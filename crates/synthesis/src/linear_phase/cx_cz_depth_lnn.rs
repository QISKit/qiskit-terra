// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::linear::lnn::synth_cnot_lnn_instructions;
use crate::linear::utils::calc_inverse_matrix_inner;

use hashbrown::HashSet;
use ndarray::{Array2, ArrayView2, s};
use numpy::PyReadonlyArray2;
use smallvec::smallvec;
use std::cmp::{max, min};

use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardGate};

enum CircuitInstructions {
    CX(u32, u32),
    Z(u32),
    S(u32),
    Sdg(u32),
}

/// Given a CZ layer (represented as an `n*n` CZ matrix `Mz`)
/// return a schedule of phase gates implementing `Mz` in a SWAP-only network
///
/// (c.f. Alg 1, [2])
fn _initialize_phase_schedule(mat_z: ArrayView2<bool>) -> Array2<usize> {
    let n = mat_z.nrows();
    let mut phase_schedule = Array2::<usize>::from_elem((n, n), 0);
    (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .filter(|(i, j)| mat_z[[*i, *j]])
        .for_each(|(i, j)| {
            phase_schedule[[i, j]] += 3;
            phase_schedule[[i, i]] += 1;
            phase_schedule[[j, j]] += 1;
        });
    phase_schedule
}

/// Shuffle the indices in labels by swapping adjacent elements
///
/// (c.f. Fig.2, [2])
fn _shuffle(labels: &[usize], start_from: usize) -> Vec<usize> {
    let mut shuffled_labels = labels.to_owned();
    shuffled_labels[start_from..]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .for_each(|pair| pair.swap(0, 1));
    shuffled_labels
}

/// Return the labels of the boxes in order from left to right, top to bottom
///
/// Given the width of the circuit `n`, (c.f. Fig.2, [2])
fn _make_seq(n: usize) -> Vec<(usize, usize)> {
    (0..n)
        .scan((0..n).rev().collect::<Vec<usize>>(), |wire_labels, i| {
            let wire_labels_new = _shuffle(wire_labels, i % 2);
            let seq_slice: Vec<(usize, usize)> = wire_labels
                .iter()
                .zip(wire_labels_new.iter())
                .step_by(2)
                .filter(|(a, b)| a != b)
                .map(|(&a, &b)| (min(a, b), max(a, b)))
                .collect();
            *wire_labels = wire_labels_new;
            Some(seq_slice)
        })
        .flatten()
        .collect()
}

/// Return a list of labels of the boxes that is SWAP+ in descending order
/// Given CX instructions (c.f. Thm 7.1, [1]) and the labels of all boxes.
///
/// - Assumes the instruction gives gates in the order from top to bottom,
///   from left to right
/// - SWAP+ is defined in section 3.A. of [2]. Note the northwest
///   diagonalization procedure of [1] consists exactly n layers of boxes,
///   each being either a SWAP or a SWAP+. That is, each northwest
///   diagonalization circuit can be uniquely represented by which of its
///   n(n-1)/2 boxes are SWAP+ and which are SWAP.
fn _swap_plus(instructions: &[(usize, usize)], seq: &[(usize, usize)]) -> HashSet<(usize, usize)> {
    (0..seq.len())
        .scan(0, |inst_index, i| {
            if (*inst_index + 2 >= instructions.len())
                || (instructions[*inst_index] != instructions[*inst_index + 2])
            {
                // Only two CNOTs on same set of controls -> this box is SWAP+
                *inst_index += 2;
                Some(Some(i))
            } else {
                *inst_index += 3;
                // simply returning None will stop the scan; flatten() will remove this case
                Some(None)
            }
        })
        .flatten()
        .map(|i| seq[i])
        .collect()
}

/// Given phase_schedule initialized to induce a CZ circuit in SWAP-only network and list of SWAP+ boxes
/// updates phase_schedule for each SWAP+ according to Algorithm 2, [2]
fn _update_phase_schedule(
    n: usize,
    phase_schedule: &mut Array2<usize>,
    swap_plus: &HashSet<(usize, usize)>,
) {
    if n <= 1 {
        return;
    }
    let mut layer_order: Vec<usize> = ((1 - (n % 2))..n - 2).step_by(2).rev().collect();
    layer_order.extend((n % 2..n - 2).step_by(2));
    if n > 1 {
        layer_order.extend(vec![n - 2]);
    }

    // this is like doing np.argsort(layer_order[::-1]) in Python
    let mut order_comp: Vec<usize> = (0..n - 1).collect();
    order_comp.sort_by_key(|&i| layer_order[n - 2 - i]);

    // Go through each box by descending layer order
    layer_order
        .iter()
        .flat_map(|&i| (i + 1..n).map(move |j| (i, j)))
        .filter(|&(i, j)| swap_plus.contains(&(i, j)))
        .for_each(|(i, j)| {
            // we need to correct for the effected linear functions:

            // We first correct type 1 and type 2 by switching
            // the phase applied to c_j and c_i+c_j
            let mut slice = phase_schedule.slice_mut(s![.., j]);
            slice.swap(i, j);

            // Then, we go through all the boxes that permutes j BEFORE box(i,j) and update:
            let valid_indices: Vec<usize> = (0..n)
                .filter(|&k| {
                    (k != i)
                        && (k != j)
                        && (order_comp[min(k, j)] < order_comp[i])
                        && !phase_schedule[[min(k, j), max(k, j)]].is_multiple_of(4)
                })
                .collect();

            for k in valid_indices {
                let phase = phase_schedule[[min(k, j), max(k, j)]];
                phase_schedule[[min(k, j), max(k, j)]] = 0;
                // Step 1, apply phase to c_i, c_j, c_k
                for l_s in [i, j, k] {
                    phase_schedule[[l_s, l_s]] = (phase_schedule[[l_s, l_s]] + phase * 3) % 4;
                }
                // Step 2, apply phase to c_i+ c_j, c_i+c_k, c_j+c_k:
                for (l1, l2) in [(i, j), (i, k), (j, k)] {
                    let ls = min(l1, l2);
                    let lb = max(l1, l2);
                    phase_schedule[[ls, lb]] = (phase_schedule[[ls, lb]] + phase * 3) % 4;
                }
            }
        });
}

/// Return a QuantumCircuit that computes the phase schedule S inside CX
///
/// Given
/// - Width of the circuit (int `n`)
/// - A CZ circuit, represented by the `n*n` phase schedule phase_schedule
/// - A CX circuit, represented by box-labels (seq) and whether the box is SWAP+ (swap_plus)
///   - This circuit corresponds to the CX transformation that transforms a matrix to
///     a NW matrix (c.f. Prop.7.4, [1])
///   - SWAP+ is defined in section 3.A. of [2].
///   - As previously noted, the northwest diagonalization procedure of [1] consists
///     of exactly n layers of boxes, each being either a SWAP or a SWAP+. That is,
///     each northwest diagonalization circuit can be uniquely represented by which
///     of its `n(n-1)/2` boxes are SWAP+ and which are SWAP.
fn _apply_phase_to_nw_circuit(
    n: usize,
    phase_schedule: &Array2<usize>,
    seq: &[(usize, usize)],
    swap_plus: &HashSet<(usize, usize)>,
) -> Vec<CircuitInstructions> {
    let wires: Vec<_> = (0..n.saturating_sub(1))
        .step_by(2)
        .zip((1..n).step_by(2))
        .chain((1..n.saturating_sub(1)).step_by(2).zip((2..n).step_by(2)))
        .collect();

    let mut cir: Vec<CircuitInstructions> = Vec::new();
    for (i, &(j, k)) in (0..seq.len()).rev().zip(seq.iter().rev()) {
        let (w1, w2) = wires[i % (n.saturating_sub(1))];
        if !swap_plus.contains(&(j, k)) {
            cir.push(CircuitInstructions::CX(w1 as u32, w2 as u32));
        }
        cir.push(CircuitInstructions::CX(w2 as u32, w1 as u32));
        match phase_schedule[[j, k]] % 4 {
            0 => {}
            1 => cir.push(CircuitInstructions::Sdg(w2 as u32)),
            2 => cir.push(CircuitInstructions::Z(w2 as u32)),
            3 => cir.push(CircuitInstructions::S(w2 as u32)),
            _ => unreachable!(),
        }
        cir.push(CircuitInstructions::CX(w1 as u32, w2 as u32));
    }
    for i in 0..n {
        match phase_schedule[[n - 1 - i, n - 1 - i]] % 4 {
            0 => {}
            1 => cir.push(CircuitInstructions::Sdg(i as u32)),
            2 => cir.push(CircuitInstructions::Z(i as u32)),
            3 => cir.push(CircuitInstructions::S(i as u32)),
            _ => unreachable!(),
        }
    }
    cir
}

/// Joint synthesis of a -CZ-CX- circuit for linear nearest neighbor (LNN) connectivity,
/// with 2-qubit depth at most 5n, based on Maslov and Yang.
///
/// This method computes the CZ circuit inside the CX circuit via phase gate insertions.
///
/// # Arguments
/// - mat_z : a boolean symmetric matrix representing a CZ circuit.
///    `mat_z[i][j]=1` represents a `cz(i,j)` gate
///
/// - mat_x : a boolean invertible matrix representing a CX circuit.
///
/// # Returns
/// A circuit implementation of a CX circuit following a CZ circuit,
/// denoted as a -CZ-CX- circuit,in two-qubit depth at most `5n`, for LNN connectivity.
///
/// # References
/// 1. Kutin, S., Moulton, D. P., Smithline, L.,
/// *Computation at a distance*, Chicago J. Theor. Comput. Sci., vol. 2007, (2007),
/// [arXiv:quant-ph/0701194] (https://arxiv.org/abs/quant-ph/0701194)
///
/// 2. Dmitri Maslov, Willers Yang, *CNOT circuits need little help to implement arbitrary
/// Hadamard-free Clifford transformations they generate*,
/// [arXiv:2210.16195] (https://arxiv.org/abs/2210.16195).
#[pyfunction]
#[pyo3(signature = (mat_x, mat_z))]
pub fn py_synth_cx_cz_depth_line_my(
    mat_x: PyReadonlyArray2<bool>,
    mat_z: PyReadonlyArray2<bool>,
) -> PyResult<PyCircuitData> {
    // First, find circuits implementing mat_x by Proposition 7.3 and Proposition 7.4 of [1]
    let n = mat_x.as_array().nrows(); // is a quadratic matrix
    let mat_x = calc_inverse_matrix_inner(mat_x.as_array(), false).unwrap();
    let (cx_instructions_rows_m2nw, cx_instructions_rows_nw2id) =
        synth_cnot_lnn_instructions(mat_x.view());

    // Meanwhile, also build the -CZ- circuit via Phase gate insertions as per Algorithm 2 [2]
    let mut phase_schedule = _initialize_phase_schedule(mat_z.as_array());
    let seq = _make_seq(n);
    let swap_plus = _swap_plus(&cx_instructions_rows_nw2id, &seq);

    _update_phase_schedule(n, &mut phase_schedule, &swap_plus);

    let mut qc_instructions = _apply_phase_to_nw_circuit(n, &phase_schedule, &seq, &swap_plus);

    for &(i, j) in cx_instructions_rows_m2nw.iter().rev() {
        qc_instructions.push(CircuitInstructions::CX(i as u32, j as u32));
    }

    let instructions = qc_instructions.into_iter().map(|inst| match inst {
        CircuitInstructions::CX(ctrl, target) => (
            StandardGate::CX,
            smallvec![],
            smallvec![Qubit(ctrl), Qubit(target)],
        ),
        CircuitInstructions::S(qubit) => (StandardGate::S, smallvec![], smallvec![Qubit(qubit)]),
        CircuitInstructions::Sdg(qubit) => {
            (StandardGate::Sdg, smallvec![], smallvec![Qubit(qubit)])
        }
        CircuitInstructions::Z(qubit) => (StandardGate::Z, smallvec![], smallvec![Qubit(qubit)]),
    });
    Ok(CircuitData::from_standard_gates(n as u32, instructions, Param::Float(0.0))?.into())
}
