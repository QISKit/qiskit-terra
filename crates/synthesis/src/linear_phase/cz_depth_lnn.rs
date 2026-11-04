// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::iter::once;

use hashbrown::HashMap;
use itertools::Itertools;
use ndarray::{Array1, ArrayView2};

use qiskit_circuit::{
    Qubit,
    operations::{Param, StandardGate},
};
use smallvec::{SmallVec, smallvec};

use crate::permutation::{_append_cx_stage1, _append_cx_stage2};

// A sequence of Lnn gates
// Represents the return type for Lnn Synthesis algorithms
pub(crate) type LnnGatesVec = Vec<(StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)>;

/// A pattern denoted by Pj in [1] for odd number of qubits:
/// [n-2, n-4, n-4, ..., 3, 3, 1, 1, 0, 0, 2, 2, ..., n-3, n-3]
fn _odd_pattern1(n: usize) -> Vec<usize> {
    debug_assert!(n >= 3);
    once(n - 2)
        .chain((0..((n - 3) / 2)).flat_map(|i| [(n - 2 * i - 4); 2]))
        .chain((0..((n - 1) / 2)).flat_map(|i| [2 * i; 2]))
        .collect()
}

/// A pattern denoted by Pk in [1] for odd number of qubits:
/// [2, 2, 4, 4, ..., n-1, n-1, n-2, n-2, n-4, n-4, ..., 5, 5, 3, 3, 1]
fn _odd_pattern2(n: usize) -> Vec<usize> {
    debug_assert!(n >= 3);
    (0..((n - 1) / 2))
        .flat_map(|i| [(2 * i + 2); 2])
        .chain((0..((n - 3) / 2)).flat_map(|i| [n - 2 * i - 2; 2]))
        .chain(once(1))
        .collect()
}

/// A pattern denoted by Pj in [1] for even number of qubits:
/// [n-1, n-3, n-3, n-5, n-5, ..., 1, 1, 0, 0, 2, 2, ..., n-4, n-4, n-2]
fn _even_pattern1(n: usize) -> Vec<usize> {
    debug_assert!(n >= 2);
    once(n - 1)
        .chain((0..((n - 2) / 2)).flat_map(|i| [n - 2 * i - 3; 2]))
        .chain((0..((n - 2) / 2)).flat_map(|i| [2 * i; 2]))
        .chain(once(n - 2))
        .collect()
}

/// A pattern denoted by Pk in [1] for even number of qubits:
/// [2, 2, 4, 4, ..., n-2, n-2, n-1, n-1, ..., 3, 3, 1, 1]
fn _even_pattern2(n: usize) -> Vec<usize> {
    debug_assert!(n >= 2);
    (0..((n - 2) / 2))
        .flat_map(|i| [2 * (i + 1); 2])
        .chain((0..(n / 2)).flat_map(|i| [(n - 2 * i - 1); 2]))
        .collect()
}

/// Creating the patterns for the phase layers.
fn _create_patterns(n: usize) -> HashMap<(usize, usize), (usize, usize)> {
    if n <= 1 {
        return HashMap::from_iter((0..n).map(|i| ((0, i), (i, i))));
    }

    let (pat1, pat2) = if n.is_multiple_of(2) {
        (_even_pattern1(n), _even_pattern2(n))
    } else {
        (_odd_pattern1(n), _odd_pattern2(n))
    };

    let ind = if n.is_multiple_of(2) {
        (2 * n - 4) / 2
    } else {
        (2 * n - 4) / 2 - 1
    };

    HashMap::from_iter((0..n).map(|i| ((0, i), (i, i))).chain(
        (0..(n / 2)).cartesian_product(0..n).map(|(layer, i)| {
            (
                (layer + 1, i),
                (pat1[ind - (2 * layer) + i], pat2[(2 * layer) + i]),
            )
        }),
    ))
}

/// Appends correct phase gate during CZ synthesis
fn _append_phase_gate(pat_val: usize, gates: &mut LnnGatesVec, qubit: usize) {
    // Add phase gates: s, sdg or z
    let gate_id = pat_val % 4;
    if gate_id != 0 {
        let gate = match gate_id {
            1 => StandardGate::Sdg,
            2 => StandardGate::Z,
            3 => StandardGate::S,
            _ => unreachable!(), // unreachable as we have modulo 4
        };
        gates.push((gate, smallvec![], smallvec![Qubit(qubit as u32)]));
    }
}

/// Synthesis of a CZ circuit for linear nearest neighbor (LNN) connectivity,
/// based on Maslov and Roetteler.
///
/// # Panics
///
/// If `matrix` is not square.
pub(super) fn synth_cz_depth_line_mr_inner(matrix: ArrayView2<bool>) -> (usize, LnnGatesVec) {
    let num_qubits = matrix.raw_dim()[0];
    let pats = _create_patterns(num_qubits);

    // s_gates[i] = 0, 1, 2 or 3 for a gate id, sdg, z or s on qubit i respectively
    let mut s_gates = Array1::<usize>::zeros(num_qubits);

    let mut patlist: Vec<(usize, usize)> = Vec::new();

    let mut gates = LnnGatesVec::new();

    for i in 0..num_qubits {
        for j in (i + 1)..num_qubits {
            if matrix[[i, j]] {
                // CZ(i,j) gate
                s_gates[[i]] += 2; // qc.z[i]
                s_gates[[j]] += 2; // qc.z[j]
                patlist.push((i, j - 1));
                patlist.push((i, j));
                patlist.push((i + 1, j - 1));
                patlist.push((i + 1, j));
            }
        }
    }

    for i in 0..num_qubits.div_ceil(2) {
        for j in 0..num_qubits {
            let pat_val = pats[&(i, j)];
            if patlist.contains(&pat_val) {
                // patcnt should be 0 or 1, which checks if a Sdg gate should be added
                let patcnt = patlist.iter().filter(|val| **val == pat_val).count();
                s_gates[[j]] += patcnt; // qc.sdg[j]
            }

            _append_phase_gate(s_gates[[j]], &mut gates, j)
        }

        _append_cx_stage1(&mut gates, num_qubits);
        _append_cx_stage2(&mut gates, num_qubits);
        s_gates = Array1::<usize>::zeros(num_qubits);
    }

    if num_qubits.is_multiple_of(2) {
        let i = num_qubits / 2;

        for j in 0..num_qubits {
            let pat_val = pats[&(i, j)];
            if patlist.contains(&pat_val) && pat_val.0 != pat_val.1 {
                // patcnt should be 0 or 1, which checks if a Sdg gate should be added
                let patcnt = patlist.iter().filter(|val| **val == pat_val).count();

                s_gates[[j]] += patcnt; // qc.sdg[j]
            }

            _append_phase_gate(s_gates[[j]], &mut gates, j)
        }

        _append_cx_stage1(&mut gates, num_qubits);
    }

    (num_qubits, gates)
}
