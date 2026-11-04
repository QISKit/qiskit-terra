// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::QiskitError;
use crate::linear::pmh::synth_pmh;
use crate::linear::utils::_row_op;
use fixedbitset::FixedBitSet;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::{prelude::*, pybacked::PyBackedStr, types::PyList};
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardGate};
use smallvec::{SmallVec, smallvec};
use std::f64::consts::PI;
type Instruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);

#[derive(Clone, Copy)]
enum AngleSpec {
    Gate(StandardGate), // t, tdg, s, sdg, z
    Phase(f64),         // numeric angle
}

fn get_instr(angle: AngleSpec, qubit_idx: usize) -> Instruction {
    let sm_vec = smallvec![];
    let qubit = smallvec![Qubit(qubit_idx as u32)];
    match angle {
        AngleSpec::Gate(gate) => (gate, sm_vec, qubit),
        AngleSpec::Phase(angle) => (
            StandardGate::Phase,
            smallvec![Param::Float(angle.rem_euclid(2.0 * PI))],
            qubit,
        ),
    }
}

// A parity is a vector in `F_2^n`. We store the set of qubits that participates
// in all the parities, as a `FixedBitSet` of `n` bits, where bit `k` is set if
// qubit `k` participates in that particular parity of the phase polynomial.

type Data = (FixedBitSet, AngleSpec);

// `GraySynth(S, I, target)` from the paper.
struct Frame {
    // `S`: the parities paired with their angles.
    parities: Vec<Data>,

    // `I`: the bit positions still available to split on. Once this is empty
    // its phase gates are applied on `target`.
    indices: Vec<usize>,

    // The qubit that accumulates the parity and gets the phase gates applied to it,
    // for a given step in the recursion.
    target: Option<usize>,
}

/// Implements `GraySynth` algorithm by Amy, Azimzadeh, and Mosca, described in the paper
///`arXiv:1712.01859 <https://arxiv.org/abs/1712.01859>`_
#[pyfunction]
#[pyo3(signature = (cnots, angles, section_size=2))]
pub fn synth_cnot_phase_aam(
    cnots: PyReadonlyArray2<bool>,
    angles: &Bound<PyList>,
    section_size: Option<usize>,
) -> PyResult<PyCircuitData> {
    let cnots = cnots.as_array();
    let num_qubits = cnots.nrows();
    let num_parities = cnots.ncols();

    if num_parities != angles.len() {
        return Err(QiskitError::new_err(
            "Number of parities (columns of cnots) and angles do not match.",
        ));
    }

    if let Some(size) = section_size
        && size > num_parities
    {
        return Err(QiskitError::new_err(format!(
            "\"section_size\"- {size} must not exceed the number of parities- {num_parities}."
        )));
    }

    let mut angle_specs: Vec<AngleSpec> = Vec::with_capacity(angles.len());
    for data in angles.iter() {
        let spec = if let Ok(label) = data.extract::<PyBackedStr>() {
            match &*label {
                "t" => AngleSpec::Gate(StandardGate::T),
                "tdg" => AngleSpec::Gate(StandardGate::Tdg),
                "s" => AngleSpec::Gate(StandardGate::S),
                "sdg" => AngleSpec::Gate(StandardGate::Sdg),
                "z" => AngleSpec::Gate(StandardGate::Z),
                other => {
                    return Err(QiskitError::new_err(format!(
                        "invalid angle: \'{other:?}\' ,each angle must be a gate label (t, tdg, s, sdg, z) or a number"
                    )));
                }
            }
        } else if let Ok(theta) = data.extract::<f64>() {
            AngleSpec::Phase(theta)
        } else {
            return Err(QiskitError::new_err(
                "each angle must be a gate label (t, tdg, s, sdg, z) or a number",
            ));
        };
        angle_specs.push(spec);
    }
    let angles = angle_specs;

    let mut s: Vec<Data> = Vec::with_capacity(num_parities);

    for j in 0..num_parities {
        let mut p = FixedBitSet::with_capacity(num_qubits);
        for i in 0..num_qubits {
            if cnots[[i, j]] {
                p.insert(i);
            }
        }
        if !p.is_clear() {
            s.push((p, angles[j]));
        }
    }

    let mut linear_state = Array2::<bool>::from_shape_fn((num_qubits, num_qubits), |(i, j)| i == j);

    let mut circuit: Vec<Instruction> = Vec::new();

    // If a parity in `s` has exactly one set bit, at position `k`, then it's phase
    // gate can be placed directly on qubit `k` with zero CNOTs.
    // Apply those phase gates now, and drop them from `s`.

    s.retain(|(parity, angle)| {
        let mut ones = parity.ones();
        match (ones.next(), ones.next()) {
            // Exactly one set bit `k` => parity == e_k. Apply on qubit `k`, then drop it.
            (Some(bit_k), None) => {
                circuit.push(get_instr(*angle, bit_k));
                false
            }
            // Weight >= 2 is kept for the recursion.
            _ => true,
        }
    });

    // `common` will hold the bitwise AND of all parities in the current `s`.
    // Bit `k` of `common` is 1 if every y ∈ s has bit `k` set, that is, bit
    // `k` is shared by all remaining parities.
    let mut common = FixedBitSet::with_capacity(num_qubits);

    // `set_bit_count[k]` will hold the number of parities in the current `s` whose
    // bit `k` is 1. Used to pick the best split bit.
    let mut set_bit_count: Vec<usize> = vec![0usize; num_qubits];

    // We pre-allocate space for 2n+4 frames, enough to avoid most
    // reallocations. The recursion tree has depth ≤ n, and each level
    // pushes two 'Frame'.
    let mut stack: Vec<Frame> = Vec::with_capacity(2 * num_qubits + 4);

    stack.push(Frame {
        parities: s,
        indices: (0..num_qubits).collect(),
        target: None,
    });

    while let Some(Frame {
        parities: mut s,
        indices,
        target: target_opt,
    }) = stack.pop()
    {
        // Skip empty branches.
        if s.is_empty() {
            continue;
        }

        // While every remaining parity in `s` shares a `1` in some bit other than
        // the target, we can apply one CNOT from that bit as control to the bit in
        // target, after that corresponding parities on the wires are also altered
        // accordingly.
        if let Some(target) = target_opt {
            loop {
                // AND all parities in `s` into `common`. Seed with the first
                // parity.
                common.clone_from(&s[0].0);

                // If none of the bits in `common` is 1, we can stop early.
                for (parity, _) in s.iter().skip(1) {
                    common.intersect_with(parity);
                    if common.is_clear() {
                        break; // no bit set, so furthur ANDs will only be 0.
                    }
                }

                // Never pick control == target, that would be invalid.
                common.set(target, false);

                // Lowest remaining shared bit, or None. Length is exactly
                // num_qubits, so no `control < num_qubits` guard is needed.
                match common.ones().next() {
                    Some(control) => {
                        // Apply a CNOT.
                        circuit.push((
                            StandardGate::CX,
                            smallvec![],
                            smallvec![Qubit(control as u32), Qubit(target as u32)],
                        ));

                        // In the paper, Lemma 4.1 says After CNOT(control, target), every parity in every
                        // frame on the stack AND in our local `s` must be updated by
                        // y_control = y_control XOR y_target.

                        // Note: at this point `s` has been moved OUT of the stack.
                        // So `stack` and `s` are disjoint we have to update each separately.
                        apply_row_op_stack(&mut stack, control, target);
                        apply_row_op_set(&mut s, control, target);
                        _row_op(linear_state.view_mut(), control, target);
                    }
                    None => break, // No shared 1-bit anywhere means we are done.
                }
            }
        }

        if indices.is_empty() {
            if let Some(target) = target_opt {
                for (_, angle) in &s {
                    circuit.push(get_instr(*angle, target));
                }
            }
            continue;
        }

        // Pick the bit to split on (the paper's `j`, Algorithm 1 line 18) the one giving the most
        // lopsided partition of `s`. Here, `j` is called `split_idx`.
        // split_idx = argmax_{split_idx ∈ indices} max(|{y : y_split_idx = 0}|, |{y : y_split_idx = 1}|)

        // Reset `set_bit_count` to zero.
        set_bit_count.fill(0);

        // Count how many parities have each bit set.
        for (parity, _) in &s {
            for idx in parity.ones() {
                set_bit_count[idx] += 1;
            }
        }

        let count_0s_1s = s.len();
        let (mut largest_idx, mut largest_subset) = (indices[0], 0usize);
        for &idx in &indices {
            let count_1s = set_bit_count[idx];
            let larger_subset = count_1s.max(count_0s_1s - count_1s);
            if larger_subset > largest_subset {
                largest_subset = larger_subset;
                largest_idx = idx;
            }
        }
        let split_idx = largest_idx;

        let mut s0: Vec<Data> = Vec::with_capacity(s.len());
        let mut s1: Vec<Data> = Vec::with_capacity(s.len());

        for (parity, angle) in s {
            if parity.contains(split_idx) {
                s1.push((parity, angle));
            } else {
                s0.push((parity, angle));
            }
        }

        let new_indices: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&i| i != split_idx)
            .collect();

        // Push S_1 first, then S_0, so S_0 ends up on TOP of the stack and
        // is processed first, matching the paper's recursion order.
        let s1_target = target_opt.or(Some(split_idx));

        if !s1.is_empty() {
            stack.push(Frame {
                parities: s1,
                indices: new_indices.clone(),
                target: s1_target,
            });
        }
        if !s0.is_empty() {
            stack.push(Frame {
                parities: s0,
                indices: new_indices,
                target: target_opt,
            });
        }
    }

    circuit.extend(synth_pmh(linear_state, section_size).rev());

    Ok(CircuitData::from_standard_gates(num_qubits as u32, circuit, Param::Float(0.0))?.into())
}

/// Apply y_control = y_control XOR y_target to every parity in every frame on the stack.
fn apply_row_op_stack(stack: &mut [Frame], control: usize, target: usize) {
    for frame in stack.iter_mut() {
        apply_row_op_set(&mut frame.parities, control, target);
    }
}

/// Apply  y_control = y_control XOR y_target to every parity in the given slice.
/// For each parity, if bit `target` is set we flip bit `control` (that is exactly XOR).
fn apply_row_op_set(s: &mut [Data], control: usize, target: usize) {
    for (y, _) in s.iter_mut() {
        if y.contains(target) {
            y.toggle(control);
        }
    }
}
