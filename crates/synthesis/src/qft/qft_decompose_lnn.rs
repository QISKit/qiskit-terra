// This code is part of Qiskit.
//
// (C) Copyright IBM 2025.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::linear_phase::cz_depth_lnn::LnnGatesVec;
use crate::permutation::_append_reverse_permutation_lnn_kms;
use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardGate};
use smallvec::smallvec;
use std::f64::consts::PI;

/// Construct a circuit for the Quantum Fourier Transform using linear
/// neighbor connectivity.
///
/// The construction is based on Fig 2.b in Fowler et al. [1].
///
/// .. note::
///
///     With the default value of ``do_swaps = True``, this synthesis algorithm creates a
///     circuit that faithfully implements the QFT operation. When ``do_swaps = False``,
///     this synthesis algorithm creates a circuit that corresponds to "QFT-with-reversal":
///     applying the QFT and reversing the order of its output qubits.
///
/// Args:
///     num_qubits: The number of qubits on which the Quantum Fourier Transform acts.
///     approximation_degree: The degree of approximation (0 for no approximation).
///         It is possible to implement the QFT approximately by ignoring
///         controlled-phase rotations with the angle beneath a threshold. This is discussed
///         in more detail in https://arxiv.org/abs/quant-ph/9601018 or
///         https://arxiv.org/abs/quant-ph/0403071.
///     do_swaps: Whether to synthesize the "QFT" or the "QFT-with-reversal" operation.
///
/// Returns:
///     A circuit implementing the QFT operation.
///
/// References:
///     1. A. G. Fowler, S. J. Devitt, and L. C. L. Hollenberg,
///        *Implementation of Shor's algorithm on a linear nearest neighbour qubit array*,
///        Quantum Info. Comput. 4, 4 (July 2004), 237–251.
///        `arXiv:quant-ph/0402196 [quant-ph] <https://arxiv.org/abs/quant-ph/0402196>`_
#[pyfunction]
#[pyo3(signature=(num_qubits, do_swaps=true, approximation_degree=0))]
pub fn synth_qft_line(
    num_qubits: usize,
    do_swaps: bool,
    approximation_degree: usize,
) -> PyResult<PyCircuitData> {
    // Total number of compound gates required = L(L-1)/2
    // Compound gate: H + 3CX + 3P or 3CX + 3P
    // For approximation degree D, D(D+1)/2 * 3 gates will be reduced
    let mut no_of_gates = num_qubits + (num_qubits * (num_qubits - 1) / 2) * 6
        - (approximation_degree * (approximation_degree + 1) / 2) * 3;

    if !do_swaps {
        // `_append_reverse_permutation_lnn_kms` would add
        no_of_gates += num_qubits * num_qubits - 1;
    }

    let mut instructions: LnnGatesVec = Vec::with_capacity(no_of_gates);

    for i in 0..num_qubits {
        append_h(&mut instructions, num_qubits - 1);

        for j in i..num_qubits - 1 {
            let q0 = num_qubits - j + i - 1;
            let q1 = num_qubits - j + i - 2;
            let phase = PI / 2.0_f64.powi((j - i + 2) as i32);

            if j - i + 2 < num_qubits - approximation_degree + 1 {
                append_phase(&mut instructions, q0, phase);
                append_cx(&mut instructions, q0, q1);
                append_phase(&mut instructions, q1, -phase);
                append_cx(&mut instructions, q1, q0);
                append_cx(&mut instructions, q0, q1);
                append_phase(&mut instructions, q0, phase);
            } else {
                // Swap
                append_cx(&mut instructions, q0, q1);
                append_cx(&mut instructions, q1, q0);
                append_cx(&mut instructions, q0, q1);
            }
        }
    }
    if !do_swaps {
        // Add a reversal network for LNN connectivity in depth 2*n+2,
        // based on Kutin at al., https://arxiv.org/abs/quant-ph/0701194, Section 5.
        _append_reverse_permutation_lnn_kms(&mut instructions, num_qubits);
    }

    Ok(
        CircuitData::from_standard_gates(num_qubits as u32, instructions, Param::Float(0.0))?
            .into(),
    )
}

#[inline]
fn append_h(instructions: &mut LnnGatesVec, q0: usize) {
    instructions.push((StandardGate::H, smallvec![], smallvec![Qubit::new(q0)]));
}

#[inline]
fn append_cx(instructions: &mut LnnGatesVec, q0: usize, q1: usize) {
    instructions.push((
        StandardGate::CX,
        smallvec![],
        smallvec![Qubit::new(q0), Qubit::new(q1)],
    ));
}

#[inline]
fn append_phase(instructions: &mut LnnGatesVec, q0: usize, phase: f64) {
    instructions.push((
        StandardGate::Phase,
        smallvec![Param::Float(phase)],
        smallvec![Qubit::new(q0)],
    ));
}
