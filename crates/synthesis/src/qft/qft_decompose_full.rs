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

use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardGate, StandardInstruction};
use qiskit_circuit::packed_instruction::PackedOperation;
use smallvec::SmallVec;
use std::f64::consts::PI;

/// Construct the textbook Quantum Fourier Transform circuit.
///
/// .. note::
///
///     With the default value of ``do_swaps = True``, this synthesis algorithm creates a
///     circuit that faithfully implements the QFT operation. This circuit contains a sequence
///     of swap gates at the end, corresponding to reversing the order of its output qubits.
///     In some applications this reversal permutation can be avoided. Setting ``do_swaps = False``
///     creates a circuit without this reversal permutation, at the expense that this circuit
///     implements the "QFT-with-reversal" instead of QFT. Alternatively, the
///     :class:`~.ElidePermutations` transpiler pass is able to remove these swap gates.
///
/// Args:
///     num_qubits: The number of qubits on which the Quantum Fourier Transform acts.
///     do_swaps: Whether to synthesize the "QFT" or the "QFT-with-reversal" operation.
///     approximation_degree: The degree of approximation (0 for no approximation).
///         It is possible to implement the QFT approximately by ignoring
///         controlled-phase rotations with the angle beneath a threshold.
///     insert_barriers: If ``True``, barriers are inserted after each qubit's H+CP block
///         for improved visualization.
///
/// Returns:
///     A circuit implementing the QFT operation.
#[pyfunction]
#[pyo3(signature=(num_qubits, do_swaps=true, approximation_degree=0, insert_barriers=false))]
pub fn synth_qft_full(
    num_qubits: usize,
    do_swaps: bool,
    approximation_degree: usize,
    insert_barriers: bool,
) -> PyResult<PyCircuitData> {
    // Pre-calculate exact gate count to avoid reallocations.
    // H gates:    one per qubit = num_qubits
    // CP gates:   sum_{j=0}^{n-1} num_entanglements(j)
    //             = (n-1-d) * (n+d) / 2   where d = approximation_degree
    //               (0 when approximation_degree >= num_qubits - 1)
    // Barrier:    one after each qubit's H+CP block = num_qubits (only when insert_barriers)
    // Swap gates: num_qubits / 2  (only when do_swaps)
    let effective = num_qubits
        .saturating_sub(1)
        .saturating_sub(approximation_degree);
    let no_of_gates = num_qubits
        + effective * (num_qubits + approximation_degree) / 2
        + if insert_barriers { num_qubits } else { 0 }
        + if do_swaps { num_qubits / 2 } else { 0 };

    let mut circuit =
        CircuitData::with_capacity(num_qubits as u32, 0, no_of_gates, Param::Float(0.0))?;
    let (barrier_op, all_qubits) = if insert_barriers {
        let op = PackedOperation::from_standard_instruction(StandardInstruction::Barrier(
            num_qubits as u32,
        ));
        let qubits: SmallVec<[Qubit; 32]> = (0..num_qubits).map(|q| Qubit(q as u32)).collect();
        (Some(op), Some(qubits))
    } else {
        (None, None)
    };

    for j in (0..num_qubits).rev() {
        circuit.push_standard_gate(StandardGate::H, &[], &[Qubit(j as u32)])?;

        let tail = num_qubits - j - 1;
        let approx_offset = approximation_degree.saturating_sub(tail);
        let num_entanglements = j.saturating_sub(approx_offset);

        for k in (j - num_entanglements..j).rev() {
            let lam = PI * (2.0_f64).powi(k as i32 - j as i32);
            circuit.push_standard_gate(
                StandardGate::CPhase,
                &[Param::Float(lam)],
                &[Qubit(j as u32), Qubit(k as u32)],
            )?;
        }

        if let (Some(op), Some(qubits)) = (&barrier_op, &all_qubits) {
            circuit.push_packed_operation(op.clone(), None, qubits, &[])?;
        }
    }

    if do_swaps {
        // Add a reversal network
        for i in 0..num_qubits / 2 {
            circuit.push_standard_gate(
                StandardGate::Swap,
                &[],
                &[Qubit(i as u32), Qubit((num_qubits - i - 1) as u32)],
            )?;
        }
    }
    Ok(circuit.into())
}
