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

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations;
use qiskit_circuit::operations::{Param, multiply_param, radd_param};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_synthesis::pauli_evolution::sparse_term_evolution;
use smallvec::smallvec;

/// Implement a Pauli evolution circuit.
///
/// The Pauli evolution is implemented as a basis transformation to the Pauli-Z basis,
/// followed by a CX-chain and then a single Pauli-Z rotation on the last qubit. Then the CX-chain
/// is uncomputed and the inverse basis transformation applied. E.g. for the evolution under the
/// Pauli string XIYZ we have the circuit
///
///        ┌───┐      ┌───┐┌───────┐┌───┐┌───┐
///     0: ┤ H ├──────┤ X ├┤ Rz(2) ├┤ X ├┤ H ├────────
///        └───┘      └─┬─┘└───────┘└─┬─┘└───┘
///     1: ─────────────┼─────────────┼───────────────
///        ┌────┐┌───┐  │             │  ┌───┐┌──────┐
///     2: ┤ √X ├┤ X ├──■─────────────■──┤ X ├┤ √Xdg ├
///        └────┘└─┬─┘                   └─┬─┘└──────┘
///     3: ────────■───────────────────────■──────────
///
/// Args:
///     num_qubits: The number of qubits in the Hamiltonian.
///     sparse_paulis: The Paulis to implement. Given in a sparse-list format with elements
///         ``(pauli_string, qubit_indices, rz_rotation_angle)``. An element of the form
///         ``("XIYZ", [0,1,2,3], 2)``, for example, is interpreted in terms of qubit indices as
///         X_q0 I_q1 Y_q2 Z_q3 and will use a RZ rotation angle of 2.
///     insert_barriers: If ``true``, insert a barrier in between the evolution of individual
///         Pauli terms.
///     do_fountain: If ``true``, implement the CX propagation as "fountain" shape, where each
///         CX uses the top qubit as target. If ``false``, uses a "chain" shape, where CX in between
///         neighboring qubits are used.
///
/// Returns:
///     Circuit data for to implement the evolution.
#[pyfunction]
#[pyo3(name = "pauli_evolution", signature = (num_qubits, sparse_paulis, insert_barriers=false, do_fountain=false))]
pub fn py_pauli_evolution(
    num_qubits: i64,
    sparse_paulis: &Bound<PyList>,
    insert_barriers: bool,
    do_fountain: bool,
) -> PyResult<PyCircuitData> {
    let num_paulis = sparse_paulis.len();
    let mut paulis: Vec<String> = Vec::with_capacity(num_paulis);
    let mut indices: Vec<Vec<u32>> = Vec::with_capacity(num_paulis);
    let mut times: Vec<Param> = Vec::with_capacity(num_paulis);
    let mut global_phase = Param::Float(0.0);
    let mut modified_phase = false; // keep track of whether we modified the phase

    for el in sparse_paulis.iter() {
        let tuple = el.cast::<PyTuple>()?;
        let pauli = tuple.get_borrowed_item(0)?.cast::<PyString>()?.to_string();
        let time = Param::extract_no_coerce(tuple.get_borrowed_item(2)?)?;

        if pauli.as_str().chars().all(|p| p == 'i') {
            global_phase = radd_param(global_phase, time);
            modified_phase = true;
            continue;
        }

        paulis.push(pauli);
        times.push(time); // note we do not multiply by 2 here, this is already done Python side!
        indices.push(tuple.get_item(1)?.extract::<Vec<u32>>()?)
    }

    let barrier = (
        PackedOperation::from_standard_instruction(operations::StandardInstruction::Barrier(
            num_qubits as u32,
        )),
        smallvec![],
        (0..num_qubits as u32).map(Qubit).collect(),
        vec![],
    );

    let evos = paulis.iter().enumerate().zip(indices).zip(times).flat_map(
        |(((i, pauli), qubits), time)| {
            let as_packed = sparse_term_evolution(pauli, qubits, time, false, do_fountain).map(Ok);
            // this creates an iterator containing a barrier only if required, otherwise it is empty
            let maybe_barrier = (insert_barriers && i < (num_paulis - 1))
                .then_some(Ok(barrier.clone()))
                .into_iter();
            as_packed.chain(maybe_barrier)
        },
    );

    // When handling all-identity Paulis above, we added the RZ rotation angle as global phase,
    // meaning that we have implemented of exp(i 2t I). However, what we want it to implement
    // exp(-i t I). To only use a single multiplication, we apply a factor of -0.5 here.
    // This is faster, in particular as long as the parameter expressions are in Python.
    if modified_phase {
        global_phase = multiply_param(&global_phase, -0.5);
    }

    Ok(CircuitData::from_packed_operations(num_qubits as u32, 0, evos, global_phase)?.into())
}
