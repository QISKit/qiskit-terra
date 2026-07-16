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

use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};
use qiskit_circuit::{circuit_data::CircuitData, dag_circuit::DAGCircuit};
use qiskit_transpiler::{
    passes::{Optimize1qGatesDecompositionState, run_optimize_1q_gates_decomposition},
    target::Target,
};

/// @ingroup QkTranspilerPassesStandalone
/// Runs the Optimize1qGatesDecomposition pass in standalone mode on a circuit.
///
/// \qk_deprecated{2.4.0|use :c:func:`qk_transpiler_pass_standalone_optimize_1q_sequences` instead.}
///
/// This function is multithreaded and will potentially launch a thread pool
/// with threads equal to the number of CPUs by default. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable.
/// For example, setting ``RAYON_NUM_THREADS=4`` would limit the thread pool
/// to 4 threads.
///
/// @param circuit A pointer to the ``QkCircuit`` object to transform.
/// @param target A pointer to the ``QkTarget`` object or a null pointer.
/// In the case a null pointer is provided and gate errors are unknown
/// the pass will choose the sequence with the least amount of gates,
/// and will support all basis gates on its Euler basis set.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit`` and
/// if ``target`` is not a valid pointer to a ``QkTarget``.
///
/// cbindgen:qk-vtable-rules=[no-export]
#[deprecated(
    since = "2.4.0",
    note = "use `qk_transpiler_pass_standalone_optimize_1q_sequences` instead"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_standalone_optimize_1q_sequences(
    circuit: *mut CircuitData,
    target: *const Target,
) {
    unsafe { qk_transpiler_pass_standalone_optimize_1q_sequences(circuit, target) };
}

/// @ingroup QkTranspilerPassesStandalone
/// Runs the Optimize1qGatesDecomposition pass in standalone mode on a circuit.
///
/// Refer to the ``qk_transpiler_pass_optimize_1q_sequences`` function for more details about the pass.
///
/// This function is multithreaded and will potentially launch a thread pool
/// with threads equal to the number of CPUs by default. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable.
/// For example, setting ``RAYON_NUM_THREADS=4`` would limit the thread pool
/// to 4 threads.
///
///
/// @param circuit A pointer to the ``QkCircuit`` object to transform.
/// @param target A pointer to the ``QkTarget`` object or a null pointer.
/// In the case a null pointer is provided and gate errors are unknown
/// the pass will choose the sequence with the least amount of gates,
/// and will support all basis gates on its Euler basis set.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit`` and
/// if ``target`` is not a valid pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_optimize_1q_sequences(
    circuit: *mut CircuitData,
    target: *const Target,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe {
        if target.is_null() {
            None
        } else {
            Some(const_ptr_as_ref(target))
        }
    };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };

    // Convert the circuit to a DAG.
    let mut circuit_as_dag = DAGCircuit::from_circuit_data(circuit, false, None, None, None, None)
        .expect("Error while converting the circuit to a dag.");

    let state = Optimize1qGatesDecompositionState::new(
        target.map(|x| x.num_qubits.unwrap_or(0)).unwrap_or(0) as usize,
    );
    // Run the pass
    run_optimize_1q_gates_decomposition(&mut circuit_as_dag, &state, target, None, None)
        .expect("Error while running the pass.");

    // Convert the DAGCircuit back to an instance of CircuitData
    *circuit = CircuitData::from_dag_ref(&circuit_as_dag)
        .expect("Error while converting the dag to a circuit.");
}

/// @ingroup QkTranspilerPasses
/// Runs the Optimize1qGatesDecomposition pass in standalone mode on a DAG Circuit.
///
/// Optimize1qGatesDecomposition optimizes single-qubit gate sequences by re-synthesizing
/// the unitary under the constraints of the target's basis gates and error rates.
///
/// The decision of whether to replace the original chain depends on:
/// - If the original chain was out of basis.
/// - If the original chain was in basis but the replacement has lower error rates.
/// - If the original chain is an identity (chain gets removed).
///
/// The error is the combined multiplication of the errors of individual gates on the
/// qubit it operates on.
///
/// This function is multithreaded and will potentially launch a thread pool
/// with threads equal to the number of CPUs by default. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable.
/// For example, setting ``RAYON_NUM_THREADS=4`` would limit the thread pool
/// to 4 threads.
///
/// @param dag A pointer to the ``QkDag`` object to transform.
/// @param target A pointer to the ``QkTarget`` object or a null pointer.
/// In the case a null pointer is provided and gate errors are unknown
/// the pass will choose the sequence with the least amount of gates,
/// and will support all basis gates on its Euler basis set.
///
/// # Example
/// ```c
/// QkTarget *target = qk_target_new(1);
/// double u_errors[3] = {0., 1e-4, 1e-4};
/// for (int idx = 0; idx < 3; idx++) {
///     QkTargetEntry *u_entry = qk_target_entry_new(QkGate_U);
///     uint32_t qargs[1] = {0};
///     qk_target_entry_add_property(u_entry, qargs, 1, NAN, u_errors[idx]);
///     qk_target_add_instruction(target, u_entry);
/// }
///
/// // Build circuit
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1, "qr");
/// qk_dag_add_quantum_register(dag, qr);
/// uint32_t qubits[1] = {0};
/// for (int iter = 0; iter < 3; iter++) {
///     qk_dag_apply_gate(dag, QkGate_H, qubits, NULL, false);
/// }
///
/// // Run transpiler pass
/// qk_transpiler_pass_optimize_1q_sequences(dag, target);
///
/// // Clean up
/// qk_target_free(target);
/// qk_dag_free(dag);
/// qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag`` and
/// if ``target`` is not a valid pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_optimize_1q_sequences(
    dag: *mut DAGCircuit,
    target: *const Target,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe {
        if target.is_null() {
            None
        } else {
            Some(const_ptr_as_ref(target))
        }
    };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { mut_ptr_as_ref(dag) };

    let state = Optimize1qGatesDecompositionState::new(
        target
            .map(|x| {
                x.num_qubits
                    .map(|num_qubits| num_qubits as usize)
                    .unwrap_or(0)
            })
            .unwrap_or(0),
    );
    // Run the pass
    run_optimize_1q_gates_decomposition(dag, &state, target, None, None)
        .expect("Error while running the pass.");
}
