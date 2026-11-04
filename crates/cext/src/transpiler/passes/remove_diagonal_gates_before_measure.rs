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

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::run_remove_diagonal_before_measure;

/// @ingroup QkTranspilerPassesStandalone
/// Run the ``RemoveDiagonalGatesBeforeMeasure`` pass on a circuit.
///
/// Refer to the ``qk_transpiler_pass_remove_diagonal_gates_before_measure`` function for more details about the pass.
///
/// @param circuit A pointer to the circuit to run this pass on.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_remove_diagonal_gates_before_measure(
    circuit: *mut CircuitData,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { &mut *circuit };
    let mut dag = DAGCircuit::from_circuit_data(circuit, false, None, None)
        .expect("Circuit to DAG conversion failed");
    run_remove_diagonal_before_measure(&mut dag);
    *circuit = CircuitData::from_dag_ref(&dag).expect("DAG to Circuit conversion failed");
}

/// @ingroup QkTranspilerPasses
/// Run the ``RemoveDiagonalGatesBeforeMeasure`` pass on a DAG Circuit.
///
/// Transpiler pass to remove diagonal gates (like RZ, T, Z, etc) before
/// a measurement. Including diagonal 2Q gates.
///
/// @param dag A pointer to the DAG to run this pass on. The DAG will be modified in place by the pass.
///
/// # Example
///
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1, "qr");
/// QkClassicalRegister *cr = qk_classical_register_new(1, "cr");
/// qk_dag_add_quantum_register(dag, qr);
/// qk_dag_add_classical_register(dag, cr);
/// qk_dag_apply_gate(dag, QkGate_Z, (uint32_t[1]){0}, NULL, false);
/// qk_dag_apply_measure(dag, 0, 0, false);
/// qk_transpiler_pass_remove_diagonal_gates_before_measure(dag);
/// // ...
/// qk_dag_free(dag);
/// qk_quantum_register_free(qr);
/// qk_classical_register_free(cr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_remove_diagonal_gates_before_measure(
    dag: *mut DAGCircuit,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { &mut *dag };
    run_remove_diagonal_before_measure(dag);
}
