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
use qiskit_transpiler::passes::{check_direction_target, fix_direction_target};
use qiskit_transpiler::target::Target;

/// @ingroup QkTranspilerPassesStandalone
/// Run the ``CheckGateDirection`` pass on a circuit.
///
/// Refer to the ``qk_transpiler_pass_check_gate_direction`` function for more details about the pass.
///
/// @param circuit A pointer to the circuit on which to run the pass.
/// @param target A pointer to the target used for checking gate directions.
/// @return True iff all two-qubit gate directions comply with target constraints.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` are not valid, non-null pointers to ``QkCircuit`` and ``QkTarget`` objects, respectively.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_check_gate_direction(
    circuit: *const CircuitData,
    target: *const Target,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { &*circuit };
    let target = unsafe { &*target };

    let dag = DAGCircuit::from_circuit_data(circuit, false, None, None)
        .expect("Circuit to DAG conversion failed");

    check_direction_target(&dag, target).expect("Unexpected error occurred in CheckGateDirection")
}

/// @ingroup QkTranspilerPassesStandalone
/// Run the ``GateDirection`` pass on a circuit.
///
/// Refer to the ``qk_transpiler_pass_gate_direction`` function for more details about the pass.
///
/// @param circuit A pointer to the circuit to modify in place.
/// @param target A pointer to the target used for gate directions.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` are not valid, non-null pointers to ``QkCircuit`` and ``QkTarget`` objects, respectively.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_gate_direction(
    circuit: *mut CircuitData,
    target: *const Target,
) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let circuit = unsafe { &mut *circuit };
    let target = unsafe { &*target };

    let mut dag = DAGCircuit::from_circuit_data(circuit, false, None, None)
        .expect("Circuit to DAG conversion failed");

    fix_direction_target(&mut dag, target).expect("Unexpected error occurred in GateDirection");

    *circuit = CircuitData::from_dag_ref(&dag).expect("DAG to circuit conversion failed");
}

/// @ingroup QkTranspilerPasses
/// Run the ``CheckGateDirection`` pass on a DAG Circuit.
///
/// The pass checks if the directions of two-qubit gates comply with the gate directions specified in a given target.
///
/// @param dag A pointer to the DAG Circuit on which to run the CheckGateDirection pass.
/// @param target A pointer to the target used for checking gate directions.
///
/// @return bool - true iff the directions of all two-qubit gates in the circuit comply with the specified target constraints.
///
/// # Example
/// ```c
/// QkTarget *target = qk_target_new(2);
/// uint32_t qargs[3] = {0,1};
///
/// QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
/// qk_target_entry_add_property(cx_entry, qargs, 2, 0.0, 0.0);
/// qk_target_add_instruction(target, cx_entry);
///
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
/// qk_dag_add_quantum_register(dag, qr);
/// qk_dag_apply_gate(dag, QkGate_CX, (uint32_t[]){1,0}, NULL, false);
///
/// bool direction_ok = qk_transpiler_pass_check_gate_direction(dag, target);
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// qk_target_free(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` or ``target`` are not valid, non-null pointers to ``QkDag`` and ``QkTarget`` objects, respectively.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_check_gate_direction(
    dag: *const DAGCircuit,
    target: *const Target,
) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let dag = unsafe { &*dag };
    let target = unsafe { &*target };

    check_direction_target(dag, target).expect("Unexpected error occurred in CheckGateDirection")
}

/// @ingroup QkTranspilerPasses
/// Run the ``GateDirection`` pass on a DAG.
///
/// The GateDirection pass modifies asymmetric gates to match the hardware coupling directions.
/// This pass supports replacements for the ``cx``, ``cz``, ``ecr``, ``swap``, ``rzx``, ``rxx``, ``ryy`` and
/// ``rzz`` gates, using predefined identities.
///
/// @param dag A pointer to the DAG Circuit on which to run the GateDirection pass. The DAG will be modified
///     in place by the pass.
/// @param target A pointer to the target used for checking gate directions.
///
/// # Example
/// ```c
/// QkTarget *target = qk_target_new(3);
///
/// uint32_t qargs[2] = {0,1};
///
/// QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
/// qk_target_entry_add_property(cx_entry, qargs, 2, 0.0, 0.0);
/// qk_target_add_instruction(target, cx_entry);
///
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(3, "qr");
/// qk_dag_add_quantum_register(dag, qr);
/// qk_dag_apply_gate(dag, QkGate_CX, (uint32_t[]){1,0}, NULL, false);
///
/// qk_transpiler_pass_gate_direction(dag, target);
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// qk_target_free(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` or ``target`` are not valid, non-null pointers to ``QkDag`` and ``QkTarget`` objects, respectively.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_gate_direction(
    dag: *mut DAGCircuit,
    target: *const Target,
) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let dag = unsafe { &mut *dag };
    let target = unsafe { &*target };

    fix_direction_target(dag, target).expect("Unexpected error occurred in GateDirection");
}
