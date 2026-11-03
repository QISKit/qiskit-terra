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
use qiskit_transpiler::passes::run_inverse_cancellation_standard_gates;

/// @ingroup QkTranspilerPassesStandalone
/// Run the InverseCancellation transpiler pass on a circuit.
///
/// Cancels pairs of consecutive gates that are inverses of each other.
/// The cancelled gates consist of pairs of self-inverse gates:
///    - QkGate_H
///    - QkGate_X
///    - QkGate_Y
///    - QkGate_Z
///    - QkGate_CH
///    - QkGate_CX
///    - QkGate_CY
///    - QkGate_CZ
///    - QkGate_ECR
///    - QkGate_Swap
///    - QkGate_CCX
///    - QkGate_CCZ
///    - QkGate_CSwap
///    - QkGate_RCCX
///    - QkGate_C3X
///
/// and pairs of inverse gates:
///    - (QkGate_T, QkGate_Tdg)
///    - (QkGate_S, QkGate_Sdg)
///    - (QkGate_SX, QkGate_SXdg)
///    - (QkGate_CS, QkGate_CSdg)
///
/// @param circuit A pointer to the circuit to run InverseCancellation on. If the pass is able to
/// remove any gates, the original circuit will be replaced by the circuit produced by this pass.
///
/// # Example
///
/// ```c
/// QkCircuit *qc = qk_circuit_new(2, 2);
/// uint32_t qargs[1] = {0};
/// qk_circuit_gate(qc, QkGate_X, qargs, NULL);
/// qk_circuit_gate(qc, QkGate_H, qargs, NULL);
/// qk_circuit_gate(qc, QkGate_H, qargs, NULL);
/// qk_circuit_gate(qc, QkGate_Y, qargs, NULL);
/// qk_transpiler_pass_standalone_inverse_cancellation(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_inverse_cancellation(
    circuit: *mut CircuitData,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { &mut *circuit };
    let mut dag = match DAGCircuit::from_circuit_data(circuit, false, None, None) {
        Ok(dag) => dag,
        Err(_) => panic!("Internal Circuit -> DAG conversion failed"),
    };

    run_inverse_cancellation_standard_gates(&mut dag);

    *circuit = CircuitData::from_dag_ref(&dag).expect("Internal DAG -> Circuit conversion failed");
}
