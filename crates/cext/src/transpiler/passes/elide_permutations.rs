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

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::run_elide_permutations;
use qiskit_transpiler::transpile_layout::TranspileLayout;

/// @ingroup QkTranspilerPassesStandalone
/// Run the ElidePermutations transpiler pass on a circuit.
///
/// Refer to the ``qk_transpiler_pass_elide_permutations`` function for more details about the pass.
///
/// @param circuit A pointer to the circuit to run ElidePermutations on. If there are changes made
///     the object pointed to is changed in place. In case of gates being elided the original circuit's
///     allocations are freed by this function.
///
/// @return the layout object containing the output permutation induced by the elided gates in the
///         circuit. If no elisions are performed this will be a null pointer and the input circuit
///         is unchanged. The caller is responsible for freeing the returned layout by calling
///         ``qk_transpile_layout_free``.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_elide_permutations(
    circuit: *mut CircuitData,
) -> *mut TranspileLayout {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { &mut *circuit };
    let dag = match DAGCircuit::from_circuit_data(circuit, false, None, None) {
        Ok(dag) => dag,
        Err(_e) => panic!("Internal circuit to DAG conversion failed."),
    };
    let res = run_elide_permutations(&dag).expect("ElidePermutations pass failed.");
    match res {
        Some(res) => {
            let out_circuit = CircuitData::from_dag_ref(&res.0)
                .expect("Internal DAG to Circuit conversion failed.");
            let num_input_qubits = circuit.num_qubits() as u32;
            *circuit = out_circuit;
            Box::into_raw(Box::new(TranspileLayout::new(
                None,
                Some(res.1.into_iter().map(Qubit::new).collect()),
                circuit.qubits().objects().clone(),
                num_input_qubits,
                circuit.qregs().to_vec(),
            )))
        }
        None => std::ptr::null_mut(),
    }
}

/// @ingroup QkTranspilerPasses
/// Run the ElidePermutations transpiler pass on a DAG.
///
/// The ElidePermutations transpiler pass removes any permutation operations from a pre-layout
/// DAG.
///
/// This pass is intended to be run before a layout (mapping virtual qubits to physical qubits) is
/// set during the transpilation pipeline. This pass iterates over the DAG
/// and when a Swap gate is encountered it permutes the virtual qubits in
/// the DAG and removes the swap gate. This will effectively remove any
/// swap gates in the DAG prior to running layout. This optimization is
/// not valid after a layout has been set and should not be run in this case.
///
/// @param dag A pointer to the DAG to run ElidePermutations on. If there are changes made
///     the object pointed to is changed in place. In case of gates being elided the original DAG's
///     allocations are freed by this function.
///
/// @return the layout object containing the output permutation induced by the elided gates in the
///         DAG. If no elisions are performed this will be a null pointer and the input DAG
///         is unchanged. The caller is responsible for freeing the returned layout by calling
///         ``qk_transpile_layout_free``.
///
/// # Example
///
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(4, "qr");
/// qk_dag_add_quantum_register(dag, qr);
/// for (uint32_t i = 0; i < qk_dag_num_qubits(dag) - 1; i++) {
///     uint32_t qargs[2] = {i, i + 1};
///     for (uint32_t j = 0; j < i + 1; j++) {
///         qk_dag_apply_gate(dag, QkGate_CX, qargs, NULL, false);
///     }
/// }
/// QkTranspileLayout *elide_result = qk_transpiler_pass_elide_permutations(dag);
/// if (elide_result != NULL) {
///     qk_transpile_layout_free(elide_result);
/// }
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag``  is not a valid, non-null pointer to a ``QkDAG``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_elide_permutations(
    dag: *mut DAGCircuit,
) -> *mut TranspileLayout {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { &mut *dag };
    let res = run_elide_permutations(dag).expect("ElidePermutations pass failed.");
    match res {
        Some(res) => {
            *dag = res.0;
            Box::into_raw(Box::new(TranspileLayout::new(
                None,
                Some(res.1.into_iter().map(Qubit::new).collect()),
                dag.qubits().objects().clone(),
                dag.num_qubits() as u32,
                dag.qregs().to_vec(),
            )))
        }
        None => std::ptr::null_mut(),
    }
}
