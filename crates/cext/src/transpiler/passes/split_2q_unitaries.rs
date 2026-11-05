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
use qiskit_transpiler::passes::run_split_2q_unitaries;
use qiskit_transpiler::transpile_layout::TranspileLayout;

/// @ingroup QkTranspilerPassesStandalone
/// Run the Split2QUnitaries transpiler pass on a circuit.
///
/// Refer to the ``qk_transpiler_pass_split_2q_unitaries`` function for more details about the pass.
///
/// @param circuit A mutable pointer to the circuit to run Split2QUnitaries on.
/// @param requested_fidelity Allowed tolerance for splitting two-qubit unitaries and gate decompositions.
/// @param split_swaps Whether to attempt to split swap gates, resulting in a permutation of the qubits.
///
/// @return If any swap equivalent unitaries are split this function returns a pointer to a ``TranspileLayout``
///     that contains the permutation induced by this circuit optimization. If no swap equivalent
///     unitaries are split this will be a null pointer.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_split_2q_unitaries(
    circuit: *mut CircuitData,
    requested_fidelity: f64,
    split_swaps: bool,
) -> *mut TranspileLayout {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { &mut *circuit };
    let mut dag = match DAGCircuit::from_circuit_data(circuit, false, None, None) {
        Ok(dag) => dag,
        Err(_e) => panic!("Internal circuit -> DAG conversion failed."),
    };
    let result = run_split_2q_unitaries(&mut dag, requested_fidelity, split_swaps)
        .unwrap_or_else(|_| panic!("Running the Split2qUnitaries pass failed"));
    match result {
        Some((out_dag, permutation)) => {
            let out_circuit = CircuitData::from_dag_ref(&out_dag)
                .expect("Internal DAG -> circuit conversion failed.");
            let qubits = out_circuit.qubits().objects().clone();
            let qregs = out_circuit.qregs().to_vec();
            *circuit = out_circuit;
            Box::into_raw(Box::new(TranspileLayout::new(
                None,
                Some(permutation.into_iter().map(Qubit::new).collect()),
                qubits,
                dag.num_qubits() as u32,
                qregs,
            )))
        }
        None => {
            *circuit = CircuitData::from_dag_ref(&dag)
                .expect("Internal DAG -> circuit conversion failed.");
            std::ptr::null_mut()
        }
    }
}

/// @ingroup QkTranspilerPasses
/// Run the Split2QUnitaries transpiler pass on a DAG Circuit
///
/// @param dag A mutable pointer to the DAG Circuit to run Split2QUnitaries on. This will be
///     replaced with the new DAG if any gates are optimized.
/// @param requested_fidelity Allowed tolerance for splitting two-qubit unitaries and gate decompositions.
/// @param split_swaps Whether to attempt to split swap gates, resulting in a permutation of the qubits.
///
/// @return If any swap equivalent unitaries are split this function returns a pointer to a ``TranspileLayout``
///     that contains the permutation induced by this DAG optimization. If no swap equivalent
///     unitaries are split this will be a null pointer.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDAG``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_split_2q_unitaries(
    dag: *mut DAGCircuit,
    requested_fidelity: f64,
    split_swaps: bool,
) -> *mut TranspileLayout {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { &mut *dag };
    let result = run_split_2q_unitaries(dag, requested_fidelity, split_swaps)
        .unwrap_or_else(|_| panic!("Running the Split2qUnitaries pass failed"));
    match result {
        Some((out_dag, permutation)) => {
            let qubits = out_dag.qubits().objects().clone();
            let qregs = out_dag.qregs().to_vec();
            *dag = out_dag;
            Box::into_raw(Box::new(TranspileLayout::new(
                None,
                Some(permutation.into_iter().map(Qubit::new).collect()),
                qubits,
                dag.num_qubits() as u32,
                qregs,
            )))
        }
        None => std::ptr::null_mut(),
    }
}
