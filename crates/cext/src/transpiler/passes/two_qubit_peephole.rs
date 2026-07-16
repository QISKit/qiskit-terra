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

use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::two_qubit_unitary_peephole_optimize;
use qiskit_transpiler::target::Target;

/// @ingroup QkTranspilerPassesStandalone
/// Run the TwoQubitPeepholeOptimization transpiler pass.
///
/// Refer to the
/// @verbatim embed:rst:inline :c:func:`qk_transpiler_pass_2q_peephole_optimization` @endverbatim function documentation
/// for the pass details.
///
/// This function is multithreaded, and will perform the analysis in parallel
/// and use all the cores available on your local system. You can refer to
/// the [configuration guide](https://docs.quantum.ibm.com/guides/configure-qiskit-local)
/// for details on how to control the threading behavior for Qiskit more broadly
/// which will also control this pass
///
/// @param circuit A pointer to the circuit to run TwoQubitPeepholeOptimization on
/// @param target A pointer to the target to run TwoQubitPeepholeOptimization with
/// @param approximation_degree heuristic dial used for circuit approximation
///        (1.0=no approximation, 0.0=maximal approximation). Approximation can
///        make the synthesized circuit smaller at the cost of straying from
///        the original unitary. If NAN, the target approximation is based on gate fidelities
///        in the ``target``.
///
/// # Example
///
/// ```c
/// QkTarget *target = qk_target_new(2);
/// uint32_t current_num_qubits = qk_target_num_qubits(target);
/// QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
/// for (uint32_t i = 0; i < current_num_qubits - 1; i++) {
///     uint32_t qargs[2] = {i, i + 1};
///     double inst_error = 0.0090393 * (current_num_qubits - i);
///     double inst_duration = 0.020039;
///     qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
/// }
/// QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
/// QkCircuit *qc = qk_circuit_new(2, 0);
/// uint32_t forward[2] = {0, 1};
/// uint32_t reverse[2] = {1, 0};
/// for (int i = 0; i < 10; i++) {
///     if (i % 2) {
///         qk_circuit_gate(qc, QkGate_CX, forward, NULL);
///     } else {
///         qk_circuit_gate(qc, QkGate_CX, reverse, NULL);
///     }
/// }
/// qk_transpiler_pass_standalone_2q_peephole_optimization(qc, target, 1.0);
/// qk_circuit_free(qc);
/// qk_target_free(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, non-null pointer to a ``QkCircuit`` and ``QkTarget``.
///
/// This function is not safe to call in a context when the circuit contains objects owned by Python. As the
/// function is internally multi-threaded if there are any Python owned gates in the circuit the worker
/// threads may need to acquire the GIL to access the attributes of those gates and this will result in
/// a deadlock. You should use the Python space @verbatim embed:rst:inline :py:class:`.TwoQubitPeepholeOptimization` @endverbatim
/// class to run this pass if there are circuit elements owned by Python.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_2q_peephole_optimization(
    circuit: *mut CircuitData,
    target: *const Target,
    approximation_degree: f64,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let target = unsafe { const_ptr_as_ref(target) };
    let dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    let approximation = if approximation_degree.is_nan() {
        None
    } else {
        Some(approximation_degree)
    };
    let out_dag = match two_qubit_unitary_peephole_optimize(&dag, target, approximation) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    if let Some(out_dag) = out_dag {
        *circuit = CircuitData::from_dag_ref(&out_dag)
            .expect("Output circuit construction from DAG failed");
    }
}

/// @ingroup QkTranspilerPasses
/// Run the TwoQubitPeepholeOptimization transpiler pass.
///
/// This transpiler pass is designed to perform two qubit unitary peephole
/// optimization. This pass finds all the 2 qubit blocks in the circuit,
/// computes the unitary of that block, and then synthesizes that unitary.
/// If the synthesized two qubit unitary is "better" than the original
/// subcircuit that subcircuit is used to replace the original. The heuristic
/// used to determine if it's better first looks at the two qubit gate count
/// in the circuit, and prefers the synthesis with fewer two qubit gates, if
/// the two qubit gate counts are the same then it looks at the estimated
/// fidelity of the circuit and picks the subcircuit with higher estimated
/// fidelity, and finally if needed it picks the subcircuit with the fewest
/// total gates.
///
/// In case the target is overcomplete the pass will try all the
/// decomposers supported for all the gates supported on a given qubit.
/// The decomposition that has the best expected performance using the above
/// heuristic will be selected and used to replace the block.
///
/// This pass is designed to be run on a physical circuit and the details of
/// operations on a given qubit is assumed to be the hardware qubit from the
/// target. However, the output of the pass might not use hardware operations,
/// specifically single qubit gates might be emitted outside the target's supported
/// operations, typically only if a parameterized gate supported by the
/// @verbatim embed:rst:inline :class:`.TwoQubitControlledUDecomposer` @endverbatim is used for synthesis. As such if running
/// this pass in a physical optimization stage (such as @verbatim embed:rst:inline :ref:`transpiler-preset-stage-optimization` @endverbatim)
/// this should be paired with passes such as @verbatim embed:rst:inline :class:`.BasisTranslator`@endverbatim and/or
/// @verbatim embed:rst:inline :class:`.Optimize1qGatesDecomposition` @endverbatim to ensure that these errant single qubit
/// gates are replaced with hardware supported operations prior to exiting the stage.
///
/// This pass is multithreaded, and will perform the analysis in parallel
/// and use all the cores available on your local system. You can refer to
/// the [configuration guide](<https://docs.quantum.ibm.com/guides/configure-qiskit-local>)
/// for details on how to control the threading behavior for Qiskit more broadly
/// which will also control this pass
///
/// @param dag A pointer to the dag circuit to run TwoQubitPeepholeOptimization on
/// @param target A pointer to the target to run TwoQubitPeepholeOptimization with
/// @param approximation_degree heuristic dial used for circuit approximation
///        (1.0=no approximation, 0.0=maximal approximation). Approximation can
///        make the synthesized circuit smaller at the cost of straying from
///        the original unitary. If NAN, the target approximation is based on gate fidelities
///        in the ``target``.
///
/// # Example
///
/// ```c
/// QkTarget *target = qk_target_new(2);
/// uint32_t current_num_qubits = qk_target_num_qubits(target);
/// QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
/// for (uint32_t i = 0; i < current_num_qubits - 1; i++) {
///     uint32_t qargs[2] = {i, i + 1};
///     double inst_error = 0.0090393 * (current_num_qubits - i);
///     double inst_duration = 0.020039;
///     qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
/// }
/// QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
/// QkCircuit *qc = qk_circuit_new(2, 0);
/// uint32_t forward[2] = {0, 1};
/// uint32_t reverse[2] = {1, 0};
/// for (int i = 0; i < 10; i++) {
///     if (i % 2) {
///         qk_circuit_gate(qc, QkGate_CX, forward, NULL);
///     } else {
///         qk_circuit_gate(qc, QkGate_CX, reverse, NULL);
///     }
/// }
/// QkDag *dag = qk_circuit_to_dag(qc);
/// qk_circuit_free(qc);
///
/// qk_transpiler_pass_2q_peephole_optimization(dag, target, 1.0);
/// qk_dag_free(dag);
/// qk_target_free(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` or ``target`` is not a valid, non-null pointer to a ``QkDag`` and ``QkTarget``.
///
/// This function is not safe to call in a context when the DAG circuit contains objects owned by Python. As the
/// function is internally multi-threaded if there are any Python owned gates in the circuit the worker
/// threads may need to acquire the GIL to access the attributes of those gates and this will result in
/// a deadlock. You should use the Python space @verbatim embed:rst:inline :py:class:`.TwoQubitPeepholeOptimization` @endverbatim
/// class to run this pass if there are circuit elements owned by Python.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_2q_peephole_optimization(
    dag: *mut DAGCircuit,
    target: *const Target,
    approximation_degree: f64,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    let target = unsafe { const_ptr_as_ref(target) };
    let approximation = if approximation_degree.is_nan() {
        None
    } else {
        Some(approximation_degree)
    };
    let out_dag = match two_qubit_unitary_peephole_optimize(dag, target, approximation) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    if let Some(out_dag) = out_dag {
        *dag = out_dag;
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;

    use qiskit_circuit::Qubit;
    use qiskit_circuit::bit::ShareableQubit;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::instruction::Parameters;
    use qiskit_circuit::operations::{ArrayType, Param, StandardGate, UnitaryGate};
    use qiskit_circuit::packed_instruction::PackedOperation;
    use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
    use qiskit_circuit::parameter::symbol_expr::Symbol;
    use smallvec::smallvec;
    use std::sync::Arc;

    #[test]
    fn test_pass_peephole() {
        let mut qc = CircuitData::new(
            Some((0..3).map(|_| ShareableQubit::new_anonymous()).collect()),
            None,
            (0.).into(),
        )
        .unwrap();
        let array = StandardGate::CZ.matrix(&[]).unwrap();
        let gate = UnitaryGate {
            array: ArrayType::NDArray(array),
        };
        let operation = PackedOperation::from_unitary(Box::new(gate));
        qc.push_packed_operation(operation, None, &[Qubit(0), Qubit(1)], &[])
            .unwrap();
        let array = StandardGate::DCX.matrix(&[]).unwrap();
        let gate = UnitaryGate {
            array: ArrayType::NDArray(array),
        };
        let operation = PackedOperation::from_unitary(Box::new(gate));
        qc.push_packed_operation(operation, None, &[Qubit(1), Qubit(2)], &[])
            .unwrap();
        let array = StandardGate::DCX.matrix(&[]).unwrap();
        let gate = UnitaryGate {
            array: ArrayType::NDArray(array),
        };
        let operation = PackedOperation::from_unitary(Box::new(gate));
        qc.push_packed_operation(operation, None, &[Qubit(1), Qubit(2)], &[])
            .unwrap();
        let array = StandardGate::Tdg.matrix(&[]).unwrap();
        let gate = UnitaryGate {
            array: ArrayType::NDArray(array),
        };
        let operation = PackedOperation::from_unitary(Box::new(gate));
        qc.push_packed_operation(operation, None, &[Qubit(1)], &[])
            .unwrap();
        let array = StandardGate::CY.matrix(&[]).unwrap();
        let gate = UnitaryGate {
            array: ArrayType::NDArray(array),
        };
        let operation = PackedOperation::from_unitary(Box::new(gate));
        qc.push_packed_operation(operation, None, &[Qubit(0), Qubit(2)], &[])
            .unwrap();

        let mut target = Target::new(
            Some("Fake Target".to_string()),
            Some(3), // num_qubits
            None,    // dt
            None,    // granularity
            None,    // min_length
            None,    // pulse_alignment
            None,    // acquire_alignment
            None,    // qubit_properties
            None,    // concurrent_measurements
        )
        .unwrap();
        let params = Some(Parameters::Params(smallvec![
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("ϴ".to_owned(), None)
            ))),
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("φ".to_owned(), None)
            ))),
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("λ".to_owned(), None)
            ))),
        ]));
        target
            .add_instruction(
                PackedOperation::from_standard_gate(StandardGate::U),
                params,
                None,
                None,
            )
            .unwrap();
        target
            .add_instruction(
                PackedOperation::from_standard_gate(StandardGate::CX),
                None,
                None,
                None,
            )
            .unwrap();
        unsafe {
            qk_transpiler_pass_standalone_2q_peephole_optimization(&mut qc, &target, 1.0);
        };
        let mut gate_names = qc.count_ops().keys().copied().collect::<Vec<_>>();
        gate_names.sort();
        assert_eq!(gate_names, vec!["cx", "u"]);
    }
}
