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

use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::{
    UnitarySynthesisConfig, UnitarySynthesisState, run_unitary_synthesis,
    unitary_synthesis::Approximation,
};
use qiskit_transpiler::target::Target;

/// @ingroup QkTranspilerPassesStandalone
/// Run the UnitarySynthesis transpiler pass.
///
/// The UnitarySynthesis transpiler pass will synthesize any UnitaryGates in the circuit into gates
/// available in the target.
///
/// This pass is multithreaded and will potentially launch a thread pool
/// with threads equal to the number of CPUs by default. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable.
/// For example, setting ``RAYON_NUM_THREADS=4`` would limit the thread pool
/// to 4 threads.
///
/// @param circuit A pointer to the circuit to run UnitarySynthesis on
/// @param target A pointer to the target to run UnitarySynthesis on
/// @param min_qubits The minimum number of qubits in the unitary to synthesize. If the unitary
///        is less than the specified number of qubits it will not be synthesized.
/// @param approximation_degree heuristic dial used for circuit approximation
///        (1.0=no approximation, 0.0=maximal approximation). Approximation can
///        make the synthesized circuit cheaper at the cost of straying from
///        the original unitary. If NAN, the target approximation is based on gate fidelities
///        in the ``target``.
///
/// # Example
///
/// ```c
///     QkTarget *target = qk_target_new(2);
///     uint32_t current_num_qubits = qk_target_num_qubits(target);
///     QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
///     for (uint32_t i = 0; i < current_num_qubits - 1; i++) {
///         uint32_t qargs[2] = {i, i + 1};
///         double inst_error = 0.0090393 * (current_num_qubits - i);
///         double inst_duration = 0.020039;
///         qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
///     }
///     QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
///     QkCircuit *qc = qk_circuit_new(2, 0);
///     QkComplex64 c0 = {0., 0.};
///     QkComplex64 c1 = {1., 0.};
///     QkComplex64 unitary[16] = {c1, c0, c0, c0,  // row 0
///                                c0, c1, c0, c0,  // row 1
///                                c0, c0, c1, c0,  // row 2
///                                c0, c0, c0, c1}; // row 3
///     uint32_t qargs[2] = {0, 1};
///     qk_circuit_unitary(qc, unitary, qargs, 2, false);
///     qk_transpiler_pass_standalone_unitary_synthesis(qc, target, 0, 1.0);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, non-null pointer to a ``QkCircuit`` and ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_unitary_synthesis(
    circuit: *mut CircuitData,
    target: *const Target,
    min_qubits: usize,
    approximation_degree: f64,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let target = unsafe { const_ptr_as_ref(target) };
    let dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    let approximation =
        Approximation::from_py_approximation_degree(if approximation_degree.is_nan() {
            None
        } else {
            Some(approximation_degree)
        });
    let physical_qubits = (0..dag.num_qubits() as u32)
        .map(PhysicalQubit::new)
        .collect::<Vec<_>>();
    let mut synthesis_state = UnitarySynthesisState::new(UnitarySynthesisConfig {
        approximation,
        run_python_decomposers: false,
        ..Default::default()
    });
    let out_dag = match run_unitary_synthesis(
        &dag,
        &["unitary".to_string()].into_iter().collect(),
        min_qubits,
        &physical_qubits,
        &mut synthesis_state,
        target.into(),
        false,
    ) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    if let Some(out_dag) = out_dag {
        *circuit = CircuitData::from_dag_ref(&out_dag).unwrap();
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
    fn test_pass_synthesis() {
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
            qk_transpiler_pass_standalone_unitary_synthesis(&mut qc, &target, 0, 1.0);
        };
        let mut gate_names = qc.count_ops().keys().copied().collect::<Vec<_>>();
        gate_names.sort();
        assert_eq!(gate_names, vec!["cx", "u"]);
    }
}
