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

use crate::exit_codes::ExitCode;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::commutation_checker::get_standard_commutation_checker;
use qiskit_transpiler::passes::cancel_commutations;
use qiskit_transpiler::target::Target;

/// @ingroup QkTranspilerPassesStandalone
/// Run the CommutativeCancellation transpiler pass on a circuit.
///
/// This pass cancels the redundant (self-adjoint) gates through commutation relations.
///
/// This function is multithreaded and will potentially launch a thread pool
/// with threads equal to the number of CPUs by default. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable.
/// For example, setting ``RAYON_NUM_THREADS=4`` would limit the thread pool
/// to 4 threads.
///
/// @param circuit A pointer to the circuit to run CommutativeCancellation on. This circuit
/// pointer to will be updated with the modified circuit if the pass is able to remove any gates.
/// @param target This pass will attempt to accumulate all Z rotations into either
/// an RZ, P or U1 gate, depending on which is already used in the circuit. If none
/// is present in the circuit, this (optional) target argument is used as fallback to
/// decide which gate to use. If none of RZ, P or U1 are in the circuit or the target,
/// single-qubit Z rotations will not be optimized.
/// @param approximation_degree The approximation degree used when
/// analyzing commutations. Must be within ``(0, 1]``.
/// @returns The integer return code where 0 represents no error and 1 is
/// used to indicate an error was encountered during the execution of the pass.
///
/// # Example
///
/// ```c
/// QkCircuit *qc = qk_circuit_new(4, 0);
/// uint32_t cx_qargs[2] = {0, 1};
/// qk_circuit_gate(qc, QkGate_CX, cx_qargs, NULL);
/// qk_circuit_gate(qc, QkGate_Z, (uint32_t[]){0}, NULL);
/// qk_circuit_gate(qc, QkGate_CX, cx_qargs, NULL);
/// qk_transpiler_pass_standalone_commutative_cancellation(qc, NULL, 1.0);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, ``QkCircuit`` and ``QkTarget``.
/// ``QkCircuit`` is not expected to be null and behavior is undefined if it is.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_commutative_cancellation(
    circuit: *mut CircuitData,
    target: *const Target,
    approximation_degree: f64,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { &mut *circuit };
    let target = if target.is_null() {
        None
    } else {
        // SAFETY: Per documentation, the pointer is non-null and aligned.
        Some(unsafe { &*target })
    };
    if !(0.0..=1.0).contains(&approximation_degree) {
        panic!(
            "Invalid value provided for approximation degree, only NAN or values between 0.0 and 1.0 inclusive are valid"
        );
    }
    let mut dag = match DAGCircuit::from_circuit_data(circuit, false, None, None) {
        Ok(dag) => dag,
        Err(_) => panic!("Internal circuit -> DAG conversion failed"),
    };
    let mut commutation_checker = get_standard_commutation_checker();
    let basis = target.map(|t| t.operation_names().map(|n| n.to_string()).collect());
    if cancel_commutations(
        &mut dag,
        &mut commutation_checker,
        basis,
        approximation_degree,
    )
    .is_err()
    {
        return ExitCode::TranspilerError;
    }
    *circuit = CircuitData::from_dag_ref(&dag).expect("Internal DAG -> circuit conversion failed");
    ExitCode::Success
}

#[cfg(not(miri))]
#[cfg(test)]
mod tests {
    use super::*;
    use qiskit_circuit::Qubit;
    use qiskit_circuit::bit::ShareableQubit;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::operations::StandardGate;

    #[test]
    fn test_commutative_cancellation() {
        let mut qc = CircuitData::new(
            Some((0..2).map(|_| ShareableQubit::new_anonymous()).collect()),
            None,
            (0.).into(),
        )
        .unwrap();
        qc.push_standard_gate(StandardGate::CX, &[], &[Qubit(0), Qubit(1)])
            .unwrap();
        qc.push_standard_gate(StandardGate::Z, &[], &[Qubit(0)])
            .unwrap();
        qc.push_standard_gate(StandardGate::CX, &[], &[Qubit(0), Qubit(1)])
            .unwrap();
        let result = unsafe {
            qk_transpiler_pass_standalone_commutative_cancellation(&mut qc, std::ptr::null(), 1.0)
        };
        assert_eq!(result, ExitCode::Success);
        assert_eq!(qc.len(), 1);
        let Some(gate) = qc.data()[0].op.try_standard_gate() else {
            panic!("Not a standard gate");
        };
        assert_eq!(StandardGate::Z, gate);
    }
}
