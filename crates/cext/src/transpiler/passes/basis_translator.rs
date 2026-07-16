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
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::run_basis_translator;
use qiskit_transpiler::standard_equivalence_library::generate_standard_equivalence_library;
use qiskit_transpiler::target::Target;

/// @ingroup QkTranspilerPassesStandalone
/// Run the BasisTranslator transpiler pass on a circuit.
///
/// The BasisTranslator transpiler pass translates gates to a target basis by
/// searching for a set of translations from the standard EquivalenceLibrary.
///
/// @param circuit A pointer to the circuit to run BasisTranslator on.
/// The circuit will be mutated in-place, unless the circuit is already
/// in the target basis, in which case the circuit remains unchanged.
/// @param target The target where we will obtain basis gates from.
/// @param min_qubits The minimum number of qubits for operations in the input
/// circuit to translate.
///
/// # Example
///
/// ```c
///    #include <qiskit.h>
///
///    QkCircuit *circuit = qk_circuit_new(3, 0);
///    qk_circuit_gate(circuit, QkGate_CCX, (uint32_t[3]){0, 1, 2}, NULL);
///
///    // Create a Target with global properties.
///    QkTarget *target = qk_target_new(3);
///    qk_target_add_instruction(target, qk_target_entry_new(QkGate_H));
///    qk_target_add_instruction(target, qk_target_entry_new(QkGate_T));
///    qk_target_add_instruction(target, qk_target_entry_new(QkGate_Tdg));
///    qk_target_add_instruction(target, qk_target_entry_new(QkGate_CX));
///
///    // Run pass
///    qk_transpiler_pass_standalone_basis_translator(circuit, target, 0);
///
///    // Free the circuit and target pointers once you're done
///    qk_circuit_free(circuit);
///    qk_target_free(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` and/or ``target`` are not valid, non-null
/// pointers to a ``QkCircuit`` or ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_basis_translator(
    circuit: *mut CircuitData,
    target: *const Target,
    min_qubits: usize,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circ_from_ptr = unsafe { mut_ptr_as_ref(circuit) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    let dag = DAGCircuit::from_circuit_data(circ_from_ptr, false, None, None, None, None)
        .expect("Circuit to DAG conversion failed");

    let mut equiv_lib = generate_standard_equivalence_library();

    let result_dag =
        match run_basis_translator(&dag, &mut equiv_lib, min_qubits, Some(target), None) {
            Ok(Some(dag)) => dag,
            Ok(None) => return,
            Err(e) => panic!("{}", e),
        };
    let result_circ =
        CircuitData::from_dag_ref(&result_dag).expect("DAG to Circuit conversion failed");
    *circ_from_ptr = result_circ;
}
