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

use qiskit_circuit::{circuit_data::CircuitData, dag_circuit::DAGCircuit};
use qiskit_transpiler::passes::run_litinski_transformation;

use crate::pointers::mut_ptr_as_ref;

/// @ingroup QkTranspilerPassesStandalone
/// Run the ``LitinskiTransformation`` pass in-place on a circuit.
///
/// This pass commutes all Clifford gates to the end of the circuit, converting Pauli rotation
/// gates into ``QkPauliProductRotation`` gates and measurements into ``QkPauliProductMeasurement``
/// instructions. Note that this pass currently only supports circuits that have ``QkGate_T``,
/// ``QkGate_Tdg`` or ``QkGate_RZ`` gates as non-Cliffords and panics otherwise.  The suggested
/// workflow is to first transpile into a Clifford+RZ basis and then call this pass.
///
/// @param circuit A pointer to the circuit on which to run the pass.
/// @param fix_clifford If ``true``, leave the Clifford gates at the end of the circuit. If
///   ``false`` they are omitted.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not valid, non-null pointers to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_litinski_transformation(
    circuit: *mut CircuitData,
    fix_clifford: bool,
) {
    // SAFETY: The user guarantees the pointer is safe to read.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let dag = DAGCircuit::from_circuit_data(circuit, false, None, None, None, None)
        .expect("Internal Circuit -> DAG conversion failed");

    // For now, run Litinski transformation with no approximation (to avoid C API change).
    let maybe_out = run_litinski_transformation(&dag, fix_clifford, false, true, 1.0)
        .expect("Failed running Litinski transformation");
    // If a DAG is returned, the circuit has been modified. Else just leave it as is.
    if let Some(out) = maybe_out {
        *circuit =
            CircuitData::from_dag_ref(&out).expect("Internal DAG -> Circuit conversion failed")
    };
}
