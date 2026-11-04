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
use qiskit_transpiler::passes::convert_to_pauli_rotations;

/// @ingroup QkTranspilerPassesStandalone
/// Run the ``ConvertToPauliRotations`` pass in-place on a circuit.
///
/// This pass converts all standard gates (with less than 4 qubits) in the circuit into a sequence
/// of ``QkPauliProductRotation`` gates and measurements into ``QkPauliProductMeasurement``
/// instructions. Note that this pass panics if the circuit contains non-standard gates.
/// The suggested workflow is to first transpile into a standard basis, keeping rotation gates
/// (such as ``QkGate_RXX`` and others) intact where possible, and then call this pass.
///
/// @param circuit A pointer to the circuit on which to run the pass.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not valid, non-null pointers to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_convert_to_pauli_rotations(
    circuit: *mut CircuitData,
) {
    // SAFETY: The user guarantees the pointer is safe to read.
    let circuit = unsafe { &mut *circuit };
    let dag = match DAGCircuit::from_circuit_data(circuit, false, None, None) {
        Ok(dag) => dag,
        Err(_) => panic!("Internal Circuit -> DAG conversion failed"),
    };
    let out = convert_to_pauli_rotations(&dag).expect("Failed running PBC conversion.");
    // If a DAG is returned, the circuit has been modified. Else just leave it as is.
    *circuit = CircuitData::from_dag_ref(&out).expect("Internal DAG -> Circuit conversion failed");
}
