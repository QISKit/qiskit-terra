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

use qiskit_circuit::{circuit_data::CircuitData, dag_circuit::DAGCircuit};
use qiskit_transpiler::{passes::run_consolidate_blocks, target::Target};

/// @ingroup QkTranspilerPassesStandalone
/// Run the ConsolidateBlocks pass on a circuit.
///
/// ConsolidateBlocks is a transpiler pass that consolidates consecutive blocks of
/// gates operating on the same qubits into a Unitary gate, to later on be
/// resynthesized, which leads to a more optimal subcircuit.
///
/// @param circuit A pointer to the circuit to run ConsolidateBlocks on.
/// @param target A pointer to the target to run ConsolidateBlocks on.
/// @param approximation_degree A float between `[0.0, 1.0]` or a `NaN` which
/// defaults to `1.0`. Lower approximates more.
/// @param force_consolidate: Force block consolidation.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit`` and
/// if ``target`` is not a valid pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_consolidate_blocks(
    circuit: *mut CircuitData,
    target: *const Target,
    approximation_degree: f64,
    force_consolidate: bool,
) {
    // SAFETY: Per documentation, the pointer is not null and aligned.
    let circuit = unsafe { &mut *circuit };
    let target = if target.is_null() {
        None
    } else {
        // SAFETY: Per documentation, the pointer is not null and aligned.
        Some(unsafe { &*target })
    };
    let approximation_degree = if approximation_degree.is_nan() {
        1.0
    } else {
        approximation_degree
    };
    let mut circ_as_dag = DAGCircuit::from_circuit_data(circuit, true, None, None)
        .expect("Error while converting from CircuitData to DAGCircuit.");

    // Call the pass
    run_consolidate_blocks(
        &mut circ_as_dag,
        force_consolidate,
        Some(approximation_degree),
        target,
    )
    .expect("Error running the consolidate blocks pass.");

    let result_circuit = CircuitData::from_dag_ref(&circ_as_dag)
        .expect("Error while converting from DAGCircuit to CircuitData.");
    *circuit = result_circuit;
}
