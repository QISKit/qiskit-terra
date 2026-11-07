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

use crate::pointers::ExposesOwnedPointers;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit_library::random_circuit;

/// The maximum number of operands to use for instructions in the ``qk_random_circuit`` function.
#[repr(u8)]
pub enum MaxOperandsRandomCircuit {
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
}

/// @ingroup QkCircuitLibrary
/// Generate a random circuit of arbitrary size and form.
///
/// This function will generate a random circuit by randomly selecting gates
/// from the set of standard gates in Qiskit.
///
/// @param num_qubits The number qubits to use for the generated circuit.
/// @param depth The number of layers of operations (i.e. critical path length)
/// @param seed An RNG seed used for generating the random circuit. If the
///     provided number is negative the seed used will be soured from system
///     entropy.
/// @param max_operands maximum qubit operands of each gate
/// @param condition Whether to use mid-circuit measurements and if operations
///     in the circuit.
/// @param reset Whether to insert resets into the circuit.
/// @return A pointer to the random circuit
///
/// # Example
///
/// ```c
/// QkCircuit *qc = qk_circuit_library_random_circuit(10, 10, -1, 1, true, false, false)
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn qk_circuit_library_random_circuit(
    num_qubits: u32,
    depth: usize,
    seed: i64,
    max_operands: MaxOperandsRandomCircuit,
    measure: bool,
    conditional: bool,
    reset: bool,
) -> *mut CircuitData {
    let seed = if seed < 0 { None } else { Some(seed as u64) };
    random_circuit::random_circuit(
        num_qubits,
        depth,
        max_operands as u8,
        measure,
        conditional,
        reset,
        seed,
        None,
    )
    .unwrap()
    .into_leaked()
}
