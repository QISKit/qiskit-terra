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

use crate::pointers::ExposesOwnedPointers;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit_library::quantum_volume::quantum_volume;

/// @ingroup QkCircuitLibrary
/// Generate a Quantum Volume model circuit
///
/// The model circuits are random instances of circuits used to measure the Quantum Volume
/// metric, as introduced in [1]. The model circuits consist of layers of Haar random
/// elements of SU(4) applied between corresponding pairs of qubits in a random bipartition.
///
/// This function is multithreaded and will launch a thread pool with threads equal to the
/// number of CPUs by default. You can tune the number of threads with the
/// RAYON_NUM_THREADS environment variable. For example, setting RAYON_NUM_THREADS=4 would
/// limit the thread pool to 4 threads.
///
/// [1] A. Cross et al. Validating quantum computers using randomized model circuits,
///     Phys. Rev. A 100, 032328 (2019). [arXiv:1811.12926](https://arxiv.org/abs/1811.12926)
///
/// @param num_qubits The number qubits to use for the generated circuit.
/// @param depth The number of layers for the generated circuit.
/// @param seed An RNG seed used for generating the random SU(4) matrices used
///   in the output circuit. If the provided number is negative the seed used
///   will be soured from system entropy.
///
/// @return A pointer to the quantum volume circuit.
///
/// # Example
///
/// ```c
/// QkCircuit *qc = qk_circuit_library_quantum_volume(10, 10, -1)
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn qk_circuit_library_quantum_volume(
    num_qubits: u32,
    depth: usize,
    seed: i64,
) -> *mut CircuitData {
    let seed = if seed < 0 { None } else { Some(seed as u64) };
    quantum_volume(num_qubits, depth, seed)
        .unwrap()
        .inner
        .into_leaked()
}
