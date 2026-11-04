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
use ndarray::ArrayView2;
use qiskit_circuit::{circuit_data::CircuitData, operations::Param};
use qiskit_circuit_library::iqp::{check_symmetric, iqp, py_random_iqp};

/// @ingroup QkCircuitLibrary
/// Generate an Instantaneous Quantum Polynomial (IQP) circuit from an
/// integer interaction matrix.
///
/// The `interactions` matrix is interpreted as an `n × n` row-major array of
/// 64-bit integers, where `n = num_qubits`. The diagonal entries set T-like
/// phase powers, and the upper triangle encodes two-qubit CPhase interactions.
///
/// @param num_qubits  Number of logical qubits (`n`). Must match the dimension
///                    of the `interactions` matrix.
/// @param interactions  Pointer to a row-major `n × n` matrix of type
///                      `int64_t`. May be NULL only if `num_qubits == 0`.
/// @param check_input  When `true`, this function verifies that the matrix is
///                     symmetric and returns `NULL` if it is not. When `false`,
///                     no additional validation is performed.
///
/// @return A newly allocated `QkCircuit*` on success (caller must free with
///         `qk_circuit_free`), or `NULL` if `num_qubits > 0` and
///         `interactions` is `NULL`, or if `check_input` is `true` and the
///         matrix is not symmetric.
///
/// # Safety
///
/// If `num_qubits > 0`, `interactions` **must** be a valid, non-null pointer
/// to at least `num_qubits * num_qubits` contiguous `int64_t` values in row-major
/// order. The memory pointed to by `interactions` must be properly aligned,
/// readable for the duration of this call, and not mutably aliased. Passing an
/// invalid pointer or a buffer that is too small results in undefined
/// behaviour.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_library_iqp(
    num_qubits: u32,
    interactions: *const i64, // row-major n×n
    check_input: bool,
) -> *mut CircuitData {
    let num_qubits = num_qubits as usize;

    // For the zero-qubit case, return an empty, but valid, circuit.
    if num_qubits == 0 {
        let circuit = CircuitData::from_standard_gates(
            0,
            std::iter::empty(), // no instructions
            Param::Float(0.0),
        )
        .expect("qk_circuit_library_iqp: failed to build empty IQP circuit");
        return circuit.into_leaked();
    }

    // For n > 0 we require a valid interactions pointer.
    if interactions.is_null() {
        return std::ptr::null_mut();
    }

    // SAFETY: caller guarantees at least num_qubits * num_qubits elements are readable.
    let len = num_qubits * num_qubits;
    let buf = unsafe { std::slice::from_raw_parts(interactions, len) };

    // Wrap the flat buffer as an `ndarray` view with shape (n, n).
    let view: ArrayView2<'_, i64> = ndarray::ArrayView2::from_shape((num_qubits, num_qubits), buf)
        .expect("qk_circuit_library_iqp: interactions buffer is not an n×n row-major matrix");

    // Optional symmetry check on the upper triangle only.
    if check_input && !check_symmetric(&view) {
        // Non-symmetric: return NULL to signal invalid input.
        return std::ptr::null_mut();
    }

    // Build CircuitData and return an owning pointer.
    let circuit_data =
        CircuitData::from_standard_gates(num_qubits as u32, iqp(view), Param::Float(0.0))
            .expect("qk_circuit_library_iqp: failed to build CircuitData from IQP interactions");
    circuit_data.into_leaked()
}

/// @ingroup QkCircuitLibrary
/// Generate a random Instantaneous Quantum Polynomial (IQP) circuit.
///
/// This constructs a random symmetric integer interaction matrix internally
/// and uses it to build an IQP circuit, mirroring the Python
/// `qiskit.circuit.library.IQP` constructor.
///
/// @param num_qubits  Number of qubits.
/// @param seed        RNG seed. If negative, entropy is drawn from the OS;
///                    otherwise, the given value is used as a deterministic
///                    seed.
///
/// @return A newly allocated `QkCircuit*` (caller must free with `qk_circuit_free`).
#[unsafe(no_mangle)]
pub extern "C" fn qk_circuit_library_random_iqp(num_qubits: u32, seed: i64) -> *mut CircuitData {
    let seed = if seed < 0 { None } else { Some(seed as u64) };
    let circuit_data = py_random_iqp(num_qubits, seed)
        .expect("qk_circuit_library_random_iqp: failed to build random IQP circuit");
    Box::into_raw(Box::new(circuit_data.into()))
}
