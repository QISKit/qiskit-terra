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

use crate::pointers::const_ptr_as_ref;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit_library::qdrift::qdrift_evolution;
use qiskit_quantum_info::sparse_observable::SparseObservable;

/// @ingroup QkCircuitLibrary
///
/// Implements the QDrift Trotterization method, which selects Hamiltonian terms
/// randomly with probability proportional to their absolute coefficients.
/// This implementation follows the method introduced by Ref. [1].
///
/// @param op Pointer to a valid ``QkObs`` containing the sum of the Pauli terms.
/// @param reps The number of times to repeat the Trotter steps.
/// @param time Evolution time t in \f$exp(-itH)\f$.
/// @param seed An optional seed for reproducibility of the random sampling process.
///   To set no seed, pass -1.
/// @param preserve_order If ``false``, allows reordering the terms of the operator to
///   potentially yield a shallower evolution circuit. Not relevant
///   when synthesizing an observable with a single term.
/// @param insert_barriers  Whether to insert barriers between the terms evolutions.
///
/// @return A pointer to the generated circuit. ``NULL`` if the circuit generation
///   failed.
///
/// # Example
/// ```c
/// // 1-qubit observable H = X + Y
/// QkObs *obs = qk_obs_zero(1);
///
/// QkBitTerm op1_bits[1] = {QkBitTerm_X};
/// QkObsTerm term1 = {(QkComplex64){1.0, 0.0}, 1, op1_bits, (uint32_t[1]){0}, 1};
/// qk_obs_add_term(obs, &term1);
/// QkBitTerm op2_bits[1] = {QkBitTerm_Y};
/// QkObsTerm term2 = {(QkComplex64){1.0, 0.0}, 1, op2_bits, (uint32_t[1]){0}, 1};
/// qk_obs_add_term(obs, &term2);
///
/// // Passing zero as value for the seed for auto generating a seed value
/// QkCircuit *qc = qk_circuit_library_qdrift(obs, 1, 0.5, 0, true, false);
///
/// qk_obs_free(obs);
/// qk_circuit_free(qc);
/// ```
///
/// # Safety
/// Behavior is undefined ``op`` is not a valid, non-null pointer to a ``QkObs``.
///
/// # References
/// [1] E. Campbell, "A random compiler for fast Hamiltonian simulation",
/// Phys. Rev. Lett. 123, 070503 (2019).
/// [arXiv:quant-ph/1811.08017](https://arxiv.org/abs/1811.08017)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_library_qdrift(
    op: *const SparseObservable, // H in e^{-i t H}
    reps: usize,                 // n in e^{-it/n H}^n
    time: f64,                   // evolution time e in e^{-i t H}
    seed: i64,
    preserve_order: bool,
    insert_barriers: bool,
) -> *mut CircuitData {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let op = unsafe { const_ptr_as_ref(op) };

    let seed = if seed < 0 { None } else { Some(seed as u64) };

    match qdrift_evolution(op, time, reps as u32, seed, preserve_order, insert_barriers) {
        Ok(circuit) => Box::into_raw(Box::new(circuit)),
        Err(_) => std::ptr::null_mut(),
    }
}
