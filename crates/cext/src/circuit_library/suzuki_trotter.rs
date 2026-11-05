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
use qiskit_circuit_library::suzuki_trotter::suzuki_trotter_evolution;
use qiskit_quantum_info::sparse_observable::SparseObservable;

/// @ingroup QkCircuitLibrary
/// Generate a circuit using the higher order Suzuki-Trotter product formula from an observable.
///
/// The Suzuki-Trotter formulas improve the error of the Lie-Trotter approximation.
/// In this implementation, the operators are provided as sum terms of a Pauli operator.
/// Higher order decompositions are based on recursions, see Ref. [1] for more details.
///
/// @param op The ``QkObs``  containing the sum of the Pauli terms.
/// @param order The order of the product formula.
/// @param reps The number of time steps.
/// @param time The evolution time.
/// @param preserve_order If ``false``, allows reordering the terms of the operator to
///   potentially yield a shallower evolution circuit. Not relevant
///   when synthesizing an observable with a single term.
/// @param insert_barriers  Whether to insert barriers between the terms evolutions.
///
/// @return A pointer to the generated circuit.
///
/// # Example
/// ```c
/// QkObs *obs = qk_obs_zero(1);
///
/// QkBitTerm op1_bits[1] = {QkBitTerm_X};
/// QkObsTerm term1 = {(QkComplex64){1.0, 0.0}, 1, op1_bits, (uint32_t[1]){0}, 1};
/// qk_obs_add_term(obs, &term1);
///
/// QkBitTerm op2_bits[1] = {QkBitTerm_Y};
/// QkObsTerm term2 = {(QkComplex64){1.0, 0.0}, 1, op2_bits, (uint32_t[1]){0}, 1};
/// qk_obs_add_term(obs, &term2);
///
/// QkCircuit *qc = qk_circuit_library_suzuki_trotter(obs, 2, 1, 0.1, true, false);
///
/// qk_obs_free(obs);
/// qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined ``op`` is not a valid, non-null pointer to a ``QkObs``.
///
/// # References
///
/// [1]: D. Berry, G. Ahokas, R. Cleve and B. Sanders,
/// "Efficient quantum algorithms for simulating sparse Hamiltonians" (2006).
/// [arXiv:quant-ph/0508139](https://arxiv.org/abs/quant-ph/0508139)
///
/// [2]: N. Hatano and M. Suzuki,
/// "Finding Exponential Product Formulas of Higher Orders" (2005).
/// [arXiv:math-ph/0506007](https://arxiv.org/pdf/math-ph/0506007.pdf)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_library_suzuki_trotter(
    op: *const SparseObservable,
    order: u32,
    reps: u32,
    time: f64,
    preserve_order: bool,
    insert_barriers: bool,
) -> *mut CircuitData {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let operator = unsafe { &*op };

    suzuki_trotter_evolution(operator, order, reps, time, preserve_order, insert_barriers)
        .map(ExposesOwnedPointers::into_leaked)
        .unwrap_or_else(|_| std::ptr::null_mut())
}
