// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::ffi::{CString, c_char};

use crate::exit_codes::{CInputError, ExitCode};
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref, slice_from_ptr, try_slice_from_ptr};
use num_complex::Complex64;

use qiskit_quantum_info::sparse_observable::{
    BitTerm, CoherenceError, SparseObservable, SparseTermView,
};

/// A term in a ``QkObs``.
///
/// This contains the coefficient (``coeff``), the number of qubits of the observable
/// (``num_qubits``) and pointers to the ``bit_terms`` and ``indices`` arrays, which have
/// length ``len``. It's the responsibility of the user that the data is coherent,
/// see also the below section on safety.
///
/// # Safety
///
/// * ``bit_terms`` must be a non-null, aligned pointer to ``len`` elements of type ``QkBitTerm``.
/// * ``indices`` must be a non-null, aligned pointer to ``len`` elements of type ``uint32_t``.
#[repr(C)]
pub struct CSparseTerm {
    /// The coefficient of the observable term.
    coeff: Complex64,
    /// Length of the ``bit_terms`` and ``indices`` arrays.
    len: usize,
    /// An aligned pointer to ``len`` elements of type ``QkBitTerm``.  This can be ``NULL`` if and
    /// only if ``len`` is 0.
    bit_terms: *mut BitTerm,
    /// An aligned pointer to ``len`` elements of type ``uint32_t``.  This can be ``NULL`` if and
    /// only if ``len`` is 0.
    indices: *mut u32,
    /// The number of qubits the observable term is defined on.
    num_qubits: u32,
}

impl TryFrom<&CSparseTerm> for SparseTermView<'_> {
    type Error = CInputError;

    fn try_from(value: &CSparseTerm) -> Result<Self, Self::Error> {
        Ok(SparseTermView {
            num_qubits: value.num_qubits,
            coeff: value.coeff,
            // SAFETY: per struct documentation, `bit_terms` is valid for `len` initialized reads.
            bit_terms: unsafe { try_slice_from_ptr(value.bit_terms, value.len) }?,
            // SAFETY: per struct documentation, `indices` is valid for `len` initialized reads.
            indices: unsafe { try_slice_from_ptr(value.indices, value.len) }?,
        })
    }
}

/// @ingroup QkObs
/// Construct the zero observable (without any terms).
///
/// @param num_qubits The number of qubits the observable is defined on.
///
/// @return A pointer to the created observable.
///
/// # Example
/// ```c
///     QkObs *zero = qk_obs_zero(100);
/// ```
///
#[unsafe(no_mangle)]
pub extern "C" fn qk_obs_zero(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::zero(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// @ingroup QkObs
/// Construct the identity observable.
///
/// @param num_qubits The number of qubits the observable is defined on.
///
/// @return A pointer to the created observable.
///
/// # Example
/// ```c
///     QkObs *identity = qk_obs_identity(100);
/// ```
///
#[unsafe(no_mangle)]
pub extern "C" fn qk_obs_identity(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::identity(num_qubits);
    Box::into_raw(Box::new(obs))
}

/// @ingroup QkObs
/// Construct a new observable from raw data.
///
/// Any of the pointer arguments may be ``NULL`` if and only if their corresponding length is zero.
///
/// @param num_qubits The number of qubits the observable is defined on.
/// @param num_terms The number of terms.
/// @param num_bits The total number of non-identity bit terms.
/// @param coeffs A pointer to the first element of the coefficients array, which has length
///     ``num_terms``.
/// @param bit_terms A pointer to the first element of the bit terms array, which has length
///     ``num_bits``.
/// @param indices A pointer to the first element of the indices array, which has length
///     ``num_bits``. Note that, per term, these *must* be sorted incrementally.
/// @param boundaries A pointer to the first element of the boundaries array, which has length
///     ``num_terms + 1``.
///
/// @return If the input data was coherent and the construction successful, the result is a pointer
///     to the observable. Otherwise a null pointer is returned.
///
/// # Example
/// ```c
///    // define the raw data for the 100-qubit observable |01><01|_{0, 1} - |+-><+-|_{98, 99}
///    uint32_t num_qubits = 100;
///    uint64_t num_terms = 2;  // we have 2 terms: |01><01|, -1 * |+-><+-|
///    uint64_t num_bits = 4; // we have 4 non-identity bits: 0, 1, +, -
///    QkComplex64 coeffs[] = {{1, 0}, {-1, 0}};
///    QkBitTerm bits[4] = {QkBitTerm_Zero, QkBitTerm_One, QkBitTerm_Plus, QkBitTerm_Minus};
///
///    uint32_t indices[4] = {0, 1, 98, 99};  // <-- e.g. {1, 0, 99, 98} would be invalid
///    size_t boundaries[3] = {0, 2, 4};
///    QkObs *obs = qk_obs_new(
///        num_qubits, num_terms, num_bits, coeffs, bits, indices, boundaries
///    );
///    qk_obs_free(obs);
/// ```
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
///   * ``coeffs`` is a pointer to a ``QkComplex64`` array of length ``num_terms``
///   * ``bit_terms`` is a pointer to an array of valid ``QkBitTerm`` elements of length ``num_bits``
///   * ``indices`` is a pointer to a ``uint32_t`` array of length ``num_bits``, which is
///     term-wise sorted in strict ascending order, and every element is smaller than ``num_qubits``
///   * ``boundaries`` is a pointer to a ``size_t`` array of length ``num_terms + 1``, which is
///     sorted in ascending order, the first element is 0 and the last element is
///     smaller than ``num_terms``
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_new(
    num_qubits: u32,
    num_terms: u64,
    num_bits: u64,
    coeffs: *mut Complex64,
    bit_terms: *mut BitTerm,
    indices: *mut u32,
    boundaries: *mut usize,
) -> *mut SparseObservable {
    let num_terms = num_terms as usize;
    let num_bits = num_bits as usize;
    SparseObservable::new(
        num_qubits,
        // SAFETY: per documentation, each pointer is valid for the given number of reads.
        unsafe { slice_from_ptr(coeffs, num_terms) }.to_vec(),
        unsafe { slice_from_ptr(bit_terms, num_bits) }.to_vec(),
        unsafe { slice_from_ptr(indices, num_bits) }.to_vec(),
        unsafe { slice_from_ptr(boundaries, num_terms + 1) }.to_vec(),
    )
    .map(|obs| Box::into_raw(Box::new(obs)))
    .unwrap_or(::std::ptr::null_mut())
}

/// @ingroup QkObs
/// Free the observable.
///
/// @param obs A pointer to the observable to free.
///
/// # Example
/// ```c
///     QkObs *obs = qk_obs_zero(100);
///     qk_obs_free(obs);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``obs`` is not either null or a valid pointer to a ``QkObs``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_free(obs: *mut SparseObservable) {
    if !obs.is_null() {
        if !obs.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.");
        }
        // SAFETY: per documentation, `obs` points to a valid boxed `SparseObservable`.  Per above
        // checks, it is aligned and non-null.
        let _ = unsafe { Box::from_raw(obs) };
    }
}

/// @ingroup QkObs
/// Add a term to the observable.
///
/// @param obs A pointer to the observable.
/// @param cterm A pointer to the term to add.
///
/// @return An exit code. This is ``>0`` if the term is incoherent or adding the term fails.
///
/// # Example
/// ```c
/// uint32_t num_qubits = 100;
/// QkObs *obs = qk_obs_zero(num_qubits);
///
/// QkComplex64 coeff = {1, 0};
/// QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
/// uint32_t indices[3] = {0, 1, 2};
/// QkObsTerm term = {coeff, 3, bit_terms, indices, num_qubits};
///
/// QkExitCode exit_code = qk_obs_add_term(obs, &term);
/// ```
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated:
///
///   * ``obs`` is a valid, non-null pointer to a ``QkObs``
///   * ``cterm`` is a valid, non-null pointer to a ``QkObsTerm``
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_add_term(
    obs: *mut SparseObservable,
    cterm: *const CSparseTerm,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let obs = unsafe { mut_ptr_as_ref(obs) };
    let cterm = unsafe { const_ptr_as_ref(cterm) };

    let view = match cterm.try_into() {
        Ok(view) => view,
        Err(err) => return ExitCode::from(err),
    };

    match obs.add_term(view) {
        Ok(_) => ExitCode::Success,
        Err(err) => ExitCode::from(err),
    }
}

/// @ingroup QkObs
/// Get an observable term by reference.
///
/// A ``QkObsTerm`` contains pointers to the indices and bit terms in the term, which
/// can be used to modify the internal data of the observable. This can leave the observable
/// in an incoherent state and should be avoided, unless great care is taken. It is generally
/// safer to construct a new observable instead of attempting in-place modifications.
///
/// @param obs A pointer to the observable.
/// @param index The index of the term to get.
/// @param out A pointer to a ``QkObsTerm`` used to return the observable term.
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkObs *obs = qk_obs_identity(100);
///     QkObsTerm term;
///     QkExitCode exit_code = qk_obs_term(obs, 0, &term);
///     // out-of-bounds indices return an error code
///     // QkExitCode error = qk_obs_term(obs, 12, &term);
/// ```
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated
/// * ``obs`` is a valid, non-null pointer to a ``QkObs``
/// * ``out`` is a valid, non-null pointer to a ``QkObsTerm``
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_term(
    obs: *mut SparseObservable,
    index: u64,
    out: *mut CSparseTerm,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let obs = unsafe { mut_ptr_as_ref(obs) };

    let index = index as usize;
    if index > obs.num_terms() {
        return ExitCode::IndexError;
    }

    out.len = obs.boundaries()[index + 1] - obs.boundaries()[index];
    out.coeff = obs.coeffs()[index];
    out.num_qubits = obs.num_qubits();

    let start = obs.boundaries()[index];
    out.bit_terms = obs.bit_terms_mut().as_mut_ptr().wrapping_add(start);
    // SAFETY: mutating the indices can leave the observable in an incoherent state.
    out.indices = unsafe { obs.indices_mut().as_mut_ptr().wrapping_add(start) };

    ExitCode::Success
}

/// @ingroup QkObs
/// Get the number of terms in the observable.
///
/// @param obs A pointer to the observable.
///
/// @return The number of terms in the observable.
///
/// # Example
/// ```c
///     QkObs *obs = qk_obs_identity(100);
///     size_t num_terms = qk_obs_num_terms(obs);  // num_terms==1
/// ```
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_num_terms(obs: *const SparseObservable) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    obs.num_terms()
}

/// @ingroup QkObs
/// Get the number of qubits the observable is defined on.
///
/// @param obs A pointer to the observable.
///
/// @return The number of qubits the observable is defined on.
///
/// # Example
/// ```c
///     QkObs *obs = qk_obs_identity(100);
///     uint32_t num_qubits = qk_obs_num_qubits(obs);  // num_qubits==100
/// ```
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_num_qubits(obs: *const SparseObservable) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    obs.num_qubits()
}

/// @ingroup QkObs
/// Get the number of bit terms/indices in the observable.
///
/// @param obs A pointer to the observable.
///
/// @return The number of terms in the observable.
///
/// # Example
/// ```c
///     QkObs *obs = qk_obs_identity(100);
///     size_t len = qk_obs_len(obs);  // len==0, as there are no non-trivial bit terms
/// ```
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_len(obs: *const SparseObservable) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    obs.bit_terms().len()
}

/// @ingroup QkObs
/// Get a pointer to the coefficients.
///
/// This can be used to read and modify the observable's coefficients. The resulting
/// pointer is valid to read for ``qk_obs_num_terms(obs)`` elements of ``QkComplex64``.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the coefficients.
///
/// # Example
/// ```c
///    QkObs *obs = qk_obs_identity(100);
///    size_t num_terms = qk_obs_num_terms(obs);
///    QkComplex64 *coeffs = qk_obs_coeffs(obs);
///
///    for (size_t i = 0; i < num_terms; i++) {
///        printf("%f + i%f\n", coeffs[i].re, coeffs[i].im);
///    }
/// ```
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_coeffs(obs: *mut SparseObservable) -> *mut Complex64 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { mut_ptr_as_ref(obs) };

    obs.coeffs_mut().as_mut_ptr()
}

/// @ingroup QkObs
/// Get a pointer to the indices.
///
/// This can be used to read and modify the observable's indices. The resulting pointer is
/// valid to read for ``qk_obs_len(obs)`` elements of size ``uint32_t``.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the indices.
///
/// # Example
/// ```c
/// uint32_t num_qubits = 100;
/// QkObs *obs = qk_obs_zero(num_qubits);
///
/// QkComplex64 coeff = {1, 0};
/// QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
/// uint32_t term_indices[3] = {0, 1, 2};
/// QkObsTerm term = {coeff, 3, bit_terms, term_indices, num_qubits};
/// qk_obs_add_term(obs, &term);
///
/// size_t len = qk_obs_len(obs);
/// uint32_t *indices = qk_obs_indices(obs);
///
/// for (size_t i = 0; i < len; i++) {
///     printf("index %i: %i\n", i, indices[i]);
/// }
///
/// qk_obs_free(obs);
/// ```
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_indices(obs: *mut SparseObservable) -> *mut u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { mut_ptr_as_ref(obs) };

    // SAFETY: Mutating the indices can leave the observable in an incoherent state.
    unsafe { obs.indices_mut() }.as_mut_ptr()
}

/// @ingroup QkObs
/// Get a pointer to the term boundaries.
///
/// This can be used to read and modify the observable's term boundaries. The resulting pointer is
/// valid to read for ``qk_obs_num_terms(obs) + 1`` elements of size ``size_t``.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the boundaries.
///
/// # Example
/// ```c
///    uint32_t num_qubits = 100;
///    QkObs *obs = qk_obs_zero(num_qubits);
///
///    QkComplex64 coeff = {1, 0};
///    QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
///    uint32_t indices[3] = {0, 1, 2};
///    QkObsTerm term = {coeff, 3, bit_terms, indices, num_qubits};
///    qk_obs_add_term(obs, &term);
///
///    size_t num_terms = qk_obs_num_terms(obs);
///    size_t *boundaries = qk_obs_boundaries(obs);
///
///    for (size_t i = 0; i < num_terms + 1; i++) {
///        printf("boundary %i: %i\n", i, boundaries[i]);
///    }
/// ```
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_boundaries(obs: *mut SparseObservable) -> *mut usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { mut_ptr_as_ref(obs) };

    // SAFETY: Modifying the boundaries can leave the observable in an incoherent state. It is
    // the responsibility of the user that the data is coherent.
    unsafe { obs.boundaries_mut() }.as_mut_ptr()
}

/// @ingroup QkObs
/// Get a pointer to the bit terms.
///
/// This can be used to read and modify the observable's bit terms. The resulting pointer is
/// valid to read for ``qk_obs_len(obs)`` elements of size ``uint8_t``.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to the bit terms.
///
/// # Example
/// ```c
///     uint32_t num_qubits = 100;
///     QkObs *obs = qk_obs_zero(num_qubits);
///
///     QkComplex64 coeff = {1, 0};
///     QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
///     uint32_t indices[3] = {0, 1, 2};
///     QkObsTerm term = {coeff, 3, bit_terms, indices, num_qubits};
///     qk_obs_add_term(obs, &term);
///
///     size_t len = qk_obs_len(obs);
///     QkBitTerm *bits = qk_obs_bit_terms(obs);
///
///     for (size_t i = 0; i < len; i++) {
///         printf("bit term %i: %i\n", i, bits[i]);
///     }
///
///     qk_obs_free(obs);
/// ```
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``,
/// or if invalid values are written into the resulting ``QkBitTerm`` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_bit_terms(obs: *mut SparseObservable) -> *mut BitTerm {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { mut_ptr_as_ref(obs) };

    obs.bit_terms_mut().as_mut_ptr()
}

/// @ingroup QkObs
/// Multiply the observable by a complex coefficient.
///
/// @param obs A pointer to the observable.
/// @param coeff The coefficient to multiply the observable with.
///
/// @return A pointer to the result ``obs * coeff``.
///
/// # Example
/// ```c
///     QkObs *obs = qk_obs_identity(100);
///     QkComplex64 coeff = {2, 0};
///     QkObs *result = qk_obs_multiply(obs, &coeff);
/// ```
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated
/// * ``obs`` is a valid, non-null pointer to a ``QkObs``
/// * ``coeff`` is a valid, non-null pointer to a ``QkComplex64``
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_multiply(
    obs: *const SparseObservable,
    coeff: *const Complex64,
) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };
    let coeff = unsafe { const_ptr_as_ref(coeff) };

    let result = obs * (*coeff);
    Box::into_raw(Box::new(result))
}

/// @ingroup QkObs
/// Multiply the observable in-place by a complex coefficient.
///
/// @param obs A pointer to the observable.
/// @param coeff The coefficient to multiply the observable with.
///
/// # Example
/// ```c
/// QkObs *obs = qk_obs_identity(100);
/// QkComplex64 coeff = {2, 0};
/// qk_obs_multiply_inplace(obs, &coeff);
/// ```
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated
/// * ``obs`` is a valid, non-null pointer to a ``QkObs``
/// * ``coeff`` is a valid, non-null pointer to a ``QkComplex64``
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_multiply_inplace(
    obs: *mut SparseObservable,
    coeff: *const Complex64,
) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let obs = unsafe { mut_ptr_as_ref(obs) };
    let coeff = unsafe { const_ptr_as_ref(coeff) };

    *obs *= *coeff;
}

/// @ingroup QkObs
/// Add two observables.
///
/// @param left A pointer to the left observable.
/// @param right A pointer to the right observable.
///
/// @return A pointer to the result ``left + right``.
///
/// # Example
/// ```c
///     QkObs *left = qk_obs_identity(100);
///     QkObs *right = qk_obs_zero(100);
///     QkObs *result = qk_obs_add(left, right);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``left`` or ``right`` are not valid, non-null pointers to
/// ``QkObs``\ s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_add(
    left: *const SparseObservable,
    right: *const SparseObservable,
) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    let result = left + right;
    Box::into_raw(Box::new(result))
}

/// @ingroup QkObs
/// Add an observable to an existing one.
///
/// @param left A pointer to the left observable.
/// @param right A pointer to the right observable.
///
/// # Example
/// ```c
/// QkObs *left = qk_obs_identity(100);
/// QkObs *right = qk_obs_zero(100);
/// qk_obs_add_inplace(left, right);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``left`` or ``right`` are not valid, non-null pointers to
/// ``QkObs``\ s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_add_inplace(
    left: *mut SparseObservable,
    right: *const SparseObservable,
) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { mut_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };

    *left += right;
}

/// @ingroup QkObs
/// Add two observables while scaling the coefficients of the right one.
///
/// @param left A pointer to the left observable.
/// @param right A pointer to the right observable.
/// @param factor The factor to multiply the coefficients with.
///
/// @return An owned pointer to the result ``left + factor * right``.
///
/// # Example
/// ```c
/// QkObs *left = qk_obs_zero(100);
/// QkObs *right = qk_obs_identity(100);
/// QkComplex64 factor = {2, 0};
/// QkObs *result = qk_obs_scaled_add(left, right, &factor);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``left`` or ``right`` are not valid, non-null pointers to
/// ``QkObs``\ s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_scaled_add(
    left: *const SparseObservable,
    right: *const SparseObservable,
    factor: *const Complex64,
) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { const_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };
    let factor = unsafe { const_ptr_as_ref(factor) };

    let result = left.scaled_add(right, *factor);
    Box::into_raw(Box::new(result))
}

/// @ingroup QkObs
/// Add a scaled observable to an existing one.
///
/// @param left A pointer to the left observable.
/// @param right A pointer to the right observable.
/// @param factor The factor to multiply the coefficients with.
///
/// # Example
/// ```c
/// QkObs *left = qk_obs_zero(100);
/// QkObs *right = qk_obs_identity(100);
/// QkComplex64 factor = {2, 0};
/// qk_obs_scaled_add_inplace(left, right, &factor);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``left`` or ``right`` are not valid, non-null pointers to
/// ``QkObs``\ s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_scaled_add_inplace(
    left: *mut SparseObservable,
    right: *const SparseObservable,
    factor: *const Complex64,
) {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let left = unsafe { mut_ptr_as_ref(left) };
    let right = unsafe { const_ptr_as_ref(right) };
    let factor = unsafe { const_ptr_as_ref(factor) };

    left.scaled_add_inplace(right, *factor);
}

/// @ingroup QkObs
/// Compose (multiply) two observables.
///
/// @param first One observable.
/// @param second The other observable.
///
/// @return ``first.compose(second)`` which equals the observable ``result = second @ first``,
///     in terms of the matrix multiplication ``@``.
///
/// # Example
/// ```c
///     QkObs *first = qk_obs_zero(100);
///     QkObs *second = qk_obs_identity(100);
///     QkObs *result = qk_obs_compose(first, second);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``first`` or ``second`` are not valid, non-null pointers to
/// ``QkObs``\ s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_compose(
    first: *const SparseObservable,
    second: *const SparseObservable,
) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let first = unsafe { const_ptr_as_ref(first) };
    let second = unsafe { const_ptr_as_ref(second) };

    let result = first.compose(second);
    Box::into_raw(Box::new(result))
}

/// @ingroup QkObs
/// Compose (multiply) two observables according to a custom qubit order.
///
/// Notably, this allows composing two observables of different size.
///
/// @param first One observable.
/// @param second The other observable. The number of qubits must match the length of ``qargs``.
/// @param qargs The qubit arguments specified which indices in ``first`` to associate with
///     the ones in ``second``.
///
/// @return ``first.compose(second)`` which equals the observable ``result = second @ first``,
///     in terms of the matrix multiplication ``@``.
///
/// # Example
/// ```c
///     QkObs *first = qk_obs_zero(100);
///     QkObs *second = qk_obs_identity(100);
///     QkObs *result = qk_obs_compose(first, second);
/// ```
///
/// # Safety
///
/// To call this function safely
///
///   * ``first`` and ``second`` must be valid, non-null pointers to ``QkObs``\ s
///   * ``qargs`` must point to an array of ``uint32_t``, readable for ``qk_obs_num_qubits(second)``
///     elements (meaning the number of qubits in ``second``)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_compose_map(
    first: *const SparseObservable,
    second: *const SparseObservable,
    qargs: *const u32,
) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let first = unsafe { const_ptr_as_ref(first) };
    let second = unsafe { const_ptr_as_ref(second) };

    // SAFETY: per documentation, `qargs` is valid for `second.num_qubits()` reads.
    let qargs = unsafe { slice_from_ptr(qargs, second.num_qubits() as usize) };
    let qargs_map = |index: u32| qargs[index as usize];

    let result = first.compose_map(second, qargs_map);
    Box::into_raw(Box::new(result))
}

/// @ingroup QkObs
/// Apply a new qubit layout to the observable.
///
/// The layout is set by an array ``layout`` of new indices, specifying that qubit at current
/// index ``i`` is relabelled to index ``layout[i]``. The number of qubits the observable
/// acts on can be extended by setting a larger ``num_qubits`` than the current observable has.
///
/// @param obs A pointer to the observable, this observable will be modified in place upon success.
/// Check the exit code to ensure the layout was correctly applied.
/// @param layout A pointer to the layout. The pointer must point to an array to
/// ``qk_obs_num_qubits(obs)`` elements of type ``uint32_t``. Each element must have values
/// in ``[0, num_qubits)``.
/// @param num_qubits The number of output qubits.
///
/// @return An exit code.
/// * ``QkExitCode_Success`` upon success
/// * ``QkExitCode_DuplicteIndexError`` if duplicate qubit indices were found
/// * ``QkExitCode_MismatchedQubits`` if ``num_qubits`` is smaller than the number of qubits in
///   the observable
/// * ``QkExitCode_IndexError`` for any other index errors, such as invalid values in ``layout``.
///
/// # Example
///
/// This interface allows to relabel and extend the qubit indices:
///
/// ```c
/// QkObs *obs = qk_obs_zero(4);
///
/// // add a term to the observable
/// QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
/// uint32_t qubits[3] = {1, 2, 3};
/// complex double coeff = 1;
/// QkObsTerm term = {coeff, 3, bit_terms, qubits, 4};
/// qk_obs_add_term(obs, &term);
///
/// uint32_t layout[3] = {0, 10, 9};  // qubit mapping is: 0->0, 1->10, 2->9
/// uint32_t num_output_qubits = 11;
/// QkExitCode exit = qk_obs_apply_layout(obs, layout, num_output_qubits);
/// ```
///
/// In a compiler workflow, this function can conveniently be used to apply a
/// ``QkTranspileLayout*`` obtained from a transpiler pass, called ``transpile_layout``
/// in the following example:
///
/// ```c
/// // get the number of output qubits
/// uint32_t num_output_qubits = qk_transpile_layout_num_output_qubits(transpile_layout);
///
/// // get the layout including the ancillas (hence the ``false`` in the function call)
/// uint32_t *layout = malloc(sizeof(uint32_t) * num_output_qubits);
/// qk_transpile_layout_final_layout(transpile_layout, false, layout);
///
/// // apply the layout
/// QkExitCode exit = qk_obs_apply_layout(obs, layout, num_output_qubits);
///
/// // free the layout array
/// free(layout);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``obs`` is not a valid, non-null pointer to ``QkObs`` or if ``layout``
/// is not a valid, non-null pointer to a sequence of ``qk_obs_num_qubits(obs)`` consecutive
/// elements of ``uint32_t``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_apply_layout(
    obs: *mut SparseObservable,
    layout: *const u32,
    num_qubits: u32,
) -> ExitCode {
    // SAFETY: Per documentation, this is a valid, non-null pointer.
    let obs = unsafe { mut_ptr_as_ref(obs) };

    let layout = if layout.is_null() {
        None
    } else {
        let len = obs.num_qubits() as usize;
        // SAFETY: Per documentation, ``layout`` is readable for ``obs.num_qubits()`` elements.
        let indices = unsafe { ::std::slice::from_raw_parts(layout, len) };
        Some(indices)
    };

    let obs_with_layout = match obs.apply_layout(layout, num_qubits) {
        Ok(obs_with_layout) => obs_with_layout,
        Err(e) => {
            return match e {
                CoherenceError::DuplicateIndices => ExitCode::DuplicateIndexError,
                CoherenceError::NotEnoughQubits { .. } => ExitCode::MismatchedQubits,
                _ => ExitCode::IndexError,
            };
        }
    };

    *obs = obs_with_layout;
    ExitCode::Success
}

/// @ingroup QkObs
/// Calculate the canonical representation of the observable.
///
/// @param obs A pointer to the observable.
/// @param tol The tolerance below which coefficients are considered to be zero.
///
/// @return The canonical representation of the observable.
///
/// # Example
/// ```c
///    QkObs *iden = qk_obs_identity(100);
///    QkObs *two = qk_obs_add(iden, iden);
///
///    double tol = 1e-6;
///    QkObs *canonical = qk_obs_canonicalize(two, tol);
/// ```
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_canonicalize(
    obs: *const SparseObservable,
    tol: f64,
) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    let result = obs.canonicalize(tol);
    Box::into_raw(Box::new(result))
}

/// @ingroup QkObs
/// Copy the observable.
///
/// @param obs A pointer to the observable.
///
/// @return A pointer to a copy of the observable.
///
/// # Example
/// ```c
///     QkObs *original = qk_obs_identity(100);
///     QkObs *copied = qk_obs_copy(original);
/// ```
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_copy(obs: *const SparseObservable) -> *mut SparseObservable {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    let copied = obs.clone();
    Box::into_raw(Box::new(copied))
}

/// @ingroup QkObs
/// Compare two observables for equality.
///
/// Note that this does not compare mathematical equality, but data equality. This means
/// that two observables might represent the same observable but not compare as equal.
///
/// @param obs A pointer to one observable.
/// @param other A pointer to another observable.
///
/// @return ``true`` if the observables are equal, ``false`` otherwise.
///
/// # Example
/// ```c
///     QkObs *observable = qk_obs_identity(100);
///     QkObs *other = qk_obs_identity(100);
///     bool are_equal = qk_obs_equal(observable, other);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``obs`` or ``other`` are not valid, non-null pointers to
/// ``QkObs``\ s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_equal(
    obs: *const SparseObservable,
    other: *const SparseObservable,
) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };
    let other = unsafe { const_ptr_as_ref(other) };

    obs.eq(other)
}

/// @ingroup QkObs
/// Return a string representation of a ``QkObs``.
///
/// @param obs A pointer to the ``QkObs`` to get the string for.
///
/// @return A pointer to a nul-terminated char array of the string representation for ``obs``
///
/// # Example
/// ```c
///     QkObs *obs = qk_obs_identity(100);
///     char *string = qk_obs_str(obs);
///     qk_str_free(string);
///     qk_obs_free(obs);
/// ```
///
/// # Safety
///
/// Behavior is undefined ``obs`` is not a valid, non-null pointer to a ``QkObs``.
///
/// The string must not be freed with the normal C free, you must use ``qk_str_free`` to
/// free the memory consumed by the String. Not calling ``qk_str_free`` will lead to a
/// memory leak.
///
/// Do not change the length of the string after it's returned (by writing a nul byte somewhere
/// inside the string or removing the final one), although values can be mutated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obs_str(obs: *const SparseObservable) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let obs = unsafe { const_ptr_as_ref(obs) };

    let num_terms = obs.num_terms();
    let num_qubits = obs.num_qubits();

    let str_num_terms = format!(
        "{} term{}",
        num_terms,
        if num_terms == 1 { "" } else { "s" }
    );
    let str_num_qubits = format!(
        "{} qubit{}",
        num_qubits,
        if num_qubits == 1 { "" } else { "s" }
    );

    let str_terms = if num_terms == 0 {
        "0.0".to_owned()
    } else {
        obs.iter()
            .map(SparseTermView::to_sparse_str)
            .collect::<Vec<_>>()
            .join(" + ")
    };

    let obs_str = format!("<QkObs with {str_num_terms} on {str_num_qubits}: {str_terms}>");
    CString::new(obs_str).unwrap().into_raw()
}

/// @ingroup QkObs
/// Free a string representation.
///
/// @param string A pointer to the returned string representation from ``qk_obs_str`` or
///     ``qk_obsterm_str``.
///
/// # Safety
///
/// Behavior is undefined if ``str`` is not a pointer returned by ``qk_obs_str`` or
/// ``qk_obsterm_str``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_str_free(string: *mut c_char) {
    if !string.is_null() {
        if !string.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }
        // SAFETY: Per docstring the pointer is obtained by Qiskit functions, which are
        // returning strings from CString::into_raw.
        unsafe {
            let _ = CString::from_raw(string);
        }
    }
}

/// @ingroup QkObsTerm
/// Return a string representation of the sparse term.
///
/// @param term A pointer to the term.
///
/// @return A pointer to a nul-terminated char array of the string representation for ``term``
///
/// # Example
/// ```c
///     QkObs *obs = qk_obs_identity(100);
///     QkObsTerm term;
///     qk_obs_term(obs, 0, &term);
///     char *string = qk_obsterm_str(&term);
///     qk_str_free(string);
///     qk_obs_free(obs);
/// ```
///
/// # Safety
///
/// Behavior is undefined ``term`` is not a valid, non-null pointer to a ``QkObsTerm``.
///
/// The string must not be freed with the normal C free, you must use ``qk_str_free`` to
/// free the memory consumed by the String. Not calling ``qk_str_free`` will lead to a
/// memory leak.
///
/// Do not change the length of the string after it's returned, although values can be mutated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_obsterm_str(term: *const CSparseTerm) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let term = unsafe { const_ptr_as_ref(term) };

    let view: SparseTermView = term.try_into().unwrap();

    let string: String = format!(
        "<QkObsTerm on {} qubit{}: {}>",
        view.num_qubits,
        if view.num_qubits == 1 { "" } else { "s" },
        view.to_sparse_str()
    );
    CString::new(string).unwrap().into_raw()
}

/// @ingroup QkBitTerm
/// Get the label for a bit term.
///
/// @param bit_term The bit term.
///
/// @return The label as ``uint8_t``, which can be cast to ``char`` to obtain the character.
///
/// # Example
/// ```c
///     QkBitTerm bit_term = QkBitTerm_Y;
///     // cast the uint8_t to char
///     char label = qk_bitterm_label(bit_term);
/// ```
///
/// # Safety
///
/// The behavior is undefined if ``bit_term`` is not a valid ``uint8_t`` value of a ``QkBitTerm``.
#[unsafe(no_mangle)]
pub extern "C" fn qk_bitterm_label(bit_term: BitTerm) -> u8 {
    // BitTerm is implemented as u8, which is calling convention compatible with C,
    // hence we can pass ``bit_term`` by value
    bit_term
        .py_label()
        .chars()
        .next()
        .expect("Label has exactly one character") as u8
}

#[cfg(feature = "python_binding")]
mod py {
    use crate::pointers::mut_ptr_as_ref;
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;
    use qiskit_quantum_info::sparse_observable::{PySparseObservable, SparseObservable};
    use std::sync;

    fn try_project_inner_observable<'a>(
        _py: Python<'_>,
        py_obs: &'a mut PySparseObservable,
    ) -> PyResult<&'a mut SparseObservable> {
        sync::Arc::get_mut(&mut py_obs.inner)
            .ok_or_else(|| PyRuntimeError::new_err("observable is not uniquely referenced"))?
            .get_mut()
            .map_err(|_| PyRuntimeError::new_err("inner observable is poisoned"))
    }

    /// @ingroup QkObs
    /// Pass ownership of a `QkObs` object to Python.
    ///
    /// It is not safe to use the `QkObs` pointer after calling this function.  In particular, you
    /// should not attempt to clear or free it.  The caller must own the `QkObs`, not hold a
    /// borrowed reference (for example, a `QkObs *` retrieved from `qk_obs_borrow_from_python` is
    /// not owned).
    ///
    /// @param obs The owned object.
    /// @return An owned Python reference to the object.
    ///
    /// # Safety
    ///
    /// The caller must be attached to a Python interpreter.  Behavior is undefined if `obs` is not
    /// a valid non-null pointer to an initialized and owned `QkObs`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn qk_obs_to_python(
        obs: *mut SparseObservable,
    ) -> *mut ::pyo3::ffi::PyObject {
        // SAFETY: Per documentation, the pointer is aligned and points to valid owned data.
        let obs = unsafe { Box::from_raw(mut_ptr_as_ref(obs)) };
        // SAFETY: Per documentation, we are attached to a Python interpreter.
        let py = unsafe { Python::assume_attached() };
        match Bound::new(py, PySparseObservable::from(*obs)) {
            Ok(ob) => ob.into_ptr(),
            Err(e) => {
                e.restore(py);
                ::std::ptr::null_mut()
            }
        }
    }

    /// @ingroup QkObs
    /// Retrieve a `QkObs` pointer from a Python object.
    ///
    /// This borrows a Python reference and extracts the `QkObs` pointer for it, if it is of
    /// the correct type.  The returned pointer is borrowed from the `ob` pointer.  If the
    /// `PyObject` is not the correct type, the return value is `NULL` and the exception
    /// state of the Python interpreter is set.
    ///
    /// You must be attached to a Python interpreter to call this function.
    ///
    /// You can also use `qk_obs_convert_from_python`, which is logically the exact same as this
    /// function, but can be directly used as a "converter" function for the `PyArg_Parse*`
    /// family of Python converter functions.
    ///
    /// @param ob A borrowed Python object.
    /// @return A pointer to the native object, or `NULL` if the Python object is the wrong type.
    ///
    /// # Safety
    ///
    /// The caller must be attached to a Python interpreter.  Behavior is undefined if `ob` is
    /// not a valid non-null pointer to a Python object.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn qk_obs_borrow_from_python(
        ob: *mut pyo3::ffi::PyObject,
    ) -> *mut SparseObservable {
        // SAFETY: per documentation, we are attached to a Python interpreter, and `ob` is a valid
        // pointer to a PyObject.
        unsafe {
            crate::py::borrow_map_mut(Python::assume_attached(), ob, try_project_inner_observable)
        }
    }

    /// @ingroup QkObs
    /// Retrieve a `QkObs` pointer from a Python object.
    ///
    /// This borrows a Python reference and extracts the `QkObs` pointer for it into `address`, if
    /// it is of the correct type.  The returned pointer is borrowed from the `object` pointer.  If
    /// the `PyObject` is not the correct type, the return value is 1, the exception state of the
    /// Python interpreter is set, and `address` is unchanged.
    ///
    /// You must be attached to a Python interpreter to call this function.
    ///
    /// You can also use `qk_obs_borrow_from_python`, which is logically the exact same as this, but
    /// with a more natural signature for direct usage.
    ///
    /// @param object A borrowed Python object.
    /// @param address The location to write the output to.
    /// @return 1 on success, 0 on failure.
    ///
    /// # Safety
    ///
    /// The caller must be attached to a Python interpreter.  Behavior is undefined if `object`
    /// is not a valid non-null pointer to a Python object, or if `address` is not a pointer to
    /// writeable data of the correct type.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn qk_obs_convert_from_python(
        object: *mut ::pyo3::ffi::PyObject,
        address: *mut ::std::ffi::c_void,
    ) -> ::std::ffi::c_int {
        // SAFETY: per documentation, we are attached to a Python interpreter, `object` is a valid
        // pointer to a PyObject, and `address` points to enough space to write a pointer.
        unsafe {
            crate::py::convert_map_mut(
                Python::assume_attached(),
                object,
                address,
                try_project_inner_observable,
            )
        }
    }
}
#[cfg(feature = "python_binding")]
pub use py::*;
