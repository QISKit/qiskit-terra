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

mod lookup;

use hashbrown::HashSet;
use itertools::Itertools;
use lookup::conjugate_bitterm;
#[cfg(feature = "python")]
use ndarray::Array2;
use num_complex::Complex64;
#[cfg(feature = "python")]
use num_traits::Zero;
#[cfg(feature = "python")]
use numpy::{
    PyArray1, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyArrayLike1, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
#[cfg(feature = "python")]
use pyo3::{
    IntoPyObjectExt, PyErr,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError, PyZeroDivisionError},
    intern,
    prelude::*,
    sync::PyOnceLock,
    types::{IntoPyDict, PyList, PyString, PyTuple, PyType},
};
#[cfg(feature = "python")]
use qiskit_util::IndexSet;
#[cfg(feature = "python")]
use qiskit_util::py::{ImportOnceCell, PySequenceIndex, SequenceIndex};
#[cfg(feature = "python")]
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
#[cfg(feature = "python")]
use std::sync::{Arc, RwLock, RwLockReadGuard};
use std::{cmp::Ordering, collections::btree_map};
use thiserror::Error;

#[cfg(feature = "python")]
static PAULI_TYPE: ImportOnceCell = ImportOnceCell::new("qiskit.quantum_info", "Pauli");
#[cfg(feature = "python")]
static PAULI_LIST_TYPE: ImportOnceCell = ImportOnceCell::new("qiskit.quantum_info", "PauliList");
#[cfg(feature = "python")]
static SPARSE_PAULI_OP_TYPE: ImportOnceCell =
    ImportOnceCell::new("qiskit.quantum_info", "SparsePauliOp");
#[cfg(feature = "python")]
static BIT_TERM_PY_ENUM: PyOnceLock<Py<PyType>> = PyOnceLock::new();
#[cfg(feature = "python")]
static BIT_TERM_INTO_PY: PyOnceLock<[Option<Py<PyAny>>; 16]> = PyOnceLock::new();

/// Named handle to the alphabet of single-qubit terms.
///
/// This is just the Rust-space representation.  We make a separate Python-space `enum.IntEnum` to
/// represent the same information, since we enforce strongly typed interactions in Rust, including
/// not allowing the stored values to be outside the valid `BitTerm`\ s, but doing so in Python would
/// make it very difficult to use the class efficiently with Numpy array views.  We attach this
/// sister class of `BitTerm` to `SparseObservable` as a scoped class.
///
/// # Representation
///
/// The `u8` representation and the exact numerical values of these are part of the public API.  The
/// low two bits are the symplectic Pauli representation of the required measurement basis with Z in
/// the Lsb0 and X in the Lsb1 (e.g. X and its eigenstate projectors all have their two low bits as
/// `0b10`).  The high two bits are `00` for the operator, `10` for the projector to the positive
/// eigenstate, and `01` for the projector to the negative eigenstate.
///
/// The `0b00_00` representation thus ends up being the natural representation of the `I` operator,
/// but this is never stored, and is not named in the enumeration.
///
/// This operator does not store phase terms of $-i$.  `BitTerm::Y` has `(1, 1)` as its `(z, x)`
/// representation, and represents exactly the Pauli Y operator; any additional phase is stored only
/// in a corresponding coefficient.
///
/// # Dev notes
///
/// This type is required to be `u8`, but it's a subtype of `u8` because not all `u8` are valid
/// `BitTerm`\ s.  For interop with Python space, we accept Numpy arrays of `u8` to represent this,
/// which we transmute into slices of `BitTerm`, after checking that all the values are correct (or
/// skipping the check if Python space promises that it upheld the checks).
///
/// We deliberately _don't_ impl `numpy::Element` for `BitTerm` (which would let us accept and
/// return `PyArray1<BitTerm>` at Python-space boundaries) so that it's clear when we're doing
/// the transmute, and we have to be explicit about the safety of that.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum BitTerm {
    /// Pauli X operator.
    X = 0b00_10,
    /// Projector to the positive eigenstate of Pauli X.
    Plus = 0b10_10,
    /// Projector to the negative eigenstate of Pauli X.
    Minus = 0b01_10,
    /// Pauli Y operator.
    Y = 0b00_11,
    /// Projector to the positive eigenstate of Pauli Y.
    Right = 0b10_11,
    /// Projector to the negative eigenstate of Pauli Y.
    Left = 0b01_11,
    /// Pauli Z operator.
    Z = 0b00_01,
    /// Projector to the positive eigenstate of Pauli Z.
    Zero = 0b10_01,
    /// Projector to the negative eigenstate of Pauli Z.
    One = 0b01_01,
}
impl From<BitTerm> for u8 {
    fn from(value: BitTerm) -> u8 {
        value as u8
    }
}
unsafe impl ::bytemuck::CheckedBitPattern for BitTerm {
    type Bits = u8;

    #[inline(always)]
    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits <= 0b11_11 && (*bits & 0b11_00) < 0b11_00 && (*bits & 0b00_11) != 0
    }
}
unsafe impl ::bytemuck::NoUninit for BitTerm {}

impl BitTerm {
    /// Get the label of this `BitTerm` used in Python-space applications.  This is a single-letter
    /// string.
    #[inline]
    pub fn py_label(&self) -> &'static str {
        // Note: these labels are part of the stable Python API and should not be changed.
        match self {
            Self::X => "X",
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Y => "Y",
            Self::Right => "r",
            Self::Left => "l",
            Self::Z => "Z",
            Self::Zero => "0",
            Self::One => "1",
        }
    }

    /// Get the name of this `BitTerm`, which is how Python space refers to the integer constant.
    #[inline]
    pub fn py_name(&self) -> &'static str {
        // Note: these names are part of the stable Python API and should not be changed.
        match self {
            Self::X => "X",
            Self::Plus => "PLUS",
            Self::Minus => "MINUS",
            Self::Y => "Y",
            Self::Right => "RIGHT",
            Self::Left => "LEFT",
            Self::Z => "Z",
            Self::Zero => "ZERO",
            Self::One => "ONE",
        }
    }

    /// Attempt to convert a `u8` into `BitTerm`.
    ///
    /// Unlike the implementation of `TryFrom<u8>`, this allows `b'I'` as an alphabet letter,
    /// returning `Ok(None)` for it.  All other letters outside the alphabet return the complete
    /// error condition.
    #[inline]
    fn try_from_u8(value: u8) -> Result<Option<Self>, BitTermFromU8Error> {
        match value {
            b'+' => Ok(Some(BitTerm::Plus)),
            b'-' => Ok(Some(BitTerm::Minus)),
            b'0' => Ok(Some(BitTerm::Zero)),
            b'1' => Ok(Some(BitTerm::One)),
            b'I' => Ok(None),
            b'X' => Ok(Some(BitTerm::X)),
            b'Y' => Ok(Some(BitTerm::Y)),
            b'Z' => Ok(Some(BitTerm::Z)),
            b'l' => Ok(Some(BitTerm::Left)),
            b'r' => Ok(Some(BitTerm::Right)),
            _ => Err(BitTermFromU8Error(value)),
        }
    }

    /// Does this term include an X component in its ZX representation?
    ///
    /// This is true for the operators and eigenspace projectors associated with X and Y.
    pub fn has_x_component(&self) -> bool {
        ((*self as u8) & (Self::X as u8)) != 0
    }

    /// Does this term include a Z component in its ZX representation?
    ///
    /// This is true for the operators and eigenspace projectors associated with Y and Z.
    pub fn has_z_component(&self) -> bool {
        ((*self as u8) & (Self::Z as u8)) != 0
    }

    pub fn is_projector(&self) -> bool {
        !matches!(self, BitTerm::X | BitTerm::Y | BitTerm::Z)
    }
}

fn bit_term_as_pauli(bit: &BitTerm) -> &'static [(bool, Option<BitTerm>)] {
    match bit {
        BitTerm::X => &[(true, Some(BitTerm::X))],
        BitTerm::Y => &[(true, Some(BitTerm::Y))],
        BitTerm::Z => &[(true, Some(BitTerm::Z))],
        BitTerm::Plus => &[(true, None), (true, Some(BitTerm::X))],
        BitTerm::Minus => &[(true, None), (false, Some(BitTerm::X))],
        BitTerm::Right => &[(true, None), (true, Some(BitTerm::Y))],
        BitTerm::Left => &[(true, None), (false, Some(BitTerm::Y))],
        BitTerm::Zero => &[(true, None), (true, Some(BitTerm::Z))],
        BitTerm::One => &[(true, None), (false, Some(BitTerm::Z))],
    }
}

/// The error type for a failed conversion into `BitTerm`.
#[derive(Error, Debug)]
#[error("{0} is not a valid letter of the single-qubit alphabet")]
pub struct BitTermFromU8Error(u8);

// `BitTerm` allows safe `as` casting into `u8`.  This is the reverse, which is fallible, because
// `BitTerm` is a value-wise subtype of `u8`.
impl ::std::convert::TryFrom<u8> for BitTerm {
    type Error = BitTermFromU8Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        ::bytemuck::checked::try_cast(value).map_err(|_| BitTermFromU8Error(value))
    }
}

/// Error cases stemming from data coherence at the point of entry into `SparseObservable` from
/// user-provided arrays.
///
/// These most typically appear during [from_raw_parts], but can also be introduced by various
/// remapping arithmetic functions.
///
/// These are generally associated with the Python-space `ValueError` because all of the
/// `TypeError`-related ones are statically forbidden (within Rust) by the language, and conversion
/// failures on entry to Rust from Python space will automatically raise `TypeError`.
#[derive(Error, Debug)]
pub enum CoherenceError {
    #[error("`boundaries` ({boundaries}) must be one element longer than `coeffs` ({coeffs})")]
    MismatchedTermCount { coeffs: usize, boundaries: usize },
    #[error("`bit_terms` ({bit_terms}) and `indices` ({indices}) must be the same length")]
    MismatchedItemCount { bit_terms: usize, indices: usize },
    #[error("the first item of `boundaries` ({0}) must be 0")]
    BadInitialBoundary(usize),
    #[error(
        "the last item of `boundaries` ({last}) must match the length of `bit_terms` and `indices` ({items})"
    )]
    BadFinalBoundary { last: usize, items: usize },
    #[error("all qubit indices must be less than the number of qubits")]
    BitIndexTooHigh,
    #[error("the values in `boundaries` include backwards slices")]
    DecreasingBoundaries,
    #[error("the values in `indices` are not term-wise increasing")]
    UnsortedIndices,
    #[error("the input contains duplicate qubits")]
    DuplicateIndices,
    #[error("the provided qubit mapping does not account for all contained qubits")]
    IndexMapTooSmall,
    #[error("cannot shrink the qubit count in an observable from {current} to {target}")]
    NotEnoughQubits { current: usize, target: usize },
}

/// An error related to processing of a string label (both dense and sparse).
#[derive(Error, Debug)]
pub enum LabelError {
    #[error("label with length {label} cannot be added to a {num_qubits}-qubit operator")]
    WrongLengthDense { num_qubits: u32, label: usize },
    #[error("label with length {label} does not match indices of length {indices}")]
    WrongLengthIndices { label: usize, indices: usize },
    #[error("index {index} is out of range for a {num_qubits}-qubit operator")]
    BadIndex { index: u32, num_qubits: u32 },
    #[error("index {index} is duplicated in a single specifier")]
    DuplicateIndex { index: u32 },
    #[error("labels must only contain letters from the alphabet 'IXYZ+-rl01'")]
    OutsideAlphabet,
}

#[derive(Error, Debug)]
pub enum ArithmeticError {
    #[error("mismatched numbers of qubits: {left}, {right}")]
    MismatchedQubits { left: u32, right: u32 },

    #[error("invalid operation: {0}")]
    InvalidOperation(String),

    #[error("duplicate indices in qargs")]
    DuplicatedIndex,

    #[error("{0}")]
    OutOfBounds(String),
}

/// One part of the type of the iteration value from [PairwiseOrdered].
///
/// The struct iterates over two sorted lists, and returns values from the left iterator, the right
/// iterator, or both simultaneously, depending on some "ordering" key attached to each.  This
/// `enum` is to pass on the information on which iterator is being returned from.
enum Paired<T> {
    Left(T),
    Right(T),
    Both(T, T),
}

/// An iterator combinator that zip-merges two sorted iterators.
///
/// This is created by [pairwise_ordered]; see that method for the description.
struct PairwiseOrdered<C, T, I1, I2>
where
    C: Ord,
    I1: Iterator<Item = (C, T)>,
    I2: Iterator<Item = (C, T)>,
{
    left: ::std::iter::Peekable<I1>,
    right: ::std::iter::Peekable<I2>,
}
impl<C, T, I1, I2> Iterator for PairwiseOrdered<C, T, I1, I2>
where
    C: Ord,
    I1: Iterator<Item = (C, T)>,
    I2: Iterator<Item = (C, T)>,
{
    type Item = (C, Paired<T>);

    fn next(&mut self) -> Option<Self::Item> {
        let order = match (self.left.peek(), self.right.peek()) {
            (None, None) => return None,
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some((left, _)), Some((right, _))) => left.cmp(right),
        };
        match order {
            Ordering::Less => self.left.next().map(|(i, value)| (i, Paired::Left(value))),
            Ordering::Greater => self
                .right
                .next()
                .map(|(i, value)| (i, Paired::Right(value))),
            Ordering::Equal => {
                let (index, left) = self.left.next().unwrap();
                let (_, right) = self.right.next().unwrap();
                Some((index, Paired::Both(left, right)))
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left.size_hint();
        let right = self.right.size_hint();
        (
            left.0.max(right.0),
            left.1.and_then(|left| right.1.map(|right| left.max(right))),
        )
    }
}
/// An iterator combinator that zip-merges two sorted iterators.
///
/// The two iterators must yield the same items, where each item comprises some totally ordered
/// index, and an associated value.  Both input iterators must be sorted in terms of the index.  The
/// output iteration is over 2-tuples, also in sorted order, of the seen ordered index values, and a
/// [Paired] object indicating which iterator (or both) the values were drawn from.
fn pairwise_ordered<C, T, I1, I2>(
    left: I1,
    right: I2,
) -> PairwiseOrdered<C, T, <I1 as IntoIterator>::IntoIter, <I2 as IntoIterator>::IntoIter>
where
    C: Ord,
    I1: IntoIterator<Item = (C, T)>,
    I2: IntoIterator<Item = (C, T)>,
{
    PairwiseOrdered {
        left: left.into_iter().peekable(),
        right: right.into_iter().peekable(),
    }
}

/// An observable over Pauli bases that stores its data in a qubit-sparse format.
///
/// See [PySparseObservable] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseObservable {
    /// The number of qubits the operator acts on.  This is not inferable from any other shape or
    /// values, since identities are not stored explicitly.
    num_qubits: u32,
    /// The coefficients of each abstract term in the sum.  This has as many elements as terms in
    /// the sum.
    coeffs: Vec<Complex64>,
    /// A flat list of single-qubit terms.  This is more naturally a list of lists, but is stored
    /// flat for memory usage and locality reasons, with the sublists denoted by `boundaries.`
    bit_terms: Vec<BitTerm>,
    /// A flat list of the qubit indices that the corresponding entries in `bit_terms` act on.  This
    /// list must always be term-wise sorted, where a term is a sublist as denoted by `boundaries`.
    indices: Vec<u32>,
    /// Indices that partition `bit_terms` and `indices` into sublists for each individual term in
    /// the sum.  `boundaries[0]..boundaries[1]` is the range of indices into `bit_terms` and
    /// `indices` that correspond to the first term of the sum.  All unspecified qubit indices are
    /// implicitly the identity.  This is one item longer than `coeffs`, since `boundaries[0]` is
    /// always an explicit zero (for algorithmic ease).
    boundaries: Vec<usize>,
}

impl SparseObservable {
    /// Create a new observable from the raw components that make it up.
    ///
    /// This checks the input values for data coherence on entry.  If you are certain you have the
    /// correct values, you can call `new_unchecked` instead.
    pub fn new(
        num_qubits: u32,
        coeffs: Vec<Complex64>,
        bit_terms: Vec<BitTerm>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Result<Self, CoherenceError> {
        if coeffs.len() + 1 != boundaries.len() {
            return Err(CoherenceError::MismatchedTermCount {
                coeffs: coeffs.len(),
                boundaries: boundaries.len(),
            });
        }
        if bit_terms.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                bit_terms: bit_terms.len(),
                indices: indices.len(),
            });
        }
        // We already checked that `boundaries` is at least length 1.
        if *boundaries.first().unwrap() != 0 {
            return Err(CoherenceError::BadInitialBoundary(boundaries[0]));
        }
        if *boundaries.last().unwrap() != indices.len() {
            return Err(CoherenceError::BadFinalBoundary {
                last: *boundaries.last().unwrap(),
                items: indices.len(),
            });
        }
        for (&left, &right) in boundaries[..].iter().zip(&boundaries[1..]) {
            if right < left {
                return Err(CoherenceError::DecreasingBoundaries);
            }
            let indices = &indices[left..right];
            if !indices.is_empty() {
                for (index_left, index_right) in indices[..].iter().zip(&indices[1..]) {
                    if index_left == index_right {
                        return Err(CoherenceError::DuplicateIndices);
                    } else if index_left > index_right {
                        return Err(CoherenceError::UnsortedIndices);
                    }
                }
            }
            if indices.last().map(|&ix| ix >= num_qubits).unwrap_or(false) {
                return Err(CoherenceError::BitIndexTooHigh);
            }
        }
        // SAFETY: we've just done the coherence checks.
        Ok(unsafe { Self::new_unchecked(num_qubits, coeffs, bit_terms, indices, boundaries) })
    }

    /// Create a new observable from the raw components without checking data coherence.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the data-coherence requirements, as enumerated in the
    /// struct-level documentation, have been upheld.
    #[inline(always)]
    pub unsafe fn new_unchecked(
        num_qubits: u32,
        coeffs: Vec<Complex64>,
        bit_terms: Vec<BitTerm>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Self {
        Self {
            num_qubits,
            coeffs,
            bit_terms,
            indices,
            boundaries,
        }
    }

    /// Create a zero operator with pre-allocated space for the given number of summands and
    /// single-qubit bit terms.
    #[inline]
    pub fn with_capacity(num_qubits: u32, num_terms: usize, num_bit_terms: usize) -> Self {
        Self {
            num_qubits,
            coeffs: Vec::with_capacity(num_terms),
            bit_terms: Vec::with_capacity(num_bit_terms),
            indices: Vec::with_capacity(num_bit_terms),
            boundaries: {
                let mut boundaries = Vec::with_capacity(num_terms + 1);
                boundaries.push(0);
                boundaries
            },
        }
    }

    /// Get an iterator over the individual terms of the operator.
    ///
    /// Recall that two [SparseObservable]s that have different term orders can still represent the
    /// same object.  Use [canonicalize] to apply a canonical ordering to the terms.
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = SparseTermView<'_>> + '_ {
        self.coeffs.iter().enumerate().map(|(i, coeff)| {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            SparseTermView {
                num_qubits: self.num_qubits,
                coeff: *coeff,
                bit_terms: &self.bit_terms[start..end],
                indices: &self.indices[start..end],
            }
        })
    }

    /// Get an iterator over the individual terms of the operator that allows in-place mutation.
    ///
    /// The length and indices of these views cannot be mutated, since both would allow breaking
    /// data coherence.
    pub fn iter_mut(&mut self) -> IterMut<'_> {
        self.into()
    }

    /// Get the number of qubits the observable is defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Get the number of terms in the observable.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.coeffs.len()
    }

    /// Get the coefficients of the terms.
    #[inline]
    pub fn coeffs(&self) -> &[Complex64] {
        &self.coeffs
    }

    /// Get a mutable slice of the coefficients.
    #[inline]
    pub fn coeffs_mut(&mut self) -> &mut [Complex64] {
        &mut self.coeffs
    }

    /// Get the indices of each [BitTerm].
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Get a mutable slice of the indices.
    ///
    /// # Safety
    ///
    /// Modifying the indices can cause an incoherent state of the [SparseObservable].
    /// It should be ensured that the indices are consistent with the coeffs, bit_terms, and
    /// boundaries.
    #[inline]
    pub unsafe fn indices_mut(&mut self) -> &mut [u32] {
        &mut self.indices
    }

    /// Get the boundaries of each term.
    #[inline]
    pub fn boundaries(&self) -> &[usize] {
        &self.boundaries
    }

    /// Get a mutable slice of the boundaries.
    ///
    /// # Safety
    ///
    /// Modifying the boundaries can cause an incoherent state of the [SparseObservable].
    /// It should be ensured that the boundaries are sorted and the length/elements are consistent
    /// with the coeffs, bit_terms, and indices.
    #[inline]
    pub unsafe fn boundaries_mut(&mut self) -> &mut [usize] {
        &mut self.boundaries
    }

    /// Get the [BitTerm]s in the observable.
    #[inline]
    pub fn bit_terms(&self) -> &[BitTerm] {
        &self.bit_terms
    }

    /// Get a muitable slice of the bit terms.
    #[inline]
    pub fn bit_terms_mut(&mut self) -> &mut [BitTerm] {
        &mut self.bit_terms
    }

    /// Create a zero operator on ``num_qubits`` qubits.
    pub fn zero(num_qubits: u32) -> Self {
        Self::with_capacity(num_qubits, 0, 0)
    }

    /// Create an identity operator on ``num_qubits`` qubits.
    pub fn identity(num_qubits: u32) -> Self {
        Self {
            num_qubits,
            coeffs: vec![Complex64::new(1.0, 0.0)],
            bit_terms: vec![],
            indices: vec![],
            boundaries: vec![0, 0],
        }
    }

    /// Clear all the terms from this operator, making it equal to the zero operator again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction operations may not need to reallocate.
    pub fn clear(&mut self) {
        self.coeffs.clear();
        self.bit_terms.clear();
        self.indices.clear();
        self.boundaries.truncate(1);
    }

    /// Reduce the observable to its canonical form.
    ///
    /// This sums like terms, removing them if the final complex coefficient's absolute value is
    /// less than or equal to the tolerance.  The terms are reordered to some canonical ordering.
    ///
    /// This function is idempotent.
    pub fn canonicalize(&self, tol: f64) -> SparseObservable {
        let mut terms = btree_map::BTreeMap::new();
        for term in self.iter() {
            terms
                .entry((term.indices, term.bit_terms))
                .and_modify(|c| *c += term.coeff)
                .or_insert(term.coeff);
        }
        let mut out = SparseObservable::zero(self.num_qubits);
        for ((indices, bit_terms), coeff) in terms {
            if coeff.norm_sqr() <= tol * tol {
                continue;
            }
            out.coeffs.push(coeff);
            out.bit_terms.extend_from_slice(bit_terms);
            out.indices.extend_from_slice(indices);
            out.boundaries.push(out.indices.len());
        }
        out
    }

    /// Tensor product of `self` with `other`.
    ///
    /// The bit ordering is defined such that the qubit indices of `other` will remain the same, and
    /// the indices of `self` will be offset by the number of qubits in `other`.  This is the same
    /// convention as used by the rest of Qiskit's `quantum_info` operators.
    ///
    /// Put another way, in the simplest case of two observables formed of dense labels:
    ///
    /// ```
    /// let mut left = SparseObservable::zero(5);
    /// left.add_dense_label("IXY+Z", Complex64::new(1.0, 0.0));
    /// let mut right = SparseObservable::zero(6);
    /// right.add_dense_label("IIrl01", Complex64::new(1.0, 0.0));
    ///
    /// // The result is the concatenation of the two labels.
    /// let mut out = SparseObservable::zero(11);
    /// out.add_dense_label("IXY+ZIIrl01", Complex64::new(1.0, 0.0));
    ///
    /// assert_eq!(left.tensor(right), out);
    /// ```
    pub fn tensor(&self, other: &SparseObservable) -> SparseObservable {
        let mut out = SparseObservable::with_capacity(
            self.num_qubits + other.num_qubits,
            self.coeffs.len() * other.coeffs.len(),
            other.coeffs.len() * self.bit_terms.len() + self.coeffs.len() * other.bit_terms.len(),
        );
        let mut self_indices = Vec::new();
        for self_term in self.iter() {
            self_indices.clear();
            self_indices.extend(self_term.indices.iter().map(|i| i + other.num_qubits));
            for other_term in other.iter() {
                out.coeffs.push(self_term.coeff * other_term.coeff);
                out.indices.extend_from_slice(other_term.indices);
                out.indices.extend_from_slice(&self_indices);
                out.bit_terms.extend_from_slice(other_term.bit_terms);
                out.bit_terms.extend_from_slice(self_term.bit_terms);
                out.boundaries.push(out.bit_terms.len());
            }
        }
        out
    }

    /// Calculate the adjoint of this observable.
    ///
    /// This is well defined in the abstract mathematical sense.  All the terms of the single-qubit
    /// alphabet are self-adjoint, so the result of this operation is the same observable, except
    /// its coefficients are all their complex conjugates.
    pub fn adjoint(&self) -> SparseObservable {
        SparseObservable {
            num_qubits: self.num_qubits,
            coeffs: self.coeffs.iter().map(|c| c.conj()).collect(),
            bit_terms: self.bit_terms.clone(),
            indices: self.indices.clone(),
            boundaries: self.boundaries.clone(),
        }
    }

    /// Calculate the transpose.
    ///
    /// This operation transposes the individual bit terms but does not directly act
    /// on the coefficients.
    pub fn transpose(&self) -> SparseObservable {
        let mut out = self.clone();
        for term in out.iter_mut() {
            for bit_term in term.bit_terms {
                match bit_term {
                    BitTerm::Y => {
                        *term.coeff = -*term.coeff;
                    }
                    BitTerm::Right => {
                        *bit_term = BitTerm::Left;
                    }
                    BitTerm::Left => {
                        *bit_term = BitTerm::Right;
                    }
                    _ => (),
                }
            }
        }
        out
    }

    /// Calculate the complex conjugate.
    ///
    /// This operation equals transposing the observable and complex conjugating the coefficients.
    pub fn conjugate(&self) -> SparseObservable {
        let mut out = self.transpose();
        for coeff in out.coeffs.iter_mut() {
            *coeff = coeff.conj();
        }
        out
    }

    /// Compose another [SparseObservable] onto this one.
    ///
    /// In terms of operator algebras, composition corresponds to left-multiplication:
    /// ``let c = a.compose(b);`` corresponds to $C = B A$.
    ///
    /// Beware that this function can cause exponential explosion of the memory usage of the
    /// observable, as the alphabet of [SparseObservable] is not closed under composition; the
    /// composition of two single-bit terms can be a sum, which multiplies the total number of
    /// terms.  This memory usage is not _necessarily_ inherent to the resultant observable, but
    /// finding an efficient re-factorization of the sum is generally equally computationally hard.
    /// It's better to use domain knowledge of your observables to minimize the number of terms that
    /// ever exist, rather than trying to simplify them after the fact.
    ///
    /// # Panics
    ///
    /// If `self` and `other` have different numbers of qubits.
    pub fn compose(&self, other: &SparseObservable) -> SparseObservable {
        if other.num_qubits != self.num_qubits {
            panic!(
                "operand ({}) has a different number of qubits to the base ({})",
                other.num_qubits, self.num_qubits
            );
        }
        let mut out = SparseObservable::zero(self.num_qubits);
        let mut term_state = compose::Iter::new(self.num_qubits);

        for left in other.iter() {
            for right in self.iter() {
                term_state.load_from(
                    left.coeff * right.coeff,
                    left.indices
                        .iter()
                        .copied()
                        .zip(left.bit_terms.iter().copied()),
                    right
                        .indices
                        .iter()
                        .copied()
                        .zip(right.bit_terms.iter().copied()),
                );
                out.boundaries.reserve(term_state.num_terms());
                out.coeffs.reserve(term_state.num_terms());
                out.indices
                    .reserve(term_state.num_terms() * term_state.term_len());
                out.bit_terms
                    .reserve(term_state.num_terms() * term_state.term_len());
                while let Some(term) = term_state.next() {
                    out.add_term(term)
                        .expect("qubit counts were checked during initialisation");
                }
            }
        }
        out
    }

    /// Compose another [SparseObservable] onto this one, remapping the qubits.
    ///
    /// The `qubit_fn` is called for each qubit in each term in `other` to determine which qubit in
    /// `self` it corresponds to; this should typically be a very cheap function to call, like a
    /// getter from a slice.
    ///
    /// See [compose] for further information.
    ///
    /// # Panics
    ///
    /// If `other` has more qubits than `self`, if the `qubit_fn` returns a qubit index greater
    /// or equal to the number of qubits in `self`, or if `qubit_fn` returns a duplicate index (this
    /// is only guaranteed to be detected if an individual term contains duplicates).
    pub fn compose_map(
        &self,
        other: &SparseObservable,
        mut qubit_fn: impl FnMut(u32) -> u32,
    ) -> SparseObservable {
        if other.num_qubits > self.num_qubits {
            panic!(
                "operand has more qubits ({}) than the base ({})",
                other.num_qubits, self.num_qubits
            );
        }
        let mut out = SparseObservable::zero(self.num_qubits);
        let mut mapped_term = btree_map::BTreeMap::<u32, BitTerm>::new();
        let mut term_state = compose::Iter::new(self.num_qubits);

        // This choice of loop ordering is because we already know that `self`'s `indices` are
        // sorted, but there's no reason that that the output of `qubit_fn` should maintain order,
        // and this way round, we sort as few times as possible.
        for left in other.iter() {
            mapped_term.clear();
            for (index, term) in left.indices.iter().zip(left.bit_terms) {
                let qubit = qubit_fn(*index);
                if qubit >= self.num_qubits {
                    panic!("remapped {index} to {qubit}, which is out of bounds");
                }
                if mapped_term.insert(qubit, *term).is_some() {
                    panic!("duplicate qubit {qubit} in remapped term");
                };
            }
            for right in self.iter() {
                term_state.load_from(
                    left.coeff * right.coeff,
                    mapped_term.iter().map(|(k, v)| (*k, *v)),
                    right
                        .indices
                        .iter()
                        .copied()
                        .zip(right.bit_terms.iter().copied()),
                );
                out.boundaries.reserve(term_state.num_terms());
                out.coeffs.reserve(term_state.num_terms());
                out.indices
                    .reserve(term_state.num_terms() * term_state.term_len());
                out.bit_terms
                    .reserve(term_state.num_terms() * term_state.term_len());
                while let Some(term) = term_state.next() {
                    out.add_term(term)
                        .expect("qubit counts were checked during initialisation");
                }
            }
        }
        out
    }

    /// Evolve this [SparseObservable] by another one.
    ///
    /// In terms of operator algebra, evolution corresponds to conjugation:
    /// ``let out = q.evolve(p);`` corresponds to $P^\dagger Q P$.
    ///
    /// This implements Heisenberg-picture evolution of the observable.  Unlike a
    /// literal implementation via two full compositions, this method performs the
    /// conjugation directly at the single-qubit level using a fixed lookup table
    /// for $P^\dagger Q P$.  This avoids materializing any intermediate
    /// [SparseObservable] and computes the evolved observable in a single pass.
    ///
    /// Currently, this method supports evolution only by a *single-term* [SparseObservable].
    pub fn evolve(
        &self,
        op: &SparseObservable,
        qargs: Option<&[u32]>,
    ) -> Result<SparseObservable, ArithmeticError> {
        if op.num_terms() != 1 {
            return Err(ArithmeticError::InvalidOperation(
                "evolve only supports single-term operators".to_string(),
            ));
        }

        let t = op.iter().next().unwrap();
        let op_coeff = t.coeff;
        let mut layout = vec![None; self.num_qubits as usize];

        if let Some(qargs) = qargs {
            if op.num_qubits > self.num_qubits {
                return Err(ArithmeticError::OutOfBounds(format!(
                    "operator has more qubits ({}) than the base ({})",
                    op.num_qubits, self.num_qubits
                )));
            }
            // Handling the zero-qubit scalar edge case (Identity Operator evolution).
            if op.num_qubits == 0 {
                let scalar = op.coeffs()[0];
                return Ok(self * (scalar.conj() * scalar));
            }

            if qargs.len() != op.num_qubits as usize {
                return Err(ArithmeticError::OutOfBounds(format!(
                    "qargs has length {}, but operator has {} qubit(s)",
                    qargs.len(),
                    op.num_qubits
                )));
            }

            let qargs_set = HashSet::<&u32>::from_iter(qargs.iter());
            if qargs_set.len() != qargs.len() {
                return Err(ArithmeticError::DuplicatedIndex);
            }

            if let Some(&max_q) = qargs.iter().max() {
                if max_q >= self.num_qubits {
                    return Err(ArithmeticError::OutOfBounds(
                        "qargs contains out-of-range qubits".to_string(),
                    ));
                }
            }

            // This maps operator bit terms to observable qubits via qargs, considering
            // qargs[i] specifies which observable qubit (at index i), the next operator qubit
            // in consideration acts on.  Operator qubits are numbered 0 to (num_qubits - 1),
            // where qubit 0 is considered, the rightmost (least significant) qubit.
            for (op_qubit, &self_qubit) in qargs.iter().enumerate() {
                if let Some(bit_term_idx) = t.indices.iter().position(|&q| q as usize == op_qubit) {
                    layout[self_qubit as usize] = Some(t.bit_terms[bit_term_idx]);
                }
            }
        } else {
            if self.num_qubits != op.num_qubits {
                return Err(ArithmeticError::MismatchedQubits {
                    left: self.num_qubits,
                    right: op.num_qubits,
                });
            }

            for (q, bt) in t.indices.iter().zip(t.bit_terms.iter()) {
                layout[*q as usize] = Some(*bt);
            }
        }

        let mut out = SparseObservable::zero(self.num_qubits);

        for term in self.iter() {
            let mut frontier = vec![(term.coeff, Vec::<u32>::new(), Vec::<BitTerm>::new())];
            let mut term_map = vec![None; self.num_qubits as usize];
            for (i, &q) in term.indices.iter().enumerate() {
                term_map[q as usize] = Some(term.bit_terms[i]);
            }

            for (q, &op_bt) in layout.iter().enumerate() {
                let term_bt = term_map[q];

                let mut next_frontier = Vec::new();

                for (coeff, indices, bit_terms) in frontier {
                    match (op_bt, term_bt) {
                        (None, None) => {
                            next_frontier.push((coeff, indices, bit_terms));
                        }
                        (None, Some(bt)) => {
                            let mut indices = indices;
                            let mut bit_terms = bit_terms;
                            indices.push(q as u32);
                            bit_terms.push(bt);
                            next_frontier.push((coeff, indices, bit_terms));
                        }
                        (Some(_), None) => {
                            next_frontier.push((coeff, indices, bit_terms));
                        }
                        (Some(p), Some(qbt)) => {
                            let outputs = conjugate_bitterm(p, qbt);
                            for &(c, new_bt) in outputs {
                                let mut new_indices = indices.clone();
                                let mut new_bit_terms = bit_terms.clone();
                                new_indices.push(q as u32);
                                new_bit_terms.push(new_bt);
                                next_frontier.push((coeff * c, new_indices, new_bit_terms));
                            }
                        }
                    }
                }

                frontier = next_frontier;
                if frontier.is_empty() {
                    break;
                }
            }

            for (coeff, indices, bit_terms) in frontier {
                if coeff == Complex64::new(0.0, 0.0) {
                    continue;
                }
                out.coeffs.push(op_coeff.conj() * coeff * op_coeff);
                out.indices.extend(indices);
                out.bit_terms.extend(bit_terms);
                out.boundaries.push(out.indices.len());
            }
        }

        Ok(out)
    }

    /// Add another [SparseObservable] onto this one, while scaling its coefficients.
    ///
    /// # Panics
    ///
    /// If the number of qubits of `rhs` and `self` differ.
    pub fn scaled_add_inplace(&mut self, rhs: &SparseObservable, factor: Complex64) {
        if rhs.num_qubits != self.num_qubits {
            panic!(
                "operand ({}) has a different number of qubits to the base ({})",
                rhs.num_qubits, self.num_qubits
            );
        }
        self.coeffs.extend(rhs.coeffs.iter().map(|c| c * factor));
        self.bit_terms.extend_from_slice(&rhs.bit_terms);
        self.indices.extend_from_slice(&rhs.indices);
        // We only need to write out the new endpoints, not the initial zero.
        let offset = self.boundaries[self.boundaries.len() - 1];
        self.boundaries
            .extend(rhs.boundaries[1..].iter().map(|boundary| offset + boundary));
    }

    /// Add two [SparseObservable] instances while scaling the coefficients of `rhs`
    /// with `factor`.
    ///
    /// # Panics
    ///
    /// If the number of qubits of `other` and `self` differ.
    pub fn scaled_add(&self, rhs: &SparseObservable, factor: Complex64) -> SparseObservable {
        let mut out = SparseObservable::with_capacity(
            self.num_qubits,
            self.coeffs.len() + rhs.coeffs.len(),
            self.bit_terms.len() + rhs.bit_terms.len(),
        );
        out += self;
        out.scaled_add_inplace(rhs, factor);
        out
    }

    /// Get a view onto a representation of a single sparse term.
    ///
    /// This is effectively an indexing operation into the [SparseObservable].  Recall that two
    /// [SparseObservable]s that have different term orders can still represent the same object.
    /// Use [canonicalize] to apply a canonical ordering to the terms.
    ///
    /// # Panics
    ///
    /// If the index is out of bounds.
    pub fn term(&self, index: usize) -> SparseTermView<'_> {
        debug_assert!(index < self.num_terms(), "index {index} out of bounds");
        let start = self.boundaries[index];
        let end = self.boundaries[index + 1];
        SparseTermView {
            num_qubits: self.num_qubits,
            coeff: self.coeffs[index],
            bit_terms: &self.bit_terms[start..end],
            indices: &self.indices[start..end],
        }
    }

    /// Expand all projectors into Pauli representation.
    ///
    /// # Warning
    ///
    /// This representation is highly inefficient for projectors. For example, a term with
    /// :math:`n` projectors :math:`|+\rangle\langle +|` will use :math:`2^n` Pauli terms.
    pub fn as_paulis(&self) -> Self {
        let mut paulis: Vec<BitTerm> = Vec::new(); // maybe get capacity here
        let mut indices: Vec<u32> = Vec::new();
        let mut coeffs: Vec<Complex64> = Vec::new();
        let mut boundaries: Vec<usize> = vec![0];

        for view in self.iter() {
            let num_projectors = view
                .bit_terms
                .iter()
                .filter(|&bit| bit.is_projector())
                .count();
            let div = 2_f64.powi(num_projectors as i32);

            let combinations = view
                .bit_terms
                .iter()
                .map(bit_term_as_pauli)
                .multi_cartesian_product();

            for combination in combinations {
                let mut positive = true; // keep track of the global sign

                for (index, (sign, bit)) in combination.iter().enumerate() {
                    positive ^= !sign; // accumulate the sign; global_sign *= local_sign
                    if let Some(bit) = bit {
                        paulis.push(*bit);
                        indices.push(view.indices[index]);
                    }
                }
                boundaries.push(paulis.len());

                let coeff = if positive { view.coeff } else { -view.coeff };
                coeffs.push(coeff / div)
            }
        }

        Self {
            num_qubits: self.num_qubits,
            coeffs,
            bit_terms: paulis,
            indices,
            boundaries,
        }
    }

    /// Add the term implied by a dense string label onto this observable.
    pub fn add_dense_label<L: AsRef<[u8]>>(
        &mut self,
        label: L,
        coeff: Complex64,
    ) -> Result<(), LabelError> {
        let label = label.as_ref();
        if label.len() != self.num_qubits() as usize {
            return Err(LabelError::WrongLengthDense {
                num_qubits: self.num_qubits(),
                label: label.len(),
            });
        }
        // The only valid characters in the alphabet are ASCII, so if we see something other than
        // ASCII, we're already in the failure path.
        for (i, letter) in label.iter().rev().enumerate() {
            match BitTerm::try_from_u8(*letter) {
                Ok(Some(term)) => {
                    self.bit_terms.push(term);
                    self.indices.push(i as u32);
                }
                Ok(None) => (),
                Err(_) => {
                    // Undo any modifications to ourselves so we stay in a consistent state.
                    let num_single_terms = self.boundaries[self.boundaries.len() - 1];
                    self.bit_terms.truncate(num_single_terms);
                    self.indices.truncate(num_single_terms);
                    return Err(LabelError::OutsideAlphabet);
                }
            }
        }
        self.coeffs.push(coeff);
        self.boundaries.push(self.bit_terms.len());
        Ok(())
    }

    /// Relabel the `indices` in the operator to new values.
    ///
    /// This fails if any of the new indices are too large, or if any mapping would cause a term to
    /// contain duplicates of the same index.  It may not detect if multiple qubits are mapped to
    /// the same index, if those qubits never appear together in the same term.  Such a mapping
    /// would not cause data-coherence problems (the output observable will be valid), but is
    /// unlikely to be what you intended.
    ///
    /// *Panics* if `new_qubits` is not long enough to map every index used in the operator.
    pub fn relabel_qubits_from_slice(&mut self, new_qubits: &[u32]) -> Result<(), CoherenceError> {
        for qubit in new_qubits {
            if *qubit >= self.num_qubits {
                return Err(CoherenceError::BitIndexTooHigh);
            }
        }
        let mut order = btree_map::BTreeMap::new();
        for i in 0..self.num_terms() {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            for j in start..end {
                order.insert(new_qubits[self.indices[j] as usize], self.bit_terms[j]);
            }
            if order.len() != end - start {
                return Err(CoherenceError::DuplicateIndices);
            }
            for (index, dest) in order.keys().zip(&mut self.indices[start..end]) {
                *dest = *index;
            }
            for (bit_term, dest) in order.values().zip(&mut self.bit_terms[start..end]) {
                *dest = *bit_term;
            }
            order.clear();
        }
        Ok(())
    }

    /// Apply a transpiler layout.
    pub fn apply_layout(
        &self,
        layout: Option<&[u32]>,
        num_qubits: u32,
    ) -> Result<Self, CoherenceError> {
        match layout {
            None => {
                let mut out = self.clone();
                if num_qubits < self.num_qubits {
                    return Err(CoherenceError::NotEnoughQubits {
                        current: self.num_qubits as usize,
                        target: num_qubits as usize,
                    });
                }
                out.num_qubits = num_qubits;
                Ok(out)
            }
            Some(layout) => {
                if layout.len() < self.num_qubits as usize {
                    return Err(CoherenceError::IndexMapTooSmall);
                }
                if layout.iter().any(|qubit| *qubit >= num_qubits) {
                    return Err(CoherenceError::BitIndexTooHigh);
                }
                if layout.iter().collect::<HashSet<_>>().len() != layout.len() {
                    return Err(CoherenceError::DuplicateIndices);
                }
                let mut out = self.clone();
                out.num_qubits = num_qubits;
                out.relabel_qubits_from_slice(layout)?;
                Ok(out)
            }
        }
    }

    /// Add a single term to this operator.
    pub fn add_term(&mut self, term: SparseTermView) -> Result<(), ArithmeticError> {
        if self.num_qubits != term.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: term.num_qubits,
            });
        }
        self.coeffs.push(term.coeff);
        self.bit_terms.extend_from_slice(term.bit_terms);
        self.indices.extend_from_slice(term.indices);
        self.boundaries.push(self.bit_terms.len());
        Ok(())
    }

    /// Return a suitable Python error if two observables do not have equal numbers of qubits.
    pub fn check_equal_qubits(&self, other: &SparseObservable) -> Result<(), ArithmeticError> {
        if self.num_qubits != other.num_qubits {
            Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: other.num_qubits,
            })
        } else {
            Ok(())
        }
    }

    /// Check whether the observable commutes with another one.
    ///
    /// # Arguments
    ///
    /// * `tol` - If coefficients in the product of `self` and `other` are below the tolerance
    ///   (in magnitude), the terms are ignored.
    ///
    /// # Returns
    ///
    /// `true` if the observables commute, `false` otherwise.
    pub fn commutes(&self, other: &SparseObservable, tol: f64) -> bool {
        let ab = self.compose(other).canonicalize(tol);
        let ba = other.compose(self).canonicalize(tol);
        ab == ba
    }
}

impl ::std::ops::Add<&SparseObservable> for SparseObservable {
    type Output = SparseObservable;

    fn add(mut self, rhs: &SparseObservable) -> SparseObservable {
        self += rhs;
        self
    }
}
impl ::std::ops::Add for &SparseObservable {
    type Output = SparseObservable;

    fn add(self, rhs: &SparseObservable) -> SparseObservable {
        let mut out = SparseObservable::with_capacity(
            self.num_qubits,
            self.coeffs.len() + rhs.coeffs.len(),
            self.bit_terms.len() + rhs.bit_terms.len(),
        );
        out += self;
        out += rhs;
        out
    }
}
impl ::std::ops::AddAssign<&SparseObservable> for SparseObservable {
    fn add_assign(&mut self, rhs: &SparseObservable) {
        if self.num_qubits != rhs.num_qubits {
            panic!("attempt to add two operators with incompatible qubit counts");
        }
        self.coeffs.extend_from_slice(&rhs.coeffs);
        self.bit_terms.extend_from_slice(&rhs.bit_terms);
        self.indices.extend_from_slice(&rhs.indices);
        // We only need to write out the new endpoints, not the initial zero.
        let offset = self.boundaries[self.boundaries.len() - 1];
        self.boundaries
            .extend(rhs.boundaries[1..].iter().map(|boundary| offset + boundary));
    }
}

impl ::std::ops::Sub<&SparseObservable> for SparseObservable {
    type Output = SparseObservable;

    fn sub(mut self, rhs: &SparseObservable) -> SparseObservable {
        self -= rhs;
        self
    }
}
impl ::std::ops::Sub for &SparseObservable {
    type Output = SparseObservable;

    fn sub(self, rhs: &SparseObservable) -> SparseObservable {
        let mut out = SparseObservable::with_capacity(
            self.num_qubits,
            self.coeffs.len() + rhs.coeffs.len(),
            self.bit_terms.len() + rhs.bit_terms.len(),
        );
        out += self;
        out -= rhs;
        out
    }
}
impl ::std::ops::SubAssign<&SparseObservable> for SparseObservable {
    fn sub_assign(&mut self, rhs: &SparseObservable) {
        if self.num_qubits != rhs.num_qubits {
            panic!("attempt to subtract two operators with incompatible qubit counts");
        }
        self.coeffs.extend(rhs.coeffs.iter().map(|coeff| -coeff));
        self.bit_terms.extend_from_slice(&rhs.bit_terms);
        self.indices.extend_from_slice(&rhs.indices);
        // We only need to write out the new endpoints, not the initial zero.
        let offset = self.boundaries[self.boundaries.len() - 1];
        self.boundaries
            .extend(rhs.boundaries[1..].iter().map(|boundary| offset + boundary));
    }
}

impl ::std::ops::Mul<Complex64> for SparseObservable {
    type Output = SparseObservable;

    fn mul(mut self, rhs: Complex64) -> SparseObservable {
        self *= rhs;
        self
    }
}
impl ::std::ops::Mul<Complex64> for &SparseObservable {
    type Output = SparseObservable;

    fn mul(self, rhs: Complex64) -> SparseObservable {
        if rhs == Complex64::new(0.0, 0.0) {
            SparseObservable::zero(self.num_qubits)
        } else {
            SparseObservable {
                num_qubits: self.num_qubits,
                coeffs: self.coeffs.iter().map(|c| c * rhs).collect(),
                bit_terms: self.bit_terms.clone(),
                indices: self.indices.clone(),
                boundaries: self.boundaries.clone(),
            }
        }
    }
}
impl ::std::ops::Mul<SparseObservable> for Complex64 {
    type Output = SparseObservable;

    fn mul(self, mut rhs: SparseObservable) -> SparseObservable {
        rhs *= self;
        rhs
    }
}
impl ::std::ops::Mul<&SparseObservable> for Complex64 {
    type Output = SparseObservable;

    fn mul(self, rhs: &SparseObservable) -> SparseObservable {
        rhs * self
    }
}
impl ::std::ops::MulAssign<Complex64> for SparseObservable {
    fn mul_assign(&mut self, rhs: Complex64) {
        if rhs == Complex64::new(0.0, 0.0) {
            self.coeffs.clear();
            self.bit_terms.clear();
            self.indices.clear();
            self.boundaries.clear();
            self.boundaries.push(0);
        } else {
            self.coeffs.iter_mut().for_each(|c| *c *= rhs)
        }
    }
}

impl ::std::ops::Div<Complex64> for SparseObservable {
    type Output = SparseObservable;

    fn div(mut self, rhs: Complex64) -> SparseObservable {
        self /= rhs;
        self
    }
}
impl ::std::ops::Div<Complex64> for &SparseObservable {
    type Output = SparseObservable;

    fn div(self, rhs: Complex64) -> SparseObservable {
        SparseObservable {
            num_qubits: self.num_qubits,
            coeffs: self.coeffs.iter().map(|c| c / rhs).collect(),
            bit_terms: self.bit_terms.clone(),
            indices: self.indices.clone(),
            boundaries: self.boundaries.clone(),
        }
    }
}
impl ::std::ops::DivAssign<Complex64> for SparseObservable {
    fn div_assign(&mut self, rhs: Complex64) {
        self.coeffs.iter_mut().for_each(|c| *c /= rhs)
    }
}

impl ::std::ops::Neg for &SparseObservable {
    type Output = SparseObservable;

    fn neg(self) -> SparseObservable {
        SparseObservable {
            num_qubits: self.num_qubits,
            coeffs: self.coeffs.iter().map(|c| -c).collect(),
            bit_terms: self.bit_terms.clone(),
            indices: self.indices.clone(),
            boundaries: self.boundaries.clone(),
        }
    }
}
impl ::std::ops::Neg for SparseObservable {
    type Output = SparseObservable;

    fn neg(mut self) -> SparseObservable {
        self.coeffs.iter_mut().for_each(|c| *c = -*c);
        self
    }
}

/// Worker objects for the [SparseObservable::compose]-alike functions.
mod compose {
    use super::*;

    /// An non-scalar entry in the multi-Cartesian product iteration.
    ///
    /// Each [MultipleItem] corresponds to a bit index that becomes a summation as part of the
    /// composition, so the multi-Cartesian product has to keep iterating through it.
    struct MultipleItem {
        /// The index into the full `bit_terms` [Vec] that this item refers to.
        loc: usize,
        /// The next item in the slice that should be written out.  If this equal to the length
        /// of the slice, the implication is that it needs to be reset to 0 by a lower-index
        /// item getting incremented and flowing forwards.
        cur: usize,
        /// The underlying slice of the multiple iteration, from the compose lookup table.
        slice: &'static [(Complex64, BitTerm)],
    }

    /// An implementation of the multiple Cartesian-product iterator that produces the sum terms
    /// that make up the composition of two individual [SparseObservable] terms.
    ///
    /// This mutates itself in-place to keep track of the state, which allows us to re-use
    /// partial results and to avoid allocations per item (since we have to copy out of the
    /// buffers to the output [SparseObservable] each time anyway).
    pub struct Iter {
        num_qubits: u32,
        /// Tracking data for the places in the output that have multiple branches to take.
        multiples: Vec<MultipleItem>,
        /// Stack of the coefficients to this point.  We could recalculate by a full
        /// multiplication on each go, but most steps will be in the low indices, where we can
        /// re-use all the multiplications that came before.  This is one longer than the length
        /// of `multiples`, because it starts off populated with the product of the
        /// non-multiple coefficients (or 1).
        coeffs: Vec<Complex64>,
        /// The full set of indices (including ones that don't correspond to multiples).  Within
        /// a given iteration, this never changes; it's just stored to make it easier to memcpy
        /// out of.
        indices: Vec<u32>,
        /// The full set of [BitTerm]s (including ones that don't correspond to multiples).  The
        /// multiple ones get updated inplace during iteration.
        bit_terms: Vec<BitTerm>,
        /// Whether iteration has started.
        started: bool,
        /// Whether iteration will yield any further items.
        exhausted: bool,
    }
    impl Iter {
        pub fn new(num_qubits: u32) -> Self {
            Self {
                num_qubits,
                multiples: Vec::new(),
                coeffs: vec![Complex64::new(1., 0.)],
                indices: Vec::new(),
                bit_terms: Vec::new(),
                started: false,
                exhausted: false,
            }
        }
        /// Load up the next terms an iterator over the `(index, term)` pairs from the left-hand
        /// side and the right-hand side.  The iterators must return indices in strictly increasing
        /// order.
        pub fn load_from(
            &mut self,
            coeff: Complex64,
            left: impl IntoIterator<Item = (u32, BitTerm)>,
            right: impl IntoIterator<Item = (u32, BitTerm)>,
        ) {
            self.multiples.clear();
            self.coeffs.clear();
            self.coeffs.push(coeff);
            self.indices.clear();
            self.bit_terms.clear();
            self.started = false;
            self.exhausted = false;

            for (index, values) in pairwise_ordered(left, right) {
                match values {
                    Paired::Left(term) | Paired::Right(term) => {
                        self.indices.push(index);
                        self.bit_terms.push(term);
                    }
                    Paired::Both(left, right) => {
                        let Some(slice) = lookup::matmul(left, right) else {
                            self.exhausted = true;
                            return;
                        };
                        match slice {
                            &[] => (),
                            &[(coeff, term)] => {
                                self.coeffs[0] *= coeff;
                                self.indices.push(index);
                                self.bit_terms.push(term);
                            }
                            slice => {
                                self.multiples.push(MultipleItem {
                                    loc: self.bit_terms.len(),
                                    cur: 0,
                                    slice,
                                });
                                self.indices.push(index);
                                self.coeffs.push(Default::default());
                                self.bit_terms.push(slice[0].1);
                            }
                        }
                    }
                }
            }
        }
        /// Expose the current iteration item, assuming the state has been updated.
        fn iter_item(&self) -> SparseTermView<'_> {
            SparseTermView {
                num_qubits: self.num_qubits,
                coeff: *self.coeffs.last().expect("coeffs is never empty"),
                indices: &self.indices,
                bit_terms: &self.bit_terms,
            }
        }
        // Not actually the iterator method, because we're borrowing from `self`.
        /// Get the next term in the iteration.
        pub fn next(&mut self) -> Option<SparseTermView<'_>> {
            if self.exhausted {
                return None;
            }
            if !self.started {
                self.started = true;
                // Initialising the struct has to put us in a place where the indices and bit terms
                // are ready, but we still need to initialise the coefficient stack.
                for (i, item) in self.multiples.iter().enumerate() {
                    // `item.cur` should always be 0 at this point.
                    self.coeffs[i + 1] = self.coeffs[i] * item.slice[item.cur].0;
                }
                return Some(self.iter_item());
            }
            // The lowest index into `multiples` that didn't overflow as we were updating the
            // bit-terms state.
            let non_overflow_index = 'inc_index: {
                for (i, item) in self.multiples.iter_mut().enumerate().rev() {
                    // The slices are always non-empty.
                    if item.cur == item.slice.len() - 1 {
                        item.cur = 0;
                    } else {
                        item.cur += 1;
                    }
                    self.bit_terms[item.loc] = item.slice[item.cur].1;
                    if item.cur > 0 {
                        break 'inc_index i;
                    }
                }
                self.exhausted = true;
                return None;
            };
            // Now run forwards updating the coefficient tree from the point it changes.
            for (offset, item) in self.multiples[non_overflow_index..].iter_mut().enumerate() {
                let base = non_overflow_index + offset;
                self.coeffs[base + 1] = self.coeffs[base] * item.slice[item.cur].0;
            }
            Some(self.iter_item())
        }
        /// The total number of items in the iteration (this ignores the iteration state; only
        /// [load_from] changes it).  If a 0 multiplier is encountered during the load, the
        /// iterator is empty.
        pub fn num_terms(&self) -> usize {
            if self.exhausted {
                0
            } else {
                self.multiples.iter().map(|item| item.slice.len()).product()
            }
        }
        /// The length of each individual term in the iteration.
        pub fn term_len(&self) -> usize {
            self.bit_terms.len()
        }
    }
}

/// A view object onto a single term of a `SparseObservable`.
///
/// The lengths of `bit_terms` and `indices` are guaranteed to be created equal, but might be zero
/// (in the case that the term is proportional to the identity).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SparseTermView<'a> {
    pub num_qubits: u32,
    pub coeff: Complex64,
    pub bit_terms: &'a [BitTerm],
    pub indices: &'a [u32],
}
impl SparseTermView<'_> {
    /// Convert this `SparseTermView` into an owning [SparseTerm] of the same data.
    pub fn to_term(&self) -> SparseTerm {
        SparseTerm {
            num_qubits: self.num_qubits,
            coeff: self.coeff,
            bit_terms: self.bit_terms.into(),
            indices: self.indices.into(),
        }
    }

    pub fn to_sparse_str(self) -> String {
        let coeff = format!("{}", self.coeff).replace('i', "j");
        let paulis = self
            .indices
            .iter()
            .zip(self.bit_terms)
            .rev()
            .map(|(i, op)| format!("{}_{}", op.py_label(), i))
            .collect::<Vec<String>>()
            .join(" ");
        format!("({coeff})({paulis})")
    }
}

/// A mutable view object onto a single term of a [SparseObservable].
///
/// The lengths of [bit_terms] and [indices] are guaranteed to be created equal, but might be zero
/// (in the case that the term is proportional to the identity).  [indices] is not mutable because
/// this would allow data coherence to be broken.
#[derive(Debug)]
pub struct SparseTermViewMut<'a> {
    pub num_qubits: u32,
    pub coeff: &'a mut Complex64,
    pub bit_terms: &'a mut [BitTerm],
    pub indices: &'a [u32],
}

/// Iterator type allowing in-place mutation of the [SparseObservable].
///
/// Created by [SparseObservable::iter_mut].
#[derive(Debug)]
pub struct IterMut<'a> {
    num_qubits: u32,
    coeffs: &'a mut [Complex64],
    bit_terms: &'a mut [BitTerm],
    indices: &'a [u32],
    boundaries: &'a [usize],
    i: usize,
}
impl<'a> From<&'a mut SparseObservable> for IterMut<'a> {
    fn from(value: &mut SparseObservable) -> IterMut<'_> {
        IterMut {
            num_qubits: value.num_qubits,
            coeffs: &mut value.coeffs,
            bit_terms: &mut value.bit_terms,
            indices: &value.indices,
            boundaries: &value.boundaries,
            i: 0,
        }
    }
}
impl<'a> Iterator for IterMut<'a> {
    type Item = SparseTermViewMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // The trick here is that the lifetime of the 'self' borrow is shorter than the lifetime of
        // the inner borrows.  We can't give out mutable references to our inner borrow, because
        // after the lifetime on 'self' expired, there'd be nothing preventing somebody using the
        // 'self' borrow to see _another_ mutable borrow of the inner data, which would be an
        // aliasing violation.  Instead, we keep splitting the inner views we took out so the
        // mutable references we return don't overlap with the ones we continue to hold.
        let coeffs = ::std::mem::take(&mut self.coeffs);
        let (coeff, other_coeffs) = coeffs.split_first_mut()?;
        self.coeffs = other_coeffs;

        let len = self.boundaries[self.i + 1] - self.boundaries[self.i];
        self.i += 1;

        let all_bit_terms = ::std::mem::take(&mut self.bit_terms);
        let all_indices = ::std::mem::take(&mut self.indices);
        let (bit_terms, rest_bit_terms) = all_bit_terms.split_at_mut(len);
        let (indices, rest_indices) = all_indices.split_at(len);
        self.bit_terms = rest_bit_terms;
        self.indices = rest_indices;

        Some(SparseTermViewMut {
            num_qubits: self.num_qubits,
            coeff,
            bit_terms,
            indices,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.coeffs.len(), Some(self.coeffs.len()))
    }
}
impl ExactSizeIterator for IterMut<'_> {}
impl ::std::iter::FusedIterator for IterMut<'_> {}

/// A single term from a complete :class:`SparseObservable`.
///
/// These are typically created by indexing into or iterating through a :class:`SparseObservable`.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseTerm {
    /// Number of qubits the entire term applies to.
    num_qubits: u32,
    /// The complex coefficient of the term.
    coeff: Complex64,
    bit_terms: Box<[BitTerm]>,
    indices: Box<[u32]>,
}
impl SparseTerm {
    pub fn new(
        num_qubits: u32,
        coeff: Complex64,
        bit_terms: Box<[BitTerm]>,
        indices: Box<[u32]>,
    ) -> Result<Self, CoherenceError> {
        if bit_terms.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                bit_terms: bit_terms.len(),
                indices: indices.len(),
            });
        }

        if indices.iter().any(|index| *index >= num_qubits) {
            return Err(CoherenceError::BitIndexTooHigh);
        }

        Ok(Self {
            num_qubits,
            coeff,
            bit_terms,
            indices,
        })
    }

    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    pub fn coeff(&self) -> Complex64 {
        self.coeff
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn bit_terms(&self) -> &[BitTerm] {
        &self.bit_terms
    }

    pub fn view(&self) -> SparseTermView<'_> {
        SparseTermView {
            num_qubits: self.num_qubits,
            coeff: self.coeff,
            bit_terms: &self.bit_terms,
            indices: &self.indices,
        }
    }

    /// Convert this term to a complete :class:`SparseObservable`.
    pub fn to_observable(&self) -> SparseObservable {
        SparseObservable {
            num_qubits: self.num_qubits,
            coeffs: vec![self.coeff],
            bit_terms: self.bit_terms.to_vec(),
            indices: self.indices.to_vec(),
            boundaries: vec![0, self.bit_terms.len()],
        }
    }
}

#[derive(Error, Debug)]
pub struct InnerReadError;

#[cfg(feature = "python")] // Only currently used by python, remove if needed from rust
#[derive(Error, Debug)]
struct InnerWriteError;

impl ::std::fmt::Display for InnerReadError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "Failed acquiring lock for reading.")
    }
}

#[cfg(feature = "python")] // Only currently used by python, remove if needed from rust
impl ::std::fmt::Display for InnerWriteError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "Failed acquiring lock for writing.")
    }
}

#[cfg(feature = "python")]
impl From<InnerReadError> for PyErr {
    fn from(value: InnerReadError) -> PyErr {
        PyRuntimeError::new_err(value.to_string())
    }
}
#[cfg(feature = "python")]
impl From<InnerWriteError> for PyErr {
    fn from(value: InnerWriteError) -> PyErr {
        PyRuntimeError::new_err(value.to_string())
    }
}

#[cfg(feature = "python")]
impl From<BitTermFromU8Error> for PyErr {
    fn from(value: BitTermFromU8Error) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

#[cfg(feature = "python")]
impl From<CoherenceError> for PyErr {
    fn from(value: CoherenceError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

#[cfg(feature = "python")]
impl From<LabelError> for PyErr {
    fn from(value: LabelError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

#[cfg(feature = "python")]
impl From<ArithmeticError> for PyErr {
    fn from(value: ArithmeticError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

/// The single-character string label used to represent this term in the :class:`SparseObservable`
/// alphabet.

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "label")]
fn bit_term_label(py: Python<'_>, slf: BitTerm) -> &Bound<'_, PyString> {
    // This doesn't use `py_label` so we can use `intern!`.
    match slf {
        BitTerm::X => intern!(py, "X"),
        BitTerm::Plus => intern!(py, "+"),
        BitTerm::Minus => intern!(py, "-"),
        BitTerm::Y => intern!(py, "Y"),
        BitTerm::Right => intern!(py, "r"),
        BitTerm::Left => intern!(py, "l"),
        BitTerm::Z => intern!(py, "Z"),
        BitTerm::Zero => intern!(py, "0"),
        BitTerm::One => intern!(py, "1"),
    }
}
/// Construct the Python-space `IntEnum` that represents the same values as the Rust-spce `BitTerm`.
///
/// We don't make `BitTerm` a direct `pyclass` because we want the behaviour of `IntEnum`, which
/// specifically also makes its variants subclasses of the Python `int` type; we use a type-safe
/// enum in Rust, but from Python space we expect people to (carefully) deal with the raw ints in
/// Numpy arrays for efficiency.
///
/// The resulting class is attached to `SparseObservable` as a class attribute, and its
/// `__qualname__` is set to reflect this.
#[cfg(feature = "python")]
fn make_py_bit_term(py: Python) -> PyResult<Py<PyType>> {
    let terms = [
        BitTerm::X,
        BitTerm::Plus,
        BitTerm::Minus,
        BitTerm::Y,
        BitTerm::Right,
        BitTerm::Left,
        BitTerm::Z,
        BitTerm::Zero,
        BitTerm::One,
    ]
    .into_iter()
    .flat_map(|term| {
        let mut out = vec![(term.py_name(), term as u8)];
        if term.py_name() != term.py_label() {
            // Also ensure that the labels are created as aliases.  These can't be (easily) accessed
            // by attribute-getter (dot) syntax, but will work with the item-getter (square-bracket)
            // syntax, or programmatically with `getattr`.
            out.push((term.py_label(), term as u8));
        }
        out
    })
    .collect::<Vec<_>>();
    let obj = py.import("enum")?.getattr("IntEnum")?.call(
        ("BitTerm", terms),
        Some(
            &[
                ("module", "qiskit.quantum_info"),
                ("qualname", "SparseObservable.BitTerm"),
            ]
            .into_py_dict(py)?,
        ),
    )?;
    let label_property = py
        .import("builtins")?
        .getattr("property")?
        .call1((wrap_pyfunction!(bit_term_label, py)?,))?;
    obj.setattr("label", label_property)?;
    Ok(obj.cast_into::<PyType>()?.unbind())
}

// Return the relevant value from the Python-space sister enumeration.  These are Python-space
// singletons and subclasses of Python `int`.  We only use this for interaction with "high level"
// Python space; the efficient Numpy-like array paths use `u8` directly so Numpy can act on it
// efficiently.
#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for BitTerm {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let terms = BIT_TERM_INTO_PY.get_or_init(py, || {
            let py_enum = BIT_TERM_PY_ENUM
                .get_or_try_init(py, || make_py_bit_term(py))
                .expect("creating a simple Python enum class should be infallible")
                .bind(py);
            ::std::array::from_fn(|val| {
                ::bytemuck::checked::try_cast(val as u8)
                    .ok()
                    .map(|term: BitTerm| {
                        py_enum
                            .getattr(term.py_name())
                            .expect("the created `BitTerm` enum should have matching attribute names to the terms")
                            .unbind()
                    })
            })
        });
        Ok(terms[self as usize]
            .as_ref()
            .expect("the lookup table initializer populated a 'Some' in all valid locations")
            .bind(py)
            .clone())
    }
}

#[cfg(feature = "python")]
impl<'a, 'py> FromPyObject<'a, 'py> for BitTerm {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let value = ob
            .extract::<isize>()
            .map_err(|_| match ob.get_type().repr() {
                Ok(repr) => PyTypeError::new_err(format!("bad type for 'BitTerm': {repr}")),
                Err(err) => err,
            })?;
        let value_error = || {
            PyValueError::new_err(format!(
                "value {value} is not a valid letter of the single-qubit alphabet for 'BitTerm'"
            ))
        };
        let value: u8 = value.try_into().map_err(|_| value_error())?;
        value.try_into().map_err(|_| value_error())
    }
}

/// A single term from a complete :class:`SparseObservable`.
///
/// These are typically created by indexing into or iterating through a :class:`SparseObservable`.
#[cfg(feature = "python")]
#[pyclass(
    name = "Term",
    frozen,
    module = "qiskit.quantum_info",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
struct PySparseTerm {
    inner: SparseTerm,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySparseTerm {
    // Mark the Python class as being defined "within" the `SparseObservable` class namespace.
    #[classattr]
    #[pyo3(name = "__qualname__")]
    fn type_qualname() -> &'static str {
        "SparseObservable.Term"
    }

    #[new]
    #[pyo3(signature = (/, num_qubits, coeff, bit_terms, indices))]
    fn py_new(
        num_qubits: u32,
        coeff: Complex64,
        bit_terms: Vec<BitTerm>,
        indices: Vec<u32>,
    ) -> PyResult<Self> {
        if bit_terms.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                bit_terms: bit_terms.len(),
                indices: indices.len(),
            }
            .into());
        }
        let mut order = (0..bit_terms.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|a| indices[*a]);
        let bit_terms = order.iter().map(|i| bit_terms[*i]).collect();
        let mut sorted_indices = Vec::<u32>::with_capacity(order.len());
        for i in order {
            let index = indices[i];
            if sorted_indices
                .last()
                .map(|prev| *prev >= index)
                .unwrap_or(false)
            {
                return Err(CoherenceError::UnsortedIndices.into());
            }
            sorted_indices.push(index)
        }
        let inner = SparseTerm::new(
            num_qubits,
            coeff,
            bit_terms,
            sorted_indices.into_boxed_slice(),
        )?;
        Ok(PySparseTerm { inner })
    }

    /// Convert this term to a complete :class:`SparseObservable`.
    fn to_observable(&self) -> PyResult<PySparseObservable> {
        let obs = SparseObservable::new(
            self.inner.num_qubits(),
            vec![self.inner.coeff()],
            self.inner.bit_terms().to_vec(),
            self.inner.indices().to_vec(),
            vec![0, self.inner.bit_terms().len()],
        )?;
        Ok(obs.into())
    }

    fn to_label(&self) -> PyResult<String> {
        Ok(self.inner.view().to_sparse_str())
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> PyResult<bool> {
        if slf.is(&other) {
            return Ok(true);
        }
        let Ok(other) = other.cast_into::<Self>() else {
            return Ok(false);
        };
        let slf = slf.borrow();
        let other = other.borrow();
        Ok(slf.inner.eq(&other.inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "<{} on {} qubit{}: {}>",
            Self::type_qualname(),
            self.inner.num_qubits(),
            if self.inner.num_qubits() == 1 {
                ""
            } else {
                "s"
            },
            self.inner.view().to_sparse_str(),
        ))
    }

    fn __getnewargs__(slf_: Bound<Self>) -> PyResult<Bound<PyTuple>> {
        let py = slf_.py();
        let borrowed = slf_.borrow();
        (
            borrowed.inner.num_qubits(),
            borrowed.inner.coeff(),
            Self::get_bit_terms(slf_.clone()),
            Self::get_indices(slf_),
        )
            .into_pyobject(py)
    }

    /// Get a copy of this term.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Read-only view onto the individual single-qubit terms.
    ///
    /// The only valid values in the array are those with a corresponding
    /// :class:`~SparseObservable.BitTerm`.
    #[getter]
    fn get_bit_terms(slf_: Bound<Self>) -> Bound<PyArray1<u8>> {
        let borrowed = slf_.borrow();
        let bit_terms = borrowed.inner.bit_terms();
        let arr = ::ndarray::aview1(::bytemuck::cast_slice::<_, u8>(bit_terms));
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[BitTerm]>` allocation (including dropping or reallocating it) other than the entire
        // object getting dropped, which Python will keep safe.
        let out = unsafe { PyArray1::borrow_from_array(&arr, slf_.into_any()) };
        out.readwrite().make_nonwriteable();
        out
    }

    /// The number of qubits the term is defined on.
    #[getter]
    fn get_num_qubits(&self) -> u32 {
        self.inner.num_qubits()
    }

    /// The term's coefficient.
    #[getter]
    fn get_coeff(&self) -> Complex64 {
        self.inner.coeff()
    }

    /// Read-only view onto the indices of each non-identity single-qubit term.
    ///
    /// The indices will always be in sorted order.
    #[getter]
    fn get_indices(slf_: Bound<Self>) -> Bound<PyArray1<u32>> {
        let borrowed = slf_.borrow();
        let indices = borrowed.inner.indices();
        let arr = ::ndarray::aview1(indices);
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[u32]>` allocation (including dropping or reallocating it) other than the entire
        // object getting dropped, which Python will keep safe.
        let out = unsafe { PyArray1::borrow_from_array(&arr, slf_.into_any()) };
        out.readwrite().make_nonwriteable();
        out
    }

    /// Get a :class:`~.quantum_info.Pauli` object that represents the measurement basis needed for
    /// this term.
    ///
    /// For example, the projector ``0l+`` will return a Pauli ``ZYX``.  The resulting
    /// :class:`~.quantum_info.Pauli` is dense, in the sense that explicit identities are stored.
    /// An identity in the Pauli output does not require a concrete measurement.
    ///
    /// Returns: :class:`~.quantum_info.Pauli`: the Pauli operator representing the necessary
    ///     measurement basis.
    ///
    /// See also:
    ///     :meth:`SparseObservable.pauli_bases`
    ///         A similar method for an entire observable at once.
    fn pauli_base<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut x = vec![false; self.inner.num_qubits() as usize];
        let mut z = vec![false; self.inner.num_qubits() as usize];
        for (bit_term, index) in self
            .inner
            .bit_terms()
            .iter()
            .zip(self.inner.indices().iter())
        {
            x[*index as usize] = bit_term.has_x_component();
            z[*index as usize] = bit_term.has_z_component();
        }
        PAULI_TYPE
            .get_bound(py)
            .call1(((PyArray1::from_vec(py, z), PyArray1::from_vec(py, x)),))
    }

    /// Return the bit labels of the term as string.
    ///
    /// The bit labels will match the order of :attr:`.SparseTerm.indices`, such that the
    /// i-th character in the string is applied to the qubit index at ``term.indices[i]``.
    ///
    /// Returns:
    ///     The non-identity bit terms as concatenated string.
    fn bit_labels<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        let string: String = self
            .inner
            .bit_terms()
            .iter()
            .map(|bit| bit.py_label())
            .collect();
        PyString::new(py, string.as_str())
    }
}

/// An observable over Pauli bases that stores its data in a qubit-sparse format.
///
/// Mathematics
/// ===========
///
/// This observable represents a sum over strings of the Pauli operators and Pauli-eigenstate
/// projectors, with each term weighted by some complex number.  That is, the full observable is
///
/// .. math::
///
///     \text{\texttt{SparseObservable}} = \sum_i c_i \bigotimes_n A^{(n)}_i
///
/// for complex numbers :math:`c_i` and single-qubit operators acting on qubit :math:`n` from a
/// restricted alphabet :math:`A^{(n)}_i`.  The sum over :math:`i` is the sum of the individual
/// terms, and the tensor product produces the operator strings.
///
/// The alphabet of allowed single-qubit operators that the :math:`A^{(n)}_i` are drawn from is the
/// Pauli operators and the Pauli-eigenstate projection operators.  Explicitly, these are:
///
/// .. _sparse-observable-alphabet:
/// .. table:: Alphabet of single-qubit terms used in :class:`SparseObservable`
///
///   =======  =======================================  ===============  ===========================
///   Label    Operator                                 Numeric value    :class:`.BitTerm` attribute
///   =======  =======================================  ===============  ===========================
///   ``"I"``  :math:`I` (identity)                     Not stored.      Not stored.
///
///   ``"X"``  :math:`X` (Pauli X)                      ``0b0010`` (2)   :attr:`~.BitTerm.X`
///
///   ``"Y"``  :math:`Y` (Pauli Y)                      ``0b0011`` (3)   :attr:`~.BitTerm.Y`
///
///   ``"Z"``  :math:`Z` (Pauli Z)                      ``0b0001`` (1)   :attr:`~.BitTerm.Z`
///
///   ``"+"``  :math:`\lvert+\rangle\langle+\rvert`     ``0b1010`` (10)  :attr:`~.BitTerm.PLUS`
///            (projector to positive eigenstate of X)
///
///   ``"-"``  :math:`\lvert-\rangle\langle-\rvert`     ``0b0110`` (6)   :attr:`~.BitTerm.MINUS`
///            (projector to negative eigenstate of X)
///
///   ``"r"``  :math:`\lvert r\rangle\langle r\rvert`   ``0b1011`` (11)  :attr:`~.BitTerm.RIGHT`
///            (projector to positive eigenstate of Y)
///
///   ``"l"``  :math:`\lvert l\rangle\langle l\rvert`   ``0b0111`` (7)   :attr:`~.BitTerm.LEFT`
///            (projector to negative eigenstate of Y)
///
///   ``"0"``  :math:`\lvert0\rangle\langle0\rvert`     ``0b1001`` (9)   :attr:`~.BitTerm.ZERO`
///            (projector to positive eigenstate of Z)
///
///   ``"1"``  :math:`\lvert1\rangle\langle1\rvert`     ``0b0101`` (5)   :attr:`~.BitTerm.ONE`
///            (projector to negative eigenstate of Z)
///   =======  =======================================  ===============  ===========================
///
/// The allowed alphabet forms an overcomplete basis of the operator space.  This means that there
/// is not a unique summation to represent a given observable.  By comparison,
/// :class:`.SparsePauliOp` uses a precise basis of the operator space, so (after combining terms of
/// the same Pauli string, removing zeros, and sorting the terms to :ref:`some canonical order
/// <sparse-observable-canonical-order>`) there is only one representation of any operator.
///
/// :class:`SparseObservable` uses its particular overcomplete basis with the aim of making
/// "efficiency of measurement" equivalent to "efficiency of representation".  For example, the
/// observable :math:`{\lvert0\rangle\langle0\rvert}^{\otimes n}` can be efficiently measured on
/// hardware with simple :math:`Z` measurements, but can only be represented by
/// :class:`.SparsePauliOp` as :math:`{(I + Z)}^{\otimes n}/2^n`, which requires :math:`2^n` stored
/// terms.  :class:`SparseObservable` requires only a single term to store this.
///
/// The downside to this is that it is impractical to take an arbitrary matrix or
/// :class:`.SparsePauliOp` and find the *best* :class:`SparseObservable` representation.  You
/// typically will want to construct a :class:`SparseObservable` directly, rather than trying to
/// decompose into one.
///
///
/// Representation
/// ==============
///
/// The internal representation of a :class:`SparseObservable` stores only the non-identity qubit
/// operators.  This makes it significantly more efficient to represent observables such as
/// :math:`\sum_{n\in \text{qubits}} Z^{(n)}`; :class:`SparseObservable` requires an amount of
/// memory linear in the total number of qubits, while :class:`.SparsePauliOp` scales quadratically.
///
/// The terms are stored compressed, similar in spirit to the compressed sparse row format of sparse
/// matrices.  In this analogy, the terms of the sum are the "rows", and the qubit terms are the
/// "columns", where an absent entry represents the identity rather than a zero.  More explicitly,
/// the representation is made up of four contiguous arrays:
///
/// .. _sparse-observable-arrays:
/// .. table:: Data arrays used to represent :class:`.SparseObservable`
///
///   ==================  ===========  =============================================================
///   Attribute           Length       Description
///   ==================  ===========  =============================================================
///   :attr:`coeffs`      :math:`t`    The complex scalar multiplier for each term.
///
///   :attr:`bit_terms`   :math:`s`    Each of the non-identity single-qubit terms for all of the
///                                    operators, in order.  These correspond to the non-identity
///                                    :math:`A^{(n)}_i` in the sum description, where the entries
///                                    are stored in order of increasing :math:`i` first, and in
///                                    order of increasing :math:`n` within each term.
///
///   :attr:`indices`     :math:`s`    The corresponding qubit (:math:`n`) for each of the operators
///                                    in :attr:`bit_terms`.  :class:`SparseObservable` requires
///                                    that this list is term-wise sorted, and algorithms can rely
///                                    on this invariant being upheld.
///
///   :attr:`boundaries`  :math:`t+1`  The indices that partition :attr:`bit_terms` and
///                                    :attr:`indices` into complete terms.  For term number
///                                    :math:`i`, its complex coefficient is ``coeffs[i]``, and its
///                                    non-identity single-qubit operators and their corresponding
///                                    qubits are the slice ``boundaries[i] : boundaries[i+1]`` into
///                                    :attr:`bit_terms` and :attr:`indices` respectively.
///                                    :attr:`boundaries` always has an explicit 0 as its first
///                                    element.
///   ==================  ===========  =============================================================
///
/// The length parameter :math:`t` is the number of terms in the sum, and the parameter :math:`s` is
/// the total number of non-identity single-qubit terms.
///
/// As illustrative examples:
///
/// * in the case of a zero operator, :attr:`boundaries` is length 1 (a single 0) and all other
///   vectors are empty.
/// * in the case of a fully simplified identity operator, :attr:`boundaries` is ``[0, 0]``,
///   :attr:`coeffs` has a single entry, and :attr:`bit_terms` and :attr:`indices` are empty.
/// * for the operator :math:`Z_2 Z_0 - X_3 Y_1`, :attr:`boundaries` is ``[0, 2, 4]``,
///   :attr:`coeffs` is ``[1.0, -1.0]``, :attr:`bit_terms` is ``[BitTerm.Z, BitTerm.Z, BitTerm.Y,
///   BitTerm.X]`` and :attr:`indices` is ``[0, 2, 1, 3]``.  The operator might act on more than
///   four qubits, depending on the :attr:`num_qubits` parameter.  The :attr:`bit_terms` are integer
///   values, whose magic numbers can be accessed via the :class:`BitTerm` attribute class.  Note
///   that the single-bit terms and indices are sorted into termwise sorted order.  This is a
///   requirement of the class.
///
/// These cases are not special, they're fully consistent with the rules and should not need special
/// handling.
///
/// The scalar item of the :attr:`bit_terms` array is stored as a numeric byte.  The numeric values
/// are related to the symplectic Pauli representation that :class:`.SparsePauliOp` uses, and are
/// accessible with named access by an enumeration:
///
/// ..
///     This is documented manually here because the Python-space `Enum` is generated
///     programmatically from Rust - it'd be _more_ confusing to try and write a docstring somewhere
///     else in this source file. The use of `autoattribute` is because it pulls in the numeric
///     value.
///
/// .. py:class:: SparseObservable.BitTerm
///
///     An :class:`~enum.IntEnum` that provides named access to the numerical values used to
///     represent each of the single-qubit alphabet terms enumerated in
///     :ref:`sparse-observable-alphabet`.
///
///     This class is attached to :class:`.SparseObservable`.  Access it as
///     :class:`.SparseObservable.BitTerm`.  If this is too much typing, and you are solely dealing
///     with :class:¬SparseObservable` objects and the :class:`BitTerm` name is not ambiguous, you
///     might want to shorten it as::
///
///         >>> ops = SparseObservable.BitTerm
///         >>> assert ops.X is SparseObservable.BitTerm.X
///
///     You can access all the values of the enumeration by either their full all-capitals name, or
///     by their single-letter label.  The single-letter labels are not generally valid Python
///     identifiers, so you must use indexing notation to access them::
///
///         >>> assert SparseObservable.BitTerm.ZERO is SparseObservable.BitTerm["0"]
///
///     The numeric structure of these is that they are all four-bit values of which the low two
///     bits are the (phase-less) symplectic representation of the Pauli operator related to the
///     object, where the low bit denotes a contribution by :math:`Z` and the second lowest a
///     contribution by :math:`X`, while the upper two bits are ``00`` for a Pauli operator, ``01``
///     for the negative-eigenstate projector, and ``10`` for the positive-eigenstate projector.
///
///     Values
///     ------
///
///     .. autoattribute:: qiskit.quantum_info::SparseObservable.BitTerm.X
///
///         The Pauli :math:`X` operator.  Uses the single-letter label ``"X"``.
///
///     .. autoattribute:: qiskit.quantum_info::SparseObservable.BitTerm.PLUS
///
///         The projector to the positive eigenstate of the :math:`X` operator:
///         :math:`\lvert+\rangle\langle+\rvert`.  Uses the single-letter label ``"+"``.
///
///     .. autoattribute:: qiskit.quantum_info::SparseObservable.BitTerm.MINUS
///
///         The projector to the negative eigenstate of the :math:`X` operator:
///         :math:`\lvert-\rangle\langle-\rvert`.  Uses the single-letter label ``"-"``.
///
///     .. autoattribute:: qiskit.quantum_info::SparseObservable.BitTerm.Y
///
///         The Pauli :math:`Y` operator.  Uses the single-letter label ``"Y"``.
///
///     .. autoattribute:: qiskit.quantum_info::SparseObservable.BitTerm.RIGHT
///
///         The projector to the positive eigenstate of the :math:`Y` operator:
///         :math:`\lvert r\rangle\langle r\rvert`.  Uses the single-letter label ``"r"``.
///
///     .. autoattribute:: qiskit.quantum_info::SparseObservable.BitTerm.LEFT
///
///         The projector to the negative eigenstate of the :math:`Y` operator:
///         :math:`\lvert l\rangle\langle l\rvert`.  Uses the single-letter label ``"l"``.
///
///     .. autoattribute:: qiskit.quantum_info::SparseObservable.BitTerm.Z
///
///         The Pauli :math:`Z` operator.  Uses the single-letter label ``"Z"``.
///
///     .. autoattribute:: qiskit.quantum_info::SparseObservable.BitTerm.ZERO
///
///         The projector to the positive eigenstate of the :math:`Z` operator:
///         :math:`\lvert0\rangle\langle0\rvert`.  Uses the single-letter label ``"0"``.
///
///     .. autoattribute:: qiskit.quantum_info::SparseObservable.BitTerm.ONE
///
///         The projector to the negative eigenstate of the :math:`Z` operator:
///         :math:`\lvert1\rangle\langle1\rvert`.  Uses the single-letter label ``"1"``.
///
///     Attributes
///     ----------
///
///     .. autoproperty:: qiskit.quantum_info::SparseObservable.BitTerm.label
///
/// Each of the array-like attributes behaves like a Python sequence.  You can index and slice these
/// with standard :class:`list`-like semantics.  Slicing an attribute returns a Numpy
/// :class:`~numpy.ndarray` containing a copy of the relevant data with the natural ``dtype`` of the
/// field; this lets you easily do mathematics on the results, like bitwise operations on
/// :attr:`bit_terms`.  You can assign to indices or slices of each of the attributes, but beware
/// that you must uphold :ref:`the data coherence rules <sparse-observable-arrays>` while doing
/// this.  For example::
///
///     >>> obs = SparseObservable.from_list([("XZY", 1.5j), ("+1r", -0.5)])
///     >>> assert isinstance(obs.coeffs[:], np.ndarray)
///     >>> # Reduce all single-qubit terms to the relevant Pauli operator, if they are a projector.
///     >>> obs.bit_terms[:] = obs.bit_terms[:] & 0b00_11
///     >>> assert obs == SparseObservable.from_list([("XZY", 1.5j), ("XZY", -0.5)])
///
/// .. note::
///
///     The above reduction to the Pauli bases can also be achieved with :meth:`pauli_bases`.
///
/// .. _sparse-observable-canonical-order:
///
/// Canonical ordering
/// ------------------
///
/// For any given mathematical observable, there are several ways of representing it with
/// :class:`SparseObservable`.  For example, the same set of single-bit terms and their
/// corresponding indices might appear multiple times in the observable.  Mathematically, this is
/// equivalent to having only a single term with all the coefficients summed.  Similarly, the terms
/// of the sum in a :class:`SparseObservable` can be in any order while representing the same
/// observable, since addition is commutative (although while floating-point addition is not
/// associative, :class:`SparseObservable` makes no guarantees about the summation order).
///
/// These two categories of representation degeneracy can cause the ``==`` operator to claim that
/// two observables are not equal, despite representing the same object.  In these cases, it can
/// be convenient to define some *canonical form*, which allows observables to be compared
/// structurally.
///
/// You can put a :class:`SparseObservable` in canonical form by using the :meth:`simplify` method.
/// The precise ordering of terms in canonical ordering is not specified, and may change between
/// versions of Qiskit.  Within the same version of Qiskit, however, you can compare two observables
/// structurally by comparing their simplified forms.
///
/// .. note::
///
///     If you wish to account for floating-point tolerance in the comparison, it is safest to use
///     a recipe such as::
///
///         def equivalent(left, right, tol):
///             return (left - right).simplify(tol) == SparseObservable.zero(left.num_qubits)
///
/// .. note::
///
///     The canonical form produced by :meth:`simplify` alone will not universally detect all
///     observables that are equivalent due to the over-complete basis alphabet. To obtain a
///     unique expression, you can first represent the observable using Pauli terms only by
///     calling :meth:`as_paulis`, followed by :meth:`simplify`. Note that the projector
///     expansion (e.g. ``+`` into ``I`` and ``X``) is not computationally feasible at scale.
///
/// Indexing
/// --------
///
/// :class:`SparseObservable` behaves as `a Python sequence
/// <https://docs.python.org/3/glossary.html#term-sequence>`__ (the standard form, not the expanded
/// :class:`collections.abc.Sequence`).  The observable can be indexed by integers, and iterated
/// through to yield individual terms.
///
/// Each term appears as an instance a self-contained class.  The individual terms are copied out of
/// the base observable; mutations to them will not affect the observable.
///
/// .. autoclass:: qiskit.quantum_info::SparseObservable.Term
///     :members:
///
/// Construction
/// ============
///
/// :class:`SparseObservable` defines several constructors.  The default constructor will attempt to
/// delegate to one of the more specific constructors, based on the type of the input.  You can
/// always use the specific constructors to have more control over the construction.
///
/// .. _sparse-observable-convert-constructors:
/// .. table:: Construction from other objects
///
///   ============================  ================================================================
///   Method                        Summary
///   ============================  ================================================================
///   :meth:`from_label`            Convert a dense string label into a single-term
///                                 :class:`.SparseObservable`.
///
///   :meth:`from_list`             Sum a list of tuples of dense string labels and the associated
///                                 coefficients into an observable.
///
///   :meth:`from_sparse_list`      Sum a list of tuples of sparse string labels, the qubits they
///                                 apply to, and their coefficients into an observable.
///
///   :meth:`from_pauli`            Raise a single :class:`~.quantum_info.Pauli` into a single-term
///                                 :class:`.SparseObservable`.
///
///   :meth:`from_sparse_pauli_op`  Raise a :class:`.SparsePauliOp` into a
///                                 :class:`SparseObservable`.
///
///   :meth:`from_terms`            Sum explicit single :class:`Term` instances.
///
///   :meth:`from_raw_parts`        Build the observable from :ref:`the raw data arrays
///                                 <sparse-observable-arrays>`.
///   ============================  ================================================================
///
/// .. py:function:: SparseObservable.__new__(data, /, num_qubits=None)
///
///     The default constructor of :class:`SparseObservable`.
///
///     This delegates to one of :ref:`the explicit conversion-constructor methods
///     <sparse-observable-convert-constructors>`, based on the type of the ``data`` argument.  If
///     ``num_qubits`` is supplied and constructor implied by the type of ``data`` does not accept a
///     number, the given integer must match the input.
///
///     :param data: The data type of the input.  This can be another :class:`SparseObservable`, in
///         which case the input is copied, a :class:`~.quantum_info.Pauli` or :class:`.SparsePauliOp`, in which
///         case :meth:`from_pauli` or :meth:`from_sparse_pauli_op` are called as appropriate, or it
///         can be a list in a valid format for either :meth:`from_list` or
///         :meth:`from_sparse_list`.
///     :param int|None num_qubits: Optional number of qubits for the operator.  For most data
///         inputs, this can be inferred and need not be passed.  It is only necessary for empty
///         lists or the sparse-list format.  If given unnecessarily, it must match the data input.
///
/// In addition to the conversion-based constructors, there are also helper methods that construct
/// special forms of observables.
///
/// .. table:: Construction of special observables
///
///   ============================  ================================================================
///   Method                        Summary
///   ============================  ================================================================
///   :meth:`zero`                  The zero operator on a given number of qubits.
///
///   :meth:`identity`              The identity operator on a given number of qubits.
///   ============================  ================================================================
///
/// Conversions
/// ===========
///
/// An existing :class:`SparseObservable` can be converted into other :mod:`~qiskit.quantum_info`
/// operators or generic formats.  Beware that other objects may not be able to represent the same
/// observable as efficiently as :class:`SparseObservable`, including potentially needed
/// exponentially more memory.
///
/// .. table:: Conversion methods to other observable forms.
///
///   ===========================  =================================================================
///   Method                       Summary
///   ===========================  =================================================================
///   :meth:`as_paulis`            Create a new :class:`SparseObservable`, expanding in terms
///                                of Pauli operators only.
///
///   :meth:`to_sparse_list`       Express the observable in a sparse list format with elements
///                                ``(bit_terms, indices, coeff)``.
///   ===========================  =================================================================
///
/// In addition, :meth:`.SparsePauliOp.from_sparse_observable` is available for conversion from this
/// class to :class:`.SparsePauliOp`.  Beware that this method suffers from the same
/// exponential-memory usage concerns as :meth:`as_paulis`.
///
/// Mathematical manipulation
/// =========================
///
/// :class:`SparseObservable` supports the standard set of Python mathematical operators like other
/// :mod:`~qiskit.quantum_info` operators.
///
/// In basic arithmetic, you can:
///
/// * add two observables using ``+``
/// * subtract two observables using ``-``
/// * multiply or divide by an :class:`int`, :class:`float` or :class:`complex` using ``*`` and ``/``
/// * negate all the coefficients in an observable with unary ``-``
///
/// Each of the basic binary arithmetic operators has a corresponding specialized in-place method,
/// which mutates the left-hand side in-place.  Using these is typically more efficient than the
/// infix operators, especially for building an observable in a loop.
///
/// The tensor product is calculated with :meth:`tensor` (for standard, juxtaposition ordering of
/// Pauli labels) or :meth:`expand` (for the reverse order).  The ``^`` operator is overloaded to be
/// equivalent to :meth:`tensor`.
///
/// .. note::
///
///     When using the binary operators ``^`` (:meth:`tensor`) and ``&`` (:meth:`compose`), beware
///     that `Python's operator-precedence rules
///     <https://docs.python.org/3/reference/expressions.html#operator-precedence>`__ may cause the
///     evaluation order to be different to your expectation.  In particular, the operator ``+``
///     binds more tightly than ``^`` or ``&``, just like ``*`` binds more tightly than ``+``.
///
///     When using the operators in mixed expressions, it is safest to use parentheses to group the
///     operands of tensor products.
///
/// A :class:`SparseObservable` has a well-defined :meth:`adjoint`.  The notions of scalar complex
/// conjugation (:meth:`conjugate`) and real-value transposition (:meth:`transpose`) are defined
/// analogously to the matrix representation of other Pauli operators in Qiskit.
///
///
/// Efficiency notes
/// ----------------
///
/// Internally, :class:`SparseObservable` is in-place mutable, including using over-allocating
/// growable vectors for extending the number of terms.  This means that the cost of appending to an
/// observable using ``+=`` is amortised linear in the total number of terms added, rather than
/// the quadratic complexity that the binary ``+`` would require.
///
/// Additions and subtractions are implemented by a term-stacking operation; there is no automatic
/// "simplification" (summing of like terms), because the majority of additions to build up an
/// observable generate only a small number of duplications, and like-term detection has additional
/// costs.  If this does not fit your use cases, you can either periodically call :meth:`simplify`,
/// or discuss further APIs with us for better building of observables.
#[cfg(feature = "python")]
#[pyclass(name = "SparseObservable", module = "qiskit.quantum_info", sequence)]
#[derive(Debug)]
pub struct PySparseObservable {
    // This class keeps a pointer to a pure Rust-SparseTerm and serves as interface from Python.
    pub inner: Arc<RwLock<SparseObservable>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySparseObservable {
    #[pyo3(signature = (data, /, num_qubits=None))]
    #[new]
    fn py_new(data: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let py = data.py();
        let check_num_qubits = |data: &Bound<PyAny>| -> PyResult<()> {
            let Some(num_qubits) = num_qubits else {
                return Ok(());
            };
            let other_qubits = data.getattr(intern!(py, "num_qubits"))?.extract::<u32>()?;
            if num_qubits == other_qubits {
                return Ok(());
            }
            Err(PyValueError::new_err(format!(
                "explicitly given 'num_qubits' ({num_qubits}) does not match operator ({other_qubits})"
            )))
        };

        if data.is_instance(PAULI_TYPE.get_bound(py))? {
            check_num_qubits(data)?;
            return Self::from_pauli(data);
        }
        if data.is_instance(SPARSE_PAULI_OP_TYPE.get_bound(py))? {
            check_num_qubits(data)?;
            return Self::from_sparse_pauli_op(data);
        }
        if let Ok(label) = data.extract::<String>() {
            let num_qubits = num_qubits.unwrap_or(label.len() as u32);
            if num_qubits as usize != label.len() {
                return Err(PyValueError::new_err(format!(
                    "explicitly given 'num_qubits' ({}) does not match label ({})",
                    num_qubits,
                    label.len(),
                )));
            }
            return Self::from_label(&label).map_err(PyErr::from);
        }
        if let Ok(observable) = data.cast_exact::<Self>() {
            check_num_qubits(data)?;
            let borrowed = observable.borrow();
            let inner = borrowed.inner.read().map_err(|_| InnerReadError)?;
            return Ok(inner.clone().into());
        }
        // The type of `vec` is inferred from the subsequent calls to `Self::from_list` or
        // `Self::from_sparse_list` to be either the two-tuple or the three-tuple form during the
        // `extract`.  The empty list will pass either, but it means the same to both functions.
        if let Ok(vec) = data.extract() {
            return Self::from_list(vec, num_qubits);
        }
        if let Ok(vec) = data.extract() {
            let Some(num_qubits) = num_qubits else {
                return Err(PyValueError::new_err(
                    "if using the sparse-list form, 'num_qubits' must be provided",
                ));
            };
            return Self::from_sparse_list(vec, num_qubits);
        }
        if let Ok(term) = data.cast_exact::<PySparseTerm>() {
            return term.borrow().to_observable();
        };
        if let Ok(observable) = Self::from_terms(data, num_qubits) {
            return Ok(observable);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'SparseObservable': {}",
            data.get_type().repr()?,
        )))
    }

    /// Get a copy of this observable.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> obs = SparseObservable.from_list([("IXZ+lr01", 2.5), ("ZXI-rl10", 0.5j)])
    ///         >>> assert obs == obs.copy()
    ///         >>> assert obs is not obs.copy()
    fn copy(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.clone().into())
    }

    /// The number of qubits the operator acts on.
    ///
    /// This is not inferable from any other shape or values, since identities are not stored
    /// explicitly.
    #[getter]
    #[inline]
    pub fn num_qubits(&self) -> PyResult<u32> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_qubits())
    }

    /// The number of terms in the sum this operator is tracking.
    #[getter]
    #[inline]
    pub fn num_terms(&self) -> PyResult<usize> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_terms())
    }

    /// The coefficients of each abstract term in in the sum.  This has as many elements as terms in
    /// the sum.
    #[getter]
    fn get_coeffs(slf_: &Bound<Self>) -> ArrayView {
        let borrowed = slf_.borrow();
        ArrayView {
            base: borrowed.inner.clone(),
            slot: ArraySlot::Coeffs,
        }
    }

    /// A flat list of single-qubit terms.  This is more naturally a list of lists, but is stored
    /// flat for memory usage and locality reasons, with the sublists denoted by `boundaries.`
    #[getter]
    fn get_bit_terms(slf_: &Bound<Self>) -> ArrayView {
        let borrowed = slf_.borrow();
        ArrayView {
            base: borrowed.inner.clone(),
            slot: ArraySlot::BitTerms,
        }
    }

    /// A flat list of the qubit indices that the corresponding entries in :attr:`bit_terms` act on.
    /// This list must always be term-wise sorted, where a term is a sublist as denoted by
    /// :attr:`boundaries`.
    ///
    /// .. warning::
    ///
    ///     If writing to this attribute from Python space, you *must* ensure that you only write in
    ///     indices that are term-wise sorted.
    #[getter]
    fn get_indices(slf_: &Bound<Self>) -> ArrayView {
        let borrowed = slf_.borrow();
        ArrayView {
            base: borrowed.inner.clone(),
            slot: ArraySlot::Indices,
        }
    }

    /// Indices that partition :attr:`bit_terms` and :attr:`indices` into sublists for each
    /// individual term in the sum.  ``boundaries[0] : boundaries[1]`` is the range of indices into
    /// :attr:`bit_terms` and :attr:`indices` that correspond to the first term of the sum.  All
    /// unspecified qubit indices are implicitly the identity.  This is one item longer than
    /// :attr:`coeffs`, since ``boundaries[0]`` is always an explicit zero (for algorithmic ease).
    #[getter]
    fn get_boundaries(slf_: &Bound<Self>) -> ArrayView {
        let borrowed = slf_.borrow();
        ArrayView {
            base: borrowed.inner.clone(),
            slot: ArraySlot::Boundaries,
        }
    }

    /// Get the zero operator over the given number of qubits.
    ///
    /// The zero operator is the operator whose expectation value is zero for all quantum states.
    /// It has no terms.  It is the identity element for addition of two :class:`SparseObservable`
    /// instances; anything added to the zero operator is equal to itself.
    ///
    /// If you want the projector onto the all zeros state, use::
    ///
    ///     >>> num_qubits = 10
    ///     >>> all_zeros = SparseObservable.from_label("0" * num_qubits)
    ///
    /// Examples:
    ///
    ///     Get the zero operator for 100 qubits::
    ///
    ///         >>> SparseObservable.zero(100)
    ///         <SparseObservable with 0 terms on 100 qubits: 0.0>
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn zero(num_qubits: u32) -> Self {
        SparseObservable::zero(num_qubits).into()
    }

    /// Get the identity operator over the given number of qubits.
    ///
    /// Examples:
    ///
    ///     Get the identity operator for 100 qubits::
    ///
    ///         >>> SparseObservable.identity(100)
    ///         <SparseObservable with 1 term on 100 qubits: (1+0j)()>
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn identity(num_qubits: u32) -> Self {
        SparseObservable::identity(num_qubits).into()
    }

    /// Construct a :class:`.SparseObservable` from a single :class:`~.quantum_info.Pauli` instance.
    ///
    /// The output observable will have a single term, with a unitary coefficient dependent on the
    /// phase.
    ///
    /// Args:
    ///     pauli (:class:`~.quantum_info.Pauli`): the single Pauli to convert.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> SparseObservable.from_pauli(pauli)
    ///         <SparseObservable with 1 term on 5 qubits: (1+0j)(Y_3 X_2 Z_1)>
    ///         >>> assert SparseObservable.from_label(label) == SparseObservable.from_pauli(pauli)
    #[staticmethod]
    #[pyo3(signature = (pauli, /))]
    fn from_pauli(pauli: &Bound<PyAny>) -> PyResult<Self> {
        let py = pauli.py();
        let num_qubits = pauli.getattr(intern!(py, "num_qubits"))?.extract::<u32>()?;
        let z = pauli
            .getattr(intern!(py, "z"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let x = pauli
            .getattr(intern!(py, "x"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let mut bit_terms = Vec::new();
        let mut indices = Vec::new();
        let mut num_ys = 0;
        for (i, (x, z)) in x.as_array().iter().zip(z.as_array().iter()).enumerate() {
            // The only failure case possible here is the identity, because of how we're
            // constructing the value to convert.
            let Ok(term) = ::bytemuck::checked::try_cast(((*x as u8) << 1) | (*z as u8)) else {
                continue;
            };
            num_ys += (term == BitTerm::Y) as isize;
            indices.push(i as u32);
            bit_terms.push(term);
        }
        let boundaries = vec![0, indices.len()];
        // The "empty" state of a `Pauli` represents the identity, which isn't our empty state
        // (that's zero), so we're always going to have a coefficient.
        let group_phase = pauli
            // `Pauli`'s `_phase` is a Numpy array ...
            .getattr(intern!(py, "_phase"))?
            // ... that should have exactly 1 element ...
            .call_method0(intern!(py, "item"))?
            // ... which is some integral type.
            .extract::<isize>()?;
        let phase = match (group_phase - num_ys).rem_euclid(4) {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, -1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, 1.0),
            _ => unreachable!("`x % 4` has only four values"),
        };
        let coeffs = vec![phase];
        let inner = SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)?;
        Ok(inner.into())
    }

    /// Construct a single-term observable from a dense string label.
    ///
    /// The resulting operator will have a coefficient of 1.  The label must be a sequence of the
    /// alphabet ``'IXYZ+-rl01'``.  The label is interpreted analogously to a bitstring.  In other
    /// words, the right-most letter is associated with qubit 0, and so on.  This is the same as the
    /// labels for :class:`~.quantum_info.Pauli` and :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     label (str): the dense label.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> SparseObservable.from_label("IIII+ZI")
    ///         <SparseObservable with 1 term on 7 qubits: (1+0j)(+_2 Z_1)>
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> assert SparseObservable.from_label(label) == SparseObservable.from_pauli(pauli)
    ///
    /// See also:
    ///     :meth:`from_list`
    ///         A generalization of this method that constructs a sum operator from multiple labels
    ///         and their corresponding coefficients.
    #[staticmethod]
    #[pyo3(signature = (label, /))]
    pub fn from_label(label: &str) -> Result<Self, LabelError> {
        let mut inner = SparseObservable::zero(label.len() as u32);
        inner.add_dense_label(label, Complex64::new(1.0, 0.0))?;
        Ok(inner.into())
    }

    /// Construct an observable from a list of dense labels and coefficients.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_list`, except it uses
    /// :ref:`the extended alphabet <sparse-observable-alphabet>` of :class:`.SparseObservable`.  In
    /// this dense form, you must supply all identities explicitly in each label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ+-rl01'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`~.quantum_info.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     iter (list[tuple[str, complex]]): Pairs of labels and their associated coefficients to
    ///         sum. The labels are interpreted the same way as in :meth:`from_label`.
    ///     num_qubits (int | None): It is not necessary to specify this if you are sure that
    ///         ``iter`` is not an empty sequence, since it can be inferred from the label lengths.
    ///         If ``iter`` may be empty, you must specify this argument to disambiguate how many
    ///         qubits the observable is for.  If this is given and ``iter`` is not empty, the value
    ///         must match the label lengths.
    ///
    /// Examples:
    ///
    ///     Construct an observable from a list of labels of the same length::
    ///
    ///         >>> SparseObservable.from_list([
    ///         ...     ("III++", 1.0),
    ///         ...     ("II--I", 1.0j),
    ///         ...     ("I++II", -0.5),
    ///         ...     ("--III", -0.25j),
    ///         ... ])
    ///         <SparseObservable with 4 terms on 5 qubits:
    ///             (1+0j)(+_1 +_0) + (0+1j)(-_2 -_1) + (-0.5+0j)(+_3 +_2) + (-0-0.25j)(-_4 -_3)>
    ///
    ///     Use ``num_qubits`` to disambiguate potentially empty inputs::
    ///
    ///         >>> SparseObservable.from_list([], num_qubits=10)
    ///         <SparseObservable with 0 terms on 10 qubits: 0.0>
    ///
    ///     This method is equivalent to calls to :meth:`from_sparse_list` with the explicit
    ///     qubit-arguments field set to decreasing integers::
    ///
    ///         >>> labels = ["XY+Z", "rl01", "-lXZ"]
    ///         >>> coeffs = [1.5j, 2.0, -0.5]
    ///         >>> from_list = SparseObservable.from_list(list(zip(labels, coeffs)))
    ///         >>> from_sparse_list = SparseObservable.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0), coeff)
    ///         ...     for label, coeff in zip(labels, coeffs)
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`from_label`
    ///         A similar constructor, but takes only a single label and always has its coefficient
    ///         set to ``1.0``.
    ///
    ///     :meth:`from_sparse_list`
    ///         Construct the observable from a list of labels without explicit identities, but with
    ///         the qubits each single-qubit term applies to listed explicitly.
    #[staticmethod]
    #[pyo3(signature = (iter, /, *, num_qubits=None))]
    fn from_list(iter: Vec<(String, Complex64)>, num_qubits: Option<u32>) -> PyResult<Self> {
        if iter.is_empty() && num_qubits.is_none() {
            return Err(PyValueError::new_err(
                "cannot construct an observable from an empty list without knowing `num_qubits`",
            ));
        }
        let num_qubits = match num_qubits {
            Some(num_qubits) => num_qubits,
            None => iter[0].0.len() as u32,
        };
        let mut inner = SparseObservable::with_capacity(num_qubits, iter.len(), 0);
        for (label, coeff) in iter {
            inner.add_dense_label(&label, coeff)?;
        }
        Ok(inner.into())
    }

    /// Construct an observable from a list of labels, the qubits each item applies to, and the
    /// coefficient of the whole term.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`, except it uses
    /// :ref:`the extended alphabet <sparse-observable-alphabet>` of :class:`.SparseObservable`.
    ///
    /// The "labels" and "indices" fields of the triples are associated by zipping them together.
    /// For example, this means that a call to :meth:`from_list` can be converted to the form used
    /// by this method by setting the "indices" field of each triple to ``(num_qubits-1, ..., 1,
    /// 0)``.
    ///
    /// Args:
    ///     iter (list[tuple[str, Sequence[int], complex]]): triples of labels, the qubits
    ///         each single-qubit term applies to, and the coefficient of the entire term.
    ///
    ///     num_qubits (int): the number of qubits in the operator.
    ///
    /// Examples:
    ///
    ///     Construct a simple operator::
    ///
    ///         >>> SparseObservable.from_sparse_list(
    ///         ...     [("ZX", (1, 4), 1.0), ("YY", (0, 3), 2j)],
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <SparseObservable with 2 terms on 5 qubits: (1+0j)(X_4 Z_1) + (0+2j)(Y_3 Y_0)>
    ///
    ///     Construct the identity observable (though really, just use :meth:`identity`)::
    ///
    ///         >>> SparseObservable.from_sparse_list([("", (), 1.0)], num_qubits=100)
    ///         <SparseObservable with 1 term on 100 qubits: (1+0j)()>
    ///
    ///     This method can replicate the behavior of :meth:`from_list`, if the qubit-arguments
    ///     field of the triple is set to decreasing integers::
    ///
    ///         >>> labels = ["XY+Z", "rl01", "-lXZ"]
    ///         >>> coeffs = [1.5j, 2.0, -0.5]
    ///         >>> from_list = SparseObservable.from_list(list(zip(labels, coeffs)))
    ///         >>> from_sparse_list = SparseObservable.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0), coeff)
    ///         ...     for label, coeff in zip(labels, coeffs)
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`to_sparse_list`
    ///         The reverse of this method.
    #[staticmethod]
    #[pyo3(signature = (iter, /, num_qubits))]
    fn from_sparse_list(
        iter: Vec<(String, Vec<u32>, Complex64)>,
        num_qubits: u32,
    ) -> PyResult<Self> {
        let coeffs = iter.iter().map(|(_, _, coeff)| *coeff).collect();
        let mut boundaries = Vec::with_capacity(iter.len() + 1);
        boundaries.push(0);
        let mut indices = Vec::new();
        let mut bit_terms = Vec::new();
        // Insertions to the `BTreeMap` keep it sorted by keys, so we use this to do the termwise
        // sorting on-the-fly.
        let mut sorted = btree_map::BTreeMap::new();
        for (label, qubits, _) in iter {
            sorted.clear();
            let label: &[u8] = label.as_ref();
            if label.len() != qubits.len() {
                return Err(LabelError::WrongLengthIndices {
                    label: label.len(),
                    indices: indices.len(),
                }
                .into());
            }
            for (letter, index) in label.iter().zip(qubits) {
                if index >= num_qubits {
                    return Err(LabelError::BadIndex { index, num_qubits }.into());
                }
                let btree_map::Entry::Vacant(entry) = sorted.entry(index) else {
                    return Err(LabelError::DuplicateIndex { index }.into());
                };
                entry.insert(
                    BitTerm::try_from_u8(*letter).map_err(|_| LabelError::OutsideAlphabet)?,
                );
            }
            for (index, term) in sorted.iter() {
                let Some(term) = term else {
                    continue;
                };
                indices.push(*index);
                bit_terms.push(*term);
            }
            boundaries.push(bit_terms.len());
        }
        let inner = SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)?;
        Ok(inner.into())
    }

    /// Express the observable in Pauli terms only, by writing each projector as sum of Pauli terms.
    ///
    /// Note that there is no guarantee of the order the resulting Pauli terms. Use
    /// :meth:`SparseObservable.simplify` in addition to obtain a canonical representation.
    ///
    /// .. warning::
    ///
    ///     Beware that this will use at least :math:`2^n` terms if there are :math:`n`
    ///     single-qubit projectors present, which can lead to an exponential number of terms.
    ///
    /// Returns:
    ///     The same observable, but expressed in Pauli terms only.
    ///
    /// Examples:
    ///
    ///     Rewrite an observable in terms of projectors into Pauli operators::
    ///
    ///         >>> obs = SparseObservable("+")
    ///         >>> obs.as_paulis()
    ///         <SparseObservable with 2 terms on 1 qubit: (0.5+0j)() + (0.5+0j)(X_0)>
    ///         >>> direct = SparseObservable.from_list([("I", 0.5), ("Z", 0.5)])
    ///         >>> assert direct.simplify() == obs.as_paulis().simplify()
    ///
    ///     For small operators, this can be used with :meth:`simplify` as a unique canonical form::
    ///
    ///         >>> left = SparseObservable.from_list([("+", 0.5), ("-", 0.5)])
    ///         >>> right = SparseObservable.from_list([("r", 0.5), ("l", 0.5)])
    ///         >>> assert left.as_paulis().simplify() == right.as_paulis().simplify()
    ///
    /// See also:
    ///     :meth:`.SparsePauliOp.from_sparse_observable`
    ///         A constructor of :class:`.SparsePauliOp` that can convert a
    ///         :class:`SparseObservable` in the :class:`.SparsePauliOp` dense Pauli representation.
    fn as_paulis(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.as_paulis().into())
    }

    /// Express the observable in terms of a sparse list format.
    ///
    /// This can be seen as counter-operation of :meth:`.SparseObservable.from_sparse_list`, however
    /// the order of terms is not guaranteed to be the same at after a roundtrip to a sparse
    /// list and back.
    ///
    /// Examples:
    ///
    ///     >>> obs = SparseObservable.from_list([("IIXIZ", 2j), ("IIZIX", 2j)])
    ///     >>> reconstructed = SparseObservable.from_sparse_list(obs.to_sparse_list(), obs.num_qubits)
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         The constructor that can interpret these lists.
    #[pyo3(signature = ())]
    fn to_sparse_list(&self, py: Python) -> PyResult<Py<PyList>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        // turn a SparseView into a Python tuple of (bit terms, indices, coeff)
        let to_py_tuple = |view: SparseTermView| {
            let mut pauli_string = String::with_capacity(view.bit_terms.len());

            for bit in view.bit_terms.iter() {
                pauli_string.push_str(bit.py_label());
            }
            let py_string = PyString::new(py, &pauli_string).unbind();
            let py_indices = PyList::new(py, view.indices.iter())?.unbind();
            let py_coeff = view.coeff.into_py_any(py)?;

            PyTuple::new(py, vec![py_string.as_any(), py_indices.as_any(), &py_coeff])
        };

        let out = PyList::empty(py);
        for view in inner.iter() {
            out.append(to_py_tuple(view)?)?;
        }
        Ok(out.unbind())
    }

    /// Construct a :class:`.SparseObservable` from a :class:`.SparsePauliOp` instance.
    ///
    /// This will be a largely direct translation of the :class:`.SparsePauliOp`; in particular,
    /// there is no on-the-fly summing of like terms, nor any attempt to refactorize sums of Pauli
    /// terms into equivalent projection operators.
    ///
    /// Args:
    ///     op (:class:`.SparsePauliOp`): the operator to convert.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> spo = SparsePauliOp.from_list([("III", 1.0), ("IIZ", 0.5), ("IZI", 0.5)])
    ///         >>> SparseObservable.from_sparse_pauli_op(spo)
    ///         <SparseObservable with 3 terms on 3 qubits: (1+0j)() + (0.5+0j)(Z_0) + (0.5+0j)(Z_1)>
    #[staticmethod]
    #[pyo3(signature = (op, /))]
    fn from_sparse_pauli_op(op: &Bound<PyAny>) -> PyResult<Self> {
        let py = op.py();
        let pauli_list_ob = op.getattr(intern!(py, "paulis"))?;
        let coeffs = op
            .getattr(intern!(py, "coeffs"))?
            .extract::<PyReadonlyArray1<Complex64>>()
            .map_err(|_| PyTypeError::new_err("only 'SparsePauliOp' with complex-typed coefficients can be converted to 'SparseObservable'"))?
            .as_array()
            .to_vec();
        let op_z = pauli_list_ob
            .getattr(intern!(py, "z"))?
            .extract::<PyReadonlyArray2<bool>>()?;
        let op_x = pauli_list_ob
            .getattr(intern!(py, "x"))?
            .extract::<PyReadonlyArray2<bool>>()?;
        // We don't extract the `phase`, because that's supposed to be 0 for all `SparsePauliOp`
        // instances - they use the symplectic convention in the representation with any phase term
        // absorbed into the coefficients (like us).
        let [num_terms, num_qubits] = *op_z.shape() else {
            unreachable!("shape is statically known to be 2D")
        };
        if op_x.shape() != [num_terms, num_qubits] {
            return Err(PyValueError::new_err(format!(
                "'x' and 'z' have different shapes ({:?} and {:?})",
                op_x.shape(),
                op_z.shape()
            )));
        }
        if num_terms != coeffs.len() {
            return Err(PyValueError::new_err(format!(
                "'x' and 'z' have a different number of operators to 'coeffs' ({} and {})",
                num_terms,
                coeffs.len(),
            )));
        }

        let mut bit_terms = Vec::new();
        let mut indices = Vec::new();
        let mut boundaries = Vec::with_capacity(num_terms + 1);
        boundaries.push(0);
        for (term_x, term_z) in op_x
            .as_array()
            .rows()
            .into_iter()
            .zip(op_z.as_array().rows())
        {
            for (i, (x, z)) in term_x.iter().zip(term_z.iter()).enumerate() {
                // The only failure case possible here is the identity, because of how we're
                // constructing the value to convert.
                let Ok(term) = ::bytemuck::checked::try_cast(((*x as u8) << 1) | (*z as u8)) else {
                    continue;
                };
                indices.push(i as u32);
                bit_terms.push(term);
            }
            boundaries.push(indices.len());
        }

        let inner =
            SparseObservable::new(num_qubits as u32, coeffs, bit_terms, indices, boundaries)?;
        Ok(inner.into())
    }

    /// Construct a :class:`SparseObservable` out of individual terms.
    ///
    /// All the terms must have the same number of qubits.  If supplied, the ``num_qubits`` argument
    /// must match the terms.
    ///
    /// No simplification is done as part of the observable creation.
    ///
    /// Args:
    ///     obj (Iterable[Term]): Iterable of individual terms to build the observable from.
    ///     num_qubits (int | None): The number of qubits the observable should act on.  This is
    ///         usually inferred from the input, but can be explicitly given to handle the case
    ///         of an empty iterable.
    ///
    /// Returns:
    ///     The corresponding observable.
    #[staticmethod]
    #[pyo3(signature = (obj, /, num_qubits=None))]
    fn from_terms(obj: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let mut iter = obj.try_iter()?;
        let mut inner = match num_qubits {
            Some(num_qubits) => SparseObservable::zero(num_qubits),
            None => {
                let Some(first) = iter.next() else {
                    return Err(PyValueError::new_err(
                        "cannot construct an observable from an empty list without knowing `num_qubits`",
                    ));
                };
                let py_term = first?.cast::<PySparseTerm>()?.borrow();
                py_term.inner.to_observable()
            }
        };
        for bound_py_term in iter {
            let py_term = bound_py_term?.cast::<PySparseTerm>()?.borrow();
            inner.add_term(py_term.inner.view())?;
        }
        Ok(inner.into())
    }

    // SAFETY: this cannot invoke undefined behaviour if `check = true`, but if `check = false` then
    // the `bit_terms` must all be valid `BitTerm` representations.
    /// Construct a :class:`.SparseObservable` from raw Numpy arrays that match :ref:`the required
    /// data representation described in the class-level documentation <sparse-observable-arrays>`.
    ///
    /// The data from each array is copied into fresh, growable Rust-space allocations.
    ///
    /// Args:
    ///     num_qubits: number of qubits in the observable.
    ///     coeffs: complex coefficients of each term of the observable.  This should be a Numpy
    ///         array with dtype :attr:`~numpy.complex128`.
    ///     bit_terms: flattened list of the single-qubit terms comprising all complete terms.  This
    ///         should be a Numpy array with dtype :attr:`~numpy.uint8` (which is compatible with
    ///         :class:`.BitTerm`).
    ///     indices: flattened term-wise sorted list of the qubits each single-qubit term corresponds
    ///         to.  This should be a Numpy array with dtype :attr:`~numpy.uint32`.
    ///     boundaries: the indices that partition ``bit_terms`` and ``indices`` into terms.  This
    ///         should be a Numpy array with dtype :attr:`~numpy.uintp`.
    ///     check: if ``True`` (the default), validate that the data satisfies all coherence
    ///         guarantees.  If ``False``, no checks are done.
    ///
    ///         .. warning::
    ///
    ///             If ``check=False``, the ``bit_terms`` absolutely *must* be all be valid values
    ///             of :class:`.SparseObservable.BitTerm`.  If they are not, Rust-space undefined
    ///             behavior may occur, entirely invalidating the program execution.
    ///
    /// Examples:
    ///
    ///     Construct a sum of :math:`Z` on each individual qubit::
    ///
    ///         >>> num_qubits = 100
    ///         >>> terms = np.full((num_qubits,), SparseObservable.BitTerm.Z, dtype=np.uint8)
    ///         >>> indices = np.arange(num_qubits, dtype=np.uint32)
    ///         >>> coeffs = np.ones((num_qubits,), dtype=complex)
    ///         >>> boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
    ///         >>> SparseObservable.from_raw_parts(num_qubits, coeffs, terms, indices, boundaries)
    ///         <SparseObservable with 100 terms on 100 qubits: (1+0j)(Z_0) + ... + (1+0j)(Z_99)>
    #[staticmethod]
    #[pyo3(
        signature = (/, num_qubits, coeffs, bit_terms, indices, boundaries, check=true),
    )]
    unsafe fn from_raw_parts<'py>(
        num_qubits: u32,
        coeffs: PyArrayLike1<'py, Complex64, numpy::AllowTypeChange>,
        bit_terms: PyArrayLike1<'py, u8, numpy::AllowTypeChange>,
        indices: PyArrayLike1<'py, u32, numpy::AllowTypeChange>,
        boundaries: PyArrayLike1<'py, usize, numpy::AllowTypeChange>,
        check: bool,
    ) -> PyResult<Self> {
        let coeffs = coeffs.as_array().to_vec();
        let bit_terms = if check {
            bit_terms
                .as_array()
                .into_iter()
                .copied()
                .map(BitTerm::try_from)
                .collect::<Result<_, _>>()?
        } else {
            let bit_terms_as_u8 = bit_terms.as_array().to_vec();
            // SAFETY: the caller enforced that each `u8` is a valid `BitTerm`, and `BitTerm` is be
            // represented by a `u8`.  We can't use `bytemuck` because we're casting a `Vec`.
            unsafe { ::std::mem::transmute::<Vec<u8>, Vec<BitTerm>>(bit_terms_as_u8) }
        };
        let indices = indices.as_array().to_vec();
        let boundaries = boundaries.as_array().to_vec();

        let inner = if check {
            SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)
                .map_err(PyErr::from)
        } else {
            // SAFETY: the caller promised they have upheld the coherence guarantees.
            Ok(unsafe {
                SparseObservable::new_unchecked(num_qubits, coeffs, bit_terms, indices, boundaries)
            })
        }?;
        Ok(inner.into())
    }

    /// Clear all the terms from this operator, making it equal to the zero operator again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction operations may not need to reallocate.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> obs = SparseObservable.from_list([("IX+-rl", 2.0), ("01YZII", -1j)])
    ///         >>> obs.clear()
    ///         >>> assert obs == SparseObservable.zero(obs.py_num_qubits())
    pub fn clear(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.clear();
        Ok(())
    }

    /// Sum any like terms in this operator, removing them if the resulting complex coefficient has
    /// an absolute value within tolerance of zero.
    ///
    /// As a side effect, this sorts the operator into :ref:`canonical order
    /// <sparse-observable-canonical-order>`.
    ///
    /// .. note::
    ///
    ///     When using this for equality comparisons, note that floating-point rounding and the
    ///     non-associativity of floating-point addition may cause non-zero coefficients of summed
    ///     terms to compare unequal.  To compare two observables up to a tolerance, it is safest to
    ///     compare the canonicalized difference of the two observables to zero.
    ///
    /// Args:
    ///     tol (float): after summing like terms, any coefficients whose absolute value is less
    ///         than the given absolute tolerance will be suppressed from the output.
    ///
    /// Examples:
    ///
    ///     Using :meth:`simplify` to compare two operators that represent the same observable, but
    ///     would compare unequal due to the structural tests by default::
    ///
    ///         >>> base = SparseObservable.from_sparse_list([
    ///         ...     ("XZ", (2, 1), 1e-10),  # value too small
    ///         ...     ("+-", (3, 1), 2j),
    ///         ...     ("+-", (3, 1), 2j),     # can be combined with the above
    ///         ...     ("01", (3, 1), 0.5),    # out of order compared to `expected`
    ///         ... ], num_qubits=5)
    ///         >>> expected = SparseObservable.from_list([("I0I1I", 0.5), ("I+I-I", 4j)])
    ///         >>> assert base != expected  # non-canonical comparison
    ///         >>> assert base.simplify() == expected.simplify()
    ///
    ///     Note that in the above example, the coefficients are chosen such that all floating-point
    ///     calculations are exact, and there are no intermediate rounding or associativity
    ///     concerns.  If this cannot be guaranteed to be the case, the safer form is::
    ///
    ///         >>> left = SparseObservable.from_list([("XYZ", 1.0/3.0)] * 3)   # sums to 1.0
    ///         >>> right = SparseObservable.from_list([("XYZ", 1.0/7.0)] * 7)  # doesn't sum to 1.0
    ///         >>> assert left.simplify() != right.simplify()
    ///         >>> assert (left - right).simplify() == SparseObservable.zero(left.num_qubits)
    #[pyo3(
        signature = (/, tol=1e-8),
    )]
    fn simplify(&self, tol: f64) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let simplified = inner.canonicalize(tol);
        Ok(simplified.into())
    }

    /// Calculate the adjoint of this observable.
    ///
    ///
    /// This is well defined in the abstract mathematical sense.  All the terms of the single-qubit
    /// alphabet are self-adjoint, so the result of this operation is the same observable, except
    /// its coefficients are all their complex conjugates.
    ///
    /// Examples:
    ///
    ///     .. code-block::
    ///
    ///         >>> left = SparseObservable.from_list([("XY+-", 1j)])
    ///         >>> right = SparseObservable.from_list([("XY+-", -1j)])
    ///         >>> assert left.adjoint() == right
    fn adjoint(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.adjoint().into())
    }

    /// Calculate the matrix transposition of this observable.
    ///
    /// This operation is defined in terms of the standard matrix conventions of Qiskit, in that the
    /// matrix form is taken to be in the $Z$ computational basis.  The $X$- and $Z$-related
    /// alphabet terms are unaffected by the transposition, but $Y$-related terms modify their
    /// alphabet terms.  Precisely:
    ///
    /// * :math:`Y` transposes to :math:`-Y`
    /// * :math:`\lvert r\rangle\langle r\rvert` transposes to :math:`\lvert l\rangle\langle l\rvert`
    /// * :math:`\lvert l\rangle\langle l\rvert` transposes to :math:`\lvert r\rangle\langle r\rvert`
    ///
    /// Examples:
    ///
    ///     .. code-block::
    ///
    ///         >>> obs = SparseObservable([("III", 1j), ("Yrl", 0.5)])
    ///         >>> assert obs.transpose() == SparseObservable([("III", 1j), ("Ylr", -0.5)])
    fn transpose(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.transpose().into())
    }

    /// Calculate the complex conjugation of this observable.
    ///
    /// This operation is defined in terms of the standard matrix conventions of Qiskit, in that the
    /// matrix form is taken to be in the $Z$ computational basis.  The $X$- and $Z$-related
    /// alphabet terms are unaffected by the complex conjugation, but $Y$-related terms modify their
    /// alphabet terms.  Precisely:
    ///
    /// * :math:`Y` conjguates to :math:`-Y`
    /// * :math:`\lvert r\rangle\langle r\rvert` conjugates to :math:`\lvert l\rangle\langle l\rvert`
    /// * :math:`\lvert l\rangle\langle l\rvert` conjugates to :math:`\lvert r\rangle\langle r\rvert`
    ///
    /// Additionally, all coefficients are conjugated.
    ///
    /// Examples:
    ///
    ///     .. code-block::
    ///
    ///         >>> obs = SparseObservable([("III", 1j), ("Yrl", 0.5)])
    ///         >>> assert obs.conjugate() == SparseObservable([("III", -1j), ("Ylr", -0.5)])
    fn conjugate(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.conjugate().into())
    }

    /// Tensor product of two observables.
    ///
    /// The bit ordering is defined such that the qubit indices of the argument will remain the
    /// same, and the indices of ``self`` will be offset by the number of qubits in ``other``.  This
    /// is the same convention as used by the rest of Qiskit's :mod:`~qiskit.quantum_info`
    /// operators.
    ///
    /// This function is used for the infix ``^`` operator.  If using this operator, beware that
    /// `Python's operator-precedence rules
    /// <https://docs.python.org/3/reference/expressions.html#operator-precedence>`__ may cause the
    /// evaluation order to be different to your expectation.  In particular, the operator ``+``
    /// binds more tightly than ``^``, just like ``*`` binds more tightly than ``+``.  Use
    /// parentheses to fix the evaluation order, if needed.
    ///
    /// The argument will be cast to :class:`SparseObservable` using its default constructor, if it
    /// is not already in the correct form.
    ///
    /// Args:
    ///
    ///     other: the observable to put on the right-hand side of the tensor product.
    ///
    /// Examples:
    ///
    ///     The bit ordering of this is such that the tensor product of two observables made from a
    ///     single label "looks like" an observable made by concatenating the two strings::
    ///
    ///         >>> left = SparseObservable.from_label("XYZ")
    ///         >>> right = SparseObservable.from_label("+-IIrl")
    ///         >>> assert left.tensor(right) == SparseObservable.from_label("XYZ+-IIrl")
    ///
    ///     You can also use the infix ``^`` operator for tensor products, which will similarly cast
    ///     the right-hand side of the operation if it is not already a :class:`SparseObservable`::
    ///
    ///         >>> assert SparseObservable("rl") ^ Pauli("XYZ") == SparseObservable("rlXYZ")
    ///
    /// See also:
    ///     :meth:`expand`
    ///
    ///         The same function, but with the order of arguments flipped.  This can be useful if
    ///         you like using the casting behavior for the argument, but you want your existing
    ///         :class:`SparseObservable` to be on the right-hand side of the tensor ordering.
    #[pyo3(signature = (other, /))]
    fn tensor(&self, other: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Err(PyTypeError::new_err(format!(
                "unknown type for tensor: {}",
                other.get_type().repr()?
            )));
        };

        let other = other.borrow();
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        inner.tensor(&other_inner).into_py_any(py)
    }

    /// Reverse-order tensor product.
    ///
    /// This is equivalent to ``other.tensor(self)``, except that ``other`` will first be type-cast
    /// to :class:`SparseObservable` if it isn't already one (by calling the default constructor).
    ///
    /// Args:
    ///
    ///     other: the observable to put on the left-hand side of the tensor product.
    ///
    /// Examples:
    ///
    ///     This is equivalent to :meth:`tensor` with the order of the arguments flipped::
    ///
    ///         >>> left = SparseObservable.from_label("XYZ")
    ///         >>> right = SparseObservable.from_label("+-IIrl")
    ///         >>> assert left.tensor(right) == right.expand(left)
    ///
    /// See also:
    ///     :meth:`tensor`
    ///
    ///         The same function with the order of arguments flipped.  :meth:`tensor` is the more
    ///         standard argument ordering, and matches Qiskit's other conventions.
    #[pyo3(signature = (other, /))]
    fn expand<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PySparseObservable>> {
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Err(PyTypeError::new_err(format!(
                "unknown type for expand: {}",
                other.get_type().repr()?
            )));
        };

        let other = other.borrow();
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        other_inner.tensor(&inner).into_pyobject(py)
    }

    /// Compose another :class:`SparseObservable` onto this one.
    ///
    /// In terms of operator algebras, composition corresponds to left-multiplication:
    /// ``c = a.compose(b)`` corresponds to $C = B A$.  In other words, ``a.compose(b)`` returns an
    /// operator that "performs ``a``, and then performs ``b`` on the result".  The argument
    /// ``front=True`` instead makes this a right multiplication.
    ///
    /// ``self`` and ``other`` must be the same size, unless ``qargs`` is given, in which case
    /// ``other`` can be smaller than ``self``, provided the number of qubits in ``other`` and the
    /// length of ``qargs`` match.  ``qargs`` can never contain duplicates or indices of qubits that
    /// do not exist in ``self``.
    ///
    /// Beware that this function can cause exponential explosion of the memory usage of the
    /// observable, as the alphabet of :class:`SparseObservable` is not closed under composition;
    /// the composition of two single-bit terms can be a sum, which multiplies the total number of
    /// terms.  This memory usage is not _necessarily_ inherent to the resultant observable, but
    /// finding an efficient re-factorization of the sum is generally equally computationally hard.
    /// It's better to use domain knowledge of your observables to minimize the number of terms that
    /// ever exist, rather than trying to simplify them after the fact.
    ///
    /// Args:
    ///     other: the observable used to left-multiply ``self``.
    ///     qargs: if given, the qubits in ``self`` to be associated with the qubits in ``other``.
    ///         Put another way: if this is given, it is similar to a more efficient implementation
    ///         of::
    ///
    ///             self.compose(other.apply_layout(qargs, self.num_qubits))
    ///
    ///         as no temporary observable is created to store the applied-layout form of ``other``.
    ///     front: if ``True``, then right-multiply by ``other`` instead of left-multiplying
    ///         (default ``False``).  The ``qargs`` are still applied to ``other``.  This is most
    ///         useful when ``qargs`` is set, or ``other`` might be an object that must be coerced
    ///         to :class:`SparseObservable`.
    #[pyo3(signature = (other, /, qargs=None, *, front=false))]
    fn compose<'py>(
        &self,
        other: &Bound<'py, PyAny>,
        qargs: Option<&Bound<'py, PyAny>>,
        front: bool,
    ) -> PyResult<Bound<'py, PySparseObservable>> {
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Err(PyTypeError::new_err(format!(
                "unknown type for compose: {}",
                other.get_type().repr()?
            )));
        };
        let other = other.borrow();
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        if let Some(order) = qargs {
            if other_inner.num_qubits() > inner.num_qubits() {
                return Err(PyValueError::new_err(format!(
                    "argument has more qubits ({}) than the base ({})",
                    other_inner.num_qubits(),
                    inner.num_qubits()
                )));
            }
            let in_length = order.len()?;
            if other_inner.num_qubits() as usize != in_length {
                return Err(PyValueError::new_err(format!(
                    "input qargs has length {}, but the observable is on {} qubit(s)",
                    in_length,
                    other_inner.num_qubits()
                )));
            }
            let order = order
                .try_iter()?
                .map(|obj| obj.and_then(|obj| obj.extract::<u32>()))
                .collect::<PyResult<IndexSet<u32>>>()?;
            if order.len() != in_length {
                return Err(PyValueError::new_err("duplicate indices in qargs"));
            }
            if order
                .iter()
                .max()
                .is_some_and(|max| *max >= inner.num_qubits())
            {
                return Err(PyValueError::new_err("qargs contains out-of-range qubits"));
            }
            if front {
                // This implementation can be improved if it turns out to be needed a lot and the
                // extra copy is a bottleneck.
                let other_to_self = order.iter().copied().collect::<Vec<_>>();
                other_inner
                    .apply_layout(Some(other_to_self.as_slice()), inner.num_qubits)?
                    .compose(&inner)
                    .into_pyobject(py)
            } else {
                inner
                    .compose_map(&other_inner, |bit| {
                        *order
                            .get_index(bit as usize)
                            .expect("order has the same length and no duplicates")
                    })
                    .into_pyobject(py)
            }
        } else {
            if other_inner.num_qubits() != inner.num_qubits() {
                return Err(PyValueError::new_err(format!(
                    "mismatched numbers of qubits: {} (base) and {} (argument)",
                    inner.num_qubits(),
                    other_inner.num_qubits()
                )));
            }
            if front {
                other_inner.compose(&inner).into_pyobject(py)
            } else {
                inner.compose(&other_inner).into_pyobject(py)
            }
        }
    }

    /// Evolve this observable by a Pauli term.
    ///
    /// An evolution of the observable :math:`O` by a Pauli :math:`P` corresponds to
    /// :math:`P^\dagger O P`.
    ///
    /// Unlike a literal implementation via two full compositions, this method
    /// performs the conjugation directly at the single-qubit level using a fixed
    /// lookup table.  This avoids materializing any intermediate
    /// :class:`SparseObservable` and computes the evolved observable in a single
    /// pass over the terms.
    /// ``self`` and ``other`` must have the same number of qubits, unless ``qargs`` is given,
    /// in which case ``other`` can be smaller than ``self``, provided the number of qubits
    /// in ``other`` and the length of ``qargs`` match. ``qargs`` specifies which qubits of
    /// ``self`` are evolved by ``other``.
    ///
    /// Currently, this method supports evolution only by a *single-term* operator, meaning
    /// that ``other`` must be a Pauli represented by :class:`~.quantum_info.Pauli`.
    ///
    /// Args:
    ///     other: the Pauli Operator used to conjugate ``self``.
    ///     qargs: if given, the qubits in ``self`` to be evolved by ``other``.
    ///         The length must match the number of qubits in ``other``.
    ///
    /// Returns:
    ///      A new evolved :class:`SparseObservable` with applied conjugations.
    ///
    /// Raises:
    ///     TypeError : if ``other`` is not of Type :class:`~.quantum_info.Pauli`.
    ///     ValueError: if ``self`` and ``other`` have different numbers of qubits (and ``qargs`` is not given).
    ///     ValueError: if ``qargs`` length doesn't match ``other`` number of qubits.
    ///     ValueError: if ``qargs`` contains duplicates or out-of-range indices.
    ///     ValueError: if ``other`` contains more than one term.
    #[pyo3(signature = (other, /, qargs=None))]
    fn evolve<'py>(
        &self,
        other: &Bound<'py, PyAny>,
        qargs: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PySparseObservable>> {
        let py = other.py();

        if !other.is_instance(PAULI_TYPE.get_bound(py))? {
            return Err(PyTypeError::new_err(format!(
                "evolve only accepts Pauli instances, got: {}",
                other.get_type().repr()?
            )));
        }

        let base = self.as_inner()?;
        let u_obs = Self::from_pauli(other)?;
        let u_inner = u_obs.as_inner()?;

        let qargs_vec = if let Some(qargs) = qargs {
            let vec = qargs
                .try_iter()?
                .map(|obj| obj.and_then(|obj| obj.extract::<u32>()))
                .collect::<PyResult<Vec<u32>>>()?;
            Some(vec)
        } else {
            None
        };

        let out = base
            .evolve(&u_inner, qargs_vec.as_deref())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        out.into_pyobject(py)
    }

    /// Apply a transpiler layout to this :class:`SparseObservable`.
    ///
    /// Typically you will have defined your observable in terms of the virtual qubits of the
    /// circuits you will use to prepare states.  After transpilation, the virtual qubits are mapped
    /// to particular physical qubits on a device, which may be wider than your circuit.  That
    /// mapping can also change over the course of the circuit.  This method transforms the input
    /// observable on virtual qubits to an observable that is suitable to apply immediately after
    /// the fully transpiled *physical* circuit.
    ///
    /// Args:
    ///     layout (TranspileLayout | list[int] | None): The layout to apply.  Most uses of this
    ///         function should pass the :attr:`.QuantumCircuit.layout` field from a circuit that
    ///         was transpiled for hardware.  In addition, you can pass a list of new qubit indices.
    ///         If given as explicitly ``None``, no remapping is applied (but you can still use
    ///         ``num_qubits`` to expand the observable).
    ///     num_qubits (int | None): The number of qubits to expand the observable to.  If not
    ///         supplied, the output will be as wide as the given :class:`.TranspileLayout`, or the
    ///         same width as the input if the ``layout`` is given in another form.
    ///
    /// Returns:
    ///     A new :class:`SparseObservable` with the provided layout applied.
    #[pyo3(signature = (/, layout, num_qubits=None))]
    fn apply_layout(&self, layout: Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let py = layout.py();
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        // A utility to check the number of qubits is compatible with the observable.
        let check_inferred_qubits = |inferred: u32| -> PyResult<u32> {
            if inferred < inner.num_qubits() {
                return Err(CoherenceError::NotEnoughQubits {
                    current: inner.num_qubits() as usize,
                    target: inferred as usize,
                }
                .into());
            }
            Ok(inferred)
        };

        // Normalize the number of qubits in the layout and the layout itself, depending on the
        // input types, before calling SparseObservable.apply_layout to do the actual work.
        let (num_qubits, layout): (u32, Option<Vec<u32>>) = if layout.is_none() {
            (num_qubits.unwrap_or(inner.num_qubits()), None)
        } else if layout.is_instance(
            &py.import(intern!(py, "qiskit.transpiler"))?
                .getattr(intern!(py, "TranspileLayout"))?,
        )? {
            (
                check_inferred_qubits(
                    layout.getattr(intern!(py, "_output_qubit_list"))?.len()? as u32
                )?,
                Some(
                    layout
                        .call_method0(intern!(py, "final_index_layout"))?
                        .extract::<Vec<u32>>()?,
                ),
            )
        } else {
            (
                check_inferred_qubits(num_qubits.unwrap_or(inner.num_qubits()))?,
                Some(layout.extract()?),
            )
        };

        let out = inner.apply_layout(layout.as_deref(), num_qubits)?;
        Ok(out.into())
    }

    /// Get a :class:`.PauliList` object that represents the measurement basis needed for each term
    /// (in order) in this observable.
    ///
    /// For example, the projector ``0l+`` will return a Pauli ``ZXY``.  The resulting
    /// :class:`~.quantum_info.Pauli` is dense, in the sense that explicit identities are stored.
    /// An identity in the Pauli output does not require a concrete measurement.
    ///
    /// This will return an entry in the Pauli list for every term in the sum.
    ///
    /// Returns:
    ///     :class:`.PauliList`: the Pauli operator list representing the necessary measurement
    ///     bases.
    fn pauli_bases<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let mut x = Array2::from_elem([inner.num_terms(), inner.num_qubits() as usize], false);
        let mut z = Array2::from_elem([inner.num_terms(), inner.num_qubits() as usize], false);
        for (loc, term) in inner.iter().enumerate() {
            let mut x_row = x.row_mut(loc);
            let mut z_row = z.row_mut(loc);
            for (bit_term, index) in term.bit_terms.iter().zip(term.indices) {
                x_row[*index as usize] = bit_term.has_x_component();
                z_row[*index as usize] = bit_term.has_z_component();
            }
        }
        PAULI_LIST_TYPE
            .get_bound(py)
            .getattr(intern!(py, "from_symplectic"))?
            .call1((
                PyArray2::from_owned_array(py, z),
                PyArray2::from_owned_array(py, x),
            ))
    }

    /// Check whether the observable commutes with another one.
    ///
    /// Args:
    ///     other (SparseObservable): The other observable to check commutation with.
    ///     tol (float): If coefficients in the product of `self` and `other` are below the
    ///         tolerance (in magnitude), the terms are ignored.
    ///
    /// Returns:
    ///     ``True`` if the terms commute, up to tolerance, ``False`` otherwise.
    ///
    /// Raises:
    ///     TypeError: If ``other`` could not be coerced to ``SparseObservable``.
    #[pyo3(signature = (other, tol=1e-12))]
    pub fn commutes(&self, other: &Bound<PyAny>, tol: f64) -> PyResult<bool> {
        let Some(other) = coerce_to_observable(other)? else {
            return Err(PyTypeError::new_err("Invalid type of other."));
        };

        let self_inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other = other.borrow();
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;

        Ok(self_inner.commutes(&other_inner, tol))
    }

    fn __len__(&self) -> PyResult<usize> {
        self.num_terms()
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: PySequenceIndex<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let indices = match index.with_len(inner.num_terms())? {
            SequenceIndex::Int(index) => {
                return PySparseTerm {
                    inner: inner.term(index).to_term(),
                }
                .into_bound_py_any(py);
            }
            indices => indices,
        };
        let mut out = SparseObservable::zero(inner.num_qubits());
        for index in indices.iter() {
            out.add_term(inner.term(index))?;
        }
        out.into_bound_py_any(py)
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> PyResult<bool> {
        // this is also important to check before trying to read both slf and other
        if slf.is(&other) {
            return Ok(true);
        }
        let Ok(other) = other.cast_into::<Self>() else {
            return Ok(false);
        };
        let slf_borrowed = slf.borrow();
        let other_borrowed = other.borrow();
        let slf_inner = slf_borrowed.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other_borrowed.inner.read().map_err(|_| InnerReadError)?;
        Ok(slf_inner.eq(&other_inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        let num_terms = self.num_terms()?;
        let num_qubits = self.num_qubits()?;

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

        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let str_terms = if num_terms == 0 {
            "0.0".to_owned()
        } else {
            inner
                .iter()
                .map(SparseTermView::to_sparse_str)
                .collect::<Vec<_>>()
                .join(" + ")
        };
        Ok(format!(
            "<SparseObservable with {str_num_terms} on {str_num_qubits}: {str_terms}>"
        ))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let bit_terms: &[u8] = ::bytemuck::cast_slice(inner.bit_terms());
        (
            py.get_type::<Self>().getattr("from_raw_parts")?,
            (
                inner.num_qubits(),
                PyArray1::from_slice(py, inner.coeffs()),
                PyArray1::from_slice(py, bit_terms),
                PyArray1::from_slice(py, inner.indices()),
                PyArray1::from_slice(py, inner.boundaries()),
                false,
            ),
        )
            .into_pyobject(py)
    }

    fn __add__<'py>(
        slf_: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf_.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented().into_bound(py));
        };

        let other = other.borrow();
        let slf_ = slf_.borrow();
        if Arc::ptr_eq(&slf_.inner, &other.inner) {
            // This fast path is for consistency with the in-place `__iadd__`, which would otherwise
            // struggle to do the addition to itself.
            let inner = slf_.inner.read().map_err(|_| InnerReadError)?;
            return <&SparseObservable as ::std::ops::Mul<_>>::mul(
                &inner,
                Complex64::new(2.0, 0.0),
            )
            .into_bound_py_any(py);
        }
        let slf_inner = slf_.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        slf_inner.check_equal_qubits(&other_inner)?;
        <&SparseObservable as ::std::ops::Add>::add(&slf_inner, &other_inner).into_bound_py_any(py)
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        // No need to handle the `self is other` case here, because `__add__` will get it.
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented().into_bound(py));
        };

        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other = other.borrow();
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        inner.check_equal_qubits(&other_inner)?;
        <&SparseObservable as ::std::ops::Add>::add(&other_inner, &inner).into_bound_py_any(py)
    }

    fn __iadd__(slf_: Bound<PySparseObservable>, other: &Bound<PyAny>) -> PyResult<()> {
        let Some(other) = coerce_to_observable(other)? else {
            // This is not well behaved - we _should_ return `NotImplemented` to Python space
            // without an exception, but limitations in PyO3 prevent this at the moment.  See
            // https://github.com/PyO3/pyo3/issues/4605.
            return Err(PyTypeError::new_err(format!(
                "invalid object for in-place addition of 'SparseObservable': {}",
                other.repr()?
            )));
        };

        let other = other.borrow();
        let slf_ = slf_.borrow();
        let mut slf_inner = slf_.inner.write().map_err(|_| InnerWriteError)?;

        // Check if slf_ and other point to the same SparseObservable object, in which case
        // we just multiply it by 2
        if Arc::ptr_eq(&slf_.inner, &other.inner) {
            *slf_inner *= Complex64::new(2.0, 0.0);
            return Ok(());
        }

        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        slf_inner.check_equal_qubits(&other_inner)?;
        slf_inner.add_assign(&other_inner);
        Ok(())
    }

    fn __sub__<'py>(
        slf_: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf_.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented().into_bound(py));
        };

        let other = other.borrow();
        let slf_ = slf_.borrow();
        if Arc::ptr_eq(&slf_.inner, &other.inner) {
            return PySparseObservable::zero(slf_.num_qubits()?).into_bound_py_any(py);
        }

        let slf_inner = slf_.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        slf_inner.check_equal_qubits(&other_inner)?;
        <&SparseObservable as ::std::ops::Sub>::sub(&slf_inner, &other_inner).into_bound_py_any(py)
    }

    fn __rsub__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented().into_bound(py));
        };
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other = other.borrow();
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        inner.check_equal_qubits(&other_inner)?;
        <&SparseObservable as ::std::ops::Sub>::sub(&other_inner, &inner).into_bound_py_any(py)
    }

    fn __isub__(slf_: Bound<PySparseObservable>, other: &Bound<PyAny>) -> PyResult<()> {
        let Some(other) = coerce_to_observable(other)? else {
            // This is not well behaved - we _should_ return `NotImplemented` to Python space
            // without an exception, but limitations in PyO3 prevent this at the moment.  See
            // https://github.com/PyO3/pyo3/issues/4605.
            return Err(PyTypeError::new_err(format!(
                "invalid object for in-place subtraction of 'SparseObservable': {}",
                other.repr()?
            )));
        };
        let other = other.borrow();
        let slf_ = slf_.borrow();
        let mut slf_inner = slf_.inner.write().map_err(|_| InnerWriteError)?;

        if Arc::ptr_eq(&slf_.inner, &other.inner) {
            // This is not strictly the same thing as `a - a` if `a` contains non-finite
            // floating-point values (`inf - inf` is `NaN`, for example); we don't really have a
            // clear view on what floating-point guarantees we're going to make right now.
            slf_inner.clear();
            return Ok(());
        }

        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        slf_inner.check_equal_qubits(&other_inner)?;
        slf_inner.sub_assign(&other_inner);
        Ok(())
    }

    fn __pos__(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.clone().into())
    }

    fn __neg__(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let neg = <&SparseObservable as ::std::ops::Neg>::neg(&inner);
        Ok(neg.into())
    }

    fn __mul__(&self, other: Complex64) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let mult = <&SparseObservable as ::std::ops::Mul<_>>::mul(&inner, other);
        Ok(mult.into())
    }
    fn __rmul__(&self, other: Complex64) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __imul__(&mut self, other: Complex64) -> PyResult<()> {
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.mul_assign(other);
        Ok(())
    }

    fn __truediv__(&self, other: Complex64) -> PyResult<Self> {
        if other.is_zero() {
            return Err(PyZeroDivisionError::new_err("complex division by zero"));
        }
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let div = <&SparseObservable as ::std::ops::Div<_>>::div(&inner, other);
        Ok(div.into())
    }
    fn __itruediv__(&mut self, other: Complex64) -> PyResult<()> {
        if other.is_zero() {
            return Err(PyZeroDivisionError::new_err("complex division by zero"));
        }
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.div_assign(other);
        Ok(())
    }

    fn __xor__(&self, other: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        // we cannot just delegate this to ``tensor`` since ``other`` might allow
        // right-hand-side arithmetic and we have to try deferring to that object,
        // which is done by returning ``NotImplemented``
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented());
        };

        self.tensor(&other)
    }

    fn __rxor__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = other.py();
        let Some(other) = coerce_to_observable(other)? else {
            return Ok(py.NotImplemented().into_bound(py));
        };
        self.expand(&other).map(|obj| obj.into_any())
    }

    // The documentation for this is inlined into the class-level documentation of
    // `SparseObservable`.
    #[allow(non_snake_case)]
    #[classattr]
    fn BitTerm(py: Python) -> PyResult<Py<PyType>> {
        BIT_TERM_PY_ENUM
            .get_or_try_init(py, || make_py_bit_term(py))
            .map(|obj| obj.clone_ref(py))
    }

    // The documentation for this is inlined into the class-level documentation of
    // `SparseObservable`.
    #[allow(non_snake_case)]
    #[classattr]
    fn Term(py: Python) -> Bound<PyType> {
        py.get_type::<PySparseTerm>()
    }
}

#[cfg(feature = "python")]
impl PySparseObservable {
    /// This is an immutable reference as opposed to a `copy`.
    pub fn as_inner(&self) -> Result<RwLockReadGuard<'_, SparseObservable>, InnerReadError> {
        let data = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(data)
    }
}

#[cfg(feature = "python")]
impl From<SparseObservable> for PySparseObservable {
    fn from(val: SparseObservable) -> PySparseObservable {
        PySparseObservable {
            inner: Arc::new(RwLock::new(val)),
        }
    }
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for SparseObservable {
    type Target = PySparseObservable;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PySparseObservable::from(self).into_pyobject(py)
    }
}

/// Helper class of `ArrayView` that denotes the slot of the `SparseObservable` we're looking at.
#[cfg(feature = "python")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArraySlot {
    Coeffs,
    BitTerms,
    Indices,
    Boundaries,
}

/// Custom wrapper sequence class to get safe views onto the Rust-space data.  We can't directly
/// expose Python-managed wrapped pointers without introducing some form of runtime exclusion on the
/// ability of `SparseObservable` to re-allocate in place; we can't leave dangling pointers for
/// Python space.
#[cfg(feature = "python")]
#[pyclass(frozen, sequence)]
struct ArrayView {
    base: Arc<RwLock<SparseObservable>>,
    slot: ArraySlot,
}

#[cfg(feature = "python")]
#[pymethods]
impl ArrayView {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let obs = self.base.read().map_err(|_| InnerReadError)?;
        let data = match self.slot {
            // Simple integers look the same in Rust-space debug as Python.
            ArraySlot::Indices => format!("{:?}", obs.indices()),
            ArraySlot::Boundaries => format!("{:?}", obs.boundaries()),
            // Complexes don't have a nice repr in Rust, so just delegate the whole load to Python
            // and convert back.
            ArraySlot::Coeffs => PyList::new(py, obs.coeffs())?.repr()?.to_string(),
            ArraySlot::BitTerms => format!(
                "[{}]",
                obs.bit_terms()
                    .iter()
                    .map(BitTerm::py_label)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        Ok(format!(
            "<observable {} view: {}>",
            match self.slot {
                ArraySlot::Coeffs => "coeffs",
                ArraySlot::BitTerms => "bit_terms",
                ArraySlot::Indices => "indices",
                ArraySlot::Boundaries => "boundaries",
            },
            data,
        ))
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: PySequenceIndex,
    ) -> PyResult<Bound<'py, PyAny>> {
        // The slightly verbose generic setup here is to allow the type of a scalar return to be
        // different to the type that gets put into the Numpy array, since the `BitTerm` enum can be
        // a direct scalar, but for Numpy, we need it to be a raw `u8`.
        fn get_from_slice<'py, T, S>(
            py: Python<'py>,
            slice: &[T],
            index: PySequenceIndex,
        ) -> PyResult<Bound<'py, PyAny>>
        where
            T: IntoPyObject<'py> + Copy + Into<S>,
            S: ::numpy::Element,
        {
            match index.with_len(slice.len())? {
                SequenceIndex::Int(index) => slice[index].into_bound_py_any(py),
                indices => PyArray1::from_iter(py, indices.iter().map(|index| slice[index].into()))
                    .into_bound_py_any(py),
            }
        }

        let obs = self.base.read().map_err(|_| InnerReadError)?;
        match self.slot {
            ArraySlot::Coeffs => get_from_slice::<_, Complex64>(py, obs.coeffs(), index),
            ArraySlot::BitTerms => get_from_slice::<_, u8>(py, obs.bit_terms(), index),
            ArraySlot::Indices => get_from_slice::<_, u32>(py, obs.indices(), index),
            ArraySlot::Boundaries => get_from_slice::<_, usize>(py, obs.boundaries(), index),
        }
    }

    fn __setitem__(&self, index: PySequenceIndex, values: &Bound<PyAny>) -> PyResult<()> {
        /// Set values of a slice according to the indexer, using `extract` to retrieve the
        /// Rust-space object from the collection of Python-space values.
        ///
        /// This indirects the Python extraction through an intermediate type to marginally improve
        /// the error messages for things like `BitTerm`, where Python-space extraction might fail
        /// because the user supplied an invalid alphabet letter.
        ///
        /// This allows broadcasting a single item into many locations in a slice (like Numpy), but
        /// otherwise requires that the index and values are the same length (unlike Python's
        /// `list`) because that would change the length.
        fn set_in_slice<'a, 'py, T, S>(
            slice: &mut [T],
            index: PySequenceIndex<'py>,
            values: Borrowed<'a, 'py, PyAny>,
        ) -> PyResult<()>
        where
            T: Copy + TryFrom<S>,
            S: for<'b> FromPyObject<'b, 'py, Error: Into<PyErr>>,
            PyErr: From<<T as TryFrom<S>>::Error>,
        {
            match index.with_len(slice.len())? {
                SequenceIndex::Int(index) => {
                    slice[index] = values
                        .extract::<S>()
                        .map_err(Into::<PyErr>::into)?
                        .try_into()?;
                    Ok(())
                }
                indices => {
                    if let Ok(value) = values.extract::<S>() {
                        let value = value.try_into()?;
                        for index in indices {
                            slice[index] = value;
                        }
                    } else {
                        let values = values
                            .try_iter()?
                            .map(|value| {
                                value?
                                    .extract::<S>()
                                    .map_err(Into::<PyErr>::into)?
                                    .try_into()
                                    .map_err(PyErr::from)
                            })
                            .collect::<PyResult<Vec<_>>>()?;
                        if indices.len() != values.len() {
                            return Err(PyValueError::new_err(format!(
                                "tried to set a slice of length {} with a sequence of length {}",
                                indices.len(),
                                values.len(),
                            )));
                        }
                        for (index, value) in indices.into_iter().zip(values) {
                            slice[index] = value;
                        }
                    }
                    Ok(())
                }
            }
        }

        let mut obs = self.base.write().map_err(|_| InnerWriteError)?;
        let values = values.as_borrowed();
        match self.slot {
            ArraySlot::Coeffs => set_in_slice::<_, Complex64>(obs.coeffs_mut(), index, values),
            ArraySlot::BitTerms => set_in_slice::<BitTerm, u8>(obs.bit_terms_mut(), index, values),
            ArraySlot::Indices => unsafe {
                set_in_slice::<_, u32>(obs.indices_mut(), index, values)
            },
            ArraySlot::Boundaries => unsafe {
                set_in_slice::<_, usize>(obs.boundaries_mut(), index, values)
            },
        }
    }

    fn __len__(&self, _py: Python) -> PyResult<usize> {
        let obs = self.base.read().map_err(|_| InnerReadError)?;
        let len = match self.slot {
            ArraySlot::Coeffs => obs.coeffs().len(),
            ArraySlot::BitTerms => obs.bit_terms().len(),
            ArraySlot::Indices => obs.indices().len(),
            ArraySlot::Boundaries => obs.boundaries().len(),
        };
        Ok(len)
    }

    #[pyo3(signature = (/, dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // This method always copies, so we don't leave dangling pointers lying around in Numpy
        // arrays; it's not enough just to set the `base` of the Numpy array to the
        // `SparseObservable`, since the `Vec` we're referring to might re-allocate and invalidate
        // the pointer the Numpy array is wrapping.
        if !copy.unwrap_or(true) {
            return Err(PyValueError::new_err(
                "cannot produce a safe view onto movable memory",
            ));
        }
        let obs = self.base.read().map_err(|_| InnerReadError)?;
        match self.slot {
            ArraySlot::Coeffs => cast_array_type(py, PyArray1::from_slice(py, obs.coeffs()), dtype),
            ArraySlot::Indices => {
                cast_array_type(py, PyArray1::from_slice(py, obs.indices()), dtype)
            }
            ArraySlot::Boundaries => {
                cast_array_type(py, PyArray1::from_slice(py, obs.boundaries()), dtype)
            }
            ArraySlot::BitTerms => {
                let bit_terms: &[u8] = ::bytemuck::cast_slice(obs.bit_terms());
                cast_array_type(py, PyArray1::from_slice(py, bit_terms), dtype)
            }
        }
    }
}

/// Use the Numpy Python API to convert a `PyArray` into a dynamically chosen `dtype`, copying only
/// if required.
#[cfg(feature = "python")]
fn cast_array_type<'py, T: numpy::Element>(
    py: Python<'py>,
    array: Bound<'py, PyArray1<T>>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let base_dtype = array.dtype();
    let dtype = dtype
        .map(|dtype| PyArrayDescr::new(py, dtype))
        .unwrap_or_else(|| Ok(base_dtype.clone()))?;
    if dtype.is_equiv_to(&base_dtype) {
        return Ok(array.into_any());
    }
    PyModule::import(py, intern!(py, "numpy"))?
        .getattr(intern!(py, "array"))?
        .call(
            (array,),
            Some(&[(intern!(py, "dtype"), dtype.as_any())].into_py_dict(py)?),
        )
}

/// Attempt to coerce an arbitrary Python object to a [PySparseObservable].
///
/// This returns:
///
/// * `Ok(Some(obs))` if the coercion was completely successful.
/// * `Ok(None)` if the input value was just completely the wrong type and no coercion could be
///   attempted.
/// * `Err` if the input was a valid type for coercion, but the coercion failed with a Python
///   exception.
///
/// The purpose of this is for conversion the arithmetic operations, which should return
/// [PyNotImplemented] if the type is not valid for coercion.
#[cfg(feature = "python")]
fn coerce_to_observable<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PySparseObservable>>> {
    let py = value.py();
    if let Ok(obs) = value.cast_exact::<PySparseObservable>() {
        return Ok(Some(obs.clone()));
    }
    match PySparseObservable::py_new(value, None) {
        Ok(obs) => Ok(Some(Bound::new(py, obs)?)),
        Err(e) => {
            if e.is_instance_of::<PyTypeError>(py) {
                Ok(None)
            } else {
                Err(e)
            }
        }
    }
}

#[cfg(feature = "python")]
pub fn sparse_observable(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PySparseObservable>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_error_safe_add_dense_label() {
        let base = SparseObservable::identity(5);
        let mut modified = base.clone();
        assert!(matches!(
            modified.add_dense_label("IIZ$X", Complex64::new(1.0, 0.0)),
            Err(LabelError::OutsideAlphabet)
        ));
        // `modified` should have been left in a valid state.
        assert_eq!(base, modified);
    }
}
