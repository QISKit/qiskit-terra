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

use hashbrown::HashSet;

use ndarray::Array2;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::{
    IntoPyObjectExt, PyErr,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    intern,
    prelude::*,
    sync::PyOnceLock,
    types::{IntoPyDict, PyList, PyString, PyTuple, PyType},
};
use std::collections::btree_map;

#[cfg(feature = "python")]
use std::{
    iter::zip,
    sync::{Arc, RwLock},
};
use thiserror::Error;

#[cfg(feature = "python")]
use qiskit_util::py::{PySequenceIndex, SequenceIndex};

#[cfg(feature = "python")]
use crate::imports;

#[cfg(feature = "python")]
static PAULI_PY_ENUM: PyOnceLock<Py<PyType>> = PyOnceLock::new();
#[cfg(feature = "python")]
static PAULI_INTO_PY: PyOnceLock<[Option<Py<PyAny>>; 16]> = PyOnceLock::new();

/// Named handle to the alphabet of single-qubit terms.
///
/// This is just the Rust-space representation.  We make a separate Python-space `enum.IntEnum` to
/// represent the same information, since we enforce strongly typed interactions in Rust, including
/// not allowing the stored values to be outside the valid `Pauli`\ s, but doing so in Python would
/// make it very difficult to use the class efficiently with Numpy array views.  We attach this
/// sister class of `Pauli` to `QubitSparsePauli` and `QubitSparsePauliList` as a scoped class.
///
/// # Representation
///
/// The `u8` representation and the exact numerical values of these are part of the public API.  The
/// two bits are the symplectic Pauli representation of the Pauli operator, with the associations
/// `0b10` <-> `X`, `0b01` <-> `Z`, `0b11` <-> `Y`. The `0b00` representation thus ends up being the
/// natural representation of the `I` operator, but this is never stored, and is not named in the
/// enumeration.
///
/// This operator does not store phase terms of $-i$.  `Pauli::Y` has `(1, 1)` as its `(z, x)`
/// representation, and represents exactly the Pauli Y operator. Additional phases, if needed, must
/// be stored elsewhere.
///
/// # Dev notes
///
/// This type is required to be `u8`, but it's a subtype of `u8` because not all `u8` are valid
/// `Pauli`\ s.  For interop with Python space, we accept Numpy arrays of `u8` to represent this,
/// which we transmute into slices of `Pauli`, after checking that all the values are correct (or
/// skipping the check if Python space promises that it upheld the checks).
///
/// We deliberately _don't_ impl `numpy::Element` for `Pauli` (which would let us accept and
/// return `PyArray1<Pauli>` at Python-space boundaries) so that it's clear when we're doing
/// the transmute, and we have to be explicit about the safety of that.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pauli {
    /// Pauli X operator.
    X = 0b10,
    /// Pauli Y operator.
    Y = 0b11,
    /// Pauli Z operator.
    Z = 0b01,
}
impl From<Pauli> for u8 {
    fn from(value: Pauli) -> u8 {
        value as u8
    }
}
unsafe impl ::bytemuck::CheckedBitPattern for Pauli {
    type Bits = u8;

    #[inline(always)]
    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits <= 0b11 && *bits != 0
    }
}
unsafe impl ::bytemuck::NoUninit for Pauli {}

impl Pauli {
    /// Get the label of this `Pauli` used in Python-space applications.  This is a single-letter
    /// string.
    #[inline]
    pub fn py_label(&self) -> &'static str {
        // Note: these labels are part of the stable Python API and should not be changed.
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }

    /// Attempt to convert a `u8` into `Pauli`.
    ///
    /// Unlike the implementation of `TryFrom<u8>`, this allows `b'I'` as an alphabet letter,
    /// returning `Ok(None)` for it.  All other letters outside the alphabet return the complete
    /// error condition.
    #[inline]
    pub fn try_from_u8(value: u8) -> Result<Option<Self>, PauliFromU8Error> {
        match value {
            b'I' => Ok(None),
            b'X' => Ok(Some(Pauli::X)),
            b'Y' => Ok(Some(Pauli::Y)),
            b'Z' => Ok(Some(Pauli::Z)),
            _ => Err(PauliFromU8Error(value)),
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
}

/// The error type for a failed conversion into `Pauli`.
#[derive(Error, Debug)]
#[error("{0} is not a valid letter of the single-qubit alphabet")]
pub struct PauliFromU8Error(u8);

// `Pauli` allows safe `as` casting into `u8`.  This is the reverse, which is fallible, because
// `Pauli` is a value-wise subtype of `u8`.
impl ::std::convert::TryFrom<u8> for Pauli {
    type Error = PauliFromU8Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        ::bytemuck::checked::try_cast(value).map_err(|_| PauliFromU8Error(value))
    }
}

/// Error cases stemming from data coherence at the point of entry into `QubitSparsePauli` or
/// `QubitSparsePauliList` from user-provided arrays.
///
/// These most typically appear during [from_raw_parts], but can also be introduced by various
/// remapping arithmetic functions.
///
/// These are generally associated with the Python-space `ValueError` because all of the
/// `TypeError`-related ones are statically forbidden (within Rust) by the language, and conversion
/// failures on entry to Rust from Python space will automatically raise `TypeError`.
#[derive(Error, Debug)]
pub enum CoherenceError {
    #[error("`phases` ({phases}) must be the same length as `qubit_sparse_pauli_list` ({qspl})")]
    MismatchedPhaseCount { phases: usize, qspl: usize },
    #[error("`rates` ({rates}) must be the same length as `qubit_sparse_pauli_list` ({qspl})")]
    MismatchedTermCount { rates: usize, qspl: usize },
    #[error("`paulis` ({paulis}) and `indices` ({indices}) must be the same length")]
    MismatchedItemCount { paulis: usize, indices: usize },
    #[error("the first item of `boundaries` ({0}) must be 0")]
    BadInitialBoundary(usize),
    #[error(
        "the last item of `boundaries` ({last}) must match the length of `paulis` and `indices` ({items})"
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
    #[error("cannot shrink the qubit count in a QubitSparsePauliList from {current} to {target}")]
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
    #[error("labels must only contain letters from the alphabet 'IXYZ'")]
    OutsideAlphabet,
}

#[derive(Error, Debug)]
pub enum ArithmeticError {
    #[error("mismatched numbers of qubits: {left}, {right}")]
    MismatchedQubits { left: u32, right: u32 },
    #[error("multiplying single qubit paulis resulted in bit value: {b}")]
    PauliMultiplication { b: u8 },
}

/// A list of Pauli operators stored in a qubit-sparse format.
///
/// See [PyQubitSparsePauliList] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct QubitSparsePauliList {
    /// The number of qubits the Paulis are defined on.  This is not inferable from any other shape
    /// or values, since identities are not stored explicitly.
    num_qubits: u32,
    /// A flat list of single-qubit paulis.  This is more naturally a list of lists, but is stored
    /// flat for memory usage and locality reasons, with the sublists denoted by `boundaries.`
    paulis: Vec<Pauli>,
    /// A flat list of the qubit indices that the corresponding entries in `paulis` act on.  This
    /// list must always be term-wise sorted, where a term is a sublist as denoted by `boundaries`.
    indices: Vec<u32>,
    /// Indices that partition `paulis` and `indices` into sublists for each individual sparse
    /// pauli.  `boundaries[0]..boundaries[1]` is the range of indices into `paulis` and
    /// `indices` that correspond to the first term of the sum.  All unspecified qubit indices are
    /// implicitly the identity.
    boundaries: Vec<usize>,
}

impl QubitSparsePauliList {
    /// Create a new qubit-sparse Pauli list from the raw components that make it up.
    ///
    /// This checks the input values for data coherence on entry.  If you are certain you have the
    /// correct values, you can call `new_unchecked` instead.
    pub fn new(
        num_qubits: u32,
        paulis: Vec<Pauli>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Result<Self, CoherenceError> {
        if paulis.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                paulis: paulis.len(),
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
        Ok(unsafe { Self::new_unchecked(num_qubits, paulis, indices, boundaries) })
    }

    /// Create a new [QubitSparsePauliList] from the raw components without checking data coherence.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the data-coherence requirements, as enumerated in the
    /// struct-level documentation, have been upheld.
    #[inline(always)]
    pub unsafe fn new_unchecked(
        num_qubits: u32,
        paulis: Vec<Pauli>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Self {
        Self {
            num_qubits,
            paulis,
            indices,
            boundaries,
        }
    }

    /// Create a new empty list with pre-allocated space for the given number of Paulis and
    /// single-qubit pauli terms.
    #[inline]
    pub fn with_capacity(num_qubits: u32, num_terms: usize, num_paulis: usize) -> Self {
        Self {
            num_qubits,
            paulis: Vec::with_capacity(num_paulis),
            indices: Vec::with_capacity(num_paulis),
            boundaries: {
                let mut boundaries = Vec::with_capacity(num_terms + 1);
                boundaries.push(0);
                boundaries
            },
        }
    }

    /// Get an iterator over the individual elements of the list.
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = QubitSparsePauliView<'_>> + '_ {
        std::ops::Range {
            start: 0,
            end: self.boundaries.len() - 1,
        }
        .map(|i| {
            let start = self.boundaries[i];
            let end = self.boundaries[i + 1];
            QubitSparsePauliView {
                num_qubits: self.num_qubits,
                paulis: &self.paulis[start..end],
                indices: &self.indices[start..end],
            }
        })
    }

    /// Get the number of qubits the paulis are defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Get the number of elements in the list.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.boundaries.len() - 1
    }

    /// Get the indices of each [Pauli].
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Get the boundaries of each term.
    #[inline]
    pub fn boundaries(&self) -> &[usize] {
        &self.boundaries
    }

    /// Get the [Pauli]s in the list.
    #[inline]
    pub fn paulis(&self) -> &[Pauli] {
        &self.paulis
    }

    /// Create a [QubitSparsePauliList] representing the empty list on ``num_qubits`` qubits.
    pub fn empty(num_qubits: u32) -> Self {
        Self::with_capacity(num_qubits, 0, 0)
    }

    /// Clear all the elements of the list.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction of elements in the list may not need to reallocate.
    pub fn clear(&mut self) {
        self.paulis.clear();
        self.indices.clear();
        self.boundaries.truncate(1);
    }

    /// Get a view onto a representation of a single sparse Pauli.
    ///
    /// This is effectively an indexing operation into the [QubitSparsePauliList].
    ///
    /// # Panics
    ///
    /// If the index is out of bounds.
    pub fn term(&self, index: usize) -> QubitSparsePauliView<'_> {
        debug_assert!(index < self.num_terms(), "index {index} out of bounds");
        let start = self.boundaries[index];
        let end = self.boundaries[index + 1];
        QubitSparsePauliView {
            num_qubits: self.num_qubits,
            paulis: &self.paulis[start..end],
            indices: &self.indices[start..end],
        }
    }

    /// Add an element to the list implied by a dense string label.
    pub fn add_dense_label<L: AsRef<[u8]>>(&mut self, label: L) -> Result<(), LabelError> {
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
            match Pauli::try_from_u8(*letter) {
                Ok(Some(term)) => {
                    self.paulis.push(term);
                    self.indices.push(i as u32);
                }
                Ok(None) => (),
                Err(_) => {
                    // Undo any modifications to ourselves so we stay in a consistent state.
                    let num_single_terms = self.boundaries[self.boundaries.len() - 1];
                    self.paulis.truncate(num_single_terms);
                    self.indices.truncate(num_single_terms);
                    return Err(LabelError::OutsideAlphabet);
                }
            }
        }
        self.boundaries.push(self.paulis.len());
        Ok(())
    }

    /// Add a single sparse Pauli term to the list.
    pub fn add_qubit_sparse_pauli(
        &mut self,
        term: QubitSparsePauliView,
    ) -> Result<(), ArithmeticError> {
        if self.num_qubits != term.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: term.num_qubits,
            });
        }
        self.paulis.extend_from_slice(term.paulis);
        self.indices.extend_from_slice(term.indices);
        self.boundaries.push(self.paulis.len());
        Ok(())
    }

    /// Relabel the `indices` in the map to new values.
    ///
    /// This fails if any of the new indices are too large, or if any mapping would cause a term to
    /// contain duplicates of the same index.  It may not detect if multiple qubits are mapped to
    /// the same index, if those qubits never appear together in the same term.  Such a mapping
    /// would not cause data-coherence problems (the output map will be valid), but is
    /// unlikely to be what you intended.
    ///
    /// *Panics* if `new_qubits` is not long enough to map every index used in the map.
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
                order.insert(new_qubits[self.indices[j] as usize], self.paulis[j]);
            }
            if order.len() != end - start {
                return Err(CoherenceError::DuplicateIndices);
            }
            for (index, dest) in order.keys().zip(&mut self.indices[start..end]) {
                *dest = *index;
            }
            for (pauli, dest) in order.values().zip(&mut self.paulis[start..end]) {
                *dest = *pauli;
            }
            order.clear();
        }
        Ok(())
    }

    // Return a Vec of dense labels representing this Pauli list
    pub fn to_dense_label_list(&self) -> Vec<String> {
        let mut dense_label_list = Vec::with_capacity(self.num_terms());

        for qubit_sparse_pauli in self.iter() {
            dense_label_list.push(qubit_sparse_pauli.to_term().to_dense_label());
        }
        dense_label_list
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

    // Check if the elements of `self` commute with the elements of `other`.
    pub fn commutes(&self, other: &QubitSparsePauliList) -> Result<Array2<bool>, ArithmeticError> {
        if self.num_qubits != other.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: other.num_qubits,
            });
        }

        Ok(Array2::from_shape_fn(
            (self.num_terms(), other.num_terms()),
            |(i, j)| {
                self.term(i)
                    .to_term()
                    .commutes(&other.term(j).to_term())
                    .unwrap()
            },
        ))
    }
}

type RawParts = (Vec<Pauli>, Vec<u32>, Vec<usize>);

pub fn raw_parts_from_sparse_list(
    iter: Vec<(String, Vec<u32>)>,
    num_qubits: u32,
) -> Result<RawParts, LabelError> {
    let mut boundaries = Vec::with_capacity(iter.len() + 1);
    boundaries.push(0);
    let mut indices = Vec::new();
    let mut paulis = Vec::new();
    // Insertions to the `BTreeMap` keep it sorted by keys, so we use this to do the termwise
    // sorting on-the-fly.
    let mut sorted = btree_map::BTreeMap::new();
    for (label, qubits) in iter {
        sorted.clear();
        let label: &[u8] = label.as_ref();
        if label.len() != qubits.len() {
            return Err(LabelError::WrongLengthIndices {
                label: label.len(),
                indices: qubits.len(),
            });
        }
        for (letter, index) in label.iter().zip(qubits) {
            if index >= num_qubits {
                return Err(LabelError::BadIndex { index, num_qubits });
            }
            let btree_map::Entry::Vacant(entry) = sorted.entry(index) else {
                return Err(LabelError::DuplicateIndex { index });
            };
            entry.insert(Pauli::try_from_u8(*letter).map_err(|_| LabelError::OutsideAlphabet)?);
        }
        for (index, term) in sorted.iter() {
            let Some(term) = term else {
                continue;
            };
            indices.push(*index);
            paulis.push(*term);
        }
        boundaries.push(paulis.len());
    }
    Ok((paulis, indices, boundaries))
}

/// A view object onto a single term of a `QubitSparsePauliList`.
///
/// The lengths of `paulis` and `indices` are guaranteed to be created equal, but might be zero
/// (in the case that the term is the identity).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct QubitSparsePauliView<'a> {
    pub num_qubits: u32,
    pub paulis: &'a [Pauli],
    pub indices: &'a [u32],
}
impl QubitSparsePauliView<'_> {
    /// Convert this `QubitSparsePauliView` into an owning [QubitSparsePauli] of the same data.
    pub fn to_term(&self) -> QubitSparsePauli {
        QubitSparsePauli {
            num_qubits: self.num_qubits,
            paulis: self.paulis.into(),
            indices: self.indices.into(),
        }
    }

    pub fn num_ys(self) -> isize {
        let mut num_ys = 0;
        for pauli in self.paulis {
            num_ys += (*pauli == Pauli::Y) as isize;
        }
        num_ys
    }

    pub fn to_sparse_str(self) -> String {
        let paulis = self
            .indices
            .iter()
            .zip(self.paulis)
            .rev()
            .map(|(i, op)| format!("{}_{}", op.py_label(), i))
            .collect::<Vec<String>>()
            .join(" ");
        paulis.to_string()
    }
}

/// A single qubit-spare Pauli operator.
#[derive(Clone, Debug, PartialEq)]
pub struct QubitSparsePauli {
    /// Number of qubits the Pauli operator is defined on.
    num_qubits: u32,
    /// A list of the non-identity single-qubit Paulis in the operator.
    paulis: Box<[Pauli]>,
    /// A flat list of the qubit indices that the corresponding entries in `paulis` act on.
    indices: Box<[u32]>,
}

impl QubitSparsePauli {
    /// Create a new qubit-sparse Pauli from the raw components that make it up.
    pub fn new(
        num_qubits: u32,
        paulis: Box<[Pauli]>,
        indices: Box<[u32]>,
    ) -> Result<Self, CoherenceError> {
        if paulis.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                paulis: paulis.len(),
                indices: indices.len(),
            });
        }

        if indices.iter().any(|index| *index >= num_qubits) {
            return Err(CoherenceError::BitIndexTooHigh);
        }

        Ok(Self {
            num_qubits,
            paulis,
            indices,
        })
    }

    pub fn from_dense_label(label: &str) -> Result<QubitSparsePauli, LabelError> {
        let label: &[u8] = label.as_ref();
        let num_qubits = label.len() as u32;
        let mut paulis = Vec::new();
        let mut indices = Vec::new();
        // The only valid characters in the alphabet are ASCII, so if we see something other than
        // ASCII, we're already in the failure path.
        for (i, letter) in label.iter().rev().enumerate() {
            match Pauli::try_from_u8(*letter) {
                Ok(Some(term)) => {
                    paulis.push(term);
                    indices.push(i as u32);
                }
                Ok(None) => (),
                Err(_) => {
                    return Err(LabelError::OutsideAlphabet);
                }
            }
        }
        Ok(unsafe {
            QubitSparsePauli::new_unchecked(
                num_qubits,
                paulis.into_boxed_slice(),
                indices.into_boxed_slice(),
            )
        })
    }

    // Return a dense label representing this Pauli
    pub fn to_dense_label(&self) -> String {
        let mut pauli_str = "".to_string();

        let mut current_idx = 0;

        for (index, pauli) in self.indices().iter().zip(self.paulis().iter()) {
            if *index > current_idx {
                pauli_str =
                    (0..(index - current_idx)).map(|_| "I").collect::<String>() + &pauli_str;
                current_idx = *index;
            }
            pauli_str = pauli.py_label().to_string() + &pauli_str;
            current_idx += 1;
        }

        if current_idx < self.num_qubits() {
            pauli_str = (0..(self.num_qubits() - current_idx))
                .map(|_| "I")
                .collect::<String>()
                + &pauli_str;
        }
        pauli_str
    }

    /// Create a new [QubitSparsePauli] from the raw components without checking data coherence.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the data-coherence requirements, as enumerated in the
    /// struct-level documentation, have been upheld.
    #[inline(always)]
    pub unsafe fn new_unchecked(
        num_qubits: u32,
        paulis: Box<[Pauli]>,
        indices: Box<[u32]>,
    ) -> Self {
        Self {
            num_qubits,
            paulis,
            indices,
        }
    }

    /// Get the number of qubits the paulis are defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Get the indices of each [Pauli].
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Get the [Pauli]s in the list.
    #[inline]
    pub fn paulis(&self) -> &[Pauli] {
        &self.paulis
    }

    /// Create the identity [QubitSparsePauli] on ``num_qubits`` qubits.
    pub fn identity(num_qubits: u32) -> Self {
        Self {
            num_qubits,
            paulis: Box::new([]),
            indices: Box::new([]),
        }
    }

    // Phaseless composition of two pauli operators.
    pub fn compose(&self, other: &QubitSparsePauli) -> Result<QubitSparsePauli, ArithmeticError> {
        if self.num_qubits != other.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: other.num_qubits,
            });
        }

        // if either are the identity, return a clone of the other
        if self.indices.is_empty() {
            return Ok(other.clone());
        }

        if other.indices.is_empty() {
            return Ok(self.clone());
        }

        let mut paulis = Vec::new();
        let mut indices = Vec::new();

        let mut self_idx = 0;
        let mut other_idx = 0;

        // iterate through each entry of self and other one time, incrementing based on the ordering
        // or equality of self_idx and other_idx, until one of them runs out of entries
        while self_idx < self.indices.len() && other_idx < other.indices.len() {
            if self.indices[self_idx] < other.indices[other_idx] {
                // if the current qubit index of self is strictly less than other, append the pauli
                paulis.push(self.paulis[self_idx]);
                indices.push(self.indices[self_idx]);
                self_idx += 1;
            } else if self.indices[self_idx] == other.indices[other_idx] {
                // if the indices are the same, perform multiplication and append if non-identity
                let new_pauli = (self.paulis[self_idx] as u8) ^ (other.paulis[other_idx] as u8);
                if new_pauli != 0 {
                    paulis.push(match new_pauli {
                        0b01 => Ok(Pauli::Z),
                        0b10 => Ok(Pauli::X),
                        0b11 => Ok(Pauli::Y),
                        _ => Err(ArithmeticError::PauliMultiplication { b: new_pauli }),
                    }?);
                    indices.push(self.indices[self_idx])
                }
                self_idx += 1;
                other_idx += 1;
            } else {
                // same as the first if block but with roles of self and other reversed
                paulis.push(other.paulis[other_idx]);
                indices.push(other.indices[other_idx]);
                other_idx += 1;
            }
        }

        // if any entries remain in either pauli, append them
        if other_idx != other.indices.len() {
            paulis.append(&mut other.paulis[other_idx..].to_vec());
            indices.append(&mut other.indices[other_idx..].to_vec());
        } else if self_idx != self.indices.len() {
            paulis.append(&mut self.paulis[self_idx..].to_vec());
            indices.append(&mut self.indices[self_idx..].to_vec());
        }

        Ok(QubitSparsePauli {
            num_qubits: self.num_qubits,
            paulis: paulis.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
        })
    }

    /// Get a view version of this object.
    pub fn view(&self) -> QubitSparsePauliView<'_> {
        QubitSparsePauliView {
            num_qubits: self.num_qubits,
            paulis: &self.paulis,
            indices: &self.indices,
        }
    }

    /// Convert this single Pauli into a :class:`QubitSparsePauliList`.
    pub fn to_qubit_sparse_pauli_list(&self) -> QubitSparsePauliList {
        QubitSparsePauliList {
            num_qubits: self.num_qubits,
            paulis: self.paulis.to_vec(),
            indices: self.indices.to_vec(),
            boundaries: vec![0, self.paulis.len()],
        }
    }

    // Check if `self` commutes with `other`.
    pub fn commutes(&self, other: &QubitSparsePauli) -> Result<bool, ArithmeticError> {
        if self.num_qubits != other.num_qubits {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits,
                right: other.num_qubits,
            });
        }

        let mut commutes = true;
        let mut self_idx = 0;
        let mut other_idx = 0;

        // iterate through each entry of self and other one time, incrementing based on the ordering
        // or equality of self_idx and other_idx, until one of them runs out of entries
        while self_idx < self.indices.len() && other_idx < other.indices.len() {
            if self.indices[self_idx] < other.indices[other_idx] {
                self_idx += 1;
            } else if self.indices[self_idx] == other.indices[other_idx] {
                // if the indices are the same, check commutation
                commutes = commutes == (self.paulis[self_idx] == other.paulis[other_idx]);
                self_idx += 1;
                other_idx += 1;
            } else {
                other_idx += 1;
            }
        }

        Ok(commutes)
    }
}

#[derive(Error, Debug)]
pub struct InnerReadError;

#[derive(Error, Debug)]
pub struct InnerWriteError;

impl ::std::fmt::Display for InnerReadError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "Failed acquiring lock for reading.")
    }
}

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
impl From<PauliFromU8Error> for PyErr {
    fn from(value: PauliFromU8Error) -> PyErr {
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

/// The single-character string label used to represent this term in the
/// :class:`QubitSparsePauliList` alphabet.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "label")]
fn pauli_label(py: Python<'_>, slf: Pauli) -> &Bound<'_, PyString> {
    // This doesn't use `py_label` so we can use `intern!`.
    match slf {
        Pauli::X => intern!(py, "X"),
        Pauli::Y => intern!(py, "Y"),
        Pauli::Z => intern!(py, "Z"),
    }
}
/// Construct the Python-space `IntEnum` that represents the same values as the Rust-spce `Pauli`.
///
/// We don't make `Pauli` a direct `pyclass` because we want the behaviour of `IntEnum`, which
/// specifically also makes its variants subclasses of the Python `int` type; we use a type-safe
/// enum in Rust, but from Python space we expect people to (carefully) deal with the raw ints in
/// Numpy arrays for efficiency.
///
/// The resulting class is attached to `QubitSparsePauliList` as a class attribute, and its
/// `__qualname__` is set to reflect this.
#[cfg(feature = "python")]
fn make_py_pauli(py: Python) -> PyResult<Py<PyType>> {
    let terms = [Pauli::X, Pauli::Y, Pauli::Z]
        .into_iter()
        .flat_map(|term| {
            let mut out = vec![(term.py_label(), term as u8)];
            if term.py_label() != term.py_label() {
                // Also ensure that the labels are created as aliases.  These can't be (easily) accessed
                // by attribute-getter (dot) syntax, but will work with the item-getter (square-bracket)
                // syntax, or programmatically with `getattr`.
                out.push((term.py_label(), term as u8));
            }
            out
        })
        .collect::<Vec<_>>();
    let obj = py.import("enum")?.getattr("IntEnum")?.call(
        ("Pauli", terms),
        Some(
            &[
                ("module", "qiskit.quantum_info"),
                ("qualname", "QubitSparsePauliList.Pauli"),
            ]
            .into_py_dict(py)?,
        ),
    )?;
    let label_property = py
        .import("builtins")?
        .getattr("property")?
        .call1((wrap_pyfunction!(pauli_label, py)?,))?;
    obj.setattr("label", label_property)?;
    Ok(obj.cast_into::<PyType>()?.unbind())
}

// Return the relevant value from the Python-space sister enumeration.  These are Python-space
// singletons and subclasses of Python `int`.  We only use this for interaction with "high level"
// Python space; the efficient Numpy-like array paths use `u8` directly so Numpy can act on it
// efficiently.
#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for Pauli {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let terms = PAULI_INTO_PY.get_or_init(py, || {
            let py_enum = PAULI_PY_ENUM
                .get_or_try_init(py, || make_py_pauli(py))
                .expect("creating a simple Python enum class should be infallible")
                .bind(py);
            ::std::array::from_fn(|val| {
                ::bytemuck::checked::try_cast(val as u8)
                    .ok()
                    .map(|term: Pauli| {
                        py_enum
                            .getattr(term.py_label())
                            .expect("the created `Pauli` enum should have matching attribute names to the terms")
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
impl<'a, 'py> FromPyObject<'a, 'py> for Pauli {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let value = ob
            .extract::<isize>()
            .map_err(|_| match ob.get_type().repr() {
                Ok(repr) => PyTypeError::new_err(format!("bad type for 'Pauli': {repr}")),
                Err(err) => err,
            })?;
        let value_error = || {
            PyValueError::new_err(format!(
                "value {value} is not a valid letter of the single-qubit alphabet for 'Pauli'"
            ))
        };
        let value: u8 = value.try_into().map_err(|_| value_error())?;
        value.try_into().map_err(|_| value_error())
    }
}

/// A phase-less Pauli operator stored in a qubit-sparse format.
///
/// Representation
/// ==============
///
/// A Pauli operator is a tensor product of single-qubit Pauli operators of the form :math:`P =
/// \bigotimes_n A^{(n)}_i`, for :math:`A^{(n)}_i \in \{I, X, Y, Z\}`. The internal representation
/// of a :class:`QubitSparsePauli` stores only the non-identity single-qubit Pauli operators.
///
/// Internally, each single-qubit Pauli operator is stored with a numeric value, explicitly:
///
/// .. _qubit-sparse-pauli-alphabet:
/// .. table:: Alphabet of single-qubit Pauli operators used in :class:`QubitSparsePauliList`
///
///   =======  =======================================  ===============  ===========================
///   Label    Operator                                 Numeric value    :class:`.Pauli` attribute
///   =======  =======================================  ===============  ===========================
///   ``"I"``  :math:`I` (identity)                     Not stored.      Not stored.
///
///   ``"X"``  :math:`X` (Pauli X)                      ``0b10`` (2)     :attr:`~.Pauli.X`
///
///   ``"Y"``  :math:`Y` (Pauli Y)                      ``0b11`` (3)     :attr:`~.Pauli.Y`
///
///   ``"Z"``  :math:`Z` (Pauli Z)                      ``0b01`` (1)     :attr:`~.Pauli.Z`
///
///   =======  =======================================  ===============  ===========================
///
/// .. _qubit-sparse-pauli-arrays:
/// .. table:: Data arrays used to represent :class:`.QubitSparsePauli`
///
///   ==================  ===========  =============================================================
///   Attribute           Length       Description
///   ==================  ===========  =============================================================
///   :attr:`paulis`      :math:`s`    Each of the non-identity single-qubit Pauli operators.  These
///                                    correspond to the non-identity :math:`A^{(n)}_i` in the list,
///                                    where the entries are stored in order of increasing :math:`i`
///                                    first, and in order of increasing :math:`n` within each term.
///
///   :attr:`indices`     :math:`s`    The corresponding qubit (:math:`n`) for each of the operators
///                                    in :attr:`paulis`.  :class:`QubitSparsePauli` requires
///                                    that this list is term-wise sorted, and algorithms can rely
///                                    on this invariant being upheld.
///   ==================  ===========  =============================================================
///
/// The parameter :math:`s` is the total number of non-identity single-qubit terms.
///
/// The scalar item of the :attr:`paulis` array is stored as a numeric byte.  The numeric values
/// are related to the symplectic Pauli representation that :class:`.SparsePauliOp` uses, and are
/// accessible with named access by an enumeration:
///
/// ..
///     This is documented manually here because the Python-space `Enum` is generated
///     programmatically from Rust - it'd be _more_ confusing to try and write a docstring somewhere
///     else in this source file. The use of `autoattribute` is because it pulls in the numeric
///     value.
///
/// .. py:class:: QubitSparsePauli.Pauli
///
///
///     An :class:`~enum.IntEnum` that provides named access to the numerical values used to
///     represent each of the single-qubit alphabet terms enumerated in
///     :ref:`qubit-sparse-pauli-alphabet`.
///
///     This class is attached to :class:`.QubitSparsePauli`.  Access it as
///     :class:`.QubitSparsePauli.Pauli`.  If this is too much typing, and you are solely
///     dealing with :class:`QubitSparsePauliList` objects and the :class:`Pauli` name is not
///     ambiguous, you might want to shorten it as::
///
///         >>> ops = QubitSparsePauli.Pauli
///         >>> assert ops.X is QubitSparsePauli.Pauli.X
///
///     You can access all the values of the enumeration either with attribute access, or with
///     dictionary-like indexing by string::
///
///         >>> assert QubitSparsePauli.Pauli.X is QubitSparsePauli.Pauli["X"]
///
///     The bits representing each single-qubit Pauli are the (phase-less) symplectic representation
///     of the Pauli operator.
///
///     Values
///     ------
///
///     .. autoattribute:: qiskit.quantum_info::QubitSparsePauli.Pauli.X
///
///         The Pauli :math:`X` operator.  Uses the single-letter label ``"X"``.
///
///     .. autoattribute:: qiskit.quantum_info::QubitSparsePauli.Pauli.Y
///
///         The Pauli :math:`Y` operator.  Uses the single-letter label ``"Y"``.
///
///     .. autoattribute:: qiskit.quantum_info::QubitSparsePauli.Pauli.Z
///
///         The Pauli :math:`Z` operator.  Uses the single-letter label ``"Z"``.
///
///
/// Each of the array-like attributes behaves like a Python sequence.  You can index and slice these
/// with standard :class:`list`-like semantics.  Slicing an attribute returns a Numpy
/// :class:`~numpy.ndarray` containing a copy of the relevant data with the natural ``dtype`` of the
/// field; this lets you easily do mathematics on the results, like bitwise operations on
/// :attr:`paulis`.
///
/// Construction
/// ============
///
/// :class:`QubitSparsePauli` defines several constructors.  The default constructor will
/// attempt to delegate to one of the more specific constructors, based on the type of the input.
/// You can always use the specific constructors to have more control over the construction.
///
/// .. _qubit-sparse-pauli-convert-constructors:
/// .. table:: Construction from other objects
///
///   ============================  ================================================================
///   Method                        Summary
///   ============================  ================================================================
///   :meth:`from_label`            Convert a dense string label into a :class:`~.QubitSparsePauli`.
///
///   :meth:`from_sparse_label`     Build a :class:`.QubitSparsePauli` from a tuple of a sparse
///                                 string label and the qubits they apply to.
///
///   :meth:`from_pauli`            Raise a single :class:`~.quantum_info.Pauli` into a
///                                 :class:`.QubitSparsePauli`.
///
///   :meth:`from_raw_parts`        Build the operator from :ref:`the raw data arrays
///                                 <qubit-sparse-pauli-arrays>`.
///   ============================  ================================================================
///
/// .. py:function:: QubitSparsePauli.__new__(data, /, num_qubits=None)
///
///     The default constructor of :class:`QubitSparsePauli`.
///
///     This delegates to one of :ref:`the explicit conversion-constructor methods
///     <qubit-sparse-pauli-convert-constructors>`, based on the type of the ``data`` argument.
///     If ``num_qubits`` is supplied and constructor implied by the type of ``data`` does not
///     accept a number, the given integer must match the input.
///
///     :param data: The data type of the input.  This can be another :class:`QubitSparsePauli`,
///         in which case the input is copied, or it can be a valid format for either
///         :meth:`from_label` or :meth:`from_sparse_label`.
///     :param int|None num_qubits: Optional number of qubits for the operator.  For most data
///         inputs, this can be inferred and need not be passed.  It is only necessary for the
///         sparse-label format.  If given unnecessarily, it must match the data input.
#[cfg(feature = "python")]
#[pyclass(
    name = "QubitSparsePauli",
    frozen,
    module = "qiskit.quantum_info",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyQubitSparsePauli {
    inner: QubitSparsePauli,
}

#[cfg(feature = "python")]
impl PyQubitSparsePauli {
    pub fn inner(&self) -> &QubitSparsePauli {
        &self.inner
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQubitSparsePauli {
    #[new]
    #[pyo3(signature = (data, /, num_qubits=None))]
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
        if data.is_instance(imports::PAULI_TYPE.get_bound(py))? {
            check_num_qubits(data)?;
            return Self::from_pauli(data);
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
            return Self::from_label(&label);
        }
        if let Ok(sparse_label) = data.extract() {
            let Some(num_qubits) = num_qubits else {
                return Err(PyValueError::new_err(
                    "if using the sparse-label form, 'num_qubits' must be provided",
                ));
            };
            return Self::from_sparse_label(sparse_label, num_qubits);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'QubitSparsePauli': {}",
            data.get_type().repr()?,
        )))
    }

    /// Construct a :class:`.QubitSparsePauli` from raw Numpy arrays that match :ref:`the required
    /// data representation described in the class-level documentation
    /// <qubit-sparse-pauli-arrays>`.
    ///
    /// The data from each array is copied into fresh, growable Rust-space allocations.
    ///
    /// Args:
    ///     num_qubits: number of qubits the operator acts on.
    ///     paulis: list of the single-qubit terms.  This should be a Numpy array with dtype
    ///         :attr:`~numpy.uint8` (which is compatible with :class:`.Pauli`).
    ///     indices: sorted list of the qubits each single-qubit term corresponds to.  This should
    ///         be a Numpy array with dtype :attr:`~numpy.uint32`.
    ///
    /// Examples:
    ///
    ///     Construct a :math:`Z` operator acting on qubit 50 of 100 qubits.
    ///
    ///         >>> num_qubits = 100
    ///         >>> terms = np.array([QubitSparsePauli.Pauli.Z], dtype=np.uint8)
    ///         >>> indices = np.array([50], dtype=np.uint32)
    ///         >>> QubitSparsePauli.from_raw_parts(num_qubits, terms, indices)
    ///         <QubitSparsePauli on 100 qubits: Z_50>
    #[staticmethod]
    #[pyo3(signature = (/, num_qubits, paulis, indices))]
    fn from_raw_parts(num_qubits: u32, paulis: Vec<Pauli>, indices: Vec<u32>) -> PyResult<Self> {
        if paulis.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                paulis: paulis.len(),
                indices: indices.len(),
            }
            .into());
        }
        let mut order = (0..paulis.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|a| indices[*a]);
        let paulis = order.iter().map(|i| paulis[*i]).collect();
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
        let inner = QubitSparsePauli::new(num_qubits, paulis, sorted_indices.into_boxed_slice())?;
        Ok(PyQubitSparsePauli { inner })
    }

    /// Construct from a dense string label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`~.quantum_info.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     label (str): the dense label.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> QubitSparsePauli.from_label("IIIIXZI")
    ///         <QubitSparsePauli on 7 qubits: X_2 Z_1>
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> assert QubitSparsePauli.from_label(label) == QubitSparsePauli.from_pauli(pauli)
    #[staticmethod]
    #[pyo3(signature = (label, /))]
    fn from_label(label: &str) -> PyResult<Self> {
        let inner = QubitSparsePauli::from_dense_label(label)?;
        Ok(inner.into())
    }

    /// Construct a :class:`.QubitSparsePauli` from a single :class:`~.quantum_info.Pauli` instance.
    ///
    /// Note that the phase of the Pauli is dropped.
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
    ///         >>> QubitSparsePauli.from_pauli(pauli)
    ///         <QubitSparsePauli on 5 qubits: Y_3 X_2 Z_1>
    ///         >>> assert QubitSparsePauli.from_label(label) == QubitSparsePauli.from_pauli(pauli)
    #[staticmethod]
    #[pyo3(signature = (pauli, /))]
    pub fn from_pauli(pauli: &Bound<PyAny>) -> PyResult<Self> {
        let py = pauli.py();
        let num_qubits = pauli.getattr(intern!(py, "num_qubits"))?.extract::<u32>()?;
        let z = pauli
            .getattr(intern!(py, "z"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let x = pauli
            .getattr(intern!(py, "x"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let mut paulis = Vec::new();
        let mut indices = Vec::new();
        for (i, (x, z)) in x.as_array().iter().zip(z.as_array().iter()).enumerate() {
            // The only failure case possible here is the identity, because of how we're
            // constructing the value to convert.
            let Ok(term) = ::bytemuck::checked::try_cast(((*x as u8) << 1) | (*z as u8)) else {
                continue;
            };
            indices.push(i as u32);
            paulis.push(term);
        }
        let inner = QubitSparsePauli::new(
            num_qubits,
            paulis.into_boxed_slice(),
            indices.into_boxed_slice(),
        )?;
        Ok(inner.into())
    }

    /// Construct a qubit sparse Pauli from a sparse label, given as a tuple of a string of Paulis,
    /// and the indices of the corresponding qubits.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`.
    ///
    /// Args:
    ///     sparse_label (tuple[str, Sequence[int]]): labels and the qubits each single-qubit term
    ///         applies to.
    ///
    ///     num_qubits (int): the number of qubits the operator acts on.
    ///
    /// Examples:
    ///
    ///     Construct a simple Pauli::
    ///
    ///         >>> QubitSparsePauli.from_sparse_label(
    ///         ...     ("ZX", (1, 4)),
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <QubitSparsePauli on 5 qubits: X_4 Z_1>
    ///
    ///     This method can replicate the behavior of :meth:`from_label`, if the qubit-arguments
    ///     field of the tuple is set to decreasing integers::
    ///
    ///         >>> label = "XYXZ"
    ///         >>> from_label = QubitSparsePauli.from_label(label)
    ///         >>> from_sparse_label = QubitSparsePauli.from_sparse_label(
    ///         ...     (label, (3, 2, 1, 0)),
    ///         ...     num_qubits=4
    ///         ... )
    ///         >>> assert from_label == from_sparse_label
    #[staticmethod]
    #[pyo3(signature = (/, sparse_label, num_qubits))]
    fn from_sparse_label(sparse_label: (String, Vec<u32>), num_qubits: u32) -> PyResult<Self> {
        let label = sparse_label.0;
        let indices = sparse_label.1;
        let mut paulis = Vec::new();
        let mut sorted_indices = Vec::new();

        let label: &[u8] = label.as_ref();
        let mut sorted = btree_map::BTreeMap::new();
        if label.len() != indices.len() {
            return Err(LabelError::WrongLengthIndices {
                label: label.len(),
                indices: indices.len(),
            }
            .into());
        }
        for (letter, index) in label.iter().zip(indices) {
            if index >= num_qubits {
                return Err(LabelError::BadIndex { index, num_qubits }.into());
            }
            let btree_map::Entry::Vacant(entry) = sorted.entry(index) else {
                return Err(LabelError::DuplicateIndex { index }.into());
            };
            entry.insert(Pauli::try_from_u8(*letter).map_err(|_| LabelError::OutsideAlphabet)?);
        }
        for (index, term) in sorted.iter() {
            let Some(term) = term else {
                continue;
            };
            sorted_indices.push(*index);
            paulis.push(*term);
        }

        let inner = QubitSparsePauli::new(
            num_qubits,
            paulis.into_boxed_slice(),
            sorted_indices.into_boxed_slice(),
        )?;
        Ok(inner.into())
    }

    /// Get the identity operator for a given number of qubits.
    ///
    /// Examples:
    ///
    ///     Get the identity on 100 qubits::
    ///
    ///         >>> QubitSparsePauli.identity(100)
    ///         <QubitSparsePauli on 100 qubits: >
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn identity(num_qubits: u32) -> Self {
        QubitSparsePauli::identity(num_qubits).into()
    }

    /// Convert this Pauli into a single element :class:`QubitSparsePauliList`.
    fn to_qubit_sparse_pauli_list(&self) -> PyResult<PyQubitSparsePauliList> {
        let qubit_sparse_pauli_list = QubitSparsePauliList::new(
            self.inner.num_qubits(),
            self.inner.paulis().to_vec(),
            self.inner.indices().to_vec(),
            vec![0, self.inner.paulis().len()],
        )?;
        Ok(qubit_sparse_pauli_list.into())
    }

    /// Phaseless composition with another :class:`QubitSparsePauli`.
    ///
    /// Args:
    ///     other (QubitSparsePauli): the qubit sparse Pauli to compose with.
    fn compose(&self, other: &PyQubitSparsePauli) -> PyResult<Self> {
        Ok(PyQubitSparsePauli {
            inner: self.inner.compose(&other.inner)?,
        })
    }

    fn __matmul__(&self, other: &PyQubitSparsePauli) -> PyResult<Self> {
        self.compose(other)
    }

    /// Check if `self`` commutes with another qubit sparse Pauli.
    ///
    /// Args:
    ///     other (QubitSparsePauli): the qubit sparse Pauli to check for commutation with.
    fn commutes(&self, other: &PyQubitSparsePauli) -> PyResult<bool> {
        Ok(self.inner.commutes(&other.inner)?)
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
            "QubitSparsePauli",
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
            Self::get_paulis(slf_.clone()),
            Self::get_indices(slf_),
        )
            .into_pyobject(py)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let paulis: &[u8] = ::bytemuck::cast_slice(self.inner.paulis());
        (
            py.get_type::<Self>().getattr("from_raw_parts")?,
            (
                self.inner.num_qubits(),
                PyArray1::from_slice(py, paulis),
                PyArray1::from_slice(py, self.inner.indices()),
            ),
        )
            .into_pyobject(py)
    }

    /// Return a :class:`~.quantum_info.Pauli` representing the same phaseless Pauli.
    fn to_pauli<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        imports::PAULI_TYPE
            .get_bound(py)
            .call1((self.inner.to_dense_label(),))
    }

    /// Get a copy of this term.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Read-only view onto the individual single-qubit terms.
    ///
    /// The only valid values in the array are those with a corresponding
    /// :class:`~QubitSparsePauli.Pauli`.
    #[getter]
    fn get_paulis(slf_: Bound<Self>) -> Bound<PyArray1<u8>> {
        let borrowed = slf_.borrow();
        let paulis = borrowed.inner.paulis();
        let arr = ::ndarray::aview1(::bytemuck::cast_slice::<_, u8>(paulis));
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[Pauli]>` allocation (including dropping or reallocating it) other than the entire
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

    // The documentation for this is inlined into the class-level documentation of
    // :class:`QubitSparsePauliList`.
    #[allow(non_snake_case)]
    #[classattr]
    pub fn Pauli(py: Python) -> PyResult<Py<PyType>> {
        PAULI_PY_ENUM
            .get_or_try_init(py, || make_py_pauli(py))
            .map(|obj| obj.clone_ref(py))
    }
}

/// A list of phase-less Pauli operators stored in a qubit-sparse format.
///
/// Representation
/// ==============
///
/// Each individual Pauli operator in the list is a tensor product of single-qubit Pauli operators
/// of the form :math:`P = \bigotimes_n A^{(n)}_i`, for :math:`A^{(n)}_i \in \{I, X, Y, Z\}`. The
/// internal representation of a :class:`QubitSparsePauliList` stores only the non-identity
/// single-qubit Pauli operators.  This makes it significantly more efficient to represent lists of
/// Pauli operators with low weights on a large number of qubits. For example, the list of
/// :math`n`-qubit operators :math:`[Z^{(0)}, \dots Z^{(n-1)}]`, where :math:`Z^{(j)}` represents
/// The :math:`Z` operator on qubit :math:`j` and identity on all others, can be stored in
/// :class:`QubitSparsePauliList` with a linear amount of memory in the number of qubits.
///
/// Indexing
/// --------
///
/// :class:`QubitSparsePauliList` behaves as `a Python sequence
/// <https://docs.python.org/3/glossary.html#term-sequence>`__ (the standard form, not the expanded
/// :class:`collections.abc.Sequence`).  The elements of the list can be indexed by integers, as
/// well as iterated through. Whether through indexing or iterating, elements of the list are
/// returned as :class:`QubitSparsePauli` instances.
///
/// Construction
/// ============
///
/// :class:`QubitSparsePauliList` defines several constructors.  The default constructor will
/// attempt to delegate to one of the more specific constructors, based on the type of the input.
/// You can always use the specific constructors to have more control over the construction.
///
/// .. _qubit-sparse-pauli-list-convert-constructors:
/// .. table:: Construction from other objects
///
///   ================================  ============================================================
///   Method                            Summary
///   ================================  ============================================================
///   :meth:`from_label`                Convert a dense string label into a single-element
///                                     :class:`.QubitSparsePauliList`.
///
///   :meth:`from_list`                 Construct from a list of dense string labels.
///
///   :meth:`from_sparse_list`          Elements given as a list of tuples of sparse string labels
///                                     and the qubits they apply to.
///
///   :meth:`from_pauli`                Raise a single :class:`~.quantum_info.Pauli` into a
///                                     single-element :class:`.QubitSparsePauliList`.
///
///   :meth:`from_qubit_sparse_paulis`  Construct from a list of :class:`.QubitSparsePauli`\s.
///   ================================  ============================================================
///
/// .. py:function:: QubitSparsePauliList.__new__(data, /, num_qubits=None)
///
///     The default constructor of :class:`QubitSparsePauliList`.
///
///     This delegates to one of :ref:`the explicit conversion-constructor methods
///     <qubit-sparse-pauli-list-convert-constructors>`, based on the type of the ``data`` argument.
///     If ``num_qubits`` is supplied and constructor implied by the type of ``data`` does not
///     accept a number, the given integer must match the input.
///
///     :param data: The data type of the input.  This can be another :class:`QubitSparsePauliList`,
///         in which case the input is copied, or it can be a list in a valid format for either
///         :meth:`from_list` or :meth:`from_sparse_list`.
///     :param int|None num_qubits: Optional number of qubits for the list.  For most data
///         inputs, this can be inferred and need not be passed.  It is only necessary for empty
///         lists or the sparse-list format.  If given unnecessarily, it must match the data input.
///
/// In addition to the conversion-based constructors, the method :meth:`empty` can be used to
/// construct an empty list of qubit-sparse Paulis acting on a given number of qubits.
///
/// Conversions
/// ===========
///
/// An existing :class:`QubitSparsePauliList` can be converted into other formats.
///
/// .. table:: Conversion methods to other observable forms.
///
///   ===========================  =================================================================
///   Method                       Summary
///   ===========================  =================================================================
///   :meth:`to_sparse_list`       Express the observable in a sparse list format with elements
///                                ``(paulis, indices)``.
///   ===========================  =================================================================
#[cfg(feature = "python")]
#[pyclass(
    name = "QubitSparsePauliList",
    module = "qiskit.quantum_info",
    sequence
)]
#[derive(Debug)]
pub struct PyQubitSparsePauliList {
    // This class keeps a pointer to a pure Rust-SparseTerm and serves as interface from Python.
    pub inner: Arc<RwLock<QubitSparsePauliList>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQubitSparsePauliList {
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
        if data.is_instance(imports::PAULI_TYPE.get_bound(py))? {
            check_num_qubits(data)?;
            return Self::from_pauli(data);
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
        if let Ok(pauli_list) = data.cast_exact::<Self>() {
            check_num_qubits(data)?;
            let borrowed = pauli_list.borrow();
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
        if let Ok(term) = data.cast_exact::<PyQubitSparsePauli>() {
            return term.borrow().to_qubit_sparse_pauli_list();
        };
        if let Ok(pauli_list) = Self::from_qubit_sparse_paulis(data, num_qubits) {
            return Ok(pauli_list);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'QubitSparsePauliList': {}",
            data.get_type().repr()?,
        )))
    }

    /// Get a copy of this qubit sparse Pauli list.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> qubit_sparse_pauli_list = QubitSparsePauliList.from_list(["IXZXYYZZ", "ZXIXYYZZ"])
    ///         >>> assert qubit_sparse_pauli_list == qubit_sparse_pauli_list.copy()
    ///         >>> assert qubit_sparse_pauli_list is not qubit_sparse_pauli_list.copy()
    fn copy(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.clone().into())
    }

    /// The number of qubits the operators in the list act on.
    ///
    /// This is not inferable from any other shape or values, since identities are not stored
    /// explicitly.
    #[getter]
    #[inline]
    pub fn num_qubits(&self) -> PyResult<u32> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_qubits())
    }

    /// The number of elements in the list.
    #[getter]
    #[inline]
    pub fn num_terms(&self) -> PyResult<usize> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_terms())
    }

    /// Get the empty list for a given number of qubits.
    ///
    /// The empty list contains no elements, and is the identity element for joining two
    /// :class:`QubitSparsePauliList` instances.
    ///
    /// Examples:
    ///
    ///     Get the empty list on 100 qubits::
    ///
    ///         >>> QubitSparsePauliList.empty(100)
    ///         <QubitSparsePauliList with 0 elements on 100 qubits: []>
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn empty(num_qubits: u32) -> Self {
        QubitSparsePauliList::empty(num_qubits).into()
    }

    /// Construct a :class:`.QubitSparsePauliList` from a single :class:`~.quantum_info.Pauli`
    /// instance.
    ///
    /// The output list will have a single term. Note that the phase is dropped.
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
    ///         >>> QubitSparsePauliList.from_pauli(pauli)
    ///         <QubitSparsePauliList with 1 element on 5 qubits: [Y_3 X_2 Z_1]>
    ///         >>> assert QubitSparsePauliList.from_label(label) == QubitSparsePauliList.from_pauli(pauli)
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
        let mut paulis = Vec::new();
        let mut indices = Vec::new();
        for (i, (x, z)) in x.as_array().iter().zip(z.as_array().iter()).enumerate() {
            // The only failure case possible here is the identity, because of how we're
            // constructing the value to convert.
            let Ok(term) = ::bytemuck::checked::try_cast(((*x as u8) << 1) | (*z as u8)) else {
                continue;
            };
            indices.push(i as u32);
            paulis.push(term);
        }
        let boundaries = vec![0, indices.len()];
        let inner = QubitSparsePauliList::new(num_qubits, paulis, indices, boundaries)?;
        Ok(inner.into())
    }

    /// Construct a list with a single-term from a dense string label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`~.quantum_info.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     label (str): the dense label.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> QubitSparsePauliList.from_label("IIIIXZI")
    ///         <QubitSparsePauliList with 1 element on 7 qubits: [X_2 Z_1]>
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> assert QubitSparsePauliList.from_label(label) == QubitSparsePauliList.from_pauli(pauli)
    ///
    /// See also:
    ///     :meth:`from_list`
    ///         A generalization of this method that constructs a list from multiple labels.
    #[staticmethod]
    #[pyo3(signature = (label, /))]
    fn from_label(label: &str) -> Result<Self, LabelError> {
        let mut inner = QubitSparsePauliList::empty(label.len() as u32);
        inner.add_dense_label(label)?;
        Ok(inner.into())
    }

    /// Construct a qubit-sparse Pauli list from a list of dense labels.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_list`. In this dense form, you must supply
    /// all identities explicitly in each label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`~.quantum_info.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     iter (list[str]): List of dense string labels.
    ///     num_qubits (int | None): It is not necessary to specify this if you are sure that
    ///         ``iter`` is not an empty sequence, since it can be inferred from the label lengths.
    ///         If ``iter`` may be empty, you must specify this argument to disambiguate how many
    ///         qubits the operators act on.  If this is given and ``iter`` is not empty, the value
    ///         must match the label lengths.
    ///
    /// Examples:
    ///
    ///     Construct a qubit sparse Pauli list from a list of labels::
    ///
    ///         >>> QubitSparsePauliList.from_list([
    ///         ...     "IIIXX",
    ///         ...     "IIYYI",
    ///         ...     "IXXII",
    ///         ...     "ZZIII",
    ///         ... ])
    ///         <QubitSparsePauliList with 4 elements on 5 qubits:
    ///             [X_1 X_0, Y_2 Y_1, X_3 X_2, Z_4 Z_3]>
    ///
    ///     Use ``num_qubits`` to disambiguate potentially empty inputs::
    ///
    ///         >>> QubitSparsePauliList.from_list([], num_qubits=10)
    ///         <QubitSparsePauliList with 0 elements on 10 qubits: []>
    ///
    ///     This method is equivalent to calls to :meth:`from_sparse_list` with the explicit
    ///     qubit-arguments field set to decreasing integers::
    ///
    ///         >>> labels = ["XYXZ", "YYZZ", "XYXZ"]
    ///         >>> from_list = QubitSparsePauliList.from_list(labels)
    ///         >>> from_sparse_list = QubitSparsePauliList.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0))
    ///         ...     for label in labels
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         Construct the list from labels without explicit identities, but with the qubits each
    ///         single-qubit operator term applies to listed explicitly.
    #[staticmethod]
    #[pyo3(signature = (iter, /, *, num_qubits=None))]
    fn from_list(iter: Vec<String>, num_qubits: Option<u32>) -> PyResult<Self> {
        if iter.is_empty() && num_qubits.is_none() {
            return Err(PyValueError::new_err(
                "cannot construct a QubitSparsePauliList from an empty list without knowing `num_qubits`",
            ));
        }
        let num_qubits = match num_qubits {
            Some(num_qubits) => num_qubits,
            None => iter[0].len() as u32,
        };
        let mut inner = QubitSparsePauliList::with_capacity(num_qubits, iter.len(), 0);
        for label in iter {
            inner.add_dense_label(&label)?;
        }
        Ok(inner.into())
    }

    /// Construct a :class:`QubitSparsePauliList` out of individual :class:`QubitSparsePauli`
    /// instances.
    ///
    /// All the terms must have the same number of qubits.  If supplied, the ``num_qubits`` argument
    /// must match the terms.
    ///
    /// Args:
    ///     obj (Iterable[QubitSparsePauli]): Iterable of individual terms to build the list from.
    ///     num_qubits (int | None): The number of qubits the elements of the list should act on.
    ///         This is usually inferred from the input, but can be explicitly given to handle the
    ///         case of an empty iterable.
    ///
    /// Returns:
    ///     The corresponding list.
    #[staticmethod]
    #[pyo3(signature = (obj, /, num_qubits=None))]
    fn from_qubit_sparse_paulis(obj: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let mut iter = obj.try_iter()?;
        let mut inner = match num_qubits {
            Some(num_qubits) => QubitSparsePauliList::empty(num_qubits),
            None => {
                let Some(first) = iter.next() else {
                    return Err(PyValueError::new_err(
                        "cannot construct an empty QubitSparsePauliList without knowing `num_qubits`",
                    ));
                };
                let py_term = first?.cast::<PyQubitSparsePauli>()?.borrow();
                py_term.inner.to_qubit_sparse_pauli_list()
            }
        };
        for bound_py_term in iter {
            let py_term = bound_py_term?.cast::<PyQubitSparsePauli>()?.borrow();
            inner.add_qubit_sparse_pauli(py_term.inner.view())?;
        }
        Ok(inner.into())
    }

    /// Clear all the elements from the list, making it equal to the empty list again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction operations resulting from composition may not need to reallocate.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> pauli_list = QubitSparsePauliList.from_list(["IXXXYY", "ZZYZII"])
    ///         >>> pauli_list.clear()
    ///         >>> assert pauli_list == QubitSparsePauliList.empty(pauli_list.num_qubits)
    pub fn clear(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.clear();
        Ok(())
    }

    /// Construct a qubit sparse Pauli list from a list of labels and the qubits each item applies
    /// to.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`.
    ///
    /// The "labels" and "indices" fields of the tuples are associated by zipping them together.
    /// For example, this means that a call to :meth:`from_list` can be converted to the form used
    /// by this method by setting the "indices" field of each triple to ``(num_qubits-1, ..., 1,
    /// 0)``.
    ///
    /// Args:
    ///     iter (list[tuple[str, Sequence[int]]]): tuples of labels and the qubits each
    ///         single-qubit term applies to.
    ///
    ///     num_qubits (int): the number of qubits the operators in the list act on.
    ///
    /// Examples:
    ///
    ///     Construct a simple list::
    ///
    ///         >>> QubitSparsePauliList.from_sparse_list(
    ///         ...     [("ZX", (1, 4)), ("YY", (0, 3))],
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <QubitSparsePauliList with 2 elements on 5 qubits: [X_4 Z_1, Y_3 Y_0]>
    ///
    ///     This method can replicate the behavior of :meth:`from_list`, if the qubit-arguments
    ///     field of the tuple is set to decreasing integers::
    ///
    ///         >>> labels = ["XYXZ", "YYZZ", "XYXZ"]
    ///         >>> from_list = QubitSparsePauliList.from_list(labels)
    ///         >>> from_sparse_list = QubitSparsePauliList.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0))
    ///         ...     for label in labels
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`to_sparse_list`
    ///         The reverse of this method.
    #[staticmethod]
    #[pyo3(signature = (iter, /, num_qubits))]
    fn from_sparse_list(iter: Vec<(String, Vec<u32>)>, num_qubits: u32) -> PyResult<Self> {
        let (paulis, indices, boundaries) = raw_parts_from_sparse_list(iter, num_qubits)?;
        let inner = QubitSparsePauliList::new(num_qubits, paulis, indices, boundaries)?;
        Ok(inner.into())
    }

    /// Express the list in terms of a sparse list format.
    ///
    /// This can be seen as counter-operation of :meth:`.QubitSparsePauliList.from_sparse_list`,
    /// however the order of terms is not guaranteed to be the same at after a roundtrip to a sparse
    /// list and back.
    ///
    /// Examples:
    ///
    ///     >>> qubit_sparse_list = QubitSparsePauliList.from_list(["IIXIZ", "IIZIX"])
    ///     >>> reconstructed = QubitSparsePauliList.from_sparse_list(qubit_sparse_list.to_sparse_list(), qubit_sparse_list.num_qubits)
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         The constructor that can interpret these lists.
    #[pyo3(signature = ())]
    fn to_sparse_list(&self, py: Python) -> PyResult<Py<PyList>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        // turn a SparseView into a Python tuple of (paulis, indices, coeff)
        let to_py_tuple = |view: QubitSparsePauliView| {
            let mut pauli_string = String::with_capacity(view.paulis.len());

            for bit in view.paulis.iter() {
                pauli_string.push_str(bit.py_label());
            }
            let py_string = PyString::new(py, &pauli_string).unbind();
            let py_indices = PyList::new(py, view.indices.iter())?.unbind();

            PyTuple::new(py, vec![py_string.as_any(), py_indices.as_any()])
        };

        let out = PyList::empty(py);
        for view in inner.iter() {
            out.append(to_py_tuple(view)?)?;
        }
        Ok(out.unbind())
    }

    /// Express the list in a dense array format.
    ///
    /// Each entry is a u8 following the :class:`Pauli` representation, while the rows index
    /// distinct Paulis and the columns distinct qubits.
    ///
    /// Examples:
    ///
    ///         >>> paulis = QubitSparsePauliList.from_sparse_list(
    ///         ...     [("ZX", (1, 4)), ("YY", (0, 3)), ("XX", (0, 1))],
    ///         ...     num_qubits=5,
    ///         ... )
    ///         >>> paulis.to_dense_array()
    #[pyo3(signature = ())]
    fn to_dense_array(&self, py: Python) -> PyResult<Py<PyArray2<u8>>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let mut out = Array2::zeros((inner.num_terms(), inner.num_qubits.try_into().unwrap()));
        for (idx, paulis) in inner.iter().enumerate() {
            for (p, p_idx) in zip(paulis.paulis, paulis.indices) {
                out[[idx, *p_idx as usize]] = *p as u8;
            }
        }
        Ok(out.into_pyarray(py).unbind())
    }

    /// Check if the elements of `self`` commute with another qubit sparse Pauli list.
    ///
    /// Args:
    ///     other (QubitSparsePauliList): the qubit sparse Pauli list to check for commutation with.
    #[pyo3(signature = (other))]
    fn commutes(&self, py: Python, other: &PyQubitSparsePauliList) -> PyResult<Py<PyArray2<bool>>> {
        let slf_inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        Ok(slf_inner.commutes(&other_inner)?.into_pyarray(py).unbind())
    }

    /// Return a :class:`~.quantum_info.PauliList` representing the same phaseless list of Paulis.
    fn to_pauli_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        imports::PAULI_LIST_TYPE
            .get_bound(py)
            .call1((inner.to_dense_label_list(),))
    }

    /// Apply a transpiler layout to this qubit sparse Pauli list.
    ///
    /// This enables remapping of qubit indices, e.g. if the list is defined in terms of virtual
    /// qubit labels.
    ///
    /// Args:
    ///     layout (TranspileLayout | list[int] | None): The layout to apply.  Most uses of this
    ///         function should pass the :attr:`.QuantumCircuit.layout` field from a circuit that
    ///         was transpiled for hardware.  In addition, you can pass a list of new qubit indices.
    ///         If given as explicitly ``None``, no remapping is applied (but you can still use
    ///         ``num_qubits`` to expand the qubits in the list).
    ///     num_qubits (int | None): The number of qubits to expand the list elements to.  If not
    ///         supplied, the output will be as wide as the given :class:`.TranspileLayout`, or the
    ///         same width as the input if the ``layout`` is given in another form.
    ///
    /// Returns:
    ///     A new :class:`QubitSparsePauli` with the provided layout applied.
    #[pyo3(signature = (/, layout, num_qubits=None))]
    fn apply_layout(&self, layout: Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let py = layout.py();
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        // A utility to check the number of qubits is compatible with the map.
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
        // input types, before calling QubitSparsePauliList.apply_layout to do the actual work.
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

    fn __len__(&self) -> PyResult<usize> {
        self.num_terms()
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        (
            py.get_type::<Self>().getattr("from_sparse_list")?,
            (self.to_sparse_list(py)?, inner.num_qubits()),
        )
            .into_pyobject(py)
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: PySequenceIndex<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let indices = match index.with_len(inner.num_terms())? {
            SequenceIndex::Int(index) => {
                return PyQubitSparsePauli {
                    inner: inner.term(index).to_term(),
                }
                .into_bound_py_any(py);
            }
            indices => indices,
        };
        let mut out = QubitSparsePauliList::empty(inner.num_qubits());
        for index in indices.iter() {
            out.add_qubit_sparse_pauli(inner.term(index))?;
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
            "{} element{}",
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
            "".to_owned()
        } else {
            inner
                .iter()
                .map(QubitSparsePauliView::to_sparse_str)
                .collect::<Vec<_>>()
                .join(", ")
        };
        Ok(format!(
            "<QubitSparsePauliList with {str_num_terms} on {str_num_qubits}: [{str_terms}]>"
        ))
    }
}

#[cfg(feature = "python")]
impl From<QubitSparsePauli> for PyQubitSparsePauli {
    fn from(val: QubitSparsePauli) -> PyQubitSparsePauli {
        PyQubitSparsePauli { inner: val }
    }
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for QubitSparsePauli {
    type Target = PyQubitSparsePauli;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyQubitSparsePauli::from(self).into_pyobject(py)
    }
}

#[cfg(feature = "python")]
impl From<QubitSparsePauliList> for PyQubitSparsePauliList {
    fn from(val: QubitSparsePauliList) -> PyQubitSparsePauliList {
        PyQubitSparsePauliList {
            inner: Arc::new(RwLock::new(val)),
        }
    }
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for QubitSparsePauliList {
    type Target = PyQubitSparsePauliList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyQubitSparsePauliList::from(self).into_pyobject(py)
    }
}
