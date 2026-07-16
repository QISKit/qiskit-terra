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

#[cfg(feature = "python")]
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::{
    IntoPyObjectExt, PyErr,
    exceptions::{PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{PyInt, PyList, PyString, PyTuple, PyType},
};

#[cfg(feature = "python")]
use std::{
    collections::btree_map,
    sync::{Arc, RwLock},
};

#[cfg(feature = "python")]
use qiskit_util::py::{PySequenceIndex, SequenceIndex};

use super::qubit_sparse_pauli::{
    ArithmeticError, CoherenceError, LabelError, Pauli, QubitSparsePauli, QubitSparsePauliList,
    QubitSparsePauliView,
};

#[cfg(feature = "python")]
use super::qubit_sparse_pauli::{
    InnerReadError, InnerWriteError, PyQubitSparsePauli, raw_parts_from_sparse_list,
};

#[cfg(feature = "python")]
use crate::imports;

/// A list of Pauli operators stored in a qubit-sparse format.
///
/// See [PyQubitSparsePauliList] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct PhasedQubitSparsePauliList {
    /// The paulis.
    qubit_sparse_pauli_list: QubitSparsePauliList,
    /// Phases.
    phases: Vec<isize>,
}

impl PhasedQubitSparsePauliList {
    /// Create a new phased qubit-sparse Pauli list from the raw components that make it up.
    ///
    /// This checks the input values for data coherence on entry.  If you are certain you have the
    /// correct values, you can call `new_unchecked` instead.
    pub fn new(
        qubit_sparse_pauli_list: QubitSparsePauliList,
        phases: Vec<isize>,
    ) -> Result<Self, CoherenceError> {
        if phases.len() != qubit_sparse_pauli_list.num_terms() {
            return Err(CoherenceError::MismatchedPhaseCount {
                phases: phases.len(),
                qspl: qubit_sparse_pauli_list.num_terms(),
            });
        }
        // SAFETY: we've just done the coherence checks.
        Ok(unsafe { Self::new_unchecked(qubit_sparse_pauli_list, phases) })
    }

    /// Create a new [QubitSparsePauliList] from the raw components without checking data coherence.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the data-coherence requirements, as enumerated in the
    /// struct-level documentation, have been upheld.
    #[inline(always)]
    pub unsafe fn new_unchecked(
        qubit_sparse_pauli_list: QubitSparsePauliList,
        phases: Vec<isize>,
    ) -> Self {
        Self {
            qubit_sparse_pauli_list,
            phases,
        }
    }

    /// Create a new empty list with pre-allocated space for the given number of Paulis and
    /// single-qubit pauli terms.
    #[inline]
    pub fn with_capacity(num_qubits: u32, num_terms: usize, num_paulis: usize) -> Self {
        Self {
            qubit_sparse_pauli_list: QubitSparsePauliList::with_capacity(
                num_qubits, num_terms, num_paulis,
            ),
            phases: Vec::with_capacity(num_terms),
        }
    }

    /// Get an iterator over the individual elements of the list.
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = PhasedQubitSparsePauliView<'_>> + '_ {
        self.phases
            .iter()
            .zip(self.qubit_sparse_pauli_list.iter())
            .map(|(phase, qspv)| PhasedQubitSparsePauliView {
                qubit_sparse_pauli_view: qspv,
                phase,
            })
    }

    /// Get the number of qubits the paulis are defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli_list.num_qubits()
    }

    /// Get the number of elements in the list.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.phases.len()
    }

    /// Create a [PhasedQubitSparsePauliList] representing the empty list on ``num_qubits`` qubits.
    pub fn empty(num_qubits: u32) -> Self {
        Self::with_capacity(num_qubits, 0, 0)
    }

    /// Clear all the elements of the list.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction of elements in the list may not need to reallocate.
    pub fn clear(&mut self) {
        self.qubit_sparse_pauli_list.clear();
        self.phases.clear();
    }

    /// Get a view onto a representation of a single phased sparse Pauli.
    ///
    /// This is effectively an indexing operation into the [PhasedQubitSparsePauliList].
    ///
    /// # Panics
    ///
    /// If the index is out of bounds.
    pub fn term(&self, index: usize) -> PhasedQubitSparsePauliView<'_> {
        debug_assert!(index < self.num_terms(), "index {index} out of bounds");
        PhasedQubitSparsePauliView {
            qubit_sparse_pauli_view: self.qubit_sparse_pauli_list.term(index),
            phase: &self.phases[index],
        }
    }

    /// Add an element to the list implied by a dense string label.
    pub fn add_dense_label<L: AsRef<[u8]>>(&mut self, label: L) -> Result<(), LabelError> {
        self.qubit_sparse_pauli_list.add_dense_label(label)?;
        let num_ys = self
            .qubit_sparse_pauli_list
            .term(self.qubit_sparse_pauli_list.num_terms() - 1)
            .num_ys();
        self.phases.push(num_ys.rem_euclid(4));
        Ok(())
    }

    /// Add a single phased sparse Pauli term to the list.
    pub fn add_phased_qubit_sparse_pauli(
        &mut self,
        term: PhasedQubitSparsePauliView,
    ) -> Result<(), ArithmeticError> {
        self.qubit_sparse_pauli_list
            .add_qubit_sparse_pauli(term.qubit_sparse_pauli_view)?;
        self.phases.push(*term.phase);
        Ok(())
    }

    // Check equality of operators
    #[cfg(feature = "python")] // Only currently used by python remove if needed from rust
    fn eq(&self, other: &PhasedQubitSparsePauliList) -> bool {
        if self.qubit_sparse_pauli_list != other.qubit_sparse_pauli_list {
            return false;
        }

        // assume here number of terms is equal
        for (self_phase, other_phase) in self.phases.iter().zip(other.phases.iter()) {
            if (self_phase - other_phase).rem_euclid(4) != 0 {
                return false;
            }
        }

        true
    }
    /// Apply a transpiler layout.
    pub fn apply_layout(
        &self,
        layout: Option<&[u32]>,
        num_qubits: u32,
    ) -> Result<Self, CoherenceError> {
        let new_qubit_sparse_pauli_list = self
            .qubit_sparse_pauli_list
            .apply_layout(layout, num_qubits)?;

        Ok(PhasedQubitSparsePauliList {
            qubit_sparse_pauli_list: new_qubit_sparse_pauli_list,
            phases: self.phases.clone(),
        })
    }
}

/// A view object onto a single term of a `QubitSparsePauliList`.
///
/// The lengths of `paulis` and `indices` are guaranteed to be created equal, but might be zero
/// (in the case that the term is the identity).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PhasedQubitSparsePauliView<'a> {
    pub qubit_sparse_pauli_view: QubitSparsePauliView<'a>,
    pub phase: &'a isize,
}
impl PhasedQubitSparsePauliView<'_> {
    /// Convert this `PhasedQubitSparsePauliView` into an owning [PhasedQubitSparsePauli] of the same data.
    pub fn to_term(&self) -> PhasedQubitSparsePauli {
        PhasedQubitSparsePauli {
            qubit_sparse_pauli: self.qubit_sparse_pauli_view.to_term(),
            phase: *self.phase,
        }
    }

    pub fn to_sparse_str(self) -> String {
        let num_ys = self.qubit_sparse_pauli_view.num_ys();
        let phase_str = match (self.phase - num_ys).rem_euclid(4) {
            0 => "",
            1 => "-i",
            2 => "-",
            3 => "i",
            _ => unreachable!("`x % 4` has only four values"),
        };

        phase_str.to_owned() + &self.qubit_sparse_pauli_view.to_sparse_str()
    }
}

/// A single phased qubit-sparse Pauli operator.
#[derive(Clone, Debug, PartialEq)]
pub struct PhasedQubitSparsePauli {
    /// The qubit sparse Pauli.
    qubit_sparse_pauli: QubitSparsePauli,
    /// ZX phase. Note that this is different from the "group phase" the user interacts with in the
    /// python interface.
    phase: isize,
}

impl PhasedQubitSparsePauli {
    /// Create a new phased qubit-sparse Pauli from the raw components that make it up.
    pub fn new(qubit_sparse_pauli: QubitSparsePauli, phase: isize) -> Self {
        Self {
            qubit_sparse_pauli,
            phase,
        }
    }

    /// Get the number of qubits the paulis are defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli.num_qubits()
    }

    /// Create the identity [QubitSparsePauli] on ``num_qubits`` qubits.
    pub fn identity(num_qubits: u32) -> Self {
        Self {
            qubit_sparse_pauli: QubitSparsePauli::identity(num_qubits),
            phase: 0,
        }
    }

    // Composition of two pauli operators self @ other.
    pub fn compose(
        &self,
        other: &PhasedQubitSparsePauli,
    ) -> Result<PhasedQubitSparsePauli, ArithmeticError> {
        if self.num_qubits() != other.num_qubits() {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: other.num_qubits(),
            });
        }

        // if either are proportional to the identity, return a clone of the other with corrected
        // phase
        if self.qubit_sparse_pauli.indices().is_empty() {
            return Ok(PhasedQubitSparsePauli {
                qubit_sparse_pauli: other.qubit_sparse_pauli.clone(),
                phase: self.phase + other.phase,
            });
        }

        if other.qubit_sparse_pauli.indices().is_empty() {
            return Ok(PhasedQubitSparsePauli {
                qubit_sparse_pauli: self.qubit_sparse_pauli.clone(),
                phase: self.phase + other.phase,
            });
        }

        let mut paulis = Vec::new();
        let mut indices = Vec::new();
        let mut new_phase = self.phase + other.phase;

        let mut self_idx = 0;
        let mut other_idx = 0;

        let self_indices = self.qubit_sparse_pauli.indices();
        let self_paulis = self.qubit_sparse_pauli.paulis();
        let other_indices = other.qubit_sparse_pauli.indices();
        let other_paulis = other.qubit_sparse_pauli.paulis();

        // iterate through each entry of self and other one time, incrementing based on the ordering
        // or equality of self_idx and other_idx, until one of them runs out of entries
        while self_idx < self_indices.len() && other_idx < other_indices.len() {
            if self_indices[self_idx] < other_indices[other_idx] {
                // if the current qubit index of self is strictly less than other, append the pauli
                paulis.push(self_paulis[self_idx]);
                indices.push(self_indices[self_idx]);
                self_idx += 1;
            } else if self_indices[self_idx] == other_indices[other_idx] {
                // if the indices are the same, perform multiplication and append if non-identity
                let new_pauli = (self_paulis[self_idx] as u8) ^ (other_paulis[other_idx] as u8);
                if new_pauli != 0 {
                    paulis.push(match new_pauli {
                        0b01 => Ok(Pauli::Z),
                        0b10 => Ok(Pauli::X),
                        0b11 => Ok(Pauli::Y),
                        _ => Err(ArithmeticError::PauliMultiplication { b: new_pauli }),
                    }?);
                    indices.push(self_indices[self_idx])
                }
                if (self_paulis[self_idx] == Pauli::X || self_paulis[self_idx] == Pauli::Y)
                    && (other_paulis[other_idx] == Pauli::Z || other_paulis[other_idx] == Pauli::Y)
                {
                    new_phase += 2;
                }
                self_idx += 1;
                other_idx += 1;
            } else {
                // same as the first if block but with roles of self and other reversed
                paulis.push(other_paulis[other_idx]);
                indices.push(other_indices[other_idx]);
                other_idx += 1;
            }
        }

        // if any entries remain in either pauli, append them
        if other_idx != other_indices.len() {
            paulis.append(&mut other_paulis[other_idx..].to_vec());
            indices.append(&mut other_indices[other_idx..].to_vec());
        } else if self_idx != self_indices.len() {
            paulis.append(&mut self_paulis[self_idx..].to_vec());
            indices.append(&mut self_indices[self_idx..].to_vec());
        }

        Ok(PhasedQubitSparsePauli {
            qubit_sparse_pauli: unsafe {
                QubitSparsePauli::new_unchecked(
                    self.num_qubits(),
                    paulis.into_boxed_slice(),
                    indices.into_boxed_slice(),
                )
            },
            phase: new_phase,
        })
    }

    /// Get a view version of this object.
    pub fn view(&self) -> PhasedQubitSparsePauliView<'_> {
        PhasedQubitSparsePauliView {
            qubit_sparse_pauli_view: self.qubit_sparse_pauli.view(),
            phase: &self.phase,
        }
    }

    /// Convert this single Pauli into a :class:`PhasedQubitSparsePauliList`.
    pub fn to_phased_qubit_sparse_pauli_list(&self) -> PhasedQubitSparsePauliList {
        PhasedQubitSparsePauliList {
            qubit_sparse_pauli_list: self.qubit_sparse_pauli.to_qubit_sparse_pauli_list(),
            phases: vec![self.phase],
        }
    }

    // Check if `self` commutes with `other`
    pub fn commutes(&self, other: &PhasedQubitSparsePauli) -> Result<bool, ArithmeticError> {
        if self.num_qubits() != other.num_qubits() {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: other.num_qubits(),
            });
        }

        self.qubit_sparse_pauli.commutes(&other.qubit_sparse_pauli)
    }

    // Check equality of operators
    #[cfg(feature = "python")] // Only currently used by python remove if needed from rust
    fn eq(&self, other: &PhasedQubitSparsePauli) -> bool {
        ((self.phase - other.phase).rem_euclid(4) == 0)
            && self.qubit_sparse_pauli == other.qubit_sparse_pauli
    }
}

/// A Pauli operator stored in a qubit-sparse format.
///
/// Representation
/// ==============
///
/// A Pauli operator is a tensor product of single-qubit Pauli operators of the form :math:`P =
/// (-i)^m \bigotimes_n A^{(n)}_i`, for :math:`A^{(n)}_i \in \{I, X, Y, Z\}` and an integer
/// :math:`m` called the phase exponent. The internal representation of a
/// :class:`PhasedQubitSparsePauli` stores only the non-identity single-qubit Pauli operators.
///
/// Internally, each single-qubit Pauli operator is stored with a numeric value. See the
/// documentation of :class:`QubitSparsePauli` for a description of the formatting of the numeric
/// value associated with each Pauli, as well as descriptions of the :attr:`paulis` and
/// :attr:`indices` attributes that store each Pauli and its associated qubit index.
///
/// Additionally, the phase of the operator can be retrieved through the :attr:`phase` attribute,
/// which returns the group phase exponent, matching the behaviour of the same attribute in
/// :class:`.Pauli`.
///
/// Construction
/// ============
///
/// :class:`PhasedQubitSparsePauli` defines several constructors.  The default constructor will
/// attempt to delegate to one of the more specific constructors, based on the type of the input.
/// You can always use the specific constructors to have more control over the construction.
///
/// .. _phased-qubit-sparse-pauli-convert-constructors:
/// .. table:: Construction from other objects
///
///   ============================  ================================================================
///   Method                        Summary
///   ============================  ================================================================
///   :meth:`from_label`            Convert a dense string label into a
///                                 :class:`~.PhasedQubitSparsePauli`.
///
///   :meth:`from_sparse_label`     Build a :class:`.PhasedQubitSparsePauli` from a tuple of a
///                                 phase, a sparse string label, and the qubits they apply to.
///
///   :meth:`from_pauli`            Raise a single :class:`~.quantum_info.Pauli` into a
///                                 :class:`.PhasedQubitSparsePauli`.
///
///   :meth:`from_raw_parts`        Build the list from :ref:`the raw data arrays
///                                 <qubit-sparse-pauli-arrays>` and the phase.
///   ============================  ================================================================
///
/// .. py:function:: PhasedQubitSparsePauli.__new__(data, /, num_qubits=None)
///
///     The default constructor of :class:`QubitSparsePauli`.
///
///     This delegates to one of :ref:`the explicit conversion-constructor methods
///     <phased-qubit-sparse-pauli-convert-constructors>`, based on the type of the ``data``
///     argument. If ``num_qubits`` is supplied and constructor implied by the type of ``data`` does
///     not accept a number, the given integer must match the input.
///
///     :param data: The data type of the input.  This can be another :class:`QubitSparsePauli`,
///         in which case the input is copied, or it can be a valid format for either
///         :meth:`from_label` or :meth:`from_sparse_label`.
///     :param int|None num_qubits: Optional number of qubits for the operator.  For most data
///         inputs, this can be inferred and need not be passed.  It is only necessary for the
///         sparse-label format.  If given unnecessarily, it must match the data input.
#[cfg(feature = "python")]
#[pyclass(
    name = "PhasedQubitSparsePauli",
    frozen,
    module = "qiskit.quantum_info",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyPhasedQubitSparsePauli {
    inner: PhasedQubitSparsePauli,
}

#[cfg(feature = "python")]
impl PyPhasedQubitSparsePauli {
    pub fn inner(&self) -> &PhasedQubitSparsePauli {
        &self.inner
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPhasedQubitSparsePauli {
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
            "unknown input format for 'PhasedQubitSparsePauli': {}",
            data.get_type().repr()?,
        )))
    }

    /// Construct a :class:`.PhasedQubitSparsePauli` from raw Numpy arrays that match the
    /// required data representation described in the class-level documentation.
    ///
    /// The data from each array is copied into fresh, growable Rust-space allocations.
    ///
    /// Args:
    ///     num_qubits: number of qubits the operator acts on.
    ///     paulis: list of the single-qubit terms.  This should be a Numpy array with dtype
    ///         :attr:`~numpy.uint8` (which is compatible with :class:`.Pauli`).
    ///     indices: sorted list of the qubits each single-qubit term corresponds to.  This should
    ///         be a Numpy array with dtype :attr:`~numpy.uint32`.
    ///     phase: The phase exponent of the operator.
    ///
    /// Examples:
    ///
    ///     Construct a :math:`Z` operator acting on qubit 50 of 100 qubits.
    ///
    ///         >>> num_qubits = 100
    ///         >>> terms = np.array([PhasedQubitSparsePauli.Pauli.Z], dtype=np.uint8)
    ///         >>> indices = np.array([50], dtype=np.uint32)
    ///         >>> phase = 0
    ///         >>> PhasedQubitSparsePauli.from_raw_parts(num_qubits, terms, indices, phase)
    ///         <PhasedQubitSparsePauli on 100 qubits: Z_50>
    #[staticmethod]
    #[pyo3(signature = (/, num_qubits, paulis, indices, phase=0))]
    fn from_raw_parts(
        num_qubits: u32,
        paulis: Vec<Pauli>,
        indices: Vec<u32>,
        phase: isize,
    ) -> PyResult<Self> {
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
        let qubit_sparse_pauli =
            QubitSparsePauli::new(num_qubits, paulis, sorted_indices.into_boxed_slice())?;
        let num_ys = qubit_sparse_pauli.view().num_ys();
        let inner = PhasedQubitSparsePauli::new(qubit_sparse_pauli, (phase + num_ys).rem_euclid(4));
        Ok(PyPhasedQubitSparsePauli { inner })
    }

    /// Construct a :class:`.PhasedQubitSparsePauli` from a single :class:`~.quantum_info.Pauli`
    /// instance.
    ///
    ///
    /// Args:
    ///     pauli (:class:`~.quantum_info.Pauli`): the single Pauli to convert.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> label = "iIYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> PhasedQubitSparsePauli.from_pauli(pauli)
    ///         <PhasedQubitSparsePauli on 5 qubits: iY_3 X_2 Z_1>
    ///         >>> assert PhasedQubitSparsePauli.from_label(label) == PhasedQubitSparsePauli.from_pauli(pauli)
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
        let group_phase = pauli
            // `Pauli`'s `_phase` is a Numpy array ...
            .getattr(intern!(py, "_phase"))?
            // ... that should have exactly 1 element ...
            .call_method0(intern!(py, "item"))?
            // ... which is some integral type.
            .extract::<isize>()?;

        let inner = PhasedQubitSparsePauli::new(
            QubitSparsePauli::new(
                num_qubits,
                paulis.into_boxed_slice(),
                indices.into_boxed_slice(),
            )?,
            group_phase,
        );
        Ok(inner.into())
    }

    /// Construct a phased qubit sparse Pauli from a sparse label, given as a tuple of an int for
    /// the phase exponent, a string of Paulis, and the indices of the corresponding qubits.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`.
    ///
    /// Args:
    ///     sparse_label (tuple[int, str, Sequence[int]]): labels and the qubits each single-qubit
    ///         term applies to.
    ///
    ///     num_qubits (int): the number of qubits the operator acts on.
    ///
    /// Examples:
    ///
    ///     Construct a simple Pauli::
    ///
    ///         >>> PhasedQubitSparsePauli.from_sparse_label(
    ///         ...     (0, "ZX", (1, 4)),
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <PhasedQubitSparsePauli on 5 qubits: X_4 Z_1>
    ///
    ///     This method can replicate the behavior of :meth:`from_label`, if the qubit-arguments
    ///     field of the tuple is set to decreasing integers::
    ///
    ///         >>> label = "XYXZ"
    ///         >>> from_label = PhasedQubitSparsePauli.from_label(label)
    ///         >>> from_sparse_label = PhasedQubitSparsePauli.from_sparse_label(
    ///         ...     (0, label, (3, 2, 1, 0)),
    ///         ...     num_qubits=4
    ///         ... )
    ///         >>> assert from_label == from_sparse_label
    #[staticmethod]
    #[pyo3(signature = (/, sparse_label, num_qubits))]
    fn from_sparse_label(
        sparse_label: (isize, String, Vec<u32>),
        num_qubits: u32,
    ) -> PyResult<Self> {
        let phase = sparse_label.0;
        let label = sparse_label.1;
        let indices = sparse_label.2;
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

        let num_ys = isize::try_from(paulis.iter().filter(|&p| *p == Pauli::Y).count())?;

        let qubit_sparse_pauli = QubitSparsePauli::new(
            num_qubits,
            paulis.into_boxed_slice(),
            sorted_indices.into_boxed_slice(),
        )?;
        let inner = PhasedQubitSparsePauli::new(qubit_sparse_pauli, phase + num_ys);
        Ok(inner.into())
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
        let qubit_sparse_pauli: QubitSparsePauli = QubitSparsePauli::from_dense_label(label)?;
        let num_ys = qubit_sparse_pauli.view().num_ys();
        let inner = PhasedQubitSparsePauli {
            qubit_sparse_pauli,
            phase: num_ys.rem_euclid(4),
        };
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
        PhasedQubitSparsePauli::identity(num_qubits).into()
    }

    /// Read-only view onto the phase.
    #[getter]
    fn get_phase(slf_: Bound<Self>) -> isize {
        let borrowed = slf_.borrow();
        let phase = borrowed.inner.phase;

        let num_ys = borrowed.inner.qubit_sparse_pauli.view().num_ys();
        (phase - num_ys).rem_euclid(4)
    }

    /// Convert this Pauli into a single element :class:`PhasedQubitSparsePauliList`.
    fn to_phased_qubit_sparse_pauli_list(&self) -> PyResult<PyPhasedQubitSparsePauliList> {
        Ok(self.inner.to_phased_qubit_sparse_pauli_list().into())
    }

    /// Composition with another :class:`PhasedQubitSparsePauli`.
    ///
    /// Args:
    ///     other (PhasedQubitSparsePauli): the qubit sparse Pauli to compose with.
    fn compose(&self, other: &PyPhasedQubitSparsePauli) -> PyResult<Self> {
        Ok(PyPhasedQubitSparsePauli {
            inner: self.inner.compose(&other.inner)?,
        })
    }

    fn __matmul__(&self, other: &PyPhasedQubitSparsePauli) -> PyResult<Self> {
        self.compose(other)
    }

    /// Check if `self`` commutes with another phased qubit sparse pauli.
    ///
    /// Args:
    ///     other (PhasedQubitSparsePauli): the phased qubit sparse Pauli to check for commutation
    ///         with.
    fn commutes(&self, other: &PyPhasedQubitSparsePauli) -> PyResult<bool> {
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
            "PhasedQubitSparsePauli",
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
            Self::get_indices(slf_.clone()),
            Self::get_phase(slf_),
        )
            .into_pyobject(py)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let paulis: &[u8] = ::bytemuck::cast_slice(self.inner.qubit_sparse_pauli.paulis());
        (
            py.get_type::<Self>().getattr("from_raw_parts")?,
            (
                self.inner.num_qubits(),
                PyArray1::from_slice(py, paulis),
                PyArray1::from_slice(py, self.inner.qubit_sparse_pauli.indices()),
                (self.inner.phase - self.inner.qubit_sparse_pauli.view().num_ys()).rem_euclid(4),
            ),
        )
            .into_pyobject(py)
    }

    /// Return a :class:`~.quantum_info.Pauli` representing the same Pauli.
    fn to_pauli<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pauli = imports::PAULI_TYPE
            .get_bound(py)
            .call1((self.inner.qubit_sparse_pauli.to_dense_label(),))?;
        pauli.setattr(
            "phase",
            self.inner.phase - self.inner.qubit_sparse_pauli.view().num_ys(),
        )?;
        Ok(pauli)
    }

    /// Get a copy of this term.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Read-only view onto the individual single-qubit terms.
    ///
    /// The only valid values in the array are those with a corresponding
    /// :class:`~PhasedQubitSparsePauli.Pauli`.
    #[getter]
    fn get_paulis(slf_: Bound<Self>) -> Bound<PyArray1<u8>> {
        let borrowed = slf_.borrow();
        let paulis = borrowed.inner.qubit_sparse_pauli.paulis();
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
        let indices = borrowed.inner.qubit_sparse_pauli.indices();
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
    // :class:`PhasedQubitSparsePauliList`.
    #[allow(non_snake_case)]
    #[classattr]
    fn Pauli(py: Python) -> PyResult<Py<PyType>> {
        PyQubitSparsePauli::Pauli(py)
    }
}

/// A list of Pauli operators with phases stored in a qubit-sparse format.
///
/// Representation
/// ==============
///
/// Each individual Pauli operator in the list is a tensor product of single-qubit Pauli operators
/// of the form :math:`P = (-i)^n\bigotimes_n A^{(n)}_i`, for :math:`A^{(n)}_i \in \{I, X, Y, Z\}`,
/// and an integer :math:`n` called the phase exponent. The
/// internal representation of a :class:`PhasedQubitSparsePauliList` stores only the non-identity
/// single-qubit Pauli operators.
///
/// Indexing
/// --------
///
/// :class:`PhasedQubitSparsePauliList` behaves as `a Python sequence
/// <https://docs.python.org/3/glossary.html#term-sequence>`__ (the standard form, not the expanded
/// :class:`collections.abc.Sequence`).  The elements of the list can be indexed by integers, as
/// well as iterated through. Whether through indexing or iterating, elements of the list are
/// returned as :class:`PhasedQubitSparsePauli` instances.
///
/// Construction
/// ============
///
/// :class:`PhasedQubitSparsePauliList` defines several constructors.  The default constructor will
/// attempt to delegate to one of the more specific constructors, based on the type of the input.
/// You can always use the specific constructors to have more control over the construction.
///
/// .. _phased-qubit-sparse-pauli-list-convert-constructors:
/// .. table:: Construction from other objects
///
///   =======================================  =====================================================
///   Method                                   Summary
///   =======================================  =====================================================
///   :meth:`from_label`                       Convert a dense string label into a single-element
///                                            :class:`.PhasedQubitSparsePauliList`.
///
///   :meth:`from_list`                        Construct from a list of dense string labels.
///
///   :meth:`from_sparse_list`                 Elements given as a list of tuples of the phase
///                                            exponent, sparse string labels, and the qubits they
///                                            apply to.
///
///   :meth:`from_pauli`                       Raise a single :class:`~.quantum_info.Pauli` into a
///                                            single-element :class:`.PhasedQubitSparsePauliList`.
///
///   :meth:`from_phased_qubit_sparse_paulis`  Construct from a list of
///                                            :class:`.PhasedQubitSparsePauli`\s.
///   =======================================  =====================================================
///
/// .. py:function:: PhasedQubitSparsePauliList.__new__(data, /, num_qubits=None)
///
///     The default constructor of :class:`PhasedQubitSparsePauliList`.
///
///     This delegates to one of :ref:`the explicit conversion-constructor methods
///     <phased-qubit-sparse-pauli-list-convert-constructors>`, based on the type of the ``data``
///     argument. If ``num_qubits`` is supplied and constructor implied by the type of ``data`` does
///     not accept a number, the given integer must match the input.
///
///     :param data: The data type of the input.  This can be another
///         :class:`PhasedQubitSparsePauliList`, in which case the input is copied, or it can be a
///         list in a valid format for either :meth:`from_list` or :meth:`from_sparse_list`.
///     :param int|None num_qubits: Optional number of qubits for the list.  For most data
///         inputs, this can be inferred and need not be passed.  It is only necessary for empty
///         lists or the sparse-list format.  If given unnecessarily, it must match the data input.
///
/// In addition to the conversion-based constructors, the method :meth:`empty` can be used to
/// construct an empty list of phased qubit-sparse Paulis acting on a given number of qubits.
///
/// Conversions
/// ===========
///
/// An existing :class:`PhasedQubitSparsePauliList` can be converted into other formats.
///
/// .. table:: Conversion methods to other observable forms.
///
///   ===========================  =================================================================
///   Method                       Summary
///   ===========================  =================================================================
///   :meth:`to_sparse_list`       Express the observable in a sparse list format with elements
///                                ``(phase, paulis, indices)``.
///   ===========================  =================================================================
#[cfg(feature = "python")]
#[pyclass(
    name = "PhasedQubitSparsePauliList",
    module = "qiskit.quantum_info",
    sequence
)]
#[derive(Debug)]
pub struct PyPhasedQubitSparsePauliList {
    // This class keeps a pointer to a pure Rust-SparseTerm and serves as interface from Python.
    pub inner: Arc<RwLock<PhasedQubitSparsePauliList>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPhasedQubitSparsePauliList {
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
            return Self::from_label(&label);
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
        if let Ok(term) = data.cast_exact::<PyPhasedQubitSparsePauli>() {
            return term.borrow().to_phased_qubit_sparse_pauli_list();
        };
        if let Ok(pauli_list) = Self::from_phased_qubit_sparse_paulis(data, num_qubits) {
            return Ok(pauli_list);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'PhasedQubitSparsePauliList': {}",
            data.get_type().repr()?,
        )))
    }

    /// Get a copy of this qubit sparse Pauli list.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> phased_qubit_sparse_pauli_list = PhasedQubitSparsePauliList.from_list(["IXZXYYZZ", "ZXIXYYZZ"])
    ///         >>> assert phased_qubit_sparse_pauli_list == phased_qubit_sparse_pauli_list.copy()
    ///         >>> assert phased_qubit_sparse_pauli_list is not phased_qubit_sparse_pauli_list.copy()
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
    /// :class:`PhasedQubitSparsePauliList` instances.
    ///
    /// Examples:
    ///
    ///     Get the empty list on 100 qubits::
    ///
    ///         >>> PhasedQubitSparsePauliList.empty(100)
    ///         <PhasedQubitSparsePauliList with 0 elements on 100 qubits: []>
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn empty(num_qubits: u32) -> Self {
        PhasedQubitSparsePauliList::empty(num_qubits).into()
    }

    /// Construct a :class:`.PhasedQubitSparsePauliList` from a single :class:`~.quantum_info.Pauli`
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
    ///         >>> PhasedQubitSparsePauliList.from_pauli(pauli)
    ///         <PhasedQubitSparsePauliList with 1 element on 5 qubits: [Y_3 X_2 Z_1]>
    ///         >>> assert PhasedQubitSparsePauliList.from_label(label) == PhasedQubitSparsePauliList.from_pauli(pauli)
    #[staticmethod]
    #[pyo3(signature = (pauli, /))]
    fn from_pauli(pauli: &Bound<PyAny>) -> PyResult<Self> {
        let x = PyPhasedQubitSparsePauli::from_pauli(pauli)?;
        x.to_phased_qubit_sparse_pauli_list()
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
    ///         >>> PhasedQubitSparsePauliList.from_label("IIIIXZI")
    ///         <PhasedQubitSparsePauliList with 1 element on 7 qubits: [X_2 Z_1]>
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> assert PhasedQubitSparsePauliList.from_label(label) == PhasedQubitSparsePauliList.from_pauli(pauli)
    ///
    /// See also:
    ///     :meth:`from_list`
    ///         A generalization of this method that constructs a list from multiple labels.
    #[staticmethod]
    #[pyo3(signature = (label, /))]
    fn from_label(label: &str) -> PyResult<Self> {
        let singleton = PyPhasedQubitSparsePauli::from_label(label)?;
        singleton.to_phased_qubit_sparse_pauli_list()
    }

    /// Construct a phased qubit-sparse Pauli list from a list of dense labels.
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
    ///         >>> PhasedQubitSparsePauliList.from_list([
    ///         ...     "IIIXX",
    ///         ...     "IIYYI",
    ///         ...     "IXXII",
    ///         ...     "ZZIII",
    ///         ... ])
    ///         <PhasedQubitSparsePauliList with 4 elements on 5 qubits:
    ///             [X_1 X_0, Y_2 Y_1, X_3 X_2, Z_4 Z_3]>
    ///
    ///     Use ``num_qubits`` to disambiguate potentially empty inputs::
    ///
    ///         >>> PhasedQubitSparsePauliList.from_list([], num_qubits=10)
    ///         <PhasedQubitSparsePauliList with 0 elements on 10 qubits: []>
    ///
    ///     This method is equivalent to calls to :meth:`from_sparse_list` with the explicit
    ///     qubit-arguments field set to decreasing integers::
    ///
    ///         >>> labels = ["XYXZ", "YYZZ", "XYXZ"]
    ///         >>> from_list = PhasedQubitSparsePauliList.from_list(labels)
    ///         >>> from_sparse_list = PhasedQubitSparsePauliList.from_sparse_list([
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
                "cannot construct a PhasedQubitSparsePauliList from an empty list without knowing `num_qubits`",
            ));
        }
        let num_qubits = match num_qubits {
            Some(num_qubits) => num_qubits,
            None => iter[0].len() as u32,
        };
        let mut inner = PhasedQubitSparsePauliList::with_capacity(num_qubits, iter.len(), 0);
        for label in iter {
            inner.add_dense_label(&label)?;
        }
        Ok(inner.into())
    }

    /// Construct a :class:`PhasedQubitSparsePauliList` out of individual
    /// :class:`PhasedQubitSparsePauli` instances.
    ///
    /// All the terms must have the same number of qubits.  If supplied, the ``num_qubits`` argument
    /// must match the terms.
    ///
    /// Args:
    ///     obj (Iterable[PhasedQubitSparsePauli]): Iterable of individual terms to build the list from.
    ///     num_qubits (int | None): The number of qubits the elements of the list should act on.
    ///         This is usually inferred from the input, but can be explicitly given to handle the
    ///         case of an empty iterable.
    ///
    /// Returns:
    ///     The corresponding list.
    #[staticmethod]
    #[pyo3(signature = (obj, /, num_qubits=None))]
    fn from_phased_qubit_sparse_paulis(
        obj: &Bound<PyAny>,
        num_qubits: Option<u32>,
    ) -> PyResult<Self> {
        let mut iter = obj.try_iter()?;
        let mut inner = match num_qubits {
            Some(num_qubits) => PhasedQubitSparsePauliList::empty(num_qubits),
            None => {
                let Some(first) = iter.next() else {
                    return Err(PyValueError::new_err(
                        "cannot construct an empty PhasedQubitSparsePauliList without knowing `num_qubits`",
                    ));
                };
                let py_term = first?.cast::<PyPhasedQubitSparsePauli>()?.borrow();
                py_term.inner.to_phased_qubit_sparse_pauli_list()
            }
        };
        for bound_py_term in iter {
            let py_term = bound_py_term?.cast::<PyPhasedQubitSparsePauli>()?.borrow();
            inner.add_phased_qubit_sparse_pauli(py_term.inner.view())?;
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
    ///         >>> pauli_list = PhasedQubitSparsePauliList.from_list(["IXXXYY", "ZZYZII"])
    ///         >>> pauli_list.clear()
    ///         >>> assert pauli_list == PhasedQubitSparsePauliList.empty(pauli_list.num_qubits)
    pub fn clear(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.clear();
        Ok(())
    }

    /// Construct a phased qubit sparse Pauli list from a list of labels and the qubits each item
    /// applies to.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`.
    ///
    /// The "labels" and "indices" fields of the tuples are associated by zipping them together.
    /// For example, this means that a call to :meth:`from_list` can be converted to the form used
    /// by this method by setting the "indices" field of each triple to ``(num_qubits-1, ..., 1,
    /// 0)``.
    ///
    /// Args:
    ///     iter (list[tuple[int, str, Sequence[int]]]): tuples of phase exponents, labels, and the
    ///         qubits each single-qubit term applies to.
    ///
    ///     num_qubits (int): the number of qubits the operators in the list act on.
    ///
    /// Examples:
    ///
    ///     Construct a simple list::
    ///
    ///         >>> PhasedQubitSparsePauliList.from_sparse_list(
    ///         ...     [(0, "ZX", (1, 4)), (1, "YY", (0, 3))],
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <PhasedQubitSparsePauliList with 2 elements on 5 qubits: [X_4 Z_1, (-i)Y_3 Y_0]>
    ///
    ///     This method can replicate the behavior of :meth:`from_list`, if the qubit-arguments
    ///     field of the tuple is set to decreasing integers::
    ///
    ///         >>> labels = ["XYXZ", "YYZZ", "XYXZ"]
    ///         >>> from_list = PhasedQubitSparsePauliList.from_list(labels)
    ///         >>> from_sparse_list = PhasedQubitSparsePauliList.from_sparse_list([
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
    fn from_sparse_list(iter: Vec<(isize, String, Vec<u32>)>, num_qubits: u32) -> PyResult<Self> {
        // separate group phases and build QubitSparsePauliList
        let mut group_phases = Vec::with_capacity(iter.len());
        let mut sub_iter = Vec::with_capacity(iter.len());
        for (phase, label, indices) in iter {
            group_phases.push(phase);
            sub_iter.push((label, indices));
        }

        let (paulis, indices, boundaries) = raw_parts_from_sparse_list(sub_iter, num_qubits)?;
        let qubit_sparse_pauli_list =
            QubitSparsePauliList::new(num_qubits, paulis, indices, boundaries)?;

        // Build zx phases
        let mut phases = Vec::with_capacity(qubit_sparse_pauli_list.num_terms());
        for (group_phase, qubit_sparse_pauli) in
            group_phases.iter().zip(qubit_sparse_pauli_list.iter())
        {
            phases.push((group_phase + qubit_sparse_pauli.num_ys()).rem_euclid(4))
        }

        let inner = PhasedQubitSparsePauliList::new(qubit_sparse_pauli_list, phases)?;
        Ok(inner.into())
    }

    /// Express the list in terms of a sparse list format.
    ///
    /// This can be seen as counter-operation of
    /// :meth:`.PhasedQubitSparsePauliList.from_sparse_list`, however the order of terms is not
    /// guaranteed to be the same at after a roundtrip to a sparse list and back.
    ///
    /// Examples:
    ///
    ///     >>> phased_qubit_sparse_list = PhasedQubitSparsePauliList.from_list(["IIXIZ", "IIZIX"])
    ///     >>> reconstructed = PhasedQubitSparsePauliList.from_sparse_list(phased_qubit_sparse_list.to_sparse_list(), qubit_sparse_list.num_qubits)
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         The constructor that can interpret these lists.
    #[pyo3(signature = ())]
    fn to_sparse_list(&self, py: Python) -> PyResult<Py<PyList>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        // turn a SparseView into a Python tuple of (phase, paulis, indices)
        let to_py_tuple = |view: PhasedQubitSparsePauliView| {
            let mut pauli_string = String::with_capacity(view.qubit_sparse_pauli_view.paulis.len());

            for bit in view.qubit_sparse_pauli_view.paulis.iter() {
                pauli_string.push_str(bit.py_label());
            }
            let py_int = PyInt::new(
                py,
                (view.phase - view.qubit_sparse_pauli_view.num_ys()).rem_euclid(4),
            )
            .unbind();
            let py_string = PyString::new(py, &pauli_string).unbind();
            let py_indices = PyList::new(py, view.qubit_sparse_pauli_view.indices.iter())?.unbind();

            PyTuple::new(
                py,
                vec![py_int.as_any(), py_string.as_any(), py_indices.as_any()],
            )
        };

        let out = PyList::empty(py);
        for view in inner.iter() {
            out.append(to_py_tuple(view)?)?;
        }
        Ok(out.unbind())
    }

    /// Return a :class:`~.quantum_info.PauliList` representing the same list of Paulis.
    fn to_pauli_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let pauli_list = imports::PAULI_LIST_TYPE
            .get_bound(py)
            .call1((inner.qubit_sparse_pauli_list.to_dense_label_list(),))?;

        let mut phases = Vec::with_capacity(inner.num_terms());
        for view in inner.iter() {
            let ys = view.qubit_sparse_pauli_view.num_ys();
            phases.push(*view.phase - ys);
        }
        pauli_list.setattr("phase", phases)?;
        Ok(pauli_list)
    }

    /// Apply a transpiler layout to this phased qubit sparse Pauli list.
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
        // input types, before calling PhasedQubitSparsePauliList.apply_layout to do the actual work.
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
                return PyPhasedQubitSparsePauli {
                    inner: inner.term(index).to_term(),
                }
                .into_bound_py_any(py);
            }
            indices => indices,
        };
        let mut out = PhasedQubitSparsePauliList::empty(inner.num_qubits());
        for index in indices.iter() {
            out.add_phased_qubit_sparse_pauli(inner.term(index))?;
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
                .map(PhasedQubitSparsePauliView::to_sparse_str)
                .collect::<Vec<_>>()
                .join(", ")
        };
        Ok(format!(
            "<PhasedQubitSparsePauliList with {str_num_terms} on {str_num_qubits}: [{str_terms}]>"
        ))
    }
}

#[cfg(feature = "python")]
impl From<PhasedQubitSparsePauli> for PyPhasedQubitSparsePauli {
    fn from(val: PhasedQubitSparsePauli) -> PyPhasedQubitSparsePauli {
        PyPhasedQubitSparsePauli { inner: val }
    }
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for PhasedQubitSparsePauli {
    type Target = PyPhasedQubitSparsePauli;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyPhasedQubitSparsePauli::from(self).into_pyobject(py)
    }
}

#[cfg(feature = "python")]
impl From<PhasedQubitSparsePauliList> for PyPhasedQubitSparsePauliList {
    fn from(val: PhasedQubitSparsePauliList) -> PyPhasedQubitSparsePauliList {
        PyPhasedQubitSparsePauliList {
            inner: Arc::new(RwLock::new(val)),
        }
    }
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for PhasedQubitSparsePauliList {
    type Target = PyPhasedQubitSparsePauliList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyPhasedQubitSparsePauliList::from(self).into_pyobject(py)
    }
}
