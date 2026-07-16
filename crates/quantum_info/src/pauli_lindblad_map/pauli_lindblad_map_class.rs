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
#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
#[cfg(feature = "python")]
use pyo3::{
    IntoPyObjectExt, PyErr,
    exceptions::{PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{PyList, PyString, PyTuple, PyType},
};
use std::collections::btree_map;

#[cfg(feature = "python")]
use std::sync::{Arc, RwLock};

use rand::prelude::*;
use rand::rngs::SysRng;
use rand_distr::Bernoulli;
use rand_pcg::Pcg64Mcg;

#[cfg(feature = "python")]
use qiskit_util::py::{PySequenceIndex, SequenceIndex};

use super::qubit_sparse_pauli::{
    ArithmeticError, CoherenceError, LabelError, Pauli, QubitSparsePauli, QubitSparsePauliList,
    QubitSparsePauliView,
};

#[cfg(feature = "python")]
use super::qubit_sparse_pauli::{
    InnerReadError, InnerWriteError, PyQubitSparsePauli, PyQubitSparsePauliList,
    raw_parts_from_sparse_list,
};

/// A Pauli Lindblad map that stores its data in a qubit-sparse format. Note that gamma,
/// probabilities, and non_negative_rates are quantities derived from rates.
///
/// See [PyPauliLindbladMap] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliLindbladMap {
    /// The rates of each abstract term in the generator sum.  This has as many elements as
    /// terms in the sum.
    rates: Vec<f64>,
    /// A list of qubit sparse Paulis corresponding to the rates
    qubit_sparse_pauli_list: QubitSparsePauliList,
    /// The gamma parameter.
    gamma: f64,
    /// Probability of application of a given Pauli operator in product form. Note that if the
    /// corresponding rate is less than 0, this is a quasi-probability.
    probabilities: Vec<f64>,
    /// List of boolean values for the statement rate >= 0 for each rate in rates.
    non_negative_rates: Vec<bool>,
}

impl PauliLindbladMap {
    pub fn new(
        rates: Vec<f64>,
        qubit_sparse_pauli_list: QubitSparsePauliList,
    ) -> Result<Self, CoherenceError> {
        if rates.len() + 1 != qubit_sparse_pauli_list.boundaries().len() {
            return Err(CoherenceError::MismatchedTermCount {
                rates: rates.len(),
                qspl: qubit_sparse_pauli_list.boundaries().len() - 1,
            });
        }

        let (gamma, probabilities, non_negative_rates) = derived_values_from_rates(&rates);

        Ok(Self {
            rates,
            qubit_sparse_pauli_list,
            gamma,
            probabilities,
            non_negative_rates,
        })
    }

    /// Get an iterator over the individual generator terms of the map.
    ///
    /// Recall that two [PauliLindbladMap]s that have different term orders can still represent the
    /// same object.  Use [canonicalize] to apply a canonical ordering to the terms.
    pub fn iter(&'_ self) -> impl ExactSizeIterator<Item = GeneratorTermView<'_>> + '_ {
        self.rates
            .iter()
            .zip(self.qubit_sparse_pauli_list.iter())
            .map(|(&rate, qubit_sparse_pauli)| GeneratorTermView {
                rate,
                qubit_sparse_pauli,
            })
    }

    /// Get the number of qubits the map is defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli_list.num_qubits()
    }

    /// Get the number of generator terms in the map.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.rates.len()
    }

    /// Get the rates of the generator terms.
    #[inline]
    pub fn rates(&self) -> &[f64] {
        &self.rates
    }

    /// Get the probabilities associated with each generator term.
    #[inline]
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Get the list of booleans for which rates are non-negative.
    #[inline]
    pub fn non_negative_rates(&self) -> &[bool] {
        &self.non_negative_rates
    }

    /// Get the indices of each [Pauli].
    #[inline]
    pub fn indices(&self) -> &[u32] {
        self.qubit_sparse_pauli_list.indices()
    }

    /// Get the boundaries of each term.
    #[inline]
    pub fn boundaries(&self) -> &[usize] {
        self.qubit_sparse_pauli_list.boundaries()
    }

    /// Get the [Pauli]s in the map.
    #[inline]
    pub fn paulis(&self) -> &[Pauli] {
        self.qubit_sparse_pauli_list.paulis()
    }

    /// Create a [PauliLindbladMap] representing the identity map on ``num_qubits`` qubits.
    pub fn identity(num_qubits: u32) -> Self {
        Self::with_capacity(num_qubits, 0, 0)
    }

    /// Add the generator term implied by a dense string label onto this map.
    pub fn add_dense_label<L: AsRef<[u8]>>(
        &mut self,
        label: L,
        rate: f64,
    ) -> Result<(), LabelError> {
        self.qubit_sparse_pauli_list.add_dense_label(label)?;

        let (g, p, pr) = derived_values_from_rate(rate);
        self.rates.push(rate);
        self.gamma *= g;
        self.probabilities.push(p);
        self.non_negative_rates.push(pr);
        Ok(())
    }

    /// Apply a transpiler layout.
    pub fn apply_layout(
        &self,
        layout: Option<&[u32]>,
        num_qubits: u32,
    ) -> Result<Self, CoherenceError> {
        let qubit_sparse_pauli_list = self
            .qubit_sparse_pauli_list
            .apply_layout(layout, num_qubits)?;
        PauliLindbladMap::new(self.rates.clone(), qubit_sparse_pauli_list)
    }

    /// Create a new identity map (with zero generator) with pre-allocated space for the given
    /// number of summands and single-qubit bit terms.
    #[inline]
    pub fn with_capacity(num_qubits: u32, num_terms: usize, num_paulis: usize) -> Self {
        let qubit_sparse_pauli_list =
            QubitSparsePauliList::with_capacity(num_qubits, num_terms, num_paulis);
        Self {
            rates: Vec::with_capacity(num_terms),
            qubit_sparse_pauli_list,
            gamma: 1.0,
            probabilities: Vec::with_capacity(num_terms),
            non_negative_rates: Vec::with_capacity(num_terms),
        }
    }

    /// Clear all the generator terms from this map, making it equal to the identity map again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction of generator terms may not need to reallocate.
    pub fn clear(&mut self) {
        self.rates.clear();
        self.gamma = 1.0;
        self.probabilities.clear();
        self.non_negative_rates.clear();
        self.qubit_sparse_pauli_list.clear();
    }

    /// Add a single generator term to this map.
    pub fn add_term(&mut self, term: GeneratorTermView) -> Result<(), ArithmeticError> {
        let term = term.to_term();
        if self.num_qubits() != term.num_qubits() {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: term.num_qubits(),
            });
        }
        let (g, p, pr) = derived_values_from_rate(term.rate);
        self.rates.push(term.rate);
        self.gamma *= g;
        self.probabilities.push(p);
        self.non_negative_rates.push(pr);
        // SAFETY: at this point we already know the term needs to be valid.
        let new_pauli = unsafe {
            QubitSparsePauli::new_unchecked(
                self.num_qubits(),
                term.qubit_sparse_pauli.paulis().to_vec().into_boxed_slice(),
                term.indices().to_vec().into_boxed_slice(),
            )
        };
        self.qubit_sparse_pauli_list
            .add_qubit_sparse_pauli(new_pauli.view())?;
        Ok(())
    }

    /// Get a view onto a representation of a single sparse term.
    ///
    /// This is effectively an indexing operation into the [PauliLindbladMap].  Recall that two
    /// [PauliLindbladMap]s that have different generator term orders can still represent the same
    /// object. Use [canonicalize] to apply a canonical ordering to the terms.
    ///
    /// # Panics
    ///
    /// If the index is out of bounds.
    pub fn term(&self, index: usize) -> GeneratorTermView<'_> {
        debug_assert!(index < self.num_terms(), "index {index} out of bounds");
        GeneratorTermView {
            rate: self.rates[index],
            qubit_sparse_pauli: self.qubit_sparse_pauli_list.term(index),
        }
    }

    /// Scale the rates by a set factor.
    pub fn scale_rates(self, scale_factor: f64) -> Self {
        let new_rates = self.rates.iter().map(|r| scale_factor * r).collect();
        PauliLindbladMap::new(new_rates, self.qubit_sparse_pauli_list.clone()).unwrap()
    }

    /// Invert the map.
    pub fn inverse(self) -> Self {
        self.scale_rates(-1.)
    }

    // Compose with another PauliLindbladMap
    pub fn compose(&self, other: &PauliLindbladMap) -> Result<PauliLindbladMap, ArithmeticError> {
        if self.num_qubits() != other.num_qubits() {
            return Err(ArithmeticError::MismatchedQubits {
                left: self.num_qubits(),
                right: other.num_qubits(),
            });
        }

        let mut rates: Vec<f64> = Vec::with_capacity(self.num_terms() + other.num_terms());
        rates.extend_from_slice(&self.rates);
        rates.extend_from_slice(&other.rates);

        let mut paulis = Vec::with_capacity(self.num_terms() + other.num_terms());
        paulis.extend_from_slice(self.paulis());
        paulis.extend_from_slice(other.paulis());

        let mut indices: Vec<u32> = Vec::with_capacity(self.num_terms() + other.num_terms());
        indices.extend_from_slice(self.indices());
        indices.extend_from_slice(other.indices());

        let mut boundaries: Vec<usize> = Vec::with_capacity(self.num_terms() + other.num_terms());
        boundaries.extend_from_slice(self.boundaries());
        let offset = self.boundaries()[self.boundaries().len() - 1];
        boundaries.extend(
            other.boundaries()[1..]
                .iter()
                .map(|boundary| offset + boundary),
        );

        let qubit_sparse_pauli_list = unsafe {
            QubitSparsePauliList::new_unchecked(self.num_qubits(), paulis, indices, boundaries)
        };
        Ok(PauliLindbladMap::new(rates, qubit_sparse_pauli_list).unwrap())
    }

    /// Drop every Pauli on the given `indices`, effectively replacing them with an identity.
    ///
    /// It ignores all the indices that are larger than `self.num_qubits`.
    pub fn drop_paulis(&self, indices: HashSet<u32>) -> Result<Self, CoherenceError> {
        let mut new_paulis: Vec<Pauli> = Vec::with_capacity(self.paulis().len());
        let mut new_indices: Vec<u32> = Vec::with_capacity(self.indices().len());
        let mut new_boundaries: Vec<usize> = Vec::with_capacity(self.boundaries().len());

        new_boundaries.push(0);
        let mut boundaries_idx = 1;
        let mut current_boundary = self.boundaries()[boundaries_idx];

        let mut num_dropped_paulis = 0;
        for (i, (&pauli, &index)) in self.paulis().iter().zip(self.indices().iter()).enumerate() {
            if current_boundary == i {
                new_boundaries.push(current_boundary - num_dropped_paulis);

                boundaries_idx += 1;
                current_boundary = self.boundaries()[boundaries_idx]
            }

            if indices.contains(&index) {
                num_dropped_paulis += 1;
            } else {
                new_indices.push(index);
                new_paulis.push(pauli);
            }
        }
        new_boundaries.push(current_boundary - num_dropped_paulis);

        Self::new(
            self.rates().to_vec(),
            QubitSparsePauliList::new(self.num_qubits(), new_paulis, new_indices, new_boundaries)
                .unwrap(),
        )
    }

    /// Drop qubits corresponding to the given `indices`.
    ///
    /// It ignores all the indices that are larger than `self.num_qubits`.
    pub fn drop_qubits(&self, indices: HashSet<u32>) -> Result<Self, CoherenceError> {
        let mut new_paulis: Vec<Pauli> = Vec::with_capacity(self.paulis().len());
        let mut new_indices: Vec<u32> = Vec::with_capacity(self.indices().len());
        let mut new_boundaries: Vec<usize> = Vec::with_capacity(self.boundaries().len());

        new_boundaries.push(0);
        let mut boundaries_idx = 1;
        let mut current_boundary = self.boundaries()[boundaries_idx];

        let mut num_dropped_paulis = 0;
        for (i, (&pauli, &index)) in self.paulis().iter().zip(self.indices().iter()).enumerate() {
            if current_boundary == i {
                new_boundaries.push(current_boundary - num_dropped_paulis);

                boundaries_idx += 1;
                current_boundary = self.boundaries()[boundaries_idx]
            }

            if indices.contains(&index) {
                num_dropped_paulis += 1;
            } else {
                new_indices.push(index - (indices.iter().filter(|&&x| x < index).count() as u32));
                new_paulis.push(pauli);
            }
        }
        new_boundaries.push(current_boundary - num_dropped_paulis);

        let new_num_qubits = self.num_qubits() - (indices.len() as u32);
        Self::new(
            self.rates().to_vec(),
            QubitSparsePauliList::new(new_num_qubits, new_paulis, new_indices, new_boundaries)
                .unwrap(),
        )
    }

    /// Compute the fidelity of the map for a single pauli
    pub fn pauli_fidelity(
        &self,
        qubit_sparse_pauli: &QubitSparsePauli,
    ) -> Result<f64, ArithmeticError> {
        let mut log_fid = 0.0;

        for generator_term in self.iter() {
            if !qubit_sparse_pauli.commutes(&generator_term.qubit_sparse_pauli.to_term())? {
                log_fid += -2.0 * generator_term.rate;
            }
        }

        Ok(log_fid.exp())
    }

    /// Sample sign and Pauli operator pairs from the map.
    /// Note that here the "sign" bool is interpreted as the exponent of (-1)^b.
    #[allow(clippy::type_complexity)]
    pub fn parity_sample_with_history(
        &self,
        num_samples: u64,
        seed: Option<u64>,
        scale: Option<f64>,
        local_scale: Option<Vec<f64>>,
    ) -> (
        Vec<bool>,
        QubitSparsePauliList,
        Vec<Vec<bool>>,
        Vec<Vec<bool>>,
    ) {
        let mut rng = match seed {
            Some(seed) => Pcg64Mcg::seed_from_u64(seed),
            None => Pcg64Mcg::try_from_rng(&mut SysRng).unwrap(),
        };
        let modified_probabilities;
        let modified_non_negative_rates;
        let (probabilities, non_negative_rates) = if local_scale.is_some() || scale.is_some() {
            let global = scale.unwrap_or(1.);
            let locals = local_scale.as_ref();
            let rates = self
                .rates
                .iter()
                .enumerate()
                .map(|(i, rate)| *rate * locals.map(|locals| locals[i]).unwrap_or(1.) * global)
                .collect::<Vec<_>>();
            (_, modified_probabilities, modified_non_negative_rates) =
                derived_values_from_rates(&rates);
            (
                modified_probabilities.as_slice(),
                modified_non_negative_rates.as_slice(),
            )
        } else {
            (
                self.probabilities.as_slice(),
                self.non_negative_rates.as_slice(),
            )
        };
        let mut random_signs = Vec::with_capacity(num_samples as usize);
        let mut random_paulis = QubitSparsePauliList::empty(self.num_qubits());
        let mut pauli_history = Vec::with_capacity(num_samples as usize);
        let mut signs_history = Vec::with_capacity(num_samples as usize);

        for _ in 0..num_samples {
            let mut random_sign = false;
            let mut random_pauli = QubitSparsePauli::identity(self.num_qubits());
            let mut inner_pauli_history = vec![false; self.qubit_sparse_pauli_list.num_terms()];
            let mut inner_signs_history = vec![false; self.qubit_sparse_pauli_list.num_terms()];

            for (((idx, probability), generator), non_negative_rate) in probabilities
                .iter()
                .enumerate()
                .zip(self.qubit_sparse_pauli_list.iter())
                .zip(non_negative_rates.iter())
            {
                // Sample true or false with given probability. If false, apply the Pauli
                if !Bernoulli::new(*probability).unwrap().sample(&mut rng) {
                    random_pauli = random_pauli.compose(&generator.to_term()).unwrap();
                    // if rate is negative, flip random_sign
                    random_sign = random_sign == *non_negative_rate;
                    // keep track of sampled generator
                    inner_pauli_history[idx] = true;
                    inner_signs_history[idx] = *non_negative_rate;
                }
            }

            random_signs.push(random_sign);
            random_paulis
                .add_qubit_sparse_pauli(random_pauli.view())
                .unwrap();
            pauli_history.push(inner_pauli_history);
            signs_history.push(inner_signs_history);
        }

        (random_signs, random_paulis, pauli_history, signs_history)
    }

    /// Reduce the map to its canonical form.
    ///
    /// This sums like terms, removing them if the final rate's absolute value is less than or equal
    /// to the tolerance.  The terms are reordered to some canonical ordering.
    ///
    /// This function is idempotent.
    pub fn simplify(&self, tol: f64) -> PauliLindbladMap {
        let mut terms = btree_map::BTreeMap::new();
        for term in self.iter() {
            terms
                .entry((
                    term.qubit_sparse_pauli.indices,
                    term.qubit_sparse_pauli.paulis,
                ))
                .and_modify(|r| *r += term.rate)
                .or_insert(term.rate);
        }

        let mut new_rates = Vec::with_capacity(self.num_terms());
        let mut new_paulis = Vec::with_capacity(self.num_terms());
        let mut new_indices = Vec::with_capacity(self.num_terms());
        let mut new_boundaries = Vec::with_capacity(self.num_terms());
        new_boundaries.push(0);
        for ((indices, paulis), r) in terms {
            // Don't add terms with zero coefficient or are pure identity
            if r.abs() <= tol || paulis.is_empty() {
                continue;
            }
            new_rates.push(r);
            new_paulis.extend_from_slice(paulis);
            new_indices.extend_from_slice(indices);
            new_boundaries.push(new_indices.len());
        }
        Self::new(
            new_rates,
            QubitSparsePauliList::new(self.num_qubits(), new_paulis, new_indices, new_boundaries)
                .unwrap(),
        )
        .unwrap()
    }
}

/// Given a rate, return the corresponding gamma, probability, and boolean for whether the rate is
/// non-negative.
fn derived_values_from_rate(rate: f64) -> (f64, f64, bool) {
    let w: f64 = 0.5 * (1.0 + (-2.0 * rate).exp());
    let g: f64 = w.abs() + (1.0 - w).abs();
    let p: f64 = w / g;
    let nnr: bool = rate >= 0.0;
    (g, p, nnr)
}

/// Return the gamma, probabilities, and non-negative rates bools for a vector of rates.
fn derived_values_from_rates(rates: &[f64]) -> (f64, Vec<f64>, Vec<bool>) {
    let mut gamma = 1.0;
    let mut probabilities = Vec::with_capacity(rates.len());
    let mut non_negative_rates = Vec::with_capacity(rates.len());

    for rate in rates {
        let (g, p, nnr) = derived_values_from_rate(*rate);
        gamma *= g;
        probabilities.push(p);
        non_negative_rates.push(nnr);
    }
    (gamma, probabilities, non_negative_rates)
}

/// A view object onto a single generator term of a `PauliLindbladMap`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GeneratorTermView<'a> {
    pub rate: f64,
    pub qubit_sparse_pauli: QubitSparsePauliView<'a>,
}
impl GeneratorTermView<'_> {
    /// Convert this `GeneratorTermView` into an owning [GeneratorTerm] of the same data.
    pub fn to_term(&self) -> GeneratorTerm {
        GeneratorTerm {
            rate: self.rate,
            qubit_sparse_pauli: self.qubit_sparse_pauli.to_term(),
        }
    }

    pub fn to_sparse_str(self) -> String {
        let rate = format!("{}", self.rate).replace('i', "j");
        let paulis = self
            .qubit_sparse_pauli
            .indices
            .iter()
            .zip(self.qubit_sparse_pauli.paulis)
            .rev()
            .map(|(i, op)| format!("{}_{}", op.py_label(), i))
            .collect::<Vec<String>>()
            .join(" ");
        format!("({rate})L({paulis})")
    }
}

/// A single term from a complete `PauliLindbladMap`.
///
/// These are typically created by indexing into or iterating through a `PauliLindbladMap`.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorTerm {
    /// The real rate of the term.
    rate: f64,
    qubit_sparse_pauli: QubitSparsePauli,
}
impl GeneratorTerm {
    pub fn num_qubits(&self) -> u32 {
        self.qubit_sparse_pauli.num_qubits()
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    pub fn indices(&self) -> &[u32] {
        self.qubit_sparse_pauli.indices()
    }

    pub fn paulis(&self) -> &[Pauli] {
        self.qubit_sparse_pauli.paulis()
    }

    pub fn view(&self) -> GeneratorTermView<'_> {
        GeneratorTermView {
            rate: self.rate,
            qubit_sparse_pauli: self.qubit_sparse_pauli.view(),
        }
    }

    /// Convert this term to a complete :class:`PauliLindbladMap`.
    pub fn to_pauli_lindblad_map(&self) -> Result<PauliLindbladMap, CoherenceError> {
        PauliLindbladMap::new(
            vec![self.rate],
            self.qubit_sparse_pauli.to_qubit_sparse_pauli_list(),
        )
    }
}

/// A single term from a complete :class:`PauliLindbladMap`.
///
/// These are typically created by indexing into or iterating through a :class:`PauliLindbladMap`.
#[cfg(feature = "python")]
#[pyclass(
    name = "GeneratorTerm",
    frozen,
    module = "qiskit.quantum_info",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
struct PyGeneratorTerm {
    inner: GeneratorTerm,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGeneratorTerm {
    // Mark the Python class as being defined "within" the `PauliLindbladMap` class namespace.
    #[classattr]
    #[pyo3(name = "__qualname__")]
    fn type_qualname() -> &'static str {
        "PauliLindbladMap.GeneratorTerm"
    }

    #[new]
    #[pyo3(signature = (/, rate, qubit_sparse_pauli))]
    fn py_new(rate: f64, qubit_sparse_pauli: &PyQubitSparsePauli) -> PyResult<Self> {
        let inner = GeneratorTerm {
            rate,
            qubit_sparse_pauli: qubit_sparse_pauli.inner().clone(),
        };
        Ok(PyGeneratorTerm { inner })
    }

    /// Convert this term to a complete :class:`PauliLindbladMap`.
    fn to_pauli_lindblad_map(&self) -> PyResult<PyPauliLindbladMap> {
        let pauli_lindblad_map = PauliLindbladMap::new(
            vec![self.inner.rate()],
            self.inner.qubit_sparse_pauli.to_qubit_sparse_pauli_list(),
        )?;
        Ok(pauli_lindblad_map.into())
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

    /// Get a copy of this term.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Read-only view onto the individual single-qubit terms.
    ///
    /// The only valid values in the array are those with a corresponding
    /// :class:`~PauliLindbladMap.Pauli`.
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

    /// The term's rate.
    #[getter]
    fn get_rate(&self) -> f64 {
        self.inner.rate()
    }

    /// The term's qubit_sparse_pauli.
    #[getter]
    fn get_qubit_sparse_pauli(&self) -> PyQubitSparsePauli {
        self.inner.qubit_sparse_pauli.clone().into()
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

    fn __getnewargs__(slf_: Bound<Self>) -> PyResult<Bound<PyTuple>> {
        let py = slf_.py();
        let borrowed = slf_.borrow();
        (
            borrowed.inner.rate,
            borrowed.inner.qubit_sparse_pauli.clone(),
        )
            .into_pyobject(py)
    }

    /// Return the pauli labels of the term as a string.
    ///
    /// The pauli labels will match the order of :attr:`.GeneratorTerm.indices`, such that the
    /// i-th character in the string is applied to the qubit index at ``term.indices[i]``. E.g. the
    /// term with operator ``X`` acting on qubit 0 and ``Y`` acting on qubit ``3`` will have
    /// ``term.indices == np.array([0, 3])`` and ``term.pauli_labels == "XY"``.
    ///
    /// Returns:
    ///     The non-identity bit terms as a concatenated string.
    fn pauli_labels<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        let string: String = self
            .inner
            .paulis()
            .iter()
            .map(|bit| bit.py_label())
            .collect();
        PyString::new(py, string.as_str())
    }
}

/// A Pauli Lindblad map stored in a qubit-sparse format.
///
/// Mathematics
/// ===========
///
/// A Pauli-Lindblad map is a linear map acting on density matrices on :math:`n`-qubits of the form:
///
/// .. math::
///
///     \Lambda\bigl[\circ\bigr] = \exp\left(\sum_{P \in K} \lambda_P (P \circ P - \circ)\right)
///
/// where :math:`K` is a subset of :math:`n`-qubit Pauli operators, and the rates, or coefficients,
/// :math:`\lambda_P` are real numbers. When all the rates :math:`\lambda_P` are non-negative, this
/// corresponds to a completely positive and trace preserving map. The sum in the exponential is
/// called the generator, and each individual term the generators. To simplify notation in the rest
/// of the documentation, we denote :math:`L(P)\bigl[\circ\bigr] = P \circ P - \circ`.
///
/// Quasi-probability representation
/// ================================
///
/// The map :math:`\Lambda` can be written as a product:
///
/// .. math::
///
///     \Lambda\bigl[\circ\bigr] = \prod_{P \in K}\exp\left(\lambda_P(P \circ P - \circ)\right).
///
/// For each :math:`P`, it holds that
///
/// .. math::
///
///     \exp\left(\lambda_P(P \circ P - \circ)\right) = \omega(\lambda_P) \circ + (1 - \omega(\lambda_P)) P \circ P,
///
/// where :math:`\omega(x) = \frac{1}{2}(1 + e^{-2 x})`. Observe that if :math:`\lambda_P \geq 0`,
/// then :math:`\omega(\lambda_P) \in (\frac{1}{2}, 1]`, and this term is a completely-positive and
/// trace-preserving map. However, if :math:`\lambda_P < 0`, then :math:`\omega(\lambda_P) > 1` and
/// the map is not completely positive or trace preserving. Letting
/// :math:`\gamma_P = \omega(\lambda_P) + |1 - \omega(\lambda_P)|`,
/// :math:`p_P = \omega(\lambda_P) / \gamma_P` and :math:`b_P \in \{0, 1\}` be :math:`1` if
/// :math:`\lambda_P < 0` and :math:`0` otherwise, we rewrite the map as:
///
/// .. math::
///
///     \omega(\lambda_P) \circ + (1 - \omega(\lambda_P)) P \circ P = \gamma_P \left(p_P \circ + (-1)^{b_P}(1 - p_P) P \circ P\right).
///
/// If :math:`\lambda_P \geq 0`, :math:`\gamma_P = 1` and the expression reduces to the standard
/// mixture of the identity map and conjugation by :math:`P`. If :math:`\lambda_P < 0`,
/// :math:`\gamma_P > 1`, and the map is a scaled difference of the identity map and conjugation by
/// :math:`P`, with probability weights (hence "quasi-probability"). Note that this is a slightly
/// different presentation than in the literature, but this notation allows us to handle both
/// non-negative and negative rates simultaneously. The overall :math:`\gamma` of the channel is the
/// product :math:`\gamma = \prod_{P \in K} \gamma_P`.
///
/// See the :meth:`.PauliLindbladMap.sample` method for the sampling procedure for this map.
///
/// Representation
/// ==============
///
/// Each individual Pauli operator in the generator is a tensor product of single-qubit Pauli
/// operators of the form :math:`P = \bigotimes_n A^{(n)}_i`, for :math:`A^{(n)}_i \in \{I, X, Y,
/// Z\}`. The internal representation of a :class:`PauliLindbladMap` stores only the non-identity
/// single-qubit Pauli operators.  This makes it significantly more efficient to represent
/// generators such as :math:`\sum_{n\in \text{qubits}} c_n L(Z^{(n)})`; for which
/// :class:`PauliLindbladMap` requires an amount of memory linear in the total number of qubits.
///
/// Internally, :class:`.PauliLindbladMap` stores an array of rates and a
/// :class:`.QubitSparsePauliList` containing the corresponding sparse Pauli operators.
/// Additionally, :class:`.PauliLindbladMap` can compute the overall channel :math:`\gamma` in the
/// :meth:`gamma` method, as well as the corresponding probabilities :math:`p_P`
/// via the :meth:`probabilities` method.
///
/// Indexing
/// --------
///
/// :class:`PauliLindbladMap` behaves as `a Python sequence
/// <https://docs.python.org/3/glossary.html#term-sequence>`__ (the standard form, not the expanded
/// :class:`collections.abc.Sequence`).  The generators of the map can be indexed by integers, and
/// iterated through to yield individual generator terms.
///
/// Each generator term appears as an instance a self-contained class.  The individual terms are
/// copied out of the base map; mutations to them will not affect the original map from which they
/// are indexed.
///
/// .. autoclass:: qiskit.quantum_info::PauliLindbladMap.GeneratorTerm
///     :members:
///
/// Construction
/// ============
///
/// :class:`PauliLindbladMap` defines several constructors.  The default constructor will attempt to
/// delegate to one of the more specific constructors, based on the type of the input.  You can
/// always use the specific constructors to have more control over the construction.
///
/// .. _pauli-lindblad-map-convert-constructors:
/// .. table:: Construction from other objects
///
///   ============================  ================================================================
///   Method                        Summary
///   ============================  ================================================================
///   :meth:`from_list`             Generators given as a list of tuples of dense string labels and
///                                 the associated rates.
///
///   :meth:`from_sparse_list`      Generators given as a list of tuples of sparse string labels,
///                                 the qubits they apply to, and their rates.
///
///   :meth:`from_terms`            Sum explicit single :class:`GeneratorTerm` instances.
///
///   :meth:`from_components`       Build from an array of rates and a
///                                 :class:`.QubitSparsePauliList` instance.
///   ============================  ================================================================
///
/// .. py:function:: PauliLindbladMap.__new__(data, /, num_qubits=None)
///
///     The default constructor of :class:`PauliLindbladMap`.
///
///     This delegates to one of :ref:`the explicit conversion-constructor methods
///     <pauli-lindblad-map-convert-constructors>`, based on the type of the ``data`` argument.  If
///     ``num_qubits`` is supplied and constructor implied by the type of ``data`` does not accept a
///     number, the given integer must match the input.
///
///     :param data: The data type of the input.  This can be another :class:`PauliLindbladMap`, in
///         which case the input is copied, or it can be a list in a valid format for either
///         :meth:`from_list` or :meth:`from_sparse_list`.
///     :param int|None num_qubits: Optional number of qubits for the map.  For most data
///         inputs, this can be inferred and need not be passed.  It is only necessary for empty
///         lists or the sparse-list format.  If given unnecessarily, it must match the data input.
///
/// In addition to the conversion-based constructors, there are also helper methods that construct
/// special forms of maps.
///
/// .. table:: Construction of special maps
///
///   ============================  ================================================================
///   Method                        Summary
///   ============================  ================================================================
///   :meth:`identity`              The identity map on a given number of qubits.
///   ============================  ================================================================
///
/// Conversions
/// ===========
///
/// An existing :class:`PauliLindbladMap` can be converted into other formats.
///
/// .. table:: Conversion methods to other forms.
///
///   ===========================  =================================================================
///   Method                       Summary
///   ===========================  =================================================================
///   :meth:`to_sparse_list`       Express the map in a sparse list format with elements
///                                ``(paulis, indices, rate)``.
///   ===========================  =================================================================
#[cfg(feature = "python")]
#[pyclass(name = "PauliLindbladMap", module = "qiskit.quantum_info", sequence)]
#[derive(Debug)]
pub struct PyPauliLindbladMap {
    // This class keeps a pointer to a pure Rust-GeneratorTerm and serves as interface from Python.
    inner: Arc<RwLock<PauliLindbladMap>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPauliLindbladMap {
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
        if let Ok(pauli_lindblad_map) = data.cast_exact::<Self>() {
            check_num_qubits(data)?;
            let borrowed = pauli_lindblad_map.borrow();
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
        if let Ok(term) = data.cast_exact::<PyGeneratorTerm>() {
            return term.borrow().to_pauli_lindblad_map();
        };
        if let Ok(pauli_lindblad_map) = Self::from_terms(data, num_qubits) {
            return Ok(pauli_lindblad_map);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'PauliLindbladMap': {}",
            data.get_type().repr()?,
        )))
    }

    #[staticmethod]
    fn from_components(
        rates: Vec<f64>,
        qubit_sparse_pauli_list: &PyQubitSparsePauliList,
    ) -> PyResult<Self> {
        let qubit_sparse_pauli_list = qubit_sparse_pauli_list
            .inner
            .read()
            .map_err(|_| InnerReadError)?;
        let inner = PauliLindbladMap::new(rates, qubit_sparse_pauli_list.clone())?;

        Ok(inner.into())
    }

    /// Construct a Pauli Lindblad map from a list of dense generator labels and rates.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_list`. In this dense form, you must supply
    /// all identities explicitly in each label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     iter (list[tuple[str, float]]): Pairs of labels and their associated rates in the
    ///         generator sum.
    ///     num_qubits (int | None): It is not necessary to specify this if you are sure that
    ///         ``iter`` is not an empty sequence, since it can be inferred from the label lengths.
    ///         If ``iter`` may be empty, you must specify this argument to disambiguate how many
    ///         qubits the map acts on.  If this is given and ``iter`` is not empty, the value
    ///         must match the label lengths.
    ///
    /// Examples:
    ///
    ///     Construct a Pauli Lindblad map from a list of labels::
    ///
    ///         >>> PauliLindbladMap.from_list([
    ///         ...     ("IIIXX", 1.0),
    ///         ...     ("IIYYI", 1.0),
    ///         ...     ("IXXII", -0.5),
    ///         ...     ("ZZIII", -0.25),
    ///         ... ])
    ///         <PauliLindbladMap with 4 terms on 5 qubits:
    ///             (1)L(X_1 X_0) + (1)L(Y_2 Y_1) + (-0.5)L(X_3 X_2) + (-0.25)L(Z_4 Z_3)>
    ///
    ///     Use ``num_qubits`` to disambiguate potentially empty inputs::
    ///
    ///         >>> PauliLindbladMap.from_list([], num_qubits=10)
    ///         <PauliLindbladMap with 0 terms on 10 qubits: 0.0>
    ///
    ///     This method is equivalent to calls to :meth:`from_sparse_list` with the explicit
    ///     qubit-arguments field set to decreasing integers::
    ///
    ///         >>> labels = ["XYXZ", "YYZZ", "XYXZ"]
    ///         >>> rates = [1.5, 2.0, -0.5]
    ///         >>> from_list = PauliLindbladMap.from_list(list(zip(labels, rates)))
    ///         >>> from_sparse_list = PauliLindbladMap.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0), rate)
    ///         ...     for label, rate in zip(labels, rates)
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         Construct the map from a list of labels without explicit identities, but with
    ///         the qubits each single-qubit generator term applies to listed explicitly.
    #[staticmethod]
    #[pyo3(signature = (iter, /, *, num_qubits=None))]
    fn from_list(iter: Vec<(String, f64)>, num_qubits: Option<u32>) -> PyResult<Self> {
        if iter.is_empty() && num_qubits.is_none() {
            return Err(PyValueError::new_err(
                "cannot construct a PauliLindbladMap from an empty list without knowing `num_qubits`",
            ));
        }
        let num_qubits = match num_qubits {
            Some(num_qubits) => num_qubits,
            None => iter[0].0.len() as u32,
        };
        let mut inner = PauliLindbladMap::with_capacity(num_qubits, iter.len(), 0);
        for (label, rate) in iter {
            inner.add_dense_label(&label, rate)?;
        }
        Ok(inner.into())
    }

    /// Get the identity map on the given number of qubits.
    ///
    /// The identity map contains no generator terms, and is the identity element for composition of
    /// two :class:`PauliLindbladMap` instances; anything composed with the identity map is equal to
    /// itself.
    ///
    /// Examples:
    ///
    ///     Get the identity map on 100 qubits::
    ///
    ///         >>> PauliLindbladMap.identity(100)
    ///         <PauliLindbladMap with 0 terms on 100 qubits: 0.0>
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn identity(num_qubits: u32) -> Self {
        PauliLindbladMap::identity(num_qubits).into()
    }

    /// Construct a :class:`PauliLindbladMap` out of individual terms.
    ///
    /// All the terms must have the same number of qubits.  If supplied, the ``num_qubits`` argument
    /// must match the terms.
    ///
    /// No simplification is done as part of the map creation.
    ///
    /// Args:
    ///     obj (Iterable[Term]): Iterable of individual terms to build the map generator from.
    ///     num_qubits (int | None): The number of qubits the map should act on.  This is
    ///         usually inferred from the input, but can be explicitly given to handle the case
    ///         of an empty iterable.
    ///
    /// Returns:
    ///     The corresponding map.
    #[staticmethod]
    #[pyo3(signature = (obj, /, num_qubits=None))]
    fn from_terms(obj: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let mut iter = obj.try_iter()?;
        let mut inner = match num_qubits {
            Some(num_qubits) => PauliLindbladMap::identity(num_qubits),
            None => {
                let Some(first) = iter.next() else {
                    return Err(PyValueError::new_err(
                        "cannot construct a PauliLindbladMap from an empty list without knowing `num_qubits`",
                    ));
                };
                let py_term = first?.cast::<PyGeneratorTerm>()?.borrow();
                py_term.inner.to_pauli_lindblad_map()?
            }
        };
        for bound_py_term in iter {
            let py_term = bound_py_term?.cast::<PyGeneratorTerm>()?.borrow();
            inner.add_term(py_term.inner.view())?;
        }
        Ok(inner.into())
    }

    /// Clear all the generator terms from this map, making it equal to the identity map again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction operations resulting from composition may not need to reallocate.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> pauli_lindblad_map = PauliLindbladMap.from_list([("IXXXYY", 2.0), ("ZZYZII", -1)])
    ///         >>> pauli_lindblad_map.clear()
    ///         >>> assert pauli_lindblad_map == PauliLindbladMap.identity(pauli_lindblad_map.py_num_qubits())
    pub fn clear(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.clear();
        Ok(())
    }

    /// Construct a Pauli Lindblad map from a list of labels, the qubits each item applies to, and
    /// the rate of the whole term.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`.
    ///
    /// The "labels" and "indices" fields of the triples are associated by zipping them together.
    /// For example, this means that a call to :meth:`from_list` can be converted to the form used
    /// by this method by setting the "indices" field of each triple to ``(num_qubits-1, ..., 1,
    /// 0)``.
    ///
    /// Args:
    ///     iter (list[tuple[str, Sequence[int], float]]): triples of labels, the qubits
    ///         each single-qubit term applies to, and the rate of the entire term.
    ///
    ///     num_qubits (int): the number of qubits the map acts on.
    ///
    /// Examples:
    ///
    ///     Construct a simple map::
    ///
    ///         >>> PauliLindbladMap.from_sparse_list(
    ///         ...     [("ZX", (1, 4), 1.0), ("YY", (0, 3), 2)],
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <PauliLindbladMap with 2 terms on 5 qubits: (1)L(X_4 Z_1) + (2)L(Y_3 Y_0)>
    ///
    ///     This method can replicate the behavior of :meth:`from_list`, if the qubit-arguments
    ///     field of the triple is set to decreasing integers::
    ///
    ///         >>> labels = ["XYXZ", "YYZZ", "XYXZ"]
    ///         >>> rates = [1.5, 2.0, -0.5]
    ///         >>> from_list = PauliLindbladMap.from_list(list(zip(labels, rates)))
    ///         >>> from_sparse_list = PauliLindbladMap.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0), rate)
    ///         ...     for label, rate in zip(labels, rates)
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`to_sparse_list`
    ///         The reverse of this method.
    #[staticmethod]
    #[pyo3(signature = (iter, /, num_qubits))]
    fn from_sparse_list(iter: Vec<(String, Vec<u32>, f64)>, num_qubits: u32) -> PyResult<Self> {
        let rates = iter.iter().map(|(_, _, rate)| *rate).collect();
        let op_iter = iter
            .iter()
            .map(|(label, indices, _)| (label.clone(), indices.clone()))
            .collect();
        let (paulis, indices, boundaries) = raw_parts_from_sparse_list(op_iter, num_qubits)?;
        let qubit_sparse_pauli_list =
            QubitSparsePauliList::new(num_qubits, paulis, indices, boundaries)?;
        let inner: PauliLindbladMap = PauliLindbladMap::new(rates, qubit_sparse_pauli_list)?;
        Ok(inner.into())
    }

    /// Get a copy of this Pauli Lindblad map.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> pauli_lindblad_map = PauliLindbladMap.from_list([("IXZXYYZZ", 2.5), ("ZXIXYYZZ", 0.5)])
    ///         >>> assert pauli_lindblad_map == pauli_lindblad_map.copy()
    ///         >>> assert pauli_lindblad_map is not pauli_lindblad_map.copy()
    fn copy(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.clone().into())
    }

    /// The number of qubits the map acts on.
    ///
    /// This is not inferable from any other shape or values, since identities are not stored
    /// explicitly.
    #[getter]
    #[inline]
    pub fn num_qubits(&self) -> PyResult<u32> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_qubits())
    }

    /// The number of generator terms in the exponent for this map.
    #[getter]
    #[inline]
    pub fn num_terms(&self) -> PyResult<usize> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_terms())
    }

    /// Calculate the :math:`\gamma` for the map.
    #[inline]
    fn gamma(&self) -> PyResult<f64> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.gamma)
    }

    /// The rates for the map.
    #[getter]
    fn get_rates(slf_: Bound<Self>) -> Bound<PyArray1<f64>> {
        let borrowed = &slf_.borrow();
        let inner = borrowed.inner.read().unwrap();
        let rates = inner.rates();
        let arr = ::ndarray::aview1(rates);
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[u32]>` allocation (including dropping or reallocating it) other than the entire
        // object getting dropped, which Python will keep safe.
        let out = unsafe { PyArray1::borrow_from_array(&arr, slf_.into_any()) };
        out.readwrite().make_nonwriteable();
        out
    }

    /// Calculate the probabilities :math:`p_P` for the map.
    /// These can be interpreted as the probabilities each generator is not applied,
    /// and are defined to be independent of the sign of each Lindblad rate.
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let inner = self.inner.read().unwrap();
        inner.probabilities().to_pyarray(py)
    }

    /// Get a copy of the map's qubit sparse pauli list.
    fn get_qubit_sparse_pauli_list_copy(&self) -> PyQubitSparsePauliList {
        let inner = self.inner.read().unwrap();
        inner.qubit_sparse_pauli_list.clone().into()
    }

    /// Get the generators of the map.
    ///
    /// This is an alias for :meth:`get_qubit_sparse_pauli_list_copy`, providing a more
    /// convenient name that aligns with the naming conventions used in Aer and Runtime
    /// for similar classes.
    ///
    /// Returns:
    ///     QubitSparsePauliList: A copy of the map's generator terms.
    fn generators(&self) -> PyQubitSparsePauliList {
        self.get_qubit_sparse_pauli_list_copy()
    }

    /// Express the map in terms of a sparse list format.
    ///
    /// This can be seen as counter-operation of :meth:`.PauliLindbladMap.from_sparse_list`, however
    /// the order of terms is not guaranteed to be the same at after a roundtrip to a sparse
    /// list and back.
    ///
    /// Examples:
    ///
    ///     >>> pauli_lindblad_map = PauliLindbladMap.from_list([("IIXIZ", 2), ("IIZIX", 3)])
    ///     >>> reconstructed = PauliLindbladMap.from_sparse_list(pauli_lindblad_map.to_sparse_list(), pauli_lindblad_map.num_qubits)
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         The constructor that can interpret these lists.
    #[pyo3(signature = ())]
    fn to_sparse_list(&self, py: Python) -> PyResult<Py<PyList>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        // turn a SparseView into a Python tuple of (bit terms, indices, rate)
        let to_py_tuple = |view: GeneratorTermView| {
            let mut pauli_string = String::with_capacity(view.qubit_sparse_pauli.paulis.len());

            for bit in view.qubit_sparse_pauli.paulis.iter() {
                pauli_string.push_str(bit.py_label());
            }
            let py_string = PyString::new(py, &pauli_string).unbind();
            let py_indices = PyList::new(py, view.qubit_sparse_pauli.indices.iter())?.unbind();
            let py_rate = view.rate.into_py_any(py)?;

            PyTuple::new(py, vec![py_string.as_any(), py_indices.as_any(), &py_rate])
        };

        let out = PyList::empty(py);
        for view in inner.iter() {
            out.append(to_py_tuple(view)?)?;
        }
        Ok(out.unbind())
    }

    /// Apply a transpiler layout to this Pauli Lindblad map.
    ///
    /// This enables remapping of qubit indices, e.g. if the map is defined in terms of virtual
    /// qubit labels.
    ///
    /// Args:
    ///     layout (TranspileLayout | list[int] | None): The layout to apply.  Most uses of this
    ///         function should pass the :attr:`.QuantumCircuit.layout` field from a circuit that
    ///         was transpiled for hardware.  In addition, you can pass a list of new qubit indices.
    ///         If given as explicitly ``None``, no remapping is applied (but you can still use
    ///         ``num_qubits`` to expand the map).
    ///     num_qubits (int | None): The number of qubits to expand the map to.  If not
    ///         supplied, the output will be as wide as the given :class:`.TranspileLayout`, or the
    ///         same width as the input if the ``layout`` is given in another form.
    ///
    /// Returns:
    ///     A new :class:`PauliLindbladMap` with the provided layout applied.
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
        // input types, before calling PauliLindbladMap.apply_layout to do the actual work.
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

    /// Return a new :class:`PauliLindbladMap` with rates scaled by `scale_factor`.
    ///
    /// Args:
    ///     scale_factor (float): the scaling coefficient.
    #[pyo3(signature = (scale_factor))]
    fn scale_rates<'py>(
        &self,
        py: Python<'py>,
        scale_factor: f64,
    ) -> PyResult<Bound<'py, PyPauliLindbladMap>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let scaled = inner.clone().scale_rates(scale_factor);
        scaled.into_pyobject(py)
    }

    /// Return a new :class:`PauliLindbladMap` that is the mathematical inverse of `self`.
    #[pyo3(signature = ())]
    fn inverse<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyPauliLindbladMap>> {
        self.scale_rates(py, -1.)
    }

    /// Drop Paulis out of this Pauli Lindblad map.
    ///
    /// Drop every Pauli on the given `indices`, effectively replacing them with an identity.
    ///
    /// The resulting map may contain duplicates, which can be removed using the :meth:`PauliLindbladMap.simplify`
    /// method.
    ///
    /// Args:
    ///     indices (Sequence[int]): The indices for which Paulis must be dropped.
    ///
    /// Returns:
    ///     A new Pauli Lindblad map where every Pauli on the given `indices` has been dropped.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> pauli_map_in = PauliLindbladMap.from_list([("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)])
    ///         >>> pauli_map_out = pauli_map_in.keep_paulis([1, 2, 4])
    ///         >>> assert pauli_map_out == PauliLindbladMap.from_list([("IXIII", 2.0), ("IIIIZ", 0.5), ("IIIIY", -0.25)])
    #[pyo3(signature = (/, indices))]
    pub fn drop_paulis(&self, indices: Vec<u32>) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        let max_index = match indices.iter().max() {
            Some(&index) => index,
            None => 0,
        };
        if max_index >= inner.num_qubits() {
            let num_qubits = inner.num_qubits();
            return Err(PyValueError::new_err(format!(
                "cannot drop Paulis for index {max_index} in a {num_qubits}-qubit PauliLindbladMap"
            )));
        }

        Ok(inner.drop_paulis(indices.into_iter().collect())?.into())
    }

    /// Keep every Pauli on the given `indices` and drop all others.
    ///
    /// This is equivalent to using :meth:`PauliLindbladMap.drop_paulis` on the complement set of indices.
    ///
    /// Args:
    ///     indices (Sequence[int]): The indices for which Paulis must be kept.
    ///
    /// Returns:
    ///     A new Pauli Lindblad map where every Pauli on the given `indices` has been kept and all other
    ///     Paulis have been dropped.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> pauli_map_in = PauliLindbladMap.from_list([("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)])
    ///         >>> pauli_map_out = pauli_map_in.keep_paulis([0, 3])
    ///         >>> assert pauli_map_out == PauliLindbladMap.from_list([("IXIII", 2.0), ("IIIIZ", 0.5), ("IIIIY", -0.25)])
    #[pyo3(signature = (/, indices))]
    pub fn keep_paulis(&self, indices: Vec<u32>) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        let max_index = match indices.iter().max() {
            Some(&index) => index,
            None => 0,
        };
        if max_index >= inner.num_qubits() {
            let num_qubits = inner.num_qubits();
            return Err(PyValueError::new_err(format!(
                "cannot keep Paulis for index {max_index} in a {num_qubits}-qubit PauliLindbladMap"
            )));
        }

        Ok(inner
            .drop_paulis(
                (0..self.num_qubits()?)
                    .filter(|index| !indices.contains(index))
                    .collect(),
            )?
            .into())
    }

    /// Drop qubits out of this Pauli Lindblad map, effectively performing a trace-out operation.
    ///
    /// The resulting map may contain duplicates, which can be removed using the :meth:`PauliLindbladMap.simplify`
    /// method.
    ///
    /// Args:
    ///     indices (Sequence[int]): The indices of the qubits to trace out.
    ///
    /// Returns:
    ///     A new Pauli Lindblad map where every Pauli on the given `indices` has been traced out.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> pauli_map_in = PauliLindbladMap.from_list([("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)])
    ///         >>> pauli_map_out = pauli_map_in.drop_qubits([1, 2, 4])
    ///         >>> assert pauli_map_out == PauliLindbladMap.from_list([("XI", 2.0), ("IZ", 0.5), ("IY", -0.25)])
    #[pyo3(signature = (/, indices))]
    pub fn drop_qubits(&self, indices: Vec<u32>) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        let max_index = match indices.iter().max() {
            Some(&index) => index,
            None => 0,
        };
        if max_index >= inner.num_qubits() {
            let num_qubits = inner.num_qubits();
            return Err(PyValueError::new_err(format!(
                "cannot drop qubits for index {max_index} in a {num_qubits}-qubit PauliLindbladMap"
            )));
        }

        if (indices.len() as u32) == inner.num_qubits() {
            return Err(PyValueError::new_err(
                "cannot drop every qubit in the given PauliLindbladMap",
            ));
        }

        Ok(inner.drop_qubits(indices.into_iter().collect())?.into())
    }

    /// Keep every qubit on the given `indices` and trace out all other qubits.
    ///
    /// This is equivalent to using :meth:`PauliLindbladMap.drop_qubits` on the complement set of indices.
    ///
    /// Args:
    ///     indices (Sequence[int]): The indices for which qubits must be kept.
    ///
    /// Returns:
    ///     A new Pauli Lindblad map where every qubit on the given `indices` has been kept and all other
    ///     qubits have been traced out.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> pauli_map_in = PauliLindbladMap.from_list([("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)])
    ///         >>> pauli_map_out = pauli_map_in.keep_qubits([0, 3])
    ///         >>> assert pauli_map_out == PauliLindbladMap.from_list([("XI", 2.0), ("IZ", 0.5), ("IY", -0.25)])
    #[pyo3(signature = (/, indices))]
    pub fn keep_qubits(&self, indices: Vec<u32>) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        let max_index = match indices.iter().max() {
            Some(&index) => index,
            None => 0,
        };
        if max_index >= inner.num_qubits() {
            let num_qubits = inner.num_qubits();
            return Err(PyValueError::new_err(format!(
                "cannot keep qubits for index {max_index} in a {num_qubits}-qubit PauliLindbladMap"
            )));
        }

        if indices.is_empty() {
            return Err(PyValueError::new_err(
                "cannot drop every qubit in the given PauliLindbladMap",
            ));
        }

        Ok(inner
            .drop_qubits(
                (0..self.num_qubits()?)
                    .filter(|index| !indices.contains(index))
                    .collect(),
            )?
            .into())
    }

    /// Sample sign and Pauli operator pairs from the map. Note that the boolean sign convention in
    /// this method is non-standard. The preferred method for this kind of sampling is
    /// :meth:`.PauliLindbladMap.parity_sample`, which is also more featureful.
    ///
    /// Each sign is represented by a boolean, with ``True`` representing ``+1``, and ``False``
    /// representing ``-1``.
    ///
    /// Given the quasi-probability representation given in the class-level documentation, each
    /// sample is drawn via the following process:
    ///
    /// * Initialize the sign boolean, and a :class:`~.QubitSparsePauli` instance to the identity
    ///   operator.
    ///
    /// * Iterate through each Pauli in the map. Using the pseudo-probability associated with
    ///   each operator, randomly choose between applying the operator or not.
    ///
    /// * If the operator is applied, update the :class`QubitSparsePauli` by multiplying it with
    ///   the Pauli. If the rate associated with the Pauli is negative, flip the sign boolean.
    ///
    /// The results are returned as a 1d array of booleans, and the corresponding sampled qubit
    /// sparse Paulis in the form of a :class:`~.QubitSparsePauliList`.
    ///
    /// Args:
    ///     num_samples (int): Number of samples to draw.
    ///     seed (int): Random seed.
    ///
    /// Returns:
    ///     signs, qubit_sparse_pauli_list: The boolean array of signs and the list of qubit sparse
    ///     paulis.
    #[pyo3(signature = (num_samples, seed=None))]
    pub fn signed_sample<'py>(
        &self,
        py: Python<'py>,
        num_samples: u64,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let (signs, paulis, _, _) =
            py.detach(|| inner.parity_sample_with_history(num_samples, seed, None, None));

        let signs = PyArray1::from_vec(py, signs.iter().map(|b| !b).collect());
        let paulis = paulis.into_pyobject(py).unwrap();

        (signs, paulis).into_pyobject(py)
    }

    /// Sample sign and Pauli operator pairs from the map.
    ///
    /// Each sign is represented by a boolean, with ``True`` representing ``-1``, and ``False``
    /// representing ``+1``.
    ///
    /// Given the quasi-probability representation given in the class-level documentation, each
    /// sample is drawn via the following process:
    ///
    /// * Initialize the sign boolean, and a :class:`~.QubitSparsePauli` instance to the identity
    ///   operator.
    ///
    /// * Iterate through each Pauli in the map. Using the pseudo-probability associated with
    ///   each operator, randomly choose between applying the operator or not.
    ///
    /// * If the operator is applied, update the :class`QubitSparsePauli` by multiplying it with
    ///   the Pauli. If the rate associated with the Pauli is negative, flip the sign boolean.
    ///
    /// The results are returned as a 1d array of booleans, and the corresponding sampled qubit
    /// sparse Paulis in the form of a :class:`~.QubitSparsePauliList`.
    ///
    /// The arguments ``scale`` and ``local_scale`` can be used to change the underlying rates used
    /// in the sampling process without modifying current instance or requiring creating a new one.
    /// The ``scale`` argument scales all rates by a fixed float, and ``local_scale`` scales rates
    /// on a term-by-term basis.
    ///
    /// Args:
    ///     num_samples (int): Number of samples to draw.
    ///     seed (int): Random seed.
    ///     scale (float): Scale to apply to all rates.
    ///     local_scale (list[float]): Local scale to apply on a term-by-term basis.
    ///
    /// Returns:
    ///     signs, qubit_sparse_pauli_list: The boolean array of signs and the list of qubit sparse
    ///     paulis.
    #[pyo3(signature = (num_samples, seed=None, scale=None, local_scale=None))]
    pub fn parity_sample<'py>(
        &self,
        py: Python<'py>,
        num_samples: u64,
        seed: Option<u64>,
        scale: Option<f64>,
        local_scale: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let (signs, paulis, _, _) =
            py.detach(|| inner.parity_sample_with_history(num_samples, seed, scale, local_scale));

        let signs = PyArray1::from_vec(py, signs);
        let paulis = paulis.into_pyobject(py).unwrap();

        (signs, paulis).into_pyobject(py)
    }

    /// Sample sign and Pauli operator pairs from the map, preserving the sampled generators.
    ///
    /// This method is identical to :meth:`parity_sample` except for also returning the information
    /// which :meth:`generators` were actually sampled to yield the final Pauli operator and sign.
    ///
    ///
    /// Args:
    ///     num_samples (int): Number of samples to draw.
    ///     seed (int): Random seed.
    ///     scale (float): Scale to apply to all rates.
    ///     local_scale (list[float]): Local scale to apply on a term-by-term basis.
    ///
    /// Returns:
    ///     signs, qubit_sparse_pauli_list, pauli_history, signs_history: The boolean array of
    ///         signs, the list of qubit sparse paulis, the two dimensional boolean array
    ///         indicating which :meth:`generators` were sampled and the two dimensional boolean
    ///         array indicating their signs.
    #[pyo3(signature = (num_samples, seed=None, scale=None, local_scale=None))]
    pub fn parity_sample_with_history<'py>(
        &self,
        py: Python<'py>,
        num_samples: u64,
        seed: Option<u64>,
        scale: Option<f64>,
        local_scale: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let (signs, paulis, pauli_history, signs_history) =
            py.detach(|| inner.parity_sample_with_history(num_samples, seed, scale, local_scale));

        let signs = PyArray1::from_vec(py, signs);
        let paulis = paulis.into_pyobject(py).unwrap();
        let pauli_history = PyArray2::from_vec2(py, &pauli_history).unwrap();
        let signs_history = PyArray2::from_vec2(py, &signs_history).unwrap();

        (signs, paulis, pauli_history, signs_history).into_pyobject(py)
    }

    /// For :class:`.PauliLindbladMap` instances with purely non-negative rates, sample Pauli
    /// operators from the map. If the map has negative rates, use
    /// :meth:`.PauliLindbladMap.parity_sample`.
    ///
    /// Given the quasi-probability representation given in the class-level documentation, each
    /// sample is drawn via the following process:
    ///
    /// * Initialize a :class`~.QubitSparsePauli` instance to the identity operator.
    ///
    /// * Iterate through each Pauli in the map. Using the pseudo-probability associated with
    ///   each operator, randomly choose between applying the operator or not.
    ///
    /// * If the operator is applied, update the :class`QubitSparsePauli` by multiplying it with
    ///   the Pauli.
    ///
    /// The sampled qubit sparse Paulis are returned in the form of a
    /// :class:`~.QubitSparsePauliList`.
    ///
    /// Args:
    ///     num_samples (int): Number of samples to draw.
    ///     seed (int): Random seed. Defaults to ``None``.
    ///
    /// Returns:
    ///     qubit_sparse_pauli_list: The list of qubit sparse paulis.
    ///
    /// Raises:
    ///     ValueError: If any of the rates in the map are negative.
    #[pyo3(signature = (num_samples, seed=None))]
    pub fn sample<'py>(
        &self,
        py: Python<'py>,
        num_samples: u64,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyQubitSparsePauliList>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        for non_negative in inner.non_negative_rates.iter() {
            if !non_negative {
                return Err(PyValueError::new_err(
                    "PauliLindbladMap.sample called for a map with negative rates. Use PauliLindbladMap.parity_sample",
                ));
            }
        }

        let (_, paulis, _, _) =
            py.detach(|| inner.parity_sample_with_history(num_samples, seed, None, None));

        paulis.into_pyobject(py)
    }

    /// Sum any like terms in the generator, removing them if the resulting rate has an absolute
    /// value within tolerance of zero. This also removes terms whose Pauli operator is proportional
    /// to the identity, as the corresponding generator is actually the zero map.
    ///
    /// As a side effect, this sorts the generators into a fixed canonical order.
    ///
    /// .. note::
    ///
    ///     When using this for equality comparisons, note that floating-point rounding and the
    ///     non-associativity of floating-point addition may cause non-zero coefficients of summed
    ///     terms to compare unequal.  To compare two observables up to a tolerance, it is safest to
    ///     compare the canonicalized difference of the two observables to zero.
    ///
    /// Args:
    ///     tol (float): after summing like terms, any rates whose absolute value is less
    ///         than the given absolute tolerance will be suppressed from the output.
    ///
    /// Examples:
    ///
    ///     Using :meth:`simplify` to compare two operators that represent the same map, but
    ///     would compare unequal due to the structural tests by default::
    ///
    ///         >>> base = PauliLindbladMap.from_sparse_list([
    ///         ...     ("XZ", (2, 1), 1e-10),  # value too small
    ///         ...     ("XX", (3, 1), 2),
    ///         ...     ("XX", (3, 1), 2),      # can be combined with the above
    ///         ...     ("ZZ", (3, 1), 0.5),    # out of order compared to `expected`
    ///         ... ], num_qubits=5)
    ///         >>> expected = PauliLindbladMap.from_list([("IZIZI", 0.5), ("IXIXI", 4)])
    ///         >>> assert base != expected  # non-canonical comparison
    ///         >>> assert base.simplify() == expected.simplify()
    ///
    ///     Note that in the above example, the coefficients are chosen such that all floating-point
    ///     calculations are exact, and there are no intermediate rounding or associativity
    ///     concerns.  If this cannot be guaranteed to be the case, the safer form is::
    ///
    ///         >>> left = PauliLindbladMap.from_list([("XYZ", 1.0/3.0)] * 3)   # sums to 1.0
    ///         >>> right = PauliLindbladMap.from_list([("XYZ", 1.0/7.0)] * 7)  # doesn't sum to 1.0
    ///         >>> assert left.simplify() != right.simplify()
    ///         >>> assert left.compose(right.inverse()).simplify() == PauliLindbladMap.identity(left.num_qubits)
    #[pyo3(
        signature = (/, tol=1e-8),
    )]
    fn simplify(&self, tol: f64) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let simplified = inner.simplify(tol);
        Ok(simplified.into())
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

    /// Compose with another :class:`PauliLindbladMap`.
    ///
    /// This appends the internal arrays of self and other, and therefore results in a map with
    /// whose enumerated terms are those of self followed by those of other.
    ///
    /// Args:
    ///     other (PauliLindbladMap): the Pauli Lindblad map to compose with.
    fn compose<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyPauliLindbladMap>> {
        let py = other.py();
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let Some(other) = coerce_to_map(other)? else {
            return Err(PyTypeError::new_err(format!(
                "unknown type for compose: {}",
                other.get_type().repr()?
            )));
        };
        let other = other.borrow();
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        let composed = inner.compose(&other_inner)?;
        composed.into_pyobject(py)
    }

    /// Compute the Pauli fidelity of this map for a qubit sparse Pauli.
    ///
    /// For a Pauli :math:`Q`, the fidelity with respect to the Pauli Lindblad map
    /// :math:`\Lambda` is the real number :math:`f(Q)` for which :math:`\Lambda(Q) = f(Q) Q`. I.e.
    /// every Pauli is an eigenvector of the linear map :math:`\Lambda`, and the fidelity is the
    /// corresponding eigenvalue. For a Pauli Lindblad map with generator set :math:`K` and rate
    /// function :math:`\lambda : K \rightarrow \mathbb{R}`, the pauli fidelity mathematically is
    ///
    /// .. math::
    ///
    ///     f(Q) = \exp\left(-2 \sum_{P \in K} \lambda(P) \langle P, Q\rangle_{sp}\right),
    ///
    /// where :math:`\langle P, Q\rangle_{sp}` is :math:`0` if :math:`P` and :math:`Q` commute, and
    /// :math:`1` if they anti-commute.
    ///
    /// Args: qubit_sparse_pauli (QubitSparsePauli): the qubit sparse Pauli to compute the Pauli
    ///     fidelity of.
    fn pauli_fidelity(&self, qubit_sparse_pauli: &PyQubitSparsePauli) -> PyResult<f64> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let result = inner.pauli_fidelity(qubit_sparse_pauli.inner())?;
        Ok(result)
    }

    fn __matmul__<'py>(
        &self,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyPauliLindbladMap>> {
        self.compose(other)
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: PySequenceIndex<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let indices = match index.with_len(inner.num_terms())? {
            SequenceIndex::Int(index) => {
                return PyGeneratorTerm {
                    inner: inner.term(index).to_term(),
                }
                .into_bound_py_any(py);
            }
            indices => indices,
        };
        let mut out = PauliLindbladMap::identity(inner.num_qubits());
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

    // The documentation for this is inlined into the class-level documentation of
    // `PauliLindbladMap`.
    #[allow(non_snake_case)]
    #[classattr]
    fn GeneratorTerm(py: Python) -> Bound<PyType> {
        py.get_type::<PyGeneratorTerm>()
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
                .map(GeneratorTermView::to_sparse_str)
                .collect::<Vec<_>>()
                .join(" + ")
        };
        Ok(format!(
            "<PauliLindbladMap with {str_num_terms} on {str_num_qubits}: {str_terms}>"
        ))
    }
}

#[cfg(feature = "python")]
impl From<PauliLindbladMap> for PyPauliLindbladMap {
    fn from(val: PauliLindbladMap) -> PyPauliLindbladMap {
        PyPauliLindbladMap {
            inner: Arc::new(RwLock::new(val)),
        }
    }
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for PauliLindbladMap {
    type Target = PyPauliLindbladMap;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyPauliLindbladMap::from(self).into_pyobject(py)
    }
}

/// Attempt to coerce an arbitrary Python object to a [PyPauliLindbladMap].
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
fn coerce_to_map<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyPauliLindbladMap>>> {
    let py = value.py();
    if let Ok(obs) = value.cast_exact::<PyPauliLindbladMap>() {
        return Ok(Some(obs.clone()));
    }
    match PyPauliLindbladMap::py_new(value, None) {
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
