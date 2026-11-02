// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

use qiskit_circuit::PhysicalQubit;

use hashbrown::HashMap;

/// A mapping that represents the avg error rate for a particular edge in
/// the connectivity graph of a backend.
///
/// This class is used to efficiently (with no iteration or copy/conversion)
/// represent an error map for a target backend to internal rust code that
/// works with error rates. For most purposes it is meant to be write only
/// from Python, as the intent is to use this to pass data to a Rust module.
/// However, this class does implement the mapping protocol so you can lookup
/// error rates from Python if needed.
///
/// Each entry consists of a key which is a 2 element tuple of qubit numbers
/// (order is significant) and a value which is a ``float`` representing the
/// error rate for the edge connecting the corresponding qubits. For 1 qubit
/// error rates, you should assign both elements of the key to the same
/// qubit index. If an edge or qubit is ideal and has no error rate, you can
/// either set it to ``0.0`` explicitly or as ``NaN``.
#[pyclass(mapping, module = "qiskit._accelerate.error_map")]
#[derive(Clone, Debug)]
pub struct ErrorMap {
    pub error_map: HashMap<[PhysicalQubit; 2], f64>,
}

#[pymethods]
impl ErrorMap {
    #[new]
    #[pyo3(signature=(size=None))]
    pub fn new(size: Option<usize>) -> Self {
        match size {
            Some(size) => ErrorMap {
                error_map: HashMap::with_capacity(size),
            },
            None => ErrorMap {
                error_map: HashMap::new(),
            },
        }
    }

    /// Initialize a new :class:`~.ErrorMap` instance from an input dictionary
    ///
    /// Unlike the default constructor this will have O(n) overhead as it has
    /// to iterate over the input dict to copy/convert each element into error
    /// map. It is generally more efficient to use ``ErrorMap(size)`` and
    /// construct the error map iteratively with :meth:`.add_error` instead of
    /// constructing an intermediate dict and using this constructor.
    #[staticmethod]
    fn from_dict(error_map: HashMap<[PhysicalQubit; 2], f64>) -> Self {
        ErrorMap { error_map }
    }

    pub fn add_error(&mut self, index: [PhysicalQubit; 2], error_rate: f64) {
        self.error_map.insert(index, error_rate);
    }

    // The pickle protocol methods can't return `HashMap<[T; 2], f64>` to Python, because by PyO3's
    // natural conversion as of 0.17.3 it will attempt to construct a `dict[list[T], float]`, where
    // `list[T]` is unhashable in Python.

    fn __getstate__(&self) -> HashMap<(PhysicalQubit, PhysicalQubit), f64> {
        self.error_map
            .iter()
            .map(|([a, b], value)| ((*a, *b), *value))
            .collect()
    }

    fn __setstate__(&mut self, state: HashMap<[PhysicalQubit; 2], f64>) {
        self.error_map = state;
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.error_map.len())
    }

    fn __getitem__(&self, key: [PhysicalQubit; 2]) -> PyResult<f64> {
        match self.error_map.get(&key) {
            Some(data) => Ok(*data),
            None => Err(PyIndexError::new_err("No node found for index")),
        }
    }

    fn __contains__(&self, key: [PhysicalQubit; 2]) -> PyResult<bool> {
        Ok(self.error_map.contains_key(&key))
    }

    #[pyo3(signature=(key, default=None))]
    fn get(
        &self,
        py: Python,
        key: [PhysicalQubit; 2],
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(match self.error_map.get(&key).copied() {
            Some(val) => val.into_py_any(py)?,
            None => match default {
                Some(val) => val,
                None => py.None(),
            },
        })
    }
}

pub fn error_map_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<ErrorMap>()?;
    Ok(())
}
