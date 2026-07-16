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

use std::{fmt::Debug, hash::Hash, sync::OnceLock};

use crate::bit::{BitLocations, Register};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
use qiskit_util::IndexMap;

/// Structure that keeps a mapping of bits and their locations within
/// the circuit.
#[derive(Debug)]
pub struct BitLocator<B, R: Register> {
    bit_locations: IndexMap<B, BitLocations<R>>,
    cached: OnceLock<Py<PyDict>>,
}

/// Custom implementation of BitLocator to skip copying the cache.
impl<B, R> Clone for BitLocator<B, R>
where
    B: Debug + Clone + Hash + Eq,
    R: Register + Debug + Clone,
{
    fn clone(&self) -> Self {
        Self {
            bit_locations: self.bit_locations.clone(),
            cached: OnceLock::new(),
        }
    }
}

impl<B, R> Default for BitLocator<B, R>
where
    B: Debug + Clone + Hash + Eq,
    R: Register + Debug + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, R> BitLocator<B, R>
where
    B: Debug + Clone + Hash + Eq,
    R: Register + Debug + Clone,
{
    /// Create an empty locator for bits.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create an empty locator for bits with pre-allocated capacity to contain a given number.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bit_locations: IndexMap::with_capacity_and_hasher(capacity, Default::default()),
            cached: OnceLock::new(),
        }
    }

    /// Track a bit at the given locations.
    ///
    /// If the bit was already tracked, its locations are updated with the new ones, and the old
    /// ones are returned.
    pub fn insert(&mut self, bit: B, location: BitLocations<R>) -> Option<BitLocations<R>> {
        self.cached.take();
        self.bit_locations.insert(bit, location)
    }

    /// Get the locations of a bit, if it is tracked.
    pub fn get(&self, bit: &B) -> Option<&BitLocations<R>> {
        self.bit_locations.get(bit)
    }

    /// Get the locations of a bit for mutation, if it is tracked.
    pub fn get_mut(&mut self, bit: &B) -> Option<&mut BitLocations<R>> {
        self.cached.take();
        self.bit_locations.get_mut(bit)
    }

    /// Is the bit tracked?
    pub fn contains_key(&self, bit: &B) -> bool {
        self.bit_locations.contains_key(bit)
    }

    /// Called during Python garbage collection, only!.
    /// Note: INVALIDATES THIS INSTANCE.
    pub fn dispose(&mut self) {
        self.bit_locations.clear();
        self.cached.take();
    }
}

impl<B, R> BitLocator<B, R>
where
    B: Debug
        + Clone
        + Hash
        + Eq
        + for<'py> IntoPyObject<'py>
        + for<'a, 'py> FromPyObject<'a, 'py, Error: Into<PyErr>>,
    R: Register + Debug + Clone + for<'py> IntoPyObject<'py> + for<'a, 'py> FromPyObject<'a, 'py>,
{
    /// Get or create the cached Python dictionary that represents this.
    pub fn cached(&self, py: Python) -> &Py<PyDict> {
        self.cached.get_or_init(|| {
            self.bit_locations
                .iter()
                .map(|(bit, loc)| (bit.clone(), loc.clone()))
                .into_py_dict(py)
                .unwrap()
                .into()
        })
    }

    /// Recreate this object from a Python dictionary.
    pub fn from_py_dict(dict: &Bound<PyDict>) -> PyResult<Self> {
        let mut locator = Self::new();
        for (key, value) in dict {
            locator.insert(
                key.extract().map_err(Into::<PyErr>::into)?,
                value.extract()?,
            );
        }
        Ok(locator)
    }

    /// Get the cache field without creating it if it doesn't exist.
    pub fn cached_raw(&self) -> Option<&Py<PyDict>> {
        self.cached.get()
    }
}
