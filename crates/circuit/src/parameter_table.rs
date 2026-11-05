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

use std::sync::OnceLock;

use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use thiserror::Error;

use pyo3::exceptions::PyTypeError;
use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::PySet;

use crate::parameter::parameter_expression::{PyParameter, PyParameterExpression};
use crate::parameter::symbol_expr::Symbol;

import_exception!(qiskit.circuit, CircuitError);

/// This struct models the error conditions that can be raised from the
/// rust methods of the [ParameterTable] struct. The goal is to explicitly
/// enumerate all the error types that are returned by these functions and
/// make it clear that the return type is part of the interface.
///
/// In the the future it is expected this single enum will be replaced by per
/// method error types to make it even more obvious, but this is a step
/// towards that as we migrate from Python -> Rust.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum ParameterTableError {
    #[error("parameter '{0:?}' is not tracked in the table")]
    ParameterNotTracked(ParameterUuid),
    #[error("usage {0:?} is not tracked by the table")]
    UsageNotTracked(ParameterUse),
    #[error("name conflict adding parameter '{0}'")]
    NameConflict(String),
}
impl From<ParameterTableError> for PyErr {
    fn from(value: ParameterTableError) -> PyErr {
        CircuitError::new_err(value.to_string())
    }
}

/// A single use of a symbolic parameter.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum ParameterUse {
    Index { instruction: usize, parameter: u32 },
    GlobalPhase,
}

/// Tracked data tied to each parameter's UUID in the table.
#[derive(Clone, Debug)]
pub struct ParameterInfo {
    uses: HashSet<ParameterUse>,
    symbol: Symbol,
}

/// Type-safe UUID for a symbolic parameter.  This does not track the name of a [Symbol]; it
/// can't be used alone to reconstruct one.  That tracking remains only within the
/// [ParameterTable].
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct ParameterUuid(u128);
impl ParameterUuid {
    /// Extract a UUID from a Python-space `Parameter` object. This assumes that the object is known
    /// to be a parameter.
    pub fn from_parameter(ob: &Bound<PyAny>) -> PyResult<Self> {
        let uuid = if let Ok(param) = ob.cast::<PyParameter>() {
            // this downcast should cover both PyParameterVectorElement and PyParameter
            param.borrow().0.uuid().as_u128()
        } else if let Ok(expr) = ob.cast::<PyParameterExpression>() {
            let expr_borrowed = expr.borrow();
            // We know the ParameterExpression is in fact representing a single Symbol
            let symbol = &expr_borrowed.inner.try_to_symbol()?;
            symbol.uuid().as_u128()
        } else {
            return Err(PyTypeError::new_err(
                "Could not downcast to Parameter or Expression (that equals a symbol)",
            ));
        };

        Ok(Self(uuid))
    }

    pub fn from_symbol(symbol: &Symbol) -> Self {
        Self(symbol.uuid().as_u128())
    }
}

#[derive(Clone, Default, Debug)]
pub struct ParameterTable {
    /// Mapping of the parameter key (its UUID) to the information on it tracked by this table.
    by_uuid: HashMap<ParameterUuid, ParameterInfo>,
    /// Mapping of the parameter names to the UUID that represents them.
    by_repr: HashMap<String, ParameterUuid>,
    /// Cache of the sort order of the parameters.  This is lexicographical for most parameters,
    /// except elements of a `ParameterVector` are sorted within the vector by numerical index.  We
    /// calculate this on demand and cache it.
    ///
    /// Any method that adds or removes a parameter needs to invalidate this.
    order_cache: OnceLock<Vec<ParameterUuid>>,
    /// Cache of a [Symbol] objects, in order.  We only generate this specifically when asked.
    /// Typically to provide a list to Python of all the Python space `Parameter` objects
    /// (which are 1:1 with a Rust space [Symbol] object).
    ///
    /// Any method that adds or removes a parameter needs to invalidate this.
    parameters_cache: OnceLock<Vec<Symbol>>,
}

impl ParameterTable {
    pub fn new() -> Self {
        Default::default()
    }

    /// Get the number of parameters tracked by the table.
    pub fn num_parameters(&self) -> usize {
        self.by_uuid.len()
    }

    /// Does this table track the given parameter?
    pub fn contains(&self, symbol: &Symbol) -> bool {
        self.by_uuid
            .contains_key(&ParameterUuid::from_symbol(symbol))
    }

    /// Add a new usage of a parameter coming in, optionally adding a first usage to it.
    ///
    /// The no-use form is useful when doing parameter assignments from Rust space, where the
    /// replacement is itself parametric; the replacement can be extracted once, then subsequent
    /// lookups and updates done without interaction with Python.
    pub fn track(
        &mut self,
        symbol: &Symbol,
        usage: Option<ParameterUse>,
    ) -> Result<ParameterUuid, ParameterTableError> {
        let uuid = ParameterUuid::from_symbol(symbol);
        match self.by_uuid.entry(uuid) {
            Entry::Occupied(mut entry) => {
                if let Some(usage) = usage {
                    entry.get_mut().uses.insert(usage);
                }
            }
            Entry::Vacant(entry) => {
                let repr = symbol.repr(false);
                if self.by_repr.contains_key(&repr) {
                    return Err(ParameterTableError::NameConflict(repr));
                }
                self.by_repr.insert(repr.clone(), uuid);
                let mut uses = HashSet::new();
                if let Some(usage) = usage {
                    unsafe {
                        uses.insert_unique_unchecked(usage);
                    }
                };
                entry.insert(ParameterInfo {
                    uses,
                    symbol: symbol.clone(),
                });
                self.invalidate_cache();
            }
        }
        Ok(uuid)
    }

    /// Untrack one use of a single [Symbol] object from the table, discarding all
    /// other tracking of that [Symbol] if this was the last usage of it.
    pub fn untrack(
        &mut self,
        symbol: &Symbol,
        usage: ParameterUse,
    ) -> Result<(), ParameterTableError> {
        self.remove_use(ParameterUuid::from_symbol(symbol), usage)
    }

    /// Lookup the Python parameter object by name.
    pub fn parameter_by_name(&self, name: &str) -> Option<&Symbol> {
        self.by_repr
            .get(name)
            .map(|uuid| &self.by_uuid[uuid].symbol)
    }

    /// Lookup the Python parameter object by uuid.
    pub fn parameter_by_uuid(&self, uuid: ParameterUuid) -> Option<&Symbol> {
        self.by_uuid.get(&uuid).map(|param| &param.symbol)
    }

    /// Get the (maybe cached) list of the sorted `Parameter` objects.
    pub fn symbols(&self) -> &[Symbol] {
        self.parameters_cache.get_or_init(|| {
            self.order_cache
                .get_or_init(|| self.sorted_order())
                .iter()
                .map(|uuid| self.by_uuid[uuid].symbol.clone())
                .collect()
        })
    }

    /// Get a set of all tracked [Symbol] objects.
    pub fn parameters_unsorted(&self) -> HashSet<&Symbol> {
        HashSet::from_iter(self.iter_symbols())
    }

    /// Iterate over the [Symbol]s in the circuit. These are iterated in unsorted order.
    pub fn iter_symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.by_uuid.values().map(|info| &info.symbol)
    }

    /// Get the sorted order of the `ParameterTable`.  This does not access the cache.
    fn sorted_order(&self) -> Vec<ParameterUuid> {
        let mut out = self.by_uuid.keys().copied().collect::<Vec<_>>();
        out.sort_unstable_by_key(|uuid| self.by_uuid[uuid].symbol.ord_key());
        out
    }

    /// Add a use of a parameter to the table.
    pub fn add_use(
        &mut self,
        uuid: ParameterUuid,
        usage: ParameterUse,
    ) -> Result<(), ParameterTableError> {
        self.by_uuid
            .get_mut(&uuid)
            .ok_or(ParameterTableError::ParameterNotTracked(uuid))?
            .uses
            .insert(usage);
        Ok(())
    }

    /// Return a use of a parameter.
    ///
    /// If the last use a parameter is discarded, the parameter is untracked.
    pub fn remove_use(
        &mut self,
        uuid: ParameterUuid,
        usage: ParameterUse,
    ) -> Result<(), ParameterTableError> {
        let Entry::Occupied(mut entry) = self.by_uuid.entry(uuid) else {
            return Err(ParameterTableError::ParameterNotTracked(uuid));
        };
        let info = entry.get_mut();
        if !info.uses.remove(&usage) {
            return Err(ParameterTableError::UsageNotTracked(usage));
        }
        if info.uses.is_empty() {
            self.by_repr.remove(&info.symbol.repr(false));
            entry.remove_entry();
            self.invalidate_cache();
        }
        Ok(())
    }

    /// Remove a parameter from the table, returning the tracked uses of it.
    pub fn pop(
        &mut self,
        uuid: ParameterUuid,
    ) -> Result<HashSet<ParameterUse>, ParameterTableError> {
        let info = self
            .by_uuid
            .remove(&uuid)
            .ok_or(ParameterTableError::ParameterNotTracked(uuid))?;
        self.by_repr
            .remove(&info.symbol.repr(false))
            .expect("each parameter should be tracked by both UUID and name");
        self.invalidate_cache();
        Ok(info.uses)
    }

    /// Clear this table, yielding the Python parameter objects and their uses in sorted order.
    ///
    /// The clearing effect is eager and not dependent on the iteration.
    pub fn drain_ordered(
        &mut self,
    ) -> impl ExactSizeIterator<Item = (Symbol, HashSet<ParameterUse>)> + use<> {
        let order = self
            .order_cache
            .take()
            .unwrap_or_else(|| self.sorted_order());
        let by_uuid = ::std::mem::take(&mut self.by_uuid);
        self.by_repr.clear();
        self.parameters_cache.take();
        ParameterTableDrain {
            order: order.into_iter(),
            by_uuid,
        }
    }

    /// Empty this `ParameterTable` of all its contents.  This does not affect the capacities of the
    /// internal data storage.
    pub fn clear(&mut self) {
        self.by_uuid.clear();
        self.by_repr.clear();
        self.invalidate_cache();
    }

    fn invalidate_cache(&mut self) {
        self.order_cache.take();
        self.parameters_cache.take();
    }

    /// Expose the tracked data for a given parameter as directly as possible to Python space.
    ///
    /// This is only really intended for use in testing.
    pub(crate) fn _py_raw_entry(&self, param: Bound<PyAny>) -> PyResult<Py<PySet>> {
        let py = param.py();
        let uuid = ParameterUuid::from_parameter(&param)?;
        let info = self
            .by_uuid
            .get(&uuid)
            .ok_or(ParameterTableError::ParameterNotTracked(uuid))?;
        // PyO3's `PySet::new` only accepts iterables of references.
        let out = PySet::empty(py)?;
        for usage in info.uses.iter() {
            match usage {
                ParameterUse::GlobalPhase => out.add((py.None(), py.None()))?,
                ParameterUse::Index {
                    instruction,
                    parameter,
                } => out.add((*instruction, *parameter))?,
            }
        }
        Ok(out.unbind())
    }
}

struct ParameterTableDrain {
    order: ::std::vec::IntoIter<ParameterUuid>,
    by_uuid: HashMap<ParameterUuid, ParameterInfo>,
}
impl Iterator for ParameterTableDrain {
    type Item = (Symbol, HashSet<ParameterUse>);

    fn next(&mut self) -> Option<Self::Item> {
        self.order.next().map(|uuid| {
            let info = self
                .by_uuid
                .remove(&uuid)
                .expect("tracked UUIDs should be consistent");
            (info.symbol, info.uses)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.order.size_hint()
    }
}
impl ExactSizeIterator for ParameterTableDrain {}
impl ::std::iter::FusedIterator for ParameterTableDrain {}
