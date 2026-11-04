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

use crate::CapacityError;
use crate::error::TryReserveError;
use hashbrown::HashMap;
use hashbrown::hash_map::OccupiedError;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

/// Wrapper for Python-side objects that implements [Hash] and [Eq], allowing them to be
/// used in Rust hash-based sets and maps.
///
/// Python's `hash()` is called on the wrapped object during construction and returned from Rust's
/// [Hash] trait impl.  The impl of [PartialEq] first compares the native Py pointers to determine
/// equality. If these are not equal, only then does it call `repr()` on both sides, which has a
/// significant performance advantage.
#[derive(Clone, Debug)]
pub struct PyObjectAsKey {
    /// Python's `hash()` of the wrapped instance.
    hash: isize,
    /// The wrapped instance.
    ob: Py<PyAny>,
}

impl PyObjectAsKey {
    pub fn new(object: &Bound<PyAny>) -> Self {
        PyObjectAsKey {
            // This really shouldn't fail, but if it does,
            // we'll just use 0.
            hash: object.hash().unwrap_or(0),
            ob: object.clone().unbind(),
        }
    }

    /// Safely clones the underlying python object reference
    pub fn clone_ref(&self, py: Python) -> Self {
        Self {
            hash: self.hash,
            ob: self.ob.clone_ref(py),
        }
    }

    /// Get a reference to the wrapped Python object.
    pub fn object(&self) -> &Py<PyAny> {
        &self.ob
    }
}

impl Hash for PyObjectAsKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_isize(self.hash);
    }
}

impl<'a, 'py> From<&'a Bound<'py, PyAny>> for PyObjectAsKey {
    fn from(value: &'a Bound<'py, PyAny>) -> Self {
        PyObjectAsKey::new(value)
    }
}

impl<'py> From<Bound<'py, PyAny>> for PyObjectAsKey {
    fn from(value: Bound<'py, PyAny>) -> Self {
        PyObjectAsKey::new(&value)
    }
}

impl<'py> IntoPyObject<'py> for PyObjectAsKey {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.ob.bind(py).clone())
    }
}

impl<'a, 'py> IntoPyObject<'py> for &'a PyObjectAsKey {
    type Target = PyAny;
    type Output = Borrowed<'a, 'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.ob.bind_borrowed(py))
    }
}

impl PartialEq for PyObjectAsKey {
    fn eq(&self, other: &Self) -> bool {
        self.ob.is(&other.ob)
            || Python::attach(|py| {
                self.ob
                    .bind(py)
                    .repr()
                    .unwrap()
                    .as_any()
                    .eq(other.ob.bind(py).repr().unwrap())
                    .unwrap()
            })
    }
}
impl Eq for PyObjectAsKey {}

/// Error types for attempts to add unique objects.
#[derive(Clone, Debug, thiserror::Error)]
pub enum AddError<T: Debug = String, B: Debug = String> {
    #[error("cannot add object {ob:?} as it is already mapped to {key:?}")]
    Duplicate { key: T, ob: B },
    #[error(transparent)]
    Capacity(#[from] CapacityError),
}
impl<T: Debug, B: Debug> AddError<T, B> {
    pub fn erase_type(self) -> AddError {
        match self {
            Self::Duplicate { key, ob } => AddError::Duplicate {
                key: format!("{key:?}"),
                ob: format!("{ob:?}"),
            },
            Self::Capacity(c) => AddError::Capacity(c),
        }
    }
}
impl<T: Debug, B: Debug> From<AddError<T, B>> for PyErr {
    fn from(val: AddError<T, B>) -> PyErr {
        match val {
            AddError::Duplicate { .. } => PyValueError::new_err(val.to_string()),
            AddError::Capacity(c) => c.into(),
        }
    }
}

/// Error return from functions that look up an object in the registry.
#[derive(Clone, Debug, thiserror::Error)]
#[error("object {0:?} is not present")]
pub struct AbsentObject<T: Debug = String>(T);
impl<T: Debug> AbsentObject<T> {
    /// Erase the internal type of the object by evaluating the debug formatting.
    pub fn erase_type(self) -> AbsentObject {
        AbsentObject(format!("{:?}", self.0))
    }
}
impl<T: Debug> From<AbsentObject<T>> for PyErr {
    fn from(val: AbsentObject<T>) -> Self {
        PyKeyError::new_err(val.to_string())
    }
}

/// A registry of unique objects, each mapped to a unique index.
///
/// This is used to associate sharable bits and other globally unique
/// objects with local indices tracked by circuits.
///
/// If type parameter `B` implements [IntoPyObject], then a cached [PyList]
/// is maintained and accessible via [ObjectRegistry::cached] and [ObjectRegistry::cached_raw],
/// which contains the unique objects, in the order they were first registered.
#[derive(Clone, Debug)]
pub struct ObjectRegistry<T, B> {
    /// Registered objects.
    objects: Vec<B>,
    /// Maps objects to native index.
    indices: HashMap<B, T>,
    /// The objects registered, cached as a PyList.
    cached: OnceLock<Py<PyList>>,
}

impl<T, B> Default for ObjectRegistry<T, B>
where
    T: From<u32> + Copy + Debug,
    u32: From<T>,
    B: Clone + Eq + Hash + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}
// The stronger `Eq` restriction here on `B` (not `PartialEq`) is because it's necessary for the
// hashmap to function correctly and consequently to implement `PartialEq`.
impl<T: PartialEq, B: Eq + Hash> PartialEq for ObjectRegistry<T, B> {
    fn eq(&self, other: &Self) -> bool {
        (self.objects == other.objects) && (self.indices == other.indices)
    }
}
impl<T: Eq, B: Eq + Hash> Eq for ObjectRegistry<T, B> {}

impl<T, B> ObjectRegistry<T, B>
where
    T: From<u32> + Copy + Debug,
    u32: From<T>,
    B: Clone + Eq + Hash + Debug,
{
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        ObjectRegistry {
            objects: Vec::with_capacity(capacity),
            indices: HashMap::with_capacity(capacity),
            cached: OnceLock::new(),
        }
    }

    /// Create a new ObjectRegistry with an initial capacity pre-allocated. This
    /// will return an error if the specified capacity can not be allocated.
    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        let mut objects = Vec::new();
        objects.try_reserve(capacity)?;
        let mut indices = HashMap::new();
        indices.try_reserve(capacity)?;
        Ok(ObjectRegistry {
            objects,
            indices,
            cached: OnceLock::new(),
        })
    }

    /// Gets the number of registered objects.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Gets a reference to the underlying vector of objects.
    #[inline]
    pub fn objects(&self) -> &Vec<B> {
        &self.objects
    }

    /// Finds the native index of the given object.
    #[inline]
    pub fn find(&self, object: &B) -> Option<T> {
        self.indices.get(object).copied()
    }

    /// Map the provided objects to their native indices.
    /// An error is returned if any object is not registered.
    pub fn map_objects<U: IntoIterator<Item = B>>(
        &self,
        objects: U,
    ) -> Result<impl Iterator<Item = T> + use<T, B, U>, AbsentObject<B>> {
        let v: Result<Vec<_>, _> = objects
            .into_iter()
            .map(|b| self.indices.get(&b).copied().ok_or(AbsentObject(b)))
            .collect();
        v.map(|x| x.into_iter())
    }

    /// Map the provided native indices to the corresponding object instances.
    /// Panics if any of the indices are out of range.
    pub fn map_indices(&self, objects: &[T]) -> impl ExactSizeIterator<Item = &B> + use<'_, T, B> {
        let v: Vec<_> = objects.iter().map(|i| self.get(*i).unwrap()).collect();
        v.into_iter()
    }

    /// Gets the object corresponding to the given native index.
    #[inline]
    pub fn get(&self, index: T) -> Option<&B> {
        self.objects.get(<u32 as From<T>>::from(index) as usize)
    }

    /// Checks if the object is registered.
    #[inline]
    pub fn contains(&self, key: &B) -> bool {
        self.indices.contains_key(key)
    }

    /// Registers a new object, automatically creating a unique index within the registry.
    ///
    /// Errors if the object is already in the registry.  To ignore duplicates, use
    /// [add_allow_existing].
    pub fn add(&mut self, object: B) -> Result<T, AddError<T, B>> {
        let idx: u32 = self.objects.len().try_into().map_err(|_| CapacityError)?;
        // Dump the cache
        self.cached.take();
        match self.indices.try_insert(object.clone(), idx.into()) {
            Ok(_) => {
                self.objects.push(object);
                Ok(idx.into())
            }
            Err(OccupiedError { value: _, entry }) => Err(AddError::Duplicate {
                key: *entry.get(),
                ob: object,
            }),
        }
    }
    /// Add an object to the registry, returning the existing key in the case of duplication.
    pub fn add_allow_existing(&mut self, object: B) -> Result<T, CapacityError> {
        match self.add(object) {
            Ok(key) | Err(AddError::Duplicate { key, ob: _ }) => Ok(key),
            Err(AddError::Capacity(c)) => Err(c),
        }
    }
    // Add an object to the registry, panicking if it is a duplicate or out of capacity.
    pub fn add_unique_within_capacity(&mut self, object: B) -> T {
        self.add(object)
            .expect("caller should ensure uniqueness and capacity bounds")
    }

    pub fn replace(&mut self, index: T, replacement: B) {
        self.cached.take();
        let to_replace = &mut self.objects[<u32 as From<T>>::from(index) as usize];
        self.indices.remove(to_replace);
        *to_replace = replacement.clone();
        self.indices.insert(replacement, index);
    }

    pub fn remove_indices(&mut self, indices: impl IntoIterator<Item = T>) {
        let mut indices_sorted: Vec<usize> = indices
            .into_iter()
            .map(|i| <u32 as From<T>>::from(i) as usize)
            .collect();
        indices_sorted.sort();
        self.cached.take();
        for index in indices_sorted.into_iter().rev() {
            let object = self.objects.remove(index);
            self.indices.remove(&object);
        }
        // Update indices.
        for (i, object) in self.objects.iter().enumerate() {
            self.indices.insert(object.clone(), (i as u32).into());
        }
    }

    /// Called during Python garbage collection, only!.
    /// Note: INVALIDATES THIS INSTANCE.
    pub fn dispose(&mut self) {
        self.indices.clear();
        self.objects.clear();
    }
}

impl<'a, T, B> ObjectRegistry<T, B>
where
    B: Clone,
    B: IntoPyObject<'a>,
{
    /// Gets a reference to the cached Python list, maintained by
    /// this instance.
    #[inline]
    pub fn cached(&self, py: Python<'a>) -> &Py<PyList> {
        self.cached.get_or_init(|| {
            PyList::new(py, self.objects.iter().cloned())
                .unwrap()
                .into()
        })
    }

    /// Gets a reference to the cached Python list, even if not initialized.
    #[inline]
    pub fn cached_raw(&self) -> Option<&Py<PyList>> {
        self.cached.get()
    }
}
