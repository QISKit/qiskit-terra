// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::marker::PhantomData;
use std::ops;

use rustworkx_core::petgraph::graph::IndexType;

/// Internal helper struct that represents a `Box<[T]>`, but is indexed by a petgraph-like indexer.
///
/// The layer structures involve several flat arrays, each representing a map from an index to a
/// value, but the index types are often different, so it's less error-prone if we _enforce_ that
/// you index using the object, rather than calling its `.index` method and erasing the type.
#[derive(Clone, Debug)]
pub struct VecMap<Idx: IndexType, T> {
    phantom: PhantomData<Idx>,
    data: Box<[T]>,
}
impl<Idx: IndexType, T> VecMap<Idx, T> {
    /// Swap the values of two indices.
    #[inline]
    pub fn swap(&mut self, a: Idx, b: Idx) {
        self.data.swap(a.index(), b.index())
    }

    /// Fill all entries in the slice with a given value.
    #[inline]
    pub fn fill(&mut self, val: T)
    where
        T: Clone,
    {
        self.data.fill(val)
    }
}

impl<Idx: IndexType, T> ops::Index<Idx> for VecMap<Idx, T> {
    type Output = <[T] as ops::Index<usize>>::Output;
    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[index.index()]
    }
}
impl<Idx: IndexType, T> ops::IndexMut<Idx> for VecMap<Idx, T> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.data[index.index()]
    }
}

impl<Idx: IndexType, T> From<Vec<T>> for VecMap<Idx, T> {
    fn from(value: Vec<T>) -> Self {
        Self {
            phantom: PhantomData,
            data: value.into_boxed_slice(),
        }
    }
}
