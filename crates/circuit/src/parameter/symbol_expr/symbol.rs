// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::borrow::Cow;
use std::cmp::{Ord, Ordering, PartialOrd};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, atomic};
use uuid::Uuid;

use pyo3::prelude::*;

use crate::parameter::parameter_expression::PyParameter;

#[derive(Debug, Clone)]
pub enum Symbol {
    Standalone {
        name: String,
        uuid: Uuid,
    },
    Element {
        index: usize,
        base: Arc<SymbolVector>,
    },
}
impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ord_key().cmp(&other.ord_key())
    }
}
// These derived versions of `PartialEq` and `Hash` are because `SymbolVector` is (unfortunately)
// internally mutable to support the Python-space `ParameterVector.resize` propagating its mutations
// through to the backreferences, but happily Python-space never actually checked if the
// `ParameterVector` backrefs inside `ParameterVectorElement` were equal, so we don't either.  Once
// `ParameterVector.resize` is gone, we can make this more sane by deriving `Eq` and `Hash`.
impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.ord_key() == other.ord_key()
    }
}
impl Eq for Symbol {}
impl Hash for Symbol {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ord_key().hash(state)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Symbol {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self::clone(&ob.cast::<PyParameter>()?.borrow().0))
    }
}

impl<'py> IntoPyObject<'py> for Symbol {
    type Target = <PyParameter as IntoPyObject<'py>>::Target;
    type Output = <PyParameter as IntoPyObject<'py>>::Output;
    type Error = <PyParameter as IntoPyObject<'py>>::Error;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyParameter(Arc::new(self)).into_pyobject(py)
    }
}

impl Symbol {
    /// Create a new standalone (not in a vector) symbol.
    pub fn standalone(name: String, uuid: Option<Uuid>) -> Self {
        let uuid = uuid.unwrap_or_else(Uuid::new_v4);
        Self::Standalone { name, uuid }
    }

    /// The base name of the symbol, without any vector-index suffix.
    ///
    /// This does not include the vector name.  Use [`fullname`] if you need that for display.
    pub fn name(&self) -> &str {
        match self {
            Self::Standalone { name, uuid: _ } => name.as_str(),
            Self::Element { base, index: _ } => base.name.as_str(),
        }
    }

    pub fn fullname(&self) -> Cow<'_, str> {
        match self {
            Self::Standalone { name, uuid: _ } => Cow::Borrowed(name.as_str()),
            Self::Element { base, index } => Cow::Owned(format!("{}[{}]", base.name, index)),
        }
    }

    pub fn uuid(&self) -> Uuid {
        match self {
            Self::Standalone { uuid, name: _ } => *uuid,
            Self::Element { base, index } => Uuid::from_u128(base.uuid.as_u128() + *index as u128),
        }
    }

    pub fn repr(&self, with_uuid: bool) -> String {
        if with_uuid {
            format!("{}_{}", self.fullname(), self.uuid().as_u128())
        } else {
            self.fullname().into_owned()
        }
    }

    /// The key object to use for all comparison operations with this object.
    pub fn ord_key(&self) -> impl Ord + Hash + use<'_> {
        (
            // This is the base name, without the vector-index appended, because we use those
            // numerically.
            self.name(),
            match self {
                Self::Standalone { .. } => None,
                // Everything about `base` that we actually use is inside `name` and `uuid`; we've
                // never actually required that the backrefs inside vector elements are to equal
                // vectors (!).  While `ParameterExpression.resize` still exists, this actually is
                // useful for us: we can maintain equality checks through serialisation roundtrips.
                Self::Element { index, base: _ } => Some(index),
            },
            self.uuid(),
        )
    }
}

/// Vector of multiple parameters.
///
/// This makes creating large numbers of parameters much more efficient; we no longer need to
/// allocate individual strings and UUIDs for each symbol, but instead derive them all from the
/// shared base.
///
/// This object is intended to be used almost exclusively behind an `Arc`, since all derived
/// symbols from it need an `Arc` backreference.
#[derive(Debug)]
pub struct SymbolVector {
    pub name: String,
    pub uuid: Uuid,
    /// Length of the vector.
    ///
    /// This is `AtomicUsize` to support the (unfortunate) Python-space method
    /// `ParameterVector.resize`.  Once that method is deprecated and removed, this can become a
    /// regular `usize` (and we'll be able to derive `Hash`, if we want).
    pub len: atomic::AtomicUsize,
}
impl SymbolVector {
    pub fn new(name: String, len: usize) -> Arc<Self> {
        Arc::new(Self {
            name,
            uuid: Uuid::new_v4(),
            len: len.into(),
        })
    }

    pub fn get(self: &Arc<Self>, index: usize) -> Option<Symbol> {
        (index < self.len.load(atomic::Ordering::Relaxed)).then(|| Symbol::Element {
            index,
            base: Arc::clone(self),
        })
    }

    pub fn iter(self: &Arc<Self>) -> impl ExactSizeIterator<Item = Symbol> {
        (0..self.len.load(atomic::Ordering::Relaxed)).map(|index| {
            self.get(index)
                .expect("the iterator is only over valid values")
        })
    }
}
impl Clone for SymbolVector {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            uuid: self.uuid,
            len: self.len.load(atomic::Ordering::Relaxed).into(),
        }
    }
}
