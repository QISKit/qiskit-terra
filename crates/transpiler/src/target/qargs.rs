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

use indexmap::Equivalent;
use pyo3::{IntoPyObject, prelude::*, types::PyTuple};
use smallvec::SmallVec;

use qiskit_circuit::PhysicalQubit;

pub type TargetQargs = SmallVec<[PhysicalQubit; 2]>;

/// Representation of quantum args for a [Target](super::Target).
///
/// An instruction stored within a [Target](super::Target) can have
/// two different types of qargs when specifying its properties:
/// - Global: If the instruction is a Variadic or can operate in any set
///   of qargs as long as they match the capacity of the instruction.
/// - Concrete: Specific combination of quantum args.
///
/// This enumeration represents these two conditions efficiently while
/// solving certain ownership issues that [Option] currently has.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Qargs {
    Global,
    Concrete(TargetQargs),
}

impl From<TargetQargs> for Qargs {
    fn from(value: TargetQargs) -> Self {
        Self::Concrete(value)
    }
}

impl FromIterator<PhysicalQubit> for Qargs {
    fn from_iter<T: IntoIterator<Item = PhysicalQubit>>(iter: T) -> Self {
        Qargs::Concrete(iter.into_iter().collect())
    }
}

impl<const N: usize> From<[PhysicalQubit; N]> for Qargs {
    fn from(value: [PhysicalQubit; N]) -> Self {
        Self::Concrete(SmallVec::from_iter(value))
    }
}

impl Qargs {
    /// Returns a reference version of a qarg.
    pub fn as_ref(&self) -> QargsRef<'_> {
        match self {
            Qargs::Global => QargsRef::Global,
            Qargs::Concrete(qargs) => QargsRef::Concrete(qargs),
        }
    }

    /// Checks if the qargs in question are [Global](Qargs::Global).
    pub fn is_global(&self) -> bool {
        matches!(self, Self::Global)
    }

    /// Checks if the qargs in question are `Concrete`.
    pub fn is_concrete(&self) -> bool {
        !self.is_global()
    }
}

impl<'py> IntoPyObject<'py> for Qargs {
    type Target = PyAny;

    type Output = Bound<'py, PyAny>;

    type Error = PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        (&self).into_pyobject(py)
    }
}

impl<'py> IntoPyObject<'py> for &Qargs {
    type Target = PyAny;

    type Output = Bound<'py, PyAny>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Qargs::Global => Ok(py.None().into_bound(py)),
            Qargs::Concrete(qargs) => Ok(PyTuple::new(py, qargs)?.into_any()),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Qargs {
    type Error = <TargetQargs as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let qargs: Option<TargetQargs> = ob.extract()?;
        match qargs {
            Some(qargs) => Ok(Self::Concrete(qargs)),
            None => Ok(Self::Global),
        }
    }
}

impl Equivalent<Qargs> for QargsRef<'_> {
    fn equivalent(&self, key: &Qargs) -> bool {
        *self == key.as_ref()
    }
}

/// Reference representation of [Qargs].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum QargsRef<'a> {
    Global,
    Concrete(&'a [PhysicalQubit]),
}

impl QargsRef<'_> {
    /// Checks if the qargs in question are `Global`.
    pub fn is_global(&self) -> bool {
        matches!(self, Self::Global)
    }

    /// Checks if the qargs in question are `Concrete`.
    pub fn is_concrete(&self) -> bool {
        !self.is_global()
    }
}

impl<'a> From<&'a Qargs> for QargsRef<'a> {
    fn from(value: &'a Qargs) -> Self {
        match value {
            Qargs::Global => Self::Global,
            Qargs::Concrete(qargs) => QargsRef::Concrete(qargs),
        }
    }
}

impl<'a, T> From<&'a T> for QargsRef<'a>
where
    T: AsRef<[PhysicalQubit]>,
{
    fn from(value: &'a T) -> Self {
        Self::Concrete(value.as_ref())
    }
}

impl<'a> From<&'a [PhysicalQubit]> for QargsRef<'a> {
    fn from(value: &'a [PhysicalQubit]) -> Self {
        Self::Concrete(value)
    }
}

impl<'py> IntoPyObject<'py> for QargsRef<'_> {
    type Target = PyAny;

    type Output = Bound<'py, PyAny>;

    type Error = PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Self::Global => Ok(py.None().into_bound(py)),
            Self::Concrete(qargs) => Ok(PyTuple::new(py, qargs)?.into_any()),
        }
    }
}
