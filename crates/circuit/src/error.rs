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

use pyo3::PyErr;
use pyo3::exceptions::PyMemoryError;
use pyo3::import_exception;

use hashbrown::TryReserveError as HashTryReserveError;
use indexmap::TryReserveError as IndexTryReserveError;
use std::collections::TryReserveError as VecTryReserveError;
use thiserror::Error;

import_exception!(qiskit.dagcircuit.exceptions, DAGCircuitError);
import_exception!(qiskit.dagcircuit.exceptions, DAGDependencyError);

#[derive(Debug, Error)]
pub enum TryReserveError {
    #[error(transparent)]
    VecTryReserve(VecTryReserveError),
    #[error("{0:?}")]
    HashTryReserve(HashTryReserveError),
    #[error(transparent)]
    IndexTryReserve(IndexTryReserveError),
}

impl From<VecTryReserveError> for TryReserveError {
    fn from(val: VecTryReserveError) -> Self {
        Self::VecTryReserve(val)
    }
}

impl From<HashTryReserveError> for TryReserveError {
    fn from(val: HashTryReserveError) -> Self {
        Self::HashTryReserve(val)
    }
}

impl From<IndexTryReserveError> for TryReserveError {
    fn from(val: IndexTryReserveError) -> Self {
        Self::IndexTryReserve(val)
    }
}

impl From<TryReserveError> for PyErr {
    fn from(val: TryReserveError) -> Self {
        PyMemoryError::new_err(val.to_string())
    }
}
