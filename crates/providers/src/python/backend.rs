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

use pyo3::exceptions::PyNotImplementedError;
use pyo3::{PyErr, PyResult};
use qiskit_circuit::converters::QuantumCircuitData;

use crate::tensor::Tensor;
/// The Python binding of the backend.
use crate::{BackendV3, DataTree, Job, QuantumProgram, RunResult};

pub enum PyJobError {}

// pub struct PyJob {}

// impl Job for PyJob  {
//     type JobError = PyErr;
//     fn result(&self) -> PyResult<RunResult> {
//         Ok(RunResult{data:DataTree::new()})
//     }
// }
// pub struct PyBackendV3 {}

// impl BackendV3 for PyBackendV3 {
//     type JobError = PyErr;
//     fn run(&self, program: &QuantumProgram) -> PyJob {PyJob{}};
// }
