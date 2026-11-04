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

use pyo3::{prelude::*, pyclass};

/**
 A representation of an ``InstructionProperties`` object.
*/
#[pyclass(
    subclass,
    name = "BaseInstructionProperties",
    module = "qiskit._accelerate.target",
    from_py_object
)]
#[derive(Clone, Debug, PartialEq)]
pub struct InstructionProperties {
    #[pyo3(get, set)]
    pub duration: Option<f64>,
    #[pyo3(get, set)]
    pub error: Option<f64>,
}

#[pymethods]
impl InstructionProperties {
    /// Create a new ``BaseInstructionProperties`` object
    ///
    /// Args:
    ///     duration (Option<f64>): The duration, in seconds, of the instruction on the
    ///         specified set of qubits
    ///     error (Option<f64>): The average error rate for the instruction on the specified
    ///         set of qubits.
    #[new]
    #[pyo3(signature = (duration=None, error=None))]
    pub fn new(duration: Option<f64>, error: Option<f64>) -> Self {
        Self { error, duration }
    }

    fn __getstate__(&self) -> PyResult<(Option<f64>, Option<f64>)> {
        Ok((self.duration, self.error))
    }

    fn __setstate__(&mut self, state: (Option<f64>, Option<f64>)) -> PyResult<()> {
        self.duration = state.0;
        self.error = state.1;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "InstructionProperties(duration={}, error={})",
            if let Some(duration) = self.duration {
                duration.to_string()
            } else {
                "None".to_string()
            },
            if let Some(error) = self.error {
                error.to_string()
            } else {
                "None".to_string()
            }
        )
    }
}
