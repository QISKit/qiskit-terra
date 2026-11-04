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

use pyo3::{prelude::*, pyclass};
/**
    A representation of a ``QubitProperties`` object.
*/
#[pyclass(subclass, module = "qiskit._accelerate.target", from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct QubitProperties {
    #[pyo3(get, set)]
    pub t1: Option<f64>,
    #[pyo3(get, set)]
    pub t2: Option<f64>,
    #[pyo3(get, set)]
    pub frequency: Option<f64>,
}

#[pymethods]
impl QubitProperties {
    /// Create a new ``QubitProperties`` object
    ///
    /// Args:
    ///     t1 (Option<f64>): The T1 relaxation time for the qubit, in seconds.
    ///     t2 (Option<f64>): The T2 dephasing time for the qubit, in seconds.
    ///     frequency (Option<f64>): The resonance frequency of the qubit, in Hz.
    #[new]
    #[pyo3(signature = (t1=None, t2=None, frequency=None))]
    pub fn new(t1: Option<f64>, t2: Option<f64>, frequency: Option<f64>) -> Self {
        Self { t1, t2, frequency }
    }

    fn __getstate__(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
        (self.t1, self.t2, self.frequency)
    }

    fn __setstate__(&mut self, state: (Option<f64>, Option<f64>, Option<f64>)) {
        self.t1 = state.0;
        self.t2 = state.1;
        self.frequency = state.2;
    }

    fn __repr__(&self) -> String {
        format!(
            "QubitProperties(t1={}, t2={}, frequency={})",
            if let Some(t1) = self.t1 {
                t1.to_string()
            } else {
                "None".to_string()
            },
            if let Some(t2) = self.t2 {
                t2.to_string()
            } else {
                "None".to_string()
            },
            if let Some(frequency) = self.frequency {
                frequency.to_string()
            } else {
                "None".to_string()
            }
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_qubit_properties_creation() {
        // Test creation of QubitProperties with all fields set
        let qubit_props = QubitProperties::new(Some(100.0), Some(200.0), Some(5.0));
        assert_eq!(qubit_props.t1, Some(100.0));
        assert_eq!(qubit_props.t2, Some(200.0));
        assert_eq!(qubit_props.frequency, Some(5.0));
    }

    #[test]
    fn test_qubit_properties_none_fields() {
        // Test creation of QubitProperties with all fields as None
        let qubit_props = QubitProperties::new(None, None, None);
        assert_eq!(qubit_props.t1, None);
        assert_eq!(qubit_props.t2, None);
        assert_eq!(qubit_props.frequency, None);
    }
}
