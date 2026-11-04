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

use hashbrown::HashMap;

use pyo3::Python;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::TranspilerError;
use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, PyDAGCircuit};

#[derive(Clone)]
pub(crate) enum CallbackType {
    Python(Py<PyAny>),
    Native(fn(&[f64], &[PhysicalQubit]) -> DAGCircuit),
}

impl CallbackType {
    fn call(&self, angles: &[f64], qubits: &[PhysicalQubit]) -> PyResult<DAGCircuit> {
        match self {
            Self::Python(inner) => {
                let qubits: Vec<usize> = qubits.iter().map(|x| x.index()).collect();
                Python::attach(|py| {
                    inner
                        .bind(py)
                        .call1((angles, qubits))?
                        .cast_into::<PyDAGCircuit>()
                        .map_err(PyErr::from)
                        // Since there is currently no other way of safely obtaining
                        // the owned DAGCircuit object. We will explicitly clone the
                        // DAG obtained in Python.
                        .map(|dag| dag.borrow().try_read().cloned())?
                })
            }
            Self::Native(inner) => Ok(inner(angles, qubits)),
        }
    }
}

/// Registry of Angle Wrapping function
///
/// This class internally contains a mapping of instruction names from a :class:`.Target` to
/// callbacks for wrapping angles that are outside the specified bounds.
#[pyclass(module = "qiskit._accelerate.angle_bound_registry")]
#[pyo3(name = "WrapAngleRegistry")]
pub struct PyWrapAngleRegistry(WrapAngleRegistry);

#[pymethods]
impl PyWrapAngleRegistry {
    #[new]
    pub fn new() -> Self {
        PyWrapAngleRegistry(WrapAngleRegistry::new())
    }

    /// Get a replacement circuit for
    pub fn substitute_angle_bounds(
        &self,
        name: &str,
        angles: Vec<f64>,
        qubits: Vec<PhysicalQubit>,
    ) -> PyResult<Option<PyDAGCircuit>> {
        Ok(self
            .0
            .substitute_angle_bounds(name, &angles, &qubits)?
            .map(Into::into))
    }

    pub fn add_wrapper(&mut self, name: String, callback: Py<PyAny>) {
        self.0.registry.insert(name, CallbackType::Python(callback));
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
        let bounds_dict = PyDict::new(py);
        for (name, bound) in self.get_inner().registry.iter() {
            if let CallbackType::Python(obj) = bound {
                bounds_dict.set_item(name, obj.clone_ref(py))?;
            } else {
                return Err(TranspilerError::new_err(
                    "Target contains native code bounds callbacks which can't be serialized",
                ));
            }
        }
        Ok(bounds_dict.unbind())
    }

    fn __setstate__(&mut self, data: Bound<PyDict>) -> PyResult<()> {
        for (key, val) in data.iter() {
            let name: String = key.extract()?;
            self.add_wrapper(name, val.unbind());
        }
        Ok(())
    }
}

impl PyWrapAngleRegistry {
    pub fn get_inner(&self) -> &WrapAngleRegistry {
        &self.0
    }
}

/// Store the mapping between gate names and callbacks for wrapping that instructions' angles
/// which are outside the specified bounds.
pub struct WrapAngleRegistry {
    registry: HashMap<String, CallbackType>,
}

impl WrapAngleRegistry {
    pub fn new() -> Self {
        WrapAngleRegistry {
            registry: HashMap::new(),
        }
    }

    pub fn add_native(
        &mut self,
        name: String,
        callback: fn(&[f64], &[PhysicalQubit]) -> DAGCircuit,
    ) {
        self.registry.insert(name, CallbackType::Native(callback));
    }

    /// Get a replacement circuit for an instruction outside the specified bounds.
    pub fn substitute_angle_bounds(
        &self,
        name: &str,
        angles: &[f64],
        qubits: &[PhysicalQubit],
    ) -> PyResult<Option<DAGCircuit>> {
        match self.registry.get(name) {
            Some(callback) => Some(callback.call(angles, qubits)).transpose(),
            None => Err(PyKeyError::new_err(format!(
                "Name '{name}' not in WrapAngleRegistry"
            ))),
        }
    }
}

impl Default for WrapAngleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PyWrapAngleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn angle_bound_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyWrapAngleRegistry>()?;
    Ok(())
}
