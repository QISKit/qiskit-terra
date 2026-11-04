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

use crate::mir::Mir;
use pyo3::prelude::*;

use pyo3::types::PyNone;

/// Python-facing wrapper around the Rust Mir type.
#[pyclass(name = "Mir", module = "qiskit._accelerate.mir")]
#[derive(Debug)]
pub struct PyMir(pub Mir);

#[pymethods]
impl PyMir {
    #[new]
    pub fn new() -> Self {
        PyMir(Mir::new())
    }

    #[classattr]
    fn _qiskit_ir_builtin_(py: Python) -> Bound<PyNone> {
        // Marker for the pass-manager system to treat `Mir` as a natively-handled IR type.
        PyNone::get(py).to_owned()
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Mir:\n{}", self.0)
    }

    fn num_instructions(&self) -> usize {
        self.0.instructions().node_count()
    }

    fn num_qubits(&self) -> usize {
        self.0.resource_table().iter_qids().len()
    }
}

impl Default for PyMir {
    fn default() -> Self {
        Self::new()
    }
}

/// Registers the qiskit._accelerate.mir Python submodule.
#[pymodule(name = "mir")]
pub fn mir_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMir>()?;
    Ok(())
}
