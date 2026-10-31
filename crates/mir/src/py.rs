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
use crate::passes::lowering::lower;
use pyo3::prelude::*;

use pyo3::types::{PyNone, PyType};
use qiskit_circuit::dag_circuit::PyDAGCircuit;

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
}

impl Default for PyMir {
    fn default() -> Self {
        Self::new()
    }
}

/// PyO3 wrapper that exposes DagToMirLowering as a Python pass.
#[pyclass(name = "DagToMirPass", module = "qiskit._accelerate.mir")]
pub struct PyDagToMirPass;

#[pymethods]
impl PyDagToMirPass {
    #[new]
    pub fn new() -> Self {
        PyDagToMirPass
    }

    // Pass protocol attributes consumed by the pass-manager system.
    #[classattr]
    fn _qiskit_pass_name_() -> &'static str {
        "DagToMirPass"
    }

    #[classattr]
    fn _qiskit_pass_ir_in_(py: Python) -> Bound<PyType> {
        py.get_type::<PyDAGCircuit>()
    }

    #[classattr]
    fn _qiskit_pass_ir_out_(py: Python) -> Bound<PyType> {
        py.get_type::<PyMir>()
    }

    fn _qiskit_pass_run_(&self, ir: &PyDAGCircuit, _context: &Bound<PyAny>) -> PyResult<PyMir> {
        // Acquire a read lock on the DAG before lowering.
        let dag_ref = ir.try_read()?;
        Ok(PyMir(lower(dag_ref)))
    }
}

impl Default for PyDagToMirPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Registers the qiskit._accelerate.mir Python submodule.
#[pymodule(name = "mir")]
pub fn mir_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMir>()?;
    m.add_class::<PyDagToMirPass>()?;
    Ok(())
}
