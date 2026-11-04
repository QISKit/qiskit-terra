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

use crate::dag_to_mir::lower_dag;
use pyo3::prelude::*;
use qiskit_mir::py::PyMir;

use pyo3::types::PyType;
use qiskit_circuit::dag_circuit::PyDAGCircuit;

/// PyO3 wrapper that exposes DagToMirPass as a Python pass.
#[pyclass(name = "DagToMirPass", module = "qiskit._accelerate.mir_passes")]
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
        let mir = lower_dag(dag_ref)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyMir(mir))
    }
}

impl Default for PyDagToMirPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Registers the qiskit._accelerate.mir_passes Python submodule.
#[pymodule(name = "mir_passes")]
pub fn mir_passes_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDagToMirPass>()?;
    Ok(())
}
