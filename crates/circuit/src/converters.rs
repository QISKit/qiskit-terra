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

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;

use crate::bit::{ShareableClbit, ShareableQubit};
use crate::circuit_data::{CircuitData, CircuitDataError, PyCircuitData};
use crate::dag_circuit::DAGCircuit;
use crate::{Clbit, Qubit};

/// An extractable representation of a QuantumCircuit reserved only for
/// conversion purposes.
#[derive(Debug, Clone)]
pub struct QuantumCircuitData<'py> {
    pub data: CircuitData,
    pub name: Option<String>,
    pub metadata: Option<Bound<'py, PyAny>>,
    pub transpile_layout: Option<Bound<'py, PyAny>>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for QuantumCircuitData<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        ob.getattr("data")?; // in case _data is lazily generated in python
        let circuit_data = ob.getattr("_data")?;
        let data_borrowed = circuit_data.cast::<PyCircuitData>()?;
        Ok(QuantumCircuitData {
            data: data_borrowed.borrow().inner.clone(),
            name: ob.getattr(intern!(py, "name"))?.extract()?,
            metadata: ob.getattr(intern!(py, "metadata")).ok(),
            transpile_layout: ob.getattr(intern!(py, "layout")).ok(),
        })
    }
}

#[pyfunction(signature = (quantum_circuit, copy_operations = true, qubit_order = None, clbit_order = None))]
pub fn circuit_to_dag(
    quantum_circuit: QuantumCircuitData,
    copy_operations: bool,
    qubit_order: Option<Vec<ShareableQubit>>,
    clbit_order: Option<Vec<ShareableClbit>>,
) -> PyResult<DAGCircuit> {
    // Convert ShareableQubit/ShareableClbit to internal indices
    let qubit_indices = qubit_order
        .as_ref()
        .map(|qubits| {
            qubits
                .iter()
                .map(|shareable_qubit| {
                    quantum_circuit
                        .data
                        .qubits()
                        .find(shareable_qubit)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "Qubit {:?} not found in circuit",
                                shareable_qubit
                            ))
                        })
                })
                .collect::<PyResult<Vec<Qubit>>>()
        })
        .transpose()?;

    let clbit_indices = clbit_order
        .as_ref()
        .map(|clbits| {
            clbits
                .iter()
                .map(|shareable_clbit| {
                    quantum_circuit
                        .data
                        .clbits()
                        .find(shareable_clbit)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "Clbit {:?} not found in circuit",
                                shareable_clbit
                            ))
                        })
                })
                .collect::<PyResult<Vec<Clbit>>>()
        })
        .transpose()?;

    DAGCircuit::from_circuit(
        quantum_circuit,
        copy_operations,
        qubit_indices,
        clbit_indices,
    )
    .map_err(Into::into)
}

/// Convert a :class:`.DAGCircuit` to a :class:`.PyCircuitData`.
///
/// `copy_operations` refers to Python-space operations; if set true, we'll attach to a Python
/// interpreter to ensure we can copy any objects.  If we're not running in a Python context, pass
/// `false` to that argument (or better, in Rust space, use `CircuitData::from_dag_ref`).
#[pyfunction(name = "dag_to_circuit", signature = (dag, copy_operations = true))]
pub fn py_dag_to_circuit(
    dag: &DAGCircuit,
    copy_operations: bool,
) -> Result<PyCircuitData, CircuitDataError> {
    if copy_operations {
        Python::attach(|py| CircuitData::from_dag_ref_deepcopy(py, dag))
    } else {
        CircuitData::from_dag_ref(dag)
    }
    .map(PyCircuitData::from)
}

pub fn converters(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(circuit_to_dag, m)?)?;
    m.add_function(wrap_pyfunction!(py_dag_to_circuit, m)?)?;
    Ok(())
}
