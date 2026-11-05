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

use numpy::PyReadonlyArray2;
use pyo3::{
    Bound, PyResult,
    exceptions::PyValueError,
    pyfunction,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};
use qiskit_circuit::{circuit_data::CircuitData, circuit_data::PyCircuitData, operations::Param};
mod cnot_phase_synth;
mod cx_cz_depth_lnn;

pub(crate) mod cz_depth_lnn;

/// Synthesis of a CZ circuit for linear nearest neighbor (LNN) connectivity,
/// based on Maslov and Roetteler.
///
///  Note that this method *reverts* the order of qubits in the circuit,
///  and returns a circuit containing :class:`.CXGate`\s and phase gates
/// (:class:`.SGate`, :class:`.SdgGate` or :class:`.ZGate`).
///
/// References:
///     1. Dmitri Maslov, Martin Roetteler,
///        *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,
///        `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
#[pyfunction]
#[pyo3(signature = (mat))]
fn synth_cz_depth_line_mr(mat: PyReadonlyArray2<bool>) -> PyResult<PyCircuitData> {
    let view = mat.as_array();
    let dim = view.raw_dim();
    if dim[0] != dim[1] {
        return Err(PyValueError::new_err(format!(
            "matrix must be square, but has dimensions ({}, {})",
            dim[0], dim[1]
        )));
    }
    let (num_qubits, lnn_gates) = cz_depth_lnn::synth_cz_depth_line_mr_inner(view);
    Ok(CircuitData::from_standard_gates(num_qubits as u32, lnn_gates, Param::Float(0.0))?.into())
}

pub fn linear_phase(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(synth_cz_depth_line_mr))?;
    m.add_wrapped(wrap_pyfunction!(
        cx_cz_depth_lnn::py_synth_cx_cz_depth_line_my
    ))?;
    m.add_wrapped(wrap_pyfunction!(cnot_phase_synth::synth_cnot_phase_aam))?;
    Ok(())
}
