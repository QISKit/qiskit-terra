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

use mcx::{
    c3x, c4x, synth_mcp_noaux_sp22, synth_mcx_1_clean_b95, synth_mcx_1_kg24, synth_mcx_2_kg24,
    synth_mcx_explicit, synth_mcx_n_clean_m15, synth_mcx_n_dirty_i15, synth_mcx_n_dirty_m15,
    synth_mcx_noaux_hp24, synth_mcx_noaux_sp22,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use qiskit_circuit::circuit_data::PyCircuitData;
use qiskit_circuit::operations::Param;

mod mcmt;
mod mcx;

#[pyfunction]
#[pyo3(name="synth_mcx_n_dirty_i15", signature = (num_controls, relative_phase=false, action_only=false))]
fn py_synth_mcx_n_dirty_i15(
    num_controls: usize,
    relative_phase: bool,
    action_only: bool,
) -> PyResult<PyCircuitData> {
    Ok(synth_mcx_n_dirty_i15(num_controls, relative_phase, action_only)?.into())
}

#[pyfunction]
#[pyo3(name="synth_mcx_n_dirty_m15", signature = (num_controls))]
fn py_synth_mcx_n_dirty_m15(num_controls: usize) -> PyResult<PyCircuitData> {
    Ok(synth_mcx_n_dirty_m15(num_controls)?.into())
}

#[pyfunction]
#[pyo3(name="synth_mcx_1_kg24", signature = (num_controls, clean))]
fn py_synth_mcx_1_kg24(num_controls: usize, clean: bool) -> PyResult<PyCircuitData> {
    Ok(synth_mcx_1_kg24(num_controls, clean)?.into())
}

#[pyfunction]
#[pyo3(name="synth_mcx_2_kg24", signature = (num_controls, clean))]
fn py_synth_mcx_2_kg24(num_controls: usize, clean: bool) -> PyResult<PyCircuitData> {
    Ok(synth_mcx_2_kg24(num_controls, clean)?.into())
}

#[pyfunction]
#[pyo3(name="synth_mcx_n_clean_m15", signature = (num_controls))]
fn py_synth_mcx_n_clean_m15(num_controls: usize) -> PyResult<PyCircuitData> {
    Ok(synth_mcx_n_clean_m15(num_controls)?.into())
}

#[pyfunction]
#[pyo3(name="synth_mcx_noaux_hp24", signature = (num_controls))]
fn py_synth_mcx_noaux_hp24(num_controls: usize) -> PyResult<PyCircuitData> {
    synth_mcx_noaux_hp24(num_controls).map(Into::into)
}

#[pyfunction]
#[pyo3(name="synth_mcx_1_clean_b95", signature = (num_controls))]
fn py_synth_mcx_1_clean_b95(num_controls: usize) -> PyResult<PyCircuitData> {
    Ok(synth_mcx_1_clean_b95(num_controls)?.into())
}

#[pyfunction]
#[pyo3(name = "synth_mcp_noaux_sp22")]
fn py_synth_mcp_noaux_sp22(num_controls: usize, phase: Param) -> PyResult<PyCircuitData> {
    // Reject unsupported types early: PyO3 silently maps unrecognised Python objects to
    // ``Param::Obj``, which would later panic.
    if matches!(phase, Param::Obj(_)) {
        return Err(PyTypeError::new_err(
            "synth_mcp_noaux_sp22 requires phase to be a float or a ParameterExpression.",
        ));
    }
    Ok(synth_mcp_noaux_sp22(num_controls, phase)?.into())
}

#[pyfunction]
#[pyo3(name = "synth_mcx_noaux_sp22")]
fn py_synth_mcx_noaux_sp22(num_controls: usize) -> PyResult<PyCircuitData> {
    Ok(synth_mcx_noaux_sp22(num_controls)?.into())
}

#[pyfunction]
#[pyo3(name = "c3x")]
fn py_c3x() -> PyResult<PyCircuitData> {
    Ok(c3x().into())
}

#[pyfunction]
#[pyo3(name = "c4x")]
fn py_c4x() -> PyResult<PyCircuitData> {
    Ok(c4x().into())
}

#[pyfunction]
#[pyo3(name = "synth_mcx_explicit")]
fn py_synth_mcx_explicit(num_controls: usize) -> PyResult<PyCircuitData> {
    Ok(synth_mcx_explicit(num_controls)?.into())
}

pub fn multi_controlled(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_c3x, m)?)?;
    m.add_function(wrap_pyfunction!(py_c4x, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_explicit, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_n_dirty_i15, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_n_dirty_m15, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_noaux_hp24, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_1_clean_b95, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_1_kg24, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_2_kg24, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcp_noaux_sp22, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_noaux_sp22, m)?)?;
    m.add_function(wrap_pyfunction!(mcmt::mcmt_v_chain, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_n_clean_m15, m)?)?;
    Ok(())
}
