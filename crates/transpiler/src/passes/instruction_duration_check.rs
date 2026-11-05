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

use crate::TranspilerError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use qiskit_circuit::dag_circuit::PyDAGCircuit;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::{DelayUnit, OperationRef, StandardInstruction};

/// Run duration validation passes.
///
/// Args:
///     dag: DAG circuit to check instruction durations.
///     acquire_align: Integer number representing the minimum time resolution to
///         trigger acquisition instruction in units of dt.
///     pulse_align: Integer number representing the minimum time resolution to
///         trigger gate instruction in units of ``dt``.
/// Returns:
///     True if rescheduling is required, False otherwise.

#[pyfunction]
#[pyo3(signature=(dag, acquire_align, pulse_align))]
pub fn run_instruction_duration_check(
    dag: &PyDAGCircuit,
    acquire_align: u32,
    pulse_align: u32,
) -> PyResult<bool> {
    let dag = dag.try_read()?;
    let num_stretches = dag.num_stretches();

    let acquire_align = i64::from(acquire_align);
    let pulse_align = i64::from(pulse_align);

    // Rescheduling is not necessary
    if (acquire_align == 1 && pulse_align == 1) || num_stretches != 0 {
        return Ok(false);
    }

    // Check delay durations
    for (_, packed_op) in dag.op_nodes(false) {
        if let OperationRef::StandardInstruction(StandardInstruction::Delay(unit)) =
            packed_op.op.view()
        {
            let params = packed_op.params_view();
            let param = params.first().ok_or_else(|| {
                PyValueError::new_err("Delay instruction missing duration parameter")
            })?;

            if unit != DelayUnit::DT {
                return Err(TranspilerError::new_err(
                    "Delay duration must have dt unit for checking alignment.",
                ));
            }
            let duration = match param {
                Param::Int(val) => *val,
                _ => {
                    return Err(TranspilerError::new_err(
                        "The provided Delay duration is not an integer.",
                    ));
                }
            };

            if !(duration % acquire_align == 0 || duration % pulse_align == 0) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub fn instruction_duration_check_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_instruction_duration_check))?;
    Ok(())
}
