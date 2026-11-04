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

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qiskit_circuit::dag_circuit::{NodeType, PyDAGCircuit, Wire};
use qiskit_circuit::operations::{DelayUnit, Operation, OperationRef, Param, StandardInstruction};

use qiskit_transpiler::target::Target;

use crate::QiskitError;
use qiskit_circuit::PhysicalQubit;
use rustworkx_core::dag_algo::longest_path;
use rustworkx_core::petgraph::stable_graph::StableDiGraph;
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences};

/// Estimate the duration of a scheduled circuit in seconds
#[pyfunction]
pub(crate) fn compute_estimated_duration(dag: &PyDAGCircuit, target: &Target) -> PyResult<f64> {
    let dt = target.dt;
    let dag = dag.try_read()?;
    let get_duration =
        |edge: <&StableDiGraph<NodeType, Wire> as IntoEdgeReferences>::EdgeRef| -> PyResult<f64> {
            let node_weight = &dag[edge.target()];
            match node_weight {
                NodeType::Operation(inst) => {
                    let name = inst.op.name();
                    let qubits = dag.get_qargs(inst.qubits);
                    let physical_qubits: Vec<PhysicalQubit> =
                        qubits.iter().map(|x| PhysicalQubit::new(x.0)).collect();

                    if let OperationRef::StandardInstruction(op) = inst.op.view() {
                        if let StandardInstruction::Delay(unit) = op {
                            let dur = &inst.params_view()[0];
                            return if unit == DelayUnit::DT {
                                if let Some(dt) = dt {
                                    match dur {
                                        Param::Float(val) => Ok(val * dt),
                                        Param::Obj(val) => Python::attach(|py| {
                                            let dur_float: f64 = val.extract(py)?;
                                            Ok(dur_float * dt)
                                        }),
                                        Param::ParameterExpression(_) => Err(QiskitError::new_err(
                                            "Circuit contains parameterized delays, can't compute a duration estimate with this circuit",
                                        )),
                                        Param::Int(value) => Ok(*value as f64 * dt),
                                    }
                                } else {
                                    Err(QiskitError::new_err(
                                        "Circuit contains delays in dt but the target doesn't specify dt",
                                    ))
                                }
                            } else if unit == DelayUnit::S {
                                match dur {
                                    Param::Float(val) => Ok(*val),
                                    _ => Err(QiskitError::new_err(
                                        "Invalid type for parameter value for delay in circuit",
                                    )),
                                }
                            } else {
                                Err(QiskitError::new_err(
                                    "Circuit contains delays in units other then seconds or dt, the circuit is not scheduled.",
                                ))
                            };
                        } else if let StandardInstruction::Barrier(_) = op {
                            return Ok(0.);
                        }
                    }
                    match target.get_duration(name, &physical_qubits) {
                        Some(dur) => Ok(dur),
                        None => Err(QiskitError::new_err(format!(
                            "Duration not found for {name} on qubits: {qubits:?}"
                        ))),
                    }
                }
                NodeType::QubitOut(_) | NodeType::ClbitOut(_) => Ok(0.),
                NodeType::ClbitIn(_) | NodeType::QubitIn(_) => {
                    Err(QiskitError::new_err("Invalid circuit provided"))
                }
                _ => Err(QiskitError::new_err(
                    "Circuit contains Vars, duration can't be calculated with classical variables",
                )),
            }
        };
    match longest_path(dag.dag(), get_duration)? {
        Some((_, weight)) => Ok(weight),
        None => Err(QiskitError::new_err("Invalid circuit provided")),
    }
}

pub fn compute_duration(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(compute_estimated_duration))?;
    Ok(())
}
