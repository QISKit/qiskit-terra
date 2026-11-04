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

use rustworkx_core::petgraph::prelude::*;

use crate::angle_bound_registry::{PyWrapAngleRegistry, WrapAngleRegistry};
use crate::target::Target;
use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, PyDAGCircuit};
use qiskit_circuit::operations::{Operation, Param};

#[pyfunction]
#[pyo3(name = "wrap_angles")]
pub fn py_run_wrap_angles(
    dag: &mut PyDAGCircuit,
    target: &Target,
    bounds_registry: &PyWrapAngleRegistry,
) -> PyResult<()> {
    run_wrap_angles(dag.try_write()?, target, bounds_registry.get_inner())
}

pub fn run_wrap_angles(
    dag: &mut DAGCircuit,
    target: &Target,
    bounds_registry: &WrapAngleRegistry,
) -> PyResult<()> {
    if !target.has_angle_bounds() {
        return Ok(());
    }
    let nodes_to_sub: Vec<NodeIndex> = dag
        .op_nodes(false)
        .filter_map(|(index, inst)| {
            if target.gate_has_angle_bounds(inst.op.name()) && !inst.is_parameterized() {
                Some(index)
            } else {
                None
            }
        })
        .collect();
    for node in nodes_to_sub {
        let inst = dag[node].unwrap_operation();
        let params: Vec<_> = inst
            .params_view()
            .iter()
            .map(|param| {
                let Param::Float(param) = param else {
                    unreachable!()
                };
                *param
            })
            .collect();
        if !target.gate_supported_angle_bound(inst.op.name(), &params) {
            let qargs: Vec<_> = dag
                .get_qargs(inst.qubits)
                .iter()
                .map(|x| PhysicalQubit(x.0))
                .collect();
            let new_dag =
                bounds_registry.substitute_angle_bounds(inst.op.name(), &params, &qargs)?;
            if let Some(new_dag) = new_dag {
                dag.substitute_node_with_dag(node, &new_dag, None, None, None, None)?;
            }
        }
    }
    Ok(())
}

pub fn wrap_angles_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_wrap_angles))?;
    Ok(())
}
