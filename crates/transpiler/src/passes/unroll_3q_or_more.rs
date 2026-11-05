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
use qiskit_circuit::dag_circuit::DAGError;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use crate::QiskitError;
use crate::target::Target;
use qiskit_circuit::dag_circuit::{DAGCircuit, PyDAGCircuit};
use qiskit_circuit::operations::Operation;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Unroll3qError {
    #[error("Cannot unroll all 3q or more gates. No rule to expand {0}")]
    NoDefinition(String),
    #[error("Failed to substitute the definition")]
    SubstitutionError(#[from] DAGError),
}

#[pyfunction]
#[pyo3(name = "unroll_3q_or_more")]
pub fn py_unroll_3q_or_more(dag: &mut PyDAGCircuit, target: Option<&Target>) -> PyResult<()> {
    run_unroll_3q_or_more(dag.try_write()?, target).map_err(|err| match err {
        Unroll3qError::NoDefinition(e) => QiskitError::new_err(format!(
            "Cannot unroll all 3q or more gates. No rule to expand {}",
            e
        )),
        Unroll3qError::SubstitutionError(e) => e.into(),
    })
}

pub fn run_unroll_3q_or_more(
    dag: &mut DAGCircuit,
    target: Option<&Target>,
) -> Result<(), Unroll3qError> {
    let remove_list: Result<Vec<(NodeIndex, DAGCircuit)>, Unroll3qError> = dag
        .op_nodes(false)
        .filter_map(
            |(idx, inst)| -> Option<Result<(NodeIndex, DAGCircuit), Unroll3qError>> {
                if inst.op.num_qubits() < 3 || inst.op.try_control_flow().is_some() {
                    return None;
                }
                if let Some(target) = target
                    && target.contains_key(inst.op.name())
                {
                    return None;
                }
                let definition = match inst.try_definition() {
                    Some(def) => def,
                    None => {
                        return Some(Err(Unroll3qError::NoDefinition(inst.op.name().to_string())));
                    }
                };
                let mut decomp_dag =
                    match DAGCircuit::from_circuit_data(&definition, false, None, None) {
                        Ok(dag) => dag,
                        Err(e) => return Some(Err(Unroll3qError::SubstitutionError(e))),
                    };
                if let Err(e) = run_unroll_3q_or_more(&mut decomp_dag, target) {
                    return Some(Err(e));
                }
                Some(Ok((idx, decomp_dag)))
            },
        )
        .collect();
    for (idx, decomp_dag) in remove_list? {
        dag.substitute_node_with_dag(idx, &decomp_dag, None, None, None, None)
            .map_err(Unroll3qError::SubstitutionError)?;
    }
    Ok(())
}

pub fn unroll_3q_or_more_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_unroll_3q_or_more))?;
    Ok(())
}
