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

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qiskit_circuit::dag_circuit::{DAGCircuit, PyDAGCircuit};
use qiskit_circuit::packed_instruction::PackedInstruction;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

#[pyfunction]
#[pyo3(name = "filter_op_nodes")]
pub fn py_filter_op_nodes(
    py: Python,
    dag: &mut PyDAGCircuit,
    predicate: &Bound<PyAny>,
) -> PyResult<()> {
    let dag_borrowed = dag.try_read()?;
    let callable = |node: NodeIndex| -> PyResult<bool> {
        let dag_op_node = dag_borrowed.get_node(py, node)?;
        predicate.call1((dag_op_node,))?.extract()
    };
    let mut remove_nodes: Vec<NodeIndex> = Vec::new();
    for node in dag_borrowed.op_node_indices(true) {
        if !callable(node)? {
            remove_nodes.push(node);
        }
    }
    let dag_mut = dag.try_write()?;
    for node in remove_nodes {
        dag_mut.remove_op_node(node);
    }
    Ok(())
}

/// Remove any nodes that have the provided label set
///
/// Args:
///     dag (DAGCircuit): The dag circuit to filter the ops from
///     label (str): The label to filter nodes on
#[pyfunction(name = "filter_labeled_op")]
fn py_filter_labeled_op(dag: &mut PyDAGCircuit, label: String) -> PyResult<()> {
    filter_labeled_op(dag.try_write()?, label);
    Ok(())
}

/// Remove any nodes that have the provided label set
///
/// # Arguments:
/// * `dag` - The dag circuit to filter the ops from
/// * `label` - The label to filter nodes on
pub fn filter_labeled_op(dag: &mut DAGCircuit, label: String) {
    let predicate = |node: &PackedInstruction| -> bool {
        match node.label.as_deref() {
            Some(inst_label) => inst_label != &label,
            None => false,
        }
    };
    dag.filter_op_nodes(predicate);
}

pub fn filter_op_nodes_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_filter_op_nodes))?;
    m.add_wrapped(wrap_pyfunction!(py_filter_labeled_op))?;
    Ok(())
}
