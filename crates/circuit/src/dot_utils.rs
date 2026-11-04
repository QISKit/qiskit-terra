// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// This module is forked from rustworkx at:
// https://github.com/Qiskit/rustworkx/blob/c4256daf96fc3c08c392450ed33bc0987cdb15ff/src/dot_utils.rs
// and has been modified to generate a dot file from a Rust DAGCircuit instead
// of a rustworkx PyGraph object

use std::collections::BTreeMap;
use std::io::prelude::*;

use crate::dag_circuit::{DAGCircuit, Wire};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use rustworkx_core::petgraph::visit::{
    EdgeRef, IntoEdgeReferences, IntoNodeReferences, NodeIndexable, NodeRef,
};

static TYPE: [&str; 2] = ["graph", "digraph"];
static EDGE: [&str; 2] = ["--", "->"];

pub fn build_dot<T>(
    py: Python,
    dag: &DAGCircuit,
    file: &mut T,
    graph_attrs: Option<BTreeMap<String, String>>,
    node_attrs: Option<Py<PyAny>>,
    edge_attrs: Option<Py<PyAny>>,
) -> PyResult<()>
where
    T: Write,
{
    let graph = dag.dag();
    writeln!(file, "{} {{", TYPE[graph.is_directed() as usize])?;
    if let Some(graph_attr_map) = graph_attrs {
        for (key, value) in graph_attr_map.iter() {
            writeln!(file, "{key}={value} ;")?;
        }
    }

    for node in graph.node_references() {
        let node_weight = dag.get_node(py, node.id())?;
        writeln!(
            file,
            "{} {};",
            graph.to_index(node.id()),
            attr_map_to_string(py, node_attrs.as_ref(), node_weight)?
        )?;
    }
    for edge in graph.edge_references() {
        let edge_weight = match edge.weight() {
            Wire::Qubit(qubit) => dag.qubits().get(*qubit).cloned().into_bound_py_any(py)?,
            Wire::Clbit(clbit) => dag.clbits().get(*clbit).cloned().into_bound_py_any(py)?,
            Wire::Var(var) => dag.vars().get(*var).cloned().into_bound_py_any(py)?,
        };
        writeln!(
            file,
            "{} {} {} {};",
            graph.to_index(edge.source()),
            EDGE[graph.is_directed() as usize],
            graph.to_index(edge.target()),
            attr_map_to_string(py, edge_attrs.as_ref(), edge_weight)?
        )?;
    }
    writeln!(file, "}}")?;
    Ok(())
}

static ATTRS_TO_ESCAPE: [&str; 2] = ["label", "tooltip"];

/// Convert an attr map to an output string
fn attr_map_to_string<'py, T: IntoPyObject<'py>>(
    py: Python<'py>,
    attrs: Option<&'py Py<PyAny>>,
    weight: T,
) -> PyResult<String>
where
    <T as pyo3::IntoPyObject<'py>>::Output: pyo3::IntoPyObject<'py>,
    <T as pyo3::IntoPyObject<'py>>::Error: std::fmt::Debug,
{
    if attrs.is_none() {
        return Ok("".to_string());
    }
    let attr_callable = |node: T| -> PyResult<BTreeMap<String, String>> {
        let res = attrs
            .unwrap()
            .call1(py, (node.into_pyobject(py).unwrap(),))?;
        res.extract(py)
    };

    let attrs = attr_callable(weight)?;
    if attrs.is_empty() {
        return Ok("".to_string());
    }
    let attr_string = attrs
        .iter()
        .map(|(key, value)| {
            if ATTRS_TO_ESCAPE.contains(&key.as_str()) {
                format!("{key}=\"{value}\"")
            } else {
                format!("{key}={value}")
            }
        })
        .collect::<Vec<String>>()
        .join(", ");
    Ok(format!("[{attr_string}]"))
}
