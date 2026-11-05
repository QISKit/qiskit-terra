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

pub mod alap_schedule_analysis;
pub mod asap_schedule_analysis;

use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign,
};

use hashbrown::HashMap;
use pyo3::{
    IntoPyObjectExt,
    exceptions::{PyKeyError, PyTypeError},
    prelude::*,
    types::{PyDict, PyIterator, PyList},
};
use qiskit_circuit::{
    dag_circuit::DAGCircuit,
    dag_node::{DAGNode, DAGOpNode},
};
use qiskit_util::IndexMap;
use rustworkx_core::petgraph::graph::NodeIndex;

use crate::TranspilerError;

pub trait Number:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
{
}

impl<
    T: Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Rem<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign,
> Number for T
{
}

pub trait TimeOps: Copy + PartialOrd + Number {
    fn zero() -> Self;
    fn max<'a>(a: &'a Self, b: &'a Self) -> &'a Self;
}

impl TimeOps for u64 {
    fn zero() -> Self {
        0
    }
    fn max<'a>(a: &'a Self, b: &'a Self) -> &'a Self {
        if a >= b { a } else { b }
    }
}

impl TimeOps for f64 {
    fn zero() -> Self {
        0.0
    }
    fn max<'a>(a: &'a Self, b: &'a Self) -> &'a Self {
        if a >= b { a } else { b }
    }
}

/// Mapping between :class:`.DAGopNode` and its durations either in values
/// of DT (`int`) or Seconds (`float`).
#[pyclass(
    mapping,
    name = "NodeDurations",
    module = "qiskit._accelerate.scheduling",
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyNodeDurations {
    inner: NodeDurations,
    nodes_mapping: HashMap<NodeIndex<u32>, Py<DAGOpNode>>,
}

impl Deref for PyNodeDurations {
    type Target = NodeDurations;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for PyNodeDurations {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[pymethods]
impl PyNodeDurations {
    #[new]
    fn new(mapping: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut idx_to_node = HashMap::default();
        if let Some((_, key)) = mapping.iter().next() {
            if key.extract::<u64>().is_ok() {
                // All durations are of type u64
                let mut op_durations = IndexMap::default();
                for (py_node, py_duration) in mapping.iter() {
                    let node = py_node.cast_into::<DAGOpNode>()?;
                    let node_idx = node
                        .cast::<DAGNode>()?
                        .borrow()
                        .node
                        .expect("Node index not found.");
                    let val = py_duration.extract()?;
                    op_durations.insert(node_idx, val);
                    idx_to_node.insert(node_idx, node.unbind());
                }
                Ok(PyNodeDurations {
                    inner: NodeDurations::Dt(op_durations),
                    nodes_mapping: idx_to_node,
                })
            } else if key.extract::<f64>().is_ok() {
                // All durations are of type f64
                let mut op_durations = IndexMap::default();
                for (py_node, py_duration) in mapping.iter() {
                    let node = py_node.cast_into::<DAGOpNode>()?;
                    let node_idx = node
                        .cast::<DAGNode>()?
                        .borrow()
                        .node
                        .expect("Node index not found.");
                    let val = py_duration.extract()?;
                    op_durations.insert(node_idx, val);
                    idx_to_node.insert(node_idx, node.unbind());
                }
                Ok(PyNodeDurations {
                    inner: NodeDurations::Seconds(op_durations),
                    nodes_mapping: idx_to_node,
                })
            } else {
                Err(PyTypeError::new_err(
                    "Only integer or float types allowed for durations",
                ))
            }
        } else {
            Ok(PyNodeDurations {
                inner: NodeDurations::Dt(Default::default()),
                nodes_mapping: Default::default(),
            })
        }
    }

    fn __getitem__<'py>(&'py self, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let node_as_base: &Bound<DAGNode> = node.cast()?;
        let py = node.py();
        let idx = node_as_base.borrow().node.expect("Node index not found.");
        match &self.inner {
            NodeDurations::Dt(map) => map
                .get(&idx)
                .ok_or_else(|| PyKeyError::new_err(format!("key {} not in mapping", node)))?
                .into_bound_py_any(py),
            NodeDurations::Seconds(map) => map
                .get(&idx)
                .ok_or_else(|| PyKeyError::new_err(format!("key {} not in mapping", node)))?
                .into_bound_py_any(py),
        }
    }

    fn __setitem__<'py>(
        &'py mut self,
        node: Bound<'py, DAGOpNode>,
        value: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let node_as_base: &Bound<DAGNode> = node.cast()?;
        let idx = node_as_base.borrow().node.expect("Node index not found.");
        match &mut self.inner {
            NodeDurations::Dt(map) => {
                let value = value.extract()?;
                map.entry(idx)
                    .and_modify(|val| *val = value)
                    .or_insert(value);
            }
            NodeDurations::Seconds(map) => {
                let value = value.extract()?;
                map.entry(node_as_base.borrow().node.expect("Node index not found."))
                    .and_modify(|val| *val = value)
                    .or_insert(value);
            }
        }
        self.nodes_mapping
            .entry(idx)
            .and_modify(|val| *val = node.clone().unbind())
            .or_insert(node.unbind());
        Ok(())
    }

    fn __delitem__<'py>(&'py mut self, object: &Bound<'py, PyAny>) -> PyResult<()> {
        let node_as_base: &Bound<DAGNode> = object.cast()?;
        let idx = node_as_base.borrow().node.expect("Node index not found.");
        if self.nodes_mapping.remove(&idx).is_some() {
            match &mut self.inner {
                NodeDurations::Dt(hash_map) => {
                    hash_map.swap_remove(&idx);
                }
                NodeDurations::Seconds(hash_map) => {
                    hash_map.swap_remove(&idx);
                }
            }
            Ok(())
        } else {
            Err(PyKeyError::new_err(format!(
                "key '{}' not present in mapping",
                object.repr()?
            )))
        }
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.keys(py)?.as_any().try_iter()
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match &self.inner {
            NodeDurations::Dt(index_map) => {
                PyList::new(py, index_map.keys().map(|key| &self.nodes_mapping[key]))
            }
            NodeDurations::Seconds(index_map) => {
                PyList::new(py, index_map.keys().map(|key| &self.nodes_mapping[key]))
            }
        }
    }

    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match &self.inner {
            NodeDurations::Dt(index_map) => PyList::new(py, index_map.values()),
            NodeDurations::Seconds(index_map) => PyList::new(py, index_map.values()),
        }
    }

    fn items<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match &self.inner {
            NodeDurations::Dt(map) => {
                PyList::new(py, map.iter().map(|(k, v)| (&self.nodes_mapping[k], *v)))
            }
            NodeDurations::Seconds(map) => {
                PyList::new(py, map.iter().map(|(k, v)| (&self.nodes_mapping[k], *v)))
            }
        }
    }

    fn __contains__<'py>(&'py self, object: Bound<'py, PyAny>) -> PyResult<bool> {
        Ok(object
            .cast_into::<DAGOpNode>()? // Extra check to make sure we are retrieving an op node.
            .cast::<DAGNode>()
            .map(|node| {
                let node_index = node.borrow().node.expect("Node index not found.");
                self.nodes_mapping.contains_key(&node_index)
            })?)
    }

    fn clear(&mut self) {
        self.inner.clear();
        self.nodes_mapping.clear();
    }

    fn __eq__(&self, other: Bound<PyAny>) -> PyResult<bool> {
        if self.len() != other.len()? {
            return Ok(false);
        }
        if let Ok(other_nodes) = other.cast::<PyNodeDurations>() {
            Ok(other_nodes.borrow().eq(self))
        } else if let Ok(as_dict) = other.cast::<PyDict>() {
            let as_node_durations = PyNodeDurations::new(as_dict)?;
            Ok(as_node_durations.inner == self.inner)
        } else {
            Err(PyTypeError::new_err(format!(
                "'{}' is not an instance of 'NodeDurations'",
                other.get_type(),
            )))
        }
    }

    fn copy(&self) -> Self {
        self.__copy__()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn pop<'py>(&'py mut self, node: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let node_as_base: &Bound<DAGNode> = node.cast()?;
        let idx = node_as_base.borrow().node.expect("Node index not found.");
        let py = node.py();
        self.nodes_mapping.remove(&idx);

        match &mut self.inner {
            NodeDurations::Dt(hash_map) => hash_map.swap_remove(&idx).into_bound_py_any(py),
            NodeDurations::Seconds(hash_map) => hash_map.swap_remove(&idx).into_bound_py_any(py),
        }
    }
}

impl PyNodeDurations {
    pub fn from_durations(py: Python, dag: &DAGCircuit, updated: NodeDurations) -> PyResult<Self> {
        let nodes_mapping = match &updated {
            NodeDurations::Dt(new) => new
                .keys()
                .map(|idx| -> PyResult<_> {
                    Ok((
                        *idx,
                        dag.get_node(py, *idx)?.cast_bound(py)?.clone().unbind(),
                    ))
                })
                .collect::<PyResult<_>>()?,
            NodeDurations::Seconds(new) => new
                .keys()
                .map(|idx| -> PyResult<_> {
                    Ok((
                        *idx,
                        dag.get_node(py, *idx)?.cast_bound(py)?.clone().unbind(),
                    ))
                })
                .collect::<PyResult<_>>()?,
        };
        Ok(Self {
            inner: updated,
            nodes_mapping,
        })
    }

    pub fn update_durations(&mut self, updated: NodeDurations) -> PyResult<()> {
        if updated.len() > self.len() {
            return Err(TranspilerError::new_err(format!(
                "Mismatched number of durations provided. Expected '<={}', got '{}'",
                self.len(),
                updated.len()
            )));
        }
        match (&mut self.inner, updated) {
            (NodeDurations::Dt(old), NodeDurations::Dt(new)) => {
                for (node, duration) in new {
                    let Some(value) = old.get_mut(&node) else {
                        return Err(PyKeyError::new_err(format!(
                            "Node index '{}' not present in durations.",
                            node.index()
                        )));
                    };
                    *value = duration;
                }
                Ok(())
            }
            (NodeDurations::Seconds(old), NodeDurations::Seconds(new)) => {
                for (node, duration) in new {
                    let Some(value) = old.get_mut(&node) else {
                        return Err(PyKeyError::new_err(format!(
                            "Node index '{}' not present in durations.",
                            node.index()
                        )));
                    };
                    *value = duration;
                }
                Ok(())
            }
            (NodeDurations::Dt(_), NodeDurations::Seconds(_)) => Err(PyTypeError::new_err(
                "The provided durations are not of the expected type. Expected 'Dt' got 'Seconds'",
            )),
            (NodeDurations::Seconds(_), NodeDurations::Dt(_)) => Err(PyTypeError::new_err(
                "The provided durations are not of the expected type. Expected 'Seconds' got 'Dt'",
            )),
        }
    }
}

/// A mapping between a DAG's [NodeIndex] and its duration values.
///
/// A duration may be in units of Dt, represented by a `u64`, or Seconds,
/// representing by an `f64`.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeDurations {
    Dt(IndexMap<NodeIndex<u32>, u64>),
    Seconds(IndexMap<NodeIndex<u32>, f64>),
}

impl NodeDurations {
    pub fn clear(&mut self) {
        match self {
            NodeDurations::Dt(map) => map.clear(),
            NodeDurations::Seconds(map) => map.clear(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            NodeDurations::Dt(hash_map) => hash_map.len(),
            NodeDurations::Seconds(hash_map) => hash_map.len(),
        }
    }
}

impl From<IndexMap<NodeIndex<u32>, u64>> for NodeDurations {
    fn from(value: IndexMap<NodeIndex<u32>, u64>) -> Self {
        Self::Dt(value)
    }
}

impl From<IndexMap<NodeIndex<u32>, f64>> for NodeDurations {
    fn from(value: IndexMap<NodeIndex<u32>, f64>) -> Self {
        Self::Seconds(value)
    }
}

pub fn scheduling_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyNodeDurations>()?;
    Ok(())
}
