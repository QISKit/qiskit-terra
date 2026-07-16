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

use std::hash::Hasher;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use crate::TupleLikeArg;
use crate::circuit_data::CircuitData;
use crate::circuit_instruction::{CircuitInstruction, OperationFromPython, extract_params};
use crate::operations::{Operation, OperationRef, Param};

use ahash::AHasher;
use approx::relative_eq;
use num_complex::Complex64;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use crate::instruction::Instruction;
use numpy::IntoPyArray;
use numpy::PyArray2;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{PyResult, intern};

/// Parent class for DAGOpNode, DAGInNode, and DAGOutNode.
#[pyclass(module = "qiskit._accelerate.circuit", subclass, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct DAGNode {
    pub node: Option<NodeIndex>,
}

impl DAGNode {
    #[inline]
    pub fn py_nid(&self) -> isize {
        self.node
            .map(|node| node.index().try_into().unwrap())
            .unwrap_or(-1)
    }
}

#[pymethods]
impl DAGNode {
    #[new]
    #[pyo3(signature=(nid=-1))]
    fn py_new(nid: isize) -> PyResult<Self> {
        Ok(DAGNode {
            node: match nid {
                -1 => None,
                nid => {
                    let index: usize = match nid.try_into() {
                        Ok(index) => index,
                        Err(_) => {
                            return Err(PyValueError::new_err(
                                "Invalid node index, must be -1 or a non-negative integer",
                            ));
                        }
                    };
                    Some(NodeIndex::new(index))
                }
            },
        })
    }

    #[getter(_node_id)]
    fn get_py_node_id(&self) -> isize {
        self.py_nid()
    }

    #[setter(_node_id)]
    fn set_py_node_id(&mut self, nid: isize) {
        self.node = match nid {
            -1 => None,
            nid => Some(NodeIndex::new(nid.try_into().unwrap())),
        }
    }

    fn __getstate__(&self) -> Option<usize> {
        self.node.map(|node| node.index())
    }

    #[pyo3(signature=(index=None))]
    fn __setstate__(&mut self, index: Option<usize>) {
        self.node = index.map(NodeIndex::new);
    }

    fn __lt__(&self, other: &DAGNode) -> bool {
        self.py_nid() < other.py_nid()
    }

    fn __gt__(&self, other: &DAGNode) -> bool {
        self.py_nid() > other.py_nid()
    }

    fn __str__(_self: &Bound<DAGNode>) -> String {
        format!("{}", _self.as_ptr() as usize)
    }

    fn __hash__(&self, py: Python) -> PyResult<isize> {
        self.py_nid().into_pyobject(py)?.hash()
    }
}

/// Object to represent an Instruction at a node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.circuit", extends=DAGNode)]
pub struct DAGOpNode {
    pub instruction: CircuitInstruction,
}

#[pymethods]
impl DAGOpNode {
    #[new]
    #[pyo3(signature = (op, qargs=None, cargs=None))]
    pub fn py_new(
        py: Python,
        op: Bound<PyAny>,
        qargs: Option<TupleLikeArg>,
        cargs: Option<TupleLikeArg>,
    ) -> PyResult<Py<Self>> {
        let py_op = op.extract::<OperationFromPython<CircuitData>>()?;
        let qargs = qargs.map_or_else(|| PyTuple::empty(py), |q| q.value);
        let cargs = cargs.map_or_else(|| PyTuple::empty(py), |c| c.value);
        let instruction = CircuitInstruction {
            operation: py_op.operation,
            qubits: qargs.unbind(),
            clbits: cargs.unbind(),
            params: py_op.params,
            label: py_op.label,
            #[cfg(feature = "cache_pygates")]
            py_op: op.unbind().into(),
        };

        Py::new(py, (DAGOpNode { instruction }, DAGNode { node: None }))
    }

    fn __hash__(slf: PyRef<'_, Self>) -> PyResult<u64> {
        let super_ = slf.as_ref();
        let mut hasher = AHasher::default();
        hasher.write_isize(super_.py_nid());
        hasher.write(slf.instruction.operation.name().as_bytes());
        Ok(hasher.finish())
    }

    fn __eq__(slf: PyRef<Self>, py: Python, other: &Bound<PyAny>) -> PyResult<bool> {
        // This check is more restrictive by design as it's intended to replace
        // object identitity for set/dict membership and not be a semantic equivalence
        // check. We have an implementation of that as part of `DAGCircuit.__eq__` and
        // this method is specifically to ensure nodes are the same. This means things
        // like parameter equality are stricter to reject things like
        // Param::Float(0.1) == Param::ParameterExpression(0.1) (if the expression was
        // a python parameter equivalent to a bound value).
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let borrowed_other = other.borrow();
        let other_super = borrowed_other.as_ref();
        let super_ = slf.as_ref();

        if super_.py_nid() != other_super.py_nid() {
            return Ok(false);
        }
        if !slf
            .instruction
            .operation
            .py_eq(py, &borrowed_other.instruction.operation)?
        {
            return Ok(false);
        }
        let params_eq = match (
            slf.instruction.operation.view(),
            borrowed_other.instruction.operation.view(),
        ) {
            (OperationRef::StandardGate(_), OperationRef::StandardGate(_))
            | (OperationRef::StandardInstruction(_), OperationRef::StandardInstruction(_)) => {
                let slf_params = slf.instruction.params_view();
                let other_params = borrowed_other.instruction.params_view();
                let mut params_eq = true;
                for (a, b) in slf_params.iter().zip(other_params) {
                    let res = match [a, b] {
                        [Param::Float(float_a), Param::Float(float_b)] => {
                            relative_eq!(float_a, float_b, max_relative = 1e-10)
                        }
                        [
                            Param::ParameterExpression(param_a),
                            Param::ParameterExpression(param_b),
                        ] => param_a == param_b,
                        [Param::Obj(param_a), Param::Obj(param_b)] => {
                            param_a.bind(py).eq(param_b)?
                        }
                        _ => false,
                    };
                    if !res {
                        params_eq = false;
                        break;
                    }
                }
                params_eq
            }
            (OperationRef::ControlFlow(_), OperationRef::ControlFlow(_)) => {
                let slf_blocks = slf.instruction.blocks_view();
                let other_blocks = borrowed_other.instruction.blocks_view();
                let mut params_eq = true;
                // TODO: we should be able to do the semantic-equality comparison from Rust
                // space in the future, without going via Python.  See gh-15267.
                for (a, b) in slf_blocks.iter().zip(other_blocks) {
                    if !a
                        .clone()
                        .into_py_quantum_circuit(py)?
                        .eq(b.clone().into_py_quantum_circuit(py)?)?
                    {
                        params_eq = false;
                        break;
                    }
                }
                params_eq
            }
            _ => {
                // For our Python object types (i.e. PyInstruction, PyGate, PyOperation), we've
                // already evaluated the parameters are equal here via the Python space operation
                // equality check.
                true
            }
        };

        Ok(params_eq
            && slf
                .instruction
                .qubits
                .bind(py)
                .eq(borrowed_other.instruction.qubits.clone_ref(py))?
            && slf
                .instruction
                .clbits
                .bind(py)
                .eq(borrowed_other.instruction.clbits.clone_ref(py))?)
    }

    #[pyo3(signature = (instruction, /, *, deepcopy=false))]
    #[staticmethod]
    fn from_instruction(
        py: Python,
        mut instruction: CircuitInstruction,
        deepcopy: bool,
    ) -> PyResult<Py<PyAny>> {
        if deepcopy {
            instruction.operation = instruction.operation.py_deepcopy(py, None)?;
            #[cfg(feature = "cache_pygates")]
            {
                instruction.py_op = OnceLock::new();
            }
        }
        let base = PyClassInitializer::from(DAGNode { node: None });
        let sub = base.add_subclass(DAGOpNode { instruction });
        Py::new(py, sub)?.into_py_any(py)
    }

    fn __reduce__(slf: PyRef<Self>, py: Python) -> PyResult<Py<PyAny>> {
        let state = slf.as_ref().node.map(|node| node.index());
        let temp = (
            slf.instruction.get_operation(py)?,
            &slf.instruction.qubits,
            &slf.instruction.clbits,
        );
        (py.get_type::<Self>(), temp, state).into_py_any(py)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let index: Option<usize> = state.extract()?;
        slf.as_mut().node = index.map(NodeIndex::new);
        Ok(())
    }

    /// Get a `CircuitInstruction` that represents the same information as this `DAGOpNode`.  If
    /// `deepcopy`, any internal Python objects are deep-copied.
    ///
    /// Note: this ought to be a temporary method, while the DAG/QuantumCircuit converters still go
    /// via Python space; this still involves copy-out and copy-in of the data, whereas doing it all
    /// within Rust space could directly re-pack the instruction from a `DAGOpNode` to a
    /// `PackedInstruction` with no intermediate copy.
    #[pyo3(signature = (/, *, deepcopy=false))]
    fn _to_circuit_instruction(&self, py: Python, deepcopy: bool) -> PyResult<CircuitInstruction> {
        Ok(CircuitInstruction {
            operation: if deepcopy {
                self.instruction.operation.py_deepcopy(py, None)?
            } else {
                self.instruction.operation.clone()
            },
            qubits: self.instruction.qubits.clone_ref(py),
            clbits: self.instruction.clbits.clone_ref(py),
            params: self.instruction.params.clone(),
            label: self.instruction.label.clone(),
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        })
    }

    #[getter]
    fn get_op(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.instruction.get_operation(py)
    }

    #[setter]
    fn set_op(&mut self, op: &Bound<PyAny>) -> PyResult<()> {
        let res = op.extract::<OperationFromPython<CircuitData>>()?;
        self.instruction.operation = res.operation;
        self.instruction.params = res.params;
        self.instruction.label = res.label;
        #[cfg(feature = "cache_pygates")]
        {
            self.instruction.py_op = op.clone().unbind().into();
        }
        Ok(())
    }

    #[getter]
    fn num_qubits(&self) -> u32 {
        self.instruction.operation.num_qubits()
    }

    #[getter]
    fn num_clbits(&self) -> u32 {
        self.instruction.operation.num_clbits()
    }

    #[getter]
    pub fn get_qargs(&self, py: Python) -> Py<PyTuple> {
        self.instruction.qubits.clone_ref(py)
    }

    #[setter]
    fn set_qargs(&mut self, qargs: Py<PyTuple>) {
        self.instruction.qubits = qargs;
    }

    #[getter]
    pub fn get_cargs(&self, py: Python) -> Py<PyTuple> {
        self.instruction.clbits.clone_ref(py)
    }

    #[setter]
    fn set_cargs(&mut self, cargs: Py<PyTuple>) {
        self.instruction.clbits = cargs;
    }

    /// Returns the Instruction name corresponding to the op for this node
    #[getter]
    fn get_name(&self) -> &str {
        self.instruction.operation.name()
    }

    #[getter]
    fn get_params(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.instruction.get_params(py)
    }

    #[setter]
    fn set_params(&mut self, val: Bound<PyAny>) -> PyResult<()> {
        self.instruction.params = extract_params(self.instruction.op(), &val)?;
        Ok(())
    }

    #[getter]
    fn matrix<'py>(&'py self, py: Python<'py>) -> Option<Bound<'py, PyArray2<Complex64>>> {
        let matrix = self.instruction.try_matrix();
        matrix.map(|mat| mat.into_pyarray(py))
    }

    #[getter]
    fn label(&self) -> Option<&str> {
        self.instruction.label.as_ref().map(|x| x.as_str())
    }

    /// Is the :class:`.Operation` contained in this node a Qiskit standard gate?
    pub fn is_standard_gate(&self) -> bool {
        self.instruction.is_standard_gate()
    }

    /// Is the :class:`.Operation` contained in this node a subclass of :class:`.ControlledGate`?
    pub fn is_controlled_gate(&self, py: Python) -> PyResult<bool> {
        self.instruction.is_controlled_gate(py)
    }

    /// Is the :class:`.Operation` contained in this node a directive?
    pub fn is_directive(&self) -> bool {
        self.instruction.is_directive()
    }

    /// Is the :class:`.Operation` contained in this node a control-flow operation (i.e. an instance
    /// of :class:`.ControlFlowOp`)?
    pub fn is_control_flow(&self) -> bool {
        self.instruction.is_control_flow()
    }

    /// Does this node contain any :class:`.ParameterExpression` parameters?
    pub fn is_parameterized(&self) -> bool {
        self.instruction.is_parameterized()
    }

    #[setter]
    fn set_label(&mut self, val: Option<String>) {
        self.instruction.label = val.map(Box::new);
    }

    #[getter]
    fn definition<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let definition = match self.instruction.operation.view() {
            OperationRef::StandardGate(g) => g.definition(self.instruction.params_view()),
            OperationRef::PyCustom(i) => i.definition(),
            _ => None,
        };
        definition
            .map(|data| data.into_py_quantum_circuit(py))
            .transpose()
    }

    /// Sets the Instruction name corresponding to the op for this node
    #[setter]
    fn set_name(&mut self, py: Python, new_name: Py<PyAny>) -> PyResult<()> {
        let op = self.instruction.get_operation_mut(py)?;
        op.setattr(intern!(py, "name"), new_name)?;
        self.instruction.operation = op.extract::<OperationFromPython<CircuitData>>()?.operation;
        Ok(())
    }

    /// Returns a representation of the DAGOpNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "DAGOpNode(op={}, qargs={}, cargs={})",
            self.instruction.get_operation(py)?.bind(py).repr()?,
            self.instruction.qubits.bind(py).repr()?,
            self.instruction.clbits.bind(py).repr()?
        ))
    }
}

/// Object to represent an incoming wire node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.circuit", extends=DAGNode)]
pub struct DAGInNode {
    #[pyo3(get)]
    pub wire: Py<PyAny>,
}

impl DAGInNode {
    pub fn new(node: NodeIndex, wire: Py<PyAny>) -> (Self, DAGNode) {
        (DAGInNode { wire }, DAGNode { node: Some(node) })
    }
}

#[pymethods]
impl DAGInNode {
    #[new]
    fn py_new(wire: Py<PyAny>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(DAGNode { node: None }).add_subclass(Self { wire })
    }

    fn __reduce__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let state = slf.as_ref().node.map(|node| node.index());
        (py.get_type::<Self>(), (&slf.wire,), state).into_pyobject(py)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let index: Option<usize> = state.extract()?;
        slf.as_mut().node = index.map(NodeIndex::new);
        Ok(())
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python) -> PyResult<u64> {
        let super_ = slf.as_ref();
        let mut hasher = AHasher::default();
        hasher.write_isize(super_.py_nid());
        hasher.write_isize(slf.wire.bind(py).hash()?);
        Ok(hasher.finish())
    }

    fn __eq__(slf: PyRef<Self>, py: Python, other: &Bound<PyAny>) -> PyResult<bool> {
        match other.cast::<Self>() {
            Ok(other) => {
                let borrowed_other = other.borrow();
                let other_super = borrowed_other.as_ref();
                let super_ = slf.as_ref();
                Ok(super_.py_nid() == other_super.py_nid()
                    && slf.wire.bind(py).eq(borrowed_other.wire.clone_ref(py))?)
            }
            Err(_) => Ok(false),
        }
    }

    /// Returns a representation of the DAGInNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!("DAGInNode(wire={})", self.wire.bind(py).repr()?))
    }
}

/// Object to represent an outgoing wire node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.circuit", extends=DAGNode)]
pub struct DAGOutNode {
    #[pyo3(get)]
    pub wire: Py<PyAny>,
}

impl DAGOutNode {
    pub fn new(node: NodeIndex, wire: Py<PyAny>) -> (Self, DAGNode) {
        (DAGOutNode { wire }, DAGNode { node: Some(node) })
    }
}

#[pymethods]
impl DAGOutNode {
    #[new]
    fn py_new(wire: Py<PyAny>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(DAGNode { node: None }).add_subclass(Self { wire })
    }

    fn __reduce__(slf: PyRef<Self>, py: Python) -> PyResult<Py<PyAny>> {
        let state = slf.as_ref().node.map(|node| node.index());
        (py.get_type::<Self>(), (&slf.wire,), state).into_py_any(py)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let index: Option<usize> = state.extract()?;
        slf.as_mut().node = index.map(NodeIndex::new);
        Ok(())
    }

    fn __hash__(slf: PyRef<'_, Self>, py: Python) -> PyResult<u64> {
        let super_ = slf.as_ref();
        let mut hasher = AHasher::default();
        hasher.write_isize(super_.py_nid());
        hasher.write_isize(slf.wire.bind(py).hash()?);
        Ok(hasher.finish())
    }

    /// Returns a representation of the DAGOutNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!("DAGOutNode(wire={})", self.wire.bind(py).repr()?))
    }

    fn __eq__(slf: PyRef<Self>, py: Python, other: &Bound<PyAny>) -> PyResult<bool> {
        match other.cast::<Self>() {
            Ok(other) => {
                let borrowed_other = other.borrow();
                let other_super = borrowed_other.as_ref();
                let super_ = slf.as_ref();
                Ok(super_.py_nid() == other_super.py_nid()
                    && slf.wire.bind(py).eq(borrowed_other.wire.clone_ref(py))?)
            }
            Err(_) => Ok(false),
        }
    }
}
