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

use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use foldhash::fast::RandomState;
use smallvec::SmallVec;

use crate::bit::{
    BitLocations, ClassicalRegister, PyClbit, PyQubit, QuantumRegister, Register, ShareableClbit,
    ShareableQubit,
};
use crate::bit_locator::BitLocator;
use crate::circuit_data::CircuitData;
use crate::circuit_instruction::{CircuitInstruction, OperationFromPython};
use crate::classical::expr;
use crate::converters::QuantumCircuitData;
use crate::dag_node::{DAGInNode, DAGNode, DAGOpNode, DAGOutNode};
use crate::dot_utils::build_dot;
use crate::error::DAGCircuitError;
use crate::interner::{Interned, InternedMap, Interner};
use crate::object_registry::ObjectRegistry;
use crate::operations::{
    ArrayType, BoxDuration, Condition, ControlFlow, ControlFlowInstruction, ControlFlowView,
    LoopParam, Operation, OperationRef, Param, PyInstruction, PyOpKind, StandardGate,
    StandardInstruction, SwitchTarget,
};
use crate::packed_instruction::{PackedInstruction, PackedOperation};
use crate::parameter::parameter_expression::ParameterExpression;
use crate::register_data::{RegisterAlreadyExists, RegisterData};
use crate::var_stretch_container::{
    StretchType, VarStretchContainer, VarStretchContainerError, VarType,
};
use crate::variable_mapper::VariableMapper;
use crate::{
    Block, BlockMapper, BlocksMode, Clbit, ControlFlowBlocks, Qubit, Stretch, TupleLikeArg, Var,
    VarsMode, imports, instruction, vf2,
};
use qiskit_util::py::{PySequenceIndex, SequenceIndex};

use hashbrown::{HashMap, HashSet};
use itertools::{EitherOrBoth, Itertools};
use qiskit_util::{IndexMap, IndexSet};

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{
    PyDeprecationWarning, PyIndexError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::intern;
use pyo3::prelude::*;

use pyo3::types::{IntoPyDict, PyDict, PyInt, PyIterator, PyList, PySet, PyTuple, PyType};

use rustworkx_core::dag_algo::layers;
use rustworkx_core::err::ContractError;
use rustworkx_core::graph_ext::ContractNodesDirected;
use rustworkx_core::petgraph;
use rustworkx_core::petgraph::Incoming;
use rustworkx_core::petgraph::prelude::StableDiGraph;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::stable_graph::{EdgeReference, IndexType};
use rustworkx_core::petgraph::unionfind::UnionFind;
use rustworkx_core::petgraph::visit::{
    EdgeIndexable, IntoEdgeReferences, IntoNodeReferences, NodeFiltered, NodeIndexable,
};
use rustworkx_core::traversal::{
    ancestors as core_ancestors, bfs_predecessors as core_bfs_predecessors,
    bfs_successors as core_bfs_successors, descendants as core_descendants,
};

#[inline]
fn dag_compose_width_error(bit_kind: &str, other: usize, dest: usize) -> String {
    format!("Cannot compose onto a DAGCircuit with fewer {bit_kind} ({other} > {dest}).")
}

use crate::imports::PARAMETER;
use crate::instruction::Parameters;
use crate::parameter_table::ParameterUuid;
use approx::relative_eq;
use std::collections::{BTreeMap, VecDeque};
use std::convert::Infallible;
use std::f64::consts::PI;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;
use uuid::Uuid;

static CONTROL_FLOW_OP_NAMES: [&str; 5] =
    ["for_loop", "while_loop", "if_else", "switch_case", "box"];
static SEMANTIC_EQ_SYMMETRIC: [&str; 4] = ["barrier", "swap", "break_loop", "continue_loop"];

pub use rustworkx_core::petgraph::stable_graph::NodeIndex;

/// An error indicating that an input would result in an invalid dag with
/// duplicate wires or that the input that takes wires contains duplicates
#[derive(Debug, thiserror::Error)]
#[error("Wire {0:?} already exists in circuit")]
pub struct DuplicateWireError(Wire);

impl From<DuplicateWireError> for PyErr {
    fn from(value: DuplicateWireError) -> Self {
        DAGCircuitError::new_err(value.to_string())
    }
}

/// Possible errors that can be received from any operation on the
/// [`DAGCircuit`]
#[derive(Debug, thiserror::Error)]
pub enum DAGError {
    #[error("qubits not idle: {0:?}")]
    QubitsNotIdle(Vec<ShareableQubit>),
    #[error("clbits not idle: {0:?}")]
    ClbitsNotIdle(Vec<ShareableClbit>),
    #[error("Wire {0:?} not found in circuit")]
    WireNotInCircuit(Wire),
    #[error(transparent)]
    DuplicateWire(#[from] DuplicateWireError),
    #[error(
        "{ty} not in circuit. {0}",
        ty = match .0 {
            RegisterType::Quantum(_) => "qreg",
            RegisterType::Classical(_) => "creg",
        }
    )]
    RegisterNotInCircuit(RegisterType),
    #[error("{message}", message = {
        match .0 {
            Wire::Qubit(qubit) => format!("Qubit index {} is out of range. This DAGCircuit currently has only {} qubits.", qubit.index(), .1),
            Wire::Clbit(clbit) => format!("Clbit index {} is out of range. This DAGCircuit currently has only {} clbits.", clbit.index(), .1),
            Wire::Var(var) => format!("Var index {} is out of range. This DAGCircuit currently has only {} vars.", var.index(), .1),
        }
    })]
    WireOutOfRange(Wire, usize),
    #[error(
        "Invalid parameter type for global phase, only float and parameter expression are supported"
    )]
    ObjGlobalPhase,
    #[error(transparent)]
    RegistryAddError(#[from] crate::object_registry::AddError),
    #[error(transparent)]
    RegistryAbsentObject(crate::object_registry::AbsentObject),
    #[error(
        "{ty} '{name}' to be inlined is not in the base DAG. If you wanted it to be automatically added, use `inline_captures=False`.",
        ty = (if *is_var {"Variable"} else {"Stretch"})
    )]
    VariableNotInlined { name: String, is_var: bool },
    #[error("Wire '{0:?}' not in self")]
    MissingWire(ShareableWire),
    #[error("Specified node {} is not in this graph", .0.index())]
    NodeNotInGraph(NodeIndex),
    /// Errors that don't have anything that needs to be tracked. Only a message.
    #[error("{0}")]
    General(String),
    #[error(transparent)]
    VarStretchContainer(#[from] VarStretchContainerError),
    #[error("Wire {0:?} not found in output map")]
    WireNotInOutput(ShareableWire),
    #[error("No register with '{0:?}' to map this expression onto.")]
    RejectNewRegister(Vec<ShareableClbit>),
    #[error(transparent)]
    RegisterData(#[from] RegisterAlreadyExists),
    #[error(transparent)]
    Circuit(#[from] crate::circuit_data::CircuitDataError),
    // For special Python cases.
    #[error(transparent)]
    Python(PyErr),
}

impl From<Infallible> for DAGError {
    fn from(value: Infallible) -> Self {
        match value {}
    }
}

impl From<String> for DAGError {
    fn from(value: String) -> Self {
        DAGError::General(value)
    }
}

impl<T: Debug> From<crate::object_registry::AbsentObject<T>> for DAGError {
    fn from(val: crate::object_registry::AbsentObject<T>) -> Self {
        Self::RegistryAbsentObject(val.erase_type())
    }
}

impl From<DAGError> for PyErr {
    fn from(value: DAGError) -> Self {
        match value {
            DAGError::RegistryAddError(err) => err.into(),
            DAGError::RegistryAbsentObject(err) => err.into(),
            DAGError::Python(err) => err,
            DAGError::WireOutOfRange(_, _total) => PyValueError::new_err(value.to_string()),
            DAGError::NodeNotInGraph(_) => PyIndexError::new_err(value.to_string()),
            DAGError::ObjGlobalPhase => PyTypeError::new_err(value.to_string()),
            DAGError::Circuit(err) => err.into(),
            _ => DAGCircuitError::new_err(value.to_string()),
        }
    }
}

/// Enumeration to capture concrete wires
#[derive(Debug)]
pub enum ShareableWire {
    Qubit(ShareableQubit),
    Clbit(ShareableClbit),
    Var(Var),
}

/// Enumeration to capture all possible registers.
#[derive(Debug)]
pub enum RegisterType {
    Quantum(QuantumRegister),
    Classical(ClassicalRegister),
}

impl std::fmt::Display for RegisterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegisterType::Quantum(quantum_register) => quantum_register.fmt(f),
            RegisterType::Classical(classical_register) => classical_register.fmt(f),
        }
    }
}

#[derive(Clone, Debug)]
pub enum NodeType {
    QubitIn(Qubit),
    QubitOut(Qubit),
    ClbitIn(Clbit),
    ClbitOut(Clbit),
    VarIn(Var),
    VarOut(Var),
    Operation(PackedInstruction),
}

impl NodeType {
    /// Unwraps this node as an operation and returns a reference to
    /// the contained [PackedInstruction].
    ///
    /// Panics if this is not an operation node.
    pub fn unwrap_operation(&self) -> &PackedInstruction {
        match self {
            NodeType::Operation(instr) => instr,
            _ => panic!("Node is not an operation!"),
        }
    }

    /// Are these two nodes equal, using the given closure to compare [PackedInstruction] variants.
    pub fn equal_with<F, E>(&self, other: &Self, cmp: F) -> Result<bool, E>
    where
        F: FnOnce(&PackedInstruction, &PackedInstruction) -> Result<bool, E>,
    {
        match (self, other) {
            (Self::QubitIn(left), Self::QubitIn(right)) => Ok(left == right),
            (Self::QubitOut(left), Self::QubitOut(right)) => Ok(left == right),
            (Self::ClbitIn(left), Self::ClbitIn(right)) => Ok(left == right),
            (Self::ClbitOut(left), Self::ClbitOut(right)) => Ok(left == right),
            (Self::VarIn(left), Self::VarIn(right)) => Ok(left == right),
            (Self::VarOut(left), Self::VarOut(right)) => Ok(left == right),
            (Self::Operation(left), Self::Operation(right)) => cmp(left, right),
            _ => Ok(false),
        }
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum Wire {
    Qubit(Qubit),
    Clbit(Clbit),
    Var(Var),
}
impl From<Qubit> for Wire {
    fn from(wire: Qubit) -> Self {
        Self::Qubit(wire)
    }
}
impl From<Clbit> for Wire {
    fn from(wire: Clbit) -> Self {
        Self::Clbit(wire)
    }
}
impl From<Var> for Wire {
    fn from(wire: Var) -> Self {
        Self::Var(wire)
    }
}

impl Wire {
    fn to_pickle(self, py: Python) -> PyResult<Py<PyAny>> {
        match self {
            Self::Qubit(bit) => (0, bit.0.into_py_any(py)?),
            Self::Clbit(bit) => (1, bit.0.into_py_any(py)?),
            Self::Var(var) => (2, var.0.into_py_any(py)?),
        }
        .into_py_any(py)
    }

    fn from_pickle(b: &Bound<PyAny>) -> PyResult<Self> {
        let tuple: Bound<PyTuple> = b.extract()?;
        let wire_type: usize = tuple.get_item(0)?.extract()?;
        if wire_type == 0 {
            Ok(Self::Qubit(Qubit(tuple.get_item(1)?.extract()?)))
        } else if wire_type == 1 {
            Ok(Self::Clbit(Clbit(tuple.get_item(1)?.extract()?)))
        } else if wire_type == 2 {
            Ok(Self::Var(Var(tuple.get_item(1)?.extract()?)))
        } else {
            Err(PyTypeError::new_err("Invalid wire type"))
        }
    }
}

/// Quantum circuit as a directed acyclic graph.
///
/// There are 3 types of nodes in the graph: inputs, outputs, and operations.
/// The nodes are connected by directed edges that correspond to qubits and
/// bits.
#[pyclass(module = "qiskit._accelerate.circuit", from_py_object)]
#[derive(Clone, Debug)]
pub struct DAGCircuit {
    /// Circuit name.  Generally, this corresponds to the name
    /// of the QuantumCircuit from which the DAG was generated.
    #[pyo3(get, set)]
    pub name: Option<String>,
    /// Circuit metadata
    #[pyo3(get, set)]
    pub metadata: Option<Py<PyAny>>,

    dag: StableDiGraph<NodeType, Wire>,

    qregs: RegisterData<QuantumRegister>,
    cregs: RegisterData<ClassicalRegister>,

    /// The cache used to intern instruction qargs.
    qargs_interner: Interner<[Qubit]>,
    /// The cache used to intern instruction cargs.
    cargs_interner: Interner<[Clbit]>,
    /// Qubits registered in the circuit.
    qubits: ObjectRegistry<Qubit, ShareableQubit>,
    /// Clbits registered in the circuit.
    clbits: ObjectRegistry<Clbit, ShareableClbit>,
    /// Basic blocks registered in the circuit.
    blocks: ControlFlowBlocks<DAGCircuit>,
    /// Variables and stretches registered in the circuit
    vars_stretches: VarStretchContainer,
    /// Global phase.
    global_phase: Param,
    /// Duration.
    duration: Option<Py<PyAny>>,
    /// Unit of duration.
    unit: String,

    // Note: these are tracked separately from `qubits` and `clbits`
    // because it's not yet clear if the Rust concept of a native Qubit
    // and Clbit should correspond directly to the numerical Python
    // index that users see in the Python API.
    /// The index locations of bits, and their positions within
    /// registers.
    qubit_locations: BitLocator<ShareableQubit, QuantumRegister>,
    clbit_locations: BitLocator<ShareableClbit, ClassicalRegister>,

    /// Map from qubit to input and output nodes of the graph.
    qubit_io_map: Vec<[NodeIndex; 2]>,

    /// Map from clbit to input and output nodes of the graph.
    clbit_io_map: Vec<[NodeIndex; 2]>,

    /// Map from var to input and output nodes of the graph.
    var_io_map: Vec<[NodeIndex; 2]>,

    /// Operation kind to count
    op_names: IndexMap<String, usize>,
}

#[derive(Clone, Debug)]
struct PyLegacyResources {
    clbits: Py<PyTuple>,
    cregs: Py<PyTuple>,
}

fn condition_resources(condition: &Bound<PyAny>) -> PyResult<PyLegacyResources> {
    let res = imports::CONTROL_FLOW_CONDITION_RESOURCES
        .get_bound(condition.py())
        .call1((condition,))?;
    Ok(PyLegacyResources {
        clbits: res.getattr("clbits")?.cast_into_exact()?.unbind(),
        cregs: res.getattr("cregs")?.cast_into_exact()?.unbind(),
    })
}

fn reject_new_register(reg: &ClassicalRegister) -> Result<(), DAGError> {
    Err(DAGError::RejectNewRegister(reg.bits().collect_vec()))
}

#[pyclass(
    name = "BitLocations",
    module = "qiskit._accelerate.circuit",
    sequence,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyBitLocations {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub registers: Py<PyList>,
}

#[pymethods]
impl PyBitLocations {
    #[new]
    /// Creates a new instance of [PyBitLocations]
    pub fn new(index: usize, registers: Py<PyList>) -> Self {
        Self { index, registers }
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> PyResult<bool> {
        let borrowed = slf.borrow();
        if let Ok(other) = other.cast::<Self>() {
            let other_borrowed = other.borrow();
            Ok(borrowed.index == other_borrowed.index
                && slf.getattr("registers")?.eq(other.getattr("registers")?)?)
        } else if let Ok(other) = other.cast::<PyTuple>() {
            Ok(slf.getattr("index")?.eq(other.get_item(0)?)?
                && slf.getattr("registers")?.eq(other.get_item(1)?)?)
        } else {
            Ok(false)
        }
    }

    fn __iter__(slf: Bound<Self>) -> PyResult<Bound<PyIterator>> {
        (slf.getattr("index")?, slf.getattr("registers")?)
            .into_bound_py_any(slf.py())?
            .try_iter()
    }

    fn __repr__(slf: Bound<Self>) -> PyResult<String> {
        Ok(format!(
            "{}(index={} registers={})",
            slf.get_type().name()?,
            slf.getattr("index")?.repr()?,
            slf.getattr("registers")?.repr()?
        ))
    }

    fn __getnewargs__(slf: Bound<Self>) -> PyResult<(Bound<PyAny>, Bound<PyAny>)> {
        Ok((slf.getattr("index")?, slf.getattr("registers")?))
    }

    fn __getitem__(&self, py: Python, index: PySequenceIndex<'_>) -> PyResult<Py<PyAny>> {
        let getter = |index: usize| -> PyResult<Py<PyAny>> {
            match index {
                0 => self.index.into_py_any(py),
                1 => Ok(self.registers.clone_ref(py).into_any()),
                _ => Err(PyIndexError::new_err("index out of range")),
            }
        };
        if let Ok(index) = index.with_len(2) {
            match index {
                SequenceIndex::Int(index) => getter(index),
                _ => PyTuple::new(py, index.iter().map(|idx| getter(idx).unwrap()))
                    .map(|obj| obj.into_any().unbind()),
            }
        } else {
            Err(PyIndexError::new_err("index out of range"))
        }
    }

    #[staticmethod]
    fn __len__() -> usize {
        2
    }
}

#[pymethods]
impl DAGCircuit {
    #[new]
    pub fn py_new(py: Python) -> PyResult<Self> {
        let mut out = Self::new();
        out.metadata = Some(PyDict::new(py).unbind().into());
        Ok(out)
    }

    /// Returns the dict containing the QuantumRegisters in the circuit
    #[getter]
    fn get_qregs(&self, py: Python) -> &Py<PyDict> {
        self.qregs.cached(py)
    }

    /// Returns a dict mapping Qubit instances to tuple comprised of 0) the
    /// corresponding index in circuit.qubits and 1) a list of
    /// Register-int pairs for each Register containing the Bit and its index
    /// within that register.
    #[getter("_qubit_indices")]
    pub fn get_qubit_locations(&self, py: Python) -> &Py<PyDict> {
        self.qubit_locations.cached(py)
    }

    /// Returns the dict containing the ClassicalRegisters in the circuit
    #[getter]
    fn get_cregs(&self, py: Python) -> &Py<PyDict> {
        self.cregs.cached(py)
    }

    /// Returns a dict mapping Clbit instances to tuple comprised of 0) the
    /// corresponding index in circuit.clbits and 1) a list of
    /// Register-int pairs for each Register containing the Bit and its index
    /// within that register.
    #[getter("_clbit_indices")]
    pub fn get_clbit_locations(&self, py: Python) -> &Py<PyDict> {
        self.clbit_locations.cached(py)
    }

    /// Returns the total duration of the circuit, set by a scheduling transpiler pass. Its unit is
    /// specified by :attr:`.unit`
    ///
    /// DEPRECATED since Qiskit 1.3.0 and will be removed in Qiskit 3.0.0
    #[getter("duration")]
    fn get_duration(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        imports::WARNINGS_WARN.get_bound(py).call1((
            intern!(
                py,
                concat!(
                    "The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.duration`` is ",
                    "deprecated as of Qiskit 1.3.0. It will be removed in Qiskit 3.0.0.",
                )
            ),
            py.get_type::<PyDeprecationWarning>(),
            1,
        ))?;
        self.get_internal_duration(py)
    }

    /// Returns the total duration of the circuit for internal use (no deprecation warning).
    ///
    /// To be removed with get_duration.
    #[getter("_duration")]
    fn get_internal_duration(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        Ok(self.duration.as_ref().map(|x| x.clone_ref(py)))
    }

    /// Sets the total duration of the circuit, set by a scheduling transpiler pass. Its unit is
    /// specified by :attr:`.unit`
    ///
    /// DEPRECATED since Qiskit 1.3.0 and will be removed in Qiskit 3.0.0
    #[setter("duration")]
    fn set_duration(&mut self, py: Python, duration: Option<Py<PyAny>>) -> PyResult<()> {
        imports::WARNINGS_WARN.get_bound(py).call1((
            intern!(
                py,
                concat!(
                    "The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.duration`` is ",
                    "deprecated as of Qiskit 1.3.0. It will be removed in Qiskit 3.0.0.",
                )
            ),
            py.get_type::<PyDeprecationWarning>(),
            1,
        ))?;
        self.set_internal_duration(duration);
        Ok(())
    }

    /// Sets the total duration of the circuit for internal use (no deprecation warning).
    ///
    /// To be removed with set_duration.
    #[setter("_duration")]
    fn set_internal_duration(&mut self, duration: Option<Py<PyAny>>) {
        self.duration = duration
    }

    /// Returns the unit that duration is specified in.
    ///
    /// DEPRECATED since Qiskit 1.3.0 and will be removed in Qiskit 3.0.0
    #[getter]
    fn get_unit(&self, py: Python) -> PyResult<String> {
        imports::WARNINGS_WARN.get_bound(py).call1((
            intern!(
                py,
                concat!(
                    "The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.unit`` is ",
                    "deprecated as of Qiskit 1.3.0. It will be removed in Qiskit 3.0.0.",
                )
            ),
            py.get_type::<PyDeprecationWarning>(),
            1,
        ))?;
        self.get_internal_unit()
    }

    /// Returns the unit that duration is specified in for internal use (no deprecation warning).
    ///
    /// To be removed with get_unit.
    #[getter("_unit")]
    fn get_internal_unit(&self) -> PyResult<String> {
        Ok(self.unit.clone())
    }

    /// Sets the unit that duration is specified in.
    ///
    /// DEPRECATED since Qiskit 1.3.0 and will be removed in Qiskit 3.0.0
    #[setter("unit")]
    fn set_unit(&mut self, py: Python, unit: String) -> PyResult<()> {
        imports::WARNINGS_WARN.get_bound(py).call1((
            intern!(
                py,
                concat!(
                    "The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.unit`` is ",
                    "deprecated as of Qiskit 1.3.0. It will be removed in Qiskit 3.0.0.",
                )
            ),
            py.get_type::<PyDeprecationWarning>(),
            1,
        ))?;
        self.set_internal_unit(unit);
        Ok(())
    }

    /// Sets the unit that duration is specified in for internal use (no deprecation warning).
    ///
    /// To be removed with set_unit.
    #[setter("_unit")]
    fn set_internal_unit(&mut self, unit: String) {
        self.unit = unit
    }

    #[getter]
    fn input_map(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new(py);
        for (qubit, indices) in self
            .qubit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Qubit::new(idx), indices))
        {
            out_dict.set_item(
                self.qubits.get(qubit).unwrap(),
                self.get_node(py, indices[0])?,
            )?;
        }
        for (clbit, indices) in self
            .clbit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Clbit::new(idx), indices))
        {
            out_dict.set_item(
                self.clbits.get(clbit).unwrap(),
                self.get_node(py, indices[0])?,
            )?;
        }
        for (var, indices) in self
            .var_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Var::new(idx), indices))
        {
            out_dict.set_item(
                self.vars_stretches
                    .vars()
                    .get(var)
                    .unwrap()
                    .clone()
                    .into_pyobject(py)?,
                self.get_node(py, indices[0])?,
            )?;
        }
        Ok(out_dict.unbind())
    }

    #[getter]
    fn output_map(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new(py);
        for (qubit, indices) in self
            .qubit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Qubit::new(idx), indices))
        {
            out_dict.set_item(
                self.qubits.get(qubit).unwrap(),
                self.get_node(py, indices[1])?,
            )?;
        }
        for (clbit, indices) in self
            .clbit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Clbit::new(idx), indices))
        {
            out_dict.set_item(
                self.clbits.get(clbit).unwrap(),
                self.get_node(py, indices[1])?,
            )?;
        }
        for (var, indices) in self
            .var_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Var::new(idx), indices))
        {
            out_dict.set_item(
                self.vars_stretches
                    .vars()
                    .get(var)
                    .unwrap()
                    .clone()
                    .into_pyobject(py)?,
                self.get_node(py, indices[1])?,
            )?;
        }
        Ok(out_dict.unbind())
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new(py);
        out_dict.set_item("name", self.name.clone())?;
        out_dict.set_item("metadata", self.metadata.as_ref().map(|x| x.clone_ref(py)))?;
        out_dict.set_item("qregs", self.qregs.cached(py))?;
        out_dict.set_item("cregs", self.cregs.cached(py))?;
        out_dict.set_item("global_phase", self.global_phase.clone())?;
        out_dict.set_item(
            "qubit_io_map",
            self.qubit_io_map
                .iter()
                .enumerate()
                .map(|(k, v)| (k, [v[0].index(), v[1].index()]))
                .into_py_dict(py)?,
        )?;
        out_dict.set_item(
            "clbit_io_map",
            self.clbit_io_map
                .iter()
                .enumerate()
                .map(|(k, v)| (k, [v[0].index(), v[1].index()]))
                .into_py_dict(py)?,
        )?;
        out_dict.set_item(
            "var_io_map",
            self.var_io_map
                .iter()
                .enumerate()
                .map(|(k, v)| (k, [v[0].index(), v[1].index()]))
                .into_py_dict(py)?,
        )?;
        out_dict.set_item("op_name", self.op_names.clone())?;
        out_dict.set_item("qubits", self.qubits.objects())?;
        out_dict.set_item("clbits", self.clbits.objects())?;

        let vars_stretches_state = self.vars_stretches.to_pickle(py);
        out_dict.set_item("identifier_info", vars_stretches_state.0)?;
        out_dict.set_item("vars", vars_stretches_state.1)?;
        out_dict.set_item("stretches", vars_stretches_state.2)?;
        let mut nodes: Vec<Py<PyAny>> = Vec::with_capacity(self.dag.node_count());
        for node_idx in self.dag.node_indices() {
            let node_data = self.get_node(py, node_idx)?;
            nodes.push((node_idx.index(), node_data).into_py_any(py)?);
        }
        out_dict.set_item("nodes", nodes)?;
        out_dict.set_item(
            "nodes_removed",
            self.dag.node_count() != self.dag.node_bound(),
        )?;
        let mut edges: Vec<Py<PyAny>> = Vec::with_capacity(self.dag.edge_bound());
        // edges are saved with none (deleted edges) instead of their index to save space
        for i in 0..self.dag.edge_bound() {
            let idx = EdgeIndex::new(i);
            let edge = match self.dag.edge_weight(idx) {
                Some(edge_w) => {
                    let endpoints = self.dag.edge_endpoints(idx).unwrap();
                    (
                        endpoints.0.index(),
                        endpoints.1.index(),
                        edge_w.to_pickle(py)?,
                    )
                        .into_py_any(py)?
                }
                None => py.None(),
            };
            edges.push(edge);
        }
        out_dict.set_item("edges", edges)?;
        Ok(out_dict.unbind())
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let dict_state = state.cast_bound::<PyDict>(py)?;
        self.name = dict_state.get_item("name")?.unwrap().extract()?;
        self.metadata = dict_state.get_item("metadata")?.unwrap().extract()?;
        self.qregs = RegisterData::from_mapping(
            dict_state
                .get_item("qregs")?
                .unwrap()
                .extract::<IndexMap<String, QuantumRegister>>()?,
        );
        self.cregs = RegisterData::from_mapping(
            dict_state
                .get_item("cregs")?
                .unwrap()
                .extract::<IndexMap<String, ClassicalRegister>>()?,
        );
        self.global_phase = dict_state.get_item("global_phase")?.unwrap().extract()?;
        self.op_names = dict_state.get_item("op_name")?.unwrap().extract()?;
        let binding = dict_state.get_item("qubits")?.unwrap();
        let qubits_raw = binding.extract::<Vec<ShareableQubit>>()?;
        for bit in qubits_raw.into_iter() {
            self.qubits.add(bit)?;
        }
        let binding = dict_state.get_item("clbits")?.unwrap();
        let clbits_raw = binding.extract::<Vec<ShareableClbit>>()?;
        for bit in clbits_raw.into_iter() {
            self.clbits.add(bit)?;
        }

        let binding = dict_state.get_item("identifier_info")?.unwrap();
        let identifier_info = binding.extract::<Vec<(String, Py<PyAny>)>>()?;
        let binding = dict_state.get_item("vars")?.unwrap();
        let vars_raw = binding.extract::<Vec<expr::Var>>()?;
        let binding = dict_state.get_item("stretches")?.unwrap();
        let stretches_raw = binding.extract::<Vec<expr::Stretch>>()?;

        self.vars_stretches =
            VarStretchContainer::from_pickle(py, (identifier_info, vars_raw, stretches_raw))?;

        let binding = dict_state.get_item("qubit_io_map")?.unwrap();
        let qubit_index_map_raw = binding.cast::<PyDict>().unwrap();
        self.qubit_io_map = Vec::with_capacity(qubit_index_map_raw.len());
        for (_k, v) in qubit_index_map_raw.iter() {
            let indices: [usize; 2] = v.extract()?;
            self.qubit_io_map
                .push([NodeIndex::new(indices[0]), NodeIndex::new(indices[1])]);
        }
        let binding = dict_state.get_item("clbit_io_map")?.unwrap();
        let clbit_index_map_raw = binding.cast::<PyDict>().unwrap();
        self.clbit_io_map = Vec::with_capacity(clbit_index_map_raw.len());
        for (_k, v) in clbit_index_map_raw.iter() {
            let indices: [usize; 2] = v.extract()?;
            self.clbit_io_map
                .push([NodeIndex::new(indices[0]), NodeIndex::new(indices[1])]);
        }
        let binding = dict_state.get_item("var_io_map")?.unwrap();
        let var_index_map_raw = binding.cast::<PyDict>().unwrap();
        self.var_io_map = Vec::with_capacity(var_index_map_raw.len());
        for (_k, v) in var_index_map_raw.iter() {
            let indices: [usize; 2] = v.extract()?;
            self.var_io_map
                .push([NodeIndex::new(indices[0]), NodeIndex::new(indices[1])]);
        }
        // Rebuild Graph preserving index holes:
        let binding = dict_state.get_item("nodes")?.unwrap();
        let nodes_lst = binding.cast::<PyList>()?;
        let binding = dict_state.get_item("edges")?.unwrap();
        let edges_lst = binding.cast::<PyList>()?;
        let node_removed: bool = dict_state.get_item("nodes_removed")?.unwrap().extract()?;
        self.dag = StableDiGraph::default();
        if !node_removed {
            for item in nodes_lst.iter() {
                let node_w = item.cast::<PyTuple>().unwrap().get_item(1).unwrap();
                let weight = self.pack_into(py, &node_w)?;
                self.dag.add_node(weight);
            }
        } else if nodes_lst.len() == 1 {
            // graph has only one node, handle logic here to save one if in the loop later
            let binding = nodes_lst.get_item(0).unwrap();
            let item = binding.cast::<PyTuple>().unwrap();
            let node_idx: usize = item.get_item(0).unwrap().extract().unwrap();
            let node_w = item.get_item(1).unwrap();

            for _i in 0..node_idx {
                self.dag.add_node(NodeType::QubitIn(Qubit(u32::MAX)));
            }
            let weight = self.pack_into(py, &node_w)?;
            self.dag.add_node(weight);
            for i in 0..node_idx {
                self.dag.remove_node(NodeIndex::new(i));
            }
        } else {
            let binding = nodes_lst.get_item(nodes_lst.len() - 1).unwrap();
            let last_item = binding.cast::<PyTuple>().unwrap();

            // list of temporary nodes that will be removed later to re-create holes
            let node_bound_1: usize = last_item.get_item(0).unwrap().extract().unwrap();
            let mut tmp_nodes: Vec<NodeIndex> =
                Vec::with_capacity(node_bound_1 + 1 - nodes_lst.len());

            for item in nodes_lst {
                let item = item.cast::<PyTuple>().unwrap();
                let next_index: usize = item.get_item(0).unwrap().extract().unwrap();
                let weight: Py<PyAny> = item.get_item(1).unwrap().extract().unwrap();
                while next_index > self.dag.node_bound() {
                    // node does not exist
                    let tmp_node = self.dag.add_node(NodeType::QubitIn(Qubit(u32::MAX)));
                    tmp_nodes.push(tmp_node);
                }
                // add node to the graph, and update the next available node index
                let weight = self.pack_into(py, weight.bind(py))?;
                self.dag.add_node(weight);
            }
            // Remove any temporary nodes we added
            for tmp_node in tmp_nodes {
                self.dag.remove_node(tmp_node);
            }
        }

        // to ensure O(1) on edge deletion, use a temporary node to store missing edges
        let tmp_node = self.dag.add_node(NodeType::QubitIn(Qubit(u32::MAX)));

        for item in edges_lst {
            if item.is_none() {
                // add a temporary edge that will be deleted later to re-create the hole
                self.dag
                    .add_edge(tmp_node, tmp_node, Wire::Qubit(Qubit(u32::MAX)));
            } else {
                let triple = item.cast::<PyTuple>().unwrap();
                let edge_p: usize = triple.get_item(0).unwrap().extract().unwrap();
                let edge_c: usize = triple.get_item(1).unwrap().extract().unwrap();
                let edge_w = Wire::from_pickle(&triple.get_item(2).unwrap())?;
                self.dag
                    .add_edge(NodeIndex::new(edge_p), NodeIndex::new(edge_c), edge_w);
            }
        }
        self.dag.remove_node(tmp_node);
        self.qubit_locations = BitLocator::with_capacity(self.qubits.len());
        for (index, qubit) in self.qubits.objects().iter().enumerate() {
            let registers = self
                .qregs
                .registers()
                .iter()
                .filter_map(|x| x.index_of(qubit).map(|y| (x.clone(), y)));
            self.qubit_locations
                .insert(qubit.clone(), BitLocations::new(index as u32, registers));
        }
        self.clbit_locations = BitLocator::with_capacity(self.clbits.len());
        for (index, clbit) in self.clbits.objects().iter().enumerate() {
            let registers = self
                .cregs
                .registers()
                .iter()
                .filter_map(|x| x.index_of(clbit).map(|y| (x.clone(), y)));
            self.clbit_locations
                .insert(clbit.clone(), BitLocations::new(index as u32, registers));
        }
        Ok(())
    }

    pub fn __copy__(&self) -> Self {
        self.clone()
    }

    pub fn __deepcopy__<'py>(
        &self,
        py: Python<'py>,
        memo: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Self> {
        let mut out = self.clone();
        let deepcopy = imports::DEEPCOPY.get_bound(py);
        // We only need to pass the deep-copying nature on to places where we store Python objects.
        out.metadata = out
            .metadata
            .map(|dict| deepcopy.call1((dict, memo)).map(|ob| ob.unbind()))
            .transpose()?;
        out.duration = out
            .duration
            .map(|dict| deepcopy.call1((dict, memo)).map(|ob| ob.unbind()))
            .transpose()?;
        for node in out.dag.node_weights_mut() {
            if let NodeType::Operation(inst) = node {
                inst.py_deepcopy_inplace(py, memo)?;
            };
        }
        Ok(out)
    }

    /// Returns the current sequence of registered :class:`.Qubit` instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`DAGCircuit` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Qubit`): The current sequence of registered qubits.
    #[getter(qubits)]
    pub fn py_qubits(&self, py: Python<'_>) -> Py<PyList> {
        self.qubits.cached(py).clone_ref(py)
    }

    /// Returns the current sequence of registered :class:`.Clbit`
    /// instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`DAGCircuit` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Clbit`): The current sequence of registered clbits.
    #[getter(clbits)]
    pub fn py_clbits(&self, py: Python<'_>) -> Py<PyList> {
        self.clbits.cached(py).clone_ref(py)
    }

    /// Return a list of the wires in order.
    #[getter]
    fn get_wires(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let wires: Bound<PyList> = PyList::new(py, self.qubits.objects().iter())?;

        for clbit in self.clbits.objects().iter() {
            wires.append(clbit)?
        }

        let out_list = PyList::new(py, wires)?;
        for var in self.vars_stretches.vars().objects() {
            out_list.append(var.clone().into_py_any(py)?)?;
        }
        Ok(out_list.unbind())
    }

    /// Returns the number of nodes in the dag.
    #[getter]
    fn get_node_counter(&self) -> usize {
        self.dag.node_count()
    }

    /// Return the global phase of the circuit.
    #[getter]
    pub fn get_global_phase(&self) -> Param {
        self.global_phase.clone()
    }

    /// Set the global phase of the circuit.
    ///
    /// Args:
    ///     angle (float, :class:`.ParameterExpression`): The phase angle.
    #[setter(global_phase)]
    pub fn set_global_phase(&mut self, angle: Param) -> Result<(), DAGError> {
        self.set_global_phase_param(angle).map(|_| ())
    }

    /// Remove all operation nodes with the given name.
    fn remove_all_ops_named(&mut self, opname: &str) {
        let mut to_remove = Vec::new();
        for (id, weight) in self.dag.node_references() {
            if let NodeType::Operation(packed) = &weight {
                if opname == packed.op.name() {
                    to_remove.push(id);
                }
            }
        }
        for node in to_remove {
            self.remove_op_node(node);
        }
    }

    /// Add individual qubit wires.
    fn add_qubits(&mut self, qubits: Vec<Bound<PyAny>>) -> PyResult<()> {
        for bit in qubits.into_iter() {
            let Ok(bit) = bit.extract::<ShareableQubit>() else {
                return Err(DAGCircuitError::new_err("not a Qubit instance."));
            };
            if self.qubits.find(&bit).is_some() {
                return Err(DAGCircuitError::new_err(format!(
                    "duplicate qubits {bit:?}"
                )));
            }
            self.add_qubit_unchecked(bit)?;
        }
        Ok(())
    }

    /// Add individual clbit wires.
    fn add_clbits(&mut self, clbits: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        for bit in clbits.into_iter() {
            let Ok(bit) = bit.extract::<ShareableClbit>() else {
                return Err(DAGCircuitError::new_err("not a Clbit instance."));
            };
            if self.clbits.find(&bit).is_some() {
                return Err(DAGCircuitError::new_err(format!(
                    "duplicate clbits {bit:?}"
                )));
            }
            self.add_clbit_unchecked(bit)?;
        }
        Ok(())
    }

    /// Add all wires in a quantum register.
    pub fn add_qreg(&mut self, qreg: QuantumRegister) -> Result<(), DAGError> {
        self.qregs.add_register(qreg.clone(), true)?;

        for (index, bit) in qreg.bits().enumerate() {
            if self.qubits.find(&bit).is_none() {
                self.add_qubit_unchecked(bit.clone())?;
            }
            let locations: &mut BitLocations<QuantumRegister> =
                self.qubit_locations.get_mut(&bit).unwrap();
            locations.add_register(qreg.clone(), index);
        }
        Ok(())
    }

    /// Add all wires in a classical register.
    pub fn add_creg(&mut self, creg: ClassicalRegister) -> Result<(), DAGError> {
        self.cregs.add_register(creg.clone(), true)?;

        for (index, bit) in creg.bits().enumerate() {
            if self.clbits.find(&bit).is_none() {
                self.add_clbit_unchecked(bit.clone())?;
            }
            let locations: &mut BitLocations<ClassicalRegister> =
                self.clbit_locations.get_mut(&bit).unwrap();
            locations.add_register(creg.clone(), index);
        }
        Ok(())
    }

    /// Finds locations in the circuit, by mapping the Qubit and Clbit to positional index
    /// BitLocations is defined as: BitLocations = namedtuple("BitLocations", ("index", "registers"))
    ///
    /// Args:
    ///     bit (Bit): The bit to locate.
    ///
    /// Returns:
    ///     namedtuple(int, List[Tuple(Register, int)]): A 2-tuple. The first element (``index``)
    ///         contains the index at which the ``Bit`` can be found (in either
    ///         :obj:`~DAGCircuit.qubits`, :obj:`~DAGCircuit.clbits`, depending on its
    ///         type). The second element (``registers``) is a list of ``(register, index)``
    ///         pairs with an entry for each :obj:`~Register` in the circuit which contains the
    ///         :obj:`~Bit` (and the index in the :obj:`~Register` at which it can be found).
    ///
    ///   Raises:
    ///     DAGCircuitError: If the supplied :obj:`~Bit` was of an unknown type.
    ///     DAGCircuitError: If the supplied :obj:`~Bit` could not be found on the circuit.
    fn find_bit<'py>(
        &self,
        py: Python<'py>,
        bit: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBitLocations>> {
        if let Ok(qubit) = bit.extract::<ShareableQubit>() {
            self.qubit_locations
                .get(&qubit)
                .map(|location| location.clone().into_pyobject(py))
                .transpose()?
                .ok_or_else(|| {
                    DAGCircuitError::new_err(format!(
                        "Could not locate provided bit: {bit}. Has it been added to the DAGCircuit?"
                    ))
                })
        } else if let Ok(clbit) = bit.extract::<ShareableClbit>() {
            self.clbit_locations
                .get(&clbit)
                .map(|location| location.clone().into_pyobject(py))
                .transpose()?
                .ok_or_else(|| {
                    DAGCircuitError::new_err(format!(
                        "Could not locate provided bit: {bit}. Has it been added to the DAGCircuit?"
                    ))
                })
        } else {
            Err(DAGCircuitError::new_err(format!(
                "Could not locate bit of unknown type: {}",
                bit.get_type()
            )))
        }
    }

    /// Remove classical bits from the circuit. All bits MUST be idle.
    /// Any registers with references to at least one of the specified bits will
    /// also be removed.
    ///
    /// .. warning::
    ///     This method is rather slow, since it must iterate over the entire
    ///     DAG to fix-up bit indices.
    ///
    /// Args:
    ///     clbits (List[Clbit]): The bits to remove.
    ///
    /// Raises:
    ///     DAGCircuitError: a clbit is not a :obj:`.Clbit`, is not in the circuit,
    ///         or is not idle.
    #[pyo3(name = "remove_clbits", signature = (*clbits))]
    fn py_remove_clbits(&mut self, clbits: Vec<ShareableClbit>) -> PyResult<()> {
        let bit_iter = match self.clbits.map_objects(clbits.iter().cloned()) {
            Ok(bit_iter) => bit_iter,
            Err(_) => {
                return Err(DAGCircuitError::new_err(format!(
                    "clbits not in circuit: {clbits:?}"
                )));
            }
        };
        self.remove_clbits(bit_iter).map_err(Into::into)
    }

    /// Remove classical registers from the circuit, leaving underlying bits
    /// in place.
    ///
    /// Raises:
    ///     DAGCircuitError: a creg is not a ClassicalRegister, or is not in
    ///     the circuit.
    #[pyo3(name = "remove_cregs", signature = (*cregs))]
    fn py_remove_cregs(&mut self, cregs: Vec<ClassicalRegister>) -> PyResult<()> {
        self.remove_cregs(cregs).map_err(Into::into)
    }

    /// Remove quantum bits from the circuit. All bits MUST be idle.
    /// Any registers with references to at least one of the specified bits will
    /// also be removed.
    ///
    /// .. warning::
    ///     This method is rather slow, since it must iterate over the entire
    ///     DAG to fix-up bit indices.
    ///
    /// Args:
    ///     qubits (List[~qiskit.circuit.Qubit]): The bits to remove.
    ///
    /// Raises:
    ///     DAGCircuitError: a qubit is not a :obj:`~.circuit.Qubit`, is not in the circuit,
    ///         or is not idle.
    #[pyo3(name = "remove_qubits", signature = (*qubits))]
    pub fn py_remove_qubits(&mut self, qubits: Vec<ShareableQubit>) -> PyResult<()> {
        let bit_iter = match self.qubits.map_objects(qubits.iter().cloned()) {
            Ok(bit_iter) => bit_iter,
            Err(_) => {
                return Err(DAGCircuitError::new_err(format!(
                    "qubits not in circuit: {qubits:?}"
                )));
            }
        };
        self.remove_qubits(bit_iter).map_err(Into::into)
    }

    /// Remove quantum registers from the circuit, leaving underlying bits
    /// in place.
    ///
    /// Raises:
    ///     DAGCircuitError: a qreg is not a QuantumRegister, or is not in
    ///     the circuit.
    #[pyo3(name = "remove_qregs", signature = (*qregs))]
    fn py_remove_qregs(&mut self, qregs: Vec<QuantumRegister>) -> PyResult<()> {
        // let self_bound_cregs = self.cregs.bind(py);
        let mut valid_regs: Vec<QuantumRegister> = Vec::new();
        for qregs in qregs.into_iter() {
            if let Some(reg) = self.qregs.get(qregs.name()) {
                if reg != &qregs {
                    return Err(DAGCircuitError::new_err(format!(
                        "creg not in circuit: {reg:?}"
                    )));
                }
                valid_regs.push(qregs);
            } else {
                return Err(DAGCircuitError::new_err(format!(
                    "creg not in circuit: {qregs:?}"
                )));
            }
        }

        // Use an iterator that will remove the registers from the circuit as it iterates.
        let valid_names = valid_regs.iter().map(|reg| {
            for (index, bit) in reg.bits().enumerate() {
                let bit_position = self.qubit_locations.get_mut(&bit).unwrap();
                bit_position.remove_register(reg, index);
            }
            reg.name().to_string()
        });
        self.qregs.remove_registers(valid_names);
        Ok(())
    }

    /// Verify that the condition is valid.
    ///
    /// Args:
    ///     name (string): used for error reporting
    ///     condition (tuple or None): a condition tuple (ClassicalRegister, int) or (Clbit, bool)
    ///
    /// Raises:
    ///     DAGCircuitError: if conditioning on an invalid register
    fn _check_condition(&self, py: Python, name: &str, condition: &Bound<PyAny>) -> PyResult<()> {
        if condition.is_none() {
            return Ok(());
        }

        let resources = condition_resources(condition)?;
        for reg in resources.cregs.bind(py) {
            if !self
                .cregs
                .contains_key(reg.getattr(intern!(py, "name"))?.to_string().as_str())
            {
                return Err(DAGCircuitError::new_err(format!(
                    "invalid creg in condition for {name}"
                )));
            }
        }

        for bit in resources.clbits.bind(py) {
            let bit: ShareableClbit = bit.extract()?;
            if self.clbits.find(&bit).is_none() {
                return Err(DAGCircuitError::new_err(format!(
                    "invalid clbits in condition for {name}"
                )));
            }
        }

        Ok(())
    }

    /// Return a copy of self with the same structure but empty.
    ///
    /// That structure includes:
    ///     * name and other metadata
    ///     * global phase
    ///     * duration
    ///     * all the qubits and clbits, including the registers.
    ///
    /// Returns:
    ///     DAGCircuit: An empty copy of self.
    // NOTE: this is typically not want you want from Rust, because you should be making decisions
    // about the `BlocksMode`.  Python always wants `Drop` (because you can't reproduce references
    // to existing blocks from Python space), but in Rust you should decide.
    //
    // You likely want `copy_empty_like_with_same_capacity`.
    #[pyo3(name = "copy_empty_like", signature = (*, vars_mode=VarsMode::Alike))]
    pub fn py_copy_empty_like(&self, vars_mode: VarsMode) -> Self {
        self.copy_empty_like_with_capacity(0, 0, vars_mode, BlocksMode::Drop)
    }

    /// Put ``self`` into the canonical physical form, with the given number of qubits.
    ///
    /// This acts in place, and does not need to traverse the DAG.  It is intended for use when the
    /// DAG is known to already represent a physical circuit, and we just need to assert that it is
    /// canonical physical form.
    ///
    /// This erases any information about virtual qubits in the :class:`DAGCircuit`; if using this
    /// yourself, you may need to ensure you have created and stored a suitable :class:`.Layout`.
    /// Effectively, this applies the "trivial" layout mapping virtual qubit 0 to physical qubit 0,
    /// and so on.
    ///
    /// Args:
    ///     num_qubits: if given, the total number of physical qubits in the output; it must be at
    ///         least as large as the number of qubits in the DAG.  If not given, the number of
    ///         qubits is unchanged.
    #[pyo3(name = "make_physical", signature = (num_qubits=None))]
    pub fn py_make_physical(&mut self, num_qubits: Option<u32>) -> PyResult<()> {
        let num_qubits = match num_qubits {
            Some(num_qubits) => {
                if (num_qubits as usize) < self.num_qubits() {
                    return Err(PyValueError::new_err(format!(
                        "cannot have fewer physical qubits ({}) than virtual ({})",
                        num_qubits,
                        self.num_qubits()
                    )));
                }
                num_qubits as usize
            }
            None => self.num_qubits(),
        };
        self.make_physical(num_qubits);
        Ok(())
    }

    #[pyo3(signature=(node, check=false))]
    fn _apply_op_node_back(
        &mut self,
        py: Python,
        node: &Bound<PyAny>,
        check: bool,
    ) -> PyResult<()> {
        if let NodeType::Operation(inst) = self.pack_into(py, node)? {
            if check {
                self.check_op_addition(&inst)?;
            }

            self.push_back(inst)?;
            Ok(())
        } else {
            Err(PyTypeError::new_err("Invalid node type input"))
        }
    }

    /// Apply an operation to the output of the circuit.
    ///
    /// Args:
    ///     op (qiskit.circuit.Operation): the operation associated with the DAG node
    ///     qargs (tuple[~qiskit.circuit.Qubit]): qubits that op will be applied to
    ///     cargs (tuple[Clbit]): cbits that op will be applied to
    ///     check (bool): If ``True`` (default), this function will enforce that the
    ///         :class:`.DAGCircuit` data-structure invariants are maintained (all ``qargs`` are
    ///         :class:`~.circuit.Qubit`\\ s, all are in the DAG, etc).  If ``False``, the caller *must*
    ///         uphold these invariants itself, but the cost of several checks will be skipped.
    ///         This is most useful when building a new DAG from a source of known-good nodes.
    /// Returns:
    ///     DAGOpNode: the node for the op that was added to the dag
    ///
    /// Raises:
    ///     DAGCircuitError: if a leaf node is connected to multiple outputs
    #[pyo3(name = "apply_operation_back", signature = (op, qargs=None, cargs=None, *, check=true))]
    pub fn py_apply_operation_back(
        &mut self,
        py: Python,
        op: Bound<PyAny>,
        qargs: Option<TupleLikeArg>,
        cargs: Option<TupleLikeArg>,
        check: bool,
    ) -> PyResult<Py<PyAny>> {
        let py_op = op.extract::<OperationFromPython<DAGCircuit>>()?;
        let qargs = qargs
            .map(|q| q.value.extract::<Vec<ShareableQubit>>())
            .transpose()?;
        let cargs = cargs
            .map(|c| c.value.extract::<Vec<ShareableClbit>>())
            .transpose()?;
        let node = {
            let qubits_id = self.qargs_interner.insert_owned(
                self.qubits
                    .map_objects(qargs.into_iter().flatten())?
                    .collect(),
            );
            let clbits_id = self.cargs_interner.insert_owned(
                self.clbits
                    .map_objects(cargs.into_iter().flatten())?
                    .collect(),
            );
            let instr = PackedInstruction {
                op: py_op.operation,
                qubits: qubits_id,
                clbits: clbits_id,
                params: self.take_parameter_blocks(py_op.params),
                label: py_op.label,
                #[cfg(feature = "cache_pygates")]
                py_op: op.unbind().into(),
            };

            if check {
                self.check_op_addition(&instr)?;
            }
            self.push_back(instr)?
        };

        self.get_node(py, node)
    }

    /// Apply an operation to the input of the circuit.
    ///
    /// Args:
    ///     op (qiskit.circuit.Operation): the operation associated with the DAG node
    ///     qargs (tuple[~qiskit.circuit.Qubit]): qubits that op will be applied to
    ///     cargs (tuple[Clbit]): cbits that op will be applied to
    ///     check (bool): If ``True`` (default), this function will enforce that the
    ///         :class:`.DAGCircuit` data-structure invariants are maintained (all ``qargs`` are
    ///         :class:`~.circuit.Qubit`\\ s, all are in the DAG, etc).  If ``False``, the caller *must*
    ///         uphold these invariants itself, but the cost of several checks will be skipped.
    ///         This is most useful when building a new DAG from a source of known-good nodes.
    /// Returns:
    ///     DAGOpNode: the node for the op that was added to the dag
    ///
    /// Raises:
    ///     DAGCircuitError: if initial nodes connected to multiple out edges
    #[pyo3(name = "apply_operation_front", signature = (op, qargs=None, cargs=None, *, check=true))]
    fn py_apply_operation_front(
        &mut self,
        py: Python,
        op: Bound<PyAny>,
        qargs: Option<TupleLikeArg>,
        cargs: Option<TupleLikeArg>,
        check: bool,
    ) -> PyResult<Py<PyAny>> {
        let py_op = op.extract::<OperationFromPython<DAGCircuit>>()?;
        let qargs = qargs
            .map(|q| q.value.extract::<Vec<ShareableQubit>>())
            .transpose()?;
        let cargs = cargs
            .map(|c| c.value.extract::<Vec<ShareableClbit>>())
            .transpose()?;
        let node = {
            let qubits_id = self.qargs_interner.insert_owned(
                self.qubits
                    .map_objects(qargs.into_iter().flatten())?
                    .collect(),
            );
            let clbits_id = self.cargs_interner.insert_owned(
                self.clbits
                    .map_objects(cargs.into_iter().flatten())?
                    .collect(),
            );
            let instr = PackedInstruction {
                op: py_op.operation,
                qubits: qubits_id,
                clbits: clbits_id,
                params: self.take_parameter_blocks(py_op.params),
                label: py_op.label,
                #[cfg(feature = "cache_pygates")]
                py_op: op.unbind().into(),
            };

            if check {
                self.check_op_addition(&instr)?;
            }
            self.push_front(instr)?
        };

        self.get_node(py, node)
    }

    /// Compose the ``other`` circuit onto the output of this circuit.
    ///
    /// A subset of input wires of ``other`` are mapped
    /// to a subset of output wires of this circuit.
    ///
    /// ``other`` can be narrower or of equal width to ``self``.
    ///
    /// Args:
    ///     other (DAGCircuit): circuit to compose with self
    ///     qubits (list[~qiskit.circuit.Qubit|int]): qubits of self to compose onto.
    ///     clbits (list[Clbit|int]): clbits of self to compose onto.
    ///     front (bool): If True, front composition will be performed (not implemented yet)
    ///     inplace (bool): If True, modify the object. Otherwise return composed circuit.
    ///     inline_captures (bool): If ``True``, variables marked as "captures" in the ``other`` DAG
    ///         will be inlined onto existing uses of those same variables in ``self``.  If ``False``,
    ///         all variables in ``other`` are required to be distinct from ``self``, and they will
    ///         be added to ``self``.
    ///
    /// ..
    ///     Note: unlike `QuantumCircuit.compose`, there's no `var_remap` argument here.  That's
    ///     because the `DAGCircuit` inner-block structure isn't set up well to allow the recursion,
    ///     and `DAGCircuit.compose` is generally only used to rebuild a DAG from layers within
    ///     itself than to join unrelated circuits.  While there's no strong motivating use-case
    ///     (unlike the `QuantumCircuit` equivalent), it's safer and more performant to not provide
    ///     the option.
    ///
    /// Returns:
    ///    DAGCircuit: the composed dag (returns None if inplace==True).
    ///
    /// Raises:
    ///     DAGCircuitError: if ``other`` is wider or there are duplicate edge mappings.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(name="compose", signature = (other, qubits=None, clbits=None, front=false, inplace=true, *, inline_captures=false))]
    fn py_compose(
        &mut self,
        py: Python,
        other: &DAGCircuit,
        qubits: Option<Bound<PyList>>,
        clbits: Option<Bound<PyList>>,
        front: bool,
        inplace: bool,
        inline_captures: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        if front {
            return Err(DAGCircuitError::new_err(
                "Front composition not supported yet.",
            ));
        }

        let qubits = qubits
            .map(|qubits| {
                qubits
                    .iter()
                    .map(|q| -> PyResult<Qubit> {
                        if q.is_instance_of::<PyInt>() {
                            let index = q.extract()?;
                            Ok(Qubit::new(index))
                        } else {
                            let shareable_qubit = q.extract::<ShareableQubit>()?;
                            self.qubits.find(&shareable_qubit).ok_or_else(|| {
                                DAGCircuitError::new_err(format!(
                                    "Qubit {:?} not found in DAG",
                                    shareable_qubit
                                ))
                            })
                        }
                    })
                    .collect::<PyResult<Vec<Qubit>>>()
            })
            .transpose()?;

        let clbits = clbits
            .map(|clbits| {
                clbits
                    .iter()
                    .map(|c| -> PyResult<Clbit> {
                        if c.is_instance_of::<PyInt>() {
                            let index = c.extract()?;
                            Ok(Clbit::new(index))
                        } else {
                            let shareable_clbit = c.extract::<ShareableClbit>()?;
                            self.clbits.find(&shareable_clbit).ok_or_else(|| {
                                DAGCircuitError::new_err(format!(
                                    "Clbit {:?} not found in DAG",
                                    shareable_clbit
                                ))
                            })
                        }
                    })
                    .collect::<PyResult<Vec<Clbit>>>()
            })
            .transpose()?;

        // TODO: since operations own their blocks in Python land, the best we can do without extra
        //       tracking is to create a new block in 'self' for every block in 'other'. This will
        //       lead to duplication if a Python user composes with a DAG that already uses blocks
        //       registered in 'self'.
        let block_map = other
            .blocks()
            .items()
            .map(|(index, block)| (index, self.add_block(block.clone())))
            .collect();

        // Compose
        if inplace {
            self.compose(
                other,
                qubits.as_deref(),
                clbits.as_deref(),
                block_map,
                inline_captures,
            )?;
            Ok(None)
        } else {
            let mut dag = self.clone();
            dag.compose(
                other,
                qubits.as_deref(),
                clbits.as_deref(),
                block_map,
                inline_captures,
            )?;
            let out_obj = dag.into_py_any(py)?;
            Ok(Some(out_obj))
        }
    }

    /// Reverse the operations in the ``self`` circuit.
    ///
    /// Returns:
    ///     DAGCircuit: the reversed dag.
    fn reverse_ops<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let qc = imports::DAG_TO_CIRCUIT.get_bound(py).call1((slf,))?;
        let reversed = qc.call_method0("reverse_ops")?;
        imports::CIRCUIT_TO_DAG.get_bound(py).call1((reversed,))
    }

    /// Return idle wires.
    ///
    /// Args:
    ///     ignore (list(str)): List of node names to ignore. Default: []
    ///
    /// Yields:
    ///     Bit: Bit in idle wire.
    ///
    /// Raises:
    ///     DAGCircuitError: If the DAG is invalid
    #[pyo3(signature=(ignore=None))]
    fn idle_wires(&self, py: Python, ignore: Option<&Bound<PyList>>) -> PyResult<Py<PyIterator>> {
        let mut result: Vec<Py<PyAny>> = Vec::new();
        let wires = (0..self.qubit_io_map.len())
            .map(|idx| Wire::Qubit(Qubit::new(idx)))
            .chain((0..self.clbit_io_map.len()).map(|idx| Wire::Clbit(Clbit::new(idx))))
            .chain((0..self.var_io_map.len()).map(|idx| Wire::Var(Var::new(idx))));
        match ignore {
            Some(ignore) => {
                // Convert the list to a Rust set.
                let ignore_set = ignore
                    .into_iter()
                    .map(|s| s.extract())
                    .collect::<PyResult<HashSet<String>>>()?;
                for wire in wires {
                    let nodes_found = self
                        .nodes_on_wire(wire)
                        .filter(|node| matches!(self.dag[*node], NodeType::Operation(_)))
                        .any(|node| {
                            let weight = self.dag.node_weight(node).unwrap();
                            if let NodeType::Operation(packed) = weight {
                                !ignore_set.contains(packed.op.name())
                            } else {
                                false
                            }
                        });

                    if !nodes_found {
                        result.push(match wire {
                            Wire::Qubit(qubit) => {
                                self.qubits.get(qubit).unwrap().into_py_any(py)?
                            }
                            Wire::Clbit(clbit) => {
                                self.clbits.get(clbit).unwrap().into_py_any(py)?
                            }
                            Wire::Var(var) => self
                                .vars_stretches
                                .vars()
                                .get(var)
                                .unwrap()
                                .clone()
                                .into_py_any(py)?,
                        });
                    }
                }
            }
            None => {
                for wire in wires {
                    if self.is_wire_idle(wire) {
                        result.push(match wire {
                            Wire::Qubit(qubit) => {
                                self.qubits.get(qubit).unwrap().into_py_any(py)?
                            }
                            Wire::Clbit(clbit) => {
                                self.clbits.get(clbit).unwrap().into_py_any(py)?
                            }
                            Wire::Var(var) => self
                                .vars_stretches
                                .vars()
                                .get(var)
                                .unwrap()
                                .clone()
                                .into_py_any(py)?,
                        });
                    }
                }
            }
        }
        Ok(PyTuple::new(py, result)?.into_any().try_iter()?.unbind())
    }

    /// Return `true` if there are no operation nodes in the graph.
    fn is_empty(&self) -> bool {
        self.dag.node_count()
            == 2 * (self.qubits.len() + self.clbits.len() + self.vars_stretches.vars().len())
    }

    /// Return the number of operations.  If there is control flow present, this count may only
    /// be an estimate, as the complete control-flow path cannot be statically known.
    ///
    /// Args:
    ///     recurse: if ``True``, then recurse into control-flow operations.  For loops with
    ///         known-length iterators are counted unrolled.  If-else blocks sum both of the two
    ///         branches.  While loops are counted as if the loop body runs once only.  Defaults to
    ///         ``False`` and raises :class:`.DAGCircuitError` if any control flow is present, to
    ///         avoid silently returning a mostly meaningless number.
    ///
    /// Returns:
    ///     int: the circuit size
    ///
    /// Raises:
    ///     DAGCircuitError: if an unknown :class:`.ControlFlowOp` is present in a call with
    ///         ``recurse=True``, or any control flow is present in a non-recursive call.
    #[pyo3(signature= (*, recurse=false))]
    pub fn size(&self, recurse: bool) -> Result<usize, DAGError> {
        let mut length = self.num_ops();
        if !self.has_control_flow() {
            return Ok(length);
        }
        if !recurse {
            return Err(DAGError::General(
                concat!(
                    "Size with control flow is ambiguous.",
                    " You may use `recurse=True` to get a result",
                    " but see this method's documentation for the meaning of this."
                )
                .into(),
            ));
        }

        // Handle recursively.
        for control_flow in self
            .op_nodes(false)
            .filter_map(|(_, node)| self.try_view_control_flow(node))
        {
            match control_flow {
                ControlFlowView::ForLoop {
                    collection, body, ..
                } => {
                    // TODO: is this the intended logic?
                    length += collection.len() * body.size(true)?;
                }
                _ => {
                    for block in control_flow.blocks() {
                        length += block.size(true)?;
                    }
                }
            }
            // We don't count a control-flow node itself!
            length -= 1;
        }
        Ok(length)
    }

    /// Return the circuit depth.  If there is control flow present, this count may only be an
    /// estimate, as the complete control-flow path cannot be statically known.
    ///
    /// Args:
    ///     recurse: if ``True``, then recurse into control-flow operations.  For loops
    ///         with known-length iterators are counted as if the loop had been manually unrolled
    ///         (*i.e.* with each iteration of the loop body written out explicitly).
    ///         If-else blocks take the longer case of the two branches.  While loops are counted as
    ///         if the loop body runs once only.  Defaults to ``False`` and raises
    ///         :class:`.DAGCircuitError` if any control flow is present, to avoid silently
    ///         returning a nonsensical number.
    ///
    /// Returns:
    ///     int: the circuit depth
    ///
    /// Raises:
    ///     DAGCircuitError: if not a directed acyclic graph
    ///     DAGCircuitError: if unknown control flow is present in a recursive call, or any control
    ///         flow is present in a non-recursive call.
    #[pyo3(signature= (*, recurse=false))]
    pub fn depth(&self, recurse: bool) -> Result<usize, DAGError> {
        if self.qubits.is_empty() && self.clbits.is_empty() && self.num_vars() == 0 {
            return Ok(0);
        }
        if !self.has_control_flow() {
            let weight_fn = |_| -> Result<usize, Infallible> { Ok(1) };
            return match rustworkx_core::dag_algo::longest_path_length(&self.dag, weight_fn)
                .unwrap()
            {
                Some(res) => Ok(res - 1),
                None => panic!("not a DAG"),
            };
        }
        if !recurse {
            return Err(DAGError::General(
                concat!(
                    "Depth with control flow is ambiguous.",
                    " You may use `recurse=True` to get a result",
                    " but see this method's documentation for the meaning of this."
                )
                .into(),
            ));
        }
        // Handle recursively.
        let mut node_lookup: HashMap<NodeIndex, usize> = HashMap::new();
        for (node_index, control_flow) in self
            .op_nodes(false)
            .filter_map(|(index, node)| self.try_view_control_flow(node).map(|cf| (index, cf)))
        {
            let weight = if let ControlFlowView::ForLoop { collection, .. } = control_flow {
                collection.len()
            } else {
                1
            };
            if weight == 0 {
                node_lookup.insert(node_index, 0);
            } else {
                let blocks = control_flow.blocks();
                let mut block_weights: Vec<usize> = Vec::with_capacity(blocks.len());
                for block in blocks {
                    block_weights.push(block.depth(true)?);
                    node_lookup.insert(node_index, weight * block_weights.iter().max().unwrap());
                }
            }
        }

        let weight_fn = |edge: EdgeReference<'_, Wire>| -> Result<usize, Infallible> {
            Ok(*node_lookup.get(&edge.target()).unwrap_or(&1))
        };
        match rustworkx_core::dag_algo::longest_path_length(&self.dag, weight_fn).unwrap() {
            Some(res) => Ok(res - 1),
            None => panic!("not a DAG"),
        }
    }

    /// Return the total number of qubits + clbits used by the circuit.
    /// This function formerly returned the number of qubits by the calculation
    /// return len(self._wires) - self.num_clbits()
    /// but was changed by issue #2564 to return number of qubits + clbits
    /// with the new function DAGCircuit.num_qubits replacing the former
    /// semantic of DAGCircuit.width().
    pub fn width(&self) -> usize {
        self.qubits.len() + self.clbits.len() + self.num_vars()
    }

    /// Return the total number of qubits used by the circuit.
    /// num_qubits() replaces former use of width().
    /// DAGCircuit.width() now returns qubits + clbits for
    /// consistency with Circuit.width() [qiskit-terra #2564].
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Return the total number of classical bits used by the circuit.
    pub fn num_clbits(&self) -> usize {
        self.clbits.len()
    }

    /// Return the number of basic blocks in this circuit.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the number of op nodes in the DAG.
    #[inline]
    pub fn num_ops(&self) -> usize {
        self.dag.node_count() - 2 * self.width()
    }

    /// Compute how many components the circuit can decompose into.
    fn num_tensor_factors(&self) -> usize {
        // This function was forked from rustworkx's
        // number_weekly_connected_components() function as of 0.15.0:
        // https://github.com/Qiskit/rustworkx/blob/0.15.0/src/connectivity/mod.rs#L215-L235

        let mut weak_components = self.dag.node_count();
        let mut vertex_sets = UnionFind::new(self.dag.node_bound());
        for edge in self.dag.edge_references() {
            let (a, b) = (edge.source(), edge.target());
            // union the two vertices of the edge
            if vertex_sets.union(a.index(), b.index()) {
                weak_components -= 1
            };
        }
        weak_components
    }

    fn __eq__(&self, py: Python, other: &DAGCircuit) -> PyResult<bool> {
        fn eq_inner(
            py: Python,
            slf: &DAGCircuit,
            slf_qubit_map: &HashMap<Qubit, Qubit>,
            slf_clbit_map: &HashMap<Clbit, Clbit>,
            other: &DAGCircuit,
            other_qubit_map: &HashMap<Qubit, Qubit>,
            other_clbit_map: &HashMap<Clbit, Clbit>,
        ) -> PyResult<bool> {
            // Try to convert to float, but in case of unbound ParameterExpressions
            // a TypeError will be raise, fallback to normal equality in those
            // cases.
            let phase_is_close = |self_phase: f64, other_phase: f64| -> bool {
                ((self_phase - other_phase + PI).rem_euclid(2. * PI) - PI).abs() <= 1.0e-10
            };
            let normalize_param = |param: &Param| {
                if let Param::ParameterExpression(ob) = param {
                    // try casting ParameterExpression to Param, prioritizing Float
                    Param::from_expr(ob.as_ref().clone(), true)
                } else {
                    Ok(param.clone())
                }
            };

            let phase_eq = match [
                normalize_param(&slf.global_phase)?,
                normalize_param(&other.global_phase)?,
            ] {
                [Param::Float(self_phase), Param::Float(other_phase)] => {
                    Ok(phase_is_close(self_phase, other_phase))
                }
                _ => slf.global_phase.eq(&other.global_phase),
            }?;
            if !phase_eq {
                return Ok(false);
            }

            // We don't do any semantic equivalence between Var nodes, as things stand; DAGs can only be
            // equal in our mind if they use the exact same UUID vars.
            if slf.vars_stretches != other.vars_stretches {
                return Ok(false);
            }

            // Check if qregs are the same.
            let self_qregs = slf.qregs.registers();
            let other_qregs = &other.qregs;
            if self_qregs.len() != other_qregs.len() {
                return Ok(false);
            }
            for (regname, self_bits) in self_qregs.iter().map(|reg| (reg.name(), reg)) {
                let self_bits: Vec<ShareableQubit> = self_bits.bits().collect();
                let other_bits: Vec<ShareableQubit> = match other_qregs.get(regname) {
                    Some(bits) => bits.bits().collect(),
                    None => return Ok(false),
                };
                if !slf
                    .qubits
                    .map_objects(self_bits)?
                    .map(|q| &slf_qubit_map[&q])
                    .eq(other
                        .qubits
                        .map_objects(other_bits)?
                        .map(|q| &other_qubit_map[&q]))
                {
                    return Ok(false);
                }
            }

            // Check if cregs are the same.
            let self_cregs = slf.cregs.registers();
            let other_cregs = &other.cregs;
            if self_cregs.len() != other_cregs.len() {
                return Ok(false);
            }

            for (regname, self_bits) in self_cregs.iter().map(|reg| (reg.name(), reg)) {
                let self_bits: Vec<ShareableClbit> = self_bits.bits().collect();
                let other_bits: Vec<ShareableClbit> = match other_cregs.get(regname) {
                    Some(bits) => bits.bits().collect(),
                    None => return Ok(false),
                };
                if !slf
                    .clbits
                    .map_objects(self_bits)?
                    .map(|c| &slf_clbit_map[&c])
                    .eq(other
                        .clbits
                        .map_objects(other_bits)?
                        .map(|c| &other_clbit_map[&c]))
                {
                    return Ok(false);
                }
            }

            // Check for VF2 isomorphic match.
            let node_match = |n1: &NodeType, n2: &NodeType| -> PyResult<bool> {
                match [n1, n2] {
                    [NodeType::Operation(inst1), NodeType::Operation(inst2)] => {
                        if inst1.op.name() != inst2.op.name() {
                            return Ok(false);
                        }
                        let check_args = || -> bool {
                            let node1_qargs = slf
                                .qargs_interner
                                .get(inst1.qubits)
                                .iter()
                                .map(|q| &slf_qubit_map[q]);
                            let node2_qargs = other
                                .qargs_interner
                                .get(inst2.qubits)
                                .iter()
                                .map(|q| &other_qubit_map[q]);
                            let node1_cargs = slf
                                .cargs_interner
                                .get(inst1.clbits)
                                .iter()
                                .map(|c| &slf_clbit_map[c]);
                            let node2_cargs = other
                                .cargs_interner
                                .get(inst2.clbits)
                                .iter()
                                .map(|c| &other_clbit_map[c]);
                            if SEMANTIC_EQ_SYMMETRIC.contains(&inst1.op.name()) {
                                let node1_qargs = node1_qargs.collect::<HashSet<&Qubit>>();
                                let node2_qargs = node2_qargs.collect::<HashSet<&Qubit>>();
                                let node1_cargs = node1_cargs.collect::<HashSet<&Clbit>>();
                                let node2_cargs = node2_cargs.collect::<HashSet<&Clbit>>();
                                if node1_qargs != node2_qargs || node1_cargs != node2_cargs {
                                    return false;
                                }
                            } else if node1_qargs.ne(node2_qargs) || node1_cargs.ne(node2_cargs) {
                                return false;
                            }
                            true
                        };
                        if let (Some(cf1), Some(cf2)) = (
                            slf.try_view_control_flow(inst1),
                            other.try_view_control_flow(inst2),
                        ) {
                            if inst1.op.num_qubits() != inst2.op.num_qubits() {
                                return Ok(false);
                            }
                            // These map the bit indices used in blocks to their indices in the
                            // root DAG.
                            let slf_block_qubit_map = (0..inst1.op.num_qubits())
                                .map(Qubit)
                                .zip(
                                    slf.qargs_interner
                                        .get(inst1.qubits)
                                        .iter()
                                        .map(|q| slf_qubit_map[q]),
                                )
                                .collect();
                            let slf_block_clbit_map = (0..inst1.op.num_clbits())
                                .map(Clbit)
                                .zip(
                                    slf.cargs_interner
                                        .get(inst1.clbits)
                                        .iter()
                                        .map(|c| slf_clbit_map[c]),
                                )
                                .collect();
                            let other_block_qubit_map = (0..inst2.op.num_qubits())
                                .map(Qubit)
                                .zip(
                                    other
                                        .qargs_interner
                                        .get(inst2.qubits)
                                        .iter()
                                        .map(|q| other_qubit_map[q]),
                                )
                                .collect();
                            let other_block_clbit_map = (0..inst2.op.num_clbits())
                                .map(Clbit)
                                .zip(
                                    other
                                        .cargs_interner
                                        .get(inst2.clbits)
                                        .iter()
                                        .map(|c| other_clbit_map[c]),
                                )
                                .collect();

                            let block_eq = |slf_block: &DAGCircuit,
                                            other_block: &DAGCircuit|
                             -> PyResult<bool> {
                                eq_inner(
                                    py,
                                    slf_block,
                                    &slf_block_qubit_map,
                                    &slf_block_clbit_map,
                                    other_block,
                                    &other_block_qubit_map,
                                    &other_block_clbit_map,
                                )
                            };

                            #[derive(Debug, PartialEq)]
                            enum VarKey {
                                Bit(Clbit),
                                Register(Vec<Clbit>),
                                Var(expr::Var),
                            }

                            let slf_var_key = |v: &expr::Var| -> VarKey {
                                match v {
                                    expr::Var::Standalone { .. } => VarKey::Var(v.clone()),
                                    expr::Var::Bit { bit } => {
                                        VarKey::Bit(slf_clbit_map[&slf.clbits.find(bit).unwrap()])
                                    }
                                    expr::Var::Register { register, .. } => VarKey::Register(
                                        register
                                            .iter()
                                            .map(|bit| {
                                                slf_clbit_map[&slf.clbits.find(&bit).unwrap()]
                                            })
                                            .collect(),
                                    ),
                                }
                            };
                            let other_var_key = |v: &expr::Var| -> VarKey {
                                match v {
                                    expr::Var::Standalone { .. } => VarKey::Var(v.clone()),
                                    expr::Var::Bit { bit } => VarKey::Bit(
                                        other_clbit_map[&other.clbits.find(bit).unwrap()],
                                    ),
                                    expr::Var::Register { register, .. } => VarKey::Register(
                                        register
                                            .iter()
                                            .map(|bit| {
                                                other_clbit_map[&other.clbits.find(&bit).unwrap()]
                                            })
                                            .collect(),
                                    ),
                                }
                            };

                            let condition_eq = |condition_a: &Condition,
                                                condition_b: &Condition|
                             -> bool {
                                match condition_a {
                                    Condition::Bit(bit_a, value_a) => match condition_b {
                                        Condition::Bit(bit_b, value_b) => {
                                            value_a == value_b
                                                && slf_clbit_map[&slf.clbits.find(bit_a).unwrap()]
                                                    == other_clbit_map
                                                        [&other.clbits.find(bit_b).unwrap()]
                                        }
                                        _ => false,
                                    },
                                    Condition::Register(reg_a, value_a) => match condition_b {
                                        Condition::Register(reg_b, value_b) => {
                                            value_a == value_b
                                                && reg_a
                                                    .iter()
                                                    .map(|a| {
                                                        slf_clbit_map[&slf.clbits.find(&a).unwrap()]
                                                    })
                                                    .eq(reg_b.iter().map(|b| {
                                                        other_clbit_map
                                                            [&other.clbits.find(&b).unwrap()]
                                                    }))
                                        }
                                        _ => false,
                                    },
                                    Condition::Expr(expr_a) => match condition_b {
                                        Condition::Expr(expr_b) => expr_a
                                            .structurally_equivalent_by_key(
                                                slf_var_key,
                                                expr_b,
                                                other_var_key,
                                            ),
                                        _ => false,
                                    },
                                }
                            };

                            match (cf1, cf2) {
                                (
                                    ControlFlowView::Box {
                                        duration: duration_a,
                                        body: body_a,
                                    },
                                    ControlFlowView::Box {
                                        duration: duration_b,
                                        body: body_b,
                                    },
                                ) => {
                                    let duration_eq = match duration_a {
                                        Some(BoxDuration::Expr(duration_a)) => match duration_b {
                                            Some(BoxDuration::Expr(duration_b)) => duration_a
                                                .structurally_equivalent_by_key(
                                                    &slf_var_key,
                                                    duration_b,
                                                    &other_var_key,
                                                ),
                                            _ => false,
                                        },
                                        Some(BoxDuration::Duration(duration_a)) => match duration_b
                                        {
                                            Some(BoxDuration::Duration(duration_b)) => {
                                                duration_a == duration_b
                                            }
                                            _ => false,
                                        },
                                        None => duration_b.is_none(),
                                    };
                                    Ok(duration_eq && block_eq(body_a, body_b)?)
                                }
                                (ControlFlowView::BreakLoop, ControlFlowView::BreakLoop) => {
                                    Ok(true)
                                }
                                (ControlFlowView::ContinueLoop, ControlFlowView::ContinueLoop) => {
                                    Ok(true)
                                }
                                (
                                    ControlFlowView::ForLoop {
                                        collection: collection_a,
                                        loop_param: loop_param_a,
                                        body: body_a,
                                    },
                                    ControlFlowView::ForLoop {
                                        collection: collection_b,
                                        loop_param: loop_param_b,
                                        body: body_b,
                                    },
                                ) => {
                                    if collection_a != collection_b {
                                        return Ok(false);
                                    }
                                    match (loop_param_a, loop_param_b) {
                                        (
                                            Some(LoopParam::Parameter(loop_param_a)),
                                            Some(LoopParam::Parameter(loop_param_b)),
                                        ) => {
                                            // Until we have a way to assign parameters in a DAG, we need
                                            // to convert a for loop's body DAG back to a circuit.
                                            let sentinel = PARAMETER
                                                .get_bound(py)
                                                .call1((Uuid::new_v4().to_string(),))?;
                                            let mut body_a_circuit =
                                                CircuitData::from_dag_ref(body_a)?;
                                            if body_a_circuit.uses_parameter(loop_param_a) {
                                                body_a_circuit.assign_parameters_from_mapping([
                                                    (
                                                        ParameterUuid::from_symbol(loop_param_a),
                                                        Param::ParameterExpression(Arc::new(
                                                            ParameterExpression::from_symbol(
                                                                sentinel.clone().extract()?,
                                                            ),
                                                        )),
                                                    ),
                                                ])?;
                                            }
                                            let body_a = DAGCircuit::from_circuit(
                                                QuantumCircuitData {
                                                    data: body_a_circuit,
                                                    name: body_a.name.clone(),
                                                    metadata: body_a
                                                        .metadata
                                                        .as_ref()
                                                        .map(|m| m.bind(py).clone()),
                                                    transpile_layout: None,
                                                },
                                                false,
                                                None,
                                                None,
                                            )?;

                                            let mut body_b_circuit =
                                                CircuitData::from_dag_ref(body_b)?;
                                            if body_b_circuit.uses_parameter(loop_param_b) {
                                                body_b_circuit.assign_parameters_from_mapping([
                                                    (
                                                        ParameterUuid::from_symbol(loop_param_b),
                                                        Param::ParameterExpression(Arc::new(
                                                            ParameterExpression::from_symbol(
                                                                sentinel.clone().extract()?,
                                                            ),
                                                        )),
                                                    ),
                                                ])?;
                                            }
                                            let body_b = DAGCircuit::from_circuit(
                                                QuantumCircuitData {
                                                    data: body_b_circuit,
                                                    name: body_b.name.clone(),
                                                    metadata: body_b
                                                        .metadata
                                                        .as_ref()
                                                        .map(|m| m.bind(py).clone()),
                                                    transpile_layout: None,
                                                },
                                                false,
                                                None,
                                                None,
                                            )?;
                                            block_eq(&body_a, &body_b)
                                        }
                                        (
                                            Some(LoopParam::Variable(var_a)),
                                            Some(LoopParam::Variable(var_b)),
                                        ) => {
                                            if var_a != var_b {
                                                return Ok(false);
                                            }
                                            block_eq(body_a, body_b)
                                        }
                                        (None, None) => block_eq(body_a, body_b),
                                        _ => Ok(false),
                                    }
                                }
                                (
                                    ControlFlowView::IfElse {
                                        condition: condition_a,
                                        true_body: true_body_a,
                                        false_body: false_body_a,
                                    },
                                    ControlFlowView::IfElse {
                                        condition: condition_b,
                                        true_body: true_body_b,
                                        false_body: false_body_b,
                                    },
                                ) => {
                                    let false_body_eq = || -> PyResult<bool> {
                                        match (false_body_a, false_body_b) {
                                            (Some(false_body_a), Some(false_body_b)) => {
                                                block_eq(false_body_a, false_body_b)
                                            }
                                            (None, None) => Ok(true),
                                            _ => Ok(false),
                                        }
                                    };
                                    Ok(condition_eq(condition_a, condition_b)
                                        && block_eq(true_body_a, true_body_b)?
                                        && false_body_eq()?)
                                }
                                (
                                    ControlFlowView::Switch {
                                        target: target_a,
                                        cases_specifier: cases_a,
                                    },
                                    ControlFlowView::Switch {
                                        target: target_b,
                                        cases_specifier: cases_b,
                                    },
                                ) => {
                                    let target_eq = || -> bool {
                                        match target_a {
                                            SwitchTarget::Bit(bit_a) => match target_b {
                                                SwitchTarget::Bit(bit_b) => {
                                                    slf_clbit_map[&slf.clbits.find(bit_a).unwrap()]
                                                        == other_clbit_map
                                                            [&other.clbits.find(bit_b).unwrap()]
                                                }
                                                _ => false,
                                            },
                                            SwitchTarget::Register(reg_a) => match target_b {
                                                SwitchTarget::Register(reg_b) => reg_a
                                                    .iter()
                                                    .map(|a| {
                                                        slf_clbit_map[&slf.clbits.find(&a).unwrap()]
                                                    })
                                                    .eq(reg_b.iter().map(|b| {
                                                        other_clbit_map
                                                            [&other.clbits.find(&b).unwrap()]
                                                    })),
                                                _ => false,
                                            },
                                            SwitchTarget::Expr(expr_a) => match target_b {
                                                SwitchTarget::Expr(expr_b) => expr_a
                                                    .structurally_equivalent_by_key(
                                                        slf_var_key,
                                                        expr_b,
                                                        other_var_key,
                                                    ),
                                                _ => false,
                                            },
                                        }
                                    };
                                    if cases_a.len() != cases_b.len() || !target_eq() {
                                        return Ok(false);
                                    }
                                    for ((a_label_spec, a_block), (b_label_spec, b_block)) in
                                        cases_a.iter().zip(cases_b.iter())
                                    {
                                        if a_label_spec.iter().collect::<HashSet<_>>()
                                            != b_label_spec.iter().collect::<HashSet<_>>()
                                        {
                                            return Ok(false);
                                        }
                                        if !block_eq(a_block, b_block)? {
                                            return Ok(false);
                                        }
                                    }
                                    Ok(true)
                                }
                                (
                                    ControlFlowView::While {
                                        condition: condition_a,
                                        body: body_a,
                                    },
                                    ControlFlowView::While {
                                        condition: condition_b,
                                        body: body_b,
                                    },
                                ) => Ok(condition_eq(condition_a, condition_b)
                                    && block_eq(body_a, body_b)?),
                                _ => Ok(false),
                            }
                        } else {
                            match [inst1.op.view(), inst2.op.view()] {
                                [
                                    OperationRef::StandardGate(gate1),
                                    OperationRef::StandardGate(gate2),
                                ] => Ok(gate1 == gate2
                                    && check_args()
                                    && inst1
                                        .params_view()
                                        .iter()
                                        .zip(inst2.params_view())
                                        .all(|(a, b)| a.is_close(b, 1e-10).unwrap())),
                                [
                                    OperationRef::StandardInstruction(op1),
                                    OperationRef::StandardInstruction(op2),
                                ] => Ok(match [op1, op2] {
                                    [
                                        StandardInstruction::Barrier(n1),
                                        StandardInstruction::Barrier(n2),
                                    ] => n1 == n2,
                                    [
                                        StandardInstruction::Delay(unit1),
                                        StandardInstruction::Delay(unit2),
                                    ] => {
                                        let duration1 = &inst1.params_view()[0];
                                        let duration2 = &inst2.params_view()[0];
                                        unit1 == unit2 && duration1.is_close(duration2, 1e-10)?
                                    }
                                    [
                                        StandardInstruction::Measure,
                                        StandardInstruction::Measure,
                                    ] => true,
                                    [StandardInstruction::Reset, StandardInstruction::Reset] => {
                                        true
                                    }
                                    _ => false,
                                } && check_args()),
                                [OperationRef::PyCustom(op1), OperationRef::PyCustom(op2)] => {
                                    Ok(op1.ob.bind(py).eq(&op2.ob)? && check_args())
                                }
                                // Handle the edge case where we end up with a Python object and a standard
                                // gate/instruction.
                                // This typically only happens if we have a ControlledGate in Python
                                // and we have mutable state set.
                                [
                                    OperationRef::StandardGate(_),
                                    OperationRef::PyCustom(PyInstruction {
                                        kind: PyOpKind::Gate,
                                        ob,
                                        ..
                                    }),
                                ] => Ok(slf.unpack_py_op(py, inst1)?.eq(ob)? && check_args()),
                                [
                                    OperationRef::PyCustom(PyInstruction {
                                        kind: PyOpKind::Gate,
                                        ob,
                                        ..
                                    }),
                                    OperationRef::StandardGate(_),
                                ] => Ok(other.unpack_py_op(py, inst2)?.eq(ob)? && check_args()),
                                [
                                    OperationRef::StandardInstruction(_),
                                    OperationRef::PyCustom(PyInstruction {
                                        kind: PyOpKind::Instruction,
                                        ob,
                                        ..
                                    }),
                                ] => Ok(slf.unpack_py_op(py, inst1)?.eq(ob)? && check_args()),
                                [
                                    OperationRef::PyCustom(PyInstruction {
                                        kind: PyOpKind::Instruction,
                                        ob,
                                        ..
                                    }),
                                    OperationRef::StandardInstruction(_),
                                ] => Ok(other.unpack_py_op(py, inst2)?.eq(ob)? && check_args()),
                                [OperationRef::Unitary(op_a), OperationRef::Unitary(op_b)] => {
                                    match [&op_a.array, &op_b.array] {
                                        [ArrayType::NDArray(a), ArrayType::NDArray(b)] => Ok(
                                            relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-8),
                                        ),
                                        [ArrayType::OneQ(a), ArrayType::NDArray(b)]
                                        | [ArrayType::NDArray(b), ArrayType::OneQ(a)] => {
                                            if b.shape()[0] == 2 {
                                                for i in 0..2 {
                                                    for j in 0..2 {
                                                        if !relative_eq!(
                                                            b[[i, j]],
                                                            a[(i, j)],
                                                            max_relative = 1e-5,
                                                            epsilon = 1e-8
                                                        ) {
                                                            return Ok(false);
                                                        }
                                                    }
                                                }
                                                Ok(true)
                                            } else {
                                                Ok(false)
                                            }
                                        }
                                        [ArrayType::TwoQ(a), ArrayType::NDArray(b)]
                                        | [ArrayType::NDArray(b), ArrayType::TwoQ(a)] => {
                                            if b.shape()[0] == 4 {
                                                for i in 0..4 {
                                                    for j in 0..4 {
                                                        if !relative_eq!(
                                                            b[[i, j]],
                                                            a[(i, j)],
                                                            max_relative = 1e-5,
                                                            epsilon = 1e-8
                                                        ) {
                                                            return Ok(false);
                                                        }
                                                    }
                                                }
                                                Ok(true)
                                            } else {
                                                Ok(false)
                                            }
                                        }
                                        [ArrayType::OneQ(a), ArrayType::OneQ(b)] => Ok(
                                            relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-8),
                                        ),
                                        [ArrayType::TwoQ(a), ArrayType::TwoQ(b)] => Ok(
                                            relative_eq!(a, b, max_relative = 1e-5, epsilon = 1e-8),
                                        ),
                                        _ => Ok(false),
                                    }
                                }
                                [
                                    OperationRef::PauliProductMeasurement(op_a),
                                    OperationRef::PauliProductMeasurement(op_b),
                                ] => Ok(op_a == op_b),
                                [
                                    OperationRef::PauliProductRotation(op_a),
                                    OperationRef::PauliProductRotation(op_b),
                                ] => Ok(op_a == op_b),
                                _ => Ok(false),
                            }
                        }
                    }
                    [NodeType::QubitIn(bit1), NodeType::QubitIn(bit2)] => {
                        Ok(slf_qubit_map[bit1] == other_qubit_map[bit2])
                    }
                    [NodeType::ClbitIn(bit1), NodeType::ClbitIn(bit2)] => {
                        Ok(slf_clbit_map[bit1] == other_clbit_map[bit2])
                    }
                    [NodeType::QubitOut(bit1), NodeType::QubitOut(bit2)] => {
                        Ok(slf_qubit_map[bit1] == other_qubit_map[bit2])
                    }
                    [NodeType::ClbitOut(bit1), NodeType::ClbitOut(bit2)] => {
                        Ok(slf_clbit_map[bit1] == other_clbit_map[bit2])
                    }
                    [NodeType::VarIn(var1), NodeType::VarIn(var2)]
                    | [NodeType::VarOut(var1), NodeType::VarOut(var2)] => {
                        // TODO: do we need to convert classical bits and registers here?
                        //   no if Var wires are always standalone
                        Ok(slf.vars_stretches.vars().get(*var1).unwrap()
                            == other.vars_stretches.vars().get(*var2).unwrap())
                    }
                    _ => Ok(false),
                }
            };
            let node_match = |n1: &NodeType, n2: &NodeType| -> PyResult<Option<()>> {
                node_match(n1, n2).map(|ok| ok.then_some(()))
            };

            vf2::is_isomorphic_with_semantics(
                &slf.dag,
                &other.dag,
                (node_match, vf2::NoSemantics::new()),
                true,
                vf2::Problem::Exact,
                None,
            )
            .map_err(|e| match e {
                vf2::IsIsomorphicError::NodeMatcher(e) => e,
                vf2::IsIsomorphicError::EdgeMatcher(_) => unreachable!(),
            })
        }
        if self.num_qubits() != other.num_qubits() || self.num_clbits() != other.num_clbits() {
            return Ok(false);
        }
        let slf_qubit_map: HashMap<Qubit, Qubit> = (0..self.num_qubits())
            .map(|i| (Qubit::new(i), Qubit::new(i)))
            .collect();
        let slf_clbit_map: HashMap<Clbit, Clbit> = (0..self.num_clbits())
            .map(|i| (Clbit::new(i), Clbit::new(i)))
            .collect();
        eq_inner(
            py,
            self,
            &slf_qubit_map,
            &slf_clbit_map,
            other,
            &slf_qubit_map,
            &slf_clbit_map,
        )
    }

    /// Are these two DAGs structurally equal?
    ///
    /// This function returns true iff the graph structures are precisely the same as each other,
    /// including the valid node indices, edge orders, and so on.  This is a much stricter check
    /// than graph equivalence, and is mostly useful for testing if two :class:`DAGCircuit`
    /// instances have been constructed and manipulated in the exact same ways.  For example, this
    /// method can be used to test whether a sequence of manipulations of a DAG is deterministic.
    ///
    /// This method does not consider tracking metadata such as :attr:`metadata` or :attr:`name`,
    /// but does consider many low-level implementation details of the internal representation, many
    /// of which do not change the semantics of the circuit.
    ///
    /// This method should, in general, be much faster than graph-equivalence checks, but will
    /// return ``False`` in many more situations.  This method should never return ``True`` when a
    /// graph-equivalence check would return ``False``.
    ///
    /// .. note::
    ///
    ///     This currently does not handle control flow, because of technical limitations in the
    ///     internal representation of control flow, and will return `false` if any control-flow
    ///     operation is present, even if they are individually equal.
    ///
    /// .. seealso::
    ///     The ``==`` operator
    ///         :class:`DAGCircuit` implements :func:`~object.__eq__` between itself and other
    ///         :class:`DAGCircuit` instances (this same method also powers
    ///         :class:`.QuantumCircuit`'s equality check).  This implements a semantic
    ///         data-flow equality check, which is less sensitive to the order operations were
    ///         defined.  This is typically what a user cares about with respect to equality.
    fn structurally_equal(&self, other: &DAGCircuit) -> PyResult<bool> {
        if self.qubits != other.qubits {
            return Ok(false);
        }
        if self.clbits != other.clbits {
            return Ok(false);
        }
        if !self
            .vars_stretches
            .structurally_equal(&other.vars_stretches)
        {
            return Ok(false);
        }
        if self.qregs != other.qregs {
            return Ok(false);
        }
        if self.cregs != other.cregs {
            return Ok(false);
        }
        // This is a stricter check than `Param::eq`, since we don't allow equality between explicit
        // floats and Python-object representations of the same float.
        let param_eq = |left: &Param, right: &Param| -> PyResult<bool> {
            match (left, right) {
                (Param::Float(a), Param::Float(b)) => Ok(a.total_cmp(b) == Ordering::Equal),
                (Param::ParameterExpression(a), Param::ParameterExpression(b)) => Ok(a.eq(b)),
                (Param::Obj(a), Param::Obj(b)) => Python::attach(|py| a.bind(py).eq(b.bind(py))),
                _ => Ok(false),
            }
        };
        if !param_eq(&self.global_phase, &other.global_phase)? {
            return Ok(false);
        }
        // This is stricter than `PackedInstruction::py_op_eq` because it doesn't allow equality
        // between `StandardGate` and Python versions of that.  Additionally, it short-circuits out
        // of control-flow, rather than recursing through while we currently don't have a clean
        // representation of DAG-like structured control flow.
        let inst_eq = |from_self: &PackedInstruction,
                       from_other: &PackedInstruction|
         -> PyResult<bool> {
            if from_self.op.try_control_flow().is_some()
                || from_other.op.try_control_flow().is_some()
            {
                return Ok(false);
            }
            if self.qargs_interner.get(from_self.qubits)
                != other.qargs_interner.get(from_other.qubits)
            {
                return Ok(false);
            }
            if self.cargs_interner.get(from_self.clbits)
                != other.cargs_interner.get(from_other.clbits)
            {
                return Ok(false);
            }
            let from_self_params = from_self.params_view();
            let from_other_params = from_other.params_view();
            if from_self_params.len() != from_other_params.len() {
                return Ok(false);
            }
            for (left, right) in from_self_params.iter().zip(from_other_params.iter()) {
                if !param_eq(left, right)? {
                    return Ok(false);
                }
            }
            match (from_self.op.view(), from_other.op.view()) {
                (OperationRef::StandardGate(left), OperationRef::StandardGate(right)) => {
                    Ok(left == right)
                }
                (
                    OperationRef::StandardInstruction(left),
                    OperationRef::StandardInstruction(right),
                ) => Ok(left == right),
                (OperationRef::Unitary(left), OperationRef::Unitary(right)) => Ok(left == right),
                (OperationRef::PyCustom(left), OperationRef::PyCustom(right)) => {
                    Python::attach(|py| left.ob.bind(py).eq(&right.ob))
                }
                _ => Ok(false),
            }
        };
        // We use this helper because the default impl of `PartialEq` doesn't check the source and
        // target, since it usually doesn't make sense to compare references between graphs.  It
        // makes sense for us, because we're checking that the graphs are structurally identical.
        let edgeref_eq = |left: EdgeReference<'_, Wire>, right: EdgeReference<'_, Wire>| -> bool {
            left.source() == right.source()
                && left.target() == right.target()
                && left.id() == right.id()
                && left.weight() == right.weight()
        };

        // Now the meat of the actual comparison.  This is deliberately non-topological and
        // index-sensitive - that's exactly what we're testing for.
        if self.dag.node_count() != other.dag.node_count()
            || self.dag.edge_count() != other.dag.edge_count()
        {
            return Ok(false);
        }
        for (self_index, other_index) in self.dag.node_indices().zip(other.dag.node_indices()) {
            // The `NodeIndex` values should be the same for both; for the example of two DAGs that
            // have undergone a deterministic compilation pipeline, this means that they've seen the
            // same sequence of additions, removals and substitutions (or at least a comparable
            // sequence), which they should have.
            if self_index != other_index {
                return Ok(false);
            }
            if !self.dag[self_index].equal_with(&other.dag[other_index], inst_eq)? {
                return Ok(false);
            }
            for pair in self
                .dag
                .edges_directed(self_index, Direction::Incoming)
                .zip_longest(other.dag.edges_directed(other_index, Direction::Incoming))
            {
                match pair {
                    EitherOrBoth::Both(left, right) if edgeref_eq(left, right) => (),
                    _ => return Ok(false),
                }
            }
            for pair in self
                .dag
                .edges_directed(self_index, Direction::Outgoing)
                .zip_longest(other.dag.edges_directed(other_index, Direction::Outgoing))
            {
                match pair {
                    EitherOrBoth::Both(left, right) if edgeref_eq(left, right) => (),
                    _ => return Ok(false),
                }
            }
        }
        Ok(true)
    }

    /// Yield nodes in topological order.
    ///
    /// Args:
    ///     key (Callable): A callable which will take a DAGNode object and
    ///         return a string sort key. If not specified the bit qargs and
    ///         cargs of a node will be used for sorting.
    ///     reverse (bool): If True, yield nodes in reverse topological order.
    ///
    /// Returns:
    ///     generator(DAGOpNode, DAGInNode, or DAGOutNode): node in topological order
    #[pyo3(name = "topological_nodes", signature=(key=None, reverse=false))]
    fn py_topological_nodes(
        &self,
        py: Python,
        key: Option<Bound<PyAny>>,
        reverse: bool,
    ) -> PyResult<Py<PyIterator>> {
        let nodes: PyResult<Vec<_>> = if let Some(key) = key {
            self.topological_key_sort(py, &key, reverse)?
                .map(|node| self.get_node(py, node))
                .collect()
        } else {
            // Good path, using interner IDs.
            self.topological_nodes(reverse)
                .into_iter()
                .map(|n| self.get_node(py, n))
                .collect()
        };

        Ok(PyTuple::new(py, nodes?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Yield op nodes in topological order.
    ///
    /// Allowed to pass in specific key to break ties in top order
    ///
    /// Args:
    ///     key (Callable): A callable which will take a DAGNode object and
    ///         return a string sort key. If not specified the qargs and
    ///         cargs of a node will be used for sorting.
    ///     reverse (bool): If True, yield op nodes in reverse topological order.
    ///
    /// Returns:
    ///     generator(DAGOpNode): op node in topological order
    #[pyo3(name = "topological_op_nodes", signature=(key=None, reverse=false))]
    fn py_topological_op_nodes(
        &self,
        py: Python,
        key: Option<Bound<PyAny>>,
        reverse: bool,
    ) -> PyResult<Py<PyIterator>> {
        let nodes: PyResult<Vec<_>> = if let Some(key) = key {
            self.topological_key_sort(py, &key, reverse)?
                .filter_map(|node| match self.dag.node_weight(node) {
                    Some(NodeType::Operation(_)) => Some(self.get_node(py, node)),
                    _ => None,
                })
                .collect()
        } else {
            // Good path, using interner IDs.
            self.topological_op_nodes(reverse)
                .map(|n| self.get_node(py, n))
                .collect()
        };

        Ok(PyTuple::new(py, nodes?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Replace a block of nodes with a single node.
    ///
    /// This is used to consolidate a block of DAGOpNodes into a single
    /// operation. A typical example is a block of gates being consolidated
    /// into a single ``UnitaryGate`` representing the unitary matrix of the
    /// block.
    ///
    /// Args:
    ///     node_block (List[DAGNode]): A list of dag nodes that represents the
    ///         node block to be replaced
    ///     op (qiskit.circuit.Operation): The operation to replace the
    ///         block with
    ///     wire_pos_map (Dict[Bit, int]): The dictionary mapping the bits to their positions in the
    ///         output ``qargs`` or ``cargs``. This is necessary to reconstruct the arg order over
    ///         multiple gates in the combined single op node.  If a :class:`.Bit` is not in the
    ///         dictionary, it will not be added to the args; this can be useful when dealing with
    ///         control-flow operations that have inherent bits in their ``condition`` or ``target``
    ///         fields.
    ///     cycle_check (bool): When set to True this method will check that
    ///         replacing the provided ``node_block`` with a single node
    ///         would introduce a cycle (which would invalidate the
    ///         ``DAGCircuit``) and will raise a ``DAGCircuitError`` if a cycle
    ///         would be introduced. This checking comes with a run time
    ///         penalty. If you can guarantee that your input ``node_block`` is
    ///         a contiguous block and won't introduce a cycle when it's
    ///         contracted to a single node, this can be set to ``False`` to
    ///         improve the runtime performance of this method.
    ///
    /// Raises:
    ///     DAGCircuitError: if ``cycle_check`` is set to ``True`` and replacing
    ///         the specified block introduces a cycle or if ``node_block`` is
    ///         empty.
    ///
    /// Returns:
    ///     DAGOpNode: The op node that replaces the block.
    #[pyo3(signature = (node_block, op, wire_pos_map, cycle_check=true))]
    fn replace_block_with_op(
        &mut self,
        py: Python,
        node_block: Vec<PyRef<DAGNode>>,
        op: Bound<PyAny>,
        wire_pos_map: &Bound<PyDict>,
        cycle_check: bool,
    ) -> PyResult<Py<PyAny>> {
        // If node block is empty return early
        if node_block.is_empty() {
            return Err(DAGCircuitError::new_err(
                "Can't replace an empty 'node_block'",
            ));
        }

        let mut qubit_pos_map: HashMap<Qubit, usize> = HashMap::new();
        let mut clbit_pos_map: HashMap<Clbit, usize> = HashMap::new();
        for (bit, index) in wire_pos_map.iter() {
            if bit.cast::<PyQubit>().is_ok() {
                qubit_pos_map.insert(
                    self.qubits.find(&bit.extract::<ShareableQubit>()?).unwrap(),
                    index.extract()?,
                );
            } else if bit.cast::<PyClbit>().is_ok() {
                clbit_pos_map.insert(
                    self.clbits.find(&bit.extract::<ShareableClbit>()?).unwrap(),
                    index.extract()?,
                );
            } else {
                return Err(DAGCircuitError::new_err(
                    "Wire map keys must be Qubit or Clbit instances.",
                ));
            }
        }

        let block_ids: Vec<_> = node_block.iter().map(|n| n.node.unwrap()).collect();
        let py_op = op.extract::<OperationFromPython<DAGCircuit>>()?;
        let params = self.take_parameter_blocks(py_op.params);
        let new_node = self.replace_block(
            &block_ids,
            py_op.operation,
            params,
            py_op.label.as_ref().map(|v| v.as_str()),
            cycle_check,
            &qubit_pos_map,
            &clbit_pos_map,
        )?;
        self.get_node(py, new_node)
    }

    /// Replace one node with dag.
    ///
    /// Args:
    ///     node (DAGOpNode): node to substitute
    ///     input_dag (DAGCircuit): circuit that will substitute the node
    ///     wires (list[Bit] | Dict[Bit, Bit]): gives an order for (qu)bits
    ///         in the input circuit. If a list, then the bits refer to those in the ``input_dag``,
    ///         and the order gets matched to the node wires by qargs first, then cargs, then
    ///         conditions.  If a dictionary, then a mapping of bits in the ``input_dag`` to those
    ///         that the ``node`` acts on.
    ///     propagate_condition (bool): DEPRECATED a legacy option that used
    ///         to control the behavior of handling control flow. It has no
    ///         effect anymore, left it for backwards compatibility. Will be
    ///         removed in Qiskit 3.0.
    ///
    /// Returns:
    ///     dict: maps node IDs from `input_dag` to their new node incarnations in `self`.
    ///
    /// Raises:
    ///     DAGCircuitError: if met with unexpected predecessor/successors
    #[pyo3(name = "substitute_node_with_dag", signature = (node, input_dag, wires=None, propagate_condition=None))]
    pub fn py_substitute_node_with_dag(
        &mut self,
        py: Python,
        node: &Bound<PyAny>,
        input_dag: &DAGCircuit,
        wires: Option<Bound<PyAny>>,
        propagate_condition: Option<bool>,
    ) -> PyResult<Py<PyDict>> {
        if propagate_condition.is_some() {
            imports::WARNINGS_WARN.get_bound(py).call1((
                intern!(
                    py,
                    concat!(
                        "The propagate_condition argument is deprecated as of Qiskit 2.0.0.",
                        "It has no effect anymore and will be removed in Qiskit 3.0.0.",
                    )
                ),
                py.get_type::<PyDeprecationWarning>(),
                2,
            ))?;
        }
        let node_index = match node.cast::<DAGOpNode>() {
            Ok(bound_node) => bound_node.borrow().as_ref().node.unwrap(),
            Err(_) => return Err(DAGCircuitError::new_err("expected node DAGOpNode")),
        };

        let node = match &self.dag[node_index] {
            NodeType::Operation(op) => op.clone(),
            _ => return Err(DAGCircuitError::new_err("expected node")),
        };

        // We need to lift any blocks registered with the `input_circuit` up to
        // the outer circuit.
        // TODO: since Python expects control flow to be owned by the instruction, it'd be nice
        //       if we had some special tracking to "deduplicate" Python instances that we've seen
        //       before.
        let block_map = input_dag
            .blocks()
            .items()
            .map(|(index, block)| (index, self.add_block(block.clone())))
            .collect();

        type WireMapsTuple = (
            HashMap<Qubit, Qubit>,
            HashMap<Clbit, Clbit>,
            HashMap<expr::Var, expr::Var>,
        );

        let build_wire_map = |wires: &Bound<PyList>| -> PyResult<WireMapsTuple> {
            let qargs_list: Vec<&ShareableQubit> = self
                .qubits
                .map_indices(self.qargs_interner.get(node.qubits))
                .collect();
            let mut cargs_list: Vec<&ShareableClbit> = self
                .clbits
                .map_indices(self.cargs_interner.get(node.clbits))
                .collect();
            let cargs_set: HashSet<&ShareableClbit> =
                HashSet::from_iter(cargs_list.iter().cloned());
            if self.may_have_additional_wires(&node) {
                let (add_cargs, _add_vars) = Python::attach(|py| self.additional_wires(py, &node))?;
                for wire in add_cargs {
                    let clbit = self.clbits.get(wire).unwrap();
                    if !cargs_set.contains(clbit) {
                        cargs_list.push(clbit);
                    }
                }
            }
            let qargs_len = qargs_list.len();
            let cargs_len = cargs_list.len();

            if qargs_len + cargs_len != wires.len() {
                return Err(DAGCircuitError::new_err(format!(
                    "bit mapping invalid: expected {}, got {}",
                    qargs_len + cargs_len,
                    wires.len()
                )));
            }
            let mut qubit_wire_map = HashMap::new();
            let mut clbit_wire_map = HashMap::new();
            let var_map = HashMap::new();
            for (index, wire) in wires.iter().enumerate() {
                if wire.cast::<PyQubit>().is_ok() {
                    if index >= qargs_len {
                        unreachable!()
                    }
                    let input_qubit: Qubit = input_dag
                        .qubits
                        .find(&wire.extract::<ShareableQubit>()?)
                        .unwrap();
                    let self_qubit: Qubit = self.qubits.find(qargs_list[index]).unwrap();
                    qubit_wire_map.insert(input_qubit, self_qubit);
                } else if wire.cast::<PyClbit>().is_ok() {
                    if index < qargs_len {
                        unreachable!()
                    }
                    clbit_wire_map.insert(
                        input_dag
                            .clbits
                            .find(&wire.extract::<ShareableClbit>()?)
                            .unwrap(),
                        self.clbits.find(cargs_list[index - qargs_len]).unwrap(),
                    );
                } else {
                    return Err(DAGCircuitError::new_err(
                        "`Var` nodes cannot be remapped during substitution",
                    ));
                }
            }
            Ok((qubit_wire_map, clbit_wire_map, var_map))
        };

        let (qubit_wire_map, clbit_wire_map, mut var_map): (
            HashMap<Qubit, Qubit>,
            HashMap<Clbit, Clbit>,
            HashMap<expr::Var, expr::Var>,
        ) = match wires {
            Some(wires) => match wires.cast::<PyDict>() {
                Ok(bound_wires) => {
                    let mut qubit_wire_map = HashMap::new();
                    let mut clbit_wire_map = HashMap::new();
                    let mut var_map = HashMap::new();
                    for (source_wire, target_wire) in bound_wires.iter() {
                        if source_wire.cast::<PyQubit>().is_ok() {
                            qubit_wire_map.insert(
                                input_dag
                                    .qubits
                                    .find(&source_wire.extract::<ShareableQubit>()?)
                                    .unwrap(),
                                self.qubits
                                    .find(&target_wire.extract::<ShareableQubit>()?)
                                    .unwrap(),
                            );
                        } else if source_wire.cast::<PyClbit>().is_ok() {
                            clbit_wire_map.insert(
                                input_dag
                                    .clbits
                                    .find(&source_wire.extract::<ShareableClbit>()?)
                                    .unwrap(),
                                self.clbits
                                    .find(&target_wire.extract::<ShareableClbit>()?)
                                    .unwrap(),
                            );
                        } else {
                            var_map.insert(source_wire.extract()?, target_wire.extract()?);
                        }
                    }
                    (qubit_wire_map, clbit_wire_map, var_map)
                }
                Err(_) => {
                    let wires: Bound<PyList> = match wires.cast::<PyList>() {
                        Ok(bound_list) => bound_list.clone(),
                        // If someone passes a sequence instead of an exact list (tuple is
                        // occasionally used) cast that to a list and then use it.
                        Err(_) => {
                            let raw_wires = imports::BUILTIN_LIST.get_bound(py).call1((wires,))?;
                            raw_wires.extract()?
                        }
                    };
                    build_wire_map(&wires)?
                }
            },
            None => {
                let raw_wires = input_dag.get_wires(py);
                let binding = raw_wires?;
                let wires = binding.bind(py);
                build_wire_map(wires)?
            }
        };

        let input_dag_var_set: HashSet<&expr::Var> =
            input_dag.vars_stretches.vars().objects().iter().collect();

        let node_vars = if self.may_have_additional_wires(&node) {
            let (_additional_clbits, additional_vars) = self.additional_wires(py, &node)?;
            let var_set: HashSet<&expr::Var> = additional_vars
                .into_iter()
                .map(|v| self.vars_stretches.vars().get(v).unwrap())
                .collect();
            if input_dag_var_set.difference(&var_set).count() > 0 {
                return Err(DAGCircuitError::new_err(format!(
                    "Cannot replace a node with a DAG with more variables. Variables in node: {:?}. Variables in dag: {:?}",
                    var_set, input_dag_var_set,
                )));
            }
            var_set
        } else {
            HashSet::default()
        };
        for contracted_var in node_vars.difference(&input_dag_var_set) {
            let pred = self
                .dag
                .edges_directed(node_index, Incoming)
                .find(|edge| {
                    if let Wire::Var(var) = edge.weight() {
                        *contracted_var == self.vars_stretches.vars().get(*var).unwrap()
                    } else {
                        false
                    }
                })
                .unwrap();
            let succ = self
                .dag
                .edges_directed(node_index, Outgoing)
                .find(|edge| {
                    if let Wire::Var(var) = edge.weight() {
                        *contracted_var == self.vars_stretches.vars().get(*var).unwrap()
                    } else {
                        false
                    }
                })
                .unwrap();
            self.dag.add_edge(
                pred.source(),
                succ.target(),
                Wire::Var(self.vars_stretches.vars().find(contracted_var).unwrap()),
            );
        }

        for var in input_dag_var_set {
            var_map.insert(var.clone(), var.clone());
        }

        // Now that we don't need these, we've got to drop them since they hold
        // references to variables owned by the DAG, which we'll need to mutate
        // when perform the substitution.
        drop(node_vars);

        // It doesn't make sense to try and propagate a condition from a control-flow op; a
        // replacement for the control-flow op should implement the operation completely.
        let node_map = self.substitute_node_with_dag(
            node_index,
            input_dag,
            Some(&qubit_wire_map),
            Some(&clbit_wire_map),
            Some(&var_map),
            Some(&block_map),
        )?;

        let out_dict = PyDict::new(py);
        for (old_index, new_index) in node_map {
            out_dict.set_item(old_index.index(), self.get_node(py, new_index)?)?;
        }
        Ok(out_dict.unbind())
    }

    /// Replace a DAGOpNode with a single operation. qargs, cargs and
    /// conditions for the new operation will be inferred from the node to be
    /// replaced. The new operation will be checked to match the shape of the
    /// replaced operation.
    ///
    /// Args:
    ///     node (DAGOpNode): Node to be replaced
    ///     op (qiskit.circuit.Operation): The :class:`qiskit.circuit.Operation`
    ///         instance to be added to the DAG
    ///     inplace (bool): Optional, default False. If True, existing DAG node
    ///         will be modified to include op. Otherwise, a new DAG node will
    ///         be used.
    ///     propagate_condition (bool): DEPRECATED a legacy option that used
    ///         to control the behavior of handling control flow. It has no
    ///         effect anymore, left it for backwards compatibility. Will be
    ///         removed in Qiskit 3.0.
    ///
    ///
    /// Returns:
    ///     DAGOpNode: the new node containing the added operation.
    ///
    /// Raises:
    ///     DAGCircuitError: If replacement operation was incompatible with
    ///     location of target node.
    #[pyo3(name = "substitute_node", signature = (node, op, inplace=false, propagate_condition=None))]
    pub fn py_substitute_node(
        &mut self,
        py: Python,
        node: &Bound<PyAny>,
        op: &Bound<PyAny>,
        inplace: bool,
        propagate_condition: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if propagate_condition.is_some() {
            imports::WARNINGS_WARN.get_bound(py).call1((
                intern!(
                    py,
                    concat!(
                        "The propagate_condition argument is deprecated as of Qiskit 2.0.0.",
                        "It has no effect anymore and will be removed in Qiskit 3.0.0.",
                    )
                ),
                py.get_type::<PyDeprecationWarning>(),
                2,
            ))?;
        }
        let mut node: PyRefMut<DAGOpNode> = match node.cast() {
            Ok(node) => node.borrow_mut(),
            Err(_) => return Err(DAGCircuitError::new_err("Only DAGOpNodes can be replaced.")),
        };
        let py = op.py();
        let node_index = node.as_ref().node.unwrap();
        self.substitute_node_with_py_op(node_index, op)?;
        if inplace {
            let temp = op.extract::<OperationFromPython<CircuitData>>()?;
            node.instruction.operation = temp.operation;
            node.instruction.params = temp.params;
            node.instruction.label = temp.label;
            #[cfg(feature = "cache_pygates")]
            {
                node.instruction.py_op = OnceLock::from(op.clone().unbind());
            }
            node.into_py_any(py)
        } else {
            self.get_node(py, node_index)
        }
    }

    /// Decompose the circuit into sets of qubits with no gates connecting them.
    ///
    /// Args:
    ///     remove_idle_qubits (bool): Flag denoting whether to remove idle qubits from
    ///         the separated circuits. If ``False``, each output circuit will contain the
    ///         same number of qubits as ``self``.
    ///
    /// Returns:
    ///     List[DAGCircuit]: The circuits resulting from separating ``self`` into sets
    ///         of disconnected qubits
    ///
    /// Each :class:`~.DAGCircuit` instance returned by this method will contain the same number of
    /// clbits as ``self``. The global phase information in ``self`` will not be maintained
    /// in the subcircuits returned by this method.
    #[pyo3(signature = (remove_idle_qubits=false, *, vars_mode=VarsMode::Alike))]
    fn separable_circuits(
        &self,
        py: Python,
        remove_idle_qubits: bool,
        vars_mode: VarsMode,
    ) -> PyResult<Py<PyList>> {
        let connected_components = rustworkx_core::connectivity::connected_components(&self.dag);
        let dags = PyList::empty(py);

        for comp_nodes in connected_components.iter() {
            let mut new_dag = self.copy_empty_like(vars_mode, BlocksMode::Drop);
            new_dag.global_phase = Param::Float(0.);

            // A map from nodes in the this DAGCircuit to nodes in the new dag. Used for adding edges
            let mut node_map: HashMap<NodeIndex, NodeIndex> =
                HashMap::with_capacity(comp_nodes.len());
            let mut block_map = BlockMapper::new();

            // Adding the nodes to the new dag
            let mut non_classical = false;
            for node in comp_nodes {
                match self.dag.node_weight(*node) {
                    Some(w) => match w {
                        NodeType::ClbitIn(b) => {
                            let clbit_in = new_dag.clbit_io_map[b.index()][0];
                            node_map.insert(*node, clbit_in);
                        }
                        NodeType::ClbitOut(b) => {
                            let clbit_out = new_dag.clbit_io_map[b.index()][1];
                            node_map.insert(*node, clbit_out);
                        }
                        NodeType::QubitIn(q) => {
                            let qbit_in = new_dag.qubit_io_map[q.index()][0];
                            node_map.insert(*node, qbit_in);
                            non_classical = true;
                        }
                        NodeType::QubitOut(q) => {
                            let qbit_out = new_dag.qubit_io_map[q.index()][1];
                            node_map.insert(*node, qbit_out);
                            non_classical = true;
                        }
                        NodeType::VarIn(v) => {
                            let var_in = new_dag.var_io_map[v.index()][0];
                            node_map.insert(*node, var_in);
                        }
                        NodeType::VarOut(v) => {
                            let var_out = new_dag.var_io_map[v.index()][1];
                            node_map.insert(*node, var_out);
                        }
                        NodeType::Operation(pi) => {
                            let pi = block_map
                                .map_instruction(pi, |b| new_dag.add_block(self.blocks[b].clone()));
                            new_dag.track_instruction(&pi);
                            let new_node = new_dag.dag.add_node(NodeType::Operation(pi));
                            node_map.insert(*node, new_node);
                            non_classical = true;
                        }
                    },
                    None => panic!("DAG node without payload!"),
                }
            }
            if !non_classical {
                continue;
            }
            let node_filter = |node: NodeIndex| -> bool { node_map.contains_key(&node) };

            let filtered = NodeFiltered(&self.dag, node_filter);

            // Remove the edges added by copy_empty_like (as idle wires) to avoid duplication
            new_dag.dag.clear_edges();
            for edge in filtered.edge_references() {
                let new_source = node_map[&edge.source()];
                let new_target = node_map[&edge.target()];
                new_dag.dag.add_edge(new_source, new_target, *edge.weight());
            }
            // Add back any edges for idle wires
            for (qubit, [in_node, out_node]) in new_dag
                .qubit_io_map
                .iter()
                .enumerate()
                .map(|(idx, indices)| (Qubit::new(idx), indices))
            {
                if new_dag.dag.edges(*in_node).next().is_none() {
                    new_dag
                        .dag
                        .add_edge(*in_node, *out_node, Wire::Qubit(qubit));
                }
            }
            for (clbit, [in_node, out_node]) in new_dag
                .clbit_io_map
                .iter()
                .enumerate()
                .map(|(idx, indices)| (Clbit::new(idx), indices))
            {
                if new_dag.dag.edges(*in_node).next().is_none() {
                    new_dag
                        .dag
                        .add_edge(*in_node, *out_node, Wire::Clbit(clbit));
                }
            }
            for (var_index, &[in_node, out_node]) in new_dag.var_io_map.iter().enumerate() {
                if new_dag.dag.edges(in_node).next().is_none() {
                    new_dag
                        .dag
                        .add_edge(in_node, out_node, Wire::Var(Var::new(var_index)));
                }
            }
            if remove_idle_qubits {
                let idle_wires: Vec<Bound<PyAny>> = new_dag
                    .idle_wires(py, None)?
                    .into_bound(py)
                    .map(|q| q.unwrap())
                    .filter(|e| e.cast::<PyQubit>().is_ok())
                    .collect();

                let qubits = PyTuple::new(py, idle_wires)?;
                let bit_iter = match self
                    .qubits
                    .map_objects(qubits.iter().map(|x| x.extract().unwrap()))
                {
                    Ok(bit_iter) => bit_iter,
                    Err(_) => {
                        return Err(DAGCircuitError::new_err(format!(
                            "qubits not in circuit: {qubits:?}"
                        )));
                    }
                };
                new_dag.remove_qubits(bit_iter)?; // TODO: this does not really work, some issue with remove_qubits itself
            }

            dags.append(pyo3::Py::new(py, new_dag)?)?;
        }

        Ok(dags.unbind())
    }

    /// Swap connected nodes e.g. due to commutation.
    ///
    /// Args:
    ///     node1 (OpNode): predecessor node
    ///     node2 (OpNode): successor node
    ///
    /// Raises:
    ///     DAGCircuitError: if either node is not an OpNode or nodes are not connected
    fn swap_nodes(&mut self, node1: &DAGNode, node2: &DAGNode) -> PyResult<()> {
        let node1 = node1.node.unwrap();
        let node2 = node2.node.unwrap();

        // Check that both nodes correspond to operations
        if !matches!(self.dag.node_weight(node1).unwrap(), NodeType::Operation(_))
            || !matches!(self.dag.node_weight(node2).unwrap(), NodeType::Operation(_))
        {
            return Err(DAGCircuitError::new_err(
                "Nodes to swap are not both DAGOpNodes",
            ));
        }

        // Gather all wires connecting node1 and node2.
        // This functionality was extracted from rustworkx's 'get_edge_data'
        let wires: Vec<Wire> = self
            .dag
            .edges(node1)
            .filter(|edge| edge.target() == node2)
            .map(|edge| *edge.weight())
            .collect();

        if wires.is_empty() {
            return Err(DAGCircuitError::new_err(
                "Attempt to swap unconnected nodes",
            ));
        };

        // Closure that finds the first parent/child node connected to a reference node by given wire
        // and returns relevant edge information depending on the specified direction:
        //  - Incoming -> parent -> outputs (parent_edge_id, parent_source_node_id)
        //  - Outgoing -> child -> outputs (child_edge_id, child_target_node_id)
        // This functionality was inspired in rustworkx's 'find_predecessors_by_edge' and 'find_successors_by_edge'.
        let directed_edge_for_wire = |node: NodeIndex, direction: Direction, wire: Wire| {
            for edge in self.dag.edges_directed(node, direction) {
                if wire == *edge.weight() {
                    match direction {
                        Incoming => return Some((edge.id(), edge.source())),
                        Outgoing => return Some((edge.id(), edge.target())),
                    }
                }
            }
            None
        };

        // Vector that contains a tuple of (wire, edge_info, parent_info, child_info) per wire in wires
        let relevant_edges = wires
            .iter()
            .rev()
            .map(|&wire| {
                (
                    wire,
                    directed_edge_for_wire(node1, Outgoing, wire).unwrap(),
                    directed_edge_for_wire(node1, Incoming, wire).unwrap(),
                    directed_edge_for_wire(node2, Outgoing, wire).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        // Iterate over relevant edges and modify self.dag
        for (wire, (node1_to_node2, _), (parent_to_node1, parent), (node2_to_child, child)) in
            relevant_edges
        {
            self.dag.remove_edge(parent_to_node1);
            self.dag.add_edge(parent, node2, wire);
            self.dag.remove_edge(node1_to_node2);
            self.dag.add_edge(node2, node1, wire);
            self.dag.remove_edge(node2_to_child);
            self.dag.add_edge(node1, child, wire);
        }
        Ok(())
    }

    /// Get the node in the dag.
    ///
    /// Args:
    ///     node_id(int): Node identifier.
    ///
    /// Returns:
    ///     node: the node.
    fn node(&self, py: Python, node_id: isize) -> PyResult<Py<PyAny>> {
        self.get_node(py, NodeIndex::new(node_id as usize))
    }

    /// Iterator for node values.
    ///
    /// Yield:
    ///     node: the node.
    fn nodes(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let result: PyResult<Vec<_>> = self
            .dag
            .node_references()
            .map(|(node, weight)| self.unpack_into(py, node, weight))
            .collect();
        let tup = PyTuple::new(py, result?)?;
        Ok(tup.into_any().try_iter().unwrap().unbind())
    }

    /// Iterator for edge values with source and destination node.
    ///
    /// This works by returning the outgoing edges from the specified nodes. If
    /// no nodes are specified all edges from the graph are returned.
    ///
    /// Args:
    ///     nodes(DAGOpNode, DAGInNode, or DAGOutNode|list(DAGOpNode, DAGInNode, or DAGOutNode):
    ///         Either a list of nodes or a single input node. If none is specified,
    ///         all edges are returned from the graph.
    ///
    /// Yield:
    ///     edge: the edge as a tuple with the format
    ///         (source node, destination node, edge wire)
    #[pyo3(signature=(nodes=None))]
    fn edges(&self, py: Python, nodes: Option<Bound<PyAny>>) -> PyResult<Py<PyIterator>> {
        let get_node_index = |obj: &Bound<PyAny>| -> PyResult<NodeIndex> {
            Ok(obj.cast::<DAGNode>()?.borrow().node.unwrap())
        };

        let actual_nodes: Vec<_> = match nodes {
            None => self.dag.node_indices().collect(),
            Some(nodes) => {
                let mut out = Vec::new();
                if let Ok(node) = get_node_index(&nodes) {
                    out.push(node);
                } else {
                    for node in nodes.try_iter()? {
                        out.push(get_node_index(&node?)?);
                    }
                }
                out
            }
        };

        let mut edges = Vec::new();
        for node in actual_nodes {
            for edge in self.dag.edges_directed(node, Outgoing) {
                edges.push((
                    self.get_node(py, edge.source())?,
                    self.get_node(py, edge.target())?,
                    match edge.weight() {
                        Wire::Qubit(qubit) => {
                            self.qubits.get(*qubit).unwrap().into_bound_py_any(py)?
                        }
                        Wire::Clbit(clbit) => {
                            self.clbits.get(*clbit).unwrap().into_bound_py_any(py)?
                        }
                        Wire::Var(var) => self
                            .vars_stretches
                            .vars()
                            .get(*var)
                            .unwrap()
                            .clone()
                            .into_bound_py_any(py)?,
                    },
                ))
            }
        }

        Ok(PyTuple::new(py, edges)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Get the list of "op" nodes in the dag.
    ///
    /// Args:
    ///     op (Type): :class:`qiskit.circuit.Operation` subclass op nodes to
    ///         return. If None, return all op nodes.
    ///     include_directives (bool): include `barrier`, `snapshot` etc.
    ///
    /// Returns:
    ///     list[DAGOpNode]: the list of dag nodes containing the given op.
    #[pyo3(name= "op_nodes", signature=(op=None, include_directives=true))]
    fn py_op_nodes(
        &self,
        py: Python,
        op: Option<&Bound<PyType>>,
        include_directives: bool,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let mut nodes = Vec::new();
        let filter_is_nonstandard = if let Some(op) = op {
            op.getattr(intern!(py, "_standard_gate")).ok().is_none()
        } else {
            true
        };
        for (node, weight) in self.dag.node_references() {
            if let NodeType::Operation(packed) = &weight {
                if !include_directives && packed.op.directive() {
                    continue;
                }
                if let Some(op_type) = op {
                    // This middle catch is to avoid Python-space operation creation for most uses of
                    // `op`; we're usually just looking for control-flow ops, and standard gates
                    // aren't control-flow ops.
                    if !(filter_is_nonstandard && packed.op.try_standard_gate().is_some())
                        && packed.op.py_op_is_instance(op_type)?
                    {
                        nodes.push(self.unpack_into(py, node, weight)?);
                    }
                } else {
                    nodes.push(self.unpack_into(py, node, weight)?);
                }
            }
        }
        Ok(nodes)
    }

    /// Get a list of "op" nodes in the dag that contain control flow instructions.
    ///
    /// Returns:
    ///     list[DAGOpNode]: The list of dag nodes containing control flow ops.
    fn control_flow_op_nodes(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        if !self.has_control_flow() {
            return Ok(vec![]);
        }
        self.dag
            .node_references()
            .filter_map(|(node_index, node_type)| match node_type {
                NodeType::Operation(node) => {
                    if node.op.try_control_flow().is_some_and(|control_flow| {
                        !matches!(
                            control_flow.control_flow,
                            ControlFlow::BreakLoop | ControlFlow::ContinueLoop
                        )
                    }) {
                        Some(self.unpack_into(py, node_index, node_type))
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect()
    }

    /// Get the list of gate nodes in the dag.
    ///
    /// Returns:
    ///     list[DAGOpNode]: the list of DAGOpNodes that represent gates.
    fn gate_nodes(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        self.dag
            .node_references()
            .filter_map(|(node, weight)| match weight {
                NodeType::Operation(packed) => match packed.op.view() {
                    OperationRef::PyCustom(PyInstruction {
                        kind: PyOpKind::Gate,
                        ..
                    })
                    | OperationRef::StandardGate(_) => Some(self.unpack_into(py, node, weight)),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    /// Get the set of "op" nodes with the given name.
    #[pyo3(signature = (*names))]
    fn named_nodes(&self, py: Python<'_>, names: &Bound<PyTuple>) -> PyResult<Vec<Py<PyAny>>> {
        let mut names_set: HashSet<String> = HashSet::with_capacity(names.len());
        for name_obj in names.iter() {
            names_set.insert(name_obj.extract::<String>()?);
        }
        let mut result: Vec<Py<PyAny>> = Vec::new();
        for (id, weight) in self.dag.node_references() {
            if let NodeType::Operation(packed) = weight {
                if names_set.contains(packed.op.name()) {
                    result.push(self.unpack_into(py, id, weight)?);
                }
            }
        }
        Ok(result)
    }

    /// Get list of 2 qubit operations. Ignore directives like snapshot and barrier.
    #[pyo3(name = "two_qubit_ops")]
    pub fn py_two_qubit_ops(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        self.two_qubit_ops()
            .map(|(index, _)| self.unpack_into(py, index, &self.dag[index]))
            .collect()
    }

    /// Get list of 3+ qubit operations. Ignore directives like snapshot and barrier.
    fn multi_qubit_ops(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let mut nodes = Vec::new();
        for (node, weight) in self.dag.node_references() {
            if let NodeType::Operation(packed) = weight {
                if packed.op.directive() {
                    continue;
                }

                let qargs = self.qargs_interner.get(packed.qubits);
                if qargs.len() >= 3 {
                    nodes.push(self.unpack_into(py, node, weight)?);
                }
            }
        }
        Ok(nodes)
    }

    /// Returns the longest path in the dag as a list of DAGOpNodes, DAGInNodes, and DAGOutNodes.
    fn longest_path(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let weight_fn = |_| -> Result<usize, Infallible> { Ok(1) };
        match rustworkx_core::dag_algo::longest_path(&self.dag, weight_fn).unwrap() {
            Some(res) => res.0,
            None => panic!("not a DAG"),
        }
        .into_iter()
        .map(|node_index| self.get_node(py, node_index))
        .collect()
    }

    /// Returns iterator of the successors of a node as :class:`.DAGOpNode`\ s and
    /// :class:`.DAGOutNode`\ s.
    #[pyo3(name = "successors")]
    fn py_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let successors: PyResult<Vec<_>> = self
            .successors(node.node.unwrap())
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new(py, successors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the predecessors of a node as :class:`.DAGOpNode`\ s and
    /// :class:`.DAGInNode`\ s.
    #[pyo3(name = "predecessors")]
    fn py_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .predecessors(node.node.unwrap())
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of "op" successors of a node in the dag.
    fn op_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Outgoing)
            .unique()
            .filter_map(|i| match self.dag[i] {
                NodeType::Operation(_) => Some(self.get_node(py, i)),
                _ => None,
            })
            .collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns the iterator of "op" predecessors of a node in the dag.
    fn op_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Incoming)
            .unique()
            .filter_map(|i| match self.dag[i] {
                NodeType::Operation(_) => Some(self.get_node(py, i)),
                _ => None,
            })
            .collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Checks if a second node is in the successors of node.
    fn is_successor(&self, node: &DAGNode, node_succ: &DAGNode) -> bool {
        self.dag
            .find_edge(node.node.unwrap(), node_succ.node.unwrap())
            .is_some()
    }

    /// Checks if a second node is in the predecessors of node.
    fn is_predecessor(&self, node: &DAGNode, node_pred: &DAGNode) -> bool {
        self.dag
            .find_edge(node_pred.node.unwrap(), node.node.unwrap())
            .is_some()
    }

    /// Returns iterator of the predecessors of a node that are
    /// connected by a quantum edge as DAGOpNodes and DAGInNodes.
    #[pyo3(name = "quantum_predecessors")]
    fn py_quantum_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .quantum_predecessors(node.node.unwrap())
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the successors of a node that are
    /// connected by a quantum edge as DAGOpNodes and DAGOutNodes.
    #[pyo3(name = "quantum_successors")]
    fn py_quantum_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let successors: PyResult<Vec<_>> = self
            .quantum_successors(node.node.unwrap())
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new(py, successors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the predecessors of a node that are
    /// connected by a classical edge as DAGOpNodes and DAGInNodes.
    fn classical_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let edges = self.dag.edges_directed(node.node.unwrap(), Incoming);
        let filtered = edges.filter_map(|e| match e.weight() {
            Wire::Qubit(_) => None,
            _ => Some(e.source()),
        });
        let predecessors: PyResult<Vec<_>> =
            filtered.unique().map(|i| self.get_node(py, i)).collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Returns set of the ancestors of a node as :class:`.DAGOpNode`\ s and :class:`.DAGInNode`\ s.
    ///
    /// The ancestors are the set of all nodes that can reach the target node. Whereas the
    /// :meth:`.DAGCircuit.predecessors` only contains the immediate predecessors, the ancestors
    /// recursively contain the predecessors of each predecessor.
    #[pyo3(name = "ancestors")]
    fn py_ancestors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PySet>> {
        let ancestors: PyResult<Vec<Py<PyAny>>> = self
            .ancestors(node.node.unwrap())
            .map(|node| self.get_node(py, node))
            .collect();
        Ok(PySet::new(py, &ancestors?)?.unbind())
    }

    /// Returns set of the descendants of a node as :class:`.DAGOpNode`\ s and :class:`.DAGOutNode`\ s.
    ///
    /// The descendants are the set of all nodes that can be reached from the target node. In
    /// comparison, :meth:`.DAGCircuit.successors` is an iterator over the immediate successors,
    /// whereas this method contains all the successors' successors.
    #[pyo3(name = "descendants")]
    fn py_descendants(&self, py: Python, node: &DAGNode) -> PyResult<Py<PySet>> {
        let descendants: PyResult<Vec<Py<PyAny>>> = self
            .descendants(node.node.unwrap())
            .map(|node| self.get_node(py, node))
            .collect();
        Ok(PySet::new(py, &descendants?)?.unbind())
    }

    /// Returns an iterator of tuples of ``(DAGNode, [DAGNodes])`` where the ``DAGNode`` is the
    /// current node and ``[DAGNodes]`` is a list of the successors in BFS order.
    #[pyo3(name = "bfs_successors")]
    fn py_bfs_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        type PyIteratorVec = Vec<(Py<PyAny>, Vec<Py<PyAny>>)>;

        let successor_index: PyResult<PyIteratorVec> = self
            .bfs_successors(node.node.unwrap())
            .map(|(node, nodes)| -> PyResult<(Py<PyAny>, Vec<Py<PyAny>>)> {
                Ok((
                    self.get_node(py, node)?,
                    nodes
                        .iter()
                        .map(|sub_node| self.get_node(py, *sub_node))
                        .collect::<PyResult<Vec<_>>>()?,
                ))
            })
            .collect();
        Ok(PyList::new(py, successor_index?)?
            .into_any()
            .try_iter()?
            .unbind())
    }

    /// Returns iterator of the successors of a node that are
    /// connected by a classical edge as DAGOpNodes and DAGOutNodes.
    fn classical_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let edges = self.dag.edges_directed(node.node.unwrap(), Outgoing);
        let filtered = edges.filter_map(|e| match e.weight() {
            Wire::Qubit(_) => None,
            _ => Some(e.target()),
        });
        let predecessors: PyResult<Vec<_>> =
            filtered.unique().map(|i| self.get_node(py, i)).collect();
        Ok(PyTuple::new(py, predecessors?)?
            .into_any()
            .try_iter()
            .unwrap()
            .unbind())
    }

    /// Remove an operation node n.
    ///
    /// Add edges from predecessors to successors.
    #[pyo3(name = "remove_op_node")]
    fn py_remove_op_node(&mut self, node: &Bound<PyAny>) -> PyResult<()> {
        let node: PyRef<DAGOpNode> = match node.cast::<DAGOpNode>() {
            Ok(node) => node.borrow(),
            Err(_) => return Err(DAGCircuitError::new_err("Node not an DAGOpNode")),
        };
        let index = node.as_ref().node.unwrap();
        if self.dag.node_weight(index).is_none() {
            return Err(DAGCircuitError::new_err("Node not in DAG"));
        }
        self.remove_op_node(index);
        Ok(())
    }

    /// Remove all of the ancestor operation nodes of node.
    fn remove_ancestors_of(&mut self, node: &DAGNode) {
        let ancestors: Vec<_> = core_ancestors(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        for a in ancestors {
            self.dag.remove_node(a);
        }
    }

    /// Remove all of the descendant operation nodes of node.
    fn remove_descendants_of(&mut self, node: &DAGNode) {
        let descendants: Vec<_> = core_descendants(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        for d in descendants {
            self.dag.remove_node(d);
        }
    }

    /// Remove all of the non-ancestors operation nodes of node.
    fn remove_nonancestors_of(&mut self, node: &DAGNode) {
        let ancestors: HashSet<_> = core_ancestors(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        let non_ancestors: Vec<_> = self
            .dag
            .node_indices()
            .filter(|node_id| !ancestors.contains(node_id))
            .collect();
        for na in non_ancestors {
            self.dag.remove_node(na);
        }
    }

    /// Remove all of the non-descendants operation nodes of node.
    fn remove_nondescendants_of(&mut self, node: &DAGNode) {
        let descendants: HashSet<_> = core_descendants(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        let non_descendants: Vec<_> = self
            .dag
            .node_indices()
            .filter(|node_id| !descendants.contains(node_id))
            .collect();
        for nd in non_descendants {
            self.dag.remove_node(nd);
        }
    }

    /// Return a list of op nodes in the first layer of this dag.
    #[pyo3(name = "front_layer")]
    fn py_front_layer(&self, py: Python) -> PyResult<Py<PyList>> {
        let native_front_layer = self.front_layer();
        let front_layer_list = PyList::empty(py);
        for node in native_front_layer {
            front_layer_list.append(self.get_node(py, node)?)?;
        }
        Ok(front_layer_list.into())
    }

    /// Yield a shallow view on a layer of this DAGCircuit for all d layers of this circuit.
    ///
    /// A layer is a circuit whose gates act on disjoint qubits, i.e.,
    /// a layer has depth 1. The total number of layers equals the
    /// circuit depth d. The layers are indexed from 0 to d-1 with the
    /// earliest layer at index 0. The layers are constructed using a
    /// greedy algorithm. Each returned layer is a dict containing
    /// {"graph": circuit graph, "partition": list of qubit lists}.
    ///
    /// The returned layer contains new (but semantically equivalent) DAGOpNodes, DAGInNodes,
    /// and DAGOutNodes. These are not the same as nodes of the original dag, but are equivalent
    /// via DAGNode.semantic_eq(node1, node2).
    ///
    /// TODO: Gates that use the same cbits will end up in different
    /// layers as this is currently implemented. This may not be
    /// the desired behavior.
    #[pyo3(signature = (*, vars_mode=VarsMode::Captures))]
    fn layers(&self, py: Python, vars_mode: VarsMode) -> PyResult<Py<PyIterator>> {
        let layer_list = PyList::empty(py);
        let mut graph_layers = self.multigraph_layers();
        if graph_layers.next().is_none() {
            return Ok(PyIterator::from_object(&layer_list)?.into());
        }

        for graph_layer in graph_layers {
            let layer_dict = PyDict::new(py);
            // Sort to make sure they are in the order they were added to the original DAG
            // It has to be done by node_id as graph_layer is just a list of nodes
            // with no implied topology
            // Drawing tools rely on _node_id to infer order of node creation
            // so we need this to be preserved by layers()
            // Get the op nodes from the layer, removing any input and output nodes.
            let mut op_nodes: Vec<(&PackedInstruction, &NodeIndex)> = graph_layer
                .iter()
                .filter_map(|node| self.dag.node_weight(*node).map(|dag_node| (dag_node, node)))
                .filter_map(|(node, index)| match node {
                    NodeType::Operation(oper) => Some((oper, index)),
                    _ => None,
                })
                .collect();
            op_nodes.sort_by_key(|(_, node_index)| **node_index);

            if op_nodes.is_empty() {
                return Ok(PyIterator::from_object(&layer_list)?.into());
            }

            let mut new_layer = self.copy_empty_like(vars_mode, BlocksMode::Drop);
            let mut block_map = BlockMapper::new();
            let data: Vec<_> = op_nodes
                .iter()
                .map(|(inst, _)| {
                    block_map.map_instruction(inst, |b| new_layer.add_block(self.blocks[b].clone()))
                })
                .collect();
            new_layer.extend(data)?;

            let support_iter = new_layer.op_nodes(false).map(|(_, instruction)| {
                PyTuple::new(
                    py,
                    new_layer
                        .qubits
                        .map_indices(new_layer.qargs_interner.get(instruction.qubits)),
                )
                .unwrap()
            });
            let support_list = PyList::empty(py);
            for support_qarg in support_iter {
                support_list.append(support_qarg)?;
            }
            layer_dict.set_item("graph", new_layer)?;
            layer_dict.set_item("partition", support_list)?;
            layer_list.append(layer_dict)?;
        }
        Ok(layer_list.into_any().try_iter()?.into())
    }

    /// Yield a layer for all gates of this circuit.
    ///
    /// A serial layer is a circuit with one gate. The layers have the
    /// same structure as in layers().
    #[pyo3(signature = (*, vars_mode=VarsMode::Captures))]
    fn serial_layers(&self, py: Python, vars_mode: VarsMode) -> PyResult<Py<PyIterator>> {
        let layer_list = PyList::empty(py);
        for next_node in self.topological_op_nodes(false) {
            let retrieved_node: &PackedInstruction = match self.dag.node_weight(next_node) {
                Some(NodeType::Operation(node)) => node,
                _ => unreachable!("A non-operation node was obtained from topological_op_nodes."),
            };
            let mut new_layer = self.copy_empty_like(vars_mode, BlocksMode::Drop);
            let mut block_map = BlockMapper::new();

            // Save the support of the operation we add to the layer
            let support_list = PyList::empty(py);
            let qubits = PyTuple::new(
                py,
                self.qargs_interner
                    .get(retrieved_node.qubits)
                    .iter()
                    .map(|qubit| self.qubits.get(*qubit)),
            )?
            .unbind();
            let inst = block_map.map_instruction(retrieved_node, |b| {
                new_layer.add_block(self.blocks[b].clone())
            });
            new_layer.push_back(inst)?;

            if !retrieved_node.op.directive() {
                support_list.append(qubits)?;
            }

            let layer_dict = [
                ("graph", new_layer.into_py_any(py)?),
                ("partition", support_list.into_any().unbind()),
            ]
            .into_py_dict(py)?;
            layer_list.append(layer_dict)?;
        }

        Ok(layer_list.into_any().try_iter()?.into())
    }

    /// Yield layers of the multigraph.
    #[pyo3(name = "multigraph_layers")]
    fn py_multigraph_layers(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let graph_layers = self.multigraph_layers().map(|layer| -> Vec<Py<PyAny>> {
            layer
                .into_iter()
                .filter_map(|index| self.get_node(py, index).ok())
                .collect()
        });
        let list: Bound<PyList> = PyList::new(py, graph_layers.collect::<Vec<Vec<Py<PyAny>>>>())?;
        Ok(PyIterator::from_object(&list)?.unbind())
    }

    /// Return a set of non-conditional runs of "op" nodes with the given names.
    ///
    /// For example, "... h q[0]; cx q[0],q[1]; cx q[0],q[1]; h q[1]; .."
    /// would produce the tuple of cx nodes as an element of the set returned
    /// from a call to collect_runs(["cx"]). If instead the cx nodes were
    /// "cx q[0],q[1]; cx q[1],q[0];", the method would still return the
    /// pair in a tuple. The namelist can contain names that are not
    /// in the circuit's basis.
    ///
    /// Nodes must have only one successor to continue the run.
    #[pyo3(name = "collect_runs")]
    fn py_collect_runs(&self, py: Python, namelist: &Bound<PyList>) -> PyResult<Py<PySet>> {
        let mut name_list_set = HashSet::with_capacity(namelist.len());
        for name in namelist.iter() {
            name_list_set.insert(name.extract::<String>()?);
        }

        let out_set = PySet::empty(py)?;

        for run in self.collect_runs(name_list_set) {
            let run_tuple = PyTuple::new(
                py,
                run.into_iter()
                    .map(|node_index| self.get_node(py, node_index).unwrap()),
            )?;
            out_set.add(run_tuple)?;
        }
        Ok(out_set.unbind())
    }

    /// Return a set of non-conditional runs of 1q "op" nodes.
    #[pyo3(name = "collect_1q_runs")]
    fn py_collect_1q_runs(&self, py: Python) -> PyResult<Py<PyList>> {
        match self.collect_1q_runs() {
            Some(runs) => {
                let runs_iter = runs.map(|node_indices| {
                    PyList::new(
                        py,
                        node_indices
                            .into_iter()
                            .map(|node_index| self.get_node(py, node_index).unwrap()),
                    )
                    .unwrap()
                    .unbind()
                });
                let out_list = PyList::empty(py);
                for run_list in runs_iter {
                    out_list.append(run_list)?;
                }
                Ok(out_list.unbind())
            }
            None => Err(PyRuntimeError::new_err(
                "Invalid DAGCircuit, cycle encountered",
            )),
        }
    }

    /// Return a set of non-conditional runs of 2q "op" nodes.
    #[pyo3(name = "collect_2q_runs")]
    fn py_collect_2q_runs(&self, py: Python) -> PyResult<Py<PyList>> {
        match self.collect_2q_runs() {
            Some(runs) => {
                let runs_iter = runs.into_iter().map(|node_indices| {
                    PyList::new(
                        py,
                        node_indices
                            .into_iter()
                            .map(|node_index| self.get_node(py, node_index).unwrap()),
                    )
                    .unwrap()
                    .unbind()
                });
                let out_list = PyList::empty(py);
                for run_list in runs_iter {
                    out_list.append(run_list)?;
                }
                Ok(out_list.unbind())
            }
            None => Err(PyRuntimeError::new_err(
                "Invalid DAGCircuit, cycle encountered",
            )),
        }
    }

    /// Iterator for nodes that affect a given wire.
    ///
    /// Args:
    ///     wire (Bit): the wire to be looked at.
    ///     only_ops (bool): True if only the ops nodes are wanted;
    ///                 otherwise, all nodes are returned.
    /// Yield:
    ///      Iterator: the successive nodes on the given wire
    ///
    /// Raises:
    ///     DAGCircuitError: if the given wire doesn't exist in the DAG
    #[pyo3(name = "nodes_on_wire", signature = (wire, only_ops=false))]
    fn py_nodes_on_wire(
        &self,
        py: Python,
        wire: &Bound<PyAny>,
        only_ops: bool,
    ) -> PyResult<Py<PyIterator>> {
        let wire = if wire.cast::<PyQubit>().is_ok() {
            let wire = wire.extract::<ShareableQubit>()?;
            self.qubits.find(&wire).map(Wire::Qubit)
        } else if wire.cast::<PyClbit>().is_ok() {
            let wire = wire.extract::<ShareableClbit>()?;
            self.clbits.find(&wire).map(Wire::Clbit)
        } else {
            let wire = wire.extract::<expr::Var>()?;
            self.vars_stretches.vars().find(&wire).map(Wire::Var)
        }
        .ok_or_else(|| {
            DAGCircuitError::new_err(format!(
                "The given wire {wire:?} is not present in the circuit"
            ))
        })?;

        let nodes = self
            .nodes_on_wire(wire)
            .filter(|node| !only_ops || matches!(self.dag[*node], NodeType::Operation(_)))
            .map(|n| self.get_node(py, n))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, nodes)?.into_any().try_iter()?.unbind())
    }

    /// Count the occurrences of operation names.
    ///
    /// Args:
    ///     recurse: if ``True`` (default), then recurse into control-flow operations.  In all
    ///         cases, this counts only the number of times the operation appears in any possible
    ///         block; both branches of if-elses are counted, and for- and while-loop blocks are
    ///         only counted once.
    ///
    /// Returns:
    ///     Mapping[str, int]: a mapping of operation names to the number of times it appears.
    #[pyo3(name = "count_ops", signature = (*, recurse=true))]
    fn py_count_ops(&self, py: Python, recurse: bool) -> PyResult<Py<PyAny>> {
        self.count_ops(recurse)?.into_py_any(py)
    }

    /// Count the occurrences of operation names on the longest path.
    ///
    /// Returns a dictionary of counts keyed on the operation name.
    fn count_ops_longest_path(&self) -> PyResult<HashMap<&str, usize>> {
        if self.dag.node_count() == 0 {
            return Ok(HashMap::new());
        }
        let weight_fn = |_| -> Result<usize, Infallible> { Ok(1) };
        let longest_path =
            match rustworkx_core::dag_algo::longest_path(&self.dag, weight_fn).unwrap() {
                Some(res) => res.0,
                None => panic!("not a DAG"),
            };
        // Allocate for worst case where all operations are unique
        let mut op_counts: HashMap<&str, usize> = HashMap::with_capacity(longest_path.len() - 2);
        for node_index in &longest_path[1..longest_path.len() - 1] {
            if let NodeType::Operation(ref packed) = self.dag[*node_index] {
                let name = packed.op.name();
                op_counts
                    .entry(name)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }
        Ok(op_counts)
    }

    /// Returns causal cone of a qubit.
    ///
    /// A qubit's causal cone is the set of qubits that can influence the output of that
    /// qubit through interactions, whether through multi-qubit gates or operations. Knowing
    /// the causal cone of a qubit can be useful when debugging faulty circuits, as it can
    /// help identify which wire(s) may be causing the problem.
    ///
    /// This method does not consider any classical data dependency in the ``DAGCircuit``,
    /// classical bit wires are ignored for the purposes of building the causal cone.
    ///
    /// Args:
    ///     qubit (~qiskit.circuit.Qubit): The output qubit for which we want to find the causal cone.
    ///
    /// Returns:
    ///     Set[~qiskit.circuit.Qubit]: The set of qubits whose interactions affect ``qubit``.
    fn quantum_causal_cone(&self, py: Python, qubit: &Bound<PyAny>) -> PyResult<Py<PySet>> {
        // Retrieve the output node from the qubit
        let qubit_nat: ShareableQubit = qubit.extract()?;
        let output_qubit = self.qubits.find(&qubit_nat).ok_or_else(|| {
            DAGCircuitError::new_err(format!(
                "The given qubit {qubit:?} is not present in the circuit"
            ))
        })?;
        let output_node_index = self
            .qubit_io_map
            .get(output_qubit.index())
            .map(|x| x[1])
            .ok_or_else(|| {
                DAGCircuitError::new_err(format!(
                    "The given qubit {qubit:?} is not present in qubit_output_map"
                ))
            })?;

        let mut qubits_in_cone: HashSet<&Qubit> = HashSet::from([&output_qubit]);
        let mut queue: VecDeque<NodeIndex> = self.quantum_predecessors(output_node_index).collect();

        // The processed_non_directive_nodes stores the set of processed non-directive nodes.
        // This is an optimization to avoid considering the same non-directive node multiple
        // times when reached from different paths.
        // The directive nodes (such as barriers or measures) are trickier since when processing
        // them we only add their predecessors that intersect qubits_in_cone. Hence, directive
        // nodes have to be considered multiple times.
        let mut processed_non_directive_nodes: HashSet<NodeIndex> = HashSet::new();

        while !queue.is_empty() {
            let cur_index = queue.pop_front().unwrap();

            if let NodeType::Operation(packed) = self.dag.node_weight(cur_index).unwrap() {
                if !packed.op.directive() {
                    // If the operation is not a directive (in particular not a barrier nor a measure),
                    // we do not do anything if it was already processed. Otherwise, we add its qubits
                    // to qubits_in_cone, and append its predecessors to queue.
                    if processed_non_directive_nodes.contains(&cur_index) {
                        continue;
                    }
                    qubits_in_cone.extend(self.qargs_interner.get(packed.qubits));
                    processed_non_directive_nodes.insert(cur_index);

                    for pred_index in self.quantum_predecessors(cur_index) {
                        if let NodeType::Operation(_pred_packed) =
                            self.dag.node_weight(pred_index).unwrap()
                        {
                            queue.push_back(pred_index);
                        }
                    }
                } else {
                    // Directives (such as barriers and measures) may be defined over all the qubits,
                    // yet not all of these qubits should be considered in the causal cone. So we
                    // only add those predecessors that have qubits in common with qubits_in_cone.
                    for pred_index in self.quantum_predecessors(cur_index) {
                        if let NodeType::Operation(pred_packed) =
                            self.dag.node_weight(pred_index).unwrap()
                        {
                            if self
                                .qargs_interner
                                .get(pred_packed.qubits)
                                .iter()
                                .any(|x| qubits_in_cone.contains(x))
                            {
                                queue.push_back(pred_index);
                            }
                        }
                    }
                }
            }
        }

        let qubits_in_cone_vec: Vec<_> = qubits_in_cone.iter().map(|&&qubit| qubit).collect();
        let elements = self.qubits.map_indices(&qubits_in_cone_vec);
        Ok(PySet::new(py, elements)?.unbind())
    }

    /// Return a dictionary of circuit properties.
    fn properties(&self, py: Python) -> PyResult<HashMap<&str, Py<PyAny>>> {
        Ok(HashMap::from_iter([
            ("size", self.size(false)?.into_py_any(py)?),
            ("depth", self.depth(false)?.into_py_any(py)?),
            ("width", self.width().into_py_any(py)?),
            ("qubits", self.num_qubits().into_py_any(py)?),
            ("bits", self.num_clbits().into_py_any(py)?),
            ("factors", self.num_tensor_factors().into_py_any(py)?),
            ("operations", self.py_count_ops(py, true)?),
        ]))
    }

    /// Convert this DAG to a :class:`.QuantumCircuit`.
    ///
    /// This is a simple wrapper around :func:`.dag_to_circuit`.
    ///
    /// Args:
    ///     copy_operations: whether to deep copy the individual instructions.  If set to ``False``,
    ///         the operation is cheaper but mutations to the instructions in the circuit will
    ///         affect the original circuit.
    ///
    /// Returns:
    ///     a :class:`.QuantumCircuit` representing this same DAG.
    // If updating this signature, update `dag_to_circuit` as well.
    #[pyo3(signature=(*, copy_operations=true))]
    fn to_circuit(slf_: PyRef<Self>, copy_operations: bool) -> PyResult<Bound<PyAny>> {
        // We go via Python space here to make sure all the extra Python-space components not yet
        // tracked in `CircuitData` are copied too.
        imports::DAG_TO_CIRCUIT
            .get_bound(slf_.py())
            .call1((slf_, copy_operations))
    }

    /// Draws the dag circuit.
    ///
    /// This function needs `Graphviz <https://www.graphviz.org/>`_ to be
    /// installed. Graphviz is not a python package and can't be pip installed
    /// (the ``graphviz`` package on PyPI is a Python interface library for
    /// Graphviz and does not actually install Graphviz). You can refer to
    /// `the Graphviz documentation <https://www.graphviz.org/download/>`__ on
    /// how to install it.
    ///
    /// .. warning::
    ///     This function will call the system Graphviz tool on a file involving user-controllable
    ///     strings (such as gate labels or register names).  It is recommended to only call this
    ///     function on trusted input.
    ///
    /// Args:
    ///     scale (float): scaling factor
    ///     filename (str): file path to save image to (format inferred from name)
    ///     style (str):
    ///         'plain': B&W graph;
    ///         'color' (default): color input/output/op nodes
    ///
    /// Returns:
    ///     Ipython.display.Image: if in Jupyter notebook and not saving to file,
    ///     otherwise None.
    #[pyo3(signature=(scale=0.7, filename=None, style="color"))]
    fn draw<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        scale: f64,
        filename: Option<&str>,
        style: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = PyModule::import(py, "qiskit.visualization.dag_visualization")?;
        module.call_method1("dag_drawer", (slf, scale, filename, style))
    }

    #[pyo3(signature=(graph_attrs=None, node_attrs=None, edge_attrs=None))]
    fn _to_dot<'py>(
        &self,
        py: Python<'py>,
        graph_attrs: Option<BTreeMap<String, String>>,
        node_attrs: Option<Py<PyAny>>,
        edge_attrs: Option<Py<PyAny>>,
    ) -> PyResult<String> {
        let mut buffer = Vec::<u8>::new();
        build_dot(py, self, &mut buffer, graph_attrs, node_attrs, edge_attrs)?;
        Ok(String::from_utf8(buffer)?)
    }

    /// Add an input variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    #[pyo3(name = "add_input_var")]
    fn py_add_input_var(&mut self, var: expr::Var) -> PyResult<()> {
        self.add_var(var, VarType::Input)?;
        Ok(())
    }

    /// Add a captured variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    #[pyo3(name = "add_captured_var")]
    fn py_add_captured_var(&mut self, var: expr::Var) -> PyResult<()> {
        self.add_var(var, VarType::Capture)?;
        Ok(())
    }

    /// Add a captured stretch to the circuit.
    ///
    /// Args:
    ///     stretch: the stretch to add.
    #[pyo3(name = "add_captured_stretch")]
    fn py_add_captured_stretch(&mut self, stretch: expr::Stretch) -> PyResult<()> {
        self.add_stretch(stretch, StretchType::Capture)
            .map_err(Into::into)
    }

    /// Add a declared local variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    #[pyo3(name = "add_declared_var")]
    fn py_add_declared_var(&mut self, var: expr::Var) -> PyResult<()> {
        self.add_var(var, VarType::Declare)?;
        Ok(())
    }

    /// Add a declared stretch to the circuit.
    ///
    /// Args:
    ///     stretch: the stretch to add.
    #[pyo3(name = "add_declared_stretch")]
    fn py_add_declared_stretch(&mut self, stretch: expr::Stretch) -> PyResult<()> {
        self.add_stretch(stretch, StretchType::Declare)
            .map_err(Into::into)
    }

    /// Total number of classical variables tracked by the circuit.
    #[getter]
    fn num_vars(&self) -> usize {
        self.num_input_vars() + self.num_captured_vars() + self.num_declared_vars()
    }

    /// Number of input classical variables tracked by the circuit.
    #[getter]
    fn num_input_vars(&self) -> usize {
        self.vars_stretches.num_vars(VarType::Input)
    }

    /// Number of captured classical variables tracked by the circuit.
    #[getter]
    fn num_captured_vars(&self) -> usize {
        self.vars_stretches.num_vars(VarType::Capture)
    }

    /// Number of declared local classical variables tracked by the circuit.
    #[getter]
    fn num_declared_vars(&self) -> usize {
        self.vars_stretches.num_vars(VarType::Declare)
    }

    /// Total number of stretches tracked by the circuit.
    #[getter]
    pub fn num_stretches(&self) -> usize {
        self.num_captured_stretches() + self.num_declared_stretches()
    }

    /// Number of captured stretches tracked by the circuit.
    #[getter]
    fn num_captured_stretches(&self) -> usize {
        self.vars_stretches.num_stretches(StretchType::Capture)
    }

    /// Number of declared local stretches tracked by the circuit.
    #[getter]
    fn num_declared_stretches(&self) -> usize {
        self.vars_stretches.num_stretches(StretchType::Declare)
    }

    /// Is this realtime variable in the DAG?
    ///
    /// Args:
    ///     var: the variable or name to check.
    #[pyo3(name = "has_var")]
    fn py_has_var(&self, var: &Bound<PyAny>) -> PyResult<bool> {
        if let Ok(name) = var.extract::<String>() {
            Ok(self.vars_stretches.has_var(&name))
        } else {
            let var = var.extract::<expr::Var>()?;
            Ok(self.vars_stretches.vars().contains(&var))
        }
    }

    /// Is this stretch in the DAG?
    ///
    /// Args:
    ///     var: the stretch or name to check.
    #[pyo3(name = "has_stretch")]
    fn py_has_stretch(&self, stretch: &Bound<PyAny>) -> PyResult<bool> {
        if let Ok(name) = stretch.extract::<String>() {
            Ok(self.vars_stretches.has_stretch(&name))
        } else {
            let stretch = stretch.extract::<expr::Stretch>()?;
            Ok(self.vars_stretches.stretches().contains(&stretch))
        }
    }

    /// Is this identifier in the DAG?
    ///
    /// Args:
    ///     var: the identifier or name to check.
    #[pyo3(name = "has_identifier")]
    fn py_has_identifier(&self, var: &Bound<PyAny>) -> PyResult<bool> {
        if let Ok(name) = var.extract::<String>() {
            Ok(self.vars_stretches.has_identifier(&name))
        } else if let Ok(var) = var.extract::<expr::Var>() {
            let expr::Var::Standalone { .. } = var else {
                return Ok(false);
            };
            Ok(self.vars_stretches.vars().contains(&var))
        } else if let Ok(stretch) = var.extract::<expr::Stretch>() {
            Ok(self.vars_stretches.stretches().contains(&stretch))
        } else {
            Err(PyValueError::new_err(
                "identifier must be a name or expression kind Var or Stretch",
            ))
        }
    }

    /// Iterable over the input classical variables tracked by the circuit.
    fn iter_input_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let result = PySet::new(
            py,
            self.input_vars()
                .map(|v| v.clone().into_py_any(py).unwrap()),
        )?;
        Ok(result.into_any().try_iter()?.unbind())
    }

    /// Iterable over the captured classical variables tracked by the circuit.
    fn iter_captured_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let result = PySet::new(
            py,
            self.captured_vars()
                .map(|v| v.clone().into_py_any(py).unwrap()),
        )?;
        Ok(result.into_any().try_iter()?.unbind())
    }

    /// Iterable over the captured stretches tracked by the circuit.
    fn iter_captured_stretches(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let result = PySet::new(
            py,
            self.captured_stretches()
                .map(|v| v.clone().into_py_any(py).unwrap()),
        )?;
        Ok(result.into_any().try_iter()?.unbind())
    }

    /// Iterable over all captured identifiers tracked by the circuit.
    fn iter_captures(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let out_set = PySet::empty(py)?;
        for var in self.captured_vars() {
            out_set.add(var.clone())?;
        }
        for stretch in self.captured_stretches() {
            out_set.add(stretch.clone())?;
        }
        Ok(out_set.into_any().try_iter()?.unbind())
    }

    /// Iterable over the declared classical variables tracked by the circuit.
    fn iter_declared_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let result = PySet::new(
            py,
            self.declared_vars()
                .map(|v| v.clone().into_py_any(py).unwrap()),
        )?;
        Ok(result.into_any().try_iter()?.unbind())
    }

    /// Iterable over the declared stretches tracked by the circuit.
    fn iter_declared_stretches(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let result = PyList::new(
            py,
            self.declared_stretches()
                .map(|v| v.clone().into_py_any(py).unwrap()),
        )?;
        Ok(result.into_any().try_iter()?.unbind())
    }

    /// Iterable over all the classical variables tracked by the circuit.
    fn iter_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let out_set = PySet::empty(py)?;
        for var in self.vars_stretches.vars().objects() {
            out_set.add(var.clone())?;
        }
        Ok(out_set.into_any().try_iter()?.unbind())
    }

    /// Iterable over all the stretches tracked by the circuit.
    fn iter_stretches(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let out_set = PySet::empty(py)?;
        for stretch in self.vars_stretches.stretches().objects() {
            out_set.add(stretch.clone())?;
        }
        Ok(out_set.into_any().try_iter()?.unbind())
    }

    fn _has_edge(&self, source: usize, target: usize) -> bool {
        self.dag
            .contains_edge(NodeIndex::new(source), NodeIndex::new(target))
    }

    fn _is_dag(&self) -> bool {
        rustworkx_core::petgraph::algo::toposort(&self.dag, None).is_ok()
    }

    fn _in_edges(&self, py: Python, node_index: usize) -> Vec<Py<PyTuple>> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Incoming)
            .map(|wire| {
                (
                    wire.source().index(),
                    wire.target().index(),
                    match wire.weight() {
                        Wire::Qubit(qubit) => {
                            self.qubits.get(*qubit).into_bound_py_any(py).unwrap()
                        }
                        Wire::Clbit(clbit) => {
                            self.clbits.get(*clbit).into_bound_py_any(py).unwrap()
                        }
                        Wire::Var(var) => self
                            .vars_stretches
                            .vars()
                            .get(*var)
                            .cloned()
                            .into_bound_py_any(py)
                            .unwrap(),
                    },
                )
                    .into_pyobject(py)
                    .unwrap()
                    .unbind()
            })
            .collect()
    }

    fn _out_edges(&self, py: Python, node_index: usize) -> Vec<Py<PyTuple>> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Outgoing)
            .map(|wire| {
                (
                    wire.source().index(),
                    wire.target().index(),
                    match wire.weight() {
                        Wire::Qubit(qubit) => {
                            self.qubits.get(*qubit).into_bound_py_any(py).unwrap()
                        }
                        Wire::Clbit(clbit) => {
                            self.clbits.get(*clbit).into_bound_py_any(py).unwrap()
                        }
                        Wire::Var(var) => self
                            .vars_stretches
                            .vars()
                            .get(*var)
                            .cloned()
                            .into_bound_py_any(py)
                            .unwrap(),
                    },
                )
                    .into_pyobject(py)
                    .unwrap()
                    .unbind()
            })
            .collect()
    }

    fn _in_wires(&self, py: Python, node_index: usize) -> Vec<Py<PyAny>> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Incoming)
            .map(|wire| match wire.weight() {
                Wire::Qubit(qubit) => self.qubits.get(*qubit).into_py_any(py).unwrap(),
                Wire::Clbit(clbit) => self.clbits.get(*clbit).into_py_any(py).unwrap(),
                Wire::Var(var) => self
                    .vars_stretches
                    .vars()
                    .get(*var)
                    .cloned()
                    .into_py_any(py)
                    .unwrap(),
            })
            .collect()
    }

    fn _out_wires(&self, py: Python, node_index: usize) -> Vec<Py<PyAny>> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Outgoing)
            .map(|wire| match wire.weight() {
                Wire::Qubit(qubit) => self.qubits.get(*qubit).into_py_any(py).unwrap(),
                Wire::Clbit(clbit) => self.clbits.get(*clbit).into_py_any(py).unwrap(),
                Wire::Var(var) => self
                    .vars_stretches
                    .vars()
                    .get(*var)
                    .cloned()
                    .into_py_any(py)
                    .unwrap(),
            })
            .collect()
    }

    fn _find_successors_by_edge(
        &self,
        py: Python,
        node_index: usize,
        edge_checker: &Bound<PyAny>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let mut result = Vec::new();
        for e in self
            .dag
            .edges_directed(NodeIndex::new(node_index), Outgoing)
            .unique_by(|e| e.id())
        {
            let weight = match e.weight() {
                Wire::Qubit(qubit) => self.qubits.get(*qubit).into_py_any(py)?,
                Wire::Clbit(clbit) => self.clbits.get(*clbit).into_py_any(py)?,
                Wire::Var(var) => self
                    .vars_stretches
                    .vars()
                    .get(*var)
                    .cloned()
                    .into_py_any(py)?,
            };
            if edge_checker.call1((weight,))?.extract::<bool>()? {
                result.push(self.get_node(py, e.target())?);
            }
        }
        Ok(result)
    }

    fn _edges(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        self.dag
            .edge_indices()
            .map(|index| {
                let wire = self.dag.edge_weight(index).unwrap();
                match wire {
                    Wire::Qubit(qubit) => self.qubits.get(*qubit).into_py_any(py),
                    Wire::Clbit(clbit) => self.clbits.get(*clbit).into_py_any(py),
                    Wire::Var(var) => self
                        .vars_stretches
                        .vars()
                        .get(*var)
                        .cloned()
                        .into_py_any(py),
                }
            })
            .collect()
    }
}

impl<'a> DAGCircuit {
    /// Return an iterator of gate runs with op nodes that match a specified filter function
    pub fn collect_runs_by<F: Fn(&PackedInstruction) -> bool + 'a>(
        &'a self,
        filter: F,
    ) -> impl Iterator<Item = Vec<NodeIndex>> + 'a {
        let filter_fn = move |node_index: NodeIndex| -> Result<bool, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => Ok(filter(inst)),
                _ => Ok(false),
            }
        };

        match rustworkx_core::dag_algo::collect_runs(&self.dag, filter_fn) {
            Some(iter) => iter.map(|result| result.unwrap()),
            None => panic!("Invalid DAG cycle(s) detected"),
        }
    }
}

impl Default for DAGCircuit {
    fn default() -> Self {
        Self::new()
    }
}

impl DAGCircuit {
    /// Gives the DAG ownership of the provided basic block and returns a
    /// unique identifier that can be used to retrieve a reference to it
    /// later.
    ///
    /// No attempt is made to deduplicate the given block.
    /// No validation is performed to ensure that the given block is valid
    /// within the DAG.
    pub fn add_block(&mut self, block: DAGCircuit) -> Block {
        self.blocks.push(block)
    }

    /// Take the blocks from an extraction from an `OperationFromPython<DAGCircuit>` and add them to
    /// this DAG.
    ///
    /// Returns the parameters in terms of block references.
    pub fn take_parameter_blocks(
        &mut self,
        params: Option<Parameters<DAGCircuit>>,
    ) -> Option<Box<Parameters<Block>>> {
        params.map(|params| Box::new(params.map_blocks(|block| self.add_block(block))))
    }

    /// Extract the blocks from a Python-space `CircuitInstruction` and add them to this DAG.
    ///
    /// The inverse of this is [unpack_blocks_to_circuit_parameters].
    fn extract_blocks_from_circuit_parameters(
        &mut self,
        params: Option<&Parameters<CircuitData>>,
    ) -> Result<Option<Box<Parameters<Block>>>, DAGError> {
        params
            .map(|params| {
                params
                    .try_map_blocks_ref(|block| {
                        DAGCircuit::from_circuit_data(block, false, None, None, None, None)
                            .map(|dag| self.add_block(dag))
                    })
                    .map(Box::new)
            })
            .transpose()
    }

    /// Given a `params` object from an instruction packed into this DAG, extract any relevant
    /// blocks into the owned-object `CircuitData` block type, suitable for passing back to Python
    /// space.
    ///
    /// The inverse of this is [extract_blocks_from_circuit_parameters].
    fn unpack_blocks_to_circuit_parameters(
        &self,
        params: Option<&Parameters<Block>>,
    ) -> Result<Option<Parameters<CircuitData>>, DAGError> {
        Ok(params
            .map(|params| {
                params.try_map_blocks_ref(|block| CircuitData::from_dag_ref(&self.blocks[*block]))
            })
            .transpose()?)
    }

    /// Gets a mutable reference to the given basic block.
    ///
    /// Panics if the block is not found.
    pub fn view_block_mut(&mut self, block: Block) -> &mut DAGCircuit {
        &mut self.blocks[block]
    }

    /// Gets an immutable view of a control flow operation.
    ///
    /// Panics or produces incorrect results if `instr` is not from this DAG (or compatible with it,
    /// e.g. from a mapped [ControlFlowBlocks]).
    pub fn try_view_control_flow<'a>(
        &'a self,
        instr: &'a PackedInstruction,
    ) -> Option<ControlFlowView<'a, DAGCircuit>> {
        ControlFlowView::try_from_instruction(instr, &self.blocks)
    }

    /// Build a reference to the Python-space operation object (the `Gate`, etc) packed into an
    /// instruction.  This may construct the reference if the `Instruction` is a standard
    /// gate or instruction with no already stored operation.
    ///
    /// A standard-gate or standard-instruction operation object returned by this function is
    /// disconnected from the dag; updates to its parameters, label, duration, unit
    /// and condition will not be propagated back.
    ///
    /// Panics if `node` does not refer to an operation.
    pub fn unpack_py_op<'py>(
        &self,
        py: Python<'py>,
        instr: &PackedInstruction,
    ) -> PyResult<Bound<'py, PyAny>> {
        // `OnceLock::get_or_init` and the non-stabilised `get_or_try_init`, which would otherwise
        // be nice here are both non-reentrant.  This is a problem if the init yields control to the
        // Python interpreter as this one does, since that can allow CPython to freeze the thread
        // and for another to attempt the initialisation.
        #[cfg(feature = "cache_pygates")]
        {
            if let Some(ob) = instr.py_op.get() {
                return Ok(ob.bind(py).clone());
            }
        }
        let out = instruction::create_py_op(
            py,
            instr.op.view(),
            self.unpack_blocks_to_circuit_parameters(instr.params.as_deref())?,
            instr.label.as_deref().map(String::as_str),
        )?;
        #[cfg(feature = "cache_pygates")]
        // The unpacking operation can cause a thread pause and concurrency, since it can call
        // interpreted Python code for a standard gate, so we need to take care that some other
        // Python thread might have populated the cache before we do.
        let _ = instr.py_op.set(out.clone_ref(py));
        Ok(out.into_bound(py))
    }

    pub fn new() -> Self {
        DAGCircuit {
            name: None,
            metadata: None,
            dag: StableDiGraph::default(),
            qregs: RegisterData::new(),
            cregs: RegisterData::new(),
            qargs_interner: Interner::new(),
            cargs_interner: Interner::new(),
            qubits: ObjectRegistry::new(),
            clbits: ObjectRegistry::new(),
            blocks: ControlFlowBlocks::new(),
            global_phase: Param::Float(0.),
            duration: None,
            unit: "dt".to_string(),
            qubit_locations: BitLocator::new(),
            clbit_locations: BitLocator::new(),
            vars_stretches: VarStretchContainer::new(),
            qubit_io_map: Vec::new(),
            clbit_io_map: Vec::new(),
            var_io_map: Vec::new(),
            op_names: IndexMap::default(),
        }
    }
    /// Create an empty DAG, but with all the same qubit data, classical data and metadata
    /// (including global phase).
    ///
    /// This method clones both the `qargs_interner` and `cargs_interner` of `self`;
    /// `Interned<[Qubit]>` and `Interned<[Clbit]>` keys from `self` are valid in the output DAG.
    ///
    /// This does not include any pre-allocated capacity.
    pub fn copy_empty_like(&self, vars_mode: VarsMode, blocks_mode: BlocksMode) -> Self {
        self.copy_empty_like_with_capacity(0, 0, vars_mode, blocks_mode)
    }

    /// Create an empty DAG, but with all the same qubit data, classical data and metadata
    /// (including global phase).
    ///
    /// This method clones both the `qargs_interner` and `cargs_interner` of `self`;
    /// `Interned<[Qubit]>` and `Interned<[Clbit]>` keys from `self` are valid in the output DAG.
    pub fn copy_empty_like_with_same_capacity(
        &self,
        vars_mode: VarsMode,
        blocks_mode: BlocksMode,
    ) -> Self {
        self.copy_empty_like_with_capacity(
            self.dag.node_count().saturating_sub(2 * self.width()),
            self.dag.edge_count(),
            vars_mode,
            blocks_mode,
        )
    }

    /// Create an empty DAG, but with all the same qubit data, classical data and metadata
    /// (including global phase).
    ///
    /// This method clones both the `qargs_interner` and `cargs_interner` of `self`;
    /// `Interned<[Qubit]>` and `Interned<[Clbit]>` keys from `self` are valid in the output DAG.
    pub fn copy_empty_like_with_capacity(
        &self,
        num_ops: usize,
        num_edges: usize,
        vars_mode: VarsMode,
        blocks_mode: BlocksMode,
    ) -> Self {
        let mut out = self.qubitless_empty_like_with_capacity(
            self.num_qubits(),
            num_ops,
            num_edges,
            vars_mode,
            blocks_mode,
        );
        for bit in self.qubits.objects() {
            out.add_qubit_unchecked(bit.clone())
                .expect("There are no duplicate qubits in a valid dag");
        }
        for reg in self.qregs.registers() {
            out.add_qreg(reg.clone())
                .expect("There are no duplicate qubits/qregs in a valid dag");
        }
        // `copy_empty_like` has historically made a strong assumption that the exact same qargs
        // will be used in the output.  Some Qiskit functions rely on this undocumented behaviour.
        out.qargs_interner.clone_from(&self.qargs_interner);
        out
    }

    /// Create an empty DAG with the canonical "physical" register of the correct length, with all
    /// classical data and metadata retained.
    ///
    /// This is similar to [copy_empty_like_with_capacity] with [VarsMode::Alike], and copies the
    /// same things over (global phase, metadata, etc) it does, except for replacing the qubits.
    ///
    /// This method clones the `cargs_interner` of `self`; `Interned<[Clbit]>` keys from `self` are
    /// valid in the output DAG.  The `qargs_interner` is empty.
    ///
    /// This method is intended for use by passes that are converting a virtual DAG to a physical
    /// one.
    pub fn physical_empty_like_with_capacity(
        &self,
        num_qubits: usize,
        num_ops: usize,
        num_edges: usize,
        blocks_mode: BlocksMode,
    ) -> Result<Self, DAGError> {
        let mut out = self.qubitless_empty_like_with_capacity(
            num_qubits,
            num_ops,
            num_edges,
            VarsMode::Alike,
            blocks_mode,
        );
        out.add_qreg(QuantumRegister::new_owning("q", num_qubits as u32))?;
        Ok(out)
    }

    /// Create an empty DAG without any qubits, but with all the same classical data and metadata
    /// (including global phase).
    ///
    /// This is the base of all the `copy_empty_like` methods.
    ///
    /// This method clones the `cargs_interner` of `self`; `Interned<[Clbit]>` keys from `self` are
    /// valid in the output DAG.  The `qargs_interner` is empty.
    ///
    /// The graph will always have sufficient capacity to store the in and out nodes of the
    /// classical data.  `num_qubits` and `num_ops` together form the _additional_ capacity the
    /// graph will have preallocated to expand into.  `num_edges` should be the total number of
    /// edges expected because of the additional of op nodes; the minimal set of edges joining the
    /// wire in nodes to the out nodes is automatically accounted for.
    ///
    /// The resulting DAG has _no_ qubits.  The `num_qubits` argument is for defining how many
    /// qubits are expected to be added later.
    fn qubitless_empty_like_with_capacity(
        &self,
        num_qubits: usize,
        num_ops: usize,
        num_edges: usize,
        vars_mode: VarsMode,
        blocks_mode: BlocksMode,
    ) -> Self {
        let (num_vars, num_stretches) = match vars_mode {
            VarsMode::Drop => (0, 0),
            _ => (self.num_vars(), self.num_stretches()),
        };
        let mut target_dag = Self::with_capacity(
            num_qubits,
            self.num_clbits(),
            Some(num_vars),
            Some(num_ops),
            Some(num_edges),
            Some(num_stretches),
        );
        target_dag.name.clone_from(&self.name);
        target_dag.global_phase = self.global_phase.clone();
        target_dag.duration.clone_from(&self.duration);
        target_dag.unit.clone_from(&self.unit);
        target_dag.metadata.clone_from(&self.metadata);
        // We strongly expect the cargs to be copied over verbatim.  We don't know about qargs, so
        // we leave that with its default capacity.
        target_dag.cargs_interner = self.cargs_interner.clone();

        for bit in self.clbits.objects() {
            target_dag
                .add_clbit_unchecked(bit.clone())
                .expect("Duplicate clbits not possible in a valid DAG");
        }
        for reg in self.cregs.registers() {
            target_dag
                .add_creg(reg.clone())
                .expect("Duplicate creg/clbits not possible in a valid DAG");
        }
        match vars_mode {
            VarsMode::Alike => {
                target_dag.vars_stretches = self.vars_stretches.clone();
            }
            VarsMode::Captures => {
                target_dag.vars_stretches = self.vars_stretches.clone_as_captures();
            }
            VarsMode::Drop => {}
        }
        target_dag
            .add_var_wires(target_dag.vars_stretches.vars().len())
            .expect("Duplicate vars not possible in a valid DAG");
        if blocks_mode == BlocksMode::Keep {
            target_dag.blocks = self.blocks.clone();
        }
        target_dag
    }

    /// Modify `self` to mark its qubits as physical.
    ///
    /// This deletes the information about the virtual registers, and replaces it with the single
    /// (implicitly) physical register.  This method does not need to traverse the DAG, other than
    /// to add any ancilla in/out nodes.
    ///
    /// The qubit indices all stay the same; effectively, this is the application of the "trivial"
    /// layout.  If the incoming DAG is supposed to be considered physical, this method can be used
    /// to ensure it is in the canonical physical form.
    ///
    /// # Panics
    ///
    /// If `num_qubits` is less than the number of qubits in the DAG already.
    pub fn make_physical(&mut self, num_qubits: usize) {
        // If this method needs updating, `CircuitData::make_physical` probably does too.
        assert!(
            num_qubits >= self.num_qubits(),
            "number of qubits {num_qubits} too small for DAG"
        );
        let num_virtuals = self.num_qubits() as u32;
        let num_qubits: u32 = num_qubits
            .try_into()
            .expect("number of qubits must fit in a u32");
        // The strategy here is just to modify the qubit and quantum register objects entirely
        // inplace; we maintain all relative indices, so we don't need to modify any interner keys.
        let register = QuantumRegister::new_owning("q", num_qubits);
        let mut registry = ObjectRegistry::with_capacity(num_qubits as usize);
        let mut locator = BitLocator::with_capacity(num_qubits as usize);
        for (index, bit) in register.iter().enumerate() {
            registry.add_unique_within_capacity(bit.clone());
            locator.insert(
                bit,
                BitLocations::new(index as u32, [(register.clone(), index)]),
            );
        }
        let mut register_data = RegisterData::with_capacity(1);
        register_data
            .add_register(register, false)
            .expect("infallible when 'strict=false'");
        for qubit in num_virtuals..num_qubits {
            self.add_wire(Wire::Qubit(Qubit(qubit)))
                .expect("this qubit has the next sequential index");
        }
        self.qubits = registry;
        self.qregs = register_data;
        self.qubit_locations = locator;
    }

    /// Returns an immutable view of the [QuantumRegister] instances in the circuit.
    #[inline(always)]
    pub fn qregs(&self) -> &[QuantumRegister] {
        self.qregs.registers()
    }

    /// Returns an immutable view of the [ClassicalRegister] instances in the circuit.
    #[inline(always)]
    pub fn cregs(&self) -> &[ClassicalRegister] {
        self.cregs.registers()
    }

    /// Returns an immutable view of the [QuantumRegister] data struct in the circuit.
    #[inline(always)]
    pub fn qregs_data(&self) -> &RegisterData<QuantumRegister> {
        &self.qregs
    }

    /// Returns an immutable view of the [ClassicalRegister] data struct in the circuit.
    #[inline(always)]
    pub fn cregs_data(&self) -> &RegisterData<ClassicalRegister> {
        &self.cregs
    }

    /// Returns an immutable view of the qubit locations of the [DAGCircuit]
    #[inline(always)]
    pub fn qubit_locations(&self) -> &BitLocator<ShareableQubit, QuantumRegister> {
        &self.qubit_locations
    }

    /// Returns an immutable view of the clbit locations of the [DAGCircuit]
    #[inline(always)]
    pub fn clbit_locations(&self) -> &BitLocator<ShareableClbit, ClassicalRegister> {
        &self.clbit_locations
    }

    /// Returns an immutable view of the qubit io map
    #[inline(always)]
    pub fn qubit_io_map(&self) -> &[[NodeIndex; 2]] {
        &self.qubit_io_map
    }

    /// Returns an immutable view of the clbit io map
    #[inline(always)]
    pub fn clbit_io_map(&self) -> &[[NodeIndex; 2]] {
        &self.clbit_io_map
    }

    /// Returns an immutable view of the inner StableGraph managed by the circuit.
    #[inline(always)]
    pub fn dag(&self) -> &StableDiGraph<NodeType, Wire> {
        &self.dag
    }

    /// Returns an immutable view of the Interner used for Qargs
    #[inline(always)]
    pub fn qargs_interner(&self) -> &Interner<[Qubit]> {
        &self.qargs_interner
    }

    /// Returns an immutable view of the Interner used for Cargs
    #[inline(always)]
    pub fn cargs_interner(&self) -> &Interner<[Clbit]> {
        &self.cargs_interner
    }

    #[inline(always)]
    pub fn blocks(&self) -> &ControlFlowBlocks<DAGCircuit> {
        &self.blocks
    }

    /// Returns an immutable view of the Global Phase `Param` of the circuit
    #[inline(always)]
    pub fn global_phase(&self) -> &Param {
        &self.global_phase
    }

    /// Returns an immutable view of the Qubits registered in the circuit
    #[inline(always)]
    pub fn qubits(&self) -> &ObjectRegistry<Qubit, ShareableQubit> {
        &self.qubits
    }

    /// Returns an immutable view of the Classical bits registered in the circuit
    #[inline(always)]
    pub fn clbits(&self) -> &ObjectRegistry<Clbit, ShareableClbit> {
        &self.clbits
    }

    /// Returns an immutable view of the Variable wires registered in the circuit
    #[inline(always)]
    pub fn vars(&self) -> &ObjectRegistry<Var, expr::Var> {
        self.vars_stretches.vars()
    }

    /// Returns an iterator over the input variables used by the circuit.
    pub fn input_vars(&self) -> impl ExactSizeIterator<Item = &expr::Var> {
        self.vars_stretches.iter_vars(VarType::Input)
    }

    /// Returns an iterator over the variables captured by the circuit.
    pub fn captured_vars(&self) -> impl ExactSizeIterator<Item = &expr::Var> {
        self.vars_stretches.iter_vars(VarType::Capture)
    }

    /// Returns an iterator over the variables declared within the circuit.
    pub fn declared_vars(&self) -> impl ExactSizeIterator<Item = &expr::Var> {
        self.vars_stretches.iter_vars(VarType::Declare)
    }

    /// Returns an iterator over the stretches captured by the circuit.
    pub fn captured_stretches(&self) -> impl ExactSizeIterator<Item = &expr::Stretch> {
        self.vars_stretches.iter_stretches(StretchType::Capture)
    }

    /// Returns an iterator over the stretches declared within the circuit.
    pub fn declared_stretches(&self) -> impl ExactSizeIterator<Item = &expr::Stretch> {
        self.vars_stretches.iter_stretches(StretchType::Declare)
    }

    pub fn remove_qubits<T: IntoIterator<Item = Qubit>>(
        &mut self,
        qubits: T,
    ) -> Result<(), DAGError> {
        let qubits: HashSet<Qubit> = qubits.into_iter().collect();

        let mut busy_bits = Vec::new();
        for bit in qubits.iter() {
            if !self.is_wire_idle(Wire::Qubit(*bit)) {
                busy_bits.push(
                    self.qubits
                        .get(*bit)
                        .cloned()
                        .expect("Qubit index should exist."),
                );
            }
        }

        if !busy_bits.is_empty() {
            return Err(DAGError::QubitsNotIdle(busy_bits));
        }

        // Remove any references to bits.
        let mut qregs_to_remove = Vec::new();
        for qreg in self.qregs.registers() {
            for bit in qreg.bits() {
                if qubits.contains(&self.qubits.find(&bit).unwrap()) {
                    qregs_to_remove.push(qreg.clone());
                    break;
                }
            }
        }
        self.remove_qregs(qregs_to_remove)?;

        // Remove DAG in/out nodes etc.
        for bit in qubits.iter() {
            self.remove_idle_wire(Wire::Qubit(*bit));
        }

        // Copy the current qubit mapping so we can use it while remapping
        // wires used on edges and in operation qargs.
        let old_qubits = self.qubits.clone();

        // Remove the qubit indices, which will invalidate our mapping of Qubit to
        // Python bits throughout the entire DAG.
        self.qubits.remove_indices(qubits.clone());

        // Update input/output maps to use new Qubits.
        let io_mapping: HashMap<Qubit, [NodeIndex; 2]> = self
            .qubit_io_map
            .drain(..)
            .enumerate()
            .filter_map(|(k, v)| {
                let qubit = Qubit::new(k);
                if qubits.contains(&qubit) {
                    None
                } else {
                    Some((self.qubits.find(old_qubits.get(qubit).unwrap()).unwrap(), v))
                }
            })
            .collect();

        self.qubit_io_map = (0..io_mapping.len())
            .map(|idx| {
                let qubit = Qubit::new(idx);
                io_mapping[&qubit]
            })
            .collect();

        // Update edges to use the new Qubits.
        for edge_weight in self.dag.edge_weights_mut() {
            if let Wire::Qubit(b) = edge_weight {
                *b = self.qubits.find(old_qubits.get(*b).unwrap()).unwrap();
            }
        }

        // Update operation qargs to use the new Qubits.
        for node_weight in self.dag.node_weights_mut() {
            match node_weight {
                NodeType::Operation(op) => {
                    let qargs = self.qargs_interner.get(op.qubits);
                    let qarg_bits = old_qubits.map_indices(qargs).cloned();
                    op.qubits = self
                        .qargs_interner
                        .insert_owned(self.qubits.map_objects(qarg_bits)?.collect());
                }
                NodeType::QubitIn(q) | NodeType::QubitOut(q) => {
                    *q = self.qubits.find(old_qubits.get(*q).unwrap()).unwrap();
                }
                _ => (),
            }
        }

        // Update bit locations.
        for (i, bit) in self.qubits.objects().iter().enumerate() {
            let raw_loc = self.qubit_locations.get_mut(bit).unwrap();
            raw_loc.index = i as u32;
        }
        Ok(())
    }

    /// Remove the specified quantum registers
    fn remove_qregs<T: IntoIterator<Item = QuantumRegister>>(
        &mut self,
        qregs: T,
    ) -> Result<(), DAGError> {
        // let self_bound_cregs = self.cregs.bind(py);
        let mut valid_regs: Vec<QuantumRegister> = Vec::new();
        for qregs in qregs.into_iter() {
            if let Some(reg) = self.qregs.get(qregs.name()) {
                if reg != &qregs {
                    return Err(DAGError::RegisterNotInCircuit(RegisterType::Quantum(qregs)));
                }
                valid_regs.push(qregs);
            } else {
                return Err(DAGError::RegisterNotInCircuit(RegisterType::Quantum(qregs)));
            }
        }

        // Use an iterator that will remove the registers from the circuit as it iterates.
        let valid_names = valid_regs.iter().map(|reg| {
            for (index, bit) in reg.bits().enumerate() {
                let bit_position = self.qubit_locations.get_mut(&bit).unwrap();
                bit_position.remove_register(reg, index);
            }
            reg.name().to_string()
        });
        self.qregs.remove_registers(valid_names);
        Ok(())
    }

    /// Remove the given clbits in the circuit
    ///
    /// This will reorder all the bits in the circuit.
    pub fn remove_clbits<T: IntoIterator<Item = Clbit>>(
        &mut self,
        clbits: T,
    ) -> Result<(), DAGError> {
        let clbits: HashSet<Clbit> = clbits.into_iter().collect();
        let mut busy_bits = Vec::new();
        for bit in clbits.iter() {
            if !self.is_wire_idle(Wire::Clbit(*bit)) {
                busy_bits.push(
                    self.clbits
                        .get(*bit)
                        .cloned()
                        .expect("Clbit index should exist."),
                );
            }
        }

        if !busy_bits.is_empty() {
            return Err(DAGError::ClbitsNotIdle(busy_bits));
        }

        // Remove any references to bits.
        let mut cregs_to_remove = Vec::new();
        for creg in self.cregs.registers() {
            for bit in creg.bits() {
                if clbits.contains(&self.clbits.find(&bit).unwrap()) {
                    cregs_to_remove.push(creg.clone());
                    break;
                }
            }
        }
        self.remove_cregs(cregs_to_remove)?;

        // Remove DAG in/out nodes etc.
        for bit in clbits.iter() {
            self.remove_idle_wire(Wire::Clbit(*bit));
        }

        // Copy the current clbit mapping so we can use it while remapping
        // wires used on edges and in operation cargs.
        let old_clbits = self.clbits.clone();

        // Remove the clbit indices, which will invalidate our mapping of Clbit to
        // Python bits throughout the entire DAG.
        self.clbits.remove_indices(clbits.clone());

        // Update input/output maps to use new Clbits.
        let io_mapping: HashMap<Clbit, [NodeIndex; 2]> = self
            .clbit_io_map
            .drain(..)
            .enumerate()
            .filter_map(|(k, v)| {
                let clbit = Clbit::new(k);
                if clbits.contains(&clbit) {
                    None
                } else {
                    Some((
                        self.clbits
                            .find(old_clbits.get(Clbit::new(k)).unwrap())
                            .unwrap(),
                        v,
                    ))
                }
            })
            .collect();

        self.clbit_io_map = (0..io_mapping.len())
            .map(|idx| {
                let clbit = Clbit::new(idx);
                io_mapping[&clbit]
            })
            .collect();

        // Update edges to use the new Clbits.
        for edge_weight in self.dag.edge_weights_mut() {
            if let Wire::Clbit(c) = edge_weight {
                *c = self.clbits.find(old_clbits.get(*c).unwrap()).unwrap();
            }
        }

        // Update operation cargs to use the new Clbits.
        for node_weight in self.dag.node_weights_mut() {
            match node_weight {
                NodeType::Operation(op) => {
                    let cargs = self.cargs_interner.get(op.clbits);
                    let carg_bits = old_clbits.map_indices(cargs).cloned();
                    op.clbits = self
                        .cargs_interner
                        .insert_owned(self.clbits.map_objects(carg_bits)?.collect());
                }
                NodeType::ClbitIn(c) | NodeType::ClbitOut(c) => {
                    *c = self.clbits.find(old_clbits.get(*c).unwrap()).unwrap();
                }
                _ => (),
            }
        }

        // Update bit locations.
        for (i, bit) in self.clbits.objects().iter().enumerate() {
            let raw_loc = self.clbit_locations.get_mut(bit).unwrap();
            raw_loc.index = i as u32;
        }
        Ok(())
    }

    /// Remove the specified classical registers
    pub fn remove_cregs<T: IntoIterator<Item = ClassicalRegister>>(
        &mut self,
        cregs: T,
    ) -> Result<(), DAGError> {
        let mut valid_regs: Vec<ClassicalRegister> = Vec::new();
        for creg in cregs {
            if let Some(reg) = self.cregs.get(creg.name()) {
                if reg != &creg {
                    return Err(DAGError::RegisterNotInCircuit(RegisterType::Classical(
                        creg,
                    )));
                }
                valid_regs.push(creg);
            } else {
                return Err(DAGError::RegisterNotInCircuit(RegisterType::Classical(
                    creg,
                )));
            }
        }

        // Use an iterator that will remove the registers from the circuit as it iterates.
        let valid_names = valid_regs.iter().map(|reg| {
            for (index, bit) in reg.bits().enumerate() {
                let bit_position = self.clbit_locations.get_mut(&bit).unwrap();
                bit_position.remove_register(reg, index);
            }
            reg.name().to_string()
        });
        self.cregs.remove_registers(valid_names);
        Ok(())
    }

    /// Permute the qubit indices used by instructions.
    ///
    /// This does not modify the order of the [ShareableQubit]s, so "qubit 0" will still point to
    /// the same [ShareableQubit] object.  It only rewrites the relative qubit indices that any
    /// given qubit acts upon.
    ///
    /// `map_fn` must implement a permutation.  It is called many times on the same qubit without
    /// caching; if the computation is expensive, consider using a function that indexes into a
    /// slice.
    ///
    /// This is useful in situations where we're applying an initial layout.
    ///
    /// This takes time (individually) proportional to the number of qubits, the number of edges in
    /// the graph, and the sum of the lengths of all distinct qargs.
    ///
    /// # Panics
    ///
    /// If `map_fn` returns an out-of-bounds qubit, or maps two different qubits to the same index.
    pub fn reindex_qargs(&mut self, mut map_fn: impl FnMut(Qubit) -> Qubit) {
        // The Python-space spiritual successors of this code (the `ApplyLayout` pass, e.g.) used to
        // have to rebuild the whole DAG including reinferring the wire structure.  That is a
        // foolproof way of doing things, but recalculates and reallocates more than necessary.  We
        // can leave the graph structure unchanged, provided we change the labelling of the wires
        // (including the in- and out-nodes), and update the interned copies of all the qargs.

        // Prepare the new in- and out-node structure.  This doubles as the validity check of the
        // input, so we avoid mutating the DAG until it's complete.
        let mut new_io_map = vec![[<NodeIndex as IndexType>::max(); 2]; self.qubit_io_map.len()];
        for (old, [in_, out]) in self.qubit_io_map.iter().enumerate() {
            let new = map_fn(Qubit::new(old));
            assert!(
                new_io_map[new.index()][0] == <NodeIndex as IndexType>::max(),
                "qubit {old} was reindexed to {new:?}, which was already assigned"
            );
            new_io_map[new.index()] = [*in_, *out];
        }

        // From here on, everything should be infallible.
        self.qubit_io_map = new_io_map;
        for (new, [in_, out]) in self.qubit_io_map.iter().enumerate() {
            let new = Qubit::new(new);
            self.dag[*in_] = NodeType::QubitIn(new);
            self.dag[*out] = NodeType::QubitOut(new);
        }
        for wire in self.dag.edge_weights_mut() {
            if let Wire::Qubit(qubit) = *wire {
                *wire = Wire::Qubit(map_fn(qubit));
            }
        }
        self.qargs_interner.map_inplace(|mut qargs| {
            for qubit in qargs.iter_mut() {
                *qubit = map_fn(*qubit)
            }
            qargs
        })
    }

    /// Merge the `qargs` in a different [Interner] into this DAG, remapping the qubits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one DAG to
    /// another, like in substitution methods, or rebuilding a new DAG out of a lot of smaller ones.
    /// See [Interner::merge_map_slice] for more information on the mapping function.
    ///
    /// The input [InternedMap] is cleared of its previous entries by this method, and then we
    /// re-use the allocation.
    pub fn merge_qargs_using(
        &mut self,
        other: &Interner<[Qubit]>,
        map_fn: impl FnMut(&Qubit) -> Option<Qubit>,
        map: &mut InternedMap<[Qubit]>,
    ) {
        // 4 is an arbitrary guess for the amount of stack space to allocate for mapping the
        // `qargs`, but it doesn't matter if it's too short because it'll safely spill to the heap.
        self.qargs_interner
            .merge_map_slice_using::<4>(other, map_fn, map);
    }

    /// Merge the `qargs` in a different [Interner] into this DAG, remapping the qubits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one DAG to
    /// another, like in substitution methods, or rebuilding a new DAG out of a lot of smaller ones.
    /// See [Interner::merge_map_slice] for more information on the mapping function.
    pub fn merge_qargs(
        &mut self,
        other: &Interner<[Qubit]>,
        map_fn: impl FnMut(&Qubit) -> Option<Qubit>,
    ) -> InternedMap<[Qubit]> {
        let mut out = InternedMap::new();
        self.merge_qargs_using(other, map_fn, &mut out);
        out
    }

    /// Merge the `cargs` in a different [Interner] into this DAG, remapping the clbits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one DAG to
    /// another, like in substitution methods, or rebuilding a new DAG out of a lot of smaller ones.
    /// See [Interner::merge_map_slice] for more information on the mapping function.
    ///
    /// The input [InternedMap] is cleared of its previous entries by this method, and then we
    /// re-use the allocation.
    pub fn merge_cargs_using(
        &mut self,
        other: &Interner<[Clbit]>,
        map_fn: impl FnMut(&Clbit) -> Option<Clbit>,
        map: &mut InternedMap<[Clbit]>,
    ) {
        // 4 is an arbitrary guess for the amount of stack space to allocate for mapping the
        // `cargs`, but it doesn't matter if it's too short because it'll safely spill to the heap.
        self.cargs_interner
            .merge_map_slice_using::<4>(other, map_fn, map);
    }

    /// Merge the `cargs` in a different [Interner] into this DAG, remapping the clbits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one DAG to
    /// another, like in substitution methods, or rebuilding a new DAG out of a lot of smaller ones.
    /// See [Interner::merge_map_slice] for more information on the mapping function.
    pub fn merge_cargs(
        &mut self,
        other: &Interner<[Clbit]>,
        map_fn: impl FnMut(&Clbit) -> Option<Clbit>,
    ) -> InternedMap<[Clbit]> {
        let mut out = InternedMap::new();
        self.merge_cargs_using(other, map_fn, &mut out);
        out
    }

    /// Return an iterator of gate runs with non-conditional op nodes of given names
    pub fn collect_runs(
        &self,
        namelist: HashSet<String>,
    ) -> impl Iterator<Item = Vec<NodeIndex>> + '_ {
        let filter_fn = move |node_index: NodeIndex| -> Result<bool, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => Ok(namelist.contains(inst.op.name())),
                _ => Ok(false),
            }
        };

        match rustworkx_core::dag_algo::collect_runs(&self.dag, filter_fn) {
            Some(iter) => iter.map(|result| result.unwrap()),
            None => panic!("invalid DAG: cycle(s) detected!"),
        }
    }

    /// Return a set of non-conditional runs of 1q "op" nodes.
    pub fn collect_1q_runs(&self) -> Option<impl Iterator<Item = Vec<NodeIndex>> + '_> {
        let filter_fn = move |node_index: NodeIndex| -> Result<bool, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => Ok(inst.op.num_qubits() == 1
                    && inst.op.num_clbits() == 0
                    && !inst.is_parameterized()
                    && (inst.op.try_standard_gate().is_some() || inst.try_matrix().is_some())),
                _ => Ok(false),
            }
        };
        rustworkx_core::dag_algo::collect_runs(&self.dag, filter_fn)
            .map(|node_iter| node_iter.map(|x| x.unwrap()))
    }

    /// Return a set of non-conditional runs of 2q "op" nodes.
    pub fn collect_2q_runs(&self) -> Option<Vec<Vec<NodeIndex>>> {
        let filter_fn = move |node_index: NodeIndex| -> Result<Option<bool>, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => match inst.op.view() {
                    OperationRef::StandardGate(gate) => {
                        Ok(Some(gate.num_qubits() <= 2 && !inst.is_parameterized()))
                    }
                    OperationRef::PyCustom(PyInstruction {
                        kind: PyOpKind::Gate,
                        qubits: ..=2,
                        ..
                    }) => Ok(Some(!inst.is_parameterized())),
                    OperationRef::Unitary(gate) => Ok(Some(gate.num_qubits() <= 2)),
                    _ => Ok(Some(false)),
                },
                _ => Ok(None),
            }
        };

        let color_fn = move |edge_index: EdgeIndex| -> Result<Option<usize>, Infallible> {
            let wire = self.dag.edge_weight(edge_index).unwrap();
            match wire {
                Wire::Qubit(index) => Ok(Some(index.index())),
                _ => Ok(None),
            }
        };
        rustworkx_core::dag_algo::collect_bicolor_runs(&self.dag, filter_fn, color_fn).unwrap()
    }

    /// Track an instruction from the [DAGCircuit].  This updates the name-count and blocks tracking
    /// information.
    ///
    /// The borrow-check rules of this function require you to track the instruction before you pass
    /// over ownership of the instruction to the DAG; you can't track an instruction that's already
    /// in place (use [track_instruction_in_place] if you need that).
    #[inline]
    fn track_instruction(&mut self, inst: &PackedInstruction) {
        track_instruction(inst, &mut self.op_names, &mut self.blocks)
    }
    /// Track an instruction from the [DAGCircuit] by node index.  This updates the name-count and
    /// blocks tracking information.
    ///
    /// Panics if the given `node` is not an operation node.
    ///
    /// Typically you should prefer to call [track_instruction] before you insert the node, since
    /// the type system will then ensure you only track correctly typed nodes.  You might need to
    /// call this function when using complex graph-manipulation functions (like `contract_nodes`)
    /// that do bulk operations on the DAG, where you never easily hold all the owned instructions.
    #[inline]
    fn track_instruction_in_place(&mut self, node: NodeIndex) {
        track_instruction(
            self.dag[node].unwrap_operation(),
            &mut self.op_names,
            &mut self.blocks,
        )
    }

    /// Untrack an instruction from the [DAGCircuit].  This updates the name-count and blocks
    /// tracking information.
    ///
    /// The borrow-check rules of this function require you to untrack the instruction after you
    /// have removed it from the DAG; you can't untrack an instruction that's still reachable
    /// through the `&mut self` reference.  If you need that, use [untrack_instruction_in_place].
    #[inline]
    fn untrack_instruction(&mut self, inst: &PackedInstruction) {
        untrack_instruction(inst, &mut self.op_names, &mut self.blocks)
    }
    /// Unrtack an instruction from the [DAGCircuit] by node index.  This updates the name-count and
    /// blocks tracking information.
    ///
    /// Panics if the given `node` is not an operation node.
    ///
    /// Typically you should prefer to call [untrack_instruction] after you remove the node, since
    /// the type system will then ensure you only track correctly typed nodes.  You might need to
    /// call this function when using complex graph-manipulation functions (like `contract_nodes`)
    /// that do bulk operations on the DAG, where you never easily hold all the owned instructions.
    #[inline]
    fn untrack_instruction_in_place(&mut self, node: NodeIndex) {
        untrack_instruction(
            self.dag[node].unwrap_operation(),
            &mut self.op_names,
            &mut self.blocks,
        )
    }

    pub fn quantum_predecessors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.dag
            .edges_directed(node, Incoming)
            .filter_map(|e| match e.weight() {
                Wire::Qubit(_) => Some(e.source()),
                _ => None,
            })
            .unique()
    }

    pub fn quantum_successors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.dag
            .edges_directed(node, Outgoing)
            .filter_map(|e| match e.weight() {
                Wire::Qubit(_) => Some(e.target()),
                _ => None,
            })
            .unique()
    }

    /// Apply a [PackedInstruction] to the back of the circuit.
    ///
    /// The caller must already have arranged for any interned tracking fields (such as `qubits`,
    /// `clbits` and any `Block`s) to have been added to the DAG.  The function may panic, or other
    /// produce an invalid data structure if not.
    #[inline]
    pub fn push_back(&mut self, instr: PackedInstruction) -> Result<NodeIndex, DAGError> {
        self.push_external(instr, Direction::Outgoing)
    }

    /// Apply a [PackedInstruction] to the front of the circuit.
    ///
    /// The caller must already have arranged for any interned tracking fields (such as `qubits`,
    /// `clbits` and any `Block`s) to have been added to the DAG.  The function may panic, or other
    /// produce an invalid data structure if not.
    #[inline]
    pub fn push_front(&mut self, instr: PackedInstruction) -> Result<NodeIndex, DAGError> {
        self.push_external(instr, Direction::Incoming)
    }

    /// Push a [PackedInstruction] to become an external operation node on the DAG, with the edges
    /// between the instruction and the DAG operations are the given direction, relative to the
    /// DAG.
    ///
    /// Explicitly, [Direction::Outgoing] is equivalent to [push_back], whereas
    /// [Direction::Incoming] is equivalent to [push_front].
    fn push_external(
        &mut self,
        instr: PackedInstruction,
        dir: Direction,
    ) -> Result<NodeIndex, DAGError> {
        self.track_instruction(&instr);
        let (all_cbits, vars) = self
            .get_classical_resources(&instr)
            .map_err(DAGError::Python)?;
        let qubits_id = instr.qubits;
        let new_node = self.dag.add_node(NodeType::Operation(instr));
        let terminus_index = match dir {
            Direction::Incoming => 0, // the "in" nodes
            Direction::Outgoing => 1, // the "out" nodes
        };
        // Put the new node in-between the previously "last" nodes on each wire
        // and the terminal map.
        let termini: IndexSet<NodeIndex> = self
            .qargs_interner
            .get(qubits_id)
            .iter()
            .map(|q| self.qubit_io_map[q.index()][terminus_index])
            .chain(
                all_cbits
                    .iter()
                    .map(|c| self.clbit_io_map[c.index()][terminus_index]),
            )
            .chain(
                vars.iter()
                    .flatten()
                    .map(|v| self.var_io_map[v.index()][terminus_index]),
            )
            .collect();

        for terminus in termini {
            let last_edges: Vec<_> = self
                .dag
                .edges_directed(terminus, dir.opposite())
                .map(|e| {
                    (
                        match dir {
                            Direction::Outgoing => e.source(),
                            Direction::Incoming => e.target(),
                        },
                        e.id(),
                        *e.weight(),
                    )
                })
                .collect();
            for (op_node, old_edge, weight) in last_edges.into_iter() {
                match dir {
                    Direction::Outgoing => {
                        self.dag.add_edge(op_node, new_node, weight);
                        self.dag.add_edge(new_node, terminus, weight);
                    }
                    Direction::Incoming => {
                        self.dag.add_edge(terminus, new_node, weight);
                        self.dag.add_edge(new_node, op_node, weight);
                    }
                }
                self.dag.remove_edge(old_edge);
            }
        }

        Ok(new_node)
    }

    fn get_classical_resources(
        &self,
        instr: &PackedInstruction,
    ) -> PyResult<(Vec<Clbit>, Option<Vec<Var>>)> {
        let (all_clbits, vars): (Vec<Clbit>, Option<Vec<Var>>) = {
            if self.may_have_additional_wires(instr) {
                let mut clbits: IndexSet<Clbit> =
                    IndexSet::from_iter(self.cargs_interner.get(instr.clbits).iter().copied());
                let (additional_clbits, additional_vars) =
                    Python::attach(|py| self.additional_wires(py, instr))?;
                for clbit in additional_clbits {
                    clbits.insert(clbit);
                }
                (clbits.into_iter().collect(), Some(additional_vars))
            } else {
                (self.cargs_interner.get(instr.clbits).to_vec(), None)
            }
        };
        Ok((all_clbits, vars))
    }

    /// Apply a [PackedOperation] to the back of the circuit.
    pub fn apply_operation_back(
        &mut self,
        op: PackedOperation,
        qargs: &[Qubit],
        cargs: &[Clbit],
        params: Option<Parameters<Block>>,
        label: Option<String>,
        #[cfg(feature = "cache_pygates")] py_op: Option<Py<PyAny>>,
    ) -> Result<NodeIndex, DAGError> {
        self.inner_apply_op(
            op,
            qargs,
            cargs,
            params,
            label,
            #[cfg(feature = "cache_pygates")]
            py_op,
            false,
        )
    }

    /// Apply a [PackedOperation] to the front of the circuit.
    pub fn apply_operation_front(
        &mut self,
        op: PackedOperation,
        qargs: &[Qubit],
        cargs: &[Clbit],
        params: Option<Parameters<Block>>,
        label: Option<String>,
        #[cfg(feature = "cache_pygates")] py_op: Option<Py<PyAny>>,
    ) -> Result<NodeIndex, DAGError> {
        self.inner_apply_op(
            op,
            qargs,
            cargs,
            params,
            label,
            #[cfg(feature = "cache_pygates")]
            py_op,
            true,
        )
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn inner_apply_op(
        &mut self,
        op: PackedOperation,
        qargs: &[Qubit],
        cargs: &[Clbit],
        params: Option<Parameters<Block>>,
        label: Option<String>,
        #[cfg(feature = "cache_pygates")] py_op: Option<Py<PyAny>>,
        front: bool,
    ) -> Result<NodeIndex, DAGError> {
        // Check that all qargs are within an acceptable range
        qargs.iter().try_for_each(|qarg| {
            if qarg.index() >= self.num_qubits() {
                return Err(DAGError::WireOutOfRange(
                    Wire::Qubit(*qarg),
                    self.num_qubits(),
                ));
            }
            Ok(())
        })?;

        // Check that all cargs are within an acceptable range
        cargs.iter().try_for_each(|carg| {
            if carg.index() >= self.num_clbits() {
                return Err(DAGError::WireOutOfRange(
                    Wire::Clbit(*carg),
                    self.num_clbits(),
                ));
            }
            Ok(())
        })?;

        #[cfg(feature = "cache_pygates")]
        let py_op = if let Some(py_op) = py_op {
            py_op.into()
        } else {
            OnceLock::new()
        };
        let instr = PackedInstruction {
            op,
            qubits: self.qargs_interner.insert(qargs),
            clbits: self.cargs_interner.insert(cargs),
            params: params.map(Box::new),
            label: label.map(Box::new),
            #[cfg(feature = "cache_pygates")]
            py_op,
        };

        if front {
            self.push_front(instr)
        } else {
            self.push_back(instr)
        }
    }

    fn sort_key(&self, node: NodeIndex) -> SortKeyType<'_> {
        match &self.dag[node] {
            NodeType::Operation(packed) => (
                self.qargs_interner.get(packed.qubits),
                self.cargs_interner.get(packed.clbits),
            ),
            NodeType::QubitIn(q) => (std::slice::from_ref(q), &[Clbit(u32::MAX)]),
            NodeType::QubitOut(_q) => (&[Qubit(u32::MAX)], &[Clbit(u32::MAX)]),
            NodeType::ClbitIn(c) => (&[Qubit(u32::MAX)], std::slice::from_ref(c)),
            NodeType::ClbitOut(_c) => (&[Qubit(u32::MAX)], &[Clbit(u32::MAX)]),
            _ => (&[], &[]),
        }
    }

    fn topological_nodes(&self, reverse: bool) -> Vec<NodeIndex> {
        let key = |node: NodeIndex| -> Result<SortKeyType, Infallible> { Ok(self.sort_key(node)) };
        let Ok(nodes) = rustworkx_core::dag_algo::lexicographical_topological_sort(
            &self.dag, key, reverse, None,
        )
        .map_err(|e| match e {
            rustworkx_core::dag_algo::TopologicalSortError::CycleOrBadInitialState => {
                panic!("DAG should prevent itself from becoming cyclic");
            }
        });
        nodes
    }

    pub fn topological_op_nodes(&self, reverse: bool) -> impl Iterator<Item = NodeIndex> + '_ {
        self.topological_nodes(reverse)
            .into_iter()
            .filter(|node: &NodeIndex| matches!(&self.dag[*node], NodeType::Operation(_)))
    }

    fn topological_key_sort(
        &self,
        py: Python,
        key: &Bound<PyAny>,
        reverse: bool,
    ) -> PyResult<impl Iterator<Item = NodeIndex> + use<>> {
        // This path (user provided key func) is not ideal, since we no longer
        // use a string key after moving to Rust, in favor of using a tuple
        // of the qargs and cargs interner IDs of the node.
        let key = |node: NodeIndex| -> PyResult<String> {
            let node = self.get_node(py, node)?;
            key.call1((node,))?.extract()
        };
        Ok(rustworkx_core::dag_algo::lexicographical_topological_sort(
            &self.dag, key, reverse, None,
        )
        .map_err(|e| match e {
            rustworkx_core::dag_algo::TopologicalSortError::CycleOrBadInitialState => {
                PyValueError::new_err(format!("{e}"))
            }
            rustworkx_core::dag_algo::TopologicalSortError::KeyError(ref e) => e.clone_ref(py),
        })?
        .into_iter())
    }

    #[inline]
    pub fn has_control_flow(&self) -> bool {
        CONTROL_FLOW_OP_NAMES
            .iter()
            .any(|x| self.op_names.contains_key(&x.to_string()))
    }

    /// Is the given [Wire] idle?
    ///
    /// # Panics
    ///
    /// If the [Wire] isn't in the [DAGCircuit].
    pub fn is_wire_idle(&self, wire: Wire) -> bool {
        let [input_node, output_node] = match wire {
            Wire::Qubit(qubit) => self.qubit_io_map[qubit.index()],
            Wire::Clbit(clbit) => self.clbit_io_map[clbit.index()],
            Wire::Var(var) => self.var_io_map[var.index()],
        };
        self.dag
            .neighbors_directed(input_node, Outgoing)
            .next()
            .expect("input node must at least be connected to output")
            == output_node
    }

    fn may_have_additional_wires(&self, instr: &PackedInstruction) -> bool {
        match instr.op.view() {
            OperationRef::ControlFlow(_) => true,
            OperationRef::PyCustom(PyInstruction {
                kind: PyOpKind::Instruction,
                op_name,
                ..
            }) => op_name == "store",
            _ => false,
        }
    }

    fn additional_wires(
        &self,
        py: Python,
        instr: &PackedInstruction,
    ) -> PyResult<(Vec<Clbit>, Vec<Var>)> {
        let wires_from_expr = |node: &expr::Expr| -> PyResult<(Vec<Clbit>, Vec<Var>)> {
            let mut clbits = Vec::new();
            let mut vars: Vec<Var> = Vec::new();
            for var in node.vars() {
                match var {
                    expr::Var::Bit { bit } => {
                        clbits.push(self.clbits.find(bit).unwrap());
                    }
                    expr::Var::Register { register, .. } => {
                        for bit in register.bits() {
                            clbits.push(self.clbits.find(&bit).unwrap());
                        }
                    }
                    expr::Var::Standalone { .. } => {
                        vars.push(self.vars_stretches.vars().find(var).unwrap())
                    }
                }
            }
            Ok((clbits, vars))
        };

        let mut clbits = Vec::new();
        let mut vars = Vec::new();

        if let Some(instr) = self.try_view_control_flow(instr) {
            match instr {
                ControlFlowView::IfElse { condition, .. }
                | ControlFlowView::While { condition, .. } => match condition {
                    Condition::Bit(bit, _) => {
                        clbits.push(self.clbits.find(bit).unwrap());
                    }
                    Condition::Register(reg, _) => {
                        for bit in reg.bits() {
                            clbits.push(self.clbits.find(&bit).unwrap());
                        }
                    }
                    Condition::Expr(condition) => {
                        let (expr_clbits, expr_vars) = wires_from_expr(condition)?;
                        for bit in expr_clbits {
                            clbits.push(bit);
                        }
                        for var in expr_vars {
                            vars.push(var);
                        }
                    }
                },
                ControlFlowView::Switch { target, .. } => match target {
                    SwitchTarget::Bit(bit) => {
                        clbits.push(self.clbits.find(bit).unwrap());
                    }
                    SwitchTarget::Register(reg) => {
                        for bit in reg.bits() {
                            clbits.push(self.clbits.find(&bit).unwrap());
                        }
                    }
                    SwitchTarget::Expr(expr) => {
                        let (expr_clbits, expr_vars) = wires_from_expr(expr)?;
                        for bit in expr_clbits {
                            clbits.push(bit);
                        }
                        for var in expr_vars {
                            vars.push(var);
                        }
                    }
                },

                // These don't have any classical wires
                ControlFlowView::Box { .. }
                | ControlFlowView::BreakLoop
                | ControlFlowView::ContinueLoop
                | ControlFlowView::ForLoop { .. } => {}
            }
            for block in instr.blocks() {
                for var in block.captured_vars() {
                    vars.push(self.vars_stretches.vars().find(var).unwrap());
                }
            }
        } else if let OperationRef::PyCustom(instr) = instr.op.view() {
            let op = instr.ob.bind(py);
            if op.is_instance(imports::STORE_OP.get_bound(py))? {
                let (expr_clbits, expr_vars) = wires_from_expr(&op.getattr("lvalue")?.extract()?)?;
                for bit in expr_clbits {
                    clbits.push(bit);
                }
                for var in expr_vars {
                    vars.push(var);
                }
                let (expr_clbits, expr_vars) = wires_from_expr(&op.getattr("rvalue")?.extract()?)?;
                for bit in expr_clbits {
                    clbits.push(bit);
                }
                for var in expr_vars {
                    vars.push(var);
                }
            }
        }
        Ok((clbits, vars))
    }

    /// Add a qubit or bit to the circuit.
    ///
    /// Args:
    ///     wire: the wire to be added
    ///
    ///     This adds a pair of in and out nodes connected by an edge.
    ///
    /// Returns:
    ///     The input and output node indices of the added wire, respectively.
    ///
    /// Raises:
    ///     DAGCircuitError: if trying to add duplicate wire
    fn add_wire(&mut self, wire: Wire) -> Result<(NodeIndex, NodeIndex), DuplicateWireError> {
        let (in_node, out_node) = match wire {
            Wire::Qubit(qubit) => {
                if qubit.index() < self.qubit_io_map.len() {
                    return Err(DuplicateWireError(wire));
                }
                let in_node = self.dag.add_node(NodeType::QubitIn(qubit));
                let out_node = self.dag.add_node(NodeType::QubitOut(qubit));
                self.qubit_io_map.push([in_node, out_node]);
                (in_node, out_node)
            }
            Wire::Clbit(clbit) => {
                if clbit.index() < self.clbit_io_map.len() {
                    return Err(DuplicateWireError(wire));
                }
                let in_node = self.dag.add_node(NodeType::ClbitIn(clbit));
                let out_node = self.dag.add_node(NodeType::ClbitOut(clbit));
                self.clbit_io_map.push([in_node, out_node]);
                (in_node, out_node)
            }
            Wire::Var(var) => {
                if var.index() < self.var_io_map.len() {
                    return Err(DuplicateWireError(wire));
                }
                let in_node = self.dag.add_node(NodeType::VarIn(var));
                let out_node = self.dag.add_node(NodeType::VarOut(var));
                self.var_io_map.push([in_node, out_node]);
                (in_node, out_node)
            }
        };
        self.dag.add_edge(in_node, out_node, wire);
        Ok((in_node, out_node))
    }

    /// Set the global phase to a float value.  Returns the old phase.
    ///
    /// Unlike the general [set_global_phase_param], this is infallible.
    #[inline]
    pub fn set_global_phase_f64(&mut self, angle: f64) -> Param {
        std::mem::replace(
            &mut self.global_phase,
            Param::Float(angle.rem_euclid(::std::f64::consts::TAU)),
        )
    }

    /// Set the global phase to a general [Param].  Returns the old phase.
    ///
    /// Unlike the general [set_global_phase_param], this is infallible.
    #[inline]
    pub fn set_global_phase_param(&mut self, angle: Param) -> Result<Param, DAGError> {
        match angle {
            Param::Float(angle) => Ok(self.set_global_phase_f64(angle)),
            Param::ParameterExpression(angle) => Ok(std::mem::replace(
                &mut self.global_phase,
                Param::ParameterExpression(angle),
            )),
            Param::Obj(_) => Err(DAGError::ObjGlobalPhase),
        }
    }

    /// Get the nodes on a given wire as a non-allocating iterator
    ///
    /// This will panic if the wire is not in the circuit
    pub fn nodes_on_wire(&self, wire: Wire) -> impl Iterator<Item = NodeIndex> {
        let start_node = match wire {
            Wire::Qubit(qubit) => self.qubit_io_map[qubit.index()][0],
            Wire::Clbit(clbit) => self.clbit_io_map[clbit.index()][0],
            Wire::Var(var) => self.var_io_map[var.index()][0],
        };
        NodesOnWireIter {
            dag: self,
            wire,
            next_node: Some(start_node),
        }
    }

    fn remove_idle_wire(&mut self, wire: Wire) {
        let [in_node, out_node] = match wire {
            Wire::Qubit(qubit) => self.qubit_io_map[qubit.index()],
            Wire::Clbit(clbit) => self.clbit_io_map[clbit.index()],
            Wire::Var(var) => self.var_io_map[var.index()],
        };
        self.dag.remove_node(in_node);
        self.dag.remove_node(out_node);
    }

    pub fn add_qubit_unchecked(
        &mut self,
        bit: ShareableQubit,
    ) -> Result<Qubit, DuplicateWireError> {
        let qubit = self.qubits.add_unique_within_capacity(bit.clone());
        self.qubit_locations
            .insert(bit, BitLocations::new((self.qubits.len() - 1) as u32, []));
        self.add_wire(Wire::Qubit(qubit))?;
        Ok(qubit)
    }

    pub fn add_clbit_unchecked(
        &mut self,
        bit: ShareableClbit,
    ) -> Result<Clbit, DuplicateWireError> {
        let clbit = self.clbits.add_unique_within_capacity(bit.clone());
        self.clbit_locations
            .insert(bit, BitLocations::new((self.clbits.len() - 1) as u32, []));
        self.add_wire(Wire::Clbit(clbit))?;
        Ok(clbit)
    }

    pub fn get_node(&self, py: Python, node: NodeIndex) -> PyResult<Py<PyAny>> {
        self.unpack_into(py, node, self.dag.node_weight(node).unwrap())
    }

    /// Remove an operation node n.
    ///
    /// Add edges from predecessors to successors.
    ///
    /// # Returns
    ///
    /// The removed [PackedInstruction] is returned
    pub fn remove_op_node(&mut self, index: NodeIndex) -> PackedInstruction {
        let mut edge_list: Vec<(NodeIndex, NodeIndex, Wire)> = Vec::new();
        for (source, in_weight) in self
            .dag
            .edges_directed(index, Incoming)
            .map(|x| (x.source(), *x.weight()))
        {
            for (target, out_weight) in self
                .dag
                .edges_directed(index, Outgoing)
                .map(|x| (x.target(), *x.weight()))
            {
                if in_weight == out_weight {
                    edge_list.push((source, target, in_weight));
                }
            }
        }
        for (source, target, weight) in edge_list {
            self.dag.add_edge(source, target, weight);
        }

        let Some(NodeType::Operation(packed)) = self.dag.remove_node(index) else {
            panic!("Must be called with valid operation node!");
        };
        self.untrack_instruction(&packed);
        packed
    }

    /// Returns an iterator of the ancestors indices of a node.
    pub fn ancestors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        core_ancestors(&self.dag, node).filter(move |next| next != &node)
    }

    /// Returns an iterator of the descendants of a node as DAGOpNodes and DAGOutNodes.
    pub fn descendants(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        core_descendants(&self.dag, node).filter(move |next| next != &node)
    }

    /// Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
    /// and [DAGNode] is its successors in  BFS order.
    pub fn bfs_successors(
        &self,
        node: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, Vec<NodeIndex>)> + '_ {
        core_bfs_successors(&self.dag, node).filter(move |(_, others)| !others.is_empty())
    }

    /// Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
    /// and [DAGNode] is its predecessors in BFS order.
    pub fn bfs_predecessors(
        &self,
        node: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, Vec<NodeIndex>)> + '_ {
        core_bfs_predecessors(&self.dag, node).filter(move |(_, others)| !others.is_empty())
    }

    fn pack_into(&mut self, py: Python, b: &Bound<PyAny>) -> Result<NodeType, PyErr> {
        Ok(if let Ok(in_node) = b.cast::<DAGInNode>() {
            let in_node = in_node.borrow();
            let wire = in_node.wire.bind(py);
            if let Ok(qubit) = wire.extract::<ShareableQubit>() {
                NodeType::QubitIn(self.qubits.find(&qubit).unwrap())
            } else if let Ok(clbit) = wire.extract::<ShareableClbit>() {
                NodeType::ClbitIn(self.clbits.find(&clbit).unwrap())
            } else {
                let var = wire.extract::<expr::Var>()?;
                NodeType::VarIn(self.vars_stretches.vars().find(&var).unwrap())
            }
        } else if let Ok(out_node) = b.cast::<DAGOutNode>() {
            let out_node = out_node.borrow();
            let wire = out_node.wire.bind(py);
            if let Ok(qubit) = wire.extract::<ShareableQubit>() {
                NodeType::QubitOut(self.qubits.find(&qubit).unwrap())
            } else if let Ok(clbit) = wire.extract::<ShareableClbit>() {
                NodeType::ClbitOut(self.clbits.find(&clbit).unwrap())
            } else {
                let var = wire.extract::<expr::Var>()?;
                NodeType::VarOut(self.vars_stretches.vars().find(&var).unwrap())
            }
        } else if let Ok(op_node) = b.cast::<DAGOpNode>() {
            let op_node = op_node.borrow();
            let qubits = self.qargs_interner.insert_owned(
                self.qubits
                    .map_objects(
                        op_node
                            .instruction
                            .qubits
                            .extract::<Vec<ShareableQubit>>(py)?,
                    )?
                    .collect(),
            );
            let clbits = self.cargs_interner.insert_owned(
                self.clbits
                    .map_objects(
                        op_node
                            .instruction
                            .clbits
                            .extract::<Vec<ShareableClbit>>(py)?,
                    )?
                    .collect(),
            );
            let params =
                self.extract_blocks_from_circuit_parameters(op_node.instruction.params.as_ref())?;
            let inst = PackedInstruction {
                op: op_node.instruction.operation.clone(),
                qubits,
                clbits,
                params,
                label: op_node.instruction.label.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: op_node.instruction.py_op.clone(),
            };
            NodeType::Operation(inst)
        } else {
            return Err(PyTypeError::new_err("Invalid type for DAGNode"));
        })
    }

    fn unpack_into(&self, py: Python, id: NodeIndex, weight: &NodeType) -> PyResult<Py<PyAny>> {
        let dag_node = match weight {
            NodeType::QubitIn(qubit) => Py::new(
                py,
                DAGInNode::new(id, self.qubits.get(*qubit).unwrap().into_py_any(py)?),
            )?
            .into_any(),
            NodeType::QubitOut(qubit) => Py::new(
                py,
                DAGOutNode::new(id, self.qubits.get(*qubit).unwrap().into_py_any(py)?),
            )?
            .into_any(),
            NodeType::ClbitIn(clbit) => Py::new(
                py,
                DAGInNode::new(id, self.clbits.get(*clbit).unwrap().into_py_any(py)?),
            )?
            .into_any(),
            NodeType::ClbitOut(clbit) => Py::new(
                py,
                DAGOutNode::new(id, self.clbits.get(*clbit).unwrap().into_py_any(py)?),
            )?
            .into_any(),
            NodeType::Operation(packed) => {
                let qubits = self.qargs_interner.get(packed.qubits);
                let clbits = self.cargs_interner.get(packed.clbits);
                let params = self.unpack_blocks_to_circuit_parameters(packed.params.as_deref())?;
                Py::new(
                    py,
                    (
                        DAGOpNode {
                            instruction: CircuitInstruction {
                                operation: packed.op.clone(),
                                qubits: PyTuple::new(py, self.qubits.map_indices(qubits))?.unbind(),
                                clbits: PyTuple::new(py, self.clbits.map_indices(clbits))?.unbind(),
                                params,
                                label: packed.label.clone(),
                                #[cfg(feature = "cache_pygates")]
                                py_op: packed.py_op.clone(),
                            },
                        },
                        DAGNode { node: Some(id) },
                    ),
                )?
                .into_any()
            }
            NodeType::VarIn(var) => Py::new(
                py,
                DAGInNode::new(
                    id,
                    self.vars_stretches
                        .vars()
                        .get(*var)
                        .unwrap()
                        .clone()
                        .into_py_any(py)?,
                ),
            )?
            .into_any(),
            NodeType::VarOut(var) => Py::new(
                py,
                DAGOutNode::new(
                    id,
                    self.vars_stretches
                        .vars()
                        .get(*var)
                        .unwrap()
                        .clone()
                        .into_py_any(py)?,
                ),
            )?
            .into_any(),
        };
        Ok(dag_node)
    }

    /// An iterator of the DAG indices and corresponding `PackedInstruction` references for
    /// the `NodeType::Operation` variants stored in the DAG.
    ///
    /// See also [op_node_indices], which provides only the indices.
    pub fn op_nodes(
        &self,
        include_directives: bool,
    ) -> impl Iterator<Item = (NodeIndex, &PackedInstruction)> + '_ {
        self.dag
            .node_references()
            .filter_map(move |(node_index, node_type)| match node_type {
                NodeType::Operation(node) => {
                    (include_directives || !node.op.directive()).then_some((node_index, node))
                }
                _ => None,
            })
    }

    /// An iterator of the DAG indices corresponding to `NodeType::Operation` variants.
    ///
    /// See also [op_nodes], which also provides a reference to the contained `PackedInstruction`.
    pub fn op_node_indices(
        &self,
        include_directives: bool,
    ) -> impl Iterator<Item = NodeIndex> + '_ {
        self.op_nodes(include_directives).map(|(index, _)| index)
    }

    /// Return an iterator of 2 qubit operations. Ignore directives like snapshot and barrier.
    pub fn two_qubit_ops(&self) -> impl Iterator<Item = (NodeIndex, &PackedInstruction)> + '_ {
        self.op_nodes(false)
            .filter(|(_, instruction)| self.qargs_interner.get(instruction.qubits).len() == 2)
    }

    // Filter any nodes that don't match a given predicate function
    pub fn filter_op_nodes<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&PackedInstruction) -> bool,
    {
        let remove_indices = self
            .op_nodes(true)
            .filter_map(|(index, instruction)| (!predicate(instruction)).then_some(index))
            .collect::<Vec<_>>();
        for node in remove_indices {
            self.remove_op_node(node);
        }
    }

    /// Returns an iterator over a list layers of the `DAGCircuit``.
    pub fn multigraph_layers(&self) -> impl Iterator<Item = Vec<NodeIndex>> + '_ {
        let mut first_layer: Vec<_> = self.qubit_io_map.iter().map(|x| x[0]).collect();
        first_layer.extend(self.clbit_io_map.iter().map(|x| x[0]));
        first_layer.extend(self.var_io_map.iter().map(|x| x[0]));
        // A DAG is by definition acyclical, therefore unwrapping the layer should never fail.
        layers(&self.dag, first_layer).map(|layer| match layer {
            Ok(layer) => layer,
            Err(_) => unreachable!("Not a DAG."),
        })
    }

    /// Returns an iterator over the first layer of the `DAGCircuit``.
    pub fn front_layer(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        let mut graph_layers = self.multigraph_layers();
        graph_layers.next();
        graph_layers
            .next()
            .into_iter()
            .flatten()
            .filter(|node| matches!(self.dag.node_weight(*node).unwrap(), NodeType::Operation(_)))
    }

    /// Substitutes a node by a given DAGCircuit and adds the replacement DAG's global phase to the current DAG.
    ///
    /// # Arguments
    ///
    /// * node_index: The node in the DAGCircuit to replace.
    /// * other: The replacement DAGCircuit.
    /// * qubit_map: A mapping from the replacement DAGCircuit qubits to the replaced node's qargs.
    ///   If None, trivial mapping will be used.
    /// * clbit_map: A mapping from the replacement DAGCircuit clbits to the replaced node's clbits.
    ///   If None, trivial mapping will be used.
    /// * var_map: A mapping from the replacement DAGCircuit variables to the replaced node's variables.
    ///   Note: Inferring variable mapping automatically is currently not implemented.
    /// * block_map: A mapping from the replacement DAGCircuit block IDs to the block IDs of this
    ///   DAGCircuit. If None, and the replacement DAG has at least one block, the replacement DAG
    ///   is assumed to use the same block IDs as this DAG, and must have the same number of blocks.
    ///
    /// # Returns
    ///
    /// A mapping of the node indices in the replacement DAGCircuit to their corresponding node indices in the
    /// current DAGCircuit.
    pub fn substitute_node_with_dag(
        &mut self,
        node_index: NodeIndex,
        other: &DAGCircuit,
        qubit_map: Option<&HashMap<Qubit, Qubit>>,
        clbit_map: Option<&HashMap<Clbit, Clbit>>,
        var_map: Option<&HashMap<expr::Var, expr::Var>>,
        block_map: Option<&HashMap<Block, Block>>,
    ) -> Result<IndexMap<NodeIndex, NodeIndex>, DAGError> {
        if self.dag.node_weight(node_index).is_none() {
            return Err(DAGError::NodeNotInGraph(node_index));
        }

        let node = match &self.dag[node_index] {
            NodeType::Operation(op) => op.clone(),
            _ => return Err(format!("Expected op-node for node {node_index:?}").into()),
        };

        let qubit_map = match qubit_map {
            Some(qubit_map) => qubit_map,
            None => {
                let node_qubits = self.qargs_interner.get(node.qubits);
                let other_qubits = (0..other.num_qubits()).map(Qubit::new);
                if node_qubits.len() != other_qubits.len() {
                    return Err(format!(
                        "Replacement DAG has {} qubits, expected {}",
                        other_qubits.len(),
                        node_qubits.len()
                    )
                    .into());
                }
                &HashMap::<Qubit, Qubit>::from_iter(other_qubits.zip(node_qubits.iter().copied()))
            }
        };

        let clbit_map = match clbit_map {
            Some(clbit_map) => clbit_map,
            None => {
                let node_clbits = self.cargs_interner.get(node.clbits);
                let other_clbits = (0..other.num_clbits()).map(Clbit::new);
                if node_clbits.len() != other_clbits.len() {
                    return Err(format!(
                        "Replacement DAG has {} clbits, expected {}",
                        other_clbits.len(),
                        node_clbits.len()
                    )
                    .into());
                }
                &HashMap::<Clbit, Clbit>::from_iter(other_clbits.zip(node_clbits.iter().copied()))
            }
        };

        let var_map = match var_map {
            Some(var_map) => var_map,
            None => {
                if self.num_vars() > 0 || other.num_vars() > 0 {
                    unimplemented!(
                        "Inferring variable mapping in substitute_node_with_dag is not implemented yet. Consider using py_substitute_node_with_dag instead."
                    );
                    // TODO: implement once additional_wires becomes Python-free
                }
                &HashMap::<expr::Var, expr::Var>::new()
            }
        };

        let block_map = match block_map {
            Some(block_map) => block_map,
            None => {
                if other.num_blocks() > 0 && self.num_blocks() != other.num_blocks() {
                    panic!(
                        "'block_map' must be provided when 'other' has a different non-empty set of blocks"
                    );
                }
                let self_blocks = (0..self.num_blocks()).map(Block::new);
                let other_blocks = (0..other.num_blocks()).map(Block::new);
                &HashMap::from_iter(other_blocks.zip(self_blocks))
            }
        };

        let out_map = self.substitute_node_with_graph(
            node_index, other, qubit_map, clbit_map, var_map, block_map,
        )?;
        self.global_phase = add_global_phase(&self.global_phase, &other.global_phase);

        let mut wire_map_dict = HashMap::new();
        for (source, target) in clbit_map.iter() {
            let source_bit = other.clbits.get(*source);
            let target_bit = self.clbits.get(*target);
            wire_map_dict.insert(source_bit.cloned().unwrap(), target_bit.cloned().unwrap());
        }

        let variable_mapper = VariableMapper::new(
            self.cregs.registers().to_vec(),
            wire_map_dict,
            var_map.clone(),
            None,
        );

        for (old_node_index, new_node_index) in out_map.iter() {
            let old_node = &other.dag[*old_node_index];
            if let NodeType::Operation(old_inst) = old_node {
                if let OperationRef::ControlFlow(cf) = old_inst.op.view() {
                    match &cf.control_flow {
                        ControlFlow::Switch {
                            target,
                            label_spec,
                            cases,
                        } => {
                            let mapped_target = variable_mapper
                                .map_target(target, |new_reg| self.add_creg(new_reg.clone()))?;

                            if let NodeType::Operation(new_inst) = &mut self.dag[*new_node_index] {
                                #[cfg(feature = "cache_pygates")]
                                {
                                    new_inst.py_op.take();
                                }
                                new_inst.op = ControlFlowInstruction {
                                    control_flow: ControlFlow::Switch {
                                        target: mapped_target,
                                        label_spec: label_spec.clone(),
                                        cases: *cases,
                                    },
                                    num_qubits: cf.num_qubits,
                                    num_clbits: cf.num_clbits,
                                }
                                .into();
                            }
                        }
                        ControlFlow::IfElse { condition } | ControlFlow::While { condition } => {
                            let mapped_condition =
                                variable_mapper.map_condition(condition, false, |new_reg| {
                                    self.add_creg(new_reg.clone())
                                })?;

                            if let NodeType::Operation(new_inst) = &mut self.dag[*new_node_index] {
                                #[cfg(feature = "cache_pygates")]
                                {
                                    new_inst.py_op.take();
                                }
                                if matches!(&cf.control_flow, ControlFlow::While { .. }) {
                                    new_inst.op = ControlFlowInstruction {
                                        control_flow: ControlFlow::While {
                                            condition: mapped_condition,
                                        },
                                        num_qubits: cf.num_qubits,
                                        num_clbits: cf.num_clbits,
                                    }
                                    .into();
                                } else {
                                    new_inst.op = ControlFlowInstruction {
                                        control_flow: ControlFlow::IfElse {
                                            condition: mapped_condition,
                                        },
                                        num_qubits: cf.num_qubits,
                                        num_clbits: cf.num_clbits,
                                    }
                                    .into();
                                }
                            }
                        }
                        _ => (),
                    }
                }
            }
        }
        Ok(out_map)
    }

    fn substitute_node_with_graph(
        &mut self,
        node: NodeIndex,
        other: &DAGCircuit,
        qubit_map: &HashMap<Qubit, Qubit>,
        clbit_map: &HashMap<Clbit, Clbit>,
        var_map: &HashMap<expr::Var, expr::Var>,
        block_map: &HashMap<Block, Block>,
    ) -> Result<IndexMap<NodeIndex, NodeIndex>, DAGError> {
        if self.dag.node_weight(node).is_none() {
            return Err(DAGError::NodeNotInGraph(node));
        }

        // Add wire from pred to succ if no ops on mapped wire on ``other``
        for (in_dag_wire, self_wire) in qubit_map.iter() {
            let [input_node, out_node] = other.qubit_io_map[in_dag_wire.index()];
            if other.dag.find_edge(input_node, out_node).is_some() {
                let pred = self
                    .dag
                    .edges_directed(node, Incoming)
                    .find(|edge| {
                        if let Wire::Qubit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                let succ = self
                    .dag
                    .edges_directed(node, Outgoing)
                    .find(|edge| {
                        if let Wire::Qubit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                self.dag
                    .add_edge(pred.source(), succ.target(), Wire::Qubit(*self_wire));
            }
        }
        for (in_dag_wire, self_wire) in clbit_map.iter() {
            let [input_node, out_node] = other.clbit_io_map[in_dag_wire.index()];
            if other.dag.find_edge(input_node, out_node).is_some() {
                let pred = self
                    .dag
                    .edges_directed(node, Incoming)
                    .find(|edge| {
                        if let Wire::Clbit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                let succ = self
                    .dag
                    .edges_directed(node, Outgoing)
                    .find(|edge| {
                        if let Wire::Clbit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                self.dag
                    .add_edge(pred.source(), succ.target(), Wire::Clbit(*self_wire));
            }
        }

        let node_filter = |node: NodeIndex| -> bool {
            match other.dag[node] {
                NodeType::Operation(_) => !other
                    .dag
                    .edges_directed(node, petgraph::Direction::Outgoing)
                    .any(|edge| match edge.weight() {
                        Wire::Qubit(qubit) => !qubit_map.contains_key(qubit),
                        Wire::Clbit(clbit) => !clbit_map.contains_key(clbit),
                        Wire::Var(var) => {
                            !var_map.contains_key(other.vars_stretches.vars().get(*var).unwrap())
                        }
                    }),
                _ => false,
            }
        };
        let reverse_qubit_map: HashMap<Qubit, Qubit> =
            qubit_map.iter().map(|(x, y)| (*y, *x)).collect();
        let reverse_clbit_map: HashMap<Clbit, Clbit> =
            clbit_map.iter().map(|(x, y)| (*y, *x)).collect();
        let reverse_var_map: HashMap<&expr::Var, &expr::Var> =
            var_map.iter().map(|(x, y)| (y, x)).collect();
        // Copy nodes from other to self
        let mut out_map: IndexMap<NodeIndex, NodeIndex> =
            IndexMap::with_capacity_and_hasher(other.dag.node_count(), RandomState::default());
        for old_index in other.dag.node_indices() {
            if !node_filter(old_index) {
                continue;
            }
            let mut new_node = other.dag[old_index].clone();
            if let NodeType::Operation(ref mut new_inst) = new_node {
                let new_qubit_indices: Vec<Qubit> = other
                    .qargs_interner
                    .get(new_inst.qubits)
                    .iter()
                    .map(|old_qubit| qubit_map[old_qubit])
                    .collect();
                let new_clbit_indices: Vec<Clbit> = other
                    .cargs_interner
                    .get(new_inst.clbits)
                    .iter()
                    .map(|old_clbit| clbit_map[old_clbit])
                    .collect();
                new_inst.qubits = self.qargs_interner.insert_owned(new_qubit_indices);
                new_inst.clbits = self.cargs_interner.insert_owned(new_clbit_indices);
                new_inst.params = new_inst
                    .params
                    .as_ref()
                    .map(|params| Box::new(params.map_blocks_ref(|b| block_map[b])));
                self.track_instruction(new_inst);
            }
            let new_index = self.dag.add_node(new_node);
            out_map.insert(old_index, new_index);
        }
        // If no nodes are copied bail here since there is nothing left
        // to do.
        if out_map.is_empty() {
            match self.dag.remove_node(node) {
                Some(NodeType::Operation(packed)) => self.untrack_instruction(&packed),
                _ => panic!("node must be an op node"),
            }
            // Return a new empty map to clear allocation from out_map
            return Ok(IndexMap::default());
        }
        // Copy edges from other to self
        for edge in other.dag.edge_references().filter(|edge| {
            out_map.contains_key(&edge.target()) && out_map.contains_key(&edge.source())
        }) {
            self.dag.add_edge(
                out_map[&edge.source()],
                out_map[&edge.target()],
                match edge.weight() {
                    Wire::Qubit(qubit) => Wire::Qubit(qubit_map[qubit]),
                    Wire::Clbit(clbit) => Wire::Clbit(clbit_map[clbit]),
                    Wire::Var(var) => Wire::Var(
                        self.vars_stretches
                            .vars()
                            .find(
                                var_map
                                    .get(other.vars_stretches.vars().get(*var).unwrap())
                                    .unwrap(),
                            )
                            .unwrap(),
                    ),
                },
            );
        }
        // Add edges to/from node to nodes in other
        let edges: Vec<(NodeIndex, NodeIndex, Wire)> = self
            .dag
            .edges_directed(node, Incoming)
            .map(|x| (x.source(), x.target(), *x.weight()))
            .collect();
        for (source, _target, weight) in edges {
            let wire_input_id = match weight {
                Wire::Qubit(qubit) => other
                    .qubit_io_map
                    .get(reverse_qubit_map[&qubit].index())
                    .map(|x| x[0]),
                Wire::Clbit(clbit) => other
                    .clbit_io_map
                    .get(reverse_clbit_map[&clbit].index())
                    .map(|x| x[0]),
                Wire::Var(var) => {
                    let index = other
                        .vars_stretches
                        .vars()
                        .find(reverse_var_map[self.vars_stretches.vars().get(var).unwrap()])
                        .unwrap()
                        .index();
                    other.var_io_map.get(index).map(|x| x[0])
                }
            };
            let old_index =
                wire_input_id.and_then(|x| other.dag.neighbors_directed(x, Outgoing).next());
            let target_out = match old_index {
                Some(old_index) => match out_map.get(&old_index) {
                    Some(new_index) => *new_index,
                    None => {
                        // If the index isn't in the node map we've already added the edges as
                        // part of the idle wire handling at the top of this method so just
                        // move on.
                        continue;
                    }
                },
                None => continue,
            };
            self.dag.add_edge(source, target_out, weight);
        }
        let edges: Vec<(NodeIndex, NodeIndex, Wire)> = self
            .dag
            .edges_directed(node, Outgoing)
            .map(|x| (x.source(), x.target(), *x.weight()))
            .collect();
        for (_source, target, weight) in edges {
            let wire_output_id = match weight {
                Wire::Qubit(qubit) => other
                    .qubit_io_map
                    .get(reverse_qubit_map[&qubit].index())
                    .map(|x| x[1]),
                Wire::Clbit(clbit) => other
                    .clbit_io_map
                    .get(reverse_clbit_map[&clbit].index())
                    .map(|x| x[1]),
                Wire::Var(var) => {
                    let index = other
                        .vars_stretches
                        .vars()
                        .find(reverse_var_map[self.vars_stretches.vars().get(var).unwrap()])
                        .unwrap()
                        .index();
                    other.var_io_map.get(index).map(|x| x[1])
                }
            };
            let old_index =
                wire_output_id.and_then(|x| other.dag.neighbors_directed(x, Incoming).next());
            let source_out = match old_index {
                Some(old_index) => match out_map.get(&old_index) {
                    Some(new_index) => *new_index,
                    None => {
                        // If the index isn't in the node map we've already added the edges as
                        // part of the idle wire handling at the top of this method so just
                        // move on.
                        continue;
                    }
                },
                None => continue,
            };
            self.dag.add_edge(source_out, target, weight);
        }
        // Remove node
        if let NodeType::Operation(inst) = self
            .dag
            .remove_node(node)
            .expect("node is already checked for presence")
        {
            self.untrack_instruction(&inst);
        }
        Ok(out_map)
    }

    /// Retrieve a variable given its unique [Var] key within the DAG.
    ///
    /// The provided [Var] must be from this [DAGCircuit].
    pub fn get_var(&self, var: Var) -> Option<&expr::Var> {
        self.vars_stretches.vars().get(var)
    }

    /// Retrieve a stretch given its unique [Stretch] key within the DAG.
    ///
    /// The provided [Stretch] must be from this [DAGCircuit].
    pub fn get_stretch(&self, stretch: Stretch) -> Option<&expr::Stretch> {
        self.vars_stretches.stretches().get(stretch)
    }

    /// Return an immutable view of the vars and stretches in the DAGCircuit
    pub fn vars_stretches_view(&self) -> &VarStretchContainer {
        &self.vars_stretches
    }

    /// Adds a variable to the DAGCircuit.
    ///
    /// # Arguments:
    ///
    /// * `var`: the new variable to add.
    /// * `var_type`: the type the variable should have in the DAGCircuit.
    ///
    /// # Returns:
    ///
    /// The [Var] index of the variable in the DAGCircuit.
    fn add_var(&mut self, var: expr::Var, var_type: VarType) -> Result<Var, DAGError> {
        // The setup of the initial graph structure between an "in" and an "out" node is the same as
        // the bit-related `_add_wire`, but this logically needs to do different bookkeeping around
        // tracking the properties

        let var_idx = self.vars_stretches.add_var(var, var_type)?;
        self.add_wire(Wire::Var(var_idx))?;
        Ok(var_idx)
    }

    /// Add a stretch variable to the DAGCircuit.
    ///
    /// # Arguments:
    ///
    /// * `stretch`: the new stretch to add.
    /// * `stretch_type`: the type the stretch should have in the DAGCircuit.
    ///
    /// # Returns:
    ///
    /// The [Stretch] index of the stretch in the DAGCircuit.
    fn add_stretch(
        &mut self,
        stretch: expr::Stretch,
        stretch_type: StretchType,
    ) -> Result<(), DAGError> {
        self.vars_stretches.add_stretch(stretch, stretch_type)?;
        Ok(())
    }

    fn check_op_addition(&self, inst: &PackedInstruction) -> Result<(), DAGError> {
        for b in self.qargs_interner.get(inst.qubits) {
            if self.qubit_io_map.len() - 1 < b.index() {
                return Err(DAGError::WireNotInOutput(ShareableWire::Qubit(
                    self.qubits.get(*b).cloned().unwrap(),
                )));
            }
        }

        for b in self.cargs_interner.get(inst.clbits) {
            if !self.clbit_io_map.len() - 1 < b.index() {
                return Err(DAGError::WireNotInOutput(ShareableWire::Clbit(
                    self.clbits.get(*b).cloned().unwrap(),
                )));
            }
        }

        if self.may_have_additional_wires(inst) {
            let (clbits, vars) =
                Python::attach(|py| self.additional_wires(py, inst).map_err(DAGError::Python))?;
            for b in clbits {
                if !self.clbit_io_map.len() - 1 < b.index() {
                    return Err(DAGError::WireNotInOutput(ShareableWire::Clbit(
                        self.clbits.get(b).cloned().unwrap(),
                    )));
                }
            }
            for v in vars {
                if !self.var_io_map.len() - 1 < v.index() {
                    return Err(DAGError::WireNotInOutput(ShareableWire::Var(v)));
                }
            }
        }
        Ok(())
    }

    /// Alternative constructor, builds a DAGCircuit with a fixed capacity.
    ///
    /// # Arguments:
    /// - `num_qubits`: Number of qubits in the circuit
    /// - `num_clbits`: Number of classical bits in the circuit.
    /// - `num_vars`: (Optional) number of variables in the circuit.
    /// - `num_ops`: (Optional) number of operations in the circuit.
    /// - `num_edges`: (Optional) number of edges in the circuit.
    /// - `num_stretches`: (Optional) number of stretches in the circuit.
    pub fn with_capacity(
        num_qubits: usize,
        num_clbits: usize,
        num_vars: Option<usize>,
        num_ops: Option<usize>,
        num_edges: Option<usize>,
        num_stretches: Option<usize>,
    ) -> Self {
        let num_ops: usize = num_ops.unwrap_or_default();
        let num_vars = num_vars.unwrap_or_default();
        let num_stretches = num_stretches.unwrap_or_default();
        let num_edges = num_edges.unwrap_or(
            num_qubits +    // 1 edge between the input node and the output node or 1st op node.
            num_clbits +    // 1 edge between the input node and the output node or 1st op node.
            num_vars +      // 1 edge between the input node and the output node or 1st op node.
            num_ops, // In Average there will be 3 edges (2 qubits and 1 clbit, or 3 qubits) per op_node.
        );

        let num_nodes = num_qubits * 2 + // One input + One output node per qubit
            num_clbits * 2 +    // One input + One output node per clbit
            num_vars * 2 +  // One input + output node per variable
            num_ops;

        Self {
            name: None,
            metadata: None,
            dag: StableDiGraph::with_capacity(num_nodes, num_edges),
            qregs: RegisterData::new(),
            cregs: RegisterData::new(),
            qargs_interner: Interner::with_capacity(num_qubits),
            cargs_interner: Interner::with_capacity(num_clbits),
            qubits: ObjectRegistry::with_capacity(num_qubits),
            clbits: ObjectRegistry::with_capacity(num_clbits),
            blocks: ControlFlowBlocks::new(),
            vars_stretches: VarStretchContainer::with_capacity(Some(num_vars), Some(num_stretches)),
            global_phase: Param::Float(0.),
            duration: None,
            unit: "dt".to_string(),
            qubit_locations: BitLocator::with_capacity(num_qubits),
            clbit_locations: BitLocator::with_capacity(num_clbits),
            qubit_io_map: Vec::with_capacity(num_qubits),
            clbit_io_map: Vec::with_capacity(num_clbits),
            var_io_map: Vec::with_capacity(num_vars),
            op_names: IndexMap::default(),
        }
    }

    /// Get qargs from an intern index
    pub fn get_qargs(&self, index: Interned<[Qubit]>) -> &[Qubit] {
        self.qargs_interner.get(index)
    }

    /// Insert qargs into the interner and return the interned value
    pub fn add_qargs(&mut self, qubits: &[Qubit]) -> Interned<[Qubit]> {
        self.qargs_interner.insert(qubits)
    }

    /// Get cargs from an intern index
    pub fn get_cargs(&self, index: Interned<[Clbit]>) -> &[Clbit] {
        self.cargs_interner.get(index)
    }

    /// Insert a new 1q standard gate on incoming qubit
    pub fn insert_1q_on_incoming_qubit(
        &mut self,
        new_gate: (StandardGate, &[f64]),
        old_index: NodeIndex,
    ) {
        let inst = if let NodeType::Operation(old_node) = &self.dag[old_index] {
            PackedInstruction {
                op: new_gate.0.into(),
                qubits: old_node.qubits,
                clbits: old_node.clbits,
                params: (!new_gate.1.is_empty()).then(|| {
                    Box::new(Parameters::Params(
                        new_gate.1.iter().map(|x| Param::Float(*x)).collect(),
                    ))
                }),
                label: None,
                #[cfg(feature = "cache_pygates")]
                py_op: OnceLock::new(),
            }
        } else {
            panic!("This method only works if provided index is an op node");
        };
        self.track_instruction(&inst);
        let new_index = self.dag.add_node(NodeType::Operation(inst));
        let (parent_index, edge_index, weight) = self
            .dag
            .edges_directed(old_index, Incoming)
            .map(|edge| (edge.source(), edge.id(), *edge.weight()))
            .next()
            .unwrap();
        self.dag.add_edge(parent_index, new_index, weight);
        self.dag.add_edge(new_index, old_index, weight);
        self.dag.remove_edge(edge_index);
    }

    /// Remove a sequence of 1 qubit nodes from the dag
    /// This must only be called if all the nodes operate
    /// on a single qubit with no other wires in or out of any nodes
    pub fn remove_1q_sequence(&mut self, sequence: &[NodeIndex]) {
        let (parent_index, weight) = self
            .dag
            .edges_directed(*sequence.first().unwrap(), Incoming)
            .map(|edge| (edge.source(), *edge.weight()))
            .next()
            .unwrap();
        let child_index = self
            .dag
            .edges_directed(*sequence.last().unwrap(), Outgoing)
            .map(|edge| edge.target())
            .next()
            .unwrap();
        self.dag.add_edge(parent_index, child_index, weight);
        for node in sequence {
            match self.dag.remove_node(*node) {
                Some(NodeType::Operation(packed)) => self.untrack_instruction(&packed),
                _ => panic!("This method only works if provided index is an op node"),
            }
        }
    }

    /// Replace a node with individual operations from a provided callback
    /// function on each qubit of that node.
    pub fn replace_node_with_1q_ops<F>(&mut self, node: NodeIndex, insert: F)
    where
        F: Fn(Wire) -> (PackedOperation, SmallVec<[Param; 3]>),
    {
        let mut edge_list: Vec<(NodeIndex, NodeIndex, Wire)> = Vec::with_capacity(2);
        for (source, in_weight) in self
            .dag
            .edges_directed(node, Incoming)
            .map(|x| (x.source(), *x.weight()))
        {
            for (target, out_weight) in self
                .dag
                .edges_directed(node, Outgoing)
                .map(|x| (x.target(), *x.weight()))
            {
                if in_weight == out_weight {
                    edge_list.push((source, target, in_weight));
                }
            }
        }
        for (source, target, weight) in edge_list {
            let (new_op, params) = insert(weight);
            let qubits = if let Wire::Qubit(qubit) = weight {
                vec![qubit]
            } else {
                panic!(
                    "This method only works if the gate being replaced has no classical incident wires"
                )
            };
            let inst = PackedInstruction {
                op: new_op,
                qubits: self.qargs_interner.insert_owned(qubits),
                clbits: self.cargs_interner.get_default(),
                params: (!params.is_empty()).then(|| Box::new(Parameters::Params(params))),
                label: None,
                #[cfg(feature = "cache_pygates")]
                py_op: OnceLock::new(),
            };
            self.track_instruction(&inst);
            let new_index = self.dag.add_node(NodeType::Operation(inst));
            self.dag.add_edge(source, new_index, weight);
            self.dag.add_edge(new_index, target, weight);
        }

        match self.dag.remove_node(node) {
            Some(NodeType::Operation(packed)) => self.untrack_instruction(&packed),
            _ => panic!("Must be called with valid operation node"),
        }
    }

    pub fn add_global_phase(&mut self, value: &Param) -> Result<(), DAGError> {
        match value {
            Param::Obj(_) => Err(DAGError::ObjGlobalPhase),
            _ => self
                .set_global_phase_param(add_global_phase(&self.global_phase, value))
                .map(|_| ()),
        }
    }

    /// Return the op name counts in the circuit
    ///
    /// Args:
    ///     py: The python token necessary for control flow recursion
    ///     recurse: Whether to recurse into control flow ops or not
    pub fn count_ops(&self, recurse: bool) -> Result<IndexMap<String, usize>, DAGError> {
        if !recurse || !self.has_control_flow() {
            Ok(self.op_names.clone())
        } else {
            fn inner(
                dag: &DAGCircuit,
                counts: &mut IndexMap<String, usize>,
            ) -> Result<(), DAGError> {
                for (key, value) in dag.op_names.iter() {
                    counts
                        .entry(key.clone())
                        .and_modify(|count| *count += value)
                        .or_insert(*value);
                }
                for (_, node) in dag.op_nodes(false) {
                    let Some(control_flow) = dag.try_view_control_flow(node) else {
                        continue;
                    };
                    for block in control_flow.blocks() {
                        inner(block, counts)?;
                    }
                }
                Ok(())
            }
            let mut counts =
                IndexMap::with_capacity_and_hasher(self.op_names.len(), RandomState::default());
            inner(self, &mut counts)?;
            Ok(counts)
        }
    }

    /// Get an immutable reference to the op counts for this DAGCircuit
    ///
    /// This differs from count_ops() in that it doesn't handle control flow recursion at all
    /// and it returns a reference instead of an owned copy. If you don't need to work with
    /// control flow or ownership of the counts this is a more efficient alternative to
    /// `DAGCircuit::count_ops(py, false)`
    pub fn get_op_counts(&self) -> &IndexMap<String, usize> {
        &self.op_names
    }

    /// Extends the DAG with valid instances of [PackedInstruction].
    pub fn extend<I>(&mut self, iter: I) -> Result<Vec<NodeIndex>, DAGError>
    where
        I: IntoIterator<Item = PackedInstruction>,
    {
        self.try_extend(
            iter.into_iter()
                .map(|inst| -> Result<PackedInstruction, Infallible> { Ok(inst) }),
        )
    }

    /// Extends the DAG with valid instances of [PackedInstruction], where the iterator produces the
    /// results in a fallible manner.
    pub fn try_extend<I, E>(&mut self, iter: I) -> Result<Vec<NodeIndex>, DAGError>
    where
        I: IntoIterator<Item = Result<PackedInstruction, E>>,
        DAGError: From<E>,
    {
        let mut new_nodes = Vec::new();
        let mut replacement_dag = DAGCircuit::new();
        std::mem::swap(self, &mut replacement_dag);
        let mut dag_builder = replacement_dag.into_builder();
        for inst in iter {
            new_nodes.push(dag_builder.push_back(inst?)?);
        }
        *self = dag_builder.build();
        Ok(new_nodes)
    }

    /// Alternative constructor to build an instance of [DAGCircuit] from a `QuantumCircuit`.
    pub fn from_circuit(
        qc: QuantumCircuitData,
        copy_op: bool,
        qubit_order: Option<Vec<Qubit>>,
        clbit_order: Option<Vec<Clbit>>,
    ) -> Result<Self, DAGError> {
        // Extract necessary attributes
        let qc_data = qc.data;
        Self::from_circuit_data(
            &qc_data,
            copy_op,
            qc.name,
            qc.metadata.map(|x| x.unbind()),
            qubit_order,
            clbit_order,
        )
    }

    /// Builds a [DAGCircuit] based on an instance of [CircuitData].
    pub fn from_circuit_data(
        qc_data: &CircuitData,
        copy_op: bool,
        name: Option<String>,
        metadata: Option<Py<PyAny>>,
        qubit_order: Option<Vec<Qubit>>,
        clbit_order: Option<Vec<Clbit>>,
    ) -> Result<Self, DAGError> {
        let num_qubits = qc_data.num_qubits();
        let num_clbits = qc_data.num_clbits();
        let num_ops = qc_data.len();
        let num_vars = qc_data.vars_stretches_view().vars().len();
        let num_stretches = qc_data.vars_stretches_view().stretches().len();

        // Build DAGCircuit with capacity
        let mut new_dag = DAGCircuit::with_capacity(
            num_qubits,
            num_clbits,
            Some(num_vars),
            Some(num_ops),
            None,
            Some(num_stretches),
        );

        // Assign other necessary data
        new_dag.name = name;

        // Avoid manually acquiring the GIL.
        new_dag.global_phase = match qc_data.global_phase() {
            // The clone here implicitly requires the gil while ParameterExpression is defined in
            // Python.
            Param::ParameterExpression(exp) => Param::ParameterExpression(exp.clone()),
            Param::Float(float) => Param::Float(*float),
            _ => unreachable!("Incorrect parameter assigned for global phase"),
        };

        new_dag.metadata = metadata;

        // Add the qubits depending on order, and produce the qargs map.
        let qarg_map =
            if let Some(qubit_ordering) = qubit_order {
                let mut ordered_vec = Vec::from_iter((0..num_qubits as u32).map(Qubit));
                qubit_ordering
                    .into_iter()
                    .try_for_each(|qubit| -> Result<(), DAGError> {
                        let shareable_qubit = qc_data
                            .qubits()
                            .get(qubit)
                            .ok_or_else(|| DAGError::WireNotInCircuit(qubit.into()))?;

                        if new_dag.qubits.find(shareable_qubit).is_some() {
                            return Err(DuplicateWireError(qubit.into()).into());
                        }
                        let qubit_index = qc_data.qubits().find(shareable_qubit).unwrap();
                        ordered_vec[qubit_index.index()] =
                            new_dag.add_qubit_unchecked(shareable_qubit.clone())?;
                        Ok(())
                    })?;
                // The `Vec::get` use is because an arbitrary interner might contain old references to
                // bit instances beyond `num_qubits`, such as if it's from a DAG that had wires removed.
                new_dag.merge_qargs(qc_data.qargs_interner(), |bit| {
                    ordered_vec.get(bit.index()).copied()
                })
            } else {
                qc_data.qubits().objects().iter().try_for_each(
                    |qubit| -> Result<(), DAGError> {
                        new_dag.add_qubit_unchecked(qubit.clone())?;
                        Ok(())
                    },
                )?;
                new_dag.merge_qargs(qc_data.qargs_interner(), |bit| Some(*bit))
            };

        // Add the clbits depending on order, and produce the cargs map.
        let carg_map =
            if let Some(clbit_ordering) = clbit_order {
                let mut ordered_vec = Vec::from_iter((0..num_clbits as u32).map(Clbit));
                clbit_ordering
                    .into_iter()
                    .try_for_each(|clbit| -> Result<(), DAGError> {
                        let shareable_clbit = qc_data
                            .clbits()
                            .get(clbit)
                            .ok_or_else(|| DAGError::WireNotInCircuit(clbit.into()))?;

                        if new_dag.clbits.find(shareable_clbit).is_some() {
                            return Err(DuplicateWireError(clbit.into()).into());
                        };
                        let clbit_index = qc_data.clbits().find(shareable_clbit).unwrap();
                        ordered_vec[clbit_index.index()] =
                            new_dag.add_clbit_unchecked(shareable_clbit.clone())?;
                        Ok(())
                    })?;
                // The `Vec::get` use is because an arbitrary interner might contain old references to
                // bit instances beyond `num_clbits`, such as if it's from a DAG that had wires removed.
                new_dag.merge_cargs(qc_data.cargs_interner(), |bit| {
                    ordered_vec.get(bit.index()).copied()
                })
            } else {
                qc_data.clbits().objects().iter().try_for_each(
                    |clbit| -> Result<(), DAGError> {
                        new_dag.add_clbit_unchecked(clbit.clone())?;
                        Ok(())
                    },
                )?;
                new_dag.merge_cargs(qc_data.cargs_interner(), |bit| Some(*bit))
            };

        new_dag.vars_stretches = qc_data.vars_stretches_view().clone();
        new_dag.add_var_wires(new_dag.vars_stretches.vars().len())?;

        // Add all the registers
        for qreg in qc_data.qregs() {
            new_dag.add_qreg(qreg.clone())?;
        }

        for creg in qc_data.cregs() {
            new_dag.add_creg(creg.clone())?;
        }

        // After bits and registers are added, copy bitlocations
        new_dag.qubit_locations = qc_data.qubit_indices().clone();
        new_dag.clbit_locations = qc_data.clbit_indices().clone();
        new_dag.blocks = qc_data.blocks().try_map_without_references(|block| {
            DAGCircuit::from_circuit_data(block, copy_op, None, None, None, None)
        })?;
        new_dag.try_extend(qc_data.iter().map(|instr| -> Result<_, DAGError> {
            Ok(PackedInstruction {
                op: if copy_op {
                    instr
                        .op
                        .clone_with_py_deepcopy()
                        .map_err(DAGError::Python)?
                } else {
                    instr.op.clone()
                },
                qubits: qarg_map[instr.qubits],
                clbits: carg_map[instr.clbits],
                params: instr.params.clone(),
                label: instr.label.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: OnceLock::new(),
            })
        }))?;
        Ok(new_dag)
    }

    #[allow(clippy::too_many_arguments)]
    /// Replace a block of node indices with a new packed operation
    pub fn replace_block(
        &mut self,
        block_ids: &[NodeIndex],
        op: PackedOperation,
        params: Option<Box<Parameters<Block>>>,
        label: Option<&str>,
        cycle_check: bool,
        qubit_pos_map: &HashMap<Qubit, usize>,
        clbit_pos_map: &HashMap<Clbit, usize>,
    ) -> Result<NodeIndex, DAGError> {
        let mut block_qargs: HashSet<Qubit> = HashSet::new();
        let mut block_cargs: HashSet<Clbit> = HashSet::new();
        for nd in block_ids {
            let NodeType::Operation(instr) = self.dag.node_weight(*nd).ok_or_else(|| {
                DAGError::General("Node in 'node_block' not found in DAG.".to_string())
            })?
            else {
                return Err("Nodes in 'node_block' must be of type 'DAGOpNode'."
                    .to_string()
                    .into());
            };
            block_qargs.extend(self.qargs_interner.get(instr.qubits));
            block_cargs.extend(self.cargs_interner.get(instr.clbits));
            if self.may_have_additional_wires(instr) {
                let (additional_clbits, _) = Python::attach(|py| {
                    self.additional_wires(py, instr).map_err(DAGError::Python)
                })?;
                for clbit in additional_clbits {
                    block_cargs.insert(clbit);
                }
            }
        }

        let mut block_qargs: Vec<Qubit> = block_qargs
            .into_iter()
            .filter(|q| qubit_pos_map.contains_key(q))
            .collect();
        block_qargs.sort_by_key(|q| qubit_pos_map[q]);

        let mut block_cargs: Vec<Clbit> = block_cargs
            .into_iter()
            .filter(|c| clbit_pos_map.contains_key(c))
            .collect();
        block_cargs.sort_by_key(|c| clbit_pos_map[c]);

        if op.num_qubits() as usize != block_qargs.len() {
            return Err(format!(
                "Number of qubits in the replacement operation ({}) is not equal to the number of qubits in the block ({})!",
                op.num_qubits(),
                block_qargs.len()
            ).into());
        }

        let qubits = self.qargs_interner.insert_owned(block_qargs);
        let clbits = self.cargs_interner.insert_owned(block_cargs);
        let weight = NodeType::Operation(PackedInstruction {
            op,
            qubits,
            clbits,
            params,
            label: label.map(|label| Box::new(label.to_string())),
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        });

        // The `contract_nodes` call removes a bunch of weights from the DAG without returning them,
        // so we need to untrack them before making the call.  If the call fails, though, we have to
        // restore the tracking to those, so we can't let any `Block`s get garbage collected.  We'll
        // increment the refcount on any blocks in them now, and then undo it again before
        // returning.
        let extra_block_refs = block_ids
            .iter()
            .flat_map(|node| self.dag[*node].unwrap_operation().blocks_view())
            .copied()
            .inspect(|block| self.blocks.increment(*block))
            .collect::<Vec<_>>();
        block_ids
            .iter()
            .for_each(|node| self.untrack_instruction_in_place(*node));
        let out: Result<NodeIndex, DAGError> =
            match self
                .dag
                .contract_nodes(block_ids.iter().copied(), weight, cycle_check)
            {
                Ok(node) => {
                    self.track_instruction_in_place(node);
                    Ok(node)
                }
                Err(e) => {
                    for old in block_ids {
                        self.track_instruction_in_place(*old);
                    }
                    Err(match e {
                        ContractError::DAGWouldCycle => {
                            "Replacing the specified node block would introduce a cycle"
                                .to_string()
                                .into()
                        }
                    })
                }
            };
        extra_block_refs
            .into_iter()
            .for_each(|block| _ = self.blocks.decrement(block));
        out
    }

    pub fn compose(
        &mut self,
        other: &DAGCircuit,
        qubits: Option<&[Qubit]>,
        clbits: Option<&[Clbit]>,
        block_map: HashMap<Block, Block>,
        inline_captures: bool,
    ) -> Result<(), DAGError> {
        if other.qubits.len() > self.qubits.len() || other.clbits.len() > self.clbits.len() {
            if other.qubits.len() > self.qubits.len() {
                return Err(dag_compose_width_error(
                    "qubits",
                    other.qubits.len(),
                    self.qubits.len(),
                )
                .into());
            }
            return Err(dag_compose_width_error(
                "classical bits",
                other.clbits.len(),
                self.clbits.len(),
            )
            .into());
        }

        let qubit_map = match qubits {
            None => other
                .qubits
                .objects()
                .iter()
                .cloned()
                .zip(self.qubits.objects().iter().cloned())
                .collect(),
            Some(qubits) => {
                if qubits.len() != other.qubits.len() {
                    return Err(format!(
                        "Qubits parameter has {} qubits, expected {}",
                        qubits.len(),
                        other.qubits.len(),
                    )
                    .into());
                }
                let other_qubits = other.qubits.objects();

                other_qubits
                    .iter()
                    .cloned()
                    .zip(qubits.iter().map(|qubit| {
                        self.qubits
                            .get(*qubit)
                            .ok_or_else(|| DAGError::WireNotInCircuit((*qubit).into()))
                    }))
                    .map(|(other_qubit, self_qubit_result)| {
                        self_qubit_result.map(|self_qubit| (other_qubit, self_qubit.clone()))
                    })
                    .collect::<Result<HashMap<ShareableQubit, ShareableQubit>, DAGError>>()?
            }
        };

        let clbit_map = match clbits {
            None => other
                .clbits
                .objects()
                .iter()
                .cloned()
                .zip(self.clbits.objects().iter().cloned())
                .collect(),
            Some(clbits) => {
                if clbits.len() != other.clbits.len() {
                    return Err(format!(
                        "Clbits parameter has {} clbits, expected {}",
                        clbits.len(),
                        other.clbits.len(),
                    )
                    .into());
                }
                let other_clbits = other.clbits.objects();
                other_clbits
                    .iter()
                    .cloned()
                    .zip(clbits.iter().map(|clbit| {
                        self.clbits
                            .get(*clbit)
                            .ok_or_else(|| DAGError::WireNotInCircuit((*clbit).into()))
                    }))
                    .map(|(other_clbit, self_clbit_result)| {
                        self_clbit_result.map(|self_clbit| (other_clbit, self_clbit.clone()))
                    })
                    .collect::<Result<HashMap<ShareableClbit, ShareableClbit>, DAGError>>()?
            }
        };

        self.global_phase = add_global_phase(&self.global_phase, &other.global_phase);

        // This is all the handling we need for realtime variables, if there's no remapping. They:
        //
        // * get added to the DAG and then operations involving them get appended on normally.
        // * get inlined onto an existing variable, then operations get appended normally.
        // * there's a clash or a failed inlining, and we just raise an error.
        //
        // Notably if there's no remapping, there's no need to recurse into control-flow or to do any
        // Var rewriting during the Expr visits.
        for var in other.vars_stretches.iter_vars(VarType::Input) {
            self.add_var(var.clone(), VarType::Input)?;
        }
        if inline_captures {
            for var in other.vars_stretches.iter_vars(VarType::Capture) {
                if self.vars_stretches.vars().find(var).is_none() {
                    let expr::Var::Standalone { name, .. } = var else {
                        panic!("var capture not standalone");
                    };
                    return Err(DAGError::VariableNotInlined {
                        name: name.clone(),
                        is_var: true,
                    });
                }
            }
            for stretch in other.vars_stretches.iter_stretches(StretchType::Capture) {
                if self.vars_stretches.stretches().find(stretch).is_none() {
                    return Err(DAGError::VariableNotInlined {
                        name: stretch.name.clone(),
                        is_var: false,
                    });
                }
            }
        } else {
            for var in other.vars_stretches.iter_vars(VarType::Capture) {
                self.add_var(var.clone(), VarType::Capture)?;
            }
            for stretch in other.vars_stretches.iter_stretches(StretchType::Capture) {
                self.add_stretch(stretch.clone(), StretchType::Capture)?;
            }
        }
        for var in other.vars_stretches.iter_vars(VarType::Declare) {
            self.add_var(var.clone(), VarType::Declare)?;
        }
        for stretch in other.vars_stretches.iter_stretches(StretchType::Declare) {
            self.add_stretch(stretch.clone(), StretchType::Declare)?;
        }
        let build_var_mapper = |cregs: &RegisterData<ClassicalRegister>| -> VariableMapper {
            let mut edge_map = HashMap::new();
            for (a, b) in clbit_map.iter() {
                edge_map.insert(a.clone(), b.clone());
            }

            VariableMapper::new(
                cregs.registers().to_vec(),
                edge_map,
                HashMap::default(),
                None,
            )
        };
        let mut variable_mapper: Option<VariableMapper> = None;

        for node in other.topological_nodes(false) {
            match &other.dag[node] {
                NodeType::QubitIn(q) => {
                    let bit = other.qubits.get(*q).unwrap();
                    let m_wire = &qubit_map[bit];
                    let wire_in_dag = self.qubits.find(m_wire);
                    if wire_in_dag.is_none()
                        || (self.qubit_io_map.len() - 1 < wire_in_dag.unwrap().index())
                    {
                        return Err(DAGError::MissingWire(ShareableWire::Qubit(m_wire.clone())));
                    }
                }
                NodeType::ClbitIn(c) => {
                    let bit = other.clbits.get(*c).unwrap();
                    let m_wire = &clbit_map[bit];
                    let wire_in_dag = self.clbits.find(m_wire);
                    if wire_in_dag.is_none()
                        || self.clbit_io_map.len() - 1 < wire_in_dag.unwrap().index()
                    {
                        return Err(DAGError::MissingWire(ShareableWire::Clbit(m_wire.clone())));
                    }
                }
                NodeType::Operation(inst) => {
                    let qubits = other
                        .qubits
                        .map_indices(other.qargs_interner.get(inst.qubits));
                    let mapped_qargs = qubits
                        .into_iter()
                        .map(|bit| self.qubits.find(&qubit_map[bit]).unwrap())
                        .collect::<Vec<Qubit>>();
                    let clbits = other
                        .clbits
                        .map_indices(other.cargs_interner.get(inst.clbits));
                    let mapped_cargs = clbits
                        .into_iter()
                        .map(|bit| self.clbits.find(&clbit_map[bit]).unwrap())
                        .collect::<Vec<Clbit>>();

                    let instr = if let Some(cf) = inst.op.try_control_flow() {
                        let new_cf = ControlFlowInstruction {
                            control_flow: match &cf.control_flow {
                                ControlFlow::IfElse { condition }
                                | ControlFlow::While { condition } => {
                                    let mapped_condition = match variable_mapper {
                                        Some(ref variable_mapper) => variable_mapper
                                            .map_condition(condition, true, reject_new_register)?,
                                        None => {
                                            let var_mapper = build_var_mapper(&self.cregs);
                                            let condition = var_mapper.map_condition(
                                                condition,
                                                true,
                                                reject_new_register,
                                            )?;
                                            variable_mapper = Some(var_mapper);
                                            condition
                                        }
                                    };
                                    if matches!(&cf.control_flow, ControlFlow::While { .. }) {
                                        ControlFlow::While {
                                            condition: mapped_condition,
                                        }
                                    } else {
                                        ControlFlow::IfElse {
                                            condition: mapped_condition,
                                        }
                                    }
                                }
                                ControlFlow::Switch {
                                    target,
                                    label_spec,
                                    cases,
                                } => {
                                    let mapped_target = match variable_mapper {
                                        Some(ref variable_mapper) => variable_mapper
                                            .map_target(target, reject_new_register)?,
                                        None => {
                                            let var_mapper = build_var_mapper(&self.cregs);
                                            let target = var_mapper
                                                .map_target(target, reject_new_register)?;
                                            variable_mapper = Some(var_mapper);
                                            target
                                        }
                                    };
                                    ControlFlow::Switch {
                                        target: mapped_target,
                                        label_spec: label_spec.clone(),
                                        cases: *cases,
                                    }
                                }
                                _ => cf.control_flow.clone(),
                            },
                            num_qubits: cf.num_qubits,
                            num_clbits: cf.num_clbits,
                        };
                        PackedInstruction {
                            op: new_cf.into(),
                            qubits: self.qargs_interner.insert_owned(mapped_qargs),
                            clbits: self.cargs_interner.insert_owned(mapped_cargs),
                            params: inst
                                .params
                                .as_ref()
                                .map(|params| Box::new(params.map_blocks_ref(|b| block_map[b]))),
                            label: inst.label.clone(),
                            #[cfg(feature = "cache_pygates")]
                            py_op: OnceLock::new(),
                        }
                    } else {
                        PackedInstruction {
                            op: inst.op.clone(),
                            qubits: self.qargs_interner.insert_owned(mapped_qargs),
                            clbits: self.cargs_interner.insert_owned(mapped_cargs),
                            params: inst.params.clone(),
                            label: inst.label.clone(),
                            #[cfg(feature = "cache_pygates")]
                            py_op: inst.py_op.clone(),
                        }
                    };
                    self.push_back(instr)?;
                }
                // If its a Var wire, we already checked that it exists in the destination.
                NodeType::VarIn(_)
                | NodeType::VarOut(_)
                | NodeType::QubitOut(_)
                | NodeType::ClbitOut(_) => (),
            }
        }
        Ok(())
    }

    /// Substitute an operation in a node with a new one. The wire counts must match and the same
    /// argument order will be used.
    pub fn substitute_op(
        &mut self,
        node_index: NodeIndex,
        new_op: PackedOperation,
        params: Option<Parameters<Block>>,
        label: Option<&str>,
    ) -> Result<(), DAGError> {
        let old_packed = self.dag[node_index].unwrap_operation();

        if old_packed.op.num_qubits() != new_op.num_qubits()
            || old_packed.op.num_clbits() != new_op.num_clbits()
        {
            return Err(format!(
                "Cannot replace node of width ({} qubits, {} clbits) with operation of mismatched width ({} qubits, {} clbits)",
                old_packed.op.num_qubits(),
                old_packed.op.num_clbits(),
                new_op.num_qubits(),
                new_op.num_clbits()
            ).into());
        }
        let new_inst = PackedInstruction {
            op: new_op,
            qubits: old_packed.qubits,
            clbits: old_packed.clbits,
            params: params.map(Box::new),
            label: label.map(|label| Box::new(label.to_string())),
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        };
        self.track_instruction(&new_inst);
        let NodeType::Operation(old_packed) =
            ::std::mem::replace(&mut self.dag[node_index], NodeType::Operation(new_inst))
        else {
            unreachable!("already asserted this index is an op node during initial checks");
        };
        self.untrack_instruction(&old_packed);
        Ok(())
    }

    /// Substitute a give node in the dag with a new operation from python
    pub fn substitute_node_with_py_op(
        &mut self,
        node_index: NodeIndex,
        op: &Bound<PyAny>,
    ) -> PyResult<()> {
        // Extract information from new op
        let new_op = op.extract::<OperationFromPython<DAGCircuit>>()?;
        let params = self.take_parameter_blocks(new_op.params);
        // Extract information from node that is going to be replaced
        let old_packed = self.dag[node_index].unwrap_operation();
        let current_wires: HashSet<Wire> =
            self.dag.edges(node_index).map(|e| *e.weight()).collect();
        let mut new_wires: HashSet<Wire> = self
            .qargs_interner
            .get(old_packed.qubits)
            .iter()
            .map(|x| Wire::Qubit(*x))
            .chain(
                self.cargs_interner
                    .get(old_packed.clbits)
                    .iter()
                    .map(|x| Wire::Clbit(*x)),
            )
            .collect();

        if old_packed.op.num_qubits() != new_op.operation.num_qubits()
            || old_packed.op.num_clbits() != new_op.operation.num_clbits()
        {
            return Err(DAGCircuitError::new_err(format!(
                "Cannot replace node of width ({} qubits, {} clbits) with operation of mismatched width ({} qubits, {} clbits)",
                old_packed.op.num_qubits(),
                old_packed.op.num_clbits(),
                new_op.operation.num_qubits(),
                new_op.operation.num_clbits()
            )));
        }
        let label = new_op.label.clone();

        let new_instr = PackedInstruction {
            op: new_op.operation,
            qubits: old_packed.qubits,
            clbits: old_packed.clbits,
            params,
            label,
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::from(op.clone().unbind()),
        };

        let (additional_clbits, additional_vars) = Python::attach(|py| {
            self.additional_wires(py, &new_instr)
                .map_err(DAGError::Python)
        })?;
        new_wires.extend(additional_clbits.iter().map(|x| Wire::Clbit(*x)));
        new_wires.extend(additional_vars.iter().map(|x| Wire::Var(*x)));

        if new_wires != current_wires {
            // The new wires must be a non-strict subset of the current wires; if they add new
            // wires, we'd not know where to cut the existing wire to insert the new dependency.
            return Err(DAGCircuitError::new_err(format!(
                "New operation '{:?}' does not span the same wires as the old node '{:?}'. New wires: {:?}, old_wires: {:?}.",
                op.str(),
                old_packed.op.view(),
                new_wires,
                current_wires
            )));
        }
        self.track_instruction(&new_instr);
        let NodeType::Operation(old_instr) =
            ::std::mem::replace(&mut self.dag[node_index], NodeType::Operation(new_instr))
        else {
            unreachable!("already asserted this index is an op node during initial checks");
        };
        self.untrack_instruction(&old_instr);
        Ok(())
    }

    /// Returns version of the DAGCircuit optimized for efficient addition
    /// of multiple new instructions to the [DAGCircuit].
    pub fn into_builder(self) -> DAGCircuitBuilder {
        DAGCircuitBuilder::new(self)
    }

    // Returns an immutable reference to 'name', if it exists
    pub fn get_name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    // Returns an immutable reference to 'metadata'
    pub fn get_metadata(&self) -> Option<&Py<PyAny>> {
        self.metadata.as_ref()
    }

    /// Returns an iterator over the unique successors of the given node
    pub fn successors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> {
        self.dag.neighbors_directed(node, Outgoing).unique()
    }

    /// Returns an iterator over the unique predecessors of the given node
    pub fn predecessors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> {
        self.dag.neighbors_directed(node, Incoming).unique()
    }

    // Adds `num_vars` var wires in bulk
    fn add_var_wires(&mut self, num_vars: usize) -> Result<(), DAGError> {
        for var_idx in 0..num_vars {
            self.add_wire(Wire::Var(Var::new(var_idx)))?;
        }
        Ok(())
    }

    /// Build a new dag copy by iterating over this dag and building up nodes in the new dag from
    /// them.
    ///
    /// This function will build a new output dag builder with the same capacity and iterate over
    /// the nodes in this dag in a topological order. The given callback is called at each operation
    /// node with a mutable reference to the [`DAGCircuitBuilder`] which is where the new dag is built
    /// and a [`PackedInstruction`] of the current op node in the dag. The caller is responsible for
    /// adding any nodes to the dag as part of the callback.
    ///
    /// Typical use cases for this method are for transpiler passes that rebuild a dag by iterating over all
    /// nodes. The callback lets you handle removing or replacing a node (either with another single
    /// node or a sequence of nodes) while iterating over the original DAG.
    pub fn rebuild_dag_with<F, E>(
        &self,
        mut callback: F,
        vars_mode: VarsMode,
        blocks_mode: BlocksMode,
    ) -> Result<Self, E>
    where
        F: FnMut(&mut DAGCircuitBuilder, &PackedInstruction, NodeIndex) -> Result<(), E>,
    {
        let new_dag = self.copy_empty_like_with_same_capacity(vars_mode, blocks_mode);
        let mut builder = new_dag.into_builder();
        for node in
            petgraph::algo::toposort(&self.dag, None).expect("DAGCircuit can't have a cycle")
        {
            if let NodeType::Operation(ref inst) = self.dag[node] {
                callback(&mut builder, inst, node)?;
            }
        }
        Ok(builder.build())
    }
}

struct NodesOnWireIter<'a> {
    dag: &'a DAGCircuit,
    wire: Wire,
    next_node: Option<NodeIndex>,
}

impl<'a> Iterator for NodesOnWireIter<'a> {
    type Item = NodeIndex;

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.next_node.map(|n| &self.dag[n]) {
            // If the wire is empty there are the input and output nodes which is the minimum
            // If the dag is all on a single wire we would have all the operations + the 2 io
            // nodes.
            Some(NodeType::QubitIn(_) | NodeType::ClbitIn(_) | NodeType::VarIn(_)) => {
                (2, Some(self.dag.num_ops() + 2))
            }
            // If the iterator is on the output node for the next there is only 1 node left
            Some(NodeType::QubitOut(_) | NodeType::ClbitOut(_) | NodeType::VarOut(_)) => {
                (1, Some(1))
            }
            // If there is an operation the minimum is 2 for the operation and then the output node
            // while the maximum is num_ops + 1 if the next node is the first operation node and all
            // the nodes are on a single wire + the output node
            Some(NodeType::Operation(_)) => (2, Some(self.dag.num_ops() + 1)),
            // If next is none we've finished iterating.
            None => (0, Some(0)),
        }
    }

    fn next(&mut self) -> Option<Self::Item> {
        let out_node = self.next_node?;
        self.next_node = self.dag.dag.edges_directed(out_node, Outgoing).find_map(
            |e: EdgeReference<'a, Wire>| (e.weight() == &self.wire).then_some(e.target()),
        );
        Some(out_node)
    }
}

pub struct DAGCircuitBuilder {
    dag: DAGCircuit,
    last_clbits: Vec<Option<NodeIndex>>,
    last_qubits: Vec<Option<NodeIndex>>,
    last_vars: Vec<Option<NodeIndex>>,
}

impl DAGCircuitBuilder {
    /// Creates a new instance of [DAGCircuitBuilder] which allows instructions to
    /// be added continuously into the [DAGCircuit].
    pub fn new(dag: DAGCircuit) -> DAGCircuitBuilder {
        let num_qubits = dag.num_qubits();
        let num_clbits = dag.num_clbits();
        let num_vars = dag.num_vars();
        Self {
            dag,
            last_qubits: vec![None; num_qubits],
            last_clbits: vec![None; num_clbits],
            last_vars: vec![None; num_vars],
        }
    }

    /// Finishes up the changes by re-connecting all of the output nodes back to the last
    /// recorded nodes.
    pub fn build(mut self) -> DAGCircuit {
        // Re-connects all of the output nodes with their respective last nodes.
        // Add the output_nodes back to qargs
        for (qubit, node) in self
            .last_qubits
            .into_iter()
            .enumerate()
            .filter_map(|(qubit, node)| node.map(|node| (qubit, node)))
        {
            let output_node = self.dag.qubit_io_map[qubit][1];
            self.dag
                .dag
                .add_edge(node, output_node, Wire::Qubit(Qubit(qubit as u32)));
        }

        // Add the output_nodes back to cargs
        for (clbit, node) in self
            .last_clbits
            .into_iter()
            .enumerate()
            .filter_map(|(clbit, node)| node.map(|node| (clbit, node)))
        {
            let output_node = self.dag.clbit_io_map[clbit][1];
            self.dag
                .dag
                .add_edge(node, output_node, Wire::Clbit(Clbit(clbit as u32)));
        }

        // Add the output_nodes back to vars
        for (var, node) in self
            .last_vars
            .into_iter()
            .enumerate()
            .filter_map(|(var, node)| node.map(|node| (var, node)))
        {
            let output_node = self.dag.var_io_map[var][1];
            self.dag
                .dag
                .add_edge(node, output_node, Wire::Var(Var(var as u32)));
        }
        self.dag
    }

    /// Applies a new operation to the back of the circuit. This variant works with non-owned bit indices.
    pub fn apply_operation_back(
        &mut self,
        op: PackedOperation,
        qubits: &[Qubit],
        clbits: &[Clbit],
        params: Option<Parameters<Block>>,
        label: Option<String>,
        #[cfg(feature = "cache_pygates")] py_op: Option<Py<PyAny>>,
    ) -> Result<NodeIndex, DAGError> {
        let instruction = self.pack_instruction(
            op,
            qubits,
            clbits,
            params,
            label,
            #[cfg(feature = "cache_pygates")]
            py_op,
        );
        self.push_back(instruction)
    }

    /// Pushes a valid [PackedInstruction] to the back of the circuit.
    pub fn push_back(&mut self, instr: PackedInstruction) -> Result<NodeIndex, DAGError> {
        self.dag.track_instruction(&instr);
        let (all_cbits, vars) = self
            .dag
            .get_classical_resources(&instr)
            .map_err(DAGError::Python)?;
        let qubits_id = instr.qubits;
        let new_node = self.dag.dag.add_node(NodeType::Operation(instr));

        // Check all the qubits in this instruction.
        for qubit in self.dag.qargs_interner.get(qubits_id) {
            // Retrieve each qubit's last node
            let qubit_last_node = *self.last_qubits[qubit.index()].get_or_insert_with(|| {
                // If the qubit is not in the last nodes collection, the edge between the output node and its predecessor.
                // Then, store the predecessor's NodeIndex in the last nodes collection.
                let output_node = self.dag.qubit_io_map[qubit.index()][1];
                let (edge_id, predecessor_node) = self
                    .dag
                    .dag
                    .edges_directed(output_node, Incoming)
                    .next()
                    .map(|edge| (edge.id(), edge.source()))
                    .unwrap();
                self.dag.dag.remove_edge(edge_id);
                predecessor_node
            });
            self.last_qubits[qubit.index()] = Some(new_node);
            self.dag
                .dag
                .add_edge(qubit_last_node, new_node, Wire::Qubit(*qubit));
        }

        // Check all the clbits in this instruction.
        for clbit in all_cbits {
            let clbit_last_node = *self.last_clbits[clbit.index()].get_or_insert_with(|| {
                // If the qubit is not in the last nodes collection, the edge between the output node and its predecessor.
                // Then, store the predecessor's NodeIndex in the last nodes collection.
                let output_node = self.dag.clbit_io_map[clbit.index()][1];
                let (edge_id, predecessor_node) = self
                    .dag
                    .dag
                    .edges_directed(output_node, Incoming)
                    .next()
                    .map(|edge| (edge.id(), edge.source()))
                    .unwrap();
                self.dag.dag.remove_edge(edge_id);
                predecessor_node
            });
            self.last_clbits[clbit.index()] = Some(new_node);
            self.dag
                .dag
                .add_edge(clbit_last_node, new_node, Wire::Clbit(clbit));
        }

        // If available, check all the vars in this instruction
        for var in vars.iter().flatten() {
            let var_last_node = *self.last_vars[var.index()].get_or_insert_with(|| {
                // If the var is not in the last nodes collection, the edge between the output node and its predecessor.
                // Then, store the predecessor's NodeIndex in the last nodes collection.
                let output_node = self.dag.var_io_map.get(var.index()).unwrap()[1];
                let (edge_id, predecessor_node) = self
                    .dag
                    .dag
                    .edges_directed(output_node, Incoming)
                    .next()
                    .map(|edge| (edge.id(), edge.source()))
                    .unwrap();
                self.dag.dag.remove_edge(edge_id);
                predecessor_node
            });

            // Because `DAGCircuit::additional_wires` can return repeated instances of vars,
            // we need to make sure to skip those to avoid cycles.
            self.last_vars[var.index()] = Some(new_node);
            if var_last_node == new_node {
                continue;
            }
            self.dag
                .dag
                .add_edge(var_last_node, new_node, Wire::Var(*var));
        }
        Ok(new_node)
    }

    /// Packs a [PackedOperation] into a valid [PackedInstruction] within the circuit.
    #[inline]
    pub fn pack_instruction(
        &mut self,
        op: PackedOperation,
        qubits: &[Qubit],
        clbits: &[Clbit],
        params: Option<Parameters<Block>>,
        label: Option<String>,
        #[cfg(feature = "cache_pygates")] py_op: Option<Py<PyAny>>,
    ) -> PackedInstruction {
        #[cfg(feature = "cache_pygates")]
        let py_op = if let Some(py_op) = py_op {
            py_op.into()
        } else {
            OnceLock::new()
        };
        let qubits = if !qubits.is_empty() {
            self.insert_qargs(qubits)
        } else {
            self.dag.qargs_interner.get_default()
        };
        let clbits = if !clbits.is_empty() {
            self.insert_cargs(clbits)
        } else {
            self.dag.cargs_interner.get_default()
        };
        PackedInstruction {
            op,
            qubits,
            clbits,
            params: params.map(Box::new),
            label: label.map(|label| label.into()),
            #[cfg(feature = "cache_pygates")]
            py_op,
        }
    }

    /// Returns an immutable view to the qubit interner
    pub fn qargs_interner(&self) -> &Interner<[Qubit]> {
        &self.dag.qargs_interner
    }

    /// Returns an immutable view to the clbit interner
    pub fn cargs_interner(&self) -> &Interner<[Clbit]> {
        &self.dag.cargs_interner
    }

    /// Packs qargs into the circuit.
    pub fn insert_qargs(&mut self, qargs: &[Qubit]) -> Interned<[Qubit]> {
        self.dag.qargs_interner.insert(qargs)
    }
    /// Merge the `qargs` in a different [Interner] into this DAG, remapping the qubits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one DAG to
    /// another, like in substitution methods, or rebuilding a new DAG out of a lot of smaller ones.
    /// See [Interner::merge_map_slice] for more information on the mapping function.
    pub fn merge_qargs(
        &mut self,
        other: &Interner<[Qubit]>,
        map_fn: impl FnMut(&Qubit) -> Option<Qubit>,
    ) -> InternedMap<[Qubit]> {
        self.dag.merge_qargs(other, map_fn)
    }

    /// Packs qargs into the circuit.
    pub fn insert_cargs(&mut self, cargs: &[Clbit]) -> Interned<[Clbit]> {
        self.dag.cargs_interner.insert(cargs)
    }
    /// Merge the `cargs` in a different [Interner] into this DAG, remapping the clbits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one DAG to
    /// another, like in substitution methods, or rebuilding a new DAG out of a lot of smaller ones.
    /// See [Interner::merge_map_slice] for more information on the mapping function.
    pub fn merge_cargs(
        &mut self,
        other: &Interner<[Clbit]>,
        map_fn: impl FnMut(&Clbit) -> Option<Clbit>,
    ) -> InternedMap<[Clbit]> {
        self.dag.merge_cargs(other, map_fn)
    }

    /// Adds a new value to the global phase of the inner [DAGCircuit].
    pub fn add_global_phase(&mut self, param: &Param) -> Result<(), DAGError> {
        self.dag.add_global_phase(param)
    }

    /// Gives the DAG ownership of the provided basic block and returns a
    /// unique identifier that can be used to retrieve a reference to it
    /// later.
    ///
    /// No attempt is made to deduplicate the given block.
    /// No validation is performed to ensure that the given block is valid
    /// within this DAG.
    #[inline]
    pub fn add_block(&mut self, block: DAGCircuit) -> Block {
        self.dag.add_block(block)
    }
}

impl ::std::ops::Index<NodeIndex> for DAGCircuit {
    type Output = NodeType;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        self.dag.index(index)
    }
}

/// Add to global phase. Global phase can only be Float or ParameterExpression so this
/// does not handle the full possibility of parameter values.
/// TODO replace/merge this with add_param/radd_param
pub(crate) fn add_global_phase(phase: &Param, other: &Param) -> Param {
    match [phase, other] {
        [Param::Float(a), Param::Float(b)] => Param::Float(a + b),
        [Param::Float(a), Param::ParameterExpression(b)] => {
            Param::ParameterExpression(Arc::new(b.add(&ParameterExpression::from_f64(*a)).unwrap()))
        }
        [Param::ParameterExpression(a), Param::Float(b)] => {
            Param::ParameterExpression(Arc::new(a.add(&ParameterExpression::from_f64(*b)).unwrap()))
        }
        [Param::ParameterExpression(a), Param::ParameterExpression(b)] => {
            Param::ParameterExpression(Arc::new(a.add(b).expect("Name conflict in add.")))
        }
        _ => panic!("Invalid global phase"),
    }
}

/// Update the tracking fields of the [DAGCircuit].
///
/// This is tightly coupled to the [DAGCircuit] internals and should be updated to match them, but
/// is separate to allow safe borrow checking of instructions that are in-place in the DAG.
fn track_instruction(
    inst: &PackedInstruction,
    op_names: &mut IndexMap<String, usize>,
    blocks: &mut ControlFlowBlocks<DAGCircuit>,
) {
    let name = inst.op.name();
    match op_names.get_mut(name) {
        Some(count) => {
            *count += 1;
        }
        None => {
            op_names.insert(name.to_string(), 1);
        }
    }
    for block in inst.blocks_view() {
        blocks.increment(*block);
    }
}
/// Update the tracking fields of the [DAGCircuit].
///
/// This is tightly coupled to the [DAGCircuit] internals and should be updated to match them, but
/// is separate to allow safe borrow checking of instructions that are in-place in the DAG.
fn untrack_instruction(
    inst: &PackedInstruction,
    op_names: &mut IndexMap<String, usize>,
    blocks: &mut ControlFlowBlocks<DAGCircuit>,
) {
    let name = inst.op.name();
    let count = op_names
        .get_mut(name)
        .expect("tracked instructions should have a referenced name");
    *count -= 1;
    if *count == 0 {
        op_names.swap_remove(name);
    }
    for block in inst.blocks_view() {
        blocks.decrement(*block);
    }
}

type SortKeyType<'a> = (&'a [Qubit], &'a [Clbit]);

#[cfg(test)]
mod test {
    use crate::bit::{ClassicalRegister, QuantumRegister};
    use crate::dag_circuit::{BlocksMode, DAGCircuit, Wire};
    use crate::operations::{StandardGate, StandardInstruction};
    use crate::packed_instruction::{PackedInstruction, PackedOperation};
    use crate::{Clbit, Qubit};
    use hashbrown::HashSet;
    use pyo3::prelude::*;
    use rustworkx_core::petgraph::prelude::*;
    use rustworkx_core::petgraph::stable_graph::DefaultIx;
    use rustworkx_core::petgraph::visit::IntoEdgeReferences;

    fn new_dag(qubits: u32, clbits: u32) -> DAGCircuit {
        let qreg = QuantumRegister::new_owning("q".to_owned(), qubits);
        let creg = ClassicalRegister::new_owning("c".to_owned(), clbits);
        let mut dag = DAGCircuit::new();
        dag.add_qreg(qreg).unwrap();
        dag.add_creg(creg).unwrap();
        dag
    }

    macro_rules! cx_gate {
        ($dag:expr, $q0:expr, $q1:expr) => {
            PackedInstruction::from_standard_gate(
                StandardGate::CX,
                None,
                $dag.qargs_interner
                    .insert_owned(vec![Qubit($q0), Qubit($q1)]),
            )
        };
    }

    macro_rules! measure {
        ($dag:expr, $qarg:expr, $carg:expr) => {{
            let qubits = $dag.qargs_interner.insert_owned(vec![Qubit($qarg)]);
            let clbits = $dag.cargs_interner.insert_owned(vec![Clbit($qarg)]);
            PackedInstruction {
                op: PackedOperation::from_standard_instruction(StandardInstruction::Measure),
                qubits,
                clbits,
                params: None,
                label: None,
                #[cfg(feature = "cache_pygates")]
                py_op: Default::default(),
            }
        }};
    }

    #[test]
    fn test_push_back() -> PyResult<()> {
        let mut dag = new_dag(2, 2);

        // IO nodes.
        let [q0_in_node, q0_out_node] = dag.qubit_io_map[0];
        let [q1_in_node, q1_out_node] = dag.qubit_io_map[1];
        let [c0_in_node, c0_out_node] = dag.clbit_io_map[0];
        let [c1_in_node, c1_out_node] = dag.clbit_io_map[1];

        // Add a CX to the otherwise empty circuit.
        let cx = cx_gate!(dag, 0, 1);
        let cx_node = dag.push_back(cx)?;
        assert!(matches!(dag.op_names.get("cx"), Some(1)));

        let expected_wires = HashSet::from_iter([
            // q0In => CX => q0Out
            (q0_in_node, cx_node, Wire::Qubit(Qubit(0))),
            (cx_node, q0_out_node, Wire::Qubit(Qubit(0))),
            // q1In => CX => q1Out
            (q1_in_node, cx_node, Wire::Qubit(Qubit(1))),
            (cx_node, q1_out_node, Wire::Qubit(Qubit(1))),
            // No clbits used, so in goes straight to out.
            (c0_in_node, c0_out_node, Wire::Clbit(Clbit(0))),
            (c1_in_node, c1_out_node, Wire::Clbit(Clbit(1))),
        ]);

        let actual_wires: HashSet<_> = dag
            .dag
            .edge_references()
            .map(|e| (e.source(), e.target(), *e.weight()))
            .collect();

        assert_eq!(actual_wires, expected_wires, "unexpected DAG structure");

        // Add measures after CX.
        let measure_q0 = measure!(dag, 0, 0);
        let measure_q0_node = dag.push_back(measure_q0)?;

        let measure_q1 = measure!(dag, 1, 1);
        let measure_q1_node = dag.push_back(measure_q1)?;

        let expected_wires = HashSet::from_iter([
            // q0In -> CX -> M -> q0Out
            (q0_in_node, cx_node, Wire::Qubit(Qubit(0))),
            (cx_node, measure_q0_node, Wire::Qubit(Qubit(0))),
            (measure_q0_node, q0_out_node, Wire::Qubit(Qubit(0))),
            // q1In -> CX -> M -> q1Out
            (q1_in_node, cx_node, Wire::Qubit(Qubit(1))),
            (cx_node, measure_q1_node, Wire::Qubit(Qubit(1))),
            (measure_q1_node, q1_out_node, Wire::Qubit(Qubit(1))),
            // c0In -> M -> c0Out
            (c0_in_node, measure_q0_node, Wire::Clbit(Clbit(0))),
            (measure_q0_node, c0_out_node, Wire::Clbit(Clbit(0))),
            // c1In -> M -> c1Out
            (c1_in_node, measure_q1_node, Wire::Clbit(Clbit(1))),
            (measure_q1_node, c1_out_node, Wire::Clbit(Clbit(1))),
        ]);

        let actual_wires: HashSet<_> = dag
            .dag
            .edge_references()
            .map(|e| (e.source(), e.target(), *e.weight()))
            .collect();

        assert_eq!(actual_wires, expected_wires, "unexpected DAG structure");
        Ok(())
    }

    #[test]
    fn test_push_front() -> PyResult<()> {
        let mut dag = new_dag(2, 2);

        // IO nodes.
        let [q0_in_node, q0_out_node] = dag.qubit_io_map[0];
        let [q1_in_node, q1_out_node] = dag.qubit_io_map[1];
        let [c0_in_node, c0_out_node] = dag.clbit_io_map[0];
        let [c1_in_node, c1_out_node] = dag.clbit_io_map[1];

        // Add measures first (we'll add something before them afterwards).
        let measure_q0 = measure!(dag, 0, 0);
        let measure_q0_node = dag.push_back(measure_q0)?;

        let measure_q1 = measure!(dag, 1, 1);
        let measure_q1_node = dag.push_back(measure_q1)?;

        let expected_wires = HashSet::from_iter([
            // q0In => M => q0Out
            (q0_in_node, measure_q0_node, Wire::Qubit(Qubit(0))),
            (measure_q0_node, q0_out_node, Wire::Qubit(Qubit(0))),
            // q1In => M => q1Out
            (q1_in_node, measure_q1_node, Wire::Qubit(Qubit(1))),
            (measure_q1_node, q1_out_node, Wire::Qubit(Qubit(1))),
            // c0In -> M -> c0Out
            (c0_in_node, measure_q0_node, Wire::Clbit(Clbit(0))),
            (measure_q0_node, c0_out_node, Wire::Clbit(Clbit(0))),
            // c1In -> M -> c1Out
            (c1_in_node, measure_q1_node, Wire::Clbit(Clbit(1))),
            (measure_q1_node, c1_out_node, Wire::Clbit(Clbit(1))),
        ]);

        let actual_wires: HashSet<_> = dag
            .dag
            .edge_references()
            .map(|e| (e.source(), e.target(), *e.weight()))
            .collect();

        assert_eq!(actual_wires, expected_wires);

        // Add a CX before the measures.
        let cx = cx_gate!(dag, 0, 1);
        let cx_node = dag.push_front(cx)?;
        assert!(matches!(dag.op_names.get("cx"), Some(1)));

        let expected_wires = HashSet::from_iter([
            // q0In -> CX -> M -> q0Out
            (q0_in_node, cx_node, Wire::Qubit(Qubit(0))),
            (cx_node, measure_q0_node, Wire::Qubit(Qubit(0))),
            (measure_q0_node, q0_out_node, Wire::Qubit(Qubit(0))),
            // q1In -> CX -> M -> q1Out
            (q1_in_node, cx_node, Wire::Qubit(Qubit(1))),
            (cx_node, measure_q1_node, Wire::Qubit(Qubit(1))),
            (measure_q1_node, q1_out_node, Wire::Qubit(Qubit(1))),
            // c0In -> M -> c0Out
            (c0_in_node, measure_q0_node, Wire::Clbit(Clbit(0))),
            (measure_q0_node, c0_out_node, Wire::Clbit(Clbit(0))),
            // c1In -> M -> c1Out
            (c1_in_node, measure_q1_node, Wire::Clbit(Clbit(1))),
            (measure_q1_node, c1_out_node, Wire::Clbit(Clbit(1))),
        ]);

        let actual_wires: HashSet<_> = dag
            .dag
            .edge_references()
            .map(|e| (e.source(), e.target(), *e.weight()))
            .collect();

        assert_eq!(actual_wires, expected_wires, "unexpected DAG structure");
        Ok(())
    }

    #[test]
    fn test_physical_empty_like() -> PyResult<()> {
        let mut dag = DAGCircuit::new();
        let qr = QuantumRegister::new_owning("virtual".to_owned(), 5);
        let cr = ClassicalRegister::new_owning("classical".to_owned(), 5);
        dag.name = Some("my dag".to_owned());
        dag.add_creg(cr.clone())?;
        dag.add_qreg(qr)?;
        dag.apply_operation_back(
            StandardGate::H.into(),
            &[Qubit(0)],
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )?;
        dag.apply_operation_back(
            StandardGate::CX.into(),
            &[Qubit(0), Qubit(1)],
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )?;
        let empty = dag.physical_empty_like_with_capacity(10, 0, 0, BlocksMode::Drop)?;
        assert_eq!(empty.name.as_deref(), Some("my dag"));
        assert_eq!(
            empty
                .qregs()
                .iter()
                .map(|reg| (reg.name(), reg.len()))
                .collect::<Vec<_>>(),
            vec![("q", 10)]
        );
        assert_eq!(empty.cregs(), &[cr]);
        assert_eq!(empty.num_ops(), 0);
        Ok(())
    }

    #[test]
    fn verify_default_ix_type() {
        // This test will prevent Qiskit from compiling in the unlikely scenario
        // that petgraph's NodeIndex type will change from u32 to something else.
        // If this does happen, pointer conversions to `*const NodeIndex` in `dag.rs`
        // will need to be updated as well.
        let _: DefaultIx = 0u32;
    }
}
