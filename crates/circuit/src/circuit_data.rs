// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use crate::bit::{
    BitLocations, ClassicalRegister, PyBit, QuantumRegister, Register, ShareableClbit,
    ShareableQubit,
};
use crate::bit_locator::BitLocator;
use crate::circuit_instruction::{CircuitInstruction, OperationFromPython};
use crate::classical::expr;
use crate::dag_circuit::{DAGCircuit, add_global_phase};
use crate::imports::{ANNOTATED_OPERATION, QUANTUM_CIRCUIT};
use crate::instruction::Parameters;
use crate::interner::{Interned, InternedMap, Interner};
use crate::object_registry::{self, ObjectRegistry};
use crate::operations::{
    BoxedCustomOperation, ControlFlow, ControlFlowView, LoopParam, Operation, OperationRef, Param,
    PauliBased, PauliProductRotation, PyOpKind, PythonOperation, StandardGate,
};
use crate::packed_instruction::{PackedInstruction, PackedOperation};
use crate::parameter::parameter_expression::{ParameterError, ParameterExpression};
use crate::parameter::symbol_expr::{Symbol, Value};
use crate::parameter_table::{ParameterTable, ParameterTableError, ParameterUse, ParameterUuid};
use crate::register_data::{RegisterAlreadyExists, RegisterData};
use crate::var_stretch_container::{
    StretchType, VarStretchContainer, VarStretchContainerError, VarType,
};
use crate::{
    Block, BlocksMode, CapacityError, Clbit, ControlFlowBlocks, Qubit, Stretch, Var, VarsMode,
    instruction,
};
use qiskit_util::py::{PySequenceIndex, SequenceIndex};

use ndarray::ArrayView1;
use num_complex::Complex64;
use numpy::PyReadonlyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList, PySet, PyTuple, PyType};
use pyo3::{PyTraverseError, PyVisit, import_exception, intern};

use hashbrown::{HashMap, HashSet};
use qiskit_util::IndexMap;
use smallvec::SmallVec;
use thiserror::Error;

import_exception!(qiskit.circuit.exceptions, CircuitError);

/// This struct models the error conditions that can be raised from the
/// rust methods of the [CircuitData] struct. The goal is to explicitly
/// enumerate all the error types that are returned by these functions and
/// make it clear that the return type is part of the interface.
///
/// In the the future it is expected this single enum will be replaced by per
/// method error types to make it even more obvious, but this is a step
/// towards that as we migrate from Python -> Rust.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum CircuitDataError {
    #[error(transparent)]
    Capacity(#[from] CapacityError),
    #[error("invalid type for global phase")]
    InvalidGlobalPhaseType,
    #[error(transparent)]
    AbsentObject(object_registry::AbsentObject),
    #[error(transparent)]
    AddObjectRegistry(object_registry::AddError),
    #[error(transparent)]
    RegisterNameExists(#[from] RegisterAlreadyExists),
    #[error("{0} at index {1} exceeds circuit capacity.")]
    BitExceedsCapacity(String, usize),
    // Explicitly an error returned from calling Python
    #[error(transparent)]
    ErrorFromPython(#[from] PyErr),
    #[error(transparent)]
    ParameterTableError(#[from] ParameterTableError),
    #[error(transparent)]
    VarStretchContainerError(#[from] VarStretchContainerError),
    #[error(
        "Mismatching number of values and parameters. For partial binding please pass a mapping of parameter to value pairs."
    )]
    ParameterSliceLenMismatch,
    #[error(transparent)]
    ParameterError(#[from] ParameterError),
    #[error("Invalid Parameter")]
    InvalidParameter,
    #[error("bad type after binding for gate '{0}': '{1}'")]
    StandardGateParameterIsComplex(String, String),
}
impl<T: Debug> From<object_registry::AbsentObject<T>> for CircuitDataError {
    fn from(val: object_registry::AbsentObject<T>) -> Self {
        Self::AbsentObject(val.erase_type())
    }
}
impl<T: Debug, B: Debug> From<object_registry::AddError<T, B>> for CircuitDataError {
    fn from(val: object_registry::AddError<T, B>) -> Self {
        Self::AddObjectRegistry(val.erase_type())
    }
}

impl From<CircuitDataError> for PyErr {
    fn from(error: CircuitDataError) -> Self {
        match error {
            CircuitDataError::Capacity(c) => c.into(),
            CircuitDataError::InvalidGlobalPhaseType => {
                PyTypeError::new_err("invalid type for global phase")
            }
            CircuitDataError::AbsentObject(e) => e.into(),
            CircuitDataError::AddObjectRegistry(e) => e.into(),
            CircuitDataError::ErrorFromPython(e) => e,
            CircuitDataError::ParameterTableError(e) => e.into(),
            CircuitDataError::RegisterNameExists(name) => {
                CircuitError::new_err(format!("register name {name} already exists"))
            }
            CircuitDataError::BitExceedsCapacity(bit_type, bit_index) => CircuitError::new_err(
                format!("{bit_type} at index {bit_index} exceds circuit capacity."),
            ),
            CircuitDataError::VarStretchContainerError(e) => CircuitError::new_err(e.to_string()),
            CircuitDataError::ParameterSliceLenMismatch => PyValueError::new_err(
                "Mismatching number of values and parameters. For partial binding please pass a mapping of {parameter: value} pairs.",
            ),
            CircuitDataError::ParameterError(e) => e.into(),
            CircuitDataError::InvalidParameter => {
                PyValueError::new_err("An invalid parameter was provided.")
            }
            CircuitDataError::StandardGateParameterIsComplex(gate_name, expr) => {
                CircuitError::new_err(format!(
                    "bad type after binding for gate '{gate_name}': '{expr}'"
                ))
            }
        }
    }
}

/// A tuple of a `CircuitData`'s internal state used for pickle's `__setstate__()` method.
type CircuitDataState<'py> = (
    Vec<QuantumRegister>,
    Vec<ClassicalRegister>,
    Bound<'py, PyDict>,       // qubit location data
    Bound<'py, PyDict>,       // clbit location data
    Vec<(String, Py<PyAny>)>, // vector of (variable name, var_stretch_container::IdentifierInfo object) pairs
    Vec<expr::Var>,
    Vec<expr::Stretch>,
);

/// The core data structure representing a quantum circuit's instruction sequence and metadata.
///
/// `CircuitData` is the internal Rust representation of a quantum circuit, storing the complete
/// state including instructions, quantum and classical bits, registers, variables, and parameters.
/// This struct is designed for efficient manipulation and traversal of circuit data from Rust code.
///
/// The data is stored in a packed format for memory efficiency, with interned bit arguments to
/// reduce duplication. Instructions are stored as [`PackedInstruction`] objects in a vector,
/// allowing for fast sequential access and modification.
///
/// # Fields
///
/// The struct contains:
/// - Instruction data and interned bit caches
/// - Quantum and classical bit registries
/// - Quantum and classical registers
/// - Control flow blocks for nested circuit structures
/// - Variable and stretch registries for dynamic circuits
/// - Parameter table for parameterized gates
/// - Global phase information
///
/// # Usage
///
/// This struct is primarily used internally by the circuit implementation and is wrapped by
/// [`PyCircuitData`] for Python interoperability. Direct manipulation should be done carefully
/// to maintain invariants around bit indices, parameter tracking, and register consistency.
#[derive(Clone, Debug)]
pub struct CircuitData {
    /// The packed instruction listing.
    data: Vec<PackedInstruction>,
    /// The cache used to intern instruction bits.
    qargs_interner: Interner<[Qubit]>,
    /// The cache used to intern instruction bits.
    cargs_interner: Interner<[Clbit]>,
    /// Qubits registered in the circuit.
    qubits: ObjectRegistry<Qubit, ShareableQubit>,
    /// Clbits registered in the circuit.
    clbits: ObjectRegistry<Clbit, ShareableClbit>,
    /// Basic blocks registered in the circuit.
    blocks: ControlFlowBlocks<CircuitData>,
    /// QuantumRegisters stored in the circuit
    qregs: RegisterData<QuantumRegister>,
    /// ClassicalRegisters stored in the circuit
    cregs: RegisterData<ClassicalRegister>,
    /// Mapping between [ShareableQubit] and its locations in
    /// the circuit
    qubit_indices: BitLocator<ShareableQubit, QuantumRegister>,
    /// Mapping between [ShareableClbit] and its locations in
    /// the circuit
    clbit_indices: BitLocator<ShareableClbit, ClassicalRegister>,
    /// Variables and stretches registered in the circuit
    vars_stretches: VarStretchContainer,

    param_table: ParameterTable,
    global_phase: Param,
}

/// A container for :class:`.QuantumCircuit` instruction listings that stores
/// :class:`.CircuitInstruction` instances in a packed form by interning
/// their :attr:`~.CircuitInstruction.qubits` and
/// :attr:`~.CircuitInstruction.clbits` to native vectors of indices.
///
/// Before adding a :class:`.CircuitInstruction` to this container, its
/// :class:`.Qubit` and :class:`.Clbit` instances MUST be registered via the
/// constructor or via :meth:`.CircuitData.add_qubit` and
/// :meth:`.CircuitData.add_clbit`. This is because the order in which
/// bits of the same type are added to the container determines their
/// associated indices used for storage and retrieval.
///
/// Once constructed, this container behaves like a Python list of
/// :class:`.CircuitInstruction` instances. However, these instances are
/// created and destroyed on the fly, and thus should be treated as ephemeral.
///
/// For example,
///
/// .. plot::
///    :include-source:
///    :no-figs:
///
///     qubits = [Qubit()]
///     data = CircuitData(qubits)
///     data.append(CircuitInstruction(XGate(), (qubits[0],), ()))
///     assert(data[0] == data[0]) # => Ok.
///     assert(data[0] is data[0]) # => PANICS!
///
/// .. warning::
///
///     This is an internal interface and no part of it should be relied upon
///     outside of Qiskit.
///
/// Args:
///     qubits (Iterable[:class:`.Qubit`] | None): The initial sequence of
///         qubits, used to map :class:`.Qubit` instances to and from its
///         indices.
///     clbits (Iterable[:class:`.Clbit`] | None): The initial sequence of
///         clbits, used to map :class:`.Clbit` instances to and from its
///         indices.
///     data (Iterable[:class:`.CircuitInstruction`]): An initial instruction
///         listing to add to this container. All bits appearing in the
///         instructions in this iterable must also exist in ``qubits`` and
///         ``clbits``.
///     reserve (int): The container's initial capacity. This is reserved
///         before copying instructions into the container when ``data``
///         is provided, so the initialized container's unused capacity will
///         be ``max(0, reserve - len(data))``.
///
/// Raises:
///     KeyError: if ``data`` contains a reference to a bit that is not present
///         in ``qubits`` or ``clbits``.
#[pyclass(
    name = "CircuitData",
    sequence,
    module = "qiskit._accelerate.circuit",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyCircuitData {
    pub inner: CircuitData,
}

impl From<CircuitData> for PyCircuitData {
    fn from(data: CircuitData) -> Self {
        PyCircuitData { inner: data }
    }
}

impl From<PyCircuitData> for CircuitData {
    fn from(data: PyCircuitData) -> Self {
        data.inner
    }
}

impl Deref for PyCircuitData {
    type Target = CircuitData;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl CircuitData {
    pub fn new(
        qubits: Option<Vec<ShareableQubit>>,
        clbits: Option<Vec<ShareableClbit>>,
        global_phase: Param,
    ) -> Result<Self, CircuitDataError> {
        let qubit_size = qubits.as_ref().map_or(0, |bits| bits.len());
        let clbit_size = clbits.as_ref().map_or(0, |bits| bits.len());
        let qubits_registry = ObjectRegistry::with_capacity(qubit_size);
        let clbits_registry = ObjectRegistry::with_capacity(clbit_size);
        let qubit_indices = BitLocator::with_capacity(qubit_size);
        let clbit_indices = BitLocator::with_capacity(clbit_size);

        let mut self_ = CircuitData {
            data: Vec::new(),
            qargs_interner: Interner::new(),
            cargs_interner: Interner::new(),
            qubits: qubits_registry,
            clbits: clbits_registry,
            blocks: ControlFlowBlocks::new(),
            param_table: ParameterTable::new(),
            global_phase: Param::Float(0.),
            qregs: RegisterData::new(),
            cregs: RegisterData::new(),
            qubit_indices,
            clbit_indices,
            vars_stretches: VarStretchContainer::new(),
        };
        self_.set_global_phase_param(global_phase)?;
        if let Some(qubits) = qubits {
            for bit in qubits.into_iter() {
                self_.add_qubit(bit, true)?;
            }
        }
        if let Some(clbits) = clbits {
            for bit in clbits.into_iter() {
                self_.add_clbit(bit, true)?;
            }
        }
        Ok(self_)
    }

    pub fn add_qubit(
        &mut self,
        bit: ShareableQubit,
        strict: bool,
    ) -> Result<Qubit, CircuitDataError> {
        let index = if strict {
            self.qubits.add(bit.clone())?
        } else {
            self.qubits.add_allow_existing(bit.clone())?
        };
        self.qubit_indices
            .insert(bit, BitLocations::new(index.0, []));
        Ok(index)
    }

    pub fn add_clbit(
        &mut self,
        bit: ShareableClbit,
        strict: bool,
    ) -> Result<Clbit, CircuitDataError> {
        let index = if strict {
            self.clbits.add(bit.clone())?
        } else {
            self.clbits.add_allow_existing(bit.clone())?
        };
        self.clbit_indices
            .insert(bit, BitLocations::new(index.0, []));
        Ok(index)
    }

    #[inline]
    pub fn blocks(&self) -> &ControlFlowBlocks<CircuitData> {
        &self.blocks
    }

    /// Return the number of qubits. This is equivalent to the length of the list returned by
    /// :meth:`.CircuitData.qubits`
    ///
    /// Returns:
    ///     int: The number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Return the number of clbits. This is equivalent to the length of the list returned by
    /// :meth:`.CircuitData.clbits`.
    ///
    /// Returns:
    ///     int: The number of clbits.
    pub fn num_clbits(&self) -> usize {
        self.clbits.len()
    }

    /// Return the number of unbound compile-time symbolic parameters tracked by the circuit.
    pub fn num_parameters(&self) -> usize {
        self.param_table.num_parameters()
    }

    /// Return the width of the circuit. This is the number of qubits plus the
    /// number of clbits.
    ///
    /// Returns:
    ///     int: The width of the circuit.
    pub fn width(&self) -> usize {
        self.num_qubits() + self.num_clbits()
    }

    /// Registers a :class:`.QuantumRegister` instance.
    ///
    /// Args:
    ///     bit (:class:`.QuantumRegister`): The register to add.
    pub fn add_qreg(
        &mut self,
        register: QuantumRegister,
        strict: bool,
    ) -> Result<(), CircuitDataError> {
        self.qregs
            .add_register(register.clone(), strict)
            .map_err(CircuitDataError::RegisterNameExists)?;

        for (index, bit) in register.bits().enumerate() {
            if let Some(entry) = self.qubit_indices.get_mut(&bit) {
                entry.add_register(register.clone(), index);
            } else if let Some(bit_idx) = self.qubits.find(&bit) {
                self.qubit_indices.insert(
                    bit,
                    BitLocations::new(bit_idx.0, [(register.clone(), index)]),
                );
            } else {
                let bit_idx = self.qubits.len();
                self.add_qubit(bit.clone(), true)?;
                self.qubit_indices.insert(
                    bit,
                    BitLocations::new(
                        bit_idx.try_into().map_err(|_| {
                            CircuitDataError::BitExceedsCapacity("Qubit".to_string(), bit_idx)
                        })?,
                        [(register.clone(), index)],
                    ),
                );
            }
        }
        Ok(())
    }

    /// Registers a :class:`.ClassicalRegister` instance.
    ///
    /// Args:
    ///     bit (:class:`.ClassicalRegister`): The register to add.
    pub fn add_creg(
        &mut self,
        register: ClassicalRegister,
        strict: bool,
    ) -> Result<(), CircuitDataError> {
        self.cregs
            .add_register(register.clone(), strict)
            .map_err(CircuitDataError::RegisterNameExists)?;
        for (index, bit) in register.bits().enumerate() {
            if let Some(entry) = self.clbit_indices.get_mut(&bit) {
                entry.add_register(register.clone(), index);
            } else if let Some(bit_idx) = self.clbits.find(&bit) {
                self.clbit_indices.insert(
                    bit,
                    BitLocations::new(bit_idx.0, [(register.clone(), index)]),
                );
            } else {
                let bit_idx = self.clbits.len();
                self.add_clbit(bit.clone(), true)?;
                self.clbit_indices.insert(
                    bit,
                    BitLocations::new(
                        bit_idx.try_into().map_err(|_| {
                            CircuitDataError::BitExceedsCapacity("Clbit".to_string(), bit_idx)
                        })?,
                        [(register.clone(), index)],
                    ),
                );
            }
        }
        Ok(())
    }

    /// Reserves capacity for at least ``additional`` more
    /// :class:`.CircuitInstruction` instances to be added to this container.
    ///
    /// Args:
    ///     additional (int): The additional capacity to reserve. If the
    ///         capacity is already sufficient, does nothing.
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Move this [CircuitData] into a complete Python `QuantumCircuit` object.
    pub fn into_py_quantum_circuit(self, py: Python) -> PyResult<Bound<PyAny>> {
        let py_circuit_data: PyCircuitData = self.into();
        py_circuit_data.into_py_quantum_circuit(py)
    }

    /// Gives the circuit ownership of the provided basic block and returns a
    /// unique identifier that can be used to retrieve a reference to it
    /// later.
    ///
    /// No attempt is made to deduplicate the given block.
    /// No validation is performed to ensure that the given block is valid
    /// within the circuit.
    pub fn add_block(&mut self, block: CircuitData) -> Block {
        self.blocks.push(block)
    }

    /// Take the blocks from an extraction from an `OperationFromPython<CircuitData>` and add them
    /// to this circuit.
    ///
    /// Returns the parameters in terms of block references.
    fn take_parameter_blocks(
        &mut self,
        params: Option<Parameters<CircuitData>>,
    ) -> Option<Box<Parameters<Block>>> {
        params.map(|params| Box::new(params.map_blocks(|block| self.add_block(block))))
    }

    /// Extract the blocks from a Python-space `CircuitInstruction` and add them to this circuit.
    ///
    /// The inverse of this is [unpack_blocks_to_circuit_parameters].  The version for when you can
    /// take the blocks directly is [take_parameter_blocks].
    pub fn extract_blocks_from_circuit_parameters(
        &mut self,
        params: Option<&Parameters<CircuitData>>,
    ) -> Option<Box<Parameters<Block>>> {
        params.map(|params| Box::new(params.map_blocks_ref(|block| self.add_block(block.clone()))))
    }

    /// Given a `params` object from an instruction packed into this circuit, extract any relevant
    /// blocks into the owned-object `CircuitData` block type, suitable for passing back to Python
    /// space.
    ///
    /// The inverse of this method is [extract_blocks_from_circuit_parameters].
    pub fn unpack_blocks_to_circuit_parameters(
        &self,
        params: Option<&Parameters<Block>>,
    ) -> Option<Parameters<CircuitData>> {
        params.map(|params| params.map_blocks_ref(|block| self.blocks[*block].clone()))
    }

    /// Gets an immutable view of a control flow operation.
    ///
    /// Panics or produces incorrect results if `instr` is not from this circuit (or compatible with
    /// it, e.g. from a mapped [ControlFlowBlocks]).
    pub fn try_view_control_flow<'a>(
        &'a self,
        instr: &'a PackedInstruction,
    ) -> Option<ControlFlowView<'a, CircuitData>> {
        ControlFlowView::try_from_instruction(instr, &self.blocks)
    }

    pub fn copy_empty_like(
        &self,
        vars_mode: VarsMode,
        blocks_mode: BlocksMode,
    ) -> Result<Self, CircuitDataError> {
        let mut res = CircuitData::new(
            Some(self.qubits.objects().clone()),
            Some(self.clbits.objects().clone()),
            self.global_phase.clone(),
        )?;

        res.qargs_interner = self.qargs_interner.clone();
        res.cargs_interner = self.cargs_interner.clone();

        if blocks_mode == BlocksMode::Keep && !self.blocks.is_empty() {
            res.blocks = self.blocks.map_without_references(|block| block.clone());
        }

        // After initialization, copy register info.
        res.qregs = self.qregs.clone();
        res.cregs = self.cregs.clone();
        res.qubit_indices = self.qubit_indices.clone();
        res.clbit_indices = self.clbit_indices.clone();

        match vars_mode {
            VarsMode::Alike => {
                res.vars_stretches = self.vars_stretches.clone();
            }
            VarsMode::Captures => {
                res.vars_stretches = self.vars_stretches.clone_as_captures();
            }
            VarsMode::Drop => {}
        };

        Ok(res)
    }

    /// An alternate constructor to build a new `CircuitData` from an iterator
    /// of packed operations. This can be used to build a circuit from a sequence
    /// of `PackedOperation` without needing to involve Python.
    ///
    /// This can be connected with the Python space
    /// QuantumCircuit.from_circuit_data() constructor to build a full
    /// QuantumCircuit from Rust.
    ///
    /// This constructor does not support control flow operations and will panic
    /// if they are provided. If you need this, you should construct an empty
    /// circuit, register any basic blocks explicitly, and then append control flow
    /// operations.
    ///
    /// # Arguments
    ///
    /// * num_qubits: The number of qubits in the circuit.
    /// * num_clbits: The number of classical bits in the circuit.
    /// * instructions: An iterator of the (packed operation, params, qubits, clbits) to
    ///   add to the circuit
    /// * global_phase: The global phase to use for the circuit
    pub fn from_packed_operations<I>(
        num_qubits: u32,
        num_clbits: u32,
        instructions: I,
        global_phase: Param,
    ) -> Result<Self, CircuitDataError>
    where
        I: IntoIterator<
            Item = Result<
                (
                    PackedOperation,
                    SmallVec<[Param; 3]>,
                    Vec<Qubit>,
                    Vec<Clbit>,
                ),
                CircuitDataError,
            >,
        >,
    {
        let instruction_iter = instructions.into_iter();
        let mut res = Self::with_capacity(
            num_qubits,
            num_clbits,
            instruction_iter.size_hint().0,
            global_phase,
        )?;

        for item in instruction_iter {
            let (operation, params, qargs, cargs) = item?;
            let qubits = res.qargs_interner.insert_owned(qargs);
            let clbits = res.cargs_interner.insert_owned(cargs);
            let params = (!params.is_empty()).then(|| Box::new(Parameters::Params(params)));
            res.push(PackedInstruction {
                op: operation,
                qubits,
                clbits,
                params,
                label: None,
                #[cfg(feature = "cache_pygates")]
                py_op: OnceLock::new(),
            })?;
        }
        Ok(res)
    }

    /// Clone a new [CircuitData] from a [DAGCircuit].
    ///
    /// This is the logical equivalent of Python's `dag_to_circuit`.
    pub fn from_dag_ref(dag: &DAGCircuit) -> Result<Self, CircuitDataError> {
        let mut out = Self::empty_like_from_dag(dag)?;
        out.data.reserve(dag.num_ops());
        for node in dag.topological_op_nodes(false) {
            out.push(dag[node].unwrap_operation().clone())?;
        }
        Ok(out)
    }

    /// Clone a new [CircuitData] from a [DAGCircuit], but applying a Python deepcopy
    ///
    /// This is the logical equivalent of Python's `dag_to_circuit`.
    pub fn from_dag_ref_deepcopy(py: Python, dag: &DAGCircuit) -> Result<Self, CircuitDataError> {
        let memo = PyDict::new(py);
        let mut out = Self::empty_like_from_dag(dag)?;
        out.data.reserve(dag.num_ops());
        for node in dag.topological_op_nodes(false) {
            let inst = dag[node].unwrap_operation();
            out.push(PackedInstruction {
                op: inst.op.py_deepcopy(py, Some(&memo))?,
                ..inst.clone()
            })?;
        }
        Ok(out)
    }

    fn empty_like_from_dag(dag: &DAGCircuit) -> Result<Self, CircuitDataError> {
        let mut out = Self {
            data: Vec::new(),
            qargs_interner: dag.qargs_interner().clone(),
            cargs_interner: dag.cargs_interner().clone(),
            qubits: dag.qubits().clone(),
            clbits: dag.clbits().clone(),
            blocks: dag
                .blocks()
                .try_map_without_references(Self::from_dag_ref)?,
            qregs: dag.qregs_data().clone(),
            cregs: dag.cregs_data().clone(),
            qubit_indices: dag.qubit_locations().clone(),
            clbit_indices: dag.clbit_locations().clone(),
            vars_stretches: dag.vars_stretches_view().clone(),
            param_table: ParameterTable::new(),
            global_phase: Param::Float(0.0),
        };
        out.set_global_phase_param(dag.global_phase().clone())?;

        Ok(out)
    }

    /// An alternate constructor to build a new `CircuitData` from an iterator
    /// of standard gates. This can be used to build a circuit from a sequence
    /// of standard gates, such as for a `StandardGate` definition or circuit
    /// synthesis without needing to involve Python.
    ///
    /// This can be connected with the Python space
    /// QuantumCircuit.from_circuit_data() constructor to build a full
    /// QuantumCircuit from Rust.
    ///
    /// # Arguments
    ///
    /// * num_qubits: The number of qubits in the circuit. These will be created
    ///   in Python as loose bits without a register.
    /// * instructions: An iterator of the standard gate params and qubits to
    ///   add to the circuit
    /// * global_phase: The global phase to use for the circuit
    pub fn from_standard_gates<I>(
        num_qubits: u32,
        instructions: I,
        global_phase: Param,
    ) -> Result<Self, CircuitDataError>
    where
        I: IntoIterator<Item = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)>,
    {
        let instruction_iter = instructions.into_iter();
        let mut res =
            Self::with_capacity(num_qubits, 0, instruction_iter.size_hint().0, global_phase)?;

        for (operation, params, qargs) in instruction_iter {
            let qubits = res.qargs_interner.insert(&qargs);
            let params = (!params.is_empty()).then(|| Box::new(params));
            res.push(PackedInstruction::from_standard_gate(
                operation, params, qubits,
            ))?;
        }
        Ok(res)
    }

    /// Build an empty CircuitData object with an initially allocated instruction capacity
    pub fn with_capacity(
        num_qubits: u32,
        num_clbits: u32,
        instruction_capacity: usize,
        global_phase: Param,
    ) -> Result<Self, CircuitDataError> {
        let mut res = CircuitData {
            data: Vec::with_capacity(instruction_capacity),
            qargs_interner: Interner::new(),
            cargs_interner: Interner::new(),
            qubits: ObjectRegistry::with_capacity(num_qubits as usize),
            clbits: ObjectRegistry::with_capacity(num_clbits as usize),
            blocks: ControlFlowBlocks::new(),
            param_table: ParameterTable::new(),
            global_phase: Param::Float(0.0),
            qregs: RegisterData::new(),
            cregs: RegisterData::new(),
            qubit_indices: BitLocator::with_capacity(num_qubits as usize),
            clbit_indices: BitLocator::with_capacity(num_clbits as usize),
            vars_stretches: VarStretchContainer::new(),
        };

        // use the global phase setter to ensure parameters are registered
        // in the parameter table
        res.set_global_phase_param(global_phase)?;
        res.add_anonymous_qubits(num_qubits)
            .expect("cannot represent a too-large count");
        res.add_anonymous_clbits(num_clbits)
            .expect("cannot represent a too-large count");
        Ok(res)
    }

    /// Add multiple new anonymous qubits.
    ///
    /// This can only fail due to circuit capacity issues, since new anonymous qubits are guaranteed
    /// to be unique.
    pub fn add_anonymous_qubits(&mut self, num: u32) -> Result<(), CapacityError> {
        if self.qubits.len() > (u32::MAX - num) as usize {
            return Err(CapacityError);
        }
        for bit in ShareableQubit::iter_anonymous(num) {
            let index = self.qubits.add_unique_within_capacity(bit.clone());
            self.qubit_indices
                .insert(bit, BitLocations::new(index.0, []));
        }
        Ok(())
    }

    /// Add multiple new anonymous clbits.
    ///
    /// This can only fail due to circuit capacity issues, since new anonymous qubits are guaranteed
    /// to be unique.
    pub fn add_anonymous_clbits(&mut self, num: u32) -> Result<(), CapacityError> {
        if self.clbits.len() > (u32::MAX - num) as usize {
            return Err(CapacityError);
        }
        for bit in ShareableClbit::iter_anonymous(num) {
            let index = self.clbits.add_unique_within_capacity(bit.clone());
            self.clbit_indices
                .insert(bit, BitLocations::new(index.0, []));
        }
        Ok(())
    }

    /// Modify `self` to mark its qubits as physical.
    ///
    /// This deletes the information about the virtual registers, and replaces it with the single
    /// (implicitly) physical register.  This method does not need to traverse the circuit.
    ///
    /// The qubit indices all stay the same; effectively, this is the application of the "trivial"
    /// layout.  If the incoming circuit is supposed to be considered physical, this method can be
    /// used to ensure it is in the canonical physical form.
    ///
    /// # Panics
    ///
    /// If `num_qubits` is less than the number of qubits in the circuit already.
    pub fn make_physical(&mut self, num_qubits: u32) {
        // If this method needs updating, `DAGCircuit::make_physical` probably does too.
        assert!(
            num_qubits as usize >= self.num_qubits(),
            "number of qubits {num_qubits} too small for circuit"
        );
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
        self.qubits = registry;
        self.qregs = register_data;
        self.qubit_indices = locator;
    }

    /// Append a standard gate to this CircuitData
    pub fn push_standard_gate(
        &mut self,
        operation: StandardGate,
        params: &[Param],
        qargs: &[Qubit],
    ) -> Result<(), CircuitDataError> {
        let params = (!params.is_empty()).then(|| Box::new(params.iter().cloned().collect()));
        let qubits = self.qargs_interner.insert(qargs);
        self.push(PackedInstruction::from_standard_gate(
            operation, params, qubits,
        ))
    }

    /// Append a packed operation to this CircuitData.
    ///
    /// If a [ControlFlow] operation is provided, the blocks given in
    /// `params` must already be registered with the circuit.
    pub fn push_packed_operation(
        &mut self,
        operation: PackedOperation,
        params: Option<Parameters<Block>>,
        qargs: &[Qubit],
        cargs: &[Clbit],
    ) -> Result<(), CircuitDataError> {
        let qubits = self.qargs_interner.insert(qargs);
        let clbits = self.cargs_interner.insert(cargs);
        self.push(PackedInstruction {
            op: operation,
            qubits,
            clbits,
            params: params.map(Box::new),
            label: None,
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        })
    }

    pub fn set_global_phase_f64(&mut self, phase: f64) {
        self.clear_global_phase();
        self.global_phase = Param::Float(phase.rem_euclid(::std::f64::consts::TAU));
    }

    /// Reset the global phase of the circuit to zero, updating any parameter tracking.
    fn clear_global_phase(&mut self) {
        let old = ::std::mem::replace(&mut self.global_phase, Param::Float(0.0));
        let Param::ParameterExpression(expr) = old else {
            return;
        };
        for symbol in expr.iter_symbols() {
            match self.param_table.remove_use(
                ParameterUuid::from_symbol(symbol),
                ParameterUse::GlobalPhase,
            ) {
                Ok(_)
                | Err(ParameterTableError::ParameterNotTracked(_))
                | Err(ParameterTableError::UsageNotTracked(_))
                | Err(ParameterTableError::NameConflict(_)) => (),
                // Any errors added later might want propagating.
            }
        }
    }

    #[inline]
    fn track_instruction_blocks(&mut self, index: usize) {
        self.data[index]
            .blocks_view()
            .iter()
            .for_each(|block| self.blocks.increment(*block))
    }
    #[inline]
    fn untrack_instruction_blocks(&mut self, index: usize) {
        self.data[index]
            .blocks_view()
            .iter()
            .for_each(|block| _ = self.blocks.decrement(*block))
    }

    /// Add the entries from the `PackedInstruction` at the given index to the internal parameter
    /// table.
    fn track_instruction_parameters(
        &mut self,
        instruction_index: usize,
    ) -> Result<(), CircuitDataError> {
        let instr = &self.data[instruction_index];
        let Some(parameters) = self.data[instruction_index].params.as_deref() else {
            return Ok(());
        };
        match parameters {
            Parameters::Params(params) => {
                for (index, param) in params.iter().enumerate() {
                    if matches!(param, Param::Float(_)) {
                        continue;
                    }
                    let usage = ParameterUse::Index {
                        instruction: instruction_index,
                        parameter: index as u32,
                    };
                    for symbol in param.iter_parameters()? {
                        self.param_table.track(&symbol, Some(usage))?;
                    }
                }
            }
            Parameters::Blocks(_) => {
                let view = ControlFlowView::try_from_instruction(instr, &self.blocks)
                    .expect("all instructions with blocks should be control flow");
                for_each_symbol_use_in_control_flow(instruction_index, view, |symbol, usage| {
                    self.param_table.track(symbol, Some(usage))?;
                    Ok(())
                })?
            }
        }
        Ok(())
    }

    /// Remove the entries from the `PackedInstruction` at the given index from the internal
    /// parameter table.
    fn untrack_instruction_parameters(
        &mut self,
        instruction_index: usize,
    ) -> Result<(), CircuitDataError> {
        let instr = &self.data[instruction_index];
        let Some(parameters) = self.data[instruction_index].params.as_deref() else {
            return Ok(());
        };
        match parameters {
            Parameters::Params(params) => {
                for (index, param) in params.iter().enumerate() {
                    if matches!(param, Param::Float(_)) {
                        continue;
                    }
                    let usage = ParameterUse::Index {
                        instruction: instruction_index,
                        parameter: index as u32,
                    };
                    for symbol in param.iter_parameters()? {
                        self.param_table.untrack(&symbol, usage)?;
                    }
                }
            }
            Parameters::Blocks(_) => {
                let view = ControlFlowView::try_from_instruction(instr, &self.blocks)
                    .expect("all instructions with blocks should be control flow");
                for_each_symbol_use_in_control_flow(instruction_index, view, |symbol, usage| {
                    self.param_table.untrack(symbol, usage)?;
                    Ok(())
                })?
            }
        }
        Ok(())
    }

    /// Retrack the entire `ParameterTable`.
    ///
    /// This is necessary each time an insertion or removal occurs on `self.data` other than in the
    /// last position.
    fn reindex_parameter_table(&mut self) -> Result<(), CircuitDataError> {
        self.param_table.clear();

        for inst_index in 0..self.len() {
            self.track_instruction_parameters(inst_index)?;
        }
        if matches!(self.global_phase, Param::Float(_)) {
            return Ok(());
        }
        for symbol in self.global_phase.iter_parameters()? {
            self.param_table
                .track(&symbol, Some(ParameterUse::GlobalPhase))?;
        }
        Ok(())
    }

    /// Native internal driver of `__delitem__` that uses a Rust-space version of the
    /// `SequenceIndex`.  This assumes that the `SequenceIndex` contains only in-bounds indices, and
    /// panics if not.
    fn delitem(&mut self, indices: SequenceIndex) -> Result<(), CircuitDataError> {
        // We need to delete in reverse order so we don't invalidate higher indices with a deletion.
        for index in indices.descending() {
            self.untrack_instruction_blocks(index);
            self.data.remove(index);
        }
        if !indices.is_empty() {
            self.reindex_parameter_table()?;
        }
        Ok(())
    }

    /// Returns an iterator over all the instructions present in the circuit.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &PackedInstruction> {
        self.data.iter()
    }

    /// Get the sorted symbols in this circuit.
    pub fn parameters(&self) -> &[Symbol] {
        self.param_table.symbols()
    }

    /// Does the circuit use this `Symbol` as a parameter?
    pub fn uses_parameter(&self, sym: &Symbol) -> bool {
        self.param_table.contains(sym)
    }

    /// Get the unsorted symbols in this circuit.
    pub fn iter_parameters(&self) -> impl Iterator<Item = &Symbol> {
        self.param_table.iter_symbols()
    }

    /// Assigns parameters to circuit data based on a mapping of `ParameterUuid` : `Param`.
    /// This mapping assumes that the provided `ParameterUuid` keys are instances
    /// of `ParameterExpression`.
    pub fn assign_parameters_from_mapping<I, T>(&mut self, iter: I) -> Result<(), CircuitDataError>
    where
        I: IntoIterator<Item = (ParameterUuid, T)>,
        T: AsRef<Param>,
    {
        let mut items = Vec::new();
        for (param_uuid, value) in iter {
            // Assume all the Parameters are already in the circuit
            let symbol = self
                .get_parameter_by_uuid(param_uuid)
                .ok_or(ParameterTableError::ParameterNotTracked(param_uuid))?;
            items.push((
                symbol.clone(),
                value.as_ref().clone(),
                self.param_table.pop(param_uuid)?,
            ));
        }
        self.assign_parameters_inner(items)
    }

    /// Assign a single parameter to a value.
    ///
    /// This is not generally efficient, and mostly just a convenience for the recursive case of
    /// control flow.
    fn assign_single_parameter(
        &mut self,
        symbol: Symbol,
        value: &Param,
    ) -> Result<(), CircuitDataError> {
        let Ok(uses) = self.param_table.pop(ParameterUuid::from_symbol(&symbol)) else {
            return Ok(());
        };
        self.assign_parameters_inner(Some((symbol, value, uses)))
    }

    /// Returns an immutable view of the Interner used for Qargs
    pub fn qargs_interner(&self) -> &Interner<[Qubit]> {
        &self.qargs_interner
    }

    /// Merge the `qargs` in a different [Interner] into this Circuit, remapping the qubits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one circuit to
    /// another, like when composing two circuits. See [Interner::merge_map_slice] for more
    /// information on the mapping function.
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

    /// Merge the `qargs` in a different [Interner] into this circuit, remapping the qubits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one circuit to
    /// another, like when composing two circuits. See [Interner::merge_map_slice] for more
    /// information on the mapping function.
    pub fn merge_qargs(
        &mut self,
        other: &Interner<[Qubit]>,
        map_fn: impl FnMut(&Qubit) -> Option<Qubit>,
    ) -> InternedMap<[Qubit]> {
        let mut out = InternedMap::new();
        self.merge_qargs_using(other, map_fn, &mut out);
        out
    }

    /// Returns an immutable view of the Interner used for Cargs
    pub fn cargs_interner(&self) -> &Interner<[Clbit]> {
        &self.cargs_interner
    }

    /// Returns an immutable view of the Global Phase `Param` of the circuit
    pub fn global_phase(&self) -> &Param {
        &self.global_phase
    }

    /// Returns an immutable view of the Qubits registered in the circuit
    pub fn qubits(&self) -> &ObjectRegistry<Qubit, ShareableQubit> {
        &self.qubits
    }

    /// Returns an immutable view of the Classical bits registered in the circuit
    pub fn clbits(&self) -> &ObjectRegistry<Clbit, ShareableClbit> {
        &self.clbits
    }

    /// Returns an immutable view of the [QuantumRegister] instances in the circuit.
    pub fn qregs(&self) -> &[QuantumRegister] {
        self.qregs.registers()
    }

    /// Returns an immutable view of the [ClassicalRegister] instances in the circuit.
    pub fn cregs(&self) -> &[ClassicalRegister] {
        self.cregs.registers()
    }

    /// Returns the index of the qubit in the circuit
    pub fn qubit_index(&self, qubit: &ShareableQubit) -> Option<u32> {
        self.qubits.find(qubit).map(|qubit| qubit.0)
    }

    /// Returns the index of the clbit in the circuit
    pub fn clbit_index(&self, clbit: &ShareableClbit) -> Option<u32> {
        self.clbits.find(clbit).map(|clbit| clbit.0)
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

    /// Returns an immutable view of the qubit locations of the [CircuitData]
    #[inline(always)]
    pub fn qubit_indices(&self) -> &BitLocator<ShareableQubit, QuantumRegister> {
        &self.qubit_indices
    }

    /// Returns an immutable view of the clbit locations of the [CircuitData]
    #[inline(always)]
    pub fn clbit_indices(&self) -> &BitLocator<ShareableClbit, ClassicalRegister> {
        &self.clbit_indices
    }

    /// Unpacks from interned value to `[Qubit]`
    pub fn get_qargs(&self, index: Interned<[Qubit]>) -> &[Qubit] {
        self.qargs_interner().get(index)
    }

    /// Insert qargs into the interner and return the interned value
    pub fn add_qargs(&mut self, qubits: &[Qubit]) -> Interned<[Qubit]> {
        self.qargs_interner.insert(qubits)
    }

    /// Unpacks from InternerIndex to `[Clbit]`
    pub fn get_cargs(&self, index: Interned<[Clbit]>) -> &[Clbit] {
        self.cargs_interner().get(index)
    }

    /// Insert cargs into the interner and return the interned value
    pub fn add_cargs(&mut self, clbits: &[Clbit]) -> Interned<[Clbit]> {
        self.cargs_interner.insert(clbits)
    }
    /// Internal method for assigning parameters.
    ///
    /// Note that currently if any [ParameterUse] identifies a basic block,
    /// the block will be replaced for all instructions in the circuit, even if
    /// they aren't included in the use set. This is because we would otherwise
    /// have to keep a mapping of [Block] to all instructions that use it and
    /// add/create an additional block if only some uses are replaced. This is
    /// not a use-case that we currently have.
    fn assign_parameters_inner<I, T>(&mut self, iter: I) -> Result<(), CircuitDataError>
    where
        I: IntoIterator<Item = (Symbol, T, HashSet<ParameterUse>)>,
        T: AsRef<Param> + Clone,
    {
        let inconsistent = || panic!("internal error: parameter table is in an inconsistent state");
        // Bind a single `Parameter` into a `ParameterExpression`.
        let bind_expr = |expr: &ParameterExpression,
                         symbol: &Symbol,
                         value: &Param,
                         coerce: bool|
         -> Result<Param, CircuitDataError> {
            let new_expr = match value {
                Param::Float(f) => {
                    let map: HashMap<&Symbol, Value> = HashMap::from([(symbol, Value::Real(*f))]);
                    expr.bind(&map, false)?
                }
                Param::ParameterExpression(e) => {
                    let map: HashMap<Symbol, ParameterExpression> =
                        HashMap::from([(symbol.clone(), e.as_ref().clone())]);
                    expr.subs(&map, false)?
                }
                Param::Obj(ob) => {
                    Python::attach(|py| -> Result<_, CircuitDataError> {
                        // The integer handling is only needed to support the case where an int is
                        // passed in directly instead of a float. This will be handled when we add
                        // int to the param enum to support dt target.
                        if let Ok(int) = ob.extract::<i64>(py) {
                            let map: HashMap<&Symbol, Value> =
                                HashMap::from([(symbol, Value::Int(int))]);
                            Ok(expr.bind(&map, false)?)
                        } else if let Ok(c) = ob.extract::<Complex64>(py) {
                            let map: HashMap<&Symbol, Value> =
                                HashMap::from([(symbol, Value::Complex(c))]);
                            Ok(expr.bind(&map, false)?)
                        } else {
                            Err(PyTypeError::new_err(format!(
                                "Cannot assign object ({ob}) object to parameter."
                            ))
                            .into())
                        }
                    })?
                }
            };
            // Param::from_expr() only errors in the python path when calling Python
            Ok(Param::from_expr(new_expr, coerce)?)
        };

        let mut user_operations = HashMap::new();
        let mut uuids = Vec::new();
        // Mark blocks that we've already edited.
        let mut seen_blocks = HashSet::new();
        for (symbol, value, uses) in iter {
            debug_assert!(!uses.is_empty());
            seen_blocks.clear();
            uuids.clear();
            for inner_symbol in value.as_ref().iter_parameters()? {
                uuids.push(self.param_table.track(&inner_symbol, None)?)
            }
            for usage in uses {
                let (instruction, parameter) = match usage {
                    // Peel off the global-phase handling first because it's much simpler.
                    ParameterUse::GlobalPhase => {
                        let Param::ParameterExpression(expr) = &self.global_phase else {
                            inconsistent()
                        };
                        self.set_global_phase_param(bind_expr(
                            expr,
                            &symbol,
                            value.as_ref(),
                            true,
                        )?)?;
                        continue;
                    }
                    ParameterUse::Index {
                        instruction,
                        parameter,
                    } => (instruction, parameter as usize),
                };
                let previous = &mut self.data[instruction];
                match previous.op.view() {
                    OperationRef::StandardGate(_)
                    | OperationRef::StandardInstruction(_)
                    | OperationRef::Unitary(_)
                    | OperationRef::PauliProductMeasurement(_)
                    | OperationRef::PauliProductRotation(_)
                    | OperationRef::CustomOperation(_) => {
                        // TODO: `PauliProductRotation` actually stores a `Param` inside itself,
                        // which this code does not account for.  We most likely need to make an
                        // `OperationRefMut` in order to modify that without cloning the whole
                        // object.

                        let params = previous.params_mut();
                        let Param::ParameterExpression(expr) = &params[parameter] else {
                            panic!("internal error: parameter table contains non-parameters");
                        };
                        let new_param = bind_expr(expr, &symbol, value.as_ref(), true)?;

                        // standard gates don't allow for complex parameters
                        if let Param::Obj(expr) = &new_param {
                            return Err(CircuitDataError::StandardGateParameterIsComplex(
                                previous.op.name().to_string(),
                                format!("{expr:?}"),
                            ));
                        }

                        params[parameter] = new_param.clone();

                        // PauliProductRotation also stores its parameter within itself.
                        if let OperationRef::PauliProductRotation(ppr) = previous.op.view() {
                            // TODO: if we had an `OperationMut` we wouldn't need this clone, but
                            // that's too much of an addition at the time this is added.
                            previous.op = PauliBased::PauliProductRotation(PauliProductRotation {
                                angle: new_param,
                                ..ppr.clone()
                            })
                            .into();
                        }
                        for uuid in uuids.iter() {
                            self.param_table.add_use(*uuid, usage)?;
                        }
                        #[cfg(feature = "cache_pygates")]
                        {
                            // Standard gates can all rebuild their definitions, so if the
                            // cached py_op exists, discard it to prompt the instruction
                            // to rebuild its cached python gate upon request later on. This is
                            // done to avoid an unintentional duplicated reference to the same gate
                            // instance in python. For more information, see
                            // https://github.com/Qiskit/qiskit/issues/13504
                            previous.py_op.take();
                        }
                    }
                    OperationRef::ControlFlow(op) => {
                        let blocks = previous.blocks_view();
                        let block_to_edit = match &op.control_flow {
                            ControlFlow::BreakLoop => inconsistent(),
                            ControlFlow::ContinueLoop => inconsistent(),
                            ControlFlow::ForLoop { .. } => {
                                match parameter {
                                    // In Python land, the loop body exists at parameter
                                    // position 2.
                                    2 => blocks[0],
                                    _ => inconsistent(),
                                }
                            }
                            // Most control flow instructions use the parameters for
                            // *just* their blocks.
                            _ => blocks[parameter],
                        };
                        if !seen_blocks.contains(&block_to_edit) {
                            self.blocks[block_to_edit]
                                .assign_single_parameter(symbol.clone(), value.as_ref())?;
                            seen_blocks.insert(block_to_edit);
                        }
                        for uuid in uuids.iter() {
                            self.param_table.add_use(*uuid, usage)?
                        }
                        #[cfg(feature = "cache_pygates")]
                        {
                            let previous = &mut self.data[instruction];
                            previous.py_op.take();
                        }
                    }
                    OperationRef::PyCustom(inst) => {
                        // Track user operations we've seen so we can rebind their definitions.
                        // Strictly this can add the same binding pair more than once, if an
                        // instruction has the same `Parameter` in several of its `params`, but
                        // we're going to turn that into a `dict` anyway, so it doesn't matter.
                        user_operations
                            .entry(instruction)
                            .or_insert_with(Vec::new)
                            .push((symbol.clone(), value.as_ref().clone()));
                        // The separate Python-only attachment here is just a lifetime trick because
                        // `previous` and `inst` share a (mutable) lifetime and so can't be captured
                        // by the same closure.
                        let py_ob = Python::attach(|py| inst.ob.clone_ref(py));
                        Python::attach(|py| -> Result<_, CircuitDataError> {
                            let py_ob = py_ob.into_bound(py);
                            let validate_parameter_attr = intern!(py, "validate_parameter");
                            let assign_parameters_attr = intern!(py, "assign_parameters");
                            // All "user" operations (e.g. PyOperation) use Parameters::Param.
                            let previous_param = &previous.params_view()[parameter];
                            let new_param = match previous_param {
                                Param::Float(_) => inconsistent(),
                                Param::ParameterExpression(expr) => {
                                    let new_param =
                                        bind_expr(expr, &symbol, value.as_ref(), false)?;
                                    // Historically, `assign_parameters` called `validate_parameter`
                                    // only when a `ParameterExpression` became fully bound.  Some
                                    // "generalised" (or user) gates fail without this, though
                                    // arguably, that's them indicating they shouldn't be allowed to
                                    // be parametric.
                                    //
                                    // Our `bind_expr` coercion means that a non-parametric
                                    // `ParameterExperssion` after binding would have been coerced
                                    // to a numeric quantity already, so the match here is
                                    // definitely parameterized.
                                    match &new_param {
                                        Param::ParameterExpression(expr) => {
                                            match expr.try_to_value(true) {
                                                Ok(_) => {
                                                    // fully bound, validate parameters
                                                    Param::extract_no_coerce(
                                                        py_ob
                                                            .call_method1(
                                                                validate_parameter_attr,
                                                                (new_param,),
                                                            )?
                                                            .as_borrowed(),
                                                    )?
                                                }
                                                Err(_) => new_param, // not bound yet, cannot validate
                                            }
                                        }
                                        new_param => Param::extract_no_coerce(
                                            py_ob
                                                .call_method1(
                                                    validate_parameter_attr,
                                                    (new_param,),
                                                )?
                                                .as_borrowed(),
                                        )?,
                                    }
                                }
                                Param::Obj(obj) => {
                                    let obj = obj.bind_borrowed(py);
                                    if !obj.is_instance(QUANTUM_CIRCUIT.get_bound(py))? {
                                        inconsistent()
                                    }
                                    Param::extract_no_coerce(
                                        obj.call_method(
                                            assign_parameters_attr,
                                            ([(symbol.clone(), value.as_ref().clone_ref(py))]
                                                .into_py_dict(py)?,),
                                            Some(
                                                &[("inplace", false), ("flat_input", true)]
                                                    .into_py_dict(py)?,
                                            ),
                                        )?
                                        .as_borrowed(),
                                    )?
                                }
                            };
                            py_ob
                                .getattr(intern!(py, "params"))?
                                .set_item(parameter, new_param)?;
                            let new_op = py_ob.extract::<OperationFromPython<CircuitData>>()?;
                            previous.op = new_op.operation;
                            previous.params = new_op.params.map(|params| {
                                Box::new(params.map_blocks(|_| panic!("unexpected control flow")))
                            });
                            previous.label = new_op.label;
                            #[cfg(feature = "cache_pygates")]
                            {
                                previous.py_op = py_ob.unbind().into();
                            }
                            for uuid in uuids.iter() {
                                self.param_table.add_use(*uuid, usage)?
                            }
                            Ok(())
                        })?;
                    }
                }
            }
        }

        // handle custom gates, this can only happen in Py-space
        if !user_operations.is_empty() {
            Python::attach(|py| -> PyResult<()> {
                let _definition_attr = intern!(py, "_definition");
                let assign_parameters_attr = intern!(py, "assign_parameters");

                let assign_kwargs = [("inplace", true), ("flat_input", true), ("strict", false)]
                    .into_py_dict(py)
                    .unwrap();
                for (instruction, bindings) in user_operations {
                    let OperationRef::PyCustom(inst) = self.data[instruction].op.view() else {
                        return Err(PyRuntimeError::new_err(
                            "internal logic error: a 'user operation' was stored in the wrong type",
                        ));
                    };
                    let definition_cache = match inst.kind {
                        PyOpKind::Operation => {
                            // `Operation` instances don't have a `definition` as part of their
                            // interfaces, but they might be an `AnnotatedOperation`, which is one
                            // of our special built-ins.  This should be handled more completely in
                            // the user-customisation interface by a delegating method, but that's
                            // not the data model we currently have.
                            let ob = inst.ob.bind(py);
                            if ob.is_instance(ANNOTATED_OPERATION.get_bound(py))? {
                                ob.getattr(intern!(py, "base_op"))?
                                    .getattr(_definition_attr)?
                            } else {
                                continue;
                            }
                        }
                        PyOpKind::Instruction | PyOpKind::Gate => {
                            inst.ob.bind(py).getattr(_definition_attr)?
                        }
                    };
                    if !definition_cache.is_none() {
                        definition_cache.call_method(
                            assign_parameters_attr,
                            (bindings.into_py_dict(py)?.into_any().unbind(),),
                            Some(&assign_kwargs),
                        )?;
                    }
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Retrieves the python `Param` object based on its `ParameterUuid`.
    pub fn get_parameter_by_uuid(&self, uuid: ParameterUuid) -> Option<Symbol> {
        self.param_table.parameter_by_uuid(uuid).cloned()
    }

    /// Get an immutable view of the instructions in the circuit data
    pub fn data(&self) -> &[PackedInstruction] {
        &self.data
    }

    /// Consume the CircuitData and create an iterator of the [`PackedInstruction`] objects in the
    /// circuit.
    pub fn into_data_iter(self) -> impl ExactSizeIterator<Item = PackedInstruction> {
        self.data.into_iter()
    }

    /// Returns an immutable view of the vars and stretches in the circuit
    pub fn vars_stretches_view(&self) -> &VarStretchContainer {
        &self.vars_stretches
    }

    /// Remove the label for an instruction in the circuit
    ///
    /// This modifies the circuit in place and sets the label
    /// field of an instruction to ``None``.
    ///
    /// # Arguments
    ///
    /// * index: The index of the instruction in the circuit to remove the label of.
    pub fn invalidate_label(&mut self, index: usize) {
        self.data[index].label = None;
    }

    /// Clone an empty CircuitData from a given reference.
    ///
    /// The new copy will have the global properties from the provided `CircuitData`.
    /// The bit data fields and interners, global phase, etc will be copied to
    /// the new returned `CircuitData`, but the `data` field's instruction list will
    /// be empty. This can be useful for scenarios where you want to rebuild a copy
    /// of the circuit from a reference but insert new gates in the middle.
    ///
    /// # Arguments
    ///
    /// * other - The other `CircuitData` to clone an empty `CircuitData` from.
    /// * capacity - The capacity for instructions to use in the output `CircuitData`
    ///   If `None` the length of `other` will be used, if `Some` the integer
    ///   value will be used as the capacity.
    /// * vars_mode - The mode to use for handling variables.
    /// * blocks_mode - The mode to use for handling basic blocks registered with the circuit.
    pub fn clone_empty_like(
        other: &Self,
        capacity: Option<usize>,
        vars_mode: VarsMode,
        blocks_mode: BlocksMode,
    ) -> Result<Self, CircuitDataError> {
        let res = CircuitData {
            data: Vec::with_capacity(capacity.unwrap_or(other.data.len())),
            qargs_interner: other.qargs_interner.clone(),
            cargs_interner: other.cargs_interner.clone(),
            qubits: other.qubits.clone(),
            clbits: other.clbits.clone(),
            blocks: if blocks_mode == BlocksMode::Keep {
                other.blocks.clone()
            } else {
                Default::default()
            },
            param_table: ParameterTable::new(),
            global_phase: Param::Float(0.0),
            qregs: other.qregs.clone(),
            cregs: other.cregs.clone(),
            qubit_indices: other.qubit_indices.clone(),
            clbit_indices: other.clbit_indices.clone(),
            vars_stretches: match vars_mode {
                VarsMode::Alike => other.vars_stretches.clone(),
                VarsMode::Captures => other.vars_stretches.clone_as_captures(),
                VarsMode::Drop => VarStretchContainer::new(),
            },
        };
        Ok(res)
    }

    /// Append a PackedInstruction to the circuit data.
    ///
    /// # Arguments
    ///
    /// * packed: The new packed instruction to insert to the end of the CircuitData
    ///   The qubits and clbits **must** already be present in the interner for this
    ///   function to work. If they are not this will corrupt the circuit.
    pub fn push(&mut self, packed: PackedInstruction) -> Result<(), CircuitDataError> {
        let new_index = self.len();
        self.data.push(packed);
        self.track_instruction_blocks(new_index);
        self.track_instruction_parameters(new_index)
    }

    /// Add a param to the current global phase of the circuit
    pub fn add_global_phase(&mut self, value: &Param) -> Result<(), CircuitDataError> {
        match value {
            Param::Obj(_) => Err(CircuitDataError::InvalidGlobalPhaseType),
            _ => self.set_global_phase_param(add_global_phase(&self.global_phase, value)),
        }
    }

    /// Adds a variable to the circuit and register it as a var of type `var_type`.
    ///
    /// # Arguments
    ///
    /// * `var` - The [expr::Var] object to add.
    /// * `var_type` - The classification of the added variable within the circuit.
    ///
    /// # Returns
    ///
    /// The [Var] index of the newly added var within the circuit on success, or an error otherwise.
    #[inline]
    pub fn add_var(&mut self, var: expr::Var, var_type: VarType) -> Result<Var, CircuitDataError> {
        Ok(self.vars_stretches.add_var(var, var_type)?)
    }

    /// Adds a stretch to the circuit and register it as a stretch of type `stretch_type`.
    ///
    /// # Arguments
    ///
    /// * `stretch` - The [expr::Stretch] object to add.
    /// * `stretch_type` - The classification of the added stretch within the circuit.
    ///
    /// # Returns
    ///
    /// The [Stretch] index of the newly added stretch within the circuit on success, or an error otherwise.
    #[inline]
    pub fn add_stretch(
        &mut self,
        stretch: expr::Stretch,
        stretch_type: StretchType,
    ) -> Result<Stretch, CircuitDataError> {
        Ok(self.vars_stretches.add_stretch(stretch, stretch_type)?)
    }

    /// Return a copy of the circuit with instructions in reverse order
    pub fn reverse(self) -> Result<Self, CircuitDataError> {
        let mut out = Self::clone_empty_like(
            &self,
            Some(self.data().len()),
            VarsMode::Alike,
            BlocksMode::Keep,
        )?;
        for inst in self.data().iter().rev() {
            out.push(inst.clone())?;
        }
        Ok(out)
    }

    /// Checks whether the circuit has an instance of :class:`.ControlFlowOp`
    /// present amongst its operations.
    pub fn has_control_flow_op(&self) -> bool {
        self.data
            .iter()
            .any(|inst| inst.op.try_control_flow().is_some())
    }

    pub fn insert(
        &mut self,
        mut index: isize,
        packed: PackedInstruction,
    ) -> Result<(), CircuitDataError> {
        // `list.insert` has special-case extra clamping logic for its index argument.
        let index = {
            if index < 0 {
                // This can't exceed `isize::MAX` because `self.data[0]` is larger than a byte.
                index += self.len() as isize;
            }
            if index < 0 {
                0
            } else if index as usize > self.len() {
                self.len()
            } else {
                index as usize
            }
        };
        self.data.insert(index, packed);
        self.track_instruction_blocks(index);
        if index == self.data.len() - 1 {
            self.track_instruction_parameters(index)?;
        } else {
            self.reindex_parameter_table()?;
        }
        Ok(())
    }

    pub fn extend(&mut self, other: &CircuitData) -> Result<(), CircuitDataError> {
        // Fast path to avoid unnecessary construction of CircuitInstruction instances.
        self.data.reserve(other.data.len());
        for inst in other.data.iter() {
            let qubits = other
                .qargs_interner
                .get(inst.qubits)
                .iter()
                .map(|b| self.qubits.find(other.qubits.get(*b).unwrap()).unwrap())
                .collect::<Vec<Qubit>>();
            let clbits = other
                .cargs_interner
                .get(inst.clbits)
                .iter()
                .map(|b| self.clbits.find(other.clbits.get(*b).unwrap()).unwrap())
                .collect::<Vec<Clbit>>();
            let qubits_id = self.qargs_interner.insert_owned(qubits);
            let clbits_id = self.cargs_interner.insert_owned(clbits);
            let params = inst.params.as_ref().map(|params| {
                Box::new(params.map_blocks_ref(|b| self.blocks.push(other.blocks[*b].clone())))
            });
            self.push(PackedInstruction {
                op: inst.op.clone(),
                qubits: qubits_id,
                clbits: clbits_id,
                params,
                label: inst.label.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: inst.py_op.clone(),
            })?;
        }
        Ok(())
    }

    pub fn assign_parameters_from_array(
        &mut self,
        array: ArrayView1<f64>,
    ) -> Result<(), CircuitDataError> {
        if array.len() != self.param_table.num_parameters() {
            return Err(CircuitDataError::ParameterSliceLenMismatch);
        }
        let mut old_table = std::mem::take(&mut self.param_table);
        self.assign_parameters_inner(
            array
                .iter()
                .map(|value| Param::Float(*value))
                .zip(old_table.drain_ordered())
                .map(|(value, (obj, uses))| (obj, value, uses)),
        )
    }

    pub fn assign_parameters_from_slice(
        &mut self,
        slice: &[Param],
    ) -> Result<(), CircuitDataError> {
        if slice.len() != self.param_table.num_parameters() {
            return Err(CircuitDataError::ParameterSliceLenMismatch);
        }
        let mut old_table = std::mem::take(&mut self.param_table);
        self.assign_parameters_inner(
            slice
                .iter()
                .zip(old_table.drain_ordered())
                .map(|(value, (symbol, uses))| (symbol, value.clone(), uses)),
        )
    }

    pub fn clear(&mut self) {
        std::mem::take(&mut self.data);
        self.param_table.clear();
    }

    /// Counts the number of times each operation is used in the circuit.
    ///
    /// # Parameters
    /// - `self` - A mutable reference to the CircuitData struct.
    ///
    /// # Returns
    /// An IndexMap containing the operation names as keys and their respective counts as values.
    pub fn count_ops(&self) -> IndexMap<&str, usize> {
        let mut ops_count: IndexMap<&str, usize> = IndexMap::default();
        for instruction in &self.data {
            *ops_count.entry(instruction.op.name()).or_insert(0) += 1;
        }
        ops_count.par_sort_by(|_k1, v1, _k2, v2| v2.cmp(v1));
        ops_count
    }

    fn clear_all(&mut self) {
        // Clear anything that could have a reference cycle.
        self.data.clear();
        self.qubits.dispose();
        self.clbits.dispose();
        self.qregs.dispose();
        self.cregs.dispose();
        self.clbit_indices.dispose();
        self.qubit_indices.dispose();
        self.param_table.clear();
    }

    /// Set the global phase of the circuit.
    ///
    /// This method assumes that the parameter table is either fully consistent, or contains zero
    /// entries for the global phase, regardless of what value is currently stored there.  It's not
    /// uncommon for subclasses and other parts of Qiskit to have filled in the global phase field
    /// by copies or other means, before making the parameter table consistent.
    pub fn set_global_phase_param(&mut self, angle: Param) -> Result<(), CircuitDataError> {
        self.clear_global_phase();
        match &angle {
            Param::Float(angle) => {
                self.set_global_phase_f64(*angle);
                Ok(())
            }
            Param::ParameterExpression(expr) => {
                for symbol in expr.iter_symbols() {
                    self.param_table
                        .track(symbol, Some(ParameterUse::GlobalPhase))?;
                }
                self.global_phase = angle;
                Ok(())
            }
            _ => Err(CircuitDataError::InvalidGlobalPhaseType),
        }
    }

    pub fn num_nonlocal_gates(&self) -> usize {
        self.data
            .iter()
            .filter(|inst| inst.op.num_qubits() > 1 && !inst.op.directive())
            .count()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Helper struct for `assign_parameters` to allow use of `Param::extract_no_coerce` in
/// PyO3-provided `FromPyObject` implementations on containers.
#[repr(transparent)]
struct AssignParam(Param);
impl<'a, 'py> FromPyObject<'a, 'py> for AssignParam {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self(Param::extract_no_coerce(ob)?))
    }
}

/// Get the `Vec` of bits referred to by the specifier `specifier`.
///
/// Valid types for `specifier` are integers, bits of the correct type (as given in `type_`), or
/// iterables of one of those two scalar types.  Integers are interpreted as indices into the
/// sequence `bit_sequence`.  All allowed bits must be in `bit_set` (which should implement
/// fast lookup), which is assumed to contain the same bits as `bit_sequence`.
///
/// # Args
///   `specifier` - the Python object specifier to look up.
///   `bit_sequence` - The sequence of bit objects assumed to be the same bits as `bit_set`
///   ` bit_set` - The bit locator, contains the same bits as `bit_sequence`
///
/// # Returns
///     A list of the specified bits from `bits`. The `PyResult` error will contain a Python
///     `CircuitError` if an incorrect type or index is encountered, if the same bit is specified
///     more than once, or if the specifier is to a bit not in the `bit_set`.
fn bit_argument_conversion<B, R>(
    specifier: &Bound<PyAny>,
    bit_sequence: &[B],
    bit_set: &BitLocator<B, R>,
) -> PyResult<Vec<B>>
where
    B: Debug + Clone + Hash + Eq + for<'a, 'py> FromPyObject<'a, 'py>,
    R: Register + Debug + Clone + Hash + for<'a, 'py> FromPyObject<'a, 'py>,
{
    // The duplication between this function and `_bit_argument_conversion_scalar` is so that fast
    // paths return as quickly as possible, and all valid specifiers will resolve without needing to
    // try/catch exceptions (which is too slow for inner-loop code).
    if let Ok(bit) = specifier.extract() {
        if bit_set.contains_key(&bit) {
            return Ok(vec![bit]);
        }
        Err(CircuitError::new_err(format!(
            "Bit '{specifier}' is not in the circuit."
        )))
    } else if let Ok(sequence) = specifier.extract::<PySequenceIndex>() {
        match sequence {
            PySequenceIndex::Int(index) => {
                if let Ok(index) = PySequenceIndex::convert_idx(index, bit_sequence.len()) {
                    if let Some(bit) = bit_sequence.get(index).cloned() {
                        return Ok(vec![bit]);
                    }
                }
                Err(CircuitError::new_err(format!(
                    "Index {specifier} out of range for size {}.",
                    bit_sequence.len()
                )))
            }
            _ => {
                let Ok(sequence) = sequence.with_len(bit_sequence.len()) else {
                    return Ok(vec![]);
                };
                Ok(sequence
                    .iter()
                    .map(|index| &bit_sequence[index])
                    .cloned()
                    .collect())
            }
        }
    } else {
        if let Ok(iter) = specifier.try_iter() {
            return iter
                .map(|spec| -> PyResult<B> {
                    bit_argument_conversion_scalar(&spec?, bit_sequence, bit_set)
                })
                .collect::<PyResult<_>>();
        }
        let err_message = if let Ok(bit) = specifier.cast::<PyBit>() {
            format!(
                "Incorrect bit type: expected '{}' but got '{}'",
                stringify!(B),
                bit.get_type().name()?
            )
        } else {
            format!(
                "Invalid bit index: '{specifier}' of type '{}'",
                specifier.get_type().name()?
            )
        };
        Err(CircuitError::new_err(err_message))
    }
}

fn bit_argument_conversion_scalar<B, R>(
    specifier: &Bound<PyAny>,
    bit_sequence: &[B],
    bit_set: &BitLocator<B, R>,
) -> PyResult<B>
where
    B: Debug + Clone + Hash + Eq + for<'a, 'py> FromPyObject<'a, 'py>,
    R: Register + Debug + Clone + Hash + for<'a, 'py> FromPyObject<'a, 'py>,
{
    if let Ok(bit) = specifier.extract() {
        if bit_set.contains_key(&bit) {
            return Ok(bit);
        }
        Err(CircuitError::new_err(format!(
            "Bit '{specifier}' is not in the circuit."
        )))
    } else if let Ok(index) = specifier.extract::<isize>() {
        if let Some(bit) = PySequenceIndex::convert_idx(index, bit_sequence.len())
            .map(|index| bit_sequence.get(index).cloned())
            .map_err(|_| {
                CircuitError::new_err(format!(
                    "Index {specifier} out of range for size {}.",
                    bit_sequence.len()
                ))
            })?
        {
            Ok(bit)
        } else {
            Err(CircuitError::new_err(format!(
                "Index {specifier} out of range for size {}.",
                bit_sequence.len()
            )))
        }
    } else {
        let err_message = if let Ok(bit) = specifier.cast::<PyBit>() {
            format!(
                "Incorrect bit type: expected '{}' but got '{}'",
                stringify!(B),
                bit.get_type().name()?
            )
        } else {
            format!(
                "Invalid bit index: '{specifier}' of type '{}'",
                specifier.get_type().name()?
            )
        };
        Err(CircuitError::new_err(err_message))
    }
}

/// Perform an action for each `ParameterUse` of a `Symbol` within a control-flow view object from
/// the given circuit index.
///
/// This encapsulates the logic of both [CircuitData::track_parameters] and
/// [CircuitData::untrack_parameters].
fn for_each_symbol_use_in_control_flow(
    instruction_index: usize,
    cf: ControlFlowView<CircuitData>,
    mut action: impl FnMut(&Symbol, ParameterUse) -> Result<(), CircuitDataError>,
) -> Result<(), CircuitDataError> {
    match cf {
        ControlFlowView::ForLoop {
            loop_param,
            body,
            collection: _,
        } => {
            // The body is at `params[2]`.
            let usage = ParameterUse::Index {
                instruction: instruction_index,
                parameter: 2,
            };
            for symbol in body.parameters() {
                // Skip the loop variable itself — it is runtime-bound.
                if let Some(LoopParam::Parameter(loop_symbol)) = loop_param {
                    if symbol == loop_symbol {
                        continue;
                    }
                }
                action(symbol, usage)?;
            }
        }
        // For all these guys, the `params` field is the same as the `blocks` list.
        ControlFlowView::Box { .. }
        | ControlFlowView::BreakLoop
        | ControlFlowView::ContinueLoop
        | ControlFlowView::IfElse { .. }
        | ControlFlowView::Switch { .. }
        | ControlFlowView::While { .. } => {
            for (idx, body) in cf.blocks().iter().enumerate() {
                let usage = ParameterUse::Index {
                    instruction: instruction_index,
                    parameter: idx as u32,
                };
                for symbol in body.parameters() {
                    action(symbol, usage)?;
                }
            }
        }
    }
    Ok(())
}

#[pymethods]
impl PyCircuitData {
    #[new]
    #[pyo3(signature = (qubits=None, clbits=None, data=None, reserve=0, global_phase=Param::Float(0.0)))]
    pub fn py_new(
        qubits: Option<Vec<ShareableQubit>>,
        clbits: Option<Vec<ShareableClbit>>,
        data: Option<&Bound<PyAny>>,
        reserve: usize,
        global_phase: Param,
    ) -> PyResult<Self> {
        let mut self_: PyCircuitData = CircuitData::new(qubits, clbits, global_phase)?.into();
        self_.reserve(reserve);
        if let Some(data) = data {
            self_.extend(data)?;
        }
        Ok(self_)
    }

    pub fn __reduce__(self_: &Bound<PyCircuitData>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ty: Bound<PyType> = self_.get_type();
        let args = {
            let self_ = self_.borrow();
            (
                (!self_.inner.qubits.is_empty()).then_some(self_.inner.qubits.objects().clone()),
                (!self_.inner.clbits.is_empty()).then_some(self_.inner.clbits.objects().clone()),
                None::<()>,
                self_.len(),
                self_.inner.global_phase.clone(),
            )
        };
        let state = {
            let borrowed = self_.borrow();
            let vars_stretches_state = borrowed.vars_stretches.to_pickle(py);

            (
                borrowed.qregs.registers().to_vec(),
                borrowed.cregs.registers().to_vec(),
                borrowed.qubit_indices.cached(py).clone_ref(py),
                borrowed.clbit_indices.cached(py).clone_ref(py),
                vars_stretches_state.0, // identifiers vector
                vars_stretches_state.1, // Var objects
                vars_stretches_state.2, // Stretch objects
            )
        };
        (ty, args, state, self_.try_iter()?).into_py_any(py)
    }

    pub fn __setstate__(slf: &Bound<PyCircuitData>, state: CircuitDataState) -> PyResult<()> {
        let mut borrowed_mut = slf.borrow_mut();
        // Add the registers directly to the `RegisterData` struct
        // to not modify the bit indices.
        for qreg in state.0.into_iter() {
            borrowed_mut.inner.qregs.add_register(qreg, false)?;
        }
        for creg in state.1.into_iter() {
            borrowed_mut.inner.cregs.add_register(creg, false)?;
        }

        // After the registers are added, reset bit locations.
        borrowed_mut.inner.qubit_indices = BitLocator::from_py_dict(&state.2)?;
        borrowed_mut.inner.clbit_indices = BitLocator::from_py_dict(&state.3)?;

        borrowed_mut.inner.vars_stretches =
            VarStretchContainer::from_pickle(slf.py(), (state.4, state.5, state.6))?;

        Ok(())
    }

    /// Invokes callable ``func`` with each instruction's operation.
    ///
    /// Args:
    ///     func (Callable[[:class:`~.Operation`], None]):
    ///         The callable to invoke.
    #[pyo3(signature = (func))]
    pub fn foreach_op(&self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for inst in self.data().iter() {
            func.call1((self.unpack_py_op(py, inst)?,))?;
        }
        Ok(())
    }

    /// Invokes callable ``func`` with the positional index and operation
    /// of each instruction.
    ///
    /// Args:
    ///     func (Callable[[int, :class:`~.Operation`], None]):
    ///         The callable to invoke.
    #[pyo3(signature = (func))]
    pub fn foreach_op_indexed(&self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for (index, inst) in self.data().iter().enumerate() {
            func.call1((index, self.unpack_py_op(py, inst)?))?;
        }
        Ok(())
    }

    /// Invokes callable ``func`` with each instruction's operation, replacing the operation with
    /// the result, if the operation is not a standard gate without a condition.
    ///
    /// .. warning::
    ///
    ///     This is a shim for while there are still important components of the circuit still
    ///     implemented in Python space.  This method **skips** any instruction that contains an
    ///     non-conditional standard gate (which is likely to be most instructions).
    ///
    /// Args:
    ///     func (Callable[[:class:`~.Operation`], :class:`~.Operation`]):
    ///         A callable used to map original operations to their replacements.
    #[pyo3(signature = (func))]
    pub fn map_nonstandard_ops(&mut self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for index in 0..self.inner.data.len() {
            let instr = &self.data()[index];
            if instr.op.try_standard_gate().is_some() {
                continue;
            }
            let py_op = func.call1((self.unpack_py_op(py, instr)?,))?;
            let result = py_op.extract::<OperationFromPython<CircuitData>>()?;
            let params = self.inner.take_parameter_blocks(result.params);
            let inst = &mut self.inner.data[index];
            inst.op = result.operation;
            inst.params = params;
            inst.label = result.label;
            #[cfg(feature = "cache_pygates")]
            {
                inst.py_op = py_op.unbind().into();
            }
        }
        Ok(())
    }

    /// Replaces the bits of this container with the given ``qubits``
    /// and/or ``clbits``.
    ///
    /// The `:attr:`~.CircuitInstruction.qubits` and
    /// :attr:`~.CircuitInstruction.clbits` of existing instructions are
    /// reinterpreted using the new bit sequences on access.
    /// As such, the primary use-case for this method is to remap a circuit to
    /// a different set of bits in constant time relative to the number of
    /// instructions in the circuit.
    ///
    /// Args:
    ///     qubits (Iterable[:class:`.Qubit] | None):
    ///         The qubit sequence which should replace the container's
    ///         existing qubits, or ``None`` to skip replacement.
    ///     clbits (Iterable[:class:`.Clbit] | None):
    ///         The clbit sequence which should replace the container's
    ///         existing qubits, or ``None`` to skip replacement.
    ///
    /// Raises:
    ///     ValueError: A replacement sequence is smaller than the bit list
    ///         its contents would replace.
    ///
    /// .. note::
    ///
    ///     Instruction operations themselves are NOT adjusted.
    ///     To modify bits referenced by an operation, use
    ///     :meth:`~.CircuitData.foreach_op` or
    ///     :meth:`~.CircuitData.foreach_op_indexed` or
    ///     :meth:`~.CircuitData.map_nonstandard_ops` to adjust the operations manually
    ///     after calling this method.
    ///
    /// Examples:
    ///
    ///     The following :class:`.CircuitData` is reinterpreted as if its bits
    ///     were originally added in reverse.
    ///
    ///     .. code-block::
    ///
    ///         qr = QuantumRegister(3)
    ///         data = CircuitData(qubits=qr, data=[
    ///             CircuitInstruction(XGate(), [qr[0]], []),
    ///             CircuitInstruction(XGate(), [qr[1]], []),
    ///             CircuitInstruction(XGate(), [qr[2]], []),
    ///         ])
    ///
    ///         data.replace_bits(qubits=reversed(qr))
    ///         assert(data == [
    ///             CircuitInstruction(XGate(), [qr[2]], []),
    ///             CircuitInstruction(XGate(), [qr[1]], []),
    ///             CircuitInstruction(XGate(), [qr[0]], []),
    ///         ])
    #[pyo3(signature = (qubits=None, clbits=None, qregs=None, cregs=None))]
    pub fn replace_bits(
        &mut self,
        qubits: Option<Vec<ShareableQubit>>,
        clbits: Option<Vec<ShareableClbit>>,
        qregs: Option<Vec<QuantumRegister>>,
        cregs: Option<Vec<ClassicalRegister>>,
    ) -> PyResult<()> {
        let qubits_is_some = qubits.is_some();
        let clbits_is_some = clbits.is_some();
        let mut temp = CircuitData::new(qubits, clbits, self.inner.global_phase.clone())?;

        // Add qregs if provided.
        if let Some(qregs) = qregs {
            for qreg in qregs {
                temp.add_qreg(qreg, true)?;
            }
        }
        // Add cregs if provided.
        if let Some(cregs) = cregs {
            for creg in cregs {
                temp.add_creg(creg, true)?;
            }
        }

        if qubits_is_some {
            if temp.num_qubits() < self.num_qubits() {
                return Err(PyValueError::new_err(format!(
                    "Replacement 'qubits' of size {:?} must contain at least {:?} bits.",
                    temp.num_qubits(),
                    self.num_qubits(),
                )));
            }
            std::mem::swap(&mut temp.qubits, &mut self.inner.qubits);
            std::mem::swap(&mut temp.qregs, &mut self.inner.qregs);
            std::mem::swap(&mut temp.qubit_indices, &mut self.inner.qubit_indices);
        }
        if clbits_is_some {
            if temp.num_clbits() < self.num_clbits() {
                return Err(PyValueError::new_err(format!(
                    "Replacement 'clbits' of size {:?} must contain at least {:?} bits.",
                    temp.num_clbits(),
                    self.num_clbits(),
                )));
            }
            std::mem::swap(&mut temp.clbits, &mut self.inner.clbits);
            std::mem::swap(&mut temp.cregs, &mut self.inner.cregs);
            std::mem::swap(&mut temp.clbit_indices, &mut self.inner.clbit_indices);
        }
        Ok(())
    }

    #[getter]
    pub fn global_phase(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.inner.global_phase.clone().into_py_any(py)
    }

    /// A dict mapping Qubit instances to tuple comprised of 0) the corresponding index in
    /// circuit.qubits and 1) a list of Register-int pairs for each Register containing the Bit and
    /// its index within that register.
    #[getter("_qubit_indices")]
    pub fn get_qubit_indices(&self, py: Python) -> &Py<PyDict> {
        self.inner.qubit_indices.cached(py)
    }

    #[setter("qregs")]
    fn set_qregs(&mut self, other: Vec<QuantumRegister>) -> PyResult<()> {
        self.inner.qregs.dispose();
        for register in other {
            self.inner.add_qreg(register, true)?;
        }

        for (index, qubit) in self.inner.qubits.objects().iter().enumerate() {
            if !self.inner.qubit_indices.contains_key(qubit) {
                self.inner
                    .qubit_indices
                    .insert(qubit.clone(), BitLocations::new(index as u32, []));
            }
        }
        Ok(())
    }

    /// Returns the current sequence of registered :class:`.Qubit` instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`CircuitData` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Qubit`): The current sequence of registered qubits.
    #[getter("qubits")]
    pub fn py_qubits(&self, py: Python<'_>) -> Py<PyList> {
        self.inner.qubits.cached(py).clone_ref(py)
    }

    /// Return the number of qubits. This is equivalent to the length of the list returned by
    /// :meth:`.CircuitData.qubits`
    ///
    /// Returns:
    ///     int: The number of qubits.
    #[getter]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// The list of registered :class:`.ClassicalRegisters` instances.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate/corrupt :attr:`.data` for this circuit.
    ///
    /// Returns:
    ///     list[:class:`.ClassicalRegister`]: The current sequence of registered qubits.
    #[getter("cregs")]
    pub fn py_cregs<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyList> {
        self.inner.cregs.cached_list(py)
    }

    /// A dict mapping Clbit instances to tuple comprised of 0) the corresponding index in
    /// circuit.clbits and 1) a list of Register-int pairs for each Register containing the Bit and
    /// its index within that register.
    #[getter("_clbit_indices")]
    pub fn get_clbit_indices(&self, py: Python) -> &Py<PyDict> {
        self.inner.clbit_indices.cached(py)
    }

    #[setter("cregs")]
    fn set_cregs(&mut self, other: Vec<ClassicalRegister>) -> PyResult<()> {
        self.inner.cregs.dispose();
        for register in other {
            self.inner.add_creg(register, true)?;
        }

        for (index, clbit) in self.inner.clbits.objects().iter().enumerate() {
            if !self.inner.clbit_indices.contains_key(clbit) {
                self.inner
                    .clbit_indices
                    .insert(clbit.clone(), BitLocations::new(index as u32, []));
            }
        }
        Ok(())
    }

    /// Returns the current sequence of registered :class:`.Clbit`
    /// instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`CircuitData` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Clbit`): The current sequence of registered clbits.
    #[getter("clbits")]
    pub fn py_clbits(&self, py: Python<'_>) -> Py<PyList> {
        self.inner.clbits.cached(py).clone_ref(py)
    }

    /// Return the number of clbits. This is equivalent to the length of the list returned by
    /// :meth:`.CircuitData.clbits`.
    ///
    /// Returns:
    ///     int: The number of clbits.
    #[getter]
    pub fn num_clbits(&self) -> usize {
        self.inner.num_clbits()
    }

    /// Get a (cached) sorted list of the Python-space `Parameter` instances tracked by this circuit
    /// data's parameter table.
    #[getter]
    pub fn get_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.inner.parameters().iter().cloned())
    }

    pub fn unsorted_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, self.inner.param_table.iter_symbols().cloned())
    }

    fn _raw_parameter_table_entry(&self, param: Bound<PyAny>) -> PyResult<Py<PySet>> {
        self.inner.param_table._py_raw_entry(param)
    }

    pub fn get_parameter_by_name(&self, py: Python, name: &str) -> Option<Py<PyAny>> {
        self.inner
            .param_table
            .parameter_by_name(name)
            .map(|ob| ob.clone().into_py_any(py).unwrap())
    }

    /// Put ``self`` into the canonical physical form, with the given number of qubits.
    ///
    /// This acts in place, and does not need to traverse the circuit.  It is intended for use when
    /// the circuit is known to already represent a physical circuit, and we just need to assert
    /// that it is canonical physical form.
    ///
    /// This erases any information about virtual qubits in the :class:`CircuitData`.  Effectively,
    /// this applies the "trivial" layout mapping virtual qubit 0 to physical qubit 0, and so on.
    ///
    /// Args:
    ///     num_qubits: if given, the total number of physical qubits in the output; it must be at
    ///         least as large as the number of qubits in the circuit.  If not given, the number of
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
                num_qubits
            }
            None => self
                .num_qubits()
                .try_into()
                .expect("qubits are stored in 32-bit integers"),
        };
        self.inner.make_physical(num_qubits);
        Ok(())
    }

    /// Registers a :class:`.Qubit` instance.
    ///
    /// Args:
    ///     bit (:class:`.Qubit`): The qubit to register.
    ///     strict (bool): When set, raises an error if ``bit`` is already present.
    ///
    /// Raises:
    ///     ValueError: The specified ``bit`` is already present and flag ``strict``
    ///         was provided.
    #[pyo3(name="add_qubit", signature = (bit, *, strict=true))]
    pub fn py_add_qubit(&mut self, bit: ShareableQubit, strict: bool) -> PyResult<u32> {
        Ok(self.inner.add_qubit(bit, strict)?.0)
    }

    /// Registers a :class:`.QuantumRegister` instance.
    ///
    /// Args:
    ///     bit (:class:`.QuantumRegister`): The register to add.
    #[pyo3(signature = (register, *,strict = true))]
    pub fn add_qreg(&mut self, register: QuantumRegister, strict: bool) -> PyResult<()> {
        Ok(self.inner.add_qreg(register, strict)?)
    }

    /// Registers a :class:`.ClassicalRegister` instance.
    ///
    /// Args:
    ///     bit (:class:`.ClassicalRegister`): The register to add.
    #[pyo3(signature = (register, *,strict = true))]
    pub fn add_creg(&mut self, register: ClassicalRegister, strict: bool) -> PyResult<()> {
        Ok(self.inner.add_creg(register, strict)?)
    }

    /// Registers a :class:`.Clbit` instance.
    ///
    /// Args:
    ///     bit (:class:`.Clbit`): The clbit to register.
    ///     strict (bool): When set, raises an error if ``bit`` is already present.
    ///
    /// Raises:
    ///     ValueError: The specified ``bit`` is already present and flag ``strict``
    ///         was provided.
    #[pyo3(name="add_clbit", signature = (bit, *, strict=true))]
    pub fn py_add_clbit(&mut self, bit: ShareableClbit, strict: bool) -> PyResult<()> {
        self.inner.add_clbit(bit, strict)?;
        Ok(())
    }

    /// Performs a shallow copy.
    ///
    /// Returns:
    ///     PyCircuitData: The shallow copy.
    #[pyo3(signature = (copy_instructions=true, deepcopy=false))]
    pub fn copy(&self, py: Python<'_>, copy_instructions: bool, deepcopy: bool) -> PyResult<Self> {
        let mut res = self.copy_empty_like(VarsMode::Alike, BlocksMode::Keep)?;
        res.qargs_interner = self.qargs_interner.clone();
        res.cargs_interner = self.cargs_interner.clone();
        res.reserve(self.data().len());
        res.param_table.clone_from(&self.param_table);

        if deepcopy {
            let memo = PyDict::new(py);
            for inst in &self.data {
                let new_op = match inst.op.view() {
                    OperationRef::PyCustom(inst) => inst.py_deepcopy(py, Some(&memo))?.into(),
                    OperationRef::ControlFlow(cf) => cf.clone().into(),
                    OperationRef::StandardGate(gate) => gate.into(),
                    OperationRef::StandardInstruction(instruction) => instruction.into(),
                    OperationRef::Unitary(unitary) => unitary.clone().into(),
                    OperationRef::PauliProductMeasurement(ppm) => {
                        PauliBased::PauliProductMeasurement(ppm.clone()).into()
                    }
                    OperationRef::PauliProductRotation(ppr) => {
                        PauliBased::PauliProductRotation(ppr.clone()).into()
                    }
                    OperationRef::CustomOperation(custom_operation) => {
                        BoxedCustomOperation::from(custom_operation.clone_dyn()).into()
                    }
                };
                res.data.push(PackedInstruction {
                    op: new_op,
                    qubits: inst.qubits,
                    clbits: inst.clbits,
                    params: inst.params.clone(),
                    label: inst.label.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                });
            }
        } else if copy_instructions {
            for inst in &self.data {
                let new_op = match inst.op.view() {
                    OperationRef::PyCustom(inst) => inst.py_copy(py)?.into(),
                    OperationRef::ControlFlow(cf) => cf.clone().into(),
                    OperationRef::StandardGate(gate) => gate.into(),
                    OperationRef::StandardInstruction(instruction) => instruction.into(),
                    OperationRef::Unitary(unitary) => unitary.clone().into(),
                    OperationRef::PauliProductMeasurement(ppm) => {
                        PauliBased::PauliProductMeasurement(ppm.clone()).into()
                    }
                    OperationRef::PauliProductRotation(ppr) => {
                        PauliBased::PauliProductRotation(ppr.clone()).into()
                    }
                    OperationRef::CustomOperation(custom_operation) => {
                        BoxedCustomOperation::from(custom_operation.clone_dyn()).into()
                    }
                };
                res.data.push(PackedInstruction {
                    op: new_op,
                    qubits: inst.qubits,
                    clbits: inst.clbits,
                    params: inst.params.clone(),
                    label: inst.label.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                });
            }
        } else {
            res.data.extend(self.data.iter().cloned());
        }
        Ok(res.into())
    }

    /// Performs a copy with no instructions.
    ///
    /// # Arguments:
    ///
    /// * vars_mode: specifies realtime variables copy mode.
    ///     * VarsMode::Alike: variables will be copied following declaration semantics in self.
    ///     * VarsMode::Captures: variables will be copied as captured variables.
    ///     * VarsMode::Drop: variables will not be copied.
    ///
    /// # Returns:
    ///
    /// CircuitData: The empty copy like self.
    #[pyo3(name = "copy_empty_like", signature = (*, vars_mode=VarsMode::Alike))]
    pub fn py_copy_empty_like(&self, vars_mode: VarsMode) -> PyResult<Self> {
        Ok(self
            .inner
            .copy_empty_like(vars_mode, BlocksMode::Drop)?
            .into())
    }

    /// Reserves capacity for at least ``additional`` more
    /// :class:`.CircuitInstruction` instances to be added to this container.
    ///
    /// Args:
    ///     additional (int): The additional capacity to reserve. If the
    ///         capacity is already sufficient, does nothing.
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    #[pyo3(signature = (index=None))]
    pub fn pop(&mut self, py: Python<'_>, index: Option<PySequenceIndex>) -> PyResult<Py<PyAny>> {
        let index = index.unwrap_or(PySequenceIndex::Int(-1));
        let native_index = index.with_len(self.inner.data.len())?;
        let item = self.__getitem__(py, index)?;
        self.inner.delitem(native_index)?;
        Ok(item)
    }

    /// Returns a tuple of the sets of :class:`.Qubit` and :class:`.Clbit` instances
    /// that appear in at least one instruction's bit lists.
    ///
    /// Returns:
    ///     tuple[set[:class:`.Qubit`], set[:class:`.Clbit`]]: The active qubits and clbits.
    pub fn active_bits(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let qubits = PySet::empty(py)?;
        let clbits = PySet::empty(py)?;
        for inst in self.inner.data.iter() {
            for b in self.inner.qargs_interner.get(inst.qubits) {
                qubits.add(self.inner.qubits.get(*b).unwrap())?;
            }
            for b in self.inner.cargs_interner.get(inst.clbits) {
                clbits.add(self.inner.clbits.get(*b).unwrap())?;
            }
        }

        Ok((qubits, clbits).into_pyobject(py)?.unbind())
    }

    /// Checks whether the circuit has an instance of :class:`.ControlFlowOp`
    /// present amongst its operations.
    pub fn has_control_flow_op(&self) -> bool {
        self.inner.has_control_flow_op()
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    // Note: we also rely on this to make us iterable!
    pub fn __getitem__(&self, py: Python, index: PySequenceIndex) -> PyResult<Py<PyAny>> {
        // Get a single item, assuming the index is validated as in bounds.
        let get_single = |index: usize| {
            let inst = &self.inner.data[index];
            let qubits = self.inner.qargs_interner.get(inst.qubits);
            let clbits = self.inner.cargs_interner.get(inst.clbits);
            let params = self
                .inner
                .unpack_blocks_to_circuit_parameters(inst.params.as_deref());
            CircuitInstruction {
                operation: inst.op.clone(),
                qubits: PyTuple::new(py, self.inner.qubits.map_indices(qubits))
                    .unwrap()
                    .unbind(),
                clbits: PyTuple::new(py, self.inner.clbits.map_indices(clbits))
                    .unwrap()
                    .unbind(),
                params,
                label: inst.label.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: inst.py_op.clone(),
            }
            .into_py_any(py)
            .unwrap()
        };
        match index.with_len(self.data().len())? {
            SequenceIndex::Int(index) => Ok(get_single(index)),
            indices => PyList::new(py, indices.iter().map(get_single))?.into_py_any(py),
        }
    }

    pub fn __delitem__(&mut self, index: PySequenceIndex) -> PyResult<()> {
        Ok(self.inner.delitem(index.with_len(self.inner.data.len())?)?)
    }

    pub fn __setitem__(&mut self, index: PySequenceIndex, value: &Bound<PyAny>) -> PyResult<()> {
        fn set_single(slf: &mut PyCircuitData, index: usize, value: &Bound<PyAny>) -> PyResult<()> {
            let py = value.py();
            slf.inner.untrack_instruction_parameters(index)?;
            slf.inner.untrack_instruction_blocks(index);
            slf.inner.data[index] = slf.pack(py, &value.cast::<CircuitInstruction>()?.borrow())?;
            slf.inner.track_instruction_blocks(index);
            slf.inner.track_instruction_parameters(index)?;
            Ok(())
        }

        match index.with_len(self.inner.data.len())? {
            SequenceIndex::Int(index) => set_single(self, index, value),
            indices @ SequenceIndex::PosRange {
                start,
                stop,
                step: 1,
            } => {
                // `list` allows setting a slice with step +1 to an arbitrary length.
                let values = value.try_iter()?.collect::<PyResult<Vec<_>>>()?;
                for (index, value) in indices.iter().zip(values.iter()) {
                    set_single(self, index, value)?;
                }
                if indices.len() > values.len() {
                    self.inner.delitem(SequenceIndex::PosRange {
                        start: start + values.len(),
                        stop,
                        step: 1,
                    })?
                } else {
                    for value in values[indices.len()..].iter().rev() {
                        self.insert(stop as isize, value.cast()?.borrow())?;
                    }
                }
                Ok(())
            }
            indices => {
                let values = value.try_iter()?.collect::<PyResult<Vec<_>>>()?;
                if indices.len() == values.len() {
                    for (index, value) in indices.iter().zip(values.iter()) {
                        set_single(self, index, value)?;
                    }
                    Ok(())
                } else {
                    Err(PyValueError::new_err(format!(
                        "attempt to assign sequence of size {:?} to extended slice of size {:?}",
                        values.len(),
                        indices.len(),
                    )))
                }
            }
        }
    }

    pub fn insert(&mut self, index: isize, value: PyRef<CircuitInstruction>) -> PyResult<()> {
        let py = value.py();
        let packed = self.pack(py, &value)?;
        Ok(self.inner.insert(index, packed)?)
    }

    /// Primary entry point for appending an instruction from Python space.
    pub fn append(&mut self, value: &Bound<CircuitInstruction>) -> PyResult<()> {
        let packed = self.pack(value.py(), &value.borrow())?;
        Ok(self.inner.push(packed)?)
    }

    /// Backup entry point for appending an instruction from Python space, in the unusual case that
    /// one of the instruction parameters contains a cyclical reference to the circuit itself.
    ///
    /// In this case, the `params` field should be a list of `(index, parameters)` tuples, where the
    /// index is into the instruction's `params` attribute, and `parameters` is a Python iterable
    /// of `Parameter` objects.
    pub fn append_manual_params(
        &mut self,
        value: &Bound<CircuitInstruction>,
        params: &Bound<PyList>,
    ) -> PyResult<()> {
        let instruction_index = self.len();
        let packed = self.pack(value.py(), &value.borrow())?;
        self.inner.data.push(packed);
        for item in params.iter() {
            let (parameter_index, parameters) = item.extract::<(u32, Bound<PyAny>)>()?;
            let usage = ParameterUse::Index {
                instruction: instruction_index,
                parameter: parameter_index,
            };
            for param in parameters.try_iter()? {
                let symbol = param?.extract::<Symbol>()?;
                self.inner.param_table.track(&symbol, Some(usage))?;
            }
        }
        Ok(())
    }

    /// Copy `CircuitInstruction` instances from a Python iterator into this object.
    ///
    /// This method (with this magic name) forms part of the Python "sequence" API.
    pub fn extend(&mut self, other: &Bound<PyAny>) -> PyResult<()> {
        for inst in other.try_iter()? {
            self.append(inst?.cast()?)?;
        }
        Ok(())
    }
    /// Copy instructions from one `PyCircuitData`  onto this one, optionally remapping the qubits and
    /// clbits by index.
    ///
    /// This assumes that all the variables and registers used in `other` have been set up
    /// correctly, but will copy across control-flow blocks as required.
    #[pyo3(signature = (other, *, qubits=None, clbits=None))]
    pub fn native_extend(
        &mut self,
        other: &PyCircuitData,
        qubits: Option<Vec<Qubit>>,
        clbits: Option<Vec<Clbit>>,
    ) -> PyResult<()> {
        if qubits
            .as_ref()
            .is_some_and(|q| q.len() != other.num_qubits())
        {
            return Err(PyValueError::new_err(
                "'qubits' argument is the wrong length",
            ));
        } else if other.num_qubits() > self.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "too many input qubits ({}) for circuit ({})",
                other.num_qubits(),
                self.num_qubits(),
            )));
        }
        if clbits
            .as_ref()
            .is_some_and(|q| q.len() != other.num_clbits())
        {
            return Err(PyValueError::new_err(
                "'clbits' argument is the wrong length",
            ));
        } else if other.num_clbits() > self.num_clbits() {
            return Err(PyValueError::new_err(format!(
                "too many input clbits ({}) for circuit ({})",
                other.num_clbits(),
                self.num_clbits(),
            )));
        }
        let qargs_map = self
            .inner
            .qargs_interner
            .merge_map_slice(&other.qargs_interner, |q| {
                Some(qubits.as_ref().map_or(*q, |qubits| qubits[q.index()]))
            });
        let cargs_map = self
            .inner
            .cargs_interner
            .merge_map_slice(&other.cargs_interner, |c| {
                Some(clbits.as_ref().map_or(*c, |clbits| clbits[c.index()]))
            });
        let block_map = other
            .blocks
            .items()
            .map(|(bid, block)| (bid, self.inner.add_block(block.clone())))
            .collect::<HashMap<_, _>>();
        self.inner.data.reserve(other.data.len());
        for inst in other.iter() {
            self.inner.push(PackedInstruction {
                qubits: qargs_map[inst.qubits],
                clbits: cargs_map[inst.clbits],
                params: inst
                    .params
                    .as_ref()
                    .map(|params| Box::new(params.map_blocks_ref(|bid| block_map[bid]))),
                ..inst.clone()
            })?;
        }
        Ok(())
    }

    /// Assign all the circuit parameters, given an iterable input of `Param` instances.
    fn assign_parameters_iterable(&mut self, sequence: Bound<PyAny>) -> PyResult<()> {
        if let Ok(readonly) = sequence.extract::<PyReadonlyArray1<f64>>() {
            // Fast path for Numpy arrays; in this case we can easily handle them without copying
            // the data across into a Rust-space `Vec` first.
            let array = readonly.as_array();
            Ok(self.inner.assign_parameters_from_array(array)?)
        } else {
            let values = sequence
                .try_iter()?
                .map(|ob| Param::extract_no_coerce(ob?.as_borrowed()))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(self.inner.assign_parameters_from_slice(&values)?)
        }
    }

    /// Assign all uses of the circuit parameters as keys `mapping` to their corresponding values.
    ///
    /// Any items in the mapping that are not present in the circuit are skipped; it's up to Python
    /// space to turn extra bindings into an error, if they choose to do it.
    fn assign_parameters_mapping(&mut self, mapping: Bound<PyAny>) -> PyResult<()> {
        let mut items = Vec::new();
        for item in mapping.call_method0("items")?.try_iter()? {
            let (symbol, value) = item?.extract::<(Symbol, AssignParam)>()?;
            let uuid = ParameterUuid::from_symbol(&symbol);
            // It's fine if the mapping contains parameters that we don't have - just skip those.
            if let Ok(uses) = self.inner.param_table.pop(uuid) {
                items.push((symbol, value.0, uses));
            }
        }
        Ok(self.inner.assign_parameters_inner(items)?)
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Counts the number of times each operation is used in the circuit.
    ///
    /// # Parameters
    /// - `self` - A mutable reference to the CircuitData struct.
    ///
    /// # Returns
    /// An IndexMap containing the operation names as keys and their respective counts as values.
    pub fn count_ops(&self) -> IndexMap<&str, usize> {
        self.inner.count_ops()
    }

    // Marks this pyclass as NOT hashable.
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __eq__(slf: &Bound<Self>, other: &Bound<PyAny>) -> PyResult<bool> {
        let self_cd = slf.borrow();
        let slf = slf.as_any();
        if slf.is(other) {
            return Ok(true);
        }
        if slf.len()? != other.len()? {
            return Ok(false);
        }

        if let Ok(other_cd) = other.cast::<PyCircuitData>() {
            if !slf
                .getattr("global_phase")?
                .eq(other_cd.getattr("global_phase")?)?
            {
                return Ok(false);
            }

            if self_cd.vars_stretches != other_cd.borrow().vars_stretches {
                return Ok(false);
            }
        }

        // Implemented using generic iterators on both sides
        // for simplicity.
        let mut ours_itr = slf.try_iter()?;
        let mut theirs_itr = other.try_iter()?;
        loop {
            match (ours_itr.next(), theirs_itr.next()) {
                (Some(ours), Some(theirs)) => {
                    if !ours?.eq(theirs?)? {
                        return Ok(false);
                    }
                }
                (None, None) => {
                    return Ok(true);
                }
                _ => {
                    return Ok(false);
                }
            }
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        // Note:
        //   There's no need to visit the native Rust data
        //   structures used for internal tracking: the only Python
        //   references they contain are to the bits in these lists!
        if let Some(bits) = self.inner.qubits.cached_raw() {
            visit.call(bits)?;
        }
        if let Some(bits) = self.inner.clbits.cached_raw() {
            visit.call(bits)?;
        }
        if let Some(regs) = self.inner.qregs.cached_raw() {
            visit.call(regs)?;
        }
        if let Some(regs) = self.inner.cregs.cached_raw() {
            visit.call(regs)?;
        }
        if let Some(locations) = self.inner.qubit_indices.cached_raw() {
            visit.call(locations)?;
        }
        if let Some(locations) = self.inner.clbit_indices.cached_raw() {
            visit.call(locations)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.inner.clear_all();
    }

    /// Set the global phase of the circuit.
    ///
    /// This method assumes that the parameter table is either fully consistent, or contains zero
    /// entries for the global phase, regardless of what value is currently stored there.  It's not
    /// uncommon for subclasses and other parts of Qiskit to have filled in the global phase field
    /// by copies or other means, before making the parameter table consistent.
    #[setter(global_phase)]
    pub fn set_global_phase_param(&mut self, angle: Param) -> Result<(), CircuitDataError> {
        self.inner.set_global_phase_param(angle)
    }

    pub fn num_nonlocal_gates(&self) -> usize {
        self.inner.num_nonlocal_gates()
    }

    /// Converts several qubit representations (such as indexes, range, etc.)
    /// into a list of qubits.
    ///
    /// Args:
    ///     qubit_representation: Representation to expand.
    ///
    /// Returns:
    ///     The resolved instances of the qubits.
    fn _qbit_argument_conversion(
        &self,
        qubit_representation: Bound<PyAny>,
    ) -> PyResult<Vec<ShareableQubit>> {
        bit_argument_conversion(
            &qubit_representation,
            self.inner.qubits.objects(),
            &self.inner.qubit_indices,
        )
    }

    /// Converts several clbit representations (such as indexes, range, etc.)
    /// into a list of qubits.
    ///
    /// Args:
    ///     clbit_representation: Representation to expand.
    ///
    /// Returns:
    ///     The resolved instances of the qubits.
    fn _cbit_argument_conversion(
        &self,
        clbit_representation: Bound<PyAny>,
    ) -> PyResult<Vec<ShareableClbit>> {
        bit_argument_conversion(
            &clbit_representation,
            self.inner.clbits.objects(),
            &self.inner.clbit_indices,
        )
    }

    /// Add an input variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    #[pyo3(name = "add_input_var")]
    fn py_add_input_var(&mut self, var: expr::Var) -> PyResult<()> {
        self.inner.add_var(var, VarType::Input)?;
        Ok(())
    }

    /// Add a captured variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    #[pyo3(name = "add_captured_var")]
    fn py_add_captured_var(&mut self, var: expr::Var) -> PyResult<()> {
        self.inner.add_var(var, VarType::Capture)?;
        Ok(())
    }

    /// Add a local variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    #[pyo3(name = "add_declared_var")]
    fn py_add_declared_var(&mut self, var: expr::Var) -> PyResult<()> {
        self.inner.add_var(var, VarType::Declare)?;
        Ok(())
    }

    /// Check if this realtime variable is in the circuit.
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

    /// Check if the circuit contains an input variable with the given name.
    #[pyo3(name = "has_input_var")]
    fn py_has_input_var(&self, name: &str) -> PyResult<bool> {
        Ok(self.vars_stretches.has_var_by_type(name, VarType::Input))
    }

    /// Check if the circuit contains a local variable with the given name.
    #[pyo3(name = "has_declared_var")]
    fn py_has_declared_var(&self, name: &str) -> PyResult<bool> {
        Ok(self.vars_stretches.has_var_by_type(name, VarType::Declare))
    }

    /// Check if the circuit contains a capture variable with the given name.
    #[pyo3(name = "has_captured_var")]
    fn py_has_captured_var(&self, name: &str) -> PyResult<bool> {
        Ok(self.vars_stretches.has_var_by_type(name, VarType::Capture))
    }

    /// Return a list of the captured variables tracked in this circuit.
    #[pyo3(name = "get_captured_vars")]
    fn py_get_captured_vars(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new(
            py,
            self.vars_stretches
                .iter_vars(VarType::Capture)
                .map(|var| var.clone().into_pyobject(py).unwrap()),
        )?
        .unbind())
    }

    /// Return a list of the local variables tracked in this circuit.
    #[pyo3(name = "get_declared_vars")]
    fn py_get_declared_vars(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new(
            py,
            self.vars_stretches
                .iter_vars(VarType::Declare)
                .map(|var| var.clone().into_pyobject(py).unwrap()),
        )?
        .unbind())
    }

    // Return the variable in the circuit corresponding to the given name, or None if no such variable.
    #[pyo3(name = "get_var")]
    fn py_get_var(&self, py: Python, name: &str) -> PyResult<Py<PyAny>> {
        if let Some(var) = self.vars_stretches.get_var(name) {
            var.clone().into_py_any(py)
        } else {
            Ok(py.None())
        }
    }

    #[pyo3(name = "get_input_vars")]
    fn py_get_input_vars(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new(
            py,
            self.vars_stretches
                .iter_vars(VarType::Input)
                .map(|var| var.clone().into_pyobject(py).unwrap()),
        )?
        .unbind())
    }

    /// Return the number of classical input variables in the circuit.
    #[getter]
    pub fn num_input_vars(&self) -> usize {
        self.vars_stretches.num_vars(VarType::Input)
    }

    /// Return the number of captured variables in the circuit.
    #[getter]
    pub fn num_captured_vars(&self) -> usize {
        self.vars_stretches.num_vars(VarType::Capture)
    }

    /// Return the number of local variables in the circuit.
    #[getter]
    pub fn num_declared_vars(&self) -> usize {
        self.vars_stretches.num_vars(VarType::Declare)
    }

    /// Add a captured stretch to the circuit.
    ///
    /// Args:
    ///     stretch: the stretch variable to add.
    #[pyo3(name = "add_captured_stretch")]
    fn py_add_captured_stretch(&mut self, stretch: expr::Stretch) -> PyResult<()> {
        self.inner.add_stretch(stretch, StretchType::Capture)?;
        Ok(())
    }

    /// Add a local stretch to the circuit.
    ///
    /// Args:
    ///     stretch: the stretch variable to add.
    #[pyo3(name = "add_declared_stretch")]
    fn py_add_declared_stretch(&mut self, stretch: expr::Stretch) -> PyResult<()> {
        self.inner.add_stretch(stretch, StretchType::Declare)?;
        Ok(())
    }

    /// Check if this stretch variable is in the circuit.
    ///
    /// Args:
    ///     stretch: the stretch or name to check.
    #[pyo3(name = "has_stretch")]
    fn py_has_stretch(&self, stretch: &Bound<PyAny>) -> PyResult<bool> {
        if let Ok(name) = stretch.extract::<String>() {
            Ok(self.vars_stretches.has_stretch(&name))
        } else {
            let stretch = stretch.extract::<expr::Stretch>()?;
            Ok(self.vars_stretches.stretches().contains(&stretch))
        }
    }

    /// Check if the circuit contains a captured stretch with the specified name.
    #[pyo3(name = "has_captured_stretch")]
    fn py_has_captured_stretch(&self, name: &str) -> PyResult<bool> {
        Ok(self
            .vars_stretches
            .has_stretch_by_type(name, StretchType::Capture))
    }

    /// Check if the circuit contains a local stretch with the specified name.
    #[pyo3(name = "has_declared_stretch")]
    fn py_has_declared_stretch(&self, name: &str) -> PyResult<bool> {
        Ok(self
            .vars_stretches
            .has_stretch_by_type(name, StretchType::Declare))
    }

    // Return the stretch in the circuit corresponding to the specified name, or None if no such variable.
    #[pyo3(name = "get_stretch")]
    fn py_get_stretch(&self, py: Python, name: &str) -> PyResult<Py<PyAny>> {
        if let Some(stretch) = self.vars_stretches.get_stretch(name) {
            stretch.clone().into_py_any(py)
        } else {
            Ok(py.None())
        }
    }

    /// Return a list of the captured stretch variables tracked in this circuit.
    #[pyo3(name = "get_captured_stretches")]
    fn py_get_captured_stretches(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new(
            py,
            self.vars_stretches
                .iter_stretches(StretchType::Capture)
                .map(|stretch| stretch.clone().into_pyobject(py).unwrap()),
        )?
        .unbind())
    }

    /// Return a list of the local stretch variables tracked in this circuit.
    #[pyo3(name = "get_declared_stretches")]
    fn py_get_declared_stretches(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new(
            py,
            self.vars_stretches
                .iter_stretches(StretchType::Declare)
                .map(|stretch| stretch.clone().into_pyobject(py).unwrap()),
        )?
        .unbind())
    }

    /// Return the number of local stretch variables in the circuit.
    #[getter]
    pub fn num_declared_stretches(&self) -> usize {
        self.vars_stretches.num_stretches(StretchType::Declare)
    }

    /// Return the number of captured stretch variables in the circuit.
    #[getter]
    pub fn num_captured_stretches(&self) -> usize {
        self.vars_stretches.num_stretches(StretchType::Capture)
    }

    #[getter]
    pub fn qregs(&self) -> &[QuantumRegister] {
        self.inner.qregs()
    }

    #[getter]
    pub fn cregs(&self) -> &[ClassicalRegister] {
        self.inner.cregs()
    }

    /// Return the number of unbound compile-time symbolic parameters tracked by the circuit.
    pub fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    /// Return the width of the circuit. This is the number of qubits plus the
    /// number of clbits.
    ///
    /// Returns:
    ///     int: The width of the circuit.
    pub fn width(&self) -> usize {
        self.inner.width()
    }
}

impl PyCircuitData {
    pub fn pack(&mut self, py: Python, inst: &CircuitInstruction) -> PyResult<PackedInstruction> {
        let qubits = self.inner.qargs_interner.insert_owned(
            self.qubits
                .map_objects(inst.qubits.extract::<Vec<ShareableQubit>>(py)?)?
                .collect(),
        );
        let clbits = self.inner.cargs_interner.insert_owned(
            self.clbits
                .map_objects(inst.clbits.extract::<Vec<ShareableClbit>>(py)?)?
                .collect(),
        );
        let params = self
            .inner
            .extract_blocks_from_circuit_parameters(inst.params.as_ref());
        Ok(PackedInstruction {
            op: inst.operation.clone(),
            qubits,
            clbits,
            params,
            label: inst.label.clone(),
            #[cfg(feature = "cache_pygates")]
            py_op: inst.py_op.clone(),
        })
    }

    /// Build a reference to the Python-space operation object (the `Gate`, etc) packed into an
    /// instruction.  This may construct the reference if the `Instruction` is a standard
    /// gate or instruction with no already stored operation.
    ///
    /// A standard-gate or standard-instruction operation object returned by this function is
    /// disconnected from the circuit; updates to its parameters, label, duration, unit
    /// and condition will not be propagated back.
    ///
    /// The provided `instr` MUST belong to this circuit.
    pub fn unpack_py_op(&self, py: Python, instr: &PackedInstruction) -> PyResult<Py<PyAny>> {
        // `OnceLock::get_or_init` and the non-stabilised `get_or_try_init`, which would otherwise
        // be nice here are both non-reentrant.  This is a problem if the init yields control to the
        // Python interpreter as this one does, since that can allow CPython to freeze the thread
        // and for another to attempt the initialisation.
        #[cfg(feature = "cache_pygates")]
        {
            if let Some(ob) = instr.py_op.get() {
                return Ok(ob.clone_ref(py));
            }
        }
        let params = self
            .inner
            .unpack_blocks_to_circuit_parameters(instr.params.as_deref());
        let out = instruction::create_py_op(
            py,
            instr.op.view(),
            params,
            instr.label.as_deref().map(String::as_str),
        )?;
        #[cfg(feature = "cache_pygates")]
        // The unpacking operation can cause a thread pause and concurrency, since it can call
        // interpreted Python code for a standard gate, so we need to take care that some other
        // Python thread might have populated the cache before we do.
        let _ = instr.py_op.set(out.clone_ref(py));
        Ok(out)
    }

    /// Move this [PyCircuitData] into a complete Python `QuantumCircuit` object.
    pub fn into_py_quantum_circuit(self, py: Python) -> PyResult<Bound<PyAny>> {
        // TODO: setting the name to a fixed value is a gross hack for Qiskit 2.3 to prevent
        // QuantumCircuit's awkward "unique name" logic from coming in to play, which would cause
        // problems with reproducibility in QPY streams when gate-caching is turned off.
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "name"), "unnamed")?;
        QUANTUM_CIRCUIT.get_bound(py).call_method(
            intern!(py, "_from_circuit_data"),
            (self,),
            Some(&kwargs),
        )
    }

    /// Returns an immutable view of the Qubits registered in the circuit
    pub fn qubits(&self) -> &ObjectRegistry<Qubit, ShareableQubit> {
        self.inner.qubits()
    }

    /// Returns an immutable view of the Classical bits registered in the circuit
    pub fn clbits(&self) -> &ObjectRegistry<Clbit, ShareableClbit> {
        self.inner.clbits()
    }

    pub fn qubit_indices(&self) -> &BitLocator<ShareableQubit, QuantumRegister> {
        self.inner.qubit_indices()
    }

    pub fn clbit_indices(&self) -> &BitLocator<ShareableClbit, ClassicalRegister> {
        self.inner.clbit_indices()
    }

    pub fn data(&self) -> &[PackedInstruction] {
        self.inner.data()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn qargs_interner(&self) -> &Interner<[Qubit]> {
        self.inner.qargs_interner()
    }

    pub fn cargs_interner(&self) -> &Interner<[Clbit]> {
        self.inner.cargs_interner()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::converters::py_dag_to_circuit;
    use crate::dag_circuit::DAGCircuit;
    use crate::operations::{ArrayType, PauliProductMeasurement, UnitaryGate};
    use nalgebra::{Matrix2, Matrix4};

    #[test]
    fn packed_pointer_types_behave() -> PyResult<()> {
        // This is largely to exercise the packed-pointer logic under debug builds (since the
        // Python-space tests run with Rust in release mode) and Miri.

        use crate::operations::PauliProductRotation;
        let mut qc = CircuitData::from_packed_operations(4, 1, [], Param::Float(0.0))?;
        qc.push_packed_operation(
            Box::new(PauliBased::PauliProductRotation(PauliProductRotation {
                z: vec![false, false, true, true],
                x: vec![false, true, true, false],
                angle: Param::Float(1.0),
            }))
            .into(),
            None,
            &[Qubit(0), Qubit(1), Qubit(2), Qubit(3)],
            &[],
        )?;
        qc.push_packed_operation(
            Box::new(PauliBased::PauliProductMeasurement(
                PauliProductMeasurement {
                    z: vec![true, true, true],
                    x: vec![false, false, true],
                    neg: false,
                },
            ))
            .into(),
            None,
            &[Qubit(0), Qubit(1), Qubit(2)],
            &[Clbit(0)],
        )?;
        qc.push_packed_operation(
            Box::new(UnitaryGate {
                array: ArrayType::OneQ(Matrix2::identity()),
            })
            .into(),
            None,
            &[Qubit(2)],
            &[],
        )?;
        qc.push_packed_operation(
            Box::new(UnitaryGate {
                array: ArrayType::TwoQ(Matrix4::identity()),
            })
            .into(),
            None,
            &[Qubit(2), Qubit(3)],
            &[],
        )?;
        let check = |left: &CircuitData, right: &CircuitData| {
            for (a, b) in ::std::iter::zip(left.data(), right.data()) {
                match (a.op.view(), b.op.view()) {
                    (OperationRef::Unitary(a), OperationRef::Unitary(b)) => assert_eq!(a, b),
                    (
                        OperationRef::PauliProductMeasurement(a),
                        OperationRef::PauliProductMeasurement(b),
                    ) => assert_eq!(a, b),
                    (
                        OperationRef::PauliProductRotation(a),
                        OperationRef::PauliProductRotation(b),
                    ) => assert_eq!(a, b),
                    (a, b) => panic!("unexpected types in iterator:\n{a:?}\n{b:?}"),
                }
            }
        };
        let other = qc.clone();
        check(&qc, &other);
        let roundtrip = py_dag_to_circuit(
            &DAGCircuit::from_circuit_data(&qc, false, None, None, None, None)?,
            false,
        )?;
        check(&qc, &roundtrip);
        Ok(())
    }
}
