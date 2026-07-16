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

#![allow(clippy::too_many_arguments)]

mod bounds;
mod errors;
mod instruction_properties;
mod qargs;
mod qubit_properties;

pub use errors::TargetError;
use foldhash::fast::RandomState;
pub use instruction_properties::InstructionProperties;
use pyo3::types::IntoPyDict;
pub use qargs::{Qargs, QargsRef};
pub use qubit_properties::QubitProperties;

use std::{ops::Index, sync::OnceLock};

use hashbrown::HashMap;
use hashbrown::HashSet;
use pyo3::{
    IntoPyObjectExt,
    exceptions::{PyAttributeError, PyIndexError, PyKeyError, PyValueError},
    prelude::*,
    pyclass,
    types::{PyDict, PyList, PySet},
};
use qiskit_util::IndexMap;
use rustworkx_core::petgraph::prelude::*;
use smallvec::SmallVec;
use thiserror::Error;

use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::instruction::{Instruction, Parameters, create_py_op};
use qiskit_circuit::operations::{Operation, OperationRef, Param};
use qiskit_circuit::packed_instruction::PackedOperation;

use crate::TranspilerError;
use bounds::AngleBound;

/// Represents a Qiskit `Gate` object or a Variadic instruction.
/// Keeps a reference to its Python instance for caching purposes.
#[derive(FromPyObject, Debug, Clone, IntoPyObjectRef)]
pub enum TargetOperation {
    Normal(NormalOperation),
    Variadic(Py<PyAny>),
}

impl TargetOperation {
    /// Gets the number of qubits of a [TargetOperation], will panic if the operation is [TargetOperation::Variadic].
    pub fn num_qubits(&self) -> u32 {
        match &self {
            Self::Normal(normal) => normal.operation.num_qubits(),
            Self::Variadic(_) => {
                panic!("'num_qubits' property doesn't exist for Variadic operations")
            }
        }
    }

    /// Creates a [TargetOperation] from an instance of [PackedOperation]
    pub fn from_packed_operation(
        operation: PackedOperation,
        params: Option<Parameters<CircuitData>>,
    ) -> Self {
        NormalOperation::from_packed_operation(operation, params).into()
    }

    /// Checks if the operation is variadic.
    pub fn is_variadic(&self) -> bool {
        matches!(self, Self::Variadic(_))
    }
}

impl From<NormalOperation> for TargetOperation {
    fn from(value: NormalOperation) -> Self {
        TargetOperation::Normal(value)
    }
}

/// Represents a Qiskit `Gate` object, keeps a reference to its Python
/// instance for caching purposes.
#[derive(Debug)]
pub struct NormalOperation {
    pub operation: PackedOperation,
    pub params: Option<Parameters<CircuitData>>,
    op_object: OnceLock<PyResult<Py<PyAny>>>,
}

impl NormalOperation {
    /// Creates a of [TargetOperation] from an instance of [PackedOperation]
    pub fn from_packed_operation(
        operation: PackedOperation,
        params: Option<Parameters<CircuitData>>,
    ) -> Self {
        Self {
            operation,
            params,
            op_object: OnceLock::new(),
        }
    }
}

impl Instruction for NormalOperation {
    type Block = CircuitData;

    fn op(&self) -> OperationRef<'_> {
        self.operation.view()
    }

    fn parameters(&self) -> Option<&Parameters<CircuitData>> {
        self.params.as_ref()
    }

    fn label(&self) -> Option<&str> {
        None
    }
}

impl<'py> IntoPyObject<'py> for NormalOperation {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(mut self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let op = self.op_object.take();
        let params = self.params.take();
        op.unwrap_or_else(|| create_py_op(py, self.op(), params, None))
            .map(|o| o.into_bound(py))
    }
}

impl<'a, 'py> IntoPyObject<'py> for &'a NormalOperation {
    type Target = PyAny;
    type Output = Borrowed<'a, 'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self
            .op_object
            .get_or_init(|| create_py_op(py, self.op(), self.parameters().cloned(), None))
        {
            Ok(op) => Ok(op.bind_borrowed(py)),
            Err(err) => Err(err.clone_ref(py)),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for NormalOperation {
    type Error = <OperationFromPython<CircuitData> as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let operation: OperationFromPython<CircuitData> = ob.extract()?;
        Ok(Self {
            operation: operation.operation,
            params: operation.params,
            op_object: Ok(ob.to_owned().unbind()).into(),
        })
    }
}

// Custom impl for Clone to avoid cloning the `OnceLock`.
impl Clone for NormalOperation {
    fn clone(&self) -> Self {
        Self {
            operation: self.operation.clone(),
            params: self.params.clone(),
            op_object: OnceLock::new(),
        }
    }
}

/// A collection of all the properties of an instruction in the [Target]
#[derive(Debug, Clone)]
struct TargetProperties {
    /// Contains the mapping of qargs and properties for the instruction.
    pub properties: IndexMap<Qargs, Option<InstructionProperties>>,
    /// Contains the instruction's original instance.
    pub instruction: TargetOperation,
    /// Contains the specified angle constraints for a parametric instruction.
    pub angle_bounds: Option<AngleBound>,
}

/**
The base class for a Python ``Target`` object. Contains data representing the
constraints of a particular backend.

The intent of this struct is to contain data that can be representable and
accessible through both Rust and Python, so it can be used for rust-based
transpiler processes.

This structure contains duplicates of every element in the Python counterpart of
`gate_map`. Which improves access for Python while sacrificing a small amount of
memory.
 */
#[pyclass(
    mapping,
    subclass,
    name = "BaseTarget",
    module = "qiskit._accelerate.target",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct Target {
    #[pyo3(get, set)]
    pub description: Option<String>,
    #[pyo3(get)]
    pub num_qubits: Option<u32>,
    pub dt: Option<f64>,
    #[pyo3(get, set)]
    pub granularity: u32,
    #[pyo3(get, set)]
    pub min_length: u32,
    #[pyo3(get, set)]
    pub pulse_alignment: u32,
    #[pyo3(get, set)]
    pub acquire_alignment: u32,
    #[pyo3(get, set)]
    pub qubit_properties: Option<Vec<QubitProperties>>,
    #[pyo3(get, set)]
    pub concurrent_measurements: Option<Vec<Vec<PhysicalQubit>>>,
    gate_map: IndexMap<String, TargetProperties>,
    global_operations: HashMap<u32, HashSet<String>>,
    // This uses `IndexMap` not because it's necessary for determinism (though it will help), but so
    // it retains the specific iteration order it is constructed with.  The order `qargs` are
    // encountered during construction are _usually_ going to be quite structured, and structure
    // here means that graphs built from qargs (like the coupling graph) will have their edges
    // ordered in much cache- and branch-prediction-friendlier orders than if they are randomised.
    qarg_gate_map: IndexMap<Qargs, HashSet<String>>,
    has_angle_bounds: bool,
}

#[pymethods]
impl Target {
    /// Create a new ``Target`` object
    ///
    ///Args:
    ///    description (str): An optional string to describe the Target.
    ///    num_qubits (int): An optional int to specify the number of qubits
    ///        the backend target has. If not set it will be implicitly set
    ///        based on the qargs when :meth:`~qiskit.Target.add_instruction`
    ///        is called. Note this must be set if the backend target is for a
    ///        noiseless simulator that doesn't have constraints on the
    ///        instructions so the transpiler knows how many qubits are
    ///        available.
    ///    dt (float): The system time resolution of input signals in seconds
    ///    granularity (int): An integer value representing minimum pulse gate
    ///        resolution in units of ``dt``. A user-defined pulse gate should
    ///        have duration of a multiple of this granularity value.
    ///    min_length (int): An integer value representing minimum pulse gate
    ///        length in units of ``dt``. A user-defined pulse gate should be
    ///        longer than this length.
    ///    pulse_alignment (int): An integer value representing a time
    ///        resolution of gate instruction starting time. Gate instruction
    ///        should start at time which is a multiple of the alignment
    ///        value.
    ///    acquire_alignment (int): An integer value representing a time
    ///        resolution of measure instruction starting time. Measure
    ///        instruction should start at time which is a multiple of the
    ///        alignment value.
    ///    qubit_properties (list): A list of :class:`~.QubitProperties`
    ///        objects defining the characteristics of each qubit on the
    ///        target device. If specified the length of this list must match
    ///        the number of qubits in the target, where the index in the list
    ///        matches the qubit number the properties are defined for. If some
    ///        qubits don't have properties available you can set that entry to
    ///        ``None``
    ///    concurrent_measurements(list): A list of sets of qubits that must be
    ///        measured together. This must be provided
    ///        as a nested list like ``[[0, 1], [2, 3, 4]]``.
    ///Raises:
    ///    ValueError: If both ``num_qubits`` and ``qubit_properties`` are both
    ///        defined and the value of ``num_qubits`` differs from the length of
    ///        ``qubit_properties``.
    #[new]
    #[pyo3(signature = (
        description = None,
        num_qubits = 0,
        dt = None,
        granularity = 1,
        min_length = 1,
        pulse_alignment = 1,
        acquire_alignment = 1,
        qubit_properties = None,
        concurrent_measurements = None,
    ))]
    pub fn new(
        description: Option<String>,
        mut num_qubits: Option<u32>,
        dt: Option<f64>,
        granularity: Option<u32>,
        min_length: Option<u32>,
        pulse_alignment: Option<u32>,
        acquire_alignment: Option<u32>,
        qubit_properties: Option<Vec<QubitProperties>>,
        concurrent_measurements: Option<Vec<Vec<PhysicalQubit>>>,
    ) -> PyResult<Self> {
        if let Some(qubit_properties) = qubit_properties.as_ref() {
            if num_qubits.is_some_and(|num_qubits| num_qubits > 0) {
                if num_qubits.unwrap() as usize != qubit_properties.len() {
                    return Err(PyValueError::new_err(
                        "The value of num_qubits specified does not match the \
                            length of the input qubit_properties list",
                    ));
                }
            } else {
                num_qubits = Some(qubit_properties.len() as u32)
            }
        }
        Ok(Target {
            description,
            num_qubits,
            dt,
            granularity: granularity.unwrap_or(1),
            min_length: min_length.unwrap_or(1),
            pulse_alignment: pulse_alignment.unwrap_or(1),
            acquire_alignment: acquire_alignment.unwrap_or(1),
            qubit_properties,
            concurrent_measurements,
            gate_map: IndexMap::default(),
            global_operations: HashMap::default(),
            qarg_gate_map: IndexMap::default(),
            has_angle_bounds: false,
        })
    }

    /// Add a new instruction to the `Target` after it has been processed in python.
    ///
    /// Args:
    ///     instruction: An instance of `Instruction` or the class representing said instructionm
    ///         if representing a variadic.
    ///     properties: A mapping of qargs and ``InstructionProperties``.
    ///     name: A name assigned to the provided gate.
    ///     bound_list: The bounds on the parameters for a given gate. This is specified by a list
    ///         of tuples (low, high) which represent the low and high bound (inclusively) on what
    ///         float values are allowed for the parameter in that position. If a parameter
    ///         doesn't have an angle bound you can use ``None`` to represent that. For example if
    ///         a 3 parameter gate only had a bound on the second parameter you would represent
    ///         that with: ``[None, [0, 3.14], None]`` which means the first and third parameter
    ///         allow any value but the second parameter only accepts values between 0 and 3.14.
    /// Raises:
    ///     AttributeError: If gate is already in map
    ///     TranspilerError: If an operation class is passed in for ``instruction`` and no name
    ///         is specified or ``properties`` is set.
    #[pyo3(name="add_instruction", signature = (instruction, name, properties=None, *, angle_bounds=None))]
    fn py_add_instruction(
        &mut self,
        instruction: TargetOperation,
        name: String,
        properties: Option<IndexMap<Qargs, Option<InstructionProperties>>>,
        angle_bounds: Option<SmallVec<[Option<[f64; 2]>; 3]>>,
    ) -> PyResult<()> {
        if self.gate_map.contains_key(&name) {
            return Err(PyAttributeError::new_err(format!(
                "Instruction {name} is already in the target"
            )));
        }
        let props_map = properties.unwrap_or_else(|| IndexMap::from_iter([(Qargs::Global, None)]));

        self.inner_add_instruction(name, instruction, props_map, angle_bounds)
            .map_err(|err| TranspilerError::new_err(err.to_string()))?;
        Ok(())
    }

    /// Update the property object for an instruction qarg pair already in the `Target`
    ///
    /// Args:
    ///     instruction (str): The instruction name to update
    ///     qargs (tuple): The qargs to update the properties of
    ///     properties (InstructionProperties): The properties to set for this instruction
    /// Raises:
    ///     KeyError: If ``instruction`` or ``qarg`` are not in the target
    #[pyo3(name = "update_instruction_properties", signature = (instruction, qargs, properties))]
    fn py_update_instruction_properties(
        &mut self,
        instruction: String,
        qargs: Qargs,
        properties: Option<InstructionProperties>,
    ) -> PyResult<()> {
        self.update_instruction_properties(&instruction, &qargs, properties)
            .map_err(|err| PyKeyError::new_err(err.to_string()))
    }

    /// Get the qargs for a given operation name
    ///
    /// Args:
    ///     operation (str): The operation name to get qargs for
    /// Returns:
    ///     list: The list of qargs the gate instance applies to.
    #[pyo3(name = "qargs_for_operation_name")]
    pub fn py_qargs_for_operation_name(&self, operation: &str) -> PyResult<Option<Vec<&Qargs>>> {
        match self.qargs_for_operation_name(operation) {
            Ok(option_set) => Ok(option_set.map(|qargs| qargs.collect())),
            Err(e) => Err(PyKeyError::new_err(e.to_string())),
        }
    }

    /// Get the operation class object for a given name
    ///
    /// Args:
    ///     instruction (str): The instruction name to get the
    ///         :class:`~qiskit.circuit.Instruction` instance for
    /// Returns:
    ///     qiskit.circuit.Instruction: The Instruction instance corresponding to the
    ///     name. This also can also be the class for globally defined variable with
    ///     operations.
    #[pyo3(name = "operation_from_name")]
    pub fn py_operation_from_name<'py>(
        &'py self,
        py: Python<'py>,
        instruction: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self.operation_from_name(instruction) {
            Some(op) => op.into_bound_py_any(py),
            None => Err(PyKeyError::new_err(format!(
                "Instruction {instruction} not in target"
            ))),
        }
    }

    /// Get the operation class object for a specified qargs tuple
    ///
    /// Args:
    ///     qargs (tuple): A qargs tuple of the qubits to get the gates that apply
    ///         to it. For example, ``(0,)`` will return the set of all
    ///         instructions that apply to qubit 0. If set to ``None`` this will
    ///         return any globally defined operations in the target.
    /// Returns:
    ///     list: The list of :class:`~qiskit.circuit.Instruction` instances
    ///     that apply to the specified qarg. This may also be a class if
    ///     a variable width operation is globally defined.
    ///
    /// Raises:
    ///     KeyError: If qargs is not in target
    #[pyo3(name = "operations_for_qargs", signature=(qargs, /))]
    pub fn py_operations_for_qargs(&self, py: Python, qargs: Qargs) -> PyResult<Vec<Py<PyAny>>> {
        // Move to rust native once Gates are in rust
        Ok(self
            .py_operation_names_for_qargs(qargs)?
            .into_iter()
            .map(|x| {
                self.gate_map[x]
                    .instruction
                    .into_pyobject(py)
                    .as_ref()
                    .unwrap()
                    .clone()
                    .unbind()
            })
            .collect())
    }

    /// Get the operation names for a specified qargs tuple
    ///
    /// Args:
    ///     qargs (tuple): A ``qargs`` tuple of the qubits to get the gates that apply
    ///         to it. For example, ``(0,)`` will return the set of all
    ///         instructions that apply to qubit 0. If set to ``None`` this will
    ///         return the names for any globally defined operations in the target.
    /// Returns:
    ///     set: The set of operation names that apply to the specified ``qargs``.
    ///
    /// Raises:
    ///     KeyError: If ``qargs`` is not in target
    #[pyo3(name = "operation_names_for_qargs", signature=(qargs, /))]
    pub fn py_operation_names_for_qargs(&self, qargs: Qargs) -> PyResult<HashSet<&str>> {
        match self.operation_names_for_qargs(&qargs) {
            Ok(set) => Ok(set),
            Err(e) => Err(PyKeyError::new_err(e.to_string())),
        }
    }

    /// Return whether the instruction (operation + qubits) is supported by the target
    ///
    /// Args:
    ///     operation_name (str): The name of the operation for the instruction. Either
    ///         this or ``operation_class`` must be specified, if both are specified
    ///         ``operation_class`` will take priority and this argument will be ignored.
    ///     qargs (tuple): The tuple of qubit indices for the instruction. If this is
    ///         not specified then this method will return ``True`` if the specified
    ///         operation is supported on any qubits. The typical application will
    ///         always have this set (otherwise it's the same as just checking if the
    ///         target contains the operation). Normally you would not set this argument
    ///         if you wanted to check more generally that the target supports an operation
    ///         with the ``parameters`` on any qubits.
    ///     operation_class (Type[qiskit.circuit.Instruction]): The operation class to check whether
    ///         the target supports a particular operation by class rather
    ///         than by name. This lookup is more expensive as it needs to
    ///         iterate over all operations in the target instead of just a
    ///         single lookup. If this is specified it will supersede the
    ///         ``operation_name`` argument. The typical use case for this
    ///         operation is to check whether a specific variant of an operation
    ///         is supported on the backend. For example, if you wanted to
    ///         check whether a :class:`~.RXGate` was supported on a specific
    ///         qubit with a fixed angle. That fixed angle variant will
    ///         typically have a name different from the object's
    ///         :attr:`~.Instruction.name` attribute (``"rx"``) in the target.
    ///         This can be used to check if any instances of the class are
    ///         available in such a case.
    ///     parameters (list): A list of parameters to check if the target
    ///         supports them on the specified qubits. If the instruction
    ///         supports the parameter values specified in the list on the
    ///         operation and qargs specified this will return ``True`` but
    ///         if the parameters are not supported on the specified
    ///         instruction it will return ``False``. If this argument is not
    ///         specified this method will return ``True`` if the instruction
    ///         is supported independent of the instruction parameters. If
    ///         specified with any :class:`~.Parameter` objects in the list,
    ///         that entry will be treated as supporting any value, however parameter names
    ///         will not be checked (for example if an operation in the target
    ///         is listed as parameterized with ``"theta"`` and ``"phi"`` is
    ///         passed into this function that will return ``True``). For
    ///         example, if called with::
    ///
    ///             parameters = [Parameter("theta")]
    ///             target.instruction_supported("rx", (0,), parameters=parameters)
    ///
    ///         will return ``True`` if an :class:`~.RXGate` is supported on qubit 0
    ///         that will accept any parameter. If you need to check for a fixed numeric
    ///         value parameter this argument is typically paired with the ``operation_class``
    ///         argument. For example::
    ///
    ///             target.instruction_supported("rx", (0,), RXGate, parameters=[pi / 4])
    ///
    ///         will return ``True`` if an RXGate(pi/4) exists on qubit 0.
    ///     check_angle_bounds (bool): If set to True (the default) the value of ``parameters`` will
    ///         be validated against any angle bounds set in the target.
    ///         If any of the values in ``parameters`` are set to be :class:`.ParameterExpression`
    ///         instances this flag will have no effect as angle bounds only impact
    ///         non-parameterized operations in the circuit.
    ///
    /// Returns:
    ///     bool: Returns ``True`` if the instruction is supported and ``False`` if it isn't.
    #[pyo3(
        name = "instruction_supported",
        signature = (operation_name=None, qargs=Qargs::Global, operation_class=None, parameters=None, check_angle_bounds=true)
    )]
    pub fn py_instruction_supported(
        &self,
        operation_name: Option<String>,
        qargs: Qargs,
        operation_class: Option<&Bound<PyAny>>,
        parameters: Option<Vec<Param>>,
        check_angle_bounds: bool,
    ) -> PyResult<bool> {
        let mut qargs = qargs;
        let num_qubits = if let Some(num_qubits) = self.num_qubits {
            num_qubits
        } else {
            qargs = Qargs::Global;
            0
        };
        if let Some(operation_class) = operation_class {
            for (_, obj) in self.gate_map.iter() {
                match &obj.instruction {
                    TargetOperation::Variadic(variable) => {
                        if !operation_class.eq(variable)? {
                            continue;
                        }
                        // If no qargs operation class is supported
                        if let Qargs::Concrete(qargs) = &qargs {
                            return Ok(qargs.iter().all(|qarg| qarg.0 <= num_qubits));
                        } else {
                            return Ok(true);
                        }
                    }
                    TargetOperation::Normal(normal) => {
                        let py = operation_class.py();
                        if normal.into_pyobject(py)?.is_instance(operation_class)? {
                            if let Some(parameters) = &parameters {
                                if parameters.len() != normal.params_view().len() {
                                    continue;
                                }
                                if !check_obj_params(parameters, normal) {
                                    continue;
                                }
                            }
                            if let Qargs::Concrete(qargs_as_vec) = &qargs {
                                let gate_map_name = &obj.properties;
                                if gate_map_name.contains_key(&qargs.as_ref()) {
                                    return Ok(true);
                                }
                                if gate_map_name.contains_key(&Qargs::Global) {
                                    let qubit_comparison = obj.instruction.num_qubits();
                                    return Ok(qubit_comparison == qargs_as_vec.len() as u32
                                        && qargs_as_vec.iter().all(|x| x.0 < num_qubits));
                                }
                            } else {
                                return Ok(true);
                            }
                        }
                    }
                }
            }
            Ok(false)
        } else if let Some(operation_name) = operation_name {
            Ok(self.instruction_supported(
                &operation_name,
                &qargs,
                parameters.as_deref().unwrap_or_default(),
                check_angle_bounds,
            ))
        } else {
            Ok(false)
        }
    }

    /// Get the instruction properties for a specific instruction tuple
    ///
    /// This method is to be used in conjunction with the
    /// :attr:`~qiskit.transpiler.Target.instructions` attribute of a
    /// :class:`~qiskit.transpiler.Target` object. You can use this method to quickly
    /// get the instruction properties for an element of
    /// :attr:`~qiskit.transpiler.Target.instructions` by using the index in that list.
    /// However, if you're not working with :attr:`~qiskit.transpiler.Target.instructions`
    /// directly it is likely more efficient to access the target directly via the name
    /// and qubits to get the instruction properties. For example, if
    /// :attr:`~qiskit.transpiler.Target.instructions` returned::
    ///
    ///     [(XGate(), (0,)), (XGate(), (1,))]
    ///
    /// you could get the properties of the ``XGate`` on qubit 1 with::
    ///
    ///     props = target.instruction_properties(1)
    ///
    /// but just accessing it directly via the name would be more efficient::
    ///
    ///     props = target['x'][(1,)]
    ///
    /// (assuming the ``XGate``'s canonical name in the target is ``'x'``)
    /// This is especially true for larger targets as this will scale worse with the number
    /// of instruction tuples in a target.
    ///
    /// Args:
    ///     index (int): The index of the instruction tuple from the
    ///         :attr:`~qiskit.transpiler.Target.instructions` attribute. For, example
    ///         if you want the properties from the third element in
    ///         :attr:`~qiskit.transpiler.Target.instructions` you would set this to be ``2``.
    /// Returns:
    ///     InstructionProperties: The instruction properties for the specified instruction tuple
    pub fn instruction_properties(&self, index: usize) -> PyResult<Option<InstructionProperties>> {
        let mut index_counter = 0;
        for (_, props_map) in self.gate_map.iter() {
            let gate_map_oper = props_map.properties.values();
            for inst_props in gate_map_oper {
                if index_counter == index {
                    return Ok(inst_props.clone());
                }
                index_counter += 1;
            }
        }
        Err(PyIndexError::new_err(format!(
            "Index: {index:?} is out of range."
        )))
    }

    /// Return the non-global operation names for the target
    ///
    /// The non-global operations are those in the target which don't apply
    /// on all qubits (for single qubit operations) or all multi-qubit qargs
    /// (for multi-qubit operations).
    ///
    /// Args:
    ///     strict_direction (bool): If set to ``True`` the multi-qubit
    ///         operations considered as non-global respect the strict
    ///         direction (or order of qubits in the qargs is significant). For
    ///         example, if ``cx`` is defined on ``(0, 1)`` and ``ecr`` is
    ///         defined over ``(1, 0)`` by default neither would be considered
    ///         non-global, but if ``strict_direction`` is set ``True`` both
    ///         ``cx`` and ``ecr`` would be returned.
    ///
    /// Returns:
    ///     List[str]: A list of operation names for operations that aren't global in this target
    #[pyo3(name = "_get_non_global_operation_names", signature = (/, strict_direction=false,))]
    fn py_get_non_global_operation_names(&self, strict_direction: bool) -> Vec<&str> {
        self.get_non_global_operation_names(strict_direction)
    }

    // Instance attributes

    /// The dt attribute.
    #[getter(_dt)]
    fn get_dt(&self) -> Option<f64> {
        self.dt
    }

    #[setter(_dt)]
    fn set_dt(&mut self, dt: Option<f64>) {
        self.dt = dt
    }

    /// The set of qargs in the target.
    #[getter]
    #[pyo3(name = "qargs")]
    fn py_qargs(&self, py: Python) -> PyResult<Py<PyAny>> {
        if let Some(qargs) = self.qargs() {
            let set = PySet::new(py, qargs)?;
            Ok(set.into_any().unbind())
        } else {
            Ok(py.None())
        }
    }

    /// Get the list of tuples ``(:class:`~qiskit.circuit.Instruction`, (qargs))``
    /// for the target
    ///
    /// For globally defined variable width operations the tuple will be of the form
    /// ``(class, None)`` where class is the actual operation class that
    /// is globally defined.
    #[getter]
    #[pyo3(name = "instructions")]
    pub fn py_instructions(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for (inst, qargs) in self._instructions() {
            let out_inst = match inst {
                TargetOperation::Normal(op) => {
                    create_py_op(py, op.op(), op.parameters().cloned(), op.label())?
                }
                TargetOperation::Variadic(op_cls) => op_cls.clone_ref(py),
            };
            list.append((out_inst, qargs))?;
        }
        Ok(list.unbind())
    }
    /// Get the operation names in the target.
    #[getter]
    #[pyo3(name = "operation_names")]
    fn py_operation_names(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        Ok(PyList::new(py, self.operation_names())?.unbind())
    }

    /// Get the operation objects in the target.
    #[getter]
    #[pyo3(name = "operations")]
    fn py_operations(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        Ok(PyList::new(py, self.gate_map.values().map(|op| &op.instruction))?.unbind())
    }

    /// Returns a sorted list of physical qubits.
    #[getter]
    #[pyo3(name = "physical_qubits")]
    fn py_physical_qubits(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        Ok(PyList::new(py, self.physical_qubits())?.unbind())
    }

    // Magic methods:

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.gate_map.len())
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result_list = PyDict::new(py);
        result_list.set_item("description", self.description.clone())?;
        result_list.set_item("num_qubits", self.num_qubits)?;
        result_list.set_item("dt", self.dt)?;
        result_list.set_item("granularity", self.granularity)?;
        result_list.set_item("min_length", self.min_length)?;
        result_list.set_item("pulse_alignment", self.pulse_alignment)?;
        result_list.set_item("acquire_alignment", self.acquire_alignment)?;
        result_list.set_item("qubit_properties", self.qubit_properties.clone())?;
        result_list.set_item(
            "concurrent_measurements",
            self.concurrent_measurements.clone(),
        )?;
        let gate_map: Vec<(String, Bound<PyDict>)> = self
            .gate_map
            .iter()
            .map(|entry| -> PyResult<_> {
                let entry_dict = PyDict::new(py);
                entry_dict.set_item("properties", entry.1.properties.clone())?;
                entry_dict.set_item("instruction", &entry.1.instruction)?;
                entry
                    .1
                    .angle_bounds
                    .as_ref()
                    .map(|bounds| -> PyResult<_> {
                        entry_dict.set_item("angle_bounds", bounds.bounds().to_vec())
                    })
                    .transpose()?;
                Ok((entry.0.clone(), entry_dict))
            })
            .collect::<PyResult<_>>()?;
        result_list.set_item("gate_map", gate_map.into_py_dict(py)?)?;
        result_list.set_item("global_operations", self.global_operations.clone())?;
        result_list.set_item("qarg_gate_map", self.qarg_gate_map.clone())?;
        Ok(result_list.unbind())
    }

    fn __setstate__(&mut self, state: Bound<PyDict>) -> PyResult<()> {
        self.description = state
            .get_item("description")?
            .unwrap()
            .extract::<Option<String>>()?;
        self.num_qubits = state
            .get_item("num_qubits")?
            .unwrap()
            .extract::<Option<u32>>()?;
        self.dt = state.get_item("dt")?.unwrap().extract::<Option<f64>>()?;
        self.granularity = state.get_item("granularity")?.unwrap().extract::<u32>()?;
        self.min_length = state.get_item("min_length")?.unwrap().extract::<u32>()?;
        self.pulse_alignment = state
            .get_item("pulse_alignment")?
            .unwrap()
            .extract::<u32>()?;
        self.acquire_alignment = state
            .get_item("acquire_alignment")?
            .unwrap()
            .extract::<u32>()?;
        self.qubit_properties = state
            .get_item("qubit_properties")?
            .unwrap()
            .extract::<Option<Vec<QubitProperties>>>()?;
        self.concurrent_measurements = state
            .get_item("concurrent_measurements")?
            .unwrap()
            .extract::<Option<Vec<Vec<PhysicalQubit>>>>()?;
        let source_map = state.get_item("gate_map")?.unwrap().cast_into::<PyDict>()?;
        let mut gate_map =
            IndexMap::with_capacity_and_hasher(source_map.len(), RandomState::default());
        type Bounds = SmallVec<[Option<[f64; 2]>; 3]>;
        let mut bounds_map: Vec<(String, Bounds)> = Vec::new();
        for (key, value) in source_map.iter() {
            let value = value.cast::<PyDict>()?;
            let angle_bound = value
                .get_item("angle_bounds")?
                .map(|out| -> PyResult<_> { out.extract::<SmallVec<[Option<[f64; 2]>; 3]>>() })
                .transpose()?;
            let key: String = key.extract()?;
            if let Some(angle_bound) = angle_bound {
                bounds_map.push((key.clone(), angle_bound));
            }
            gate_map.insert(
                key,
                TargetProperties {
                    properties: value.get_item("properties")?.unwrap().extract()?,
                    instruction: value.get_item("instruction")?.unwrap().extract()?,
                    angle_bounds: None,
                },
            );
        }
        self.gate_map = gate_map;
        self.qarg_gate_map = state.get_item("qarg_gate_map")?.unwrap().extract()?;
        self.global_operations = state
            .get_item("global_operations")?
            .unwrap()
            .extract::<HashMap<u32, HashSet<String>>>()?;
        for (gate, bounds) in bounds_map {
            self.add_owned_angle_bound(gate, bounds)
                .map_err(|err| TranspilerError::new_err(err.to_string()))?;
        }
        Ok(())
    }

    /// Check if there are any angle bounds set in the target
    ///
    /// Returns:
    ///     bool: This will return ``True`` if there are angle bounds set on any instructions in
    ///     the circuit
    pub fn has_angle_bounds(&self) -> bool {
        self.has_angle_bounds
    }

    /// Check if a specific gate gate has an angle bound set
    ///
    /// Args:
    ///     name (str): The instruction name to check if it has an angle bound set
    ///
    /// Returns:
    ///     bool: This will return ``True`` if the gate is in the target and has angle bounds
    ///     defined. It will return ``False`` if the gate does not have angle bounds defined
    ///     or is not in the target.
    pub fn gate_has_angle_bounds(&self, name: &str) -> bool {
        self.gate_map
            .get(name)
            .is_some_and(|props| props.angle_bounds.is_some())
    }

    /// Check that parameters on a specific gate conform to the angle bounds
    ///
    /// Args:
    ///     name (str): The instruction name to check the angle bounds of
    ///     angles (list): A list of float parameter values for ``name``
    ///         to see if they conform to the defined angle bounds.
    ///
    /// Returns:
    ///     bool: Returns ``True`` if the parameter values specified are compatible with the
    ///     angle bounds. ``False`` is returned if the any of the parameters
    ///     are outside the defined bounds.
    ///
    /// Raises:
    ///     TranspilerError: If ``name`` is not in the target or does not
    ///     have angle bounds defined.
    ///
    pub fn supported_angle_bound(&self, name: &str, angles: Vec<f64>) -> PyResult<bool> {
        if !self.gate_has_angle_bounds(name) {
            Err(TranspilerError::new_err(format!(
                "The specified gate {name} does not have angle bounds defined or is not in the Target"
            )))
        } else {
            Ok(self.gate_map[name]
                .angle_bounds
                .as_ref()
                .is_some_and(|bound| bound.angles_supported(&angles)))
        }
    }
}

// Rust native methods
impl Target {
    /// Adds a [PackedOperation] to the [Target].
    ///
    /// Said addition results in a [NormalOperation] in the [Target] as variadics
    /// are not yet supported natively. If no properties are specified the operation
    /// is believed to be `Global` with properties `{Qargs::Global: None}`.
    ///
    /// # Arguments
    ///
    /// * `operation` - The [PackedOperation] to be added.
    /// * `params` - The [Parameter]s collection assigned to the instruction.
    /// * `name` - The name of the instruction if differs from the [PackedOperation]
    ///   instance. If set to `None` it defaults to the string returned by [`Operation::name`] for `operation`.
    /// * `props_map`: The optional property mapping between [Qargs] and
    ///   [InstructionProperties]. If set to `None` the instruction is treated as a global ideal instruction.
    ///
    /// # Returns
    ///
    /// * `Ok`: if the instruction property is successfully added.
    /// * `Err`: (if the instruction already exists or any of the qargs do not match
    ///   the instruction's number of qubits) [TargetError].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use qiskit_transpiler::target::Target;
    /// use qiskit_circuit::operations::StandardGate;
    ///
    /// let mut target = Target::default();
    /// let result = target.add_instruction(
    ///     StandardGate::X.into(),
    ///     None,
    ///     None,
    ///     None,
    /// );
    ///
    /// assert!(matches!(result, Ok(())));
    /// ```
    pub fn add_instruction(
        &mut self,
        operation: PackedOperation,
        params: Option<Parameters<CircuitData>>,
        name: Option<&str>,
        props_map: Option<IndexMap<Qargs, Option<InstructionProperties>>>,
    ) -> Result<(), TargetError> {
        let parsed_name = if let Some(name) = name {
            name.to_string()
        } else {
            operation.name().to_string()
        };
        let argument_num = params.as_ref().map(|p| p.len()).unwrap_or(0);
        if argument_num != operation.num_params() as usize {
            return Err(TargetError::ParamsMismatch {
                instruction: parsed_name,
                instruction_num: operation.num_params() as usize,
                argument_num,
            });
        }

        if self.gate_map.contains_key(&parsed_name) {
            return Err(TargetError::AlreadyExists(parsed_name));
        }
        let operation = TargetOperation::from_packed_operation(operation, params);
        let props_map = if let Some(props_map) = props_map {
            props_map
        } else {
            IndexMap::from_iter([(Qargs::Global, None)])
        };

        self.inner_add_instruction(parsed_name, operation, props_map, None)
    }

    fn inner_add_instruction(
        &mut self,
        name: String,
        instruction: TargetOperation,
        properties: IndexMap<Qargs, Option<InstructionProperties>>,
        angle_bounds: Option<SmallVec<[Option<[f64; 2]>; 3]>>,
    ) -> Result<(), TargetError> {
        let properties = match instruction {
            TargetOperation::Variadic(_) => IndexMap::from_iter([(Qargs::Global, None)]),
            TargetOperation::Normal(_) => {
                if properties.contains_key(&Qargs::Global) {
                    self.global_operations
                        .entry(instruction.num_qubits())
                        .and_modify(|e| {
                            e.insert(name.to_string());
                        })
                        .or_insert(HashSet::from_iter([name.to_string()]));
                }
                for qarg in properties.keys() {
                    if let QargsRef::Concrete(qarg_slice) = qarg.as_ref() {
                        if qarg_slice.len() != instruction.num_qubits() as usize {
                            return Err(TargetError::QargsMismatch {
                                instruction: name.to_string(),
                                arguments: format!("{qarg:?}"),
                            });
                        }
                        let max = qarg_slice.iter().max().map(|idx| idx.0 + 1);
                        if max > self.num_qubits {
                            self.num_qubits = max;
                        }
                    }
                    if let Some(value) = self.qarg_gate_map.get_mut(qarg) {
                        value.insert(name.to_string());
                    } else {
                        self.qarg_gate_map
                            .insert(qarg.clone(), HashSet::from_iter([name.to_string()]));
                    }
                }
                properties
            }
        };

        self.gate_map.insert(
            name.to_string(),
            TargetProperties {
                properties,
                instruction,
                angle_bounds: None,
            },
        );
        angle_bounds
            .map(|angle_bounds| {
                self.check_bounds_inputs(&name, &angle_bounds)?;
                self.add_owned_angle_bound(name, angle_bounds)
            })
            .transpose()?;
        Ok(())
    }

    /// Update the property object for an instruction qarg pair already in the [Target].
    ///
    /// # Arguments
    ///
    /// * `instruction` - The instruction's name within this instance.
    /// * `qargs` - A collection of [PhysicalQubit] or an instance of [Qargs::Global]
    ///   that the instruction operated on.
    /// * `properties` - The properties to use for updating the specified instruction in the target.
    ///
    /// # Returns
    ///
    /// * `Ok`: if the instruction property is successfully updated.
    /// * `Err`: (if neither the instruction name or qarg aren't found) [TargetError].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use qiskit_transpiler::target::{Target, InstructionProperties, Qargs};
    /// use qiskit_circuit::operations::StandardGate;
    /// use qiskit_circuit::PhysicalQubit;
    /// use qiskit_util::IndexMap;
    ///
    /// let mut target = Target::default();
    /// target.add_instruction(
    ///     StandardGate::X.into(),
    ///     &[],
    ///     None,
    ///     Some(IndexMap::from_iter([([PhysicalQubit(0)].into(), None)])),
    /// );
    /// let result = target.update_instruction_properties("x", &[PhysicalQubit(0)], Some(InstructionProperties::new(Some(0.0001), Some(0.0002))));
    ///
    /// assert!(matches!(result, Ok(())));
    /// ```
    pub fn update_instruction_properties<'a, T>(
        &mut self,
        instruction: &'a str,
        qargs: T,
        properties: Option<InstructionProperties>,
    ) -> Result<(), TargetError>
    where
        T: Into<QargsRef<'a>>,
    {
        if !self.contains_key(instruction) {
            return Err(TargetError::InvalidKey(instruction.to_string()));
        };
        let qargs: QargsRef = qargs.into();
        let prop_map = self.gate_map.get_mut(instruction).unwrap();
        if !prop_map.properties.contains_key(&qargs) {
            return Err(TargetError::InvalidQargsKey {
                instruction: instruction.to_string(),
                arguments: format!("{qargs:?}"),
            });
        }
        if let Some(e) = prop_map.properties.get_mut(&qargs) {
            *e = properties;
        }
        Ok(())
    }

    /// Returns an iterator over all the instructions present in the `Target`
    /// as pair of `&OperationType`, `&SmallVec<[Param; 3]>` and `Option<&Qargs>`.
    pub fn instructions(&self) -> impl Iterator<Item = (&NormalOperation, &Qargs)> {
        self._instructions()
            .filter_map(|(operation, qargs)| match &operation {
                TargetOperation::Normal(oper) => Some((oper, qargs)),
                _ => None,
            })
    }

    /// Returns an iterator over all the instructions present in the `Target`
    /// as pair of `&TargetOperation` and `Option<&Qargs>`.
    fn _instructions(&self) -> impl Iterator<Item = (&TargetOperation, &Qargs)> {
        self.gate_map.iter().flat_map(move |(_, props_map)| {
            props_map
                .properties
                .keys()
                .map(move |qargs| (&props_map.instruction, qargs))
        })
    }

    /// Returns an iterator over the operation names in the target.
    pub fn operation_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.gate_map.keys().map(|x| x.as_str())
    }

    /// Get the `OperationType` objects present in the target.
    pub fn operations(&self) -> impl Iterator<Item = &NormalOperation> {
        self.gate_map
            .values()
            .filter_map(|oper| match &oper.instruction {
                TargetOperation::Normal(oper) => Some(oper),
                _ => None,
            })
    }

    /// Get the complete [InstructionProperties] from the [Target] for the given instruction key and
    /// qargs.
    pub fn get_instruction_properties<'a, T>(
        &self,
        name: &str,
        qargs: T,
    ) -> Option<&InstructionProperties>
    where
        T: Into<QargsRef<'a>>,
    {
        self.gate_map.get(name).and_then(|gate_props| {
            gate_props
                .properties
                .get(&qargs.into())
                .and_then(|props| props.as_ref())
        })
    }

    /// Get the error rate of a given instruction in the target
    #[inline]
    pub fn get_error<'a, T>(&self, name: &str, qargs: T) -> Option<f64>
    where
        T: Into<QargsRef<'a>>,
    {
        self.get_instruction_properties(name, qargs)
            .and_then(|props| props.error)
    }

    /// Get the duration of a given instruction in the target
    #[inline]
    pub fn get_duration<'a, T>(&self, name: &str, qargs: T) -> Option<f64>
    where
        T: Into<QargsRef<'a>>,
    {
        self.get_instruction_properties(name, qargs)
            .and_then(|props| props.duration)
    }

    /// Get an iterator over the indices of all physical qubits of the target
    pub fn physical_qubits(&self) -> impl ExactSizeIterator<Item = PhysicalQubit> + use<> {
        (0..self.num_qubits.unwrap_or_default()).map(PhysicalQubit)
    }

    /// Get all non_global operation names.
    pub fn get_non_global_operation_names(&self, strict_direction: bool) -> Vec<&str> {
        let mut search_set: HashSet<SmallVec<[PhysicalQubit; 2]>> = HashSet::default();
        if strict_direction {
            // Build search set
            search_set = self
                .qarg_gate_map
                .keys()
                .filter_map(|qargs| match qargs {
                    Qargs::Global => None,
                    Qargs::Concrete(vec) => Some(vec.clone()),
                })
                .collect();
        } else {
            for qarg_key in self
                .qarg_gate_map
                .keys()
                .filter_map(|qargs| match qargs {
                    Qargs::Global => None,
                    Qargs::Concrete(vec) => Some(vec),
                })
                .cloned()
            {
                if qarg_key.len() != 1 {
                    let mut vec = qarg_key;
                    vec.sort_unstable();
                    search_set.insert(vec);
                }
            }
        }
        let mut incomplete_basis_gates: Vec<&str> = Vec::new();
        let mut size_dict: IndexMap<u32, u32> = IndexMap::default();
        *size_dict
            .entry(1)
            .or_insert(self.num_qubits.unwrap_or_default()) = self.num_qubits.unwrap_or_default();
        for qarg in &search_set {
            if qarg.len() == 1 {
                continue;
            }
            *size_dict.entry(qarg.len() as u32).or_insert(0) += 1;
        }
        for (inst, qargs_props) in self.gate_map.iter() {
            let mut qarg_len = qargs_props.properties.len() as u32;
            let mut qargs_keys = qargs_props.properties.keys().peekable();
            let qarg_sample = qargs_keys.peek().cloned();
            if let Some(qarg_sample) = qarg_sample {
                if qarg_sample.is_global() {
                    continue;
                }
                if !strict_direction {
                    let mut deduplicated_qargs: HashSet<SmallVec<[PhysicalQubit; 2]>> =
                        HashSet::default();
                    for qarg in qargs_keys.filter_map(|qargs| match qargs {
                        Qargs::Global => None,
                        Qargs::Concrete(qargs) => Some(qargs),
                    }) {
                        let mut ordered_qargs = qarg.clone();
                        ordered_qargs.sort_unstable();
                        deduplicated_qargs.insert(ordered_qargs);
                    }
                    qarg_len = deduplicated_qargs.len() as u32;
                }
                if let Qargs::Concrete(qarg_sample) = qarg_sample {
                    if qarg_len != *size_dict.entry(qarg_sample.len() as u32).or_insert(0) {
                        incomplete_basis_gates.push(inst.as_str());
                    }
                }
            }
        }
        incomplete_basis_gates
    }

    /// Gets all the operation names that use these qargs. Rust native equivalent of ``BaseTarget.operation_names_for_qargs()``
    pub fn operation_names_for_qargs<'a, T>(&self, qargs: T) -> Result<HashSet<&str>, TargetError>
    where
        T: Into<QargsRef<'a>>,
    {
        // When num_qubits == 0 we return globally defined operators
        let mut res: HashSet<&str> = HashSet::default();
        let mut qargs: QargsRef = qargs.into();
        if self.num_qubits.unwrap_or_default() == 0 || self.num_qubits.is_none() {
            qargs = QargsRef::Global;
        }
        if let QargsRef::Concrete(qargs) = qargs {
            if qargs
                .iter()
                .any(|x| !(0..self.num_qubits.unwrap_or_default()).contains(&x.0))
            {
                return Err(TargetError::QargsWithoutInstruction(format!("{qargs:?}")));
            }
        }
        if let Some(qarg_gate_map_arg) = self.qarg_gate_map.get(&qargs) {
            res.extend(qarg_gate_map_arg.iter().map(|key| key.as_str()));
        }
        for (name, obj) in self.gate_map.iter() {
            if matches!(obj.instruction, TargetOperation::Variadic(_)) {
                res.insert(name);
            }
        }
        if let QargsRef::Concrete(qargs) = qargs {
            if let Some(global_gates) = self.global_operations.get(&(qargs.len() as u32)) {
                res.extend(global_gates.iter().map(|key| key.as_str()))
            }
        }
        if res.is_empty() {
            return Err(TargetError::QargsWithoutInstruction(format!("{qargs:?}")));
        }
        Ok(res)
    }

    /// Returns an iterator of `OperationType` instances and parameters present in the Target that
    /// affect the provided qargs.
    pub fn operations_for_qargs<'a, T>(
        &self,
        qargs: T,
    ) -> Result<impl Iterator<Item = &NormalOperation> + use<'_, T>, TargetError>
    where
        T: Into<QargsRef<'a>>,
    {
        self.operation_names_for_qargs(qargs).map(|operations| {
            operations
                .into_iter()
                .filter_map(|oper| match &self.gate_map[oper].instruction {
                    TargetOperation::Normal(normal) => Some(normal),
                    _ => None,
                })
        })
    }

    pub fn num_qargs(&self) -> usize {
        self.qarg_gate_map.len()
    }

    /// Gets an iterator with all the qargs used by the specified operation name.
    ///
    /// Rust native equivalent of ``BaseTarget.qargs_for_operation_name()``
    pub fn qargs_for_operation_name(
        &self,
        operation: &str,
    ) -> Result<Option<impl Iterator<Item = &Qargs> + use<'_>>, TargetError> {
        match self.gate_map.get(operation) {
            Some(gate_map_oper) => {
                if gate_map_oper.properties.contains_key(&Qargs::Global) {
                    return Ok(None);
                }
                let qargs = gate_map_oper
                    .properties
                    .keys()
                    .filter(|qargs| qargs.is_concrete());
                Ok(Some(qargs))
            }
            None => Err(TargetError::InvalidKey(operation.to_string())),
        }
    }

    /// Retrieve the backing representation of an operation name in the target, if it exists.
    pub fn operation_from_name(&self, instruction: &str) -> Option<&TargetOperation> {
        self.gate_map
            .get(instruction)
            .map(|entry| &entry.instruction)
    }

    /// Returns an iterator over all the qargs of a specific Target object
    pub fn qargs(&self) -> Option<impl Iterator<Item = &Qargs>> {
        let qargs = self.qarg_gate_map.keys();
        if qargs.len() == 1 && self.qarg_gate_map.contains_key(&Qargs::Global) {
            return None;
        }
        Some(qargs)
    }

    /// Checks whether an instruction is supported by the Target based on instruction name and qargs.
    /// # Arguments
    ///
    /// * `operation_name` - The instruction's name to check for.
    /// * `qargs` - A collection of [PhysicalQubit] or an instance of [Qargs::Global] that the instruction
    ///   might operate on.
    /// * `parameters` - The parameters that will be assigned to the gate.
    /// * `check_angle_bounds` - To decide if we will check the angle bounds of the provided parameters.
    ///
    /// # Returns
    ///
    /// * `true` if the instruction is compatible with the target, `false` if otherwise.
    pub fn instruction_supported<'a, T>(
        &self,
        operation_name: &str,
        qargs: T,
        parameters: &[Param],
        check_angle_bounds: bool,
    ) -> bool
    where
        T: Into<QargsRef<'a>>,
    {
        // Unwrap the num_qubits and cache it
        let num_qubits = self.num_qubits.unwrap_or_default();
        // Handle case where num_qubits is None by checking globally supported operations
        let qargs: QargsRef = if self.num_qubits.is_none() {
            QargsRef::Global
        } else {
            qargs.into()
        };
        if let Some(obj) = self.gate_map.get(operation_name) {
            if !parameters.is_empty() {
                let obj_params = match &obj.instruction {
                    TargetOperation::Variadic(_) => {
                        return match qargs {
                            QargsRef::Concrete(qargs) => {
                                qargs.iter().all(|qarg| qarg.0 <= num_qubits)
                            }
                            QargsRef::Global => true,
                        };
                    }
                    TargetOperation::Normal(normal) => normal.parameters(),
                };
                let Some(Parameters::Params(obj_params)) = obj_params else {
                    // We've either got parameters incident to the method, but the operation we've
                    // got stored doesn't take any, or the parameters we have stored are irregular.
                    return false;
                };
                if parameters.len() != obj_params.len() {
                    return false;
                }

                for (params, orig_params) in parameters.iter().zip(obj_params) {
                    let matching_params = match (orig_params, params) {
                        (Param::Float(obj_f), Param::Float(param_f)) => obj_f == param_f,
                        (Param::ParameterExpression(_), _) => true,
                        (Param::Float(obj_f), Param::ParameterExpression(expr)) => {
                            expr.try_to_value(true).is_ok_and(|value| value.eq(obj_f))
                        }
                        _ => Python::attach(|py| python_compare(py, params, orig_params))
                            .expect("Error comparing Python parameters."),
                    };

                    if !matching_params {
                        return false;
                    }
                }
                if check_angle_bounds
                    && self.has_angle_bounds()
                    && parameters.iter().all(|x| matches!(x, Param::Float(_)))
                {
                    let params: Vec<f64> = parameters
                        .iter()
                        .map(|x| {
                            let Param::Float(val) = x else { unreachable!() };
                            *val
                        })
                        .collect();
                    if obj.angle_bounds.is_some()
                        && !self.gate_supported_angle_bound(operation_name, &params)
                    {
                        return false;
                    }
                }
            }
            let QargsRef::Concrete(qargs_as_vec) = qargs else {
                return true;
            };
            if obj.properties.contains_key(&qargs) {
                return true;
            }
            if obj.properties.contains_key(&QargsRef::Global) {
                match &obj.instruction {
                    TargetOperation::Variadic(_) => {
                        return qargs_as_vec.iter().all(|qarg| qarg.0 <= num_qubits);
                    }
                    TargetOperation::Normal(obj) => {
                        let qubit_comparison = obj.operation.num_qubits();
                        return qubit_comparison == qargs_as_vec.len() as u32
                            && qargs_as_vec.iter().all(|qarg| qarg.0 < num_qubits);
                    }
                }
            }
        }
        false
    }

    /// Get a directionless coupling-graph representation of the target connectivity.
    ///
    /// This only makes sense for targets without all-to-all connectivity, and that do not have any
    /// interactions that are more than 2q.  In either of these cases, the relevant error state is
    /// returned.
    ///
    /// Since information about the actual instructions is erased, it does not make sense to attempt
    /// to preserve directionality.
    pub fn coupling_graph(&self) -> Result<Graph<(), (), Undirected>, TargetCouplingError> {
        let Some(num_qubits) = self.num_qubits else {
            // Actually, this mostly means that nothing has set it yet, so there's no explicit
            // number given, and the only possible operations are all-to-all.  It doesn't matter a
            // lot, though, because `None` mostly just means that nothing has initialised it, so
            // construction of the object isn't complete.
            return Err(TargetCouplingError::AllToAll);
        };
        let num_qubits = num_qubits as usize;
        let mut coupling = Graph::with_capacity(num_qubits, num_qubits);
        for _ in 0..num_qubits {
            coupling.add_node(());
        }
        let Some(qargs) = self.qargs() else {
            return Err(TargetCouplingError::AllToAll);
        };
        let mut multi_q = false;
        for qargs in qargs {
            let Qargs::Concrete(qargs) = qargs else {
                if self.global_operations.keys().any(|x| *x == 2) {
                    return Err(TargetCouplingError::AllToAll);
                }
                if self.global_operations.keys().any(|x| *x > 2) {
                    multi_q = true;
                }
                continue;
            };
            match qargs.as_slice() {
                &[] | &[_] => (),
                &[a, b] => {
                    coupling.update_edge(NodeIndex::new(a.index()), NodeIndex::new(b.index()), ());
                }
                _ => {
                    multi_q = true;
                }
            }
        }
        if multi_q {
            Err(TargetCouplingError::MultiQ(coupling))
        } else {
            Ok(coupling)
        }
    }

    // IndexMap methods

    /// Retreive all the gate names in the Target
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &str> {
        self.gate_map.keys().map(|x| x.as_str())
    }

    /// Retrieves an iterator over the property maps stored within the Target
    pub fn values(
        &self,
    ) -> impl ExactSizeIterator<Item = &IndexMap<Qargs, Option<InstructionProperties>>> {
        self.gate_map.values().map(|props| &props.properties)
    }

    /// Checks if a key exists in the Target
    pub fn contains_key(&self, key: &str) -> bool {
        self.gate_map.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.gate_map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gate_map.is_empty()
    }

    fn check_bounds_inputs(
        &self,
        name: &str,
        bounds: &[Option<[f64; 2]>],
    ) -> Result<(), TargetError> {
        let num_bounds = bounds.len();
        let Some(operation) = self.operation_from_name(name) else {
            return Err(TargetError::InvalidKey(format!(
                "{name} is not an instruction in the target."
            )));
        };
        let num_params = match operation {
            TargetOperation::Normal(op) => {
                let params = op.params_view();
                if params
                    .iter()
                    .zip(bounds)
                    .any(|(param, bound)| bound.is_some() && matches!(param, Param::Float(_)))
                {
                    return Err(TargetError::InvalidKey(
                        "Angle bound set on a fixed value".to_string(),
                    ));
                }
                params.len()
            }
            TargetOperation::Variadic(_) => 0,
        };
        if num_bounds != num_params {
            return Err(TargetError::InvalidKey(format!(
                "The number of bounds {num_bounds} doesn't match the gate's {num_params}"
            )));
        }
        Ok(())
    }

    /// Add an angle bound to the parameter of a gate in the target
    pub fn add_angle_bound(
        &mut self,
        name: String,
        bounds: &[Option<[f64; 2]>],
    ) -> Result<(), TargetError> {
        self.check_bounds_inputs(&name, bounds)?;
        let new_bound = AngleBound::new(bounds.iter().copied().collect())?;
        if !self.has_angle_bounds {
            self.has_angle_bounds = true;
        }
        self.gate_map[&name].angle_bounds = Some(new_bound);
        Ok(())
    }

    /// Add an owned angle bound constraint on gate.
    fn add_owned_angle_bound(
        &mut self,
        name: String,
        bounds: SmallVec<[Option<[f64; 2]>; 3]>,
    ) -> Result<(), TargetError> {
        self.check_bounds_inputs(&name, &bounds)?;
        let new_bound = AngleBound::new(bounds)?;
        if !self.has_angle_bounds {
            self.has_angle_bounds = true;
        }
        self.gate_map[&name].angle_bounds = Some(new_bound);
        Ok(())
    }

    /// Check that a gates angle bounds are supported
    pub fn gate_supported_angle_bound(&self, name: &str, angles: &[f64]) -> bool {
        self.gate_map[name]
            .angle_bounds
            .as_ref()
            .is_some_and(|bound| bound.angles_supported(angles))
    }

    /// Check that a given qargs is present in the target
    pub fn contains_qargs<'a, T: Into<QargsRef<'a>>>(&self, qargs: T) -> bool {
        self.qarg_gate_map.contains_key(&qargs.into())
    }

    /// Retrieves a gate location in the gate map by index
    pub fn get_gate_index(&self, gate_name: &str) -> Option<usize> {
        self.gate_map.get_index_of(gate_name)
    }

    /// Retrieves a gate location in the gate map by index
    pub fn get_by_index(&self, index: usize) -> Option<(&str, &<Self as Index<&str>>::Output)> {
        self.gate_map
            .get_index(index)
            .map(|(name, props)| (name.as_str(), &props.properties))
    }
    /// Retrieves a gate location in the gate map by index
    pub fn get_op_by_index(&self, index: usize) -> Option<&TargetOperation> {
        self.gate_map
            .get_index(index)
            .map(|(_, op)| &op.instruction)
    }
}

// To access the Target's gate map by gate name.
impl Index<&str> for Target {
    type Output = IndexMap<Qargs, Option<InstructionProperties>>;
    fn index(&self, index: &str) -> &Self::Output {
        &self.gate_map.index(index).properties
    }
}

impl Default for Target {
    fn default() -> Self {
        Self {
            description: None,
            num_qubits: Default::default(),
            dt: None,
            granularity: 1,
            min_length: 1,
            pulse_alignment: 1,
            acquire_alignment: 1,
            qubit_properties: None,
            concurrent_measurements: None,
            gate_map: Default::default(),
            global_operations: Default::default(),
            qarg_gate_map: Default::default(),
            has_angle_bounds: false,
        }
    }
}

#[derive(Error, Debug)]
pub enum TargetCouplingError {
    #[error("target contains short-hand all-to-all connectivity")]
    AllToAll,
    #[error("target contains multi-qubit operations")]
    MultiQ(Graph<(), (), Undirected>),
}

// For instruction_supported
fn check_obj_params(parameters: &[Param], obj: &NormalOperation) -> bool {
    for (index, param) in parameters.iter().enumerate() {
        let param_at_index = &obj.params_view()[index];
        match (param, param_at_index) {
            (Param::Float(p1), Param::Float(p2)) => {
                if p1 != p2 {
                    return false;
                }
            }
            (&Param::ParameterExpression(_), Param::Float(_)) => return false,
            (&Param::ParameterExpression(_), Param::Obj(_)) => return false,
            _ => continue,
        }
    }
    true
}

pub fn python_compare<'a, T, U>(py: Python<'a>, obj: T, other: U) -> PyResult<bool>
where
    T: IntoPyObject<'a>,
    U: IntoPyObject<'a>,
{
    let obj = obj.into_bound_py_any(py)?;
    obj.eq(other.into_bound_py_any(py)?)
}

/// Estimate the fidelity of a circuit by taking the product of the error rates
///
/// It is assumed that, when this function is called, the circuit is already physical (already transpiled)
/// such that the qubits for each gate correspond to the physical qubits in the target.
///
/// Args:
///     circuit: The circuit to estimate the fidelity of
///     target: The target the circuit will be run on
///
/// Returns:
///     The estimated fidelity. If any operation can't be found in the target it will return None.
#[pyfunction(name = "estimate_fidelity")]
pub fn py_estimate_fidelity(circuit: &PyCircuitData, target: &Target) -> Option<f64> {
    estimate_fidelity(circuit, target)
}

/// Estimate the fidelity of a circuit by taking the product of the error rates
///
/// It is assumed that, when this function is called, the circuit is already physical (already transpiled)
/// such that the qubits for each gate correspond to the physical qubits in the target.
///
/// # Args
///
/// `circuit`: The circuit to estimate the fidelity of
/// `target`: The target the circuit will be run on
///
/// # Returns
///
/// The estimated fidelity. If any operation can't be found in the target it will return None.
pub fn estimate_fidelity(circuit: &CircuitData, target: &Target) -> Option<f64> {
    circuit
        .data()
        .iter()
        .filter(|inst| !inst.op.directive())
        .map(|inst| {
            let qubits = circuit.get_qargs(inst.qubits);
            let gate_name = inst.op.name();
            let physical_qubits: &[PhysicalQubit] = PhysicalQubit::lift_slice(qubits);
            match target.get_instruction_properties(gate_name, physical_qubits) {
                Some(props) => Some(1. - props.error.unwrap_or(0.)),
                None => {
                    // If there is no instruction properties this either is because either the instruction
                    // isn't supported or it is global and ideal. Check if it's supported then
                    // treat as ideal, otherwise invalidate the fidelity because the instruction
                    // isn't supported.
                    if target.instruction_supported(
                        gate_name,
                        physical_qubits,
                        inst.params_view(),
                        true,
                    ) {
                        // Check that there aren't any instruction properties for a global entry
                        // (which applies to all valid qargs) otherwise treat as ideal.
                        if let Some(props) =
                            target.get_instruction_properties(gate_name, &Qargs::Global)
                        {
                            Some(1. - props.error.unwrap_or(0.))
                        } else {
                            Some(1.)
                        }
                    } else {
                        None
                    }
                }
            }
        })
        .product()
}

pub fn target(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<InstructionProperties>()?;
    m.add_class::<Target>()?;
    m.add_class::<QubitProperties>()?;
    m.add_wrapped(wrap_pyfunction!(py_estimate_fidelity))?;
    Ok(())
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;
    use std::sync::Arc;

    use crate::target::QargsRef;
    use qiskit_circuit::PhysicalQubit;
    use qiskit_circuit::instruction::Parameters;
    use qiskit_circuit::operations::{
        Operation, Param, STANDARD_GATE_SIZE, StandardGate, get_standard_gate_names,
    };
    use qiskit_circuit::packed_instruction::PackedOperation;
    use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
    use qiskit_circuit::parameter::symbol_expr::Symbol;
    use smallvec::{SmallVec, smallvec};

    use super::{Qargs, Target, TargetError, instruction_properties::InstructionProperties};

    #[test]
    fn test_invalid_params_instruction() {
        let params = smallvec![
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("ϴ".to_owned(), None,)
            ))),
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("φ".to_owned(), None,)
            ))),
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("λ".to_owned(), None,)
            ))),
        ];
        let mut target = Target::default();
        let result = target.add_instruction(
            PackedOperation::from_standard_gate(StandardGate::CX),
            Some(Parameters::Params(params)),
            None,
            None,
        );
        let Err(res) = result else {
            panic!("The operation was unexpectedly successful");
        };
        if !matches!(res, TargetError::ParamsMismatch { .. }) {
            panic!("Returned an unexpected error type");
        }
        assert_eq!(
            res.to_string(),
            "The number of parameters for cx: 0 does not match the provided number of parameters: 3.",
        );
    }

    #[test]
    fn test_mismatch_params_count_instruction() {
        let params = smallvec![
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("ϴ".to_owned(), None,)
            ))),
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("φ".to_owned(), None,)
            ))),
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("λ".to_owned(), None,)
            ))),
        ];
        let mut target = Target::default();
        let result = target.add_instruction(
            PackedOperation::from_standard_gate(StandardGate::RZ),
            Some(Parameters::Params(params)),
            None,
            None,
        );
        let Err(res) = result else {
            panic!("The operation was unexpectedly successful");
        };
        if !matches!(res, TargetError::ParamsMismatch { .. }) {
            panic!("Returned an unexpected error type");
        }
        assert_eq!(
            res.to_string(),
            "The number of parameters for rz: 1 does not match the provided number of parameters: 3.",
        );
    }

    #[test]
    fn test_add_invalid_qargs_insruction() {
        let qargs: SmallVec<[PhysicalQubit; 2]> = (0..4).map(PhysicalQubit).collect();
        let inst_prop: Option<InstructionProperties> = None;

        let mut target = Target::default();
        let result = target.add_instruction(
            StandardGate::CZ.into(),
            None,
            None,
            Some([(qargs.clone().into(), inst_prop)].into_iter().collect()),
        );
        let Err(res) = result else {
            panic!("The operation did not fail as expected.");
        };
        let expected_message = format!(
            "The number of qubits for cz does not match the number of qubits in the properties dictionary: {:?}.",
            Qargs::Concrete(qargs)
        );
        assert_eq!(res.to_string(), expected_message);
    }

    #[test]
    fn test_add_invalid_repeated_insruction() {
        let mut target = Target::default();
        let result = target.add_instruction(StandardGate::CX.into(), None, None, None);
        assert!(result.is_ok());

        let result = target.add_instruction(StandardGate::CX.into(), None, None, None);
        // Re-add instruction
        let Err(res) = result else {
            panic!("The operation did not fail as expected.");
        };
        let expected_message = "Instruction 'cx' is already in the target.".to_string();
        assert_eq!(res.to_string(), expected_message);
    }

    #[test]
    fn test_add_all_standard_gates() {
        let mut all_standard_target = Target::default();
        // Update this if any standard gates are added.

        for gate in 0..STANDARD_GATE_SIZE {
            // Safety: `STANDARD_GATE_SIZE` will always be in range for StandardGate.
            let gate: StandardGate = unsafe { std::mem::transmute(gate as u8) };
            let num_qubits = gate.num_qubits();
            let num_params = gate.num_params();

            let qargs: Qargs = (0..num_qubits).map(PhysicalQubit).collect();
            let params: SmallVec<[Param; 3]> = (0..num_params)
                .map(|val| Param::from(PI / (val as f64)))
                .collect();

            let res = all_standard_target.add_instruction(
                gate.into(),
                Some(Parameters::Params(params)),
                None,
                Some([(qargs, None)].into_iter().collect()),
            );
            assert!(res.is_ok())
        }

        let std_gate_names: Vec<&str> = get_standard_gate_names().to_vec();
        let operation_names: Vec<&str> = all_standard_target.operation_names().collect();

        assert_eq!(std_gate_names, operation_names)
    }

    #[test]
    fn test_update_inst_properties() {
        let mut test_target = Target::default();
        let qargs: Qargs = (0..2).map(PhysicalQubit).collect();
        // Add instruction with None as property
        let result = test_target.add_instruction(
            StandardGate::CX.into(),
            None,
            None,
            Some([(qargs.clone(), None)].into_iter().collect()),
        );
        assert!(result.is_ok(), "Error message: {result:?}");

        assert_eq!(test_target["cx"][&qargs], None);

        // Modify instruction property to a concrete value.
        let result = test_target.update_instruction_properties(
            "cx",
            &qargs,
            Some(InstructionProperties::new(Some(0.00122), Some(0.00001023))),
        );
        assert!(result.is_ok(), "Error message: {result:?}");

        assert_eq!(
            test_target["cx"][&qargs],
            Some(InstructionProperties::new(Some(0.00122), Some(0.00001023)))
        );

        // Modify instruction property back to None.
        let result = test_target.update_instruction_properties("cx", &qargs, None);
        assert!(result.is_ok(), "Error message: {result:?}");
        assert_eq!(test_target["cx"][&qargs], None);
    }

    #[test]
    fn test_update_inst_properties_invalid_inst() {
        let mut test_target = Target::default();
        let qargs: SmallVec<[PhysicalQubit; 2]> = (0..2).map(PhysicalQubit).collect();
        // Add instruction with None as property
        let result = test_target.add_instruction(
            StandardGate::CX.into(),
            None,
            None,
            Some([(qargs.clone().into(), None)].into_iter().collect()),
        );
        assert!(result.is_ok(), "Error message: {result:?}");

        assert_eq!(test_target["cx"][&QargsRef::from(&qargs)], None);

        // Try to update instruction property that is not present in the circuit.
        let result = test_target.update_instruction_properties(
            "cy",
            &qargs,
            Some(InstructionProperties::new(Some(0.00122), Some(0.00001023))),
        );
        // Check error message.
        let Err(res) = result else {
            panic!("The operation did not fail as expected.");
        };
        let expected_message = "Provided instruction: 'cy' not in this Target.".to_string();
        assert_eq!(res.to_string(), expected_message);
        // Check that no changes were made.
        assert_eq!(test_target["cx"][&QargsRef::from(&qargs)], None);

        let reverse_qargs: SmallVec<[PhysicalQubit; 2]> = qargs.iter().rev().copied().collect();
        // Try to update instruction property with qargs that are not present in the circuit.
        let result = test_target.update_instruction_properties(
            "cx",
            &reverse_qargs,
            Some(InstructionProperties::new(Some(0.00122), Some(0.00001023))),
        );
        // Check error message.
        let Err(res) = result else {
            panic!("The operation did not fail as expected.");
        };
        let expected_message = format!(
            "Provided qarg {:?} not in this Target for '{}'.",
            QargsRef::from(&reverse_qargs),
            "cx"
        );
        assert_eq!(res.to_string(), expected_message);
        // Check that no changes were made.
        assert_eq!(test_target["cx"][&QargsRef::from(&qargs)], None);
    }

    #[test]
    fn test_set_and_get_qubit_properties() {
        use super::QubitProperties;
        let props = vec![
            QubitProperties {
                t1: Some(10.0),
                t2: Some(20.0),
                frequency: Some(5.0),
            },
            QubitProperties {
                t1: Some(11.0),
                t2: Some(21.0),
                frequency: Some(6.0),
            },
        ];
        let target = Target {
            qubit_properties: Some(props.clone()),
            num_qubits: Some(2),
            ..Default::default()
        };
        assert_eq!(target.qubit_properties.as_ref().unwrap().len(), 2);
        assert_eq!(target.qubit_properties.as_ref().unwrap()[0].t1, Some(10.0));
        assert_eq!(
            target.qubit_properties.as_ref().unwrap()[1].frequency,
            Some(6.0)
        );
    }

    #[test]
    fn test_qubit_properties_num_qubits_mismatch() {
        use super::QubitProperties;
        let props = vec![QubitProperties {
            t1: Some(10.0),
            t2: Some(20.0),
            frequency: Some(5.0),
        }];
        // num_qubits is 2, but only 1 qubit_properties
        let result = Target::new(
            None,
            Some(2),
            None,
            Some(1),
            Some(1),
            Some(1),
            Some(1),
            Some(props),
            None,
        );
        assert!(result.is_err());
    }
}
