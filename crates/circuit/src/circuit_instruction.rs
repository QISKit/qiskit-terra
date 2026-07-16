// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyDeprecationWarning, PyTypeError, PyValueError};
use pyo3::prelude::*;

use pyo3::IntoPyObjectExt;
use pyo3::types::{PyBool, PyList, PyTuple, PyType};
use pyo3::{PyResult, intern};

use crate::circuit_data::{CircuitData, PyCircuitData};
use crate::dag_circuit::DAGCircuit;
use crate::duration::Duration;
use crate::imports::{CONTROLLED_GATE, WARNINGS_WARN};
use crate::instruction::{Instruction, Parameters, create_py_op};
use crate::operations::{
    ArrayType, BoxDuration, ControlFlow, ControlFlowInstruction, ControlFlowType, Operation,
    OperationRef, Param, PauliBased, PauliProductMeasurement, PauliProductRotation, PyInstruction,
    PyOpKind, StandardGate, StandardInstruction, StandardInstructionType, UnitaryGate,
};
use crate::packed_instruction::PackedOperation;
use crate::parameter::parameter_expression::ParameterExpression;
use nalgebra::{Dyn, MatrixView2, MatrixView4};
use num_complex::Complex64;
use smallvec::{SmallVec, smallvec};

/// A single instruction in a :class:`.QuantumCircuit`, comprised of the :attr:`operation` and
/// various operands.
///
/// .. note::
///
///     There is some possible confusion in the names of this class, :class:`~.circuit.Instruction`,
///     and :class:`~.circuit.Operation`, and this class's attribute :attr:`operation`.  Our
///     preferred terminology is by analogy to assembly languages, where an "instruction" is made up
///     of an "operation" and its "operands".
///
///     Historically, :class:`~.circuit.Instruction` came first, and originally contained the qubits
///     it operated on and any parameters, so it was a true "instruction".  Over time,
///     :class:`.QuantumCircuit` became responsible for tracking qubits and clbits, and the class
///     became better described as an "operation".  Changing the name of such a core object would be
///     a very unpleasant API break for users, and so we have stuck with it.
///
///     This class was created to provide a formal "instruction" context object in
///     :class:`.QuantumCircuit.data`, which had long been made of ad-hoc tuples.  With this, and
///     the advent of the :class:`~.circuit.Operation` interface for adding more complex objects to
///     circuits, we took the opportunity to correct the historical naming.  For the time being,
///     this leads to an awkward case where :attr:`.CircuitInstruction.operation` is often an
///     :class:`~.circuit.Instruction` instance (:class:`~.circuit.Instruction` implements the
///     :class:`.Operation` interface), but as the :class:`.Operation` interface gains more use,
///     this confusion will hopefully abate.
///
/// .. warning::
///
///     This is a lightweight internal class and there is minimal error checking; you must respect
///     the type hints when using it.  It is the user's responsibility to ensure that direct
///     mutations of the object do not invalidate the types, nor the restrictions placed on it by
///     its context.  Typically this will mean, for example, that :attr:`qubits` must be a sequence
///     of distinct items, with no duplicates.
#[pyclass(
    freelist = 20,
    sequence,
    module = "qiskit._accelerate.circuit",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct CircuitInstruction {
    pub operation: PackedOperation,
    /// A sequence of the qubits that the operation is applied to.
    #[pyo3(get)]
    pub qubits: Py<PyTuple>,
    /// A sequence of the classical bits that this operation reads from or writes to.
    #[pyo3(get)]
    pub clbits: Py<PyTuple>,
    pub params: Option<Parameters<CircuitData>>,
    pub label: Option<Box<String>>,
    #[cfg(feature = "cache_pygates")]
    pub py_op: OnceLock<Py<PyAny>>,
}

impl CircuitInstruction {
    /// Get the Python-space operation, ensuring that it is mutable from Python space (singleton
    /// gates might not necessarily satisfy this otherwise).
    ///
    /// This returns the cached instruction if valid, but does not replace the cache if it created a
    /// new mutable object; the expectation is that any mutations to the Python object need
    /// assigning back to the `CircuitInstruction` completely to ensure data coherence between Rust
    /// and Python spaces.  We can't protect entirely against that, but we can make it a bit harder
    /// for standard-gate getters to accidentally do the wrong thing.
    pub fn get_operation_mut<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = self.get_operation(py)?.into_bound(py);
        if out.getattr(intern!(py, "mutable"))?.is_truthy()? {
            Ok(out)
        } else {
            out.call_method0(intern!(py, "to_mutable"))
        }
    }
}

impl Instruction for CircuitInstruction {
    type Block = CircuitData;

    fn op(&self) -> OperationRef<'_> {
        self.operation.view()
    }

    fn parameters(&self) -> Option<&Parameters<CircuitData>> {
        self.params.as_ref()
    }

    fn label(&self) -> Option<&str> {
        self.label()
    }
}

#[pymethods]
impl CircuitInstruction {
    #[new]
    #[pyo3(signature = (operation, qubits=None, clbits=None))]
    pub fn py_new(
        operation: &Bound<PyAny>,
        qubits: Option<Bound<PyAny>>,
        clbits: Option<Bound<PyAny>>,
    ) -> PyResult<Self> {
        let py = operation.py();
        let op_parts = operation.extract::<OperationFromPython<CircuitData>>()?;

        Ok(Self {
            operation: op_parts.operation,
            qubits: as_tuple(py, qubits)?.unbind(),
            clbits: as_tuple(py, clbits)?.unbind(),
            params: op_parts.params,
            label: op_parts.label,
            #[cfg(feature = "cache_pygates")]
            py_op: operation.clone().unbind().into(),
        })
    }

    #[pyo3(signature = (standard, qubits, params, label=None))]
    #[staticmethod]
    pub fn from_standard(
        py: Python,
        standard: StandardGate,
        qubits: Option<Bound<PyAny>>,
        params: SmallVec<[Param; 3]>,
        label: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            operation: standard.into(),
            qubits: as_tuple(py, qubits)?.unbind(),
            clbits: PyTuple::empty(py).unbind(),
            params: (!params.is_empty()).then(|| Parameters::Params(params)),
            label: label.map(Box::new),
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        })
    }

    /// Returns a shallow copy.
    ///
    /// Returns:
    ///     CircuitInstruction: The shallow copy.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// The logical operation that this instruction represents an execution of.
    #[getter]
    pub fn get_operation(&self, py: Python) -> PyResult<Py<PyAny>> {
        // This doesn't use `get_or_init` because a) the initialiser is fallible and
        // `get_or_try_init` isn't stable, and b) the initialiser can yield to the Python
        // interpreter, which might suspend the thread and allow another to inadvertently attempt to
        // re-enter the cache setter, which isn't safe.

        #[cfg(feature = "cache_pygates")]
        {
            if let Some(cached_op) = self.py_op.get() {
                return Ok(cached_op.clone_ref(py));
            }
        }

        let out = create_py_op(py, self.op(), self.parameters().cloned(), self.label())?;

        #[cfg(feature = "cache_pygates")]
        {
            self.py_op.get_or_init(|| out.clone_ref(py));
        }

        Ok(out)
    }

    /// Returns the Instruction name corresponding to the op for this node
    #[getter]
    fn get_name(&self) -> &str {
        self.operation.name()
    }

    #[getter]
    pub fn get_params(&self, py: Python) -> PyResult<Py<PyAny>> {
        if self.params.is_none() {
            return Ok(PyList::empty(py).into_any().unbind());
        };
        match self.operation.view() {
            OperationRef::ControlFlow(cf) => match &cf.control_flow {
                ControlFlow::ForLoop {
                    collection,
                    loop_param,
                    ..
                } => [
                    collection.into_py_any(py)?,
                    loop_param.clone().into_py_any(py)?,
                    self.blocks_view()[0]
                        .clone()
                        .into_py_quantum_circuit(py)?
                        .unbind(),
                ]
                .into_py_any(py),
                _ => self
                    .blocks_view()
                    .iter()
                    .map(|block| block.clone().into_py_quantum_circuit(py))
                    .collect::<PyResult<Vec<_>>>()?
                    .into_py_any(py),
            },
            _ => self.params_view().into_py_any(py),
        }
    }

    #[getter]
    fn matrix<'py>(&'py self, py: Python<'py>) -> Option<Bound<'py, PyArray2<Complex64>>> {
        let matrix = self.try_matrix();
        matrix.map(move |mat| mat.into_pyarray(py))
    }

    #[getter]
    fn label(&self) -> Option<&str> {
        self.label.as_ref().map(|label| label.as_str())
    }

    /// Is the :class:`.Operation` contained in this instruction a Qiskit standard gate?
    pub fn is_standard_gate(&self) -> bool {
        self.operation.try_standard_gate().is_some()
    }

    /// Is the :class:`.Operation` contained in this instruction a subclass of
    /// :class:`.ControlledGate`?
    pub fn is_controlled_gate(&self, py: Python) -> PyResult<bool> {
        match self.operation.view() {
            OperationRef::StandardGate(standard) => Ok(standard.num_ctrl_qubits() != 0),
            OperationRef::PyCustom(inst) => {
                inst.ob.bind(py).is_instance(CONTROLLED_GATE.get_bound(py))
            }
            _ => Ok(false),
        }
    }

    /// Is the :class:`.Operation` contained in this node a directive?
    pub fn is_directive(&self) -> bool {
        self.operation.directive()
    }

    /// Is the :class:`.Operation` contained in this instruction a control-flow operation (i.e. an
    /// instance of :class:`.ControlFlowOp`)?
    pub fn is_control_flow(&self) -> bool {
        self.operation
            .try_control_flow()
            .is_some_and(|control_flow| {
                !matches!(
                    control_flow.control_flow,
                    ControlFlow::BreakLoop | ControlFlow::ContinueLoop
                )
            })
    }

    /// Does this instruction contain any :class:`.ParameterExpression` parameters?
    pub fn is_parameterized(&self) -> bool {
        let Some(params) = self.params.as_ref() else {
            return false;
        };
        match params {
            Parameters::Params(p) => p.iter().any(|x| matches!(x, Param::ParameterExpression(_))),
            Parameters::Blocks(_) => false,
        }
    }

    /// Creates a shallow copy with the given fields replaced.
    ///
    /// Returns:
    ///     CircuitInstruction: A new instance with the given fields replaced.
    #[pyo3(signature=(operation=None, qubits=None, clbits=None, params=None))]
    pub fn replace(
        &self,
        py: Python,
        operation: Option<&Bound<PyAny>>,
        qubits: Option<Bound<PyAny>>,
        clbits: Option<Bound<PyAny>>,
        params: Option<Bound<PyAny>>,
    ) -> PyResult<Self> {
        let qubits = match qubits {
            None => self.qubits.clone_ref(py),
            Some(qubits) => as_tuple(py, Some(qubits))?.unbind(),
        };
        let clbits = match clbits {
            None => self.clbits.clone_ref(py),
            Some(clbits) => as_tuple(py, Some(clbits))?.unbind(),
        };

        if let Some(operation) = operation {
            let op_parts = operation.extract::<OperationFromPython<CircuitData>>()?;
            let params = if let Some(params) = params {
                extract_params(op_parts.operation.view(), &params)?
            } else {
                op_parts.params
            };

            Ok(Self {
                operation: op_parts.operation,
                qubits,
                clbits,
                params,
                label: op_parts.label,
                #[cfg(feature = "cache_pygates")]
                py_op: operation.clone().unbind().into(),
            })
        } else {
            let params = if let Some(params) = params {
                extract_params(self.operation.view(), &params)?
            } else {
                self.params.clone()
            };
            Ok(Self {
                operation: self.operation.clone(),
                qubits,
                clbits,
                params,
                label: self.label.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: self.py_op.clone(),
            })
        }
    }

    pub fn __getnewargs__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        (
            self.get_operation(py)?,
            self.qubits.bind(py),
            self.clbits.bind(py),
        )
            .into_py_any(py)
    }

    pub fn __repr__(self_: &Bound<Self>, py: Python<'_>) -> PyResult<String> {
        let type_name = self_.get_type().qualname()?;
        let r = self_.try_borrow()?;
        Ok(format!(
            "{}(operation={}, qubits={}, clbits={})",
            type_name,
            r.get_operation(py)?.bind(py).repr()?,
            r.qubits.bind(py).repr()?,
            r.clbits.bind(py).repr()?
        ))
    }

    // Legacy tuple-like interface support.
    //
    // For a best attempt at API compatibility during the transition to using this new class, we need
    // the interface to behave exactly like the old 3-tuple `(inst, qargs, cargs)` if it's treated
    // like that via unpacking or similar.  That means that the `parameters` field is completely
    // absent, and the qubits and clbits must be converted to lists.
    pub fn _legacy_format<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                self.get_operation(py)?,
                self.qubits.bind(py).to_list().into(),
                self.clbits.bind(py).to_list().into(),
            ],
        )
    }

    pub fn __getitem__(&self, py: Python<'_>, key: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        self._legacy_format(py)?
            .as_any()
            .get_item(key)?
            .into_py_any(py)
    }

    pub fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        self._legacy_format(py)?
            .as_any()
            .try_iter()?
            .into_py_any(py)
    }

    pub fn __len__(&self, py: Python) -> PyResult<usize> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        Ok(3)
    }

    pub fn __richcmp__(
        self_: &Bound<Self>,
        other: &Bound<PyAny>,
        op: CompareOp,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        fn params_eq(
            py: Python,
            left: Option<&Parameters<CircuitData>>,
            right: Option<&Parameters<CircuitData>>,
        ) -> PyResult<bool> {
            if left.is_none() && right.is_none() {
                return Ok(true);
            }
            let (Some(left), Some(right)) = (left, right) else {
                return Ok(false);
            };

            match (left, right) {
                (Parameters::Params(left), Parameters::Params(right)) => {
                    if left.len() != right.len() {
                        return Ok(false);
                    }
                    for (left, right) in left.iter().zip(right) {
                        let eq = match left {
                            Param::Float(left) => match right {
                                Param::Float(right) => left == right,
                                Param::ParameterExpression(right) => {
                                    &ParameterExpression::from_f64(*left) == right.as_ref()
                                }
                                Param::Obj(right) => right.bind(py).eq(left)?,
                            },
                            Param::ParameterExpression(left) => match right {
                                Param::Float(right) => {
                                    left.as_ref() == &ParameterExpression::from_f64(*right)
                                }
                                Param::ParameterExpression(right) => left == right,
                                Param::Obj(right) => right.bind(py).eq(left.as_ref().clone())?,
                            },
                            Param::Obj(left) => left.bind(py).eq(right)?,
                        };
                        if !eq {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                (Parameters::Blocks(blocks_a), Parameters::Blocks(blocks_b)) => {
                    if blocks_a.len() != blocks_b.len() {
                        return Ok(false);
                    }
                    // TODO: we should be able to do the semantic-equality comparison from Rust
                    // space in the future, without going via Python.  See gh-15267.
                    for (a, b) in blocks_a.iter().zip(blocks_b) {
                        if !a
                            .clone()
                            .into_py_quantum_circuit(py)?
                            .eq(b.clone().into_py_quantum_circuit(py)?)?
                        {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                _ => Ok(false),
            }
        }

        fn eq(
            py: Python<'_>,
            self_: &Bound<CircuitInstruction>,
            other: &Bound<PyAny>,
        ) -> PyResult<Option<bool>> {
            if self_.is(other) {
                return Ok(Some(true));
            }

            let self_ = self_.try_borrow()?;

            if other.is_instance_of::<PyTuple>() {
                return Ok(Some(self_._legacy_format(py)?.eq(other)?));
            }
            let Ok(other) = other.cast::<CircuitInstruction>() else {
                return Ok(None);
            };
            let other = other.try_borrow()?;

            Ok(Some(
                self_.qubits.bind(py).eq(other.qubits.bind(py))?
                    && self_.clbits.bind(py).eq(other.clbits.bind(py))?
                    && self_.operation.py_eq(py, &other.operation)?
                    && (self_.operation.try_standard_gate().is_none()
                        || params_eq(py, self_.params.as_ref(), other.params.as_ref())?),
            ))
        }

        Ok(match op {
            CompareOp::Eq => match eq(py, self_, other)? {
                Some(res) => PyBool::new(py, res).to_owned().into_any().unbind(),
                None => py.NotImplemented(),
            },
            CompareOp::Ne => match eq(py, self_, other)? {
                Some(res) => PyBool::new(py, !res).to_owned().into_any().unbind(),
                None => py.NotImplemented(),
            },
            _ => py.NotImplemented(),
        })
    }
}

/// A container struct that contains the conversion from some `Operation` subclass input, on its way
/// to becoming a `PackedInstruction`.
///
/// This is the primary way of converting an incoming `Gate` / `Instruction` / `Operation` from
/// Python space into Rust-space data.  A typical access pattern is:
///
/// ```rust
/// #[pyfunction]
/// fn accepts_op_from_python(ob: &Bound<PyAny>) -> PyResult<()> {
///     let py_op = ob.extract::<OperationFromPython<CircuitData>>()?;
///     // ... use `py_op.operation`, `py_op.params`, etc.
///     Ok(())
/// }
/// ```
///
/// though you can also accept `ob: OperationFromPython` directly, if you don't also need a handle
/// to the Python object that it came from.  The handle is useful for the Python-operation caching.
///
/// The generic argument controls how control-flow blocks are extracted.  Typically you choose this
/// to match the place you'll be putting the blocks.  If you want to fail the extraction if there's
/// control-flow blocks, use `qiskit_circuit::NoBlocks`.  If you want to leave the blocks
/// unextracted, use `Py<PyAny>`.
#[derive(Debug)]
pub struct OperationFromPython<T> {
    pub operation: PackedOperation,
    pub params: Option<Parameters<T>>,
    pub label: Option<Box<String>>,
}

impl<T: CircuitBlock> OperationFromPython<T> {
    /// Takes the params out of [OperationFromPython::params].
    ///
    /// Panics if params is not a parameter list.
    pub fn take_params(&mut self) -> Option<SmallVec<[Param; 3]>> {
        self.params.take().map(|p| p.unwrap_params())
    }

    /// Takes the blocks out of [OperationFromPython::params].
    ///
    /// Panics if params is not a block list.
    pub fn take_blocks(&mut self) -> Option<Vec<T>> {
        self.params.take().map(|p| p.unwrap_blocks())
    }
}

/// Marker object for use with `OperationFromPython` that marks that the extraction should fail if
/// the object contains circuit blocks.
pub struct NoBlocks;

/// Helper trait implemented by `PyCircuitData` and `DAGCircuit` to implement extraction from
/// a Python-owned control-flow block into a suitable Rust type.
///
/// This shouldn't need to be imported anywhere nor implemented by anything else; it's only intended
/// to let the `OperationFromPython` extraction be generic.
pub trait CircuitBlock: Sized {
    fn extract_py_block(ob: Bound<PyCircuitData>) -> PyResult<Self>;
}
impl CircuitBlock for PyCircuitData {
    fn extract_py_block(ob: Bound<PyCircuitData>) -> PyResult<Self> {
        Ok(ob.borrow().clone())
    }
}
// TODO: in the long run we don't need to extract from python directly to CircuitData
// But for now it's needed in assing_parameters_inner
impl CircuitBlock for CircuitData {
    fn extract_py_block(ob: Bound<PyCircuitData>) -> PyResult<Self> {
        Ok(ob.borrow().clone().inner)
    }
}
impl CircuitBlock for DAGCircuit {
    fn extract_py_block(ob: Bound<PyCircuitData>) -> PyResult<Self> {
        Self::from_circuit_data(&ob.borrow().inner, false, None, None, None, None)
            .map_err(Into::into)
    }
}
impl CircuitBlock for NoBlocks {
    fn extract_py_block(_ob: Bound<PyCircuitData>) -> PyResult<Self> {
        Err(PyTypeError::new_err("control-flow ops are not valid here"))
    }
}
impl CircuitBlock for Py<PyAny> {
    fn extract_py_block(ob: Bound<PyCircuitData>) -> PyResult<Self> {
        Ok(ob.into_any().unbind())
    }
}

impl<T> Instruction for OperationFromPython<T> {
    type Block = T;

    fn op(&self) -> OperationRef<'_> {
        self.operation.view()
    }

    fn parameters(&self) -> Option<&Parameters<T>> {
        self.params.as_ref()
    }

    fn label(&self) -> Option<&str> {
        self.label.as_ref().map(|label| label.as_str())
    }
}

impl<'a, 'py, T: CircuitBlock> FromPyObject<'a, 'py> for OperationFromPython<T> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let py = ob.py();
        let ob_type = ob
            .getattr(intern!(py, "base_class"))
            .ok()
            .map(|base| base.cast_into::<PyType>())
            .transpose()?
            .unwrap_or_else(|| ob.get_type());

        let get_params = || -> PyResult<Bound<PyAny>> {
            Ok(ob
                .getattr_opt(intern!(py, "params"))?
                .unwrap_or_else(|| PyTuple::empty(py).into_any()))
        };
        let extract_label = || -> PyResult<Option<Box<String>>> {
            let raw = ob.getattr(intern!(py, "label"))?;
            Ok(raw.extract::<Option<String>>()?.map(Box::new))
        };

        'standard_gate: {
            // Our Python standard gates have a `_standard_gate` field at the class level so we can
            // quickly identify them here without an `isinstance` check.
            let Some(standard) = ob_type
                .getattr(intern!(py, "_standard_gate"))
                .ok()
                .and_then(|standard| standard.extract::<StandardGate>().ok())
            else {
                break 'standard_gate;
            };

            // If the instruction is a controlled gate with a not-all-ones control state, it doesn't
            // fit our definition of standard.  We abuse the fact that we know our standard-gate
            // mapping to avoid an `isinstance` check on `ControlledGate` - a standard gate has
            // nonzero `num_ctrl_qubits` iff it is a `ControlledGate`.
            //
            // `ControlledGate` also has a `base_gate` attribute related to its historical
            // implementation, which technically allows mutations from Python space.  The only
            // mutation of a standard gate's `base_gate` that wouldn't have already broken the
            // Python-space data model is setting a label, so we just catch that case and default
            // back to non-standard-gate handling in that case.
            if standard.num_ctrl_qubits() != 0
                && ((ob.getattr(intern!(py, "ctrl_state"))?.extract::<usize>()?
                    != (1 << standard.num_ctrl_qubits()) - 1)
                    || !ob
                        .getattr(intern!(py, "base_gate"))?
                        .getattr(intern!(py, "label"))?
                        .is_none())
            {
                break 'standard_gate;
            }
            let operation = PackedOperation::from_standard_gate(standard);
            let params = extract_params(operation.view(), &get_params()?)?;
            return Ok(OperationFromPython {
                operation,
                params,
                label: extract_label()?,
            });
        }
        'standard_instr: {
            // Our Python standard instructions have a `_standard_instruction_type` field at the
            // class level so we can quickly identify them here without an `isinstance` check.
            // Once we know the type, we query the object for any type-specific fields we need to
            // read (e.g. a Barrier's number of qubits) to build the Rust representation.
            let Some(standard_type) = ob_type
                .getattr(intern!(py, "_standard_instruction_type"))
                .ok()
                .and_then(|standard| standard.extract::<StandardInstructionType>().ok())
            else {
                break 'standard_instr;
            };
            let standard = match standard_type {
                StandardInstructionType::Barrier => {
                    let num_qubits = ob.getattr(intern!(py, "num_qubits"))?.extract()?;
                    StandardInstruction::Barrier(num_qubits)
                }
                StandardInstructionType::Delay => {
                    let unit = ob.getattr(intern!(py, "unit"))?.extract()?;
                    StandardInstruction::Delay(unit)
                }
                StandardInstructionType::Measure => StandardInstruction::Measure,
                StandardInstructionType::Reset => StandardInstruction::Reset,
            };
            let operation = PackedOperation::from_standard_instruction(standard);
            let params = extract_params(operation.view(), &get_params()?)?;
            return Ok(OperationFromPython {
                operation,
                params,
                label: extract_label()?,
            });
        }
        'control_flow: {
            // Our Python control flow instructions have a `_control_flow_type` field at the
            // class level so we can quickly identify them here without an `isinstance` check.
            // Once we know the type, we query the object for any type-specific fields we need to
            // read to build the Rust representation.
            let Some(control_flow_type) = ob_type
                .getattr(intern!(py, "_control_flow_type"))
                .ok()
                .and_then(|cf| cf.extract::<ControlFlowType>().ok())
            else {
                break 'control_flow;
            };
            let params = get_params()?;
            let control_flow = ControlFlowInstruction {
                control_flow: match control_flow_type {
                    ControlFlowType::Box => {
                        let py_duration: Option<Bound<PyAny>> =
                            ob.getattr(intern!(py, "duration"))?.extract()?;
                        let unit: Option<String> = ob.getattr(intern!(py, "unit"))?.extract()?;
                        let duration = if let Some(py_duration) = py_duration {
                            Some(match unit.as_deref().unwrap_or("dt") {
                                "dt" => BoxDuration::Duration(Duration::dt(
                                    py_duration.extract::<f64>()? as i64,
                                )),
                                "s" => BoxDuration::Duration(Duration::s(py_duration.extract()?)),
                                "ms" => BoxDuration::Duration(Duration::ms(py_duration.extract()?)),
                                "us" => BoxDuration::Duration(Duration::us(py_duration.extract()?)),
                                "ns" => BoxDuration::Duration(Duration::ns(py_duration.extract()?)),
                                "ps" => BoxDuration::Duration(Duration::ps(py_duration.extract()?)),
                                "expr" => BoxDuration::Expr(py_duration.extract()?),
                                _ => {
                                    return Err(PyValueError::new_err(format!(
                                        "duration unit '{}' is unsupported",
                                        unit.unwrap()
                                    )));
                                }
                            })
                        } else {
                            None
                        };
                        let annotations = ob.getattr(intern!(py, "annotations"))?.extract()?;
                        ControlFlow::Box {
                            duration,
                            annotations,
                        }
                    }
                    ControlFlowType::BreakLoop => ControlFlow::BreakLoop,
                    ControlFlowType::ContinueLoop => ControlFlow::ContinueLoop,
                    ControlFlowType::ForLoop => {
                        // We lift for-loop's collection and loop parameter from `params` to the
                        // operation itself for Rust since it's nicer to work with.
                        let mut params = params.try_iter()?;
                        ControlFlow::ForLoop {
                            collection: params
                                .next()
                                .ok_or_else(|| {
                                    PyValueError::new_err(
                                        "not enough values to unpack (expected 3, got 0)",
                                    )
                                })??
                                .extract()?,
                            loop_param: params
                                .next()
                                .ok_or_else(|| {
                                    PyValueError::new_err(
                                        "not enough values to unpack (expected 3, got 1)",
                                    )
                                })??
                                .extract()?,
                        }
                    }
                    ControlFlowType::IfElse => ControlFlow::IfElse {
                        condition: ob.getattr(intern!(py, "condition"))?.extract()?,
                    },
                    ControlFlowType::SwitchCase => ControlFlow::Switch {
                        target: ob.getattr(intern!(py, "target"))?.extract()?,
                        label_spec: ob.getattr(intern!(py, "_label_spec"))?.extract()?,
                        cases: params.len()? as u32,
                    },
                    ControlFlowType::WhileLoop => ControlFlow::While {
                        condition: ob.getattr(intern!(py, "condition"))?.extract()?,
                    },
                },
                num_qubits: ob.getattr("num_qubits")?.extract()?,
                num_clbits: ob.getattr("num_clbits")?.extract()?,
            };
            let operation = PackedOperation::from_control_flow(control_flow.into());
            let params = extract_params(operation.view(), &params)?;
            return Ok(OperationFromPython {
                operation,
                params,
                label: extract_label()?,
            });
        }

        // We need to check by name here to avoid a circular import during initial loading
        let ob_name = ob.getattr(intern!(py, "name"))?.extract::<String>()?;
        if ob_name == "unitary" {
            let params: SmallVec<[Param; 3]> = get_params()?.extract()?;
            if let Some(Param::Obj(data)) = params.first() {
                let py_matrix: PyReadonlyArray2<Complex64> = data.extract(py)?;
                let matrix: Option<MatrixView2<Complex64, Dyn, Dyn>> = py_matrix.try_as_matrix();
                if let Some(x) = matrix {
                    let unitary_gate = Box::new(UnitaryGate {
                        array: ArrayType::OneQ(x.into_owned()),
                    });
                    return Ok(OperationFromPython {
                        operation: PackedOperation::from_unitary(unitary_gate),
                        params: None,
                        label: extract_label()?,
                    });
                }
                let matrix: Option<MatrixView4<Complex64, Dyn, Dyn>> = py_matrix.try_as_matrix();
                if let Some(x) = matrix {
                    let unitary_gate = Box::new(UnitaryGate {
                        array: ArrayType::TwoQ(x.into_owned()),
                    });
                    return Ok(OperationFromPython {
                        operation: PackedOperation::from_unitary(unitary_gate),
                        params: None,
                        label: extract_label()?,
                    });
                } else {
                    let unitary_gate = Box::new(UnitaryGate {
                        array: ArrayType::NDArray(py_matrix.as_array().to_owned()),
                    });
                    return Ok(OperationFromPython {
                        operation: PackedOperation::from_unitary(unitary_gate),
                        params: None,
                        label: extract_label()?,
                    });
                };
            }
        } else if ob_name == "pauli_product_measurement" {
            let z = ob
                .getattr(intern!(py, "_pauli_z"))?
                .extract::<PyReadonlyArray1<bool>>()?
                .as_slice()?
                .to_vec();

            let x = ob
                .getattr(intern!(py, "_pauli_x"))?
                .extract::<PyReadonlyArray1<bool>>()?
                .as_slice()?
                .to_vec();

            let phase = ob.getattr(intern!(py, "_pauli_phase"))?.extract::<u8>()?;

            let pauli_product_measurement = PauliProductMeasurement {
                z: z.to_owned(),
                x: x.to_owned(),
                neg: phase == 2, // phase is only 0 (represents 1) or 2 (represents -1)
            };
            let pbc = Box::new(PauliBased::PauliProductMeasurement(
                pauli_product_measurement,
            ));

            return Ok(OperationFromPython {
                operation: PackedOperation::from_pauli_based(pbc),
                params: None,
                label: extract_label()?,
            });
        } else if ob_name == "pauli_product_rotation" {
            let z = ob
                .getattr(intern!(py, "_pauli_z"))?
                .extract::<PyReadonlyArray1<bool>>()?
                .as_slice()?
                .to_vec();

            let x = ob
                .getattr(intern!(py, "_pauli_x"))?
                .extract::<PyReadonlyArray1<bool>>()?
                .as_slice()?
                .to_vec();

            let py_angle = get_params()?.get_item(0)?;
            let angle = Param::extract(py_angle.as_borrowed())?;
            if matches!(angle, Param::Obj(_)) {
                return Err(PyTypeError::new_err(
                    "invalid type for angle in PauliProductRotation",
                ));
            }

            let pauli_rotation = PauliProductRotation {
                z: z.to_owned(),
                x: x.to_owned(),
                angle: angle.clone(),
            };
            let pbc = Box::new(PauliBased::PauliProductRotation(pauli_rotation));

            return Ok(OperationFromPython {
                operation: PackedOperation::from_pauli_based(pbc),
                params: Some(Parameters::Params(smallvec![angle])),
                label: extract_label()?,
            });
        }

        let Some(kind) = PyOpKind::from_type(ob_type.as_borrowed())? else {
            return Err(PyTypeError::new_err(format!(
                "invalid input: {}",
                ob.to_owned()
            )));
        };
        let params = get_params()?;
        let operation = PackedOperation::from(PyInstruction {
            kind,
            qubits: ob.getattr(intern!(py, "num_qubits"))?.extract()?,
            clbits: ob.getattr(intern!(py, "num_clbits"))?.extract()?,
            params: params.len()? as u32,
            op_name: ob.getattr(intern!(py, "name"))?.extract()?,
            ob: ob.to_owned().unbind(),
        });
        let params = extract_params(operation.view(), &params)?;
        Ok(OperationFromPython {
            operation,
            params,
            label: match kind {
                PyOpKind::Gate | PyOpKind::Instruction => extract_label()?,
                PyOpKind::Operation => None,
            },
        })
    }
}

/// Extracts a Python-space params list into an optional [Parameters] list, given
/// the corresponding operation reference.
pub fn extract_params<T: CircuitBlock>(
    op: OperationRef,
    params: &Bound<PyAny>,
) -> PyResult<Option<Parameters<T>>> {
    let data_attr = intern!(params.py(), "_data");
    Ok(match op {
        OperationRef::ControlFlow(cf) => match &cf.control_flow {
            ControlFlow::BreakLoop => None,
            ControlFlow::ContinueLoop => None,
            ControlFlow::ForLoop { .. } => {
                // We skip the first two parameters (collection and loop_param) since we
                // store those directly on the operation in Rust.
                let mut params = params.try_iter()?.skip(2);
                Some(Parameters::Blocks(vec![T::extract_py_block(
                    params
                        .next()
                        .ok_or_else(|| {
                            PyValueError::new_err("not enough values to unpack (expected 3)")
                        })??
                        .getattr(data_attr)?
                        .cast_into()?,
                )?]))
            }
            _ => {
                // For all other control flow operations with blocks, the 'params' in Python land
                // are exactly the blocks.
                let blocks = params
                    .try_iter()?
                    .take_while(|p| match p {
                        // In the case of IfElse, the "false" body might be None.
                        Ok(block) if !block.is_none() => true,
                        _ => false,
                    })
                    .map(|p| T::extract_py_block(p?.getattr(data_attr)?.cast_into()?))
                    .collect::<PyResult<_>>()?;
                Some(Parameters::Blocks(blocks))
            }
        },
        OperationRef::StandardGate(_) => {
            let params: SmallVec<[Param; 3]> = params.extract()?;
            (!params.is_empty()).then(|| Parameters::Params(params))
        }
        OperationRef::StandardInstruction(i) => {
            match &i {
                StandardInstruction::Barrier(_) => None,
                StandardInstruction::Delay(_) => {
                    // If the delay's duration is a Python int, we preserve it rather than
                    // coercing it to a float (e.g. when unit is 'dt').
                    Some(Parameters::Params(
                        params
                            .try_iter()?
                            .map(|p| Param::extract_no_coerce(p?.as_borrowed()))
                            .collect::<PyResult<_>>()?,
                    ))
                }
                StandardInstruction::Measure => None,
                StandardInstruction::Reset => None,
            }
        }
        OperationRef::Unitary(_) | OperationRef::PauliProductMeasurement(_) => None,
        OperationRef::PyCustom(_) | OperationRef::CustomOperation(_) => {
            let params: SmallVec<[Param; 3]> = params.extract()?;
            (!params.is_empty()).then(|| Parameters::Params(params))
        }
        OperationRef::PauliProductRotation(_) => {
            let params: SmallVec<[Param; 3]> = params.extract()?;
            Some(Parameters::Params(params))
        }
    })
}

/// Convert a sequence-like Python object to a tuple.
fn as_tuple<'py>(py: Python<'py>, seq: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyTuple>> {
    let Some(seq) = seq else {
        return Ok(PyTuple::empty(py));
    };
    if seq.is_instance_of::<PyTuple>() {
        Ok(seq.cast_into_exact::<PyTuple>()?)
    } else if seq.is_instance_of::<PyList>() {
        Ok(seq.cast_exact::<PyList>()?.to_tuple())
    } else {
        // New tuple from iterable.
        PyTuple::new(
            py,
            seq.try_iter()?
                .map(|o| Ok(o?.unbind()))
                .collect::<PyResult<Vec<Py<PyAny>>>>()?,
        )
    }
}

/// Issue a Python `DeprecationWarning` about using the legacy tuple-like interface to
/// `CircuitInstruction`.
///
/// Beware the `stacklevel` here doesn't work quite the same way as it does in Python as Rust-space
/// calls are completely transparent to Python.
#[inline]
fn warn_on_legacy_circuit_instruction_iteration(py: Python) -> PyResult<()> {
    WARNINGS_WARN
        .get_bound(py)
        .call1((
            intern!(
                py,
                concat!(
                    "Treating CircuitInstruction as an iterable is deprecated legacy behavior",
                    " since Qiskit 1.2, and will be removed in Qiskit 3.0.",
                    " Instead, use the `operation`, `qubits` and `clbits` named attributes."
                )
            ),
            py.get_type::<PyDeprecationWarning>(),
            // Stack level.  Compared to Python-space calls to `warn`, this is unusually low
            // because all our internal call structure is now Rust-space and invisible to Python.
            1,
        ))
        .map(|_| ())
}
