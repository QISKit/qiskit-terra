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

use approx::relative_eq;
use qiskit_quantum_info::sparse_pauli_op::MatrixCompressedPaulis;
use std::any::Any;
use std::fmt::Debug;
use std::num::NonZero;
use std::ops::{Deref, DerefMut};
use std::str::FromStr;
use std::sync::Arc;
use std::{fmt, vec};

use crate::bit::{ClassicalRegister, ShareableClbit};
use crate::circuit_data::{CircuitData, PyCircuitData};
use crate::classical::expr;
use crate::classical::expr::Var;
use crate::converters::QuantumCircuitData;
use crate::duration::Duration;
use crate::operations::custom_traits::{ClonableOp, ComparableOp};
use crate::packed_instruction::{PackedInstruction, PackedOperation};
use crate::parameter::parameter_expression::{
    ParameterExpression, PyParameter, PyParameterExpression,
};
use crate::parameter::symbol_expr::{Symbol, Value};
use crate::{ControlFlowBlocks, imports};

use nalgebra::{Matrix2, Matrix4};
use ndarray::{Array1, Array2, ArrayView2, Dim, ShapeBuilder, array};
use num_bigint::BigUint;
use num_complex::{Complex64, c64};
use smallvec::SmallVec;

use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyFloat, PyTuple, PyType};
use pyo3::{IntoPyObjectExt, Python, intern};

// This is a convenience re-export, since basically everywhere in Qiskit expects all the
// `StandardGate` definitions to be in this file.
pub use crate::standard_gate::*;

#[derive(Clone, Debug)]
pub enum Param {
    ParameterExpression(Arc<ParameterExpression>),
    Float(f64),
    Obj(Py<PyAny>),
}

impl<'py> IntoPyObject<'py> for &Param {
    type Target = PyAny; // target type is PyAny to cover f64, Py<PyAny> and PyParameterExpression
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Param::Float(value) => value.into_bound_py_any(py),
            Param::Obj(py_obj) => py_obj.into_bound_py_any(py),
            Param::ParameterExpression(expr) => {
                let py_expr = PyParameterExpression::from(expr.as_ref().clone());
                py_expr.coerce_into_py(py)?.into_bound_py_any(py)
            }
        }
    }
}

impl<'py> IntoPyObject<'py> for Param {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (&self).into_pyobject(py)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Param {
    type Error = ::std::convert::Infallible;

    fn extract(b: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(if let Ok(py_expr) = b.extract::<PyParameterExpression>() {
            Param::ParameterExpression(Arc::new(py_expr.inner))
        } else if b.is_instance_of::<PyArray1<i32>>() {
            Param::Obj(b.to_owned().unbind())
        } else if let Ok(val) = b.extract::<f64>() {
            // TODO: remove this branch when we raise the NumPy version to 2.4.
            Param::Float(val)
        } else {
            Param::Obj(b.to_owned().unbind())
        })
    }
}

impl Param {
    /// Get the float value, if one is stored.
    pub fn try_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }
    pub fn eq(&self, other: &Param) -> PyResult<bool> {
        match [self, other] {
            [Self::Float(a), Self::Float(b)] => Ok(a == b),
            [Self::Float(a), Self::ParameterExpression(b)] => {
                Ok(&ParameterExpression::from_f64(*a) == b.as_ref())
            }
            [Self::ParameterExpression(a), Self::Float(b)] => {
                Ok(a.as_ref() == &ParameterExpression::from_f64(*b))
            }
            [Self::ParameterExpression(a), Self::ParameterExpression(b)] => Ok(a == b),
            [Self::Obj(a), Self::Obj(b)] => Python::attach(|py| a.bind(py).eq(b)),
            [Self::Obj(_), Self::Float(_)] => Ok(false),
            [Self::Float(_), Self::Obj(_)] => Ok(false),
            [Self::Obj(_a), Self::ParameterExpression(_b)] => Ok(false),
            [Self::ParameterExpression(_a), Self::Obj(_b)] => Ok(false),
        }
    }

    pub fn is_close(&self, other: &Param, max_relative: f64) -> PyResult<bool> {
        match [self, other] {
            [Self::Float(a), Self::Float(b)] => Ok(relative_eq!(a, b, max_relative = max_relative)),
            _ => self.eq(other),
        }
    }

    // TODO: Replace `Box<dyn Iterator>` with a custom iterator struct. The
    // Box<dyn Iterator> is an anti-pattern which shouldn't really be needed.
    /// Get an iterator over any `Symbol` instances tracked within this `Param`.
    pub fn iter_parameters(&self) -> PyResult<Box<dyn Iterator<Item = Symbol> + '_>> {
        match self {
            Param::Float(_) => Ok(Box::new(::std::iter::empty())),
            Param::ParameterExpression(expr) => Ok(Box::new(expr.iter_symbols().cloned())),
            Param::Obj(obj) => {
                Python::attach(|py| -> PyResult<Box<dyn Iterator<Item = Symbol>>> {
                    let parameters_attr = intern!(py, "parameters");
                    let obj = obj.bind(py);
                    if obj.is_instance(imports::QUANTUM_CIRCUIT.get_bound(py))? {
                        // Note: this code-path is only potentially used by custom user operations
                        let collected: Vec<Symbol> = obj
                            .getattr(parameters_attr)?
                            .try_iter()?
                            .map(|elem| {
                                Ok(Symbol::clone(&elem?.cast_into::<PyParameter>()?.borrow().0))
                            })
                            .collect::<PyResult<_>>()?;
                        Ok(Box::new(collected.into_iter()))
                    } else {
                        Ok(Box::new(::std::iter::empty()))
                    }
                })
            }
        }
    }

    /// Construct a [Param] from a [ParameterExpression]. Allows type coercion.
    ///
    /// # Arguments
    ///
    /// * expr - The expression to construct the [Param] from.
    /// * coerce_to_float - If `true`, coerce integers and complex (with 0 imaginary part) types to
    ///   [Param::Float]. If `false`, only float types are [Param::Float] and integers and
    ///   complex numbers are represented as [Param::ParameterExpression].
    ///
    /// # Returns
    ///
    /// - `Param` - The [Param] object.
    pub fn from_expr(expr: ParameterExpression, coerce_to_float: bool) -> PyResult<Self> {
        match expr.try_to_value(true) {
            Ok(value) => match value {
                Value::Int(i) => {
                    if coerce_to_float {
                        Ok(Self::Float(i as f64)) // coerce integer to float
                    } else {
                        // Int is not a param type and only comes from Python so dump it in
                        // there until we support DT unit delay from C
                        Python::attach(|py| Ok(Self::Obj(i.into_py_any(py)?)))
                    }
                }
                Value::Real(f) => Ok(Self::Float(f)),
                Value::Complex(c) => {
                    if coerce_to_float && value.is_real() {
                        Ok(Self::Float(c.re))
                    } else {
                        // Complex numbers are only defined in Python custom
                        // objects and aren't valid gate parameters for
                        // anything else so return it as an object
                        Python::attach(|py| Ok(Self::Obj(c.into_py_any(py)?)))
                    }
                }
            },
            Err(_) => Ok(Self::ParameterExpression(Arc::new(expr))),
        }
    }

    /// Extract from a Python object without numeric coercion to float.  The default conversion will
    /// coerce integers into floats, but in things like `assign_parameters`, this is not always
    /// desirable.
    pub fn extract_no_coerce(ob: Borrowed<PyAny>) -> PyResult<Self> {
        Ok(if ob.is_instance_of::<PyFloat>() {
            Param::Float(ob.extract()?)
        } else if let Ok(py_expr) = PyParameterExpression::extract_coerce(ob) {
            // don't get confused by the `coerce` name here -- we promise to not coerce to
            // Param::Float. But if it's an int or complex we need to store it as an Obj.
            if Some(true) == py_expr.inner.is_int() || Some(true) == py_expr.inner.is_complex() {
                Param::Obj(ob.to_owned().unbind())
            } else {
                Param::ParameterExpression(Arc::new(py_expr.inner))
            }
        } else {
            Param::Obj(ob.to_owned().unbind())
        })
    }

    /// Clones the [Param] object safely by reference count or copying.
    pub fn clone_ref(&self, py: Python) -> Self {
        match self {
            Param::ParameterExpression(exp) => Param::ParameterExpression(exp.clone()),
            Param::Float(float) => Param::Float(*float),
            Param::Obj(obj) => Param::Obj(obj.clone_ref(py)),
        }
    }

    pub fn py_deepcopy<'py>(
        &self,
        py: Python<'py>,
        memo: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Self> {
        match self {
            Param::Float(f) => Ok(Param::Float(*f)),
            _ => Self::extract_no_coerce(
                imports::DEEPCOPY
                    .get_bound(py)
                    .call1((self.clone(), memo))?
                    .as_borrowed(),
            ),
        }
    }
}

// This impl allows for shared usage between [Param] and &[Param].
// Such blanked impl doesn't exist inherently due to Rust's type system limitations.
// See https://doc.rust-lang.org/std/convert/trait.AsRef.html#reflexivity for more information.
impl AsRef<Param> for Param {
    fn as_ref(&self) -> &Param {
        self
    }
}

// Conveniently converts an f64 into a `Param`.
impl From<f64> for Param {
    fn from(value: f64) -> Self {
        Param::Float(value)
    }
}

/// Trait for generic circuit operations these define the common attributes
/// needed for something to be addable to the circuit struct
pub trait Operation {
    fn name(&self) -> &str;
    fn num_qubits(&self) -> u32;
    fn num_clbits(&self) -> u32;
    fn num_params(&self) -> u32;
    fn directive(&self) -> bool;
}

/// Unpacked view object onto a `PackedOperation`.  This is the return value of
/// `PackedInstruction::op`, and in turn is a view object onto a `PackedOperation`.
///
/// This is the main way that we interact immutably with general circuit operations from Rust space.
#[derive(Debug)]
pub enum OperationRef<'a> {
    ControlFlow(&'a ControlFlowInstruction),
    StandardGate(StandardGate),
    StandardInstruction(StandardInstruction),
    PyCustom(&'a PyInstruction),
    Unitary(&'a UnitaryGate),
    PauliProductMeasurement(&'a PauliProductMeasurement),
    PauliProductRotation(&'a PauliProductRotation),
    CustomOperation(&'a dyn CustomOperation),
}

impl Operation for OperationRef<'_> {
    #[inline]
    fn name(&self) -> &str {
        match self {
            Self::ControlFlow(op) => op.name(),
            Self::StandardGate(standard) => standard.name(),
            Self::StandardInstruction(instruction) => instruction.name(),
            Self::PyCustom(inst) => inst.name(),
            Self::Unitary(unitary) => unitary.name(),
            Self::PauliProductMeasurement(ppm) => ppm.name(),
            Self::PauliProductRotation(rotation) => rotation.name(),
            Self::CustomOperation(operation) => operation.name(),
        }
    }
    #[inline]
    fn num_qubits(&self) -> u32 {
        match self {
            Self::ControlFlow(op) => op.num_qubits(),
            Self::StandardGate(standard) => standard.num_qubits(),
            Self::StandardInstruction(instruction) => instruction.num_qubits(),
            Self::PyCustom(inst) => inst.num_qubits(),
            Self::Unitary(unitary) => unitary.num_qubits(),
            Self::PauliProductMeasurement(ppm) => ppm.num_qubits(),
            Self::PauliProductRotation(rotation) => rotation.num_qubits(),
            Self::CustomOperation(operation) => operation.num_qubits(),
        }
    }
    #[inline]
    fn num_clbits(&self) -> u32 {
        match self {
            Self::ControlFlow(op) => op.num_clbits(),
            Self::StandardGate(standard) => standard.num_clbits(),
            Self::StandardInstruction(instruction) => instruction.num_clbits(),
            Self::PyCustom(inst) => inst.num_clbits(),
            Self::Unitary(unitary) => unitary.num_clbits(),
            Self::PauliProductMeasurement(ppm) => ppm.num_clbits(),
            Self::PauliProductRotation(rotation) => rotation.num_clbits(),
            Self::CustomOperation(operation) => operation.num_clbits(),
        }
    }
    #[inline]
    fn num_params(&self) -> u32 {
        match self {
            Self::ControlFlow(op) => op.num_params(),
            Self::StandardGate(standard) => standard.num_params(),
            Self::StandardInstruction(instruction) => instruction.num_params(),
            Self::PyCustom(inst) => inst.num_params(),
            Self::Unitary(unitary) => unitary.num_params(),
            Self::PauliProductMeasurement(ppm) => ppm.num_params(),
            Self::PauliProductRotation(rotation) => rotation.num_params(),
            Self::CustomOperation(operation) => operation.num_params(),
        }
    }
    #[inline]
    fn directive(&self) -> bool {
        match self {
            Self::ControlFlow(op) => op.directive(),
            Self::StandardGate(standard) => standard.directive(),
            Self::StandardInstruction(instruction) => instruction.directive(),
            Self::PyCustom(inst) => inst.directive(),
            Self::Unitary(unitary) => unitary.directive(),
            Self::PauliProductMeasurement(ppm) => ppm.directive(),
            Self::PauliProductRotation(rotation) => rotation.directive(),
            Self::CustomOperation(operation) => operation.directive(),
        }
    }
}

/// Used to tag control flow instructions via the `_control_flow_type` class
/// attribute in the corresponding Python class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass(module = "qiskit._accelerate.circuit", eq, eq_int, from_py_object)]
#[repr(u8)]
pub enum ControlFlowType {
    Box = 0,
    BreakLoop = 1,
    ContinueLoop = 2,
    ForLoop = 3,
    IfElse = 4,
    SwitchCase = 5,
    WhileLoop = 6,
}

impl ControlFlowType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ControlFlowType::Box => "box",
            ControlFlowType::BreakLoop => "break_loop",
            ControlFlowType::ContinueLoop => "continue_loop",
            ControlFlowType::ForLoop => "for_loop",
            ControlFlowType::IfElse => "if_else",
            ControlFlowType::SwitchCase => "switch_case",
            ControlFlowType::WhileLoop => "while_loop",
        }
    }
}

impl FromStr for ControlFlowType {
    type Err = ();

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "box" => Ok(ControlFlowType::Box),
            "break_loop" => Ok(ControlFlowType::BreakLoop),
            "continue_loop" => Ok(ControlFlowType::ContinueLoop),
            "for_loop" => Ok(ControlFlowType::ForLoop),
            "if_else" => Ok(ControlFlowType::IfElse),
            "switch_case" => Ok(ControlFlowType::SwitchCase),
            "while_loop" => Ok(ControlFlowType::WhileLoop),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug, IntoPyObject, PartialEq)]
pub enum BoxDuration {
    Duration(Duration),
    Expr(expr::Expr),
}

/// A literal Python range extracted to a Rust object.
///
/// This is separate to `PyO3`'s `PyRange` type, since that keeps everything internally safe for
/// subclassing and modification the Python heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PyRange {
    pub start: isize,
    pub stop: isize,
    pub step: NonZero<isize>,
}
impl PyRange {
    pub fn is_empty(&self) -> bool {
        let step = self.step.unsigned_abs().get();
        let diff = self.start.abs_diff(self.stop);
        (self.step.get() > 0 && self.start < self.stop
            || (self.step.get() < 0 && self.start > self.stop))
            && diff >= step
    }
    pub fn len(&self) -> usize {
        let step = self.step.unsigned_abs().get();
        let diff = self.start.abs_diff(self.stop);
        if (self.step.get() > 0 && self.start < self.stop)
            || (self.step.get() < 0 && self.start > self.stop)
        {
            // The `diff-1` is guaranteed safe because the `start < stop` or `start > stop`
            // conditions guarantee that `diff` is at least 1.
            1 + (diff - 1) / step
        } else {
            0
        }
    }
}
impl<'py> IntoPyObject<'py> for PyRange {
    type Target = ::pyo3::types::PyRange;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        ::pyo3::types::PyRange::new_with_step(py, self.start, self.stop, self.step.get())
    }
}
impl<'py> IntoPyObject<'py> for &'_ PyRange {
    type Target = <PyRange as IntoPyObject<'py>>::Target;
    type Output = <PyRange as IntoPyObject<'py>>::Output;
    type Error = <PyRange as IntoPyObject<'py>>::Error;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (*self).into_pyobject(py)
    }
}
impl<'a, 'py> FromPyObject<'a, 'py> for PyRange {
    type Error = PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        use pyo3::types::PyRangeMethods;

        let ob = ob.cast::<pyo3::types::PyRange>()?;
        Ok(Self {
            start: ob.start()?,
            stop: ob.stop()?,
            step: NonZero::new(ob.step()?).expect("Python does not allow zero steps"),
        })
    }
}

/// Possible specifications of the "collection" that a for loop iterates over.
#[derive(Clone, Debug, IntoPyObject, IntoPyObjectRef, FromPyObject, PartialEq, Eq)]
pub enum ForCollection {
    /// A literal Python `range` object extracted to Rust.
    PyRange(PyRange),
    /// Some ordered collection of integers.
    List(Vec<isize>),
}
impl ForCollection {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::PyRange(xs) => xs.is_empty(),
            Self::List(xs) => xs.is_empty(),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::PyRange(xs) => xs.len(),
            Self::List(xs) => xs.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ControlFlowInstruction {
    pub control_flow: ControlFlow,
    pub num_qubits: u32,
    pub num_clbits: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LoopParam {
    Parameter(Symbol),
    Variable(Var),
}

impl<'a, 'py> FromPyObject<'a, 'py> for LoopParam {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(parameter) = ob.extract::<Symbol>() {
            Ok(LoopParam::Parameter(parameter))
        } else {
            ob.extract::<Var>()
                .map(LoopParam::Variable)
                .map_err(PyErr::from)
        }
    }
}

impl<'py> IntoPyObject<'py> for LoopParam {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            LoopParam::Parameter(symbol) => symbol.into_pyobject(py),
            LoopParam::Variable(var) => var.into_pyobject(py),
        }
    }
}

#[derive(Clone, Debug)]
#[repr(align(8))]
pub enum ControlFlow {
    Box {
        duration: Option<BoxDuration>,
        annotations: Vec<Py<PyAny>>,
    },
    BreakLoop,
    ContinueLoop,
    ForLoop {
        collection: ForCollection,
        loop_param: Option<LoopParam>,
    },
    IfElse {
        condition: Condition,
    },
    Switch {
        target: SwitchTarget,
        label_spec: Vec<Vec<CaseSpecifier>>,
        cases: u32,
    },
    While {
        condition: Condition,
    },
}

impl ControlFlowInstruction {
    /// Check if another control flow operations is equivalent to this one.
    ///
    /// This can be removed and [ControlFlowInstruction] can be made to implement [PartialEq]
    /// instead once `annotations` gets moved to the instruction.
    pub fn py_eq(&self, py: Python, other: &ControlFlowInstruction) -> PyResult<bool> {
        if self.num_qubits != other.num_qubits || self.num_clbits != other.num_clbits {
            return Ok(false);
        }
        match &self.control_flow {
            ControlFlow::Box {
                duration: self_duration,
                annotations: self_annotations,
            } => match &other.control_flow {
                ControlFlow::Box {
                    duration: other_duration,
                    annotations: other_annotations,
                } => {
                    if self_duration != other_duration
                        || self_annotations.len() != other_annotations.len()
                    {
                        return Ok(false);
                    }
                    for (a, b) in self_annotations.iter().zip(other_annotations) {
                        if !a.bind(py).eq(b)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                _ => Ok(false),
            },
            ControlFlow::BreakLoop => match &other.control_flow {
                ControlFlow::BreakLoop => Ok(true),
                _ => Ok(false),
            },
            ControlFlow::ContinueLoop => match &other.control_flow {
                ControlFlow::ContinueLoop => Ok(true),
                _ => Ok(false),
            },
            ControlFlow::ForLoop {
                collection: self_collection,
                loop_param: self_loop_param,
            } => match &other.control_flow {
                ControlFlow::ForLoop {
                    collection: other_collection,
                    loop_param: other_loop_param,
                } => Ok(self_collection == other_collection && self_loop_param == other_loop_param),
                _ => Ok(false),
            },
            ControlFlow::IfElse {
                condition: self_condition,
            } => match &other.control_flow {
                ControlFlow::IfElse {
                    condition: other_condition,
                } => Ok(self_condition == other_condition),
                _ => Ok(false),
            },
            ControlFlow::Switch {
                target: self_target,
                label_spec: self_label_spec,
                cases: self_cases,
            } => match &other.control_flow {
                ControlFlow::Switch {
                    target: other_target,
                    label_spec: other_label_spec,
                    cases: other_cases,
                } => Ok(self_cases == other_cases
                    && self_target == other_target
                    && self_label_spec == other_label_spec),
                _ => Ok(false),
            },
            ControlFlow::While {
                condition: self_condition,
            } => match &other.control_flow {
                ControlFlow::While {
                    condition: other_condition,
                } => Ok(self_condition == other_condition),
                _ => Ok(false),
            },
        }
    }

    pub fn create_py_op(
        &self,
        py: Python,
        blocks: Option<Vec<CircuitData>>,
        label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let mut blocks = blocks.into_iter().flatten();
        let kwargs = label
            .map(|label| [("label", label.into_py_any(py)?)].into_py_dict(py))
            .transpose()?;
        match &self.control_flow {
            ControlFlow::Box {
                duration,
                annotations,
            } => {
                let (duration, unit) = match duration {
                    Some(duration) => match duration {
                        BoxDuration::Duration(duration) => {
                            (Some(duration.py_value(py)), Some(duration.unit()))
                        }
                        BoxDuration::Expr(expr) => {
                            (Some(expr.clone().into_bound_py_any(py)?), Some("expr"))
                        }
                    },
                    None => (None, None),
                };
                imports::BOX_OP.get(py).call1(
                    py,
                    (
                        blocks
                            .next()
                            .expect("box should have a body")
                            .into_py_quantum_circuit(py)?,
                        duration,
                        unit,
                        label,
                        PyTuple::new(py, annotations)?,
                    ),
                )
            }
            ControlFlow::BreakLoop => imports::BREAK_LOOP_OP.get(py).call(
                py,
                (self.num_qubits, self.num_clbits),
                kwargs.as_ref(),
            ),
            ControlFlow::ContinueLoop => imports::CONTINUE_LOOP_OP.get(py).call(
                py,
                (self.num_qubits, self.num_clbits),
                kwargs.as_ref(),
            ),
            ControlFlow::ForLoop {
                collection,
                loop_param,
            } => imports::FOR_LOOP_OP.get(py).call(
                py,
                (
                    collection,
                    loop_param.clone(),
                    blocks
                        .next()
                        .expect("for loop should have a body")
                        .into_py_quantum_circuit(py)?,
                ),
                kwargs.as_ref(),
            ),
            ControlFlow::IfElse { condition } => imports::IF_ELSE_OP.get(py).call(
                py,
                (
                    condition.clone(),
                    blocks
                        .next()
                        .expect("if should have a true body")
                        .into_py_quantum_circuit(py)?,
                    blocks
                        .next()
                        .map(|circuit| circuit.into_py_quantum_circuit(py))
                        .transpose()?,
                ),
                kwargs.as_ref(),
            ),
            ControlFlow::Switch {
                target, label_spec, ..
            } => {
                let cases_specifier: Vec<(Vec<CaseSpecifier>, Py<PyAny>)> = label_spec
                    .iter()
                    .cloned()
                    .zip(blocks)
                    .map(|(cases, body)| {
                        body.into_py_quantum_circuit(py)
                            .map(|ob| (cases, ob.unbind()))
                    })
                    .collect::<PyResult<_>>()?;
                imports::SWITCH_CASE_OP.get(py).call(
                    py,
                    (target.clone(), cases_specifier),
                    kwargs.as_ref(),
                )
            }
            ControlFlow::While { condition, .. } => imports::WHILE_LOOP_OP.get(py).call(
                py,
                (
                    condition.clone(),
                    blocks
                        .next()
                        .expect("while should have a body")
                        .into_py_quantum_circuit(py)?,
                ),
                kwargs.as_ref(),
            ),
        }
    }
}

impl Operation for ControlFlowInstruction {
    fn name(&self) -> &str {
        match &self.control_flow {
            ControlFlow::Box { .. } => ControlFlowType::Box.as_str(),
            ControlFlow::BreakLoop => ControlFlowType::BreakLoop.as_str(),
            ControlFlow::ContinueLoop => ControlFlowType::ContinueLoop.as_str(),
            ControlFlow::ForLoop { .. } => ControlFlowType::ForLoop.as_str(),
            ControlFlow::IfElse { .. } => ControlFlowType::IfElse.as_str(),
            ControlFlow::Switch { .. } => ControlFlowType::SwitchCase.as_str(),
            ControlFlow::While { .. } => ControlFlowType::WhileLoop.as_str(),
        }
    }

    fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    fn num_clbits(&self) -> u32 {
        self.num_clbits
    }

    fn num_params(&self) -> u32 {
        match &self.control_flow {
            ControlFlow::Box { .. } => 1,
            ControlFlow::BreakLoop => 0,
            ControlFlow::ContinueLoop => 0,
            ControlFlow::ForLoop { .. } => 3,
            ControlFlow::IfElse { .. } => 2,
            ControlFlow::Switch { cases, .. } => *cases,
            ControlFlow::While { .. } => 1,
        }
    }

    fn directive(&self) -> bool {
        false
    }
}

/// An ergonomic view of a control flow operation and its blocks.
#[derive(Clone, Debug)]
pub enum ControlFlowView<'a, T> {
    Box {
        duration: Option<&'a BoxDuration>,
        body: &'a T,
    },
    BreakLoop,
    ContinueLoop,
    ForLoop {
        collection: &'a ForCollection,
        loop_param: Option<&'a LoopParam>,
        body: &'a T,
    },
    IfElse {
        condition: &'a Condition,
        true_body: &'a T,
        false_body: Option<&'a T>,
    },
    Switch {
        target: &'a SwitchTarget,
        cases_specifier: Vec<(&'a [CaseSpecifier], &'a T)>,
    },
    While {
        condition: &'a Condition,
        body: &'a T,
    },
}

impl<'a, T> ControlFlowView<'a, T> {
    /// Produce a complete control-flow view object from the given instruction.
    ///
    /// While [CircuitData] and [DAGCircuit] both provide `try_view_control_flow` methods which just
    /// delegate to this internally, this function is useful for a) code de-duplication and b)
    /// finer-grained borrow-check control from within the `impl` blocks of [CircuitData] and
    /// [DAGCircuit].
    ///
    /// Panics or produces invalid results if `inst` and `blocks` aren't from compatible sources
    /// (e.g. the same [CircuitData]).
    pub fn try_from_instruction(
        inst: &'a PackedInstruction,
        blocks: &'a ControlFlowBlocks<T>,
    ) -> Option<Self> {
        let OperationRef::ControlFlow(cf) = inst.op.view() else {
            return None;
        };
        let block_ids = inst.blocks_view();
        let view = match &cf.control_flow {
            ControlFlow::Box {
                duration,
                annotations: _,
            } => Self::Box {
                duration: duration.as_ref(),
                body: &blocks[block_ids[0]],
            },
            ControlFlow::BreakLoop => Self::BreakLoop,
            ControlFlow::ContinueLoop => Self::ContinueLoop,
            ControlFlow::ForLoop {
                collection,
                loop_param,
            } => Self::ForLoop {
                collection,
                loop_param: loop_param.as_ref(),
                body: &blocks[block_ids[0]],
            },
            ControlFlow::IfElse { condition } => Self::IfElse {
                condition,
                true_body: &blocks[block_ids[0]],
                false_body: block_ids.get(1).map(|bid| &blocks[*bid]),
            },
            ControlFlow::Switch {
                target,
                label_spec,
                cases: _,
            } => Self::Switch {
                target,
                cases_specifier: label_spec
                    .iter()
                    .zip(block_ids)
                    .map(|(cases, bid)| (cases.as_slice(), &blocks[*bid]))
                    .collect(),
            },
            ControlFlow::While { condition } => Self::While {
                condition,
                body: &blocks[block_ids[0]],
            },
        };
        Some(view)
    }

    pub fn blocks(&self) -> Vec<&'a T> {
        match self {
            ControlFlowView::Box { body, .. } => vec![*body],
            ControlFlowView::BreakLoop => vec![],
            ControlFlowView::ContinueLoop => vec![],
            ControlFlowView::ForLoop { body, .. } => vec![*body],
            ControlFlowView::IfElse {
                true_body,
                false_body,
                ..
            } => {
                if let Some(false_body) = false_body {
                    vec![*true_body, *false_body]
                } else {
                    vec![*true_body]
                }
            }
            ControlFlowView::Switch {
                cases_specifier, ..
            } => cases_specifier.iter().map(|(_, block)| *block).collect(),
            ControlFlowView::While { body, .. } => vec![*body],
        }
    }
}

/// A control flow operation's condition.
#[derive(Clone, Debug, PartialEq, IntoPyObject)]
pub enum Condition {
    Bit(ShareableClbit, bool),
    Register(ClassicalRegister, BigUint),
    Expr(expr::Expr),
}

impl<'a, 'py> FromPyObject<'a, 'py> for Condition {
    type Error = <expr::Expr as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok((bit, value)) = ob.extract::<(ShareableClbit, Bound<PyAny>)>() {
            Ok(Condition::Bit(bit, value.is_truthy()?))
        } else if let Ok((register, value)) = ob.extract::<(ClassicalRegister, BigUint)>() {
            Ok(Condition::Register(register, value))
        } else {
            Ok(Condition::Expr(ob.extract()?))
        }
    }
}

/// A control flow operation's target.
#[derive(Clone, Debug, PartialEq, IntoPyObject)]
pub enum SwitchTarget {
    Bit(ShareableClbit),
    Register(ClassicalRegister),
    Expr(expr::Expr),
}

impl<'a, 'py> FromPyObject<'a, 'py> for SwitchTarget {
    type Error = <expr::Expr as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(bit) = ob.extract::<ShareableClbit>() {
            Ok(SwitchTarget::Bit(bit))
        } else if let Ok(register) = ob.extract::<ClassicalRegister>() {
            Ok(SwitchTarget::Register(register))
        } else {
            Ok(SwitchTarget::Expr(ob.extract()?))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CaseSpecifier {
    Uint(BigUint),
    Default,
}

impl<'a, 'py> FromPyObject<'a, 'py> for CaseSpecifier {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(i) = ob.extract::<BigUint>() {
            Ok(CaseSpecifier::Uint(i))
        } else if ob.is(imports::SWITCH_CASE_DEFAULT.get_bound(ob.py())) {
            Ok(CaseSpecifier::Default)
        } else {
            Err(PyValueError::new_err("invalid case specifier"))
        }
    }
}

impl<'py> IntoPyObject<'py> for CaseSpecifier {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            CaseSpecifier::Uint(u) => u.into_bound_py_any(py),
            CaseSpecifier::Default => Ok(imports::SWITCH_CASE_DEFAULT.get_bound(py).clone()),
        }
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum DelayUnit {
    NS,
    PS,
    US,
    MS,
    S,
    DT,
    EXPR,
}

unsafe impl ::bytemuck::CheckedBitPattern for DelayUnit {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 7
    }
}
unsafe impl ::bytemuck::NoUninit for DelayUnit {}

impl fmt::Display for DelayUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DelayUnit::NS => "ns",
                DelayUnit::PS => "ps",
                DelayUnit::US => "us",
                DelayUnit::MS => "ms",
                DelayUnit::S => "s",
                DelayUnit::DT => "dt",
                DelayUnit::EXPR => "expr",
            }
        )
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for DelayUnit {
    type Error = PyErr;

    fn extract(b: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let str: String = b.extract()?;
        Ok(match str.as_str() {
            "ns" => DelayUnit::NS,
            "ps" => DelayUnit::PS,
            "us" => DelayUnit::US,
            "ms" => DelayUnit::MS,
            "s" => DelayUnit::S,
            "dt" => DelayUnit::DT,
            "expr" => DelayUnit::EXPR,
            unknown_unit => {
                return Err(PyValueError::new_err(format!(
                    "Unit '{unknown_unit}' is invalid."
                )));
            }
        })
    }
}

/// An internal type used to further discriminate the payload of a `PackedOperation` when its
/// discriminant is `PackedOperationType::StandardInstruction`.
///
/// This is also used to tag standard instructions via the `_standard_instruction_type` class
/// attribute in the corresponding Python class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass(module = "qiskit._accelerate.circuit", eq, eq_int, from_py_object)]
#[repr(u8)]
pub enum StandardInstructionType {
    Barrier = 0,
    Delay = 1,
    Measure = 2,
    Reset = 3,
}

unsafe impl ::bytemuck::CheckedBitPattern for StandardInstructionType {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 4
    }
}
unsafe impl ::bytemuck::NoUninit for StandardInstructionType {}

impl StandardInstructionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            StandardInstructionType::Barrier => "barrier",
            StandardInstructionType::Delay => "delay",
            StandardInstructionType::Measure => "measure",
            StandardInstructionType::Reset => "reset",
        }
    }
}

impl FromStr for StandardInstructionType {
    type Err = ();

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "barrier" => Ok(StandardInstructionType::Barrier),
            "delay" => Ok(StandardInstructionType::Delay),
            "measure" => Ok(StandardInstructionType::Measure),
            "reset" => Ok(StandardInstructionType::Reset),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
pub enum StandardInstruction {
    Barrier(u32),
    Delay(DelayUnit),
    Measure,
    Reset,
}

// This must be kept up-to-date with `StandardInstruction` when adding or removing
// gates from the enum
//
// Remove this when std::mem::variant_count() is stabilized (see
// https://github.com/rust-lang/rust/issues/73662 )
pub const STANDARD_INSTRUCTION_SIZE: usize = 4;

impl Operation for StandardInstruction {
    fn name(&self) -> &str {
        match self {
            StandardInstruction::Barrier(_) => StandardInstructionType::Barrier.as_str(),
            StandardInstruction::Delay(_) => StandardInstructionType::Delay.as_str(),
            StandardInstruction::Measure => StandardInstructionType::Measure.as_str(),
            StandardInstruction::Reset => StandardInstructionType::Reset.as_str(),
        }
    }

    fn num_qubits(&self) -> u32 {
        match self {
            StandardInstruction::Barrier(num_qubits) => *num_qubits,
            StandardInstruction::Delay(_) => 1,
            StandardInstruction::Measure => 1,
            StandardInstruction::Reset => 1,
        }
    }

    fn num_clbits(&self) -> u32 {
        match self {
            StandardInstruction::Barrier(_) => 0,
            StandardInstruction::Delay(_) => 0,
            StandardInstruction::Measure => 1,
            StandardInstruction::Reset => 0,
        }
    }

    fn num_params(&self) -> u32 {
        match self {
            StandardInstruction::Delay(_) => 1,
            _ => 0,
        }
    }

    fn directive(&self) -> bool {
        match self {
            StandardInstruction::Barrier(_) => true,
            StandardInstruction::Delay(_) => false,
            StandardInstruction::Measure => false,
            StandardInstruction::Reset => false,
        }
    }
}

impl StandardInstruction {
    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<SmallVec<[Param; 3]>>,
        label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let mut params = params.into_iter().flatten();
        let kwargs = label
            .map(|label| [("label", label.into_py_any(py)?)].into_py_dict(py))
            .transpose()?;
        let out = match self {
            StandardInstruction::Barrier(num_qubits) => imports::BARRIER
                .get_bound(py)
                .call((num_qubits,), kwargs.as_ref())?,
            StandardInstruction::Delay(unit) => {
                let duration = params.next().unwrap();
                imports::DELAY
                    .get_bound(py)
                    .call1((duration.into_py_any(py)?, unit.to_string()))?
            }
            StandardInstruction::Measure => {
                imports::MEASURE.get_bound(py).call((), kwargs.as_ref())?
            }
            StandardInstruction::Reset => imports::RESET.get_bound(py).call((), kwargs.as_ref())?,
        };

        Ok(out.unbind())
    }
}

// Return explicitly requested copy of `param`, handling
// each variant separately.
pub fn clone_param(param: &Param) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta),
        Param::ParameterExpression(theta) => Param::ParameterExpression(theta.clone()),
        Param::Obj(_) => unreachable!(),
    }
}

/// Multiply a ``Param`` with a float.
pub fn multiply_param(param: &Param, mult: f64) -> Param {
    match param {
        Param::Float(theta) => Param::Float(theta * mult),
        Param::ParameterExpression(theta) => {
            // safe to unwrap as multiplication with float does not have name conflicts
            Param::ParameterExpression(Arc::new(
                theta.mul(&ParameterExpression::from_f64(mult)).unwrap(),
            ))
        }
        Param::Obj(_) => unreachable!("Unsupported multiplication of a Param::Obj."),
    }
}

/// Multiply two ``Param``s.
pub fn multiply_params(param1: Param, param2: Param) -> Param {
    match (&param1, &param2) {
        (Param::Float(theta), Param::Float(lambda)) => Param::Float(theta * lambda),
        (param, Param::Float(theta)) => multiply_param(param, *theta),
        (Param::Float(theta), param) => multiply_param(param, *theta),
        (Param::ParameterExpression(p1), Param::ParameterExpression(p2)) => {
            // TODO we could properly propagate the error here
            Param::ParameterExpression(Arc::new(p1.mul(p2).expect("Name conflict during mul.")))
        }
        _ => unreachable!("Unsupported multiplication."),
    }
}

pub fn add_param(param: &Param, summand: f64) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta + summand),
        Param::ParameterExpression(theta) => Param::ParameterExpression(
            // safe to unwrap as addition with float does not have name conflicts
            Arc::new(theta.add(&ParameterExpression::from_f64(summand)).unwrap()),
        ),
        Param::Obj(_) => unreachable!("Unsupported addition of a Param::Obj."),
    }
}

pub fn radd_param(param1: Param, param2: Param) -> Param {
    match [&param1, &param2] {
        [Param::Float(theta), Param::Float(lambda)] => Param::Float(theta + lambda),
        [Param::Float(theta), Param::ParameterExpression(_lambda)] => add_param(&param2, *theta),
        [Param::ParameterExpression(_theta), Param::Float(lambda)] => add_param(&param1, *lambda),
        [
            Param::ParameterExpression(theta),
            Param::ParameterExpression(lambda),
        ] => {
            // TODO we could properly propagate the error here
            Param::ParameterExpression(Arc::new(
                theta.add(lambda).expect("Name conflict during add."),
            ))
        }
        _ => unreachable!("Unsupported addition."),
    }
}

/// This trait is defined on operation types in the circuit that are defined in Python.
/// It contains the methods for managing the Python aspect
pub trait PythonOperation: Sized {
    /// Copy this operation, including a Python-space deep copy
    fn py_deepcopy(&self, py: Python, memo: Option<&Bound<'_, PyDict>>) -> PyResult<Self>;

    /// Copy this operation, including a Python-space call to `copy` on the `Operation` subclass.
    fn py_copy(&self, py: Python) -> PyResult<Self>;
}

/// Which Python subclass a [`PyInstruction`] is associated with.
///
/// This is the same as the `Operation`/`Instruction`/`Gate` split that Python-space has.  `Gate`
/// incorporates all of `Instruction`, which in turn incorporates all of `Operation`.  `Gate` means
/// the operation represents a unitary action, `Instruction` is general and (usually) has a built-in
/// hierarchical definition, while `Operation` is nearly entirely opaque.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub enum PyOpKind {
    /// The instruction implements only `Operation`.
    Operation,
    /// The instruction is a subclass of `Instruction`.
    Instruction,
    /// The instruction is a subclass of `Gate`.
    Gate,
}
impl PyOpKind {
    /// Get the operation kind from a Python type.
    ///
    /// The `Err` variant is only for the propagation of Python errors during `issubclass` checks.
    /// The function returns `Ok(None)` if the type is not in the `Operation` hierarchy.
    pub fn from_type(ob: Borrowed<PyType>) -> PyResult<Option<Self>> {
        let py = ob.py();
        if ob.is_subclass(imports::GATE.get_bound(py))? {
            Ok(Some(Self::Gate))
        } else if ob.is_subclass(imports::INSTRUCTION.get_bound(py))? {
            Ok(Some(Self::Instruction))
        } else {
            ob.is_subclass(imports::OPERATION.get_bound(py))
                .map(|ok| ok.then_some(Self::Operation))
        }
    }
}

/// A custom operation from Python space.
///
/// These could be backed by "generalized" instructions from Qiskit's Python code, or completely
/// arbitrary user code.  Rust code, in general, can only understand these objects through use of
/// `Operation` trait and similar methods.
///
/// If you find yourself deeply inspecting or traversing the internal Python object manually,
/// something is probably not right; there is likely either something missing from Rust space, or
/// the Rust-space code is too tightly coupled to a custom Python object.
#[derive(Clone, Debug)]
#[repr(align(8))] // This is a `PackedOperation` packed pointer, so needs a fixed alignment.
pub struct PyInstruction {
    /// What Python-space subclass the operation is associated with.  This field represents the same
    /// `Operation`/`Instruction`/`Gate` split that Python space has.
    pub kind: PyOpKind,
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub ob: Py<PyAny>,
}

impl PythonOperation for PyInstruction {
    fn py_deepcopy(&self, py: Python, memo: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let deepcopy = imports::DEEPCOPY.get_bound(py);
        Ok(PyInstruction {
            ob: deepcopy.call1((&self.ob, memo))?.unbind(),
            ..self.clone()
        })
    }

    fn py_copy(&self, py: Python) -> PyResult<Self> {
        let copy_attr = intern!(py, "copy");
        Ok(PyInstruction {
            ob: self.ob.call_method0(py, copy_attr)?,
            ..self.clone()
        })
    }
}

impl Operation for PyInstruction {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }

    fn directive(&self) -> bool {
        Python::attach(|py| -> bool {
            match self.ob.getattr(py, intern!(py, "_directive")) {
                Ok(directive) => {
                    let res: bool = directive.extract(py).unwrap();
                    res
                }
                Err(_) => false,
            }
        })
    }
}

impl PyInstruction {
    /// returns the number of control qubits in the instruction, if any.
    pub fn num_ctrl_qubits(&self) -> Option<u32> {
        if self.kind != PyOpKind::Gate {
            return None;
        }
        Python::attach(|py| {
            self.ob
                .getattr(py, "num_ctrl_qubits")
                .and_then(|py_num_ctrl_qubits| py_num_ctrl_qubits.extract::<u32>(py))
                .ok()
        })
    }

    /// returns the control state of the gate as a decimal number, if any.
    pub fn ctrl_state(&self) -> Option<u32> {
        if self.kind != PyOpKind::Gate {
            return None;
        }
        Python::attach(|py| {
            self.ob
                .getattr(py, "ctrl_state")
                .and_then(|py_ctrl_state| py_ctrl_state.extract::<u32>(py))
                .ok()
        })
    }
    /// returns the class name of the python gate
    pub fn class_name(&self, py: Python) -> PyResult<String> {
        self.ob
            .bind(py)
            .getattr(intern!(py, "__class__"))?
            .getattr(intern!(py, "__name__"))?
            .extract::<String>()
    }

    pub fn matrix(&self) -> Option<Array2<Complex64>> {
        if self.kind != PyOpKind::Gate {
            return None;
        }
        Python::attach(|py| -> Option<Array2<Complex64>> {
            match self.ob.getattr(py, intern!(py, "to_matrix")) {
                Ok(to_matrix) => {
                    let res: Option<Py<PyAny>> = to_matrix.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let array: PyReadonlyArray2<Complex64> = x.extract(py).ok()?;
                            Some(array.as_array().to_owned())
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }

    /// Get the complete definition field, if it exists.
    pub fn py_definition<'py>(&self, py: Python<'py>) -> PyResult<Option<QuantumCircuitData<'py>>> {
        if self.kind == PyOpKind::Operation {
            return Ok(None);
        }
        self.ob
            .bind(py)
            .getattr(intern!(py, "definition"))
            .and_then(|ob| ob.extract())
    }

    pub fn definition(&self) -> Option<CircuitData> {
        // The `definition` attribute isn't part of the `Operation` interface, so it's invalid for
        // us to access it.
        if self.kind == PyOpKind::Operation {
            return None;
        }
        Python::attach(|py| -> Option<CircuitData> {
            match self.ob.getattr(py, intern!(py, "definition")) {
                Ok(definition) => definition
                    .bind(py)
                    .getattr(intern!(py, "_data"))
                    .ok()?
                    .cast::<PyCircuitData>()
                    .map(|data| data.borrow().inner.clone())
                    .ok(),
                Err(_) => None,
            }
        })
    }

    pub fn matrix_as_static_1q(&self) -> Option<[[Complex64; 2]; 2]> {
        if self.kind != PyOpKind::Gate || self.num_qubits() != 1 {
            return None;
        }
        Python::attach(|py| -> Option<[[Complex64; 2]; 2]> {
            let array = self
                .ob
                .call_method0(py, intern!(py, "to_matrix"))
                .ok()?
                .extract::<PyReadonlyArray2<Complex64>>(py)
                .ok()?;
            let arr = array.as_array();
            Some([[arr[[0, 0]], arr[[0, 1]]], [arr[[1, 0]], arr[[1, 1]]]])
        })
    }

    pub fn matrix_as_static_2q(&self) -> Option<[[Complex64; 4]; 4]> {
        if self.kind != PyOpKind::Gate || self.num_qubits() != 2 {
            return None;
        }
        Python::attach(|py| -> Option<[[Complex64; 4]; 4]> {
            let array = self
                .ob
                .call_method0(py, intern!(py, "to_matrix"))
                .ok()?
                .extract::<PyReadonlyArray2<Complex64>>(py)
                .ok()?;
            let arr = array.as_array();
            Some([
                [arr[[0, 0]], arr[[0, 1]], arr[[0, 2]], arr[[0, 3]]],
                [arr[[1, 0]], arr[[1, 1]], arr[[1, 2]], arr[[1, 3]]],
                [arr[[2, 0]], arr[[2, 1]], arr[[2, 2]], arr[[2, 3]]],
                [arr[[3, 0]], arr[[3, 1]], arr[[3, 2]], arr[[3, 3]]],
            ])
        })
    }

    /// Reference the Python object backing this object, if it is a gate.
    pub fn gate_object(&self) -> Option<&Py<PyAny>> {
        (self.kind == PyOpKind::Gate).then_some(&self.ob)
    }
}

#[derive(Clone, Debug)]
pub enum ArrayType {
    NDArray(Array2<Complex64>),
    OneQ(Matrix2<Complex64>),
    TwoQ(Matrix4<Complex64>),
}

/// This class is a rust representation of a UnitaryGate in Python,
/// a gate represented solely by it's unitary matrix.
#[derive(Clone, Debug)]
#[repr(align(8))]
pub struct UnitaryGate {
    pub array: ArrayType,
}

impl PartialEq for UnitaryGate {
    fn eq(&self, other: &Self) -> bool {
        match (&self.array, &other.array) {
            (ArrayType::OneQ(mat1), ArrayType::OneQ(mat2)) => mat1 == mat2,
            (ArrayType::TwoQ(mat1), ArrayType::TwoQ(mat2)) => mat1 == mat2,
            // we could also slightly optimize comparisons between NDArray and OneQ/TwoQ if
            // this becomes performance critical
            _ => self.matrix() == other.matrix(),
        }
    }
}

impl Operation for UnitaryGate {
    fn name(&self) -> &str {
        "unitary"
    }
    fn num_qubits(&self) -> u32 {
        match &self.array {
            ArrayType::NDArray(arr) => arr.shape()[0].ilog2(),
            ArrayType::OneQ(_) => 1,
            ArrayType::TwoQ(_) => 2,
        }
    }
    fn num_clbits(&self) -> u32 {
        0
    }
    fn num_params(&self) -> u32 {
        0
    }
    fn directive(&self) -> bool {
        false
    }
}

impl UnitaryGate {
    pub fn matrix(&self) -> Option<Array2<Complex64>> {
        match &self.array {
            ArrayType::NDArray(arr) => Some(arr.clone()),
            ArrayType::OneQ(mat) => Some(array!(
                [mat[(0, 0)], mat[(0, 1)]],
                [mat[(1, 0)], mat[(1, 1)]],
            )),
            ArrayType::TwoQ(mat) => Some(array!(
                [mat[(0, 0)], mat[(0, 1)], mat[(0, 2)], mat[(0, 3)]],
                [mat[(1, 0)], mat[(1, 1)], mat[(1, 2)], mat[(1, 3)]],
                [mat[(2, 0)], mat[(2, 1)], mat[(2, 2)], mat[(2, 3)]],
                [mat[(3, 0)], mat[(3, 1)], mat[(3, 2)], mat[(3, 3)]],
            )),
        }
    }

    pub fn matrix_as_static_1q(&self) -> Option<[[Complex64; 2]; 2]> {
        match &self.array {
            ArrayType::OneQ(mat) => Some([[mat[(0, 0)], mat[(0, 1)]], [mat[(1, 0)], mat[(1, 1)]]]),
            ArrayType::NDArray(arr) => {
                if self.num_qubits() == 1 {
                    Some([[arr[(0, 0)], arr[(0, 1)]], [arr[(1, 0)], arr[(1, 1)]]])
                } else {
                    None
                }
            }
            ArrayType::TwoQ(_) => None,
        }
    }

    pub fn matrix_as_static_2q(&self) -> Option<[[Complex64; 4]; 4]> {
        match &self.array {
            ArrayType::OneQ(_mat) => None,
            ArrayType::NDArray(arr) => {
                if self.num_qubits() == 2 {
                    Some([
                        [arr[[0, 0]], arr[[0, 1]], arr[[0, 2]], arr[[0, 3]]],
                        [arr[[1, 0]], arr[[1, 1]], arr[[1, 2]], arr[[1, 3]]],
                        [arr[[2, 0]], arr[[2, 1]], arr[[2, 2]], arr[[2, 3]]],
                        [arr[[3, 0]], arr[[3, 1]], arr[[3, 2]], arr[[3, 3]]],
                    ])
                } else {
                    None
                }
            }
            ArrayType::TwoQ(mat) => Some([
                [mat[(0, 0)], mat[(0, 1)], mat[(0, 2)], mat[(0, 3)]],
                [mat[(1, 0)], mat[(1, 1)], mat[(1, 2)], mat[(1, 3)]],
                [mat[(2, 0)], mat[(2, 1)], mat[(2, 2)], mat[(2, 3)]],
                [mat[(3, 0)], mat[(3, 1)], mat[(3, 2)], mat[(3, 3)]],
            ]),
        }
    }

    pub fn matrix_as_nalgebra_1q(&self) -> Option<Matrix2<Complex64>> {
        match &self.array {
            ArrayType::OneQ(mat) => Some(*mat),
            ArrayType::NDArray(arr) => {
                if self.num_qubits() == 1 {
                    Some(Matrix2::new(
                        arr[[0, 0]],
                        arr[[0, 1]],
                        arr[[1, 0]],
                        arr[[1, 1]],
                    ))
                } else {
                    None
                }
            }
            ArrayType::TwoQ(_) => None,
        }
    }
    pub fn matrix_as_nalgebra_2q(&self) -> Option<Matrix4<Complex64>> {
        match &self.array {
            ArrayType::OneQ(_mat) => None,
            ArrayType::NDArray(arr) => {
                if self.num_qubits() == 2 {
                    Some(Matrix4::new(
                        arr[[0, 0]],
                        arr[[0, 1]],
                        arr[[0, 2]],
                        arr[[0, 3]],
                        arr[[1, 0]],
                        arr[[1, 1]],
                        arr[[1, 2]],
                        arr[[1, 3]],
                        arr[[2, 0]],
                        arr[[2, 1]],
                        arr[[2, 2]],
                        arr[[2, 3]],
                        arr[[3, 0]],
                        arr[[3, 1]],
                        arr[[3, 2]],
                        arr[[3, 3]],
                    ))
                } else {
                    None
                }
            }
            ArrayType::TwoQ(mat) => Some(*mat),
        }
    }
}

impl UnitaryGate {
    pub fn create_py_op(&self, py: Python, label: Option<&str>) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        if let Some(label) = label {
            kwargs.set_item(intern!(py, "label"), label.into_py_any(py)?)?;
        }
        let out_array = match &self.array {
            ArrayType::NDArray(arr) => arr.to_pyarray(py),
            ArrayType::OneQ(arr) => arr.to_pyarray(py),
            ArrayType::TwoQ(arr) => arr.to_pyarray(py),
        };
        kwargs.set_item(intern!(py, "check_input"), false)?;
        kwargs.set_item(intern!(py, "num_qubits"), self.num_qubits())?;
        let gate = imports::UNITARY_GATE
            .get_bound(py)
            .call((out_array,), Some(&kwargs))?;
        Ok(gate.unbind())
    }

    /// Get a read-only ndarray view of the matrix stored in the `UnitaryGate`
    ///
    /// Regardless of the underlying array type `Matrix2`, `Matrix4`, or `Array2` it returns
    /// a read-only an ndarray `ArrayView2` view to the underlying matrix by reference.
    #[inline]
    pub fn matrix_view(&self) -> ArrayView2<'_, Complex64> {
        match &self.array {
            ArrayType::NDArray(arr) => arr.view(),
            ArrayType::OneQ(mat) => {
                let dim = Dim(mat.shape());
                let strides = Dim(mat.strides());
                // SAFETY: We know the array is a 2x2 and contiguous block so we don't need to
                // check for invalid format
                unsafe { ArrayView2::from_shape_ptr(dim.strides(strides), mat.get_unchecked(0)) }
            }
            ArrayType::TwoQ(mat) => {
                let dim = Dim(mat.shape());
                let strides = Dim(mat.strides());
                // SAFETY: We know the array is a 4x4 and contiguous block so we don't need to
                // check for invalid format
                unsafe { ArrayView2::from_shape_ptr(dim.strides(strides), mat.get_unchecked(0)) }
            }
        }
    }
}

/// A Pauli-based gate model, consisting of [PauliProductRotation] and [PauliProductMeasurement] ops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PauliBased {
    PauliProductRotation(PauliProductRotation),
    PauliProductMeasurement(PauliProductMeasurement),
}

#[derive(Clone, Debug)]
#[repr(align(8))]
pub struct PauliProductRotation {
    /// The z-component of the pauli.
    pub z: Vec<bool>,
    /// The x-component of the pauli.
    pub x: Vec<bool>,
    /// The rotation angle, exp(i theta / 2 P)
    pub angle: Param,
}

impl Operation for PauliProductRotation {
    fn name(&self) -> &str {
        "pauli_product_rotation"
    }
    fn num_qubits(&self) -> u32 {
        self.z.len() as u32
    }
    fn num_clbits(&self) -> u32 {
        0
    }
    fn num_params(&self) -> u32 {
        1
    }
    fn directive(&self) -> bool {
        false
    }
}

impl PauliProductRotation {
    pub fn create_py_op(&self, py: Python, label: Option<&str>) -> PyResult<Py<PyAny>> {
        let z = self.z.to_pyarray(py);
        let x = self.x.to_pyarray(py);

        let py_label = if let Some(label) = label {
            label.into_py_any(py)?
        } else {
            py.None()
        };

        let gate = imports::PAULI_PRODUCT_ROTATION_GATE
            .get_bound(py)
            .call_method1(
                intern!(py, "_from_pauli_data"),
                (z, x, self.angle.clone(), py_label),
            )?;
        Ok(gate.unbind())
    }

    /// Attempts to merge `self` and `other`.
    /// If successful, returns the merged [PauliProductRotation].
    /// If not successful, returns `None`.
    pub fn merge_with(&self, other: &Self) -> Option<Self> {
        if self.x == other.x && self.z == other.z {
            Some(PauliProductRotation {
                z: self.z.clone(),
                x: self.x.clone(),
                angle: radd_param(self.angle.clone(), other.angle.clone()),
            })
        } else {
            None
        }
    }

    /// For a [PauliProductRotation] gate with a floating-point angle return a tuple `(Tr(gate) / dim, dim)`.
    /// Return `None` if the angle is parameterized.
    pub fn rotation_trace_and_dim(&self) -> Option<(Complex64, f64)> {
        let Param::Float(angle) = self.angle else {
            return None;
        };

        let num_qubits = self
            .z
            .iter()
            .zip(self.x.iter())
            .filter(|(z, x)| **z || **x)
            .count();
        let dim = 2u32.pow(num_qubits as u32);
        let tr_over_dim = if num_qubits == 0 {
            // This is an identity Pauli rotation.
            (Complex64::new(0.0, -angle / 2.)).exp()
        } else {
            Complex64::new((angle / 2.).cos(), 0.)
        };

        Some((tr_over_dim, dim as f64))
    }

    /// Return a dense matrix representation of the matrix.
    ///
    /// # Returns
    ///
    /// * Some(matrix) - If the matrix was successfully computed.
    /// * None - If the angle is not a [Param::Float] or the number of qubits exceeds 63.
    pub fn matrix(&self) -> Option<Array2<Complex64>> {
        let Param::Float(coeff) = self.angle else {
            // We cannot compute a matrix representation for a parameterized angle
            return None;
        };
        let x = ArrayView2::from_shape((1, self.x.len()), &self.x)
            .expect("1 x x.len() is a compatible shape");
        let z = ArrayView2::from_shape((1, self.z.len()), &self.z)
            .expect("1 x z.len() is a compatible shape");
        let phases = Array1::zeros(self.x.len());
        let coeffs = Array1::ones(1);

        let Ok(compressed) =
            MatrixCompressedPaulis::from_zx_arrays(x, z, phases.view(), coeffs.view())
        else {
            return None;
        };

        let mut out = c64(0.0, -(coeff / 2.0).sin()) * compressed.to_matrix_dense(false);
        let cos = c64((coeff / 2.0).cos(), 0.0);
        for i in 0..out.ncols() {
            out[(i, i)] += cos;
        }
        Some(out)
    }

    pub fn matrix_as_static_1q(&self) -> Option<[[Complex64; 2]; 2]> {
        if self.num_qubits() == 1 {
            let arr = self.matrix()?;
            Some([[arr[(0, 0)], arr[(0, 1)]], [arr[(1, 0)], arr[(1, 1)]]])
        } else {
            None
        }
    }

    pub fn matrix_as_static_2q(&self) -> Option<[[Complex64; 4]; 4]> {
        if self.num_qubits() == 2 {
            let arr = self.matrix()?;
            Some([
                [arr[[0, 0]], arr[[0, 1]], arr[[0, 2]], arr[[0, 3]]],
                [arr[[1, 0]], arr[[1, 1]], arr[[1, 2]], arr[[1, 3]]],
                [arr[[2, 0]], arr[[2, 1]], arr[[2, 2]], arr[[2, 3]]],
                [arr[[3, 0]], arr[[3, 1]], arr[[3, 2]], arr[[3, 3]]],
            ])
        } else {
            None
        }
    }
}

impl PartialEq for PauliProductRotation {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
            && self.z == other.z
            && self
                .angle
                .eq(&other.angle)
                .expect("Angles are float or symbol, for which eq is infallible")
    }
}

impl Eq for PauliProductRotation {}

/// This class represents a PauliProductMeasurement instruction.
#[derive(Clone, Debug)]
#[repr(align(8))]
pub struct PauliProductMeasurement {
    /// The z-component of the pauli.
    pub z: Vec<bool>,
    /// The x-component of the pauli.
    pub x: Vec<bool>,
    /// For a PauliProductMeasurement instruction, the phase of the Pauli can be either 0 or 2,
    /// where the value of 2 corresponds to a sign of `-1`.
    pub neg: bool,
}

impl Operation for PauliProductMeasurement {
    fn name(&self) -> &str {
        "pauli_product_measurement"
    }
    fn num_qubits(&self) -> u32 {
        self.z.len() as u32
    }
    fn num_clbits(&self) -> u32 {
        1
    }
    fn num_params(&self) -> u32 {
        0
    }
    fn directive(&self) -> bool {
        false
    }
}

impl PauliProductMeasurement {
    pub fn create_py_op(&self, py: Python, label: Option<&str>) -> PyResult<Py<PyAny>> {
        let z = self.z.to_pyarray(py);
        let x = self.x.to_pyarray(py);

        let phase = if self.neg { 2 } else { 0 };

        let py_label = if let Some(label) = label {
            label.into_py_any(py)?
        } else {
            py.None()
        };

        let gate = imports::PAULI_PRODUCT_MEASUREMENT
            .get_bound(py)
            .call_method1(intern!(py, "_from_pauli_data"), (z, x, phase, py_label))?;
        Ok(gate.unbind())
    }
}

impl PartialEq for PauliProductMeasurement {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.z == other.z && self.neg == other.neg
    }
}

impl Eq for PauliProductMeasurement {}

/// Private module with especific traits that allow for the implementation
/// of non dyn-compatible traits for [`CustomOperation`]. Namely [`PartialEq`]
/// and [`Clone`].
mod custom_traits {
    use crate::operations::CustomOperation;

    /// A trait which implements comparisons between [`CustomOperation`] instances.
    /// If the operation implements [`PartialEq`], this trait will be automatically implemented.
    /// Otherwise, the user is responsible for implementing this trait.
    #[diagnostic::on_unimplemented(
        message = "PartialEq is required to correctly implement CustomOperation on {Self}.",
        label = "This type needs an implementation of PartialEq",
        note = "Consider annotating {Self} with `#[derive(PartialEq)]`"
    )]
    pub trait ComparableOp {
        fn rich_eq(&self, other: &dyn CustomOperation) -> bool;
    }

    impl<Op: PartialEq + CustomOperation> ComparableOp for Op {
        fn rich_eq(&self, other: &dyn CustomOperation) -> bool {
            let Some(other) = other.downcast_ref() else {
                return false;
            };
            self.eq(other)
        }
    }

    /// A trait which implements dynamically cloning [`CustomOperation`] dyn objects.
    /// If the operation implements [`Clone`], this trait will be automatically implemented.
    /// Otherwise, the user is responsible for implementing [`Clone`].
    #[diagnostic::on_unimplemented(
        message = "Clone is required to correctly implement CustomOperation on {Self}.",
        label = "This type needs an implementation of Clone",
        note = "Consider annotating {Self} with `#[derive(Clone)]`"
    )]
    pub trait ClonableOp {
        fn clone_dyn(&self) -> Box<dyn CustomOperation>;
    }

    impl<Op: Clone + CustomOperation> ClonableOp for Op {
        fn clone_dyn(&self) -> Box<dyn CustomOperation> {
            Box::new(self.clone())
        }
    }
}

/// Trait that implements common methods found in operations that, in conjunction with
/// the [Operation] trait, allows a struct to operate in a circuit.
///
/// Unlike [Operation] alone, this trait focuses on the specific functions that are
/// typically available for the two main Qiskit operation types: Gate and Instruction.
///
/// To classify an operation, you must implement the [`CustomOperation::is_unitary`] method,
/// which returns a [`bool`] object with two variants:
/// - [`true`]: For unitary instructions.
///     - In addition to this, the implementor should define required methods for
///       the `Gate` to function properly:
///       - [`CustomOperation::matrix`]
///       - [`CustomOperation::num_ctrl_qubits`]
///       - [`CustomOperation::is_controlled_gate`]
/// - [`false`]: For non-unitary instruction.
///
/// This trait has an implicit requirement to [`PartialEq`] and [`Clone`] to allow for
/// comparison between opaque gates and dynamic cloning.
pub trait CustomOperation:
    Operation + Any + Debug + Send + Sync + ComparableOp + ClonableOp
{
    /// Return the custom label assigned to this instruction.
    fn label(&self) -> Option<&str> {
        None
    }

    /// Returns an inverted version of this instruction and the computed parameters.
    fn inverse(&self, _params: &[Param]) -> Option<(PackedOperation, SmallVec<[Param; 3]>)> {
        None
    }

    /// Returns a circuit representing the possible list of instructions that
    /// this operation is composed of.
    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        None
    }

    /// If the instance is a gate, returns the unitary matrix that represents it,
    /// if the parameters are correct. Otherwise, it returns None.
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        // TODO: Make fallible.
        None
    }

    /// If the instance is a gate, returns the number of control qubits.
    fn num_ctrl_qubits(&self) -> Option<NonZero<u32>> {
        None
    }

    /// If the instance is a gate, checks if it contains any control Qubits.
    fn is_controlled_gate(&self) -> bool {
        self.num_ctrl_qubits().is_some()
    }

    /// Returns whether the operation is based on a unitary matrix.
    fn is_unitary(&self) -> bool;
}

impl PartialEq for dyn CustomOperation {
    fn eq(&self, other: &Self) -> bool {
        ComparableOp::rich_eq(self, other)
    }
}

impl Clone for Box<dyn CustomOperation> {
    fn clone(&self) -> Self {
        self.clone_dyn()
    }
}

impl ToOwned for dyn CustomOperation {
    type Owned = Box<dyn CustomOperation>;

    fn to_owned(&self) -> Self::Owned {
        self.clone_dyn()
    }
}

impl dyn CustomOperation + 'static {
    /// Casts a reference to a CustomOperation to its original type if the correct
    /// type is specified.
    pub fn downcast_ref<T: CustomOperation + 'static>(&self) -> Option<&T> {
        let self_as_any: &dyn Any = self;
        self_as_any.downcast_ref()
    }
}

/// Internal representation of a custom operation within a Circuit.
///
/// It acts as a wrapper for a CustomOperation which ensures that
/// the operation is wrapped within a [`Box`] pointer and is aligned
/// to 8 bytes, which enables it to be safely represented as a
/// [`PackedOperation`].
#[derive(Debug)]
#[repr(align(8))]
pub(crate) struct BoxedCustomOperation(Box<dyn CustomOperation>);

impl Deref for BoxedCustomOperation {
    type Target = dyn CustomOperation;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl DerefMut for BoxedCustomOperation {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut()
    }
}

impl Clone for BoxedCustomOperation {
    fn clone(&self) -> Self {
        Self(self.0.clone_dyn())
    }
}

impl<T: CustomOperation> From<T> for BoxedCustomOperation {
    fn from(value: T) -> Self {
        Self(value.clone_dyn())
    }
}

impl From<Box<dyn CustomOperation>> for BoxedCustomOperation {
    fn from(value: Box<dyn CustomOperation>) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, arr2, linalg::kron};
    use qiskit_util::complex::{C_ONE, C_ZERO, IM};

    use crate::operations::{Param, PauliProductRotation};

    #[test]
    fn test_ppr_matrix() {
        // Prepare I X Y Z with some rotation angle
        let z = vec![false, false, true, true];
        let x = vec![false, true, true, false];
        let dim = 2usize.pow(x.len() as u32);
        let angle = -0.5;
        let ppr = PauliProductRotation {
            z,
            x,
            angle: Param::Float(angle),
        };

        let pauli_i = arr2(&[[C_ONE, C_ZERO], [C_ZERO, C_ONE]]);
        let pauli_x = arr2(&[[C_ZERO, C_ONE], [C_ONE, C_ZERO]]);
        let pauli_y = arr2(&[[C_ZERO, -IM], [IM, C_ZERO]]);
        let pauli_z = arr2(&[[C_ONE, C_ZERO], [C_ZERO, -C_ONE]]);

        let pauli_product = kron(&pauli_z, &kron(&pauli_y, &kron(&pauli_x, &pauli_i)));
        let expected_matrix =
            C_ONE * (angle / 2.).cos() * Array2::eye(dim) - IM * (angle / 2.).sin() * pauli_product;

        let matrix = ppr.matrix().unwrap();
        // Loosen the tolerance in Miri mode allows for larger roundoff errors
        // to mimic different hardware / OS configs, but keep 1e-17 for tight checks
        let epsilon = if cfg!(miri) { 1e-12 } else { 1e-17 };

        for i in 0..dim {
            for j in 0..dim {
                assert_abs_diff_eq!(expected_matrix[(i, j)], matrix[(i, j)], epsilon = epsilon);
            }
        }
    }
}

#[cfg(test)]
mod test_custom_operations {
    use crate::circuit_data::CircuitData;
    use crate::gate_matrix::{H_GATE, rz_gate};
    use crate::instruction::Parameters;
    use crate::operations::{CustomOperation, Operation, OperationRef, Param, StandardGate};
    use crate::packed_instruction::PackedOperation;
    use crate::{Clbit, Qubit};
    use ndarray::aview2;
    use smallvec::smallvec;
    use std::f64::consts::PI;

    macro_rules! impl_static_operation {
        ($ty:ident; $name:expr, $qubits:expr, $clbits:expr, $params:expr, $directive:expr) => {
            impl $crate::operations::Operation for $ty {
                fn name(&self) -> &str {
                    $name
                }
                fn num_qubits(&self) -> u32 {
                    $qubits
                }
                fn num_clbits(&self) -> u32 {
                    $clbits
                }
                fn num_params(&self) -> u32 {
                    $params
                }
                fn directive(&self) -> bool {
                    $directive
                }
            }
        };
    }

    /// HGate-like implementor
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct CustomH;
    impl_static_operation!(CustomH; "h", 1, 0, 0, false);

    impl CustomOperation for CustomH {
        fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
            CircuitData::from_standard_gates(
                1,
                [(StandardGate::H, smallvec![], smallvec![Qubit(0)])],
                0.0.into(),
            )
            .ok()
        }

        fn matrix(&self, params: &[Param]) -> Option<ndarray::Array2<numpy::Complex64>> {
            params.is_empty().then_some(aview2(&H_GATE).to_owned())
        }

        fn is_unitary(&self) -> bool {
            true
        }
    }

    /// Parameterized Z gate
    #[derive(Debug, Clone, PartialEq, Default)]
    struct ParametrizedAndLabeled {
        label: Option<String>,
    }
    impl_static_operation!(ParametrizedAndLabeled; "custom_rz", 1, 0, 1, false);
    impl ParametrizedAndLabeled {
        pub fn new<T: Into<String>>(label: Option<T>) -> Self {
            Self {
                label: label.map(Into::into),
            }
        }
    }
    impl CustomOperation for ParametrizedAndLabeled {
        fn is_unitary(&self) -> bool {
            true
        }

        fn matrix(&self, params: &[Param]) -> Option<ndarray::Array2<numpy::Complex64>> {
            match params {
                [Param::Float(theta)] => Some(aview2(&rz_gate(*theta)).to_owned()),
                _ => None,
            }
        }

        fn definition(&self, params: &[Param]) -> Option<CircuitData> {
            match params {
                [Param::Float(theta)] => CircuitData::from_standard_gates(
                    1,
                    [(
                        StandardGate::RZ,
                        smallvec![(*theta).into()],
                        smallvec![Qubit(0)],
                    )],
                    0.0.into(),
                )
                .ok(),
                _ => None,
            }
        }

        fn label(&self) -> Option<&str> {
            self.label.as_deref()
        }
    }

    /// Custom controlled gate
    #[derive(Debug, Copy, Clone, PartialEq)]
    struct Controlled;
    impl_static_operation!(Controlled; "controlled", 1, 0, 0, false);
    impl CustomOperation for Controlled {
        fn is_unitary(&self) -> bool {
            false
        }

        fn num_ctrl_qubits(&self) -> Option<std::num::NonZero<u32>> {
            std::num::NonZero::new(1)
        }
    }
    /// Custom implementation of measure
    #[derive(Debug, Copy, Clone, PartialEq)]
    struct Measure2;
    impl_static_operation!(Measure2; "measure", 2, 2, 0, false);
    impl CustomOperation for Measure2 {
        fn is_unitary(&self) -> bool {
            false
        }
    }

    /// Operation that can be reversed
    #[derive(Debug, Copy, Clone, PartialEq)]
    struct Reversible;
    impl_static_operation!(Reversible; "rev", 1, 0, 0, false);
    impl CustomOperation for Reversible {
        fn is_unitary(&self) -> bool {
            false
        }

        fn inverse(
            &self,
            params: &[Param],
        ) -> Option<(
            crate::packed_instruction::PackedOperation,
            smallvec::SmallVec<[Param; 3]>,
        )> {
            match params {
                [] => Some((
                    PackedOperation::from_custom_operation(Box::new(Self)),
                    smallvec![],
                )),
                _ => None,
            }
        }
    }

    /// Fully opaque gate with no matrix
    #[derive(Debug, Clone, PartialEq)]
    struct OpaqueGate;
    impl_static_operation!(OpaqueGate; "foo", 1, 0, 0, false);
    impl CustomOperation for OpaqueGate {
        fn is_unitary(&self) -> bool {
            true
        }
    }

    #[test]
    fn try_custom_h_gate() {
        let gate: Box<dyn CustomOperation> = Box::new(CustomH);

        // Try downcasting
        let gate = gate
            .downcast_ref::<CustomH>()
            .expect("Should downcast to an H gate");

        assert_eq!(gate.name(), "h");
        assert_eq!(gate.num_qubits(), 1);
        assert_eq!(gate.num_params(), 0);
        assert_eq!(gate.label(), None);

        assert!(gate.is_unitary());

        let matrix_res = gate.matrix(&[]);
        let matrix_exp = Some(aview2(&H_GATE));
        assert_eq!(matrix_res.as_ref().map(|mat| mat.view()), matrix_exp);

        let matrix_res = gate.matrix(&[Param::Float(PI)]);
        let matrix_exp = None;
        assert_eq!(matrix_res, matrix_exp,);

        let circuit = gate.definition(&[]).expect("Circuit should exist.");
        assert_eq!(circuit.len(), 1);

        let hgate = circuit.iter().next().expect("Should be H gate");
        assert_eq!(hgate.op.standard_gate(), StandardGate::H);
    }

    #[test]
    fn try_add_to_circuit() {
        let mut circuit = CircuitData::with_capacity(1, 0, 1, 0.0.into())
            .expect("Circuit with small capacity should be built.");
        let as_operation = PackedOperation::from_custom_operation(Box::new(CustomH));
        circuit
            .push_packed_operation(as_operation, None, &[Qubit(0)], &[])
            .expect("Instruction should be added to the circuit.");

        // Retrieve operation
        let retrieved_gate = &circuit.data()[0];

        let OperationRef::CustomOperation(gate_as_h) = retrieved_gate.op.view() else {
            panic!("Gate should be a custom gate of type CustomH");
        };

        let Some(downcast_gate) = gate_as_h.downcast_ref::<CustomH>() else {
            panic!("Gate should be a custom gate of type CustomH");
        };

        // Check that the retreived gate is still valid.
        assert_eq!(gate_as_h.num_qubits(), 1);
        assert!(gate_as_h.is_unitary());
        assert_eq!(gate_as_h.matrix(&[]), Some(aview2(&H_GATE).to_owned()));

        // Final instance equality check.
        assert_eq!(Some(&CustomH), Some(downcast_gate))
    }

    // Test a custom gate with varying labels.
    #[test]
    fn test_custom_gate_with_label() {
        let no_label = ParametrizedAndLabeled::default();
        let labeled = ParametrizedAndLabeled::new(Some("label"));

        // Make into boxed
        let boxed_no_label: Box<dyn CustomOperation> = Box::new(no_label.clone());
        let boxed_labeled: Box<dyn CustomOperation> = Box::new(labeled.clone());

        assert_ne!(&boxed_labeled, &boxed_no_label);
        assert_eq!(boxed_labeled.label(), labeled.label());
        assert_eq!(boxed_no_label.label(), no_label.label());

        // Try adding to circuit
        let mut circuit =
            CircuitData::with_capacity(2, 0, 2, 0.0.into()).expect("Empty circuit should work");
        circuit
            .push_packed_operation(
                PackedOperation::from_custom_operation(boxed_no_label),
                None,
                &[Qubit(0)],
                &[],
            )
            .expect("Operation should be added successfully");
        circuit
            .push_packed_operation(
                PackedOperation::from_custom_operation(boxed_labeled),
                None,
                &[Qubit(0)],
                &[],
            )
            .expect("Operation should be added successfully");

        // Test roundtrip
        let ops_ordered: [&dyn CustomOperation; 2] = [&no_label, &labeled];
        for (idx, op) in circuit.data().iter().enumerate() {
            let OperationRef::CustomOperation(op_ref) = op.op.view() else {
                panic!("Incorrect operation variant found in circuit!");
            };
            assert_eq!(op_ref, ops_ordered[idx]);
        }
    }

    /// Test comparison between custom operations
    #[test]
    fn test_gate_equality() {
        let custom_h = CustomH;
        let custom_h_as_dyn: Box<dyn CustomOperation> = Box::new(CustomH);
        let labeled = ParametrizedAndLabeled::new(Some("fee"));
        let labeled_fee = ParametrizedAndLabeled::new(Some("fee"));
        let labeled_fi = ParametrizedAndLabeled::new(Some("fi"));

        // identicals as opaques
        assert_eq!(&custom_h as &dyn CustomOperation, custom_h_as_dyn.as_ref());
        // two identicals
        assert_eq!(labeled, labeled_fee);
        // two non-identical
        assert_ne!(labeled, labeled_fi);
        // two different instances
        assert_ne!(
            custom_h_as_dyn.as_ref(),
            &labeled_fi as &dyn CustomOperation
        );
    }

    // Test dynamic cloning of operations with data within.
    #[test]
    fn test_clone_dyn() {
        let labeled_fee = ParametrizedAndLabeled::new(Some("fee"));
        let labeled_fi = ParametrizedAndLabeled::new(Some("fi"));

        // Try cloning as dyn refs using `ToOwned`
        let cloned_fee: Box<dyn CustomOperation> =
            (&labeled_fee as &dyn CustomOperation).to_owned();
        let cloned_fi: Box<dyn CustomOperation> = (&labeled_fi as &dyn CustomOperation).to_owned();

        assert_eq!(cloned_fee.as_ref(), &labeled_fee as &dyn CustomOperation);
        assert_eq!(cloned_fi.as_ref(), &labeled_fi as &dyn CustomOperation);

        // Check if label data is still the same.
        assert_eq!(cloned_fee.label(), labeled_fee.label());
        assert_eq!(cloned_fi.label(), labeled_fi.label());
    }

    // Test downcasting
    #[test]
    fn test_downcast() {
        let measure_boxed: Box<dyn CustomOperation> = Box::new(Measure2);
        let control_boxed: Box<dyn CustomOperation> = Box::new(Controlled);

        // Check if downcasting to the right type works
        assert_eq!(measure_boxed.downcast_ref::<Measure2>(), Some(&Measure2));
        assert_eq!(
            control_boxed.downcast_ref::<Controlled>(),
            Some(&Controlled)
        );

        // Check if downcasting to the wrong type doesn't work
        assert!(measure_boxed.downcast_ref::<Controlled>().is_none());
        assert!(control_boxed.downcast_ref::<Measure2>().is_none());
    }

    // Test parametrized matrix
    #[test]
    fn parameterized_gate_matrix() {
        let labeled_rz = ParametrizedAndLabeled::new(Some("rz"));
        let theta: Param = (PI / 4.0).into();

        let Some(matrix) = labeled_rz.matrix(&[theta]) else {
            panic!("Matrix should exist");
        };
        // Compare matrices
        assert!(approx::abs_diff_eq!(
            matrix,
            aview2(&rz_gate(PI / 4.0)),
            epsilon = 1e-12
        ));

        // Compare null case
        assert_eq!(labeled_rz.matrix(&[]), None,);
    }

    // Test inversed gate
    #[test]
    fn test_inverse() {
        let reversible = Reversible;

        // Retrieve the reverse, should be itself
        let Some((reversed, params)) = reversible.inverse(&[]) else {
            panic!("A reverse was not obtained")
        };

        // Parameters should be empty
        assert!(params.is_empty());

        let OperationRef::CustomOperation(roundtrip) = reversed.view() else {
            panic!("Obtained operation is not custom")
        };

        // Compare matrices
        assert_eq!(roundtrip.downcast_ref::<Reversible>(), Some(&Reversible));

        // Try with invalid params
        assert!(roundtrip.inverse(&[0.0.into()]).is_none());
    }

    // Adds all custom instructions to a Circuit
    #[test]
    fn test_multiple_custom_ops_in_circuit() {
        let h = CustomH;
        let rz = ParametrizedAndLabeled::new(Some("rz"));
        let reversible = Reversible;
        let meas = Measure2;
        let opaque = OpaqueGate;

        let ops: [&dyn CustomOperation; 5] = [&h, &rz, &reversible, &meas, &opaque];
        let params: [_; 5] = [
            None,
            Some(Parameters::Params(smallvec![Param::from(PI)])),
            None,
            None,
            None,
        ];

        let mut circuit = CircuitData::with_capacity(2, 2, 5, 0.0.into())
            .expect("Circuit creation should succeed.");
        for (op, params) in ops.iter().zip(params) {
            let qubits: Vec<Qubit> = (0..op.num_qubits()).map(Qubit).collect();
            let clbits: Vec<Clbit> = (0..op.num_clbits()).map(Clbit).collect();
            circuit
                .push_packed_operation(
                    PackedOperation::from_custom_operation((*op).to_owned()),
                    params,
                    &qubits,
                    &clbits,
                )
                .expect("Operation should be added to circuit.");
        }

        for (idx, op) in (0..circuit.len())
            .map(|idx| &circuit.data()[idx])
            .enumerate()
        {
            let OperationRef::CustomOperation(comparison) = op.op.view() else {
                panic!("Non-custom operation found")
            };

            // Check that each instance is the same.
            assert_eq!(comparison, ops[idx]);
        }
    }

    // Tests that `OperationRef` delegates each function call correctly for
    // the `Operation` trait when it refers to a custom operation.
    #[test]
    fn test_operation_ref_delegates_correctly() {
        let h = CustomH;
        let rz = ParametrizedAndLabeled::new(Some("rz"));
        let reversible = Reversible;
        let meas = Measure2;
        let opaque = OpaqueGate;

        let ops: [&dyn CustomOperation; 5] = [&h, &rz, &reversible, &meas, &opaque];
        let packed_ops: Vec<PackedOperation> = ops
            .iter()
            .map(|op| PackedOperation::from_custom_operation((*op).to_owned()))
            .collect();

        for (op, packed) in ops.iter().zip(&packed_ops) {
            let view = packed.view();
            assert_eq!(op.name(), view.name());
            assert_eq!(op.num_qubits(), view.num_qubits());
            assert_eq!(op.num_clbits(), view.num_clbits());
            assert_eq!(op.num_params(), view.num_params());
            assert_eq!(op.directive(), view.directive());
        }
    }
}
