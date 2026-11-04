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

use crate::classical::expr::{Expr, ExprKind, PyExpr};
use crate::classical::types::Type;
use crate::imports;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{IntoPyObjectExt, intern};

/// A unary expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Unary {
    pub op: UnaryOp,
    pub operand: Expr,
    pub ty: Type,
    pub constant: bool,
}

/// The Rust-side enum indicating a [Unary] expression's kind.
///
/// The values are part of the public Qiskit Python interface, since
/// they are public in the sister Python enum `_UnaryOp` in `expr.py`
/// and used in our QPY serialization format.
///
/// WARNING: If you add more, **be sure to update expr.py** as well
/// as the implementation of [::bytemuck::CheckedBitPattern]
/// below.
#[repr(u8)]
#[derive(Copy, Hash, Clone, Debug, PartialEq)]
pub enum UnaryOp {
    BitNot = 1,
    LogicNot = 2,
    Negate = 3,
}

unsafe impl ::bytemuck::CheckedBitPattern for UnaryOp {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits > 0 && *bits < 4
    }
}

impl UnaryOp {
    pub fn try_from_u8(value: u8) -> Result<UnaryOp, bytemuck::checked::CheckedCastError> {
        bytemuck::checked::try_cast::<u8, UnaryOp>(value)
    }
}

impl<'py> IntoPyObject<'py> for Unary {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyUnary(self), PyExpr(ExprKind::Unary)))?.into_any())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Unary {
    type Error = <PyUnary as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let PyUnary(u) = ob.extract()?;
        Ok(u)
    }
}

impl<'py> IntoPyObject<'py> for UnaryOp {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        imports::UNARY_OP.get_bound(py).call1((self as usize,))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for UnaryOp {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let value = ob.getattr(intern!(ob.py(), "value"))?;
        bytemuck::checked::try_cast(value.extract::<u8>()?)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// A Python descriptor to prevent PyO3 from attempting to import the Python-side
/// enum before we're initialized.
#[pyclass(module = "qiskit._accelerate.circuit.classical.expr")]
struct PyUnaryOp;

#[pymethods]
impl PyUnaryOp {
    fn __get__(&self, obj: &Bound<PyAny>, _obj_type: &Bound<PyAny>) -> Py<PyAny> {
        imports::UNARY_OP.get_bound(obj.py()).clone().unbind()
    }
}

/// A unary expression.
///
/// Args:
///     op: The opcode describing which operation is being done.
///     operand: The operand of the operation.
///     type: The resolved type of the result.
#[pyclass(
    eq,
    extends = PyExpr,
    name = "Unary",
    module = "qiskit._accelerate.circuit.classical.expr",
    from_py_object
)]
#[derive(PartialEq, Clone, Debug)]
pub struct PyUnary(Unary);

#[pymethods]
impl PyUnary {
    // The docstring for 'Op' is defined in Python (expr.py).
    #[classattr]
    #[allow(non_snake_case)]
    fn Op(py: Python) -> PyResult<Py<PyAny>> {
        PyUnaryOp.into_py_any(py)
    }

    #[new]
    #[pyo3(text_signature = "(op, operand, type)")]
    fn new(py: Python, op: UnaryOp, operand: Expr, ty: Type) -> PyResult<Py<Self>> {
        let constant = operand.is_const();
        Py::new(
            py,
            (
                PyUnary(Unary {
                    op,
                    operand,
                    ty,
                    constant,
                }),
                PyExpr(ExprKind::Unary),
            ),
        )
    }

    #[getter]
    fn get_op(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.op.into_py_any(py)
    }

    #[getter]
    fn get_operand(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.operand.clone().into_py_any(py)
    }

    #[getter]
    fn get_const(&self) -> bool {
        self.0.constant
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.into_py_any(py)
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_unary"), (slf,))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (self.get_op(py)?, self.get_operand(py)?, self.get_type(py)?),
        )
            .into_pyobject(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "Unary({}, {}, {})",
            self.get_op(py)?.bind(py).repr()?,
            self.get_operand(py)?.bind(py).repr()?,
            self.get_type(py)?.bind(py).repr()?,
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_unary_op_from_u8_invalid_input() {
        assert!(UnaryOp::try_from_u8(123).is_err())
    }
}
