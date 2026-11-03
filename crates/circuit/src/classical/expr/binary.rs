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

/// A binary expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Binary {
    pub op: BinaryOp,
    pub left: Expr,
    pub right: Expr,
    pub ty: Type,
    pub constant: bool,
}

/// The Rust-side enum indicating a [Binary] expression's kind.
///
/// The values are part of the public Qiskit Python interface, since
/// they are public in the sister Python enum `_BinaryOp` in `expr.py`
/// and used in our QPY serialization format.
///
/// WARNING: If you add more, **be sure to update expr.py** as well
/// as the implementation of [::bytemuck::CheckedBitPattern]
/// below.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    BitAnd = 1,
    BitOr = 2,
    BitXor = 3,
    LogicAnd = 4,
    LogicOr = 5,
    Equal = 6,
    NotEqual = 7,
    Less = 8,
    LessEqual = 9,
    Greater = 10,
    GreaterEqual = 11,
    ShiftLeft = 12,
    ShiftRight = 13,
    Add = 14,
    Sub = 15,
    Mul = 16,
    Div = 17,
}

unsafe impl ::bytemuck::CheckedBitPattern for BinaryOp {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits > 0 && *bits < 18
    }
}

impl BinaryOp {
    pub fn try_from_u8(value: u8) -> Result<BinaryOp, bytemuck::checked::CheckedCastError> {
        bytemuck::checked::try_cast::<u8, BinaryOp>(value)
    }
}

impl<'py> IntoPyObject<'py> for Binary {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyBinary(self), PyExpr(ExprKind::Binary)))?.into_any())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Binary {
    type Error = <PyBinary as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let PyBinary(b) = ob.extract()?;
        Ok(b)
    }
}

impl<'py> IntoPyObject<'py> for BinaryOp {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        imports::BINARY_OP.get_bound(py).call1((self as usize,))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for BinaryOp {
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
struct PyBinaryOp;

#[pymethods]
impl PyBinaryOp {
    fn __get__(&self, obj: &Bound<PyAny>, _obj_type: &Bound<PyAny>) -> Py<PyAny> {
        imports::BINARY_OP.get_bound(obj.py()).clone().unbind()
    }
}

/// A binary expression.
///
/// Args:
///     op: The opcode describing which operation is being done.
///     left: The left-hand operand.
///     right: The right-hand operand.
///     type: The resolved type of the result.
#[pyclass(
    eq,
    extends = PyExpr,
    name = "Binary",
    module = "qiskit._accelerate.circuit.classical.expr",
    from_py_object
)]
#[derive(PartialEq, Clone, Debug)]
pub struct PyBinary(Binary);

#[pymethods]
impl PyBinary {
    // The docstring for 'Op' is defined in Python (expr.py).
    #[classattr]
    #[allow(non_snake_case)]
    fn Op(py: Python) -> PyResult<Py<PyAny>> {
        PyBinaryOp.into_py_any(py)
    }

    #[new]
    #[pyo3(text_signature = "(op, left, right, type)")]
    fn new(py: Python, op: BinaryOp, left: Expr, right: Expr, ty: Type) -> PyResult<Py<Self>> {
        let constant = left.is_const() && right.is_const();
        Py::new(
            py,
            (
                PyBinary(Binary {
                    op,
                    left,
                    right,
                    ty,
                    constant,
                }),
                PyExpr(ExprKind::Binary),
            ),
        )
    }

    #[getter]
    fn get_op(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.op.into_py_any(py)
    }

    #[getter]
    fn get_left(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.left.clone().into_py_any(py)
    }

    #[getter]
    fn get_right(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.right.clone().into_py_any(py)
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
        visitor.call_method1(intern!(visitor.py(), "visit_binary"), (slf,))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (
                self.get_op(py)?,
                self.get_left(py)?,
                self.get_right(py)?,
                self.get_type(py)?,
            ),
        )
            .into_pyobject(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "Binary({}, {}, {}, {})",
            self.get_op(py)?.bind(py).repr()?,
            self.get_left(py)?.bind(py).repr()?,
            self.get_right(py)?.bind(py).repr()?,
            self.get_type(py)?.bind(py).repr()?,
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_binary_op_from_u8_invalid_input() {
        assert!(BinaryOp::try_from_u8(123).is_err())
    }
}
