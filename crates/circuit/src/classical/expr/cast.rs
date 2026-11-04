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
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{IntoPyObjectExt, intern};

/// A cast from one type to another, implied by the use of an expression in a different
/// context.
#[derive(Clone, Debug, PartialEq)]
pub struct Cast {
    pub operand: Expr,
    pub ty: Type,
    pub constant: bool,
    pub implicit: bool,
}

impl<'py> IntoPyObject<'py> for Cast {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyCast(self), PyExpr(ExprKind::Cast)))?.into_any())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Cast {
    type Error = <PyCast as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let PyCast(c) = ob.extract()?;
        Ok(c)
    }
}

/// A cast from one type to another, implied by the use of an expression in a different
/// context.
#[pyclass(
    eq,
    extends = PyExpr,
    name = "Cast",
    module = "qiskit._accelerate.circuit.classical.expr",
    from_py_object
)]
#[derive(PartialEq, Clone, Debug)]
pub struct PyCast(Cast);

#[pymethods]
impl PyCast {
    #[new]
    #[pyo3(signature=(operand, ty, implicit=false), text_signature="(operand, type, implicit=False)")]
    fn new(py: Python, operand: Expr, ty: Type, implicit: bool) -> PyResult<Py<Self>> {
        let constant = operand.is_const();
        Py::new(
            py,
            (
                PyCast(Cast {
                    operand,
                    ty,
                    constant,
                    implicit,
                }),
                PyExpr(ExprKind::Cast),
            ),
        )
    }

    #[getter]
    fn get_operand(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.operand.clone().into_py_any(py)
    }

    #[getter]
    fn get_implicit(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.implicit.into_py_any(py)
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
        visitor.call_method1(intern!(visitor.py(), "visit_cast"), (slf,))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (
                self.get_operand(py)?,
                self.get_type(py)?,
                self.get_implicit(py)?,
            ),
        )
            .into_pyobject(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "Cast({}, {}, implicit={})",
            self.get_operand(py)?.bind(py).repr()?,
            self.get_type(py)?.bind(py).repr()?,
            self.get_implicit(py)?.bind(py).repr()?,
        ))
    }
}
