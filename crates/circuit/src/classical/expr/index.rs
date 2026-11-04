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

/// An indexing expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Index {
    pub target: Expr,
    pub index: Expr,
    pub ty: Type,
    pub constant: bool,
}

impl<'py> IntoPyObject<'py> for Index {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyIndex(self), PyExpr(ExprKind::Index)))?.into_any())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Index {
    type Error = <PyIndex as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let PyIndex(i) = ob.extract()?;
        Ok(i)
    }
}

/// An indexing expression.
///
/// Args:
///     target: The object being indexed.
///     index: The expression doing the indexing.
///     type: The resolved type of the result.
#[pyclass(
    eq,
    extends = PyExpr,
    name = "Index",
    module = "qiskit._accelerate.circuit.classical.expr",
    from_py_object,
)]
#[derive(PartialEq, Clone, Debug)]
pub struct PyIndex(Index);

#[pymethods]
impl PyIndex {
    #[new]
    #[pyo3(text_signature = "(target, index, type)")]
    fn new(py: Python, target: Expr, index: Expr, ty: Type) -> PyResult<Py<Self>> {
        let constant = target.is_const() && index.is_const();
        Py::new(
            py,
            (
                PyIndex(Index {
                    target,
                    index,
                    ty,
                    constant,
                }),
                PyExpr(ExprKind::Index),
            ),
        )
    }

    #[getter]
    fn get_target(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.target.clone().into_py_any(py)
    }

    #[getter]
    fn get_index(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.index.clone().into_py_any(py)
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
        visitor.call_method1(intern!(visitor.py(), "visit_index"), (slf,))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (
                self.get_target(py)?,
                self.get_index(py)?,
                self.get_type(py)?,
            ),
        )
            .into_pyobject(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "Index({}, {}, {})",
            self.get_target(py)?.bind(py).repr()?,
            self.get_index(py)?.bind(py).repr()?,
            self.get_type(py)?.bind(py).repr()?,
        ))
    }
}
