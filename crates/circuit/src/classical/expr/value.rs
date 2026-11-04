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

use crate::classical::expr::{ExprKind, PyExpr};
use crate::classical::types::Type;
use crate::duration::Duration;
use num_bigint::BigUint;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{IntoPyObjectExt, intern};

/// A single scalar value expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Duration(Duration),
    Float { raw: f64, ty: Type },
    Uint { raw: BigUint, ty: Type },
}

impl<'py> IntoPyObject<'py> for Value {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyValue(self), PyExpr(ExprKind::Value)))?.into_any())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Value {
    type Error = <PyValue as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let PyValue(v) = ob.extract()?;
        Ok(v)
    }
}

/// A single scalar value.
#[pyclass(
    eq,
    extends = PyExpr,
    name = "Value",
    module = "qiskit._accelerate.circuit.classical.expr",
    from_py_object
)]
#[derive(PartialEq, Clone, Debug)]
pub struct PyValue(Value);

#[pymethods]
impl PyValue {
    #[new]
    #[pyo3(text_signature = "(value, type)")]
    fn new(py: Python, value: Bound<PyAny>, ty: Type) -> PyResult<Py<Self>> {
        let value = if let Ok(raw) = value.extract::<BigUint>() {
            Value::Uint { raw, ty }
        } else if let Ok(raw) = value.extract::<f64>() {
            Value::Float { raw, ty }
        } else {
            Value::Duration(value.extract()?)
        };
        Py::new(py, (PyValue(value), PyExpr(ExprKind::Value)))
    }

    #[getter]
    fn get_value(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.0 {
            Value::Duration(d) => d.into_py_any(py),
            Value::Float { raw, .. } => raw.into_py_any(py),
            Value::Uint { raw, .. } => raw.into_py_any(py),
        }
    }

    #[getter]
    fn get_const(&self) -> bool {
        true
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.0 {
            Value::Duration(_) => Type::Duration.into_py_any(py),
            Value::Float { ty, .. } | Value::Uint { ty, .. } => ty.into_py_any(py),
        }
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_value"), (slf,))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (self.get_value(py)?, self.get_type(py)?),
        )
            .into_pyobject(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "Value({}, {})",
            self.get_value(py)?.bind(py).repr()?,
            self.get_type(py)?.bind(py).repr()?,
        ))
    }
}
