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
use crate::imports::UUID;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyTuple};
use pyo3::{IntoPyObjectExt, intern};
use uuid::Uuid;

/// A stretch variable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stretch {
    pub uuid: u128,
    pub name: String,
}

impl<'py> IntoPyObject<'py> for Stretch {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyStretch(self), PyExpr(ExprKind::Stretch)))?.into_any())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Stretch {
    type Error = <PyStretch as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let PyStretch(s) = ob.extract()?;
        Ok(s)
    }
}

/// A stretch variable.
///
/// In general, construction of stretch variables for use in programs should use :meth:`Stretch.new`
/// or :meth:`.QuantumCircuit.add_stretch`
#[pyclass(
    eq,
    hash,
    frozen,
    extends = PyExpr,
    name = "Stretch",
    module = "qiskit._accelerate.circuit.classical.expr",
    from_py_object
)]
#[derive(PartialEq, Clone, Debug, Hash)]
pub struct PyStretch(Stretch);

#[pymethods]
impl PyStretch {
    #[new]
    fn py_new(var: &Bound<PyAny>, name: String) -> PyResult<Py<Self>> {
        Py::new(
            var.py(),
            (
                PyStretch(Stretch {
                    uuid: var.getattr(intern!(var.py(), "int"))?.extract()?,
                    name,
                }),
                PyExpr(ExprKind::Stretch),
            ),
        )
    }

    /// Generate a new named stretch variable.
    #[classmethod]
    fn new(cls: &Bound<'_, pyo3::types::PyType>, name: String) -> PyResult<Py<Self>> {
        Py::new(
            cls.py(),
            (
                PyStretch(Stretch {
                    uuid: Uuid::new_v4().as_u128(),
                    name,
                }),
                PyExpr(ExprKind::Stretch),
            ),
        )
    }

    /// A :class:`~uuid.UUID` to uniquely identify this stretch.
    #[getter]
    fn get_var(&self, py: Python) -> PyResult<Py<PyAny>> {
        let kwargs = [("int", self.0.uuid)].into_py_dict(py)?;
        Ok(UUID.get_bound(py).call((), Some(&kwargs))?.unbind())
    }

    /// The name of the stretch variable.
    #[getter]
    fn get_name(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.name.clone().into_py_any(py)
    }

    #[getter]
    fn get_const(&self) -> bool {
        true
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        Type::Duration.into_py_any(py)
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_stretch"), (slf,))
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // I am immutable...
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // ... as are all my constituent parts.
        slf
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (self.get_var(py)?, self.get_name(py)?),
        )
            .into_pyobject(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "Stretch({}, '{}')",
            self.get_var(py)?.bind(py).repr()?,
            self.get_name(py)?
        ))
    }
}
