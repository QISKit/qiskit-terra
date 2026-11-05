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

use pyo3::PyTypeInfo;
use pyo3::exceptions::PyAttributeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyTuple;

static BOOL_TYPE: PyOnceLock<Py<PyBool>> = PyOnceLock::new();
static DURATION_TYPE: PyOnceLock<Py<PyDuration>> = PyOnceLock::new();
static FLOAT_TYPE: PyOnceLock<Py<PyFloat>> = PyOnceLock::new();

/// A classical expression's "type".
///
/// This is the only struct that Rust code should be using when working with classical expression
/// types. Everything else in this file is to support our Python API, and is intentionally
/// private.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Type {
    Bool,
    Duration,
    Float,
    Uint(u32),
}

impl<'py> IntoPyObject<'py> for Type {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Type::Bool => Ok(PyBool::new(py).into_bound(py).into_any()),
            Type::Duration => Ok(PyDuration::new(py).into_bound(py).into_any()),
            Type::Float => Ok(PyFloat::new(py).into_bound(py).into_any()),
            Type::Uint(n) => Ok(PyUint::new(py, n).into_bound(py).into_any()),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Type {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let PyType(kind) = ob.extract()?;
        Ok(match kind {
            TypeKind::Bool => Type::Bool,
            TypeKind::Duration => Type::Duration,
            TypeKind::Float => Type::Float,
            TypeKind::Uint => {
                let PyUint(n) = ob.extract()?;
                Type::Uint(n)
            }
        })
    }
}

/// Root base class of all nodes in the type tree.  The base case should never be instantiated
/// directly.
///
/// This must not be subclassed by users; subclasses form the internal data of the representation
/// of expressions, and it does not make sense to add more outside of Qiskit library code.
#[pyclass(
    eq,
    hash,
    subclass,
    frozen,
    name = "Type",
    module = "qiskit._accelerate.circuit.classical.types",
    from_py_object
)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyType(TypeKind);

#[pymethods]
impl PyType {
    /// Get the kind of this type.
    ///
    /// This is exactly equal to the Python type object that defines
    /// this type, that is ``t.kind is type(t)``, but is exposed like this to make it clear that
    /// this a hashable enum-like discriminator you can rely on.
    #[getter]
    fn get_kind(&self, py: Python) -> Py<PyAny> {
        match self.0 {
            TypeKind::Bool => PyBool::type_object(py).into_any().unbind(),
            TypeKind::Duration => PyDuration::type_object(py).into_any().unbind(),
            TypeKind::Float => PyFloat::type_object(py).into_any().unbind(),
            TypeKind::Uint => PyUint::type_object(py).into_any().unbind(),
        }
    }

    fn __setattr__(&self, _key: Bound<PyAny>, _value: Bound<PyAny>) -> PyResult<()> {
        Err(PyAttributeError::new_err(format!(
            "'{:?}' instances are immutable",
            self.0
        )))
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        slf
    }
}

#[repr(u8)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
enum TypeKind {
    Bool,
    Duration,
    Float,
    Uint,
}

/// The Boolean type.  This has exactly two values: ``True`` and ``False``.
#[pyclass(
    eq,
    hash,
    extends = PyType,
    frozen,
    name = "Bool",
    module = "qiskit._accelerate.circuit.classical.types",
    from_py_object
)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyBool;

#[pymethods]
impl PyBool {
    #[new]
    fn new(py: Python) -> Py<Self> {
        BOOL_TYPE
            .get_or_init(py, || {
                Py::new(py, (PyBool, PyType(TypeKind::Bool))).unwrap()
            })
            .clone_ref(py)
    }

    fn __repr__(&self) -> &str {
        "Bool()"
    }

    fn __reduce__<'py>(_slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (py.get_type::<Self>(), ()).into_pyobject(py)
    }
}

/// A length of time, possibly negative.
#[pyclass(
    eq,
    hash,
    extends = PyType,
    frozen,
    name = "Duration",
    module = "qiskit._accelerate.circuit.classical.types",
    from_py_object
)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyDuration;

#[pymethods]
impl PyDuration {
    #[new]
    fn new(py: Python) -> Py<Self> {
        DURATION_TYPE
            .get_or_init(py, || {
                Py::new(py, (PyDuration, PyType(TypeKind::Duration))).unwrap()
            })
            .clone_ref(py)
    }

    fn __repr__(&self) -> &str {
        "Duration()"
    }

    fn __reduce__<'py>(_slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (py.get_type::<Self>(), ()).into_pyobject(py)
    }
}

/// An IEEE-754 double-precision floating point number.
///
/// In the future, this may also be used to represent other fixed-width floats.
#[pyclass(
    eq,
    hash,
    extends = PyType,
    frozen,
    name = "Float",
    module = "qiskit._accelerate.circuit.classical.types",
    from_py_object
)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyFloat;

#[pymethods]
impl PyFloat {
    #[new]
    fn new(py: Python) -> Py<Self> {
        FLOAT_TYPE
            .get_or_init(py, || {
                Py::new(py, (PyFloat, PyType(TypeKind::Float))).unwrap()
            })
            .clone_ref(py)
    }

    fn __repr__(&self) -> &str {
        "Float()"
    }

    fn __reduce__<'py>(_slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (py.get_type::<Self>(), ()).into_pyobject(py)
    }
}

/// An unsigned integer of fixed bit width.
#[pyclass(
    eq,
    hash,
    extends = PyType,
    frozen,
    name = "Uint",
    module = "qiskit._accelerate.circuit.classical.types",
    from_py_object
)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyUint(u32);

#[pymethods]
impl PyUint {
    #[new]
    fn new(py: Python, width: u32) -> Py<Self> {
        Py::new(py, (PyUint(width), PyType(TypeKind::Uint))).unwrap()
    }

    #[getter]
    fn get_width(&self) -> u32 {
        self.0
    }

    fn __repr__(&self) -> String {
        format!("Uint({})", self.0)
    }

    fn __reduce__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (py.get_type::<Self>(), (slf.0,)).into_pyobject(py)
    }
}

pub(crate) fn register_python(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyType>()?;
    m.add_class::<PyBool>()?;
    m.add_class::<PyDuration>()?;
    m.add_class::<PyFloat>()?;
    m.add_class::<PyUint>()?;
    Ok(())
}
