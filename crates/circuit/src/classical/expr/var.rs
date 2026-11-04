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

use crate::bit::{ClassicalRegister, ShareableClbit};
use crate::classical::expr::{ExprKind, PyExpr};
use crate::classical::types::Type;
use crate::imports::UUID;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyTuple};
use pyo3::{IntoPyObjectExt, intern};
use uuid::Uuid;

/// A classical variable expression.
///
/// Note that the type of variant [Var::Bit] is always assumed to be a bool,
/// so we don't store it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Var {
    Standalone {
        uuid: u128,
        name: String,
        ty: Type,
    },
    Bit {
        bit: ShareableClbit,
    },
    Register {
        register: ClassicalRegister,
        ty: Type,
    },
}

impl Var {
    pub fn ty(&self) -> Type {
        match self {
            Var::Standalone { ty, .. } => *ty,
            Var::Bit { .. } => Type::Bool,
            Var::Register { ty, .. } => *ty,
        }
    }
}

impl<'py> IntoPyObject<'py> for Var {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyVar(self), PyExpr(ExprKind::Var)))?.into_any())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Var {
    type Error = <PyVar as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let PyVar(v) = ob.extract()?;
        Ok(v)
    }
}

/// A classical variable.
///
/// These variables take two forms: a new-style variable that owns its storage location and has an
/// associated name; and an old-style variable that wraps a :class:`.Clbit` or
/// :class:`.ClassicalRegister` instance that is owned by some containing circuit.  In general,
/// construction of variables for use in programs should use :meth:`Var.new` or
/// :meth:`.QuantumCircuit.add_var`.
///
/// Variables are immutable after construction, so they can be used as dictionary keys.
#[pyclass(
    eq,
    hash,
    frozen,
    extends = PyExpr,
    name = "Var",
    module = "qiskit._accelerate.circuit.classical.expr",
    from_py_object
)]
#[derive(PartialEq, Clone, Debug, Hash)]
pub struct PyVar(Var);

#[pymethods]
impl PyVar {
    #[new]
    #[pyo3(signature = (var, ty, *, name = None), text_signature="(var, type, *, name=None)")]
    fn py_new(var: &Bound<PyAny>, ty: Type, name: Option<String>) -> PyResult<Py<Self>> {
        let v = if let Some(name) = name {
            Var::Standalone {
                uuid: var.getattr(intern!(var.py(), "int"))?.extract()?,
                name,
                ty,
            }
        } else if let Ok(register) = var.extract::<ClassicalRegister>() {
            Var::Register { register, ty }
        } else {
            Var::Bit {
                bit: var.extract()?,
            }
        };
        Py::new(var.py(), (PyVar(v), PyExpr(ExprKind::Var)))
    }

    /// Generate a new named variable that owns its own backing storage.
    #[classmethod]
    fn new(cls: &Bound<'_, pyo3::types::PyType>, name: String, ty: Type) -> PyResult<Py<Self>> {
        Py::new(
            cls.py(),
            (
                PyVar(Var::Standalone {
                    uuid: Uuid::new_v4().as_u128(),
                    name,
                    ty,
                }),
                PyExpr(ExprKind::Var),
            ),
        )
    }

    #[getter]
    fn get_const(&self) -> bool {
        false
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self.0 {
            Var::Bit { .. } => Type::Bool.into_py_any(py),
            Var::Standalone { ty, .. } | Var::Register { ty, .. } => ty.into_py_any(py),
        }
    }

    /// A reference to the backing data storage of the :class:`Var` instance.  When lifting
    /// old-style :class:`.Clbit` or :class:`.ClassicalRegister` instances into a :class:`Var`,
    /// this is exactly the :class:`.Clbit` or :class:`.ClassicalRegister`.  If the variable is a
    /// new-style classical variable (one that owns its own storage separate to the old
    /// :class:`.Clbit`/:class:`.ClassicalRegister` model), this field will be a :class:`~uuid.UUID`
    /// to uniquely identify it.
    #[getter]
    fn get_var(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.0 {
            Var::Standalone { uuid, .. } => {
                let kwargs = [("int", uuid)].into_py_dict(py)?;
                Ok(UUID.get_bound(py).call((), Some(&kwargs))?.unbind())
            }
            Var::Bit { bit } => bit.into_py_any(py),
            Var::Register { register, .. } => register.into_py_any(py),
        }
    }

    /// The name of the variable.  This is required to exist if the backing :attr:`var` attribute
    /// is a :class:`~uuid.UUID`, i.e. if it is a new-style variable, and must be ``None`` if it is
    /// an old-style variable.
    #[getter]
    fn get_name(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.0 {
            Var::Standalone { name, .. } => Some(name),
            Var::Bit { .. } | Var::Register { .. } => None,
        }
        .into_py_any(py)
    }

    /// Whether this :class:`Var` is a standalone variable that owns its storage
    /// location, if applicable. If false, this is a wrapper :class:`Var` around a
    /// pre-existing circuit object.
    #[getter]
    fn get_standalone(&self) -> bool {
        match self.0 {
            Var::Standalone { .. } => true,
            Var::Bit { .. } | Var::Register { .. } => false,
        }
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_var"), (slf,))
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // I am immutable...
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // ... as are all my constituent parts.
        slf
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        if matches!(self.0, Var::Bit { .. } | Var::Register { .. }) {
            return Ok(format!(
                "Var({}, {})",
                self.get_var(py)?.bind(py).repr()?,
                self.get_type(py)?.bind(py).repr()?,
            ));
        };
        Ok(format!(
            "Var({}, {}, name='{}')",
            self.get_var(py)?.bind(py).repr()?,
            self.get_type(py)?.bind(py).repr()?,
            self.get_name(py)?
        ))
    }

    // This is needed to handle the 'name' kwarg for __reduce__.
    #[classmethod]
    fn _from_pickle(
        _cls: &Bound<'_, pyo3::types::PyType>,
        var: &Bound<PyAny>,
        ty: Type,
        name: &Bound<PyAny>,
    ) -> PyResult<Py<Self>> {
        PyVar::py_new(var, ty, name.extract()?)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>().getattr(intern!(py, "_from_pickle"))?,
            (self.get_var(py)?, self.get_type(py)?, self.get_name(py)?),
        )
            .into_pyobject(py)
    }
}
