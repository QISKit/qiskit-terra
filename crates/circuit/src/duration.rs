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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// A length of time used to express circuit timing.
///
/// It defines a group of classes which are all subclasses of itself (functionally, an
/// enumeration carrying data).
///
/// You can use it in a match statement::
///
///   match duration:
///      case Duration.dt(dt):
///          return dt
///      case Duration.s(seconds):
///          return seconds / 5e-7
///      case _:
///          raise ValueError("expected dt or seconds")
///
/// You can also use :meth:`value` and :meth:`unit` to get the information separately.
#[pyclass(eq, module = "qiskit._accelerate.circuit", from_py_object)]
#[derive(PartialEq, Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub enum Duration {
    dt(i64),
    ps(f64),
    ns(f64),
    us(f64),
    ms(f64),
    s(f64),
}

#[pymethods]
impl Duration {
    #[new]
    fn py_new(unit: &str, value: Bound<PyAny>) -> PyResult<Self> {
        match unit {
            "dt" => value.extract().map(Self::dt),
            "ps" => value.extract().map(Self::ps),
            "ns" => value.extract().map(Self::ns),
            "us" => value.extract().map(Self::us),
            "ms" => value.extract().map(Self::ms),
            "s" => value.extract().map(Self::s),
            _ => Err(PyValueError::new_err(format!("unknown unit: {unit}"))),
        }
    }

    /// The corresponding ``unit`` of the duration.
    pub fn unit(&self) -> &'static str {
        match self {
            Duration::dt(_) => "dt",
            Duration::ps(_) => "ps",
            Duration::us(_) => "us",
            Duration::ns(_) => "ns",
            Duration::ms(_) => "ms",
            Duration::s(_) => "s",
        }
    }

    /// The ``value`` of the duration.
    ///
    /// This will be a Python ``int`` if the :meth:`~Duration.unit` is ``"dt"``,
    /// else a ``float``.
    #[pyo3(name = "value")]
    pub fn py_value<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match *self {
            Duration::dt(v) => {
                let Ok(v) = v.into_pyobject(py);
                v.into_any()
            }
            Duration::ps(v)
            | Duration::us(v)
            | Duration::ns(v)
            | Duration::ms(v)
            | Duration::s(v) => {
                let Ok(v) = v.into_pyobject(py);
                v.into_any()
            }
        }
    }

    fn __repr__(&self) -> String {
        match self {
            Duration::ps(t) => format!("Duration.ps({t})"),
            Duration::ns(t) => format!("Duration.ns({t})"),
            Duration::us(t) => format!("Duration.us({t})"),
            Duration::ms(t) => format!("Duration.ms({t})"),
            Duration::s(t) => format!("Duration.s({t})"),
            Duration::dt(t) => format!("Duration.dt({t})"),
        }
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (py.get_type::<Duration>(), (self.unit(), self.py_value(py))).into_pyobject(py)
    }
}
