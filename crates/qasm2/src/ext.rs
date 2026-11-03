// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#[cfg(feature = "py")]
use pyo3::prelude::*;
use std::sync::Arc;

use crate::error::ParseError;

/// Information about a custom instruction that Python space is able to construct to pass down to
/// us.
#[cfg_attr(feature = "py", pyclass(from_py_object))]
#[derive(Clone)]
pub struct CustomInstruction {
    pub name: String,
    pub num_params: usize,
    pub num_qubits: usize,
    pub builtin: bool,
}

#[cfg(feature = "py")]
#[pymethods]
impl CustomInstruction {
    #[new]
    fn __new__(name: String, num_params: usize, num_qubits: usize, builtin: bool) -> Self {
        Self {
            name,
            num_params,
            num_qubits,
            builtin,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ClassicalBuiltinExt {
    Asin,
    Acos,
    Atan,
}
impl ClassicalBuiltinExt {
    /// How many parameters the builtin expects.
    pub fn num_params(&self) -> usize {
        match self {
            Self::Asin | Self::Acos | Self::Atan => 1,
        }
    }
    /// The name that Qiskit historically gave this extension function.
    pub fn natural_name(&self) -> &'static str {
        match self {
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
        }
    }

    /// Call the built-in function on the given parameter list.
    ///
    /// Returns the expected number of parameters, if the given slice is the wrong length.
    pub fn call(&self, params: &[f64]) -> Result<f64, usize> {
        (params.len() == self.num_params())
            .then(|| match self {
                Self::Asin => params[0].asin(),
                Self::Acos => params[0].acos(),
                Self::Atan => params[0].atan(),
            })
            .ok_or(self.num_params())
    }
}

/// The only thing permitted to invoke a [ClassicalCallableExt].
///
/// Encapsulating the call this way means call sites don't have to know which arm of the enum they
/// hold, nor whether an interpreter is available.
#[derive(Clone, Copy)]
pub struct ClassicalEvaluator<'py> {
    /// `None` when no interpreter is attached; see [ClassicalEvaluator::detached].
    #[cfg(feature = "py")]
    py: Option<Python<'py>>,
    #[cfg(not(feature = "py"))]
    _lifetime: std::marker::PhantomData<&'py ()>,
}

impl<'py> ClassicalEvaluator<'py> {
    /// An evaluator that can call every kind of callable, including Python ones.
    #[cfg(feature = "py")]
    pub fn attached(py: Python<'py>) -> Self {
        Self { py: Some(py) }
    }

    /// An evaluator for callers with no attached interpreter, such as the C API.
    ///
    /// Python callables can't be invoked through this; that is reported as an error rather than
    /// trying to attach, since the interpreter may not even be initialised.
    pub fn detached() -> Self {
        Self {
            #[cfg(feature = "py")]
            py: None,
            #[cfg(not(feature = "py"))]
            _lifetime: std::marker::PhantomData,
        }
    }

    #[cfg(feature = "py")]
    pub fn eval(&self, callable: &ClassicalCallableExt, params: &[f64]) -> Result<f64, ParseError> {
        match self.py {
            Some(py) => callable.call_attached(py, params),
            None => callable.call(params),
        }
    }

    #[cfg(not(feature = "py"))]
    pub fn eval(&self, callable: &ClassicalCallableExt, params: &[f64]) -> Result<f64, ParseError> {
        callable.call(params)
    }
}

/// A pure-Rust callable type for custom classical functions.
pub type ClassicalFn = Arc<dyn Fn(&[f64]) -> Result<f64, ParseError> + Send + Sync>;

/// A classical callable used during expression constant-folding in the qasm2 parser.
#[derive(Clone)]
pub enum ClassicalCallableExt {
    /// An extension to OpenQASM 2 that's built into Qiskit.
    Builtin(ClassicalBuiltinExt),
    /// A user-supplied callable wrapped in an Arc closure.
    Custom { num_params: usize, f: ClassicalFn },
    #[cfg(feature = "py")]
    Python {
        num_params: usize,
        callable: Py<PyAny>,
    },
}

impl ClassicalCallableExt {
    pub fn num_params(&self) -> usize {
        match self {
            Self::Builtin(builtin) => builtin.num_params(),
            Self::Custom { num_params, .. } => *num_params,
            #[cfg(feature = "py")]
            Self::Python { num_params, .. } => *num_params,
        }
    }

    fn call(&self, params: &[f64]) -> Result<f64, ParseError> {
        match self {
            Self::Builtin(builtin) => builtin.call(params).map_err(|expected| {
                ParseError::new(format!(
                    "argument mismatch: expected {expected}, got {}",
                    params.len()
                ))
            }),
            Self::Custom { f, .. } => f(params),
            #[cfg(feature = "py")]
            Self::Python { .. } => Err(ParseError::new(
                "cannot constant-fold with a Python callable without an attached Python interpreter",
            )),
        }
    }

    /// Call the function on the given parameters, with an attached Python interpreter available for
    /// the callables that need one.
    #[cfg(feature = "py")]
    fn call_attached(&self, py: Python<'_>, params: &[f64]) -> Result<f64, ParseError> {
        match self {
            Self::Python { callable, .. } => {
                let pyargs =
                    pyo3::types::PyTuple::new(py, params).expect("f64 to pyfloat wont fail");
                let result = callable.call1(py, pyargs).map_err(|e| {
                    ParseError::with_source(
                        format!("caught exception when constant folding: {e}"),
                        e,
                    )
                })?;
                result.extract::<f64>(py).map_err(|e| {
                    ParseError::with_source(
                        "user provided classical function returned non-float",
                        e,
                    )
                })
            }
            _ => self.call(params),
        }
    }
}

/// Information about a custom classical function that should be defined in mathematical
/// expressions.
///
/// The given `callable` must be a Python function that takes `num_params` floats, and returns a
/// float.  The `name` is the identifier that refers to it in the OpenQASM 2 program.  This cannot
/// clash with any defined gates.
#[cfg_attr(feature = "py", pyclass(from_py_object))]
#[derive(Clone)]
pub struct CustomClassical {
    pub name: String,
    pub callable: ClassicalCallableExt,
}

#[cfg(feature = "py")]
#[pymethods]
impl CustomClassical {
    #[new]
    #[pyo3(text_signature = "(name, num_params, callable, /)")]
    fn __new__(name: String, num_params: usize, callable: Py<PyAny>) -> Self {
        Self {
            name,
            callable: ClassicalCallableExt::Python {
                num_params,
                callable,
            },
        }
    }

    /// Get a list of all the custom classical instructions that are built into Qiskit, but not part
    /// of the original OpenQASM 2 specification.
    #[staticmethod]
    fn builtins() -> Vec<CustomClassical> {
        [
            ClassicalBuiltinExt::Asin,
            ClassicalBuiltinExt::Acos,
            ClassicalBuiltinExt::Atan,
        ]
        .into_iter()
        .map(|builtin| CustomClassical {
            name: builtin.natural_name().to_owned(),
            callable: ClassicalCallableExt::Builtin(builtin),
        })
        .collect()
    }
}

/// The callables themselves can't be printed, so only their arity is.
impl std::fmt::Debug for ClassicalCallableExt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Builtin(builtin) => f.debug_tuple("Builtin").field(builtin).finish(),
            Self::Custom { num_params, .. } => f
                .debug_struct("Custom")
                .field("num_params", num_params)
                .finish(),
            #[cfg(feature = "py")]
            Self::Python { num_params, .. } => f
                .debug_struct("Python")
                .field("num_params", num_params)
                .finish(),
        }
    }
}
