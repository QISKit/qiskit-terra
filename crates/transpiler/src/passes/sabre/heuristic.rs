// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::IntoPyObjectExt;
use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use qiskit_circuit::impl_intopyobject_for_copy_pyclass;

/// Affect the dynamic scaling of the weight of node-set-based heuristics (basic and lookahead).
#[pyclass(module = "qiskit._accelerate.sabre", frozen, eq, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SetScaling {
    /// No dynamic scaling of the weight.
    Constant,
    /// Scale the weight by the current number of nodes in the set (e.g., if it contains 5 nodes,
    /// the weight will be multiplied by ``0.2``).
    Size,
}
impl_intopyobject_for_copy_pyclass!(SetScaling);
#[pymethods]
impl SetScaling {
    pub fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let name = match self {
            SetScaling::Constant => "Constant",
            SetScaling::Size => "Size",
        };
        (
            py.import("builtins")?.getattr("getattr")?,
            (py.get_type::<Self>(), name),
        )
            .into_py_any(py)
    }
}

/// Define the characteristics of the basic heuristic.  This is a simple sum of the physical
/// distances of every gate in the front layer.
#[pyclass(module = "qiskit._accelerate.sabre", frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BasicHeuristic {
    /// The relative weighting of this heuristic to others.  Typically you should just set this to
    /// 1.0 and define everything else in terms of this.
    pub weight: f64,
    /// Set the dynamic scaling of the weight based on the layer it is applying to.
    pub scale: SetScaling,
}
impl_intopyobject_for_copy_pyclass!(BasicHeuristic);
#[pymethods]
impl BasicHeuristic {
    #[new]
    pub fn new(weight: f64, scale: SetScaling) -> Self {
        Self { weight, scale }
    }

    pub fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (self.weight, self.scale).into_py_any(py)
    }

    pub fn __eq__(&self, py: Python, other: Py<PyAny>) -> bool {
        other.extract::<Self>(py).is_ok_and(|other| self == &other)
    }

    pub fn __repr__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let fmt = "BasicHeuristic(weight={!r}, scale={!r})";
        PyString::new(py, fmt)
            .call_method1("format", (self.weight, self.scale))?
            .into_py_any(py)
    }
}

/// Define the characteristics of the lookahead heuristic.
#[pyclass(module = "qiskit._accelerate.sabre", frozen, from_py_object)]
#[derive(Clone, PartialEq, Debug)]
pub struct LookaheadHeuristic {
    /// The relative weights of each sequential layer in this heuristic.  Typically each of these is
    /// defined relative to the :class:`.BasicHeuristic`, which generally has its weight set to 1.0.
    weights: Vec<f64>,
    /// Dynamic scaling of the heuristic weight depending on the size of each layer.
    pub scale: SetScaling,
}
impl LookaheadHeuristic {
    /// Construct a new lookahead heuristic.
    ///
    /// Fails if `weights` has $2^{16}$ or more elements (these are layers - 65,536 is too many!).
    pub fn new(weights: Vec<f64>, scale: SetScaling) -> Option<Self> {
        let _: u16 = weights.len().try_into().ok()?;
        Some(Self { weights, scale })
    }

    pub fn num_layers(&self) -> u16 {
        self.weights
            .len()
            .try_into()
            .expect("constructor enforces sufficiently few layers")
    }

    #[inline]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}
#[pymethods]
impl LookaheadHeuristic {
    #[new]
    pub fn py_new(weights: Vec<f64>, scale: SetScaling) -> PyResult<Self> {
        Self::new(weights, scale)
            .ok_or_else(|| PyValueError::new_err("must have fewer than 65,536 layers"))
    }

    pub fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (self.weights.as_slice().into_pyobject(py)?, self.scale).into_py_any(py)
    }

    pub fn __eq__(&self, py: Python, other: Py<PyAny>) -> bool {
        other.extract::<Self>(py).is_ok_and(|other| self == &other)
    }

    pub fn __repr__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let fmt = "LookaheadHeuristic(weights={!r}, scale={!r})";
        PyString::new(py, fmt)
            .call_method1(
                "format",
                (self.weights.as_slice().into_pyobject(py)?, self.scale),
            )?
            .into_py_any(py)
    }
}

/// Define the characteristics of the "decay" heuristic.  In this, each physical qubit has a
/// multiplier associated with it, beginning at 1.0, and has :attr:`increment` added to it each time
/// the qubit is involved in a swap.  The final heuristic is calculated by multiplying all other
/// components by the maximum multiplier involved in a given swap.
#[pyclass(from_py_object)]
#[pyo3(module = "qiskit._accelerate.sabre", frozen)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DecayHeuristic {
    /// The amount to add onto the multiplier of a physical qubit when it is used.
    pub increment: f64,
    /// How frequently (in terms of swaps in the layer) to reset all qubit multipliers back to 1.0.
    pub reset: usize,
}
impl_intopyobject_for_copy_pyclass!(DecayHeuristic);
#[pymethods]
impl DecayHeuristic {
    #[new]
    pub fn new(increment: f64, reset: usize) -> Self {
        Self { increment, reset }
    }

    pub fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (self.increment, self.reset).into_py_any(py)
    }

    pub fn __eq__(&self, py: Python, other: Py<PyAny>) -> bool {
        other.extract::<Self>(py).is_ok_and(|other| self == &other)
    }

    pub fn __repr__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let fmt = "DecayHeuristic(increment={!r}, reset={!r})";
        PyString::new(py, fmt)
            .call_method1("format", (self.increment, self.reset))?
            .into_py_any(py)
    }
}

/// A complete description of the heuristic that Sabre will use.  See the individual elements for a
/// greater description.
///
/// .. note::
///
///     This is an internal Qiskit object, not a formal part of the API.  You can use this for
///     fine-grained control over the Sabre heuristic, including for research, but beware that the
///     available options and configuration of it may change without warning in minor versions of
///     Qiskit.  If you are doing research using this, be sure to pin the version of Qiskit in your
///     requirements.
#[pyclass(module = "qiskit._accelerate.sabre", frozen, eq)]
#[derive(Clone, Debug, PartialEq)]
pub struct Heuristic {
    pub basic: Option<BasicHeuristic>,
    pub lookahead: Option<LookaheadHeuristic>,
    pub decay: Option<DecayHeuristic>,
    pub best_epsilon: f64,
    pub attempt_limit: usize,
}

#[pymethods]
impl Heuristic {
    /// Construct a new Sabre heuristic.  This can either be made directly of the desired
    /// components, or you can make an empty heuristic and use the ``with_*`` methods to add
    /// components to it.
    ///
    /// Args:
    ///     attempt_limit (int): the maximum number of swaps to attempt before using a fallback
    ///         "escape" mechanism to forcibly route a gate.  Set this to ``None`` to entirely
    ///         disable the mechanism, but beware that it's possible (on large coupling maps with a
    ///         lookahead heuristic component) for Sabre to get stuck in an inescapable arbitrarily
    ///         deep local minimum of the heuristic.  If this happens, and the escape mechanism is
    ///         disabled entirely, Sabre will enter an infinite loop.
    ///     best_epsilon (float): the floating-point epsilon to use when comparing scores to find
    ///         the best value.
    #[new]
    #[pyo3(signature = (basic=None, lookahead=None, decay=None, attempt_limit=1000, best_epsilon=1e-10))]
    pub fn new(
        basic: Option<BasicHeuristic>,
        lookahead: Option<LookaheadHeuristic>,
        decay: Option<DecayHeuristic>,
        attempt_limit: Option<usize>,
        best_epsilon: f64,
    ) -> Self {
        Self {
            basic,
            lookahead,
            decay,
            best_epsilon,
            attempt_limit: attempt_limit.unwrap_or(usize::MAX),
        }
    }

    pub fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (
            self.basic,
            self.lookahead.clone(),
            self.decay,
            self.attempt_limit,
            self.best_epsilon,
        )
            .into_py_any(py)
    }

    /// Set the weight of the ``basic`` heuristic (the sum of distances of gates in the front
    /// layer).  This is often set to ``1.0``.  You almost certainly should enable this part of the
    /// heuristic, or it's highly unlikely that Sabre will be able to make any progress.
    pub fn with_basic(&self, weight: f64, scale: SetScaling) -> Self {
        Self {
            basic: Some(BasicHeuristic { weight, scale }),
            ..self.clone()
        }
    }

    /// Set the layer weights of the ``lookahead`` heuristic.  The weight here should typically be
    /// less than that of ``basic``.  The number of weights dictates the number of layers.
    pub fn with_lookahead(&self, weights: Vec<f64>, scale: SetScaling) -> Self {
        Self {
            lookahead: Some(LookaheadHeuristic { weights, scale }),
            ..self.clone()
        }
    }

    /// Set the multiplier increment and reset interval of the decay heuristic.  The reset interval
    /// must be non-zero.
    pub fn with_decay(&self, increment: f64, reset: usize) -> PyResult<Self> {
        if reset == 0 {
            Err(PyValueError::new_err("decay reset interval cannot be zero"))
        } else {
            Ok(Self {
                decay: Some(DecayHeuristic { increment, reset }),
                ..self.clone()
            })
        }
    }

    pub fn __repr__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let fmt = "Heuristic(basic={!r}, lookahead={!r}, decay={!r}, attempt_limit={!r}, best_epsilon={!r})";
        PyString::new(py, fmt)
            .call_method1(
                "format",
                (
                    self.basic,
                    self.lookahead.clone(),
                    self.decay,
                    self.attempt_limit,
                    self.best_epsilon,
                ),
            )?
            .into_py_any(py)
    }
}
