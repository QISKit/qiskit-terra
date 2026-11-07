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

use crate::classical::expr::cast::Cast;
use crate::classical::expr::{Expr, ExprKind, PyExpr, Value};
use crate::classical::types::Type;
use num_bigint::BigUint;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use pyo3::{IntoPyObjectExt, intern};
use std::boxed::Box;

/// A range expression that represents a sequence of values.
///
/// The step is always Some(Expr) as it defaults to 1 when not explicitly provided.
/// This avoids the need for unwrap() calls throughout the codebase.
#[derive(Clone, Debug, PartialEq)]
pub struct Range {
    pub start: Expr,
    pub stop: Expr,
    pub step: Expr, // Always Some(Expr) - defaults to 1 when not provided
    pub ty: Type,
    pub constant: bool,
}

// Manual Eq implementation since Expr only implements PartialEq, not Eq.
// This is safe because Eq is just a marker trait indicating that PartialEq
// is an equivalence relation (reflexive, symmetric, transitive).
impl Eq for Range {}

impl<'py> IntoPyObject<'py> for Range {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyRangeExpr(self), PyExpr(ExprKind::Range)))?.into_any())
    }
}

impl<'py> IntoPyObject<'py> for &'_ Range {
    type Target = <Range as IntoPyObject<'py>>::Target;
    type Output = <Range as IntoPyObject<'py>>::Output;
    type Error = <Range as IntoPyObject<'py>>::Error;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.clone().into_pyobject(py)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Range {
    type Error = <PyRangeExpr as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let PyRangeExpr(r) = ob.extract()?;
        Ok(r)
    }
}

/// Errors when materializing constant range bounds without calling Python.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializeError {
    NonConstant,
    NonConstantBound,
    InvalidType,
    Overflow,
    ZeroStep,
}

/// Length of an OpenQASM 3 range ``[start:step:stop]`` (inclusive on both ends).
///
/// ``step`` must be positive; materialization rejects zero and unsigned bounds cannot
/// produce a negative step.
fn inclusive_range_len(start: isize, stop: isize, step: isize) -> usize {
    debug_assert!(step > 0);
    if start > stop {
        0
    } else {
        // Inclusive: start, start+step, ..., last ≤ stop.
        1 + (stop - start) as usize / step as usize
    }
}

/// Extract a constant unsigned integer from an expression, peeling implicit casts.
fn extract_const_uint(expr: &Expr) -> Result<isize, MaterializeError> {
    match expr {
        Expr::Value(Value::Uint { raw, ty }) => {
            if !matches!(ty, Type::Uint(_)) {
                return Err(MaterializeError::InvalidType);
            }
            let value: u64 = raw.try_into().map_err(|_| MaterializeError::Overflow)?;
            isize::try_from(value).map_err(|_| MaterializeError::Overflow)
        }
        Expr::Cast(cast) if cast.constant => extract_const_uint(&cast.operand),
        _ => Err(MaterializeError::NonConstantBound),
    }
}

impl Range {
    /// Returns the length of the range.
    ///
    /// For constant ranges, matches OpenQASM 3 inclusive-stop semantics
    /// (``len([start:step:stop])``).  For non-constant ranges the iteration count is unknown
    /// at compile time; returns ``1`` as a conservative placeholder for ``ForCollection``
    /// size/depth heuristics.
    pub fn len(&self) -> usize {
        if self.constant
            && let Ok((start, stop, step)) = self.try_materialize_bounds()
        {
            return inclusive_range_len(start, stop, step);
        }
        1
    }

    /// Returns whether the range is empty.
    ///
    /// For constant ranges, matches OpenQASM 3 emptiness (``len() == 0``).  For
    /// non-constant ranges, returns ``false`` because emptiness cannot be determined statically.
    pub fn is_empty(&self) -> bool {
        if self.constant {
            return self.len() == 0;
        }
        false
    }

    /// Materialize constant range bounds to integers (inclusive ``stop`` semantics).
    ///
    /// Returns the stored ``(start, stop, step)`` values as OpenQASM 3 would interpret them:
    /// both ends are inclusive. This supports internal size and depth estimates.
    pub fn try_materialize_bounds(&self) -> Result<(isize, isize, isize), MaterializeError> {
        if !self.constant {
            return Err(MaterializeError::NonConstant);
        }
        let start = extract_const_uint(&self.start)?;
        let stop = extract_const_uint(&self.stop)?;
        let step = extract_const_uint(&self.step)?;
        if step == 0 {
            return Err(MaterializeError::ZeroStep);
        }
        Ok((start, stop, step))
    }
}

/// Helper function to convert Python values to Expr::Value
fn py_value_to_expr(_py: Python, value: &Bound<PyAny>) -> PyResult<Expr> {
    if let Ok(raw) = value.extract::<i64>() {
        if raw < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Range values must be non-negative integers",
            ));
        }
        Ok(Value::Uint {
            raw: BigUint::from(raw as u64),
            ty: Type::Uint(64),
        }
        .into())
    } else if let Ok(expr) = value.extract::<Expr>() {
        // Ensure the expression is of unsigned integer type
        if !matches!(expr.ty(), Type::Uint(_)) {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Range values must be of unsigned integer type",
            ));
        }
        Ok(expr)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected non-negative integer or Expr of unsigned integer type, got {}",
            value.get_type()
        )))
    }
}

/// Helper function to determine the common type with maximum bit width
fn determine_common_max_type(types: &[Type]) -> Type {
    let mut max_type = types[0];

    for &ty in types.iter().skip(1) {
        if let (Type::Uint(max_width), Type::Uint(width)) = (&max_type, &ty) {
            // If both types are Uint, take the one with the largest bit width
            if width > max_width {
                max_type = ty;
            }
        }
        // For any other type combinations, keep the first type
    }

    max_type
}

/// A range expression with OpenQASM 3 inclusive-stop semantics.
///
/// Unlike Python's built-in :class:`range`, both ``start`` and ``stop`` are inclusive.
/// For example, ``Range(0, 5)`` iterates ``0, 1, 2, 3, 4, 5`` (matching OpenQASM 3
/// ``[0:1:5]``), not Python ``range(0, 5)``.
///
/// Args:
///     start: The inclusive start value of the range.
///     stop: The inclusive stop value of the range.
///     step: Optional step value for the range. Defaults to 1.
///     type: The resolved type of the result.
#[pyclass(
    eq,
    from_py_object,
    subclass,
    frozen,
    extends = PyExpr,
    name = "Range",
    module = "qiskit._accelerate.circuit.classical.expr"
)]
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct PyRangeExpr(pub Range);

#[pymethods]
impl PyRangeExpr {
    #[new]
    #[pyo3(signature=(start, stop, step=None, ty=None), text_signature="(start, stop, step=None, type=None)")]
    fn new(
        py: Python,
        start: &Bound<PyAny>,
        stop: &Bound<PyAny>,
        step: Option<&Bound<PyAny>>,
        ty: Option<Type>,
    ) -> PyResult<Py<Self>> {
        let start_expr = py_value_to_expr(py, start)?;
        let stop_expr = py_value_to_expr(py, stop)?;
        // Store whether the type was explicitly specified or implicitly determined
        let is_implicit_promotion = ty.is_none();

        // Determine the target type for the Range and its expressions
        let target_ty = if let Some(explicit_ty) = ty {
            // Verify that the explicit type is indeed a Uint
            if !matches!(explicit_ty, Type::Uint(_)) {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Range type must be an unsigned integer type",
                ));
            }
            explicit_ty
        } else {
            // When no explicit type is provided, determine the common type from start and stop
            // If step is provided, include it in the type determination
            let mut types = vec![start_expr.ty(), stop_expr.ty()];
            if let Some(step) = step {
                // If step is provided, we need to convert it first to get its type
                let step_expr_temp = py_value_to_expr(py, step)?;
                types.push(step_expr_temp.ty());
            }
            determine_common_max_type(&types)
        };

        // Create step expression - either from provided value or default to 1 in the target type
        let step_expr = if let Some(step) = step {
            py_value_to_expr(py, step)?
        } else {
            // Default step of 1 in the target type (Uint-only validation follows below).
            Value::Uint {
                raw: BigUint::from(1u64),
                ty: target_ty,
            }
            .into()
        };

        let constant = start_expr.is_const() && stop_expr.is_const() && step_expr.is_const();

        // Apply casts to any expressions with types different from the target type
        // For Range expressions, we only handle Uint types
        if !matches!(target_ty, Type::Uint(_)) {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Range type must be an unsigned integer type",
            ));
        }

        let (start_expr, stop_expr, step_expr) = {
            // Create necessary Cast expressions for start if needed
            let start_ty = start_expr.ty();
            let start_const = start_expr.is_const();
            let start_expr = if target_ty != start_ty {
                Expr::Cast(Box::new(Cast {
                    operand: start_expr,
                    ty: target_ty,
                    constant: start_const,
                    // Mark as implicit if the type was determined via promotion
                    implicit: is_implicit_promotion,
                }))
            } else {
                start_expr
            };

            // Create necessary Cast expressions for stop if needed
            let stop_ty = stop_expr.ty();
            let stop_const = stop_expr.is_const();
            let stop_expr = if target_ty != stop_ty {
                Expr::Cast(Box::new(Cast {
                    operand: stop_expr,
                    ty: target_ty,
                    constant: stop_const,
                    // Mark as implicit if the type was determined via promotion
                    implicit: is_implicit_promotion,
                }))
            } else {
                stop_expr
            };

            // Create necessary Cast expressions for step if needed
            let step_ty = step_expr.ty();
            let step_const = step_expr.is_const();
            let step_expr = if target_ty != step_ty {
                Expr::Cast(Box::new(Cast {
                    operand: step_expr,
                    ty: target_ty,
                    constant: step_const,
                    // Mark as implicit if the type was determined via promotion
                    implicit: is_implicit_promotion,
                }))
            } else {
                step_expr
            };

            (start_expr, stop_expr, step_expr)
        };

        Py::new(
            py,
            (
                PyRangeExpr(Range {
                    start: start_expr,
                    stop: stop_expr,
                    step: step_expr,
                    ty: target_ty,
                    constant,
                }),
                PyExpr(ExprKind::Range),
            ),
        )
    }

    #[getter]
    fn get_start(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.start.clone().into_py_any(py)
    }

    #[getter]
    fn get_stop(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.stop.clone().into_py_any(py)
    }

    #[getter]
    fn get_step(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.step.clone().into_py_any(py)
    }

    #[getter]
    fn get_const(&self) -> bool {
        self.0.constant
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.into_py_any(py)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_range"), (slf,))
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let start = self.get_start(py)?.bind(py).repr()?;
        let stop = self.get_stop(py)?.bind(py).repr()?;
        let step = self.get_step(py)?.bind(py).repr()?;
        let ty = self.get_type(py)?.bind(py).repr()?;
        Ok(format!("Range({}, {}, {}, {})", start, stop, step, ty))
    }

    fn __str__(&self, py: Python) -> PyResult<String> {
        // Use get_start and get_stop methods for consistency
        let start_py = self.get_start(py)?;
        let start = start_py.bind(py);

        let stop_py = self.get_stop(py)?;
        let stop = stop_py.bind(py);
        let start_str = match start.getattr("name") {
            Ok(name) => name.extract::<String>()?,
            Err(_) => start.str()?.to_string(),
        };
        let stop_str = match stop.getattr("name") {
            Ok(name) => name.extract::<String>()?,
            Err(_) => stop.str()?.to_string(),
        };
        let step_py = self.get_step(py)?;
        let step = step_py.bind(py);
        let step_str = match step.getattr("name") {
            Ok(name) => format!(", step={}", name.extract::<String>()?),
            Err(_) => format!(", step={}", step.str()?),
        };

        Ok(format!("Range({}, {}{})", start_str, stop_str, step_str))
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
            (
                self.get_start(py)?,
                self.get_stop(py)?,
                self.get_step(py)?,
                self.get_type(py)?,
            ),
        )
            .into_pyobject(py)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::expr::Var;
    use uuid::Uuid;

    #[test]
    fn test_materialize_bounds() {
        let ty = Type::Uint(8);
        let range = Range {
            start: Value::Uint {
                raw: BigUint::from(0u64),
                ty,
            }
            .into(),
            stop: Value::Uint {
                raw: BigUint::from(5u64),
                ty,
            }
            .into(),
            step: Value::Uint {
                raw: BigUint::from(1u64),
                ty,
            }
            .into(),
            ty,
            constant: true,
        };
        assert_eq!(range.try_materialize_bounds().unwrap(), (0, 5, 1));
    }

    #[test]
    fn test_materialize_bounds_with_step() {
        let ty = Type::Uint(8);
        let range = Range {
            start: Value::Uint {
                raw: BigUint::from(0u64),
                ty,
            }
            .into(),
            stop: Value::Uint {
                raw: BigUint::from(10u64),
                ty,
            }
            .into(),
            step: Value::Uint {
                raw: BigUint::from(2u64),
                ty,
            }
            .into(),
            ty,
            constant: true,
        };
        assert_eq!(range.try_materialize_bounds().unwrap(), (0, 10, 2));
    }

    #[test]
    fn test_materialize_bounds_with_cast() {
        let ty8 = Type::Uint(8);
        let ty32 = Type::Uint(32);
        let start = Value::Uint {
            raw: BigUint::from(5u64),
            ty: ty8,
        }
        .into();
        let stop = Value::Uint {
            raw: BigUint::from(10u64),
            ty: ty32,
        }
        .into();
        let range = Range {
            start: Expr::Cast(Box::new(Cast {
                operand: start,
                ty: ty32,
                constant: true,
                implicit: true,
            })),
            stop,
            step: Value::Uint {
                raw: BigUint::from(2u64),
                ty: ty32,
            }
            .into(),
            ty: ty32,
            constant: true,
        };
        assert_eq!(range.try_materialize_bounds().unwrap(), (5, 10, 2));
    }

    #[test]
    fn test_materialize_non_constant() {
        let ty = Type::Uint(8);
        let range = Range {
            start: Var::Standalone {
                uuid: Uuid::new_v4().as_u128(),
                name: "start".to_string(),
                ty,
            }
            .into(),
            stop: Value::Uint {
                raw: BigUint::from(10u64),
                ty,
            }
            .into(),
            step: Value::Uint {
                raw: BigUint::from(1u64),
                ty,
            }
            .into(),
            ty,
            constant: false,
        };
        assert_eq!(
            range.try_materialize_bounds(),
            Err(MaterializeError::NonConstant)
        );
    }

    #[test]
    fn test_len_constant_matches_inclusive_openqasm_range() {
        let ty = Type::Uint(8);
        let make = |start: u64, stop: u64, step: u64| Range {
            start: Value::Uint {
                raw: BigUint::from(start),
                ty,
            }
            .into(),
            stop: Value::Uint {
                raw: BigUint::from(stop),
                ty,
            }
            .into(),
            step: Value::Uint {
                raw: BigUint::from(step),
                ty,
            }
            .into(),
            ty,
            constant: true,
        };
        // Inclusive: [0:1:5] → 0..=5
        assert_eq!(make(0, 5, 1).len(), 6);
        assert!(!make(0, 5, 1).is_empty());
        // Inclusive: [0:2:10] → 0,2,4,6,8,10
        assert_eq!(make(0, 10, 2).len(), 6);
        // Inclusive singleton: [5:1:5] → [5]
        assert_eq!(make(5, 5, 1).len(), 1);
        assert!(!make(5, 5, 1).is_empty());
        // Empty when start > stop with positive step
        assert_eq!(make(6, 5, 1).len(), 0);
        assert!(make(6, 5, 1).is_empty());
        // Stop not on the step grid is still included as upper bound for membership,
        // but length counts only values ≤ stop that are reachable from start.
        assert_eq!(make(0, 9, 2).len(), 5); // 0,2,4,6,8
    }

    #[test]
    fn test_len_non_constant_placeholder() {
        let ty = Type::Uint(8);
        let range = Range {
            start: Var::Standalone {
                uuid: Uuid::new_v4().as_u128(),
                name: "start".to_string(),
                ty,
            }
            .into(),
            stop: Value::Uint {
                raw: BigUint::from(10u64),
                ty,
            }
            .into(),
            step: Value::Uint {
                raw: BigUint::from(1u64),
                ty,
            }
            .into(),
            ty,
            constant: false,
        };
        assert_eq!(range.len(), 1);
        assert!(!range.is_empty());
    }

    #[test]
    fn test_materialize_zero_step() {
        let ty = Type::Uint(8);
        let range = Range {
            start: Value::Uint {
                raw: BigUint::from(0u64),
                ty,
            }
            .into(),
            stop: Value::Uint {
                raw: BigUint::from(5u64),
                ty,
            }
            .into(),
            step: Value::Uint {
                raw: BigUint::from(0u64),
                ty,
            }
            .into(),
            ty,
            constant: true,
        };
        assert_eq!(
            range.try_materialize_bounds(),
            Err(MaterializeError::ZeroStep)
        );
    }
}
