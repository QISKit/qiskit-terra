// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use num_complex::Complex64;
use std::sync::{Arc, atomic};
use thiserror::Error;
use uuid::Uuid;

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

use pyo3::exceptions::{
    PyDeprecationWarning, PyRuntimeError, PyTypeError, PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    IntoPyDict, PyComplex, PyFloat, PyInt, PyIterator, PyList, PyNotImplemented, PySet, PyString,
    PyTuple,
};
use pyo3::{IntoPyObjectExt, intern};

use crate::circuit_data::CircuitError;
use crate::imports::{BUILTIN_HASH, SYMPIFY_PARAMETER_EXPRESSION, UUID, WARNINGS_WARN};
use crate::parameter::symbol_expr::{self, SymbolExpr, SymbolVector};
use crate::parameter::symbol_parser::parse_expression;
use qiskit_util::IndexSet;

use super::symbol_expr::{SYMEXPR_EPSILON, Symbol, Value};

/// Errors for dealing with parameters and parameter expressions.
#[derive(Error, Debug)]
pub enum ParameterError {
    #[error("Division by zero.")]
    ZeroDivisionError,
    #[error("Binding to infinite value.")]
    BindingInf,
    #[error("Binding to NaN.")]
    BindingNaN,
    #[error("Invalid value: NaN or infinite.")]
    InvalidValue,
    #[error("Cannot bind following parameters not present in expression: {0:?}")]
    UnknownParameters(HashSet<Symbol>),
    #[error("Parameter expression with unbound parameters {0:?} is not numeric.")]
    UnboundParameters(HashSet<Symbol>),
    #[error("Name conflict adding parameters.")]
    NameConflict,
    #[error("Invalid cast to OpCode: {0}")]
    InvalidU8ToOpCode(u8),
    #[error("Could not cast to Symbol.")]
    NotASymbol,
    #[error("Derivative not supported on expression: {0}")]
    DerivativeNotSupported(String),
}

impl From<ParameterError> for PyErr {
    fn from(value: ParameterError) -> Self {
        match value {
            ParameterError::ZeroDivisionError => {
                PyZeroDivisionError::new_err("zero division occurs while binding parameter")
            }
            ParameterError::BindingInf => {
                PyZeroDivisionError::new_err("attempted to bind infinite value to parameter")
            }
            ParameterError::UnknownParameters(_) | ParameterError::NameConflict => {
                CircuitError::new_err(value.to_string())
            }
            ParameterError::UnboundParameters(_) => PyTypeError::new_err(value.to_string()),
            ParameterError::InvalidValue => PyValueError::new_err(value.to_string()),
            _ => PyRuntimeError::new_err(value.to_string()),
        }
    }
}

/// A parameter expression.
///
/// This is backed by Qiskit's symbolic expression engine and a cache
/// for the parameters inside the expression.
#[derive(Clone, Debug)]
pub struct ParameterExpression {
    // The symbolic expression.
    expr: SymbolExpr,
    // A map keeping track of all symbols, with their name. This map *must* have
    // exactly one entry per symbol used in the expression (no more, no less).
    name_map: HashMap<String, Symbol>,
}

impl Hash for ParameterExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // The string representation of a tree is unique.
        self.expr.string_id().hash(state);
    }
}

impl PartialEq for ParameterExpression {
    fn eq(&self, other: &Self) -> bool {
        self.expr.eq(&other.expr)
    }
}

impl Eq for ParameterExpression {}

impl Default for ParameterExpression {
    /// The default constructor returns zero.
    fn default() -> Self {
        Self {
            expr: SymbolExpr::Value(Value::Int(0)),
            name_map: HashMap::new(), // no parameters, hence empty name map
        }
    }
}

impl fmt::Display for ParameterExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", {
            if let SymbolExpr::Symbol(s) = &self.expr {
                s.repr(false)
            } else {
                match self.expr.eval(true) {
                    Some(e) => e.to_string(),
                    None => self.expr.to_string(),
                }
            }
        })
    }
}

impl ParameterExpression {
    pub fn qpy_replay(&self) -> Vec<OPReplay> {
        let mut replay = Vec::new();
        let mut unused: IndexSet<_> = self.name_map.values().cloned().collect();
        // The recursive inner `qpy_replay_inner` assumes it starts from a containing operation, so
        // fails to build a complete replay in the case it starts from a single symbol or value.
        match &self.expr {
            SymbolExpr::Value(v) => {
                let item = match *v {
                    Value::Int(v) => OPReplay {
                        op: OpCode::ADD,
                        lhs: Some(ParameterValueType::Int(v)),
                        rhs: Some(ParameterValueType::Int(0)),
                    },
                    Value::Real(v) => OPReplay {
                        op: OpCode::ADD,
                        lhs: Some(ParameterValueType::Float(v)),
                        // `-0.0` is technically the identity element of floating-point addition;
                        // `0.0 + x` is not bit-for-bit equal to `x` solely if `x` is `-0.0`.
                        rhs: Some(ParameterValueType::Float(-0.0)),
                    },
                    Value::Complex(v) => OPReplay {
                        op: OpCode::ADD,
                        lhs: Some(ParameterValueType::Complex(v)),
                        rhs: Some(ParameterValueType::Complex(Complex64 {
                            re: -0.0,
                            im: -0.0,
                        })),
                    },
                };
                replay.push(item);
            }
            SymbolExpr::Symbol(sym) => {
                unused.swap_remove(sym.as_ref());
                replay.push(OPReplay {
                    op: OpCode::ADD,
                    lhs: ParameterValueType::extract_from_expr(&self.expr),
                    rhs: Some(ParameterValueType::Int(0)),
                });
            }
            SymbolExpr::Unary { .. } | SymbolExpr::Binary { .. } => {
                qpy_replay_inner(self, &self.name_map, &mut replay, &mut unused);
            }
        }
        // For any unused symbols, we'll add something like `(x * 0) + expr`.  This sort of
        // cancellation is how unused symbols appear; it doesn't matter if the _actual_ cause was
        // `x - x` or whatever, because the end observable effect is the same.
        for symbol in unused {
            replay.push(OPReplay {
                op: OpCode::MUL,
                lhs: Some(ParameterValueType::Parameter(PyParameter(Arc::new(symbol)))),
                rhs: Some(ParameterValueType::Int(0)),
            });
            replay.push(OPReplay {
                op: OpCode::ADD,
                lhs: None,
                rhs: None,
            });
        }
        replay
    }
}
// This needs to be implemented manually, because PyO3 does not provide built-in
// conversions for the subclasses of ParameterExpression in Python. Specifically
// the Python classes Parameter and ParameterVector are subclasses of
// ParameterExpression and the default trait impl would not handle the specialization
// there.
impl<'py> IntoPyObject<'py> for ParameterExpression {
    type Target = PyParameterExpression;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let expr = PyParameterExpression::from(self.clone());
        expr.into_pyobject(py)
    }
}

/// Lookup for which operations are binary (i.e. require two operands).
static BINARY_OPS: [OpCode; 8] = [
    // a HashSet would be better but requires unstable features
    OpCode::ADD,
    OpCode::SUB,
    OpCode::MUL,
    OpCode::DIV,
    OpCode::POW,
    OpCode::RSUB,
    OpCode::RDIV,
    OpCode::RPOW,
];

impl ParameterExpression {
    /// Initialize with an existing [SymbolExpr] and its valid name map.
    ///
    /// Caution: The caller **guarantees** that ``name_map`` is consistent with ``expr``.
    /// If uncertain, call [Self::from_symbol_expr], which automatically builds the correct name map.
    pub fn new(expr: SymbolExpr, name_map: HashMap<String, Symbol>) -> Self {
        Self { expr, name_map }
    }

    /// Construct from a [Symbol].
    pub fn from_symbol(symbol: Symbol) -> Self {
        Self {
            expr: SymbolExpr::Symbol(Arc::new(symbol.clone())),
            name_map: [(symbol.repr(false), symbol)].into(),
        }
    }

    pub fn from_arc_symbol(symbol: Arc<Symbol>) -> Self {
        Self {
            name_map: [(symbol.repr(false), Symbol::clone(&*symbol))].into(),
            expr: SymbolExpr::Symbol(symbol),
        }
    }

    /// Try casting to a [Symbol].
    ///
    /// This only succeeds if the underlying expression is, in fact, only a symbol.
    pub fn try_to_symbol(&self) -> Result<Symbol, ParameterError> {
        self.try_to_symbol_ref().cloned()
    }

    /// Simplify the expression. This, for example, attempts to cancel
    /// variables in the expression.
    pub fn simplify(&self) -> Self {
        ParameterExpression::from_symbol_expr(self.expr.optimize())
    }

    /// Try casting to a [Symbol], returning a reference.
    ///
    /// This only succeeds if the underlying expression is, in fact, only a symbol.
    pub fn try_to_symbol_ref(&self) -> Result<&Symbol, ParameterError> {
        match &self.expr {
            SymbolExpr::Symbol(sym) if self.name_map.len() == 1 => Ok(sym.as_ref()),
            _ => Err(ParameterError::NotASymbol),
        }
    }

    /// Try casting to a [Value].
    ///
    /// Attempt to evaluate the expression recursively and return a [Value] if fully bound.
    ///
    /// # Arguments
    ///
    /// * strict - If ``true``, only allow returning a value if all symbols are bound. If
    ///   ``false``, allow casting expressions to values, even though symbols might still exist.
    ///   For example, ``0 * x`` will return ``0`` for ``strict=false`` and otherwise return
    ///   an error.
    pub fn try_to_value(&self, strict: bool) -> Result<Value, ParameterError> {
        if strict && !self.name_map.is_empty() {
            let free_symbols = self.expr.iter_symbols().cloned().collect();
            return Err(ParameterError::UnboundParameters(free_symbols));
        }

        match self.expr.eval(true) {
            Some(value) => {
                // we try to restrict complex to real, if possible
                if let Value::Complex(c) = value {
                    if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON).contains(&c.im)
                    {
                        return Ok(Value::Real(c.re));
                    }
                }
                Ok(value)
            }
            None => {
                let free_symbols = self.expr.iter_symbols().cloned().collect();
                Err(ParameterError::UnboundParameters(free_symbols))
            }
        }
    }

    /// Construct from a [SymbolExpr].
    ///
    /// This populates the name map with the symbols in the expression.
    pub fn from_symbol_expr(expr: SymbolExpr) -> Self {
        let name_map = expr.name_map();
        Self { expr, name_map }
    }

    /// Initialize from an f64.
    pub fn from_f64(value: f64) -> Self {
        Self {
            expr: SymbolExpr::Value(Value::Real(value)),
            name_map: HashMap::new(),
        }
    }

    /// Load from a sequence of [OPReplay]s. Used in serialization.
    pub fn from_qpy(replay: &[OPReplay]) -> Result<Self, ParameterError> {
        // the stack contains the latest lhs and rhs values
        let mut stack: Vec<ParameterExpression> = Vec::new();
        for OPReplay { op, lhs, rhs } in replay {
            // put the values on the stack, if they exist
            if let Some(value) = lhs {
                stack.push(value.clone().into());
            }
            if let Some(value) = rhs {
                stack.push(value.clone().into());
            }

            // if we need two operands, pop rhs from the stack
            let rhs = if BINARY_OPS.contains(op) {
                Some(stack.pop().expect("Pop from empty stack"))
            } else {
                None
            };

            // pop lhs from the stack, this we always need
            let lhs = stack.pop().expect("Pop from empty stack");

            // apply the operation and put the result onto the stack for the next replay
            let result: ParameterExpression = match op {
                OpCode::ADD => lhs.add(&rhs.unwrap())?,
                OpCode::MUL => lhs.mul(&rhs.unwrap())?,
                OpCode::SUB => lhs.sub(&rhs.unwrap())?,
                OpCode::RSUB => rhs.unwrap().sub(&lhs)?,
                OpCode::POW => lhs.pow(&rhs.unwrap())?,
                OpCode::RPOW => rhs.unwrap().pow(&lhs)?,
                OpCode::DIV => lhs.div(&rhs.unwrap())?,
                OpCode::RDIV => rhs.unwrap().div(&lhs)?,
                OpCode::ABS => lhs.abs(),
                OpCode::SIN => lhs.sin(),
                OpCode::ASIN => lhs.asin(),
                OpCode::COS => lhs.cos(),
                OpCode::ACOS => lhs.acos(),
                OpCode::TAN => lhs.tan(),
                OpCode::ATAN => lhs.atan(),
                OpCode::CONJ => lhs.conjugate(),
                OpCode::LOG => lhs.log(),
                OpCode::EXP => lhs.exp(),
                OpCode::SIGN => lhs.sign(),
                OpCode::GRAD | OpCode::SUBSTITUTE => {
                    panic!("GRAD and SUBSTITUTE are not supported.")
                }
            };
            stack.push(result);
        }

        // once we're done, just return the last element in the stack
        Ok(stack
            .pop()
            .expect("Invalid QPY replay encountered during deserialization: empty OPReplay."))
    }

    pub fn iter_symbols(&self) -> impl Iterator<Item = &Symbol> + '_ {
        self.name_map.values()
    }

    /// Get the number of [Symbol]s in the expression.
    pub fn num_symbols(&self) -> usize {
        self.name_map.len()
    }

    /// Whether the expression represents a complex number. None if cannot be determined.
    pub fn is_complex(&self) -> Option<bool> {
        self.expr.is_complex()
    }

    /// Whether the expression represents a int. None if cannot be determined.
    pub fn is_int(&self) -> Option<bool> {
        self.expr.is_int()
    }

    /// Add an expression; ``self + rhs``.
    pub fn add(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.merged_name_map(rhs)?;
        Ok(Self {
            expr: &self.expr + &rhs.expr,
            name_map,
        })
    }

    /// Multiply with an expression; ``self * rhs``.
    pub fn mul(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.merged_name_map(rhs)?;
        Ok(Self {
            expr: &self.expr * &rhs.expr,
            name_map,
        })
    }

    /// Subtract another expression; ``self - rhs``.
    pub fn sub(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.merged_name_map(rhs)?;
        Ok(Self {
            expr: &self.expr - &rhs.expr,
            name_map,
        })
    }

    /// Divide by another expression; ``self / rhs``.
    pub fn div(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        if rhs.expr.is_zero() {
            return Err(ParameterError::ZeroDivisionError);
        }

        let name_map = self.merged_name_map(rhs)?;
        Ok(Self {
            expr: &self.expr / &rhs.expr,
            name_map,
        })
    }

    /// Raise this expression to a power; ``self ^ rhs``.
    pub fn pow(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.merged_name_map(rhs)?;
        Ok(Self {
            expr: self.expr.pow(&rhs.expr),
            name_map,
        })
    }

    /// Apply the sine to this expression; ``sin(self)``.
    pub fn sin(&self) -> Self {
        Self {
            expr: self.expr.sin(),
            name_map: self.name_map.clone(),
        }
    }

    /// Apply the cosine to this expression; ``cos(self)``.
    pub fn cos(&self) -> Self {
        Self {
            expr: self.expr.cos(),
            name_map: self.name_map.clone(),
        }
    }

    /// Apply the tangent to this expression; ``tan(self)``.
    pub fn tan(&self) -> Self {
        Self {
            expr: self.expr.tan(),
            name_map: self.name_map.clone(),
        }
    }

    /// Apply the arcsine to this expression; ``asin(self)``.
    pub fn asin(&self) -> Self {
        Self {
            expr: self.expr.asin(),
            name_map: self.name_map.clone(),
        }
    }

    /// Apply the arccosine to this expression; ``acos(self)``.
    pub fn acos(&self) -> Self {
        Self {
            expr: self.expr.acos(),
            name_map: self.name_map.clone(),
        }
    }

    /// Apply the arctangent to this expression; ``atan(self)``.
    pub fn atan(&self) -> Self {
        Self {
            expr: self.expr.atan(),
            name_map: self.name_map.clone(),
        }
    }

    /// Exponentiate this expression; ``exp(self)``.
    pub fn exp(&self) -> Self {
        Self {
            expr: self.expr.exp(),
            name_map: self.name_map.clone(),
        }
    }

    /// Take the (natural) logarithm of this expression; ``log(self)``.
    pub fn log(&self) -> Self {
        Self {
            expr: self.expr.log(),
            name_map: self.name_map.clone(),
        }
    }

    /// Take the absolute value of this expression; ``|self|``.
    pub fn abs(&self) -> Self {
        Self {
            expr: self.expr.abs(),
            name_map: self.name_map.clone(),
        }
    }

    /// Return the sign of this expression; ``sign(self)``.
    pub fn sign(&self) -> Self {
        Self {
            expr: self.expr.sign(),
            name_map: self.name_map.clone(),
        }
    }

    /// Complex conjugate the expression.
    pub fn conjugate(&self) -> Self {
        Self {
            expr: self.expr.conjugate(),
            name_map: self.name_map.clone(),
        }
    }

    /// negate the expression.
    pub fn neg(&self) -> Self {
        Self {
            expr: -&self.expr,
            name_map: self.name_map.clone(),
        }
    }

    /// Compute the derivative of the expression with respect to the provided symbol.
    ///
    /// Note that this keeps the name map unchanged. Meaning that computing the derivative
    /// of ``x`` will yield ``1`` but the expression still owns the symbol ``x``. This is
    /// done such that we can still bind the value ``x`` in an automated process.
    pub fn derivative(&self, param: &Symbol) -> Result<Self, ParameterError> {
        Ok(Self {
            expr: self
                .expr
                .derivative(param)
                .map_err(ParameterError::DerivativeNotSupported)?,
            name_map: self.name_map.clone(),
        })
    }

    /// Substitute symbols with [ParameterExpression]s.
    ///
    /// # Arguments
    ///
    /// * map - A hashmap with [Symbol] keys and [ParameterExpression]s to replace these
    ///   symbols with.
    /// * allow_unknown_parameters - If `false`, returns an error if any symbol in the
    ///   hashmap is not present in the expression. If `true`, unknown symbols are ignored.
    ///   Setting to `true` is slightly faster as it does not involve additional checks.
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - A parameter expression with the substituted expressions.
    /// * `Err(ParameterError)` - An error if the substitution failed.
    pub fn subs(
        &self,
        map: &HashMap<Symbol, Self>,
        allow_unknown_parameters: bool,
    ) -> Result<Self, ParameterError> {
        // Build the outgoing name map. In the process we check for any duplicates.
        let mut name_map: HashMap<String, Symbol> = HashMap::new();
        let mut symbol_map: HashMap<Symbol, SymbolExpr> = HashMap::new();

        // If we don't allow for unknown parameters, check if there are any.
        if !allow_unknown_parameters {
            let existing: HashSet<&Symbol> = self.name_map.values().collect();
            let to_replace: HashSet<&Symbol> = map.keys().collect();
            let mut difference = to_replace.difference(&existing).peekable();

            if difference.peek().is_some() {
                let different_symbols = difference.map(|s| (**s).clone()).collect();
                return Err(ParameterError::UnknownParameters(different_symbols));
            }
        }

        for (name, symbol) in self.name_map.iter() {
            // check if the symbol will get replaced
            if let Some(replacement) = map.get(symbol) {
                // If yes, update the name_map. This also checks for duplicates.
                for (replacement_name, replacement_symbol) in replacement.name_map.iter() {
                    if let Some(duplicate) = name_map.get(replacement_name) {
                        // If a symbol with the same name already exists, check whether it is
                        // the same symbol (fine) or a different symbol with the same name (conflict)!
                        if duplicate != replacement_symbol {
                            return Err(ParameterError::NameConflict);
                        }
                    } else {
                        // SAFETY: We know the key does not exist yet.
                        unsafe {
                            name_map.insert_unique_unchecked(
                                replacement_name.clone(),
                                replacement_symbol.clone(),
                            )
                        };
                    }
                }

                // If we got until here, there were no duplicates, so we are safe to
                // add this symbol to the internal replacement map.
                symbol_map.insert(symbol.clone(), replacement.expr.clone());
            } else {
                // no replacement for this symbol, carry on
                match name_map.entry(name.clone()) {
                    Entry::Occupied(duplicate) => {
                        if duplicate.get() != symbol {
                            return Err(ParameterError::NameConflict);
                        }
                    }
                    Entry::Vacant(e) => {
                        e.insert(symbol.clone());
                    }
                }
            }
        }

        let res = self.expr.subs(&symbol_map);
        Ok(Self {
            expr: res,
            name_map,
        })
    }

    /// Bind symbols to values.
    ///
    /// # Arguments
    ///
    /// * map - A hashmap with [Symbol] keys and [Value]s to replace these
    ///   symbols with.
    /// * allow_unknown_parameter - If `false`, returns an error if any symbol in the
    ///   hashmap is not present in the expression. If `true`, unknown symbols are ignored.
    ///   Setting to `true` is slightly faster as it does not involve additional checks.
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - A parameter expression with the bound symbols.
    /// * `Err(ParameterError)` - An error if binding failed.
    pub fn bind(
        &self,
        map: &HashMap<&Symbol, Value>,
        allow_unknown_parameters: bool,
    ) -> Result<Self, ParameterError> {
        // The set of symbols we will bind. Used twice, hence pre-computed here.
        let bind_symbols: HashSet<&Symbol> = map.keys().cloned().collect();

        // If we don't allow for unknown parameters, check if there are any.
        if !allow_unknown_parameters {
            let existing: HashSet<&Symbol> = self.name_map.values().collect();
            let mut difference = bind_symbols.difference(&existing).peekable();

            if difference.peek().is_some() {
                let different_symbols = difference.map(|s| (**s).clone()).collect();
                return Err(ParameterError::UnknownParameters(different_symbols));
            }
        }

        // bind the symbol expression and then check the outcome for inf/nan, or numeric values
        let bound_expr = self.expr.bind(map);
        let bound = match bound_expr.eval(true) {
            Some(v) => match &v {
                Value::Real(r) => {
                    if r.is_infinite() {
                        Err(ParameterError::BindingInf)
                    } else if r.is_nan() {
                        Err(ParameterError::BindingNaN)
                    } else {
                        Ok(SymbolExpr::Value(v))
                    }
                }
                Value::Int(_) => Ok(SymbolExpr::Value(v)),
                Value::Complex(c) => {
                    if c.re.is_infinite() || c.im.is_infinite() {
                        Err(ParameterError::BindingInf)
                    } else if c.re.is_nan() || c.im.is_nan() {
                        Err(ParameterError::BindingNaN)
                    } else if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON)
                        .contains(&c.im)
                    {
                        Ok(SymbolExpr::Value(Value::Real(c.re)))
                    } else {
                        Ok(SymbolExpr::Value(v))
                    }
                }
            },
            None => Ok(bound_expr),
        }?;

        // update the name map by removing the bound parameters
        let bound_name_map: HashMap<String, Symbol> = self
            .name_map
            .iter()
            .filter(|(_, symbol)| !bind_symbols.contains(symbol))
            .map(|(name, symbol)| (name.clone(), symbol.clone()))
            .collect();

        Ok(Self {
            expr: bound,
            name_map: bound_name_map,
        })
    }

    /// Merge name maps.
    ///
    /// # Arguments
    ///
    /// * `other` - The other parameter expression whose symbols we add to self.
    ///
    /// # Returns
    ///
    /// * `Ok(HashMap<String, Symbol>)` - The merged name map.
    /// * `Err(ParameterError)` - An error if there was a name conflict.
    fn merged_name_map(&self, other: &Self) -> Result<HashMap<String, Symbol>, ParameterError> {
        let mut merged = self.name_map.clone();
        for (name, param) in other.name_map.iter() {
            match merged.get(name) {
                Some(existing_param) => {
                    if param != existing_param {
                        return Err(ParameterError::NameConflict);
                    }
                }
                None => {
                    // SAFETY: We ensured the key is unique
                    let _ = unsafe { merged.insert_unique_unchecked(name.clone(), param.clone()) };
                }
            }
        }
        Ok(merged)
    }
}

/// A parameter expression.
///
/// This is backed by Qiskit's symbolic expression engine and a cache
/// for the parameters inside the expression.
#[pyclass(
    subclass,
    module = "qiskit._accelerate.circuit",
    name = "ParameterExpression",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyParameterExpression {
    pub inner: ParameterExpression,
}

impl Default for PyParameterExpression {
    /// The default constructor returns zero.
    fn default() -> Self {
        Self {
            inner: ParameterExpression::default(),
        }
    }
}

impl fmt::Display for PyParameterExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl From<ParameterExpression> for PyParameterExpression {
    fn from(value: ParameterExpression) -> Self {
        Self { inner: value }
    }
}

impl PyParameterExpression {
    /// Attempt to extract a `PyParameterExpression` from a bound `PyAny`.
    ///
    /// This will try to coerce to the strictest data type:
    /// Int - Real - Complex - PyParameterVectorElement - PyParameter - PyParameterExpression.
    ///
    /// # Arguments:
    ///
    /// * ob - The bound `PyAny` to extract from.
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - The extracted expression.
    /// * `Err(PyResult)` - An error if extraction to all above types failed.
    pub fn extract_coerce(ob: Borrowed<PyAny>) -> PyResult<Self> {
        if let Ok(i) = ob.cast::<PyInt>() {
            Ok(ParameterExpression::new(
                SymbolExpr::Value(Value::from(i.extract::<i64>()?)),
                HashMap::new(),
            )
            .into())
        } else if let Ok(r) = ob.cast::<PyFloat>() {
            let r: f64 = r.extract()?;
            if r.is_infinite() || r.is_nan() {
                return Err(ParameterError::InvalidValue.into());
            }
            Ok(ParameterExpression::new(SymbolExpr::Value(Value::from(r)), HashMap::new()).into())
        } else if let Ok(c) = ob.cast::<PyComplex>() {
            let c: Complex64 = c.extract()?;
            if c.is_infinite() || c.is_nan() {
                return Err(ParameterError::InvalidValue.into());
            }
            Ok(ParameterExpression::new(SymbolExpr::Value(Value::from(c)), HashMap::new()).into())
        } else if let Ok(parameter) = ob.cast::<PyParameter>() {
            Ok(ParameterExpression::from_symbol(Symbol::clone(&parameter.borrow().0)).into())
        } else {
            ob.extract::<PyParameterExpression>().map_err(Into::into)
        }
    }

    pub fn coerce_into_py(&self, py: Python) -> PyResult<Py<PyAny>> {
        if let Ok(value) = self.inner.try_to_value(true) {
            match value {
                Value::Int(i) => Ok(PyInt::new(py, i).unbind().into_any()),
                Value::Real(r) => Ok(PyFloat::new(py, r).unbind().into_any()),
                Value::Complex(c) => Ok(PyComplex::from_complex_bound(py, c).unbind().into_any()),
            }
        } else if let Ok(symbol) = self.inner.try_to_symbol() {
            PyParameter(Arc::new(symbol)).into_py_any(py)
        } else {
            self.clone().into_py_any(py)
        }
    }
}

#[pymethods]
impl PyParameterExpression {
    /// Simplify the expression. This, for example, attempts to cancel
    /// variables in the expression.
    pub fn simplify(&self) -> Self {
        self.inner.simplify().into()
    }

    /// This is a **strictly internal** constructor and **should not be used**.
    /// It is subject to arbitrary change in between Qiskit versions and cannot be relied on.
    /// Parameter expressions should always be constructed from applying operations on
    /// parameters, or by loading via QPY.
    #[new]
    #[pyo3(signature = (name_map, expr))]
    pub fn py_new(name_map: HashMap<String, PyParameter>, expr: String) -> PyResult<Self> {
        // We first parse the expression and then update the symbols with the ones
        // the user provided. The replacement relies on the names to match.
        // This is hacky and we likely want a more reliably conversion from a SymPy object,
        // if we decide we want to continue supporting this.
        let expr = parse_expression(&expr)
            .map_err(|_| PyRuntimeError::new_err("Failed parsing input expression"))?;
        let symbol_map: HashMap<String, Symbol> = name_map
            .iter()
            .map(|(string, param)| (string.clone(), Symbol::clone(&param.0)))
            .collect();

        let replaced_expr = symbol_expr::replace_symbol(&expr, &symbol_map);

        let inner = ParameterExpression::new(replaced_expr, symbol_map);
        Ok(Self { inner })
    }

    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn _Value(value: &Bound<PyAny>) -> PyResult<Self> {
        Self::extract_coerce(value.as_borrowed())
    }

    /// Check if the expression corresponds to a plain symbol.
    ///
    /// Returns:
    ///     ``True`` is this expression corresponds to a symbol, ``False`` otherwise.
    pub fn is_symbol(&self) -> bool {
        matches!(self.inner.expr, SymbolExpr::Symbol(_))
    }

    /// Cast this expression to a numeric value.
    ///
    /// Args:
    ///     strict: If ``True`` (default) this function raises an error if there are any
    ///         unbound symbols in the expression. If ``False``, this allows casting
    ///         if the expression represents a numeric value, regardless of unbound symbols.
    ///         For example ``(0 * Parameter("x"))`` is 0 but has the symbol ``x`` present.
    #[pyo3(signature = (strict=true))]
    pub fn numeric(&self, py: Python, strict: bool) -> PyResult<Py<PyAny>> {
        match self.inner.try_to_value(strict)? {
            Value::Real(r) => r.into_py_any(py),
            Value::Int(i) => i.into_py_any(py),
            Value::Complex(c) => c.into_py_any(py),
        }
    }

    /// Return a SymPy equivalent of this expression.
    ///
    /// Returns:
    ///     A SymPy equivalent of this expression.
    pub fn sympify(&self, py: Python) -> PyResult<Py<PyAny>> {
        let py_sympify = SYMPIFY_PARAMETER_EXPRESSION.get(py);
        py_sympify.call1(py, (self.clone(),))
    }

    /// The number of unbound parameters in the expression.
    ///
    /// This is equivalent to ``len(expr.parameters)`` but does not involve the overhead of creating
    /// a set and counting its length.
    #[getter]
    pub fn num_parameters(&self) -> usize {
        self.inner.num_symbols()
    }

    /// Get the parameters present in the expression.
    ///
    /// .. note::
    ///
    ///     Qiskit guarantees equality (via ``==``) of parameters retrieved from an expression
    ///     with the original :class:`.Parameter` objects used to create this expression,
    ///     but does **not guarantee** ``is`` comparisons to succeed.
    ///
    #[getter]
    pub fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let py_parameters: Vec<Py<PyAny>> = self
            .inner
            .iter_symbols()
            .map(|symbol| PyParameter(Arc::new(symbol.clone())).into_py_any(py))
            .collect::<PyResult<_>>()?;
        PySet::new(py, py_parameters)
    }

    /// Sine of the expression.
    #[inline]
    #[pyo3(name = "sin")]
    pub fn py_sin(&self) -> Self {
        self.inner.sin().into()
    }

    /// Cosine of the expression.
    #[inline]
    #[pyo3(name = "cos")]
    pub fn py_cos(&self) -> Self {
        self.inner.cos().into()
    }

    /// Tangent of the expression.
    #[inline]
    #[pyo3(name = "tan")]
    pub fn py_tan(&self) -> Self {
        self.inner.tan().into()
    }

    /// Arcsine of the expression.
    #[inline]
    pub fn arcsin(&self) -> Self {
        self.inner.asin().into()
    }

    /// Arccosine of the expression.
    #[inline]
    pub fn arccos(&self) -> Self {
        self.inner.acos().into()
    }

    /// Arctangent of the expression.
    #[inline]
    pub fn arctan(&self) -> Self {
        self.inner.atan().into()
    }

    /// Exponentiate the expression.
    #[inline]
    #[pyo3(name = "exp")]
    pub fn py_exp(&self) -> Self {
        self.inner.exp().into()
    }

    /// Take the natural logarithm of the expression.
    #[inline]
    #[pyo3(name = "log")]
    pub fn py_log(&self) -> Self {
        self.inner.log().into()
    }

    /// Take the absolute value of the expression.
    #[inline]
    #[pyo3(name = "abs")]
    pub fn py_abs(&self) -> Self {
        self.inner.abs().into()
    }

    /// Return the sign of the expression.
    #[inline]
    #[pyo3(name = "sign")]
    pub fn py_sign(&self) -> Self {
        self.inner.sign().into()
    }

    /// Return the complex conjugate of the expression.
    #[inline]
    #[pyo3(name = "conjugate")]
    pub fn py_conjugate(&self) -> Self {
        self.inner.conjugate().into()
    }

    /// Check whether the expression represents a real number.
    ///
    /// Note that this will return ``None`` if there are unbound parameters, in which case
    /// it cannot be determined whether the expression is real.
    #[inline]
    #[pyo3(name = "is_real")]
    pub fn py_is_real(&self) -> Option<bool> {
        self.inner.expr.is_real()
    }

    /// Return derivative of this expression with respect to the input parameter.
    ///
    /// Args:
    ///     param: The parameter with respect to which the derivative is calculated.
    ///
    /// Returns:
    ///     The derivative as either a constant numeric value or a symbolic
    ///     :class:`.ParameterExpression`.
    pub fn gradient(&self, py: Python, param: &PyParameter) -> PyResult<Py<PyAny>> {
        let d_expr = self.inner.derivative(&param.0)?;

        // try converting to value and return as built-in numeric type
        match d_expr.try_to_value(false) {
            Ok(val) => match val {
                Value::Real(r) => r.into_py_any(py),
                Value::Int(i) => i.into_py_any(py),
                Value::Complex(c) => c.into_py_any(py),
            },
            Err(_) => PyParameterExpression::from(d_expr).into_py_any(py),
        }
    }

    /// Return all values in this equation.
    pub fn _values(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .expr
            .values()
            .iter()
            .map(|val| match val {
                Value::Real(r) => r.into_py_any(py),
                Value::Int(i) => i.into_py_any(py),
                Value::Complex(c) => c.into_py_any(py),
            })
            .collect()
    }

    /// Returns a new expression with replacement parameters.
    ///
    /// Args:
    ///     parameter_map: Mapping from :class:`.Parameter`\ s in ``self`` to the
    ///         :class:`.ParameterExpression` instances with which they should be replaced.
    ///     allow_unknown_parameters: If ``False``, raises an error if ``parameter_map``
    ///         contains :class:`.Parameter`\ s in the keys outside those present in the expression.
    ///         If ``True``, any such parameters are simply ignored.
    ///
    /// Raises:
    ///     CircuitError:
    ///         - If parameter_map contains parameters outside those in self.
    ///         - If the replacement parameters in ``parameter_map`` would result in
    ///           a name conflict in the generated expression.
    ///
    /// Returns:
    ///     A new expression with the specified parameters replaced.
    #[pyo3(name = "subs")]
    #[pyo3(signature = (parameter_map, allow_unknown_parameters=false))]
    pub fn py_subs(
        &self,
        parameter_map: HashMap<PyParameter, Self>,
        allow_unknown_parameters: bool,
    ) -> PyResult<Self> {
        // reduce the map to a HashMap<Symbol, ParameterExpression>
        let map = parameter_map
            .into_iter()
            .map(|(param, expr)| Ok((Arc::unwrap_or_clone(param.0), expr.inner)))
            .collect::<PyResult<_>>()?;

        // apply to the inner expression
        match self.inner.subs(&map, allow_unknown_parameters) {
            Ok(subbed) => Ok(subbed.into()),
            Err(e) => Err(e.into()),
        }
    }

    /// Binds the provided set of parameters to their corresponding values.
    ///
    /// Args:
    ///     parameter_values: Mapping of :class:`.Parameter` instances to the numeric value to which
    ///         they will be bound.
    ///     allow_unknown_parameters: If ``False``, raises an error if ``parameter_values``
    ///         contains :class:`.Parameter`\ s in the keys outside those present in the expression.
    ///         If ``True``, any such parameters are simply ignored.
    ///
    /// Raises:
    ///     CircuitError:
    ///         - If parameter_values contains parameters outside those in self.
    ///         - If a non-numeric value is passed in ``parameter_values``.
    ///     ZeroDivisionError:
    ///         - If binding the provided values requires division by zero.
    ///
    /// Returns:
    ///     A new expression parameterized by any parameters which were not bound by
    ///     ``parameter_values``.
    #[pyo3(name = "bind")]
    #[pyo3(signature = (parameter_values, allow_unknown_parameters=false))]
    pub fn py_bind(
        &self,
        parameter_values: HashMap<PyParameter, Bound<PyAny>>,
        allow_unknown_parameters: bool,
    ) -> PyResult<Self> {
        // reduce the map to a HashMap<Symbol, Value>
        let map = parameter_values
            .iter()
            .map(|(param, value)| {
                let value = value.extract()?;
                Ok((param.0.as_ref(), value))
            })
            .collect::<PyResult<_>>()?;

        // apply to the inner expression
        match self.inner.bind(&map, allow_unknown_parameters) {
            Ok(bound) => Ok(bound.into()),
            Err(e) => Err(e.into()),
        }
    }

    /// Bind all of the parameters in ``self`` to numeric values in the dictionary, returning a
    /// numeric value.
    ///
    /// This is a special case of :meth:`bind` which can reach higher performance.  It is no problem
    /// for the ``values`` dictionary to contain parameters that are not used in this expression;
    /// the expectation is that the same bindings dictionary will be fed to other expressions as
    /// well.
    ///
    /// It is an error to call this method with a ``values`` dictionary that does not bind all of
    /// the values, or to call this method with non-numeric values, but this is not explicitly
    /// checked, since this method is intended for performance-sensitive use.  Passing an incorrect
    /// dictionary may result in unexpected behavior.
    ///
    /// Unlike :meth:`bind`, this method will not raise an exception if non-finite floating-point
    /// values are encountered.
    ///
    /// Args:
    ///     values: mapping of parameters to numeric values.
    #[pyo3(name = "bind_all")]
    #[pyo3(signature = (values, *))]
    pub fn py_bind_all(&self, values: Bound<PyAny>) -> PyResult<Value> {
        let mut partial_map = HashMap::with_capacity(self.inner.name_map.len());
        for symbol in self.inner.name_map.values() {
            let py_parameter = symbol.clone().into_pyobject(values.py())?;
            partial_map.insert(symbol, values.get_item(py_parameter)?.extract()?);
        }
        let bound = self.inner.expr.bind(&partial_map);
        bound.eval(true).ok_or_else(|| {
            PyTypeError::new_err(format!(
                "binding did not produce a numeric quantity: {bound:?}"
            ))
        })
    }

    /// Assign one parameter to a value, which can either be numeric or another parameter
    /// expression.
    ///
    /// Args:
    ///     parameter: A parameter in this expression whose value will be updated.
    ///     value: The new value to bind to.
    ///
    /// Returns:
    ///     A new expression parameterized by any parameters which were not bound by assignment.
    #[pyo3(name = "assign")]
    pub fn py_assign(&self, parameter: PyParameter, value: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(expr) = value.cast::<Self>() {
            let map = [(parameter, expr.borrow().clone())].into_iter().collect();
            self.py_subs(map, false)
        } else if value.extract::<Value>().is_ok() {
            let map = [(parameter, value.clone())].into_iter().collect();
            self.py_bind(map, false)
        } else {
            Err(PyValueError::new_err(
                "Unexpected value in assign: {replacement:?}",
            ))
        }
    }

    #[inline]
    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // ParameterExpression is immutable.
        slf
    }

    #[inline]
    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // Everything a ParameterExpression contains is immutable.
        slf
    }

    pub fn __eq__(&self, rhs: &Bound<PyAny>) -> PyResult<bool> {
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            match rhs.inner.expr {
                SymbolExpr::Value(v) => match self.inner.try_to_value(false) {
                    Ok(e) => Ok(e == v),
                    Err(_) => Ok(false),
                },
                _ => Ok(self.inner.expr == rhs.inner.expr),
            }
        } else {
            Ok(false)
        }
    }

    #[inline]
    pub fn __abs__(&self) -> Self {
        self.inner.abs().into()
    }

    #[inline]
    pub fn __pos__(&self) -> Self {
        self.clone()
    }

    pub fn __neg__(&self) -> Self {
        Self {
            inner: ParameterExpression::new(-&self.inner.expr, self.inner.name_map.clone()),
        }
    }

    pub fn __add__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            Ok(self.inner.add(&rhs.inner)?.into())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __add__",
            ))
        }
    }

    pub fn __radd__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(lhs) = Self::extract_coerce(lhs.as_borrowed()) {
            Ok(lhs.inner.add(&self.inner)?.into())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __radd__",
            ))
        }
    }

    pub fn __sub__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            Ok(self.inner.sub(&rhs.inner)?.into())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __sub__",
            ))
        }
    }

    pub fn __rsub__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(lhs) = Self::extract_coerce(lhs.as_borrowed()) {
            Ok(lhs.inner.sub(&self.inner)?.into())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rsub__",
            ))
        }
    }

    pub fn __mul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            match self.inner.mul(&rhs.inner) {
                Ok(result) => PyParameterExpression::from(result).into_bound_py_any(py),
                Err(e) => Err(PyErr::from(e)),
            }
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }

    pub fn __rmul__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(lhs) = Self::extract_coerce(lhs.as_borrowed()) {
            Ok(lhs.inner.mul(&self.inner)?.into())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rmul__",
            ))
        }
    }

    pub fn __truediv__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            Ok(self.inner.div(&rhs.inner)?.into())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __truediv__",
            ))
        }
    }

    pub fn __rtruediv__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(lhs) = Self::extract_coerce(lhs.as_borrowed()) {
            Ok(lhs.inner.div(&self.inner)?.into())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rtruediv__",
            ))
        }
    }

    pub fn __pow__(&self, rhs: &Bound<PyAny>, _modulo: Option<i32>) -> PyResult<Self> {
        if let Ok(rhs) = Self::extract_coerce(rhs.as_borrowed()) {
            Ok(self.inner.pow(&rhs.inner)?.into())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __pow__",
            ))
        }
    }

    pub fn __rpow__(&self, lhs: &Bound<PyAny>, _modulo: Option<i32>) -> PyResult<Self> {
        if let Ok(lhs) = Self::extract_coerce(lhs.as_borrowed()) {
            Ok(lhs.inner.pow(&self.inner)?.into())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rpow__",
            ))
        }
    }

    pub fn __int__(&self) -> PyResult<i64> {
        match self.inner.try_to_value(false)? {
            Value::Complex(_) => Err(PyTypeError::new_err(
                "Cannot cast complex parameter to int.",
            )),
            Value::Real(r) => {
                let rounded = r.floor();
                Ok(rounded as i64)
            }
            Value::Int(i) => Ok(i),
        }
    }

    pub fn __float__(&self) -> PyResult<f64> {
        match self.inner.try_to_value(false)? {
            Value::Complex(c) => {
                if c.im.abs() > SYMEXPR_EPSILON {
                    Err(PyTypeError::new_err(
                        "Could not cast complex parameter expression to float.",
                    ))
                } else {
                    Ok(c.re)
                }
            }
            Value::Real(r) => Ok(r),
            Value::Int(i) => Ok(i as f64),
        }
    }

    pub fn __complex__(&self) -> PyResult<Complex64> {
        match self.inner.try_to_value(false)? {
            Value::Complex(c) => Ok(c),
            Value::Real(r) => Ok(Complex64::new(r, 0.)),
            Value::Int(i) => Ok(Complex64::new(i as f64, 0.)),
        }
    }

    pub fn __str__(&self) -> String {
        self.to_string()
    }

    pub fn __hash__(&self, py: Python) -> PyResult<u64> {
        match self.inner.try_to_value(false) {
            // if a value, we promise to match the hash of the raw value!
            Ok(value) => {
                let py_hash = BUILTIN_HASH.get_bound(py);
                match value {
                    Value::Complex(c) => py_hash.call1((c,))?.extract::<u64>(),
                    Value::Real(r) => py_hash.call1((r,))?.extract::<u64>(),
                    Value::Int(i) => py_hash.call1((i,))?.extract::<u64>(),
                }
            }
            Err(_) => {
                let mut hasher = DefaultHasher::new();
                self.inner.expr.string_id().hash(&mut hasher);
                Ok(hasher.finish())
            }
        }
    }

    #[staticmethod]
    fn _reconstruct(replay: Vec<OPReplay>) -> PyResult<Self> {
        Ok(Self {
            inner: ParameterExpression::from_qpy(&replay)?,
        })
    }
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        py.get_type::<Self>()
            .getattr("_reconstruct")
            .and_then(|meth| (meth, (self.inner.qpy_replay(),)).into_pyobject(py))
    }

    #[getter]
    fn _qpy_replay(&self) -> Vec<OPReplay> {
        self.inner.qpy_replay()
    }
}

/// A compile-time symbolic parameter.
///
/// The value of a :class:`.Parameter` must be entirely determined before a circuit begins execution.
/// Typically this will mean that you should supply values for all :class:`.Parameter`\ s in a
/// circuit using :meth:`.QuantumCircuit.assign_parameters`, though certain hardware vendors may
/// allow you to give them a circuit in terms of these parameters, provided you also pass the values
/// separately.
///
/// This is the atom of :class:`.ParameterExpression`, and is itself an expression.  The numeric
/// value of a parameter need not be fixed while the circuit is being defined.
///
/// Examples:
///
///     Construct a variable-rotation X gate using circuit parameters.
///
///     .. plot::
///         :alt: Circuit diagram output by the previous code.
///         :include-source:
///
///         from qiskit.circuit import QuantumCircuit, Parameter
///
///         # create the parameter
///         phi = Parameter("phi")
///         qc = QuantumCircuit(1)
///
///         # parameterize the rotation
///         qc.rx(phi, 0)
///         qc.draw("mpl")
///
///         # bind the parameters after circuit to create a bound circuit
///         bc = qc.assign_parameters({phi: 3.14})
///         bc.measure_all()
///         bc.draw("mpl")
#[pyclass(
    sequence,
    subclass,
    module="qiskit._accelerate.circuit",
    extends=PyParameterExpression,
    name="Parameter",
    from_py_object,
    frozen,
)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Hash)]
pub struct PyParameter(pub Arc<Symbol>);
impl From<Symbol> for PyParameter {
    fn from(val: Symbol) -> Self {
        Self(Arc::new(val))
    }
}

// This needs to be implemented manually, since PyO3 does not provide this conversion
// for subclasses.
impl<'py> IntoPyObject<'py> for PyParameter {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    /// `PyParameter`'s implementation of this is responsible for sorting out the vector-element
    /// logic as well.
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let is_element = matches!(&*self.0, Symbol::Element { .. });
        let inner = ParameterExpression::from_symbol_expr(SymbolExpr::Symbol(self.0.clone()));
        let init = PyClassInitializer::from(PyParameterExpression { inner }).add_subclass(self);
        if is_element {
            Bound::new(py, init.add_subclass(PyParameterVectorElement)).map(|ob| ob.into_any())
        } else {
            Bound::new(py, init).map(|ob| ob.into_any())
        }
    }
}

#[pymethods]
impl PyParameter {
    /// Args:
    ///     name: name of the parameter, used for visual representation. This can
    ///         be any Unicode string, e.g. ``"ϕ"``.
    ///     uuid: For advanced usage only.  Override the UUID of this parameter, in order to make it
    ///         compare equal to some other parameter object.  By default, two parameters with the
    ///         same name do not compare equal to help catch shadowing bugs when two circuits
    ///         containing the same named parameters are spurious combined.  Setting the ``uuid``
    ///         field when creating two parameters to the same thing (along with the same name)
    ///         allows them to be equal.  This is useful during serialization and deserialization.
    #[new]
    #[pyo3(signature = (name, uuid=None))]
    fn py_new(name: String, uuid: Option<Bound<PyAny>>) -> PyResult<PyClassInitializer<Self>> {
        let uuid = uuid.map(uuid_from_py).transpose()?;
        let symbol = Arc::new(Symbol::standalone(name, uuid));
        let inner = ParameterExpression::from_symbol_expr(SymbolExpr::Symbol(symbol.clone()));
        Ok(PyClassInitializer::from(PyParameterExpression { inner }).add_subclass(Self(symbol)))
    }

    /// Returns the name of the :class:`.Parameter`.
    #[getter]
    fn name<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        PyString::new(py, &self.0.repr(false))
    }

    /// Returns the :class:`~uuid.UUID` of the :class:`Parameter`.
    ///
    /// In advanced use cases, this property can be passed to the
    /// :class:`.Parameter` constructor to produce an instance that compares
    /// equal to another instance.
    #[getter]
    fn uuid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        uuid_to_py(py, self.0.uuid())
    }

    pub fn __repr__<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        let str = format!("Parameter({})", self.0.repr(false),);
        PyString::new(py, str.as_str())
    }

    pub fn __getnewargs__(&self) -> (String, u128) {
        (self.0.repr(false), self.0.uuid().as_u128())
    }

    // In Python space we inherit from `PyParameterExpression`, but we don't want its `__reduce__`
    // to take effect.  This statement is the same as the Python statement:
    //
    // class Parameter:
    //     __reduce__ = object.__reduce__
    //
    // which is a standard way of overriding with the default.
    #[classattr]
    fn __reduce__(py: Python) -> PyResult<Bound<PyAny>> {
        py.get_type::<PyAny>().getattr("__reduce__")
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // Parameter is immutable. Note that this **cannot** be deferred to the parent class
        // since PyO3 would then always return the parent type.
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // Everything inside a Parameter is immutable. Note that this **cannot** be deferred to the
        // parent class since PyO3 would then always return the parent type.
        slf
    }

    #[pyo3(name = "subs")]
    #[pyo3(signature = (parameter_map, allow_unknown_parameters=false))]
    pub fn py_subs<'py>(
        &self,
        py: Python<'py>,
        parameter_map: HashMap<PyParameter, Bound<'py, PyAny>>,
        allow_unknown_parameters: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // We implement this method on this class, and do not defer to the parent, such
        // that x.subs({x: y}) remains a Parameter, and is not upgraded to an expression.
        // Also this should be faster than going via ParameterExpression, which constructs
        // intermediary HashMaps we don't need here.
        match parameter_map.get(self) {
            None => {
                if allow_unknown_parameters {
                    self.clone().into_bound_py_any(py)
                } else {
                    Err(CircuitError::new_err(
                        "Cannot bind parameters not present in parameter.",
                    ))
                }
            }
            Some(replacement) => {
                if allow_unknown_parameters || parameter_map.len() == 1 {
                    Ok(replacement.clone())
                } else {
                    Err(CircuitError::new_err(
                        "Cannot bind parameters not present in parameter.",
                    ))
                }
            }
        }
    }

    #[pyo3(name = "bind")]
    #[pyo3(signature = (parameter_values, allow_unknown_parameters=false))]
    pub fn py_bind<'py>(
        &self,
        py: Python<'py>,
        parameter_values: HashMap<PyParameter, Bound<'py, PyAny>>,
        allow_unknown_parameters: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Returns PyAny to cover Parameter and ParameterExpression(value).
        match parameter_values.get(self) {
            None => {
                if allow_unknown_parameters {
                    self.clone().into_bound_py_any(py)
                } else {
                    Err(CircuitError::new_err(
                        "Cannot bind parameters not present in parameter.",
                    ))
                }
            }
            Some(replacement) => {
                if allow_unknown_parameters || parameter_values.len() == 1 {
                    let expr = PyParameterExpression::extract_coerce(replacement.as_borrowed())?;
                    if let SymbolExpr::Value(_) = &expr.inner.expr {
                        expr.clone().into_bound_py_any(py)
                    } else {
                        Err(PyValueError::new_err("Invalid binding value."))
                    }
                } else {
                    Err(CircuitError::new_err(
                        "Cannot bind parameters not present in parameter.",
                    ))
                }
            }
        }
    }

    #[pyo3(name = "bind_all")]
    #[pyo3(signature = (values, *))]
    pub fn py_bind_all<'py>(
        slf_: Bound<'py, Self>,
        values: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        values.get_item(slf_)
    }

    #[pyo3(name = "assign")]
    pub fn py_assign<'py>(
        &self,
        py: Python<'py>,
        parameter: PyParameter,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if value.cast::<PyParameterExpression>().is_ok() {
            let map = [(parameter, value.clone())].into_iter().collect();
            self.py_subs(py, map, false)
        } else if value.extract::<Value>().is_ok() {
            let map = [(parameter, value.clone())].into_iter().collect();
            self.py_bind(py, map, false)
        } else {
            Err(PyValueError::new_err(
                "Unexpected value in assign: {replacement:?}",
            ))
        }
    }
}

/// An element of a :class:`.ParameterVector`.
///
/// .. note::
///     There is very little reason to ever construct this class directly.  Objects of this type are
///     automatically constructed efficiently as part of creating a :class:`.ParameterVector`.
#[pyclass(
    sequence,
    subclass,
    module="qiskit._accelerate.circuit",
    extends=PyParameter,
    name="ParameterVectorElement",
    skip_from_py_object
)]
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd)]
// This carries no data in Rust space; `PyParameter`/`Symbol` already has it all.  It's just a
// Python type-level marker.  Nothing in Rust space should actually use this.
pub(crate) struct PyParameterVectorElement;

#[pymethods]
impl PyParameterVectorElement {
    #[new]
    #[pyo3(signature = (vector, index, uuid=None))]
    pub fn py_new<'py>(
        py: Python<'py>,
        vector: &PyParameterVector,
        index: usize,
        uuid: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if uuid.is_some() {
            let msg = intern!(
                py,
                "constructing a `ParameterVectorElement` with an explicit UUID is no longer \
                supported; UUIDs are always derived from the base vector."
            );
            WARNINGS_WARN
                .get_bound(py)
                .call1((msg, py.get_type::<PyDeprecationWarning>()))?;
        }
        let symbol = Symbol::Element {
            index,
            base: vector.base.clone(),
        };
        PyParameter(Arc::new(symbol)).into_py_any(py)
    }

    pub fn __getnewargs__(self_: PyRef<Self>) -> (PyParameterVector, usize) {
        match &*self_.as_super().0 {
            Symbol::Standalone { .. } => {
                panic!("internal logic error: standalone symbols cannot become vector elements")
            }
            Symbol::Element { base, index } => (PyParameterVector::new(base.clone()), *index),
        }
    }

    pub fn __repr__(self_: PyRef<Self>) -> String {
        format!("ParameterVectorElement({})", self_.as_super().0.repr(false))
    }

    /// Get the index of this element in the parent vector.
    #[getter]
    pub fn index(self_: PyRef<Self>) -> usize {
        match &*self_.as_super().0 {
            Symbol::Standalone { .. } => {
                panic!("internal logic error: standalone symbols cannot become vector elements")
            }
            Symbol::Element { index, base: _ } => *index,
        }
    }

    /// Get the parent vector instance.
    #[getter]
    pub fn vector(self_: PyRef<Self>) -> PyParameterVector {
        match &*self_.as_super().0 {
            Symbol::Standalone { .. } => {
                panic!("internal logic error: standalone symbols cannot become vector elements")
            }
            Symbol::Element { base, index: _ } => PyParameterVector::new(base.clone()),
        }
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // ParameterVectorElement is immutable.
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // Everything a ParameterVectorElement contains is immutable.
        slf
    }
}

/// A container of many related :class:`Parameter` objects.
///
/// This class is faster to construct than constructing many :class:`.Parameter` objects
/// individually, and the individual names of the parameters will all share a common stem (the name
/// of the vector).  For a vector called ``v`` with length 3, the individual elements will have
/// names ``v[0]``, ``v[1]`` and ``v[2]``.
///
/// The elements of a vector are sorted by the name of the vector, then the numeric value of their
/// index.
///
/// This class fulfills the :class:`collections.abc.Sequence` interface.
#[pyclass(
    sequence,
    module = "qiskit._accelerate.circuit",
    name = "ParameterVector",
    skip_from_py_object
)]
pub struct PyParameterVector {
    base: Arc<SymbolVector>,
    /// Cache of the name list.
    name: PyOnceLock<Py<PyString>>,
    /// Cache of the parameter list.
    params: PyOnceLock<Py<PyList>>,
}
impl PyParameterVector {
    pub fn new(base: Arc<SymbolVector>) -> Self {
        Self {
            base,
            name: PyOnceLock::new(),
            params: PyOnceLock::new(),
        }
    }
}
#[pymethods]
impl PyParameterVector {
    #[new]
    #[pyo3(signature = (name, length=0, uuid=None))]
    fn py_new(name: Bound<PyString>, length: usize, uuid: Option<Bound<PyAny>>) -> PyResult<Self> {
        let base = Arc::new(SymbolVector {
            name: name.to_str()?.to_owned(),
            uuid: uuid
                .map(uuid_from_py)
                .transpose()?
                .unwrap_or_else(Uuid::new_v4),
            len: length.into(),
        });
        // Annoyingly, PyO3 doesn't have the same `impl From<T> for OnceLock<T>`.
        let name_cell = PyOnceLock::new();
        name_cell
            .set(name.py(), name.unbind())
            .expect("the line above is the only reference, so we cannot be racing a thread");
        Ok(Self {
            base,
            name: name_cell,
            params: PyOnceLock::new(),
        })
    }

    /// The base name of the vector.
    #[getter]
    fn name<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        self.name
            .get_or_init(py, || PyString::new(py, &self.base.name).unbind())
            .bind(py)
            .clone()
    }

    /// A sequence of the contained :class:`.ParameterVectorElement` instances.
    ///
    /// Mutations to this object are not tracked.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.params
            .get_or_try_init(py, || {
                PyList::new(py, self.base.iter().map(PyParameter::from)).map(|x| x.unbind())
            })
            .map(|ob| ob.bind(py).clone())
    }

    /// Get the root UUID of this vector.
    #[getter]
    fn uuid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        uuid_to_py(py, self.base.uuid)
    }

    // Historically, `ParameterVector` _did not_ implement `__eq__` and `__hash__`, or more
    // precisely, it inherited the `object` ones, which are based on pointer identity.  We can't
    // have the natural definitions of `__eq__` and `__hash__` while the mutable `resize` exists, so
    // instead we reinvent pointer-identity equality and hashing in a way that works with the
    // on-the-fly Python-object creation from raw `Arc<SymbolVector>` instances.
    fn __eq__(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.base, &other.base)
    }
    fn __hash__(&self) -> usize {
        Arc::as_ptr(&self.base).addr()
    }

    /// Find the index of a :class:`.ParameterVectorElement` within the sequence.
    ///
    /// This is an inefficient way of accessing the :attr:`~.ParameterVectorElement.index` field on
    /// the given value, which you should use instead.
    fn index(&self, val: &Bound<PyAny>) -> PyResult<usize> {
        self.params(val.py())?.index(val)
    }
    fn __getitem__<'py>(&self, val: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.params(val.py())?.as_any().get_item(val)
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.params(py)?.as_any().try_iter()
    }
    fn __len__(&self) -> usize {
        self.base.len.load(atomic::Ordering::Relaxed)
    }

    fn __str__(&self) -> String {
        // Yeah, I don't know why we do this either, but it's been like this ever since
        // `ParameterVector` was first introduced.
        format!(
            "{}, [{}]",
            self.base.name,
            self.base
                .iter()
                .map(|s| format!("'{}'", s.fullname()))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
    fn __repr__(&self) -> String {
        format!(
            "ParameterVector(name='{}', length={})",
            self.base.name,
            self.base.len.load(atomic::Ordering::Relaxed)
        )
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            self.name(py),
            self.base.len.load(atomic::Ordering::Relaxed),
            uuid_to_py(py, self.base.uuid)?,
        )
            .into_pyobject(py)
    }

    /// Change the length of this vector.
    ///
    /// Any references to :class:`.ParameterVectorElement` instances with lengths greater than
    /// ``length`` are invalidated by calling this.
    ///
    /// The UUIDs of :class:`.ParameterVectorElement` instances that are deleted or created by this
    /// method are stable; the UUIDs are always derived as index offsets from the vector's UUID.
    fn resize(&mut self, length: usize) {
        // This can actually allow existing Rust-space `Symbol` objects derived from the prior
        // `SymbolVector` to become out of sync with Python space, because we replace the internal
        // vector here rather than propagating it through everywhere.
        //
        // This whole method is not ideal, but it has very limited uses remaining in
        // `BlueprintCircuit`, which is deprecated pending removal in Qiskit 3.0.
        _ = self.params.take();
        self.base.len.store(length, atomic::Ordering::Relaxed)
    }
}

/// Try to extract a Uuid from a Python object, which could be a Python UUID or int.
fn uuid_from_py(uuid: Bound<PyAny>) -> PyResult<Uuid> {
    let int = if uuid.is_exact_instance(UUID.get_bound(uuid.py())) {
        uuid.getattr("int")?
    } else {
        uuid
    };
    Ok(Uuid::from_u128(int.extract::<u128>()?))
}

/// Convert a Rust Uuid object to a Python UUID object.
fn uuid_to_py(py: Python<'_>, uuid: Uuid) -> PyResult<Bound<'_, PyAny>> {
    let uuid = uuid.as_u128();
    let kwargs = [("int", uuid)].into_py_dict(py)?;
    UUID.get_bound(py).call((), Some(&kwargs))
}

/// A singular parameter value used for QPY serialization. This covers anything
/// but a [PyParameterExpression], which is represented by [None] in the serialization.
#[derive(IntoPyObject, FromPyObject, Clone, Debug)]
pub enum ParameterValueType {
    Int(i64),
    Float(f64),
    Complex(Complex64),
    Parameter(PyParameter),
}

impl ParameterValueType {
    fn extract_from_expr(expr: &SymbolExpr) -> Option<ParameterValueType> {
        if let Some(value) = expr.eval(true) {
            match value {
                Value::Int(i) => Some(ParameterValueType::Int(i)),
                Value::Real(r) => Some(ParameterValueType::Float(r)),
                Value::Complex(c) => Some(ParameterValueType::Complex(c)),
            }
        } else if let SymbolExpr::Symbol(symbol) = expr {
            Some(Self::Parameter(PyParameter(symbol.clone())))
        } else {
            // ParameterExpressions have the value None, as they must be constructed
            None
        }
    }
}

impl From<ParameterValueType> for ParameterExpression {
    fn from(value: ParameterValueType) -> Self {
        match value {
            ParameterValueType::Parameter(PyParameter(symbol)) => {
                let expr = SymbolExpr::Symbol(symbol);
                Self::from_symbol_expr(expr)
            }
            ParameterValueType::Int(i) => {
                let expr = SymbolExpr::Value(Value::Int(i));
                Self::from_symbol_expr(expr)
            }
            ParameterValueType::Float(f) => {
                let expr = SymbolExpr::Value(Value::Real(f));
                Self::from_symbol_expr(expr)
            }
            ParameterValueType::Complex(c) => {
                let expr = SymbolExpr::Value(Value::Complex(c));
                Self::from_symbol_expr(expr)
            }
        }
    }
}

#[pyclass(module = "qiskit._accelerate.circuit", from_py_object)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum OpCode {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    POW = 4,
    SIN = 5,
    COS = 6,
    TAN = 7,
    ASIN = 8,
    ACOS = 9,
    EXP = 10,
    LOG = 11,
    SIGN = 12,
    GRAD = 13, // for backward compatibility, unused in Rust's ParameterExpression
    CONJ = 14,
    SUBSTITUTE = 15, // for backward compatibility, unused in Rust's ParameterExpression
    ABS = 16,
    ATAN = 17,
    RSUB = 18,
    RDIV = 19,
    RPOW = 20,
}

impl From<OpCode> for u8 {
    fn from(value: OpCode) -> Self {
        value as u8
    }
}

unsafe impl ::bytemuck::CheckedBitPattern for OpCode {
    type Bits = u8;

    #[inline(always)]
    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits <= 20
    }
}

unsafe impl ::bytemuck::NoUninit for OpCode {}

impl OpCode {
    pub fn from_u8(value: u8) -> PyResult<OpCode> {
        Ok(bytemuck::checked::cast::<u8, OpCode>(value))
    }
}

#[pymethods]
impl OpCode {
    #[new]
    fn py_new(value: u8) -> PyResult<Self> {
        let code: OpCode = ::bytemuck::checked::try_cast(value)
            .map_err(|_| ParameterError::InvalidU8ToOpCode(value))?;
        Ok(code)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        if let Ok(code) = other.cast::<OpCode>() {
            *code.borrow() == *self
        } else {
            false
        }
    }

    fn __hash__(&self) -> u8 {
        *self as u8
    }

    fn __getnewargs__(&self) -> (u8,) {
        (*self as u8,)
    }
}

// enum for QPY replay
#[pyclass(sequence, module = "qiskit._accelerate.circuit", from_py_object)]
#[derive(Clone, Debug)]
pub struct OPReplay {
    pub op: OpCode,
    pub lhs: Option<ParameterValueType>,
    pub rhs: Option<ParameterValueType>,
}

#[pymethods]
impl OPReplay {
    #[new]
    pub fn py_new(
        op: OpCode,
        lhs: Option<ParameterValueType>,
        rhs: Option<ParameterValueType>,
    ) -> OPReplay {
        OPReplay { op, lhs, rhs }
    }

    #[getter]
    fn op(&self) -> OpCode {
        self.op
    }

    #[getter]
    fn lhs(&self) -> Option<ParameterValueType> {
        self.lhs.clone()
    }

    #[getter]
    fn rhs(&self) -> Option<ParameterValueType> {
        self.rhs.clone()
    }

    fn __getnewargs__(
        &self,
    ) -> (
        OpCode,
        Option<ParameterValueType>,
        Option<ParameterValueType>,
    ) {
        (self.op, self.lhs.clone(), self.rhs.clone())
    }
}

/// Internal helper. Extract one part of the expression tree, keeping the name map up to date.
///
/// Example: Given expr1 + expr2, each being [PyParameterExpression], we need the ability to
/// extract one of the expressions with the proper name map.
///
/// Args:
///     - joint_parameter_expr: The full expression, e.g. expr1 + expr2.
///     - sub_expr: The sub expression, on whose symbols we restrict the name map.
fn filter_name_map(
    sub_expr: &SymbolExpr,
    name_map: &HashMap<String, Symbol>,
) -> ParameterExpression {
    let sub_symbols: HashSet<&Symbol> = sub_expr.iter_symbols().collect();
    let restricted_name_map: HashMap<String, Symbol> = name_map
        .iter()
        .filter(|(_, symbol)| sub_symbols.contains(*symbol))
        .map(|(name, symbol)| (name.clone(), symbol.clone()))
        .collect();

    ParameterExpression {
        expr: sub_expr.clone(),
        name_map: restricted_name_map,
    }
}

fn qpy_replay_inner(
    expr: &ParameterExpression,
    name_map: &HashMap<String, Symbol>,
    replay: &mut Vec<OPReplay>,
    unused: &mut IndexSet<Symbol>,
) {
    match &expr.expr {
        // This function is written under the assumption that the top-level expression involves an
        // operation, since `OPReplay` items correspond to operations that own their operands.
        SymbolExpr::Value(_) => (),
        SymbolExpr::Symbol(sym) => {
            unused.swap_remove(sym.as_ref());
        }
        SymbolExpr::Unary { op, expr } => {
            let op = match op {
                symbol_expr::UnaryOp::Abs => OpCode::ABS,
                symbol_expr::UnaryOp::Acos => OpCode::ACOS,
                symbol_expr::UnaryOp::Asin => OpCode::ASIN,
                symbol_expr::UnaryOp::Atan => OpCode::ATAN,
                symbol_expr::UnaryOp::Conj => OpCode::CONJ,
                symbol_expr::UnaryOp::Cos => OpCode::COS,
                symbol_expr::UnaryOp::Exp => OpCode::EXP,
                symbol_expr::UnaryOp::Log => OpCode::LOG,
                symbol_expr::UnaryOp::Neg => OpCode::MUL,
                symbol_expr::UnaryOp::Sign => OpCode::SIGN,
                symbol_expr::UnaryOp::Sin => OpCode::SIN,
                symbol_expr::UnaryOp::Tan => OpCode::TAN,
            };
            // TODO filter shouldn't be necessary for unary ops
            let lhs = filter_name_map(expr, name_map);

            // recurse on the instruction
            qpy_replay_inner(&lhs, name_map, replay, unused);

            let lhs_value = ParameterValueType::extract_from_expr(expr);

            // MUL is special: we implement ``neg`` as multiplication by -1
            if let OpCode::MUL = &op {
                replay.push(OPReplay {
                    op,
                    lhs: lhs_value,
                    rhs: Some(ParameterValueType::Int(-1)),
                });
            } else {
                replay.push(OPReplay {
                    op,
                    lhs: lhs_value,
                    rhs: None,
                });
            }
        }
        SymbolExpr::Binary { op, lhs, rhs } => {
            let lhs_value = ParameterValueType::extract_from_expr(lhs);
            let rhs_value = ParameterValueType::extract_from_expr(rhs);

            // recurse on the parameter expressions
            let lhs = filter_name_map(lhs, name_map);
            let rhs = filter_name_map(rhs, name_map);
            qpy_replay_inner(&lhs, name_map, replay, unused);
            qpy_replay_inner(&rhs, name_map, replay, unused);

            // add the expression to the replay
            match lhs_value {
                Some(ParameterValueType::Parameter(_)) => {
                    // For non-commutative operations (SUB, DIV, POW): if LHS is a Parameter and RHS is
                    // an expression (None), we need to use reverse operations (RSUB, RDIV, RPOW)
                    let op = match op {
                        symbol_expr::BinaryOp::Add => OpCode::ADD,
                        symbol_expr::BinaryOp::Sub => {
                            // If RHS is None (an expression), use RSUB
                            if rhs_value.is_none() {
                                OpCode::RSUB
                            } else {
                                OpCode::SUB
                            }
                        }
                        symbol_expr::BinaryOp::Mul => OpCode::MUL,
                        symbol_expr::BinaryOp::Div => {
                            // If RHS is None (an expression), use RDIV
                            if rhs_value.is_none() {
                                OpCode::RDIV
                            } else {
                                OpCode::DIV
                            }
                        }
                        symbol_expr::BinaryOp::Pow => {
                            // If RHS is None (an expression), use RPOW
                            if rhs_value.is_none() {
                                OpCode::RPOW
                            } else {
                                OpCode::POW
                            }
                        }
                    };
                    if op == OpCode::RPOW || op == OpCode::RDIV || op == OpCode::RSUB {
                        // For reverse operations, swap lhs and rhs (Python's sympify will swap again to get correct order)
                        replay.push(OPReplay {
                            op,
                            lhs: rhs_value,
                            rhs: lhs_value,
                        });
                    } else {
                        replay.push(OPReplay {
                            op,
                            lhs: lhs_value,
                            rhs: rhs_value,
                        });
                    }
                }
                None => {
                    // When LHS is an expression (None), use normal operations
                    let op = match op {
                        symbol_expr::BinaryOp::Add => OpCode::ADD,
                        symbol_expr::BinaryOp::Sub => OpCode::SUB,
                        symbol_expr::BinaryOp::Mul => OpCode::MUL,
                        symbol_expr::BinaryOp::Div => OpCode::DIV,
                        symbol_expr::BinaryOp::Pow => OpCode::POW,
                    };
                    replay.push(OPReplay {
                        op,
                        lhs: lhs_value,
                        rhs: rhs_value,
                    });
                }
                _ => {
                    let op = match op {
                        symbol_expr::BinaryOp::Add => OpCode::ADD,
                        symbol_expr::BinaryOp::Sub => OpCode::RSUB,
                        symbol_expr::BinaryOp::Mul => OpCode::MUL,
                        symbol_expr::BinaryOp::Div => OpCode::RDIV,
                        symbol_expr::BinaryOp::Pow => OpCode::RPOW,
                    };
                    if let OpCode::ADD | OpCode::MUL = op {
                        replay.push(OPReplay {
                            op,
                            lhs: lhs_value,
                            rhs: rhs_value,
                        });
                    } else {
                        // this covers RSUB, RDIV, RPOW, hence we swap lhs and rhs
                        replay.push(OPReplay {
                            op,
                            lhs: rhs_value,
                            rhs: lhs_value,
                        });
                    }
                }
            }
        }
    }
}
