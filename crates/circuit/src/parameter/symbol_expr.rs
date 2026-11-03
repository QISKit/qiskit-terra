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

use std::borrow::Cow;
use std::cmp::{Ord, Ordering, PartialOrd};
use std::convert::From;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, LazyLock, atomic};
use std::{fmt, mem};
use uuid::Uuid;

use hashbrown::HashMap;
use num_complex::Complex64;
use pyo3::prelude::*;

use crate::parameter::parameter_expression::PyParameter;

// epsilon for SymbolExpr is heuristically defined
pub const SYMEXPR_EPSILON: f64 = f64::EPSILON * 8.0;

#[derive(Debug, Clone)]
pub enum Symbol {
    Standalone {
        name: String,
        uuid: Uuid,
    },
    Element {
        index: usize,
        base: Arc<SymbolVector>,
    },
}
impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ord_key().cmp(&other.ord_key())
    }
}
// These derived versions of `PartialEq` and `Hash` are because `SymbolVector` is (unfortunately)
// internally mutable to support the Python-space `ParameterVector.resize` propagating its mutations
// through to the backreferences, but happily Python-space never actually checked if the
// `ParameterVector` backrefs inside `ParameterVectorElement` were equal, so we don't either.  Once
// `ParameterVector.resize` is gone, we can make this more sane by deriving `Eq` and `Hash`.
impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.ord_key() == other.ord_key()
    }
}
impl Eq for Symbol {}
impl Hash for Symbol {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ord_key().hash(state)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Symbol {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self::clone(&ob.cast::<PyParameter>()?.borrow().0))
    }
}

impl<'py> IntoPyObject<'py> for Symbol {
    type Target = <PyParameter as IntoPyObject<'py>>::Target;
    type Output = <PyParameter as IntoPyObject<'py>>::Output;
    type Error = <PyParameter as IntoPyObject<'py>>::Error;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyParameter(Arc::new(self)).into_pyobject(py)
    }
}

impl Symbol {
    /// Create a new standalone (not in a vector) symbol.
    pub fn standalone(name: String, uuid: Option<Uuid>) -> Self {
        let uuid = uuid.unwrap_or_else(Uuid::new_v4);
        Self::Standalone { name, uuid }
    }

    /// The base name of the symbol, without any vector-index suffix.
    ///
    /// This does not include the vector name.  Use [`fullname`] if you need that for display.
    pub fn name(&self) -> &str {
        match self {
            Self::Standalone { name, uuid: _ } => name.as_str(),
            Self::Element { base, index: _ } => base.name.as_str(),
        }
    }

    pub fn fullname(&self) -> Cow<'_, str> {
        match self {
            Self::Standalone { name, uuid: _ } => Cow::Borrowed(name.as_str()),
            Self::Element { base, index } => Cow::Owned(format!("{}[{}]", base.name, index)),
        }
    }

    pub fn uuid(&self) -> Uuid {
        match self {
            Self::Standalone { uuid, name: _ } => *uuid,
            Self::Element { base, index } => Uuid::from_u128(base.uuid.as_u128() + *index as u128),
        }
    }

    pub fn repr(&self, with_uuid: bool) -> String {
        if with_uuid {
            format!("{}_{}", self.fullname(), self.uuid().as_u128())
        } else {
            self.fullname().into_owned()
        }
    }

    /// The key object to use for all comparison operations with this object.
    pub fn ord_key(&self) -> impl Ord + Hash + use<'_> {
        (
            // This is the base name, without the vector-index appended, because we use those
            // numerically.
            self.name(),
            match self {
                Self::Standalone { .. } => None,
                // Everything about `base` that we actually use is inside `name` and `uuid`; we've
                // never actually required that the backrefs inside vector elements are to equal
                // vectors (!).  While `ParameterExpression.resize` still exists, this actually is
                // useful for us: we can maintain equality checks through serialisation roundtrips.
                Self::Element { index, base: _ } => Some(index),
            },
            self.uuid(),
        )
    }
}

/// Vector of multiple parameters.
///
/// This makes creating large numbers of parameters much more efficient; we no longer need to
/// allocate individual strings and UUIDs for each symbol, but instead derive them all from the
/// shared base.
///
/// This object is intended to be used almost exclusively behind an `Arc`, since all derived
/// symbols from it need an `Arc` backreference.
#[derive(Debug)]
pub struct SymbolVector {
    pub name: String,
    pub uuid: Uuid,
    /// Length of the vector.
    ///
    /// This is `AtomicUsize` to support the (unfortunate) Python-space method
    /// `ParameterVector.resize`.  Once that method is deprecated and removed, this can become a
    /// regular `usize` (and we'll be able to derive `Hash`, if we want).
    pub len: atomic::AtomicUsize,
}
impl SymbolVector {
    pub fn new(name: String, len: usize) -> Arc<Self> {
        Arc::new(Self {
            name,
            uuid: Uuid::new_v4(),
            len: len.into(),
        })
    }

    pub fn get(self: &Arc<Self>, index: usize) -> Option<Symbol> {
        (index < self.len.load(atomic::Ordering::Relaxed)).then(|| Symbol::Element {
            index,
            base: Arc::clone(self),
        })
    }

    pub fn iter(self: &Arc<Self>) -> impl ExactSizeIterator<Item = Symbol> {
        (0..self.len.load(atomic::Ordering::Relaxed)).map(|index| {
            self.get(index)
                .expect("the iterator is only over valid values")
        })
    }
}
impl Clone for SymbolVector {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            uuid: self.uuid,
            len: self.len.load(atomic::Ordering::Relaxed).into(),
        }
    }
}

/// node types of expression tree
#[derive(Debug, Clone)]
pub enum SymbolExpr {
    Symbol(Arc<Symbol>),
    Value(Value),
    Unary {
        op: UnaryOp,
        expr: Arc<SymbolExpr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Arc<SymbolExpr>,
        rhs: Arc<SymbolExpr>,
    },
}

// The derived `Drop` is naturally recursive, which is slow and will overflow the stack on
// deeply nested expressions.  We instead need to iterate through a dropping expression, clearing
// out the recursive structure iteratively in a safe order before allowing the final drop glue.
impl Drop for SymbolExpr {
    fn drop(&mut self) {
        // There's no recursive worries for `Symbol` or `Value`, or the `UnaryOp` or `BinaryOp`
        // elements; it's just the `Arc<SymbolExpr>` parts that are an issue.  `Unary` is easy
        // enough; we replace `expr` with a non-recursive object, allow _that_ `Unary` to drop, then
        // iterate onto the previous content of `expr`.  With `Binary`, we need to depth-first
        // traverse the drops, but we _must_ do that non-recursively, and we _want_ to do that
        // without allocating in the `Drop` implementation.
        //
        // Without loss of generality, let's assume we're now dealing only with `Binary`.  We
        // traverse the nodes in depth-first, left-first order, and drop each node in post order.
        // We "reuse" the `lhs` pointer of each `Binary` we need to walk through to store its
        // parent.

        // Arbitrary `Arc<SymbolExpr>` we use that never drops (or would be non-recursive if it
        // does).  We use this in the way a sole-owning tree might use a null pointer.  There's only
        // ever one of these per process, so on average there's no heap allocation per `Drop` call.
        static NULL: LazyLock<Arc<SymbolExpr>> =
            LazyLock::new(|| Arc::new(SymbolExpr::Value(Value::Int(0))));

        let mut cur: Option<Arc<Self>>;
        // Always a `Binary` variant; we never need to backtrack to it if it's 0- or 1-ary.
        let mut cur_parent: Option<Arc<Self>> = None;
        // If the root node is a `Binary`, we put its rhs here, because we can't store `self` in
        // `cur_parent` without an `Arc`.
        let mut root_rhs: Option<Arc<Self>>;

        (cur, root_rhs) = match self {
            Self::Symbol(_) | Self::Value(_) => (None, None),
            Self::Unary { op: _, expr } => (Some(mem::replace(expr, Arc::clone(&NULL))), None),
            Self::Binary { op: _, lhs, rhs } => (
                Some(mem::replace(lhs, Arc::clone(&NULL))),
                Some(mem::replace(rhs, Arc::clone(&NULL))),
            ),
        };

        // Call when `cur` is completely ready or unable to drop, and we need to walk up the tree
        // and find the next node that needs to be cleared out.  `cur` should always be `None` at
        // the point that this function is called, but we have to pass ownership of the reference
        // into the function to prove the lifetimes around access to it are valid.
        let mut backtrack = |parent: &mut Option<Arc<Self>>| {
            while let Some(mut parent_arc) = parent.take() {
                let parent_inner = Arc::get_mut(&mut parent_arc)
                    .expect("only `Arc`s set by `Arc::make_mut` can be backtrack parents");
                let Self::Binary { op: _, lhs, rhs } = parent_inner else {
                    panic!("internal logic error: backtrack parents must always be `Binary`");
                };
                // During a backtrack, the `lhs` is always fully visited.  The `rhs` might not be.
                if Arc::ptr_eq(rhs, &NULL) {
                    // `rhs` is visited; let `parent_arc` drop and go up a level (if there is one).
                    *parent =
                        (!Arc::ptr_eq(lhs, &NULL)).then(|| mem::replace(lhs, Arc::clone(&NULL)));
                } else {
                    // `rhs` isn't visited; put ourselves back as the parent, and return the `rhs`
                    // for iteration.
                    let cur = Some(mem::replace(rhs, Arc::clone(&NULL)));
                    *parent = Some(parent_arc);
                    return cur;
                }
            }
            // If we get here, we've exhausted the whole `cur` tree, so if there's any remaining
            // `rhs` from the root, swap to it.
            root_rhs.take()
        };

        // Walk down the left edges of the current tree until we reach something that can either
        // drop non-recursively, or isn't eligible to drop.  Drop our reference to it, and then walk
        // back up the tree to the nearest `rhs` edge that hasn't been taken yet.
        while let Some(mut cur_arc) = cur.take() {
            // This `strong_count`/`make_mut` form is imperfect and if there are `Weak` pointers, it
            // might cause us to do a fairly cheap extra Clone+Drop (if a racing `Weak` upgrades to
            // an `Arc` between the count and the `make_mut`), or make a new extra `Arc` allocation
            // (otherwise).  However, we don't expect any `Weak`s to exist, and if they do, this
            // should still be safe, just a little less efficient.
            //
            // We have to mutate the inner `cur` both to clear out the recursive items, and to
            // (temporarily) use its `lhs` space as a storage location for a linked list of back
            // refs through the parents.
            if Arc::strong_count(&cur_arc) > 1 {
                // `cur_arc` isn't eligible to drop because of other references.  We drop our copy
                // by letting it go out of scope, and continue.
                cur = backtrack(&mut cur_parent);
                continue;
            }
            match Arc::make_mut(&mut cur_arc) {
                Self::Symbol(_) | Self::Value(_) => {
                    // `cur_arc`/`cur_inner` can drop without risking recursion because it's a base
                    // case.  It drops simply by going out of scope.
                    cur = backtrack(&mut cur_parent)
                }
                Self::Unary { op: _, expr } => {
                    // This is like "contracting" the edge through a `Unary`; we put a null object
                    // into the actual `Unary`, let `cur_inner` drop (so guaranteed one level of
                    // recursion), and pretend its parent node pointed directly to its child.
                    cur = Some(mem::replace(expr, Arc::clone(&NULL)));
                }
                Self::Binary { op: _, lhs, rhs: _ } => {
                    // `parent` is the parent of this node; we store a backref in `lhs` so we can
                    // walk back up later.
                    let parent = cur_parent.take().unwrap_or_else(|| Arc::clone(&NULL));
                    cur = Some(mem::replace(lhs, parent));
                    cur_parent = Some(cur_arc);
                }
            }
        }
    }
}

/// Value type, can be integer, real or complex number
#[derive(Debug, Clone, Copy, IntoPyObject, IntoPyObjectRef)]
pub enum Value {
    Real(f64),
    Int(i64),
    Complex(Complex64),
}

impl<'a, 'py> FromPyObject<'a, 'py> for Value {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(i) = ob.extract::<i64>() {
            Ok(Value::Int(i))
        } else if let Ok(r) = ob.extract::<f64>() {
            Ok(Value::Real(r))
        } else {
            ob.extract::<Complex64>().map(Value::Complex)
        }
    }
}

/// definition of unary operations
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Abs,
    Neg,
    Sin,
    Asin,
    Cos,
    Acos,
    Tan,
    Atan,
    Exp,
    Log,
    Sign,
    Conj,
}

/// definition of binary operations
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

// functions to make new expr for add
#[inline(always)]
fn _add(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    if rhs.is_negative() {
        match rhs.neg_opt() {
            Some(e) => SymbolExpr::Binary {
                op: BinaryOp::Sub,
                lhs: Arc::new(lhs),
                rhs: Arc::new(e),
            },
            None => SymbolExpr::Binary {
                op: BinaryOp::Sub,
                lhs: Arc::new(lhs),
                rhs: Arc::new(_neg(rhs)),
            },
        }
    } else if matches!(lhs, SymbolExpr::Value(_)) || !matches!(rhs, SymbolExpr::Value(_)) {
        SymbolExpr::Binary {
            op: BinaryOp::Add,
            lhs: Arc::new(lhs),
            rhs: Arc::new(rhs),
        }
    } else {
        SymbolExpr::Binary {
            op: BinaryOp::Add,
            lhs: Arc::new(rhs),
            rhs: Arc::new(lhs),
        }
    }
}

// functions to make new expr for sub
#[inline(always)]
fn _sub(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    if rhs.is_negative() {
        match rhs.neg_opt() {
            Some(e) => SymbolExpr::Binary {
                op: BinaryOp::Add,
                lhs: Arc::new(lhs),
                rhs: Arc::new(e),
            },
            None => SymbolExpr::Binary {
                op: BinaryOp::Add,
                lhs: Arc::new(lhs),
                rhs: Arc::new(_neg(rhs)),
            },
        }
    } else {
        SymbolExpr::Binary {
            op: BinaryOp::Sub,
            lhs: Arc::new(lhs),
            rhs: Arc::new(rhs),
        }
    }
}

// functions to make new expr for mul
#[inline(always)]
fn _mul(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary {
        op: BinaryOp::Mul,
        lhs: Arc::new(lhs),
        rhs: Arc::new(rhs),
    }
}

// functions to make new expr for div
#[inline(always)]
fn _div(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary {
        op: BinaryOp::Div,
        lhs: Arc::new(lhs),
        rhs: Arc::new(rhs),
    }
}

// functions to make new expr for pow
#[inline(always)]
fn _pow(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary {
        op: BinaryOp::Pow,
        lhs: Arc::new(lhs),
        rhs: Arc::new(rhs),
    }
}

// functions to make new expr for neg
#[inline(always)]
fn _neg(expr: SymbolExpr) -> SymbolExpr {
    match expr.neg_opt() {
        Some(e) => e,
        None => SymbolExpr::Unary {
            op: UnaryOp::Neg,
            expr: Arc::new(expr),
        },
    }
}

impl fmt::Display for SymbolExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr(false))
    }
}

impl SymbolExpr {
    /// bind value to symbol node
    pub fn bind(&self, maps: &HashMap<&Symbol, Value>) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => match maps.get(e.as_ref()) {
                Some(v) => SymbolExpr::Value(*v),
                None => self.clone(),
            },
            SymbolExpr::Value(e) => SymbolExpr::Value(*e),
            SymbolExpr::Unary { op, expr } => SymbolExpr::Unary {
                op: op.clone(),
                expr: Arc::new(expr.bind(maps)),
            },
            SymbolExpr::Binary { op, lhs, rhs } => {
                let new_lhs = lhs.bind(maps);
                let new_rhs = rhs.bind(maps);
                match op {
                    BinaryOp::Add => new_lhs + new_rhs,
                    BinaryOp::Sub => new_lhs - new_rhs,
                    BinaryOp::Mul => new_lhs * new_rhs,
                    BinaryOp::Div => new_lhs / new_rhs,
                    BinaryOp::Pow => _pow(new_lhs, new_rhs),
                }
            }
        }
    }

    /// substitute symbol node to other expression
    /// allows unknown expressions
    /// does not allow duplicate names with different UUID
    pub fn subs(&self, maps: &HashMap<Symbol, SymbolExpr>) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => match maps.get(e.as_ref()) {
                Some(v) => v.clone(),
                None => self.clone(),
            },
            SymbolExpr::Value(e) => SymbolExpr::Value(*e),
            SymbolExpr::Unary { op, expr } => SymbolExpr::Unary {
                op: op.clone(),
                expr: Arc::new(expr.subs(maps)),
            },
            SymbolExpr::Binary { op, lhs, rhs } => {
                let new_lhs = lhs.subs(maps);
                let new_rhs = rhs.subs(maps);
                match op {
                    BinaryOp::Add => new_lhs + new_rhs,
                    BinaryOp::Sub => new_lhs - new_rhs,
                    BinaryOp::Mul => new_lhs * new_rhs,
                    BinaryOp::Div => new_lhs / new_rhs,
                    BinaryOp::Pow => _pow(new_lhs, new_rhs),
                }
            }
        }
    }

    /// Are these two expressions precisely equal in structure?
    ///
    /// This is stricter but faster form of the [`PartialEq`] implementation that does not consider
    /// symbolic equality via simplification, but instead requires that the exact structures are
    /// constructed in the same manner.
    pub fn eq_exact(&self, other: &Self) -> bool {
        // Helper so `rustfmt` doesn't turn `if (a != b) { return false; }` into 3 lines.
        macro_rules! return_if_unequal {
            ($a:expr, $b:expr) => {
                if $a != $b {
                    return false;
                }
            };
        }
        // `backtrack` stores the right-hand sides of `Binary`s, so we can walk back up to them.
        // `cur` is a minor optimisation to avoid allocations in the case of long chains of `Unary`,
        // and when walking down the left-hand edges of `Binary`s; instead of pushing into the heap
        // and immediately popping it back again, we just stay purely in stack space.
        let mut backtrack = Vec::new();
        let mut cur = Some((self, other));
        while let Some((a, b)) = cur.take().or_else(|| backtrack.pop()) {
            // Short-circuit this entire tree branch if `a` and `b` point to the same object.
            if std::ptr::eq(a, b) {
                continue;
            }
            match (a, b) {
                (Self::Symbol(a_sym), Self::Symbol(b_sym)) => return_if_unequal!(a_sym, b_sym),
                (Self::Value(a_val), Self::Value(b_val)) => return_if_unequal!(a_val, b_val),
                (
                    Self::Unary {
                        op: a_op,
                        expr: a_expr,
                    },
                    Self::Unary {
                        op: b_op,
                        expr: b_expr,
                    },
                ) => {
                    return_if_unequal!(a_op, b_op);
                    cur = Some((a_expr, b_expr));
                }
                (
                    Self::Binary {
                        op: a_op,
                        lhs: a_lhs,
                        rhs: a_rhs,
                    },
                    Self::Binary {
                        op: b_op,
                        lhs: b_lhs,
                        rhs: b_rhs,
                    },
                ) => {
                    return_if_unequal!(a_op, b_op);
                    // This is the same `:std::ptr::eq` trick from the top of the loop, but we
                    // repeat it here to avoid allocations if possible.
                    if !Arc::ptr_eq(a_rhs, b_rhs) {
                        backtrack.push((a_rhs, b_rhs));
                    }
                    cur = Some((a_lhs, b_lhs));
                }
                // The `SymbolExpr` variants don't match. This doesn't use a full `_` so we still
                // get compiler protection against expansion of the enum.
                (Self::Symbol(_), _)
                | (Self::Value(_), _)
                | (Self::Unary { .. }, _)
                | (Self::Binary { .. }, _) => return false,
            }
        }
        true
    }

    /// evaluate the equation
    /// if recursive is false, only this node will be evaluated
    pub fn eval(&self, recurse: bool) -> Option<Value> {
        let process_unary = |op: UnaryOp, val: Value| -> Value {
            let ret = match op {
                UnaryOp::Abs => val.abs(),
                UnaryOp::Neg => -val,
                UnaryOp::Sin => val.sin(),
                UnaryOp::Asin => val.asin(),
                UnaryOp::Cos => val.cos(),
                UnaryOp::Acos => val.acos(),
                UnaryOp::Tan => val.tan(),
                UnaryOp::Atan => val.atan(),
                UnaryOp::Exp => val.exp(),
                UnaryOp::Log => val.log(),
                UnaryOp::Sign => val.sign(),
                UnaryOp::Conj => match val {
                    Value::Complex(v) => Value::Complex(v.conj()),
                    _ => val,
                },
            };
            match ret {
                Value::Real(_) => ret,
                Value::Int(_) => ret,
                Value::Complex(c) => {
                    if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
                        Value::Real(c.re)
                    } else {
                        ret
                    }
                }
            }
        };

        let process_binary = |op: BinaryOp, lval: Value, rval: Value| -> Value {
            let ret = match op {
                BinaryOp::Add => lval + rval,
                BinaryOp::Sub => lval - rval,
                BinaryOp::Mul => lval * rval,
                BinaryOp::Div => lval / rval,
                BinaryOp::Pow => lval.pow(&rval),
            };
            match ret {
                Value::Real(_) => ret,
                Value::Int(_) => ret,
                Value::Complex(c) => {
                    if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
                        Value::Real(c.re)
                    } else {
                        ret
                    }
                }
            }
        };

        #[derive(Clone, Copy)]
        enum ParentPosition {
            None,
            Lhs,
            Rhs,
        }

        let mut current: Option<(Arc<SymbolExpr>, ParentPosition)> = match self {
            SymbolExpr::Symbol(_) => return None,
            SymbolExpr::Value(e) => return Some(*e),
            SymbolExpr::Unary { op, expr } => {
                if !recurse {
                    return match expr.as_ref() {
                        SymbolExpr::Value(e) => Some(process_unary(op.clone(), *e)),
                        _ => None,
                    };
                }
                if let SymbolExpr::Symbol(_) = expr.as_ref() {
                    return None;
                } else if let SymbolExpr::Value(val) = expr.as_ref() {
                    return Some(process_unary(op.clone(), *val));
                }
                Some((Arc::new(self.clone()), ParentPosition::None))
            }
            SymbolExpr::Binary { op, lhs, rhs } => {
                if !recurse {
                    return match (lhs.as_ref(), rhs.as_ref()) {
                        (SymbolExpr::Value(lval), SymbolExpr::Value(rval)) => {
                            Some(process_binary(op.clone(), *lval, *rval))
                        }
                        _ => None,
                    };
                }
                match (lhs.as_ref(), rhs.as_ref()) {
                    (SymbolExpr::Symbol(_), _) | (_, SymbolExpr::Symbol(_)) => {
                        return None;
                    }
                    (SymbolExpr::Value(lval), SymbolExpr::Value(rval)) => {
                        return Some(process_binary(op.clone(), *lval, *rval));
                    }
                    _ => Some((Arc::new(self.clone()), ParentPosition::None)),
                }
            }
        };
        let process_value = |stack: &mut Vec<(Arc<SymbolExpr>, ParentPosition)>,
                             node_position: ParentPosition,
                             value: Value| {
            if let Some(parent) = stack.last_mut() {
                // make_mut is CoW and internally clones the symbol expr so we can mutate it
                let parent = Arc::<SymbolExpr>::make_mut(&mut parent.0);
                match parent {
                    SymbolExpr::Unary { op: _, expr } => {
                        *expr = Arc::new(SymbolExpr::Value(value));
                    }
                    SymbolExpr::Binary { op: _, lhs, rhs } => match node_position {
                        ParentPosition::None => unreachable!(
                            "Values with a binary parent always have a defined position"
                        ),
                        ParentPosition::Lhs => *lhs = Arc::new(SymbolExpr::Value(value)),
                        ParentPosition::Rhs => *rhs = Arc::new(SymbolExpr::Value(value)),
                    },
                    _ => {
                        unreachable!(
                            "Only operations need processing, Value and Symbol are already handled"
                        );
                    }
                };
                None
            } else {
                Some(value)
            }
        };

        let mut stack: Vec<(Arc<SymbolExpr>, ParentPosition)> = Vec::new();
        while current.is_some() || !stack.is_empty() {
            if let Some(ref curr) = current {
                match curr.0.as_ref() {
                    SymbolExpr::Symbol(_) => return None,
                    SymbolExpr::Value(_) => {
                        stack.push(curr.clone());
                        current = None;
                    }
                    SymbolExpr::Unary { op: _, expr } => {
                        stack.push(curr.clone());
                        current = Some((expr.clone(), ParentPosition::None));
                    }
                    SymbolExpr::Binary { op: _, lhs, rhs: _ } => {
                        stack.push(curr.clone());
                        current = Some((lhs.clone(), ParentPosition::Lhs));
                    }
                }
                continue;
            }
            let node = stack.pop().expect("Stack is checked to not be empty");
            match node.0.as_ref() {
                SymbolExpr::Symbol(_) => return None,
                SymbolExpr::Value(e) => {
                    if let Some(val) = process_value(&mut stack, node.1, *e) {
                        return Some(val);
                    }
                }
                SymbolExpr::Unary { op, expr } => match expr.as_ref() {
                    SymbolExpr::Symbol(_) => return None,
                    SymbolExpr::Value(val) => {
                        let val = process_unary(op.clone(), *val);
                        if let Some(val) = process_value(&mut stack, node.1, val) {
                            return Some(val);
                        }
                    }
                    SymbolExpr::Unary { op: _, expr } => {
                        current = Some((expr.clone(), ParentPosition::None));
                        stack.push(node.clone());
                    }
                    SymbolExpr::Binary { op: _, lhs: _, rhs } => {
                        current = Some((rhs.clone(), ParentPosition::Rhs));
                        stack.push(node.clone())
                    }
                },
                SymbolExpr::Binary { op, lhs, rhs } => match (lhs.as_ref(), rhs.as_ref()) {
                    (SymbolExpr::Value(lval), SymbolExpr::Value(rval)) => {
                        let val = process_binary(op.clone(), *lval, *rval);
                        if let Some(val) = process_value(&mut stack, node.1, val) {
                            return Some(val);
                        }
                    }
                    _ => {
                        if !matches!(rhs.as_ref(), SymbolExpr::Value(_)) {
                            current = Some((rhs.clone(), ParentPosition::Rhs));
                            stack.push(node.clone());
                        } else {
                            stack.push((rhs.clone(), ParentPosition::Rhs));
                            current = None;
                        }
                    }
                },
            }
        }
        None
    }

    /// calculate derivative of the equantion for a symbol passed by param
    pub fn derivative(&self, param: &Symbol) -> Result<SymbolExpr, String> {
        if let SymbolExpr::Symbol(s) = self
            && s.as_ref() == param
        {
            return Ok(SymbolExpr::Value(Value::Real(1.0)));
        }

        match self {
            SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => Ok(SymbolExpr::Value(Value::Real(0.0))),
            SymbolExpr::Unary { op, expr } => {
                let expr_d = expr.derivative(param)?;
                match op {
                    UnaryOp::Abs => Ok(&(expr.as_ref() * &expr_d)
                        / &SymbolExpr::Unary {
                            op: op.clone(),
                            expr: Arc::new(expr.as_ref().clone()),
                        }),
                    UnaryOp::Neg => Ok(SymbolExpr::Unary {
                        op: UnaryOp::Neg,
                        expr: Arc::new(expr_d),
                    }),
                    UnaryOp::Sin => {
                        let lhs = SymbolExpr::Unary {
                            op: UnaryOp::Cos,
                            expr: Arc::new(expr.as_ref().clone()),
                        };
                        Ok(lhs * expr_d)
                    }
                    UnaryOp::Asin => {
                        let d =
                            &SymbolExpr::Value(Value::Real(1.0)) - &(expr.as_ref() * expr.as_ref());
                        let rhs = match d {
                            SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
                            _ => _pow(d, SymbolExpr::Value(Value::Real(0.5))),
                        };
                        Ok(&expr_d / &rhs)
                    }
                    UnaryOp::Cos => {
                        let lhs = SymbolExpr::Unary {
                            op: UnaryOp::Sin,
                            expr: Arc::new(expr.as_ref().clone()),
                        };
                        Ok(&-&lhs * &expr_d)
                    }
                    UnaryOp::Acos => {
                        let d =
                            &SymbolExpr::Value(Value::Real(1.0)) - &(expr.as_ref() * expr.as_ref());
                        let rhs = match d {
                            SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
                            _ => _pow(d, SymbolExpr::Value(Value::Real(0.5))),
                        };
                        Ok(&-&expr_d / &rhs)
                    }
                    UnaryOp::Tan => {
                        let d = SymbolExpr::Unary {
                            op: UnaryOp::Cos,
                            expr: Arc::new(expr.as_ref().clone()),
                        };
                        Ok(&(&expr_d / &d) / &d)
                    }
                    UnaryOp::Atan => {
                        let d =
                            &SymbolExpr::Value(Value::Real(1.0)) + &(expr.as_ref() * expr.as_ref());
                        Ok(&expr_d / &d)
                    }
                    UnaryOp::Exp => Ok(&SymbolExpr::Unary {
                        op: UnaryOp::Exp,
                        expr: Arc::new(expr.as_ref().clone()),
                    } * &expr_d),
                    UnaryOp::Log => Ok(&expr_d / expr.as_ref()),
                    UnaryOp::Sign => {
                        Err("SymbolExpr::derivative does not support sign function.".to_string())
                    }
                    UnaryOp::Conj => {
                        // we assume real parameters, hence Conj acts as identity
                        Ok(expr_d)
                    }
                }
            }
            SymbolExpr::Binary { op, lhs, rhs } => match op {
                BinaryOp::Add => Ok(lhs.derivative(param)? + rhs.derivative(param)?),
                BinaryOp::Sub => Ok(lhs.derivative(param)? - rhs.derivative(param)?),
                BinaryOp::Mul => Ok(&(&lhs.derivative(param)? * rhs.as_ref())
                    + &(lhs.as_ref() * &rhs.derivative(param)?)),
                BinaryOp::Div => Ok(&(&(&(&lhs.derivative(param)? * rhs.as_ref())
                    - &(lhs.as_ref() * &rhs.derivative(param)?))
                    / rhs.as_ref())
                    / rhs.as_ref()),
                BinaryOp::Pow => {
                    if !lhs.has_symbol(param) {
                        if !rhs.has_symbol(param) {
                            Ok(SymbolExpr::Value(Value::Real(0.0)))
                        } else {
                            // (a^g(x))' = ln(a) * a^g(x) * g'(x)
                            Ok(_mul(
                                _mul(
                                    SymbolExpr::Binary {
                                        op: BinaryOp::Pow,
                                        lhs: Arc::new(lhs.as_ref().clone()),
                                        rhs: Arc::new(rhs.as_ref().clone()),
                                    },
                                    SymbolExpr::Unary {
                                        op: UnaryOp::Log,
                                        expr: Arc::new(lhs.as_ref().clone()),
                                    },
                                ),
                                rhs.derivative(param)?,
                            ))
                        }
                    } else if !rhs.has_symbol(param) {
                        // (g(x)^n)' = n*g(x)^(n-1) * g'(x)
                        Ok(_mul(
                            rhs.as_ref()
                                * &SymbolExpr::Binary {
                                    op: BinaryOp::Pow,
                                    lhs: Arc::new(lhs.as_ref().clone()),
                                    rhs: Arc::new(
                                        rhs.as_ref() - &SymbolExpr::Value(Value::Real(1.0)),
                                    ),
                                },
                            lhs.derivative(param)?,
                        ))
                    } else {
                        let new_expr = SymbolExpr::Unary {
                            op: UnaryOp::Exp,
                            expr: Arc::new(_mul(
                                SymbolExpr::Unary {
                                    op: UnaryOp::Log,
                                    expr: Arc::new(lhs.as_ref().clone()),
                                },
                                rhs.as_ref().clone(),
                            )),
                        };
                        new_expr.derivative(param)
                    }
                }
            },
        }
    }

    /// expand the equation
    pub fn expand(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(_) => self.clone(),
            SymbolExpr::Value(_) => self.clone(),
            SymbolExpr::Unary { op, expr } => {
                let ex = expr.expand();
                match op {
                    UnaryOp::Neg => match ex.neg_opt() {
                        Some(ne) => ne,
                        None => _neg(ex),
                    },
                    _ => SymbolExpr::Unary {
                        op: op.clone(),
                        expr: Arc::new(ex),
                    },
                }
            }
            SymbolExpr::Binary { op, lhs, rhs } => match op {
                BinaryOp::Mul => match lhs.mul_expand(rhs) {
                    Some(e) => e,
                    None => _mul(lhs.as_ref().clone(), rhs.as_ref().clone()),
                },
                BinaryOp::Div => match lhs.div_expand(rhs) {
                    Some(e) => e,
                    None => _div(lhs.as_ref().clone(), rhs.as_ref().clone()),
                },
                BinaryOp::Add => match lhs.add_opt(rhs, true) {
                    Some(e) => e,
                    None => _add(lhs.as_ref().clone(), rhs.as_ref().clone()),
                },
                BinaryOp::Sub => match lhs.sub_opt(rhs, true) {
                    Some(e) => e,
                    None => _sub(lhs.as_ref().clone(), rhs.as_ref().clone()),
                },
                BinaryOp::Pow => {
                    let base = lhs.expand();
                    let exponent = rhs.expand();
                    match &exponent {
                        SymbolExpr::Value(v) if v.is_minus_one() => base.rcp(),
                        SymbolExpr::Value(v) if v.is_negative() => {
                            let pos_exponent = SymbolExpr::Value(-*v);
                            base.pow(&pos_exponent).rcp()
                        }
                        _ => _pow(base, exponent),
                    }
                }
            },
        }
    }

    /// sign operator
    pub fn sign(&self) -> SymbolExpr {
        SymbolExpr::Unary {
            op: UnaryOp::Sign,
            expr: Arc::new(self.clone()),
        }
    }

    /// return real number if equation can be evaluated
    pub fn real(&self) -> Option<f64> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Some(r),
                Value::Int(r) => Some(r as f64),
                Value::Complex(c) => Some(c.re),
            },
            None => None,
        }
    }
    /// return imaginary part of the value if equation can be evaluated as complex number
    pub fn imag(&self) -> Option<f64> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(_) => Some(0.0),
                Value::Int(_) => Some(0.0),
                Value::Complex(c) => Some(c.im),
            },
            None => None,
        }
    }
    /// return complex number if equation can be evaluated as complex
    pub fn complex(&self) -> Option<Complex64> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Some(r.into()),
                Value::Int(i) => Some((i as f64).into()),
                Value::Complex(c) => Some(c),
            },
            None => None,
        }
    }

    /// Iterate over all symbols this equation contains.
    pub fn iter_symbols(&self) -> Box<dyn Iterator<Item = &Symbol> + '_> {
        // This could maybe be more elegantly resolved with a SymbolIter type>
        match self {
            SymbolExpr::Symbol(e) => Box::new(::std::iter::once(e.as_ref())),
            SymbolExpr::Value(_) => Box::new(::std::iter::empty()),
            SymbolExpr::Unary { op: _, expr } => expr.iter_symbols(),
            SymbolExpr::Binary { op: _, lhs, rhs } => {
                Box::new(lhs.iter_symbols().chain(rhs.iter_symbols()))
            }
        }
    }

    /// Map of parameter name to the parameter.
    pub fn name_map(&self) -> HashMap<String, Symbol> {
        self.iter_symbols()
            .map(|param| (param.repr(false), param.clone()))
            .collect()
    }

    /// return all numbers in the equation
    pub fn values(&self) -> Vec<Value> {
        match self {
            SymbolExpr::Symbol(_) => Vec::<Value>::new(),
            SymbolExpr::Value(v) => Vec::<Value>::from([*v]),
            SymbolExpr::Unary { op: _, expr } => expr.values(),
            SymbolExpr::Binary { op: _, lhs, rhs } => {
                let mut l = lhs.values();
                let r = rhs.values();
                l.extend(r);
                l
            }
        }
    }

    /// check if a symbol is in this equation
    pub fn has_symbol(&self, param: &Symbol) -> bool {
        match self {
            SymbolExpr::Symbol(e) => e.as_ref().eq(param),
            SymbolExpr::Value(_) => false,
            SymbolExpr::Unary { op: _, expr } => expr.has_symbol(param),
            SymbolExpr::Binary { op: _, lhs, rhs } => lhs.has_symbol(param) | rhs.has_symbol(param),
        }
    }

    /// return reciprocal of the equation
    pub fn rcp(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => _div(
                SymbolExpr::Value(Value::Real(1.0)),
                SymbolExpr::Symbol(e.clone()),
            ),
            SymbolExpr::Value(e) => SymbolExpr::Value(e.rcp()),
            SymbolExpr::Unary { .. } => _div(SymbolExpr::Value(Value::Real(1.0)), self.clone()),
            SymbolExpr::Binary { op, lhs, rhs } => match op {
                BinaryOp::Div => SymbolExpr::Binary {
                    op: op.clone(),
                    lhs: rhs.clone(),
                    rhs: lhs.clone(),
                },
                _ => _div(SymbolExpr::Value(Value::Real(1.0)), self.clone()),
            },
        }
    }
    /// return square root of the equation
    pub fn sqrt(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
            _ => self.pow(&SymbolExpr::Value(Value::Real(0.5))),
        }
    }

    /// return conjugate of the equation
    pub fn conjugate(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(_) => SymbolExpr::Unary {
                op: UnaryOp::Conj,
                expr: Arc::new(self.clone()),
            },
            SymbolExpr::Value(e) => match e {
                Value::Complex(c) => SymbolExpr::Value(Value::Complex(c.conj())),
                _ => SymbolExpr::Value(*e),
            },
            SymbolExpr::Unary { op, expr } => SymbolExpr::Unary {
                op: op.clone(),
                expr: Arc::new(expr.conjugate()),
            },
            SymbolExpr::Binary { op, lhs, rhs } => SymbolExpr::Binary {
                op: op.clone(),
                lhs: Arc::new(lhs.conjugate()),
                rhs: Arc::new(rhs.conjugate()),
            },
        }
    }

    /// check if complex number or not
    pub fn is_complex(&self) -> Option<bool> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Complex(c) => Some(!(-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im)),
                _ => Some(false),
            },
            None => None,
        }
    }

    /// check if real number or not
    pub fn is_real(&self) -> Option<bool> {
        self.eval(true).map(|value| value.is_real())
    }

    /// check if integer or not
    pub fn is_int(&self) -> Option<bool> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Int(_) => Some(true),
                _ => Some(false),
            },
            None => None,
        }
    }

    /// check if evaluated result is 0
    pub fn is_zero(&self) -> bool {
        match self.eval(true) {
            Some(v) => v.is_zero(),
            None => false,
        }
    }

    /// check if evaluated result is 1
    pub fn is_one(&self) -> bool {
        match self.eval(true) {
            Some(v) => v.is_one(),
            None => false,
        }
    }

    /// check if evaluated result is -1
    pub fn is_minus_one(&self) -> bool {
        match self.eval(true) {
            Some(v) => v.is_minus_one(),
            None => false,
        }
    }

    /// check if evaluated result is negative
    fn is_negative(&self) -> bool {
        match self {
            SymbolExpr::Value(v) => v.is_negative(),
            SymbolExpr::Symbol(_) => false,
            SymbolExpr::Unary { op, expr } => match op {
                UnaryOp::Abs => false,
                UnaryOp::Neg => !expr.is_negative(),
                _ => false, // TO DO add heuristic determination
            },
            SymbolExpr::Binary { op, lhs, rhs } => match op {
                BinaryOp::Mul | BinaryOp::Div => lhs.is_negative() ^ rhs.is_negative(),
                BinaryOp::Add | BinaryOp::Sub => lhs.is_negative(),
                _ => false, // TO DO add heuristic determination for pow
            },
        }
    }

    /// unary operations
    pub fn abs(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.abs()),
            SymbolExpr::Unary {
                op: UnaryOp::Abs | UnaryOp::Neg,
                expr,
            } => expr.abs(),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Abs,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn sin(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.sin()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Sin,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn asin(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.asin()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Asin,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn cos(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.cos()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Cos,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn acos(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.acos()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Acos,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn tan(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.tan()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Tan,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn atan(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.atan()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Atan,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn exp(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.exp()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Exp,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn log(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.log()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Log,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn pow(&self, rhs: &SymbolExpr) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => match rhs {
                SymbolExpr::Value(r) => SymbolExpr::Value(l.pow(r)),
                _ => SymbolExpr::Binary {
                    op: BinaryOp::Pow,
                    lhs: Arc::new(SymbolExpr::Value(*l)),
                    rhs: Arc::new(rhs.clone()),
                },
            },
            _ => SymbolExpr::Binary {
                op: BinaryOp::Pow,
                lhs: Arc::new(self.clone()),
                rhs: Arc::new(rhs.clone()),
            },
        }
    }

    pub fn string_id(&self) -> String {
        self.repr(true)
    }

    // Add with heuristic optimization
    fn add_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if self.is_zero() {
            Some(rhs.clone())
        } else if rhs.is_zero() {
            Some(self.clone())
        } else {
            // if neg operation, call sub_opt
            if let SymbolExpr::Unary { op, expr } = rhs {
                if let UnaryOp::Neg = op {
                    return self.sub_opt(expr, recursive);
                }
            } else if recursive
                && let SymbolExpr::Binary {
                    op,
                    lhs: r_lhs,
                    rhs: r_rhs,
                } = rhs
            {
                // recursive optimization for add and sub
                if let BinaryOp::Add = &op {
                    if let Some(e) = self.add_opt(r_lhs, true) {
                        return match e.add_opt(r_rhs, true) {
                            Some(ee) => Some(ee),
                            None => Some(_add(e, r_rhs.as_ref().clone())),
                        };
                    }
                    if let Some(e) = self.add_opt(r_rhs, true) {
                        return match e.add_opt(r_lhs, true) {
                            Some(ee) => Some(ee),
                            None => Some(_add(e, r_lhs.as_ref().clone())),
                        };
                    }
                }
                if let BinaryOp::Sub = &op {
                    if let Some(e) = self.add_opt(r_lhs, true) {
                        return match e.sub_opt(r_rhs, true) {
                            Some(ee) => Some(ee),
                            None => Some(_sub(e, r_rhs.as_ref().clone())),
                        };
                    }
                    if let Some(e) = self.sub_opt(r_rhs, true) {
                        return match e.add_opt(r_lhs, true) {
                            Some(ee) => Some(ee),
                            None => Some(_add(e, r_lhs.as_ref().clone())),
                        };
                    }
                }
            }

            // optimization for each node type
            match self {
                SymbolExpr::Value(l) => match rhs {
                    SymbolExpr::Value(r) => Some(SymbolExpr::Value(l + r)),
                    SymbolExpr::Binary {
                        op,
                        lhs: r_lhs,
                        rhs: r_rhs,
                    } => {
                        if let SymbolExpr::Value(v) = r_lhs.as_ref() {
                            let t = l + v;
                            match op {
                                BinaryOp::Add => {
                                    if t.is_zero() {
                                        Some(r_rhs.as_ref().clone())
                                    } else {
                                        Some(_add(SymbolExpr::Value(t), r_rhs.as_ref().clone()))
                                    }
                                }
                                BinaryOp::Sub => {
                                    if t.is_zero() {
                                        match r_rhs.neg_opt() {
                                            Some(e) => Some(e),
                                            None => Some(_neg(r_rhs.as_ref().clone())),
                                        }
                                    } else {
                                        Some(_sub(SymbolExpr::Value(t), r_rhs.as_ref().clone()))
                                    }
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                },
                SymbolExpr::Symbol(l) => match rhs {
                    SymbolExpr::Value(_) => Some(_add(rhs.clone(), self.clone())),
                    SymbolExpr::Symbol(r) => {
                        if r == l {
                            Some(_mul(SymbolExpr::Value(Value::Int(2)), self.clone()))
                        } else if r < l {
                            Some(_add(rhs.clone(), self.clone()))
                        } else {
                            None
                        }
                    }
                    SymbolExpr::Binary {
                        op,
                        lhs: r_lhs,
                        rhs: r_rhs,
                    } => {
                        if let (BinaryOp::Mul, SymbolExpr::Value(v), SymbolExpr::Symbol(s)) =
                            (op, r_lhs.as_ref(), r_rhs.as_ref())
                        {
                            if l == s {
                                let t = v + &Value::Int(1);
                                if t.is_zero() {
                                    Some(SymbolExpr::Value(Value::Int(0)))
                                } else {
                                    Some(_mul(SymbolExpr::Value(t), r_rhs.as_ref().clone()))
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                },
                SymbolExpr::Unary { op, expr } => {
                    if let UnaryOp::Neg = op {
                        if let Some(e) = expr.sub_opt(rhs, recursive) {
                            return match e.neg_opt() {
                                Some(ee) => Some(ee),
                                None => Some(_neg(e)),
                            };
                        }
                    } else if let SymbolExpr::Unary {
                        op: rop,
                        expr: rexpr,
                    } = rhs
                        && op == rop
                        && let Some(t) = expr.expand().add_opt(&rexpr.expand(), true)
                        && t.is_zero()
                    {
                        return Some(SymbolExpr::Value(Value::Int(0)));
                    }

                    // swap nodes by sorting rule
                    match rhs {
                        SymbolExpr::Binary { op: rop, .. } => {
                            if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
                                if self > rhs {
                                    Some(_add(rhs.clone(), self.clone()))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        _ => {
                            if self > rhs {
                                Some(_add(rhs.clone(), self.clone()))
                            } else {
                                None
                            }
                        }
                    }
                }
                SymbolExpr::Binary {
                    op,
                    lhs: l_lhs,
                    rhs: l_rhs,
                } => {
                    // Binary + Binary
                    if let SymbolExpr::Binary {
                        op: rop,
                        lhs: r_lhs,
                        rhs: r_rhs,
                    } = rhs
                    {
                        match (
                            l_lhs.as_ref(),
                            l_rhs.as_ref(),
                            r_lhs.as_ref(),
                            r_rhs.as_ref(),
                        ) {
                            // shared terms (lv op X) + (rv op X)
                            (SymbolExpr::Value(lv), _, SymbolExpr::Value(rv), _) => {
                                if l_rhs.expand().string_id() == r_rhs.expand().string_id() {
                                    let t = SymbolExpr::Value(lv + rv);
                                    match (op, rop) {
                                        // (lv + X) + (rv + X)  ->  t + 2*X
                                        // (lv - X) + (rv - X)  ->  t - 2*X
                                        (BinaryOp::Add, BinaryOp::Add)
                                        | (BinaryOp::Sub, BinaryOp::Sub) => {
                                            let two_signed =
                                                if matches!(op, BinaryOp::Add) { 2 } else { -2 };
                                            let doubled_signed = _mul(
                                                SymbolExpr::Value(Value::Int(two_signed)),
                                                l_rhs.as_ref().clone(),
                                            );
                                            return Some(if t.is_zero() {
                                                doubled_signed
                                            } else {
                                                _add(t, doubled_signed)
                                            });
                                        }
                                        // (lv + X) + (rv - X)  ->  X canceled -> (lv+rv) = t
                                        // (lv - X) + (rv + X)  ->  X canceled -> (lv+rv) = t
                                        (BinaryOp::Sub, BinaryOp::Add)
                                        | (BinaryOp::Add, BinaryOp::Sub) => {
                                            return Some(t);
                                        }
                                        // (lv * X) + (rv * X)  ->  (lv+rv) * X = t * X
                                        (BinaryOp::Mul, BinaryOp::Mul) => {
                                            if t.is_zero() {
                                                return Some(SymbolExpr::Value(Value::Int(0)));
                                            }
                                            return match t.mul_opt(l_rhs, recursive) {
                                                Some(e) => Some(e),
                                                None => Some(_mul(t, l_rhs.as_ref().clone())),
                                            };
                                        }
                                        // (lv / X) + (rv / X)  ->  (lv+rv) / X = t / X
                                        (BinaryOp::Div, BinaryOp::Div) => {
                                            if t.is_zero() {
                                                return Some(SymbolExpr::Value(Value::Int(0)));
                                            }
                                            return match t.div_opt(l_rhs, recursive) {
                                                Some(e) => Some(e),
                                                None => Some(_div(t, l_rhs.as_ref().clone())),
                                            };
                                        }
                                        (_, _) => (),
                                    }
                                }
                            }
                            // (X / lv) + (Y / rv) ->  (rv*X + lv*Y) / (lv*rv)
                            // only when X == +-Y, so add_opt collapses (rv*X + lv*Y) into a single term
                            (_, SymbolExpr::Value(lv), _, SymbolExpr::Value(rv)) => {
                                if let (BinaryOp::Div, BinaryOp::Div) = (op, rop)
                                    && (l_lhs.expand().string_id() == r_lhs.expand().string_id()
                                        || _neg(l_lhs.as_ref().clone()).expand().string_id()
                                            == r_lhs.expand().string_id())
                                {
                                    let tl = _mul(SymbolExpr::Value(*rv), l_lhs.as_ref().clone());
                                    let tr = _mul(SymbolExpr::Value(*lv), r_lhs.as_ref().clone());
                                    let b = SymbolExpr::Value(lv * rv);
                                    return match tl.add_opt(&tr, recursive) {
                                        Some(e) => Some(_div(e, b)),
                                        None => Some(_div(_add(tl, tr), b)),
                                    };
                                }
                            }
                            // (lv * X) + (Y / rv) ->  (lv * X) + ((1/rv) * Y)
                            // rewrite Div as Mul when X == +-Y so add_opt collapses them into a single term
                            (SymbolExpr::Value(_), _, _, SymbolExpr::Value(rv)) => {
                                if let (BinaryOp::Mul, BinaryOp::Div) = (op, rop)
                                    && (l_rhs.expand().string_id() == r_lhs.expand().string_id()
                                        || _neg(l_rhs.as_ref().clone()).expand().string_id()
                                            == r_lhs.expand().string_id())
                                {
                                    let r = _mul(
                                        SymbolExpr::Value(Value::Real(1.0) / *rv),
                                        r_lhs.as_ref().clone(),
                                    );
                                    if let Some(e) = self.add_opt(&r, recursive) {
                                        return Some(e);
                                    }
                                }
                            }
                            // (X / lv) + (rv * Y) -> ((1/lv) * X) + (rv * Y)
                            // rewrite Div as Mul when X == +-Y so add_opt collapses them into a single term
                            (_, SymbolExpr::Value(lv), SymbolExpr::Value(_), _) => {
                                if let (BinaryOp::Div, BinaryOp::Mul) = (op, rop)
                                    && (l_lhs.expand().string_id() == r_rhs.expand().string_id()
                                        || _neg(l_lhs.as_ref().clone()).expand().string_id()
                                            == r_rhs.expand().string_id())
                                {
                                    let l = _mul(
                                        SymbolExpr::Value(Value::Real(1.0) / *lv),
                                        l_lhs.as_ref().clone(),
                                    );
                                    if let Some(e) = l.add_opt(rhs, recursive) {
                                        return Some(e);
                                    }
                                }
                            }
                            (_, _, _, _) => (),
                        }

                        if op == rop
                            && let Some(e) = rhs.neg_opt()
                            && self.expand().string_id() == e.expand().string_id()
                        {
                            return Some(SymbolExpr::Value(Value::Int(0)));
                        }
                    } else if let SymbolExpr::Symbol(r) = rhs
                        && let (BinaryOp::Mul, SymbolExpr::Value(v), SymbolExpr::Symbol(s)) =
                            (op, l_lhs.as_ref(), l_rhs.as_ref())
                        && s == r
                    {
                        let t = v + &Value::Int(1);
                        if t.is_zero() {
                            return Some(SymbolExpr::Value(Value::Int(0)));
                        } else {
                            return Some(_mul(SymbolExpr::Value(t), l_rhs.as_ref().clone()));
                        }
                    } else if let SymbolExpr::Value(rv) = rhs {
                        if matches!(op, BinaryOp::Add) {
                            // implements l_lhs + l_rhs + rv
                            match (l_lhs.as_ref(), l_rhs.as_ref()) {
                                // case: (lv + rv) + l_rhs
                                (SymbolExpr::Value(lv), _) => {
                                    return Some(_add(
                                        SymbolExpr::Value(lv + rv),
                                        l_rhs.as_ref().clone(),
                                    ));
                                }
                                // case: l_lhs + (lv + rv)
                                (_, SymbolExpr::Value(lv)) => {
                                    return Some(_add(
                                        SymbolExpr::Value(lv + rv),
                                        l_lhs.as_ref().clone(),
                                    ));
                                }
                                (_, _) => (),
                            }
                        } else if matches!(op, BinaryOp::Sub) {
                            // implements l_lhs - l_rhs + rv
                            match (l_lhs.as_ref(), l_rhs.as_ref()) {
                                // case: (lv + rv) - l_rhs
                                (SymbolExpr::Value(lv), _) => {
                                    return Some(_sub(
                                        SymbolExpr::Value(lv + rv),
                                        l_rhs.as_ref().clone(),
                                    ));
                                }
                                // case: (rv - lv) + l_lhs
                                (_, SymbolExpr::Value(lv)) => {
                                    return Some(_add(
                                        SymbolExpr::Value(rv - lv),
                                        l_lhs.as_ref().clone(),
                                    ));
                                }
                                (_, _) => (),
                            }
                        }
                    }
                    if recursive {
                        if let BinaryOp::Add = op {
                            if let Some(e) = l_lhs.add_opt(rhs, true) {
                                return match e.add_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_add(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.add_opt(rhs, true) {
                                return match l_lhs.add_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_add(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        } else if let BinaryOp::Sub = op {
                            if let Some(e) = l_lhs.add_opt(rhs, true) {
                                return match e.sub_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_sub(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.sub_opt(rhs, true) {
                                return match l_lhs.sub_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_sub(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        }
                    }
                    // swap nodes by sorting rule
                    if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = op {
                        match rhs {
                            SymbolExpr::Binary { op: rop, .. } => {
                                if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
                                    if self > rhs {
                                        Some(_add(rhs.clone(), self.clone()))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => {
                                if self > rhs {
                                    Some(_add(rhs.clone(), self.clone()))
                                } else {
                                    None
                                }
                            }
                        }
                    } else {
                        None
                    }
                }
            }
        }
    }

    /// Sub with heuristic optimization
    fn sub_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if self.is_zero() {
            match rhs.neg_opt() {
                Some(e) => Some(e),
                None => Some(_neg(rhs.clone())),
            }
        } else if rhs.is_zero() {
            Some(self.clone())
        } else {
            // if neg, call add_opt
            if let SymbolExpr::Unary { op, expr } = rhs {
                if let UnaryOp::Neg = op {
                    return self.add_opt(expr, recursive);
                }
            } else if recursive
                && let SymbolExpr::Binary {
                    op,
                    lhs: r_lhs,
                    rhs: r_rhs,
                } = rhs
            {
                // recursive optimization for add and sub
                if let BinaryOp::Add = &op {
                    if let Some(e) = self.sub_opt(r_lhs, true) {
                        return match e.sub_opt(r_rhs, true) {
                            Some(ee) => Some(ee),
                            None => Some(_sub(e, r_rhs.as_ref().clone())),
                        };
                    }
                    if let Some(e) = self.sub_opt(r_rhs, true) {
                        return match e.sub_opt(r_lhs, true) {
                            Some(ee) => Some(ee),
                            None => Some(_sub(e, r_lhs.as_ref().clone())),
                        };
                    }
                }
                if let BinaryOp::Sub = &op {
                    if let Some(e) = self.sub_opt(r_lhs, true) {
                        return match e.add_opt(r_rhs, true) {
                            Some(ee) => Some(ee),
                            None => Some(_add(e, r_rhs.as_ref().clone())),
                        };
                    }
                    if let Some(e) = self.add_opt(r_rhs, true) {
                        return match e.sub_opt(r_lhs, true) {
                            Some(ee) => Some(ee),
                            None => Some(_sub(e, r_lhs.as_ref().clone())),
                        };
                    }
                }
            }

            // optimization for each type
            match self {
                SymbolExpr::Value(l) => match &rhs {
                    SymbolExpr::Value(r) => Some(SymbolExpr::Value(l - r)),
                    SymbolExpr::Binary {
                        op,
                        lhs: r_lhs,
                        rhs: r_rhs,
                    } => {
                        if let SymbolExpr::Value(v) = r_lhs.as_ref() {
                            let t = l - v;
                            match op {
                                BinaryOp::Add => {
                                    if t.is_zero() {
                                        match r_rhs.neg_opt() {
                                            Some(e) => Some(e),
                                            None => Some(_neg(r_rhs.as_ref().clone())),
                                        }
                                    } else {
                                        Some(_sub(SymbolExpr::Value(t), r_rhs.as_ref().clone()))
                                    }
                                }
                                BinaryOp::Sub => {
                                    if t.is_zero() {
                                        Some(r_rhs.as_ref().clone())
                                    } else {
                                        Some(_add(SymbolExpr::Value(t), r_rhs.as_ref().clone()))
                                    }
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                },
                SymbolExpr::Symbol(l) => match &rhs {
                    SymbolExpr::Value(r) => Some(_add(SymbolExpr::Value(-r), self.clone())),
                    SymbolExpr::Symbol(r) => {
                        if r == l {
                            Some(SymbolExpr::Value(Value::Int(0)))
                        } else if r < l {
                            Some(_add(_neg(rhs.clone()), self.clone()))
                        } else {
                            None
                        }
                    }
                    SymbolExpr::Binary {
                        op,
                        lhs: r_lhs,
                        rhs: r_rhs,
                    } => {
                        if let (BinaryOp::Mul, SymbolExpr::Value(v), SymbolExpr::Symbol(s)) =
                            (op, r_lhs.as_ref(), r_rhs.as_ref())
                        {
                            if l == s {
                                let t = &Value::Int(1) - v;
                                if t.is_zero() {
                                    Some(SymbolExpr::Value(Value::Int(0)))
                                } else {
                                    Some(_mul(SymbolExpr::Value(t), r_rhs.as_ref().clone()))
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                },
                SymbolExpr::Unary { op, expr } => {
                    if let UnaryOp::Neg = op
                        && let Some(e) = expr.add_opt(rhs, recursive)
                    {
                        return match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        };
                    }
                    if let SymbolExpr::Unary {
                        op: rop,
                        expr: rexpr,
                    } = rhs
                        && op == rop
                        && let Some(t) = expr.expand().sub_opt(&rexpr.expand(), true)
                        && t.is_zero()
                    {
                        return Some(SymbolExpr::Value(Value::Int(0)));
                    }

                    // swap nodes by sorting rule
                    match rhs {
                        SymbolExpr::Binary { op: rop, .. } => {
                            if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
                                if self > rhs {
                                    match rhs.neg_opt() {
                                        Some(e) => Some(_add(e, self.clone())),
                                        None => Some(_add(_neg(rhs.clone()), self.clone())),
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        _ => {
                            if self > rhs {
                                match rhs.neg_opt() {
                                    Some(e) => Some(_add(e, self.clone())),
                                    None => Some(_add(_neg(rhs.clone()), self.clone())),
                                }
                            } else {
                                None
                            }
                        }
                    }
                }
                // Binary - Binary
                SymbolExpr::Binary {
                    op,
                    lhs: l_lhs,
                    rhs: l_rhs,
                } => {
                    if let SymbolExpr::Binary {
                        op: rop,
                        lhs: r_lhs,
                        rhs: r_rhs,
                    } = rhs
                    {
                        match (
                            l_lhs.as_ref(),
                            l_rhs.as_ref(),
                            r_lhs.as_ref(),
                            r_rhs.as_ref(),
                        ) {
                            // shared terms: (lv op X) - (rv op X)
                            (SymbolExpr::Value(lv), _, SymbolExpr::Value(rv), _) => {
                                if l_rhs.expand().string_id() == r_rhs.expand().string_id() {
                                    let t = SymbolExpr::Value(lv - rv);
                                    match (op, rop) {
                                        // (lv + X) - (rv + X)  ->  X canceled -> (lv-rv) = t
                                        // (lv - X) - (rv - X)  ->  X canceled -> (lv-rv) = t
                                        (BinaryOp::Add, BinaryOp::Add)
                                        | (BinaryOp::Sub, BinaryOp::Sub) => {
                                            return Some(t);
                                        }
                                        // (lv - X) - (rv + X)  ->  (lv-rv) - 2*X = t - 2*X
                                        // (lv + X) - (rv - X)  ->  (lv-rv) + 2*X = t + 2*X
                                        (BinaryOp::Sub, BinaryOp::Add)
                                        | (BinaryOp::Add, BinaryOp::Sub) => {
                                            let two_signed =
                                                if matches!(op, BinaryOp::Sub) { -2 } else { 2 };
                                            let doubled_signed = _mul(
                                                SymbolExpr::Value(Value::Int(two_signed)),
                                                l_rhs.as_ref().clone(),
                                            );
                                            return Some(if t.is_zero() {
                                                doubled_signed
                                            } else {
                                                _add(t, doubled_signed)
                                            });
                                        }
                                        // (lv * X) - (rv * X)  ->  (lv-rv) * X = t * X
                                        (BinaryOp::Mul, BinaryOp::Mul) => {
                                            if t.is_zero() {
                                                return Some(SymbolExpr::Value(Value::Int(0)));
                                            }
                                            return match t.mul_opt(l_rhs, recursive) {
                                                Some(e) => Some(e),
                                                None => Some(_mul(t, l_rhs.as_ref().clone())),
                                            };
                                        }
                                        // (lv / X) - (rv / X)  ->  (lv-rv) / X = t / X
                                        (BinaryOp::Div, BinaryOp::Div) => {
                                            if t.is_zero() {
                                                return Some(SymbolExpr::Value(Value::Int(0)));
                                            }
                                            return match t.div_opt(l_rhs, recursive) {
                                                Some(e) => Some(e),
                                                None => Some(_div(t, l_rhs.as_ref().clone())),
                                            };
                                        }
                                        (_, _) => (),
                                    }
                                }
                            }
                            // (X / lv) - (Y / rv) -> (rv*X - lv*Y) / (lv*rv)
                            // only when X == +-Y, so sub_opt collapses rv*X - lv*Y into a single term
                            (_, SymbolExpr::Value(lv), _, SymbolExpr::Value(rv)) => {
                                if let (BinaryOp::Div, BinaryOp::Div) = (op, rop)
                                    && (l_lhs.expand().string_id() == r_lhs.expand().string_id()
                                        || _neg(l_lhs.as_ref().clone()).expand().string_id()
                                            == r_lhs.expand().string_id())
                                {
                                    let tl = _mul(SymbolExpr::Value(*rv), l_lhs.as_ref().clone());
                                    let tr = _mul(SymbolExpr::Value(*lv), r_lhs.as_ref().clone());
                                    let b = SymbolExpr::Value(lv * rv);
                                    return match tl.sub_opt(&tr, recursive) {
                                        Some(e) => Some(_div(e, b)),
                                        None => Some(_div(_sub(tl, tr), b)),
                                    };
                                }
                            }
                            // (lv * X) - (Y / rv) ->  (lv * X) - ((1/rv) * Y)
                            // rewrite the Div as a Mul only when X == +-Y so sub_opt collapses them
                            (SymbolExpr::Value(_), _, _, SymbolExpr::Value(rv)) => {
                                if let (BinaryOp::Mul, BinaryOp::Div) = (op, rop)
                                    && (l_rhs.expand().string_id() == r_lhs.expand().string_id()
                                        || _neg(l_rhs.as_ref().clone()).expand().string_id()
                                            == r_lhs.expand().string_id())
                                {
                                    let r = _mul(
                                        SymbolExpr::Value(Value::Real(1.0) / *rv),
                                        r_lhs.as_ref().clone(),
                                    );
                                    if let Some(e) = self.sub_opt(&r, recursive) {
                                        return Some(e);
                                    }
                                }
                            }
                            // (X / lv) - (rv * Y) -> ((1/lv) * X) - (rv * Y)
                            // rewrite the Div as a Mul only when X == +-Y so sub_opt collapses them
                            (_, SymbolExpr::Value(lv), SymbolExpr::Value(_), _) => {
                                if let (BinaryOp::Div, BinaryOp::Mul) = (op, rop)
                                    && (l_lhs.expand().string_id() == r_rhs.expand().string_id()
                                        || _neg(l_lhs.as_ref().clone()).expand().string_id()
                                            == r_rhs.expand().string_id())
                                {
                                    let l = _mul(
                                        SymbolExpr::Value(Value::Real(1.0) / *lv),
                                        l_lhs.as_ref().clone(),
                                    );
                                    if let Some(e) = l.sub_opt(rhs, recursive) {
                                        return Some(e);
                                    }
                                }
                            }
                            (_, _, _, _) => (),
                        }

                        if op == rop && self.expand().string_id() == rhs.expand().string_id() {
                            return Some(SymbolExpr::Value(Value::Int(0)));
                        }
                    } else if let SymbolExpr::Symbol(r) = rhs
                        && let (BinaryOp::Mul, SymbolExpr::Value(v), SymbolExpr::Symbol(s)) =
                            (op, l_lhs.as_ref(), l_rhs.as_ref())
                        && s == r
                    {
                        let t = v - &Value::Int(1);
                        if t.is_zero() {
                            return Some(SymbolExpr::Value(Value::Int(0)));
                        } else {
                            return Some(_mul(SymbolExpr::Value(t), l_rhs.as_ref().clone()));
                        }
                    } else if let SymbolExpr::Value(rv) = rhs {
                        if matches!(op, BinaryOp::Add) {
                            // implements l_lhs + l_rhs - rv
                            match (l_lhs.as_ref(), l_rhs.as_ref()) {
                                // case: (lv - rv) + l_rhs
                                (SymbolExpr::Value(lv), _) => {
                                    return Some(_add(
                                        SymbolExpr::Value(lv - rv),
                                        l_rhs.as_ref().clone(),
                                    ));
                                }
                                // case: l_lhs + (lv - rv)
                                (_, SymbolExpr::Value(lv)) => {
                                    return Some(_add(
                                        SymbolExpr::Value(lv - rv),
                                        l_lhs.as_ref().clone(),
                                    ));
                                }
                                (_, _) => (),
                            }
                        } else if matches!(op, BinaryOp::Sub) {
                            // implements l_lhs - l_rhs - rv
                            match (l_lhs.as_ref(), l_rhs.as_ref()) {
                                // case: (lv - rv) - l_rhs
                                (SymbolExpr::Value(lv), _) => {
                                    return Some(_sub(
                                        SymbolExpr::Value(lv - rv),
                                        l_rhs.as_ref().clone(),
                                    ));
                                }
                                // case: l_lhs - (lv + rv)
                                (_, SymbolExpr::Value(lv)) => {
                                    return Some(_sub(
                                        l_lhs.as_ref().clone(),
                                        SymbolExpr::Value(rv + lv),
                                    ));
                                }
                                (_, _) => (),
                            }
                        }
                    }
                    if recursive {
                        if let BinaryOp::Add = op {
                            if let Some(e) = l_lhs.sub_opt(rhs, true) {
                                return match e.add_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_add(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.sub_opt(rhs, true) {
                                return match l_lhs.add_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_add(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        }
                        if let BinaryOp::Sub = op {
                            if let Some(e) = l_lhs.sub_opt(rhs, true) {
                                return match e.sub_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_sub(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.add_opt(rhs, true) {
                                return match l_lhs.sub_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_sub(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        }
                    }
                    // swap nodes by sorting rule
                    if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = op {
                        match rhs {
                            SymbolExpr::Binary { op: rop, .. } => {
                                if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
                                    if self > rhs {
                                        match rhs.neg_opt() {
                                            Some(e) => Some(_add(e, self.clone())),
                                            None => Some(_add(_neg(rhs.clone()), self.clone())),
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => {
                                if self > rhs {
                                    match rhs.neg_opt() {
                                        Some(e) => Some(_add(e, self.clone())),
                                        None => Some(_add(_neg(rhs.clone()), self.clone())),
                                    }
                                } else {
                                    None
                                }
                            }
                        }
                    } else {
                        None
                    }
                }
            }
        }
    }

    /// Mul with heuristic optimization
    fn mul_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if self.is_zero() {
            Some(self.clone())
        } else if rhs.is_zero() || self.is_one() {
            Some(rhs.clone())
        } else if rhs.is_one() {
            Some(self.clone())
        } else if self.is_minus_one() {
            match rhs.neg_opt() {
                Some(e) => Some(e),
                None => Some(_neg(rhs.clone())),
            }
        } else if rhs.is_minus_one() {
            match self.neg_opt() {
                Some(e) => Some(e),
                None => Some(_neg(self.clone())),
            }
        } else {
            if let SymbolExpr::Value(_) | SymbolExpr::Symbol(_) = rhs
                && let SymbolExpr::Unary { .. } = self
            {
                return match rhs.mul_opt(self, recursive) {
                    Some(e) => Some(e),
                    None => Some(_mul(rhs.clone(), self.clone())),
                };
            }

            match self {
                SymbolExpr::Value(e) => e.mul_opt(rhs, recursive),
                SymbolExpr::Symbol(e) => match rhs {
                    SymbolExpr::Value(_) => Some(_mul(rhs.clone(), self.clone())),
                    SymbolExpr::Symbol(r) => {
                        if r < e {
                            Some(_mul(rhs.clone(), self.clone()))
                        } else {
                            None
                        }
                    }
                    SymbolExpr::Unary {
                        op: UnaryOp::Neg,
                        expr,
                    } => match expr.as_ref() {
                        SymbolExpr::Value(v) => Some(_mul(SymbolExpr::Value(-v), self.clone())),
                        SymbolExpr::Symbol(s) => {
                            if s < e {
                                Some(_neg(_mul(expr.as_ref().clone(), self.clone())))
                            } else {
                                Some(_neg(_mul(self.clone(), expr.as_ref().clone())))
                            }
                        }
                        SymbolExpr::Binary { .. } => match self.mul_opt(expr, recursive) {
                            Some(e) => match e.neg_opt() {
                                Some(ee) => Some(ee),
                                None => Some(_neg(e)),
                            },
                            None => None,
                        },
                        _ => None,
                    },
                    _ => None,
                },
                SymbolExpr::Unary { op, expr } => match op {
                    UnaryOp::Neg => match expr.mul_opt(rhs, recursive) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => None,
                    },
                    UnaryOp::Abs => match rhs {
                        SymbolExpr::Unary {
                            op: UnaryOp::Abs,
                            expr: rexpr,
                        } => match expr.mul_opt(rexpr, recursive) {
                            Some(e) => Some(SymbolExpr::Unary {
                                op: UnaryOp::Abs,
                                expr: Arc::new(e),
                            }),
                            None => Some(SymbolExpr::Unary {
                                op: UnaryOp::Abs,
                                expr: Arc::new(_mul(expr.as_ref().clone(), rexpr.as_ref().clone())),
                            }),
                        },
                        _ => None,
                    },
                    _ => None,
                },
                SymbolExpr::Binary {
                    op,
                    lhs: l_lhs,
                    rhs: l_rhs,
                } => {
                    if recursive {
                        if let SymbolExpr::Binary {
                            op: rop,
                            lhs: r_lhs,
                            rhs: r_rhs,
                        } = rhs
                        {
                            if let BinaryOp::Mul = &rop {
                                if let Some(e) = self.mul_opt(r_lhs, true) {
                                    return match e.mul_opt(r_rhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_mul(e, r_rhs.as_ref().clone())),
                                    };
                                }
                                if let Some(e) = self.mul_opt(r_rhs, true) {
                                    return match e.mul_opt(r_lhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_mul(e, r_lhs.as_ref().clone())),
                                    };
                                }
                            }
                            if let BinaryOp::Div = &rop {
                                if let Some(e) = self.mul_opt(r_lhs, true) {
                                    return match e.div_opt(r_rhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_div(e, r_rhs.as_ref().clone())),
                                    };
                                }
                                if let Some(e) = self.div_opt(r_rhs, true) {
                                    return match e.mul_opt(r_lhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_mul(e, r_lhs.as_ref().clone())),
                                    };
                                }
                            }
                        }

                        if let BinaryOp::Mul = &op {
                            if let Some(e) = l_lhs.mul_opt(rhs, true) {
                                return match e.mul_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.mul_opt(rhs, true) {
                                return match l_lhs.mul_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        } else if let BinaryOp::Div = &op {
                            if let Some(e) = l_lhs.mul_opt(rhs, true) {
                                return match e.div_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_div(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = rhs.div_opt(l_rhs, true) {
                                return match l_lhs.mul_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        }
                        None
                    } else {
                        match rhs {
                            SymbolExpr::Value(v) => match l_lhs.as_ref() {
                                SymbolExpr::Value(lv) => match op {
                                    BinaryOp::Mul => Some(_mul(
                                        SymbolExpr::Value(lv * v),
                                        l_rhs.as_ref().clone(),
                                    )),
                                    BinaryOp::Div => Some(_div(
                                        SymbolExpr::Value(lv * v),
                                        l_rhs.as_ref().clone(),
                                    )),
                                    _ => None,
                                },
                                _ => match l_rhs.as_ref() {
                                    SymbolExpr::Value(rv) => match op {
                                        BinaryOp::Mul => Some(_mul(
                                            SymbolExpr::Value(rv * v),
                                            l_lhs.as_ref().clone(),
                                        )),
                                        BinaryOp::Div => Some(_mul(
                                            SymbolExpr::Value(v / rv),
                                            l_lhs.as_ref().clone(),
                                        )),
                                        _ => None,
                                    },
                                    _ => None,
                                },
                            },
                            SymbolExpr::Binary {
                                op: rop,
                                lhs: r_lhs,
                                rhs: r_rhs,
                            } => match (op, rop) {
                                (BinaryOp::Mul, BinaryOp::Mul) => match (
                                    l_lhs.as_ref(),
                                    l_rhs.as_ref(),
                                    r_lhs.as_ref(),
                                    r_rhs.as_ref(),
                                ) {
                                    (SymbolExpr::Value(lv), _, SymbolExpr::Value(rv), _) => {
                                        Some(_mul(
                                            SymbolExpr::Value(lv * rv),
                                            _mul(l_rhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    }
                                    (SymbolExpr::Value(lv), _, _, SymbolExpr::Value(rv)) => {
                                        Some(_mul(
                                            SymbolExpr::Value(lv * rv),
                                            _mul(l_rhs.as_ref().clone(), r_lhs.as_ref().clone()),
                                        ))
                                    }
                                    (_, SymbolExpr::Value(lv), SymbolExpr::Value(rv), _) => {
                                        Some(_mul(
                                            SymbolExpr::Value(lv * rv),
                                            _mul(l_lhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    }
                                    (_, SymbolExpr::Value(lv), _, SymbolExpr::Value(rv)) => {
                                        Some(_mul(
                                            SymbolExpr::Value(lv * rv),
                                            _mul(l_lhs.as_ref().clone(), r_lhs.as_ref().clone()),
                                        ))
                                    }
                                    (_, _, _, _) => None,
                                },
                                (BinaryOp::Mul, BinaryOp::Div) => match (
                                    l_lhs.as_ref(),
                                    l_rhs.as_ref(),
                                    r_lhs.as_ref(),
                                    r_rhs.as_ref(),
                                ) {
                                    (SymbolExpr::Value(lv), _, SymbolExpr::Value(rv), _) => {
                                        Some(_mul(
                                            SymbolExpr::Value(lv * rv),
                                            _div(l_rhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    }
                                    (SymbolExpr::Value(lv), _, _, SymbolExpr::Value(rv)) => {
                                        Some(_mul(
                                            SymbolExpr::Value(lv / rv),
                                            _mul(l_rhs.as_ref().clone(), r_lhs.as_ref().clone()),
                                        ))
                                    }
                                    (_, SymbolExpr::Value(lv), SymbolExpr::Value(rv), _) => {
                                        Some(_mul(
                                            SymbolExpr::Value(lv * rv),
                                            _div(l_lhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    }
                                    (_, SymbolExpr::Value(lv), _, SymbolExpr::Value(rv)) => {
                                        Some(_mul(
                                            SymbolExpr::Value(lv / rv),
                                            _mul(l_lhs.as_ref().clone(), r_lhs.as_ref().clone()),
                                        ))
                                    }
                                    (_, _, _, _) => None,
                                },
                                (BinaryOp::Div, BinaryOp::Mul) => match (
                                    l_lhs.as_ref(),
                                    l_rhs.as_ref(),
                                    r_lhs.as_ref(),
                                    r_rhs.as_ref(),
                                ) {
                                    (SymbolExpr::Value(lv), _, SymbolExpr::Value(rv), _) => {
                                        Some(_mul(
                                            SymbolExpr::Value(lv * rv),
                                            _div(r_rhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                        ))
                                    }
                                    (SymbolExpr::Value(lv), _, _, SymbolExpr::Value(rv)) => {
                                        Some(_mul(
                                            SymbolExpr::Value(lv * rv),
                                            _div(r_lhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                        ))
                                    }
                                    (_, SymbolExpr::Value(lv), SymbolExpr::Value(rv), _) => {
                                        Some(_mul(
                                            SymbolExpr::Value(rv / lv),
                                            _mul(l_lhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    }
                                    (_, SymbolExpr::Value(lv), _, SymbolExpr::Value(rv)) => {
                                        Some(_mul(
                                            SymbolExpr::Value(rv / lv),
                                            _mul(l_lhs.as_ref().clone(), r_lhs.as_ref().clone()),
                                        ))
                                    }
                                    (_, _, _, _) => None,
                                },
                                (BinaryOp::Div, BinaryOp::Div) => match (
                                    l_lhs.as_ref(),
                                    l_rhs.as_ref(),
                                    r_lhs.as_ref(),
                                    r_rhs.as_ref(),
                                ) {
                                    (SymbolExpr::Value(lv), _, SymbolExpr::Value(rv), _) => {
                                        Some(_div(
                                            SymbolExpr::Value(lv * rv),
                                            _mul(l_rhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    }
                                    (SymbolExpr::Value(lv), _, _, SymbolExpr::Value(rv)) => {
                                        Some(_mul(
                                            SymbolExpr::Value(lv / rv),
                                            _div(r_lhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                        ))
                                    }
                                    (_, SymbolExpr::Value(lv), SymbolExpr::Value(rv), _) => {
                                        Some(_mul(
                                            SymbolExpr::Value(rv / lv),
                                            _div(l_lhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    }
                                    (_, SymbolExpr::Value(lv), _, SymbolExpr::Value(rv)) => {
                                        Some(_div(
                                            _mul(l_lhs.as_ref().clone(), r_lhs.as_ref().clone()),
                                            SymbolExpr::Value(lv * rv),
                                        ))
                                    }
                                    (_, _, _, _) => None,
                                },
                                (_, _) => None,
                            },
                            _ => None,
                        }
                    }
                }
            }
        }
    }
    /// expand with optimization for mul operation
    fn mul_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let SymbolExpr::Binary {
            op: rop,
            lhs: r_lhs,
            rhs: r_rhs,
        } = rhs
        {
            if let BinaryOp::Add | BinaryOp::Sub = &rop {
                let el = match self.mul_expand(r_lhs) {
                    Some(e) => e,
                    None => match self.mul_opt(r_lhs, true) {
                        Some(e) => e,
                        None => _mul(self.clone(), r_lhs.as_ref().clone()),
                    },
                };
                let er = match self.mul_expand(r_rhs) {
                    Some(e) => e,
                    None => match self.mul_opt(r_rhs, true) {
                        Some(e) => e,
                        None => _mul(self.clone(), r_rhs.as_ref().clone()),
                    },
                };
                return match &rop {
                    BinaryOp::Sub => match el.sub_opt(&er, true) {
                        Some(e) => Some(e),
                        None => Some(_sub(el, er)),
                    },
                    _ => match el.add_opt(&er, true) {
                        Some(e) => Some(e),
                        None => Some(_add(el, er)),
                    },
                };
            }
            if let BinaryOp::Mul = &rop {
                return match self.mul_expand(r_lhs) {
                    Some(e) => match e.mul_expand(r_rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, r_rhs.as_ref().clone())),
                    },
                    None => self
                        .mul_expand(r_rhs)
                        .map(|e| _mul(e, r_lhs.as_ref().clone())),
                };
            }
            if let BinaryOp::Div = &rop {
                return match self.mul_expand(r_lhs) {
                    Some(e) => match e.div_expand(r_rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_div(e, r_rhs.as_ref().clone())),
                    },
                    None => self
                        .div_expand(r_rhs)
                        .map(|e| _mul(e, r_lhs.as_ref().clone())),
                };
            }
        }
        if let SymbolExpr::Unary {
            op: UnaryOp::Neg,
            expr: rexpr,
        } = rhs
        {
            return match self.mul_expand(rexpr) {
                Some(e) => match e.neg_opt() {
                    Some(ee) => Some(ee),
                    None => Some(_neg(e)),
                },
                None => match self.mul_opt(rexpr, true) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => Some(_neg(_mul(self.clone(), rexpr.as_ref().clone()))),
                },
            };
        }

        match self {
            SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr,
            } => match expr.mul_expand(rhs) {
                Some(e) => match e.neg_opt() {
                    Some(ee) => Some(ee),
                    None => Some(_neg(e)),
                },
                None => match expr.mul_opt(rhs, true) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => None,
                },
            },
            SymbolExpr::Binary {
                op,
                lhs: l_lhs,
                rhs: l_rhs,
            } => match &op {
                BinaryOp::Add | BinaryOp::Sub => {
                    let l = match l_lhs.mul_expand(rhs) {
                        Some(e) => e,
                        None => match l_lhs.mul_opt(rhs, true) {
                            Some(e) => e,
                            None => _mul(l_lhs.as_ref().clone(), rhs.clone()),
                        },
                    };
                    let r = match l_rhs.mul_expand(rhs) {
                        Some(e) => e,
                        None => match l_rhs.mul_opt(rhs, true) {
                            Some(e) => e,
                            None => _mul(l_rhs.as_ref().clone(), rhs.clone()),
                        },
                    };
                    match &op {
                        BinaryOp::Sub => match l.sub_opt(&r, true) {
                            Some(e) => Some(e),
                            None => Some(_sub(l, r)),
                        },
                        _ => match l.add_opt(&r, true) {
                            Some(e) => Some(e),
                            None => Some(_add(l, r)),
                        },
                    }
                }
                BinaryOp::Mul => match l_lhs.mul_expand(rhs) {
                    Some(e) => match e.mul_expand(l_rhs) {
                        Some(ee) => Some(ee),
                        None => match e.mul_opt(l_rhs, true) {
                            Some(ee) => Some(ee),
                            None => Some(_mul(e, l_rhs.as_ref().clone())),
                        },
                    },
                    None => match l_rhs.mul_expand(rhs) {
                        Some(e) => match l_lhs.mul_expand(&e) {
                            Some(ee) => Some(ee),
                            None => match l_lhs.mul_opt(&e, true) {
                                Some(ee) => Some(ee),
                                None => Some(_mul(l_lhs.as_ref().clone(), e)),
                            },
                        },
                        None => None,
                    },
                },
                BinaryOp::Div => match l_lhs.mul_expand(rhs) {
                    Some(e) => match e.div_expand(l_rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_div(e, l_rhs.as_ref().clone())),
                    },
                    None => l_rhs
                        .div_expand(rhs)
                        .map(|e| _div(l_lhs.as_ref().clone(), e)),
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// Div with heuristic optimization
    fn div_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if rhs.is_zero() {
            // return inf to detect divide by zero without panic
            Some(SymbolExpr::Value(Value::Real(f64::INFINITY)))
        } else if rhs.is_one() {
            Some(self.clone())
        } else if rhs.is_minus_one() {
            match self.neg_opt() {
                Some(e) => Some(e),
                None => Some(_neg(self.clone())),
            }
        } else if *self == *rhs {
            let l_is_int = self.is_int().unwrap_or_default();
            let r_is_int = rhs.is_int().unwrap_or_default();
            if l_is_int || r_is_int {
                Some(SymbolExpr::Value(Value::Int(1)))
            } else {
                Some(SymbolExpr::Value(Value::Real(1.0)))
            }
        } else {
            if let SymbolExpr::Value(Value::Real(r)) = rhs {
                let t = 1.0 / r;
                if &(1.0 / t) == r {
                    if recursive {
                        return self.mul_opt(&SymbolExpr::Value(Value::Real(t)), recursive);
                    } else {
                        return Some(&SymbolExpr::Value(Value::Real(t)) * self);
                    }
                }
            }

            match self {
                SymbolExpr::Value(e) => e.div_opt(rhs, recursive),
                SymbolExpr::Symbol(_) => None,
                SymbolExpr::Unary { op, expr } => match op {
                    UnaryOp::Neg => match expr.div_opt(rhs, recursive) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => None,
                    },
                    UnaryOp::Abs => match rhs {
                        SymbolExpr::Unary {
                            op: UnaryOp::Abs,
                            expr: rexpr,
                        } => match expr.div_opt(rexpr, recursive) {
                            Some(e) => Some(SymbolExpr::Unary {
                                op: UnaryOp::Abs,
                                expr: Arc::new(e),
                            }),
                            None => Some(SymbolExpr::Unary {
                                op: UnaryOp::Abs,
                                expr: Arc::new(_div(expr.as_ref().clone(), rexpr.as_ref().clone())),
                            }),
                        },
                        _ => None,
                    },
                    _ => None,
                },
                SymbolExpr::Binary {
                    op,
                    lhs: l_lhs,
                    rhs: l_rhs,
                } => {
                    if recursive {
                        if let SymbolExpr::Binary {
                            op: rop,
                            lhs: r_lhs,
                            rhs: r_rhs,
                        } = rhs
                        {
                            if let BinaryOp::Mul = &rop {
                                if let Some(e) = self.div_opt(r_lhs, true) {
                                    return match e.div_opt(r_rhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_div(e, r_rhs.as_ref().clone())),
                                    };
                                }
                                if let Some(e) = self.div_opt(r_rhs, true) {
                                    return match e.div_opt(r_lhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_div(e, r_lhs.as_ref().clone())),
                                    };
                                }
                            }
                            if let BinaryOp::Div = &rop {
                                if let Some(e) = self.mul_opt(r_rhs, true) {
                                    return match e.div_opt(r_lhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_div(e, r_lhs.as_ref().clone())),
                                    };
                                }
                                if let Some(e) = self.div_opt(r_lhs, true) {
                                    return match e.mul_opt(r_rhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_mul(e, r_rhs.as_ref().clone())),
                                    };
                                }
                            }
                        }

                        if let BinaryOp::Mul = &op {
                            if let Some(e) = l_lhs.div_opt(rhs, true) {
                                return match e.mul_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.div_opt(rhs, true) {
                                return match l_lhs.mul_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        } else if let BinaryOp::Div = &op {
                            if let Some(e) = l_rhs.mul_opt(rhs, true) {
                                return match l_lhs.div_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_div(l_lhs.as_ref().clone(), e)),
                                };
                            }
                            if let Some(e) = l_lhs.div_opt(rhs, true) {
                                return match e.div_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_div(e, l_rhs.as_ref().clone())),
                                };
                            }
                        }
                        None
                    } else {
                        match rhs {
                            SymbolExpr::Value(v) => match l_lhs.as_ref() {
                                SymbolExpr::Value(lv) => match op {
                                    BinaryOp::Mul => Some(_mul(
                                        SymbolExpr::Value(lv / v),
                                        l_rhs.as_ref().clone(),
                                    )),
                                    BinaryOp::Div => Some(_div(
                                        SymbolExpr::Value(lv / v),
                                        l_rhs.as_ref().clone(),
                                    )),
                                    _ => None,
                                },
                                _ => match l_rhs.as_ref() {
                                    SymbolExpr::Value(rv) => match op {
                                        BinaryOp::Mul => Some(_mul(
                                            SymbolExpr::Value(rv / v),
                                            l_lhs.as_ref().clone(),
                                        )),
                                        BinaryOp::Div => Some(_mul(
                                            SymbolExpr::Value(v * rv).rcp(),
                                            l_lhs.as_ref().clone(),
                                        )),
                                        _ => None,
                                    },
                                    _ => None,
                                },
                            },
                            SymbolExpr::Binary {
                                op: rop,
                                lhs: r_lhs,
                                rhs: r_rhs,
                            } => match (l_lhs.as_ref(), r_lhs.as_ref()) {
                                (SymbolExpr::Value(lv), SymbolExpr::Value(rv)) => match (op, rop) {
                                    (BinaryOp::Mul, BinaryOp::Mul) => Some(_mul(
                                        SymbolExpr::Value(lv / rv),
                                        _div(l_rhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                    )),
                                    (BinaryOp::Mul, BinaryOp::Div) => Some(_mul(
                                        SymbolExpr::Value(lv / rv),
                                        _mul(r_rhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                    )),
                                    (BinaryOp::Div, BinaryOp::Mul) => Some(_div(
                                        SymbolExpr::Value(lv / rv),
                                        _mul(r_rhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                    )),
                                    (BinaryOp::Div, BinaryOp::Div) => Some(_mul(
                                        SymbolExpr::Value(lv / rv),
                                        _div(r_rhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                    )),
                                    (_, _) => None,
                                },
                                (_, _) => None,
                            },
                            _ => None,
                        }
                    }
                }
            }
        }
    }

    /// expand with optimization for div operation
    fn div_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Unary { op, expr } => match op {
                UnaryOp::Neg => match expr.div_expand(rhs) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => match expr.div_opt(rhs, true) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => None,
                    },
                },
                _ => None,
            },
            SymbolExpr::Binary {
                op,
                lhs: l_lhs,
                rhs: l_rhs,
            } => match &op {
                BinaryOp::Add | BinaryOp::Sub => {
                    let l = match l_lhs.div_expand(rhs) {
                        Some(e) => e,
                        None => match l_lhs.div_opt(rhs, true) {
                            Some(e) => e,
                            None => _div(l_lhs.as_ref().clone(), rhs.clone()),
                        },
                    };
                    let r = match l_rhs.div_expand(rhs) {
                        Some(e) => e,
                        None => match l_rhs.div_opt(rhs, true) {
                            Some(e) => e,
                            None => _div(l_rhs.as_ref().clone(), rhs.clone()),
                        },
                    };
                    match &op {
                        BinaryOp::Sub => match l.sub_opt(&r, true) {
                            Some(e) => Some(e),
                            None => Some(_sub(l, r)),
                        },
                        _ => match l.add_opt(&r, true) {
                            Some(e) => Some(e),
                            None => Some(_add(l, r)),
                        },
                    }
                }
                _ => None,
            },
            _ => self.div_opt(rhs, true),
        }
    }

    /// optimization for neg
    fn neg_opt(&self) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Value(v) => Some(SymbolExpr::Value(-v)),
            SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr,
            } => Some(expr.as_ref().clone()),
            SymbolExpr::Binary { op, lhs, rhs } => match &op {
                BinaryOp::Add => match lhs.neg_opt() {
                    Some(ln) => match rhs.neg_opt() {
                        Some(rn) => Some(_add(ln, rn)),
                        None => Some(_sub(ln, rhs.as_ref().clone())),
                    },
                    None => match rhs.neg_opt() {
                        Some(rn) => Some(_add(_neg(lhs.as_ref().clone()), rn)),
                        None => Some(_sub(_neg(lhs.as_ref().clone()), rhs.as_ref().clone())),
                    },
                },
                BinaryOp::Sub => match lhs.neg_opt() {
                    Some(ln) => Some(_add(ln, rhs.as_ref().clone())),
                    None => Some(_add(_neg(lhs.as_ref().clone()), rhs.as_ref().clone())),
                },
                BinaryOp::Mul => match lhs.neg_opt() {
                    Some(ln) => Some(_mul(ln, rhs.as_ref().clone())),
                    None => rhs.neg_opt().map(|rn| _mul(lhs.as_ref().clone(), rn)),
                },
                BinaryOp::Div => match lhs.neg_opt() {
                    Some(ln) => Some(_div(ln, rhs.as_ref().clone())),
                    None => rhs.neg_opt().map(|rn| _div(lhs.as_ref().clone(), rn)),
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// optimize the equation
    pub fn optimize(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(_) => self.clone(),
            SymbolExpr::Symbol(_) => self.clone(),
            SymbolExpr::Unary { op, expr } => {
                let opt = expr.optimize();
                match op {
                    UnaryOp::Neg => match opt.neg_opt() {
                        Some(e) => e,
                        None => _neg(opt),
                    },
                    _ => SymbolExpr::Unary {
                        op: op.clone(),
                        expr: Arc::new(opt),
                    },
                }
            }
            SymbolExpr::Binary { op, lhs, rhs } => {
                let opt_lhs = lhs.optimize();
                let opt_rhs = rhs.optimize();
                match op {
                    BinaryOp::Add => match opt_lhs.add_opt(&opt_rhs, true) {
                        Some(e) => e,
                        None => _add(opt_lhs, opt_rhs),
                    },
                    BinaryOp::Sub => match opt_lhs.sub_opt(&opt_rhs, true) {
                        Some(e) => e,
                        None => _sub(opt_lhs, opt_rhs),
                    },
                    BinaryOp::Mul => match opt_lhs.mul_opt(&opt_rhs, true) {
                        Some(e) => e,
                        None => _mul(opt_lhs, opt_rhs),
                    },
                    BinaryOp::Div => match opt_lhs.div_opt(&opt_rhs, true) {
                        Some(e) => e,
                        None => _div(opt_lhs, opt_rhs),
                    },
                    BinaryOp::Pow => _pow(opt_lhs, opt_rhs),
                }
            }
        }
    }

    // convert sympy compatible format
    pub fn sympify(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol { .. } => self.clone(),
            SymbolExpr::Value(e) => e.sympify(),
            SymbolExpr::Unary { op, expr } => SymbolExpr::Unary {
                op: op.clone(),
                expr: Arc::new(expr.sympify()),
            },
            SymbolExpr::Binary { op, lhs, rhs } => SymbolExpr::Binary {
                op: op.clone(),
                lhs: Arc::new(lhs.sympify()),
                rhs: Arc::new(rhs.sympify()),
            },
        }
    }

    fn repr(&self, with_uuid: bool) -> String {
        match self {
            SymbolExpr::Symbol(e) => e.repr(with_uuid),
            SymbolExpr::Value(e) => e.to_string(),
            SymbolExpr::Unary { op, expr } => {
                let s = expr.repr(with_uuid);
                match op {
                    UnaryOp::Abs => format!("abs({s})"),
                    UnaryOp::Neg => match expr.as_ref() {
                        SymbolExpr::Value(e) => (-e).to_string(),
                        SymbolExpr::Binary {
                            op: BinaryOp::Add | BinaryOp::Sub,
                            ..
                        } => format!("-({s})"),
                        _ => format!("-{s}"),
                    },
                    UnaryOp::Sin => format!("sin({s})"),
                    UnaryOp::Asin => format!("asin({s})"),
                    UnaryOp::Cos => format!("cos({s})"),
                    UnaryOp::Acos => format!("acos({s})"),
                    UnaryOp::Tan => format!("tan({s})"),
                    UnaryOp::Atan => format!("atan({s})"),
                    UnaryOp::Exp => format!("exp({s})"),
                    UnaryOp::Log => format!("log({s})"),
                    UnaryOp::Sign => format!("sign({s})"),
                    UnaryOp::Conj => format!("conj({s})"),
                }
            }
            SymbolExpr::Binary { op, lhs, rhs } => {
                let s_lhs = lhs.repr(with_uuid);
                let s_rhs = rhs.repr(with_uuid);
                let op_lhs = match lhs.as_ref() {
                    SymbolExpr::Binary { op: lop, .. } => {
                        matches!(lop, BinaryOp::Add | BinaryOp::Sub)
                    }
                    SymbolExpr::Value(e) => match e {
                        Value::Real(v) => *v < 0.0,
                        Value::Int(v) => *v < 0,
                        Value::Complex(_) => true,
                    },
                    _ => false,
                };
                let op_rhs = match rhs.as_ref() {
                    SymbolExpr::Binary { op: rop, .. } => match rop {
                        BinaryOp::Add | BinaryOp::Sub => true,
                        _ => matches!(op, BinaryOp::Div),
                    },
                    SymbolExpr::Value(e) => match e {
                        Value::Real(v) => *v < 0.0,
                        Value::Int(v) => *v < 0,
                        Value::Complex(_) => true,
                    },
                    _ => false,
                };

                match op {
                    BinaryOp::Add => match rhs.as_ref() {
                        SymbolExpr::Unary {
                            op: UnaryOp::Neg,
                            expr: _,
                        } => {
                            if s_rhs.as_str().char_indices().nth(0).unwrap().1 == '-' {
                                format!("{s_lhs} {s_rhs}")
                            } else {
                                format!("{s_lhs} + {s_rhs}")
                            }
                        }
                        _ => format!("{s_lhs} + {s_rhs}"),
                    },
                    BinaryOp::Sub => match rhs.as_ref() {
                        SymbolExpr::Unary {
                            op: UnaryOp::Neg,
                            expr: _,
                        } => {
                            if s_rhs.as_str().char_indices().nth(0).unwrap().1 == '-' {
                                let st = s_rhs.char_indices().nth(0).unwrap().0;
                                let ed = s_rhs.char_indices().nth(1).unwrap().0;
                                let s_rhs_new: &str = &s_rhs.as_str()[st..ed];
                                format!("{s_lhs} + {s_rhs_new}")
                            } else if op_rhs {
                                format!("{s_lhs} -({s_rhs})")
                            } else {
                                format!("{s_lhs} - {s_rhs}")
                            }
                        }
                        _ => {
                            if op_rhs {
                                format!("{s_lhs} -({s_rhs})")
                            } else {
                                format!("{s_lhs} - {s_rhs}")
                            }
                        }
                    },
                    BinaryOp::Mul => {
                        if op_lhs {
                            if op_rhs {
                                format!("({s_lhs})*({s_rhs})")
                            } else {
                                format!("({s_lhs})*{s_rhs}")
                            }
                        } else if op_rhs {
                            format!("{s_lhs}*({s_rhs})")
                        } else {
                            format!("{s_lhs}*{s_rhs}")
                        }
                    }
                    BinaryOp::Div => {
                        if op_lhs {
                            if op_rhs {
                                format!("({s_lhs})/({s_rhs})")
                            } else {
                                format!("({s_lhs})/{s_rhs}")
                            }
                        } else if op_rhs {
                            format!("{s_lhs}/({s_rhs})")
                        } else {
                            format!("{s_lhs}/{s_rhs}")
                        }
                    }
                    BinaryOp::Pow => match lhs.as_ref() {
                        SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. } => {
                            match rhs.as_ref() {
                                SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. } => {
                                    format!("({s_lhs})**({s_rhs})")
                                }
                                SymbolExpr::Value(r) => {
                                    if r.as_real() < 0.0 {
                                        format!("({s_lhs})**({s_rhs})")
                                    } else {
                                        format!("({s_lhs})**{s_rhs}")
                                    }
                                }
                                _ => format!("({s_lhs})**{s_rhs}"),
                            }
                        }
                        SymbolExpr::Value(l) => {
                            if l.as_real() < 0.0 {
                                match rhs.as_ref() {
                                    SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. } => {
                                        format!("({s_lhs})**({s_rhs})")
                                    }
                                    _ => format!("({s_lhs})**{s_rhs}"),
                                }
                            } else {
                                match rhs.as_ref() {
                                    SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. } => {
                                        format!("{s_lhs}**({s_rhs})")
                                    }
                                    _ => format!("{s_lhs}**{s_rhs}"),
                                }
                            }
                        }
                        _ => match rhs.as_ref() {
                            SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. } => {
                                format!("{s_lhs}**({s_rhs})")
                            }
                            SymbolExpr::Value(r) => {
                                if r.as_real() < 0.0 {
                                    format!("{s_lhs}**({s_rhs})")
                                } else {
                                    format!("{s_lhs}**{s_rhs}")
                                }
                            }
                            _ => format!("{s_lhs}**{s_rhs}"),
                        },
                    },
                }
            }
        }
    }
}

impl Add for SymbolExpr {
    type Output = SymbolExpr;
    fn add(self, rhs: Self) -> SymbolExpr {
        match self.add_opt(&rhs, false) {
            Some(e) => e,
            None => _add(self, rhs),
        }
    }
}

impl Add for &SymbolExpr {
    type Output = SymbolExpr;
    fn add(self, rhs: Self) -> SymbolExpr {
        match self.add_opt(rhs, false) {
            Some(e) => e,
            None => _add(self.clone(), rhs.clone()),
        }
    }
}

impl Sub for SymbolExpr {
    type Output = SymbolExpr;
    fn sub(self, rhs: Self) -> SymbolExpr {
        match self.sub_opt(&rhs, false) {
            Some(e) => e,
            None => _sub(self, rhs),
        }
    }
}

impl Sub for &SymbolExpr {
    type Output = SymbolExpr;
    fn sub(self, rhs: Self) -> SymbolExpr {
        match self.sub_opt(rhs, false) {
            Some(e) => e,
            None => _sub(self.clone(), rhs.clone()),
        }
    }
}

impl Mul for SymbolExpr {
    type Output = SymbolExpr;
    fn mul(self, rhs: Self) -> SymbolExpr {
        match self.mul_opt(&rhs, false) {
            Some(e) => e,
            None => _mul(self, rhs),
        }
    }
}

impl Mul for &SymbolExpr {
    type Output = SymbolExpr;
    fn mul(self, rhs: Self) -> SymbolExpr {
        match self.mul_opt(rhs, false) {
            Some(e) => e,
            None => _mul(self.clone(), rhs.clone()),
        }
    }
}

impl Div for SymbolExpr {
    type Output = SymbolExpr;
    fn div(self, rhs: Self) -> SymbolExpr {
        match self.div_opt(&rhs, false) {
            Some(e) => e,
            None => _div(self, rhs),
        }
    }
}

impl Div for &SymbolExpr {
    type Output = SymbolExpr;
    fn div(self, rhs: Self) -> SymbolExpr {
        match self.div_opt(rhs, false) {
            Some(e) => e,
            None => _div(self.clone(), rhs.clone()),
        }
    }
}

impl Neg for SymbolExpr {
    type Output = SymbolExpr;
    fn neg(self) -> SymbolExpr {
        match self.neg_opt() {
            Some(e) => e,
            None => _neg(self),
        }
    }
}

impl Neg for &SymbolExpr {
    type Output = SymbolExpr;
    fn neg(self) -> SymbolExpr {
        match self.neg_opt() {
            Some(e) => e,
            None => _neg(self.clone()),
        }
    }
}

impl PartialEq for SymbolExpr {
    fn eq(&self, rexpr: &Self) -> bool {
        if let (Some(l), Some(r)) = (self.eval(true), rexpr.eval(true)) {
            return l == r;
        }
        match (self, rexpr) {
            (SymbolExpr::Symbol(l), SymbolExpr::Symbol(r)) => l == r,
            (SymbolExpr::Value(l), SymbolExpr::Value(r)) => l == r,
            (
                SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. },
                SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. },
            ) => {
                let ex_lhs = self.expand();
                let ex_rhs = rexpr.expand();
                match ex_lhs.sub_opt(&ex_rhs, true) {
                    Some(e) => e.is_zero(),
                    None => {
                        let t = &ex_lhs - &ex_rhs;
                        t.is_zero()
                    }
                }
            }
            (SymbolExpr::Binary { .. }, _) => {
                let ex_lhs = self.expand();
                match ex_lhs.sub_opt(rexpr, true) {
                    Some(e) => e.is_zero(),
                    None => {
                        let t = &ex_lhs - rexpr;
                        t.is_zero()
                    }
                }
            }
            (_, SymbolExpr::Binary { .. }) => {
                let ex_rhs = rexpr.expand();
                match self.sub_opt(&ex_rhs, true) {
                    Some(e) => e.is_zero(),
                    None => {
                        let t = self - &ex_rhs;
                        t.is_zero()
                    }
                }
            }
            (_, _) => false,
        }
    }
}

impl PartialEq<f64> for SymbolExpr {
    fn eq(&self, r: &f64) -> bool {
        match self.eval(true) {
            Some(v) => v == *r,
            None => false,
        }
    }
}

impl PartialEq<Complex64> for SymbolExpr {
    fn eq(&self, r: &Complex64) -> bool {
        match self.eval(true) {
            Some(v) => v == *r,
            None => false,
        }
    }
}

// comparison rules for sorting equation
impl PartialOrd for SymbolExpr {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match self {
            SymbolExpr::Value(l) => match rhs {
                SymbolExpr::Value(r) => l.partial_cmp(r),
                _ => Some(Ordering::Less),
            },
            SymbolExpr::Symbol(l) => match rhs {
                SymbolExpr::Value(_) => Some(Ordering::Greater),
                SymbolExpr::Symbol(r) => l.partial_cmp(r),
                SymbolExpr::Unary { op: _, expr } => self.partial_cmp(expr),
                _ => Some(Ordering::Less),
            },
            SymbolExpr::Unary { op: _, expr } => match rhs {
                SymbolExpr::Value(_) => Some(Ordering::Greater),
                SymbolExpr::Unary { op: _, expr: rexpr } => expr.partial_cmp(rexpr),
                _ => (expr.as_ref()).partial_cmp(rhs),
            },
            SymbolExpr::Binary {
                op,
                lhs: ll,
                rhs: lr,
            } => match rhs {
                SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => match op {
                    BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow => Some(Ordering::Greater),
                    _ => Some(Ordering::Equal),
                },
                SymbolExpr::Unary { op: _, expr } => self.partial_cmp(expr),
                SymbolExpr::Binary {
                    op: _,
                    lhs: rl,
                    rhs: rr,
                } => {
                    let ls = match ll.as_ref() {
                        SymbolExpr::Value(_) => lr.string_id(),
                        _ => self.string_id(),
                    };
                    let rs = match rl.as_ref() {
                        SymbolExpr::Value(_) => rr.string_id(),
                        _ => rhs.string_id(),
                    };
                    if rs > ls && rs.len() > ls.len() {
                        Some(Ordering::Less)
                    } else if rs < ls && rs.len() < ls.len() {
                        Some(Ordering::Greater)
                    } else {
                        Some(Ordering::Equal)
                    }
                }
            },
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Value::Real(e) => e.to_string(),
                Value::Int(e) => e.to_string(),
                Value::Complex(e) => {
                    if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&e.re) {
                        if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&e.im) {
                            0.to_string()
                        } else {
                            format!("{}i", e.im)
                        }
                    } else if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&e.im) {
                        e.re.to_string()
                    } else {
                        e.to_string()
                    }
                }
            }
        )
    }
}

// ===============================================================
//  implementations for Value
// ===============================================================
impl Value {
    pub fn as_real(&self) -> f64 {
        match self {
            Value::Real(e) => *e,
            Value::Int(e) => *e as f64,
            Value::Complex(e) => e.re,
        }
    }

    pub fn is_real(&self) -> bool {
        match self {
            Value::Real(_) | Value::Int(_) => true,
            Value::Complex(c) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im),
        }
    }

    pub fn abs(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.abs()),
            Value::Int(e) => Value::Int(e.abs()),
            Value::Complex(e) => Value::Real((e.re * e.re + e.im * e.im).sqrt()),
        }
    }

    pub fn sin(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.sin()),
            Value::Int(e) => Value::Real((*e as f64).sin()),
            Value::Complex(e) => {
                let t = Value::Complex(e.sin());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn asin(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.asin()),
            Value::Int(e) => Value::Real((*e as f64).asin()),
            Value::Complex(e) => {
                let t = Value::Complex(e.asin());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn cos(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.cos()),
            Value::Int(e) => Value::Real((*e as f64).cos()),
            Value::Complex(e) => {
                let t = Value::Complex(e.cos());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn acos(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.acos()),
            Value::Int(e) => Value::Real((*e as f64).acos()),
            Value::Complex(e) => {
                let t = Value::Complex(e.acos());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn tan(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.tan()),
            Value::Int(e) => Value::Real((*e as f64).tan()),
            Value::Complex(e) => {
                let t = Value::Complex(e.tan());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn atan(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.atan()),
            Value::Int(e) => Value::Real((*e as f64).atan()),
            Value::Complex(e) => {
                let t = Value::Complex(e.atan());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn exp(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.exp()),
            Value::Int(e) => Value::Real((*e as f64).exp()),
            Value::Complex(e) => {
                let t = Value::Complex(e.exp());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn log(&self) -> Value {
        match self {
            Value::Real(e) => {
                if *e < 0.0 {
                    Value::Complex(Complex64::from(e)).log()
                } else {
                    Value::Real(e.ln())
                }
            }
            Value::Int(e) => Value::Real(*e as f64).log(),
            Value::Complex(e) => {
                let t = Value::Complex(e.ln());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn sqrt(&self) -> Value {
        match self {
            Value::Real(e) => {
                if *e < 0.0 {
                    Value::Complex(Complex64::from(e)).sqrt()
                } else {
                    Value::Real(e.sqrt())
                }
            }
            Value::Int(e) => {
                if *e < 0 {
                    Value::Complex(Complex64::from(*e as f64)).pow(&Value::Real(0.5))
                } else {
                    let t = (*e as f64).sqrt();
                    let d = t.floor() - t;
                    if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&d) {
                        Value::Int(t as i64)
                    } else {
                        Value::Real(t)
                    }
                }
            }
            Value::Complex(e) => {
                let t = Value::Complex(e.sqrt());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn pow(&self, p: &Value) -> Value {
        match self {
            Value::Real(e) => match p {
                Value::Real(r) => {
                    if *e < 0.0 && r.fract() != 0. {
                        Value::Complex(Complex64::from(e)).pow(p)
                    } else {
                        Value::Real(e.powf(*r))
                    }
                }
                Value::Int(i) => Value::Real(e.powf(*i as f64)),
                Value::Complex(_) => Value::Complex(Complex64::from(e)).pow(p),
            },
            Value::Int(e) => match p {
                Value::Real(r) => {
                    if *e < 0 && r.fract() != 0. {
                        Value::Complex(Complex64::from(*e as f64)).pow(p)
                    } else {
                        let t = (*e as f64).powf(*r);
                        let d = t.floor() - t;
                        if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&d) {
                            Value::Int(t as i64)
                        } else {
                            Value::Real(t)
                        }
                    }
                }
                Value::Int(r) => {
                    if *r < 0 {
                        Value::Real(*e as f64).pow(p)
                    } else {
                        Value::Int(e.pow(*r as u32))
                    }
                }
                Value::Complex(_) => Value::Complex(Complex64::from(*e as f64)).pow(p),
            },
            Value::Complex(e) => {
                let t = match p {
                    Value::Real(r) => Value::Complex(e.powf(*r)),
                    Value::Int(r) => Value::Complex(e.powf(*r as f64)),
                    Value::Complex(r) => Value::Complex(e.powc(*r)),
                };
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn rcp(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(1.0 / e),
            Value::Int(e) => {
                let t = 1.0 / (*e as f64);
                let d = t.floor() - t;
                if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&d) {
                    Value::Int(t as i64)
                } else {
                    Value::Real(t)
                }
            }
            Value::Complex(e) => Value::Complex(1.0 / e),
        }
    }
    pub fn sign(&self) -> Value {
        match self {
            Value::Real(e) => {
                if *e > SYMEXPR_EPSILON {
                    Value::Real(1.0)
                } else if *e < -SYMEXPR_EPSILON {
                    Value::Real(-1.0)
                } else {
                    Value::Real(0.0)
                }
            }
            Value::Int(e) => {
                if *e > 0 {
                    Value::Int(1)
                } else if *e < 0 {
                    Value::Int(-1)
                } else {
                    Value::Int(0)
                }
            }
            Value::Complex(_) => *self,
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Value::Real(r) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(r),
            Value::Int(i) => *i == 0,
            Value::Complex(c) => {
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.re)
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im)
            }
        }
    }
    pub fn is_one(&self) -> bool {
        match self {
            Value::Real(r) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(*r - 1.0)),
            Value::Int(i) => *i == 1,
            Value::Complex(c) => {
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(c.re - 1.0))
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im)
            }
        }
    }
    pub fn is_minus_one(&self) -> bool {
        match self {
            Value::Real(r) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(*r + 1.0)),
            Value::Int(i) => *i == -1,
            Value::Complex(c) => {
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(c.re + 1.0))
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im)
            }
        }
    }

    pub fn is_negative(&self) -> bool {
        match self {
            Value::Real(r) => *r < 0.0,
            Value::Int(i) => *i < 0,
            Value::Complex(c) => {
                (c.re < 0.0 && c.im < SYMEXPR_EPSILON && c.im > -SYMEXPR_EPSILON)
                    || (c.im < 0.0 && c.re < SYMEXPR_EPSILON && c.re > -SYMEXPR_EPSILON)
            }
        }
    }

    fn mul_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(r) => Some(SymbolExpr::Value(self * r)),
            SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr,
            } => {
                let l = SymbolExpr::Value(-self);
                match l.mul_opt(expr, recursive) {
                    Some(e) => Some(e),
                    None => Some(_mul(l, expr.as_ref().clone())),
                }
            }
            SymbolExpr::Binary { op, lhs: l, rhs: r } => {
                if recursive {
                    match op {
                        BinaryOp::Mul => match self.mul_opt(l, recursive) {
                            Some(e) => match e.mul_opt(r, recursive) {
                                Some(ee) => Some(ee),
                                None => Some(_mul(e, r.as_ref().clone())),
                            },
                            None => self
                                .mul_opt(r, recursive)
                                .map(|e| _mul(e, l.as_ref().clone())),
                        },
                        BinaryOp::Div => match self.mul_opt(l, recursive) {
                            Some(e) => Some(_div(e, r.as_ref().clone())),
                            None => self
                                .div_opt(r, recursive)
                                .map(|e| _mul(e, l.as_ref().clone())),
                        },
                        _ => None,
                    }
                } else {
                    match l.as_ref() {
                        SymbolExpr::Value(v) => match op {
                            BinaryOp::Mul => {
                                Some(_mul(SymbolExpr::Value(self * v), r.as_ref().clone()))
                            }
                            BinaryOp::Div => {
                                Some(_div(SymbolExpr::Value(self * v), r.as_ref().clone()))
                            }
                            _ => None,
                        },
                        _ => match r.as_ref() {
                            SymbolExpr::Value(v) => match op {
                                BinaryOp::Mul => {
                                    Some(_mul(SymbolExpr::Value(self * v), l.as_ref().clone()))
                                }
                                BinaryOp::Div => {
                                    Some(_mul(SymbolExpr::Value(self / v), l.as_ref().clone()))
                                }
                                _ => None,
                            },
                            _ => None,
                        },
                    }
                }
            }
            _ => None,
        }
    }

    fn div_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(r) => Some(SymbolExpr::Value(self / r)),
            SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr,
            } => {
                if recursive {
                    self.div_opt(expr, recursive).map(_neg)
                } else {
                    None
                }
            }
            SymbolExpr::Binary { op, lhs: l, rhs: r } => match l.as_ref() {
                SymbolExpr::Value(v) => match op {
                    BinaryOp::Mul => Some(_div(SymbolExpr::Value(self / v), r.as_ref().clone())),
                    BinaryOp::Div => Some(_mul(SymbolExpr::Value(self / v), r.as_ref().clone())),
                    _ => None,
                },
                _ => match r.as_ref() {
                    SymbolExpr::Value(v) => match op {
                        BinaryOp::Mul => {
                            Some(_div(SymbolExpr::Value(self / v), l.as_ref().clone()))
                        }
                        BinaryOp::Div => {
                            Some(_div(SymbolExpr::Value(self * v), l.as_ref().clone()))
                        }
                        _ => None,
                    },
                    _ => None,
                },
            },
            _ => None,
        }
    }

    pub fn opt_complex(&self) -> Option<Value> {
        match self {
            Value::Complex(c) => {
                if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
                    Some(Value::Real(c.re))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    // convert sympy compatible format
    pub fn sympify(&self) -> SymbolExpr {
        match self {
            // imaginary number is converted to value * symbol 'I'
            Value::Complex(c) => _add(
                SymbolExpr::Value(Value::Real(c.re)),
                _mul(
                    SymbolExpr::Value(Value::Real(c.im)),
                    SymbolExpr::Symbol(Arc::new(Symbol::standalone("I".to_owned(), None))),
                ),
            ),
            _ => SymbolExpr::Value(*self),
        }
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Real(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}

impl TryFrom<u64> for Value {
    fn try_from(value: u64) -> Result<Self, Self::Error> {
        let as_i64: i64 = value.try_into()?;
        Ok(as_i64.into())
    }

    type Error = <i64 as TryFrom<u64>>::Error;
}

impl From<Complex64> for Value {
    fn from(v: Complex64) -> Self {
        if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&v.im) {
            Value::Real(v.re)
        } else {
            Value::Complex(v)
        }
    }
}

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Value {
        *self + *rhs
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l + r),
                Value::Int(r) => Value::Real(l + r as f64),
                Value::Complex(r) => Value::Complex(l + r),
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => Value::Real(l as f64 + r),
                Value::Int(r) => Value::Int(l + r),
                Value::Complex(r) => Value::Complex(l as f64 + r),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l + r),
                Value::Int(r) => Value::Complex(l + r as f64),
                Value::Complex(r) => Value::Complex(l + r),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Value {
        *self - *rhs
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l - r),
                Value::Int(r) => Value::Real(l - r as f64),
                Value::Complex(r) => Value::Complex(l - r),
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => Value::Real(l as f64 - r),
                Value::Int(r) => Value::Int(l - r),
                Value::Complex(r) => Value::Complex(l as f64 - r),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l - r),
                Value::Int(r) => Value::Complex(l - r as f64),
                Value::Complex(r) => Value::Complex(l - r),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Value {
        *self * *rhs
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l * r),
                Value::Int(r) => Value::Real(l * r as f64),
                Value::Complex(r) => Value::Complex(l * r),
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => Value::Real(l as f64 * r),
                Value::Int(r) => Value::Int(l * r),
                Value::Complex(r) => Value::Complex(l as f64 * r),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l * r),
                Value::Int(r) => Value::Complex(l * r as f64),
                Value::Complex(r) => Value::Complex(l * r),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Value {
        *self / *rhs
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l / r),
                Value::Int(r) => Value::Real(l / r as f64),
                Value::Complex(r) => Value::Complex(l / r),
            },
            Value::Int(l) => {
                if rhs == 0.0 {
                    return Value::Real(f64::INFINITY);
                }
                match rhs {
                    Value::Real(r) => Value::Real(l as f64 / r),
                    Value::Int(r) => {
                        let t = l as f64 / r as f64;
                        let d = t.floor() - t;
                        if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&d) {
                            Value::Int(t as i64)
                        } else {
                            Value::Real(t)
                        }
                    }
                    Value::Complex(r) => Value::Complex(l as f64 / r),
                }
            }
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l / r),
                Value::Int(r) => Value::Complex(l / r as f64),
                Value::Complex(r) => Value::Complex(l / r),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        -*self
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        match self {
            Value::Real(v) => Value::Real(-v),
            Value::Int(v) => Value::Int(-v),
            Value::Complex(v) => Value::Complex(-v),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, r: &Self) -> bool {
        match self {
            Value::Real(e) => match r {
                Value::Real(rv) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(e - rv)),
                Value::Int(rv) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(e - *rv as f64)),
                Value::Complex(rv) => {
                    let t = Complex64::from(*e) - rv;
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
            },
            Value::Int(e) => match r {
                Value::Int(rv) => e == rv,
                Value::Real(rv) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(*e as f64 - rv)),
                Value::Complex(rv) => {
                    let t = Complex64::from(*e as f64) - rv;
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
            },
            Value::Complex(e) => match r {
                Value::Real(rv) => {
                    let t = *e - Complex64::from(rv);
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
                Value::Int(rv) => {
                    let t = *e - Complex64::from(*rv as f64);
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
                Value::Complex(rv) => {
                    let t = *e - rv;
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
            },
        }
    }
}

impl PartialEq<f64> for Value {
    fn eq(&self, r: &f64) -> bool {
        match self {
            Value::Real(e) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(e - r)),
            Value::Int(e) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(*e as f64 - r)),
            Value::Complex(e) => {
                let t = *e - Complex64::from(r);
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
            }
        }
    }
}

impl PartialEq<Complex64> for Value {
    fn eq(&self, r: &Complex64) -> bool {
        match self {
            Value::Real(e) => {
                let t = Complex64::from(*e) - r;
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
            }
            Value::Int(e) => {
                let t = Complex64::from(*e as f64) - r;
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
            }
            Value::Complex(e) => {
                let t = *e - r;
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
            }
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => l.partial_cmp(r),
                Value::Int(r) => l.partial_cmp(&(*r as f64)),
                Value::Complex(_) => None,
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => (*l as f64).partial_cmp(r),
                Value::Int(r) => l.partial_cmp(r),
                Value::Complex(_) => None,
            },
            Value::Complex(_) => None,
        }
    }
}

#[cfg(test)]
mod test {
    use std::mem;
    use std::sync::Arc;

    use super::*;

    /// In normal use, a number that's a problem if a recursive call stack gets this deep.  If
    /// running under Miri, though, a much smaller number so that the runtime isn't excessive.
    ///
    /// Used in tests of implementations like `Drop`.  When under Miri, we're testing directly by
    /// instrumentation for leaks and undefined behaviour.  When not under Miri, we're deliberately
    /// stressing the system such that a crash/segfault is indicative that the algorithm is
    /// accidentally recursive, rather than iterative.
    const BIG_CALL_STACK: usize = if cfg!(miri) { 100 } else { 20_000 };

    #[test]
    fn test_drop_unary_large() {
        let base = Arc::new(SymbolExpr::Value(Value::Real(0.0)));
        let mut expr = Arc::new(SymbolExpr::Unary {
            op: UnaryOp::Neg,
            expr: Arc::clone(&base),
        });
        for _ in 0..BIG_CALL_STACK {
            expr = Arc::new(SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr,
            });
        }
        assert_eq!(Arc::strong_count(&base), 2);
        mem::drop(expr);
        assert_eq!(Arc::strong_count(&base), 1);
    }

    #[test]
    fn test_drop_binary_left() {
        let base = Arc::new(SymbolExpr::Value(Value::Real(0.0)));
        let count = BIG_CALL_STACK;
        let mut expr = Arc::new(SymbolExpr::Binary {
            op: BinaryOp::Add,
            lhs: Arc::clone(&base),
            rhs: Arc::clone(&base),
        });
        for _ in 0..count {
            expr = Arc::new(SymbolExpr::Binary {
                op: BinaryOp::Add,
                lhs: expr,
                rhs: Arc::clone(&base),
            });
        }
        // `count` from the loop, two from the initialiser, and one from our `base`.
        assert_eq!(Arc::strong_count(&base), count + 3);
        mem::drop(expr);
        assert_eq!(Arc::strong_count(&base), 1);
    }

    #[test]
    fn test_drop_binary_right() {
        let base = Arc::new(SymbolExpr::Value(Value::Real(0.0)));
        let count = BIG_CALL_STACK;
        let mut expr = Arc::new(SymbolExpr::Binary {
            op: BinaryOp::Add,
            lhs: Arc::clone(&base),
            rhs: Arc::clone(&base),
        });
        for _ in 0..count {
            expr = Arc::new(SymbolExpr::Binary {
                op: BinaryOp::Add,
                lhs: Arc::clone(&base),
                rhs: expr,
            });
        }
        // `count` from the loop, two from the initialiser, and one from our `base`.
        assert_eq!(Arc::strong_count(&base), count + 3);
        mem::drop(expr);
        assert_eq!(Arc::strong_count(&base), 1);
    }

    #[test]
    fn test_drop_binary_balanced() {
        let base = Arc::new(SymbolExpr::Value(Value::Real(0.0)));
        let mut expr = Arc::new(SymbolExpr::Binary {
            op: BinaryOp::Add,
            lhs: Arc::clone(&base),
            rhs: Arc::clone(&base),
        });
        let count = BIG_CALL_STACK / 10;
        for _ in 0..count {
            expr = Arc::new(SymbolExpr::Binary {
                op: BinaryOp::Add,
                lhs: Arc::clone(&expr),
                rhs: expr,
            });
        }
        // Two in the initialiser of `expr`, one in `base`.
        assert_eq!(Arc::strong_count(&base), 3);
        mem::drop(expr);
        assert_eq!(Arc::strong_count(&base), 1);
    }

    #[test]
    fn test_drop_mixed() {
        // A messy tree that on average is balanced left and right, and includes a couple of nested
        // `Unary`s at some levels, just for good measure.
        let base = Arc::new(SymbolExpr::Value(Value::Real(0.0)));
        let mut expr = Arc::new(SymbolExpr::Binary {
            op: BinaryOp::Add,
            lhs: Arc::clone(&base),
            rhs: Arc::clone(&base),
        });
        let count = BIG_CALL_STACK / 2;
        for half in 0..count {
            let mut unary = Arc::new(SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr: Arc::clone(&expr),
            });
            for _ in 0..half {
                unary = Arc::new(SymbolExpr::Unary {
                    op: UnaryOp::Neg,
                    expr: unary,
                });
            }
            if half % 2 == 0 {
                expr = Arc::new(SymbolExpr::Binary {
                    op: BinaryOp::Add,
                    lhs: unary,
                    rhs: expr,
                });
            } else {
                expr = Arc::new(SymbolExpr::Binary {
                    op: BinaryOp::Add,
                    lhs: expr,
                    rhs: unary,
                });
            }
        }
        assert_eq!(Arc::strong_count(&base), 3);
        mem::drop(expr);
        assert_eq!(Arc::strong_count(&base), 1);
    }

    #[test]
    fn test_drop_partial() {
        // We'll keep some refs to partial expressions, and make sure that drops all happen at the
        // correct time.  This test can't catch the case that we make a spurious clone-and-drop
        // because we can't observe the temporary increase in strong counts because of the clone,
        // unless we attempt to race the drop in a separate thread, which would be disgusting.
        let base = Arc::new(SymbolExpr::Value(Value::Real(0.0)));
        let bottom_unary = Arc::new(SymbolExpr::Unary {
            op: UnaryOp::Neg,
            expr: Arc::clone(&base),
        });
        let bottom_binary = Arc::new(SymbolExpr::Binary {
            op: BinaryOp::Add,
            lhs: Arc::clone(&base),
            rhs: Arc::clone(&base),
        });
        let mut expr = Arc::new(SymbolExpr::Binary {
            op: BinaryOp::Add,
            // We keep _our_ refs to `bottom_unary` and `bottom_binary` alive here.
            lhs: Arc::clone(&bottom_unary),
            rhs: Arc::clone(&bottom_binary),
        });

        for _ in 0..BIG_CALL_STACK {
            expr = Arc::new(SymbolExpr::Binary {
                op: BinaryOp::Add,
                lhs: Arc::clone(&expr),
                rhs: expr,
            });
        }

        let mid_binary = expr;
        let mut mid_unaries = Vec::new();
        let mut mid_unary = Arc::new(SymbolExpr::Unary {
            op: UnaryOp::Neg,
            expr: Arc::clone(&mid_binary),
        });
        for _ in 0..BIG_CALL_STACK {
            mid_unary = Arc::new(SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr: mid_unary,
            });
            // Keep loads of layers of these alive to catch the case of accidental recursion when
            // many strong references are alive.
            mid_unaries.push(Arc::clone(&mid_unary));
        }

        expr = Arc::new(SymbolExpr::Binary {
            op: BinaryOp::Add,
            lhs: Arc::clone(&mid_unary),
            rhs: Arc::clone(&mid_binary),
        });
        for _ in 0..BIG_CALL_STACK {
            expr = Arc::new(SymbolExpr::Binary {
                op: BinaryOp::Add,
                rhs: Arc::clone(&expr),
                lhs: expr,
            });
        }

        // `base`, `bottom_unary` and 2x `bottom_binary`.
        assert_eq!(Arc::strong_count(&base), 4);
        // `bottom_unary` and initialiser of `expr`.
        assert_eq!(Arc::strong_count(&bottom_unary), 2);
        // `bottom_binary` and initialiser of `expr`.
        assert_eq!(Arc::strong_count(&bottom_binary), 2);
        // `mid_unary`, the `mid_unaries` vec, and the relevant layer of `expr`.
        assert_eq!(Arc::strong_count(&mid_unary), 3);
        // `mid_binary`, the relevant layer of `expr`, and the bottom of `mid_unary`.
        assert_eq!(Arc::strong_count(&mid_binary), 3);

        mem::drop(expr);
        assert_eq!(Arc::strong_count(&base), 4); // Unchanged; all refs should surive.
        assert_eq!(Arc::strong_count(&bottom_unary), 2); // Unchanged: the `mid`s aren't dropped.
        assert_eq!(Arc::strong_count(&bottom_binary), 2); // Unchanged: same reason.
        assert_eq!(Arc::strong_count(&mid_unary), 2); // Dropped everything above.
        assert_eq!(Arc::strong_count(&mid_binary), 2); // Dropped everything above.

        mem::drop(mid_unaries); // Mostly no effect because `mid_unary` survives.
        assert_eq!(Arc::strong_count(&mid_unary), 1);
        mem::drop(mid_unary);
        assert_eq!(Arc::strong_count(&base), 4); // Unchanged; all refs should surive.
        assert_eq!(Arc::strong_count(&bottom_unary), 2); // Unchanged: `mid_binary` still survives.
        assert_eq!(Arc::strong_count(&bottom_binary), 2); // Unchanged: same reason.

        mem::drop(mid_binary);
        assert_eq!(Arc::strong_count(&base), 4); // Unchanged; all refs should surive.
        assert_eq!(Arc::strong_count(&bottom_unary), 1);
        assert_eq!(Arc::strong_count(&bottom_binary), 1);

        mem::drop(bottom_unary);
        mem::drop(bottom_binary);
        assert_eq!(Arc::strong_count(&base), 1);
    }

    #[test]
    fn test_drop_weak() {
        // We'll keep hold of a `Weak` ref to some partial expressions during the drop, and make
        // sure that everything still drops right without a segfault; `Weak` shouldn't prevent the
        // drop from occurring or cause it to become recursive.
        let base = Arc::new(SymbolExpr::Value(Value::Real(0.0)));

        let mut expr = Arc::new(SymbolExpr::Binary {
            op: BinaryOp::Add,
            lhs: Arc::clone(&base),
            rhs: Arc::clone(&base),
        });
        for _ in 0..BIG_CALL_STACK {
            expr = Arc::new(SymbolExpr::Binary {
                op: BinaryOp::Add,
                lhs: Arc::clone(&expr),
                rhs: expr,
            });
        }

        // We keep _loads_ of intermediate `Weak`s alive to catch any unbounded recursion caused by
        // holding a `Weak`.
        let mut weaks = Vec::new();
        for _ in 0..BIG_CALL_STACK {
            weaks.push(Arc::downgrade(&expr));
            expr = Arc::new(SymbolExpr::Binary {
                op: BinaryOp::Add,
                lhs: Arc::clone(&expr),
                rhs: expr,
            });
        }
        for _ in 0..BIG_CALL_STACK {
            weaks.push(Arc::downgrade(&expr));
            expr = Arc::new(SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr,
            });
        }

        assert_eq!(Arc::strong_count(&base), 3);
        mem::drop(expr);
        assert_eq!(Arc::strong_count(&base), 1);
        assert_eq!(weaks[0].strong_count(), 0);
    }

    #[test]
    fn test_eq_exact_unary_non_simplification() {
        let unary_negate = |expr| SymbolExpr::Unary {
            op: UnaryOp::Neg,
            expr: Arc::new(expr),
        };
        let val = Value::Int(1);
        let direct = SymbolExpr::Value(-val);
        let indirect = unary_negate(SymbolExpr::Value(val));
        assert_eq!(direct, indirect);
        assert!(!direct.eq_exact(&indirect));

        let two_extra_negations = unary_negate(unary_negate(indirect.clone()));
        assert_eq!(indirect, two_extra_negations);
        assert!(!direct.eq_exact(&indirect));
    }

    #[test]
    fn test_eq_exact_unary_deep() {
        let make_big = |mut expr| {
            for _ in 0..BIG_CALL_STACK {
                expr = SymbolExpr::Unary {
                    op: UnaryOp::Neg,
                    expr: Arc::new(expr),
                };
            }
            expr
        };

        let one = Value::Int(1);
        let two = Value::Int(2);
        let base = make_big(SymbolExpr::Value(one));
        assert!(base.eq_exact(&make_big(SymbolExpr::Value(one))));
        assert!(!base.eq_exact(&make_big(SymbolExpr::Value(two))));
    }

    #[test]
    fn test_eq_exact_binary_deep() {
        let make_big_lhs = |mut expr| {
            for _ in 0..BIG_CALL_STACK {
                expr = SymbolExpr::Binary {
                    op: BinaryOp::Pow,
                    lhs: Arc::new(expr),
                    rhs: Arc::new(SymbolExpr::Value(Value::Int(2))),
                }
            }
            expr
        };
        let make_big_rhs = |mut expr| {
            for _ in 0..BIG_CALL_STACK {
                expr = SymbolExpr::Binary {
                    op: BinaryOp::Pow,
                    lhs: Arc::new(SymbolExpr::Value(Value::Int(2))),
                    rhs: Arc::new(expr),
                }
            }
            expr
        };
        let one = Value::Int(1);
        let two = Value::Int(2);
        let base = make_big_lhs(SymbolExpr::Value(one));
        assert!(base.eq_exact(&make_big_lhs(SymbolExpr::Value(one))));
        assert!(!base.eq_exact(&make_big_lhs(SymbolExpr::Value(two))));

        let base = make_big_rhs(SymbolExpr::Value(one));
        assert!(base.eq_exact(&make_big_rhs(SymbolExpr::Value(one))));
        assert!(!base.eq_exact(&make_big_rhs(SymbolExpr::Value(two))));
    }

    #[test]
    fn test_eq_exact_binary_backtracks() {
        let extend_lhs = |mut expr, size: usize| {
            for _ in 0..size {
                expr = SymbolExpr::Binary {
                    op: BinaryOp::Pow,
                    lhs: Arc::new(expr),
                    rhs: Arc::new(SymbolExpr::Value(Value::Int(2))),
                }
            }
            expr
        };
        let extend_rhs = |mut expr, size: usize| {
            for _ in 0..size {
                expr = SymbolExpr::Binary {
                    op: BinaryOp::Pow,
                    lhs: Arc::new(SymbolExpr::Value(Value::Int(2))),
                    rhs: Arc::new(expr),
                }
            }
            expr
        };

        let one = Value::Int(1);
        let two = Value::Int(2);

        let left = SymbolExpr::Binary {
            op: BinaryOp::Pow,
            lhs: Arc::new(SymbolExpr::Value(one)),
            rhs: Arc::new(SymbolExpr::Value(one)),
        };
        let right = SymbolExpr::Binary {
            op: BinaryOp::Pow,
            lhs: Arc::new(SymbolExpr::Value(one)),
            rhs: Arc::new(SymbolExpr::Value(two)),
        };

        // Failure deep on the lhs.
        assert!(!extend_lhs(left.clone(), 5).eq_exact(&extend_lhs(right.clone(), 5)));
        // Failure deep on the rhs.
        assert!(!extend_rhs(left.clone(), 5).eq_exact(&extend_rhs(right.clone(), 5)));

        // Failure in a middle right edge of a deep left tree.
        let shared_base = extend_lhs(left.clone(), 5);
        let left_mid = extend_lhs(
            SymbolExpr::Binary {
                op: BinaryOp::Pow,
                lhs: Arc::new(shared_base.clone()),
                rhs: Arc::new(SymbolExpr::Value(one)),
            },
            5,
        );
        let right_mid = extend_lhs(
            SymbolExpr::Binary {
                op: BinaryOp::Pow,
                lhs: Arc::new(shared_base.clone()),
                rhs: Arc::new(SymbolExpr::Value(two)),
            },
            5,
        );
        assert!(!extend_lhs(left_mid, 5).eq_exact(&extend_lhs(right_mid, 5)));

        // Failure on a middle left edge of a deep right tree.
        let shared_base = extend_rhs(right.clone(), 5);
        let left_mid = extend_rhs(
            SymbolExpr::Binary {
                op: BinaryOp::Pow,
                lhs: Arc::new(SymbolExpr::Value(one)),
                rhs: Arc::new(shared_base.clone()),
            },
            5,
        );
        let right_mid = extend_rhs(
            SymbolExpr::Binary {
                op: BinaryOp::Pow,
                lhs: Arc::new(SymbolExpr::Value(two)),
                rhs: Arc::new(shared_base.clone()),
            },
            5,
        );
        assert!(!extend_rhs(left_mid, 5).eq_exact(&extend_rhs(right_mid, 5)));
    }
}
