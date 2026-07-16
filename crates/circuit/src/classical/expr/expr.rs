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

use std::error::Error;

use crate::classical::expr::{Binary, Cast, Index, Stretch, Unary, Value, Var};
use crate::classical::types::Type;
use pyo3::prelude::*;
use pyo3::{IntoPyObjectExt, intern};

/// A classical expression.
///
/// Variants that themselves contain [Expr]s are boxed. This is done instead
/// of boxing the contained [Expr]s within the specific type to reduce the
/// number of boxes we need (e.g. Binary would otherwise contain two boxed
/// expressions).
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Unary(Box<Unary>),
    Binary(Box<Binary>),
    Cast(Box<Cast>),
    Value(Value),
    Var(Var),
    Stretch(Stretch),
    Index(Box<Index>),
}

#[derive(Debug, PartialEq)]
pub enum ExprRef<'a> {
    Unary(&'a Unary),
    Binary(&'a Binary),
    Cast(&'a Cast),
    Value(&'a Value),
    Var(&'a Var),
    Stretch(&'a Stretch),
    Index(&'a Index),
}

#[derive(Debug, PartialEq)]
pub enum ExprRefMut<'a> {
    Unary(&'a mut Unary),
    Binary(&'a mut Binary),
    Cast(&'a mut Cast),
    Value(&'a mut Value),
    Var(&'a mut Var),
    Stretch(&'a mut Stretch),
    Index(&'a mut Index),
}

#[derive(Clone, Debug, PartialEq)]
pub enum IdentifierRef<'a> {
    Var(&'a Var),
    Stretch(&'a Stretch),
}

impl Expr {
    /// Converts from `&Expr` to `ExprRef`.
    pub fn as_ref(&self) -> ExprRef<'_> {
        match self {
            Expr::Unary(u) => ExprRef::Unary(u.as_ref()),
            Expr::Binary(b) => ExprRef::Binary(b.as_ref()),
            Expr::Cast(c) => ExprRef::Cast(c.as_ref()),
            Expr::Value(v) => ExprRef::Value(v),
            Expr::Var(v) => ExprRef::Var(v),
            Expr::Stretch(s) => ExprRef::Stretch(s),
            Expr::Index(i) => ExprRef::Index(i.as_ref()),
        }
    }

    /// Converts from `&mut Expr` to `ExprRefMut`.
    pub fn as_mut(&mut self) -> ExprRefMut<'_> {
        match self {
            Expr::Unary(u) => ExprRefMut::Unary(u.as_mut()),
            Expr::Binary(b) => ExprRefMut::Binary(b.as_mut()),
            Expr::Cast(c) => ExprRefMut::Cast(c.as_mut()),
            Expr::Value(v) => ExprRefMut::Value(v),
            Expr::Var(v) => ExprRefMut::Var(v),
            Expr::Stretch(s) => ExprRefMut::Stretch(s),
            Expr::Index(i) => ExprRefMut::Index(i.as_mut()),
        }
    }

    /// The const-ness of the expression.
    pub fn is_const(&self) -> bool {
        match self {
            Expr::Unary(u) => u.constant,
            Expr::Binary(b) => b.constant,
            Expr::Cast(c) => c.constant,
            Expr::Value(_) => true,
            Expr::Var(_) => false,
            Expr::Stretch(_) => true,
            Expr::Index(i) => i.constant,
        }
    }

    /// The expression's [Type].
    pub fn ty(&self) -> Type {
        match self {
            Expr::Unary(u) => u.ty,
            Expr::Binary(b) => b.ty,
            Expr::Cast(c) => c.ty,
            Expr::Value(v) => match v {
                Value::Duration(_) => Type::Duration,
                Value::Float { ty, .. } => *ty,
                Value::Uint { ty, .. } => *ty,
            },
            Expr::Var(v) => match v {
                Var::Standalone { ty, .. } => *ty,
                Var::Bit { .. } => Type::Bool,
                Var::Register { ty, .. } => *ty,
            },
            Expr::Stretch(_) => Type::Duration,
            Expr::Index(i) => i.ty,
        }
    }

    /// Returns an iterator over the identifier nodes in this expression in some
    /// deterministic order.
    pub fn identifiers(&self) -> impl Iterator<Item = IdentifierRef<'_>> {
        IdentIterator(ExprIterator { stack: vec![self] })
    }

    /// Returns an iterator over the [Var] nodes in this expression in some
    /// deterministic order.
    pub fn vars(&self) -> impl Iterator<Item = &Var> {
        VarIterator(ExprIterator { stack: vec![self] })
    }

    /// Returns an iterator over all nodes in this expression in some deterministic
    /// order.
    pub fn iter(&self) -> impl Iterator<Item = ExprRef<'_>> {
        ExprIterator { stack: vec![self] }
    }

    /// Visits all nodes by mutable reference, in a post-order traversal.
    pub fn visit_mut<F, E>(&mut self, mut visitor: F) -> Result<(), E>
    where
        F: FnMut(ExprRefMut) -> Result<(), E>,
        E: Error,
    {
        self.visit_mut_impl(&mut visitor)
    }

    fn visit_mut_impl<F, E>(&mut self, visitor: &mut F) -> Result<(), E>
    where
        F: FnMut(ExprRefMut) -> Result<(), E>,
        E: Error,
    {
        match self {
            Expr::Unary(u) => u.operand.visit_mut_impl(visitor)?,
            Expr::Binary(b) => {
                b.left.visit_mut_impl(visitor)?;
                b.right.visit_mut_impl(visitor)?;
            }
            Expr::Cast(c) => c.operand.visit_mut_impl(visitor)?,
            Expr::Value(_) => {}
            Expr::Var(_) => {}
            Expr::Stretch(_) => {}
            Expr::Index(i) => {
                i.target.visit_mut_impl(visitor)?;
                i.index.visit_mut_impl(visitor)?;
            }
        }
        visitor(self.as_mut())
    }

    /// Do these two expressions have exactly the same tree structure?
    pub fn structurally_equivalent(&self, other: &Expr) -> bool {
        let identity_key = |v: &Var| v.clone();
        self.structurally_equivalent_by_key(identity_key, other, identity_key)
    }

    /// Do these two expressions have exactly the same tree structure, up to some key function for
    /// [Var] nodes?
    ///
    /// In other words, are these two expressions the exact same trees, except we compare the
    /// [Var] nodes by calling the appropriate `*_var_key` function on them, and comparing
    /// that output for equality.  This function does not allow any semantic "equivalences" such as
    /// asserting that `a == b` is equivalent to `b == a`; the evaluation order of the operands
    /// could, in general, cause such a statement to be false.
    pub fn structurally_equivalent_by_key<F1, F2, K>(
        &self,
        mut self_var_key: F1,
        other: &Expr,
        mut other_var_key: F2,
    ) -> bool
    where
        F1: FnMut(&Var) -> K,
        F2: FnMut(&Var) -> K,
        K: PartialEq,
    {
        let mut self_nodes = self.iter();
        let mut other_nodes = other.iter();
        loop {
            match (self_nodes.next(), other_nodes.next()) {
                (Some(a), Some(b)) => match (a, b) {
                    (ExprRef::Unary(a), ExprRef::Unary(b)) => {
                        if a.op != b.op || a.ty != b.ty || a.constant != b.constant {
                            return false;
                        }
                    }
                    (ExprRef::Binary(a), ExprRef::Binary(b)) => {
                        if a.op != b.op || a.ty != b.ty || a.constant != b.constant {
                            return false;
                        }
                    }
                    (ExprRef::Cast(a), ExprRef::Cast(b)) => {
                        if a.ty != b.ty || a.constant != b.constant {
                            return false;
                        }
                    }
                    (ExprRef::Value(a), ExprRef::Value(b)) => {
                        if a != b {
                            return false;
                        }
                    }
                    (ExprRef::Var(a), ExprRef::Var(b)) => {
                        if a.ty() != b.ty() {
                            return false;
                        }
                        if self_var_key(a) != other_var_key(b) {
                            return false;
                        }
                    }
                    (ExprRef::Stretch(a), ExprRef::Stretch(b)) => {
                        if a.uuid != b.uuid {
                            return false;
                        }
                    }
                    (ExprRef::Index(a), ExprRef::Index(b)) => {
                        if a.ty != b.ty || a.constant != b.constant {
                            return false;
                        }
                    }
                    _ => return false,
                },
                (None, None) => return true,
                _ => return false,
            }
        }
    }
}

/// A private iterator over the [Expr] nodes of an expression
/// by reference.
///
/// The first node reference returned is the [Expr] itself.
struct ExprIterator<'a> {
    stack: Vec<&'a Expr>,
}

impl<'a> Iterator for ExprIterator<'a> {
    type Item = ExprRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let expr = self.stack.pop()?;
        match expr {
            Expr::Unary(u) => {
                self.stack.push(&u.operand);
            }
            Expr::Binary(b) => {
                self.stack.push(&b.left);
                self.stack.push(&b.right);
            }
            Expr::Cast(c) => self.stack.push(&c.operand),
            Expr::Value(_) => {}
            Expr::Var(_) => {}
            Expr::Stretch(_) => {}
            Expr::Index(i) => {
                self.stack.push(&i.index);
                self.stack.push(&i.target);
            }
        }
        Some(expr.as_ref())
    }
}

/// A private iterator over the [Var] nodes contained within an [Expr].
struct VarIterator<'a>(ExprIterator<'a>);

impl<'a> Iterator for VarIterator<'a> {
    type Item = &'a Var;

    fn next(&mut self) -> Option<Self::Item> {
        for expr in self.0.by_ref() {
            if let ExprRef::Var(v) = expr {
                return Some(v);
            }
        }
        None
    }
}

/// A private iterator over the [Var] and [Stretch] nodes contained within an [Expr].
struct IdentIterator<'a>(ExprIterator<'a>);

impl<'a> Iterator for IdentIterator<'a> {
    type Item = IdentifierRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        for expr in self.0.by_ref() {
            if let ExprRef::Var(v) = expr {
                return Some(IdentifierRef::Var(v));
            }
            if let ExprRef::Stretch(s) = expr {
                return Some(IdentifierRef::Stretch(s));
            }
        }
        None
    }
}

impl From<Unary> for Expr {
    fn from(value: Unary) -> Self {
        Expr::Unary(Box::new(value))
    }
}

impl From<Box<Unary>> for Expr {
    fn from(value: Box<Unary>) -> Self {
        Expr::Unary(value)
    }
}

impl From<Binary> for Expr {
    fn from(value: Binary) -> Self {
        Expr::Binary(Box::new(value))
    }
}

impl From<Box<Binary>> for Expr {
    fn from(value: Box<Binary>) -> Self {
        Expr::Binary(value)
    }
}

impl From<Cast> for Expr {
    fn from(value: Cast) -> Self {
        Expr::Cast(Box::new(value))
    }
}

impl From<Box<Cast>> for Expr {
    fn from(value: Box<Cast>) -> Self {
        Expr::Cast(value)
    }
}

impl From<Value> for Expr {
    fn from(value: Value) -> Self {
        Expr::Value(value)
    }
}

impl From<Var> for Expr {
    fn from(value: Var) -> Self {
        Expr::Var(value)
    }
}

impl From<Stretch> for Expr {
    fn from(value: Stretch) -> Self {
        Expr::Stretch(value)
    }
}

impl From<Index> for Expr {
    fn from(value: Index) -> Self {
        Expr::Index(Box::new(value))
    }
}

impl From<Box<Index>> for Expr {
    fn from(value: Box<Index>) -> Self {
        Expr::Index(value)
    }
}

/// Root base class of all nodes in the expression tree.  The base case should never be
/// instantiated directly.
///
/// This must not be subclassed by users; subclasses form the internal data of the representation of
/// expressions, and it does not make sense to add more outside of Qiskit library code.
///
/// All subclasses are responsible for setting their ``type`` attribute in their ``__init__``, and
/// should not call the parent initializer.
#[pyclass(
    eq,
    hash,
    subclass,
    frozen,
    name = "Expr",
    module = "qiskit._accelerate.circuit.classical.expr",
    skip_from_py_object
)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
pub struct PyExpr(pub ExprKind); // ExprKind is used for fast extraction from Python

#[pymethods]
impl PyExpr {
    /// Call the relevant ``visit_*`` method on the given :class:`ExprVisitor`.  The usual entry
    /// point for a simple visitor is to construct it, and then call :meth:`accept` on the root
    /// object to be visited.  For example::
    ///
    ///     expr = ...
    ///     visitor = MyVisitor()
    ///     visitor.accept(expr)
    ///
    /// Subclasses of :class:`Expr` should override this to call the correct virtual method on the
    /// visitor.  This implements double dispatch with the visitor.
    /// return visitor.visit_generic(self)
    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_generic"), (slf,))
    }
}

/// The expression's kind, used internally during Python instance extraction to avoid
/// `isinstance` checks.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExprKind {
    Unary,
    Binary,
    Value,
    Var,
    Cast,
    Stretch,
    Index,
}

impl<'py> IntoPyObject<'py> for Expr {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Expr::Unary(u) => u.into_bound_py_any(py),
            Expr::Binary(b) => b.into_bound_py_any(py),
            Expr::Cast(c) => c.into_bound_py_any(py),
            Expr::Value(v) => v.into_bound_py_any(py),
            Expr::Var(v) => v.into_bound_py_any(py),
            Expr::Stretch(s) => s.into_bound_py_any(py),
            Expr::Index(i) => i.into_bound_py_any(py),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Expr {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let expr: PyRef<'_, PyExpr> = ob.cast()?.borrow();
        match expr.0 {
            ExprKind::Unary => Ok(Expr::Unary(Box::new(ob.extract()?))),
            ExprKind::Binary => Ok(Expr::Binary(Box::new(ob.extract()?))),
            ExprKind::Value => Ok(Expr::Value(ob.extract()?)),
            ExprKind::Var => Ok(Expr::Var(ob.extract()?)),
            ExprKind::Cast => Ok(Expr::Cast(Box::new(ob.extract()?))),
            ExprKind::Stretch => Ok(Expr::Stretch(ob.extract()?)),
            ExprKind::Index => Ok(Expr::Index(Box::new(ob.extract()?))),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::bit::{ClassicalRegister, ShareableClbit};
    use crate::classical::expr::{
        Binary, BinaryOp, Cast, Expr, ExprRef, ExprRefMut, IdentifierRef, Index, Stretch, Unary,
        UnaryOp, Value, Var,
    };
    use crate::classical::types::Type;
    use crate::duration::Duration;
    use num_bigint::BigUint;
    use pyo3::{PyErr, PyResult};
    use uuid::Uuid;

    #[test]
    fn test_vars() {
        let expr: Expr = Binary {
            op: BinaryOp::BitAnd,
            left: Unary {
                op: UnaryOp::BitNot,
                operand: Var::Standalone {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "test".to_string(),
                    ty: Type::Bool,
                }
                .into(),
                ty: Type::Bool,
                constant: false,
            }
            .into(),
            right: Var::Bit {
                bit: ShareableClbit::new_anonymous(),
            }
            .into(),
            ty: Type::Bool,
            constant: false,
        }
        .into();

        let vars: Vec<&Var> = expr.vars().collect();
        assert!(matches!(
            vars.as_slice(),
            [Var::Bit { .. }, Var::Standalone { .. }]
        ));
    }

    #[test]
    fn test_identifiers() {
        let expr: Expr = Binary {
            op: BinaryOp::Mul,
            left: Binary {
                op: BinaryOp::Add,
                left: Var::Standalone {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "test".to_string(),
                    ty: Type::Duration,
                }
                .into(),
                right: Stretch {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "test".to_string(),
                }
                .into(),
                ty: Type::Duration,
                constant: false,
            }
            .into(),
            right: Binary {
                op: BinaryOp::Div,
                left: Stretch {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "test".to_string(),
                }
                .into(),
                right: Value::Duration(Duration::dt(1000)).into(),
                ty: Type::Float,
                constant: true,
            }
            .into(),
            ty: Type::Bool,
            constant: false,
        }
        .into();

        let identifiers: Vec<IdentifierRef> = expr.identifiers().collect();
        assert!(matches!(
            identifiers.as_slice(),
            [
                IdentifierRef::Stretch(Stretch { .. }),
                IdentifierRef::Stretch(Stretch { .. }),
                IdentifierRef::Var(Var::Standalone { .. }),
            ]
        ));
    }

    #[test]
    fn test_iter() {
        let expr: Expr = Binary {
            op: BinaryOp::Mul,
            left: Binary {
                op: BinaryOp::Add,
                left: Var::Standalone {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "test".to_string(),
                    ty: Type::Duration,
                }
                .into(),
                right: Stretch {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "test".to_string(),
                }
                .into(),
                ty: Type::Duration,
                constant: false,
            }
            .into(),
            right: Binary {
                op: BinaryOp::Div,
                left: Stretch {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "test".to_string(),
                }
                .into(),
                right: Value::Duration(Duration::dt(1000)).into(),
                ty: Type::Float,
                constant: true,
            }
            .into(),
            ty: Type::Bool,
            constant: false,
        }
        .into();

        let exprs: Vec<ExprRef> = expr.iter().collect();
        assert!(matches!(
            exprs.as_slice(),
            [
                ExprRef::Binary(Binary {
                    op: BinaryOp::Mul,
                    ..
                }),
                ExprRef::Binary(Binary {
                    op: BinaryOp::Div,
                    ..
                }),
                ExprRef::Value(Value::Duration(..)),
                ExprRef::Stretch(Stretch { .. }),
                ExprRef::Binary(Binary {
                    op: BinaryOp::Add,
                    ..
                }),
                ExprRef::Stretch(Stretch { .. }),
                ExprRef::Var(Var::Standalone { .. }),
            ]
        ));
    }

    #[test]
    fn test_visit_mut_ordering() -> PyResult<()> {
        let mut expr: Expr = Binary {
            op: BinaryOp::Mul,
            left: Binary {
                op: BinaryOp::Add,
                left: Var::Standalone {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "test".to_string(),
                    ty: Type::Duration,
                }
                .into(),
                right: Stretch {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "test".to_string(),
                }
                .into(),
                ty: Type::Duration,
                constant: false,
            }
            .into(),
            right: Binary {
                op: BinaryOp::Div,
                left: Stretch {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "test".to_string(),
                }
                .into(),
                right: Value::Duration(Duration::dt(1000)).into(),
                ty: Type::Float,
                constant: true,
            }
            .into(),
            ty: Type::Bool,
            constant: false,
        }
        .into();

        // These get *consumed* by every visit, so by the end we expect this
        // iterator to be empty. The ordering here is post-order, LRN.
        let mut order = [
            |x: &ExprRefMut| matches!(x, ExprRefMut::Var(Var::Standalone { .. })),
            |x: &ExprRefMut| matches!(x, ExprRefMut::Stretch(Stretch { .. })),
            |x: &ExprRefMut| {
                matches!(
                    x,
                    ExprRefMut::Binary(Binary {
                        op: BinaryOp::Add,
                        ..
                    })
                )
            },
            |x: &ExprRefMut| matches!(x, ExprRefMut::Stretch(Stretch { .. })),
            |x: &ExprRefMut| matches!(x, ExprRefMut::Value(Value::Duration(..))),
            |x: &ExprRefMut| {
                matches!(
                    x,
                    ExprRefMut::Binary(Binary {
                        op: BinaryOp::Div,
                        ..
                    })
                )
            },
            |x: &ExprRefMut| {
                matches!(
                    x,
                    ExprRefMut::Binary(Binary {
                        op: BinaryOp::Mul,
                        ..
                    })
                )
            },
        ]
        .into_iter();

        expr.visit_mut(|x| {
            assert!(order.next().unwrap()(&x));
            Ok::<_, PyErr>(())
        })?;

        assert!(order.next().is_none());
        Ok(())
    }

    #[test]
    fn test_visit_mut() -> PyResult<()> {
        let mut expr: Expr = Binary {
            op: BinaryOp::BitAnd,
            left: Unary {
                op: UnaryOp::BitNot,
                operand: Var::Standalone {
                    uuid: Uuid::new_v4().as_u128(),
                    name: "test".to_string(),
                    ty: Type::Bool,
                }
                .into(),
                ty: Type::Bool,
                constant: false,
            }
            .into(),
            right: Value::Uint {
                raw: BigUint::from(1u8),
                ty: Type::Bool,
            }
            .into(),
            ty: Type::Bool,
            constant: false,
        }
        .into();

        expr.visit_mut(|x| match x {
            ExprRefMut::Var(Var::Standalone { name, .. }) => {
                *name = "updated".to_string();
                Ok::<_, PyErr>(())
            }
            _ => Ok(()),
        })?;

        let Var::Standalone { name, .. } = expr.vars().next().unwrap() else {
            panic!("wrong var type")
        };
        assert_eq!(name.as_str(), "updated");
        Ok(())
    }

    #[test]
    fn test_structurally_eq_to_self() -> PyResult<()> {
        let exprs = [
            Expr::Var(Var::Bit {
                bit: ShareableClbit::new_anonymous(),
            }),
            Expr::Var(Var::Register {
                register: ClassicalRegister::new_owning("a", 3),
                ty: Type::Uint(3),
            }),
            Expr::Value(Value::Uint {
                raw: BigUint::from(3u8),
                ty: Type::Uint(2),
            }),
            Expr::Cast(
                Cast {
                    operand: Expr::Var(Var::Register {
                        register: ClassicalRegister::new_owning("a", 3),
                        ty: Type::Uint(3),
                    }),
                    ty: Type::Bool,
                    constant: false,
                    implicit: false,
                }
                .into(),
            ),
            Expr::Unary(
                Unary {
                    op: UnaryOp::LogicNot,
                    operand: Expr::Var(Var::Bit {
                        bit: ShareableClbit::new_anonymous(),
                    }),
                    ty: Type::Bool,
                    constant: false,
                }
                .into(),
            ),
            Expr::Binary(
                Binary {
                    op: BinaryOp::BitAnd,
                    left: Expr::Value(Value::Uint {
                        raw: BigUint::from(5u8),
                        ty: Type::Uint(3),
                    }),
                    right: Expr::Var(Var::Register {
                        register: ClassicalRegister::new_owning("a", 3),
                        ty: Type::Uint(3),
                    }),
                    ty: Type::Uint(3),
                    constant: false,
                }
                .into(),
            ),
            Expr::Binary(
                Binary {
                    op: BinaryOp::LogicAnd,
                    left: Expr::Binary(
                        Binary {
                            op: BinaryOp::Less,
                            left: Expr::Value(Value::Uint {
                                raw: BigUint::from(2u8),
                                ty: Type::Uint(3),
                            }),
                            right: Expr::Var(Var::Register {
                                register: ClassicalRegister::new_owning("a", 3),
                                ty: Type::Uint(3),
                            }),
                            ty: Type::Bool,
                            constant: false,
                        }
                        .into(),
                    ),
                    right: Expr::Var(Var::Bit {
                        bit: ShareableClbit::new_anonymous(),
                    }),
                    ty: Type::Bool,
                    constant: false,
                }
                .into(),
            ),
            Expr::Binary(
                Binary {
                    op: BinaryOp::ShiftLeft,
                    left: Expr::Binary(
                        Binary {
                            op: BinaryOp::ShiftRight,
                            left: Expr::Value(Value::Uint {
                                raw: BigUint::from(255u8),
                                ty: Type::Uint(8),
                            }),
                            right: Expr::Value(Value::Uint {
                                raw: BigUint::from(3u8),
                                ty: Type::Uint(8),
                            }),
                            ty: Type::Uint(8),
                            constant: true,
                        }
                        .into(),
                    ),
                    right: Expr::Value(Value::Uint {
                        raw: BigUint::from(3u8),
                        ty: Type::Uint(8),
                    }),
                    ty: Type::Uint(8),
                    constant: true,
                }
                .into(),
            ),
            Expr::Index(
                Index {
                    target: Expr::Var(Var::Standalone {
                        uuid: Uuid::new_v4().as_u128(),
                        name: "a".to_string(),
                        ty: Type::Uint(8),
                    }),
                    index: Expr::Value(Value::Uint {
                        raw: BigUint::from(0u8),
                        ty: Type::Uint(8),
                    }),
                    ty: Type::Uint(1),
                    constant: false,
                }
                .into(),
            ),
            Expr::Binary(
                Binary {
                    op: BinaryOp::Greater,
                    left: Expr::Stretch(Stretch {
                        uuid: Uuid::new_v4().as_u128(),
                        name: "a".to_string(),
                    }),
                    right: Expr::Value(Value::Duration(Duration::dt(100))),
                    ty: Type::Bool,
                    constant: false,
                }
                .into(),
            ),
        ];

        for expr in exprs.iter() {
            assert!(expr.structurally_equivalent(expr));
        }

        Ok(())
    }

    #[test]
    fn test_structurally_eq_does_not_compare_symmetrically() {
        let all_ops = [
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::LogicAnd,
            BinaryOp::LogicOr,
            BinaryOp::Equal,
            BinaryOp::NotEqual,
            BinaryOp::Less,
            BinaryOp::LessEqual,
            BinaryOp::Greater,
            BinaryOp::GreaterEqual,
            BinaryOp::ShiftLeft,
            BinaryOp::ShiftRight,
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
        ];

        for op in all_ops.iter().copied() {
            let (left, right, out_ty) = match op {
                BinaryOp::LogicAnd | BinaryOp::LogicOr => (
                    Expr::Value(Value::Uint {
                        raw: BigUint::from(1u8),
                        ty: Type::Bool,
                    }),
                    Expr::Var(Var::Bit {
                        bit: ShareableClbit::new_anonymous(),
                    }),
                    Type::Bool,
                ),
                _ => (
                    Expr::Value(Value::Uint {
                        raw: BigUint::from(5u8),
                        ty: Type::Uint(3),
                    }),
                    Expr::Var(Var::Register {
                        register: ClassicalRegister::new_owning("a", 3),
                        ty: Type::Uint(3),
                    }),
                    match op {
                        BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => Type::Uint(3),
                        _ => Type::Bool,
                    },
                ),
            };

            let cis = Expr::Binary(
                Binary {
                    op,
                    left: left.clone(),
                    right: right.clone(),
                    ty: out_ty,
                    constant: false,
                }
                .into(),
            );

            let trans = Expr::Binary(
                Binary {
                    op,
                    left: right,
                    right: left,
                    ty: out_ty,
                    constant: false,
                }
                .into(),
            );

            assert!(
                !cis.structurally_equivalent(&trans),
                "Expected {op:?} to not be structurally equivalent to its flipped form"
            );
            assert!(
                !trans.structurally_equivalent(&cis),
                "Expected flipped {op:?} to not be structurally equivalent to original form"
            );
        }
    }

    #[test]
    fn test_structurally_eq_key_function_both() -> PyResult<()> {
        let left_clbit = ShareableClbit::new_anonymous();
        let left_cr = ClassicalRegister::new_owning("a", 3);
        let right_clbit = ShareableClbit::new_anonymous();
        let right_cr = ClassicalRegister::new_owning("b", 3);
        assert_ne!(left_clbit, right_clbit);
        assert_ne!(left_cr, right_cr);

        let left = Expr::Unary(
            Unary {
                op: UnaryOp::LogicNot,
                operand: Expr::Binary(
                    Binary {
                        op: BinaryOp::LogicAnd,
                        left: Expr::Binary(
                            Binary {
                                op: BinaryOp::Less,
                                left: Expr::Value(Value::Uint {
                                    raw: BigUint::from(5u8),
                                    ty: Type::Uint(3),
                                }),
                                right: Expr::Var(Var::Register {
                                    register: left_cr.clone(),
                                    ty: Type::Uint(3),
                                }),
                                ty: Type::Bool,
                                constant: false,
                            }
                            .into(),
                        ),
                        right: Expr::Var(Var::Bit {
                            bit: left_clbit.clone(),
                        }),
                        ty: Type::Bool,
                        constant: false,
                    }
                    .into(),
                ),
                ty: Type::Bool,
                constant: false,
            }
            .into(),
        );

        let right = Expr::Unary(
            Unary {
                op: UnaryOp::LogicNot,
                operand: Expr::Binary(
                    Binary {
                        op: BinaryOp::LogicAnd,
                        left: Expr::Binary(
                            Binary {
                                op: BinaryOp::Less,
                                left: Expr::Value(Value::Uint {
                                    raw: BigUint::from(5u8),
                                    ty: Type::Uint(3),
                                }),
                                right: Expr::Var(Var::Register {
                                    register: right_cr.clone(),
                                    ty: Type::Uint(3),
                                }),
                                ty: Type::Bool,
                                constant: false,
                            }
                            .into(),
                        ),
                        right: Expr::Var(Var::Bit {
                            bit: right_clbit.clone(),
                        }),
                        ty: Type::Bool,
                        constant: false,
                    }
                    .into(),
                ),
                ty: Type::Bool,
                constant: false,
            }
            .into(),
        );

        assert!(
            !left.structurally_equivalent(&right),
            "Expressions using different registers/clbits should not be structurally equivalent"
        );

        // We're happy as long as the variables are of the same kind.
        let key_func = |v: &Var| -> &str {
            match v {
                Var::Standalone { .. } => "standalone",
                Var::Bit { .. } => "bit",
                Var::Register { .. } => "register",
            }
        };

        assert!(
            left.structurally_equivalent_by_key(key_func, &right, key_func),
            "Expressions that only care that their vars are of the same kind should be equivalent"
        );

        Ok(())
    }
}
