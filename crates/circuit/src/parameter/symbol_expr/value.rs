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

use std::cmp::{Ordering, PartialOrd};
use std::convert::From;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use num_complex::Complex64;
use pyo3::prelude::*;

use super::{_div, _mul, _neg, BinaryOp, SYMEXPR_EPSILON, SymbolExpr, UnaryOp};

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

    pub(super) fn mul_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
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

    pub(super) fn div_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
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
