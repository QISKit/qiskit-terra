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

use std::ptr;

use crate::pointers::const_ptr_as_ref;
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use qiskit_circuit::{
    bit::{ClassicalRegister, ShareableClbit},
    classical::{
        expr::{Binary, BinaryOp, Cast, Expr, Index, Stretch, Unary, UnaryOp, Value, Var},
        types::Type,
    },
    duration::Duration,
};
use std::ffi::{CString, c_char};
use uuid::Uuid;

/// The different types of expression nodes that can appear in a classical expression tree.
#[repr(u8)]
pub enum CExprNodeKind {
    /// Unary operation expression (e.g., NOT, negation)
    Unary = 0,
    /// Binary operation expression (e.g., AND, OR, arithmetic operations)
    Binary = 1,
    /// Type cast expression
    Cast = 2,
    /// Literal value expression
    Value = 3,
    /// Variable reference expression
    Var = 4,
    /// Stretch expression (timing-related)
    Stretch = 5,
    /// Index/subscript expression
    Index = 6,
}

impl From<&Expr> for CExprNodeKind {
    fn from(value: &Expr) -> Self {
        match value {
            Expr::Unary(_) => Self::Unary,
            Expr::Binary(_) => Self::Binary,
            Expr::Cast(_) => Self::Cast,
            Expr::Index(_) => Self::Index,
            Expr::Stretch(_) => Self::Stretch,
            Expr::Value(_) => Self::Value,
            Expr::Var(_) => Self::Var,
        }
    }
}

/// The data types that can be used in classical expressions.
#[repr(u8)]
#[derive(Copy, Clone)]
pub enum CExprType {
    /// Boolean type
    Bool = 0,
    /// Duration type
    Duration = 1,
    /// Floating-point type
    Float = 2,
    /// Unsigned integer type
    Uint = 3,
}

impl From<&Type> for CExprType {
    fn from(value: &Type) -> Self {
        match value {
            Type::Bool => Self::Bool,
            Type::Duration => Self::Duration,
            Type::Float => Self::Float,
            Type::Uint(_) => Self::Uint,
        }
    }
}

impl From<&Value> for CExprType {
    fn from(value: &Value) -> Self {
        match value {
            Value::Duration(_) => Self::Duration,
            Value::Float { ty, .. } => Self::from(ty),
            Value::Uint { ty, .. } => Self::from(ty),
        }
    }
}

/// The complete representation of the data type used for an expression.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CExprTypeInfo {
    /// The expression type
    ty: CExprType,
    /// Bit width for the Uint expression type
    width: u32,
}

impl From<CExprTypeInfo> for Type {
    fn from(value: CExprTypeInfo) -> Type {
        match value.ty {
            CExprType::Bool => Type::Bool,
            CExprType::Duration => Type::Duration,
            CExprType::Float => Type::Float,
            CExprType::Uint => Type::Uint(value.width),
        }
    }
}

impl From<&Type> for CExprTypeInfo {
    fn from(ty: &Type) -> Self {
        match ty {
            Type::Bool => CExprTypeInfo {
                ty: CExprType::Bool,
                width: 0,
            },
            Type::Duration => CExprTypeInfo {
                ty: CExprType::Duration,
                width: 0,
            },
            Type::Float => CExprTypeInfo {
                ty: CExprType::Float,
                width: 0,
            },
            Type::Uint(w) => CExprTypeInfo {
                ty: CExprType::Uint,
                width: *w,
            },
        }
    }
}

/// The operation types a unary expression can hold.
/// Values are one-based to match the Python convention of the classical expression system.
#[repr(u8)]
pub enum CUnaryOpType {
    /// Bitwise NOT operation
    BitNot = 1,
    /// Logical NOT operation
    LogicNot = 2,
    /// Arithmetic negation
    Negate = 3,
}

impl From<UnaryOp> for CUnaryOpType {
    fn from(value: UnaryOp) -> Self {
        match value {
            UnaryOp::BitNot => Self::BitNot,
            UnaryOp::LogicNot => Self::LogicNot,
            UnaryOp::Negate => Self::Negate,
        }
    }
}

impl From<CUnaryOpType> for UnaryOp {
    fn from(value: CUnaryOpType) -> Self {
        match value {
            CUnaryOpType::BitNot => UnaryOp::BitNot,
            CUnaryOpType::LogicNot => UnaryOp::LogicNot,
            CUnaryOpType::Negate => UnaryOp::Negate,
        }
    }
}

/// Describes a unary expression, including its operator, operand, result type and whether it is constant.
/// Returned by the `qk_expr_unary_info` function. The `operand` field is a borrowed pointer to the operand
/// of the unary operation expression.
#[repr(C)]
pub struct CUnaryExprInfo {
    /// The unary operator
    op: CUnaryOpType,
    /// Borrowed pointer to the operand expression
    operand: *const Expr,
    /// Result type of the operation
    ty: CExprTypeInfo,
    /// Whether the expression is constant
    constant: bool,
}

/// The operation types a binary expression can hold.
/// Values are one-based to match Python convention.
#[repr(u8)]
#[derive(Copy, Clone)]
pub enum CBinaryOpType {
    /// Bitwise AND operation
    BitAnd = 1,
    /// Bitwise OR operation
    BitOr = 2,
    /// Bitwise XOR operation
    BitXor = 3,
    /// Logical AND operation
    LogicAnd = 4,
    /// Logical OR operation
    LogicOr = 5,
    /// Equality comparison
    Equal = 6,
    /// Inequality comparison
    NotEqual = 7,
    /// Less than comparison
    Less = 8,
    /// Less than or equal comparison
    LessEqual = 9,
    /// Greater than comparison
    Greater = 10,
    /// Greater than or equal comparison
    GreaterEqual = 11,
    /// Left shift operation
    ShiftLeft = 12,
    /// Right shift operation
    ShiftRight = 13,
    /// Addition operation
    Add = 14,
    /// Subtraction operation
    Sub = 15,
    /// Multiplication operation
    Mul = 16,
    /// Division operation
    Div = 17,
}

impl From<BinaryOp> for CBinaryOpType {
    fn from(value: BinaryOp) -> Self {
        match value {
            BinaryOp::BitAnd => Self::BitAnd,
            BinaryOp::BitOr => Self::BitOr,
            BinaryOp::BitXor => Self::BitXor,
            BinaryOp::LogicAnd => Self::LogicAnd,
            BinaryOp::LogicOr => Self::LogicOr,
            BinaryOp::Equal => Self::Equal,
            BinaryOp::NotEqual => Self::NotEqual,
            BinaryOp::Less => Self::Less,
            BinaryOp::LessEqual => Self::LessEqual,
            BinaryOp::Greater => Self::Greater,
            BinaryOp::GreaterEqual => Self::GreaterEqual,
            BinaryOp::ShiftLeft => Self::ShiftLeft,
            BinaryOp::ShiftRight => Self::ShiftRight,
            BinaryOp::Add => Self::Add,
            BinaryOp::Sub => Self::Sub,
            BinaryOp::Mul => Self::Mul,
            BinaryOp::Div => Self::Div,
        }
    }
}

impl From<CBinaryOpType> for BinaryOp {
    fn from(value: CBinaryOpType) -> Self {
        match value {
            CBinaryOpType::BitAnd => BinaryOp::BitAnd,
            CBinaryOpType::BitOr => BinaryOp::BitOr,
            CBinaryOpType::BitXor => BinaryOp::BitXor,
            CBinaryOpType::LogicAnd => BinaryOp::LogicAnd,
            CBinaryOpType::LogicOr => BinaryOp::LogicOr,
            CBinaryOpType::Equal => BinaryOp::Equal,
            CBinaryOpType::NotEqual => BinaryOp::NotEqual,
            CBinaryOpType::Less => BinaryOp::Less,
            CBinaryOpType::LessEqual => BinaryOp::LessEqual,
            CBinaryOpType::Greater => BinaryOp::Greater,
            CBinaryOpType::GreaterEqual => BinaryOp::GreaterEqual,
            CBinaryOpType::ShiftLeft => BinaryOp::ShiftLeft,
            CBinaryOpType::ShiftRight => BinaryOp::ShiftRight,
            CBinaryOpType::Add => BinaryOp::Add,
            CBinaryOpType::Sub => BinaryOp::Sub,
            CBinaryOpType::Mul => BinaryOp::Mul,
            CBinaryOpType::Div => BinaryOp::Div,
        }
    }
}

/// Describes a binary expression, including its operator, operands, result type and whether it is constant.
/// Returned by the `qk_expr_binary_info` function. The `left` and `right` fields are borrowed pointers to
/// the operands of the binary operation expression.
#[repr(C)]
pub struct CBinaryExprInfo {
    /// The binary operator
    op: CBinaryOpType,
    /// Borrowed pointer to the left operand expression
    left: *const Expr,
    /// Borrowed pointer to the right operand expression
    right: *const Expr,
    /// Result type of the operation
    ty: CExprTypeInfo,
    /// Whether the expression is constant
    constant: bool,
}

/// Describes a cast expression, including the operand, target type, whether it is implicit, and whether it is constant.
/// Returned by the `qk_expr_cast_info` function. The `operand` field is a borrowed pointer to the operand
/// of the cast expression.
#[repr(C)]
pub struct CCastExprInfo {
    /// Borrowed pointer to the operand expression being cast
    operand: *const Expr,
    /// Target type of the cast
    ty: CExprTypeInfo,
    /// Whether the cast is implicit (automatic) or explicit
    implicit: bool,
    /// Whether the expression is constant
    constant: bool,
}

/// Describes an index expression, including the target, index, result type and whether it is constant.
/// Returned by the `qk_expr_index_info` function. The `target` and `index` fields are borrowed pointers
/// to the target and index of the index operation expression.
#[repr(C)]
pub struct CIndexExprInfo {
    /// Borrowed pointer to the target expression being indexed
    target: *const Expr,
    /// Borrowed pointer to the index expression
    index: *const Expr,
    /// Result type of the indexing operation
    ty: CExprTypeInfo,
    /// Whether the expression is constant
    constant: bool,
}

/// Represents different time units used in duration expressions.
#[repr(u8)]
#[derive(Copy, Clone)]
pub enum CDurationType {
    /// System time units
    Dt = 0,
    /// Picoseconds
    Ps = 1,
    /// Nanoseconds
    Ns = 2,
    /// Microseconds
    Us = 3,
    /// Milliseconds
    Ms = 4,
    /// Seconds
    S = 5,
}

impl From<&Duration> for CDurationType {
    fn from(duration: &Duration) -> Self {
        match duration {
            Duration::dt(_) => Self::Dt,
            Duration::ps(_) => Self::Ps,
            Duration::ns(_) => Self::Ns,
            Duration::us(_) => Self::Us,
            Duration::ms(_) => Self::Ms,
            Duration::s(_) => Self::S,
        }
    }
}

/// A union to hold either system time units (dt) as an integer or real time as a float.
///
/// This union is part of the `QkDurationInfo` struct and should not be used directly.
#[repr(C)]
#[derive(Copy, Clone)]
pub union CDurationValue {
    /// System time units (active when `ty` in `QkDurationInfo` is `QkDurationType_Dt`)
    dt: i64,
    /// Real time value (active for all other duration types)
    time: f64,
}

/// The complete representation of a duration value.
///
/// The `ty` field acts as a discriminant that determines which field of the `value` union is active.
///
/// When initialized from C, it is the user's responsibility to ensure that:
/// - When `ty` is `QkDurationType_Dt`, the duration is stored as an integer in `value.dt`.
/// - For all other duration types the duration is stored as a floating-point value in `value.time`.
///
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CDurationInfo {
    /// The duration unit type (discriminant for the union)
    ty: CDurationType,
    /// The duration value
    value: CDurationValue,
}

impl From<&Duration> for CDurationInfo {
    fn from(duration: &Duration) -> Self {
        match duration {
            Duration::dt(v) => CDurationInfo {
                ty: CDurationType::Dt,
                value: CDurationValue { dt: *v },
            },
            Duration::ps(v) => CDurationInfo {
                ty: CDurationType::Ps,
                value: CDurationValue { time: *v },
            },
            Duration::ns(v) => CDurationInfo {
                ty: CDurationType::Ns,
                value: CDurationValue { time: *v },
            },
            Duration::us(v) => CDurationInfo {
                ty: CDurationType::Us,
                value: CDurationValue { time: *v },
            },
            Duration::ms(v) => CDurationInfo {
                ty: CDurationType::Ms,
                value: CDurationValue { time: *v },
            },
            Duration::s(v) => CDurationInfo {
                ty: CDurationType::S,
                value: CDurationValue { time: *v },
            },
        }
    }
}

impl From<CDurationInfo> for Duration {
    fn from(value: CDurationInfo) -> Self {
        match value.ty {
            // SAFETY: Per documentation, ty must correctly discriminate the union.
            CDurationType::Dt => Duration::dt(unsafe { value.value.dt }),
            CDurationType::Ps => Duration::ps(unsafe { value.value.time }),
            CDurationType::Ns => Duration::ns(unsafe { value.value.time }),
            CDurationType::Us => Duration::us(unsafe { value.value.time }),
            CDurationType::Ms => Duration::ms(unsafe { value.value.time }),
            CDurationType::S => Duration::s(unsafe { value.value.time }),
        }
    }
}

/// @ingroup QkClassicalExpressions
/// Return the kind of a classical expression node.
///
/// @param expr A pointer to the expression node to inspect.
///
/// @return The kind enum describing which concrete expression variant ``expr`` contains.
///
/// # Example
/// ```c
/// QKExprNodeKind kind = qk_expr_kind(expr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``expr`` is not a valid, non-null pointer to a ``QkExprNode``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_kind(expr: *const Expr) -> CExprNodeKind {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let expr = unsafe { const_ptr_as_ref(expr) };

    CExprNodeKind::from(expr)
}

/// @ingroup QkClassicalExpressions
///
/// Extract information from a binary expression node.
///
/// @param expr A pointer to a binary expression node.
///
/// @return A `QkBinaryExprInfo` struct describing the operator, operands,
/// result type, and whether the expression is constant.
///
/// Panics if ``expr`` does not point to a binary expression node.
///
/// # Example
/// ```c
/// QkBinaryExprInfo info = qk_expr_binary_info(expr);
/// const QkExprNode *lhs = info.left;
/// const QkExprNode *rhs = info.right;
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``expr`` is not a valid, non-null pointer to a ``QkExprNode``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_binary_info(expr: *const Expr) -> CBinaryExprInfo {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Binary(binary) = expr else {
        panic!("qk_expr_binary_info called on non-binary expression")
    };

    CBinaryExprInfo {
        op: CBinaryOpType::from(binary.op),
        left: ptr::from_ref(&binary.left),
        right: ptr::from_ref(&binary.right),
        ty: CExprTypeInfo::from(&binary.ty),
        constant: binary.constant,
    }
}

/// @ingroup QkClassicalExpressions
/// Extract information from a unary expression node.
///
/// @param expr A pointer to a unary expression node.
///
/// @return A `QkUnaryExprInfo` struct describing the operator, operand,
/// result type, and whether the expression is constant.
///
/// Panics if ``expr`` does not point to a unary expression node.
///
/// # Example
/// ```c
/// QkUnaryExprInfo info = qk_expr_unary_info(expr);
/// QkUnaryOpType op = info.op;
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``expr`` is not a valid, non-null pointer to a ``QkExprNode``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_unary_info(expr: *const Expr) -> CUnaryExprInfo {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Unary(unary) = expr else {
        panic!("qk_expr_unary_info called on non-unary expression")
    };

    CUnaryExprInfo {
        op: CUnaryOpType::from(unary.op),
        operand: ptr::from_ref(&unary.operand),
        ty: CExprTypeInfo::from(&unary.ty),
        constant: unary.constant,
    }
}

/// @ingroup QkClassicalExpressions
/// Extract information from a cast expression node.
///
/// @param expr A pointer to a cast expression node.
///
/// @return A `QkCastExprInfo` struct describing the operand, destination
/// type, whether the cast is implicit, and whether the expression is constant.
///
/// Panics if ``expr`` does not point to a cast expression node.
///
/// # Example
/// ```c
/// QkCastExprInfo info = qk_expr_cast_info(expr);
/// const QkExprNode *operand = info.operand;
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``expr`` is not a valid, non-null pointer to a ``QkExprNode``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_cast_info(expr: *const Expr) -> CCastExprInfo {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Cast(cast) = expr else {
        panic!("qk_expr_cast_info called on non-cast expression")
    };

    CCastExprInfo {
        operand: ptr::from_ref(&cast.operand),
        ty: CExprTypeInfo::from(&cast.ty),
        implicit: cast.implicit,
        constant: cast.constant,
    }
}

/// @ingroup QkClassicalExpressions
/// Extract information from an index expression node.
///
/// @param expr A pointer to an index expression node.
///
/// @return A `QkIndexExprInfo` struct describing the indexed target,
/// index expression, result type, and whether the expression is constant.
///
/// Panics if ``expr`` does not point to an index expression node.
///
/// # Example
/// ```c
/// QkIndexExprInfo info = qk_expr_index_info(expr);
/// const QkExprNode *target = info.target;
/// const QkExprNode *index = info.index;
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``expr`` is not a valid, non-null pointer to a ``QkExprNode``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_index_info(expr: *const Expr) -> CIndexExprInfo {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Index(index) = expr else {
        panic!("qk_expr_index_info called on non-index expression")
    };

    CIndexExprInfo {
        target: ptr::from_ref(&index.target),
        index: ptr::from_ref(&index.index),
        ty: CExprTypeInfo::from(&index.ty),
        constant: index.constant,
    }
}

/// @ingroup QkClassicalExpressions
/// Return a view into the underlying value of the expression node.
///
/// @param expr A pointer to a value expression node.
///
/// @return A pointer to the ``QkValue`` stored inside ``expr``.
///
/// Panics if ``expr`` does not point to a value expression node.
///
/// # Example
/// ```c
/// const QkValue *value = qk_expr_as_value(expr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``expr`` is not a valid, non-null pointer to a ``QkExprNode``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_as_value(expr: *const Expr) -> *const Value {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Value(val) = expr else {
        panic!("qk_expr_as_value called on non-value expression")
    };

    ptr::from_ref(val)
}

/// @ingroup QkClassicalExpressions
/// Return a view into the underlying variable of the expression node.
///
/// @param expr A pointer to a variable expression node.
///
/// @return A pointer to the ``QkVar`` stored inside ``expr``.
///
/// Panics if ``expr`` does not point to a variable expression node.
///
/// # Example
/// ```c
/// const QkVar *var = qk_expr_as_var(expr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``expr`` is not a valid, non-null pointer to a ``QkExprNode``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_as_var(expr: *const Expr) -> *const Var {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Var(var) = expr else {
        panic!("qk_expr_as_var called on non-variable expression")
    };

    ptr::from_ref(var)
}

/// @ingroup QkClassicalExpressions
/// Return a view into the underlying stretch of the expression node.
///
/// @param expr A pointer to a stretch expression node.
///
/// @return A pointer to the ``QkStretch`` stored inside ``expr``.
///
/// Panics if ``expr`` does not point to a stretch expression node.
///
/// # Example
/// ```c
/// const QkStretch *stretch = qk_expr_as_stretch(expr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``expr`` is not a valid, non-null pointer to a ``QkExprNode``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_expr_as_stretch(expr: *const Expr) -> *const Stretch {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let expr = unsafe { const_ptr_as_ref(expr) };

    let Expr::Stretch(stretch) = expr else {
        panic!("qk_expr_as_stretch called on non-stretch expression")
    };

    ptr::from_ref(stretch)
}

/// @ingroup QkClassicalExpressions
/// Return the type information of a value.
///
/// @param value A pointer to the `QkValue` to inspect.
///
/// @return A `QkExprTypeInfo` struct containing the value type information.
///
/// # Example
/// ```c
/// QkExprTypeInfo type_info = qk_value_type_info(value);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``value`` is not a valid, non-null pointer to a ``Value``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_value_type_info(value: *const Value) -> CExprTypeInfo {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let value = unsafe { const_ptr_as_ref(value) };

    match value {
        Value::Duration(_) => CExprTypeInfo {
            ty: CExprType::Duration,
            width: 0,
        },
        Value::Float { ty, .. } | Value::Uint { ty, .. } => CExprTypeInfo::from(ty),
    }
}

/// @ingroup QkClassicalExpressions
/// Extract structured information from a duration value.
///
/// @param value A pointer to a duration value.
///
/// @return A `QkDurationInfo` struct containing the duration unit and raw value.
///
/// Panics if ``value`` does not point to a duration value.
///
/// # Example
/// ```c
/// QkDurationInfo info = qk_value_duration_info(value);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``value`` is not a valid, non-null pointer to a ``QkValue``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_value_duration_info(value: *const Value) -> CDurationInfo {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let value = unsafe { const_ptr_as_ref(value) };

    let Value::Duration(duration) = value else {
        panic!("qk_value_duration_info called on non-duration value")
    };

    CDurationInfo::from(duration)
}

/// @ingroup QkClassicalExpressions
/// Extract the floating-point value from a ``QkValue``.
///
/// @param value A pointer to a float value.
///
/// @return The ``double`` value stored in ``value``.
///
/// Panics if ``value`` does not point to a float value.
///
/// # Example
/// ```c
/// double raw = qk_value_float(value);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``value`` is not a valid, non-null pointer to a ``QkValue``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_value_float(value: *const Value) -> f64 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let value = unsafe { const_ptr_as_ref(value) };

    let Value::Float {
        raw,
        ty: Type::Float,
    } = value
    else {
        panic!("qk_value_float called on non-float value")
    };

    *raw
}

/// @ingroup QkClassicalExpressions
/// Extract the unsigned integer value from a ``QkValue`` of type ``QkExprType_Uint``.
///
/// you should use `qk_value_type_info` to ensure the value will fit in ``uint64_t``
/// before calling this function.
///
/// @param value A pointer to a uint value.
///
/// @return The integer value converted to ``uint64_t``.
///
/// Panics if ``value`` does not point to a ``QkExprType_Uint`` value or if the
/// stored integer does not fit in ``uint64_t``.
///
/// # Example
/// ```c
/// uint64_t raw = qk_value_uint(value);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``value`` is not a valid, non-null pointer to a ``QkValue``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_value_uint(value: *const Value) -> u64 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let value = unsafe { const_ptr_as_ref(value) };

    let Value::Uint {
        raw,
        ty: Type::Uint(_),
    } = value
    else {
        panic!("qk_value_uint called on non-uint value")
    };

    raw.to_u64()
        .expect("Integer value too large to fit in uint64_t")
}

/// @ingroup QkClassicalExpressions
/// Extract the value from a ``QkValue`` of type ``QkExprType_Bool``.
///
/// @param value A pointer to a bool value.
///
/// @return ``true`` if the stored integer representation is nonzero, otherwise ``false``.
///
/// Panics if ``value`` does not point to a bool value.
///
/// # Example
/// ```c
/// bool raw = qk_value_bool(value);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``value`` is not a valid, non-null pointer to a ``QkValue``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_value_bool(value: *const Value) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let value = unsafe { const_ptr_as_ref(value) };

    let Value::Uint {
        raw,
        ty: Type::Bool,
    } = value
    else {
        panic!("qk_value_bool called on non-bool value")
    };

    !raw.is_zero()
}

/// @ingroup QkClassicalExpressions
/// Return the name of a variable as a newly allocated C string.
///
/// @param var A pointer to the variable to inspect.
///
/// @return A null-terminated string containing the variable name, or ``NULL``
/// if ``var`` refers to a non-standalone variable (i.e. based on a classical bit or a classical register).
/// The caller owns the returned string and must free it with `qk_str_free`.
///
/// # Example
/// ```c
/// char *name = qk_var_name(var);
/// if (name != NULL) {
///     // Use the name...
///     qk_str_free(name);
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``var`` is not a valid, non-null pointer to a ``QkVar``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_var_name(var: *const Var) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let var = unsafe { const_ptr_as_ref(var) };

    let name = match var {
        Var::Standalone { name, .. } => name.as_str(),
        Var::Register { .. } | Var::Bit { .. } => return ptr::null_mut(),
    };

    CString::new(name)
        .expect("Var should have a valid name")
        .into_raw()
}

/// @ingroup QkClassicalExpressions
/// Return full type information for a variable.
///
/// @param var A pointer to the variable to inspect.
///
/// @return A `QkExprTypeInfo` struct containing the variable type information.
///
/// Panics if ``var`` is a bit variable, which is not yet supported by this API.
///
/// # Example
/// ```c
/// QkExprTypeInfo type_info = qk_var_type_info(var);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``var`` is not a valid, non-null pointer to a ``QkVar``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_var_type_info(var: *const Var) -> CExprTypeInfo {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let var = unsafe { const_ptr_as_ref(var) };

    let ty = match var {
        Var::Standalone { ty, .. } => ty,
        Var::Register { ty, .. } => ty,
        Var::Bit { .. } => &Type::Bool,
    };

    CExprTypeInfo {
        ty: CExprType::from(ty),
        width: match ty {
            Type::Uint(width) => *width,
            _ => 0,
        },
    }
}

/// @ingroup QkClassicalExpressions
/// Return the name of a stretch.
///
/// @param stretch A pointer to the stretch to inspect.
///
/// @return A null-terminated string containing the stretch name. The caller owns
/// the returned string and must free it with `qk_str_free`.
///
/// # Example
/// ```c
/// char *name = qk_stretch_name(stretch);
/// // Use the name...
/// qk_str_free(name);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``stretch`` is not a valid, non-null pointer to a ``QkStretch``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_stretch_name(stretch: *const Stretch) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let stretch = unsafe { const_ptr_as_ref(stretch) };

    CString::new(stretch.name.as_str())
        .expect("Stretch should have a valid name")
        .into_raw()
}

//////////////////////////////////////////////////////////////////
// The functions below are used in the C testing, to generate   //
// various objects for testing the C API. These functions       //
// should be removed once we have the actual C API for creating //
// classical expression constructs.                             //
//////////////////////////////////////////////////////////////////

/// cbindgen:qk-vtable-rules=[no-export]
/// cbindgen:no-export
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inner_test_expression_structs() -> *mut Expr {
    let v1 = Expr::Var(Var::Standalone {
        uuid: Uuid::new_v4().as_u128(),
        name: "v1".to_owned(),
        ty: Type::Uint(3),
    });

    let five = Expr::Value(Value::Uint {
        raw: BigUint::from(5u32),
        ty: Type::Uint(3),
    });

    let gt = Expr::Binary(Box::new(Binary {
        op: BinaryOp::Greater,
        left: v1.clone(),
        right: five,
        ty: Type::Bool,
        constant: false,
    }));

    let lt_eq = Expr::Unary(Box::new(Unary {
        op: UnaryOp::LogicNot,
        operand: gt,
        ty: Type::Bool,
        constant: false,
    }));

    let idx = Expr::Cast(Box::new(Cast {
        operand: lt_eq,
        ty: Type::Uint(1),
        implicit: false,
        constant: false,
    }));

    let index = Expr::Index(Box::new(Index {
        target: v1,
        index: idx,
        ty: Type::Bool,
        constant: false,
    }));

    Box::into_raw(Box::new(index))
}

/// cbindgen:qk-vtable-rules=[no-export]
/// cbindgen:no-export
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inner_test_binary_expr_ops(op: CBinaryOpType) -> *mut Expr {
    let zero = Expr::Value(Value::Float {
        raw: 0.0,
        ty: Type::Float,
    });

    let expr = Expr::Binary(Box::new(Binary {
        op: op.into(),
        left: zero.clone(),
        right: zero.clone(),
        ty: Type::Float,
        constant: true,
    }));

    Box::into_raw(Box::new(expr))
}

/// cbindgen:qk-vtable-rules=[no-export]
/// cbindgen:no-export
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inner_test_unary_expr_ops(op: CUnaryOpType) -> *mut Expr {
    let zero = Expr::Value(Value::Float {
        raw: 0.0,
        ty: Type::Float,
    });

    let expr = Expr::Unary(Box::new(Unary {
        op: op.into(),
        operand: zero,
        ty: Type::Float,
        constant: true,
    }));

    Box::into_raw(Box::new(expr))
}

/// cbindgen:qk-vtable-rules=[no-export]
/// cbindgen:no-export
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inner_test_expr_kinds_and_types(
    kind: CExprNodeKind,
    ty: CExprTypeInfo,
) -> *mut Expr {
    let rust_type = ty.into();

    let dummy_value = match &rust_type {
        Type::Bool => Expr::Value(Value::Uint {
            raw: BigUint::from(0u32),
            ty: Type::Bool,
        }),
        Type::Duration => Expr::Value(Value::Duration(Duration::dt(0))),
        Type::Float => Expr::Value(Value::Float {
            raw: 0.0,
            ty: Type::Float,
        }),
        Type::Uint(w) => Expr::Value(Value::Uint {
            raw: BigUint::from(0u32),
            ty: Type::Uint(*w),
        }),
    };

    let var = Expr::Var(Var::Standalone {
        uuid: Uuid::new_v4().as_u128(),
        name: "test_var".to_owned(),
        ty: ty.into(),
    });

    let index = Expr::Value(Value::Uint {
        raw: BigUint::from(0u32),
        ty: Type::Uint(3),
    });

    let expr = match kind {
        CExprNodeKind::Unary => Expr::Unary(Box::new(Unary {
            op: UnaryOp::BitNot,
            operand: dummy_value,
            ty: rust_type,
            constant: true,
        })),
        CExprNodeKind::Binary => Expr::Binary(Box::new(Binary {
            op: BinaryOp::BitAnd,
            left: dummy_value.clone(),
            right: dummy_value,
            ty: rust_type,
            constant: true,
        })),
        CExprNodeKind::Cast => Expr::Cast(Box::new(Cast {
            operand: dummy_value,
            ty: rust_type,
            implicit: false,
            constant: true,
        })),
        CExprNodeKind::Index => Expr::Index(Box::new(Index {
            target: var,
            index,
            ty: rust_type,
            constant: false,
        })),
        CExprNodeKind::Value => dummy_value,
        CExprNodeKind::Var => var,
        CExprNodeKind::Stretch => Expr::Stretch(Stretch {
            uuid: Uuid::new_v4().as_u128(),
            name: "test_stretch".to_owned(),
        }),
    };

    Box::into_raw(Box::new(expr))
}

/// cbindgen:qk-vtable-rules=[no-export]
/// cbindgen:no-export
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inner_test_value(
    ty: CExprType,
    b: bool,
    duration: CDurationInfo,
    f_val: f64,
    u_val: u64,
) -> *mut Expr {
    let value = match ty {
        CExprType::Bool => Value::Uint {
            raw: BigUint::from(if b { 1u32 } else { 0u32 }),
            ty: Type::Bool,
        },
        CExprType::Duration => Value::Duration(duration.into()),
        CExprType::Float => Value::Float {
            raw: f_val,
            ty: Type::Float,
        },
        CExprType::Uint => Value::Uint {
            raw: BigUint::from(u_val),
            ty: Type::Uint(4),
        },
    };

    Box::into_raw(Box::new(Expr::Value(value)))
}

/// cbindgen:qk-vtable-rules=[no-export]
/// cbindgen:no-export
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inned_test_old_style_vars(out_vars: *mut *mut Expr) {
    let bit_var = Var::Bit {
        bit: ShareableClbit::new_anonymous(),
    };
    let reg_var = Var::Register {
        register: ClassicalRegister::new_owning("c1", 2),
        ty: Type::Uint(2),
    };

    unsafe {
        out_vars.write(Box::into_raw(Box::new(Expr::Var(bit_var))));
        out_vars
            .add(1)
            .write(Box::into_raw(Box::new(Expr::Var(reg_var))));
    }
}

/// cbindgen:qk-vtable-rules=[no-export]
/// cbindgen:no-export
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inner_expr_free(expr: *mut Expr) {
    drop(unsafe { Box::from_raw(expr) })
}
