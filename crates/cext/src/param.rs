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

use num_complex::Complex64;
use std::ffi::{CStr, CString, c_char};
use std::sync::Arc;

use crate::exit_codes::ExitCode;
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

use qiskit_circuit::operations::Param;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::{Symbol, SymbolExpr, Value};

/// @ingroup QkParam
/// Construct a new ``QkParam`` representing an unbound symbol.
///
/// @param name The name of symbol. This cannot be empty.
///
/// @return A pointer to the created ``QkParam``.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// ```
///
/// # Safety
///
/// The `name` parameter must be a pointer to memory that contains a valid
/// nul terminator at the end of the string. It also must be valid for reads of
/// bytes up to and including the nul terminator.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_new_symbol(name: *const c_char) -> *mut Param {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let name = unsafe { CStr::from_ptr(name).to_str().unwrap() };
    if name.is_empty() {
        // Per documentation, the name cannot be empty.
        panic!("Invalid empty name.");
    } else {
        let symbol = Symbol::standalone(name.to_owned(), None);
        let expr = ParameterExpression::from_symbol(symbol);
        let param = Param::ParameterExpression(Arc::new(expr));
        Box::into_raw(Box::new(param))
    }
}

/// @ingroup QkParam
/// Construct a new ``QkParam`` with a value zero.
///
/// The ``QkParam`` returned from this function can be used
/// to store the result of binary or unary operations.
///
/// @return A pointer to the created ``QkParam``.
///
/// # Example
///
/// ```c
/// QkParam *t = qk_param_zero();
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *b = qk_param_new_symbol("b");
/// qk_param_add(t, a, b);
/// ```
///
#[unsafe(no_mangle)]
pub extern "C" fn qk_param_zero() -> *mut Param {
    Box::into_raw(Box::new(Param::Float(0.)))
}

/// @ingroup QkParam
/// Free the ``QkParam``.
///
/// @param param A pointer to the ``QkParam`` to free.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// qk_param_free(a);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``param`` is not either null or a valid pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_free(param: *mut Param) {
    if !param.is_null() {
        if !param.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(param);
        }
    }
}

/// @ingroup QkParam
/// Construct a new ``QkParam`` from a ``double``.
///
/// @param value A ``double`` to initialize the ``QkParam``.
///
/// @return A pointer to the created ``QkParam``.
///
/// # Example
///
/// ```c
/// QkParam *r = qk_param_from_double(2.5);
/// ```
///
#[unsafe(no_mangle)]
pub extern "C" fn qk_param_from_double(value: f64) -> *mut Param {
    let value = Param::Float(value);
    Box::into_raw(Box::new(value))
}

/// @ingroup QkParam
/// Construct a new ``QkParam`` from a complex number, given as ``QkComplex64``.
///
/// @param value A ``QkComplex64`` to initialize the ``QkParam``.
///
/// @return A pointer to the created ``QkParam``.
///
/// # Example
///
/// ```c
/// QkComplex64 c = {1.0, 2.0};  // 1 + 2i
/// QkParam *param = qk_param_from_complex(c);
/// ```
///
#[unsafe(no_mangle)]
pub extern "C" fn qk_param_from_complex(value: Complex64) -> *mut Param {
    let value = SymbolExpr::Value(Value::Complex(value));
    let expr = ParameterExpression::from_symbol_expr(value);
    let param = Param::ParameterExpression(Arc::new(expr));
    Box::into_raw(Box::new(param))
}

/// @ingroup QkParam
/// Copy a ``QkParam``.
///
/// @param param The ``QkParam`` to copy.
///
/// @return A pointer to the created ``QkParam``.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *b = qk_param_copy(a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if ``param`` is not a valid pointer to a non-null ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_copy(param: *const Param) -> *mut Param {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let expr = unsafe { const_ptr_as_ref(param) };
    Box::into_raw(Box::new(expr.clone()))
}

/// @ingroup QkParam
/// Get a string representation of the ``QkParam``.
///
/// @param param A pointer to the ``QkParam``.
///
/// @return A pointer to a nul-terminated char array of the string representation for ``param``.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// char* str = qk_param_str(a);
/// printf(str);
/// qk_str_free(str);
/// qk_param_free(a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if ``param`` is not a valid pointer to a non-null ``QkParam``.
///
/// The string must not be freed with the normal C free, you must use ``qk_str_free`` to
/// free the memory consumed by the String. Not calling ``qk_str_free`` will lead to a
/// memory leak.
///
/// Do not change the length of the string after it's returned (by writing a nul byte somewhere
/// inside the string or removing the final one), although values can be mutated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_str(param: *const Param) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let expr = unsafe { const_ptr_as_ref(param) };
    let str = match expr {
        Param::ParameterExpression(expr) => expr.to_string(),
        Param::Float(f) => f.to_string(),
        Param::Obj(_) => panic!("Param::Obj is not supported in the C API"),
    };
    let out = CString::new(str.to_string()).unwrap();
    out.into_raw()
}

/// @ingroup QkParam
/// Add two ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store the result of ``lhs + rhs``.
/// @param lhs A pointer to the left hand side ``QkParam``.
/// @param rhs A pointer to the right hand side ``QkParam``.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *b = qk_param_new_symbol("b");
/// QkParam *out = qk_param_zero();
/// qk_param_add(out, a, b);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out``, ``lhs`` or ``rhs`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_add(
    out: *mut Param,
    lhs: *const Param,
    rhs: *const Param,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };

    let ret = match (lhs, rhs) {
        (Param::ParameterExpression(lhs), Param::ParameterExpression(rhs)) => lhs
            .add(rhs)
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::ParameterExpression(lhs), Param::Float(rhs)) => lhs
            .add(&ParameterExpression::from_f64(*rhs))
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::Float(lhs), Param::ParameterExpression(rhs)) => ParameterExpression::from_f64(*lhs)
            .add(rhs)
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::Float(lhs), Param::Float(rhs)) => Ok(Param::Float(*lhs + *rhs)),
        _ => return ExitCode::ArithmeticError, // PyObj case is not supported
    };

    match ret {
        Ok(ret) => {
            *out = ret;
            ExitCode::Success
        }
        Err(_) => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Subtract two ``QkParam``.
///
/// @param out A pointer to a ``QkParam`` to store the result of ``lhs - rhs``.
/// @param lhs A pointer to the left hand side ``QkParam``.
/// @param rhs A pointer to the right hand side ``QkParam``.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *b = qk_param_new_symbol("b");
/// QkParam *out = qk_param_zero();
/// qk_param_sub(out, a, b);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out``, ``lhs`` or ``rhs`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_sub(
    out: *mut Param,
    lhs: *const Param,
    rhs: *const Param,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };

    let ret = match (lhs, rhs) {
        (Param::ParameterExpression(lhs), Param::ParameterExpression(rhs)) => lhs
            .sub(rhs)
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::ParameterExpression(lhs), Param::Float(rhs)) => lhs
            .sub(&ParameterExpression::from_f64(*rhs))
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::Float(lhs), Param::ParameterExpression(rhs)) => ParameterExpression::from_f64(*lhs)
            .sub(rhs)
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::Float(lhs), Param::Float(rhs)) => Ok(Param::Float(*lhs - *rhs)),
        _ => return ExitCode::ArithmeticError, // PyObj case is not supported
    };
    match ret {
        Ok(ret) => {
            *out = ret;
            ExitCode::Success
        }
        Err(_) => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Multiply two ``QkParam``.
///
/// @param out A pointer to a ``QkParam`` to store the result of ``lhs * rhs``.
/// @param lhs A pointer to the left hand side ``QkParam``.
/// @param rhs A pointer to the right hand side ``QkParam``.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *b = qk_param_new_symbol("b");
/// QkParam *out = qk_param_zero();
/// qk_param_mul(out, a, b);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out``, ``lhs`` or ``rhs`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_mul(
    out: *mut Param,
    lhs: *const Param,
    rhs: *const Param,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };

    let ret = match (lhs, rhs) {
        (Param::ParameterExpression(lhs), Param::ParameterExpression(rhs)) => lhs
            .mul(rhs)
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::ParameterExpression(lhs), Param::Float(rhs)) => lhs
            .mul(&ParameterExpression::from_f64(*rhs))
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::Float(lhs), Param::ParameterExpression(rhs)) => ParameterExpression::from_f64(*lhs)
            .mul(rhs)
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::Float(lhs), Param::Float(rhs)) => Ok(Param::Float(*lhs * *rhs)),
        _ => return ExitCode::ArithmeticError, // PyObj case is not supported
    };
    match ret {
        Ok(ret) => {
            *out = ret;
            ExitCode::Success
        }
        Err(_) => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Divide a ``QkParam`` by another.
///
/// @param out A pointer to a ``QkParam`` to store the result of ``lhs / rhs``.
/// @param num A pointer to the numerator ``QkParam``.
/// @param den A pointer to the denominator ``QkParam``.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *b = qk_param_new_symbol("b");
/// QkParam *out = qk_param_zero();
/// qk_param_div(out, a, b);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out``, ``num`` or ``den`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_div(
    out: *mut Param,
    num: *const Param,
    den: *const Param,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let num = unsafe { const_ptr_as_ref(num) };
    let den = unsafe { const_ptr_as_ref(den) };

    let ret = match (num, den) {
        (Param::ParameterExpression(num), Param::ParameterExpression(den)) => num
            .div(den)
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::ParameterExpression(num), Param::Float(den)) => num
            .div(&ParameterExpression::from_f64(*den))
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::Float(num), Param::ParameterExpression(den)) => ParameterExpression::from_f64(*num)
            .div(den)
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::Float(num), Param::Float(den)) => Ok(Param::Float(*num / *den)),
        _ => return ExitCode::ArithmeticError, // PyObj case is not supported
    };
    match ret {
        Ok(ret) => {
            *out = ret;
            ExitCode::Success
        }
        Err(_) => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Raise a ``QkParam`` to the power of another.
///
/// @param out A pointer to a ``QkParam`` to store the result of ``lhs ** rhs``.
/// @param base A pointer to the left hand side ``QkParam``.
/// @param pow A pointer to the right hand side ``QkParam``.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *base = qk_param_new_symbol("a");
/// QkParam *pow = qk_param_new_symbol("b");
/// QkParam *out = qk_param_zero();
/// qk_param_pow(out, base, pow);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out``, ``base`` or ``pow`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_pow(
    out: *mut Param,
    base: *const Param,
    pow: *const Param,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let base = unsafe { const_ptr_as_ref(base) };
    let pow = unsafe { const_ptr_as_ref(pow) };

    let ret = match (base, pow) {
        (Param::ParameterExpression(base), Param::ParameterExpression(pow)) => base
            .pow(pow)
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::ParameterExpression(base), Param::Float(pow)) => base
            .pow(&ParameterExpression::from_f64(*pow))
            .map(|expr| Param::ParameterExpression(expr.into())),
        (Param::Float(base), Param::ParameterExpression(pow)) => {
            ParameterExpression::from_f64(*base)
                .pow(pow)
                .map(|expr| Param::ParameterExpression(expr.into()))
        }
        (Param::Float(base), Param::Float(pow)) => Ok(Param::Float(base.powf(*pow))),
        _ => return ExitCode::ArithmeticError, // PyObj case is not supported
    };
    match ret {
        Ok(ret) => {
            *out = ret;
            ExitCode::Success
        }
        Err(_) => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Calculate the sine of a ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store the result, ``sin(src)``.
/// @param src A pointer to the ``QkParam`` to apply the sine to.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_sin(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_sin(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.sin()));
            ExitCode::Success
        }
        Param::Float(f) => {
            *out = Param::Float(f.sin());
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Calculate the cosine of a ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store the result, ``cos(src)``.
/// @param src A pointer to the ``QkParam`` to apply the cosine to.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_cos(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_cos(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.cos()));
            ExitCode::Success
        }
        Param::Float(f) => {
            *out = Param::Float(f.cos());
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Calculate the tangent of a ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store the result, ``tan(src)``.
/// @param src A pointer to the ``QkParam`` to apply the tangent to.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_tan(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_tan(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.tan()));
            ExitCode::Success
        }
        Param::Float(f) => {
            *out = Param::Float(f.tan());
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Calculate the arcsine of a ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store the result, ``asin(src)``.
/// @param src A pointer to the ``QkParam`` to apply the arcsine to.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_asin(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_asin(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.asin()));
            ExitCode::Success
        }
        Param::Float(f) => {
            *out = Param::Float(f.asin());
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Calculate the arccosine of a ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store the result, ``acos(src)``.
/// @param src A pointer to the ``QkParam`` to apply the cosine to.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_acos(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_acos(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.acos()));
            ExitCode::Success
        }
        Param::Float(f) => {
            *out = Param::Float(f.acos());
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Calculate the arctangent of a ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store a result, ``atan(src)``.
/// @param src A pointer to the ``QkParam`` to apply the arctangent to.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_atan(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_atan(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.atan()));
            ExitCode::Success
        }
        Param::Float(f) => {
            *out = Param::Float(f.atan());
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Calculate the natural logarithm of a ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store the result, ``log(src)``.
/// @param src A pointer to the ``QkParam`` to apply the logarithm to.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_log(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_log(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.log()));
            ExitCode::Success
        }
        Param::Float(f) => {
            *out = Param::Float(f.ln());
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Apply the exponential function to a ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store the result, ``exp(src)``.
/// @param src A pointer to the ``QkParam`` to compute the exponential of.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_exp(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_exp(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.exp()));
            ExitCode::Success
        }
        Param::Float(f) => {
            *out = Param::Float(f.exp());
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Calculate the absolute value of a ``QkParam``.
///
/// @param out A pointer to the QkParam to store a result.
/// @param src A pointer to the QkParam.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_abs(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_abs(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.abs()));
            ExitCode::Success
        }
        Param::Float(f) => {
            *out = Param::Float(f.abs());
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Get the sign of a ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store the result, ``sign(src)``.
/// @param src A pointer to the ``QkParam`` to get the sign of.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_sign(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_sign(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.sign()));
            ExitCode::Success
        }
        Param::Float(f) => {
            *out = Param::Float(f.signum());
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Negate a ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store the result, ``-src``.
/// @param src A pointer to the ``QkParam`` to negate.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_neg(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_neg(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.neg()));
            ExitCode::Success
        }
        Param::Float(f) => {
            *out = Param::Float(-f);
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Calculate the complex conjugate of a ``QkParam``.
///
/// @param out A pointer to the ``QkParam`` to store the result, ``conj(src)``.
/// @param src A pointer to the ``QkParam`` to conjugate.
///
/// @return An exit code indicating ``QkExitCode_Success`` upon success and an error otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *out = qk_param_zero();
/// qk_param_conjugate(out, a);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``out`` or ``src`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_conjugate(out: *mut Param, src: *const Param) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let out = unsafe { mut_ptr_as_ref(out) };
    let src = unsafe { const_ptr_as_ref(src) };

    match src {
        Param::ParameterExpression(expr) => {
            *out = Param::ParameterExpression(Arc::new(expr.conjugate()));
            ExitCode::Success
        }
        Param::Float(_) => {
            *out = Param::Float(0.);
            ExitCode::Success
        }
        _ => ExitCode::ArithmeticError,
    }
}

/// @ingroup QkParam
/// Compare two ``QkParam`` for equality.
///
/// @param lhs A pointer to the left hand side ``QkParam``.
/// @param rhs A pointer to the right hand side ``QkParam``.
///
/// @return ``true`` if the ``QkParam`` objects are equal, ``false`` otherwise.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *one = qk_param_from_value(1.0);
/// QkParam *mone = qk_param_from_value(-1.0);
///
/// QkParam *x = qk_param_add(a, one);
/// QkParam *y = qk_param_sub(a, mone);
///
/// bool equal = qk_param_equal(x, y);
/// ```
///
/// # Safety
///
/// The behavior is undefined if any of ``lhs`` or ``rhs`` is not a valid, non-null
/// pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_equal(lhs: *const Param, rhs: *const Param) -> bool {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };

    lhs.eq(rhs).unwrap()
}

/// @ingroup QkParam
/// Attempt casting the ``QkParam`` as ``double``.
///
/// If the parameter could not be cast to a ``double``, because there were unbound parameters,
/// ``NAN`` is returned. Note that for ``QkParam`` representing complex values the real part is
/// returned.
///
/// @param param A pointer to the ``QkParam`` to evaluate.
///
/// @return The value, if casting was successful, otherwise ``NAN``.
///
/// # Example
///
/// ```c
/// QkParam *a = qk_param_new_symbol("a");
/// QkParam *b = qk_param_new_symbol("b");
/// QkParam *x = qk_param_zero();
/// qk_param_add(x, a, b);
///
/// const QkParam* keys[2] = {a, b};
/// QkParam *y = qk_param_zero();
/// qk_param_bind(y, x, keys, {1.0, 2.0}, 2);
///
/// double out = qk_param_as_real(y);
/// ```
///
/// # Safety
///
/// The behavior is undefined if ``param`` is not a valid, non-null pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_param_as_real(param: *const Param) -> f64 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let param = unsafe { const_ptr_as_ref(param) };

    match param {
        Param::ParameterExpression(expr) => match expr.try_to_value(true) {
            Ok(v) => v.as_real(),
            Err(_) => f64::NAN,
        },
        Param::Float(f) => *f,
        Param::Obj(_) => panic!("Param::Obj is not supported in the C API"),
    }
}
