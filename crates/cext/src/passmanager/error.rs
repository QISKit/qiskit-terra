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

use crate::pointers::{ExposesOwnedPointers, expose_by_box};
use crate::util;
use std::ffi::{CStr, c_char};

/// @ingroup pass-manager
/// A detailed error raised during compilation.
///
/// Compilation pipelines, including custom compiler passes, may want to provide detailed error
/// information back to the caller.  These errors are not typically intended for programmatic
/// handling, but are intended to be read and acted on by an end user.
pub struct CompilationError(pub anyhow::Error);
// SAFETY: `CompilationError` is always exposed and freed by `Box`.
const _: () = unsafe { expose_by_box!(CompilationError) };

/// @ingroup pass-manager
/// Create a new compilation error with a fixed message.
///
/// @param msg A borrowed pointer to a nul-terminated string.
/// @return An owned compiler error.
///
/// # Safety
///
/// `msg` must point to nul-terminated bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_compilation_error_new(msg: *const c_char) -> *mut CompilationError {
    // SAFETY: per documentation, `msg` points to nul-terminated bytes.
    let msg = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    CompilationError(anyhow::Error::msg(msg)).into_leaked()
}

/// @ingroup pass-manager
/// Get a simple owned string representation of the error.
///
/// The exact format of this representation is not stable, but is generally intended for display to
/// a user.  Error messages may include arbitrary strings that are controlled by custom passes
/// defined outside of Qiskit.
///
/// The return value must be freed with `qk_str_free`.
///
/// @param error The error to get the message for.
/// @return An owned UTF-8 string representation of the error.
///
/// # Safety
///
/// Behavior is undefined if `error` does not point to a valid `QkCompilationError`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_compilation_error_str(error: *const CompilationError) -> *mut c_char {
    // SAFETY: per documentation, `error` points to an aligned valid error.
    let error = unsafe { &*error };
    util::cstring_from_utf8_lossy(error.0.to_string()).into_raw()
}

/// @ingroup pass-manager
/// Free the given owned pointer.
///
/// Does nothing if `error` is null.
///
/// @param error The owned pointer to free.
///
/// # Safety
///
/// Behavior is undefined is `error` is neither null nor points to a valid owned `CompilationError`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_compilation_error_free(error: *mut CompilationError) {
    // SAFETY: per documentation, `error` is either `null` or a valid owned `CompilationError`.
    _ = (!error.is_null()).then(|| unsafe { CompilationError::steal(error) });
}
