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

use std::ffi::{CStr, CString, c_char};
use std::path::PathBuf;
use std::ptr;

use qiskit_circuit::circuit_data::CircuitData;

use crate::pointers::{const_ptr_as_ref, try_slice_from_ptr};

/// Options for the OpenQASM 2 importer.
///
/// Use ``qk_openqasm2_default_options`` to get a value with the defaults filled in, then override
/// the fields you care about.
///
/// # Safety
///
/// Each entry of ``include_path`` must be a pointer to memory containing a valid nul terminator,
/// and be valid for reads up to and including that terminator.  ``include_path`` itself must be
/// valid for ``num_include_paths`` reads of `char *`, or ``num_include_paths`` must be zero.
#[repr(C)]
pub struct OpenQasm2Options {
    /// The number of entries in ``include_path``.
    pub num_include_paths: usize,
    /// The directories searched, in order, to resolve an `include` statement.  Qiskit's built-in
    /// `qelib1.inc` is handled without consulting this, so a program that includes only that file
    /// needs no search path at all.
    pub include_path: *const *const c_char,
    /// Whether to run the parser in strict mode.  The default (``false``) accepts the same
    /// deviations from the specification that Qiskit has historically tolerated; ``true`` demands a
    /// leading `OPENQASM 2.0;` statement and is stricter throughout.
    pub strict: bool,
}

impl Default for OpenQasm2Options {
    fn default() -> Self {
        Self {
            num_include_paths: 0,
            include_path: std::ptr::null(),
            strict: false,
        }
    }
}

/// @ingroup QkOpenQasm2
///
/// Generate the default OpenQASM 2 importer options.
///
/// The defaults are an empty include path and non-strict parsing.
///
/// @return A ``QkOpenQasm2Options`` with the default settings.
#[unsafe(no_mangle)]
pub extern "C" fn qk_openqasm2_default_options() -> OpenQasm2Options {
    OpenQasm2Options::default()
}

/// Write an owned copy of `message` through `error`, if the caller asked for one.
///
/// # Safety
///
/// `error` must be null, or a valid pointer to write a `char *` to.
unsafe fn set_error(error: *mut *mut c_char, message: &str) {
    if error.is_null() {
        return;
    }
    // An interior nul would make `CString::new` fail; none of our messages should contain one, but
    // losing the whole diagnostic to that would be worse than mangling it.
    let owned = CString::new(message.replace('\0', "\\0"))
        .expect("interior nul bytes were already removed");
    // SAFETY: per documentation, `error` is non-null and valid for a pointer write.
    unsafe { *error = owned.into_raw() };
}

/// @ingroup QkOpenQasm2
///
/// Load a circuit from an OpenQASM 2 program.
///
/// This is a native importer; it does not go through Python.  It does not yet support the custom
/// classical functions or custom instructions that ``qiskit.qasm2.loads`` accepts, so a program
/// relying on those extensions will fail to parse.
///
/// @param program A nul-terminated string containing the OpenQASM 2 program.
/// @param options A pointer to the importer options.  If this is a null pointer, the defaults from
///     ``qk_openqasm2_default_options`` are used.
/// @param error A pointer to a pointer to a nul-terminated string.  On failure, a pointer to a
///     string describing the failure is written here, and must be released with ``qk_str_free``.
///     This can be a null pointer, in which case the description is discarded.  Nothing is written
///     here on success.
///
/// @returns A pointer to the newly built circuit, which it becomes the caller's responsibility to
///     release with ``qk_circuit_free``, or a null pointer if the program could not be loaded.
///
/// # Example
/// ```c
///     QkOpenQasm2Options options = qk_openqasm2_default_options();
///     char *error = NULL;
///     QkCircuit *qc = qk_circuit_from_openqasm2(
///         "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; h q[0]; cx q[0], q[1];",
///         &options, &error);
///     if (qc == NULL) {
///         printf("failed to load: %s\n", error);
///         qk_str_free(error);
///     } else {
///         qk_circuit_free(qc);
///     }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``program`` is not a valid pointer to a nul-terminated string.
/// ``options`` must be null or a valid pointer to a ``QkOpenQasm2Options`` satisfying that struct's
/// own safety requirements.  ``error`` must be null or a valid pointer to write a ``char *`` to; any
/// value already pointed to by it is overwritten without being released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_from_openqasm2(
    program: *const c_char,
    options: *const OpenQasm2Options,
    error: *mut *mut c_char,
) -> *mut CircuitData {
    if program.is_null() {
        // SAFETY: per documentation, `error` is null or valid for a pointer write.
        unsafe { set_error(error, "`program` must not be null") };
        return ptr::null_mut();
    }
    let defaults = OpenQasm2Options::default();
    let options = if options.is_null() {
        &defaults
    } else {
        // SAFETY: we checked the pointer is not null, then, per documentation, it is a valid and
        // aligned pointer.
        unsafe { const_ptr_as_ref(options) }
    };

    // SAFETY: per documentation, `program` points to a nul-terminated string.
    let Ok(program) = unsafe { CStr::from_ptr(program) }.to_str() else {
        // SAFETY: per documentation, `error` is null or valid for a pointer write.
        unsafe { set_error(error, "the OpenQASM 2 program was not valid UTF-8") };
        return ptr::null_mut();
    };

    // SAFETY: per the documentation on `QkOpenQasm2Options`, `include_path` is valid for
    // `num_include_paths` reads.
    let raw_include_path =
        match unsafe { try_slice_from_ptr(options.include_path, options.num_include_paths) } {
            Ok(raw_include_path) => raw_include_path,
            Err(_) => {
                // SAFETY: per documentation, `error` is null or valid for a pointer write.
                unsafe {
                    set_error(
                        error,
                        "`include_path` was not valid for `num_include_paths`",
                    )
                };
                return ptr::null_mut();
            }
        };
    let mut include_path = Vec::with_capacity(raw_include_path.len());
    for &entry in raw_include_path {
        if entry.is_null() {
            // SAFETY: per documentation, `error` is null or valid for a pointer write.
            unsafe { set_error(error, "an include path entry was null") };
            return ptr::null_mut();
        }
        // SAFETY: per the documentation on `QkOpenQasm2Options`, each entry is nul-terminated.
        let Ok(entry) = unsafe { CStr::from_ptr(entry) }.to_str() else {
            // SAFETY: per documentation, `error` is null or valid for a pointer write.
            unsafe { set_error(error, "an include path entry was not valid UTF-8") };
            return ptr::null_mut();
        };
        include_path.push(PathBuf::from(entry));
    }

    match qiskit_qasm2::circuit_from_string(
        program.to_owned(),
        include_path,
        &[],
        &[],
        options.strict,
    ) {
        Ok(built) => Box::into_raw(Box::new(built)),
        Err(err) => {
            // SAFETY: per documentation, `error` is null or valid for a pointer write.
            unsafe { set_error(error, &err.message) };
            ptr::null_mut()
        }
    }
}
