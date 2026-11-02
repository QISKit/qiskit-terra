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

use std::ffi::{CStr, c_char};
use std::path::PathBuf;

use qiskit_circuit::circuit_data::CircuitData;

use crate::exit_codes::ExitCode;
use crate::pointers::{ExposesOwnedPointers, check_ptr, const_ptr_as_ref, mut_ptr_as_ref};
use crate::util;

/// Options for the OpenQASM 2 importer.
///
/// Create a set with ``qk_openqasm2_options_new``, configure it with the setter functions, and
/// release it with ``qk_openqasm2_options_free``.
///
/// The type is deliberately opaque: the setters are the only supported way to change an option,
/// so fields can be added later without changing the layout that programs compiled against an
/// older header expect.
#[derive(Default)]
pub struct OpenQasm2Options {
    /// Directories searched, in order, to resolve an `include` statement.
    include_path: Vec<PathBuf>,
    /// Whether to run the parser in strict mode.
    strict: bool,
}

/// Write an owned copy of `message` through `error`, if the caller asked for one, and return
/// `code`.
///
/// # Safety
///
/// `error` must be null, or a valid pointer to write a `char *` to.
unsafe fn return_error(error: *mut *mut c_char, code: ExitCode, message: &str) -> ExitCode {
    if !error.is_null() {
        // SAFETY: per documentation, `error` is non-null and valid for a pointer write.
        unsafe { error.write(util::cstring_from_utf8_lossy(message).into_raw()) };
    }
    code
}

/// @ingroup QkOpenQasm2
///
/// Create importer options with the defaults filled in: an empty include path and non-strict
/// parsing.
///
/// @return A handle, never null, to be released with ``qk_openqasm2_options_free``.
#[unsafe(no_mangle)]
pub extern "C" fn qk_openqasm2_options_new() -> *mut OpenQasm2Options {
    Box::into_raw(Box::new(OpenQasm2Options::default()))
}

/// @ingroup QkOpenQasm2
///
/// Release a set of importer options.  A null pointer is accepted and ignored.
///
/// @param options A handle from ``qk_openqasm2_options_new``.
///
/// # Safety
///
/// ``options`` must not have been released already.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_openqasm2_options_free(options: *mut OpenQasm2Options) {
    if options.is_null() {
        return;
    }
    // SAFETY: per documentation, `options` is a live handle from `qk_openqasm2_options_new`.
    drop(unsafe { Box::from_raw(options) });
}

/// @ingroup QkOpenQasm2
///
/// Set whether the importer runs in strict mode.  The default (``false``) accepts the same
/// deviations from the specification that Qiskit has historically tolerated; ``true`` demands a
/// leading `OPENQASM 2.0;` statement and is stricter throughout.
///
/// @param options A handle to the options to modify.
/// @param strict Whether to run in strict mode.
///
/// # Safety
///
/// ``options`` must be a valid handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_openqasm2_options_set_strict(
    options: *mut OpenQasm2Options,
    strict: bool,
) {
    // SAFETY: per documentation, `options` is a valid handle.
    unsafe { mut_ptr_as_ref(options) }.strict = strict;
}

/// @ingroup QkOpenQasm2
///
/// Append a directory to search when resolving an `include` statement.  Directories are searched
/// in the order they are appended.  Qiskit's built-in `qelib1.inc` is handled without consulting
/// the search path.
///
/// @param options A handle to the options to modify.
/// @param path A nul-terminated string giving the directory to search.
/// @param error Receives a description on failure, released with ``qk_str_free``.  This may be a
///     null pointer, in which case the description is discarded.
///
/// @return ``QkExitCode_Success`` if the directory was appended,
///     ``QkExitCode_NullPointerError`` if ``path`` is null, or ``QkExitCode_CInputError`` if
///     ``path`` is not valid UTF-8.
///
/// # Safety
///
/// ``options`` must be a valid handle, ``path`` null or a valid pointer to a nul-terminated
/// string, and ``error`` null or valid for a pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_openqasm2_options_append_include_path(
    options: *mut OpenQasm2Options,
    path: *const c_char,
    error: *mut *mut c_char,
) -> ExitCode {
    if path.is_null() {
        return ExitCode::NullPointerError;
    }
    // SAFETY: per documentation, `path` points to a nul-terminated string.
    let Ok(path) = unsafe { CStr::from_ptr(path) }.to_str() else {
        return unsafe {
            return_error(
                error,
                ExitCode::CInputError,
                "the include path was not valid UTF-8",
            )
        };
    };
    // SAFETY: per documentation, `options` is a valid handle.
    unsafe { mut_ptr_as_ref(options) }
        .include_path
        .push(PathBuf::from(path));
    ExitCode::Success
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
/// @param options A handle to the importer options, which is borrowed only for the duration of the
///     call.  If this is a null pointer, the default options are used.
/// @param circuit On success, receives a newly built circuit, to be released with
///     ``qk_circuit_free``.  It is not written to on failure.
/// @param error Receives a description on failure, released with ``qk_str_free``.  This may be a
///     null pointer, in which case the description is discarded.  Not every failure describes
///     itself.
///
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` if ``program`` or
///     ``circuit`` is null, ``QkExitCode_AlignmentError`` if ``circuit`` is misaligned, or
///     ``QkExitCode_CInputError`` if the program could not be imported.
///
/// # Example
/// ```c
///     QkOpenQasm2Options *options = qk_openqasm2_options_new();
///     QkCircuit *qc = NULL;
///     char *error = NULL;
///     QkExitCode code = qk_openqasm2_load(
///         "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; h q[0]; cx q[0], q[1];",
///         options, &qc, &error);
///     qk_openqasm2_options_free(options);
///     if (code != QkExitCode_Success) {
///         printf("failed to load: %s\n", error);
///         qk_str_free(error);
///     } else {
///         qk_circuit_free(qc);
///     }
/// ```
///
/// # Safety
///
/// ``program`` must be null or a valid pointer to a nul-terminated string, ``options`` null or a
/// valid handle, and ``error`` null or valid for a pointer write; any value ``error`` already
/// points to is overwritten without being released.  ``circuit`` must point to a writable
/// ``QkCircuit *`` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_openqasm2_load(
    program: *const c_char,
    options: *const OpenQasm2Options,
    circuit: *mut *mut CircuitData,
    error: *mut *mut c_char,
) -> ExitCode {
    if program.is_null() {
        return ExitCode::NullPointerError;
    }
    if let Err(err) = check_ptr(circuit) {
        return err.into();
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
        return unsafe {
            return_error(
                error,
                ExitCode::CInputError,
                "the OpenQASM 2 program was not valid UTF-8",
            )
        };
    };

    match qiskit_qasm2::circuit_from_string(
        program.to_owned(),
        options.include_path.clone(),
        &[],
        &[],
        options.strict,
    ) {
        Ok(built) => {
            // SAFETY: checked non-null and aligned above, and, per documentation, otherwise valid
            // for one pointer write.
            unsafe { circuit.write(built.into_leaked()) };
            ExitCode::Success
        }
        Err(err) => unsafe { return_error(error, ExitCode::CInputError, &err.message) },
    }
}
