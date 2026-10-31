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
use std::fs;
use std::slice;

use qiskit_circuit::circuit_data::CircuitData;

use crate::exit_codes::ExitCode;
use crate::pointers::check_ptr;
use crate::util;

/// @ingroup QkQpy
/// Get the oldest QPY format version readable by the loaded library.
///
/// @return The oldest QPY format version supported for reading.
#[unsafe(no_mangle)]
pub extern "C" fn qk_qpy_read_min_version() -> u8 {
    qiskit_qpy::QPY_READ_MIN_VERSION
}

/// @ingroup QkQpy
/// Get the oldest QPY format version writable by the loaded library.
///
///  Use this value as the lower bound for the ``*_with_version`` dump functions.
///
/// This is the equivalent of Python's
/// @verbatim embed:rst:inline :attr:`~.qpy.QPY_COMPATIBILITY_VERSION` @endverbatim
///
/// @return The oldest QPY format version supported for writing.
#[unsafe(no_mangle)]
pub extern "C" fn qk_qpy_write_min_version() -> u8 {
    qiskit_qpy::QPY_WRITE_MIN_VERSION
}

enum QpyCError {
    ExitCode(ExitCode),
    Diagnostic(String),
}

impl From<ExitCode> for QpyCError {
    fn from(value: ExitCode) -> Self {
        Self::ExitCode(value)
    }
}

fn return_error(error: *mut *mut c_char, value: QpyCError) -> ExitCode {
    match value {
        QpyCError::ExitCode(code) => code,
        QpyCError::Diagnostic(message) => {
            if !error.is_null() {
                let message = util::cstring_from_utf8_lossy(message);
                // SAFETY: the caller guarantees that a non-null `error` is valid for one write.
                unsafe { error.write(message.into_raw()) };
            }
            ExitCode::QpyError
        }
    }
}

fn dump_circuits(circuits: &[&CircuitData], version: Option<u8>) -> Result<Vec<u8>, QpyCError> {
    qiskit_qpy::native_dump_qpy(circuits, version)
        .map_err(|err| QpyCError::Diagnostic(err.to_string()))
}

/// Generate a QPY payload from an array of QkCircuit pointers
///
/// # Safety
///
/// Every element of ``circuits`` must point to a valid ``QkCircuit``. If any entry is not a valid
/// aligned, non-null pointer the behavior is undefined. ``filename`` must point to a valid
/// nul-terminated string for the duration of this call. ``error`` must be null or valid
/// writeable pointer to a char pointer.
unsafe fn dump(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    version: Option<u8>,
) -> Result<Vec<u8>, QpyCError> {
    // SAFETY: this function's caller upholds the circuit-array contract.
    let circuits = unsafe { slice::from_raw_parts(circuits as *const &CircuitData, num_circuits) };
    dump_circuits(circuits, version)
}

fn load(payload: &[u8]) -> Result<Vec<CircuitData>, QpyCError> {
    qiskit_qpy::native_load_qpy(payload).map_err(|err| QpyCError::Diagnostic(err.to_string()))
}

unsafe fn dump_file_impl(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    filename: *const c_char,
    version: Option<u8>,
    error: *mut *mut c_char,
) -> ExitCode {
    if filename.is_null() {
        return ExitCode::NullPointerError;
    }
    // SAFETY: upheld by the caller contract and checked non-null above.
    let Ok(filename) = unsafe { CStr::from_ptr(filename) }.to_str() else {
        return return_error(
            error,
            QpyCError::Diagnostic("filename is not valid UTF-8".into()),
        );
    };
    // SAFETY: upheld by the caller contract and checked for alignment/null above.
    let result = unsafe { dump(circuits, num_circuits, version) };
    match result.and_then(|payload| {
        fs::write(filename, payload).map_err(|err| QpyCError::Diagnostic(err.to_string()))
    }) {
        Ok(()) => ExitCode::Success,
        Err(value) => return_error(error, value),
    }
}

unsafe fn dump_buffer_impl(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    buffer: *mut *mut u8,
    size: *mut usize,
    version: Option<u8>,
    error: *mut *mut c_char,
) -> ExitCode {
    if let Err(error) = check_ptr(buffer) {
        return error.into();
    }
    if let Err(error) = check_ptr(size) {
        return error.into();
    }
    // SAFETY: upheld by the caller contract and checked for alignment/null above.
    match unsafe { dump(circuits, num_circuits, version) } {
        Ok(payload) => {
            let payload = payload.into_boxed_slice();
            let payload_size = payload.len();
            // SAFETY: the caller guarantees both output locations are writable.
            unsafe {
                buffer.write(Box::into_raw(payload) as *mut u8);
                size.write(payload_size);
            }
            ExitCode::Success
        }
        Err(value) => return_error(error, value),
    }
}

/// @ingroup QkQpy
/// Write circuits to a QPY file.
///
/// @param circuits A valid, non-null pointer to an array of ``num_circuits`` circuit pointers.
/// @param num_circuits The number of circuit pointers in ``circuits``.
/// @param filename A valid, non-null, nul-terminated UTF-8 path.
/// @param error Optional output location for an error description. Free a returned string with
/// ``qk_str_free``.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` for an invalid path, unsupported QPY data, or I/O failure.
///
/// This function will return an error if any aspect of the circuit requires the Python interpreter
/// to generate the QPY payload. If you are using this function in a Python extension module you
/// should use the Python function @verbatim embed:rst:inline :func:`.qpy.dump` @endverbatim instead
/// to avoid this limitation.
///
/// # Safety
/// Every element of ``circuits`` must point to a valid ``QkCircuit``. If any entry is not a valid
/// aligned, non-null pointer the behavior is undefined. ``filename`` must point to a valid
/// nul-terminated string for the duration of this call. ``error`` must be null or valid
/// writeable pointer to a char pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_dump_file(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    filename: *const c_char,
    error: *mut *mut c_char,
) -> ExitCode {
    // SAFETY: this function has the same pointer requirements as the implementation.
    unsafe { dump_file_impl(circuits, num_circuits, filename, None, error) }
}

/// @ingroup QkQpy
/// Write circuits to a QPY file using a specific format version.
///
/// @param circuits A valid, non-null pointer to an array of ``num_circuits`` circuit pointers.
/// @param num_circuits The number of circuit pointers in ``circuits``.
/// @param filename A valid, non-null, nul-terminated UTF-8 path.
/// @param version The QPY format version. It must be at least the value returned by
/// ``qk_qpy_write_min_version``.
/// @param error Optional output location for an error description. Free a returned string with
/// ``qk_str_free``.
/// @return The same exit codes as ``qk_qpy_dump_file``.
///
/// This function will return an error if any aspect of the circuit requires the Python interpreter
/// to generate the QPY payload. If you are using this function in a Python extension module you
/// should use the Python function @verbatim embed:rst:inline :func:`.qpy.dump` @endverbatim instead
/// to avoid this limitation.
///
/// # Safety
/// Every element of ``circuits`` must point to a valid ``QkCircuit``. If any entry is not a valid
/// aligned, non-null pointer the behavior is undefined. ``filename`` must point to a valid
/// nul-terminated string for the duration of this call. ``error`` must be null or valid
/// writeable pointer to a char pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_dump_file_with_version(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    filename: *const c_char,
    version: u8,
    error: *mut *mut c_char,
) -> ExitCode {
    // SAFETY: this function has the same pointer requirements as the implementation.
    unsafe { dump_file_impl(circuits, num_circuits, filename, Some(version), error) }
}

/// @ingroup QkQpy
/// The output data from `qk_qpy_load_file` or `qk_qpy_load_buffer`
///
/// This contains an array of pointers to `QkCircuit` and a length.
///
/// The array as populated by the QPY load functions are owned by qiskit and must be freed by
/// calling `qk_qpy_loaded_circuits_clear`.
#[repr(C)]
pub struct QpyLoadedCircuits {
    /// The array of ``QkCircuit`` pointers that were loaded from the QPY payload.
    data: *mut *mut CircuitData,
    /// The number of circuits in ``data``.
    len: usize,
}

/// @ingroup QkQpy
/// Load all circuits from a QPY file.
///
/// @param filename A valid, non-null, nul-terminated UTF-8 path.
/// @param circuits Output location for loaded circuits.
/// @param error Optional output location for an error description. Free a returned string with
/// ``qk_str_free``.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` if the file cannot be read or is not supported QPY.
/// The returned array and all its circuits must be released with
/// `qk_qpy_loaded_circuits_clear`.
///
/// This function will return an error if any aspect of the circuits in the QPY payload require the
/// Python interpreter to create the `QkCircuit` object. If you are using this function in a Python
/// extension module you should use the Python function
/// @verbatim embed:rst:inline :func:`.qpy.dump` @endverbatim instead to avoid this limitation.
///
/// # Safety
/// ``filename`` must point to a valid nul-terminated string; ``circuits`` must be valid non-null,
/// aligned valid pointer to a `QpyLoadedCircuits` or behavior is undefined. ``error`` must be null
/// or a valid pointer to a `char` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_load_file(
    circuits: *mut QpyLoadedCircuits,
    filename: *const c_char,
    error: *mut *mut c_char,
) -> ExitCode {
    if filename.is_null() {
        return ExitCode::NullPointerError;
    }
    if let Err(error) = check_ptr(circuits) {
        return error.into();
    }
    // SAFETY: upheld by the caller contract and checked non-null above.
    let Ok(filename) = unsafe { CStr::from_ptr(filename) }.to_str() else {
        return return_error(
            error,
            QpyCError::Diagnostic("filename is not valid UTF-8".into()),
        );
    };
    let result = fs::read(filename)
        .map_err(|err| QpyCError::Diagnostic(err.to_string()))
        .and_then(|payload| load(&payload));
    match result {
        Ok(loaded) => {
            let loaded: Box<[*mut CircuitData]> = loaded
                .into_iter()
                .map(|circuit| Box::into_raw(Box::new(circuit)))
                .collect();
            let len = loaded.len();
            // SAFETY: the caller guarantees the output locations are writable.
            unsafe {
                circuits.write(QpyLoadedCircuits {
                    data: Box::into_raw(loaded) as *mut *mut CircuitData,
                    len,
                });
            }
            ExitCode::Success
        }
        Err(value) => return_error(error, value),
    }
}

/// @ingroup QkQpy
/// Serialize circuits into a newly allocated QPY buffer.
///
/// @param circuits A valid, non-null pointer to an array of ``num_circuits`` circuit pointers.
/// @param num_circuits The number of circuit pointers in ``circuits``.
/// @param buffer Output location for the newly allocated buffer. It is unchanged on failure.
/// @param size Output location for the buffer size in bytes. It is unchanged on failure.
/// @param error Optional output location for an error description. Free a returned string with
/// ``qk_str_free``.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` if serialization fails. The returned buffer must be
/// released with ``qk_qpy_free_buffer``.
///
/// This function will return an error if any aspect of the circuit requires the Python interpreter
/// to generate the QPY payload. If you are using this function in a Python extension module you
/// should use the Python function @verbatim embed:rst:inline :func:`.qpy.dump` @endverbatim instead
/// to avoid this limitation.
///
/// # Safety
/// Every element of ``circuits`` must point to a valid ``QkCircuit``. ``buffer`` and ``size`` must
/// each be valid for one pointer-sized write for the duration of this call. ``error`` must be null
/// or valid for one pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_dump_buffer(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    buffer: *mut *mut u8,
    size: *mut usize,
    error: *mut *mut c_char,
) -> ExitCode {
    // SAFETY: this function has the same pointer requirements as the implementation.
    unsafe { dump_buffer_impl(circuits, num_circuits, buffer, size, None, error) }
}

/// @ingroup QkQpy
/// Serialize circuits into a newly allocated QPY buffer using a specific format version.
///
/// @param circuits A valid, non-null pointer to an array of ``num_circuits`` circuit pointers.
/// @param num_circuits The number of circuit pointers in ``circuits``.
/// @param buffer Output location for the newly allocated buffer. It is unchanged on failure.
/// @param size Output location for the buffer size in bytes. It is unchanged on failure.
/// @param version The QPY format version. It must be at least the value returned by
/// ``qk_qpy_write_min_version``.
/// @param error Optional output location for an error description. Free a returned string with
/// ``qk_str_free``.
/// @return The same exit codes as ``qk_qpy_dump_buffer``.
///
/// This function will return an error if any aspect of the circuit requires the Python interpreter
/// to generate the QPY payload. If you are using this function in a Python extension module you
/// should use the Python function @verbatim embed:rst:inline :func:`.qpy.dump` @endverbatim instead
/// to avoid this limitation.
///
/// # Safety
/// The pointer requirements are the same as for ``qk_qpy_dump_buffer``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_dump_buffer_with_version(
    circuits: *const *const CircuitData,
    num_circuits: usize,
    buffer: *mut *mut u8,
    size: *mut usize,
    version: u8,
    error: *mut *mut c_char,
) -> ExitCode {
    // SAFETY: this function has the same pointer requirements as the implementation.
    unsafe { dump_buffer_impl(circuits, num_circuits, buffer, size, Some(version), error) }
}

/// @ingroup QkQpy
/// Load all circuits from a QPY buffer.
///
/// @param circuits Output location for loaded circuits.
/// @param buffer A valid buffer containing ``size`` bytes of QPY data.
/// @param size The size of ``buffer`` in bytes.
/// @param error Optional output location for an error description. Free a returned string with
/// ``qk_str_free``.
/// @return ``QkExitCode_Success`` on success, ``QkExitCode_NullPointerError`` for a null
/// pointer, or ``QkExitCode_QpyError`` if the buffer is not supported QPY. The returned array and
/// all its circuits must be released with `qk_qpy_loaded_circuits_clear`.
///
/// This function will return
/// an error if any aspect of the circuits in the QPY payload require the Python interpreter to create
/// the `QkCircuit` object. If you are using this function in a Python extension module you should use
/// the Python function @verbatim embed:rst:inline :func:`.qpy.dump` @endverbatim instead to avoid
/// this limitation.
///
/// # Safety
/// ``buffer`` must be valid for reads of ``size`` bytes; ``circuits`` must be valid non-null,
/// aligned valid pointer to a `QpyLoadedCircuits` or behavior is undefined. ``error`` must be null
/// or a valid pointer to a `char` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_load_buffer(
    circuits: *mut QpyLoadedCircuits,
    buffer: *const u8,
    size: usize,
    error: *mut *mut c_char,
) -> ExitCode {
    if buffer.is_null() {
        return ExitCode::NullPointerError;
    }
    if let Err(error) = check_ptr(circuits) {
        return error.into();
    }
    // SAFETY: upheld by the caller contract and checked non-null above.
    match load(unsafe { slice::from_raw_parts(buffer.cast::<u8>(), size) }) {
        Ok(loaded) => {
            let loaded: Box<[*mut CircuitData]> = loaded
                .into_iter()
                .map(|circuit| Box::into_raw(Box::new(circuit)))
                .collect();
            let len = loaded.len();
            // SAFETY: the caller guarantees the output locations are writable.
            unsafe {
                circuits.write(QpyLoadedCircuits {
                    data: Box::into_raw(loaded) as *mut *mut CircuitData,
                    len,
                })
            }
            ExitCode::Success
        }
        Err(value) => return_error(error, value),
    }
}

/// @ingroup QkQpy
/// Free circuits returned by ``qk_qpy_load_file`` or ``qk_qpy_load_buffer``.
///
/// @param circuits The pointer to the loaded circuits object populated by a QPY load function, or null.
///
/// # Safety
/// ``circuits`` must either be null, or valid aligned pointer to a `QkQpyLoadedCircuits`
/// object with the `.data` field populated with an array of `QkCircuit` pointers as populated
/// by `qk_qpy_load_file` or `qk_qpy_load_buffer`. Otherwise the behavior of this function is
/// undefined.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_loaded_circuits_clear(circuits: *mut QpyLoadedCircuits) {
    if !circuits.is_null() {
        // SAFETY: Per the documentation the pointer must be a valid pointer to a QpyLoadedCircuits
        // object
        let circuits = unsafe { &mut *circuits };
        // SAFETY: Per the documentation the QpyLoadCircuits object must have a valid circuits
        let circuit_array: Box<[*mut CircuitData]> = unsafe {
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                circuits.data,
                circuits.len,
            ))
        };
        for circuit in circuit_array.into_iter() {
            if !circuit.is_null() {
                // SAFETY: per the QPY documentation each pointer was allocated by a QPY load function.
                unsafe { drop(Box::from_raw(circuit)) };
            }
        }
        circuits.data = std::ptr::null_mut();
        circuits.len = 0;
    }
}

/// @ingroup QkQpy
/// Free a buffer returned by ``qk_qpy_dump_buffer``.
///
/// @param buffer A buffer returned by ``qk_qpy_dump_buffer``, or null.
/// @param size The buffer size returned by ``qk_qpy_dump_buffer``.
///
/// # Safety
/// A non-null ``buffer`` must have been returned by ``qk_qpy_dump_buffer`` with exactly this
/// ``size`` and must not have been freed previously.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_qpy_free_buffer(buffer: *mut u8, size: usize) {
    if !buffer.is_null() {
        // SAFETY: upheld by the caller contract; dump_buffer allocated this as a boxed slice.
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                buffer.cast::<u8>(),
                size,
            )));
        }
    }
}
