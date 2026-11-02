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

use std::ffi::c_void;
use std::ptr;

use super::{CPass, CompilationError, IrHandle};
use crate::ExitCode;
use crate::pointers::{ExposesOwnedPointers, expose_by_box};
use anyhow::anyhow;
use qiskit_passmanager::{Pass, PassManager, Task};

// SAFETY: `PassManager` is always exposed and freed by `Box`.
const _: () = unsafe { expose_by_box!(PassManager) };

/// @ingroup pass-manager
/// Create an empty pass manager.
///
/// @return An owned, empty pass manager.
///
/// This object must be freed by the user.
#[unsafe(no_mangle)]
pub extern "C" fn qk_passmanager_new() -> *mut PassManager {
    PassManager::new().into_leaked()
}

/// @ingroup pass-manager
/// Free the pass manager.
///
/// @param pm An owned pointer to the pass manager.
///
/// # Safety
///
/// Behavior is undefined if `pm` is not either null or a valid pointer to a `QkPassManager`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_free(pm: *mut PassManager) {
    _ = (!pm.is_null()).then(|| unsafe { PassManager::steal(pm) });
}

/// @ingroup pass-manager
/// Push a pass onto the pass manager.
///
/// @param pm A borrowed pointer to the pass manager.
/// @param pass An owned pass to push.  This steals ownership of the given pass.
///
/// @return `QkExitCode_Success` if the type was added successfully, or
///     `QkExitCode_IncompatibleTypes` if the pass could not be added because its expected IR types
///     do not match the pipeline's expectation.
///
/// # Safety
///
/// Behavior is undefined if any of the following are violated:
///
/// * `pm` points to a valid `QkPassManager`
/// * `pass` points to an owned `QkPass` instance.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_push_pass(
    pm: *mut PassManager,
    pass: *mut CPass,
) -> ExitCode {
    // SAFETY: per documentation, `pm` points to a valid `PassManager`.
    let pm = unsafe { &mut *pm };
    // SAFETY: per documentation, `pass` points to a valid owned `CPass`.
    let pass = unsafe { CPass::steal(pass) };
    match pm.try_push_task(Task::Transformation(pass as Box<dyn Pass>)) {
        Ok(()) => ExitCode::Success,
        Err(_) => ExitCode::IncompatibleTypes,
    }
}

// TODO: should we move the "expose" logic into the core `qiskit-passmanager` crate, and remove the
// handles from `run_simple`?  Pro: simpler signature and less chance for disagreement.  Cons: moves
// C-specific exposure code into the core; motivates exposing the dynamic-type comparison logic to
// C, for it to check safety.

/// @ingroup pass-manager
/// Run the pass manager.
///
/// `ir_in_handle` and `ir_out_handle` must match the type expectations of the pipeline.  This
/// function will return `NULL` and (optionally) set the error state if this is violated.
///
/// @param pm A borrowed pointer to the pass manager to run.
/// @param ir An owned data pointer of the type expected by `ir_in_handle`.  This function steals
///     ownership of the pointer, and will arrange for it to be destructed, even if an error occurs.
/// @param ir_in_handle A borrowed type object corresponding to the input type of the pass manager,
///     and the type of `ir`.
/// @param ir_out_handle A borrowed type object corresponding to the output type of the pass
///     manager.
/// @param[out] error If not `NULL`, then a location to write out a `QkCompilationError *` result.
///
/// @return If no fatal error occurred, then an owned data pointer of the type corresponding to
///     `ir_out_handle`.  `NULL` if a fatal error occurred.
///
/// # Safety
///
/// Behavior is undefined if any of the following are violated:
///
/// * `pm` is a valid pointer to a `QkPassManager`.
/// * `ir` is an owning data pointer of the type indicated by `ir_in_handle`.
/// * `ir_in_handle` points to a valid `IrHandle`.
/// * `ir_out_handle` points to a valid `IrHandle`.
/// * `error` is either null, or points to a storage location valid for a single write of one
///   `QkCompilationError *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_run_simple(
    pm: *mut PassManager,
    ir: *mut c_void,
    ir_in_handle: *const IrHandle,
    ir_out_handle: *const IrHandle,
    error: *mut *mut CompilationError,
) -> *mut c_void {
    // SAFETY: Per documentation, `pm` is non-null and valid
    let pm = unsafe { &mut *pm };
    let ir_in_handle = unsafe { &*ir_in_handle };
    let ir_out_handle = unsafe { &*ir_out_handle };
    let ir = unsafe { ir_in_handle.0.steal(ir) };

    let parse_error = |e| {
        if !error.is_null() {
            let e = CompilationError(e).into_leaked();
            unsafe { error.write(e) };
        }
        ptr::null_mut()
    };

    match pm.run_erased(ir) {
        Ok((ir_out, _)) => {
            if ir_out.dyn_type_id() != ir_out_handle.0.object_dyn_type_id() {
                parse_error(anyhow!(
                    "declared output IR type does not match the pipeline result"
                ))
            } else {
                ir_out_handle.0.leak(ir_out)
            }
        }
        Err(e) => parse_error(e.into()),
    }
}
