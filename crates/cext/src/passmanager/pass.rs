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

use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;
use std::{mem, ptr};

use super::{CompilationError, IrHandle};
use crate::dyn_types::*;
use crate::pointers::{ExposesOwnedPointers, arc_clone_from_raw, expose_by_arc, expose_by_box};
use qiskit_passmanager::{IR, Pass, PassContext, PassError};
use qiskit_util::dyn_types::*;

/// @ingroup pass-manager
/// The available methods that can be implemented by a custom pass.
///
/// These are the valid values for `QkVtableEntry::slot` and the associated function-pointer types
/// required for the `table` argument of `qk_pass_new`.
///
/// # Safety
///
/// All functions, including the destructor, must be callable from any thread.
#[derive(Clone, Copy, derive_more::TryFrom, Debug)]
#[try_from(repr)]
#[repr(u32)]
pub enum PassSlot {
    /// Run the pass on an owned IR of the correct type. *Required*.
    ///
    /// Signature:
    /// ```c
    /// void *run(void *this, void *ir, QkPassContext *context, QkCompilationError **error);
    /// ```
    ///
    /// # Implementation
    ///
    /// This implements the pass behavior.  `this` will be equal to a value passed to `qk_pass_new`
    /// with this vtable.  `ir` will point to owned data of the type specified by the vtable's
    /// `ir_in` field.  It is the responsibility of the implementer to ensure `ir` is freed, if
    /// necessary.
    ///
    /// The function must return owned data of the type specified by `ir_out`, or `NULL` if an error
    /// as occurred.
    ///
    /// If the function wants to indicate an error state, it must write a valid owned
    /// `QkCompilationError *` object into the `error` pointer, ensure the `ir` pointer is freed,
    /// and return `NULL`.
    RunOwned = 0,
    /// A destructor for the `this` argument of a pass, at the time that the pass is destructed.
    /// *Optional*.
    ///
    /// This will not be called if the data pointer is `NULL`.
    ///
    /// Signature:
    /// ```c
    /// void delete(void *this);
    /// ```
    Delete = 1,
}

/// @ingroup pass-manager
/// A "type object" representing the behavior of a custom pass.
///
/// These objects represent only the behavior; they must be combined with a data pointer of the
/// correct type to produce a complete pass.
///
/// These are constructed using `qk_pass_vtable_new`, and are given to `qk_pass_new`.
///
/// See `QkPassSlot` for an enumeration of the possible behavior and its semantics.
pub struct PassVtable {
    /// A human-readable name for the pass.
    name: String,
    /// A handle to the complete IR methods of the input type, including the static constructor.
    ir_in: Arc<dyn DynTraitExposer<dyn IR>>,
    /// A handle to the complete IR methods of the output type, including the static constructor.
    ir_out: Arc<dyn DynTraitExposer<dyn IR>>,
    /// Run the pass on an owned copy of the IR.  See [`PassSlot::RunOwned`].
    run_owned: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut PassContext,
        *mut *mut CompilationError,
    ) -> *mut c_void,
    /// The destructor of the [`CPass::this`] pointer.  See [`PassSlot::Delete`].
    delete: Option<unsafe extern "C" fn(*mut c_void)>,
}
// SAFETY: `PassVtable` is always exposed and freed via an `Arc`.
const _: () = unsafe { expose_by_arc!(PassVtable) };
/// The primary constructor of `PassVtable` from Rust.
impl TryFrom<PassVtablePartial> for PassVtable {
    /// The first slot encountered that's required but absent.
    type Error = PassSlot;

    fn try_from(partial: PassVtablePartial) -> Result<Self, Self::Error> {
        Ok(Self {
            name: partial.name,
            ir_in: partial.ir_in,
            ir_out: partial.ir_out,
            run_owned: partial.run_owned.ok_or(PassSlot::RunOwned)?,
            delete: partial.delete,
        })
    }
}

/// @ingroup pass-manager
/// Create a new "type object" for a custom pass.
///
/// Once you have the output of this function, you create instances of the pass by combining it with
/// a data pointer using `qk_pass_new`.
///
/// @param name A borrowed nul-terminated human-readable name for the pass. Used for debugging.
/// @param ir_in A borrowed handle to the IR "type object" representing the expected input type for
///     the pass.
/// @param ir_out A borrowed handle to the IR "type object" representing the expected output type
///     for the pass.
/// @param table A borrowed table of `QkVtableEntry` objects terminated by an entry with `.slot =
///     -1`.  See `QkPassSlot` for the allowed slot values and associated function-pointer types.
///     Unknown values of `slot` are ignored.
/// @return An owned type object that can lent to `qk_pass_new` to create instances of the pass.
///
/// # Safety
///
/// Behavior is undefined if any of the following are violated:
///
/// * `name` is a nul-terminated string.
/// * `ir_in` points to a valid `QkIrHandle` object.
/// * `ir_out` points to a valid `QkIrHandle` object.
/// * `table` points to contiguous, valid instances of `VtableEntry` that are all consistent entries
///   as defined by `QkPassSlot`, with no duplicate `slot` entries and terminated by `.slot = -1`.
/// * `table` contains slot items for all required slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_vtable_new(
    name: *const c_char,
    ir_in: *const IrHandle,
    ir_out: *const IrHandle,
    mut table: *const VtableEntry,
) -> *mut PassVtable {
    // SAFETY: per documentation `name` is a pointer to nul-terminated `char`s.
    let name = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: per documentation, `ir_in` points to a valid `IrHandle` object.
    let ir_in = unsafe { &*ir_in };
    // SAFETY: per documentation, `ir_out` points to a valid `IrHandle` object.
    let ir_out = unsafe { &*ir_out };
    let mut partial = PassVtablePartial::new(name, Arc::clone(&ir_in.0), Arc::clone(&ir_out.0));
    loop {
        // SAFETY: per documentation, `table` is valid for reads until we see the sentinel all-ones
        // pattern in a `slot`.
        let entry = unsafe { table.read() };
        table = if entry.slot == u32::MAX {
            break;
        } else {
            table.wrapping_add(1)
        };
        let Ok(slot) = PassSlot::try_from(entry.slot) else {
            // We assume this is a slot from a later version of Qiskit.
            // TODO: add an envvar / global to turn on debug information in these cases?
            continue;
        };
        // SAFETY: per documentation, `entry,ptr` is of the expected function-pointer type and valid
        // to call, because `entry.slot` was not all-ones.
        if unsafe { partial.set(slot, entry.ptr) } {
            // This a documented UB case.
            return ptr::dangling_mut();
        }
    }
    // SAFETY: per documentation, all required methods were set.
    (unsafe { PassVtable::try_from(partial).unwrap_unchecked() }).into_leaked()
}

/// @ingroup pass-manager
/// Free a single reference to a `PassVtable`.
///
/// You can call this once you have finished constructing instances of passes that need this vtable;
/// each constructed pass owns its own reference to the table.
///
/// Does nothing if `vtable` is `NULL`.
///
/// @param vtable The owned reference to release.
///
/// # Safety
///
/// Behavior is undefined if `vtable` is not null or a valid owned reference to a `PassVtable`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_vtable_free(vtable: *mut PassVtable) {
    _ = (!vtable.is_null()).then(|| unsafe { PassVtable::steal(vtable) });
}

// TODO: we can very likely do a bit of macro trickery to simplify the creation of vtable objects,
// taking care that the filled public one needs to be visible to `cbindgen`.  For initial
// implementation, hard-coding is good enough.
/// Internal Rust builder object for constructing `PassVtable` via its [`TryFrom`] implementation.
struct PassVtablePartial {
    name: String,
    ir_in: Arc<dyn DynTraitExposer<dyn IR>>,
    ir_out: Arc<dyn DynTraitExposer<dyn IR>>,
    run_owned: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut PassContext,
            *mut *mut CompilationError,
        ) -> *mut c_void,
    >,
    delete: Option<unsafe extern "C" fn(*mut c_void)>,
}
impl PassVtablePartial {
    /// Initialize the builder.
    ///
    /// After this, call [`set`](Self::set) repeatedly to fill in the slots.
    fn new(
        name: String,
        ir_in: Arc<dyn DynTraitExposer<dyn IR>>,
        ir_out: Arc<dyn DynTraitExposer<dyn IR>>,
    ) -> Self {
        Self {
            name,
            ir_in,
            ir_out,
            run_owned: None,
            delete: None,
        }
    }

    /// Set the `slot` to the corresponding function `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid function pointer of the type expected by the corresponding method in
    /// [`PassVtable`].
    unsafe fn set(&mut self, slot: PassSlot, ptr: *mut c_void) -> bool {
        // This lint suppression is because there is no actual safety provided by writing out the
        // entire type again; the safety is only guaranteed as a long-range interaction of a C
        // caller matching the correct type signature in the documentation of `PassSlot`.
        #[allow(clippy::missing_transmute_annotations)]
        match slot {
            PassSlot::RunOwned => {
                // SAFETY: per documentation, caller ensures pointer type validity.
                let ptr = unsafe { mem::transmute::<*mut c_void, _>(ptr) };
                self.run_owned.replace(ptr).is_some()
            }
            PassSlot::Delete => {
                // SAFETY: per documentation, caller ensures pointer type validity.
                let ptr = unsafe { mem::transmute::<*mut c_void, _>(ptr) };
                self.delete.replace(ptr).is_some()
            }
        }
    }
}

/// @ingroup pass-manager
/// A pass with custom behavior defined through the C API.
///
/// This is a complete instance of a pass whose general behavior was previously defined using
/// `qk_pass_vtable_new`.  You construct instances of this struct by calling `qk_pass_new`.
///
/// These passes can be added to instances of `QkPassManager` using `qk_passmanager_push_pass`.
///
/// # Safety
///
/// The `this` data pointer must be safe to send and share between threads.  All methods in the
/// `QkPassVtable` must be safely callable from any thread.
pub struct CPass {
    /// The data pointer provided by C, and used as the `this` parameter in all [`PassVtable`]
    /// methods.
    this: *mut c_void,
    vtable: Arc<PassVtable>,
}
impl CPass {
    /// Create a new instance of the pass.
    ///
    /// # Safety
    ///
    /// 1. `this` must point to data of the type that is expected by all "`this`" arguments in
    ///    functions in the `vtable`.
    /// 2. the data pointed to by `this` must be safe to share and send between threads.
    unsafe fn new(this: *mut c_void, vtable: Arc<PassVtable>) -> Self {
        Self { this, vtable }
    }
}
// SAFETY: per struct documentation, the `this` pointer must be safe to send between threads.
unsafe impl Send for CPass {}
// SAFETY: per struct documentation, the `this` pointer must be safe to share between threads.
unsafe impl Sync for CPass {}
impl Pass for CPass {
    fn ir_id_in(&self) -> DynTypeId<'_> {
        self.vtable.ir_in.object_dyn_type_id()
    }
    fn ir_id_out(&self) -> DynTypeId<'_> {
        self.vtable.ir_out.object_dyn_type_id()
    }
    fn name(&self) -> &str {
        &self.vtable.name
    }
    fn run(&self, ir: Box<dyn IR>, context: &mut PassContext) -> Result<Box<dyn IR>, PassError> {
        let mut error = None::<ptr::NonNull<CompilationError>>;
        let error_ptr = (&raw mut error).cast::<*mut CompilationError>();
        let ir_in = self.vtable.ir_in.leak(ir).cast::<c_void>();
        let ir_out = unsafe { (self.vtable.run_owned)(self.this, ir_in, context, error_ptr) };
        match error {
            Some(error) => {
                let error = unsafe { Box::from_raw(error.as_ptr()) };
                Err(PassError::Runtime(error.0))
            }
            None => Ok(unsafe { self.vtable.ir_out.steal(ir_out) }),
        }
    }
}
impl Drop for CPass {
    fn drop(&mut self) {
        // The no-op on `NULL` is documented in `PassSlot::Delete` and used to mark moved ownership
        // of the data pointer.
        if let Some(delete) = self.vtable.delete
            && !self.this.is_null()
        {
            // SAFETY: per documentation of `PassVtable`, if the `delete` method is set, it is valid
            // to be passed `self.this` from any thread.
            unsafe { delete(self.this) };
        }
    }
}
// SAFETY: `CPass` is only created and freed via `Box`.
const _: () = unsafe { expose_by_box!(CPass) };

// TODO: having everything exposed as `CPass` is awkward for a future world where we have handles to
// built-in passes?

/// @ingroup pass-manager
/// Create a new instance of a pass whose behavior was previously defined.
///
/// The result pass object is typically then given to `qk_passmanager_push_pass`, or a related
/// method.
///
/// @param data The owned data pointer for this `vtable`.
/// @param vtable A borrowed "type object" created by `qk_pass_vtable_new`.
/// @return An owned pass object.
///
/// # Safety
///
/// Behavior is undefined if any of the following are violated:
///
/// * `this` points to data of the correct type for the methods in `vtable`.
/// * `vtable` points to a valid instance of `QkPassVtable`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_new(data: *mut c_void, vtable: *const PassVtable) -> *mut CPass {
    // SAFETY: per documentation, `vtable` is the still-valid result of `qk_pass_vtable_new`, which
    // returns the result of `Arc::into_raw`.
    let vtable = unsafe { arc_clone_from_raw(vtable) };
    // SAFETY: per documentation, `this` points to data of the type expected by `vtable` methods.
    (unsafe { CPass::new(data, vtable) }).into_leaked()
}

/// @ingroup pass-manager
/// Free a custom pass instance.
///
/// You typically do not need to call this, because the pass-manager building functions steal
/// ownership of `QkPass` arguments.
///
/// @param pass An owned pointer to the pass to free.
///
/// # Safety
///
/// Behavior is undefined if `pass` is not either null or a valid owned pointer to a `QkPass`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_free(pass: *mut CPass) {
    _ = (!pass.is_null()).then(|| unsafe { CPass::steal(pass) });
}
