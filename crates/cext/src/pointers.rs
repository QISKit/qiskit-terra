// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::exit_codes::CInputError;
use std::sync::Arc;

/// Trait that declares how a Rust-native object exposes "owned" versions of itself to the C API.
///
/// # Implementing
///
/// You typically don't need to implement this by hand; use the [`expose_by_box!`] or
/// [`expose_by_arc!`] macros to do it for you.  Since the trait is unsafe to implement, these two
/// macros require you to wrap them in `unsafe` blocks to work.  You do this like
/// ```no_run
/// # #[macro_use] extern crate qiskit_cext;
/// struct MyType;
/// // SAFETY: `MyType` is always exposed and freed by `Box`.
/// const _: () = unsafe { expose_by_box!(MyType) };
///
/// #[unsafe(no_mangle)]
/// pub extern "C" fn qk_mytype_new() -> *mut MyType {
///     MyType.into_leaked()
/// }
/// #[unsafe(no_mangle)]
/// pub unsafe extern "C" fn qk_mytype_free(ptr: *mut MyType) {
///     _ = (!ptr.is_null()).then(|| unsafe { MyType::steal(ptr) })
/// }
/// ```
///
/// # Safety
///
/// By implementing this trait, you are asserting that every time an object passes ownership to C,
/// it uses the same smart-pointer type.  If you have a `qk_*_*free` method, this almost certainly
/// holds.
///
/// The [`Owner`] type needs to be able to leak itself to a single thin pointer, and then recreate
/// itself entirely from the same pointer. [`Box`] and [`Arc`] both satisfy this.
pub unsafe trait ExposesOwnedPointers: Sized + Sync + Send + 'static {
    /// The Rust-native owning "smart pointer" type.  This is generally going to be [`Box`] or
    /// [`Arc`].
    type Owner;

    /// Assume Rust-native ownership of `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a non-null, aligned pointer to an initialized `Self` which has no other
    /// outstanding valid reference, where the pointer was produced by [`Self::leak`] (or
    /// equivalent).
    unsafe fn steal(ptr: *mut Self) -> Self::Owner;
    /// Move the owned value into a suitable heap allocation for exposure.
    fn into_owned(self) -> Self::Owner;
    /// Leak the Rust-native owning object into a pointer.
    fn leak(ob: Self::Owner) -> *mut Self;
    /// Helper method that moves `self` into a suitable heap allocation for exposure, then
    /// immediately leaks it.
    #[inline]
    fn into_leaked(self) -> *mut Self {
        Self::leak(self.into_owned())
    }
}
/// Mark the given Rust type as being exposed to C only by [`Box<Self>`].
///
/// See the [implementation section](ExposesOwnedPointers#implementing) for detail on how to use
/// this macro and the considerations.
///
/// This implements [`ExposesOwnedPointers<Owner = Box<Self>>`](ExposesOwnedPointers) for the given
/// type.  You probably want to use that trait's [`into_leaked`](ExposesOwnedPointers::into_leaked)
/// and [`steal`](ExposesOwnedPointers::steal) methods exclusively at the C FFI boundary.
///
/// # Safety
///
/// All functions that expose and receive ownership of the Rust type use [`Box`] as the
/// owning heap allocation.
#[macro_export]
macro_rules! expose_by_box {
    ($ty:ty) => {{
        $crate::pointers::__noop_unsafe(); // Force the macro to need `unsafe {}`.
        //
        // SAFETY: per macro documentation, all exposures to C are guaranteed to use `Box`.
        unsafe impl $crate::pointers::ExposesOwnedPointers for $ty {
            type Owner = ::std::boxed::Box<Self>;

            #[inline]
            unsafe fn steal(ptr: *mut Self) -> Self::Owner {
                // SAFETY: per (trait) documentation, `ptr` is the only owner resulting from a
                // leaked `Box<Self>`.
                unsafe { Self::Owner::from_raw(ptr) }
            }
            #[inline]
            fn into_owned(self) -> Self::Owner {
                Self::Owner::new(self)
            }
            #[inline]
            fn leak(ob: Self::Owner) -> *mut Self {
                Self::Owner::into_raw(ob)
            }
        }
    }};
}
/// Mark the given Rust type as being exposed to C only by [`Arc<Self>`].
///
/// See the [implementation section](ExposesOwnedPointers#implementing) for detail on how to use
/// this macro and the considerations.
///
/// This implements [`ExposesOwnedPointers<Owner = Arc<Self>>`](ExposesOwnedPointers) for the given
/// type.  You probably want to use that trait's [`into_leaked`](ExposesOwnedPointers::into_leaked)
/// and [`steal`](ExposesOwnedPointers::steal) methods exclusively at the C FFI boundary.
///
/// # Safety
///
/// All functions that expose and receive ownership of the Rust type use [`Arc`] as the
/// owning heap allocation.
#[macro_export]
macro_rules! expose_by_arc {
    ($ty:ty) => {{
        $crate::pointers::__noop_unsafe(); // Force the macro to need `unsafe {}`.

        // SAFETY: per macro documentation, all exposures to C are guaranteed to use `Arc`.
        unsafe impl $crate::pointers::ExposesOwnedPointers for $ty {
            type Owner = ::std::sync::Arc<Self>;

            #[inline]
            unsafe fn steal(ptr: *mut Self) -> Self::Owner {
                // SAFETY: per (trait) documentation, `ptr` is an owner of a single `strong_count`
                // resulting from a leaked `Arc<Self>`.
                unsafe { Self::Owner::from_raw(ptr.cast_const()) }
            }
            #[inline]
            fn into_owned(self) -> Self::Owner {
                Self::Owner::new(self)
            }
            #[inline]
            fn leak(ob: Self::Owner) -> *mut Self {
                Self::Owner::into_raw(ob).cast_mut()
            }
        }
    }};
}
pub use {expose_by_arc, expose_by_box};

/// No-op function whose only purpose is to force use of wrapping `unsafe {}` blocks around certain
/// macro calls, which in turn implies necessary "SAFETY" comments.
#[doc(hidden)]
#[inline(always)]
pub const unsafe fn __noop_unsafe() {}

/// Check the pointer is not null and is aligned.
pub(crate) fn check_ptr<T>(ptr: *const T) -> Result<(), CInputError> {
    if ptr.is_null() {
        return Err(CInputError::NullPointerError);
    };
    if !ptr.is_aligned() {
        return Err(CInputError::AlignmentError);
    };
    Ok(())
}

/// Create a slice of length `len` from a given pointer.
///
/// If the length is zero, the function is infallible, though the returned slice may not be backed
/// by the same pointer.  Otherwise, check if the pointer is non-null and aligned.
///
/// # Safety
///
/// If `len` is non-zero, `ptr` must be valid for `len` reads of initialized memory for lifetime
/// `'a`.
pub(crate) unsafe fn try_slice_from_ptr<'a, T>(
    ptr: *const T,
    len: usize,
) -> Result<&'a [T], CInputError> {
    if len == 0 {
        Ok(&[])
    } else {
        // SAFETY: per documentation, pointer is valid for `len` reads of initialised memory at the
        // lifetime of the function.
        check_ptr(ptr).map(|_| unsafe { ::std::slice::from_raw_parts(ptr, len) })
    }
}
/// Create a slice of length `len` from a given pointer.
///
/// Panicking variant of [try_slice_from_ptr].
///
/// # Safety
///
/// If `len` is non-zero, `ptr` must be valid for `len` reads of initialized memory for lifetime
/// `'a`.
pub(crate) unsafe fn slice_from_ptr<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    unsafe { try_slice_from_ptr(ptr, len) }.expect("caller should ensure a valid pointer")
}

/// Clone a new [`Arc`] from a pointer.
///
/// The given `ptr` is only borrowed; it is valid to be given to `Arc::from_raw` (or this function)
/// again after this function returns.
///
/// # Safety
///
/// `ptr` must be the result of a call to [`Arc<T>::into_raw`] and still be valid to pass to
/// [`Arc::from_raw`].
pub unsafe fn arc_clone_from_raw<T: ?Sized>(ptr: *const T) -> Arc<T> {
    // SAFETY: per documentation, `ptr` is from `Arc::into_raw` and still valid.
    unsafe { Arc::increment_strong_count(ptr) };
    // SAFETY: per documentation, `ptr` is from `Arc::into_raw`, still valid, and we just
    // incremented the strong count for this `from_raw` call to take ownership of.
    unsafe { Arc::from_raw(ptr) }
}
