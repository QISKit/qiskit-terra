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

//! Additional tools for working with the dynamic runtime-type information system across the C FFI.

use qiskit_util::dyn_types::DynTypeId;
use std::ffi::c_void;

/// Define behavior for sending and receiving `dyn Trait` objects across the C FFI boundary.
///
/// The `Trait` type parameter is always expected to be some `dyn Trait`, the instances of which
/// participate in the dynamic-typing system.  The [`object_dyn_type_id`](Self::object_dyn_type_id)
/// corresponds to the dynamic type of the _produced_ object.
///
/// This trait is dyn compatible, with the intention that it is implemented on cheaper "marker"
/// structures that hold effectively the information needed for the (unstable)
/// [`Pointee::Metadata`](std::ptr::Pointee) component of the fat `Box<dyn T>` pointers.  For
/// objects that implement
///
/// # Safety
///
/// The [`leak`](Self::leak) and [`steal`](Self::steal) functions must agree on the type
/// constructed, and this type must use the same [`DynTypeId`] as produced by the
/// [`object_dyn_type_id`](Self::object_dyn_type_id) function.
///
/// Ensure that your implementation of [`leak`](Self::leak) does not allow a concrete `Box<T>` with
/// to run a custom destructor in its [`Drop`] implementation after you transfer ownership of the
/// data pointer to the output.  Consider using [`ManuallyDrop`](std::mem::ManuallyDrop) to suppress
/// a virtual destructor (though take care to drop the other fields and any containing smart
/// pointer), or arrange the destructor to check for ownership, such as a non-null pointer, before
/// attempting to run.
///
/// Other unsafe FFI code relies on the correctness and soundness of this trait to avoid undefined
/// behavior across the FFI boundary.
pub unsafe trait DynTraitExposer<Trait: ?Sized>: Send + Sync + 'static {
    /// Get the type identifier for the concrete objects that are produced by
    /// [`steal`](Self::steal) and consumed by [`leak`](Self::leak).
    fn object_dyn_type_id(&self) -> DynTypeId<'_>;
    /// Leak the raw data pointer of `ob` to a type-erased C pointer.
    ///
    /// *Warning*: when implementating this method, make sure that you do not trigger a virtual
    /// destructor after transferring ownership of the data pointer.  See the trait's "Safety"
    /// section.
    ///
    /// # Panics
    ///
    /// Implementers of the trait may assume that `ob` will always downcast to the known concrete
    /// type expected by the base object.
    fn leak(&self, ob: Box<Trait>) -> *mut ::std::ffi::c_void;
    /// Steal the ownership of the raw data pointer `ptr`, and combine it with the necessary dynamic
    /// trait vtables to produce a complete [`Box<dyn T>`] object.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid pointer that solely owns a single instance of the concrete type
    /// expected by this trait implementer.
    unsafe fn steal(&self, ptr: *mut ::std::ffi::c_void) -> Box<Trait>;
}

/// Create a marker struct that implements [`DynTraitExposer`] for some chosen trait.
///
/// # Examples
///
/// ```
/// use qiskit_util::dyn_types::*;
/// use qiskit_cext::dyn_types::*;
///
/// pub trait MyTrait: DynTyped + Send + Sync + 'static {}
/// make_static_trait_exposer!(pub struct StaticMyTraitExposer<T> for dyn MyTrait);
/// ```
///
/// The produced struct is constructed using a `const` `new` method, or via its [`Default`]
/// implementation.
#[macro_export]
macro_rules! make_static_trait_exposer {
    ($vis:vis struct $ty:ident<$T:ident> for dyn $trait:ident) => {
        /// A zero-sized marker struct for implementing
        /// [`DynTraitExposer`](qiskit_cext::dyn_types::DynTraitExposer) for
        #[doc = concat!("[`", stringify!($trait), "`].")]
        #[derive(Debug)]
        $vis struct $ty<$T>(::std::marker::PhantomData<$T>);
        impl<$T> $ty<$T> {
            $vis const fn new() -> Self {
                Self(::std::marker::PhantomData)
            }
        }
        impl<$T> ::std::clone::Clone for $ty<$T> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<$T> ::std::marker::Copy for $ty<$T> {}
        impl<$T> ::std::default::Default for $ty<$T> {
            fn default() -> Self {
                Self::new()
            }
        }

        // SAFETY: since `T` is static, we easily make all the required methods agree simply by
        // having the compiler fill in the correct information statically.
        unsafe impl<$T> $crate::dyn_types::DynTraitExposer<dyn $trait> for $ty<$T>
        where $T: $trait
            + ::qiskit_util::dyn_types::StaticDynTyped
            + $crate::pointers::ExposesOwnedPointers<Owner = ::std::boxed::Box<$T>>
        {
            fn object_dyn_type_id(&self) -> ::qiskit_util::dyn_types::DynTypeId<'_> {
                $T::static_dyn_type_id()
            }
            fn leak(&self, ob: ::std::boxed::Box<dyn $trait>) -> *mut ::std::ffi::c_void {
                use std::{any::Any, boxed::Box};
                use qiskit_util::dyn_types::DynTyped;

                debug_assert_eq!(
                    (&*ob as &dyn DynTyped).dyn_type_id(),
                    self.object_dyn_type_id()
                );
                let typed = (ob as Box<dyn Any>)
                    .downcast::<$T>()
                    .expect("caller should ensure correct type");
                $T::leak(typed).cast()
            }
            unsafe fn steal(&self, ptr: *mut ::std::ffi::c_void) -> ::std::boxed::Box<dyn $trait> {
                (unsafe { $T::steal(ptr.cast()) }) as ::std::boxed::Box<dyn $trait>
            }
        }
    };
}
pub use make_static_trait_exposer;

/// @ingroup dynamic-types
/// An entry in a vtable for defining objects with custom behavior.
///
/// This same structure is used in several places when defining "custom behavior" for objects
/// dynamically at runtime of your C program.  The valid values of `slot`, `flag` and the
/// function-pointer type of `ptr` will vary based on the context you are passing it to.
///
/// Typically, you will defining program statics of tables of these, terminating in the sentinel
/// value `{-1, 0, NULL}`.  Various Qiskit C API functions will take arguments of this form, and
/// return a "vtable" handle back, which can then be used to define "instances" of the object with
/// this attached behavior.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct VtableEntry {
    /// The "slot" of the function, or the sentinel `(uint32_t)-1` to mark the final array entry.
    ///
    /// This is typically set to some `enum` value, where the particular `enum` varies depending on
    /// which vtable you are defining.
    pub slot: u32,
    /// Any additional "flags" for the particular table entry.  These will typically be or'd (`|`)
    /// together, and the valid set of flags will be documented by the table user.
    pub flags: u32,
    /// A function pointer implementing the correct signature for the combination of the `slot` and
    /// `flags`.
    ///
    /// This can be `NULL` only in the case of `slots` being the sentinel `-1`.
    pub ptr: *mut c_void,
}
