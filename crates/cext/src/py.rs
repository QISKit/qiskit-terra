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

// The `boolean_struct` module here is `doc(hidden)` but `pub`.  We need it to specify the trait
// bound correctly to make `borrow_mut` available.
use pyo3::PyClass;
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::False;

/// Borrow a pointer to a Rust-native object extracted from a Python object.
///
/// The `map_fn` projects the desired reference from a temporary one.  This is useful in cases where
/// the object exposed to Python contains the raw Rust type (for example, `PySparseObservable`
/// contains `SparseObservable`).  We do this with a `map_fn` so we can control the safety of the
/// lifetime of the temporary reference.
///
/// You can use [`borrow_map_mut`] if you need mutable access to the underlying Rust struct.
///
/// The returned pointer derives its lifetime from the lifetime of `ob`.  The reference `ob` is only
/// borrowed by the function.
///
/// If the `PyObject` is not of the correct type or the `map_fn` returns an error variant, the null
/// pointer is returned and the Python exception state is set.
///
/// # Safety
///
/// `ob` must point to a valid `PyObject`.
pub unsafe fn borrow_map<'py, T, S>(
    py: Python<'py>,
    ob: *mut ::pyo3::ffi::PyObject,
    map_fn: impl for<'a> FnOnce(Python<'py>, &'a T) -> PyResult<&'a S>,
) -> *const S
where
    T: PyClass,
{
    let borrow_map = || -> PyResult<*const S> {
        // SAFETY: per documentation, `ob` points to a valid PyObject.  The lifetime of the
        // `Borrowed` is valid because we either drop it or consume it back into a pointer before
        // the function returns.
        let ob = unsafe { Borrowed::from_ptr(py, ob) }.cast::<T>()?;
        let handle = &*ob.borrow();
        map_fn(py, handle).map(::std::ptr::from_ref)
    };
    borrow_map().unwrap_or_else(|e| {
        e.restore(py);
        ::std::ptr::null()
    })
}

/// Borrow a pointer to a Rust-native object from a Python object.
///
/// The returned pointer derives its lifetime from the lifetime of `ob`.  The reference `ob` is only
/// borrowed by the function.
///
/// If the `PyObject` is not of the correct type, the null pointer is returned and the Python
/// exception state is set.
///
/// # Safety
///
/// `ob` must point to a valid `PyObject`.
pub unsafe fn borrow_mut<T>(py: Python, ob: *mut ::pyo3::ffi::PyObject) -> *mut T
where
    T: PyClass<Frozen = False>,
{
    // SAFETY: per documentation, `ob` satisfies the same requirements as it does in `borrow_map_mut`.
    unsafe { borrow_map_mut::<T, T>(py, ob, |_py, x| Ok(x)) }
}

/// Borrow a pointer to a Rust-native object extracted from a Python object.
///
/// The `map_fn` projects the desired reference from a temporary one.  This is useful in cases where
/// the object exposed to Python contains the raw Rust type (for example, `PySparseObservable`
/// contains `SparseObservable`).  We do this with a `map_fn` so we can control the safety of the
/// lifetime of the temporary reference.
///
/// You can use [`borrow_map`] if you don't need mutable access, which makes it possible to use this
/// with structs marked `pyclass(frozen)`.
///
/// The returned pointer derives its lifetime from the lifetime of `ob`.  The reference `ob` is only
/// borrowed by the function.
///
/// If the `PyObject` is not of the correct type or the `map_fn` returns an error variant, the null
/// pointer is returned and the Python exception state is set.
///
/// # Safety
///
/// `ob` must point to a valid `PyObject`.
pub unsafe fn borrow_map_mut<'py, T, S>(
    py: Python<'py>,
    ob: *mut ::pyo3::ffi::PyObject,
    map_fn: impl for<'a> FnOnce(Python<'py>, &'a mut T) -> PyResult<&'a mut S>,
) -> *mut S
where
    T: PyClass<Frozen = False>,
{
    let borrow_map = || -> PyResult<*mut S> {
        // SAFETY: per documentation, `ob` points to a valid PyObject.  The lifetime of the
        // `Borrowed` is valid because we either drop it or consume it back into a pointer before
        // the function returns.
        let ob = unsafe { Borrowed::from_ptr(py, ob) }.cast::<T>()?;
        let handle = &mut *ob.borrow_mut();
        map_fn(py, handle).map(::std::ptr::from_mut)
    };
    borrow_map().unwrap_or_else(|e| {
        e.restore(py);
        ::std::ptr::null_mut()
    })
}

/// Extract a pointer to a Rust-native object from a `PyObject`, storing the result in `address`.
///
/// The exact object stored can be extracted from a `PyClass` by projecting a reference out of some
/// outer Python-exposed type using `map_fn`.  For example, the `object` might be a
/// `PySparseObservable`, but the `map_fn` extracts a reference to the inner `SparseObservable`.
///
/// This is used to define Python-space "converter" functions for use with the `PyArg_Parse*` family
/// of functions.
///
/// On success, returns 1 and writes out the pointer in `address`.  On failure, returns 0, sets the
/// Python exception state and leaves `address` untouched.
///
/// # Safety
///
/// `object` must point to a valid `PyObject`.  `address` must point to enough space to write a
/// pointer to.
pub unsafe fn convert_map<'py, T, S>(
    py: Python<'py>,
    object: *mut ::pyo3::ffi::PyObject,
    address: *mut ::std::ffi::c_void,
    map_fn: impl for<'a> FnOnce(Python<'py>, &'a T) -> PyResult<&'a S>,
) -> ::std::ffi::c_int
where
    T: PyClass,
{
    let native = unsafe { borrow_map(py, object, map_fn) };
    if native.is_null() {
        0
    } else {
        unsafe { address.cast::<*const S>().write(native) };
        1
    }
}

/// Extract a pointer to a Rust-native object from a PyObject representing a PyClass, storing the
/// result in `address`.
///
/// This is used to define Python-space "converter" functions for use with the `PyArg_Parse*` family
/// of functions.
///
/// On success, returns 1 and writes out the pointer in `address`.  On failure, returns 0, sets the
/// Python exception state and leaves `address` untouched.
///
/// # Safety
///
/// `object` must point to a valid PyObject.  `address` must point to enough space to write a
/// pointer to.
pub unsafe fn convert_mut<T>(
    py: Python,
    object: *mut ::pyo3::ffi::PyObject,
    address: *mut ::std::ffi::c_void,
) -> ::std::ffi::c_int
where
    T: PyClass<Frozen = False>,
{
    // SAFETY: per documentation, `object` and `address` satisfy the same requirements as
    // `convert_map`.
    unsafe { convert_map_mut::<T, T>(py, object, address, |_py, x| Ok(x)) }
}

/// Extract a pointer to a Rust-native object from a `PyObject`, storing the result in `address`.
///
/// The exact object stored can be extracted from a `PyClass` by projecting a reference out of some
/// outer Python-exposed type using `map_fn`.  For example, the `object` might be a
/// `PySparseObservable`, but the `map_fn` extracts a reference to the inner `SparseObservable`.
///
/// This is used to define Python-space "converter" functions for use with the `PyArg_Parse*` family
/// of functions.
///
/// On success, returns 1 and writes out the pointer in `address`.  On failure, returns 0, sets the
/// Python exception state and leaves `address` untouched.
///
/// # Safety
///
/// `object` must point to a valid `PyObject`.  `address` must point to enough space to write a
/// pointer to.
pub unsafe fn convert_map_mut<'py, T, S>(
    py: Python<'py>,
    object: *mut ::pyo3::ffi::PyObject,
    address: *mut ::std::ffi::c_void,
    map_fn: impl for<'a> FnOnce(Python<'py>, &'a mut T) -> PyResult<&'a mut S>,
) -> ::std::ffi::c_int
where
    T: PyClass<Frozen = False>,
{
    let native = unsafe { borrow_map_mut(py, object, map_fn) };
    if native.is_null() {
        0
    } else {
        unsafe { address.cast::<*mut S>().write(native) };
        1
    }
}
