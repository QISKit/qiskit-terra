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

use std::ffi::{CStr, CString, c_void};
use std::ptr::NonNull;
use std::sync::LazyLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyModule};
use qiskit_cext_vtable::ExportedFunctions;

struct VTable {
    name: &'static str,
    table: Vec<usize>,
}
impl VTable {
    fn new(name: &'static str, base: &'static ExportedFunctions) -> Self {
        let mut table = vec![0; base.len()];
        for export in base.exports(0) {
            table[export.slot] = export.addr;
        }
        Self { name, table }
    }

    fn attach_pycapsule(&'static self, m: &Bound<'_, PyModule>, modname: &str) -> PyResult<()> {
        // We do `[T]::as_ptr().cast_mut()` here rather than `.as_mut_ptr()` to avoid, even
        // temporarily, breaking Rust's rules on aliasing mutable references; we _cannot_ retrieve a
        // safe `&mut T` pointer out of the static, so we mustn't attempt it.  It's fine to hold a
        // mutable pointer, though - it's only UB to attempt to mutate through that.
        let ptr = NonNull::new(self.table.as_ptr().cast_mut().cast::<c_void>())
            .expect("slices should be backed by non-null pointers");
        let last_modname = modname
            .rsplit_once(".")
            .map(|(_, last)| last)
            .unwrap_or(modname);
        let base_modname = m.name()?;
        if base_modname != last_modname {
            return Err(PyValueError::new_err(format!(
                "internal error: mismatched names between module ('{base_modname}') and requested submodule ('{modname}')"
            )));
        }

        let fullname = {
            // We need to leak the CString because Python needs the name to last until the end of
            // time.
            let alloc = CString::new(format!("{}.{}", modname, self.name))?;
            // SAFETY: the input to `from_ptr` is a pointer from a `CString` created above.
            unsafe { CStr::from_ptr::<'static>(alloc.into_raw()) }
        };

        // SAFETY: the pointer is to static-lifetimed data, and no destructor is necessary (only the
        // complete memory-space cleanup on process termination.
        let capsule = unsafe { PyCapsule::new_with_pointer(m.py(), ptr, fullname) }?;
        m.add(self.name, capsule)?;
        Ok(())
    }
}

static FFI_TRANSPILE: LazyLock<VTable> =
    LazyLock::new(|| VTable::new("QK_FFI_TRANSPILE", &qiskit_cext_vtable::FUNCTIONS_TRANSPILE));
static FFI_CIRCUIT: LazyLock<VTable> =
    LazyLock::new(|| VTable::new("QK_FFI_CIRCUIT", &qiskit_cext_vtable::FUNCTIONS_CIRCUIT));
static FFI_QI: LazyLock<VTable> =
    LazyLock::new(|| VTable::new("QK_FFI_QI", &qiskit_cext_vtable::FUNCTIONS_QI));
static FFI_MIR: LazyLock<VTable> =
    LazyLock::new(|| VTable::new("QK_FFI_MIR", &qiskit_cext_vtable::FUNCTIONS_MIR));

#[pymodule(name = "capi")]
pub fn capi_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    static MODNAME: &str = "qiskit._accelerate.capi";

    m.add("API_VERSION", qiskit_cext::qk_api_version())?;
    FFI_TRANSPILE.attach_pycapsule(m, MODNAME)?;
    FFI_CIRCUIT.attach_pycapsule(m, MODNAME)?;
    FFI_QI.attach_pycapsule(m, MODNAME)?;
    FFI_MIR.attach_pycapsule(m, MODNAME)?;
    Ok(())
}
