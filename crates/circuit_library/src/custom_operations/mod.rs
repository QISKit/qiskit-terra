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

//! Custom operations implemented natively in Rust.
//!
//! Each operation implements [`qiskit_circuit::operations::CustomOperation`]. Operations that also
//! have a Python representation implement [`PyConvertible`], which provides the conversions between
//! the Rust and Python representations.
//!
//! This module registers those conversions with `qiskit-circuit`. To add an operation, implement it
//! in a submodule, implement [`PyConvertible`] if it has a Python representation, and add its
//! conversion functions to the appropriate tables below.

use std::any::TypeId;

use pyo3::prelude::*;

use qiskit_circuit::py_convertible::{
    ConversionFromPythonEntry, ConversionToPythonEntry, create_py_op_for, extract_from_py_for,
    register_conversions_from_python, register_conversions_to_python,
};

pub mod qft;

// Operations with a Python representation need an entry in both tables. The Rust type identifies
// the operation in the Rust-to-Python direction; the Python operation name identifies it in the
// Python-to-Rust direction.

/// All Rust-to-Python conversions provided by this crate.
static CONVERSIONS_TO_PYTHON_TABLE: &[ConversionToPythonEntry] = &[ConversionToPythonEntry {
    type_id: TypeId::of::<qft::QftGate>,
    create: create_py_op_for::<qft::QftGate>,
}];

/// All Python-to-Rust conversions provided by this crate.
static CONVERSIONS_FROM_PYTHON_TABLE: &[ConversionFromPythonEntry] = &[ConversionFromPythonEntry {
    name: "qft",
    extract: extract_from_py_for::<qft::QftGate>,
}];

/// Register the Python conversions for all custom operations in this crate.
///
/// This must be called during module initialization before any custom operation crosses the
/// Python boundary.
pub fn register_custom_operations() -> PyResult<()> {
    register_conversions_to_python(CONVERSIONS_TO_PYTHON_TABLE);
    register_conversions_from_python(CONVERSIONS_FROM_PYTHON_TABLE);
    Ok(())
}
