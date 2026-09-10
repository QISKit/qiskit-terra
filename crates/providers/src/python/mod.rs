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

//! Bindings for the `qiskit.quantum_program` Python package.

mod backend;
mod data_tree;
mod ops;
mod program;
mod stepper;
mod tensor;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use crate::program::InstructionRole;
use crate::tensor::{DType, TensorType};
//pub use backend::PyBackendV3;
pub use data_tree::PyDataTree;
use ops::{
    PyAdd, PyBindParameters, PyBitwiseAnd, PyBitwiseNot, PyBitwiseOr, PyBitwiseXor, PyBroadcastTo,
    PyCast, PyConstant, PyDivide, PyMean, PyMultiply, PyParity, PyPower, PyProgramOp, PyRemainder,
    PyShotLoop, PyStd, PySubtract, PyVariance,
};
use program::{PyFunctionBuilder, PyInstruction, PyProgramFunction, PyQuantumProgram, PyValue};
use stepper::{PyProgramStepper, PyRequest};
use tensor::PyBounded;

/// Return `error` and everything that caused it, as one message.
///
/// Nothing sets a Python exception's cause here, so each source is appended to the message instead.
fn chain(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(error) = source {
        message.push_str(": ");
        message.push_str(&error.to_string());
        source = error.source();
    }
    message
}

/// Return `error` as a `ValueError`.
fn value_error(error: &dyn std::error::Error) -> PyErr {
    PyValueError::new_err(chain(error))
}

/// Return the position `index` addresses among `length` items of kind `what`, counting from the end
/// when `index` is negative.
pub(super) fn position(index: isize, length: usize, what: &str) -> PyResult<usize> {
    let refuse = || PyIndexError::new_err(format!("{what} {index} is out of range"));
    let length = isize::try_from(length).map_err(|_| refuse())?;
    let shifted = if index < 0 { index + length } else { index };
    (0..length)
        .contains(&shifted)
        .then(|| usize::try_from(shifted).expect("a position in range is not negative"))
        .ok_or_else(refuse)
}

/// Register the `quantum_program` submodule.
pub fn quantum_program(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBounded>()?;
    m.add_class::<PyDataTree>()?;
    m.add_class::<PyFunctionBuilder>()?;
    m.add_class::<PyInstruction>()?;
    m.add_class::<PyProgramFunction>()?;
    m.add_class::<PyProgramStepper>()?;
    m.add_class::<PyQuantumProgram>()?;
    m.add_class::<PyRequest>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<InstructionRole>()?;
    m.add_class::<DType>()?;
    m.add_class::<TensorType>()?;
    // The op catalogue, whose base class must be registered before the classes extending it.
    m.add_class::<PyProgramOp>()?;
    m.add_class::<PyAdd>()?;
    m.add_class::<PyBindParameters>()?;
    m.add_class::<PyBitwiseAnd>()?;
    m.add_class::<PyBitwiseNot>()?;
    m.add_class::<PyBitwiseOr>()?;
    m.add_class::<PyBitwiseXor>()?;
    m.add_class::<PyBroadcastTo>()?;
    m.add_class::<PyCast>()?;
    m.add_class::<PyConstant>()?;
    m.add_class::<PyDivide>()?;
    m.add_class::<PyMean>()?;
    m.add_class::<PyMultiply>()?;
    m.add_class::<PyParity>()?;
    m.add_class::<PyPower>()?;
    m.add_class::<PyRemainder>()?;
    m.add_class::<PyShotLoop>()?;
    m.add_class::<PyStd>()?;
    m.add_class::<PySubtract>()?;
    m.add_class::<PyVariance>()?;
    Ok(())
}
