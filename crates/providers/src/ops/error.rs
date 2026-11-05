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

//! Errors raised by ops.

use crate::tensor::{DType, Dim, TensorError};
use thiserror::Error;

/// Errors returned by the [`ProgramOp`](super::ProgramOp) implementations Qiskit defines.
///
/// Operands are positional, so an offending one is named by index. The op itself is named by
/// whatever wraps the error: a [`ProgramFunction`](crate::ProgramFunction) attaches its type name.
#[derive(Debug, Error)]
pub enum MathOpError {
    /// An operand's dtype is not the single one this op accepts.
    #[error("operand {operand}: expected dtype {expected}, got {actual}")]
    WrongDType {
        operand: usize,
        expected: DType,
        actual: DType,
    },
    /// An operand's dtype is not among those this op accepts.
    #[error("operand {operand}: dtype {dtype} is not supported")]
    UnsupportedDType { operand: usize, dtype: DType },
    /// Two operands promote to a dtype this op does not compute.
    #[error("operands of dtype {lhs} and {rhs} promote to {dtype}, which is not supported")]
    UnsupportedPromotion {
        lhs: DType,
        rhs: DType,
        dtype: DType,
    },
    /// Complex dtypes cannot be cast to real dtypes.
    #[error("cannot cast {from} to {to}")]
    UnsupportedCast { from: DType, to: DType },
    /// A tensor operation failed (dtype or shape mismatch).
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// The requested axis was out of bounds for the tensor's number of dimensions.
    #[error("axis {axis} is out of bounds for tensor with {ndim} dimension(s)")]
    InvalidAxis { axis: usize, ndim: usize },
}

/// Validate that an operand's dtype is `expected`, naming the offending operand by its position.
pub(super) fn check_dtype(
    operand: usize,
    actual: DType,
    expected: DType,
) -> Result<(), MathOpError> {
    if actual != expected {
        return Err(MathOpError::WrongDType {
            operand,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Validate that `axis` is a valid axis index for a tensor with `ndim` dimensions.
pub(super) fn check_axis(axis: usize, ndim: usize) -> Result<(), MathOpError> {
    if axis >= ndim {
        return Err(MathOpError::InvalidAxis { axis, ndim });
    }
    Ok(())
}

/// Remove `axis` from a shape.
pub(super) fn reduced_shape(axis: usize, shape: &[Dim]) -> Result<Vec<Dim>, MathOpError> {
    check_axis(axis, shape.len())?;
    let mut shape = shape.to_vec();
    shape.remove(axis);
    Ok(shape)
}
