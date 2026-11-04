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

//! The tensor domain's error type, shared by tensor operations and by the type-level rules.

use thiserror::Error;

use super::tensor_type::fmt_shape;
use super::{DType, Dim};

/// Errors returned by tensor operations.
#[derive(Debug, Error)]
pub enum TensorError {
    /// The two operand tensors have different dtypes.
    #[error("dtype mismatch in Tensor::{op}: lhs={lhs}, rhs={rhs}")]
    DTypeMismatch {
        op: &'static str,
        lhs: DType,
        rhs: DType,
    },
    /// The operand does not support the dtype.
    #[error("Tensor::{op} does not support dtype {dtype}")]
    UnsupportedDType { op: &'static str, dtype: DType },
    /// The two operand shapes are not broadcast-compatible.
    #[error("shapes {lhs:?} and {rhs:?} are not broadcast-compatible")]
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },
    /// The two operand [`Dim`] shapes are not broadcast-compatible.
    #[error(
        "shapes {} and {} are not broadcast-compatible",
        fmt_shape(lhs),
        fmt_shape(rhs)
    )]
    DimShapeMismatch { lhs: Vec<Dim>, rhs: Vec<Dim> },
    /// An exponent of an integer dtype is negative.
    #[error("an exponent of dtype {dtype} cannot be negative")]
    NegativeExponent { dtype: DType },
    /// A [`Dim::Bounded`] axis reached a position that needs a true size.
    #[error(
        "shape {} has an axis whose size is only bounded above, where a true size is required",
        fmt_shape(shape)
    )]
    DynamicDim { shape: Vec<Dim> },
}
