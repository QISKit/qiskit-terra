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
    /// A sequence of bits does not hold one bit per element of its shape.
    #[error("a Bit tensor of shape {shape:?} holds {expected} bit(s), not {actual}")]
    BitCount {
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },
    /// A [`Dim::Bounded`] axis reached a position that needs a true size.
    #[error(
        "shape {} has an axis whose size is only bounded above, where a true size is required",
        fmt_shape(shape)
    )]
    DynamicDim { shape: Vec<Dim> },
    /// A ragged axis is not an axis of the shape it is ragged over, or is its last axis.
    ///
    /// The last axis is excluded because that is the one a [`BitTensor`](super::BitTensor) packs
    /// along, so a ragged final axis has no byte boundary to land on.
    #[error(
        "axis {axis} cannot be ragged in a shape of {ndim} dimension(s): \
         a ragged axis must be one of the first {} and not the last",
        ndim.saturating_sub(1)
    )]
    CsrAxis { axis: usize, ndim: usize },
    /// A ragged array's offsets do not describe runs over its bounds.
    #[error("invalid offsets for a ragged array: {reason}")]
    CsrOffsets { reason: String },
    /// An operand is ragged, and the operation is defined only over dense tensors.
    #[error("Tensor::{op} is defined only over dense tensors, and an operand is ragged")]
    RaggedOperand { op: &'static str },
    /// A ragged array's values do not match the runs its offsets describe.
    #[error("a ragged array's values should have shape {expected:?}, got {actual:?}")]
    CsrValues {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}
