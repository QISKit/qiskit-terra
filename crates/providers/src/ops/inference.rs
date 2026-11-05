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

//! Various rules for output tensor type inference.

use super::error::MathOpError;
use crate::tensor::{DType, Dim, TensorType, broadcast_dims, broadcast_dims_to, promotion};

/// Which dtypes an op admits.
pub(super) type Accepts = fn(DType) -> bool;

/// How an op's result dtype follows from its operand's.
pub(super) type ResultDType = fn(DType) -> DType;

/// Admit every dtype.
pub(super) fn any(_dtype: DType) -> bool {
    true
}

/// Infer the result type of an elementwise operation over two operands.
///
/// The operand dtypes promote and the shapes broadcast. `accepts` is checked against the promoted
/// dtype rather than against each operand, returning an error if rejected.
pub(super) fn elementwise_binary(
    x: &TensorType,
    y: &TensorType,
    accepts: Accepts,
) -> Result<TensorType, MathOpError> {
    let dtype = promotion(x.dtype, y.dtype);
    if !accepts(dtype) {
        return Err(MathOpError::UnsupportedPromotion {
            lhs: x.dtype,
            rhs: y.dtype,
            dtype,
        });
    }
    Ok(TensorType {
        dtype,
        shape: broadcast_dims(&x.shape, &y.shape)?,
    })
}

/// Infer the result type of an elementwise operation over one operand.
///
/// The dtype and shape are unchanged.
pub(super) fn elementwise_unary(
    x: &TensorType,
    accepts: Accepts,
) -> Result<TensorType, MathOpError> {
    check_accepts(x.dtype, accepts)?;
    Ok(x.clone())
}

/// Infer the result type of a reduction along `axis`.
pub(super) fn reduce(
    x: &TensorType,
    axis: usize,
    accepts: Accepts,
    result_dtype: ResultDType,
) -> Result<TensorType, MathOpError> {
    check_accepts(x.dtype, accepts)?;
    Ok(TensorType {
        dtype: result_dtype(x.dtype),
        shape: super::error::reduced_shape(axis, &x.shape)?,
    })
}

/// Return the tensor type of a dtype cast to `target`.
pub(super) fn cast(x: &TensorType, target: DType) -> Result<TensorType, MathOpError> {
    Ok(TensorType {
        dtype: cast_dtype(x.dtype, target)?,
        shape: x.shape.clone(),
    })
}

/// Return the tensor type of a broadcast to `target`.
pub(super) fn broadcast_to(x: &TensorType, target: &[Dim]) -> Result<TensorType, MathOpError> {
    Ok(TensorType {
        dtype: x.dtype,
        shape: broadcast_dims_to(&x.shape, target)?,
    })
}

/// Validate that a single operand's dtype is admitted.
fn check_accepts(dtype: DType, accepts: Accepts) -> Result<(), MathOpError> {
    if !accepts(dtype) {
        return Err(MathOpError::UnsupportedDType { operand: 0, dtype });
    }
    Ok(())
}

/// Return the dtype a cast from `from` to `to` produces.
///
/// This is the evaluation-time counterpart of [`cast`]. Every cast is supported except from a
/// complex dtype to a real one.
pub(super) fn cast_dtype(from: DType, to: DType) -> Result<DType, MathOpError> {
    let complex = |dtype| matches!(dtype, DType::C64 | DType::C128);
    if complex(from) && !complex(to) {
        return Err(MathOpError::UnsupportedCast { from, to });
    }
    Ok(to)
}

/// Return the dtype an elementwise binary operation computes in.
///
/// This is the evaluation-time counterpart of [`elementwise_binary`]. Evaluation casts both operands
/// to this dtype first.
pub(super) fn promoted_dtype(x: DType, y: DType, accepts: Accepts) -> Result<DType, MathOpError> {
    let dtype = promotion(x, y);
    if !accepts(dtype) {
        return Err(MathOpError::UnsupportedPromotion {
            lhs: x,
            rhs: y,
            dtype,
        });
    }
    Ok(dtype)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::{Dim, TensorError};

    /// Return a `TensorType` of `dtype` over fixed axes `shape`.
    fn ty(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: shape.iter().copied().map(Dim::Fixed).collect(),
        }
    }

    /// Admit only the floating-point dtypes.
    fn floats(dtype: DType) -> bool {
        matches!(dtype, DType::F32 | DType::F64)
    }

    #[test]
    fn test_elementwise_binary_promotes_the_dtypes_and_broadcasts_the_shapes() {
        assert_eq!(
            elementwise_binary(&ty(DType::Bit, &[3, 1]), &ty(DType::F32, &[4]), any).unwrap(),
            ty(DType::F32, &[3, 4])
        );
        assert_eq!(
            elementwise_binary(&ty(DType::I32, &[2]), &ty(DType::U32, &[2]), any).unwrap(),
            ty(DType::I64, &[2])
        );
    }

    #[test]
    fn test_elementwise_binary_admits_the_promoted_dtype_rather_than_each_operand() {
        assert_eq!(
            elementwise_binary(&ty(DType::Bit, &[2]), &ty(DType::F64, &[2]), floats).unwrap(),
            ty(DType::F64, &[2]),
            "a Bit operand beside a float promotes to a float"
        );
        assert!(
            matches!(
                elementwise_binary(&ty(DType::Bit, &[2]), &ty(DType::Bit, &[2]), floats)
                    .unwrap_err(),
                MathOpError::UnsupportedPromotion {
                    lhs: DType::Bit,
                    rhs: DType::Bit,
                    dtype: DType::Bit,
                }
            ),
            "both operand dtypes and the promotion they were refused for are named"
        );
    }

    #[test]
    fn test_elementwise_binary_reports_shapes_that_do_not_broadcast() {
        assert!(matches!(
            elementwise_binary(&ty(DType::F64, &[3]), &ty(DType::F64, &[4]), any).unwrap_err(),
            MathOpError::Tensor(TensorError::DimShapeMismatch { lhs, rhs })
                if lhs == [Dim::Fixed(3)] && rhs == [Dim::Fixed(4)]
        ));
    }

    #[test]
    fn test_elementwise_unary_keeps_the_operand_type() {
        let x = TensorType {
            dtype: DType::C64,
            shape: vec![Dim::Bounded { max: 10 }, Dim::Fixed(2)],
        };
        assert_eq!(elementwise_unary(&x, any).unwrap(), x);

        assert!(matches!(
            elementwise_unary(&x, floats).unwrap_err(),
            MathOpError::UnsupportedDType {
                operand: 0,
                dtype: DType::C64,
            }
        ));
    }

    #[test]
    fn test_reduce_drops_the_axis_and_maps_the_dtype() {
        let x = TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(4), Dim::Bounded { max: 10 }, Dim::Fixed(2)],
        };
        assert_eq!(
            reduce(&x, 1, any, |_| DType::F64).unwrap(),
            ty(DType::F64, &[4, 2])
        );
        assert_eq!(
            reduce(&x, 0, any, |dtype| dtype).unwrap(),
            TensorType {
                dtype: DType::Bit,
                shape: vec![Dim::Bounded { max: 10 }, Dim::Fixed(2)],
            },
            "an axis the reduction keeps is unchanged, bound and all"
        );
    }

    #[test]
    fn test_reduce_reports_an_axis_the_operand_does_not_have() {
        assert!(matches!(
            reduce(&ty(DType::F64, &[2, 3]), 2, any, |dtype| dtype).unwrap_err(),
            MathOpError::InvalidAxis { axis: 2, ndim: 2 }
        ));
    }

    #[test]
    fn test_promoted_dtype_is_the_dtype_elementwise_binary_infers() {
        for (lhs, rhs) in [
            (DType::Bit, DType::F64),
            (DType::I32, DType::U64),
            (DType::F32, DType::C64),
        ] {
            assert_eq!(
                promoted_dtype(lhs, rhs, any).unwrap(),
                elementwise_binary(&ty(lhs, &[2]), &ty(rhs, &[2]), any)
                    .unwrap()
                    .dtype
            );
        }
        assert!(
            matches!(
                promoted_dtype(DType::Bit, DType::Bit, floats).unwrap_err(),
                MathOpError::UnsupportedPromotion {
                    lhs: DType::Bit,
                    rhs: DType::Bit,
                    dtype: DType::Bit,
                }
            ),
            "a promotion the op does not admit is refused at evaluation time too"
        );
    }

    #[test]
    fn test_cast_replaces_the_dtype_and_keeps_the_shape() {
        let shape = vec![Dim::Bounded { max: 10 }, Dim::Fixed(2)];
        assert_eq!(
            cast(
                &TensorType {
                    dtype: DType::Bit,
                    shape: shape.clone(),
                },
                DType::C128
            )
            .unwrap(),
            TensorType {
                dtype: DType::C128,
                shape,
            }
        );
    }

    #[test]
    fn test_cast_refuses_a_complex_operand_and_a_real_target() {
        assert!(matches!(
            cast(&ty(DType::C64, &[2]), DType::F64).unwrap_err(),
            MathOpError::UnsupportedCast {
                from: DType::C64,
                to: DType::F64,
            }
        ));
        assert_eq!(
            cast(&ty(DType::C64, &[2]), DType::C128).unwrap(),
            ty(DType::C128, &[2]),
            "a complex target is admitted in either width"
        );
    }

    #[test]
    fn test_cast_dtype_is_the_dtype_cast_infers() {
        assert_eq!(cast_dtype(DType::Bit, DType::C128).unwrap(), DType::C128);
        assert!(
            matches!(
                cast_dtype(DType::C128, DType::I8).unwrap_err(),
                MathOpError::UnsupportedCast {
                    from: DType::C128,
                    to: DType::I8,
                }
            ),
            "a cast the op does not admit is refused at evaluation time too"
        );
    }

    #[test]
    fn test_broadcast_to_makes_the_target_the_result_shape() {
        let target = [Dim::Fixed(4), Dim::Fixed(2), Dim::Fixed(3)];
        assert_eq!(
            broadcast_to(&ty(DType::F32, &[1, 3]), &target).unwrap(),
            ty(DType::F32, &[4, 2, 3]),
            "the dtype is unchanged and the shape is the target"
        );
        assert!(matches!(
            broadcast_to(&ty(DType::F32, &[2]), &target).unwrap_err(),
            MathOpError::Tensor(TensorError::DimShapeMismatch { lhs, rhs })
                if lhs == [Dim::Fixed(2)] && rhs == target
        ));
    }
}
