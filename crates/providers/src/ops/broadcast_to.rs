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

//! The op that broadcasts an operand to a chosen shape.

use super::error::MathOpError;
use super::inference::broadcast_to;
use super::{ProgramOp, QISKIT};
use crate::tensor::{Dim, Tensor, TensorError, TensorType};

/// Broadcast a tensor to a target shape, right-aligning the two.
#[derive(Clone)]
pub struct BroadcastTo {
    target: Vec<Dim>,
}

impl BroadcastTo {
    /// Construct a `BroadcastTo` op producing `target`.
    pub fn new(target: Vec<Dim>) -> Self {
        Self { target }
    }

    /// Return the shape an operand of shape `shape` is broadcast to.
    ///
    /// A fixed target axis gives its own size. A bounded one takes the operand's size along the axis
    /// it aligns with.
    fn concrete_target(&self, shape: &[usize]) -> Result<Vec<usize>, MathOpError> {
        let Some(offset) = self.target.len().checked_sub(shape.len()) else {
            return Err(TensorError::DimShapeMismatch {
                lhs: shape.iter().copied().map(Dim::Fixed).collect(),
                rhs: self.target.clone(),
            }
            .into());
        };
        self.target
            .iter()
            .enumerate()
            .map(|(axis, dim)| match *dim {
                Dim::Fixed(size) => Ok(size),
                Dim::Bounded { .. } if axis >= offset => Ok(shape[axis - offset]),
                Dim::Bounded { .. } => Err(TensorError::DynamicDim {
                    shape: self.target.clone(),
                }
                .into()),
            })
            .collect()
    }
}

impl ProgramOp for BroadcastTo {
    type Error = MathOpError;

    fn name(&self) -> &str {
        "broadcast_to"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn arity(&self) -> usize {
        1
    }
    fn has_builtin_eval(&self) -> bool {
        true
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        crate::unpack_operands!(self, inputs, [x]);
        Ok(vec![broadcast_to(x, &self.target)?])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x]);
        Ok(vec![x.broadcast_to(&self.concrete_target(x.shape())?)?])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::DType;
    use ndarray::arr2;

    /// Return a `TensorType` over `shape`.
    fn ty(shape: Vec<Dim>) -> TensorType {
        TensorType {
            dtype: DType::F64,
            shape,
        }
    }

    /// Return a shape of fixed axes.
    fn fixed(sizes: &[usize]) -> Vec<Dim> {
        sizes.iter().copied().map(Dim::Fixed).collect()
    }

    #[test]
    fn test_broadcast_to_full_name_and_arity() {
        let op = BroadcastTo::new(fixed(&[3]));
        assert_eq!(op.full_name(), "qiskit.broadcast_to");
        assert_eq!(op.arity(), 1);
        assert!(op.has_builtin_eval());
    }

    #[test]
    fn test_infer_output_types_is_the_target_and_keeps_the_dtype() {
        // An axis of size one reaches the target's size, and a missing leading axis is added.
        for shape in [fixed(&[1, 5]), fixed(&[5]), fixed(&[3, 5])] {
            assert_eq!(
                BroadcastTo::new(fixed(&[3, 5]))
                    .infer_output_types(&[ty(shape.clone())])
                    .unwrap(),
                vec![ty(fixed(&[3, 5]))],
                "for operand shape {shape:?}"
            );
        }
        let bit = TensorType {
            dtype: DType::Bit,
            shape: fixed(&[1]),
        };
        assert_eq!(
            BroadcastTo::new(fixed(&[4]))
                .infer_output_types(&[bit])
                .unwrap()[0]
                .dtype,
            DType::Bit
        );
    }

    #[test]
    fn test_infer_output_types_copies_a_bounded_axis_from_the_operand() {
        let shots = Dim::Bounded { max: 4000 };
        assert_eq!(
            BroadcastTo::new(vec![shots, Dim::Fixed(5)])
                .infer_output_types(&[ty(vec![shots, Dim::Fixed(1)])])
                .unwrap(),
            vec![ty(vec![shots, Dim::Fixed(5)])]
        );
    }

    #[test]
    fn test_infer_output_types_rejects_a_target_it_cannot_reach() {
        let shots = Dim::Bounded { max: 4000 };
        assert!(matches!(
            BroadcastTo::new(fixed(&[4]))
                .infer_output_types(&[ty(fixed(&[3]))])
                .unwrap_err(),
            MathOpError::Tensor(TensorError::DimShapeMismatch { lhs, rhs })
                if lhs == fixed(&[3]) && rhs == fixed(&[4])
        ));
        assert!(matches!(
            BroadcastTo::new(fixed(&[5]))
                .infer_output_types(&[ty(vec![shots])])
                .unwrap_err(),
            MathOpError::Tensor(TensorError::DynamicDim { shape }) if shape == [shots]
        ));
        assert!(matches!(
            BroadcastTo::new(vec![shots])
                .infer_output_types(&[ty(fixed(&[1]))])
                .unwrap_err(),
            MathOpError::Tensor(TensorError::DynamicDim { shape }) if shape == [shots]
        ));
    }

    #[test]
    fn test_eval_duplicates_the_operand() {
        let x = Tensor::from([1.0_f64, 2.0, 3.0]);
        assert_eq!(
            BroadcastTo::new(fixed(&[2, 3])).eval(&[x]).unwrap(),
            vec![Tensor::F64(
                arr2(&[[1.0_f64, 2.0, 3.0], [1.0, 2.0, 3.0]])
                    .into_dyn()
                    .into_shared()
            )]
        );
    }

    #[test]
    fn test_eval_takes_a_bounded_axis_size_from_the_operand() {
        let x = Tensor::F64(arr2(&[[1.0_f64], [2.0], [3.0]]).into_dyn().into_shared());
        let op = BroadcastTo::new(vec![Dim::Bounded { max: 4000 }, Dim::Fixed(2)]);
        assert_eq!(
            op.eval(&[x]).unwrap(),
            vec![Tensor::F64(
                arr2(&[[1.0_f64, 1.0], [2.0, 2.0], [3.0, 3.0]])
                    .into_dyn()
                    .into_shared()
            )]
        );
    }

    #[test]
    fn test_eval_rejects_an_operand_that_does_not_fit_the_target() {
        assert!(matches!(
            BroadcastTo::new(fixed(&[4]))
                .eval(&[Tensor::from([1.0_f64, 2.0, 3.0])])
                .unwrap_err(),
            MathOpError::Tensor(TensorError::ShapeMismatch { lhs, rhs })
                if lhs == [3] && rhs == [4]
        ));
        assert!(matches!(
            BroadcastTo::new(fixed(&[3]))
                .eval(&[Tensor::F64(
                    arr2(&[[1.0_f64], [2.0]]).into_dyn().into_shared()
                )])
                .unwrap_err(),
            MathOpError::Tensor(TensorError::DimShapeMismatch { lhs, rhs })
                if lhs == fixed(&[2, 1]) && rhs == fixed(&[3])
        ));
        let shots = Dim::Bounded { max: 4000 };
        assert!(matches!(
            BroadcastTo::new(vec![shots, Dim::Fixed(3)])
                .eval(&[Tensor::from([1.0_f64, 2.0, 3.0])])
                .unwrap_err(),
            MathOpError::Tensor(TensorError::DynamicDim { shape })
                if shape == [shots, Dim::Fixed(3)]
        ));
    }
}
