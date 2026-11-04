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

use super::error::{MathOpError, check_axis, check_dtype};
use super::inference::{elementwise_binary, elementwise_unary, reduce};
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Tensor, TensorType};
use ndarray::Axis;

/// Return whether a dtype is `Bit`.
#[inline]
fn is_bit(dtype: DType) -> bool {
    dtype == DType::Bit
}

/// Generate a [`ProgramOp`] struct for an elementwise binary bitwise operation on `Bit` tensors.
///
/// `Bit` promotes only to itself, so the promoted dtype is `Bit` exactly when both operands are it.
/// Shapes broadcast, as they do for arithmetic.
macro_rules! bitwise_binary_op {
    ($name:ident, $op_name:literal, $eval_fn:expr) => {
        #[doc = concat!("Elementwise `", $op_name, "` of two `Bit` tensors of identical shape.")]
        #[derive(Clone)]
        pub struct $name;

        impl ProgramOp for $name {
            type Error = MathOpError;

            fn name(&self) -> &str {
                $op_name
            }
            fn namespace(&self) -> &str {
                QISKIT
            }
            fn arity(&self) -> usize {
                2
            }
            fn has_builtin_eval(&self) -> bool {
                true
            }
            fn infer_output_types(
                &self,
                inputs: &[TensorType],
            ) -> Result<Vec<TensorType>, Self::Error> {
                crate::unpack_operands!(self, inputs, [x, y]);
                Ok(vec![elementwise_binary(x, y, is_bit)?])
            }
            fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
                crate::unpack_operands!(self, args, [x, y]);
                check_dtype(0, x.dtype(), DType::Bit)?;
                check_dtype(1, y.dtype(), DType::Bit)?;
                Ok(vec![$eval_fn(x, y)?])
            }
        }
    };
}

bitwise_binary_op!(BitwiseAnd, "bitwise_and", Tensor::bitand_tensor);
bitwise_binary_op!(BitwiseOr, "bitwise_or", Tensor::bitor_tensor);
bitwise_binary_op!(BitwiseXor, "bitwise_xor", Tensor::bitxor_tensor);

/// Elementwise bitwise NOT of a `Bit` tensor.
#[derive(Clone)]
pub struct BitwiseNot;

impl ProgramOp for BitwiseNot {
    type Error = MathOpError;

    fn name(&self) -> &str {
        "bitwise_not"
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
        Ok(vec![elementwise_unary(x, is_bit)?])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x]);
        check_dtype(0, x.dtype(), DType::Bit)?;
        let Tensor::Bit(arr) = x else {
            unreachable!("dtype checked above")
        };
        Ok(vec![Tensor::Bit(arr.mapv(|b| b ^ 1).into_shared())])
    }
}

/// XOR-reduction of a `Bit` tensor along a specified axis, removing that axis.
#[derive(Clone)]
pub struct Parity {
    axis: usize,
}

impl Parity {
    /// Construct a `Parity` op that reduces along `axis`.
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl ProgramOp for Parity {
    type Error = MathOpError;

    fn name(&self) -> &str {
        "parity"
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
        Ok(vec![reduce(x, self.axis, is_bit, |dtype| dtype)?])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x]);
        check_dtype(0, x.dtype(), DType::Bit)?;
        check_axis(self.axis, x.shape().len())?;
        let Tensor::Bit(arr) = x else {
            unreachable!("dtype checked above")
        };
        Ok(vec![Tensor::Bit(
            arr.fold_axis(Axis(self.axis), 0u8, |&acc, &b| acc ^ b)
                .into_shared(),
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim, TensorError};
    use ndarray::{arr1, arr2};

    fn bit(data: &[u8]) -> Tensor {
        Tensor::Bit(arr1(data).into_dyn().into_shared())
    }

    fn bit_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(len)],
        }
    }

    #[test]
    fn test_bitwise_and() {
        let result = BitwiseAnd
            .eval(&[bit(&[1, 0, 1, 1]), bit(&[1, 1, 0, 1])])
            .unwrap();
        assert_eq!(result, vec![bit(&[1, 0, 0, 1])]);
    }

    #[test]
    fn test_bitwise_or() {
        let result = BitwiseOr
            .eval(&[bit(&[1, 0, 1, 0]), bit(&[0, 1, 0, 1])])
            .unwrap();
        assert_eq!(result, vec![bit(&[1, 1, 1, 1])]);
    }

    #[test]
    fn test_bitwise_xor() {
        let result = BitwiseXor
            .eval(&[bit(&[1, 0, 1, 1]), bit(&[1, 1, 0, 1])])
            .unwrap();
        assert_eq!(result, vec![bit(&[0, 1, 1, 0])]);
    }

    #[test]
    fn test_bitwise_not() {
        let result = BitwiseNot.eval(&[bit(&[1, 0, 1, 0])]).unwrap();
        assert_eq!(result, vec![bit(&[0, 1, 0, 1])]);
    }

    #[test]
    fn test_parity_axis0() {
        // [[1,0,1],[0,1,1],[0,0,0]] axis 0 → [1, 1, 0]
        let x = Tensor::Bit(
            arr2(&[[1u8, 0, 1], [0, 1, 1], [0, 0, 0]])
                .into_dyn()
                .into_shared(),
        );
        let result = Parity::new(0).eval(&[x]).unwrap();
        assert_eq!(result, vec![bit(&[1, 1, 0])]);
    }

    #[test]
    fn test_infer_output_types_forwards_the_operand_shape() {
        assert_eq!(
            BitwiseAnd
                .infer_output_types(&[bit_1d(4), bit_1d(4)])
                .unwrap(),
            vec![bit_1d(4)]
        );
        assert_eq!(
            BitwiseNot.infer_output_types(&[bit_1d(4)]).unwrap(),
            vec![bit_1d(4)]
        );
    }

    #[test]
    fn test_bitwise_not_forwards_a_bounded_axis() {
        let bounded = TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Bounded { max: 8 }],
        };
        assert_eq!(
            BitwiseNot
                .infer_output_types(std::slice::from_ref(&bounded))
                .unwrap(),
            vec![bounded]
        );
    }

    #[test]
    fn test_parity_removes_the_reduced_axis() {
        let ty = TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Bounded { max: 4000 }, Dim::Fixed(3)],
        };
        assert_eq!(
            Parity::new(0).infer_output_types(&[ty]).unwrap(),
            vec![bit_1d(3)]
        );
    }

    #[test]
    fn test_a_non_bit_operand_is_rejected_naming_both_dtypes() {
        let f64_1 = TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(1)],
        };
        assert!(matches!(
            BitwiseAnd
                .infer_output_types(&[bit_1d(1), f64_1])
                .unwrap_err(),
            MathOpError::UnsupportedPromotion {
                lhs: DType::Bit,
                rhs: DType::F64,
                dtype: DType::F64,
            }
        ));
    }

    #[test]
    fn test_bitwise_not_names_the_operand_it_rejects() {
        let f64_1 = TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(1)],
        };
        assert!(matches!(
            BitwiseNot.infer_output_types(&[f64_1]).unwrap_err(),
            MathOpError::UnsupportedDType {
                operand: 0,
                dtype: DType::F64,
            }
        ));
    }

    #[test]
    fn test_eval_rejects_a_wrong_dtype() {
        assert!(matches!(
            BitwiseAnd
                .eval(&[Tensor::from([1.0_f64]), bit(&[1])])
                .unwrap_err(),
            MathOpError::WrongDType {
                operand: 0,
                expected: DType::Bit,
                actual: DType::F64,
            }
        ));
    }

    #[test]
    fn test_eval_broadcasts_its_operands() {
        assert_eq!(
            BitwiseAnd.eval(&[bit(&[1, 0, 1]), bit(&[1])]).unwrap(),
            vec![bit(&[1, 0, 1])]
        );
        assert_eq!(
            BitwiseAnd.eval(&[bit(&[1]), bit(&[1, 0, 1])]).unwrap(),
            vec![bit(&[1, 0, 1])],
            "broadcasting is symmetric, unlike ndarray's own operators"
        );
    }

    #[test]
    fn test_eval_rejects_shapes_that_do_not_broadcast() {
        assert!(matches!(
            BitwiseAnd
                .eval(&[bit(&[1, 0, 1]), bit(&[1, 0])])
                .unwrap_err(),
            MathOpError::Tensor(TensorError::ShapeMismatch { lhs, rhs })
                if lhs == [3] && rhs == [2]
        ));
    }

    #[test]
    fn test_parity_axis_out_of_bounds_errors() {
        assert!(matches!(
            Parity::new(1).infer_output_types(&[bit_1d(3)]).unwrap_err(),
            MathOpError::InvalidAxis { axis: 1, ndim: 1 }
        ));
        assert!(matches!(
            Parity::new(1).eval(&[bit(&[1, 0, 1])]).unwrap_err(),
            MathOpError::InvalidAxis { axis: 1, ndim: 1 }
        ));
    }
}
