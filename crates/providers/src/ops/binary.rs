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

use super::error::MathOpError;
use super::inference::{elementwise_binary, promoted_dtype};
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Tensor, TensorType};

/// Generate a [`ProgramOp`] struct for an elementwise binary operation.
///
/// These ops coerce and promote dtypes like NumPy.
macro_rules! elementwise_binary_op {
    ($name:ident, $op_name:literal, $eval_fn:expr, $accepts:expr) => {
        #[doc = concat!(
                    "Elementwise `",
                    $op_name,
                    "` of two tensors, promoting their dtypes and broadcasting their shapes."
                )]
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
                Ok(vec![elementwise_binary(x, y, $accepts)?])
            }
            fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
                crate::unpack_operands!(self, args, [x, y]);
                let dtype = promoted_dtype(x.dtype(), y.dtype(), $accepts)?;
                let (x, y) = (x.clone().cast(dtype), y.clone().cast(dtype));
                Ok(vec![$eval_fn(&x, &y)?])
            }
        }
    };
}

/// Admit every dtype except `Bit`.
#[inline]
fn numeric(dtype: DType) -> bool {
    dtype != DType::Bit
}

/// Admit every real dtype.
fn real(dtype: DType) -> bool {
    !matches!(dtype, DType::Bit | DType::C64 | DType::C128)
}

elementwise_binary_op!(Add, "add", Tensor::add_tensor, numeric);
elementwise_binary_op!(Subtract, "subtract", Tensor::sub_tensor, numeric);
elementwise_binary_op!(Multiply, "multiply", Tensor::mul_tensor, numeric);
elementwise_binary_op!(Divide, "divide", Tensor::div_tensor, numeric);
elementwise_binary_op!(Remainder, "remainder", Tensor::rem_tensor, real);
elementwise_binary_op!(Power, "power", Tensor::pow, numeric);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim, TensorError};

    fn f64_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(len)],
        }
    }

    #[test]
    fn test_add_same_dtype() {
        let result = Add
            .eval(&[
                Tensor::from([1.0_f64, 2.0, 3.0]),
                Tensor::from([4.0_f64, 5.0, 6.0]),
            ])
            .unwrap();
        assert_eq!(result, vec![Tensor::from([5.0_f64, 7.0, 9.0])]);
    }

    #[test]
    fn test_subtract() {
        let result = Subtract
            .eval(&[
                Tensor::from([5.0_f64, 6.0, 7.0]),
                Tensor::from([1.0_f64, 2.0, 3.0]),
            ])
            .unwrap();
        assert_eq!(result, vec![Tensor::from([4.0_f64, 4.0, 4.0])]);
    }

    #[test]
    fn test_multiply() {
        let result = Multiply
            .eval(&[
                Tensor::from([2.0_f64, 3.0, 4.0]),
                Tensor::from([10.0_f64, 10.0, 10.0]),
            ])
            .unwrap();
        assert_eq!(result, vec![Tensor::from([20.0_f64, 30.0, 40.0])]);
    }

    #[test]
    fn test_divide() {
        let result = Divide
            .eval(&[
                Tensor::from([10.0_f64, 9.0, 8.0]),
                Tensor::from([2.0_f64, 3.0, 4.0]),
            ])
            .unwrap();
        assert_eq!(result, vec![Tensor::from([5.0_f64, 3.0, 2.0])]);
    }

    #[test]
    fn test_remainder() {
        let result = Remainder
            .eval(&[
                Tensor::from([7.0_f64, 8.0, 9.0]),
                Tensor::from([3.0_f64, 3.0, 3.0]),
            ])
            .unwrap();
        assert_eq!(result, vec![Tensor::from([1.0_f64, 2.0, 0.0])]);
    }

    #[test]
    fn test_power() {
        let result = Power
            .eval(&[
                Tensor::from([2.0_f64, 3.0, 4.0]),
                Tensor::from([3.0_f64, 2.0, 1.0]),
            ])
            .unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        for (a, b) in arr.as_slice().unwrap().iter().zip(&[8.0_f64, 9.0, 4.0]) {
            assert!(approx::abs_diff_eq!(a, b, epsilon = 1e-12));
        }
    }

    #[test]
    fn test_infer_output_types_forwards_the_operand_type() {
        assert_eq!(
            Add.infer_output_types(&[f64_1d(3), f64_1d(3)]).unwrap(),
            vec![f64_1d(3)]
        );
    }

    #[test]
    fn test_infer_output_types_promotes_differing_dtypes() {
        let f32_3 = TensorType {
            dtype: DType::F32,
            shape: vec![Dim::Fixed(3)],
        };
        assert_eq!(
            Add.infer_output_types(&[f64_1d(3), f32_3]).unwrap(),
            vec![f64_1d(3)]
        );
    }

    #[test]
    fn test_infer_output_types_broadcasts_differing_shapes() {
        assert_eq!(
            Add.infer_output_types(&[f64_1d(3), f64_1d(1)]).unwrap(),
            vec![f64_1d(3)]
        );
    }

    #[test]
    fn test_infer_output_types_rejects_shapes_that_do_not_broadcast() {
        assert!(matches!(
            Add.infer_output_types(&[f64_1d(3), f64_1d(4)]).unwrap_err(),
            MathOpError::Tensor(TensorError::DimShapeMismatch { lhs, rhs })
                if lhs == [Dim::Fixed(3)] && rhs == [Dim::Fixed(4)]
        ));
    }

    #[test]
    fn test_infer_output_types_rejects_two_bounded_axes() {
        let bounded = TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Bounded { max: 8 }],
        };
        assert!(matches!(
            Add.infer_output_types(&[bounded.clone(), bounded.clone()])
                .unwrap_err(),
            MathOpError::Tensor(TensorError::DynamicDim { shape })
                if shape == bounded.shape
        ));
        assert_eq!(
            Add.infer_output_types(&[bounded.clone(), f64_1d(1)])
                .unwrap(),
            vec![bounded]
        );
    }

    #[test]
    fn test_eval_broadcasts_and_promotes_to_match_the_inferred_type() {
        assert_eq!(
            Add.eval(&[Tensor::from([1.0_f64, 2.0]), Tensor::from([10.0_f32])])
                .unwrap(),
            vec![Tensor::from([11.0_f64, 12.0])]
        );
    }

    #[test]
    fn test_a_dtype_the_operation_cannot_compute_is_rejected_when_inferring() {
        // `add` has no `Bit` implementation and `remainder` no complex one.
        let bit = TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(2)],
        };
        assert!(matches!(
            Add.infer_output_types(&[bit.clone(), bit.clone()])
                .unwrap_err(),
            MathOpError::UnsupportedPromotion {
                lhs: DType::Bit,
                rhs: DType::Bit,
                dtype: DType::Bit,
            }
        ));

        let c128 = TensorType {
            dtype: DType::C128,
            shape: vec![Dim::Fixed(2)],
        };
        assert!(matches!(
            Remainder
                .infer_output_types(&[c128.clone(), c128.clone()])
                .unwrap_err(),
            MathOpError::UnsupportedPromotion {
                lhs: DType::C128,
                rhs: DType::C128,
                dtype: DType::C128,
            }
        ));
        assert_eq!(
            Add.infer_output_types(&[c128.clone(), c128.clone()])
                .unwrap(),
            vec![c128]
        );
    }

    #[test]
    fn test_a_bit_operand_is_accepted_where_it_promotes_to_something_computable() {
        // A `Bit` operand is rejected only when the other one is also `Bit`.
        let bit = TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(3)],
        };
        assert_eq!(
            Add.infer_output_types(&[bit, f64_1d(3)]).unwrap(),
            vec![f64_1d(3)]
        );
    }
}
