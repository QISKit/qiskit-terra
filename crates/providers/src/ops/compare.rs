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

//! The ops that compare two tensors element-wise, producing bits.

use super::error::MathOpError;
use super::inference::{elementwise_binary, promoted_dtype};
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Tensor, TensorType};

/// Generate a [`ProgramOp`] for an element-wise comparison of two tensors.
///
/// A comparison promotes and broadcasts its operands and produces `Bit`.
macro_rules! compare_op {
    ($name:ident, $op_name:literal, $tensor_method:ident, $accepts:expr) => {
        #[doc = concat!(
                    "Element-wise `",
                    $op_name,
                    "` of two tensors, promoting their dtypes and broadcasting their shapes.\n\n",
                    "The result is always `Bit`."
                )]
        #[derive(Clone, Debug)]
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
                Ok(vec![TensorType {
                    dtype: DType::Bit,
                    ..elementwise_binary(x, y, $accepts)?
                }])
            }
            fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
                crate::unpack_operands!(self, args, [x, y]);
                // promote first, the same way the arithmetic ops do.
                let dtype = promoted_dtype(x.dtype(), y.dtype(), $accepts)?;
                let (x, y) = (x.clone().cast(dtype), y.clone().cast(dtype));
                Ok(vec![Tensor::$tensor_method(&x, &y)?])
            }
        }
    };
}

/// Admit every dtype.
#[inline]
fn any(_dtype: DType) -> bool {
    true
}

/// Admit every dtype that has a natural order.
#[inline]
fn ordered(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F64
            | DType::F32
            | DType::I64
            | DType::I32
            | DType::I16
            | DType::I8
            | DType::U64
            | DType::U32
            | DType::U16
            | DType::U8
            | DType::Bit
    )
}

compare_op!(Equal, "equal", eq_tensor, any);
compare_op!(NotEqual, "not_equal", ne_tensor, any);
compare_op!(Less, "less", lt_tensor, ordered);
compare_op!(LessEqual, "less_equal", le_tensor, ordered);
compare_op!(Greater, "greater", gt_tensor, ordered);
compare_op!(GreaterEqual, "greater_equal", ge_tensor, ordered);

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::{BitTensor, Dim};

    /// Return a `TensorType` of `dtype` over fixed axes `shape`.
    fn ty(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: shape.iter().copied().map(Dim::Fixed).collect(),
        }
    }

    /// Return a 1-D `Bit` tensor holding `bits`.
    fn bits(values: &[u8]) -> Tensor {
        bit_tensor(values, &[values.len()])
    }

    /// Return a `Bit` tensor of logical shape `shape` over `values`, one bit per entry.
    fn bit_tensor(values: &[u8], shape: &[usize]) -> Tensor {
        Tensor::Bit(BitTensor::from_bools(values.iter().map(|&v| v == 1), shape).unwrap())
    }

    /// Return the single output of `op` applied to `x` and `y`.
    fn compared(op: &impl ProgramOp<Error = MathOpError>, x: Tensor, y: Tensor) -> Tensor {
        op.eval(&[x, y]).unwrap().into_iter().next().unwrap()
    }

    #[test]
    fn test_compare_full_names_and_arity() {
        assert_eq!(Equal.full_name(), "qiskit.equal");
        assert_eq!(NotEqual.full_name(), "qiskit.not_equal");
        assert_eq!(Less.full_name(), "qiskit.less");
        assert_eq!(LessEqual.full_name(), "qiskit.less_equal");
        assert_eq!(Greater.full_name(), "qiskit.greater");
        assert_eq!(GreaterEqual.full_name(), "qiskit.greater_equal");
        assert_eq!(Equal.arity(), 2);
        assert!(Equal.has_builtin_eval());
    }

    #[test]
    fn test_infer_output_types_is_always_bit() {
        // Whatever the operands, the answer to a comparison is a bit.
        for operands in [DType::F64, DType::I32, DType::U8, DType::Bit] {
            assert_eq!(
                Equal
                    .infer_output_types(&[ty(operands, &[3]), ty(operands, &[3])])
                    .unwrap(),
                vec![ty(DType::Bit, &[3])],
                "comparing two {operands} operands"
            );
        }
    }

    #[test]
    fn test_infer_output_types_promotes_and_broadcasts_like_arithmetic() {
        // The dtypes still promote, which is what decides the comparison's own arithmetic, and the
        // shapes still broadcast -- only the result dtype differs from an arithmetic op.
        assert_eq!(
            Less.infer_output_types(&[ty(DType::I32, &[3, 1]), ty(DType::F32, &[4])])
                .unwrap(),
            vec![ty(DType::Bit, &[3, 4])]
        );
    }

    #[test]
    fn test_infer_output_types_keeps_a_bounded_axis() {
        let bounded = |dtype| TensorType {
            dtype,
            shape: vec![Dim::Bounded { max: 100 }],
        };
        assert_eq!(
            Equal
                .infer_output_types(&[bounded(DType::Bit), ty(DType::Bit, &[1])])
                .unwrap(),
            vec![bounded(DType::Bit)],
            "an axis of unknown length is still unknown after comparing"
        );
    }

    #[test]
    fn test_an_ordering_refuses_the_complex_dtypes() {
        // A complex number has no natural order, so these have no meaning; equality still does.
        assert!(matches!(
            Less.infer_output_types(&[ty(DType::C128, &[2]), ty(DType::C128, &[2])])
                .unwrap_err(),
            MathOpError::UnsupportedPromotion {
                dtype: DType::C128,
                ..
            }
        ));
        assert_eq!(
            Equal
                .infer_output_types(&[ty(DType::C128, &[2]), ty(DType::C128, &[2])])
                .unwrap(),
            vec![ty(DType::Bit, &[2])]
        );
    }

    #[test]
    fn test_eval_compares_elementwise() {
        let x = Tensor::from([1.0_f64, 2.0, 3.0]);
        let y = Tensor::from([3.0_f64, 2.0, 1.0]);
        assert_eq!(compared(&Equal, x.clone(), y.clone()), bits(&[0, 1, 0]));
        assert_eq!(compared(&NotEqual, x.clone(), y.clone()), bits(&[1, 0, 1]));
        assert_eq!(compared(&Less, x.clone(), y.clone()), bits(&[1, 0, 0]));
        assert_eq!(compared(&LessEqual, x.clone(), y.clone()), bits(&[1, 1, 0]));
        assert_eq!(compared(&Greater, x.clone(), y.clone()), bits(&[0, 0, 1]));
        assert_eq!(compared(&GreaterEqual, x, y), bits(&[0, 1, 1]));
    }

    #[test]
    fn test_eval_promotes_mismatched_dtypes() {
        // An integer beside a float promotes to the float, so the comparison is exact rather than
        // truncating.
        assert_eq!(
            compared(
                &Equal,
                Tensor::from([1_i32, 2, 3]),
                Tensor::from([1.0_f64, 2.5, 3.0])
            ),
            bits(&[1, 0, 1])
        );
    }

    #[test]
    fn test_eval_broadcasts_a_scalar_operand() {
        // The post-selection shape: compare a reduced quantity against a single threshold.
        assert_eq!(
            compared(&Equal, Tensor::from([0_u8, 1, 0, 1]), Tensor::from([0_u8])),
            bits(&[1, 0, 1, 0])
        );
    }

    #[test]
    fn test_eval_result_satisfies_the_inferred_type() {
        let operands = [ty(DType::I32, &[4]), ty(DType::I32, &[1])];
        let inferred = Greater.infer_output_types(&operands).unwrap();
        let out = compared(
            &Greater,
            Tensor::from([1_i32, 5, 3, 9]),
            Tensor::from([3_i32]),
        );
        assert!(out.matches(&inferred[0]));
        assert_eq!(out, bits(&[0, 1, 0, 1]));
    }

    #[test]
    fn test_comparing_a_parity_against_zero_builds_a_mask() {
        // The reason these ops exist: `parity(x) == 0` is the post-selection criterion, and until
        // now a mask could only enter a program as a parameter.
        use crate::ops::Parity;

        let shots = bit_tensor(&[0, 1, 1, 1, 1, 0, 0, 0], &[4, 2]);
        let parities = Parity::new(1).eval(&[shots]).unwrap().remove(0);
        assert_eq!(parities, bits(&[1, 0, 1, 0]), "odd parity sets the bit");

        let mask = compared(&Equal, parities, bit_tensor(&[0], &[]));
        assert_eq!(mask, bits(&[0, 1, 0, 1]), "the even-parity shots are kept");
    }
    #[test]
    fn test_eval_compares_two_bit_operands() {
        // A `Bit` operand is unpacked before comparing, since the packed form is addressable only
        // by byte. A row wider than eight bits exercises that crossing.
        let a: Vec<u8> = (0..20).map(|i| u8::from(i % 3 == 0)).collect();
        let b: Vec<u8> = (0..20).map(|i| u8::from(i % 2 == 0)).collect();
        let expected: Vec<u8> = a.iter().zip(&b).map(|(x, y)| u8::from(x == y)).collect();
        assert_eq!(compared(&Equal, bits(&a), bits(&b)), bits(&expected));

        let less: Vec<u8> = a.iter().zip(&b).map(|(x, y)| u8::from(x < y)).collect();
        assert_eq!(
            compared(&Less, bits(&a), bits(&b)),
            bits(&less),
            "`Bit` is ordered, so the orderings accept it too"
        );
    }
}
