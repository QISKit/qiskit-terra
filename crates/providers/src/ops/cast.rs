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

//! An op to cast an operand's dtype.

use super::error::MathOpError;
use super::inference::{cast, cast_dtype};
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Tensor, TensorType};

/// Cast a tensor to a target dtype, keeping its shape.
#[derive(Clone)]
pub struct Cast {
    target: DType,
}

impl Cast {
    /// Construct a `Cast` op producing `target`.
    pub fn new(target: DType) -> Self {
        Self { target }
    }
}

impl ProgramOp for Cast {
    type Error = MathOpError;

    fn name(&self) -> &str {
        "cast"
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
        Ok(vec![cast(x, self.target)?])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x]);
        let target = cast_dtype(x.dtype(), self.target)?;
        Ok(vec![x.clone().cast(target)])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::Dim;
    use num_complex::Complex64;

    /// Return the type of a 1-D tensor of `len` elements.
    fn ty_1d(dtype: DType, len: usize) -> TensorType {
        TensorType {
            dtype,
            shape: vec![Dim::Fixed(len)],
        }
    }

    #[test]
    fn test_cast_full_name_and_arity() {
        let op = Cast::new(DType::F64);
        assert_eq!(op.full_name(), "qiskit.cast");
        assert_eq!(op.arity(), 1);
        assert!(op.has_builtin_eval());
    }

    #[test]
    fn test_infer_output_types_replaces_the_dtype_and_keeps_the_shape() {
        assert_eq!(
            Cast::new(DType::I32)
                .infer_output_types(&[ty_1d(DType::F64, 3)])
                .unwrap(),
            vec![ty_1d(DType::I32, 3)]
        );
        let bounded = |dtype| TensorType {
            dtype,
            shape: vec![Dim::Bounded { max: 4000 }, Dim::Fixed(2)],
        };
        assert_eq!(
            Cast::new(DType::F64)
                .infer_output_types(&[bounded(DType::Bit)])
                .unwrap(),
            vec![bounded(DType::F64)]
        );
    }

    #[test]
    fn test_cast_to_the_dtype_an_operand_already_has_is_admitted() {
        assert_eq!(
            Cast::new(DType::F64)
                .infer_output_types(&[ty_1d(DType::F64, 2)])
                .unwrap(),
            vec![ty_1d(DType::F64, 2)]
        );
        assert_eq!(
            Cast::new(DType::C64)
                .infer_output_types(&[ty_1d(DType::C64, 2)])
                .unwrap(),
            vec![ty_1d(DType::C64, 2)]
        );
    }

    #[test]
    fn test_eval_casts_its_operand() {
        assert_eq!(
            Cast::new(DType::I32)
                .eval(&[Tensor::from([1.7_f64, -2.2, 3.0])])
                .unwrap(),
            vec![Tensor::from([1_i32, -2, 3])]
        );
        assert_eq!(
            Cast::new(DType::F64)
                .eval(&[Tensor::from([1_i8, 2])])
                .unwrap(),
            vec![Tensor::from([1.0_f64, 2.0])]
        );
    }

    #[test]
    fn test_eval_casts_to_bit_by_testing_against_zero() {
        assert_eq!(
            Cast::new(DType::Bit)
                .eval(&[Tensor::from([0.0_f64, 2.5, -1.0])])
                .unwrap(),
            vec![Tensor::Bit(
                ndarray::arr1(&[0u8, 1, 1]).into_dyn().into_shared()
            )]
        );
    }

    #[test]
    fn test_a_complex_operand_cannot_be_cast_to_a_real_dtype() {
        let op = Cast::new(DType::F64);
        for err in [
            op.infer_output_types(&[ty_1d(DType::C128, 2)]).unwrap_err(),
            op.eval(&[Tensor::from([Complex64::new(1.0, 2.0)])])
                .unwrap_err(),
        ] {
            assert!(matches!(
                err,
                MathOpError::UnsupportedCast {
                    from: DType::C128,
                    to: DType::F64,
                }
            ));
        }

        // A complex target is fine, in either direction between the two widths.
        assert_eq!(
            Cast::new(DType::C64)
                .infer_output_types(&[ty_1d(DType::C128, 2)])
                .unwrap(),
            vec![ty_1d(DType::C64, 2)]
        );
        assert_eq!(
            Cast::new(DType::C128)
                .infer_output_types(&[ty_1d(DType::F32, 2)])
                .unwrap(),
            vec![ty_1d(DType::C128, 2)]
        );
    }
}
