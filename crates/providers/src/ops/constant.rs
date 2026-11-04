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

//! The op that supplies a fixed tensor.

use crate::ops::ProgramOp;
use crate::tensor::{Tensor, TensorType};

/// A program op that owns one tensor and produces it unconditionally.
#[derive(Clone)]
pub struct Constant {
    value: Tensor,
}

impl Constant {
    /// Construct a new `Constant` producing `value`.
    pub fn new(value: Tensor) -> Self {
        Self { value }
    }

    /// Return the tensor this op produces.
    pub fn value(&self) -> &Tensor {
        &self.value
    }
}

impl ProgramOp for Constant {
    type Error = std::convert::Infallible;

    fn name(&self) -> &str {
        "constant"
    }

    fn namespace(&self) -> &str {
        super::QISKIT
    }

    fn arity(&self) -> usize {
        0
    }

    fn has_builtin_eval(&self) -> bool {
        true
    }

    fn infer_output_types(&self, _inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        Ok(vec![self.value.tensor_type()])
    }

    fn eval(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        Ok(vec![self.value.clone()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, Dim};

    #[test]
    fn test_constant_eval_returns_its_tensor() {
        let constant = Constant::new(Tensor::from([1.0_f64, 2.0, 3.0]));
        assert_eq!(
            constant.eval(&[]).unwrap(),
            vec![Tensor::from([1.0_f64, 2.0, 3.0])]
        );
        assert_eq!(constant.value(), &Tensor::from([1.0_f64, 2.0, 3.0]));
    }

    #[test]
    fn test_constant_full_name_and_arity() {
        let constant = Constant::new(Tensor::from([1.0_f64]));
        assert_eq!(constant.arity(), 0);
        assert!(constant.has_builtin_eval());
        assert_eq!(constant.full_name(), "qiskit.constant");
    }

    #[test]
    fn test_constant_output_type_is_its_tensor_type() {
        use ndarray::arr2;
        let constant = Constant::new(Tensor::F64(
            arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]).into_dyn().into_shared(),
        ));
        assert_eq!(
            constant.infer_output_types(&[]).unwrap(),
            vec![TensorType {
                dtype: DType::F64,
                shape: vec![Dim::Fixed(2), Dim::Fixed(2)],
            }]
        );
    }
}
