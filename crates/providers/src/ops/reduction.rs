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

use super::error::{MathOpError, check_axis};
use super::inference::{any, reduce};
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Tensor, TensorType};
use ndarray::{ArrayBase, ArrayD, Axis, Data, IxDyn, NdFloat, Zip};
use num_complex::Complex;

/// Return the result dtype of [`Mean`]: `F32` and the complex dtypes are unchanged, and everything
/// else becomes `F64`.
fn mean_out_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F32 => DType::F32,
        DType::C64 => DType::C64,
        DType::C128 => DType::C128,
        _ => DType::F64,
    }
}

/// Return the result dtype of [`Variance`] and [`Std`]: as [`mean_out_dtype`], except that a
/// complex operand produces its real counterpart, since a squared modulus is real.
fn real_out_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F32 => DType::F32,
        DType::C64 => DType::F32,
        DType::C128 => DType::F64,
        _ => DType::F64,
    }
}

/// Smallest output length at which the slice traversal in [`sum_sq_deviations`] is
/// faster than the lane traversal.
///
/// The slice traversal runs one [`Zip`] per position along the reduced axis, so its
/// fixed per-slice cost is amortized only once the output holds enough elements. The
/// crossover is therefore set by the ratio of that setup cost to the per-element
/// work, and measurement on `C64` data puts it near 16. The per-element work is
/// smaller on a target with wider vectors, so the crossover there should be larger.
///
/// The exact value matters little, and any of 8, 16 or 32 would do. Some threshold is
/// still needed. The output of a one-dimensional reduction is a single element, and
/// the slice traversal on it is more than forty times slower.
const MIN_SLICE_OUTPUT_LEN: usize = 16;

/// Compute the mean of `a` along `axis`, with that axis removed.
///
/// A zero-length axis gives a NaN mean.
fn complex_mean<A, S>(a: &ArrayBase<S, IxDyn>, axis: Axis) -> ArrayD<Complex<A>>
where
    A: NdFloat,
    S: Data<Elem = Complex<A>>,
{
    let n = A::from(a.len_of(axis)).expect("an axis length converts to a float");
    a.sum_axis(axis)
        .mapv_into(|c| Complex::new(c.re / n, c.im / n))
}

/// Sum the squared moduli of the deviations of `a` from its mean along `axis`,
/// with that axis removed.
///
/// Peak memory is independent of the length of the reduced axis.
/// The sum is accumulated in `f64` for both `f32` and `f64` to hold the summation error
/// below the rounding error of `f32` inputs.
fn sum_sq_deviations<A, S>(a: &ArrayBase<S, IxDyn>, axis: Axis) -> ArrayD<f64>
where
    A: NdFloat + Into<f64>,
    S: Data<Elem = Complex<A>>,
{
    let mean = complex_mean(a, axis);
    // This function contains two implementations. One does a reduction over each lane,
    // which is fast when lanes are contiguous in memory. The other accumulates over
    // slices, which is fast when slices are contiguous in memory. We inspect stride
    // information to form a heuristic about which one to choose: the difference in
    // speed can be as much as 40x.
    let reduced_stride = a.strides()[axis.index()].unsigned_abs();
    let fastest_stride = a
        .shape()
        .iter()
        .zip(a.strides())
        .filter_map(|(&len, &stride)| (len > 1).then_some(stride.unsigned_abs()))
        .min()
        .unwrap_or(1);
    if mean.len() >= MIN_SLICE_OUTPUT_LEN && reduced_stride > fastest_stride {
        // Perform an accumulation, one slice at a time.
        let mut accumulated = ArrayD::<f64>::zeros(mean.raw_dim());
        for slice in a.axis_iter(axis) {
            Zip::from(&mut accumulated)
                .and(&slice)
                .and(&mean)
                .for_each(|total, &x, &m| *total += (x - m).norm_sqr().into());
        }
        accumulated
    } else {
        // Perform a reduction of each lane separately.
        Zip::from(a.lanes(axis))
            .and(&mean)
            .map_collect(|lane, &m| lane.iter().map(|&x| (x - m).norm_sqr().into()).sum())
    }
}

/// Generate the [`ProgramOp`] implementation for a reduction along one axis.
///
/// Each reduction takes one operand, accepts every dtype, and folds along `self.axis`. They differ
/// only in the dtype they produce, in the fold itself and in the fields they have, so each op
/// supplies `$result_dtype`, the `$field`s it is described by, and an inherent
/// `fn reduce_axis(&self, x: &Tensor) -> Tensor`. `reduce_axis` may assume `self.axis` is in bounds
/// for `x`.
macro_rules! reduction_op {
    ($name:ident, $op_name:literal, $result_dtype:expr, [$($field:ident),+]) => {
        impl ProgramOp for $name {
            type Error = MathOpError;

            fn name(&self) -> &str {
                $op_name
            }
            fn namespace(&self) -> &str {
                QISKIT
            }
            fn describe(&self) -> Option<String> {
                Some([$(format!("{}={}", stringify!($field), self.$field)),+].join(", "))
            }
            fn arity(&self) -> usize {
                1
            }
            fn has_builtin_eval(&self) -> bool {
                true
            }
            fn infer_output_types(
                &self,
                inputs: &[TensorType],
            ) -> Result<Vec<TensorType>, Self::Error> {
                crate::unpack_operands!(self, inputs, [x]);
                Ok(vec![reduce(x, self.axis, any, $result_dtype)?])
            }
            fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
                crate::unpack_operands!(self, args, [x]);
                check_axis(self.axis, x.shape().len())?;
                Ok(vec![self.reduce_axis(x)])
            }
        }
    };
}

/// Mean of a tensor along a specified axis, removing that axis.
///
/// An `F32` operand produces `F32` and a complex one keeps its dtype. Every other dtype produces
/// `F64`. Averaging a zero-length axis divides by zero and gives `NaN`.
#[derive(Clone)]
pub struct Mean {
    axis: usize,
}

impl Mean {
    /// Construct a `Mean` op that reduces along `axis`.
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }

    /// Compute the mean of `x` along `self.axis`, which must be in bounds for `x`.
    fn reduce_axis(&self, x: &Tensor) -> Tensor {
        // can't use `ndarray::mean_axis` because it returns `None` for a zero-length axis.
        let n = x.shape()[self.axis];
        match x {
            Tensor::F32(a) => Tensor::F32((a.sum_axis(Axis(self.axis)) / n as f32).into_shared()),
            Tensor::F64(a) => Tensor::F64((a.sum_axis(Axis(self.axis)) / n as f64).into_shared()),
            Tensor::C64(a) => Tensor::C64(
                (a.sum_axis(Axis(self.axis)) / Complex::new(n as f32, 0.0)).into_shared(),
            ),
            Tensor::C128(a) => Tensor::C128(
                (a.sum_axis(Axis(self.axis)) / Complex::new(n as f64, 0.0)).into_shared(),
            ),
            other => {
                let Tensor::F64(a) = other.clone().cast(DType::F64) else {
                    unreachable!("Value cast as F64 can't be another dtype")
                };
                Tensor::F64((a.sum_axis(Axis(self.axis)) / n as f64).into_shared())
            }
        }
    }
}

reduction_op!(Mean, "mean", mean_out_dtype, [axis]);

/// Variance of a tensor along a specified axis, removing that axis.
///
/// The divisor is `n - ddof`, where `n` is the length of the axis. `ddof=0` gives the population
/// variance and `ddof=1` the sample variance.
///
/// An `F32` operand produces `F32`, `C64` produces `F32` and `C128` produces `F64`. Every other
/// dtype produces `F64`. The variance of a complex tensor is the mean squared modulus of its
/// deviations.
#[derive(Clone)]
pub struct Variance {
    axis: usize,
    ddof: f64,
}

impl Variance {
    /// Construct a `Variance` op that reduces along `axis` with degrees-of-freedom
    /// correction `ddof`.
    pub fn new(axis: usize, ddof: f64) -> Self {
        Self { axis, ddof }
    }

    /// Compute the variance of `x` along `self.axis`.
    fn reduce_axis(&self, x: &Tensor) -> Tensor {
        match x {
            Tensor::F32(a) => {
                let (n, ddof) = (a.shape()[self.axis] as f32, self.ddof as f32);
                let mean = (a.sum_axis(Axis(self.axis)) / n).insert_axis(Axis(self.axis));
                let sq = (a - &mean).mapv(|v| v * v);
                Tensor::F32((sq.sum_axis(Axis(self.axis)) / (n - ddof)).into_shared())
            }
            Tensor::C64(a) => {
                let denom = a.shape()[self.axis] as f64 - self.ddof;
                let var = sum_sq_deviations(a, Axis(self.axis));
                Tensor::F32(var.mapv(|total| (total / denom) as f32).into_shared())
            }
            Tensor::C128(a) => {
                let denom = a.shape()[self.axis] as f64 - self.ddof;
                let var = sum_sq_deviations(a, Axis(self.axis));
                Tensor::F64(var.mapv_into(|total| total / denom).into_shared())
            }
            other => {
                let Tensor::F64(a) = other.clone().cast(DType::F64) else {
                    unreachable!("Value cast as F64 can't be another dtype")
                };
                let n = a.shape()[self.axis] as f64;
                let mean = (a.sum_axis(Axis(self.axis)) / n).insert_axis(Axis(self.axis));
                let sq = (a - &mean).mapv(|v| v * v);
                Tensor::F64((sq.sum_axis(Axis(self.axis)) / (n - self.ddof)).into_shared())
            }
        }
    }
}

reduction_op!(Variance, "variance", real_out_dtype, [axis, ddof]);

/// Standard deviation of a tensor along a specified axis, removing that axis.
///
/// This is the square root of [`Variance`].
#[derive(Clone)]
pub struct Std {
    axis: usize,
    ddof: f64,
}

impl Std {
    /// Construct a `Std` op that reduces along `axis` with degrees-of-freedom
    /// correction `ddof`.
    pub fn new(axis: usize, ddof: f64) -> Self {
        Self { axis, ddof }
    }

    /// Compute the standard deviation of `x` along `self.axis`.
    fn reduce_axis(&self, x: &Tensor) -> Tensor {
        match Variance::new(self.axis, self.ddof).reduce_axis(x) {
            Tensor::F32(v) => Tensor::F32(v.mapv(f32::sqrt).into_shared()),
            Tensor::F64(v) => Tensor::F64(v.mapv(f64::sqrt).into_shared()),
            other => unreachable!("a variance is real, got {:?}", other.dtype()),
        }
    }
}

reduction_op!(Std, "std", real_out_dtype, [axis, ddof]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim;
    use ndarray::{ArrayView, ShapeBuilder, arr2};
    use num_traits::{Float, NumCast, Signed, abs, cast};
    use std::ops::Sub;

    fn approx_eq_slice<'a, T>(a: &'a [T], b: &'a [T])
    where
        T: Float + NumCast + std::fmt::Display,
        &'a T: Sub<&'a T>,
        <&'a T as Sub>::Output: Signed + Float,
    {
        assert_eq!(a.len(), b.len(), "slice lengths differ");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(abs(x - y) < cast(1e-10).unwrap(), "{x} != {y}");
        }
    }

    /// Return a 2-D operand type, so that a reduction has an axis to remove and one to keep.
    fn ty_2d(dtype: DType, rows: Dim, cols: Dim) -> TensorType {
        TensorType {
            dtype,
            shape: vec![rows, cols],
        }
    }

    // --- Mean tests ---

    #[test]
    fn test_mean_f64_axis0() {
        // [[1,2,3],[4,5,6]] along axis 0 → [2.5, 3.5, 4.5]
        let x = Tensor::F64(
            arr2(&[[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]])
                .into_dyn()
                .into_shared(),
        );
        let result = Mean::new(0).eval(&[x]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 leaf");
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_mean_f32_axis0() {
        // [[1,2,3],[4,5,6]] along axis 0 → [2.5, 3.5, 4.5]
        let x = Tensor::F32(
            arr2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]])
                .into_dyn()
                .into_shared(),
        );
        let result = Mean::new(0).eval(&[x]).unwrap();
        let Tensor::F32(arr) = &result[0] else {
            panic!("expected F32 leaf");
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_mean_i32_casts_to_f64() {
        let x = Tensor::from([1_i32, 2, 3, 4]);
        let result = Mean::new(0).eval(&[x]).unwrap();
        assert_eq!(
            result[0].dtype(),
            DType::F64,
            "integer input should produce F64 mean"
        );
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.5]);
    }

    #[test]
    fn test_mean_c128() {
        let data: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];
        let x = Tensor::C128(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Mean::new(0).eval(&[x]).unwrap();
        let Tensor::C128(arr) = &result[0] else {
            panic!("expected C128 leaf");
        };
        let v = arr.as_slice().unwrap()[0];
        assert!((v.re - 3.0).abs() < 1e-10);
        assert!((v.im - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_c64() {
        let data: Vec<Complex<f32>> = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];
        let x = Tensor::C64(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Mean::new(0).eval(&[x]).unwrap();
        let Tensor::C64(arr) = &result[0] else {
            panic!("expected C64 leaf");
        };
        let v = arr.as_slice().unwrap()[0];
        assert!((v.re - 3.0).abs() < 1e-10);
        assert!((v.im - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_large_strided_mean() {
        let raw_array: [Complex<f64>; 48] = [
            2.0.into(),
            4.0.into(),
            4.0.into(),
            4.0.into(),
            5.0.into(),
            5.0.into(),
            7.0.into(),
            9.0.into(),
            2.0.into(),
            4.0.into(),
            4.0.into(),
            4.0.into(),
            5.0.into(),
            5.0.into(),
            7.0.into(),
            9.0.into(),
            2.0.into(),
            4.0.into(),
            4.0.into(),
            4.0.into(),
            5.0.into(),
            7.0.into(),
            9.0.into(),
            1.0.into(),
            2.0.into(),
            4.0.into(),
            4.0.into(),
            4.0.into(),
            5.0.into(),
            5.0.into(),
            7.0.into(),
            9.0.into(),
            2.0.into(),
            4.0.into(),
            4.0.into(),
            4.0.into(),
            5.0.into(),
            5.0.into(),
            7.0.into(),
            9.0.into(),
            2.0.into(),
            4.0.into(),
            4.0.into(),
            4.0.into(),
            5.0.into(),
            7.0.into(),
            9.0.into(),
            1.0.into(),
        ];
        // For shape that triggers accumulation over slice path
        let strided = ArrayView::from_shape((4, 5, 4).strides((1, 4, 2)), &raw_array).unwrap();
        let x = Tensor::C128(strided.into_dyn().into_owned().into_shared());
        let result = Variance::new(0, 0.).eval(&[x]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("Expected F64 leaf")
        };
        approx_eq_slice(
            arr.as_slice().unwrap(),
            strided
                .mapv(|x| x.re)
                .var_axis(Axis(0), 0.)
                .as_slice()
                .unwrap(),
        );
    }

    // --- Variance tests ---

    #[test]
    fn test_variance_f64_ddof0() {
        let x = Tensor::from([2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let result = Variance::new(0, 0.0).eval(&[x]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 leaf");
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[4.0]);
    }

    #[test]
    fn test_variance_f64_ddof1() {
        let x = Tensor::from([2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let result = Variance::new(0, 1.0).eval(&[x]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected F64 leaf");
        };
        // sample variance = population variance * n / (n-1) = 4.0 * 8/7
        approx_eq_slice(arr.as_slice().unwrap(), &[4.0 * 8.0 / 7.0]);
    }

    #[test]
    fn test_variance_f32_ddof0() {
        let x = Tensor::from([2.0_f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let result = Variance::new(0, 0.0).eval(&[x]).unwrap();
        let Tensor::F32(arr) = &result[0] else {
            panic!("expected F32 leaf");
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[4.0]);
    }

    #[test]
    fn test_variance_f32_ddof1() {
        // Sample variance (ddof=1) of the same sequence
        let x = Tensor::from([2.0_f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let result = Variance::new(0, 1.0).eval(&[x]).unwrap();
        let Tensor::F32(arr) = &result[0] else {
            panic!("expected F32 leaf");
        };
        // sample variance = population variance * n / (n-1) = 4.0 * 8/7
        approx_eq_slice(arr.as_slice().unwrap(), &[4.0 * 8.0 / 7.0]);
    }

    #[test]
    fn test_variance_c128_returns_real() {
        let data: Vec<Complex<f64>> = vec![Complex::new(1.0, 1.0), Complex::new(3.0, 3.0)];
        let x = Tensor::C128(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Variance::new(0, 0.0).eval(&[x]).unwrap();
        assert_eq!(
            result[0].dtype(),
            DType::F64,
            "C128 variance should return F64"
        );
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.0]);
    }

    #[test]
    fn test_variance_c64_returns_real() {
        let data: Vec<Complex<f32>> = vec![Complex::new(1.0, 1.0), Complex::new(3.0, 3.0)];
        let x = Tensor::C64(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Variance::new(0, 0.0).eval(&[x]).unwrap();
        assert_eq!(
            result[0].dtype(),
            DType::F32,
            "C64 variance should return F32"
        );
        let Tensor::F32(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.0]);
    }

    // --- Std tests ---

    #[test]
    fn test_std_matches_sqrt_of_variance() {
        // Verify std = sqrt(variance) numerically
        let x = Tensor::from([1.0_f64, 3.0, 5.0, 7.0, 9.0]);
        let var_result = Variance::new(0, 0.0)
            .eval(std::slice::from_ref(&x))
            .unwrap();
        let std_result = Std::new(0, 0.0).eval(&[x]).unwrap();

        let Tensor::F64(var_arr) = &var_result[0] else {
            panic!()
        };
        let Tensor::F64(std_arr) = &std_result[0] else {
            panic!()
        };

        let var_val = var_arr.as_slice().unwrap()[0];
        let std_val = std_arr.as_slice().unwrap()[0];
        assert!((std_val - var_val.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_std_c128_returns_real() {
        let data: Vec<Complex<f64>> = vec![Complex::new(1.0, 1.0), Complex::new(3.0, 3.0)];
        let x = Tensor::C128(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Std::new(0, 0.0).eval(&[x]).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "C128 std should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        // std = sqrt(2.0)
        approx_eq_slice(arr.as_slice().unwrap(), &[2.0_f64.sqrt()]);
    }

    #[test]
    fn test_std_c64_returns_real() {
        let data: Vec<Complex<f32>> = vec![Complex::new(1.0, 1.0), Complex::new(3.0, 3.0)];
        let x = Tensor::C64(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Std::new(0, 0.0).eval(&[x]).unwrap();
        assert_eq!(result[0].dtype(), DType::F32, "C64 std should return F32");
        let Tensor::F32(arr) = &result[0] else {
            panic!()
        };
        // std = sqrt(2.0)
        approx_eq_slice(arr.as_slice().unwrap(), &[2.0_f32.sqrt()]);
    }

    #[test]
    fn test_i8_cast_to_float() {
        let data: Vec<i8> = vec![1, 3];
        let x = Tensor::I8(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Std::new(0, 0.0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 std should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.0]);
        let result = Mean::new(0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.]);
        let result = Variance::new(0, 0.).eval(&[x]).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.])
    }

    #[test]
    fn test_i16_cast_to_float() {
        let data: Vec<i16> = vec![1, 3];
        let x = Tensor::I16(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Std::new(0, 0.0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 std should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.0]);
        let result = Mean::new(0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.]);
        let result = Variance::new(0, 0.).eval(&[x]).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.])
    }

    #[test]
    fn test_i32_cast_to_float() {
        let data: Vec<i32> = vec![1, 3];
        let x = Tensor::I32(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Std::new(0, 0.0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 std should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.0]);
        let result = Mean::new(0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.]);
        let result = Variance::new(0, 0.).eval(&[x]).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.])
    }

    #[test]
    fn test_i64_cast_to_float() {
        let data: Vec<i64> = vec![1, 3];
        let x = Tensor::I64(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Std::new(0, 0.0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 std should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.0]);
        let result = Mean::new(0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.]);
        let result = Variance::new(0, 0.).eval(&[x]).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.])
    }

    #[test]
    fn test_u8_cast_to_float() {
        let data: Vec<u8> = vec![1, 3];
        let x = Tensor::U8(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Std::new(0, 0.0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 std should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.0]);
        let result = Mean::new(0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.]);
        let result = Variance::new(0, 0.).eval(&[x]).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.])
    }

    #[test]
    fn test_u16_cast_to_float() {
        let data: Vec<u16> = vec![1, 3];
        let x = Tensor::U16(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Std::new(0, 0.0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 std should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.0]);
        let result = Mean::new(0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.]);
        let result = Variance::new(0, 0.).eval(&[x]).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.])
    }

    #[test]
    fn test_u32_cast_to_float() {
        let data: Vec<u32> = vec![1, 3];
        let x = Tensor::U32(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Std::new(0, 0.0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 std should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.0]);
        let result = Mean::new(0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.]);
        let result = Variance::new(0, 0.).eval(&[x]).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.])
    }

    #[test]
    fn test_u64_cast_to_float() {
        let data: Vec<u64> = vec![1, 3];
        let x = Tensor::U64(ndarray::Array1::from(data).into_dyn().into_shared());
        let result = Std::new(0, 0.0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 std should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.0]);
        let result = Mean::new(0).eval(std::slice::from_ref(&x)).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[2.]);
        let result = Variance::new(0, 0.).eval(&[x]).unwrap();
        assert_eq!(result[0].dtype(), DType::F64, "I64 mean should return F64");
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        approx_eq_slice(arr.as_slice().unwrap(), &[1.])
    }

    // --- Output types ---

    #[test]
    fn test_mean_output_type_removes_the_reduced_axis() {
        assert_eq!(
            Mean::new(0)
                .infer_output_types(&[ty_2d(DType::I32, Dim::Fixed(2), Dim::Fixed(3))])
                .unwrap(),
            vec![TensorType {
                dtype: DType::F64,
                shape: vec![Dim::Fixed(3)],
            }]
        );
    }

    #[test]
    fn test_mean_forwards_a_bounded_axis_and_folds_one() {
        // A reduction divides by the reduced axis's size, but only at run time, which is when
        // that size is known. Either axis may be bounded.
        let bounded = Dim::Bounded { max: 4000 };
        assert_eq!(
            Mean::new(0)
                .infer_output_types(&[ty_2d(DType::F64, bounded, bounded)])
                .unwrap(),
            vec![TensorType {
                dtype: DType::F64,
                shape: vec![bounded],
            }]
        );
    }

    #[test]
    fn test_reduction_output_dtypes() {
        // Mean keeps a complex operand complex; variance and std take its squared modulus, which
        // is real.
        let cases = [
            (DType::F32, DType::F32, DType::F32),
            (DType::F64, DType::F64, DType::F64),
            (DType::I32, DType::F64, DType::F64),
            (DType::Bit, DType::F64, DType::F64),
            (DType::C64, DType::C64, DType::F32),
            (DType::C128, DType::C128, DType::F64),
        ];
        for (operand, mean, real) in cases {
            let ty = ty_2d(operand, Dim::Fixed(2), Dim::Fixed(3));
            for (op, expected) in [
                (
                    Mean::new(0).infer_output_types(std::slice::from_ref(&ty)),
                    mean,
                ),
                (
                    Variance::new(0, 0.0).infer_output_types(std::slice::from_ref(&ty)),
                    real,
                ),
                (
                    Std::new(0, 0.0).infer_output_types(std::slice::from_ref(&ty)),
                    real,
                ),
            ] {
                assert_eq!(op.unwrap()[0].dtype, expected, "for operand {operand}");
            }
        }
    }

    // --- Axis validation ---

    #[test]
    fn test_a_degenerate_ddof_yields_a_non_finite_value() {
        // `ddof` can only be weighed against the reduced axis's length, which a bounded axis does
        // not have until the reduction runs, so a divisor of zero is answered the way a division by
        // zero is: with a value the caller can see is not a number.
        let x = Tensor::from([1.0_f64, 2.0, 3.0]);
        for op in [Variance::new(0, 3.0), Variance::new(0, 4.0)] {
            let result = op.eval(std::slice::from_ref(&x)).unwrap();
            let Tensor::F64(v) = &result[0] else { panic!() };
            assert!(v.iter().all(|value| !value.is_finite() || *value < 0.0));
        }
        // A negative `ddof` divides by more than the axis's length, which is finite and meaningful.
        let result = Variance::new(0, -1.0)
            .eval(std::slice::from_ref(&x))
            .unwrap();
        let Tensor::F64(v) = &result[0] else { panic!() };
        approx_eq_slice(v.as_slice().unwrap(), &[2.0 / 4.0]);
    }

    #[test]
    fn test_output_type_axis_out_of_bounds_errors() {
        let ty = TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(3)],
        };
        assert!(matches!(
            Mean::new(1).infer_output_types(&[ty]).unwrap_err(),
            MathOpError::InvalidAxis { axis: 1, ndim: 1 }
        ));
    }

    #[test]
    fn test_mean_axis_out_of_bounds_errors() {
        let x = Tensor::from([1.0_f64, 2.0, 3.0]);
        assert!(matches!(
            Mean::new(1).eval(&[x]).unwrap_err(),
            MathOpError::InvalidAxis { axis: 1, ndim: 1 }
        ));
    }

    #[test]
    fn test_variance_axis_out_of_bounds_errors() {
        let x = Tensor::from([1.0_f64, 2.0, 3.0]);
        assert!(matches!(
            Variance::new(1, 0.0).eval(&[x]).unwrap_err(),
            MathOpError::InvalidAxis { axis: 1, ndim: 1 }
        ));
    }

    #[test]
    fn test_std_axis_out_of_bounds_errors() {
        let x = Tensor::from([1.0_f64, 2.0, 3.0]);
        assert!(matches!(
            Std::new(1, 0.0).eval(&[x]).unwrap_err(),
            MathOpError::InvalidAxis { axis: 1, ndim: 1 }
        ));
    }
}
