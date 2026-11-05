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

//! A tensor value: a dense array over one of the supported dtypes.

use ndarray::{ArcArrayD, ArrayD, IxDyn};
use num_complex::{Complex32, Complex64};

use super::broadcast::{broadcast_elementwise, broadcast_shape};
use super::{DType, Dim, TensorError, TensorType};

/// A tensor of one of the supported dtypes.
///
/// Each variant wraps a reference-counted  copy-on-write dynamic ndarray (`ArcArrayD`).
#[derive(Debug, Clone, PartialEq)]
pub enum Tensor {
    C64(ArcArrayD<Complex32>), // complex
    C128(ArcArrayD<Complex64>),
    F32(ArcArrayD<f32>), // real
    F64(ArcArrayD<f64>),
    I8(ArcArrayD<i8>), // signed integer
    I16(ArcArrayD<i16>),
    I32(ArcArrayD<i32>),
    I64(ArcArrayD<i64>),
    U8(ArcArrayD<u8>), // unsigned integer
    U16(ArcArrayD<u16>),
    U32(ArcArrayD<u32>),
    U64(ArcArrayD<u64>),
    Bit(ArcArrayD<u8>), // bool
}

/// Return the error raised when an `op` has no implementation for some type(s).
///
/// Operands of different dtypes are a [`TensorError::DTypeMismatch`], and a pair that shares a
/// dtype the op does not implement is a [`TensorError::UnsupportedDType`].
fn dtype_error(op: &'static str, lhs: DType, rhs: DType) -> TensorError {
    if lhs == rhs {
        TensorError::UnsupportedDType { op, dtype: lhs }
    } else {
        TensorError::DTypeMismatch { op, lhs, rhs }
    }
}

/// Cast an array of a real numeric type to any supported dtype.
///
/// A cast to `Bit` compares against zero, like NumPy's cast to `bool`. A `Bit` tensor holds only 0
/// or 1, and truncating `2.5` to `2` would leave a value the bitwise operations cannot read.
macro_rules! cast_real {
    ($arr:expr, $src:ty, $target:expr) => {
        match $target {
            DType::Bit => Tensor::Bit($arr.mapv(|x: $src| u8::from(x != 0 as $src)).into_shared()),
            DType::U8 => Tensor::U8($arr.mapv(|x: $src| x as u8).into_shared()),
            DType::U16 => Tensor::U16($arr.mapv(|x: $src| x as u16).into_shared()),
            DType::U32 => Tensor::U32($arr.mapv(|x: $src| x as u32).into_shared()),
            DType::U64 => Tensor::U64($arr.mapv(|x: $src| x as u64).into_shared()),
            DType::I8 => Tensor::I8($arr.mapv(|x: $src| x as i8).into_shared()),
            DType::I16 => Tensor::I16($arr.mapv(|x: $src| x as i16).into_shared()),
            DType::I32 => Tensor::I32($arr.mapv(|x: $src| x as i32).into_shared()),
            DType::I64 => Tensor::I64($arr.mapv(|x: $src| x as i64).into_shared()),
            DType::F32 => Tensor::F32($arr.mapv(|x: $src| x as f32).into_shared()),
            DType::F64 => Tensor::F64($arr.mapv(|x: $src| x as f64).into_shared()),
            DType::C64 => Tensor::C64(
                $arr.mapv(|x: $src| Complex32::new(x as f32, 0.0))
                    .into_shared(),
            ),
            DType::C128 => Tensor::C128(
                $arr.mapv(|x: $src| Complex64::new(x as f64, 0.0))
                    .into_shared(),
            ),
        }
    };
}

/// Cast an array of a complex type to a complex dtype.
///
/// # Panics
///
/// Panics for real targets.
macro_rules! cast_complex {
    ($arr:expr, $target:expr) => {
        match $target {
            DType::C64 => Tensor::C64(
                $arr.mapv(|x| Complex32::new(x.re as f32, x.im as f32))
                    .into_shared(),
            ),
            DType::C128 => Tensor::C128(
                $arr.mapv(|x| Complex64::new(x.re as f64, x.im as f64))
                    .into_shared(),
            ),
            _ => panic!("cannot cast complex tensor to a real dtype"),
        }
    };
}

impl Tensor {
    /// Return the dtype of this tensor.
    pub fn dtype(&self) -> DType {
        match self {
            Tensor::C128(_) => DType::C128,
            Tensor::C64(_) => DType::C64,
            Tensor::F64(_) => DType::F64,
            Tensor::F32(_) => DType::F32,
            Tensor::I64(_) => DType::I64,
            Tensor::I32(_) => DType::I32,
            Tensor::I16(_) => DType::I16,
            Tensor::I8(_) => DType::I8,
            Tensor::U64(_) => DType::U64,
            Tensor::U32(_) => DType::U32,
            Tensor::U16(_) => DType::U16,
            Tensor::U8(_) => DType::U8,
            Tensor::Bit(_) => DType::Bit,
        }
    }

    /// Return the shape of this tensor as a slice of dimension sizes.
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::C128(a) => a.shape(),
            Tensor::C64(a) => a.shape(),
            Tensor::F64(a) => a.shape(),
            Tensor::F32(a) => a.shape(),
            Tensor::I64(a) => a.shape(),
            Tensor::I32(a) => a.shape(),
            Tensor::I16(a) => a.shape(),
            Tensor::I8(a) => a.shape(),
            Tensor::U64(a) => a.shape(),
            Tensor::U32(a) => a.shape(),
            Tensor::U16(a) => a.shape(),
            Tensor::U8(a) => a.shape(),
            Tensor::Bit(a) => a.shape(),
        }
    }

    /// Return the [`TensorType`] that describes this tensor's dtype and concrete shape.
    pub fn tensor_type(&self) -> TensorType {
        TensorType {
            dtype: self.dtype(),
            shape: self.shape().iter().map(|&n| Dim::Fixed(n)).collect(),
        }
    }

    /// Return whether this tensor satisfies `ty`.
    ///
    /// A [`Dim::Fixed`] axis admits exactly its size and a [`Dim::Bounded`] axis admits any
    /// size up to and including its bound.
    pub fn matches(&self, ty: &TensorType) -> bool {
        self.dtype() == ty.dtype
            && self.shape().len() == ty.shape.len()
            && self
                .shape()
                .iter()
                .zip(&ty.shape)
                .all(|(&size, &dim)| dim.admits(Dim::Fixed(size)))
    }

    /// Compute the element-wise power with NumPy-style broadcasting.
    ///
    /// An integer result wraps on overflow. Returns [`TensorError::NegativeExponent`] if an
    /// exponent of a signed integer dtype is negative, [`TensorError::DTypeMismatch`] if the
    /// operands have different dtypes, [`TensorError::UnsupportedDType`] if they share a dtype
    /// that does not support `pow`, and [`TensorError::ShapeMismatch`] if the shapes are not
    /// broadcast-compatible.
    pub fn pow(&self, rhs: &Tensor) -> Result<Tensor, TensorError> {
        /// Raise a signed integer tensor to a non-negative exponent.
        macro_rules! signed_pow {
            ($variant:ident, $base:expr, $exponent:expr) => {{
                if $exponent.iter().any(|&y| y < 0) {
                    return Err(TensorError::NegativeExponent { dtype: rhs.dtype() });
                }
                broadcast_elementwise($base, $exponent, |&x, &y| x.wrapping_pow(y as u32))
                    .map(Tensor::$variant)
            }};
        }
        match (self, rhs) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.powf(y)).map(Tensor::F32)
            }
            (Tensor::F64(a), Tensor::F64(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.powf(y)).map(Tensor::F64)
            }
            (Tensor::C64(a), Tensor::C64(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.powc(y)).map(Tensor::C64)
            }
            (Tensor::C128(a), Tensor::C128(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.powc(y)).map(Tensor::C128)
            }
            (Tensor::I8(a), Tensor::I8(b)) => signed_pow!(I8, a, b),
            (Tensor::I16(a), Tensor::I16(b)) => signed_pow!(I16, a, b),
            (Tensor::I32(a), Tensor::I32(b)) => signed_pow!(I32, a, b),
            (Tensor::I64(a), Tensor::I64(b)) => signed_pow!(I64, a, b),
            (Tensor::U8(a), Tensor::U8(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.wrapping_pow(y as u32)).map(Tensor::U8)
            }
            (Tensor::U16(a), Tensor::U16(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.wrapping_pow(y as u32)).map(Tensor::U16)
            }
            (Tensor::U32(a), Tensor::U32(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.wrapping_pow(y)).map(Tensor::U32)
            }
            (Tensor::U64(a), Tensor::U64(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.wrapping_pow(y as u32)).map(Tensor::U64)
            }
            _ => Err(dtype_error("pow", self.dtype(), rhs.dtype())),
        }
    }

    /// Broadcast this tensor to `shape`.
    ///
    /// The shapes are right-aligned, so leading axes may be added and an axis of size `1` grows to
    /// any size. Returns [`TensorError::ShapeMismatch`] if `shape` cannot be reached that way. This
    /// is the value-level counterpart of [`broadcast_dims_to`](super::broadcast_dims_to).
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor, TensorError> {
        if self.shape() == shape {
            return Ok(self.clone());
        }
        let ix = IxDyn(shape);
        macro_rules! broadcast {
            ($variant:ident, $arr:expr) => {
                $arr.broadcast(ix)
                    .map(|view| Tensor::$variant(view.to_owned().into_shared()))
            };
        }
        match self {
            Tensor::C128(a) => broadcast!(C128, a),
            Tensor::C64(a) => broadcast!(C64, a),
            Tensor::F64(a) => broadcast!(F64, a),
            Tensor::F32(a) => broadcast!(F32, a),
            Tensor::I64(a) => broadcast!(I64, a),
            Tensor::I32(a) => broadcast!(I32, a),
            Tensor::I16(a) => broadcast!(I16, a),
            Tensor::I8(a) => broadcast!(I8, a),
            Tensor::U64(a) => broadcast!(U64, a),
            Tensor::U32(a) => broadcast!(U32, a),
            Tensor::U16(a) => broadcast!(U16, a),
            Tensor::U8(a) => broadcast!(U8, a),
            Tensor::Bit(a) => broadcast!(Bit, a),
        }
        .ok_or_else(|| TensorError::ShapeMismatch {
            lhs: self.shape().to_vec(),
            rhs: shape.to_vec(),
        })
    }

    /// Cast this tensor to `target`, consuming it. Returns `self` unchanged if already that dtype.
    pub fn cast(self, target: DType) -> Tensor {
        if self.dtype() == target {
            return self;
        }
        match &self {
            Tensor::Bit(a) | Tensor::U8(a) => cast_real!(a, u8, target),
            Tensor::U16(a) => cast_real!(a, u16, target),
            Tensor::U32(a) => cast_real!(a, u32, target),
            Tensor::U64(a) => cast_real!(a, u64, target),
            Tensor::I8(a) => cast_real!(a, i8, target),
            Tensor::I16(a) => cast_real!(a, i16, target),
            Tensor::I32(a) => cast_real!(a, i32, target),
            Tensor::I64(a) => cast_real!(a, i64, target),
            Tensor::F32(a) => cast_real!(a, f32, target),
            Tensor::F64(a) => cast_real!(a, f64, target),
            Tensor::C64(a) => cast_complex!(a, target),
            Tensor::C128(a) => cast_complex!(a, target),
        }
    }
}

/// Implement `From<&[T]>`, `From<&[T; N]>`, `From<ArrayD<T>>`, and
/// `From<ArcArrayD<T>>` for a given `Tensor` variant.
macro_rules! impl_tensor_from {
    ($variant:ident, $t:ty) => {
        impl From<&[$t]> for Tensor {
            fn from(data: &[$t]) -> Self {
                Tensor::$variant(ndarray::arr1(data).into_dyn().into_shared())
            }
        }
        impl<const N: usize> From<[$t; N]> for Tensor {
            fn from(data: [$t; N]) -> Self {
                Tensor::$variant(ndarray::arr1(&data).into_dyn().into_shared())
            }
        }
        impl From<ArrayD<$t>> for Tensor {
            fn from(data: ArrayD<$t>) -> Self {
                Tensor::$variant(data.into_shared())
            }
        }
        impl From<ArcArrayD<$t>> for Tensor {
            fn from(data: ArcArrayD<$t>) -> Self {
                Tensor::$variant(data)
            }
        }
    };
}

impl_tensor_from!(C128, Complex64);
impl_tensor_from!(C64, Complex32);
impl_tensor_from!(F64, f64);
impl_tensor_from!(F32, f32);
impl_tensor_from!(I64, i64);
impl_tensor_from!(I32, i32);
impl_tensor_from!(I16, i16);
impl_tensor_from!(I8, i8);
impl_tensor_from!(U64, u64);
impl_tensor_from!(U32, u32);
impl_tensor_from!(U16, u16);
impl_tensor_from!(U8, u8); // u8 → U8; Bit requires explicit construction

/// Integer division and remainder where a zero divisor gives zero, as in NumPy.
trait DivideByZero: Sized {
    /// Return `self / rhs`, or zero if `rhs` is zero.
    fn div_or_zero(self, rhs: Self) -> Self;
    /// Return `self % rhs`, or zero if `rhs` is zero.
    fn rem_or_zero(self, rhs: Self) -> Self;
}

macro_rules! impl_divide_by_zero {
    ($($t:ty),*) => {
        $(
            impl DivideByZero for $t {
                fn div_or_zero(self, rhs: Self) -> Self {
                    if rhs == 0 { 0 } else { self.wrapping_div(rhs) }
                }
                fn rem_or_zero(self, rhs: Self) -> Self {
                    if rhs == 0 { 0 } else { self.wrapping_rem(rhs) }
                }
            }
        )*
    };
}

impl_divide_by_zero!(i8, i16, i32, i64, u8, u16, u32, u64);

/// Define a fallible element-wise binary method on [`Tensor`] (e.g. `add_tensor`),
/// plus the corresponding [`std::ops`] trait impls that unwrap the `Result`.
///
/// The operand shapes are pre-validated with [`broadcast_shape`] so that the underlying
/// ndarray operator (which broadcasts but panics on shape mismatch) cannot panic. An integer arm
/// applies `$integer` element by element, since the plain operator panics on an integer overflow
/// or a zero divisor.
macro_rules! impl_tensor_binop {
    ($trait:ident, $method:ident, $tensor_method:ident, $op:tt, $integer:ident, $op_name:literal) => {
        impl Tensor {
            #[doc = concat!(
                "Compute the element-wise `",
                $op_name,
                "` with NumPy-style broadcasting.\n\n",
                "Returns [`TensorError::DTypeMismatch`] if the operand dtypes differ, ",
                "[`TensorError::UnsupportedDType`] if they share a dtype this op does not ",
                "support, and [`TensorError::ShapeMismatch`] if the shapes are not ",
                "broadcast-compatible."
            )]
            pub fn $tensor_method(&self, rhs: &Tensor) -> Result<Tensor, TensorError> {
                broadcast_shape(self.shape(), rhs.shape())?;
                match (self, rhs) {
                    (Tensor::C128(a), Tensor::C128(b)) => Ok(Tensor::C128((a $op b).into_shared())),
                    (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64((a $op b).into_shared())),
                    (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64((a $op b).into_shared())),
                    (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32((a $op b).into_shared())),
                    (Tensor::I64(a), Tensor::I64(b)) => {
                        broadcast_elementwise(a, b, |&x, &y| x.$integer(y)).map(Tensor::I64)
                    }
                    (Tensor::I32(a), Tensor::I32(b)) => {
                        broadcast_elementwise(a, b, |&x, &y| x.$integer(y)).map(Tensor::I32)
                    }
                    (Tensor::I16(a), Tensor::I16(b)) => {
                        broadcast_elementwise(a, b, |&x, &y| x.$integer(y)).map(Tensor::I16)
                    }
                    (Tensor::I8(a), Tensor::I8(b)) => {
                        broadcast_elementwise(a, b, |&x, &y| x.$integer(y)).map(Tensor::I8)
                    }
                    (Tensor::U64(a), Tensor::U64(b)) => {
                        broadcast_elementwise(a, b, |&x, &y| x.$integer(y)).map(Tensor::U64)
                    }
                    (Tensor::U32(a), Tensor::U32(b)) => {
                        broadcast_elementwise(a, b, |&x, &y| x.$integer(y)).map(Tensor::U32)
                    }
                    (Tensor::U16(a), Tensor::U16(b)) => {
                        broadcast_elementwise(a, b, |&x, &y| x.$integer(y)).map(Tensor::U16)
                    }
                    (Tensor::U8(a), Tensor::U8(b)) => {
                        broadcast_elementwise(a, b, |&x, &y| x.$integer(y)).map(Tensor::U8)
                    }
                    _ => Err(dtype_error($op_name, self.dtype(), rhs.dtype())),
                }
            }
        }
        impl std::ops::$trait for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Self) -> Tensor {
                self.$tensor_method(rhs).unwrap_or_else(|e| panic!("{e}"))
            }
        }
        impl std::ops::$trait for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Self) -> Tensor { &self $op &rhs }
        }
    };
}

impl_tensor_binop!(Add, add, add_tensor, +, wrapping_add, "add");
impl_tensor_binop!(Sub, sub, sub_tensor, -, wrapping_sub, "sub");
impl_tensor_binop!(Mul, mul, mul_tensor, *, wrapping_mul, "mul");
impl_tensor_binop!(Div, div, div_tensor, /, div_or_zero, "div");

// `Rem` is hand-written rather than going through `impl_tensor_binop!` because
// `num_complex` does not implement `%`, so the complex variants must be omitted.
impl Tensor {
    /// Compute the element-wise `%` with NumPy-style broadcasting (real dtypes only).
    ///
    /// Returns [`TensorError::DTypeMismatch`] if the operand dtypes differ,
    /// [`TensorError::UnsupportedDType`] if they share a dtype this op does not support (a complex
    /// one, for instance), and [`TensorError::ShapeMismatch`] if the shapes are not
    /// broadcast-compatible.
    pub fn rem_tensor(&self, rhs: &Tensor) -> Result<Tensor, TensorError> {
        broadcast_shape(self.shape(), rhs.shape())?;
        match (self, rhs) {
            (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64((a % b).into_shared())),
            (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32((a % b).into_shared())),
            (Tensor::I64(a), Tensor::I64(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.rem_or_zero(y)).map(Tensor::I64)
            }
            (Tensor::I32(a), Tensor::I32(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.rem_or_zero(y)).map(Tensor::I32)
            }
            (Tensor::I16(a), Tensor::I16(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.rem_or_zero(y)).map(Tensor::I16)
            }
            (Tensor::I8(a), Tensor::I8(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.rem_or_zero(y)).map(Tensor::I8)
            }
            (Tensor::U64(a), Tensor::U64(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.rem_or_zero(y)).map(Tensor::U64)
            }
            (Tensor::U32(a), Tensor::U32(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.rem_or_zero(y)).map(Tensor::U32)
            }
            (Tensor::U16(a), Tensor::U16(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.rem_or_zero(y)).map(Tensor::U16)
            }
            (Tensor::U8(a), Tensor::U8(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.rem_or_zero(y)).map(Tensor::U8)
            }
            _ => Err(dtype_error("rem", self.dtype(), rhs.dtype())),
        }
    }
}

impl std::ops::Rem for &Tensor {
    type Output = Tensor;
    fn rem(self, rhs: Self) -> Tensor {
        self.rem_tensor(rhs).unwrap_or_else(|e| panic!("{e}"))
    }
}

impl std::ops::Rem for Tensor {
    type Output = Tensor;
    fn rem(self, rhs: Self) -> Tensor {
        &self % &rhs
    }
}

/// Define an element-wise bitwise method on [`Tensor`] over `Bit` operands.
///
/// Bitwise operations are defined for `Bit` alone, so these are separate from
/// [`impl_tensor_binop`], whose arms cover the numeric dtypes and not `Bit`. As there, the operand
/// shapes are pre-validated with [`broadcast_shape`] so that the underlying ndarray operator, which
/// broadcasts but panics on a shape mismatch, cannot panic.
macro_rules! impl_tensor_bitop {
    ($method:ident, $op:tt, $op_name:literal) => {
        impl Tensor {
            #[doc = concat!(
                "Element-wise `",
                $op_name,
                "` of two `Bit` tensors, with NumPy-style broadcasting.\n\n",
                "Returns [`TensorError::DTypeMismatch`] unless both operands are `Bit`, and ",
                "[`TensorError::ShapeMismatch`] if the shapes are not broadcast-compatible."
            )]
            pub fn $method(&self, rhs: &Tensor) -> Result<Tensor, TensorError> {
                broadcast_shape(self.shape(), rhs.shape())?;
                match (self, rhs) {
                    (Tensor::Bit(a), Tensor::Bit(b)) => Ok(Tensor::Bit((a $op b).into_shared())),
                    _ => Err(TensorError::DTypeMismatch {
                        op: $op_name,
                        lhs: self.dtype(),
                        rhs: rhs.dtype(),
                    }),
                }
            }
        }
    };
}

impl_tensor_bitop!(bitand_tensor, &, "bitand");
impl_tensor_bitop!(bitor_tensor, |, "bitor");
impl_tensor_bitop!(bitxor_tensor, ^, "bitxor");

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn test_tensor_equality() {
        assert_eq!(Tensor::from([1.0_f64, 2.0]), Tensor::from([1.0_f64, 2.0]));
        assert_ne!(Tensor::from([1.0_f64, 2.0]), Tensor::from([1.0_f64, 3.0]));

        // Bit and U8 share a storage type but are distinct dtypes.
        let bits = ndarray::ArrayD::from_elem(IxDyn(&[2]), 1u8).into_shared();
        assert_ne!(Tensor::Bit(bits.clone()), Tensor::U8(bits));
    }

    // Construction, dtype, shape.

    #[test]
    fn test_from_slice() {
        let t = Tensor::from(&[1.0f64, 2.0, 3.0][..]);
        assert_eq!(t.dtype(), DType::F64);
        assert_eq!(t.shape(), &[3]);

        let t = Tensor::from(&[1i32, -2, 3][..]);
        assert_eq!(t.dtype(), DType::I32);
        assert_eq!(t.shape(), &[3]);

        let t = Tensor::from(&[10u8, 20, 30][..]);
        assert_eq!(t.dtype(), DType::U8);
        assert_eq!(t.shape(), &[3]);

        let t = Tensor::from(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)][..]);
        assert_eq!(t.dtype(), DType::C128);
        assert_eq!(t.shape(), &[2]);
    }

    #[test]
    fn test_from_array() {
        let t = Tensor::from([0.5f32, 1.5, 2.5]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.shape(), &[3]);

        let t = Tensor::from([1i64, 2, 3, 4]);
        assert_eq!(t.dtype(), DType::I64);
        assert_eq!(t.shape(), &[4]);
    }

    #[test]
    fn test_clone_shares_buffer() {
        // ArcArray storage means Tensor::clone() is a refcount bump, not a deep
        // copy. Verify by comparing the underlying buffer pointer between the
        // original and a clone.
        let t = Tensor::from([1.0_f64, 2.0, 3.0]);
        let cloned = t.clone();
        let Tensor::F64(orig) = &t else {
            panic!("expected F64 tensor")
        };
        let Tensor::F64(copy) = &cloned else {
            panic!("expected F64 tensor")
        };
        assert_eq!(orig.as_ptr(), copy.as_ptr());
    }

    #[test]
    fn test_from_arrayd() {
        let arr = ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f64; 6]).unwrap();
        let t = Tensor::from(arr);
        assert_eq!(t.dtype(), DType::F64);
        assert_eq!(t.shape(), &[2, 3]);

        let arr = ndarray::Array::from_shape_vec(IxDyn(&[4, 1, 2]), vec![0u32; 8]).unwrap();
        let t = Tensor::from(arr);
        assert_eq!(t.dtype(), DType::U32);
        assert_eq!(t.shape(), &[4, 1, 2]);
    }

    #[test]
    fn test_tensor_type() {
        let t = Tensor::from([1.0f64, 2.0, 3.0]);
        assert_eq!(
            t.tensor_type(),
            TensorType {
                dtype: DType::F64,
                shape: vec![Dim::Fixed(3)],
            }
        );

        let arr = ndarray::Array::from_shape_vec(IxDyn(&[2, 4]), vec![0i16; 8]).unwrap();
        let t = Tensor::from(arr);
        assert_eq!(
            t.tensor_type(),
            TensorType {
                dtype: DType::I16,
                shape: vec![Dim::Fixed(2), Dim::Fixed(4)],
            }
        );
    }

    #[test]
    fn test_matches_its_own_type() {
        let t = Tensor::from([1.0f64, 2.0, 3.0]);
        assert!(t.matches(&t.tensor_type()));
    }

    #[test]
    fn test_matches_rejects_a_different_dtype_rank_or_size() {
        let t = Tensor::from([1.0f64, 2.0, 3.0]);
        for shape in [
            vec![Dim::Fixed(2)],
            vec![Dim::Fixed(3), Dim::Fixed(1)],
            vec![],
        ] {
            assert!(
                !t.matches(&TensorType {
                    dtype: DType::F64,
                    shape: shape.clone(),
                }),
                "for shape {shape:?}"
            );
        }
        assert!(!t.matches(&TensorType {
            dtype: DType::F32,
            shape: vec![Dim::Fixed(3)],
        }));
    }

    #[test]
    fn test_matches_a_bounded_axis_up_to_its_bound() {
        let bounded = |max| TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Bounded { max }],
        };
        let t = Tensor::from([1.0f64, 2.0, 3.0]);
        assert!(t.matches(&bounded(4)));
        assert!(t.matches(&bounded(3)));
        assert!(!t.matches(&bounded(2)));
    }

    // The cast method.

    #[test]
    fn test_cast_identity() {
        let t = Tensor::from([1.0f64, 2.0, 3.0]);
        let t2 = t.cast(DType::F64);
        assert_eq!(t2.dtype(), DType::F64);
        if let Tensor::F64(a) = t2 {
            assert_eq!(a.as_slice().unwrap(), &[1.0f64, 2.0, 3.0]);
        } else {
            panic!("expected F64 tensor");
        }
    }

    #[test]
    fn test_cast_real_widening() {
        let t = Tensor::from([1i8, 2, 3]);
        let t2 = t.cast(DType::I64);
        assert_eq!(t2.dtype(), DType::I64);
        if let Tensor::I64(a) = t2 {
            assert_eq!(a.as_slice().unwrap(), &[1i64, 2, 3]);
        } else {
            panic!("expected I64 tensor");
        }

        let t = Tensor::from([1.0f32, 2.0]);
        let t2 = t.cast(DType::F64);
        assert_eq!(t2.dtype(), DType::F64);
        if let Tensor::F64(a) = t2 {
            assert!(approx::abs_diff_eq!(a[0], 1.0f64, epsilon = 1e-6));
            assert!(approx::abs_diff_eq!(a[1], 2.0f64, epsilon = 1e-6));
        } else {
            panic!("expected F64 tensor");
        }
    }

    #[test]
    fn test_cast_real_to_complex() {
        let t = Tensor::from([3.0f64, 4.0]);
        let t2 = t.cast(DType::C128);
        assert_eq!(t2.dtype(), DType::C128);
        if let Tensor::C128(a) = t2 {
            assert!(approx::abs_diff_eq!(a[0].re, 3.0f64, epsilon = 1e-12));
            assert!(approx::abs_diff_eq!(a[0].im, 0.0f64, epsilon = 1e-12));
            assert!(approx::abs_diff_eq!(a[1].re, 4.0f64, epsilon = 1e-12));
        } else {
            panic!("expected C128 tensor");
        }
    }

    #[test]
    fn test_cast_complex_to_complex() {
        let t = Tensor::from([Complex64::new(1.0, -1.0), Complex64::new(0.5, 2.0)]);
        let t2 = t.cast(DType::C64);
        assert_eq!(t2.dtype(), DType::C64);
        if let Tensor::C64(a) = t2 {
            assert!(approx::abs_diff_eq!(a[0].re, 1.0f32, epsilon = 1e-5));
            assert!(approx::abs_diff_eq!(a[0].im, -1.0f32, epsilon = 1e-5));
        } else {
            panic!("expected C64 tensor");
        }
    }

    #[test]
    fn test_cast_to_bit_tests_against_zero() {
        // Any non-zero value becomes 1, so a Bit tensor never holds anything else.
        let t = Tensor::from([0.0_f64, 0.5, 1.0, 2.5, -3.0]);
        let Tensor::Bit(bits) = t.cast(DType::Bit) else {
            panic!("expected Bit tensor")
        };
        assert_eq!(bits.as_slice().unwrap(), &[0, 1, 1, 1, 1]);

        let Tensor::Bit(bits) = Tensor::from([0_i32, 7, -7]).cast(DType::Bit) else {
            panic!("expected Bit tensor")
        };
        assert_eq!(bits.as_slice().unwrap(), &[0, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "cannot cast complex")]
    fn test_cast_complex_to_real_panics() {
        let t = Tensor::from([Complex64::new(1.0, 2.0)]);
        let _ = t.cast(DType::F64);
    }

    // The broadcast_to method.

    #[test]
    fn test_broadcast_to_duplicates_along_the_axes_that_grow() {
        // [1, 2, 3] to [2, 3] repeats the row; a leading axis may be added.
        let t = Tensor::from([1.0_f64, 2.0, 3.0]);
        let Tensor::F64(arr) = t.broadcast_to(&[2, 3]).unwrap() else {
            panic!("expected F64 tensor")
        };
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // A single element reaches any shape.
        let Tensor::Bit(arr) =
            Tensor::Bit(ndarray::ArrayD::from_elem(IxDyn(&[1]), 1u8).into_shared())
                .broadcast_to(&[2, 2])
                .unwrap()
        else {
            panic!("expected Bit tensor")
        };
        assert_eq!(arr.as_slice().unwrap(), &[1, 1, 1, 1]);
    }

    #[test]
    fn test_broadcast_to_its_own_shape_shares_the_buffer() {
        let t = Tensor::from([1.0_f64, 2.0, 3.0]);
        let broadcast = t.broadcast_to(&[3]).unwrap();
        let (Tensor::F64(orig), Tensor::F64(copy)) = (&t, &broadcast) else {
            panic!("expected F64 tensors")
        };
        assert_eq!(orig.as_ptr(), copy.as_ptr());
    }

    #[test]
    fn test_broadcast_to_a_shape_it_cannot_reach_reports_both() {
        let t = Tensor::from([1.0_f64, 2.0, 3.0]);
        for shape in [vec![4], vec![1], vec![3, 2], vec![]] {
            assert!(
                matches!(
                    t.broadcast_to(&shape).unwrap_err(),
                    TensorError::ShapeMismatch { lhs, rhs } if lhs == [3] && rhs == shape
                ),
                "for target {shape:?}"
            );
        }
    }

    // The pow method.

    #[test]
    fn test_pow_float() {
        let base = Tensor::from([4.0f64, 9.0, 16.0]);
        let exp = Tensor::from([0.5f64, 0.5, 0.5]);
        let result = base.pow(&exp).unwrap();
        assert_eq!(result.dtype(), DType::F64);
        if let Tensor::F64(a) = result {
            assert!(approx::abs_diff_eq!(a[0], 2.0f64, epsilon = 1e-12));
            assert!(approx::abs_diff_eq!(a[1], 3.0f64, epsilon = 1e-12));
            assert!(approx::abs_diff_eq!(a[2], 4.0f64, epsilon = 1e-12));
        } else {
            panic!("expected F64 tensor");
        }
    }

    #[test]
    fn test_pow_int() {
        let base = Tensor::from([2i32, 3, 4]);
        let exp = Tensor::from([3i32, 2, 1]);
        let result = base.pow(&exp).unwrap();
        assert_eq!(result.dtype(), DType::I32);
        if let Tensor::I32(a) = result {
            assert_eq!(a.as_slice().unwrap(), &[8i32, 9, 4]);
        } else {
            panic!("expected I32 tensor");
        }
    }

    #[test]
    fn test_pow_broadcast() {
        // shape [3] ^ shape [1] -> shape [3]
        let base = Tensor::from([2.0f64, 3.0, 4.0]);
        let exp = Tensor::from([2.0f64]);
        let result = base.pow(&exp).unwrap();
        assert_eq!(result.shape(), &[3]);
        if let Tensor::F64(a) = result {
            assert!(approx::abs_diff_eq!(a[0], 4.0f64, epsilon = 1e-12));
            assert!(approx::abs_diff_eq!(a[1], 9.0f64, epsilon = 1e-12));
            assert!(approx::abs_diff_eq!(a[2], 16.0f64, epsilon = 1e-12));
        } else {
            panic!("expected F64 tensor");
        }
    }

    // Arithmetic operators.

    #[test]
    fn test_add() {
        let a = Tensor::from([1.0f64, 2.0, 3.0]);
        let b = Tensor::from([4.0f64, 5.0, 6.0]);
        let c = &a + &b;
        assert_eq!(c.dtype(), DType::F64);
        if let Tensor::F64(arr) = c {
            assert_eq!(arr.as_slice().unwrap(), &[5.0f64, 7.0, 9.0]);
        } else {
            panic!("expected F64 tensor");
        }
    }

    #[test]
    fn test_sub() {
        let a = Tensor::from([10.0f64, 5.0, 3.0]);
        let b = Tensor::from([1.0f64, 2.0, 3.0]);
        let c = &a - &b;
        assert_eq!(c.dtype(), DType::F64);
        if let Tensor::F64(arr) = c {
            assert_eq!(arr.as_slice().unwrap(), &[9.0f64, 3.0, 0.0]);
        } else {
            panic!("expected F64 tensor");
        }
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from([2.0f64, 3.0, 4.0]);
        let b = Tensor::from([5.0f64, 6.0, 7.0]);
        let c = &a * &b;
        assert_eq!(c.dtype(), DType::F64);
        if let Tensor::F64(arr) = c {
            assert_eq!(arr.as_slice().unwrap(), &[10.0f64, 18.0, 28.0]);
        } else {
            panic!("expected F64 tensor");
        }
    }

    #[test]
    fn test_div() {
        let a = Tensor::from([6.0f64, 9.0, 12.0]);
        let b = Tensor::from([2.0f64, 3.0, 4.0]);
        let c = &a / &b;
        assert_eq!(c.dtype(), DType::F64);
        if let Tensor::F64(arr) = c {
            assert_eq!(arr.as_slice().unwrap(), &[3.0f64, 3.0, 3.0]);
        } else {
            panic!("expected F64 tensor");
        }
    }

    #[test]
    fn test_rem() {
        let a = Tensor::from([7i32, 10, 13]);
        let b = Tensor::from([3i32, 4, 5]);
        let c = &a % &b;
        assert_eq!(c.dtype(), DType::I32);
        if let Tensor::I32(arr) = c {
            assert_eq!(arr.as_slice().unwrap(), &[1i32, 2, 3]);
        } else {
            panic!("expected I32 tensor");
        }
    }

    #[test]
    fn test_arithmetic_complex() {
        let a = Tensor::from([Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);
        let b = Tensor::from([Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)]);
        let sum = &a + &b;
        assert_eq!(sum.dtype(), DType::C128);
        if let Tensor::C128(arr) = &sum {
            assert!(approx::abs_diff_eq!(arr[0].re, 6.0f64, epsilon = 1e-12));
            assert!(approx::abs_diff_eq!(arr[0].im, 8.0f64, epsilon = 1e-12));
        } else {
            panic!("expected C128 tensor");
        }

        let prod = &a * &b;
        assert_eq!(prod.dtype(), DType::C128);
        if let Tensor::C128(arr) = prod {
            // (1+2i)(5+6i) = 5+6i+10i+12i^2 = 5+16i-12 = -7+16i
            assert!(approx::abs_diff_eq!(arr[0].re, -7.0f64, epsilon = 1e-12));
            assert!(approx::abs_diff_eq!(arr[0].im, 16.0f64, epsilon = 1e-12));
        } else {
            panic!("expected C128 tensor");
        }
    }

    #[test]
    fn test_arithmetic_owned() {
        let a = Tensor::from([1.0f64, 2.0]);
        let b = Tensor::from([3.0f64, 4.0]);
        let c = a + b; // owned Tensor + Tensor path
        assert_eq!(c.dtype(), DType::F64);
        if let Tensor::F64(arr) = c {
            assert_eq!(arr.as_slice().unwrap(), &[4.0f64, 6.0]);
        } else {
            panic!("expected F64 tensor");
        }
    }

    #[test]
    #[should_panic(expected = "dtype mismatch")]
    fn test_arithmetic_type_mismatch_panics() {
        let a = Tensor::from([1.0f64, 2.0]);
        let b = Tensor::from([1i32, 2]);
        let _ = &a + &b;
    }

    #[test]
    fn test_add_tensor_dtype_mismatch_returns_err() {
        let a = Tensor::from([1.0f64, 2.0]);
        let b = Tensor::from([1i32, 2]);
        let err = a.add_tensor(&b).unwrap_err();
        assert!(matches!(
            err,
            TensorError::DTypeMismatch {
                op: "add",
                lhs: DType::F64,
                rhs: DType::I32
            }
        ));
    }

    #[test]
    fn test_add_tensor_shape_mismatch_returns_err() {
        let a = Tensor::from([1.0f64, 2.0, 3.0]);
        let b = Tensor::from([1.0f64, 2.0, 3.0, 4.0]);
        let err = a.add_tensor(&b).unwrap_err();
        match err {
            TensorError::ShapeMismatch { lhs, rhs } => {
                assert_eq!(lhs, vec![3]);
                assert_eq!(rhs, vec![4]);
            }
            _ => panic!("expected ShapeMismatch, got {err:?}"),
        }
    }

    #[test]
    fn test_pow_dtype_mismatch_returns_err() {
        let base = Tensor::from([1.0f64, 2.0]);
        let exp = Tensor::from([1i32, 2]);
        let err = base.pow(&exp).unwrap_err();
        assert!(matches!(
            err,
            TensorError::DTypeMismatch {
                op: "pow",
                lhs: DType::F64,
                rhs: DType::I32
            }
        ));
    }

    // Broadcasting.

    #[test]
    fn test_arithmetic_broadcast() {
        // shape [2,3] + shape [3] -> shape [2,3]
        let a_data = ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f64; 6]).unwrap();
        let b_data = ndarray::Array::from_shape_vec(IxDyn(&[3]), vec![1.0f64, 2.0, 3.0]).unwrap();
        let a = Tensor::from(a_data);
        let b = Tensor::from(b_data);
        let c = &a + &b;
        assert_eq!(c.shape(), &[2, 3]);
        if let Tensor::F64(arr) = c {
            // row 0: [2.0, 3.0, 4.0], row 1: [2.0, 3.0, 4.0]
            assert!(approx::abs_diff_eq!(arr[[0, 0]], 2.0f64, epsilon = 1e-12));
            assert!(approx::abs_diff_eq!(arr[[0, 2]], 4.0f64, epsilon = 1e-12));
            assert!(approx::abs_diff_eq!(arr[[1, 1]], 3.0f64, epsilon = 1e-12));
        } else {
            panic!("expected F64 tensor");
        }
    }

    #[test]
    fn test_broadcast_scalar() {
        // shape [4] * shape [1] -> shape [4]
        let a = Tensor::from([1.0f64, 2.0, 3.0, 4.0]);
        let b = Tensor::from([10.0f64]);
        let c = &a * &b;
        assert_eq!(c.shape(), &[4]);
        if let Tensor::F64(arr) = c {
            assert_eq!(arr.as_slice().unwrap(), &[10.0f64, 20.0, 30.0, 40.0]);
        } else {
            panic!("expected F64 tensor");
        }
    }

    #[test]
    fn test_pow_shape_mismatch_returns_err() {
        let a = Tensor::from([1.0f64, 2.0, 3.0]);
        let b = Tensor::from([1.0f64, 2.0, 3.0, 4.0]);
        let err = a.pow(&b).unwrap_err();
        match err {
            TensorError::ShapeMismatch { lhs, rhs } => {
                assert_eq!(lhs, vec![3]);
                assert_eq!(rhs, vec![4]);
            }
            _ => panic!("expected ShapeMismatch, got {err:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "not broadcast-compatible")]
    fn test_op_panics_on_shape_mismatch() {
        let a = Tensor::from([1.0f64, 2.0, 3.0]);
        let b = Tensor::from([1.0f64, 2.0, 3.0, 4.0]);
        let _ = &a + &b;
    }

    // Per-dtype binop, pow, and cast dispatch coverage.

    #[test]
    fn test_binops_dtype_dispatch() {
        let mut fails: Vec<String> = vec![];

        // Check Add/Sub/Mul/Div/Rem with all real dtypes
        macro_rules! check_real {
            ($variant:ident, $t:ty) => {{
                let a = Tensor::from([6 as $t, 4 as $t]);
                let b = Tensor::from([3 as $t, 2 as $t]);
                for (op_name, got, want) in [
                    ("add", &a + &b, [9 as $t, 6 as $t]),
                    ("sub", &a - &b, [3 as $t, 2 as $t]),
                    ("mul", &a * &b, [18 as $t, 8 as $t]),
                    ("div", &a / &b, [2 as $t, 2 as $t]),
                    ("rem", &a % &b, [0 as $t, 0 as $t]),
                ] {
                    if let Tensor::$variant(arr) = got {
                        if arr.as_slice().unwrap() != want {
                            fails.push(format!(
                                "{} {op_name}: got {arr:?}, want {want:?}",
                                stringify!($variant),
                            ));
                        }
                    } else {
                        fails.push(format!("{} {op_name}: wrong variant", stringify!($variant)));
                    }
                }
            }};
        }
        check_real!(I8, i8);
        check_real!(I16, i16);
        check_real!(I32, i32);
        check_real!(I64, i64);
        check_real!(U8, u8);
        check_real!(U16, u16);
        check_real!(U32, u32);
        check_real!(U64, u64);
        check_real!(F32, f32);
        check_real!(F64, f64);

        // Check the same ops, but not Rem, with complex dtypes
        macro_rules! check_complex {
            ($variant:ident, $ctor:ident, $t:ty) => {{
                let c = |re: $t| $ctor::new(re, 0.0);
                let a = Tensor::from([c(6.0), c(4.0)]);
                let b = Tensor::from([c(3.0), c(2.0)]);
                for (op_name, got, want) in [
                    ("add", &a + &b, [c(9.0), c(6.0)]),
                    ("sub", &a - &b, [c(3.0), c(2.0)]),
                    ("mul", &a * &b, [c(18.0), c(8.0)]),
                    ("div", &a / &b, [c(2.0), c(2.0)]),
                ] {
                    if let Tensor::$variant(arr) = got {
                        if arr.as_slice().unwrap() != want {
                            fails.push(format!(
                                "{} {op_name}: got {arr:?}, want {want:?}",
                                stringify!($variant),
                            ));
                        }
                    } else {
                        fails.push(format!("{} {op_name}: wrong variant", stringify!($variant)));
                    }
                }
            }};
        }
        check_complex!(C64, Complex32, f32);
        check_complex!(C128, Complex64, f64);

        assert_eq!(fails, Vec::<String>::new(), "binop failures: {fails:?}");
    }

    #[test]
    fn test_rem_complex_returns_err() {
        // C128 % C128 is unsupported
        let a = Tensor::from([Complex64::new(1.0, 0.0)]);
        let b = Tensor::from([Complex64::new(1.0, 0.0)]);
        let err = a.rem_tensor(&b).unwrap_err();
        assert!(matches!(
            err,
            TensorError::UnsupportedDType {
                op: "rem",
                dtype: DType::C128
            }
        ));
    }

    #[test]
    fn test_integer_zero_divisor_gives_zero() {
        let a = Tensor::from([7_i64, 7]);
        let b = Tensor::from([0_i64, 2]);
        assert_eq!(a.div_tensor(&b).unwrap(), Tensor::from([0_i64, 3]));
        assert_eq!(a.rem_tensor(&b).unwrap(), Tensor::from([0_i64, 1]));

        // The one division that overflows wraps, as it does in NumPy.
        let min = Tensor::from([i8::MIN]);
        assert_eq!(min.div_tensor(&Tensor::from([-1_i8])).unwrap(), min);
    }

    #[test]
    fn test_integer_arithmetic_wraps() {
        let a = Tensor::from([1_u8]);
        assert_eq!(
            a.sub_tensor(&Tensor::from([5_u8])).unwrap(),
            Tensor::from([252_u8])
        );
        assert_eq!(
            Tensor::from([1_i64 << 62])
                .mul_tensor(&Tensor::from([4_i64]))
                .unwrap(),
            Tensor::from([0_i64])
        );
        assert_eq!(
            Tensor::from([2_i64]).pow(&Tensor::from([100_i64])).unwrap(),
            Tensor::from([0_i64])
        );
    }

    #[test]
    fn test_negative_exponent_returns_err() {
        assert!(matches!(
            Tensor::from([2_i64])
                .pow(&Tensor::from([-1_i64]))
                .unwrap_err(),
            TensorError::NegativeExponent { dtype: DType::I64 }
        ));
    }

    #[test]
    fn test_pow_dtype_dispatch() {
        let mut fails: Vec<String> = vec![];

        macro_rules! check_int {
            ($variant:ident, $t:ty) => {{
                let base = Tensor::from([2 as $t, 3 as $t]);
                let exp = Tensor::from([3 as $t, 2 as $t]);
                match base.pow(&exp).unwrap() {
                    Tensor::$variant(arr) => {
                        if arr.as_slice().unwrap() != [8 as $t, 9 as $t] {
                            fails.push(format!("{} pow: got {arr:?}", stringify!($variant)));
                        }
                    }
                    other => fails.push(format!(
                        "{} pow: wrong variant {}",
                        stringify!($variant),
                        other.dtype()
                    )),
                }
            }};
        }
        check_int!(I8, i8);
        check_int!(I16, i16);
        check_int!(I32, i32);
        check_int!(I64, i64);
        check_int!(U8, u8);
        check_int!(U16, u16);
        check_int!(U32, u32);
        check_int!(U64, u64);

        macro_rules! check_float {
            ($variant:ident, $t:ty, $eps:expr) => {{
                let base = Tensor::from([2.0 as $t, 3.0 as $t]);
                let exp = Tensor::from([3.0 as $t, 2.0 as $t]);
                match base.pow(&exp).unwrap() {
                    Tensor::$variant(arr) => {
                        if !approx::abs_diff_eq!(arr[0], 8.0 as $t, epsilon = $eps)
                            || !approx::abs_diff_eq!(arr[1], 9.0 as $t, epsilon = $eps)
                        {
                            fails.push(format!("{} pow: got {arr:?}", stringify!($variant)));
                        }
                    }
                    other => fails.push(format!(
                        "{} pow: wrong variant {}",
                        stringify!($variant),
                        other.dtype()
                    )),
                }
            }};
        }
        check_float!(F32, f32, 1e-4);
        check_float!(F64, f64, 1e-10);

        macro_rules! check_complex {
            ($variant:ident, $ctor:ident, $t:ty, $eps:expr) => {{
                let c = |re: $t| $ctor::new(re, 0.0);
                let base = Tensor::from([c(2.0), c(3.0)]);
                let exp = Tensor::from([c(3.0), c(2.0)]);
                match base.pow(&exp).unwrap() {
                    Tensor::$variant(arr) => {
                        if !approx::abs_diff_eq!(arr[0].re, 8.0 as $t, epsilon = $eps)
                            || !approx::abs_diff_eq!(arr[1].re, 9.0 as $t, epsilon = $eps)
                            || !approx::abs_diff_eq!(arr[0].im, 0.0 as $t, epsilon = $eps)
                            || !approx::abs_diff_eq!(arr[1].im, 0.0 as $t, epsilon = $eps)
                        {
                            fails.push(format!("{} pow: got {arr:?}", stringify!($variant)));
                        }
                    }
                    other => fails.push(format!(
                        "{} pow: wrong variant {}",
                        stringify!($variant),
                        other.dtype()
                    )),
                }
            }};
        }
        check_complex!(C64, Complex32, f32, 1e-4);
        check_complex!(C128, Complex64, f64, 1e-10);

        assert_eq!(fails, Vec::<String>::new(), "pow failures: {fails:?}");
    }

    #[test]
    fn test_cast_dispatch() {
        // Loop every real-source dtype against every target to cover every arm
        // of `cast_real!`. The complex-source arms are covered by
        // `test_cast_complex_to_complex` and the explicit `C64 -> C128` check
        // below.
        let mut fails: Vec<String> = vec![];

        let all_targets = [
            DType::Bit,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::F32,
            DType::F64,
            DType::C64,
            DType::C128,
        ];
        let sources = [
            Tensor::Bit(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1u8).into_shared()),
            Tensor::U8(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1u8).into_shared()),
            Tensor::U16(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1u16).into_shared()),
            Tensor::U32(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1u32).into_shared()),
            Tensor::U64(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1u64).into_shared()),
            Tensor::I8(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1i8).into_shared()),
            Tensor::I16(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1i16).into_shared()),
            Tensor::I32(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1i32).into_shared()),
            Tensor::I64(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1i64).into_shared()),
            Tensor::F32(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1.0f32).into_shared()),
            Tensor::F64(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1.0f64).into_shared()),
        ];
        for src in sources {
            let src_dtype = src.dtype();
            for target in all_targets {
                let casted = src.clone().cast(target);
                if casted.dtype() != target {
                    fails.push(format!(
                        "{src_dtype} -> {target}: dtype was {}",
                        casted.dtype()
                    ));
                }
            }
        }

        // C64 -> C128.
        let c64_src = Tensor::from([Complex32::new(1.0, 2.0)]);
        let casted = c64_src.cast(DType::C128);
        assert_eq!(casted.dtype(), DType::C128);
        if let Tensor::C128(arr) = casted {
            assert!(approx::abs_diff_eq!(arr[0].re, 1.0_f64, epsilon = 1e-6));
            assert!(approx::abs_diff_eq!(arr[0].im, 2.0_f64, epsilon = 1e-6));
        }

        // Spot-check a numeric value (Bit(1) -> F64 -> 1.0).
        let bit_to_f64 = Tensor::Bit(ndarray::ArrayD::from_elem(IxDyn(&[2]), 1u8).into_shared())
            .cast(DType::F64);
        if let Tensor::F64(arr) = bit_to_f64 {
            assert_eq!(arr.as_slice().unwrap(), &[1.0_f64, 1.0]);
        } else {
            fails.push("Bit -> F64 produced wrong variant".into());
        }

        assert_eq!(fails, Vec::<String>::new(), "cast failures: {fails:?}");
    }
}
