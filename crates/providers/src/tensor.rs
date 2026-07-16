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

use ndarray::{ArcArray, ArrayD, IxDyn, Zip};
use num_complex::{Complex32, Complex64};
use std::fmt;
use thiserror::Error;

/// Dynamic-dimensional [`ArcArray`]; the storage type for every [`Tensor`] variant.
type ArcArrayD<T> = ArcArray<T, IxDyn>;

/// Errors returned by [`Tensor`] operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TensorError {
    /// The two operand tensors have different dtypes or a dtype that does not support the op.
    #[error("dtype mismatch in Tensor::{op}: lhs={lhs}, rhs={rhs}")]
    DTypeMismatch {
        op: &'static str,
        lhs: DType,
        rhs: DType,
    },
    /// The two operand shapes are not broadcast-compatible.
    #[error("shapes {lhs:?} and {rhs:?} are not broadcast-compatible")]
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },
}

/// The possible data types for a Tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    C128, // complex
    C64,
    F64, // real
    F32,
    I64, // signed integer
    I32,
    I16,
    I8,
    U64, // unsigned integer
    U32,
    U16,
    U8,
    Bit, // bool
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let string_repr = match self {
            DType::C128 => "C128",
            DType::C64 => "C64",
            DType::F64 => "F64",
            DType::F32 => "F32",
            DType::I64 => "I64",
            DType::I32 => "I32",
            DType::I16 => "I16",
            DType::I8 => "I8",
            DType::U64 => "U64",
            DType::U32 => "U32",
            DType::U16 => "U16",
            DType::U8 => "U8",
            DType::Bit => "Bit",
        };
        write!(f, "{string_repr}")
    }
}

/// A tensor dtype that is unknown but identified by name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DTypeVar {
    /// The variable name.
    pub name: String,
}

impl<T: Into<String>> From<T> for DTypeVar {
    fn from(value: T) -> Self {
        Self { name: value.into() }
    }
}

/// A tensor data type whose value is yet unknown, but will be the promotion of others.
#[derive(Debug, Clone)]
pub struct DTypePromotion {
    /// The dtype arguments to promote over.
    pub args: Vec<DTypeLike>,
}

impl<T: Into<Vec<DTypeLike>>> From<T> for DTypePromotion {
    fn from(args: T) -> Self {
        Self { args: args.into() }
    }
}

/// A tensor data type, known or unknown.
#[derive(Debug, Clone)]
pub enum DTypeLike {
    /// A fully resolved dtype.
    Concrete(DType),
    /// A dtype identified by a variable name, to be resolved later.
    Var(DTypeVar),
    /// A dtype that is the promotion of one or more other dtypes.
    Promotion(DTypePromotion),
}

/// Promote a pair of DTypes to the smallest type compatible with both.
///
/// QuantumProgram nodes often, but not necessarily, use this promotion rule
/// to determine their output type.
///
/// This function implements the same promotion rules as NumPy, modulo that we don't
/// need to contend with the arbitrary precision types for each type kind, and that
/// we omit F16 entirely because it's unstable in Rust:
/// https://numpy.org/doc/stable/reference/arrays.promotion.html#numerical-promotion
/// In short, if you view the linked diagram as a DAG, this function hard-codes the
/// least-common-descendant algorithm.
pub fn promotion(lhs: DType, rhs: DType) -> DType {
    use DType::*;

    match lhs {
        C128 => C128,

        C64 => match rhs {
            U32 | U64 | I32 | I64 | F64 | C128 => C128,
            _ => C64,
        },

        F64 => match rhs {
            C64 | C128 => C128,
            _ => F64,
        },

        F32 => match rhs {
            C128 => C128,
            C64 => C64,
            U32 | U64 | I32 | I64 | F64 => F64,
            _ => F32,
        },

        I64 => match rhs {
            C64 | C128 => C128,
            U64 | F32 | F64 => F64,
            _ => I64,
        },

        I32 => match rhs {
            C64 | C128 => C128,
            U64 | F32 | F64 => F64,
            U32 | I64 => I64,
            _ => I32,
        },

        I16 => match rhs {
            U64 => F64,
            U32 => I64,
            U16 => I32,
            Bit | U8 | I8 => I16,
            _ => rhs,
        },

        I8 => match rhs {
            U64 => F64,
            U32 => I64,
            U16 => I32,
            U8 => I16,
            Bit => I8,
            _ => rhs,
        },

        U64 => match rhs {
            C128 | C64 => C128,
            F32 | F64 | I8 | I16 | I32 | I64 => F64,
            _ => U64,
        },

        U32 => match rhs {
            C64 | C128 => C128,
            F32 | F64 => F64,
            I8 | I16 | I32 | I64 => I64,
            U64 => U64,
            _ => U32,
        },

        U16 => match rhs {
            I8 | I16 => I32,
            Bit | U8 => U16,
            _ => rhs,
        },

        U8 => match rhs {
            I8 => I16,
            Bit => U8,
            _ => rhs,
        },

        Bit => rhs,
    }
}

/// A tensor axis dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    /// A known size.
    Fixed(usize),
    /// An unresolved, named size.
    Named(String),
}

/// A specification of a tensor without any data.
#[derive(Debug, Clone)]
pub struct TensorType {
    /// The type of the tensor.
    pub dtype: DTypeLike,
    /// The shape of the tensor, possibly with axes of unknown size.
    pub shape: Vec<Dim>,
    /// Whether the tensor supports leading-axis (i.e. NumPy-style) broadcasting semantics.
    pub broadcastable: bool,
}

impl TensorType {
    /// Return a dimension vector if all sizes are fixed, or `None` if any are named.
    pub fn concrete_shape(&self) -> Option<Vec<usize>> {
        self.shape
            .iter()
            .map(|d| match d {
                Dim::Fixed(n) => Some(*n),
                Dim::Named(_) => None,
            })
            .collect()
    }
}

/// A tensor of one of the supported dtypes.
///
/// Each variant wraps a reference-counted dynamic ndarray ([`ArcArray`]).
///
/// This allows [`Tensor::clone`] to cause a refcount bump rather than a copy of
/// underlying data. Note that mutating the underlying buffer in place (via ndarray
/// methods that require `DataMut`) clones-on-write when the buffer is shared.
#[derive(Debug, Clone)]
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

/// Cast an array of a real numeric type to any supported dtype.
macro_rules! cast_real {
    ($arr:expr, $src:ty, $target:expr) => {
        match $target {
            DType::Bit => Tensor::Bit($arr.mapv(|x: $src| x as u8).into_shared()),
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

/// Cast an array of a complex type to a complex dtype (panics for real targets).
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

/// Compute the NumPy-style broadcast shape for two operand shapes, or
/// return [`TensorError::ShapeMismatch`] if they are not broadcast-compatible.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, TensorError> {
    let ndim = a.len().max(b.len());
    (0..ndim)
        .map(|i| {
            let dim_a = if i >= ndim - a.len() {
                a[i - (ndim - a.len())]
            } else {
                1
            };
            let dim_b = if i >= ndim - b.len() {
                b[i - (ndim - b.len())]
            } else {
                1
            };
            match (dim_a, dim_b) {
                (x, y) if x == y => Ok(x),
                (1, y) => Ok(y),
                (x, 1) => Ok(x),
                _ => Err(TensorError::ShapeMismatch {
                    lhs: a.to_vec(),
                    rhs: b.to_vec(),
                }),
            }
        })
        .collect()
}

/// Element-wise binary operation on two arrays with NumPy-style broadcasting.
///
/// Unlike ndarray's built-in arithmetic operators which handle broadcasting automatically,
/// this helper is needed for operations without a Rust operator (e.g. `pow`). Returns
/// [`TensorError::ShapeMismatch`] if the operand shapes are not broadcast-compatible.
fn broadcast_elementwise<T, F>(
    a: &ArcArrayD<T>,
    b: &ArcArrayD<T>,
    op: F,
) -> Result<ArcArrayD<T>, TensorError>
where
    T: Clone,
    F: Fn(&T, &T) -> T,
{
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let out_ix = IxDyn(&out_shape);
    let a_bc = a.broadcast(out_ix.clone()).expect("broadcast failed");
    let b_bc = b.broadcast(out_ix).expect("broadcast failed");
    Ok(Zip::from(a_bc).and(b_bc).map_collect(op).into_shared())
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
            dtype: DTypeLike::Concrete(self.dtype()),
            shape: self.shape().iter().map(|&n| Dim::Fixed(n)).collect(),
            broadcastable: false,
        }
    }

    /// Element-wise power with NumPy-style broadcasting.
    ///
    /// For integer types the exponent is cast to `u32`; negative integer exponents
    /// are not supported. Returns [`TensorError::DTypeMismatch`] if the operands have
    /// different dtypes (or a dtype that does not support `pow`), and
    /// [`TensorError::ShapeMismatch`] if the shapes are not broadcast-compatible.
    pub fn pow(&self, rhs: &Tensor) -> Result<Tensor, TensorError> {
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
            (Tensor::I8(a), Tensor::I8(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)).map(Tensor::I8)
            }
            (Tensor::I16(a), Tensor::I16(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)).map(Tensor::I16)
            }
            (Tensor::I32(a), Tensor::I32(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)).map(Tensor::I32)
            }
            (Tensor::I64(a), Tensor::I64(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)).map(Tensor::I64)
            }
            (Tensor::U8(a), Tensor::U8(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)).map(Tensor::U8)
            }
            (Tensor::U16(a), Tensor::U16(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)).map(Tensor::U16)
            }
            (Tensor::U32(a), Tensor::U32(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.pow(y)).map(Tensor::U32)
            }
            (Tensor::U64(a), Tensor::U64(b)) => {
                broadcast_elementwise(a, b, |&x, &y| x.pow(y as u32)).map(Tensor::U64)
            }
            _ => Err(TensorError::DTypeMismatch {
                op: "pow",
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            }),
        }
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

/// Define a fallible element-wise binary method on [`Tensor`] (e.g. `add_tensor`),
/// plus the corresponding [`std::ops`] trait impls that unwrap the `Result`.
///
/// The operand shapes are pre-validated with [`broadcast_shape`] so that the underlying
/// ndarray operator (which broadcasts but panics on shape mismatch) cannot panic.
macro_rules! impl_tensor_binop {
    ($trait:ident, $method:ident, $tensor_method:ident, $op:tt, $op_name:literal) => {
        impl Tensor {
            #[doc = concat!(
                "Element-wise `",
                $op_name,
                "` with NumPy-style broadcasting.\n\n",
                "Returns [`TensorError::DTypeMismatch`] if the operand dtypes differ ",
                "(or do not support this op), and [`TensorError::ShapeMismatch`] if ",
                "the shapes are not broadcast-compatible."
            )]
            pub fn $tensor_method(&self, rhs: &Tensor) -> Result<Tensor, TensorError> {
                broadcast_shape(self.shape(), rhs.shape())?;
                match (self, rhs) {
                    (Tensor::C128(a), Tensor::C128(b)) => Ok(Tensor::C128((a $op b).into_shared())),
                    (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64((a $op b).into_shared())),
                    (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64((a $op b).into_shared())),
                    (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32((a $op b).into_shared())),
                    (Tensor::I64(a), Tensor::I64(b)) => Ok(Tensor::I64((a $op b).into_shared())),
                    (Tensor::I32(a), Tensor::I32(b)) => Ok(Tensor::I32((a $op b).into_shared())),
                    (Tensor::I16(a), Tensor::I16(b)) => Ok(Tensor::I16((a $op b).into_shared())),
                    (Tensor::I8(a), Tensor::I8(b)) => Ok(Tensor::I8((a $op b).into_shared())),
                    (Tensor::U64(a), Tensor::U64(b)) => Ok(Tensor::U64((a $op b).into_shared())),
                    (Tensor::U32(a), Tensor::U32(b)) => Ok(Tensor::U32((a $op b).into_shared())),
                    (Tensor::U16(a), Tensor::U16(b)) => Ok(Tensor::U16((a $op b).into_shared())),
                    (Tensor::U8(a), Tensor::U8(b)) => Ok(Tensor::U8((a $op b).into_shared())),
                    _ => Err(TensorError::DTypeMismatch {
                        op: $op_name,
                        lhs: self.dtype(),
                        rhs: rhs.dtype(),
                    }),
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

impl_tensor_binop!(Add, add, add_tensor, +, "add");
impl_tensor_binop!(Sub, sub, sub_tensor, -, "sub");
impl_tensor_binop!(Mul, mul, mul_tensor, *, "mul");
impl_tensor_binop!(Div, div, div_tensor, /, "div");

// `Rem` is hand-written rather than going through `impl_tensor_binop!` because
// `num_complex` does not implement `%`, so the complex variants must be omitted.
impl Tensor {
    /// Element-wise `%` with NumPy-style broadcasting (real dtypes only).
    ///
    /// Returns [`TensorError::DTypeMismatch`] if the operand dtypes differ or are
    /// not supported by this op (e.g. complex), and [`TensorError::ShapeMismatch`]
    /// if the shapes are not broadcast-compatible.
    pub fn rem_tensor(&self, rhs: &Tensor) -> Result<Tensor, TensorError> {
        broadcast_shape(self.shape(), rhs.shape())?;
        match (self, rhs) {
            (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64((a % b).into_shared())),
            (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32((a % b).into_shared())),
            (Tensor::I64(a), Tensor::I64(b)) => Ok(Tensor::I64((a % b).into_shared())),
            (Tensor::I32(a), Tensor::I32(b)) => Ok(Tensor::I32((a % b).into_shared())),
            (Tensor::I16(a), Tensor::I16(b)) => Ok(Tensor::I16((a % b).into_shared())),
            (Tensor::I8(a), Tensor::I8(b)) => Ok(Tensor::I8((a % b).into_shared())),
            (Tensor::U64(a), Tensor::U64(b)) => Ok(Tensor::U64((a % b).into_shared())),
            (Tensor::U32(a), Tensor::U32(b)) => Ok(Tensor::U32((a % b).into_shared())),
            (Tensor::U16(a), Tensor::U16(b)) => Ok(Tensor::U16((a % b).into_shared())),
            (Tensor::U8(a), Tensor::U8(b)) => Ok(Tensor::U8((a % b).into_shared())),
            _ => Err(TensorError::DTypeMismatch {
                op: "rem",
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            }),
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

#[cfg(test)]
mod test {
    use super::*;

    const ALL_DTYPES: [DType; 13] = [
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

    #[test]
    fn test_promotion_against_promotion_dag() {
        use DType::*;
        use hashbrown::{HashMap, HashSet};
        use rustworkx_core::dag_algo::lexicographical_topological_sort;
        use rustworkx_core::petgraph::graph::{DiGraph, NodeIndex};
        use rustworkx_core::traversal::descendants;

        // define a DAG that implements all promotion rules; two DTypes
        // should be promoted to their least common descendant in the DAG
        let mut g: DiGraph<DType, ()> = DiGraph::new();
        let mut idx: HashMap<DType, NodeIndex> = HashMap::new();

        for &dtype in &ALL_DTYPES {
            idx.insert(dtype, g.add_node(dtype));
        }

        // within-kind promotions
        g.add_edge(idx[&U8], idx[&U16], ());
        g.add_edge(idx[&U16], idx[&U32], ());
        g.add_edge(idx[&U32], idx[&U64], ());

        g.add_edge(idx[&I8], idx[&I16], ());
        g.add_edge(idx[&I16], idx[&I32], ());
        g.add_edge(idx[&I32], idx[&I64], ());

        g.add_edge(idx[&F32], idx[&F64], ());

        g.add_edge(idx[&C64], idx[&C128], ());

        // bit promotions
        g.add_edge(idx[&Bit], idx[&U8], ());
        g.add_edge(idx[&Bit], idx[&I8], ());

        // uint promotions
        g.add_edge(idx[&U8], idx[&I16], ());
        g.add_edge(idx[&U16], idx[&I32], ());
        g.add_edge(idx[&U16], idx[&F32], ());
        g.add_edge(idx[&U32], idx[&I64], ());
        g.add_edge(idx[&U64], idx[&F64], ());

        // int promotions
        g.add_edge(idx[&I16], idx[&F32], ());
        g.add_edge(idx[&I32], idx[&F64], ());
        g.add_edge(idx[&I64], idx[&F64], ());

        // float promotions
        g.add_edge(idx[&F32], idx[&C64], ());
        g.add_edge(idx[&F64], idx[&C128], ());

        let order = lexicographical_topological_sort(
            &g,
            |n: NodeIndex| Ok::<usize, std::convert::Infallible>(n.index()),
            false,
            None,
        )
        .unwrap();

        let least_common_descendant = move |a: &DType, b: &DType| -> DType {
            let da: HashSet<_> = descendants(&g, idx[a]).collect();
            let db: HashSet<_> = descendants(&g, idx[b]).collect();
            let common: HashSet<NodeIndex> = da.intersection(&db).copied().collect();
            let least_idx = order.iter().find(|n| common.contains(*n)).unwrap();
            ALL_DTYPES[least_idx.index()]
        };

        for &a in &ALL_DTYPES {
            for &b in &ALL_DTYPES {
                assert_eq!(
                    promotion(a, b),
                    least_common_descendant(&a, &b),
                    "For promotion ({a}, {b})"
                )
            }
        }
    }

    #[test]
    fn test_promotion_idempotence() {
        for &a in &ALL_DTYPES {
            assert_eq!(promotion(a, a), a, "For promotion ({a}, {a})")
        }
    }

    #[test]
    fn test_promotion_commutativity() {
        for &a in &ALL_DTYPES {
            for &b in &ALL_DTYPES {
                assert_eq!(promotion(a, b), promotion(b, a), "For promotion ({a}, {b})")
            }
        }
    }

    #[test]
    fn test_tensor_type_concrete_shape() {
        assert_eq!(
            TensorType {
                dtype: DTypeLike::Concrete(DType::Bit),
                shape: vec![Dim::Fixed(3)],
                broadcastable: false,
            }
            .concrete_shape(),
            Some(vec![3])
        );

        assert_eq!(
            TensorType {
                dtype: DTypeLike::Concrete(DType::Bit),
                shape: vec![Dim::Fixed(3), Dim::Fixed(8)],
                broadcastable: true,
            }
            .concrete_shape(),
            Some(vec![3, 8])
        );

        assert_eq!(
            TensorType {
                dtype: DTypeLike::Concrete(DType::Bit),
                shape: vec![Dim::Fixed(3), Dim::Named("foo".into())],
                broadcastable: false,
            }
            .concrete_shape(),
            None
        );
    }

    // -----------------------------------------------------------------------
    // Construction, dtype, shape
    // -----------------------------------------------------------------------

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
        let tt = t.tensor_type();
        assert!(
            matches!(tt.dtype, DTypeLike::Concrete(DType::F64)),
            "expected Concrete(F64)"
        );
        assert_eq!(tt.shape, vec![Dim::Fixed(3)]);
        assert!(!tt.broadcastable);

        let arr = ndarray::Array::from_shape_vec(IxDyn(&[2, 4]), vec![0i16; 8]).unwrap();
        let t = Tensor::from(arr);
        let tt = t.tensor_type();
        assert!(
            matches!(tt.dtype, DTypeLike::Concrete(DType::I16)),
            "expected Concrete(I16)"
        );
        assert_eq!(tt.shape, vec![Dim::Fixed(2), Dim::Fixed(4)]);
    }

    // -----------------------------------------------------------------------
    // cast
    // -----------------------------------------------------------------------

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
    #[should_panic(expected = "cannot cast complex")]
    fn test_cast_complex_to_real_panics() {
        let t = Tensor::from([Complex64::new(1.0, 2.0)]);
        let _ = t.cast(DType::F64);
    }

    // -----------------------------------------------------------------------
    // pow
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Arithmetic operators
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Broadcasting
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Display & conversion helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_dtype_display() {
        use DType::*;
        let cases = [
            (C128, "C128"),
            (C64, "C64"),
            (F64, "F64"),
            (F32, "F32"),
            (I64, "I64"),
            (I32, "I32"),
            (I16, "I16"),
            (I8, "I8"),
            (U64, "U64"),
            (U32, "U32"),
            (U16, "U16"),
            (U8, "U8"),
            (Bit, "Bit"),
        ];
        let mut fails = vec![];
        for (dtype, expected) in cases {
            let got = format!("{dtype}");
            if got != expected {
                fails.push((dtype, expected, got));
            }
        }
        assert_eq!(fails, [], "DType Display mismatches: {fails:?}");
    }

    #[test]
    fn test_dtype_var_from() {
        let v = DTypeVar::from("x");
        assert_eq!(v.name, "x");

        let v = DTypeVar::from(String::from("alpha"));
        assert_eq!(v.name, "alpha");
    }

    #[test]
    fn test_dtype_promotion_from() {
        let args = vec![
            DTypeLike::Concrete(DType::F32),
            DTypeLike::Concrete(DType::I16),
        ];
        let p = DTypePromotion::from(args);
        assert_eq!(p.args.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Per-dtype binop, pow, and cast dispatch coverage
    // -----------------------------------------------------------------------

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
            TensorError::DTypeMismatch {
                op: "rem",
                lhs: DType::C128,
                rhs: DType::C128
            }
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
