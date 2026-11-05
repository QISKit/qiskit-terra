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

//! NumPy-style broadcasting helpers.

use ndarray::{ArcArrayD, IxDyn, Zip};

use super::TensorError;

/// Pair up the axes of two shapes, right-aligned, padding the shorter one with `pad`.
pub(super) fn align_axes<'a, T: Copy>(
    a: &'a [T],
    b: &'a [T],
    pad: T,
) -> impl Iterator<Item = (T, T)> + 'a {
    let ndim = a.len().max(b.len());
    (0..ndim).map(move |i| {
        let axis = |shape: &[T]| {
            let offset = ndim - shape.len();
            if i >= offset { shape[i - offset] } else { pad }
        };
        (axis(a), axis(b))
    })
}

/// Compute the NumPy-style broadcast shape for two operand shapes, or
/// return [`TensorError::ShapeMismatch`] if they are not broadcast-compatible.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, TensorError> {
    align_axes(a, b, 1)
        .map(|pair| match pair {
            (x, y) if x == y => Ok(x),
            (1, y) => Ok(y),
            (x, 1) => Ok(x),
            _ => Err(TensorError::ShapeMismatch {
                lhs: a.to_vec(),
                rhs: b.to_vec(),
            }),
        })
        .collect()
}

/// Apply an element-wise binary operation to two arrays with NumPy-style broadcasting.
///
/// ndarray's arithmetic operators broadcast on their own, so this is for an operation with no Rust
/// operator, such as `pow`.
pub(super) fn broadcast_elementwise<T, F>(
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

#[cfg(test)]
mod test {
    use ndarray::ArrayD;

    use super::*;

    fn arr(shape: &[usize], data: Vec<f64>) -> ArcArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(shape), data)
            .unwrap()
            .into_shared()
    }

    #[test]
    fn test_align_axes_pairs_from_the_right() {
        let pairs: Vec<_> = align_axes(&[2, 3], &[3], 1).collect();
        assert_eq!(pairs, [(2, 1), (3, 3)]);
    }

    #[test]
    fn test_align_axes_pads_either_side() {
        assert_eq!(
            align_axes(&[4], &[2, 4], 1).collect::<Vec<_>>(),
            [(1, 2), (4, 4)]
        );
        assert!(align_axes::<usize>(&[], &[], 1).next().is_none());
    }

    #[test]
    fn test_broadcast_shape_stretches_a_unit_axis() {
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap(), [2, 3]);
        assert_eq!(broadcast_shape(&[4], &[1]).unwrap(), [4]);
        assert_eq!(broadcast_shape(&[1], &[4]).unwrap(), [4]);
        assert!(broadcast_shape(&[], &[]).unwrap().is_empty());
    }

    #[test]
    fn test_broadcast_shape_reports_both_operands() {
        assert!(matches!(
            broadcast_shape(&[3], &[4]).unwrap_err(),
            TensorError::ShapeMismatch { lhs, rhs } if lhs == [3] && rhs == [4]
        ));
    }

    #[test]
    fn test_broadcast_elementwise_spans_the_result_shape() {
        let a = arr(&[2, 1], vec![1.0, 2.0]);
        let b = arr(&[3], vec![10.0, 20.0, 30.0]);
        let out = broadcast_elementwise(&a, &b, |&x, &y| x + y).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            [11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
        );
    }

    #[test]
    fn test_broadcast_elementwise_forwards_a_shape_mismatch() {
        let a = arr(&[3], vec![1.0, 2.0, 3.0]);
        let b = arr(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(
            broadcast_elementwise(&a, &b, |&x, &y| x + y).unwrap_err(),
            TensorError::ShapeMismatch { lhs, rhs } if lhs == [3] && rhs == [4]
        ));
    }
}
