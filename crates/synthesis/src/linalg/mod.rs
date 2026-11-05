// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use approx::{abs_diff_eq, relative_ne};
use faer::Mat;
use faer::MatRef;
use nalgebra::{DMatrix, DMatrixView, Dim, Dyn, MatrixView, Scalar, ViewStorage};
use ndarray::ArrayView2;
use ndarray::ShapeBuilder;
use num_complex::Complex64;
use pyo3::PyErr;
use thiserror::Error;

use crate::QiskitError;

pub mod cos_sin_decomp;

const ATOL_DEFAULT: f64 = 1e-8;
const RTOL_DEFAULT: f64 = 1e-5;

/// Tolerance used for debug checking
pub const VERIFY_TOL: f64 = 1e-7;

/// Errors that might occur in linear algebra computations
#[derive(Error, Debug)]
pub enum LinAlgError {
    #[error("eigen decomposition failed")]
    EigenDecompositionFailed,

    #[error("SVD decomposition failed")]
    SVDDecompositionFailed,
}

impl From<LinAlgError> for PyErr {
    fn from(error: LinAlgError) -> Self {
        match error {
            LinAlgError::EigenDecompositionFailed => QiskitError::new_err(
                "Internal eigendecomposition failed. \
                This can point to a numerical tolerance issue.",
            ),

            LinAlgError::SVDDecompositionFailed => QiskitError::new_err(
                "Internal SVD decomposition failed. \
                This can point to a numerical tolerance issue.",
            ),
        }
    }
}

#[inline]
pub fn nalgebra_array_view<T: Scalar, R: Dim, C: Dim>(mat: MatrixView<T, R, C>) -> ArrayView2<T> {
    let dim = ndarray::Dim(mat.shape());
    let strides = ndarray::Dim(mat.strides());
    // SAFETY: We know the array is a 2d array from nalgebra and we get the pointer and memory
    // layout description from nalgebra so we don't need to check for invalid format as nalgebra
    // has already validated this
    unsafe { ArrayView2::from_shape_ptr(dim.strides(strides), mat.get_unchecked(0)) }
}

#[inline]
pub fn ndarray_to_faer<T>(array: ArrayView2<'_, T>) -> MatRef<'_, T> {
    // This function's code is based on faer-ext's IntoFaer::into_faer implementation for ArrayView2:
    // https://codeberg.org/sarah-quinones/faer-ext/src/commit/0f055b39529c94d1a000982df745cb9ce170f994/src/lib.rs#L108-L114
    let nrows = array.nrows();
    let ncols = array.ncols();
    let strides: [isize; 2] = array.strides().try_into().unwrap();
    let ptr = array.as_ptr();
    // SAFETY: We know the array is a 2d array from ndarray and we get the pointer and memory layout
    // description from ndarray and can be assumed to be valid.
    unsafe { faer::MatRef::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
}

#[inline]
pub fn faer_to_ndarray<T>(mat: MatRef<'_, T>) -> ArrayView2<'_, T> {
    // This function's code is based on faer-ext's IntoNdarray::into_ndarray implementation at:
    // https://codeberg.org/sarah-quinones/faer-ext/src/commit/0f055b39529c94d1a000982df745cb9ce170f994/src/lib.rs#L134-L141
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let row_stride: usize = mat.row_stride().try_into().unwrap();
    let col_stride: usize = mat.col_stride().try_into().unwrap();
    let ptr = mat.as_ptr();
    // SAFETY: We know the array is a 2d array from nalgebra and we get the pointer and memory layout
    // description from faer and can be assumed to be valid since the constraints on
    // `ArrayView2::from_shape_ptr()`
    // (https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html#method.from_shape_ptr)
    // should be be met for a faer Mat.
    unsafe { ArrayView2::from_shape_ptr((nrows, ncols).strides((row_stride, col_stride)), ptr) }
}

#[inline]
pub(crate) fn nalgebra_to_faer<R: Dim, C: Dim, RStride: Dim, CStride: Dim, T: nalgebra::Scalar>(
    mat: MatrixView<'_, T, R, C, RStride, CStride>,
) -> MatRef<'_, T> {
    let dim = ::ndarray::Dim(mat.shape());
    let strides = ::ndarray::Dim(mat.strides());

    // SAFETY: We know the array is a 2d array from nalgebra and we get the pointer and memory layout
    // description from nalgebra and can be assumed to be valid since the constraints on
    // `ArrayView2::from_shape_ptr()`
    // (https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html#method.from_shape_ptr)
    // should be be met for a valid nalgebra matrix.
    let array = unsafe { ArrayView2::from_shape_ptr(dim.strides(strides), mat.get_unchecked(0)) };
    ndarray_to_faer(array)
}

/// Convert a faer MatRef into a nalgebra MatrixView without copying
///
/// This function has some potential sharp edeges around converting strided
/// views. If you are using a strided `MatRef` you'll typically want to ensure
/// the underlying matrix the view is over is contiguous. Specifically if it's
/// not there is a potential path to undefined behavior when using nalgebra
/// methods like `MatrixView::into_slice()` which doesn't understand
/// striding and will access the memory as if the array was a contiguous R x C
/// matrix. If using striding here it's best to not ever call `into_slice()`. There
/// might also be similar sharp edges with strided matrices.
#[inline]
pub(crate) fn faer_to_nalgebra(
    mat: MatRef<'_, Complex64>,
) -> MatrixView<'_, Complex64, Dyn, Dyn, Dyn, Dyn> {
    // This function's code is based on faer-ext's IntoNalgebra::into_nalgebra implementation at:
    // https://codeberg.org/sarah-quinones/faer-ext/src/commit/0f055b39529c94d1a000982df745cb9ce170f994/src/lib.rs#L77-L96

    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let row_stride = mat.row_stride();
    let col_stride = mat.col_stride();

    let ptr = mat.as_ptr();
    // SAFETY: Pointer came from a faer MatRef as does the description of the memory layout
    // so creating a view of the data from the from the pointer is safe unless the faer object
    // is already corrupt. nalgebra doesn't support negative striding so that panics and we
    // only work with positive strides
    unsafe {
        MatrixView::<'_, Complex64, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
            '_,
            Complex64,
            Dyn,
            Dyn,
            Dyn,
            Dyn,
        >::from_raw_parts(
            ptr,
            (Dyn(nrows), Dyn(ncols)),
            (
                Dyn(row_stride
                    .try_into()
                    .expect("only works for positive strides")),
                Dyn(col_stride
                    .try_into()
                    .expect("only works for positive strides")),
            ),
        ))
    }
}

/// Check whether the given matrix is hermitian by comparing it (up to tolerance) with its hermitian adjoint.
pub fn is_hermitian_matrix(mat: DMatrixView<Complex64>) -> bool {
    let shape = mat.shape();
    let adjoint = mat.adjoint();
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            if relative_ne!(
                mat[(i, j)],
                adjoint[(i, j)],
                epsilon = ATOL_DEFAULT,
                max_relative = RTOL_DEFAULT
            ) {
                return false;
            }
        }
    }
    true
}

/// Verify SVD decomposition gives the same unitary
fn verify_svd_decomp(
    mat: DMatrixView<Complex64>,
    v: DMatrixView<Complex64>,
    s: DMatrixView<Complex64>,
    w: DMatrixView<Complex64>,
) -> bool {
    let mat_check = v * s * w;
    abs_diff_eq!(mat, mat_check.as_view(), epsilon = VERIFY_TOL)
}

/// Verifies the given matrix U is unitary by comparing U*U to the identity matrix
pub fn verify_unitary(u: &DMatrix<Complex64>) -> bool {
    let n = u.shape().0;

    let id_mat = DMatrix::identity(n, n);
    let uu = u.adjoint() * u;

    abs_diff_eq!(uu, id_mat, epsilon = VERIFY_TOL)
}

/// Given a matrix that is "close" to unitary, returns the closest
/// unitary matrix.
/// See https://michaelgoerz.net/notes/finding-the-closest-unitary-for-a-given-matrix/,
pub fn closest_unitary(mat: DMatrixView<Complex64>) -> DMatrix<Complex64> {
    let (u, _sigma, v_t) = svd_decomposition(mat);
    &u * &v_t
}

/// Calculate the condition number of a matrix w.r.t the L2 norm
/// using SVD
pub fn condition_number(mat: DMatrix<Complex64>) -> Option<f64> {
    let svd = mat.svd(false, false);
    let singular_values = svd.singular_values;

    if singular_values.is_empty() {
        return None;
    }

    let max_sv = singular_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_sv = singular_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    if min_sv == 0.0 {
        return None; // Singular matrix
    }

    Some(max_sv / min_sv)
}

/// Returns the SVD decomposition of the given matrix M as three matrices A,S,B such that
/// M=ASB (In the usual notations, M=USV*, and this function returns A=U and B=V*)
pub fn svd_decomposition(
    mat: DMatrixView<Complex64>,
) -> (DMatrix<Complex64>, DMatrix<Complex64>, DMatrix<Complex64>) {
    let mat_view: DMatrixView<Complex64> = mat.as_view();
    let faer_mat: MatRef<Complex64> = nalgebra_to_faer(mat_view);

    let faer_svd = faer_mat.svd().unwrap();

    let u_faer = faer_svd.U();
    let s_faer = faer_svd.S();
    let v_faer = faer_svd.V();

    let s_na = DMatrix::from_fn(u_faer.ncols(), v_faer.nrows(), |i, j| {
        if i == j {
            s_faer[i]
        } else {
            Complex64::new(0.0, 0.0)
        }
    });

    let u_na = faer_to_nalgebra(u_faer);
    let v_na = faer_to_nalgebra(v_faer).adjoint();

    debug_assert!(verify_svd_decomp(
        mat_view,
        u_na.as_view(),
        s_na.as_view(),
        v_na.as_view()
    ));

    (u_na.into(), s_na, v_na)
}

/// Result for the singular value decomposition (SVD) of a matrix `A`.
///
/// The decomposition is given by three matrices `(U, S, V)` such that `A = U * S * V^\dagger`.
pub struct SVDResult {
    pub u: Mat<Complex64>,
    pub s: Mat<Complex64>,
    pub v: Mat<Complex64>,
}

/// Runs singular valued decomposition on `mat`.
pub fn svd_decomposition_faer(mat: MatRef<Complex64>) -> Result<SVDResult, LinAlgError> {
    let svd = mat.svd().map_err(|_| LinAlgError::SVDDecompositionFailed)?;

    let n = mat.nrows();
    let u = svd.U().to_owned();
    let v = svd.V().to_owned();
    let mut s = Mat::zeros(n, n);
    s.diagonal_mut().copy_from(svd.S());

    let svd_result = SVDResult { u, s, v };
    debug_assert!(verify_svd_decomposition_faer(mat.as_ref(), &svd_result));

    Ok(svd_result)
}

/// Computes the eigenvalues and the eigenvectors of a square matrix
pub fn eigendecomposition_faer(
    mat: MatRef<Complex64>,
) -> Result<(Vec<Complex64>, Mat<Complex64>), LinAlgError> {
    let eigh = mat
        .eigen()
        .map_err(|_| LinAlgError::EigenDecompositionFailed)?;

    let vmat = eigh.U().to_owned();
    // unfortunately, we need to call closest_unitary_faer here
    let vmat = closest_unitary_faer(vmat.as_ref())?;
    let eigvals: Vec<Complex64> = eigh.S().column_vector().iter().copied().collect();
    Ok((eigvals, vmat))
}

pub fn closest_unitary_faer(mat: MatRef<Complex64>) -> Result<Mat<Complex64>, LinAlgError> {
    let svd = mat.svd().map_err(|_| LinAlgError::SVDDecompositionFailed)?;
    Ok(svd.U() * svd.V().adjoint())
}

pub fn from_diagonal_faer(diag: &[Complex64]) -> Mat<Complex64> {
    let n = diag.len();
    let mut mat = Mat::zeros(n, n);
    mat.diagonal_mut()
        .column_vector_mut()
        .iter_mut()
        .zip(diag)
        .for_each(|(x, y)| *x = *y);
    mat
}

/// Returns a block matrix `[a, b; c, d]`.
/// The matrices `a`, `b`, `c`, `d` are all assumed to be square matrices of the same size
pub fn block_matrix_faer(
    a: MatRef<Complex64>,
    b: MatRef<Complex64>,
    c: MatRef<Complex64>,
    d: MatRef<Complex64>,
) -> Mat<Complex64> {
    let n = a.nrows();
    let mut block_matrix = Mat::<Complex64>::zeros(2 * n, 2 * n);
    block_matrix.as_mut().submatrix_mut(0, 0, n, n).copy_from(a);
    block_matrix.as_mut().submatrix_mut(0, n, n, n).copy_from(b);
    block_matrix.as_mut().submatrix_mut(n, 0, n, n).copy_from(c);
    block_matrix.as_mut().submatrix_mut(n, n, n, n).copy_from(d);
    block_matrix
}

/// Verify SVD decomposition gives the same unitary
fn verify_svd_decomposition_faer(mat: MatRef<Complex64>, svd: &SVDResult) -> bool {
    let mat_check = svd.u.as_ref() * svd.s.as_ref() * svd.v.as_ref().adjoint();
    (mat - mat_check).norm_max() < VERIFY_TOL
}

/// Verifies the given matrix U is unitary by comparing U*U to the identity matrix
pub fn verify_unitary_faer(u: MatRef<Complex64>) -> bool {
    let n = u.shape().0;

    let id_mat = Mat::<Complex64>::identity(n, n);
    let uu = u.adjoint() * u;

    (uu.as_ref() - id_mat.as_ref()).norm_max() < VERIFY_TOL
}

// check whether a matrix is zero (up to tolerance)
pub fn is_zero_matrix_faer(mat: MatRef<Complex64>, atol: Option<f64>) -> bool {
    let atol = atol.unwrap_or(1e-12);
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            if !abs_diff_eq!(mat[(i, j)], Complex64::ZERO, epsilon = atol) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_basic_faer_to_nalgebra_conversion() {
        let matrix = Mat::from_fn(10, 10, |i, j| {
            Complex64::new(i as f64, 0.0) + Complex64::new(j as f64 * 10., 0.)
        });
        let mat_view = faer_to_nalgebra(matrix.as_ref());
        let expected = DMatrix::from_fn(10, 10, |i, j| {
            Complex64::new(i as f64, 0.0) + Complex64::new(j as f64 * 10., 0.)
        });
        assert_eq!(mat_view, expected);
    }

    #[cfg(not(miri))] // TODO: Remove this after dimforge/nalgebra#1562 is released
    #[test]
    fn test_transpose_faer_to_nalgebra_conversion() {
        let matrix = Mat::from_fn(10, 10, |i, j| {
            Complex64::new(i as f64, 0.0) + Complex64::new(j as f64 * 10., 0.)
        });
        let mat_view = faer_to_nalgebra(matrix.transpose());
        let expected = DMatrix::from_fn(10, 10, |i, j| {
            Complex64::new(j as f64, 0.0) + Complex64::new(i as f64 * 10., 0.)
        });
        assert_eq!(mat_view, expected);
    }

    #[test]
    #[should_panic]
    fn test_negative_strided_view_faer_to_nalgebra_conversion() {
        let matrix = Mat::identity(10, 10);
        faer_to_nalgebra(matrix.reverse_rows());
    }
}
