// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::QiskitError;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

pub mod lnn;
pub mod pmh;
pub mod utils;

#[pyfunction]
#[pyo3(signature = (mat, ncols=None, full_elim=false))]
/// Gauss elimination of a matrix mat with m rows and n columns.
/// If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
/// Modifies the matrix mat in-place, and returns the permutation perm that was done
/// on the rows during the process. perm[0 : rank] represents the indices of linearly
/// independent rows in the original matrix.
/// Args:
///     mat: a boolean matrix with n rows and m columns
///     ncols: the number of columns for the gaussian elimination,
///            if ncols=None, then the elimination is done over all the columns
///     full_elim: whether to do a full elimination, or partial (upper triangular form)
/// Returns:
///     perm: the permutation perm that was done on the rows during the process
fn gauss_elimination_with_perm(
    py: Python,
    mut mat: PyReadwriteArray2<bool>,
    ncols: Option<usize>,
    full_elim: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let matmut = mat.as_array_mut();
    let perm = utils::gauss_elimination_with_perm_inner(matmut, ncols, full_elim);
    perm.into_py_any(py)
}

#[pyfunction]
#[pyo3(signature = (mat, ncols=None, full_elim=false))]
/// Gauss elimination of a matrix mat with m rows and n columns.
/// If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
/// This function modifies the input matrix in-place.
/// Args:
///     mat: a boolean matrix with n rows and m columns
///     ncols: the number of columns for the gaussian elimination,
///            if ncols=None, then the elimination is done over all the columns
///     full_elim: whether to do a full elimination, or partial (upper triangular form)
fn gauss_elimination(
    mut mat: PyReadwriteArray2<bool>,
    ncols: Option<usize>,
    full_elim: Option<bool>,
) {
    let matmut = mat.as_array_mut();
    let _perm = utils::gauss_elimination_with_perm_inner(matmut, ncols, full_elim);
}

#[pyfunction]
#[pyo3(signature = (mat))]
/// Given a boolean matrix mat after Gaussian elimination, computes its rank
/// (i.e. simply the number of nonzero rows)
/// Args:
///     mat: a boolean matrix after gaussian elimination
/// Returns:
///     rank: the rank of the matrix
fn compute_rank_after_gauss_elim(py: Python, mat: PyReadonlyArray2<bool>) -> PyResult<Py<PyAny>> {
    let view = mat.as_array();
    let rank = utils::compute_rank_after_gauss_elim_inner(view);
    rank.into_py_any(py)
}

#[pyfunction]
#[pyo3(signature = (mat))]
/// Given a boolean matrix mat computes its rank
/// Args:
///     mat: a boolean matrix
/// Returns:
///     rank: the rank of the matrix
fn compute_rank(py: Python, mat: PyReadonlyArray2<bool>) -> PyResult<Py<PyAny>> {
    let rank = utils::compute_rank_inner(mat.as_array());
    rank.into_py_any(py)
}

#[pyfunction]
#[pyo3(signature = (mat, verify=false))]
/// Given a boolean matrix mat, tries to calculate its inverse matrix
/// Args:
///    mat: a boolean square matrix.
///    verify: if True asserts that the multiplication of mat and its inverse is the identity matrix.
/// Returns:
///   the inverse matrix.
/// Raises:
///  QiskitError: if the matrix is not square or not invertible.
pub fn calc_inverse_matrix(
    py: Python,
    mat: PyReadonlyArray2<bool>,
    verify: Option<bool>,
) -> PyResult<Py<PyArray2<bool>>> {
    let view = mat.as_array();
    let invmat =
        utils::calc_inverse_matrix_inner(view, verify.is_some()).map_err(QiskitError::new_err)?;
    Ok(invmat.into_pyarray(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (mat1, mat2))]
/// Binary matrix multiplication
/// Args:
///     mat1: a boolean matrix
///     mat2: a boolean matrix
/// Returns:
///     a boolean matrix which is the multiplication of mat1 and mat2
/// Raises:
///     QiskitError: if the dimensions of mat1 and mat2 do not match
pub fn binary_matmul(
    py: Python,
    mat1: PyReadonlyArray2<bool>,
    mat2: PyReadonlyArray2<bool>,
) -> PyResult<Py<PyArray2<bool>>> {
    let view1 = mat1.as_array();
    let view2 = mat2.as_array();
    let result = utils::binary_matmul_inner(view1, view2).map_err(QiskitError::new_err)?;
    Ok(result.into_pyarray(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (mat, ctrl, trgt))]
/// Perform ROW operation on a matrix mat
fn row_op(mut mat: PyReadwriteArray2<bool>, ctrl: usize, trgt: usize) {
    let matmut = mat.as_array_mut();
    utils::_add_row_or_col(matmut, &false, ctrl, trgt)
}

#[pyfunction]
#[pyo3(signature = (mat, ctrl, trgt))]
/// Perform COL operation on a matrix mat (in the inverse direction)
fn col_op(mut mat: PyReadwriteArray2<bool>, ctrl: usize, trgt: usize) {
    let matmut = mat.as_array_mut();
    utils::_add_row_or_col(matmut, &true, trgt, ctrl)
}

#[pyfunction]
#[pyo3(signature = (num_qubits, seed=None))]
/// Generate a random invertible n x n binary matrix.
///  Args:
///     num_qubits: the matrix size.
///     seed: a random seed.
///  Returns:
///     np.ndarray: A random invertible binary matrix of size num_qubits.
fn random_invertible_binary_matrix(
    py: Python,
    num_qubits: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyArray2<bool>>> {
    let matrix = utils::random_invertible_binary_matrix_inner(num_qubits, seed);
    Ok(matrix.into_pyarray(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (mat))]
/// Check that a binary matrix is invertible.
/// Args:
///     mat: a binary matrix.
/// Returns:
///     bool: True if mat in invertible and False otherwise.
fn check_invertible_binary_matrix(mat: PyReadonlyArray2<bool>) -> bool {
    let view = mat.as_array();
    utils::check_invertible_binary_matrix_inner(view)
}

pub fn linear(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(gauss_elimination_with_perm))?;
    m.add_wrapped(wrap_pyfunction!(gauss_elimination))?;
    m.add_wrapped(wrap_pyfunction!(compute_rank_after_gauss_elim))?;
    m.add_wrapped(wrap_pyfunction!(compute_rank))?;
    m.add_wrapped(wrap_pyfunction!(calc_inverse_matrix))?;
    m.add_wrapped(wrap_pyfunction!(row_op))?;
    m.add_wrapped(wrap_pyfunction!(col_op))?;
    m.add_wrapped(wrap_pyfunction!(binary_matmul))?;
    m.add_wrapped(wrap_pyfunction!(random_invertible_binary_matrix))?;
    m.add_wrapped(wrap_pyfunction!(check_invertible_binary_matrix))?;
    m.add_wrapped(wrap_pyfunction!(pmh::synth_cnot_count_full_pmh))?;
    m.add_wrapped(wrap_pyfunction!(lnn::py_synth_cnot_depth_line_kms))?;
    m.add_wrapped(wrap_pyfunction!(lnn::py_synth_cnot_lnn_instructions))?;
    Ok(())
}
