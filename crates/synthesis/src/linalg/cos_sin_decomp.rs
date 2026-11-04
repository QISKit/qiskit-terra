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

// Based on the Cosine Sine decomposition TKET uses for unitary matrices:
//
// https://github.com/CQCL/tket/blob/a61bbac1c07487521acd8da463023f3c85c799e8/tket/src/Utils/CosSinDecomposition.cpp

use numpy::PyReadonlyArray2;
use numpy::ToPyArray;
use pyo3::prelude::*;

use crate::linalg::faer_to_ndarray;
use faer::{Mat, MatRef};
use num_complex::{Complex64, ComplexFloat};

const EPS: f64 = 1e-11;

pub struct CosSinDecompReturn {
    pub l0: Mat<Complex64>,
    pub l1: Mat<Complex64>,
    pub r0: Mat<Complex64>,
    pub r1: Mat<Complex64>,
    pub thetas: Vec<f64>,
}

/// Computes the cosine-sin decomposition (CSD) of a unitary matrix.
///
/// # Args
///
/// - `u`: The input unitary to decompose
///
/// # Returns
///
/// The tuple `(l0, l1, r0, r1, theta)` representing the CSD:
///
/// ```text
/// [l0   ] [c -s] [r0   ]
/// [   l1] [s  c] [   r1]
/// ```
///
/// - `l0`, `l1`, `r0`, and `r1` are unitaries of equal size,
/// - `theta` is the array of angles, with each angle in in the interval :math:`[0, \pi/2]`.
///
/// The diagonal matrices `c` and `s` contain the cosines and sines of angles in `theta`,
/// respectively. In particular, `c^2 + s^2 = I`.
///
/// Furthermore, the angles in `theta` are sorted in descending order, so:
/// - cosines are in ascending order,
/// - sines are in descending order.
pub fn cos_sin_decomposition(u: MatRef<Complex64>) -> CosSinDecompReturn {
    let shape = u.shape();
    let n = shape.0 / 2;
    // Upper left corner
    let u00 = u.submatrix(0, 0, n, n);
    // Upper right corner
    let u01 = u.submatrix(0, n, n, n);
    // Lower left corner
    let u10 = u.submatrix(n, 0, n, n);
    // Lower right corner
    let u11 = u.submatrix(n, n, n, n);

    // For the desired decomposition to exist, we must have:
    //   u00 = l0 c r0
    //   u01 = -l0 s r1
    //   u10 = l1 s r0
    //   u11 = l1 c r1
    // We will first find l0, c, and r0, then s and l1, and finally r1.

    // Apply SVD to u00
    let svd = u00.svd().expect("Problem with SVD decomposition");
    let l0 = svd.U();
    let r0 = svd.V().adjoint();
    let mut c: Vec<f64> = svd.S().column_vector().iter().map(|z| z.re()).collect();

    // We have u00 = l0 c r0, where l0 and r0 are unitary, and c is a diagonal matrix
    // with positive entries in the descending order. Also note that u00 is a submatrix
    // of a unitary matrix, and hence its singular values (the entries of c) are all <= 1
    // (well, due to floating-point precision errors, it is possible that some of
    // the c-values are just a tiny bit larger than 1, so we have to be careful about that).
    // However, we want the entries of c to be in the ascending order instead (otherwise,
    // we will not be able to guarantee that s is a diagonal matrix too). Fortunately,
    // it is easy to modify l0, c, and r0, so that this becomes true.
    let r0 = r0.reverse_rows().to_owned();
    let l0 = l0.reverse_cols().to_owned();
    c.reverse();

    // Apply QR to u10 r0*.
    // We have u10 r0* = l1 s, where l1 is unitary and s is upper-triangular.
    // Equivalently, u10 = l1 s r0.
    let r0_dag = r0.adjoint();
    let u10_r0_dag = u10 * r0_dag;
    let qr = u10_r0_dag.qr();
    let mut l1 = qr.compute_Q();
    let s = qr.R();

    // Claim: s is diagonal.
    // Proof: Since u is unitary, we have
    //     I = u00* u00 + u10* u10
    //       = (l0 c r0)* (l0 c r0) + (l1 s r0)* (l1 s r0)
    //       = r0* c* l0* l0 c r0 + r0* s* l1* l1 s r0
    //       = r0* c* c r0 + r0* s* s r0
    //       = r0* (c^2 + s* s) r0
    // So I = c^2 + (s* s), so (s* s) = I - c^2 is a diagonal matrix with non-
    // increasing entries in the range [0,1). As s is upper triangular, this
    // implies that s must be diagonal. (Proof by induction over the dimension n
    // of s: consider the two cases s_00 = 0 and s_00 != 0 and reduce to the n-1
    // case. Note: it is important that the entries of s are in descending order
    // for this proof to work.)

    // We want s to be real. This is not guaranteed, though it seems to be always
    // true in practice. In either case, it can be made possible by suitable adjusting
    // both s and l1 together.
    let mut s: Vec<Complex64> = s.diagonal().column_vector().iter().copied().collect();
    if s.iter().any(|x| x.im != 0.) {
        for (j, z) in s.iter_mut().enumerate() {
            let r = z.abs();
            if r > EPS {
                let w = z.conj() / r;
                *z *= w;
                l1.col_mut(j).iter_mut().for_each(|x| *x /= w);
            }
        }
    }

    // Now s is real and diagonal, and c^2 + s^2 = I.
    let mut s: Vec<f64> = s.iter().map(|x| x.re).collect();

    // Additionally, adjust l1 and s so that all entries in s are non-negative.
    // Again, this seems to be never needed in practice.
    for (j, val) in s.iter_mut().enumerate() {
        if *val < 0. {
            *val = -(*val);
            l1.col_mut(j).iter_mut().for_each(|x| *x = -(*x));
        }
    }

    // Finally compute r1, being careful not to divide by small things.
    // r1 should satisfy u01 = -l0 s r1 and u11 = l1 c r1, with these two
    // conditions being equivalent due to u being unitary.
    let mut r1 = Mat::<Complex64>::zeros(n, n);
    let l0_dag_u01 = l0.adjoint() * u01;
    let l1_dag_u11 = l1.adjoint() * u11;
    for i in 0..n {
        if s[i] > c[i] {
            for j in 0..n {
                r1[(i, j)] = -l0_dag_u01[(i, j)] / s[i];
            }
        } else {
            for j in 0..n {
                r1[(i, j)] = l1_dag_u11[(i, j)] / c[i];
            }
        }
    }

    // Compute the angles theta given approximate values of their cosines (entries of c)
    // and their sines (entries of s).
    // We can compute theta either as c.acos() or as s.asin(), however the first formula
    // is not very accurate when c is close to 1 (an epsilon error in c leads to a
    // sqrt(epsilon) error in theta), while the second formula is not very accurate when
    // c is close to 0.
    let thetas: Vec<f64> = c
        .iter()
        .zip(s.iter())
        .map(|(&ci, &si)| si.atan2(ci))
        .collect();

    CosSinDecompReturn {
        l0,
        l1,
        r0,
        r1,
        thetas,
    }
}

#[pyfunction]
pub fn cossin<'py>(py: Python<'py>, u: PyReadonlyArray2<Complex64>) -> PyResult<Bound<'py, PyAny>> {
    let array = u.as_array();
    let shape = array.shape();
    let mat: Mat<Complex64> = Mat::from_fn(shape[0], shape[1], |i, j| array[[i, j]]);
    let res = cos_sin_decomposition(mat.as_ref());

    let arr0 = faer_to_ndarray(res.l0.as_ref()).to_pyarray(py);
    let arr1 = faer_to_ndarray(res.l1.as_ref()).to_pyarray(py);
    let arr2 = faer_to_ndarray(res.r0.as_ref()).to_pyarray(py);
    let arr3 = faer_to_ndarray(res.r1.as_ref()).to_pyarray(py);

    Ok(((arr0, arr1), res.thetas, (arr2, arr3))
        .into_pyobject(py)?
        .into_any())
}

pub fn cos_sin_decomp(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(cossin))?;
    Ok(())
}
