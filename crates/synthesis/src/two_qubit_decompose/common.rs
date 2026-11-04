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

use std::f64::consts::FRAC_1_SQRT_2;

use nalgebra::{Matrix2, Matrix4};
use ndarray::{Array2, ArrayView2, array, aview2};
use num_complex::{Complex64, ComplexFloat};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::linalg::ndarray_to_faer;
use qiskit_circuit::gate_matrix::{H_GATE, S_GATE, SDG_GATE};
use qiskit_util::alias::GateArray2Q;
use qiskit_util::complex::{C_M_ONE, C_ONE, C_ZERO, IM, M_IM, c64};
pub(super) const DEFAULT_FIDELITY: f64 = 1.0 - 1.0e-9;

pub trait TraceToFidelity {
    /// Average gate fidelity is :math:`Fbar = (d + |Tr (Utarget \\cdot U^dag)|^2) / d(d+1)`
    /// M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)
    fn trace_to_fid(self) -> f64;
}

impl TraceToFidelity for Complex64 {
    fn trace_to_fid(self) -> f64 {
        (4.0 + self.abs().powi(2)) / 20.0
    }
}

#[pyfunction]
#[pyo3(name = "trace_to_fid")]
/// Average gate fidelity is :math:`Fbar = (d + |Tr (Utarget \\cdot U^dag)|^2) / d(d+1)`
/// M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)
pub fn py_trace_to_fid(trace: Complex64) -> PyResult<f64> {
    let fid = trace.trace_to_fid();
    Ok(fid)
}

static MAGIC: GateArray2Q = [
    [
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
    ],
    [
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
    ],
    [
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
        c64(-FRAC_1_SQRT_2, 0.),
        C_ZERO,
    ],
    [
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
        C_ZERO,
        c64(0., -FRAC_1_SQRT_2),
    ],
];

static MAGIC_DAGGER: GateArray2Q = [
    [
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
        C_ZERO,
        c64(FRAC_1_SQRT_2, 0.),
    ],
    [
        C_ZERO,
        c64(0., -FRAC_1_SQRT_2),
        c64(0., -FRAC_1_SQRT_2),
        C_ZERO,
    ],
    [
        C_ZERO,
        c64(FRAC_1_SQRT_2, 0.),
        c64(-FRAC_1_SQRT_2, 0.),
        C_ZERO,
    ],
    [
        c64(0., -FRAC_1_SQRT_2),
        C_ZERO,
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
    ],
];

pub(super) static IPZ: Matrix2<Complex64> = Matrix2::new(IM, C_ZERO, C_ZERO, M_IM);

pub(super) static IPY: Matrix2<Complex64> = Matrix2::new(C_ZERO, C_ONE, C_M_ONE, C_ZERO);

pub(super) static IPX: Matrix2<Complex64> = Matrix2::new(C_ZERO, IM, IM, C_ZERO);

/// A good approximation to the best value x to get the minimum
/// trace distance for :math:`U_d(x, x, x)` from :math:`U_d(a, b, c)`.
pub(crate) fn closest_partial_swap(a: f64, b: f64, c: f64) -> f64 {
    let m = (a + b + c) / 3.;
    let [am, bm, cm] = [a - m, b - m, c - m];
    let [ab, bc, ca] = [a - b, b - c, c - a];
    m + am * bm * cm * (6. + ab * ab + bc * bc + ca * ca) / 18.
}

pub(crate) fn rx_matrix(theta: f64) -> Matrix2<Complex64> {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let isin = c64(0., -half_theta.sin());
    Matrix2::new(cos, isin, isin, cos)
}

pub(crate) fn ry_matrix(theta: f64) -> Matrix2<Complex64> {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let sin = c64(half_theta.sin(), 0.);
    Matrix2::new(cos, -sin, sin, cos)
}

pub(crate) fn rz_matrix(theta: f64) -> Matrix2<Complex64> {
    let ilam2 = c64(0., 0.5 * theta);
    Matrix2::new((-ilam2).exp(), C_ZERO, C_ZERO, ilam2.exp())
}

/// Generates the array :math:`e^{(i a XX + i b YY + i c ZZ)}`
pub(crate) fn ud(a: f64, b: f64, c: f64) -> Array2<Complex64> {
    array![
        [
            (IM * c).exp() * (a - b).cos(),
            C_ZERO,
            C_ZERO,
            IM * (IM * c).exp() * (a - b).sin()
        ],
        [
            C_ZERO,
            (M_IM * c).exp() * (a + b).cos(),
            IM * (M_IM * c).exp() * (a + b).sin(),
            C_ZERO
        ],
        [
            C_ZERO,
            IM * (M_IM * c).exp() * (a + b).sin(),
            (M_IM * c).exp() * (a + b).cos(),
            C_ZERO
        ],
        [
            IM * (IM * c).exp() * (a - b).sin(),
            C_ZERO,
            C_ZERO,
            (IM * c).exp() * (a - b).cos()
        ]
    ]
}

#[pyfunction]
#[pyo3(name = "Ud")]
pub fn py_ud(py: Python, a: f64, b: f64, c: f64) -> Py<PyArray2<Complex64>> {
    let ud_mat = ud(a, b, c);
    ud_mat.into_pyarray(py).unbind()
}

/// Computes the local invariants for a two-qubit unitary.
///
/// Based on:
///
/// Y. Makhlin, Quant. Info. Proc. 1, 243-252 (2002).
///
/// Zhang et al., Phys Rev A. 67, 042313 (2003).
#[pyfunction]
pub fn two_qubit_local_invariants(unitary: PyReadonlyArray2<Complex64>) -> [f64; 3] {
    let mat = unitary.as_array();
    // Transform to bell basis
    let bell_basis_unitary = aview2(&MAGIC_DAGGER).dot(&mat.dot(&aview2(&MAGIC)));
    // Get determinate since +- one is allowed.
    let det_bell_basis = ndarray_to_faer(bell_basis_unitary.view()).determinant();
    let m = bell_basis_unitary.t().dot(&bell_basis_unitary);
    let mut m_tr2 = m.diag().sum();
    m_tr2 *= m_tr2;
    // Table II of Ref. 1 or Eq. 28 of Ref. 2.
    let g1 = m_tr2 / (16. * det_bell_basis);
    let g2 = (m_tr2 - m.dot(&m).diag().sum()) / (4. * det_bell_basis);

    // Here we split the real and imag pieces of G1 into two so as
    // to better equate to the Weyl chamber coordinates (c0,c1,c2)
    // and explore the parameter space.
    // Also do a FP trick -0.0 + 0.0 = 0.0
    [g1.re + 0., g1.im + 0., g2.re + 0.]
}

#[pyfunction]
pub fn local_equivalence(weyl: PyReadonlyArray1<f64>) -> PyResult<[f64; 3]> {
    let weyl = weyl.as_array();
    let weyl_2_cos_squared_product: f64 = weyl.iter().map(|x| (x * 2.).cos().powi(2)).product();
    let weyl_2_sin_squared_product: f64 = weyl.iter().map(|x| (x * 2.).sin().powi(2)).product();
    let g0_equiv = weyl_2_cos_squared_product - weyl_2_sin_squared_product;
    let g1_equiv = weyl.iter().map(|x| (x * 4.).sin()).product::<f64>() / 4.;
    let g2_equiv = 4. * weyl_2_cos_squared_product
        - 4. * weyl_2_sin_squared_product
        - weyl.iter().map(|x| (4. * x).cos()).product::<f64>();
    Ok([g0_equiv + 0., g1_equiv + 0., g2_equiv + 0.])
}

/// Copy the input array view into a Matrix2 output
///
/// This function assumes the input is a 2x2 matrix. If a matrix of a
/// different shape is passed in it will pick the first 4 elements in logical
/// order (row major) and if it doesn't have 4 elements it will panic. This
/// should only really be used for copying a 2x2 ndarray view into a Matrix2.
#[inline]
pub(super) fn ndarray_to_matrix2<T: Copy>(view: ArrayView2<T>) -> Matrix2<T> {
    Matrix2::new(view[[0, 0]], view[(0, 1)], view[(1, 0)], view[(1, 1)])
}

/// Copy the input array view into a Matrix4 output
///
/// This function assumes the input is a 4x4 matrix. If a matrix of a
/// different shape is passed in it will pick the first 16 elements in logical
/// order (row major) and if it doesn't have 16 elements it will panic. This
/// should only really be used for copying a 4x4 ndarray view into a Matrix4.
#[inline]
pub(super) fn ndarray_to_matrix4(view: ArrayView2<Complex64>) -> Matrix4<Complex64> {
    Matrix4::from_row_iterator(view.iter().copied())
}

pub static HGATE: Matrix2<Complex64> =
    Matrix2::new(H_GATE[0][0], H_GATE[0][1], H_GATE[1][0], H_GATE[1][1]);
pub static SGATE: Matrix2<Complex64> =
    Matrix2::new(S_GATE[0][0], S_GATE[0][1], S_GATE[1][0], S_GATE[1][1]);
pub static SDGGATE: Matrix2<Complex64> = Matrix2::new(
    SDG_GATE[0][0],
    SDG_GATE[0][1],
    SDG_GATE[1][0],
    SDG_GATE[1][1],
);
