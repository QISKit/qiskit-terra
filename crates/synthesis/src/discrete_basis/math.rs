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

use nalgebra::{Matrix2, Matrix3, Matrix3x1};
use ndarray::ArrayView2;
use num_complex::{Complex64, ComplexFloat};
use qiskit_circuit::operations::{Param, StandardGate};
use std::{
    f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, FRAC_PI_8},
    ops::Div,
};

use super::basic_approximations::DiscreteBasisError;

/// Solve equation (10) in https://arxiv.org/pdf/quant-ph/0505030 by using the
/// substitution sin(u/2) = sin^2(phi/4).
pub(crate) fn solve_decomposition_angle(matrix: &Matrix3<f64>) -> f64 {
    let trace = matrix.trace().min(3.0); // avoid roundoff errors
    let angle = ((trace - 1.) / 2.).acos();

    2. * (angle / 4.).sin().sqrt().abs().asin()
}

/// Add coeff * V to the out matrix, where V is the skew-symmetric representation of the
/// cross product with the vector v.
#[inline]
fn add_cross_prod(coeff: f64, v: &Matrix3x1<f64>, out: &mut Matrix3<f64>) {
    out[(0, 1)] -= coeff * v[2];
    out[(0, 2)] += coeff * v[1];
    out[(1, 0)] += coeff * v[2];
    out[(1, 2)] -= coeff * v[0];
    out[(2, 0)] -= coeff * v[1];
    out[(2, 1)] += coeff * v[0];
}

/// Compute the group commutator UVU^{-1}V^{-1}. This code leverages the fact that
/// the matrices are in SO(3) and U^{-1} = U^T.
fn group_commutator(left: &Matrix3<f64>, right: &Matrix3<f64>) -> Matrix3<f64> {
    left * right * left.transpose() * right.transpose()
}

/// Compute the rotation axis of a SO(3) matrix.
fn rotation_axis_from_so3(matrix: &Matrix3<f64>, do_checks: bool) -> Matrix3x1<f64> {
    let trace = matrix.trace();

    // Try to obtain the matrix via Rodrigues formula. If the matrix is diagonal (or the Rodrigues
    // approach fails, likely because the matrix was diagonal up to numerical error), determine
    // the axis from the diagonal.
    let eps = 1e-15;
    if trace >= 3. - eps {
        // matrix is the identity, ie. no rotation
        return Matrix3x1::identity();
    }

    if trace >= eps - 1. {
        // there's a skew symmetric part which we can use to get the rotation axis
        let theta = ((trace - 1.) / 2.).acos();

        if theta.sin() > eps {
            let coeff = 1. / 2. / theta.sin();
            let axis = Matrix3x1::new(
                coeff * (matrix[(2, 1)] - matrix[(1, 2)]),
                coeff * (matrix[(0, 2)] - matrix[(2, 0)]),
                coeff * (matrix[(1, 0)] - matrix[(0, 1)]),
            );

            // this might fail due to numerical error, in that case go to diagonal case
            if !axis.iter().any(|el| el.is_nan()) {
                return axis.normalize();
            }
            if do_checks {
                panic!("Encountered NaN in rotation axis.");
            }
        }
    }

    // this is a 180 degree rotation
    let mut axis = Matrix3x1::new(
        ((1. + matrix[(0, 0)]) / 2.).sqrt(),
        ((1. + matrix[(1, 1)]) / 2.).sqrt(),
        ((1. + matrix[(2, 2)]) / 2.).sqrt(),
    );

    // fix the signs by setting the first non-zero element to +1 and determine the rest from there
    let index = axis
        .iter()
        .enumerate()
        .find(|&(_, &el)| el.abs() > eps)
        .expect("At least one element must be nonzero.")
        .0;
    match index {
        0 => {
            if matrix[(0, 1)] < 0. {
                axis[1] *= -1.;
            };
            if matrix[(0, 2)] < 0. {
                axis[2] *= -1.;
            }
        }
        1 if matrix[(1, 2)] < 0. => {
            axis[2] *= -1.;
        }
        _ => (),
    };
    axis
}

/// Compute the SO(3) matrix that rotates from ``from`` to ``to``.
fn rotation_matrix(from: &Matrix3x1<f64>, to: &Matrix3x1<f64>, do_checks: bool) -> Matrix3<f64> {
    let from = from.normalize();
    let to = to.normalize();
    let dot = from.dot(&to);

    let mut cross = Matrix3::zeros();
    add_cross_prod(1., &from.cross(&to), &mut cross);

    let out = Matrix3::identity() + cross + cross * cross / (1. + dot);
    if do_checks {
        assert_so3("rotation matrix", &out);
    }
    out
}

/// Compute the SO(3) matrix implementing rotation about ``axis`` for an ``angle``.
fn so3_from_angle_axis(angle: f64, axis: &Matrix3x1<f64>) -> Matrix3<f64> {
    let mut out = Matrix3::<f64>::identity() * angle.cos();
    add_cross_prod(angle.sin(), axis, &mut out);

    let outer = axis * axis.transpose();
    out + (1. - angle.cos()) * outer
}

/// Decompose the SO(3) input matrix M into a balanced group commutator, that is
///
///     M = V W V^T W^T
///
/// for two SO(3) matrices V and W. See section 4.1 of https://arxiv.org/abs/quant-ph/0505030.
pub fn group_commutator_decomposition(
    matrix_so3: &Matrix3<f64>,
    do_checks: bool,
) -> (Matrix3<f64>, Matrix3<f64>) {
    if do_checks {
        assert_so3("GC input", matrix_so3);
    }

    let angle = solve_decomposition_angle(matrix_so3);

    let e1 = Matrix3x1::new(1., 0., 0.);
    let e2 = Matrix3x1::new(0., 1., 0.);

    let vx = so3_from_angle_axis(angle, &e1);
    let wy = so3_from_angle_axis(angle, &e2);

    let group_comm = group_commutator(&vx, &wy);
    let group_comm_axis = rotation_axis_from_so3(&group_comm, do_checks);
    let matrix_axis = rotation_axis_from_so3(matrix_so3, do_checks);

    let sim_matrix = rotation_matrix(&group_comm_axis, &matrix_axis, do_checks);
    let sim_matrix_t = sim_matrix.transpose();

    let v = sim_matrix * vx * sim_matrix_t;
    let w = sim_matrix * wy * sim_matrix_t;

    if do_checks {
        assert_so3("group commutator", &group_comm);
        assert_so3("vx", &vx);
        assert_so3("wy", &wy);
        assert_so3("sim_matrix", &sim_matrix);
        assert_so3("v", &v);
        assert_so3("w", &w);
    }

    (v, w)
}

pub(super) fn assert_so3(name: &str, matrix: &Matrix3<f64>) {
    if matrix.iter().any(|el| el.is_nan()) {
        panic!("{name} has NaN value.");
    }
    if (1. - matrix.determinant()) > 1e-5 {
        panic!(
            "{} is not SO(3): Determinant is {}, not 1.",
            name,
            matrix.determinant()
        );
    }
    let diff = matrix * matrix.transpose() - Matrix3::<f64>::identity();
    if diff.iter().any(|el| el.abs() > 1e-5) {
        panic!("{name} is not SO(3): Matrix is not orthogonal.")
    }
}

pub fn su2_to_so3(view: &Matrix2<Complex64>) -> Matrix3<f64> {
    let a = view[(0, 0)].re;
    let b = view[(0, 0)].im;
    let c = -view[(0, 1)].re;
    let d = -view[(0, 1)].im;

    Matrix3::new(
        a.powi(2) - b.powi(2) - c.powi(2) + d.powi(2),
        2.0 * (a * b + c * d),
        2.0 * (b * d - a * c),
        2.0 * (c * d - a * b),
        a.powi(2) - b.powi(2) + c.powi(2) - d.powi(2),
        2.0 * (a * d + b * c),
        2.0 * (a * c + b * d),
        2.0 * (b * c - a * d),
        a.powi(2) + b.powi(2) - c.powi(2) - d.powi(2),
    )
}

/// Convert a unitary input matrix into an SO(3) matrix + phase.
///
/// Note that this is not a bijective mapping, as the SO(3) representation loses
/// the sign information.
pub fn u2_to_so3(matrix_u2: &Matrix2<Complex64>) -> (Matrix3<f64>, f64) {
    let determinant = matrix_u2.determinant();
    let matrix_su2 = matrix_u2.div(determinant.sqrt());
    let matrix_so3 = su2_to_so3(&matrix_su2);
    let z = 1. / determinant.sqrt();
    let phase = z.im().atan2(z.re());

    (matrix_so3, phase)
}

#[inline]
pub fn array2_to_matrix2<T: Copy>(view: &ArrayView2<T>) -> Matrix2<T> {
    Matrix2::new(view[[0, 0]], view[(0, 1)], view[(1, 0)], view[(1, 1)])
}

/// Get the U(2) representation of a standard gate.
pub fn standard_gates_to_u2(
    gate: &StandardGate,
    params: &[Param],
) -> Result<Matrix2<Complex64>, DiscreteBasisError> {
    let matrix_c64 = gate.matrix(params).expect("Failed to get matrix.");
    Ok(array2_to_matrix2(&matrix_c64.view()))
}

/// Get the SO(3) representation of a standard gate.
///
/// Attempts to directly construct the matrix using [f64] accuracy, otherwise falls back
/// to matrix construction and conversion.
pub fn standard_gates_to_so3(
    gate: &StandardGate,
    params: &[Param],
) -> Result<(Matrix3<f64>, f64), DiscreteBasisError> {
    match gate {
        StandardGate::T => {
            let so3 = Matrix3::new(
                FRAC_1_SQRT_2,
                -FRAC_1_SQRT_2,
                0.,
                FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                0.,
                0.,
                0.,
                1.,
            );
            let phase = -FRAC_PI_8;
            Ok((so3, phase))
        }
        StandardGate::Tdg => {
            let so3 = Matrix3::new(
                FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                0.,
                -FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
                0.,
                0.,
                0.,
                1.,
            );
            let phase = FRAC_PI_8;
            Ok((so3, phase))
        }
        StandardGate::S => {
            let so3 = Matrix3::new(0., -1., 0., 1., 0., 0., 0., 0., 1.);
            let phase = -FRAC_PI_4;
            Ok((so3, phase))
        }
        StandardGate::Sdg => {
            let so3 = Matrix3::new(0., 1., 0., -1., 0., 0., 0., 0., 1.);
            let phase = FRAC_PI_4;
            Ok((so3, phase))
        }
        StandardGate::H => {
            let so3 = Matrix3::new(0., 0., -1., 0., -1., 0., -1., 0., 0.);
            let phase = FRAC_PI_2;
            Ok((so3, phase))
        }
        StandardGate::RX | StandardGate::RY | StandardGate::RZ => {
            let angle = match params[0] {
                Param::Float(angle) => angle,
                _ => return Err(DiscreteBasisError::ParameterizedGate),
            };
            let cos = angle.cos();
            let sin = angle.sin();
            let so3 = match gate {
                StandardGate::RX => Matrix3::new(1., 0., 0., 0., cos, sin, 0., -sin, cos),
                StandardGate::RY => Matrix3::new(cos, 0., -sin, 0., 1., 0., sin, 0., cos),
                StandardGate::RZ => Matrix3::new(cos, -sin, 0., sin, cos, 0., 0., 0., 1.),
                _ => unreachable!(),
            };
            Ok((so3, 0.))
        }
        _ => {
            let array_u2 = gate
                .matrix(params)
                .expect("Failed to get matrix representation.");
            let matrix_u2 = array2_to_matrix2(&array_u2.view());
            let (so3, phase) = u2_to_so3(&matrix_u2);

            Ok((so3, phase))
        }
    }
}
