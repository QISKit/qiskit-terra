// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

const PI: f64 = std::f64::consts::PI;

///     Return a triple theta, phi, lambda for the product.
///
///         u3(theta, phi, lambda)
///            = u3(theta1, phi1, lambda1).u3(theta2, phi2, lambda2)
///            = Rz(phi1).Ry(theta1).Rz(lambda1+phi2).Ry(theta2).Rz(lambda2)
///            = Rz(phi1).Rz(phi').Ry(theta').Rz(lambda').Rz(lambda2)
///            = u3(theta', phi1 + phi', lambda2 + lambda')
///
///         Return theta, phi, lambda.
#[pyfunction]
#[pyo3(text_signature = "(theta1, phi1, lambda1, theta2, phi2, lambda2, /)")]
pub fn compose_u3_rust(
    theta1: f64,
    phi1: f64,
    lambda1: f64,
    theta2: f64,
    phi2: f64,
    lambda2: f64,
) -> [f64; 3] {
    let q = [(theta1 / 2.0).cos(), 0., (theta1 / 2.0).sin(), 0.];
    let r = [
        ((lambda1 + phi2) / 2.0).cos(),
        0.,
        0.,
        ((lambda1 + phi2) / 2.0).sin(),
    ];
    let s = [(theta2 / 2.0).cos(), 0., (theta2 / 2.0).sin(), 0.];

    // Compute YZY decomp (q.r.s in variable names)
    let temp: [f64; 4] = [
        r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
        r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
        r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
        r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0],
    ];

    let out: [f64; 4] = [
        s[0] * temp[0] - s[1] * temp[1] - s[2] * temp[2] - s[3] * temp[3],
        s[0] * temp[1] + s[1] * temp[0] - s[2] * temp[3] + s[3] * temp[2],
        s[0] * temp[2] + s[1] * temp[3] + s[2] * temp[0] - s[3] * temp[1],
        s[0] * temp[3] - s[1] * temp[2] + s[2] * temp[1] + s[3] * temp[0],
    ];

    // out is now in YZY decomp, make into ZYZ
    let mat: [f64; 9] = [
        1. - 2. * out[2] * out[2] - 2. * out[3] * out[3],
        2. * out[1] * out[2] - 2. * out[3] * out[0],
        2. * out[1] * out[3] + 2. * out[2] * out[0],
        2. * out[1] * out[2] + 2. * out[3] * out[0],
        1. - 2. * out[1] * out[1] - 2. * out[3] * out[3],
        2. * out[2] * out[3] - 2. * out[1] * out[0],
        2. * out[1] * out[3] - 2. * out[2] * out[0],
        2. * out[2] * out[3] + 2. * out[1] * out[0],
        1. - 2. * out[1] * out[1] - 2. * out[2] * out[2],
    ];

    // Grab the euler angles
    let mut euler: [f64; 3] = if mat[8] < 1.0 {
        if mat[8] > -1.0 {
            [mat[5].atan2(mat[2]), (mat[8]).acos(), mat[7].atan2(-mat[6])]
        } else {
            [-(mat[3].atan2(mat[4])), PI, 0.]
        }
    } else {
        [mat[3].atan2(mat[4]), 0., 0.]
    };
    euler
        .iter_mut()
        .filter(|k| k.abs() < 1e-15)
        .for_each(|k| *k = 0.0);

    let out_angles: [f64; 3] = [euler[1], phi1 + euler[0], lambda2 + euler[2]];
    out_angles
}

pub fn optimize_1q_gates(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(compose_u3_rust))?;
    Ok(())
}
