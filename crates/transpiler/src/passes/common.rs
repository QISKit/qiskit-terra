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

use num_complex::Complex64;
use num_complex::ComplexFloat;
use std::f64::consts::PI;

use crate::gate_metrics::rotation_trace_and_dim;
use qiskit_circuit::operations::StandardGate;

// Default value for tolerance level. It is used to check the fidely tolerance.
pub const MINIMUM_TOL: f64 = 1e-12;

/// Fidelity-based computation to check whether an operation `G` is equivalent
/// to identity up to a global phase.
///
/// # Arguments
///
/// * `tr_over_dim`: `|Tr(G)| / dim(G)`.
/// * `dim`: `dim(G)`.
/// * `tol`: tolerance.
///
/// # Returns
///
/// * `Some(update to the global phase)` if the operation can be removed.
/// * `None` if the operation cannot be removed.
pub fn average_gate_fidelity_below_tol(tr_over_dim: Complex64, dim: f64, tol: f64) -> Option<f64> {
    let f_pro = tr_over_dim.abs().powi(2);
    let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
    if (1. - gate_fidelity).abs() < tol {
        Some(tr_over_dim.arg())
    } else {
        None
    }
}

/// For a given angle, if it is a multiple of PI/k, calculate the multiple mod (4*k),
/// Otherwise, return `None`.
/// E.g, if the angle is a multiple m of PI/4 then it returns m, where 0 <= m < 16,
/// and if the angle is a multiple m of PI/2 then it returns m, where 0 <= m < 8.
/// Panics if the StandardGate is not a rotation gate.
pub fn is_angle_close_to_multiple_of_pi_k(
    gate: StandardGate,
    k: usize,
    angle: f64,
    tol: f64,
) -> Option<usize> {
    let closest_ratio = angle * (k as f64) / PI;
    let closest_integer = closest_ratio.round();
    let closest_angle = closest_integer * PI / (k as f64);
    let theta = angle - closest_angle;

    // Trace and dimension calculation of rotation matrices
    let (tr_over_dim, dim) = rotation_trace_and_dim(gate, theta)
        .expect("Since only supported rotation gates are given, the result is not None");

    // fidelity-based tolerance
    let f_pro = tr_over_dim.abs().powi(2);
    let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
    let rem = 4 * k as i64;
    if (1. - gate_fidelity).abs() < tol {
        Some((closest_integer as i64).rem_euclid(rem) as usize)
    } else {
        None
    }
}
