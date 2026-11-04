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

use super::StandardGate;
use crate::operations::{Operation, Param};
use qiskit_quantum_info::versor_u2::{VersorSU2, VersorU2, VersorU2Error};

use nalgebra::{Quaternion, Unit};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, FRAC_PI_8};

const COS_FRAC_PI_8: f64 = 0.9238795325112867;
const SIN_FRAC_PI_8: f64 = 0.3826834323650898;

/// Conversion logic of `StandardGate::versor_u2`.
pub fn versor_u2(gate: StandardGate, params: &[Param]) -> Result<VersorU2, VersorU2Error> {
    debug_assert_eq!(params.len(), gate.num_params() as usize);
    match gate {
        StandardGate::GlobalPhase => {
            let &[Param::Float(phase)] = params else {
                return Err(VersorU2Error::Symbolic);
            };
            Ok(VersorSU2::identity().with_phase(phase))
        }
        StandardGate::H => {
            Ok(
                VersorSU2::from_quaternion_unchecked(0., -FRAC_1_SQRT_2, 0., -FRAC_1_SQRT_2)
                    .with_phase(FRAC_PI_2),
            )
        }
        StandardGate::I => Ok(VersorU2::identity()),
        StandardGate::X => {
            Ok(VersorSU2::from_quaternion_unchecked(0., 0., 0., -1.).with_phase(FRAC_PI_2))
        }
        StandardGate::Y => {
            Ok(VersorSU2::from_quaternion_unchecked(0., 0., -1., 0.).with_phase(FRAC_PI_2))
        }
        StandardGate::Z => {
            Ok(VersorSU2::from_quaternion_unchecked(0., -1., 0., 0.).with_phase(FRAC_PI_2))
        }
        StandardGate::Phase | StandardGate::U1 => {
            let &[Param::Float(angle)] = params else {
                return Err(VersorU2Error::Symbolic);
            };
            Ok(VersorSU2::from_rz(angle).with_phase(angle * 0.5))
        }
        StandardGate::R => {
            let &[Param::Float(angle), Param::Float(axis)] = params else {
                return Err(VersorU2Error::Symbolic);
            };
            let (sin_angle, cos_angle) = (angle * 0.5).sin_cos();
            let (sin_axis, cos_axis) = axis.sin_cos();
            Ok(VersorSU2::from_quaternion_unchecked(
                cos_angle,
                0.,
                -sin_axis * sin_angle,
                -cos_axis * sin_angle,
            )
            .into())
        }
        StandardGate::RX => {
            let &[Param::Float(angle)] = params else {
                return Err(VersorU2Error::Symbolic);
            };
            Ok(VersorSU2::from_rx(angle).into())
        }
        StandardGate::RY => {
            let &[Param::Float(angle)] = params else {
                return Err(VersorU2Error::Symbolic);
            };
            Ok(VersorSU2::from_ry(angle).into())
        }
        StandardGate::RZ => {
            let &[Param::Float(angle)] = params else {
                return Err(VersorU2Error::Symbolic);
            };
            Ok(VersorSU2::from_rz(angle).into())
        }
        StandardGate::S => {
            Ok(
                VersorSU2::from_quaternion_unchecked(FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0., 0.)
                    .with_phase(FRAC_PI_4),
            )
        }
        StandardGate::Sdg => {
            Ok(
                VersorSU2::from_quaternion_unchecked(FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0., 0.)
                    .with_phase(-FRAC_PI_4),
            )
        }
        StandardGate::SX => {
            Ok(
                VersorSU2::from_quaternion_unchecked(FRAC_1_SQRT_2, 0., 0., -FRAC_1_SQRT_2)
                    .with_phase(FRAC_PI_4),
            )
        }
        StandardGate::SXdg => {
            Ok(
                VersorSU2::from_quaternion_unchecked(FRAC_1_SQRT_2, 0., 0., FRAC_1_SQRT_2)
                    .with_phase(-FRAC_PI_4),
            )
        }
        StandardGate::T => {
            Ok(
                VersorSU2::from_quaternion_unchecked(COS_FRAC_PI_8, -SIN_FRAC_PI_8, 0., 0.)
                    .with_phase(FRAC_PI_8),
            )
        }
        StandardGate::Tdg => {
            Ok(
                VersorSU2::from_quaternion_unchecked(COS_FRAC_PI_8, SIN_FRAC_PI_8, 0., 0.)
                    .with_phase(-FRAC_PI_8),
            )
        }
        StandardGate::U | StandardGate::U3 => {
            let &[Param::Float(theta), Param::Float(phi), Param::Float(lambda)] = params else {
                return Err(VersorU2Error::Symbolic);
            };
            Ok(
                (VersorSU2::from_rz(phi) * VersorSU2::from_ry(theta) * VersorSU2::from_rz(lambda))
                    .with_phase((phi + lambda) * 0.5),
            )
        }
        StandardGate::U2 => {
            let &[Param::Float(phi), Param::Float(lambda)] = params else {
                return Err(VersorU2Error::Symbolic);
            };
            let (sin, cos) = (lambda * 0.5).sin_cos();
            // The RY(pi/2).RZ(lambda) part of the decomposition, without the phase term.
            let ry_rz = VersorSU2(Unit::new_unchecked(
                FRAC_1_SQRT_2 * Quaternion::new(cos, -sin, -cos, -sin),
            ));
            Ok((VersorSU2::from_rz(phi) * ry_rz).with_phase((phi + lambda) * 0.5))
        }
        _ => Err(VersorU2Error::MultiQubit),
    }
}
