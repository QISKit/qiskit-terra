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

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::dag_circuit::NodeIndex;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::packed_instruction::PackedOperation;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8, PI};

use crate::passes::common::{MINIMUM_TOL, is_angle_close_to_multiple_of_pi_k};
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;

static ROTATION_GATE_NAMES: [&str; 14] = [
    "rx", "ry", "rz", "p", "cu1", "u1", "rzz", "rxx", "rzx", "ryy", "cp", "crx", "cry", "crz",
];

type SubstituteSequencePi4<'a> = [(&'a [(StandardGate, &'a [u32])], f64); 16];
type SubstituteSequencePi2<'a> = [(&'a [(StandardGate, &'a [u32])], f64); 8];

/// Table for RZ(k * pi / 4) substitutions, with 0 <= k < 16
static RZ_SUBSTITUTIONS: [(&[StandardGate], f64); 16] = [
    (&[], 0.0),
    (&[StandardGate::T], -FRAC_PI_8),
    (&[StandardGate::S], -FRAC_PI_4),
    (&[StandardGate::S, StandardGate::T], -3.0 * FRAC_PI_8),
    (&[StandardGate::Z], -FRAC_PI_2),
    (&[StandardGate::Z, StandardGate::T], -5.0 * FRAC_PI_8),
    (&[StandardGate::Sdg], -3.0 * FRAC_PI_4),
    (&[StandardGate::Tdg], -7.0 * FRAC_PI_8),
    (&[], -PI),
    (&[StandardGate::T], 7.0 * FRAC_PI_8),
    (&[StandardGate::S], 3.0 * FRAC_PI_4),
    (&[StandardGate::S, StandardGate::T], 5.0 * FRAC_PI_8),
    (&[StandardGate::Z], FRAC_PI_2),
    (&[StandardGate::Z, StandardGate::T], 3.0 * FRAC_PI_8),
    (&[StandardGate::Sdg], FRAC_PI_4),
    (&[StandardGate::Tdg], FRAC_PI_8),
];

/// Table for RX(k * pi / 4) substitutions, with 0 <= k < 16
static RX_SUBSTITUTIONS: [(&[StandardGate], f64); 16] = [
    (&[], 0.0),
    (
        &[StandardGate::H, StandardGate::T, StandardGate::H],
        -FRAC_PI_8,
    ),
    (&[StandardGate::SX], -FRAC_PI_4),
    (
        &[
            StandardGate::SX,
            StandardGate::H,
            StandardGate::T,
            StandardGate::H,
        ],
        -3.0 * FRAC_PI_8,
    ),
    (&[StandardGate::X], -FRAC_PI_2),
    (
        &[
            StandardGate::X,
            StandardGate::H,
            StandardGate::T,
            StandardGate::H,
        ],
        -5.0 * FRAC_PI_8,
    ),
    (&[StandardGate::SXdg], -3.0 * FRAC_PI_4),
    (
        &[StandardGate::H, StandardGate::Tdg, StandardGate::H],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[StandardGate::H, StandardGate::T, StandardGate::H],
        7.0 * FRAC_PI_8,
    ),
    (&[StandardGate::SX], 3.0 * FRAC_PI_4),
    (
        &[
            StandardGate::SX,
            StandardGate::H,
            StandardGate::T,
            StandardGate::H,
        ],
        5.0 * FRAC_PI_8,
    ),
    (&[StandardGate::X], FRAC_PI_2),
    (
        &[
            StandardGate::X,
            StandardGate::H,
            StandardGate::T,
            StandardGate::H,
        ],
        3.0 * FRAC_PI_8,
    ),
    (&[StandardGate::SXdg], FRAC_PI_4),
    (
        &[StandardGate::H, StandardGate::Tdg, StandardGate::H],
        FRAC_PI_8,
    ),
];

/// Table for RY(k * pi / 4) substitutions, with 0 <= k < 16
static RY_SUBSTITUTIONS: [(&[StandardGate], f64); 16] = [
    (&[], 0.0),
    (
        &[StandardGate::SX, StandardGate::T, StandardGate::SXdg],
        -FRAC_PI_8,
    ),
    (&[StandardGate::Z, StandardGate::H], 0.0),
    (
        &[
            StandardGate::SX,
            StandardGate::T,
            StandardGate::S,
            StandardGate::SXdg,
        ],
        -3.0 * FRAC_PI_8,
    ),
    (&[StandardGate::Y], -FRAC_PI_2),
    (
        &[
            StandardGate::Y,
            StandardGate::SX,
            StandardGate::T,
            StandardGate::SXdg,
        ],
        -5.0 * FRAC_PI_8,
    ),
    (&[StandardGate::H, StandardGate::Z], -PI),
    (
        &[StandardGate::SX, StandardGate::Tdg, StandardGate::SXdg],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[StandardGate::SX, StandardGate::T, StandardGate::SXdg],
        7.0 * FRAC_PI_8,
    ),
    (&[StandardGate::Z, StandardGate::H], -PI),
    (
        &[
            StandardGate::SX,
            StandardGate::T,
            StandardGate::S,
            StandardGate::SXdg,
        ],
        5.0 * FRAC_PI_8,
    ),
    (&[StandardGate::Y], FRAC_PI_2),
    (
        &[
            StandardGate::Y,
            StandardGate::SX,
            StandardGate::T,
            StandardGate::SXdg,
        ],
        3.0 * FRAC_PI_8,
    ),
    (&[StandardGate::H, StandardGate::Z], 0.0),
    (
        &[StandardGate::SX, StandardGate::Tdg, StandardGate::SXdg],
        FRAC_PI_8,
    ),
];

/// Table for P(k * pi / 4) substitutions, with 0 <= k < 16
static P_SUBSTITUTIONS: [(&[StandardGate], f64); 16] = [
    (&[], 0.0),
    (&[StandardGate::T], 0.0),
    (&[StandardGate::S], 0.0),
    (&[StandardGate::S, StandardGate::T], 0.0),
    (&[StandardGate::Z], 0.0),
    (&[StandardGate::Z, StandardGate::T], 0.0),
    (&[StandardGate::Sdg], 0.0),
    (&[StandardGate::Tdg], 0.0),
    (&[], 0.0),
    (&[StandardGate::T], 0.0),
    (&[StandardGate::S], 0.0),
    (&[StandardGate::S, StandardGate::T], 0.0),
    (&[StandardGate::Z], 0.0),
    (&[StandardGate::Z, StandardGate::T], 0.0),
    (&[StandardGate::Sdg], 0.0),
    (&[StandardGate::Tdg], 0.0),
];

/// Table for RZZ(k * pi / 4) substitutions, with 0 <= k < 16
static RZZ_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        -FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
        ],
        -FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        -3.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::Z, &[0]), (StandardGate::Z, &[1])],
        -FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        -5.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
        ],
        -3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        7.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
        ],
        3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        5.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::Z, &[0]), (StandardGate::Z, &[1])],
        FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        3.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
        ],
        FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        FRAC_PI_8,
    ),
];

/// Table for RXX(k * pi / 4) substitutions, with 0 <= k < 16
static RXX_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        -FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::SX, &[1]),
            (StandardGate::H, &[0]),
        ],
        -FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        -3.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::X, &[0]), (StandardGate::X, &[1])],
        -FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        -5.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::H, &[0]),
        ],
        -3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        7.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::SX, &[1]),
            (StandardGate::H, &[0]),
        ],
        3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        5.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::X, &[0]), (StandardGate::X, &[1])],
        FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        3.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::H, &[0]),
        ],
        FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[0]),
            (StandardGate::H, &[1]),
        ],
        FRAC_PI_8,
    ),
];

/// Table for RZX(k * pi / 4) substitutions, with 0 <= k < 16
static RZX_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        -FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        -3.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::Z, &[0]), (StandardGate::X, &[1])],
        -FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        -5.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::SXdg, &[1]),
        ],
        -3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        7.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::SX, &[1]),
        ],
        3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        5.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::Z, &[0]), (StandardGate::X, &[1])],
        FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        3.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::SXdg, &[1]),
        ],
        FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        FRAC_PI_8,
    ),
];

/// Table for RYY(k * pi / 4) substitutions, with 0 <= k < 16
static RYY_SUBSTITUTIONS: SubstituteSequencePi4 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -3.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::Y, &[0]), (StandardGate::Y, &[1])],
        -FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -5.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        7.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::S, &[0]),
            (StandardGate::S, &[1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        3.0 * FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        5.0 * FRAC_PI_8,
    ),
    (
        &[(StandardGate::Y, &[0]), (StandardGate::Y, &[1])],
        FRAC_PI_2,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Z, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        3.0 * FRAC_PI_8,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CZ, &[0, 1]),
            (StandardGate::Sdg, &[0]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        FRAC_PI_4,
    ),
    (
        &[
            (StandardGate::SXdg, &[0]),
            (StandardGate::SXdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SX, &[0]),
            (StandardGate::SX, &[1]),
        ],
        FRAC_PI_8,
    ),
];

/// Table for CPhase(k * pi / 2) substitutions, with 0 <= k < 8
static CP_SUBSTITUTIONS: SubstituteSequencePi2 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::T, &[0]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (&[(StandardGate::CZ, &[0, 1])], 0.0),
    (
        &[
            (StandardGate::Tdg, &[0]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (&[], 0.0),
    (
        &[
            (StandardGate::T, &[0]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (&[(StandardGate::CZ, &[0, 1])], 0.0),
    (
        &[
            (StandardGate::Tdg, &[0]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
];

/// Table for CRZ(k * pi / 2) substitutions, with 0 <= k < 8
static CRZ_SUBSTITUTIONS: SubstituteSequencePi2 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (
        &[(StandardGate::CZ, &[0, 1]), (StandardGate::Sdg, &[0])],
        0.0,
    ),
    (
        &[
            (StandardGate::T, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (&[(StandardGate::Z, &[0])], 0.0),
    (
        &[
            (StandardGate::Tdg, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
    (&[(StandardGate::CZ, &[0, 1]), (StandardGate::S, &[0])], 0.0),
    (
        &[
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
        ],
        0.0,
    ),
];

/// Table for CRX(k * pi / 2) substitutions, with 0 <= k < 8
static CRX_SUBSTITUTIONS: SubstituteSequencePi2 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        0.0,
    ),
    (
        &[(StandardGate::CX, &[0, 1]), (StandardGate::Sdg, &[0])],
        0.0,
    ),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        0.0,
    ),
    (&[(StandardGate::Z, &[0])], 0.0),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        0.0,
    ),
    (&[(StandardGate::CX, &[0, 1]), (StandardGate::S, &[0])], 0.0),
    (
        &[
            (StandardGate::H, &[1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::H, &[1]),
        ],
        0.0,
    ),
];

/// Table for CRY(k * pi / 2) substitutions, with 0 <= k < 8
static CRY_SUBSTITUTIONS: SubstituteSequencePi2 = [
    (&[], 0.0),
    (
        &[
            (StandardGate::SX, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SXdg, &[1]),
        ],
        0.0,
    ),
    (
        &[(StandardGate::CY, &[0, 1]), (StandardGate::Sdg, &[0])],
        0.0,
    ),
    (
        &[
            (StandardGate::SX, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SXdg, &[1]),
        ],
        0.0,
    ),
    (&[(StandardGate::Z, &[0])], 0.0),
    (
        &[
            (StandardGate::SX, &[1]),
            (StandardGate::Sdg, &[1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::S, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SXdg, &[1]),
        ],
        0.0,
    ),
    (&[(StandardGate::CY, &[0, 1]), (StandardGate::S, &[0])], 0.0),
    (
        &[
            (StandardGate::SX, &[1]),
            (StandardGate::Tdg, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::T, &[1]),
            (StandardGate::CX, &[0, 1]),
            (StandardGate::SXdg, &[1]),
        ],
        0.0,
    ),
];

/// The following two functions get a rotation gate and outputs an equivalent vector of
/// standard gates in {Clifford, T, Tdg} and a global phase.
/// For single-qubit rotation gates (RX, RY, RZ, Phase) and two-qubit rotation gates
/// (RXX, RYY, RZZ, RZX), when the angle is a multiple of pi/4, the decomposition requires
/// a single T or Tdg gate as well as some Clifford gates.
/// For two-qubit controlled rotation gates (CPhase, CRX, CRY, CRZ), when the angle
/// is a multiple of pi/2, the decomposition requires three T or Tdg gates for CPhase and
/// two otherwise, as well as some Clifford gates.
/// Note that even multiples of pi/4 (resectively pi/2 for controlled rotations),
/// or equivalently, integer multiples of pi/2 (respectively pi for controlled rotations),
/// can be written using only Clifford gates.
///
/// Function for 1-qubit gates
fn replace_1q_rotation_by_discrete(
    gate: StandardGate,
    multiple: usize,
) -> (&'static [StandardGate], f64) {
    match gate {
        StandardGate::RX => RX_SUBSTITUTIONS[multiple],
        StandardGate::RY => RY_SUBSTITUTIONS[multiple],
        StandardGate::RZ => RZ_SUBSTITUTIONS[multiple],
        StandardGate::Phase => P_SUBSTITUTIONS[multiple],
        StandardGate::U1 => P_SUBSTITUTIONS[multiple],
        _ => unreachable!("This is only called for 1-qubit rotation gates."),
    }
}

/// Function for 2-qubit gates
fn replace_2q_rotation_by_discrete(
    gate: StandardGate,
    multiple: usize,
) -> (&'static [(StandardGate, &'static [u32])], f64) {
    match gate {
        StandardGate::RZZ => RZZ_SUBSTITUTIONS[multiple],
        StandardGate::RXX => RXX_SUBSTITUTIONS[multiple],
        StandardGate::RZX => RZX_SUBSTITUTIONS[multiple],
        StandardGate::RYY => RYY_SUBSTITUTIONS[multiple],
        StandardGate::CPhase => CP_SUBSTITUTIONS[multiple],
        StandardGate::CU1 => CP_SUBSTITUTIONS[multiple],
        StandardGate::CRZ => CRZ_SUBSTITUTIONS[multiple],
        StandardGate::CRX => CRX_SUBSTITUTIONS[multiple],
        StandardGate::CRY => CRY_SUBSTITUTIONS[multiple],
        _ => unreachable!("This is only called for 2-qubit rotation gates."),
    }
}

/// Matches 4 to the single-qubit and two-qubit rotation gates, and 2 to the controlled-rotation gates
fn rotation_to_pi_div(gate: StandardGate) -> usize {
    match gate {
        StandardGate::RX => 4,
        StandardGate::RY => 4,
        StandardGate::RZ => 4,
        StandardGate::Phase => 4,
        StandardGate::U1 => 4,
        StandardGate::RZZ => 4,
        StandardGate::RXX => 4,
        StandardGate::RZX => 4,
        StandardGate::RYY => 4,
        StandardGate::CPhase => 2,
        StandardGate::CU1 => 2,
        StandardGate::CRZ => 2,
        StandardGate::CRX => 2,
        StandardGate::CRY => 2,
        _ => unreachable!("This is only called for rotation gates."),
    }
}

#[pyfunction]
#[pyo3(name = "substitute_pi4_rotations")]
pub fn py_run_substitute_pi4_rotations(
    dag: &mut DAGCircuit,
    approximation_degree: f64,
) -> PyResult<()> {
    // Skip the pass if there are no rotation gates.
    if dag
        .get_op_counts()
        .keys()
        .all(|k| !ROTATION_GATE_NAMES.contains(&k.as_str()))
    {
        return Ok(());
    }

    // Iterate over nodes in the DAG and collect nodes that are rotation gates
    // with an angle that is sufficiently close to a multiple of pi/4
    // (respectively pi/2 for controlled rotations)
    let tol = MINIMUM_TOL.max(1.0 - approximation_degree);
    let mut global_phase_update: f64 = 0.;

    // We first collect all nodes that need changing and store their pi/4 multiple. Then
    // we iterate over the DAG to apply these changes.
    let nodes_to_replace: Vec<(NodeIndex, StandardGate, usize)> = dag
        .op_nodes(false)
        .filter_map(|(node_index, inst)| {
            let OperationRef::StandardGate(gate) = inst.op.view() else {
                return None;
            };

            if gate.num_params() != 1 {
                return None;
            };

            let Param::Float(angle) = inst.params_view()[0] else {
                return None;
            };

            let k = rotation_to_pi_div(gate);
            let multiple = is_angle_close_to_multiple_of_pi_k(gate, k, angle, tol)?;
            Some((node_index, gate, multiple))
        })
        .collect();

    for (node_index, gate, multiple) in nodes_to_replace {
        match gate.num_qubits() {
            1 => {
                let (sequence, phase_update) = replace_1q_rotation_by_discrete(gate, multiple);
                let num_gates = sequence.len();
                if num_gates == 0 {
                    // in the special case that we have a 0-length sequence, remove the gate
                    dag.remove_1q_sequence(&[node_index]);
                } else {
                    // add all gates except the last one, and then substitute the existing op
                    // for the very last gate
                    for gate in sequence[..num_gates - 1].iter() {
                        dag.insert_1q_on_incoming_qubit((*gate, &[]), node_index);
                    }
                    let last_op = PackedOperation::from_standard_gate(
                        *sequence.last().expect("sequence has at least 1 element"),
                    );
                    dag.substitute_op(node_index, last_op, None, None)?;
                }
                global_phase_update += phase_update;
            }
            2 => {
                let (sequence, phase_update) = replace_2q_rotation_by_discrete(gate, multiple);
                let mut local_dag =
                    DAGCircuit::with_capacity(2, 0, None, Some(sequence.len()), None, None);
                local_dag.add_qubit_unchecked(ShareableQubit::new_anonymous())?;
                local_dag.add_qubit_unchecked(ShareableQubit::new_anonymous())?;

                for (gate, qubits) in sequence {
                    let qargs = ::bytemuck::cast_slice(qubits);
                    let interned = local_dag.add_qargs(qargs);
                    let packed = PackedInstruction::from_standard_gate(*gate, None, interned);
                    local_dag.push_back(packed)?;
                }
                dag.substitute_node_with_dag(node_index, &local_dag, None, None, None, None)?;
                global_phase_update += phase_update;
            }
            _ => {
                // There's no standard rotation gates with more than 2 qubits.
                continue;
            }
        };
    }
    dag.add_global_phase(&Param::Float(global_phase_update))?;

    Ok(())
}

pub fn substitute_pi4_rotations_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_substitute_pi4_rotations))?;
    Ok(())
}
