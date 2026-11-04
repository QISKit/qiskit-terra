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

mod convert;
pub mod standard_generators;

use crate::circuit_data::{CircuitData, PyCircuitData};
use crate::operations::{Operation, Param, add_param, clone_param, multiply_param, radd_param};
use crate::{Qubit, gate_matrix, impl_intopyobject_for_copy_pyclass, imports};
use qiskit_quantum_info::versor_u2::{VersorU2, VersorU2Error};

use ndarray::{Array2, aview2};
use num_complex::Complex64;
use smallvec::{SmallVec, smallvec};
use std::f64::consts::PI;

use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyList, PyTuple};

const FLOAT_ZERO: Param = Param::Float(0.0);

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[repr(u8)]
#[pyclass(module = "qiskit._accelerate.circuit", eq, eq_int, from_py_object)]
pub enum StandardGate {
    GlobalPhase = 0,
    H = 1,
    I = 2,
    X = 3,
    Y = 4,
    Z = 5,
    Phase = 6,
    R = 7,
    RX = 8,
    RY = 9,
    RZ = 10,
    S = 11,
    Sdg = 12,
    SX = 13,
    SXdg = 14,
    T = 15,
    Tdg = 16,
    U = 17,
    U1 = 18,
    U2 = 19,
    U3 = 20,
    CH = 21,
    CX = 22,
    CY = 23,
    CZ = 24,
    DCX = 25,
    ECR = 26,
    Swap = 27,
    ISwap = 28,
    CPhase = 29,
    CRX = 30,
    CRY = 31,
    CRZ = 32,
    CS = 33,
    CSdg = 34,
    CSX = 35,
    CU = 36,
    CU1 = 37,
    CU3 = 38,
    RXX = 39,
    RYY = 40,
    RZZ = 41,
    RZX = 42,
    XXMinusYY = 43,
    XXPlusYY = 44,
    CCX = 45,
    CCZ = 46,
    CSwap = 47,
    RCCX = 48,
    C3X = 49,
    C3SX = 50,
    RC3X = 51,
    // Remember to update StandardGate::is_valid_bit_pattern below
    // if you add or remove this enum's variants!
}
impl_intopyobject_for_copy_pyclass!(StandardGate);

unsafe impl ::bytemuck::CheckedBitPattern for StandardGate {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < (STANDARD_GATE_SIZE as u8)
    }
}
unsafe impl ::bytemuck::NoUninit for StandardGate {}

static STANDARD_GATE_NUM_QUBITS: [u32; STANDARD_GATE_SIZE] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 0-9
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 10-19
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, // 20-29
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // 30-39
    2, 2, 2, 2, 2, 3, 3, 3, 3, 4, // 40-49
    4, 4, // 50-51
];

static STANDARD_GATE_NUM_PARAMS: [u32; STANDARD_GATE_SIZE] = [
    1, 0, 0, 0, 0, 0, 1, 2, 1, 1, // 0-9
    1, 0, 0, 0, 0, 0, 0, 3, 1, 2, // 10-19
    3, 0, 0, 0, 0, 0, 0, 0, 0, 1, // 20-29
    1, 1, 1, 0, 0, 0, 4, 1, 3, 1, // 30-39
    1, 1, 1, 2, 2, 0, 0, 0, 0, 0, // 40-49
    0, 0, // 50-51
];

static STANDARD_GATE_NUM_CTRL_QUBITS: [u32; STANDARD_GATE_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0-9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 10-19
    0, 1, 1, 1, 1, 0, 0, 0, 0, 1, // 20-29
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, // 30-39
    0, 0, 0, 0, 0, 2, 2, 1, 0, 3, // 40-49
    3, 0, // 50-51
];

static STANDARD_GATE_NAME: [&str; STANDARD_GATE_SIZE] = [
    "global_phase", // 0
    "h",            // 1
    "id",           // 2
    "x",            // 3
    "y",            // 4
    "z",            // 5
    "p",            // 6
    "r",            // 7
    "rx",           // 8
    "ry",           // 9
    "rz",           // 10
    "s",            // 11
    "sdg",          // 12
    "sx",           // 13
    "sxdg",         // 14
    "t",            // 15
    "tdg",          // 16
    "u",            // 17
    "u1",           // 18
    "u2",           // 19
    "u3",           // 20
    "ch",           // 21
    "cx",           // 22
    "cy",           // 23
    "cz",           // 24
    "dcx",          // 25
    "ecr",          // 26
    "swap",         // 27
    "iswap",        // 28
    "cp",           // 29
    "crx",          // 30
    "cry",          // 31
    "crz",          // 32
    "cs",           // 33
    "csdg",         // 34
    "csx",          // 35
    "cu",           // 36
    "cu1",          // 37
    "cu3",          // 38
    "rxx",          // 39
    "ryy",          // 40
    "rzz",          // 41
    "rzx",          // 42
    "xx_minus_yy",  // 43
    "xx_plus_yy",   // 44
    "ccx",          // 45
    "ccz",          // 46
    "cswap",        // 47
    "rccx",         // 48
    "mcx",          // 49 ("c3x")
    "c3sx",         // 50
    "rcccx",        // 51 ("rc3x")
];

/// Get a slice of all standard gate names.
pub fn get_standard_gate_names() -> &'static [&'static str] {
    &STANDARD_GATE_NAME
}

impl StandardGate {
    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<SmallVec<[Param; 3]>>,
        label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let gate_class = imports::get_std_gate_class(py, *self)?;
        let args = match params {
            None => PyTuple::empty(py),
            Some(params) => {
                PyTuple::new(py, params.into_iter().map(|x| x.into_pyobject(py).unwrap()))?
            }
        };
        if let Some(label) = label {
            let kwargs = [("label", label.into_pyobject(py)?)].into_py_dict(py)?;
            gate_class.call(py, args, Some(&kwargs))
        } else {
            gate_class.call(py, args, None)
        }
    }

    pub fn num_ctrl_qubits(&self) -> u32 {
        STANDARD_GATE_NUM_CTRL_QUBITS[*self as usize]
    }

    pub fn inverse(&self, params: &[Param]) -> Option<(StandardGate, SmallVec<[Param; 3]>)> {
        match self {
            Self::GlobalPhase => Some((
                Self::GlobalPhase,
                smallvec![multiply_param(&params[0], -1.0)],
            )),
            Self::H => Some((Self::H, smallvec![])),
            Self::I => Some((Self::I, smallvec![])),
            Self::X => Some((Self::X, smallvec![])),
            Self::Y => Some((Self::Y, smallvec![])),
            Self::Z => Some((Self::Z, smallvec![])),
            Self::Phase => Some((Self::Phase, smallvec![multiply_param(&params[0], -1.0)])),
            Self::R => Some((
                Self::R,
                smallvec![multiply_param(&params[0], -1.0), params[1].clone()],
            )),
            Self::RX => Some((Self::RX, smallvec![multiply_param(&params[0], -1.0)])),
            Self::RY => Some((Self::RY, smallvec![multiply_param(&params[0], -1.0)])),
            Self::RZ => Some((Self::RZ, smallvec![multiply_param(&params[0], -1.0)])),
            Self::S => Some((Self::Sdg, smallvec![])),
            Self::Sdg => Some((Self::S, smallvec![])),
            Self::SX => Some((Self::SXdg, smallvec![])),
            Self::SXdg => Some((Self::SX, smallvec![])),
            Self::T => Some((Self::Tdg, smallvec![])),
            Self::Tdg => Some((Self::T, smallvec![])),
            Self::U => Some((
                Self::U,
                smallvec![
                    multiply_param(&params[0], -1.0),
                    multiply_param(&params[2], -1.0),
                    multiply_param(&params[1], -1.0),
                ],
            )),
            Self::U1 => Some((Self::U1, smallvec![multiply_param(&params[0], -1.0)])),
            Self::U2 => Some((
                Self::U2,
                smallvec![
                    add_param(&multiply_param(&params[1], -1.0), -PI),
                    add_param(&multiply_param(&params[0], -1.0), PI),
                ],
            )),
            Self::U3 => Some((
                Self::U3,
                smallvec![
                    multiply_param(&params[0], -1.0),
                    multiply_param(&params[2], -1.0),
                    multiply_param(&params[1], -1.0),
                ],
            )),
            Self::CH => Some((Self::CH, smallvec![])),
            Self::CX => Some((Self::CX, smallvec![])),
            Self::CY => Some((Self::CY, smallvec![])),
            Self::CZ => Some((Self::CZ, smallvec![])),
            Self::DCX => None, // the inverse in not a StandardGate
            Self::ECR => Some((Self::ECR, smallvec![])),
            Self::Swap => Some((Self::Swap, smallvec![])),
            Self::ISwap => None, // the inverse in not a StandardGate
            Self::CPhase => Some((Self::CPhase, smallvec![multiply_param(&params[0], -1.0)])),
            Self::CRX => Some((Self::CRX, smallvec![multiply_param(&params[0], -1.0)])),
            Self::CRY => Some((Self::CRY, smallvec![multiply_param(&params[0], -1.0)])),
            Self::CRZ => Some((Self::CRZ, smallvec![multiply_param(&params[0], -1.0)])),
            Self::CS => Some((Self::CSdg, smallvec![])),
            Self::CSdg => Some((Self::CS, smallvec![])),
            Self::CSX => None, // the inverse in not a StandardGate
            Self::CU => Some((
                Self::CU,
                smallvec![
                    multiply_param(&params[0], -1.0),
                    multiply_param(&params[2], -1.0),
                    multiply_param(&params[1], -1.0),
                    multiply_param(&params[3], -1.0),
                ],
            )),
            Self::CU1 => Some((Self::CU1, smallvec![multiply_param(&params[0], -1.0)])),
            Self::CU3 => Some((
                Self::CU3,
                smallvec![
                    multiply_param(&params[0], -1.0),
                    multiply_param(&params[2], -1.0),
                    multiply_param(&params[1], -1.0),
                ],
            )),
            Self::RXX => Some((Self::RXX, smallvec![multiply_param(&params[0], -1.0)])),
            Self::RYY => Some((Self::RYY, smallvec![multiply_param(&params[0], -1.0)])),
            Self::RZZ => Some((Self::RZZ, smallvec![multiply_param(&params[0], -1.0)])),
            Self::RZX => Some((Self::RZX, smallvec![multiply_param(&params[0], -1.0)])),
            Self::XXMinusYY => Some((
                Self::XXMinusYY,
                smallvec![multiply_param(&params[0], -1.0), params[1].clone()],
            )),
            Self::XXPlusYY => Some((
                Self::XXPlusYY,
                smallvec![multiply_param(&params[0], -1.0), params[1].clone()],
            )),
            Self::CCX => Some((Self::CCX, smallvec![])),
            Self::CCZ => Some((Self::CCZ, smallvec![])),
            Self::CSwap => Some((Self::CSwap, smallvec![])),
            Self::RCCX => Some((Self::RCCX, smallvec![])),
            Self::C3X => Some((Self::C3X, smallvec![])),
            Self::C3SX => None, // the inverse in not a StandardGate
            Self::RC3X => None, // the inverse in not a StandardGate
        }
    }

    pub fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>> {
        match self {
            Self::GlobalPhase => match params {
                [Param::Float(theta)] => {
                    Some(aview2(&gate_matrix::global_phase_gate(*theta)).to_owned())
                }
                _ => None,
            },
            Self::H => match params {
                [] => Some(aview2(&gate_matrix::H_GATE).to_owned()),
                _ => None,
            },
            Self::I => match params {
                [] => Some(aview2(&gate_matrix::ONE_QUBIT_IDENTITY).to_owned()),
                _ => None,
            },
            Self::X => match params {
                [] => Some(aview2(&gate_matrix::X_GATE).to_owned()),
                _ => None,
            },
            Self::Y => match params {
                [] => Some(aview2(&gate_matrix::Y_GATE).to_owned()),
                _ => None,
            },
            Self::Z => match params {
                [] => Some(aview2(&gate_matrix::Z_GATE).to_owned()),
                _ => None,
            },
            Self::Phase => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::phase_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::R => match params {
                [Param::Float(theta), Param::Float(phi)] => {
                    Some(aview2(&gate_matrix::r_gate(*theta, *phi)).to_owned())
                }
                _ => None,
            },
            Self::RX => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::rx_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::RY => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::ry_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::RZ => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::rz_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::S => match params {
                [] => Some(aview2(&gate_matrix::S_GATE).to_owned()),
                _ => None,
            },
            Self::Sdg => match params {
                [] => Some(aview2(&gate_matrix::SDG_GATE).to_owned()),
                _ => None,
            },
            Self::SX => match params {
                [] => Some(aview2(&gate_matrix::SX_GATE).to_owned()),
                _ => None,
            },
            Self::SXdg => match params {
                [] => Some(aview2(&gate_matrix::SXDG_GATE).to_owned()),
                _ => None,
            },
            Self::T => match params {
                [] => Some(aview2(&gate_matrix::T_GATE).to_owned()),
                _ => None,
            },
            Self::Tdg => match params {
                [] => Some(aview2(&gate_matrix::TDG_GATE).to_owned()),
                _ => None,
            },
            Self::U => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::U1 => match params[0] {
                Param::Float(val) => Some(aview2(&gate_matrix::u1_gate(val)).to_owned()),
                _ => None,
            },
            Self::U2 => match params {
                [Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u2_gate(*phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::U3 => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u3_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::CH => match params {
                [] => Some(aview2(&gate_matrix::CH_GATE).to_owned()),
                _ => None,
            },
            Self::CX => match params {
                [] => Some(aview2(&gate_matrix::CX_GATE).to_owned()),
                _ => None,
            },
            Self::CY => match params {
                [] => Some(aview2(&gate_matrix::CY_GATE).to_owned()),
                _ => None,
            },
            Self::CZ => match params {
                [] => Some(aview2(&gate_matrix::CZ_GATE).to_owned()),
                _ => None,
            },
            Self::DCX => match params {
                [] => Some(aview2(&gate_matrix::DCX_GATE).to_owned()),
                _ => None,
            },
            Self::ECR => match params {
                [] => Some(aview2(&gate_matrix::ECR_GATE).to_owned()),
                _ => None,
            },
            Self::Swap => match params {
                [] => Some(aview2(&gate_matrix::SWAP_GATE).to_owned()),
                _ => None,
            },
            Self::ISwap => match params {
                [] => Some(aview2(&gate_matrix::ISWAP_GATE).to_owned()),
                _ => None,
            },
            Self::CPhase => match params {
                [Param::Float(lam)] => Some(aview2(&gate_matrix::cp_gate(*lam)).to_owned()),
                _ => None,
            },
            Self::CRX => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::crx_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CRY => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::cry_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CRZ => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::crz_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CS => match params {
                [] => Some(aview2(&gate_matrix::CS_GATE).to_owned()),
                _ => None,
            },
            Self::CSdg => match params {
                [] => Some(aview2(&gate_matrix::CSDG_GATE).to_owned()),
                _ => None,
            },
            Self::CSX => match params {
                [] => Some(aview2(&gate_matrix::CSX_GATE).to_owned()),
                _ => None,
            },
            Self::CU => match params {
                [
                    Param::Float(theta),
                    Param::Float(phi),
                    Param::Float(lam),
                    Param::Float(gamma),
                ] => Some(aview2(&gate_matrix::cu_gate(*theta, *phi, *lam, *gamma)).to_owned()),
                _ => None,
            },
            Self::CU1 => match params[0] {
                Param::Float(lam) => Some(aview2(&gate_matrix::cu1_gate(lam)).to_owned()),
                _ => None,
            },
            Self::CU3 => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::cu3_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::RXX => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rxx_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RYY => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::ryy_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RZZ => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rzz_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RZX => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rzx_gate(theta)).to_owned()),
                _ => None,
            },
            Self::XXMinusYY => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_minus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            Self::XXPlusYY => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_plus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            Self::CCX => match params {
                [] => Some(aview2(&gate_matrix::CCX_GATE).to_owned()),
                _ => None,
            },
            Self::CCZ => match params {
                [] => Some(aview2(&gate_matrix::CCZ_GATE).to_owned()),
                _ => None,
            },
            Self::CSwap => match params {
                [] => Some(aview2(&gate_matrix::CSWAP_GATE).to_owned()),
                _ => None,
            },
            Self::RCCX => match params {
                [] => Some(aview2(&gate_matrix::RCCX_GATE).to_owned()),
                _ => None,
            },
            Self::C3X => match params {
                [] => Some(aview2(&gate_matrix::C3X_GATE).to_owned()),
                _ => None,
            },
            Self::C3SX => match params {
                [] => Some(aview2(&gate_matrix::C3SX_GATE).to_owned()),
                _ => None,
            },
            Self::RC3X => match params {
                [] => Some(aview2(&gate_matrix::RC3X_GATE).to_owned()),
                _ => None,
            },
        }
    }

    pub fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        match self {
            Self::GlobalPhase => Some(
                CircuitData::from_standard_gates(0, [], params[0].clone())
                    .expect("Unexpected Qiskit python bug"),
            ),
            Self::H => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        smallvec![Param::Float(PI / 2.), FLOAT_ZERO, Param::Float(PI)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::I => None,
            Self::X => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        smallvec![Param::Float(PI), FLOAT_ZERO, Param::Float(PI)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::Y => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        smallvec![
                            Param::Float(PI),
                            Param::Float(PI / 2.),
                            Param::Float(PI / 2.),
                        ],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),

            Self::Z => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        smallvec![Param::Float(PI)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::Phase => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        smallvec![FLOAT_ZERO, FLOAT_ZERO, params[0].clone()],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::R => {
                let theta_expr = clone_param(&params[0]);
                let phi_expr1 = add_param(&params[1], -PI / 2.);
                let phi_expr2 = multiply_param(&phi_expr1, -1.0);
                let defparams = smallvec![theta_expr, phi_expr1, phi_expr2];
                Some(
                    CircuitData::from_standard_gates(
                        1,
                        [(Self::U, defparams, smallvec![Qubit(0)])],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RX => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        1,
                        [(
                            Self::R,
                            smallvec![theta.clone(), FLOAT_ZERO],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RY => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        1,
                        [(
                            Self::R,
                            smallvec![theta.clone(), Param::Float(PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RZ => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        1,
                        [(Self::Phase, smallvec![theta.clone()], smallvec![Qubit(0)])],
                        multiply_param(theta, -0.5),
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::S => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        smallvec![Param::Float(PI / 2.)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::Sdg => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        smallvec![Param::Float(-PI / 2.)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::SX => Some(
                CircuitData::from_standard_gates(
                    1,
                    [
                        (Self::Sdg, smallvec![], smallvec![Qubit(0)]),
                        (Self::H, smallvec![], smallvec![Qubit(0)]),
                        (Self::Sdg, smallvec![], smallvec![Qubit(0)]),
                    ],
                    Param::Float(PI / 4.),
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::SXdg => Some(
                CircuitData::from_standard_gates(
                    1,
                    [
                        (Self::S, smallvec![], smallvec![Qubit(0)]),
                        (Self::H, smallvec![], smallvec![Qubit(0)]),
                        (Self::S, smallvec![], smallvec![Qubit(0)]),
                    ],
                    Param::Float(-PI / 4.),
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::T => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        smallvec![Param::Float(PI / 4.)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::Tdg => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        smallvec![Param::Float(-PI / 4.)],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::U => None,
            Self::U1 => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::Phase,
                        params.iter().cloned().collect(),
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::U2 => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        smallvec![Param::Float(PI / 2.), params[0].clone(), params[1].clone()],
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::U3 => Some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        Self::U,
                        params.iter().cloned().collect(),
                        smallvec![Qubit(0)],
                    )],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::CH => {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::S, smallvec![], q1.clone()),
                            (Self::H, smallvec![], q1.clone()),
                            (Self::T, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::Tdg, smallvec![], q1.clone()),
                            (Self::H, smallvec![], q1.clone()),
                            (Self::Sdg, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }

            Self::CX => None,
            Self::CY => {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::Sdg, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::S, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CZ => {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::H, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::H, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::DCX => Some(
                CircuitData::from_standard_gates(
                    2,
                    [
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::ECR => Some(
                CircuitData::from_standard_gates(
                    2,
                    [
                        (Self::S, smallvec![], smallvec![Qubit(0)]),
                        (Self::SX, smallvec![], smallvec![Qubit(1)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::X, smallvec![], smallvec![Qubit(0)]),
                    ],
                    Param::Float(-PI / 4.),
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::Swap => Some(
                CircuitData::from_standard_gates(
                    2,
                    [
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::ISwap => Some(
                CircuitData::from_standard_gates(
                    2,
                    [
                        (Self::S, smallvec![], smallvec![Qubit(0)]),
                        (Self::S, smallvec![], smallvec![Qubit(1)]),
                        (Self::H, smallvec![], smallvec![Qubit(0)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                        (Self::H, smallvec![], smallvec![Qubit(1)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::CPhase => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::Phase, smallvec![multiply_param(&params[0], 0.5)], q0),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (
                                Self::Phase,
                                smallvec![multiply_param(&params[0], -0.5)],
                                q1.clone(),
                            ),
                            (Self::CX, smallvec![], q0_1),
                            (Self::Phase, smallvec![multiply_param(&params[0], 0.5)], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CRX => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::S, smallvec![], smallvec![Qubit(1)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, -0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, 0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::Sdg, smallvec![], smallvec![Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }
            Self::CRY => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, 0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RY,
                                smallvec![multiply_param(theta, -0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }
            Self::CRZ => {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (
                                Self::RZ,
                                smallvec![multiply_param(theta, 0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RZ,
                                smallvec![multiply_param(theta, -0.5)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }
            Self::CS => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::T, smallvec![], q0),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (Self::Tdg, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::T, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CSdg => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::Tdg, smallvec![], q0),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (Self::T, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::Tdg, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CSX => {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::H, smallvec![], q1.clone()),
                            (Self::CS, smallvec![], q0_1),
                            (Self::H, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CU => {
                let param_second_p = radd_param(
                    multiply_param(&params[2], 0.5),
                    multiply_param(&params[1], 0.5),
                );
                let param_third_p = radd_param(
                    multiply_param(&params[2], 0.5),
                    multiply_param(&params[1], -0.5),
                );
                let param_first_u = radd_param(
                    multiply_param(&params[1], -0.5),
                    multiply_param(&params[2], -0.5),
                );
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (
                                Self::Phase,
                                smallvec![params[3].clone()],
                                smallvec![Qubit(0)],
                            ),
                            (Self::Phase, smallvec![param_second_p], smallvec![Qubit(0)]),
                            (Self::Phase, smallvec![param_third_p], smallvec![Qubit(1)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U,
                                smallvec![
                                    multiply_param(&params[0], -0.5),
                                    FLOAT_ZERO,
                                    param_first_u
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U,
                                smallvec![
                                    multiply_param(&params[0], 0.5),
                                    params[1].clone(),
                                    FLOAT_ZERO
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CU1 => Some(
                CircuitData::from_standard_gates(
                    2,
                    [
                        (
                            Self::Phase,
                            smallvec![multiply_param(&params[0], 0.5)],
                            smallvec![Qubit(0)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (
                            Self::Phase,
                            smallvec![multiply_param(&params[0], -0.5)],
                            smallvec![Qubit(1)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (
                            Self::Phase,
                            smallvec![multiply_param(&params[0], 0.5)],
                            smallvec![Qubit(1)],
                        ),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::CU3 => {
                let param_first_u1 = radd_param(
                    multiply_param(&params[2], 0.5),
                    multiply_param(&params[1], 0.5),
                );
                let param_second_u1 = radd_param(
                    multiply_param(&params[2], 0.5),
                    multiply_param(&params[1], -0.5),
                );
                let param_first_u3 = radd_param(
                    multiply_param(&params[1], -0.5),
                    multiply_param(&params[2], -0.5),
                );
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::Phase, smallvec![param_first_u1], smallvec![Qubit(0)]),
                            (Self::Phase, smallvec![param_second_u1], smallvec![Qubit(1)]),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U,
                                smallvec![
                                    multiply_param(&params[0], -0.5),
                                    FLOAT_ZERO,
                                    param_first_u3
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U,
                                smallvec![
                                    multiply_param(&params[0], 0.5),
                                    params[1].clone(),
                                    FLOAT_ZERO
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RXX => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::H, smallvec![], q0.clone()),
                            (Self::H, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_q1.clone()),
                            (Self::RZ, smallvec![theta.clone()], q1.clone()),
                            (Self::CX, smallvec![], q0_q1),
                            (Self::H, smallvec![], q1),
                            (Self::H, smallvec![], q0),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RYY => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::SXdg, smallvec![], q0.clone()),
                            (Self::SXdg, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_q1.clone()),
                            (Self::RZ, smallvec![theta.clone()], q1.clone()),
                            (Self::CX, smallvec![], q0_q1),
                            (Self::SX, smallvec![], q0),
                            (Self::SX, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RZZ => {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::CX, smallvec![], q0_q1.clone()),
                            (Self::RZ, smallvec![theta.clone()], q1),
                            (Self::CX, smallvec![], q0_q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::RZX => {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::H, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_q1.clone()),
                            (Self::RZ, smallvec![theta.clone()], q1.clone()),
                            (Self::CX, smallvec![], q0_q1),
                            (Self::H, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::XXMinusYY => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::RZ, smallvec![multiply_param(beta, -1.0)], q1.clone()),
                            (Self::Sdg, smallvec![], q0.clone()),
                            (Self::SX, smallvec![], q0.clone()),
                            (Self::S, smallvec![], q0.clone()),
                            (Self::S, smallvec![], q1.clone()),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (Self::RY, smallvec![multiply_param(theta, 0.5)], q0.clone()),
                            (Self::RY, smallvec![multiply_param(theta, -0.5)], q1.clone()),
                            (Self::CX, smallvec![], q0_1),
                            (Self::Sdg, smallvec![], q1.clone()),
                            (Self::Sdg, smallvec![], q0.clone()),
                            (Self::SXdg, smallvec![], q0.clone()),
                            (Self::S, smallvec![], q0),
                            (Self::RZ, smallvec![beta.clone()], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::XXPlusYY => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q1_0 = smallvec![Qubit(1), Qubit(0)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        2,
                        [
                            (Self::RZ, smallvec![beta.clone()], q0.clone()),
                            (Self::Sdg, smallvec![], q1.clone()),
                            (Self::SX, smallvec![], q1.clone()),
                            (Self::S, smallvec![], q1.clone()),
                            (Self::S, smallvec![], q0.clone()),
                            (Self::CX, smallvec![], q1_0.clone()),
                            (Self::RY, smallvec![multiply_param(theta, -0.5)], q1.clone()),
                            (Self::RY, smallvec![multiply_param(theta, -0.5)], q0.clone()),
                            (Self::CX, smallvec![], q1_0),
                            (Self::Sdg, smallvec![], q0.clone()),
                            (Self::Sdg, smallvec![], q1.clone()),
                            (Self::SXdg, smallvec![], q1.clone()),
                            (Self::S, smallvec![], q1),
                            (Self::RZ, smallvec![multiply_param(beta, -1.0)], q0),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::CCX => {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q2 = smallvec![Qubit(2)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        3,
                        [
                            (Self::H, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q1_2.clone()),
                            (Self::Tdg, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q0_2.clone()),
                            (Self::T, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q1_2),
                            (Self::Tdg, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q0_2),
                            (Self::T, smallvec![], q1.clone()),
                            (Self::T, smallvec![], q2.clone()),
                            (Self::H, smallvec![], q2),
                            (Self::CX, smallvec![], q0_1.clone()),
                            (Self::T, smallvec![], q0),
                            (Self::Tdg, smallvec![], q1),
                            (Self::CX, smallvec![], q0_1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }

            Self::CCZ => Some(
                CircuitData::from_standard_gates(
                    3,
                    [
                        (Self::H, smallvec![], smallvec![Qubit(2)]),
                        (
                            Self::CCX,
                            smallvec![],
                            smallvec![Qubit(0), Qubit(1), Qubit(2)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(2)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::CSwap => Some(
                CircuitData::from_standard_gates(
                    3,
                    [
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(1)]),
                        (
                            Self::CCX,
                            smallvec![],
                            smallvec![Qubit(0), Qubit(1), Qubit(2)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(1)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),

            Self::RCCX => {
                let q2 = smallvec![Qubit(2)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        3,
                        [
                            (Self::H, smallvec![], q2.clone()),
                            (Self::T, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q1_2.clone()),
                            (Self::Tdg, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q0_2),
                            (Self::T, smallvec![], q2.clone()),
                            (Self::CX, smallvec![], q1_2),
                            (Self::Tdg, smallvec![], q2.clone()),
                            (Self::H, smallvec![], q2),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }
            Self::C3X => Some(
                CircuitData::from_standard_gates(
                    4,
                    [
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(0)],
                        ),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(1)],
                        ),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(2)],
                        ),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(1)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(2)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(2)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(2)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (
                            Self::Phase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(3)],
                        ),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),

            Self::C3SX => Some(
                CircuitData::from_standard_gates(
                    4,
                    [
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(0), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(1), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(1), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(2), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(2), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(-PI / 8.)],
                            smallvec![Qubit(2), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (
                            Self::CPhase,
                            smallvec![Param::Float(PI / 8.)],
                            smallvec![Qubit(2), Qubit(3)],
                        ),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
            Self::RC3X => Some(
                CircuitData::from_standard_gates(
                    4,
                    [
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::T, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (Self::Tdg, smallvec![], smallvec![Qubit(3)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                        (Self::T, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                        (Self::Tdg, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                        (Self::T, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                        (Self::Tdg, smallvec![], smallvec![Qubit(3)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                        (Self::T, smallvec![], smallvec![Qubit(3)]),
                        (Self::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                        (Self::Tdg, smallvec![], smallvec![Qubit(3)]),
                        (Self::H, smallvec![], smallvec![Qubit(3)]),
                    ],
                    FLOAT_ZERO,
                )
                .expect("Unexpected Qiskit python bug"),
            ),
        }
    }

    pub fn matrix_as_static_1q(&self, params: &[Param]) -> Option<[[Complex64; 2]; 2]> {
        match self {
            Self::GlobalPhase => None,
            Self::H => match params {
                [] => Some(gate_matrix::H_GATE),
                _ => None,
            },
            Self::I => match params {
                [] => Some(gate_matrix::ONE_QUBIT_IDENTITY),
                _ => None,
            },
            Self::X => match params {
                [] => Some(gate_matrix::X_GATE),
                _ => None,
            },
            Self::Y => match params {
                [] => Some(gate_matrix::Y_GATE),
                _ => None,
            },
            Self::Z => match params {
                [] => Some(gate_matrix::Z_GATE),
                _ => None,
            },
            Self::Phase => match params {
                [Param::Float(theta)] => Some(gate_matrix::phase_gate(*theta)),
                _ => None,
            },
            Self::R => match params {
                [Param::Float(theta), Param::Float(phi)] => Some(gate_matrix::r_gate(*theta, *phi)),
                _ => None,
            },
            Self::RX => match params {
                [Param::Float(theta)] => Some(gate_matrix::rx_gate(*theta)),
                _ => None,
            },
            Self::RY => match params {
                [Param::Float(theta)] => Some(gate_matrix::ry_gate(*theta)),
                _ => None,
            },
            Self::RZ => match params {
                [Param::Float(theta)] => Some(gate_matrix::rz_gate(*theta)),
                _ => None,
            },
            Self::S => match params {
                [] => Some(gate_matrix::S_GATE),
                _ => None,
            },
            Self::Sdg => match params {
                [] => Some(gate_matrix::SDG_GATE),
                _ => None,
            },
            Self::SX => match params {
                [] => Some(gate_matrix::SX_GATE),
                _ => None,
            },
            Self::SXdg => match params {
                [] => Some(gate_matrix::SXDG_GATE),
                _ => None,
            },
            Self::T => match params {
                [] => Some(gate_matrix::T_GATE),
                _ => None,
            },
            Self::Tdg => match params {
                [] => Some(gate_matrix::TDG_GATE),
                _ => None,
            },
            Self::U => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(gate_matrix::u_gate(*theta, *phi, *lam))
                }
                _ => None,
            },
            Self::U1 => match params[0] {
                Param::Float(val) => Some(gate_matrix::u1_gate(val)),
                _ => None,
            },
            Self::U2 => match params {
                [Param::Float(phi), Param::Float(lam)] => Some(gate_matrix::u2_gate(*phi, *lam)),
                _ => None,
            },
            Self::U3 => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(gate_matrix::u3_gate(*theta, *phi, *lam))
                }
                _ => None,
            },
            Self::CH => None,
            Self::CX => None,
            Self::CY => None,
            Self::CZ => None,
            Self::DCX => None,
            Self::ECR => None,
            Self::Swap => None,
            Self::ISwap => None,
            Self::CPhase => None,
            Self::CRX => None,
            Self::CRY => None,
            Self::CRZ => None,
            Self::CS => None,
            Self::CSdg => None,
            Self::CSX => None,
            Self::CU => None,
            Self::CU1 => None,
            Self::CU3 => None,
            Self::RXX => None,
            Self::RYY => None,
            Self::RZZ => None,
            Self::RZX => None,
            Self::XXMinusYY => None,
            Self::XXPlusYY => None,
            Self::CCX => None,
            Self::CCZ => None,
            Self::CSwap => None,
            Self::RCCX => None,
            Self::C3X => None,
            Self::C3SX => None,
            Self::RC3X => None,
        }
    }

    pub fn matrix_as_static_2q(&self, params: &[Param]) -> Option<[[Complex64; 4]; 4]> {
        match self {
            Self::GlobalPhase => None,
            Self::H => None,
            Self::I => None,
            Self::X => None,
            Self::Y => None,
            Self::Z => None,
            Self::Phase => None,
            Self::R => None,
            Self::RX => None,
            Self::RY => None,
            Self::RZ => None,
            Self::S => None,
            Self::Sdg => None,
            Self::SX => None,
            Self::SXdg => None,
            Self::T => None,
            Self::Tdg => None,
            Self::U => None,
            Self::U1 => None,
            Self::U2 => None,
            Self::U3 => None,
            Self::CH => match params {
                [] => Some(gate_matrix::CH_GATE),
                _ => None,
            },
            Self::CX => match params {
                [] => Some(gate_matrix::CX_GATE),
                _ => None,
            },
            Self::CY => match params {
                [] => Some(gate_matrix::CY_GATE),
                _ => None,
            },
            Self::CZ => match params {
                [] => Some(gate_matrix::CZ_GATE),
                _ => None,
            },
            Self::DCX => match params {
                [] => Some(gate_matrix::DCX_GATE),
                _ => None,
            },
            Self::ECR => match params {
                [] => Some(gate_matrix::ECR_GATE),
                _ => None,
            },
            Self::Swap => match params {
                [] => Some(gate_matrix::SWAP_GATE),
                _ => None,
            },
            Self::ISwap => match params {
                [] => Some(gate_matrix::ISWAP_GATE),
                _ => None,
            },
            Self::CPhase => match params {
                [Param::Float(lam)] => Some(gate_matrix::cp_gate(*lam)),
                _ => None,
            },
            Self::CRX => match params {
                [Param::Float(theta)] => Some(gate_matrix::crx_gate(*theta)),
                _ => None,
            },
            Self::CRY => match params {
                [Param::Float(theta)] => Some(gate_matrix::cry_gate(*theta)),
                _ => None,
            },
            Self::CRZ => match params {
                [Param::Float(theta)] => Some(gate_matrix::crz_gate(*theta)),
                _ => None,
            },
            Self::CS => match params {
                [] => Some(gate_matrix::CS_GATE),
                _ => None,
            },
            Self::CSdg => match params {
                [] => Some(gate_matrix::CSDG_GATE),
                _ => None,
            },
            Self::CSX => match params {
                [] => Some(gate_matrix::CSX_GATE),
                _ => None,
            },
            Self::CU => match params {
                [
                    Param::Float(theta),
                    Param::Float(phi),
                    Param::Float(lam),
                    Param::Float(gamma),
                ] => Some(gate_matrix::cu_gate(*theta, *phi, *lam, *gamma)),
                _ => None,
            },
            Self::CU1 => match params[0] {
                Param::Float(lam) => Some(gate_matrix::cu1_gate(lam)),
                _ => None,
            },
            Self::CU3 => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(gate_matrix::cu3_gate(*theta, *phi, *lam))
                }
                _ => None,
            },
            Self::RXX => match params[0] {
                Param::Float(theta) => Some(gate_matrix::rxx_gate(theta)),
                _ => None,
            },
            Self::RYY => match params[0] {
                Param::Float(theta) => Some(gate_matrix::ryy_gate(theta)),
                _ => None,
            },
            Self::RZZ => match params[0] {
                Param::Float(theta) => Some(gate_matrix::rzz_gate(theta)),
                _ => None,
            },
            Self::RZX => match params[0] {
                Param::Float(theta) => Some(gate_matrix::rzx_gate(theta)),
                _ => None,
            },
            Self::XXMinusYY => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(gate_matrix::xx_minus_yy_gate(*theta, *beta))
                }
                _ => None,
            },
            Self::XXPlusYY => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(gate_matrix::xx_plus_yy_gate(*theta, *beta))
                }
                _ => None,
            },
            Self::CCX => None,
            Self::CCZ => None,
            Self::CSwap => None,
            Self::RCCX => None,
            Self::C3X => None,
            Self::C3SX => None,
            Self::RC3X => None,
        }
    }

    /// Get the versor representation of a 1q [StandardGate] without constructing a matrix.
    ///
    /// Returns the error state if `gate` is not 1q, or if any of the parameters are symbolic.
    pub fn versor_u2(&self, params: &[Param]) -> Result<VersorU2, VersorU2Error> {
        convert::versor_u2(*self, params)
    }
}

#[pymethods]
impl StandardGate {
    pub fn copy(&self) -> Self {
        *self
    }

    // These pymethods are for testing:
    pub fn _to_matrix<'py>(
        &self,
        py: Python<'py>,
        params: Vec<Param>,
    ) -> Option<Bound<'py, PyArray2<Complex64>>> {
        self.matrix(&params).map(|x| x.into_pyarray(py))
    }

    pub fn _num_params(&self) -> u32 {
        self.num_params()
    }

    pub fn _get_definition(&self, params: Vec<Param>) -> Option<PyCircuitData> {
        self.definition(&params).map(Into::into)
    }

    pub fn _inverse(&self, params: Vec<Param>) -> Option<(StandardGate, SmallVec<[Param; 3]>)> {
        self.inverse(&params)
    }

    #[getter]
    pub fn get_num_qubits(&self) -> u32 {
        self.num_qubits()
    }

    #[getter]
    pub fn get_num_ctrl_qubits(&self) -> u32 {
        self.num_ctrl_qubits()
    }

    #[getter]
    pub fn get_num_clbits(&self) -> u32 {
        self.num_clbits()
    }

    #[getter]
    pub fn get_num_params(&self) -> u32 {
        self.num_params()
    }

    #[getter]
    pub fn get_name(&self) -> &str {
        self.name()
    }

    #[getter]
    pub fn is_controlled_gate(&self) -> bool {
        self.num_ctrl_qubits() > 0
    }

    #[getter]
    pub fn get_gate_class(&self, py: Python) -> PyResult<&'static Py<PyAny>> {
        imports::get_std_gate_class(py, *self)
    }

    #[staticmethod]
    pub fn all_gates(py: Python) -> PyResult<Bound<PyList>> {
        PyList::new(
            py,
            (0..STANDARD_GATE_SIZE as u8).map(::bytemuck::checked::cast::<_, Self>),
        )
    }

    pub fn __hash__(&self) -> isize {
        *self as isize
    }
}

// This must be kept up-to-date with `StandardGate` when adding or removing
// gates from the enum
//
// Remove this when std::mem::variant_count() is stabilized (see
// https://github.com/rust-lang/rust/issues/73662 )
pub const STANDARD_GATE_SIZE: usize = 52;

impl Operation for StandardGate {
    fn name(&self) -> &str {
        STANDARD_GATE_NAME[*self as usize]
    }
    fn num_qubits(&self) -> u32 {
        STANDARD_GATE_NUM_QUBITS[*self as usize]
    }
    fn num_clbits(&self) -> u32 {
        0
    }
    fn num_params(&self) -> u32 {
        STANDARD_GATE_NUM_PARAMS[*self as usize]
    }
    fn directive(&self) -> bool {
        false
    }
}
