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
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8, SQRT_2};

#[cfg(feature = "py")]
use pyo3::prelude::*;
#[cfg(feature = "py")]
use pyo3::wrap_pyfunction;

use crate::operations::{Operation, Param, StandardGate};
use qiskit_quantum_info::sparse_observable::SparseObservable;
use qiskit_util::complex::{C_ZERO, c64};

const C_FRAC_PI_2: Complex64 = c64(FRAC_PI_2, 0.0);
const C_FRAC_PI_4: Complex64 = c64(FRAC_PI_4, 0.0);
const C_FRAC_PI_8: Complex64 = c64(FRAC_PI_8, 0.0);
const C_FRAC_PI_2_SQRT_2: Complex64 = c64(FRAC_PI_2 / SQRT_2, 0.0);
const C_M_FRAC_PI_4: Complex64 = c64(-FRAC_PI_4, 0.0);
const C_M_FRAC_PI_8: Complex64 = c64(-FRAC_PI_8, 0.0);
const C_M_FRAC_PI_2_SQRT_2: Complex64 = c64(-FRAC_PI_2 / SQRT_2, 0.0);

// A constant cutoff below which we ignore the beta parameter of [StandardGate::XXPlusYY]
// and [StandardGate::XXMinusYY]
const BETA_TOLERANCE: f64 = 1e-10;

/// Return the exponent representation of a [StandardGate], if it is available.
///
/// We define the exponent $E$ of a gate $U$ as $U = \exp(-i E)$ up to a global phase.
/// For example, the [StandardGate::RX] has $E = \theta/2 X$. Since the return type is
/// [SparseObservable], which does not support parameterized coefficients, parameter values
/// of type [Param::ParameterExpression] default to 1.0.
///
/// # Arguments
///
/// * gate - The standard gate whose exponent we return.
/// * params - The gate's parameters. The length must equal the number of parameters the gate has.
///
/// # Returns
///
/// * Some(SparseObservable) - The exponent.
/// * None - If the exponent is not supported.
pub fn standard_gate_exponent(gate: StandardGate, params: &[Param]) -> Option<SparseObservable> {
    let fixed_params = params
        .iter()
        .map(|p| match p {
            Param::Float(f) => *f,
            Param::ParameterExpression(_) => 1.0,
            #[cfg(feature = "py")]
            Param::Obj(_) => panic!("StandardGate does not have Param::Obj parameters"),
            Param::Int(_) => panic!("StandardGate does not have Param::Int parameters"),
        })
        .collect::<Vec<f64>>();

    let num_qubits = gate.num_qubits();

    use qiskit_quantum_info::sparse_observable::BitTerm::*;

    let (coeffs, bit_terms, indices, boundaries) = match gate {
        StandardGate::GlobalPhase => (vec![c64(-fixed_params[0], 0.)], vec![], vec![], vec![0, 0]),
        // H = exp(-i pi/sqrt(8) (X + Z))
        StandardGate::H => (
            vec![C_FRAC_PI_2_SQRT_2, C_FRAC_PI_2_SQRT_2],
            vec![X, Z],
            vec![0, 0],
            vec![0, 1, 2],
        ),
        // X = exp(-i pi/2 X), Y = exp(-i pi/2 Y), Z = exp(-i pi/2 Z)
        StandardGate::X => (vec![C_FRAC_PI_2], vec![X], vec![0], vec![0, 1]),
        StandardGate::Y => (vec![C_FRAC_PI_2], vec![Y], vec![0], vec![0, 1]),
        StandardGate::Z => (vec![C_FRAC_PI_2], vec![Z], vec![0], vec![0, 1]),
        // Identity: exp(-i 0) = I.
        StandardGate::I => (vec![C_ZERO], vec![], vec![], vec![0, 0]),
        // S = exp(-i pi/4 Z), Sdg = exp(-i (-pi/4) Z)
        StandardGate::S => (vec![C_FRAC_PI_4], vec![Z], vec![0], vec![0, 1]),
        StandardGate::Sdg => (vec![C_M_FRAC_PI_4], vec![Z], vec![0], vec![0, 1]),
        // T = exp(-i pi/8 Z), Tdg = exp(-i (-pi/8) Z)
        StandardGate::T => (vec![C_FRAC_PI_8], vec![Z], vec![0], vec![0, 1]),
        StandardGate::Tdg => (vec![C_M_FRAC_PI_8], vec![Z], vec![0], vec![0, 1]),
        // SX = exp(-i pi/4 X), SXdg = exp(-i (-pi/4) X)
        StandardGate::SX => (vec![C_FRAC_PI_4], vec![X], vec![0], vec![0, 1]),
        StandardGate::SXdg => (vec![C_M_FRAC_PI_4], vec![X], vec![0], vec![0, 1]),
        // RX(t) = exp(-i t/2 X), RY, RZ=Phase, equivalently
        StandardGate::RX => (
            vec![c64(fixed_params[0] / 2.0, 0.0)],
            vec![X],
            vec![0],
            vec![0, 1],
        ),
        StandardGate::RY => (
            vec![c64(fixed_params[0] / 2.0, 0.0)],
            vec![Y],
            vec![0],
            vec![0, 1],
        ),
        StandardGate::RZ | StandardGate::Phase | StandardGate::U1 => (
            vec![c64(fixed_params[0] / 2.0, 0.0)],
            vec![Z],
            vec![0],
            vec![0, 1],
        ),
        // CX = exp(-i pi/4 (ZX - ZI - IX))
        StandardGate::CX => (
            vec![C_M_FRAC_PI_4, C_FRAC_PI_4, C_FRAC_PI_4],
            vec![Z, X, Z, X],
            vec![0, 1, 0, 1],
            vec![0, 2, 3, 4],
        ),
        // CY = exp(-i pi/4 (ZY - ZI - IY))
        StandardGate::CY => (
            vec![C_M_FRAC_PI_4, C_FRAC_PI_4, C_FRAC_PI_4],
            vec![Z, Y, Z, Y],
            vec![0, 1, 0, 1],
            vec![0, 2, 3, 4],
        ),
        // CZ = exp(-i pi/4 (ZZ - ZI - IZ))
        StandardGate::CZ => (
            vec![C_M_FRAC_PI_4, C_FRAC_PI_4, C_FRAC_PI_4],
            vec![Z, Z, Z, Z],
            vec![0, 1, 0, 1],
            vec![0, 2, 3, 4],
        ),
        // CS = exp(-i (-pi/8) (ZZ - ZI - IZ))
        StandardGate::CS => (
            vec![C_FRAC_PI_8, C_FRAC_PI_8, C_M_FRAC_PI_8],
            vec![Z, Z, Z, Z],
            vec![0, 1, 0, 1],
            vec![0, 1, 2, 4],
        ),
        // CSdg (inv sqrt(CZ)): factor +pi/8
        StandardGate::CSdg => (
            vec![C_M_FRAC_PI_8, C_M_FRAC_PI_8, C_FRAC_PI_8],
            vec![Z, Z, Z, Z],
            vec![0, 1, 0, 1],
            vec![0, 1, 2, 4],
        ),
        // CSX (sqrt(CX)/controlled-SX): factor -pi/8
        StandardGate::CSX => (
            vec![C_FRAC_PI_8, C_FRAC_PI_8, C_M_FRAC_PI_8],
            vec![Z, X, Z, X],
            vec![0, 1, 0, 1],
            vec![0, 1, 2, 4],
        ),
        // CRX(t) = exp(-i t/4 (-ZX + IX))
        StandardGate::CRX => (
            vec![
                c64(-fixed_params[0] / 4.0, 0.0),
                c64(fixed_params[0] / 4.0, 0.0),
            ],
            vec![Z, X, X],
            vec![0, 1, 1],
            vec![0, 2, 3],
        ),
        // CRY(t) = exp(-i t/4 (-ZY + IY))
        StandardGate::CRY => (
            vec![
                c64(-fixed_params[0] / 4.0, 0.0),
                c64(fixed_params[0] / 4.0, 0.0),
            ],
            vec![Z, Y, Y],
            vec![0, 1, 1],
            vec![0, 2, 3],
        ),
        // CRZ(t) = exp(-i t/4 (-ZZ + IZ))
        StandardGate::CRZ => (
            vec![
                c64(-fixed_params[0] / 4.0, 0.0),
                c64(fixed_params[0] / 4.0, 0.0),
            ],
            vec![Z, Z, Z],
            vec![0, 1, 1],
            vec![0, 2, 3],
        ),
        // CPhase(t) = exp(-i  t/4 (-ZZ + ZI + IZ))
        // (same as CRZ but shifts both Z0 and Z1, not just Z1)
        StandardGate::CPhase => (
            vec![
                c64(-fixed_params[0] / 4.0, 0.0),
                c64(fixed_params[0] / 4.0, 0.0),
                c64(fixed_params[0] / 4.0, 0.0),
            ],
            vec![Z, Z, Z, Z],
            vec![0, 1, 0, 1],
            vec![0, 2, 3, 4],
        ),
        // Swap = exp(-i pi/4 (XX + YY + ZZ))
        StandardGate::Swap => (
            vec![C_FRAC_PI_4, C_FRAC_PI_4, C_FRAC_PI_4],
            vec![X, X, Y, Y, Z, Z],
            vec![0, 1, 0, 1, 0, 1],
            vec![0, 2, 4, 6],
        ),
        // ISwap = exp(-i (-pi/4) (XX + YY))
        StandardGate::ISwap => (
            vec![C_M_FRAC_PI_4, C_M_FRAC_PI_4],
            vec![X, X, Y, Y],
            vec![0, 1, 0, 1],
            vec![0, 2, 4],
        ),
        // RXX(t) = exp(-i  t/2 XX)
        StandardGate::RXX => (
            vec![c64(fixed_params[0] / 2.0, 0.0)],
            vec![X, X],
            vec![0, 1],
            vec![0, 2],
        ),
        // RYY(t) = exp(-i t/2 YY)
        StandardGate::RYY => (
            vec![c64(fixed_params[0] / 2.0, 0.0)],
            vec![Y, Y],
            vec![0, 1],
            vec![0, 2],
        ),
        // RZZ(t) = exp(-i t/2 ZZ)
        StandardGate::RZZ => (
            vec![c64(fixed_params[0] / 2.0, 0.0)],
            vec![Z, Z],
            vec![0, 1],
            vec![0, 2],
        ),
        // RZX(t) = exp(-i t/2 ZX)
        StandardGate::RZX => (
            vec![c64(fixed_params[0] / 2.0, 0.0)],
            vec![Z, X],
            vec![0, 1],
            vec![0, 2],
        ),
        // XX+YY and XX-YY are just handled if the beta parameter is 0, in which case
        // the generator is XX +- YY.
        StandardGate::XXPlusYY | StandardGate::XXMinusYY => {
            // This covers both parametric ``beta`` and if they are above tolerance
            if fixed_params[1].abs() > BETA_TOLERANCE {
                return None;
            }

            match gate {
                StandardGate::XXPlusYY => (
                    vec![
                        c64(fixed_params[0] / 4.0, 0.0),
                        c64(fixed_params[0] / 4.0, 0.0),
                    ],
                    vec![X, X, Y, Y],
                    vec![0, 1, 0, 1],
                    vec![0, 2, 4],
                ),
                StandardGate::XXMinusYY => (
                    vec![
                        c64(fixed_params[0] / 4.0, 0.0),
                        c64(-fixed_params[0] / 4.0, 0.0),
                    ],
                    vec![X, X, Y, Y],
                    vec![0, 1, 0, 1],
                    vec![0, 2, 4],
                ),
                _ => unreachable!(),
            }
        }
        // CCX = exp(-i pi/8 (ZZX - ZIX - IZX - ZZI + ZII + IZI + IIX))
        StandardGate::CCX => (
            vec![
                C_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
            ],
            vec![Z, Z, X, Z, X, Z, X, Z, Z, Z, Z, X],
            vec![0, 1, 2, 0, 2, 1, 2, 0, 1, 0, 1, 2],
            vec![0, 3, 5, 7, 9, 10, 11, 12],
        ),
        // Same as CCX but with Z on the target
        StandardGate::CCZ => (
            vec![
                C_M_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
            ],
            vec![Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z],
            vec![0, 1, 2, 0, 2, 1, 2, 0, 1, 0, 1, 2],
            vec![0, 3, 5, 7, 9, 10, 11, 12],
        ),
        // CSwap = exp(-i pi/8 (ZII - ZXX - ZYY - ZZZ + IXX + IYY + IZZ))
        StandardGate::CSwap => (
            vec![
                C_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_M_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
                C_FRAC_PI_8,
            ],
            vec![Z, Z, X, X, Z, Y, Y, Z, Z, Z, X, X, Y, Y, Z, Z],
            vec![0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2],
            vec![0, 1, 4, 7, 10, 12, 14, 16],
        ),
        // ECR = exp(-i pi/sqrt(8) (IX - XY))
        StandardGate::ECR => (
            vec![C_FRAC_PI_2_SQRT_2, C_M_FRAC_PI_2_SQRT_2],
            vec![X, Y, X],
            vec![0, 0, 1],
            vec![0, 1, 3],
        ),
        _ => return None,
    };

    // SAFETY: The internal data was constructed manually and is consistent.
    Some(unsafe {
        SparseObservable::new_unchecked(num_qubits, coeffs, bit_terms, indices, boundaries)
    })
}

#[cfg(feature = "py")]
#[pyfunction(name = "_standard_gate_exponent")]
#[pyo3(signature = (gate, params=None))]
pub fn py_standard_gate_exponent(
    gate: StandardGate,
    params: Option<Vec<Param>>,
) -> Option<SparseObservable> {
    let params = params.unwrap_or_default();
    standard_gate_exponent(gate, &params)
}

#[cfg(feature = "py")]
pub fn standard_generators(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_standard_gate_exponent, m)?)?;
    Ok(())
}
