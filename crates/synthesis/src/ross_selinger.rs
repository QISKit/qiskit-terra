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

use ndarray::ArrayView2;
use num_complex::Complex64;

use pyo3::prelude::*;

use numpy::PyReadonlyArray2;

use rsgridsynth::clear_caches;
use rsgridsynth::config::config_from_theta_epsilon;
use rsgridsynth::gridsynth::gridsynth_gates;

use crate::QiskitError;
use crate::euler_one_qubit_decomposer::params_zxz_inner;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardGate};

use std::f64::consts::{FRAC_PI_4, FRAC_PI_8};

/// Creates a circuit from the (iterator over) gate names that are returned by rsgridsynth.
fn circuit_from_string<S>(s: S, phase: f64, instruction_capacity: usize) -> PyResult<CircuitData>
where
    S: IntoIterator<Item = char>,
{
    let qubit = [Qubit(0)];
    let mut circuit =
        CircuitData::with_capacity(1u32, 0, instruction_capacity, Param::Float(phase))?;
    for c in s {
        match c {
            'I' => {}
            'H' => {
                circuit.push_standard_gate(StandardGate::H, &[], &qubit)?;
            }
            'S' => {
                circuit.push_standard_gate(StandardGate::S, &[], &qubit)?;
            }
            'T' => {
                circuit.push_standard_gate(StandardGate::T, &[], &qubit)?;
            }
            'X' => {
                circuit.push_standard_gate(StandardGate::X, &[], &qubit)?;
            }
            'W' => {
                // Note that W in gridsynth represents the global phase update by pi/4,
                // and not the Clifford W-gate used in Qiskit.
                circuit.add_global_phase(&Param::Float(FRAC_PI_4))?;
            }
            _ => {
                return Err(QiskitError::new_err(format!(
                    "Unknown character in gridsynth output: {c}"
                )));
            }
        };
    }
    Ok(circuit)
}

pub fn gridsynth_rz(theta: f64, epsilon: f64) -> PyResult<CircuitData> {
    let res = gridsynth_gates(&mut config_from_theta_epsilon(
        theta, epsilon, 0u64, false, true,
    ));
    let gates_iter = res.gates.chars();
    let phase = if res.global_phase { FRAC_PI_8 } else { 0. };
    let instruction_capacity = res.gates.len();
    circuit_from_string(gates_iter, phase, instruction_capacity)
}

// rsgridsynth internally caches results of various computations that depend on the
// number of precision bits. Generally speaking, these caches become invalid when
// precision changes. Fortunately, rsgridsynth exposes a method to clear some (though not all)
// of these caches.
pub fn gridsynth_cleanup() {
    clear_caches();
}

#[pyfunction]
#[pyo3(name = "gridsynth_rz")]
pub fn py_gridsynth_rz(theta: f64, epsilon: f64) -> PyResult<PyCircuitData> {
    gridsynth_rz(theta, epsilon).map(Into::into)
}

/// Approximates 1q unitary matrix using Ross-Selinger algorithm
/// as implemented in https://github.com/qiskit-community/rsgridsynth.
pub fn gridsynth_unitary(mat: ArrayView2<Complex64>, epsilon: f64) -> PyResult<CircuitData> {
    // Run ZXZ decomposition
    let [theta, phi, lambda, euler_phase] = params_zxz_inner(mat);

    // Approximate each of the RZ, RX, RZ rotations using rsgridsynth and join the results.
    let res_gates_theta = gridsynth_gates(&mut config_from_theta_epsilon(
        theta, epsilon, 0u64, false, true,
    ));
    let res_gates_phi = gridsynth_gates(&mut config_from_theta_epsilon(
        phi, epsilon, 0u64, false, true,
    ));
    let res_gates_lambda = gridsynth_gates(&mut config_from_theta_epsilon(
        lambda, epsilon, 0u64, false, true,
    ));
    let instruction_capacity =
        res_gates_theta.gates.len() + res_gates_phi.gates.len() + res_gates_lambda.gates.len() + 2;
    let gates_iter = res_gates_lambda
        .gates
        .chars()
        .chain(std::iter::once('H'))
        .chain(res_gates_theta.gates.chars())
        .chain(std::iter::once('H'))
        .chain(res_gates_phi.gates.chars());

    let phase = euler_phase
        + if res_gates_theta.global_phase {
            FRAC_PI_8
        } else {
            0.
        }
        + if res_gates_phi.global_phase {
            FRAC_PI_8
        } else {
            0.
        }
        + if res_gates_lambda.global_phase {
            FRAC_PI_8
        } else {
            0.
        };
    circuit_from_string(gates_iter, phase, instruction_capacity)
}

/// Approximates 1q unitary matrix using Ross-Selinger algorithm
/// as implemented in https://github.com/qiskit-community/rsgridsynth.
#[pyfunction]
#[pyo3(name = "gridsynth_unitary")]
pub fn py_gridsynth_unitary(
    unitary: PyReadonlyArray2<Complex64>,
    epsilon: f64,
) -> PyResult<PyCircuitData> {
    gridsynth_unitary(unitary.as_array(), epsilon).map(Into::into)
}

pub fn ross_selinger_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_gridsynth_rz, m)?)?;
    m.add_function(wrap_pyfunction!(py_gridsynth_unitary, m)?)?;
    Ok(())
}
