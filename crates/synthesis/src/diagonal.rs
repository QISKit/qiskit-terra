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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::ucrz::get_ucrz;
use crate::utils::append_with_qubit_map;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError, PyCircuitData};
use qiskit_circuit::operations::Param;

/// Synthesize a `CircuitData` implementing a diagonal gate with phases `diag_phases`,
/// via recursive UCRZ decomposition (see [`get_ucrz`]). Follows [1]. `diag_phases` is
/// consumed/overwritten as scratch space during the recursion.
///
/// [1]: https://arxiv.org/pdf/quant-ph/0406176.pdf
pub(crate) fn diagonal_gate_circuit(diag_phases: &[f64]) -> Result<CircuitData, CircuitDataError> {
    let num_qubits = diag_phases.len().trailing_zeros() as usize;
    let mut diag_phases = diag_phases.to_vec();

    // Worst-case instruction count across all recursion levels is 2 * diag_phases.len() - 3
    let capacity = (2 * diag_phases.len()).saturating_sub(3);
    let mut circuit = CircuitData::with_capacity(num_qubits as u32, 0, capacity, Param::Float(0.))?;

    let mut n = diag_phases.len();

    while n >= 2 {
        let mut angles_rz = Vec::new();
        for i in (0..n).step_by(2) {
            let phi1 = diag_phases[i];
            let phi2 = diag_phases[i + 1];
            diag_phases[i / 2] = (phi1 + phi2) / 2.0;
            angles_rz.push(phi2 - phi1);
        }
        let num_act_qubits = n.trailing_zeros() as usize;
        let target_qubit = num_qubits - num_act_qubits;
        let ucrz = get_ucrz(num_act_qubits, &mut angles_rz, true)?;

        let qubit_map: Vec<Qubit> = (0..num_act_qubits)
            .map(|q| Qubit((q + target_qubit) as u32))
            .collect();
        append_with_qubit_map(&mut circuit, ucrz, &qubit_map)?;
        n /= 2;
    }
    circuit.add_global_phase(&Param::Float(diag_phases[0]))?;
    Ok(circuit)
}

/// Python-exposed entry point for synthesizing a `DiagonalGate`. `diag_phases` must have
/// exactly `2^num_qubits` entries.
#[pyfunction]
pub fn synth_diagonal(diag_phases: Vec<f64>, num_qubits: u32) -> PyResult<PyCircuitData> {
    let expected = 1usize
        .checked_shl(num_qubits)
        .ok_or_else(|| PyValueError::new_err(format!("num_qubits={num_qubits} is too large")))?;
    let got = diag_phases.len();
    if got != expected {
        return Err(PyValueError::new_err(format!(
            "expected {expected} diagonal phases for {num_qubits} qubits, got {got}"
        )));
    }
    let circuit = diagonal_gate_circuit(&diag_phases).map_err(PyErr::from)?;
    Ok(circuit.into())
}

pub fn diagonal(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synth_diagonal, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::diagonal_gate_circuit;
    use approx::abs_diff_eq;
    use ndarray::Array2;
    use num_complex::Complex64;

    use crate::matrix::sim::sim_unitary_circuit;

    #[test]
    fn test_diagonal_gate_circuit_synthesizes_diagonal() {
        let phases = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let circuit = diagonal_gate_circuit(phases.as_ref()).unwrap();
        let unitary = sim_unitary_circuit(&circuit).unwrap();

        let mut expected = Array2::zeros((8, 8));
        for (index, phase) in phases.iter().enumerate() {
            expected[[index, index]] = Complex64::new(0.0, *phase).exp();
        }
        assert!(abs_diff_eq!(unitary, expected, epsilon = 1e-12));
    }
}
