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

//! Multi-controlled single-qubit gate decomposition up to diagonal (`MCGupDiag`).
//!
//! Decomposes a multi-controlled unitary $U$ acting on $k$ control qubits and 1 target
//! qubit into a circuit $U'$ such that $U = D \cdot U'$, where $D$ is a diagonal gate.
//! Implemented by reducing to [`dec_ucg_inner`] with $2^k$ single-qubit unitaries
//! where all slots are identity except the last (all controls active) which receives the target gate.

use numpy::PyReadonlyArray2;
use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::uc_gate::dec_ucg_inner;

use nalgebra::{Matrix2, MatrixView2};
use num_complex::Complex64;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};

/// Synthesizes a multi-controlled single-qubit gate up to a diagonal gate.
///
/// Constructs a multiplexed gate sequence of length `2^num_ctrls` with identities in all
/// positions except the last (all controls set to 1), then delegates to [`dec_ucg_inner`]
/// with `up_to_diagonal = true` and `mux_simp = true`.
///
/// # Arguments
///
/// * `gate` - The 2x2 unitary matrix acting on the target qubit when all controls are 1.
/// * `num_ctrls` - The number of control qubits.
///
/// # Returns
///
/// A tuple `(circuit, diag)` where `circuit` is the `CircuitData` implementing the gate
/// up to diagonal, and `diag` is the diagonal phases vector of length `2^(num_ctrls + 1)`.
pub(crate) fn mcg_up_to_diagonal_inner(
    gate: Matrix2<Complex64>,
    num_ctrls: u32,
) -> Result<(CircuitData, Vec<Complex64>), CircuitDataError> {
    let num_qubits = num_ctrls + 1;

    let mut gates = vec![Matrix2::identity(); 2usize.pow(num_ctrls)];
    let last = gates.len() - 1;
    gates[last] = gate;
    dec_ucg_inner(gates, num_qubits, true, true)
}

/// Python-exposed entry point for decomposing a multi-controlled single-qubit gate up to diagonal.
///
/// See [`mcg_up_to_diagonal_inner`] for the core synthesis logic. This wrapper converts the
/// input 2x2 NumPy matrix to Rust and the resulting `CircuitData` to a Python `QuantumCircuit`.
///
/// # Arguments
///
/// * `gate` - A 2x2 complex numpy array representing the target unitary.
/// * `num_ctrls` - The number of control qubits.
///
/// # Returns
///
/// A tuple `(circuit, diag)` containing the `QuantumCircuit` and the list of complex diagonal elements.
#[pyfunction]
pub fn mcg_up_to_diagonal_synth(
    py: Python,
    gate: PyReadonlyArray2<Complex64>,
    num_ctrls: u32,
) -> PyResult<(Py<PyAny>, Vec<Complex64>)> {
    let gate_rs: Matrix2<Complex64> = gate
        .try_as_matrix()
        .map(|m: MatrixView2<Complex64>| m.into_owned())
        .ok_or_else(|| {
            PyValueError::new_err("expected a 2x2 unitary matrix for each single-qubit gate")
        })?;

    let (circuit, diag) = mcg_up_to_diagonal_inner(gate_rs, num_ctrls).map_err(PyErr::from)?;
    let qc = circuit.into_py_quantum_circuit(py)?;
    qc.setattr("name", "mcg")?;
    Ok((qc.unbind(), diag))
}

/// Registers the `mcg_up_to_diagonal` Python submodule functions.
pub fn mcg_up_to_diagonal(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mcg_up_to_diagonal_synth, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::mcg_up_to_diagonal_inner;
    use crate::matrix::sim::sim_unitary_circuit;
    use crate::uc_gate::dec_ucg_inner;
    use approx::abs_diff_eq;
    use nalgebra::Matrix2;
    use ndarray::Array2;
    use num_complex::Complex64;

    // Helper: build a 2x2 unitary from a real angle (Ry-like rotation).
    // Matches the ry() helper in uc_gate.rs's tests.
    fn ry(theta: f64) -> Matrix2<Complex64> {
        let c = Complex64::new((theta / 2.0).cos(), 0.0);
        let s = Complex64::new((theta / 2.0).sin(), 0.0);
        Matrix2::new(c, -s, s, c)
    }

    // A generic (non-real) SU(2) gate, so tests aren't only exercising the
    // real-orthogonal `ry` gates above.
    fn su2(theta: f64, phi: f64, lam: f64) -> Matrix2<Complex64> {
        let c = Complex64::new((theta / 2.0).cos(), 0.0);
        let s = Complex64::new((theta / 2.0).sin(), 0.0);
        let e_i = |x: f64| Complex64::new(0.0, x).exp();
        Matrix2::new(c, -e_i(lam) * s, e_i(phi) * s, e_i(phi + lam) * c)
    }

    fn diag_matrix(diag: &[Complex64]) -> Array2<Complex64> {
        let n = diag.len();
        let mut out = Array2::zeros((n, n));
        for (i, d) in diag.iter().enumerate() {
            out[[i, i]] = *d;
        }
        out
    }

    #[test]
    fn test_mcg_up_to_diagonal_inner_no_controls() {
        let gate = ry(0.5);
        let (circuit, diag) = mcg_up_to_diagonal_inner(gate, 0).unwrap();
        assert_eq!(circuit.num_qubits(), 1);
        // With no controls there is nothing to defer to a trailing diagonal.
        assert_eq!(diag, vec![Complex64::ONE; 2]);

        let unitary = sim_unitary_circuit(&circuit).unwrap();
        let mut expected = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                expected[[i, j]] = gate[(i, j)];
            }
        }
        assert!(abs_diff_eq!(unitary, expected, epsilon = 1e-12));
    }

    #[test]
    fn test_mcg_up_to_diagonal_inner_sizes() {
        // For `num_ctrls` controls, the circuit acts on `num_ctrls + 1` qubits
        // and the diagonal has one entry per basis state.
        for num_ctrls in 0..4_u32 {
            let gate = ry(0.3 + num_ctrls as f64);
            let (circuit, diag) = mcg_up_to_diagonal_inner(gate, num_ctrls).unwrap();
            assert_eq!(circuit.num_qubits(), (num_ctrls + 1) as usize);
            assert_eq!(diag.len(), 1usize << (num_ctrls + 1));
        }
    }

    // A multi-controlled gate is a uniformly controlled gate where every basis
    // state except the all-ones one gets the identity. Folding the diagonal
    // `mcg_up_to_diagonal_inner` returns back onto its circuit must therefore
    // reproduce exactly what `dec_ucg_inner` computes when asked to fold the
    // diagonal in itself (`up_to_diagonal = false`) on that same gate list.
    fn check_matches_full_ucg_decomposition(gate: Matrix2<Complex64>, num_ctrls: u32) {
        let (up_to_diag_circuit, diag) = mcg_up_to_diagonal_inner(gate, num_ctrls).unwrap();
        let up_to_diag_unitary = sim_unitary_circuit(&up_to_diag_circuit).unwrap();
        let reconstructed = diag_matrix(&diag).dot(&up_to_diag_unitary);

        let num_qubits = num_ctrls + 1;
        let mut gates = vec![Matrix2::identity(); 2usize.pow(num_ctrls)];
        let last = gates.len() - 1;
        gates[last] = gate;
        let (full_circuit, _) = dec_ucg_inner(gates, num_qubits, false, true).unwrap();
        let full_unitary = sim_unitary_circuit(&full_circuit).unwrap();

        assert!(abs_diff_eq!(reconstructed, full_unitary, epsilon = 1e-10));
    }

    #[test]
    fn test_mcg_up_to_diagonal_inner_one_control() {
        check_matches_full_ucg_decomposition(ry(0.9), 1);
    }

    #[test]
    fn test_mcg_up_to_diagonal_inner_two_controls() {
        check_matches_full_ucg_decomposition(ry(1.3), 2);
    }

    #[test]
    fn test_mcg_up_to_diagonal_inner_three_controls() {
        check_matches_full_ucg_decomposition(ry(-0.4), 3);
    }

    #[test]
    fn test_mcg_up_to_diagonal_inner_complex_gate() {
        check_matches_full_ucg_decomposition(su2(0.7, 0.2, -0.5), 2);
    }
}
