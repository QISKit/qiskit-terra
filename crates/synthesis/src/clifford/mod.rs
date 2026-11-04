// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

mod bm_synthesis;
pub(crate) mod greedy_synthesis;
mod random_clifford;
pub(crate) mod utils;

use crate::QiskitError;
use crate::clifford::bm_synthesis::synth_clifford_bm_inner;
use crate::clifford::greedy_synthesis::GreedyCliffordSynthesis;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::Param;

/// Create a circuit that synthesizes a given Clifford operator represented as a tableau.
///
/// This is an implementation of the "greedy Clifford compiler" presented in
/// Appendix A of the paper "Clifford Circuit Optimization with Templates and Symbolic
/// Pauli Gates" by Bravyi, Shaydulin, Hu, and Maslov (2021), `<https://arxiv.org/abs/2105.02291>`__.
///
/// This method typically yields better CX cost compared to the Aaronson-Gottesman method.
///
/// Note that this function only implements the greedy Clifford compiler and not the
/// templates and symbolic Pauli gates optimizations that are also described in the paper.
#[pyfunction]
#[pyo3(signature = (clifford))]
fn synth_clifford_greedy(clifford: PyReadonlyArray2<bool>) -> PyResult<PyCircuitData> {
    let tableau = clifford.as_array();
    let mut greedy_synthesis =
        GreedyCliffordSynthesis::new(tableau.view()).map_err(QiskitError::new_err)?;
    let (num_qubits, clifford_gates) = greedy_synthesis.run().map_err(QiskitError::new_err)?;

    Ok(
        CircuitData::from_standard_gates(num_qubits as u32, clifford_gates, Param::Float(0.0))?
            .into(),
    )
}

/// Generate a random Clifford tableau.
///
/// The Clifford is sampled using the method of the paper "Hadamard-free circuits
/// expose the structure of the Clifford group" by S. Bravyi and D. Maslov (2020),
/// `https://arxiv.org/abs/2003.09412`__.
///
/// Args:
///     num_qubits: the number of qubits.
///     seed: an optional random seed.
/// Returns:
///     result: a random clifford tableau.
#[pyfunction]
#[pyo3(signature = (num_qubits, seed=None))]
fn random_clifford_tableau(
    py: Python,
    num_qubits: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyArray2<bool>>> {
    let tableau = random_clifford::random_clifford_tableau_inner(num_qubits, seed);
    Ok(tableau.into_pyarray(py).unbind())
}

/// Create a circuit that optimally synthesizes a given Clifford operator represented as
/// a tableau for Cliffords up to 3 qubits.
///
/// This implementation follows the paper "Hadamard-free circuits expose the structure
/// of the Clifford group" by S. Bravyi, D. Maslov (2020), `<https://arxiv.org/abs/2003.09412>`__.
#[pyfunction]
#[pyo3(signature = (clifford))]
fn synth_clifford_bm(clifford: PyReadonlyArray2<bool>) -> PyResult<PyCircuitData> {
    let tableau = clifford.as_array();
    let (num_qubits, clifford_gates) =
        synth_clifford_bm_inner(tableau).map_err(QiskitError::new_err)?;
    Ok(
        CircuitData::from_standard_gates(num_qubits as u32, clifford_gates, Param::Float(0.0))?
            .into(),
    )
}

pub fn clifford(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synth_clifford_greedy, m)?)?;
    m.add_function(wrap_pyfunction!(synth_clifford_bm, m)?)?;
    m.add_function(wrap_pyfunction!(random_clifford_tableau, m)?)?;
    Ok(())
}
