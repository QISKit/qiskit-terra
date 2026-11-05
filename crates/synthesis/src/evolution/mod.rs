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

mod chunks;
mod mcts;
mod pauli_network;

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};

use qiskit_circuit::circuit_data::{CircuitDataError, PyCircuitData};
use qiskit_circuit::operations::Param;
use qiskit_quantum_info::clifford::PauliListError;

use crate::QiskitError;
use crate::evolution::mcts::pauli_network_mcts_inner;
use crate::evolution::pauli_network::pauli_network_synthesis_inner;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EvolutionSynthesisError {
    // wraps PauliList error
    #[error(transparent)]
    ErrorFromPauliList(#[from] PauliListError),

    // wraps CircuitData error
    #[error(transparent)]
    ErrorFromCircuitData(#[from] CircuitDataError),

    // wraps a Clifford (re)synthesis error, which is reported as a plain string
    #[error("Clifford synthesis failed: {0}")]
    ErrorFromCliffordSynthesis(String),

    // invalid input arguments
    #[error("Invalid input argument: {0}")]
    ErrorInvalidInputArguments(String),
}

impl From<EvolutionSynthesisError> for PyErr {
    fn from(error: EvolutionSynthesisError) -> Self {
        match error {
            EvolutionSynthesisError::ErrorFromPauliList(err) => {
                QiskitError::new_err(err.to_string())
            }
            EvolutionSynthesisError::ErrorFromCircuitData(err) => err.into(),
            EvolutionSynthesisError::ErrorFromCliffordSynthesis(msg) => QiskitError::new_err(msg),
            EvolutionSynthesisError::ErrorInvalidInputArguments(msg) => QiskitError::new_err(msg),
        }
    }
}

/// Expands the sparse pauli string representation to the full representation.
///
/// For example: for the input `sparse_pauli = "XY", qubits = [1, 3], num_qubits = 6`,
/// the function returns `"IXIYII"`.
fn expand_pauli(num_qubits: usize, sparse_pauli: String, qubits: Vec<u32>) -> String {
    let mut v: Vec<char> = vec!['I'; num_qubits];
    for (q, p) in qubits.iter().zip(sparse_pauli.chars()) {
        v[*q as usize] = p;
    }
    v.into_iter().collect()
}

/// The input to Pauli network synthesis algorithm in Rust.
/// This includes a list of Paulis in the sparse format and a list of rotation angles.
type PauliNetworkInput = (Vec<(String, Vec<u32>)>, Vec<Param>);

/// Extracts sparse paulis and angles.
fn extract_sparse_paulis(pauli_network: &Bound<PyList>) -> PyResult<PauliNetworkInput> {
    let mut sparse_paulis: Vec<(String, Vec<u32>)> = Vec::with_capacity(pauli_network.len());
    let mut angles: Vec<Param> = Vec::with_capacity(pauli_network.len());

    // This does not allow minus signs, preserving the existing behavior.
    // We can extend this if needed.
    let allowed_chars = ['I', 'X', 'Y', 'Z'];

    // Go over the input pauli network and extract a list of pauli rotations,
    // qubits, and angles.
    for item in pauli_network {
        let tuple = item.cast::<PyTuple>()?;

        let sparse_pauli: String = tuple.get_item(0)?.cast::<PyString>()?.extract()?;
        let qubits: Vec<u32> = tuple.get_item(1)?.extract()?;
        let angle: Param = tuple.get_item(2)?.extract()?;

        if sparse_pauli.chars().any(|c| !allowed_chars.contains(&c)) {
            return Err(QiskitError::new_err(format!(
                "Pauli network contains invalid Pauli string {sparse_pauli}"
            )));
        }

        sparse_paulis.push((sparse_pauli, qubits));
        angles.push(angle);
    }
    Ok((sparse_paulis, angles))
}

// Converts sparse pauli representation into dense pauli representation.
fn paulis_to_dense(num_qubits: usize, sparse_paulis: Vec<(String, Vec<u32>)>) -> Vec<String> {
    sparse_paulis
        .into_iter()
        .map(|(sparse_pauli, qubits)| expand_pauli(num_qubits, sparse_pauli, qubits))
        .collect()
}

/// Calls Rustiq's Pauli network synthesis algorithm.
///
/// See python documentation for ``synth_pauli_network_rustiq`` for details.
#[pyfunction]
#[pyo3(signature = (num_qubits, pauli_network, optimize_count=true, preserve_order=true, upto_clifford=false, upto_phase=false, resynth_clifford_method=1))]
#[allow(clippy::too_many_arguments)]
pub fn pauli_network_synthesis(
    num_qubits: usize,
    pauli_network: &Bound<PyList>,
    optimize_count: bool,
    preserve_order: bool,
    upto_clifford: bool,
    upto_phase: bool,
    resynth_clifford_method: usize,
) -> PyResult<PyCircuitData> {
    let (sparse_paulis, angles) = extract_sparse_paulis(pauli_network)?;
    let paulis: Vec<String> = paulis_to_dense(num_qubits, sparse_paulis);

    pauli_network_synthesis_inner(
        num_qubits,
        paulis,
        angles,
        optimize_count,
        preserve_order,
        upto_clifford,
        upto_phase,
        resynth_clifford_method,
    )
    .map(Into::into)
    .map_err(Into::into)
}

/// Calls Monte Carlo Tree Search Pauli network synthesis algorithm.
///
/// See python documentation for ``synth_pauli_network_mcts`` for details.
#[pyfunction]
#[pyo3(signature = (num_qubits, pauli_network, preserve_order=true, upto_clifford=false, upto_phase=false, num_simulations=1, max_parallel_simulations=None))]
#[allow(clippy::too_many_arguments)]
pub fn pauli_network_mcts(
    num_qubits: usize,
    pauli_network: &Bound<PyList>,
    preserve_order: bool,
    upto_clifford: bool,
    upto_phase: bool,
    num_simulations: usize,
    max_parallel_simulations: Option<usize>,
) -> PyResult<PyCircuitData> {
    let (sparse_paulis, angles) = extract_sparse_paulis(pauli_network)?;
    pauli_network_mcts_inner(
        num_qubits,
        sparse_paulis,
        angles,
        preserve_order,
        upto_clifford,
        upto_phase,
        num_simulations,
        max_parallel_simulations,
    )
    .map(Into::into)
    .map_err(Into::into)
}

pub fn evolution(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pauli_network_synthesis, m)?)?;
    m.add_function(wrap_pyfunction!(pauli_network_mcts, m)?)?;
    Ok(())
}

pub mod suzuki_trotter;
