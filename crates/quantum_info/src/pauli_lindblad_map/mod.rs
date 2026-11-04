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

pub mod pauli_lindblad_map_class;
pub mod phased_qubit_sparse_pauli;
pub mod qubit_sparse_pauli;

#[cfg(feature = "python")]
use pauli_lindblad_map_class::PyPauliLindbladMap;
#[cfg(feature = "python")]
use phased_qubit_sparse_pauli::{PyPhasedQubitSparsePauli, PyPhasedQubitSparsePauliList};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use qubit_sparse_pauli::{PyQubitSparsePauli, PyQubitSparsePauliList};

#[cfg(feature = "python")]
pub fn pauli_lindblad_map(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPauliLindbladMap>()?;
    m.add_class::<PyQubitSparsePauli>()?;
    m.add_class::<PyQubitSparsePauliList>()?;
    m.add_class::<PyPhasedQubitSparsePauli>()?;
    m.add_class::<PyPhasedQubitSparsePauliList>()?;
    Ok(())
}
