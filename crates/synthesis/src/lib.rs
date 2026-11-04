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

pub mod clifford;
pub mod discrete_basis;
pub mod euler_one_qubit_decomposer;
pub mod evolution;
pub mod linalg;
pub mod linear;
pub mod linear_phase;
pub mod matrix;
mod multi_controlled;
pub mod pauli_evolution;
pub mod pauli_products;
mod permutation;
mod qft;
pub mod qsd;
pub mod ross_selinger;
pub mod two_qubit_decompose;

use pyo3::import_exception;
use pyo3::prelude::*;

import_exception!(qiskit.exceptions, QiskitError);

pub fn synthesis(m: &Bound<PyModule>) -> PyResult<()> {
    let linear_mod = PyModule::new(m.py(), "linear")?;
    linear::linear(&linear_mod)?;
    m.add_submodule(&linear_mod)?;

    let linear_phase_mod = PyModule::new(m.py(), "linear_phase")?;
    linear_phase::linear_phase(&linear_phase_mod)?;
    m.add_submodule(&linear_phase_mod)?;

    let permutation_mod = PyModule::new(m.py(), "permutation")?;
    permutation::permutation(&permutation_mod)?;
    m.add_submodule(&permutation_mod)?;

    let clifford_mod = PyModule::new(m.py(), "clifford")?;
    clifford::clifford(&clifford_mod)?;
    m.add_submodule(&clifford_mod)?;

    let mc_mod = PyModule::new(m.py(), "multi_controlled")?;
    multi_controlled::multi_controlled(&mc_mod)?;
    m.add_submodule(&mc_mod)?;

    let pbc_mod = PyModule::new(m.py(), "pauli_products")?;
    pauli_products::pauli_products_mod(&pbc_mod)?;
    m.add_submodule(&pbc_mod)?;

    let evolution_mod = PyModule::new(m.py(), "evolution")?;
    evolution::evolution(&evolution_mod)?;
    m.add_submodule(&evolution_mod)?;

    let discrete_basis_mod = PyModule::new(m.py(), "discrete_basis")?;
    discrete_basis::discrete_basis(&discrete_basis_mod)?;
    m.add_submodule(&discrete_basis_mod)?;

    let qft_mod = PyModule::new(m.py(), "qft")?;
    qft::qft(&qft_mod)?;
    m.add_submodule(&qft_mod)?;

    Ok(())
}
