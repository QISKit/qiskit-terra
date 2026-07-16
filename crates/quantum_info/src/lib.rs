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

pub mod clifford;
pub mod pauli_lindblad_map;
pub mod sparse_observable;
#[cfg(feature = "python")]
pub mod sparse_pauli_op;
pub mod unitary_compose;
pub mod versor_u2;

#[cfg(feature = "python")] // Only currently used by python remove if needed from rust
mod rayon_ext;
#[cfg(test)]
mod test;

#[cfg(feature = "python")]
use pyo3::import_exception;

#[cfg(feature = "python")]
pub(crate) mod imports {
    use qiskit_util::py::ImportOnceCell;

    pub static PAULI_TYPE: ImportOnceCell = ImportOnceCell::new("qiskit.quantum_info", "Pauli");
    pub static PAULI_LIST_TYPE: ImportOnceCell =
        ImportOnceCell::new("qiskit.quantum_info", "PauliList");
}
#[cfg(feature = "python")]
import_exception!(qiskit.exceptions, QiskitError);
