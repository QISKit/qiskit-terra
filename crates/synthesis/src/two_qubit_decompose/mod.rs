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

use pyo3::prelude::*;

mod basis_decomposer;
mod common;
mod controlled_u_decomposer;
mod gate_sequence;
mod weyl_decomposition;

pub use basis_decomposer::{TwoQubitBasisDecomposer, two_qubit_decompose_up_to_diagonal};
pub use controlled_u_decomposer::{RXXEquivalent, TwoQubitControlledUDecomposer};
pub use gate_sequence::TwoQubitGateSequence;
pub use weyl_decomposition::{Specialization, TwoQubitWeylDecomposition};

pub fn two_qubit_decompose(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(weyl_decomposition::_num_basis_gates))?;
    m.add_wrapped(wrap_pyfunction!(
        weyl_decomposition::py_decompose_two_qubit_product_gate
    ))?;
    m.add_wrapped(wrap_pyfunction!(
        basis_decomposer::py_two_qubit_decompose_up_to_diagonal
    ))?;
    m.add_wrapped(wrap_pyfunction!(common::two_qubit_local_invariants))?;
    m.add_wrapped(wrap_pyfunction!(common::local_equivalence))?;
    m.add_wrapped(wrap_pyfunction!(common::py_trace_to_fid))?;
    m.add_wrapped(wrap_pyfunction!(common::py_ud))?;
    m.add_wrapped(wrap_pyfunction!(weyl_decomposition::weyl_coordinates))?;
    m.add_class::<weyl_decomposition::TwoQubitWeylDecomposition>()?;
    m.add_class::<weyl_decomposition::Specialization>()?;
    m.add_class::<basis_decomposer::TwoQubitBasisDecomposer>()?;
    m.add_class::<controlled_u_decomposer::TwoQubitControlledUDecomposer>()?;
    Ok(())
}
