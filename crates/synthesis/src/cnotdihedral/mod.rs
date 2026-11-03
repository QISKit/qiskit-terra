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

pub mod dihedral;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::Param;

use crate::QiskitError;
use dihedral::CNOTDihedral;

/// Build a Rust-space ``CNOTDihedral`` element from the components of the
/// Python-space one: the binary ``linear`` matrix and ``shift`` vector of its
/// affine part, and the packed phase-polynomial coefficients (reduced mod 8).
#[allow(clippy::too_many_arguments)]
fn build_elem(
    linear: PyReadonlyArray2<bool>,
    shift: PyReadonlyArray1<bool>,
    weight0: u8,
    weight1: PyReadonlyArray1<u8>,
    weight2: PyReadonlyArray1<u8>,
    weight3: PyReadonlyArray1<u8>,
) -> PyResult<CNOTDihedral> {
    CNOTDihedral::from_parts(
        linear.as_array(),
        shift.as_array(),
        weight0,
        weight1.as_array(),
        weight2.as_array(),
        weight3.as_array(),
    )
    .map_err(QiskitError::new_err)
}

macro_rules! synth_pyfunction {
    ($name:ident, $inner:path, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        #[pyo3(signature = (linear, shift, weight0, weight1, weight2, weight3))]
        #[allow(clippy::too_many_arguments)]
        fn $name(
            linear: PyReadonlyArray2<bool>,
            shift: PyReadonlyArray1<bool>,
            weight0: u8,
            weight1: PyReadonlyArray1<u8>,
            weight2: PyReadonlyArray1<u8>,
            weight3: PyReadonlyArray1<u8>,
        ) -> PyResult<PyCircuitData> {
            let elem = build_elem(linear, shift, weight0, weight1, weight2, weight3)?;
            let gates = $inner(&elem).map_err(QiskitError::new_err)?;
            Ok(
                CircuitData::from_standard_gates(elem.num_qubits as u32, gates, Param::Float(0.0))?
                    .into(),
            )
        }
    };
}

synth_pyfunction!(
    synth_cnotdihedral_two_qubits,
    dihedral::synth_cnotdihedral_two_qubits_inner,
    "Optimal-CX decomposition of a 1-qubit or 2-qubit CNOTDihedral element, based on the \
     structure of the CNOT-Dihedral group described in Garion & Cross, \
     `arXiv:2006.12042 <https://arxiv.org/abs/2006.12042>`__."
);
synth_pyfunction!(
    synth_cnotdihedral_general,
    dihedral::synth_cnotdihedral_general_inner,
    "Decomposition of a general CNOTDihedral element, based on the scalable randomized \
     benchmarking construction of Cross, Magesan, Bishop, Smolin & Gambetta, \
     npj Quantum Inf 2, 16012 (2016).  The CX count is not necessarily optimal."
);
synth_pyfunction!(
    synth_cnotdihedral_full,
    dihedral::synth_cnotdihedral_full_inner,
    "Decomposition of a CNOTDihedral element: optimal for up to 2 qubits, general otherwise."
);

pub fn cnotdihedral(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synth_cnotdihedral_two_qubits, m)?)?;
    m.add_function(wrap_pyfunction!(synth_cnotdihedral_general, m)?)?;
    m.add_function(wrap_pyfunction!(synth_cnotdihedral_full, m)?)?;
    Ok(())
}
