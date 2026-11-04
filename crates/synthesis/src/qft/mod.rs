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

mod qft_decompose_full;
mod qft_decompose_lnn;

use pyo3::prelude::*;
use qft_decompose_full::synth_qft_full;
use qft_decompose_lnn::synth_qft_line;

pub fn qft(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synth_qft_full, m)?)?;
    m.add_function(wrap_pyfunction!(synth_qft_line, m)?)?;
    Ok(())
}
