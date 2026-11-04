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

pub mod expr;
pub mod types;

use pyo3::prelude::*;

pub(crate) fn register_python(m: &Bound<PyModule>) -> PyResult<()> {
    let expr_mod = PyModule::new(m.py(), "expr")?;
    expr::register_python(&expr_mod)?;
    m.add_submodule(&expr_mod)?;

    let types_mod = PyModule::new(m.py(), "types")?;
    types::register_python(&types_mod)?;
    m.add_submodule(&types_mod)?;
    Ok(())
}
