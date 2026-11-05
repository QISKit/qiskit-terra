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

#![allow(
    clippy::self_named_module_files,
    // See https://github.com/rust-lang/rust-clippy/issues/11916
    reason = "false positive in clippy for Rust <= 1.91",
)]

pub mod angle_bound_registry;
pub mod commutation_checker;
pub mod equivalence;
pub mod neighbors;
pub mod passes;
pub mod standard_equivalence_library;
pub mod standard_gates_commutations;
pub mod target;
pub mod transpile_layout;

pub mod transpiler;

pub use transpiler::transpile;

mod gate_metrics;

use pyo3::import_exception;

import_exception! {qiskit.exceptions, QiskitError}
import_exception! {qiskit.transpiler.exceptions, TranspilerError}
