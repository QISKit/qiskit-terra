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

//! Qiskit middle-end intermediate representation (MIR) for fault-tolerant quantum programs.
//!
//! This is the base crate that defines the IR, its data structures and built-in operations and
//! algorithms.

pub mod instruction;
pub mod line_graph;
pub mod mir;
pub mod ops;
pub mod passes;
pub mod py;
pub mod qid;
pub mod symbol_table;

pub use py::mir_mod;
