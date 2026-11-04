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

//! A whole computation: the dataflow graphs it is made of, and the entry point a caller reaches it
//! through.

mod program_function;
mod quantum_program;

pub use program_function::{
    FunctionError, FunctionEvalError, InstructionId, InstructionRef, InstructionRole,
    InstructionView, ProgramFunction, Signature, Value,
};
pub use quantum_program::{FunctionId, ProgramError, ProgramEvalError, QuantumProgram};
