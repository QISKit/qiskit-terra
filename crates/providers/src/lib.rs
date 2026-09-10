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

//! The quantum program: the dataflow IR that Qiskit's `BackendV3` interface consumes.
//!
//! A program describes a hybrid quantum-classical computation as typed tensor values produced and
//! consumed by instructions.
//!
//! - [`tensor`] is the value domain. A [`Tensor`](tensor::Tensor) is a dense array over one of a
//!   fixed set of dtypes, and a [`TensorType`](tensor::TensorType) is its data-less counterpart used during
//!   static analysis.
//! - [`ops`] defines the [`ProgramOp`] trait and its Qiskit implementations. An op
//!   may be defined outside this crate, in its own namespace.
//! - [`program`] defines [`QuantumProgram`], which is a list of [`ProgramFunction`]s.
//! - [`data_tree`] defines [`DataTree`], the container for nested structured values used to describe the
//!   IO contract of a quantum program.
//! - [`render`] writes a program out for a person to read, as a listing or as a drawing.
//! - [`python`] binds the parts of all this that `qiskit.quantum_program` exposes, behind the
//!   `python` feature.
pub mod backend_v3;
pub mod data_tree;
pub mod ops;
pub mod program;
#[cfg(feature = "python")]
pub mod python;
pub mod render;
pub mod tensor;

pub use backend_v3::{BackendV3, Job, RunResult};
pub use data_tree::{ArityMismatch, DataTree, InvalidName, PathEntry, TreeMatchError};
pub use ops::{BoxedOpError, BoxedProgramOp, Constant, ErasedProgramOp, ProgramOp};
pub use program::{
    FunctionError, FunctionId, InstructionId, InstructionRef, InstructionRole, InstructionView,
    PartitionError, ProgramError, ProgramFunction, ProgramStepper, QuantumProgram, Request,
    RequestId, Signature, StepperError, Value, partition,
};
