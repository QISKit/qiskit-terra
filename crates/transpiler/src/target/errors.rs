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

use thiserror::Error;

/// A collection of the Errors possible in the [Target].
#[derive(Debug, Error)]
pub enum TargetError {
    /// An invalid instruction name being queried into the [Target].
    #[error["Provided instruction: '{0}' not in this Target."]]
    InvalidKey(String),
    /// An already existing instruction name being queried into the [Target].
    #[error["Instruction '{0}' is already in the target."]]
    AlreadyExists(String),
    /// An attempt to add collection of qargs to the [Target] that does
    /// not match the source instruction's number of qubits.
    #[error["The number of qubits for {instruction} does not match the number of qubits in the properties dictionary: {arguments}."]]
    QargsMismatch {
        instruction: String,
        arguments: String,
    },
    #[error(
        "The number of parameters for {instruction}: {instruction_num} does not match the provided number of parameters: {argument_num}."
    )]
    ParamsMismatch {
        instruction: String,
        instruction_num: usize,
        argument_num: usize,
    },
    /// An attempt to query collection of qargs to the [Target] that are
    /// not operated on by the specified instruction.
    #[error["Provided qarg {arguments} not in this Target for '{instruction}'."]]
    InvalidQargsKey {
        instruction: String,
        arguments: String,
    },
    /// An attempt to query collection of qargs to the [Target] that are
    /// not operated on by any instruction.
    #[error["{0} not in Target."]]
    QargsWithoutInstruction(String),
    ///The specified bounds for the instruction are not valid.
    #[error["Lower bound {low} is not less than higher bound {high}."]]
    InvalidBounds { low: f64, high: f64 },
}

impl From<TargetError> for ::pyo3::PyErr {
    fn from(val: TargetError) -> Self {
        crate::TranspilerError::new_err(val.to_string())
    }
}
