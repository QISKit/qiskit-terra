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

use qiskit_circuit::parameter::parameter_expression::ParameterError;
use qiskit_quantum_info::sparse_observable::ArithmeticError;
use qiskit_transpiler::target::TargetError;
use thiserror::Error;

/// Errors related to C input.
#[derive(Error, Debug)]
pub enum CInputError {
    #[error("Unexpected null pointer.")]
    NullPointerError,
    #[error("Non-aligned memory.")]
    AlignmentError,
    #[error("Index out of bounds.")]
    IndexError,
}

/// Integer exit codes returned to C.
#[repr(u32)]
#[derive(PartialEq, Eq, Debug)]
pub enum ExitCode {
    /// Success.
    Success = 0,
    /// Error related to data input.
    CInputError = 100,
    /// Unexpected null pointer.
    NullPointerError = 101,
    /// Pointer is not aligned to expected data.
    AlignmentError = 102,
    /// Index out of bounds.
    IndexError = 103,
    /// Duplicate index.
    DuplicateIndexError = 104,
    /// Invalid ``QkOperationKind``.
    InvalidOperationKind = 105,
    /// Error related to arithmetic operations or similar.
    ArithmeticError = 200,
    /// Mismatching number of qubits.
    MismatchedQubits = 201,
    /// Matrix is not unitary.
    ExpectedUnitary = 202,
    /// Target related error
    TargetError = 300,
    /// Instruction already exists in the Target
    TargetInstAlreadyExists = 301,
    /// Properties with incorrect qargs was added
    TargetQargMismatch = 302,
    /// Trying to query into the target with non-existent qargs.
    TargetInvalidQargsKey = 303,
    /// Querying an operation that doesn't exist in the Target.
    TargetInvalidInstKey = 304,
    /// Transpilation failed
    TranspilerError = 400,
    /// Incompatible types.
    IncompatibleTypes = 401,
    /// Type casting error.
    CastingError = 402,
    /// QkDag operation error
    DagError = 500,
    /// The DAGs have mismatching qubit/clbit amounts during compose.
    DagComposeMismatch = 501,
    /// One or more bit indices were not found during compose.
    DagComposeMissingBit = 502,
    /// Errors concerning parameter handling.
    ParameterError = 600,
    /// Parameter name conflict.
    ParameterNameConflict = 601,
    /// QPY serialization, deserialization, or file I/O failed.
    QpyError = 700,
    /// QkCircuit operation error
    CircuitError = 800,
    /// Incorrect delay unit added to the circuit. Use specialized function.
    IncorrectDelayUnit = 801,
    /// CustomOperation error
    CustomOperation = 900,
    /// Repeated vtable slot error
    CustomOperationRepeatedSlot = 901,
}

impl From<ArithmeticError> for ExitCode {
    fn from(value: ArithmeticError) -> Self {
        match value {
            ArithmeticError::MismatchedQubits { left: _, right: _ } => ExitCode::MismatchedQubits,
            ArithmeticError::InvalidOperation(_) => ExitCode::ArithmeticError,
            ArithmeticError::DuplicatedIndex => ExitCode::DuplicateIndexError,
            ArithmeticError::OutOfBounds(_) => ExitCode::IndexError,
        }
    }
}

impl From<CInputError> for ExitCode {
    fn from(value: CInputError) -> Self {
        match value {
            CInputError::AlignmentError => ExitCode::AlignmentError,
            CInputError::NullPointerError => ExitCode::NullPointerError,
            CInputError::IndexError => ExitCode::IndexError,
        }
    }
}

impl From<TargetError> for ExitCode {
    fn from(value: TargetError) -> Self {
        match value {
            TargetError::InvalidKey(_) => ExitCode::TargetInvalidInstKey,
            TargetError::AlreadyExists(_) => ExitCode::TargetInstAlreadyExists,
            TargetError::QargsMismatch {
                instruction: _,
                arguments: _,
            } => ExitCode::TargetQargMismatch,
            TargetError::InvalidQargsKey {
                instruction: _,
                arguments: _,
            } => ExitCode::TargetInvalidQargsKey,
            _ => ExitCode::TargetError,
        }
    }
}

impl From<ParameterError> for ExitCode {
    fn from(_value: ParameterError) -> Self {
        ExitCode::ArithmeticError
    }
}
