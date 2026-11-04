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

use pyo3::PyErr;
use qiskit_circuit::{
    circuit_data::CircuitError, error::DAGCircuitError,
    parameter::parameter_expression::ParameterError,
};
use thiserror::Error;

use crate::TranspilerError;

/// A collection of the Errors possible in the [run_basis_translator] function
#[derive(Debug, Error)]
pub enum BasisTranslatorError {
    #[error[
        "Unable to translate the operations in the circuit: \
        {basis} to the backend's (or manually specified) target \
        basis: {expanded}. This likely means the target basis is not universal \
        or there are additional equivalence rules needed in the EquivalenceLibrary being \
        used. For more details on this error see: \
        https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.passes.\
        BasisTranslator#translation-errors"
    ]]
    TargetMissingEquivalence { basis: String, expanded: String },
    // TODO: Use rust native CircuitError
    #[error[
        "{0}"
    ]]
    BasisCircuitError(String),
    // TODO: Use rust native DAGCircuitError
    #[error[
    "{0}"
    ]]
    BasisDAGCircuitError(String),
    #[error[
        "BasisTranslator did not map {0}"
    ]]
    ApplyTranslationMappingError(String),
    #[error[
        "Translation num_params not equal to op num_params. \
            Op: {:?} {} Translation: {:?}\n{:?}",
            node_params,
            node_name,
            target_params,
            target_dag
    ]]
    ReplaceNodeParamMismatch {
        node_params: String,
        node_name: String,
        target_params: String,
        target_dag: String,
    },
    #[error[
        "Global phase must be real, but got {0}"
    ]]
    ReplaceNodeGlobalPhaseComplex(String),
    #[error(transparent)]
    BasisParameterError(#[from] ParameterError),
}

impl From<BasisTranslatorError> for PyErr {
    fn from(value: BasisTranslatorError) -> Self {
        match value {
            BasisTranslatorError::BasisCircuitError(_) => CircuitError::new_err(value.to_string()),
            BasisTranslatorError::BasisDAGCircuitError(message) => {
                DAGCircuitError::new_err(message)
            }
            _ => TranspilerError::new_err(value.to_string()),
        }
    }
}
