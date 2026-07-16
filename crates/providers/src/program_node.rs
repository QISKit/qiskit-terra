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

use crate::data_tree::{ArityMismatch, DataTree, TreeMatchError};
use crate::tensor::{DType, Tensor, TensorType};
use thiserror::Error;

/// Errors returned when a tree-shaped argument does not match [`ProgramNode::input_types`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CallInputError {
    #[error("missing required input {key:?}")]
    MissingInput { key: String },

    #[error("expected a leaf at {key:?}, found a branch")]
    ExpectedLeaf { key: String },

    #[error("unexpected dtype at {key:?}: expected {expected}, found {actual}")]
    UnexpectedDType {
        key: String,
        expected: String,
        actual: DType,
    },
}

impl From<TreeMatchError> for CallInputError {
    fn from(e: TreeMatchError) -> Self {
        match e {
            TreeMatchError::MissingPath { path } => Self::MissingInput { key: path },
            TreeMatchError::ExpectedLeaf { path } => Self::ExpectedLeaf { key: path },
        }
    }
}

/// Returned by implementations with a missing call implementation when called.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("node {0:?} does not implement call()")]
pub struct MissingCallError(pub String);

impl MissingCallError {
    /// Construct a new [`MissingCallError`] tagged with the node's full name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// Errors returned by [`ProgramNodeExt::call`].
#[derive(Debug, Error)]
pub enum CallError<E> {
    /// The input tree did not match the contract declared by `input_types()`.
    #[error(transparent)]
    Input(CallInputError),
    /// The node's [`ProgramNode::call_flat`] returned an error.
    #[error(transparent)]
    Call(E),
    /// The node's [`ProgramNode::call_flat`] returned a vector whose length
    /// did not match the leaf count of `output_types()`.
    #[error("call_flat returned {actual} outputs, expected {expected}")]
    OutputArityMismatch { expected: usize, actual: usize },
}

impl<E> From<ArityMismatch> for CallError<E> {
    fn from(e: ArityMismatch) -> Self {
        Self::OutputArityMismatch {
            expected: e.expected,
            actual: e.actual,
        }
    }
}

/// A node in a quantum program graph that transforms tensors.
pub trait ProgramNode {
    type CallError;

    /// The name of this program node.
    fn name(&self) -> &str;

    /// The namespace this program node belongs to.
    fn namespace(&self) -> &str;

    /// The namespace and name as one string.
    fn full_name(&self) -> String {
        format!("{}.{}", self.namespace(), self.name())
    }

    /// The inputs expected at call time.
    fn input_types(&self) -> &DataTree<TensorType>;

    /// The outputs promised on call return.
    fn output_types(&self) -> &DataTree<TensorType>;

    /// Whether this program node implements the call method.
    fn implements_call(&self) -> bool;

    /// The action of this program node with flattened I/O.
    ///
    /// `args` is in input-tree DFS leaf order matching `input_types()` and
    /// the returned vector is in output-tree DFS leaf order matching
    /// `output_types()`.
    ///
    /// # Panics
    ///
    /// Implementations are allowed to panic if `args.len()` does not equal
    /// the leaf count of `input_types()`; callers are responsible for upholding
    /// this invariant. On the other hand, implementations should raise a call
    /// error if they find tensors that they don't like.
    /// [`ProgramNodeExt::call`] and [`QuantumProgram::call_flat`] both do.
    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError>;
}

/// Extension with the wrapper over [`ProgramNode::call_flat`] whose I/O are data trees.
///
/// Provided via a blanket impl over every `T: ProgramNode` so that it cannot
/// be overridden in stable Rust.
pub trait ProgramNodeExt: ProgramNode {
    /// The action of this program node.
    fn call(
        &self,
        args: &DataTree<Tensor>,
    ) -> Result<DataTree<Tensor>, CallError<Self::CallError>> {
        let flat = self
            .input_types()
            .flatten_against(args)
            .map_err(|e| CallError::Input(e.into()))?;
        let out = self.call_flat(&flat).map_err(CallError::Call)?;
        self.output_types().unflatten(out).map_err(Into::into)
    }
}

impl<T: ProgramNode + ?Sized> ProgramNodeExt for T {}
