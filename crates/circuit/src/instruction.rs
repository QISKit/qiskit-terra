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

use crate::circuit_data::CircuitData;
use crate::operations::{OperationRef, Param};
use ndarray::Array2;
use num_complex::Complex64;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use smallvec::SmallVec;

/// The parameter list of an instruction.
#[derive(Clone, Debug)]
pub enum Parameters<T> {
    Params(SmallVec<[Param; 3]>),
    Blocks(Vec<T>),
}

impl<T> Parameters<T> {
    /// Get the number of parameters in this parameter list.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Parameters::Params(params) => params.len(),
            Parameters::Blocks(blocks) => blocks.len(),
        }
    }

    /// Check if the parameter list is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Unwraps the parameter list as params.
    ///
    /// Panics if this is not a params list.
    #[inline]
    pub fn unwrap_params(self) -> SmallVec<[Param; 3]> {
        match self {
            Parameters::Params(params) => params,
            Parameters::Blocks(_) => panic!("expected params, got blocks"),
        }
    }

    /// Unwraps the parameter list as a slice of blocks.
    ///
    /// Panics if this is not a block list.
    #[inline]
    pub fn unwrap_blocks(self) -> Vec<T> {
        match self {
            Parameters::Params(_) => panic!("expected blocks, got params"),
            Parameters::Blocks(blocks) => blocks,
        }
    }

    /// Get a cloned version of these parameters, using a fallible mapping function to map any
    /// contained blocks into a new type.
    pub fn try_map_blocks_ref<S, E>(
        &self,
        block_map_fn: impl FnMut(&T) -> Result<S, E>,
    ) -> Result<Parameters<S>, E> {
        match self {
            Self::Params(params) => Ok(Parameters::Params(params.clone())),
            Self::Blocks(blocks) => blocks
                .iter()
                .map(block_map_fn)
                .collect::<Result<_, _>>()
                .map(Parameters::Blocks),
        }
    }

    /// Get a mapped version of these parameters, using a fallible mapping function to map any
    /// contained blocks into a new type.
    pub fn try_map_blocks<S, E>(
        self,
        block_map_fn: impl FnMut(T) -> Result<S, E>,
    ) -> Result<Parameters<S>, E> {
        match self {
            Self::Params(params) => Ok(Parameters::Params(params)),
            Self::Blocks(blocks) => blocks
                .into_iter()
                .map(block_map_fn)
                .collect::<Result<_, _>>()
                .map(Parameters::Blocks),
        }
    }

    /// Get a cloned version of these parameters, using an infallible mapping function to map any
    /// contained blocks into a new type.
    #[inline]
    pub fn map_blocks_ref<S>(&self, mut block_map_fn: impl FnMut(&T) -> S) -> Parameters<S> {
        let Ok(out) = self.try_map_blocks_ref(|block| -> Result<S, ::std::convert::Infallible> {
            Ok(block_map_fn(block))
        });
        out
    }

    /// Get a mapped version of these parameters, using an infallible mapping function to map any
    /// contained blocks into a new type.
    #[inline]
    pub fn map_blocks<S>(self, mut block_map_fn: impl FnMut(T) -> S) -> Parameters<S> {
        let Ok(out) = self.try_map_blocks(|block| -> Result<S, ::std::convert::Infallible> {
            Ok(block_map_fn(block))
        });
        out
    }
}

/// Represents an instruction that is directly convertible to our Python API
/// instruction type (i.e. owns its `params` data and label).
///
/// It's implemented by our unpacked instruction types like
/// [CircuitInstruction] and [OperationFromPython] which own all the data they
/// need to be converted back to a Python instance.
pub trait Instruction {
    type Block;

    /// Gets a reference to this instruction's operation.
    fn op(&self) -> OperationRef<'_>;

    /// Get a reference to this instruction's parameter list, if applicable.
    ///
    /// For standard gates without parameters this may be [None] or a
    /// `Some(Parameters::Param(smallvec![]))`.
    fn parameters(&self) -> Option<&Parameters<Self::Block>>;

    /// Get the label for this instruction.
    fn label(&self) -> Option<&str>;

    /// Get a slice view onto the contained parameters.
    #[inline]
    fn params_view(&self) -> &[Param] {
        self.parameters()
            .and_then(|p| match p {
                Parameters::Params(p) => Some(p.as_slice()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Get a slice view onto the contained blocks.
    #[inline]
    fn blocks_view(&self) -> &[Self::Block] {
        self.parameters()
            .and_then(|p| match p {
                Parameters::Blocks(b) => Some(b.as_slice()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Gets an owned matrix from the instruction, if applicable.
    fn try_matrix(&self) -> Option<Array2<Complex64>> {
        match self.op() {
            OperationRef::StandardGate(g) => g.matrix(self.params_view()),
            OperationRef::PyCustom(i) => i.matrix(),
            OperationRef::Unitary(u) => u.matrix(),
            OperationRef::PauliProductRotation(ppr) => ppr.matrix(),
            _ => None,
        }
    }
}

pub fn create_py_op(
    py: Python,
    op: OperationRef,
    params: Option<Parameters<CircuitData>>,
    label: Option<&str>,
) -> PyResult<Py<PyAny>> {
    match op {
        OperationRef::ControlFlow(cf) => {
            cf.create_py_op(py, params.map(|p| p.unwrap_blocks()), label)
        }
        OperationRef::PauliProductMeasurement(ppm) => ppm.create_py_op(py, label),
        OperationRef::PauliProductRotation(rotation) => rotation.create_py_op(py, label),
        OperationRef::StandardGate(gate) => {
            gate.create_py_op(py, params.map(|p| p.unwrap_params()), label)
        }
        OperationRef::StandardInstruction(instruction) => {
            instruction.create_py_op(py, params.map(|p| p.unwrap_params()), label)
        }
        OperationRef::PyCustom(inst) => Ok(inst.ob.clone_ref(py)),
        OperationRef::Unitary(unitary) => unitary.create_py_op(py, label),
        OperationRef::CustomOperation(_) => Err(PyNotImplementedError::new_err(
            "Custom operations from Rust cannot be exposed to Python",
        )),
    }
}
