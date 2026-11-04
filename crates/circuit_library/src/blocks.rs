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

use pyo3::{
    prelude::*,
    types::{PyList, PyTuple},
};
use qiskit_circuit::{
    NoBlocks,
    circuit_instruction::OperationFromPython,
    operations::{Operation, Param, StandardGate},
    packed_instruction::PackedOperation,
};
use smallvec::SmallVec;

use crate::{QiskitError, entanglement::get_entanglement};

#[derive(Debug, Clone)]
pub enum BlockOperation {
    Standard { gate: StandardGate },
    PyCustom { builder: Py<PyAny> },
}

impl BlockOperation {
    pub fn assign_parameters(
        &self,
        params: &[&Param],
    ) -> PyResult<(PackedOperation, SmallVec<[Param; 3]>)> {
        match self {
            Self::Standard { gate } => Ok((
                (*gate).into(),
                SmallVec::from_iter(params.iter().map(|&p| p.clone())),
            )),
            Self::PyCustom { builder } => Python::attach(|py| {
                // the builder returns a Python operation plus the bound parameters
                let py_params = PyList::new(py, params.iter().map(|&p| p.clone()))?.into_any();

                let job = builder.call1(py, (py_params,))?;
                let result = job.cast_bound::<PyTuple>(py)?;

                let operation: OperationFromPython<NoBlocks> = result.get_item(0)?.extract()?;
                let bound_params = result
                    .get_item(1)?
                    .try_iter()?
                    .map(|ob| Param::extract_no_coerce(ob?.as_borrowed()))
                    .collect::<PyResult<SmallVec<[Param; 3]>>>()?;

                Ok((operation.operation, bound_params))
            }),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct Block {
    pub operation: BlockOperation,
    pub num_qubits: u32,
    pub num_parameters: usize,
}

#[pymethods]
impl Block {
    #[staticmethod]
    #[pyo3(signature = (gate,))]
    pub fn from_standard_gate(gate: StandardGate) -> Self {
        Block {
            operation: BlockOperation::Standard { gate },
            num_qubits: gate.num_qubits(),
            num_parameters: gate.num_params() as usize,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (num_qubits, num_parameters, builder,))]
    pub fn from_callable(
        num_qubits: i64,
        num_parameters: i64,
        builder: &Bound<PyAny>,
    ) -> PyResult<Self> {
        if !builder.is_callable() {
            return Err(QiskitError::new_err(
                "builder must be a callable: parameters->(bound gate, bound gate params)",
            ));
        }
        let block = Block {
            operation: BlockOperation::PyCustom {
                builder: builder.clone().unbind(),
            },
            num_qubits: num_qubits as u32,
            num_parameters: num_parameters as usize,
        };

        Ok(block)
    }
}

// We introduce typedefs to make the types more legible. We can understand the hierarchy
// as follows:
// Connection: Vec<u32> -- indices that the multi-qubit gate acts on
// BlockEntanglement: Vec<Connection> -- entanglement for single block
// LayerEntanglement: Vec<BlockEntanglement> -- entanglements for all blocks in the layer
// Entanglement: Vec<LayerEntanglement> -- entanglement for every layer
type BlockEntanglement = Vec<Vec<u32>>;
pub(super) type LayerEntanglement = Vec<BlockEntanglement>;

/// Represent the entanglement in an n-local circuit.
pub struct Entanglement {
    // Possible optimization in future: This eagerly expands the full entanglement for every layer.
    // This could be done more efficiently, e.g., by creating entanglement objects that store
    // their underlying representation (e.g. a string or a list of connections) and returning
    // these when given a layer-index.
    pub entanglement_vec: Vec<LayerEntanglement>,
}

impl Entanglement {
    /// Create an entanglement from the input of an n_local circuit.
    pub fn from_py(
        num_qubits: u32,
        reps: usize,
        entanglement: &Bound<PyAny>,
        entanglement_blocks: &[&Block],
    ) -> PyResult<Self> {
        let entanglement_vec = (0..reps)
            .map(|layer| -> PyResult<LayerEntanglement> {
                if entanglement.is_callable() {
                    let as_any = entanglement.call1((layer,))?;
                    let as_list = as_any.cast::<PyList>()?;
                    unpack_entanglement(num_qubits, layer, as_list, entanglement_blocks)
                } else {
                    let as_list = entanglement.cast::<PyList>()?;
                    unpack_entanglement(num_qubits, layer, as_list, entanglement_blocks)
                }
            })
            .collect::<PyResult<_>>()?;

        Ok(Self { entanglement_vec })
    }

    pub fn get_layer(&self, layer: usize) -> &LayerEntanglement {
        &self.entanglement_vec[layer]
    }

    pub fn iter(&self) -> impl Iterator<Item = &LayerEntanglement> {
        self.entanglement_vec.iter()
    }
}

fn unpack_entanglement(
    num_qubits: u32,
    layer: usize,
    entanglement: &Bound<PyList>,
    entanglement_blocks: &[&Block],
) -> PyResult<LayerEntanglement> {
    entanglement_blocks
        .iter()
        .zip(entanglement.iter())
        .map(|(block, ent)| -> PyResult<Vec<Vec<u32>>> {
            get_entanglement(num_qubits, block.num_qubits, &ent, layer)?.collect()
        })
        .collect()
}
