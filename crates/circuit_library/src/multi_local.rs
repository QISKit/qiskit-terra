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

use std::ops::Deref;

use hashbrown::HashSet;
use pyo3::prelude::*;
use pyo3::types::PyString;
use qiskit_circuit::packed_instruction::PackedOperation;
use smallvec::{SmallVec, smallvec};

use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardInstruction};
use qiskit_circuit::{Clbit, Qubit};

use itertools::izip;

use crate::parameter_ledger::ParameterLedger;

use super::blocks::{Block, Entanglement, LayerEntanglement};
use super::parameter_ledger::{LayerParameters, LayerType};

type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

/// Construct a rotation layer.
///
/// # Arguments
///
/// - `num_qubits`: The number of qubits in the circuit.
/// - `rotation_blocks`: A reference to a vector containing the instructions to insert.
///   This is a vector (since we can have multiple operations per layer), with
///   3-tuple elements containing (packed_operation, num_qubits, num_params).
/// - `parameters`: The set of parameter objects to use for the operations. This is a 3x nested
///   vector, organized as operation -> block -> param. That means that for operation `i`
///   and block `j`, the parameters are given by `parameters[i][j]`.
/// - skipped_qubits: A hash-set containing which qubits are skipped in the rotation layer.
///
/// # Returns
///
/// An iterator for the rotation instructions.
fn rotation_layer<'a>(
    num_qubits: u32,
    rotation_blocks: &'a [&'a Block],
    parameters: Vec<Vec<Vec<&'a Param>>>,
    skipped_qubits: &'a HashSet<u32>,
) -> impl Iterator<Item = Result<Instruction, CircuitDataError>> + 'a {
    rotation_blocks
        .iter()
        .zip(parameters)
        .flat_map(move |(block, block_params)| {
            (0..num_qubits)
                .step_by(block.num_qubits as usize)
                .filter(move |start_idx| {
                    skipped_qubits.is_disjoint(&HashSet::from_iter(
                        *start_idx..(*start_idx + block.num_qubits),
                    ))
                })
                .zip(block_params)
                .map(move |(start_idx, params)| {
                    let (bound_op, bound_params) = block
                        .operation
                        .assign_parameters(&params)
                        .expect("Failed to rebind");
                    Ok((
                        bound_op,
                        bound_params,
                        (0..block.num_qubits)
                            .map(|i| Qubit(start_idx + i))
                            .collect(),
                        vec![] as Vec<Clbit>,
                    ))
                })
        })
}

/// Construct an entanglement layer.
///
/// # Arguments
///
/// - `entanglement`: The entanglement structure in this layer. Given as 3x nested vector, which
///   for each entanglement block contains a vector of connections, where each connection
///   is a vector of indices.
/// - `entanglement_blocks`: A reference to a vector containing the instructions to insert.
///   This is a vector (since we can have multiple entanglement operations per layer), with
///   3-tuple elements containing (packed_operation, num_qubits, num_params).
/// - `parameters`: The set of parameter objects to use for the operations. This is a 3x nested
///   vector, organized as operation -> block -> param. That means that for operation `i`
///   and block `j`, the parameters are given by `parameters[i][j]`.
///
/// # Returns
///
/// An iterator for the entanglement instructions.
fn entanglement_layer<'a>(
    entanglement: &'a LayerEntanglement,
    entanglement_blocks: &'a [&'a Block],
    parameters: LayerParameters<'a>,
) -> impl Iterator<Item = Result<Instruction, CircuitDataError>> + 'a {
    let zipped = izip!(entanglement_blocks, parameters, entanglement);
    zipped.flat_map(move |(block, block_params, block_entanglement)| {
        block_entanglement
            .iter()
            .zip(block_params)
            .map(move |(indices, params)| {
                let (bound_op, bound_params) = block
                    .operation
                    .assign_parameters(&params)
                    .expect("Failed to rebind");
                Ok((
                    bound_op,
                    bound_params,
                    indices.iter().map(|i| Qubit(*i)).collect(),
                    vec![] as Vec<Clbit>,
                ))
            })
    })
}

/// # Arguments
///
/// - `num_qubits`: The number of qubits of the circuit.
/// - `rotation_blocks`: The blocks used in the rotation layers. If multiple are passed,
///   these will be applied one after another (like new sub-layers).
/// - `entanglement_blocks`: The blocks used in the entanglement layers. If multiple are passed,
///   these will be applied one after another.
/// - `entanglement`: The indices specifying on which qubits the input blocks act. This is
///   specified by string describing an entanglement strategy (see the additional info)
///   or a list of qubit connections.
///   If a list of entanglement blocks is passed, different entanglement for each block can
///   be specified by passing a list of entanglements. To specify varying entanglement for
///   each repetition, pass a callable that takes as input the layer and returns the
///   entanglement for that layer.
///   Defaults to ``"full"``, meaning an all-to-all entanglement structure.
/// - `reps`: Specifies how often the rotation blocks and entanglement blocks are repeated.
/// - `insert_barriers`: If ``True``, barriers are inserted in between each layer. If ``False``,
///   no barriers are inserted.
/// - `parameter_prefix`: The prefix used if default parameters are generated.
/// - `skip_final_rotation_layer`: Whether a final rotation layer is added to the circuit.
/// - `skip_unentangled_qubits`: If ``True``, the rotation gates act only on qubits that
///   are entangled. If ``False``, the rotation gates act on all qubits.
///
/// # Returns
///
/// An N-local circuit.
#[allow(clippy::too_many_arguments)]
pub fn n_local(
    num_qubits: u32,
    rotation_blocks: &[&Block],
    entanglement_blocks: &[&Block],
    entanglement: &Entanglement,
    reps: usize,
    insert_barriers: bool,
    parameter_prefix: &str,
    skip_final_rotation_layer: bool,
    skip_unentangled_qubits: bool,
) -> PyResult<CircuitData> {
    // Construct the parameter ledger, which will define all free parameters and provide
    // access to them, given an index for a layer and the current gate to implement.
    let ledger = ParameterLedger::from_nlocal(
        num_qubits,
        reps,
        entanglement,
        rotation_blocks,
        entanglement_blocks,
        skip_final_rotation_layer,
        &parameter_prefix.to_string(),
    );

    // Compute the qubits that are skipped in the rotation layer. If this is set,
    // we skip qubits that do not appear in any of the entanglement layers.
    let skipped_qubits = if skip_unentangled_qubits {
        let active: HashSet<&u32> =
            HashSet::from_iter(entanglement.iter().flatten().flatten().flatten());
        HashSet::from_iter((0..num_qubits).filter(|i| !active.contains(i)))
    } else {
        HashSet::new()
    };

    // This struct can be used to yield barrier if insert_barriers is true, otherwise
    // it returns an empty iterator. For conveniently injecting barriers in-between operations.
    let maybe_barrier = MaybeBarrier::new(num_qubits, insert_barriers)?;

    let packed_insts = (0..reps).flat_map(|layer| {
        rotation_layer(
            num_qubits,
            rotation_blocks,
            ledger.get_parameters(LayerType::Rotation, layer),
            &skipped_qubits,
        )
        .chain(maybe_barrier.get())
        .chain(entanglement_layer(
            entanglement.get_layer(layer),
            entanglement_blocks,
            ledger.get_parameters(LayerType::Entangle, layer),
        ))
        .chain(maybe_barrier.get())
    });
    if !skip_final_rotation_layer {
        let packed_insts = packed_insts.chain(rotation_layer(
            num_qubits,
            rotation_blocks,
            ledger.get_parameters(LayerType::Rotation, reps),
            &skipped_qubits,
        ));
        Ok(CircuitData::from_packed_operations(
            num_qubits,
            0,
            packed_insts,
            Param::Float(0.0),
        )?)
    } else {
        Ok(CircuitData::from_packed_operations(
            num_qubits,
            0,
            packed_insts,
            Param::Float(0.0),
        )?)
    }
}

#[pyfunction]
#[pyo3(signature = (num_qubits, rotation_blocks, entanglement_blocks, entanglement, reps, insert_barriers, parameter_prefix, skip_final_rotation_layer, skip_unentangled_qubits))]
#[allow(clippy::too_many_arguments)]
pub fn py_n_local(
    num_qubits: u32,
    rotation_blocks: Vec<PyRef<Block>>,
    entanglement_blocks: Vec<PyRef<Block>>,
    entanglement: &Bound<PyAny>,
    reps: usize,
    insert_barriers: bool,
    parameter_prefix: &Bound<PyString>,
    skip_final_rotation_layer: bool,
    skip_unentangled_qubits: bool,
) -> PyResult<PyCircuitData> {
    // Normalize the Python data.
    let parameter_prefix = parameter_prefix.to_string();
    let rotation_blocks: Vec<&Block> = rotation_blocks
        .iter()
        .map(|py_block| py_block.deref())
        .collect();
    let entanglement_blocks: Vec<&Block> = entanglement_blocks
        .iter()
        .map(|py_block| py_block.deref())
        .collect();

    // Expand the entanglement. This will (currently) eagerly expand the entanglement for each
    // circuit layer.
    let entanglement = Entanglement::from_py(num_qubits, reps, entanglement, &entanglement_blocks)?;

    Ok(n_local(
        num_qubits,
        &rotation_blocks,
        &entanglement_blocks,
        &entanglement,
        reps,
        insert_barriers,
        &parameter_prefix,
        skip_final_rotation_layer,
        skip_unentangled_qubits,
    )?
    .into())
}

/// A convenient struct to optionally yield barriers to inject in-between circuit layers.
///
/// If constructed with ``insert_barriers=false``, then the method ``.get`` yields empty iterators,
/// otherwise it will yield a barrier. This is a struct such that the call to Python that
/// creates the Barrier object can be done a single time, but barriers can be yielded multiple times.
struct MaybeBarrier {
    barrier: Option<Instruction>,
}

impl MaybeBarrier {
    fn new(num_qubits: u32, insert_barriers: bool) -> PyResult<Self> {
        if !insert_barriers {
            Ok(Self { barrier: None })
        } else {
            let inst = (
                PackedOperation::from_standard_instruction(StandardInstruction::Barrier(
                    num_qubits,
                )),
                smallvec![],
                (0..num_qubits).map(Qubit).collect(),
                vec![] as Vec<Clbit>,
            );

            Ok(Self {
                barrier: Some(inst),
            })
        }
    }

    fn get(&self) -> Box<dyn Iterator<Item = Result<Instruction, CircuitDataError>>> {
        match &self.barrier {
            None => Box::new(std::iter::empty()),
            Some(inst) => Box::new(std::iter::once(Ok(inst.clone()))),
        }
    }
}
