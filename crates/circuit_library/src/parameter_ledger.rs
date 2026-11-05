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

use qiskit_circuit::{
    operations::Param,
    parameter::{parameter_expression::ParameterExpression, symbol_expr::SymbolVector},
};

use super::blocks::{Block, Entanglement};

/// Enum to determine the type of circuit layer.
pub(super) enum LayerType {
    Rotation,
    Entangle,
}

type BlockParameters<'a> = Vec<Vec<&'a Param>>; // parameters for each gate in the block
pub(super) type LayerParameters<'a> = Vec<BlockParameters<'a>>; // parameter in a layer

/// The ParameterLedger stores the parameter objects contained in the n-local circuit.
///
/// Internally, the parameters are stored in a 1-D vector and the ledger keeps track of
/// which indices belong to which layer. For example, a 2-qubit circuit where both the
/// rotation and entanglement layer have 1 block with 2 parameters each, we would store
///
///     [x0 x1 x2 x3 x4 x5 x6 x7 ....]
///      ----- ----- ----- -----
///      rep0  rep0  rep1  rep2
///      rot   ent   rot   ent
///
/// This allows accessing the parameters by index of the rotation or entanglement layer by means
/// of the ``get_parameters`` method, e.g. as
///
///     let layer: usize = 4;
///     let params_in_that_layer: LayerParameters =
///         ledger.get_parameter(LayerType::Rotation, layer);
///
pub(super) struct ParameterLedger {
    parameter_vector: Vec<Param>, // all parameters
    rotation_indices: Vec<usize>, // indices where rotation blocks start
    entangle_indices: Vec<usize>,
    rotations: Vec<(u32, usize)>, // (#blocks per layer, #params per block) for each block
    entanglements: Vec<Vec<(u32, usize)>>, // this might additionally change per layer
}

impl ParameterLedger {
    /// Initialize the ledger n-local input data. This will call Python to create a new
    /// ``ParameterVector`` of adequate size and compute all required indices to access
    /// parameter of a specific layer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_nlocal(
        num_qubits: u32,
        reps: usize,
        entanglement: &Entanglement,
        rotation_blocks: &[&Block],
        entanglement_blocks: &[&Block],
        skip_final_rotation_layer: bool,
        parameter_prefix: &String,
    ) -> Self {
        // if we keep the final layer (i.e. skip=false), add parameters on the final layer
        let final_layer_rep = match skip_final_rotation_layer {
            true => 0,
            false => 1,
        };

        // compute the number of parameters used for the rotation layers
        let mut num_rotation_params_per_layer: usize = 0;
        let mut rotations: Vec<(u32, usize)> = Vec::new();

        for block in rotation_blocks {
            let num_blocks = num_qubits / block.num_qubits;
            rotations.push((num_blocks, block.num_parameters));
            num_rotation_params_per_layer += (num_blocks as usize) * block.num_parameters;
        }

        // compute the number of parameters used for the entanglement layers
        let mut num_entangle_params_per_layer: Vec<usize> = Vec::with_capacity(reps);
        let mut entanglements: Vec<Vec<(u32, usize)>> = Vec::with_capacity(reps);
        for this_entanglement in entanglement.iter() {
            let mut this_entanglements: Vec<(u32, usize)> = Vec::new();
            let mut this_num_params: usize = 0;
            for (block, block_entanglement) in entanglement_blocks.iter().zip(this_entanglement) {
                let num_blocks = block_entanglement.len();
                this_num_params += num_blocks * block.num_parameters;
                this_entanglements.push((num_blocks as u32, block.num_parameters));
            }
            num_entangle_params_per_layer.push(this_num_params);
            entanglements.push(this_entanglements);
        }

        let num_rotation_params: usize = (reps + final_layer_rep) * num_rotation_params_per_layer;
        let num_entangle_params: usize = num_entangle_params_per_layer.iter().sum();

        // generate a ParameterVector Python-side, containing all parameters, and then
        // map it onto Rust-space parameters
        let num_parameters = num_rotation_params + num_entangle_params;

        let parameter_vector: Vec<Param> =
            SymbolVector::new(parameter_prefix.to_string(), num_parameters)
                .iter()
                .map(|sym| Param::ParameterExpression(ParameterExpression::from_symbol(sym).into()))
                .collect();

        // finally, distribute the parameters onto the repetitions and blocks for each
        // rotation layer and entanglement layer
        let mut rotation_indices: Vec<usize> = Vec::with_capacity(reps + final_layer_rep);
        let mut entangle_indices: Vec<usize> = Vec::with_capacity(reps);
        let mut index: usize = 0;
        for num_entangle in num_entangle_params_per_layer {
            rotation_indices.push(index);
            index += num_rotation_params_per_layer;
            entangle_indices.push(index);
            index += num_entangle;
        }
        if !skip_final_rotation_layer {
            rotation_indices.push(index);
        }

        ParameterLedger {
            parameter_vector,
            rotation_indices,
            entangle_indices,
            rotations,
            entanglements,
        }
    }

    /// Get the parameters in the rotation or entanglement layer.
    pub(super) fn get_parameters(&self, kind: LayerType, layer: usize) -> LayerParameters<'_> {
        let (mut index, blocks) = match kind {
            LayerType::Rotation => (
                *self
                    .rotation_indices
                    .get(layer)
                    .expect("Out of bounds in rotation_indices."),
                &self.rotations,
            ),
            LayerType::Entangle => (
                *self
                    .entangle_indices
                    .get(layer)
                    .expect("Out of bounds in entangle_indices."),
                &self.entanglements[layer],
            ),
        };

        let mut parameters: LayerParameters = Vec::new();
        for (num_blocks, num_params) in blocks {
            let mut per_block: BlockParameters = Vec::new();
            for _ in 0..*num_blocks {
                let gate_params: Vec<&Param> = (index..index + num_params)
                    .map(|i| {
                        self.parameter_vector
                            .get(i)
                            .expect("Ran out of parameters!")
                    })
                    .collect();
                index += num_params;
                per_block.push(gate_params);
            }
            parameters.push(per_block);
        }

        parameters
    }
}
