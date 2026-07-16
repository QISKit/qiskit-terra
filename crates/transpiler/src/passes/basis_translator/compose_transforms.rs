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

use super::errors::BasisTranslatorError;
use qiskit_circuit::Qubit;

use qiskit_circuit::bit::QuantumRegister;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::CustomOperation;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::SymbolVector;
use qiskit_circuit::parameter_table::ParameterUuid;
use qiskit_circuit::{
    dag_circuit::DAGCircuit,
    operations::{Operation, Param},
};
use qiskit_util::{IndexMap, IndexSet};
use smallvec::SmallVec;

// Custom types
pub type GateIdentifier = (String, u32);
pub type BasisTransformIn = (SmallVec<[Param; 3]>, CircuitData);
pub type BasisTransformOut = (SmallVec<[Param; 3]>, DAGCircuit);

pub(super) fn compose_transforms<'a>(
    basis_transforms: &'a [(GateIdentifier, BasisTransformIn)],
    source_basis: &'a IndexSet<GateIdentifier>,
    source_dag: &'a DAGCircuit,
) -> Result<IndexMap<GateIdentifier, BasisTransformOut>, BasisTranslatorError> {
    let mut gate_param_counts: IndexMap<GateIdentifier, usize> = IndexMap::default();
    get_gates_num_params(source_dag, &mut gate_param_counts);
    let mut mapped_instructions: IndexMap<GateIdentifier, BasisTransformOut> = IndexMap::default();

    for (gate_name, gate_num_qubits) in source_basis.iter().cloned() {
        let num_params = gate_param_counts[&(gate_name.clone(), gate_num_qubits)];
        let placeholder_params: SmallVec<[Param; 3]> =
            SymbolVector::new(gate_name.clone(), num_params)
                .iter()
                .map(|sym| Param::ParameterExpression(ParameterExpression::from_symbol(sym).into()))
                .collect();

        let mut dag = DAGCircuit::new();
        // Create the mock gate and add to the circuit, use Python if necessary.
        let qubits = QuantumRegister::new_owning("q", gate_num_qubits);
        dag.add_qreg(qubits).map_err(|_| {
            BasisTranslatorError::BasisDAGCircuitError(
                "Error while adding register to the circuit".to_string(),
            )
        })?;
        let gate = PackedOperation::from_custom_operation(Box::new(CustomDummy {
            name: gate_name.clone(),
            num_qubits: gate_num_qubits,
            num_params: placeholder_params.len() as u32,
        }));
        let qubits: Vec<Qubit> = (0..dag.num_qubits() as u32).map(Qubit).collect();
        dag.apply_operation_back(
            gate,
            &qubits,
            &[],
            Some(Parameters::Params(placeholder_params.clone())),
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .map_err(|_| {
            BasisTranslatorError::BasisDAGCircuitError(
                "Error applying operation to DAGCircuit".to_string(),
            )
        })?;
        mapped_instructions.insert((gate_name, gate_num_qubits), (placeholder_params, dag));
    }

    for ((gate_name, gate_num_qubits), (equiv_params, equiv)) in basis_transforms {
        for (_, dag) in &mut mapped_instructions.values_mut() {
            let nodes_to_replace = dag
                .op_nodes(true)
                .filter(|(_, op)| {
                    (op.op.num_qubits() == *gate_num_qubits) && (op.op.name() == gate_name.as_str())
                })
                .map(|(node, op)| {
                    (
                        node,
                        op.params_view()
                            .iter()
                            .cloned()
                            .collect::<SmallVec<[Param; 3]>>(),
                    )
                })
                .collect::<Vec<_>>();
            for (node, params) in nodes_to_replace {
                let param_mapping: IndexMap<ParameterUuid, Param> = equiv_params
                    .iter()
                    .map(|x| match x {
                        Param::ParameterExpression(parameter_expression) => {
                            let symbol = parameter_expression.try_to_symbol().unwrap();
                            ParameterUuid::from_symbol(&symbol)
                        }
                        _ => unreachable!("A non parameter-expression has snuck in"),
                    })
                    .zip(params)
                    .collect();
                let mut replacement = equiv.clone();
                replacement
                    .assign_parameters_from_mapping(param_mapping)
                    .map_err(|_| {
                        BasisTranslatorError::BasisCircuitError(
                            "Error during parameter assignment".to_string(),
                        )
                    })?;
                let replace_dag: DAGCircuit =
                    DAGCircuit::from_circuit_data(&replacement, true, None, None, None, None)
                        .map_err(|_| {
                            BasisTranslatorError::BasisDAGCircuitError(
                                "Error converting circuit to dag".to_string(),
                            )
                        })?;
                dag.substitute_node_with_dag(node, &replace_dag, None, None, None, None)
                    .map_err(|_| {
                        BasisTranslatorError::BasisDAGCircuitError(
                            "Error during node substitution with DAG.".to_string(),
                        )
                    })?;
            }
        }
    }
    Ok(mapped_instructions)
}

/// `DAGCircuit` variant.
///
/// Gets the identifier of a gate instance (name, number of qubits) mapped to the
/// number of parameters it contains currently.
fn get_gates_num_params(dag: &DAGCircuit, example_gates: &mut IndexMap<GateIdentifier, usize>) {
    for (_, inst) in dag.op_nodes(true) {
        if let Some(control_flow) = dag.try_view_control_flow(inst) {
            example_gates.insert(
                (inst.op.name().to_string(), inst.op.num_qubits()),
                inst.op.num_params() as usize,
            );
            for block in control_flow.blocks() {
                get_gates_num_params(block, example_gates);
            }
        } else {
            example_gates.insert(
                (inst.op.name().to_string(), inst.op.num_qubits()),
                inst.params_view().len(),
            );
        }
    }
}

/// Dummy gate used exclusively for the Basis Translator as a placeholder.
#[derive(Debug, Clone, PartialEq)]
struct CustomDummy {
    name: String,
    num_qubits: u32,
    num_params: u32,
}

impl Operation for CustomDummy {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        self.num_params
    }

    fn directive(&self) -> bool {
        false
    }
}

impl CustomOperation for CustomDummy {
    fn is_unitary(&self) -> bool {
        true
    }
}
