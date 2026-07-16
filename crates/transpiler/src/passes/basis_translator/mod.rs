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

use compose_transforms::BasisTransformIn;
use compose_transforms::BasisTransformOut;
use compose_transforms::GateIdentifier;

use basis_search::basis_search;
use compose_transforms::compose_transforms;
use errors::BasisTranslatorError;
use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use qiskit_util::{IndexMap, IndexSet};
use rustworkx_core::petgraph::algo::toposort;
use rustworkx_core::petgraph::graph::NodeIndex;

mod basis_search;
mod compose_transforms;
mod errors;

use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::parameter::parameter_expression::ParameterError;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;
use qiskit_circuit::parameter::symbol_expr::SymbolExpr;
use qiskit_circuit::parameter::symbol_expr::Value;
use qiskit_circuit::{
    BlocksMode, VarsMode,
    dag_circuit::{DAGCircuit, DAGCircuitBuilder, NodeType},
    operations::{Operation, OperationRef, Param, PauliBased, PythonOperation},
};
use qiskit_circuit::{Clbit, PhysicalQubit, Qubit};
use smallvec::SmallVec;

use crate::equivalence::EquivalenceLibrary;
use crate::target::Qargs;
use crate::target::QargsRef;
use crate::target::Target;

type AhashIndexMap<K, V> = IndexMap<K, V>;
type AhashIndexSet<O> = IndexSet<O>;
type InstMap = AhashIndexMap<GateIdentifier, BasisTransformOut>;
type ExtraInstructionMap<'a> = AhashIndexMap<&'a PhysicalQargs, InstMap>;
type PhysicalQargs = SmallVec<[PhysicalQubit; 2]>;

#[pyfunction(name = "base_run", signature = (dag, equiv_lib, min_qubits, target=None, target_basis=None))]
fn py_run_basis_translator(
    dag: &DAGCircuit,
    equiv_lib: &mut EquivalenceLibrary,
    min_qubits: usize,
    target: Option<&Target>,
    target_basis: Option<HashSet<String>>,
) -> PyResult<Option<DAGCircuit>> {
    let target_basis_ref: Option<HashSet<&str>> = target_basis
        .as_ref()
        .map(|set| set.iter().map(|obj| obj.as_str()).collect());
    run_basis_translator(dag, equiv_lib, min_qubits, target, target_basis_ref).map_err(|e| e.into())
}

pub fn run_basis_translator(
    dag: &DAGCircuit,
    equiv_lib: &mut EquivalenceLibrary,
    min_qubits: usize,
    target: Option<&Target>,
    target_basis: Option<HashSet<&str>>,
) -> Result<Option<DAGCircuit>, BasisTranslatorError> {
    if target_basis.is_none() && target.is_none() {
        return Ok(None);
    }

    let (non_global_operations, qargs_with_non_global_operation): (
        Option<AhashIndexSet<&str>>,
        AhashIndexMap<Qargs, AhashIndexSet<&str>>,
    ) = if let Some(target) = target {
        let mut qargs_mapping: AhashIndexMap<Qargs, AhashIndexSet<&str>> = AhashIndexMap::default();
        let global_set: AhashIndexSet<&str> =
            AhashIndexSet::from_iter(target.get_non_global_operation_names(false));
        for name in global_set.iter() {
            for qarg in target[name].keys().cloned() {
                qargs_mapping
                    .entry(qarg)
                    .and_modify(|val| {
                        val.insert(name);
                    })
                    .or_insert(AhashIndexSet::from_iter([*name]));
            }
        }
        (Some(global_set), qargs_mapping)
    } else {
        (None, AhashIndexMap::default())
    };

    let basic_instrs: AhashIndexSet<&str>;
    let mut source_basis: AhashIndexSet<GateIdentifier> = AhashIndexSet::default();
    let mut new_target_basis: AhashIndexSet<&str>;
    let mut qargs_local_source_basis: AhashIndexMap<PhysicalQargs, AhashIndexSet<GateIdentifier>> =
        AhashIndexMap::default();
    if let Some(target) = target.as_ref() {
        basic_instrs = ["barrier", "snapshot", "store"].into_iter().collect();
        let non_global_str: AhashIndexSet<&str> =
            if let Some(operations) = non_global_operations.as_ref() {
                operations.clone()
            } else {
                AhashIndexSet::default()
            };
        let target_keys = target.keys().collect::<AhashIndexSet<_>>();
        new_target_basis = target_keys.difference(&non_global_str).copied().collect();
        extract_basis_target(
            dag,
            &mut source_basis,
            &mut qargs_local_source_basis,
            min_qubits,
            &qargs_with_non_global_operation,
            None,
        );
    } else {
        basic_instrs = ["measure", "reset", "barrier", "snapshot", "delay", "store"]
            .into_iter()
            .collect();
        source_basis = extract_basis(dag, min_qubits);
        new_target_basis = target_basis
            .as_ref()
            .unwrap()
            .into_iter()
            .copied()
            .collect();
    }
    new_target_basis = new_target_basis.union(&basic_instrs).copied().collect();
    // If the source basis is a subset of the target basis and we have no circuit
    // instructions on qargs that have non-global operations there is nothing to
    // translate and we can exit early.
    let source_basis_names: AhashIndexSet<&str> =
        source_basis.iter().map(|x| x.0.as_str()).collect();
    if source_basis_names.is_subset(&new_target_basis) && qargs_local_source_basis.is_empty() {
        return Ok(None);
    }
    let basis_transforms = basis_search(equiv_lib, &source_basis, &new_target_basis);
    let mut qarg_local_basis_transforms: AhashIndexMap<
        &PhysicalQargs,
        Vec<(GateIdentifier, BasisTransformIn)>,
    > = AhashIndexMap::default();
    for (qargs, local_source_basis) in qargs_local_source_basis.iter() {
        // For any multiqubit operation that contains a subset of qubits that
        // has a non-local operation, include that non-local operation in the
        // search. This matches with the check we did above to include those
        // subset non-local operations in the check here.
        let mut expanded_target = new_target_basis.clone();
        // Qargs are always guaranteed to be concrete based on `extract_basis_target`.
        if qargs.len() > 1 {
            let qarg_as_set: AhashIndexSet<PhysicalQubit> =
                AhashIndexSet::from_iter(qargs.iter().copied());
            for (non_local_qarg, local_basis) in qargs_with_non_global_operation.iter() {
                if let Qargs::Concrete(non_local_qarg) = non_local_qarg {
                    let non_local_qarg_as_set: AhashIndexSet<PhysicalQubit> =
                        AhashIndexSet::from_iter(non_local_qarg.iter().copied());
                    if qarg_as_set.is_superset(&non_local_qarg_as_set) {
                        expanded_target = expanded_target.union(local_basis).cloned().collect();
                    }
                }
            }
        } else {
            expanded_target = expanded_target
                .union(&qargs_with_non_global_operation[&QargsRef::Concrete(qargs)])
                .cloned()
                .collect();
        }
        let local_basis_transforms = basis_search(equiv_lib, local_source_basis, &expanded_target);
        if let Some(local_basis_transforms) = local_basis_transforms {
            qarg_local_basis_transforms.insert(qargs, local_basis_transforms);
        } else {
            return Err(BasisTranslatorError::TargetMissingEquivalence {
                basis: format!("{:?}", local_source_basis),
                expanded: format!("{:?}", expanded_target),
            });
        }
    }

    let Some(basis_transforms) = basis_transforms else {
        return Err(BasisTranslatorError::TargetMissingEquivalence {
            basis: format!("{:?}", source_basis),
            expanded: format!("{:?}", new_target_basis),
        });
    };

    let instr_map: InstMap = compose_transforms(&basis_transforms, &source_basis, dag)?;
    let extra_inst_map: ExtraInstructionMap = qarg_local_basis_transforms
        .iter()
        .map(|(qarg, transform)| -> Result<_, BasisTranslatorError> {
            Ok((
                *qarg,
                compose_transforms(transform, &qargs_local_source_basis[*qarg], dag)?,
            ))
        })
        .collect::<Result<_, BasisTranslatorError>>()?;

    let out_dag = apply_translation(
        dag,
        &new_target_basis,
        &instr_map,
        &extra_inst_map,
        min_qubits,
        &qargs_with_non_global_operation,
        None,
    )?;
    Ok(Some(out_dag))
}

/// Method that extracts all gate instances identifiers from a DAGCircuit.
fn extract_basis(circuit: &DAGCircuit, min_qubits: usize) -> AhashIndexSet<GateIdentifier> {
    let mut basis = AhashIndexSet::default();
    // Recurse for DAGCircuit
    fn recurse_dag(
        circuit: &DAGCircuit,
        basis: &mut AhashIndexSet<GateIdentifier>,
        min_qubits: usize,
    ) {
        for (_, operation) in circuit.op_nodes(true) {
            if circuit.get_qargs(operation.qubits).len() >= min_qubits {
                basis.insert((operation.op.name().to_string(), operation.op.num_qubits()));
            }
            if let Some(control_flow) = circuit.try_view_control_flow(operation) {
                for block in control_flow.blocks() {
                    recurse_dag(block, basis, min_qubits);
                }
            }
        }
    }

    recurse_dag(circuit, &mut basis, min_qubits);
    basis
}

/// Method that extracts a mapping of all the qargs in the local_source basis
/// obtained from the [Target], to all gate instances identifiers from a DAGCircuit.
/// When dealing with `ControlFlowOp` instances the function will perform a recursion call
/// to a variant design to handle instances of `QuantumCircuit`.
fn extract_basis_target(
    dag: &DAGCircuit,
    source_basis: &mut AhashIndexSet<GateIdentifier>,
    qargs_local_source_basis: &mut AhashIndexMap<PhysicalQargs, AhashIndexSet<GateIdentifier>>,
    min_qubits: usize,
    qargs_with_non_global_operation: &AhashIndexMap<Qargs, AhashIndexSet<&str>>,
    qarg_mapping: Option<&HashMap<Qubit, Qubit>>,
) {
    let qarg_mapping = |q: &Qubit| qarg_mapping.map(|map| map[q]).unwrap_or(*q);
    for (_, node) in dag.op_nodes(true) {
        let qargs: &[Qubit] = dag.get_qargs(node.qubits);
        if qargs.len() < min_qubits {
            continue;
        }
        // Treat the instruction as on an incomplete basis if the qargs are in the
        // qargs_with_non_global_operation dictionary or if any of the qubits in qargs
        // are a superset for a non-local operation. For example, if the qargs
        // are (0, 1) and that's a global (ie no non-local operations on (0, 1)
        // operation but there is a non-local operation on (1,) we need to
        // do an extra non-local search for this op to ensure we include any
        // single qubit operation for (1,) as valid. This pattern also holds
        // true for > 2q ops too (so for 4q operations we need to check for 3q, 2q,
        // and 1q operations in the same manner)
        let physical_qargs: PhysicalQargs = qargs
            .iter()
            .map(|x| PhysicalQubit(qarg_mapping(x).0))
            .collect();
        let physical_qargs_as_set: AhashIndexSet<PhysicalQubit> =
            AhashIndexSet::from_iter(physical_qargs.iter().copied());
        let physical_qargs: Qargs = physical_qargs.into();
        if qargs_with_non_global_operation.contains_key(&physical_qargs)
            || qargs_with_non_global_operation
                .keys()
                .filter_map(|qargs| {
                    if let QargsRef::Concrete(qargs) = qargs.as_ref() {
                        Some(qargs)
                    } else {
                        None
                    }
                })
                .any(|incomplete_qargs| {
                    let incomplete_qargs: AhashIndexSet<PhysicalQubit> =
                        AhashIndexSet::from_iter(incomplete_qargs.iter().copied());
                    physical_qargs_as_set.is_superset(&incomplete_qargs)
                })
        {
            let qargs_from_set: PhysicalQargs = physical_qargs_as_set.into_iter().collect();
            qargs_local_source_basis
                .entry(qargs_from_set)
                .and_modify(|set| {
                    set.insert((node.op.name().to_string(), node.op.num_qubits()));
                })
                .or_insert(AhashIndexSet::from_iter([(
                    node.op.name().to_string(),
                    node.op.num_qubits(),
                )]));
        } else {
            source_basis.insert((node.op.name().to_string(), node.op.num_qubits()));
        }
        if let Some(control_flow) = dag.try_view_control_flow(node) {
            for block in control_flow.blocks() {
                // Generate a mapping with the absolute indices and the local ones.
                let qarg_mapping: HashMap<Qubit, Qubit> = dag
                    .qargs_interner()
                    .get(node.qubits)
                    .iter()
                    .copied()
                    .zip((0..(block.num_qubits() as u32)).map(Qubit))
                    .map(|(k, v)| (v, qarg_mapping(&k)))
                    .collect();
                extract_basis_target(
                    block,
                    source_basis,
                    qargs_local_source_basis,
                    min_qubits,
                    qargs_with_non_global_operation,
                    Some(&qarg_mapping),
                );
            }
        }
    }
}

fn apply_translation(
    dag: &DAGCircuit,
    target_basis: &AhashIndexSet<&str>,
    instr_map: &InstMap,
    extra_inst_map: &ExtraInstructionMap,
    min_qubits: usize,
    qargs_with_non_global_operation: &AhashIndexMap<Qargs, AhashIndexSet<&str>>,
    qarg_mapping: Option<&HashMap<Qubit, Qubit>>,
) -> Result<DAGCircuit, BasisTranslatorError> {
    let rebuilder_callback = |out_dag_builder: &mut DAGCircuitBuilder,
                              node_obj: &PackedInstruction,
                              _node: NodeIndex| {
        let node_qarg = dag.get_qargs(node_obj.qubits);
        let node_carg = dag.get_cargs(node_obj.clbits);
        let qubit_set: AhashIndexSet<Qubit> = AhashIndexSet::from_iter(node_qarg.iter().copied());
        if target_basis.contains(node_obj.op.name()) || node_qarg.len() < min_qubits {
            if let Some(control_flow) = dag.try_view_control_flow(node_obj) {
                let mut flow_blocks = vec![];
                for dag_block in control_flow.blocks() {
                    let updated_dag: DAGCircuit;
                    // Generate a mapping between the absolute and local qubit indices
                    // which will be used to correctly map the operation onto the dag.
                    let qarg_mapping: HashMap<Qubit, Qubit> =
                        if let Some(qarg_mapping) = qarg_mapping {
                            dag.qargs_interner()
                                .get(node_obj.qubits)
                                .iter()
                                .zip((0..(dag_block.num_qubits() as u32)).map(Qubit))
                                .map(|(k, v)| (v, qarg_mapping[k]))
                                .collect()
                        } else {
                            dag.qargs_interner()
                                .get(node_obj.qubits)
                                .iter()
                                .zip((0..(dag_block.num_qubits() as u32)).map(Qubit))
                                .map(|(k, v)| (v, *k))
                                .collect()
                        };
                    updated_dag = apply_translation(
                        dag_block,
                        target_basis,
                        instr_map,
                        extra_inst_map,
                        min_qubits,
                        qargs_with_non_global_operation,
                        Some(&qarg_mapping),
                    )?;
                    flow_blocks.push(out_dag_builder.add_block(updated_dag));
                }
                let new_instr = PackedInstruction::from_control_flow(
                    node_obj.op.control_flow().clone(),
                    flow_blocks,
                    node_obj.qubits,
                    node_obj.clbits,
                    node_obj.label.as_deref().cloned(),
                );
                out_dag_builder.push_back(new_instr).map_err(|_| {
                    BasisTranslatorError::BasisDAGCircuitError(
                        "Error applying operation to DAGCircuit".to_string(),
                    )
                })?;
            } else {
                out_dag_builder
                    .apply_operation_back(
                        node_obj.op.clone(),
                        node_qarg,
                        node_carg,
                        node_obj.params.as_ref().map(|x| *x.clone()),
                        node_obj.label.as_deref().cloned(),
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )
                    .map_err(|_| {
                        BasisTranslatorError::BasisDAGCircuitError(
                            "Error applying operation to DAGCircuit".to_string(),
                        )
                    })?;
            }
            return Ok(());
        }
        // Map to the absolute indices when provided to avoid mistakenly tracking
        // the operation as global.
        let node_qarg_as_physical: Qargs = if let Some(qarg_mapping) = qarg_mapping {
            node_qarg
                .iter()
                .map(|x| PhysicalQubit(qarg_mapping[x].0))
                .collect()
        } else {
            node_qarg.iter().map(|x| PhysicalQubit(x.0)).collect()
        };
        if qargs_with_non_global_operation.contains_key(&node_qarg_as_physical)
            && qargs_with_non_global_operation[&node_qarg_as_physical].contains(node_obj.op.name())
        {
            out_dag_builder
                .apply_operation_back(
                    node_obj.op.clone(),
                    node_qarg,
                    node_carg,
                    node_obj.params.as_ref().map(|x| *x.clone()),
                    node_obj.label.as_deref().cloned(),
                    #[cfg(feature = "cache_pygates")]
                    None,
                )
                .map_err(|_| {
                    BasisTranslatorError::BasisDAGCircuitError(
                        "Error applying operation to DAGCircuit".to_string(),
                    )
                })?;
            return Ok(());
        }

        // Map the unique qargs with the absolute indices as well
        let unique_qargs: PhysicalQargs = if let Some(qarg_mapping) = qarg_mapping {
            qubit_set
                .iter()
                .map(|x| PhysicalQubit(qarg_mapping[x].0))
                .collect()
        } else {
            qubit_set.iter().map(|x| PhysicalQubit(x.0)).collect()
        };
        if extra_inst_map.contains_key(&unique_qargs) {
            replace_node(
                out_dag_builder,
                node_obj.clone(),
                &extra_inst_map[&unique_qargs],
            )?;
        } else if instr_map
            .contains_key(&(node_obj.op.name().to_string(), node_obj.op.num_qubits()))
        {
            replace_node(out_dag_builder, node_obj.clone(), instr_map)?;
        } else {
            return Err(BasisTranslatorError::ApplyTranslationMappingError(
                node_obj.op.name().to_string(),
            ));
        }
        Ok(())
    };
    dag.rebuild_dag_with(rebuilder_callback, VarsMode::Alike, BlocksMode::Keep)
}

fn replace_node(
    dag: &mut DAGCircuitBuilder,
    node: PackedInstruction,
    instr_map: &AhashIndexMap<GateIdentifier, (SmallVec<[Param; 3]>, DAGCircuit)>,
) -> Result<(), BasisTranslatorError> {
    let (target_params, target_dag) =
        &instr_map[&(node.op.name().to_string(), node.op.num_qubits())];
    let params_view = node.params_view();

    if params_view.len() != target_params.len() {
        return Err(BasisTranslatorError::ReplaceNodeParamMismatch {
            node_params: format!("{:?}", params_view),
            node_name: node.op.name().to_string(),
            target_params: format!("{:?}", target_params),
            target_dag: format!("{:?}", target_dag),
        });
    }
    if params_view.is_empty() {
        for inner_index in toposort(target_dag.dag(), None).expect("DAGCircuit can't have a cycle")
        {
            let NodeType::Operation(ref inner_node) = target_dag[inner_index] else {
                continue;
            };
            let old_qargs = dag.qargs_interner().get(node.qubits);
            let old_cargs = dag.cargs_interner().get(node.clbits);
            let new_qubits: Vec<Qubit> = target_dag
                .get_qargs(inner_node.qubits)
                .iter()
                .map(|qubit| old_qargs[qubit.0 as usize])
                .collect();
            let new_clbits: Vec<Clbit> = target_dag
                .get_cargs(inner_node.clbits)
                .iter()
                .map(|clbit| old_cargs[clbit.0 as usize])
                .collect();
            let new_op: PackedOperation = match inner_node.op.view() {
                OperationRef::PyCustom(inst) => Python::attach(|py| inst.py_copy(py))
                    .map_err(|err| BasisTranslatorError::BasisDAGCircuitError(err.to_string()))?
                    .into(),
                OperationRef::ControlFlow(_) => panic!("control flow should not be present here"),
                OperationRef::StandardGate(gate) => gate.into(),
                OperationRef::StandardInstruction(instruction) => instruction.into(),
                OperationRef::Unitary(unitary) => unitary.clone().into(),
                OperationRef::PauliProductMeasurement(ppm) => {
                    PauliBased::PauliProductMeasurement(ppm.clone()).into()
                }
                OperationRef::PauliProductRotation(rotation) => {
                    PauliBased::PauliProductRotation(rotation.clone()).into()
                }
                OperationRef::CustomOperation(_) => inner_node.op.clone(),
            };
            let new_params: Option<Parameters<_>> = inner_node.params.as_deref().cloned();
            dag.apply_operation_back(
                new_op,
                &new_qubits,
                &new_clbits,
                new_params,
                node.label.as_deref().cloned(),
                #[cfg(feature = "cache_pygates")]
                None,
            )
            .map_err(|_| {
                BasisTranslatorError::BasisDAGCircuitError(
                    "Error applying operation to DAGCircuit".to_string(),
                )
            })?;
        }
        dag.add_global_phase(target_dag.global_phase())
            .map_err(|_| {
                BasisTranslatorError::BasisDAGCircuitError(
                    "Error while adding a new global phase".to_string(),
                )
            })?;
    } else {
        let parameter_map: HashMap<Symbol, Param> = HashMap::from_iter(
            target_params
                .iter()
                .zip(params_view.iter().cloned())
                .filter_map(|(key, val)| match key {
                    Param::ParameterExpression(param) => param
                        .try_to_symbol()
                        .ok()
                        .map(|param| (param.clone(), val.clone())),
                    _ => None,
                }),
        );
        for inner_index in toposort(target_dag.dag(), None).expect("DAGCircuit can't have a cycle")
        {
            let NodeType::Operation(ref inner_node) = target_dag[inner_index] else {
                continue;
            };
            let old_qargs = dag.qargs_interner().get(node.qubits);
            let old_cargs = dag.cargs_interner().get(node.clbits);
            let new_qubits: Vec<Qubit> = target_dag
                .get_qargs(inner_node.qubits)
                .iter()
                .map(|qubit| old_qargs[qubit.0 as usize])
                .collect();
            let new_clbits: Vec<Clbit> = target_dag
                .get_cargs(inner_node.clbits)
                .iter()
                .map(|clbit| old_cargs[clbit.0 as usize])
                .collect();
            let new_op: PackedOperation = match inner_node.op.view() {
                OperationRef::ControlFlow(cf) => cf.clone().into(),
                OperationRef::PyCustom(inst) => Python::attach(|py| inst.py_copy(py))
                    .map_err(|err| BasisTranslatorError::BasisDAGCircuitError(err.to_string()))?
                    .into(),
                OperationRef::StandardGate(gate) => gate.into(),
                OperationRef::StandardInstruction(instruction) => instruction.into(),
                OperationRef::Unitary(unitary) => unitary.clone().into(),
                OperationRef::PauliProductMeasurement(ppm) => {
                    PauliBased::PauliProductMeasurement(ppm.clone()).into()
                }
                OperationRef::PauliProductRotation(rotation) => {
                    PauliBased::PauliProductRotation(rotation.clone()).into()
                }
                OperationRef::CustomOperation(_) => inner_node.op.clone(),
            };

            let mut new_params: Option<Parameters<_>> = inner_node.params.as_deref().cloned();
            if let Some(Parameters::Params(inner_node_params)) = inner_node.params.as_deref() {
                if inner_node_params
                    .iter()
                    .any(|param| matches!(param, Param::ParameterExpression(_)))
                {
                    let new_params_inner: SmallVec<[Param; 3]> = inner_node_params
                        .iter()
                        .map(|param| match param {
                            Param::ParameterExpression(parameter_expression) => {
                                param_assignment_expr(parameter_expression, &parameter_map, true)
                            }
                            _ => Ok(param.clone()),
                        })
                        .collect::<Result<_, BasisTranslatorError>>()?;
                    if let OperationRef::PyCustom(inst) = new_op.view() {
                        // TODO: Remove this.
                        // Acquire the gil if the operation is not native to set the operation parameters in
                        // Python.
                        Python::attach(|py| {
                            inst.ob
                                .bind(py)
                                .setattr("params", new_params_inner.clone())
                                .map_err(|err| {
                                    BasisTranslatorError::BasisDAGCircuitError(err.to_string())
                                })
                        })?;
                    }
                    new_params = Some(Parameters::Params(new_params_inner));
                }
            }
            dag.apply_operation_back(
                new_op,
                &new_qubits,
                &new_clbits,
                new_params,
                inner_node.label.as_deref().cloned(),
                #[cfg(feature = "cache_pygates")]
                None,
            )
            .map_err(|err| BasisTranslatorError::BasisDAGCircuitError(err.to_string()))?;
        }

        match target_dag.global_phase() {
            Param::ParameterExpression(expr) => {
                let param = param_assignment_expr(expr, &parameter_map, false)?;
                dag.add_global_phase(&param)
                    .map_err(|e| BasisTranslatorError::BasisDAGCircuitError(e.to_string()))
            }
            Param::Float(_) => dag
                .add_global_phase(target_dag.global_phase())
                .map_err(|e| BasisTranslatorError::BasisDAGCircuitError(e.to_string())),
            Param::Obj(_) => Ok(()),
        }?
    }

    Ok(())
}

fn param_expr_assignment(
    param_obj: &ParameterExpression,
    parameter_map: &HashMap<Symbol, Param>,
) -> Result<ParameterExpression, ParameterError> {
    let mut subs_map: HashMap<Symbol, ParameterExpression> = HashMap::new();
    let mut bind_map: HashMap<&Symbol, Value> = HashMap::new();
    for key in param_obj.iter_symbols() {
        match &parameter_map[key].clone() {
            Param::ParameterExpression(val) => {
                subs_map.insert(key.clone(), val.as_ref().clone());
            }
            Param::Float(val) => {
                bind_map.insert(key, Value::Real(*val));
            }
            Param::Obj(val) => {
                let val = Python::attach(|py| val.extract::<Value>(py))
                    .map_err(|_| ParameterError::InvalidValue)?;
                bind_map.insert(key, val);
            }
        }
    }
    // Apply substitution and binding in that order.
    let mut new_value: ParameterExpression = param_obj.subs(&subs_map, true)?;
    new_value = new_value.bind(&bind_map, true)?;
    Ok(new_value)
}

fn param_assignment_expr(
    param: &ParameterExpression,
    parameter_map: &HashMap<Symbol, Param>,
    allow_complex: bool,
) -> Result<Param, BasisTranslatorError> {
    let new_value = param_expr_assignment(param, parameter_map)
        .map_err(BasisTranslatorError::BasisParameterError)?;
    match (new_value.try_to_value(true), allow_complex) {
        (Ok(Value::Complex(parsed)), false) => Err(
            BasisTranslatorError::ReplaceNodeGlobalPhaseComplex(parsed.to_string()),
        ),
        (Ok(Value::Real(num)), _) => Ok(Param::Float(num)),
        (Ok(parsed), _) => Ok(Param::ParameterExpression(
            ParameterExpression::from_symbol_expr(SymbolExpr::Value(parsed)).into(),
        )),
        (Err(_), _) => Ok(Param::ParameterExpression(new_value.into())),
    }
}

pub fn basis_translator_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_basis_translator))?;
    Ok(())
}
