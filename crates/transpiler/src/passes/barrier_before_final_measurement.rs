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

use pyo3::prelude::*;
use rayon::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::Qubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGError, NodeType};
use qiskit_circuit::operations::{OperationRef, StandardInstruction};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};

const PARALLEL_THRESHOLD: usize = 150;

#[pyfunction]
#[pyo3(name = "barrier_before_final_measurements", signature=(dag, label=None))]
pub fn py_run_barrier_before_final_measurements(
    dag: &mut DAGCircuit,
    label: Option<String>,
) -> PyResult<()> {
    run_barrier_before_final_measurements(dag, label).map_err(Into::into)
}

pub fn run_barrier_before_final_measurements(
    dag: &mut DAGCircuit,
    label: Option<String>,
) -> Result<(), DAGError> {
    // Get a list of the node indices which are final measurement or barriers that are ancestors
    // of a given qubit's output node.
    let find_final_nodes = |[_in_index, out_index]: &[NodeIndex; 2]| -> Vec<NodeIndex> {
        // Next nodes is the stack of parent nodes to investigate. It starts with any predecessors
        // of a qubit's output node that are Barrier or Measure
        let mut next_nodes: Vec<NodeIndex> = dag
            .quantum_predecessors(*out_index)
            .filter(|index| {
                let node = &dag[*index];
                match node {
                    NodeType::Operation(inst) => {
                        if let OperationRef::StandardInstruction(op) = inst.op.view() {
                            if matches!(
                                op,
                                StandardInstruction::Measure | StandardInstruction::Barrier(_)
                            ) {
                                dag.bfs_successors(*index).all(|(_, child_successors)| {
                                    child_successors.iter().all(|suc| match &dag[*suc] {
                                        NodeType::Operation(suc_inst) => match suc_inst.op.view() {
                                            OperationRef::StandardInstruction(suc_op) => {
                                                matches!(
                                                    suc_op,
                                                    StandardInstruction::Measure
                                                        | StandardInstruction::Barrier(_)
                                                )
                                            }
                                            _ => false,
                                        },
                                        _ => true,
                                    })
                                })
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    _ => false,
                }
            })
            .collect();
        let mut nodes: Vec<NodeIndex> = Vec::new();
        // Reverse traverse the dag from next nodes until we encounter no more barriers or measures
        while let Some(node_index) = next_nodes.pop() {
            // If node on the stack is a barrier or measure we can add it to the output list
            if node_index != *out_index
                && dag.bfs_successors(node_index).all(|(_, child_successors)| {
                    child_successors.iter().all(|suc| match &dag[*suc] {
                        NodeType::Operation(suc_inst) => match suc_inst.op.view() {
                            OperationRef::StandardInstruction(suc_op) => matches!(
                                suc_op,
                                StandardInstruction::Measure | StandardInstruction::Barrier(_)
                            ),
                            _ => false,
                        },
                        _ => true,
                    })
                })
            {
                nodes.push(node_index);
            }
            // For this node if any parent nodes are barrier or measure add those to the stack
            for pred in dag.quantum_predecessors(node_index) {
                match &dag[pred] {
                    NodeType::Operation(inst) => {
                        if let OperationRef::StandardInstruction(op) = inst.op.view() {
                            if matches!(
                                op,
                                StandardInstruction::Measure | StandardInstruction::Barrier(_)
                            ) {
                                next_nodes.push(pred)
                            }
                        }
                    }
                    _ => continue,
                }
            }
        }
        nodes.reverse();
        nodes
    };

    let final_ops: Vec<NodeIndex> =
        if dag.num_qubits() >= PARALLEL_THRESHOLD && ::qiskit_util::getenv_use_multiple_threads() {
            dag.qubit_io_map()
                .par_iter()
                .flat_map(find_final_nodes)
                .collect()
        } else {
            dag.qubit_io_map()
                .iter()
                .flat_map(find_final_nodes)
                .collect()
        };

    if final_ops.is_empty() {
        return Ok(());
    }
    let final_packed_ops: Vec<PackedInstruction> = final_ops
        .into_iter()
        .filter_map(|node| match dag.dag().node_weight(node) {
            Some(weight) => {
                let NodeType::Operation(_) = weight else {
                    return None;
                };
                let res = dag.remove_op_node(node);
                Some(res)
            }
            None => None,
        })
        .collect();
    let qargs: Vec<Qubit> = (0..dag.num_qubits() as u32).map(Qubit).collect();
    dag.apply_operation_back(
        PackedOperation::from_standard_instruction(StandardInstruction::Barrier(
            dag.num_qubits() as u32
        )),
        qargs.as_slice(),
        &[],
        None,
        label,
        #[cfg(feature = "cache_pygates")]
        None,
    )?;
    for inst in final_packed_ops {
        dag.push_back(inst)?;
    }
    Ok(())
}

pub fn barrier_before_final_measurements_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_barrier_before_final_measurements))?;
    Ok(())
}
