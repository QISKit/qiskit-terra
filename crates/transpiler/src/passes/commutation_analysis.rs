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

use pyo3::prelude::PyModule;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::{Bound, PyResult, Python, pyfunction, wrap_pyfunction};

use qiskit_util::IndexMap;
use rayon::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use crate::commutation_checker::{CommutationChecker, CommutationError};
use qiskit_circuit::Qubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};

// Custom types to store the commutation sets and node indices,
// see the docstring below for more information.
type CommutationSet = Vec<Vec<Vec<NodeIndex>>>;
type NodeIndices = Vec<IndexMap<NodeIndex, usize>>;

// the maximum number of qubits we check commutativity for
const MAX_NUM_QUBITS: u32 = 3;

/// Compute the commutation sets for a given DAG.
///
/// We return two HashMaps:
///  * {wire: commutation_sets}: For each wire, we keep a vector of index sets, where each index
///    set contains mutually commuting nodes. Note that these include the input and output nodes
///    which do not commute with anything.
///  * {(node, wire): index}: For each (node, wire) pair we store the index indicating in which
///    commutation set the node appears on a given wire.
///
/// For example, if we have a circuit
///
///     |0> -- X -- SX -- Z (out)
///      0     2    3     4   1   <-- node indices including input (0) and output (1) nodes
///
/// Then we would have
///
///     commutation_set = {0: [[0], [2, 3], [4], [1]]}
///     node_indices = {(0, 0): 0, (1, 0): 3, (2, 0): 1, (3, 0): 1, (4, 0): 2}
///
pub fn analyze_commutations(
    dag: &mut DAGCircuit,
    commutation_checker: &CommutationChecker,
    approximation_degree: f64,
) -> Result<(CommutationSet, NodeIndices), CommutationError> {
    let mut commutation_set = vec![vec![]; dag.num_qubits()];
    let mut node_indices: NodeIndices = vec![IndexMap::default(); dag.num_qubits()];

    let evaluate_qubit = |qubit: usize,
                          commutation_entry: &mut Vec<Vec<NodeIndex>>,
                          node_indices: &mut IndexMap<NodeIndex, usize>|
     -> Result<(), CommutationError> {
        let wire = Wire::Qubit(Qubit(qubit as u32));

        for current_gate_idx in dag.nodes_on_wire(wire) {
            // create a new inner vec if the entry is empty
            if commutation_entry.is_empty() {
                commutation_entry.push(vec![current_gate_idx]);
            }

            // we can unwrap as we know the commutation entry has at least one element
            let last = commutation_entry.last_mut().unwrap();

            // if the current gate index is not in the set, check whether it commutes with
            // the previous nodes -- if yes, add it to the commutation set
            if !last.contains(&current_gate_idx) {
                let mut all_commute = true;

                for prev_gate_idx in last.iter() {
                    // if the node is an input/output node, they do not commute, so we only
                    // continue if the nodes are operation nodes
                    if let (NodeType::Operation(packed_inst0), NodeType::Operation(packed_inst1)) =
                        (&dag[current_gate_idx], &dag[*prev_gate_idx])
                    {
                        let op1 = packed_inst0.op.view();
                        let op2 = packed_inst1.op.view();

                        if packed_inst0.op.try_control_flow().is_some()
                            || packed_inst1.op.try_control_flow().is_some()
                        {
                            all_commute = false;
                            break;
                        }

                        let qargs1 = dag.get_qargs(packed_inst0.qubits);
                        let qargs2 = dag.get_qargs(packed_inst1.qubits);
                        let cargs1 = dag.get_cargs(packed_inst0.clbits);
                        let cargs2 = dag.get_cargs(packed_inst1.clbits);

                        all_commute = commutation_checker.commute(
                            &op1,
                            packed_inst0.params.as_deref(),
                            qargs1,
                            cargs1,
                            &op2,
                            packed_inst1.params.as_deref(),
                            qargs2,
                            cargs2,
                            None,
                            MAX_NUM_QUBITS,
                            approximation_degree,
                        )?;
                        if !all_commute {
                            break;
                        }
                    } else {
                        all_commute = false;
                        break;
                    }
                }

                if all_commute {
                    // all commute, add to current list
                    last.push(current_gate_idx);
                } else {
                    // does not commute, create new list
                    commutation_entry.push(vec![current_gate_idx]);
                }
            }
            node_indices.insert(current_gate_idx, commutation_entry.len() - 1);
        }
        Ok(())
    };

    if qiskit_util::getenv_use_multiple_threads() {
        commutation_set
            .par_iter_mut()
            .zip(node_indices.par_iter_mut())
            .enumerate()
            .map(
                |(qubit, (local_commutation_set, local_indices))| -> Result<(), CommutationError> {
                    evaluate_qubit(qubit, local_commutation_set, local_indices)?;
                    Ok(())
                },
            )
            .collect::<Result<(), CommutationError>>()?;
        Ok((commutation_set, node_indices))
    } else {
        for qubit in 0..dag.num_qubits() {
            evaluate_qubit(qubit, &mut commutation_set[qubit], &mut node_indices[qubit])?;
        }
        Ok((commutation_set, node_indices))
    }
}

#[pyfunction]
#[pyo3(name = "analyze_commutations", signature = (dag, commutation_checker, approximation_degree=1.))]
pub fn py_analyze_commutations(
    py: Python,
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
    approximation_degree: f64,
) -> PyResult<Py<PyDict>> {
    // This returns two HashMaps:
    //   * The commuting nodes per wire: {wire: [commuting_nodes_1, commuting_nodes_2, ...]}
    //   * The index in which commutation set a given node is located on a wire: {(node, wire): index}
    // The Python dict will store both of these dictionaries in one.
    let (commutation_set, node_indices) =
        analyze_commutations(dag, commutation_checker, approximation_degree)?;

    let out_dict = PyDict::new(py);

    // First set the {wire: [commuting_nodes_1, ...]} bit
    for (wire_index, commutations) in commutation_set.into_iter().enumerate() {
        if commutations.is_empty() {
            continue;
        }
        // we know all wires are of type Wire::Qubit, since in analyze_commutations_inner
        // we only iterate over the qubits
        let py_wire = dag
            .qubits()
            .get(Qubit::new(wire_index))
            .unwrap()
            .into_pyobject(py)?;

        out_dict.set_item(
            py_wire,
            PyList::new(
                py,
                commutations.iter().map(|inner| {
                    PyList::new(
                        py,
                        inner
                            .iter()
                            .map(|node_index| dag.get_node(py, *node_index).unwrap()),
                    )
                    .unwrap()
                }),
            )?,
        )?;
    }
    // Then we add the {(node, wire): index} dictionary
    for (qubit, node_index_map) in node_indices.iter().enumerate() {
        if node_index_map.is_empty() {
            continue;
        }
        for (node_index, index) in node_index_map {
            let py_wire = dag
                .qubits()
                .get(Qubit::new(qubit))
                .unwrap()
                .into_pyobject(py)?;
            out_dict.set_item((dag.get_node(py, *node_index)?, py_wire), index)?;
        }
    }

    Ok(out_dict.unbind())
}

pub fn commutation_analysis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_analyze_commutations))?;
    Ok(())
}
