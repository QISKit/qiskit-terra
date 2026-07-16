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

use std::cell::RefCell;

use qiskit_util::{IndexMap, IndexSet};

use crate::equivalence::{EdgeData, Equivalence, EquivalenceLibrary, Key, NodeData};
use qiskit_circuit::operations::Operation;
use rustworkx_core::petgraph::stable_graph::{EdgeReference, NodeIndex, StableDiGraph};
use rustworkx_core::petgraph::visit::Control;
use rustworkx_core::traversal::{DijkstraEvent, dijkstra_search};

use super::compose_transforms::{BasisTransformIn, GateIdentifier};

type BasisTransforms = Vec<(GateIdentifier, BasisTransformIn)>;
/// Search for a set of transformations from source_basis to target_basis.
///
/// Performs a Dijkstra search algorithm on the `EquivalenceLibrary`'s core graph
/// to rate and classify different possible equivalent circuits to the provided gates.
///
/// This is done by connecting all the nodes represented in the `target_basis` to a dummy
/// node, and then traversing the graph until all the nodes described in the `source
/// basis` are reached.
pub(crate) fn basis_search(
    equiv_lib: &mut EquivalenceLibrary,
    source_basis: &IndexSet<GateIdentifier>,
    target_basis: &IndexSet<&str>,
) -> Option<BasisTransforms> {
    // Build the visitor attributes:
    let mut num_gates_remaining_for_rule: IndexMap<usize, usize> = IndexMap::default();
    let predecessors: RefCell<IndexMap<GateIdentifier, Equivalence>> =
        RefCell::new(IndexMap::default());
    let opt_cost_map: RefCell<IndexMap<GateIdentifier, u32>> = RefCell::new(IndexMap::default());
    let mut basis_transforms: Vec<(GateIdentifier, BasisTransformIn)> = vec![];

    // Initialize visitor attributes:
    initialize_num_gates_remain_for_rule(equiv_lib.graph(), &mut num_gates_remaining_for_rule);

    let mut source_basis_remain: IndexSet<Key> = source_basis
        .iter()
        .filter_map(|(gate_name, gate_num_qubits)| {
            if !target_basis.contains(gate_name.as_str()) {
                Some(Key {
                    name: gate_name.to_string(),
                    num_qubits: *gate_num_qubits,
                })
            } else {
                None
            }
        })
        .collect();

    // If source_basis is empty, no work needs to be done.
    if source_basis_remain.is_empty() {
        return Some(vec![]);
    }

    // This is only necessary since gates in target basis are currently reported by
    // their names and we need to have in addition the number of qubits they act on.
    let target_basis_keys: Vec<Key> = equiv_lib
        .keys()
        .filter(|&key| target_basis.contains(key.name.as_str()))
        .cloned()
        .collect();

    // Dummy node is inserted in the graph. Which is where the search will start
    let dummy: NodeIndex = equiv_lib.graph_mut().add_node(NodeData {
        equivs: vec![],
        key: Key {
            name: "key".to_string(),
            num_qubits: u32::MAX,
        },
    });

    // Extract indices for the target_basis gates, to avoid borrowing from graph.
    let target_basis_indices: Vec<NodeIndex> = target_basis_keys
        .iter()
        .map(|key| equiv_lib.node_index(key))
        .collect();

    // Connect each edge in the target_basis to the dummy node.
    for node in target_basis_indices {
        equiv_lib.graph_mut().add_edge(dummy, node, None);
    }

    // Edge cost function for Visitor
    let edge_weight = |edge: EdgeReference<Option<EdgeData>>| -> Result<u32, ()> {
        if edge.weight().is_none() {
            return Ok(1);
        }
        let edge_data = edge.weight().as_ref().unwrap();
        let mut cost_tot = 0;
        let borrowed_cost = opt_cost_map.borrow();
        for instruction in edge_data.rule.circuit.iter() {
            let instruction_op = instruction.op.view();
            cost_tot += borrowed_cost[&(
                instruction_op.name().to_string(),
                instruction_op.num_qubits(),
            )];
        }
        Ok(cost_tot
            - borrowed_cost[&(
                edge_data.source.name.to_string(),
                edge_data.source.num_qubits,
            )])
    };

    let event_matcher = |event: DijkstraEvent<NodeIndex, &Option<EdgeData>, u32>| {
        match event {
            DijkstraEvent::Discover(n, score) => {
                let gate_key = &equiv_lib.graph()[n].key;
                let gate = (gate_key.name.to_string(), gate_key.num_qubits);
                source_basis_remain.swap_remove(gate_key);
                let mut borrowed_cost_map = opt_cost_map.borrow_mut();
                if let Some(entry) = borrowed_cost_map.get_mut(&gate) {
                    *entry = score;
                } else {
                    borrowed_cost_map.insert(gate.clone(), score);
                }
                if let Some(rule) = predecessors.borrow().get(&gate) {
                    basis_transforms.push((
                        (gate_key.name.to_string(), gate_key.num_qubits),
                        (rule.params.clone(), rule.circuit.clone()),
                    ));
                }

                if source_basis_remain.is_empty() {
                    basis_transforms.reverse();
                    return Control::Break(());
                }
            }
            DijkstraEvent::EdgeRelaxed(_, target, Some(edata)) => {
                let gate = &equiv_lib.graph()[target].key;
                predecessors
                    .borrow_mut()
                    .entry((gate.name.to_string(), gate.num_qubits))
                    .and_modify(|value| *value = edata.rule.clone())
                    .or_insert(edata.rule.clone());
            }
            DijkstraEvent::ExamineEdge(_, target, Some(edata)) => {
                num_gates_remaining_for_rule
                    .entry(edata.index)
                    .and_modify(|val| *val -= 1)
                    .or_insert(0);
                let target = &equiv_lib.graph()[target].key;

                // If there are gates in this `rule` that we have not yet generated, we can't apply
                // this `rule`. if `target` is already in basis, it's not beneficial to use this rule.
                if num_gates_remaining_for_rule[&edata.index] > 0
                    || target_basis_keys.contains(target)
                {
                    return Control::Prune;
                }
            }
            _ => {}
        };
        Control::Continue
    };

    let basis_transforms =
        match dijkstra_search(&equiv_lib.graph(), [dummy], edge_weight, event_matcher) {
            Ok(Control::Break(_)) => Some(basis_transforms),
            _ => None,
        };
    equiv_lib.graph_mut().remove_node(dummy);
    basis_transforms
}

fn initialize_num_gates_remain_for_rule(
    graph: &StableDiGraph<NodeData, Option<EdgeData>>,
    source: &mut IndexMap<usize, usize>,
) {
    let mut save_index = usize::MAX;
    // When iterating over the edges, ignore any none-valued ones by calling `flatten`
    for edge_data in graph.edge_weights().flatten() {
        if save_index == edge_data.index {
            continue;
        }
        source.insert(edge_data.index, edge_data.num_gates);
        save_index = edge_data.index;
    }
}
