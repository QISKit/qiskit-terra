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

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate, multiply_param, radd_param};
use qiskit_quantum_info::clifford::{PauliLabelOrder, PauliList};

use qiskit_util::getenv_use_multiple_threads;
use rayon::prelude::*;
use rustworkx_core::petgraph::Direction::Outgoing;
use rustworkx_core::petgraph::Incoming;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::prelude::StableDiGraph;

use hashbrown::HashMap;
use smallvec::{SmallVec, smallvec};

use std::f64::consts::SQRT_2;
use std::fmt;

use crate::clifford::greedy_synthesis::resynthesize_clifford_circuit;
use crate::evolution::chunks::{ALL_CHUNKS, REDUCING_CHUNKS, SUPPORT_DELTA};
use crate::evolution::{EvolutionSynthesisError, paulis_to_dense};

// The MCTS algorithm implemented in this file is based on the papers [1], [2],
// their respecitive implementations in [3], [4], but with substantial modifications
// and performance optimizations beyond the original algorithms.
//
// References:
//
// 1. Mulundano Machiya, Matt Menickelly, Paul Hovland, Ji Liu,
// *MonteQ: A Monte Carlo Tree Search Based Quantum Circuit Synthesis Framework*,
// [arXiv:2604.19029](https://arxiv.org/abs/2604.19029).
//
// 2. Timothée Goubault de Brugière and Simon Martiel,
// *Faster and shorter synthesis of Hamiltonian simulation circuits*,
// [arXiv:2404.03280](https://arxiv.org/abs/2404.03280).
//
// 3. https://github.com/Mulundano/MonteQ,
// The code is licensed under the MIT license.
//
// 4. https://github.com/qiskit-community/rustiq-core,
// The code is licensed under the MIT license.

/// The multiplicative scaling parameter used in the MCTS algorithm.
const MCTS_PARAM: f64 = SQRT_2;

/// Sequence of standard gates constructed by the algorithm, including
/// both Clifford gates and single-qubit rotations.
type GateSequence = Vec<(StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)>;

/// A particular point during Pauli network synthesis.
#[derive(Clone)]
struct PauliSynthesisState {
    /// The tableau storing Pauli rotations corresponding to this state
    /// (the initial tableau conjugated by Clifford gates leading to this state).
    /// The Paulis that have already been synthesized are stored as ``None`` in
    /// ``in_degrees``.
    tab: PauliList,

    /// Support sizes for Paulis in the pauli list. This is a performance optimization that
    /// avoid recomputing these and instead updates them on the fly for active Paulis.
    support_sizes: Vec<usize>,

    /// Number of processed Pauli rotations.
    num_processed: usize,

    /// Sequence of gates (including Clifford and rotation gates)
    /// leading to this state.
    gate_sequence: GateSequence,

    /// Number of unprocessed predecessors for each node in the commutativity DAG
    /// (with ``None`` nodes considered as removed).
    in_degrees: Vec<Option<usize>>,
}

impl fmt::Display for PauliSynthesisState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tab = {}, num_processed = {}, in_degrees = {:?}",
            self.tab, self.num_processed, self.in_degrees
        )
    }
}

/// A node in the MCTS tree.
struct MctsNode {
    /// Number of times this node was visited across simulations.
    ni: usize,

    /// Cumulative cost (e.g. number of CX-gates) from root to terminal node
    /// across simulations visiting this node.
    qi: usize,

    /// Parent node in the MCTS tree.
    parent: Option<usize>,

    /// Child nodes in the MCTS tree.
    children: Vec<usize>,

    /// The corresponding state of the Pauli network synthesis algorithm.
    synthesis_state: PauliSynthesisState,

    /// Unexplored actions at this node. Correspond to Pauli indices that have not yet been
    /// explored by the MCTS algorithm when starting from this state.
    unexplored_actions: Vec<usize>,
}

/// Count the number of CX-gates in [GateSequence].
///
/// All non-CX gates are ignored.
fn cx_count(gate_seq: &GateSequence) -> usize {
    gate_seq
        .iter()
        .filter(|(gate, _, _)| *gate == StandardGate::CX)
        .count()
}

/// Count the number of CX-gates in [GateSequence].
///
/// This also handles SWAPs, counting each as 3 CX-gates. Note that SWAP gates can be produced
/// by Clifford resynthesis.
fn cx_count_with_swaps(gate_seq: &GateSequence) -> usize {
    // This function may also be called on the resynthesized Clifford circuit,
    // and so also considers SWAP gates.
    gate_seq
        .iter()
        .filter_map(|(gate, _, _)| match gate {
            StandardGate::CX => Some(1),
            StandardGate::Swap => Some(3),
            _ => None,
        })
        .sum()
}

/// Score of applying a 2-qubit Clifford circuit on a pair of qubits in the PauliList.
/// Larger scores are better.
type Score = (isize, isize);

/// Count how many times each of the 16 possible 2-qubit Paulis appear for a fixed
/// pair `(ctrl, trgt)`.
fn count_active_pairs(
    synthesis_state: &PauliSynthesisState,
    ctrl: usize,
    trgt: usize,
) -> [u32; 16] {
    let mut pair_counts = [0u32; 16];
    for pauli_idx in 0..synthesis_state.tab.num_paulis {
        if synthesis_state.in_degrees[pauli_idx].is_some() {
            let pair_idx = synthesis_state.tab.pauli_pair_index(pauli_idx, ctrl, trgt);
            pair_counts[pair_idx] += 1;
        }
    }
    pair_counts
}

/// Computes the score of applying a Clifford circuit represented by `chunk_idx` on a pair of
/// qubits, where `pair_counts` represent the numbers of different types of 2-qubit Paulis
/// (obtained by restricting the active Pauli roations to this pair of qubits).
fn compute_score(pair_counts: &[u32; 16], chunk_idx: usize) -> Score {
    let deltas = &SUPPORT_DELTA[chunk_idx];
    let mut total_decrease: isize = 0;
    let mut num_decreased: isize = 0;

    for (pair_idx, &pair_count) in pair_counts.iter().enumerate() {
        if pair_count == 0 {
            continue;
        }
        let delta = deltas[pair_idx] as isize;
        total_decrease -= delta * (pair_count as isize);
        num_decreased += ((delta < 0) as isize) * (pair_count as isize);
    }
    (total_decrease, num_decreased)
}

/// Synthesizes a given Pauli, that is finds a sequence of Clifford gates that brings this Pauli to a single-qubit
/// Pauli rotation (note that the same sequence can bring other Paulis to single-qubit Paulis as well).
/// Updates the synthesis state in-place, adjusting the Pauli tableau (conjugating by Clifford gates) and extending
/// the gate sequence.
fn synthesize_pauli(synthesis_state: &mut PauliSynthesisState, ndx: usize) {
    // Caches previously computed information. For pairs (ctrl, trgt) with ctrl < trgt,
    // store how many times each of the 16 possible 2-qubit Paulis appear.
    let mut counts_cache: HashMap<(usize, usize), [u32; 16]> = HashMap::new();

    // On each iteration of the while-loop, the support size of the given Pauli rotation should
    // decrease by 1. We stop iterating when the Pauli is reduced to a single-qubit rotation.
    while synthesis_state.support_sizes[ndx] > 1 {
        let support = synthesis_state.tab.get_pauli_support(ndx);

        // Loop to cycle over all qubit indices in the support to get all combinations of control and target.
        let mut best_score = (isize::MIN, isize::MIN);
        let mut best_ctrl = 0;
        let mut best_trgt = 0;
        let mut best_chunk_idx = 0;
        for ii in 0..support.len() {
            for jj in ii + 1..support.len() {
                // note that ctrl < trgt
                let ctrl = support[ii];
                let trgt = support[jj];

                // get the index 0..16
                let pair_idx = synthesis_state.tab.pauli_pair_index(ndx, ctrl, trgt);

                // Count the numbers of different 2-qubit Paulis on (ctrl, trgt). The
                // counts only depend on `ctrl` and `trgt`, and change only when the applied
                // chunk involves one (or both) of the qubits.
                let pair_counts = *counts_cache
                    .entry((ctrl, trgt))
                    .or_insert_with(|| count_active_pairs(synthesis_state, ctrl, trgt));

                // get the indices of chunks that can reduce this
                for chunk_idx in REDUCING_CHUNKS[pair_idx] {
                    let score = compute_score(&pair_counts, *chunk_idx);
                    if score > best_score {
                        best_score = score;
                        best_chunk_idx = *chunk_idx;
                        best_ctrl = ctrl;
                        best_trgt = trgt;
                    }
                }
            }
        }

        // Updates following applying the best chunk -- pauli support sizes
        for pauli_idx in 0..synthesis_state.tab.num_paulis {
            if synthesis_state.in_degrees[pauli_idx].is_some() {
                let pair_idx = synthesis_state
                    .tab
                    .pauli_pair_index(pauli_idx, best_ctrl, best_trgt);

                let delta = SUPPORT_DELTA[best_chunk_idx][pair_idx];
                synthesis_state.support_sizes[pauli_idx] =
                    ((synthesis_state.support_sizes[pauli_idx] as isize) + (delta as isize))
                        as usize;
            }
        }

        // Updates following applying the best chunk -- pauli list
        for (gate, qubits) in ALL_CHUNKS[best_chunk_idx] {
            let mapped_qubits: SmallVec<[Qubit; 2]> = qubits
                .iter()
                .map(|q| match q {
                    0 => Qubit(best_ctrl as u32),
                    1 => Qubit(best_trgt as u32),
                    _ => {
                        unreachable!("can only have 0/1");
                    }
                })
                .collect();

            match gate {
                StandardGate::H => {
                    synthesis_state.tab.append_h(mapped_qubits[0].index());
                }
                StandardGate::S => {
                    synthesis_state.tab.append_s(mapped_qubits[0].index());
                }
                StandardGate::SX => {
                    synthesis_state.tab.append_sx(mapped_qubits[0].index());
                }
                StandardGate::CX => {
                    synthesis_state
                        .tab
                        .append_cx(mapped_qubits[0].index(), mapped_qubits[1].index());
                }
                _ => {
                    unreachable!("chunks only contain s/sx/h/cx gates");
                }
            }

            synthesis_state
                .gate_sequence
                .push((*gate, smallvec![], mapped_qubits));
        }

        // Evict stale entries
        counts_cache.retain(|&(i, j), _| {
            i != best_ctrl && i != best_trgt && j != best_ctrl && j != best_trgt
        });
    }
}

/// Updates in-degree in-place with newly processed nodes.
fn update_in_degrees(
    dag: &StableDiGraph<usize, ()>,
    in_degrees: &mut [Option<usize>],
    new_processed: &[usize],
) {
    for i in new_processed {
        for n in dag.neighbors_directed(NodeIndex::new(*i), Outgoing) {
            let degree = in_degrees[n.index()]
                .as_mut()
                .expect("the successor node should not be processed");
            *degree -= 1;
        }
        in_degrees[*i] = None;
    }
}

/// Given a DAG, and a list of in-degrees (see the algorithm for detail),
/// computes frontier nodes.
fn compute_frontier_nodes(
    dag: &StableDiGraph<usize, ()>,
    in_degrees: &[Option<usize>],
) -> Vec<usize> {
    dag.node_indices()
        .filter(|n| in_degrees[n.index()] == Some(0))
        .map(|n| n.index())
        .collect()
}

// Computes support sizes (numbers of non-I terms) for every Pauli in the tableau.
fn compute_support_sizes(tab: &PauliList) -> Vec<usize> {
    (0..tab.num_paulis)
        .map(|i| tab.get_pauli_support_size(i))
        .collect()
}

/// Sorts a sparse Pauli by qubits, e.g., sorting ("XIZ", [5, 1, 2]) should produce ("IZX", [1, 2, 5]).
fn sort_sparse_pauli(pauli: String, qubits: Vec<u32>) -> (String, Vec<u32>) {
    let mut zipped: Vec<(u8, u32)> = pauli
        .into_bytes()
        .into_iter()
        .zip(qubits.iter().copied())
        .collect();
    zipped.sort_unstable_by_key(|&(_, q)| q);
    let (sorted_bytes, sorted_qubits): (Vec<u8>, Vec<u32>) = zipped.into_iter().unzip();
    let sorted_paulis = String::from_utf8(sorted_bytes).unwrap();
    (sorted_paulis, sorted_qubits)
}

/// Checks whether two sparse sorted paulis commute.
#[inline]
fn sorted_sparse_paulis_commute(
    pauli1: &str,
    qubits1: &[u32],
    pauli2: &str,
    qubits2: &[u32],
) -> bool {
    let mut idx1 = 0;
    let mut idx2 = 0;
    let mut parity = false;
    while (idx1 < qubits1.len()) && (idx2 < qubits2.len()) {
        if qubits1[idx1] < qubits2[idx2] {
            idx1 += 1;
        } else if qubits1[idx1] > qubits2[idx2] {
            idx2 += 1;
        } else {
            // same qubit
            parity ^= (pauli1.as_bytes()[idx1] != b'I')
                && (pauli2.as_bytes()[idx2] != b'I')
                && (pauli1.as_bytes()[idx1] != pauli2.as_bytes()[idx2]);
            idx1 += 1;
            idx2 += 1;
        }
    }
    !parity
}

/// The main algorithm.
struct MctsAlgorithm {
    /// Total number of Pauli rotations to synthesize.
    num_paulis: usize,

    /// Paulis to be synthesized.
    paulis: PauliList,

    /// Parameters for rotation angles.
    angles: Vec<Param>,

    /// MCTS nodes (forming a tree).
    mcts_nodes: Vec<MctsNode>,

    /// Anti-commutativity DAG for Pauli rotations.
    dag: StableDiGraph<usize, ()>,

    /// Global phase of the circuit.
    global_phase: Param,
}

impl MctsAlgorithm {
    /// Obtains the input in sparse format, consuming the input.
    /// Creates a tree with a single root node.
    fn new(
        num_qubits: usize,
        sparse_paulis: Vec<(String, Vec<u32>)>,
        angles: Vec<Param>,
        preserve_order: bool,
    ) -> Self {
        let num_paulis = sparse_paulis.len();

        // Build anti-commutativity DAG.
        // In the future, we may either parallelize this, or implement a slightly more clever algorithm
        // from python's DAGDependency code that avoids adding transitive edges.
        let mut dag = StableDiGraph::<usize, ()>::new();
        let node_indices: Vec<NodeIndex> = (0..num_paulis).map(|i| dag.add_node(i)).collect();

        let dense_paulis = if preserve_order {
            // We build the commutativity DAG, and for efficiency we check commutativity between sorted Paulis
            // in sparse format.

            let sorted_sparse_paulis: Vec<(String, Vec<u32>)> = sparse_paulis
                .into_iter()
                .map(|(p, q)| sort_sparse_pauli(p, q))
                .collect();

            for i in 0..num_paulis {
                for j in i + 1..num_paulis {
                    if !sorted_sparse_paulis_commute(
                        &sorted_sparse_paulis[i].0,
                        &sorted_sparse_paulis[i].1,
                        &sorted_sparse_paulis[j].0,
                        &sorted_sparse_paulis[j].1,
                    ) {
                        dag.add_edge(node_indices[i], node_indices[j], ());
                    }
                }
            }
            paulis_to_dense(num_qubits, sorted_sparse_paulis)
        } else {
            paulis_to_dense(num_qubits, sparse_paulis)
        };

        let paulis =
            PauliList::from_pauli_labels(num_qubits, &dense_paulis, PauliLabelOrder::LeftToRight)
                .expect("When MCTS algorithm is called, the Pauli labels have been validated.");

        Self {
            global_phase: Param::Float(0.0),
            num_paulis,
            paulis,
            angles,
            mcts_nodes: vec![],
            dag,
        }
    }

    /// Adds the root MCTS node.
    /// The root includes the tableau after processing all-identity and all active single-qubit rotations
    /// in the original tableau.
    fn add_root_mcts_node(&mut self, tab: &PauliList) {
        let mut in_degrees: Vec<Option<usize>> = vec![None; self.num_paulis];
        self.dag.node_indices().for_each(|i| {
            in_degrees[i.index()] = Some(self.dag.neighbors_directed(i, Incoming).count());
        });
        let support_sizes = compute_support_sizes(tab);
        let mut synthesis_state = PauliSynthesisState {
            tab: tab.clone(),
            support_sizes,
            num_processed: 0,
            gate_sequence: vec![],
            in_degrees,
        };
        self.global_phase = self.process_all_identity_paulis(&mut synthesis_state);
        self.process_synthesized_paulis(&mut synthesis_state);

        let unexplored_actions = {
            let mut frontier_nodes = compute_frontier_nodes(&self.dag, &synthesis_state.in_degrees);
            // Reversing to allow efficient popping of nodes from the back of the vector, while preserving
            // the original order.
            frontier_nodes.reverse();
            frontier_nodes
        };

        let node = MctsNode {
            ni: 0,
            qi: 0,
            parent: None,
            children: vec![],
            synthesis_state,
            unexplored_actions,
        };
        self.mcts_nodes.push(node);
    }

    /// Returns the "upper confidence bound" score for a given node in the tree.
    /// These scores balance the exploration of the previously unexplored vertices and
    /// the exploitation of the previously explored promising virtices. Higher scores
    /// are better. See [1] for additional details.
    fn upper_confidence_bound(&self, node_idx: usize) -> f64 {
        let node = &self.mcts_nodes[node_idx];

        if node.ni == 0 {
            // it has been completely unexplored
            f64::INFINITY
        } else {
            match node.parent {
                None => {
                    // it is a root
                    -((node.qi as f64) / (node.ni as f64))
                }
                Some(parent_idx) => {
                    let parent_node = &self.mcts_nodes[parent_idx];
                    -((node.qi as f64) / (node.ni as f64))
                        + MCTS_PARAM * ((parent_node.ni as f64).ln() / (node.ni as f64)).sqrt()
                }
            }
        }
    }

    /// Backpropagate estimated value from the terminal state up through the tree.
    fn backpropagate(&mut self, leaf_node_id: usize, value: usize) {
        let mut node_id = Some(leaf_node_id);
        while let Some(id) = node_id {
            let node = &mut self.mcts_nodes[id];
            node.ni += 1;
            node.qi += value;
            node_id = node.parent;
        }
    }

    /// Runs simulation from each leaf node. In parallel, if there is more than one leaf node.
    /// Returns the solution and the number of CX-gates in the solution.
    fn run_next_batch(&mut self, leaf_node_ids: &[usize]) -> (GateSequence, usize) {
        let rollout = |&leaf_node_id: &usize| {
            let solution = self.rollout_policy(leaf_node_id);
            let value = cx_count(&solution);
            (leaf_node_id, solution, value)
        };

        // Run rollout (greedy synthesis) from each leaf.
        let results: Vec<(usize, GateSequence, usize)> = if leaf_node_ids.len() == 1 {
            leaf_node_ids.iter().map(rollout).collect()
        } else {
            leaf_node_ids.par_iter().map(rollout).collect()
        };

        // Back-propagate values using all of the discovered solutions.
        for &(leaf_node_id, _, value) in &results {
            self.backpropagate(leaf_node_id, value);
        }

        // Pick the solution with the minimum value.
        results
            .into_iter()
            .min_by_key(|(_leaf, _solution, value)| *value)
            .map(|(_leaf, solution, value)| (solution, value))
            .expect("we should have at least one solution")
    }

    /// The main function which runs everything.
    fn run(
        &mut self,
        num_simulations: usize,
        max_parallel_simulations: Option<usize>,
    ) -> (GateSequence, Param) {
        // Add the root state (this also removes all-identity and synthesizable 1-qubit rotations).
        self.add_root_mcts_node(&self.paulis.clone());

        // Number of simulations per batch. The simulation in one batch run in parallel.
        let run_in_parallel = getenv_use_multiple_threads();
        let batch_size = match (run_in_parallel, max_parallel_simulations) {
            (false, _) => 1,
            (true, Some(k)) => k.min(num_simulations),
            (true, None) => num_simulations,
        };
        // Number of batches to run. Batches run sequentially.
        let num_batches = num_simulations.div_ceil(batch_size);

        // The first simulation starts from the root MCTS node. Each consecutive simulation
        // either selects an existing MCTS state with unexplored actions or creates a new MCTS
        // state. For each selected state, the rollout strategy (heuristic greedy synthesis)
        // is called, starting at this state.
        let mut first_run = true;
        let best_solution = (0..num_batches)
            .map(|_| {
                let leaf_node_ids = self.select_next_k_child_states(batch_size, first_run);
                first_run = false;
                self.run_next_batch(&leaf_node_ids)
            })
            .min_by_key(|(_solution, value)| *value)
            .map(|(solution, _value)| solution)
            .expect("we should have at least one solution");
        (best_solution, self.global_phase.clone())
    }

    /// Find or create the next MCTS nodes to start the rollout from (called the "tree policy" in the paper).
    /// If it's the very first run of this function, returns the root node.
    fn select_next_child_state(&mut self, first_run: bool) -> usize {
        if first_run {
            return 0;
        }

        let mut mcts_node_id = 0; // root
        loop {
            // If all the Paulis have been processed, return the current id.
            if self.mcts_nodes[mcts_node_id].synthesis_state.num_processed == self.num_paulis {
                return mcts_node_id;
            }

            // Node includes unexplored actions (Paulis that can be immediately synthesized).
            // We will create a new MCTS by synthesizing one of the unexplored Paulis and removing
            // newly synthesized rotations.
            if !self.mcts_nodes[mcts_node_id].unexplored_actions.is_empty() {
                // Choose one of the unprocessed paulis in the front layer.
                let chosen_pauli_idx = self.mcts_nodes[mcts_node_id]
                    .unexplored_actions
                    .last()
                    .unwrap();

                // Synthesize it.
                let mut synthesis_state = self.mcts_nodes[mcts_node_id].synthesis_state.clone();
                synthesize_pauli(&mut synthesis_state, *chosen_pauli_idx);
                self.process_synthesized_paulis(&mut synthesis_state);

                let unexplored_actions = {
                    let mut frontier_nodes =
                        compute_frontier_nodes(&self.dag, &synthesis_state.in_degrees);
                    // Reversing to allow efficient popping of nodes from the back of the vector, while preserving
                    // the original order.
                    frontier_nodes.reverse();
                    frontier_nodes
                };

                // Create the child mcts node with the result of the above synthesis.
                let child_node = MctsNode {
                    ni: 0,
                    qi: 0,
                    parent: Some(mcts_node_id),
                    children: vec![],
                    synthesis_state,
                    unexplored_actions,
                };

                // Add child to tree and remove the synthesized pauli from the set of unexplored actions.
                self.mcts_nodes.push(child_node);
                let child_node_id = self.mcts_nodes.len() - 1;
                self.mcts_nodes[mcts_node_id].children.push(child_node_id);
                self.mcts_nodes[mcts_node_id].unexplored_actions.pop();
                return child_node_id;
            };

            // All the children have been explored; choose the child with highest upper confidence bound
            // (and proceed recursively examining this child).
            mcts_node_id = *self.mcts_nodes[mcts_node_id]
                .children
                .iter()
                .map(|node_idx| {
                    let score = self.upper_confidence_bound(*node_idx);
                    (node_idx, score)
                })
                .max_by(|&a, &b| a.1.total_cmp(&b.1))
                .unwrap()
                .0;
        }
    }

    /// Find or create the next k MCTS nodes to start the rollout from
    /// (for parallelization).
    fn select_next_k_child_states(&mut self, k: usize, first_run: bool) -> Vec<usize> {
        (0..k)
            .map(|i| {
                let is_first_run = first_run && (i == 0);
                self.select_next_child_state(is_first_run)
            })
            .collect()
    }

    /// Finds all-identity paulis and updates the synthesis state accordingly.
    /// Called once at the start of the algorithm.
    /// Returns the global phase update.
    fn process_all_identity_paulis(&self, synthesis_state: &mut PauliSynthesisState) -> Param {
        let mut global_phase = Param::Float(0.0);
        let mut new_processed: Vec<usize> = Vec::new();
        for pauli_idx in 0..self.num_paulis {
            if synthesis_state.support_sizes[pauli_idx] == 0 {
                new_processed.push(pauli_idx);
                synthesis_state.num_processed += 1;
                global_phase =
                    radd_param(global_phase, multiply_param(&self.angles[pauli_idx], -0.5));
            }
        }
        if !new_processed.is_empty() {
            update_in_degrees(&self.dag, &mut synthesis_state.in_degrees, &new_processed);
        }
        global_phase
    }

    /// Recursively find synthesized paulis and update the synthesis state accordingly.
    fn process_synthesized_paulis(&self, state: &mut PauliSynthesisState) {
        // Runs recursively because processing one single-qubit Pauli in the front layer may enable
        // additional single-qubit Paulis in the following layers.
        loop {
            // All the Paulis have been processed.
            if state.num_processed == self.num_paulis {
                break;
            }

            // Find Paulis of weight 1 in the front layer,
            let mut new_processed: Vec<usize> = Vec::new();
            let frontier_nodes = compute_frontier_nodes(&self.dag, &state.in_degrees);
            for idx in &frontier_nodes {
                if state.support_sizes[*idx] == 1 {
                    let q = state
                        .tab
                        .get_pauli_support_if_size_1(*idx)
                        .expect("We are guaranteed to have a single qubit in the support");
                    new_processed.push(*idx);

                    // update number of processed nodes
                    state.num_processed += 1;

                    // update the circuit, including the single-qubit rotation gate
                    let angle = &self.angles[*idx];
                    match (
                        state.tab.get_pauli_x(*idx, q),
                        state.tab.get_pauli_z(*idx, q),
                        state.tab.get_pauli_phase(*idx),
                    ) {
                        (true, false, false) => state.gate_sequence.push((
                            StandardGate::RX,
                            smallvec![angle.clone()],
                            smallvec![Qubit(q as u32)],
                        )),
                        (true, false, true) => state.gate_sequence.push((
                            StandardGate::RX,
                            smallvec![multiply_param(angle, -1.0)],
                            smallvec![Qubit(q as u32)],
                        )),
                        (false, true, false) => state.gate_sequence.push((
                            StandardGate::RZ,
                            smallvec![angle.clone()],
                            smallvec![Qubit(q as u32)],
                        )),
                        (false, true, true) => state.gate_sequence.push((
                            StandardGate::RZ,
                            smallvec![multiply_param(angle, -1.0)],
                            smallvec![Qubit(q as u32)],
                        )),
                        (true, true, false) => state.gate_sequence.push((
                            StandardGate::RY,
                            smallvec![angle.clone()],
                            smallvec![Qubit(q as u32)],
                        )),
                        (true, true, true) => state.gate_sequence.push((
                            StandardGate::RY,
                            smallvec![multiply_param(angle, -1.0)],
                            smallvec![Qubit(q as u32)],
                        )),
                        _ => {
                            unreachable!("The Pauli support qubit cannot be I");
                        }
                    }
                }
            }

            if new_processed.is_empty() {
                break;
            }

            // recompute in_degrees
            update_in_degrees(&self.dag, &mut state.in_degrees, &new_processed);
        }
    }

    /// Starting from an MCTS node, implements the rollout (synthesizing all Paulis).
    /// Returns the solution.
    /// A possible follow-up: also implement Rustiq's heuristic for minimizing CX-count
    /// (by changing the internal scoring function).
    fn rollout_policy(&self, mcts_node_id: usize) -> GateSequence {
        // We are cloning this state, since are going to update it in-place.
        let mut synthesis_state = self.mcts_nodes[mcts_node_id].synthesis_state.clone();

        while synthesis_state.num_processed < self.num_paulis {
            // Compute front nodes.
            let front_nodes = compute_frontier_nodes(&self.dag, &synthesis_state.in_degrees);

            // Find active pauli of minimum weight.
            let pauli_idx = front_nodes
                .iter()
                .min_by_key(|&&idx| synthesis_state.support_sizes[idx])
                .expect("The frontier cannot be empty.");

            // We should have filtered all already synthesized paulis.
            synthesize_pauli(&mut synthesis_state, *pauli_idx);

            // Update the state by finding all Paulis that got synthesized.
            self.process_synthesized_paulis(&mut synthesis_state);
        }
        synthesis_state.gate_sequence
    }
}

static ROTATION_GATES: [StandardGate; 3] = [StandardGate::RX, StandardGate::RY, StandardGate::RZ];

/// See python documentation for ``synth_pauli_network_mcts`` for details.
#[allow(clippy::too_many_arguments)]
pub fn pauli_network_mcts_inner(
    num_qubits: usize,
    sparse_paulis: Vec<(String, Vec<u32>)>,
    angles: Vec<Param>,
    preserve_order: bool,
    upto_clifford: bool,
    upto_phase: bool,
    num_simulations: usize,
    max_parallel_simulations: Option<usize>,
) -> Result<CircuitData, EvolutionSynthesisError> {
    if num_simulations == 0 {
        return Err(EvolutionSynthesisError::ErrorInvalidInputArguments(
            "Number of simulations must be at least 1.".to_string(),
        ));
    }
    if max_parallel_simulations == Some(0) {
        return Err(EvolutionSynthesisError::ErrorInvalidInputArguments(
            "Number of simulations that can be executed in parallel must be at least 1."
                .to_string(),
        ));
    }

    let mut mcts = MctsAlgorithm::new(num_qubits, sparse_paulis, angles, preserve_order);
    let (mut circuit, global_phase) = mcts.run(num_simulations, max_parallel_simulations);

    // If upto_clifford is true, we just return the above circuit. However, if upto_clifford is false, we need to
    // add the inverse clifford part of the circuit.
    if !upto_clifford {
        // Extract & inverse the Clifford part of the circuit
        let inverse_clifford_circuit: GateSequence = circuit
            .iter()
            .filter(|(gate, _params, _qubits)| !ROTATION_GATES.contains(gate))
            .rev()
            .map(|(gate, params, qubits)| {
                let (inverse_gate, inverse_params) = gate
                    .inverse(params)
                    .expect("All Clifford gates that can appear here are invertible");
                (inverse_gate, inverse_params, qubits.clone())
            })
            .collect();

        if !upto_phase {
            // If the circuit needs to be synthesized exactly, we cannot use Clifford resynthesis methods
            // since they do not preserve the global phase.
            circuit.extend(inverse_clifford_circuit);
        } else {
            //  However, if upto_phase is true, we can attempt to resynthesize the final clifford.
            let resynthesized =
                resynthesize_clifford_circuit(num_qubits, &inverse_clifford_circuit)
                    .map_err(EvolutionSynthesisError::ErrorFromCliffordSynthesis)?;
            if cx_count_with_swaps(&inverse_clifford_circuit) <= cx_count_with_swaps(&resynthesized)
            {
                circuit.extend(inverse_clifford_circuit);
            } else {
                circuit.extend(resynthesized);
            }
        }
    }

    Ok(CircuitData::from_standard_gates(
        num_qubits as u32,
        circuit,
        global_phase,
    )?)
}
