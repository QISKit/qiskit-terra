// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashMap;
use pyo3::prelude::*;
use rustworkx_core::petgraph::prelude::*;
use thiserror::Error;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::ControlFlowView;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Qubit, VirtualQubit};

/// The type of a node in the Sabre interactions graph.
#[derive(Clone, Debug)]
pub enum InteractionKind {
    /// The node is some sort of synchronisation point, but not something that imposes routing
    /// constraints itself.  It can be pushed into the output as soon as all its predecessors are
    /// satisfied.  This might be a directive, or a single-qubit gate conditioned on clbits that
    /// depend on more than one 2q gate, or even just a single-qubit gate. In the simple 1q case,
    /// the synchronisation is trivial, and the gate will be automatically folded into a parent.
    Synchronize,
    /// A two-qubit interaction.  This is the principal component that we're concerned with.
    TwoQ([VirtualQubit; 2]),
    /// A control-flow operation on more than two qubits.  If it's only on a single qubit, it'll be
    /// handled by `Synchronize`.  If it's on 2q, it'll be handled by `TwoQ`.  We need to store both
    /// the Sabre version and the DAG representation of the block, so we can reconstruct things
    /// later.  When control-flow ops are represented with native DAGs, we won't need to store the
    /// temporary.
    ControlFlow(Box<[(SabreDAG, DAGCircuit)]>),
}
impl InteractionKind {
    fn from_control_flow(cf: ControlFlowView<DAGCircuit>) -> Result<Self, SabreDAGError> {
        let blocks: Box<[_]> = cf
            .blocks()
            .into_iter()
            .map(|dag| Ok((SabreDAG::from_dag(dag)?, dag.clone())))
            .collect::<Result<_, SabreDAGError>>()?;
        Ok(Self::ControlFlow(blocks))
    }

    fn from_op(op: &PackedOperation, qargs: &[Qubit]) -> Result<Self, SabreDAGError> {
        if op.directive() {
            return Ok(Self::Synchronize);
        }
        match qargs {
            // We're assuming that if the instruction has classical wires (like a `PyInstruction` or
            // something), then it's still going to need routing, even though we can't see inside
            // the operation to actually _know_ what it is.
            &[left, right] => Ok(Self::TwoQ([
                VirtualQubit::new(left.0),
                VirtualQubit::new(right.0),
            ])),
            // TODO: multi-q gates _should_ be an error, but for the initial patch, we're
            // maintaining the historical behaviour of Sabre which is to ignore multi-q gates.
            _ => Ok(Self::Synchronize),
        }
    }
}

/// Named access to the node elements in the [SabreDAG].
#[derive(Clone, Debug)]
pub struct SabreNode {
    /// Indices into the original [DAGCircuit] in topological order that become routable once the
    /// node as a whole is routable.
    pub indices: Vec<NodeIndex>,
    pub kind: InteractionKind,
}
impl SabreNode {
    fn new(initial: NodeIndex, kind: InteractionKind) -> Self {
        Self {
            indices: vec![initial],
            kind,
        }
    }
}

#[derive(Error, Debug)]
pub enum SabreDAGError {
    #[error("Python error: {0}")]
    Python(#[from] PyErr),
}
impl From<SabreDAGError> for PyErr {
    fn from(err: SabreDAGError) -> PyErr {
        match err {
            SabreDAGError::Python(err) => err,
        }
    }
}

/// A DAG representation of the logical circuit to be routed.
///
/// This interaction representation retains only information about routing necessities; when
/// possible, nodes in the input [DAGCircuit] are combined into a single node for routing purposes
/// (for example, 1q gates are always folded into a preceding node and runs of 2q gates on the same
/// qubits are combined).
///
/// Note that all the qubit references here are to "virtual" qubits, that is, the qubits are those
/// specified by the user.  This DAG does not need to be full-width on the hardware.
#[derive(Clone, Debug)]
pub struct SabreDAG {
    /// An ordered sequence of node indices in the original [DAGCircuit] that are automatically
    /// routable immediately at the start of the circuit, regardless of layout (for example 1q
    /// gates and directives that are preceded only by 1q gates).
    pub initial: Vec<NodeIndex>,
    pub dag: DiGraph<SabreNode, ()>,
    /// The incoming externals of the `dag`.  These will form the first "front layer" of the Sabre
    /// search (unless they're already routable with the chosen layout or control-flow ops).  These
    /// are indices into the [dag] field.
    pub first_layer: Vec<NodeIndex>,
}

impl SabreDAG {
    pub fn from_dag(dag: &DAGCircuit) -> Result<Self, SabreDAGError> {
        // The `NodeIndex` here is into `dag`.
        let mut initial = Vec::<NodeIndex>::new();
        let mut sabre = DiGraph::new();
        // The `NodeIndex`es in here are into `sabre`.
        let mut wire_pos: HashMap<Wire, NodeIndex> = HashMap::with_capacity(dag.width());
        let mut first_layer = Vec::<NodeIndex>::new();

        enum Predecessors {
            /// All the predecessors of a node are unmapped, so the node is an incoming external.
            AllUnmapped,
            /// All the predecessors of the node are the same concrete node.
            Single(NodeIndex),
            /// The node is not immediately dominated by a single concrete node (i.e. it's got more
            /// than one in edge, or one but not all of its wires are unmapped).
            Multiple,
        }
        let predecessors =
            |dag_node: NodeIndex, sabre_pos: &HashMap<Wire, NodeIndex>| -> Predecessors {
                let mut edges = dag.dag().edges_directed(dag_node, Direction::Incoming);
                let Some(first) = edges.next() else {
                    // 0-arity node, so it's always immediately eligible for routing.
                    return Predecessors::AllUnmapped;
                };
                let single = sabre_pos.get(first.weight()).copied();
                for edge in edges {
                    if single != sabre_pos.get(edge.weight()).copied() {
                        return Predecessors::Multiple;
                    }
                }
                single.map_or(Predecessors::AllUnmapped, Predecessors::Single)
            };

        for dag_node in dag.topological_op_nodes(false) {
            let NodeType::Operation(inst) = &dag[dag_node] else {
                panic!("op nodes should always be of type `Operation`");
            };
            let kind = if let Some(cf) = dag.try_view_control_flow(inst) {
                InteractionKind::from_control_flow(cf)?
            } else {
                InteractionKind::from_op(&inst.op, dag.get_qargs(inst.qubits))?
            };
            match predecessors(dag_node, &wire_pos) {
                Predecessors::AllUnmapped => match kind {
                    InteractionKind::Synchronize => {
                        initial.push(dag_node);
                    }
                    kind => {
                        let sabre_node = sabre.add_node(SabreNode::new(dag_node, kind));
                        first_layer.push(sabre_node);
                        for edge in dag.dag().edges_directed(dag_node, Direction::Incoming) {
                            wire_pos.insert(*edge.weight(), sabre_node);
                        }
                    }
                },
                Predecessors::Multiple => {
                    // It's possible that `kind` is `Synchronize` here and it only has a single
                    // routing-enforced predecessor (and other wires that are automatically
                    // satisfied). In that case, _technically_ we could fold it onto the single
                    // constraining predecessor, but failing to do that doesn't cause correctness
                    // issues, makes the logic easier, and will have next-to-no performance cost.
                    let sabre_node = sabre.add_node(SabreNode::new(dag_node, kind));
                    for edge in dag.dag().edges_directed(dag_node, Direction::Incoming) {
                        if let Some(parent) = wire_pos.insert(*edge.weight(), sabre_node) {
                            sabre.add_edge(parent, sabre_node, ());
                        }
                    }
                }
                Predecessors::Single(prev_sabre_node) => {
                    let prev = sabre
                        .node_weight_mut(prev_sabre_node)
                        .expect("derived from 'edges_directed'");
                    match (&prev.kind, kind) {
                        // A "synchronise" that only has one predecessor isn't actually imposing any
                        // synchronisation.  If a 2q gate depends only on another 2q gate, they've
                        // got to be the same qubits, and therefore it's automatically routable.
                        (_, InteractionKind::Synchronize)
                        | (InteractionKind::TwoQ(_), InteractionKind::TwoQ(_)) => {
                            prev.indices.push(dag_node);
                        }
                        // Otherwise we need the router to evaluate it.
                        (_, kind) => {
                            let sabre_node = sabre.add_node(SabreNode::new(dag_node, kind));
                            sabre.add_edge(prev_sabre_node, sabre_node, ());
                            for edge in dag.dag().edges_directed(dag_node, Direction::Incoming) {
                                wire_pos.insert(*edge.weight(), sabre_node);
                            }
                        }
                    }
                }
            }
        }
        Ok(SabreDAG {
            initial,
            dag: sabre,
            first_layer,
        })
    }

    /// Reduce the [SabreDAG] down to only consider the interaction structure.
    ///
    /// The resulting [SabreDAG] contains no actual node indices from the original [DAGCircuit], so
    /// cannot be used to reconstruct an output (you'd just get the swaps, no gates).
    ///
    /// This converts all [InteractionKind::ControlFlow] nodes in the graph into simple
    /// [InteractionKind::Synchronize] entries, because the current Sabre algorithm ensures
    /// control-flow blocks return the layout to their starting position, which makes them
    /// indistinguishable from simple synchronisation points from the top level of routing.
    pub fn only_interactions(&self) -> Self {
        let mut dag = DiGraph::with_capacity(self.dag.node_count(), self.dag.edge_count());
        for node in self.dag.node_weights() {
            let kind = match &node.kind {
                InteractionKind::Synchronize | InteractionKind::TwoQ(_) => node.kind.clone(),
                InteractionKind::ControlFlow(_) => InteractionKind::Synchronize,
            };
            // `NodeWeights` guarantees that the weights come out in index order, so the new indexes
            // must match those in `self`.
            dag.add_node(SabreNode {
                indices: vec![],
                kind,
            });
        }
        for edge in self.dag.edge_references() {
            dag.add_edge(edge.source(), edge.target(), ());
        }
        Self {
            initial: vec![],
            dag,
            first_layer: self.first_layer.clone(),
        }
    }

    pub fn reverse_dag(&self) -> Self {
        let mut out_dag = self.clone();
        out_dag.dag.reverse();
        out_dag.first_layer = out_dag.dag.externals(Direction::Incoming).collect();
        out_dag
    }
}
