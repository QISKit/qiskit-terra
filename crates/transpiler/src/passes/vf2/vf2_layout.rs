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

use std::convert::Infallible;
use std::time::Instant;

use hashbrown::HashMap;
use qiskit_util::{IndexMap, IndexSet};
use rand::prelude::*;
use rand::rngs::SysRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use rustworkx_core::petgraph::data::Create;
use rustworkx_core::petgraph::prelude::*;

use pyo3::prelude::*;
use pyo3::{IntoPyObjectExt, create_exception, wrap_pyfunction};

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::interner::{Interned, Interner};
use qiskit_circuit::operations::{ControlFlowView, Operation};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{PhysicalQubit, VirtualQubit, vf2};

use super::error_map::ErrorMap;
use crate::target::{Qargs, QargsRef, Target, TargetOperation};

create_exception!(qiskit, MultiQEncountered, pyo3::exceptions::PyException);

#[pyclass(name = "VF2PassConfiguration", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct Vf2PassConfiguration {
    /// The maximum numbers of times VF2 is allowed to extend the mapping (before, after) the first
    /// match.  In both cases, `None` means "no limit".  Steps taken before the first match is found
    /// still count against the "after" limit.
    pub call_limit: (Option<usize>, Option<usize>),
    /// Time in seconds to spend optimizing.  This is not a tight limit; control only returns to the
    /// iterator to check for the time limit when a successful _improved_ match is found.
    pub time_limit: Option<f64>,
    /// In `max_trials`, `None` means "based on the number of graph edges" and `0` means
    /// "unbounded".
    pub max_trials: Option<usize>,
    /// If set, shuffle the node indices of the input graphs using a specified random seed.  If
    /// `None`, perform no shuffling.  You probably want this to be `None`.
    pub shuffle_seed: Option<u64>,
    /// Whether the initial "trivial" layout should be immediately scored, as used as the base.  If
    /// true, the result will return [Vf2PassReturn::NoImprovement] if the initial layout was the
    /// best-scoring match.  Scoring the initial layout is useful for seeding the tree-pruner
    /// component of the search, if the incoming layout is expected to already be valid.
    pub score_initial_layout: bool,
}
impl Vf2PassConfiguration {
    /// A set of defaults that just runs everything completely unbounded.
    pub fn default_unbounded() -> Self {
        Self {
            call_limit: (None, None),
            time_limit: None,
            max_trials: Some(0),
            shuffle_seed: None,
            score_initial_layout: false,
        }
    }

    /// A set of defaults suitable for calling the VF2 layout passes in "search" mode for a circuit
    /// that has not been lowered to hardware instructions.
    pub fn default_abstract() -> Self {
        Self {
            call_limit: (None, None),
            time_limit: None,
            // There's no need to attempt to improve the layout, since everything's so approximate
            // anyway.
            max_trials: Some(1),
            shuffle_seed: None,
            score_initial_layout: false,
        }
    }

    /// A set of defaults for calling VF2 passes on circuits that are already lowered to hardware,
    /// and we want to find the _best_ layout.
    pub fn default_concrete() -> Self {
        Self {
            call_limit: (None, None),
            time_limit: None,
            // Unbounded trials.
            max_trials: Some(0),
            shuffle_seed: None,
            score_initial_layout: true,
        }
    }
}
#[pymethods]
impl Vf2PassConfiguration {
    #[new]
    #[pyo3(signature = (*, call_limit=(None, None), time_limit=None, max_trials=None, shuffle_seed=None, score_initial_layout=false))]
    fn py_new(
        call_limit: (Option<usize>, Option<usize>),
        time_limit: Option<f64>,
        max_trials: Option<usize>,
        shuffle_seed: Option<u64>,
        score_initial_layout: bool,
    ) -> Self {
        Self {
            call_limit,
            time_limit,
            max_trials,
            shuffle_seed,
            score_initial_layout,
        }
    }

    /// Construct the VF2 configuration from the legacy interface to the Python passes.
    #[staticmethod]
    #[pyo3(signature = (*, call_limit=None, time_limit=None, max_trials=None, shuffle_seed=None, score_initial_layout=false))]
    fn from_legacy_api(
        call_limit: Option<Bound<PyAny>>,
        time_limit: Option<f64>,
        max_trials: Option<isize>,
        shuffle_seed: Option<i64>,
        score_initial_layout: bool,
    ) -> PyResult<Self> {
        let call_limit = match call_limit {
            Some(call_limit) => {
                if let Ok(call_limit) = call_limit.extract::<usize>() {
                    (Some(call_limit), Some(call_limit))
                } else {
                    call_limit.extract()?
                }
            }
            None => (None, None),
        };
        // In the legacy API, negative `max_trials` means unbounded (which we represent as 0) and
        // `None` means "choose some values based on the size of the graph structures".
        let max_trials = max_trials.map(|value| value.try_into().unwrap_or(0));
        let shuffle_seed = match shuffle_seed {
            None => {
                // In Python space, `None` means "seed with OS entropy" because seeding was expected
                // to be the default.
                Some(Pcg64Mcg::try_from_rng(&mut SysRng).unwrap().next_u64())
            }
            Some(-1) => None,
            Some(seed) => {
                // Python accepted negative seeds other than -1.  Since we're working with
                // fixed-size integers, we'll just reinterpret the bits as a `u64`.
                Some(u64::from_ne_bytes(seed.to_ne_bytes()))
            }
        };
        Ok(Self {
            call_limit,
            time_limit,
            max_trials,
            shuffle_seed,
            score_initial_layout,
        })
    }
}

/// The possible success-path returns from a VF2-layout based path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Vf2PassReturn {
    /// No solution could be found.
    NoSolution,
    /// The configuration asked us to score the identity layout, it resulted in a valid mapping, and
    /// it was the best mapping seen (or at least tied for the best).
    NoImprovement,
    /// A solution was found (other than `NoImprovement`, if relevant).
    Solution(HashMap<VirtualQubit, PhysicalQubit>),
}
impl<'py> IntoPyObject<'py> for Vf2PassReturn {
    type Target = VF2PassReturn;
    type Output = <Self::Target as IntoPyObject<'py>>::Output;
    type Error = <Self::Target as IntoPyObject<'py>>::Error;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        VF2PassReturn(self).into_pyobject(py)
    }
}

#[pyclass]
pub struct VF2PassReturn(Vf2PassReturn);
#[pymethods]
impl VF2PassReturn {
    /// True iff there exists a valid solution.  If ``True``, ``new_mapping`` can still be ``None``
    /// if there was no improvement over the trivial mapping.
    #[getter]
    fn has_solution(&self) -> bool {
        match &self.0 {
            Vf2PassReturn::NoSolution => false,
            Vf2PassReturn::NoImprovement | Vf2PassReturn::Solution(_) => true,
        }
    }
    /// Get the improved mapping, if one exists.  Returns ``None`` if there was no solution or no
    /// improvement.
    fn new_mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.0 {
            Vf2PassReturn::NoSolution | Vf2PassReturn::NoImprovement => {
                Ok(py.None().into_bound(py))
            }
            Vf2PassReturn::Solution(mapping) => mapping.into_bound_py_any(py),
        }
    }
}

/// Build an average error map for a given target.
///
/// Returns `None` if there is a global 2q operation to avoid attempting to construct a meaningless
/// all-to-all connectivity graph.
fn build_average_error_map(target: &Target) -> Option<ErrorMap> {
    if let Ok(mut globals) = target.operations_for_qargs(QargsRef::Global) {
        if globals.any(|op| op.operation.num_qubits() == 2) {
            return None;
        }
    }
    let mut error_map = ErrorMap::new(Some(target.num_qargs()));
    let mut target_without_errors = true;
    for qargs in target.qargs()? {
        let Qargs::Concrete(qargs) = qargs else {
            // TODO: limitations in the Rust-space `Target` don't let us handle errors on
            // globally-defined 1q operations well, and we already handled the case of a 2q global,
            // so we can skip from here.
            continue;
        };
        let mut qarg_error: f64 = 0.;
        let mut count: usize = 0;
        for op in target
            .operation_names_for_qargs(QargsRef::Concrete(qargs))
            .expect("these qargs came from `target.qargs()`")
        {
            count += 1;
            // If the `target` has no error recorded for an operation, we treat it as errorless.
            qarg_error += target
                .get_error(op, qargs)
                .inspect(|_| target_without_errors = false)
                .unwrap_or(0.);
        }
        if count > 0 {
            let out_qargs = if qargs.len() == 1 {
                [qargs[0], qargs[0]]
            } else {
                [qargs[0], qargs[1]]
            };
            error_map
                .error_map
                .insert(out_qargs, qarg_error / count as f64);
        }
    }
    if target_without_errors {
        // Legacy back-up condition, which penalises high-degree nodes.  This was originally part of
        // VF2Layout to better support old IBM devices, like the bowties, which were unreliable at
        // reporting their error rates, but typically had significantly worse performance on the
        // high-degree nodes.
        let num_qubits = target.num_qubits.unwrap() as usize;
        // Using an undirected graph because we don't want to double-count neighbors in directed
        // graphs.  Most directed graphs likely have no reversed edges or are totally symmetric, but
        // this avoids making any assumptions.
        let mut coupling =
            Graph::<(), (), Undirected>::with_capacity(num_qubits, error_map.error_map.len());
        for _ in 0..num_qubits {
            coupling.add_node(());
        }
        for qargs in target.qargs()? {
            let QargsRef::Concrete(&[left, right]) = qargs.as_ref() else {
                continue;
            };
            coupling.update_edge(
                NodeIndex::new(left.index()),
                NodeIndex::new(right.index()),
                (),
            );
        }
        for index in coupling.node_indices() {
            let qubit = PhysicalQubit::new(index.index() as u32);
            let degree = coupling.neighbors(index).count();
            error_map
                .error_map
                .entry([qubit, qubit])
                .insert(degree as f64 / num_qubits as f64);
        }
        // We use the `target` here rather than the graph edges because we want the directionality.
        for qargs in target.qargs()? {
            let QargsRef::Concrete(&[left, right]) = qargs.as_ref() else {
                continue;
            };
            let avg =
                0.5 * (error_map.error_map[&[left, left]] + error_map.error_map[&[right, right]]);
            error_map.error_map.entry([left, right]).insert(avg);
        }
    }
    Some(error_map)
}

/// A full set of a virtual interaction graph, and any loose 1q qubits.
#[derive(Default, Debug, Clone)]
struct VirtualInteractions<T> {
    /// The graph of actual interactions.  Nodes each correspond to 1q operations on a single
    /// virtual qubit (the mapping `nodes` stores _which_ qubits), and edges between virtual qubits
    /// correspond to 2q operations.  Edges are always directed, even for when `strict_direction` is
    /// unset; we handle the fuzzy directional matching by setting the edge weights of the coupling
    /// graph appropriately.
    graph: Graph<T, T>,
    /// Map of node index to the qubit it represents.  We could store this on the nodes themselves,
    /// but then all the scorers would need different weight types between the nodes and the edges.
    nodes: IndexSet<VirtualQubit>,
    /// The qubits that have only single-qubit operations on them, mapped to the interaction summary
    /// associated with them.  We iterate through this at the end, so need a consistent order.
    uncoupled: IndexMap<VirtualQubit, T>,
    /// The qubits that have no operations on them at all.
    idle: IndexSet<VirtualQubit>,
}
impl<T: Default> VirtualInteractions<T> {
    /// Create a set of virtual interactions from a DAG, and a weighter function for a single
    /// interaction.
    ///
    /// The weighter should return `true` on success, or `false` if no weight could be created
    /// (indicating a necessary failure of the layout pass).
    fn try_from_dag<W>(dag: &DAGCircuit, weighter: W) -> PyResult<Option<Self>>
    where
        W: Fn(&mut T, &PackedInstruction, usize) -> bool,
    {
        let id_qubit_map = (0..dag.num_qubits())
            .map(|q| VirtualQubit(q as u32))
            .collect::<Vec<_>>();
        let mut out = Self::default();
        if !out.add_interactions_from(dag, &id_qubit_map, 1, &weighter)? {
            return Ok(None);
        }
        out.idle.extend(
            (0..dag.num_qubits() as u32)
                .map(VirtualQubit)
                .filter(|q| !(out.nodes.contains(q) || out.uncoupled.contains_key(q))),
        );
        Ok(Some(out))
    }

    /// Create a set of virtual interactions from a DAG, and a weighter function for a single
    /// interaction.
    fn from_dag<W>(dag: &DAGCircuit, weighter: W) -> PyResult<Self>
    where
        W: Fn(&mut T, &PackedInstruction, usize),
    {
        Self::try_from_dag(dag, |weight, inst, count| {
            weighter(weight, inst, count);
            true
        })
        .map(|res| res.expect("weighter is infallible so should always produce interactions"))
    }

    /// Add interactions from a given DAG.  Returns `false` if the weighter ever returned `false`.
    fn add_interactions_from<W>(
        &mut self,
        dag: &DAGCircuit,
        wire_map: &[VirtualQubit],
        repeats: usize,
        weighter: &W,
    ) -> PyResult<bool>
    where
        W: Fn(&mut T, &PackedInstruction, usize) -> bool,
    {
        for (_, inst) in dag.op_nodes(false) {
            let qubits = dag.get_qargs(inst.qubits);
            if let Some(control_flow) = dag.try_view_control_flow(inst) {
                let repeats = if let ControlFlowView::ForLoop { collection, .. } = control_flow {
                    repeats * collection.len()
                } else {
                    repeats
                };
                let wire_map: Vec<_> = qubits.iter().map(|i| wire_map[i.index()]).collect();
                for block in control_flow.blocks() {
                    if !self.add_interactions_from(block, &wire_map, repeats, weighter)? {
                        return Ok(false);
                    }
                }
                continue;
            }
            match qubits {
                [] => (),
                [q] => {
                    let q = wire_map[q.index()];
                    if let Some(index) = self.nodes.get_index_of(&q) {
                        let weight = self
                            .graph
                            .node_weight_mut(NodeIndex::new(index))
                            .expect("node must be in graph if tracked in 'nodes'");
                        if !weighter(weight, inst, repeats) {
                            return Ok(false);
                        };
                    } else {
                        let weight = self.uncoupled.entry(q).or_default();
                        weighter(weight, inst, repeats);
                    }
                }
                [q0, q1] => {
                    let q0 = wire_map[q0.index()];
                    let q1 = wire_map[q1.index()];
                    let node0 = self.ensure_1q_in_graph(q0);
                    let node1 = self.ensure_1q_in_graph(q1);
                    if let Some(edge) = self.graph.find_edge(node0, node1) {
                        let weight = self
                            .graph
                            .edge_weight_mut(edge)
                            .expect("this index came from a call to 'find_edge'");
                        if !weighter(weight, inst, repeats) {
                            return Ok(false);
                        };
                    } else {
                        let mut weight = T::default();
                        if !weighter(&mut weight, inst, repeats) {
                            return Ok(false);
                        };
                        self.graph.add_edge(node0, node1, weight);
                    }
                }
                _ => return Err(MultiQEncountered::new_err("")),
            }
        }
        Ok(true)
    }

    fn ensure_1q_in_graph(&mut self, q: VirtualQubit) -> NodeIndex {
        if let Some(index) = self.nodes.get_index_of(&q) {
            return NodeIndex::new(index);
        }
        assert!(self.nodes.insert(q));
        self.graph
            .add_node(self.uncoupled.swap_remove(&q).unwrap_or_default())
    }
}
impl<T> VirtualInteractions<T> {
    /// Move all the uncoupled qubits into the graph.
    ///
    /// This is useful for the case that the interaction weights have sufficient semantics that we
    /// wouldn't be able to reliably match them separately to the rest of the interactions.
    fn move_all_uncoupled_to_graph(&mut self) {
        for (qubit, interactions) in self.uncoupled.drain(..) {
            assert!(self.nodes.insert(qubit));
            self.graph.add_node(interactions);
        }
    }
}

/// Calculate a safe version of `-ln(1 - e)`, where the result is guaranteed to be finite.
///
/// This treats `nan` as `0.0`, and clamps errors to the `[0.0, 1.0]` interval.  An error of `1.0`
/// should arguably have an infinite cost, but we use `f64::MAX` instead so the score is guaranteed
/// to be finite, and safe to multiply by any other floating-point value.
fn neg_log_fidelity(error: f64) -> f64 {
    if error.is_nan() || error <= 0. {
        0.0
    } else if error >= 1. {
        // Logically this would be cleaner as `INFINITY`, but it's better to allow the downstream
        // scoring to rely on this value being finite, so we don't have to branch in inner-loop
        // scoring code.
        f64::MAX
    } else {
        -((-error).ln_1p())
    }
}

fn build_average_coupling_map(target: &Target, errors: &ErrorMap) -> Option<Graph<f64, f64>> {
    let num_qubits = target.num_qubits.unwrap_or_default() as usize;
    if target.num_qargs() == 0 {
        return None;
    }
    let mut cm_graph =
        Graph::with_capacity(num_qubits, target.num_qargs().saturating_sub(num_qubits));
    for qubit in 0..num_qubits as u32 {
        let qubit = PhysicalQubit::new(qubit);
        cm_graph.add_node(neg_log_fidelity(
            *errors.error_map.get(&[qubit, qubit]).unwrap_or(&0.0),
        ));
    }
    for qargs in target.qargs()? {
        let QargsRef::Concrete(&[left, right]) = qargs.as_ref() else {
            // We just ignore globals; we're assuming that somewhere else has eagerly terminated in
            // the case of global 2q operations.  We also don't care about non-2q operations in this
            // loop (since we already did the single-qubit bits).
            continue;
        };
        let error = errors.error_map.get(&[left, right]).unwrap_or(&0.0);
        cm_graph.add_edge(
            NodeIndex::new(left.index()),
            NodeIndex::new(right.index()),
            neg_log_fidelity(*error),
        );
    }
    Some(cm_graph)
}

#[allow(clippy::type_complexity)]
fn build_exact_coupling_map(
    target: &Target,
) -> Option<(
    Graph<HashMap<Interned<str>, f64>, HashMap<Interned<str>, f64>>,
    Interner<str>,
)> {
    let num_qubits = target.num_qubits.unwrap_or_default() as usize;
    if target.num_qargs() == 0 {
        return None;
    }
    let mut cm_graph: Graph<HashMap<_, _>, HashMap<_, _>> =
        Graph::with_capacity(num_qubits, target.num_qargs().saturating_sub(num_qubits));
    let mut interner = Interner::new();
    for _ in 0..num_qubits {
        cm_graph.add_node(Default::default());
    }
    // If there's _only_ global operations in the target, then either any mapping is valid (with the
    // same score) or no mapping is valid.  Rather than adding a bunch of extra logic to handle
    // that, we just give up; it doesn't make much sense to call VF2Layout/VF2PostLayout in those
    // situations.
    for qargs in target.qargs()? {
        let QargsRef::Concrete(qargs) = qargs.as_ref() else {
            // We'll handle globals afterwards, so we can match the Python-space version's unusual
            // handling of "global" operations.
            continue;
        };
        let instructions = match qargs {
            [qubit] => cm_graph
                .node_weight_mut(NodeIndex::new(qubit.index()))
                .expect("previous loop added all nodes"),
            [left, right] => {
                let (left, right) = (NodeIndex::new(left.index()), NodeIndex::new(right.index()));
                let edge = cm_graph
                    .find_edge(left, right)
                    .unwrap_or_else(|| cm_graph.add_edge(left, right, Default::default()));
                cm_graph
                    .edge_weight_mut(edge)
                    .expect("edge created in previous statement")
            }
            _ => continue,
        };
        for name in target
            .operation_names_for_qargs(QargsRef::Concrete(qargs))
            .expect("these qargs come from `Target::qargs`")
        {
            instructions.insert(
                interner.insert(name),
                neg_log_fidelity(target.get_error(name, qargs).unwrap_or(0.)),
            );
        }
    }
    for name in target
        .operation_names_for_qargs(QargsRef::Global)
        .into_iter()
        .flatten()
    {
        let TargetOperation::Normal(operation) = target
            .operation_from_name(name)
            .expect("name comes from target")
        else {
            // A variadic that's valid globally on both 1q and 2q would have the same effect on
            // the score even if it had an error, regardless of the isomorphism.
            continue;
        };
        // TODO: the `Target` API currently doesn't let us access the error of a global, assuming
        // that it's ideal.  We treat things the same here for now.
        let score = neg_log_fidelity(0.0);
        match operation.operation.num_qubits() {
            1 => {
                let key = interner.insert(name);
                for weight in cm_graph.node_weights_mut() {
                    weight.insert(key, score);
                }
            }
            2 => {
                // TODO: the Python-space version of `VF2PostLayout` in strict mode has an unusual
                // interpretation of "global" 2q operations; it defines the operation on all 2q
                // links that _also_ have concrete instructions.  For now, we replicate that.
                let key = interner.insert(name);
                for weight in cm_graph.edge_weights_mut() {
                    weight.insert(key, score);
                }
            }
            _ => (),
        }
    }
    Some((cm_graph, interner))
}

/// If an edge does not have a parallel but reversed counterpart, add one with the same weight.
fn loosen_directionality<S, T: Clone>(graph: &mut Graph<S, T>) {
    graph
        .edge_references()
        .filter(|edge| graph.find_edge(edge.target(), edge.source()).is_none())
        .map(|edge| (edge.target(), edge.source(), edge.weight().clone()))
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|(source, target, weight)| {
            graph.add_edge(source, target, weight);
        })
}

// This function assumes that there's a way of "sorting" the 1q interactions relative to each other,
// so it only really makes sense for the VF2 averaging, not VF2Post exact matching.
/// Assign the free isolated qubits to the physical qubits with the best error rates.
fn map_free_qubits(
    num_physical_qubits: usize,
    interactions: VirtualInteractions<usize>,
    mut partial_layout: HashMap<VirtualQubit, PhysicalQubit>,
    avg_error_map: &ErrorMap,
) -> Option<HashMap<VirtualQubit, PhysicalQubit>> {
    if num_physical_qubits
        < partial_layout.len() + interactions.uncoupled.len() + interactions.idle.len()
    {
        return None;
    }

    let normalize = |err: Option<&f64>| -> f64 {
        let err = err.copied().unwrap_or(f64::INFINITY);
        if err.is_nan() { 0.0 } else { err }
    };

    let mut free_physical = (0..num_physical_qubits)
        .map(|qubit| PhysicalQubit::new(qubit as u32))
        .collect::<IndexSet<_>>();
    partial_layout.values().for_each(|phys| {
        free_physical.swap_remove(phys);
    });
    free_physical.par_sort_by(|a, b| {
        let score_a = normalize(avg_error_map.error_map.get(&[*a, *a]));
        let score_b = normalize(avg_error_map.error_map.get(&[*b, *b]));
        score_a.partial_cmp(&score_b).expect("NaNs treated as zero")
    });

    let mut uncoupled_virtual = interactions.uncoupled.into_iter().collect::<Vec<_>>();
    uncoupled_virtual.par_sort_by_key(|(_, interactions)| *interactions);
    partial_layout.extend(
        uncoupled_virtual
            .into_iter()
            .rev() // We want the most used virtuals to get first pick.
            .map(|(virt, _)| virt)
            .chain(interactions.idle)
            .zip(free_physical),
    );
    Some(partial_layout)
}

fn minimize_vf2<N, H, NG, HG, NO, HO, NS, ES>(
    vf2: vf2::Vf2<N, H, NG, HG, NO, HO, NS, ES>,
    config: &Vf2PassConfiguration,
) -> Option<IndexMap<N::NodeId, H::NodeId>>
where
    N: vf2::alias::IntoVf2Graph,
    H: vf2::alias::IntoVf2Graph<EdgeType = N::EdgeType>,
    NG: for<'a> vf2::alias::Vf2Graph<
            'a,
            NodeWeight = N::NodeWeight,
            EdgeWeight = N::EdgeWeight,
            EdgeType = N::EdgeType,
        > + Create,
    HG: for<'a> vf2::alias::Vf2Graph<
            'a,
            NodeWeight = H::NodeWeight,
            EdgeWeight = H::EdgeWeight,
            EdgeType = H::EdgeType,
        > + Create,
    NO: vf2::NodeSorter<N>,
    HO: vf2::NodeSorter<H>,
    NS: vf2::Semantics<N::NodeWeight, H::NodeWeight, Error = Infallible>,
    ES: vf2::Semantics<N::EdgeWeight, H::EdgeWeight, Score = NS::Score, Error = Infallible>,
{
    let start_time = Instant::now();
    let mut times_up = false;
    let mut trials: usize = 0;
    let max_trials = config
        .max_trials
        .unwrap_or_else(|| 15 + vf2.needle().edge_count().max(vf2.haystack().edge_count()));
    let time_limit = config.time_limit.unwrap_or(f64::INFINITY);
    let mut can_continue = || {
        if times_up {
            return false;
        }
        times_up = start_time.elapsed().as_secs_f64() >= time_limit;
        trials += 1;
        max_trials == 0 || trials <= max_trials
    };
    let mut vf2 = vf2.with_call_limit(config.call_limit.0).into_iter();
    let Ok((mut mapping, _score)) = vf2.next()?;
    if can_continue() {
        vf2.call_limit = config.call_limit.1;
        if let Some(result) = vf2.take_while(|_| can_continue()).last() {
            let Ok((new_mapping, _)) = result;
            mapping = new_mapping;
        }
    }
    Some(mapping)
}

/// Produce an initial score for the identity mapping of the interaction graph onto the coupling
/// graph.
///
///
/// This assumes that the input graphs are not multigraphs; the results will certainly be incorrect
/// if the virtual-interactions graph is a multigraph.
fn score_identity_layout<S, T, W>(
    interactions: &VirtualInteractions<S>,
    coupling: &Graph<T, T>,
    scorer: W,
) -> Option<W::Score>
where
    W: vf2::Semantics<S, T, Error = Infallible>,
{
    use vf2::Vf2Score;

    // Map from nodes in the interactions graph to the coupling graph.
    let node_map = |node: NodeIndex| -> NodeIndex {
        NodeIndex::new(
            interactions
                .nodes
                .get_index(node.index())
                .expect("all nodes should have an entry")
                .index(),
        )
    };
    let mut score = W::Score::id();
    for node in interactions.graph.node_indices() {
        let needle = interactions.graph.node_weight(node)?;
        let haystack = coupling.node_weight(node_map(node))?;
        let Ok(partial) = scorer.score(needle, haystack);
        score = W::Score::combine(&score, &partial?);
    }
    for edge in interactions.graph.edge_references() {
        let needle = edge.weight();
        // Making a strong assumption that the virtual interactions are not a multigraph here.
        let haystack = coupling
            .edges_connecting(node_map(edge.source()), node_map(edge.target()))
            .next()?
            .weight();
        let Ok(partial) = scorer.score(needle, haystack);
        score = W::Score::combine(&score, &partial?);
    }
    Some(score)
}

#[pyfunction]
#[pyo3(signature = (dag, target, config, *, strict_direction=false, avg_error_map=None))]
pub fn vf2_layout_pass_average(
    dag: &DAGCircuit,
    target: &Target,
    config: &Vf2PassConfiguration,
    strict_direction: bool,
    avg_error_map: Option<&ErrorMap>,
) -> PyResult<Vf2PassReturn> {
    let add_interaction = |count: &mut usize, _: &PackedInstruction, repeats: usize| {
        *count += repeats;
    };
    let interactions = VirtualInteractions::from_dag(dag, add_interaction)?;
    let score =
        |count: &usize, err: &f64| -> Result<f64, Infallible> { Ok(*err * (*count as f64)) };
    let target_error_map = avg_error_map
        .is_none()
        .then(|| build_average_error_map(target))
        .flatten();
    let Some(avg_error_map) = avg_error_map.or(target_error_map.as_ref()) else {
        return Ok(Vf2PassReturn::NoSolution);
    };
    let Some(mut coupling_graph) = build_average_coupling_map(target, avg_error_map) else {
        return Ok(Vf2PassReturn::NoSolution);
    };
    if !strict_direction {
        loosen_directionality(&mut coupling_graph);
    }
    let best_score = if config.score_initial_layout {
        score_identity_layout(&interactions, &coupling_graph, vf2::Scorer(score))
    } else {
        None
    };
    let num_physical_qubits = coupling_graph.node_count();
    let mut coupling_qubits = (0..num_physical_qubits)
        .map(|k| PhysicalQubit::new(k as u32))
        .collect::<Vec<_>>();
    if let Some(seed) = config.shuffle_seed {
        coupling_qubits.shuffle(&mut Pcg64Mcg::seed_from_u64(seed));
        let order = coupling_qubits
            .iter()
            .map(|qubit| NodeIndex::new(qubit.index()))
            .collect::<Vec<_>>();
        coupling_graph = vf2::reorder_nodes(&coupling_graph, &order);
    }

    let vf2 = vf2::Vf2::new(&interactions.graph, &coupling_graph, vf2::Problem::Subgraph)
        .with_scoring(score, score)
        .with_restriction(vf2::Restriction::Decreasing(best_score))
        .with_vf2pp_ordering();
    let Some(mapping) = minimize_vf2(vf2, config) else {
        if best_score.is_some() {
            return Ok(Vf2PassReturn::NoImprovement);
        } else {
            return Ok(Vf2PassReturn::NoSolution);
        }
    };
    // Remap node indices back to virtual/physical qubits.
    let mapping = mapping
        .iter()
        .map(|(k, v)| (interactions.nodes[k.index()], coupling_qubits[v.index()]))
        .collect();
    match map_free_qubits(num_physical_qubits, interactions, mapping, avg_error_map) {
        Some(mapping) => Ok(Vf2PassReturn::Solution(mapping)),
        None => Ok(Vf2PassReturn::NoSolution),
    }
}

#[pyfunction]
#[pyo3(signature = (dag, target, config))]
pub fn vf2_layout_pass_exact(
    dag: &DAGCircuit,
    target: &Target,
    config: &Vf2PassConfiguration,
) -> PyResult<Vf2PassReturn> {
    let Some((mut coupling_graph, interner)) = build_exact_coupling_map(target) else {
        return Ok(Vf2PassReturn::NoSolution);
    };
    let add_interaction =
        |uses: &mut Vec<(Interned<str>, usize)>, inst: &PackedInstruction, repeats: usize| {
            let Some(key) = interner.try_key(inst.op.name()) else {
                return false;
            };
            if let Some((_, count)) = uses.iter_mut().find(|(name, _)| key == *name) {
                *count += repeats;
            } else {
                uses.push((key, repeats));
            }
            true
        };
    let Some(mut interactions) = VirtualInteractions::try_from_dag(dag, add_interaction)? else {
        return Ok(Vf2PassReturn::NoSolution);
    };

    // The optimisation we have in the "average" case where we assign loose 1q gates after matching
    // the rest of the graph doesn't hold in the "exact" case, so we have to have VF2 match them all
    // at once.  For example, consider a heterogeneous target where only one qubit has `rx`
    // available, but VF2 matches a non-`rx`-using qubit onto that one during the first step,
    // because it isn't aware there's a necessary `rx` gate in the free list.
    if interactions.uncoupled.len() > 1 {
        // ... but if there's too many uncoupled qubits, there's a combinatorial explosion in the
        // matching time, so we just bail out.  It would be nice to deal with this better.
        return Ok(Vf2PassReturn::NoSolution);
    }
    interactions.move_all_uncoupled_to_graph();
    let score = |counts: &Vec<(Interned<str>, usize)>,
                 errs: &HashMap<Interned<str>, f64>|
     -> Result<Option<f64>, Infallible> {
        Ok(counts.iter().try_fold(0.0, |tot, (key, count)| {
            errs.get(key).map(|err| tot + err * *count as f64)
        }))
    };
    let best_score = if config.score_initial_layout {
        score_identity_layout(&interactions, &coupling_graph, score)
    } else {
        None
    };
    let num_physical_qubits = coupling_graph.node_count();
    let mut coupling_qubits = (0..num_physical_qubits)
        .map(|k| PhysicalQubit::new(k as u32))
        .collect::<Vec<_>>();
    if let Some(seed) = config.shuffle_seed {
        coupling_qubits.shuffle(&mut Pcg64Mcg::seed_from_u64(seed));
        let order = coupling_qubits
            .iter()
            .map(|qubit| NodeIndex::new(qubit.index()))
            .collect::<Vec<_>>();
        coupling_graph = vf2::reorder_nodes(&coupling_graph, &order);
    }
    let vf2 = vf2::Vf2::new(&interactions.graph, &coupling_graph, vf2::Problem::Subgraph)
        .with_semantics(score, score)
        .with_restriction(vf2::Restriction::Decreasing(best_score))
        .with_vf2pp_ordering();
    let Some(mapping) = minimize_vf2(vf2, config) else {
        if best_score.is_some() {
            return Ok(Vf2PassReturn::NoImprovement);
        } else {
            return Ok(Vf2PassReturn::NoSolution);
        }
    };
    // Remap node indices back to virtual/physical qubits.
    // Keep track of which PhysicalQubits are left idle.
    let mut coupling_indices: Vec<bool> = vec![true; coupling_qubits.len()];
    let mut mapping: HashMap<VirtualQubit, PhysicalQubit> = mapping
        .iter()
        .map(|(k, v)| {
            coupling_indices[v.index()] = false;
            (interactions.nodes[k.index()], coupling_qubits[v.index()])
        })
        .collect();
    // Use the idle virtual qubits and map them to the idle physical ones.
    mapping.extend(
        interactions.idle.iter().copied().zip(
            coupling_indices
                .iter()
                .enumerate()
                .filter_map(|(idx, val)| val.then_some(coupling_qubits[idx])),
        ),
    );
    Ok(Vf2PassReturn::Solution(mapping))
}

pub fn vf2_layout_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(vf2_layout_pass_average))?;
    m.add_wrapped(wrap_pyfunction!(vf2_layout_pass_exact))?;
    m.add("MultiQEncountered", m.py().get_type::<MultiQEncountered>())?;
    m.add(
        "VF2PassConfiguration",
        m.py().get_type::<Vf2PassConfiguration>(),
    )?;
    m.add("VF2PassReturn", m.py().get_type::<VF2PassReturn>())?;
    Ok(())
}
