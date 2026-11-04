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

// This module was originally forked into Rustworkx from petgraph's isomorphism module @ v0.5.0
// to handle Rustworkx PyDiGraph inputs instead of petgraph's generic Graph.  It was modified
// substantially in Rustworkx.  It was then forked from Rustworkx to Qiskit from
//
//    https://github.com/Qiskit/rustworkx/blob/9f0646e8886cfecc55e59b96532c6f7f798524c0/src/isomorphism/vf2.rs
//
// It has subsequently been near-completely written within Qiskit.

use std::cmp::{Ordering, Reverse};
use std::convert::Infallible;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::iter::Iterator;
use std::marker;
use std::num::NonZero;

use hashbrown::{HashMap, hash_map::Entry};
use qiskit_util::IndexMap;
use smallvec::SmallVec;

use rustworkx_core::petgraph::data::Create;
use rustworkx_core::petgraph::graph::IndexType;
use rustworkx_core::petgraph::visit::{
    EdgeRef, GraphBase, GraphProp, IntoNeighborsDirected, IntoNodeIdentifiers,
    NodeCompactIndexable, NodeCount, NodeIndexable,
};
use rustworkx_core::petgraph::{Direction, Graph, Incoming, Outgoing};

pub mod alias {
    use std::hash::Hash;

    use rustworkx_core::petgraph::Direction;
    use rustworkx_core::petgraph::data::DataMap;
    use rustworkx_core::petgraph::graph::IndexType;
    use rustworkx_core::petgraph::visit::{
        Data, EdgeCount, EdgeRef, GraphBase, GraphProp, GraphRef, IntoEdgeReferences,
        IntoEdgesDirected, IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers,
        NodeCompactIndexable, NodeCount, NodeIndexable,
    };

    pub trait IntoVf2Graph:
        GraphBase<NodeId: Hash + Eq + 'static, EdgeId: 'static>
        + DataMap<NodeWeight: Clone + 'static, EdgeWeight: Clone + 'static>
        + EdgeCount
        + GraphProp<EdgeType: 'static>
        + GraphRef
        + IntoEdgeReferences
        + IntoNeighborsDirected
        + IntoNodeIdentifiers
        + NodeCount
        + NodeIndexable
    {
    }
    impl<G> IntoVf2Graph for G where
        G: GraphBase<NodeId: Hash + Eq + 'static, EdgeId: 'static>
            + DataMap<NodeWeight: Clone + 'static, EdgeWeight: Clone + 'static>
            + EdgeCount
            + GraphProp<EdgeType: 'static>
            + GraphRef
            + IntoEdgeReferences
            + IntoNeighborsDirected
            + IntoNodeIdentifiers
            + NodeCount
            + NodeIndexable
    {
    }

    /// This is _intended_ to be a metatrait that just defines a bunch of bounds on references to
    /// implementers of [Vf2Graph].  Unfortunately, I couldn't get a `where &'a Self` bound on
    /// [Vf2Graph] itself to work correctly with the blanket implementation, so I got stuck writing
    /// this boilerplate that duplicates the trait into a lifetime-bound one that we can then use in
    /// higher-ranked trait bounds.
    pub trait Vf2GraphRef<'a>: GraphBase + Data
    where
        Self: 'a,
    {
        type EdgeRef: EdgeRef<NodeId = Self::NodeId, EdgeId = Self::EdgeId, Weight = Self::EdgeWeight>;
        type EdgeReferences: Iterator<Item = Self::EdgeRef>;
        type EdgesDirected: Iterator<Item = Self::EdgeRef>;
        type Neighbors: Iterator<Item = Self::NodeId>;
        type NeighborsDirected: Iterator<Item = Self::NodeId>;

        fn edge_references(&'a self) -> Self::EdgeReferences;
        fn edges_directed(
            &'a self,
            node: Self::NodeId,
            direction: Direction,
        ) -> Self::EdgesDirected;
        fn neighbors(&'a self, node: Self::NodeId) -> Self::Neighbors;
        fn neighbors_directed(
            &'a self,
            node: Self::NodeId,
            direction: Direction,
        ) -> Self::NeighborsDirected;
    }
    impl<'a, G> Vf2GraphRef<'a> for G
    where
        G: GraphBase + Data + 'a,
        &'a G: GraphBase<NodeId = G::NodeId, EdgeId = G::EdgeId>
            + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>
            + IntoEdgesDirected,
    {
        type EdgeRef = <&'a G as IntoEdgeReferences>::EdgeRef;
        type EdgeReferences = <&'a G as IntoEdgeReferences>::EdgeReferences;
        type EdgesDirected = <&'a G as IntoEdgesDirected>::EdgesDirected;
        type Neighbors = <&'a G as IntoNeighbors>::Neighbors;
        type NeighborsDirected = <&'a G as IntoNeighborsDirected>::NeighborsDirected;

        fn edge_references(&'a self) -> Self::EdgeReferences {
            IntoEdgeReferences::edge_references(self)
        }
        fn edges_directed(&'a self, node: G::NodeId, direction: Direction) -> Self::EdgesDirected {
            IntoEdgesDirected::edges_directed(self, node, direction)
        }
        fn neighbors(&'a self, node: G::NodeId) -> Self::Neighbors {
            IntoNeighbors::neighbors(self, node)
        }
        fn neighbors_directed(
            &'a self,
            node: G::NodeId,
            direction: Direction,
        ) -> Self::NeighborsDirected {
            IntoNeighborsDirected::neighbors_directed(self, node, direction)
        }
    }

    /// The operations that a graph must implement to be be
    pub trait Vf2Graph<'a>:
        GraphProp<NodeId: IndexType> + DataMap + EdgeCount + NodeCompactIndexable + Vf2GraphRef<'a>
    {
    }
    impl<'a, G> Vf2Graph<'a> for G where
        G: GraphProp<NodeId: IndexType>
            + DataMap<NodeWeight: 'a, EdgeWeight: 'a>
            + EdgeCount
            + NodeCompactIndexable
            + Vf2GraphRef<'a>
    {
    }
}

pub trait NodeSorter<G: GraphBase> {
    /// Produce the priority list that the nodes should be matched in.  The highest priority node
    /// ids should be first in the output.
    fn sort(&self, _: G) -> Vec<G::NodeId>;
}

/// Rearrange the nodes in `graph` so that the id in position `i` in `order` has index `i` in the
/// output graph.
pub fn reorder_nodes<In, Out>(graph: In, order: &[In::NodeId]) -> Out
where
    In: alias::IntoVf2Graph,
    Out: Create<NodeWeight = In::NodeWeight, EdgeWeight = In::EdgeWeight>
        + GraphProp<EdgeType = In::EdgeType>
        + NodeCompactIndexable,
{
    let node_count = graph.node_count();
    let mut new_graph = Out::with_capacity(node_count, graph.edge_count());
    let mut id_map = HashMap::with_capacity(node_count);
    for &node_index in order {
        let node_data = graph.node_weight(node_index).unwrap();
        let new_index = new_graph.add_node(node_data.clone());
        id_map.insert(graph.to_index(node_index), new_index);
    }
    for edge in graph.edge_references() {
        let edge_w = edge.weight();
        let p_index = id_map[&graph.to_index(edge.source())];
        let c_index = id_map[&graph.to_index(edge.target())];
        new_graph.add_edge(p_index, c_index, edge_w.clone());
    }
    new_graph
}

/// Sort nodes based on node ids.
pub struct DefaultIdSorter;
impl<G> NodeSorter<G> for DefaultIdSorter
where
    // This bound could probably be relaxed to `NodeIndexable + DataMap` because we assume in
    // [reorder_nodes] that we can produce a [Vec] (potentially with holes) over the node indices.
    G: GraphBase + IntoNodeIdentifiers,
{
    fn sort(&self, graph: G) -> Vec<G::NodeId> {
        graph.node_identifiers().collect()
    }
}

/// Sort nodes based on VF2++ heuristic.
pub struct Vf2ppSorter;
impl<G> NodeSorter<G> for Vf2ppSorter
where
    G: GraphProp + IntoNodeIdentifiers + NodeIndexable + NodeCount + IntoNeighborsDirected,
{
    fn sort(&self, graph: G) -> Vec<G::NodeId> {
        let max_nodes = graph.node_bound();

        let degree_out = (0..max_nodes)
            .map(|idx| {
                graph
                    .neighbors_directed(graph.from_index(idx), Outgoing)
                    .count()
            })
            .collect::<Vec<_>>();
        let degree_in = if graph.is_directed() {
            (0..max_nodes)
                .map(|idx| {
                    graph
                        .neighbors_directed(graph.from_index(idx), Incoming)
                        .count()
                })
                .collect()
        } else {
            vec![0; max_nodes]
        };

        let mut conn_in: Vec<usize> = vec![0; max_nodes];
        let mut conn_out: Vec<usize> = vec![0; max_nodes];

        let mut order = Vec::with_capacity(graph.node_count());

        // Process BFS level
        let mut process = |mut vd: Vec<G::NodeId>| -> Vec<G::NodeId> {
            // repeatedly bring largest element in front.
            for i in 0..vd.len() {
                let (index, &item) = vd[i..]
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, node)| {
                        let index = graph.to_index(*node);
                        (
                            conn_in[index] + conn_out[index],
                            degree_out[index] + degree_in[index],
                            Reverse(index),
                        )
                    })
                    .expect("the slice is guaranteed to be non-empty by the loop condition");

                vd.swap(i, i + index);
                order.push(item);

                for neigh in graph.neighbors_directed(item, Outgoing) {
                    conn_in[graph.to_index(neigh)] += 1;
                }
                if graph.is_directed() {
                    for neigh in graph.neighbors_directed(item, Incoming) {
                        conn_out[graph.to_index(neigh)] += 1;
                    }
                }
            }
            vd
        };

        let mut seen: Vec<bool> = vec![false; max_nodes];

        // Create BFS Tree from root and process each level.
        let mut bfs_tree = |root: G::NodeId| {
            let root_index = graph.to_index(root);
            if seen[root_index] {
                return;
            }
            seen[root_index] = true;
            let mut next_level = Vec::new();
            next_level.push(root);
            while !next_level.is_empty() {
                let this_level = process(next_level);

                next_level = Vec::new();
                for bfs_node in this_level {
                    for neighbor in graph.neighbors_directed(bfs_node, Outgoing) {
                        let neighbor_index = graph.to_index(neighbor);
                        if !seen[neighbor_index] {
                            next_level.push(neighbor);
                        }
                        seen[neighbor_index] = true;
                    }
                    if graph.is_directed() {
                        for neighbor in graph.neighbors_directed(bfs_node, Incoming) {
                            let neighbor_index = graph.to_index(neighbor);
                            if !seen[neighbor_index] {
                                next_level.push(neighbor);
                            }
                            seen[neighbor_index] = true;
                        }
                    }
                }
            }
        };

        let mut sorted_nodes = graph.node_identifiers().collect::<Vec<_>>();
        sorted_nodes.sort_by_key(|&node| {
            let index = graph.to_index(node);
            Reverse((degree_out[index] + degree_in[index], Reverse(index)))
        });
        for node in sorted_nodes {
            bfs_tree(node);
        }
        order
    }
}

#[derive(Debug)]
pub enum IsIsomorphicError<NME, EME> {
    NodeMatcher(NME),
    EdgeMatcher(EME),
}
impl<NME: Error, EME: Error> Display for IsIsomorphicError<NME, EME> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            IsIsomorphicError::NodeMatcher(e) => write!(f, "Node match callback failed with: {e}"),
            IsIsomorphicError::EdgeMatcher(e) => write!(f, "Edge match callback failed with: {e}"),
        }
    }
}
impl<NME: Error, EME: Error> Error for IsIsomorphicError<NME, EME> {}

// The `Clone` impl _could_ be spelled `combine(&self, &Self::id())`, but why bother?
pub trait Vf2Score: Clone {
    /// Combine two scores together.  This is typically addition.
    ///
    /// The combination of two scores must not be less than either, as measured by `cmp`.
    fn combine(left: &Self, right: &Self) -> Self;
    /// The identity value of the `combine` operation.
    fn id() -> Self;
    /// Compare two values.
    fn cmp(left: &Self, right: &Self) -> Ordering;
}

impl Vf2Score for () {
    #[inline]
    fn combine(_: &(), _: &()) {}
    #[inline]
    fn id() {}
    #[inline]
    fn cmp(_: &(), _: &()) -> Ordering {
        Ordering::Equal
    }
}
macro_rules! impl_vf2_score_int {
    ($($t:ty)*) => ($(
        impl Vf2Score for $t {
            #[inline]
            fn combine(left: &Self, right: &Self) -> Self {
                *left + *right
            }
            #[inline]
            fn id() -> Self {
                0
            }
            #[inline]
            fn cmp(left: &Self, right: &Self) -> Ordering {
                left.cmp(right)
            }
        }
    )*)
}
impl_vf2_score_int! { u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize }
macro_rules! impl_vf2_score_float {
    ($($t:ty)*) => ($(
        impl Vf2Score for $t {
            #[inline]
            fn combine(left: &Self, right: &Self) -> Self {
                *left + *right
            }
            #[inline]
            fn id() -> Self {
                0.0
            }
            #[inline]
            fn cmp(left: &Self, right: &Self) -> Ordering {
                left.partial_cmp(right).expect("float scores must not return nan")
            }
        }
    )*)
}
impl_vf2_score_float! { f32 f64 }

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SemanticType {
    /// The matcher function would always match and its scores are meaningless; only the
    /// non-semantic structure of the graph is important.
    Disabled,
    /// The matcher function will always match, but its scores are meaningful.  The semantics
    /// of the graph is not important for correctness, only for assigning scores.
    Score,
    /// The structure alone is insufficient to tell if a potential pair are valid to map to each
    /// other; the matching function must be called to validate matches.  This is the strictest and
    /// the slowest, because it requires pairwise comparisons in all cases, even when only a single
    /// isomorphism is requested.
    Semantic,
}

/// Semantic matching for VF2.
///
/// The semantic scoring is implemented for nodes and edges separately.  The degree of matching is
/// set by the [SemanticType] enum, which allows the VF2 algorithm to optimise itself for certain
/// operations, if the full semantics are not necessary.
///
/// The scoring functions do not have access to the general graph structures; they are only
/// permitted to access the weights.  This is deliberate; the scores cannot be suitably
/// combined and pruned if they are anything other than completely local.
pub trait Semantics<N, H> {
    type Score: Vf2Score;
    type Error: Error;
    const MATCH: SemanticType;
    fn score(&self, needle: &N, haystack: &H) -> Result<Option<Self::Score>, Self::Error>;
}

/// Explicitly provide no enforced semantics for node- or edge-matching.
///
/// Typically you don't need to construct this at all, but you can do if you only want to apply
/// semantics to one of the nodes or edges when calling [Vf2::with_semantics].  In that case, use
/// [NoSemantics::new] and let type inference handle the score typing for you.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoSemantics<S>(marker::PhantomData<S>);
impl<S> NoSemantics<S> {
    pub fn new() -> Self {
        Self(marker::PhantomData)
    }
}
impl<N, H, S: Vf2Score> Semantics<N, H> for NoSemantics<S> {
    type Score = S;
    type Error = Infallible;
    const MATCH: SemanticType = SemanticType::Disabled;
    #[inline(always)]
    fn score(&self, _: &N, _: &H) -> Result<Option<Self::Score>, Self::Error> {
        Ok(Some(Self::Score::id()))
    }
}

/// Semantics for the VF2 algorithm that produce a score for any candidate node pair / edge pair
/// that is structurally valid in a partial mapping.
///
/// Typically this is constructed automatically by [Vf2::with_scoring], but you can also construct
/// it directly with a scoring function.
pub struct Scorer<F>(pub F);
impl<F, N, H, S, E> Semantics<N, H> for Scorer<F>
where
    F: Fn(&N, &H) -> Result<S, E>,
    S: Vf2Score,
    E: Error,
{
    type Score = S;
    type Error = E;
    const MATCH: SemanticType = SemanticType::Score;
    #[inline(always)]
    fn score(&self, needle: &N, haystack: &H) -> Result<Option<Self::Score>, Self::Error> {
        (self.0)(needle, haystack).map(|score| Some(score))
    }
}

/// Semantics for the VF2 algorithm that match node- or edge-pairs without an associated score.
///
/// Typically this is constructed automatically by [Vf2::with_matching] or
/// [Vf2::with_node_matching], but you can also construct it directly with a matching function.
pub struct Matcher<F>(pub F);
impl<F, N, H, E> Semantics<N, H> for Matcher<F>
where
    F: Fn(&N, &H) -> Result<bool, E>,
    E: Error,
{
    type Score = ();
    type Error = E;
    const MATCH: SemanticType = SemanticType::Semantic;
    #[inline(always)]
    fn score(&self, needle: &N, haystack: &H) -> Result<Option<Self::Score>, Self::Error> {
        (self.0)(needle, haystack).map(|is_match| is_match.then_some(()))
    }
}

/// Implementation of full semantic matching for an arbitrary function.
///
/// This simply avoids needing an extra helper struct in [Vf2::with_semantics], if the function is
/// already in the exact required form.
impl<F, N, H, S, E> Semantics<N, H> for F
where
    F: Fn(&N, &H) -> Result<Option<S>, E>,
    S: Vf2Score,
    E: Error,
{
    type Score = S;
    type Error = E;
    const MATCH: SemanticType = SemanticType::Semantic;
    #[inline(always)]
    fn score(&self, needle: &N, haystack: &H) -> Result<Option<Self::Score>, Self::Error> {
        self(needle, haystack)
    }
}

/// Return `true` if the `needle` is isomorphic to `haystack` under the (sub)graph constraints of
/// the [Problem].
pub fn is_isomorphic<N, H>(
    needle: N,
    haystack: H,
    id_order: bool,
    problem: Problem,
    call_limit: Option<usize>,
) -> bool
where
    N: alias::IntoVf2Graph<NodeId: IndexType>,
    H: alias::IntoVf2Graph<NodeId: IndexType, EdgeType = N::EdgeType>,
{
    is_isomorphic_with_semantics(
        needle,
        haystack,
        (NoSemantics::<()>::new(), NoSemantics::<()>::new()),
        id_order,
        problem,
        call_limit,
    )
    .expect("error type is infallible")
}

/// Return `true` if the `needle` is isomorphic to `haystack` under the (sub)graph constraints of
/// the [Problem], using the given semantics.
pub fn is_isomorphic_with_semantics<N, H, NS, ES>(
    needle: N,
    haystack: H,
    semantics: (NS, ES),
    id_order: bool,
    problem: Problem,
    call_limit: Option<usize>,
) -> Result<bool, IsIsomorphicError<NS::Error, ES::Error>>
where
    N: alias::IntoVf2Graph<NodeId: IndexType>,
    H: alias::IntoVf2Graph<NodeId: IndexType, EdgeType = N::EdgeType>,
    NS: Semantics<N::NodeWeight, H::NodeWeight>,
    ES: Semantics<N::EdgeWeight, H::EdgeWeight, Score = NS::Score>,
{
    let vf2 = Vf2::new(needle, haystack, problem)
        .with_call_limit(call_limit)
        .with_semantics(semantics.0, semantics.1);
    if id_order {
        vf2.into_iter()
            .next()
            .map(|res| res.map(|_| true))
            .unwrap_or(Ok(false))
    } else {
        vf2.with_vf2pp_ordering()
            .into_iter()
            .next()
            .map(|res| res.map(|_| true))
            .unwrap_or(Ok(false))
    }
}

/// A description of the isomorphism problem to be solved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Problem {
    /// Exact isomorphism between the two graphs.
    Exact,
    /// The left graph is isomorphic to an induced subgraph of the right graph.
    InducedSubgraph,
    /// The left graph is isomorphic to any subgraph of the right graph.  This differs from the
    /// "induced subgraph" case in that there can be extra unmatched edges in the right graph.
    ///
    /// The match/score functions will not scan through every possible matching of edges in the case
    /// of a multigraph; the chosen edge matching is arbitrary and may not optimise the score.
    Subgraph,
}

/// A semantic restriction on the returned values from the VF2 iterator.
#[derive(Clone, Copy, Debug)]
pub enum Restriction<S> {
    /// Only isomorphisms with a score strictly less than the given value are returned.
    BetterThan(S),
    /// Only isomorphisms that score strictly less than the previously returned isomorphism are
    /// returned.  An initial maximum can be supplied.
    Decreasing(Option<S>),
}
impl<S> Restriction<S> {
    /// The current upper bound, if any.
    pub fn upper_bound(&self) -> Option<&S> {
        match self {
            Self::BetterThan(s) => Some(s),
            Self::Decreasing(s) => s.as_ref(),
        }
    }
}

/// An iterator which uses the VF2(++) algorithm to produce isomorphic matches between two graphs,
/// examining both syntactic and semantic graph isomorphism (graph structure and matching node and
/// edge weights).
///
/// The problem is initial created using [Vf2::new], and the algorithm implementation is then
/// refined by (optionally) calling one or more builder methods:
///
/// * [with_matching] lets you specify node- and edge-matching semantics during the algorithm.
///
/// * [with_scoring] lets you specify a scoring function for nodes and edges, which is calculated
///   on-the-fly and returned with the isomorphisms.
///
/// * [with_semantic_scoring] combines [with_matching] and [with_scoring]; the result is scored,
///   but the scoring functions may also indicate that the mapped pairs are not a semantic
///   match.
///
/// * [with_restriction] adds a restriction on the returned iterator, based on the scoring function
///   previously specified.
///
/// * [with_call_limit] limits the number of times the partial mapping can be extended before
///   terminating the isomorphism search.
///
/// * [with_vf2pp_ordering] causes the algorithm to first use the VF2++ ordering heuristic to
///   generate the initial mapping priority for the nodes.
///
/// * [with_representation] lets you control the internal graph representation used by the
///   matcher.
pub struct Vf2<N, H, NG, HG, NO, HO, NS, ES>
where
    N: alias::IntoVf2Graph,
    H: alias::IntoVf2Graph,
    NO: NodeSorter<N>,
    HO: NodeSorter<H>,
    NS: Semantics<N::NodeWeight, H::NodeWeight>,
    ES: Semantics<N::EdgeWeight, H::EdgeWeight, Score = NS::Score>,
{
    needle: N,
    needle_sorter: NO,
    haystack: H,
    haystack_sorter: HO,
    node_semantics: NS,
    edge_semantics: ES,
    restriction: Option<Restriction<NS::Score>>,
    problem: Problem,
    call_limit: Option<usize>,
    marker: marker::PhantomData<(NG, HG)>,
}

impl<N, H>
    Vf2<
        N,
        H,
        Graph<N::NodeWeight, N::EdgeWeight, N::EdgeType, N::NodeId>,
        Graph<H::NodeWeight, H::EdgeWeight, H::EdgeType, H::NodeId>,
        DefaultIdSorter,
        DefaultIdSorter,
        NoSemantics<()>,
        NoSemantics<()>,
    >
where
    N: alias::IntoVf2Graph<NodeId: IndexType>,
    H: alias::IntoVf2Graph<NodeId: IndexType, EdgeType = N::EdgeType>,
{
    /// Define the simplest VF2 problem.
    ///
    /// By default, the algorithm will find (sub)graph isomorphisms that match the given [Problem],
    /// with no additional semantics, scoring, or tracking during the execution.
    ///
    /// You can chain calls to additional builder methods to add further constraints or tracking in
    /// the isomorphism finder, or to modify internal details of how the algorithm will proceed.
    pub fn new(needle: N, haystack: H, problem: Problem) -> Self {
        Self {
            needle,
            needle_sorter: DefaultIdSorter,
            haystack,
            haystack_sorter: DefaultIdSorter,
            node_semantics: NoSemantics::new(),
            edge_semantics: NoSemantics::new(),
            restriction: None,
            problem,
            call_limit: None,
            marker: marker::PhantomData,
        }
    }
}

/// Builder functions that set the semantic matching of the problem.
impl<N, H, NG, HG, NO, HO> Vf2<N, H, NG, HG, NO, HO, NoSemantics<()>, NoSemantics<()>>
where
    N: alias::IntoVf2Graph<NodeId: IndexType>,
    H: alias::IntoVf2Graph<NodeId: IndexType, EdgeType = N::EdgeType>,
    NO: NodeSorter<N>,
    HO: NodeSorter<H>,
{
    /// Add node-matching semantics to the problem.
    ///
    /// This is only as expensive as the matcher function itself; it doesn't change anything about
    /// the underlying algorithm.
    pub fn with_node_matching<NM>(
        self,
        node: NM,
    ) -> Vf2<N, H, NG, HG, NO, HO, Matcher<NM>, NoSemantics<()>>
    where
        Matcher<NM>: Semantics<N::NodeWeight, H::NodeWeight, Score = ()>,
    {
        Vf2 {
            needle: self.needle,
            needle_sorter: self.needle_sorter,
            haystack: self.haystack,
            haystack_sorter: self.haystack_sorter,
            node_semantics: Matcher(node),
            edge_semantics: self.edge_semantics,
            restriction: self.restriction,
            problem: self.problem,
            call_limit: self.call_limit,
            marker: marker::PhantomData,
        }
    }
    /// Add both node- and edge-matching to the problem.
    ///
    /// Adding edge matching fundamentally modifies how the algorithm proceeds.  If the needle has
    /// no parallel edges with matching directions, the only modification is that edges in the
    /// haystack are tried in turn for a match, so the cost is mostly just the cost of the
    /// edge-matching function.
    ///
    /// If the needle is a _multigraph_, the algorithm changes substantially because we have to
    /// attempt to one group of edges to another.  Currently this is done by a greedy match, and is
    /// not guaranteed to find a valid matching, even if one is possible.
    pub fn with_matching<NM, EM>(
        self,
        node: NM,
        edge: EM,
    ) -> Vf2<N, H, NG, HG, NO, HO, Matcher<NM>, Matcher<EM>>
    where
        Matcher<NM>: Semantics<N::NodeWeight, H::NodeWeight, Score = ()>,
        Matcher<EM>: Semantics<N::EdgeWeight, H::EdgeWeight, Score = ()>,
    {
        Vf2 {
            needle: self.needle,
            needle_sorter: self.needle_sorter,
            haystack: self.haystack,
            haystack_sorter: self.haystack_sorter,
            node_semantics: Matcher(node),
            edge_semantics: Matcher(edge),
            restriction: self.restriction,
            problem: self.problem,
            call_limit: self.call_limit,
            marker: marker::PhantomData,
        }
    }

    /// Add scoring functions to the nodes and edges.
    ///
    /// In the case of a multigraph, the edge-scoring is node guaranteed to minimise the score.  One
    /// score will be found, but it is arbitrary.
    pub fn with_scoring<NS, ES>(
        self,
        node: NS,
        edge: ES,
    ) -> Vf2<N, H, NG, HG, NO, HO, Scorer<NS>, Scorer<ES>>
    where
        Scorer<NS>: Semantics<N::NodeWeight, H::NodeWeight>,
        Scorer<ES>: Semantics<
                N::EdgeWeight,
                H::EdgeWeight,
                Score = <Scorer<NS> as Semantics<N::NodeWeight, H::NodeWeight>>::Score,
            >,
    {
        Vf2 {
            needle: self.needle,
            needle_sorter: self.needle_sorter,
            haystack: self.haystack,
            haystack_sorter: self.haystack_sorter,
            node_semantics: Scorer(node),
            edge_semantics: Scorer(edge),
            restriction: None,
            problem: self.problem,
            call_limit: self.call_limit,
            marker: marker::PhantomData,
        }
    }

    /// Add full semantic scoring to the nodes and edges.
    ///
    /// These can be any combination of matching and scoring between the two components.  The
    /// limitations when dealing with multigraphs discussed in [with_matching] and [with_scoring]
    /// apply in the same way.
    pub fn with_semantics<NS, ES>(self, node: NS, edge: ES) -> Vf2<N, H, NG, HG, NO, HO, NS, ES>
    where
        NS: Semantics<N::NodeWeight, H::NodeWeight>,
        ES: Semantics<N::EdgeWeight, H::EdgeWeight, Score = NS::Score>,
    {
        Vf2 {
            needle: self.needle,
            needle_sorter: self.needle_sorter,
            haystack: self.haystack,
            haystack_sorter: self.haystack_sorter,
            node_semantics: node,
            edge_semantics: edge,
            restriction: None,
            problem: self.problem,
            call_limit: self.call_limit,
            marker: marker::PhantomData,
        }
    }
}

impl<N, H, NG, HG, NO, HO, NS, ES> Vf2<N, H, NG, HG, NO, HO, NS, ES>
where
    N: alias::IntoVf2Graph,
    H: alias::IntoVf2Graph,
    NO: NodeSorter<N>,
    HO: NodeSorter<H>,
    NS: Semantics<N::NodeWeight, H::NodeWeight>,
    ES: Semantics<N::EdgeWeight, H::EdgeWeight, Score = NS::Score>,
{
    pub fn needle(&self) -> &N {
        &self.needle
    }
    pub fn haystack(&self) -> &H {
        &self.haystack
    }

    /// Add a limit to the number of candidate extensions the algorithm is allowed to attempt.
    ///
    /// This can be used to upper-bound the runtime of the algorithm while keeping it deterministic.
    pub fn with_call_limit(self, call_limit: Option<usize>) -> Self {
        Self { call_limit, ..self }
    }

    /// Apply a restriction to the returned values from the iterator.
    pub fn with_restriction(self, restriction: Restriction<NS::Score>) -> Self {
        Self {
            restriction: Some(restriction),
            ..self
        }
    }

    /// Use the VF2++ ordering heuristic to seed the initial priority queue for node matching.
    pub fn with_vf2pp_ordering(self) -> Vf2<N, H, NG, HG, Vf2ppSorter, Vf2ppSorter, NS, ES> {
        Vf2 {
            needle: self.needle,
            needle_sorter: Vf2ppSorter,
            haystack: self.haystack,
            haystack_sorter: Vf2ppSorter,
            node_semantics: self.node_semantics,
            edge_semantics: self.edge_semantics,
            restriction: self.restriction,
            problem: self.problem,
            call_limit: self.call_limit,
            marker: marker::PhantomData,
        }
    }

    /// Fix the internal representation of the graphs used during the isomorphism iteration.
    pub fn with_representation<NG2, HG2>(self) -> Vf2<N, H, NG2, HG2, NO, HO, NS, ES>
    where
        NG2: for<'a> alias::Vf2Graph<
                'a,
                NodeWeight = N::NodeWeight,
                EdgeWeight = N::EdgeWeight,
                EdgeType = N::EdgeType,
            > + Create,
        HG2: for<'a> alias::Vf2Graph<
                'a,
                NodeWeight = H::NodeWeight,
                EdgeWeight = H::EdgeWeight,
                EdgeType = H::EdgeType,
            > + Create,
    {
        Vf2 {
            needle: self.needle,
            needle_sorter: self.needle_sorter,
            haystack: self.haystack,
            haystack_sorter: self.haystack_sorter,
            node_semantics: self.node_semantics,
            edge_semantics: self.edge_semantics,
            restriction: self.restriction,
            problem: self.problem,
            call_limit: self.call_limit,
            marker: marker::PhantomData,
        }
    }
}

impl<N, H, NG, HG, NO, HO, NS, ES> IntoIterator for Vf2<N, H, NG, HG, NO, HO, NS, ES>
where
    N: alias::IntoVf2Graph,
    H: alias::IntoVf2Graph<EdgeType = N::EdgeType>,
    NG: for<'a> alias::Vf2Graph<
            'a,
            NodeWeight = N::NodeWeight,
            EdgeWeight = N::EdgeWeight,
            EdgeType = N::EdgeType,
        > + Create,
    HG: for<'a> alias::Vf2Graph<
            'a,
            NodeWeight = H::NodeWeight,
            EdgeWeight = H::EdgeWeight,
            EdgeType = H::EdgeType,
        > + Create,
    NO: NodeSorter<N>,
    HO: NodeSorter<H>,
    NS: Semantics<N::NodeWeight, H::NodeWeight>,
    ES: Semantics<N::EdgeWeight, H::EdgeWeight, Score = NS::Score>,
{
    type Item = Result<
        (IndexMap<N::NodeId, H::NodeId>, NS::Score),
        IsIsomorphicError<NS::Error, ES::Error>,
    >;
    type IntoIter = Vf2IntoIter<NG, HG, N::NodeId, H::NodeId, NS, ES>;

    fn into_iter(self) -> Self::IntoIter {
        let needle_reorder = self.needle_sorter.sort(self.needle);
        let needle = reorder_nodes::<N, NG>(self.needle, &needle_reorder);
        let haystack_reorder = self.haystack_sorter.sort(self.haystack);
        let haystack = reorder_nodes::<H, HG>(self.haystack, &haystack_reorder);

        // Fast-path short circuits.
        let possible = match self.problem {
            Problem::Exact => {
                needle.node_count() == haystack.node_count()
                    && needle.edge_count() == haystack.edge_count()
            }
            Problem::InducedSubgraph | Problem::Subgraph => {
                needle.node_count() <= haystack.node_count()
                    && needle.edge_count() <= haystack.edge_count()
            }
        };
        // The stack typically will need to grow to account to store a "haystack" frame for each
        // node in the needle graph in a success path (and it's shorter in the failure path).
        let loop_stack = if possible {
            let mut stack = Vec::with_capacity(needle.node_count());
            stack.push(Frame::ChooseNextNeedle);
            stack
        } else {
            // If the short-circuit checks say the problem is clearly impossible, then we don't
            // even bother beginning the search.
            vec![]
        };
        let score_stack = {
            let capacity =
                if NS::MATCH == SemanticType::Disabled && ES::MATCH == SemanticType::Disabled {
                    1
                } else {
                    needle.node_count() + 1
                };
            let mut stack = Vec::with_capacity(capacity);
            stack.push(NS::Score::id());
            stack
        };

        Vf2IntoIter {
            needle: State::new(needle),
            needle_reorder,
            haystack: State::new(haystack),
            haystack_reorder,
            problem: self.problem,
            node_semantics: self.node_semantics,
            edge_semantics: self.edge_semantics,
            restriction: self.restriction,
            score_stack,
            loop_stack,
            num_calls: 0,
            call_limit: self.call_limit,
        }
    }
}

struct State<G: GraphBase, OtherId> {
    graph: G,
    /// The current mapping from indices in this graph to indices in the other graph.  If a node is
    /// not yet mapped, the other index is stored as `I::max()`.
    mapping: Vec<OtherId>,
    /// Mapping from node index to the generation at which a node was first added to the mapping
    /// that had an `[outbound, inbound]` edge to that index.  This can be used to find new
    /// candidate nodes to add to the mapping; you typically want your next node to be one that has
    /// edges linking it to the existing mapping (but isn't yet _in_ the mapping).
    neighbor_since: Vec<[Option<NonZero<usize>>; 2]>,
    /// The number of neighbors to the mapping in the `[outgoing, incoming]` direction.
    // TODO: currently this count isn't decremented when a neighbor is added to the mapping, so it
    // becomes less useful as a metric as the mapping completes.
    num_neighbors: [usize; 2],
    /// The edge multiplicity of a given node pair.  If the graph is directed, the keys are
    /// `(source, target)`.  If the graph is undirected, the keys are always in sorted order, and
    /// the multiplicity includes both "directions" of the edge.
    adjacency_matrix: HashMap<(G::NodeId, G::NodeId), usize>,
    /// Is this a multigraph?
    multigraph: bool,
    /// The number of nodes in currently in the mapping.
    generation: usize,
}

impl<G, OtherId> State<G, OtherId>
where
    G: for<'a> alias::Vf2Graph<'a>,
    OtherId: IndexType,
{
    pub fn new(graph: G) -> Self {
        let node_count = graph.node_count();
        let mut adjacency_matrix = HashMap::with_capacity(graph.edge_count());
        let mut multigraph = false;
        for edge in graph.edge_references() {
            let item = if graph.is_directed() || edge.source() <= edge.target() {
                (edge.source(), edge.target())
            } else {
                (edge.target(), edge.source())
            };
            match adjacency_matrix.entry(item) {
                Entry::Vacant(entry) => {
                    entry.insert(1);
                }
                Entry::Occupied(mut entry) => {
                    multigraph = true;
                    *entry.get_mut() += 1;
                }
            }
        }
        Self {
            mapping: vec![<OtherId as IndexType>::max(); node_count],
            neighbor_since: vec![[None, None]; node_count],
            num_neighbors: [0, 0],
            adjacency_matrix,
            multigraph,
            generation: 0,
            graph,
        }
    }

    /// Find the mapping (in the graph) of the `target` of a local edge.
    ///
    /// If the edge is a self loop, return the mapping of `source`, if provided.
    pub fn map_target(
        &self,
        source: G::NodeId,
        target: G::NodeId,
        their_source: Option<OtherId>,
    ) -> Option<OtherId> {
        if source == target {
            their_source
        } else {
            let other = self.mapping[target.index()];
            (other != <OtherId as IndexType>::max()).then_some(other)
        }
    }

    /// Is every node in the graph mapped?
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.generation == self.mapping.len()
    }

    /// Add a new entry into the mapping.
    pub fn push_mapping(&mut self, ours: G::NodeId, theirs: OtherId) {
        self.generation += 1;
        debug_assert_eq!(self.mapping[ours.index()], <OtherId as IndexType>::max());
        self.mapping[ours.index()] = theirs;
        // Mark any nodes that are newly neighbors of the set of mapped nodes.  To be _newly_ a
        // neighbor, it must not already be a neighbor.
        let directions: &[Direction] = if self.graph.is_directed() {
            &[Outgoing, Incoming]
        } else {
            &[Outgoing]
        };
        for direction in directions {
            let dir_index = *direction as usize;
            for neighbor in self.graph.neighbors_directed(ours, *direction) {
                let neighbor = neighbor.index();
                if self.neighbor_since[neighbor][dir_index].is_none() {
                    // The `NonZero` here will always be `Some` since we just incremented the
                    // generation, but it'll all compile out to be simple integers anyway.
                    self.neighbor_since[neighbor][dir_index] = NonZero::new(self.generation);
                    self.num_neighbors[dir_index] += 1;
                }
            }
        }
    }

    /// Undo the mapping of node `ours`.  The node `ours` must be the last one given to
    /// `push_mapping` for this to make sense.
    pub fn pop_mapping(&mut self, ours: G::NodeId) {
        // Any neighbors of ours that became neighbors of the mapping at our generation are now no
        // longer neighbors of the mapping, since all the nodes that got added to the mapping after
        // us are already popped.
        let directions: &[Direction] = if self.graph.is_directed() {
            &[Outgoing, Incoming]
        } else {
            &[Outgoing]
        };
        let my_generation = NonZero::new(self.generation); // not trying to cause a big s-sensation
        for direction in directions {
            let dir_index = *direction as usize;
            for neighbor in self.graph.neighbors_directed(ours, *direction) {
                let neighbor = neighbor.index();
                if self.neighbor_since[neighbor][dir_index] == my_generation {
                    self.neighbor_since[neighbor][dir_index] = None;
                    self.num_neighbors[dir_index] -= 1;
                }
            }
        }
        self.mapping[ours.index()] = <OtherId as IndexType>::max();
        self.generation -= 1;
    }

    /// Get the next unmapped node in the priority queue from a specific list whose index is at
    /// least `skip`, and its neighboring state to the partial mapping matches the predicate.
    fn next_unmapped_from(
        &self,
        skip: usize,
        mut pred: impl FnMut(NeighborKind) -> bool,
    ) -> Option<G::NodeId> {
        self.neighbor_since
            .iter()
            .enumerate()
            .skip(skip)
            .filter(|(index, generations)| {
                self.mapping[*index] == <OtherId as IndexType>::max()
                    && pred(NeighborKind::from_neighbor_generations(generations))
            })
            .map(|(index, _)| G::NodeId::new(index))
            .next()
    }

    /// Get the next unmapped node
    ///
    /// # Warning
    ///
    /// This only makes sense to call for the haystack graph; the `problem`-matching logic is
    /// inverted for the needle.  (Ideally this would be a method on `Vf2IntoIter` instead, but
    /// getting the generics to work and keeping the borrow-checker happy is verbose.)
    #[inline]
    pub fn next_haystack_from(
        &mut self,
        skip: usize,
        needle_kind: NeighborKind,
        problem: Problem,
    ) -> Option<G::NodeId> {
        match problem {
            Problem::Exact | Problem::InducedSubgraph => {
                self.next_unmapped_from(skip, |haystack_kind| haystack_kind == needle_kind)
            }
            Problem::Subgraph => self.next_unmapped_from(skip, |haystack_kind| {
                haystack_kind
                    .partial_cmp(&needle_kind)
                    .is_some_and(|ord| ord != Ordering::Less)
            }),
        }
    }

    /// Number of edges from `source` to `target` (including the reverse, if the graph is
    /// undirected).
    ///
    /// If you already have an edge reference and want to know its multiplicity, use
    /// [edge_multiplicity_of], which is optimised in the case of non-multigraphs.
    #[inline]
    fn edge_multiplicity(&self, source: G::NodeId, target: G::NodeId) -> usize {
        let item = if self.graph.is_directed() || source <= target {
            (source, target)
        } else {
            (target, source)
        };
        *self.adjacency_matrix.get(&item).unwrap_or(&0)
    }

    /// What is the multiplicity of the given edge reference?
    ///
    /// This is optimised to avoid hash-map lookups in the case of a non-multigraph (since the
    /// answer is simply always 1; you've already proved you've got the edge, so it can't be zero).
    #[inline]
    fn edge_multiplicity_of<'a>(&'a self, edge: <G as alias::Vf2GraphRef<'a>>::EdgeRef) -> usize {
        if self.multigraph {
            self.edge_multiplicity(edge.source(), edge.target())
        } else {
            1
        }
    }
}

/// What kind of neighbor the needle node we're matching against is, relative to the partial
/// mapping.  In the case of an undirected graph, the only variants used are [Neither] and
/// [Outgoing].
#[derive(Copy, Clone, PartialEq, Debug)]
enum NeighborKind {
    /// Needle node is not a neighbor of the partial mapping at all.
    Neither,
    /// Needle node is the target of an outgoing edge from the mapping.
    Outgoing,
    /// Needle node is the source of an incoming edge into the mapping.
    Incoming,
    /// Needle node is both the source and target of edges from the mapping.
    Both,
}
impl NeighborKind {
    /// Get the kind of neighbor from an entry in [State::neighbor_since].
    #[inline]
    fn from_neighbor_generations(generations: &[Option<NonZero<usize>>; 2]) -> Self {
        match generations {
            [None, None] => NeighborKind::Neither,
            [Some(_), None] => NeighborKind::Outgoing,
            [None, Some(_)] => NeighborKind::Incoming,
            [Some(_), Some(_)] => NeighborKind::Both,
        }
    }
}
impl PartialOrd for NeighborKind {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Neither, Self::Neither) => Some(Ordering::Equal),
            (Self::Neither, _) => Some(Ordering::Less),
            (_, Self::Neither) => Some(Ordering::Greater),
            (Self::Both, Self::Both) => Some(Ordering::Equal),
            (Self::Both, _) => Some(Ordering::Greater),
            (_, Self::Both) => Some(Ordering::Less),
            // This rule now only handles when both are in `[Outgoing, Incoming]`.
            (_, _) => (self == other).then_some(Ordering::Equal),
        }
    }
}

#[derive(Debug)]
enum Frame<N, H> {
    ChooseNextHaystack { nodes: (N, H), kind: NeighborKind },
    ChooseNextNeedle,
}

pub struct Vf2IntoIter<N, H, NId, HId, NS, ES>
where
    N: for<'a> alias::Vf2Graph<'a>,
    H: for<'a> alias::Vf2Graph<'a>,
    NS: Semantics<N::NodeWeight, H::NodeWeight>,
    ES: Semantics<N::EdgeWeight, H::EdgeWeight, Score = NS::Score>,
{
    needle: State<N, H::NodeId>,
    /// Mapping of indices in the rewritten `needle` back to the original node ids of the input.
    needle_reorder: Vec<NId>,
    haystack: State<H, N::NodeId>,
    /// Mapping of indices in the rewritten `haystack` back to the original node ids of the input.
    haystack_reorder: Vec<HId>,
    problem: Problem,
    node_semantics: NS,
    edge_semantics: ES,
    restriction: Option<Restriction<NS::Score>>,
    score_stack: Vec<NS::Score>,
    loop_stack: Vec<Frame<N::NodeId, H::NodeId>>,
    num_calls: usize,
    pub call_limit: Option<usize>,
}

impl<N, H, NId, HId, NS, ES> Vf2IntoIter<N, H, NId, HId, NS, ES>
where
    N: for<'a> alias::Vf2Graph<'a>,
    H: for<'a> alias::Vf2Graph<'a, EdgeType = N::EdgeType>,
    NId: Hash + Eq + Copy,
    HId: Copy,
    NS: Semantics<N::NodeWeight, H::NodeWeight>,
    ES: Semantics<N::EdgeWeight, H::EdgeWeight, Score = NS::Score>,
{
    fn mapping(&self) -> IndexMap<NId, HId> {
        self.needle
            .mapping
            .iter()
            .enumerate()
            .map(|(needle, haystack)| {
                debug_assert!(*haystack != <H::NodeId as IndexType>::max());
                (
                    self.needle_reorder[needle],
                    self.haystack_reorder[haystack.index()],
                )
            })
            .collect()
    }

    /// Find the unmapped candidates with the highest priority from both graphs.
    ///
    /// If found, the two candidates are guaranteed to satisfy the most basic compatibility
    /// requirements with respect to neighboring the partial mapping (i.e. the haystack node will be
    /// a neighbor in the same direction(s) of the mapping as the needle in an exact or
    /// induced-subgraph problem, and will be _at least_ a neighbor in the same direction(s) for
    /// non-induced subgraphs).
    fn next_candidates(&mut self) -> Option<(N::NodeId, H::NodeId, NeighborKind)> {
        let mut needle_isolated: Option<N::NodeId> = None;
        let mut needle_outgoing: Option<N::NodeId> = None;
        let mut needle_incoming: Option<N::NodeId> = None;
        let mut needle_candidate_from = |skip: usize| -> Option<(N::NodeId, NeighborKind)> {
            if needle_outgoing.is_some() && needle_incoming.is_some() && needle_isolated.is_some() {
                return needle_isolated.map(|neighbor| (neighbor, NeighborKind::Neither));
            }
            for (neighbor, kind) in self
                .needle
                .neighbor_since
                .iter()
                .enumerate()
                .skip(skip)
                .filter(|(index, _)| self.needle.mapping[*index] == <H::NodeId as IndexType>::max())
                .map(|(index, generations)| {
                    (
                        N::NodeId::new(index),
                        NeighborKind::from_neighbor_generations(generations),
                    )
                })
            {
                match kind {
                    NeighborKind::Neither => {
                        needle_isolated.get_or_insert(neighbor);
                        if needle_outgoing.is_some() && needle_incoming.is_some() {
                            return Some((neighbor, kind));
                        }
                    }
                    NeighborKind::Outgoing => {
                        if needle_outgoing.is_none() {
                            needle_outgoing = Some(neighbor);
                            return Some((neighbor, kind));
                        }
                    }
                    NeighborKind::Incoming => {
                        if needle_incoming.is_none() {
                            needle_incoming = Some(neighbor);
                            return Some((neighbor, kind));
                        }
                    }
                    NeighborKind::Both => {
                        if needle_outgoing.is_none() || needle_incoming.is_none() {
                            // It doesn't matter if we overwrite a previous value; we will already
                            // have returned it.
                            needle_outgoing = Some(neighbor);
                            needle_incoming = Some(neighbor);
                            return Some((neighbor, kind));
                        }
                    }
                }
            }
            needle_isolated.map(|neighbor| (neighbor, NeighborKind::Neither))
        };
        let mut needle_pos = 0;
        loop {
            let (needle_node, needle_kind) = needle_candidate_from(needle_pos)?;
            needle_pos = needle_node.index() + 1;
            // Strictly we could probably save the iteration state of the `haystack` search here,
            // but the performance difference is likely negligible in practice.
            if let Some(haystack_node) =
                self.haystack
                    .next_haystack_from(0, needle_kind, self.problem)
            {
                return Some((needle_node, haystack_node, needle_kind));
            };
            if needle_kind == NeighborKind::Neither {
                // `Neither` is the lowest priority, and if we fail to match it, we're done.
                return None;
            }
        }
    }

    /// Remove this pair of nodes from the mapping.
    ///
    /// The pair of nodes must be on the top of the stack of pushes.
    fn pop_state(&mut self, nodes: (N::NodeId, H::NodeId)) {
        // Restore state.
        self.needle.pop_mapping(nodes.0);
        self.haystack.pop_mapping(nodes.1);
        if !(NS::MATCH == SemanticType::Disabled && ES::MATCH == SemanticType::Disabled) {
            self.score_stack.pop();
        }
    }

    /// Add a new pair of nodes to the mapping, and set the total score.
    fn push_state(&mut self, nodes: (N::NodeId, H::NodeId), new_score: NS::Score) {
        // Add mapping nx <-> mx to the state
        self.needle.push_mapping(nodes.0, nodes.1);
        self.haystack.push_mapping(nodes.1, nodes.0);
        if !(NS::MATCH == SemanticType::Disabled && ES::MATCH == SemanticType::Disabled) {
            // If both are disabled then the score has no meaning, and we can pretend it's some
            // arbitrary ZST.  This `if` block just ensures that the `Vec` doesn't waste any cycles
            // updating its inner `len`; it'll get compiled out entirely.
            self.score_stack.push(new_score);
        }
    }

    #[inline]
    fn directed(&self) -> bool {
        self.needle.graph.is_directed()
    }

    /// If we added this pair of nodes to the mapping, would it still be feasible to reach a full
    /// solution in the future, and if so, what woould the new score of the partial mapping be?
    ///
    /// The `Result` is related to the semantic node- and edge-matching in the scoring.  The inner
    /// `Option` is `Some(new_score)` if the pair is feasible, and `None` if it is not.
    #[allow(clippy::type_complexity)]
    fn is_feasible(
        &self,
        nodes: (N::NodeId, H::NodeId),
    ) -> Result<Option<NS::Score>, IsIsomorphicError<NS::Error, ES::Error>> {
        // This is the core correctness component; are the nodes of this mapping and all the edges
        // that would be newly mapped consistent with a solution to the problem?  This must be
        // precise; we _must_ cut if the mapping would not be consistent.
        let Some(score) = self.is_consistent(nodes)? else {
            return Ok(None);
        };

        // Everything else in this function is lookahead feasibility tests.  It's important for
        // correctness that we never report a mapping as unfeasible if actually it does lead to a
        // valid solution.  If in doubt, say it's feasible; we attempt to eagerly cut as a
        // performance optimisation only.

        if !self.unmapped_existing_neighbors_feasible(nodes, Outgoing)
            || !self.unmapped_new_neighbors_feasible(nodes, Outgoing)
        {
            return Ok(None);
        }
        if self.directed()
            && (!self.unmapped_existing_neighbors_feasible(nodes, Incoming)
                || !self.unmapped_new_neighbors_feasible(nodes, Incoming))
        {
            return Ok(None);
        }
        Ok(Some(score))
    }

    #[allow(clippy::type_complexity)]
    fn is_consistent(
        &self,
        nodes: (N::NodeId, H::NodeId),
    ) -> Result<Option<NS::Score>, IsIsomorphicError<NS::Error, ES::Error>> {
        // Are the semantics of these two nodes consistent?
        let node_score = if NS::MATCH == SemanticType::Disabled {
            // If there's no node-matching enabled, then the two are always semantically feasible.
            // The only reason they might fail is if the newly mapped connecting edges aren't.
            NS::Score::id()
        } else {
            let (Some(needle_weight), Some(haystack_weight)) = (
                self.needle.graph.node_weight(nodes.0),
                self.haystack.graph.node_weight(nodes.1),
            ) else {
                panic!("internal logic error: nodes not in graph");
            };
            match self
                .node_semantics
                .score(needle_weight, haystack_weight)
                .map_err(IsIsomorphicError::NodeMatcher)?
            {
                Some(score) => score,
                None => return Ok(None),
            }
        };

        // Are the semantics of the edges that would become fully mapped consistent?
        let edge_score = if ES::MATCH == SemanticType::Disabled {
            if !self.mapped_edges_counts_match(nodes, Outgoing)
                || (self.directed() && !self.mapped_edges_counts_match(nodes, Incoming))
            {
                return Ok(None);
            }
            NS::Score::id()
        } else {
            let Some(outgoing_score) = self
                .mapped_edges_semantic_match(nodes, Outgoing)
                .map_err(IsIsomorphicError::EdgeMatcher)?
            else {
                return Ok(None);
            };
            if self.directed() {
                let Some(incoming_score) = self
                    .mapped_edges_semantic_match(nodes, Incoming)
                    .map_err(IsIsomorphicError::EdgeMatcher)?
                else {
                    return Ok(None);
                };
                NS::Score::combine(&outgoing_score, &incoming_score)
            } else {
                outgoing_score
            }
        };

        let new_score = NS::Score::combine(
            &NS::Score::combine(
                self.score_stack.last().expect("always non-empty"),
                &node_score,
            ),
            &edge_score,
        );
        // We're not consistent if the score breaks the limit.
        if self
            .restriction
            .as_ref()
            .and_then(|restriction| restriction.upper_bound())
            .map(|bound| NS::Score::cmp(&new_score, bound) == Ordering::Less)
            .unwrap_or(true)
        {
            Ok(Some(new_score))
        } else {
            Ok(None)
        }
    }

    /// Are the directed edges of these two paired nodes consistent with all nodes already in the
    /// mapping?
    ///
    /// This is the more expensive semantic check, where we have to run pairwise through the edge
    /// lists to match them up against each other.  If there's no semantic checking enabled, use
    /// `mapped_edges_counts_match`, which short-circuits based on the pre-calculated adjacency
    /// matrices.
    fn mapped_edges_semantic_match(
        &self,
        nodes: (N::NodeId, H::NodeId),
        direction: Direction,
    ) -> Result<Option<NS::Score>, ES::Error> {
        let needle = &self.needle;
        let haystack = &self.haystack;
        // We only handle self loops in the `Outgoing` direction to avoid double-counting the score.
        let handle_self = direction == Direction::Outgoing;

        macro_rules! neighbor {
            ($edge:expr) => {
                match direction {
                    Direction::Outgoing => $edge.target(),
                    Direction::Incoming => $edge.source(),
                }
            };
        }

        if !needle.multigraph {
            let mut score = NS::Score::id();
            let mapped_source = handle_self.then_some(nodes.1);
            for needle_edge in needle.graph.edges_directed(nodes.0, direction) {
                let Some(haystack_neighbor) =
                    needle.map_target(nodes.0, neighbor!(needle_edge), mapped_source)
                else {
                    continue;
                };
                let haystack_multiplicity = match direction {
                    Direction::Outgoing => haystack.edge_multiplicity(nodes.1, haystack_neighbor),
                    Direction::Incoming => haystack.edge_multiplicity(haystack_neighbor, nodes.1),
                };
                match self.problem {
                    Problem::Exact | Problem::InducedSubgraph => {
                        if haystack_multiplicity != 1 {
                            return Ok(None);
                        }
                    }
                    Problem::Subgraph => {
                        if haystack_multiplicity == 0 {
                            return Ok(None);
                        }
                    }
                }
                let Some(edge_score) = haystack
                    .graph
                    .edges_directed(nodes.1, direction)
                    .filter_map(|haystack_edge| {
                        if (neighbor!(haystack_edge) == haystack_neighbor) {
                            self.edge_semantics
                                .score(needle_edge.weight(), haystack_edge.weight())
                                .transpose()
                        } else {
                            None
                        }
                    })
                    .next()
                    .transpose()?
                else {
                    return Ok(None);
                };
                score = NS::Score::combine(&score, &edge_score);
            }
            return Ok(Some(score));
        }

        let mut needle_edges = HashMap::<_, SmallVec<[_; 4]>>::new();
        let mapped_source = handle_self.then_some(nodes.1);
        for edge in needle.graph.edges_directed(nodes.0, direction) {
            let Some(haystack_neighbor) =
                needle.map_target(nodes.0, neighbor!(edge), mapped_source)
            else {
                continue;
            };
            needle_edges
                .entry(haystack_neighbor.index())
                .or_insert_with(|| SmallVec::with_capacity(needle.edge_multiplicity_of(edge)))
                .push(edge);
        }
        let mut haystack_edges = HashMap::<_, SmallVec<[_; 4]>>::new();
        let mapped_source = handle_self.then_some(nodes.0);
        for edge in haystack.graph.edges_directed(nodes.1, direction) {
            let Some(needle_neighbor) =
                haystack.map_target(nodes.1, neighbor!(edge), mapped_source)
            else {
                continue;
            };
            haystack_edges
                .entry(needle_neighbor.index())
                .or_insert_with(|| SmallVec::with_capacity(haystack.edge_multiplicity_of(edge)))
                .push(edge);
        }

        // In all problems, every edge between two mapped nodes in the needle must map to an edge
        // between the two paired nodes in the haystack.  In the `Exact` and `InducedSubgraph`
        // problems, we also need the reciprocal condition; there must be no unmapped edges on the
        // haystack side.
        let mut score = NS::Score::id();
        for (haystack_neighbor, needle_edges) in needle_edges {
            let needle_neighbor = haystack.mapping[haystack_neighbor.index()];
            let Some(mut haystack_edges) = haystack_edges.remove(&needle_neighbor.index()) else {
                // Actually this shouldn't ever trigger if we already checked that the _number_ of
                // edges is consistent.
                return Ok(None);
            };
            for needle_edge in needle_edges {
                // The intent of this is easier to read with an imperative-loop closure, but
                // spelling out the types for that is pretty disgusting.  We're finding the first
                // haystack edge that matches the needle edge, propagating _any_ observed failure
                // from the edge matcher.
                //
                // TODO: this is a greedy algorithm that is not guaranteed to find an isomorphism,
                // even if one exists, and certainly is not guaranteed to minimise the score.
                let Some((pos, edge_score)) = haystack_edges
                    .iter()
                    .enumerate()
                    .filter_map(|(pos, haystack_edge)| {
                        self.edge_semantics
                            .score(needle_edge.weight(), haystack_edge.weight())
                            .map(|score| score.map(|score| (pos, score)))
                            .transpose()
                    })
                    .next()
                    .transpose()?
                else {
                    return Ok(None);
                };
                score = NS::Score::combine(&score, &edge_score);
                haystack_edges.swap_remove(pos);
            }
            if !haystack_edges.is_empty()
                && (self.problem == Problem::Exact || self.problem == Problem::InducedSubgraph)
            {
                return Ok(None);
            }
        }
        Ok(Some(score))
    }

    fn mapped_edges_counts_match(
        &self,
        nodes: (N::NodeId, H::NodeId),
        direction: Direction,
    ) -> bool {
        let needle = &self.needle;
        let haystack = &self.haystack;

        macro_rules! order {
            ($node:expr, $other:expr) => {
                match direction {
                    Direction::Outgoing => ($node, $other),
                    Direction::Incoming => ($other, $node),
                }
            };
        }

        let mapped_self = (direction == Direction::Outgoing).then_some(nodes.1);
        for needle_neighbor in needle.graph.neighbors_directed(nodes.0, direction) {
            let Some(haystack_neighbor) = needle.map_target(nodes.0, needle_neighbor, mapped_self)
            else {
                continue;
            };
            let (needle_source, needle_target) = order!(nodes.0, needle_neighbor);
            let (haystack_source, haystack_target) = order!(nodes.1, haystack_neighbor);
            let needle_multiplicity = match needle.multigraph {
                true => needle.edge_multiplicity(needle_source, needle_target),
                false => 1,
            };
            let haystack_multiplicity =
                haystack.edge_multiplicity(haystack_source, haystack_target);
            match self.problem {
                Problem::Exact | Problem::InducedSubgraph => {
                    if needle_multiplicity != haystack_multiplicity {
                        return false;
                    }
                }
                Problem::Subgraph => {
                    if needle_multiplicity > haystack_multiplicity {
                        return false;
                    }
                }
            }
        }
        if let Problem::Subgraph = self.problem {
            // It's not a problem if there extra edges between mapped nodes in the haystack if we're
            // looking for a non-induced subgraph, but for everything else we have to check.
            return true;
        }

        let mapped_self = (direction == Direction::Outgoing).then_some(nodes.0);
        for haystack_neighbor in haystack.graph.neighbors_directed(nodes.1, direction) {
            let Some(needle_neighbor) =
                haystack.map_target(nodes.1, haystack_neighbor, mapped_self)
            else {
                continue;
            };
            let (needle_source, needle_target) = order!(nodes.0, needle_neighbor);
            let (haystack_source, haystack_target) = order!(nodes.1, haystack_neighbor);
            let needle_multiplicity = needle.edge_multiplicity(needle_source, needle_target);
            let haystack_multiplicity = match haystack.multigraph {
                true => haystack.edge_multiplicity(haystack_source, haystack_target),
                false => 1,
            };
            if needle_multiplicity != haystack_multiplicity {
                return false;
            }
        }
        true
    }

    /// Do the nodes in this mapping have a compatible number of directed neighbors that are
    /// unmapped, and already directed neighbors of the partial mapping?
    fn unmapped_existing_neighbors_feasible(
        &self,
        nodes: (N::NodeId, H::NodeId),
        direction: Direction,
    ) -> bool {
        let needle = &self.needle;
        let haystack = &self.haystack;

        fn inc_pair(mut state: [usize; 2], inc: [bool; 2]) -> [usize; 2] {
            state[0] += inc[0] as usize;
            state[1] += inc[1] as usize;
            state
        }

        let needle_neighbors = needle
            .graph
            .neighbors_directed(nodes.0, direction)
            .filter(|node| needle.mapping[node.index()] == <H::NodeId as IndexType>::max())
            .fold([0, 0], |counts, node| {
                inc_pair(
                    counts,
                    needle.neighbor_since[node.index()].map(|x| x.is_some()),
                )
            });
        let haystack_neighbors = haystack
            .graph
            .neighbors_directed(nodes.1, direction)
            .filter(|node| haystack.mapping[node.index()] == <N::NodeId as IndexType>::max())
            .fold([0, 0], |counts, node| {
                inc_pair(
                    counts,
                    haystack.neighbor_since[node.index()].map(|x| x.is_some()),
                )
            });
        match self.problem {
            Problem::Exact => needle_neighbors == haystack_neighbors,
            Problem::InducedSubgraph | Problem::Subgraph => {
                needle_neighbors[0] <= haystack_neighbors[0]
                    && needle_neighbors[1] <= haystack_neighbors[1]
            }
        }
    }

    /// Would this extension of the mapping add a compatible number of new neighbors to the entire
    /// mapping?
    fn unmapped_new_neighbors_feasible(
        &self,
        nodes: (N::NodeId, H::NodeId),
        direction: Direction,
    ) -> bool {
        if let Problem::Subgraph = self.problem {
            // If we're looking for a non-induced subgraph, the counts here don't mean anything.
            return true;
        }

        let needle = &self.needle;
        let haystack = &self.haystack;

        // TODO: this bound could be tighter; we should be checking the two directions separately.
        let needle_neighbors = needle
            .graph
            .neighbors_directed(nodes.0, direction)
            .filter(|node| matches!(needle.neighbor_since[node.index()], [None, None]))
            .count();
        let haystack_neighbors = haystack
            .graph
            .neighbors_directed(nodes.1, direction)
            .filter(|node| matches!(haystack.neighbor_since[node.index()], [None, None]))
            .count();

        match self.problem {
            Problem::Exact => needle_neighbors == haystack_neighbors,
            Problem::InducedSubgraph => needle_neighbors <= haystack_neighbors,
            Problem::Subgraph => unreachable!("already handled by early return"),
        }
    }

    /// Are the cardinalities of the sets of neighbors of the current mapping state still feasible
    /// for a complete solution?
    ///
    /// This is a cheap lookahead heuristic; if the nodes currently mapped in the needle graph have
    /// more neighbors than the nodes currently mapped in the haystack graph, there can't possibly
    /// be a solution, regardless of the problem.  If we're not looking for a subgraph, then those
    /// two sets of objects have to map exactly.
    fn neighbor_counts_feasible(&self) -> bool {
        let needle_outs = self.needle.num_neighbors[0];
        let needle_ins = self.needle.num_neighbors[1];
        let haystack_outs = self.haystack.num_neighbors[0];
        let haystack_ins = self.haystack.num_neighbors[1];

        match self.problem {
            Problem::Exact => (needle_outs == haystack_outs) && (needle_ins == haystack_ins),
            Problem::InducedSubgraph | Problem::Subgraph => {
                (needle_outs <= haystack_outs) && (needle_ins <= haystack_ins)
            }
        }
    }

    /// Increase the call count of the mapper.  Returns `None` if we're already exhausted.
    #[inline]
    fn try_add_call(&mut self) -> Option<()> {
        self.num_calls += 1;
        self.call_limit
            .is_none_or(|limit| self.num_calls < limit)
            .then_some(())
    }
}

impl<N, H, NId, HId, NS, ES> Iterator for Vf2IntoIter<N, H, NId, HId, NS, ES>
where
    N: for<'a> alias::Vf2Graph<'a>,
    H: for<'a> alias::Vf2Graph<'a, EdgeType = N::EdgeType>,
    NId: Hash + Eq + Copy,
    HId: Copy,
    NS: Semantics<N::NodeWeight, H::NodeWeight>,
    ES: Semantics<N::EdgeWeight, H::EdgeWeight, Score = NS::Score>,
{
    type Item = Result<(IndexMap<NId, HId>, NS::Score), IsIsomorphicError<NS::Error, ES::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        // The overall strategy is a nested loop, where the "outer" loop is over unmapped nodes in
        // the "needle" graph, and the "inner" loop is over unmapped nodes in the "haystack" graph.
        // In both cases, the ordering of the nodes was set earlier, during the initial setup of
        // `self`, depending on the base ordering heuristic.  The "stack" is to save our position
        // in the loop iteration if/when we cede control back to the calling function.
        loop {
            let (mut nodes, kind) = match self.loop_stack.pop()? {
                Frame::ChooseNextHaystack { nodes, kind } => {
                    self.pop_state(nodes);
                    if let Some(haystack) =
                        self.haystack
                            .next_haystack_from(nodes.1.index() + 1, kind, self.problem)
                    {
                        ((nodes.0, haystack), kind)
                    } else {
                        continue;
                    }
                }
                Frame::ChooseNextNeedle => {
                    if let Some((needle, haystack, kind)) = self.next_candidates() {
                        ((needle, haystack), kind)
                    } else if self.needle.is_complete() {
                        // This only triggers if the needle graph is empty, and we're after a
                        // subgraph problem (since we'll already have exited if we're after an exact
                        // match and the numbers of nodes didn't match.
                        return Some(Ok((
                            self.mapping(),
                            self.score_stack
                                .last()
                                .expect("stack is always nonempty")
                                .clone(),
                        )));
                    } else {
                        continue;
                    }
                }
            };
            // At this point, we've got a pair of a needle node and a haystack node to try.  We loop
            // through the available haystack nodes until we find one that's feasible, at which
            // point we save our state in the haystack loop and move to finding the match to the
            // next needle node.  If none are feasible, we've finished this stack frame.
            loop {
                let feasible = match self.is_feasible(nodes) {
                    Ok(f) => f,
                    Err(e) => return Some(Err(e)),
                };
                if let Some(new_score) = feasible {
                    self.push_state(nodes, new_score);
                    if self.neighbor_counts_feasible() {
                        self.try_add_call()?;
                        self.loop_stack
                            .push(Frame::ChooseNextHaystack { nodes, kind });
                        if self.needle.is_complete() {
                            let score = self.score_stack.last().expect("always non-empty");
                            if let Some(Restriction::Decreasing(best)) = self.restriction.as_mut() {
                                *best = Some(score.clone());
                            };
                            return Some(Ok((self.mapping(), score.clone())));
                        }
                        self.loop_stack.push(Frame::ChooseNextNeedle);
                        break;
                    }
                    self.pop_state(nodes);
                }
                if let Some(haystack) =
                    self.haystack
                        .next_haystack_from(nodes.1.index() + 1, kind, self.problem)
                {
                    nodes.1 = haystack;
                } else {
                    break;
                }
            }
        }
    }
}
