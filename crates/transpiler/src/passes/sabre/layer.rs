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

use ndarray::prelude::*;
use rustworkx_core::petgraph::graph::IndexType;
use rustworkx_core::petgraph::prelude::*;
use std::num::NonZero;

use super::vec_map::VecMap;
use qiskit_circuit::{PhysicalQubit, VirtualQubit, nlayout::NLayout};

#[derive(Clone, Debug)]
pub struct Layers {
    layers: Box<[Layer]>,
    /// Mapping of `NodeIndex` to the location it is in the layer structure (by index).
    locations: VecMap<NodeIndex, Option<Location>>,
}
impl Layers {
    /// Create a new set of layers.
    ///
    /// # Panics
    ///
    /// If `num_layers` is zero - you have to have at least a front layer!
    pub fn new(num_layers: u16, num_qubits: u32, num_nodes: u32) -> Self {
        assert!(num_layers > 0, "must have at least a front layer!");
        Self {
            layers: vec![Layer::new(num_qubits); num_layers as usize].into_boxed_slice(),
            locations: vec![None; num_nodes as usize].into(),
        }
    }

    #[inline]
    pub fn num_layers(&self) -> NonZero<u16> {
        self.layers
            .len()
            .try_into()
            .and_then(|x: u16| x.try_into())
            .expect("constructor enforces front layer and upper bound")
    }

    /// The front layer.
    #[inline]
    pub fn front(&self) -> &Layer {
        self.layers
            .first()
            .expect("constructor enforces that the front layer exists")
    }

    #[inline]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    #[inline]
    pub fn apply_swap(&mut self, swap: [PhysicalQubit; 2]) {
        for layer in self.layers.iter_mut() {
            layer.apply_swap(swap);
        }
    }

    /// Set the layer of a given node.
    ///
    /// This removes the node from any layer it may already be in.
    #[inline]
    pub fn insert(
        &mut self,
        layer: u16,
        node: NodeIndex,
        qubits: [VirtualQubit; 2],
        layout: &NLayout,
    ) {
        if let Some(Location { layer, position }) = self.locations[node] {
            if let Some(moved) = self.layers[layer as usize].remove(position, layout) {
                self.locations[moved] = Some(Location { layer, position });
            }
        }
        let position = self.layers[layer as usize].insert(node, qubits, layout);
        self.locations[node] = Some(Location { layer, position });
    }

    /// Remove a node from the layer it is in.
    ///
    /// Returns the layer the node was in, if any.
    #[inline]
    pub fn remove(&mut self, node: NodeIndex, layout: &NLayout) -> Option<u16> {
        let Location { layer, position } = self.locations[node].take()?;
        if let Some(moved) = self.layers[layer as usize].remove(position, layout) {
            self.locations[moved] = Some(Location { layer, position });
        }
        Some(layer)
    }
}

/// Location of a node within the layer structure.
///
/// Both attributes are indices into arrays: `layer` says which layer the node is in, and `position`
/// says which position in [Layer::gates] corresponds to the node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Location {
    // There's deliberate padding in this struct to allow `Option<Location>` to be only 8 bytes.  It
    // seems somewhat unlikely we'd need 65,536 layers...
    /// Which layer the node is in.
    layer: u16,
    /// Which position in [Layer::gates] corresponds to this node.
    position: u32,
}

/// A container for the current non-routable parts of a layer.  This only ever holds two-qubit
/// gates, which are the only ones that contribute to routing costs.
#[derive(Clone, Debug)]
pub struct Layer {
    /// The qubits that each stored gate in the layer is active on.  For any given pair of qubits
    /// making up a gate in the layer, all three equalities hold:
    ///
    /// ```
    /// let virtuals: [VirtualQubit; 2];
    /// let physicals = virtuals.map(|q| layout[q]);
    /// physicals[0] == layer.other[physicals[1]];
    /// physicals[1] == layer.other[physicals[0]];
    /// layer.node[virtuals[0]] == layer.node[virtuals[1]];
    /// ```
    ///
    /// The indexing into this `Vec` is arbitrary but deterministic for a given input and random
    /// seed.  The indexing is tracked by [Layers::locations].
    gates: Vec<[VirtualQubit; 2]>,
    /// If a given qubit is active in the layer, `other[qubit]` will be the other `PhysicalQubit` in
    /// the corresponding gate.  If the qubit is not active, the maximum physical qubit is stored
    /// (we assume you don't use the absolute maximum).
    other: VecMap<PhysicalQubit, PhysicalQubit>,
    /// If a given qubit is active in the layer, the corresponding `node` will be the [SabreDAG]
    /// node corresponding to its gate.  If not active, the maximum node index is stored.
    node: VecMap<VirtualQubit, NodeIndex>,
}

impl Layer {
    #[inline]
    pub fn new(num_qubits: u32) -> Self {
        assert!(num_qubits < <PhysicalQubit as IndexType>::max().index() as u32);
        Layer {
            gates: Vec::with_capacity(num_qubits as usize / 2),
            other: vec![IndexType::max(); num_qubits as usize].into(),
            node: vec![IndexType::max(); num_qubits as usize].into(),
        }
    }

    /// Apply a physical swap to the current layout data structure.
    #[inline]
    fn apply_swap(&mut self, swap: [PhysicalQubit; 2]) {
        if self.other[swap[0]] == swap[1] {
            // The two qubits of the swap are both involved in the gate in this layer, so nothing
            // actually changes; they're both still each other's "other".
            return;
        }
        if let Some(other) = self.other_qubit(swap[0]) {
            debug_assert_eq!(self.other[other], swap[0]);
            self.other[other] = swap[1];
        }
        if let Some(other) = self.other_qubit(swap[1]) {
            debug_assert_eq!(self.other[other], swap[1]);
            self.other[other] = swap[0];
        }
        self.other.swap(swap[0], swap[1]);
    }

    /// Number of gates currently stored in the layer.
    #[inline]
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    /// Add a node into the layer, with the two qubits it operates on.
    ///
    /// Returns the location into its tracking vector that the gate was inserted at.
    #[inline]
    fn insert(&mut self, index: NodeIndex, qubits: [VirtualQubit; 2], layout: &NLayout) -> u32 {
        let [a, b] = qubits;
        debug_assert_eq!(self.other[layout[a]], IndexType::max());
        debug_assert_eq!(self.other[layout[b]], IndexType::max());
        debug_assert_eq!(self.node[a], IndexType::max());
        debug_assert_eq!(self.node[b], IndexType::max());
        self.other[layout[a]] = layout[b];
        self.other[layout[b]] = layout[a];
        self.node[a] = index;
        self.node[b] = index;
        self.gates.push(qubits);
        self.gates.len() as u32 - 1
    }

    /// Remove a node from the layer, given by position into the tracking vector.
    ///
    /// Returns the node that is now in this location in the tracking vector, if any.
    #[must_use]
    #[inline]
    fn remove(&mut self, location: u32, layout: &NLayout) -> Option<NodeIndex> {
        let replacement = if location as usize + 1 == self.len() {
            None
        } else {
            self.gates.last().map(|[a, _]| self.node[*a])
        };
        let [a, b] = self.gates.swap_remove(location as usize);
        debug_assert_eq!(self.other[layout[a]], layout[b]);
        debug_assert_eq!(self.other[layout[b]], layout[a]);
        self.other[layout[a]] = IndexType::max();
        self.other[layout[b]] = IndexType::max();
        self.node[a] = IndexType::max();
        self.node[b] = IndexType::max();
        replacement
    }

    /// Get the node index of an active qubit.
    #[inline]
    pub fn node_of(&self, qubit: VirtualQubit) -> Option<NodeIndex> {
        (self.node[qubit] != IndexType::max()).then_some(self.node[qubit])
    }

    /// Get the pair of the given qubit, if it's active in the layer.
    #[inline]
    pub fn other_qubit(&self, qubit: PhysicalQubit) -> Option<PhysicalQubit> {
        (self.other[qubit] != IndexType::max()).then_some(self.other[qubit])
    }

    /// Query whether a qubit has an active node.
    #[inline]
    pub fn is_active(&self, qubit: PhysicalQubit) -> bool {
        self.other_qubit(qubit).is_some()
    }

    /// Calculate the score _difference_ caused by this swap, compared to not making the swap.
    #[inline]
    pub fn score(&self, swap: [PhysicalQubit; 2], dist: &ArrayView2<f64>) -> f64 {
        let [a, b] = swap;
        let score = |cur: PhysicalQubit, new: PhysicalQubit| {
            let other = self.other[cur];
            if other == IndexType::max() {
                0.0
            } else {
                dist[[other.index(), new.index()]] - dist[[other.index(), cur.index()]]
            }
        };
        if self.other[a] == b {
            0.0
        } else {
            score(a, b) + score(b, a)
        }
    }

    /// Calculate the total absolute score of the layer for the set layout.
    #[inline]
    pub fn total_score(&self, layout: &NLayout, dist: &ArrayView2<f64>) -> f64 {
        self.gates
            .iter()
            .map(|&[a, b]| dist[[layout[a].index(), layout[b].index()]])
            .sum()
    }

    /// True if there are no nodes in the current layer.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Iterator over gates and the qubits they act on.
    pub fn iter_gates(&self) -> impl ExactSizeIterator<Item = (NodeIndex, [VirtualQubit; 2])> + '_ {
        self.gates
            .iter()
            .copied()
            .map(|qubits| (self.node[qubits[0]], qubits))
    }

    /// Iterator over the qubits that have active nodes on them.
    pub fn iter_active<'a>(
        &'a self,
        layout: &'a NLayout,
    ) -> impl Iterator<Item = PhysicalQubit> + 'a {
        self.gates.iter().flatten().map(|q| layout[*q])
    }
}
