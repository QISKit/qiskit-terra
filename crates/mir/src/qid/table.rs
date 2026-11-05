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

use std::ops;

use hashbrown::HashSet;
use rustworkx_core::petgraph::{acyclic::Acyclic, data::Build, prelude::*, visit};

use super::*;
use qiskit_util::IndexMap;

/// An error encountered while trying to allocate a new [`Qid`] in a resource table.
///
/// This is returned by [`ResourceTable::add_qid`].
#[derive(thiserror::Error, Clone, Debug)]
#[error("{kind}")]
pub struct AddQidError {
    info: QidInfo,
    kind: AddQidErrorKind,
}
impl AddQidError {
    /// View the [`QidInfo`] attached to this error.
    #[inline]
    pub fn info(&self) -> &QidInfo {
        &self.info
    }
    /// Retrieve the ownership of the info object that couldn't be added.
    #[inline]
    pub fn into_info(self) -> QidInfo {
        self.info
    }
    /// Get the underlying error.
    #[inline]
    pub fn kind(&self) -> AddQidErrorKind {
        self.kind
    }
}

/// The kind of error that triggered a [`AddQidError`]
#[derive(thiserror::Error, Clone, Copy, Debug)]
pub enum AddQidErrorKind {
    #[error("{0} is not present")]
    AbsentQid(Qid),
    #[error("{0} is not a bit owner of known size")]
    BadBase(Qid),
    #[error("index {} is out of bounds for {}", .0.bit(), .0.qid())]
    BadIndex(QidBit),
    #[error("a bit was used more than once")]
    DuplicateBit,
    #[error("the layout did not use all the bits in {0}")]
    IncompleteUse(Qid),
    #[error("too many qids")]
    NoSpace,
}
impl AddQidErrorKind {
    pub fn with_info(self, info: QidInfo) -> AddQidError {
        AddQidError { info, kind: self }
    }
}

/// The failure mechanism of [`ResourceTable::mark_contains`].
#[derive(thiserror::Error, Clone, Copy, Debug)]
pub enum ContainsError {
    #[error("{0} is not present")]
    AbsentQid(Qid),
    #[error("adding this relationship would cause a cycle")]
    Cycle,
}

/// The "context" for an entire table of [`Qid`]s.
///
/// This is responsible for:
/// - storing the [`QidInfo`] information for every [`Qid`] in a given [`QidSpace`];
/// - ensuring bit layouts are valid within the context;
/// - tracking the aliasing relations between different [`Qid`]s.
///
/// Generally, one can consider a [`Qid`] to be a key into an instance of this table.  The table is
/// associated with exactly one [`QidSpace`].  All qids stored in the table must match this space.
#[derive(Clone, Debug)]
pub struct ResourceTable {
    space: QidSpace,
    /// Mapping containing the per-qid info (nodes), with the direct "contains" relations pointing
    /// from the "bigger" qid to the smaller one.
    ///
    /// The graph is indexed by the table's [`Qid::value`]s.
    qids: Acyclic<Graph<QidInfo, ()>>,
}
impl ResourceTable {
    /// Initialize a new resource table with no capacity.
    #[inline]
    pub fn new(space: QidSpace) -> Self {
        Self::with_capacity(space, 0, 0)
    }

    /// Initialize a new resource table with capacity to expand to the given number of `qids`,
    /// including a number of `subrelations` without allocating.
    #[inline]
    pub fn with_capacity(space: QidSpace, qids: usize, subrelations: usize) -> Self {
        Self {
            space,
            qids: Acyclic::try_from_graph(Graph::with_capacity(qids, subrelations))
                .expect("empty graphs cannot have cycles"),
        }
    }

    /// Get the stored "information" of this [`Qid`], if it is in the table.
    ///
    /// The panicking variant of this is the implementation of [`Index`](std::ops::Index).
    #[inline]
    pub fn info(&self, qid: Qid) -> Option<&QidInfo> {
        self.index_from_qid(qid).map(|idx| &self.qids[idx])
    }

    /// Get the graph index that corresponds to this qid, if any.
    #[inline]
    fn index_from_qid(&self, qid: Qid) -> Option<NodeIndex> {
        (qid.space() == self.space && (qid.value() as usize) < self.qids.node_count())
            .then_some(NodeIndex::new(qid.value() as usize))
    }

    /// Get the qid that corresponds to this graph index.
    ///
    /// Assumes that the graph index is valid.
    #[inline]
    fn qid_from_index(&self, index: NodeIndex) -> Qid {
        Qid::from_parts(self.space, index.index() as u32)
    }

    /// Get the value of the next [`Qid`] that will be returned by [`add_qid`](Self::add_qid).
    #[inline]
    pub fn next_qid(&self) -> Option<Qid> {
        (self.qids.node_count() <= Qid::MAX_VALUE as usize)
            .then_some(Qid::from_parts(self.space, self.qids.node_count() as u32))
    }

    /// Validate a given explicit bit layout, returning an iterator over unique node indices that the
    /// layout completely contains.
    fn validate_layout(
        &self,
        layout: &BitLayout,
    ) -> Result<impl IntoIterator<Item = NodeIndex> + use<>, AddQidErrorKind> {
        let mut contains = IndexMap::<NodeIndex, u32>::default();
        let layout = match layout {
            BitLayout::Abstract | BitLayout::Base(_) => return Ok(contains.into_keys()),
            BitLayout::Alias(layout) => layout,
        };
        let set_bits = || layout.iter().filter_map(|x| *x);
        if set_bits().collect::<HashSet<_>>().len() != set_bits().count() {
            return Err(AddQidErrorKind::DuplicateBit);
        }
        for bit in set_bits() {
            let Some(idx) = self.index_from_qid(bit.qid()) else {
                return Err(AddQidErrorKind::AbsentQid(bit.qid()));
            };
            let other = &self.qids[idx];
            let &BitLayout::Base(width) = other.layout() else {
                return Err(AddQidErrorKind::BadBase(bit.qid()));
            };
            if bit.bit() >= width {
                return Err(AddQidErrorKind::BadIndex(bit));
            }
            // Since we know now that every `bit` use is
            // - unique in the layout,
            // - ... of a valid, owning qid in this table,
            // - ... which has an in-bounds index
            // then we can conclude that if we see the same qid for `width` times, then we've used
            // bits `0..width` of it, and thus it is used completely.  We can't overflow the
            // subtraction for the same reason.
            *contains.entry(idx).or_insert(width) -= 1;
        }
        if let Some(idx) = contains
            .iter()
            .find_map(|(idx, v)| (*v > 0).then_some(*idx))
        {
            let qid = self.qid_from_index(idx);
            return Err(AddQidErrorKind::IncompleteUse(qid));
        }
        Ok(contains.into_keys())
    }

    /// Add a new [`Qid`] to the graph.
    ///
    /// [`Qid`]s are assigned in consecutive, natural-number order.
    pub fn add_qid(&mut self, info: QidInfo) -> Result<Qid, AddQidError> {
        let Some(qid) = self.next_qid() else {
            return Err(AddQidErrorKind::NoSpace.with_info(info));
        };
        let contains = match self.validate_layout(info.layout()) {
            Ok(contains) => contains,
            Err(kind) => return Err(kind.with_info(info)),
        };
        let idx = self.qids.add_node(info);
        for contained in contains {
            self.qids
                .try_add_edge(idx, contained, ())
                .expect("adding an edge to a leaf not should not cause a cycle");
        }
        Ok(qid)
    }

    /// Mark a `bigger` [`Qid`] as containing the `smaller` one, for the purposes of alias analysis.
    ///
    /// Adding this "contains" relationship must not introduce a cycle in the hierarchy of qids.
    pub fn mark_contains(&mut self, bigger: Qid, smaller: Qid) -> Result<(), ContainsError> {
        let big_idx = self
            .index_from_qid(bigger)
            .ok_or(ContainsError::AbsentQid(bigger))?;
        let small_idx = self
            .index_from_qid(smaller)
            .ok_or(ContainsError::AbsentQid(smaller))?;
        self.qids
            .try_update_edge(big_idx, small_idx, ())
            .map_err(|_| ContainsError::Cycle)?;
        Ok(())
    }

    /// Check if there is any aliasing in a series of [`Qid`]s.
    ///
    /// Returns the error variant with a [`Qid`] that was not present in the table, if one is
    /// encountered before the question of overlap is resolved.
    pub fn try_overlap(&self, qids: impl IntoIterator<Item = Qid>) -> Result<bool, Qid> {
        let mut qids = qids.into_iter();
        // We need at least two qids for it to be possible to have an overlap.
        let Some(first) = qids.next() else {
            return Ok(false);
        };
        let Some(second) = qids.next() else {
            // There can't be an overlap, but `first` might not be a valid qid for this table.
            return self.index_from_qid(first).map(|_| false).ok_or(first);
        };
        // TODO: this is a pretty clear candidate for future optimisation.  There are several
        // straightforward possibilities that might be suitable:
        // - share the workspace between multiple calls (little-to-no downside other than potential
        //   cache invalidation of the `VisitMap`).
        // - cache the reachable leaves for each node directly (risks huge memory explosion).
        // - use a custom DFS that tracks the visit map with a "generation", so we don't have to
        //   reset it between nodes (though that uses more memory).
        let mut leaves = HashSet::<NodeIndex>::new();
        // Use pre-order not post-order because that (in theory) short-circuits faster.
        let mut dfs = visit::Dfs::empty(&self.qids);
        for qid in [first, second].into_iter().chain(qids) {
            dfs.reset(&self.qids);
            dfs.move_to(self.index_from_qid(qid).ok_or(qid)?);
            for idx in visit::Walker::iter(&mut dfs, &self.qids) {
                if self.qids.first_edge(idx, Direction::Outgoing).is_none() && !leaves.insert(idx) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Check if there is any aliasing in a series of [`Qid`]s.
    ///
    /// # Panics
    ///
    /// If any of the qids are not present in the table.  If you can't ensure that, use
    /// [`try_overlap`](Self::try_overlap) instead.
    #[inline]
    pub fn overlap(&self, qids: impl IntoIterator<Item = Qid>) -> bool {
        self.try_overlap(qids)
            .expect("caller should ensure all qids are present")
    }

    /// Iterate over the stored qids and their information objects.
    #[inline]
    pub fn iter_qids(&self) -> impl ExactSizeIterator<Item = (Qid, &QidInfo)> {
        self.qids
            .node_indices()
            .map(|idx| (self.qid_from_index(idx), &self.qids[idx]))
    }
}

impl ops::Index<Qid> for ResourceTable {
    type Output = QidInfo;

    fn index(&self, index: Qid) -> &Self::Output {
        &self.qids[self
            .index_from_qid(index)
            .expect("caller enforces qid should be present")]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn add_qid_invalid() {
        let single_alias_info =
            |qid, bit| QidInfo::new(BitLayout::Alias([Some(QidBit::new(qid, bit))].into()));

        let mut table = ResourceTable::new(QidSpace::Virtual);

        let next = table.next_qid().unwrap();
        assert!(
            matches!(
                table
                    .add_qid(single_alias_info(next, 0))
                    .unwrap_err()
                    .kind(),
                AddQidErrorKind::AbsentQid(q) if q == next,
            ),
            "qid not present at all"
        );
        assert_eq!(table.next_qid().unwrap(), next, "failure should not mutate");

        assert!(
            matches!(
                table
                    .add_qid(single_alias_info(Qid::phys(0), 0))
                    .unwrap_err()
                    .kind(),
                AddQidErrorKind::AbsentQid(q) if q == Qid::phys(0),
            ),
            "qid from wrong space"
        );
        assert_eq!(table.next_qid().unwrap(), next, "failure should not mutate");

        let good = table.add_qid(QidInfo::new(BitLayout::Base(1))).unwrap();
        let next = table.next_qid().unwrap();
        let duplicate_qid =
            BitLayout::Alias([Some(QidBit::new(good, 0)), Some(QidBit::new(good, 0))].into());
        assert!(
            matches!(
                table
                    .add_qid(QidInfo::new(duplicate_qid))
                    .unwrap_err()
                    .kind(),
                AddQidErrorKind::DuplicateBit,
            ),
            "layout references same bit twice"
        );
        assert_eq!(table.next_qid().unwrap(), next, "failure should not mutate");

        let oob_bit =
            BitLayout::Alias([Some(QidBit::new(good, 0)), Some(QidBit::new(good, 1))].into());
        assert!(
            matches!(
                table.add_qid(QidInfo::new(oob_bit)).unwrap_err().kind(),
                AddQidErrorKind::BadIndex(bit) if bit == QidBit::new(good, 1)
            ),
            "layout references out-of-range bit"
        );
        assert_eq!(table.next_qid().unwrap(), next, "failure should not mutate");

        let abstract_qid = table.add_qid(QidInfo::new(BitLayout::Abstract)).unwrap();
        let next = table.next_qid().unwrap();
        assert!(
            matches!(
                table
                    .add_qid(single_alias_info(abstract_qid, 0))
                    .unwrap_err()
                    .kind(),
                AddQidErrorKind::BadBase(q) if q == abstract_qid,
            ),
            "abstract qids can't be layout bases"
        );
        assert_eq!(table.next_qid().unwrap(), next, "failure should not mutate");

        let alias_qid = table.add_qid(single_alias_info(good, 0)).unwrap();
        let next = table.next_qid().unwrap();
        assert!(
            matches!(
                table
                    .add_qid(single_alias_info(alias_qid, 0))
                    .unwrap_err()
                    .kind(),
                AddQidErrorKind::BadBase(q) if q == alias_qid,
            ),
            "alias qids can't be layout bases"
        );
        assert_eq!(table.next_qid().unwrap(), next, "failure should not mutate");

        let two_bit_qid = table.add_qid(QidInfo::new(BitLayout::Base(2))).unwrap();
        let next = table.next_qid().unwrap();
        assert!(
            matches!(
                table
                    .add_qid(single_alias_info(two_bit_qid, 1))
                    .unwrap_err()
                    .kind(),
                AddQidErrorKind::IncompleteUse(q) if q == two_bit_qid,
            ),
            "must use all bits"
        );
        assert_eq!(table.next_qid().unwrap(), next, "failure should not mutate");
    }

    #[test]
    fn add_qid_valid_alias() {
        let mut table = ResourceTable::with_capacity(QidSpace::Virtual, 8, 16);
        let base_width = 2;
        let bases = (0..4)
            .map(|_| {
                table
                    .add_qid(QidInfo::new(BitLayout::Base(base_width)))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(!table.overlap(bases.iter().copied()));

        let flat_layout = QidInfo::new(BitLayout::Alias(
            bases
                .iter()
                .flat_map(|&qid| (0..base_width).map(move |bit| Some(QidBit::new(qid, bit))))
                .collect(),
        ));
        let flat_qid = table.add_qid(flat_layout).unwrap();
        assert!(bases.iter().all(|&qid| table.overlap([qid, flat_qid])));

        let with_holes_layout = QidInfo::new(BitLayout::Alias(
            [
                Some(QidBit::new(bases[0], 0)),
                Some(QidBit::new(bases[0], 1)),
                None,
                None,
                Some(QidBit::new(bases[1], 1)),
                Some(QidBit::new(bases[1], 0)),
            ]
            .into(),
        ));
        let with_holes_qid = table.add_qid(with_holes_layout).unwrap();
        assert!(
            [bases[0], bases[1]]
                .iter()
                .all(|&qid| table.overlap([qid, with_holes_qid]))
        );
        assert!(table.overlap([flat_qid, with_holes_qid]));
        assert!(
            ![bases[2], bases[3]]
                .iter()
                .any(|&qid| table.overlap([qid, with_holes_qid]))
        );
        // We should be able to mark an "aliasing" qid as containing another one "abstractly", even
        // if they've both got explicit layouts; this can model mutual exclusion in (e.g.) control
        // systems.
        table.mark_contains(with_holes_qid, bases[2]).unwrap();
        assert!(table.overlap([bases[2], with_holes_qid]));
        assert!(!table.overlap([bases[3], with_holes_qid]));
    }

    #[test]
    fn overlap_simple() {
        let mut table = ResourceTable::with_capacity(QidSpace::Physical, 8, 16);
        let qids = (0..8)
            .map(|_| table.add_qid(QidInfo::new(BitLayout::Abstract)).unwrap())
            .collect::<Vec<_>>();
        assert!(!table.overlap(qids.iter().copied()));

        table.mark_contains(qids[2], qids[0]).unwrap();
        assert!(
            table.mark_contains(qids[2], qids[0]).is_ok(),
            "should be idempotent"
        );
        table.mark_contains(qids[2], qids[1]).unwrap();
        assert!(table.overlap([qids[2], qids[0]]));
        assert!(table.overlap([qids[0], qids[2]]));
        assert!(table.overlap([qids[2], qids[1]]));
        assert!(table.overlap([qids[1], qids[2]]));
        assert!(!table.overlap([qids[0], qids[1]]));

        table.mark_contains(qids[3], qids[0]).unwrap();
        assert!(table.overlap([qids[2], qids[3]]), "both touch Qid(p:0)");

        table.mark_contains(qids[4], qids[2]).unwrap();
        assert!(
            table.overlap([qids[3], qids[4]]),
            "Qid(p:4) transitively overlaps Qid(p:0)"
        );

        assert!(!table.overlap([]), "empty set has no overlap");
        assert!(
            !qids.iter().any(|&qid| table.overlap([qid])),
            "single qids have no overlap"
        );

        assert!(
            qids.iter().all(|&qid| table.overlap([qid, qid])),
            "all qids overlap themselves"
        );

        let failures = [qids[0], qids[2], qids[5], qids[6]]
            .into_iter()
            .permutations(4)
            .filter(|perm| !table.overlap(perm.iter().copied()))
            .collect::<Vec<_>>();
        assert_eq!(
            &failures,
            &[] as &[Vec<_>],
            "overlap of Qid(p:0) and Qid(p:2) should be detected regardless of permutation"
        );

        assert!(matches!(
            table.try_overlap([Qid::virt(0)]),
            Err(q) if q == Qid::virt(0)
        ));
        assert!(matches!(
            table.try_overlap([qids[0], Qid::virt(0)]),
            Err(q) if q == Qid::virt(0)
        ));
        assert!(matches!(
            table.try_overlap([Qid::virt(0), qids[0]]),
            Err(q) if q == Qid::virt(0)
        ));
        assert!(matches!(
            table.try_overlap([qids[0], qids[1], Qid::virt(0)]),
            Err(q) if q == Qid::virt(0)
        ));
        assert!(matches!(
            table.try_overlap([qids[0], qids[1], table.next_qid().unwrap()]),
            Err(q) if q == table.next_qid().unwrap(),
        ));
    }

    #[test]
    fn overlap_implicit_layout() {
        let mut table = ResourceTable::with_capacity(QidSpace::Physical, 8, 16);
        let bases = (0..4)
            .map(|_| table.add_qid(QidInfo::new(BitLayout::Base(1))).unwrap())
            .collect::<Vec<_>>();
        let aliases = [
            [bases[0], bases[1]],
            [bases[2], bases[3]],
            [bases[1], bases[2]],
        ]
        .map(|[l, r]| {
            table
                .add_qid(QidInfo::new(BitLayout::Alias(
                    [Some(QidBit::new(l, 0)), Some(QidBit::new(r, 0))].into(),
                )))
                .unwrap()
        });
        assert!(table.overlap([aliases[0], aliases[2]]));
        assert!(table.overlap([aliases[1], aliases[2]]));
        assert!(!table.overlap([aliases[0], aliases[1]]));
    }

    #[test]
    fn mark_contains_invalids() {
        let mut table = ResourceTable::new(QidSpace::Virtual);
        let qids = (0..4)
            .map(|_| table.add_qid(QidInfo::new(BitLayout::Abstract)).unwrap())
            .collect::<Vec<_>>();

        assert!(
            matches!(
                table.mark_contains(qids[0], qids[0]),
                Err(ContainsError::Cycle)
            ),
            "self-loops forbidden"
        );
        table.mark_contains(qids[1], qids[0]).unwrap();
        assert!(matches!(
            table.mark_contains(qids[0], qids[1]),
            Err(ContainsError::Cycle)
        ));
        table.mark_contains(qids[2], qids[1]).unwrap();
        assert!(matches!(
            table.mark_contains(qids[0], qids[1]),
            Err(ContainsError::Cycle)
        ));
        assert!(matches!(
            table.mark_contains(qids[0], qids[2]),
            Err(ContainsError::Cycle)
        ));

        assert!(matches!(
            table.mark_contains(Qid::phys(100), qids[0]),
            Err(ContainsError::AbsentQid(q)) if q == Qid::phys(100)
        ));
    }
}
