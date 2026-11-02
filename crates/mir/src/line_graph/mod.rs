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

//! Definition of the base [`LineGraph`] data structure.

pub mod iter;

use std::{mem, num, ops};

/// A graph that supports only "lines", and whose indices are stable under node removal.
///
/// Each node in the graph can have zero or one successor, and zero or one predecessor.  Edges are
/// directed, and cannot carry data.
///
/// The graph tracks one "main sequence" of nodes.  Operations like [`push_front`](Self::push_front)
/// and [`push_back`](Self::push_back) modify this "main sequence".  It is permissible to add
/// "orphan" sequences, initialized with [`push_orphan`](Self::push_orphan), which will not appear
/// in the main-sequence iteration.
///
/// # Iteration
///
/// There are several supported iteration methods on the graph for shared references, such as
/// [`iter_from`](Self::iter_from), [`iter_main`](Self::iter_main) and
/// [`iter_range_back`](Self::iter_range_back).  There are no mutable-reference iterators because
/// this is near-impossible to achieve with the require random access through the graph's backing
/// data storage.
///
/// The various iteration methods typically have an [`indexed`](iter::Iter::indexed) method, which
/// attaches the [`Index`] of each node, similar to [`Iterator::enumerate`].
#[derive(Clone, Debug)]
pub struct LineGraph<T> {
    /// The individual slots that make up the graph.
    ///
    /// Each slot is either part of the data, or part of a free list.  Removals to `self` have to
    /// retain index stability, but we don't want them to leak permanently or require expensive
    /// scans to recover, so we track them in a free stack.
    ///
    /// As a performance/storage optimisation, we always allocate a `Free` into slot zero, and skip
    /// it in the `free_head`.  This is so `Index` can internally be a `NonZero` type, and
    /// `Option<Index>` is the same width.  Alternatives are to use something like `nonmax`, which
    /// adds an `xor` overhead to all indexes, or to wait for [Rust feature
    /// `pattern_types`](https://github.com/rust-lang/rust/issues/123646).
    slots: Vec<Slot<T>>,
    /// The first and last indices in `slots` that contain data nodes.  We only track a single line
    /// ourselves, though we permit the case that a user of the graph object has other "orphaned"
    /// lines within our data storage that they separately track (via
    /// [`push_orphan`](Self::push_orphan)).
    data_ends: Option<[Index; 2]>,
    /// The first slot in `slots` that should truly be considered "free".  This ignores index 0,
    /// which is leaked.
    free_head: Option<Index>,
    /// The length of the free chain.  This length already has slot allocations in the `Vec`, but
    /// they're eligible to be filled in.
    free_len: usize,
}
impl<T> LineGraph<T> {
    /// Create a new empty graph with no guaranteed additional capacity.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create a new graph with enough pre-allocated capacity to hold `cap` data nodes.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        let mut slots = Vec::with_capacity(cap.saturating_add(1));
        slots.push(Slot::Free(None));
        Self {
            slots,
            data_ends: None,
            free_head: None,
            free_len: 0,
        }
    }

    /// The first node in the main sequence.
    #[inline]
    pub fn head(&self) -> Option<&Node<T>> {
        self.data_ends.map(|[head, _]| self.data_unwrap(head))
    }
    /// The index of the first node in the main sequence.
    #[inline]
    pub fn head_index(&self) -> Option<Index> {
        self.data_ends.map(|[head, _]| head)
    }
    /// The last node in the main sequence.
    #[inline]
    pub fn tail(&self) -> Option<&Node<T>> {
        self.data_ends.map(|[_, tail]| self.data_unwrap(tail))
    }
    /// The index of the last node in the main sequence.
    #[inline]
    pub fn tail_index(&self) -> Option<Index> {
        self.data_ends.map(|[_, tail]| tail)
    }

    /// Place this node into an available slot and return its index.
    ///
    /// This will preferentially fill from the free chain, and only resize the underlying vector if
    /// required.
    fn allocate(&mut self, node: Node<T>) -> Index {
        let slot = Slot::Data(node);
        match self.free_head.take() {
            Some(free) => {
                let Slot::Free(next) = mem::replace(&mut self.slots[free.index()], slot) else {
                    panic!("free list pointed to a data node");
                };
                self.free_len -= 1;
                self.free_head = next;
                free
            }
            None => {
                self.slots.push(slot);
                Index::try_from(self.slots.len() - 1)
                    .expect("we assume `Index` is big enough in practice")
            }
        }
    }

    /// Get a reference to the data node corresponding to the given index, if it is valid.
    #[inline]
    pub fn data(&self, idx: Index) -> Option<&Node<T>> {
        self.slots.get(idx.index()).and_then(|slot| match slot {
            Slot::Data(node) => Some(node),
            Slot::Free(_) => None,
        })
    }

    /// Get a mutable reference to the data node corresponding to the given index, if it is valid.
    #[inline]
    pub fn data_mut(&mut self, idx: Index) -> Option<&mut Node<T>> {
        self.slots.get_mut(idx.index()).and_then(|slot| match slot {
            Slot::Data(node) => Some(node),
            Slot::Free(_) => None,
        })
    }

    /// Get a reference to the data at the given index.
    ///
    /// This is the same as the [`ops::Index`] implementation.  See [`Self::data`] for a
    /// non-panicking variant.
    ///
    /// **Panics** if the index is out of bounds, or does not point to a filled slot.
    #[inline]
    pub fn data_unwrap(&self, idx: Index) -> &Node<T> {
        self.data(idx)
            .unwrap_or_else(|| panic!("slot {} should be a data node", idx.index()))
    }

    /// Get a mutable reference to the data at the given index.
    ///
    /// This the same as the [`ops::IndexMut`] implementation.  See [`Self::data_mut`] for a
    /// non-panicking variant.
    ///
    /// **Panics** if the index is out of bounds, or does not point to a filled slot.
    #[inline]
    pub fn data_unwrap_mut(&mut self, idx: Index) -> &mut Node<T> {
        self.data_mut(idx)
            .unwrap_or_else(|| panic!("slot {} should be a data node", idx.index()))
    }

    /// Insert this weight as a new node that comes before `next`.
    ///
    /// **Panics** if `next` isn't a data node.
    pub fn insert_before(&mut self, next: Index, weight: T) -> Index {
        let prev = self[next].prev();
        let cur = self.allocate(Node {
            weight,
            next: Some(next),
            prev,
        });
        self.data_unwrap_mut(next).prev = Some(cur);
        if let Some(prev) = prev {
            self.data_unwrap_mut(prev).next = Some(cur);
        }
        if let Some([head, _]) = self.data_ends.as_mut()
            && *head == next
        {
            *head = cur;
        }
        cur
    }

    /// Insert this weight as a new node that comes after `prev`.
    ///
    /// **Panics** if `prev` isn't a data node.
    pub fn insert_after(&mut self, prev: Index, weight: T) -> Index {
        let next = self[prev].next();
        let cur = self.allocate(Node {
            weight,
            prev: Some(prev),
            next,
        });
        self.data_unwrap_mut(prev).next = Some(cur);
        if let Some(next) = next {
            self.data_unwrap_mut(next).prev = Some(cur);
        }
        if let Some([_, tail]) = self.data_ends.as_mut()
            && *tail == prev
        {
            *tail = cur;
        }
        cur
    }

    /// Add a new data node to the end of the graph.
    #[inline]
    pub fn push_back(&mut self, weight: T) -> Index {
        match self.data_ends {
            Some([head, tail]) => {
                let cur = self.allocate(Node {
                    weight,
                    prev: Some(tail),
                    next: None,
                });
                self.data_unwrap_mut(tail).next = Some(cur);
                self.data_ends = Some([head, cur]);
                cur
            }
            None => self.replace_main_with_node(weight),
        }
    }

    /// Add a new data node to the start of the graph.
    #[inline]
    pub fn push_front(&mut self, weight: T) -> Index {
        match self.data_ends {
            Some([head, tail]) => {
                let cur = self.allocate(Node {
                    weight,
                    next: Some(head),
                    prev: None,
                });
                self.data_unwrap_mut(head).prev = Some(cur);
                self.data_ends = Some([cur, tail]);
                cur
            }
            None => self.replace_main_with_node(weight),
        }
    }

    /// Add a new orphan node to the graph.  This will not be part of the "main sequence" implied by
    /// [`Self::head`] and [`Self::tail`]; it is not reachable from them.
    #[inline]
    pub fn push_orphan(&mut self, weight: T) -> Index {
        self.allocate(Node {
            weight,
            prev: None,
            next: None,
        })
    }

    /// Set the main sequence to be exactly this weight.
    ///
    /// Orphans any existing main sequence.  This is safe to use with an empty main sequence.
    #[inline]
    pub fn replace_main_with_node(&mut self, weight: T) -> Index {
        let cur = self.allocate(Node {
            weight,
            prev: None,
            next: None,
        });
        self.data_ends = Some([cur, cur]);
        cur
    }

    /// Remove the given node from the graph and return it.
    ///
    /// The predecessor and successor nodes (if any) are adjusted to link to each other.
    ///
    /// **Panics** if the node is not a data node.
    pub fn remove(&mut self, idx: Index) -> Node<T> {
        let Slot::Data(node) =
            mem::replace(&mut self.slots[idx.index()], Slot::Free(self.free_head))
        else {
            panic!("slot {} is not a data node", idx.index());
        };
        if let Some(prev) = node.prev {
            self.data_unwrap_mut(prev).next = node.next;
        }
        if let Some(next) = node.next {
            self.data_unwrap_mut(next).prev = node.prev;
        }
        self.free_head = Some(idx);
        self.free_len += 1;
        self.data_ends = self.data_ends.and_then(|[mut head, mut tail]| {
            if head == tail {
                // There's only one node in the main sequence.  If it's `idx`, then the main
                // sequence is removed and everything else is orphaned.  If not, then `idx` was on
                // an orphan line.
                return (head != idx).then_some([head, tail]);
            } else if head == idx {
                head = node.next.expect(
                    "the node is the head of a sequence of 2+ nodes, so it has a successor",
                );
            } else if tail == idx {
                tail = node.prev.expect(
                    "the node is the tail of a sequence of 2+ nodes, so it has a predecessor",
                );
            }
            Some([head, tail])
        });
        node
    }

    /// The number of data nodes stored in the graph.
    ///
    /// Note that this is greater than or equal to the length of the main sequence.  Equality holds
    /// when there are no "orphan" sequences.
    #[inline]
    pub fn node_count(&self) -> usize {
        // Remove 1 for the dummy `Free` slot, then any free capacity.  These should be `strict`
        // from MSRV 1.91; it's an internal logic error if they overflow
        self.slots
            .len()
            .checked_sub(1 + self.free_len)
            .expect("slots should always include one leaked free slot plus its free capacity")
    }

    /// The current capacity for data nodes in the graph.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots
            .capacity()
            // The leaked free slot is never fillable, so doesn't count to capacity.
            .checked_sub(1)
            .expect("capacity should always include a leaked free slot")
    }

    /// Iterate over data nodes starting from (and including) `idx`, in successor order.
    ///
    /// This is identical to [`iter_range(idx, None)`](Self::iter_range).  If you need the whole
    /// main sequence, try [`iter_main`](Self::iter_main).
    ///
    /// This returns an empty iterator if `idx` is not a valid data node.
    #[inline]
    pub fn iter_from(&self, idx: Index) -> iter::Iter<'_, T, iter::Successors> {
        iter::Iter::new(self, idx, None)
    }

    /// Iterate over data nodes starting from `from` (inclusive), until the sequence ends or the
    /// next index would be `to` (a half-open range), following the successor edges.
    ///
    /// If you need a double-inclusive iterator, use `self.data(to).and_then(|n| n.next())` as the
    /// bound instead.
    ///
    /// The iteration is empty if `from` is not a valid data node.  It is not an error if `to` is not
    /// reachable from `from`, the iteration will just run to the end of the sequence.
    #[inline]
    pub fn iter_range(
        &self,
        from: Index,
        to: Option<Index>,
    ) -> iter::Iter<'_, T, iter::Successors> {
        iter::Iter::new(self, from, to)
    }

    /// Iterate over data nodes starting from (and including) `idx`, in predecessor order.
    ///
    /// This is identical to [`iter_range_back(idx, None)`](Self::iter_range_back).  If you need the
    /// whole main sequence, try [`self.iter_main().rev()`](Self::iter_main).
    ///
    /// This returns an empty iterator if `idx` is not a valid data node.
    #[inline]
    pub fn iter_back_from(&self, idx: Index) -> iter::Iter<'_, T, iter::Predecessors> {
        iter::Iter::new(self, idx, None)
    }

    /// Iterate over data nodes starting from `from` (inclusive), until the sequence ends or the
    /// next index would be `to` (a half-open range), following the predecessor edges.
    ///
    /// If you need a double-inclusive iterator, use `self.data(to).and_then(|n| n.prev())` as the
    /// bound instead.
    ///
    /// The iteration is empty if `from` is not a valid data node.  It is not an error if `to` is not
    /// reachable from `from`, the iteration will just run to the end of the sequence.
    #[inline]
    pub fn iter_range_back(
        &self,
        from: Index,
        to: Option<Index>,
    ) -> iter::Iter<'_, T, iter::Predecessors> {
        iter::Iter::new(self, from, to)
    }

    /// Iterate over the entire main sequence, following successors.
    ///
    /// This will not include any orphan sequences.  The resulting iterator implements
    /// [`DoubleEndedIterator`], so you can iterate the main sequence following predecessors using
    /// its [`rev`](Iterator::rev) method.
    #[inline]
    pub fn iter_main(&self) -> iter::IterMain<'_, T> {
        iter::IterMain::new(self)
    }
}

impl<T> Default for LineGraph<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
impl<T> ops::Index<Index> for LineGraph<T> {
    type Output = Node<T>;
    #[inline]
    fn index(&self, idx: Index) -> &Self::Output {
        self.data_unwrap(idx)
    }
}
impl<T> ops::IndexMut<Index> for LineGraph<T> {
    #[inline]
    fn index_mut(&mut self, idx: Index) -> &mut Self::Output {
        self.data_unwrap_mut(idx)
    }
}

/// A slot in the graph.
#[derive(Clone, Copy, Debug)]
enum Slot<T> {
    /// Actual data.
    Data(Node<T>),
    /// An entry in the free list.  The payload is the next index (if any) in the chain.
    Free(Option<Index>),
}

/// An index into a [`LineGraph`].
///
/// This almost always points to a filled slot at the point of its public exposure from
/// [`LineGraph`] methods, though internal uses of it within [`LineGraph`] will use it to point to
/// free slots too, and mutations to the graph can invalidate existing indices.
///
/// [`Option<Index>`] is guaranteed to have the same size as [`Index`]; you can rely on the niche
/// optimization happening.  The value of the niche is **not** guaranteed and subject to change.
#[derive(
    Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, bytemuck::TransparentWrapper,
)]
#[repr(transparent)]
// The use of `NonZeroU32` is not intended to be API exposed or binding.  It would be better to use
// `u32::MAX` as the sentinel "bad" value, but it's hard to spell that _and_ permit the `Option<T>`
// niche optimisation that `Node` relies on to avoid being too large.
pub struct Index(num::NonZeroU32);
impl Index {
    /// Construct a new index from a regular numeric value.
    ///
    /// You can also use the [`TryFrom`] implementations for [`u32`] and [`usize`].
    ///
    /// Fails if `val` is not a valid value for an index.
    #[inline]
    pub fn new(val: u32) -> Option<Self> {
        Self::try_from(val).ok()
    }

    /// The numeric value of the index.
    #[inline]
    pub fn index(self) -> usize {
        self.0.get() as usize
    }
}
impl TryFrom<u32> for Index {
    type Error = num::TryFromIntError;
    #[inline]
    fn try_from(val: u32) -> Result<Self, Self::Error> {
        num::NonZeroU32::try_from(val).map(Self)
    }
}
impl TryFrom<usize> for Index {
    type Error = num::TryFromIntError;
    #[inline]
    fn try_from(val: usize) -> Result<Self, Self::Error> {
        Self::try_from(u32::try_from(val)?)
    }
}
// Assert that the niche optimization occurs with the type.  This is a documented contract of the
// `Index` type.
const _: () = assert!(mem::size_of::<Option<Index>>() == mem::size_of::<Index>());
// SAFETY: `Index` is a transparent wrapper around `NonZeroU32`, which implements this trait.
unsafe impl bytemuck::ZeroableInOption for Index {}
// SAFETY: `Index` is a transparent wrapper around `NonZeroU32`, which implements this trait.
unsafe impl bytemuck::PodInOption for Index {}

/// A node in a line graph.
///
/// This includes references to the node weight and its predecessor and successor edges (if any).
#[derive(Clone, Copy, Debug)]
pub struct Node<T> {
    weight: T,
    next: Option<Index>,
    prev: Option<Index>,
}
impl<T> Node<T> {
    /// Reference to the inner weight.
    #[inline]
    pub fn weight(&self) -> &T {
        &self.weight
    }

    /// Mutable reference to the inner weight.
    #[inline]
    pub fn weight_mut(&mut self) -> &mut T {
        &mut self.weight
    }

    /// Consume `self` into its weight.
    #[inline]
    pub fn into_weight(self) -> T {
        self.weight
    }

    /// The index of the successor node.
    #[inline]
    pub fn next(&self) -> Option<Index> {
        self.next
    }

    /// The index of the predecessor node.
    #[inline]
    pub fn prev(&self) -> Option<Index> {
        self.prev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_empty<T>(g: &LineGraph<T>) {
        assert_eq!(g.node_count(), 0);
        assert!(g.head().is_none());
        assert!(g.head_index().is_none());
        assert!(g.tail().is_none());
        assert!(g.tail_index().is_none());
    }

    #[test]
    fn basic_construction() {
        let mut g = LineGraph::<u8>::new();
        assert_empty(&g);
        assert_empty(&LineGraph::<()>::default());

        let idx = g.push_back(5);
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.head().map(Node::weight).copied(), Some(5));
        assert_eq!(g.tail().map(Node::weight).copied(), Some(5));
        assert_eq!(g.head_index(), Some(idx));
        assert_eq!(g.tail_index(), Some(idx));
        let node = g.data(idx).unwrap();
        assert_eq!(*node.weight(), 5);
        assert!(node.next().is_none());
        assert!(node.prev().is_none());
        let popped = g.remove(idx);
        assert_eq!(*popped.weight(), 5);
        assert!(popped.next().is_none());
        assert!(popped.prev().is_none());
        assert_empty(&g);

        let idx = g.push_front(4);
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.head().map(Node::weight).copied(), Some(4));
        assert_eq!(g.tail().map(Node::weight).copied(), Some(4));
        assert_eq!(g.head_index(), Some(idx));
        assert_eq!(g.tail_index(), Some(idx));
        let popped = g.remove(idx);
        assert_eq!(*popped.weight(), 4);
        assert!(popped.next().is_none());
        assert!(popped.prev().is_none());
        assert_empty(&g);
    }

    #[test]
    fn push_back_simple_chain() {
        let weights = [4u8, 8, 2, 3, 4, 5];
        let mut g = LineGraph::<u8>::new();
        assert!(
            // Nothing about the API guarantees this, but the point of this assertion is to ensure
            // that there's a test that _is_ pushing nodes outside the capacity of the free list.
            // You can just add more weights to the test.
            g.capacity() < weights.len(),
            "smoke test to ensure pushes outside capacity are covered"
        );
        let forward_indices = weights.iter().map(|w| g.push_back(*w)).collect::<Vec<_>>();
        assert_eq!(g.node_count(), weights.len());
        assert_eq!(g.head_index(), forward_indices.first().copied());
        assert_eq!(g.tail_index(), forward_indices.last().copied());
        let received_weights = forward_indices
            .iter()
            .map(|idx| *g[*idx].weight())
            .collect::<Vec<_>>();
        assert_eq!(weights.as_slice(), received_weights.as_slice());

        let mut iterated_indices = Vec::with_capacity(forward_indices.len());
        let mut cur = g.head_index();
        while let Some(idx) = cur {
            iterated_indices.push(idx);
            cur = g[idx].next();
        }
        assert_eq!(forward_indices.as_slice(), iterated_indices.as_slice());

        let mut iterated_indices_back = Vec::with_capacity(forward_indices.len());
        let mut cur = g.tail_index();
        while let Some(idx) = cur {
            iterated_indices_back.push(idx);
            cur = g[idx].prev();
        }
        iterated_indices_back.reverse();
        assert_eq!(forward_indices.as_slice(), iterated_indices_back.as_slice());
    }

    #[test]
    fn push_front_simple_chain() {
        let weights = [4u8, 8, 2, 3, 4, 5];
        let mut g = LineGraph::<u8>::new();
        let backward_indices = weights.iter().map(|w| g.push_front(*w)).collect::<Vec<_>>();
        assert_eq!(g.node_count(), weights.len());
        assert_eq!(g.head_index(), backward_indices.last().copied());
        assert_eq!(g.tail_index(), backward_indices.first().copied());
        let received_weights = backward_indices
            .iter()
            .map(|idx| *g[*idx].weight())
            .collect::<Vec<_>>();
        assert_eq!(weights.as_slice(), received_weights.as_slice());

        let mut iterated_indices = Vec::with_capacity(backward_indices.len());
        let mut cur = g.head_index();
        while let Some(idx) = cur {
            iterated_indices.push(idx);
            cur = g[idx].next();
        }
        iterated_indices.reverse();
        assert_eq!(backward_indices.as_slice(), iterated_indices.as_slice());

        let mut iterated_indices_back = Vec::with_capacity(backward_indices.len());
        let mut cur = g.tail_index();
        while let Some(idx) = cur {
            iterated_indices_back.push(idx);
            cur = g[idx].prev();
        }
        assert_eq!(
            backward_indices.as_slice(),
            iterated_indices_back.as_slice()
        );
    }

    #[test]
    fn mixed_push_front_push_back() {
        let weights = [1u8, 2, 3, 4, 5, 6, 7];
        let mut g = LineGraph::<u8>::with_capacity(weights.len());
        let cap = g.capacity();
        g.push_back(5);
        g.push_front(4);
        g.push_front(3);
        g.push_back(6);
        g.push_back(7);
        g.push_front(2);
        g.push_front(1);
        assert_eq!(g.capacity(), cap); // Not a tight test, but we shouldn't re-allocate.
        assert_eq!(g.node_count(), weights.len());

        let mut iterated_weights = Vec::with_capacity(weights.len());
        let mut cur = g.head_index();
        while let Some(idx) = cur {
            iterated_weights.push(*g[idx].weight());
            cur = g[idx].next();
        }
        assert_eq!(weights.as_slice(), iterated_weights.as_slice());

        let mut iterated_weights_back = Vec::with_capacity(weights.len());
        let mut cur = g.tail_index();
        while let Some(idx) = cur {
            iterated_weights_back.push(*g[idx].weight());
            cur = g[idx].prev();
        }
        iterated_weights_back.reverse();
        assert_eq!(weights.as_slice(), iterated_weights_back.as_slice());
    }

    #[test]
    fn orphans_do_not_affect_main() {
        let mut g = LineGraph::<u8>::new();
        let first_orphan_idx = g.push_orphan(5);
        assert_eq!(g.head_index(), None);
        assert_eq!(g.tail_index(), None);
        assert_eq!(g.node_count(), 1);
        assert_eq!(*g[first_orphan_idx].weight(), 5);

        let main_idx = g.push_back(2);
        assert_eq!(g.head_index(), Some(main_idx));
        assert_eq!(g.tail_index(), Some(main_idx));
        assert_eq!(g.node_count(), 2);

        let second_orphan_idx = g.push_orphan(7);
        assert_eq!(g.head_index(), Some(main_idx));
        assert_eq!(g.tail_index(), Some(main_idx));
        assert_eq!(g.node_count(), 3);
        assert_eq!(*g[second_orphan_idx].weight(), 7);

        assert_eq!(g.remove(first_orphan_idx).into_weight(), 5);
        assert_eq!(g.head_index(), Some(main_idx));
        assert_eq!(g.tail_index(), Some(main_idx));
        assert_eq!(g.node_count(), 2);

        let orphan_root = g.insert_before(second_orphan_idx, 3);
        let orphan_tail = g.insert_after(second_orphan_idx, 11);
        assert_eq!(g.head_index(), Some(main_idx));
        assert_eq!(g.tail_index(), Some(main_idx));
        assert_eq!(g.node_count(), 4);
        let expected_orphan_weights = [3u8, 7, 11];

        let mut cur = Some(orphan_root);
        let mut iterated_orphan_weights = Vec::new();
        while let Some(idx) = cur {
            iterated_orphan_weights.push(*g[idx].weight());
            cur = g[idx].next();
        }
        assert_eq!(
            expected_orphan_weights.as_slice(),
            iterated_orphan_weights.as_slice()
        );

        let mut cur = Some(orphan_tail);
        let mut iterated_orphan_weights_back = Vec::new();
        while let Some(idx) = cur {
            iterated_orphan_weights_back.push(*g[idx].weight());
            cur = g[idx].prev();
        }
        iterated_orphan_weights_back.reverse();
        assert_eq!(
            expected_orphan_weights.as_slice(),
            iterated_orphan_weights_back.as_slice()
        );
    }

    #[test]
    fn inserts_removes_in_main() {
        let mut g = LineGraph::<u8>::new();
        let head = g.push_front(1);
        let tail = g.push_back(4);
        let before_head = g.insert_before(head, 0);
        let after_tail = g.insert_after(tail, 5);
        assert_eq!(
            [g.head_index(), g.tail_index()],
            [Some(before_head), Some(after_tail)],
            "inserts before (after) the head (tail) should update the end tracking",
        );

        let after_head = g.insert_after(head, 2);
        let before_tail = g.insert_before(tail, 3);
        assert_eq!(
            [g.head_index(), g.tail_index()],
            [Some(before_head), Some(after_tail)],
            "inserts in the middle shouldn't affect the ends",
        );

        let weights_expected = [0u8, 1, 2, 3, 4, 5];

        let mut weights_forward = Vec::new();
        let mut cur = Some(before_head);
        while let Some(idx) = cur {
            weights_forward.push(*g[idx].weight());
            cur = g[idx].next();
        }
        assert_eq!(
            weights_expected.as_slice(),
            weights_forward.as_slice(),
            "after insertions the edges should be correct"
        );
        let mut weights_backward = Vec::new();
        let mut cur = Some(after_tail);
        while let Some(idx) = cur {
            weights_backward.push(*g[idx].weight());
            cur = g[idx].prev();
        }
        weights_backward.reverse();
        assert_eq!(
            weights_expected.as_slice(),
            weights_backward.as_slice(),
            "after insertions the edges should be correct"
        );

        assert_eq!(
            g.node_count(),
            weights_expected.len(),
            "node count should be tracked"
        );
        let node = g.remove(after_head);
        assert_eq!(
            (*node.weight(), node.prev(), node.next()),
            (2, Some(head), Some(before_tail)),
            "removed node should retain all its information",
        );
        let head_node = g[head];
        assert_eq!(
            (*head_node.weight(), head_node.prev(), head_node.next()),
            (1, Some(before_head), Some(before_tail)),
            "head node's successor should have been updated",
        );
        let before_tail_node = g[before_tail];
        assert_eq!(
            (
                *before_tail_node.weight(),
                before_tail_node.prev(),
                before_tail_node.next()
            ),
            (3, Some(head), Some(tail)),
            "before_tail node's predecessor should have been updated",
        );

        let before_head_node = g.remove(before_head);
        assert_eq!(
            (
                *before_head_node.weight(),
                before_head_node.prev(),
                before_head_node.next()
            ),
            (0, None, Some(head)),
        );
        assert_eq!(g.head_index(), Some(head), "graph's head should be updated");
        let head_node = g[head];
        assert_eq!(
            (*head_node.weight(), head_node.prev(), head_node.next()),
            (1, None, Some(before_tail)),
            "head node's predecessor should have been updated",
        );

        let after_tail_node = g.remove(after_tail);
        assert_eq!(
            (
                *after_tail_node.weight(),
                after_tail_node.prev(),
                after_tail_node.next()
            ),
            (5, Some(tail), None),
        );
        assert_eq!(g.tail_index(), Some(tail), "graph's tail should be updated");
        let tail_node = g[tail];
        assert_eq!(
            (*tail_node.weight(), tail_node.prev(), tail_node.next()),
            (4, Some(before_tail), None),
            "tail node's successor should have been updated",
        );
    }

    #[test]
    fn fallible_getters_do_not_panic() {
        let mut g = LineGraph::<u8>::new();
        let idx = g.push_back(1);
        g.remove(idx);
        assert!(g.data(idx).is_none());
        assert!(g.data_mut(idx).is_none());
    }

    #[test]
    #[should_panic]
    fn double_remove_panics() {
        let mut g = LineGraph::<u8>::new();
        let idx = g.push_back(1);
        g.remove(idx);
        g.remove(idx);
    }

    #[test]
    #[should_panic]
    fn accessing_removed_node_panics() {
        let mut g = LineGraph::<u8>::new();
        let idx = g.push_back(1);
        g.remove(idx);
        let _ = g[idx];
    }

    #[test]
    fn capacity_is_reused_after_free() {
        let requested = 8;
        let mut g = LineGraph::<u8>::with_capacity(requested);
        let actual = g.capacity();
        assert!(
            actual >= requested,
            "actual capacity must not be less than requested"
        );
        let mut indices = (0..actual).map(|_| g.push_back(0)).collect::<Vec<_>>();
        assert_eq!(g.node_count(), actual);
        assert_eq!(
            g.capacity(),
            actual,
            "capacity should not be changed by pushes within it"
        );
        for idx in &indices {
            g.remove(*idx);
        }
        assert_eq!(g.node_count(), 0);
        assert_eq!(
            g.capacity(),
            actual,
            "capacity should not be changed by removals"
        );

        let mut indices_again = (0..actual).map(|_| g.push_back(1)).collect::<Vec<_>>();
        indices.sort();
        indices_again.sort();
        assert_eq!(
            indices.as_slice(),
            indices_again.as_slice(),
            "pushing after removes should re-use the indices (in arbitrary order)"
        );
    }

    #[test]
    fn iter_basic() {
        let weights = [0, 1, 2, 3, 4, 5];
        let mut g = LineGraph::<u8>::new();
        for weight in weights {
            g.push_back(weight);
        }

        let main_fwd = g.iter_main().map(|n| *n.weight()).collect::<Vec<_>>();
        assert_eq!(weights.as_slice(), main_fwd.as_slice());
        let mut main_rev = g.iter_main().rev().map(|n| *n.weight()).collect::<Vec<_>>();
        main_rev.reverse();
        assert_eq!(weights.as_slice(), main_rev.as_slice());

        let from_head = g
            .iter_from(g.head_index().unwrap())
            .map(|n| *n.weight())
            .collect::<Vec<_>>();
        assert_eq!(weights.as_slice(), from_head.as_slice());

        let mut from_tail = g
            .iter_back_from(g.tail_index().unwrap())
            .map(|n| *n.weight())
            .collect::<Vec<_>>();
        from_tail.reverse();
        assert_eq!(weights.as_slice(), from_tail.as_slice());

        let range_fwd = g
            .iter_range(
                g.head().and_then(Node::next).unwrap(),
                g.tail().and_then(Node::prev),
            )
            .map(|n| *n.weight())
            .collect::<Vec<_>>();
        assert_eq!(&weights[1..4], range_fwd.as_slice());

        let mut range_back = g
            .iter_range_back(
                g.tail().and_then(Node::prev).unwrap(),
                g.head().and_then(Node::next),
            )
            .map(|n| *n.weight())
            .collect::<Vec<_>>();
        range_back.reverse();
        assert_eq!(&weights[2..5], range_back.as_slice());
    }

    #[test]
    fn iter_from_allows_orphans() {
        let mut g = LineGraph::<u8>::new();
        let head_idx = g.push_orphan(0);
        let mid_idx = g.insert_after(head_idx, 1);
        let tail_idx = g.insert_after(mid_idx, 2);

        let fwd = g
            .iter_from(head_idx)
            .map(|n| *n.weight())
            .collect::<Vec<_>>();
        assert_eq!([0, 1, 2].as_slice(), fwd.as_slice());
        let back = g
            .iter_back_from(tail_idx)
            .map(|n| *n.weight())
            .collect::<Vec<_>>();
        assert_eq!([2, 1, 0].as_slice(), back.as_slice());
    }

    #[test]
    fn iter_range_empty_range() {
        let mut g = LineGraph::<u8>::new();
        g.push_back(0);
        let mid = g.push_back(1);
        g.push_back(2);

        assert_eq!(g.iter_range(mid, Some(mid)).count(), 0);
        assert_eq!(g.iter_range_back(mid, Some(mid)).count(), 0);
    }

    #[test]
    fn iterators_ignore_invalid_indices() {
        let weights = [0, 1, 2, 3, 4];
        let mut g = LineGraph::<u8>::new();
        for weight in weights {
            g.push_back(weight);
        }
        let bad_head = g.head_index().unwrap();
        g.remove(bad_head);
        let bad_tail = g.tail_index().unwrap();
        g.remove(bad_tail);

        assert_eq!(g.iter_from(bad_head).count(), 0);
        assert_eq!(g.iter_from(bad_tail).count(), 0);
        assert_eq!(g.iter_back_from(bad_head).count(), 0);
        assert_eq!(g.iter_back_from(bad_tail).count(), 0);
        assert_eq!(g.iter_range(bad_head, None).count(), 0);
        assert_eq!(g.iter_range_back(bad_tail, None).count(), 0);

        let from_head = g
            .iter_range(g.head_index().unwrap(), Some(bad_head))
            .map(|n| *n.weight())
            .collect::<Vec<_>>();
        assert_eq!(&weights[1..4], from_head.as_slice());

        let mut from_tail = g
            .iter_range_back(g.tail_index().unwrap(), Some(bad_tail))
            .map(|n| *n.weight())
            .collect::<Vec<_>>();
        from_tail.reverse();
        assert_eq!(&weights[1..4], from_tail.as_slice());
    }

    #[test]
    fn iterators_work_on_empty() {
        let g = LineGraph::<u8>::new();
        let idx = Index::new(123).unwrap();
        assert_eq!(g.iter_from(idx).count(), 0);
        assert_eq!(g.iter_back_from(idx).count(), 0);
        assert_eq!(g.iter_range(idx, None).count(), 0);
        assert_eq!(g.iter_range_back(idx, None).count(), 0);
        assert_eq!(g.iter_main().count(), 0);
        assert_eq!(g.iter_main().rev().count(), 0);
    }

    #[test]
    fn iter_main_ignores_orphans() {
        let mut g = LineGraph::<u8>::new();
        g.push_orphan(10);
        assert_eq!(g.iter_main().count(), 0);
        assert_eq!(g.iter_main().rev().count(), 0);

        g.push_back(5);
        assert_eq!(
            g.iter_main()
                .map(|n| *n.weight())
                .collect::<Vec<_>>()
                .as_slice(),
            &[5]
        );
        assert_eq!(
            g.iter_main()
                .rev()
                .map(|n| *n.weight())
                .collect::<Vec<_>>()
                .as_slice(),
            &[5]
        );
    }

    #[test]
    fn iter_main_double_ended_interleave() {
        fn case(weights: &[u8], fwd_first: bool, desc: &str) {
            let mut g = LineGraph::with_capacity(weights.len());
            for weight in weights {
                g.push_back(*weight);
            }

            let mut fwd = Vec::new();
            let mut back = Vec::new();
            let mut iter = g.iter_main();
            if fwd_first {
                fwd.push(*iter.next().unwrap().weight());
            }
            // This is just a loop with two exit conditions.
            None::<std::convert::Infallible> = (|| loop {
                back.push(*iter.next_back()?.weight());
                fwd.push(*iter.next()?.weight());
            })();
            fwd.extend(back.drain(..).rev());

            assert_eq!(weights, fwd.as_slice(), "{desc}");
        }
        case(&[0, 1, 2, 3, 4, 5], false, "even, back first");
        case(&[0, 1, 2, 3, 4, 5], true, "even, fwd first");
        case(&[0, 1, 2, 3, 4, 5, 6], false, "odd, back first");
        case(&[0, 1, 2, 3, 4, 5, 6], true, "odd, fwd first");
    }

    #[test]
    fn indexed_iterators() {
        let mut g = LineGraph::<u8>::new();
        let indices = [4, 8, 1, 2, 3]
            .into_iter()
            .map(|w| g.push_back(w))
            .collect::<Vec<_>>();

        let indices_fwd = g
            .iter_from(g.head().and_then(Node::next).unwrap())
            .indexed()
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        assert_eq!(&indices[1..], indices_fwd.as_slice());

        let mut indices_back = g
            .iter_range_back(g.tail().and_then(Node::prev).unwrap(), g.head_index())
            .indexed()
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        indices_back.reverse();
        assert_eq!(&indices[1..4], indices_back.as_slice());

        let indices_main = g.iter_main().indexed().map(|(i, _)| i).collect::<Vec<_>>();
        assert_eq!(indices.as_slice(), indices_main.as_slice());

        let mut indices_main_rev = g
            .iter_main()
            .indexed()
            .rev()
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        indices_main_rev.reverse();
        assert_eq!(indices.as_slice(), indices_main_rev.as_slice());
    }
}
