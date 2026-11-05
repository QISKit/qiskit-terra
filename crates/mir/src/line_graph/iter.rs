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

//! Implementations of the iterator parts of [`LineGraph`].

use super::{Index, LineGraph, Node};
use std::marker::PhantomData;

mod sealed {
    use crate::line_graph::{Index, Node};

    /// Implementation of the direction-switching logic within iterators.
    ///
    /// This is used as a monomorphization code-generation trick to avoid having to manually define
    /// both "successor" and "predecessor" forms of iterators.
    pub trait Direction<T> {
        fn next(n: &Node<T>) -> Option<Index>;
    }

    /// Trait that powers the [`Indexed`](super::Indexed) iterator combinator.
    pub trait NextIndex {
        /// Get the tentative index of the next node to be emitted.  If this is `None`, the iterator
        /// terminates.  This is allowed to be `Some` even if the iterator doesn't emit the node.
        fn next_index(&self) -> Option<Index>;
    }
    /// Trait that powers the [`Indexed`](super::Indexed) iterator combinator for
    /// [`DoubleEndedIterator`].
    pub trait NextIndexBack {
        /// Get the tentative index of the next node to be emitted.  If this is `None`, the iterator
        /// terminates.  This is allowed to be `Some` even if the iterator doesn't emit the node.
        fn next_index_back(&self) -> Option<Index>;
    }
}
/// Marker object for use with iterators that indicates we're walking following the successors.
pub struct Successors;
impl<T> sealed::Direction<T> for Successors {
    fn next(n: &Node<T>) -> Option<Index> {
        n.next()
    }
}
/// Marker object for use with iterators that indicates we're walking following the predecessors.
pub struct Predecessors;
impl<T> sealed::Direction<T> for Predecessors {
    fn next(n: &Node<T>) -> Option<Index> {
        n.prev()
    }
}

macro_rules! impl_indexed {
    () => {
        /// Create an iterator which gives the current node index as well as the node itself.
        ///
        /// This is analogous to [`Iterator::enumerate`].
        #[inline]
        pub fn indexed(self) -> Indexed<Self> {
            Indexed(self)
        }
    };
}

/// An iterator that walks the graph from a given index until a given one (exclusive), or the end of
/// the sequence.
///
/// The order of iteration is determined by the `D` parameter, which is exactly one of
/// [`Successors`] or [`Predecessors`].
///
/// # Limitations
///
/// This is not an [`ExactSizeIterator`] because we can't know in constant time how long the chain
/// is.  It is not [`DoubleEndedIterator`] because we don't know whether the start and end points
/// are actually in the same sequence (or the main sequence, if the end point was `None`).
#[derive(Debug)]
pub struct Iter<'a, T, D: sealed::Direction<T>> {
    g: &'a LineGraph<T>,
    cur: Option<Index>,
    end: Option<Index>,
    seen: usize,
    dir: PhantomData<D>,
}
impl<'a, T, D: sealed::Direction<T>> Iter<'a, T, D> {
    #[inline]
    pub(super) fn new(g: &'a LineGraph<T>, from: Index, to: Option<Index>) -> Self {
        Self {
            g,
            cur: Some(from),
            end: to,
            seen: 0,
            dir: PhantomData,
        }
    }

    impl_indexed! {}
}
impl<'a, T, D: sealed::Direction<T>> Iterator for Iter<'a, T, D> {
    type Item = &'a Node<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.end {
            self.cur = None;
        }
        let cur = self.cur?;
        let node = self.g.data(cur)?;
        self.cur = <D as sealed::Direction<T>>::next(node);
        self.seen += 1;
        Some(node)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.cur.is_some() && self.cur != self.end {
            // This isn't very tight, it's just the best upper bound we have.
            (0, Some(self.g.node_count().saturating_sub(self.seen)))
        } else {
            (0, Some(0))
        }
    }
}
impl<T, D: sealed::Direction<T>> std::iter::FusedIterator for Iter<'_, T, D> {}
impl<T, D: sealed::Direction<T>> sealed::NextIndex for Iter<'_, T, D> {
    fn next_index(&self) -> Option<Index> {
        self.cur
    }
}

/// An iterator that walks the complete main sequence of the graph.
///
/// This implements [`DoubleEndedIterator`]; the graph automatically knows both ends of the main
/// sequence.
#[derive(Debug)]
pub struct IterMain<'a, T> {
    g: &'a LineGraph<T>,
    cur_front: Option<Index>,
    cur_back: Option<Index>,
    seen: usize,
}
impl<'a, T> IterMain<'a, T> {
    #[inline]
    pub(super) fn new(g: &'a LineGraph<T>) -> Self {
        Self {
            g,
            cur_front: g.head_index(),
            cur_back: g.tail_index(),
            seen: 0,
        }
    }

    impl_indexed! {}
}
impl<'a, T> Iterator for IterMain<'a, T> {
    type Item = &'a Node<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let cur_front = self.cur_front?;
        let node = self.g.data(cur_front)?;
        if self.cur_front == self.cur_back {
            self.cur_front = None;
        } else {
            self.cur_front = node.next();
        }
        if self.cur_front.is_none() {
            self.cur_back = None;
        }
        self.seen += 1;
        Some(node)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.cur_front.is_some() {
            if self.cur_front == self.cur_back {
                (0, Some(1))
            } else {
                // This isn't very tight, it's just the best upper bound we have.
                (0, Some(self.g.node_count().saturating_sub(self.seen)))
            }
        } else {
            (0, Some(0))
        }
    }
}
impl<'a, T> DoubleEndedIterator for IterMain<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let cur_back = self.cur_back?;
        let node = self.g.data(cur_back)?;
        if self.cur_front == self.cur_back {
            self.cur_back = None;
        } else {
            self.cur_back = node.prev();
        }
        if self.cur_back.is_none() {
            self.cur_front = None;
        }
        self.seen += 1;
        Some(node)
    }
}
impl<'a, T> std::iter::FusedIterator for IterMain<'a, T> {}
impl<T> sealed::NextIndex for IterMain<'_, T> {
    fn next_index(&self) -> Option<Index> {
        self.cur_front
    }
}
impl<T> sealed::NextIndexBack for IterMain<'_, T> {
    fn next_index_back(&self) -> Option<Index> {
        self.cur_back
    }
}

#[derive(Debug)]
pub struct Indexed<I>(I);
impl<I> Iterator for Indexed<I>
where
    I: Iterator + sealed::NextIndex,
{
    type Item = (Index, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.0.next_index()?;
        self.0.next().map(|x| (idx, x))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<I> DoubleEndedIterator for Indexed<I>
where
    I: DoubleEndedIterator + sealed::NextIndex + sealed::NextIndexBack,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let idx = self.0.next_index_back()?;
        self.0.next_back().map(|x| (idx, x))
    }
}
impl<I> std::iter::FusedIterator for Indexed<I> where
    I: Iterator + sealed::NextIndex + std::iter::FusedIterator
{
}
