// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Extension structs for use with Rayon parallelism.

// See https://github.com/rayon-rs/rayon/blob/v1.10.0/src/iter/plumbing/README.md (or a newer
// version) for more of an explanation of how Rayon's plumbing works.

use rayon::iter::plumbing::*;
use rayon::prelude::*;

pub trait ParallelSliceMutExt<T: Send>: ParallelSliceMut<T> {
    /// Create a parallel iterator over mutable chunks of uneven lengths for this iterator.
    ///
    /// # Panics
    ///
    /// Panics if the sums of the given lengths do not add up to the length of the slice.
    #[track_caller]
    fn par_uneven_chunks_mut<'len, 'data>(
        &'data mut self,
        chunk_lengths: &'len [usize],
    ) -> ParUnevenChunksMut<'len, 'data, T> {
        let mut_slice = self.as_parallel_slice_mut();
        let chunk_sum = chunk_lengths.iter().sum::<usize>();
        let slice_len = mut_slice.len();
        if chunk_sum != slice_len {
            panic!("given slices of total size {chunk_sum} for a chunk of length {slice_len}");
        }
        ParUnevenChunksMut {
            chunk_lengths,
            data: mut_slice,
        }
    }
}

impl<T: Send, S: ?Sized> ParallelSliceMutExt<T> for S where S: ParallelSliceMut<T> {}

/// Very similar to Rayon's [rayon::slice::ChunksMut], except that the lengths of the individual
/// chunks are arbitrary, provided they sum to the total length of the slice.
#[derive(Debug)]
pub struct ParUnevenChunksMut<'len, 'data, T> {
    chunk_lengths: &'len [usize],
    data: &'data mut [T],
}

impl<'data, T: Send + 'data> ParallelIterator for ParUnevenChunksMut<'_, 'data, T> {
    type Item = &'data mut [T];

    #[track_caller]
    fn drive_unindexed<C: UnindexedConsumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }
}

impl<'data, T: Send + 'data> IndexedParallelIterator for ParUnevenChunksMut<'_, 'data, T> {
    #[track_caller]
    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.chunk_lengths.len()
    }

    #[track_caller]
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(UnevenChunksMutProducer {
            chunk_lengths: self.chunk_lengths,
            data: self.data,
        })
    }
}

struct UnevenChunksMutProducer<'len, 'data, T: Send> {
    chunk_lengths: &'len [usize],
    data: &'data mut [T],
}

impl<'len, 'data, T: Send + 'data> Producer for UnevenChunksMutProducer<'len, 'data, T> {
    type Item = &'data mut [T];
    type IntoIter = UnevenChunksMutIter<'len, 'data, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self.chunk_lengths, self.data)
    }

    #[track_caller]
    fn split_at(self, index: usize) -> (Self, Self) {
        // Technically quadratic for a full-depth split, but let's worry about that later if needed.
        let data_mid = self.chunk_lengths[..index].iter().sum();
        let (chunks_left, chunks_right) = self.chunk_lengths.split_at(index);
        let (data_left, data_right) = self.data.split_at_mut(data_mid);
        (
            Self {
                chunk_lengths: chunks_left,
                data: data_left,
            },
            Self {
                chunk_lengths: chunks_right,
                data: data_right,
            },
        )
    }
}

#[must_use = "iterators do nothing unless consumed"]
struct UnevenChunksMutIter<'len, 'data, T> {
    chunk_lengths: &'len [usize],
    // The extra `Option` wrapper here is to satisfy the borrow checker while we're splitting the
    // `data` reference.  We need to consume `self`'s reference during the split before replacing
    // it, which means we need to temporarily set the `data` ref to some unowned value.
    // `Option<&mut [T]>` means we can replace it temporarily with the null reference, ensuring the
    // mutable aliasing rules are always upheld.
    data: Option<&'data mut [T]>,
}

impl<'len, 'data, T> UnevenChunksMutIter<'len, 'data, T> {
    fn new(chunk_lengths: &'len [usize], data: &'data mut [T]) -> Self {
        Self {
            chunk_lengths,
            data: Some(data),
        }
    }
}

impl<'data, T> Iterator for UnevenChunksMutIter<'_, 'data, T> {
    type Item = &'data mut [T];

    #[track_caller]
    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk_lengths.is_empty() {
            return None;
        }
        let (out_data, rem_data) = self
            .data
            .take()
            .unwrap()
            .split_at_mut(self.chunk_lengths[0]);
        self.chunk_lengths = &self.chunk_lengths[1..];
        self.data = Some(rem_data);
        Some(out_data)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.chunk_lengths.len(), Some(self.chunk_lengths.len()))
    }
}
impl<T> ExactSizeIterator for UnevenChunksMutIter<'_, '_, T> {}
impl<T> DoubleEndedIterator for UnevenChunksMutIter<'_, '_, T> {
    #[track_caller]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.chunk_lengths.is_empty() {
            return None;
        }
        let pos = self.chunk_lengths.len() - 1;
        let data_pos = self.data.as_ref().map(|x| x.len()).unwrap() - self.chunk_lengths[pos];
        let (rem_data, out_data) = self.data.take().unwrap().split_at_mut(data_pos);
        self.chunk_lengths = &self.chunk_lengths[..pos];
        self.data = Some(rem_data);
        Some(out_data)
    }
}
