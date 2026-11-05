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

//! Pseudo-random number generators.

use std::fmt;

use ndarray::{ArcArrayD, IxDyn};

use super::TensorError;

/// A counter-based pseudo-random number generator[^random123].
///
/// Such a generator maps a key and a counter to a *block*, a fixed run of `u32` words. The
/// length of each block is given by [`Self::block_words`], and the total number of
/// addressable blocks is limited by the protocol's counter type (e.g. `u128`) and
/// queryable by [`Self::blocks`].
///
/// Since [`crate::QuantumProgram`] has a mandate to avoid runtime failures, a consumer needing
/// more blocks than the counter addresses is refused when the program is built, which is
/// statically checkable because every size is known by then.
///
/// [^random123]: John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw. 2011. "Parallel
///     random numbers: as easy as 1, 2, 3." <https://doi.org/10.1145/2063384.2063405>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrngKind {
    /// Philox 4x32-10: a two-word key, a 128-bit counter and blocks of four `u32`s.
    Philox4x32,
}

impl PrngKind {
    /// Return the number of `u32` words in a key.
    pub fn key_words(self) -> usize {
        match self {
            PrngKind::Philox4x32 => 2,
        }
    }

    /// Return the number of `u32` words one block of the stream yields.
    pub fn block_words(self) -> usize {
        match self {
            PrngKind::Philox4x32 => 4,
        }
    }

    /// Return how many blocks the counter addresses, or `None` if more than `usize::MAX`.
    ///
    /// Multiplied by [`Self::block_words`], this bounds the number of words a sampler of fixed
    /// draw length can access from one key.
    pub fn blocks(self) -> Option<usize> {
        match self {
            // 2^128 blocks, which no `usize` holds.
            PrngKind::Philox4x32 => None,
        }
    }
}

impl fmt::Display for PrngKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrngKind::Philox4x32 => write!(f, "philox4x32"),
        }
    }
}

/// A tensor of pseudo-random number generator keys for some [`PrngKind`].
///
/// The words of the key are stored as `u32` in the last axis, but the length of this axis
/// depends on [`PrngKind::key_words`] and is therefore opaque; [`Self::shape`] is the
/// shape of the key-tensor itself, ignoring this last axis, for example.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyTensor {
    kind: PrngKind,
    words: ArcArrayD<u32>,
}

impl KeyTensor {
    /// Construct a tensor of `kind` keys from `words` whose trailing axis holds the key words.
    ///
    /// Returns [`TensorError::KeyWords`] unless the trailing axis is [`PrngKind::key_words`].
    pub fn new(kind: PrngKind, words: ArcArrayD<u32>) -> Result<Self, TensorError> {
        let expected = kind.key_words();
        match words.shape().split_last() {
            Some((&last, _)) if last == expected => Ok(Self { kind, words }),
            _ => Err(TensorError::KeyWords {
                kind,
                expected,
                shape: words.shape().to_vec(),
            }),
        }
    }

    /// Return the kind of generator these keys are for.
    pub fn kind(&self) -> PrngKind {
        self.kind
    }

    /// Return the shape of the key tensor.
    pub fn shape(&self) -> &[usize] {
        let shape = self.words.shape();
        &shape[..shape.len() - 1]
    }

    /// Return the raw key words with their trailing axis.
    pub fn words(&self) -> &ArcArrayD<u32> {
        &self.words
    }

    /// Broadcast to a key tensor with the given `shape` if possible.
    pub(super) fn broadcast_to(&self, shape: &[usize]) -> Option<Self> {
        let mut target = shape.to_vec();
        target.push(self.kind.key_words());
        self.words.broadcast(IxDyn(&target)).map(|view| Self {
            kind: self.kind,
            words: view.to_owned().into_shared(),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Return a key tensor over `batch` keys, whose words count up from zero.
    fn keys(batch: &[usize]) -> KeyTensor {
        let kind = PrngKind::Philox4x32;
        let mut shape = batch.to_vec();
        shape.push(kind.key_words());
        let count = shape.iter().product::<usize>();
        let words = ndarray::ArrayD::from_shape_vec(IxDyn(&shape), (0..count as u32).collect())
            .unwrap()
            .into_shared();
        KeyTensor::new(kind, words).unwrap()
    }

    #[test]
    fn test_prng_kind_widths() {
        assert_eq!(PrngKind::Philox4x32.key_words(), 2);
        assert_eq!(PrngKind::Philox4x32.block_words(), 4);
    }

    #[test]
    fn test_a_wide_counter_addresses_more_than_usize_blocks() {
        // Philox's counter is 2^128 blocks, which overflows `usize`.
        assert_eq!(PrngKind::Philox4x32.blocks(), None);
    }

    #[test]
    fn test_prng_kind_display() {
        assert_eq!(PrngKind::Philox4x32.to_string(), "philox4x32");
    }

    #[test]
    fn test_shape_hides_the_words_axis() {
        // A lone key is a scalar, so indexing a tensor of keys cannot reach the words.
        assert_eq!(keys(&[]).shape(), &[] as &[usize]);
        assert_eq!(keys(&[3]).shape(), &[3]);
        assert_eq!(keys(&[2, 3]).shape(), &[2, 3]);

        assert_eq!(keys(&[3]).words().shape(), &[3, 2]);
    }

    #[test]
    fn test_new_rejects_a_wrong_width() {
        let kind = PrngKind::Philox4x32;
        for shape in [vec![3], vec![3, 1], vec![3, 4], vec![]] {
            let count = shape.iter().product::<usize>();
            let words = ndarray::ArrayD::from_shape_vec(IxDyn(&shape), vec![0u32; count])
                .unwrap()
                .into_shared();
            assert!(
                matches!(
                    KeyTensor::new(kind, words).unwrap_err(),
                    TensorError::KeyWords { expected: 2, shape: reported, .. }
                        if reported == shape
                ),
                "for words of shape {shape:?}"
            );
        }
    }

    #[test]
    fn test_broadcast_to_carries_the_words_axis() {
        let broadcast = keys(&[1]).broadcast_to(&[3]).unwrap();
        assert_eq!(broadcast.shape(), &[3]);

        // Every copy is the same key, which is what makes an extent-1 axis mean "shared".
        assert_eq!(
            broadcast.words().iter().copied().collect::<Vec<u32>>(),
            vec![0, 1, 0, 1, 0, 1]
        );
        assert_eq!(keys(&[3]).broadcast_to(&[4]), None);
    }
}
