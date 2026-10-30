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

//! Ops that produce rng keys and pseudo-random values.
//!
//! Every [`ProgramOp`] in this module (except [`Entropy`] which has no arguments) uses the same
//! broadcast semantics. For a key operand of shape `B` and an op holding a payload shape `S`:
//!
//!  1. The output shape is `broadcast(B, S)`.
//!  2. The value at output index `p` is a function of two indices: the key index, which is `p`
//!     projected onto `B`'s axes, and the payload index, which is `p` projected onto `S`'s axes.
//!     An axis of extent one projects to zero, so the key is shared along it.
//!  3. The key index selects the key. The payload index selects the counter position, and the
//!     key index never enters the counter.
//!

use ndarray::{ArcArrayD, IxDyn};
use rand_philox::Philox;
use thiserror::Error;

use super::{ProgramOp, QISKIT};
use crate::tensor::{
    DType, Dim, KeyTensor, PrngKind, Tensor, TensorError, TensorType, broadcast_dims,
    broadcast_shape,
};

/// An open stream of one [`PrngKind`], positioned at a block.
enum Stream {
    Philox4x32(Philox),
}

impl Stream {
    /// Open the stream of `kind` for `key_words` at `block`.
    ///
    /// # Panics
    ///
    /// Panics unless `key_words` is [`PrngKind::key_words`] long, which under normal usage of a
    /// stream is guaranteed by [`KeyTensor`].
    fn at_block(kind: PrngKind, key_words: &[u32], block: usize) -> Self {
        match kind {
            PrngKind::Philox4x32 => {
                let [lo, hi] = key_words else {
                    panic!("a philox4x32 key is two words")
                };
                Stream::Philox4x32(Philox::new([*lo, *hi], block as u128))
            }
        }
    }

    /// Fill `dest` with the words at the stream's position.
    fn fill(&mut self, dest: &mut [u32]) {
        match self {
            Stream::Philox4x32(philox) => philox.fill_u32(dest),
        }
    }
}

/// Return the block holding word `word` of a stream, and the word's offset within it.
fn word_block(kind: PrngKind, word: usize) -> (usize, usize) {
    let per_block = kind.block_words();
    (word / per_block, word % per_block)
}

/// Return how many blocks a payload of `shape` takes at `bits_per_position` bits each.
fn blocks_needed(kind: PrngKind, shape: &[usize], bits_per_position: usize) -> u128 {
    let bits = positions(shape).saturating_mul(bits_per_position as u128);
    bits.div_ceil((32 * kind.block_words()) as u128)
}

/// Check that we will not run out of words when requiring `bits_per_position` bits for every payload position.
fn check_blocks(
    kind: PrngKind,
    shape: &[usize],
    bits_per_position: usize,
) -> Result<(), RandomOpError> {
    let needed = blocks_needed(kind, shape, bits_per_position);

    match kind.blocks() {
        Some(afforded) if needed > afforded as u128 => Err(RandomOpError::CounterExhausted {
            kind,
            unit: CounterUnit::Blocks,
            needed,
            afforded: afforded as u128,
        }),
        _ => Ok(()),
    }
}

/// Return how many positions `shape` holds, saturating at `u128::MAX`.
fn positions(shape: &[usize]) -> u128 {
    shape
        .iter()
        .try_fold(1u128, |total, &size| total.checked_mul(size as u128))
        .unwrap_or(u128::MAX)
}

/// Return a key operand's generator and its batch axes.
///
/// Returns [`RandomOpError::KeyType`] if the operand is not a key.
fn key_type(operand: &TensorType) -> Result<(PrngKind, &[Dim]), RandomOpError> {
    match operand.dtype.prng_kind() {
        Some(kind) => Ok((kind, operand.shape.as_slice())),
        None => Err(RandomOpError::KeyType {
            actual: operand.clone(),
        }),
    }
}

/// Return the keys a key operand holds.
///
/// Returns [`RandomOpError::KeyType`] if the operand is not a key.
fn key_operand(operand: &Tensor) -> Result<&KeyTensor, RandomOpError> {
    operand.keys().ok_or_else(|| RandomOpError::KeyType {
        actual: operand.tensor_type(),
    })
}

/// Return the shape a key operand's `batch` and a payload `shape` produce together.
fn output_shape(batch: &[Dim], shape: &[usize]) -> Result<Vec<Dim>, TensorError> {
    let payload: Vec<Dim> = shape.iter().copied().map(Dim::Fixed).collect();
    broadcast_dims(batch, &payload)
}

/// Return every key's words, as one flat vector in row-major order.
fn flat_key_words(keys: &KeyTensor) -> std::borrow::Cow<'_, [u32]> {
    match keys.keys().as_slice() {
        Some(slice) => std::borrow::Cow::Borrowed(slice),
        None => std::borrow::Cow::Owned(keys.keys().iter().copied().collect()),
    }
}

/// Assemble key words into an array whose trailing axis holds them.
fn key_words_array(batch: &[usize], kind: PrngKind, key_words: Vec<u32>) -> ArcArrayD<u32> {
    let mut shape = batch.to_vec();
    shape.push(kind.key_words());
    ndarray::ArrayD::from_shape_vec(IxDyn(&shape), key_words)
        .expect("one key of the right width per batch position")
        .into_shared()
}

/// How a key tensor and a payload shape pair up, position by position, once broadcast.
///
/// Operations in this file generally use broadcast semantics. For example, a key of shape [6, 1]
/// given to a random integer sampler that internally holds dimension [1, 20] will result in
/// an integer array of shape [6, 20], but all values in the second axis will use the same key.
/// This struct helps keep track of this relationship.
struct Layout {
    /// The shape of the output.
    shape: Vec<usize>,
    /// The output strides of the key batch's axes, zero where an axis is shared.
    key_strides: Vec<usize>,
    /// The output strides of the payload shape's axes, zero where an axis is shared.
    draw_strides: Vec<usize>,
    /// The number of values in the output.
    len: usize,
}

impl Layout {
    /// Pair up `keys` with `shape`, or return [`TensorError::ShapeMismatch`] if they do not
    /// broadcast.
    fn new(keys: &KeyTensor, shape: &[usize]) -> Result<Self, TensorError> {
        let batch = keys.shape();
        let out = broadcast_shape(batch, shape)?;
        let rank = out.len();

        // Right-align a shape against the output and give each of its axes a stride. An axis of
        // extent one is shared along, so it contributes nothing to the position.
        let strides = |from: &[usize]| {
            let offset = rank - from.len();
            let mut strides = vec![0usize; rank];
            let mut stride = 1;
            for axis in (0..from.len()).rev() {
                strides[offset + axis] = if from[axis] == 1 { 0 } else { stride };
                stride *= from[axis];
            }
            strides
        };

        Ok(Self {
            key_strides: strides(batch),
            draw_strides: strides(shape),
            len: out.iter().product(),
            shape: out,
        })
    }

    /// Return how many values are in the output.
    fn len(&self) -> usize {
        self.len
    }

    /// Return which key, and which of its values, each output reads, in row-major order.
    ///
    /// The indices are carried along rather than recovered from each position, so that stepping to the
    /// next output is an increment.
    fn sources(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let rank = self.shape.len();
        let mut index = vec![0usize; rank];
        let mut key = 0usize;
        let mut counter = 0usize;
        let mut remaining = self.len;
        std::iter::from_fn(move || {
            if remaining == 0 {
                return None;
            }
            remaining -= 1;
            let current = (key, counter);
            // Step the odometer, undoing each axis that wraps and advancing the first that does
            // not. An axis contributes its stride on the way up and loses it all on the way round.
            for axis in (0..rank).rev() {
                index[axis] += 1;
                key += self.key_strides[axis];
                counter += self.draw_strides[axis];
                if index[axis] < self.shape[axis] {
                    break;
                }
                key -= self.key_strides[axis] * index[axis];
                counter -= self.draw_strides[axis] * index[axis];
                index[axis] = 0;
            }
            Some(current)
        })
    }
}

/// Keys drawn from the operating system's entropy.
#[derive(Clone)]
pub struct Entropy {
    kind: PrngKind,
    shape: Vec<usize>,
}

impl Entropy {
    /// Construct an `Entropy` op producing a tensor of `shape` keys for `kind`.
    pub fn new(kind: PrngKind, shape: Vec<usize>) -> Self {
        Self { kind, shape }
    }

    /// Return the generator the keys are for.
    pub fn kind(&self) -> PrngKind {
        self.kind
    }
}

impl ProgramOp for Entropy {
    type Error = RandomOpError;

    fn name(&self) -> &str {
        "entropy"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn arity(&self) -> usize {
        0
    }
    fn has_builtin_eval(&self) -> bool {
        true
    }
    fn infer_output_types(&self, _inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        Ok(vec![TensorType {
            dtype: DType::Key(self.kind),
            shape: self.shape.iter().copied().map(Dim::Fixed).collect(),
        }])
    }
    fn eval(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        let count = self.shape.iter().product::<usize>() * self.kind.key_words();
        // `fill` takes the bulk path, where a word-at-a-time loop would not.
        let mut key_words = vec![0u32; count];
        rand::RngExt::fill(&mut rand::rng(), key_words.as_mut_slice());
        Ok(vec![Tensor::Key(KeyTensor::new(
            self.kind,
            key_words_array(&self.shape, self.kind, key_words),
        )?)])
    }
}

/// Derive independent keys from a key.
#[derive(Clone)]
pub struct Split {
    shape: Vec<usize>,
}

impl Split {
    /// Construct a `Split` op producing `shape` keys per input key.
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl ProgramOp for Split {
    type Error = RandomOpError;

    fn name(&self) -> &str {
        "split"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn arity(&self) -> usize {
        1
    }
    fn has_builtin_eval(&self) -> bool {
        true
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        crate::unpack_operands!(self, inputs, [x]);
        let (kind, batch) = key_type(x)?;
        check_blocks(kind, &self.shape, 32 * kind.key_words())?;
        Ok(vec![TensorType {
            dtype: DType::Key(kind),
            shape: output_shape(batch, &self.shape)?,
        }])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x]);
        let keys = key_operand(x)?;
        let kind = keys.kind();
        let layout = Layout::new(keys, &self.shape)?;
        let width = kind.key_words();
        let source = flat_key_words(keys);

        // One draw per input key, reused by every output position that reads that key.
        let mut key_words = vec![0u32; layout.len() * width];
        // A derived key spans at most one block boundary, so reading from its first word's block
        // takes this much room at the most. The buffer is reused across outputs.
        let mut block = vec![0u32; kind.block_words() + width];
        for ((key, payload), into) in layout.sources().zip(key_words.chunks_mut(width)) {
            let words = &source[key * width..][..width];
            // A derived key takes `width` words starting at word `payload * width` of the stream.
            // Several keys can share a block.
            let (block_index, offset) = word_block(kind, payload * width);
            let read = &mut block[..offset + width];
            Stream::at_block(kind, words, block_index).fill(read);
            into.copy_from_slice(&read[offset..]);
        }
        Ok(vec![Tensor::Key(KeyTensor::new(
            kind,
            key_words_array(&layout.shape, kind, key_words),
        )?)])
    }
}

/// What a [`RandomOpError::CounterExhausted`] count is measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterUnit {
    Blocks,
    Substreams,
}

impl std::fmt::Display for CounterUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CounterUnit::Blocks => write!(f, "blocks"),
            CounterUnit::Substreams => write!(f, "substreams"),
        }
    }
}

/// Errors returned by the random ops.
#[derive(Debug, Error)]
pub enum RandomOpError {
    /// The operand is not a key.
    #[error("expected a key operand, got {actual}")]
    KeyType { actual: TensorType },
    /// The dtype does not hold raw bits.
    #[error("{kind} affords {afforded} {unit}, and this payload needs {needed}")]
    CounterExhausted {
        kind: PrngKind,
        unit: CounterUnit,
        needed: u128,
        afforded: u128,
    },
    /// A tensor operation failed.
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[cfg(test)]
mod test {
    use super::*;

    /// Return a tensor of `shape` keys laid out row-major.
    fn keys_of(shape: &[usize], key_words: Vec<u32>) -> Tensor {
        let kind = PrngKind::Philox4x32;
        Tensor::Key(KeyTensor::new(kind, key_words_array(shape, kind, key_words)).unwrap())
    }

    /// Known-answer vectors for philox4x32-10, from Random123's `tests/kat_vectors`.
    ///
    /// Each row is a counter, a key, and the four words the block function yields. These pin the
    /// protocol itself, so a change of generator crate or of our counter convention is caught here
    /// rather than by a downstream test asserting against whatever we happen to produce.
    const PHILOX4X32_10_KAT: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
        (
            [0, 0, 0, 0],
            [0, 0],
            [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
        ),
        (
            [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
            [0xffffffff, 0xffffffff],
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
        ),
        (
            [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
            [0xa4093822, 0x299f31d0],
            [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
        ),
    ];

    #[test]
    fn test_the_block_function_matches_the_published_vectors() {
        for (counter, key, expected) in PHILOX4X32_10_KAT {
            assert_eq!(
                rand_philox::philox4x32_10(counter, key),
                expected,
                "for counter {counter:08x?} and key {key:08x?}"
            );
        }
    }

    #[test]
    fn test_a_stream_is_addressed_by_block_and_by_substream() {
        // A block index and a substream index are the only two ways a stream is opened, and each
        // protocol widens them to its own counter. Both are checked against the block function,
        // so neither depends on how a counter is represented.
        let kind = PrngKind::Philox4x32;
        let key = [0x1234_5678u32, 0x9abc_def0];
        let block_of = |stream: &mut Stream| {
            let mut words = vec![0u32; kind.block_words()];
            stream.fill(&mut words);
            words
        };

        for block in [0usize, 1, 2, 4095, 4096] {
            assert_eq!(
                block_of(&mut Stream::at_block(kind, &key, block)),
                rand_philox::philox4x32_10([block as u32, 0, 0, 0], key),
                "at block {block}"
            );
        }
    }

    /// A key whose words are arbitrary, used by the golden-value tests below.
    fn golden_key() -> Tensor {
        keys_of(&[], vec![0xdead_beef, 0x1234_5678])
    }

    /// Three keys in a `[3, 1]` batch, used by the golden-value tests below.
    fn golden_batch() -> Tensor {
        keys_of(&[3, 1], vec![1, 2, 3, 4, 5, 6])
    }

    #[test]
    fn test_split_matches_its_recorded_output() {
        let derived = Split::new(vec![3]).eval(&[golden_key()]).unwrap().remove(0);
        assert_eq!(
            derived
                .keys()
                .unwrap()
                .keys()
                .iter()
                .copied()
                .collect::<Vec<u32>>(),
            [
                0xd8830217, 0xd18bb386, 0x0cb1c71d, 0xbdd96e0b, 0xeb24d1e9, 0x57890589,
            ],
            "three keys of two words each, taken from the key's stream in order"
        );

        let derived = Split::new(vec![2])
            .eval(&[golden_batch()])
            .unwrap()
            .remove(0);
        assert_eq!(
            derived
                .keys()
                .unwrap()
                .keys()
                .iter()
                .copied()
                .collect::<Vec<u32>>(),
            [
                0x0598de3a, 0x98d2802e, 0x270f8f9e, 0xeab709d3, 0x1955b60c, 0x46814a70, 0x682e0b3d,
                0x30193e3c, 0xa82aad38, 0xf48e6bfb, 0xab96c3d4, 0xc9bd190e,
            ]
        );
    }
}
