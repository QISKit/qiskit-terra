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
use rand::distr::{Distribution, Uniform};
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

    /// Open the stream of `kind` for `key_words` at the start of `output`'s substream.
    ///
    /// A *substream* is a run of `stride` consecutive blocks, and substream `output` begins at
    /// block `output * stride`.
    ///
    /// A common use for substreams is rejection sampling. For example, drawing a uniform
    /// random integers between two values separated by a value that is not a multiple of 2 typically
    /// use rejection sampling, where rejection consumes words in geometrically distributed runs: an
    /// attempt takes a fixed few
    /// words and is retried with some probability `p < 1`, so a draw exceeds `n` attempts with
    /// probability `p^n`. [`Uniform`] and the other [`rand`](https://docs.rs/rand) integer
    /// distributions use Lemire's method over a 32-bit word, where `p < 1/2` and an attempt takes
    /// two words. A stride of `n` blocks therefore holds `n * w / 2` attempts, for `w` words per
    /// block from [`PrngKind::block_words`], and is overrun with probability under
    /// `2^(-n * w / 2)`. A draw that does overrun continues into the next output's substream, and
    /// the two variates are then correlated. This is a risk we accept.
    ///
    /// # Panics
    ///
    /// Panics unless `key_words` is [`PrngKind::key_words`] long.
    fn at_substream(kind: PrngKind, key_words: &[u32], output: usize, stride: usize) -> Self {
        Self::at_block(kind, key_words, output.wrapping_mul(stride))
    }

    /// Fill `dest` with the words at the stream's position.
    fn fill(&mut self, dest: &mut [u32]) {
        match self {
            Stream::Philox4x32(philox) => philox.fill_u32(dest),
        }
    }

    /// Return the stream as an [`rand::Rng`], so that a distribution can sample from it.
    fn as_rng(&mut self) -> &mut dyn rand::Rng {
        match self {
            Stream::Philox4x32(philox) => philox,
        }
    }
}

/// Return the block holding word `word` of a stream, and the word's offset within it.
fn word_block(kind: PrngKind, word: usize) -> (usize, usize) {
    let per_block = kind.block_words();
    (word / per_block, word % per_block)
}

/// Return how many substreams of `stride` blocks a key affords, or `None` if more than
/// `usize::MAX`.
fn substreams(kind: PrngKind, stride: usize) -> Option<usize> {
    Some(kind.blocks()? / stride)
}

/// Check that we will not run out of substreams given a `stride` size and sample shape.
fn check_substreams(kind: PrngKind, shape: &[usize], stride: usize) -> Result<(), RandomOpError> {
    let needed = positions(shape);
    match substreams(kind, stride) {
        Some(afforded) if needed > afforded as u128 => Err(RandomOpError::CounterExhausted {
            kind,
            unit: CounterUnit::Substreams,
            needed,
            afforded: afforded as u128,
        }),
        _ => Ok(()),
    }
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
    match keys.words().as_slice() {
        Some(slice) => std::borrow::Cow::Borrowed(slice),
        None => std::borrow::Cow::Owned(keys.words().iter().copied().collect()),
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

/// Draw uniformly distributed bits from a key into a specific dtype.
#[derive(Debug, Clone)]
pub struct RandomBits {
    dtype: DType,
    shape: Vec<usize>,
}

impl RandomBits {
    /// Construct a `RandomBits` op producing `shape` values of `dtype` per key.
    ///
    /// Returns [`RandomOpError::NotBits`] unless `dtype` is [`DType::Bit`] or an unsigned integer.
    pub fn new(dtype: DType, shape: Vec<usize>) -> Result<Self, RandomOpError> {
        if bits_per_value(dtype).is_none() {
            return Err(RandomOpError::NotBits { dtype });
        }
        Ok(Self { dtype, shape })
    }

    /// Return the dtype of the values drawn.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Return how many bits one value of `dtype` takes, or `None` if it does not hold raw bits.
fn bits_per_value(dtype: DType) -> Option<usize> {
    match dtype {
        DType::Bit => Some(1),
        DType::U8 => Some(8),
        DType::U16 => Some(16),
        DType::U32 => Some(32),
        DType::U64 => Some(64),
        _ => None,
    }
}

impl ProgramOp for RandomBits {
    type Error = RandomOpError;

    fn name(&self) -> &str {
        "random_bits"
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
        let bits = bits_per_value(self.dtype).expect("the constructor checked the dtype");
        check_blocks(kind, &self.shape, bits)?;
        Ok(vec![TensorType {
            dtype: self.dtype,
            shape: output_shape(batch, &self.shape)?,
        }])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x]);
        let keys = key_operand(x)?;
        let layout = Layout::new(keys, &self.shape)?;
        let bits = bits_per_value(self.dtype).expect("the constructor checked the dtype");
        let width = keys.kind().key_words();
        let source = flat_key_words(keys);
        // How many values one block holds, and so how far one block advances the payload index. A
        // value never straddles a block, since every width divides a block.
        let per_block = (32 * keys.kind().block_words()) / bits;

        /// Draw each key's values, then gather them in output order.
        ///
        /// `$value` reads value `i` out of the block of words a key drew.
        macro_rules! draw {
            ($variant:ident, $t:ty, $value:expr) => {{
                let mut block = vec![0u32; keys.kind().block_words()];
                let mut cached: Option<(usize, usize)> = None;
                let values: Vec<$t> = layout
                    .sources()
                    .map(|(key, payload)| {
                        let wanted = (key, payload / per_block);
                        if cached != Some(wanted) {
                            let words = &source[key * width..][..width];
                            Stream::at_block(keys.kind(), words, wanted.1).fill(&mut block);
                            cached = Some(wanted);
                        }
                        $value(&block, payload % per_block)
                    })
                    .collect();
                Tensor::$variant(
                    ndarray::ArrayD::from_shape_vec(IxDyn(&layout.shape), values)
                        .expect("one value per output position")
                        .into_shared(),
                )
            }};
        }

        // A value takes the next `bits` bits of its key's stream, least significant first.
        Ok(vec![match self.dtype {
            DType::U64 => draw!(U64, u64, |b: &[u32], i: usize| (b[2 * i] as u64)
                | ((b[2 * i + 1] as u64) << 32)),
            DType::U32 => draw!(U32, u32, |b: &[u32], i: usize| b[i]),
            DType::U16 => {
                draw!(U16, u16, |b: &[u32], i: usize| (b[i / 2] >> (16 * (i % 2)))
                    as u16)
            }
            DType::U8 => draw!(U8, u8, |b: &[u32], i: usize| (b[i / 4] >> (8 * (i % 4)))
                as u8),
            DType::Bit => {
                draw!(Bit, u8, |b: &[u32], i: usize| ((b[i / 32] >> (i % 32)) & 1)
                    as u8)
            }
            dtype => unreachable!("the constructor refuses {dtype}"),
        }])
    }
}

/// How many blocks [`UniformInt`] reserves for each value it draws.
///
/// See [`Stream::at_substream`] for how a stride is chosen. At four words per block this holds
/// 256 attempts, so it is overrun with probability under `2^-256`. The measured rejection rate
/// of `Uniform<i64>` over its worst range is 1/4, which gives `4^-256`.
const UNIFORM_INT_STRIDE: usize = 1 << 8;

/// Draw integers uniformly from an inclusive range.
#[derive(Debug, Clone)]
pub struct UniformInt {
    shape: Vec<usize>,
    low: i64,
    high: i64,
}

impl UniformInt {
    /// Construct a `UniformInt` op drawing `shape` values from `low..=high` per key.
    ///
    /// Returns [`RandomOpError::EmptyRange`] if `low` exceeds `high`.
    pub fn new(shape: Vec<usize>, low: i64, high: i64) -> Result<Self, RandomOpError> {
        if low > high {
            return Err(RandomOpError::EmptyRange { low, high });
        }
        Ok(Self { shape, low, high })
    }

    /// Return the inclusive bounds values are drawn from.
    pub fn range(&self) -> (i64, i64) {
        (self.low, self.high)
    }
}

impl ProgramOp for UniformInt {
    type Error = RandomOpError;

    fn name(&self) -> &str {
        "uniform_int"
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
        check_substreams(kind, &self.shape, UNIFORM_INT_STRIDE)?;
        Ok(vec![TensorType {
            dtype: DType::I64,
            shape: output_shape(batch, &self.shape)?,
        }])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x]);
        let keys = key_operand(x)?;
        let layout = Layout::new(keys, &self.shape)?;
        let range = Uniform::new_inclusive(self.low, self.high)
            .expect("low <= high is checked in the constructor");

        let width = keys.kind().key_words();
        let source = flat_key_words(keys);
        let mut values = Vec::with_capacity(layout.len());
        for (which, position) in layout.sources() {
            let key = &source[which * width..][..width];
            let mut stream = Stream::at_substream(keys.kind(), key, position, UNIFORM_INT_STRIDE);
            values.push(range.sample(stream.as_rng()));
        }
        let array = ndarray::ArrayD::from_shape_vec(IxDyn(&layout.shape), values)
            .expect("one value per output position");
        Ok(vec![Tensor::I64(array.into_shared())])
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
    #[error("{dtype} does not hold raw bits: expected Bit or an unsigned integer")]
    NotBits { dtype: DType },
    /// The range holds no values.
    #[error("the range {low}..={high} is empty")]
    EmptyRange { low: i64, high: i64 },
    /// The generator's counter is too narrow for the payload shape.
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

    /// Return a shape of fixed axes.
    fn fixed(sizes: &[usize]) -> Vec<Dim> {
        sizes.iter().copied().map(Dim::Fixed).collect()
    }

    /// Return a tensor of `shape` keys laid out row-major.
    fn keys_of(shape: &[usize], key_words: Vec<u32>) -> Tensor {
        let kind = PrngKind::Philox4x32;
        Tensor::Key(KeyTensor::new(kind, key_words_array(shape, kind, key_words)).unwrap())
    }

    /// Return a tensor of `shape` distinct keys, numbered from `first`.
    fn numbered_keys(shape: &[usize], first: u32) -> Tensor {
        let count: usize = shape.iter().product();
        let key_words = (0..count as u32)
            .flat_map(|i| [first + 2 * i, first + 2 * i + 1])
            .collect();
        keys_of(shape, key_words)
    }

    /// Return a single key, distinct for each `label`.
    fn key_from(label: u32) -> Tensor {
        keys_of(&[], vec![label, label.wrapping_mul(2_654_435_761)])
    }

    /// Return the `U32` words a tensor holds.
    fn words_of(tensor: &Tensor) -> Vec<u32> {
        let Tensor::U32(array) = tensor else {
            panic!("expected U32, got {}", tensor.dtype())
        };
        array.iter().copied().collect()
    }

    /// Return the `I64` values a tensor holds.
    fn ints_of(tensor: &Tensor) -> Vec<i64> {
        let Tensor::I64(array) = tensor else {
            panic!("expected I64, got {}", tensor.dtype())
        };
        array.iter().copied().collect()
    }

    #[test]
    fn test_names_arities_and_builtin_eval() {
        for (op, name, arity) in [
            (
                &Entropy::new(PrngKind::Philox4x32, vec![])
                    as &dyn ProgramOp<Error = RandomOpError>,
                "qiskit.entropy",
                0,
            ),
            (&Split::new(vec![]), "qiskit.split", 1),
            (
                &RandomBits::new(DType::U32, vec![]).unwrap(),
                "qiskit.random_bits",
                1,
            ),
            (
                &UniformInt::new(vec![], 0, 9).unwrap(),
                "qiskit.uniform_int",
                1,
            ),
        ] {
            assert_eq!(op.full_name(), name);
            assert_eq!(op.arity(), arity, "for {name}");
            assert!(op.has_builtin_eval(), "for {name}");
        }
    }

    #[test]
    fn test_infer_broadcasts_the_batch_against_the_payload_shape() {
        let key_type = |shape: Vec<Dim>| TensorType {
            dtype: DType::Key(PrngKind::Philox4x32),
            shape,
        };

        // A batch of keys and a payload shape broadcast, right-aligned.
        assert_eq!(
            RandomBits::new(DType::U32, vec![4])
                .unwrap()
                .infer_output_types(&[key_type(fixed(&[3, 1]))])
                .unwrap(),
            vec![TensorType {
                dtype: DType::U32,
                shape: fixed(&[3, 4]),
            }]
        );
        // A split keeps the key dtype, and uniform_int produces I64.
        assert_eq!(
            Split::new(vec![3])
                .infer_output_types(&[key_type(vec![])])
                .unwrap(),
            vec![key_type(fixed(&[3]))]
        );
        assert_eq!(
            UniformInt::new(vec![2], 0, 5)
                .unwrap()
                .infer_output_types(&[key_type(vec![])])
                .unwrap(),
            vec![TensorType {
                dtype: DType::I64,
                shape: fixed(&[2]),
            }]
        );
    }

    #[test]
    fn test_infer_rejects_an_operand_that_is_not_a_key() {
        let not_a_key = TensorType {
            dtype: DType::U32,
            shape: fixed(&[2]),
        };
        for op in [
            &Split::new(vec![2]) as &dyn ProgramOp<Error = RandomOpError>,
            &RandomBits::new(DType::U32, vec![2]).unwrap(),
            &UniformInt::new(vec![2], 0, 5).unwrap(),
        ] {
            assert!(
                matches!(
                    op.infer_output_types(std::slice::from_ref(&not_a_key)).unwrap_err(),
                    RandomOpError::KeyType { actual } if actual == not_a_key
                ),
                "for {}",
                op.full_name()
            );
        }
    }

    #[test]
    fn test_a_word_is_addressed_by_its_flat_position() {
        // The word at flat index i is lane i % 4 of block i / 4, whatever shape it sits in. This
        // is checked against the library's own block function rather than against our output.
        let key = key_from(11);
        let flat = words_of(
            &RandomBits::new(DType::U32, vec![8])
                .unwrap()
                .eval(std::slice::from_ref(&key))
                .unwrap()[0],
        );
        let nested = words_of(
            &RandomBits::new(DType::U32, vec![2, 4])
                .unwrap()
                .eval(std::slice::from_ref(&key))
                .unwrap()[0],
        );
        assert_eq!(flat, nested, "the shape does not move a word");

        let key_words: Vec<u32> = key.keys().unwrap().words().iter().copied().collect();
        let key_pair = [key_words[0], key_words[1]];
        for (index, &word) in flat.iter().enumerate() {
            // A draw starts at block zero, and the block counter maps to Philox's four counter
            // words little-endian.
            let block = rand_philox::philox4x32_10([(index / 4) as u32, 0, 0, 0], key_pair);
            assert_eq!(word, block[index % 4], "at flat index {index}");
        }
    }

    #[test]
    fn test_distinct_keys_give_distinct_words_and_equal_keys_agree() {
        // A [3, 1] batch of keys against a [4] payload gives each key its own row of four words.
        let op = RandomBits::new(DType::U32, vec![4]).unwrap();
        let rows_of = |keys: Tensor| words_of(&op.eval(&[keys]).unwrap()[0]);

        let distinct = rows_of(numbered_keys(&[3, 1], 100));
        let rows: Vec<&[u32]> = distinct.chunks(4).collect();
        assert_ne!(rows[0], rows[1]);
        assert_ne!(rows[1], rows[2]);

        // Equal keys produce identical rows, which is what makes an extent-1 axis mean "shared".
        let equal = rows_of(keys_of(&[3, 1], vec![5, 6, 5, 6, 5, 6]));
        let rows: Vec<&[u32]> = equal.chunks(4).collect();
        assert_eq!(rows[0], rows[1]);
        assert_eq!(rows[1], rows[2]);
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
    fn test_every_dtype_reads_the_same_bits() {
        // One stream, read at five widths. A U64 is two words low-first, a U16 or U8 is a slice of
        // one word from its low end, and a Bit is one bit, so every narrower dtype has to agree
        // with U32 on the bits they share.
        let key = key_from(55);
        let draw = |dtype: DType, count: usize| {
            RandomBits::new(dtype, vec![count])
                .unwrap()
                .eval(std::slice::from_ref(&key))
                .unwrap()
                .remove(0)
        };

        let words = words_of(&draw(DType::U32, 4));

        let Tensor::U64(wide) = draw(DType::U64, 2) else {
            panic!("expected U64")
        };
        for (i, &value) in wide.iter().enumerate() {
            let expected = (words[2 * i] as u64) | ((words[2 * i + 1] as u64) << 32);
            assert_eq!(value, expected, "U64 {i} is two words, low word first");
        }

        let Tensor::U16(narrow) = draw(DType::U16, 4) else {
            panic!("expected U16")
        };
        for (i, &value) in narrow.iter().enumerate() {
            let expected = (words[i / 2] >> (16 * (i % 2))) as u16;
            assert_eq!(value, expected, "U16 {i} is half a word");
        }

        let Tensor::U8(bytes) = draw(DType::U8, 8) else {
            panic!("expected U8")
        };
        for (i, &value) in bytes.iter().enumerate() {
            let expected = (words[i / 4] >> (8 * (i % 4))) as u8;
            assert_eq!(value, expected, "U8 {i} is a byte, low byte first");
        }

        let Tensor::Bit(bits) = draw(DType::Bit, 64) else {
            panic!("expected Bit")
        };
        for (i, &value) in bits.iter().enumerate() {
            let expected = ((words[i / 32] >> (i % 32)) & 1) as u8;
            assert_eq!(value, expected, "bit {i} is LSB-first within a word");
            assert!(value <= 1, "a Bit holds 0 or 1");
        }
    }

    #[test]
    fn test_random_bits_refuses_a_dtype_that_is_not_bits() {
        // A signed or floating draw is a distribution, not raw bits, so it is not this op's job.
        for dtype in [DType::I32, DType::I64, DType::F64, DType::C64] {
            let Err(error) = RandomBits::new(dtype, vec![2]) else {
                panic!("{dtype} is not raw bits")
            };
            assert!(matches!(
                error,
                RandomOpError::NotBits { dtype: reported } if reported == dtype
            ));
        }
    }

    #[test]
    fn test_the_dtype_reaches_the_output_type() {
        let key_type = TensorType {
            dtype: DType::Key(PrngKind::Philox4x32),
            shape: vec![],
        };
        for dtype in [DType::Bit, DType::U8, DType::U16, DType::U32, DType::U64] {
            assert_eq!(
                RandomBits::new(dtype, vec![3])
                    .unwrap()
                    .infer_output_types(std::slice::from_ref(&key_type))
                    .unwrap(),
                vec![TensorType {
                    dtype,
                    shape: fixed(&[3]),
                }]
            );
        }
    }

    #[test]
    fn test_the_batch_and_the_payload_shape_broadcast() {
        // The key batch and the payload shape broadcast against each other, so the two may share
        // axes. A [2] batch against a payload of [2] gives two outputs, not four.
        let keys = keys_of(&[2], vec![10, 11, 20, 21]);
        let draw = RandomBits::new(DType::U32, vec![2]).unwrap();
        assert_eq!(
            draw.infer_output_types(&[keys.tensor_type()]).unwrap()[0].shape,
            fixed(&[2])
        );
        let drawn = draw.eval(std::slice::from_ref(&keys)).unwrap();
        assert_eq!(drawn[0].shape(), &[2], "eval agrees with inference");

        // On the diagonal, output i reads key i at payload index i, so it is not key i's first
        // value. `test_a_draw_follows_the_documented_semantics` pins the rule itself; this only
        // records that the two indices move together here.
        let first_values = words_of(
            &RandomBits::new(DType::U32, vec![1])
                .unwrap()
                .eval(&[keys_of(&[2], vec![10, 11, 20, 21])])
                .unwrap()[0],
        );
        let diagonal = words_of(&drawn[0].clone());
        assert_eq!(diagonal[0], first_values[0], "output 0 is payload index 0");
        assert_ne!(diagonal[1], first_values[1], "output 1 is payload index 1");

        // Every op agrees on the layout.
        for op in [
            &Split::new(vec![2]) as &dyn ProgramOp<Error = RandomOpError>,
            &RandomBits::new(DType::Bit, vec![2]).unwrap(),
            &UniformInt::new(vec![2], 0, 7).unwrap(),
        ] {
            let inferred = op
                .infer_output_types(&[keys.tensor_type()])
                .unwrap()
                .remove(0);
            let evaluated = op.eval(std::slice::from_ref(&keys)).unwrap().remove(0);
            assert!(
                evaluated.matches(&inferred),
                "{} inferred {inferred} and produced {}",
                op.full_name(),
                evaluated.tensor_type()
            );
        }
    }

    #[test]
    fn test_a_shared_key_axis_gives_every_output_the_same_value() {
        // An extent-one batch axis means the key is shared along it, so a [1] key against a [4]
        // payload draws four values from one stream, and a [4] batch against a [1] payload draws
        // one value from each of four streams.
        let shared = words_of(
            &RandomBits::new(DType::U32, vec![4])
                .unwrap()
                .eval(&[keys_of(&[1], vec![3, 4])])
                .unwrap()[0],
        );
        assert_eq!(shared.len(), 4);
        assert_eq!(
            shared,
            words_of(
                &RandomBits::new(DType::U32, vec![4])
                    .unwrap()
                    .eval(&[keys_of(&[], vec![3, 4])])
                    .unwrap()[0]
            ),
            "a batch of one is the same as no batch"
        );

        let per_key = words_of(
            &RandomBits::new(DType::U32, vec![1])
                .unwrap()
                .eval(&[numbered_keys(&[4], 50)])
                .unwrap()[0],
        );
        assert_eq!(per_key.len(), 4);
        assert_eq!(
            per_key.iter().collect::<hashbrown::HashSet<_>>().len(),
            4,
            "four distinct keys give four distinct values"
        );
    }

    /// Project a flat output index onto `shape`'s axes, as the module documentation defines it.
    ///
    /// Written out independently of [`Layout`], so that the two agreeing is evidence rather than a
    /// restatement.
    fn project(out: &[usize], position: usize, shape: &[usize]) -> usize {
        let offset = out.len() - shape.len();
        let mut strides = vec![0usize; out.len()];
        let mut stride = 1;
        for axis in (0..shape.len()).rev() {
            strides[offset + axis] = if shape[axis] == 1 { 0 } else { stride };
            stride *= shape[axis];
        }
        let mut remaining = position;
        let mut index = 0;
        for axis in (0..out.len()).rev() {
            index += (remaining % out[axis]) * strides[axis];
            remaining /= out[axis];
        }
        index
    }

    #[test]
    fn test_a_draw_follows_the_documented_semantics() {
        // The module documents three properties, and this checks all of them against a reference
        // written from the prose: the output shape, the two projections, and where the payload
        // index sits in the counter. The diagonal cases are the ones a per-key draw gets wrong.
        let kind = PrngKind::Philox4x32;
        for (batch, payload) in [
            (vec![], vec![8]),
            (vec![1], vec![4]),
            (vec![3, 1], vec![4]),
            (vec![2, 1], vec![2]),
            (vec![4], vec![4]),
            (vec![6], vec![6]),
        ] {
            let count = batch.iter().product::<usize>().max(1);
            let words: Vec<u32> = (0..(2 * count) as u32).map(|w| w + 100).collect();
            let keys = Tensor::Key(
                KeyTensor::new(kind, key_words_array(&batch, kind, words.clone())).unwrap(),
            );
            let op = RandomBits::new(DType::U32, payload.clone()).unwrap();

            // Property 1: the output shape is the broadcast of the two.
            let out = broadcast_shape(&batch, &payload).unwrap();
            let drawn = words_of(&op.eval(std::slice::from_ref(&keys)).unwrap()[0]);
            assert_eq!(drawn.len(), out.iter().product::<usize>());

            for (position, &word) in drawn.iter().enumerate() {
                // Property 2: the value depends on exactly these two indices.
                let key_index = project(&out, position, &batch);
                let payload_index = project(&out, position, &payload);
                // Property 3: the key index picks the key, and the payload index picks the
                // counter position.
                let key = [words[key_index * 2], words[key_index * 2 + 1]];
                let counter = [(payload_index / 4) as u32, 0, 0, 0];
                assert_eq!(
                    word,
                    rand_philox::philox4x32_10(counter, key)[payload_index % 4],
                    "B={batch:?} S={payload:?} at output {position}"
                );
            }
        }
    }

    #[test]
    fn test_a_bounded_key_batch_flows_through() {
        // A key operand's axis may be bounded, since its true size arrives with the operand. The
        // result carries the bound, and evaluation reads the size the keys actually have. A
        // payload axis is never bounded, because an op holds its payload shape as sizes.
        let shots = Dim::Bounded { max: 8 };
        let op = RandomBits::new(DType::U32, vec![4]).unwrap();
        let operand = TensorType {
            dtype: DType::Key(PrngKind::Philox4x32),
            shape: vec![shots, Dim::Fixed(1)],
        };
        assert_eq!(
            op.infer_output_types(std::slice::from_ref(&operand))
                .unwrap(),
            vec![TensorType {
                dtype: DType::U32,
                shape: vec![shots, Dim::Fixed(4)],
            }]
        );

        let drawn = op.eval(&[numbered_keys(&[3, 1], 0)]).unwrap();
        assert_eq!(
            drawn[0].shape(),
            &[3, 4],
            "three keys within a bound of eight"
        );
        assert!(drawn[0].matches(&op.infer_output_types(&[operand]).unwrap()[0]));
    }

    #[test]
    fn test_a_key_is_single_use() {
        // A value is a function of its key and its position, so two ops drawing from one key
        // read the same blocks.
        let key = key_from(7);
        let children: Vec<u32> = Split::new(vec![3])
            .eval(std::slice::from_ref(&key))
            .unwrap()[0]
            .keys()
            .unwrap()
            .words()
            .iter()
            .copied()
            .collect();
        let drawn = words_of(
            &RandomBits::new(DType::U32, vec![6])
                .unwrap()
                .eval(std::slice::from_ref(&key))
                .unwrap()[0],
        );
        assert_eq!(
            children, drawn,
            "a split and a draw from one key read the same words"
        );

        // Each op addresses from block zero.
        let key_pair = {
            let words: Vec<u32> = key.keys().unwrap().words().iter().copied().collect();
            [words[0], words[1]]
        };
        let mut stream = Philox::new(key_pair, 0);
        let mut first = [0u32; 4];
        stream.fill_u32(&mut first);
        assert_eq!(&drawn[..4], &first, "a draw starts at block zero");

        // A derived key drives a stream independent of the parent's and of its siblings'.
        let derived = Split::new(vec![2])
            .eval(std::slice::from_ref(&key))
            .unwrap();
        let from_child = |index: usize| {
            let kind = PrngKind::Philox4x32;
            let words: Vec<u32> = derived[0].keys().unwrap().words().iter().copied().collect();
            let one = Tensor::Key(
                KeyTensor::new(
                    kind,
                    key_words_array(&[], kind, words[index * 2..index * 2 + 2].to_vec()),
                )
                .unwrap(),
            );
            words_of(
                &RandomBits::new(DType::U32, vec![4])
                    .unwrap()
                    .eval(&[one])
                    .unwrap()[0],
            )
        };
        assert_ne!(from_child(0), from_child(1));
        assert_ne!(from_child(0), drawn[..4].to_vec());
    }

    #[test]
    fn test_a_counter_too_narrow_for_the_payload_is_refused_at_inference() {
        // A payload shape is fixed when the program is built, and the key's batch shape never
        // enters the counter, so a payload a key cannot afford is refused during inference.
        let kind = PrngKind::Philox4x32;

        // Philox's counter affords more positions than `usize` holds, so no payload exhausts it.
        assert_eq!(
            substreams(kind, UNIFORM_INT_STRIDE),
            None,
            "2^128 / 2^12 exceeds usize"
        );
        assert_eq!(kind.blocks(), None, "2^128 blocks exceeds usize");

        // So the largest payload a shape can express is admitted.
        let operand = TensorType {
            dtype: DType::Key(kind),
            shape: vec![],
        };
        let huge = vec![usize::MAX];
        for op in [
            &UniformInt::new(huge.clone(), 0, 5).unwrap() as &dyn ProgramOp<Error = RandomOpError>,
            &RandomBits::new(DType::U64, huge.clone()).unwrap(),
            &Split::new(huge),
        ] {
            assert!(
                op.infer_output_types(std::slice::from_ref(&operand))
                    .is_ok(),
                "{} refused a payload Philox affords",
                op.full_name()
            );
        }
    }

    #[test]
    fn test_a_block_holds_every_value_that_fits_in_it() {
        // The capacity count is what a draw actually consumes, so it has to agree with the
        // packing `eval` uses. Counting a word per value instead would charge a Bit payload 32
        // blocks where it takes one, which no Philox payload is large enough to expose.
        let kind = PrngKind::Philox4x32;
        let block_bits = 32 * kind.block_words();
        for dtype in [DType::Bit, DType::U8, DType::U16, DType::U32, DType::U64] {
            let bits = bits_per_value(dtype).unwrap();
            let per_block = block_bits / bits;

            // A payload filling one block exactly takes one, and one value more takes two.
            assert_eq!(
                blocks_needed(kind, &[per_block], bits),
                1,
                "{per_block} {dtype} values are one block"
            );
            assert_eq!(
                blocks_needed(kind, &[per_block + 1], bits),
                2,
                "{dtype} past a block boundary"
            );
        }

        // A derived key is whole words, so a Philox key is two of the four in a block.
        assert_eq!(blocks_needed(kind, &[2], 32 * kind.key_words()), 1);
        assert_eq!(blocks_needed(kind, &[3], 32 * kind.key_words()), 2);

        // The product saturates rather than wrapping, so a huge payload stays huge. Two
        // `usize::MAX` axes still fit in a `u128`, so it takes three to reach the ceiling.
        assert_eq!(
            blocks_needed(kind, &[usize::MAX; 3], 1),
            u128::MAX.div_ceil(block_bits as u128)
        );
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

        // An output's substream starts a stride of blocks past the previous one's, for whatever
        // stride its sampler chose.
        for stride in [1usize, 8, UNIFORM_INT_STRIDE] {
            for output in [0usize, 1, 2] {
                assert_eq!(
                    block_of(&mut Stream::at_substream(kind, &key, output, stride)),
                    block_of(&mut Stream::at_block(kind, &key, output * stride)),
                    "substream {output} of stride {stride}"
                );
            }
        }
    }

    #[test]
    fn test_a_split_key_drives_an_independent_stream() {
        let split = Split::new(vec![3]).eval(&[key_from(99)]).unwrap();
        let keys = split[0].keys().unwrap();
        assert_eq!(keys.shape(), &[3]);

        // Each derived key is usable as a key, and the three streams differ.
        let source = flat_key_words(keys);
        let draws: Vec<Vec<u32>> = source
            .chunks(keys.kind().key_words())
            .map(|key_words| {
                let one = Tensor::Key(
                    KeyTensor::new(
                        keys.kind(),
                        key_words_array(&[], keys.kind(), key_words.to_vec()),
                    )
                    .unwrap(),
                );
                words_of(
                    &RandomBits::new(DType::U32, vec![4])
                        .unwrap()
                        .eval(&[one])
                        .unwrap()[0],
                )
            })
            .collect();
        assert_ne!(draws[0], draws[1]);
        assert_ne!(draws[1], draws[2]);
        assert_ne!(draws[0], draws[2]);
    }

    #[test]
    fn test_uniform_int_honours_its_range() {
        // The bounds are inclusive, and a single-value range is constant.
        let key = key_from(3);
        let drawn = |low, high| {
            ints_of(
                &UniformInt::new(vec![64], low, high)
                    .unwrap()
                    .eval(std::slice::from_ref(&key))
                    .unwrap()[0],
            )
        };
        let samples = drawn(-5, 5);
        assert!(samples.iter().all(|&v| (-5..=5).contains(&v)));
        assert!(
            samples.contains(&-5) || samples.contains(&5),
            "the bounds are reachable"
        );
        assert_eq!(drawn(7, 7), vec![7; 64]);

        // An empty range is refused. `unwrap_err` would need UniformInt to be Debug, which no op
        // in the catalogue is.
        let Err(error) = UniformInt::new(vec![2], 5, 4) else {
            panic!("an empty range is refused")
        };
        assert!(matches!(
            error,
            RandomOpError::EmptyRange { low: 5, high: 4 }
        ));
    }

    #[test]
    fn test_a_uniform_sample_reads_its_own_substream() {
        // Each sample comes from the substream its own position opens. Were positions adjacent
        // blocks instead, a sample would read words belonging to its neighbour, so this compares
        // the op's output against substreams opened independently of it.
        let key = key_from(41);
        let key_words: Vec<u32> = key.keys().unwrap().words().iter().copied().collect();
        let key_pair = [key_words[0], key_words[1]];

        let samples = ints_of(
            &UniformInt::new(vec![6], 0, 2)
                .unwrap()
                .eval(std::slice::from_ref(&key))
                .unwrap()[0],
        );
        let range = Uniform::new_inclusive(0i64, 2).unwrap();
        for (position, &sample) in samples.iter().enumerate() {
            // Spelled out rather than taken from `UNIFORM_INT_STRIDE`, so that a change to the
            // stride has to be made here too rather than passing silently.
            let counter = (position as u128) * (1 << 8);
            assert_eq!(UNIFORM_INT_STRIDE, 1 << 8, "the stride this test assumes");
            let mut substream = Philox::new(key_pair, counter);
            assert_eq!(
                sample,
                range.sample(&mut substream),
                "at position {position}"
            );
        }
        assert!(
            samples.windows(2).any(|w| w[0] != w[1]),
            "every position reading one substream would give one repeated value"
        );
    }

    #[test]
    fn test_entropy_gives_fresh_keys_of_the_payload_shape() {
        let op = Entropy::new(PrngKind::Philox4x32, vec![2, 3]);
        assert_eq!(
            op.infer_output_types(&[]).unwrap(),
            vec![TensorType {
                dtype: DType::Key(PrngKind::Philox4x32),
                shape: fixed(&[2, 3]),
            }]
        );

        let first = op.eval(&[]).unwrap().remove(0);
        assert_eq!(first.shape(), &[2, 3]);
        assert_eq!(first.dtype(), DType::Key(PrngKind::Philox4x32));
        // Two evaluations differ, so no particular value is asserted.
        assert_ne!(first, op.eval(&[]).unwrap().remove(0));

        // The keys need no conversion to be drawn from, and each opens its own stream.
        let draws = words_of(
            &RandomBits::new(DType::U32, vec![1])
                .unwrap()
                .eval(&[first])
                .unwrap()[0],
        );
        let distinct: hashbrown::HashSet<u32> = draws.iter().copied().collect();
        assert_eq!(distinct.len(), 6, "each of the six keys differs");
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
    fn test_random_bits_matches_its_recorded_output() {
        // Every value an op produces is recorded here, so a change to the composition of key,
        // counter and dtype is caught even where it keeps the documented semantics self
        // consistent. The all-zero key anchors the numbers to the published vectors: the first
        // four words below are `PHILOX4X32_10_KAT[0]`, which is what makes these more than a
        // record of whatever the code did when they were written.
        assert_eq!(
            words_of(
                &RandomBits::new(DType::U32, vec![8])
                    .unwrap()
                    .eval(&[keys_of(&[], vec![0, 0])])
                    .unwrap()[0]
            ),
            [
                0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8, 0xf8e4cca4, 0x5cb200db, 0xb1a574eb,
                0x097eff67,
            ]
        );
        assert_eq!(
            &words_of(
                &RandomBits::new(DType::U32, vec![8])
                    .unwrap()
                    .eval(&[keys_of(&[], vec![0, 0])])
                    .unwrap()[0]
            )[..4],
            PHILOX4X32_10_KAT[0].2,
            "the first block of the zero key is the published vector"
        );

        let key = golden_key();
        assert_eq!(
            words_of(
                &RandomBits::new(DType::U32, vec![8])
                    .unwrap()
                    .eval(std::slice::from_ref(&key))
                    .unwrap()[0]
            ),
            [
                0xd8830217, 0xd18bb386, 0x0cb1c71d, 0xbdd96e0b, 0xeb24d1e9, 0x57890589, 0x4c52e006,
                0x94d7f26b,
            ]
        );

        // The narrower dtypes read those same words, so each recorded value is also a check on
        // the packing: a U64 is two words low first, a U8 is a byte of one word from its low end,
        // and a Bit is one bit.
        let Tensor::U64(wide) = &RandomBits::new(DType::U64, vec![4])
            .unwrap()
            .eval(std::slice::from_ref(&key))
            .unwrap()[0]
        else {
            panic!("expected U64")
        };
        assert_eq!(
            wide.iter().copied().collect::<Vec<u64>>(),
            [
                0xd18bb386_d8830217,
                0xbdd96e0b_0cb1c71d,
                0x57890589_eb24d1e9,
                0x94d7f26b_4c52e006,
            ]
        );

        let Tensor::U8(bytes) = &RandomBits::new(DType::U8, vec![8])
            .unwrap()
            .eval(std::slice::from_ref(&key))
            .unwrap()[0]
        else {
            panic!("expected U8")
        };
        assert_eq!(
            bytes.iter().copied().collect::<Vec<u8>>(),
            [0x17, 0x02, 0x83, 0xd8, 0x86, 0xb3, 0x8b, 0xd1]
        );

        let Tensor::Bit(bits) = &RandomBits::new(DType::Bit, vec![16])
            .unwrap()
            .eval(std::slice::from_ref(&key))
            .unwrap()[0]
        else {
            panic!("expected Bit")
        };
        assert_eq!(
            bits.iter().copied().collect::<Vec<u8>>(),
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "the low 16 bits of 0xd8830217, least significant first"
        );

        // A batch of keys against a payload, where each output reads its own key.
        assert_eq!(
            words_of(
                &RandomBits::new(DType::U32, vec![2])
                    .unwrap()
                    .eval(&[golden_batch()])
                    .unwrap()[0]
            ),
            [
                0x0598de3a, 0x98d2802e, 0x1955b60c, 0x46814a70, 0xa82aad38, 0xf48e6bfb,
            ],
            "output (i, j) is word j of key i's stream"
        );
    }

    #[test]
    fn test_split_matches_its_recorded_output() {
        let derived = Split::new(vec![3]).eval(&[golden_key()]).unwrap().remove(0);
        assert_eq!(
            derived
                .keys()
                .unwrap()
                .words()
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
                .words()
                .iter()
                .copied()
                .collect::<Vec<u32>>(),
            [
                0x0598de3a, 0x98d2802e, 0x270f8f9e, 0xeab709d3, 0x1955b60c, 0x46814a70, 0x682e0b3d,
                0x30193e3c, 0xa82aad38, 0xf48e6bfb, 0xab96c3d4, 0xc9bd190e,
            ]
        );
    }

    #[test]
    fn test_uniform_int_matches_its_recorded_output() {
        // These pin the substream addressing and the distribution together, so a change to the
        // stride, to the counter layout, or to how `Uniform` is driven shows up here.
        let key = golden_key();
        assert_eq!(
            ints_of(
                &UniformInt::new(vec![8], 0, 99)
                    .unwrap()
                    .eval(std::slice::from_ref(&key))
                    .unwrap()[0]
            ),
            [81, 6, 66, 28, 83, 21, 51, 59]
        );
        assert_eq!(
            ints_of(
                &UniformInt::new(vec![8], -1000, 1000)
                    .unwrap()
                    .eval(std::slice::from_ref(&key))
                    .unwrap()[0]
            ),
            [637, -873, 336, -438, 662, -565, 28, 195]
        );
        assert_eq!(
            ints_of(
                &UniformInt::new(vec![2], 0, 7)
                    .unwrap()
                    .eval(&[golden_batch()])
                    .unwrap()[0]
            ),
            [4, 6, 2, 0, 7, 0]
        );
    }

    #[test]
    fn test_a_key_reaches_an_op_from_every_origin() {
        // A key needs no op of its own to construct: a parameter declares one, a constant holds
        // one, and entropy mints one. All three feed the drawing ops directly.
        use crate::ops::Constant;
        use crate::program::ProgramFunction;

        let kind = PrngKind::Philox4x32;
        let key = key_from(1234);
        let draw = RandomBits::new(DType::U32, vec![4]).unwrap();

        // A caller supplies the key as an argument.
        let mut function = ProgramFunction::default();
        let parameter = function.add_parameter(TensorType {
            dtype: DType::Key(kind),
            shape: vec![],
        });
        let drawn = function.add_op(draw.clone(), &[parameter]).unwrap();
        function.add_result(drawn[0]).unwrap();
        let from_parameter = function.eval(std::slice::from_ref(&key)).unwrap();

        // The program holds the key itself.
        let mut function = ProgramFunction::default();
        let constant = function.add_op(Constant::new(key.clone()), &[]).unwrap();
        let drawn = function.add_op(draw.clone(), &[constant[0]]).unwrap();
        function.add_result(drawn[0]).unwrap();
        let from_constant = function.eval(&[]).unwrap();

        // The same key by either route gives the same variates.
        assert_eq!(from_parameter, from_constant);
        assert_eq!(from_parameter[0].dtype(), DType::U32);

        // Entropy mints its own, so it differs and is not asserted against a value.
        let mut function = ProgramFunction::default();
        let entropy = function.add_op(Entropy::new(kind, vec![]), &[]).unwrap();
        let drawn = function.add_op(draw, &[entropy[0]]).unwrap();
        function.add_result(drawn[0]).unwrap();
        assert_eq!(function.eval(&[]).unwrap()[0].dtype(), DType::U32);
    }
}
