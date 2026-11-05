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

//! Packed storage for an array of bits.

use std::iter::repeat_n;

use ndarray::{ArcArrayD, ArrayD, Axis, IxDyn};

use super::TensorError;
use super::broadcast::broadcast_shape;

/// Return the shape of the packed buffer of a tensor of logical shape `shape`.
fn packed_shape(shape: &[usize]) -> Vec<usize> {
    let mut packed = shape.to_vec();
    if let Some(axis) = packed.last_mut() {
        *axis = axis.div_ceil(8);
    }
    packed
}

/// Return the number of bits in one row of a tensor of logical shape `shape`.
fn num_bits(shape: &[usize]) -> usize {
    // a 0-d tensor has 1 member, the scalar itself
    shape.last().copied().unwrap_or(1)
}

/// Return the number of whole bytes one row of a tensor of logical shape `shape` occupies.
fn bytes_per_row(shape: &[usize]) -> usize {
    num_bits(shape).div_ceil(8)
}

/// Return the number of rows a tensor of logical shape `shape` holds.
fn num_rows(shape: &[usize]) -> usize {
    shape
        .split_last()
        .map_or(1, |(_, leading)| leading.iter().product())
}

/// Return the mask of the bits a row of `num_bits` bits uses in its final byte, or `None` if that
/// byte has no padding to clear.
fn final_byte_mask(num_bits: usize) -> Option<u8> {
    match num_bits % 8 {
        0 => None,
        used => Some((1u8 << used) - 1),
    }
}

/// A dense array of bits.
///
/// This object presents an array of bits with a shape, storing them packed eight bits to a
/// byte along the last axis.
///
/// The storage format is the following. A "row" is the last axis of the logical shape, of
/// [`num_bits`](Self::num_bits) bits, and occupies `ceil(num_bits / 8)` bytes. Bit `i` of a row is
/// bit `i % 8` of byte `i / 8`, so the order within a byte is least significant first. The bits of
/// a row's final byte at or above `num_bits` are padding, and are zero. Rows follow one another in
/// row-major order of the leading axes, each starting on a byte boundary. A zero-dimensional
/// tensor is one row of one bit.
#[derive(Clone, Debug, PartialEq)]
pub struct BitTensor {
    data: ArcArrayD<u8>,
    /// Store the entire logical shape so that `shape()` is not an allocation.
    shape: Vec<usize>,
}

impl BitTensor {
    /// Return a tensor of logical shape `shape` over `bits`, taken in row-major order.
    ///
    /// Returns [`TensorError::BitCount`] if `bits` does not yield one bit per element of `shape`.
    pub fn from_bools<I>(bits: I, shape: &[usize]) -> Result<Self, TensorError>
    where
        I: IntoIterator<Item = bool>,
        I::IntoIter: ExactSizeIterator,
    {
        let bits = bits.into_iter();
        let expected = shape.iter().product::<usize>();
        if bits.len() != expected {
            return Err(TensorError::BitCount {
                shape: shape.to_vec(),
                expected,
                actual: bits.len(),
            });
        }
        Ok(pack(bits, shape.to_vec()))
    }

    /// Return this tensor's bits in row-major order, one `bool` per element.
    pub fn to_bool_vec(&self) -> Vec<bool> {
        let num_bits = self.num_bits();
        self.rows()
            .flat_map(|row| (0..num_bits).map(|bit| row[bit / 8] >> (bit % 8) & 1 == 1))
            .collect()
    }

    /// Return the packed bits, in the format this type documents.
    fn as_bytes(&self) -> &[u8] {
        self.data
            .as_slice()
            .expect("the packed buffer is contiguous")
    }

    /// Return the logical shape of this tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the number of bits per row, i.e. the size of the last logical axis.
    pub fn num_bits(&self) -> usize {
        num_bits(&self.shape)
    }

    /// Return the number of whole bytes one row occupies, i.e. `num_bits` divided by eight and
    /// rounded up.
    pub fn bytes_per_row(&self) -> usize {
        bytes_per_row(&self.shape)
    }

    /// Broadcast this tensor to `shape`, right aligning the shapes.
    ///
    /// Returns [`TensorError::ShapeMismatch`] if `shape` cannot be reached that way.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self, TensorError> {
        if self.shape == shape {
            return Ok(self.clone());
        }
        let mismatch = || TensorError::ShapeMismatch {
            lhs: self.shape.clone(),
            rhs: shape.to_vec(),
        };
        // The row is stretched first, because the bits within one are not addressable as bytes.
        let stretched = self.stretch_row(num_bits(shape)).ok_or_else(mismatch)?;
        let data = stretched
            .data
            .broadcast(IxDyn(&packed_shape(shape)))
            .ok_or_else(mismatch)?;
        Ok(Self {
            data: data.to_owned().into_shared(),
            shape: shape.to_vec(),
        })
    }

    /// Return the element-wise `and` of two tensors.
    ///
    /// Returns [`TensorError::ShapeMismatch`] if the shapes are not broadcast-compatible.
    pub fn and(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.zip_bytes(rhs, |a, b| a & b)
    }

    /// Return the element-wise `or` of two tensors.
    ///
    /// Returns [`TensorError::ShapeMismatch`] if the shapes are not broadcast-compatible.
    pub fn or(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.zip_bytes(rhs, |a, b| a | b)
    }

    /// Return the element-wise `xor` of two tensors.
    ///
    /// Returns [`TensorError::ShapeMismatch`] if the shapes are not broadcast-compatible.
    pub fn xor(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.zip_bytes(rhs, |a, b| a ^ b)
    }

    /// Return the element-wise complement of this tensor.
    pub fn not(&self) -> Self {
        let mut bytes: Vec<u8> = self.as_bytes().iter().map(|byte| !byte).collect();
        if let Some(mask) = final_byte_mask(self.num_bits()) {
            let bytes_per_row = self.bytes_per_row();
            for row in 0..num_rows(&self.shape) {
                bytes[(row + 1) * bytes_per_row - 1] &= mask;
            }
        }
        Self::new(bytes, self.shape.clone())
    }

    /// Return the XOR-reduction of this tensor along `axis`, with that axis removed.
    ///
    /// # Panics
    ///
    /// Panics if `axis` is out of bounds.
    pub fn parity(&self, axis: usize) -> Self {
        assert!(
            axis < self.shape.len(),
            "axis {axis} is out of bounds for a tensor of {} dimension(s)",
            self.shape.len()
        );
        let mut shape = self.shape.clone();
        shape.remove(axis);
        if axis + 1 == self.shape.len() {
            // Reducing the row itself, so each row contributes one bit of a repacked result.
            let parities = self.popcount().into_iter().map(|count| count % 2 == 1);
            return pack(parities, shape);
        }
        // A leading axis indexes whole rows, so the reduction is a byte-wise XOR.
        Self {
            data: self
                .data
                .fold_axis(Axis(axis), 0u8, |&total, &byte| total ^ byte)
                .into_shared(),
            shape,
        }
    }

    /// Return the number of set bits in each row, in row-major order of the leading axes.
    fn popcount(&self) -> Vec<usize> {
        self.rows()
            .map(|row| row.iter().map(|byte| byte.count_ones() as usize).sum())
            .collect()
    }

    /// Return a tensor of logical shape `shape` over `bytes`, which must be packed for it.
    fn new(bytes: Vec<u8>, shape: Vec<usize>) -> Self {
        let packed = packed_shape(&shape);
        let this = Self {
            data: ArrayD::from_shape_vec(IxDyn(&packed), bytes)
                .expect("the bytes are packed for this shape")
                .into_shared(),
            shape,
        };
        debug_assert!(
            this.padding_is_zero(),
            "the padding bits of a packed row are zero"
        );
        this
    }

    /// Return an iterator over the rows of the packed buffer.
    fn rows(&self) -> impl Iterator<Item = &[u8]> {
        let bytes_per_row = self.bytes_per_row();
        let bytes = self.as_bytes();
        (0..num_rows(&self.shape))
            .map(move |row| &bytes[row * bytes_per_row..(row + 1) * bytes_per_row])
    }

    /// Return whether the padding bits of every row are zero.
    fn padding_is_zero(&self) -> bool {
        let Some(mask) = final_byte_mask(self.num_bits()) else {
            return true;
        };
        self.rows()
            .all(|row| row.last().is_none_or(|byte| byte & !mask == 0))
    }

    /// Return this tensor with each of its rows stretched to `num_bits` bits, or `None` if a row
    /// does not hold the single bit that one is stretched from.
    fn stretch_row(&self, num_bits: usize) -> Option<Self> {
        if !self.shape.is_empty() && self.num_bits() == num_bits {
            return Some(self.clone());
        }
        if self.num_bits() != 1 {
            return None;
        }
        let mut shape = self.shape.clone();
        match shape.last_mut() {
            Some(axis) => *axis = num_bits,
            None => shape.push(num_bits),
        }
        let bytes_per_row = num_bits.div_ceil(8);
        let mask = final_byte_mask(num_bits);
        let mut bytes = Vec::with_capacity(num_rows(&shape) * bytes_per_row);
        for row in self.rows() {
            let fill = if row[0] & 1 == 1 { u8::MAX } else { 0 };
            bytes.extend(repeat_n(fill, bytes_per_row));
            if let Some(mask) = mask {
                *bytes.last_mut().expect("a padded row holds a byte") &= mask;
            }
        }
        Some(Self::new(bytes, shape))
    }

    /// Return the result of applying `op` to the bytes of two tensors, broadcast against one
    /// another NumPy-style.
    fn zip_bytes(&self, rhs: &Self, op: impl Fn(u8, u8) -> u8) -> Result<Self, TensorError> {
        let shape = broadcast_shape(&self.shape, &rhs.shape)?;
        let lhs = self.broadcast_to(&shape)?;
        let rhs = rhs.broadcast_to(&shape)?;
        let bytes = lhs
            .as_bytes()
            .iter()
            .zip(rhs.as_bytes())
            .map(|(&a, &b)| op(a, b))
            .collect();
        Ok(Self::new(bytes, shape))
    }
}

/// Pack `bits`, taken in row-major order, into a tensor of logical shape `shape`.
fn pack(bits: impl IntoIterator<Item = bool>, shape: Vec<usize>) -> BitTensor {
    let (num_bits, bytes_per_row) = (num_bits(&shape), bytes_per_row(&shape));
    let mut bytes = vec![0u8; num_rows(&shape) * bytes_per_row];
    let mut bits = bits.into_iter();
    for row in 0..num_rows(&shape) {
        for bit in 0..num_bits {
            if bits.next().expect("one bit per element of the shape") {
                bytes[row * bytes_per_row + bit / 8] |= 1 << (bit % 8);
            }
        }
    }
    BitTensor::new(bytes, shape)
}

#[cfg(test)]
mod test {
    use super::*;

    /// Return a tensor of logical shape `shape` over `bits`, taken in row-major order.
    fn bits(bits: &[bool], shape: &[usize]) -> BitTensor {
        BitTensor::from_bools(bits.iter().copied(), shape).unwrap()
    }

    /// Return a tensor of one row of `num_bits` bits, set at the positions `set` names.
    ///
    /// Writing the set positions is shorter than a list of bools once a row is wide, and it is how
    /// the expected bytes of a wide row are worked out by hand.
    fn row(num_bits: usize, set: &[usize]) -> BitTensor {
        let values: Vec<bool> = (0..num_bits).map(|bit| set.contains(&bit)).collect();
        bits(&values, &[num_bits])
    }

    /// Return whether the padding bits of every row of `tensor` are zero, read through the public
    /// buffer rather than through the type's own check.
    fn padding_is_zero(tensor: &BitTensor) -> bool {
        let used = tensor.num_bits() % 8;
        if used == 0 {
            return true;
        }
        let mask = !((1u8 << used) - 1);
        tensor
            .as_bytes()
            .chunks(tensor.bytes_per_row())
            .all(|row| row[row.len() - 1] & mask == 0)
    }

    // The published byte format.

    #[test]
    fn test_a_row_packs_least_significant_bit_first() {
        assert_eq!(bits(&[true, false, true], &[3]).as_bytes(), [0b101]);
        // Bit 0 is the low bit of the byte and bit 7 is the high one.
        assert_eq!(row(8, &[0]).as_bytes(), [0b0000_0001]);
        assert_eq!(row(8, &[7]).as_bytes(), [0b1000_0000]);
        assert_eq!(row(8, &[1, 3, 6]).as_bytes(), [0b0100_1010]);
    }

    #[test]
    fn test_the_buffer_is_what_numpy_packbits_writes() {
        // The expected bytes are `numpy.packbits(rows, axis=-1, bitorder="little")`, which is the
        // interchange the format promises.
        let rows = [
            [
                true, true, true, false, false, false, false, false, false, true, true,
            ],
            [
                true, true, true, true, true, true, true, true, true, false, true,
            ],
            [
                true, false, false, true, true, false, true, true, true, false, false,
            ],
        ];
        let values: Vec<bool> = rows.iter().flatten().copied().collect();
        let tensor = bits(&values, &[3, 11]);
        assert_eq!(tensor.as_bytes(), [7, 6, 255, 5, 217, 1]);
        assert_eq!(tensor.to_bool_vec(), values);
    }

    #[test]
    fn test_a_row_is_padded_to_a_byte_boundary() {
        for (num_bits, bytes_per_row) in [(1, 1), (7, 1), (8, 1), (9, 2), (100, 13)] {
            let tensor = row(num_bits, &[]);
            assert_eq!(tensor.bytes_per_row(), bytes_per_row, "for {num_bits} bits");
            assert_eq!(
                tensor.as_bytes().len(),
                bytes_per_row,
                "for {num_bits} bits"
            );
        }
    }

    #[test]
    fn test_a_row_begins_a_byte_whatever_its_width() {
        // Bit 0 of every row is the low bit of a byte of its own, so one row never spills into
        // another's bytes.
        let tensor = bits(&[true, false, false, true, true, false], &[3, 2]);
        assert_eq!(tensor.as_bytes(), [0b01, 0b10, 0b01]);
        assert_eq!(tensor.bytes_per_row(), 1);
    }

    #[test]
    fn test_rows_follow_in_row_major_order() {
        let tensor = bits(
            &[true, false, false, false, false, true, true, true],
            &[2, 2, 2],
        );
        assert_eq!(tensor.as_bytes(), [0b01, 0b00, 0b10, 0b11]);
    }

    #[test]
    fn test_a_row_wider_than_a_byte_packs_low_byte_first() {
        assert_eq!(row(16, &[0, 8]).as_bytes(), [0b0000_0001, 0b0000_0001]);
        assert_eq!(row(12, &[11]).as_bytes(), [0, 0b0000_1000]);
    }

    // Shape and the zero-dimensional case.

    #[test]
    fn test_shape_is_logical_rather_than_packed() {
        let tensor = bits(&[false; 24], &[2, 12]);
        assert_eq!(tensor.shape(), [2, 12]);
        assert_eq!(tensor.num_bits(), 12);
        assert_eq!(tensor.as_bytes().len(), 4);
    }

    #[test]
    fn test_a_zero_dimensional_tensor_is_one_bit() {
        let tensor = bits(&[true], &[]);
        assert_eq!(tensor.shape(), [0usize; 0]);
        assert_eq!(tensor.num_bits(), 1);
        assert_eq!(tensor.as_bytes(), [1]);
        assert_eq!(tensor.to_bool_vec(), [true]);
    }

    #[test]
    fn test_a_zero_bit_row_holds_no_bytes() {
        let tensor = bits(&[], &[4, 0]);
        assert_eq!(tensor.shape(), [4, 0]);
        assert_eq!(tensor.num_bits(), 0);
        assert_eq!(tensor.bytes_per_row(), 0);
        assert!(tensor.as_bytes().is_empty());
        assert!(tensor.to_bool_vec().is_empty());
    }

    #[test]
    fn test_an_empty_leading_axis_holds_no_rows() {
        let tensor = bits(&[], &[0, 5]);
        assert_eq!(tensor.shape(), [0, 5]);
        assert_eq!(tensor.bytes_per_row(), 1);
        assert!(tensor.as_bytes().is_empty());
    }

    #[test]
    fn test_from_bools_rejects_a_wrong_count() {
        assert!(matches!(
            BitTensor::from_bools([true; 5], &[2, 3]).unwrap_err(),
            TensorError::BitCount { shape, expected: 6, actual: 5 } if shape == [2, 3]
        ));
    }

    #[test]
    fn test_to_bool_vec_is_row_major_and_skips_the_padding() {
        let values: Vec<bool> = (0..27).map(|index| index % 4 == 0).collect();
        let tensor = bits(&values, &[3, 9]);
        assert_eq!(tensor.to_bool_vec(), values);
    }

    // The element-wise kernels.

    #[test]
    fn test_and_or_and_xor_combine_the_bytes() {
        let lhs = bits(&[true, true, false, false], &[4]);
        let rhs = bits(&[true, false, true, false], &[4]);
        assert_eq!(lhs.and(&rhs).unwrap().as_bytes(), [0b0001]);
        assert_eq!(lhs.or(&rhs).unwrap().as_bytes(), [0b0111]);
        assert_eq!(lhs.xor(&rhs).unwrap().as_bytes(), [0b0110]);
    }

    #[test]
    fn test_not_complements_the_bits() {
        assert_eq!(row(8, &[1, 3, 6]).not().as_bytes(), [0b1011_0101]);
        assert_eq!(
            bits(&[true, false, true], &[3]).not().to_bool_vec(),
            [false, true, false]
        );
    }

    #[test]
    fn test_a_kernel_holds_the_padding_at_zero() {
        // `and`, `or` and `xor` cannot dirty the padding of zero-padded operands, and `not` has to
        // re-clear it, so every kernel is checked at a width that has padding.
        let lhs = row(11, &[0, 10]);
        let rhs = row(11, &[1, 10]);
        for tensor in [
            lhs.and(&rhs).unwrap(),
            lhs.or(&rhs).unwrap(),
            lhs.xor(&rhs).unwrap(),
            lhs.not(),
            lhs.parity(0),
        ] {
            assert!(padding_is_zero(&tensor), "for {:?}", tensor.as_bytes());
        }
    }

    #[test]
    fn test_a_kernel_reads_a_row_that_is_not_a_whole_number_of_bytes() {
        let lhs = row(11, &[0, 7, 8, 10]);
        let rhs = row(11, &[7, 10]);
        assert_eq!(lhs.and(&rhs).unwrap().to_bool_vec(), rhs.to_bool_vec());
        assert_eq!(lhs.or(&rhs).unwrap().to_bool_vec(), lhs.to_bool_vec());
        assert_eq!(
            lhs.xor(&rhs).unwrap().to_bool_vec(),
            row(11, &[0, 8]).to_bool_vec()
        );
        assert_eq!(
            lhs.not().to_bool_vec(),
            row(11, &[1, 2, 3, 4, 5, 6, 9]).to_bool_vec()
        );
    }

    #[test]
    fn test_not_of_a_zero_bit_row_is_empty() {
        let tensor = bits(&[], &[2, 0]).not();
        assert_eq!(tensor.shape(), [2, 0]);
        assert!(tensor.as_bytes().is_empty());
    }

    #[test]
    fn test_a_kernel_reaches_every_row() {
        let lhs = bits(&[true, false, false, true, true, true], &[3, 2]);
        let rhs = bits(&[true, true, false, false, true, true], &[3, 2]);
        assert_eq!(lhs.and(&rhs).unwrap().as_bytes(), [0b01, 0b00, 0b11]);
        assert_eq!(lhs.not().as_bytes(), [0b10, 0b01, 0b00]);
    }

    // Broadcasting.

    #[test]
    fn test_broadcast_to_its_own_shape_shares_the_buffer() {
        let tensor = bits(&[true, false, true], &[3]);
        let broadcast = tensor.broadcast_to(&[3]).unwrap();
        assert_eq!(tensor.as_bytes().as_ptr(), broadcast.as_bytes().as_ptr());
    }

    #[test]
    fn test_broadcast_to_repeats_a_row_along_a_leading_axis() {
        let tensor = bits(&[true, false, true], &[3]);
        let broadcast = tensor.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(broadcast.as_bytes(), [0b101, 0b101]);
        let tensor = bits(&[true, false, false, true], &[2, 1, 2]);
        let broadcast = tensor.broadcast_to(&[2, 3, 2]).unwrap();
        assert_eq!(broadcast.as_bytes(), [0b01, 0b01, 0b01, 0b10, 0b10, 0b10]);
    }

    #[test]
    fn test_broadcast_to_stretches_a_single_bit_across_a_row() {
        // A one-bit row is the case the byte layout cannot handle by striding, since the copies
        // land within a byte rather than at byte offsets.
        assert_eq!(
            bits(&[true], &[1]).broadcast_to(&[11]).unwrap().as_bytes(),
            [0xff, 0b0000_0111]
        );
        assert_eq!(
            bits(&[false], &[1]).broadcast_to(&[11]).unwrap().as_bytes(),
            [0, 0]
        );
        let stretched = bits(&[true, false], &[2, 1]).broadcast_to(&[2, 3]).unwrap();
        assert_eq!(stretched.as_bytes(), [0b111, 0b000]);
    }

    #[test]
    fn test_broadcast_to_grows_both_axes_at_once() {
        let broadcast = bits(&[true], &[1, 1]).broadcast_to(&[2, 9]).unwrap();
        assert_eq!(broadcast.as_bytes(), [0xff, 0b1, 0xff, 0b1]);
        assert_eq!(broadcast.to_bool_vec(), [true; 18]);
    }

    #[test]
    fn test_broadcast_to_adds_an_axis_to_a_zero_dimensional_tensor() {
        let broadcast = bits(&[true], &[]).broadcast_to(&[2, 3]).unwrap();
        assert_eq!(broadcast.as_bytes(), [0b111, 0b111]);
    }

    #[test]
    fn test_broadcast_to_a_shape_it_cannot_reach_reports_both() {
        let tensor = bits(&[true, false, true], &[3]);
        for shape in [vec![4], vec![1], vec![3, 2], vec![]] {
            assert!(
                matches!(
                    tensor.broadcast_to(&shape).unwrap_err(),
                    TensorError::ShapeMismatch { lhs, rhs } if lhs == [3] && rhs == shape
                ),
                "for target {shape:?}"
            );
        }
    }

    #[test]
    fn test_a_kernel_broadcasts_its_operands_either_way_round() {
        let wide = bits(&[true, false, true], &[3]);
        let single = bits(&[true], &[1]);
        assert_eq!(wide.and(&single).unwrap(), wide);
        assert_eq!(single.and(&wide).unwrap(), wide);
        let column = bits(&[true, false], &[2, 1]);
        assert_eq!(
            wide.and(&column).unwrap().to_bool_vec(),
            [true, false, true, false, false, false]
        );
    }

    #[test]
    fn test_a_kernel_takes_a_zero_dimensional_operand() {
        let single = bits(&[true], &[]);
        assert_eq!(single.and(&single).unwrap(), single);
        assert_eq!(single.not().to_bool_vec(), [false]);
        let wide = bits(&[true, false, true], &[3]);
        assert_eq!(single.and(&wide).unwrap(), wide);
        assert_eq!(wide.or(&single).unwrap().to_bool_vec(), [true; 3]);
    }

    #[test]
    fn test_a_kernel_reports_shapes_that_do_not_broadcast() {
        let lhs = bits(&[false; 3], &[3]);
        let rhs = bits(&[false; 2], &[2]);
        assert!(matches!(
            lhs.xor(&rhs).unwrap_err(),
            TensorError::ShapeMismatch { lhs, rhs } if lhs == [3] && rhs == [2]
        ));
    }

    // Parity and popcount.

    #[test]
    fn test_parity_of_a_leading_axis_xors_whole_rows() {
        // [[1,0,1],[0,1,1],[0,0,0]] along axis 0 is [1, 1, 0].
        let values = [
            true, false, true, //
            false, true, true, //
            false, false, false,
        ];
        let parity = bits(&values, &[3, 3]).parity(0);
        assert_eq!(parity.shape(), [3]);
        assert_eq!(parity.to_bool_vec(), [true, true, false]);
    }

    #[test]
    fn test_parity_of_the_last_axis_repacks_one_bit_per_row() {
        let values = [
            true, false, true, //
            false, true, true, //
            true, true, true,
        ];
        // Two set bits per row is even and three is odd, so only the last row's parity is set.
        let parity = bits(&values, &[3, 3]).parity(1);
        assert_eq!(parity.shape(), [3]);
        assert_eq!(parity.as_bytes(), [0b100]);
    }

    #[test]
    fn test_parity_of_a_row_wider_than_a_byte() {
        // 3 set bits across two bytes is odd, and 4 is even.
        assert_eq!(row(11, &[0, 8, 10]).parity(0).to_bool_vec(), [true]);
        assert_eq!(row(11, &[0, 7, 8, 10]).parity(0).to_bool_vec(), [false]);
    }

    #[test]
    fn test_parity_of_the_only_axis_leaves_a_zero_dimensional_tensor() {
        let parity = bits(&[true, true, true], &[3]).parity(0);
        assert_eq!(parity.shape(), [0usize; 0]);
        assert_eq!(parity.as_bytes(), [1]);
        assert_eq!(parity.to_bool_vec(), [true]);
    }

    #[test]
    fn test_parity_of_a_middle_axis_keeps_the_others() {
        let values: Vec<bool> = (0..24).map(|index| index % 5 == 0).collect();
        let parity = bits(&values, &[2, 3, 4]).parity(1);
        assert_eq!(parity.shape(), [2, 4]);
        let mut expected = Vec::new();
        for outer in 0..2 {
            for bit in 0..4 {
                let set = (0..3).filter(|row| values[(outer * 3 + row) * 4 + bit]);
                expected.push(set.count() % 2 == 1);
            }
        }
        assert_eq!(parity.to_bool_vec(), expected);
    }

    #[test]
    fn test_parity_of_an_empty_axis_is_all_clear() {
        assert_eq!(bits(&[], &[0, 3]).parity(0).to_bool_vec(), [false; 3]);
        let parity = bits(&[], &[3, 0]).parity(1);
        assert_eq!(parity.shape(), [3]);
        assert_eq!(parity.to_bool_vec(), [false; 3]);
    }

    #[test]
    #[should_panic(expected = "axis 2 is out of bounds")]
    fn test_parity_of_an_axis_out_of_bounds_panics() {
        bits(&[false; 6], &[2, 3]).parity(2);
    }

    #[test]
    fn test_popcount_counts_each_row() {
        let values = [
            true, false, true, //
            false, false, false, //
            true, true, true,
        ];
        assert_eq!(bits(&values, &[3, 3]).popcount(), [2, 0, 3]);
    }

    #[test]
    fn test_popcount_spans_the_bytes_of_a_wide_row() {
        assert_eq!(
            row(100, &(0..100).step_by(3).collect::<Vec<_>>()).popcount(),
            [34]
        );
        assert_eq!(bits(&[], &[2, 0]).popcount(), [0, 0]);
        assert_eq!(bits(&[true], &[]).popcount(), [1]);
    }
}
