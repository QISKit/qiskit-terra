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

//! How an array of bits is stored: densely, or compressed along a ragged axis.

use super::{BitTensor, CsrBits, Layout};

#[cfg(test)]
use super::Tensor;

/// The storage backing a `Bit` tensor.
///
/// `Bit` is the one dtype with two representations. Post-selection keeps a different number of
/// shots per sweep point, and a dense array cannot say that, so the result is compressed along the
/// shots axis instead -- which is the data reduction that post-selection exists to achieve.
#[derive(Clone, Debug, PartialEq)]
pub enum BitStorage {
    /// Every position the shape describes holds a bit.
    Dense(BitTensor),
    /// The bits along one axis vary in number, compressed as in [`CsrBits`].
    Sparse(CsrBits),
}

impl BitStorage {
    /// Return the extent of each axis.
    ///
    /// For [`Dense`](Self::Dense) this is the logical shape, every position of which holds a bit.
    /// For [`Sparse`](Self::Sparse) it is the *bounding box*: the ragged axis reports its bound,
    /// and the true number of entries there varies per outer index. Callers that need a true
    /// extent must check [`layout`](Self::layout) first.
    pub fn shape(&self) -> &[usize] {
        match self {
            BitStorage::Dense(bits) => bits.shape(),
            BitStorage::Sparse(csr) => csr.bounds(),
        }
    }

    /// Return whether every position this storage's shape describes holds an entry.
    pub fn layout(&self) -> Layout {
        match self {
            BitStorage::Dense(_) => Layout::Dense,
            BitStorage::Sparse(csr) => Layout::Ragged { axis: csr.axis() },
        }
    }

    /// Return whether the entries vary in number along some axis.
    pub fn is_ragged(&self) -> bool {
        matches!(self, BitStorage::Sparse(_))
    }

    /// Return the dense bits, or `None` if this storage is ragged.
    pub fn dense(&self) -> Option<&BitTensor> {
        match self {
            BitStorage::Dense(bits) => Some(bits),
            BitStorage::Sparse(_) => None,
        }
    }

    /// Return the compressed bits, or `None` if this storage is dense.
    pub fn sparse(&self) -> Option<&CsrBits> {
        match self {
            BitStorage::Dense(_) => None,
            BitStorage::Sparse(csr) => Some(csr),
        }
    }
}

impl From<BitTensor> for BitStorage {
    fn from(bits: BitTensor) -> Self {
        BitStorage::Dense(bits)
    }
}

impl From<CsrBits> for BitStorage {
    fn from(csr: CsrBits) -> Self {
        BitStorage::Sparse(csr)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::{Dim, TensorError};
    use ndarray::ArcArrayD;

    /// Return all-clear packed values of logical shape `shape`.
    fn values(shape: &[usize]) -> BitTensor {
        let count = shape.iter().product();
        BitTensor::from_bools(std::iter::repeat_n(false, count), shape).unwrap()
    }

    /// Return a 1-D offsets array.
    fn offsets(values: &[u64]) -> ArcArrayD<u64> {
        ndarray::arr1(values).into_dyn().into_shared()
    }

    /// Return the `[4, 5, 2]` post-selection example from the `CsrBits` docs, ragged on axis 1.
    fn post_selection() -> CsrBits {
        CsrBits::new(vec![4, 5, 2], 1, values(&[7, 2]), offsets(&[0, 2, 4, 5, 7])).unwrap()
    }

    #[test]
    fn test_dense_reports_its_shape_and_a_dense_layout() {
        let storage = BitStorage::Dense(values(&[4, 5, 2]));
        assert_eq!(storage.shape(), [4, 5, 2]);
        assert_eq!(storage.layout(), Layout::Dense);
        assert!(!storage.is_ragged());
    }

    #[test]
    fn test_sparse_reports_its_bounding_box_and_ragged_axis() {
        let storage = BitStorage::Sparse(post_selection());
        // The shape is the bounding box: axis 1 holds at most 5, and actually holds 2, 2, 1, 2.
        assert_eq!(storage.shape(), [4, 5, 2]);
        assert_eq!(storage.layout(), Layout::Ragged { axis: 1 });
        assert!(storage.is_ragged());
    }

    #[test]
    fn test_dense_accessor_admits_only_dense_storage() {
        // This is the gate an op defined only over dense bits uses: a `None` means refuse, so
        // that no ragged value ever reaches a kernel that would read its shape as a true extent.
        let dense = BitStorage::Dense(values(&[4, 5, 2]));
        assert_eq!(dense.dense(), Some(&values(&[4, 5, 2])));
        assert_eq!(dense.sparse(), None);

        let sparse = BitStorage::Sparse(post_selection());
        assert_eq!(sparse.dense(), None);
        assert_eq!(sparse.sparse(), Some(&post_selection()));
    }

    #[test]
    fn test_from_conversions_pick_the_matching_variant() {
        assert_eq!(
            BitStorage::from(values(&[3])),
            BitStorage::Dense(values(&[3]))
        );
        assert_eq!(
            BitStorage::from(post_selection()),
            BitStorage::Sparse(post_selection())
        );
    }

    /// The bitwise methods are dense-only until the ragged arms land, and say so.
    #[test]
    fn test_a_ragged_operand_is_refused_by_the_inherent_bitwise_methods() {
        let ragged = Tensor::Bit(post_selection().into());
        let dense = Tensor::bit(values(&[4, 5, 2]));
        assert!(matches!(
            ragged.bitand_tensor(&dense),
            Err(TensorError::RaggedOperand { op: "bitand" })
        ));
        assert!(matches!(
            dense.bitxor_tensor(&ragged),
            Err(TensorError::RaggedOperand { op: "bitxor" })
        ));
        assert!(matches!(
            ragged.broadcast_to(&[2, 4, 5, 2]),
            Err(TensorError::RaggedOperand { op: "broadcast_to" })
        ));
    }

    /// Layout comparison is Stage 3; until then `matches` rejects raggedness on either side.
    ///
    /// This is conservative, not correct: a ragged value genuinely does satisfy a ragged type of
    /// the same dtype, axis and bounds. Pinning it here means the change of behaviour is visible
    /// in the diff that fixes it.
    #[test]
    fn test_matches_does_not_yet_compare_layouts() {
        let ragged = Tensor::Bit(post_selection().into());
        let ragged_ty = ragged.tensor_type();
        assert!(ragged_ty.is_ragged());
        assert!(
            !ragged.matches(&ragged_ty),
            "a ragged value does not yet satisfy even its own type"
        );

        let dense = Tensor::bit(values(&[4, 5, 2]));
        assert!(
            dense.matches(&dense.tensor_type()),
            "a dense value still satisfies its own type"
        );
        assert!(!dense.matches(&ragged_ty));
    }

    /// A ragged value's type reports the ragged axis as a bound, not as a length.
    #[test]
    fn test_tensor_type_of_a_ragged_value_is_ragged() {
        let ragged = Tensor::Bit(post_selection().into());
        let ty = ragged.tensor_type();
        assert_eq!(ty.layout, Layout::Ragged { axis: 1 });
        assert!(ty.is_ragged());
        assert_eq!(
            ty.shape,
            vec![Dim::Fixed(4), Dim::Bounded { max: 5 }, Dim::Fixed(2)]
        );
        assert!(ragged.is_ragged());
        assert_eq!(ragged.layout(), Layout::Ragged { axis: 1 });

        // A dense value of the same bounding box is unchanged: every axis a true length.
        let dense = Tensor::bit(values(&[4, 5, 2])).tensor_type();
        assert_eq!(dense.layout, Layout::Dense);
        assert!(!dense.is_ragged());
    }
}
