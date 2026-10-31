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

//! A partition of elements.

use std::fmt;

use thiserror::Error;

/// Why a partition could not be built.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PartitionError {
    #[error("a part must contain at least one element")]
    EmptyPart,
    #[error("an element appears in more than one part, or twice in one")]
    DuplicateElement,
}

/// A grouping of elements into non-overlapping subsets.
///
/// Order in a given part is meaningful. For instance, when sampling from a distribution over
/// multiple elements, it may not be symmetric under permutations of those elements.
///
/// Order of parts is the caller's responsibility and maintained by [`Partition::iter`].
///
/// While a partition can have no parts, each part must have an element.
#[derive(Clone, PartialEq, Eq)]
pub struct Partition<T> {
    /// The elements, in part order then slot order.
    elements: Box<[T]>,
    /// Where each part after the first begins.
    splits: Box<[u32]>,
}

impl<T: Copy + Ord> Partition<T> {
    /// One part per element.
    pub fn singletons(elements: impl IntoIterator<Item = T>) -> Result<Self, PartitionError> {
        Self::new(elements.into_iter().map(|element| vec![element]))
    }

    /// Every element in one part.
    pub fn whole(elements: impl IntoIterator<Item = T>) -> Result<Self, PartitionError> {
        let elements: Vec<T> = elements.into_iter().collect();
        if elements.is_empty() {
            return Err(PartitionError::EmptyPart);
        }
        Self::new([elements])
    }

    /// Specify a new partition from constituent parts.
    pub fn new(
        parts: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
    ) -> Result<Self, PartitionError> {
        let parts = parts.into_iter();
        let mut elements = Vec::new();
        let mut splits = Vec::with_capacity(parts.size_hint().0 + 1);
        splits.push(0);
        for part in parts {
            let start = elements.len();
            elements.extend(part);
            if elements.len() == start {
                return Err(PartitionError::EmptyPart);
            }
            splits.push(elements.len() as u32);
        }

        // Elements must be unique across the whole partition.
        let mut seen = elements.clone();
        seen.sort_unstable();
        if seen.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(PartitionError::DuplicateElement);
        }

        Ok(Partition {
            elements: elements.into_boxed_slice(),
            splits: splits.into_boxed_slice(),
        })
    }

    /// How many parts there are.
    pub fn num_parts(&self) -> usize {
        self.splits.len() - 1
    }

    /// How many elements the partition covers, which may be fewer than its carrier has.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// The elements.
    pub fn elements(&self) -> &[T] {
        &self.elements
    }

    /// Get an iterator over the individual parts.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[T]> {
        std::ops::Range {
            start: 0,
            end: self.splits.len() - 1,
        }
        .map(|i| &self.elements[self.splits[i] as usize..self.splits[i + 1] as usize])
    }

    /// Whether every part is a single element.
    pub fn is_singletons(&self) -> bool {
        self.num_parts() == self.elements.len()
    }

    /// Whether this partition contains `element`.
    pub fn contains(&self, element: &T) -> bool {
        self.elements.contains(element)
    }
}

impl<T: fmt::Debug> fmt::Debug for Partition<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for i in 0..self.splits.len() - 1 {
            list.entry(&&self.elements[self.splits[i] as usize..self.splits[i + 1] as usize]);
        }
        list.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A partition of `usize` elements from explicit parts, which must be valid.
    fn parts(parts: &[&[usize]]) -> Partition<usize> {
        Partition::new(parts.iter().map(|part| part.to_vec())).expect("a valid partition")
    }

    #[test]
    fn test_singletons() {
        let partition = Partition::singletons([3, 5, 4]).unwrap();
        assert_eq!(partition.num_parts(), 3);
        assert_eq!(partition.num_elements(), 3);
        assert!(partition.is_singletons());
        assert_eq!(
            partition.iter().collect::<Vec<_>>(),
            vec![[3].as_slice(), [5].as_slice(), [4].as_slice()]
        );
    }

    #[test]
    fn test_whole() {
        let partition = Partition::whole([5, 3]).unwrap();
        assert_eq!(partition.num_parts(), 1);
        assert_eq!(partition.num_elements(), 2);
        assert!(!partition.is_singletons());
        assert_eq!(
            partition.iter().collect::<Vec<_>>(),
            vec![[5, 3].as_slice()]
        );
        assert!(Partition::whole([7]).unwrap().is_singletons());
    }

    #[test]
    fn test_a_parts_own_order_is_kept_and_is_meaningful() {
        let forward = parts(&[&[0, 1]]);
        let reversed = parts(&[&[1, 0]]);
        assert_eq!(forward.iter().collect::<Vec<_>>(), vec![[0, 1].as_slice()]);
        assert_eq!(reversed.iter().collect::<Vec<_>>(), vec![[1, 0].as_slice()]);
        assert_ne!(forward, reversed);

        assert_eq!(
            Partition::new([vec![0usize, 1], vec![1, 0]]).unwrap_err(),
            PartitionError::DuplicateElement
        );
    }

    #[test]
    fn test_a_partition_with_arbitrary_elements() {
        let partition = parts(&[&[1], &[3]]);
        assert_eq!(partition.num_elements(), 2);
        assert_eq!(partition.elements(), [1, 3]);
        assert!(partition.contains(&1) && partition.contains(&3));
        assert!(!partition.contains(&0) && !partition.contains(&2));
    }

    #[test]
    fn test_order_of_parts_matters() {
        let increasing = parts(&[&[0], &[1, 2]]);
        let decreasing = parts(&[&[1, 2], &[0]]);
        assert_ne!(increasing, decreasing);
    }

    #[test]
    fn test_iter_round_trips_the_parts_it_was_built_from() {
        let partition = parts(&[&[0], &[7, 5, 6], &[2, 1]]);
        assert_eq!(
            partition.iter().collect::<Vec<_>>(),
            vec![[0].as_slice(), [7, 5, 6].as_slice(), [2, 1].as_slice()]
        );
        assert_eq!(partition.num_parts(), 3);
        assert_eq!(partition.iter().len(), 3);
        assert_eq!(partition.num_elements(), 6);
    }

    #[test]
    fn test_whole_does_not_accept_no_elemets() {
        assert_eq!(
            Partition::<i32>::whole([]).unwrap_err(),
            PartitionError::EmptyPart
        )
    }

    #[test]
    fn test_new_rejects_what_is_not_a_partition() {
        assert_eq!(
            Partition::new([vec![0usize], vec![]]).unwrap_err(),
            PartitionError::EmptyPart
        );
        assert_eq!(
            Partition::new([vec![1usize, 1]]).unwrap_err(),
            PartitionError::DuplicateElement
        );
        assert_eq!(
            Partition::new([vec![0usize, 1], vec![1, 2]]).unwrap_err(),
            PartitionError::DuplicateElement
        );
    }

    #[test]
    fn test_new_accepts_no_parts_at_all() {
        let empty = Partition::<usize>::new(Vec::<Vec<usize>>::new()).unwrap();
        assert_eq!(empty.num_parts(), 0);
        assert_eq!(empty.num_elements(), 0);
        assert_eq!(empty.iter().next(), None);
        assert_eq!(format!("{empty:?}"), "[]");
    }

    #[test]
    fn test_debug_shows_the_parts() {
        assert_eq!(format!("{:?}", parts(&[&[0], &[2, 1]])), "[[0], [2, 1]]");
        assert_eq!(format!("{:?}", parts(&[&[7]])), "[[7]]");
    }
}
