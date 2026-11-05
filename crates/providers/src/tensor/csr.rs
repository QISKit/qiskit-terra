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

//! A ragged array of bits, stored in compressed-sparse-row form.

use ndarray::ArcArrayD;

use super::{BitTensor, TensorError};

/// A ragged array of bits: one whose entries along a single axis vary in number.
///
/// Post-selection produces one of these. Masking the shots axis of a `Bit[4, 5, 2]` result keeps a
/// different number of shots per sweep point, and a dense array cannot say that: it has one length
/// per axis, and there is no single number to put in the shots slot.
///
/// The encoding is the one sparse matrices use. The entries that survive are stored contiguously,
/// flattened across the ragged axis, and a second array of `offsets` says where each run begins:
///
/// ```text
/// bounds   [4, 5, 2]          the bounding box: at most 5 shots per sweep point
/// axis     1                  the shots axis is the ragged one
/// values   Bit[7, 2]          2 + 2 + 1 + 2 surviving shots, each still 2 bits
/// offsets  [0, 2, 4, 5, 7]    sweep point i is values[offsets[i]..offsets[i + 1]]
/// ```
///
/// The axes before the ragged one are *outer* axes and are dense; `offsets` carries one entry per
/// outer index, plus a final total. The axes after it are *inner* axes, also dense, and pass
/// through onto `values` whole. A ragged axis with no outer axes is the degenerate case of a single
/// run, which is just a dense tensor whose length is not known until run time.
///
/// The values are a [`BitTensor`], so they are bitpacked along their own last axis just like any
/// other array of bits. That is why the ragged axis may not be the last one: packing applies to
/// the final axis, and a ragged final axis would make every run a different number of bits, with
/// no byte boundary to land on. Post-selection satisfies this, since the bits of a shot are the
/// last axis and the shots being selected are the one before it.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrBits {
    /// The largest shape this array can take, with `bounds[axis]` the bound on the ragged axis.
    bounds: Vec<usize>,
    /// Which axis is ragged.
    axis: usize,
    /// The surviving entries, flattened across the ragged axis: `[total, inner axes...]`.
    values: BitTensor,
    /// Where each run begins, as `outer + 1` indices into `values`.
    offsets: ArcArrayD<u64>,
}

impl CsrBits {
    /// Construct a ragged bit array, checking it against the layout it claims.
    ///
    /// `bounds` is the bounding box, whose `axis` entry bounds the ragged axis. `values` holds the
    /// surviving entries flattened across that axis, and `offsets` says where each run begins.
    ///
    /// # Errors
    ///
    /// [`TensorError::CsrAxis`] if `axis` is not an axis of `bounds` or is its last axis, and
    /// [`TensorError::CsrOffsets`] or [`TensorError::CsrValues`] if the two arrays do not describe
    /// a ragged array of that shape. Every invariant the accessors rely on is checked here, so a
    /// constructed `CsrBits` is always well formed.
    pub fn new(
        bounds: Vec<usize>,
        axis: usize,
        values: BitTensor,
        offsets: ArcArrayD<u64>,
    ) -> Result<Self, TensorError> {
        // The last axis is excluded because that is the one `BitTensor` packs along; see the type
        // docs. `bounds.len() - 1` underflows for an empty `bounds`, hence the separate check.
        if bounds.is_empty() || axis >= bounds.len() - 1 {
            return Err(TensorError::CsrAxis {
                axis,
                ndim: bounds.len(),
            });
        }
        let reject = |reason: String| Err(TensorError::CsrOffsets { reason });
        if offsets.ndim() != 1 {
            return reject(format!(
                "they must be a single vector, but have {} axes",
                offsets.ndim()
            ));
        }
        // One offset per outer index, plus a final total, so that every run is a pair.
        let outer: usize = bounds[..axis].iter().product();
        if offsets.len() != outer + 1 {
            return reject(format!(
                "a ragged axis {axis} of bounds {bounds:?} needs {} offset(s), got {}",
                outer + 1,
                offsets.len()
            ));
        }
        let offsets_slice: Vec<u64> = offsets.iter().copied().collect();
        if offsets_slice[0] != 0 {
            return reject(format!(
                "the first offset must be 0, got {}",
                offsets_slice[0]
            ));
        }
        for (index, pair) in offsets_slice.windows(2).enumerate() {
            let (start, end) = (pair[0], pair[1]);
            if end < start {
                return reject(format!(
                    "they must not decrease, but offset {} is {end} after {start}",
                    index + 1
                ));
            }
            let run = end - start;
            let max = bounds[axis] as u64;
            if run > max {
                return reject(format!(
                    "run {index} is {run} long, over the ragged axis's bound of {max}"
                ));
            }
        }
        // `values` is the runs laid end to end, so its leading axis is the final offset and its
        // trailing axes are the inner axes, which a ragged axis leaves untouched.
        let total = *offsets_slice.last().expect("offsets are never empty") as usize;
        let expected: Vec<usize> = std::iter::once(total)
            .chain(bounds[axis + 1..].iter().copied())
            .collect();
        if values.shape() != expected.as_slice() {
            return Err(TensorError::CsrValues {
                expected,
                actual: values.shape().to_vec(),
            });
        }
        Ok(Self {
            bounds,
            axis,
            values,
            offsets,
        })
    }

    /// The largest shape this array can take.
    ///
    /// Every axis but [`axis`](Self::axis) is exact; that one is a bound, since its true length
    /// varies with the outer indices.
    pub fn bounds(&self) -> &[usize] {
        &self.bounds
    }

    /// Which axis is ragged.
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// The surviving entries, flattened across the ragged axis.
    pub fn values(&self) -> &BitTensor {
        &self.values
    }

    /// Where each run begins, as `outer + 1` indices into [`values`](Self::values).
    pub fn offsets(&self) -> &ArcArrayD<u64> {
        &self.offsets
    }

    /// How many entries are stored across every run.
    pub fn total(&self) -> usize {
        self.values
            .shape()
            .first()
            .copied()
            .expect("the values of a ragged array always have a leading axis")
    }

    /// How many runs this array holds, which is the product of its outer axes.
    pub fn runs(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The half-open range of [`values`](Self::values) belonging to run `index`.
    ///
    /// The outer axes are indexed row-major, so a `bounds` of `[2, 3, ...]` ragged on axis 2 has
    /// run `i * 3 + j` for outer index `(i, j)`.
    ///
    /// # Panics
    ///
    /// If `index` is not below [`runs`](Self::runs).
    pub fn run(&self, index: usize) -> std::ops::Range<usize> {
        assert!(
            index < self.runs(),
            "run {index} is out of bounds for {} run(s)",
            self.runs()
        );
        self.offsets[index] as usize..self.offsets[index + 1] as usize
    }

    /// How many entries run `index` holds.
    ///
    /// # Panics
    ///
    /// If `index` is not below [`runs`](Self::runs).
    pub fn run_len(&self, index: usize) -> usize {
        let run = self.run(index);
        run.end - run.start
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Return a 1-D offsets array.
    fn offsets(values: &[u64]) -> ArcArrayD<u64> {
        ndarray::arr1(values).into_dyn().into_shared()
    }

    /// Return all-clear packed values of logical shape `shape`.
    ///
    /// The bit pattern is irrelevant to every invariant here, which are all about shapes and run
    /// boundaries, so the tests say only how many bits there are.
    fn values(shape: &[usize]) -> BitTensor {
        let count = shape.iter().product();
        BitTensor::from_bools(std::iter::repeat_n(false, count), shape).unwrap()
    }

    /// Return the `[4, 5, 2]` post-selection example from the module docs.
    fn post_selection() -> CsrBits {
        // 2 + 2 + 1 + 2 = 7 surviving shots, each still 2 bits.
        CsrBits::new(vec![4, 5, 2], 1, values(&[7, 2]), offsets(&[0, 2, 4, 5, 7])).unwrap()
    }

    #[test]
    fn test_new_accepts_the_post_selection_shape_and_reads_its_runs_back() {
        let csr = post_selection();
        assert_eq!(csr.bounds(), [4, 5, 2]);
        assert_eq!(csr.axis(), 1);
        assert_eq!(csr.total(), 7);
        assert_eq!(csr.runs(), 4, "one run per sweep point");

        // Each sweep point keeps a different number of shots, which is the whole point.
        assert_eq!(
            (0..csr.runs()).map(|i| csr.run_len(i)).collect::<Vec<_>>(),
            [2, 2, 1, 2]
        );
        assert_eq!(csr.run(2), 4..5);
    }

    #[test]
    fn test_new_accepts_empty_runs_and_a_fully_empty_array() {
        // Post-selection can reject every shot of a sweep point, or of all of them.
        let some = CsrBits::new(vec![3, 4, 1], 1, values(&[2, 1]), offsets(&[0, 0, 2, 2])).unwrap();
        assert_eq!(
            (0..some.runs())
                .map(|i| some.run_len(i))
                .collect::<Vec<_>>(),
            [0, 2, 0]
        );

        let none = CsrBits::new(vec![3, 4, 1], 1, values(&[0, 1]), offsets(&[0, 0, 0, 0])).unwrap();
        assert_eq!(none.total(), 0);
    }

    #[test]
    fn test_new_counts_the_offsets_from_the_outer_axes() {
        // Two outer axes multiply: 2 * 3 runs, so 7 offsets.
        assert!(
            CsrBits::new(vec![2, 3, 4, 8], 2, values(&[0, 8]), offsets(&[0; 7])).is_ok(),
            "an array ragged on axis 2 has one run per outer index pair"
        );

        // A ragged leading axis has no outer axes, so it is a single run.
        assert!(
            CsrBits::new(vec![4, 8], 0, values(&[0, 8]), offsets(&[0, 0])).is_ok(),
            "a ragged axis with nothing before it is one run"
        );
    }

    #[test]
    fn test_new_rejects_a_ragged_axis_the_bounds_do_not_have() {
        assert!(matches!(
            CsrBits::new(vec![4, 5], 2, values(&[0, 5]), offsets(&[0])).unwrap_err(),
            TensorError::CsrAxis { axis: 2, ndim: 2 }
        ));
    }

    #[test]
    fn test_new_rejects_a_ragged_last_axis() {
        // The last axis is the one `BitTensor` packs along, so it cannot also be ragged.
        assert!(matches!(
            CsrBits::new(vec![4, 5], 1, values(&[0]), offsets(&[0; 5])).unwrap_err(),
            TensorError::CsrAxis { axis: 1, ndim: 2 }
        ));
        // A one-dimensional array has only a last axis, so nothing about it can be ragged.
        assert!(matches!(
            CsrBits::new(vec![4], 0, values(&[0]), offsets(&[0, 0])).unwrap_err(),
            TensorError::CsrAxis { axis: 0, ndim: 1 }
        ));
    }

    #[test]
    fn test_new_rejects_offsets_that_do_not_describe_runs() {
        let csr = |offs: &[u64]| CsrBits::new(vec![4, 5, 2], 1, values(&[7, 2]), offsets(offs));

        // One per sweep point plus a total is 5; anything else cannot be read as runs.
        assert!(matches!(
            csr(&[0, 2, 4, 7]).unwrap_err(),
            TensorError::CsrOffsets { .. }
        ));
        // Runs are consecutive, so the offsets cannot go backwards.
        assert!(matches!(
            csr(&[0, 4, 2, 5, 7]).unwrap_err(),
            TensorError::CsrOffsets { .. }
        ));
        // The first run starts at the beginning of `values`.
        assert!(matches!(
            csr(&[1, 2, 4, 5, 7]).unwrap_err(),
            TensorError::CsrOffsets { .. }
        ));
        // No run may be longer than the bound on the ragged axis.
        assert!(
            matches!(
                CsrBits::new(vec![2, 3, 2], 1, values(&[5, 2]), offsets(&[0, 4, 5])).unwrap_err(),
                TensorError::CsrOffsets { .. }
            ),
            "a run of 4 cannot fit a ragged axis bounded at 3"
        );
    }

    #[test]
    fn test_new_rejects_values_that_do_not_match_the_runs_or_the_inner_axes() {
        let offs = offsets(&[0, 2, 4, 5, 7]);

        // The leading axis of `values` is the total the offsets end at.
        assert!(matches!(
            CsrBits::new(vec![4, 5, 2], 1, values(&[6, 2]), offs.clone()).unwrap_err(),
            TensorError::CsrValues { expected, actual } if expected == [7, 2] && actual == [6, 2]
        ));
        // The inner axes pass through untouched, so they must agree with the bounds.
        assert!(matches!(
            CsrBits::new(vec![4, 5, 2], 1, values(&[7, 3]), offs).unwrap_err(),
            TensorError::CsrValues { expected, actual } if expected == [7, 2] && actual == [7, 3]
        ));
    }
}
