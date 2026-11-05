// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::PyErr;
use ndarray::ArrayViewMut1;
use ndarray::{Array1, ArrayView1};
use pyo3::exceptions::PyValueError;
use std::vec::Vec;

use qiskit_util::py::PySequenceIndex;

/// Possible errors that can occur when validating a permutation pattern.
#[derive(Debug, thiserror::Error)]
pub(crate) enum PermutationError {
    #[error("invalid permutation: input contains a negative number")]
    NegativeEntry,

    #[error("invalid permutation: input has length {length} and contains {value}")]
    OutOfBounds { length: usize, value: i64 },

    #[error("invalid permutation: input contains {value} more than once")]
    DuplicateEntry { value: i64 },
}

impl From<PermutationError> for PyErr {
    fn from(value: PermutationError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

pub fn validate_permutation(pattern: &ArrayView1<i64>) -> Result<(), PermutationError> {
    let n = pattern.len();
    let mut seen: Vec<bool> = vec![false; n];

    for &x in pattern {
        if x < 0 {
            return Err(PermutationError::NegativeEntry);
        }

        if x as usize >= n {
            return Err(PermutationError::OutOfBounds {
                length: n,
                value: x,
            });
        }

        if seen[x as usize] {
            return Err(PermutationError::DuplicateEntry { value: x });
        }

        seen[x as usize] = true;
    }

    Ok(())
}

pub fn invert(pattern: &ArrayView1<i64>) -> Array1<usize> {
    let mut inverse: Array1<usize> = Array1::zeros(pattern.len());
    pattern.iter().enumerate().for_each(|(ii, &jj)| {
        inverse[jj as usize] = ii;
    });
    inverse
}

/// Sorts the input permutation by iterating through the permutation list
/// and putting each element to its correct position via a SWAP (if it's not
/// at the correct position already). If ``n`` is the length of the input
/// permutation, this requires at most ``n`` SWAPs.
///
/// More precisely, if the input permutation is a cycle of length ``m``,
/// then this creates a quantum circuit with ``m-1`` SWAPs (and of depth ``m-1``);
/// if the input  permutation consists of several disjoint cycles, then each cycle
/// is essentially treated independently.
pub fn get_ordered_swap(pattern: &ArrayView1<i64>) -> Vec<(usize, usize)> {
    let mut permutation: Vec<usize> = pattern.iter().map(|&x| x as usize).collect();
    let mut index_map = invert(pattern);

    let n = permutation.len();
    let mut swaps: Vec<(usize, usize)> = Vec::with_capacity(n);
    for ii in 0..n {
        let val = permutation[ii];
        if val == ii {
            continue;
        }
        let jj = index_map[ii];
        swaps.push((ii, jj));
        (permutation[ii], permutation[jj]) = (permutation[jj], permutation[ii]);
        index_map[val] = jj;
        index_map[ii] = ii;
    }

    swaps[..].reverse();
    swaps
}

/// Explore cycles in a permutation pattern. This is probably best explained in an
/// example: let a pattern be [1, 2, 3, 0, 4, 6, 5], then it contains the two
/// cycles [1, 2, 3, 0] and [6, 5]. The index [4] does not perform a permutation and does
/// therefore not create a cycle.
pub fn pattern_to_cycles(pattern: &ArrayView1<usize>) -> Vec<Vec<usize>> {
    // vector keeping track of which elements in the permutation pattern have been visited
    let mut explored: Vec<bool> = vec![false; pattern.len()];

    // vector to store the cycles
    let mut cycles: Vec<Vec<usize>> = Vec::new();

    for pos in pattern {
        let mut cycle: Vec<usize> = Vec::new();

        // follow the cycle until we reached an entry we saw before
        let mut i = *pos;
        while !explored[i] {
            cycle.push(i);
            explored[i] = true;
            i = pattern[i];
        }
        // cycles must have more than 1 element
        if cycle.len() > 1 {
            cycles.push(cycle);
        }
    }

    cycles
}

/// Periodic (or Python-like) access to a vector.
/// Util used below in ``decompose_cycles``.
#[inline]
fn pget(vec: &[usize], index: isize) -> usize {
    vec[PySequenceIndex::convert_idx(index, vec.len()).unwrap()]
}

/// Given a disjoint cycle decomposition of a permutation pattern (see the function
/// ``pattern_to_cycles``), decomposes every cycle into a series of SWAPs to implement it.
/// In combination with ``pattern_to_cycle``, this function allows to implement a
/// full permutation pattern by applying SWAP gates on the returned index-pairs.
pub fn decompose_cycles(cycles: &Vec<Vec<usize>>) -> Vec<(usize, usize)> {
    let mut swaps: Vec<(usize, usize)> = Vec::new();

    for cycle in cycles {
        let length = cycle.len() as isize;

        for idx in 0..(length - 1) / 2 {
            swaps.push((pget(cycle, idx - 1), pget(cycle, length - 3 - idx)));
        }
        for idx in 0..length / 2 {
            swaps.push((pget(cycle, idx - 1), pget(cycle, length - 2 - idx)));
        }
    }

    swaps
}

/// Implements a single swap layer, consisting of conditional swaps between each
/// neighboring couple. The starting_point is the first qubit to use (either 0 or 1
/// for even or odd layers respectively). Mutates the permutation pattern ``pattern``.
pub fn create_swap_layer(
    pattern: &mut ArrayViewMut1<usize>,
    starting_point: usize,
) -> Vec<(usize, usize)> {
    let num_qubits = pattern.len();
    let mut gates = Vec::new();

    for j in (starting_point..num_qubits - 1).step_by(2) {
        if pattern[j] > pattern[j + 1] {
            gates.push((j, j + 1));
            pattern.swap(j, j + 1);
        }
    }
    gates
}
