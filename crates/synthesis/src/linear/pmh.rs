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

use hashbrown::HashMap;
use ndarray::{Array1, Array2, ArrayViewMut2, Axis, s};
use numpy::PyReadonlyArray2;
use smallvec::{SmallVec, smallvec};
use std::cmp;

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardGate};

use pyo3::prelude::*;

use super::utils::_add_row_or_col;

/// This helper function allows transposed access to a matrix.
fn _index(transpose: bool, i: usize, j: usize) -> (usize, usize) {
    if transpose { (j, i) } else { (i, j) }
}

fn _ceil_fraction(numerator: usize, denominator: usize) -> usize {
    let mut fraction = numerator / denominator;
    if !numerator.is_multiple_of(denominator) {
        fraction += 1;
    }
    fraction
}

/// This function is a helper function of the algorithm for optimal synthesis
/// of linear reversible circuits (the Patel–Markov–Hayes algorithm). It works
/// like gaussian elimination, except that it works a lot faster, and requires
/// fewer steps (and therefore fewer CNOTs). It takes the matrix and
/// splits it into sections of size section_size. Then it eliminates all non-zero
/// sub-rows within each section, which are the same as a non-zero sub-row
/// above. Once this has been done, it continues with normal gaussian elimination.
/// The benefit is that with small section sizes, most of the sub-rows will
/// be cleared in the first step, resulting in a factor ``section_size`` fewer row row operations
/// during Gaussian elimination.
///
/// The algorithm is described in detail in the following paper
/// "Optimal synthesis of linear reversible circuits."
/// Patel, Ketan N., Igor L. Markov, and John P. Hayes.
/// Quantum Information & Computation 8.3 (2008): 282-294.
///
/// Note:
/// This implementation tweaks the Patel, Markov, and Hayes algorithm by adding
/// a "back reduce" which adds rows below the pivot row with a high degree of
/// overlap back to it. The intuition is to avoid a high-weight pivot row
/// increasing the weight of lower rows.
///
/// Args:
///     matrix: square matrix, describing a linear quantum circuit
///     section_size: the section size the matrix columns are divided into
///
/// Returns:
///     A vector of CX locations (control, target) that need to be applied.
fn lower_cnot_synth(
    mut matrix: ArrayViewMut2<bool>,
    section_size: usize,
    transpose: bool,
) -> Vec<(usize, usize)> {
    // The vector of CNOTs to be applied. Called ``circuit`` here for consistency with the paper.
    let mut circuit: Vec<(usize, usize)> = Vec::new();
    let cutoff = 1;

    // to apply to the transposed matrix, we can just set axis = 1
    let row_axis = if transpose { Axis(1) } else { Axis(0) };

    // get number of columns (same as rows) and the number of sections
    let n = matrix.raw_dim()[0];
    let num_sections = _ceil_fraction(n, section_size);

    // iterate over the columns
    for section in 1..num_sections + 1 {
        // store sub section row patterns here, which we saw already
        let mut patterns: HashMap<Array1<bool>, usize> = HashMap::new();
        let section_slice = s![(section - 1) * section_size..cmp::min(section * section_size, n)];

        // iterate over the rows (note we only iterate from the diagonal downwards)
        for row_idx in (section - 1) * section_size..n {
            // we need to keep track of the rows we saw already, called ``pattern`` here
            let pattern: Array1<bool> = matrix
                .index_axis(row_axis, row_idx)
                .slice(section_slice)
                .to_owned();

            // skip if the row is empty (i.e. all elements are false)
            if pattern.iter().any(|&el| el) {
                if patterns.contains_key(&pattern) {
                    // store CX location
                    circuit.push((patterns[&pattern], row_idx));
                    // remove the row
                    _add_row_or_col(matrix.view_mut(), &transpose, patterns[&pattern], row_idx);
                } else {
                    // if we have not seen this pattern yet, keep track of it
                    patterns.insert(pattern, row_idx);
                }
            }
        }

        // gaussian eliminate the remainder of the section
        for col_idx in (section - 1) * section_size..cmp::min(section * section_size, n) {
            let mut diag_el = matrix[[col_idx, col_idx]];

            for r in col_idx + 1..n {
                if matrix[_index(transpose, r, col_idx)] {
                    if !diag_el {
                        _add_row_or_col(matrix.view_mut(), &transpose, r, col_idx);
                        circuit.push((r, col_idx));
                        diag_el = true
                    }
                    _add_row_or_col(matrix.view_mut(), &transpose, col_idx, r);
                    circuit.push((col_idx, r));
                }

                // back-reduce to the pivot row: this one-line-magic checks if the logical AND
                // between the two target rows has more ``true`` elements is larger than the cutoff
                if matrix
                    .index_axis(row_axis, col_idx)
                    .iter()
                    .zip(matrix.index_axis(row_axis, r).iter())
                    .map(|(&i, &j)| i & j)
                    .filter(|&x| x)
                    .count()
                    > cutoff
                {
                    _add_row_or_col(matrix.view_mut(), &transpose, r, col_idx);
                    circuit.push((r, col_idx));
                }
            }
        }
    }
    circuit
}

/// Synthesize a linear function, given by a boolean square matrix, into a circuit.
/// This function uses the Patel-Markov-Hayes algorithm, described in arXiv:quant-ph/0302002,
/// using section-wise elimination of the rows.
#[pyfunction]
#[pyo3(signature = (matrix, section_size=None))]
pub fn synth_cnot_count_full_pmh(
    matrix: PyReadonlyArray2<bool>,
    section_size: Option<usize>,
) -> PyResult<PyCircuitData> {
    let arrayview = matrix.as_array();
    let mat: Array2<bool> = arrayview.to_owned();
    let num_qubits = mat.nrows();

    let instructions = synth_pmh(mat, section_size);
    Ok(
        CircuitData::from_standard_gates(num_qubits as u32, instructions, Param::Float(0.0))?
            .into(),
    )
}

type Instruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);
pub fn synth_pmh(
    mut mat: Array2<bool>,
    section_size: Option<usize>,
) -> impl DoubleEndedIterator<Item = Instruction> {
    let num_qubits = mat.nrows(); // is a quadratic matrix

    // If given, use the user-specified input size. If None, we default to
    // alpha * log2(num_qubits), as suggested in the paper, where the coefficient alpha
    // is calibrated to minimize the upper bound on the number of row operations.
    // In addition, we at least set a block size of 2, which, in practice, minimizes the bound
    // until ~100 qubits.
    let alpha = 0.56;
    let blocksize = match section_size {
        Some(section_size) => section_size,
        None => std::cmp::max(2, (alpha * (num_qubits as f64).log2()).floor() as usize),
    };

    // compute the synthesis for the lower triangular part of the matrix, and then
    // apply it on the transposed part for the full synthesis
    let lower_cnots = lower_cnot_synth(mat.view_mut(), blocksize, false);
    let upper_cnots = lower_cnot_synth(mat.view_mut(), blocksize, true);

    // iterator over the gates
    upper_cnots
        .into_iter()
        .map(|(i, j)| (j, i))
        .chain(lower_cnots.into_iter().rev())
        .map(|(ctrl, target)| {
            (
                StandardGate::CX,
                smallvec![],
                smallvec![Qubit(ctrl as u32), Qubit(target as u32)],
            )
        })
}
