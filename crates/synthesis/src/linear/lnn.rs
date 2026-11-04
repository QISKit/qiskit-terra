// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::linear::utils::{_col_op, _row_op, _row_sum, calc_inverse_matrix_inner};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};
use numpy::PyReadonlyArray2;
use smallvec::smallvec;

use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardGate};

// Optimize the synthesis of an n-qubit circuit contains only CX gates for
// linear nearest neighbor (LNN) connectivity.
// The depth of the circuit is bounded by 5*n, while the gate count is approximately 2.5*n^2
//
// References:
// [1]: Kutin, S., Moulton, D. P., Smithline, L. (2007).
// Computation at a Distance.
// `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_.

type InstructionList = Vec<(usize, usize)>;

/// Add a cx gate to the instructions and update the matrix mat
fn _row_op_update_instructions(
    cx_instructions: &mut InstructionList,
    mat: ArrayViewMut2<bool>,
    a: usize,
    b: usize,
) {
    cx_instructions.push((a, b));
    _row_op(mat, a, b);
}

/// Get the instructions for a lower triangular basis change of a matrix mat.
/// See the proof of Proposition 7.3 in [1].
/// mat_inv needs to be the inverted matrix of mat
/// The outputs are the permuted versions of mat and mat_inv
fn _get_lower_triangular<'a>(
    n: usize,
    mat: ArrayView2<bool>,
    mut mat_inv: ArrayViewMut2<'a, bool>,
) -> (Array2<bool>, ArrayViewMut2<'a, bool>) {
    let mut mat = mat.to_owned();
    let mut mat_t = mat.to_owned();

    let mut cx_instructions_rows: InstructionList = Vec::new();
    // Use the instructions in U, which contains only gates of the form cx(a,b) a>b
    // to transform the matrix to a permuted lower-triangular matrix.
    // The original Matrix mat is unchanged, but mat_inv is

    for i in (0..n).rev() {
        // Find the last "1" in row i, use COL operations to the left in order to
        // zero out all other "1"s in that row.
        let cols_to_update: Vec<usize> = (0..n).rev().filter(|&j| mat[[i, j]]).collect();
        let (first_j, cols_to_update) = cols_to_update.split_first().unwrap();
        cols_to_update.iter().for_each(|j| {
            _col_op(mat.view_mut(), *first_j, *j);
        });

        // Use row operations directed upwards to zero out all "1"s above the remaining "1" in row i
        let rows_to_update: Vec<usize> = (0..i).rev().filter(|k| mat[[*k, *first_j]]).collect();
        rows_to_update.into_iter().for_each(|k| {
            _row_op_update_instructions(&mut cx_instructions_rows, mat.view_mut(), i, k);
        });
    }
    // Apply only U instructions to get the permuted L
    for (ctrl, trgt) in cx_instructions_rows {
        _row_op(mat_t.view_mut(), ctrl, trgt);
        _col_op(mat_inv.view_mut(), trgt, ctrl); // performs an inverted col_op
    }
    (mat_t, mat_inv)
}

/// For each row in mat_t, save the column index of the last "1"
fn _get_label_arr(n: usize, mat_t: ArrayView2<bool>) -> Vec<usize> {
    (0..n)
        .map(|i| (0..n).find(|&j| mat_t[[i, n - 1 - j]]).unwrap_or(n))
        .collect()
}

/// Check if "row" is a linear combination of all rows in mat_inv_t not including the row labeled by k
fn _in_linear_combination(
    label_arr_t: &[usize],
    mat_inv_t: ArrayView2<bool>,
    row: ArrayView1<bool>,
    k: usize,
) -> bool {
    // Find the linear combination of mat_t rows which produces "row"
    !(0..row.len())
        .filter(|&row_l| row[row_l])
        .fold(Array1::from_elem(row.len(), false), |w_needed, row_l| {
            _row_sum(w_needed.view(), mat_inv_t.row(row_l)).unwrap()
        })[label_arr_t[k]]
}

/// Returns label_arr_t = label_arr^(-1)
fn _get_label_arr_t(n: usize, label_arr: &[usize]) -> Vec<usize> {
    let mut label_arr_t: Vec<usize> = vec![0; n];
    (0..n).for_each(|i| label_arr_t[label_arr[i]] = i);
    label_arr_t
}

/// Transform an arbitrary boolean invertible matrix to a north-west triangular matrix
/// by Proposition 7.3 in [1]
fn _matrix_to_north_west(
    n: usize,
    mut mat: ArrayViewMut2<bool>,
    mut mat_inv: ArrayViewMut2<bool>,
) -> InstructionList {
    // The rows of mat_t hold all w_j vectors (see [1]). mat_inv_t is the inverted matrix of mat_t
    // To save time on needless copying, we change mat_inv into mat_inv_t, since we won't need mat_inv anymore
    let (mat_t, mat_inv_t) = _get_lower_triangular(n, mat.view(), mat_inv.view_mut());
    // Get all pi(i) labels
    let mut label_arr = _get_label_arr(n, mat_t.view());

    // Save the original labels, exchange index <-> value
    let label_arr_t = _get_label_arr_t(n, &label_arr);
    let mut first_qubit = 0;
    let mut empty_layers = 0;
    let mut done = false;
    let mut cx_instructions_rows: InstructionList = Vec::new();
    while !done {
        // At each iteration the values of i switch between even and odd
        let mut at_least_one_needed = false;
        for i in (first_qubit..n.saturating_sub(1)).step_by(2) {
            // "If j < k, we do nothing" (see [1])
            // "If j > k, we swap the two labels, and we also perform a box" (see [1])
            if label_arr[i] > label_arr[i + 1] {
                at_least_one_needed = true;
                // iterate on column indices, output rows as Vec<bool>
                let row_sum = _row_sum(mat.row(i), mat.row(i + 1)).unwrap();
                // "Let W be the span of all w_l for l!=k" (see [1])
                // " We can perform a box on <i> and <i + 1> that writes a vector in W to wire <i + 1>."
                // (see [1])
                if _in_linear_combination(
                    &label_arr_t,
                    mat_inv_t.view(),
                    mat.row(i + 1),
                    label_arr[i + 1],
                ) {
                    // do nothing
                } else if _in_linear_combination(
                    &label_arr_t,
                    mat_inv_t.view(),
                    row_sum.view(),
                    label_arr[i + 1],
                ) {
                    _row_op_update_instructions(
                        &mut cx_instructions_rows,
                        mat.view_mut(),
                        i,
                        i + 1,
                    );
                } else if _in_linear_combination(
                    &label_arr_t,
                    mat_inv_t.view(),
                    mat.row(i),
                    label_arr[i + 1],
                ) {
                    _row_op_update_instructions(
                        &mut cx_instructions_rows,
                        mat.view_mut(),
                        i + 1,
                        i,
                    );
                    _row_op_update_instructions(
                        &mut cx_instructions_rows,
                        mat.view_mut(),
                        i,
                        i + 1,
                    );
                }
                (label_arr[i], label_arr[i + 1]) = (label_arr[i + 1], label_arr[i]);
            }
        }
        if !at_least_one_needed {
            empty_layers += 1;
            if empty_layers > 1 {
                // if nothing happened twice in a row, then finished.
                done = true;
            }
        } else {
            empty_layers = 0;
        }
        first_qubit = 1 - first_qubit;
    }
    cx_instructions_rows
}

/// Transform a north-west triangular matrix to identity in depth 3*n by Proposition 7.4 of [1]
fn _north_west_to_identity(n: usize, mut mat: ArrayViewMut2<bool>) -> InstructionList {
    // At start the labels are in reversed order
    let mut label_arr: Vec<usize> = (0..n).rev().collect();
    let mut first_qubit = 0;
    let mut empty_layers = 0;
    let mut done = false;
    let mut cx_instructions_rows: InstructionList = Vec::new();
    while !done {
        let mut at_least_one_needed = false;
        for i in (first_qubit..n.saturating_sub(1)).step_by(2) {
            // Exchange the labels if needed
            if label_arr[i] > label_arr[i + 1] {
                at_least_one_needed = true;
                // If row i has "1" in column i+1, swap and remove the "1" (in depth 2)
                // otherwise, only do a swap (in depth 3)
                if !mat[[i, label_arr[i + 1]]] {
                    // Adding this turns the operation to a SWAP
                    _row_op_update_instructions(
                        &mut cx_instructions_rows,
                        mat.view_mut(),
                        i + 1,
                        i,
                    );
                }
                _row_op_update_instructions(&mut cx_instructions_rows, mat.view_mut(), i, i + 1);
                _row_op_update_instructions(&mut cx_instructions_rows, mat.view_mut(), i + 1, i);

                (label_arr[i], label_arr[i + 1]) = (label_arr[i + 1], label_arr[i]);
            }
        }

        if !at_least_one_needed {
            empty_layers += 1;
            if empty_layers > 1 {
                // if nothing happened twice in a row, then finished.
                done = true;
            }
        } else {
            empty_layers = 0;
        }
        first_qubit = 1 - first_qubit;
    }
    cx_instructions_rows
}

/// Find instruction to synthesize CX circuit in depth bounded by 5n for LNN connectivity.
/// The algorithm [1] has two steps:
/// a) transform the original matrix to a north-west matrix (m2nw),
/// b) transform the north-west matrix to identity (nw2id).
///
/// A square n-by-n matrix A is called north-west if A[i][j]=0 for all i+j>=n
/// For example, the following matrix is north-west:
/// [[0, 1, 0, 1]
/// [1, 1, 1, 0]
/// [0, 1, 0, 0]
/// [1, 0, 0, 0]]
///
/// According to [1] the synthesis is done on the inverse matrix
/// so the matrix mat is inverted at this step
///
/// References:
/// [1]: Kutin, S., Moulton, D. P., Smithline, L. (2007).
/// Computation at a Distance.
/// `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_.
pub fn synth_cnot_lnn_instructions(
    arrayview: ArrayView2<bool>,
) -> (InstructionList, InstructionList) {
    // According to [1] the synthesis is done on the inverse matrix
    // so the matrix mat is inverted at this step
    let mut mat_inv: Array2<bool> = arrayview.to_owned();
    let mut mat_cpy = calc_inverse_matrix_inner(mat_inv.view(), false).unwrap();

    let n = mat_cpy.nrows();

    // Transform an arbitrary invertible matrix to a north-west triangular matrix
    // by Proposition 7.3 of [1]

    let cx_instructions_rows_m2nw =
        _matrix_to_north_west(n, mat_cpy.view_mut(), mat_inv.view_mut());
    // Transform a north-west triangular matrix to identity in depth 3*n
    // by Proposition 7.4 of [1]

    let cx_instructions_rows_nw2id = _north_west_to_identity(n, mat_cpy.view_mut());

    (cx_instructions_rows_m2nw, cx_instructions_rows_nw2id)
}

/// Find instruction to synthesize CX circuit in depth bounded by 5n for LNN connectivity.
/// Uses the algorithm by Kutin, Moulton, Smithline
/// described in `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_.
/// Returns: Tuple with two lists of instructions for CX gates
/// Corresponding to the two parts of the algorithm
#[pyfunction]
#[pyo3(signature = (mat))]
pub fn py_synth_cnot_lnn_instructions(
    mat: PyReadonlyArray2<bool>,
) -> PyResult<(InstructionList, InstructionList)> {
    Ok(synth_cnot_lnn_instructions(mat.as_array()))
}

/// Synthesize CX circuit in depth bounded by 5n for LNN connectivity.
/// Uses the algorithm by Kutin, Moulton, Smithline
/// described in `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_.
/// Returns: The CircuitData of the synthesized circuit.
#[pyfunction]
#[pyo3(signature = (mat))]
pub fn py_synth_cnot_depth_line_kms(mat: PyReadonlyArray2<bool>) -> PyResult<PyCircuitData> {
    let num_qubits = mat.as_array().nrows(); // is a quadratic matrix
    let (cx_instructions_rows_m2nw, cx_instructions_rows_nw2id) =
        synth_cnot_lnn_instructions(mat.as_array());

    let instructions = cx_instructions_rows_m2nw
        .into_iter()
        .chain(cx_instructions_rows_nw2id)
        .map(|(ctrl, target)| {
            (
                StandardGate::CX,
                smallvec![],
                smallvec![Qubit(ctrl as u32), Qubit(target as u32)],
            )
        });
    Ok(
        CircuitData::from_standard_gates(num_qubits as u32, instructions, Param::Float(0.0))?
            .into(),
    )
}
