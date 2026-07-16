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

use ndarray::{Array, Array2, ArrayView, ArrayView2, IxDyn};
use ndarray_einsum::*;
use num_complex::{Complex, Complex64};

static LOWERCASE: [u8; 26] = *b"abcdefghijklmnopqrstuvwxyz";
static _UPPERCASE: [u8; 26] = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// Compose the operators given by `gate_unitary` and `overall_unitary`, i.e. apply one to the other
// as specified by the involved qubits given in `qubits` and the `front` parameter
pub fn compose(
    gate_unitary: &ArrayView2<Complex64>,
    overall_unitary: &ArrayView2<Complex64>,
    indices: &[usize],
    front: bool,
) -> Result<Array2<Complex64>, &'static str> {
    let gate_qubits = gate_unitary.shape()[0].ilog2() as usize;

    // Full composition of operators
    if indices.is_empty() {
        if front {
            return Ok(gate_unitary.dot(overall_unitary));
        } else {
            return Ok(overall_unitary.dot(gate_unitary));
        }
    }
    // Compose with other on subsystem
    let num_indices = gate_qubits;
    let shift = if front { gate_qubits } else { 0usize };
    let right_mul = front;

    //Reshape current matrix
    //Note that we must reverse the subsystem dimension order as
    //qubit 0 corresponds to the right-most position in the tensor
    //product, which is the last tensor wire index.
    let tensor = per_qubit_shaped(gate_unitary);
    let mat = per_qubit_shaped(overall_unitary);
    let indices = indices
        .iter()
        .map(|i| num_indices - 1 - i)
        .collect::<Vec<usize>>();
    let num_rows = usize::pow(2, num_indices as u32);

    let res = _einsum_matmul(&tensor, &mat, &indices, shift, right_mul)?
        .as_standard_layout()
        .into_shape_with_order((num_rows, num_rows))
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()
        .to_owned();
    Ok(res)
}

// Reshape an input matrix to (2, 2, ..., 2) depending on its dimensionality
fn per_qubit_shaped<'a>(array: &ArrayView2<'a, Complex<f64>>) -> ArrayView<'a, Complex64, IxDyn> {
    let overall_shape = (0..array.shape()[0].ilog2() as usize)
        .flat_map(|_| [2, 2])
        .collect::<Vec<usize>>();
    array.into_shape_with_order(overall_shape).unwrap()
}

// Determine einsum strings for perform a matrix multiplication on the input matrices
fn _einsum_matmul(
    tensor: &ArrayView<Complex64, IxDyn>,
    mat: &ArrayView<Complex64, IxDyn>,
    indices: &[usize],
    shift: usize,
    right_mul: bool,
) -> Result<Array<Complex64, IxDyn>, &'static str> {
    let rank = tensor.ndim();
    let rank_mat = mat.ndim();
    if !rank_mat.is_multiple_of(2) {
        return Err("Contracted matrix must have an even number of indices.");
    }
    // Get einsum indices for tensor
    let mut indices_tensor = (0..rank).collect::<Vec<usize>>();
    for (j, index) in indices.iter().enumerate() {
        indices_tensor[index + shift] = rank + j;
    }
    // Get einsum indices for mat
    let mat_contract = (rank..rank + indices.len()).rev().collect::<Vec<usize>>();
    let mat_free = indices
        .iter()
        .rev()
        .map(|index| index + shift)
        .collect::<Vec<usize>>();
    let indices_mat = if right_mul {
        [mat_contract, mat_free].concat()
    } else {
        [mat_free, mat_contract].concat()
    };

    let tensor_einsum = unsafe {
        String::from_utf8_unchecked(indices_tensor.iter().map(|c| LOWERCASE[*c]).collect())
    };
    let mat_einsum =
        unsafe { String::from_utf8_unchecked(indices_mat.iter().map(|c| LOWERCASE[*c]).collect()) };

    einsum(
        format!("{tensor_einsum},{mat_einsum}").as_str(),
        &[tensor, mat],
    )
}

fn _einsum_matmul_helper(qubits: &[u32], num_qubits: usize) -> [String; 4] {
    let tens_in: Vec<u8> = LOWERCASE[..num_qubits].to_vec();
    let mut tens_out: Vec<u8> = tens_in.clone();
    let mut mat_l: Vec<u8> = Vec::with_capacity(num_qubits);
    let mut mat_r: Vec<u8> = Vec::with_capacity(num_qubits);
    qubits.iter().rev().enumerate().for_each(|(pos, idx)| {
        mat_r.push(tens_in[num_qubits - 1 - pos]);
        mat_l.push(LOWERCASE[25 - pos]);
        tens_out[num_qubits - 1 - *idx as usize] = LOWERCASE[25 - pos];
    });
    unsafe {
        [
            String::from_utf8_unchecked(mat_l),
            String::from_utf8_unchecked(mat_r),
            String::from_utf8_unchecked(tens_in),
            String::from_utf8_unchecked(tens_out),
        ]
    }
}

fn _einsum_matmul_index(qubits: &[u32], num_qubits: usize) -> String {
    assert!(num_qubits > 26, "Can't compute unitary of > 26 qubits");

    let tens_r = unsafe { String::from_utf8_unchecked(_UPPERCASE[..num_qubits].to_vec()) };
    let [mat_l, mat_r, tens_lin, tens_lout] = _einsum_matmul_helper(qubits, num_qubits);
    format!("{mat_l}{mat_r}, {tens_lin}{tens_r}->{tens_lout}{tens_r}")
}

#[inline]
pub fn matmul_1q(left: &ArrayView2<Complex64>, right: &ArrayView2<Complex64>) -> Array2<Complex64> {
    let mut out = Array2::zeros((2, 2));
    out[[0, 0]] = left[[0, 0]] * right[[0, 0]] + left[[0, 1]] * right[[1, 0]];
    out[[0, 1]] = left[[0, 0]] * right[[0, 1]] + left[[0, 1]] * right[[1, 1]];
    out[[1, 0]] = left[[1, 0]] * right[[0, 0]] + left[[1, 1]] * right[[1, 0]];
    out[[1, 1]] = left[[1, 0]] * right[[0, 1]] + left[[1, 1]] * right[[1, 1]];
    out
}

#[inline]
fn _ind(i: usize, reversed: bool) -> usize {
    if reversed {
        // reverse the first two bits
        ((i & 1) << 1) + ((i & 2) >> 1)
    } else {
        i
    }
}

#[inline]
pub fn matmul_2q(
    left: &ArrayView2<Complex64>,
    right: &ArrayView2<Complex64>,
    indices: &[usize],
) -> Array2<Complex64> {
    let mut out = Array2::zeros((4, 4));

    let rev = indices[0] == 1;
    for i in 0..4usize {
        for j in 0..4usize {
            for k in 0..4usize {
                out[[i, j]] += left[[_ind(i, rev), _ind(k, rev)]] * right[[k, j]];
            }
        }
    }
    out
}
