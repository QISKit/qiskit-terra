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

use crate::linear::utils::{
    binary_matmul_inner, calc_inverse_matrix_inner, replace_row_inner, swap_rows_inner,
};
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, Axis, concatenate, s};
use rand::prelude::*;
use rand::rngs::SysRng;
use rand_pcg::Pcg64Mcg;

/// Sample from the quantum Mallows distribution.
fn sample_qmallows(n: usize, rng: &mut Pcg64Mcg) -> (Array1<bool>, Array1<usize>) {
    // Hadamard layer
    let mut had = Array1::from_elem(n, false);

    // Permutation layer
    let mut perm = Array1::from_elem(n, 0);
    let mut inds: Vec<usize> = (0..n).collect();

    for i in 0..n {
        let m = n - i;
        let eps: f64 = 4f64.powi(-(m as i32));
        let r: f64 = rng.random();
        let index: usize = -((r + (1f64 - r) * eps).log2().ceil() as isize) as usize;
        had[i] = index < m;
        let k = if index < m { index } else { 2 * m - index - 1 };
        perm[i] = inds[k];
        inds.remove(k);
    }
    (had, perm)
}

/// Add symmetric random boolean value to off diagonal entries.
fn fill_tril(mut mat: ArrayViewMut2<bool>, rng: &mut Pcg64Mcg, symmetric: bool) {
    let n = mat.shape()[0];
    for i in 0..n {
        for j in 0..i {
            mat[[i, j]] = rng.random();
            if symmetric {
                mat[[j, i]] = mat[[i, j]];
            }
        }
    }
}

/// Invert a lower-triangular matrix with unit diagonal.
fn inverse_tril(mat: ArrayView2<bool>) -> Array2<bool> {
    calc_inverse_matrix_inner(mat, false).unwrap()
}

/// Generate a random Clifford tableau.
///
/// The Clifford is sampled using the method of the paper "Hadamard-free circuits
/// expose the structure of the Clifford group" by S. Bravyi and D. Maslov (2020),
/// `https://arxiv.org/abs/2003.09412`__.
///
/// The function returns a random clifford tableau.
pub fn random_clifford_tableau_inner(num_qubits: usize, seed: Option<u64>) -> Array2<bool> {
    let mut rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::try_from_rng(&mut SysRng).unwrap(),
    };

    let (had, perm) = sample_qmallows(num_qubits, &mut rng);

    let mut gamma1: Array2<bool> = Array2::from_elem((num_qubits, num_qubits), false);
    for i in 0..num_qubits {
        gamma1[[i, i]] = rng.random();
    }
    fill_tril(gamma1.view_mut(), &mut rng, true);

    let mut gamma2: Array2<bool> = Array2::from_elem((num_qubits, num_qubits), false);
    for i in 0..num_qubits {
        gamma2[[i, i]] = rng.random();
    }
    fill_tril(gamma2.view_mut(), &mut rng, true);

    let mut delta1: Array2<bool> = Array2::from_shape_fn((num_qubits, num_qubits), |(i, j)| i == j);
    fill_tril(delta1.view_mut(), &mut rng, false);

    let mut delta2: Array2<bool> = Array2::from_shape_fn((num_qubits, num_qubits), |(i, j)| i == j);
    fill_tril(delta2.view_mut(), &mut rng, false);

    // Compute stabilizer table
    let zero = Array2::from_elem((num_qubits, num_qubits), false);
    let prod1 = binary_matmul_inner(gamma1.view(), delta1.view()).unwrap();
    let prod2 = binary_matmul_inner(gamma2.view(), delta2.view()).unwrap();
    let inv1 = inverse_tril(delta1.view()).t().to_owned();
    let inv2 = inverse_tril(delta2.view()).t().to_owned();

    let table1 = concatenate(
        Axis(0),
        &[
            concatenate(Axis(1), &[delta1.view(), zero.view()])
                .unwrap()
                .view(),
            concatenate(Axis(1), &[prod1.view(), inv1.view()])
                .unwrap()
                .view(),
        ],
    )
    .unwrap();

    let table2 = concatenate(
        Axis(0),
        &[
            concatenate(Axis(1), &[delta2.view(), zero.view()])
                .unwrap()
                .view(),
            concatenate(Axis(1), &[prod2.view(), inv2.view()])
                .unwrap()
                .view(),
        ],
    )
    .unwrap();

    // Compute the full stabilizer tableau

    // The code below is based on the original code in the referenced paper.
    let mut table = Array2::from_elem((2 * num_qubits, 2 * num_qubits), false);

    // Apply qubit permutation
    for i in 0..num_qubits {
        replace_row_inner(table.view_mut(), i, table2.slice(s![perm[i], ..]));
        replace_row_inner(
            table.view_mut(),
            i + num_qubits,
            table2.slice(s![perm[i] + num_qubits, ..]),
        );
    }

    // Apply layer of Hadamards
    for i in 0..num_qubits {
        if had[i] {
            swap_rows_inner(table.view_mut(), i, i + num_qubits);
        }
    }

    // Apply table
    let random_symplectic_mat = binary_matmul_inner(table1.view(), table.view()).unwrap();

    // Generate random phases
    let random_phases: Array2<bool> = Array2::from_shape_fn((2 * num_qubits, 1), |_| rng.random());

    let random_tableau: Array2<bool> = concatenate(
        Axis(1),
        &[random_symplectic_mat.view(), random_phases.view()],
    )
    .unwrap();
    random_tableau
}
