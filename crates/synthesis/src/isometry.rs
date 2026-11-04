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

//! Implements the column-by-column isometry decomposition of Iten et al.,
//! "Quantum Circuits for Isometries" (arXiv:1501.06911), Section IV.C,
//! Theorem 2 and Fig. 1-2, which gives the sequence of gates {G_k} that "disentangle"
//! (uncompute) the isometry to the first 2^m columns of the identity.

use std::iter;
use std::ops::BitAnd;

use approx::abs_diff_eq;
use hashbrown::HashSet;
use itertools::Itertools;
use nalgebra::Matrix2;
use ndarray::prelude::*;
use num_complex::{Complex64, ComplexFloat};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::gate_matrix::ONE_QUBIT_IDENTITY;
use qiskit_circuit::operations::Param;
use qiskit_util::complex::C_ZERO;

use crate::diagonal::diagonal_gate_circuit;
use crate::mcg_up_to_diagonal::mcg_up_to_diagonal_inner;
use crate::qsd::append;
use crate::uc_gate::dec_ucg_inner;

#[inline(always)]
fn l2_norm(vec: &[Complex64]) -> f64 {
    vec.iter()
        .fold(0., |acc, elem| acc + elem.norm_sqr())
        .sqrt()
}

#[inline(always)]
fn bin_to_int(bin: &[u8]) -> usize {
    bin.iter()
        .fold(0_usize, |acc, digit| (acc << 1) + *digit as usize)
}

#[inline(always)]
fn k_s(k: usize, s: usize) -> usize {
    if k == 0 {
        0
    } else {
        let filter = 1 << s;
        k.bitand(filter) >> s
    }
}

#[inline(always)]
fn a(k: usize, s: usize) -> usize {
    k / 2_usize.pow(s as u32)
}

#[inline(always)]
fn b(k: usize, s: usize) -> usize {
    k - (a(k, s) * 2_usize.pow(s as u32))
}

fn binary_rep_bit(k: usize, n: usize, i: usize) -> u8 {
    ((k >> (n - 1 - i)) & 1) as u8
}

fn array2_to_matrix2(a: &Array2<Complex64>) -> Matrix2<Complex64> {
    Matrix2::new(a[[0, 0]], a[[0, 1]], a[[1, 0]], a[[1, 1]])
}

fn find_squs_for_disentangling(
    v: ArrayView2<Complex64>,
    k: usize,
    s: usize,
    epsilon: f64,
    n: usize,
) -> Vec<Array2<Complex64>> {
    let k_prime = 0;
    let i_start = if b(k, s + 1) == 0 {
        a(k, s + 1)
    } else {
        a(k, s + 1) + 1
    };
    let mut output: Vec<Array2<Complex64>> = (0..i_start).map(|_| Array2::eye(2)).collect();
    let mut squs: Vec<Array2<Complex64>> = (i_start..2_usize.pow((n - s - 1) as u32))
        .map(|i| {
            reverse_qubit_state(
                &[
                    v[[2 * i * 2_usize.pow(s as u32) + b(k, s), k_prime]],
                    v[[(2 * i + 1) * 2_usize.pow(s as u32) + b(k, s), k_prime]],
                ],
                k_s(k, s),
                epsilon,
            )
        })
        .collect();
    output.append(&mut squs);
    output
}

fn ucg_is_identity_up_to_global_phase(
    single_qubit_gates: &[Array2<Complex64>],
    epsilon: f64,
) -> bool {
    let global_phase: Complex64 = if single_qubit_gates[0][[0, 0]].abs() >= epsilon {
        single_qubit_gates[0][[0, 0]].finv()
    } else {
        return false;
    };
    for gate in single_qubit_gates {
        if !abs_diff_eq!(
            gate.mapv(|x| x * global_phase),
            aview2(&ONE_QUBIT_IDENTITY),
            epsilon = 1e-8
        ) {
            return false;
        }
    }
    true
}

/// Appends a uniformly controlled gate, targeting `target_qubit` and controlled by
/// `control_qubits`, to `circuit`. Returns the trailing diagonal left over, for the caller
/// to fold into whatever comes next.
fn append_ucg_up_to_diagonal(
    circuit: &mut CircuitData,
    single_qubit_gates: &[Array2<Complex64>],
    target_qubit: Qubit,
    control_qubits: &[Qubit],
) -> Result<Vec<Complex64>, CircuitDataError> {
    let gates: Vec<Matrix2<Complex64>> = single_qubit_gates.iter().map(array2_to_matrix2).collect();
    let num_qubits = control_qubits.len() as u32 + 1;
    let (sub_circuit, diag) = dec_ucg_inner(gates, num_qubits, true, true)?;
    let qubit_map: Vec<Qubit> = iter::once(target_qubit)
        .chain(control_qubits.iter().copied())
        .collect();
    append(circuit, sub_circuit, &qubit_map)?;
    Ok(diag)
}

/// Appends a multi-controlled single-qubit `gate`, targeting `target_qubit` and controlled by
/// `control_qubits`, to `circuit`. Returns the trailing diagonal left over, for the caller
/// to fold into whatever comes next.
fn append_mcg_up_to_diagonal(
    circuit: &mut CircuitData,
    gate: &Array2<Complex64>,
    target_qubit: Qubit,
    control_qubits: &[Qubit],
) -> Result<Vec<Complex64>, CircuitDataError> {
    let (sub_circuit, diag) =
        mcg_up_to_diagonal_inner(array2_to_matrix2(gate), control_qubits.len() as u32)?;
    let qubit_map: Vec<Qubit> = iter::once(target_qubit)
        .chain(control_qubits.iter().copied())
        .collect();
    append(circuit, sub_circuit, &qubit_map)?;
    Ok(diag)
}

fn merge_ucgate_and_diag(
    single_qubit_gates: &[Array2<Complex64>],
    diag: &[Complex64],
) -> Vec<Array2<Complex64>> {
    single_qubit_gates
        .iter()
        .enumerate()
        .map(|(i, gate)| aview2(&[[diag[2 * i], C_ZERO], [C_ZERO, diag[2 * i + 1]]]).dot(gate))
        .collect()
}

fn reverse_qubit_state(
    state: &[Complex64; 2],
    basis_state: usize,
    epsilon: f64,
) -> Array2<Complex64> {
    let r = l2_norm(state);
    let r_inv = 1. / r;
    if r < epsilon {
        Array2::eye(2)
    } else if basis_state == 0 {
        array![
            [state[0].conj() * r_inv, state[1].conj() * r_inv],
            [-state[1] * r_inv, state[0] * r_inv],
        ]
    } else {
        array![
            [-state[1] * r_inv, state[0] * r_inv],
            [state[0].conj() * r_inv, state[1].conj() * r_inv],
        ]
    }
}

/// Helper for `apply_multi_controlled_gate`. Constructs the two basis-state indices the gate
/// acts on for a specific state `state_free` of the qubits that are neither controls nor target.
fn construct_basis_states(
    state_free: &[u8],
    control_set: &HashSet<usize>,
    target_label: usize,
) -> [usize; 2] {
    let size = state_free.len() + control_set.len() + 1;
    let mut e1: usize = 0;
    let mut e2: usize = 0;
    let mut j = 0;
    for i in 0..size {
        e1 <<= 1;
        e2 <<= 1;
        if control_set.contains(&i) {
            e1 += 1;
            e2 += 1;
        } else if i == target_label {
            e2 += 1;
        } else {
            e1 += state_free[j] as usize;
            e2 += state_free[j] as usize;
            j += 1
        }
    }
    [e1, e2]
}

fn apply_ucg(
    mut m: Array2<Complex64>,
    k: usize,
    single_qubit_gates: &[Array2<Complex64>],
) -> Array2<Complex64> {
    let shape = m.shape();
    let num_qubits = shape[0].ilog2();
    let num_col = shape[1];
    let spacing: usize = 2_usize.pow(num_qubits - k as u32 - 1);
    for j in 0..2_usize.pow(num_qubits - 1) {
        let i = (j / spacing) * spacing + j;
        let gate_index = i / (2_usize.pow(num_qubits - k as u32));
        for col in 0..num_col {
            let gate = single_qubit_gates[gate_index].view();
            let a = m[[i, col]];
            let b = m[[i + spacing, col]];
            m[[i, col]] = gate[[0, 0]] * a + gate[[0, 1]] * b;
            m[[i + spacing, col]] = gate[[1, 0]] * a + gate[[1, 1]] * b;
        }
    }
    m
}

fn apply_diagonal_gate(
    mut m: Array2<Complex64>,
    action_qubit_labels: &[usize],
    diag: &[Complex64],
) -> Array2<Complex64> {
    let shape = m.shape();
    let num_qubits = shape[0].ilog2();
    let num_col = shape[1];
    for state in std::iter::repeat_n([0_u8, 1_u8], num_qubits as usize).multi_cartesian_product() {
        let diag_index = action_qubit_labels
            .iter()
            .fold(0_usize, |acc, i| (acc << 1) + state[*i] as usize);
        let i = bin_to_int(&state);
        for j in 0..num_col {
            m[[i, j]] = diag[diag_index] * m[[i, j]]
        }
    }
    m
}

fn apply_diagonal_gate_to_diag(
    mut m_diagonal: Vec<Complex64>,
    action_qubit_labels: &[usize],
    diag: &[Complex64],
    num_qubits: usize,
) -> Vec<Complex64> {
    if m_diagonal.is_empty() {
        return m_diagonal;
    }
    for state in std::iter::repeat_n([0_u8, 1_u8], num_qubits)
        .multi_cartesian_product()
        .take(m_diagonal.len())
    {
        let diag_index = action_qubit_labels
            .iter()
            .fold(0_usize, |acc, i| (acc << 1) + state[*i] as usize);
        let i = bin_to_int(&state);
        m_diagonal[i] *= diag[diag_index]
    }
    m_diagonal
}

fn diag_is_identity_up_to_global_phase(diag: &[Complex64], epsilon: f64) -> bool {
    let global_phase: Complex64 = if diag[0].abs() >= epsilon {
        diag[0].finv()
    } else {
        return false;
    };
    for &d in diag {
        if (global_phase * d - 1.0).abs() >= epsilon {
            return false;
        }
    }
    true
}

fn apply_multi_controlled_gate(
    mut m: Array2<Complex64>,
    control_labels: &[usize],
    target_label: usize,
    gate: ArrayView2<Complex64>,
) -> Array2<Complex64> {
    let shape = m.shape();
    let num_qubits = shape[0].ilog2();
    let num_col = shape[1];
    let free_qubits = num_qubits as usize - control_labels.len() - 1;
    let control_set: HashSet<usize> = control_labels.iter().copied().collect();
    if free_qubits == 0 {
        let [e1, e2] = construct_basis_states(&[], &control_set, target_label);
        for i in 0..num_col {
            let temp: Vec<_> = gate
                .dot(&aview2(&[[m[[e1, i]]], [m[[e2, i]]]]))
                .into_iter()
                .take(2)
                .collect();
            m[[e1, i]] = temp[0];
            m[[e2, i]] = temp[1];
        }
        return m;
    }
    for state_free in std::iter::repeat_n([0_u8, 1_u8], free_qubits).multi_cartesian_product() {
        let [e1, e2] = construct_basis_states(&state_free, &control_set, target_label);
        for i in 0..num_col {
            let temp: Vec<_> = gate
                .dot(&aview2(&[[m[[e1, i]]], [m[[e2, i]]]]))
                .into_iter()
                .take(2)
                .collect();
            m[[e1, i]] = temp[0];
            m[[e2, i]] = temp[1];
        }
    }
    m
}

/// Zeroes out qubit `s` of column `k`, appending whatever gates that takes to `circuit`.
/// Returns the updated isometry and the diagonal correction accumulated so far.
fn disentangle(
    circuit: &mut CircuitData,
    mut v: Array2<Complex64>,
    mut diag: Vec<Complex64>,
    k: usize,
    s: usize,
    n: usize,
    epsilon: f64,
) -> Result<(Array2<Complex64>, Vec<Complex64>), CircuitDataError> {
    let k_prime = 0;
    let pow2s = 2_usize.pow(s as u32);
    let index1 = 2 * a(k, s + 1) * pow2s + b(k, s + 1);
    let index2 = (2 * a(k, s + 1) + 1) * pow2s + b(k, s + 1);
    let target_qubit = Qubit(s as u32);

    if k_s(k, s) == 0 && b(k, s + 1) != 0 && v[[index2, k_prime]].abs() > epsilon {
        let gate = reverse_qubit_state(&[v[[index1, k_prime]], v[[index2, k_prime]]], 0, epsilon);
        let control_labels: Vec<usize> = (0..n)
            .filter(|&i| binary_rep_bit(k, n, i) == 1 && i != n - s - 1)
            .collect();
        let control_qubits: Vec<Qubit> = control_labels
            .iter()
            .rev()
            .map(|&label| Qubit((n - label - 1) as u32))
            .collect();

        let diagonal_mcg =
            append_mcg_up_to_diagonal(circuit, &gate, target_qubit, &control_qubits)?;
        let control_labels_and_target: Vec<usize> =
            control_labels.iter().copied().chain([n - s - 1]).collect();

        v = apply_multi_controlled_gate(v, &control_labels, n - s - 1, gate.view());
        let diag_mcg_inverse: Vec<Complex64> = diagonal_mcg.iter().map(|z| z.conj()).collect();
        v = apply_diagonal_gate(v, &control_labels_and_target, &diag_mcg_inverse);
        diag = apply_diagonal_gate_to_diag(diag, &control_labels_and_target, &diag_mcg_inverse, n);
    }

    let single_qubit_gates = find_squs_for_disentangling(v.view(), k, s, epsilon, n);
    if !ucg_is_identity_up_to_global_phase(&single_qubit_gates, epsilon) {
        let target_label = n - s - 1;
        let control_qubits: Vec<Qubit> = ((s + 1)..n).map(|q| Qubit(q as u32)).collect();
        let control_labels: Vec<usize> = (0..target_label).collect();

        let diagonal_ucg =
            append_ucg_up_to_diagonal(circuit, &single_qubit_gates, target_qubit, &control_qubits)?;
        let diagonal_ucg_inverse: Vec<Complex64> = diagonal_ucg.iter().map(|z| z.conj()).collect();
        let merged_gates = merge_ucgate_and_diag(&single_qubit_gates, &diagonal_ucg_inverse);

        v = apply_ucg(v, control_labels.len(), &merged_gates);
        let control_labels_and_target: Vec<usize> =
            control_labels.into_iter().chain([target_label]).collect();
        diag =
            apply_diagonal_gate_to_diag(diag, &control_labels_and_target, &diagonal_ucg_inverse, n);
    }

    Ok((v, diag))
}

fn decompose_column(
    circuit: &mut CircuitData,
    mut v: Array2<Complex64>,
    mut diag: Vec<Complex64>,
    column_index: usize,
    n: usize,
    epsilon: f64,
) -> Result<(Array2<Complex64>, Vec<Complex64>), CircuitDataError> {
    for s in 0..n {
        (v, diag) = disentangle(circuit, v, diag, column_index, s, n, epsilon)?;
    }
    Ok((v, diag))
}
/// Builds the circuit that maps `iso`'s columns onto the first `2^m` basis states, column by
/// column, then widens it with `num_ancillas_zero + num_ancillas_dirty` unused ancilla qubits.
pub(crate) fn synth_isometry_inner(
    iso: ArrayView2<Complex64>,
    num_ancillas_zero: usize,
    num_ancillas_dirty: usize,
    epsilon: f64,
) -> Result<CircuitData, CircuitDataError> {
    let n = iso.shape()[0].ilog2() as usize;
    let m = iso.shape()[1].ilog2() as usize;
    let num_qubits = n + num_ancillas_zero + num_ancillas_dirty;

    let mut circuit = CircuitData::with_capacity(num_qubits as u32, 0, 0, Param::Float(0.0))?;
    let mut remaining_isometry = iso.to_owned();
    let mut diag: Vec<Complex64> = Vec::with_capacity(1 << m);

    for column_index in 0..(1usize << m) {
        let (v, d) = decompose_column(
            &mut circuit,
            remaining_isometry,
            diag,
            column_index,
            n,
            epsilon,
        )?;
        remaining_isometry = v;
        diag = d;
        diag.push(remaining_isometry[[column_index, 0]]);
        remaining_isometry = remaining_isometry.slice(s![.., 1..]).to_owned();
    }

    if diag.len() > 1 && !diag_is_identity_up_to_global_phase(&diag, epsilon) {
        // phase(conj(z)) = -phase(z) = -z.arg().
        let mut phases: Vec<f64> = diag.iter().map(|z| -z.arg()).collect();
        let diag_circuit = diagonal_gate_circuit(&mut phases, m)?;
        let qubit_map: Vec<Qubit> = (0..m as u32).map(Qubit).collect();
        append(&mut circuit, diag_circuit, &qubit_map)?;
    }

    Ok(circuit)
}

/// Validates that `iso`'s shape is a valid isometry (power-of-2 rows and columns, with at
/// least as many rows as columns), then returns the circuit decomposing it, named
/// `"isometry_to_uncompute"`.
#[pyfunction]
pub fn synth_isometry(
    py: Python,
    iso: PyReadonlyArray2<Complex64>,
    num_ancillas_zero: usize,
    num_ancillas_dirty: usize,
    epsilon: f64,
) -> PyResult<Py<PyAny>> {
    let iso_array = iso.as_array();
    let (rows, cols) = (iso_array.shape()[0], iso_array.shape()[1]);
    if !rows.is_power_of_two() {
        return Err(PyValueError::new_err(format!(
            "the number of rows ({rows}) is not a power of 2"
        )));
    }
    if !cols.is_power_of_two() {
        return Err(PyValueError::new_err(format!(
            "the number of columns ({cols}) is not a power of 2"
        )));
    }
    if cols > rows {
        return Err(PyValueError::new_err(format!(
            "the isometry has more columns ({cols}) than rows ({rows})"
        )));
    }
    let circuit = synth_isometry_inner(
        iso.as_array(),
        num_ancillas_zero,
        num_ancillas_dirty,
        epsilon,
    )
    .map_err(PyErr::from)?;
    let qc = circuit.into_py_quantum_circuit(py)?;
    qc.setattr("name", "isometry_to_uncompute")?;
    Ok(qc.unbind())
}

pub fn isometry(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synth_isometry, m)?)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::matrix::sim::sim_unitary_circuit;
    use qiskit_circuit::operations::Operation;
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    use rand_pcg::Pcg64Mcg;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[inline(always)]
    fn random_complex(rng: &mut Pcg64Mcg) -> Complex64 {
        Complex64::new(rng.sample(StandardNormal), rng.sample(StandardNormal))
    }

    /// A `rows x cols` matrix with orthonormal columns, built via Gram-Schmidt on random
    /// complex vectors.
    fn random_isometry(rows: usize, cols: usize, rng: &mut Pcg64Mcg) -> Array2<Complex64> {
        let mut m: Array2<Complex64> = Array2::zeros((rows, cols));
        for col in 0..cols {
            loop {
                let mut v: Vec<Complex64> = (0..rows).map(|_| random_complex(rng)).collect();
                for prev in 0..col {
                    let proj: Complex64 = (0..rows).map(|i| m[[i, prev]].conj() * v[i]).sum();
                    for (i, vi) in v.iter_mut().enumerate() {
                        *vi -= proj * m[[i, prev]];
                    }
                }
                let norm = l2_norm(&v);
                if norm > 1e-6 {
                    for (i, vi) in v.into_iter().enumerate() {
                        m[[i, col]] = vi / norm;
                    }
                    break;
                }
            }
        }
        m
    }

    /// The `rows x cols` top-left block of the `rows x rows` identity, the expected target of
    /// `synth_isometry_inner`.
    fn basis_columns(rows: usize, cols: usize) -> Array2<Complex64> {
        let mut out = Array2::zeros((rows, cols));
        for i in 0..cols {
            out[[i, i]] = Complex64::ONE;
        }
        out
    }

    fn id2() -> Array2<Complex64> {
        array![[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]
    }

    fn x_gate() -> Array2<Complex64> {
        array![[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]]
    }

    fn hadamard() -> Array2<Complex64> {
        let h = c(1.0 / 2.0_f64.sqrt(), 0.0);
        array![[h, h], [h, -h]]
    }

    // ------------------------------------------------------------------
    // reverse_qubit_state
    // ------------------------------------------------------------------

    #[test]
    fn test_reverse_qubit_state_degenerate_returns_identity() {
        // r < epsilon: the state is already (numerically) zero, nothing to rotate.
        let gate = reverse_qubit_state(&[c(0.0, 0.0), c(0.0, 0.0)], 0, 1e-10);
        assert!(abs_diff_eq!(gate, Array2::eye(2), epsilon = 1e-12));
    }

    #[test]
    fn test_reverse_qubit_state_basis_zero_eliminates_second_component() {
        let state = [c(0.6, 0.0), c(0.8, 0.0)];
        let gate = reverse_qubit_state(&state, 0, 1e-10);
        let rotated = gate.dot(&array![[state[0]], [state[1]]]);
        // basis_state == 0: the state is rotated onto |0>, so its second component vanishes.
        assert!(abs_diff_eq!(rotated[[1, 0]], c(0.0, 0.0), epsilon = 1e-10));
        assert!(rotated[[0, 0]].abs() > 0.99);
    }

    #[test]
    fn test_reverse_qubit_state_basis_one_eliminates_first_component() {
        let state = [c(0.6, 0.0), c(0.8, 0.0)];
        let gate = reverse_qubit_state(&state, 1, 1e-10);
        let rotated = gate.dot(&array![[state[0]], [state[1]]]);
        // basis_state == 1: the state is rotated onto |1>, so its first component vanishes.
        assert!(abs_diff_eq!(rotated[[0, 0]], c(0.0, 0.0), epsilon = 1e-10));
        assert!(rotated[[1, 0]].abs() > 0.99);
    }

    // ------------------------------------------------------------------
    // apply_ucg / apply_multi_controlled_gate
    // ------------------------------------------------------------------

    #[test]
    fn test_apply_ucg_single_qubit_no_controls() {
        // num_qubits = 1, k = 0 (no controls): a plain single-qubit gate application.
        let m: Array2<Complex64> = array![[c(1.0, 0.0)], [c(0.0, 0.0)]]; // |0>
        let out = apply_ucg(m, 0, &[hadamard()]);
        let h = c(1.0 / 2.0_f64.sqrt(), 0.0);
        assert!(abs_diff_eq!(out, array![[h], [h]], epsilon = 1e-12));
    }

    #[test]
    fn test_apply_ucg_one_control() {
        // num_qubits = 2, k = 1: label 0 controls, label 1 is the target. X only fires when
        // the control (label 0) is 1, flipping row "10"=2 into row "11"=3.
        let mut m: Array2<Complex64> = Array2::zeros((4, 1));
        m[[2, 0]] = c(1.0, 0.0);
        let out = apply_ucg(m, 1, &[id2(), x_gate()]);
        let mut expected: Array2<Complex64> = Array2::zeros((4, 1));
        expected[[3, 0]] = c(1.0, 0.0);
        assert!(abs_diff_eq!(out, expected, epsilon = 1e-12));
    }

    #[test]
    fn test_apply_multi_controlled_gate_no_free_qubits() {
        // num_qubits = 2, control_labels = [0], target_label = 1: no free qubits remain
        // (2 - 1 - 1 == 0), exercising the dedicated fast path.
        let mut m: Array2<Complex64> = Array2::zeros((4, 1));
        m[[2, 0]] = c(1.0, 0.0); // row "10": control=1, target=0
        let out = apply_multi_controlled_gate(m, &[0], 1, x_gate().view());
        let mut expected: Array2<Complex64> = Array2::zeros((4, 1));
        expected[[3, 0]] = c(1.0, 0.0); // row "11"
        assert!(abs_diff_eq!(out, expected, epsilon = 1e-12));
    }

    #[test]
    fn test_apply_multi_controlled_gate_with_free_qubits() {
        // num_qubits = 2, control_labels = [] (free_qubits = 1): the gate is applied to the
        // target unconditionally, for every value of the one free qubit.
        let mut m: Array2<Complex64> = Array2::zeros((4, 1));
        m[[1, 0]] = c(1.0, 0.0); // row "01": target(label0)=0, free(label1)=1
        let out = apply_multi_controlled_gate(m, &[], 0, x_gate().view());
        let mut expected: Array2<Complex64> = Array2::zeros((4, 1));
        expected[[3, 0]] = c(1.0, 0.0); // row "11"
        assert!(abs_diff_eq!(out, expected, epsilon = 1e-12));
    }

    // ------------------------------------------------------------------
    // apply_diagonal_gate / apply_diagonal_gate_to_diag / diag_is_identity_up_to_global_phase
    // ------------------------------------------------------------------

    #[test]
    fn test_apply_diagonal_gate() {
        // num_qubits = 2, action_qubit_labels = [0]: rows with label-0 bit 0 ("00","01") get
        // d0, rows with label-0 bit 1 ("10","11") get d1.
        let d0 = c(2.0, 0.0);
        let d1 = c(3.0, 0.0);
        let m: Array2<Complex64> = Array2::eye(4);
        let out = apply_diagonal_gate(m, &[0], &[d0, d1]);
        assert!(abs_diff_eq!(out[[0, 0]], d0, epsilon = 1e-12));
        assert!(abs_diff_eq!(out[[1, 1]], d0, epsilon = 1e-12));
        assert!(abs_diff_eq!(out[[2, 2]], d1, epsilon = 1e-12));
        assert!(abs_diff_eq!(out[[3, 3]], d1, epsilon = 1e-12));
        assert!(abs_diff_eq!(out[[0, 1]], c(0.0, 0.0), epsilon = 1e-12));
    }

    #[test]
    fn test_apply_diagonal_gate_to_diag() {
        let d0 = c(2.0, 0.0);
        let d1 = c(3.0, 0.0);
        // Only 3 of the 4 basis states are filled in, matching a partial diag mid-loop.
        let m_diagonal = vec![c(1.0, 0.0), c(0.0, 1.0), c(1.0, 1.0)];
        let out = apply_diagonal_gate_to_diag(m_diagonal, &[0], &[d0, d1], 2);
        assert!(abs_diff_eq!(out[0], c(2.0, 0.0), epsilon = 1e-12));
        assert!(abs_diff_eq!(out[1], c(0.0, 2.0), epsilon = 1e-12));
        assert!(abs_diff_eq!(out[2], c(3.0, 3.0), epsilon = 1e-12));
    }

    #[test]
    fn test_diag_is_identity_up_to_global_phase() {
        assert!(diag_is_identity_up_to_global_phase(
            &[c(1.0, 0.0); 4],
            1e-10
        ));
        // Same phase on every entry, scaled by a global factor, is still "identity".
        let phase = c(0.0, 1.0);
        assert!(diag_is_identity_up_to_global_phase(
            &[phase, phase, phase],
            1e-10
        ));
        // Different phases: not the identity up to a single global phase.
        assert!(!diag_is_identity_up_to_global_phase(
            &[c(1.0, 0.0), c(-1.0, 0.0)],
            1e-10
        ));
        // First entry below epsilon in magnitude: the global phase can't be defined.
        assert!(!diag_is_identity_up_to_global_phase(
            &[c(0.0, 0.0), c(1.0, 0.0)],
            1e-10
        ));
    }

    // ------------------------------------------------------------------
    // ucg_is_identity_up_to_global_phase / merge_ucgate_and_diag
    // ------------------------------------------------------------------

    #[test]
    fn test_ucg_is_identity_up_to_global_phase() {
        assert!(ucg_is_identity_up_to_global_phase(&[id2()], 1e-8));
        assert!(!ucg_is_identity_up_to_global_phase(
            &[id2(), x_gate()],
            1e-8
        ));
        // gate[0][[0,0]] is below epsilon: the global phase is undefined, so this is never
        // treated as the identity (even though x_gate() alone, up to a basis swap, is unitary).
        assert!(!ucg_is_identity_up_to_global_phase(&[x_gate()], 1e-8));
    }

    #[test]
    fn test_merge_ucgate_and_diag() {
        let d0 = c(2.0, 0.0);
        let d1 = c(0.0, 1.0);
        let merged = merge_ucgate_and_diag(&[id2()], &[d0, d1]);
        assert!(abs_diff_eq!(
            merged[0],
            array![[d0, c(0.0, 0.0)], [c(0.0, 0.0), d1]],
            epsilon = 1e-12
        ));
    }

    // ------------------------------------------------------------------
    // append_ucg_up_to_diagonal / append_mcg_up_to_diagonal
    // ------------------------------------------------------------------

    #[test]
    fn test_append_ucg_up_to_diagonal_matches_direct_decomposition() {
        // Folding the returned diagonal back on top must reproduce what `dec_ucg_inner`
        // computes directly with `up_to_diagonal = false`, for the same gates.
        let gates = vec![id2(), x_gate(), hadamard(), id2()];
        let mut circuit = CircuitData::with_capacity(3, 0, 0, Param::Float(0.0)).unwrap();
        let control_qubits = [Qubit(1), Qubit(2)];
        let diag =
            append_ucg_up_to_diagonal(&mut circuit, &gates, Qubit(0), &control_qubits).unwrap();

        let unitary = sim_unitary_circuit(&circuit).unwrap();
        // diag spans all 3 sub-circuit qubits (target + 2 controls), i.e. 2^3 = 8 entries,
        // not just the 2^2 = 4 entries of the 2 "logical" control/gate-selector qubits.
        let dim = diag.len();
        let mut diag_mat: Array2<Complex64> = Array2::zeros((dim, dim));
        for (i, d) in diag.iter().enumerate() {
            diag_mat[[i, i]] = *d;
        }
        let reconstructed = diag_mat.dot(&unitary);

        let matrix_gates: Vec<Matrix2<Complex64>> = gates.iter().map(array2_to_matrix2).collect();
        let (full_circuit, _) = dec_ucg_inner(matrix_gates, 3, false, true).unwrap();
        let full_unitary = sim_unitary_circuit(&full_circuit).unwrap();

        assert!(abs_diff_eq!(reconstructed, full_unitary, epsilon = 1e-10));
    }

    #[test]
    fn test_append_mcg_up_to_diagonal_matches_direct_decomposition() {
        // Ground truth: a 2-controlled Hadamard is a UCGate with identity everywhere except
        // the "both controls = 1" slot. Rebuild it directly with `up_to_diagonal = false` to
        // get the exact target unitary.
        let mut circuit = CircuitData::with_capacity(3, 0, 0, Param::Float(0.0)).unwrap();
        let control_qubits = [Qubit(1), Qubit(2)];
        let diag = append_mcg_up_to_diagonal(&mut circuit, &hadamard(), Qubit(0), &control_qubits)
            .unwrap();

        let unitary = sim_unitary_circuit(&circuit).unwrap();
        // diag spans all 3 sub-circuit qubits (target + 2 controls), i.e. 2^3 = 8 entries.
        let dim = diag.len();
        let mut diag_mat: Array2<Complex64> = Array2::zeros((dim, dim));
        for (i, d) in diag.iter().enumerate() {
            diag_mat[[i, i]] = *d;
        }
        let reconstructed = diag_mat.dot(&unitary);

        let matrix_gates = vec![
            Matrix2::identity(),
            Matrix2::identity(),
            Matrix2::identity(),
            array2_to_matrix2(&hadamard()),
        ];
        let (full_circuit, _) = dec_ucg_inner(matrix_gates, 3, false, true).unwrap();
        let full_unitary = sim_unitary_circuit(&full_circuit).unwrap();

        assert!(abs_diff_eq!(reconstructed, full_unitary, epsilon = 1e-10));
    }

    // ------------------------------------------------------------------
    // synth_isometry_inner (end-to-end)
    // ------------------------------------------------------------------

    #[test]
    fn test_synth_isometry_inner_identity_produces_empty_circuit() {
        // An isometry already equal to (a block of) the identity needs no gates at all.
        for n in 1..=3_usize {
            let iso: Array2<Complex64> = Array2::eye(1 << n);
            let circuit = synth_isometry_inner(iso.view(), 0, 0, 1e-10).unwrap();
            assert_eq!(circuit.num_qubits(), n);
            assert_eq!(
                circuit.data().len(),
                0,
                "identity isometry for n={n} should need no gates"
            );
        }
    }

    #[test]
    fn test_synth_isometry_inner_state_prep_uniform_superposition() {
        // m = 0 (state preparation): a 2-qubit uniform superposition, unlike the identity
        // case above, needs real disentangling gates.
        let half = c(0.5, 0.0);
        let iso: Array2<Complex64> = array![[half], [half], [half], [half]];
        let circuit = synth_isometry_inner(iso.view(), 0, 0, 1e-10).unwrap();
        assert!(!circuit.data().is_empty());
        let unitary = sim_unitary_circuit(&circuit).unwrap();
        let reconstructed = unitary.dot(&iso);
        assert!(abs_diff_eq!(
            reconstructed,
            basis_columns(4, 1),
            epsilon = 1e-8
        ));
    }

    #[test]
    fn test_synth_isometry_inner_full_unitary() {
        // m = n = 2 (a full unitary): H tensored onto the more-significant qubit (label 0).
        let h = c(1.0 / 2.0_f64.sqrt(), 0.0);
        let z = c(0.0, 0.0);
        let iso: Array2<Complex64> =
            array![[h, z, h, z], [z, h, z, h], [h, z, -h, z], [z, h, z, -h],];
        let circuit = synth_isometry_inner(iso.view(), 0, 0, 1e-10).unwrap();
        let unitary = sim_unitary_circuit(&circuit).unwrap();
        let reconstructed = unitary.dot(&iso);
        assert!(abs_diff_eq!(reconstructed, Array2::eye(4), epsilon = 1e-8));
    }

    #[test]
    fn test_synth_isometry_inner_random_isometries() {
        // Property test: for an arbitrary isometry, the circuit must satisfy
        // `G @ iso == [I_cols; 0]`, across a spread of m-to-n shapes.
        let mut rng = Pcg64Mcg::seed_from_u64(2024);
        let cases: [(usize, usize); 8] = [
            (2, 1),
            (4, 1),
            (4, 2),
            (4, 4),
            (8, 1),
            (8, 2),
            (8, 4),
            (8, 8),
        ];
        for (rows, cols) in cases {
            let iso = random_isometry(rows, cols, &mut rng);
            let n = rows.ilog2() as usize;
            let circuit = synth_isometry_inner(iso.view(), 0, 0, 1e-10).unwrap();
            assert_eq!(circuit.num_qubits(), n);
            let unitary = sim_unitary_circuit(&circuit).unwrap();
            let reconstructed = unitary.dot(&iso);
            assert!(
                abs_diff_eq!(reconstructed, basis_columns(rows, cols), epsilon = 1e-7),
                "failed for {rows}->{cols} isometry"
            );
        }
    }

    #[test]
    fn test_synth_isometry_inner_ancillas_are_idle() {
        // Ancillas currently only widen `num_qubits`; they must not change the gates
        // themselves or which (sub-n) qubits they act on.
        let mut rng = Pcg64Mcg::seed_from_u64(7);
        let iso = random_isometry(4, 2, &mut rng);
        let n = 2;

        fn signature(circuit: &CircuitData) -> Vec<(String, Vec<u32>)> {
            circuit
                .data()
                .iter()
                .map(|inst| {
                    let qargs = circuit
                        .get_qargs(inst.qubits)
                        .iter()
                        .map(|q| q.index() as u32)
                        .collect();
                    (inst.op.name().to_string(), qargs)
                })
                .collect()
        }

        let baseline = synth_isometry_inner(iso.view(), 0, 0, 1e-10).unwrap();
        let baseline_signature = signature(&baseline);

        for (zero, dirty) in [(2, 0), (0, 1), (1, 1)] {
            let circuit = synth_isometry_inner(iso.view(), zero, dirty, 1e-10).unwrap();
            assert_eq!(circuit.num_qubits(), n + zero + dirty);
            assert_eq!(
                signature(&circuit),
                baseline_signature,
                "ancillas=({zero},{dirty})"
            );
            for inst in circuit.data() {
                for q in circuit.get_qargs(inst.qubits) {
                    assert!(q.index() < n, "gate touched an ancilla qubit");
                }
            }
        }
    }
}
