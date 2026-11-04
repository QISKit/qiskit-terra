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

use std::ops::BitAnd;

use approx::abs_diff_eq;
use num_complex::{Complex64, ComplexFloat};
use pyo3::Python;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use hashbrown::HashSet;
use itertools::Itertools;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};

use qiskit_circuit::gate_matrix::ONE_QUBIT_IDENTITY;
use qiskit_util::complex::C_ZERO;

/// Find special unitary matrix that maps [c0,c1] to [r,0] or [0,r] if basis_state=0 or
/// basis_state=1 respectively
#[pyfunction]
pub fn reverse_qubit_state(
    py: Python,
    state: [Complex64; 2],
    basis_state: usize,
    epsilon: f64,
) -> Py<PyAny> {
    reverse_qubit_state_inner(&state, basis_state, epsilon)
        .into_pyarray(py)
        .into_any()
        .unbind()
}

#[inline(always)]
fn l2_norm(vec: &[Complex64]) -> f64 {
    vec.iter()
        .fold(0., |acc, elem| acc + elem.norm_sqr())
        .sqrt()
}

fn reverse_qubit_state_inner(
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

/// This method finds the single-qubit gates for a UCGate to disentangle a qubit:
/// we consider the n-qubit state v[:,0] starting with k zeros (in the computational basis).
/// The qubit with label n-s-1 is disentangled into the basis state k_s(k,s).

#[pyfunction]
pub fn find_squs_for_disentangling(
    py: Python,
    v: PyReadonlyArray2<Complex64>,
    k: usize,
    s: usize,
    epsilon: f64,
    n: usize,
) -> Vec<Py<PyAny>> {
    let v = v.as_array();
    let k_prime = 0;
    let i_start = if b(k, s + 1) == 0 {
        a(k, s + 1)
    } else {
        a(k, s + 1) + 1
    };
    let mut output: Vec<Array2<Complex64>> = (0..i_start).map(|_| Array2::eye(2)).collect();
    let mut squs: Vec<Array2<Complex64>> = (i_start..2_usize.pow((n - s - 1) as u32))
        .map(|i| {
            reverse_qubit_state_inner(
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
        .into_iter()
        .map(|x| x.into_pyarray(py).into_any().unbind())
        .collect()
}

#[pyfunction]
pub fn apply_ucg(
    py: Python,
    m: PyReadonlyArray2<Complex64>,
    k: usize,
    single_qubit_gates: Vec<PyReadonlyArray2<Complex64>>,
) -> Py<PyAny> {
    let mut m = m.as_array().to_owned();
    let shape = m.shape();
    let num_qubits = shape[0].ilog2();
    let num_col = shape[1];
    let spacing: usize = 2_usize.pow(num_qubits - k as u32 - 1);
    for j in 0..2_usize.pow(num_qubits - 1) {
        let i = (j / spacing) * spacing + j;
        let gate_index = i / (2_usize.pow(num_qubits - k as u32));
        for col in 0..num_col {
            let gate = single_qubit_gates[gate_index].as_array();
            let a = m[[i, col]];
            let b = m[[i + spacing, col]];
            m[[i, col]] = gate[[0, 0]] * a + gate[[0, 1]] * b;
            m[[i + spacing, col]] = gate[[1, 0]] * a + gate[[1, 1]] * b;
        }
    }
    m.into_pyarray(py).into_any().unbind()
}

#[inline(always)]
fn bin_to_int(bin: &[u8]) -> usize {
    bin.iter()
        .fold(0_usize, |acc, digit| (acc << 1) + *digit as usize)
}

#[pyfunction]
pub fn apply_diagonal_gate(
    py: Python,
    m: PyReadonlyArray2<Complex64>,
    action_qubit_labels: Vec<usize>,
    diag: PyReadonlyArray1<Complex64>,
) -> PyResult<Py<PyAny>> {
    let diag = diag.as_slice()?;
    let mut m = m.as_array().to_owned();
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
    Ok(m.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
pub fn apply_diagonal_gate_to_diag(
    mut m_diagonal: Vec<Complex64>,
    action_qubit_labels: Vec<usize>,
    diag: PyReadonlyArray1<Complex64>,
    num_qubits: usize,
) -> PyResult<Vec<Complex64>> {
    let diag = diag.as_slice()?;
    if m_diagonal.is_empty() {
        return Ok(m_diagonal);
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
    Ok(m_diagonal)
}

/// Helper method for _apply_multi_controlled_gate. This constructs the basis states the MG gate
/// is acting on for a specific state state_free of the qubits we neither control nor act on
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

#[pyfunction]
pub fn apply_multi_controlled_gate(
    py: Python,
    m: PyReadonlyArray2<Complex64>,
    control_labels: Vec<usize>,
    target_label: usize,
    gate: PyReadonlyArray2<Complex64>,
) -> Py<PyAny> {
    let mut m = m.as_array().to_owned();
    let gate = gate.as_array();
    let shape = m.shape();
    let num_qubits = shape[0].ilog2();
    let num_col = shape[1];
    let free_qubits = num_qubits as usize - control_labels.len() - 1;
    let control_set: HashSet<usize> = control_labels.into_iter().collect();
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
        return m.into_pyarray(py).into_any().unbind();
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
    m.into_pyarray(py).into_any().unbind()
}

#[pyfunction]
pub fn ucg_is_identity_up_to_global_phase(
    single_qubit_gates: Vec<PyReadonlyArray2<Complex64>>,
    epsilon: f64,
) -> bool {
    let global_phase: Complex64 = if single_qubit_gates[0].as_array()[[0, 0]].abs() >= epsilon {
        single_qubit_gates[0].as_array()[[0, 0]].finv()
    } else {
        return false;
    };
    for raw_gate in single_qubit_gates {
        let gate = raw_gate.as_array();
        if !abs_diff_eq!(
            gate.mapv(|x| x * global_phase),
            aview2(&ONE_QUBIT_IDENTITY),
            epsilon = 1e-8 // Default tolerance from numpy for allclose()
        ) {
            return false;
        }
    }
    true
}

#[pyfunction]
fn diag_is_identity_up_to_global_phase(diag: Vec<Complex64>, epsilon: f64) -> bool {
    let global_phase: Complex64 = if diag[0].abs() >= epsilon {
        diag[0].finv()
    } else {
        return false;
    };
    for d in diag {
        if (global_phase * d - 1.0).abs() >= epsilon {
            return false;
        }
    }
    true
}

#[pyfunction]
pub fn merge_ucgate_and_diag(
    py: Python,
    single_qubit_gates: Vec<PyReadonlyArray2<Complex64>>,
    diag: Vec<Complex64>,
) -> Vec<Py<PyAny>> {
    single_qubit_gates
        .iter()
        .enumerate()
        .map(|(i, raw_gate)| {
            let gate = raw_gate.as_array();
            let res = aview2(&[[diag[2 * i], C_ZERO], [C_ZERO, diag[2 * i + 1]]]).dot(&gate);
            res.into_pyarray(py).into_any().unbind()
        })
        .collect()
}

#[inline(always)]
#[pyfunction]
fn k_s(k: usize, s: usize) -> usize {
    if k == 0 {
        0
    } else {
        let filter = 1 << s;
        k.bitand(filter) >> s
    }
}

#[inline(always)]
#[pyfunction]
fn a(k: usize, s: usize) -> usize {
    k / 2_usize.pow(s as u32)
}

#[inline(always)]
#[pyfunction]
fn b(k: usize, s: usize) -> usize {
    k - (a(k, s) * 2_usize.pow(s as u32))
}

pub fn isometry(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(diag_is_identity_up_to_global_phase))?;
    m.add_wrapped(wrap_pyfunction!(find_squs_for_disentangling))?;
    m.add_wrapped(wrap_pyfunction!(reverse_qubit_state))?;
    m.add_wrapped(wrap_pyfunction!(apply_ucg))?;
    m.add_wrapped(wrap_pyfunction!(apply_diagonal_gate))?;
    m.add_wrapped(wrap_pyfunction!(apply_diagonal_gate_to_diag))?;
    m.add_wrapped(wrap_pyfunction!(apply_multi_controlled_gate))?;
    m.add_wrapped(wrap_pyfunction!(ucg_is_identity_up_to_global_phase))?;
    m.add_wrapped(wrap_pyfunction!(merge_ucgate_and_diag))?;
    m.add_wrapped(wrap_pyfunction!(a))?;
    m.add_wrapped(wrap_pyfunction!(b))?;
    m.add_wrapped(wrap_pyfunction!(k_s))?;
    Ok(())
}
