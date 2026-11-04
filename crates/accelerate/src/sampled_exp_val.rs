// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_complex::Complex64;

use hashbrown::HashMap;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::pauli_exp_val::fast_sum;
use qiskit_quantum_info::sparse_observable::BitTerm;
use qiskit_quantum_info::sparse_observable::PySparseObservable;
use qiskit_quantum_info::sparse_observable::SparseTermView;
use qiskit_util::complex::c64;

const OPER_TABLE_SIZE: usize = (b'Z' as usize) + 1;
const fn generate_oper_table() -> [[f64; 2]; OPER_TABLE_SIZE] {
    let mut table = [[0.; 2]; OPER_TABLE_SIZE];
    table[b'Z' as usize] = [1., -1.];
    table[b'0' as usize] = [1., 0.];
    table[b'1' as usize] = [0., 1.];
    table
}

static OPERS: [[f64; 2]; OPER_TABLE_SIZE] = generate_oper_table();

const fn generate_bitterm_oper_table() -> [[f64; 2]; 10] {
    let mut table = [[0.; 2]; 10];
    table[BitTerm::Z as usize] = [1., -1.];
    table[BitTerm::Zero as usize] = [1., 0.];
    table[BitTerm::One as usize] = [0., 1.];
    table
}

static BIT_TERM_OPERS: [[f64; 2]; 10] = generate_bitterm_oper_table();

fn bitstring_expval(dist: &HashMap<String, f64>, mut oper_str: String) -> f64 {
    let inds: Vec<usize> = oper_str
        .char_indices()
        .filter_map(|(index, oper)| if oper != 'I' { Some(index) } else { None })
        .collect();
    oper_str.retain(|c| !r#"I"#.contains(c));
    let denom: f64 = fast_sum(&dist.values().copied().collect::<Vec<f64>>());
    let exp_val: f64 = dist
        .iter()
        .map(|(bits, val)| {
            let temp_product: f64 = oper_str.bytes().enumerate().fold(1.0, |acc, (idx, oper)| {
                let diagonal: [f64; 2] = OPERS[oper as usize];
                let index_char: char = bits.as_bytes()[inds[idx]] as char;
                let index: usize = index_char.to_digit(10).unwrap() as usize;
                acc * diagonal[index]
            });
            val * temp_product
        })
        .sum();
    exp_val / denom
}

fn bitstring_expval_bitterm(dist: &HashMap<String, f64>, term: &SparseTermView) -> PyResult<f64> {
    let inds: Vec<usize> = term.indices.iter().map(|x| *x as usize).collect();
    let n = term.num_qubits as usize;
    let denom: f64 = fast_sum(&dist.values().copied().collect::<Vec<f64>>());
    let exp_val: PyResult<f64> = dist
        .iter()
        .map(|(bits, val)| {
            let temp_product: PyResult<f64> =
                term.bit_terms
                    .iter()
                    .enumerate()
                    .try_fold(1.0, |acc, (idx, bitterm)| {
                        if (*bitterm as usize) & 0b0011 != 0b0001 {
                            return Err(PyValueError::new_err(format!(
                                "Operator string '{:?}' contains non-diagonal terms",
                                term.bit_terms
                            )));
                        }
                        let diagonal = BIT_TERM_OPERS[*bitterm as usize];
                        let index_char: char = bits.as_bytes()[n - 1 - inds[idx]] as char;
                        let index: usize = index_char.to_digit(10).unwrap() as usize;
                        Ok(acc * diagonal[index])
                    });
            Ok(val * temp_product?)
        })
        .sum();
    Ok(exp_val? / denom)
}

/// Compute the expectation value from a sampled distribution
#[pyfunction]
#[pyo3(text_signature = "(oper_strs, coeff, dist, /)")]
pub fn sampled_expval_float(
    oper_strs: Vec<String>,
    coeff: PyReadonlyArray1<f64>,
    dist: HashMap<String, f64>,
) -> PyResult<f64> {
    let coeff_arr = coeff.as_slice()?;
    let out = oper_strs
        .into_iter()
        .enumerate()
        .map(|(idx, string)| coeff_arr[idx] * bitstring_expval(&dist, string))
        .sum();
    Ok(out)
}

/// Compute the expectation value from a sampled distribution
#[pyfunction]
#[pyo3(text_signature = "(oper_strs, coeff, dist, /)")]
pub fn sampled_expval_complex(
    oper_strs: Vec<String>,
    coeff: PyReadonlyArray1<Complex64>,
    dist: HashMap<String, f64>,
) -> PyResult<Complex64> {
    let coeff_arr = coeff.as_slice()?;
    let out: Complex64 = oper_strs
        .into_iter()
        .enumerate()
        .map(|(idx, string)| coeff_arr[idx] * c64(bitstring_expval(&dist, string), 0.))
        .sum::<Complex64>();
    Ok(out)
}

// Compute the expectation value from a sampled distribution for SparseObservable objects
#[pyfunction]
#[pyo3(text_signature = "(sparse_obs, dist, /)")]
pub fn sampled_expval_sparse_observable(
    sparse_obs: PyRef<PySparseObservable>,
    dist: HashMap<String, f64>,
) -> PyResult<Complex64> {
    // Access the SparseObservable
    let sparse_obs = sparse_obs.as_inner()?;
    let result: PyResult<Complex64> =
        sparse_obs
            .iter()
            .enumerate()
            .try_fold(Complex64::new(0.0, 0.0), |acc, (_idx, term)| {
                Ok(acc + term.coeff * Complex64::new(bitstring_expval_bitterm(&dist, &term)?, 0.0))
            });
    result
}

pub fn sampled_exp_val(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sampled_expval_float))?;
    m.add_wrapped(wrap_pyfunction!(sampled_expval_complex))?;
    m.add_wrapped(wrap_pyfunction!(sampled_expval_sparse_observable))?;
    Ok(())
}
