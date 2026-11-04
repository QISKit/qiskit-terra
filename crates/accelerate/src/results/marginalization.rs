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

use super::converters::hex_to_bin;
use hashbrown::HashMap;
use ndarray::prelude::*;
use num_bigint::BigUint;
use num_complex::Complex64;
use numpy::IntoPyArray;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use qiskit_util::getenv_use_multiple_threads;
use rayon::prelude::*;

fn marginalize<T: std::ops::AddAssign + Copy>(
    counts: HashMap<String, T>,
    indices: Option<Vec<usize>>,
) -> HashMap<String, T> {
    let mut out_counts: HashMap<String, T> = HashMap::with_capacity(counts.len());
    let clbit_size = counts.keys().next().unwrap().replace(['_', ' '], "").len();
    let all_indices: Vec<usize> = (0..clbit_size).collect();
    counts
        .iter()
        .map(|(k, v)| (k.replace(['_', ' '], ""), *v))
        .for_each(|(k, v)| match &indices {
            Some(indices) => {
                if all_indices == *indices {
                    out_counts.insert(k, v);
                } else {
                    let key_arr = k.as_bytes();
                    let new_key: String = indices
                        .iter()
                        .map(|bit| {
                            let index = clbit_size - *bit - 1;
                            match key_arr.get(index) {
                                Some(bit) => *bit as char,
                                None => '0',
                            }
                        })
                        .rev()
                        .collect();
                    out_counts
                        .entry(new_key)
                        .and_modify(|e| *e += v)
                        .or_insert(v);
                }
            }
            None => {
                out_counts.insert(k, v);
            }
        });
    out_counts
}

#[pyfunction]
#[pyo3(signature=(counts, indices=None))]
pub fn marginal_counts(
    counts: HashMap<String, u64>,
    indices: Option<Vec<usize>>,
) -> HashMap<String, u64> {
    marginalize(counts, indices)
}

#[pyfunction]
#[pyo3(signature=(counts, indices=None))]
pub fn marginal_distribution(
    counts: HashMap<String, f64>,
    indices: Option<Vec<usize>>,
) -> HashMap<String, f64> {
    marginalize(counts, indices)
}

#[inline]
fn map_memory(
    hexstring: &str,
    indices: &Option<Vec<usize>>,
    clbit_size: usize,
    return_hex: bool,
) -> String {
    let out = match indices {
        Some(indices) => {
            let bitstring = hex_to_bin(hexstring);
            let bit_array = bitstring.as_bytes();
            indices
                .iter()
                .map(|bit| {
                    let index = clbit_size - *bit - 1;
                    match bit_array.get(index) {
                        Some(bit) => *bit as char,
                        None => '0',
                    }
                })
                .rev()
                .collect()
        }
        None => hex_to_bin(hexstring),
    };
    if return_hex {
        format!("0x{:x}", BigUint::parse_bytes(out.as_bytes(), 2).unwrap())
    } else {
        out
    }
}

#[pyfunction(
    signature = (
        memory,
        indices=None,
        return_int=false,
        return_hex=false,
        parallel_threshold=1000,
    )
)]
pub fn marginal_memory(
    py: Python,
    memory: Vec<String>,
    indices: Option<Vec<usize>>,
    return_int: bool,
    return_hex: bool,
    parallel_threshold: usize,
) -> PyResult<Py<PyAny>> {
    let run_in_parallel = getenv_use_multiple_threads();
    let first_elem = memory.first();
    if first_elem.is_none() {
        let res: Vec<String> = Vec::new();
        return res.into_py_any(py);
    }

    let clbit_size = hex_to_bin(first_elem.unwrap()).len();

    let out_mem: Vec<String> = if memory.len() < parallel_threshold || !run_in_parallel {
        memory
            .iter()
            .map(|x| map_memory(x, &indices, clbit_size, return_hex))
            .collect()
    } else {
        memory
            .par_iter()
            .map(|x| map_memory(x, &indices, clbit_size, return_hex))
            .collect()
    };
    if return_int {
        if out_mem.len() < parallel_threshold || !run_in_parallel {
            Ok(out_mem
                .iter()
                .map(|x| BigUint::parse_bytes(x.as_bytes(), 2).unwrap())
                .collect::<Vec<BigUint>>()
                .into_py_any(py)?)
        } else {
            Ok(out_mem
                .par_iter()
                .map(|x| BigUint::parse_bytes(x.as_bytes(), 2).unwrap())
                .collect::<Vec<BigUint>>()
                .into_py_any(py)?)
        }
    } else {
        out_mem.into_py_any(py)
    }
}

#[pyfunction]
pub fn marginal_measure_level_0(
    py: Python,
    memory: PyReadonlyArray3<Complex64>,
    indices: Vec<usize>,
) -> Py<PyAny> {
    let mem_arr: ArrayView3<Complex64> = memory.as_array();
    let input_shape = mem_arr.shape();
    let new_shape = [input_shape[0], indices.len(), input_shape[2]];
    let out_arr: Array3<Complex64> =
        Array3::from_shape_fn(new_shape, |(i, j, k)| mem_arr[[i, indices[j], k]]);
    out_arr.into_pyarray(py).into_any().unbind()
}

#[pyfunction]
pub fn marginal_measure_level_0_avg(
    py: Python,
    memory: PyReadonlyArray2<Complex64>,
    indices: Vec<usize>,
) -> Py<PyAny> {
    let mem_arr: ArrayView2<Complex64> = memory.as_array();
    let input_shape = mem_arr.shape();
    let new_shape = [indices.len(), input_shape[1]];
    let out_arr: Array2<Complex64> =
        Array2::from_shape_fn(new_shape, |(i, j)| mem_arr[[indices[i], j]]);
    out_arr.into_pyarray(py).into_any().unbind()
}

#[pyfunction]
pub fn marginal_measure_level_1(
    py: Python,
    memory: PyReadonlyArray2<Complex64>,
    indices: Vec<usize>,
) -> Py<PyAny> {
    let mem_arr: ArrayView2<Complex64> = memory.as_array();
    let input_shape = mem_arr.shape();
    let new_shape = [input_shape[0], indices.len()];
    let out_arr: Array2<Complex64> =
        Array2::from_shape_fn(new_shape, |(i, j)| mem_arr[[i, indices[j]]]);
    out_arr.into_pyarray(py).into_any().unbind()
}

#[pyfunction]
pub fn marginal_measure_level_1_avg(
    py: Python,
    memory: PyReadonlyArray1<Complex64>,
    indices: Vec<usize>,
) -> PyResult<Py<PyAny>> {
    let mem_arr: &[Complex64] = memory.as_slice()?;
    let out_arr: Vec<Complex64> = indices.into_iter().map(|idx| mem_arr[idx]).collect();
    Ok(out_arr.into_pyarray(py).into_any().unbind())
}
