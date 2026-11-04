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
use numpy::PyReadonlyArray1;
use pulp::Simd;
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;

use qiskit_util::complex::c64;
use qiskit_util::getenv_use_multiple_threads;

const PARALLEL_THRESHOLD: usize = 19;

#[pulp::with_simd(fast_sum = pulp::Arch::new())]
#[inline(always)]
pub fn fast_sum_with_simd<S: Simd>(simd: S, values: &[f64]) -> f64 {
    let (head, tail) = S::as_simd_f64s(values);
    let sum: f64 = head
        .iter()
        .fold(0., |acc, chunk| acc + simd.reduce_sum_f64s(*chunk));
    sum + tail.iter().sum::<f64>()
}

/// Compute the pauli expectation value of a statevector without x
#[pyfunction]
#[pyo3(text_signature = "(data, num_qubits, z_mask, /)")]
pub fn expval_pauli_no_x(
    data: PyReadonlyArray1<Complex64>,
    num_qubits: usize,
    z_mask: usize,
) -> PyResult<f64> {
    if num_qubits >= usize::BITS as usize {
        return Err(PyOverflowError::new_err(format!(
            "The value for num_qubits, {num_qubits}, is too large and would overflow"
        )));
    }
    let data_arr = data.as_slice()?;
    let size = 1_usize << num_qubits;
    let run_in_parallel = getenv_use_multiple_threads();
    let map_fn = |i: usize| -> f64 {
        let mut val: f64 = data_arr[i].re * data_arr[i].re + data_arr[i].im * data_arr[i].im;
        if (i & z_mask).count_ones() & 1 != 0 {
            val *= -1.;
        }
        val
    };

    if num_qubits < PARALLEL_THRESHOLD || !run_in_parallel {
        Ok(fast_sum(&(0..size).map(map_fn).collect::<Vec<f64>>()))
    } else {
        Ok((0..size).into_par_iter().map(map_fn).sum())
    }
}

/// Compute the pauli expectation value of a statevector with x
#[pyfunction]
#[pyo3(text_signature = "(data, num_qubits, z_mask, x_mask, phase, x_max, /)")]
pub fn expval_pauli_with_x(
    data: PyReadonlyArray1<Complex64>,
    num_qubits: usize,
    z_mask: usize,
    x_mask: usize,
    phase: Complex64,
    x_max: u32,
) -> PyResult<f64> {
    if num_qubits > usize::BITS as usize {
        return Err(PyOverflowError::new_err(format!(
            "The value for num_qubits, {num_qubits}, is too large and would overflow",
        )));
    }
    let data_arr = data.as_slice()?;
    let mask_u = !(2_usize.pow(x_max + 1) - 1);
    let mask_l = 2_usize.pow(x_max) - 1;
    let size = 1_usize << (num_qubits - 1);
    let run_in_parallel = getenv_use_multiple_threads();
    let map_fn = |i: usize| -> f64 {
        let index_0 = ((i << 1) & mask_u) | (i & mask_l);
        let index_1 = index_0 ^ x_mask;
        let val_0 = (phase
            * c64(
                data_arr[index_1].re * data_arr[index_0].re
                    + data_arr[index_1].im * data_arr[index_0].im,
                data_arr[index_1].im * data_arr[index_0].re
                    - data_arr[index_1].re * data_arr[index_0].im,
            ))
        .re;
        let val_1 = (phase
            * c64(
                data_arr[index_0].re * data_arr[index_1].re
                    + data_arr[index_0].im * data_arr[index_1].im,
                data_arr[index_0].im * data_arr[index_1].re
                    - data_arr[index_0].re * data_arr[index_1].im,
            ))
        .re;
        let mut val = val_0;
        if (index_0 & z_mask).count_ones() & 1 != 0 {
            val *= -1.
        }
        if (index_1 & z_mask).count_ones() & 1 != 0 {
            val -= val_1;
        } else {
            val += val_1;
        }
        val
    };
    if num_qubits < PARALLEL_THRESHOLD || !run_in_parallel {
        Ok(fast_sum(&(0..size).map(map_fn).collect::<Vec<f64>>()))
    } else {
        Ok((0..size).into_par_iter().map(map_fn).sum())
    }
}

/// Compute the pauli expectation value of a density matrix without x
#[pyfunction]
#[pyo3(text_signature = "(data, num_qubits, z_mask, /)")]
pub fn density_expval_pauli_no_x(
    data: PyReadonlyArray1<Complex64>,
    num_qubits: usize,
    z_mask: usize,
) -> PyResult<f64> {
    if num_qubits >= usize::BITS as usize {
        return Err(PyOverflowError::new_err(format!(
            "The value for num_qubits, {num_qubits}, is too large and would overflow",
        )));
    }
    let data_arr = data.as_slice()?;
    let num_rows = 1_usize << num_qubits;
    let stride = 1 + num_rows;
    let run_in_parallel = getenv_use_multiple_threads();
    let map_fn = |i: usize| -> f64 {
        let index = i * stride;
        let mut val = data_arr[index].re;
        if (i & z_mask).count_ones() & 1 != 0 {
            val *= -1.;
        }
        val
    };
    if num_qubits < PARALLEL_THRESHOLD || !run_in_parallel {
        Ok(fast_sum(&(0..num_rows).map(map_fn).collect::<Vec<f64>>()))
    } else {
        Ok((0..num_rows).into_par_iter().map(map_fn).sum())
    }
}

/// Compute the pauli expectation value of a density matrix with x
#[pyfunction]
#[pyo3(text_signature = "(data, num_qubits, z_mask, x_mask, phase, x_max, /)")]
pub fn density_expval_pauli_with_x(
    data: PyReadonlyArray1<Complex64>,
    num_qubits: usize,
    z_mask: usize,
    x_mask: usize,
    phase: Complex64,
    x_max: u32,
) -> PyResult<f64> {
    if num_qubits >= usize::BITS as usize {
        return Err(PyOverflowError::new_err(format!(
            "The value for num_qubits, {num_qubits}, is too large and would overflow",
        )));
    }
    let data_arr = data.as_slice()?;
    let mask_u = !(2_usize.pow(x_max + 1) - 1);
    let mask_l = 2_usize.pow(x_max) - 1;
    let num_rows = 1_usize << num_qubits;
    let run_in_parallel = getenv_use_multiple_threads();
    let map_fn = |i: usize| -> f64 {
        let index_vec = ((i << 1) & mask_u) | (i & mask_l);
        let index_mat = (index_vec ^ x_mask) + num_rows * index_vec;
        let mut val = 2. * (phase * data_arr[index_mat]).re;
        if (index_vec & z_mask).count_ones() & 1 != 0 {
            val *= -1.
        }
        val
    };
    if num_qubits < PARALLEL_THRESHOLD || !run_in_parallel {
        Ok(fast_sum(
            &(0..num_rows >> 1).map(map_fn).collect::<Vec<f64>>(),
        ))
    } else {
        Ok((0..num_rows >> 1).into_par_iter().map(map_fn).sum())
    }
}

pub fn pauli_expval(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(expval_pauli_no_x))?;
    m.add_wrapped(wrap_pyfunction!(expval_pauli_with_x))?;
    m.add_wrapped(wrap_pyfunction!(density_expval_pauli_with_x))?;
    m.add_wrapped(wrap_pyfunction!(density_expval_pauli_no_x))?;
    Ok(())
}
