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

use numpy::PyArrayLike1;
use smallvec::smallvec;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Param, StandardGate};

use super::linear_phase::cz_depth_lnn::LnnGatesVec;

mod utils;

/// Checks whether an array of size N is a permutation of 0, 1, ..., N - 1.
#[pyfunction]
#[pyo3(signature = (pattern))]
pub fn _validate_permutation(
    py: Python,
    pattern: PyArrayLike1<i64, numpy::AllowTypeChange>,
) -> PyResult<Py<PyAny>> {
    let view = pattern.as_array();
    utils::validate_permutation(&view)?;
    Ok(py.None())
}

/// Finds inverse of a permutation pattern.
#[pyfunction]
#[pyo3(signature = (pattern))]
pub fn _inverse_pattern(
    py: Python,
    pattern: PyArrayLike1<i64, numpy::AllowTypeChange>,
) -> PyResult<Py<PyAny>> {
    let view = pattern.as_array();
    let inverse_i64: Vec<i64> = utils::invert(&view).iter().map(|&x| x as i64).collect();
    Ok(inverse_i64.into_pyobject(py)?.unbind())
}

#[pyfunction]
#[pyo3(signature = (pattern))]
pub fn _synth_permutation_basic(
    pattern: PyArrayLike1<i64, numpy::AllowTypeChange>,
) -> PyResult<PyCircuitData> {
    let view = pattern.as_array();
    let num_qubits = view.len();
    Ok(CircuitData::from_standard_gates(
        num_qubits as u32,
        utils::get_ordered_swap(&view).iter().map(|(i, j)| {
            (
                StandardGate::Swap,
                smallvec![],
                smallvec![Qubit::new(*i), Qubit::new(*j)],
            )
        }),
        Param::Float(0.0),
    )?
    .into())
}

#[pyfunction]
#[pyo3(signature = (pattern))]
fn _synth_permutation_acg(
    pattern: PyArrayLike1<i64, numpy::AllowTypeChange>,
) -> PyResult<PyCircuitData> {
    let inverted = utils::invert(&pattern.as_array());
    let view = inverted.view();
    let num_qubits = view.len();
    let cycles = utils::pattern_to_cycles(&view);
    let swaps = utils::decompose_cycles(&cycles);

    Ok(CircuitData::from_standard_gates(
        num_qubits as u32,
        swaps.iter().map(|(i, j)| {
            (
                StandardGate::Swap,
                smallvec![],
                smallvec![Qubit::new(*i), Qubit::new(*j)],
            )
        }),
        Param::Float(0.0),
    )?
    .into())
}

/// Synthesize a permutation circuit for a linear nearest-neighbor
/// architecture using the Kutin, Moulton, Smithline method.
#[pyfunction]
#[pyo3(signature = (pattern))]
pub fn _synth_permutation_depth_lnn_kms(
    pattern: PyArrayLike1<i64, numpy::AllowTypeChange>,
) -> PyResult<PyCircuitData> {
    let mut inverted = utils::invert(&pattern.as_array());
    let mut view = inverted.view_mut();
    let num_qubits = view.len();
    let mut swap_layers: Vec<(usize, usize)> = Vec::new();

    for i in 0..num_qubits {
        let swap_layer: Vec<(usize, usize)> = utils::create_swap_layer(&mut view, i % 2);
        swap_layers.extend(swap_layer);
    }

    Ok(CircuitData::from_standard_gates(
        num_qubits as u32,
        swap_layers.iter().map(|(i, j)| {
            (
                StandardGate::Swap,
                smallvec![],
                smallvec![Qubit::new(*i), Qubit::new(*j)],
            )
        }),
        Param::Float(0.0),
    )?
    .into())
}

/// A single layer of CX gates.
pub(crate) fn _append_cx_stage1(gates: &mut LnnGatesVec, n: usize) {
    for i in 0..(n / 2) {
        gates.push((
            StandardGate::CX,
            smallvec![],
            smallvec![Qubit::new(2 * i), Qubit::new(2 * i + 1)],
        ))
    }

    for i in 0..(n.div_ceil(2).saturating_sub(1)) {
        gates.push((
            StandardGate::CX,
            smallvec![],
            smallvec![Qubit::new(2 * i + 2), Qubit::new(2 * i + 1)],
        ))
    }
}

/// A single layer of CX gates.
pub(crate) fn _append_cx_stage2(gates: &mut LnnGatesVec, n: usize) {
    for i in 0..(n / 2) {
        gates.push((
            StandardGate::CX,
            smallvec![],
            smallvec![Qubit::new(2 * i + 1), Qubit::new(2 * i)],
        ))
    }

    for i in 0..(n.div_ceil(2) - 1) {
        gates.push((
            StandardGate::CX,
            smallvec![],
            smallvec![Qubit::new(2 * i + 1), Qubit::new(2 * i + 2)],
        ))
    }
}

/// Append reverse permutation to a QuantumCircuit for linear nearest-neighbor architectures
/// using Kutin, Moulton, Smithline method.
pub(crate) fn _append_reverse_permutation_lnn_kms(gates: &mut LnnGatesVec, num_qubits: usize) {
    (0..num_qubits.div_ceil(2)).for_each(|_| {
        _append_cx_stage1(gates, num_qubits);
        _append_cx_stage2(gates, num_qubits);
    });

    if num_qubits.is_multiple_of(2) {
        _append_cx_stage1(gates, num_qubits);
    }
}

/// Synthesize reverse permutation for linear nearest-neighbor architectures using
/// Kutin, Moulton, Smithline method.
///
/// Synthesis algorithm for reverse permutation from [1], section 5.
/// This algorithm synthesizes the reverse permutation on :math:`n` qubits over
/// a linear nearest-neighbor architecture using CX gates with depth :math:`2 * n + 2`.
///
/// References:
///     1. Kutin, S., Moulton, D. P., Smithline, L.,
///        *Computation at a distance*, Chicago J. Theor. Comput. Sci., vol. 2007, (2007),
///        `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_
#[pyfunction]
#[pyo3(signature = (num_qubits))]
fn synth_permutation_reverse_lnn_kms(num_qubits: usize) -> PyResult<PyCircuitData> {
    let mut gates = LnnGatesVec::new();
    _append_reverse_permutation_lnn_kms(&mut gates, num_qubits);
    Ok(CircuitData::from_standard_gates(num_qubits as u32, gates, Param::Float(0.0))?.into())
}

pub fn permutation(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_validate_permutation, m)?)?;
    m.add_function(wrap_pyfunction!(_inverse_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(_synth_permutation_basic, m)?)?;
    m.add_function(wrap_pyfunction!(_synth_permutation_acg, m)?)?;
    m.add_function(wrap_pyfunction!(_synth_permutation_depth_lnn_kms, m)?)?;
    m.add_function(wrap_pyfunction!(synth_permutation_reverse_lnn_kms, m)?)?;
    Ok(())
}
