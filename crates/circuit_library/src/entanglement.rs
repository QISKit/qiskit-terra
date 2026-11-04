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

use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{
    Bound, PyAny, PyResult,
    types::{PyAnyMethods, PyList, PyListMethods, PyString, PyTuple},
};

use crate::QiskitError;

/// Get all-to-all entanglement. For 4 qubits and block size 2 we have:
/// [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
fn full(num_qubits: u32, block_size: u32) -> impl Iterator<Item = Vec<u32>> {
    (0..num_qubits).combinations(block_size as usize)
}

/// Get a linear entanglement structure. For ``n`` qubits and block size ``m`` we have:
/// [(0..m-1), (1..m), (2..m+1), ..., (n-m..n-1)]
fn linear(num_qubits: u32, block_size: u32) -> impl DoubleEndedIterator<Item = Vec<u32>> {
    (0..num_qubits - block_size + 1)
        .map(move |start_index| (start_index..start_index + block_size).collect())
}

/// Get a reversed linear entanglement. This is like linear entanglement but in reversed order:
/// [(n-m..n-1), ..., (1..m), (0..m-1)]
/// This is particularly interesting, as CX+"full" uses n(n-1)/2 gates, but operationally equals
/// CX+"reverse_linear", which needs only n-1 gates.
fn reverse_linear(num_qubits: u32, block_size: u32) -> impl Iterator<Item = Vec<u32>> {
    linear(num_qubits, block_size).rev()
}

/// Return the qubit indices for circular entanglement. This is defined as tuples of length ``m``
/// starting at each possible index ``(0..n)``. Historically, Qiskit starts with index ``n-m+1``.
/// This is probably easiest understood for a concerete example of 4 qubits and block size 3:
/// [(2,3,0), (3,0,1), (0,1,2), (1,2,3)]
fn circular(num_qubits: u32, block_size: u32) -> Box<dyn Iterator<Item = Vec<u32>>> {
    if block_size == 1 || num_qubits == block_size {
        Box::new(linear(num_qubits, block_size))
    } else {
        let historic_offset = num_qubits - block_size + 1;
        Box::new((0..num_qubits).map(move |start_index| {
            (0..block_size)
                .map(|i| (historic_offset + start_index + i) % num_qubits)
                .collect()
        }))
    }
}

/// Get pairwise entanglement. This is typically used on 2 qubits and only has a depth of 2, as
/// first all odd pairs, and then even pairs are entangled. For example on 6 qubits:
/// [(0, 1), (2, 3), (4, 5), /* now the even pairs */ (1, 2), (3, 4)]
fn pairwise(num_qubits: u32) -> impl Iterator<Item = Vec<u32>> {
    // for Python-folks (like me): pairwise is equal to linear[::2] + linear[1::2]
    linear(num_qubits, 2)
        .step_by(2)
        .chain(linear(num_qubits, 2).skip(1).step_by(2))
}

/// The shifted, circular, alternating (sca) entanglement is motivated from circuits 14/15 of
/// https://arxiv.org/abs/1905.10876. It corresponds to circular entanglement, with the difference
/// that in each layer (controlled by ``offset``) the entanglement gates are shifted by one, plus
/// in each second layer, the entanglement gate is turned upside down.
/// Offset 0 -> [(2,3,0), (3,0,1), (0,1,2), (1,2,3)]
/// Offset 1 -> [(3,2,1), (0,3,2), (1,0,3), (2,1,0)]
/// Offset 2 -> [(0,1,2), (1,2,3), (2,3,0), (3,0,1)]
/// ...
fn shift_circular_alternating(
    num_qubits: u32,
    block_size: u32,
    offset: usize,
) -> Box<dyn Iterator<Item = Vec<u32>>> {
    // index at which we split the circular iterator -- we use Python-like indexing here,
    // and define ``split`` as equivalent to a Python index of ``-offset``
    let split = (num_qubits - (offset as u32 % num_qubits)) % num_qubits;
    let shifted = circular(num_qubits, block_size)
        .skip(split as usize)
        .chain(circular(num_qubits, block_size).take(split as usize));
    if offset.is_multiple_of(2) {
        Box::new(shifted)
    } else {
        // if the offset is odd, reverse the indices inside the qubit block (e.g. turn CX
        // gates upside down)
        Box::new(shifted.map(|indices| indices.into_iter().rev().collect()))
    }
}

/// Get an entangler map for an arbitrary number of qubits.
///
/// Args:
///     num_qubits: The number of qubits of the circuit.
///     block_size: The number of qubits of the entangling block.
///     entanglement: The entanglement strategy as string.
///     offset: The block offset, can be used if the entanglements differ per block,
///         for example used in the "sca" mode.
///
/// Returns:
///     The entangler map using mode ``entanglement`` to scatter a block of ``block_size``
///     qubits on ``num_qubits`` qubits.
pub fn get_entanglement_from_str(
    num_qubits: u32,
    block_size: u32,
    entanglement: &str,
    offset: usize,
) -> PyResult<Box<dyn Iterator<Item = Vec<u32>>>> {
    if num_qubits == 0 || block_size == 0 {
        return Ok(Box::new(std::iter::empty()));
    }

    if block_size > num_qubits {
        return Err(QiskitError::new_err(format!(
            "block_size ({block_size}) cannot be larger than number of qubits ({num_qubits})"
        )));
    }

    match (entanglement, block_size) {
        ("full", _) => Ok(Box::new(full(num_qubits, block_size))),
        ("linear", _) => Ok(Box::new(linear(num_qubits, block_size))),
        ("reverse_linear", _) => Ok(Box::new(reverse_linear(num_qubits, block_size))),
        ("sca", _) => Ok(shift_circular_alternating(num_qubits, block_size, offset)),
        ("circular", _) => Ok(circular(num_qubits, block_size)),
        ("pairwise", 1) => Ok(Box::new(linear(num_qubits, 1))),
        ("pairwise", 2) => Ok(Box::new(pairwise(num_qubits))),
        ("pairwise", _) => Err(QiskitError::new_err(format!(
            "block_size ({block_size}) can be at most 2 for pairwise entanglement"
        ))),
        _ => Err(QiskitError::new_err(format!(
            "Unsupported entanglement: {entanglement}"
        ))),
    }
}

/// Get an entangler map for an arbitrary number of qubits.
///
/// Args:
///     num_qubits: The number of qubits of the circuit.
///     block_size: The number of qubits of the entangling block.
///     entanglement: The entanglement strategy.
///     offset: The block offset, can be used if the entanglements differ per block,
///         for example used in the "sca" mode.
///
/// Returns:
///     The entangler map using mode ``entanglement`` to scatter a block of ``block_size``
///     qubits on ``num_qubits`` qubits.
pub fn get_entanglement<'a>(
    num_qubits: u32,
    block_size: u32,
    entanglement: &'a Bound<PyAny>,
    offset: usize,
) -> PyResult<Box<dyn Iterator<Item = PyResult<Vec<u32>>> + 'a>> {
    // unwrap the callable, if it is one
    let entanglement = if entanglement.is_callable() {
        entanglement.call1((offset,))?
    } else {
        entanglement.to_owned()
    };

    if let Ok(strategy) = entanglement.cast::<PyString>() {
        let as_str = strategy.to_string();
        return Ok(Box::new(
            get_entanglement_from_str(num_qubits, block_size, as_str.as_str(), offset)?.map(Ok),
        ));
    } else if let Ok(dict) = entanglement.cast::<PyDict>() {
        if let Some(value) = dict.get_item(block_size)? {
            let list = value.cast::<PyList>()?;
            return _check_entanglement_list(list.to_owned(), block_size);
        } else {
            return Ok(Box::new(std::iter::empty()));
        }
    } else if let Ok(list) = entanglement.cast::<PyList>() {
        return _check_entanglement_list(list.to_owned(), block_size);
    }
    Err(QiskitError::new_err(
        "Entanglement must be a string or list of qubit indices.",
    ))
}

fn _check_entanglement_list<'a>(
    list: Bound<'a, PyList>,
    block_size: u32,
) -> PyResult<Box<dyn Iterator<Item = PyResult<Vec<u32>>> + 'a>> {
    let entanglement_iter = list.iter().map(move |el| {
        let connections: Vec<u32> = el.extract()?;

        if connections.len() != block_size as usize {
            return Err(QiskitError::new_err(format!(
                "Entanglement {connections:?} does not match block size {block_size}"
            )));
        }

        Ok(connections)
    });
    Ok(Box::new(entanglement_iter))
}

/// Get the entanglement for given number of qubits and block size.
///
/// Args:
///     num_qubits: The number of qubits to entangle.
///     block_size: The entanglement block size (e.g. 2 for CX or 3 for CCX).
///     entanglement: The entanglement strategy. This can be one of:
///
///         * string: Available options are ``"full"``, ``"linear"``, ``"pairwise"``
///             ``"reverse_linear"``, ``"circular"``, or ``"sca"``.
///         * list of tuples: A list of entanglements given as tuple, e.g. [(0, 1), (1, 2)].
///         * callable: A callable that takes as input an offset as ``int`` (usually the layer
///             in the variational circuit) and returns a string or list of tuples to use as
///             entanglement in this layer.
///
///     offset: An offset used by certain entanglement strategies (e.g. ``"sca"``) or if the
///         entanglement is given as callable. This is typically used to have different
///         entanglement structures in different layers of variational quantum circuits.
///
/// Returns:
///     The entanglement as list of tuples.
///
/// Raises:
///     QiskitError: In case the entanglement is invalid.
#[pyfunction]
#[pyo3(signature = (num_qubits, block_size, entanglement, offset=0))]
pub fn get_entangler_map<'py>(
    py: Python<'py>,
    num_qubits: u32,
    block_size: u32,
    entanglement: &Bound<PyAny>,
    offset: usize,
) -> PyResult<Vec<Bound<'py, PyTuple>>> {
    // The entanglement is Result<impl Iterator<Item = Result<Vec<u32>>>>, so there's two
    // levels of errors we must handle: the outer error is handled by the outer match statement,
    // and the inner (Result<Vec<u32>>) is handled upon the PyTuple creation.
    match get_entanglement(num_qubits, block_size, entanglement, offset) {
        Ok(entanglement) => entanglement
            .into_iter()
            .map(|vec| match vec {
                Ok(vec) => PyTuple::new(py, vec),
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<_>, _>>(),
        Err(e) => Err(e),
    }
}
