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
#![allow(clippy::too_many_arguments)]

use hashbrown::{HashMap, HashSet};
use ndarray::prelude::*;
use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use qiskit_util::IndexSet;
use rayon::prelude::*;

use pyo3::Python;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qiskit_util::getenv_use_multiple_threads;

struct SubsetResult {
    pub count: usize,
    pub error: f64,
    pub map: Vec<usize>,
    pub subgraph: Vec<[usize; 2]>,
    pub index: usize,
}

fn bfs_sort(adj_matrix: ArrayView2<f64>, start: usize, num_qubits: usize) -> Vec<usize> {
    let n = adj_matrix.shape()[0];
    let mut next_level: IndexSet<usize> = IndexSet::default();
    let mut bfs_order = Vec::with_capacity(num_qubits);
    let mut seen: HashSet<usize> = HashSet::with_capacity(n);
    next_level.insert(start);
    while !next_level.is_empty() {
        let this_level = next_level;
        next_level = IndexSet::default();
        let mut found: Vec<usize> = Vec::new();
        for v in this_level {
            if !seen.contains(&v) {
                seen.insert(v);
                found.push(v);
                bfs_order.push(v);
                if bfs_order.len() == num_qubits {
                    return bfs_order;
                }
            }
        }
        if seen.len() == n {
            return bfs_order;
        }
        for node in found {
            for (idx, v) in adj_matrix.index_axis(Axis(0), node).iter().enumerate() {
                if *v != 0. {
                    next_level.insert(idx);
                }
            }
            for (idx, v) in adj_matrix.index_axis(Axis(1), node).iter().enumerate() {
                if *v != 0. {
                    next_level.insert(idx);
                }
            }
        }
    }
    bfs_order
}

/// Find the best subset in the coupling graph
///
/// This function will find the best densely connected subgraph in the
/// coupling graph to run the circuit on. It factors in measurement error and
/// cx error if specified.
///
/// Args:
///
///     num_qubits (int): The number of circuit qubits
///     coupling_adjacency (numpy.ndarray): An adjacency matrix for the
///         coupling graph.
///     num_meas (int): The number of measurement operations in the circuit
///     num_cx (int): The number of CXGates that are in the circuit
///     use_error (bool): Set to True to use the error
///     symmetric_coupling_map (bool): Is the coupling graph symmetric
///     error_matrix (numpy.ndarray): A 2D array that represents the error
///         rates on the target device, where the indices are physical qubits.
///         The diagonal (i.e. ``error_matrix[i][i]``) is the measurement error rate
///         for each qubit (``i``) and the positions where the indices differ are the
///         2q/cx error rate for the corresponding qubit pair.
///
/// Returns:
///     (rows, cols, best_map): A tuple of the rows, columns and the best
///     mapping found by the function. This can be used to efficiently create
///     a sparse matrix that maps the layout of virtual qubits
///     (0 to ``num_qubits``) to the physical qubits on the coupling graph.
#[pyfunction]
#[pyo3(name = "best_subset")]
pub fn py_best_subset(
    py: Python,
    num_qubits: usize,
    coupling_adjacency: PyReadonlyArray2<f64>,
    num_meas: usize,
    num_cx: usize,
    use_error: bool,
    symmetric_coupling_map: bool,
    error_matrix: PyReadonlyArray2<f64>,
) -> (Py<PyAny>, Py<PyAny>, Py<PyAny>) {
    let coupling_adj_mat = coupling_adjacency.as_array();
    let err = error_matrix.as_array();
    let [rows, cols, best_map] = best_subset(
        num_qubits,
        coupling_adj_mat,
        num_meas,
        num_cx,
        use_error,
        symmetric_coupling_map,
        err,
    );
    (
        rows.into_pyarray(py).into_any().unbind(),
        cols.into_pyarray(py).into_any().unbind(),
        best_map.into_pyarray(py).into_any().unbind(),
    )
}

pub fn best_subset(
    num_qubits: usize,
    coupling_adj_mat: ArrayView2<f64>,
    num_meas: usize,
    num_cx: usize,
    use_error: bool,
    symmetric_coupling_map: bool,
    err: ArrayView2<f64>,
) -> [Vec<usize>; 3] {
    let coupling_shape = coupling_adj_mat.shape();
    let avg_meas_err = err.diag().mean().unwrap();

    let map_fn = |k| -> SubsetResult {
        let mut subgraph: Vec<[usize; 2]> = Vec::with_capacity(num_qubits);
        let bfs = bfs_sort(coupling_adj_mat, k, num_qubits);
        let bfs_set: HashSet<usize> = bfs.iter().copied().collect();
        let mut connection_count = 0;
        for node_idx in &bfs {
            coupling_adj_mat
                .index_axis(Axis(0), *node_idx)
                .into_iter()
                .enumerate()
                .filter_map(|(node, j)| {
                    if *j != 0. && bfs_set.contains(&node) {
                        Some(node)
                    } else {
                        None
                    }
                })
                .for_each(|node| {
                    connection_count += 1;
                    subgraph.push([*node_idx, node]);
                });
        }
        let error = if use_error {
            let mut ret_error = 0.;
            let meas_avg = bfs
                .iter()
                .map(|i| {
                    let idx = *i;
                    err[[idx, idx]]
                })
                .sum::<f64>()
                / num_qubits as f64;
            let meas_diff = meas_avg - avg_meas_err;
            if meas_diff > 0. {
                ret_error += num_meas as f64 * meas_diff;
            }
            let cx_sum: f64 = subgraph.iter().map(|edge| err[[edge[0], edge[1]]]).sum();
            let mut cx_err = cx_sum / subgraph.len() as f64;
            if symmetric_coupling_map {
                cx_err /= 2.;
            }
            ret_error += num_cx as f64 * cx_err;
            ret_error
        } else {
            0.
        };
        SubsetResult {
            count: connection_count,
            error,
            map: bfs,
            subgraph,
            index: k,
        }
    };

    let reduce_identity_fn = || -> SubsetResult {
        SubsetResult {
            count: 0,
            map: Vec::new(),
            error: f64::INFINITY,
            subgraph: Vec::new(),
            index: usize::MAX,
        }
    };

    let reduce_fn = |best: SubsetResult, curr: SubsetResult| -> SubsetResult {
        if use_error {
            if (curr.count >= best.count && curr.error < best.error)
                || (curr.count == best.count && curr.error == best.error && curr.index < best.index)
            {
                curr
            } else {
                best
            }
        } else if curr.count > best.count || (curr.count == best.count && curr.index < best.index) {
            curr
        } else {
            best
        }
    };

    let best_result = if getenv_use_multiple_threads() {
        (0..coupling_shape[0])
            .into_par_iter()
            .map(map_fn)
            .reduce(reduce_identity_fn, reduce_fn)
    } else {
        (0..coupling_shape[0])
            .map(map_fn)
            .reduce(reduce_fn)
            .unwrap()
    };
    let best_map: Vec<usize> = best_result.map;
    let mapping: HashMap<usize, usize> = best_map
        .iter()
        .enumerate()
        .map(|(best_edge, edge)| (*edge, best_edge))
        .collect();
    let new_cmap: Vec<[usize; 2]> = best_result
        .subgraph
        .iter()
        .map(|c| [mapping[&c[0]], mapping[&c[1]]])
        .collect();
    let rows: Vec<usize> = new_cmap.iter().map(|edge| edge[0]).collect();
    let cols: Vec<usize> = new_cmap.iter().map(|edge| edge[1]).collect();

    [rows, cols, best_map]
}

pub fn dense_layout_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_best_subset))?;
    Ok(())
}
