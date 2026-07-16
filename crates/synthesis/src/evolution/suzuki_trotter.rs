// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashSet;
use itertools::Itertools;
use qiskit_quantum_info::sparse_observable::SparseTermView;
use qiskit_util::IndexMap;
use rustworkx_core::coloring::{ColoringStrategy, greedy_node_color_with_coloring_strategy};
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::{Graph, Undirected};
use std::convert::Infallible;

pub fn evolution(order: u32, num_paulis: usize) -> Vec<(usize, f64)> {
    match order {
        1 => (0..num_paulis).map(|i| (i, 1.)).collect(),
        2 => (0..num_paulis - 1)
            .map(|i| (i, 0.5))
            .chain(::std::iter::once((num_paulis - 1, 1.)))
            .chain((0..num_paulis - 1).rev().map(|i| (i, 0.5)))
            .collect(),
        _ => {
            let reduction = 1.0 / (4.0 - 4_f64.powf(1.0 / (order as f64 - 1.0)));
            let mut outer = evolution(order - 2, num_paulis);
            let outer_len = outer.len();
            outer.iter_mut().for_each(|p| {
                p.1 *= reduction;
            });
            let mut outer: Vec<(usize, f64)> =
                outer.into_iter().cycle().take(outer_len * 2).collect();
            let mut outer_r = outer.clone();

            let mut inner = evolution(order - 2, num_paulis);
            inner.iter_mut().for_each(|p| {
                p.1 *= 1.0 - 4.0 * reduction;
            });

            outer.append(&mut inner);
            outer.append(&mut outer_r);

            outer
        }
    }
}

pub fn reorder_terms<'a>(
    terms: impl Iterator<Item = SparseTermView<'a>>,
) -> Result<Vec<SparseTermView<'a>>, &'a str> {
    let sorted: Vec<SparseTermView<'a>> = terms
        .sorted_by_key(|view| (view.indices, view.bit_terms))
        .collect();
    let edges: Vec<(usize, usize)> = (0..sorted.len())
        .combinations(2)
        .map(|combination| (combination[0], combination[1]))
        .filter(|(index1, index2)| {
            let (indices1, indices2) = (
                HashSet::<&u32>::from_iter(sorted[*index1].indices),
                HashSet::<&u32>::from_iter(sorted[*index2].indices),
            );
            indices1.intersection(&indices2).count() > 0
        })
        .collect();

    let mut graph: Graph<usize, Option<u8>, Undirected> =
        Graph::with_capacity(sorted.len(), edges.len());

    sorted.iter().enumerate().for_each(|(i, _)| {
        graph.add_node(i);
    });
    edges.iter().for_each(|(index1, index2)| {
        graph.add_edge((*index1 as u32).into(), (*index2 as u32).into(), None);
    });

    let callback = |_: NodeIndex| -> Result<Option<usize>, Infallible> { Ok(None) };
    match greedy_node_color_with_coloring_strategy(&graph, callback, ColoringStrategy::Saturation) {
        Ok(colors) => {
            let mut colors_map: IndexMap<usize, Vec<&NodeIndex>> = IndexMap::default();

            for (node_index, color) in colors.iter().sorted() {
                if !colors_map.contains_key(color) {
                    colors_map.insert(*color, Vec::new());
                }
                colors_map.get_mut(color).unwrap().push(node_index);
            }

            Ok(colors_map
                .iter()
                .flat_map(|(_, node_index)| node_index)
                .map(|index| sorted[graph[**index]])
                .collect())
        }

        Err(_) => Err("Unexpected error when coloring Pauli sparse terms"),
    }
}
