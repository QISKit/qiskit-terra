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

use std::ops::DerefMut;

use hashbrown::{HashMap, HashSet};

use pyo3::create_exception;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rustworkx_core::connectivity::connected_components;
use rustworkx_core::petgraph::EdgeType;
use rustworkx_core::petgraph::algo::toposort;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::visit::{IntoEdgeReferences, IntoNodeReferences, NodeFiltered};
use uuid::Uuid;

use crate::TranspilerError;
use crate::target::{Qargs, Target};
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::{Operation, OperationRef, StandardInstruction};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{
    BlockMapper, BlocksMode, Clbit, PhysicalQubit, Qubit, VarsMode, VirtualQubit,
};
use qiskit_util::py::ImportOnceCell;

create_exception!(qiskit, MultiQEncountered, pyo3::exceptions::PyException);

static COUPLING_MAP: ImportOnceCell =
    ImportOnceCell::new("qiskit.transpiler.coupling", "CouplingMap");

/// An individual disjoint component that provides the mapping of virtual qubits in the
/// original dag to the physical qubits components to run an isolated layout on.
/// The sub dag contains a filtered dag that removes the qubits outside of virtual qubits
/// You will need to map the `ShareableQubits` in `sub_dag` to the original dag's to figure out the
/// final mapping of `VirtualQubit` -> `PhysicalQubit`.
pub struct DisjointComponent {
    pub physical_qubits: Vec<PhysicalQubit>,
    pub sub_dag: DAGCircuit,
    pub virtual_qubits: Vec<VirtualQubit>,
}

/// The possible outcomes when trying to
pub enum DisjointSplit {
    /// There is no disjoint component in the connectivity graph so you don't need to do any
    /// special handling of disjoint components
    NoneNeeded,
    /// The are disjoint components but the entire DAG can fit in a single target component. This
    /// contains a list of the physical qubits that make up that single component to filter the
    /// layout problem to.
    TargetSubset(Vec<PhysicalQubit>),
    /// There are multiple disjoint components in the DAG and the connectivity graph that need to
    /// be mapped and the combined to form a complete initial layout. This contains a list of
    /// [DisjointComponent] objects which outline the isolated layout problems to solve.
    Arbitrary(Vec<DisjointComponent>),
}

type CouplingMap = UnGraph<PhysicalQubit, ()>;

fn subgraph(graph: &CouplingMap, node_set: &HashSet<NodeIndex>) -> CouplingMap {
    let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(node_set.len());
    let node_filter = |node: NodeIndex| -> bool { node_set.contains(&node) };
    let mut out_graph = CouplingMap::with_capacity(node_set.len(), node_set.len());
    let filtered = NodeFiltered(&graph, node_filter);
    for node in filtered.node_references() {
        let new_node = out_graph.add_node(*node.1);
        node_map.insert(node.0, new_node);
    }
    for edge in filtered.edge_references() {
        let new_source = *node_map.get(&edge.source()).unwrap();
        let new_target = *node_map.get(&edge.target()).unwrap();
        out_graph.add_edge(new_source, new_target, ());
    }
    out_graph
}

#[pyfunction(name = "run_pass_over_connected_components")]
pub fn py_run_pass_over_connected_components(
    dag: Bound<DAGCircuit>,
    target: &Target,
    run_func: Bound<PyAny>,
) -> PyResult<Option<Vec<Py<PyAny>>>> {
    let py = dag.py();
    let func = |dag: Bound<DAGCircuit>, cmap: &CouplingMap| -> PyResult<Py<PyAny>> {
        let py = run_func.py();
        let coupling_map_cls = COUPLING_MAP.get_bound(py);
        let endpoints: Vec<[usize; 2]> = cmap
            .edge_indices()
            .flat_map(|edge| {
                let endpoints = cmap.edge_endpoints(edge).unwrap();
                // Return bidirectional edges here because the rust space graph is
                // undirected and Python space CouplingMap is explicitly a directed graph.
                // Adding the reverse edge here is to ensure we are representing the coupling
                // map in Python as rust is working with it.
                [
                    [endpoints.0.index(), endpoints.1.index()],
                    [endpoints.1.index(), endpoints.0.index()],
                ]
            })
            .collect();
        let out_list = PyList::new(py, endpoints)?;
        let py_cmap = coupling_map_cls.call1((out_list,))?;
        for node in cmap.node_indices() {
            py_cmap
                .getattr(intern!(py, "graph"))?
                .set_item(node.index(), cmap.node_weight(node).unwrap().index())?;
        }
        Ok(run_func.call1((dag, py_cmap))?.unbind())
    };
    let components = {
        let mut borrowed = dag.borrow_mut();
        distribute_components(borrowed.deref_mut(), target)?
    };
    match components {
        DisjointSplit::NoneNeeded => {
            let coupling_map: CouplingMap = match build_coupling_map(target) {
                Some(map) => map,
                None => return Ok(None),
            };
            Ok(Some(vec![func(dag, &coupling_map)?]))
        }
        DisjointSplit::TargetSubset(qubits) => {
            let coupling_map = build_coupling_map(target).unwrap();
            let cmap = subgraph(
                &coupling_map,
                &qubits.iter().map(|x| NodeIndex::new(x.index())).collect(),
            );
            Ok(Some(vec![func(dag, &cmap)?]))
        }
        DisjointSplit::Arbitrary(components) => Some(
            components
                .into_iter()
                .map(|component| {
                    let coupling_map = build_coupling_map(target).unwrap();
                    let cmap = subgraph(
                        &coupling_map,
                        &component
                            .physical_qubits
                            .iter()
                            .map(|x| NodeIndex::new(x.index()))
                            .collect(),
                    );
                    func(component.sub_dag.into_pyobject(py)?, &cmap)
                })
                .collect::<PyResult<Vec<_>>>(),
        )
        .transpose(),
    }
}

pub fn distribute_components(dag: &mut DAGCircuit, target: &Target) -> PyResult<DisjointSplit> {
    let coupling_map: CouplingMap = match build_coupling_map(target) {
        Some(map) => map,
        None => return Ok(DisjointSplit::NoneNeeded),
    };
    let cmap_components = connected_components(&coupling_map);
    if cmap_components.len() == 1 {
        if dag.num_qubits() > cmap_components[0].len() {
            return Err(TranspilerError::new_err(concat!(
                "A connected component of the DAGCircuit is too large for any of the connected ",
                "components in the coupling map."
            )));
        }
        return Ok(DisjointSplit::NoneNeeded);
    }
    if let Some(largest_component) = cmap_components.iter().max_by_key(|x| x.len()) {
        let num_active_qubits = dag
            .qubit_io_map()
            .iter()
            .filter(|[source, target]| dag.dag().find_edge(*source, *target).is_none())
            .count();
        if largest_component.len() >= num_active_qubits {
            return Ok(DisjointSplit::TargetSubset(
                largest_component
                    .iter()
                    .map(|x| PhysicalQubit(x.index() as u32))
                    .collect(),
            ));
        }
    }
    let dag_components = separate_dag(dag)?;
    let mapped_components = map_components(&dag_components, &cmap_components)?;
    let out_component_pairs: Vec<(DAGCircuit, CouplingMap)> = mapped_components
        .into_iter()
        .enumerate()
        .filter(|(_, dag_indices)| !dag_indices.is_empty())
        .map(|(cmap_index, dag_indices)| {
            let mut out_dag = dag_components[*dag_indices.first().unwrap()].clone();
            for dag_index in &dag_indices[1..] {
                let dag = &dag_components[*dag_index];
                for qubit in dag.qubits().objects() {
                    out_dag.add_qubit_unchecked(qubit.clone())?;
                }
                for clbit in dag.clbits().objects() {
                    out_dag.add_clbit_unchecked(clbit.clone())?;
                }
                for qreg in dag.qregs() {
                    out_dag.add_qreg(qreg.clone())?;
                }
                for creg in dag.cregs() {
                    out_dag.add_creg(creg.clone())?;
                }

                let qubits_indices: Vec<Qubit> = dag
                    .qubits()
                    .objects()
                    .iter()
                    .map(|qubit| {
                        out_dag
                            .qubits()
                            .find(qubit)
                            .expect("Qubit from dag not found in out_dag")
                    })
                    .collect();
                let clbits_indices: Vec<Clbit> = dag
                    .clbits()
                    .objects()
                    .iter()
                    .map(|clbit| {
                        out_dag
                            .clbits()
                            .find(clbit)
                            .expect("Clbit from dag not found in out_dag")
                    })
                    .collect();

                let block_map = dag
                    .blocks()
                    .items()
                    .map(|(index, block)| (index, out_dag.add_block(block.clone())))
                    .collect();
                out_dag.compose(
                    dag,
                    Some(&qubits_indices),
                    Some(&clbits_indices),
                    block_map,
                    false,
                )?;
            }
            let subgraph = subgraph(&coupling_map, &cmap_components[cmap_index]);
            Ok((out_dag, subgraph))
        })
        .collect::<PyResult<Vec<_>>>()?;
    if out_component_pairs.len() == 1 {
        return Ok(DisjointSplit::TargetSubset(
            out_component_pairs[0]
                .1
                .node_weights()
                .map(|x| PhysicalQubit::new(x.index() as u32))
                .collect(),
        ));
    }
    Ok(DisjointSplit::Arbitrary(
        out_component_pairs
            .into_iter()
            .map(|(sub_dag, coupling_map)| {
                let physical_qubits = coupling_map
                    .node_weights()
                    .map(|x| PhysicalQubit::new(x.index() as u32))
                    .collect();
                let virtual_qubits = sub_dag
                    .qubits()
                    .objects()
                    .iter()
                    .map(|x| VirtualQubit::new(dag.qubit_locations().get(x).unwrap().index()))
                    .collect();
                DisjointComponent {
                    physical_qubits,
                    sub_dag,
                    virtual_qubits,
                }
            })
            .collect(),
    ))
}

fn map_components(
    dag_components: &[DAGCircuit],
    cmap_components: &[HashSet<NodeIndex>],
) -> PyResult<Vec<Vec<usize>>> {
    let mut free_qubits: Vec<usize> = cmap_components.iter().map(|g| g.len()).collect();
    let mut out_mapping = vec![Vec::new(); cmap_components.len()];
    let mut dag_qubits: Vec<(usize, usize)> = dag_components
        .iter()
        .enumerate()
        .map(|(idx, dag)| (idx, dag.num_qubits()))
        .collect();
    dag_qubits.sort_unstable_by_key(|x| x.1);
    dag_qubits.reverse();
    let mut cmap_indices = (0..cmap_components.len()).collect::<Vec<_>>();
    cmap_indices.sort_unstable_by_key(|x| free_qubits[*x]);
    cmap_indices.reverse();
    for (dag_index, dag_num_qubits) in dag_qubits {
        let mut found = false;
        for cmap_index in &cmap_indices {
            if dag_num_qubits <= free_qubits[*cmap_index] {
                found = true;
                out_mapping[*cmap_index].push(dag_index);
                free_qubits[*cmap_index] -= dag_num_qubits;
                break;
            }
        }
        if !found {
            return Err(TranspilerError::new_err(
                "A connected component of the DAGCircuit is too large for any of the connected components in the coupling map",
            ));
        }
    }
    Ok(out_mapping)
}

fn build_coupling_map(target: &Target) -> Option<UnGraph<PhysicalQubit, ()>> {
    let num_qubits = target.num_qubits.unwrap_or_default() as usize;
    if target.num_qargs() == 0 {
        return None;
    }
    let mut cm_graph = UnGraph::with_capacity(num_qubits, target.num_qargs() - num_qubits);
    for i in 0..num_qubits {
        cm_graph.add_node(PhysicalQubit::new(i as u32));
    }
    let qargs = target.qargs()?;
    for qarg in qargs {
        let Qargs::Concrete(qarg) = qarg else {
            continue;
        };
        if qarg.len() == 2 {
            let edge_index = cm_graph.find_edge(
                NodeIndex::new(qarg[0].index()),
                NodeIndex::new(qarg[1].index()),
            );
            match edge_index {
                Some(_) => {
                    continue;
                }
                None => {
                    cm_graph.add_edge(
                        NodeIndex::new(qarg[0].index()),
                        NodeIndex::new(qarg[1].index()),
                        (),
                    );
                }
            }
        }
    }
    Some(cm_graph)
}

struct InteractionGraphData<Ty: EdgeType> {
    im_graph: Graph<(), (), Ty>,
    reverse_im_graph_node_map: Vec<Option<Qubit>>,
}

fn generate_directed_interaction(dag: &DAGCircuit) -> PyResult<InteractionGraphData<Directed>> {
    let mut im_graph_node_map: Vec<Option<NodeIndex>> = vec![None; dag.num_qubits()];
    let mut reverse_im_graph_node_map: Vec<Option<Qubit>> = vec![None; dag.num_qubits()];
    let wire_map: Vec<Qubit> = (0..dag.num_qubits()).map(Qubit::new).collect();
    let mut im_graph = DiGraph::with_capacity(dag.num_qubits(), dag.num_qubits());
    build_interaction_graph(
        dag,
        &wire_map,
        &mut im_graph,
        &mut im_graph_node_map,
        &mut reverse_im_graph_node_map,
    )?;
    Ok(InteractionGraphData {
        im_graph,
        reverse_im_graph_node_map,
    })
}

fn build_interaction_graph<Ty: EdgeType>(
    dag: &DAGCircuit,
    wire_map: &[Qubit],
    im_graph: &mut Graph<(), (), Ty>,
    im_graph_node_map: &mut [Option<NodeIndex>],
    reverse_im_graph_node_map: &mut [Option<Qubit>],
) -> PyResult<()> {
    for (_, inst) in dag.op_nodes(false) {
        if let Some(control_flow) = dag.try_view_control_flow(inst) {
            for block in control_flow.blocks() {
                let mut inner_wire_map = vec![Qubit(u32::MAX); wire_map.len()];
                let node_qargs = dag.get_qargs(inst.qubits);

                for (outer, inner) in node_qargs.iter().zip(0..inst.op.num_qubits()) {
                    inner_wire_map[inner as usize] = wire_map[outer.index()]
                }
                build_interaction_graph(
                    block,
                    &inner_wire_map,
                    im_graph,
                    im_graph_node_map,
                    reverse_im_graph_node_map,
                )?;
            }
            continue;
        }
        let len_args = inst.op.num_qubits();
        if len_args == 1 {
            let dag_qubits = dag.get_qargs(inst.qubits);
            let qargs = wire_map[dag_qubits[0].index()];
            if im_graph_node_map[qargs.index()].is_none() {
                let new_index = im_graph.add_node(());
                im_graph_node_map[qargs.index()] = Some(new_index);
                reverse_im_graph_node_map[new_index.index()] = Some(qargs);
            }
        } else if len_args == 2 {
            let dag_qubits = dag.get_qargs(inst.qubits);
            let qargs: [Qubit; 2] = [
                wire_map[dag_qubits[0].index()],
                wire_map[dag_qubits[1].index()],
            ];
            if im_graph_node_map[qargs[0].index()].is_none() {
                let new_index = im_graph.add_node(());
                im_graph_node_map[qargs[0].index()] = Some(new_index);
                reverse_im_graph_node_map[new_index.index()] = Some(qargs[0]);
            }
            if im_graph_node_map[qargs[1].index()].is_none() {
                let new_index = im_graph.add_node(());
                im_graph_node_map[qargs[1].index()] = Some(new_index);
                reverse_im_graph_node_map[new_index.index()] = Some(qargs[1]);
            }

            let edge_index = im_graph.find_edge(
                im_graph_node_map[qargs[0].index()].unwrap(),
                im_graph_node_map[qargs[1].index()].unwrap(),
            );
            if edge_index.is_none() {
                im_graph.add_edge(
                    im_graph_node_map[qargs[0].index()].unwrap(),
                    im_graph_node_map[qargs[1].index()].unwrap(),
                    (),
                );
            }
        }
        if len_args > 2 {
            return Err(MultiQEncountered::new_err(""));
        }
    }
    Ok(())
}

fn separate_dag(dag: &mut DAGCircuit) -> PyResult<Vec<DAGCircuit>> {
    split_barriers(dag)?;
    let im_graph_data = generate_directed_interaction(dag)?;
    let connected_components = connected_components(&im_graph_data.im_graph);
    let component_qubits: Vec<HashSet<Qubit>> = connected_components
        .into_iter()
        .map(|component| {
            component
                .into_iter()
                .map(|x| im_graph_data.reverse_im_graph_node_map[x.index()].unwrap())
                .collect::<HashSet<Qubit>>()
        })
        .collect();
    let qubits: HashSet<Qubit> = (0..dag.num_qubits()).map(Qubit::new).collect();
    let decomposed_dags: PyResult<Vec<DAGCircuit>> = component_qubits
        .into_iter()
        .map(|dag_qubits| -> PyResult<DAGCircuit> {
            let mut new_dag = dag.copy_empty_like(VarsMode::Alike, BlocksMode::Drop);
            let qubits_to_revmove: Vec<Qubit> = qubits.difference(&dag_qubits).copied().collect();

            new_dag.remove_qubits(qubits_to_revmove)?;
            new_dag.set_global_phase_f64(0.);
            let old_qubits = dag.qubits();
            let mut block_map = BlockMapper::new();
            for index in toposort(dag.dag(), None).expect("DAGCircuit can't have a cycle") {
                let NodeType::Operation(ref node) = dag.dag()[index] else {
                    continue;
                };
                let qargs: HashSet<Qubit> = dag.get_qargs(node.qubits).iter().copied().collect();
                if dag_qubits.is_superset(&qargs) {
                    let qargs = dag.get_qargs(node.qubits);
                    let qarg_bits = old_qubits.map_indices(qargs).cloned();
                    let mapped_qubits: Vec<Qubit> =
                        new_dag.qubits().map_objects(qarg_bits)?.collect();
                    let mapped_clbits: Vec<Clbit> =
                        new_dag.cargs_interner().get(node.clbits).to_vec();
                    let mapped_params = node.params.as_deref().map(|p| {
                        block_map.map_params(p, |b| new_dag.add_block(dag.blocks()[b].clone()))
                    });
                    new_dag.apply_operation_back(
                        node.op.clone(),
                        &mapped_qubits,
                        &mapped_clbits,
                        mapped_params,
                        node.label.as_ref().map(|x| *x.clone()),
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                }
            }

            let idle_clbits: Vec<Clbit> = new_dag
                .clbit_io_map()
                .iter()
                .enumerate()
                .filter_map(|(index, [input, output])| {
                    if new_dag.dag().find_edge(*input, *output).is_some() {
                        Some(Clbit::new(index))
                    } else {
                        None
                    }
                })
                .collect();
            new_dag.remove_clbits(idle_clbits)?;
            combine_barriers(&mut new_dag, true)?;
            Ok(new_dag)
        })
        .collect();
    combine_barriers(dag, false)?;
    decomposed_dags
}

#[pyfunction]
pub fn combine_barriers(dag: &mut DAGCircuit, retain_uuid: bool) -> PyResult<()> {
    let mut uuid_map: HashMap<String, NodeIndex> = HashMap::new();
    let barrier_nodes: Vec<NodeIndex> = dag
        .op_nodes(true)
        .filter_map(|(index, inst)| {
            if let OperationRef::StandardInstruction(op) = inst.op.view() {
                if matches!(op, StandardInstruction::Barrier(_)) {
                    if let Some(label) = inst.label.as_ref() {
                        if label.contains("_uuid=") {
                            return Some(index);
                        }
                    }
                }
            }
            None
        })
        .collect();
    for node_index in barrier_nodes {
        let num_qubits = dag[node_index].unwrap_operation().op.num_qubits();
        let label = dag[node_index].unwrap_operation().label.clone().unwrap();
        match uuid_map.get(label.as_str()) {
            Some(other_index) => {
                let num_qubits = dag[*other_index].unwrap_operation().op.num_qubits() + num_qubits;
                let new_label = if retain_uuid {
                    Some(label.to_string())
                } else if label.starts_with("_none_uuid=") {
                    None
                } else {
                    let len = label.len();
                    let components: Vec<&str> = label.split("_uuid=").collect();
                    Some(components[..len - 1].join("_uuid="))
                };
                let new_op = PackedOperation::from_standard_instruction(
                    StandardInstruction::Barrier(num_qubits),
                );
                let new_node = dag.replace_block(
                    &[*other_index, node_index],
                    new_op,
                    None,
                    new_label.as_deref(),
                    true,
                    &HashMap::new(),
                    &HashMap::new(),
                )?;
                uuid_map.insert(*label, new_node);
            }
            None => {
                uuid_map.insert(*label, node_index);
            }
        }
    }
    Ok(())
}

fn split_barriers(dag: &mut DAGCircuit) -> PyResult<()> {
    for (_index, inst) in dag.op_nodes(true) {
        let OperationRef::StandardInstruction(StandardInstruction::Barrier(num_qubits)) =
            inst.op.view()
        else {
            continue;
        };
        if num_qubits == 1 {
            continue;
        }
        let barrier_uuid = match &inst.label {
            Some(label) => format!("{}_uuid={}", label, Uuid::new_v4()),
            None => format!("_none_uuid={}", Uuid::new_v4()),
        };
        let mut split_dag = DAGCircuit::new();
        for q in 0..num_qubits {
            split_dag.add_qubit_unchecked(ShareableQubit::new_anonymous())?;
            split_dag.apply_operation_back(
                PackedOperation::from_standard_instruction(StandardInstruction::Barrier(1)),
                &[Qubit(q)],
                &[],
                None,
                Some(barrier_uuid.clone()),
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
        }
    }
    Ok(())
}

pub fn disjoint_utils_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(combine_barriers))?;
    m.add_wrapped(wrap_pyfunction!(py_run_pass_over_connected_components))?;
    Ok(())
}
