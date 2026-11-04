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

/// Remove diagonal gates (including diagonal 2Q gates) before a measurement.
use hashbrown::HashSet;
use pyo3::prelude::*;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, PyDAGCircuit};
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::StandardGate;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

/// Run the RemoveDiagonalGatesBeforeMeasure pass on `dag`.
/// Args:
///     dag (DAGCircuit): the DAG to be optimized.
/// Returns:
///     DAGCircuit: the optimized DAG.
#[pyfunction]
#[pyo3(name = "remove_diagonal_gates_before_measure")]
pub fn py_run_remove_diagonal_before_measure(dag: &mut PyDAGCircuit) -> PyResult<()> {
    run_remove_diagonal_before_measure(dag.try_write()?);
    Ok(())
}

/// Run the RemoveDiagonalGatesBeforeMeasure pass on `dag`.
/// # Arguments:
/// * `dag` - the DAG to be optimized.
///
/// # Returns:
/// The optimized DAG.
pub fn run_remove_diagonal_before_measure(dag: &mut DAGCircuit) {
    static DIAGONAL_GATES: [StandardGate; 16] = [
        // 1Q gates
        StandardGate::RZ,
        StandardGate::Z,
        StandardGate::T,
        StandardGate::S,
        StandardGate::Tdg,
        StandardGate::Sdg,
        StandardGate::U1,
        StandardGate::Phase,
        // 2Q gates
        StandardGate::CZ,
        StandardGate::CRZ,
        StandardGate::CU1,
        StandardGate::RZZ,
        StandardGate::CPhase,
        StandardGate::CS,
        StandardGate::CSdg,
        // 3Q gates
        StandardGate::CCZ,
    ];

    let is_measure = |node: NodeIndex| -> bool {
        matches!(&dag[node], NodeType::Operation(inst) if inst.op.name() == "measure")
    };
    let is_diagonal = |node: NodeIndex| -> bool {
        matches!(
            &dag[node],
            NodeType::Operation(inst)
                if inst.op.try_standard_gate().is_some_and(|gate| DIAGONAL_GATES.contains(&gate))
        )
    };

    // A diagonal gate can be removed if every one of its qubits is next acted on by either a
    // measurement or another removable gate. Starting from the gates right before each
    // measurement, we walk backwards through the DAG, so that a whole chain of diagonal gates
    // is removed in a single run. A gate whose successors were not all removable yet is
    // revisited once the last of them becomes removable.
    let mut stack: Vec<NodeIndex> = dag
        .op_nodes(true)
        .filter(|(_, inst)| inst.op.name() == "measure")
        .flat_map(|(index, _)| dag.quantum_predecessors(index))
        .collect();
    let mut removable: HashSet<NodeIndex> = HashSet::new();
    let mut nodes_to_remove: Vec<NodeIndex> = Vec::new();
    while let Some(node) = stack.pop() {
        if removable.contains(&node) || !is_diagonal(node) {
            continue;
        }
        if dag
            .quantum_successors(node)
            .all(|succ| removable.contains(&succ) || is_measure(succ))
        {
            removable.insert(node);
            nodes_to_remove.push(node);
            stack.extend(dag.quantum_predecessors(node));
        }
    }

    for node_to_remove in nodes_to_remove {
        dag.remove_op_node(node_to_remove);
    }
}

pub fn remove_diagonal_gates_before_measure_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_remove_diagonal_before_measure))?;
    Ok(())
}
