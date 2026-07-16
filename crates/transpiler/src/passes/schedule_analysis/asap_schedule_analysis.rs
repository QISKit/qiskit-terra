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

use crate::TranspilerError;
use crate::passes::schedule_analysis::{NodeDurations, PyNodeDurations, TimeOps};
use hashbrown::HashMap;
use pyo3::prelude::*;
use qiskit_circuit::dag_circuit::{DAGCircuit, Wire};
use qiskit_circuit::operations::{OperationRef, PyInstruction, PyOpKind, StandardInstruction};
use qiskit_circuit::{Clbit, Qubit};
use qiskit_util::IndexMap;
use rustworkx_core::petgraph::prelude::NodeIndex;

pub fn run_asap_schedule_analysis<T: TimeOps>(
    dag: &DAGCircuit,
    clbit_write_latency: T,
    node_durations: &IndexMap<NodeIndex, T>,
) -> PyResult<IndexMap<NodeIndex, T>> {
    if dag.qregs().len() != 1 || !dag.qregs_data().contains_key("q") {
        return Err(TranspilerError::new_err(
            "ASAP schedule runs on physical circuits only",
        ));
    }

    let mut node_start_time: IndexMap<NodeIndex, T> = IndexMap::default();
    let mut idle_after: HashMap<Wire, T> = HashMap::new();

    let zero = T::zero();

    for index in 0..dag.qubits().len() {
        idle_after.insert(Wire::Qubit(Qubit::new(index)), zero);
    }

    for index in 0..dag.clbits().len() {
        idle_after.insert(Wire::Clbit(Clbit::new(index)), zero);
    }

    for node_index in dag.topological_op_nodes(false) {
        let op = dag[node_index].unwrap_operation();

        let qargs: Vec<Wire> = dag
            .qargs_interner()
            .get(op.qubits)
            .iter()
            .map(|&q| Wire::Qubit(q))
            .collect();
        let cargs: Vec<Wire> = dag
            .cargs_interner()
            .get(op.clbits)
            .iter()
            .map(|&c| Wire::Clbit(c))
            .collect();

        let &op_duration = node_durations.get(&node_index).ok_or_else(|| {
            TranspilerError::new_err(format!(
                "No duration found for node at index {}",
                node_index.index()
            ))
        })?;
        let op_view = op.op.view();
        let is_gate_or_delay = matches!(
            op_view,
            OperationRef::PyCustom(PyInstruction {
                kind: PyOpKind::Gate,
                ..
            }) | OperationRef::StandardGate(_)
                | OperationRef::StandardInstruction(StandardInstruction::Delay(_))
        );

        // compute t0, t1: instruction interval, note that
        // t0: start time of instruction
        // t1: end time of instruction

        let (t0, t1) = if is_gate_or_delay {
            let t0 = qargs
                .iter()
                .map(|q| idle_after[q])
                .fold(zero, |acc, x| *T::max(&acc, &x));
            (t0, t0 + op_duration)
        } else if matches!(
            op_view,
            OperationRef::StandardInstruction(StandardInstruction::Measure)
        ) {
            // Measure instruction handling is bit tricky due to clbit_write_latency
            let t0q = qargs
                .iter()
                .map(|q| idle_after[q])
                .fold(zero, |acc, x| *T::max(&acc, &x));
            let t0c = cargs
                .iter()
                .map(|c| idle_after[c])
                .fold(zero, |acc, x| *T::max(&acc, &x));
            // Assume following case (t0c > t0q)
            //
            //       |t0q
            // Q ▒▒▒▒░░░░░░░░░░░░
            // C ▒▒▒▒▒▒▒▒░░░░░░░░
            //           |t0c
            //
            // In this case, there is no actual clbit access until clbit_write_latency.
            // The node t0 can be push backward by this amount.
            //
            //         |t0q' = t0c - clbit_write_latency
            // Q ▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒
            // C ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
            //           |t0c' = t0c
            //
            // rather than naively doing
            //
            //           |t0q' = t0c
            // Q ▒▒▒▒░░░░▒▒▒▒▒▒▒▒
            // C ▒▒▒▒▒▒▒▒░░░▒▒▒▒▒
            //              |t0c' = t0c + clbut_write_latency
            let t0 = *T::max(&t0q, &(t0c - clbit_write_latency));
            let t1 = t0 + op_duration;
            for clbit in cargs.iter() {
                idle_after.insert(*clbit, t1);
            }
            (t0, t1)
        } else {
            // Directives (like Barrier)
            let t0 = qargs
                .iter()
                .chain(cargs.iter())
                .map(|bit| idle_after[bit])
                .fold(zero, |acc, x| *T::max(&acc, &x));
            (t0, t0 + op_duration)
        };

        for qubit in qargs {
            idle_after.insert(qubit, t1);
        }

        node_start_time.insert(node_index, t0);
    }

    Ok(node_start_time)
}

#[pyfunction]
/// Runs the ASAPSchedule analysis pass on dag.
///
/// Args:
///     dag (DAGCircuit): DAG to schedule.
///     clbit_write_latency (u64): The latency to write classical bits.
///     node_durations (PyDict): Mapping from node indices to operation durations.
///
/// Returns:
///     PyDict: A dictionary mapping each DAGOpNode to its scheduled start time.
///
#[pyo3(name = "asap_schedule_analysis", signature= (dag, clbit_write_latency, node_durations))]
pub fn py_run_asap_schedule_analysis(
    dag: &DAGCircuit,
    clbit_write_latency: u64,
    mut node_durations: PyNodeDurations,
) -> PyResult<PyNodeDurations> {
    // Extract indices and durations from PyDict
    // Get the first duration type
    let new_durations: NodeDurations = match &*node_durations {
        NodeDurations::Dt(node_durations) => {
            run_asap_schedule_analysis(dag, clbit_write_latency, node_durations)?.into()
        }
        NodeDurations::Seconds(node_durations) => {
            run_asap_schedule_analysis::<f64>(dag, clbit_write_latency as f64, node_durations)?
                .into()
        }
    };
    node_durations.update_durations(new_durations)?;
    Ok(node_durations)
}

pub fn asap_schedule_analysis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_asap_schedule_analysis))?;
    Ok(())
}
