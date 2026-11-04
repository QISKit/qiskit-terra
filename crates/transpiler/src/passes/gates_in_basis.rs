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

use crate::target::{Qargs, Target};
use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::Qubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, PyDAGCircuit};
use qiskit_circuit::operations::Operation;
use qiskit_circuit::packed_instruction::PackedInstruction;
use rustworkx_core::petgraph::prelude::NodeIndex;

#[pyfunction]
#[pyo3(name = "any_gate_missing_from_target")]
pub fn py_gates_missing_from_target(dag: &PyDAGCircuit, target: &Target) -> PyResult<bool> {
    gates_missing_from_target(dag.try_read()?, target)
}

pub fn gates_missing_from_target(dag: &DAGCircuit, target: &Target) -> PyResult<bool> {
    #[inline]
    fn is_universal(gate: &PackedInstruction) -> bool {
        matches!(gate.op.name(), "barrier" | "store")
    }

    fn visit_gate(
        circuit: &DAGCircuit,
        target: &Target,
        gate_node: NodeIndex,
        qargs: &[Qubit],
        wire_map: &HashMap<Qubit, PhysicalQubit>,
    ) -> PyResult<bool> {
        let gate = circuit
            .dag()
            .node_weight(gate_node)
            .unwrap()
            .unwrap_operation();
        let qargs_mapped: Qargs = qargs.iter().map(|q| wire_map[q]).collect();
        // Skip parameter checking if the gate is control flow.
        if !target.instruction_supported(
            gate.op.name(),
            &qargs_mapped,
            gate.params_view(),
            !gate.is_parameterized() && gate.op.try_control_flow().is_none(),
        ) {
            return Ok(true);
        }
        if let Some(control_flow) = circuit.try_view_control_flow(gate) {
            for block in control_flow.blocks() {
                let block_qubits = (0..block.num_qubits()).map(Qubit::new);
                let inner_wire_map = qargs
                    .iter()
                    .zip(block_qubits)
                    .map(|(outer, inner)| (inner, wire_map[outer]))
                    .collect();
                if visit_circuit(target, block, &inner_wire_map)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn visit_circuit(
        target: &Target,
        circuit: &DAGCircuit,
        wire_map: &HashMap<Qubit, PhysicalQubit>,
    ) -> PyResult<bool> {
        for (gate_node, gate) in circuit.op_nodes(true) {
            if is_universal(gate) {
                continue;
            }
            let qargs = circuit.qargs_interner().get(gate.qubits);
            if visit_gate(circuit, target, gate_node, qargs, wire_map)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    // In the outer DAG, virtual and physical bits are the same thing.
    let wire_map: HashMap<Qubit, PhysicalQubit> = HashMap::from_iter(
        (0..dag.num_qubits()).map(|i| (Qubit::new(i), PhysicalQubit::new(i.try_into().unwrap()))),
    );

    // Process the DAG.
    for (gate_node, gate) in dag.op_nodes(true) {
        if is_universal(gate) {
            continue;
        }
        let qargs = dag.qargs_interner().get(gate.qubits);
        if visit_gate(dag, target, gate_node, qargs, &wire_map)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[pyfunction]
#[pyo3(name = "any_gate_missing_from_basis")]
pub fn py_gates_missing_from_basis(dag: &PyDAGCircuit, basis: HashSet<String>) -> PyResult<bool> {
    gates_missing_from_basis(dag.try_read()?, basis)
}

pub fn gates_missing_from_basis(dag: &DAGCircuit, basis: HashSet<String>) -> PyResult<bool> {
    for (gate, _) in dag.count_ops(true)? {
        if !basis.contains(gate.as_str()) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn gates_in_basis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_gates_missing_from_target))?;
    m.add_wrapped(wrap_pyfunction!(py_gates_missing_from_basis))?;
    Ok(())
}
