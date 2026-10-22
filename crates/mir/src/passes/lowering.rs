// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    instruction::MirInstruction,
    mir::Mir,
    ops::{
        MirOp,
        pbc::{AngleType, Pauli, PauliMeasure, PauliRotation, PauliString},
    },
    qid::{Qid, QidInfo, QidList},
    symbol_table::{Symbol, SymbolId, SymbolIdList, SymbolType},
};
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::OperationRef;
use qiskit_passmanager::{PassContext, StaticPass};

fn pauli_from_bits(x: bool, z: bool) -> Pauli {
    match (x, z) {
        (false, false) => Pauli::I,
        (true, false) => Pauli::X,
        (false, true) => Pauli::Z,
        (true, true) => Pauli::Y,
    }
}

pub fn lower(dag: &DAGCircuit) -> Mir {
    let mut mir = Mir::new();
    let mut qubits: BTreeSet<u32> = BTreeSet::new();
    let mut clbits: BTreeSet<u32> = BTreeSet::new();
    let mut qid_for_qubits: BTreeMap<u32, Qid> = BTreeMap::new();
    let mut sym_for_clbits: BTreeMap<u32, SymbolId> = BTreeMap::new();

    for node in dag.topological_op_nodes(false) {
        let instr = dag.dag()[node].unwrap_operation();
        for q in dag.get_qargs(instr.qubits) {
            qubits.insert(q.0);
        }
        for c in dag.get_cargs(instr.clbits) {
            clbits.insert(c.0);
        }
    }

    for qubit in qubits {
        let qid = mir
            .resource_table_mut()
            .add_qid(QidInfo::new(crate::qid::BitLayout::Base(1)))
            .expect("qid allocation should succeed");
        qid_for_qubits.insert(qubit, qid);
    }

    for clbit in clbits {
        let sym_id = mir
            .symbol_table_mut()
            .add(Symbol::new(format!("c{clbit}"), SymbolType::Bool))
            .expect("symbol allocation should succeed");
        sym_for_clbits.insert(clbit, sym_id);
    }

    for node in dag.topological_op_nodes(false) {
        let instr = dag.dag()[node].unwrap_operation();
        let qargs = dag.get_qargs(instr.qubits);
        let qids = qargs.iter().map(|q| qid_for_qubits[&q.0]).collect();
        match instr.op.view() {
            OperationRef::PauliProductRotation(rotation) => {
                let paulis = rotation
                    .x
                    .iter()
                    .zip(rotation.z.iter())
                    .map(|(&x, &z)| pauli_from_bits(x, z))
                    .collect();
                let angle = rotation
                    .angle
                    .try_float()
                    .expect("Currently MIR lowering supports only float angles");
                let op = MirOp::PauliRotation(PauliRotation::new(
                    PauliString::new(paulis),
                    AngleType::new(angle),
                ));
                mir.instructions_mut().push_back(MirInstruction::new(
                    op,
                    QidList::new(qids),
                    SymbolIdList::default(),
                    SymbolIdList::default(),
                ));
            }
            OperationRef::PauliProductMeasurement(measurement) => {
                let paulis: Vec<Pauli> = measurement
                    .x
                    .iter()
                    .zip(measurement.z.iter())
                    .map(|(&x, &z)| pauli_from_bits(x, z))
                    .collect();
                let node_clbits = dag.get_cargs(instr.clbits);
                assert_eq!(
                    node_clbits.len(),
                    1,
                    "PauliProductMeasurement should have 1 clbits"
                );
                let sym_id = sym_for_clbits[&node_clbits[0].0];
                let op = MirOp::PauliMeasure(PauliMeasure::new(
                    PauliString::new(paulis),
                    measurement.neg,
                ));
                mir.instructions_mut().push_back(MirInstruction::new(
                    op,
                    QidList::new(qids),
                    SymbolIdList::default(),
                    SymbolIdList::new(vec![sym_id]),
                ));
            }
            other => {
                panic!("MIR lowering doesn't support operation {other:?}");
            }
        }
    }
    mir
}

pub struct DagToMirLowering;

impl StaticPass<DAGCircuit, Mir> for DagToMirLowering {
    fn run(&self, ir: Box<DAGCircuit>, _context: &mut PassContext) -> anyhow::Result<Box<Mir>> {
        Ok(Box::new(lower(&ir)))
    }
}
