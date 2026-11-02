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

//! The following integration test demonstrates how to construct MIR for a Bell circuit.
//! Using the following Qiskit Circuit, and `preset_pbc_pass_manager` con convert a QiskitCircuit to PBC form.
//!
//! #!/usr/bin/env python3
//! from qiskit import QuantumCircuit
//! from qiskit.transpiler import generate_preset_pbc_pass_manager
//!
//! def make_bell_circuit():
//!     qc = QuantumCircuit(2)
//!     qc.h(0)
//!     qc.cx(0, 1)
//!     qc.measure_all()
//!     return qc
//!
//! if __name__ == "__main__":
//!     qc = make_bell_circuit()
//!     pm = generate_preset_pbc_pass_manager()
//!     qc_pbc = pm.run(qc)
//!     print(qc_pbc)
//!
//! global phase: π/4
//!         ┌──────────┐┌────────┐┌────────────┐┌───────────┐ ░ ┌─────────┐
//!    q_0: ┤ R_Y(π/2) ├┤ R_X(π) ├┤0           ├┤ R_Z(-π/2) ├─░─┤0        ├───────────
//!         └──────────┘└────────┘│  R_XZ(π/2) │├───────────┤ ░ │         │┌─────────┐
//!    q_1: ──────────────────────┤1           ├┤ R_X(-π/2) ├─░─┤  PPM(Z) ├┤0        ├
//!                               └────────────┘└───────────┘ ░ │         ││         │
//! meas_0: ════════════════════════════════════════════════════╡0        ╞╡         ╞
//!                                                             └─────────┘│         │
//! meas_1: ═══════════════════════════════════════════════════════════════╡0        ╞
//!                                                                        └─────────┘

use qiskit_mir::{
    instruction::MirInstruction,
    mir::Mir,
    ops::{
        MirOp,
        pbc::{AngleType, Pauli, PauliRotation, PauliString},
    },
    qid::{BitLayout, QidInfo, QidList},
    symbol_table::SymbolIdList,
};
use std::f64::consts::PI;

#[test]
fn construct_bell_mir() {
    let mut mir = Mir::new();

    // Create Qids for q_0 and q_1.

    // These will be virtual to start, and then mapped to physical through
    // transforms on the Mir.
    // They could be created as physical to start with also.
    let resource_table: &mut qiskit_mir::qid::ResourceTable = mir.resource_table_mut();
    let qid_0 = resource_table
        .add_qid(QidInfo::new(BitLayout::Base(1))) // use Base here to represent individual qubits with no aliasing/overlap
        .unwrap();
    let qid_1 = resource_table
        .add_qid(QidInfo::new(BitLayout::Base(1))) // use Base here to represent individual qubits with no aliasing/overlap
        .unwrap();

    // Here the instructions are created manually, from visually inspecting the
    // circuit above.
    //
    // TODO: Write a lowering pass that iterates over the QiskitCircuit (in PBC)
    // and for each operation create the corresponding MIR instruction, adding
    // the Qid to the ResourceTable if they have not already been added.
    let instr_list: &mut qiskit_mir::line_graph::LineGraph<MirInstruction> = mir.instructions_mut();
    instr_list.push_back(MirInstruction::new(
        MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![Pauli::Y]),
            AngleType::new(PI / 2.0),
        )),
        QidList::new(vec![qid_0]),
        SymbolIdList::default(),
        SymbolIdList::default(),
    ));

    instr_list.push_back(MirInstruction::new(
        MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![Pauli::X]),
            AngleType::new(PI),
        )),
        QidList::new(vec![qid_0]),
        SymbolIdList::default(),
        SymbolIdList::default(),
    ));

    instr_list.push_back(MirInstruction::new(
        MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![Pauli::Z, Pauli::X]),
            AngleType::new(PI / 2.0),
        )),
        QidList::new(vec![qid_0, qid_1]),
        SymbolIdList::default(),
        SymbolIdList::default(),
    ));

    instr_list.push_back(MirInstruction::new(
        MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![Pauli::Z]),
            AngleType::new(-PI / 2.0),
        )),
        QidList::new(vec![qid_0]),
        SymbolIdList::default(),
        SymbolIdList::default(),
    ));

    instr_list.push_back(MirInstruction::new(
        MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![Pauli::X]),
            AngleType::new(-PI / 2.0),
        )),
        QidList::new(vec![qid_1]),
        SymbolIdList::default(),
        SymbolIdList::default(),
    ));

    //
    // Measures go here -- Not Defined Yet

    // TODO: Add back when verify implemented.
    //instr_list.verify().unwrap();
}
