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

//! Definition of the Qiskit Middle IR.

use crate::instruction::MirInstruction;
use crate::line_graph::LineGraph;
use crate::qid::{QidSpace, ResourceTable};
use crate::symbol_table::SymbolTable;

/// The Qiskit Middle IR
///
/// This represents a quantum Mir compilation unit. It consists of an instruction list, a resource
/// table and a symbol table.
#[derive(Debug)]
pub struct Mir {
    instructions: LineGraph<MirInstruction>,
    resource_table: ResourceTable,
    symbol_table: SymbolTable,
}

impl Mir {
    /// Create an empty `Mir` with virtual qubits; use [`Mir::with_space`] for physical qubits.
    pub fn new() -> Self {
        Self::with_space(QidSpace::Virtual)
    }

    /// Create an empty `Mir` that uses qubits from the given space.
    pub fn with_space(space: QidSpace) -> Self {
        Self {
            instructions: LineGraph::new(),
            resource_table: ResourceTable::new(space),
            symbol_table: SymbolTable::default(),
        }
    }

    /// The instruction list of this `Mir`.
    pub fn instructions(&self) -> &LineGraph<MirInstruction> {
        &self.instructions
    }

    /// Mutable access to the instruction list of this `Mir`.
    pub fn instructions_mut(&mut self) -> &mut LineGraph<MirInstruction> {
        &mut self.instructions
    }

    /// The resource table of this `Mir`.
    pub fn resource_table(&self) -> &ResourceTable {
        &self.resource_table
    }

    /// Mutable access to the resource table of this `Mir`.
    pub fn resource_table_mut(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }

    /// The symbol table of this `Mir`.
    pub fn symbol_table(&self) -> &SymbolTable {
        &self.symbol_table
    }

    /// Mutable access to the symbol table of this `Mir`.
    pub fn symbol_table_mut(&mut self) -> &mut SymbolTable {
        &mut self.symbol_table
    }
}

impl Default for Mir {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::MirOp;
    use crate::ops::pbc::{AngleType, Pauli, PauliRotation, PauliString};
    use crate::qid::{BitLayout, QidInfo, QidList};
    use crate::symbol_table::{Symbol, SymbolIdList};

    #[test]
    fn test_populate_mir() {
        let mut mir = Mir::new();

        // Add a qubit, and an instruction using it to `Mir`
        let qid_info = QidInfo::new(BitLayout::Base(1));
        let qid = mir.resource_table_mut().add_qid(qid_info.clone()).unwrap();

        let op = MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![Pauli::X]),
            AngleType::new(0.5),
        ));

        let instruction = MirInstruction::new(
            op,
            QidList::new(vec![qid]),
            SymbolIdList::default(),
            SymbolIdList::default(),
        );
        mir.instructions_mut().push_back(instruction.clone());

        assert_eq!(mir.resource_table().iter_qids().len(), 1);
        assert_eq!(mir.resource_table().info(qid), Some(&qid_info));

        assert_eq!(mir.instructions().node_count(), 1);
        assert_eq!(
            *mir.instructions().iter_main().next().unwrap().weight(),
            instruction
        );

        // Add a symbol to `Mir`
        let sym_id = mir
            .symbol_table_mut()
            .add(Symbol::new("a".to_string()))
            .unwrap();

        let symbol = mir.symbol_table().get(sym_id).unwrap();
        assert_eq!(symbol.name(), "a");
        assert_eq!(symbol.id(), Some(sym_id));
    }

    #[test]
    fn test_create_mir_with_qid_space() {
        let mut virtual_mir = Mir::new();
        let qid = virtual_mir
            .resource_table_mut()
            .add_qid(QidInfo::new(BitLayout::Base(1)))
            .unwrap();
        assert_eq!(qid.space(), QidSpace::Virtual);

        let mut physical_mir = Mir::with_space(QidSpace::Physical);
        let qid = physical_mir
            .resource_table_mut()
            .add_qid(QidInfo::new(BitLayout::Base(1)))
            .unwrap();
        assert_eq!(qid.space(), QidSpace::Physical);
    }
}
