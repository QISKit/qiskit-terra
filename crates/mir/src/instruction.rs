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

//! Definition of Instruction and InstructionList used in Qiskit Middle IR.

use crate::line_graph;
use crate::ops::MirOp;
use crate::qid::QidList;
use crate::symbol_table::SymbolIdList;
use std::fmt;

/// Instruction in Qiskit Middle IR
///
/// This represents an instruction in MIR. Each instruction consists of an
/// operation, a list of Quantum IDs, a list of SymbolIds of arguments and a
/// list of SymbolIds for returns. Any of these lists may be empty, indicating
/// that the specific field is not required for the given operation. For
/// example, a `PauliRotation` does not "return" any values and thus the
/// `returns` field should be an empty vector. Similarly, a branch operation
/// does not operate on, or effect any QuantumIDs and thus the `qids` list
/// should be empty.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize, mem_dbg::MemDbg))]
pub struct MirInstruction {
    op: MirOp,
    qids: QidList,
    args: SymbolIdList,
    returns: SymbolIdList, // Or just SymbolId
}

impl MirInstruction {
    pub fn new(op: MirOp, qids: QidList, args: SymbolIdList, returns: SymbolIdList) -> Self {
        Self {
            op,
            qids,
            args,
            returns,
        }
    }

    pub fn op(&self) -> &MirOp {
        &self.op
    }

    pub fn qids(&self) -> &QidList {
        &self.qids
    }

    pub fn args(&self) -> &SymbolIdList {
        &self.args
    }
    pub fn returns(&self) -> &SymbolIdList {
        &self.returns
    }
}

impl fmt::Display for MirInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "op: {:?}, qids: {:?}, args: {:?}, returns: {:?}",
            self.op, self.qids, self.args, self.returns
        )
    }
}

pub struct MirInstructionIndex(line_graph::Index);

impl MirInstructionIndex {
    pub fn inner(&self) -> line_graph::Index {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::line_graph::LineGraph;
    use crate::ops::pbc::{AngleType, Pauli, PauliRotation, PauliString};
    use crate::qid::Qid;
    use crate::symbol_table::SymbolIdList;

    fn create_pauli_rotation_op(pauli: Pauli, size: usize, angle: f64) -> MirOp {
        MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![pauli; size]),
            AngleType::new(angle),
        ))
    }

    #[test]
    fn test_create_instruction() {
        let instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::I, 8, 0.5),
            qids: QidList::new(vec![Qid::virt(0); 8]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        assert_eq!(*instr.op(), create_pauli_rotation_op(Pauli::I, 8, 0.5));
        assert_eq!(instr.qids().len(), 8);
        assert_eq!(instr.args().len(), 0);
        assert_eq!(instr.returns().len(), 0);
    }

    #[test]
    fn test_instruction_list() {
        let mut instr_list = LineGraph::<MirInstruction>::new();

        let instr1 = MirInstruction {
            op: create_pauli_rotation_op(Pauli::X, 8, 0.5),
            qids: QidList::new(vec![Qid::virt(0); 8]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        let instr2 = MirInstruction {
            op: create_pauli_rotation_op(Pauli::Y, 8, 0.5),
            qids: QidList::new(vec![Qid::virt(0); 8]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        let instr3 = MirInstruction {
            op: create_pauli_rotation_op(Pauli::Z, 8, 0.5),
            qids: QidList::new(vec![Qid::virt(0); 8]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        let check_list = [instr3.clone(), instr1.clone(), instr2.clone()];

        instr_list.push_back(instr1);
        instr_list.push_back(instr2);
        instr_list.push_front(instr3);

        // Iterate through the instruction list.
        // Expect to find instr3, instr1, instr2
        for (index, instr) in instr_list.iter_main().enumerate() {
            assert_eq!(*instr.weight(), check_list[index])
        }

        // TODO: Add back when verify implemented.
        //instr_list.verify().unwrap();
    }

    // TODO: Add back when verify implemented.
    // #[test]
    // fn test_invalid_pauli_instruction() {
    //     let instr = MirInstruction {
    //         op: create_pauli_rotation_op(Pauli::X, 4, 0.5),
    //         qids: QidList::new(vec![Qid::virt(0); 8]),
    //         args: SymbolIdList::default(),
    //         returns: SymbolIdList::default(),
    //     };

    //     let result = instr.verify();

    //     assert!(matches!(
    //         result,
    //         Err(MirVerifierError::MismatchedOperands { len1: 8, len2: 4 })
    //     ));
    // }
}
