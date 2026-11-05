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
use crate::ops::{
    MirOp,
    MirOp::{PauliMeasure, PauliRotation},
    pbc::PauliString,
};
use crate::qid::{Qid, QidList, table::ResourceTable};
use crate::symbol_table::SymbolIdList;
use std::fmt;
use thiserror::Error;

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

    /// Run verification on an Mir instruction.
    ///
    /// This verification ensures that assumptions for each instruction, and its
    /// containing operation, hold. Currently, the following assumptions must
    /// hold for all `MirInstruction`s:
    ///   - All [`Qid`]s must exist in the [`ResourceTable`]
    ///   - None of the [`Qid`]s can be aliased
    ///
    /// In addition to general verification for every instruction, op-specific
    /// verification is also run. Currently this is encapsulated in
    /// `MirInstruction::verify_pauli_op()` method, but will expand as more
    /// operations are added.
    ///
    /// TODO: The following verifications need to be added:
    ///   - All [SymbolIds](crate::symbol_table::SymbolId) in the `args` and `returns`
    ///     list must exist in the [SymbolTable](crate::symbol_table::SymbolTable).
    pub fn verify(
        &self,
        resource_table: &ResourceTable,
    ) -> Result<(), InstructionVerificationError> {
        // Verify all Qids exist in the ResourceTable
        for &q in self.qids().inner().iter() {
            if resource_table.info(q).is_none() {
                return Err(InstructionVerificationError::QidNotFound { qid: q })?;
            }
        }

        // Verify none of the Qids are aliased to each other
        if resource_table.overlap(self.qids().inner().iter().copied()) {
            return Err(InstructionVerificationError::AliasedQids);
        }

        // TODO: Verify all SymbolIds exist in the SymbolTable.

        // Perform op-specific verification
        match &self.op {
            PauliRotation(pauli_rotation) => {
                self.verify_pauli_op(pauli_rotation.paulis(), resource_table)?;
            }
            PauliMeasure(pauli_measure) => {
                self.verify_pauli_op(pauli_measure.paulis(), resource_table)?;
            }
        }
        Ok(())
    }

    /// Perform verification on the
    /// [PauliMeasure](crate::ops::pbc::PauliMeasure) and
    /// [PauliRotation](crate::ops::pbc::PauliRotation) operations.
    ///
    /// For [PauliMeasure](crate::ops::pbc::PauliMeasure) and
    /// [PauliRotation](crate::ops::pbc::PauliRotation) operations, the width of
    /// the [`Qid`]s on the instruction must match the number of
    /// [Pauli](crate::ops::pbc::Pauli)s in the provided [`PauliString`] on the
    /// operation. This also prevents any [`Qid`] from having an
    /// [`BitLayout::Abstract`] layout (i.e., the [`Qid`] must have a known
    /// width).
    fn verify_pauli_op(
        &self,
        paulis: &PauliString,
        resource_table: &ResourceTable,
    ) -> Result<(), InstructionVerificationError> {
        // Walk the QidList and check all Qids exist in the ResourceTable, are
        // not Abstract, and collect the total number.
        let mut total_qids = 0;
        for q in self.qids().inner().iter() {
            total_qids += resource_table
                .info(*q)
                .ok_or(InstructionVerificationError::QidNotFound { qid: *q })?
                .width()
                .ok_or(InstructionVerificationError::AbstractQidNotAllowed { qid: *q })?;
        }

        // Ensure total Qids match the number of Paulis
        if total_qids != paulis.len() as u32 {
            return Err(InstructionVerificationError::QidWidthMismatch {
                len1: total_qids,
                len2: paulis.len() as u32,
            });
        }

        Ok(())
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

#[derive(Error, Debug)]
pub enum InstructionVerificationError {
    #[error("Qid width ({}) does not match pauli count ({})", len1, len2)]
    QidWidthMismatch { len1: u32, len2: u32 },
    #[error("Qid {} not found in resource table.", qid)]
    QidNotFound { qid: Qid },
    #[error("Abstract Qid {} not allowed (no width specified)", qid)]
    AbstractQidNotAllowed { qid: Qid },
    #[error("Overlapping Qids")]
    AliasedQids,
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
    use crate::mir::Mir;
    use crate::ops::pbc::{AngleType, Pauli, PauliMeasure, PauliRotation, PauliString};
    use crate::qid::{BitLayout, Qid, QidInfo, QidSpace};
    use crate::symbol_table::{Symbol, SymbolTable, SymbolType};

    fn create_pauli_rotation_op(pauli: Pauli, size: usize, angle: f64) -> MirOp {
        MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![pauli; size]),
            AngleType::new(angle),
        ))
    }

    #[test]
    fn test_create_measurement_instruction_with_bool_symbol() {
        let mut symbol_table = SymbolTable::default();
        let sym_id = symbol_table
            .add(Symbol::new("c0".to_string(), SymbolType::Bool))
            .unwrap();
        let op = MirOp::PauliMeasure(PauliMeasure::new(PauliString::new(vec![Pauli::I])));
        let instr = MirInstruction {
            op: op.clone(),
            qids: QidList::new(vec![Qid::virt(0)]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::new(vec![sym_id]),
        };

        assert_eq!(*instr.op(), op);
        assert_eq!(instr.returns().len(), 1);
        let ret_id = instr.returns().inner()[0];
        assert_eq!(ret_id, sym_id);
        let symbol = symbol_table.get(ret_id).unwrap();
        assert_eq!(symbol.name(), "c0");
        assert_eq!(symbol.symbol_type(), SymbolType::Bool);
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
        let mut mir = Mir::new();

        let resource_table = mir.resource_table_mut();
        let qids = (0..8)
            .map(|_| {
                resource_table
                    .add_qid(QidInfo::new(BitLayout::Base(1)))
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let instr_list = mir.instructions_mut();

        let instr1 = MirInstruction {
            op: create_pauli_rotation_op(Pauli::X, 8, 0.5),
            qids: QidList::new(qids.clone()),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        let instr2 = MirInstruction {
            op: create_pauli_rotation_op(Pauli::Y, 8, 0.5),
            qids: QidList::new(qids.clone()),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        let instr3 = MirInstruction {
            op: create_pauli_rotation_op(Pauli::Z, 8, 0.5),
            qids: QidList::new(qids),
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

        mir.verify().unwrap();
    }

    #[test]
    fn test_invalid_pauli_instruction() {
        let mut resource_table = ResourceTable::new(QidSpace::Virtual);
        let qid_len = 8;
        let pauli_len = 4;
        let qids = (0..qid_len)
            .map(|_| {
                resource_table
                    .add_qid(QidInfo::new(BitLayout::Base(1)))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::X, pauli_len, 0.5),
            qids: QidList::new(qids),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        let result = instr.verify(&resource_table);

        assert!(matches!(
            result,
            Err(InstructionVerificationError::QidWidthMismatch {
                len1: _qid_len,
                len2: _pauli_len
            })
        ));
    }
    #[test]
    fn test_missing_qid() {
        let resource_table = ResourceTable::new(QidSpace::Virtual);
        let qid = Qid::virt(0);
        let instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::X, 1, 0.5),
            qids: QidList::new(vec![qid; 1]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        let result = instr.verify(&resource_table);

        assert!(matches!(
            result,
            Err(InstructionVerificationError::QidNotFound { qid: _ })
        ));
    }

    #[test]
    fn test_abstract_qid() {
        let mut resource_table = ResourceTable::new(QidSpace::Virtual);
        // Qid with Abstract layout cannot be used in instructions because they
        // have no width associated with them
        let qid = resource_table
            .add_qid(QidInfo::new(BitLayout::Abstract))
            .unwrap();

        let bad_instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::X, 1, 0.5),
            qids: QidList::new(vec![qid]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        assert!(
            matches!(
                bad_instr.verify(&resource_table),
                Err(InstructionVerificationError::AbstractQidNotAllowed { qid: _ })
            ),
            "Missed identifying abstract Qid in instruction"
        );
    }

    #[test]
    fn test_wide_qids() {
        let mut resource_table = ResourceTable::new(QidSpace::Virtual);
        let qid_container = resource_table
            .add_qid(QidInfo::new(BitLayout::Base(2)))
            .unwrap();

        let qid_member1 = resource_table
            .add_qid(QidInfo::new(BitLayout::Base(1)))
            .unwrap();
        let qid_member2 = resource_table
            .add_qid(QidInfo::new(BitLayout::Base(1)))
            .unwrap();

        resource_table
            .mark_contains(qid_container, qid_member1)
            .unwrap();
        resource_table
            .mark_contains(qid_container, qid_member2)
            .unwrap();

        let mut good_instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::X, 2, 0.5),
            qids: QidList::new(vec![qid_container]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        good_instr.verify(&resource_table).unwrap();

        good_instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::Y, 2, 0.5),
            qids: QidList::new(vec![qid_member1, qid_member2]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        good_instr.verify(&resource_table).unwrap();

        let mut bad_instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::Z, 3, 0.5),
            qids: QidList::new(vec![qid_member1, qid_member2]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        assert!(
            matches!(
                bad_instr.verify(&resource_table),
                Err(InstructionVerificationError::QidWidthMismatch { len1: 2, len2: 3 })
            ),
            "Mismatched operand lengths"
        );

        bad_instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::I, 1, 0.5),
            qids: QidList::new(vec![qid_container]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        assert!(
            matches!(
                bad_instr.verify(&resource_table),
                Err(InstructionVerificationError::QidWidthMismatch { len1: 2, len2: 1 }),
            ),
            "Mismatched operand lengths",
        );
    }

    #[test]
    fn test_aliased_qids() {
        let mut resource_table = ResourceTable::new(QidSpace::Virtual);
        let qid_container = resource_table
            .add_qid(QidInfo::new(BitLayout::Base(2)))
            .unwrap();
        let qid_member1 = resource_table
            .add_qid(QidInfo::new(BitLayout::Base(1)))
            .unwrap();
        let qid_member2 = resource_table
            .add_qid(QidInfo::new(BitLayout::Base(1)))
            .unwrap();

        resource_table
            .mark_contains(qid_container, qid_member1)
            .unwrap();
        resource_table
            .mark_contains(qid_container, qid_member2)
            .unwrap();

        let good_instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::X, 2, 0.5),
            qids: QidList::new(vec![qid_member1, qid_member2]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        // This should pass verification because Qids do not overlap
        good_instr.verify(&resource_table).unwrap();

        let bad_instr = MirInstruction {
            op: create_pauli_rotation_op(Pauli::X, 3, 0.5),
            qids: QidList::new(vec![qid_container, qid_member1]),
            args: SymbolIdList::default(),
            returns: SymbolIdList::default(),
        };

        assert!(
            matches!(
                bad_instr.verify(&resource_table),
                Err(InstructionVerificationError::AliasedQids)
            ),
            "Missed identifying aliased qids"
        );
    }
}
