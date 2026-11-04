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

use qiskit_circuit::packed_instruction::PackedOperation;
use smallvec::SmallVec;

pub(crate) type TwoQubitSequenceVec = Vec<(PackedOperation, SmallVec<[f64; 3]>, SmallVec<[u8; 2]>)>;

#[derive(Clone, Debug)]
pub struct TwoQubitGateSequence {
    pub gates: TwoQubitSequenceVec,
    pub global_phase: f64,
}

impl TwoQubitGateSequence {
    pub fn gates(&self) -> &TwoQubitSequenceVec {
        &self.gates
    }

    pub fn into_gates(self) -> TwoQubitSequenceVec {
        self.gates
    }

    pub fn global_phase(&self) -> f64 {
        self.global_phase
    }

    pub fn set_state(&mut self, state: (TwoQubitSequenceVec, f64)) {
        self.gates = state.0;
        self.global_phase = state.1;
    }

    pub fn new() -> Self {
        TwoQubitGateSequence {
            gates: Vec::new(),
            global_phase: 0.,
        }
    }
    /// Create this sequence from the constituent parts.
    pub fn from_sequence(gates: TwoQubitSequenceVec, global_phase: f64) -> Self {
        Self {
            gates,
            global_phase,
        }
    }
}

impl Default for TwoQubitGateSequence {
    fn default() -> Self {
        Self::new()
    }
}
