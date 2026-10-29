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

//! Definition of operations in Qiskit MIR.
//!
//! Each file represents a different logical category of operations.

pub mod pbc;

use crate::ops::pbc::{PauliMeasure, PauliRotation};

/// Enum to track the various Operations in MIR.
///
/// This is intended to be grouped by logical category of operations (e.g.,
/// quantum, control flow, classical, etc).
#[derive(Clone, Debug, PartialEq)]
pub enum MirOp {
    // Quantum operations
    PauliRotation(PauliRotation),
    PauliMeasure(PauliMeasure),
}

impl std::fmt::Display for MirOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MirOp::PauliRotation(rotation) => write!(f, "{rotation}"),
            MirOp::PauliMeasure(measurement) => write!(f, "{measurement}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::MirOp;
    use crate::ops::PauliRotation;
    use crate::ops::pbc::PauliMeasure;
    use crate::ops::pbc::{AngleType, Pauli, PauliString};

    #[test]
    fn test_mir_op() {
        let pauli_rotation_op =
            PauliRotation::new(PauliString::new(vec![Pauli::I; 8]), AngleType::new(0.5));
        let mir_op = MirOp::PauliRotation(pauli_rotation_op);

        let mir_op2 = MirOp::PauliRotation(PauliRotation::new(
            PauliString::new(vec![Pauli::I; 8]),
            AngleType::new(0.5),
        ));

        let mir_op3 = MirOp::PauliMeasure(PauliMeasure::new(
            PauliString::new(vec![Pauli::I; 8]),
            false,
        ));

        assert_eq!(mir_op, mir_op2);
        assert_ne!(mir_op2, mir_op3);
    }
}
