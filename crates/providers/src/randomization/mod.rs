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

//! Generate randomizations of quantum circuits and programs.

pub mod annotations;
pub mod emissions;
pub mod partition;
pub mod virtual_type;

use partition::Partition;
use virtual_type::VirtualType;

/// The synthesizer to use for virtual gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SynthesizerType {
    RzSx,
    RzRx,
}

impl SynthesizerType {
    /// The number of qubits the synthesized gate acts on.
    pub fn num_qubits(&self) -> u32 {
        match self {
            Self::RzSx | Self::RzRx => 1,
        }
    }

    /// Whether this synthesizer can synthesize a given virtual type.
    pub fn accepts(&self, _virtual_type: VirtualType) -> bool {
        match self {
            Self::RzSx | Self::RzRx => true,
        }
    }
}

/// The distribution to sample virtual gates from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributionType {
    UniformPauli,
    BalancedUniformPauli,
    UniformPauliSubset,
    UniformC1,
    UniformLocalC1,
    HaarU2,
}

impl DistributionType {
    /// The algebraic type of the sampled virtual gates.
    pub fn virtual_type(&self) -> VirtualType {
        match self {
            Self::UniformPauli | Self::BalancedUniformPauli | Self::UniformPauliSubset => {
                VirtualType::Pauli
            }
            Self::UniformC1 | Self::UniformLocalC1 => VirtualType::C1,
            Self::HaarU2 => VirtualType::U2,
        }
    }

    /// How samples from this distribution factorize.
    pub fn factorization(&self) -> Partition<u32> {
        match self {
            Self::UniformPauli
            | Self::BalancedUniformPauli
            | Self::UniformPauliSubset
            | Self::UniformC1
            | Self::HaarU2 => Partition::whole([0]),
            Self::UniformLocalC1 => Partition::singletons([0, 1]),
        }
        .expect("a distribution's factorization covers at least one qubit")
    }

    /// The total number of qubits.
    pub fn num_qubits(&self) -> u32 {
        self.factorization().num_elements() as u32
    }

    /// The number of factors in this distribution.
    pub fn num_slots(&self) -> usize {
        self.factorization().num_parts()
    }
}

/// Which way virtual state flows temporally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Forward,
    Backward,
}
