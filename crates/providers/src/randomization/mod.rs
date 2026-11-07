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
pub mod partition;
pub mod virtual_type;

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
    /// The number qubits the sampled virtual gate acts on.
    pub fn num_qubits(&self) -> u32 {
        match self {
            Self::UniformPauli
            | Self::BalancedUniformPauli
            | Self::UniformPauliSubset
            | Self::UniformC1
            | Self::HaarU2 => 1,
            Self::UniformLocalC1 => 2,
        }
    }
}

/// Which way virtual state flows temporally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Forward,
    Backward,
}
