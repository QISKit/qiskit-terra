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

//! Emit and collect operations.

use std::sync::Arc;

use uuid::Uuid;

use qiskit_circuit::operations::{CustomOperation, Operation};
use qiskit_circuit::packed_instruction::PackedInstruction;

use super::virtual_type::VirtualType;
use super::{Direction, DistributionType, SynthesizerType};

/// The name of the emit operation.
pub const EMIT_NAME: &str = "emit";

/// The name of the collect operation.
pub const COLLECT_NAME: &str = "collect";

/// A distribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Distribution {
    Noise {
        reference: Box<str>,
        modifier: Option<Box<str>>,
        num_qubits: u32,
    },
    Sample(DistributionType),
}

impl Distribution {
    /// How many virtual gates each sample decomposes into.
    pub fn num_slots(&self) -> usize {
        match self {
            Self::Noise { num_qubits, .. } => *num_qubits as usize,
            Self::Sample(distribution) => distribution.num_slots(),
        }
    }

    /// The width of the virtual gate of `slot`, or `None` if it has no such slot.
    pub fn slot_qubits(&self, slot: u32) -> Option<u32> {
        match self {
            // Every slot of a noise map is one qubit's local Pauli.
            Self::Noise { num_qubits, .. } => (slot < *num_qubits).then_some(1),
            Self::Sample(distribution) => distribution
                .factorization()
                .part(slot as usize)
                .map(|part| part.len() as u32),
        }
    }

    /// The virtual type.
    pub fn virtual_type(&self) -> VirtualType {
        match self {
            Self::Noise { .. } => VirtualType::Pauli,
            Self::Sample(distribution) => distribution.virtual_type(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emit {
    pub direction: Option<Direction>,
    /// What the sample this belongs to draws from.
    pub distribution: Arc<Distribution>,
    /// Which sample.
    pub uuid: Uuid,
    /// Which part of that sample's factorization this emission places.
    pub slot: u32,
    /// Whether to invert the sampled value before composing it.
    pub adjoint: bool,
}

impl Operation for Emit {
    fn name(&self) -> &str {
        EMIT_NAME
    }

    fn num_qubits(&self) -> u32 {
        self.distribution
            .slot_qubits(self.slot)
            .expect("an emission places a slot its distribution has")
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        0
    }

    fn directive(&self) -> bool {
        true
    }
}

impl CustomOperation for Emit {
    fn is_unitary(&self) -> bool {
        false
    }
}

pub type CollectStep = (Vec<u32>, PackedInstruction);

#[derive(Debug, Clone)]
pub struct Collect {
    /// How gathered gates are synthesized.
    pub synthesizer: SynthesizerType,
    pub steps: Vec<CollectStep>,
}

impl PartialEq for Collect {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Collect {
    pub fn new(synthesizer: SynthesizerType) -> Self {
        Collect {
            synthesizer,
            steps: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn accepts(&self, virtual_type: VirtualType) -> bool {
        self.synthesizer.accepts(virtual_type)
    }
}

impl Operation for Collect {
    fn name(&self) -> &str {
        COLLECT_NAME
    }

    fn num_qubits(&self) -> u32 {
        self.synthesizer.num_qubits()
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        0
    }

    fn directive(&self) -> bool {
        true
    }
}

impl CustomOperation for Collect {
    fn is_unitary(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Pauli-Lindblad map over `width` qubits which factorizes into one slot each.
    fn noise(width: u32) -> Distribution {
        Distribution::Noise {
            reference: "layer_1".into(),
            modifier: None,
            num_qubits: width,
        }
    }

    fn emission(distribution: Distribution, uuid: Uuid) -> Emit {
        emission_at(distribution, uuid, 0)
    }

    fn emission_at(distribution: Distribution, uuid: Uuid, slot: u32) -> Emit {
        Emit {
            direction: Some(Direction::Forward),
            distribution: Arc::new(distribution),
            uuid,
            slot,
            adjoint: false,
        }
    }

    #[test]
    fn test_an_emission_is_as_wide_as_one_slot() {
        let pauli = emission(
            Distribution::Sample(DistributionType::UniformPauli),
            Uuid::new_v4(),
        );
        assert_eq!(pauli.num_qubits(), 1);
        for slot in 0..2 {
            let local = emission_at(
                Distribution::Sample(DistributionType::UniformLocalC1),
                Uuid::new_v4(),
                slot,
            );
            assert_eq!(local.num_qubits(), 1, "a local-C1 slot is one qubit");
        }
        // A noise slot is one qubit's Pauli however wide the map is.
        assert_eq!(emission(noise(3), Uuid::new_v4()).num_qubits(), 1);
    }

    #[test]
    fn test_emissions_of_one_sample_share_a_uuid_and_differ_otherwise() {
        let uuid = Uuid::new_v4();
        let distribution = Distribution::Sample(DistributionType::UniformPauli);
        let near = emission(distribution, uuid);
        let far = Emit {
            adjoint: true,
            ..near.clone()
        };
        assert_eq!(near.uuid, far.uuid);
        assert_ne!(near, far);
    }

    #[test]
    fn test_two_uuids_from_one_distribution_are_independent() {
        let first = emission(
            Distribution::Sample(DistributionType::UniformPauli),
            Uuid::new_v4(),
        );
        let second = emission(
            Distribution::Sample(DistributionType::UniformPauli),
            Uuid::new_v4(),
        );
        assert_ne!(first.uuid, second.uuid);
        assert_ne!(first, second);
    }

    #[test]
    fn test_noise_emits_are_told_apart_by_their_slot() {
        let uuid = Uuid::new_v4();
        let distribution = Arc::new(noise(2));
        let first = Emit {
            direction: Some(Direction::Forward),
            distribution: Arc::clone(&distribution),
            uuid,
            slot: 0,
            adjoint: false,
        };
        let second = Emit {
            slot: 1,
            ..first.clone()
        };
        assert_eq!(first.uuid, second.uuid);
        assert!(Arc::ptr_eq(&first.distribution, &second.distribution));
        assert_ne!(first, second);
    }

    #[test]
    fn test_distribution_virtual_type() {
        assert_eq!(noise(1).virtual_type(), VirtualType::Pauli);
        assert_eq!(
            Distribution::Sample(DistributionType::HaarU2).virtual_type(),
            VirtualType::U2
        );
    }

    #[test]
    fn test_a_collector_starts_empty_and_as_wide_as_its_synthesizer() {
        let collector = Collect::new(SynthesizerType::RzSx);
        assert!(collector.is_empty());
        assert_eq!(collector.num_qubits(), 1);
        assert_eq!(collector.name(), COLLECT_NAME);
    }

    #[test]
    fn test_a_collector_accepts_what_its_synthesizer_can_express() {
        let collector = Collect::new(SynthesizerType::RzSx);
        for virtual_type in [VirtualType::Pauli, VirtualType::C1, VirtualType::U2] {
            assert!(collector.accepts(virtual_type), "{virtual_type:?}");
        }
    }

    #[test]
    fn test_neither_operation_is_unitary() {
        assert!(!emission(noise(0), Uuid::new_v4()).is_unitary());
        assert!(!Collect::new(SynthesizerType::RzRx).is_unitary());
    }
}
