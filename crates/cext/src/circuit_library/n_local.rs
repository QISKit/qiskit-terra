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

use crate::pointers::ExposesOwnedPointers;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit_library::blocks::{Block, Entanglement};
use qiskit_circuit_library::entanglement::get_entanglement_from_str;
use qiskit_circuit_library::multi_local::n_local;
use std::ffi::{CStr, c_char};

/// Settings for generating a n-local variational circuit.
///
/// This contains the entanglement strategy (``entanglement_strategy``),
/// the repetitions of the rotation and entanglement layers (``reps``),
/// the parameter prefix for gates that have parameters (``parameter_prefix``),
/// whether it's wanted to add barriers after each layer (``insert_barriers``)
/// and if it's wanted to skip the last layer (``skip_final_rotation_layer``)
///
/// It's the responsibility of the user that the data is coherent,
/// see also the below section on safety.
///
/// # Safety
///
/// ``parameter_prefix`` must be a pointer to memory that contains
/// a valid nul terminator at the end of the string. It also must be valid for reads of
/// bytes up to and including the nul terminator.
#[repr(C)]
pub struct NLocalSettings {
    /// The entanglement strategy to be used in the generated circuit.
    /// See ``QkEntanglementStrategy`` for more details.
    entanglement_strategy: EntanglementStrategy,
    /// Specifies how often the rotation blocks and entanglement blocks are repeated.
    reps: usize,
    /// The prefix used for the default parameters generated.
    parameter_prefix: *const c_char,
    /// If ``true``, barriers are inserted in between each layer.
    /// If ``false``, no barriers are inserted.
    insert_barriers: bool,
    /// Whether a final rotation layer is added to the circuit.
    skip_final_rotation_layer: bool,
}

impl Default for NLocalSettings {
    fn default() -> Self {
        Self {
            entanglement_strategy: EntanglementStrategy::Full,
            reps: 3,
            parameter_prefix: std::ptr::null_mut(),
            insert_barriers: false,
            skip_final_rotation_layer: false,
        }
    }
}

/// Enum representing all the possible strategies for n_local entanglement blocks.
#[repr(u8)]
pub enum EntanglementStrategy {
    /// Entanglement where each qubit is entangled with all the others.
    Full = 0,
    /// Entanglement where qubit \f$i\f$ is entangled with qubit \f$i + 1\f$
    /// for all \f$i \in \{0, 1, ... , n - 2\}\f$ where \f$n\f$ is the total
    /// number of qubits.
    Linear = 1,
    /// Entanglement where qubit \f$i\f$ is entangled with qubit \f$i + 1\f$
    /// for all \f$i \in \{n-2, n-3, ... , 1, 0\}\f$ where \f$n\f$ is the total
    /// number of qubits.
    ReverseLinear = 2,
    /// Entanglement that behaves the same as linear entanglement but with an additional
    /// entanglement of the first and last qubit before the linear part.
    Circular = 3,
    /// Entanglement where one layer contains a qubit \f$i\f$ entangled with
    /// qubit \f$i + 1\f$, for all even values of \f$i\f$, and then a second layer
    /// where a qubit \f$i\f$ is entangled with qubit \f$i + 1\f$, for all odd values of
    /// \f$i\f$.
    Pairwise = 4,
    /// Entanglement shifted-circular-alternating. It's a generalized and modified
    /// version of the proposed circuit 14 in
    /// [Sim et al.](https://arxiv.org/abs/1905.10876)
    /// It consists of circular entanglement where the "long" entanglement connecting
    /// the first with the last qubit is shifted by one each block.  
    /// Furthermore the role of control and target qubits are swapped every block
    /// (therefore alternating).
    Sca = 5,
}

/// @ingroup QkCircuitLibrary
/// Construct an n-local variational circuit.
///
/// The structure of the n-local circuit are alternating rotation and entanglement layers.
/// In both layers, parameterized circuit-blocks act on the circuit in a defined way.
/// In the rotation layer, the blocks are applied stacked on top of each other, while in the
/// entanglement layer according to the ``entanglement`` strategy.
/// The circuit blocks can have arbitrary sizes (smaller equal to the number of qubits in the
/// circuit). Each layer is repeated ``reps`` times, and by default a final rotation layer is
/// appended.
///
/// For instance, a rotation block on 2 qubits and an entanglement block on 4 qubits using
/// ``QkEntanglementStrategy_Linear`` entanglement yields the following circuit.
///
/// ```
///
///     ┌──────┐ ░ ┌──────┐                      ░ ┌──────┐
///     ┤0     ├─░─┤0     ├──────────────── ... ─░─┤0     ├
///     │  Rot │ ░ │      │┌──────┐              ░ │  Rot │
///     ┤1     ├─░─┤1     ├┤0     ├──────── ... ─░─┤1     ├
///     ├──────┤ ░ │  Ent ││      │┌──────┐      ░ ├──────┤
///     ┤0     ├─░─┤2     ├┤1     ├┤0     ├ ... ─░─┤0     ├
///     │  Rot │ ░ │      ││  Ent ││      │      ░ │  Rot │
///     ┤1     ├─░─┤3     ├┤2     ├┤1     ├ ... ─░─┤1     ├
///     ├──────┤ ░ └──────┘│      ││  Ent │      ░ ├──────┤
///     ┤0     ├─░─────────┤3     ├┤2     ├ ... ─░─┤0     ├
///     │  Rot │ ░         └──────┘│      │      ░ │  Rot │
///     ┤1     ├─░─────────────────┤3     ├ ... ─░─┤1     ├
///     └──────┘ ░                 └──────┘      ░ └──────┘
///     
///     |                                 |
///     +---------------------------------+
///            repeated reps times
///
/// ```
/// Entanglement:
///
/// The entanglement describes the connections of the gates in the entanglement layer.
/// For a two-qubit gate for example, the entanglement contains pairs of qubits on which the
/// gate should acts, e.g. ``[[ctrl0, target0], [ctrl1, target1], ...]``.
/// To know more about the available entanglement strategies see ``QkEntanglementStrategy``.
///
/// @param num_qubits The number of qubits of the circuit.
/// @param rotation_blocks The blocks used in the rotation layers.
/// @param rotation_blocks_size Length of the array of rotation blocks provided.
/// @param entanglement_blocks The blocks used in the entanglement layers.
/// @param entanglement_blocks_size Length of the list of entanglement blocks provided.
/// @param settings A ``QkNLocalSettings`` pointer that is the settings to be applied
///    to the generated circuit. If ``NULL``, the default settings are used,
///    see ``QkNLocalSettings`` for more details.
///
/// @return A pointer to the generated circuit.
///
/// # Example
/// ```c
/// size_t num_qubits = 2;
/// QkGate rotation_blocks[1] = {QkGate_H};
/// QkGate entanglement_blocks[1] = {QkGate_CRX};
///
/// QkNLocalSettings settings = qk_circuit_library_n_local_settings_default();
/// settings.reps = 2;
/// // For this example we use QkEntanglementStrategy_Linear since
/// // the default is QkEntanglementStrategy_Full
/// settings.entanglement_strategy = QkEntanglementStrategy_Linear;
///
/// QkCircuit *qc = qk_circuit_library_n_local(num_qubits, rotation_blocks, 1,
///                                            entanglement_blocks, 1, &settings);
///
/// qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// * Behavior is undefined if ``rotation_blocks`` is not a valid, non-null pointer
///   to a sequence of ``rotation_blocks_size`` consecutive elements of ``StandardGate``.
/// * Behavior is undefined if ``entanglement_blocks`` is not a valid, non-null pointer
///   to a sequence of ``entanglement_blocks_size`` consecutive elements of ``StandardGate``.
/// * Behavior is undefined if ``settings`` is not a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_library_n_local(
    num_qubits: u32,
    rotation_blocks: *const StandardGate,
    rotation_blocks_size: usize,
    entanglement_blocks: *const StandardGate,
    entanglement_blocks_size: usize,
    settings: *const NLocalSettings,
) -> *mut CircuitData {
    // SAFETY: per documentation, `rotation_blocks` is aligned and and points to a valid
    // aligned block of memory valid for `num_rotation_blocks` reads of `StandardGate`s
    let rotation_blocks: Vec<Block> =
        unsafe { ::std::slice::from_raw_parts(rotation_blocks, rotation_blocks_size) }
            .iter()
            .map(|&gate| Block::from_standard_gate(gate))
            .collect();
    let rotation_blocks: Vec<&Block> = rotation_blocks.iter().collect();

    // SAFETY: per documentation, `entanglement_blocks` is aligned and and points to a valid
    // aligned block of memory valid for `num_entanglement_blocks` reads of `StandardGate`s
    let entanglement_blocks: Vec<Block> =
        unsafe { ::std::slice::from_raw_parts(entanglement_blocks, entanglement_blocks_size) }
            .iter()
            .map(|&gate| Block::from_standard_gate(gate))
            .collect();
    let entanglement_blocks: Vec<&Block> = entanglement_blocks.iter().collect();

    let settings = if settings.is_null() {
        &NLocalSettings::default()
    } else {
        // SAFETY: Per documentation, the pointer is non-null and aligned.
        unsafe { &*settings }
    };

    // Create entanglement
    let entanglement_strategy = match settings.entanglement_strategy {
        EntanglementStrategy::Full => "full",
        EntanglementStrategy::Linear => "linear",
        EntanglementStrategy::ReverseLinear => "reverse_linear",
        EntanglementStrategy::Sca => "sca",
        EntanglementStrategy::Circular => "circular",
        EntanglementStrategy::Pairwise => "pairwise",
    };
    let is_sca = matches!(settings.entanglement_strategy, EntanglementStrategy::Sca);

    let entanglement = match (0..settings.reps)
        .map(|layer| {
            // Create entanglement layer
            entanglement_blocks
                .iter()
                .map(|gate| {
                    // Create block entanglement
                    get_entanglement_from_str(
                        num_qubits,
                        gate.num_qubits,
                        entanglement_strategy,
                        if is_sca { layer } else { 0 },
                    )
                    .map(|connection| connection.collect())
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(entanglement_vector) => Entanglement {
            entanglement_vec: entanglement_vector,
        },
        Err(_) => return std::ptr::null_mut(),
    };

    n_local(
        num_qubits,
        &rotation_blocks,
        &entanglement_blocks,
        &entanglement,
        settings.reps,
        settings.insert_barriers,
        if settings.parameter_prefix.is_null() {
            "θ"
        } else {
            // SAFETY: Per documentation, the pointer is non-null and aligned.
            unsafe {
                CStr::from_ptr(settings.parameter_prefix)
                    .to_str()
                    .expect("Invalid UTF-8 character")
            }
        },
        settings.skip_final_rotation_layer,
        false,
    )
    .map(ExposesOwnedPointers::into_leaked)
    .unwrap_or_else(|_| std::ptr::null_mut())
}

/// @ingroup QkCircuitLibrary
///
/// Generate default options for ``qk_circuit_library_n_local``.
///
/// @return A ``QkNLocalSettings`` object with default settings.
#[unsafe(no_mangle)]
pub extern "C" fn qk_circuit_library_n_local_settings_default() -> NLocalSettings {
    NLocalSettings::default()
}
