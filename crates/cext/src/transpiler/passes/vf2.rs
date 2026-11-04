// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use qiskit_circuit::VirtualQubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::vf2::{
    Vf2PassConfiguration, Vf2PassReturn, vf2_layout_pass_average, vf2_layout_pass_exact,
};
use qiskit_transpiler::target::Target;

/// The result from ``qk_transpiler_pass_standalone_vf2_layout_average()`` and
/// ``qk_transpile_pass_standalone_vf2_layout_exact()``.
pub struct VF2LayoutResult(Vf2PassReturn);

/// @ingroup QkVF2LayoutResult
/// Check whether a result was found.
///
/// A ``true`` value includes the situation where the configuration specified to try the "trivial"
/// layout and it was found to be the best (and consequently no qubit relabelling is necessary,
/// other than ancilla expansion if appropriate).  See ``qk_vf2_layout_result_has_improvement`` to
/// distinguish whether an explicit remapping is stored.
///
/// @param layout a pointer to the layout
///
/// @returns ``true`` if the VF2-based layout pass found any match.
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkVF2LayoutResult``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_vf2_layout_result_has_match(layout: *const VF2LayoutResult) -> bool {
    // SAFETY: per documentation this is a valid pointer to a layout.
    let layout = unsafe { &*layout };
    layout.0 != Vf2PassReturn::NoSolution
}

/// @ingroup QkVF2LayoutResult
/// Check whether the result is an improvement to the trivial layout.
///
/// @param layout a pointer to the layout
///
/// @returns ``true`` if the VF2-based layout pass found an improved match.
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkVF2LayoutResult``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_vf2_layout_result_has_improvement(
    layout: *const VF2LayoutResult,
) -> bool {
    // SAFETY: per documentation this is a valid pointer to a layout.
    let layout = unsafe { &*layout };
    matches!(layout.0, Vf2PassReturn::Solution(_))
}

/// @ingroup QkVF2LayoutResult
/// Get the physical qubit for a given virtual qubit
///
/// @param layout a pointer to the layout
/// @param qubit the virtual qubit to get the physical qubit of
///
/// @returns The physical qubit mapped to by the specified virtual qubit
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkVF2LayoutResult`` containing a result, or if the qubit is out of range for the initial
/// circuit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_vf2_layout_result_map_virtual_qubit(
    layout: *const VF2LayoutResult,
    qubit: u32,
) -> u32 {
    // SAFETY: per documentation this is a valid pointer to a layout.
    let layout = unsafe { &*layout };
    match &layout.0 {
        Vf2PassReturn::NoSolution => panic!("There was no layout found!"),
        // It's undefined behaviour to pass a qubit that's out of range, so it's not our problem
        // that we can't tell if this is valid.
        Vf2PassReturn::NoImprovement => qubit,
        Vf2PassReturn::Solution(mapping) => mapping[&VirtualQubit::new(qubit)].0,
    }
}

/// @ingroup QkVF2LayoutResult
/// Free a ``QkVF2LayoutResult`` object
///
/// @param layout a pointer to the layout to free
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(1, 0);
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a ``QkVF2Layout``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_vf2_layout_result_free(layout: *mut VF2LayoutResult) {
    if !layout.is_null() {
        if !layout.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }
        // SAFETY: We have verified the pointer is non-null and aligned, so
        // it should be readable by Box.
        unsafe {
            let _ = Box::from_raw(layout);
        }
    }
}

/// A set of configurations for the VF2 layout passes.
///
/// See the setter methods associated with this `struct` for the available configurations.
pub struct VF2LayoutConfiguration(Vf2PassConfiguration);
/// @ingroup QkVF2LayoutConfiguration
/// Create a new configuration for the VF2 passes that runs everything completely unbounded.
///
/// Call ``qk_vf2_layout_configuration_free`` with the return value to free the memory when done.
///
/// @return A pointer to the configuration.
#[unsafe(no_mangle)]
pub extern "C" fn qk_vf2_layout_configuration_new() -> *mut VF2LayoutConfiguration {
    Box::into_raw(Box::new(VF2LayoutConfiguration(
        Vf2PassConfiguration::default_unbounded(),
    )))
}
/// @ingroup QkVF2LayoutConfiguration
/// Free a `QkVf2LayoutConfiguration` object.
///
/// @param config A pointer to the configuration.
///
/// # Safety
///
/// Behavior is undefined if ``config`` is a non-null pointer, but does not point to a valid,
/// aligned `QkVF2LayoutConfiguration` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_vf2_layout_configuration_free(config: *mut VF2LayoutConfiguration) {
    if !config.is_null() {
        if !config.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.");
        }
        // SAFETY: per documentation and above checks, this points to valid data that was previously
        // boxed.
        let _ = unsafe { Box::from_raw(config) };
    }
}
/// @ingroup QkVF2LayoutConfiguration
/// Limit the numbers of times that the VF2 algorithm will attempt to extend its mapping before and
/// after it finds the first match.
///
/// The VF2 algorithm keeps track of the number of steps it has taken, and terminates when it
/// reaches the limit.  After the first match is found, the limit swaps from the "before" limit to
/// the "after" limit without resetting the number of steps taken.
///
/// @param config The configuration to update.
/// @param before The number of attempts to allow before the first match is found.  Set to a
///     negative number to have no bound.
/// @param after The number of attempts to allow after the first match (if any) is found.  Set to a
///     negative number to have no bound.
///
/// # Safety
///
/// Behavior is undefined if `config` is not a valid, aligned, non-null pointer to a
/// `QkVF2LayoutConfiguration`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_vf2_layout_configuration_set_call_limit(
    config: *mut VF2LayoutConfiguration,
    before: i64,
    after: i64,
) {
    let lift = |limit: i64| usize::try_from(limit).ok();
    // SAFETY: per documentation this is a valid configuration pointer.
    unsafe { (*config).0.call_limit = (lift(before), lift(after)) };
}
/// @ingroup QkVF2LayoutConfiguration
/// Limit the runtime of the VF2 search.
///
/// This is not a hard limit; it is only checked when an improved layout is encountered.  Using this
/// option also makes the pass non-deterministic. It is generally recommended to use
/// `qk_vf2_layout_configuration_set_call_limit` instead.
///
/// @param config The configuration to update.
/// @param limit The time in seconds to allow.  Set to a non-positive value to run with no limit.
///
/// # Safety
///
/// Behavior is undefined if `config` is not a valid, aligned, non-null pointer to a
/// `QkVF2LayoutConfiguration`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_vf2_layout_configuration_set_time_limit(
    config: *mut VF2LayoutConfiguration,
    limit: f64,
) {
    // SAFETY: per documentation this is a valid pointer to a configuration.
    unsafe { (*config).0.time_limit = (limit > 0.0).then_some(limit) };
}
/// @ingroup QkVF2LayoutConfiguration
/// Limit the total number of complete improvements found.
///
/// Since the VF2 search tree is pruned on-the-fly based on scoring in the `QkTarget`, this limit
/// is not especially powerful.  See `qk_vf2_layout_configuration_set_call_limit` for a tighter
/// bound.
///
/// @param config The configuration to update.
/// @param limit The number of complete layouts to allow before terminating.  Set to 0 to run
///     unbounded.
///
/// # Safety
///
/// Behavior is undefined if `config` is not a valid, aligned, non-null pointer to a
/// `QkVF2LayoutConfiguration`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_vf2_layout_configuration_set_max_trials(
    config: *mut VF2LayoutConfiguration,
    limit: u64,
) {
    // SAFETY: per documentation this is a valid pointer to a configuration.
    unsafe { (*config).0.max_trials = Some(limit.try_into().unwrap_or(usize::MAX)) };
}
/// @ingroup QkVF2LayoutConfiguration
/// Activate node shuffling of the input graphs with a given seed.
///
/// This effectively drives a modification of the matching order of VF2, which in theory means that
/// the space of a bounded search is not biased based on the node indices.  In practice, Qiskit uses
/// the VF2++ ordering improvements when running in "average" mode (corresponding to initial layout
/// search), and starts from the identity mapping in "exact" made.  Both of these ordering
/// heuristics are typically far more likely to find results for the given problem than
/// randomization.
///
/// If this function was not called, no node shuffling takes place.
///
/// @param config The configuration to update.
/// @param seed The seed to use for the activated shuffling.
///
/// # Safety
///
/// Behavior is undefined if `config` is not a valid, aligned, non-null pointer to a
/// `QkVF2LayoutConfiguration`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_vf2_layout_configuration_set_shuffle_seed(
    config: *mut VF2LayoutConfiguration,
    seed: u64,
) {
    // SAFETY: per documentation this is a valid pointer to a configuration.
    unsafe { (*config).0.shuffle_seed = Some(seed) };
}
/// @ingroup QkVF2LayoutConfiguration
/// Whether to eagerly score the initial "trivial" layout of the interaction graph.
///
/// You typically want to set this ``true`` if you are using the VF2 passes to improve a circuit
/// that is already lowered to hardware, in order to set a baseline for the score-based pruning.  If
/// not, you can leave this as ``false`` (the default), to avoid a calculation that likely will not
/// have any impact.
///
/// @param config The configuration to update.
/// @param score_initial Whether to eagerly score the initial trivial layout.
///
/// # Safety
///
/// Behavior is undefined if `config` is not a valid, aligned, non-null pointer to a
/// `VF2LayoutConfiguration`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_vf2_layout_configuration_set_score_initial(
    config: *mut VF2LayoutConfiguration,
    score_initial: bool,
) {
    unsafe { (*config).0.score_initial_layout = score_initial };
}

/// @ingroup QkTranspilerPassesStandalone
/// Use the VF2 algorithm to choose a layout (if possible) for the input circuit, using a
/// noise-aware scoring heuristic based only on hardware error rates, and not the specific gates in
/// the circuit.
///
/// This function corresponds to the Python-space ``VF2Layout`` pass.
///
/// This function is suitable for use on circuits that have not yet been fully lowered to hardware.
/// If your circuit has already been completely lowered to hardware and you are looking to _improve_
/// the layout for an exact interaction graph, use ``qk_transpile_pass_standalone_vf2_layout_exact``
/// instead.
///
/// If this pass finds a solution that means there is a "perfect layout" and that no
/// further swap mapping or routing is needed. However, there is not always a possible
/// solution, or a solution might exist but it is not found within the limits specified
/// when the pass is called.
///
/// @param circuit A pointer to the circuit to run VF2Layout on
/// @param target A pointer to the target to run the VF2Layout pass on
/// @param config A pointer to the ``QkVF2LayoutConfiguration`` configuration structure.  If this
///     pointer is null, the pass defaults are used.
/// @param strict_direction If ``true``, the pass will consider the edge direction in the
///     connectivity described in the ``target``. Typically, setting this to ``false``
///     is desirable as the error heuristic is already very approximate, and two-qubit gates can
///     almost invariably be synthesised to "flip" direction using only local one-qubit gates and
///     the native-direction two-qubit gate.
///
/// @return A pointer to a result object that contains the results of the pass. This object is heap
///     allocated and will need to be freed with the `qk_vf2_layout_result_free` function.
///
/// # Example
///
/// ```c
/// QkTarget *target = qk_target_new(5);
/// uint32_t current_num_qubits = qk_target_num_qubits(target);
/// QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
/// for (uint32_t i = 0; i < current_num_qubits - 1; i++) {
///     uint32_t qargs[2] = {i, i + 1};
///     double inst_error = 0.0090393 * (current_num_qubits - i);
///     double inst_duration = 0.020039;
///     qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
/// }
/// QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
/// QkCircuit *qc = qk_circuit_new(4, 0);
/// for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
///     uint32_t qargs[2] = {i, i + 1};
///     for (uint32_t j = 0; j<i+1; j++) {
///         qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
///     }
/// }
/// QkVF2LayoutConfiguration *config = qk_vf2_layout_configuration_new();
/// qk_vf2_layout_configuration_set_call_limit(config, 10000, 10000);
/// QkVF2LayoutResult *layout_result = qk_transpiler_pass_standalone_vf2_layout_average(qc, target, config, false);
/// qk_vf2_layout_result_free(layout_result);
/// qk_vf2_layout_configuration_free(config);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, non-null pointer to a
/// `QkCircuit` and `QkTarget`.  Behavior is undefined if ``config`` is a non-null pointer that
/// does not point to a valid `QkVF2LayoutConfiguration` object (but a null pointer is fine).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_vf2_layout_average(
    circuit: *const CircuitData,
    target: *const Target,
    config: *const VF2LayoutConfiguration,
    strict_direction: bool,
) -> *mut VF2LayoutResult {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { &*circuit };
    let target = unsafe { &*target };
    let dag = match DAGCircuit::from_circuit_data(circuit, false, None, None) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    let config_default = Vf2PassConfiguration::default_abstract();
    let config = if config.is_null() {
        &config_default
    } else {
        // SAFETY: per documentation this is a valid pointer to a configuration.
        unsafe { &(*config).0 }
    };
    vf2_layout_pass_average(&dag, target, config, strict_direction, None)
        .map(|result| Box::into_raw(Box::new(VF2LayoutResult(result))))
        .unwrap()
}

/// @ingroup QkTranspilerPassesStandalone
/// Use the VF2 algorithm to choose a layout (if possible) for the input circuit, using a
/// noise-aware scoring heuristic that requires the result is already fully compatible with
/// the hardware.
///
/// This function corresponds to the Python-space ``VF2PostLayout`` pass with
/// ``strict_direction=True``.
///
/// This function is suitable for use on circuits that have already been fully lowered to hardware,
/// and you are now looking to see if a qubit permutation can lead to better estimated error rates.
/// If your circuit is still in terms of non-hardware-supported operations, use
/// `qk_transpile_pass_standalone_vf2_layout_average` instead.
///
/// Typically, you call this pass after layout, routing, translation to a native basis set and
/// optimization, such that the input circuit is already executable on hardware with the qubit
/// indices referring to physical qubits.  The pass will return a result indicating one of:
///
/// * there is a better choice of initial virtual-to-physical qubit mapping than what the circuit is
///   currently using.
/// * the current choice of physical qubits is the best the pass found within its call limit.
/// * there is no valid choice of virtual-to-physical mapping that results in an executable circuit
///   (or at least, the pass failed to find one within its specified limits).
///
/// In both of the first two cases, `qk_vf2_layout_has_match` will return ``true``.  In only the
/// first case, `qk_vf2_layout_has_improvement` will return ``true``.
///
/// @param circuit A pointer to the circuit to run the layout search on.
/// @param target A pointer to the target representing the QPU.
/// @param config A pointer to the `QkVF2LayoutConfiguration` configuration structure.  If this
///     pointer is null, the pass defaults are used.
///
/// @return A pointer to a result object that contains the results of the pass. This object is heap
///     allocated and will need to be freed with the `qk_vf2_layout_result_free` function.
///
/// # Example
///
/// ```c
/// QkTarget *target = qk_target_new(5)
/// QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
/// for (uint32_t i = 0; i < current_num_qubits - 1; i++) {
///     uint32_t qargs[2] = {i, i + 1};
///     double inst_error = 0.0090393 * (current_num_qubits - i);
///     double inst_duration = 0.020039;
///     qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
/// }
/// QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
/// QkCircuit *qc = qk_circuit_new(4, 0);
/// for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
///     uint32_t qargs[2] = {i, i + 1};
///     for (uint32_t j = 0; j<i+1; j++) {
///         qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
///     }
/// }
/// QkVF2LayoutConfiguration *config = qk_vf2_layout_configuration_new();
/// qk_vf2_layout_configuration_call_limit(config, 10000, 10000);
/// QkVF2LayoutResult *layout_result = qk_transpiler_pass_standalone_vf2_layout_exact(qc, target, config);
/// qk_vf2_layout_result_free(layout_result);
/// qk_vf2_layout_configuration_free(config);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, non-null pointer to a
/// `QkCircuit` and `QkTarget`.  Behavior is undefined if ``config`` is a non-null pointer that
/// does not point to a valid `QkVF2LayoutConfiguration` object (but a null pointer is fine).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_vf2_layout_exact(
    circuit: *const CircuitData,
    target: *const Target,
    config: *const VF2LayoutConfiguration,
) -> *mut VF2LayoutResult {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { &*circuit };
    let target = unsafe { &*target };
    let dag = match DAGCircuit::from_circuit_data(circuit, false, None, None) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    let config_default = Vf2PassConfiguration::default_concrete();
    let config = if config.is_null() {
        &config_default
    } else {
        unsafe { &(*config).0 }
    };
    vf2_layout_pass_exact(&dag, target, config)
        .map(|result| Box::into_raw(Box::new(VF2LayoutResult(result))))
        .unwrap()
}

/// @ingroup QkTranspilerPassesStandalone
/// Deprecated version of `qk_transpiler_pass_standalone_vf2_layout_average`.
///
/// This legacy interface does not use `QkVf2LayoutConfiguration`, and has a name that is not clear
/// about how it handles the error heuristic (it averages over all gates in the `QkTarget` for a
/// given qubit or link).
///
/// \qk_deprecated{2.3.0|Replaced by :c:func:`qk_transpiler_pass_standalone_vf2_layout_average`.}
///
/// @param circuit As in `qk_transpiler_pass_standalone_vf2_layout_average`.
/// @param target As in `qk_transpiler_pass_standalone_vf2_layout_average`.
/// @param strict_direction As in `qk_transpiler_pass_standalone_vf2_layout_average`.
/// @param call_limit As in `qk_vf2_layout_configuration_set_call_limit`, but the same value is used
///     for both `before` and `after`.
/// @param time_limit As in `qk_vf2_layout_configuration_set_time_limit`.
/// @param max_trials As in `qk_vf2_layout_configuration_set_max_trials`.
///
/// @return As in `qk_transpiler_pass_standalone_vf2_layout_average`.
///
/// # Safety
///
/// The safety requirements of `qk_transpiler_pass_standalone_vf2_layout_average` must be respected
/// for `circuit` and `target`.
/// cbindgen:qk-vtable-rules=[no-export]
#[deprecated(
    since = "2.3.0",
    note = "use `qk_transpiler_pass_standalone_vf2_layout_average` instead"
)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_vf2_layout(
    circuit: *const CircuitData,
    target: *const Target,
    strict_direction: bool,
    call_limit: i64,
    time_limit: f64,
    max_trials: i64,
) -> *mut VF2LayoutResult {
    let call_limit: Option<usize> = call_limit.try_into().ok();
    let max_trials = if max_trials == 0 {
        None
    } else {
        max_trials.try_into().ok().or(Some(0))
    };
    let config = VF2LayoutConfiguration(Vf2PassConfiguration {
        call_limit: (call_limit, call_limit),
        time_limit: (time_limit > 0.0).then_some(time_limit),
        max_trials,
        shuffle_seed: None,
        score_initial_layout: false,
    });
    // SAFETY: this function is a deprecated thin wrapper around `_average`, and per documentation
    // the caller has upheld the requirements of that function.  `config` is safe to point to as it
    // is constructed in safe code and lasts for the duration of this function.
    unsafe {
        qk_transpiler_pass_standalone_vf2_layout_average(circuit, target, &config, strict_direction)
    }
}
