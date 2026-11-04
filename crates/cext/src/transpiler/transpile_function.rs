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

use std::ffi::c_char;
use std::ptr::null_mut;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::commutation_checker::get_standard_commutation_checker;
use qiskit_transpiler::passes::{UnitarySynthesisConfig, UnitarySynthesisState, unitary_synthesis};
use qiskit_transpiler::standard_equivalence_library::generate_standard_equivalence_library;
use qiskit_transpiler::target::Target;
use qiskit_transpiler::transpile;
use qiskit_transpiler::transpile_layout::TranspileLayout;
use qiskit_transpiler::transpiler::{
    LayoutSource, get_sabre_heuristic, init_stage, layout_stage, optimization_stage, routing_stage,
    translation_stage,
};

use crate::exit_codes::CInputError;
use crate::exit_codes::ExitCode;
use crate::pointers::check_ptr;
use crate::util;

/// The container result object from ``qk_transpile``
///
/// When the transpiler successfully compiles a quantum circuit for a given target it
/// returns the transpiled circuit and the layout. The ``qk_transpile`` function will
/// write pointers to the fields in this struct when it successfully executes, you can
/// initialize this struct with null pointers or leave them unset as the values are never
/// read by ``qk_transpile`` and only written to. After calling ``qk_transpile`` you are
/// responsible for calling ``qk_circuit_free`` and ``qk_transpile_layout_free`` on the
/// members of this struct.
#[repr(C)]
pub struct TranspileResult {
    /// The compiled circuit.
    circuit: *mut CircuitData,
    /// Metadata about the initial and final virtual-to-physical layouts.
    layout: *mut TranspileLayout,
}

/// A container collecting individual attributes shared by the transpiler stages.
///
/// When transpiling correctly, each individual stage writes specific attributes
/// to this container that will be needed by other stages in sequence. If the container
/// is not initialized, each stage will initialize a new object when necessary.
#[derive(Default)]
pub struct TranspilerStageState {
    /// Metadata about the initial and final virtual-to-physical layouts.
    pub layout: Option<TranspileLayout>,
}

/// @ingroup QkTranspilerStageState
/// Create a pointer to an empty ``QkTranspilerStageState`` object
///
/// @param state a pointer to the allocated memory space to store the
///     pointer in.
///
///
/// # Safety
///
/// Behavior is undefined if ``state`` is not a valid pointer allocated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_state_new(state: *mut *mut TranspilerStageState) {
    match check_ptr(state) {
        Ok(_) => (),
        Err(CInputError::AlignmentError) => {
            panic!("Tried referring to a QkTranspilerStageState in an unaligned pointer.")
        }
        _ => unreachable!(),
    }
    let transpile_state = TranspilerStageState::default();
    unsafe {
        // SAFETY: We have verified
        state.write(Box::into_raw(Box::new(transpile_state)));
    };
}

/// @ingroup QkTranspilerStageState
/// Free a ``QkTranspilerStageState`` object
///
/// @param state a pointer to the state to free
///
/// # Safety
///
/// Behavior is undefined if ``state`` is not either null or a valid pointer to a ``QkTranspilerStageState``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_state_free(state: *mut TranspilerStageState) {
    if !state.is_null() {
        if !state.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }
        // SAFETY: We have verified the pointer is non-null and aligned, so
        // it should be readable by Box.
        unsafe {
            let _ = Box::from_raw(state);
        }
    }
}

/// @ingroup QkTranspilerStageState
/// Obtains a ``QkTranspileLayout`` object from a ``QkTranspilerStageState`` object.
///
/// This pointer is owned by the ``state`` object and should not be freed using
/// ``qk_transpile_layout_free``. Instead, free the original ``state`` object using
/// ``qk_transpile_state_free``.
///
/// @param state a pointer to the state to retrieve the layout from.
///
/// @return a pointer to a ``QkTranspileLayout`` object owned by the state.
///
/// # Safety
///
/// Behavior is undefined if ``state`` is not a valid, non-null pointer to a ``QkTranspilerStageState``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_state_layout(
    state: *mut TranspilerStageState,
) -> *mut TranspileLayout {
    let borrowed_state = unsafe { &mut *state };

    match borrowed_state.layout.as_mut() {
        Some(layout) => layout,
        None => null_mut(),
    }
}

/// @ingroup QkTranspilerStageState
/// Sets a ``QkTranspileLayout`` object as the layout for a ``QkTranspilerStageState`` object.
///
/// Calling this method consumes the ``QkTranspileLayout`` object which means the user will
/// not need to call ``qk_transpile_layout_free``. The user should still de-allocate the
/// space allotted for it using ``free``.
///
///
/// @param state a pointer to the state to set the layout for.
/// @param layout a pointer to the layout to use. May be ``NULL`` to unset the layout.
///
/// # Safety
///
/// Behavior is undefined if ``state`` is not a valid, non-null pointer to a ``QkTranspilerStageState``.
/// Behavior is undefined if ``state`` is not a valid pointer to a ``QkTranspileLayout``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_state_layout_set(
    state: *mut TranspilerStageState,
    layout: *mut TranspileLayout,
) {
    // SAFETY: As per documentation, the pointer should be non-null and aligned.
    let state = unsafe { &mut *state };

    if layout.is_null() {
        state.layout = None;
    } else {
        // SAFETY: `layout` is non-null and aligned in `else` arm
        let layout = unsafe { Box::from_raw(layout) };
        state.layout = Some(*layout);
    }
}

/// The options for running the transpiler
#[repr(C)]
pub struct TranspileOptions {
    /// The optimization level to run the transpiler with. Valid values are 0, 1, 2, or 3.
    optimization_level: u8,
    /// The seed for the transpiler. If set to a negative number this means no seed will be
    /// set and the RNGs used in the transpiler will be seeded from system entropy.
    seed: i64,
    /// The approximation degree a heuristic dial where 1.0 means no approximation (up to numerical
    /// tolerance) and 0.0 means the maximum approximation. A `NAN` value indicates that
    /// approximation is allowed up to the reported error rate for an operation in the target.
    approximation_degree: f64,
}

impl Default for TranspileOptions {
    fn default() -> Self {
        TranspileOptions {
            optimization_level: 2,
            seed: -1,
            approximation_degree: 1.0,
        }
    }
}

/// @ingroup QkTranspiler
///
/// Generate transpiler options defaults
///
/// This function generates a QkTranspileOptions with the default settings
/// This currently is ``optimization_level`` 2, no seed, and no approximation.
///
/// @return A ``QkTranspileOptions`` object with default settings.
#[unsafe(no_mangle)]
pub extern "C" fn qk_transpiler_default_options() -> TranspileOptions {
    TranspileOptions::default()
}

/// @ingroup QkTranspiler
/// Run the preset init stage of the transpiler on a circuit
///
/// The Qiskit transpiler is a quantum circuit compiler that rewrites a given
/// input circuit to match the constraints of a QPU and optimizes the circuit
/// for execution. This function runs the first stage of the transpiler,
/// **init**, which runs abstract-circuit optimizations, and reduces multi-qubit
/// operations into one- and two-qubit operations. You can refer to
/// @verbatim embed:rst:inline :ref:`transpiler-preset-stage-init` @endverbatim for more details.
///
/// This function should only be used with circuits constructed
/// using Qiskit's C API. It makes assumptions on the circuit only using features exposed via C,
/// if you are in a mixed Python and C environment it is typically better to invoke the transpiler
/// via Python.
///
/// This function is multithreaded internally and will launch a thread pool
/// with threads equal to the number of CPUs reported by the operating system by default.
/// This will include logical cores on CPUs with simultaneous multithreading. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
/// ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.
///
/// @param dag A pointer to the circuit to run the transpiler on.
/// @param target A pointer to the target to compile the circuit for.
/// @param options A pointer to an options object that defines user options. If this is a null
///   pointer the default values will be used. See ``qk_transpile_default_options``
///   for more details on the default values.
/// @param state A pointer to a pointer to a ``QkTranspilerStageState`` object. On a successful
///   execution (return code 0) a pointer to the state object created transpiler will be written
///   to this pointer.
/// @param error A pointer to a pointer with an nul terminated string with an error description.
///   If the transpiler fails a pointer to the string with the error description will be written
///   to this pointer. That pointer needs to be freed with ``qk_str_free``. This can be a null
///   pointer in which case the error will not be written out.
///
/// @returns The return code for the transpiler, ``QkExitCode_Success`` means success and all
///   other values indicate an error.
///
/// # Safety
///
/// Behavior is undefined if ``dag``, ``target``, or ``state``, are not valid, non-null
/// pointers to a ``QkDag``, ``QkTarget``, or a ``QkTranspileLayout`` pointer
/// respectively. ``options`` must be a valid pointer a to a ``QkTranspileOptions`` or ``NULL``.
/// ``error`` must be a valid pointer to a ``char`` pointer or ``NULL``. The value of the inner
/// pointer for ``state`` will be overwritten by this function. If the value pointed to needs to
/// be freed this must be done outside of this function as it will not be freed by this function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_stage_init(
    dag: *mut DAGCircuit,
    target: *const Target,
    options: *const TranspileOptions,
    state: *mut *mut TranspilerStageState,
    error: *mut *mut c_char,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { &mut *dag };
    let target = unsafe { &*target };
    let mut out_layout = TranspileLayout::new(
        None,
        None,
        dag.qubits().objects().to_owned(),
        dag.num_qubits() as u32,
        dag.qregs().to_vec(),
    );
    let options = if options.is_null() {
        &TranspileOptions::default()
    } else {
        // SAFETY: We checked the pointer is not null, then, per documentation, it is a valid
        // and aligned pointer.
        unsafe { &*options }
    };

    let approximation_degree = if options.approximation_degree.is_nan() {
        None
    } else {
        if !(0.0..=1.0).contains(&options.approximation_degree) {
            panic!(
                "Invalid value provided for approximation degree, only NAN or values between 0.0 and 1.0 inclusive are valid"
            );
        }
        Some(options.approximation_degree)
    };
    let mut synthesis_state = UnitarySynthesisState::new(UnitarySynthesisConfig {
        approximation: unitary_synthesis::Approximation::from_py_approximation_degree(
            approximation_degree,
        ),
        run_python_decomposers: false,
        ..Default::default()
    });
    let mut commutation_checker = get_standard_commutation_checker();

    match init_stage(
        dag,
        target,
        options.optimization_level.into(),
        approximation_degree,
        &mut synthesis_state,
        &mut out_layout,
        &mut commutation_checker,
    ) {
        Ok(_) => {
            // SAFETY: Per the documentation result is a non-null aligned pointer to a pointer to
            // a QKTranspileLayout
            let deref_state = unsafe { *state };
            if deref_state.is_null() {
                unsafe {
                    *state = Box::into_raw(Box::new(TranspilerStageState {
                        layout: Some(out_layout),
                    }))
                }
            } else {
                let borrowed = unsafe { &mut **state };
                borrowed.layout.replace(out_layout);
            }
            ExitCode::Success
        }
        Err(e) => {
            if !error.is_null() {
                // Right now we return a backtrace of the error. This at least gives a hint as to
                // which pass failed when we have rust errors normalized we can actually have error
                // messages which are user facing. But most likely this will be a PyErr and panic
                // when trying to extract the string.
                let out_string = util::cstring_from_utf8_lossy(format!(
                    "Transpilation failed with this backtrace: {}",
                    e.backtrace()
                ))
                .into_raw();
                // SAFETY: Per documentation, error is a char* (and we checked it's not NULL)
                unsafe { *error = out_string };
            }
            ExitCode::TranspilerError
        }
    }
}

/// @ingroup QkTranspiler
/// Run the preset routing stage of the transpiler on a circuit
///
/// The Qiskit transpiler is a quantum circuit compiler that rewrites a given
/// input circuit to match the constraints of a QPU and optimizes the circuit
/// for execution. This function runs the third stage of the preset pass manager,
/// **routing**, which translates all the instructions in the circuit into
/// those supported by the target. You can refer to
/// @verbatim embed:rst:inline :ref:`transpiler-preset-stage-routing` @endverbatim for more details.
///
/// This function should only be used with circuits constructed
/// using Qiskit's C API. It makes assumptions on the circuit only using features exposed via C,
/// if you are in a mixed Python and C environment it is typically better to invoke the transpiler
/// via Python.
///
/// This function is multithreaded internally and will launch a thread pool
/// with threads equal to the number of CPUs reported by the operating system by default.
/// This will include logical cores on CPUs with simultaneous multithreading. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
/// ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.
///
/// @param dag A pointer to the circuit to run the transpiler on.
/// @param target A pointer to the target to compile the circuit for.
/// @param options A pointer to an options object that defines user options. If this is a null
///   pointer the default values will be used. See ``qk_transpile_default_options``
///   for more details on the default values.
/// @param state A pointer to a pointer to a ``QkTranspilerStageState`` object containing the layout.
///   Typically you will need to run the `qk_transpile_stage_layout` prior to this function and that
///   will provide a `QkTranspilerStageState` object with the initial layout set. You want to take that output state from
///   that function and use it as the input for this. If you don't have a layout object (e.g. you ran
///   your own layout pass). You can run ``qk_transpile_layout_generate_from_mapping`` to generate a trivial
///   layout (where virtual qubit 0 in the circuit is mapped to physical qubit 0 in the target,
///   1->1, 2->2, etc) for the dag at it's current state. This will enable you to generate a layout
///   object for the routing stage if you generate your own layout.
/// @param error A pointer to a pointer with an nul terminated string with an error description.
///   If the transpiler fails a pointer to the string with the error description will be written
///   to this pointer. That pointer needs to be freed with ``qk_str_free``. This can be a null
///   pointer in which case the error will not be written out.
///
/// @returns The return code for the transpiler, ``QkExitCode_Success`` means success and all
///   other values indicate an error.
///
/// # Safety
///
/// Behavior is undefined if ``dag``, ``target``, or ``layout``, are not valid, non-null
/// pointers to a ``QkDag``, ``QkTarget``, or a ``QkTranspileLayout`` pointer
/// respectively. ``options`` must be a valid pointer a to a ``QkTranspileOptions`` or ``NULL``.
/// ``error`` must be a valid pointer to a ``char`` pointer or ``NULL``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_stage_routing(
    dag: *mut DAGCircuit,
    target: *const Target,
    options: *const TranspileOptions,
    state: *mut TranspilerStageState,
    error: *mut *mut c_char,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers is non-null and aligned.
    let dag = unsafe { &mut *dag };
    let target = unsafe { &*target };
    let options = if options.is_null() {
        &TranspileOptions::default()
    } else {
        // SAFETY: We checked the pointer is not null, then, per documentation, it is a valid
        // and aligned pointer.
        unsafe { &*options }
    };
    let seed = if options.seed < 0 {
        None
    } else {
        Some(options.seed as u64)
    };
    let sabre_heuristic = match get_sabre_heuristic(target) {
        Ok(val) => val,
        Err(e) => {
            if !error.is_null() {
                // Right now we return a backtrace of the error. This at least gives a hint as to
                // which pass failed when we have rust errors normalized we can actually have error
                // messages which are user facing. But most likely this will be a PyErr and panic
                // when trying to extract the string.
                let out_string = util::cstring_from_utf8_lossy(format!(
                    "Transpilation failed with this backtrace: {}",
                    e.backtrace()
                ))
                .into_raw();
                // SAFETY: Per documentation, error is a char* (and we checked it's not NULL)
                unsafe { *error = out_string };
            }
            return ExitCode::TranspilerError;
        }
    };
    // SAFETY: Per the documentation this is a valid pointer to a transpile layout
    let borrowed_state = unsafe { &mut *state };
    let out_layout = borrowed_state
        .layout
        .as_mut()
        .expect("A layout should have been set by now.");
    match routing_stage(
        dag,
        target,
        options.optimization_level.into(),
        seed,
        &sabre_heuristic,
        out_layout,
        // Use Sabre here to ensure we run VF2PostLayout
        LayoutSource::Sabre,
    ) {
        Err(e) => {
            if !error.is_null() {
                // Right now we return a backtrace of the error. This at least gives a hint as to
                // which pass failed when we have rust errors normalized we can actually have error
                // messages which are user facing. But most likely this will be a PyErr and panic
                // when trying to extract the string.
                let out_string = util::cstring_from_utf8_lossy(format!(
                    "Transpilation failed with this backtrace: {}",
                    e.backtrace()
                ))
                .into_raw();
                unsafe {
                    *error = out_string;
                }
            }
            ExitCode::TranspilerError
        }
        Ok(_) => ExitCode::Success,
    }
}

/// @ingroup QkTranspiler
/// Run the preset optimization stage of the transpiler on a circuit
///
/// The Qiskit transpiler is a quantum circuit compiler that rewrites a given
/// input circuit to match the constraints of a QPU and optimizes the circuit
/// for execution. This function runs the fourth stage of the preset pass manager,
/// **optimization**, which optimizes the circuit for the given target after the
/// circuit has been transformed into a physical circuit. You can refer to
/// @verbatim embed:rst:inline :ref:`transpiler-preset-stage-optimization` @endverbatim for
/// more details.
///
/// This function should only be used with circuits constructed
/// using Qiskit's C API. It makes assumptions on the circuit only using features exposed via C,
/// if you are in a mixed Python and C environment it is typically better to invoke the transpiler
/// via Python.
///
/// This function is multithreaded internally and will launch a thread pool
/// with threads equal to the number of CPUs reported by the operating system by default.
/// This will include logical cores on CPUs with simultaneous multithreading. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
/// ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.
///
/// @param dag A pointer to the circuit to run the transpiler on.
/// @param target A pointer to the target to compile the circuit for.
/// @param options A pointer to an options object that defines user options. If this is a null
///   pointer the default values will be used. See ``qk_transpile_default_options``
///   for more details on the default values.
/// @param error A pointer to a pointer with an nul terminated string with an error description.
///   If the transpiler fails a pointer to the string with the error description will be written
///   to this pointer. That pointer needs to be freed with ``qk_str_free``. This can be a null
///   pointer in which case the error will not be written out.
/// @param state A pointer to a ``QkTranspilerStageState`` object containing the layout. Typically you will need
///   to run the `qk_transpile_stage_layout` prior to this function and that will provide a
///   `QkTranspileLayout` object with the initial layout set you want to take that output layout from
///   that function and use this as the input for this. If you don't have a layout object (e.g. you ran
///   your own layout pass). You can run ``qk_transpile_layout_generate_from_mapping`` to generate a trivial
///   layout (where virtual qubit 0 in the circuit is mapped to physical qubit 0 in the target,
///   1->1, 2->2, etc) for the dag at it's current state. This will enable you to generate a layout
///   object for the optimization stage if you generate your own layout.
///
/// @returns The return code for the transpiler, ``QkExitCode_Success`` means success and all
///   other values indicate an error.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` and ``target`` are not valid, non-null
/// pointers to a ``QkDag``, or a ``QkTarget`` respectively. ``options`` must
/// be a valid pointer a to a ``QkTranspileOptions`` or ``NULL``. ``error`` must
/// be a valid pointer to a ``char`` pointer or ``NULL``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_stage_optimization(
    dag: *mut DAGCircuit,
    target: *const Target,
    options: *const TranspileOptions,
    error: *mut *mut c_char,
    state: *mut TranspilerStageState,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let dag = unsafe { &mut *dag };
    let target = unsafe { &*target };
    let state = unsafe { &mut *state };
    let layout = state
        .layout
        .as_mut()
        .expect("A layout must exist if running optimization stage.");
    let options = if options.is_null() {
        &TranspileOptions::default()
    } else {
        // SAFETY: We checked the pointer is not null, then, per documentation, it is a valid
        // and aligned pointer.
        unsafe { &*options }
    };
    let approximation_degree = if options.approximation_degree.is_nan() {
        None
    } else {
        if !(0.0..=1.0).contains(&options.approximation_degree) {
            panic!(
                "Invalid value provided for approximation degree, only NAN or values between 0.0 and 1.0 inclusive are valid"
            );
        }
        Some(options.approximation_degree)
    };
    let mut synthesis_state = UnitarySynthesisState::new(UnitarySynthesisConfig {
        approximation: unitary_synthesis::Approximation::from_py_approximation_degree(
            approximation_degree,
        ),
        run_python_decomposers: false,
        ..Default::default()
    });
    let mut equiv_lib = generate_standard_equivalence_library();
    let mut commutation_checker = get_standard_commutation_checker();

    match optimization_stage(
        dag,
        target,
        options.optimization_level.into(),
        approximation_degree,
        &mut synthesis_state,
        &mut commutation_checker,
        &mut equiv_lib,
        layout,
    ) {
        Ok(_) => ExitCode::Success,
        Err(e) => {
            if !error.is_null() {
                // Right now we return a backtrace of the error. This at least gives a hint as to
                // which pass failed when we have rust errors normalized we can actually have error
                // messages which are user facing. But most likely this will be a PyErr and panic
                // when trying to extract the string.
                let msg = util::cstring_from_utf8_lossy(format!(
                    "Transpilation failed with this backtrace: {}",
                    e.backtrace()
                ))
                .into_raw();
                // SAFETY: per documentation, if `error` is not null, it accepts a single write.
                unsafe { error.write(msg) };
            }
            ExitCode::TranspilerError
        }
    }
}

/// @ingroup QkTranspiler
/// Run the preset translation stage of the transpiler on a circuit
///
/// The Qiskit transpiler is a quantum circuit compiler that rewrites a given
/// input circuit to match the constraints of a QPU and optimizes the circuit
/// for execution. This function runs the fourth stage of the preset pass manager,
/// **translation**, which translates all the instructions in the circuit into
/// those supported by the target. You can refer to
/// @verbatim embed:rst:inline :ref:`transpiler-preset-stage-translation` @endverbatim for more details.
///
/// This function should only be used with circuits constructed
/// using Qiskit's C API. It makes assumptions on the circuit only using features exposed via C,
/// if you are in a mixed Python and C environment it is typically better to invoke the transpiler
/// via Python.
///
/// This function is multithreaded internally and will launch a thread pool
/// with threads equal to the number of CPUs reported by the operating system by default.
/// This will include logical cores on CPUs with simultaneous multithreading. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
/// ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.
///
/// @param dag A pointer to the circuit to run the transpiler on.
/// @param target A pointer to the target to compile the circuit for.
/// @param options A pointer to an options object that defines user options. If this is a null
///   pointer the default values will be used. See ``qk_transpile_default_options``
///   for more details on the default values.
/// @param error A pointer to a pointer with an nul terminated string with an error description.
///   If the transpiler fails a pointer to the string with the error description will be written
///   to this pointer. That pointer needs to be freed with ``qk_str_free``. This can be a null
///   pointer in which case the error will not be written out.
///
/// @returns The return code for the transpiler, ``QkExitCode_Success`` means success and all
///   other values indicate an error.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` and ``target`` are not valid, non-null
/// pointers to a ``QkDag``, ``QkTarget`` respectively. ``options`` must be a valid pointer a to
/// a ``QkTranspileOptions`` or ``NULL``. ``error`` must be a valid pointer to a ``char`` pointer
/// or ``NULL``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_stage_translation(
    dag: *mut DAGCircuit,
    target: *const Target,
    options: *const TranspileOptions,
    error: *mut *mut c_char,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { &mut *dag };
    let target = unsafe { &*target };
    let options = if options.is_null() {
        &TranspileOptions::default()
    } else {
        // SAFETY: We checked the pointer is not null, then, per documentation, it is a valid
        // and aligned pointer.
        unsafe { &*options }
    };

    let approximation_degree = if options.approximation_degree.is_nan() {
        None
    } else {
        if !(0.0..=1.0).contains(&options.approximation_degree) {
            panic!(
                "Invalid value provided for approximation degree, only NAN or values between 0.0 and 1.0 inclusive are valid"
            );
        }
        Some(options.approximation_degree)
    };
    let mut synthesis_state = UnitarySynthesisState::new(UnitarySynthesisConfig {
        approximation: unitary_synthesis::Approximation::from_py_approximation_degree(
            approximation_degree,
        ),
        run_python_decomposers: false,
        ..Default::default()
    });
    let mut equiv_lib = generate_standard_equivalence_library();

    match translation_stage(dag, target, &mut synthesis_state, &mut equiv_lib) {
        Ok(_) => ExitCode::Success,
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    // Right now we return a backtrace of the error. This at least gives a hint as to
                    // which pass failed when we have rust errors normalized we can actually have error
                    // messages which are user facing. But most likely this will be a PyErr and panic
                    // when trying to extract the string.
                    *error = util::cstring_from_utf8_lossy(format!(
                        "Transpilation failed with this backtrace: {}",
                        e.backtrace()
                    ))
                    .into_raw();
                }
            }
            ExitCode::TranspilerError
        }
    }
}

/// @ingroup QkTranspiler
/// Run the preset layout stage of the transpiler on a circuit
///
/// The Qiskit transpiler is a quantum circuit compiler that rewrites a given
/// input circuit to match the constraints of a QPU and optimizes the circuit
/// for execution. This function runs the second stage of the preset pass manager
/// **layout**, which chooses the initial mapping of virtual qubits to
/// physical qubits, including expansion of the circuit to contain explicit
/// ancillas. You can refer to
/// @verbatim embed:rst:inline :ref:`transpiler-preset-stage-layout` @endverbatim for more details.
///
/// This function should only be used with circuits constructed
/// using Qiskit's C API. It makes assumptions on the circuit only using features exposed via C,
/// if you are in a mixed Python and C environment it is typically better to invoke the transpiler
/// via Python.
///
/// This function is multithreaded internally and will launch a thread pool
/// with threads equal to the number of CPUs reported by the operating system by default.
/// This will include logical cores on CPUs with simultaneous multithreading. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
/// ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.
///
/// @param dag A pointer to the circuit to run the transpiler on.
/// @param target A pointer to the target to compile the circuit for.
/// @param options A pointer to an options object that defines user options. If this is a null
///   pointer the default values will be used. See ``qk_transpile_default_options``
///   for more details on the default values.
/// @param state A pointer to a pointer to a ``QkTranspilerStageState`` object. On a successful
///   execution (return code 0) the layout object created by the transpiler will be written to the
///   state object in this pointer. The inner pointer for this can be null if there is no existing state
///   or layout object. Typically if you run `qk_transpile_stage_init` you would take the output state
///   from that function and use its layout as the input for this. But if you don't have a layout, the
///   inner pointer can be null and a new `QkTranspilerStageState`, with a layout, will be allocated and
///   that pointer will be set for the inner value of the layout here.
/// @param error A pointer to a pointer with an nul terminated string with an error description.
///   If the transpiler fails a pointer to the string with the error description will be written
///   to this pointer. That pointer needs to be freed with ``qk_str_free``. This can be a null
///   pointer in which case the error will not be written out.
///
/// @returns The return code for the transpiler, ``QkExitCode_Success`` means success and all
///   other values indicate an error.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` or ``target``, are not valid, non-null
/// pointers to a ``QkDag``, or a ``QkTarget`` respectively. Behavior is also undefined if ``layout``
/// is not a valid, aligned, pointer to a pointer to a ``QkTranspileLayout`` or a pointer to a
/// ``NULL`` pointer. ``options`` must be a valid pointer a to a ``QkTranspileOptions`` or ``NULL``.
/// ``error`` must be a valid pointer to a ``char`` pointer or ``NULL``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_stage_layout(
    dag: *mut DAGCircuit,
    target: *const Target,
    options: *const TranspileOptions,
    state: *mut *mut TranspilerStageState,
    error: *mut *mut c_char,
) -> ExitCode {
    // SAFETY: Per documentation, the pointers are non-null and aligned.
    let dag = unsafe { &mut *dag };
    let target = unsafe { &*target };

    let options = if options.is_null() {
        &TranspileOptions::default()
    } else {
        // SAFETY: We checked the pointer is not null, then, per documentation, it is a valid
        // and aligned pointer.
        unsafe { &*options }
    };

    let seed = if options.seed < 0 {
        None
    } else {
        Some(options.seed as u64)
    };
    let sabre_heuristic = match get_sabre_heuristic(target) {
        Ok(val) => val,
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    // Right now we return a backtrace of the error. This at least gives a hint as to
                    // which pass failed when we have rust errors normalized we can actually have error
                    // messages which are user facing. But most likely this will be a PyErr and panic
                    // when trying to extract the string.
                    *error = util::cstring_from_utf8_lossy(format!(
                        "Transpilation failed with this backtrace: {}",
                        e.backtrace()
                    ))
                    .into_raw();
                }
            }
            return ExitCode::TranspilerError;
        }
    };

    // SAFETY: As per documentation, state should be a non-null pointer to an address
    // containing another valid pointer.
    let deref_state = unsafe { *state };
    let mut new_layout = || -> Result<TranspileLayout, anyhow::Error> {
        let mut out_layout = TranspileLayout::new(
            None,
            None,
            dag.qubits().objects().to_owned(),
            dag.num_qubits() as u32,
            dag.qregs().to_vec(),
        );
        layout_stage(
            dag,
            target,
            options.optimization_level.into(),
            seed,
            &sabre_heuristic,
            &mut out_layout,
        )
        .map(|_| out_layout)
    };
    let err = |e: anyhow::Error| -> ExitCode {
        if !error.is_null() {
            // Right now we return a backtrace of the error. This at least gives a hint as to
            // which pass failed when we have rust errors normalized we can actually have error
            // messages which are user facing. But most likely this will be a PyErr and panic
            // when trying to extract the string.
            let out_string = util::cstring_from_utf8_lossy(format!(
                "Transpilation failed with this backtrace: {}",
                e.backtrace()
            ))
            .into_raw();
            unsafe {
                *error = out_string;
            }
        }
        ExitCode::TranspilerError
    };
    if deref_state.is_null() {
        // SAFETY: As per documentation state is a pointer to an address of
        // enough size.
        match new_layout() {
            Ok(new_layout) => {
                unsafe {
                    *state = Box::into_raw(Box::new(TranspilerStageState {
                        layout: Some(new_layout),
                    }))
                }
                ExitCode::Success
            }
            Err(e) => err(e),
        }
    } else {
        let layout_inner = unsafe { &mut **state };
        if let Some(layout) = layout_inner.layout.as_mut() {
            match layout_stage(
                dag,
                target,
                options.optimization_level.into(),
                seed,
                &sabre_heuristic,
                layout,
            ) {
                Err(e) => {
                    if !error.is_null() {
                        // Right now we return a backtrace of the error. This at least gives a hint as to
                        // which pass failed when we have rust errors normalized we can actually have error
                        // messages which are user facing. But most likely this will be a PyErr and panic
                        // when trying to extract the string.
                        let out_string = util::cstring_from_utf8_lossy(format!(
                            "Transpilation failed with this backtrace: {}",
                            e.backtrace()
                        ))
                        .into_raw();
                        unsafe {
                            *error = out_string;
                        }
                    }
                    ExitCode::TranspilerError
                }
                Ok(_) => ExitCode::Success,
            }
        } else {
            match new_layout() {
                Ok(new_layout) => {
                    layout_inner.layout.replace(new_layout);
                    ExitCode::Success
                }
                Err(e) => err(e),
            }
        }
    }
}

/// @ingroup QkTranspiler
/// Transpile a single circuit.
///
/// The Qiskit transpiler is a quantum circuit compiler that rewrites a given
/// input circuit to match the constraints of a QPU and optimizes the circuit
/// for execution. This function should only be used with circuits constructed
/// using Qiskit's C API. It makes assumptions on the circuit only using features exposed via C,
/// if you are in a mixed Python and C environment it is typically better to invoke the transpiler
/// via Python.
///
/// This function is multithreaded internally and will launch a thread pool
/// with threads equal to the number of CPUs reported by the operating system by default.
/// This will include logical cores on CPUs with simultaneous multithreading. You can tune the
/// number of threads with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
/// ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.
///
/// @param qc A pointer to the circuit to run the transpiler on.
/// @param target A pointer to the target to compile the circuit for.
/// @param options A pointer to an options object that defines user options. If this is a null
///   pointer the default values will be used. See ``qk_transpile_default_options``
///   for more details on the default values.
/// @param result A pointer to the memory location of the transpiler result. On a successful
///   execution (return code 0) the output of the transpiler will be written to the pointer. The
///   members of the result struct are owned by the caller and you are responsible for freeing
///   the members using the respective free functions.
/// @param error A pointer to a pointer with an nul terminated string with an error description.
///   If the transpiler fails a pointer to the string with the error description will be written
///   to this pointer. That pointer needs to be freed with ``qk_str_free``. This can be a null
///   pointer in which case the error will not be written out.
///
/// @returns The return code for the transpiler, ``QkExitCode_Success`` means success and all
///   other values indicate an error.
///
/// # Safety
///
/// Behavior is undefined if ``circuit``, ``target``, or ``result``, are not valid, non-null
/// pointers to a ``QkCircuit``, ``QkTarget``, or ``QkTranspileResult`` respectively.
/// ``options`` must be a valid pointer a to a ``QkTranspileOptions`` or ``NULL``.
/// ``error`` must be a valid pointer to a ``char`` pointer or ``NULL``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile(
    qc: *const CircuitData,
    target: *const Target,
    options: *const TranspileOptions,
    result: *mut TranspileResult,
    error: *mut *mut c_char,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let qc = unsafe { &*qc };
    let target = unsafe { &*target };
    let options = if options.is_null() {
        &TranspileOptions::default()
    } else {
        // SAFETY: We checked the pointer is not null, then, per documentation, it is a valid
        // and aligned pointer.
        unsafe { &*options }
    };

    if !(0..=3u8).contains(&options.optimization_level) {
        panic!(
            "Invalid optimization level specified {}",
            options.optimization_level
        );
    }

    let seed = if options.seed < 0 {
        None
    } else {
        Some(options.seed as u64)
    };
    let approximation_degree = if options.approximation_degree.is_nan() {
        None
    } else {
        if !(0.0..=1.0).contains(&options.approximation_degree) {
            panic!(
                "Invalid value provided for approximation degree, only NAN or values between 0.0 and 1.0 inclusive are valid"
            );
        }
        Some(options.approximation_degree)
    };

    if let Some(target_qubits) = target.num_qubits
        && target_qubits < qc.num_qubits() as u32
    {
        if !error.is_null() {
            let msg = util::cstring_from_utf8_lossy(format!(
                "Insufficient qubits in target: {}, the circuit uses {}",
                target_qubits,
                qc.num_qubits()
            ))
            .into_raw();
            // SAFETY: per documentation, if `error` is not null, it accepts a single write.
            unsafe { error.write(msg) };
        }
        return ExitCode::TranspilerError;
    }

    match transpile(
        qc,
        target,
        options.optimization_level.into(),
        approximation_degree,
        seed,
    ) {
        Ok(transpile_result) => {
            unsafe {
                *result = TranspileResult {
                    circuit: Box::into_raw(Box::new(transpile_result.0)),
                    layout: Box::into_raw(Box::new(transpile_result.1)),
                };
            }
            ExitCode::Success
        }
        Err(e) => {
            if !error.is_null() {
                // Right now we return a backtrace of the error. This at least gives a hint as to
                // which pass failed when we have rust errors normalized we can actually have error
                // messages which are user facing. But most likely this will be a PyErr and panic
                // when trying to extract the string.
                let msg = util::cstring_from_utf8_lossy(format!(
                    "Transpilation failed with this backtrace: {}",
                    e.backtrace()
                ))
                .into_raw();
                // SAFETY: per documentation, if `error` is not null, it accepts a single write.
                unsafe { error.write(msg) };
            }
            ExitCode::TranspilerError
        }
    }
}
