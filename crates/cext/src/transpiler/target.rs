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

use std::ffi::{CStr, CString, c_char};
use std::ptr::null_mut;
use std::sync::Arc;

use crate::dag::COperationKind;
use crate::exit_codes::{CInputError, ExitCode};
use crate::pointers::{check_ptr, const_ptr_as_ref, mut_ptr_as_ref};
use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::instruction::{Instruction, Parameters};
use qiskit_circuit::operations::StandardInstruction;
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;
use qiskit_transpiler::target::{InstructionProperties, Qargs, Target, TargetOperation};
use qiskit_util::IndexMap;
use smallvec::{SmallVec, smallvec};

/// @ingroup QkTarget
/// Construct a new ``QkTarget`` with the given number of qubits.
/// The number of qubits is bound to change if an instruction is added with properties
/// that apply to a collection of qargs in which any index is higher than the specified
/// number of qubits
///
/// @param num_qubits The number of qubits the ``QkTarget`` will explicitly support.
///
/// @return A pointer to the new ``QkTarget``
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
/// ```
///
#[unsafe(no_mangle)]
pub extern "C" fn qk_target_new(num_qubits: u32) -> *mut Target {
    let target = Target::new(
        None,
        Some(num_qubits),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    Box::into_raw(Box::new(target))
}

/// @ingroup QkTarget
/// Retrieve a `QkTarget` pointer from a Python object.
///
/// This borrows a Python reference and extracts the `QkTarget` pointer for it, if it is of
/// the correct type.  The returned pointer is borrowed from the `ob` pointer.  If the
/// ``PyObject`` is not the correct type, the return value is ``NULL`` and the exception
/// state of the Python interpreter is set.
///
/// You must be attached to a Python interpreter to call this function.
///
/// You can also use `qk_target_convert_from_python`, which is logically the exact same as this
/// function, but can be directly used as a "converter" function for the `PyArg_Parse*`
/// family of Python converter functions.
///
/// @param ob A borrowed Python object.
/// @return A pointer to the native object, or `NULL` if the Python object is the wrong type.
///
/// # Safety
///
/// The caller must be attached to a Python interpreter.  Behavior is undefined if `ob` is
/// not a valid non-null pointer to a Python object.
#[unsafe(no_mangle)]
#[cfg(feature = "python_binding")]
pub unsafe extern "C" fn qk_target_borrow_from_python(ob: *mut pyo3::ffi::PyObject) -> *mut Target {
    // SAFETY: per documentation, we are attached to a Python interpreter and `ob` points to a valid
    // Python object.
    unsafe { crate::py::borrow_mut(::pyo3::Python::assume_attached(), ob) }
}

/// @ingroup QkTarget
/// Retrieve a Target pointer from a Python object.
///
/// This borrows a Python reference and extracts the `QkTarget` pointer for it into ``address``, if
/// it is of the correct type.  The returned pointer is borrowed from the `object` pointer.  If the
/// ``PyObject`` is not the correct type, the return value is 1, the exception state of the Python
/// interpreter is set, and ``address`` is unchanged.
///
/// You must be attached to a Python interpreter to call this function.
///
/// You can also use `qk_target_borrow_from_python`, which is logically the exact same as this, but
/// with a more natural signature for direct usage.
///
/// @param object A borrowed Python object.
/// @param address The location to write the output to.
/// @return 1 on success, 0 on failure.
///
/// # Safety
///
/// The caller must be attached to a Python interpreter.  Behavior is undefined if `object`
/// is not a valid non-null pointer to a Python object, or if `address` is not a pointer to
/// writeable data of the correct type.
#[unsafe(no_mangle)]
#[cfg(feature = "python_binding")]
pub unsafe extern "C" fn qk_target_convert_from_python(
    object: *mut ::pyo3::ffi::PyObject,
    address: *mut ::std::ffi::c_void,
) -> ::std::ffi::c_int {
    // SAFETY: per documentation, we are attached to a Python interpreter, `object` points to a
    // valid Python object and `address` points to enough space to write a pointer.
    unsafe { crate::py::convert_mut::<Target>(::pyo3::Python::assume_attached(), object, address) }
}

/// @ingroup QkTarget
/// Returns the number of qubits of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The number of qubits this target can use.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     uint32_t num_qubits = qk_target_num_qubits(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_num_qubits(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.num_qubits.unwrap_or_default()
}

/// @ingroup QkTarget
/// Returns the dt value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The dt value of this ``QkTarget`` or ``NAN`` if not assigned.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     qk_target_set_dt(target, 10e-9);
///     double dt = qk_target_dt(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_dt(target: *const Target) -> f64 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.dt.unwrap_or(f64::NAN)
}

/// @ingroup QkTarget
/// Returns the granularity value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The ``granularity`` value of this ``QkTarget``.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     uint32_t granularity = qk_target_granularity(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_granularity(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.granularity
}

/// @ingroup QkTarget
/// Returns the ``min_length`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The ``min_length`` value of this ``QkTarget``.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     size_t min_length = qk_target_min_length(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_min_length(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.min_length
}

/// @ingroup QkTarget
/// Returns the ``pulse_alignment`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The ``pulse_alignment`` value of this ``QkTarget``.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     uint32_t pulse_alignment = qk_target_pulse_alignment(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_pulse_alignment(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.pulse_alignment
}

/// @ingroup QkTarget
/// Returns the ``acquire_alignment`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The ``acquire_alignment`` value of this ``QkTarget``.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 0
///     uint32_t acquire_alignment = qk_target_pulse_alignment(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_acquire_alignment(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.acquire_alignment
}

/// @ingroup QkTarget
/// Sets the dt value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param dt The ``dt`` value for the system time resolution of input.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
///
/// ```c
///     QkTarget *target = qk_target_new(5);
///     double dt = qk_target_set_dt(target, 10e-9);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_set_dt(target: *mut Target, dt: f64) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.dt = if dt.is_nan() { None } else { Some(dt) };
    ExitCode::Success
}

/// @ingroup QkTarget
/// Sets the ``granularity`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param granularity The value for the minimum pulse gate resolution in
///     units of ``dt``.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_granularity(target, 2);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_set_granularity(
    target: *mut Target,
    granularity: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.granularity = granularity;
    ExitCode::Success
}

/// @ingroup QkTarget
/// Sets the ``min_length`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param min_length The minimum pulse gate length value in units of ``dt``.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
///
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_min_length(target, 3);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_set_min_length(
    target: *mut Target,
    min_length: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.min_length = min_length;
    ExitCode::Success
}

/// @ingroup QkTarget
/// Returns the ``pulse_alignment`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param pulse_alignment value representing a time resolution of gate.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_pulse_alignment(target, 4);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_set_pulse_alignment(
    target: *mut Target,
    pulse_alignment: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.pulse_alignment = pulse_alignment;
    ExitCode::Success
}

/// @ingroup QkTarget
/// Sets the ``acquire_alignment`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param acquire_alignment value representing a time resolution of measure instruction
///     starting time.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 0
///     qk_target_set_acquire_alignment(target, 5);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_set_acquire_alignment(
    target: *mut Target,
    acquire_alignment: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.acquire_alignment = acquire_alignment;
    ExitCode::Success
}

/// @ingroup QkTarget
/// Creates a copy of the ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget`` to copy.
///
/// @return A pointer to the new copy of the ``QkTarget``.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     QkExitCode result = qk_target_add_instruction(target, entry);
///
///     QkTarget *copied = qk_target_copy(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_copy(target: *mut Target) -> *mut Target {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    Box::into_raw(target.clone().into())
}

/// @ingroup QkTarget
/// Free the ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget`` to free.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     qk_target_free(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_free(target: *mut Target) {
    if !target.is_null() {
        if !target.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(target);
        }
    }
}

#[derive(Debug)]
enum StandardOperation {
    Gate(StandardGate),
    Instruction(StandardInstruction),
}

impl StandardOperation {
    pub fn num_qubits(&self) -> u32 {
        match &self {
            Self::Gate(gate) => gate.num_qubits(),
            Self::Instruction(inst) => inst.num_qubits(),
        }
    }
}

impl From<StandardOperation> for PackedOperation {
    fn from(value: StandardOperation) -> Self {
        match value {
            StandardOperation::Gate(gate) => gate.into(),
            StandardOperation::Instruction(inst) => inst.into(),
        }
    }
}

/// Represents the mapping between qargs and ``InstructionProperties``
#[derive(Debug)]
pub struct TargetEntry {
    operation: StandardOperation,
    params: Option<SmallVec<[Param; 3]>>,
    map: IndexMap<Qargs, Option<InstructionProperties>>,
    name: Option<String>,
}

impl TargetEntry {
    pub fn new(operation: StandardGate) -> Self {
        let params = if operation.num_params() > 0 {
            Some(
                (0..operation.num_params())
                    .map(|i| {
                        let op_name = operation.name();
                        Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                            Symbol::standalone(format!("{op_name}_param_{i}"), None),
                        )))
                    })
                    .collect(),
            )
        } else {
            None
        };
        Self {
            operation: StandardOperation::Gate(operation),
            params,
            map: Default::default(),
            name: None,
        }
    }

    pub fn new_fixed(
        operation: StandardGate,
        params: SmallVec<[Param; 3]>,
        name: Option<String>,
    ) -> Self {
        Self {
            operation: StandardOperation::Gate(operation),
            params: Some(params),
            map: Default::default(),
            name,
        }
    }

    pub fn new_instruction(instruction: StandardInstruction) -> Self {
        Self {
            operation: StandardOperation::Instruction(instruction),
            params: None,
            map: Default::default(),
            name: None,
        }
    }
}

/// @ingroup QkTargetEntry
/// Creates an entry to the ``QkTarget`` based on a ``QkGate`` instance.
///
/// @param operation The ``QkGate`` whose properties this target entry defines. If the ``QkGate``
/// takes parameters (which can be checked with ``qk_gate_num_params``) it will be added as a
/// an instruction on the target which accepts any parameter value. If the gate only accepts a
/// fixed parameter value you can use ``qk_target_entry_new_fixed`` instead.
///
/// @return A pointer to the new ``QkTargetEntry``.
///
/// # Example
/// ```c
///     QkTargetEntry *had_entry = qk_target_entry_new(QkGate_H);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn qk_target_entry_new(operation: StandardGate) -> *mut TargetEntry {
    Box::into_raw(Box::new(TargetEntry::new(operation)))
}

/// @ingroup QkTargetEntry
/// Creates a new entry for adding a measurement instruction to a ``QkTarget``.
///
/// @return A pointer to the new ``QkTargetEntry`` for a measurement instruction.
///
/// # Example
/// ```c
///     QkTargetEntry *entry = qk_target_entry_new_measure();
///     // Add fixed duration and error rates from qubits at index 0 to 4.
///     for (uint32_t i = 0; i < 5; i++) {
///         // Measure is a single qubit instruction
///         uint32_t qargs[1] = {i};
///         qk_target_entry_add_property(entry, qargs, 1, 1.928e-10, 7.9829e-11);
///     }
///
///     // Add the entry to a target with 5 qubits
///     QkTarget *measure_target = qk_target_new(5);
///     qk_target_add_instruction(measure_target, entry);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn qk_target_entry_new_measure() -> *mut TargetEntry {
    Box::into_raw(Box::new(TargetEntry::new_instruction(
        StandardInstruction::Measure,
    )))
}

/// @ingroup QkTargetEntry
/// Creates a new entry for adding a reset instruction to a ``QkTarget``.
///
/// @return A pointer to the new ``QkTargetEntry`` for a reset instruction.
///
/// # Example
/// ```c
///     QkTargetEntry *entry = qk_target_entry_new_reset();
///     // Add fixed duration and error rates from qubits at index 0 to 2.
///     for (uint32_t i = 0; i < 3; i++) {
///         // Reset is a single qubit instruction
///         uint32_t qargs[1] = {i};
///         qk_target_entry_add_property(entry, qargs, 1, 1.2e-11, 5.9e-13);
///     }
///
///     // Add the entry to a target with 3 qubits
///     QkTarget *reset_target = qk_target_new(3);
///     qk_target_add_instruction(reset_target, entry);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn qk_target_entry_new_reset() -> *mut TargetEntry {
    Box::into_raw(Box::new(TargetEntry::new_instruction(
        StandardInstruction::Reset,
    )))
}

/// @ingroup QkTargetEntry
/// Creates an entry in the ``QkTarget`` based on a ``QkGate`` instance with
/// no parameters.
///
/// @note Adding a ``QkGate`` with regular parameters is not currently supported.
///
/// @param operation The ``QkGate`` whose properties this target entry defines.
/// @param params A pointer to the parameters that the instruction is calibrated for.
/// @param name An optional name for the instruction in the target. If a null pointer is
/// provided, the entry will use the operation's default name.
///
/// @return A pointer to the new ``QkTargetEntry``.
///
/// # Example
/// ```c
///     double crx_params[1] = {3.14};
///     QkTargetEntry *entry = qk_target_entry_new_fixed(QkGate_CRX, crx_params, "crx_fixed")";
/// ```
///
/// # Safety
///
/// The ``params`` type is expected to be a pointer to an array of ``double`` where the length
/// matches the expectations of the ``QkGate``. If the array is insufficiently long the
/// behavior of this function is undefined as this will read outside the bounds of the array.
/// It can be a null pointer if there are no params for a given gate. You can check
/// ``qk_gate_num_params`` to determine how many qubits are required for a given gate.
///
/// The ``name`` pointer is expected to be either a C string comprising of valid UTF-8 characters
/// or a null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_entry_new_fixed(
    operation: StandardGate,
    params: *mut f64,
    name: *const c_char,
) -> *mut TargetEntry {
    // SAFETY: per documentation, name points to a valid UTF-8 null-terminated string.
    let name_fixed: Option<String> = if name.is_null() {
        None
    } else {
        Some(unsafe {
            CStr::from_ptr(name)
                .to_str()
                .expect("Error while extracting the given name.")
                .to_string()
        })
    };
    unsafe {
        Box::into_raw(Box::new(TargetEntry::new_fixed(
            operation,
            parse_params(operation, params),
            name_fixed,
        )))
    }
}

/// @ingroup QkTargetEntry
/// Retrieves the number of properties stored in the target entry.
///
/// @param entry The pointer to the mapping object.
///
/// @return The number of properties in the ``QkTargetEntry``.
///
/// # Example
/// ```c
///     // Create an entry for an H gate
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_H);
///     size_t props_size = qk_target_entry_num_properties(entry);
/// ```
///
/// # Safety
///
/// The behavior is undefined if ``entry`` is not a valid,
/// non-null pointer to a ``QkTargetEntry`` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_entry_num_properties(entry: *const TargetEntry) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { const_ptr_as_ref(entry) };
    prop_map.map.len()
}

/// @ingroup QkTargetEntry
/// Frees the entry.
///
/// @note An entry pointer will be freed when added to a ``QkTarget`` via
/// ``qk_target_add_instruction``, this function is only meant to be used
/// alternatively if an entry is never added to a ``QkTarget`` instance.
///
/// @param entry The pointer to the mapping object to be freed.
///
/// # Example
/// ```c
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_H);
///     qk_target_entry_free(entry);
/// ```
///
/// # Safety
///
/// The behavior is undefined if ``entry`` is not a valid,
/// non-null pointer to a ``QkTargetEntry`` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_entry_free(entry: *mut TargetEntry) {
    if !entry.is_null() {
        if !entry.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(entry);
        }
    }
}

/// @ingroup QkTargetEntry
/// Adds an instruction property instance based on its assigned qargs.
///
/// @param entry The pointer to the entry object.
/// @param qargs A pointer to the array of ``uint32_t`` qubit indices to add the
///     gate on, can be a null pointer to check for global properties.
/// @param num_qubits The length of the qargs array.
/// @param duration The instruction's duration in seconds on the specific set of
///     qubits.
/// @param error The instruction's average error rate on the specific set of qubits.
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
/// ```
///
/// # Safety
///
/// The behavior is undefined if ``entry`` is not a valid, non-null pointer
/// to a ``QkTargetEntry`` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_entry_add_property(
    entry: *mut TargetEntry,
    qargs: *mut u32,
    num_qubits: u32,
    duration: f64,
    error: f64,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let entry = unsafe { mut_ptr_as_ref(entry) };
    if num_qubits != entry.operation.num_qubits() {
        return ExitCode::TargetQargMismatch;
    }
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qubits: Qargs = unsafe { parse_qargs(qargs, num_qubits) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let duration = if duration.is_nan() {
        None
    } else {
        Some(duration)
    };
    let error = if error.is_nan() { None } else { Some(error) };
    entry
        .map
        .insert(qubits, Some(InstructionProperties::new(duration, error)));
    ExitCode::Success
}

/// @ingroup QkTargetEntry
/// Sets a custom name to the target entry.
///
/// @param entry The pointer to the entry object.
/// @param name The name to be set for the target entry.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
/// ```c
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     qk_target_entry_set_name(entry, "cx_gate");
/// ```
///
/// # Safety
///
/// The behavior is undefined if ``entry`` is not a valid, non-null pointer
/// to a ``QkTargetEntry`` object.
/// The ``name`` pointer is expected to be either a C string comprising
/// of valid UTF-8 characters or a null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_entry_set_name(
    entry: *mut TargetEntry,
    name: *const c_char,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let entry = unsafe { mut_ptr_as_ref(entry) };
    // SAFETY: per documentation, name points to a valid UTF-8 null-terminated string.
    let name_set: Option<String> = if name.is_null() {
        None
    } else {
        Some(unsafe {
            CStr::from_ptr(name)
                .to_str()
                .expect("Error while extracting the given name.")
                .to_string()
        })
    };
    entry.name = name_set;
    ExitCode::Success
}

/// @ingroup QkTarget
/// Adds a gate to the ``QkTarget`` through a ``QkTargetEntry``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param target_entry A pointer to the ``QkTargetEntry``. The pointer
/// gets freed when added to the ``QkTarget``.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     QkExitCode result = qk_target_add_instruction(target, entry);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// Behavior is undefined if ``entry`` is not a valid, non-null pointer to a ``QkTargetEntry``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_add_instruction(
    target: *mut Target,
    target_entry: *mut TargetEntry,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let pointer_check = check_ptr(target_entry);
    if pointer_check.is_err() {
        match pointer_check {
            Err(CInputError::AlignmentError) => return ExitCode::AlignmentError,
            Err(CInputError::NullPointerError) => return ExitCode::NullPointerError,
            _ => (),
        }
    }
    let entry = unsafe { Box::from_raw(target_entry) };
    let instruction = entry.operation;

    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };

    let property_map = if entry.map.is_empty() {
        None
    } else {
        Some(entry.map)
    };

    match target.add_instruction(
        instruction.into(),
        entry.params.map(Parameters::Params),
        entry.name.as_deref(),
        property_map,
    ) {
        Ok(_) => ExitCode::Success,
        Err(e) => e.into(),
    }
}

/// @ingroup QkTarget
/// Modifies the properties of a gate in the ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param instruction The instruction to modify.
/// @param qargs The pointer to the array of ``uint32_t`` values to use as
/// qargs. Can be ``NULL`` if global.
/// @param num_qubits The number of qubits of the instruction..
/// @param duration The instruction's duration in seconds on the specific set of
///     qubits.
/// @param error The instruction's average error rate on the specific set of qubits.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     double params[1] = {3.1415};
///     QkTargetEntry *entry = qk_target_entry_new_fixed(QkGate_CRX, params, "crx_pi");
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, entry);
///
///     qk_target_update_property(target, QkGate_CRX, qargs, 2, 0.0012, 1.1);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``qargs`` type is expected to be a pointer to an array of ``uint32_t`` where the length
/// matches is specified by ``num_qubits`` and has to match the expectation of the gate. If the
/// array is insufficiently long the behavior of this function is undefined as this will read
/// outside the bounds of the array. It can be a null pointer if there are no qubits for
/// a given gate. You can check ``qk_gate_num_qubits`` to determine how many qubits are required
/// for a given gate.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_update_property(
    target: *mut Target,
    instruction: StandardGate,
    qargs: *mut u32,
    num_qubits: u32,
    duration: f64,
    error: f64,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };

    let duration = if duration.is_nan() {
        None
    } else {
        Some(duration)
    };
    let error = if error.is_nan() { None } else { Some(error) };

    match target.update_instruction_properties(
        instruction.name(),
        &qargs,
        Some(InstructionProperties::new(duration, error)),
    ) {
        Ok(_) => ExitCode::Success,
        Err(e) => e.into(),
    }
}

/// @ingroup QkTarget
/// Returns the number of instructions tracked by a ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The length of the target.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *target_entry = qk_target_entry_new(QkGate_H);
///     qk_target_add_instruction(target, target_entry);
///
///     size_t num_instructions = qk_target_num_instructions(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_num_instructions(target: *const Target) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    target.len()
}

/// @ingroup QkTarget
/// Checks if the provided instruction and its qargs are supported by this
/// ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param operation_name The instruction name to check for.
/// @param qargs The pointer to the array of ``uint32_t`` values to use as
/// qargs. Can be ``NULL`` if global.
/// @param params A pointer to an array of pointers of ``QkParam`` objects as parameters
/// to check. Can be ``NULL`` if no parameters are present.
///
/// @return Whether the instruction is supported or not.
///
/// # Example
/// ```c
///     // Create a mock target with only a global crx entry
///     // and 3.14 as its rotation parameter.
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *crx_entry = qk_target_entry_new_fixed(QkGate_CRX, (double[]){3.14});
///     qk_target_entry_add_property(crx_entry, NULL, 0, 0.0, 0.1);
///     qk_target_add_instruction(target, crx_entry);
///
///     // Check if target is compatible with a "crx" gate
///     // at [0, 1] with 3.14 rotation.
///     QkParam *params[1] = {qk_param_from_double(3.14)};
///     qk_target_instruction_supported(target, "crx", (uint32_t []){0, 1}, params);
///
///     // Free the pointers
///     qk_param_free(params[0]);
///     qk_target_free(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``qargs`` argument is expected to be a pointer to an array of ``u32int_t`` where the length
/// matches the expectation of the gate. If the array is insufficiently long the behavior of this
/// function is undefined as this will read outside the bounds of the array. It can be a null
/// pointer if there are no qubits for a given gate. You can check `qk_gate_num_qubits` to
/// determine how many qubits are required for a given gate.
///
/// The ``params`` argument is expected to be an array of ``QkParam`` where the length matches the
/// expectation of the operation in question. If the array is insufficiently long, the behavior will
/// be undefined just as mentioned above for the ``qargs`` argument. You can always check ``qk_gate_num_params``
/// in the case of a ``QkGate``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_instruction_supported(
    target: *const Target,
    operation_name: *const c_char,
    qargs: *const u32,
    params: *mut *mut Param,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    // SAFETY: Per documentation, the pointer for operation name points to a valid null terminated string.
    let name = unsafe { CStr::from_ptr(operation_name) }
        .to_str()
        .expect("Error extracting gate name from target.");
    let Some(operation) = target.operation_from_name(name) else {
        return false;
    };
    let num_params = match operation {
        TargetOperation::Variadic(_) => 0,
        TargetOperation::Normal(op) => op.params.as_ref().map(|p| p.len()).unwrap_or(0),
    };

    // SAFETY: Per documentation, the params argument points to an appropriately allocated
    // array of valid pointers to `QkParam` objects.
    let params = if params.is_null() {
        Vec::with_capacity(0)
    } else {
        unsafe {
            std::slice::from_raw_parts(params, num_params)
                .iter()
                .map(|param| const_ptr_as_ref(*param).clone())
                .collect()
        }
    };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, operation.num_qubits()) };

    target.instruction_supported(name, &qargs, &params, false)
}

/// @ingroup QkTarget
/// Return the index at which an operation is located based on its name.
///
/// @param target A pointer to the ``QkTarget``.
/// @param name The name to get the index of.
///
/// @return the index in which the operation is the maximum value of ``size_t``
///     in the case it is not in the Target.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *target_entry = qk_target_entry_new(QkGate_H);
///     qk_target_add_instruction(target, target_entry);
///
///     size_t op_idx = qk_target_op_index(target, "h");
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
/// Behavior is undefined if ``name`` is not a pointer to a valid null-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_op_index(target: *const Target, name: *const c_char) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.

    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    // SAFETY: Per documentation, this should point to a valid, null-terminated
    // string.
    let name = unsafe { CStr::from_ptr(name) }
        .to_str()
        .expect("Users must only pass valid UTF-8 strings");

    target_borrowed.get_gate_index(name).unwrap_or(usize::MAX)
}

/// @ingroup QkTarget
/// Return the name of the operation stored at that index in the ``QkTarget`` instance's
/// gate map.
///
/// @param target A pointer to the ``QkTarget``.
/// @param index The index at which the gate is stored.
///
/// @return The name of the operation associated with the provided index.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *target_entry = qk_target_entry_new(QkGate_H);
///     qk_target_add_instruction(target, target_entry);
///
///     char *op_name = qk_target_op_name(target, 0);
///     // Free after use
///     qk_str_free(op_name);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_op_name(target: *const Target, index: usize) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    let (name, _) = target_borrowed
        .get_by_index(index)
        .expect("Operation index should exist in the Target");
    CString::new(name)
        .expect("A valid rust string should not have any trailing 0 byte.")
        .into_raw()
}

/// @ingroup QkTarget
/// Return the number of properties defined for the specified operation in
/// the ``QkTarget`` instance, a.k.a. the length of the property map. Panics
/// if the operation index is not present.
///
/// @param target A pointer to the ``QkTarget``.
/// @param index The index in which the gate is stored.
///
/// @return The number of properties specified for the operation associated with
/// that index.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *target_entry = qk_target_entry_new(QkGate_H);
///     qk_target_add_instruction(target, target_entry);
///
///     size_t num_props = qk_target_op_num_properties(target, 0);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_op_num_properties(target: *const Target, index: usize) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    target_borrowed
        .get_by_index(index)
        .map(|(_, v)| v.len())
        .expect("Operation indices must be present in the Target")
}

/// @ingroup QkTarget
/// Retrieve the index at which some qargs are stored. Returns ``SIZE_MAX``
/// if not found.
///
/// @param target A pointer to the ``QkTarget``.
/// @param op_idx The index at which the operation is stored.
/// @param qargs A pointer to the array of ``uint32_t`` qubit indices to
///     check for, can be a null pointer to check for global properties.
///
/// @return The index of the qargs associated with the instruction at the
/// specified `op_idx` index or ``SIZE_MAX`` if the qargs are not present.
///
/// # Example
/// ```c
/// QkTarget *target = qk_target_new(5);
///
/// QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
/// uint32_t qargs[2] = {0, 1};
/// qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
/// qk_target_add_instruction(target, entry);
///
/// size_t idx_0_1 = qk_target_op_qargs_index(target, 0, qargs);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_op_qargs_index(
    target: *const Target,
    op_idx: usize,
    qargs: *const u32,
) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    let (_, inst_map) = target_borrowed
        .get_by_index(op_idx)
        .expect("Operation indices must be present in the Target");
    let op = target_borrowed.get_op_by_index(op_idx).unwrap();
    // SAFETY: Per the documentation the qubits pointer is an array of the size
    // associated with the operation or a NULL pointer.
    let parsed = unsafe { parse_qargs(qargs, op.num_qubits()) };
    inst_map.get_index_of(&parsed).unwrap_or(usize::MAX)
}

/// @ingroup QkTarget
/// Retrieve the qargs for the operation by index.
///
/// @param target A pointer to the ``QkTarget``.
/// @param op_idx The index at which the gate is stored.
/// @param qarg_idx The index at which the qargs are stored.
/// @param qargs_out An out pointer to an array qubits. If ``op_idx`` refers to a a global
///     operation, a null pointer will be written.  The written pointer is borrowed from the
///     target and must not be freed.  A zero-qargs instruction will write out a non-null pointer,
///     though one that is invalid for reads.
/// @param qargs_len An out pointer to the length of the qargs in `qargs_out`. If the index is
///     global, the written length is not defined.
///
/// Panics if any of the indices are out of range.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, entry);
///
///     uint32_t *qargs_retrieved;
///     uint32_t qargs_length;
///     qk_target_op_qargs(target, 0, 0, &qargs_retrieved, &qargs_length);
///     if (qargs_retrieved) {
///         // We should enter this branch.
///         printf("Number of qargs: %lu\n", qargs_length);
///     } else {
///         printf("Qargs are global\n");
///     }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
/// Behavior is undefined if each `qargs_out` or `qargs_len` are not aligned and writeable for a
/// single value of the correct type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_op_qargs(
    target: *const Target,
    op_idx: usize,
    qarg_idx: usize,
    qargs_out: *mut *mut u32,
    qargs_len: *mut u32,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    let (_, props_map) = target_borrowed
        .get_by_index(op_idx)
        .expect("Operation indices must be present in the Target.");
    let (retrieved_qargs, _) = props_map
        .get_index(qarg_idx)
        .expect("Qarg indices must be present in the operation's property map");
    let (qargs_ptr, qargs_length) = qargs_to_ptr(retrieved_qargs);

    // SAFETY: Per documentation, the pointer is not-null goes to an address
    // with enough capacity for this array.
    unsafe { qargs_out.write(qargs_ptr.cast()) };

    // SAFETY: Per documentation, the pointer goes to an aligned writable uint.
    unsafe { qargs_len.write(qargs_length) };
}

/// @ingroup QkTarget
/// Retrieve the qargs for the operation stored in its respective indices.
///
/// @param target A pointer to the ``QkTarget``.
/// @param op_idx The index in which the gate is stored.
/// @param qarg_idx The index in which the qargs are stored.
/// @param inst_props A pointer to write out the ``QkInstructionProperties`` instance.
///
/// Panics if any of the indices are out of range.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, entry);
///
///     QkInstructionProperties inst_props;
///     qk_target_op_props(target, 0, 0, &inst_props);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
/// Behavior is undefined if ``inst_props`` does not point to an address of the correct size to
/// store ``QkInstructionProperties`` in.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_op_props(
    target: *const Target,
    op_idx: usize,
    qarg_idx: usize,
    inst_props: *mut CInstructionProperties,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    let (_, props_map) = target_borrowed
        .get_by_index(op_idx)
        .expect("Operation indices must be present in the Target");
    let (_, retrieved_props) = props_map
        .get_index(qarg_idx)
        .expect("Qarg indices must be present in the operation's property map");
    // SAFETY: Per documentation, the pointer goes to an address
    // pre-allocated for a ``CInstructionProperties`` object.
    if let Some(props) = retrieved_props {
        unsafe {
            inst_props.write(CInstructionProperties {
                duration: props.duration.unwrap_or(f64::NAN),
                error: props.error.unwrap_or(f64::NAN),
            });
        }
    } else {
        // If no property is found but there are qargs, return a null instance.
        unsafe {
            inst_props.write(CInstructionProperties {
                duration: f64::NAN,
                error: f64::NAN,
            });
        }
    }
}

/// A representation of a Target operation's instruction properties.
#[repr(C)]
pub struct CInstructionProperties {
    /// The duration, in seconds, of the instruction on the specified set of qubits.
    /// Will be set to ``NaN`` if the property is not defined.
    pub duration: f64,
    /// The average error rate for the instruction on the specified set of qubits.
    /// Will be set to ``NaN`` if the property is not defined.
    pub error: f64,
}

/// Representation of an operation identified within the Target.
///
/// This struct is created natively by the underlying `Target` API
/// and users should not write to it directly nor try to free its
/// attributes manually as it would lead to undefined behavior. To
/// free this struct, users should call ``qk_target_op_clear`` instead.
#[repr(C)]
pub struct CTargetOp {
    /// The identifier for the current operation.
    pub op_type: COperationKind,
    /// The name of the operation.
    pub name: *mut c_char,
    /// The number of qubits this operation supports. Will default to
    /// `(uint32_t)-1` in the case of a variadic.
    pub num_qubits: u32,
    /// The parameters tied to this operation, as `QkParam`.
    /// If there are no parameters then this value will be represented
    /// with a `NULL` pointer.
    pub params: *mut *const Param,
    /// The number of parameters supported by this operation. Will default to
    /// `(uint32_t)-1` in the case of a variadic.
    pub num_params: u32,
}

/// @ingroup QkTarget
/// Retrieves information about an operation in the Target via index.
/// If the index is not present, this function will panic. You can check
/// the ``QkTarget`` total number of instructions using ``qk_target_num_instructions``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param index The index in which the gate is stored.
/// @param out_op A pointer to the space where the ``QkTargetOp``
///     will be stored.
///
/// # Example
/// ```c
/// QkTarget *target = qk_target_new(5);
///
/// QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
/// uint32_t qargs[2] = {0, 1};
/// qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
/// qk_target_add_instruction(target, entry);
///
/// QkTargetOp op;
/// qk_target_op_get(target, 0, &op);
///
/// // Clean up after you're done
/// qk_target_op_clear(&op);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
/// Behavior is undefined if ``out_op`` does not point to an address of the correct size to
/// store ``QkTargetOp`` in.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_op_get(
    target: *const Target,
    index: usize,
    out_op: *mut CTargetOp,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };

    let operation = target_borrowed
        .get_op_by_index(index)
        .expect("Operation index should be present in the Target.");
    match operation {
        TargetOperation::Normal(operation) => {
            let kind = match operation.operation.view() {
                qiskit_circuit::operations::OperationRef::StandardGate(_) => COperationKind::Gate,
                qiskit_circuit::operations::OperationRef::StandardInstruction(
                    standard_instruction,
                ) => match standard_instruction {
                    StandardInstruction::Barrier(_) => COperationKind::Barrier,
                    StandardInstruction::Delay(_) => COperationKind::Delay,
                    StandardInstruction::Measure => COperationKind::Measure,
                    StandardInstruction::Reset => COperationKind::Reset,
                },
                qiskit_circuit::operations::OperationRef::Unitary(_) => COperationKind::Unitary,
                _ => panic!(
                    "Unsupported operation type found: {}",
                    stringify!(normal_operation.view())
                ),
            };
            let name = CString::new(
                target_borrowed
                    .get_by_index(index)
                    .expect("An operation name should already exist.")
                    .0,
            )
            .expect("The string should be UTF-8 encoded.")
            .into_raw();
            let num_params = operation.operation.num_params();
            let mut params: Option<Box<[*const Param]>> = (num_params > 0).then_some(
                operation
                    .params_view()
                    .iter()
                    .map(|param| match param {
                        Param::Float(_) | Param::ParameterExpression(_) => {
                            std::ptr::from_ref(param)
                        }
                        Param::Obj(_) => panic!("Objects are not supported in the C API."),
                    })
                    .collect(),
            );
            // SAFETY: As per documentation, `out_op` is a pointer to a sufficient allocation.
            unsafe {
                out_op.write(CTargetOp {
                    op_type: kind,
                    name,
                    num_qubits: operation.operation.num_qubits(),
                    params: if let Some(params) = params.as_mut() {
                        params.as_mut_ptr()
                    } else {
                        null_mut()
                    },
                    num_params,
                })
            };
            let _ = params.map(Box::into_raw);
        }
        TargetOperation::Variadic(_) => {
            let name = CString::new(
                target_borrowed
                    .get_by_index(index)
                    .expect("An operation name should already exist.")
                    .0,
            )
            .expect("Names should be ")
            .into_raw();
            // SAFETY: As per documentation, `out_op` is a pointer to a sufficient allocation.
            unsafe {
                out_op.write(CTargetOp {
                    op_type: COperationKind::Unknown,
                    name,
                    num_qubits: u32::MAX,
                    params: null_mut(),
                    num_params: u32::MAX,
                })
            };
        }
    };
}

/// @ingroup QkTarget
/// Tries to retrieve a ``QkGate`` based on the operation stored in an index.
/// The user is responsible for checking whether this operation is a gate in the ``QkTarget``
/// via using ``qk_target_op_get``. If not, this function will panic.
///
/// @param target A pointer to the ``Target`` instance.
/// @param index The index at which the operation is located.
///
/// @return The ``QkGate`` enum in said index.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, entry);
///
///     QkTargetOp op;
///     qk_target_op_get(target, 0, &op);
///
///     // Check if the operation is a gate;
///     if (op.op_type == QkOperationKind_Gate) {
///         QkGate gate = qk_target_op_gate(target, 0);
///         // Do something
///     }
///
///     // Clean up after you're done.
///     qk_target_op_clear(&op);
/// ```
///
/// # Safety
///
/// Behavior is undefined if the ``target`` pointer is null or not aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_op_gate(target: *const Target, index: usize) -> StandardGate {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };

    match target_borrowed
        .get_op_by_index(index)
        .expect("Invalid index")
    {
        qiskit_transpiler::target::TargetOperation::Normal(normal_operation) => normal_operation
            .operation
            .try_standard_gate()
            .expect("Not a standard gate."),
        qiskit_transpiler::target::TargetOperation::Variadic(_) => panic!("Not a standard gate."),
    }
}

/// @ingroup QkTarget
/// Clears the ``QkTargetOp`` object.
///
/// @param op The pointer to a ``QkTargetOp`` object.
///
/// # Safety
///
/// The behavior will be undefined if the pointer is null or not-aligned.
/// The data belonging to a ``QkTargetOp`` originates in Rust and
/// can only be freed using this function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_target_op_clear(op: *mut CTargetOp) {
    // We need to consume both the name and the parameters

    // SAFETY: As per documentation, data from pointers contained in CTargetOp
    // originates from rust code and are constructed internally with vecs and CStrings.
    unsafe {
        let op_borrowed = mut_ptr_as_ref(op);
        if !op_borrowed.name.is_null() {
            let _ = CString::from_raw(op_borrowed.name);
            op_borrowed.name = std::ptr::null_mut();
        }

        if op_borrowed.num_params > 0 && !op_borrowed.params.is_null() {
            let params = std::slice::from_raw_parts_mut(
                op_borrowed.params,
                op_borrowed.num_params.try_into().unwrap(),
            );
            // Parameters don't need to be freed as they're owned by the
            // Target.
            let _ = Box::from_raw(params as *mut [*const Param]);
            op_borrowed.params = std::ptr::null_mut();
        }
        op_borrowed.num_params = 0;
        op_borrowed.num_qubits = 0;
    }
}

/// Parses qargs based on a pointer and its size.
///
/// # Arguments
///
/// * `qargs` - The pointer to the array of ``uint32_t`` values to use as
///   qargs. Can be ``NULL`` if global.
/// * `num_qubits` - The length of the array.
///
/// # Returns
///
/// A [Qargs] object parsed from the pointer.
///
/// # Safety
///
/// Behavior is undefined if the qubits pointer is non-aligned or if the specified
/// num_qubits exceeds the number of bits stored in the array.
unsafe fn parse_qargs(qargs: *const u32, num_qubits: u32) -> Qargs {
    if qargs.is_null() {
        Qargs::Global
    } else {
        // SAFETY: Per the documentation qargs points to an array of num_qubits elements
        unsafe {
            (0..num_qubits)
                .map(|idx| PhysicalQubit(*qargs.wrapping_add(idx as usize)))
                .collect()
        }
    }
}

/// Parse params based on a standard gate and a pointer to a float.
///
/// # Arguments
///
/// * `gate` - The [StandardGate] for which we will extract parameters.
/// * `params` -  A pointer to the parameters that the instruction is calibrated for.
///
/// # Returns
///
/// A collection of [Param] wrapped in a [SmallVec].
///
/// # Safety
///
/// The ``params`` type is expected to be a pointer to an array of ``double`` where the length
/// matches the expectations of the ``QkGate``. If the array is insufficiently long the
/// behavior of this function is undefined as this will read outside the bounds of the array.
/// It can be a null pointer if there are no params for a given gate. You can check
/// ``qk_gate_num_params`` to determine how many qubits are required for a given gate.
unsafe fn parse_params(gate: StandardGate, params: *mut f64) -> SmallVec<[Param; 3]> {
    // SAFETY: Per the documentation the params pointers are arrays of num_params() elements.
    unsafe {
        match gate.num_params() {
            0 => smallvec![],
            1 => smallvec![(*params.wrapping_add(0)).into()],
            2 => smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
            ],
            3 => smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
            ],
            4 => smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
                (*params.wrapping_add(3)).into(),
            ],
            // There are no standard gates that take > 4 params
            _ => unreachable!(),
        }
    }
}

/// Converts a collection of Qargs into a mutable array pointer and its length.
///
/// # Arguments
///
/// * `qargs` - A reference to a [Qargs] object to convert
///
/// # Returns
///
/// A tuple with a mutable pointer to an array of `u32` members and the length
/// as a `i32`.
fn qargs_to_ptr(qargs: &Qargs) -> (*mut PhysicalQubit, u32) {
    match qargs {
        Qargs::Global => (null_mut(), u32::MAX),
        Qargs::Concrete(small_vec) => (
            small_vec.as_ptr().cast_mut(),
            small_vec.len().try_into().unwrap_or_else(|_| {
                panic!(
                    "The length of these qargs exceeds the capacity of {}",
                    u32::MAX
                )
            }),
        ),
    }
}
