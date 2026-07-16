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

use std::ffi::{CString, c_char};
use std::num::NonZero;
use std::ptr;

use qiskit_circuit::bit::ClassicalRegister;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::classical::expr::{Binary, BinaryOp, Expr, Value, Var};
use qiskit_circuit::classical::types::Type;
use qiskit_circuit::duration::Duration;
use qiskit_circuit::operations::{
    BoxDuration, CaseSpecifier, Condition, ControlFlow, ControlFlowInstruction, ForCollection,
    LoopParam, PyRange, SwitchTarget,
};
use qiskit_circuit::parameter::symbol_expr::Symbol;
use qiskit_circuit::{Clbit, Qubit};
use uuid::Uuid;

use crate::classical_expr::CDurationInfo;
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};
use num_traits::ToPrimitive;

use num_bigint::BigUint;
use qiskit_circuit::bit::{ShareableClbit, ShareableQubit};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{Param, StandardGate, StandardInstruction};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};

/// Represents a control flow instruction in a ``QkCircuit``.
///
/// This struct holds information about a control flow instruction and provides
/// access to its properties and nested circuit blocks. It also maintains mappings
/// of qubits and classical bits relative to the top-level circuit, which is
/// necessary for correctly interpreting nested control flow structures.
///
/// A pointer to this object is only valid as long as the source circuit remains alive.
/// Users must ensure the circuit is not freed while any ``QkControlFlowInstruction``
/// referencing it still exists.
pub struct CControlFlowInstruction {
    /// A borrowed pointer to the circuit this control-flow instruction lives in
    circuit: *const CircuitData,
    /// The index of the control flow instruction in the containing circuit
    inst_idx: usize,
    /// Qubit mapping of this instruction qargs w.r.t the top-level circuit
    pub(crate) qubit_map: Vec<u32>,
    /// Clbit mapping of this instruction qargs w.r.t the top-level circuit
    pub(crate) clbit_map: Vec<u32>,
}

impl CControlFlowInstruction {
    pub(crate) fn new(
        circuit: &CircuitData,
        inst_idx: usize,
        qubit_map: Vec<u32>,
        clbit_map: Vec<u32>,
    ) -> Self {
        Self {
            circuit: circuit as *const CircuitData,
            inst_idx,
            qubit_map,
            clbit_map,
        }
    }

    // Helper to get the circuit reference
    #[inline]
    fn circuit(&self) -> &CircuitData {
        // SAFETY: Per documentation and CControlFlowInstruction invariants, self.circuit is valid and non-null
        unsafe { const_ptr_as_ref(self.circuit) }
    }

    // Helper to get the PackedInstruction for this control flow instruction
    #[inline]
    fn instruction(&self) -> &PackedInstruction {
        let circuit = self.circuit();
        // Per documentation and CControlFlowInstruction invariants, inst_idx is a valid circuit index
        &circuit.data()[self.inst_idx]
    }

    // Helper to get the ControlFlowInstruction for this control flow instruction
    #[inline]
    fn control_flow_inst(&self) -> &ControlFlowInstruction {
        let inst = self.instruction();
        inst.op
            .try_control_flow()
            .unwrap_or_else(|| panic!("inst should be a control flow instruction"))
    }
}

/// This enum represents the different kinds of control flow instructions that can appear
/// in a quantum circuit.
#[repr(u8)]
pub enum CControlFlowKind {
    /// A Box instruction
    Box = 0,
    /// Break loop instruction
    BreakLoop = 1,
    /// Continue loop instruction
    ContinueLoop = 2,
    /// For loop instruction
    ForLoop = 3,
    /// If-else instruction
    IfElse = 4,
    /// Switch case instruction
    Switch = 5,
    /// While loop instruction
    While = 6,
}

impl From<&ControlFlowInstruction> for CControlFlowKind {
    fn from(value: &ControlFlowInstruction) -> Self {
        match value.control_flow {
            ControlFlow::Box { .. } => Self::Box,
            ControlFlow::BreakLoop => Self::BreakLoop,
            ControlFlow::ContinueLoop => Self::ContinueLoop,
            ControlFlow::ForLoop { .. } => Self::ForLoop,
            ControlFlow::IfElse { .. } => Self::IfElse,
            ControlFlow::Switch { .. } => Self::Switch,
            ControlFlow::While { .. } => Self::While,
        }
    }
}

/// Represents the type of condition or Switch target in a control flow instruction.
///
/// This enum is used to identify whether a condition (in IfElse or While instructions)
/// or a Switch target (in Switch instructions) operates on a classical bit, a classical
/// register, or a classical expression.
#[repr(u8)]
pub enum CConditionType {
    /// Condition based on a classical bit
    ClBit = 0,
    /// Condition based on a classical register
    ClReg = 1,
    /// Condition based on a classical expression
    Expr = 2,
}

impl From<&Condition> for CConditionType {
    fn from(value: &Condition) -> Self {
        match value {
            Condition::Bit(_, _) => Self::ClBit,
            Condition::Register(_, _) => Self::ClReg,
            Condition::Expr(_) => Self::Expr,
        }
    }
}

impl From<&SwitchTarget> for CConditionType {
    fn from(value: &SwitchTarget) -> Self {
        match value {
            SwitchTarget::Bit(_) => Self::ClBit,
            SwitchTarget::Register(_) => Self::ClReg,
            SwitchTarget::Expr(_) => Self::Expr,
        }
    }
}

/// Information about a classical bit condition.
///
/// This struct contains the details of a condition that operates on a single classical bit
#[repr(C)]
pub struct CConditionBitInfo {
    /// The index of the classical bit in the circuit
    clbit: u32,
    /// The expected value of the classical bit (true or false)
    condition: bool,
}

/// Represents the kind of duration specification for a Box instruction.
///
/// Box instructions can have no duration, a concrete duration value, or a duration
/// specified as a classical expression.
#[repr(u8)]
pub enum CBoxDurationKind {
    /// No duration specified
    NoDuration = 0,
    /// Concrete duration value (represented as `QkDurationInfo`)
    Duration = 1,
    /// Duration specified as a classical expression
    Expr = 2,
}

impl From<Option<&BoxDuration>> for CBoxDurationKind {
    fn from(value: Option<&BoxDuration>) -> Self {
        match value {
            None => Self::NoDuration,
            Some(BoxDuration::Duration(_)) => Self::Duration,
            Some(BoxDuration::Expr(_)) => Self::Expr,
        }
    }
}

/// Contains the labels for a Switch case.
///
/// This struct holds an array of label values that a Switch case matches against.
/// For example, a case like ``case(1, 2, 3)`` would have three labels: 1, 2, and 3.
/// The memory for the labels array is allocated by `qk_control_flow_switch_case_labels_uint`
/// and must be freed using `qk_control_flow_switch_case_labels_clear`.
#[repr(C)]
pub struct CSwitchCaseLabels {
    /// Pointer to an array of label values
    labels: *const u64,
    /// Number of labels in the array
    num_labels: usize,
}

/// The kind of loop parameter in a ForLoop control flow instruction.
///
/// This enum indicates whether a for-loop has no loop parameter, uses a Parameter symbol,
/// or uses a Variable (``QkVar``) to track the iteration value.
#[repr(u8)]
pub enum CLoopParamKind {
    /// No loop parameter is specified for the ForLoop instruction
    NoLoopParam = 0,
    /// Loop parameter is a Parameter
    Parameter = 1,
    /// Loop parameter is a Variable
    Variable = 2,
}

/// The type of symbol in a ForLoop context.
///
/// This enum indicates whether a symbol is a standalone variable or an element
/// inside a parameter vector accessed by an index.
#[repr(u8)]
pub enum CSymbolType {
    /// A standalone symbol with a simple name
    Standalone = 0,
    /// An element symbol with a base name and index
    Element = 1,
}

/// Symbol information including type and data.
///
/// This struct represents a symbol in a ForLoop context, which can be
/// either a standalone variable or an indexed element.
#[repr(C)]
pub struct CSymbolInfo {
    /// The type of symbol (standalone or element)
    ty: CSymbolType,
    /// A null-terminated C string containing the symbol name. For standalone symbols,
    /// this is the variable name. For element symbols, this is the base name of the parameter
    /// vector. The caller is responsible for freeing this string using `qk_str_free`.
    name: *mut c_char,
    /// For element symbols, this is the index into the parameter vector. For standalone
    /// symbols, this field is unused and should be ignored.
    index: usize,
}

/// The type of collection used in a ForLoop control flow instruction.
#[repr(u8)]
pub enum CLoopCollectionType {
    /// The loop iterates over an explicit list of elements
    List = 0,
    /// The loop iterates over a Python-style range (start, stop, step)
    Range = 1,
}

/// A struct containing loop elements from a ForLoop control flow instruction.
///
/// This struct is returned by `qk_control_flow_loop_elements` and contains both
/// a pointer to the array of loop elements and the number of elements in the array.
/// The pointer is borrowed and must not be freed by the caller.
#[repr(C)]
pub struct CLoopElements {
    /// Pointer to the array of loop elements.
    elements: *const isize,
    /// Number of elements in the array.
    len: usize,
}

/// @ingroup QkControlFlow
/// Get the kind of a control flow instruction.
///
/// @param cf_inst A pointer to the control flow instruction.
///
/// @return The kind of control flow instruction.
///
/// # Example
/// ```c
/// // Assuming cf_inst is obtained from a circuit with control flow
/// QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
/// switch (kind) {
/// case QkControlFlowKind_Box:
///     // do something with the box instruction
///     break;
/// case QkControlFlowKind_BreakLoop:
///     // do something with the break loop instruction
///     break;
/// case QkControlFlowKind_ContinueLoop:
///     // do something with the continue loop instruction
///     break;
/// case QkControlFlowKind_ForLoop:
///     // do something with the for loop instruction
///     break;
/// case QkControlFlowKind_IfElse:
///     // do something with the if-else instruction
///     break;
/// case QkControlFlowKind_Switch:
///     // do something with the switch instruction
///     break;
/// case QkControlFlowKind_While:
///     // do something with the while loop instruction
///     break;
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_kind(
    cf_inst: *const CControlFlowInstruction,
) -> CControlFlowKind {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let cf_inst = cf_inst.control_flow_inst();

    CControlFlowKind::from(cf_inst)
}

/// @ingroup QkControlFlow
/// Get the number of circuit blocks in a control flow instruction.
///
/// @param cf_inst A pointer to the control flow instruction.
///
/// @return The number of circuit blocks contained in this control flow instruction.
///
/// # Example
/// ```c
/// // Assuming cf_inst is obtained from a circuit with control flow
/// size_t num_blocks = qk_control_flow_num_blocks(cf_inst);
/// for (size_t i = 0; i < num_blocks; i++) {
///     const QkCircuit *block_circuit = qk_control_flow_block_circuit(cf_inst, i);
///     // Process each block...
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_num_blocks(
    cf_inst: *const CControlFlowInstruction,
) -> usize {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let instruction = cf_inst.instruction();

    instruction.blocks_view().len()
}

/// @ingroup QkControlFlow
/// Get a pointer to a circuit block within a control flow instruction.
///
/// Control flow instructions contain one or more circuit blocks (e.g., IfElse has two blocks,
/// Switch may have multiple blocks). This function retrieves a specific block by index.
///
/// @param cf_inst A pointer to the control flow instruction.
/// @param block_idx The index of the block to retrieve.
///     ``block_idx`` must be within bounds (< `qk_control_flow_num_blocks`).
///
/// @return A pointer to the ``QkCircuit`` representing the requested block.
///     The circuit is valid as long as the control flow instruction exists.
///     The circuit is owned by the control flow instruction and must not be freed by the caller.
///
/// # Example
/// ```c
/// // Assuming cf_inst is an IfElse control flow instruction
/// const QkCircuit *true_block = qk_control_flow_block_circuit(cf_inst, 0);
/// const QkCircuit *false_block = qk_control_flow_block_circuit(cf_inst, 1);
/// // Process the true and false blocks...
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_block_circuit(
    cf_inst: *const CControlFlowInstruction,
    block_idx: usize,
) -> *const CircuitData {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let block_ids = cf_inst.instruction().blocks_view();
    // Per documentation, block_idx must be within bounds.
    let block_circuit = &cf_inst.circuit().blocks()[block_ids[block_idx]];
    ptr::from_ref(block_circuit)
}

/// @ingroup QkControlFlow
/// Get the qubit mapping for a control flow instruction.
///
/// Returns a pointer to an array that maps the qubits used in the control flow instruction's
/// blocks to their indices in the top-level circuit. The array length equals the number of
/// qubits used by the control flow instruction. For each qubit index ``i`` in the nested block,
/// the mapping at index ``i`` in the array gives the corresponding qubit index in the top-level circuit.
///
/// @param cf_inst A pointer to the control flow instruction.
///
/// @return A pointer to an array of ``uint32_t`` values representing the qubit mapping.
///     The array is valid as long as the control flow instruction exists.
///     The array is owned by the control flow instruction and must not be freed by the caller.
///
/// # Example
/// ```c
/// // Assuming cf_inst is obtained from a circuit with control flow
/// const uint32_t *qubit_map = qk_control_flow_qubit_map(cf_inst);
/// uint32_t num_qubits = qk_circuit_num_qubits(qk_control_flow_block_circuit(cf_inst, 0));
/// for (uint32_t i = 0; i < num_qubits; i++) {
///     printf("Block qubit %u maps to circuit qubit %u\n", i, qubit_map[i]);
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_qubit_map(
    cf_inst: *const CControlFlowInstruction,
) -> *const u32 {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    cf_inst.qubit_map.as_ptr()
}

/// @ingroup QkControlFlow
/// Get the classical bit mapping for a control flow instruction.
///
/// Returns a pointer to an array that maps the classical bits used in the control flow
/// instruction's blocks to their indices in the top-level circuit. The array length equals
/// the number of classical bits used by the control flow instruction. For each classical bit
/// index ``i`` in the nested block, the mapping at index ``i`` in the array gives the corresponding
/// classical bit index in the top-level circuit.
///
/// @param cf_inst A pointer to the control flow instruction.
///
/// @return A pointer to an array of ``uint32_t`` values representing the classical bit mapping.
///     The array is valid as long as the control flow instruction exists.
///     The array is owned by the control flow instruction and must not be freed by the caller.
///
/// # Example
/// ```c
/// // Assuming cf_inst is obtained from a circuit with control flow
/// const uint32_t *clbit_map = qk_control_flow_clbit_map(cf_inst);
/// uint32_t num_clbits = qk_circuit_num_clbits(qk_control_flow_block_circuit(cf_inst, 0));
/// for (uint32_t i = 0; i < num_clbits; i++) {
///     printf("Block clbit %u maps to circuit clbit %u\n", i, clbit_map[i]);
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_clbit_map(
    cf_inst: *const CControlFlowInstruction,
) -> *const u32 {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    cf_inst.clbit_map.as_ptr()
}

/// @ingroup QkControlFlow
/// Get the condition type for a control flow instruction.
///
/// Returns the type of condition used in an IfElse or While instruction.
/// The condition type indicates whether the condition is based on a classical bit,
/// classical register or a classical expression.
///
/// @param cf_inst A valid pointer to a ``QkControlFlowInstruction`` that must represent
///     an IfElse or While instruction.
///
/// @return A `QkConditionType` enum value indicating the condition type.
///
/// Panics if ``cf_inst`` is not an IfElse or While control flow instruction.
///
/// # Example
/// ```c
/// // Assuming cf_inst is an IfElse or While control flow instruction
/// QkConditionType cond_type = qk_control_flow_condition_type(cf_inst);
/// switch (cond_type) {
/// case QkConditionType_ClBit:
///     // do something with classical bit...
///     break;
/// case QkConditionType_ClReg:
///     // do something with classical register...
///     break;
/// case QkConditionType_Expr:
///     // Process expression...
///     break;
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_condition_type(
    cf_inst: *const CControlFlowInstruction,
) -> CConditionType {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let cf_inst = cf_inst.control_flow_inst();

    match &cf_inst.control_flow {
        ControlFlow::IfElse { condition } | ControlFlow::While { condition } => {
            CConditionType::from(condition)
        }
        _ => panic!("Expected either an IfElse or a While control flow instruction"),
    }
}

/// @ingroup QkControlFlow
/// Get the classical bit condition information for a control flow instruction.
///
/// Extracts the classical bit index and expected value from an IfElse or While
/// instruction that has a classical bit condition.
///
/// @param cf_inst A valid pointer to a ``QkControlFlowInstruction`` that must represent
///     an IfElse or While instruction with a classical bit condition.
///
/// @return A `QkConditionBitInfo` struct containing the classical bit index and expected value.
///
/// Panics if ``cf_inst`` is not an IfElse or While control flow instruction,
/// or if the condition is not a bit type.
///
/// # Example
/// ```c
/// // Assuming cf_inst is an IfElse or While instruction with a bit condition
/// QkConditionBitInfo bit_info = qk_control_flow_condition_bit_info(cf_inst);
/// printf("Condition: clbit[%u] == %s\n", bit_info.clbit, bit_info.condition ? "true" : "false");
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_condition_bit_info(
    cf_inst: *const CControlFlowInstruction,
) -> CConditionBitInfo {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let condition = match &cf_inst.control_flow_inst().control_flow {
        ControlFlow::IfElse { condition } | ControlFlow::While { condition } => condition,
        _ => panic!("Expected either an IfElse or a While control flow instruction"),
    };

    let Condition::Bit(clbit, cond) = condition else {
        panic!("Expected a classical bit condition for the instruction")
    };

    CConditionBitInfo {
        clbit: cf_inst
            .circuit()
            .clbit_index(clbit)
            .expect("Classical bit should be in the containing circuit"),
        condition: *cond,
    }
}

/// @ingroup QkControlFlow
/// Get the bit width of the classical register condition for a control flow instruction.
///
/// Extracts the bit width of the classical register from an IfElse or While
/// instruction that has a classical register condition.
///
/// @param cf_inst A valid pointer to a ``QkControlFlowInstruction`` that must represent
///     an IfElse or While instruction with a classical register condition.
///
/// @return The bit width of the condition over the classical register.
///
/// Panics if ``cf_inst`` is not an IfElse or While control flow instruction,
/// or if the condition is not a register type.
///
/// # Example
/// ```c
/// // Assuming cf_inst is an IfElse or While instruction with a register condition
/// uint64_t bit_width = qk_control_flow_condition_reg_cond_bit_width(cf_inst);
/// printf("Register bit width: %lu\n", bit_width);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_condition_reg_cond_bit_width(
    cf_inst: *const CControlFlowInstruction,
) -> u64 {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let condition = match &cf_inst.control_flow_inst().control_flow {
        ControlFlow::IfElse { condition } | ControlFlow::While { condition } => condition,
        _ => {
            panic!("Expected either an IfElse or a While control flow instruction")
        }
    };

    let Condition::Register(_, cond) = condition else {
        panic!("Expected a register condition for the instruction")
    };

    cond.bits()
}

/// @ingroup QkControlFlow
/// Get the classical register for a control flow instruction condition.
///
/// Extracts the classical register from an IfElse or While instruction that has
/// a classical register condition.
///
/// @param cf_inst A valid pointer to a ``QkControlFlowInstruction`` that must represent
///     an IfElse or While instruction with a classical register condition.
///
/// @return A borrowed pointer to the ``QkClassicalRegister``. The pointer remains valid
///     as long as the parent circuit remains valid.
///
/// Panics if ``cf_inst`` is not an IfElse or While control flow instruction,
/// or if the condition is not a register type.
///
/// # Example
/// ```c
/// // Assuming cf_inst is an IfElse or While instruction with a register condition
/// const QkClassicalRegister* creg = qk_control_flow_condition_reg(cf_inst);
/// // Use the classical register pointer
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_condition_reg(
    cf_inst: *const CControlFlowInstruction,
) -> *const ClassicalRegister {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let condition = match &cf_inst.control_flow_inst().control_flow {
        ControlFlow::IfElse { condition } | ControlFlow::While { condition } => condition,
        _ => {
            panic!("Expected either an IfElse or a While control flow instruction")
        }
    };

    let Condition::Register(creg, _) = condition else {
        panic!("Expected a register condition for the instruction")
    };

    ptr::from_ref(creg)
}

/// @ingroup QkControlFlow
/// Get the condition value of the classical register condition for a control flow instruction.
///
/// Extracts the condition as an unsigned integer value from an IfElse or While
/// instruction that has a classical register condition.
///
/// @param cf_inst A valid pointer to a ``QkControlFlowInstruction`` that must represent
///     an IfElse or While instruction with a classical register condition.
///
/// @return The condition value.
///
/// Panics if ``cf_inst`` is not an IfElse or While control flow instruction,
/// if the condition is not a register type, or if the condition value does not
/// fit in a ``uint64_t``.
///
/// # Example
/// ```c
/// // Assuming cf_inst is an IfElse or While instruction with a register condition
/// uint64_t expected_value = qk_control_flow_condition_reg_cond_uint(cf_inst);
/// printf("Expected register value: %lu\n", expected_value);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_condition_reg_cond_uint(
    cf_inst: *const CControlFlowInstruction,
) -> u64 {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let condition = match &cf_inst.control_flow_inst().control_flow {
        ControlFlow::IfElse { condition } | ControlFlow::While { condition } => condition,
        _ => {
            panic!("Expected either an IfElse or a While control flow instruction")
        }
    };

    let Condition::Register(_, cond) = condition else {
        panic!("Expected a register condition for the instruction")
    };

    cond.to_u64()
        .unwrap_or_else(|| panic!("Condition value too large to fit in uint64_t"))
}

/// @ingroup QkControlFlow
/// Get the classical expression for a control flow instruction.
///
/// Extracts the classical expression from an IfElse or While instruction that has
/// an expression-based condition.
///
/// @param cf_inst A valid pointer to a ``QkControlFlowInstruction`` that must represent
///     an IfElse or While instruction with an expression condition.
///
/// @return A borrowed pointer to the ``QkExprNode`` representing the classical expression.
///     The pointer remains valid as long as the parent circuit remains valid.
///
/// Panics if ``cf_inst`` is not an IfElse or While control flow instruction,
/// or if the condition is not an expression type.
///
/// # Example Usage
/// ```c
/// // Assuming while_inst is a While control flow instruction with an expression condition
/// QkConditionType cond_type = qk_control_flow_condition_type(while_inst);
/// if (cond_type == QkConditionType_Expr) {
///     const QkExprNode *expr = qk_control_flow_condition_expr(while_inst);
///     // Use the expression...
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_condition_expr(
    cf_inst: *const CControlFlowInstruction,
) -> *const Expr {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let expr = match &cf_inst.control_flow_inst().control_flow {
        ControlFlow::IfElse { condition } | ControlFlow::While { condition } => {
            if let Condition::Expr(expr) = condition {
                expr
            } else {
                panic!("Condition must be an expression condition")
            }
        }
        _ => panic!("Expected either an IfElse or a While control flow instruction"),
    };

    ptr::from_ref(expr)
}

/// @ingroup QkControlFlow
/// Get the duration kind of a Box control flow instruction.
///
/// Box instructions can have no duration, a concrete duration value as `QkDurationInfo`, or a duration
/// specified as an expression. This function returns which kind of duration is present.
///
/// @param cf_inst A pointer to the control flow instruction.
///     The control flow instruction must be of a Box kind.
///
/// @return The kind of duration as `QkBoxDurationKind`.
///
/// Panics if ``cf_inst`` is not a Box control flow instruction.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a Box control flow instruction
/// QkBoxDurationKind duration_kind = qk_control_flow_box_duration_kind(cf_inst);
/// switch (duration_kind) {
/// case QkBoxDurationKind_NoDuration:
///     // do something...
///     break;
/// case QkBoxDurationKind_Duration:
///     // do something...
///     break;
/// case QkBoxDurationKind_Expr:
///     // do something...
///     break;
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_box_duration_kind(
    cf_inst: *const CControlFlowInstruction,
) -> CBoxDurationKind {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::Box { duration, .. } = &cf_inst.control_flow_inst().control_flow else {
        panic!("Expected a Box control flow instruction")
    };

    CBoxDurationKind::from(duration.as_ref())
}

/// @ingroup QkControlFlow
/// Get the duration value information of a Box control flow instruction.
///
/// This function retrieves the duration value and type for a Box instruction that has
/// a concrete duration.
///
/// @param cf_inst A pointer to the control flow instruction.
///     The control flow instruction must be of a Box kind with a concrete duration.
///
/// @return A `QkDurationInfo` struct containing the duration value and unit.
///
/// Panics if ``cf_inst`` is not a Box control flow instruction with a concrete duration.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a Box instruction with a concrete duration
///  QkDurationInfo duration_info = qk_control_flow_box_duration_val_info(cf_inst);
///  if (duration_info.ty == QkDurationType_Dt) {
///      int64_t dt = duration_info.value.dt;
///  } else {
///      double time = duration_info.value.time;
///  }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_box_duration_val_info(
    cf_inst: *const CControlFlowInstruction,
) -> CDurationInfo {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::Box {
        duration: Some(BoxDuration::Duration(duration)),
        ..
    } = &cf_inst.control_flow_inst().control_flow
    else {
        panic!("Control flow instruction must be a Box with a concrete duration")
    };

    CDurationInfo::from(duration)
}

/// @ingroup QkControlFlow
/// Get the duration expression of a Box control flow instruction.
///
/// This function retrieves a pointer to the expression that defines the duration for a
/// Box instruction when the duration is specified as an expression.
///
/// @param cf_inst A pointer to the control flow instruction.
///     The control flow instruction must be of a Box kind with an expression duration.
///
/// @return A pointer to the ``QkExpr`` representing the duration expression.
///     The expression is valid as long as the control flow instruction exists.
///     The expression is owned by the control flow instruction and must not be freed by the caller.
///
/// Panics if ``cf_inst`` is not a Box control flow instruction with an expression duration.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a Box instruction with an expression duration
/// const QkExpr *duration_expr = qk_control_flow_box_duration_expr(cf_inst);
/// // Use the expression to evaluate or analyze the duration...
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_box_duration_expr(
    cf_inst: *const CControlFlowInstruction,
) -> *const Expr {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::Box {
        duration: Some(BoxDuration::Expr(expr)),
        ..
    } = &cf_inst.control_flow_inst().control_flow
    else {
        panic!("Control flow instruction must be a Box with an expression duration")
    };

    ptr::from_ref(expr)
}

/// @ingroup QkControlFlow
/// Get the type of collection used in a ForLoop control flow instruction.
///
/// This function determines whether a ForLoop iterates over an explicit list of elements
/// or a Python-style range.
///
/// @param cf_inst A pointer to a control flow instruction that must be a ForLoop.
///
/// @return A ``CLoopCollectionType`` enum value indicating the collection type.
///
/// Panics if ``cf_inst`` is not a ForLoop control flow instruction.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a ForLoop instruction
/// QkLoopCollectionType collection_type = qk_control_flow_loop_collection_type(cf_inst);
/// if (collection_type == QkLoopCollectionType_List) {
///     // Handle list-based loop
/// } else if (collection_type == QkLoopCollectionType_Range) {
///     // Handle range-based loop
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_loop_collection_type(
    cf_inst: *const CControlFlowInstruction,
) -> CLoopCollectionType {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::ForLoop { collection, .. } = &cf_inst.control_flow_inst().control_flow else {
        panic!("Expected a ForLoop control flow instruction")
    };

    match collection {
        ForCollection::List(_) => CLoopCollectionType::List,
        ForCollection::PyRange(_) => CLoopCollectionType::Range,
    }
}

/// @ingroup QkControlFlow
/// Get the list of elements that a ForLoop iterates over.
///
/// This function retrieves the list of elements from a ForLoop control flow instruction
/// that uses an explicit list collection. Use `qk_control_flow_loop_collection_type`
/// to determine the collection type before calling this function.
///
/// @param cf_inst A pointer to a control flow instruction that must be a ForLoop with a List collection.
///
/// @return A `QkLoopElements` struct containing a pointer to the array of loop elements and the number
///     of elements. The pointer is borrowed for the duration of the control flow instruction and must
///     not be freed by the caller.
///
/// Panics if ``cf_inst`` is not a ForLoop control flow instruction with a list collection.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a ForLoop instruction with a List collection type
/// QkLoopElements loop_elements = qk_control_flow_loop_elements(cf_inst);
/// for (size_t i = 0; i < loop_elements.len; i++) {
///     printf("Element %zu: %zu\n", i, loop_elements.elements[i]);
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_loop_elements(
    cf_inst: *const CControlFlowInstruction,
) -> CLoopElements {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::ForLoop {
        collection: ForCollection::List(elements),
        ..
    } = &cf_inst.control_flow_inst().control_flow
    else {
        panic!("Expected a ForLoop control flow instruction with a List collection")
    };

    CLoopElements {
        elements: elements.as_ptr(),
        len: elements.len(),
    }
}

/// @ingroup QkControlFlow
/// Get the range parameters of a ForLoop that iterates over a Python-style range.
///
/// This function retrieves the start, stop, and step values from a ForLoop control flow
/// instruction that uses a range collection. Use `qk_control_flow_loop_collection_type`
/// to determine the collection type before calling this function.
///
/// @param cf_inst A pointer to a control flow instruction that must be a ForLoop with a Range collection.
/// @param out_start An output parameter that will be set to the range start value.
/// @param out_stop An output parameter that will be set to the range stop value.
/// @param out_step An output parameter that will be set to the range step value.
///
/// Panics if ``cf_inst`` is not a ForLoop control flow instruction with a range collection.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a ForLoop instruction with a Range collection type
/// int64_t start, stop, step;
/// qk_control_flow_loop_range(cf_inst, &start, &stop, &step);
/// printf("Loop range: start=%ld, stop=%ld, step=%ld\n", start, stop, step);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``,
/// or if any of ``out_start``, ``out_stop``, or ``out_step`` are not aligned valid pointers to write to.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_loop_range(
    cf_inst: *const CControlFlowInstruction,
    out_start: *mut i64,
    out_stop: *mut i64,
    out_step: *mut i64,
) {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::ForLoop {
        collection: ForCollection::PyRange(range),
        ..
    } = &cf_inst.control_flow_inst().control_flow
    else {
        panic!("Expected a ForLoop control flow instruction with a Range collection");
    };

    // SAFETY: Per documentation, out_start, out_stop, and out_step are aligned valid pointers to write to.
    unsafe { *out_start = range.start as i64 };
    unsafe { *out_stop = range.stop as i64 };
    unsafe { *out_step = range.step.get() as i64 };
}

/// @ingroup QkControlFlow
/// Get the kind of loop parameter used in a ForLoop control flow instruction.
///
/// This function determines whether a ForLoop has no loop parameter, uses a Parameter symbol,
/// or uses a Variable (``QkVar``) to track the iteration value.
///
/// @param cf_inst A pointer to a control flow instruction that must be a ForLoop.
///
/// @return A `QkLoopParamKind` enum value indicating the loop parameter kind.
///
/// Panics if ``cf_inst`` is not a ForLoop control flow instruction.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a ForLoop instruction
/// QkLoopParamKind param_kind = qk_control_flow_loop_param_kind(cf_inst);
/// if (param_kind == QkLoopParamKind_Parameter) {
///     // Loop uses a Parameter symbol
/// } else if (param_kind == QkLoopParamKind_Variable) {
///     // Loop uses a Variable
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_loop_param_kind(
    cf_inst: *const CControlFlowInstruction,
) -> CLoopParamKind {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::ForLoop { loop_param, .. } = &cf_inst.control_flow_inst().control_flow else {
        panic!("Expected a ForLoop control flow instruction")
    };

    match loop_param {
        None => CLoopParamKind::NoLoopParam,
        Some(LoopParam::Parameter(_)) => CLoopParamKind::Parameter,
        Some(LoopParam::Variable(_)) => CLoopParamKind::Variable,
    }
}

/// @ingroup QkControlFlow
/// Get the loop parameter symbol information from a ForLoop control flow instruction.
///
/// This function retrieves the loop parameter symbol information from a ForLoop instruction
/// that uses a Parameter as its loop parameter. The returned `QkSymbolInfo` contains the symbol's
/// type, name, and index (for parameter vector element symbols).
///
/// @param cf_inst A valid pointer to a ``QkControlFlowInstruction`` that must represent a ForLoop
///     with a Parameter loop parameter.
///
/// @return A `QkSymbolInfo` struct containing the symbol information. The caller must free
///     the returned string in the ``name`` field using `qk_str_free`.
///
/// Panics if ``cf_inst`` is not a ForLoop control flow instruction with a Parameter loop parameter.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a ForLoop control flow instruction with a Parameter loop parameter
/// QkSymbolInfo symbol_info = qk_control_flow_loop_symbol_info(cf_inst);
/// if (symbol_info.ty == QkSymbolType_Standalone) {
///     printf("Loop parameter: %s\n", symbol_info.name);
/// } else if (symbol_info.ty == QkSymbolType_Element) {
///     printf("Loop parameter: %s[%zu]\n", symbol_info.name, symbol_info.index);
/// }
/// qk_str_free(symbol_info.name);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_loop_symbol_info(
    cf_inst: *const CControlFlowInstruction,
) -> CSymbolInfo {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::ForLoop {
        loop_param: Some(LoopParam::Parameter(symbol)),
        ..
    } = &cf_inst.control_flow_inst().control_flow
    else {
        panic!("Expected a ForLoop control flow instruction with a loop parameter")
    };

    match symbol {
        Symbol::Standalone { name, .. } => {
            CSymbolInfo {
                ty: CSymbolType::Standalone,
                name: CString::new(name.as_str())
                    .expect("Symbol should have a valid name")
                    .into_raw(),
                index: 0, // dummy value in the StandAlone case
            }
        }
        Symbol::Element { index, base } => CSymbolInfo {
            ty: CSymbolType::Element,
            index: *index,
            name: CString::new(base.name.as_str())
                .expect("Parameter vector should have a valid name")
                .into_raw(),
        },
    }
}

/// @ingroup QkControlFlow
/// Get the loop variable from a ForLoop control flow instruction.
///
/// This function retrieves a pointer to the Variable (``QkVar``) used as the loop parameter
/// in a ForLoop instruction. The returned pointer is borrowed for the lifetime of the
/// control flow instruction and must not be freed by the caller.
///
/// @param cf_inst A valid pointer to a ``QkControlFlowInstruction`` that must represent a ForLoop
///     with a Variable loop parameter.
///
/// @return A pointer to the ``QkVar`` representing the loop variable. The pointer is valid
///     for the lifetime of the control flow instruction.
///
/// Panics if ``cf_inst`` is not a ForLoop control flow instruction with a Variable loop parameter.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a ForLoop control flow instruction with a Variable loop parameter
/// const QkVar *loop_var = qk_control_flow_loop_variable(cf_inst);
/// // Use the loop variable pointer to access variable information
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_loop_variable(
    cf_inst: *const CControlFlowInstruction,
) -> *const Var {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::ForLoop {
        loop_param: Some(LoopParam::Variable(var)),
        ..
    } = &cf_inst.control_flow_inst().control_flow
    else {
        panic!("Expected a ForLoop control flow instruction with a loop variable")
    };

    ptr::from_ref(var)
}

/// @ingroup QkControlFlow
/// Get the type of the Switch target for a Switch control flow instruction.
///
/// Switch statements can operate on different types of targets: a classical bit,
/// a classical register, or an expression. This function returns which type of
/// target the Switch statement uses.
///
/// @param cf_inst A pointer to the control flow instruction.
///     The control flow instruction must be of a Switch kind.
///
/// @return The condition type of the Switch target as `QkConditionType`.
///
/// Panics if ``cf_inst`` is not a Switch control flow instruction.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a Switch control flow instruction
/// QkConditionType target_type = qk_control_flow_switch_target_type(cf_inst);
/// switch (target_type) {
/// case QkConditionType_ClBit:
///     // do something
///     break;
/// case QkConditionType_ClReg:
///     // do something
///     break;
/// case QkConditionType_Expr:
///     // do something
///     break;
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_switch_target_type(
    cf_inst: *const CControlFlowInstruction,
) -> CConditionType {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::Switch { target, .. } = &cf_inst.control_flow_inst().control_flow else {
        panic!("Expected a Switch control flow instruction")
    };

    CConditionType::from(target)
}

/// @ingroup QkControlFlow
/// Get the classical bit index for a Switch target.
///
/// This function retrieves the index of the classical bit that a Switch statement
/// operates on when the Switch target is a classical bit.
///
/// @param cf_inst A pointer to the control flow instruction.
///     The control flow instruction must be of a Switch kind with a classical bit target.
///
/// @return The index of the classical bit in the circuit.
///
/// Panics if ``cf_inst`` is not a Switch control flow instruction with a classical bit target.
///  
/// # Example
/// ```c
/// // Assuming cf_inst is a Switch instruction with a classical bit target
/// uint32_t clbit_idx = qk_control_flow_switch_target_bit(cf_inst);
/// printf("Switch operates on clbit %u\n", clbit_idx);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_switch_target_bit(
    cf_inst: *const CControlFlowInstruction,
) -> u32 {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::Switch {
        target: SwitchTarget::Bit(clbit),
        ..
    } = &cf_inst.control_flow_inst().control_flow
    else {
        panic!("Expected a Switch control flow instruction with a classical bit target")
    };

    cf_inst
        .circuit()
        .clbit_index(clbit)
        .expect("Classical bit should be found in circuit")
}

/// @ingroup QkControlFlow
/// Get the classical register that a Switch statement operates on.
///
/// This function retrieves the classical register used as the Switch target when
/// the Switch operates on a register.
///
/// @param cf_inst A pointer to the control flow instruction.
///     The control flow instruction must be of a Switch kind with a register target.
///
/// @return A pointer to the ``QkClassicalRegister`` that the Switch operates on.
///     The register is valid as long as the control flow instruction exists.
///     The register is owned by the control flow instruction and must not be freed by the caller.
///
/// Panics if ``cf_inst`` is not a Switch control flow instruction with a register target.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a Switch instruction with a register target
/// const QkClassicalRegister *reg = qk_control_flow_switch_target_register(cf_inst);
/// // Use the register to get its name, size, etc...
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_switch_target_register(
    cf_inst: *const CControlFlowInstruction,
) -> *const ClassicalRegister {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::Switch {
        target: SwitchTarget::Register(reg),
        ..
    } = &cf_inst.control_flow_inst().control_flow
    else {
        panic!("Expected a Switch control flow instruction with a register target")
    };

    ptr::from_ref(reg)
}

/// @ingroup QkControlFlow
/// Get the expression for a Switch target.
///
/// This function retrieves a pointer to the classical expression that a Switch statement
/// operates on when the Switch target is an expression.
///
/// @param cf_inst A pointer to the control flow instruction.
///     The control flow instruction must be of a Switch kind with an expression target.
///
/// @return A pointer to the ``QkExpr`` representing the Switch target expression.
///     The expression is valid as long as the control flow instruction exists.
///     The expression is owned by the control flow instruction and must not be freed by the caller.
///
/// Panics if ``cf_inst`` is not a Switch control flow instruction with an expression target.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a Switch instruction with an expression target
/// const QkExpr *target_expr = qk_control_flow_switch_target_expr(cf_inst);
/// // Evaluate the expression...
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_switch_target_expr(
    cf_inst: *const CControlFlowInstruction,
) -> *const Expr {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::Switch {
        target: SwitchTarget::Expr(expr),
        ..
    } = &cf_inst.control_flow_inst().control_flow
    else {
        panic!("Expected a Switch control flow instruction with an expression target")
    };

    ptr::from_ref(expr)
}

/// @ingroup QkControlFlow
/// Get the number of cases in a Switch statement.
///
/// Returns the total number of case blocks in the Switch statement, including
/// the default case if present.
///
/// @param cf_inst A pointer to the control flow instruction.
///     The control flow instruction must be of a Switch kind.
///
/// @return The number of cases in the Switch statement.
///
/// Panics if ``cf_inst`` is not a Switch control flow instruction.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a Switch control flow instruction
/// size_t num_cases = qk_control_flow_switch_num_cases(cf_inst);
/// for (size_t i = 0; i < num_cases; i++) {
///     // Analyze the case...
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_switch_num_cases(
    cf_inst: *const CControlFlowInstruction,
) -> usize {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::Switch { label_spec, .. } = &cf_inst.control_flow_inst().control_flow else {
        panic!("Expected a Switch control flow instruction")
    };

    label_spec.len()
}

/// @ingroup QkControlFlow
/// Check if a specific case in a Switch statement is the default case.
///
/// Switch statements can have a default case that matches when no other cases match.
/// This function checks whether the case at the given index is the default case.
/// The default case may also include leading labels, which can be retrieved by
/// `qk_control_flow_switch_case_labels_uint` on the default case.
///
/// @param cf_inst A pointer to the control flow instruction.
///     The control flow instruction must be of a Switch kind.
/// @param case_idx The index of the case to check. Must be less than the value
///     returned by `qk_control_flow_switch_num_cases`.
///
/// @return ``true`` if the case at ``case_idx`` is the default case, ``false`` otherwise.
///
/// Panics if ``cf_inst`` is not a Switch control flow instruction.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a Switch control flow instruction
/// for (size_t i = 0; i < qk_control_flow_switch_num_cases(cf_inst); i++) {
///     if (qk_control_flow_switch_is_case_default(cf_inst, i)) {
///         printf("Case %zu is the default case\n", i);
///     }
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_switch_is_case_default(
    cf_inst: *const CControlFlowInstruction,
    case_idx: usize,
) -> bool {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::Switch { label_spec, .. } = &cf_inst.control_flow_inst().control_flow else {
        panic!("Expected a Switch control flow instruction")
    };

    matches!(label_spec[case_idx].last(), Some(CaseSpecifier::Default))
}

/// @ingroup QkControlFlow
/// Get the maximum bit width required to represent the labels in a Switch case.
///
/// This function returns the maximum number of bits needed to represent any of the
/// unsigned integer labels in the specified case. This is useful for determining
/// the minimum bit width required to store or process the case labels.
///
/// @param cf_inst A pointer to the control flow instruction.
///     The control flow instruction must be of a Switch kind.
/// @param case_idx The index of the case whose label bit width to retrieve. Must be less than
///     the value returned by `qk_control_flow_switch_num_cases`.
///
/// @return The maximum bit width of the labels in the case, or 0 if the case has no
///     unsigned integer labels (e.g., only a DEFAULT specifier).
///
/// Panics if ``cf_inst`` is not a Switch control flow instruction.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a Switch control flow instruction
/// uint64_t bit_width = qk_control_flow_switch_case_labels_bit_width(cf_inst, 0);
/// printf("Maximum bit width for case 0: %lu\n", bit_width);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_switch_case_labels_bit_width(
    cf_inst: *const CControlFlowInstruction,
    case_idx: usize,
) -> u64 {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::Switch { label_spec, .. } = &cf_inst.control_flow_inst().control_flow else {
        panic!("Expected a Switch control flow instruction")
    };

    label_spec[case_idx]
        .iter()
        .filter_map(|label| {
            if let CaseSpecifier::Uint(biguint) = label {
                Some(biguint.bits())
            } else {
                None
            }
        })
        .max()
        .unwrap_or_default()
}

/// @ingroup QkControlFlow
/// Get the labels for a specific case in a Switch statement.
///
/// Each case in a Switch statement can have one or more labels (e.g., ``case(1, 2, 3)``)
/// has three labels: 1, 2, and 3). This function retrieves all labels for a given case.
/// Note that the default case can also include labels, which can be retrieved by this function.
/// When called on a default case without additional labels, the function sets ``num_labels = 0`` and
/// ``labels = NULL`` in the returned `QkSwitchCaseLabels` struct.
/// Before calling this function, you should use `qk_control_flow_switch_case_labels_bit_width`
/// to ensure the labels will fit in ``uint64_t``.
///
/// @param cf_inst A pointer to the control flow instruction.
///     The control flow instruction must be of a Switch kind.
/// @param case_idx The index of the case whose labels to retrieve. Must be less than
///     the value returned by `qk_control_flow_switch_num_cases`.
///
/// @return A `QkSwitchCaseLabels` struct with the label information for the given case.
///     The struct will contain a pointer to an array of labels and the number of labels.
///     If ``num_labels > 0``, you should call `qk_control_flow_switch_case_labels_clear`
///     to free the memory allocated by this function.
///
/// Panics if ``cf_inst`` is not a Switch control flow instruction or if a case label
/// does not fit in ``uint64_t``.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a Switch control flow instruction
/// QkSwitchCaseLabels case_labels = qk_control_flow_switch_case_labels_uint(cf_inst, 0);
/// for (size_t i = 0; i < case_labels.num_labels; i++) {
///     printf("Label %zu: %lu\n", i, case_labels.labels[i]);
/// }
/// qk_control_flow_switch_case_labels_clear(&case_labels);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer to a ``QkControlFlowInstruction``,
/// or if `out_labels` is not a valid pointer to a `QkSwitchCaseLabels` struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_switch_case_labels_uint(
    cf_inst: *const CControlFlowInstruction,
    case_idx: usize,
) -> CSwitchCaseLabels {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a CControlFlowInstruction.
    let cf_inst = unsafe { const_ptr_as_ref(cf_inst) };

    let ControlFlow::Switch { label_spec, .. } = &cf_inst.control_flow_inst().control_flow else {
        panic!("Expected a Switch control flow instruction")
    };

    let labels = label_spec[case_idx]
        .iter()
        .filter_map(|l| {
            if let CaseSpecifier::Uint(label) = l {
                Some(
                    label
                        .to_u64()
                        .unwrap_or_else(|| panic!("Case label too large to fit in uint64_t")),
                )
            } else {
                None
            }
        })
        .collect::<Vec<u64>>()
        .into_boxed_slice();

    CSwitchCaseLabels {
        num_labels: labels.len(),
        labels: if labels.is_empty() {
            ptr::null() // occurs for a default case without additional labels
        } else {
            Box::into_raw(labels) as *const u64
        },
    }
}

/// @ingroup QkControlFlow
/// Clear a `QkSwitchCaseLabels` struct.
///
/// This function must be called to free the memory allocated by
/// `qk_control_flow_switch_case_labels_uint`. After calling this function,
/// the labels pointer in the struct will be set to null and the
/// count will be set to zero.
///
/// @param labels A pointer to the `QkSwitchCaseLabels` struct to clear.
///     The struct must have been previously populated by
///     `qk_control_flow_switch_case_labels_uint`.
///
/// # Example
/// ```c
/// // Assuming cf_inst is a Switch control flow instruction
/// QkSwitchCaseLabels case_labels = qk_control_flow_switch_case_labels_uint(cf_inst, 0);
/// // Use the labels...
/// qk_control_flow_switch_case_labels_clear(&case_labels);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``labels`` is not a valid pointer to a `QkSwitchCaseLabels`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_switch_case_labels_clear(labels: *mut CSwitchCaseLabels) {
    // SAFETY: Per documentation, labels is a valid pointer to a CSwitchCaseLabels.
    let labels = unsafe { mut_ptr_as_ref(labels) };

    if !labels.labels.is_null() && labels.num_labels > 0 {
        drop(unsafe {
            // SAFETY: Per document unsafe, label is a valid CSwitchCaseLabels
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                labels.labels as *mut u64,
                labels.num_labels,
            ))
        });

        labels.labels = std::ptr::null();
        labels.num_labels = 0;
    }
}

//////////////////////////////////////////////////////////////////
// The functions below are used in the C testing, to generate   //
// various objects for testing the C API. These functions       //
// should be removed once we have the actual C API for creating //
// control flow operations.                                     //
//////////////////////////////////////////////////////////////////

// This function creates a CircuitData object for testing. The content of the circuit is as follows:
// +-------+------------------+------------------------------------------------------------------------------+
// | Index | Instruction Type | Description                                                                  |
// +-------+------------------+------------------------------------------------------------------------------+
// |   0   | Box              | Inner circuit with CX(0,1) and Measure(0->0), mapped to qubits [2,0],        |
// |       |                  | clbit [1], duration=0.1s                                                     |
// +-------+------------------+------------------------------------------------------------------------------+
// |   1   | For Loop         | Loop over [1,2] with loop_parameter=Parameter("x") and                       |
// |       |                  | If-Else(clbit[0]==True: Break, else: Continue), H(0) in the body             |
// +-------+------------------+------------------------------------------------------------------------------+
// |   2   | Switch           | Target: cr, Cases: {(1<<80)-1->X(0), 1,2,3->H(1), (1<<64)-1,DEFAULT->Y(2)}   |
// +-------+------------------+------------------------------------------------------------------------------+
// |   3   | While Loop       | Condition: clbit[1]==False, Body: X(0)                                       |
// +-------+------------------+------------------------------------------------------------------------------+
// |   4   | While Loop       | Condition: cr==7, Body: Y(0)                                                 |
// +-------+------------------+------------------------------------------------------------------------------+
// |   5   | While Loop       | Condition: cr<7 (expression), Body: Z(0)                                     |
// +-------+------------------+------------------------------------------------------------------------------+
// |   6   | Switch           | Target: clbit[0], Cases: {DEFAULT->X(0)}                                     |
// +-------+------------------+------------------------------------------------------------------------------+
// |   7   | Switch           | Target: cr<2 (expression), Cases: {DEFAULT->Y(0)}                            |
// +-------+------------------+------------------------------------------------------------------------------+
// |   8   | For Loop         | Loop over range(1,10,3), loop_parameter=expr.Var.new("v"), types.Uint(8)     |
// +-------+------------------+------------------------------------------------------------------------------+
// |   9   | Switch           | Condition: cr==(1<<80)-1, Body: Z(0)                                         |
// +-------+------------------+------------------------------------------------------------------------------+
/// cbindgen:qk-vtable-rules=[no-export]
/// cbindgen:no-export
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inner_test_control_flow_circuit() -> *mut CircuitData {
    let qubits: Vec<ShareableQubit> = (0..3).map(|_| ShareableQubit::new_anonymous()).collect();
    let clbits: Vec<ShareableClbit> = (0..3).map(|_| ShareableClbit::new_anonymous()).collect();

    let mut circuit = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create circuit");

    //////////////////////////////////////////////////////
    // Build a box like this:
    // ----------------------
    // inner = QuantumCircuit(2,1)
    // inner.cx(0,1)
    // inner.measure(0,0)
    // qc.box(inner, [2,0], [1], duration=0.1, unit='s')
    let inner_qubits: Vec<ShareableQubit> =
        (0..2).map(|_| ShareableQubit::new_anonymous()).collect();
    let inner_clbits: Vec<ShareableClbit> =
        (0..1).map(|_| ShareableClbit::new_anonymous()).collect();

    let mut inner_circuit = CircuitData::new(
        Some(inner_qubits.clone()),
        Some(inner_clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create inner circuit");

    inner_circuit
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::CX),
            None,
            &[Qubit(0), Qubit(1)],
            &[],
        )
        .expect("Failed to add CX gate");

    inner_circuit
        .push_packed_operation(
            PackedOperation::from_standard_instruction(StandardInstruction::Measure),
            None,
            &[Qubit(0)],
            &[Clbit(0)],
        )
        .expect("Failed to add measure");

    let box_block = circuit.add_block(inner_circuit);
    let box_op = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::Box {
            duration: Some(BoxDuration::Duration(Duration::s(0.1))),
            annotations: Default::default(),
        },
        num_qubits: 2,
        num_clbits: 1,
    });

    circuit
        .push_packed_operation(
            box_op,
            Some(Parameters::Blocks(vec![box_block])),
            &[Qubit(2), Qubit(0)],
            &[Clbit(1)],
        )
        .expect("Failed to add box");

    /////////////////////////////////////////////
    // Build a for-loop like this:
    // ----------------------
    // with qc.for_loop([1,2], loop_parameter=Parameter("x")):
    //     qc.h(0)
    //     qc.cx(0, 1)
    //     qc.measure(0, 0)
    //     with qc.if_test((0, True)) as else_:
    //         qc.break_loop()
    //     with else_:
    //         qc.continue_loop()
    //     qc.h(0)
    let mut for_loop_body = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create for loop body");

    for_loop_body
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::H),
            None,
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add H gate");

    for_loop_body
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::CX),
            None,
            &[Qubit(0), Qubit(1)],
            &[],
        )
        .expect("Failed to add CX gate");

    for_loop_body
        .push_packed_operation(
            PackedOperation::from_standard_instruction(StandardInstruction::Measure),
            None,
            &[Qubit(0)],
            &[Clbit(0)],
        )
        .expect("Failed to add measure");

    let mut then_block = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create then block");

    let break_op = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::BreakLoop,
        num_qubits: 0,
        num_clbits: 0,
    });
    then_block
        .push_packed_operation(break_op, None, &[], &[])
        .expect("Failed to add break_loop");

    let mut else_block_inner = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create else block");

    let continue_op = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::ContinueLoop,
        num_qubits: 0,
        num_clbits: 0,
    });
    else_block_inner
        .push_packed_operation(continue_op, None, &[], &[])
        .expect("Failed to add continue_loop");

    // Add if-else to for loop body
    let then_block_id = for_loop_body.add_block(then_block);
    let else_block_id = for_loop_body.add_block(else_block_inner);
    let if_else_op = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::IfElse {
            condition: Condition::Bit(clbits[0].clone(), true),
        },
        num_qubits: 2,
        num_clbits: 1,
    });
    for_loop_body
        .push_packed_operation(
            if_else_op,
            Some(Parameters::Blocks(vec![then_block_id, else_block_id])),
            &[Qubit(0), Qubit(1)],
            &[Clbit(0)],
        )
        .expect("Failed to add if-else to for loop");

    // Add final H gate after if-else
    for_loop_body
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::H),
            None,
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add final H gate");

    // Add the for loop to the main circuit
    let for_block = circuit.add_block(for_loop_body);
    let for_op = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::ForLoop {
            collection: ForCollection::List(vec![1, 2]),
            loop_param: Some(LoopParam::Parameter(Symbol::Standalone {
                name: "x".to_owned(),
                uuid: Uuid::new_v4(),
            })),
        },
        num_qubits: 2,
        num_clbits: 1,
    });
    circuit
        .push_packed_operation(
            for_op,
            Some(Parameters::Blocks(vec![for_block])),
            &[Qubit(0), Qubit(1)],
            &[Clbit(0)],
        )
        .expect("Failed to add for loop");

    ////////////////////////////////////
    // Build a switch-case like this:
    // ----------------------
    // with qc.switch(cr) as case:
    //     with case((1 << 80) - 1):
    //         qc.x(0)
    //     with case(1, 2, 3):
    //         qc.h(1)
    //     with case((1 << 64) - 1, case.DEFAULT):
    //         qc.y(2)
    let mut first_case_block = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create case 0 block");

    first_case_block
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::X),
            None,
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add X gate to case 0");

    let mut second_case_block = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create case 1,2,3 block");

    second_case_block
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::H),
            None,
            &[Qubit(1)],
            &[],
        )
        .expect("Failed to add H gate to case 1,2,3");

    let mut default_case_block = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create default case block");

    default_case_block
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::Y),
            None,
            &[Qubit(2)],
            &[],
        )
        .expect("Failed to add Y gate to default case");

    let first_case_id = circuit.add_block(first_case_block);
    let second_case_id = circuit.add_block(second_case_block);
    let default_case_id = circuit.add_block(default_case_block);

    let creg = ClassicalRegister::new_owning("cr", 2);

    let switch_op = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::Switch {
            target: SwitchTarget::Register(creg),
            cases: 3,
            label_spec: vec![
                vec![CaseSpecifier::Uint((BigUint::from(1u32) << 80) - 1u32)],
                vec![
                    CaseSpecifier::Uint(BigUint::from(1u32)),
                    CaseSpecifier::Uint(BigUint::from(2u32)),
                    CaseSpecifier::Uint(BigUint::from(3u32)),
                ],
                vec![
                    CaseSpecifier::Uint((BigUint::from(1u32) << 64) - 1u32),
                    CaseSpecifier::Default,
                ],
            ],
        },
        num_qubits: 3,
        num_clbits: 3,
    });

    circuit
        .push_packed_operation(
            switch_op,
            Some(Parameters::Blocks(vec![
                first_case_id,
                second_case_id,
                default_case_id,
            ])),
            &[Qubit(0), Qubit(1), Qubit(2)],
            &[Clbit(0), Clbit(1), Clbit(2)],
        )
        .expect("Failed to add switch instruction");

    ///////////////////////////////////////
    // Build a while loop like this:
    // ----------------------
    // with qc.while_loop((cr[1], False)):
    //     qc.x(0)
    let mut while_block1 = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create while block 1");

    while_block1
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::X),
            None,
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add X gate to while block 1");

    let while_block1_id = circuit.add_block(while_block1);
    let while_op1 = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::While {
            condition: Condition::Bit(clbits[1].clone(), false),
        },
        num_qubits: 1,
        num_clbits: 0,
    });

    circuit
        .push_packed_operation(
            while_op1,
            Some(Parameters::Blocks(vec![while_block1_id])),
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add first while loop");

    ///////////////////////////////////////
    // Build a while loop like this:
    // ----------------------
    // with qc.while_loop((cr, 7)):
    //     qc.y(0)
    let mut while_block2 = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create while block 2");

    while_block2
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::Y),
            None,
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add Y gate to while block 2");

    let while_block2_id = circuit.add_block(while_block2);
    let creg2 = ClassicalRegister::new_owning("cr", 2);
    let while_op2 = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::While {
            condition: Condition::Register(creg2, BigUint::from(7u32)),
        },
        num_qubits: 1,
        num_clbits: 0,
    });

    circuit
        .push_packed_operation(
            while_op2,
            Some(Parameters::Blocks(vec![while_block2_id])),
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add second while loop");

    ///////////////////////////////////////
    // Build a while loop like this:
    // ----------------------
    // with qc.while_loop(expr.less(cr, 7)):
    //     qc.z(0)
    //
    let mut while_block3 = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create while block 3");

    while_block3
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::Z),
            None,
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add Z gate to while block 3");

    let while_block3_id = circuit.add_block(while_block3);

    // Create a simple expression: less(cr, 7)
    let creg3 = ClassicalRegister::new_owning("cr", 2);

    let creg_var = Expr::Var(Var::Register {
        register: creg3.clone(),
        ty: Type::Uint(2),
    });

    let seven = Expr::Value(Value::Uint {
        raw: BigUint::from(7u32),
        ty: Type::Uint(2),
    });

    let expr_condition = Expr::Binary(Box::new(Binary {
        op: BinaryOp::Less,
        left: creg_var,
        right: seven,
        ty: Type::Bool,
        constant: false,
    }));

    let while_op3 = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::While {
            condition: Condition::Expr(expr_condition),
        },
        num_qubits: 1,
        num_clbits: 0,
    });

    circuit
        .push_packed_operation(
            while_op3,
            Some(Parameters::Blocks(vec![while_block3_id])),
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add third while loop");

    ///////////////////////////////////
    // Build a switch-case like this:
    // ----------------------
    // with qc.switch(cr[0]) as case:
    //     with case(case.DEFAULT):
    //         qc.x(0)
    //
    let mut switch_bit_block = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create switch bit block");

    switch_bit_block
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::X),
            None,
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add X gate to switch bit block");

    let switch_bit_block_id = circuit.add_block(switch_bit_block);

    let switch_bit_op = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::Switch {
            target: SwitchTarget::Bit(clbits[0].clone()),
            cases: 1,
            label_spec: vec![vec![CaseSpecifier::Default]],
        },
        num_qubits: 3,
        num_clbits: 3,
    });

    circuit
        .push_packed_operation(
            switch_bit_op,
            Some(Parameters::Blocks(vec![switch_bit_block_id])),
            &[Qubit(0), Qubit(1), Qubit(2)],
            &[Clbit(0), Clbit(1), Clbit(2)],
        )
        .expect("Failed to add switch on bit instruction");

    /////////////////////////////////////////////
    // Build a switch-case like this:
    // ----------------------
    // with qc.switch(expr.less(cr, 2)) as case:
    //     with case(case.DEFAULT):
    //         qc.y(0)
    let mut switch_expr_block = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create switch expr block");

    switch_expr_block
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::Y),
            None,
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add Y gate to switch expr block");

    let switch_expr_block_id = circuit.add_block(switch_expr_block);

    // Create expression: less(cr, 2)
    let creg4 = ClassicalRegister::new_owning("cr", 2);

    let creg_var2 = Expr::Var(Var::Register {
        register: creg4.clone(),
        ty: Type::Uint(2),
    });

    let two = Expr::Value(Value::Uint {
        raw: BigUint::from(2u32),
        ty: Type::Uint(2),
    });

    let expr_switch = Expr::Binary(Box::new(Binary {
        op: BinaryOp::Less,
        left: creg_var2,
        right: two,
        ty: Type::Bool,
        constant: false,
    }));

    let switch_expr_op = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::Switch {
            target: SwitchTarget::Expr(expr_switch),
            cases: 1,
            label_spec: vec![vec![CaseSpecifier::Default]],
        },
        num_qubits: 3,
        num_clbits: 3,
    });

    circuit
        .push_packed_operation(
            switch_expr_op,
            Some(Parameters::Blocks(vec![switch_expr_block_id])),
            &[Qubit(0), Qubit(1), Qubit(2)],
            &[Clbit(0), Clbit(1), Clbit(2)],
        )
        .expect("Failed to add switch on expression instruction");

    /////////////////////////////////////////////
    // Build a for-loop like this:
    // ----------------------
    // with qc.for_loop(range(1,10,3), expr.Var.new("v"), types.Uint(8)):
    //      qc.y(0)
    let mut for_loop_range_body = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create for loop range body");

    for_loop_range_body
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::Y),
            None,
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add Y gate");

    // Add the for loop with range to the main circuit
    let for_range_block = circuit.add_block(for_loop_range_body);
    let for_range_op = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::ForLoop {
            collection: ForCollection::PyRange(PyRange {
                start: 1,
                stop: 10,
                step: NonZero::new(3).unwrap(),
            }),
            loop_param: Some(LoopParam::Variable(Var::Standalone {
                uuid: Uuid::new_v4().as_u128(),
                name: "v".to_owned(),
                ty: Type::Uint(8),
            })),
        },
        num_qubits: 3,
        num_clbits: 3,
    });
    circuit
        .push_packed_operation(
            for_range_op,
            Some(Parameters::Blocks(vec![for_range_block])),
            &[Qubit(0), Qubit(1), Qubit(2)],
            &[Clbit(0), Clbit(1), Clbit(2)],
        )
        .expect("Failed to add for loop with range");

    ///////////////////////////////////////
    // Build a while loop like this:
    // ----------------------
    // with qc.while_loop((cr, (1<<80)-1)):
    //     qc.z(0)
    let mut while_block2 = CircuitData::new(
        Some(qubits.clone()),
        Some(clbits.clone()),
        Param::Float(0.0),
    )
    .expect("Failed to create while block 2");

    while_block2
        .push_packed_operation(
            PackedOperation::from_standard_gate(StandardGate::Z),
            None,
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add Z gate to while block 2");

    let while_block2_id = circuit.add_block(while_block2);
    let creg2 = ClassicalRegister::new_owning("cr", 2);
    let while_op2 = PackedOperation::from(ControlFlowInstruction {
        control_flow: ControlFlow::While {
            condition: Condition::Register(creg2, (BigUint::from(1u32) << 80) - 1u32),
        },
        num_qubits: 1,
        num_clbits: 0,
    });

    circuit
        .push_packed_operation(
            while_op2,
            Some(Parameters::Blocks(vec![while_block2_id])),
            &[Qubit(0)],
            &[],
        )
        .expect("Failed to add second while loop");

    Box::into_raw(Box::new(circuit))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_switch_label_handling() {
        let qubits = vec![ShareableQubit::new_anonymous()];

        let creg = ClassicalRegister::new_owning("cr", 2);
        let clbits: Vec<ShareableClbit> = (0..creg.len())
            .map(|i| creg.get(i).expect("index in range"))
            .collect();

        let mut circuit = CircuitData::new(Some(qubits), Some(clbits), Param::Float(0.0)).unwrap();

        // Create a case block with a simple gate
        let case_block = CircuitData::new(None, None, Param::Float(0.0)).unwrap();
        let case_block_id = circuit.add_block(case_block);

        // Create a classical register for the switch target

        // Create a switch instruction with one case and labels (1, 2, 3)
        let switch_op = PackedOperation::from(ControlFlowInstruction {
            control_flow: ControlFlow::Switch {
                target: SwitchTarget::Register(creg),
                cases: 1,
                label_spec: vec![vec![
                    CaseSpecifier::Uint(BigUint::from(1u32)),
                    CaseSpecifier::Uint(BigUint::from(2u32)),
                    CaseSpecifier::Uint(BigUint::from(3u32)),
                ]],
            },
            num_qubits: 1,
            num_clbits: 2,
        });

        circuit
            .push_packed_operation(
                switch_op,
                Some(Parameters::Blocks(vec![case_block_id])),
                &[Qubit(0)],
                &[Clbit(0), Clbit(1)],
            )
            .expect("Failed to add switch instruction");

        let cf_inst = CControlFlowInstruction::new(&circuit, 0, vec![0], vec![0, 1]);

        let mut case_labels = unsafe { qk_control_flow_switch_case_labels_uint(&cf_inst, 0) };

        assert_eq!(case_labels.num_labels, 3, "Expected 3 labels");
        assert!(
            !case_labels.labels.is_null(),
            "Labels pointer should not be null"
        );

        unsafe {
            let labels_slice =
                std::slice::from_raw_parts(case_labels.labels, case_labels.num_labels);
            assert_eq!(labels_slice[0], 1, "First label should be 1");
            assert_eq!(labels_slice[1], 2, "Second label should be 2");
            assert_eq!(labels_slice[2], 3, "Third label should be 3");
        }

        unsafe {
            qk_control_flow_switch_case_labels_clear(&mut case_labels);
        }

        assert_eq!(
            case_labels.num_labels, 0,
            "num_labels should be 0 after clear"
        );
        assert!(
            case_labels.labels.is_null(),
            "labels pointer should be null after clear"
        );
    }
}
