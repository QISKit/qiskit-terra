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

use anyhow::Error;
use hashbrown::HashMap;
use num_complex::Complex64;
use smallvec::smallvec;

use crate::exit_codes::ExitCode;
use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGError, NodeIndex, NodeType};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    ArrayType, Operation, OperationRef, Param, StandardGate, StandardInstruction, UnitaryGate,
};
use qiskit_circuit::{Clbit, Qubit};

use crate::circuit::{CBlocksMode, CInstruction, CVarsMode};

use crate::circuit::unitary_from_pointer;
use crate::pointers::{check_ptr, const_ptr_as_ref, mut_ptr_as_ref};

/// @ingroup QkDag
/// Construct a new empty DAG.
///
/// You must free the returned DAG with ``qk_dag_free`` when done with it.
///
/// @return A pointer to the created DAG.
///
/// # Example
/// ```c
/// QkDag *empty = qk_dag_new();
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn qk_dag_new() -> *mut DAGCircuit {
    let dag = DAGCircuit::new();
    Box::into_raw(Box::new(dag))
}

/// @ingroup QkDag
/// Add a quantum register to the DAG.
///
/// @param dag A pointer to the DAG.
/// @param reg A pointer to the quantum register.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1024, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag`` and
/// if ``reg`` is not a valid, non-null pointer to a ``QkQuantumRegister``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_add_quantum_register(
    dag: *mut DAGCircuit,
    reg: *const QuantumRegister,
) {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    // SAFETY: Per documentation, the pointer is to valid data.
    let qreg = unsafe { const_ptr_as_ref(reg) };

    dag.add_qreg(qreg.clone())
        .expect("Invalid register unable to be added to DAG");
}

/// @ingroup QkDag
/// Add a classical register to the DAG.
///
/// @param dag A pointer to the DAG.
/// @param reg A pointer to the classical register.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkClassicalRegister *cr = qk_classical_register_new(24, "my_register");
/// qk_dag_add_classical_register(dag, cr);
/// qk_classical_register_free(cr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag`` and
/// if ``reg`` is not a valid, non-null pointer to a ``QkClassicalRegister``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_add_classical_register(
    dag: *mut DAGCircuit,
    reg: *const ClassicalRegister,
) {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    // SAFETY: Per documentation, the pointer is to valid data.
    let creg = unsafe { const_ptr_as_ref(reg) };

    dag.add_creg(creg.clone())
        .expect("Invalid register unable to be added to DAG");
}

/// @ingroup QkDag
/// Get the number of qubits the DAG contains.
///
/// @param dag A pointer to the DAG.
///
/// @return The number of qubits the DAG is defined on.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(24, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
/// uint32_t num_qubits = qk_dag_num_qubits(dag);  // num_qubits==24
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_num_qubits(dag: *const DAGCircuit) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.num_qubits() as u32
}

/// @ingroup QkDag
/// Get the number of clbits the DAG contains.
///
/// @param dag A pointer to the DAG.
///
/// @return The number of clbits the DAG is defined on.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkClassicalRegister *cr = qk_classical_register_new(24, "my_register");
/// qk_dag_add_classical_register(dag, cr);
/// uint32_t num_clbits = qk_dag_num_clbits(dag);  // num_clbits==24
/// qk_classical_register_free(cr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_num_clbits(dag: *const DAGCircuit) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.num_clbits() as u32
}

/// @ingroup QkDag
/// Return the total number of operation nodes in the DAG.
///
/// @param dag A pointer to the DAG.
///
/// @return The total number of instructions in the DAG.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
///
/// uint32_t qubit[1] = {0};
/// qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
/// size_t num = qk_dag_num_op_nodes(dag); // 1
///
/// qk_dag_free(dag);
/// qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_num_op_nodes(dag: *const DAGCircuit) -> usize {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.num_ops()
}

/// @ingroup QkDag
/// Get the global phase of the DAG.
///
/// This function returns a copy of the DAG's global phase
/// and the value must be freed via :c:func:`qk_param_free`
/// after usage.
///
/// @param dag A pointer to the DAG.
///
/// @return The global phase of the DAG.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(24, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
/// QkParam *global_phase = qk_dag_global_phase(dag);
/// qk_param_free(global_phase);
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_global_phase(dag: *const DAGCircuit) -> *mut Param {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    Box::into_raw(Box::new(dag.global_phase().clone()))
}

/// @ingroup QkDag
/// Set the global phase of the DAG.
///
/// This function copies the new global phase upon setting it,
/// so the caller retains ownership of the ``QkParam`` phase,
/// and the value of the phase must be freed via :c:func:`qk_param_free`
/// after setting.
///
/// @param dag A pointer to the DAG.
/// @param phase A pointer to the global phase to set.
///
/// @return ``QkExitCode_Success`` upon successful setting of the global phase. Upon failure,
///     ``QkExitCode_ParameterError`` describes generic failures when attempting to
///     track the parameter symbols such as invalid parameter values.
///     Otherwise, ``QkExitCode_DagError`` indicates a DAG-specific cause of the failure.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkParam *new_global_phase = qk_param_from_double(1.23);
/// qk_dag_set_global_phase(dag, new_global_phase);
/// qk_param_free(new_global_phase);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag`` and
/// if ``phase`` is not a valid, non-null pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_set_global_phase(
    dag: *mut DAGCircuit,
    phase: *const Param,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    let phase = unsafe { const_ptr_as_ref(phase) };
    // dag.set_global_phase_param(phase.clone())
    //     .expect("Unable to set global phase");
    match dag.set_global_phase_param(phase.clone()) {
        Ok(_) => ExitCode::Success,
        Err(DAGError::ObjGlobalPhase) => ExitCode::ParameterError,
        Err(_) => ExitCode::DagError,
    }
}

/// The type of node in a ``QkDag``.
///
/// Operation nodes represent an applied instruction. The rest of the nodes are
/// considered "wire" nodes and represent the endpoints of the DAG's data dependency
/// chains.
#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum CDagNodeType {
    /// Operation node.
    Operation = 0,
    /// Qubit wire start node.
    QubitIn = 1,
    /// Qubit wire end node.
    QubitOut = 2,
    /// Clbit wire start node.
    ClbitIn = 3,
    /// Clbit wire end node.
    ClbitOut = 4,
    /// Classical variable wire start node.
    VarIn = 5,
    /// Classical variable wire end node.
    VarOut = 6,
}

/// @ingroup QkDag
/// Get the type of the specified node.
///
/// The result can be used in a switch statement to dispatch proper handling
/// when iterating over nodes of unknown type.
///
/// @param dag A pointer to the DAG.
/// @param node The node to get the type of.
///
/// @return The type of the node.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_node_type(dag: *const DAGCircuit, node: u32) -> CDagNodeType {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    match dag.dag()[NodeIndex::new(node as usize)] {
        NodeType::QubitIn(_) => CDagNodeType::QubitIn,
        NodeType::QubitOut(_) => CDagNodeType::QubitOut,
        NodeType::ClbitIn(_) => CDagNodeType::ClbitIn,
        NodeType::ClbitOut(_) => CDagNodeType::ClbitOut,
        NodeType::VarIn(_) => CDagNodeType::VarIn,
        NodeType::VarOut(_) => CDagNodeType::VarOut,
        NodeType::Operation(_) => CDagNodeType::Operation,
    }
}

/// @ingroup QkDag
/// Retrieve the index of the input node of the wire corresponding to the given qubit.
///
/// @param dag A pointer to the DAG.
/// @param qubit The qubit to get the input node index of.
///
/// @return The input node of the qubit wire.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_qubit_in_node(dag: *const DAGCircuit, qubit: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.qubit_io_map()[qubit as usize][0].index() as u32
}

/// @ingroup QkDag
/// Retrieve the index of the output node of the wire corresponding to the given qubit.
///
/// @param dag A pointer to the DAG.
/// @param qubit The qubit to get the output node index of.
///
/// @return The output node of the qubit wire.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_qubit_out_node(dag: *const DAGCircuit, qubit: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.qubit_io_map()[qubit as usize][1].index() as u32
}

/// @ingroup QkDag
/// Retrieve the index of the input node of the wire corresponding to the given clbit.
///
/// @param dag A pointer to the DAG.
/// @param clbit The clbit to get the input node index of.
///
/// @return The input node of the clbit wire.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_clbit_in_node(dag: *const DAGCircuit, clbit: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.clbit_io_map()[clbit as usize][0].index() as u32
}

/// @ingroup QkDag
/// Retrieve the index of the output node of the wire corresponding to the given clbit.
///
/// @param dag A pointer to the DAG.
/// @param clbit The clbit to get the output node index of.
///
/// @return The output node of the clbit wire.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_clbit_out_node(dag: *const DAGCircuit, clbit: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.clbit_io_map()[clbit as usize][1].index() as u32
}

/// @ingroup QkDag
/// Retrieve the value of a wire endpoint node.
///
/// @param dag A pointer to the DAG.
/// @param node The endpoint node to get the wire value of.
///
/// @return The value (e.g. qubit, clbit, or var) within the endpoint node.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_wire_node_value(dag: *const DAGCircuit, node: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    (match dag.dag()[NodeIndex::new(node as usize)] {
        NodeType::QubitIn(value) => value.index(),
        NodeType::QubitOut(value) => value.index(),
        NodeType::ClbitIn(value) => value.index(),
        NodeType::ClbitOut(value) => value.index(),
        NodeType::VarIn(value) => value.index(),
        NodeType::VarOut(value) => value.index(),
        NodeType::Operation(_) => panic!("Specified node is not a wire endpoint node."),
    }) as u32
}

/// @ingroup QkDag
/// Gets the number of qubits of the specified operation node.
///
/// Panics if the node is not an operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the number of qubits of.
///
/// @return The number of qubits of the operation.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_op_node_num_qubits(dag: *const DAGCircuit, node: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag.dag()[NodeIndex::new(node as usize)].unwrap_operation();
    instr.op.num_qubits()
}

/// @ingroup QkDag
/// Gets the number of clbits of the specified operation node.
///
/// Panics if the node is not an operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the number of clbits of.
///
/// @return The number of clbits of the operation.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_op_node_num_clbits(dag: *const DAGCircuit, node: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag.dag()[NodeIndex::new(node as usize)].unwrap_operation();
    instr.op.num_clbits()
}

/// @ingroup QkDag
/// Gets the number of params of the specified operation node.
///
/// Panics if the node is not an operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the number of params of.
///
/// @return The number of params of the operation.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_op_node_num_params(dag: *const DAGCircuit, node: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag.dag()[NodeIndex::new(node as usize)].unwrap_operation();
    instr.op.num_params()
}

/// @ingroup QkDag
/// Retrieve the qubits of the specified operation node.
///
/// Panics if the node is not an operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the qubits of.
///
/// @return A pointer to the qubits. Use ``qk_dag_op_node_num_qubits`` to determine the number of
///     elements.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_op_node_qubits(dag: *const DAGCircuit, node: u32) -> *const u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag.dag()[NodeIndex::new(node as usize)].unwrap_operation();
    dag.qargs_interner().get(instr.qubits).as_ptr().cast()
}

/// @ingroup QkDag
/// Retrieve the clbits of the specified operation node.
///
/// Panics if the node is not an operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the clbits of.
///
/// @return A pointer to the clbits. Use ``qk_dag_op_node_num_clbits`` to determine the number of
///     elements.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_op_node_clbits(dag: *const DAGCircuit, node: u32) -> *const u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag.dag()[NodeIndex::new(node as usize)].unwrap_operation();
    dag.cargs_interner().get(instr.clbits).as_ptr().cast()
}

/// @ingroup QkDag
/// Apply a ``QkGate`` to the DAG.
///
/// @param dag A pointer to the DAG to apply the gate to.
/// @param gate The StandardGate to apply.
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the gate on. This
///     can be a null pointer if there are no qubits for ``gate`` (e.g. ``QkGate_GlobalPhase``).
/// @param params The pointer to the array of ``double`` values to use for the gate parameters.
///     This can be a null pointer if there are no parameters for ``gate`` (e.g. ``QkGate_H``).
/// @param front If ``true``, the gate is applied as the first operation on the specified qubits,
///     rather than as the last.
///
/// @return The index of the newly added operation node.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
///
/// uint32_t qubit[1] = {0};
/// qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
///
/// qk_dag_free(dag);
/// qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// The ``qubits`` and ``params`` types are expected to be a pointer to an array of ``uint32_t``
/// and ``double`` respectively where the length is matching the expectations for the standard
/// gate. If the array is insufficiently long the behavior of this function is undefined as this
/// will read outside the bounds of the array. It can be a null pointer if there are no qubits
/// or params for a given gate. You can check ``qk_gate_num_qubits`` and ``qk_gate_num_params`` to
/// determine how many qubits and params are required for a given gate.
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_apply_gate(
    dag: *mut DAGCircuit,
    gate: StandardGate,
    qubits: *const u32,
    params: *const f64,
    front: bool,
) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    // SAFETY: Per the documentation the qubits and params pointers are arrays of num_qubits()
    // and num_params() elements respectively.
    unsafe {
        let qargs: &[Qubit] = match gate.num_qubits() {
            0 => &[],
            1 => &[Qubit(*qubits.wrapping_add(0))],
            2 => &[
                Qubit(*qubits.wrapping_add(0)),
                Qubit(*qubits.wrapping_add(1)),
            ],
            3 => &[
                Qubit(*qubits.wrapping_add(0)),
                Qubit(*qubits.wrapping_add(1)),
                Qubit(*qubits.wrapping_add(2)),
            ],
            4 => &[
                Qubit(*qubits.wrapping_add(0)),
                Qubit(*qubits.wrapping_add(1)),
                Qubit(*qubits.wrapping_add(2)),
                Qubit(*qubits.wrapping_add(3)),
            ],
            // There are no ``QkGate``s > 4 qubits
            _ => panic!(),
        };
        let params = match gate.num_params() {
            0 => None,
            1 => Some(smallvec![(*params.wrapping_add(0)).into()]),
            2 => Some(smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
            ]),
            3 => Some(smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
            ]),
            4 => Some(smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
                (*params.wrapping_add(3)).into(),
            ]),
            // There are no ``QkGate``s that take > 4 params
            _ => panic!(),
        };
        let new_node = if front {
            dag.apply_operation_front(
                gate.into(),
                qargs,
                &[],
                params.map(Parameters::Params),
                None,
                #[cfg(feature = "cache_pygates")]
                None,
            )
            .unwrap()
        } else {
            dag.apply_operation_back(
                gate.into(),
                qargs,
                &[],
                params.map(Parameters::Params),
                None,
                #[cfg(feature = "cache_pygates")]
                None,
            )
            .unwrap()
        };
        new_node.index() as u32
    }
}

/// @ingroup QkDag
/// Apply a measure to a DAG.
///
/// @param dag The circuit to apply to.
/// @param qubit The qubit index to measure.
/// @param clbit The clbit index to store the result in.
/// @param front Whether to apply the measure at the start of the circuit. Usually `false`.
///
/// @return The node index of the created instruction.
///
/// # Example
///
/// Measure all qubits into the corresponding clbit index at the end of the circuit.
///
/// ```c
/// uint32_t num_qubits = qk_dag_num_qubits(dag);
/// for (uint32_t i = 0; i < num_qubits; i++) {
///     qk_dag_apply_measure(dag, i, i, false);
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if `dag` is not an aligned, non-null pointer to a valid ``QkDag``,
/// or if `qubit` or `clbit` are out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_apply_measure(
    dag: *mut DAGCircuit,
    qubit: u32,
    clbit: u32,
    front: bool,
) -> u32 {
    // SAFETY: per documentation, `dag` points to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    if front {
        dag.apply_operation_front(
            StandardInstruction::Measure.into(),
            &[Qubit(qubit)],
            &[Clbit(clbit)],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .expect("caller is responsible for passing inbounds bits")
        .index() as u32
    } else {
        dag.apply_operation_back(
            StandardInstruction::Measure.into(),
            &[Qubit(qubit)],
            &[Clbit(clbit)],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .expect("caller is responsible for passing inbounds bits")
        .index() as u32
    }
}

/// @ingroup QkDag
/// Apply a reset to the DAG.
///
/// @param dag The circuit to apply to.
/// @param qubit The qubit index to reset.
/// @param front Whether to apply the reset at the start of the circuit. Usually `false`.
///
/// @return The node index of the created instruction.
///
/// # Examples
///
/// Apply initial resets on all qubits.
///
/// ```c
/// uint32_t num_qubits = qk_dag_num_qubits(dag);
/// for (uint32_t qubit = 0; qubit < num_qubits; qubit++) {
///     qk_dag_apply_reset(dag, qubit, true);
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if `dag` is not an aligned, non-null pointer to a valid ``QkDag``,
/// or if `qubit` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_apply_reset(dag: *mut DAGCircuit, qubit: u32, front: bool) -> u32 {
    // SAFETY: per documentation, `dag` points to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    if front {
        dag.apply_operation_front(
            StandardInstruction::Reset.into(),
            &[Qubit(qubit)],
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .expect("caller is responsible for passing inbounds bits")
        .index() as u32
    } else {
        dag.apply_operation_back(
            StandardInstruction::Reset.into(),
            &[Qubit(qubit)],
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .expect("caller is responsible for passing inbounds bits")
        .index() as u32
    }
}

/// @ingroup QkDag
/// Apply a barrier to the DAG.
///
/// @param dag The circuit to apply to.
/// @param qubits The qubit indices to apply the barrier to.  This can be null, in which case
///     `num_qubits` is not read, and the barrier is applied to all qubits in the DAG.
/// @param num_qubits How many qubits the barrier applies to.
/// @param front Whether to apply the barrier at the start of the circuit. Usually `false`.
///
/// @return The node index of the created instruction.
///
/// # Examples
///
/// Apply a final barrier on all qubits:
///
/// ```c
/// qk_dag_apply_barrier(dag, NULL, qk_dag_num_qubits(dag), false);
/// ```
///
/// Apply a barrier at the beginning of a circuit on specified qubit indices:
///
/// ```c
/// uint32_t qubits[] = {0, 2, 4, 5};
/// uint32_t num_qubits = sizeof(qubits) / sizeof(qubits[0]);
/// qk_dag_apply_barrier(dag, qubits, num_qubits, true);
/// ```
///
/// # Safety
///
/// Behavior is undefined if:
///
/// * `dag` is not an aligned, non-null pointer to a valid ``QkDag``,
/// * `qubits` is not aligned or is not valid for `num_qubits` reads of initialized, in-bounds and
///   unduplicated indices, unless `qubits` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_apply_barrier(
    dag: *mut DAGCircuit,
    qubits: *const u32,
    num_qubits: u32,
    front: bool,
) -> u32 {
    // SAFETY: per documentation, `dag` points to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    let all_qubits;
    let qubits = if qubits.is_null() {
        all_qubits = (0..dag.num_qubits()).map(Qubit::new).collect::<Vec<_>>();
        all_qubits.as_slice()
    } else {
        // SAFETY: per documentation, `qubits` is valid, aligned, and points to `num_qubits` valid
        // initialized u32s.
        unsafe { std::slice::from_raw_parts(qubits.cast(), num_qubits as usize) }
    };
    let barrier = StandardInstruction::Barrier(qubits.len() as u32);
    if front {
        dag.apply_operation_front(
            barrier.into(),
            qubits,
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .expect("caller is responsible for passing inbounds bits")
        .index() as u32
    } else {
        dag.apply_operation_back(
            barrier.into(),
            qubits,
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .expect("caller is responsible for passing inbounds bits")
        .index() as u32
    }
}

/// @ingroup QkDag
/// Apply a unitary gate to a DAG.
///
/// The values in `matrix` should form a row-major unitary matrix of the correct size for the number
/// of qubits.  The data is copied out of the pointer, and only needs to be valid for reads until
/// this function returns.
///
/// See @verbatim embed:rst:inline ::ref:`circuit-conventions` @endverbatim for detail on the
/// bit-labelling and matrix conventions of Qiskit.
///
/// @param dag The circuit to apply to.
/// @param matrix An initialized row-major unitary matrix of total size ``4**num_qubits``.
/// @param qubits An array of distinct ``uint32_t`` indices of the qubits.
/// @param num_qubits The number of qubits the gate applies to.
/// @param front Whether to apply the gate at the start of the circuit. Usually `false`.
///
/// @return The node index of the created instruction.
///
/// # Safety
///
/// Behavior is undefined if any of:
/// * `dag` is not an aligned, non-null pointer to a valid ``QkDag``,
/// * `matrix` is not an aligned pointer to `4**num_qubits` initialized values,
/// * `qubits` is not an aligned pointer to `num_qubits` initialized values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_apply_unitary(
    dag: *mut DAGCircuit,
    matrix: *const Complex64,
    qubits: *const u32,
    num_qubits: u32,
    front: bool,
) -> u32 {
    // SAFETY: per documentation, `dag` points to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    // SAFETY: per documentation, `matrix` is aligned and valid for `4**num_qubits` reads of
    // initialised data.
    let array = unsafe { unitary_from_pointer(matrix, num_qubits, None) }
        .expect("infallible without tolerance checking");
    let qubits = if num_qubits == 0 {
        // This handles the case of C passing us a null pointer for a scalar matrix; Rust slices
        // can't be backed by the null pointer.
        &[]
    } else {
        // SAFETY: per documentation, `qubits` is aligned and valid for `num_qubits` reads.  Per
        // previous check, `num_qubits` is nonzero so `qubits` cannot be null.
        unsafe { ::std::slice::from_raw_parts(qubits as *const Qubit, num_qubits as usize) }
    };
    if front {
        dag.apply_operation_front(
            Box::new(UnitaryGate { array }).into(),
            qubits,
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .expect("caller is responsible for passing inbounds bits")
        .index() as u32
    } else {
        dag.apply_operation_back(
            Box::new(UnitaryGate { array }).into(),
            qubits,
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .expect("caller is responsible for passing inbounds bits")
        .index() as u32
    }
}

/// @ingroup QkDag
/// Retrieve the standard gate of the specified node.
///
/// Panics if the node is not a standard gate operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the standard gate of.
/// @param out_params A buffer to be filled with the gate's params or NULL
///     if they're not wanted.
///
/// @return The gate value.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
///
/// uint32_t qubit[1] = {0};
/// uint32_t h_gate_idx = qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
///
/// QkGate gate = qk_dag_op_node_gate_op(dag, h_gate_idx, NULL);
///
/// qk_dag_free(dag);
/// qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a
/// ``QkDag``.
/// If ``out_params`` is non-NULL, it must point to a buffer large enough to
/// hold all the gate's params, otherwise behavior is undefined as this
/// function will write beyond its bounds.
/// You can check ``qk_dag_op_node_num_params`` to determine how many params
/// are required for any given operation node.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_op_node_gate_op(
    dag: *const DAGCircuit,
    node: u32,
    out_params: *mut f64,
) -> StandardGate {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag[NodeIndex::new(node as usize)].unwrap_operation();
    if !out_params.is_null() {
        let params = instr.params_view();
        for (i, param) in params
            .iter()
            .map(|x| match x {
                Param::Float(val) => *val,
                _ => panic!("Invalid parameter on instruction"),
            })
            .enumerate()
        {
            // SAFETY: Per documentation, the buffer is large enough to hold all params.
            unsafe {
                out_params.add(i).write(param);
            }
        }
    }
    instr.op.standard_gate()
}

/// @ingroup QkDag
/// Copy out the unitary matrix of the corresponding node index.
///
/// Panics if the node is not a unitary gate.
///
/// @param dag The circuit to read from.
/// @param node The node index of the unitary matrix instruction.
/// @param out Allocated and aligned memory for `4**num_qubits` complex values in row-major order,
///     where `num_qubits` is the number of qubits the gate applies to.
///
/// # Safety
///
/// Behavior is undefined if `dag` is not a non-null pointer to a valid `QkDag`, if `out` is
/// unaligned, or if `out` is not valid for `4**num_qubits` writes of `QkComplex64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_op_node_unitary(
    dag: *const DAGCircuit,
    node: u32,
    out: *mut Complex64,
) {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = &dag[NodeIndex::new(node as usize)].unwrap_operation();
    let OperationRef::Unitary(unitary) = instr.op.view() else {
        panic!("requested node {node} was not a unitary gate");
    };
    match &unitary.array {
        ArrayType::OneQ(array) => {
            let dim = 2;
            for row in 0..dim {
                for col in 0..dim {
                    // SAFETY: per documentation, `out` is aligned and valid for 4 writes.
                    unsafe { out.add(dim * row + col).write(array[(row, col)]) };
                }
            }
        }
        ArrayType::TwoQ(array) => {
            let dim = 4;
            for row in 0..dim {
                for col in 0..dim {
                    // SAFETY: per documentation, `out` is aligned and valid for 16 writes.
                    unsafe { out.add(dim * row + col).write(array[(row, col)]) };
                }
            }
        }
        ArrayType::NDArray(array) => {
            for (i, val) in array.iter().enumerate() {
                // SAFETY: per documentation, `out` is aligned and valid for `array.size()` writes.
                unsafe { out.add(i).write(*val) };
            }
        }
    }
}

/// The operation's kind.
///
/// This is returned when querying a particular node in the graph with ``qk_dag_op_node_kind``,
/// and is intended to allow the caller to dispatch (e.g. via a "switch") calls specific to
/// the contained operation's kind.
#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum COperationKind {
    Gate = 0,
    Barrier = 1,
    Delay = 2,
    Measure = 3,
    Reset = 4,
    Unitary = 5,
    PauliProductMeasurement = 6,
    ControlFlow = 7,
    /// This variant is used as an opaque type for operations not yet
    /// implemented in the native data model.
    Unknown = 8,
    PauliProductRotation = 9,
}

/// @ingroup QkDag
/// Get the "kind" of an operation node.
///
/// The result can be used in a switch statement to dispatch proper handling
/// when iterating over operation nodes.
///
/// Panics if ``node`` is not an operation node.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the "kind" of.
///
/// @return The "kind" of the node.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_op_node_kind(dag: *const DAGCircuit, node: u32) -> COperationKind {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    match dag[NodeIndex::new(node as usize)]
        .unwrap_operation()
        .op
        .view()
    {
        OperationRef::StandardGate(_) => COperationKind::Gate,
        OperationRef::StandardInstruction(instr) => match instr {
            StandardInstruction::Barrier(_) => COperationKind::Barrier,
            StandardInstruction::Delay(_) => COperationKind::Delay,
            StandardInstruction::Measure => COperationKind::Measure,
            StandardInstruction::Reset => COperationKind::Reset,
        },
        OperationRef::Unitary(_) => COperationKind::Unitary,
        OperationRef::PauliProductMeasurement(_) => COperationKind::PauliProductMeasurement,
        OperationRef::PauliProductRotation(_) => COperationKind::PauliProductRotation,
        OperationRef::ControlFlow(_) => COperationKind::ControlFlow,
        OperationRef::PyCustom(_) | OperationRef::CustomOperation(_) => COperationKind::Unknown,
    }
}

/// A struct for storing successors and predecessors information
/// retrieved from `qk_dag_successors` and `qk_dag_predecessors`, respectively.
///
/// This object is read-only from C. To satisfy the safety guarantees of `qk_dag_neighbors_clear`,
/// you must not overwrite any data initialized by `qk_dag_successors` or `qk_dag_predecessors`,
/// including any pointed-to data.
#[repr(C)]
pub struct CDagNeighbors {
    /// Array of size `num_neighbors` of node indices.
    pub neighbors: *const u32,
    /// The length of the `neighbors` array.
    pub num_neighbors: usize,
}

/// @ingroup QkDag
/// Retrieve the successors of the specified node.
///
/// The successors array and its length are returned as a `QkDagNeighbors` struct, where each element in the
/// array corresponds to a DAG node index.
/// You must call the `qk_dag_neighbors_clear` function when done to free the memory allocated for the struct.
///
/// @param dag A pointer to the DAG.
/// @param node The node to get the successors of.
///
/// @return An instance of the `QkDagNeighbors` struct with the successors information.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
/// qk_dag_add_quantum_register(dag, qr);
/// qk_quantum_register_free(qr);
///
/// uint32_t node_cx = qk_dag_apply_gate(dag, QkGate_CX, (uint32_t[]){0, 1}, NULL, false);
///
/// QkDagNeighbors successors = qk_dag_successors(dag, node_cx);
///
/// qk_dag_neighbors_clear(&successors);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_successors(dag: *const DAGCircuit, node: u32) -> CDagNeighbors {
    // SAFETY: Per documentation, the pointers are to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };

    let successors: Box<[u32]> = dag
        .successors(NodeIndex::new(node as usize))
        .map(|node| node.index() as u32)
        .collect();

    CDagNeighbors {
        num_neighbors: successors.len(),
        neighbors: Box::into_raw(successors) as *const u32,
    }
}

/// @ingroup QkDag
/// Retrieve the predecessors of the specified node.
///
/// The predecessors array and its length are returned as a `QkDagNeighbors` struct, where each element in the
/// array corresponds to a DAG node index.
/// You must call the `qk_dag_neighbors_clear` function when done to free the memory allocated for the struct.
///
/// @param dag A pointer to the DAG.
/// @param node The node to get the predecessors of.
///
/// @return An instance of the `QkDagNeighbors` struct with the predecessors information.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
/// qk_dag_add_quantum_register(dag, qr);
/// qk_quantum_register_free(qr);
///
/// uint32_t node_cx = qk_dag_apply_gate(dag, QkGate_CX, (uint32_t[]){0, 1}, NULL, false);
///
/// QkDagNeighbors predecessors = qk_dag_predecessors(dag, node_cx);
///
/// qk_dag_neighbors_clear(&predecessors);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_predecessors(dag: *const DAGCircuit, node: u32) -> CDagNeighbors {
    // SAFETY: Per documentation, the pointers are to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };

    let predecessors: Box<[u32]> = dag
        .predecessors(NodeIndex::new(node as usize))
        .map(|node| node.index() as u32)
        .collect();
    CDagNeighbors {
        num_neighbors: predecessors.len(),
        neighbors: Box::into_raw(predecessors) as *const u32,
    }
}

/// @ingroup QkDag
/// Clear the fields of the input `QkDagNeighbors` struct.
///
/// The function deallocates the memory pointed to by the `neighbors` field and sets it to NULL.
/// It also sets the `num_neighbors` field to 0.
///
/// @param neighbors A pointer to a `QkDagNeighbors` object.
///
/// # Safety
///
/// Behavior is undefined if ``neighbors`` is not a valid, non-null pointer to a QkDagNeighbors
/// object populated with either ``qk_dag_successors`` or ``qk_dag_predecessors``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_neighbors_clear(neighbors: *mut CDagNeighbors) {
    // SAFETY: Per documentation, the pointer is to a valid data.
    let neighbors = unsafe { mut_ptr_as_ref(neighbors) };

    if neighbors.num_neighbors > 0 {
        let slice = std::ptr::slice_from_raw_parts_mut(
            neighbors.neighbors as *mut u32,
            neighbors.num_neighbors,
        );
        unsafe {
            let _ = Box::from_raw(slice);
        }
    }

    neighbors.num_neighbors = 0;
    neighbors.neighbors = std::ptr::null();
}

/// @ingroup QkDag
/// Return the details for an instruction in the circuit.
///
/// This is a mirror of `qk_circuit_get_instruction`.  You can also use individual methods such as
/// `qk_dag_op_node_gate_op` to get individual properties.
///
/// You must call `qk_circuit_instruction_clear` to reset the `QkCircuitInstruction` before reusing
/// it or dropping it.
///
/// @param dag The circuit to retrieve the instruction from.
/// @param index The node index.  It is an error to pass an index that is node a valid op node.
/// @param instruction A point to where to write out the `QkCircuitInstruction`.
///
/// # Examples
///
/// Iterate through a DAG to find which qubits have measures on them:
///
/// ```c
/// bool *measured = calloc(qk_dag_num_qubits(dag), sizeof(*measured));
/// uint32_t num_ops = qk_dag_num_op_nodes(dag);
/// uint32_t *ops = malloc(num_ops * sizeof(*ops));
/// qk_dag_topological_op_nodes(dag, ops);
///
/// // Storage space for the instruction.
/// QkCircuitInstruction inst;
/// for (uint32_t i = 0; i < num_ops; i++) {
///     qk_dag_get_instruction(dag, ops[i], &inst);
///     if (!strcmp(inst.name, "measure"))
///         measured[inst.qubits[0]] = true;
///     qk_circuit_instruction_clear(&inst);
/// }
///
/// free(ops);
/// free(measured);
/// ```
///
/// # Safety
///
/// Behavior is undefined if either `dag` or `instruction` are not valid, aligned, non-null pointers
/// to the relevant data type.  The fields of `instruction` need not be initialized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_get_instruction(
    dag: *const DAGCircuit,
    index: u32,
    instruction: *mut CInstruction,
) {
    // SAFETY: per documentation, `dag` is a pointer to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let inst = CInstruction::from_packed_instruction_with_numeric(
        dag.dag()[NodeIndex::new(index as usize)].unwrap_operation(),
        dag.qargs_interner(),
        dag.cargs_interner(),
    );
    // SAFETY: per documentation, `instruction` is a pointer to a sufficient allocation.
    unsafe { instruction.write(inst) };
}

/// @ingroup QkDag
/// Compose the ``other`` DAG onto the ``dag`` instance with the option of a subset
/// of input wires of ``other`` being mapped onto a subset of output wires of ``dag``.
///
/// ``other`` may include a smaller or equal number of wires for each type.
///
/// @param dag A pointer to the DAG to be composed on.
/// @param other A pointer to the DAG to compose with ``dag``.
/// @param qubits A list of indices representing the qubit wires to compose
///     onto.
/// @param clbits A list of indices representing the clbit wires to compose
///     onto.
///
/// @return ``QkExitCode_Success`` upon successful decomposition, otherwise a DAG-specific
///     error code indicating the cause of the failure.
///
/// # Example
///
/// ```c
/// // Build the following dag
/// // rqr_0: ──■───────
/// //          │  ┌───┐
/// // rqr_1: ──┼──┤ Y ├
/// //        ┌─┴─┐└───┘
/// // rqr_2: ┤ X ├─────
/// //        └───┘
/// QkDag *dag_right = qk_dag_new();
/// QkQuantumRegister *rqr = qk_quantum_register_new(3, "rqr");
/// qk_dag_add_quantum_register(dag_right, rqr);
/// qk_dag_add_classical_register(dag_right, rcr);
/// qk_dag_apply_gate(dag_right, QkGate_CX, (uint32_t[]){0, 2}, NULL, false);
/// qk_dag_apply_gate(dag_right, QkGate_Y, (uint32_t[]){1}, NULL, false);
///
/// // Build the following dag
/// //          ┌───┐
/// // lqr_0: ──┤ H ├───
/// //        ┌─┴───┴──┐
/// // lqr_1: ┤ P(0.1) ├
/// //        └────────┘
/// QkDag *dag_left = qk_dag_new();
/// QkQuantumRegister *lqr = qk_quantum_register_new(2, "lqr");
/// qk_dag_add_quantum_register(dag_left, lqr);
/// qk_dag_add_classical_register(dag_left, lcr);
/// qk_dag_apply_gate(dag_left, QkGate_H, (uint32_t[]){0}, NULL, false);
/// qk_dag_apply_gate(dag_left, QkGate_Phase, (uint32_t[]){1}, (double[]){0.1}, false);
///
/// // Compose left circuit onto right circuit
/// // Should result in circuit
/// //             ┌───┐
/// // rqr_0: ──■──┤ H ├──────────
/// //          │  ├───┤┌────────┐
/// // rqr_1: ──┼──┤ Y ├┤ P(0.1) ├
/// //        ┌─┴─┐└───┘└────────┘
/// // rqr_2: ┤ X ├───────────────
/// //        └───┘
/// qk_dag_compose(dag_right, dag_left, NULL, NULL);
///
/// // Clean up after you're done
/// qk_dag_free(dag_left);
/// qk_dag_free(dag_right);
/// qk_quantum_register_free(lqr);
/// qk_quantum_register_free(rqr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` or ``other`` are not valid, non-null pointers to a ``QkDag``.
/// If ``qubit`` nor ``clbit`` are NULL, it must contains a less or equal amount
/// than what the circuit owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_compose(
    dag: *mut DAGCircuit,
    other: *const DAGCircuit,
    qubits: *const u32,
    clbits: *const u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    // SAFETY: Per documentation, the pointer is to valid data.
    let other_dag = unsafe { const_ptr_as_ref(other) };

    if other_dag.qubits().len() > dag.qubits().len()
        || other_dag.clbits().len() > dag.clbits().len()
    {
        return ExitCode::DagComposeMismatch;
    }

    let block_map = other_dag
        .blocks()
        .items()
        .map(|(index, block)| (index, dag.add_block(block.clone())))
        .collect();

    let local_qubits: Option<Vec<Qubit>> = if check_ptr(qubits).is_ok() {
        let qubit_slice = unsafe { std::slice::from_raw_parts(qubits, other_dag.num_qubits()) };
        Some(qubit_slice.iter().map(|&bit| Qubit(bit)).collect())
    } else {
        None
    };

    let local_clbits: Option<Vec<Clbit>> = if check_ptr(clbits).is_ok() {
        let clbit_slice = unsafe { std::slice::from_raw_parts(clbits, other_dag.num_clbits()) };
        Some(clbit_slice.iter().map(|&bit| Clbit(bit)).collect())
    } else {
        None
    };

    // Since we don't yet support vars in C, we can skip the inline_captures check.
    dag.compose(
        other_dag,
        local_qubits.as_deref(),
        local_clbits.as_deref(),
        block_map,
        false,
    )
    .expect("Error during circuit composition.");
    ExitCode::Success
}

/// @ingroup QkDag
/// Free the DAG.
///
/// @param dag A pointer to the DAG to free.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not either null or a valid pointer to a
/// ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_free(dag: *mut DAGCircuit) {
    if !dag.is_null() {
        if !dag.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(dag);
        }
    }
}

/// @ingroup QkDag
/// Convert a given DAG to a circuit.
///
/// The new circuit is copied from the DAG; the original ``dag`` reference is still owned by the
/// caller and still required to be freed with `qk_dag_free`.  You must free the returned circuit
/// with ``qk_circuit_free`` when done with it.
///
/// @param dag A pointer to the DAG from which to create the circuit.
///
/// @return A pointer to the new circuit.
///
/// # Example
///
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
/// qk_dag_add_quantum_register(dag, qr);
/// qk_quantum_register_free(qr);
///
/// QkCircuit *qc = qk_dag_to_circuit(dag);
///
/// qk_circuit_free(qc);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_to_circuit(dag: *const DAGCircuit) -> *mut CircuitData {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let circuit = CircuitData::from_dag_ref(dag)
        .expect("Error occurred while converting DAGCircuit to CircuitData");

    Box::into_raw(Box::new(circuit))
}

/// @ingroup QkDag
/// Return the operation nodes in the DAG listed in topological order.
///
/// @param dag A pointer to the DAG.
/// @param out_order A pointer to an array of ``qk_dag_num_op_nodes(dag)`` elements
/// of type ``uint32_t``, where this function will write the output to.
///
/// # Example
///
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
///
/// uint32_t qubit[1] = {0};
/// qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
/// qk_dag_apply_gate(dag, QkGate_S, qubit, NULL, false);
///
/// // get the number of operation nodes
/// uint32_t num_ops = qk_dag_num_op_nodes(dag); // 2
/// uint32_t *out_order = malloc(sizeof(uint32_t) * num_ops);
///
/// // get operation nodes listed in topological order
/// qk_dag_topological_op_nodes(dag, out_order);
///
/// // do something with the ordered nodes
/// for (uint32_t i = 0; i < num_ops; i++) {
///     QkGate gate = qk_dag_op_node_gate_op(dag, out_order[i], NULL);
///     printf("The gate at location %u is %u.\n", i, gate);
/// }
///
/// // free the out_order array, register, and dag pointer when done
/// free(out_order);
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``
/// or if ``out_order`` is not a valid, non-null pointer to a sequence of ``qk_dag_num_op_nodes(dag)``
/// consecutive elements of ``uint32_t``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_topological_op_nodes(dag: *const DAGCircuit, out_order: *mut u32) {
    // SAFETY: Per documentation, ``dag`` is non-null and valid.
    let dag = unsafe { const_ptr_as_ref(dag) };

    let out_topological_op_nodes = dag.topological_op_nodes(false);

    for (i, node) in out_topological_op_nodes.enumerate() {
        // SAFETY: per documentation, `out_order` is aligned and points to a valid
        // aligned and maybe uninitialized block of memory valid for `num_op_nodes`
        // writes of `u32`s.
        unsafe { out_order.add(i).write(node.index() as u32) }
    }
}

/// @ingroup QkDag
/// Replace a node in a `QkDag` with a subcircuit specified by another `QkDag`
///
/// @param dag A pointer to the DAG.
/// @param node The node index of the operation to replace with the other `QkDag`. This
///     must be the node index for an operation node in ``dag`` and the qargs and cargs
///     count must match the number of qubits and clbits in `replacement`.
/// @param replacement The other `QkDag` to replace `node` with. This dag must have
///     the same number of qubits as the operation for ``node``. The node
///     bit ordering will be ordering will be handled in order, so `qargs[0]` for
///     `node` will be mapped to `qubits[0]` in `replacement`, `qargs[1]` to
///     `qubits[0]`, etc. The same pattern applies to classical bits too.
///
/// # Example
///
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
///
/// uint32_t qubit[1] = {0};
/// uint32_t node_to_replace = qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
/// qk_dag_apply_gate(dag, QkGate_S, qubit, NULL, false);
///
/// // Build replacement dag for H
/// QkDag *replacement = qk_dag_new();
/// QkQuantumRegister *replacement_qr = qk_quantum_register_new(1, "other");
/// qk_dag_add_quantum_register(replacement, replacement_qr);
/// double pi_param[1] = {3.14159};
/// qk_dag_apply_gate(replacement, QkGate_RZ, qubit, pi_param, false);
/// qk_dag_apply_gate(replacement, QkGate_SX, qubit, NULL, false);
/// qk_dag_apply_gate(replacement, QkGate_RZ, qubit, pi_param, false);
///
/// qk_dag_substitute_node_with_dag(dag, node_to_replace, replacement);
///
/// // Free the replacement dag, register, dag, and register
/// qk_quantum_register_free(replacement_qr);
/// qk_dag_free(replacement);
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` and ``replacement`` are not a valid, non-null pointer to a
/// ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_substitute_node_with_dag(
    dag: *mut DAGCircuit,
    node: u32,
    replacement: *const DAGCircuit,
) {
    // SAFETY: Per documentation, ``dag`` is non-null and valid.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    // SAFETY: Per documentation, ``replacement`` is non-null and valid.
    let replacement = unsafe { const_ptr_as_ref(replacement) };

    if let Err(e) = dag.substitute_node_with_dag(
        NodeIndex::new(node as usize),
        replacement,
        None,
        None,
        None,
        None,
    ) {
        let err: Error = e.into();
        panic!("Node substitution failed with: {}", err.backtrace());
    }
}

/// @ingroup QkDag
/// Return a copy of self with the same structure but empty.
///
/// That structure includes:
/// * name and other metadata
/// * global phase
/// * duration
/// * all the qubits and clbits, including the registers.
///
/// @param dag A pointer to the DAG to copy.
/// @param vars_mode The mode for handling classical variables.
/// @param blocks_mode The mode for handling blocks.
///
/// @return The pointer to the copied DAG circuit.
///
/// # Example
///
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
///
/// uint32_t qubit[1] = {0};
/// qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
///
/// // As the DAG does not contain any control-flow instructions,
/// // vars_mode and blocks_mode do not have any effect.
/// QkDag *copied_dag = qk_dag_copy_empty_like(dag, QkVarsMode_Alike, QkBlocksMode_Drop);
/// uint32_t num_ops_in_copied_dag = qk_dag_num_op_nodes(copied_dag); // 0
///
/// // do something with copied_dag
///
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// qk_dag_free(copied_dag);
/// ```
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid pointer to a ``QkDag``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_copy_empty_like(
    dag: *const DAGCircuit,
    vars_mode: CVarsMode,
    blocks_mode: CBlocksMode,
) -> *mut DAGCircuit {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let vars_mode = vars_mode.into();
    let blocks_mode = blocks_mode.into();

    let copied_dag = dag.copy_empty_like_with_capacity(0, 0, vars_mode, blocks_mode);
    Box::into_raw(Box::new(copied_dag))
}

/// @ingroup QkDag
/// Replace a non-empty contiguous block of nodes in a ``QkDag`` with a
/// single unitary gate corresponding to the specified unitary matrix.
///
/// Upon replacement, the nodes in the block are removed and substituted by
/// a new node acting on the given qubits.
///
/// @param dag Pointer to the DAG.
/// @param num_block_ids Number of entries in ``block_ids``. This number must
///     be nonzero.
/// @param block_ids Pointer to a non-empty array of nodes to replace.
/// @param matrix Pointer to an initialized row-major unitary matrix of size
///     ``4**num_qubits``.
/// @param num_qubits The number of qubits the resulting unitary gate acts on.
/// @param qubits Pointer to an array of distinct ``uint32_t`` qubit indices.
///     Each entry specifies the index of the DAG qubit that corresponds to
///     the respective argument position in the unitary gate.
/// @param cycle_check If ``true``, the function checks whether replacing the
///     provided ``block_ids`` with a single node would introduce a cycle in
///     the DAG (which would invalidate the DAG). If a cycle would be
///     created, the DAG is left unchanged and ``UINT32_MAX`` is returned.
///     This checking comes with a run time penalty. If you can guarantee that
///     the provided ``block_ids`` is a contiguous block and won't introduce a
///     cycle when contracted to a single node, this can be set to ``false``.
///
/// @return The index of the newly added operation node, or ``UINT32_MAX`` if
///     ``cycle_check`` is ``true`` and the replacement would introduce a cycle.
///
/// # Example
///
/// ```c
/// // Create a DAG with H, T, S, T, H gates on the second qubit
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
/// qk_dag_add_quantum_register(dag, qr);
/// uint32_t qubit[1] = {1};
/// qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
/// uint32_t idx1 = qk_dag_apply_gate(dag, QkGate_T, qubit, NULL, false);
/// uint32_t idx2 = qk_dag_apply_gate(dag, QkGate_S, qubit, NULL, false);
/// uint32_t idx3 = qk_dag_apply_gate(dag, QkGate_T, qubit, NULL, false);
/// qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
///
/// // Replace the inner T, S, T gates by a unitary gate (representing Z)
/// uint32_t replaced_ids[3] = {idx1, idx2, idx3};
/// static const QkComplex64 mat_z[4] = {{1, 0}, {0, 0}, {0, 0}, {-1, 0}};
/// uint32_t new_node_idx =
///     qk_dag_replace_block_with_unitary(dag, 3, replaced_ids, mat_z, 1, qubit, false);
///
/// // free the register and dag pointer when done
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if any of:
/// * `dag` is not an aligned, non-null pointer to a valid ``QkDag``,
/// * `qubits` is not an aligned pointer to `num_qubits` initialized values.
/// * `matrix` is not an aligned pointer to `4**num_qubits` initialized values,
/// * `block_ids` is not an aligned pointer to `num_block_ids` initialized values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_replace_block_with_unitary(
    dag: *mut DAGCircuit,
    num_block_ids: u32,
    block_ids: *const u32,
    matrix: *const Complex64,
    num_qubits: u32,
    qubits: *const u32,
    cycle_check: bool,
) -> u32 {
    // SAFETY: per documentation, `dag` points to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };

    // SAFETY: per documentation, `matrix` is aligned and valid for `4**num_qubits` reads of
    // initialized data.
    let array = unsafe { unitary_from_pointer(matrix, num_qubits, None) }
        .expect("infallible without tolerance checking");

    // SAFETY: per documentation, `block_ids` is aligned and valid for `num_block_ids` reads. Per
    // documentation, `num_block_ids` is nonzero so `block_ids` cannot be null.
    // In addition, it is unlikely that petgraph's `NodeIndex` will ever change from `u32`, but
    // to be fully on the safe side the rust test `verify_default_ix_type` will prevent Qiskit
    // from compiling if this ever happens.
    let block = unsafe {
        ::std::slice::from_raw_parts(block_ids as *const NodeIndex, num_block_ids as usize)
    };

    let qubits = if num_qubits == 0 {
        // This handles the case of C passing us a null pointer for a scalar matrix; Rust slices
        // can't be backed by the null pointer.
        &[]
    } else {
        // SAFETY: per documentation, `qubits` is aligned and valid for `num_qubits` reads.  Per
        // previous check, `num_qubits` is nonzero so `qubits` cannot be null.
        unsafe { ::std::slice::from_raw_parts(qubits as *const Qubit, num_qubits as usize) }
    };

    let qubit_pos_map = qubits
        .iter()
        .enumerate()
        .map(|(i, q)| (*q, i))
        .collect::<HashMap<_, _>>();
    let clbit_pos_map = HashMap::new();

    let res = dag.replace_block(
        block,
        Box::new(UnitaryGate { array }).into(),
        None,
        None,
        cycle_check,
        &qubit_pos_map,
        &clbit_pos_map,
    );

    match res {
        Ok(new_index) => new_index.index() as u32,
        Err(_) => u32::MAX,
    }
}

/// @ingroup QkDag
/// Substitute an operation in a node in a ``QkDag`` with a unitary gate
/// corresponding to the specified unitary matrix.
///
/// The new operation should match the shape of the replaced operation.
/// The qargs and cargs for the node will remain the same.
///
/// @param dag Pointer to the DAG.
/// @param node The node whose operation is substituted. The number of qubits
///     in the substituted operation should equal to ``num_qubits`` and the
///     number of clbits should be ``0``.
/// @param matrix Pointer to an initialized row-major unitary matrix of size
///     ``4**num_qubits``.
/// @param num_qubits The number of qubits the unitary acts on.
///
/// # Example
/// ```c
/// // Create a DAG with a Z-gate
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
/// qk_dag_add_quantum_register(dag, qr);
/// uint32_t idx_z = qk_dag_apply_gate(dag, QkGate_Z, (uint32_t[]){1}, NULL, false);
///
/// static const QkComplex64 mat[4] = {{1, 0}, {0, 0}, {0, 0}, {-1, 0}};
///
/// // Replace the Z-gate by a unitary matrix
/// qk_dag_substitute_node_with_unitary(dag, idx_z, mat, 1);
///
/// // free the register and dag pointer when done
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if any of:
/// * `dag` is not an aligned, non-null pointer to a valid ``QkDag``,
/// * `matrix` is not an aligned pointer to `4**num_qubits` initialized values,
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_dag_substitute_node_with_unitary(
    dag: *mut DAGCircuit,
    node: u32,
    matrix: *const Complex64,
    num_qubits: u32,
) {
    // SAFETY: per documentation, `dag` points to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };

    // SAFETY: per documentation, `matrix` is aligned and valid for `4**num_qubits` reads of
    // initialized data.
    let array = unsafe { unitary_from_pointer(matrix, num_qubits, None) }
        .expect("infallible without tolerance checking");

    dag.substitute_op(
        NodeIndex::new(node as usize),
        UnitaryGate { array }.into(),
        None,
        None,
    )
    .expect("Failed to substitute op.")
}

/// @ingroup QkDag
/// Pass ownership of a `QkDag` object to Python.
///
/// It is not safe to use the `QkDag` pointer after calling this function.  In particular, you
/// should not attempt to clear or free it.  The caller must own the `QkDag`, not hold a borrowed
/// reference (for example, a `QkDag *` retrieved from `qk_dag_borrow_from_python` is not owned).
///
/// @param dag The owned object.
/// @return An owned Python reference to the object.
///
/// # Safety
///
/// The caller must be attached to a Python interpreter.  Behavior is undefined if `dag` is not a
/// valid non-null pointer to an initialized and owned `QkDag`.
#[unsafe(no_mangle)]
#[cfg(feature = "python_binding")]
pub unsafe extern "C" fn qk_dag_to_python(dag: *mut DAGCircuit) -> *mut ::pyo3::ffi::PyObject {
    // SAFETY: per documentation, we are attached to a Python interpreter.
    let py = unsafe { ::pyo3::Python::assume_attached() };
    // SAFETY: per documentation, `dag` points to owned and valid data.
    let dag = unsafe { Box::from_raw(dag) };
    match ::pyo3::Bound::new(py, *dag) {
        Ok(ob) => ob.into_ptr(),
        Err(e) => {
            e.restore(py);
            ::std::ptr::null_mut()
        }
    }
}

/// @ingroup QkDag
/// Retrieve a `QkDag` pointer from a Python object.
///
/// This borrows a Python reference and extracts the `QkDag` pointer for it, if it is of
/// the correct type.  The returned pointer is borrowed from the `ob` pointer.  If the
/// ``PyObject`` is not the correct type, the return value is ``NULL`` and the exception
/// state of the Python interpreter is set.
///
/// You must be attached to a Python interpreter to call this function.
///
/// You can also use `qk_dag_convert_from_python`, which is logically the exact same as this
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
pub unsafe extern "C" fn qk_dag_borrow_from_python(
    ob: *mut pyo3::ffi::PyObject,
) -> *mut DAGCircuit {
    // SAFETY: per documentation, we are attached to a Python interpreter, and `ob` points to a
    // valid PyObject.
    unsafe { crate::py::borrow_mut(::pyo3::Python::assume_attached(), ob) }
}

/// @ingroup QkDag
/// Retrieve a DAG pointer from a Python object.
///
/// This borrows a Python reference and extracts the `QkDag` pointer for it into ``address``, if it
/// is of the correct type.  The returned pointer is borrowed from the `object` pointer.  If the
/// ``PyObject`` is not the correct type, the return value is 1, the exception state of the Python
/// interpreter is set, and ``address`` is unchanged.
///
/// You must be attached to a Python interpreter to call this function.
///
/// You can also use `qk_dag_borrow_from_python`, which is logically the exact same as this, but
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
pub unsafe extern "C" fn qk_dag_convert_from_python(
    object: *mut ::pyo3::ffi::PyObject,
    address: *mut ::std::ffi::c_void,
) -> ::std::ffi::c_int {
    // SAFETY: per documentation, we are attached to a Python interpreter, `object` is a valid
    // pointer to a PyObject, and `address` points to enough space to write a pointer.
    unsafe {
        crate::py::convert_mut::<DAGCircuit>(::pyo3::Python::assume_attached(), object, address)
    }
}
