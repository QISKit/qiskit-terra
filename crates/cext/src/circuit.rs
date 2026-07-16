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
use std::ptr;

use crate::circuit_library::pbc::{CPauliProductMeasurement, CPauliProductRotation};
use crate::control_flow::CControlFlowInstruction;
use crate::dag::COperationKind;
use crate::exit_codes::ExitCode;
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

use nalgebra::{Matrix2, Matrix4};
use ndarray::{Array2, ArrayView2};
use num_complex::{Complex64, ComplexFloat};

use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister};
use qiskit_circuit::bit::{ShareableClbit, ShareableQubit};
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::circuit_drawer::draw_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::interner::Interner;
use qiskit_circuit::operations::{
    ArrayType, DelayUnit, Operation, OperationRef, Param, PauliBased, PauliProductMeasurement,
    PauliProductRotation, StandardGate, StandardInstruction, UnitaryGate,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::parameter_table::ParameterTableError;
use qiskit_circuit::{BlocksMode, Clbit, Qubit, VarsMode};
use qiskit_transpiler::target::{Target, estimate_fidelity};
use smallvec::smallvec;

/// @ingroup QkCircuit
/// Construct a new circuit with the given number of qubits and clbits.
///
/// @param num_qubits The number of qubits the circuit contains.
/// @param num_clbits The number of clbits the circuit contains.
///
/// @return A pointer to the created circuit.
///
/// # Example
///
///     QkCircuit *empty = qk_circuit_new(100, 100);
///
#[unsafe(no_mangle)]
pub extern "C" fn qk_circuit_new(num_qubits: u32, num_clbits: u32) -> *mut CircuitData {
    let qubits = if num_qubits > 0 {
        Some(
            (0..num_qubits)
                .map(|_| ShareableQubit::new_anonymous())
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let clbits = if num_clbits > 0 {
        Some(
            (0..num_clbits)
                .map(|_| ShareableClbit::new_anonymous())
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let circuit = CircuitData::new(qubits, clbits, (0.).into()).unwrap();
    Box::into_raw(Box::new(circuit))
}

/// @ingroup QkQuantumRegister
/// Construct a new owning quantum register with a given number of qubits and name
///
/// @param num_qubits The number of qubits to create the register for
/// @param name The name string for the created register. The name must be comprised of
/// valid UTF-8 characters.
///
/// @return A pointer to the created register
///
/// # Example
/// ```c
///     QkQuantumRegister *qr = qk_quantum_register_new(5, "five_qubits");
/// ```
///
/// # Safety
///
/// The `name` parameter must be a pointer to memory that contains a valid
/// nul terminator at the end of the string. It also must be valid for reads of
/// bytes up to and including the nul terminator.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_quantum_register_new(
    num_qubits: u32,
    name: *const c_char,
) -> *mut QuantumRegister {
    let name = unsafe {
        CStr::from_ptr(name)
            .to_str()
            .expect("Invalid UTF-8 character")
            .to_string()
    };
    // SAFETY: Per documentation the pointer for name is a valid CStr pointer
    let reg = QuantumRegister::new_owning(name, num_qubits);
    Box::into_raw(Box::new(reg))
}

/// @ingroup QkQuantumRegister
/// Free a quantum register.
///
/// @param reg A pointer to the register to free.
///
/// # Example
/// ```c
///     QkQuantumRegister *qr = qk_quantum_register_new(1024, "qreg");
///     qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``reg`` is not either null or a valid pointer to a
/// ``QkQuantumRegister``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_quantum_register_free(reg: *mut QuantumRegister) {
    if !reg.is_null() {
        if !reg.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(reg);
        }
    }
}

/// @ingroup QkQuantumRegister
/// Get the name of a quantum register.
///
/// Returns a newly allocated C string containing the name of the quantum register.
///
/// @param qreg A pointer to the quantum register.
///
/// @return A pointer to a newly allocated null-terminated string containing the register name.
///         The caller must free this string using `qk_str_free`.
///
/// # Example
/// ```c
///     QkQuantumRegister *qr = qk_quantum_register_new(5, "my_qreg");
///     char *name = qk_quantum_register_name(qr);
///     printf("Register name: %s\n", name);
///     qk_str_free(name);
///     qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``qreg`` is not a valid, non-null pointer to a ``QkQuantumRegister``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_quantum_register_name(qreg: *const QuantumRegister) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let qreg = unsafe { const_ptr_as_ref(qreg) };

    CString::new(qreg.name())
        .expect("Register name should not contain null bytes")
        .into_raw()
}

/// @ingroup QkQuantumRegister
/// Get the number of bits in a quantum register.
///
/// Returns the size (number of bits) of the quantum register.
///
/// @param qreg A pointer to the quantum register.
///
/// @return The number of bits in the register.
///
/// # Example
/// ```c
///     QkQuantumRegister *qr = qk_quantum_register_new(5, "my_qreg");
///     size_t num_qubits = qk_quantum_register_num_bits(qr);
///     printf("Number of qubits: %zu\n", num_qubits);  // Prints: 5
///     qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``qreg`` is not a valid, non-null pointer to a ``QkQuantumRegister``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_quantum_register_num_bits(qreg: *const QuantumRegister) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let qreg = unsafe { const_ptr_as_ref(qreg) };

    qreg.len()
}

/// @ingroup QkQuantumRegister
/// Get the circuit bit indices for all qubits in a quantum register.
///
/// Outputs a mapping from each qubit in the quantum register to its corresponding index in the circuit's
/// qubit list. If a qubit from the register is not present in the circuit, its index
/// in the mapping will be ``UINT32_MAX``.
///
/// @param qreg A pointer to the quantum register to map.
/// @param circuit A pointer to the circuit containing the qubits.
/// @param out_bits A pointer to an array where the mapped indices will be written.
///     The array must be pre-allocated with at least `qk_quantum_register_num_bits` elements
///     for ``qreg``. Each element will contain either the circuit bit index or ``UINT32_MAX``
///     if the qubit is not in the circuit.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(2, 0);
///     QkQuantumRegister *qr = qk_quantum_register_new(3, "my_qreg");
///     qk_circuit_add_quantum_register(qc, qr);
///
///     uint32_t bit_indices[3];
///
///     qk_quantum_register_circuit_bits(qr, qc, bit_indices);
///
///     // bit_indices now contains [2, 3, 4] since all qubits are in the circuit
///     // and the circuit has 2 anonymous qubits
///
///     qk_quantum_register_free(qr);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if:
/// - ``qreg`` is not a valid, non-null pointer to a ``QkQuantumRegister``.
/// - ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
/// - ``qreg`` has one or more bits and ``out_bits`` is not an aligned, non-null pointer to an array with at least
///   ``qk_quantum_register_num_bits(qreg)`` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_quantum_register_circuit_bits(
    qreg: *const QuantumRegister,
    circuit: *const CircuitData,
    out_bits: *mut u32,
) {
    // SAFETY: Per documentation, qreg is a valid, non-null pointer.
    let qreg = unsafe { const_ptr_as_ref(qreg) };
    // SAFETY: Per documentation, circuit is a valid, non-null pointer.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    qreg.iter().enumerate().for_each(|(i, qubit)| {
        let mapped_qubit = circuit.qubit_index(&qubit).map_or(u32::MAX, |q| q);
        // SAFETY: Per documentation, out_bits is aligned and has at least qreg.len() elements
        unsafe { out_bits.add(i).write(mapped_qubit) };
    });
}

/// @ingroup QkClassicalRegister
/// Construct a new owning classical register with a given number of clbits and name
///
/// @param num_clbits The number of clbits to create the register for
/// @param name The name string for the created register. The name must be comprised of
/// valid UTF-8 characters.
///
/// @return A pointer to the created register
///
/// # Example
/// ```c
///     QkClassicalRegister *cr = qk_classical_register_new(5, "five_qubits");
/// ```
///
/// # Safety
///
/// The `name` parameter must be a pointer to memory that contains a valid
/// nul terminator at the end of the string. It also must be valid for reads of
/// bytes up to and including the nul terminator.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_classical_register_new(
    num_clbits: u32,
    name: *const c_char,
) -> *mut ClassicalRegister {
    // SAFETY: Per documentation the pointer for name is a valid CStr pointer
    let name = unsafe {
        CStr::from_ptr(name)
            .to_str()
            .expect("Invalid UTF-8 character")
            .to_string()
    };
    let reg = ClassicalRegister::new_owning(name, num_clbits);
    Box::into_raw(Box::new(reg))
}

/// @ingroup QkClassicalRegister
/// Free a classical register.
///
/// @param reg A pointer to the register to free.
///
/// # Example
/// ```c
///     QkClassicalRegister *cr = qk_classical_register_new(1024, "creg");
///     qk_classical_register_free(cr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``reg`` is not either null or a valid pointer to a
/// ``QkClassicalRegister``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_classical_register_free(reg: *mut ClassicalRegister) {
    if !reg.is_null() {
        if !reg.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(reg);
        }
    }
}

/// @ingroup QkClassicalRegister
/// Get the name of a classical register.
///
/// Returns a newly allocated C string containing the name of the classical register.
///
/// @param creg A pointer to the classical register.
///
/// @return A pointer to a newly allocated null-terminated string containing the register name.
///         The caller must free this string using `qk_str_free`.
///
/// # Example
/// ```c
///     QkClassicalRegister *cr = qk_classical_register_new(3, "my_creg");
///     char *name = qk_classical_register_name(cr);
///     printf("Register name: %s\n", name);
///     qk_str_free(name);
///     qk_classical_register_free(cr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``creg`` is not a valid, non-null pointer to a ``QkClassicalRegister``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_classical_register_name(creg: *const ClassicalRegister) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let creg = unsafe { const_ptr_as_ref(creg) };

    CString::new(creg.name())
        .expect("Register name should not contain null bytes")
        .into_raw()
}

/// @ingroup QkClassicalRegister
/// Get the number of classical bits in a classical register.
///
/// Returns the size (number of classical bits) of the classical register.
///
/// @param creg A pointer to the classical register.
///
/// @return The number of classical bits in the register.
///
/// # Example
/// ```c
///     QkClassicalRegister *cr = qk_classical_register_new(3, "my_creg");
///     size_t num_clbits = qk_classical_register_num_bits(cr);
///     printf("Number of clbits: %zu\n", num_clbits);  // Prints: 3
///     qk_classical_register_free(cr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``creg`` is not a valid, non-null pointer to a ``QkClassicalRegister``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_classical_register_num_bits(creg: *const ClassicalRegister) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let creg = unsafe { const_ptr_as_ref(creg) };

    creg.len()
}

/// @ingroup QkClassicalRegister
/// Get the circuit bit indices for all clbits in a classical register.
///
/// Outputs a mapping from each clbit in the classical register to its corresponding index in the circuit's
/// clbit list. If a clbit from the register is not present in the circuit, its index
/// in the mapping will be ``UINT32_MAX``.
///
/// @param creg A pointer to the classical register to map.
/// @param circuit A pointer to the circuit containing the clbits.
/// @param out_bits A pointer to an array where the mapped indices will be written.
///     The array must be pre-allocated with at least `qk_classical_register_num_bits` elements
///     for ``creg``. Each element will contain either the circuit bit index or ``UINT32_MAX``
///     if the clbit is not in the circuit.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(0, 2);
///     QkClassicalRegister *cr = qk_classical_register_new(3, "my_creg");
///     qk_circuit_add_classical_register(qc, cr);
///
///     uint32_t bit_indices[3];
///
///     qk_classical_register_circuit_bits(cr, qc, bit_indices);
///
///     // bit_indices now contains [2, 3, 4] since all clbits are in the circuit
///     // and the circuit has 2 anonymous clbits
///
///     qk_classical_register_free(cr);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if:
/// - ``creg`` is not a valid, non-null pointer to a ``QkClassicalRegister``.
/// - ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
/// - ``creg`` has one or more bits and ``out_bits`` is not an aligned, non-null pointer to an array with at least
///   ``qk_classical_register_num_bits(creg)`` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_classical_register_circuit_bits(
    creg: *const ClassicalRegister,
    circuit: *const CircuitData,
    out_bits: *mut u32,
) {
    // SAFETY: Per documentation, creg is a valid, non-null pointer.
    let creg = unsafe { const_ptr_as_ref(creg) };
    // SAFETY: Per documentation, circuit is a valid, non-null pointer.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    creg.iter().enumerate().for_each(|(i, clbit)| {
        let mapped_clbit = circuit.clbit_index(&clbit).map_or(u32::MAX, |c| c);
        // SAFETY: Per documentation, out_bits is aligned and has at least creg.len() elements
        unsafe { out_bits.add(i).write(mapped_clbit) };
    });
}

/// @ingroup QkCircuit
/// Add a quantum register to a given quantum circuit
///
/// @param circuit A pointer to the circuit.
/// @param reg A pointer to the quantum register
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(0, 0);
///     QkQuantumRegister *qr = qk_quantum_register_new(1024, "my_little_register");
///     qk_circuit_add_quantum_register(qc, qr);
///     qk_quantum_register_free(qr);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit`` and
/// if ``reg`` is not a valid, non-null pointer to a ``QkQuantumRegister``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_add_quantum_register(
    circuit: *mut CircuitData,
    reg: *const QuantumRegister,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let qreg = unsafe { const_ptr_as_ref(reg) };

    circuit
        .add_qreg(qreg.clone(), true)
        .expect("Invalid register unable to be added to circuit");
}

/// @ingroup QkCircuit
/// Get the number of quantum registers in the circuit.
///
/// Returns the number of quantum registers that have been added to the circuit.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of quantum registers in the circuit.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(0, 0);
///     QkQuantumRegister *qr1 = qk_quantum_register_new(2, "qr1");
///     QkQuantumRegister *qr2 = qk_quantum_register_new(3, "qr2");
///     qk_circuit_add_quantum_register(qc, qr1);
///     qk_circuit_add_quantum_register(qc, qr2);
///
///     size_t num_qregs = qk_circuit_num_quantum_registers(qc);
///     printf("Number of quantum registers: %zu\n", num_qregs);  // Prints: 2
///
///     qk_quantum_register_free(qr1);
///     qk_quantum_register_free(qr2);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_num_quantum_registers(circuit: *const CircuitData) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    circuit.qregs().len()
}

/// @ingroup QkCircuit
/// Returns the quantum register at the specified index.
///
/// Returns a borrowed pointer to the quantum register at the specified index.
/// The returned pointer is valid as long as the circuit exists and is not modified.
/// The caller should not free the returned pointer.
///
/// @param circuit A pointer to the circuit.
/// @param qreg_idx The index of the quantum register to retrieve.
///
/// @return A borrowed pointer to the quantum register at the specified index.
///         Do not free this pointer.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(0, 0);
///     QkQuantumRegister *qr1 = qk_quantum_register_new(2, "qr1");
///     qk_circuit_add_quantum_register(qc, qr1);
///
///     const QkQuantumRegister *retrieved = qk_circuit_get_quantum_register(qc, 0);
///
///     qk_quantum_register_free(qr1);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``,
/// or if ``qreg_idx`` is out of bounds (>= the number of quantum registers in the circuit).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_get_quantum_register(
    circuit: *const CircuitData,
    qreg_idx: usize,
) -> *const QuantumRegister {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    ptr::from_ref(&circuit.qregs()[qreg_idx])
}

/// @ingroup QkCircuit
/// Add a classical register to a given quantum circuit
///
/// @param circuit A pointer to the circuit.
/// @param reg A pointer to the classical register
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(0, 0);
///     QkClassicalRegister *cr = qk_classical_register_new(24, "my_big_register");
///     qk_circuit_add_classical_register(qc, cr);
///     qk_classical_register_free(cr);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit`` and
/// if ``reg`` is not a valid, non-null pointer to a ``QkClassicalRegister``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_add_classical_register(
    circuit: *mut CircuitData,
    reg: *const ClassicalRegister,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let creg = unsafe { const_ptr_as_ref(reg) };

    circuit
        .add_creg(creg.clone(), true)
        .expect("Invalid register unable to be added to circuit");
}

/// @ingroup QkCircuit
/// Get the number of classical registers in the circuit.
///
/// Returns the number of classical registers that have been added to the circuit.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of classical registers in the circuit.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(0, 0);
///     QkClassicalRegister *cr1 = qk_classical_register_new(2, "cr1");
///     QkClassicalRegister *cr2 = qk_classical_register_new(3, "cr2");
///     qk_circuit_add_classical_register(qc, cr1);
///     qk_circuit_add_classical_register(qc, cr2);
///
///     size_t num_cregs = qk_circuit_num_classical_registers(qc);
///     printf("Number of classical registers: %zu\n", num_cregs);  // Prints: 2
///
///     qk_classical_register_free(cr1);
///     qk_classical_register_free(cr2);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_num_classical_registers(circuit: *const CircuitData) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    circuit.cregs().len()
}

/// @ingroup QkCircuit
/// Returns the classical register at the specified index.
///
/// Returns a borrowed pointer to the classical register at the specified index.
/// The returned pointer is valid as long as the circuit exists and is not modified.
/// The caller should not free the returned pointer.
///
/// @param circuit A pointer to the circuit.
/// @param creg_idx The index of the classical register to retrieve.
///
/// @return A borrowed pointer to the classical register at the specified index.
///         Do not free this pointer.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(0, 0);
///     QkClassicalRegister *cr1 = qk_classical_register_new(2, "cr1");
///     qk_circuit_add_classical_register(qc, cr1);
///
///     const QkClassicalRegister *retrieved = qk_circuit_get_classical_register(qc, 0);
///
///     qk_classical_register_free(cr1);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``,
/// or if ``creg_idx`` is out of bounds (>= the number of classical registers in the circuit).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_get_classical_register(
    circuit: *const CircuitData,
    creg_idx: usize,
) -> *const ClassicalRegister {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    ptr::from_ref(&circuit.cregs()[creg_idx])
}

/// @ingroup QkCircuit
/// Create a copy of a ``QkCircuit``.
///
/// @param circuit A pointer to the circuit to copy.
///
/// @return A new pointer to a copy of the input ``circuit``.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 100);
///     QkCircuit *copy = qk_circuit_copy(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_copy(circuit: *const CircuitData) -> *mut CircuitData {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    Box::into_raw(Box::new(circuit.clone()))
}

/// @ingroup QkCircuit
/// Get the number of qubits the circuit contains.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of qubits the circuit is defined on.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 100);
///     uint32_t num_qubits = qk_circuit_num_qubits(qc);  // num_qubits==100
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_num_qubits(circuit: *const CircuitData) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    circuit.num_qubits() as u32
}

/// @ingroup QkCircuit
/// Get the number of clbits the circuit contains.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of clbits the circuit is defined on.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 50);
///     uint32_t num_clbits = qk_circuit_num_clbits(qc);  // num_clbits==50
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_num_clbits(circuit: *const CircuitData) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    circuit.num_clbits() as u32
}

/// @ingroup QkCircuit
/// Get the number of unbound symbols in ``QkParam`` objects in the circuit.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of unbound ``QkParam`` symbols in the circuit.
///
/// # Example
///
/// ```c
/// QkCircuit *qc = qk_circuit_new(2, 0);
/// QkParam *x = qk_param_new_symbol("x");
/// QkParam *y = qk_param_new_symbol("y");
///
/// uint32_t q0[1] = {0};
/// const QkParam *rx_param[1] = {x};
/// const QkParam *ry_param[1] = {y};
///
/// qk_circuit_parameterized_gate(qc, QkGate_RX, q0, rx_param);
/// qk_circuit_parameterized_gate(qc, QkGate_RY, q0, ry_param);
///
/// // check the number of QkParam symbols
/// size_t num_symbols = qk_circuit_num_param_symbols(qc); // == 2
///
/// qk_param_free(x);
/// qk_param_free(y);
/// qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_num_param_symbols(circuit: *const CircuitData) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    circuit.num_parameters()
}

/// @ingroup QkCircuit
/// Get the global phase of the circuit.
///
/// This function returns a copy of the circuit's global phase
/// and the value must be freed via :c:func:`qk_param_free`
/// after usage.
///
/// @param circuit A pointer to the circuit.
///
/// @return The global phase of the circuit.
///
/// # Example
/// ```c
/// QkCircuit *qc = qk_circuit_new(100, 100);
/// QkParam *phase = qk_circuit_global_phase(qc);
/// qk_param_free(phase);
/// qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_global_phase(circuit: *const CircuitData) -> *mut Param {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    Box::into_raw(Box::new(circuit.global_phase().clone()))
}

/// @ingroup QkCircuit
/// Set the global phase of the circuit.
///
/// This function copies the new global phase upon setting it,
/// so the caller retains ownership of the ``QkParam`` phase,
/// and the value of the phase must be freed via :c:func:`qk_param_free`
/// after setting.
///
/// @param circuit A pointer to the circuit.
/// @param phase A pointer to the global phase to set.
///
/// @return ``QkExitCode_Success`` upon successful setting of the global phase. Upon failure,
///     ``QkExitCode_ParameterNameConflict`` indicates that a new parameter symbol has a name
///     conflict with an existing one. ``QkExitCode_ParameterError`` describes other generic
///     failures when attempting to track the parameter symbols.
///
/// # Example
/// ```c
/// QkCircuit *qc = qk_circuit_new(100, 100);
/// QkParam *new_global_phase = qk_param_from_double(1.23);
/// qk_circuit_set_global_phase(qc, new_global_phase);
/// qk_param_free(new_global_phase);
/// qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit`` and
/// if ``phase`` is not a valid, non-null pointer to a ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_set_global_phase(
    circuit: *mut CircuitData,
    phase: *const Param,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let phase = unsafe { const_ptr_as_ref(phase) };

    match circuit.set_global_phase_param(phase.clone()) {
        Ok(()) => ExitCode::Success,
        Err(CircuitDataError::ParameterTableError(ParameterTableError::NameConflict(_))) => {
            ExitCode::ParameterNameConflict
        }
        Err(_) => ExitCode::ParameterError,
    }
}

/// @ingroup QkCircuit
/// Free the circuit.
///
/// @param circuit A pointer to the circuit to free.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 100);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not either null or a valid pointer to a
/// ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_free(circuit: *mut CircuitData) {
    if !circuit.is_null() {
        if !circuit.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(circuit);
        }
    }
}

/// @ingroup QkCircuit
/// Append a ``QkGate`` to the circuit.
///
/// @param circuit A pointer to the circuit to add the gate to.
/// @param gate The StandardGate to add to the circuit.
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the gate on. This
///     can be a null pointer if there are no qubits for ``gate`` (e.g. ``QkGate_GlobalPhase``).
/// @param params The pointer to the array of ``double`` values to use for the gate parameters.
///     This can be a null pointer if there are no parameters for ``gate`` (e.g. ``QkGate_H``).
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubit, NULL);
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
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_gate(
    circuit: *mut CircuitData,
    gate: StandardGate,
    qubits: *const u32,
    params: *const f64,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
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
            _ => unreachable!(),
        };
        let params: &[Param] = match gate.num_params() {
            0 => &[],
            1 => &[(*params.wrapping_add(0)).into()],
            2 => &[
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
            ],
            3 => &[
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
            ],
            4 => &[
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
                (*params.wrapping_add(3)).into(),
            ],
            // There are no ``QkGate``s that take > 4 params
            _ => unreachable!(),
        };
        circuit.push_standard_gate(gate, params, qargs).unwrap()
    }
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Append a ``QkGate`` with ``QkParam*`` parameters to the circuit.
///
/// @param circuit A pointer to the circuit to add the gate to.
/// @param gate The ``QkGate`` to add to the circuit.
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the gate on. This
/// can be a null pointer if there are no qubits for ``gate`` (e.g. ``QkGate_GlobalPhase``).
/// @param params The pointer to the array of ``QkParam*`` parameters to use for the gate parameters.
/// This can be a null pointer if there are no parameters for ``gate`` (e.g. ``QkGate_H``).
///
/// @return ``QkExitCode_Success`` upon successful append. Upon failure,
///     ``QkExitCode_ParameterNameConflict`` indicates that a new parameter symbol has a name
///     conflict with an existing one. ``QkExitCode_ParameterError`` describes other generic
///     failures when attempting to track the parameter symbols.
///
/// # Example
///
/// ```c
/// QkCircuit *qc = qk_circuit_new(100, 0);
/// QkParam *theta = qk_param_new_symbol("theta");
/// uint32_t qubit[1] = {0};
/// const QkParam* params[1] = {theta};
/// qk_circuit_parameterized_gate(qc, QkGate_RX, qubit, params); // add RX(theta) to the circuit
/// ```
///
/// # Safety
///
/// The ``qubits`` and ``params`` types are expected to be a pointer to an array of ``uint32_t``
/// and ``QkParam*`` respectively where the length is matching the expectations for the standard
/// gate. If the array is insufficiently long the behavior of this function is undefined as this
/// will read outside the bounds of the array. It can be a null pointer if there are no qubits
/// or params for a given gate. You can check ``qk_gate_num_qubits`` and ``qk_gate_num_params`` to
/// determine how many qubits and params are required for a given gate.
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``,
/// or if any of the elements in the ``params`` array is not a valid, non-null pointer to a
/// ``QkParam``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_parameterized_gate(
    circuit: *mut CircuitData,
    gate: StandardGate,
    qubits: *const u32,
    params: *const *const Param,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits() elements.
    let qargs: &[Qubit] = unsafe {
        match gate.num_qubits() {
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
            _ => unreachable!(),
        }
    };

    // SAFETY: Per documentation, the params pointer is an array of num_params() elements, and
    // each element is a valid, non-null pointer to a Param.
    let params: Vec<Param> =
        unsafe { ::std::slice::from_raw_parts(params, gate.num_params() as usize) }
            .iter()
            .map(|&ptr| unsafe { const_ptr_as_ref(ptr) }.clone())
            .collect();

    match circuit.push_standard_gate(gate, &params, qargs) {
        Ok(()) => ExitCode::Success,
        Err(CircuitDataError::ParameterTableError(ParameterTableError::NameConflict(_))) => {
            ExitCode::ParameterNameConflict
        }
        Err(_) => ExitCode::ParameterError,
    }
}

/// @ingroup QkCircuit
/// Get the number of qubits for a ``QkGate``.
///
/// @param gate The ``QkGate`` to get the number of qubits for.
///
/// @return The number of qubits the gate acts on.
///
/// # Example
/// ```c
///     uint32_t num_qubits = qk_gate_num_qubits(QkGate_CCX);
/// ```
///
#[unsafe(no_mangle)]
pub extern "C" fn qk_gate_num_qubits(gate: StandardGate) -> u32 {
    gate.num_qubits()
}

/// @ingroup QkCircuit
/// Get the number of parameters for a ``QkGate``.
///
/// @param gate The ``QkGate`` to get the number of qubits for.
///
/// @return The number of parameters the gate has.
///
/// # Example
/// ```c
///     uint32_t num_params = qk_gate_num_params(QkGate_R);
/// ```
///
#[unsafe(no_mangle)]
pub extern "C" fn qk_gate_num_params(gate: StandardGate) -> u32 {
    gate.num_params()
}

/// @ingroup QkCircuit
/// Append a measurement to the circuit
///
/// @param circuit A pointer to the circuit to add the measurement to
/// @param qubit The ``uint32_t`` for the qubit to measure
/// @param clbit The ``uint32_t`` for the clbit to store the measurement outcome in
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 1);
///     qk_circuit_measure(qc, 0, 0);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_measure(
    circuit: *mut CircuitData,
    qubit: u32,
    clbit: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    circuit
        .push_packed_operation(
            PackedOperation::from_standard_instruction(StandardInstruction::Measure),
            None,
            &[Qubit(qubit)],
            &[Clbit(clbit)],
        )
        .unwrap();
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Append a reset to the circuit
///
/// @param circuit A pointer to the circuit to add the reset to
/// @param qubit The ``uint32_t`` for the qubit to reset
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     qk_circuit_reset(qc, 0);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_reset(circuit: *mut CircuitData, qubit: u32) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    circuit
        .push_packed_operation(
            PackedOperation::from_standard_instruction(StandardInstruction::Reset),
            None,
            &[Qubit(qubit)],
            &[],
        )
        .unwrap();
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Append a barrier to the circuit.
///
/// @param circuit A pointer to the circuit to add the barrier to.
/// @param num_qubits The number of qubits wide the barrier is.
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the barrier on.
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 1);
///     uint32_t qubits[5] = {0, 1, 2, 3, 4};
///     qk_circuit_barrier(qc, qubits, 5);
/// ```
///
/// # Safety
///
/// The length of the array ``qubits`` points to must be ``num_qubits``. If there is
/// a mismatch the behavior is undefined.
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_barrier(
    circuit: *mut CircuitData,
    qubits: *const u32,
    num_qubits: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qubits: Vec<Qubit> = unsafe {
        (0..num_qubits)
            .map(|idx| Qubit(*qubits.wrapping_add(idx as usize)))
            .collect()
    };
    circuit
        .push_packed_operation(
            PackedOperation::from_standard_instruction(StandardInstruction::Barrier(num_qubits)),
            None,
            &qubits,
            &[],
        )
        .unwrap();
    ExitCode::Success
}

/// An individual operation count represented by the operation name
/// and the number of instances in the circuit.
#[repr(C)]
pub struct OpCount {
    /// A nul terminated string representing the operation name
    name: *const c_char,
    /// The number of instances of this operation in the circuit
    count: usize,
}

/// An array of ``OpCount`` objects representing the total counts of all
/// the operation types in a circuit.
#[repr(C)]
pub struct OpCounts {
    /// A array of size ``len`` containing ``OpCount`` objects for each
    /// type of operation in the circuit
    data: *mut OpCount,
    /// The number of elements in ``data``
    len: usize,
}

#[inline]
fn conjugate(matrix: ArrayView2<Complex64>) -> Array2<Complex64> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(i, j)| {
        matrix[(j, i)].conj()
    })
}

/// Check an [ArrayType] represents a unitary matrix. Uses an element-wise check; if
/// any element in ``conjugate(matrix) * matrix`` differs from the identity by more than ``tol``
/// (in magnitude), the matrix is not considered unitary.
fn is_unitary(matrix: &ArrayType, tol: f64) -> bool {
    let not_unitary = match matrix {
        ArrayType::OneQ(mat) => (mat.adjoint() * mat - Matrix2::identity())
            .iter()
            .any(|val| val.abs() > tol),
        ArrayType::TwoQ(mat) => (mat.adjoint() * mat - Matrix4::identity())
            .iter()
            .any(|val| val.abs() > tol),
        ArrayType::NDArray(mat) => {
            let product = mat.dot(&conjugate(mat.view()));
            product.indexed_iter().any(|((row, col), value)| {
                if row == col {
                    (value - Complex64::ONE).abs() > tol
                } else {
                    value.abs() > tol
                }
            })
        }
    };
    !not_unitary // using double negation to use ``any`` (faster) instead of ``all``
}

/// Create a unitary matrix `ArrayType` from a pointer to a row-major contiguous matrix of the
/// correct dimensions.
///
/// If `tol` is `Some`, the unitary matrix is checked for tolerance against the given value.  If the
/// tolerance check fails, no array is returned.
///
/// The data is copied out of `matrix`.
///
/// # Safety
///
/// `matrix` must be aligned and valid for `4 ** num_qubits` reads.
pub(crate) unsafe fn unitary_from_pointer(
    matrix: *const Complex64,
    num_qubits: u32,
    tol: Option<f64>,
) -> Option<ArrayType> {
    let dim = 1 << num_qubits;
    // SAFETY: per documentation, `matrix` is aligned and valid for `4**num_qubits` reads.
    let raw = unsafe { ::std::slice::from_raw_parts(matrix, dim * dim) };
    let mat = match num_qubits {
        1 => ArrayType::OneQ(Matrix2::from_fn(|i, j| raw[i * dim + j])),
        2 => ArrayType::TwoQ(Matrix4::from_fn(|i, j| raw[i * dim + j])),
        _ => ArrayType::NDArray(Array2::from_shape_fn((dim, dim), |(i, j)| raw[i * dim + j])),
    };
    match tol {
        Some(tol) => is_unitary(&mat, tol).then_some(mat),
        None => Some(mat),
    }
}

/// @ingroup QkCircuit
/// Append an arbitrary unitary matrix to the circuit.
///
/// @param circuit A pointer to the circuit to append the unitary to.
/// @param matrix A pointer to the ``QkComplex64`` array representing the unitary matrix.
///     This must be a row-major, unitary matrix of dimension ``2 ^ num_qubits x 2 ^ num_qubits``.
///     More explicitly: the ``(i, j)``-th element is given by ``matrix[i * 2^n + j]``.
///     The contents of ``matrix`` are copied inside this function before being added to the circuit,
///     so caller keeps ownership of the original memory that ``matrix`` points to and can reuse it
///     after the call and the caller is responsible for freeing it.
/// @param qubits A pointer to array of qubit indices, of length ``num_qubits``.
/// @param num_qubits The number of qubits the unitary acts on.
/// @param check_input When true, the function verifies that the matrix is unitary.
///     If set to False the caller is responsible for ensuring the matrix is unitary, if
///     the matrix is not unitary this is undefined behavior and will result in a corrupt
///     circuit.
///
/// @return An exit code.
///
/// # Example
/// ```c
/// QkComplex64 c0 = {0, 0};  // 0+0i
/// QkComplex64 c1 = {1, 0};  // 1+0i
///
/// const uint32_t num_qubits = 1;
/// QkComplex64 unitary[2*2] = {c0, c1,  // row 0
///                             c1, c0}; // row 1
///
/// QkCircuit *circuit = qk_circuit_new(1, 0);  // 1 qubit circuit
/// uint32_t qubit[1] = {0};  // qubit to apply the unitary on
/// qk_circuit_unitary(circuit, unitary, qubit, num_qubits, true);
/// ```
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated:
///
/// * ``circuit`` is a valid, non-null pointer to a ``QkCircuit``
/// * ``matrix`` is an aligned pointer to ``4**num_qubits`` initialized ``QkComplex64`` values
/// * ``qubits`` is an aligned pointer to ``num_qubits`` initialized ``uint32_t`` values
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_unitary(
    circuit: *mut CircuitData,
    matrix: *const Complex64,
    qubits: *const u32,
    num_qubits: u32,
    check_input: bool,
) -> ExitCode {
    // SAFETY: Caller guarantees pointer validation, alignment
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let mat = unsafe { unitary_from_pointer(matrix, num_qubits, check_input.then_some(1e-12)) };
    let Some(mat) = mat else {
        return ExitCode::ExpectedUnitary;
    };
    let qubits = if num_qubits == 0 {
        // This handles the case of C passing us a null pointer for the qubits; Rust slices
        // can't be backed by the null pointer even when empty.
        &[]
    } else {
        // SAFETY: per documentation, `qubits` is aligned and valid for `num_qubits` reads.  Per
        // previous check, `num_qubits` is nonzero so `qubits` cannot be null.
        unsafe { ::std::slice::from_raw_parts(qubits as *const Qubit, num_qubits as usize) }
    };

    // Create PackedOperation -> push to circuit_data
    let u_gate = Box::new(UnitaryGate { array: mat });
    let op = PackedOperation::from_unitary(u_gate);
    circuit
        .push_packed_operation(op, None, qubits, &[])
        .unwrap();
    // Return success
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Copy out the unitary matrix for a given instruction in the circuit.
///
/// Panics if the instruction at the given index is not a unitary.
///
/// @param circuit A pointer to the circuit to get the unitary from.
/// @param index The index of the instruction to get the unitary for.
/// @param out Allocated and aligned pointer to write the unitary matrix to.
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated:
/// if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``,
/// if ``index`` is out of bounds for the number of instructions in the circuit,
/// if `out` is not valid for `4**num_qubits` writes of `QkComplex64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_inst_unitary(
    circuit: *mut CircuitData,
    index: usize,
    out: *mut Complex64,
) {
    // SAFETY: Per documentation the pointer is to a valid, non-null QkCircuit.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let instr: &PackedInstruction = &circuit.data()[index];
    let OperationRef::Unitary(unitary) = instr.op.view() else {
        panic!("Instruction at index {index} is not a unitary");
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

/// @ingroup QkCircuit
/// Get the "kind" of circuit instruction.
///
/// @param circuit A pointer to the circuit to get the instruction kind from.
/// @param index The index of the instruction to get the kind for.
///
/// @return A ``QkOperationKind`` enum value representing the kind of instruction at the given
/// index.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``
/// and if ``index`` is out of bounds for the number of instructions in the circuit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_instruction_kind(
    circuit: *const CircuitData,
    index: usize,
) -> COperationKind {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let instr: &PackedInstruction = &circuit.data()[index];
    match instr.op.view() {
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

/// @ingroup QkCircuit
/// Return a list of string names for instructions in a circuit and their counts.
///
/// To properly free the memory allocated by the struct, you should call ``qk_opcounts_clear``.
/// Dropping the ``QkOpCounts`` struct without doing so will leave the stored array of ``QkOpCount``
/// allocated and produce a memory leak.
///
/// @param circuit A pointer to the circuit to get the counts for.
///
/// @return An ``QkOpCounts`` struct containing the circuit operation counts.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t qubits[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubits, NULL);
///     QkOpCounts counts = qk_circuit_count_ops(qc);
///     // .. once done
///     qk_opcounts_clear(&counts);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_count_ops(circuit: *const CircuitData) -> OpCounts {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let count_ops = circuit.count_ops();
    let output = {
        let vec: Vec<OpCount> = count_ops
            .into_iter()
            .map(|(name, count)| OpCount {
                name: CString::new(name).unwrap().into_raw(),
                count,
            })
            .collect();
        vec.into_boxed_slice()
    };
    let len = output.len();
    let data = Box::into_raw(output) as *mut OpCount;
    OpCounts { data, len }
}

/// @ingroup QkCircuit
/// Return the total number of instructions in the circuit.
///
/// @param circuit A pointer to the circuit to get the total number of instructions for.
///
/// @return The total number of instructions in the circuit.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubit, NULL);
///     size_t num = qk_circuit_num_instructions(qc); // 1
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_num_instructions(circuit: *const CircuitData) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    circuit.len()
}

/// A circuit instruction representation.
///
/// This struct represents the data contained in an individual instruction in a ``QkCircuit``.
/// It is not a pointer to the underlying object, but contains a copy of the properties of the
/// instruction for inspection.
#[repr(C)]
pub struct CInstruction {
    /// The instruction name
    name: *mut c_char,
    /// A pointer to an array of qubit indices this instruction operates on.
    qubits: *mut u32,
    /// A pointer to an array of clbit indices this instruction operates on.
    clbits: *mut u32,
    /// A pointer to an array of parameter values for this instruction.
    params: *mut *mut Param,
    /// The number of qubits for this instruction.
    num_qubits: u32,
    /// The number of clbits for this instruction.
    num_clbits: u32,
    /// The number of parameters for this instruction.
    num_params: u32,
}
impl CInstruction {
    /// Create a `CInstruction` that owns pointers to copies of the information in the given
    /// `PackedInstruction`.
    ///
    /// This must be cleared by a call to `qk_circuit_instruction_clear` to avoid leaking its
    /// allocations.
    ///
    /// Panics if the operation name contains a nul, or if the instruction has non-numeric
    /// parameters (not [Param::Float] or [Param::ParameterExpression]).
    pub(crate) fn from_packed_instruction_with_numeric(
        packed: &PackedInstruction,
        qargs_interner: &Interner<[Qubit]>,
        cargs_interner: &Interner<[Clbit]>,
    ) -> Self {
        let name = CString::new(packed.op.name())
            .expect("names do not contain nul")
            .into_raw();
        let qargs = qargs_interner.get(packed.qubits);
        let cargs = cargs_interner.get(packed.clbits);
        let params = packed
            .params_view()
            .iter()
            .map(|p| match p {
                Param::Float(_) | Param::ParameterExpression(_) => {
                    Some(Box::into_raw(Box::new(p.clone())))
                }
                _ => None,
            })
            .collect::<Option<Box<[*mut Param]>>>()
            .expect("caller is responsible for ensuring all parameters are numeric");
        Self {
            name,
            num_qubits: qargs.len() as u32,
            qubits: Box::leak(qargs.iter().map(|q| q.0).collect::<Box<[u32]>>()).as_mut_ptr(),
            num_clbits: cargs.len() as u32,
            clbits: Box::leak(cargs.iter().map(|c| c.0).collect::<Box<[u32]>>()).as_mut_ptr(),
            num_params: params.len() as u32,
            params: Box::leak(params).as_mut_ptr(),
        }
    }
}

/// @ingroup QkCircuit
/// Return the instruction details for an instruction in the circuit.
///
/// This function is used to get the instruction details for a given instruction in
/// the circuit.
///
/// This function allocates memory internally for the provided ``QkCircuitInstruction``
/// and thus you are responsible for calling ``qk_circuit_instruction_clear`` to
/// free it.
///
/// @param circuit A pointer to the circuit to get the instruction details for.
/// @param index The instruction index to get the instruction details of.
/// @param instruction A pointer to where to write out the ``QkCircuitInstruction``
///
///
/// # Example
/// ```c
///     QkCircuitInstruction inst;
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubit, NULL);
///     qk_circuit_get_instruction(qc, 0, &inst);
///     qk_circuit_instruction_clear(&inst);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``. The
/// value for ``index`` must be less than the value returned by ``qk_circuit_num_instructions``
/// otherwise this function will panic. Behavior is undefined if ``instruction`` is not a valid,
/// non-null pointer to a memory allocation with sufficient space for a ``QkCircuitInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_get_instruction(
    circuit: *const CircuitData,
    index: usize,
    instruction: *mut CInstruction,
) {
    // SAFETY: Per documentation, `circuit` is a pointer to valid data.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let inst = CInstruction::from_packed_instruction_with_numeric(
        &circuit.data()[index],
        circuit.qargs_interner(),
        circuit.cargs_interner(),
    );
    // SAFETY: per documentation, `instruction` is a pointer to a sufficient allocation.
    unsafe { instruction.write(inst) };
}

/// @ingroup QkCircuit
/// Apply a ``QkPauliProductRotation`` to a circuit.
///
/// @param circuit The circuit to apply the operation to.
/// @param rotation A pointer to the ``QkPauliProductRotation`` to apply.
/// @param qubits A pointer to the qubit array.
///
/// # Example
///
/// ```c
/// // build a IXYZ Pauli rotation
/// bool x[4] = {false, true, true, false};
/// bool z[4] = {false, false, true, true};
/// QkParam *angle = qk_param_from_double(1.0);
/// QkPauliProductRotation rotation = {x, z, 4, angle};
/// // append it to a circuit
/// QkCircuit *circuit = qk_circuit_new(10, 1);
/// uint32_t qubits[4] = {0, 1, 2, 3};
/// qk_circuit_pauli_product_rotation(circuit, &rotation, qubits);
/// // do something with the circuit... and then free it
/// qk_param_free(angle);
/// qk_circuit_free(circuit);
/// ```
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated:
/// * ``circuit`` is a valid, non-null pointer to a ``QkCircuit``
/// * ``rotation`` is a valid, non-null pointer to a coherent ``QkPauliProductRotation``.
///   Specifically, the ``rotation->z`` and ``rotation->x`` data arrays must be readable for
///   ``rotation->len`` elements.
/// * ``qubits`` is an aligned pointer to ``rotation->len`` initialized ``uint32_t`` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_pauli_product_rotation(
    circuit: *mut CircuitData,
    rotation: *const CPauliProductRotation,
    qubits: *const u32,
) {
    // SAFETY: The user guarantees the circuit pointer is valid.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };

    // SAFETY: The user guarantees the rotation pointer is valid and the data
    // is coherent. This allows us reading the Z and X arrays.
    let c_data = unsafe { const_ptr_as_ref(rotation) };
    let angle = unsafe { const_ptr_as_ref(c_data.angle) };
    let pbc_rotation = PauliProductRotation {
        z: unsafe { ::std::slice::from_raw_parts(c_data.z, c_data.len) }.to_vec(),
        x: unsafe { ::std::slice::from_raw_parts(c_data.x, c_data.len) }.to_vec(),
        angle: angle.clone(),
    };

    // SAFETY: The user guarantees the qubit
    let qubits = unsafe {
        ::std::slice::from_raw_parts(qubits as *const Qubit, pbc_rotation.num_qubits() as usize)
    };
    let params = Some(Parameters::Params(smallvec![angle.clone()]));

    let pbc = PauliBased::PauliProductRotation(pbc_rotation);
    let packed = PackedOperation::from_pauli_based(Box::new(pbc));
    circuit
        .push_packed_operation(packed, params, qubits, &[])
        .expect("Failed pushing packed QkPauliProductRotation");
}

/// @ingroup QkCircuit
/// Get the ``QkPauliProductRotation`` data from a circuit instruction.
///
/// For a circuit with a ``QkPauliProductRotation`` instruction at index ``index``, this function
/// will populate the ``instruction`` pointer with a copy of ``QkPauliProductRotation`` data. Note that
/// this data lives independently of the circuit and must be freed manually with
/// ``qk_pauli_product_rotation_clear``.
///
/// If the instruction at the provided ``index`` **is not** a ``QkPauliProductRotation``, this function
/// will return ``QkExitCode_InvalidOperationKind`` error. You can verify that the instruction has the
/// right kind using ``qk_circuit_instruction_kind``.
///
/// @param circuit A pointer to the circuit to retrieve the instruction details from.
/// @param index The circuit instruction index.
/// @param instruction A pointer to an allocated ``QkPauliProductRotation`` to store the data.
///
/// @return ``QkExitCode_Success`` if the data was written into the instruction, or
///    ``QkExitCode_InvalidOperationKind`` if the index did not point to a ``QkPauliProductRotation``.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``. The
/// value for ``index`` must be less than the value returned by ``qk_circuit_num_instructions``
/// otherwise this function will panic. Behavior is undefined if ``instruction`` is not a valid,
/// non-null pointer to a memory allocation with sufficient space for a ``QkPauliProductRotation``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_inst_pauli_product_rotation(
    circuit: *const CircuitData,
    index: usize,
    instruction: *mut CPauliProductRotation,
) -> ExitCode {
    // SAFETY: The user guarantees the circuit pointer is valid to read.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    // Ensure the operation has the correct type, otherwise return early
    let OperationRef::PauliProductRotation(rotation) = circuit.data()[index].op.view() else {
        return ExitCode::InvalidOperationKind;
    };

    // We clone the internal data and then leak the box to give C access to the memory.
    // This means the user has to manually free the allocated memory in the Pauli rotation.
    let len = rotation.x.len();
    let x = rotation.x.clone().into_boxed_slice();
    let z = rotation.z.clone().into_boxed_slice();
    let angle = Box::into_raw(Box::new(rotation.angle.clone()));
    let out = CPauliProductRotation {
        x: Box::into_raw(x) as *mut bool,
        z: Box::into_raw(z) as *mut bool,
        len,
        angle,
    };

    // SAFETY: The user guarantees `instruction` points to sufficiently allocated memory.
    unsafe { instruction.write(out) };

    ExitCode::Success
}

/// @ingroup QkCircuit
/// Apply a ``QkPauliProductMeasurement`` to a circuit.
///
/// @param circuit The circuit to apply the operation to.
/// @param measurement A pointer to the ``QkPauliProductMeasurement`` to apply.
/// @param qubits A pointer to the qubit array.
/// @param clbit A single ``uint32_t`` specifying the measurement qubit.
///
/// # Example
///
/// ```c
/// // build a XZ Pauli measurement
/// bool x[2] = {true, false};
/// bool z[2] = {false, true};
/// QkPauliProductMeasurement measure = {x, z, 2, false};
/// // append it to a circuit
/// QkCircuit *circuit = qk_circuit_new(10, 1);
/// uint32_t qubits[2] = {0, 2};
/// uint32_t clbit = 0;
/// qk_circuit_pauli_product_measurement(circuit, &measure, qubits, clbit);
/// // do something with the circuit... and then free it
/// qk_circuit_free(circuit);
/// ```
///
/// # Safety
///
/// Behavior is undefined if any of the following is violated:
/// * ``circuit`` is a valid, non-null pointer to a ``QkCircuit``
/// * ``measurement`` is a valid, non-null pointer to a coherent ``QkPauliProductMeasurement``.
///   Specifically, the ``measurement->z`` and ``measurement->x`` data arrays must be readable for
///   ``measurement->len`` elements.
/// * ``qubits`` is an aligned pointer to ``rotation->len`` initialized ``uint32_t`` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_pauli_product_measurement(
    circuit: *mut CircuitData,
    measurement: *const CPauliProductMeasurement,
    qubits: *const u32,
    clbit: u32,
) {
    // SAFETY: The user guarantees the circuit pointer is valid.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };

    // SAFETY: The user guarantees the rotation pointer is valid and the data
    // is coherent. This allows us reading the Z and X arrays.
    let c_data = unsafe { const_ptr_as_ref(measurement) };
    let pbc_measure = PauliProductMeasurement {
        z: unsafe { ::std::slice::from_raw_parts(c_data.z, c_data.len) }.to_vec(),
        x: unsafe { ::std::slice::from_raw_parts(c_data.x, c_data.len) }.to_vec(),
        neg: c_data.flip_outcome,
    };

    // SAFETY: The user guarantees the qubit
    let qubits = unsafe {
        ::std::slice::from_raw_parts(qubits as *const Qubit, pbc_measure.num_qubits() as usize)
    };
    let clbits = &[Clbit(clbit)];

    let pbc = PauliBased::PauliProductMeasurement(pbc_measure);
    let packed = PackedOperation::from_pauli_based(Box::new(pbc));
    circuit
        .push_packed_operation(packed, None, qubits, clbits)
        .expect("Failed pushing packed QkPauliProductMeasurement");
}

/// @ingroup QkCircuit
/// Get the ``QkPauliProductMeasurement`` data from a circuit instruction.
///
/// For a circuit with a ``QkPauliProductMeasurement`` instruction at index ``index``, this function
/// will populate the ``instruction`` pointer with a copy of ``QkPauliProductMeasurement`` data.
/// Note that this data lives independently of the circuit must be freed manually with
/// ``qk_pauli_product_measurement_clear``.
///
/// If the instruction at the provided ``index`` **is not** a ``QkPauliProductMeasurement``, this
/// function will return ``QkExitCode_InvalidOperationKind`` error. Verify that the instruction is
/// of the correct kind using ``qk_circuit_instruction_kind``.
///
/// @param circuit A pointer to the circuit to retrieve the instruction details from.
/// @param index The circuit instruction index.
/// @param instruction A pointer to an allocated ``QkPauliProductMeasurement`` to store the data.
///
/// @return ``QkExitCode_Success`` if the data was written into the instruction, or
///     ``QkExitCode_InvalidOperationKind`` if the index did not point to a
///     ``QkPauliProductMeasurement``.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``. The
/// value for ``index`` must be less than the value returned by ``qk_circuit_num_instructions``
/// otherwise this function will panic. Behavior is undefined if ``instruction`` is not a valid,
/// non-null pointer to a memory allocation with sufficient space for a ``QkPauliProductMeasurement``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_inst_pauli_product_measurement(
    circuit: *const CircuitData,
    index: usize,
    instruction: *mut CPauliProductMeasurement,
) -> ExitCode {
    // SAFETY: The user guarantees the circuit pointer is valid to read.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    // Ensure the operation has the correct type, otherwise return early
    let OperationRef::PauliProductMeasurement(measure) = circuit.data()[index].op.view() else {
        return ExitCode::InvalidOperationKind;
    };

    // We clone the internal data and then leak the box to give C access to the memory.
    // This means the user has to manually free the allocated memory in the Pauli rotation.
    let len = measure.x.len();
    let x = measure.x.clone().into_boxed_slice();
    let z = measure.z.clone().into_boxed_slice();
    let out = CPauliProductMeasurement {
        x: Box::into_raw(x) as *mut bool,
        z: Box::into_raw(z) as *mut bool,
        len,
        flip_outcome: measure.neg,
    };

    // SAFETY: The user guarantees `instruction` points to sufficiently allocated memory.
    unsafe { instruction.write(out) };

    ExitCode::Success
}

/// @ingroup QkCircuit
/// Clear the data in circuit instruction object.
///
/// This function doesn't free the allocation for the provided ``QkCircuitInstruction`` pointer, it
/// only frees the internal allocations for the data contained in the instruction. You are
/// responsible for allocating and freeing the actual allocation used to store a
/// ``QkCircuitInstruction``.
///
/// @param inst A pointer to the instruction to free.
///
/// # Example
/// ```c
///     QkCircuitInstruction *inst = malloc(sizeof(QkCircuitInstruction));
///     QkCircuit *qc = qk_circuit_new(100, 0);
///     uint32_t q0[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, q0, NULL);
///     qk_circuit_get_instruction(qc, 0, inst);
///     qk_circuit_instruction_clear(inst); // clear internal allocations
///     free(inst); // free struct
///     qk_circuit_free(qc); // free the circuit
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``inst`` is not a valid, non-null pointer to a ``QkCircuitInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_instruction_clear(inst: *mut CInstruction) {
    // SAFETY: Loading the data from pointers contained in a CInstruction. These should only be
    // created by rust code and are constructed from Vecs internally or CStrings.
    unsafe {
        let inst = mut_ptr_as_ref(inst);
        if inst.num_qubits > 0 && !inst.qubits.is_null() {
            let qubits = std::slice::from_raw_parts_mut(inst.qubits, inst.num_qubits as usize);
            let _: Box<[u32]> = Box::from_raw(qubits as *mut [u32]);
            inst.qubits = std::ptr::null_mut();
        }
        inst.num_qubits = 0;
        if inst.num_clbits > 0 && !inst.clbits.is_null() {
            let clbits = std::slice::from_raw_parts_mut(inst.clbits, inst.num_clbits as usize);
            let _: Box<[u32]> = Box::from_raw(clbits as *mut [u32]);
            inst.clbits = std::ptr::null_mut();
        }
        inst.num_clbits = 0;
        if inst.num_params > 0 && !inst.params.is_null() {
            let params = std::slice::from_raw_parts_mut(inst.params, inst.num_params as usize);
            for param in params.iter() {
                let _ = Box::from_raw(*param);
            }
            let _ = Box::from_raw(params);
            inst.params = std::ptr::null_mut();
        }
        inst.num_params = 0;
        if !inst.name.is_null() {
            let _ = CString::from_raw(inst.name);
            inst.name = std::ptr::null_mut();
        }
    }
}

/// @ingroup QkCircuit
/// Clear the content in a circuit operation count list.
///
/// @param op_counts The returned op count list from ``qk_circuit_count_ops``.
///
/// # Safety
///
/// Behavior is undefined if ``op_counts`` is not the object returned by ``qk_circuit_count_ops``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_opcounts_clear(op_counts: *mut OpCounts) {
    // SAFETY: The user guarantees the input is a valid OpCounts pointer.
    let op_counts = unsafe { mut_ptr_as_ref(op_counts) };

    if op_counts.len > 0 && !op_counts.data.is_null() {
        // SAFETY: We load the box from a slice pointer created from
        // the raw parts from the OpCounts::data attribute.
        unsafe {
            let slice: Box<[OpCount]> = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                op_counts.data,
                op_counts.len,
            ));
            // free the allocated strings in each OpCount
            for count in slice.iter() {
                if !count.name.is_null() {
                    let _ = CString::from_raw(count.name as *mut c_char);
                }
            }
            // the variable vec goes out of bounds and is freed too
        }
    }
    op_counts.len = 0;
    op_counts.data = std::ptr::null_mut();
}

#[cfg(feature = "python_binding")]
mod py {
    use crate::circuit::mut_ptr_as_ref;
    use pyo3::prelude::*;
    use qiskit_circuit::bit::{
        ClassicalRegister, PyClassicalRegister, PyQuantumRegister, QuantumRegister,
    };
    use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};

    /// @ingroup QkCircuit
    /// Pass ownership of a `QkCircuit` object to Python.
    ///
    /// The resulting Python object is *not* `QuantumCircuit`, it is the inner `CircuitData`, which
    /// is typically accessed as `QuantumCircuit._data`.  You can use `qk_circuit_to_python_full` to
    /// produce a complete `QuantumCircuit` object.
    ///
    /// It is not safe to use the `QkCircuit` pointer after calling this function.  In particular,
    /// you should not attempt to clear or free it.  The caller must own the `QkCircuit`, not hold a
    /// borrowed reference (for example, a `QkCircuit *` retrieved from
    /// `qk_circuit_borrow_from_python` is not owned).
    ///
    /// @param qc The owned object.
    /// @return An owned Python reference to the object.
    ///
    /// # Safety
    ///
    /// The caller must be attached to a Python interpreter.  Behavior is undefined if `circuit` is not
    /// a valid non-null pointer to an initialized and owned `QkCircuit`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn qk_circuit_to_python(
        qc: *mut CircuitData,
    ) -> *mut ::pyo3::ffi::PyObject {
        // SAFETY: per documentation, we are attached to a Python interpreter.
        let py = unsafe { Python::assume_attached() };
        // SAFETY: per documentation, `dag` points to owned and valid data.
        let qc = unsafe { Box::from_raw(mut_ptr_as_ref(qc)) };
        match Bound::new(py, PyCircuitData::from(*qc)) {
            Ok(ob) => ob.into_ptr(),
            Err(e) => {
                e.restore(py);
                ::std::ptr::null_mut()
            }
        }
    }

    /// @ingroup QkCircuit
    /// Pass ownership of a `QkCircuit` object to Python and create a complete `QuantumCircuit`.
    ///
    /// This includes additional Python-space logic to produce the complete `QuantumCircuit`, since
    /// `QkCircuit` corresponds only to the internal `QuantumCircuit._data` field.
    ///
    /// It is not safe to use the `QkCircuit` pointer after calling this function.  In particular,
    /// you should not attempt to clear or free it.  The caller must own the `QkCircuit`, not hold a
    /// borrowed reference (for example, a `QkCircuit *` retrieved from
    /// `qk_circuit_borrow_from_python` is not owned).
    ///
    /// @param qc The owned object.
    /// @return An owned Python reference to the object.
    ///
    /// # Safety
    ///
    /// The caller must be attached to a Python interpreter.  Behavior is undefined if `circuit` is not
    /// a valid non-null pointer to an initialized and owned `QkCircuit`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn qk_circuit_to_python_full(
        qc: *mut CircuitData,
    ) -> *mut ::pyo3::ffi::PyObject {
        // SAFETY: per documentation, we are attached to a Python interpreter.
        let py = unsafe { Python::assume_attached() };
        // SAFETY: per documentation, `dag` points to owned and valid data.
        let qc = unsafe { Box::from_raw(mut_ptr_as_ref(qc)) };
        match PyCircuitData::from(*qc).into_py_quantum_circuit(py) {
            Ok(ob) => ob.into_ptr(),
            Err(e) => {
                e.restore(py);
                ::std::ptr::null_mut()
            }
        }
    }

    /// @ingroup QkCircuit
    /// Retrieve a `QkCircuit` pointer from a Python object.
    ///
    /// Note that the input to this function should _not_ be `QuantumCircuit`, but the output of
    /// `QuantumCircuit._data`.  This is necessary to enforce correct reference-counting semantics.
    ///
    /// This borrows a Python reference and extracts the `QkCircuit` pointer for it, if it is of
    /// the correct type.  The returned pointer is borrowed from the `ob` pointer.  If the
    /// ``PyObject`` is not the correct type, the return value is ``NULL`` and the exception
    /// state of the Python interpreter is set.
    ///
    /// You must be attached to a Python interpreter to call this function.
    ///
    /// You can also use `qk_circuit_convert_from_python`, which is logically the exact same as this
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
    pub unsafe extern "C" fn qk_circuit_borrow_from_python(
        ob: *mut pyo3::ffi::PyObject,
    ) -> *mut CircuitData {
        // SAFETY: per documentation, we are attached to a Python interpreter, and `ob` points to a
        // valid PyObject.
        unsafe {
            crate::py::borrow_map_mut::<PyCircuitData, CircuitData>(
                Python::assume_attached(),
                ob,
                // If in the future we change `PyCircuitData` to store an `Arc<RwLock>`, look at
                // `QkObs`/`SparseObservable` for how to change this Python-space function and the
                // new ones to be added.
                |_py, qc| Ok(&mut qc.inner),
            )
        }
    }

    /// @ingroup QkCircuit
    /// Retrieve a `QkCircuit` pointer from a Python object.
    ///
    /// Note that the input to this function should _not_ be `QuantumCircuit`, but the output of
    /// `QuantumCircuit._data`.  This is necessary to enforce correct reference-counting semantics.
    ///
    /// This borrows a Python reference and extracts the `QkCircuit` pointer for it into
    /// ``address``, if it is of the correct type.  The returned pointer is borrowed from the
    /// `object` pointer.  If the `PyObject` is not the correct type, the return value is 0, the
    /// exception state of the Python interpreter is set, and `address` is unchanged.
    ///
    /// You must be attached to a Python interpreter to call this function.
    ///
    /// You can also use `qk_circuit_borrow_from_python`, which is logically the exact same as this,
    /// but with a more natural signature for direct usage.
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
    pub unsafe extern "C" fn qk_circuit_convert_from_python(
        object: *mut ::pyo3::ffi::PyObject,
        address: *mut ::std::ffi::c_void,
    ) -> ::std::ffi::c_int {
        // SAFETY: per documentation, we are attached to a Python interpreter, `object` is a valid
        // pointer to a PyObject, and `address` points to enough space to write a pointer.
        unsafe {
            crate::py::convert_map_mut::<PyCircuitData, CircuitData>(
                Python::assume_attached(),
                object,
                address,
                |_py, qc| Ok(&mut qc.inner),
            )
        }
    }

    /// @ingroup QkCircuit
    /// Pass ownership of a `QkQuantumRegister` object to Python.
    ///
    /// It is not safe to use the `QkQuantumRegister` pointer after calling this function.  In
    /// particular, you should not attempt to clear or free it.  The caller must own the
    /// `QkQuantumRegister`, not hold a borrowed reference (for example, a `QkQuantumRegister *`
    /// retrieved from `qk_quantum_register_borrow_from_python` is not owned).
    ///
    /// @param qr The owned object.
    /// @return An owned Python reference to the object.
    ///
    /// # Safety
    ///
    /// The caller must be attached to a Python interpreter.  Behavior is undefined if `qr` is not
    /// a valid non-null pointer to an initialized and owned `QkQuantumRegister`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn qk_quantum_register_to_python(
        qr: *mut QuantumRegister,
    ) -> *mut ::pyo3::ffi::PyObject {
        // SAFETY: per documentation, we are attached to a Python interpreter.
        let py = unsafe { Python::assume_attached() };
        // SAFETY: per documentation, `dag` points to owned and valid data.
        let qreg = unsafe { Box::from_raw(mut_ptr_as_ref(qr)) };
        match qreg.into_pyobject(py) {
            Ok(ob) => ob.into_ptr(),
            Err(e) => {
                e.restore(py);
                ::std::ptr::null_mut()
            }
        }
    }

    /// @ingroup QkCircuit
    /// Retrieve a `QkQuantumRegister` pointer from a Python object.
    ///
    /// This borrows a Python reference and extracts the `QkQuantumRegister` pointer for it, if it
    /// is of the correct type.  The returned pointer is borrowed from the `ob` pointer.  If the
    /// `PyObject` is not the correct type, the return value is `NULL` and the exception state of
    /// the Python interpreter is set.
    ///
    /// You must be attached to a Python interpreter to call this function.
    ///
    /// You can also use `qk_quantum_register_convert_from_python`, which is logically the exact
    /// same as this function, but can be directly used as a "converter" function for the
    /// `PyArg_Parse*` family of Python converter functions.
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
    pub unsafe extern "C" fn qk_quantum_register_borrow_from_python(
        ob: *mut pyo3::ffi::PyObject,
    ) -> *const QuantumRegister {
        // SAFETY: per documentation, we are attached to a Python interpreter, and `ob` points to a
        // valid PyObject.
        unsafe {
            crate::py::borrow_map::<PyQuantumRegister, QuantumRegister>(
                Python::assume_attached(),
                ob,
                |_py, qr| Ok(qr),
            )
        }
    }

    /// @ingroup QkCircuit
    /// Retrieve a `QkQuantumRegister` pointer from a Python object.
    ///
    /// This borrows a Python reference and extracts the `QkQuantumRegister` pointer for it into
    /// `address`, if it is of the correct type.  The returned pointer is borrowed from the
    /// `object` pointer.  If the `PyObject` is not the correct type, the return value is 0, the
    /// exception state of the Python interpreter is set, and `address` is unchanged.
    ///
    /// You must be attached to a Python interpreter to call this function.
    ///
    /// You can also use `qk_quantum_register_borrow_from_python`, which is logically the exact same
    /// as this, but with a more natural signature for direct usage.
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
    pub unsafe extern "C" fn qk_quantum_register_convert_from_python(
        object: *mut ::pyo3::ffi::PyObject,
        address: *mut ::std::ffi::c_void,
    ) -> ::std::ffi::c_int {
        // SAFETY: per documentation, we are attached to a Python interpreter, `object` is a valid
        // pointer to a PyObject, and `address` points to enough space to write a pointer.
        unsafe {
            crate::py::convert_map::<PyQuantumRegister, QuantumRegister>(
                Python::assume_attached(),
                object,
                address,
                |_py, qr| Ok(qr),
            )
        }
    }

    /// @ingroup QkCircuit
    /// Pass ownership of a `QkClassicalRegister` object to Python.
    ///
    /// It is not safe to use the `QkClassicalRegister` pointer after calling this function.  In
    /// particular, you should not attempt to clear or free it.  The caller must own the
    /// `QkClassicalRegister`, not hold a borrowed reference (for example, a `QkClassicalRegister *`
    /// retrieved from `qk_classical_register_borrow_from_python` is not owned).
    ///
    /// @param cr The owned object.
    /// @return An owned Python reference to the object.
    ///
    /// # Safety
    ///
    /// The caller must be attached to a Python interpreter.  Behavior is undefined if `cr` is not
    /// a valid non-null pointer to an initialized and owned `QkClassicalRegister`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn qk_classical_register_to_python(
        cr: *mut ClassicalRegister,
    ) -> *mut ::pyo3::ffi::PyObject {
        // SAFETY: per documentation, we are attached to a Python interpreter.
        let py = unsafe { Python::assume_attached() };
        // SAFETY: per documentation, `dag` points to owned and valid data.
        let cr = unsafe { Box::from_raw(mut_ptr_as_ref(cr)) };
        match cr.into_pyobject(py) {
            Ok(ob) => ob.into_ptr(),
            Err(e) => {
                e.restore(py);
                ::std::ptr::null_mut()
            }
        }
    }

    /// @ingroup QkCircuit
    /// Retrieve a `QkClassicalRegister` pointer from a Python object.
    ///
    /// This borrows a Python reference and extracts the `QkClassicalRegister` pointer for it, if it
    /// is of the correct type.  The returned pointer is borrowed from the `ob` pointer.  If the
    /// `PyObject` is not the correct type, the return value is `NULL` and the exception
    /// state of the Python interpreter is set.
    ///
    /// You must be attached to a Python interpreter to call this function.
    ///
    /// You can also use `qk_classical_register_convert_from_python`, which is logically the exact
    /// same as this function, but can be directly used as a "converter" function for the
    /// `PyArg_Parse*` family of Python converter functions.
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
    pub unsafe extern "C" fn qk_classical_register_borrow_from_python(
        ob: *mut pyo3::ffi::PyObject,
    ) -> *const ClassicalRegister {
        // SAFETY: per documentation, we are attached to a Python interpreter, and `ob` points to a
        // valid PyObject.
        unsafe {
            crate::py::borrow_map::<PyClassicalRegister, ClassicalRegister>(
                Python::assume_attached(),
                ob,
                |_py, cr| Ok(cr),
            )
        }
    }

    /// @ingroup QkCircuit
    /// Retrieve a `QkClassicalRegister` pointer from a Python object.
    ///
    /// This borrows a Python reference and extracts the `QkClassicalRegister` pointer for it into
    /// `address`, if it is of the correct type.  The returned pointer is borrowed from the
    /// `object` pointer.  If the `PyObject` is not the correct type, the return value is 0, the
    /// exception state of the Python interpreter is set, and `address` is unchanged.
    ///
    /// You must be attached to a Python interpreter to call this function.
    ///
    /// You can also use `qk_classical_register_borrow_from_python`, which is logically the exact same as this,
    /// but with a more natural signature for direct usage.
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
    pub unsafe extern "C" fn qk_classical_register_convert_from_python(
        object: *mut ::pyo3::ffi::PyObject,
        address: *mut ::std::ffi::c_void,
    ) -> ::std::ffi::c_int {
        // SAFETY: per documentation, we are attached to a Python interpreter, `object` is a valid
        // pointer to a PyObject, and `address` points to enough space to write a pointer.
        unsafe {
            crate::py::convert_map::<PyClassicalRegister, ClassicalRegister>(
                Python::assume_attached(),
                object,
                address,
                |_py, cr| Ok(cr),
            )
        }
    }
}
#[cfg(feature = "python_binding")]
pub use py::*;

/// @ingroup QkCircuit
///
/// Units for circuit delays.
#[repr(u8)]
pub enum CDelayUnit {
    /// Seconds.
    S = 0,
    /// Milliseconds.
    MS = 1,
    /// Microseconds.
    US = 2,
    /// Nanoseconds.
    NS = 3,
    /// Picoseconds.
    PS = 4,
}

impl From<CDelayUnit> for DelayUnit {
    fn from(value: CDelayUnit) -> Self {
        match value {
            CDelayUnit::S => DelayUnit::S,
            CDelayUnit::MS => DelayUnit::MS,
            CDelayUnit::US => DelayUnit::US,
            CDelayUnit::NS => DelayUnit::NS,
            CDelayUnit::PS => DelayUnit::PS,
        }
    }
}

/// @ingroup QkCircuit
/// Append a delay instruction to the circuit.
///
/// @param circuit A pointer to the circuit to add the delay to.
/// @param qubit The ``uint32_t`` index of the qubit to apply the delay to.
/// @param duration The duration of the delay.
/// @param unit An enum representing the unit of the duration.
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(1, 0);
///     qk_circuit_delay(qc, 0, 100.0, QkDelayUnit_NS);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_delay(
    circuit: *mut CircuitData,
    qubit: u32,
    duration: f64,
    unit: CDelayUnit,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };

    let delay_unit_variant = unit.into();

    let duration_param: Param = duration.into();
    let delay_instruction = StandardInstruction::Delay(delay_unit_variant);

    let params = Parameters::Params(smallvec![duration_param]);
    circuit
        .push_packed_operation(
            PackedOperation::from_standard_instruction(delay_instruction),
            Some(params),
            &[Qubit(qubit)],
            &[],
        )
        .unwrap();

    ExitCode::Success
}

/// The configuration options for the ``qk_circuit_draw`` function.
#[repr(C)]
pub struct CircuitDrawerConfig {
    /// If `true`, bundles classical registers into single wires.
    bundle_cregs: bool,
    /// If `true`, merges the bottom and top lines of adjacent wires.
    merge_wires: bool,
    /// Sets the line length for wrapping the rendered text. Use 0
    /// to auto-detect console width. Use `SIZE_MAX` to effectively skip
    /// wrapping altogether.
    fold: usize,
}

/// @ingroup QkCircuit
/// Draw the circuit as text.
///
/// @param circuit A pointer to the circuit to draw.
/// @param config A pointer to the ``QkCircuitDrawerConfig`` structure or NULL. If NULL,
///     the drawer will use these defaults:
///     * ``bundle_cregs = true``
///     * ``merge_wires = true``
///     * ``fold = 0``
///
/// @return A pointer to a null-terminated string containing the circuit representation.
///     You must use ``qk_str_free`` to release the allocated memory when done.
///
/// # Example
/// ```c
///     QkCircuit *circuit = qk_circuit_new(2, 1);
///
///     qk_circuit_gate(circuit, QkGate_H, (uint32_t[]){0}, NULL);
///     qk_circuit_gate(circuit, QkGate_CX, (uint32_t[]){0, 1}, NULL);
///     qk_circuit_measure(circuit, 0, 0);
///     qk_circuit_measure(circuit, 1, 0);
///
///     QkCircuitDrawerConfig config = {false, true, 0};
///
///     char *circ_str = qk_circuit_draw(circuit, &config);
///
///     printf("%s", circ_str);
///
///     qk_str_free(circ_str);
///     qk_circuit_free(circuit);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``, or
/// if ``config`` is not NULL and a non-valid pointer to a ``QkCircuitDrawerConfig`` struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_draw(
    circuit: *const CircuitData,
    config: *const CircuitDrawerConfig,
) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    let (bundle_cregs, merge_wires, fold) = if !config.is_null() {
        // SAFETY: Per documentation, the pointer is to a valid QkCircuitDrawerConfig struct.
        let config = unsafe { const_ptr_as_ref(config) };
        (
            config.bundle_cregs,
            config.merge_wires,
            if config.fold != 0 {
                Some(config.fold)
            } else {
                None
            },
        )
    } else {
        (true, true, None)
    };

    let circuit_str = draw_circuit(circuit, bundle_cregs, merge_wires, fold).unwrap();

    CString::new(circuit_str).unwrap().into_raw()
}

/// @ingroup QkCircuit
/// Convert a given circuit to a DAG.
///
/// The new DAG is copied from the circuit; the original ``circuit`` reference is still owned by the
/// caller and still required to be freed with `qk_circuit_free`.  You must free the returned DAG
/// with ``qk_dag_free`` when done with it.
///
/// @param circuit A pointer to the circuit from which to create the DAG.
///
/// @return A pointer to the new DAG.
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(0, 0);
///     QkQuantumRegister *qr = qk_quantum_register_new(3, "qr");
///     qk_circuit_add_quantum_register(qc, qr);
///     qk_quantum_register_free(qr);
///
///     QkDag *dag = qk_circuit_to_dag(qc);
///
///     qk_dag_free(dag);
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_to_dag(circuit: *const CircuitData) -> *mut DAGCircuit {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    let dag = DAGCircuit::from_circuit_data(circuit, true, None, None, None, None)
        .expect("Error occurred while converting CircuitData to DAGCircuit");

    Box::into_raw(Box::new(dag))
}

/// @ingroup QkCircuit
///
/// The mode to copy the classical variables, for operations that create a new
/// circuit based on an existing one.
#[repr(u8)]
pub enum CVarsMode {
    /// Each variable has the same type it had in the input.
    Alike = 0,
    /// Each variable becomes a "capture".
    Captures = 1,
    /// Do not copy the variable data.
    Drop = 2,
}

impl From<CVarsMode> for VarsMode {
    fn from(value: CVarsMode) -> Self {
        match value {
            CVarsMode::Alike => VarsMode::Alike,
            CVarsMode::Captures => VarsMode::Captures,
            CVarsMode::Drop => VarsMode::Drop,
        }
    }
}

/// @ingroup QkCircuit
///
/// The mode to use to copy blocks in control-flow instructions, for operations that
/// create a new circuit based on an existing one.
#[repr(u8)]
pub enum CBlocksMode {
    /// Drop the blocks.
    Drop = 0,
    /// Keep the blocks.
    Keep = 1,
}

impl From<CBlocksMode> for BlocksMode {
    fn from(value: CBlocksMode) -> Self {
        match value {
            CBlocksMode::Drop => BlocksMode::Drop,
            CBlocksMode::Keep => BlocksMode::Keep,
        }
    }
}

/// @ingroup QkCircuit
/// Return a copy of self with the same structure but empty.
///
/// That structure includes:
/// * global phase
/// * all the qubits and clbits, including the registers.
///
/// @param circuit A pointer to the circuit to copy.
/// @param vars_mode The mode for handling classical variables.
/// @param blocks_mode The mode for handling blocks.
///
/// @return The pointer to the copied circuit.
///
/// # Example
/// ```c
/// QkCircuit *qc = qk_circuit_new(10, 10);
/// for (int i = 0; i < 10; i++) {
///     qk_circuit_measure(qc, i, i);
///     uint32_t qubits[1] = {i};
///     qk_circuit_gate(qc, QkGate_H, qubits, NULL);
/// }
///
/// // As the circuit does not contain any control-flow instructions,
/// // vars_mode and blocks_mode do not have any effect.
/// QkCircuit *copy = qk_circuit_copy_empty_like(qc, QkVarsMode_Alike, QkBlocksMode_Drop);
///
/// size_t num_copy_instructions = qk_circuit_num_instructions(copy); // 0
///
/// // do something with the copy
///
/// qk_circuit_free(qc);
/// qk_circuit_free(copy);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_copy_empty_like(
    circuit: *const CircuitData,
    vars_mode: CVarsMode,
    blocks_mode: CBlocksMode,
) -> *mut CircuitData {
    // SAFETY: Per documentation, the pointer is to valid data.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let vars_mode = vars_mode.into();
    let blocks_mode = blocks_mode.into();

    let copied_circuit = circuit
        .copy_empty_like(vars_mode, blocks_mode)
        .expect("Failed to copy the circuit.");
    Box::into_raw(Box::new(copied_circuit))
}

/// @ingroup QkCircuit
/// Estimate the fidelity of a physical circuit.
///
/// This function will compute the product of the error rates for each
/// gate in the circuit to estimate the fidelity of the circuit. This method is not
/// intended to compute a realistic simulation of the fidelity of execution on real hardware. It is
/// designed to provide an estimate of how the transpiler would work with the fidelity for various
/// heuristics in its operation. It is typically only useful for comparing different compilation
/// outputs against each other to estimate which one would produce a better quality execution on
/// hardware.
///
/// @param circuit A pointer to the circuit to estimate the fidelity of.
/// @param target A pointer to the target that the circuit will be executed on. This is
///     used to get the error rates for the instructions in the circuit.
///
/// @return The computed fidelity of the circuit. This will return NaN if the circuit is not
///     physical, meaning there are instructions in `circuit` not supported by `target`.
///
/// # Safety
///
/// Behavior is undefined if `circuit` and `target` are not a valid, non-null pointer to a
/// `QkCircuit` and `QkTarget` respectively.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_estimate_fidelity(
    circuit: *const CircuitData,
    target: *const Target,
) -> f64 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    // SAFETY: Per documentation, the pointer is to valid data.
    let target = unsafe { const_ptr_as_ref(target) };
    estimate_fidelity(circuit, target).unwrap_or(f64::NAN)
}

/// @ingroup QkCircuit
/// Get a control flow instruction from a circuit at the specified index.
///
/// This function returns a pointer to an opaque ``QkControlFlowInstruction`` struct, which holds
/// information about a control flow instruction at the specified index. The instruction at
/// ``inst_idx`` must be a control flow instruction (e.g. IfElse, ForLoop etc.).
///
/// @param circuit A pointer to the circuit containing the control flow instruction.
/// @param inst_idx The index of the instruction in the circuit's instruction list.
/// @param parent_cf A pointer to the enclosing control flow instruction, or ``NULL`` if this
/// instruction is at the top level of the circuit. This is used to compute the correct qubit
/// and clbit mappings relative to the top-level circuit.
///
/// @return A pointer to a newly allocated ``QkControlFlowInstruction`` object.
///
/// # Example
/// ```c
///     QkCircuit *circuit = ...; // Assume circuit contains a control flow instruction at index 0
///     QkControlFlowInstruction *cf_inst = qk_circuit_get_control_flow_instruction(circuit, 0, NULL);
///     QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
///     qk_control_flow_instruction_free(cf_inst);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid pointer to a ``QkCircuit`` object,
/// or if ``inst_idx`` is not a valid index within the circuit's instruction list, or if the
/// instruction at ``inst_idx`` is not a control flow instruction. If ``parent_cf`` is not null,
/// it must be a valid pointer to a ``QkControlFlowInstruction``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_circuit_get_control_flow_instruction(
    circuit: *const CircuitData,
    inst_idx: usize,
    parent_cf: *const CControlFlowInstruction,
) -> *mut CControlFlowInstruction {
    // SAFETY: Per documentation, circuit is a valid pointer to a QkCircuit object.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    let inst = &circuit.data()[inst_idx];

    // Mapping is done already here, since we need the circuit context
    let (qubit_map, clbit_map) = if parent_cf.is_null() {
        (
            // No enclosing block, use the trivial mapping
            circuit
                .get_qargs(inst.qubits)
                .iter()
                .map(|q| q.index() as u32)
                .collect(),
            circuit
                .get_cargs(inst.clbits)
                .iter()
                .map(|c| c.index() as u32)
                .collect(),
        )
    } else {
        // SAFETY: Per documentation, parent_cf is a valid pointer to a QkControlFlowInstruction.
        let parent_cf = unsafe { const_ptr_as_ref(parent_cf) };
        (
            circuit
                .get_qargs(inst.qubits)
                .iter()
                .map(|q| parent_cf.qubit_map[q.index()])
                .collect(),
            circuit
                .get_cargs(inst.clbits)
                .iter()
                .map(|c| parent_cf.clbit_map[c.index()])
                .collect(),
        )
    };

    // Per documentation, we assume that this is a control flow instruction
    Box::into_raw(Box::new(CControlFlowInstruction::new(
        circuit, inst_idx, qubit_map, clbit_map,
    )))
}

/// @ingroup QkCircuit
/// Free a `QkControlFlowInstruction` object.
///
/// @param cf_inst A pointer to the control flow instruction to free.
///
/// # Example
/// ```c
///     QkCircuit *circuit = ...; // Assume circuit contains a control flow instruction at index 0
///     QkControlFlowInstruction *cf_inst = qk_circuit_get_control_flow_instruction(circuit, 0, NULL);
///     qk_control_flow_instruction_free(cf_inst);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``cf_inst`` is not a valid pointer that was returned by
/// ``qk_circuit_get_control_flow_instruction``, or if this function is called more than
/// once on the same pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_control_flow_instruction_free(cf_inst: *mut CControlFlowInstruction) {
    // SAFETY: Per documentation, cf_inst is a valid pointer to a QkControlFlowInstruction
    unsafe { drop(Box::from_raw(cf_inst)) };
}

#[cfg(test)]
mod test {
    use super::*;
    use qiskit_circuit::{
        bit::{ClassicalRegister, QuantumRegister, ShareableClbit, ShareableQubit},
        circuit_data::CircuitData,
        operations::Param,
    };
    use std::mem::MaybeUninit;

    #[test]
    fn test_register_circuit_bits() {
        let qubits = vec![ShareableQubit::new_anonymous()];
        let clbits = vec![ShareableClbit::new_anonymous()];
        let mut circuit = CircuitData::new(Some(qubits), Some(clbits), Param::Float(0.0)).unwrap();

        let qr1 = QuantumRegister::new_owning("QR1", 2);
        let _ = circuit.add_qubit(qr1.get(0).unwrap(), false);

        let mut out_bits = vec![MaybeUninit::<u32>::uninit(), MaybeUninit::<u32>::uninit()];
        unsafe {
            qk_quantum_register_circuit_bits(&qr1, &circuit, out_bits.as_mut_ptr() as *mut u32)
        };
        let out_bits = unsafe {
            out_bits
                .into_iter()
                .map(|b| b.assume_init())
                .collect::<Vec<u32>>()
        };

        assert_eq!(out_bits[0], 1); // Bit was explicitly added to the circuit
        assert_eq!(out_bits[1], u32::MAX); // Bit was not added to the circuit

        let cr1 = ClassicalRegister::new_owning("CR1", 2);
        let _ = circuit.add_clbit(cr1.get(1).unwrap(), false);

        let mut out_bits = vec![MaybeUninit::<u32>::uninit(), MaybeUninit::<u32>::uninit()];
        unsafe {
            qk_classical_register_circuit_bits(&cr1, &circuit, out_bits.as_mut_ptr() as *mut u32)
        };
        let out_bits = unsafe {
            out_bits
                .into_iter()
                .map(|b| b.assume_init())
                .collect::<Vec<u32>>()
        };

        assert_eq!(out_bits[1], 1); // Bit was explicitly added to the circuit
        assert_eq!(out_bits[0], u32::MAX); // Bit was not added to the circuit
    }
}
