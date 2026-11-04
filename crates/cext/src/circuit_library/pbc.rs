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

use qiskit_circuit::operations::Param;

/// A representation of Pauli product rotation data.
///
/// A Pauli product rotation implements a rotation of an ``angle`` about an axis defined
/// by a Pauli product on ``len`` qubits. The Pauli here is represented in ZX format with
/// two Boolean arrays representing the Z and X components.
#[repr(C)]
pub struct CPauliProductRotation {
    /// Pointer to an length-`len` array of Z components.
    pub z: *mut bool,
    /// Pointer to an length-`len` array of X components.
    pub x: *mut bool,
    /// The number of Pauli terms.
    pub len: usize,
    /// The rotation angle.
    pub angle: *mut Param,
}

/// @ingroup QkCircuitLibrary
/// Clear the internal data of Rust-allocated ``QkPauliProductRotation``.
///
/// This frees the memory of the ``z`` and ``x`` arrays and frees the ``angle``. This function
/// should only be called for ``QkPauliProductRotation`` objects whose data has been populated by Rust.
///
/// # Example
///
/// ```c
/// // let `circuit` be a QkCircuit* and `index` a size_t at the position of a QkPauliProductRotation
/// QkPauliProductRotation inst;
///
/// // query the QkPauliProductRotation data
/// assert(qk_circuit_operation_kind(circuit, index) == QkOperationKind_PauliProductRotation);
/// qk_circuit_inst_pauli_product_rotation(circuit, index, &inst);
///
/// // do something with `inst`, and then clear the Rust-allocated data
/// qk_pauli_product_rotation_clear(&inst);
/// ```
///
/// In contrast, this function should not be called if C already takes care of clearing the data.
/// ```c
/// bool z[4] = {false, false, true, true};
/// bool x[4] = {false, true, true, false};
/// QkParam *angle = qk_param_from_double(1.0);
/// QkPauliProductRotation rotation = {z, x, 4, angle};
///
/// // since this data is allocated by C, we do not call `qk_pauli_product_rotation_clear(&rotation)`!
/// qk_param_free(angle);
/// ```
///
/// @param inst A pointer to the ``QkPauliProductRotation`` to clear.
///
/// # Safety
///
/// Behavior is undefined if ``inst`` is not a valid, non-null pointer to a ``QkPauliProductRotation``,
/// or if the internal data of ``QkPauliProductRotation`` is incoherent.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pauli_product_rotation_clear(inst: *mut CPauliProductRotation) {
    // SAFETY: The user guarantees `inst` is a valid, non-null pointer to a [CPauliProductRotation].
    let inst = unsafe { &mut *inst };

    // SAFETY: The user guarantees the instruction is coherent, i.e. the Z and X arrays are
    // readable for the correct length and `angle` is a valid, non-null Param pointer.
    // Note that every single line here is unsafe, hence we wrap it in a single block.
    unsafe {
        let x = ::std::slice::from_raw_parts_mut(inst.x, inst.len);
        let _: Box<[bool]> = Box::from_raw(x);

        let z = ::std::slice::from_raw_parts_mut(inst.z, inst.len);
        let _: Box<[bool]> = Box::from_raw(z);

        let _ = Box::from_raw(inst.angle);
    }

    inst.x = ::std::ptr::null_mut();
    inst.z = ::std::ptr::null_mut();
    inst.len = 0;
    inst.angle = ::std::ptr::null_mut();
}

/// A representation of Pauli product measurement data.
///
/// A Pauli product measurement implements a projection onto the eigenspace of the defined
/// Pauli product on ``len`` qubits. The Pauli here is represented in ZX format with
/// two Boolean arrays representing the Z and X components and can include a minus sign, indicated
/// by ``flip_outcome``.
#[repr(C)]
pub struct CPauliProductMeasurement {
    /// Pointer to an length-`len` array of Z components.
    pub z: *mut bool,
    /// Pointer to an length-`len` array of X components.
    pub x: *mut bool,
    /// The number of Pauli terms.
    pub len: usize,
    /// Whether the measurement outcome has a minus sign.
    pub flip_outcome: bool,
}

/// @ingroup QkCircuitLibrary
/// Clear the internal data of Rust-allocated ``QkPauliProductMeasurement``.
///
/// This frees the memory of the ``z`` and ``x`` arrays. This function should only be called for
/// ``QkPauliProductMeasurement`` objects whose data has been populated by Rust.
///
/// # Example
///
/// ```c
/// // let `circuit` be a QkCircuit* and `index` a size_t at the position
/// // of a QkPauliProductMeasurement
/// QkPauliProductMeasurement inst;
///
/// // query the QkPauliProductMeasurement data
/// assert(qk_circuit_operation_kind(circuit, index) == QkOperationKind_PauliProductMeasurement);
/// qk_circuit_inst_pauli_product_measurement(circuit, index, &inst);
///
/// // do something with `inst`, and then clear the Rust-allocated data
/// qk_pauli_product_measurement_clear(&inst);
/// ```
///
/// In contrast, this function should not be called if C already takes care of clearing the data.
/// ```c
/// bool z[4] = {false, false, true, true};
/// bool x[4] = {false, true, true, false};
/// QkPauliProductMeasurement inst = {z, x, 4, true};
///
/// // since this data is allocated by C, we do not call `qk_pauli_product_measurement_clear(&inst)`
/// ```
///
/// @param inst A pointer to the ``QkPauliProductMeasurement`` to clear.
///
/// # Safety
///
/// Behavior is undefined if ``inst`` is not a valid, non-null pointer to a
/// ``QkPauliProductMeasurement``, or if the internal data of ``QkPauliProductMeasurement`` is
/// incoherent.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pauli_product_measurement_clear(inst: *mut CPauliProductMeasurement) {
    // SAFETY: The user guarantees `inst` is a valid, non-null pointer to a [CPauliProductMeasurement].
    let inst = unsafe { &mut *inst };

    // SAFETY: The user guarantees the instruction is coherent, i.e. the Z and X arrays are
    // readable for the correct length and `angle` is a valid, non-null Param pointer.
    // Note that every single line here is unsafe, hence we wrap it in a single block.
    unsafe {
        let x = ::std::slice::from_raw_parts_mut(inst.x, inst.len);
        let _: Box<[bool]> = Box::from_raw(x);

        let z = ::std::slice::from_raw_parts_mut(inst.z, inst.len);
        let _: Box<[bool]> = Box::from_raw(z);
    }

    inst.x = ::std::ptr::null_mut();
    inst.z = ::std::ptr::null_mut();
    inst.len = 0;
    inst.flip_outcome = false;
}
