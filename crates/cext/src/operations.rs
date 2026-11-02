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

use std::{
    ffi::{CStr, c_char, c_void},
    num::NonZero,
    ptr::{null, null_mut},
    sync::Arc,
};

use qiskit_circuit::{
    circuit_data::CircuitData,
    operations::{BoxedCustomOperation, CustomOperation, Operation, Param},
};

use crate::{
    dyn_types::VtableEntry,
    expose_by_arc, expose_by_box,
    pointers::{ExposesOwnedPointers, arc_clone_from_raw},
};

// SAFETY: all owned `BoxedCustomOperation` objects are exposed and freed using `Box`.
const _: () = unsafe { expose_by_box!(BoxedCustomOperation) };

/// Represents a quantum operation fully defined in C.
///
/// This operation object contains the minimal functionality an object
/// should adhere to in order operate on a ``QkCircuit``.
///
/// Any object that can be implemented using ``QkCustomOp`` will be
/// dynamically dispatched to be added to the circuit. In other words,
/// the circuit is unaware of the type of object it is accepting, but
/// it will work with it as long as it has the functionality expected
/// from any operation.
///
/// To achieve this, an operation is defined by two parts:
/// - The original pointer to the operation struct.
/// - The pointer to a vtable with the function slots that define
///   the functionality of this operation. See ``qk_custom_operation_vtable_new``
///   for more details.
///
/// Here's a quick example of what that looks like:
///
/// ```c
///
/// // Define an operation with a single attribute.
/// struct foo_gate {
///     uint32_t num_qubits;
/// }
///
/// // Represents the name of the operation.
/// const char *FOO_NAME = "foo";
///
/// // Design the required methods for the vtable.
///
/// const char *foo_name(const void *gate) {
///     // Cast void to original pointer.
///     struct foo_gate *_self = (struct foo_gate *)gate;
///     // Cast once more to consume it
///     (void)_self;
///     return FOO_NAME;
/// }
/// uint32_t foo_num_qubits(const void *gate) {
///     struct foo_gate *self = (struct foo_gate *)gate;
///     // Used stored attirbute as return value.
///     return self->num_qubits;
/// }
/// // Use same logic below for required methods that have
/// // fixed values.
/// uint32_t foo_num_clbits(const void *gate) {
///     struct foo_gate *self = (struct foo_gate *)gate;
///     (void)_self;
///     return 0;
/// }
/// // Implement all required methods.
///
/// // Build list of entries for the vtable (at least 7 required entries)
/// QkVtableEntry entries[7] = {
///     {.slot = 0, .func = foo_name},
///     {.slot = 1, .func = foo_num_qubits},
///     {.slot = 2, .func = foo_num_clbits},
///     // ...
///     // End with sentinel value
///     {.slot = -1, .func = NULL},
/// };
///
/// // Create a vtable
/// QkCustomOpVtable *foo_vtable = qk_custom_operation_vtable_new(entries);
///
/// // Declare a sample instance
/// struct foo_gate foo_3q = {
///     .num_qubits = 3,
/// };
///
/// // Create the custom operation
/// QkCustomOp foo_3q_custom = {
///     .orig = &foo_3q,
///     .v_table = foo_vtable,
/// };
///
/// // Add to a circuit
/// QkCircuit *circuit = qk_circuit_new(3, 0);
/// uint32_t qubits[3] = {0, 1, 2};
///
/// qk_circuit_custom_operation(circuit, foo_3q_custom, qubits, NULL, NULL);
/// ```
///
/// # Safety:
///
/// This struct contains raw pointers, which are not [`Send`] or [`Sync`].
///
/// It falls on the responsability of the implementors to ensure that the
/// data enclosed in the operation can:
/// - Be accessed safely by multiple threads concurrently.
/// - Be immutably borrowed by other threads without causing race conditions.
/// - Be preserved throughout the runtime of the program.
///
/// Failure to comply with these conditions may result in undefined behavior.
#[derive(Debug)]
struct CustomOp {
    /// A pointer to the original gate.
    orig: *mut c_void,
    /// A pointer to a vtable designed for the original gate.
    v_table: Arc<CustomOpVtable>,
}

impl PartialEq for CustomOp {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.v_table, &other.v_table)
            && (unsafe { (self.v_table.eq)(self.orig, other.orig) })
    }
}

impl Clone for CustomOp {
    fn clone(&self) -> Self {
        let orig = match self.v_table.clone {
            // SAFETY: per slot documentation, `clone` accepts all pointers stored in the `orig`
            // field and returns a new owning pointer.
            Some(clone) => unsafe { clone(self.orig.cast_const()) },
            None => self.orig,
        };
        Self {
            orig,
            v_table: Arc::clone(&self.v_table),
        }
    }
}
impl Drop for CustomOp {
    fn drop(&mut self) {
        // The no-op on `NULL` is documented in the slot documentation, and marks moved ownership of
        // the data pointer.
        if let Some(delete) = self.v_table.delete
            && !self.orig.is_null()
        {
            // SAFETY: per documentation of `CustomOpVtable`, if the `delete` method is set, it is
            // valid to be passed `self.orig` from any thread.
            unsafe { delete(self.orig) };
        }
    }
}
// SAFETY: per struct documentation, the data pointer is safe to send to other threads.
unsafe impl Send for CustomOp {}
// SAFETY: per struct documentation, the data pointer is safe to share with other threads.
unsafe impl Sync for CustomOp {}

impl Operation for CustomOp {
    fn name(&self) -> &str {
        let name = unsafe { (self.v_table.name)(self.orig) };
        // Safety violation on lifetimes of the name here
        // Document the lifetime bounds here, these pointers must only be borrowed.
        // C should not mutate origin while Rust is accessing it.
        let name_parsed = unsafe { CStr::from_ptr(name) };
        name_parsed.to_str().unwrap_or_default()
    }

    fn num_qubits(&self) -> u32 {
        unsafe { (({ &*self.v_table }).num_qubits)(self.orig) }
    }

    fn num_clbits(&self) -> u32 {
        unsafe { (({ &*self.v_table }).num_clbits)(self.orig) }
    }

    fn num_params(&self) -> u32 {
        unsafe { (({ &*self.v_table }).num_params)(self.orig) }
    }

    fn directive(&self) -> bool {
        unsafe { (({ &*self.v_table }).directive)(self.orig) }
    }
}

impl CustomOperation for CustomOp {
    fn is_unitary(&self) -> bool {
        unsafe { (({ &*self.v_table }).is_unitary)(self.orig) }
    }

    fn num_ctrl_qubits(&self) -> Option<std::num::NonZero<u32>> {
        let num_ctrl_qubits = unsafe { (({ &*self.v_table }).num_ctrl_qubits)(self.orig) };
        NonZero::new(num_ctrl_qubits)
    }

    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        let params: Vec<*const Param> = params.iter().map(|obj| obj as *const Param).collect();
        let definition = unsafe { (({ &*self.v_table }).definition)(self.orig, params.as_ptr()) };
        if definition.is_null() {
            return None;
        }
        let circ = unsafe { CircuitData::steal(definition) };
        Some(*circ)
    }

    fn label(&self) -> Option<&str> {
        let label = unsafe { (({ &*self.v_table }).label)(self.orig) };
        if label.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(label) }.to_str().ok()
        }
    }
}

/// Represents a vtable containing all the function pointers
/// pertaining to the methods associated with the instance of
/// [``CustomOp``] coming from C.
///
/// All methods provided require a void pointer representing
/// the original instance to be passed as an argument, which
/// is always packed together with the vtable in [`CustomOp`].
/// An implementor is expected to provide the pointers to
/// the following required methods for implementing the [`CustomOperation`]
/// trait:
///
/// * ``name(*const c_void)`` -> ``*const c_char``,
/// * ``num_qubits(*const c_void)`` -> ``u32``,
///
/// There are also functional methods that are optional but
/// implementors are expected to provide.
///
/// * ``num_clbits(*const c_void)`` -> ``u32``,
/// * ``num_params(*const c_void)`` -> ``u32``,
/// * ``directive(*const c_void)`` -> ``bool``,
/// * ``is_unitary(*const c_void)`` -> ``bool``,
/// * ``num_ctrl_qubits(*const c_void)`` -> ``u32``,
/// * ``label(*const c_void)`` ->  ``*const c_char``,
/// * ``definition(*const c_void, *const *const Param)`` -> ``*mut CircuitData``,
/// * ``eq(*const c_void, *const c_void)`` -> ``bool``, to compare two operations of the same kind.
///
/// In any custom operation, the data pointer is _owned_.  If your data pointer involves an
/// allocation with memory-management requirements, or includes mutable data, you most likely need
/// to implement the cloning and destruction behavior via the `Clone` and `Delete` slots.
///
/// `Clone` must accept the data pointer, and return a new owning allocation of the same type.
/// `Delete` must accept the data pointer and free all memory or resources used.  `Delete` is not
/// called if the data pointer is null.
#[derive(Debug, Clone)]
pub struct CustomOpVtable {
    clone: Option<unsafe extern "C" fn(*const c_void) -> *mut c_void>,
    delete: Option<unsafe extern "C" fn(*mut c_void)>,
    name: unsafe extern "C" fn(*const c_void) -> *const c_char,
    num_qubits: unsafe extern "C" fn(*const c_void) -> u32,
    num_clbits: unsafe extern "C" fn(*const c_void) -> u32,
    num_params: unsafe extern "C" fn(*const c_void) -> u32,
    directive: unsafe extern "C" fn(*const c_void) -> bool,
    is_unitary: unsafe extern "C" fn(*const c_void) -> bool,
    num_ctrl_qubits: unsafe extern "C" fn(*const c_void) -> u32,
    label: unsafe extern "C" fn(*const c_void) -> *const c_char,
    definition: unsafe extern "C" fn(*const c_void, *const *const Param) -> *mut CircuitData,
    eq: unsafe extern "C" fn(*const c_void, *const c_void) -> bool,
}

// SAFETY: all owned `CustomOpVtable` objects are exposed and freed using `Arc`.
const _: () = unsafe { expose_by_arc!(CustomOpVtable) };

extern "C" fn default_num_clbits(_op: *const c_void) -> u32 {
    0
}
extern "C" fn default_num_params(_op: *const c_void) -> u32 {
    0
}
extern "C" fn default_directive(_op: *const c_void) -> bool {
    false
}
extern "C" fn default_is_unitary(_op: *const c_void) -> bool {
    true
}
extern "C" fn default_num_ctrl_qubits(_op: *const c_void) -> u32 {
    0
}

extern "C" fn default_label(_op: *const c_void) -> *const c_char {
    null()
}

extern "C" fn default_definition(
    _op: *const c_void,
    _params: *const *const Param,
) -> *mut CircuitData {
    null_mut()
}

extern "C" fn default_eq(slf: *const c_void, other: *const c_void) -> bool {
    slf.eq(&other)
}

impl TryFrom<CustomOpVtablePartial> for CustomOpVtable {
    type Error = CustomOpSlot;

    fn try_from(value: CustomOpVtablePartial) -> Result<Self, Self::Error> {
        use CustomOpSlot::*;
        Ok(Self {
            name: value.name.ok_or(Name)?,
            num_qubits: value.num_qubits.ok_or(NumQubits)?,
            num_clbits: value.num_clbits.unwrap_or(default_num_clbits),
            num_params: value.num_params.unwrap_or(default_num_params),
            directive: value.directive.unwrap_or(default_directive),
            is_unitary: value.is_unitary.unwrap_or(default_is_unitary),
            num_ctrl_qubits: value.num_ctrl_qubits.unwrap_or(default_num_ctrl_qubits),
            label: value.label.unwrap_or(default_label),
            definition: value.definition.unwrap_or(default_definition),
            eq: value.eq.unwrap_or(default_eq),
            clone: value.clone,
            delete: value.delete,
        })
    }
}

/// A partial implementation of [`CustomOpVtable`] as referred
/// to by its namesake.
///
/// This implementation is only used during construction of the
/// vtable and should always be converted to a [`CustomOpVtable`].
/// The conversion will fail if any of the required methods listed
/// in the documentation are not provided, and an error code with
/// the first missing slot's [``CustomOpSlot``] index will be provided.
#[derive(Debug, Clone, Default)]
pub struct CustomOpVtablePartial {
    name: Option<unsafe extern "C" fn(*const c_void) -> *const c_char>,
    clone: Option<unsafe extern "C" fn(*const c_void) -> *mut c_void>,
    delete: Option<unsafe extern "C" fn(*mut c_void)>,
    num_qubits: Option<unsafe extern "C" fn(*const c_void) -> u32>,
    num_clbits: Option<unsafe extern "C" fn(*const c_void) -> u32>,
    num_params: Option<unsafe extern "C" fn(*const c_void) -> u32>,
    directive: Option<unsafe extern "C" fn(*const c_void) -> bool>,
    is_unitary: Option<unsafe extern "C" fn(*const c_void) -> bool>,
    num_ctrl_qubits: Option<unsafe extern "C" fn(*const c_void) -> u32>,
    label: Option<unsafe extern "C" fn(*const c_void) -> *const c_char>,
    definition:
        Option<unsafe extern "C" fn(*const c_void, *const *const Param) -> *mut CircuitData>,
    eq: Option<unsafe extern "C" fn(*const c_void, *const c_void) -> bool>,
}

impl CustomOpVtablePartial {
    unsafe fn set(&mut self, slot: CustomOpSlot, ptr: *const c_void) -> bool {
        match slot {
            CustomOpSlot::Name => {
                let ptr = unsafe {
                    std::mem::transmute::<
                        *const c_void,
                        unsafe extern "C" fn(*const c_void) -> *const c_char,
                    >(ptr)
                };
                self.name.replace(ptr).is_some()
            }
            CustomOpSlot::Clone => {
                let ptr = unsafe {
                    std::mem::transmute::<
                        *const c_void,
                        unsafe extern "C" fn(*const c_void) -> *mut c_void,
                    >(ptr)
                };
                self.clone.replace(ptr).is_some()
            }
            CustomOpSlot::Delete => {
                let ptr = unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*mut c_void)>(ptr)
                };
                self.delete.replace(ptr).is_some()
            }
            CustomOpSlot::NumQubits => {
                let ptr = unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const c_void) -> u32>(
                        ptr,
                    )
                };
                self.num_qubits.replace(ptr).is_some()
            }
            CustomOpSlot::NumClbits => {
                let ptr = unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const c_void) -> u32>(
                        ptr,
                    )
                };
                self.num_clbits.replace(ptr).is_some()
            }
            CustomOpSlot::NumParams => {
                let ptr = unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const c_void) -> u32>(
                        ptr,
                    )
                };
                self.num_params.replace(ptr).is_some()
            }
            CustomOpSlot::Directive => {
                let ptr = unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const c_void) -> bool>(
                        ptr,
                    )
                };
                self.directive.replace(ptr).is_some()
            }
            CustomOpSlot::IsUnitary => {
                let ptr = unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const c_void) -> bool>(
                        ptr,
                    )
                };
                self.is_unitary.replace(ptr).is_some()
            }
            CustomOpSlot::NumCtrlQubits => {
                let ptr = unsafe {
                    std::mem::transmute::<*const c_void, unsafe extern "C" fn(*const c_void) -> u32>(
                        ptr,
                    )
                };
                self.num_ctrl_qubits.replace(ptr).is_some()
            }
            CustomOpSlot::Label => {
                let ptr = unsafe {
                    std::mem::transmute::<
                        *const c_void,
                        unsafe extern "C" fn(*const c_void) -> *const c_char,
                    >(ptr)
                };
                self.label.replace(ptr).is_some()
            }
            CustomOpSlot::Definition => {
                let ptr = unsafe {
                    std::mem::transmute::<
                        *const c_void,
                        unsafe extern "C" fn(
                            *const c_void,
                            *const *const Param,
                        ) -> *mut CircuitData,
                    >(ptr)
                };
                self.definition.replace(ptr).is_some()
            }
            CustomOpSlot::Eq => {
                let ptr = unsafe {
                    std::mem::transmute::<
                        *const c_void,
                        unsafe extern "C" fn(*const c_void, *const c_void) -> bool,
                    >(ptr)
                };
                self.eq.replace(ptr).is_some()
            }
        }
    }
}

/// Represents the Vtable index of a ``QkCustomOp`` coming from the
/// C domain.
///
/// Each named index refers to a required/optional method of the `Operation`
/// and `CustomOperation` traits in Rust.
#[repr(u32)]
#[derive(Debug, Clone, Copy, derive_more::TryFrom)]
#[try_from(repr)]
pub enum CustomOpSlot {
    Name = 0,
    NumQubits = 1,
    NumClbits = 2,
    NumParams = 3,
    Directive = 4,
    IsUnitary = 5,
    NumCtrlQubits = 6,
    Label = 7,
    Definition = 8,
    Eq = 9,
    Clone = 10,
    Delete = 11,
}

/// @ingroup QkCustomOp
/// Builds a ``QkCustomOp`` based on a quantum operation fully
/// defined in C.
///
/// Here's a quick example of what that looks like:
///
/// ```c
///
/// // Define an operation with a single attribute.
/// struct foo_gate {
///     uint32_t num_qubits;
/// }
///
/// // Implement all required methods
/// uint32_t foo_num_qubits(const void *gate) {
///     struct foo_gate *self = (struct foo_gate *)gate;
///     // Used stored attirbute as return value.
///     return self->num_qubits;
/// }
///
/// // Build list of entries for the vtable (at least 7 required entries)
/// QkVtableEntry entries[7] = {
///     {.slot = QkCustomOpSlot_NumQubits, .func = foo_num_qubits},
///     // ...
///     // End with sentinel value
///     {.slot = -1, .func = NULL},
/// };
///
/// // Create a vtable
/// QkCustomOpVtable *foo_vtable = qk_custom_operation_vtable_new(entries);
///
/// // Declare a sample instance
/// struct foo_gate foo_3q = {
///     .num_qubits = 3,
/// };
///
/// // Create the custom operation
/// QkCustomOp foo_3q_custom = qk_custom_operation_new(&foo_3q, foo_vtable);
/// ```
///
/// @param operation An owned pointer to the operation struct.
/// @param v_table A pointer to a correctly constructed v_table designed to
/// work with the data of the struct `operation` points to.
///
/// @return A pointer to ``QkCustomOp``.
///
/// # Safety
///
/// It falls on the responsibility of the implementors to ensure that the
/// data enclosed in the `operation` struct can:
/// - Be accessed safely by multiple threads concurrently.
/// - Be immutably borrowed by other threads without causing race conditions.
/// - Be preserved throughout the lifetime of the operation.
///
/// Behavior is undefined if the provided `v_table` pointer is null or non-aligned.
///
/// Failure to comply with these conditions may result in undefined behavior.
///
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_new(
    operation: *mut c_void,
    v_table: *const CustomOpVtable,
) -> *mut BoxedCustomOperation {
    let as_custom_op = CustomOp {
        orig: operation,
        // SAFETY: as established by the documentation this pointer must be non-null
        // and aligned.
        v_table: unsafe { arc_clone_from_raw(v_table) },
    };

    BoxedCustomOperation::from(as_custom_op).into_leaked()
}

/// @ingroup QkCustomOp
/// Free an owned :c:type:`QkCustomOp`.
///
/// You typically do not need to call this, because :c:func:`qk_circuit_custom_operation` takes
/// ownership of the object.
///
/// This function is a no-op if given `NULL`.
///
/// @param data An owning pointer to data to free.
///
/// # Safety
///
/// Behavior is undefined if `data` is not null or a valid owning pointer to a :c:type:`QkCustomOp`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_free(data: *mut BoxedCustomOperation) {
    // SAFETY: per documentation, `data` is either null or a valid owning pointer.
    _ = (!data.is_null()).then(|| unsafe { BoxedCustomOperation::steal(data) });
}

/// @ingroup QkCustomOp
/// Builds a ``QkCustomOpVtable`` based on a list of ``QkVtableEntry``
/// instances.
///
/// The vtable is built from a collection of slots that hold an index and a
/// pointer to a function of the correct argument and return types.
///
/// Refer to the following table to identify the correct slots.
///
/// | Slot                 | Arg(s) type                    | Return type   |               Index                  | Required |       Default       |
/// |----------------------|--------------------------------|---------------|--------------------------------------|----------|---------------------|
/// | ``name``             | `const void *`                 | `char *`      | ``QkCustomOpSlot_Name``            |    Yes   |        n/a          |
/// | ``num_qubits``       | `const void *`                 | `uint32_t`    | ``QkCustomOpSlot_NumQubits``       |    Yes   |        n/a          |
/// | ``num_clbits``       | `const void *`                 | `uint32_t`    | ``QkCustomOpSlot_NumClbits``       |    No    |       `0`           |
/// | ``num_params``       | `const void *`                 | `uint32_t`    | ``QkCustomOpSlot_NumParams``       |    No    |       `0`           |
/// | ``directive``        | `const void *`                 | `bool`        | ``QkCustomOpSlot_Directive``       |    No    |      `false`        |
/// | ``is_unitary``       | `const void *`                 | `bool`        | ``QkCustomOpSlot_IsUnitary``       |    No    |      `true`         |
/// | ``num_ctrl_qubits``  | `const void *`                 | `uint32_t`    | ``QkCustomOpSlot_NumCtrlQubits``   |    No    |       `0`           |
/// | ``label``            | `const void *`                 | `char *`      | ``QkCustomOpSlot_Label``           |    No    |       `NULL`        |
/// | ``definition``       | `const void *`, `QkParam **`   | `QkCircuit *` | ``QkCustomOpSlot_Definition``      |    No    |       `NULL`        |
/// | ``eq``               | `const void *`, `const void *` | `bool`        | ``QkCustomOpSlot_Eq``              |    No    | Pointer comparison  |
/// | ``clone``            | `const void *`                 | `void *`      | ``QkCustomOpSlot_Clone``           |    No    |     pointer alias   |
/// | ``delete``           | `void *`                       | `void`        | ``QkCustomOpSlot_Delete``          |    No    |       no-op         |
///
/// Each function will be seen as a `void` pointer to Rust and will be transmuted
/// to a function pointer of the correct signature.
///
/// If a required slot is not received, the vtable will not be constructed
/// and this function will return a `NULL` pointer. If an optional slot is not
/// included, the vtable will still be built and its slots will point to default
/// implementations of the said method(s).
///
/// If a slot does not have a valid index (other than the sentinel value), the provided
/// function pointer will be ignored. This ensures that if any non-required methods are
/// added or removed from the chart above, the program should still be able to run
/// without issues.
///
/// Every list of slots should be delimited by a sentinel valued
/// ``QkVtableEntry`` at the end. The sentinel should look as follows:
///
/// ```c
/// QkVtableEntry sentinel = {.slot = -1, .func = NULL};
/// ```
///
/// This function will stop reading any slots located after the sentinel is found.
///
/// @param slots A pointer to a list of entries delimited by an entry with
/// a sentinel value.
///
/// @return A pointer to a constructed vtable or a null pointer if any
/// required entries are absent.
///
/// # Safety
///
/// Behavior is undefined if a list of entries without delimiting sentinel
/// value are provided.
///
/// Undefined behavior may also happen during transmutation if the provided
/// function pointer does not have the correct signature.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_vtable_new(
    mut slots: *const VtableEntry,
) -> *const CustomOpVtable {
    let mut vtable = CustomOpVtablePartial::default();
    loop {
        // SAFETY: per documentation, `slots` is valid for reads until we see the sentinel all-ones
        // pattern in a `slot`.
        let entry = unsafe { slots.read() };
        slots = if entry.slot == u32::MAX {
            break;
        } else {
            slots.wrapping_add(1)
        };
        let Ok(slot) = CustomOpSlot::try_from(entry.slot) else {
            // We assume this is a slot from a later version of Qiskit.
            // TODO: add an envvar / global to turn on debug information in these cases?
            continue;
        };
        // SAFETY: per documentation, `entry.ptr` is of the expected function-pointer type and valid
        // to call, because `entry.slot` was not all-ones.
        if unsafe { vtable.set(slot, entry.ptr) } {
            // This a documented UB case.
            return std::ptr::dangling_mut();
        }
    }

    match CustomOpVtable::try_from(vtable) {
        Ok(full_table) => full_table.into_leaked(),
        Err(_missing_slot) => std::ptr::null_mut(),
    }
}

/// @ingroup QkCustomOp
/// Frees the `QkCustomOpVtable` pointer
///
/// @param v_table The pointer to a `QkCustomOpVtable` object.
///
/// # Safety
///
/// Undefined behavior may occur if `v_table` is a `NULL` or unaligned invalid pointer
/// to a `QkCustomOpVtable`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_custom_operation_vtable_free(v_table: *const CustomOpVtable) {
    // SAFETY: if `v_table` is not nul, then it is an owned pointer as per documentation
    // all owned pointers can be given to `steal`.
    _ = (!v_table.is_null()).then(|| unsafe { CustomOpVtable::steal(v_table.cast_mut()) })
}
