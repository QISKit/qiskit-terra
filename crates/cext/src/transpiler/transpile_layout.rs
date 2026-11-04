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

use crate::pointers::{ExposesOwnedPointers, expose_by_box};
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::nlayout::{NLayout, PhysicalQubit};
use qiskit_transpiler::target::Target;
use qiskit_transpiler::transpile_layout::TranspileLayout;

#[cfg(feature = "python_binding")]
use pyo3::Python;
#[cfg(feature = "python_binding")]
use pyo3::ffi::PyObject;
#[cfg(feature = "python_binding")]
use qiskit_circuit::circuit_data::CircuitData;

// SAFETY: all owned `TranspileLayout` objects are exposed and freed using `Box`.
const _: () = unsafe { expose_by_box!(TranspileLayout) };

/// @ingroup QkTranspileLayout
/// Return the number of qubits in the input circuit to the transpiler.
///
/// @param layout A pointer to the ``QkTranspileLayout``.
///
/// @return The number of input qubits
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkTranspileLayout``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_layout_num_input_qubits(
    layout: *const TranspileLayout,
) -> u32 {
    let layout = unsafe { &*layout };
    layout.num_input_qubits()
}

/// @ingroup QkTranspileLayout
/// Return the number of qubits in the output circuit from the transpiler.
///
/// @param layout A pointer to the ``QkTranspileLayout``.
///
/// @return The number of output qubits
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkTranspileLayout``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_layout_num_output_qubits(
    layout: *const TranspileLayout,
) -> u32 {
    let layout = unsafe { &*layout };
    layout.num_output_qubits()
}

/// @ingroup QkTranspileLayout
/// Query the initial layout of a ``QkTranspileLayout``.
///
/// The output array from this function represents the mapping from the virtual qubits in the
/// original input circuit to the physical qubit in the output circuit. The
/// index in the array is the virtual qubit and the value is the physical qubit. For example an
/// output array of:
///
/// ```
/// [1, 0, 2]
/// ```
///
/// indicates that the layout maps virtual qubit 0 -> physical qubit 1, virtual qubit 1 -> physical
/// qubit -> 0, and virtual qubit 2 -> physical qubit 2.
///
/// @param layout A pointer to the ``QkTranspileLayout``.
/// @param filter_ancillas If set to true the output array will not include any indices for any
/// ancillas added by the transpiler.
/// @param initial_layout A pointer to the array where this function will write the initial layout
/// to. This must have sufficient space for the full array which will either be
/// ``qk_transpile_layout_num_input_qubits()`` or ``qk_transpile_layout_num_output_qubits()`` for
/// ``filter_ancillas`` being true or false respectively.
///
/// @returns True if there was a initial_layout written to ``initial_layout`` and false if there
/// is no initial layout.
///
/// # Safety
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkTranspileLayout``. ``initial_layout`` must be a valid, non-null pointer with a large enough
/// allocation to store the size necessary for the initial layout. If ``filter_ancillas`` is true
/// this will be number of input qubits (which can be checked with
/// ``qk_transpile_layout_num_input_qubits()``) or the number of output qubits if ``filter_ancillas``
/// is false (which can be queried with ``qk_transpile_layout_num_output_qubits()``).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_layout_initial_layout(
    layout: *const TranspileLayout,
    filter_ancillas: bool,
    initial_layout: *mut u32,
) -> bool {
    // SAFETY: Per the documentation layout must be a valid pointer to a TranspileLayout
    let layout = unsafe { &*layout };
    let out_initial_layout = layout.initial_physical_layout(filter_ancillas);
    if let Some(out_initial_layout) = out_initial_layout {
        // SAFETY: Per the documentation initial_layout must be a valid pointer with a sufficient
        // allocation for the output array
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(initial_layout, out_initial_layout.len()) };
        out_slice
            .iter_mut()
            .zip(out_initial_layout.iter())
            .for_each(|(dest, src)| *dest = src.0);
        true
    } else {
        false
    }
}

/// @ingroup QkTranspileLayout
/// Query the output permutation of a ``QkTranspileLayout``
///
/// The output array from this function represents the permutation induced by the transpiler where
/// the index indicates the qubit at the start of the circuit and the value is the position of the
/// qubit at the end of the circuit. For example an output array of:
///
/// ```
/// [1, 2, 0]
/// ```
///
/// indicates that qubit 0 from the start of the circuit is at qubit 1 at the end of the circuit,
/// 1 -> 2, and 2 -> 0.
///
/// @param layout A pointer to the ``QkTranspileLayout``.
/// @param output_permutation A pointer to the array where this function will write the output permutation
/// to. This must have sufficient space for the output which will be the number of output qubits in
/// the layout. This can be queried with ``qk_transpile_layout_num_output_qubits``.
///
/// @returns True if there is an output permutation that was written to ``output_permutation``
/// false if the ``QkTranspileLayout`` does not contain an output permutation.
///
/// # Safety
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkTranspileLayout``. ``output_permutation`` must be a valid, non-null pointer with a large enough
/// allocation to store the size necessary for the output_permutation. This will always be the number
/// of output qubits in the ``QkTranspileLayout`` which can be queried with
/// ``qk_transpile_layout_num_output_qubits()``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_layout_output_permutation(
    layout: *const TranspileLayout,
    output_permutation: *mut u32,
) -> bool {
    // SAFETY: Per the documentation layout must be a valid pointer to a TranspileLayout
    let layout = unsafe { &*layout };
    let permutation = layout.output_permutation();
    if let Some(permutation) = permutation {
        // SAFETY: Per the documentation output_permutation must be a valid pointer with a sufficient
        // allocation for the output array
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(output_permutation, permutation.len()) };
        out_slice
            .iter_mut()
            .zip(permutation.iter())
            .for_each(|(dest, src)| *dest = src.0);
        true
    } else {
        false
    }
}

/// @ingroup QkTranspileLayout
/// Query the final layout of a ``QkTranspileLayout``
///
/// The output array represents the mapping from the virtual qubit in the original input circuit to
/// the physical qubit at the end of the transpile circuit that has that qubit's state. The array
/// index represents the virtual qubit and the value represents the physical qubit at the end of
/// the transpiled circuit which has that virtual qubit's state. For example, an output array of:
///
/// ```
/// [2, 0, 1]
/// ```
///
/// indicates that virtual qubit 0's state in the original circuit is on
/// physical qubit 2 at the end of the transpiled circuit, 1 -> 0, and 2 -> 1.
///
/// @param layout A pointer to the ``QkTranspileLayout``.
/// @param filter_ancillas If set to true the output array will not include any indices for any
/// ancillas added by the transpiler.
/// @param final_layout A pointer to the array where this function will write the final layout to.
/// This must have sufficient space for the output which will either be the number of input or
/// output qubits depending on the value of filter_ancillas.
///
/// # Safety
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkTranspileLayout``. ``final_layout`` must be a valid, non-null pointer with a large enough
/// allocation to store the size necessary for the final layout. If ``filter_ancillas`` is true
/// this will be number of input qubits (which can be checked with
/// ``qk_transpile_layout_num_input_qubits()``) or the number of output qubits if ``filter_ancillas``
/// is false (which can be queried with ``qk_transpile_layout_num_output_qubits()``).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_layout_final_layout(
    layout: *const TranspileLayout,
    filter_ancillas: bool,
    final_layout: *mut u32,
) {
    // SAFETY: Per the documentation layout must be a valid pointer to a TranspileLayout
    let layout = unsafe { &*layout };
    let result = layout.final_index_layout(filter_ancillas);
    // SAFETY: Per the documentation final_layout must be a valid pointer with a sufficient
    // allocation for the output array
    let out_slice = unsafe { std::slice::from_raw_parts_mut(final_layout, result.len()) };
    out_slice
        .iter_mut()
        .zip(result.iter())
        .for_each(|(dest, src)| *dest = src.0);
}

/// @ingroup QkTranspileLayout
/// Generate a ``QkTranspileLayout`` from a initial layout mapping
///
/// This will generate a ``QkTranspileLayout`` with the initial layout set (and no ouptput
/// permutation) from a provided mapping. The intent of this function is to enable creating
/// a custom layout pass that also creates a ``QkTranspileLayout`` that you can use with
/// subsequent stage functions such as ``qk_transpile_stage_routing``.
///
/// @param original_dag: A pointer to the original dag prior to running a custom layout pass. This
///     dag must have fewer than or the same number of qubits as ``target``.
/// @param target: A pointer to the target that layout was run on. This target must have fixed
///     number of qubits set.
/// @param qubit_mapping: A pointer to the layout mapping array. This array must have the same
///     number of elements as there are qubits in target and each element is a unique integer and
///     the all must fall in the range of 0 to ``num_qubits`` where ``num_qubits`` is the
///     number of qubits indicated in the provided value for ``target``.
///     The first elements represent the virtual qubits in ``original_dag`` and the value
///     represents the physical qubit in the target which the virtual qubit is mapped too.
///     For example an array of ``[1, 0, 2]`` would map virtual qubit 0 -> physical qubit 1,
///     virtual qubit 1 -> physical qubit 0, and virtual qubit 2 -> physical qubit 2. For elements
///     that are not in the original dag these are treated as ancilla qubits, but still must be
///     mapped to a physical qubit. This array will be copied into the output ``QkTranspileLayout``
///     so you must still free it after calling this function.
///
/// @returns The QkTranspileLayout object with the initial layout set
///
/// # Safety
/// Behavior is undefined if ``original_dag`` and target ``target`` are not a valid, aligned,
/// non-null pointer to a ``QkDag`` or a ``QkTarget`` respectively. ``qubit_mapping`` must be a
/// valid pointer to a contiguous array of ``uint32_t`` with enough space for the number of qubits
/// indicated in ``target``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_layout_generate_from_mapping(
    original_dag: *const DAGCircuit,
    target: *const Target,
    qubit_mapping: *const u32,
) -> *mut TranspileLayout {
    // SAFETY: Per the documentation these are valid pointers to the appropriate type
    let dag = unsafe { &*original_dag };
    let target = unsafe { &*target };
    // SAFETY: Per the documentation this must be a valid pointer to a u32 array with
    // target.num_qubits elements.
    let virt_to_phys: Vec<PhysicalQubit> = unsafe {
        std::slice::from_raw_parts(
            qubit_mapping as *const PhysicalQubit,
            target.num_qubits.unwrap() as usize,
        )
    }
    .to_vec();
    let initial_layout = NLayout::from_virtual_to_physical(virt_to_phys).unwrap();
    TranspileLayout::new(
        Some(initial_layout),
        None,
        dag.qubits().objects().to_owned(),
        dag.num_qubits() as u32,
        dag.qregs().to_vec(),
    )
    .into_leaked()
}

/// @ingroup QkTranspileLayout
/// Free a ``QkTranspileLayout`` object
///
/// @param layout a pointer to the layout to free
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not either null or a valid pointer to a ``QkTranspileLayout``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpile_layout_free(layout: *mut TranspileLayout) {
    // SAFETY: if `layout` is not null, then per documentation it is an owned pointer.  Per trait
    // documentation, all owned pointers can be given to `steal`.
    _ = (!layout.is_null()).then(|| unsafe { TranspileLayout::steal(layout) });
}

/// @ingroup QkTranspileLayout
/// Generate a Python-space ``TranspileLayout`` object from a ``QkTranspileLayout``.
///
/// The created Python-space object is a copy of the ``QkTranspileLayout`` provided, the data
/// representation is different between C and Python and the data is not moved to Python like
/// for some other ``*_to_python`` functions.
///
/// @param layout a pointer to a ``QkTranspileLayout``.
/// @param circuit a pointer to the original ``QkCircuit``.
/// @return the PyObject pointer for the Python space TranspileLayout object.
///
/// # Safety
///
/// Behavior is undefined if ``layout`` and ``circuit`` are not valid, non-null pointers to a
/// ``QkTranspileLayout`` and ``QkCircuit`` respectively. It is assumed that the thread currently
/// executing this function holds the Python GIL. This is required to create the Python object
/// returned by this function.
#[unsafe(no_mangle)]
#[cfg(feature = "python_binding")]
pub unsafe extern "C" fn qk_transpile_layout_to_python(
    layout: *const TranspileLayout,
    circuit: *const CircuitData,
) -> *mut PyObject {
    // SAFETY: Per the documentation layout and circuit are valid pointers
    // and the thread running the function holds the gil.
    unsafe {
        let layout = &*layout;
        let circuit = &*circuit;
        let py = Python::assume_attached();
        let res = layout.to_py_native(py, circuit.qubits().objects()).unwrap();
        res.into_ptr()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use qiskit_circuit::Qubit;
    use qiskit_circuit::bit::ShareableQubit;
    use qiskit_circuit::nlayout::{NLayout, PhysicalQubit};
    use qiskit_transpiler::transpile_layout::TranspileLayout;

    #[test]
    fn test_final_layout() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let input_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(routing_permutation),
            input_qubits,
            3,
            vec![],
        );
        let result = layout.final_index_layout(true);
        let expected = vec![PhysicalQubit(3), PhysicalQubit(5), PhysicalQubit(2)];
        assert_eq!(expected, result);
        let mut result = vec![u32::MAX; layout.num_input_qubits() as usize];
        unsafe { qk_transpile_layout_final_layout(&layout, true, result.as_mut_ptr()) };
        let expected = expected.into_iter().map(|x| x.0).collect::<Vec<_>>();
        assert_eq!(expected.as_slice(), result.as_slice());
    }

    #[test]
    fn test_final_layout_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let input_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(routing_permutation),
            input_qubits,
            3,
            vec![],
        );
        let result = layout.final_index_layout(false);
        let expected = vec![
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
        ];
        assert_eq!(expected, result);
        let mut result = vec![u32::MAX; layout.num_output_qubits() as usize];
        unsafe { qk_transpile_layout_final_layout(&layout, false, result.as_mut_ptr()) };
        let expected = expected.into_iter().map(|x| x.0).collect::<Vec<_>>();
        assert_eq!(expected.as_slice(), result.as_slice());
    }

    #[test]
    fn test_initial_layout() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let input_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(routing_permutation),
            input_qubits,
            3,
            vec![],
        );
        let expected: Vec<u32> = [PhysicalQubit(9), PhysicalQubit(4), PhysicalQubit(0)]
            .into_iter()
            .map(|x| x.0)
            .collect();
        let mut result: Vec<u32> = vec![u32::MAX; layout.num_input_qubits() as usize];
        assert!(unsafe { qk_transpile_layout_initial_layout(&layout, true, result.as_mut_ptr()) });
        assert_eq!(expected.as_slice(), result.as_slice());
    }

    #[test]
    fn test_initial_layout_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let expected: Vec<u32> = initial_layout_vec.iter().map(|x| x.0).collect();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let input_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(routing_permutation),
            input_qubits,
            3,
            vec![],
        );
        let mut result: Vec<u32> = vec![u32::MAX; layout.num_output_qubits() as usize];
        assert!(unsafe { qk_transpile_layout_initial_layout(&layout, false, result.as_mut_ptr()) });
        assert_eq!(expected.as_slice(), result.as_slice());
    }

    #[test]
    fn test_initial_layout_no_layout() {
        let input_qubits = vec![ShareableQubit::new_anonymous(); 10000];
        let layout = TranspileLayout::new(None, None, input_qubits, 10000, vec![]);
        let mut result: Vec<u32> = vec![u32::MAX; layout.num_input_qubits() as usize];
        assert!(!unsafe { qk_transpile_layout_initial_layout(&layout, true, result.as_mut_ptr()) });
    }

    #[test]
    fn test_output_permutation() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let expected = routing_permutation.iter().map(|x| x.0).collect::<Vec<_>>();
        let input_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(routing_permutation),
            input_qubits,
            3,
            vec![],
        );
        let mut result: Vec<u32> = vec![u32::MAX; layout.num_output_qubits() as usize];
        assert!(unsafe { qk_transpile_layout_output_permutation(&layout, result.as_mut_ptr()) });
        assert_eq!(expected.as_slice(), result.as_slice());
    }

    #[test]
    fn test_output_permutation_not_set() {
        let input_qubits = vec![ShareableQubit::new_anonymous(); 10000];
        let layout = TranspileLayout::new(None, None, input_qubits, 10000, vec![]);
        let mut result: Vec<u32> = vec![u32::MAX; layout.num_output_qubits() as usize];
        assert!(!unsafe { qk_transpile_layout_output_permutation(&layout, result.as_mut_ptr()) });
    }

    #[test]
    fn test_input_num_qubits() {
        let initial_layout_vec = (0..256).rev().map(PhysicalQubit::new).collect();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let input_qubits = vec![ShareableQubit::new_anonymous(); 256];
        let layout = TranspileLayout::new(Some(initial_layout), None, input_qubits, 3, vec![]);
        unsafe {
            assert_eq!(qk_transpile_layout_num_input_qubits(&layout), 3);
        }
    }

    #[test]
    fn test_output_num_qubits() {
        let initial_layout_vec = (0..256).rev().map(PhysicalQubit::new).collect();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let input_qubits = vec![ShareableQubit::new_anonymous(); 256];
        let layout = TranspileLayout::new(Some(initial_layout), None, input_qubits, 3, vec![]);
        unsafe {
            assert_eq!(qk_transpile_layout_num_output_qubits(&layout), 256);
        }
    }
}
