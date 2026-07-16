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

use qiskit_transpiler::neighbors::Neighbors;
use qiskit_transpiler::target::{Target, TargetCouplingError};

use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

/// An adjacency-list representation of a coupling graph.
///
/// This is initialized by `qk_neighbors_from_target`.
///
/// This object is read-only from C.  To satisfy the safety guarantees of `qk_neighbors_clear`, you
/// must not overwrite any data initialized by `qk_neighbors_from_target`, including any pointed-to
/// data.
///
/// # Representation
///
/// After initialization by `qk_neighbors_from_target`, the structure will be in one of two modes:
///
/// * all-to-all connectivity
/// * limited two-qubit connectivity
///
/// In the all-to-all case, the `neighbors` and `partition` pointers will both be the null pointer,
/// and `num_qubits` will be the number of qubits in the `QkTarget`.  These objects do not have
/// backing allocations, and do not need to be given to `qk_neighbors_clear` (though this function
/// is a safe no-op in this case).
///
/// In the limited two-qubit case (which is by far the more common for real hardware), see the
/// documentation of the structure members for their interpretation.
#[repr(C)]
#[derive(Debug)]
pub struct CNeighbors {
    /// A partitioned adjacency-list representation of the neighbors of each qubit.  This pointer is
    /// valid for exactly `partition[num_qubits + 1]` reads.
    ///
    /// For qubit number `i`, its neighbors are the values between offsets `partition[i]`
    /// (inclusive) and `partition[i + 1]` (exclusive).  The values between these two offsets are
    /// sorted in ascending order and contain no duplicates.
    pub neighbors: *const u32,
    /// How the `neighbors` field is partitioned into slices.  This pointer is valid for exactly
    /// `num_qubits + 1` reads.  The first value is always 0, and values increase monotonically.
    pub partition: *const usize,
    /// The number of qubits.
    pub num_qubits: u32,
}

/// @ingroup QkNeighbors
/// Does this coupling graph represent all-to-all connectivity?
///
/// This is represented by `neighbors` and `partition` being null pointers, so they are not valid
/// for any reads.
///
/// @param neighbors The coupling graph.
/// @return Whether the graph represents all to all connectivity.
///
/// # Safety
///
/// `neighbors` must point to a valid, initialized `QkNeighbors` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_neighbors_is_all_to_all(neighbors: *const CNeighbors) -> bool {
    // SAFETY: per documentation, `neighbors` points to a valid initialized `CNeighbors`.
    unsafe { (*neighbors).neighbors.is_null() && (*neighbors).partition.is_null() }
}

/// @ingroup QkNeighbors
/// Initialize a `QkNeighbors` object from a `QkTarget`.
///
/// If the target contains multi-qubit gates, they will be ignored and the connectivity will only
/// represent the two-qubit coupling constraints.  If the target represents all-to-all connectivity,
/// the function returns `true`, and the output pointers will be initialized to be null pointers, in
/// keeping with the representation of all-to-all connectivity.
///
/// @param target The target to read the connectivity from.
/// @param neighbors The `QkNeighbors` object to initialize.
/// @return Whether the `QkTarget` represented all-to-all connectivity (`true`) or has regular
///     connectivity (`false`).
///
/// # Examples
///
/// ```c
/// QkTarget *target = build_target_from_somewhere();
/// QkNeighbors neighbors;
/// if (qk_neighbors_from_target(target, &neighbors)) {
///     printf("All-to-all connectivity on &lu qubits.\n", neighbors.num_qubits);
///     return;
/// }
/// printf("Qubit 3 has %zu neighbors.\n", neighbors.partition[4] - neighbors.partition[3]);
/// printf("Those neighbors are: [");
/// for (size_t offset = neighbors.partition[3]; offset < neighbors.partition[4]; offset++) {
///     printf("%u%s",
///            neighbors.neighbors[offset],
///            offset + 1 == neighbors.partition[4] ? "" : ", ");
/// }
/// printf("]\n");
/// qk_neighbors_clear(&neighbors);
/// ```
///
/// # Safety
///
/// `target` must point to a valid `QkTarget` object.  `neighbors` must be aligned and safe to write
/// to, but need not be initialized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_neighbors_from_target(
    target: *const Target,
    neighbors: *mut CNeighbors,
) -> bool {
    // We don't create a reference to `neighbors` because it might not be initialised yet, so
    // creating a reference would be Rust-side UB.

    // SAFETY: per documentation, `target` points to a valid `Target`.
    let target = unsafe { const_ptr_as_ref(target) };
    let coupling = match target.coupling_graph() {
        Ok(coupling) | Err(TargetCouplingError::MultiQ(coupling)) => coupling,
        Err(TargetCouplingError::AllToAll) => {
            let num_qubits = target
                .num_qubits
                .expect("all Targets exposed to C have fixed size");
            // SAFETY: per documentation, `neighbors` points to an aligned `CNeighbors`.
            unsafe {
                (&raw mut (*neighbors).neighbors).write(std::ptr::null());
                (&raw mut (*neighbors).partition).write(std::ptr::null());
                (&raw mut (*neighbors).num_qubits).write(num_qubits);
            }
            return true;
        }
    };
    let (adjacency, partition) = Neighbors::from_coupling(&coupling).take();

    // The conversion to boxed slices is a provenance trick; we're explicitly throwing away any
    // excess capacity at this point.
    let adjacency = adjacency.into_boxed_slice();
    let partition = partition.into_boxed_slice();
    let num_qubits = partition.len() as u32 - 1;
    // SAFETY: per documentation, `neighbors` points to an aligned `CNeighbors`.
    unsafe {
        (&raw mut (*neighbors).neighbors).write(Box::into_raw(adjacency) as *const u32);
        (&raw mut (*neighbors).partition).write(Box::into_raw(partition) as *const usize);
        (&raw mut (*neighbors).num_qubits).write(num_qubits);
    }
    false
}

/// @ingroup QkNeighbors
/// Free all the allocations within the object.
///
/// After calling this function, the `QkNeighbors` object will contain null pointers in all its
/// allocations and present as if it represents all-to-all connectivity.
///
/// This should only be called on `QkNeighbors` objects that were initialized by
/// `qk_neighbors_from_target`.
///
/// @param neighbors A pointer to a ``QkNeighbors`` object.
///
/// # Safety
///
/// `neighbors` must point to a valid, initialized `QkNeighbors` object, which must have been
/// initialized by a call to `qk_neighbors_from_target` and unaltered since then.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_neighbors_clear(neighbors: *mut CNeighbors) {
    // SAFETY: per documentation, `neighbors` points to a valid initialised `CNeighbors`.
    let neighbors = unsafe { mut_ptr_as_ref(neighbors) };
    // Not strictly necessary, but doesn't really hurt to make things more obvious.
    let num_qubits = std::mem::replace(&mut neighbors.num_qubits, u32::MAX) as usize;
    let partition_ptr = std::mem::replace(&mut neighbors.partition, std::ptr::null());
    if partition_ptr.is_null() {
        return;
    }
    // SAFETY: `partition_ptr` is not null per check above.  Per `CNeighbors` struct documentation,
    // neither `partition_ptr` nor `num_qubits` have been written to, so `partition_ptr` points to
    // `num_qubits + 1` usizes and is owned by the `Box<[usize]>` allocator (it just had its `mut`
    // cast away).
    let partition = unsafe {
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            partition_ptr.cast_mut(),
            num_qubits + 1,
        ))
    };

    let neighbors_ptr = std::mem::replace(&mut neighbors.neighbors, std::ptr::null());
    // SAFETY: per `CNeighbors` struct documentation, `neighbors_ptr` was unassigned since set by
    // Rust space, and since `partition_ptr` is not null, neither is `neighbors_ptr`.  Per struct
    // documentation, nothing was written into `neighbors_ptr`, therefore it is owning data of a
    // number of `u32`
    let _ = unsafe {
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            neighbors_ptr.cast_mut(),
            *partition.last().expect("partition is always non-empty"),
        ))
    };
}

#[cfg(test)]
mod test {
    use super::*;

    use qiskit_circuit::PhysicalQubit;
    use qiskit_circuit::operations::StandardGate;
    use qiskit_transpiler::target::{Qargs, Target};
    use qiskit_util::IndexMap;

    // This is mostly for Miri.
    #[test]
    fn simple_line() {
        let num_qubits = 5;
        let mut target = Target::new(
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
        let mut line: IndexMap<_, _> = Default::default();
        for qubit in 0..num_qubits - 1 {
            line.insert(Qargs::from([qubit, qubit + 1].map(PhysicalQubit)), None);
        }
        target
            .add_instruction(StandardGate::CZ.into(), None, None, Some(line))
            .unwrap();

        let mut neighbors = CNeighbors {
            neighbors: std::ptr::null(),
            partition: std::ptr::null(),
            num_qubits: 0,
        };
        assert!(!unsafe { qk_neighbors_from_target(&target, &mut neighbors) });
        assert!(!unsafe { qk_neighbors_is_all_to_all(&neighbors) });

        let partition =
            unsafe { std::slice::from_raw_parts(neighbors.partition, num_qubits as usize + 1) };
        assert_eq!(partition, &[0, 1, 3, 5, 7, 8]);
        let adjacency =
            unsafe { std::slice::from_raw_parts(neighbors.neighbors, *partition.last().unwrap()) };
        assert_eq!(adjacency, &[1, 0, 2, 1, 3, 2, 4, 3]);

        unsafe { qk_neighbors_clear(&mut neighbors) };
        assert!(neighbors.neighbors.is_null());
        assert!(neighbors.partition.is_null());
    }

    // This is mostly for Miri.
    #[test]
    fn simple_all_to_all() {
        let num_qubits = 5;
        let mut target = Target::new(
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
        let mut line: IndexMap<_, _> = Default::default();
        line.insert(Qargs::Global, None);
        target
            .add_instruction(StandardGate::CZ.into(), None, None, Some(line))
            .unwrap();

        let mut neighbors = CNeighbors {
            neighbors: std::ptr::dangling(),
            partition: std::ptr::dangling(),
            num_qubits: 0,
        };
        assert!(unsafe { qk_neighbors_from_target(&target, &mut neighbors) });
        assert!(unsafe { qk_neighbors_is_all_to_all(&neighbors) });

        assert!(neighbors.neighbors.is_null());
        assert!(neighbors.partition.is_null());

        unsafe { qk_neighbors_clear(&mut neighbors) };
        assert!(neighbors.neighbors.is_null());
        assert!(neighbors.partition.is_null());
    }
}
