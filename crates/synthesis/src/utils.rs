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

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::packed_instruction::PackedInstruction;

/// Append `new`'s instructions into `circ`, remapping `new`'s qubits through
/// `qubit_map` (i.e. `new`'s qubit `i` becomes `circ`'s qubit `qubit_map[i]`).
pub(crate) fn append_with_qubit_map(
    circ: &mut CircuitData,
    new: CircuitData,
    qubit_map: &[Qubit],
) -> Result<(), CircuitDataError> {
    let new_qubits_map = circ.merge_qargs(new.qargs_interner(), |x| Some(qubit_map[x.index()]));
    circ.add_global_phase(new.global_phase())?;
    for inst in new.into_data_iter() {
        let out_inst = PackedInstruction {
            op: inst.op,
            params: inst.params,
            qubits: new_qubits_map[inst.qubits],
            clbits: Default::default(),
            label: inst.label,
            #[cfg(feature = "cache_pygates")]
            py_op: inst.py_op,
        };
        circ.push(out_inst)?;
    }
    Ok(())
}
