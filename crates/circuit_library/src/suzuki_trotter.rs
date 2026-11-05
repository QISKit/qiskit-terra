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

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardInstruction, multiply_param, radd_param};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use qiskit_quantum_info::sparse_observable::SparseObservable;
use qiskit_synthesis::evolution::suzuki_trotter::{evolution, reorder_terms};
use qiskit_synthesis::pauli_evolution::sparse_term_evolution;
use smallvec::{SmallVec, smallvec};
use thiserror::Error;

// custom type for a more readable code
pub type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

pub fn suzuki_trotter_evolution(
    observable: &SparseObservable,
    order: u32,
    reps: u32,
    time: f64,
    preserve_order: bool,
    insert_barriers: bool,
) -> Result<CircuitData, EvolutionError> {
    if order > 1 && !order.is_multiple_of(2) || order == 0 {
        return Err(EvolutionError::OrderSymmetry(order));
    }

    let terms_iter = observable.iter().map(|mut view| {
        view.coeff.re *= time * 2.0 / (reps as f64);
        view
    });
    let terms = if preserve_order || observable.bit_terms().len() <= 1 {
        terms_iter.collect()
    } else {
        match reorder_terms(terms_iter) {
            Ok(terms) => terms,
            Err(msg) => return Err(EvolutionError::TermsReorder(msg.to_string())),
        }
    };

    // execute evolution
    let evo: Vec<(usize, f64)> = evolution(order, terms.len());

    // convert terms into instructions for circuit creation
    let repetitions = evo.len() as u32 * reps;
    let iter_repeated = evo.iter().cycle().take(repetitions as usize);

    let mut global_phase = Param::Float(0.0);
    let mut modified_phase = false;

    let repeated_evo = iter_repeated.enumerate().flat_map(|(i, (index, coeff))| {
        let view = &terms[*index];
        if view.bit_terms.is_empty() {
            global_phase = radd_param(global_phase.clone(), (view.coeff.re * coeff).into());
            modified_phase = true;
        }
        let pauli = view
            .bit_terms
            .iter()
            .map(|bit| bit.py_label())
            .collect::<String>();
        let instructions = sparse_term_evolution(
            pauli.as_str(),
            view.indices.into(),
            (view.coeff.re * coeff).into(),
            false,
            false,
        )
        .map(Ok::<Instruction, _>);

        let maybe_barrier = (insert_barriers && i as u32 != repetitions - 1)
            .then_some(Ok(create_barrier(observable.num_qubits())))
            .into_iter();
        instructions.chain(maybe_barrier)
    });

    match CircuitData::from_packed_operations(
        observable.num_qubits(),
        0,
        repeated_evo,
        Param::Float(0.0),
    ) {
        Ok(mut circuit) => {
            if modified_phase {
                let _ = circuit.set_global_phase_param(multiply_param(&global_phase, -0.5));
            }
            Ok(circuit)
        }
        Err(_) => Err(EvolutionError::CircuitBuild),
    }
}

fn create_barrier(num_qubits: u32) -> Instruction {
    (
        PackedOperation::from_standard_instruction(StandardInstruction::Barrier(num_qubits)),
        smallvec![],
        (0..num_qubits).map(Qubit).collect(),
        vec![],
    )
}

#[derive(Debug, Error)]
pub enum EvolutionError {
    /// An invalid order value passed to the main method
    #[error["Suzuki product formulae are symmetric and therefore only defined \
            for when the order is 1 or even, not {0}."]]
    OrderSymmetry(u32),

    /// A general error when trying to build circuit from generated instructions
    #[error["Failed building circuit"]]
    CircuitBuild,

    /// A general error for terms reordering error
    #[error["Error ocurred when trying to reorder terms: {0}"]]
    TermsReorder(String),
}

#[cfg(test)]
mod test {
    use crate::suzuki_trotter::suzuki_trotter_evolution;
    use num_complex::c64;
    use qiskit_quantum_info::sparse_observable::SparseObservable;

    #[test]
    fn test_suzuki_trotter() {
        let num_qubits = 2;
        let mut obs = SparseObservable::zero(num_qubits);
        obs.add_dense_label("ZZ", c64(1.0, 0.0)).unwrap();
        obs.add_dense_label("XI", c64(1.0, 0.0)).unwrap();
        let out = suzuki_trotter_evolution(&obs, 2, 10, 1.0, true, false).unwrap();
        assert_eq!(out.num_qubits(), 2);
    }
}
