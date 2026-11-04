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

use super::evolution_error::EvolutionError;
use num_complex::Complex64;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardInstruction, multiply_param, radd_param};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_quantum_info::sparse_observable::{SparseObservable, SparseTermView};
use qiskit_synthesis::evolution::suzuki_trotter::reorder_terms;
use qiskit_synthesis::pauli_evolution::{Instruction, sparse_term_evolution};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::SysRng;
use rand_pcg::Pcg64Mcg;
use smallvec::smallvec;

pub fn qdrift_evolution(
    observable: &SparseObservable,
    time: f64,
    reps: u32,
    seed: Option<u64>,
    preserve_order: bool,
    insert_barriers: bool,
) -> Result<CircuitData, EvolutionError> {
    let num_qubits = observable.num_qubits();

    let evo = qdrift(time, reps, seed, observable.coeffs().iter())?;

    let mut global_phase = Param::Float(0.0);
    let mut modified_phase = false;

    // Since we need the term to use the coefficient obtained by the evolution,
    // we chose to build `SparseTermView` directly instead of using the
    // `observable.term` function
    let sampled_paulis = evo.map(|(index, coeff)| {
        let start = observable.boundaries()[index];
        let end = observable.boundaries()[index + 1];
        let bit_terms = &observable.bit_terms()[start..end];
        if bit_terms.is_empty() {
            global_phase = radd_param(global_phase.clone(), (coeff).into());
            modified_phase = true;
        }
        SparseTermView {
            num_qubits,
            coeff: coeff.into(),
            bit_terms,
            indices: &observable.indices()[start..end],
        }
    });

    let sampled_paulis = if preserve_order || observable.bit_terms().len() <= 1 {
        sampled_paulis.collect()
    } else {
        match reorder_terms(sampled_paulis) {
            Ok(sampled_paulis) => sampled_paulis,
            Err(msg) => return Err(EvolutionError::TermsReorder(msg.to_string())),
        }
    };

    let instructions = sampled_paulis.iter().enumerate().flat_map(|(i, view)| {
        let coeff_param: Param = view.coeff.re.into();
        let inst_iter = sparse_term_evolution(
            &view
                .bit_terms
                .iter()
                .map(|bit| bit.py_label())
                .collect::<String>(),
            view.indices.into(),
            coeff_param,
            false,
            false,
        )
        .map(Ok::<Instruction, _>);

        let maybe_barrier = (insert_barriers && i != sampled_paulis.len() - 1)
            .then_some(Ok((
                PackedOperation::from_standard_instruction(StandardInstruction::Barrier(
                    num_qubits,
                )),
                smallvec![],
                (0..num_qubits).map(Qubit).collect(),
                vec![],
            )))
            .into_iter();
        inst_iter.chain(maybe_barrier)
    });

    CircuitData::from_packed_operations(
        observable.num_qubits(),
        0,
        instructions,
        if modified_phase {
            multiply_param(&global_phase, -0.5)
        } else {
            Param::Float(0.0)
        },
    )
    .map_err(EvolutionError::CircuitBuild)
}

pub fn qdrift<'a>(
    time: f64,
    reps: u32,
    seed: Option<u64>,
    coeffs_iter: impl Iterator<Item = &'a Complex64>,
) -> Result<impl Iterator<Item = (usize, f64)>, EvolutionError> {
    let mut rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::try_from_rng(&mut SysRng).unwrap(),
    };
    let mut lambd = 0.0;

    let coeffs = coeffs_iter
        .enumerate()
        .map(|(i, coeff)| {
            let (coeff_re, coeff_im) = (coeff.re.abs(), coeff.im.abs());
            lambd += if coeff_im > 1e-12 {
                return Err(EvolutionError::UnexpectedImaginaryPart(coeff_im));
            } else {
                coeff_re
            };
            Ok((i, coeff_re))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let num_gates = (2.0 * lambd.powi(2) * time.powi(2) * reps as f64).ceil() as usize;
    let distr = WeightedIndex::new(coeffs.iter().map(|(_, coeff)| coeff.abs() / lambd))?;

    Ok((0..num_gates).map(move |_| {
        let (index, coeff) = coeffs[distr.sample(&mut rng)];
        (
            index,
            coeff.signum() * (2.0 * lambd / (num_gates as f64) * time),
        )
    }))
}
