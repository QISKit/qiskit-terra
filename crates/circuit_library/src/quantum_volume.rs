// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;

use nalgebra::Matrix4;
use num_complex::{Complex64, ComplexFloat};
use rand::prelude::*;
use rand::rngs::SysRng;
use rand_distr::StandardNormal;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError, PyCircuitData};
use qiskit_circuit::operations::{ArrayType, Param, UnitaryGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use qiskit_util::getenv_use_multiple_threads;
use smallvec::{SmallVec, smallvec};

type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

#[inline(always)]
fn random_complex(rng: &mut Pcg64Mcg) -> Complex64 {
    Complex64::new(rng.sample(StandardNormal), rng.sample(StandardNormal))
        * std::f64::consts::FRAC_1_SQRT_2
}

// This function's implementation was modeled off of the algorithm used in the
// `scipy.stats.unitary_group.rvs()` function defined here:
//
// https://github.com/scipy/scipy/blob/v1.14.1/scipy/stats/_multivariate.py#L4224-L4256
#[inline]
fn random_unitaries(seed: u64, size: usize) -> impl Iterator<Item = Matrix4<Complex64>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);

    (0..size).map(move |_| {
        let mat: Matrix4<Complex64> = [
            [
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
            ],
            [
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
            ],
            [
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
            ],
            [
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
            ],
        ]
        .into();
        let (q, r) = mat.qr().unpack();
        let diag = r.map_diagonal(|x| x / x.abs());
        q.map_with_location(|i, _j, val| val * diag[i])
    })
}

const UNITARY_PER_SEED: usize = 50;

#[pyfunction]
#[pyo3(signature=(num_qubits, depth, seed=None))]
pub fn quantum_volume(num_qubits: u32, depth: usize, seed: Option<u64>) -> PyResult<PyCircuitData> {
    let width = num_qubits as usize / 2;
    let num_unitaries = width * depth;
    let mut permutation: Vec<Qubit> = (0..num_qubits).map(Qubit).collect();

    let mut build_instruction = |(unitary_index, unitary_array): (usize, Matrix4<Complex64>),
                                 rng: &mut Pcg64Mcg|
     -> Result<Instruction, CircuitDataError> {
        let layer_index = unitary_index % width;
        if layer_index == 0 {
            permutation.shuffle(rng);
        }

        let unitary_gate = UnitaryGate {
            array: ArrayType::TwoQ(unitary_array),
        };
        let qubit = layer_index * 2;
        Ok((
            PackedOperation::from_unitary(Box::new(unitary_gate)),
            smallvec![],
            vec![permutation[qubit], permutation[qubit + 1]],
            vec![],
        ))
    };

    let mut per_thread = num_unitaries / UNITARY_PER_SEED;
    if per_thread == 0 {
        per_thread = 10;
    }
    let mut outer_rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::try_from_rng(&mut SysRng).unwrap(),
    };
    let seed_vec: Vec<u64> = rand::distr::StandardUniform
        .sample_iter(&mut outer_rng)
        .take(num_unitaries)
        .collect();

    let unitaries: Vec<Matrix4<Complex64>> = if getenv_use_multiple_threads() && num_unitaries > 200
    {
        seed_vec
            .par_chunks(per_thread)
            .flat_map_iter(|seeds| random_unitaries(seeds[0], seeds.len()))
            .collect()
    } else {
        seed_vec
            .chunks(per_thread)
            .flat_map(|seeds| random_unitaries(seeds[0], seeds.len()))
            .collect()
    };
    Ok(CircuitData::from_packed_operations(
        num_qubits,
        0,
        unitaries
            .into_iter()
            .enumerate()
            .map(|x| build_instruction(x, &mut outer_rng)),
        Param::Float(0.),
    )?
    .into())
}
