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

use std::f64::consts::PI;

use hashbrown::HashMap;
use ndarray::ArrayView2;
use ndarray::linalg::kron;
use ndarray::prelude::*;
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::gate_matrix::ONE_QUBIT_IDENTITY;
use qiskit_circuit::instruction::Instruction;
use qiskit_circuit::operations::StandardGate::{I, X, Y, Z};
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;

use crate::QiskitError;

use qiskit_circuit::{BlocksMode, NoBlocks, VarsMode};
use qiskit_transpiler::passes::{
    Optimize1qGatesDecompositionState, run_optimize_1q_gates_decomposition,
};
use qiskit_transpiler::target::Target;
use rand::prelude::*;
use rand::rngs::SysRng;
use rand_pcg::Pcg64Mcg;

static ECR_TWIRL_SET: [([StandardGate; 4], f64); 16] = [
    ([I, Z, Z, Y], 0.),
    ([I, X, I, X], 0.),
    ([I, Y, Z, Z], PI),
    ([I, I, I, I], 0.),
    ([Z, X, Z, X], PI),
    ([Z, Y, I, Z], 0.),
    ([Z, I, Z, I], PI),
    ([Z, Z, I, Y], PI),
    ([X, Y, X, Y], 0.),
    ([X, I, Y, X], PI),
    ([X, Z, X, Z], 0.),
    ([X, X, Y, I], PI),
    ([Y, I, X, X], PI),
    ([Y, Z, Y, Z], PI),
    ([Y, X, X, I], PI),
    ([Y, Y, Y, Y], PI),
];

static CX_TWIRL_SET: [([StandardGate; 4], f64); 16] = [
    ([I, Z, Z, Z], 0.),
    ([I, X, I, X], 0.),
    ([I, Y, Z, Y], 0.),
    ([I, I, I, I], 0.),
    ([Z, X, Z, X], 0.),
    ([Z, Y, I, Y], 0.),
    ([Z, I, Z, I], 0.),
    ([Z, Z, I, Z], 0.),
    ([X, Y, Y, Z], 0.),
    ([X, I, X, X], 0.),
    ([X, Z, Y, Y], PI),
    ([X, X, X, I], 0.),
    ([Y, I, Y, X], 0.),
    ([Y, Z, X, Y], 0.),
    ([Y, X, Y, I], 0.),
    ([Y, Y, X, Z], PI),
];

static CZ_TWIRL_SET: [([StandardGate; 4], f64); 16] = [
    ([I, Z, I, Z], 0.),
    ([I, X, Z, X], 0.),
    ([I, Y, Z, Y], 0.),
    ([I, I, I, I], 0.),
    ([Z, X, I, X], 0.),
    ([Z, Y, I, Y], 0.),
    ([Z, I, Z, I], 0.),
    ([Z, Z, Z, Z], 0.),
    ([X, Y, Y, X], PI),
    ([X, I, X, Z], 0.),
    ([X, Z, X, I], 0.),
    ([X, X, Y, Y], 0.),
    ([Y, I, Y, Z], 0.),
    ([Y, Z, Y, I], 0.),
    ([Y, X, X, Y], PI),
    ([Y, Y, X, X], 0.),
];

static ISWAP_TWIRL_SET: [([StandardGate; 4], f64); 16] = [
    ([I, Z, Z, I], 0.),
    ([I, X, Y, Z], 0.),
    ([I, Y, X, Z], PI),
    ([I, I, I, I], 0.),
    ([Z, X, Y, I], 0.),
    ([Z, Y, X, I], PI),
    ([Z, I, I, Z], 0.),
    ([Z, Z, Z, Z], 0.),
    ([X, Y, Y, X], 0.),
    ([X, I, Z, Y], 0.),
    ([X, Z, I, Y], 0.),
    ([X, X, X, X], 0.),
    ([Y, I, Z, X], PI),
    ([Y, Z, I, X], PI),
    ([Y, X, X, Y], 0.),
    ([Y, Y, Y, Y], 0.),
];

static TWIRLING_SETS: [&[([StandardGate; 4], f64); 16]; 4] = [
    &CX_TWIRL_SET,
    &CZ_TWIRL_SET,
    &ECR_TWIRL_SET,
    &ISWAP_TWIRL_SET,
];

const CX_MASK: u8 = 8;
const CZ_MASK: u8 = 4;
const ECR_MASK: u8 = 2;
const ISWAP_MASK: u8 = 1;

#[inline(always)]
fn diff_frob_norm_sq(array: ArrayView2<Complex64>, gate_matrix: ArrayView2<Complex64>) -> f64 {
    let mut res: f64 = 0.;
    for i in 0..4 {
        for j in 0..4 {
            let gate = gate_matrix[[i, j]];
            let twirled = array[[i, j]];
            let diff = twirled - gate;
            res += (diff.conj() * diff).re;
        }
    }
    res
}

fn generate_twirling_set(gate_matrix: ArrayView2<Complex64>) -> Vec<([StandardGate; 4], f64)> {
    let mut out_vec = Vec::with_capacity(16);
    let i_matrix = aview2(&ONE_QUBIT_IDENTITY);
    let x_matrix = aview2(&qiskit_circuit::gate_matrix::X_GATE);
    let y_matrix = aview2(&qiskit_circuit::gate_matrix::Y_GATE);
    let z_matrix = aview2(&qiskit_circuit::gate_matrix::Z_GATE);
    let iter_set = [I, X, Y, Z];
    let kron_set: [Array2<Complex64>; 16] = [
        kron(&i_matrix, &i_matrix),
        kron(&x_matrix, &i_matrix),
        kron(&y_matrix, &i_matrix),
        kron(&z_matrix, &i_matrix),
        kron(&i_matrix, &x_matrix),
        kron(&x_matrix, &x_matrix),
        kron(&y_matrix, &x_matrix),
        kron(&z_matrix, &x_matrix),
        kron(&i_matrix, &y_matrix),
        kron(&x_matrix, &y_matrix),
        kron(&y_matrix, &y_matrix),
        kron(&z_matrix, &y_matrix),
        kron(&i_matrix, &z_matrix),
        kron(&x_matrix, &z_matrix),
        kron(&y_matrix, &z_matrix),
        kron(&z_matrix, &z_matrix),
    ];
    for (i_idx, i) in iter_set.iter().enumerate() {
        for (j_idx, j) in iter_set.iter().enumerate() {
            let before_matrix = kron_set[i_idx * 4 + j_idx].view();
            let half_twirled_matrix = gate_matrix.dot(&before_matrix);
            for (k_idx, k) in iter_set.iter().enumerate() {
                for (l_idx, l) in iter_set.iter().enumerate() {
                    let after_matrix = kron_set[k_idx * 4 + l_idx].view();
                    let twirled_matrix = after_matrix.dot(&half_twirled_matrix);
                    let norm: f64 = diff_frob_norm_sq(twirled_matrix.view(), gate_matrix);
                    if norm.abs() < 1e-15 {
                        out_vec.push(([*i, *j, *k, *l], 0.));
                    } else if (norm - 16.).abs() < 1e-15 {
                        out_vec.push(([*i, *j, *k, *l], PI));
                    }
                }
            }
        }
    }
    out_vec
}

fn twirl_gate(
    circ: &CircuitData,
    rng: &mut Pcg64Mcg,
    out_circ: &mut CircuitData,
    twirl_set: &[([StandardGate; 4], f64)],
    inst: &PackedInstruction,
) -> PyResult<()> {
    let qubits = circ.get_qargs(inst.qubits);
    let (twirl, twirl_phase) = twirl_set.choose(rng).unwrap();
    let bit_zero = out_circ.add_qargs(std::slice::from_ref(&qubits[0]));
    let bit_one = out_circ.add_qargs(std::slice::from_ref(&qubits[1]));
    out_circ.push(PackedInstruction::from_standard_gate(
        twirl[0], None, bit_zero,
    ))?;
    out_circ.push(PackedInstruction::from_standard_gate(
        twirl[1], None, bit_one,
    ))?;

    out_circ.push(inst.clone())?;
    out_circ.push(PackedInstruction::from_standard_gate(
        twirl[2], None, bit_zero,
    ))?;
    out_circ.push(PackedInstruction::from_standard_gate(
        twirl[3], None, bit_one,
    ))?;

    if *twirl_phase != 0. {
        out_circ.add_global_phase(&Param::Float(*twirl_phase))?;
    }
    Ok(())
}

type CustomGateTwirlingMap = HashMap<String, Vec<([StandardGate; 4], f64)>>;

fn generate_twirled_circuit(
    circ: &CircuitData,
    rng: &mut Pcg64Mcg,
    twirling_mask: u8,
    custom_gate_map: Option<&CustomGateTwirlingMap>,
    optimizer_target: Option<&Target>,
) -> PyResult<CircuitData> {
    let mut out_circ = CircuitData::copy_empty_like(circ, VarsMode::Alike, BlocksMode::Drop)?;

    for inst in circ.data().iter() {
        if let Some(custom_gate_map) = custom_gate_map {
            if let Some(twirling_set) = custom_gate_map.get(inst.op.name()) {
                twirl_gate(circ, rng, &mut out_circ, twirling_set.as_slice(), inst)?;
                continue;
            }
        }
        if let Some(control_flow) = circ.try_view_control_flow(inst) {
            let new_blocks: Vec<_> = control_flow
                .blocks()
                .into_iter()
                .map(|block| {
                    let new_block = generate_twirled_circuit(
                        block,
                        rng,
                        twirling_mask,
                        custom_gate_map,
                        optimizer_target,
                    )?;
                    Ok(out_circ.add_block(new_block))
                })
                .collect::<PyResult<_>>()?;
            out_circ.push(PackedInstruction::from_control_flow(
                inst.op.control_flow().clone(),
                new_blocks,
                inst.qubits,
                inst.clbits,
                inst.label.as_deref().cloned(),
            ))?;
            continue;
        }
        match inst.op.view() {
            OperationRef::StandardGate(gate) => match gate {
                StandardGate::CX => {
                    if twirling_mask & CX_MASK != 0 {
                        twirl_gate(circ, rng, &mut out_circ, TWIRLING_SETS[0], inst)?;
                    } else {
                        out_circ.push(inst.clone())?;
                    }
                }
                StandardGate::CZ => {
                    if twirling_mask & CZ_MASK != 0 {
                        twirl_gate(circ, rng, &mut out_circ, TWIRLING_SETS[1], inst)?;
                    } else {
                        out_circ.push(inst.clone())?;
                    }
                }
                StandardGate::ECR => {
                    if twirling_mask & ECR_MASK != 0 {
                        twirl_gate(circ, rng, &mut out_circ, TWIRLING_SETS[2], inst)?;
                    } else {
                        out_circ.push(inst.clone())?;
                    }
                }
                StandardGate::ISwap => {
                    if twirling_mask & ISWAP_MASK != 0 {
                        twirl_gate(circ, rng, &mut out_circ, TWIRLING_SETS[3], inst)?;
                    } else {
                        out_circ.push(inst.clone())?;
                    }
                }
                _ => out_circ.push(inst.clone())?,
            },
            _ => {
                out_circ.push(inst.clone())?;
            }
        }
    }
    if let Some(optimizer_target) = optimizer_target {
        let mut dag = DAGCircuit::from_circuit_data(&out_circ, false, None, None, None, None)?;
        let state = Optimize1qGatesDecompositionState::new(
            optimizer_target.num_qubits.unwrap_or(0) as usize,
        );
        run_optimize_1q_gates_decomposition(&mut dag, &state, Some(optimizer_target), None, None)?;
        Ok(CircuitData::from_dag_ref(&dag)?)
    } else {
        Ok(out_circ)
    }
}

#[pyfunction]
#[pyo3(signature=(circ, twirled_gate=None, custom_twirled_gates=None, seed=None, num_twirls=1, optimizer_target=None))]
pub(crate) fn twirl_circuit(
    circ: &PyCircuitData,
    twirled_gate: Option<Vec<StandardGate>>,
    custom_twirled_gates: Option<Vec<OperationFromPython<NoBlocks>>>,
    seed: Option<u64>,
    num_twirls: usize,
    optimizer_target: Option<&Target>,
) -> PyResult<Vec<PyCircuitData>> {
    let mut rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::try_from_rng(&mut SysRng).unwrap(),
    };
    let twirling_mask: u8 = match twirled_gate {
        Some(gates) => {
            let mut out_mask = 0;
            for gate in gates {
                let new_mask = match gate {
                    StandardGate::CX => CX_MASK,
                    StandardGate::CZ => CZ_MASK,
                    StandardGate::ECR => ECR_MASK,
                    StandardGate::ISwap => ISWAP_MASK,
                    _ => {
                        return Err(QiskitError::new_err(format!(
                            "Provided gate to twirl, {}, is not currently supported you can only use cx, cz, ecr or iswap.",
                            gate.name()
                        )));
                    }
                };
                out_mask |= new_mask;
            }
            out_mask
        }
        None => {
            if custom_twirled_gates.is_none() {
                15
            } else {
                0
            }
        }
    };
    let custom_gate_twirling_sets: Option<CustomGateTwirlingMap> =
        custom_twirled_gates.map(|gates| {
            gates
                .into_iter()
                .filter_map(|gate| {
                    if gate.operation.num_qubits() != 2 {
                        return Some(Err(QiskitError::new_err(
                            format!(
                                "The provided gate to twirl {} operates on an invalid number of qubits {}, it can only be a two qubit gate",
                                gate.operation.name(),
                                gate.operation.num_qubits(),
                            )
                        )))
                    }
                    if gate.operation.num_params() != 0 {
                        return Some(Err(QiskitError::new_err(
                            format!(
                                "The provided gate to twirl {} takes a parameter, it can only be an unparameterized gate",
                                gate.operation.name(),
                            )
                        )))
                    }
                    let matrix = gate.try_matrix();
                    if let Some(matrix) = matrix {
                        let twirl_set = generate_twirling_set(matrix.view());
                        if twirl_set.is_empty() {
                            None
                        } else {
                            Some(Ok((gate.operation.name().to_string(), twirl_set)))
                        }
                    } else {
                        Some(Err(QiskitError::new_err(
                            format!("Provided gate to twirl, {}, does not have a matrix defined and can't be twirled", gate.operation.name())
                        )))
                    }
                })
                .collect()
        }).transpose()?;
    (0..num_twirls)
        .map(|_| {
            generate_twirled_circuit(
                circ,
                &mut rng,
                twirling_mask,
                custom_gate_twirling_sets.as_ref(),
                optimizer_target,
            )
            .map(Into::into)
        })
        .collect()
}

pub fn twirling(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(twirl_circuit))?;
    Ok(())
}
