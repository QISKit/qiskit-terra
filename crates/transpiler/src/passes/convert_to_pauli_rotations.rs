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

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use qiskit_circuit::operations::PauliBased;
use smallvec::smallvec;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8, PI};

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PauliProductMeasurement, PauliProductRotation, StandardGate,
    StandardInstruction, add_param, multiply_param, radd_param,
};
use qiskit_circuit::{BlocksMode, Qubit, VarsMode};
use qiskit_quantum_info::sparse_observable::BitTerm;

use crate::TranspilerError;

type GateToPauliRotType = (&'static [(&'static [BitTerm], f64, &'static [u32])], f64);
type GateToPauliRotVec = (Vec<(&'static [BitTerm], Param, &'static [u32])>, Param);

/// Map standard gates to a list of equivalent Pauli product rotations and a global phase.
/// Each element of the list is of the form ((pauli, phase, [qubit indices]), global_phase).
/// Namely, each standard gate S is written as a product of Pauli product rotations:
/// S = R_pauli_1(phase_1) * ... * R_pauli_k(phase_k) * exp(i*global_phase)
static STANDARD_GATE_SUBSTITUTIONS: [Option<GateToPauliRotType>; 52] = [
    None, // GlobalPhase
    Some((
        &[(&[BitTerm::Y], FRAC_PI_2, &[0]), (&[BitTerm::X], PI, &[0])],
        FRAC_PI_2,
    )), // H
    None, // I
    Some((&[(&[BitTerm::X], PI, &[0])], FRAC_PI_2)), // X
    Some((&[(&[BitTerm::Y], PI, &[0])], FRAC_PI_2)), // Y
    Some((&[(&[BitTerm::Z], PI, &[0])], FRAC_PI_2)), // Z
    Some((&[(&[BitTerm::Z], 1.0, &[0])], 0.5)), // Phase
    None, // R
    Some((&[(&[BitTerm::X], 1.0, &[0])], 0.0)), // RX
    Some((&[(&[BitTerm::Y], 1.0, &[0])], 0.0)), // RY
    Some((&[(&[BitTerm::Z], 1.0, &[0])], 0.0)), // RZ
    Some((&[(&[BitTerm::Z], FRAC_PI_2, &[0])], FRAC_PI_4)), // S
    Some((&[(&[BitTerm::Z], -FRAC_PI_2, &[0])], -FRAC_PI_4)), // Sdg
    Some((&[(&[BitTerm::X], FRAC_PI_2, &[0])], FRAC_PI_4)), // SX
    Some((&[(&[BitTerm::X], -FRAC_PI_2, &[0])], -FRAC_PI_4)), // SXdg
    Some((&[(&[BitTerm::Z], FRAC_PI_4, &[0])], FRAC_PI_8)), // T
    Some((&[(&[BitTerm::Z], -FRAC_PI_4, &[0])], -FRAC_PI_8)), // Tdg
    None, // U
    Some((&[(&[BitTerm::Z], 1.0, &[0])], 0.5)), // U1
    None, // U2
    None, // U3
    Some((
        &[
            (&[BitTerm::X], -FRAC_PI_2, &[1]),
            (&[BitTerm::Z], -FRAC_PI_4, &[1]),
            (&[BitTerm::Z, BitTerm::X], FRAC_PI_2, &[0, 1]),
            (&[BitTerm::Z], -FRAC_PI_2, &[0]),
            (&[BitTerm::Y], -FRAC_PI_4, &[1]),
        ],
        7.0 * FRAC_PI_4,
    )), // CH
    Some((
        &[
            (&[BitTerm::Z, BitTerm::X], FRAC_PI_2, &[0, 1]),
            (&[BitTerm::Z], -FRAC_PI_2, &[0]),
            (&[BitTerm::X], -FRAC_PI_2, &[1]),
        ],
        -FRAC_PI_4,
    )), // CX
    Some((
        &[
            (&[BitTerm::Z, BitTerm::Y], FRAC_PI_2, &[0, 1]),
            (&[BitTerm::Z], -FRAC_PI_2, &[0]),
            (&[BitTerm::Y], -FRAC_PI_2, &[1]),
        ],
        -FRAC_PI_4,
    )), // CY
    Some((
        &[
            (&[BitTerm::Z, BitTerm::Z], FRAC_PI_2, &[0, 1]),
            (&[BitTerm::Z], -FRAC_PI_2, &[0]),
            (&[BitTerm::Z], -FRAC_PI_2, &[1]),
        ],
        -FRAC_PI_4,
    )), // CZ
    Some((
        &[
            (&[BitTerm::Z, BitTerm::X], FRAC_PI_2, &[0, 1]),
            (&[BitTerm::Z], -FRAC_PI_2, &[0]),
            (&[BitTerm::X], -FRAC_PI_2, &[1]),
            (&[BitTerm::Z, BitTerm::X], FRAC_PI_2, &[1, 0]),
            (&[BitTerm::X], -FRAC_PI_2, &[0]),
            (&[BitTerm::Z], -FRAC_PI_2, &[1]),
        ],
        -FRAC_PI_2,
    )), // DCX
    Some((
        &[
            (&[BitTerm::Z, BitTerm::X], -FRAC_PI_2, &[0, 1]),
            (&[BitTerm::Y], -PI, &[0]),
            (&[BitTerm::Z], PI, &[1]),
            (&[BitTerm::Y], PI, &[1]),
        ],
        PI,
    )), // ECR
    Some((
        &[
            (&[BitTerm::X, BitTerm::X], FRAC_PI_2, &[0, 1]),
            (&[BitTerm::Y, BitTerm::Y], FRAC_PI_2, &[0, 1]),
            (&[BitTerm::Z, BitTerm::Z], FRAC_PI_2, &[0, 1]),
        ],
        FRAC_PI_4,
    )), // Swap
    Some((
        &[
            (&[BitTerm::X, BitTerm::X], -FRAC_PI_2, &[0, 1]),
            (&[BitTerm::Y, BitTerm::Y], -FRAC_PI_2, &[0, 1]),
        ],
        0.0,
    )), // ISwap
    Some((
        &[
            (&[BitTerm::Z, BitTerm::Z], -0.5, &[0, 1]),
            (&[BitTerm::Z], 0.5, &[0]),
            (&[BitTerm::Z], 0.5, &[1]),
        ],
        0.25,
    )), // CPhase
    Some((
        &[
            (&[BitTerm::Z, BitTerm::X], -0.5, &[0, 1]),
            (&[BitTerm::X], 0.5, &[1]),
        ],
        0.0,
    )), // CRX
    Some((
        &[
            (&[BitTerm::Z, BitTerm::Y], -0.5, &[0, 1]),
            (&[BitTerm::Y], 0.5, &[1]),
        ],
        0.0,
    )), // CRY
    Some((
        &[
            (&[BitTerm::Z, BitTerm::Z], -0.5, &[0, 1]),
            (&[BitTerm::Z], 0.5, &[1]),
        ],
        0.0,
    )), // CRZ
    Some((
        &[
            (&[BitTerm::Z, BitTerm::Z], -FRAC_PI_4, &[0, 1]),
            (&[BitTerm::Z], FRAC_PI_4, &[0]),
            (&[BitTerm::Z], FRAC_PI_4, &[1]),
        ],
        FRAC_PI_8,
    )), // CS
    Some((
        &[
            (&[BitTerm::Z, BitTerm::Z], FRAC_PI_4, &[0, 1]),
            (&[BitTerm::Z], -FRAC_PI_4, &[0]),
            (&[BitTerm::Z], -FRAC_PI_4, &[1]),
        ],
        -FRAC_PI_8,
    )), // CSdg
    Some((
        &[
            (&[BitTerm::Z, BitTerm::X], -FRAC_PI_4, &[0, 1]),
            (&[BitTerm::Z], FRAC_PI_4, &[0]),
            (&[BitTerm::X], FRAC_PI_4, &[1]),
        ],
        FRAC_PI_8,
    )), // CSX
    None, // CU
    Some((
        &[
            (&[BitTerm::Z, BitTerm::Z], -0.5, &[0, 1]),
            (&[BitTerm::Z], 0.5, &[0]),
            (&[BitTerm::Z], 0.5, &[1]),
        ],
        0.25,
    )), // CU1
    None, // CU3
    Some((&[(&[BitTerm::X, BitTerm::X], 1.0, &[0, 1])], 0.0)), // RXX
    Some((&[(&[BitTerm::Y, BitTerm::Y], 1.0, &[0, 1])], 0.0)), // RYY
    Some((&[(&[BitTerm::Z, BitTerm::Z], 1.0, &[0, 1])], 0.0)), // RZZ
    Some((&[(&[BitTerm::Z, BitTerm::X], 1.0, &[0, 1])], 0.0)), // RZX
    None, // XXMinusYY
    None, // XXPlusYY
    Some((
        &[
            (&[BitTerm::Z, BitTerm::Z, BitTerm::X], FRAC_PI_4, &[0, 1, 2]),
            (&[BitTerm::Z, BitTerm::Z], -FRAC_PI_4, &[0, 1]),
            (&[BitTerm::Z, BitTerm::X], -FRAC_PI_4, &[1, 2]),
            (&[BitTerm::Z, BitTerm::X], -FRAC_PI_4, &[0, 2]),
            (&[BitTerm::Z], FRAC_PI_4, &[0]),
            (&[BitTerm::Z], FRAC_PI_4, &[1]),
            (&[BitTerm::X], FRAC_PI_4, &[2]),
        ],
        FRAC_PI_8,
    )),
    // CCX
    Some((
        &[
            (&[BitTerm::Z, BitTerm::Z, BitTerm::Z], FRAC_PI_4, &[0, 1, 2]),
            (&[BitTerm::Z, BitTerm::Z], -FRAC_PI_4, &[0, 1]),
            (&[BitTerm::Z, BitTerm::Z], -FRAC_PI_4, &[1, 2]),
            (&[BitTerm::Z, BitTerm::Z], -FRAC_PI_4, &[0, 2]),
            (&[BitTerm::Z], FRAC_PI_4, &[0]),
            (&[BitTerm::Z], FRAC_PI_4, &[1]),
            (&[BitTerm::Z], FRAC_PI_4, &[2]),
        ],
        FRAC_PI_8,
    )),
    // CCZ
    Some((
        &[
            (
                &[BitTerm::Z, BitTerm::X, BitTerm::X],
                -FRAC_PI_4,
                &[0, 1, 2],
            ),
            (
                &[BitTerm::Z, BitTerm::Y, BitTerm::Y],
                -FRAC_PI_4,
                &[0, 1, 2],
            ),
            (
                &[BitTerm::Z, BitTerm::Z, BitTerm::Z],
                -FRAC_PI_4,
                &[0, 1, 2],
            ),
            (&[BitTerm::X, BitTerm::X], FRAC_PI_4, &[1, 2]),
            (&[BitTerm::Y, BitTerm::Y], FRAC_PI_4, &[1, 2]),
            (&[BitTerm::Z, BitTerm::Z], FRAC_PI_4, &[1, 2]),
            (&[BitTerm::Z], FRAC_PI_4, &[0]),
        ],
        FRAC_PI_8,
    )),
    // CSwap
    Some((
        &[
            (&[BitTerm::X], -3.0 * FRAC_PI_4, &[2]),
            (&[BitTerm::Y], -FRAC_PI_2, &[2]),
            (&[BitTerm::Z, BitTerm::X], FRAC_PI_2, &[1, 2]),
            (&[BitTerm::Z], -FRAC_PI_2, &[1]),
            (&[BitTerm::X], -FRAC_PI_2, &[2]),
            (&[BitTerm::Z], -FRAC_PI_4, &[2]),
            (&[BitTerm::Z, BitTerm::X], FRAC_PI_2, &[0, 2]),
            (&[BitTerm::Z], -FRAC_PI_2, &[0]),
            (&[BitTerm::X], -FRAC_PI_2, &[2]),
            (&[BitTerm::Z], FRAC_PI_4, &[2]),
            (&[BitTerm::Z, BitTerm::X], FRAC_PI_2, &[1, 2]),
            (&[BitTerm::Z], -FRAC_PI_2, &[1]),
            (&[BitTerm::Z], FRAC_PI_2, &[2]),
            (&[BitTerm::X], FRAC_PI_4, &[2]),
        ],
        5.0 * FRAC_PI_4,
    )),
    // RCCX
    None, // C3X
    None, // C3SX
    None, // RC3X
];

/// Map standard gates with more than one parameter to a list of equivalent
/// Pauli product rotations and a global phase.
/// Each element of the list is of the form ((pauli, phase, [qubit indices]), global_phase).
/// Namely, each standard gate S is written as a product of Pauli product rotations:
/// S = R_pauli_1(phase_1) * ... * R_pauli_k(phase_k) * exp(i*global_phase)
fn replace_gate_by_pauli_vec(gate: StandardGate, angles: &[Param]) -> GateToPauliRotVec {
    match gate {
        StandardGate::U | StandardGate::U3 => (
            vec![
                (&[BitTerm::Z], angles[2].clone(), &[0]),
                (&[BitTerm::Y], angles[0].clone(), &[0]),
                (&[BitTerm::Z], angles[1].clone(), &[0]),
            ],
            multiply_param(&radd_param(angles[1].clone(), angles[2].clone()), 0.5),
        ),
        StandardGate::R => (
            vec![
                (
                    &[BitTerm::Z],
                    multiply_param(&add_param(&angles[1], -FRAC_PI_2), -1.0),
                    &[0],
                ),
                (&[BitTerm::Y], angles[0].clone(), &[0]),
                (&[BitTerm::Z], add_param(&angles[1], -FRAC_PI_2), &[0]),
            ],
            Param::Float(0.0),
        ),
        StandardGate::U2 => (
            vec![
                (&[BitTerm::Z], angles[1].clone(), &[0]),
                (&[BitTerm::Y], Param::Float(FRAC_PI_2), &[0]),
                (&[BitTerm::Z], angles[0].clone(), &[0]),
            ],
            multiply_param(&radd_param(angles[0].clone(), angles[1].clone()), 0.5),
        ),
        StandardGate::CU => (
            vec![
                (
                    &[BitTerm::Z],
                    radd_param(
                        radd_param(angles[3].clone(), multiply_param(&angles[1], 0.5)),
                        multiply_param(&angles[2], 0.5),
                    ),
                    &[0],
                ),
                (
                    &[BitTerm::Z],
                    radd_param(
                        multiply_param(&angles[2], 0.5),
                        multiply_param(&angles[1], -0.5),
                    ),
                    &[1],
                ),
                (&[BitTerm::Z, BitTerm::X], Param::Float(FRAC_PI_2), &[0, 1]),
                (&[BitTerm::Z], Param::Float(-FRAC_PI_2), &[0]),
                (&[BitTerm::X], Param::Float(-FRAC_PI_2), &[1]),
                (
                    &[BitTerm::Z],
                    radd_param(
                        multiply_param(&angles[2], -0.5),
                        multiply_param(&angles[1], -0.5),
                    ),
                    &[1],
                ),
                (&[BitTerm::Y], multiply_param(&angles[0], -0.5), &[1]),
                (&[BitTerm::Z, BitTerm::X], Param::Float(FRAC_PI_2), &[0, 1]),
                (&[BitTerm::Z], Param::Float(-FRAC_PI_2), &[0]),
                (&[BitTerm::X], Param::Float(-FRAC_PI_2), &[1]),
                (&[BitTerm::Y], multiply_param(&angles[0], 0.5), &[1]),
                (&[BitTerm::Z], angles[1].clone(), &[1]),
            ],
            radd_param(
                radd_param(
                    multiply_param(&angles[3], 0.5),
                    multiply_param(&angles[1], 0.25),
                ),
                add_param(&multiply_param(&angles[2], 0.25), -FRAC_PI_2),
            ),
        ),
        StandardGate::CU3 => (
            vec![
                (
                    &[BitTerm::Z],
                    radd_param(
                        multiply_param(&angles[1], 0.5),
                        multiply_param(&angles[2], 0.5),
                    ),
                    &[0],
                ),
                (
                    &[BitTerm::Z],
                    radd_param(
                        multiply_param(&angles[2], 0.5),
                        multiply_param(&angles[1], -0.5),
                    ),
                    &[1],
                ),
                (&[BitTerm::Z, BitTerm::X], Param::Float(FRAC_PI_2), &[0, 1]),
                (&[BitTerm::Z], Param::Float(-FRAC_PI_2), &[0]),
                (&[BitTerm::X], Param::Float(-FRAC_PI_2), &[1]),
                (
                    &[BitTerm::Z],
                    radd_param(
                        multiply_param(&angles[2], -0.5),
                        multiply_param(&angles[1], -0.5),
                    ),
                    &[1],
                ),
                (&[BitTerm::Y], multiply_param(&angles[0], -0.5), &[1]),
                (&[BitTerm::Z, BitTerm::X], Param::Float(FRAC_PI_2), &[0, 1]),
                (&[BitTerm::Z], Param::Float(-FRAC_PI_2), &[0]),
                (&[BitTerm::X], Param::Float(-FRAC_PI_2), &[1]),
                (&[BitTerm::Y], multiply_param(&angles[0], 0.5), &[1]),
                (&[BitTerm::Z], angles[1].clone(), &[1]),
            ],
            radd_param(
                multiply_param(&angles[1], 0.25),
                add_param(&multiply_param(&angles[2], 0.25), -FRAC_PI_2),
            ),
        ),
        StandardGate::XXPlusYY => (
            vec![
                (&[BitTerm::Z], angles[1].clone(), &[0]),
                (
                    &[BitTerm::X, BitTerm::X],
                    multiply_param(&angles[0], 0.5),
                    &[0, 1],
                ),
                (
                    &[BitTerm::Y, BitTerm::Y],
                    multiply_param(&angles[0], 0.5),
                    &[0, 1],
                ),
                (&[BitTerm::Z], multiply_param(&angles[1], -1.0), &[0]),
            ],
            Param::Float(0.0),
        ),
        StandardGate::XXMinusYY => (
            vec![
                (&[BitTerm::Z], multiply_param(&angles[1], -1.0), &[0]),
                (
                    &[BitTerm::X, BitTerm::X],
                    multiply_param(&angles[0], 0.5),
                    &[0, 1],
                ),
                (
                    &[BitTerm::Y, BitTerm::Y],
                    multiply_param(&angles[0], -0.5),
                    &[0, 1],
                ),
                (&[BitTerm::Z], angles[1].clone(), &[0]),
            ],
            Param::Float(0.0),
        ),
        _ => {
            panic!("This is only called for one and two qubit gates with more than one parameter.")
        }
    }
}

/// Converts a BitTerm pauli into (z, x)
fn bitterm_to_zx(pauli: BitTerm) -> (bool, bool) {
    match pauli {
        BitTerm::X => (false, true),
        BitTerm::Y => (true, true),
        BitTerm::Z => (true, false),
        _ => unreachable!("BitTerm is not a valid Pauli term (X, Y or Z)"),
    }
}

/// Takes an input of the form (pauli, time, [qubit indices]),
/// and outputs a corresponding PauliProductRotationGate(z, x, angle) gate.
fn generate_pauli_product_rotation_gate(paulis: &[BitTerm], angle: Param) -> PauliProductRotation {
    let mut x = Vec::with_capacity(paulis.len());
    let mut z = Vec::with_capacity(paulis.len());

    for p in paulis {
        let (pz, px) = bitterm_to_zx(*p);
        z.push(pz);
        x.push(px);
    }
    PauliProductRotation {
        z,
        x,
        angle: angle.clone(),
    }
}

/// Convert a quantum circuit containing single-qubit, two-qubit and three-qubit standard gates,
/// barriers and measurements, into an equivalent list of `PauliProductRotation` gates
/// and a global phase, as well as `PauliProductMeasurement` instructions.
/// Raises a `TranspilerError`: if the circuit contains instructions not supported by
/// the pass.
#[pyfunction]
#[pyo3(name = "convert_to_pauli_rotations")]
pub fn py_convert_to_pauli_rotations(dag: &DAGCircuit) -> PyResult<DAGCircuit> {
    let mut new_dag = dag.copy_empty_like(VarsMode::Alike, BlocksMode::Keep);

    // Iterate over nodes in the DAG and collect nodes
    let mut global_phase = Param::Float(0.0);

    for node_index in dag.topological_op_nodes(false) {
        let NodeType::Operation(inst) = &dag[node_index] else {
            unreachable!("dag.topological_op_nodes only returns Operations");
        };
        if matches!(
            inst.op.view(),
            OperationRef::ControlFlow(_)
                | OperationRef::StandardInstruction(StandardInstruction::Barrier(_))
                | OperationRef::StandardInstruction(StandardInstruction::Reset)
                | OperationRef::StandardInstruction(StandardInstruction::Delay(_))
                | OperationRef::PauliProductRotation(_)
                | OperationRef::PauliProductMeasurement(_)
        ) {
            new_dag.push_back(inst.clone())?;
        } else if inst.op.name() == "measure" {
            let z = vec![true];
            let x = vec![false];
            let neg = false;
            let ppm = PauliProductMeasurement { z, x, neg };
            let ppm_qubits = dag.get_qargs(inst.qubits);
            let ppm_clbits = dag.get_cargs(inst.clbits);

            new_dag.apply_operation_back(
                PauliBased::PauliProductMeasurement(ppm).into(),
                ppm_qubits,
                ppm_clbits,
                None,
                None,
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
        } else if let OperationRef::StandardGate(gate) = inst.op.view() {
            if gate.num_qubits() > 3 {
                return Err(TranspilerError::new_err(format!(
                    "Unable to convert to pauli rotations as the circuit contains instructions not supported by the pass: {:?}",
                    gate.name()
                )));
            }
            // handling only 1-qubit, 2-qubit and 3-qubit gates with no parameter or with a single parameter
            if gate.num_params() <= 1 {
                if let Some((sequence, global_phase_update)) =
                    STANDARD_GATE_SUBSTITUTIONS[gate as usize]
                {
                    let angle: Param = if gate.num_params() == 1 {
                        inst.params_view()[0].clone()
                    } else {
                        Param::Float(1.0)
                    };
                    let original_qubits = dag.get_qargs(inst.qubits);
                    for (paulis, phase_rescale, qubits) in sequence {
                        let updated_qubits: Vec<Qubit> = qubits
                            .iter()
                            .map(|q| original_qubits[*q as usize])
                            .collect();
                        let time = multiply_param(&angle, *phase_rescale);
                        let ppr = generate_pauli_product_rotation_gate(paulis, time.clone());

                        new_dag.apply_operation_back(
                            PauliBased::PauliProductRotation(ppr).into(),
                            &updated_qubits,
                            &[],
                            Some(Parameters::Params(smallvec![time])),
                            None,
                            #[cfg(feature = "cache_pygates")]
                            None,
                        )?;
                    }
                    global_phase =
                        radd_param(multiply_param(&angle, global_phase_update), global_phase);
                } else if matches!(gate, StandardGate::GlobalPhase) {
                    let global_phase_update = inst.params_view()[0].clone();
                    global_phase = radd_param(global_phase, global_phase_update);
                } else if matches!(gate, StandardGate::I) {
                } else {
                    return Err(TranspilerError::new_err(format!(
                        "Unable to convert to pauli rotations as the circuit contains instructions not supported by the pass: {:?}",
                        inst.op.name()
                    )));
                }
            }
            // handling only 1-qubit and 2-qubit gates with more than one parameter
            else if matches!(
                gate,
                StandardGate::U
                    | StandardGate::U2
                    | StandardGate::U3
                    | StandardGate::R
                    | StandardGate::CU
                    | StandardGate::CU3
                    | StandardGate::XXPlusYY
                    | StandardGate::XXMinusYY
            ) {
                let angles = inst.params_view();
                let original_qubits = dag.get_qargs(inst.qubits);
                let (sequence, global_phase_update) = replace_gate_by_pauli_vec(gate, angles);
                for (paulis, phase_rescale, qubits) in sequence {
                    let updated_qubits: Vec<Qubit> = qubits
                        .iter()
                        .map(|q| original_qubits[*q as usize])
                        .collect();
                    let time = phase_rescale;
                    let ppr = generate_pauli_product_rotation_gate(paulis, time.clone());

                    new_dag.apply_operation_back(
                        PauliBased::PauliProductRotation(ppr).into(),
                        &updated_qubits,
                        &[],
                        Some(Parameters::Params(smallvec![time])),
                        None,
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                }
                global_phase = radd_param(global_phase, global_phase_update);
            }
        } else {
            return Err(TranspilerError::new_err(format!(
                "Unable to run convert to pauli rotations as the circuit contains instructions not supported by the pass: {:?}",
                inst.op.name()
            )));
        }
    }

    new_dag.add_global_phase(&global_phase)?;

    Ok(new_dag)
}

pub fn convert_to_pauli_rotations_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_convert_to_pauli_rotations))?;
    Ok(())
}
