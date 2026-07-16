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

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, pyfunction, wrap_pyfunction};

use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_util::IndexMap;
use smallvec::{SmallVec, smallvec};

use super::analyze_commutations;
use crate::commutation_checker::{CommutationChecker, CommutationError};
use approx::abs_diff_eq;
use qiskit_circuit::Qubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGError, NodeType};
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_synthesis::QiskitError;

/// Possible errors during the `CommutationCancellation` pass
#[derive(Debug, thiserror::Error)]
pub enum CommutationCancelError {
    #[error("Rotational gate with parameter expression encountered in cancellation {0:?}")]
    RotationalParameter(PackedOperation),
    #[error("Angle for operation {0} is not defined")]
    OperationUndefinedAngle(String),
    #[error(transparent)]
    DAGCircuit(#[from] DAGError),
    #[error(transparent)]
    Analysis(#[from] CommutationError),
}

impl From<CommutationCancelError> for PyErr {
    fn from(value: CommutationCancelError) -> Self {
        match value {
            CommutationCancelError::RotationalParameter(_) => {
                QiskitError::new_err(value.to_string())
            }
            CommutationCancelError::OperationUndefinedAngle(_) => {
                PyRuntimeError::new_err(value.to_string())
            }
            CommutationCancelError::Analysis(commutation_analysis_error) => {
                commutation_analysis_error.into()
            }
            CommutationCancelError::DAGCircuit(dagcircuit_inner_error) => {
                dagcircuit_inner_error.into()
            }
        }
    }
}

const _CUTOFF_PRECISION: f64 = 1e-5;
static ROTATION_GATES: [&str; 4] = ["p", "u1", "rz", "rx"];

static VAR_Z_MAP: [(&str, StandardGate); 3] = [
    ("rz", StandardGate::RZ),
    ("p", StandardGate::Phase),
    ("u1", StandardGate::U1),
];
static Z_ROTATIONS: [StandardGate; 8] = [
    StandardGate::Phase,
    StandardGate::Z,
    StandardGate::U1,
    StandardGate::RZ,
    StandardGate::T,
    StandardGate::Tdg,
    StandardGate::S,
    StandardGate::Sdg,
];
static X_ROTATIONS: [StandardGate; 4] = [
    StandardGate::X,
    StandardGate::RX,
    StandardGate::SX,
    StandardGate::SXdg,
];
static SUPPORTED_GATES: [StandardGate; 5] = [
    StandardGate::CX,
    StandardGate::CY,
    StandardGate::CZ,
    StandardGate::H,
    StandardGate::Y,
];

#[derive(Hash, Eq, PartialEq, Debug)]
enum GateOrRotation {
    Gate(StandardGate),
    ZRotation,
    XRotation,
}
#[derive(Hash, Eq, PartialEq, Debug)]
struct CancellationSetKey {
    gate: GateOrRotation,
    qubits: SmallVec<[Qubit; 2]>,
    com_set_index: usize,
    second_index: Option<usize>,
}

#[pyfunction(name = "cancel_commutations")]
#[pyo3(signature = (dag, commutation_checker, basis_gates=None, approximation_degree=1.))]
fn py_cancel_commutations(
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
    basis_gates: Option<Vec<String>>,
    approximation_degree: f64,
) -> PyResult<()> {
    cancel_commutations(dag, commutation_checker, basis_gates, approximation_degree)
        .map_err(Into::into)
}

pub fn cancel_commutations(
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
    basis_gates: Option<Vec<String>>,
    approximation_degree: f64,
) -> Result<(), CommutationCancelError> {
    let basis = basis_gates.unwrap_or_default();
    let z_var_gate = dag
        .get_op_counts()
        .keys()
        .find_map(|g| {
            VAR_Z_MAP
                .iter()
                .find(|(key, _)| *key == g.as_str())
                .map(|(_, gate)| gate)
        })
        .or_else(|| {
            // Fallback to the first matching key from basis if there is no match in dag.op_names
            basis.iter().find_map(|g| {
                VAR_Z_MAP
                    .iter()
                    .find(|(key, _)| *key == g.as_str())
                    .map(|(_, gate)| gate)
            })
        });
    let sx_supported = dag.get_op_counts().contains_key("sx") || basis.iter().any(|x| x == "sx");
    let x_supported = dag.get_op_counts().contains_key("x") || basis.iter().any(|x| x == "x");

    // RZ and P/U1 have a phase difference of angle/2, which we need to account for
    let z_phase_shift = match z_var_gate {
        Some(z_var_gate) => {
            if z_var_gate == &StandardGate::RZ {
                |gate_name: &str, angle: f64| -> f64 {
                    if ["u1", "p"].contains(&gate_name) {
                        angle / 2.
                    } else {
                        0.
                    }
                }
            } else {
                |gate_name: &str, angle: f64| -> f64 {
                    if gate_name == "rz" { -angle / 2. } else { 0. }
                }
            }
        }
        // if there's no z_var_gate detected, Z rotations are not merged, so we have no phase shifts
        None => |_name: &str, _angle: f64| -> f64 { 0. },
    };

    // Gate sets to be cancelled
    /* Traverse each qubit to generate the cancel dictionaries
     Cancel dictionaries:
      - For 1-qubit gates the key is (gate_type, qubit_id, commutation_set_id),
        the value is the list of gates that share the same gate type, qubit, commutation set.
      - For 2qbit gates the key: (gate_type, first_qbit, sec_qbit, first commutation_set_id,
        sec_commutation_set_id), the value is the list gates that share the same gate type,
        qubits and commutation sets.
    */
    let (commutation_set, node_indices) =
        analyze_commutations(dag, commutation_checker, approximation_degree)?;
    let mut cancellation_sets = IndexMap::with_hasher(::foldhash::fast::RandomState::default());

    (0..dag.num_qubits() as u32).for_each(|qubit| {
        let wire = Qubit(qubit);
        let wire_commutation_set = &commutation_set[qubit as usize];
        if !wire_commutation_set.is_empty() {
            for (com_set_idx, com_set) in wire_commutation_set.iter().enumerate() {
                if let Some(&nd) = com_set.first() {
                    if !matches!(dag[nd], NodeType::Operation(_)) {
                        continue;
                    }
                } else {
                    continue;
                }
                for node in com_set.iter() {
                    let instr = match &dag[*node] {
                        NodeType::Operation(instr) => instr,
                        _ => panic!("Unexpected type in commutation set."),
                    };
                    let num_qargs = dag.get_qargs(instr.qubits).len();
                    // no support for cancellation of parameterized gates
                    if instr.is_parameterized() {
                        continue;
                    }
                    if let Some(op_gate) = instr.op.try_standard_gate() {
                        if num_qargs == 1 && SUPPORTED_GATES.contains(&op_gate) {
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::Gate(op_gate),
                                    qubits: smallvec![wire],
                                    com_set_index: com_set_idx,
                                    second_index: None,
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }

                        if num_qargs == 1 && Z_ROTATIONS.contains(&op_gate) {
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::ZRotation,
                                    qubits: smallvec![wire],
                                    com_set_index: com_set_idx,
                                    second_index: None,
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }
                        if num_qargs == 1 && X_ROTATIONS.contains(&op_gate) {
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::XRotation,
                                    qubits: smallvec![wire],
                                    com_set_index: com_set_idx,
                                    second_index: None,
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }
                        // Don't deal with Y rotation, because Y rotation doesn't commute with
                        // CNOT, so it should be dealt with by optimized1qgate pass
                        if num_qargs == 2 && dag.get_qargs(instr.qubits)[0] == wire {
                            let second_qarg = dag.get_qargs(instr.qubits)[1];
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::Gate(op_gate),
                                    qubits: smallvec![wire, second_qarg],
                                    com_set_index: com_set_idx,
                                    second_index: node_indices[second_qarg.index()]
                                        .get(node)
                                        .copied(),
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }
                    }
                }
            }
        }
    });

    for (cancel_key, cancel_set) in &cancellation_sets {
        if cancel_set.len() > 1 {
            if let GateOrRotation::Gate(g) = cancel_key.gate {
                if SUPPORTED_GATES.contains(&g) {
                    for &c_node in &cancel_set[0..(cancel_set.len() / 2) * 2] {
                        dag.remove_op_node(c_node);
                    }
                }
                continue;
            }
            if matches!(cancel_key.gate, GateOrRotation::ZRotation) && z_var_gate.is_none() {
                continue;
            }
            if matches!(
                cancel_key.gate,
                GateOrRotation::ZRotation | GateOrRotation::XRotation
            ) {
                let mut total_angle: f64 = 0.0;
                let mut total_phase: f64 = 0.0;
                for current_node in cancel_set {
                    let node_op = match &dag[*current_node] {
                        NodeType::Operation(instr) => instr,
                        _ => panic!("Unexpected type in commutation set run."),
                    };
                    let node_op_name = node_op.op.name();

                    let (node_angle, phase_shift) = if ROTATION_GATES.contains(&node_op_name) {
                        let node_angle = match node_op.params_view().first() {
                            Some(Param::Float(f)) => *f,
                            _ => {
                                return Err(CommutationCancelError::RotationalParameter(
                                    node_op.op.clone(),
                                ));
                            }
                        };
                        let phase_shift = z_phase_shift(node_op_name, node_angle);
                        Ok((node_angle, phase_shift))
                    } else {
                        match node_op_name {
                            "t" => Ok((FRAC_PI_4, z_phase_shift("p", FRAC_PI_4))),
                            "tdg" => Ok((-FRAC_PI_4, -z_phase_shift("p", FRAC_PI_4))),
                            "s" => Ok((FRAC_PI_2, z_phase_shift("p", FRAC_PI_2))),
                            "sdg" => Ok((-FRAC_PI_2, -z_phase_shift("p", FRAC_PI_2))),
                            "z" => Ok((PI, z_phase_shift("p", PI))),
                            "x" => Ok((PI, FRAC_PI_2)),
                            "sx" => Ok((FRAC_PI_2, FRAC_PI_4)),
                            "sxdg" => Ok((-FRAC_PI_2, -FRAC_PI_4)),
                            _ => Err(CommutationCancelError::OperationUndefinedAngle(
                                node_op_name.to_owned(),
                            )),
                        }
                    }?;
                    total_angle += node_angle;
                    total_phase += phase_shift;
                }

                if is_multiple_of_pi(total_angle, 4.) {
                    // if the angle is close to a 4-pi multiple (from above or below), then the
                    // operator is equal to the identity
                } else if is_multiple_of_pi(total_angle, 2.) {
                    if cancel_key.gate == GateOrRotation::XRotation
                        || matches!(z_var_gate, Some(&StandardGate::RZ))
                    {
                        // a 2-pi multiple has a phase of pi: RX(2pi) = RZ(2pi) = -I = I exp(i pi)
                        total_phase -= PI;
                    }
                } else if cancel_key.gate == GateOrRotation::ZRotation {
                    let z_gate = z_var_gate.unwrap();
                    dag.insert_1q_on_incoming_qubit((*z_gate, &[total_angle]), cancel_set[0]);
                } else {
                    // cancel_gate.key is either ZRotation or XRotation in this block it is an
                    // XRotation.
                    if x_supported && is_multiple_of_pi(total_angle, 1.) {
                        let num_x = (total_angle / PI).round();
                        total_phase -= FRAC_PI_2 * num_x;
                        dag.insert_1q_on_incoming_qubit((StandardGate::X, &[]), cancel_set[0]);
                    } else if sx_supported && is_multiple_of_pi(total_angle, 0.5) {
                        let num_sx = (total_angle / FRAC_PI_2).round();
                        total_phase -= FRAC_PI_4 * num_sx;
                        for _ in 0..(num_sx as i64) % 4 {
                            dag.insert_1q_on_incoming_qubit((StandardGate::SX, &[]), cancel_set[0]);
                        }
                    } else {
                        dag.insert_1q_on_incoming_qubit(
                            (StandardGate::RX, &[total_angle]),
                            cancel_set[0],
                        );
                    }
                }

                dag.add_global_phase(&Param::Float(total_phase))?;

                for node in cancel_set {
                    dag.remove_op_node(*node);
                }
            }
        }
    }

    Ok(())
}

/// Checks if angle is an integer multiple of ``factor * PI``.
fn is_multiple_of_pi(angle: f64, factor: f64) -> bool {
    let modulo = angle / (factor * PI);
    let remainder = modulo.rem_euclid(1.0);
    abs_diff_eq!(remainder, 0., epsilon = _CUTOFF_PRECISION)
        || abs_diff_eq!(remainder, 1., epsilon = _CUTOFF_PRECISION)
}

pub fn commutation_cancellation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_cancel_commutations))?;
    Ok(())
}
