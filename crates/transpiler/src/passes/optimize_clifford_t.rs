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

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::f64::consts::PI;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, PyDAGCircuit};
use qiskit_circuit::operations::{OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

// The Clifford+T optimization pass only applies to circuits with Clifford+T/Tdg gates.
// We return a transpiler error when the circuit contains gates outide of the following
// list:
pub static CLIFFORD_T_GATE_NAMES: [&str; 18] = [
    "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr",
    "dcx", "t", "tdg",
];

// We need to reason about RX, RY, RZ rotations by angles pi/8 and -pi/8. We will
// represent such rotations as a pair (rotation axis, sign).
// In particular, a T-gate corresponds to RZ(pi/8) rotation, with the appropriate
// adjustment to the global phase, or with the above notation to (Pauli1q::Z, false).
#[derive(Clone, Copy, PartialEq)]
enum Pauli1q {
    X,
    Y,
    Z,
}

// We need to reason about operators that can be constructed using 1q Clifford gates,
// including the precise value of the global phase. We have a total of 192 = 24 x 8
// such objects, with 24 possible 1q-Clifford objects and 8 possible multiples of the
// global phase. The exact correspondence between the index of the Clifford and the
// corresponding 1q Clifford circuit is contained in the slice called CIRCUIT.
struct Clifford1q {
    idx: u8,
    // enumerates single-qubit Cliffords: takes values in 0..24
    w: u8,
    // global phase factor of the form w * pi * i / 4, w takes values in 0..8
}

impl Clifford1q {
    // Represents the identity operator.
    fn identity() -> Self {
        Self { idx: 0, w: 0 }
    }

    // In-place modification of the operator corresponding to adding a Clifford
    // gate after the current operator. For the implementation, we have precomputed
    // the effect of appending S and H gates, and express the remaining 1q Clifford
    // gates in terms of these.
    fn append_clifford_gate(&mut self, gate: StandardGate) {
        match gate {
            StandardGate::I => {}
            StandardGate::H => {
                let (new_idx, w_update) = APPEND_H[self.idx as usize];
                (self.idx, self.w) = (new_idx, (self.w + w_update) % 8);
            }
            StandardGate::S => {
                let (new_idx, w_update) = APPEND_S[self.idx as usize];
                (self.idx, self.w) = (new_idx, (self.w + w_update) % 8);
            }
            StandardGate::Sdg => {
                self.append_clifford_gate(StandardGate::S);
                self.append_clifford_gate(StandardGate::S);
                self.append_clifford_gate(StandardGate::S);
            }
            StandardGate::Z => {
                self.append_clifford_gate(StandardGate::S);
                self.append_clifford_gate(StandardGate::S);
            }
            StandardGate::SX => {
                self.append_clifford_gate(StandardGate::H);
                self.append_clifford_gate(StandardGate::S);
                self.append_clifford_gate(StandardGate::H);
            }
            StandardGate::SXdg => {
                self.append_clifford_gate(StandardGate::H);
                self.append_clifford_gate(StandardGate::Sdg);
                self.append_clifford_gate(StandardGate::H);
            }
            StandardGate::X => {
                self.append_clifford_gate(StandardGate::SX);
                self.append_clifford_gate(StandardGate::SX);
            }
            StandardGate::Y => {
                self.append_clifford_gate(StandardGate::X);
                self.append_clifford_gate(StandardGate::Z);
                self.w = (self.w + 6) % 8;
            }
            _ => unreachable!("should not be here, gate = {:?}", gate),
        }
    }

    // In-place modification of the operator corresponding to adding a Clifford
    // gate before the current operator. For the implementation, we have precomputed
    // the effect of prepending S and H gates, and express the remaining 1q Clifford
    // gates in terms of these.
    fn prepend_clifford_gate(&mut self, gate: StandardGate) {
        match gate {
            StandardGate::I => {}
            StandardGate::H => {
                let (new_idx, w_update) = PREPEND_H[self.idx as usize];
                (self.idx, self.w) = (new_idx, (self.w + w_update) % 8);
            }
            StandardGate::S => {
                let (new_idx, w_update) = PREPEND_S[self.idx as usize];
                (self.idx, self.w) = (new_idx, (self.w + w_update) % 8);
            }
            StandardGate::Sdg => {
                self.prepend_clifford_gate(StandardGate::S);
                self.prepend_clifford_gate(StandardGate::S);
                self.prepend_clifford_gate(StandardGate::S);
            }
            StandardGate::Z => {
                self.prepend_clifford_gate(StandardGate::S);
                self.prepend_clifford_gate(StandardGate::S);
            }
            StandardGate::SX => {
                self.prepend_clifford_gate(StandardGate::H);
                self.prepend_clifford_gate(StandardGate::S);
                self.prepend_clifford_gate(StandardGate::H);
            }
            StandardGate::SXdg => {
                self.prepend_clifford_gate(StandardGate::H);
                self.prepend_clifford_gate(StandardGate::Sdg);
                self.prepend_clifford_gate(StandardGate::H);
            }
            StandardGate::X => {
                self.prepend_clifford_gate(StandardGate::SX);
                self.prepend_clifford_gate(StandardGate::SX);
            }
            StandardGate::Y => {
                self.prepend_clifford_gate(StandardGate::Z);
                self.prepend_clifford_gate(StandardGate::X);
                self.w = (self.w + 6) % 8;
            }
            _ => unreachable!("should not be here, gate = {:?}", gate),
        }
    }

    // Evolves a rotation using the Clifford. Returns the new rotation
    // including the sign.
    fn evolve_pauli(&self, pauli: Pauli1q, sign: bool) -> (Pauli1q, bool) {
        let (new_pauli, new_sign) = match pauli {
            Pauli1q::X => EVOLVE_X[self.idx as usize],
            Pauli1q::Y => EVOLVE_Y[self.idx as usize],
            Pauli1q::Z => EVOLVE_Z[self.idx as usize],
        };
        (new_pauli, sign ^ new_sign)
    }

    /// Returns the corresponding Clifford circuit.
    fn to_circuit(&self) -> (&[StandardGate], f64) {
        let circuit = CIRCUIT[self.idx as usize];
        let phase: f64 = (self.w as f64) * PI / 4.;
        (circuit, phase)
    }
}

/// Attempts to optimize a sequence of consecutive 1-qubit Clifford and T/Tdg gates.
/// Returns `None` if the sequence can't be optimized.
fn optimize_clifford_t_1q(
    dag: &DAGCircuit,
    raw_run: &[NodeIndex],
    basis_gates: Option<&[StandardGate]>,
) -> Option<(Vec<StandardGate>, f64)> {
    let mut is_reduced = false; // was any reduction applied?
    let mut clifford1q = Clifford1q::identity(); // current 1q-clifford operator
    let mut rotations: Vec<(Pauli1q, bool)> = Vec::new(); // current list of rotations
    let mut global_phase: f64 = 0.; // current adjustment to the global phase
    let num_nodes = raw_run.len();

    let apply_ty = if basis_gates.is_some_and(|gates| !gates.contains(&StandardGate::SX)) {
        |(sequence, flip): (&mut Vec<StandardGate>, bool)| {
            sequence.push(StandardGate::Sdg);
            sequence.push(StandardGate::H);
            sequence.push(if flip {
                StandardGate::Tdg
            } else {
                StandardGate::T
            });
            sequence.push(StandardGate::H);
            sequence.push(StandardGate::S);
        }
    } else {
        |(sequence, flip): (&mut Vec<StandardGate>, bool)| {
            sequence.push(StandardGate::SX);
            sequence.push(if flip {
                StandardGate::Tdg
            } else {
                StandardGate::T
            });
            sequence.push(StandardGate::SXdg);
        }
    };

    for idx in 0..num_nodes {
        let cur_node = &dag[raw_run[idx]];
        let cur_gate = if let NodeType::Operation(inst) = cur_node {
            if let OperationRef::StandardGate(gate) = inst.op.view() {
                gate
            } else {
                unreachable!("Can only have Clifford+T gates at this point");
            }
        } else {
            unreachable!("Can only have op nodes here")
        };

        // Process the next gate. T and Tdg gates are replaced by RZ-rotations with the appropriate
        // adjustment of the global phase. Clifford gates are merged into the current clifford operator.
        if cur_gate == StandardGate::T {
            global_phase += PI / 8.;
            let evolved_rotation: (Pauli1q, bool) = clifford1q.evolve_pauli(Pauli1q::Z, false);
            rotations.push(evolved_rotation);
        } else if cur_gate == StandardGate::Tdg {
            global_phase -= PI / 8.;
            let evolved_rotation: (Pauli1q, bool) = clifford1q.evolve_pauli(Pauli1q::Z, true);
            rotations.push(evolved_rotation);
        } else {
            clifford1q.append_clifford_gate(cur_gate);
            continue;
        }

        // Check if the new rotation can be canceled or merged with the previous rotation,
        // which happens exactly when they have the same rotation axis.
        if rotations.len() >= 2 {
            let (pauli2, sign2) = rotations[rotations.len() - 2];
            let (pauli1, sign1) = rotations[rotations.len() - 1];

            if pauli1 == pauli2 {
                is_reduced = true;

                rotations.pop();
                rotations.pop();

                if sign1 != sign2 {
                    // nothing more to do
                } else {
                    match (pauli1, sign1) {
                        (Pauli1q::Z, false) => {
                            global_phase -= PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::S);
                        }
                        (Pauli1q::Z, true) => {
                            global_phase += PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::Sdg);
                        }
                        (Pauli1q::X, false) => {
                            global_phase -= PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::SX);
                        }
                        (Pauli1q::X, true) => {
                            global_phase += PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::SXdg);
                        }
                        (Pauli1q::Y, false) => {
                            global_phase -= PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::SXdg);
                            clifford1q.prepend_clifford_gate(StandardGate::S);
                            clifford1q.prepend_clifford_gate(StandardGate::SX);
                        }
                        (Pauli1q::Y, true) => {
                            global_phase += PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::SXdg);
                            clifford1q.prepend_clifford_gate(StandardGate::Sdg);
                            clifford1q.prepend_clifford_gate(StandardGate::SX);
                        }
                    }
                }
            }
        }
    }

    let mut optimized_sequence = Vec::<StandardGate>::with_capacity(num_nodes);

    // Insert rotations, converting them to Clifford+T/Tdg gates, while appropriately updating the
    // global phase.
    for (pauli, sign) in rotations {
        match (pauli, sign) {
            (Pauli1q::X, false) => {
                optimized_sequence.push(StandardGate::H);
                optimized_sequence.push(StandardGate::T);
                optimized_sequence.push(StandardGate::H);
                global_phase -= PI / 8.;
            }
            (Pauli1q::X, true) => {
                optimized_sequence.push(StandardGate::H);
                optimized_sequence.push(StandardGate::Tdg);
                optimized_sequence.push(StandardGate::H);
                global_phase += PI / 8.;
            }
            (Pauli1q::Y, false) => {
                apply_ty((&mut optimized_sequence, false));
                global_phase -= PI / 8.;
            }
            (Pauli1q::Y, true) => {
                apply_ty((&mut optimized_sequence, true));
                global_phase += PI / 8.;
            }
            (Pauli1q::Z, false) => {
                optimized_sequence.push(StandardGate::T);
                global_phase -= PI / 8.;
            }
            (Pauli1q::Z, true) => {
                optimized_sequence.push(StandardGate::Tdg);
                global_phase += PI / 8.;
            }
        }
    }

    // Insert the remaining Clifford gates
    let (clifford_gates, phase_update) = clifford1q.to_circuit();
    for gate in clifford_gates {
        optimized_sequence.push(*gate);
    }
    global_phase += phase_update;

    // Either we have fewer T-gates, or the number of T-gates remains the same
    // but we have fewer Clifford gates (as the total number of gates is the number
    // of T-gates + the number of Clifford gates).
    (is_reduced || optimized_sequence.len() < raw_run.len())
        .then_some((optimized_sequence, global_phase))
}

#[pyfunction]
#[pyo3(name = "optimize_clifford_t", signature = (dag, basis_gates=None))]
pub fn py_run_optimize_clifford_t(
    dag: &mut PyDAGCircuit,
    basis_gates: Option<Vec<StandardGate>>,
) -> PyResult<()> {
    run_optimize_clifford_t(dag.try_write()?, basis_gates)
}

pub fn run_optimize_clifford_t(
    dag: &mut DAGCircuit,
    basis_gates: Option<Vec<StandardGate>>,
) -> PyResult<()> {
    let filter = |inst: &PackedInstruction| -> bool {
        matches!(
            inst.op.view(),
            OperationRef::StandardGate(
                StandardGate::I
                    | StandardGate::X
                    | StandardGate::Y
                    | StandardGate::Z
                    | StandardGate::H
                    | StandardGate::S
                    | StandardGate::Sdg
                    | StandardGate::SX
                    | StandardGate::SXdg
                    | StandardGate::T
                    | StandardGate::Tdg
            )
        )
    };

    let runs: Vec<Vec<NodeIndex>> = dag.collect_runs_by(filter).collect();

    for raw_run in runs {
        let optimized_sequence = optimize_clifford_t_1q(dag, &raw_run, basis_gates.as_deref());
        match optimized_sequence {
            Some((optimized_sequence, global_phase_update)) => {
                for gate in optimized_sequence {
                    dag.insert_1q_on_incoming_qubit((gate, &[]), raw_run[0]);
                }

                dag.add_global_phase(&Param::Float(global_phase_update))?;
                dag.remove_1q_sequence(&raw_run);
            }
            None => {
                // No reductions were performed.
            }
        }
    }
    Ok(())
}

pub fn optimize_clifford_t_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_optimize_clifford_t))?;
    Ok(())
}

// Precomputed tables used in the algorithm.

// Index of the Clifford1q operator -> corresponding Clifford circuit
static CIRCUIT: [&[StandardGate]; 24] = [
    &[],
    &[StandardGate::H],
    &[StandardGate::S],
    &[StandardGate::H, StandardGate::S],
    &[StandardGate::S, StandardGate::H],
    &[StandardGate::S, StandardGate::H, StandardGate::S],
    &[StandardGate::X],
    &[StandardGate::H, StandardGate::X],
    &[StandardGate::S, StandardGate::X],
    &[StandardGate::H, StandardGate::S, StandardGate::X],
    &[StandardGate::S, StandardGate::H, StandardGate::X],
    &[
        StandardGate::S,
        StandardGate::H,
        StandardGate::S,
        StandardGate::X,
    ],
    &[StandardGate::Z],
    &[StandardGate::H, StandardGate::Z],
    &[StandardGate::S, StandardGate::Z],
    &[StandardGate::H, StandardGate::S, StandardGate::Z],
    &[StandardGate::S, StandardGate::H, StandardGate::Z],
    &[
        StandardGate::S,
        StandardGate::H,
        StandardGate::S,
        StandardGate::Z,
    ],
    &[StandardGate::Y],
    &[StandardGate::H, StandardGate::Y],
    &[StandardGate::S, StandardGate::Y],
    &[StandardGate::H, StandardGate::S, StandardGate::Y],
    &[StandardGate::S, StandardGate::H, StandardGate::Y],
    &[
        StandardGate::S,
        StandardGate::H,
        StandardGate::S,
        StandardGate::Y,
    ],
];

// Index of the Clifford1q operator -> its effect on the X/Y/Z-rotations
// (new rotation axis + sign)
static EVOLVE_X: [(Pauli1q, bool); 24] = [
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::Y, true),
    (Pauli1q::Y, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::Y, true),
    (Pauli1q::Y, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::Y, false),
    (Pauli1q::Y, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::Y, false),
    (Pauli1q::Y, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
];
static EVOLVE_Y: [(Pauli1q, bool); 24] = [
    (Pauli1q::Y, false),
    (Pauli1q::Y, true),
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, true),
    (Pauli1q::Z, false),
    (Pauli1q::Y, true),
    (Pauli1q::Y, false),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, false),
    (Pauli1q::Z, true),
    (Pauli1q::Y, true),
    (Pauli1q::Y, false),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, false),
    (Pauli1q::Z, true),
    (Pauli1q::Y, false),
    (Pauli1q::Y, true),
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, true),
    (Pauli1q::Z, false),
];
static EVOLVE_Z: [(Pauli1q, bool); 24] = [
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::Y, true),
    (Pauli1q::Y, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
    (Pauli1q::Y, false),
    (Pauli1q::Y, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::Y, true),
    (Pauli1q::Y, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
    (Pauli1q::Y, false),
    (Pauli1q::Y, false),
];

// Index of the Clifford1q operator -> changes when appending/prepending S/H-gates
// (index of the new operator + phase update)
static APPEND_S: [(u8, u8); 24] = [
    (2, 0),
    (3, 0),
    (12, 0),
    (13, 0),
    (5, 0),
    (16, 0),
    (20, 0),
    (21, 0),
    (6, 2),
    (7, 2),
    (23, 0),
    (10, 2),
    (14, 0),
    (15, 0),
    (0, 0),
    (1, 0),
    (17, 0),
    (4, 0),
    (8, 4),
    (9, 4),
    (18, 2),
    (19, 2),
    (11, 4),
    (22, 2),
];
static PREPEND_S: [(u8, u8); 24] = [
    (2, 0),
    (4, 0),
    (12, 0),
    (5, 0),
    (7, 0),
    (21, 0),
    (8, 0),
    (10, 0),
    (18, 6),
    (11, 0),
    (1, 0),
    (15, 2),
    (14, 0),
    (16, 0),
    (0, 0),
    (17, 0),
    (19, 2),
    (9, 6),
    (20, 0),
    (22, 0),
    (6, 2),
    (23, 0),
    (13, 6),
    (3, 0),
];
static APPEND_H: [(u8, u8); 24] = [
    (1, 0),
    (0, 0),
    (4, 0),
    (11, 7),
    (2, 0),
    (15, 1),
    (13, 0),
    (12, 0),
    (16, 0),
    (23, 1),
    (14, 0),
    (3, 1),
    (7, 0),
    (6, 0),
    (10, 0),
    (5, 7),
    (8, 0),
    (21, 7),
    (19, 4),
    (18, 4),
    (22, 4),
    (17, 1),
    (20, 4),
    (9, 7),
];
static PREPEND_H: [(u8, u8); 24] = [
    (1, 0),
    (0, 0),
    (3, 0),
    (2, 0),
    (11, 7),
    (10, 1),
    (7, 0),
    (6, 0),
    (9, 0),
    (8, 0),
    (5, 7),
    (4, 1),
    (13, 0),
    (12, 0),
    (15, 0),
    (14, 0),
    (23, 1),
    (22, 3),
    (19, 0),
    (18, 0),
    (21, 0),
    (20, 0),
    (17, 5),
    (16, 7),
];
