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

use crate::clifford::greedy_synthesis::resynthesize_clifford_circuit;
use crate::evolution::EvolutionSynthesisError;

use smallvec::{SmallVec, smallvec};

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate, multiply_param, radd_param};

use rustiq_core::structures::{
    CliffordCircuit, CliffordGate, IsometryTableau, Metric, PauliLike, PauliSet,
};
use rustiq_core::synthesis::clifford::isometry::isometry_synthesis;
use rustiq_core::synthesis::pauli_network::greedy_pauli_network;

use rustworkx_core::petgraph::Incoming;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::prelude::StableDiGraph;

/// A Qiskit gate. The quantum circuit data returned by the pauli network
/// synthesis algorithm will consist of clifford and rotation gates.
type QiskitGate = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);

/// Return the Qiskit's gate corresponding to the given Rustiq's Clifford gate.
fn to_qiskit_clifford_gate(rustiq_gate: &CliffordGate) -> QiskitGate {
    match rustiq_gate {
        CliffordGate::CNOT(i, j) => (
            StandardGate::CX,
            smallvec![],
            smallvec![Qubit(*i as u32), Qubit(*j as u32)],
        ),
        CliffordGate::CZ(i, j) => (
            StandardGate::CZ,
            smallvec![],
            smallvec![Qubit(*i as u32), Qubit(*j as u32)],
        ),
        CliffordGate::H(i) => (StandardGate::H, smallvec![], smallvec![Qubit(*i as u32)]),
        CliffordGate::S(i) => (StandardGate::S, smallvec![], smallvec![Qubit(*i as u32)]),
        CliffordGate::Sd(i) => (StandardGate::Sdg, smallvec![], smallvec![Qubit(*i as u32)]),
        CliffordGate::SqrtX(i) => (StandardGate::SX, smallvec![], smallvec![Qubit(*i as u32)]),
        CliffordGate::SqrtXd(i) => (StandardGate::SXdg, smallvec![], smallvec![Qubit(*i as u32)]),
    }
}

/// Return the Qiskit rotation gate corresponding to the single-qubit Pauli rotation.
///
/// # Arguments
///
/// * py: a GIL handle, needed to negate rotation parameters in Python space.
/// * paulis: Rustiq's data structure storing pauli rotations.
/// * i: index of the single-qubit Pauli rotation.
/// * angle: Qiskit's rotation angle.
fn qiskit_rotation_gate(paulis: &PauliSet, i: usize, angle: &Param) -> QiskitGate {
    let (phase, pauli_str) = paulis.get(i);
    for (q, c) in pauli_str.chars().enumerate() {
        if c != 'I' {
            let standard_gate = match c {
                'X' => StandardGate::RX,
                'Y' => StandardGate::RY,
                'Z' => StandardGate::RZ,
                _ => unreachable!("Only X, Y and Z are possible characters at this point."),
            };
            // We need to negate the angle when there is a phase.
            let param = match phase {
                false => angle.clone(),
                true => multiply_param(angle, -1.0),
            };
            return (standard_gate, smallvec![param], smallvec![Qubit(q as u32)]);
        }
    }
    unreachable!("The pauli rotation is guaranteed to be a single-qubit rotation.")
}

// Note:
// The Pauli network synthesis algorithm in rustiq-core 0.0.8 only returns
// the list of Clifford gates that, when simulated, turn every Pauli rotation
// at some point to a single-qubit Pauli rotation. As an additional complication,
// the order in which the Pauli rotations are turned into single-qubit Pauli
// rotations coincides with the original order only up to commutativity between
// Pauli rotations.
// As a temporary solution, we follow the approach in Simon's private rustiq-plugin
// repository: we simulate the original Pauli network using returned Clifford gates
// to find where Pauli rotations need to be inserted, and we keep a DAG representing
// commutativity relations between Pauli rotations to make sure the single-qubit
// rotations are chosen in the correct order.
// In the future we are planning to extend the algorithm in rustiq-core to
// explicitly return the circuit with single-qubit Pauli rotations already inserted.
// When this happens, we will be able to significantly simplify the code that follows.

/// A DAG that stores ordered Paulis, up to commutativity.
struct CommutativityDag {
    /// Rustworkx's DAG
    dag: StableDiGraph<usize, ()>,
}

impl CommutativityDag {
    /// Construct a DAG corresponding to `paulis`.
    /// When `add_edges` is `true`, we add an edge between pauli `i` and pauli `j`
    /// iff they commute. When `add_edges` is `false`, we do not add any edges.
    fn from_paulis(paulis: &PauliSet, add_edges: bool) -> Self {
        let mut dag = StableDiGraph::<usize, ()>::new();

        let node_indices: Vec<NodeIndex> = (0..paulis.len()).map(|i| dag.add_node(i)).collect();

        if add_edges {
            for i in 0..paulis.len() {
                let pauli_i = paulis.get_as_pauli(i);
                for j in i + 1..paulis.len() {
                    let pauli_j = paulis.get_as_pauli(j);

                    if !pauli_i.commutes(&pauli_j) {
                        dag.add_edge(node_indices[i], node_indices[j], ());
                    }
                }
            }
        }

        CommutativityDag { dag }
    }

    /// Return whether the given node is a front node (i.e. has no predecessors).
    fn is_front_node(&self, index: usize) -> bool {
        self.dag
            .neighbors_directed(NodeIndex::new(index), Incoming)
            .next()
            .is_none()
    }

    /// Remove node from the DAG.
    fn remove_node(&mut self, index: usize) {
        self.dag.remove_node(NodeIndex::new(index));
    }
}

/// Return a Qiskit circuit with Clifford gates and rotations.
///
/// The rotations are assumed to be ordered (up to commutativity).
///
/// # Arguments
///
/// * py: a GIL handle, needed to negate rotation parameters in Python space.
/// * gates: the sequence of Rustiq's Clifford gates returned by Rustiq's
///   pauli network synthesis algorithm.
/// * paulis: Rustiq's data structure storing the pauli rotations.
/// * angles: Qiskit's rotation angles corresponding to the pauli rotations.
/// * preserve_order: specifies whether the order of paulis should be preserved,
///   up to commutativity.
fn inject_rotations(
    gates: &[CliffordGate],
    paulis: &PauliSet,
    angles: &[Param],
    preserve_order: bool,
) -> (Vec<QiskitGate>, Param) {
    let mut out_gates: Vec<QiskitGate> = Vec::with_capacity(gates.len() + paulis.len());
    let mut global_phase = Param::Float(0.0);

    let mut cur_paulis = paulis.clone();
    let mut hit_paulis: Vec<bool> = vec![false; cur_paulis.len()];

    let mut dag = CommutativityDag::from_paulis(paulis, preserve_order);

    // check which paulis are hit at the very start
    for i in 0..cur_paulis.len() {
        let pauli_support_size = cur_paulis.support_size(i);
        if pauli_support_size == 0 {
            // in case of an all-identity rotation, update global phase by subtracting
            // the angle
            global_phase = radd_param(global_phase, multiply_param(&angles[i], -0.5));
            hit_paulis[i] = true;
            dag.remove_node(i);
        } else if pauli_support_size == 1 && dag.is_front_node(i) {
            out_gates.push(qiskit_rotation_gate(&cur_paulis, i, &angles[i]));
            hit_paulis[i] = true;
            dag.remove_node(i);
        }
    }

    for gate in gates {
        out_gates.push(to_qiskit_clifford_gate(gate));

        cur_paulis.conjugate_with_gate(gate);

        // check which paulis are hit now
        for i in 0..cur_paulis.len() {
            if !hit_paulis[i] && cur_paulis.support_size(i) == 1 && dag.is_front_node(i) {
                out_gates.push(qiskit_rotation_gate(&cur_paulis, i, &angles[i]));
                hit_paulis[i] = true;
                dag.remove_node(i);
            }
        }
    }

    (out_gates, global_phase)
}

/// Return the vector of Qiskit's gate corresponding to the given vector
/// of Rustiq's Clifford gate.
fn to_qiskit_clifford_gates(gates: &Vec<CliffordGate>) -> Vec<QiskitGate> {
    let mut qiskit_gates: Vec<QiskitGate> = Vec::with_capacity(gates.len());
    for gate in gates {
        qiskit_gates.push(to_qiskit_clifford_gate(gate));
    }
    qiskit_gates
}

/// Returns the number of CNOTs.
fn cnot_count(qgates: &[QiskitGate]) -> usize {
    qgates.iter().filter(|&gate| gate.2.len() == 2).count()
}

/// Given the Clifford circuit returned by Rustiq's pauli network synthesis algorithm,
/// generates a sequence of Qiskit gates that implements this circuit.
/// If `fix_clifford_method` is `0`, the original circuit is inverted; if `1`, it is
/// resynthesized using Qiskit; and if `2` it is resynthesized using Rustiq.
fn synthesize_final_clifford(
    rcircuit: &CliffordCircuit,
    resynth_clifford_method: usize,
) -> Vec<QiskitGate> {
    match resynth_clifford_method {
        0 => to_qiskit_clifford_gates(&rcircuit.gates),
        1 => {
            // Qiskit-based resynthesis
            let qcircuit = to_qiskit_clifford_gates(&rcircuit.gates);
            let new_qcircuit = resynthesize_clifford_circuit(rcircuit.nqbits, &qcircuit).unwrap();
            if cnot_count(&qcircuit) < cnot_count(&new_qcircuit) {
                qcircuit
            } else {
                new_qcircuit
            }
        }
        _ => {
            // Rustiq-based resynthesis
            let tableau = IsometryTableau::new(rcircuit.nqbits, 0);
            let new_rcircuit = isometry_synthesis(&tableau, &Metric::COUNT, 1);
            if new_rcircuit.cnot_count() < rcircuit.cnot_count() {
                to_qiskit_clifford_gates(&new_rcircuit.gates)
            } else {
                to_qiskit_clifford_gates(&rcircuit.gates)
            }
        }
    }
}

/// Calls Rustiq's pauli network synthesis algorithm and returns the
/// Qiskit circuit data with Clifford gates and rotations.
///
/// # Arguments
///
/// * py: a GIL handle, needed to add and negate rotation parameters in Python space.
/// * num_qubits: total number of qubits.
/// * pauli_network: pauli network represented in sparse format. It's a list
///   of triples such as `[("XX", [0, 3], theta), ("ZZ", [0, 1], 0.1)]`.
/// * optimize_count: if `true`, Rustiq's synthesis algorithms aims to optimize
///   the 2-qubit gate count; and if `false`, then the 2-qubit depth.
/// * preserve_order: whether the order of paulis should be preserved, up to
///   commutativity. If the order is not preserved, the returned circuit will
///   generally not be equivalent to the given pauli network.
/// * upto_clifford: if `true`, the final Clifford operator is not synthesized
///   and the returned circuit will generally not be equivalent to the given
///   pauli network. In addition, the argument `upto_phase` would be ignored.
/// * upto_phase: if `true`, the global phase of the returned circuit may differ
///   from the global phase of the given pauli network. The argument is considered
///   to be `true` when `upto_clifford` is `true`.
/// * resynth_clifford_method: describes the strategy to synthesize the final
///   Clifford operator. If `0` a naive approach is used, which doubles the number
///   of gates but preserves the global phase of the circuit. If `1`, the Clifford is
///   resynthesized using Qiskit's greedy Clifford synthesis algorithm. If `2`, it
///   is resynthesized by Rustiq itself. If `upto_phase` is `false`, the naive
///   approach is used, as neither synthesis method preserves the global phase.
///
/// If `preserve_order` is `true` and both `upto_clifford` and `upto_phase` are `false`,
/// the returned circuit is equivalent to the given pauli network.
#[allow(clippy::too_many_arguments)]
pub fn pauli_network_synthesis_inner(
    num_qubits: usize,
    paulis: Vec<String>,
    angles: Vec<Param>,
    optimize_count: bool,
    preserve_order: bool,
    upto_clifford: bool,
    upto_phase: bool,
    resynth_clifford_method: usize,
) -> Result<CircuitData, EvolutionSynthesisError> {
    let paulis = PauliSet::from_slice(&paulis);
    let metric = match optimize_count {
        true => Metric::COUNT,
        false => Metric::DEPTH,
    };

    // Call Rustiq's synthesis algorithm
    let circuit = greedy_pauli_network(&paulis, &metric, preserve_order, 0, false, false);

    // post-process algorithm's output, translating to Qiskit's gates and inserting rotation gates
    let (mut gates, global_phase) =
        inject_rotations(&circuit.gates, &paulis, &angles, preserve_order);

    // if the circuit needs to be synthesized exactly, we cannot use either Rustiq's
    // or Qiskit's synthesis methods for Cliffords, since they do not necessarily preserve
    // the global phase.
    let resynth_clifford_method = match upto_phase {
        true => resynth_clifford_method,
        false => 0,
    };

    // synthesize the final Clifford
    if !upto_clifford {
        let final_clifford = synthesize_final_clifford(&circuit.dagger(), resynth_clifford_method);
        for gate in final_clifford {
            gates.push(gate);
        }
    }

    Ok(CircuitData::from_standard_gates(
        num_qubits as u32,
        gates,
        global_phase,
    )?)
}
