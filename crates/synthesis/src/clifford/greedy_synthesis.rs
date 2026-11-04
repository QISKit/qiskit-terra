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

use ndarray::{Array2, ArrayView2, s};
use qiskit_util::IndexSet;
use smallvec::smallvec;

use crate::clifford::utils::{
    CliffordGatesVec, SymplecticMatrix, adjust_final_pauli_gates, clifford_from_gate_sequence,
};

use qiskit_circuit::Qubit;
use qiskit_circuit::operations::StandardGate;

/// Converts a pair of Paulis pauli_x and pauli_z acting on a specific qubit
/// to the corresponding index in [PauliPairsClass] or [SingleQubitGate] classes.
/// The input is given as a 4-tuple: (pauli_x stabilizer, pauli_x destabilizer,
/// pauli_z stabilizer, pauli_z destabilizer), and the output is an unsigned
/// integer from 0 to 15.
fn pauli_pair_to_index(xs: bool, xd: bool, zs: bool, zd: bool) -> usize {
    ((xs as usize) << 3) | ((xd as usize) << 2) | ((zs as usize) << 1) | (zd as usize)
}

/// The five classes of Pauli 2-qubit operators as described in the paper.
#[derive(Clone, Copy)]
enum PauliPairsClass {
    ClassA,
    ClassB,
    ClassC,
    ClassD,
    ClassE,
}

/// The 16 Pauli 2-qubit operators are divided into 5 equivalence classes
/// under the action of single-qubit Cliffords.
static PAULI_INDEX_TO_CLASS: [PauliPairsClass; 16] = [
    PauliPairsClass::ClassE, // 'II'
    PauliPairsClass::ClassD, // 'IX'
    PauliPairsClass::ClassD, // 'IZ'
    PauliPairsClass::ClassD, // 'IY'
    PauliPairsClass::ClassC, // 'XI'
    PauliPairsClass::ClassB, // 'XX'
    PauliPairsClass::ClassA, // 'XZ'
    PauliPairsClass::ClassA, // 'XY'
    PauliPairsClass::ClassC, // 'ZI'
    PauliPairsClass::ClassA, // 'ZX'
    PauliPairsClass::ClassB, // 'ZZ'
    PauliPairsClass::ClassA, // 'ZY'
    PauliPairsClass::ClassC, // 'YI'
    PauliPairsClass::ClassA, // 'YX'
    PauliPairsClass::ClassA, // 'YZ'
    PauliPairsClass::ClassB, // 'YY'
];

/// Single-qubit Clifford gates modulo Paulis.
#[derive(Clone, Copy)]
enum SingleQubitGate {
    GateI,
    GateS,
    GateH,
    GateSH,
    GateHS,
    GateSHS,
}

/// Maps pair of pauli operators to the single-qubit gate required
/// for the decoupling step.
static PAULI_INDEX_TO_1Q_GATE: [SingleQubitGate; 16] = [
    SingleQubitGate::GateI,   // 'II'
    SingleQubitGate::GateH,   // 'IX'
    SingleQubitGate::GateI,   // 'IZ'
    SingleQubitGate::GateSH,  // 'IY'
    SingleQubitGate::GateI,   // 'XI'
    SingleQubitGate::GateI,   // 'XX'
    SingleQubitGate::GateI,   // 'XZ'
    SingleQubitGate::GateSHS, // 'XY'
    SingleQubitGate::GateH,   // 'ZI'
    SingleQubitGate::GateH,   // 'ZX'
    SingleQubitGate::GateH,   // 'ZZ'
    SingleQubitGate::GateSH,  // 'ZY'
    SingleQubitGate::GateS,   // 'YI'
    SingleQubitGate::GateHS,  // 'YX'
    SingleQubitGate::GateS,   // 'YZ'
    SingleQubitGate::GateS,   // 'YY'
];

pub struct GreedyCliffordSynthesis<'a> {
    /// The Clifford tableau to be synthesized.
    tableau: ArrayView2<'a, bool>,

    /// The total number of qubits.
    num_qubits: usize,

    /// Symplectic matrix being reduced.
    symplectic_matrix: SymplecticMatrix,

    /// Unprocessed qubits.
    unprocessed_qubits: IndexSet<usize>,
}

impl GreedyCliffordSynthesis<'_> {
    pub(crate) fn new(
        tableau: ArrayView2<'_, bool>,
    ) -> Result<GreedyCliffordSynthesis<'_>, String> {
        let tableau_shape = tableau.shape();
        if (tableau_shape[0] % 2 == 1) || (tableau_shape[1] != tableau_shape[0] + 1) {
            return Err("The shape of the Clifford tableau is invalid".to_string());
        }

        let num_qubits = tableau_shape[0] / 2;

        // We are going to modify symplectic_matrix in-place until it
        // becomes the identity.
        let symplectic_matrix = SymplecticMatrix {
            num_qubits,
            smat: tableau.slice(s![.., 0..2 * num_qubits]).to_owned(),
        };

        let unprocessed_qubits = (0..num_qubits).collect();

        Ok(GreedyCliffordSynthesis {
            tableau,
            num_qubits,
            symplectic_matrix,
            unprocessed_qubits,
        })
    }

    /// Computes the CX cost of decoupling the symplectic matrix on the
    /// given qubit.
    fn compute_cost(&self, qubit: usize) -> Result<usize, String> {
        let mut a_num = 0;
        let mut b_num = 0;
        let mut c_num = 0;
        let mut d_num = 0;

        let mut qubit_is_in_a = false;

        for q in &self.unprocessed_qubits {
            let pauli_pair_index = pauli_pair_to_index(
                self.symplectic_matrix.smat[[*q, qubit + self.num_qubits]],
                self.symplectic_matrix.smat[[*q + self.num_qubits, qubit + self.num_qubits]],
                self.symplectic_matrix.smat[[*q, qubit]],
                self.symplectic_matrix.smat[[*q + self.num_qubits, qubit]],
            );
            let pauli_class = PAULI_INDEX_TO_CLASS[pauli_pair_index];

            match pauli_class {
                PauliPairsClass::ClassA => {
                    a_num += 1;
                    if *q == qubit {
                        qubit_is_in_a = true;
                    }
                }
                PauliPairsClass::ClassB => {
                    b_num += 1;
                }
                PauliPairsClass::ClassC => {
                    c_num += 1;
                }
                PauliPairsClass::ClassD => {
                    d_num += 1;
                }
                PauliPairsClass::ClassE => {}
            }
        }

        if a_num % 2 == 0 {
            return Err("Symplectic Gaussian elimination failed.".to_string());
        }

        let mut cnot_cost: usize =
            3 * (a_num - 1) / 2 + (b_num + 1) * ((b_num > 0) as usize) + c_num + d_num;

        if !qubit_is_in_a {
            cnot_cost += 3;
        }

        Ok(cnot_cost)
    }

    /// Calculate a decoupling operator D:
    /// D^{-1} * Ox * D = x1
    /// D^{-1} * Oz * D = z1
    /// and reduces the clifford such that it will act trivially on min_qubit.
    fn decouple_qubit(
        &mut self,
        gate_seq: &mut CliffordGatesVec,
        min_qubit: usize,
    ) -> Result<(), String> {
        let mut a_qubits: IndexSet<_> = IndexSet::default();
        let mut b_qubits: IndexSet<_> = IndexSet::default();
        let mut c_qubits: IndexSet<_> = IndexSet::default();
        let mut d_qubits: IndexSet<_> = IndexSet::default();

        for qubit in &self.unprocessed_qubits {
            let pauli_pair_index = pauli_pair_to_index(
                self.symplectic_matrix.smat[[*qubit, min_qubit + self.num_qubits]],
                self.symplectic_matrix.smat
                    [[*qubit + self.num_qubits, min_qubit + self.num_qubits]],
                self.symplectic_matrix.smat[[*qubit, min_qubit]],
                self.symplectic_matrix.smat[[*qubit + self.num_qubits, min_qubit]],
            );

            let single_qubit_gate = PAULI_INDEX_TO_1Q_GATE[pauli_pair_index];
            match single_qubit_gate {
                SingleQubitGate::GateS => {
                    gate_seq.push((StandardGate::S, smallvec![], smallvec![Qubit::new(*qubit)]));
                    self.symplectic_matrix.prepend_s(*qubit);
                }
                SingleQubitGate::GateH => {
                    gate_seq.push((StandardGate::H, smallvec![], smallvec![Qubit::new(*qubit)]));
                    self.symplectic_matrix.prepend_h(*qubit);
                }
                SingleQubitGate::GateSH => {
                    gate_seq.push((StandardGate::S, smallvec![], smallvec![Qubit::new(*qubit)]));
                    gate_seq.push((StandardGate::H, smallvec![], smallvec![Qubit::new(*qubit)]));
                    self.symplectic_matrix.prepend_s(*qubit);
                    self.symplectic_matrix.prepend_h(*qubit);
                }
                SingleQubitGate::GateHS => {
                    gate_seq.push((StandardGate::H, smallvec![], smallvec![Qubit::new(*qubit)]));
                    gate_seq.push((StandardGate::S, smallvec![], smallvec![Qubit::new(*qubit)]));
                    self.symplectic_matrix.prepend_h(*qubit);
                    self.symplectic_matrix.prepend_s(*qubit);
                }
                SingleQubitGate::GateSHS => {
                    gate_seq.push((StandardGate::S, smallvec![], smallvec![Qubit::new(*qubit)]));
                    gate_seq.push((StandardGate::H, smallvec![], smallvec![Qubit::new(*qubit)]));
                    gate_seq.push((StandardGate::S, smallvec![], smallvec![Qubit::new(*qubit)]));
                    self.symplectic_matrix.prepend_s(*qubit);
                    self.symplectic_matrix.prepend_h(*qubit);
                    self.symplectic_matrix.prepend_s(*qubit);
                }
                SingleQubitGate::GateI => {}
            }

            let pauli_class = PAULI_INDEX_TO_CLASS[pauli_pair_index];
            match pauli_class {
                PauliPairsClass::ClassA => {
                    a_qubits.insert(*qubit);
                }
                PauliPairsClass::ClassB => {
                    b_qubits.insert(*qubit);
                }
                PauliPairsClass::ClassC => {
                    c_qubits.insert(*qubit);
                }
                PauliPairsClass::ClassD => {
                    d_qubits.insert(*qubit);
                }
                PauliPairsClass::ClassE => {}
            }
        }

        if a_qubits.len() % 2 != 1 {
            return Err("Symplectic Gaussian elimination failed.".to_string());
        }

        if !a_qubits.contains(&min_qubit) {
            let qubit_a = a_qubits[0];
            gate_seq.push((
                StandardGate::Swap,
                smallvec![],
                smallvec![Qubit::new(min_qubit), Qubit::new(qubit_a)],
            ));
            self.symplectic_matrix.prepend_swap(min_qubit, qubit_a);

            if b_qubits.contains(&min_qubit) {
                b_qubits.swap_remove(&min_qubit);
                b_qubits.insert(qubit_a);
            } else if c_qubits.contains(&min_qubit) {
                c_qubits.swap_remove(&min_qubit);
                c_qubits.insert(qubit_a);
            } else if d_qubits.contains(&min_qubit) {
                d_qubits.swap_remove(&min_qubit);
                d_qubits.insert(qubit_a);
            }

            a_qubits.swap_remove(&qubit_a);
            a_qubits.insert(min_qubit);
        }

        for qubit in c_qubits {
            gate_seq.push((
                StandardGate::CX,
                smallvec![],
                smallvec![Qubit::new(min_qubit), Qubit::new(qubit)],
            ));
            self.symplectic_matrix.prepend_cx(min_qubit, qubit);
        }

        for qubit in d_qubits {
            gate_seq.push((
                StandardGate::CX,
                smallvec![],
                smallvec![Qubit::new(qubit), Qubit::new(min_qubit)],
            ));
            self.symplectic_matrix.prepend_cx(qubit, min_qubit);
        }

        if b_qubits.len() > 1 {
            let qubit_b = b_qubits[0];
            for qubit in &b_qubits[1..] {
                gate_seq.push((
                    StandardGate::CX,
                    smallvec![],
                    smallvec![Qubit::new(qubit_b), Qubit::new(*qubit)],
                ));
                self.symplectic_matrix.prepend_cx(qubit_b, *qubit);
            }
        }

        if !b_qubits.is_empty() {
            let qubit_b = b_qubits[0];
            gate_seq.push((
                StandardGate::CX,
                smallvec![],
                smallvec![Qubit::new(min_qubit), Qubit::new(qubit_b)],
            ));
            self.symplectic_matrix.prepend_cx(min_qubit, qubit_b);

            gate_seq.push((StandardGate::H, smallvec![], smallvec![Qubit::new(qubit_b)]));
            self.symplectic_matrix.prepend_h(qubit_b);

            gate_seq.push((
                StandardGate::CX,
                smallvec![],
                smallvec![Qubit::new(qubit_b), Qubit::new(min_qubit)],
            ));
            self.symplectic_matrix.prepend_cx(qubit_b, min_qubit);
        }

        let a_len: usize = (a_qubits.len() - 1) / 2;
        if a_len > 0 {
            a_qubits.swap_remove(&min_qubit);
        }

        for qubit in 0..a_len {
            gate_seq.push((
                StandardGate::CX,
                smallvec![],
                smallvec![
                    Qubit::new(a_qubits[2 * qubit + 1]),
                    Qubit::new(a_qubits[2 * qubit])
                ],
            ));
            self.symplectic_matrix
                .prepend_cx(a_qubits[2 * qubit + 1], a_qubits[2 * qubit]);

            gate_seq.push((
                StandardGate::CX,
                smallvec![],
                smallvec![Qubit::new(a_qubits[2 * qubit]), Qubit::new(min_qubit)],
            ));
            self.symplectic_matrix
                .prepend_cx(a_qubits[2 * qubit], min_qubit);

            gate_seq.push((
                StandardGate::CX,
                smallvec![],
                smallvec![Qubit::new(min_qubit), Qubit::new(a_qubits[2 * qubit + 1])],
            ));
            self.symplectic_matrix
                .prepend_cx(min_qubit, a_qubits[2 * qubit + 1]);
        }

        Ok(())
    }

    /// The main synthesis function.
    pub(crate) fn run(&mut self) -> Result<(usize, CliffordGatesVec), String> {
        let mut clifford_gates = CliffordGatesVec::new();

        while !self.unprocessed_qubits.is_empty() {
            let costs: Vec<(usize, usize)> = self
                .unprocessed_qubits
                .iter()
                .map(|q| self.compute_cost(*q).map(|cost| (cost, *q)))
                .collect::<Result<Vec<_>, _>>()?;

            let min_cost_qubit = costs.iter().min_by_key(|(cost, _)| cost).unwrap().1;

            self.decouple_qubit(&mut clifford_gates, min_cost_qubit)?;

            self.unprocessed_qubits.swap_remove(&min_cost_qubit);
        }

        adjust_final_pauli_gates(&mut clifford_gates, self.tableau, self.num_qubits)?;

        Ok((self.num_qubits, clifford_gates))
    }
}

/// Resynthesizes a clifford circuit using the greedy Clifford synthesis algorithm.
pub fn resynthesize_clifford_circuit(
    num_qubits: usize,
    gates: &CliffordGatesVec,
) -> Result<CliffordGatesVec, String> {
    let sim_clifford = clifford_from_gate_sequence(gates, num_qubits)?;
    let tableau = Array2::from_shape_fn((2 * num_qubits, 2 * num_qubits + 1), |(i, j)| {
        sim_clifford.get_entry(i, j)
    });
    let mut synthesis = GreedyCliffordSynthesis::new(tableau.view())?;
    let (_, new_gates) = synthesis.run()?;
    Ok(new_gates)
}
