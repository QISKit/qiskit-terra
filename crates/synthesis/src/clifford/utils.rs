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

use crate::linear::utils::calc_inverse_matrix_inner;
use ndarray::{Array1, Array2, ArrayView2, azip, s};
use qiskit_circuit::Qubit;
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_quantum_info::clifford::Clifford;
use smallvec::{SmallVec, smallvec};

/// A symplectic matrix representation.
///
/// Conceptually, a symplectic matrix is represented by a tableau in which the rows
/// recorrespond to destabilizers and stabilizers, and the columns correspond
/// to the X and Z components:
///
/// ```text
/// [ destab_x | destab_z ]
/// [  stab_x  |  stab_z  ]
/// ```
///
/// Currently this class is internal to the synthesis library.
pub struct SymplecticMatrix {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits).
    pub smat: Array2<bool>,
}

impl SymplecticMatrix {
    /// Modifies the matrix in-place by appending S-gate
    #[allow(dead_code)]
    pub fn append_s(&mut self, qubit: usize) {
        let (x, mut z) = self
            .smat
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    /// Modifies the matrix in-place by prepending S-gate
    pub fn prepend_s(&mut self, qubit: usize) {
        let (x, mut z) = self
            .smat
            .multi_slice_mut((s![self.num_qubits + qubit, ..], s![qubit, ..]));
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    /// Modifies the matrix in-place by appending H-gate
    #[allow(dead_code)]
    pub fn append_h(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .smat
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));
        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x));
    }

    /// Modifies the matrix in-place by prepending H-gate
    pub fn prepend_h(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .smat
            .multi_slice_mut((s![qubit, ..], s![self.num_qubits + qubit, ..]));
        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x));
    }

    /// Modifies the matrix in-place by appending SWAP-gate
    #[allow(dead_code)]
    pub fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        let (mut x0, mut z0, mut x1, mut z1) = self.smat.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
        ));
        azip!((x0 in &mut x0, x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
        azip!((z0 in &mut z0, z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
    }

    /// Modifies the matrix in-place by prepending SWAP-gate
    pub fn prepend_swap(&mut self, qubit0: usize, qubit1: usize) {
        let (mut x0, mut z0, mut x1, mut z1) = self.smat.multi_slice_mut((
            s![qubit0, ..],
            s![self.num_qubits + qubit0, ..],
            s![qubit1, ..],
            s![self.num_qubits + qubit1, ..],
        ));
        azip!((x0 in &mut x0, x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
        azip!((z0 in &mut z0, z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
    }

    /// Modifies the matrix in-place by appending CX-gate
    #[allow(dead_code)]
    pub fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        let (x0, mut z0, mut x1, z1) = self.smat.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
        ));
        azip!((x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
        azip!((z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
    }

    /// Modifies the matrix in-place by prepending CX-gate
    pub fn prepend_cx(&mut self, qubit0: usize, qubit1: usize) {
        let (x0, mut z0, mut x1, z1) = self.smat.multi_slice_mut((
            s![qubit1, ..],
            s![self.num_qubits + qubit1, ..],
            s![qubit0, ..],
            s![self.num_qubits + qubit0, ..],
        ));
        azip!((x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
        azip!((z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
    }
}

/// A sequence of Clifford gates.
/// Represents the return type of Clifford synthesis algorithms.
pub type CliffordGatesVec = Vec<(StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)>;

/// Given a sequence of Clifford gates that correctly implements the symplectic matrix
/// of the target clifford tableau, adds the Pauli gates to also match the phase of
/// the tableau.
pub fn adjust_final_pauli_gates(
    gate_seq: &mut CliffordGatesVec,
    target_tableau: ArrayView2<bool>,
    num_qubits: usize,
) -> Result<(), String> {
    // simulate the clifford circuit that we have constructed
    let simulated_clifford = clifford_from_gate_sequence(gate_seq, num_qubits)?;

    // compute the phase difference
    let target_phase = target_tableau.column(2 * num_qubits);
    let sim_phase = simulated_clifford.tableau.get_phase();

    let delta_phase = target_phase
        .iter()
        .zip((0..sim_phase.len()).map(|i| sim_phase[i]))
        .map(|(&a, b)| a ^ b);

    // compute inverse of the symplectic matrix
    let smat = target_tableau.slice(s![.., ..2 * num_qubits]);
    let smat_inv = calc_inverse_matrix_inner(smat, false)?;

    // compute smat_inv * delta_phase
    let arr1 = smat_inv.map(|v| *v as usize);
    let vec2: Vec<usize> = delta_phase.map(|v| v as usize).collect();
    let arr2 = Array1::from(vec2);
    let delta_phase_pre = arr1.dot(&arr2).map(|v| v % 2 == 1);

    // add pauli gates
    for qubit in 0..num_qubits {
        if delta_phase_pre[qubit] && delta_phase_pre[qubit + num_qubits] {
            gate_seq.push((StandardGate::Y, smallvec![], smallvec![Qubit::new(qubit)]));
        } else if delta_phase_pre[qubit] {
            gate_seq.push((StandardGate::Z, smallvec![], smallvec![Qubit::new(qubit)]));
        } else if delta_phase_pre[qubit + num_qubits] {
            gate_seq.push((StandardGate::X, smallvec![], smallvec![Qubit::new(qubit)]));
        }
    }

    Ok(())
}

/// Creates a Clifford from a given sequence of Clifford gates.
/// In essence, starts from the identity tableau and modifies it
/// based on the gates in the sequence.
pub fn clifford_from_gate_sequence(
    gate_seq: &CliffordGatesVec,
    num_qubits: usize,
) -> Result<Clifford, String> {
    // create the identity
    let mut clifford = Clifford::identity(num_qubits);

    gate_seq
        .iter()
        .try_for_each(|(gate, _params, qubits)| match *gate {
            StandardGate::S => {
                clifford.append_s(qubits[0].index());
                Ok(())
            }
            StandardGate::Sdg => {
                clifford.append_sdg(qubits[0].0 as usize);
                Ok(())
            }
            StandardGate::SX => {
                clifford.append_sx(qubits[0].0 as usize);
                Ok(())
            }
            StandardGate::SXdg => {
                clifford.append_sxdg(qubits[0].0 as usize);
                Ok(())
            }
            StandardGate::H => {
                clifford.append_h(qubits[0].index());
                Ok(())
            }
            StandardGate::CX => {
                clifford.append_cx(qubits[0].index(), qubits[1].index());
                Ok(())
            }
            StandardGate::Swap => {
                clifford.append_swap(qubits[0].index(), qubits[1].index());
                Ok(())
            }
            _ => Err(format!("Unsupported gate {gate:?}")),
        })?;
    Ok(clifford)
}
