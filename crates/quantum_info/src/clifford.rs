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
use std::fmt;

use fixedbitset::FixedBitSet;
use ndarray::{Array2, ArrayView2};

/// 1-qubit Paulis
#[derive(Clone, Copy, PartialEq)]
pub enum Pauli1q {
    X,
    Y,
    Z,
}

/// Symplectic matrix.
pub struct SymplecticMatrix {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits).
    pub smat: Array2<bool>,
}

/// SIMD accelerated Clifford.
///
/// Currently this class offers a reduced functionality of the python-based
/// Clifford class.
#[derive(Clone)]
pub struct Clifford {
    /// Number of qubits.
    pub num_qubits: usize,
    /// The (2 * num qubits) x (2 * num qubits + 1) stabilizer tableau stored
    /// as a vector of (2 * num_qubits) + 1 columns,
    /// each of length (2 * num_qubits). The element in row
    /// i and column j can be access as tableau[j][i].
    pub tableau: Vec<FixedBitSet>,
    scratch: FixedBitSet,
}

impl Clifford {
    /// Create a new clifford from a tableau. The size of the tableau must match the number of
    /// qubits provided otherwise an invalid Clifford object will be created.
    pub fn new(num_qubits: usize, tableau: Vec<FixedBitSet>) -> Self {
        Clifford {
            num_qubits,
            tableau,
            scratch: FixedBitSet::with_capacity(2 * num_qubits),
        }
    }

    /// Creates the identity Clifford on num_qubits
    pub fn identity(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            tableau: (0..2 * num_qubits + 1)
                .map(|i| {
                    let mut column = FixedBitSet::with_capacity(2 * num_qubits);
                    if i < 2 * num_qubits {
                        // SAFETY: We know column is large enough since it's larger than the range
                        // i is from
                        unsafe {
                            column.insert_unchecked(i);
                        }
                    }
                    column
                })
                .collect(),
            scratch: FixedBitSet::with_capacity(2 * num_qubits),
        }
    }

    /// Creates a new Clifford from a tableau array of bools
    #[inline]
    pub fn from_array(tableau_array: ArrayView2<bool>) -> Self {
        let tableau_shape = tableau_array.shape();
        let num_qubits = tableau_shape[0] / 2;
        let mut out = Self {
            num_qubits,
            tableau: (0..2 * num_qubits + 1)
                .map(|_| FixedBitSet::with_capacity(2 * num_qubits))
                .collect(),
            scratch: FixedBitSet::with_capacity(2 * num_qubits),
        };
        tableau_array
            .indexed_iter()
            .for_each(|(index, v)| out.tableau[index.1].set(index.0, *v));
        out
    }

    #[inline]
    pub fn get_phase(&self) -> &FixedBitSet {
        self.tableau.get(2 * self.num_qubits).unwrap()
    }

    #[inline]
    pub fn get_z(&self, qubit: usize) -> &FixedBitSet {
        self.tableau.get(self.num_qubits + qubit).unwrap()
    }

    #[inline]
    pub fn get_z_mut(&mut self, qubit: usize) -> &mut FixedBitSet {
        self.tableau.get_mut(self.num_qubits + qubit).unwrap()
    }

    /// Modifies the tableau in-place by appending S-gate
    pub fn append_s(&mut self, qubit: usize) {
        self.scratch.clone_from(&self.tableau[qubit]);
        self.scratch &= &self.tableau[qubit + self.num_qubits];
        self.tableau[2 * self.num_qubits] ^= &self.scratch;
        let (lhs, rhs) = self.tableau.split_at_mut(qubit + 1);
        rhs[self.num_qubits - 1] ^= lhs.last().unwrap();
    }

    /// Modifies the tableau in-place by appending Sdg-gate
    pub fn append_sdg(&mut self, qubit: usize) {
        let x = &self.tableau[qubit];
        self.scratch
            .clone_from(&self.tableau[qubit + self.num_qubits]);
        self.scratch.toggle_range(..);
        self.scratch &= x;
        self.tableau[2 * self.num_qubits] ^= &self.scratch;
        let (lhs, rhs) = self.tableau.split_at_mut(qubit + 1);
        rhs[self.num_qubits - 1] ^= lhs.last().unwrap();
    }

    /// Modifies the tableau in-place by appending SX-gate
    pub fn append_sx(&mut self, qubit: usize) {
        let x = &self.tableau[qubit];
        let z = &self.tableau[qubit + self.num_qubits];
        self.scratch.clone_from(x);
        self.scratch.toggle_range(..);
        self.scratch &= z;
        self.tableau[2 * self.num_qubits] ^= &self.scratch;
        let (lhs, rhs) = self.tableau.split_at_mut(qubit + 1);
        *lhs.last_mut().unwrap() ^= &rhs[self.num_qubits - 1];
    }

    /// Modifies the tableau in-place by appending SXDG-gate
    pub fn append_sxdg(&mut self, qubit: usize) {
        self.scratch.clone_from(&self.tableau[qubit]);
        self.scratch &= &self.tableau[qubit + self.num_qubits];
        self.tableau[2 * self.num_qubits] ^= &self.scratch;
        let (lhs, rhs) = self.tableau.split_at_mut(qubit + 1);
        *lhs.last_mut().unwrap() ^= &rhs[self.num_qubits - 1];
    }

    /// Modifies the tableau in-place by appending H-gate
    pub fn append_h(&mut self, qubit: usize) {
        let x = &self.tableau[qubit];
        let z = &self.tableau[qubit + self.num_qubits];
        self.scratch.clone_from(x);
        self.scratch &= z;
        self.tableau[2 * self.num_qubits] ^= &self.scratch;
        self.tableau.swap(qubit, self.num_qubits + qubit);
    }

    /// Modifies the tableau in-place by appending SWAP-gate
    pub fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        self.tableau.swap(qubit0, qubit1);
        self.tableau
            .swap(self.num_qubits + qubit0, self.num_qubits + qubit1);
    }

    /// Modifies the tableau in-place by appending CX-gate
    pub fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        let x0 = &self.tableau[qubit0];
        let z0 = &self.tableau[qubit0 + self.num_qubits];
        let x1 = &self.tableau[qubit1];
        let z1 = &self.tableau[qubit1 + self.num_qubits];

        self.scratch.clone_from(x1);
        self.scratch ^= z0;
        self.scratch.toggle_range(..);
        self.scratch &= z1;
        self.scratch &= x0;
        self.tableau[2 * self.num_qubits] ^= &self.scratch;
        self.scratch.clone_from(&self.tableau[qubit1]);
        self.scratch ^= &self.tableau[qubit0];
        std::mem::swap(&mut self.tableau[qubit1], &mut self.scratch);
        self.scratch
            .clone_from(&self.tableau[qubit0 + self.num_qubits]);
        self.scratch ^= &self.tableau[qubit1 + self.num_qubits];
        std::mem::swap(
            &mut self.tableau[qubit0 + self.num_qubits],
            &mut self.scratch,
        );
    }

    /// Modifies the tableau in-place by appending CZ-gate
    pub fn append_cz(&mut self, qubit0: usize, qubit1: usize) {
        let x0 = &self.tableau[qubit0];
        let z0 = &self.tableau[qubit0 + self.num_qubits];
        let x1 = &self.tableau[qubit1];
        let z1 = &self.tableau[qubit1 + self.num_qubits];
        self.scratch.clone_from(z0);
        self.scratch ^= z1;
        self.scratch &= &(x0 & x1);
        self.tableau[2 * self.num_qubits] ^= &self.scratch;
        self.scratch
            .clone_from(&self.tableau[qubit1 + self.num_qubits]);
        self.scratch ^= &self.tableau[qubit0];
        std::mem::swap(
            &mut self.tableau[qubit1 + self.num_qubits],
            &mut self.scratch,
        );
        self.scratch
            .clone_from(&self.tableau[qubit0 + self.num_qubits]);
        self.scratch ^= &self.tableau[qubit1];
        std::mem::swap(
            &mut self.tableau[qubit0 + self.num_qubits],
            &mut self.scratch,
        );
    }

    /// Modifies the tableau in-place by appending CY-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_cy(&mut self, qubit0: usize, qubit1: usize) {
        self.append_sdg(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_s(qubit1);
    }

    /// Modifies the tableau in-place by appending X-gate
    pub fn append_x(&mut self, qubit: usize) {
        let (lhs, rhs) = self.tableau.split_at_mut(qubit + self.num_qubits + 1);
        *rhs.last_mut().unwrap() ^= lhs.last().unwrap();
    }

    /// Modifies the tableau in-place by appending Z-gate
    pub fn append_z(&mut self, qubit: usize) {
        let (lhs, rhs) = self.tableau.split_at_mut(qubit + 1);
        *rhs.last_mut().unwrap() ^= lhs.last().unwrap();
    }

    /// Modifies the tableau in-place by appending Y-gate
    pub fn append_y(&mut self, qubit: usize) {
        self.scratch.clone_from(&self.tableau[qubit]);
        self.scratch ^= &self.tableau[qubit + self.num_qubits];
        self.tableau[2 * self.num_qubits] ^= &self.scratch;
    }

    /// Modifies the tableau in-place by appending iSWAP-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_iswap(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_s(qubit1);
        self.append_h(qubit0);
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
        self.append_h(qubit1);
    }

    /// Modifies the tableau in-place by appending ECR-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_ecr(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_sx(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_x(qubit0);
    }

    /// Modifies the tableau in-place by appending DCX-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_dcx(&mut self, qubit0: usize, qubit1: usize) {
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
    }

    /// Modifies the tableau in-place by appending V-gate.
    /// This is equivalent to an Sdg gate followed by an H gate.
    pub fn append_v(&mut self, qubit: usize) {
        self.scratch.clone_from(&self.tableau[qubit]);
        self.scratch ^= &self.tableau[qubit + self.num_qubits];
        self.tableau.swap(qubit, self.num_qubits + qubit);
        std::mem::swap(&mut self.tableau[qubit], &mut self.scratch);
    }

    /// Modifies the tableau in-place by appending W-gate.
    /// This is equivalent to two V gates.
    pub fn append_w(&mut self, qubit: usize) {
        self.scratch.clone_from(&self.tableau[qubit]);
        self.scratch ^= &self.tableau[qubit + self.num_qubits];
        self.tableau.swap(qubit, self.num_qubits + qubit);
        std::mem::swap(
            &mut self.tableau[qubit + self.num_qubits],
            &mut self.scratch,
        );
    }
    /// Modifies the tableau in-place by appending RZ-gate,
    /// with an angle that is an integer multiple of pi/2
    /// so RZ is necessarily a Clifford gate
    pub fn append_rz(&mut self, qubit: usize, multiple: usize) {
        let multiple = multiple.rem_euclid(4);
        match multiple {
            0 => {}
            1 => self.append_s(qubit),
            2 => self.append_z(qubit),
            3 => self.append_sdg(qubit),
            _ => unreachable!("Multiple should be an integer."),
        }
    }
    /// Modifies the tableau in-place by appending RX-gate,
    /// with an angle that is an integer multiple of pi/2
    /// so RX is necessarily a Clifford gate
    pub fn append_rx(&mut self, qubit: usize, multiple: usize) {
        let multiple = multiple.rem_euclid(4);
        match multiple {
            0 => {}
            1 => self.append_sx(qubit),
            2 => self.append_x(qubit),
            3 => self.append_sxdg(qubit),
            _ => unreachable!("Multiple should be an integer."),
        }
    }
    /// Modifies the tableau in-place by appending RY-gate,
    /// with an angle that is an integer multiple of pi/2
    /// so RY is necessarily a Clifford gate
    pub fn append_ry(&mut self, qubit: usize, multiple: usize) {
        let multiple = multiple.rem_euclid(4);
        match multiple {
            0 => {}
            1 => {
                self.append_z(qubit);
                self.append_h(qubit)
            }
            2 => self.append_y(qubit),
            3 => {
                self.append_h(qubit);
                self.append_z(qubit)
            }
            _ => unreachable!("Multiple should be an integer."),
        }
    }
    /// Applies the initial basis transformation for a Pauli Product Rotation,
    /// and modifies the tableau in-place.
    ///
    /// For each qubit in the Pauli string:
    /// - Z basis: no transformation needed
    /// - X basis: apply H gate
    /// - Y basis: apply SX gate
    ///
    /// Then applies a CX ladder to entangle the qubits, preparing for the
    /// central rotation on the first qubit.
    fn _append_initial_part_ppr(
        &mut self,
        z: &[bool],
        x: &[bool],
        indices: &[u32],
        active_indices: &[usize],
    ) {
        // initial H or SX gates (in case of pauli X or pauli Y respectively)
        for &qubit in active_indices {
            match (z[qubit], x[qubit]) {
                (true, false) => {}                                      // pauli Z on qubit
                (true, true) => self.append_sx(indices[qubit] as usize), // pauli Y on qubit
                (false, true) => self.append_h(indices[qubit] as usize), // pauli X on qubit
                (false, false) => panic!("Pauli I terms are ignored."),  // pauli I on qubit
            }
        }

        // CX ladder (reverse order)
        if active_indices.len() > 1 {
            for w in active_indices.windows(2).rev() {
                let (a, b) = (w[0], w[1]);
                self.append_cx(indices[b] as usize, indices[a] as usize);
            }
        }
    }

    /// Applies the final basis transformation for a Pauli Product Rotation,
    /// and modifies the tableau in-place.
    ///
    /// First, applies a reversed CX ladder to disentangle the qubits.
    ///
    /// Then, for each qubit in the Pauli string:
    /// - Z basis: no transformation needed
    /// - X basis: apply H gate
    /// - Y basis: apply SXdg gate
    fn _append_final_part_ppr(
        &mut self,
        z: &[bool],
        x: &[bool],
        indices: &[u32],
        active_indices: &[usize],
    ) {
        // CX ladder
        if active_indices.len() > 1 {
            for w in active_indices.windows(2) {
                let (a, b) = (w[0], w[1]);
                self.append_cx(indices[b] as usize, indices[a] as usize);
            }
        }

        // final H or SXdg gates (in case of pauli X or pauli Y respectively)
        for &qubit in active_indices {
            match (z[qubit], x[qubit]) {
                (true, false) => {}                                        // pauli Z on qubit
                (true, true) => self.append_sxdg(indices[qubit] as usize), // pauli Y on qubit
                (false, true) => self.append_h(indices[qubit] as usize),   // pauli X on qubit
                (false, false) => panic!("Pauli I terms were ignored."),   // pauli I on qubit
            }
        }
    }

    /// Modifies the tableau in-place by appending PPR gate,
    /// with an angle that is an integer multiple of pi/2
    /// so PPR is necessarily a Clifford gate
    pub fn append_ppr(
        &mut self,
        pauli_z: &[bool],
        pauli_x: &[bool],
        indices: &[u32],
        multiple: usize,
    ) {
        // Ignore I terms from a sparse Pauli list and indicate their corresponsing indices
        // For example, if the input Pauli is "XIYZ" (read left-to-right) on qubits [1, 2, 4, 7]
        // then the output is "XYZ" on qubits [1, 4, 7]
        let active_indices: Vec<usize> = pauli_z
            .iter()
            .zip(pauli_x)
            .enumerate()
            .filter_map(|(i, (&z, &x))| (z || x).then_some(i))
            .collect();

        self._append_initial_part_ppr(pauli_z, pauli_x, indices, &active_indices);

        // internal RZ gate
        if let Some(&idx) = active_indices.first() {
            self.append_rz(indices[idx] as usize, multiple);
        }

        self._append_final_part_ppr(pauli_z, pauli_x, indices, &active_indices);
    }

    /// Evolving a (dense) Pauli gate by the Clifford.
    /// This is done by appending PPR (PPM) initial and final gates to the Clifford tableau in-place,
    /// and evolving the internal RZ gate.
    pub fn evolve_pauli(
        &mut self,
        in_z: &[bool],
        in_x: &[bool],
        indices_in: &[u32],
    ) -> (bool, Vec<bool>, Vec<bool>, Vec<u32>) {
        // Ignore I terms from a sparse Pauli list and indicate their corresponsing indices
        // For example, if the input Pauli is "XIYZ" (read left-to-right) on qubits [1, 2, 4, 7]
        // then the output is "XYZ" on qubits [1, 4, 7]
        let active_indices: Vec<usize> = in_z
            .iter()
            .zip(in_x)
            .enumerate()
            .filter_map(|(i, (&z, &x))| (z || x).then_some(i))
            .collect();

        if let Some(&idx) = active_indices.first() {
            self._append_initial_part_ppr(in_z, in_x, indices_in, &active_indices);

            // Evolving RZ by the Clifford.
            let (sign, z, x, indices) =
                self.evolve_single_qubit_pauli(Pauli1q::Z, indices_in[idx] as usize);

            self._append_final_part_ppr(in_z, in_x, indices_in, &active_indices);

            return (sign, z, x, indices);
        }
        (false, in_z.to_vec(), in_x.to_vec(), indices_in.to_vec())
    }

    /// Evolving a single qubit pauli on qubit qbit by the Clifford.
    /// The pauli (X, Y or Z) is given as (pauli_z, pauli_x)
    /// Returns the evolved Pauli in the a sparse ZX format: (sign, z, x, indices).
    pub fn evolve_single_qubit_pauli(
        &self,
        pauli: Pauli1q,
        qbit: usize,
    ) -> (bool, Vec<bool>, Vec<bool>, Vec<u32>) {
        let mut z = Vec::with_capacity(self.num_qubits);
        let mut x = Vec::with_capacity(self.num_qubits);
        let mut indices = Vec::with_capacity(self.num_qubits);
        let mut pauli_indices = Vec::<usize>::with_capacity(2 * self.num_qubits);
        // Compute the y-count to avoid recomputing it later
        let mut pauli_y_count: u32 = 0;
        for i in 0..self.num_qubits {
            let (z_bit, x_bit) = match pauli {
                Pauli1q::Z => (
                    self.tableau[qbit][i],
                    self.tableau[qbit][i + self.num_qubits],
                ),
                Pauli1q::X => (
                    self.tableau[qbit + self.num_qubits][i],
                    self.tableau[qbit + self.num_qubits][i + self.num_qubits],
                ),
                Pauli1q::Y => (
                    self.tableau[qbit + self.num_qubits][i] ^ self.tableau[qbit][i],
                    self.tableau[qbit + self.num_qubits][i + self.num_qubits]
                        ^ self.tableau[qbit][i + self.num_qubits],
                ),
            };
            if z_bit || x_bit {
                z.push(z_bit);
                x.push(x_bit);
                indices.push(i as u32);
                if x_bit {
                    pauli_indices.push(i);
                }
                if z_bit {
                    pauli_indices.push(i + self.num_qubits);
                }
                pauli_y_count += (x_bit && z_bit) as u32;
            }
        }
        let phase = compute_phase_product_pauli(self, &pauli_indices, pauli_y_count);

        (phase, z, x, indices)
    }
}

/// Computes the sign (either +1 or -1) when conjugating a Pauli by a Clifford
fn compute_phase_product_pauli(
    clifford: &Clifford,
    pauli_indices: &[usize],
    pauli_y_count: u32,
) -> bool {
    let phase = pauli_indices.iter().fold(false, |acc, &pauli_index| {
        acc ^ (clifford.tableau[2 * clifford.num_qubits][pauli_index])
    });

    let mut ifact: u8 = pauli_y_count as u8 % 4;

    for j in 0..clifford.num_qubits {
        let mut x = false;
        let mut z = false;
        for &pauli_index in pauli_indices.iter() {
            let x1: bool = clifford.tableau[j][pauli_index];
            let z1: bool = clifford.tableau[j + clifford.num_qubits][pauli_index];

            match (x1, z1, x, z) {
                (false, true, true, true)
                | (true, false, false, true)
                | (true, true, true, false) => {
                    ifact += 1;
                }
                (false, true, true, false)
                | (true, false, true, true)
                | (true, true, false, true) => {
                    ifact += 3;
                }
                _ => {}
            };
            x ^= x1;
            z ^= z1;
            ifact %= 4;
        }
    }
    (((ifact % 4) >> 1) != 0) ^ phase
}

impl fmt::Debug for Clifford {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        writeln!(f, "Tableau:")?;
        for i in 0..2 * self.num_qubits {
            for j in 0..2 * self.num_qubits + 1 {
                write!(f, "{} ", self.tableau[j][i] as u8)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;
        Ok(())
    }
}
