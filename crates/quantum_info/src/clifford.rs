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
use ndarray::ArrayView2;

use thiserror::Error;

/// 1-qubit Paulis
#[derive(Clone, Copy, PartialEq)]
pub enum Pauli1q {
    X,
    Y,
    Z,
}

/// An index identifying a two-qubit Pauli operator.
///
/// A two-qubit Pauli operator can be encoded as an integer in 0..16
/// by packing the `x` and `z` components of the two qubits:
/// ```text
/// (x[0] << 3) | (z[0] << 2) | (x[1] << 1) | z[1].
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TwoQubitPauliIndex(u8);

impl TwoQubitPauliIndex {
    /// Constructor from the given index.
    pub fn from_usize(index: usize) -> Self {
        Self(index as u8)
    }

    /// Packs the bits into the index.
    pub fn from_bits(x0: bool, z0: bool, x1: bool, z1: bool) -> Self {
        Self(((x0 as u8) << 3) | ((z0 as u8) << 2) | ((x1 as u8) << 1) | (z1 as u8))
    }

    /// Unpacks bits from the index.
    pub fn bits(self) -> (bool, bool, bool, bool) {
        (
            self.0 & 0b1000 != 0,
            self.0 & 0b0100 != 0,
            self.0 & 0b0010 != 0,
            self.0 & 0b0001 != 0,
        )
    }

    /// Returns the index as a `usize`.
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// SIMD accelerated PauliList.
///
/// Stores multiple Paulis with associated phases, but in a packed format
/// optimized for vectorized conjugation by Clifford gates. The phases are
/// only allowed to be false ("+") and true ("-"); the "i" and "-i" phases
/// are not allowed.
///
/// Implementation-wise, each entry in `data` is a `FixedBitSet` corresponding
/// to some X, some Z, or the phase component of every Pauli in the list.
///
/// One can conceptually visualize a `PauliList` as the following two-dimensional
/// array, with a column in this array corresponding to an entry in `data`, and a row
/// corresponding to a specific Pauli (represented using X, Z, and phase components).
/// In this visual representation there are (2 * num_qubits + 1) columns and
/// `num_paulis` rows.
///
/// ```text
/// [ pauli_1_x pauli_1_z pauli_1_p ]
/// [ pauli_2_x pauli_2_z pauli_2_p ]
/// [ pauli_3_x pauli_3_z pauli_3_p ]
/// [ pauli_4_x pauli_4_z pauli_4_p ]
/// ```
#[derive(Clone)]
pub struct PauliList {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Number of Paulis.
    pub num_paulis: usize,
    /// List of Paulis, stored as vector of (2 * num_qubits + 1) columns,
    /// each of length num_paulis.
    pub data: Vec<FixedBitSet>,
    /// Scratch space for internal computations, of length num_paulis.
    scratch: FixedBitSet,
}

#[derive(Error, Debug)]
pub enum PauliListError {
    #[error("Invalid Pauli label: {0}")]
    InvalidLabel(String),
}

/// Specifies the order in which Pauli labels are interpreted.
/// The standard Qiskit convention inteprets labels right-to-left:
/// for Pauli label "IXYZ", the label on the first qubit is "Z". However,
/// in some cases it is more convenient to interpret labels left-to-right.
#[derive(PartialEq)]
pub enum PauliLabelOrder {
    LeftToRight,
    RightToLeft,
}

impl PauliList {
    /// A reference to the given x-column of the PauliList.
    ///
    /// Note: this function assumes that the given column exists
    /// and panics otherwise.
    #[inline]
    pub fn get_x(&self, qubit: usize) -> &FixedBitSet {
        self.data.get(qubit).unwrap()
    }

    /// A mutable reference to the given x-column of the PauliList.
    ///
    /// Note: this function assumes that the given column exists
    /// and panics otherwise.
    #[inline]
    pub fn get_x_mut(&mut self, qubit: usize) -> &mut FixedBitSet {
        self.data.get_mut(qubit).unwrap()
    }

    /// A reference to the given z-column of the PauliList.
    ///
    /// Note: this function assumes that the given column exists
    /// and panics otherwise.
    #[inline]
    pub fn get_z(&self, qubit: usize) -> &FixedBitSet {
        self.data.get(self.num_qubits + qubit).unwrap()
    }

    /// A mutable reference to the z-column of the PauliList.
    ///
    /// Note: this function assumes that the given column exists
    /// and panics otherwise.
    #[inline]
    pub fn get_z_mut(&mut self, qubit: usize) -> &mut FixedBitSet {
        self.data.get_mut(self.num_qubits + qubit).unwrap()
    }

    /// A reference to the phase column of the PauliList.
    #[inline]
    pub fn get_phase(&self) -> &FixedBitSet {
        self.data.get(2 * self.num_qubits).unwrap()
    }

    /// A mutable reference to the phase column of the PauliList.
    #[inline]
    pub fn get_phase_mut(&mut self) -> &mut FixedBitSet {
        self.data.get_mut(2 * self.num_qubits).unwrap()
    }

    /// The entry in the given x-column corresponding to the Pauli ``pauli_idx``.
    ///
    /// Note: this function assumes that the given pauli and the column exist
    /// and panics otherwise.
    #[inline]
    pub fn get_pauli_x(&self, pauli_idx: usize, qubit: usize) -> bool {
        self.data[qubit][pauli_idx]
    }

    /// The entry in the given z-column corresponding to the Pauli ``pauli_idx``.
    ///
    /// Note: this function assumes that the given pauli and the column exist
    /// and panics otherwise.
    #[inline]
    pub fn get_pauli_z(&self, pauli_idx: usize, qubit: usize) -> bool {
        self.data[qubit + self.num_qubits][pauli_idx]
    }

    /// The entry in the phase-column corresponding to the Pauli ``pauli_idx``.
    ///
    /// Note: this function assumes that the given pauli exists
    /// and panics otherwise.
    #[inline]
    pub fn get_pauli_phase(&self, pauli_idx: usize) -> bool {
        self.data[2 * self.num_qubits][pauli_idx]
    }

    /// Modifies the pauli list in-place by conjugating each pauli with S-gate
    pub fn append_s(&mut self, qubit: usize) {
        self.scratch.clone_from(&self.data[qubit]);
        self.scratch &= &self.data[qubit + self.num_qubits];
        self.data[2 * self.num_qubits] ^= &self.scratch;
        let (lhs, rhs) = self.data.split_at_mut(qubit + 1);
        rhs[self.num_qubits - 1] ^= lhs.last().unwrap();
    }

    /// Modifies the pauli list in-place by conjugating each pauli with Sdg-gate
    pub fn append_sdg(&mut self, qubit: usize) {
        let x = &self.data[qubit];
        self.scratch.clone_from(&self.data[qubit + self.num_qubits]);
        self.scratch.toggle_range(..);
        self.scratch &= x;
        self.data[2 * self.num_qubits] ^= &self.scratch;
        let (lhs, rhs) = self.data.split_at_mut(qubit + 1);
        rhs[self.num_qubits - 1] ^= lhs.last().unwrap();
    }

    /// Modifies the pauli list in-place by conjugating each pauli with SX-gate
    pub fn append_sx(&mut self, qubit: usize) {
        let x = &self.data[qubit];
        let z = &self.data[qubit + self.num_qubits];
        self.scratch.clone_from(x);
        self.scratch.toggle_range(..);
        self.scratch &= z;
        self.data[2 * self.num_qubits] ^= &self.scratch;
        let (lhs, rhs) = self.data.split_at_mut(qubit + 1);
        *lhs.last_mut().unwrap() ^= &rhs[self.num_qubits - 1];
    }

    /// Modifies the pauli list in-place by conjugating each pauli with SXdg-gate
    pub fn append_sxdg(&mut self, qubit: usize) {
        self.scratch.clone_from(&self.data[qubit]);
        self.scratch &= &self.data[qubit + self.num_qubits];
        self.data[2 * self.num_qubits] ^= &self.scratch;
        let (lhs, rhs) = self.data.split_at_mut(qubit + 1);
        *lhs.last_mut().unwrap() ^= &rhs[self.num_qubits - 1];
    }

    /// Modifies the pauli list in-place by conjugating each pauli with H-gate
    pub fn append_h(&mut self, qubit: usize) {
        let x = &self.data[qubit];
        let z = &self.data[qubit + self.num_qubits];
        self.scratch.clone_from(x);
        self.scratch &= z;
        self.data[2 * self.num_qubits] ^= &self.scratch;
        self.data.swap(qubit, self.num_qubits + qubit);
    }

    /// Modifies the pauli list in-place by conjugating each pauli with SWAP-gate
    pub fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        self.data.swap(qubit0, qubit1);
        self.data
            .swap(self.num_qubits + qubit0, self.num_qubits + qubit1);
    }

    /// Modifies the pauli list in-place by conjugating each pauli with CX-gate
    pub fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        let x0 = &self.data[qubit0];
        let z0 = &self.data[qubit0 + self.num_qubits];
        let x1 = &self.data[qubit1];
        let z1 = &self.data[qubit1 + self.num_qubits];

        self.scratch.clone_from(x1);
        self.scratch ^= z0;
        self.scratch.toggle_range(..);
        self.scratch &= z1;
        self.scratch &= x0;
        self.data[2 * self.num_qubits] ^= &self.scratch;
        self.scratch.clone_from(&self.data[qubit1]);
        self.scratch ^= &self.data[qubit0];
        std::mem::swap(&mut self.data[qubit1], &mut self.scratch);
        self.scratch
            .clone_from(&self.data[qubit0 + self.num_qubits]);
        self.scratch ^= &self.data[qubit1 + self.num_qubits];
        std::mem::swap(&mut self.data[qubit0 + self.num_qubits], &mut self.scratch);
    }

    /// Modifies the pauli list in-place by conjugating each pauli with CZ-gate
    pub fn append_cz(&mut self, qubit0: usize, qubit1: usize) {
        let x0 = &self.data[qubit0];
        let z0 = &self.data[qubit0 + self.num_qubits];
        let x1 = &self.data[qubit1];
        let z1 = &self.data[qubit1 + self.num_qubits];
        self.scratch.clone_from(z0);
        self.scratch ^= z1;
        self.scratch &= &(x0 & x1);
        self.data[2 * self.num_qubits] ^= &self.scratch;
        self.scratch
            .clone_from(&self.data[qubit1 + self.num_qubits]);
        self.scratch ^= &self.data[qubit0];
        std::mem::swap(&mut self.data[qubit1 + self.num_qubits], &mut self.scratch);
        self.scratch
            .clone_from(&self.data[qubit0 + self.num_qubits]);
        self.scratch ^= &self.data[qubit1];
        std::mem::swap(&mut self.data[qubit0 + self.num_qubits], &mut self.scratch);
    }

    /// Modifies the pauli list in-place by conjugating each pauli with CY-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_cy(&mut self, qubit0: usize, qubit1: usize) {
        self.append_sdg(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_s(qubit1);
    }

    /// Modifies the pauli list in-place by conjugating each pauli with X-gate
    pub fn append_x(&mut self, qubit: usize) {
        let (lhs, rhs) = self.data.split_at_mut(qubit + self.num_qubits + 1);
        *rhs.last_mut().unwrap() ^= lhs.last().unwrap();
    }

    /// Modifies the pauli list in-place by conjugating each pauli with Z-gate
    pub fn append_z(&mut self, qubit: usize) {
        let (lhs, rhs) = self.data.split_at_mut(qubit + 1);
        *rhs.last_mut().unwrap() ^= lhs.last().unwrap();
    }

    /// Modifies the pauli list in-place by conjugating each pauli with Y-gate
    pub fn append_y(&mut self, qubit: usize) {
        self.scratch.clone_from(&self.data[qubit]);
        self.scratch ^= &self.data[qubit + self.num_qubits];
        self.data[2 * self.num_qubits] ^= &self.scratch;
    }

    /// Modifies the pauli list in-place by conjugating each pauli with iSWAP-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_iswap(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_s(qubit1);
        self.append_h(qubit0);
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
        self.append_h(qubit1);
    }

    /// Modifies the pauli list in-place by conjugating each pauli with ECR-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_ecr(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_sx(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_x(qubit0);
    }

    /// Modifies the pauli list in-place by conjugating each pauli with DCX-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_dcx(&mut self, qubit0: usize, qubit1: usize) {
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
    }

    /// Modifies the pauli list in-place by conjugating each pauli with V-gate
    /// This is equivalent to an Sdg gate followed by an H gate.
    pub fn append_v(&mut self, qubit: usize) {
        self.scratch.clone_from(&self.data[qubit]);
        self.scratch ^= &self.data[qubit + self.num_qubits];
        self.data.swap(qubit, self.num_qubits + qubit);
        std::mem::swap(&mut self.data[qubit], &mut self.scratch);
    }

    /// Modifies the pauli list in-place by conjugating each pauli with W-gate
    /// This is equivalent to two V gates.
    pub fn append_w(&mut self, qubit: usize) {
        self.scratch.clone_from(&self.data[qubit]);
        self.scratch ^= &self.data[qubit + self.num_qubits];
        self.data.swap(qubit, self.num_qubits + qubit);
        std::mem::swap(&mut self.data[qubit + self.num_qubits], &mut self.scratch);
    }

    /// Modifies the pauli list in-place by conjugating each pauli with RZ-gate,
    /// with an angle that is an integer multiple of pi/2
    /// (so RZ is necessarily a Clifford gate)
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

    /// Modifies the pauli list in-place by conjugating each pauli with RX-gate,
    /// with an angle that is an integer multiple of pi/2
    /// (so RX is necessarily a Clifford gate)
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

    /// Modifies the pauli list in-place by conjugating each pauli with RY-gate,
    /// with an angle that is an integer multiple of pi/2
    /// (so RY is necessarily a Clifford gate)
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

    /// Computes the support size (number of non-I terms) of the Pauli given by `pauli_idx`.
    pub fn get_pauli_support_size(&self, pauli_idx: usize) -> usize {
        (0..self.num_qubits)
            .filter(|q| {
                self.data[*q].contains(pauli_idx)
                    | self.data[*q + self.num_qubits].contains(pauli_idx)
            })
            .count()
    }

    /// Computes the support (non-I terms) of the Pauli given by `pauli_idx`.
    pub fn get_pauli_support(&self, pauli_idx: usize) -> Vec<usize> {
        (0..self.num_qubits)
            .filter(|q| {
                self.data[*q].contains(pauli_idx)
                    | self.data[*q + self.num_qubits].contains(pauli_idx)
            })
            .collect()
    }

    /// If the Pauli with index `pauli_idx` has support `[q]` of exactly size 1, return `Some(q)`.
    /// Return `None` otherwise.
    pub fn get_pauli_support_if_size_1(&self, pauli_idx: usize) -> Option<usize> {
        let mut found: Option<usize> = None;
        for q in 0..self.num_qubits {
            if self.data[q].contains(pauli_idx)
                || self.data[q + self.num_qubits].contains(pauli_idx)
            {
                if found.is_some() {
                    // Support size is at least 2
                    return None;
                }
                found = Some(q);
            }
        }
        found
    }

    /// Returns whether two Paulis given `pauli_idx1` and `pauli_idx2` commute.
    ///
    /// # Panics
    ///
    /// This function assumes that `pauli_idx1 < self.num_paulis` and `pauli_idx2 < self.num_paulis`.
    pub fn commute(&self, pauli_idx1: usize, pauli_idx2: usize) -> bool {
        let mut parity = false;
        for i in 0..self.num_qubits {
            parity ^= (self.get_pauli_z(pauli_idx1, i) & self.get_pauli_x(pauli_idx2, i))
                ^ (self.get_pauli_x(pauli_idx1, i) & self.get_pauli_z(pauli_idx2, i));
        }
        !parity
    }

    /// Construct a [PauliList] from Pauli labels.
    ///
    /// # Arguments:
    ///
    /// * `num_qubits`: The number of qubits each Pauli is defined on.
    /// * `pauli_labels`: An array of Pauli labels, where each label consists
    ///   of an optional plus or minus sign followed by a sequence of `'I'`, `'X'`, `'Y'`,
    ///   or `'Z'` characters. The `'i'` factor is not allowed.
    /// * `label_order`: specifies whether each label should be interpreted left-to-right,
    ///   or right-to-left.
    ///
    /// # Errors:
    ///
    /// Returns [`PauliListError::InvalidLabel`] if the labels are invalid.
    pub fn from_pauli_labels(
        num_qubits: usize,
        pauli_labels: &[String],
        label_order: PauliLabelOrder,
    ) -> Result<Self, PauliListError> {
        let num_paulis = pauli_labels.len();

        let scratch = FixedBitSet::with_capacity(num_paulis);
        let mut data: Vec<FixedBitSet> = Vec::with_capacity(2 * num_qubits + 1);

        for _ in 0..2 * num_qubits + 1 {
            data.push(scratch.clone());
        }

        for (pauli_idx, pauli_label) in pauli_labels.iter().enumerate() {
            let s = if let Some(rest) = pauli_label.strip_prefix('-') {
                data[2 * num_qubits].set(pauli_idx, true);
                rest
            } else if let Some(rest) = pauli_label.strip_prefix('+') {
                rest
            } else {
                pauli_label.as_str()
            };

            for (j, c) in s.chars().enumerate() {
                let qubit = if label_order == PauliLabelOrder::LeftToRight {
                    j
                } else {
                    num_qubits - j - 1
                };
                match c {
                    'X' => {
                        data[qubit].set(pauli_idx, true);
                    }
                    'Z' => {
                        data[qubit + num_qubits].set(pauli_idx, true);
                    }
                    'Y' => {
                        data[qubit].set(pauli_idx, true);
                        data[qubit + num_qubits].set(pauli_idx, true);
                    }
                    'I' => {}
                    _ => {
                        return Err(PauliListError::InvalidLabel(pauli_label.clone()));
                    }
                }
            }
        }

        Ok(Self {
            num_qubits,
            num_paulis,
            data,
            scratch,
        })
    }

    /// Return Pauli labels.
    ///
    /// # Arguments:
    ///
    /// * `label_order`: specifies whether each label should be interpreted left-to-right,
    ///   or right-to-left.
    pub fn to_pauli_labels(&self, label_order: PauliLabelOrder) -> Vec<String> {
        let mut pauli_labels: Vec<String> = Vec::with_capacity(self.num_paulis);
        for pauli_idx in 0..self.num_paulis {
            let mut s: String = String::with_capacity(self.num_qubits + 1);
            let c = match self.get_pauli_phase(pauli_idx) {
                false => '+',
                true => '-',
            };
            s.push(c);
            for j in 0..self.num_qubits {
                let qubit = if label_order == PauliLabelOrder::LeftToRight {
                    j
                } else {
                    self.num_qubits - j - 1
                };
                let c = match (
                    self.get_pauli_x(pauli_idx, qubit),
                    self.get_pauli_z(pauli_idx, qubit),
                ) {
                    (false, false) => 'I',
                    (false, true) => 'Z',
                    (true, false) => 'X',
                    (true, true) => 'Y',
                };
                s.push(c);
            }
            pauli_labels.push(s);
        }
        pauli_labels
    }

    /// Returns the index of 2-qubit Pauli operator induces by `pauli_idx` on qubits `i` and `j`.
    ///
    /// The two-qubit Pauli is encoded as an integer in `0..16` using the binary representation
    /// ```text
    /// (X[i] << 3) | (Z[i] << 2) | (X[j] << 1) | Z[j].
    /// ```
    /// where `X[i]`, `Z[i]`, `X[j]`, `Z[j]` are the X and Z components of the Pauli operator on qubits
    /// `i` and `j` respectively.
    /// For example, if the Pauli on `(i, j)` is `XY`, the corresponding index is 8 + 0 + 2 + 1 = 11.
    ///
    /// # Panics
    ///
    /// This function assumes that `pauli_idx < self.num_paulis` and
    /// `i < self.num_qubits` and `j < self.num_qubits`.
    #[inline]
    pub fn pauli_pair_index(&self, pauli_idx: usize, i: usize, j: usize) -> usize {
        let xi = self.get_pauli_x(pauli_idx, i);
        let zi = self.get_pauli_z(pauli_idx, i);
        let xj = self.get_pauli_x(pauli_idx, j);
        let zj = self.get_pauli_z(pauli_idx, j);
        TwoQubitPauliIndex::from_bits(xi, zi, xj, zj).as_usize()
    }
}

impl fmt::Display for PauliList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}",
            self.to_pauli_labels(PauliLabelOrder::LeftToRight)
        )
    }
}

/// A SIMD-accelerated Clifford representation.
///
/// Conceptually, a Clifford is represented by a (2 * num qubits) x (2 * num qubits + 1)
/// stabilizer tableau in which the rows recorrespond to destabilizers and stabilizers,
/// and the columns correspond to the X, Z, and phase components:
///
/// ```text
/// [ destab_x | destab_z | destab_phase ]
/// [  stab_x  |  stab_z  |  stab_phase  ]
/// ```
///
/// Internally, this is stored as a `PauliList`, with `tableau` represented as a vector
/// of (2 * num_qubits + 1) "columns", corresponding to the `num_qubits` X, `num_qubits` Z
/// and the phase component of the tableau. Each column is of length (2 * num_qubits) with
/// entries corresponding to `num_qubits` destabilizers and `num_qubits` stabilizers.
/// This representations makes it efficient to conjugate a Clifford by Clifford gates, as all
/// of the Paulis are conjugated in a SIMD-accelerated fashion.
///
/// This type currently provides a subset of the functionality of the Python-based
/// `Clifford`` class.
#[derive(Clone)]
pub struct Clifford {
    /// Stabilizer tableau. The element in row i and column j can be accessed using
    /// ``get_entry(i, j)``.
    pub tableau: PauliList,
}

impl Clifford {
    /// Create a new clifford from a tableau. The size of the tableau must match the number of
    /// qubits provided otherwise an invalid Clifford object will be created.
    pub fn new(num_qubits: usize, tableau: Vec<FixedBitSet>) -> Self {
        Clifford {
            tableau: PauliList {
                num_qubits,
                num_paulis: 2 * num_qubits,
                data: tableau,
                scratch: FixedBitSet::with_capacity(2 * num_qubits),
            },
        }
    }

    /// Creates the identity Clifford on num_qubits
    pub fn identity(num_qubits: usize) -> Self {
        Self {
            tableau: PauliList {
                num_qubits,
                num_paulis: 2 * num_qubits,
                data: (0..2 * num_qubits + 1)
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
            },
        }
    }

    /// Creates a new Clifford from a tableau array of bools
    #[inline]
    pub fn from_array(tableau_array: ArrayView2<bool>) -> Self {
        let tableau_shape = tableau_array.shape();
        let num_qubits = tableau_shape[0] / 2;
        let mut out = Self {
            tableau: PauliList {
                num_qubits,
                num_paulis: 2 * num_qubits,
                data: (0..2 * num_qubits + 1)
                    .map(|_| FixedBitSet::with_capacity(2 * num_qubits))
                    .collect(),
                scratch: FixedBitSet::with_capacity(2 * num_qubits),
            },
        };
        tableau_array
            .indexed_iter()
            .for_each(|(index, v)| out.tableau.data[index.1].set(index.0, *v));
        out
    }

    /// Returns the number of qubits the Clifford acts upon
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.tableau.num_qubits
    }

    /// Returns the value in the given row and the given columns of the tableau.
    /// The rows correspond to stabilizers and destabilizers, and the columns correspond
    /// x, z, and phase components.
    #[inline]
    pub fn get_entry(&self, row: usize, col: usize) -> bool {
        self.tableau.data[col].contains(row)
    }

    /// Sets the value in the given row and the given column of the tableau.
    /// The rows correspond to stabilizers and destabilizers, and the columns correspond
    /// x, z, and phase components.
    #[inline]
    pub fn set_entry(&mut self, row: usize, col: usize, value: bool) {
        self.tableau.data[col].set(row, value);
    }

    /// Modifies the tableau in-place by conjugating with S-gate
    #[inline]
    pub fn append_s(&mut self, qubit: usize) {
        self.tableau.append_s(qubit);
    }

    /// Modifies the tableau in-place by conjugating with Sdg-gate
    #[inline]
    pub fn append_sdg(&mut self, qubit: usize) {
        self.tableau.append_sdg(qubit);
    }

    /// Modifies the tableau in-place by conjugating with SX-gate
    #[inline]
    pub fn append_sx(&mut self, qubit: usize) {
        self.tableau.append_sx(qubit);
    }

    /// Modifies the tableau in-place by conjugating with SXdg-gate
    #[inline]
    pub fn append_sxdg(&mut self, qubit: usize) {
        self.tableau.append_sxdg(qubit);
    }

    /// Modifies the tableau in-place by conjugating with H-gate
    #[inline]
    pub fn append_h(&mut self, qubit: usize) {
        self.tableau.append_h(qubit);
    }

    /// Modifies the tableau in-place by conjugating with SWAP-gate
    #[inline]
    pub fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        self.tableau.append_swap(qubit0, qubit1);
    }

    /// Modifies the tableau in-place by conjugating with CX-gate
    #[inline]
    pub fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        self.tableau.append_cx(qubit0, qubit1);
    }

    /// Modifies the tableau in-place by conjugating with CZ-gate
    #[inline]
    pub fn append_cz(&mut self, qubit0: usize, qubit1: usize) {
        self.tableau.append_cz(qubit0, qubit1);
    }

    /// Modifies the tableau in-place by conjugating with CY-gate
    /// (todo: rewrite using native tableau manipulations)
    #[inline]
    pub fn append_cy(&mut self, qubit0: usize, qubit1: usize) {
        self.tableau.append_cy(qubit0, qubit1);
    }

    /// Modifies the tableau in-place by conjugating with X-gate
    #[inline]
    pub fn append_x(&mut self, qubit: usize) {
        self.tableau.append_x(qubit);
    }

    /// Modifies the tableau in-place by conjugating with Z-gate
    #[inline]
    pub fn append_z(&mut self, qubit: usize) {
        self.tableau.append_z(qubit);
    }

    /// Modifies the tableau in-place by conjugating with Y-gate
    #[inline]
    pub fn append_y(&mut self, qubit: usize) {
        self.tableau.append_y(qubit);
    }

    /// Modifies the tableau in-place by conjugating with iSWAP-gate
    #[inline]
    pub fn append_iswap(&mut self, qubit0: usize, qubit1: usize) {
        self.tableau.append_iswap(qubit0, qubit1);
    }

    /// Modifies the tableau in-place by conjugating with ECR-gate
    #[inline]
    pub fn append_ecr(&mut self, qubit0: usize, qubit1: usize) {
        self.tableau.append_ecr(qubit0, qubit1);
    }

    /// Modifies the tableau in-place by conjugating with DCX-gate
    #[inline]
    pub fn append_dcx(&mut self, qubit0: usize, qubit1: usize) {
        self.tableau.append_dcx(qubit0, qubit1);
    }

    /// Modifies the tableau in-place by conjugating with V-gate
    #[inline]
    pub fn append_v(&mut self, qubit: usize) {
        self.tableau.append_v(qubit);
    }

    /// Modifies the tableau in-place by conjugating with W-gate
    #[inline]
    pub fn append_w(&mut self, qubit: usize) {
        self.tableau.append_w(qubit);
    }

    /// Modifies the tableau in-place by conjugating with RZ-gate,
    /// with an angle that is an integer multiple of pi/2
    /// (so RZ is necessarily a Clifford gate)
    #[inline]
    pub fn append_rz(&mut self, qubit: usize, multiple: usize) {
        self.tableau.append_rz(qubit, multiple);
    }

    /// Modifies the tableau in-place by conjugating with RX-gate,
    /// with an angle that is an integer multiple of pi/2
    /// (so RX is necessarily a Clifford gate)
    #[inline]
    pub fn append_rx(&mut self, qubit: usize, multiple: usize) {
        self.tableau.append_rx(qubit, multiple);
    }

    /// Modifies the tableau in-place by conjugating with RY-gate,
    /// with an angle that is an integer multiple of pi/2
    /// (so RY is necessarily a Clifford gate)
    #[inline]
    pub fn append_ry(&mut self, qubit: usize, multiple: usize) {
        self.tableau.append_ry(qubit, multiple);
    }

    /// Applies the initial basis transformation for a Pauli Product Rotation,
    /// and modifies the tableau in-place.
    ///
    /// See [`PauliList::_append_initial_part_ppr`] for details.
    #[inline]
    fn _append_initial_part_ppr(
        &mut self,
        z: &[bool],
        x: &[bool],
        indices: &[u32],
        active_indices: &[usize],
    ) {
        self.tableau
            ._append_initial_part_ppr(z, x, indices, active_indices);
    }

    /// Applies the final basis transformation for a Pauli Product Rotation,
    /// and modifies the tableau in-place.
    ///
    /// See [`PauliList::_append_final_part_ppr`] for details.
    #[inline]
    fn _append_final_part_ppr(
        &mut self,
        z: &[bool],
        x: &[bool],
        indices: &[u32],
        active_indices: &[usize],
    ) {
        self.tableau
            ._append_final_part_ppr(z, x, indices, active_indices);
    }

    /// Modifies the tableau in-place by appending PPR gate,
    /// with an angle that is an integer multiple of pi/2
    /// so PPR is necessarily a Clifford gate.
    /// See [PauliList::append_ppr] for details.
    #[inline]
    pub fn append_ppr(
        &mut self,
        pauli_z: &[bool],
        pauli_x: &[bool],
        indices: &[u32],
        multiple: usize,
    ) {
        self.tableau.append_ppr(pauli_z, pauli_x, indices, multiple);
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
            self.tableau
                ._append_initial_part_ppr(in_z, in_x, indices_in, &active_indices);

            // Evolving RZ by the Clifford.
            let (sign, z, x, indices) =
                self.evolve_single_qubit_pauli(Pauli1q::Z, indices_in[idx] as usize);

            self.tableau
                ._append_final_part_ppr(in_z, in_x, indices_in, &active_indices);

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
        let num_qubits = self.tableau.num_qubits;
        let mut z = Vec::with_capacity(num_qubits);
        let mut x = Vec::with_capacity(num_qubits);
        let mut indices = Vec::with_capacity(num_qubits);
        let mut pauli_indices = Vec::<usize>::with_capacity(2 * num_qubits);
        // Compute the y-count to avoid recomputing it later
        let mut pauli_y_count: u32 = 0;
        for i in 0..num_qubits {
            let (z_bit, x_bit) = match pauli {
                Pauli1q::Z => (
                    self.tableau.data[qbit][i],
                    self.tableau.data[qbit][i + num_qubits],
                ),
                Pauli1q::X => (
                    self.tableau.data[qbit + num_qubits][i],
                    self.tableau.data[qbit + num_qubits][i + num_qubits],
                ),
                Pauli1q::Y => (
                    self.tableau.data[qbit + num_qubits][i] ^ self.tableau.data[qbit][i],
                    self.tableau.data[qbit + num_qubits][i + num_qubits]
                        ^ self.tableau.data[qbit][i + num_qubits],
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
                    pauli_indices.push(i + num_qubits);
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
        acc ^ (clifford.tableau.data[2 * clifford.tableau.num_qubits][pauli_index])
    });

    let mut ifact: u8 = pauli_y_count as u8 % 4;

    for j in 0..clifford.tableau.num_qubits {
        let mut x = false;
        let mut z = false;
        for &pauli_index in pauli_indices.iter() {
            let x1: bool = clifford.tableau.data[j][pauli_index];
            let z1: bool = clifford.tableau.data[j + clifford.tableau.num_qubits][pauli_index];

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
        for i in 0..2 * self.tableau.num_qubits {
            for j in 0..2 * self.tableau.num_qubits + 1 {
                write!(f, "{} ", self.tableau.data[j][i] as u8)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::clifford::{PauliLabelOrder, PauliList, PauliListError};

    #[test]
    fn test_from_labels_and_back_with_left_to_right() {
        let pauli_labels = ["XIZ".to_string(), "-YYY".to_string()];
        let pauli_list =
            PauliList::from_pauli_labels(3, &pauli_labels, PauliLabelOrder::LeftToRight)
                .expect("PauliList should be created without problems (all labels are valid)");

        // XIZ
        assert!(pauli_list.get_pauli_x(0, 0));
        assert!(!pauli_list.get_pauli_z(0, 0));
        assert!(!pauli_list.get_pauli_x(0, 1));
        assert!(!pauli_list.get_pauli_z(0, 1));
        assert!(!pauli_list.get_pauli_x(0, 2));
        assert!(pauli_list.get_pauli_z(0, 2));
        assert!(!pauli_list.get_pauli_phase(0));
        // -YYY
        assert!(pauli_list.get_pauli_x(1, 0));
        assert!(pauli_list.get_pauli_z(1, 0));
        assert!(pauli_list.get_pauli_x(1, 1));
        assert!(pauli_list.get_pauli_z(1, 1));
        assert!(pauli_list.get_pauli_x(1, 2));
        assert!(pauli_list.get_pauli_z(1, 2));
        assert!(pauli_list.get_pauli_phase(1));

        let pauli_labels_roundtrip = pauli_list.to_pauli_labels(PauliLabelOrder::LeftToRight);
        let expected_pauli_labels = vec!["+XIZ".to_string(), "-YYY".to_string()];
        assert_eq!(pauli_labels_roundtrip, expected_pauli_labels);
    }

    #[test]
    fn test_from_labels_and_back_with_right_to_left() {
        let pauli_labels = ["+XIZ".to_string(), "-YYY".to_string()];
        let pauli_list =
            PauliList::from_pauli_labels(3, &pauli_labels, PauliLabelOrder::RightToLeft)
                .expect("PauliList should be created without problems (all labels are valid)");

        // ZIX (when ordered left-to-right)
        assert!(!pauli_list.get_pauli_x(0, 0));
        assert!(pauli_list.get_pauli_z(0, 0));
        assert!(!pauli_list.get_pauli_x(0, 1));
        assert!(!pauli_list.get_pauli_z(0, 1));
        assert!(pauli_list.get_pauli_x(0, 2));
        assert!(!pauli_list.get_pauli_z(0, 2));
        assert!(!pauli_list.get_pauli_phase(0));
        // -YYY
        assert!(pauli_list.get_pauli_x(1, 0));
        assert!(pauli_list.get_pauli_z(1, 0));
        assert!(pauli_list.get_pauli_x(1, 1));
        assert!(pauli_list.get_pauli_z(1, 1));
        assert!(pauli_list.get_pauli_x(1, 2));
        assert!(pauli_list.get_pauli_z(1, 2));
        assert!(pauli_list.get_pauli_phase(1));

        let pauli_labels_roundtrip = pauli_list.to_pauli_labels(PauliLabelOrder::RightToLeft);
        assert_eq!(pauli_labels_roundtrip, pauli_labels);
    }

    #[test]
    fn test_from_invalid_labels() {
        let pauli_labels = ["+XIZ".to_string(), "1XY".to_string()];
        let pauli_list =
            PauliList::from_pauli_labels(3, &pauli_labels, PauliLabelOrder::RightToLeft);
        assert!(matches!(pauli_list, Err(PauliListError::InvalidLabel(_))));
    }

    #[test]
    fn test_commutation() {
        let pauli_labels = [
            "+XIZ".to_string(),
            "-YYY".to_string(),
            "IIX".to_string(),
            "III".to_string(),
        ];
        let pauli_list =
            PauliList::from_pauli_labels(3, &pauli_labels, PauliLabelOrder::LeftToRight)
                .expect("PauliList should be created without problems (all labels are valid)");
        assert!(pauli_list.commute(0, 1));
        assert!(!pauli_list.commute(2, 0));
        assert!(pauli_list.commute(3, 0));
        assert!(!pauli_list.commute(1, 2));
        assert!(pauli_list.commute(2, 2));
        assert!(pauli_list.commute(2, 3));
    }
}
