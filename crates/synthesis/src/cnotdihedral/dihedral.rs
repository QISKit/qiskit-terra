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

use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, azip, s};
use qiskit_circuit::Qubit;
use qiskit_circuit::operations::{Param, StandardGate};
use smallvec::smallvec;

use crate::clifford::utils::CliffordGatesVec;
use crate::linear::utils::check_invertible_binary_matrix_inner;

/// A CNOT-dihedral element, mirroring the data model of the Python
/// ``qiskit.quantum_info.CNOTDihedral`` class: an affine function given by an
/// invertible binary matrix ``linear`` and a binary vector ``shift``, together
/// with a phase polynomial of degree at most 3 over ``num_qubits`` Z_2
/// variables, with coefficients in Z_8.
///
/// The phase polynomial coefficients are stored in the same packed layout as
/// the Python ``SpecialPolynomial`` class: ``weight1[i]`` is the coefficient
/// of `x_i`, ``weight2`` stores the coefficients of `x_i x_j` for `i < j` and
/// ``weight3`` those of `x_i x_j x_k` for `i < j < k`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CNOTDihedral {
    pub num_qubits: usize,
    linear: Array2<bool>,
    shift: Array1<bool>,
    weight0: u8,
    weight1: Array1<u8>,
    weight2: Array1<u8>,
    weight3: Array1<u8>,
}

impl CNOTDihedral {
    /// The identity element on ``num_qubits`` qubits.
    pub fn identity(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            linear: Array2::from_shape_fn((num_qubits, num_qubits), |(i, j)| i == j),
            shift: Array1::from_elem(num_qubits, false),
            weight0: 0,
            weight1: Array1::zeros(num_qubits),
            weight2: Array1::zeros(num_qubits * num_qubits.saturating_sub(1) / 2),
            weight3: Array1::zeros(
                num_qubits * num_qubits.saturating_sub(1) * num_qubits.saturating_sub(2) / 6,
            ),
        }
    }

    /// Build an element from its components, checking that the dimensions are
    /// consistent.  The phase polynomial coefficients must already be reduced
    /// modulo 8.
    pub fn from_parts(
        linear: ArrayView2<bool>,
        shift: ArrayView1<bool>,
        weight0: u8,
        weight1: ArrayView1<u8>,
        weight2: ArrayView1<u8>,
        weight3: ArrayView1<u8>,
    ) -> Result<Self, String> {
        let num_qubits = shift.len();
        let nc2 = num_qubits * num_qubits.saturating_sub(1) / 2;
        let nc3 = num_qubits * num_qubits.saturating_sub(1) * num_qubits.saturating_sub(2) / 6;
        if num_qubits == 0
            || linear.shape() != [num_qubits, num_qubits]
            || weight1.len() != num_qubits
            || weight2.len() != nc2
            || weight3.len() != nc3
        {
            return Err("Invalid CNOTDihedral element.".to_string());
        }
        Ok(Self {
            num_qubits,
            linear: linear.to_owned(),
            shift: shift.to_owned(),
            weight0,
            weight1: weight1.to_owned(),
            weight2: weight2.to_owned(),
            weight3: weight3.to_owned(),
        })
    }

    /// Packed index of the quadratic term `x_i x_j`, requiring `i < j`.
    fn idx2(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < j && j < self.num_qubits);
        i * self.num_qubits - (i + 1) * i / 2 + (j - i - 1)
    }

    /// Packed index of the cubic term `x_i x_j x_k`, requiring `i < j < k`.
    fn idx3(&self, i: usize, j: usize, k: usize) -> usize {
        debug_assert!(i < j && j < k && k < self.num_qubits);
        let n = self.num_qubits;
        let (t1, t2) = (n - i, n - j);
        n * (n - 1) * (n - 2) / 6
            - (t1 - 3) * (t1 - 2) * (t1 - 1) / 6
            - (t2 - 2) * (t2 - 1) / 2
            - (n - k)
    }

    fn get1(&self, i: usize) -> u8 {
        self.weight1[i]
    }

    fn get2(&self, i: usize, j: usize) -> u8 {
        self.weight2[self.idx2(i, j)]
    }

    fn get3(&self, i: usize, j: usize, k: usize) -> u8 {
        self.weight3[self.idx3(i, j, k)]
    }

    /// Left-multiply the element by `X(i)`.
    pub fn append_x(&mut self, i: usize) {
        self.shift[i] ^= true;
    }

    /// Left-multiply the element by `CX(ctrl, tgt)`.
    pub fn append_cx(&mut self, ctrl: usize, tgt: usize) {
        let (ctrl_row, mut tgt_row) = self.linear.multi_slice_mut((s![ctrl, ..], s![tgt, ..]));
        azip!((t in &mut tgt_row, &c in &ctrl_row) *t ^= c);
        let ctrl_shift = self.shift[ctrl];
        self.shift[tgt] ^= ctrl_shift;
    }

    /// Left-multiply the element by the `k`-th power of `T` on qubit `i`.
    pub fn append_phase(&mut self, k: u8, i: usize) {
        // If the i-th bit is flipped, conjugate this gate.
        let k = if self.shift[i] {
            (7 * (k % 8)) % 8
        } else {
            k % 8
        };
        // Take all subsets of the support of row i of weight up to 3 and add
        // k * (-2)^(|subset| - 1) mod 8 to the corresponding term.
        let support: Vec<usize> = (0..self.num_qubits)
            .filter(|&col| self.linear[[i, col]])
            .collect();
        for (na, &a) in support.iter().enumerate() {
            self.weight1[a] = (self.weight1[a] + k) % 8;
            for (nb, &b) in support.iter().enumerate().skip(na + 1) {
                let idx = self.idx2(a, b);
                self.weight2[idx] = (self.weight2[idx] + 8 - (2 * k) % 8) % 8;
                for &c in support.iter().skip(nb + 1) {
                    let idx = self.idx3(a, b, c);
                    self.weight3[idx] = (self.weight3[idx] + (4 * k) % 8) % 8;
                }
            }
        }
    }

    /// Whether the phase polynomials of two elements coincide.
    fn poly_eq(&self, other: &Self) -> bool {
        self.weight0 == other.weight0
            && self.weight1 == other.weight1
            && self.weight2 == other.weight2
            && self.weight3 == other.weight3
    }
}

/// Push a phase gate `P(tpow * pi / 4)`.  The angle is evaluated in the same
/// order as the Python implementation (``tpow * np.pi / 4``) so that the
/// emitted floats are identical.
fn push_phase(gates: &mut CliffordGatesVec, tpow: u8, qubit: usize) {
    gates.push((
        StandardGate::Phase,
        smallvec![Param::Float(tpow as f64 * PI / 4.0)],
        smallvec![Qubit::new(qubit)],
    ));
}

fn push1(gates: &mut CliffordGatesVec, gate: StandardGate, qubit: usize) {
    gates.push((gate, smallvec![], smallvec![Qubit::new(qubit)]));
}

fn push2(gates: &mut CliffordGatesVec, gate: StandardGate, qubit0: usize, qubit1: usize) {
    gates.push((
        gate,
        smallvec![],
        smallvec![Qubit::new(qubit0), Qubit::new(qubit1)],
    ));
}

/// Decompose a 1-qubit or 2-qubit `CNOTDihedral` element into a gate sequence
/// with an optimal number of CX gates, following the structure theorems of
/// Garion & Cross, `arXiv:2006.12042 <https://arxiv.org/abs/2006.12042>`__.
pub fn synth_cnotdihedral_two_qubits_inner(
    elem: &CNOTDihedral,
) -> Result<CliffordGatesVec, String> {
    let mut gates = CliffordGatesVec::new();

    if elem.num_qubits > 2 {
        return Err(
            "Cannot decompose a CNOT-Dihedral element with more than 2 qubits. \
             use synth_cnotdihedral_full function instead."
                .to_string(),
        );
    }

    if elem.num_qubits == 1 {
        if elem.weight0 != 0 || !elem.linear[[0, 0]] {
            return Err("1-qubit element is not CNOT-Dihedral.".to_string());
        }
        let tpow0 = elem.get1(0);
        let xpow0 = elem.shift[0];
        if tpow0 > 0 {
            push_phase(&mut gates, tpow0, 0);
        }
        if xpow0 {
            push1(&mut gates, StandardGate::X, 0);
        }
        if tpow0 == 0 && !xpow0 {
            push1(&mut gates, StandardGate::I, 0);
        }
        return Ok(gates);
    }

    // The two-qubit case.
    if elem.weight0 != 0 {
        return Err("2-qubit element is not CNOT-Dihedral.".to_string());
    }
    let weight1 = [elem.get1(0) as i16, elem.get1(1) as i16];
    let weight2 = elem.get2(0, 1) as i16;
    let linear = [
        [elem.linear[[0, 0]], elem.linear[[0, 1]]],
        [elem.linear[[1, 0]], elem.linear[[1, 1]]],
    ];
    let shift = [elem.shift[0], elem.shift[1]];

    // Push the leading single-qubit phase and X corrections shared by all the
    // cosets below.
    let emit_prefix = |gates: &mut CliffordGatesVec, tpow0: i16, xpow0: bool, tpow1: i16, xpow1| {
        if tpow0 > 0 {
            push_phase(gates, tpow0 as u8, 0);
        }
        if xpow0 {
            push1(gates, StandardGate::X, 0);
        }
        if tpow1 > 0 {
            push_phase(gates, tpow1 as u8, 1);
        }
        if xpow1 {
            push1(gates, StandardGate::X, 1);
        }
    };
    // The phase-power corrections shared by all cosets whose linear part is
    // not the identity.
    let cx_like_powers = |xpow0: bool, xpow1: bool| -> (i16, i16, i16) {
        if xpow0 == xpow1 {
            let m = ((8 - weight2) / 2) % 4;
            (
                m,
                (weight1[0] - m).rem_euclid(8),
                (weight1[1] - m).rem_euclid(8),
            )
        } else {
            let m = (weight2 / 2) % 4;
            (
                m,
                (weight1[0] + m).rem_euclid(8),
                (weight1[1] + m).rem_euclid(8),
            )
        }
    };

    // CS subgroup
    if linear == [[true, false], [false, true]] {
        let [xpow0, xpow1] = shift;
        let (x0, x1) = (xpow0 as i16, xpow1 as i16);

        // Dihedral class
        if weight2 == 0 {
            let (tpow0, tpow1) = (weight1[0], weight1[1]);
            emit_prefix(&mut gates, tpow0, xpow0, tpow1, xpow1);
            if tpow0 == 0 && tpow1 == 0 && !xpow0 && !xpow1 {
                push1(&mut gates, StandardGate::I, 0);
                push1(&mut gates, StandardGate::I, 1);
            }
        }

        // CS-like class
        if (weight2 == 2 && xpow0 == xpow1) || (weight2 == 6 && xpow0 != xpow1) {
            let tpow0 = (weight1[0] - 2 * x1 - 4 * x0 * x1).rem_euclid(8);
            let tpow1 = (weight1[1] - 2 * x0 - 4 * x0 * x1).rem_euclid(8);
            emit_prefix(&mut gates, tpow0, xpow0, tpow1, xpow1);
            // CS gate is implemented using 2 CX gates
            push_phase(&mut gates, 1, 0);
            push_phase(&mut gates, 1, 1);
            push2(&mut gates, StandardGate::CX, 0, 1);
            push_phase(&mut gates, 7, 1);
            push2(&mut gates, StandardGate::CX, 0, 1);
        }

        // CSdg-like class
        if (weight2 == 6 && xpow0 == xpow1) || (weight2 == 2 && xpow0 != xpow1) {
            let tpow0 = (weight1[0] - 6 * x1 - 4 * x0 * x1).rem_euclid(8);
            let tpow1 = (weight1[1] - 6 * x0 - 4 * x0 * x1).rem_euclid(8);
            emit_prefix(&mut gates, tpow0, xpow0, tpow1, xpow1);
            // CSdg gate is implemented using 2 CX gates
            push_phase(&mut gates, 7, 0);
            push_phase(&mut gates, 7, 1);
            push2(&mut gates, StandardGate::CX, 0, 1);
            push_phase(&mut gates, 1, 1);
            push2(&mut gates, StandardGate::CX, 0, 1);
        }

        // CZ-like class
        if weight2 == 4 {
            let tpow0 = (weight1[0] - 4 * x1).rem_euclid(8);
            let tpow1 = (weight1[1] - 4 * x0).rem_euclid(8);
            emit_prefix(&mut gates, tpow0, xpow0, tpow1, xpow1);
            push2(&mut gates, StandardGate::CZ, 1, 0);
        }
    }

    // CX01-like class
    if linear == [[true, false], [true, true]] {
        let xpow0 = shift[0];
        let xpow1 = shift[1] != xpow0;
        let (m, tpow0, tpow1) = cx_like_powers(xpow0, xpow1);
        emit_prefix(&mut gates, tpow0, xpow0, tpow1, xpow1);
        push2(&mut gates, StandardGate::CX, 0, 1);
        if m > 0 {
            push_phase(&mut gates, m as u8, 1);
        }
    }

    // CX10-like class
    if linear == [[true, true], [false, true]] {
        let xpow1 = shift[1];
        let xpow0 = shift[0] != xpow1;
        let (m, tpow0, tpow1) = cx_like_powers(xpow0, xpow1);
        emit_prefix(&mut gates, tpow0, xpow0, tpow1, xpow1);
        push2(&mut gates, StandardGate::CX, 1, 0);
        if m > 0 {
            push_phase(&mut gates, m as u8, 0);
        }
    }

    // CX01*CX10-like class
    if linear == [[false, true], [true, true]] {
        let xpow1 = shift[0];
        let xpow0 = shift[1] != xpow1;
        let (m, tpow0, tpow1) = cx_like_powers(xpow0, xpow1);
        emit_prefix(&mut gates, tpow0, xpow0, tpow1, xpow1);
        push2(&mut gates, StandardGate::CX, 0, 1);
        push2(&mut gates, StandardGate::CX, 1, 0);
        if m > 0 {
            push_phase(&mut gates, m as u8, 1);
        }
    }

    // CX10*CX01-like class
    if linear == [[true, true], [true, false]] {
        let xpow0 = shift[1];
        let xpow1 = shift[0] != xpow0;
        let (m, tpow0, tpow1) = cx_like_powers(xpow0, xpow1);
        emit_prefix(&mut gates, tpow0, xpow0, tpow1, xpow1);
        push2(&mut gates, StandardGate::CX, 1, 0);
        push2(&mut gates, StandardGate::CX, 0, 1);
        if m > 0 {
            push_phase(&mut gates, m as u8, 0);
        }
    }

    // CX01*CX10*CX01-like class
    if linear == [[false, true], [true, false]] {
        let xpow0 = shift[1];
        let xpow1 = shift[0];
        let (m, tpow0, tpow1) = cx_like_powers(xpow0, xpow1);
        emit_prefix(&mut gates, tpow0, xpow0, tpow1, xpow1);
        push2(&mut gates, StandardGate::CX, 0, 1);
        push2(&mut gates, StandardGate::CX, 1, 0);
        if m > 0 {
            push_phase(&mut gates, m as u8, 1);
        }
        push2(&mut gates, StandardGate::CX, 0, 1);
    }

    Ok(gates)
}

/// Decompose a general `CNOTDihedral` element into a gate sequence.  The
/// number of CX gates is not necessarily optimal.  This follows the
/// randomized-benchmarking construction of Cross, Magesan, Bishop, Smolin &
/// Gambetta, npj Quantum Inf 2, 16012 (2016).
pub fn synth_cnotdihedral_general_inner(elem: &CNOTDihedral) -> Result<CliffordGatesVec, String> {
    let num_qubits = elem.num_qubits;

    // Make a copy of the element as we are going to reduce it to an identity.
    let mut elem_cpy = elem.clone();

    if !check_invertible_binary_matrix_inner(elem_cpy.linear.view()) {
        return Err("Linear part is not invertible.".to_string());
    }

    let mut circuit = CliffordGatesVec::new();

    // Do x gate for each qubit i where shift[i]=1
    for i in 0..num_qubits {
        if elem.shift[i] {
            push1(&mut circuit, StandardGate::X, i);
            elem_cpy.append_x(i);
        }
    }

    // Do Gauss elimination on the linear part by adding cx gates
    for i in 0..num_qubits {
        // set i-th element to be 1
        if !elem_cpy.linear[[i, i]] {
            for j in (i + 1)..num_qubits {
                if elem_cpy.linear[[j, i]] {
                    // swap qubits i and j
                    push2(&mut circuit, StandardGate::CX, j, i);
                    push2(&mut circuit, StandardGate::CX, i, j);
                    push2(&mut circuit, StandardGate::CX, j, i);
                    elem_cpy.append_cx(j, i);
                    elem_cpy.append_cx(i, j);
                    elem_cpy.append_cx(j, i);
                    break;
                }
            }
        }
        // make all the other elements in column i zero
        for j in 0..num_qubits {
            if j != i && elem_cpy.linear[[j, i]] {
                push2(&mut circuit, StandardGate::CX, i, j);
                elem_cpy.append_cx(i, j);
            }
        }
    }

    if elem_cpy.shift.iter().any(|&shift| shift)
        || elem_cpy
            .linear
            .indexed_iter()
            .any(|((i, j), &val)| val != (i == j))
    {
        return Err("Cannot do Gauss elimination on linear part.".to_string());
    }

    // Initialize new_elem to an identity CNOTDihedral element
    let mut new_elem = CNOTDihedral::identity(num_qubits);
    let mut new_circuit = CliffordGatesVec::new();

    // Do cx and phase gates to construct all monomials of weight 3
    for i in 0..num_qubits {
        for j in (i + 1)..num_qubits {
            for k in (j + 1)..num_qubits {
                if elem_cpy.get3(i, j, k) != 0 {
                    new_elem.append_cx(i, k);
                    new_elem.append_cx(j, k);
                    new_elem.append_phase(1, k);
                    new_elem.append_cx(i, k);
                    new_elem.append_cx(j, k);
                    push2(&mut new_circuit, StandardGate::CX, i, k);
                    push2(&mut new_circuit, StandardGate::CX, j, k);
                    push_phase(&mut new_circuit, 1, k);
                    push2(&mut new_circuit, StandardGate::CX, i, k);
                    push2(&mut new_circuit, StandardGate::CX, j, k);
                }
            }
        }
    }

    // Do cx and phase gates to construct all monomials of weight 2
    for i in 0..num_qubits {
        for j in (i + 1)..num_qubits {
            let tpow1 = elem_cpy.get2(i, j) as i16;
            let tpow2 = new_elem.get2(i, j) as i16;
            let tpow = ((tpow2 - tpow1) / 2).rem_euclid(4) as u8;
            if tpow != 0 {
                new_elem.append_cx(i, j);
                new_elem.append_phase(tpow, j);
                new_elem.append_cx(i, j);
                push2(&mut new_circuit, StandardGate::CX, i, j);
                push_phase(&mut new_circuit, tpow, j);
                push2(&mut new_circuit, StandardGate::CX, i, j);
            }
        }
    }

    // Do phase gates to construct all monomials of weight 1
    for i in 0..num_qubits {
        let tpow1 = elem_cpy.get1(i) as i16;
        let tpow2 = new_elem.get1(i) as i16;
        let tpow = (tpow1 - tpow2).rem_euclid(8) as u8;
        if tpow != 0 {
            new_elem.append_phase(tpow, i);
            push_phase(&mut new_circuit, tpow, i);
        }
    }

    if !elem.poly_eq(&new_elem) {
        return Err("Could not recover phase polynomial.".to_string());
    }

    // The final circuit is ``new_circuit`` composed with the inverse of
    // ``circuit``; the latter contains only (self-inverse) X and CX gates, so
    // its inverse is just the same gates in reverse order.
    new_circuit.extend(circuit.into_iter().rev());
    Ok(new_circuit)
}

/// Decompose a `CNOTDihedral` element into a gate sequence, using the optimal
/// two-qubit decomposition for up to 2 qubits and the general decomposition
/// otherwise.
pub fn synth_cnotdihedral_full_inner(elem: &CNOTDihedral) -> Result<CliffordGatesVec, String> {
    if elem.num_qubits < 3 {
        synth_cnotdihedral_two_qubits_inner(elem)
    } else {
        synth_cnotdihedral_general_inner(elem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replay a synthesized gate sequence onto an identity element, using the
    /// same gate actions as ``qiskit.quantum_info.operators.dihedral``.
    fn replay(num_qubits: usize, gates: &CliffordGatesVec) -> CNOTDihedral {
        let mut elem = CNOTDihedral::identity(num_qubits);
        for (gate, params, qubits) in gates {
            match gate {
                StandardGate::I => (),
                StandardGate::X => elem.append_x(qubits[0].index()),
                StandardGate::CX => elem.append_cx(qubits[0].index(), qubits[1].index()),
                StandardGate::CZ => {
                    let (q0, q1) = (qubits[0].index(), qubits[1].index());
                    elem.append_phase(7, q1);
                    elem.append_phase(7, q0);
                    elem.append_cx(q1, q0);
                    elem.append_phase(2, q0);
                    elem.append_cx(q1, q0);
                    elem.append_phase(7, q1);
                    elem.append_phase(7, q0);
                }
                StandardGate::Phase => {
                    let Param::Float(theta) = params[0] else {
                        panic!("unexpected parameter type");
                    };
                    let tpow = ((theta * 4.0 / PI).round() as i64).rem_euclid(8) as u8;
                    elem.append_phase(tpow, qubits[0].index());
                }
                _ => panic!("unexpected gate {gate:?}"),
            }
        }
        elem
    }

    /// Deterministically generate an n-qubit element by applying a pseudo-random
    /// sequence of X, T^k and CX generators derived from ``seed``.
    fn pseudo_random_elem(num_qubits: usize, seed: u64) -> CNOTDihedral {
        let mut elem = CNOTDihedral::identity(num_qubits);
        let mut state = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        let mut next = |bound: usize| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound
        };
        for _ in 0..(8 * num_qubits) {
            match next(3) {
                0 => elem.append_x(next(num_qubits)),
                1 => elem.append_phase(next(8) as u8, next(num_qubits)),
                _ => {
                    if num_qubits > 1 {
                        let ctrl = next(num_qubits);
                        let mut tgt = next(num_qubits);
                        if tgt == ctrl {
                            tgt = (tgt + 1) % num_qubits;
                        }
                        elem.append_cx(ctrl, tgt);
                    }
                }
            }
        }
        elem
    }

    #[test]
    fn test_two_qubits_roundtrip() {
        for num_qubits in [1, 2] {
            for seed in 0..50 {
                let elem = pseudo_random_elem(num_qubits, seed);
                let gates = synth_cnotdihedral_two_qubits_inner(&elem).unwrap();
                assert_eq!(replay(num_qubits, &gates), elem, "seed {seed}");
            }
        }
    }

    #[test]
    fn test_general_roundtrip() {
        for num_qubits in 1..6 {
            for seed in 0..20 {
                let elem = pseudo_random_elem(num_qubits, seed);
                let gates = synth_cnotdihedral_general_inner(&elem).unwrap();
                assert_eq!(replay(num_qubits, &gates), elem, "seed {seed}");
            }
        }
    }

    #[test]
    fn test_full_roundtrip() {
        for num_qubits in 1..6 {
            for seed in 0..20 {
                let elem = pseudo_random_elem(num_qubits, seed);
                let gates = synth_cnotdihedral_full_inner(&elem).unwrap();
                assert_eq!(replay(num_qubits, &gates), elem, "seed {seed}");
            }
        }
    }

    #[test]
    fn test_two_qubits_rejects_large_elements() {
        let elem = CNOTDihedral::identity(3);
        assert!(synth_cnotdihedral_two_qubits_inner(&elem).is_err());
    }
}
