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

use qiskit_circuit::operations::StandardGate;

// The concept of chunks appears in the paper [1] and the implementation is partially
// adapted from [2].
//
// References:
//
// 1. Timothée Goubault de Brugière and Simon Martiel,
// *Faster and shorter synthesis of Hamiltonian simulation circuits*,
// [arXiv:2404.03280](https://arxiv.org/abs/2404.03280).
//
// 2. https://github.com/qiskit-community/rustiq-core/blob/main/src/synthesis/pauli_network/chunks.rs.
// The code in https://github.com/qiskit-community/rustiq-core is licensed under the MIT license.

// In what follows, Pauli labels should be read left-to-right, thus XY
// means X on the first qubit and Y on the second qubits.

// A two-qubit Pauli operator can be encoded as an integer in 0..16,
// see [TwoQubitPauliIndex].
// Internally, this packs the `x` and the `z` components of the two qubits as
// (x[0] << 3) | (z[0] << 2) | (x[1] << 1) | z[1].
// For instance, XY (read left-to-right) corresponds to 8 + 0 + 2 + 1 = 11.
// In this way, 2-qubit Paulis are given by
// [II, IZ, IX, IY, ZI, ZZ, ZX, ZY, XI, XZ, XX, XY, YI, YZ, YX, YY].

// For efficiency, the following table stores 2-qubit Pauli support sizes.
// For example, for 2-qubit Pauli ZI with index 4, its support size
// is given by PAULI_SUPPORT_SIZES[4] = 1.
pub(crate) static PAULI_SUPPORT_SIZES: [usize; 16] =
    [0, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2];

// A "chunk" is a small 2-qubit Clifford circuit constisting of some
// single-qubit Clifford gates followed either by CX(0, 1) or CX(1, 0).
// There are 18 chunks of interest, numbered 0..18, see reference [1]
// above.
pub(crate) const ALL_CHUNKS: [&[(StandardGate, &[usize])]; 18] = [
    &[(StandardGate::CX, &[0, 1])],
    &[(StandardGate::CX, &[1, 0])],
    &[(StandardGate::H, &[1]), (StandardGate::CX, &[0, 1])],
    &[(StandardGate::H, &[0]), (StandardGate::CX, &[1, 0])],
    &[(StandardGate::S, &[1]), (StandardGate::CX, &[0, 1])],
    &[(StandardGate::S, &[0]), (StandardGate::CX, &[1, 0])],
    &[(StandardGate::H, &[0]), (StandardGate::CX, &[0, 1])],
    &[(StandardGate::H, &[1]), (StandardGate::CX, &[1, 0])],
    &[
        (StandardGate::H, &[0]),
        (StandardGate::H, &[1]),
        (StandardGate::CX, &[0, 1]),
    ],
    &[
        (StandardGate::H, &[1]),
        (StandardGate::H, &[0]),
        (StandardGate::CX, &[1, 0]),
    ],
    &[
        (StandardGate::H, &[0]),
        (StandardGate::S, &[1]),
        (StandardGate::CX, &[0, 1]),
    ],
    &[
        (StandardGate::H, &[1]),
        (StandardGate::S, &[0]),
        (StandardGate::CX, &[1, 0]),
    ],
    &[(StandardGate::SX, &[0]), (StandardGate::CX, &[0, 1])],
    &[(StandardGate::SX, &[1]), (StandardGate::CX, &[1, 0])],
    &[
        (StandardGate::SX, &[0]),
        (StandardGate::H, &[1]),
        (StandardGate::CX, &[0, 1]),
    ],
    &[
        (StandardGate::SX, &[1]),
        (StandardGate::H, &[0]),
        (StandardGate::CX, &[1, 0]),
    ],
    &[
        (StandardGate::SX, &[0]),
        (StandardGate::S, &[1]),
        (StandardGate::CX, &[0, 1]),
    ],
    &[
        (StandardGate::SX, &[1]),
        (StandardGate::S, &[0]),
        (StandardGate::CX, &[1, 0]),
    ],
];

// Given a 2-qubit Pauli, we want to precompute its conjugation
// by various chunks according to the Schrödinger picture:
// C P C^\dagger.
//
// The following table is a visual representation of this data.
//
//      II IZ IX IY ZI ZZ ZX ZY XI XZ XX XY YI YZ YX YY
//   0: II ZZ IX ZY ZI IZ ZX IY XX YY XI YZ YX XY YI XZ
//   1: II IZ XX XY ZZ ZI YY YX XI XZ IX IY YZ YI ZY ZX
//   2: II IX ZZ ZY ZI ZX IZ IY XX XI YY YZ YX YI XY XZ
//   3: II IZ XX XY XI XZ IX IY ZZ ZI YY YX YZ YI ZY ZX
//   4: II ZZ ZY IX ZI IZ IY ZX XX YY YZ XI YX XY XZ YI
//   5: II IZ XX XY ZZ ZI YY YX YZ YI ZY ZX XI XZ IX IY
//   6: II ZZ IX ZY XX YY XI YZ ZI IZ ZX IY YX XY YI XZ
//   7: II XX IZ XY ZZ YY ZI YX XI IX XZ IY YZ ZY YI ZX
//   8: II IX ZZ ZY XX XI YY YZ ZI ZX IZ IY YX YI XY XZ
//   9: II XX IZ XY XI IX XZ IY ZZ YY ZI YX YZ ZY YI ZX
//  10: II ZZ ZY IX XX YY YZ XI ZI IZ IY ZX YX XY XZ YI
//  11: II XX IZ XY ZZ YY ZI YX YZ ZY YI ZX XI IX XZ IY
//  12: II ZZ IX ZY YX XY YI XZ XX YY XI YZ ZI IZ ZX IY
//  13: II XY XX IZ ZZ YX YY ZI XI IY IX XZ YZ ZX ZY YI
//  14: II IX ZZ ZY YX YI XY XZ XX XI YY YZ ZI ZX IZ IY
//  15: II XY XX IZ XI IY IX XZ ZZ YX YY ZI YZ ZX ZY YI
//  16: II ZZ ZY IX YX XY XZ YI XX YY YZ XI ZI IZ IY ZX
//  17: II XY XX IZ ZZ YX YY ZI YZ ZX ZY YI XI IY IX XZ
//
// This is the corresponding table using 2-qubit Pauli indices.
// CHUNK_CONJUGATION_TABLE[chunk_idx][pauli_pair_idx] represents
// the index of the 2-qubit Pauli we obtain by conjugation.
pub(crate) static CHUNK_CONJUGATION_TABLE: [[usize; 16]; 18] = [
    [0, 5, 2, 7, 4, 1, 6, 3, 10, 15, 8, 13, 14, 11, 12, 9],
    [0, 1, 10, 11, 5, 4, 15, 14, 8, 9, 2, 3, 13, 12, 7, 6],
    [0, 2, 5, 7, 4, 6, 1, 3, 10, 8, 15, 13, 14, 12, 11, 9],
    [0, 1, 10, 11, 8, 9, 2, 3, 5, 4, 15, 14, 13, 12, 7, 6],
    [0, 5, 7, 2, 4, 1, 3, 6, 10, 15, 13, 8, 14, 11, 9, 12],
    [0, 1, 10, 11, 5, 4, 15, 14, 13, 12, 7, 6, 8, 9, 2, 3],
    [0, 5, 2, 7, 10, 15, 8, 13, 4, 1, 6, 3, 14, 11, 12, 9],
    [0, 10, 1, 11, 5, 15, 4, 14, 8, 2, 9, 3, 13, 7, 12, 6],
    [0, 2, 5, 7, 10, 8, 15, 13, 4, 6, 1, 3, 14, 12, 11, 9],
    [0, 10, 1, 11, 8, 2, 9, 3, 5, 15, 4, 14, 13, 7, 12, 6],
    [0, 5, 7, 2, 10, 15, 13, 8, 4, 1, 3, 6, 14, 11, 9, 12],
    [0, 10, 1, 11, 5, 15, 4, 14, 13, 7, 12, 6, 8, 2, 9, 3],
    [0, 5, 2, 7, 14, 11, 12, 9, 10, 15, 8, 13, 4, 1, 6, 3],
    [0, 11, 10, 1, 5, 14, 15, 4, 8, 3, 2, 9, 13, 6, 7, 12],
    [0, 2, 5, 7, 14, 12, 11, 9, 10, 8, 15, 13, 4, 6, 1, 3],
    [0, 11, 10, 1, 8, 3, 2, 9, 5, 14, 15, 4, 13, 6, 7, 12],
    [0, 5, 7, 2, 14, 11, 9, 12, 10, 15, 13, 8, 4, 1, 3, 6],
    [0, 11, 10, 1, 5, 14, 15, 4, 13, 6, 7, 12, 8, 3, 2, 9],
];

// Precomputed change in support size for every (chunk, 2-qubit Pauli) pair
// (a negative value means the conjugation reduces the support).
pub(crate) static SUPPORT_DELTA: [[i8; 16]; 18] = build_support_delta();

const fn build_support_delta() -> [[i8; 16]; 18] {
    let mut table = [[0i8; 16]; 18];
    let mut chunk_idx = 0;
    while chunk_idx < 18 {
        let mut pair_idx = 0;
        while pair_idx < 16 {
            let conjugated_pair_idx = CHUNK_CONJUGATION_TABLE[chunk_idx][pair_idx];
            table[chunk_idx][pair_idx] = PAULI_SUPPORT_SIZES[conjugated_pair_idx] as i8
                - PAULI_SUPPORT_SIZES[pair_idx] as i8;
            pair_idx += 1;
        }
        chunk_idx += 1;
    }
    table
}

// For efficiency, we also precompute which conjugations reduce
// the size of the support set of a given 2-qubit Pauli.
pub(crate) static REDUCING_CHUNKS: [&[usize]; 16] = [
    &[],
    &[],
    &[],
    &[],
    &[],
    &[0, 1, 4, 5, 8, 9, 14, 15],
    &[2, 3, 4, 6, 7, 11, 12, 15],
    &[0, 2, 3, 9, 10, 13, 16, 17],
    &[],
    &[2, 3, 5, 6, 7, 10, 13, 14],
    &[0, 1, 8, 9, 10, 11, 12, 13],
    &[1, 4, 6, 7, 8, 15, 16, 17],
    &[],
    &[1, 2, 3, 8, 11, 12, 16, 17],
    &[0, 5, 6, 7, 9, 14, 16, 17],
    &[4, 5, 10, 11, 12, 13, 14, 15],
];

#[cfg(test)]
mod tests {

    use qiskit_circuit::Qubit;
    use qiskit_quantum_info::clifford::TwoQubitPauliIndex;
    use smallvec::SmallVec;

    use crate::clifford::utils::clifford_from_gate_sequence;
    use crate::evolution::chunks::{
        ALL_CHUNKS, CHUNK_CONJUGATION_TABLE, PAULI_SUPPORT_SIZES, REDUCING_CHUNKS, SUPPORT_DELTA,
    };

    /// Given a chunk index corresponding to a 2-qubit Clifford circuit `C`, and a two-qubit Pauli
    /// index corresponding to a Pauli `P`, computes the two-qubit Pauli index for `C^\dagger P C`.
    fn inverse_conjugate_chunk(chunk_idx: usize, pauli_idx: usize) -> usize {
        // Create a Clifford from the chunk
        let clifford_gates_vec: Vec<_> = ALL_CHUNKS[chunk_idx]
            .iter()
            .map(|(gate, indices)| {
                let qubits: SmallVec<_> = indices.iter().copied().map(Qubit::new).collect();
                (*gate, SmallVec::new(), qubits)
            })
            .collect();
        let cliff = clifford_from_gate_sequence(&clifford_gates_vec, 2);
        assert!(cliff.is_ok());
        let mut cliff = cliff.unwrap();

        // Convert pauli index to z and x components of the two qubits
        let (x0, z0, x1, z1) = TwoQubitPauliIndex::from_usize(pauli_idx).bits();

        // Use Clifford's evolve_pauli method to compute C^\dagger P C.
        let (_, evolved_z, evolved_x, indices_out) =
            cliff.evolve_pauli(&[z0, z1], &[x0, x1], &[0, 1]);

        // Explicitly compute the z and x components for the two qubits (as the output of evolve_pauli
        // is given in the sparse format).
        let (out_x0, out_z0, out_x1, out_z1) = match indices_out.as_slice() {
            [0, 1] => (evolved_x[0], evolved_z[0], evolved_x[1], evolved_z[1]),
            [0] => (evolved_x[0], evolved_z[0], false, false),
            [1] => (false, false, evolved_x[0], evolved_z[0]),
            [] => (false, false, false, false),
            _ => {
                unreachable!(
                    "The output qubits of the evolved Pauli are a sorted subset of [0, 1]."
                );
            }
        };

        // Return the index of the corresponding Pauli.
        TwoQubitPauliIndex::from_bits(out_x0, out_z0, out_x1, out_z1).as_usize()
    }

    /// Returns the size of the Pauli support (number of non-I terms) for the two-qubit Pauli
    /// with index pauli_idx.
    fn pauli_support_size(pauli_idx: usize) -> usize {
        let (x0, z0, x1, z1) = TwoQubitPauliIndex::from_usize(pauli_idx).bits();
        ((x0 || z0) as usize) + ((x1 || z1) as usize)
    }

    /// Test that the table CHUNK_CONJUGATION_TABLE is correct.
    #[test]
    fn test_chunk_conjugation_table() {
        for (chunk_idx, stored_pauli_indices) in CHUNK_CONJUGATION_TABLE.iter().enumerate() {
            for pauli_idx in 0..16 {
                // Note that CHUNK_CONJUGATION_TABLE stores results for C P C^\dagger,
                // while inverse_conjugate_chunk returns the result for C^\dagger P C.
                let evolved_id = inverse_conjugate_chunk(chunk_idx, pauli_idx);
                let expected_id = stored_pauli_indices[evolved_id];
                assert_eq!(expected_id, pauli_idx);
            }
        }
    }

    /// Test that the table PAULI_SUPPORT_SIZES is correct.    
    #[test]
    fn test_pauli_support_sizes() {
        for (pauli_idx, &stored_support_size) in PAULI_SUPPORT_SIZES.iter().enumerate() {
            assert_eq!(stored_support_size, pauli_support_size(pauli_idx));
        }
    }

    /// Test that the table REDUCING_CHUNKS is correct.
    #[test]
    fn check_reducing_chunks_table() {
        // Note that the table SUPPORT_DELTA is constructed automatically and thus is correct.
        for (pauli_idx, stored_reducing_chunks) in REDUCING_CHUNKS.iter().enumerate() {
            let stored: Vec<usize> = stored_reducing_chunks.to_vec();
            let computed: Vec<_> = (0..18)
                .filter(|chunk_idx| SUPPORT_DELTA[*chunk_idx][pauli_idx] < 0)
                .collect();
            assert_eq!(stored, computed);
        }
    }
}
