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

use bytemuck::zeroed_vec;
use ndarray::{Array2, ArrayView2, linalg::kron};
use num_complex::Complex64;

use super::{BitTerm, MatrixError, SparseTermView};

pub fn create_matrix_with_zeros(num_qubits: u32) -> Result<Array2<Complex64>, MatrixError> {
    if num_qubits == 0 {
        return Err(MatrixError::ZeroQubits);
    }

    let dim = 1usize
        .checked_shl(num_qubits)
        .ok_or(MatrixError::LimitExceeded(num_qubits))?;

    let len = dim
        .checked_mul(dim)
        .ok_or(MatrixError::LimitExceeded(num_qubits))?;

    let data = zeroed_vec(len);
    let matrix = Array2::from_shape_vec((dim, dim), data).expect("shape fits data");
    Ok(matrix)
}

/// bit-compressed pauli string. each pauli term adds to one column per row.
/// provides an efficient way to compute the term's column index and sign.
struct PauliTerm {
    /// coefficient including phase
    phase_coeff: Complex64,
    /// mapping of qubit index to enabled X or Y operation. the rightmost bit is
    /// qubit 0. is Y if X and Z are enabled for an index.
    x_qubit_ops: u32,
    /// mapping of qubit index to enabled Z or Y operation. the rightmost bit is
    /// qubit 0. is Y if X and Z are enabled for an index.
    z_qubit_ops: u32,
}

pub fn add_term(matrix: &mut Array2<Complex64>, term: &SparseTermView) {
    if let Some(pauli) = maybe_compress_pauli(term) {
        add_term_pauli(matrix, &pauli);
    } else {
        add_term_kron(matrix, term);
    }
}

fn maybe_compress_pauli(term: &SparseTermView) -> Option<PauliTerm> {
    let mut pauli = PauliTerm {
        phase_coeff: term.coeff,
        x_qubit_ops: 0,
        z_qubit_ops: 0,
    };

    for (bit_term, qubit_idx) in term.bit_terms.iter().zip(term.indices) {
        let enable_op = |qubit_ops: &mut u32| *qubit_ops |= 1 << qubit_idx;

        match bit_term {
            BitTerm::X => {
                enable_op(&mut pauli.x_qubit_ops);
            }
            BitTerm::Y => {
                enable_op(&mut pauli.x_qubit_ops);
                enable_op(&mut pauli.z_qubit_ops);
                pauli.phase_coeff *= -Complex64::i();
            }
            BitTerm::Z => {
                enable_op(&mut pauli.z_qubit_ops);
            }
            _ => return None,
        }
    }

    Some(pauli)
}

fn add_term_pauli(matrix: &mut Array2<Complex64>, term: &PauliTerm) {
    for (i, mut row) in matrix.rows_mut().into_iter().enumerate() {
        let qubit_col = i ^ term.x_qubit_ops as usize;

        if (i as u32 & term.z_qubit_ops).count_ones().is_multiple_of(2) {
            row[qubit_col] += term.phase_coeff;
        } else {
            row[qubit_col] -= term.phase_coeff;
        }
    }
}

fn add_term_kron(matrix: &mut Array2<Complex64>, term: &SparseTermView) {
    let n = matrix.nrows().trailing_zeros();

    let mut local = Array2::from_elem((1, 1), term.coeff);
    for term in term.bit_terms {
        local = kron(&local, &get_bit_term_matrix(*term));
    }

    let support: u32 = term.indices.iter().fold(0u32, |acc, &q| acc | (1 << q));
    let identity_qubits: Vec<u32> = (0..n).filter(|q| support & (1 << q) == 0).collect();

    let n_free = 1usize << identity_qubits.len();
    let mut free_patterns = vec![0usize; n_free];
    for (bit_pos, &q) in identity_qubits.iter().enumerate() {
        for (pattern_idx, pattern) in free_patterns.iter_mut().enumerate() {
            if (pattern_idx >> bit_pos) & 1 == 1 {
                *pattern |= 1usize << q;
            }
        }
    }

    let scatter = |local_idx: usize| -> usize {
        term.indices
            .iter()
            .rev()
            .enumerate()
            .fold(0usize, |acc, (bit_pos, &q)| {
                acc | (((local_idx >> bit_pos) & 1) << q)
            })
    };

    let local_dim = local.nrows();
    for local_row in 0..local_dim {
        let base_row = scatter(local_row);
        for local_col in 0..local_dim {
            let val = local[(local_row, local_col)];
            if val == Complex64::ZERO {
                continue;
            }
            let base_col = scatter(local_col);
            for &pattern in &free_patterns {
                matrix[(base_row | pattern, base_col | pattern)] += val;
            }
        }
    }
}

fn get_bit_term_matrix(bit_term: BitTerm) -> ArrayView2<'static, Complex64> {
    let data = match bit_term {
        BitTerm::X => const { &[re(0.0), re(1.0), re(1.0), re(0.0)] },
        BitTerm::Y => const { &[re(0.0), im(-1.0), im(1.0), re(0.0)] },
        BitTerm::Z => const { &[re(1.0), re(0.0), re(0.0), re(-1.0)] },
        BitTerm::Plus => const { &[re(0.5), re(0.5), re(0.5), re(0.5)] },
        BitTerm::Minus => const { &[re(0.5), re(-0.5), re(-0.5), re(0.5)] },
        BitTerm::Right => const { &[re(0.5), im(-0.5), im(0.5), re(0.5)] },
        BitTerm::Left => const { &[re(0.5), im(0.5), im(-0.5), re(0.5)] },
        BitTerm::Zero => const { &[re(1.0), re(0.0), re(0.0), re(0.0)] },
        BitTerm::One => const { &[re(0.0), re(0.0), re(0.0), re(1.0)] },
    };

    ArrayView2::from_shape((2, 2), data).expect("shape fits data")
}

const fn re(n: f64) -> Complex64 {
    Complex64::new(n, 0.0)
}

const fn im(n: f64) -> Complex64 {
    Complex64::new(0.0, n)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use num_complex::c64;

    use super::*;
    use crate::sparse_observable::SparseObservable;

    const TOL: f64 = 1e-8;

    #[test]
    fn test_too_many_qubits() {
        let num_qubits = 0usize.count_zeros();

        let res = SparseObservable::new(num_qubits, vec![], vec![], vec![], vec![0])
            .expect("is coherent")
            .to_matrix();

        assert!(matches!(res, Err(MatrixError::LimitExceeded(_))));
    }

    #[test]
    fn test_zero_qubits() {
        let num_qubits = 0;

        let res = SparseObservable::new(num_qubits, vec![], vec![], vec![], vec![0])
            .expect("is coherent")
            .to_matrix();

        assert!(matches!(res, Err(MatrixError::ZeroQubits)));
    }

    #[test]
    fn test_zero_coeff() {
        let terms = &[(c64(0.0, 0.0), "II")];
        let obs = create_observable(terms);
        let res = obs.to_matrix().expect("no errors");

        let data = &[Complex64::ZERO; 16];
        let exp = ArrayView2::from_shape((4, 4), data).expect("shape fits data");
        assert_eq!(res, exp);
    }

    #[test]
    fn test_dimension_scaling() {
        const DIMS: &[usize] = &[2, 4, 8, 16, 32, 64, 128, 256];

        for (i, dim) in DIMS.iter().copied().enumerate() {
            let num_qubits = (i + 1) as u32;
            let obs = SparseObservable::new(num_qubits, vec![], vec![], vec![], vec![0])
                .expect("is coherent");

            let matrix = obs.to_matrix().expect("no errors");
            assert_eq!(matrix.dim(), (dim, dim));
        }
    }

    #[test]
    fn test_pauli_only_1() {
        let terms = &[
            (c64(-3.0, 0.0), "XI"),
            (c64(0.0, 4.4), "YZ"),
            (c64(0.2, -0.1), "YY"),
            (c64(66.12, 0.0), "ZZ"),
        ];
        let obs = create_observable(terms);
        let res = obs.to_matrix().expect("no errors");

        let data = &[
            // Row 1
            c64(66.12, 0.0),
            c64(0.0, 0.0),
            c64(1.4, 0.0),
            c64(-0.2, 0.1),
            // Row 2
            c64(0.0, 0.0),
            c64(-66.12, 0.0),
            c64(0.2, -0.1),
            c64(-7.4, 0.0),
            // Row 3
            c64(-7.4, 0.0),
            c64(0.2, -0.1),
            c64(-66.12, 0.0),
            c64(0.0, 0.0),
            // Row 4
            c64(-0.2, 0.1),
            c64(1.4, 0.0),
            c64(0.0, 0.0),
            c64(66.12, 0.0),
        ];

        let exp = ArrayView2::from_shape((4, 4), data).expect("shape fits data");
        assert_abs_diff_eq!(res, exp, epsilon = TOL);
    }

    #[test]
    fn test_pauli_only_2() {
        let terms = &[
            (c64(0.0, -0.5), "XY"),
            (c64(2.0, 0.0), "II"),
            (c64(-15.1, 0.0), "ZX"),
            (c64(0.0, 7.0), "XX"),
        ];
        let obs = create_observable(terms);
        let res = obs.to_matrix().expect("no errors");

        let data = &[
            // Row 1
            c64(2.0, 0.0),
            c64(-15.1, 0.0),
            c64(0.0, 0.0),
            c64(-0.5, 7.0),
            // Row 2
            c64(-15.1, 0.0),
            c64(2.0, 0.0),
            c64(0.5, 7.0),
            c64(0.0, 0.0),
            // Row 3
            c64(0.0, 0.0),
            c64(-0.5, 7.0),
            c64(2.0, 0.0),
            c64(15.1, 0.0),
            // Row 4
            c64(0.5, 7.0),
            c64(0.0, 0.0),
            c64(15.1, 0.0),
            c64(2.0, 0.0),
        ];

        let exp = ArrayView2::from_shape((4, 4), data).expect("shape fits data");
        assert_abs_diff_eq!(res, exp, epsilon = TOL);
    }

    #[test]
    fn test_pauli_only_3() {
        let terms = &[
            (c64(0.0, 0.0), "XY"),
            (c64(1.5, 0.0), "YX"),
            (c64(0.0, -9.1), "ZX"),
            (c64(2.0, 0.0), "YZ"),
        ];
        let obs = create_observable(terms);
        let res = obs.to_matrix().expect("no errors");

        let data = &[
            // Row 1
            c64(0.0, 0.0),
            c64(0.0, -9.1),
            c64(0.0, -2.0),
            c64(0.0, -1.5),
            // Row 2
            c64(0.0, -9.1),
            c64(0.0, 0.0),
            c64(0.0, -1.5),
            c64(0.0, 2.0),
            // Row 3
            c64(0.0, 2.0),
            c64(0.0, 1.5),
            c64(0.0, 0.0),
            c64(0.0, 9.1),
            // Row 4
            c64(0.0, 1.5),
            c64(0.0, -2.0),
            c64(0.0, 9.1),
            c64(0.0, 0.0),
        ];

        let exp = ArrayView2::from_shape((4, 4), data).expect("shape fits data");
        assert_abs_diff_eq!(res, exp, epsilon = TOL);
    }

    #[test]
    fn test_pauli_only_larger() {
        let terms = &[
            (c64(-3.0, 0.0), "XIX"),
            (c64(0.0, 4.4), "YZX"),
            (c64(0.2, -0.1), "YYI"),
            (c64(66.12, 0.0), "ZZZ"),
        ];
        let obs = create_observable(terms);
        let res = obs.to_matrix().expect("no errors");

        let data = &[
            // Row 1
            c64(66.12, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(1.4, 0.0),
            c64(-0.2, 0.1),
            c64(0.0, 0.0),
            // Row 2
            c64(0.0, 0.0),
            c64(-66.12, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(1.4, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(-0.2, 0.1),
            // Row 3
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(-66.12, 0.0),
            c64(0.0, 0.0),
            c64(0.2, -0.1),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(-7.4, 0.0),
            // Row 4
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(66.12, 0.0),
            c64(0.0, 0.0),
            c64(0.2, -0.1),
            c64(-7.4, 0.0),
            c64(0.0, 0.0),
            // Row 5
            c64(0.0, 0.0),
            c64(-7.4, 0.0),
            c64(0.2, -0.1),
            c64(0.0, 0.0),
            c64(-66.12, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            // Row 6
            c64(-7.4, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.2, -0.1),
            c64(0.0, 0.0),
            c64(66.12, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            // Row 7
            c64(-0.2, 0.1),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(1.4, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(66.12, 0.0),
            c64(0.0, 0.0),
            // Row 8
            c64(0.0, 0.0),
            c64(-0.2, 0.1),
            c64(1.4, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(-66.12, 0.0),
        ];

        let exp = ArrayView2::from_shape((8, 8), data).expect("shape fits data");
        assert_abs_diff_eq!(res, exp, epsilon = TOL);
    }

    #[test]
    fn test_with_projectors_1() {
        let terms = &[
            (c64(0.5, -1.0), "X+"),
            (c64(8.1, 0.0), "Y-"),
            (c64(0.7, -0.1), "Zr"),
            (c64(9.1, 0.0), "Il"),
            (c64(2.0, 0.0), "I0"),
            (c64(0.5, 0.0), "I1"),
        ];
        let obs = create_observable(terms);
        let res = obs.to_matrix().expect("no errors");

        let exp = obs.as_paulis().to_matrix().expect("no errors");
        assert_abs_diff_eq!(res, exp, epsilon = TOL);
    }

    #[test]
    fn test_with_projectors_2() {
        let terms = &[
            (c64(0.5, -1.0), "-X"),
            (c64(8.1, 0.0), "+Y"),
            (c64(0.7, -0.1), "lZ"),
            (c64(9.1, 0.0), "rI"),
            (c64(2.0, 0.0), "1X"),
            (c64(0.5, 0.0), "0X"),
        ];
        let obs = create_observable(terms);
        let res = obs.to_matrix().expect("no errors");

        let exp = obs.as_paulis().to_matrix().expect("no errors");
        assert_abs_diff_eq!(res, exp, epsilon = TOL);
    }

    #[test]
    fn test_with_projectors_larger() {
        let terms = &[
            (c64(0.0, -6.0), "Y-X"),
            (c64(5.5, 0.2), "+YZ"),
            (c64(0.0, -0.7), "lIr"),
            (c64(1.2, 0.0), "rI-"),
            (c64(3.0, 0.0), "1XY"),
            (c64(0.5, 0.0), "Z0X"),
        ];
        let obs = create_observable(terms);
        let res = obs.to_matrix().expect("no errors");

        let exp = obs.as_paulis().to_matrix().expect("no errors");
        assert_abs_diff_eq!(res, exp, epsilon = TOL);
    }

    fn create_observable<S: AsRef<str>>(terms: &[(Complex64, S)]) -> SparseObservable {
        let mut num_qubits = 0;
        let mut coeffs = vec![];
        let mut bit_terms = vec![];
        let mut indices = vec![];
        let mut boundaries = vec![0];

        for (coeff, term) in terms {
            let term = term.as_ref();

            num_qubits = term.len();
            coeffs.push(*coeff);

            let mut num_bit_terms = 0;
            for (i, ch) in term.as_bytes().iter().rev().enumerate() {
                let bit_term = BitTerm::try_from_u8(*ch).expect("ch is bit term");

                if let Some(non_identity) = bit_term {
                    bit_terms.push(non_identity);
                    indices.push(i as u32);
                    num_bit_terms += 1;
                }
            }

            let next_boundary = boundaries.last().expect("non empty") + num_bit_terms;
            boundaries.push(next_boundary);
        }

        SparseObservable::new(num_qubits as u32, coeffs, bit_terms, indices, boundaries)
            .expect("is coherent")
    }
}
