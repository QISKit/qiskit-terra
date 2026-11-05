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

use foldhash::fast::RandomState;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis, s};
use num_complex::Complex64;
use num_traits::Zero;
#[cfg(feature = "python")]
use pyo3::{PyErr, exceptions::PyValueError};
use qiskit_util::{
    IndexMap,
    complex::{C_ZERO, c64},
};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PauliCompressionError {
    #[error("{0} is too many qubits to convert to a matrix")]
    TooManyQubits(usize),
    #[error("Incompatible Z and X shapes: {z:?} and {x:?}")]
    IncompatibleShapes {
        z: (usize, usize),
        x: (usize, usize),
    },
}

#[cfg(feature = "python")]
impl From<PauliCompressionError> for PyErr {
    fn from(value: PauliCompressionError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

/// Temporary bit-compressed storage of the Pauli string.  The [coeffs] are reinterpreted to
/// include the old `phase` component in them directly, plus the factors of `-i` stemming from `Y`
/// components.  The result is that the [coeffs] now more directly represent entries in a matrix,
/// while [x_like] and [z_like] are no longer direct measures of `X` and `Z` elements (as in the ZX
/// convention), but are instead only a marker of the column and parity respectively.
///
/// In other words, `row_num ^ x_like` gives the column number of an element, while
/// `(row_num & z_like).count_ones()` counts multiplicative factors of `-1` to be applied to
/// `coeff` when placing it at `(row_num, col_num)` in an output matrix.
pub struct MatrixCompressedPaulis {
    pub(crate) num_qubits: u8,
    pub(crate) x_like: Vec<u64>,
    pub(crate) z_like: Vec<u64>,
    pub(crate) coeffs: Vec<Complex64>,
}

impl MatrixCompressedPaulis {
    /// The number of qubits this operator acts on.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits as usize
    }

    /// The number of explicitly stored operators in the sum.
    pub fn num_ops(&self) -> usize {
        self.coeffs.len()
    }

    pub fn from_zx_arrays(
        x: ArrayView2<bool>,
        z: ArrayView2<bool>,
        phases: ArrayView1<u8>,
        coeffs: ArrayView1<Complex64>,
    ) -> Result<Self, PauliCompressionError> {
        if x.shape() != z.shape() {
            return Err(PauliCompressionError::IncompatibleShapes {
                z: (z.nrows(), z.ncols()),
                x: (x.ncols(), x.nrows()),
            });
        }
        let num_qubits = x.ncols();
        // This is obviously way too big for a dense operator, and SciPy limits us to using `i64`
        // for the index and indptr types, so (except for some synthetic cases), it's not possible
        // for us to work with a larger matrix than this.
        if num_qubits > 63 {
            return Err(PauliCompressionError::TooManyQubits(num_qubits));
        }

        let x_like = pack_bits(x).expect("x should already be validated as <64 qubits");
        let z_like = pack_bits(z).expect("z should already be validated as <64 qubits");
        let coeffs = x_like
            .iter()
            .zip(z_like.iter())
            .zip(phases.iter().zip(coeffs.iter()))
            .map(|((xs, zs), (&phase, &coeff))| {
                let ys = (xs & zs).count_ones();
                match (phase as u32 + ys) % 4 {
                    0 => coeff,
                    1 => c64(coeff.im, -coeff.re),
                    2 => c64(-coeff.re, -coeff.im),
                    3 => c64(-coeff.im, coeff.re),
                    _ => unreachable!(),
                }
            })
            .collect::<Vec<_>>();

        if num_qubits == 0 {
            return Ok(Self {
                num_qubits: 0,
                x_like: Vec::new(),
                z_like: Vec::new(),
                coeffs,
            });
        }

        Ok(MatrixCompressedPaulis {
            num_qubits: num_qubits as u8,
            x_like,
            z_like,
            coeffs,
        })
    }

    /// Sum coefficients that correspond to the same Pauli operator; this reduces the number of
    /// explicitly stored operations, if there are duplicates.  After the summation, any terms that
    /// have become zero are dropped.
    pub fn combine(&mut self) {
        let mut hash_table = IndexMap::<(u64, u64), Complex64>::with_capacity_and_hasher(
            self.coeffs.len(),
            RandomState::default(),
        );
        for (key, coeff) in self
            .x_like
            .drain(..)
            .zip(self.z_like.drain(..))
            .zip(self.coeffs.drain(..))
        {
            *hash_table.entry(key).or_insert(C_ZERO) += coeff;
        }
        for ((x, z), coeff) in hash_table {
            if coeff.is_zero() {
                continue;
            }
            self.x_like.push(x);
            self.z_like.push(z);
            self.coeffs.push(coeff);
        }
    }

    /// Returns a C-ordered [Vec] of the 2D matrix.
    pub fn to_matrix_dense(&self, parallel: bool) -> Array2<Complex64> {
        let side = 1usize << self.num_qubits();
        let mut out = bytemuck::allocation::zeroed_vec(side * side);
        let write_row = |(i_row, row): (usize, &mut [Complex64])| {
            for ((&x_like, &z_like), &coeff) in self
                .x_like
                .iter()
                .zip(self.z_like.iter())
                .zip(self.coeffs.iter())
            {
                // Technically this discards part of the storable data, but in practice, a dense
                // operator with more than 32 qubits needs in the region of 1 ZiB memory.  We still use
                // `u64` to help sparse-matrix construction, though.
                let coeff = if (i_row as u32 & z_like as u32)
                    .count_ones()
                    .is_multiple_of(2)
                {
                    coeff
                } else {
                    -coeff
                };
                row[i_row ^ (x_like as usize)] += coeff;
            }
        };
        if parallel {
            out.par_chunks_mut(side).enumerate().for_each(write_row);
        } else {
            out.chunks_mut(side).enumerate().for_each(write_row);
        }
        Array2::from_shape_vec((side, side), out).expect("The shape should be side x side")
    }
}

/// Pack a 2D array of Booleans into a given width.  Returns an error if the input array is
/// too large to be packed into u64.
fn pack_bits(bool_arr: ArrayView2<bool>) -> Result<Vec<u64>, ()> {
    let num_qubits = bool_arr.shape()[1];
    if num_qubits > (u64::BITS as usize) {
        return Err(());
    }
    let slack = num_qubits % 8;
    let pack_row = |row: ArrayView1<bool>| -> u64 {
        let mut val: u64 = 0;
        let mut shift = 0;
        for chunk in row.exact_chunks(8) {
            val |= ((chunk[0] as u8
                | ((chunk[1] as u8) << 1)
                | ((chunk[2] as u8) << 2)
                | ((chunk[3] as u8) << 3)
                | ((chunk[4] as u8) << 4)
                | ((chunk[5] as u8) << 5)
                | ((chunk[6] as u8) << 6)
                | ((chunk[7] as u8) << 7)) as u64)
                << shift;
            shift += 8;
        }
        if slack > 0 {
            for (i, b) in row
                .slice(s![num_qubits - slack..num_qubits])
                .iter()
                .enumerate()
            {
                val |= (*b as u64) << (shift + i);
            }
        }
        val
    };
    Ok(bool_arr
        .axis_iter(Axis(0))
        .map(pack_row)
        .collect::<Vec<_>>())
}
