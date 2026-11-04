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

use nalgebra::{Matrix2, Quaternion, Unit, UnitQuaternion};
use ndarray::ArrayView2;
use num_complex::Complex64;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VersorU2Error {
    #[error("cannot act on gates with symbolic parameters")]
    Symbolic,
    #[error("multi-qubit gates have no versor representation")]
    MultiQubit,
    #[error("non-unitary instructions have no versor representation")]
    NonUnitary,
}

/// A versor (unit-quaternion) representation of a single-qubit gate.
///
/// $SU(2)$ is representable by versors.  The convention we use internally here is to associate the
/// versor basis to the Pauli matrices as:
///
///     1 => I,   (where `nalgebra` calls the scalar term `w`)
///     i => i Z,
///     j => i Y,
///     k => i X,
///
/// so, for example, the Pauli Z gate has a possible (phase, versor) representation
/// `(pi/2, [0, -1, 0, 0])`.
///
/// For a version of this that includes a phase term to make an entry in $U(2), see [VersorU2].
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct VersorSU2(pub UnitQuaternion<f64>);
impl VersorSU2 {
    /// Get the identity operator.
    #[inline]
    pub fn identity() -> Self {
        Self(UnitQuaternion::identity())
    }

    /// Get a representation of this gate in $U(2)$ with a defined phase.
    #[inline]
    pub fn with_phase(self, phase: f64) -> VersorU2 {
        VersorU2 { phase, su2: self }
    }

    /// Create a versor representation of an $SU(2)$ matrix directly from the quaternion
    /// representiation.
    ///
    /// This does not check that the input is normalized.
    #[inline]
    pub fn from_quaternion_unchecked(scalar: f64, iz: f64, iy: f64, ix: f64) -> Self {
        Self(Unit::new_unchecked(Quaternion::new(scalar, iz, iy, ix)))
    }

    /// Get the representation of an `RZ(angle)` gate.
    ///
    /// Internally this calculates `sin` and `cos`, so discrete-angle forms will be more efficient
    /// to be written explicitly.
    #[inline]
    pub fn from_rz(angle: f64) -> Self {
        let (sin, cos) = (angle * 0.5).sin_cos();
        Self(Unit::new_unchecked(Quaternion::new(cos, -sin, 0., 0.)))
    }

    /// Get the representation of an `RY(angle)` gate.
    ///
    /// Internally this calculates `sin` and `cos`, so discrete-angle forms will be more efficient
    /// to be written explicitly.
    #[inline]
    pub fn from_ry(angle: f64) -> Self {
        let (sin, cos) = (angle * 0.5).sin_cos();
        Self(Unit::new_unchecked(Quaternion::new(cos, 0., -sin, 0.)))
    }

    /// Get the representation of an `RX(angle)` gate.
    ///
    /// Internally this calculates `sin` and `cos`, so discrete-angle forms will be more efficient
    /// to be written explicitly.
    #[inline]
    pub fn from_rx(angle: f64) -> Self {
        let (sin, cos) = (angle * 0.5).sin_cos();
        Self(Unit::new_unchecked(Quaternion::new(cos, 0., 0., -sin)))
    }

    /// Fill a pre-allocated 2x2 Rust-native contiguous array with the Z-basis representation of
    /// this versor.
    #[inline]
    pub fn matrix_contiguous_into(&self, matrix: &mut [[Complex64; 2]; 2]) {
        let q = self.0.quaternion();
        matrix[0][0] = Complex64::new(q.w, q.i);
        matrix[0][1] = Complex64::new(q.j, q.k);
        matrix[1][0] = Complex64::new(-q.j, q.k);
        matrix[1][1] = Complex64::new(q.w, -q.i);
    }
}

/// A versor-based (unit-quaternion) representation of a single-qubit gate.
///
/// In general, a single-qubit gate is a member of the group $U(2)$, and the group $SU(2)$ is
/// isomorphic to the versors.  We can keep track of the complex phase separately to the rotation
/// action, to fully describe a member of $U(2)$.
///
/// See [VersorSU2] for the underlying quaternion representation.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct VersorU2 {
    /// The phase of the gate.  This should be multiplied as `exp(i * phase) * matrix(su2)` to
    /// retrieve the explicit matrix form.
    pub phase: f64,
    /// The element of $SU(2)$ that has the same action as this gate, up to a global phase.
    pub su2: VersorSU2,
}
impl VersorU2 {
    /// Get the identity operator in quaternion form.
    #[inline]
    pub fn identity() -> Self {
        Self {
            phase: 0.,
            su2: VersorSU2::identity(),
        }
    }

    /// Calculate the versor representation of this matrix, assuming it is unitary.
    #[inline(always)]
    fn from_matrix_unchecked<M: Matrix1q>(matrix: &M) -> Self {
        // `matrix` is in U(2), therefore it has some representation
        //
        //   exp(ix) . [[a + bi, c + di], [-c + di, a - bi]],
        //
        // and the matrix is in SU(2) with determinant 1.  We can find the phase angle `x` via the
        // determinant of the given, which we've been promised has magnitude 1, so we can phase it
        // out of the matrix terms by multiplying by the root of the conjugate (since `det(aA) =
        // a^dim(A) det(A)`).  The versor representation is then just `(x/2, [a, b, c, d])`.
        let det = matrix.get(0, 0) * matrix.get(1, 1) - matrix.get(0, 1) * matrix.get(1, 0);
        let inv_rot = det.conj().sqrt();
        Self {
            phase: 0.5 * det.arg(),
            su2: VersorSU2::from_quaternion_unchecked(
                (inv_rot * matrix.get(0, 0)).re,
                (inv_rot * matrix.get(0, 0)).im,
                (inv_rot * matrix.get(0, 1)).re,
                (inv_rot * matrix.get(0, 1)).im,
            ),
        }
    }

    /// Calculate the versor representation of a unitary matrix, assuming it is unitary.
    pub fn from_contiguous_unchecked(matrix: &[[Complex64; 2]; 2]) -> Self {
        Self::from_matrix_unchecked(matrix)
    }

    /// Calculate the versor representation of a unitary matrix, assuming it is unitary.
    pub fn from_ndarray_unchecked(matrix: &ArrayView2<Complex64>) -> Self {
        Self::from_matrix_unchecked(matrix)
    }

    /// Calculate the versor representation of a unitary matrix, assuming it is unitary.
    pub fn from_nalgebra_unchecked(matrix: &Matrix2<Complex64>) -> Self {
        Self::from_matrix_unchecked(matrix)
    }

    /// Calculate the versor representation of a unitary matrix.
    ///
    /// Returns the error state if `|| A+.A - 1 ||_2 > tol` for matrix `A`, where the norm is the
    /// Frobenius norm.
    fn from_matrix_with_tol<M: Matrix1q>(matrix: &M, tol: f64) -> Result<Self, VersorU2Error> {
        if unitary_frobenius_distance_square(matrix) > tol * tol {
            return Err(VersorU2Error::NonUnitary);
        }
        Ok(Self::from_matrix_unchecked(matrix))
    }

    /// Calculate the versor representation of a unitary matrix.
    ///
    /// Returns the error state if `|| A+.A - 1 ||_2 > tol` for matrix `A`, where the norm is the
    /// Frobenius norm.
    pub fn from_contiguous(matrix: &[[Complex64; 2]; 2], tol: f64) -> Result<Self, VersorU2Error> {
        Self::from_matrix_with_tol(matrix, tol)
    }

    /// Calculate the versor representation of a unitary matrix.
    ///
    /// Returns the error state if `|| A+.A - 1 ||_2 > tol` for matrix `A`, where the norm is the
    /// Frobenius norm.
    pub fn from_ndarray(matrix: &ArrayView2<Complex64>, tol: f64) -> Result<Self, VersorU2Error> {
        Self::from_matrix_with_tol(matrix, tol)
    }

    /// Calculate the versor representation of a unitary matrix.
    ///
    /// Returns the error state if `|| A+.A - 1 ||_2 > tol` for matrix `A`, where the norm is the
    /// Frobenius norm.
    pub fn from_nalgebra(matrix: &Matrix2<Complex64>, tol: f64) -> Result<Self, VersorU2Error> {
        Self::from_matrix_with_tol(matrix, tol)
    }

    /// Fill a pre-allocated 2x2 Rust-native contiguous array with the Z-basis representation of
    /// this versor.
    ///
    /// The inverse of this function is [from_contiguous].
    #[inline]
    pub fn matrix_contiguous_into(&self, matrix: &mut [[Complex64; 2]; 2]) {
        let phase = Complex64::from_polar(1., self.phase);
        self.su2.matrix_contiguous_into(matrix);
        for row in matrix.iter_mut() {
            for element in row.iter_mut() {
                *element *= phase;
            }
        }
    }

    /// Create a new Z-basis representation of this versor as a contiguous Rust-native array.
    ///
    /// The inverse of this function is [from_contiguous].
    #[inline]
    pub fn matrix_contiguous(&self) -> [[Complex64; 2]; 2] {
        let mut out = Default::default();
        self.matrix_contiguous_into(&mut out);
        out
    }
}
impl From<VersorSU2> for VersorU2 {
    fn from(val: VersorSU2) -> Self {
        Self {
            phase: 0.,
            su2: val,
        }
    }
}

/// Implement the `Mul` traits between two `Copy` types for the pairs `[(&T, U), (T, &U), (&T, &U)]`
/// by delegating the reference-based multiplications to the owned-based version using `Copy`.
macro_rules! impl_mul_refs {
    ($right:ty, $left:ty) => {
        impl ::std::ops::Mul<$left> for &$right {
            type Output = <$right as ::std::ops::Mul<$left>>::Output;
            fn mul(self, other: $left) -> Self::Output {
                *self * other
            }
        }
        impl ::std::ops::Mul<&$left> for $right {
            type Output = <$right as ::std::ops::Mul<$left>>::Output;
            fn mul(self, other: &$left) -> Self::Output {
                self * *other
            }
        }
        impl ::std::ops::Mul<&$left> for &$right {
            type Output = <$right as ::std::ops::Mul<$left>>::Output;
            fn mul(self, other: &$left) -> Self::Output {
                *self * *other
            }
        }
    };
}

impl ::std::ops::Mul for VersorSU2 {
    type Output = VersorSU2;
    fn mul(self, other: VersorSU2) -> Self::Output {
        VersorSU2(self.0 * other.0)
    }
}
impl_mul_refs!(VersorSU2, VersorSU2);

impl ::std::ops::Mul for VersorU2 {
    type Output = VersorU2;
    fn mul(self, other: VersorU2) -> Self::Output {
        VersorU2 {
            phase: self.phase + other.phase,
            su2: self.su2 * other.su2,
        }
    }
}
impl_mul_refs!(VersorU2, VersorU2);

impl ::std::ops::Mul<VersorU2> for VersorSU2 {
    type Output = VersorU2;
    fn mul(self, other: VersorU2) -> VersorU2 {
        VersorU2 {
            phase: other.phase,
            su2: self * other.su2,
        }
    }
}
impl_mul_refs!(VersorU2, VersorSU2);

impl ::std::ops::Mul<VersorSU2> for VersorU2 {
    type Output = VersorU2;
    fn mul(self, other: VersorSU2) -> VersorU2 {
        VersorU2 {
            phase: self.phase,
            su2: self.su2 * other,
        }
    }
}
impl_mul_refs!(VersorSU2, VersorU2);

/// A module-internal trait to simplify the code-generation of both the dynamic `ndarray` and the
/// static `&[[Complex64; 2]; 2]` and `Matrix2` paths.  Rather than making the user care
/// about importing it, we just expose the concretised methods using it through `VersorU2`.
trait Matrix1q {
    fn get(&self, row: usize, col: usize) -> Complex64;
}
impl<const N: usize, const M: usize> Matrix1q for [[Complex64; N]; M] {
    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> Complex64 {
        self[row][col]
    }
}
impl Matrix1q for ArrayView2<'_, Complex64> {
    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> Complex64 {
        self[(row, col)]
    }
}
impl Matrix1q for Matrix2<Complex64> {
    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> Complex64 {
        self[(row, col)]
    }
}

/// Calculate the squared Frobenius distance `|| A+ A - 1 ||_2^2`.
///
/// As a rule of thumb, if `A+ A = 1 + eps`, then the Frobenius distance from the identity will be
/// `sqrt(2)*eps`, and this function would return `2 * eps*eps`, up to round-off error.
fn unitary_frobenius_distance_square<M: Matrix1q>(matrix: &M) -> f64 {
    let topleft = matrix.get(0, 0).norm_sqr() + matrix.get(0, 1).norm_sqr();
    let botright = matrix.get(1, 0).norm_sqr() + matrix.get(1, 1).norm_sqr();
    let off =
        matrix.get(0, 0) * matrix.get(1, 0).conj() + matrix.get(0, 1) * matrix.get(1, 1).conj();
    (topleft - 1.).powi(2) + (botright - 1.).powi(2) + 2. * off.norm_sqr()
}

#[cfg(test)]
mod test {
    use super::*;

    use approx::AbsDiffEq;
    use ndarray::aview2;
    use qiskit_circuit::operations::{Operation, Param, STANDARD_GATE_SIZE, StandardGate};

    #[cfg(not(miri))]
    const ATOL: f64 = 1e-15;
    #[cfg(miri)]
    const ATOL: f64 = 1e-8;

    fn all_1q_gates() -> Vec<StandardGate> {
        (0..STANDARD_GATE_SIZE as u8)
            .filter_map(|x| {
                ::bytemuck::checked::try_cast::<_, StandardGate>(x)
                    .ok()
                    .filter(|gate| gate.num_qubits() == 1)
            })
            .collect()
    }

    #[test]
    fn each_1q_gate_has_correct_matrix() {
        let params = [0.25, -0.75, 1.25, 0.5].map(Param::Float);
        let mut fails = Vec::new();
        for gate in all_1q_gates() {
            let params = &params[0..gate.num_params() as usize];
            let direct_matrix = gate.matrix(params).unwrap();
            let versor_matrix = gate.versor_u2(params).unwrap().matrix_contiguous();
            if direct_matrix.abs_diff_ne(&aview2(&versor_matrix), ATOL) {
                fails.push((gate, direct_matrix, versor_matrix));
            }
        }
        assert_eq!(fails, [])
    }

    #[test]
    fn can_roundtrip_1q_gate_from_matrix() {
        let params = [0.25, -0.75, 1.25, 0.5].map(Param::Float);
        let mut fails = Vec::new();
        for gate in all_1q_gates() {
            let params = &params[0..gate.num_params() as usize];
            let direct_matrix = gate.matrix(params).unwrap();
            let versor_matrix = VersorU2::from_ndarray(&direct_matrix.view(), ATOL)
                .unwrap()
                .matrix_contiguous();
            if direct_matrix.abs_diff_ne(&aview2(&versor_matrix), ATOL) {
                fails.push((gate, direct_matrix, versor_matrix));
            }
        }
        assert_eq!(fails, [])
    }

    #[test]
    fn pairwise_multiplication_gives_correct_matrices() {
        // We have two pairs just so in the (x, x) case of iteration we're including two different
        // gates to make sure that any non-commutation is accounted for.
        let left_params = [0.25, -0.75, 1.25, 0.5].map(Param::Float);
        let right_params = [0.5, 1.25, -0.75, 0.25].map(Param::Float);

        let mut fails = Vec::new();
        for (left, right) in all_1q_gates().into_iter().zip(all_1q_gates()) {
            let left_params = &left_params[0..left.num_params() as usize];
            let right_params = &right_params[0..right.num_params() as usize];

            let direct_matrix = left
                .matrix(left_params)
                .unwrap()
                .dot(&right.matrix(right_params).unwrap());
            let versor_matrix = (left.versor_u2(left_params).unwrap()
                * right.versor_u2(right_params).unwrap())
            .matrix_contiguous();
            if direct_matrix.abs_diff_ne(&aview2(&versor_matrix), ATOL) {
                fails.push((left, right, direct_matrix, versor_matrix));
            }
        }
        assert_eq!(fails, [])
    }
}
