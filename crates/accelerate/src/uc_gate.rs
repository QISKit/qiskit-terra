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

use num_complex::{Complex64, ComplexFloat};
use pyo3::Python;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

use nalgebra::{Matrix2, MatrixView2, Vector2};
use numpy::{IntoPyArray, PyReadonlyArray2, ToPyArray};

use qiskit_util::complex::{C_ZERO, IM, c64};

const EPS: f64 = 1e-10;

/// Compute the eigenvectors and eigenvalues for a 2x2 matrix
///
/// Based on formula from:
/// https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
///
/// # Returns
///
/// `(eigenvalues, eigenvectors)`
fn compute_2x2_eig(mat: Matrix2<Complex64>) -> ([Complex64; 2], Matrix2<Complex64>) {
    let a = mat[(0, 0)];
    let b = mat[(0, 1)];
    let c = mat[(1, 0)];
    let d = mat[(1, 1)];

    if c.abs() <= 1e-8 && b.abs() <= 1e-8 {
        let eigvals: [Complex64; 2] = [a, d];
        let eigenvectors = [
            [Complex64::ONE, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE],
        ]
        .into();
        return (eigvals, eigenvectors);
    }

    let trace = a + d;
    let det = (a * d) - (b * c);
    let disc_sqrt = (0.25 * trace.powi(2) - det).sqrt();
    let half_trace = 0.5 * trace;
    let l1 = half_trace + disc_sqrt;
    let l2 = half_trace - disc_sqrt;
    let eigvals: [Complex64; 2] = [l1, l2];
    let eigenvectors: Matrix2<Complex64> = if c.abs() >= 1e-8 {
        let mut v_1: Vector2<Complex64> = [l1 - d, c].into();
        let v_1_norm = v_1.norm();
        v_1.iter_mut().for_each(|x| *x /= v_1_norm);
        let mut v_2: Vector2<Complex64> = [l2 - d, c].into();
        let v_2_norm = v_2.norm();
        v_2.iter_mut().for_each(|x| *x /= v_2_norm);
        Matrix2::from_columns(&[v_1, v_2])
    } else {
        // we know b is not close to 0 due to previous check {
        let mut v_1: Vector2<Complex64> = [b, l1 - a].into();
        let v_1_norm = v_1.norm();
        v_1.iter_mut().for_each(|x| *x /= v_1_norm);
        let mut v_2: Vector2<Complex64> = [b, l2 - a].into();
        let v_2_norm = v_2.norm();
        v_2.iter_mut().for_each(|x| *x /= v_2_norm);
        Matrix2::from_columns(&[v_1, v_2])
    };
    (eigvals, eigenvectors)
}

// These constants are the non-zero elements of an RZ gate's unitary with an
// angle of pi / 2
const RZ_PI2_11: Complex64 = c64(FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
const RZ_PI2_00: Complex64 = c64(FRAC_1_SQRT_2, FRAC_1_SQRT_2);

/// This method implements the decomposition given in equation (3) in
/// https://arxiv.org/pdf/quant-ph/0410066.pdf.
///
/// The decomposition is used recursively to decompose uniformly controlled gates.
///
/// a,b = single qubit unitaries
/// v,u,r = outcome of the decomposition given in the reference mentioned above
///
/// (see there for the details).
fn demultiplex_single_uc(
    a: &Matrix2<Complex64>,
    b: &Matrix2<Complex64>,
) -> [Matrix2<Complex64>; 3] {
    let x = a * b.adjoint();
    let det_x = x.determinant();
    let x11 = x[(0, 0)] / det_x.sqrt();
    let phi = det_x.arg();

    let r1 = (IM / 2. * (PI / 2. - phi / 2. - x11.arg())).exp();
    let r2 = (IM / 2. * (PI / 2. - phi / 2. + x11.arg() + PI)).exp();

    let r = Matrix2::new(r1, C_ZERO, C_ZERO, r2);
    let (diag, mut u) = compute_2x2_eig(r * x * r);

    // If d is not equal to diag(i,-i), then we put it into this "standard" form
    // (see eq. (13) in https://arxiv.org/pdf/quant-ph/0410066.pdf) by interchanging
    // the eigenvalues and eigenvectors
    if (diag[0] + IM).abs() < EPS {
        u = Matrix2::new(u[(0, 1)], u[(0, 0)], u[(1, 1)], u[(1, 0)]);
    }

    let d = Matrix2::new(RZ_PI2_00, C_ZERO, C_ZERO, RZ_PI2_11);
    let v = d * u.adjoint() * r.adjoint() * b;
    [v, u, r]
}

#[pyfunction]
pub fn dec_ucg_help(
    py: Python,
    sq_gates: Vec<PyReadonlyArray2<Complex64>>,
    num_qubits: u32,
) -> (Vec<Py<PyAny>>, Py<PyAny>) {
    let mut single_qubit_gates: Vec<Matrix2<Complex64>> = sq_gates
        .into_iter()
        .map(|x| {
            let res: MatrixView2<Complex64> = x.try_as_matrix().unwrap();
            res.into_owned()
        })
        .collect();
    let mut diag: Vec<Complex64> = vec![Complex64::ONE; 2_usize.pow(num_qubits)];
    let num_controls = num_qubits - 1;
    for dec_step in 0..num_controls {
        let num_ucgs = 2_usize.pow(dec_step);
        // The decomposition works recursively and the following loop goes over the different
        // UCGates that arise in the decomposition
        for ucg_index in 0..num_ucgs {
            let len_ucg = 2_usize.pow(num_controls - dec_step);
            for i in 0..len_ucg / 2 {
                let shift = ucg_index * len_ucg;
                let a = single_qubit_gates[shift + i];
                let b = single_qubit_gates[shift + len_ucg / 2 + i];
                // Apply the decomposition for UCGates given in equation (3) in
                // https://arxiv.org/pdf/quant-ph/0410066.pdf
                // to demultiplex one control of all the num_ucgs uniformly-controlled gates
                // with log2(len_ucg) uniform controls

                let [v, u, r] = demultiplex_single_uc(&a, &b);

                // replace the single-qubit gates with v,u (the already existing ones
                // are not needed any more)
                single_qubit_gates[shift + i] = v;
                single_qubit_gates[shift + len_ucg / 2 + i] = u;
                // Now we decompose the gates D as described in Figure 4 in
                // https://arxiv.org/pdf/quant-ph/0410066.pdf and merge some of the gates
                // into the UCGates and the diagonal at the end of the circuit

                // Remark: The Rz(pi/2) rotation acting on the target qubit and the Hadamard
                // gates arising in the decomposition of D are ignored for the moment (they will
                // be added together with the C-NOT gates at the end of the decomposition
                // (in the method dec_ucg()))
                let r_conj_t = r.adjoint();
                if ucg_index < num_ucgs - 1 {
                    // Absorb the Rz(pi/2) rotation on the control into the UC-Rz gate and
                    // merge the UC-Rz rotation with the following UCGate,
                    // which hasn't been decomposed yet
                    let k = shift + len_ucg + i;

                    single_qubit_gates[k] *= r_conj_t;
                    single_qubit_gates[k]
                        .iter_mut()
                        .for_each(|x| *x *= RZ_PI2_00);
                    let k = k + len_ucg / 2;
                    single_qubit_gates[k] *= r;
                    single_qubit_gates[k]
                        .iter_mut()
                        .for_each(|x| *x *= RZ_PI2_11);
                } else {
                    // Absorb the Rz(pi/2) rotation on the control into the UC-Rz gate and merge
                    // the trailing UC-Rz rotation into a diagonal gate at the end of the circuit
                    for ucg_index_2 in 0..num_ucgs {
                        let shift_2 = ucg_index_2 * len_ucg;
                        let k = 2 * (i + shift_2);
                        diag[k] *= r_conj_t[(0, 0)] * RZ_PI2_00;
                        diag[k + 1] *= r_conj_t[(1, 1)] * RZ_PI2_00;
                        let k = len_ucg + k;
                        diag[k] *= r[(0, 0)] * RZ_PI2_11;
                        diag[k + 1] *= r[(1, 1)] * RZ_PI2_11;
                    }
                }
            }
        }
    }
    (
        single_qubit_gates
            .into_iter()
            .map(|x| x.to_pyarray(py).into_any().unbind())
            .collect(),
        diag.into_pyarray(py).into_any().unbind(),
    )
}

pub fn uc_gate(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(dec_ucg_help))?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::compute_2x2_eig;
    use approx::abs_diff_eq;
    use nalgebra::Matrix2;
    use num_complex::Complex64;
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    use rand_pcg::Pcg64Mcg;

    #[inline(always)]
    fn random_complex(rng: &mut Pcg64Mcg) -> Complex64 {
        Complex64::new(rng.sample(StandardNormal), rng.sample(StandardNormal))
    }

    fn check_eig(mat: Matrix2<Complex64>) {
        let (eigvals, eigenvectors) = compute_2x2_eig(mat);
        for i in [0, 1] {
            const EPS: f64 = 1e-13;
            assert!(abs_diff_eq!(
                mat * eigenvectors.column(i),
                eigenvectors.column(i).map(|x| x * eigvals[i]),
                epsilon = EPS
            ));
        }
    }

    #[test]
    fn test_eig() {
        let mut rng = Pcg64Mcg::seed_from_u64(42);
        for _ in 0..4096 {
            let mat: Matrix2<Complex64> = [
                [random_complex(&mut rng), random_complex(&mut rng)],
                [random_complex(&mut rng), random_complex(&mut rng)],
            ]
            .into();
            check_eig(mat);
        }
    }

    #[test]
    fn test_diagonal_eig() {
        let mut rng = Pcg64Mcg::seed_from_u64(43);

        for _ in 0..1024 {
            let mat = Matrix2::from_diagonal(
                &[random_complex(&mut rng), random_complex(&mut rng)].into(),
            );
            check_eig(mat);
        }
    }

    #[test]
    fn test_off_diagonal_eig() {
        let mut rng = Pcg64Mcg::seed_from_u64(44);
        for _ in 0..1024 {
            let mat: Matrix2<Complex64> = [
                [Complex64::ZERO, random_complex(&mut rng)],
                [random_complex(&mut rng), Complex64::ZERO],
            ]
            .into();
            check_eig(mat);
        }
    }
}
