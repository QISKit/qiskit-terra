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

use approx::abs_diff_eq;
use num_complex::{Complex64, ComplexFloat};
use pyo3::Python;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::collections::BTreeSet;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

use crate::diagonal::diagonal_gate_circuit;
use crate::qsd::append;
use nalgebra::{Matrix2, MatrixView2, Vector2};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::{ArrayType, StandardGate, UnitaryGate};
use qiskit_circuit::packed_instruction::PackedOperation;
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

const RZ_PI2_MAT: Matrix2<Complex64> = Matrix2::new(RZ_PI2_00, C_ZERO, C_ZERO, RZ_PI2_11);

const H_MAT: Matrix2<Complex64> = Matrix2::new(
    c64(FRAC_1_SQRT_2, 0.0),
    c64(FRAC_1_SQRT_2, 0.0),
    c64(FRAC_1_SQRT_2, 0.0),
    c64(-FRAC_1_SQRT_2, 0.0),
);

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
/// Expand a diagonal that was computed over a simplified (reduced-control) UCGate back
/// to the full `2^num_qubits` size, by duplicating entries along the qubits that were
/// dropped from `new_ctrl` during simplification.
fn expand_diagonal(mut diag: Vec<Complex64>, new_ctrl: &[u32], num_qubits: u32) -> Vec<Complex64> {
    let active: Vec<u32> = new_ctrl.iter().map(|&x| num_qubits - x).collect();
    for i in 0..num_qubits {
        if i != 0 && !active.contains(&i) {
            let d = 1_usize << i;
            let mut new_diag: Vec<Complex64> = Vec::new();
            let n = diag.len();

            for j in 0..n {
                new_diag.push(diag[j]);
                if (j + 1) % d == 0 {
                    new_diag.extend_from_slice(&diag[j + 1 - d..j + 1]);
                }
            }
            diag = new_diag
        }
    }
    diag
}

/// Append a single-qubit unitary `mat` acting on `qubit` to `circuit` as a `UnitaryGate`.
fn push_1q_unitary(
    circuit: &mut CircuitData,
    mat: Matrix2<Complex64>,
    qubit: Qubit,
) -> Result<(), CircuitDataError> {
    circuit.push_packed_operation(
        PackedOperation::from_unitary(Box::new(UnitaryGate {
            array: ArrayType::OneQ(mat),
        })),
        None,
        &[qubit],
        &[],
    )?;
    Ok(())
}

/// Recursively decompose a uniformly controlled one-qubit gate (`single_qubit_gates`,
/// indexed by the bitstring of control values) into a `CircuitData` of single-qubit
/// gates and CX gates, following the method in [1].
///
/// If `up_to_diagonal` is false, the trailing diagonal is folded back into the circuit
/// via [`diagonal_gate_circuit`]. If `mux_simp` is true, [`simplify`] is used first to
/// drop controls the gate list doesn't actually depend on.
///
/// [1]: https://arxiv.org/pdf/quant-ph/0410066.pdf
pub(crate) fn dec_ucg_inner(
    single_qubit_gates: Vec<Matrix2<Complex64>>,
    num_qubits: u32,
    up_to_diagonal: bool,
    mux_simp: bool,
) -> Result<(CircuitData, Vec<Complex64>), CircuitDataError> {
    if num_qubits == 1 {
        let mut circuit = CircuitData::with_capacity(1, 0, 1, Param::Float(0.0))?;
        push_1q_unitary(&mut circuit, single_qubit_gates[0], Qubit(0))?;
        return Ok((circuit, vec![Complex64::ONE; 2]));
    }
    let num_contr = num_qubits - 1;
    let (q_controls, new_gates, raw_ctrls) = if mux_simp {
        let (ctrls, gates) = simplify(&single_qubit_gates, num_contr);
        let mut mapped: Vec<u32> = ctrls.iter().map(|&x| num_qubits - x).collect();
        mapped.reverse();
        (mapped, gates, ctrls)
    } else {
        let ctrls: Vec<u32> = (1..num_qubits).collect();
        // clone for q_controls, move for raw_ctrls
        (ctrls.clone(), single_qubit_gates, ctrls)
    };

    let simplified_num_qubits = q_controls.len() as u32 + 1;

    if simplified_num_qubits == 1 {
        let mut circuit = CircuitData::with_capacity(num_qubits, 0, 1, Param::Float(0.0))?;
        push_1q_unitary(&mut circuit, new_gates[0], Qubit(0))?;
        return Ok((circuit, vec![Complex64::ONE; 1_usize << num_qubits]));
    }

    let mut global_phase = 0.0;
    let mut circuit = CircuitData::with_capacity(num_qubits, 0, 0, Param::Float(global_phase))?;
    let mut gates = new_gates;
    let diagonal = dec_ucg_help(&mut gates, simplified_num_qubits);
    let n = gates.len();

    for (i, gate) in gates.iter().enumerate() {
        let squ = match i {
            0 => H_MAT * gate,
            i if i == n - 1 => gate * RZ_PI2_MAT * H_MAT,
            _ => H_MAT * (gate * RZ_PI2_MAT) * H_MAT,
        };
        push_1q_unitary(&mut circuit, squ, Qubit(0))?;

        // push CX after every gate except the last
        if i < n - 1 {
            let control_ind = (i + 1).trailing_zeros() as usize;
            circuit.push_standard_gate(
                StandardGate::CX,
                &[],
                &[Qubit(q_controls[control_ind]), Qubit(0)], // control, target
            )?;

            global_phase -= 0.25 * PI
        }
    }
    circuit.add_global_phase(&Param::Float(global_phase))?;
    let diagonal = expand_diagonal(diagonal, &raw_ctrls, num_qubits);
    if !up_to_diagonal {
        let mut diag_phases: Vec<f64> = diagonal.iter().map(|x| x.arg()).collect();
        let diag_circuit: CircuitData =
            diagonal_gate_circuit(&mut diag_phases, num_qubits as usize)?;
        let qubit_map: Vec<Qubit> = (0..num_qubits).map(Qubit).collect();

        append(&mut circuit, diag_circuit, &qubit_map)?;
    }

    Ok((circuit, diagonal))
}

/// Python-exposed entry point for decomposing a uniformly controlled one-qubit gate
/// (`UCGate`). See [`dec_ucg_inner`] for the algorithm; this wrapper only converts the
/// input matrices from NumPy and the output circuit to a `QuantumCircuit`.
#[pyfunction]
pub fn dec_ucg(
    py: Python,
    single_qubit_gates: Vec<PyReadonlyArray2<Complex64>>,
    num_qubits: u32,
    up_to_diagonal: bool,
    mux_simp: bool,
) -> PyResult<(Py<PyAny>, Vec<Complex64>)> {
    let gates: Vec<Matrix2<Complex64>> = single_qubit_gates
        .into_iter()
        .map(|x| {
            let res: Option<MatrixView2<Complex64>> = x.try_as_matrix();
            res.map(|m| m.into_owned()).ok_or_else(|| {
                PyValueError::new_err("expected a 2x2 unitary matrix for each single-qubit gate")
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let (circuit, diag) =
        dec_ucg_inner(gates, num_qubits, up_to_diagonal, mux_simp).map_err(PyErr::from)?;
    let qc = circuit.into_py_quantum_circuit(py)?;
    qc.setattr("name", "uc")?;
    Ok((qc.unbind(), diag))
}

/// Find and drop control qubits that `gate_list` doesn't actually depend on, using
/// repeated-block detection (see [`repetition_search`]). Based on [1].
///
/// # Returns
///
/// `(surviving_controls, simplified_gate_list)`
///
/// [1]: https://arxiv.org/abs/2409.05618
fn simplify(
    gate_list: &[Matrix2<Complex64>],
    num_ctrls: u32,
) -> (Vec<u32>, Vec<Matrix2<Complex64>>) {
    let mut c: BTreeSet<u32> = BTreeSet::new();
    for i in 0..num_ctrls {
        c.insert(i + 1);
    }

    let (new_mux, nc) = if gate_list.len() > 1 {
        let (found_nc, mux_copy) = repetition_search(gate_list, num_ctrls);
        (mux_copy, found_nc)
    } else {
        (gate_list.to_owned(), BTreeSet::new())
    };

    let new_ctrl: Vec<u32> = c.difference(&nc).copied().collect();
    (new_ctrl, new_mux)
}

/// Detect, for each power-of-two block size `d`, whether `mux` is made of repeated
/// blocks of that size — meaning the control bit at that position can be dropped.
/// Confirmed repetitions are nulled out in `mux_copy`.
///
/// # Returns
///
/// `(removed_controls, simplified_gate_list)`
fn repetition_search(
    mux: &[Matrix2<Complex64>],
    num_ctrls: u32,
) -> (BTreeSet<u32>, Vec<Matrix2<Complex64>>) {
    let mut nc: BTreeSet<u32> = BTreeSet::new();
    let mut mux_copy: Vec<Option<Matrix2<Complex64>>> = mux.iter().map(|x| Some(*x)).collect();

    let mut d = 1;

    while d <= mux.len() / 2 {
        let mut disentanglement = false;
        if abs_diff_eq!(mux[d], mux[0], epsilon = 1e-8) {
            let mux_org = mux_copy.clone();
            let repetitions = mux.len() / (2 * d);
            let mut p = 0;
            let mut broke_early = false;

            for _ in 0..repetitions {
                let valid = repetition_verify(p, d, mux, &mut mux_copy);
                p += 2 * d;
                if !valid {
                    mux_copy = mux_org;
                    broke_early = true;
                    break;
                }
            }
            if !broke_early {
                disentanglement = true;
            }
        }
        if disentanglement {
            let removed_contr = num_ctrls - d.trailing_zeros();
            nc.insert(removed_contr);
        }
        d *= 2;
    }
    let new_mux = mux_copy.into_iter().flatten().collect();
    (nc, new_mux)
}

/// Check whether the block of `d` gates starting at `base` equals the block starting at
/// `base + d`; if so, null out the duplicate entries in `mux_copy` and return `true`.
fn repetition_verify(
    mut base: usize,
    d: usize,
    mux: &[Matrix2<Complex64>],
    mux_copy: &mut [Option<Matrix2<Complex64>>],
) -> bool {
    let mut i = 0;
    let mut next_base = base + d;

    while i < d {
        if !abs_diff_eq!(mux[base], mux[next_base], epsilon = 1e-8) {
            return false;
        }
        mux_copy[next_base] = None;
        base += 1;
        next_base += 1;
        i += 1;
    }
    true
}

/// Recursively demultiplex `single_qubit_gates` one control at a time via
/// [`demultiplex_single_uc`], accumulating the trailing diagonal that the caller
/// (`dec_ucg_inner`) still needs to fold in. See [1].
///
/// [1]: https://arxiv.org/pdf/quant-ph/0410066.pdf
fn dec_ucg_help(single_qubit_gates: &mut [Matrix2<Complex64>], num_qubits: u32) -> Vec<Complex64> {
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
    diag
}

/// Register this module's Python-exposed functions (`dec_ucg`).
pub fn uc_gate(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dec_ucg, m)?)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::compute_2x2_eig;
    use super::dec_ucg_inner;
    use super::expand_diagonal;
    use super::simplify;
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

    // Helper: build a 2x2 unitary from a real angle (Ry-like rotation).
    // Matches the ry() helper in test/python/test_simplify_porting.py.
    fn ry(theta: f64) -> Matrix2<Complex64> {
        let c = Complex64::new((theta / 2.0).cos(), 0.0);
        let s = Complex64::new((theta / 2.0).sin(), 0.0);
        Matrix2::new(c, -s, s, c)
    }

    // Case 1 — matches Python: TestSimplify::test_single_gate
    // Single gate: no controls exist, gate is returned unchanged.
    #[test]
    fn test_simplify_single_gate() {
        let gate = ry(0.5);
        let gate_list = vec![gate];
        let num_ctrls = 0_u32; // log2(1) = 0

        let (new_ctrl, new_mux) = simplify(&gate_list, num_ctrls);

        assert!(new_ctrl.is_empty());
        assert_eq!(new_mux.len(), 1);
        assert!(abs_diff_eq!(new_mux[0], gate, epsilon = 1e-12));
    }

    // Case 2 — matches Python: TestSimplify::test_two_identical_gates_stub
    // [A, A]: period-1 repetition → control 1 removable, mux collapses to [A].
    #[test]
    fn test_simplify_two_identical_gates() {
        let gate = ry(1.0);
        let gate_list = vec![gate, gate];
        let num_ctrls = 1_u32;

        let (new_ctrl, new_mux) = simplify(&gate_list, num_ctrls);

        assert_eq!(new_ctrl, vec![] as Vec<u32>);
        assert_eq!(new_mux.len(), 1);
        assert!(abs_diff_eq!(new_mux[0], gate, epsilon = 1e-12));
    }

    // Case 3 — matches Python: TestSimplify::test_four_gates_two_controls_stub
    // [A, B, A, B]: period-2 repetition → control 1 removed, mux = [A, B].
    #[test]
    fn test_simplify_four_gates_two_controls() {
        let a = ry(0.3);
        let b = ry(0.9);
        let gate_list = vec![a, b, a, b];
        let num_ctrls = 2_u32;

        let (new_ctrl, new_mux) = simplify(&gate_list, num_ctrls);

        assert_eq!(new_ctrl, vec![2_u32]); // control 1 removed
        assert_eq!(new_mux.len(), 2); // [A, B] after Nones removed
        assert!(abs_diff_eq!(new_mux[0], a, epsilon = 1e-12));
        assert!(abs_diff_eq!(new_mux[1], b, epsilon = 1e-12));
    }

    // Case 4 — matches Python: TestSimplify::test_all_identical_stub
    // [A, A, A, A]: both controls removable, mux collapses to [A].
    #[test]
    fn test_simplify_all_identical() {
        let gate = ry(0.7);
        let gate_list = vec![gate; 4];
        let num_ctrls = 2_u32;

        let (new_ctrl, new_mux) = simplify(&gate_list, num_ctrls);

        assert_eq!(new_ctrl, vec![] as Vec<u32>);
        assert_eq!(new_mux.len(), 1);
        assert!(abs_diff_eq!(new_mux[0], gate, epsilon = 1e-12));
    }

    // Case 5 — matches Python: TestSimplify::test_no_simplification_possible
    // All different gates: nothing is simplified, all controls and gates kept.
    #[test]
    fn test_simplify_no_simplification_possible() {
        let gate_list = vec![ry(0.1), ry(0.2), ry(0.3), ry(0.4)];
        let num_ctrls = 2_u32;

        let (new_ctrl, new_mux) = simplify(&gate_list, num_ctrls);

        assert_eq!(new_ctrl, vec![1_u32, 2_u32]);
        assert_eq!(new_mux.len(), 4);
    }

    // Case 6 — matches Python: TestSimplify::test_controls_are_sorted_ascending
    // Controls must always be returned in ascending order (BTreeSet guarantee).
    #[test]
    fn test_simplify_controls_sorted_ascending() {
        let gate_list: Vec<_> = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            .iter()
            .map(|&t| ry(t))
            .collect();
        let num_ctrls = 3_u32; // log2(8) = 3

        let (new_ctrl, _) = simplify(&gate_list, num_ctrls);

        let mut sorted = new_ctrl.clone();
        sorted.sort();
        assert_eq!(new_ctrl, sorted, "controls must be in ascending order");
    }

    // Helper: complex value from real part only
    fn c(re: f64) -> Complex64 {
        Complex64::new(re, 0.0)
    }

    // Case: no simplification — new_ctrl = all controls, diagonal unchanged
    // num_qubits=3, new_ctrl=[1,2] (all controls survived), diag length stays 8
    #[test]
    fn test_expand_diagonal_no_expansion() {
        let diag: Vec<Complex64> = (1..=8).map(|x| c(x as f64)).collect();
        let new_ctrl = vec![1_u32, 2_u32]; // all controls for 3 qubits survived
        let result = expand_diagonal(diag.clone(), &new_ctrl, 3);
        // nothing removed → diagonal unchanged
        assert_eq!(result, diag);
    }

    // Case: control 1 removed (d=2^1=2), num_qubits=2, new_ctrl=[2] (only ctrl 2 survived)
    // Input diag has 2^1=2 entries (simplified), expand to 2^2=4
    // Python equivalent:
    //   i=1 not in [0, num_qubits-2=0] → wait, num_qubits=2, new_ctrl=[2]
    //   active = [num_qubits - 2] = [0]  ← qubit 0 is target, always active
    //   i=1 not in [0]+[0] → expand at d=2
    //   diag=[a,b]: j=0→[a], j=1→[b,a,b] → result=[a,b,a,b]
    #[test]
    fn test_expand_diagonal_one_control_removed() {
        // simplified diag: 2 entries for 1 qubit (target only)
        let diag = vec![c(1.0), c(2.0)];
        let new_ctrl: Vec<u32> = vec![]; // no controls survived → only target qubit remains
        let result = expand_diagonal(diag, &new_ctrl, 2);
        // i=1 is not active → expand: each pair [a,b] is repeated → [1,2,1,2]
        assert_eq!(result, vec![c(1.0), c(2.0), c(1.0), c(2.0)]);
    }

    // Case: 3 qubits, control 1 removed, control 2 survived
    // new_ctrl=[2], active=[num_qubits - 2]=[1]
    // i=1 is active (skip), i=2 not active → expand at d=4
    // Input diag: 4 entries (2 qubits: target + ctrl 2)
    // Expansion at d=4: after j=3, repeat last 4 → doubles to 8 entries
    #[test]
    fn test_expand_diagonal_one_of_two_controls_removed() {
        let diag = vec![c(1.0), c(2.0), c(3.0), c(4.0)];
        let new_ctrl = vec![2_u32]; // control 2 survived, control 1 removed
        // num_qubits=3: active = [3-2] = [1]
        // i=1 → active (skip); i=2 → not active, d=4
        // expand: j=0→[1], j=1→[1,2], j=2→[1,2,3], j=3→[1,2,3,4,1,2,3,4]
        let result = expand_diagonal(diag, &new_ctrl, 3);
        assert_eq!(
            result,
            vec![
                c(1.0),
                c(2.0),
                c(3.0),
                c(4.0),
                c(1.0),
                c(2.0),
                c(3.0),
                c(4.0)
            ]
        );
    }

    // matches Python: TestSimplify::test_controls_are_sorted_ascending
    // Controls must always be returned in ascending order (BTreeSet guarantee).
    // (kept here as cross-check that expand_diagonal preserves length correctly)
    #[test]
    fn test_expand_diagonal_length() {
        // for num_qubits=3, all controls removed → diag starts at 2 (target only)
        // two expansions: i=1 (d=2) doubles to 4, i=2 (d=4) doubles to 8
        let diag = vec![c(1.0), c(2.0)];
        let new_ctrl: Vec<u32> = vec![]; // all controls removed
        let result = expand_diagonal(diag, &new_ctrl, 3);
        assert_eq!(result.len(), 8); // 2^3
    }
    #[test]
    fn test_dec_ucg_inner_single_qubit() {
        let gate = ry(0.5);
        let (circuit, diag) = dec_ucg_inner(vec![gate], 1, true, false).unwrap();
        assert_eq!(circuit.num_qubits(), 1);
        assert_eq!(diag, vec![Complex64::ONE; 2]);
    }

    // dec_ucg_inner: full decomposition up_to_diagonal=false
    #[test]
    fn test_dec_ucg_inner_two_qubits_full() {
        let gates = vec![ry(0.3), ry(0.9)];
        let (circuit, diag) = dec_ucg_inner(gates, 2, false, false).unwrap();
        assert!(circuit.num_qubits() > 0);
        assert_eq!(diag.len(), 4);
    }

    // dec_ucg_inner: simplified_num_qubits == 1 (all controls removed)
    #[test]
    fn test_dec_ucg_inner_all_controls_simplified() {
        let gate = ry(0.7);
        let gates = vec![gate; 4]; // all identical → all controls removed
        let (circuit, _diag) = dec_ucg_inner(gates, 3, true, true).unwrap();
        assert!(circuit.num_qubits() > 0);
    }
}
