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

use std::str::FromStr;

use approx::{abs_diff_eq, relative_eq};
use num_complex::{Complex64, ComplexFloat};
use num_traits::Zero;
use smallvec::{SmallVec, smallvec};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, PI, TAU};

use nalgebra::{Matrix2, MatrixView2, U2};
use ndarray::linalg::kron;
use ndarray::prelude::*;
use numpy::PyReadonlyArray2;
use numpy::{IntoPyArray, ToPyArray};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::common::{DEFAULT_FIDELITY, HGATE, IPZ, TraceToFidelity, rx_matrix, rz_matrix};
use super::gate_sequence::{TwoQubitGateSequence, TwoQubitSequenceVec};
use super::weyl_decomposition::{__num_basis_gates, _num_basis_gates, TwoQubitWeylDecomposition};

use crate::QiskitError;
use crate::euler_one_qubit_decomposer::{
    ANGLE_ZERO_EPSILON, EulerBasis, EulerBasisSet, OneQubitGateSequence, angles_from_unitary,
    unitary_to_gate_sequence_inner,
};
use crate::linalg::{nalgebra_array_view, ndarray_to_faer};
use crate::matrix::two_qubit;

use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::gate_matrix::{CX_GATE, ONE_QUBIT_IDENTITY};
use qiskit_circuit::instruction::{Instruction, Parameters};
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{NoBlocks, Qubit};
use qiskit_util::complex::{C_M_ONE, C_ONE, IM, M_IM, c64};

// Worst case length is 5x 1q gates for each 1q decomposition + 1x 2q gate
// We might overallocate a bit if the euler basis is different but
// the worst case is just 16 extra elements with just a String and 2 smallvecs
// each. This is only transient though as the circuit sequences aren't long lived
// and are just used to create a QuantumCircuit or DAGCircuit when we return to
// Python space.
const TWO_QUBIT_SEQUENCE_DEFAULT_CAPACITY: usize = 21;

static K12R: Matrix2<Complex64> = Matrix2::new(
    c64(0., FRAC_1_SQRT_2),
    c64(FRAC_1_SQRT_2, 0.),
    c64(-FRAC_1_SQRT_2, 0.),
    c64(0., -FRAC_1_SQRT_2),
);

static K12R_DG: Matrix2<Complex64> = Matrix2::new(
    c64(0., -FRAC_1_SQRT_2),
    c64(-FRAC_1_SQRT_2, 0.),
    c64(FRAC_1_SQRT_2, 0.),
    c64(0., FRAC_1_SQRT_2),
);

static K12L: Matrix2<Complex64> =
    Matrix2::new(c64(0.5, 0.5), c64(0.5, 0.5), c64(-0.5, 0.5), c64(0.5, -0.5));

static K12L_DG: Matrix2<Complex64> = Matrix2::new(
    c64(0.5, -0.5),
    c64(-0.5, -0.5),
    c64(0.5, -0.5),
    c64(0.5, 0.5),
);

static K22L: Matrix2<Complex64> = Matrix2::new(
    c64(FRAC_1_SQRT_2, 0.),
    c64(-FRAC_1_SQRT_2, 0.),
    c64(FRAC_1_SQRT_2, 0.),
    c64(FRAC_1_SQRT_2, 0.),
);
static K22R: Matrix2<Complex64> = Matrix2::new(Complex64::ZERO, C_ONE, C_M_ONE, Complex64::ZERO);

#[derive(Clone, Debug)]
#[allow(non_snake_case)]
#[pyclass(
    module = "qiskit._accelerate.two_qubit_decompose",
    subclass,
    skip_from_py_object
)]
pub struct TwoQubitBasisDecomposer {
    gate: PackedOperation,
    gate_params: SmallVec<[f64; 3]>,
    basis_fidelity: f64,
    euler_basis: EulerBasis,
    pulse_optimize: Option<bool>,
    basis_decomposer: TwoQubitWeylDecomposition,
    #[pyo3(get)]
    super_controlled: bool,
    u0l: Matrix2<Complex64>,
    u0r: Matrix2<Complex64>,
    u1l: Matrix2<Complex64>,
    u1ra: Matrix2<Complex64>,
    u1rb: Matrix2<Complex64>,
    u2la: Matrix2<Complex64>,
    u2lb: Matrix2<Complex64>,
    u2ra: Matrix2<Complex64>,
    u2rb: Matrix2<Complex64>,
    u3l: Matrix2<Complex64>,
    u3r: Matrix2<Complex64>,
    q0l: Matrix2<Complex64>,
    q0r: Matrix2<Complex64>,
    q1la: Matrix2<Complex64>,
    q1lb: Matrix2<Complex64>,
    q1ra: Matrix2<Complex64>,
    q1rb: Matrix2<Complex64>,
    q2l: Matrix2<Complex64>,
    q2r: Matrix2<Complex64>,
}
impl TwoQubitBasisDecomposer {
    /// Return the KAK gate name
    pub fn gate_name(&self) -> &str {
        self.gate.name()
    }

    /// Compute the number of basis gates needed for a given unitary
    pub fn num_basis_gates_inner(&self, unitary: ArrayView2<Complex64>) -> PyResult<usize> {
        let u = ndarray_to_faer(unitary);
        __num_basis_gates(self.basis_decomposer.b, self.basis_fidelity, u)
    }

    /// Is the gate super controlled
    #[inline]
    pub fn super_controlled(&self) -> bool {
        self.super_controlled
    }

    fn decomp1_inner(
        &self,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Matrix2<Complex64>; 8]> {
        // FIXME: fix for z!=0 and c!=0 using closest reflection (not always in the Weyl chamber)
        smallvec![
            self.basis_decomposer.K2r.adjoint() * target.K2r,
            self.basis_decomposer.K2l.adjoint() * target.K2l,
            target.K1r * self.basis_decomposer.K1r.adjoint(),
            target.K1l * self.basis_decomposer.K1l.adjoint(),
        ]
    }

    fn decomp2_supercontrolled_inner(
        &self,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Matrix2<Complex64>; 8]> {
        smallvec![
            self.q2r * target.K2r,
            self.q2l * target.K2l,
            self.q1ra * rz_matrix(2. * target.b) * self.q1rb,
            self.q1la * rz_matrix(-2. * target.a) * self.q1lb,
            target.K1r * self.q0r,
            target.K1l * self.q0l,
        ]
    }

    fn decomp3_supercontrolled_inner(
        &self,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Matrix2<Complex64>; 8]> {
        smallvec![
            self.u3r * target.K2r,
            self.u3l * target.K2l,
            self.u2ra * rz_matrix(2. * target.b) * self.u2rb,
            self.u2la * rz_matrix(-2. * target.a) * self.u2lb,
            self.u1ra * rz_matrix(-2. * target.c) * self.u1rb,
            self.u1l,
            target.K1r * self.u0r,
            target.K1l * self.u0l,
        ]
    }

    /// Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
    /// two CNOT gates are needed.
    ///
    /// This first decomposes each unitary from the KAK decomposition into ZXZ on the source
    /// qubit of the CNOTs and XZX on the targets in order to commute operators to beginning and
    /// end of decomposition. The beginning and ending single qubit gates are then
    /// collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
    /// if performance is a concern.
    fn get_sx_vz_2cx_efficient_euler(
        &self,
        decomposition: &SmallVec<[Matrix2<Complex64>; 8]>,
        target_decomposed: &TwoQubitWeylDecomposition,
        use_xgate: bool,
    ) -> Option<TwoQubitGateSequence> {
        let mut gates = Vec::new();
        let mut global_phase = target_decomposed.global_phase;
        global_phase -= 2. * self.basis_decomposer.global_phase;
        let euler_q0: Vec<[f64; 3]> = decomposition
            .iter()
            .step_by(2)
            .map(|decomp| {
                let euler_angles = angles_from_unitary(
                    nalgebra_array_view::<Complex64, U2, U2>(decomp.as_view()),
                    EulerBasis::ZXZ,
                );
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();
        let euler_q1: Vec<[f64; 3]> = decomposition
            .iter()
            .skip(1)
            .step_by(2)
            .map(|decomp| {
                let euler_angles = angles_from_unitary(
                    nalgebra_array_view::<Complex64, U2, U2>(decomp.as_view()),
                    EulerBasis::XZX,
                );
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();
        let mut euler_matrix_q0 = rx_matrix(euler_q0[0][1]) * rz_matrix(euler_q0[0][0]);
        euler_matrix_q0 = rz_matrix(euler_q0[0][2] + euler_q0[1][0] + FRAC_PI_2) * euler_matrix_q0;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q0.as_view(), 0);
        let mut euler_matrix_q1 = rz_matrix(euler_q1[0][1]) * rx_matrix(euler_q1[0][0]);
        euler_matrix_q1 = rx_matrix(euler_q1[0][2] + euler_q1[1][0]) * euler_matrix_q1;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q1.as_view(), 1);
        gates.push((StandardGate::CX.into(), smallvec![], smallvec![0, 1]));
        if (euler_q0[1][1] - PI).abs() < ANGLE_ZERO_EPSILON {
            if use_xgate {
                gates.push((StandardGate::X.into(), smallvec![], smallvec![0]));
            } else {
                gates.push((StandardGate::SX.into(), smallvec![], smallvec![0]));
                gates.push((StandardGate::SX.into(), smallvec![], smallvec![0]));
            }
        } else {
            gates.push((StandardGate::SX.into(), smallvec![], smallvec![0]));
            gates.push((
                StandardGate::RZ.into(),
                smallvec![euler_q0[1][1] - PI],
                smallvec![0],
            ));
            gates.push((StandardGate::SX.into(), smallvec![], smallvec![0]));
        }
        gates.push((
            StandardGate::RZ.into(),
            smallvec![euler_q1[1][1]],
            smallvec![1],
        ));
        global_phase += FRAC_PI_2;
        gates.push((StandardGate::CX.into(), smallvec![], smallvec![0, 1]));
        let mut euler_matrix_q0 =
            rx_matrix(euler_q0[2][1]) * rz_matrix(euler_q0[1][2] + euler_q0[2][0] + FRAC_PI_2);
        euler_matrix_q0 = rz_matrix(euler_q0[2][2]) * euler_matrix_q0;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q0.as_view(), 0);
        let mut euler_matrix_q1 =
            rz_matrix(euler_q1[2][1]) * rx_matrix(euler_q1[1][2] + euler_q1[2][0]);
        euler_matrix_q1 = rx_matrix(euler_q1[2][2]) * euler_matrix_q1;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q1.as_view(), 1);
        Some(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }

    /// Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
    /// three CNOT gates are needed.
    ///
    /// This first decomposes each unitary from the KAK decomposition into ZXZ on the source
    /// qubit of the CNOTs and XZX on the targets in order commute operators to beginning and
    /// end of decomposition. Inserting Hadamards reverses the direction of the CNOTs and transforms
    /// a variable Rx -> variable virtual Rz. The beginning and ending single qubit gates are then
    /// collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
    /// if performance is a concern.
    fn get_sx_vz_3cx_efficient_euler(
        &self,
        decomposition: &SmallVec<[Matrix2<Complex64>; 8]>,
        target_decomposed: &TwoQubitWeylDecomposition,
    ) -> Option<TwoQubitGateSequence> {
        let mut gates = Vec::new();
        let mut global_phase = target_decomposed.global_phase;
        global_phase -= 3. * self.basis_decomposer.global_phase;
        global_phase = global_phase.rem_euclid(TAU);
        let atol = 1e-10; // absolute tolerance for floats
        // Decompose source unitaries to zxz
        let euler_q0: Vec<[f64; 3]> = decomposition
            .iter()
            .step_by(2)
            .map(|decomp| {
                let euler_angles = angles_from_unitary(
                    nalgebra_array_view::<Complex64, U2, U2>(decomp.as_view()),
                    EulerBasis::ZXZ,
                );
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();
        // Decompose target unitaries to xzx
        let euler_q1: Vec<[f64; 3]> = decomposition
            .iter()
            .skip(1)
            .step_by(2)
            .map(|decomp| {
                let euler_angles = angles_from_unitary(
                    nalgebra_array_view::<Complex64, U2, U2>(decomp.as_view()),
                    EulerBasis::XZX,
                );
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();

        let x12 = euler_q0[1][2] + euler_q0[2][0];
        let x12_is_non_zero = !abs_diff_eq!(x12, 0., epsilon = atol);
        let mut x12_is_old_mult = None;
        let mut x12_phase = 0.;
        let x12_is_pi_mult = abs_diff_eq!(x12.sin(), 0., epsilon = atol);
        if x12_is_pi_mult {
            x12_is_old_mult = Some(abs_diff_eq!(x12.cos(), -1., epsilon = atol));
            x12_phase = PI * x12.cos();
        }
        let x02_add = x12 - euler_q0[1][0];
        let x12_is_half_pi = abs_diff_eq!(x12, FRAC_PI_2, epsilon = atol);

        let mut euler_matrix_q0 = rx_matrix(euler_q0[0][1]) * rz_matrix(euler_q0[0][0]);
        if x12_is_non_zero && x12_is_pi_mult {
            euler_matrix_q0 = rz_matrix(euler_q0[0][2] - x02_add) * euler_matrix_q0;
        } else {
            euler_matrix_q0 = rz_matrix(euler_q0[0][2] + euler_q0[1][0]) * euler_matrix_q0;
        }
        euler_matrix_q0 = HGATE * euler_matrix_q0;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q0.as_view(), 0);

        let rx_0 = rx_matrix(euler_q1[0][0]);
        let rz = rz_matrix(euler_q1[0][1]);
        let rx_1 = rx_matrix(euler_q1[0][2] + euler_q1[1][0]);
        let mut euler_matrix_q1 = rz * rx_0;
        euler_matrix_q1 = rx_1 * euler_matrix_q1;
        euler_matrix_q1 = HGATE * euler_matrix_q1;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q1.as_view(), 1);

        gates.push((StandardGate::CX.into(), smallvec![], smallvec![1, 0]));

        if x12_is_pi_mult {
            // even or odd multiple
            if x12_is_non_zero {
                global_phase += x12_phase;
            }
            if x12_is_non_zero && x12_is_old_mult.unwrap() {
                gates.push((
                    StandardGate::RZ.into(),
                    smallvec![-euler_q0[1][1]],
                    smallvec![0],
                ));
            } else {
                gates.push((
                    StandardGate::RZ.into(),
                    smallvec![euler_q0[1][1]],
                    smallvec![0],
                ));
                global_phase += PI;
            }
        }
        if x12_is_half_pi {
            gates.push((StandardGate::SX.into(), smallvec![], smallvec![0]));
            global_phase -= FRAC_PI_4;
        } else if x12_is_non_zero && !x12_is_pi_mult {
            if self.pulse_optimize.is_none() {
                self.append_1q_sequence(&mut gates, &mut global_phase, rx_matrix(x12).as_view(), 0);
            } else {
                return None;
            }
        }
        if abs_diff_eq!(euler_q1[1][1], FRAC_PI_2, epsilon = atol) {
            gates.push((StandardGate::SX.into(), smallvec![], smallvec![1]));
            global_phase -= FRAC_PI_4
        } else if self.pulse_optimize.is_none() {
            self.append_1q_sequence(
                &mut gates,
                &mut global_phase,
                rx_matrix(euler_q1[1][1]).as_view(),
                1,
            );
        } else {
            return None;
        }
        gates.push((
            StandardGate::RZ.into(),
            smallvec![euler_q1[1][2] + euler_q1[2][0]],
            smallvec![1],
        ));
        gates.push((StandardGate::CX.into(), smallvec![], smallvec![1, 0]));
        gates.push((
            StandardGate::RZ.into(),
            smallvec![euler_q0[2][1]],
            smallvec![0],
        ));
        if abs_diff_eq!(euler_q1[2][1], FRAC_PI_2, epsilon = atol) {
            gates.push((StandardGate::SX.into(), smallvec![], smallvec![1]));
            global_phase -= FRAC_PI_4;
        } else if self.pulse_optimize.is_none() {
            self.append_1q_sequence(
                &mut gates,
                &mut global_phase,
                rx_matrix(euler_q1[2][1]).as_view(),
                1,
            );
        } else {
            return None;
        }
        gates.push((StandardGate::CX.into(), smallvec![], smallvec![1, 0]));
        let mut euler_matrix = rz_matrix(euler_q0[2][2] + euler_q0[3][0]) * HGATE;
        euler_matrix = rx_matrix(euler_q0[3][1]) * euler_matrix;
        euler_matrix = rz_matrix(euler_q0[3][2]) * euler_matrix;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix.as_view(), 0);

        let mut euler_matrix = rx_matrix(euler_q1[2][2] + euler_q1[3][0]) * HGATE;
        euler_matrix = rz_matrix(euler_q1[3][1]) * euler_matrix;
        euler_matrix = rx_matrix(euler_q1[3][2]) * euler_matrix;
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix.as_view(), 1);

        let out_unitary = compute_unitary(&gates, global_phase);
        // TODO: fix the sign problem to avoid correction here
        if abs_diff_eq!(
            target_decomposed.unitary_matrix[(0, 0)],
            -out_unitary[[0, 0]],
            epsilon = atol
        ) {
            global_phase += PI;
        }
        Some(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }

    fn append_1q_sequence(
        &self,
        gates: &mut TwoQubitSequenceVec,
        global_phase: &mut f64,
        unitary: MatrixView2<Complex64>,
        qubit: u8,
    ) {
        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(self.euler_basis);
        let sequence = unitary_to_gate_sequence_inner(
            nalgebra_array_view::<Complex64, U2, U2>(unitary),
            &target_1q_basis_list,
            qubit as usize,
            None,
            true,
            None,
        );
        if let Some(sequence) = sequence {
            *global_phase += sequence.global_phase;
            for gate in sequence.gates {
                gates.push((gate.0.into(), gate.1, smallvec![qubit]));
            }
        }
    }

    fn pulse_optimal_chooser(
        &self,
        best_nbasis: u8,
        decomposition: &SmallVec<[Matrix2<Complex64>; 8]>,
        target_decomposed: &TwoQubitWeylDecomposition,
    ) -> PyResult<Option<TwoQubitGateSequence>> {
        if self.pulse_optimize.is_some()
            && (best_nbasis == 0 || best_nbasis == 1 || best_nbasis > 3)
        {
            return Ok(None);
        }
        let mut use_xgate = false;
        match self.euler_basis {
            EulerBasis::ZSX => (),
            EulerBasis::ZSXX => {
                use_xgate = true;
            }
            _ => {
                if self.pulse_optimize.is_some() {
                    return Err(QiskitError::new_err(format!(
                        "'pulse_optimize' currently only works with ZSX basis ({} used)",
                        self.euler_basis.as_str()
                    )));
                } else {
                    return Ok(None);
                }
            }
        }
        if !matches!(
            self.gate.view(),
            OperationRef::StandardGate(StandardGate::CX)
        ) {
            if self.pulse_optimize.is_some() {
                return Err(QiskitError::new_err(
                    "pulse_optimizer currently only works with CNOT entangling gate",
                ));
            } else {
                return Ok(None);
            }
        }
        let res = if best_nbasis == 3 {
            self.get_sx_vz_3cx_efficient_euler(decomposition, target_decomposed)
        } else if best_nbasis == 2 {
            self.get_sx_vz_2cx_efficient_euler(decomposition, target_decomposed, use_xgate)
        } else {
            None
        };
        if self.pulse_optimize.is_some() && res.is_none() {
            return Err(QiskitError::new_err(
                "Failed to compute requested pulse optimal decomposition",
            ));
        }
        Ok(res)
    }

    pub fn new_inner(
        gate: PackedOperation,
        gate_params: SmallVec<[f64; 3]>,
        gate_matrix: ArrayView2<Complex64>,
        basis_fidelity: f64,
        euler_basis: EulerBasis,
        pulse_optimize: Option<bool>,
    ) -> PyResult<Self> {
        let basis_decomposer =
            TwoQubitWeylDecomposition::new_inner(gate_matrix, Some(DEFAULT_FIDELITY), None)?;
        let super_controlled = relative_eq!(basis_decomposer.a, FRAC_PI_4, max_relative = 1e-09)
            && relative_eq!(basis_decomposer.c, 0.0, max_relative = 1e-09);

        // Create some useful matrices U1, U2, U3 are equivalent to the basis,
        // expand as Ui = Ki1.Ubasis.Ki2
        let b = basis_decomposer.b;
        let temp = c64(0.5, -0.5);
        let k11l = Matrix2::new(
            temp * (M_IM * c64(0., -b).exp()),
            temp * c64(0., -b).exp(),
            temp * (M_IM * c64(0., b).exp()),
            temp * -(c64(0., b).exp()),
        );
        let k11r = Matrix2::new(
            FRAC_1_SQRT_2 * (IM * c64(0., -b).exp()),
            FRAC_1_SQRT_2 * -c64(0., -b).exp(),
            FRAC_1_SQRT_2 * c64(0., b).exp(),
            FRAC_1_SQRT_2 * (M_IM * c64(0., b).exp()),
        );
        let k32l_k21l = Matrix2::new(
            FRAC_1_SQRT_2 * c64(1., (2. * b).cos()),
            FRAC_1_SQRT_2 * (IM * (2. * b).sin()),
            FRAC_1_SQRT_2 * (IM * (2. * b).sin()),
            FRAC_1_SQRT_2 * c64(1., -(2. * b).cos()),
        );
        let temp = c64(0.5, 0.5);
        let k21r = Matrix2::new(
            temp * (M_IM * c64(0., -2. * b).exp()),
            temp * c64(0., -2. * b).exp(),
            temp * (IM * c64(0., 2. * b).exp()),
            temp * c64(0., 2. * b).exp(),
        );
        let k31l = Matrix2::new(
            FRAC_1_SQRT_2 * c64(0., -b).exp(),
            FRAC_1_SQRT_2 * c64(0., -b).exp(),
            FRAC_1_SQRT_2 * -c64(0., b).exp(),
            FRAC_1_SQRT_2 * c64(0., b).exp(),
        );
        let k31r = Matrix2::new(
            IM * c64(0., b).exp(),
            Complex64::zero(),
            Complex64::zero(),
            M_IM * c64(0., -b).exp(),
        );
        let temp = c64(0.5, 0.5);
        let k32r = Matrix2::new(
            temp * c64(0., b).exp(),
            temp * -c64(0., -b).exp(),
            temp * (M_IM * c64(0., b).exp()),
            temp * (M_IM * c64(0., -b).exp()),
        );
        let k1ld = basis_decomposer.K1l.adjoint();
        let k1rd = basis_decomposer.K1r.adjoint();
        let k2ld = basis_decomposer.K2l.adjoint();
        let k2rd = basis_decomposer.K2r.adjoint();
        // Pre-build the fixed parts of the matrices used in 3-part decomposition
        let u0l = k31l * k1ld;
        let u0r = k31r * k1rd;
        let u1l = k2ld * k32l_k21l * k1ld;
        let u1ra = k2rd * k32r;
        let u1rb = k21r * k1rd;
        let u2la = k2ld * K22L;
        let u2lb = k11l * k1ld;
        let u2ra = k2rd * K22R;
        let u2rb = k11r * k1rd;
        let u3l = k2ld * K12L;
        let u3r = k2rd * K12R;
        // Pre-build the fixed parts of the matrices used in the 2-part decomposition
        let q0l = K12L_DG * k1ld;
        let q0r = K12R_DG * IPZ * k1rd;
        let q1la = k2ld * k11l.adjoint();
        let q1lb = k11l * k1ld;
        let q1ra = k2rd * IPZ * k11r.adjoint();
        let q1rb = k11r * k1rd;
        let q2l = k2ld * K12L;
        let q2r = k2rd * K12R;

        Ok(TwoQubitBasisDecomposer {
            gate,
            gate_params,
            basis_fidelity,
            euler_basis,
            pulse_optimize,
            basis_decomposer,
            super_controlled,
            u0l,
            u0r,
            u1l,
            u1ra,
            u1rb,
            u2la,
            u2lb,
            u2ra,
            u2rb,
            u3l,
            u3r,
            q0l,
            q0r,
            q1la,
            q1lb,
            q1ra,
            q1rb,
            q2l,
            q2r,
        })
    }

    pub fn call_inner(
        &self,
        unitary: ArrayView2<Complex64>,
        basis_fidelity: Option<f64>,
        approximate: bool,
        _num_basis_uses: Option<u8>,
    ) -> PyResult<TwoQubitGateSequence> {
        let basis_fidelity = if !approximate {
            1.0
        } else {
            basis_fidelity.unwrap_or(self.basis_fidelity)
        };
        let target_decomposed =
            TwoQubitWeylDecomposition::new_inner(unitary, Some(DEFAULT_FIDELITY), None)?;
        let traces = self.traces(&target_decomposed);
        let best_nbasis = _num_basis_uses.unwrap_or_else(|| {
            traces
                .into_iter()
                .enumerate()
                .map(|(idx, trace)| (idx, trace.trace_to_fid() * basis_fidelity.powi(idx as i32)))
                .min_by(|(_idx1, fid1), (_idx2, fid2)| fid2.partial_cmp(fid1).unwrap())
                .unwrap()
                .0 as u8
        });
        let decomposition = match best_nbasis {
            0 => decomp0_inner(&target_decomposed),
            1 => self.decomp1_inner(&target_decomposed),
            2 => self.decomp2_supercontrolled_inner(&target_decomposed),
            3 => self.decomp3_supercontrolled_inner(&target_decomposed),
            _ => unreachable!("Invalid basis to use"),
        };
        let pulse_optimize = self.pulse_optimize.unwrap_or(true);
        let sequence = if pulse_optimize {
            self.pulse_optimal_chooser(best_nbasis, &decomposition, &target_decomposed)?
        } else {
            None
        };
        if let Some(seq) = sequence {
            return Ok(seq);
        }
        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(self.euler_basis);
        let euler_decompositions: SmallVec<[Option<OneQubitGateSequence>; 8]> = decomposition
            .iter()
            .map(|decomp| {
                unitary_to_gate_sequence_inner(
                    nalgebra_array_view::<Complex64, U2, U2>(decomp.as_view()),
                    &target_1q_basis_list,
                    0,
                    None,
                    true,
                    None,
                )
            })
            .collect();
        let mut gates = Vec::with_capacity(TWO_QUBIT_SEQUENCE_DEFAULT_CAPACITY);
        let mut global_phase = target_decomposed.global_phase;
        global_phase -= best_nbasis as f64 * self.basis_decomposer.global_phase;
        if best_nbasis == 2 {
            global_phase += PI;
        }
        for i in 0..best_nbasis as usize {
            if let Some(euler_decomp) = &euler_decompositions[2 * i] {
                for gate in &euler_decomp.gates {
                    gates.push((gate.0.into(), gate.1.clone(), smallvec![0]));
                }
                global_phase += euler_decomp.global_phase
            }
            if let Some(euler_decomp) = &euler_decompositions[2 * i + 1] {
                for gate in &euler_decomp.gates {
                    gates.push((gate.0.into(), gate.1.clone(), smallvec![1]));
                }
                global_phase += euler_decomp.global_phase
            }
            gates.push((self.gate.clone(), self.gate_params.clone(), smallvec![0, 1]));
        }
        if let Some(euler_decomp) = &euler_decompositions[2 * best_nbasis as usize] {
            for gate in &euler_decomp.gates {
                gates.push((gate.0.into(), gate.1.clone(), smallvec![0]));
            }
            global_phase += euler_decomp.global_phase
        }
        if let Some(euler_decomp) = &euler_decompositions[2 * best_nbasis as usize + 1] {
            for gate in &euler_decomp.gates {
                gates.push((gate.0.into(), gate.1.clone(), smallvec![1]));
            }
            global_phase += euler_decomp.global_phase
        }
        Ok(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }
}

fn decomp0_inner(target: &TwoQubitWeylDecomposition) -> SmallVec<[Matrix2<Complex64>; 8]> {
    smallvec![target.K1r * target.K2r, target.K1l * target.K2l,]
}

type PickleNewArgs<'a> = (Py<PyAny>, Py<PyAny>, f64, &'a str, Option<bool>);

#[pymethods]
impl TwoQubitBasisDecomposer {
    fn __getnewargs__(&self, py: Python) -> PyResult<PickleNewArgs<'_>> {
        let params: SmallVec<[Param; 3]> =
            self.gate_params.iter().map(|x| Param::Float(*x)).collect();
        Ok((
            match self.gate.view() {
                OperationRef::StandardGate(standard) => {
                    standard.create_py_op(py, Some(params), None)?.into_any()
                }
                OperationRef::PyCustom(inst) => inst.ob.clone_ref(py),
                OperationRef::Unitary(unitary) => unitary.create_py_op(py, None)?.into_any(),
                _ => unreachable!("decomposer gate must be a gate"),
            },
            self.basis_decomposer.unitary_matrix(py),
            self.basis_fidelity,
            self.euler_basis.as_str(),
            self.pulse_optimize,
        ))
    }

    #[new]
    #[pyo3(signature=(gate, gate_matrix, basis_fidelity=1.0, euler_basis="U", pulse_optimize=None))]
    fn new(
        gate: OperationFromPython<NoBlocks>,
        gate_matrix: PyReadonlyArray2<Complex64>,
        basis_fidelity: f64,
        euler_basis: &str,
        pulse_optimize: Option<bool>,
    ) -> PyResult<Self> {
        if gate.operation.try_control_flow().is_some() {
            return Err(PyValueError::new_err(
                "Only gates are supported by two qubit decomposer",
            ));
        }
        let gate_params: PyResult<SmallVec<[f64; 3]>> = gate
            .params_view()
            .iter()
            .map(|x| match x {
                Param::Float(val) => Ok(*val),
                _ => Err(PyValueError::new_err(
                    "Only unparameterized gates are supported as KAK gate",
                )),
            })
            .collect();
        TwoQubitBasisDecomposer::new_inner(
            gate.operation,
            gate_params?,
            gate_matrix.as_array(),
            basis_fidelity,
            EulerBasis::from_str(euler_basis).map_err(PyValueError::new_err)?,
            pulse_optimize,
        )
    }

    fn traces(&self, target: &TwoQubitWeylDecomposition) -> [Complex64; 4] {
        [
            4. * c64(
                target.a.cos() * target.b.cos() * target.c.cos(),
                target.a.sin() * target.b.sin() * target.c.sin(),
            ),
            4. * c64(
                (FRAC_PI_4 - target.a).cos()
                    * (self.basis_decomposer.b - target.b).cos()
                    * target.c.cos(),
                (FRAC_PI_4 - target.a).sin()
                    * (self.basis_decomposer.b - target.b).sin()
                    * target.c.sin(),
            ),
            c64(4. * target.c.cos(), 0.),
            c64(4., 0.),
        ]
    }

    /// Decompose target :math:`\sim U_d(x, y, z)` with :math:`0` uses of the basis gate.
    /// Result :math:`U_r` has trace:
    ///
    /// .. math::
    ///
    ///     \Big\vert\text{Tr}(U_r\cdot U_\text{target}^{\dag})\Big\vert =
    ///     4\Big\vert (\cos(x)\cos(y)\cos(z)+ j \sin(x)\sin(y)\sin(z)\Big\vert
    ///
    /// which is optimal for all targets and bases
    #[staticmethod]
    fn decomp0(py: Python, target: &TwoQubitWeylDecomposition) -> SmallVec<[Py<PyAny>; 2]> {
        decomp0_inner(target)
            .into_iter()
            .map(|x| x.to_pyarray(py).into_any().unbind())
            .collect()
    }

    /// Decompose target :math:`\sim U_d(x, y, z)` with :math:`1` use of the basis gate
    /// math:`\sim U_d(a, b, c)`.
    /// Result :math:`U_r` has trace:
    ///
    /// .. math::
    ///
    ///     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^{\dag})\Big\vert =
    ///     4\Big\vert \cos(x-a)\cos(y-b)\cos(z-c) + j \sin(x-a)\sin(y-b)\sin(z-c)\Big\vert
    ///
    /// which is optimal for all targets and bases with ``z==0`` or ``c==0``.
    fn decomp1(&self, py: Python, target: &TwoQubitWeylDecomposition) -> SmallVec<[Py<PyAny>; 4]> {
        self.decomp1_inner(target)
            .into_iter()
            .map(|x| x.to_pyarray(py).into_any().unbind())
            .collect()
    }

    /// Decompose target :math:`\sim U_d(x, y, z)` with :math:`2` uses of the basis gate.
    ///
    /// For supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b, result :math:`U_r` has trace
    ///
    /// .. math::
    ///
    ///     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^\dag) \Big\vert = 4\cos(z)
    ///
    /// which is the optimal approximation for basis of CNOT-class :math:`\sim U_d(\pi/4, 0, 0)`
    /// or DCNOT-class :math:`\sim U_d(\pi/4, \pi/4, 0)` and any target. It may
    /// be sub-optimal for :math:`b \neq 0` (i.e. there exists an exact decomposition for any target
    /// using :math:`B \sim U_d(\pi/4, \pi/8, 0)`, but it may not be this decomposition).
    /// This is an exact decomposition for supercontrolled basis and target :math:`\sim U_d(x, y, 0)`.
    /// No guarantees for non-supercontrolled basis.
    fn decomp2_supercontrolled(
        &self,
        py: Python,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Py<PyAny>; 6]> {
        self.decomp2_supercontrolled_inner(target)
            .into_iter()
            .map(|x| x.to_pyarray(py).into_any().unbind())
            .collect()
    }

    /// Decompose target with :math:`3` uses of the basis.
    ///
    /// This is an exact decomposition for supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b,
    /// and any target. No guarantees for non-supercontrolled basis.
    fn decomp3_supercontrolled(
        &self,
        py: Python,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Py<PyAny>; 8]> {
        self.decomp3_supercontrolled_inner(target)
            .into_iter()
            .map(|x| x.to_pyarray(py).into_any().unbind())
            .collect()
    }

    /// Synthesizes a two qubit unitary matrix into a :class:`.DAGCircuit` object
    ///
    /// Args:
    ///     unitary (ndarray): A 4x4 unitary matrix in the form of a numpy complex array
    ///         representing the gate to synthesize
    ///     basis_fidelity (float): The target fidelity of the synthesis. This is a floating point
    ///         value between 1.0 and 0.0.
    ///     approximate (bool): Whether to enable approximation. If set to false this is equivalent
    ///         to setting basis_fidelity to 1.0.
    ///
    /// Returns:
    ///     DAGCircuit: The decomposed circuit for the given unitary.
    #[pyo3(signature = (unitary, basis_fidelity=None, approximate=true, _num_basis_uses=None))]
    fn to_dag(
        &self,
        unitary: PyReadonlyArray2<Complex64>,
        basis_fidelity: Option<f64>,
        approximate: bool,
        _num_basis_uses: Option<u8>,
    ) -> PyResult<DAGCircuit> {
        let array = unitary.as_array();
        let sequence = self.call_inner(array, basis_fidelity, approximate, _num_basis_uses)?;
        let mut dag = DAGCircuit::with_capacity(2, 0, None, Some(sequence.gates.len()), None, None);
        dag.set_global_phase_f64(sequence.global_phase);
        dag.add_qubit_unchecked(ShareableQubit::new_anonymous())?;
        dag.add_qubit_unchecked(ShareableQubit::new_anonymous())?;
        let mut builder = dag.into_builder();
        for (gate, params, qubits) in sequence.gates {
            let qubits: Vec<Qubit> = qubits.iter().map(|x| Qubit(*x as u32)).collect();
            let params = Parameters::Params(params.iter().map(|x| Param::Float(*x)).collect());
            builder.apply_operation_back(
                gate,
                &qubits,
                &[],
                Some(params),
                None,
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
        }
        Ok(builder.build())
    }

    /// Synthesizes a two qubit unitary matrix into a :class:`.CircuitData` object
    ///
    /// Args:
    ///     unitary (ndarray): A 4x4 unitary matrix in the form of a numply complex array
    ///         representing the gate to synthesize
    ///     basis_fidelity (float): The target fidelity of the synthesis. This is a floating point
    ///         value between 1.0 and 0.0.
    ///     approximate (bool): Whether to enable approximation. If set to false this is equivalent
    ///         to setting basis_fidelity to 1.0.
    ///
    /// Returns:
    ///     CircuitData: The decomposed circuit for the given unitary.
    #[pyo3(signature = (unitary, basis_fidelity=None, approximate=true, _num_basis_uses=None))]
    fn to_circuit(
        &self,
        unitary: PyReadonlyArray2<Complex64>,
        basis_fidelity: Option<f64>,
        approximate: bool,
        _num_basis_uses: Option<u8>,
    ) -> PyResult<PyCircuitData> {
        let array = unitary.as_array();
        let sequence = self.call_inner(array, basis_fidelity, approximate, _num_basis_uses)?;
        Ok(CircuitData::from_packed_operations(
            2,
            0,
            sequence.gates.into_iter().map(|(gate, params, qubits)| {
                Ok((
                    gate,
                    params.iter().map(|x| Param::Float(*x)).collect(),
                    qubits.iter().map(|q| Qubit(*q as u32)).collect(),
                    vec![],
                ))
            }),
            Param::Float(sequence.global_phase),
        )?
        .into())
    }

    fn num_basis_gates(&self, unitary: PyReadonlyArray2<Complex64>) -> PyResult<usize> {
        _num_basis_gates(self.basis_decomposer.b, self.basis_fidelity, unitary)
    }
}

/// Helper functions for two_qubit_decompose_up_to_diagonal
/// Convert a 4x4 unitary matrix into a unitary matrix with determinant 1
fn u4_to_su4(u4: ArrayView2<Complex64>) -> (Array2<Complex64>, f64) {
    let det_u = ndarray_to_faer(u4).determinant();
    let phase_factor = det_u.powf(-0.25).conj();
    let su4 = u4.mapv(|x| x / phase_factor);
    (su4, phase_factor.arg())
}

fn real_trace_transform(mat: ArrayView2<Complex64>) -> Array2<Complex64> {
    let a1 = -mat[[1, 3]] * mat[[2, 0]] + mat[[1, 2]] * mat[[2, 1]] + mat[[1, 1]] * mat[[2, 2]]
        - mat[[1, 0]] * mat[[2, 3]];
    let a2 = mat[[0, 3]] * mat[[3, 0]] - mat[[0, 2]] * mat[[3, 1]] - mat[[0, 1]] * mat[[3, 2]]
        + mat[[0, 0]] * mat[[3, 3]];
    let theta = 0.; // Arbitrary!
    let phi = 0.; // This is extra arbitrary!
    let psi = f64::atan2(a1.im + a2.im, a1.re - a2.re) - phi;
    let im = Complex64::new(0., -1.);
    let temp = [
        (theta * im).exp(),
        (phi * im).exp(),
        (psi * im).exp(),
        (-(theta + phi + psi) * im).exp(),
    ];
    Array2::from_diag(&arr1(&temp))
}

#[pyfunction]
#[pyo3(name = "two_qubit_decompose_up_to_diagonal")]
pub fn py_two_qubit_decompose_up_to_diagonal(
    py: Python,
    mat: PyReadonlyArray2<Complex64>,
) -> PyResult<(Py<PyAny>, PyCircuitData)> {
    let mat_arr: ArrayView2<Complex64> = mat.as_array();
    let (real_map, circ) = two_qubit_decompose_up_to_diagonal(mat_arr)?;
    Ok((real_map.into_pyarray(py).into_any().unbind(), circ.into()))
}

pub fn two_qubit_decompose_up_to_diagonal(
    mat: ArrayView2<Complex64>,
) -> PyResult<(Array2<Complex64>, CircuitData)> {
    let (su4, phase) = u4_to_su4(mat);
    let mut real_map = real_trace_transform(su4.view());
    let mapped_su4 = real_map.dot(&su4.view());
    let decomp = TwoQubitBasisDecomposer::new_inner(
        StandardGate::CX.into(),
        smallvec![],
        aview2(&CX_GATE),
        1.0,
        EulerBasis::U,
        None,
    )?;

    let circ_seq = decomp.call_inner(mapped_su4.view(), None, true, None)?;
    let circ = CircuitData::from_packed_operations(
        2,
        0,
        circ_seq
            .gates
            .into_iter()
            .map(|(gate, param_floats, qubit_index)| {
                let params: SmallVec<[Param; 3]> =
                    param_floats.into_iter().map(Param::Float).collect();
                let qubits = qubit_index.into_iter().map(|x| Qubit(x as u32)).collect();
                Ok((gate, params, qubits, vec![]))
            }),
        Param::Float(circ_seq.global_phase + phase),
    )?;
    real_map.mapv_inplace(|x| x.conj());
    Ok((real_map, circ))
}

/// Helper function for TwoQubitBasisDecomposer with rz, sx, cx gates
fn compute_unitary(sequence: &TwoQubitSequenceVec, global_phase: f64) -> Array2<Complex64> {
    let identity = aview2(&ONE_QUBIT_IDENTITY);
    let phase = c64(0., global_phase).exp();
    let mut matrix = Array2::from_diag(&arr1(&[phase, phase, phase, phase]));
    sequence
        .iter()
        .map(|inst| {
            // This only gets called by get_sx_vz_3cx_efficient_euler()
            // which only uses sx, x, rz, and cx gates for the circuit
            // sequence. If we get a different gate this is getting called
            // by something else and is invalid.
            let gate_matrix = inst
                .0
                .try_standard_gate()
                .expect("should be sx, x, rz, or cx")
                .matrix(&inst.1.iter().map(|x| Param::Float(*x)).collect::<Vec<_>>())
                .unwrap();
            (gate_matrix, &inst.2)
        })
        .for_each(|(op_matrix, q_list)| {
            let result = match q_list.as_slice() {
                [0] => Some(kron(&identity, &op_matrix)),
                [1] => Some(kron(&op_matrix, &identity)),
                [1, 0] => Some(two_qubit::change_basis(op_matrix.view())),
                [] => Some(Array2::eye(4)),
                _ => None,
            };
            matrix = match result {
                Some(result) => result.dot(&matrix),
                None => op_matrix.dot(&matrix),
            }
        });
    matrix
}
