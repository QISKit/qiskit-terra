// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use faer::{Mat, MatRef, Scale};
use hashbrown::HashMap;
use nalgebra::{Matrix4, U4};
use ndarray::prelude::*;
use num_complex::Complex64;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use smallvec::smallvec;
use thiserror::Error;

use crate::euler_one_qubit_decomposer::{
    EulerBasis, EulerBasisSet, unitary_to_gate_sequence_inner,
};
use crate::linalg::{
    LinAlgError, VERIFY_TOL, block_matrix_faer, closest_unitary_faer, eigendecomposition_faer,
    faer_to_ndarray, from_diagonal_faer, is_zero_matrix_faer, nalgebra_array_view, ndarray_to_faer,
    svd_decomposition_faer, verify_unitary_faer,
};
use crate::matrix::two_qubit;
use crate::two_qubit_decompose::{TwoQubitBasisDecomposer, two_qubit_decompose_up_to_diagonal};
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError, PyCircuitData};
use qiskit_circuit::interner::Interned;
use qiskit_circuit::operations::{ArrayType, OperationRef, Param, StandardGate, UnitaryGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{BlocksMode, Qubit, VarsMode};

const EPS: f64 = 1e-10;
const MINIMUM_TOL: f64 = 1e-12;

/// Errors that might occur during QSD synthesis algorithm
#[derive(Error, Debug)]
pub enum QSDError {
    // wraps LinAlgError, produced by linear algebra packages
    #[error(transparent)]
    ErrorFromLinAlg(#[from] LinAlgError),

    // wraps CircuitDataError, e.g. produced by demultiplex
    #[error(transparent)]
    ErrorFromCircuitData(#[from] CircuitDataError),

    // wraps PyErr, e.g. produced by 2q decomposer
    #[error(transparent)]
    ErrorFromPython(#[from] PyErr),

    // unknown one-qubit decomposer basis name
    #[error("unknown basis name {0}")]
    UnknownBasis(String),
}

impl From<QSDError> for PyErr {
    fn from(error: QSDError) -> Self {
        match error {
            QSDError::ErrorFromLinAlg(err) => err.into(),

            QSDError::ErrorFromCircuitData(err) => err.into(),

            QSDError::ErrorFromPython(err) => err,

            QSDError::UnknownBasis(_) => pyo3::exceptions::PyValueError::new_err(error.to_string()),
        }
    }
}

// when performing demultiplaxing, this enum is used to specify the actions that needs to be done
enum VWType {
    All,
    OnlyV,
    OnlyW,
}

/// Decomposes a unitary matrix into one and two qubit gates using Quantum Shannon Decomposition,
/// based on the Block ZXZ-Decomposition.
///
/// # Arguments
/// * `array`: unitary matrix to decompose
/// * `opt_a1`: whether to try optimization A.1 (if `None`, decide automatically)
/// * `opt_a2`: whether to try optimization A.2 (if `None`, decide automatically)
/// * `two_qubit_decomposer`: Optional alternative two qubit decomposer, if not specified a decomposer using CX and U is used.
/// * `one_qubit_decomposer`: Optional alternative one qubit euler basis to use for single qubit unitary decompositions. If not specified U is used.
///
/// # Returns
///
/// Decomposed quantum circuit.
pub fn quantum_shannon_decomposition(
    array: ArrayView2<Complex64>,
    opt_a1: Option<bool>,
    opt_a2: Option<bool>,
    one_qubit_decomposer_basis_set: Option<&EulerBasisSet>,
    two_qubit_decomposer: Option<&TwoQubitBasisDecomposer>,
) -> Result<CircuitData, QSDError> {
    let mat = ndarray_to_faer(array);
    let dim = mat.shape().0;
    let num_qubits = dim.ilog2() as usize;
    let mut default_1q_basis = EulerBasisSet::new();
    default_1q_basis.add_basis(EulerBasis::U);
    let default_2q_decomposer = TwoQubitBasisDecomposer::new_inner(
        StandardGate::CX.into(),
        smallvec![],
        aview2(&qiskit_circuit::gate_matrix::CX_GATE),
        1.0,
        EulerBasis::U,
        None,
    )?;
    let one_qubit_decomposer = one_qubit_decomposer_basis_set.unwrap_or(&default_1q_basis);
    let two_qubit_decomposer = two_qubit_decomposer.unwrap_or(&default_2q_decomposer);

    if (Mat::<Complex64>::identity(dim, dim).as_ref() - mat).norm_max() < MINIMUM_TOL {
        let out_qubits = (0..num_qubits)
            .map(|_| ShareableQubit::new_anonymous())
            .collect::<Vec<_>>();
        return Ok(CircuitData::new(Some(out_qubits), None, Param::Float(0.))?);
    }
    qsd_inner(
        mat,
        opt_a1,
        opt_a2,
        two_qubit_decomposer,
        one_qubit_decomposer,
        0,
    )
}

fn qsd_inner(
    mat: MatRef<Complex64>,
    opt_a1: Option<bool>,
    opt_a2: Option<bool>,
    two_qubit_decomposer: &TwoQubitBasisDecomposer,
    one_qubit_decomposer: &EulerBasisSet,
    depth: usize,
) -> Result<CircuitData, QSDError> {
    let dim = mat.shape().0;
    let num_qubits = dim.ilog2() as usize;
    let opt_a1_val = opt_a1.unwrap_or(true);
    if dim == 1 {
        // A 1x1 "unitary" is just a global phase acting on zero qubits.
        let global_phase = mat[(0, 0)].arg();
        return Ok(CircuitData::new(None, None, Param::Float(global_phase))?);
    }
    if dim == 2 {
        let array = faer_to_ndarray(mat);
        let sequence =
            unitary_to_gate_sequence_inner(array.view(), one_qubit_decomposer, 0, None, true, None);

        return match sequence {
            Some(seq) => Ok(CircuitData::from_standard_gates(
                1,
                seq.gates.into_iter().map(|(gate, params)| {
                    (
                        gate,
                        params.into_iter().map(Param::Float).collect(),
                        smallvec![Qubit(0)],
                    )
                }),
                Param::Float(seq.global_phase),
            )?),
            None => {
                let out_qubits = (0..num_qubits)
                    .map(|_| ShareableQubit::new_anonymous())
                    .collect::<Vec<_>>();
                Ok(CircuitData::new(Some(out_qubits), None, Param::Float(0.))?)
            }
        };
    } else if dim == 4 {
        if opt_a2 == Some(true) && depth > 0 {
            let out_qubits = (0..num_qubits)
                .map(|_| ShareableQubit::new_anonymous())
                .collect::<Vec<_>>();
            let mut out = CircuitData::new(Some(out_qubits), None, Param::Float(0.))?;
            let two_q_mat: Matrix4<Complex64> = Matrix4::from_fn(|i, j| mat[(i, j)]);
            let packed_inst = PackedInstruction {
                op: PackedOperation::from_unitary(Box::new(UnitaryGate {
                    array: ArrayType::TwoQ(two_q_mat),
                })),
                qubits: out.add_qargs(&[Qubit(0), Qubit(1)]),
                clbits: Default::default(),
                params: None,
                label: Some(Box::new("qsd2q".to_string())),
                #[cfg(feature = "cache_pygates")]
                py_op: OnceLock::new(),
            };
            out.push(packed_inst)?;
            return Ok(out);
        }
        let array = faer_to_ndarray(mat);
        let sequence = two_qubit_decomposer
            .call_inner(array.view(), None, false, None)
            .unwrap_or_else(|_| {
                two_qubit_decomposer
                    .call_inner(array.view(), None, false, None)
                    .unwrap()
            });
        let global_phase = sequence.global_phase();
        return Ok(CircuitData::from_packed_operations(
            num_qubits as u32,
            0,
            sequence
                .into_gates()
                .into_iter()
                .map(|(op, params, qubits)| {
                    Ok((
                        op,
                        params.into_iter().map(Param::Float).collect(),
                        qubits.into_iter().map(|q| Qubit(q as u32)).collect(),
                        vec![],
                    ))
                }),
            Param::Float(global_phase),
        )?);
    }
    // Check whether the matrix is equivalent to a block diagonal w.r.t ctrl_index
    if opt_a2 != Some(true) {
        for ctrl_index in 0..num_qubits {
            let [um00, um11, um01, um10] = extract_multiplex_blocks(mat, ctrl_index);

            if is_zero_matrix_faer(um01.as_ref(), None) && is_zero_matrix_faer(um10.as_ref(), None)
            {
                return Ok(demultiplex(
                    um00.as_ref(),
                    um11.as_ref(),
                    opt_a1_val,
                    false, // opt_a2
                    depth,
                    Some(num_qubits - 1 - ctrl_index),
                    VWType::All,
                    two_qubit_decomposer,
                    one_qubit_decomposer,
                )?
                .0);
            }
        }
    }
    let opt_a2_val = opt_a2.unwrap_or(true);
    // a rough bound on the number of gates in the circuit is as follows:
    // the number of CX gates without optimizations is N = 9/16*4^n - 3/2*2^n
    // the number of 1-qubit unitary gates is bounded by 2N
    // depending on the one-qubit decomposer, it means up to 6N gates for the 1-qubit unitaries
    // this leads to a bound of 7N = 63/16*4^n - 21/2*2^n = (63x^2-168x)/16 for x=2^n
    let x: usize = 1 << num_qubits;
    let numerator = 63 * x * x - 168 * x;
    let gates_bound = numerator.div_ceil(16);

    let mut out = CircuitData::with_capacity(num_qubits as u32, 0, gates_bound, Param::Float(0.))?;
    // perform block ZXZ decomposition from [2]
    let zxz_result = block_zxz_decomp(mat.as_ref())?;
    debug_assert!(zxz_decomp_verify(mat, &zxz_result,));

    let ZXZResult { a1, a2, b, c } = zxz_result;
    let iden = Mat::<Complex64>::identity(dim / 2, dim / 2);
    let (left_circuit, vmat_c, _) = demultiplex(
        iden.as_ref(),
        c.as_ref(),
        opt_a1_val,
        opt_a2_val,
        depth,
        None,
        VWType::OnlyW,
        two_qubit_decomposer,
        one_qubit_decomposer,
    )?;
    let (right_circuit, _, wmat_a) = demultiplex(
        a1.as_ref(),
        a2.as_ref(),
        opt_a1_val,
        opt_a2_val,
        depth,
        None,
        VWType::OnlyV,
        two_qubit_decomposer,
        one_qubit_decomposer,
    )?;

    // middle circ

    // wmatA and vmatC are combined into B1 and B2
    let b1 = &wmat_a * &vmat_c;
    let b2 = if opt_a1_val {
        // zmat is needed in order to reduce two cz gates, and combine them into the B2 matrix
        let mut zmat = Mat::<Complex64>::zeros(dim / 2, dim / 2);
        for i in 0..dim / 2 {
            zmat[(i, i)] = if i < dim / 4 {
                Complex64::ONE
            } else {
                -Complex64::ONE
            };
        }
        &zmat * &wmat_a * &b * &vmat_c * &zmat
    } else {
        &wmat_a * &b * &vmat_c
    };
    let middle_circ = demultiplex(
        b1.as_ref(),
        b2.as_ref(),
        opt_a1_val,
        opt_a2_val,
        depth,
        None,
        VWType::All,
        two_qubit_decomposer,
        one_qubit_decomposer,
    )?
    .0;

    // the output circuit of the block ZXZ decomposition from [2]
    let qr = (0..num_qubits).map(Qubit::new).collect::<Vec<_>>();
    append(&mut out, left_circuit, &qr)?;
    out.push_standard_gate(StandardGate::H, &[], &[Qubit((num_qubits - 1) as u32)])?;
    append(&mut out, middle_circ, &qr)?;
    out.push_standard_gate(StandardGate::H, &[], &[Qubit((num_qubits - 1) as u32)])?;
    append(&mut out, right_circuit, &qr)?;
    if opt_a2_val && depth == 0 && dim > 4 {
        Ok(apply_a2(&out, two_qubit_decomposer)?)
    } else {
        Ok(out)
    }
}

/// Result for the block-ZXZ decomposition of a matrix `A`.
///
/// See [2] equation (5) for details.
struct ZXZResult {
    pub a1: Mat<Complex64>,
    pub a2: Mat<Complex64>,
    pub b: Mat<Complex64>,
    pub c: Mat<Complex64>,
}

/// Run the Block-ZXZ decomposition, by Krol and Al-Ars [2].
fn block_zxz_decomp(mat: MatRef<Complex64>) -> Result<ZXZResult, QSDError> {
    debug_assert!(verify_unitary_faer(mat));

    let i = Complex64::new(0.0, 1.0);
    let n = mat.shape().0 / 2;
    let x = mat.submatrix(0, 0, n, n);
    let y = mat.submatrix(0, n, n, n);
    let u21 = mat.submatrix(n, 0, n, n);
    let u22 = mat.submatrix(n, n, n, n);

    let svdx = svd_decomposition_faer(x)?;
    let sx = svdx.u.as_ref() * svdx.s * svdx.u.adjoint();
    let ux = svdx.u * svdx.v.adjoint();

    let svdy = svd_decomposition_faer(y)?;
    let sy = svdy.u.as_ref() * svdy.s * svdy.u.adjoint();
    let uy = svdy.u * svdy.v.adjoint();

    let c = ((uy.adjoint() * &ux) * Scale(i)).adjoint().to_owned();
    let a1 = (sx + sy * Scale(i)) * &ux;
    let a2 = u21 + (u22 * (uy.adjoint() * ux) * Scale(i));
    let b = (a1.adjoint() * x) * Scale(Complex64::from(2.0)) - Mat::<Complex64>::identity(n, n);
    Ok(ZXZResult { a1, a2, b, c })
}

/// Verify ZXZ decomposition gives the same unitary
fn zxz_decomp_verify(mat: MatRef<Complex64>, zxz_result: &ZXZResult) -> bool {
    let ZXZResult { a1, a2, b, c } = zxz_result;

    let n = mat.shape().0 / 2;
    let zero = Mat::<Complex64>::zeros(n, n);
    let iden = Mat::<Complex64>::identity(n, n);

    let a_block = block_matrix_faer(a1.as_ref(), zero.as_ref(), zero.as_ref(), a2.as_ref());

    let b1 = &iden + b.as_ref();
    let b2 = &iden - b.as_ref();
    let b_block = block_matrix_faer(b1.as_ref(), b2.as_ref(), b2.as_ref(), b1.as_ref());

    let c_block = block_matrix_faer(iden.as_ref(), zero.as_ref(), zero.as_ref(), c.as_ref());

    let mat_check = &a_block * &b_block * &c_block * Scale(Complex64::from(0.5));

    (mat - mat_check).norm_max() < VERIFY_TOL
}

///  Decompose a generic multiplexer.
///
///        ────□────
///         ┌──┴──┐
///       /─┤     ├─
///         └─────┘
///
/// represented by the block diagonal matrix
///
///         ┏         ┓
///         ┃ um0     ┃
///         ┃     um1 ┃
///         ┗         ┛
///
/// to
///            ┌───┐
///     ───────┤ Rz├──────
///       ┌───┐└─┬─┘┌───┐
///     /─┤ w ├──□──┤ v ├─
///       └───┘     └───┘
///
/// where v and w are general unitaries determined from decomposition.
///
/// # Note
///
/// When the original unitary is controlled then the default value of ``opt_a2 = False``
/// as we start with the demultiplexing step that does not work with the optimization A.2 of [1, 2].
#[allow(clippy::too_many_arguments)]
fn demultiplex(
    um0: MatRef<Complex64>,
    um1: MatRef<Complex64>,
    opt_a1: bool,
    opt_a2: bool,
    depth: usize,
    _ctrl_index: Option<usize>,
    vw_type: VWType,
    two_qubit_decomposer: &TwoQubitBasisDecomposer,
    one_qubit_decomposer: &EulerBasisSet,
) -> Result<(CircuitData, Mat<Complex64>, Mat<Complex64>), QSDError> {
    let um0 = closest_unitary_faer(um0.as_ref())?;
    let um1 = closest_unitary_faer(um1.as_ref())?;

    let dim = um0.shape().0 + um1.shape().0;
    let num_qubits = dim.ilog2() as usize;
    let _ctrl_index = _ctrl_index.unwrap_or_else(|| num_qubits - 1);
    let layout: Vec<Qubit> = (0.._ctrl_index)
        .chain(_ctrl_index + 1..num_qubits)
        .chain([_ctrl_index])
        .map(Qubit::new)
        .collect();
    let um0um1 = um0.as_ref() * um1.adjoint();
    let (eigvals, vmat): (Vec<Complex64>, Mat<Complex64>) =
        eigendecomposition_faer(um0um1.as_ref())?;
    let d_values: Vec<Complex64> = eigvals.iter().map(|x| x.sqrt()).collect();
    let d_mat: Mat<Complex64> = from_diagonal_faer(&d_values);
    let wmat = d_mat.as_ref() * vmat.adjoint() * um1.as_ref();
    debug_assert!(demultiplex_verify(
        um0.as_ref(),
        um1.as_ref(),
        vmat.as_ref(),
        wmat.as_ref(),
        d_mat.as_ref()
    ));

    let out_qubits = (0..num_qubits)
        .map(|_| ShareableQubit::new_anonymous())
        .collect::<Vec<_>>();
    let mut out = CircuitData::new(Some(out_qubits), None, Param::Float(0.))?;

    // left gate. In this case we decompose wmat.
    // Otherwise, it is combined with the B matrix.
    match vw_type {
        VWType::OnlyW | VWType::All => {
            let left_circuit = qsd_inner(
                wmat.as_ref(),
                Some(opt_a1),
                Some(opt_a2),
                two_qubit_decomposer,
                one_qubit_decomposer,
                depth + 1,
            )?;
            append(&mut out, left_circuit, &layout[..num_qubits - 1])?;
        }
        VWType::OnlyV => (),
    }

    // multiplexed Rz gate
    // If opt_a1 = ``True``, then we reduce 2 ``cx`` gates per call.
    let mut angles = d_values
        .iter()
        .map(|x| 2. * x.conj().arg())
        .collect::<Vec<_>>();
    let ucrz = match (&vw_type, opt_a1) {
        (VWType::OnlyW, true) => get_ucrz(num_qubits, &mut angles, false)?,
        (VWType::OnlyV, true) => get_ucrz(num_qubits, &mut angles, false)?.reverse()?,
        _ => get_ucrz(num_qubits, &mut angles, true)?,
    };
    append(
        &mut out,
        ucrz,
        &[
            std::slice::from_ref(layout.last().unwrap()),
            &layout[..num_qubits - 1],
        ]
        .concat(),
    )?;
    // right gate. In this case we decompose vmat.
    // Otherwise, it is combined with the B matrix.
    match vw_type {
        VWType::OnlyV | VWType::All => {
            let right_circuit = qsd_inner(
                vmat.as_ref(),
                Some(opt_a1),
                Some(opt_a2),
                two_qubit_decomposer,
                one_qubit_decomposer,
                depth + 1,
            )?;
            append(&mut out, right_circuit, &layout[..num_qubits - 1])?;
        }
        VWType::OnlyW => (),
    }
    Ok((out, vmat, wmat))
}

fn demultiplex_verify(
    um0: MatRef<Complex64>,
    um1: MatRef<Complex64>,
    vmat: MatRef<Complex64>,
    wmat: MatRef<Complex64>,
    dmat: MatRef<Complex64>,
) -> bool {
    let n = um0.nrows();
    let zero = Mat::<Complex64>::zeros(n, n);

    let u_block = block_matrix_faer(um0, zero.as_ref(), zero.as_ref(), um1);
    let v_block = block_matrix_faer(vmat, zero.as_ref(), zero.as_ref(), vmat);
    let w_block = block_matrix_faer(wmat, zero.as_ref(), zero.as_ref(), wmat);
    let d_inv = dmat.adjoint().to_owned();
    let d_block = block_matrix_faer(dmat, zero.as_ref(), zero.as_ref(), d_inv.as_ref());
    let u_check = &v_block * &d_block * &w_block;

    (u_block.as_ref() - u_check.as_ref()).norm_max() < VERIFY_TOL
}

/// This function synthesizes UCRZ without the final CX gate,
/// unless _vw_type = ``all``.
fn get_ucrz(
    num_qubits: usize,
    angles: &mut [f64],
    vw_type_all: bool,
) -> Result<CircuitData, CircuitDataError> {
    let out_qubits = (0..num_qubits)
        .map(|_| ShareableQubit::new_anonymous())
        .collect::<Vec<_>>();
    let mut out = CircuitData::new(Some(out_qubits), None, Param::Float(0.))?;
    let q_target = Qubit(0);
    let q_controls: Vec<Qubit> = (1..num_qubits).map(|i| Qubit(i as u32)).collect();
    decompose_uc_rotations(angles, 0, angles.len(), false);
    for (i, angle) in angles.iter().enumerate() {
        if angle.abs() > EPS {
            let _ = out.push_standard_gate(StandardGate::RZ, &[Param::Float(*angle)], &[q_target]);
        }
        if i != angles.len() - 1 {
            let q_ctrl_index = (i + 1).trailing_zeros();
            let _ = out.push_standard_gate(
                StandardGate::CX,
                &[],
                &[q_controls[q_ctrl_index as usize], q_target],
            );
        } else if vw_type_all {
            let q_ctrl_index = num_qubits - 2;
            let _ = out.push_standard_gate(
                StandardGate::CX,
                &[],
                &[q_controls[q_ctrl_index], q_target],
            );
        };
    }
    Ok(out)
}

/// Calculates rotation angles for a uniformly controlled R_t gate with a C-NOT gate at
/// the end of the circuit. The rotation angles of the gate R_t are stored in
/// angles[start_index:end_index]. If reversed_dec == True, it decomposes the gate such that
/// there is a C-NOT gate at the start of the circuit (in fact, the circuit topology for
/// the reversed decomposition is the reversed one of the original decomposition)
fn decompose_uc_rotations(
    angles: &mut [f64],
    start_index: usize,
    end_index: usize,
    reversed_decomposition: bool,
) {
    let interval_len_half = (end_index - start_index) / 2;
    for i in start_index..start_index + interval_len_half {
        if !reversed_decomposition {
            let new_angles = update_angle(angles[i], angles[i + interval_len_half]);
            angles[i] = new_angles[0];
            angles[i + interval_len_half] = new_angles[1];
        } else {
            let new_angles = update_angle(angles[i], angles[i + interval_len_half]);
            angles[i + interval_len_half] = new_angles[0];
            angles[i] = new_angles[1];
        }
    }
    if interval_len_half > 1 {
        decompose_uc_rotations(angles, start_index, start_index + interval_len_half, false);
        decompose_uc_rotations(angles, start_index + interval_len_half, end_index, true);
    }
}

fn update_angle(angle_1: f64, angle_2: f64) -> [f64; 2] {
    [(angle_1 + angle_2) / 2., (angle_1 - angle_2) / 2.]
}

fn append(
    circ: &mut CircuitData,
    new: CircuitData,
    qubit_map: &[Qubit],
) -> Result<(), CircuitDataError> {
    let new_qubits_map = circ.merge_qargs(new.qargs_interner(), |x| Some(qubit_map[x.index()]));
    circ.add_global_phase(new.global_phase())?;
    for inst in new.into_data_iter() {
        let out_inst = PackedInstruction {
            op: inst.op,
            params: inst.params,
            qubits: new_qubits_map[inst.qubits],
            clbits: Default::default(),
            label: inst.label,
            #[cfg(feature = "cache_pygates")]
            py_op: inst.py_op,
        };
        circ.push(out_inst)?;
    }
    Ok(())
}

/// numpy's move_axis has the effect of pushing back the axis
/// before the location we move into. there's no native ndarray equivalent
fn move_axis<X>(array: ArrayD<X>, src: usize, dest: usize) -> ArrayD<X>
where
    X: Clone,
{
    let ndim = array.ndim();
    let mut axes: Vec<usize> = (0..ndim).collect();
    let axis = axes.remove(src);
    axes.insert(dest, axis);
    array.permuted_axes(axes)
}

/// A block diagonal gate is represented as:
/// [ um00 | um01 ]
/// [ ---- | ---- ]
/// [ um10 | um11 ]
fn extract_multiplex_blocks(umat: MatRef<Complex64>, k: usize) -> [Mat<Complex64>; 4] {
    let dim = umat.shape().0;
    let num_qubits = dim.ilog2() as usize;
    let half_dim = dim / 2;
    let umat_array: Array2<Complex64> = Array2::from_shape_fn(umat.shape(), |(i, j)| umat[(i, j)]);
    let mut utensor = umat_array
        .to_shape(vec![2; 2 * num_qubits])
        .unwrap()
        .into_owned();
    if k != 0 {
        utensor = move_axis(utensor, k, 0);
        utensor = move_axis(utensor, k + num_qubits, num_qubits);
    }
    let ud4 = utensor.to_shape([2, half_dim, 2, half_dim]).unwrap();
    let um00 = ud4.slice(s![0, .., 0, ..]);
    let um11 = ud4.slice(s![1, .., 1, ..]);
    let um01 = ud4.slice(s![0, .., 1, ..]);
    let um10 = ud4.slice(s![1, .., 0, ..]);
    [
        Mat::from_fn(um00.shape()[0], um00.shape()[1], |i, j| um00[[i, j]]),
        Mat::from_fn(um11.shape()[0], um11.shape()[1], |i, j| um11[[i, j]]),
        Mat::from_fn(um01.shape()[0], um01.shape()[1], |i, j| um01[[i, j]]),
        Mat::from_fn(um10.shape()[0], um10.shape()[1], |i, j| um10[[i, j]]),
    ]
}

/// The optimization A.2 from [1, 2]. This decomposes two qubit unitaries into a
/// diagonal gate and a two cx unitary and reduces overall ``cx`` count by
/// 4^(n-2) - 1. This optimization should not be done if the original unitary is controlled.
fn apply_a2(
    circ: &CircuitData,
    two_qubit_decomposer: &TwoQubitBasisDecomposer,
) -> Result<CircuitData, CircuitDataError> {
    let ind2q: Vec<usize> = circ
        .data()
        .iter()
        .enumerate()
        .filter_map(|(idx, inst)| {
            if matches!(inst.op.view(), OperationRef::Unitary(_))
                && let Some(ref label) = inst.label
                && label.as_str() == "qsd2q"
            {
                return Some(idx);
            }
            None
        })
        .collect();
    if ind2q.is_empty() {
        return Ok(circ.clone());
    } else if ind2q.len() == 1 {
        let mut circ = circ.clone();
        circ.invalidate_label(ind2q[0]);
    }
    let mut diagonal_rollover: HashMap<usize, CircuitData> = HashMap::with_capacity(ind2q.len());
    let mut new_matrices: HashMap<usize, Array2<Complex64>> = ind2q
        .iter()
        .map(|idx| {
            let OperationRef::Unitary(unitary) = circ.data()[*idx].op.view() else {
                unreachable!("diagonal unitary is not a unitary gate");
            };
            (*idx, unitary.matrix().unwrap())
        })
        .collect();
    for ind in ind2q.windows(2) {
        let mat1 = match diagonal_rollover.get(&ind[0]) {
            Some(circ) => {
                let mat: Matrix4<Complex64> = two_qubit::instructions_to_matrix(
                    circ.data().iter(),
                    [Qubit(0), Qubit(1)],
                    circ.qargs_interner(),
                )?;
                nalgebra_array_view::<Complex64, U4, U4>(mat.as_view()).to_owned()
            }
            None => new_matrices[&ind[0]].to_owned(),
        };
        let mat2 = match diagonal_rollover.get(&ind[1]) {
            Some(circ) => nalgebra_array_view::<Complex64, U4, U4>(
                two_qubit::instructions_to_matrix(
                    circ.data().iter(),
                    [Qubit(0), Qubit(1)],
                    circ.qargs_interner(),
                )?
                .as_view(),
            )
            .to_owned(),
            None => new_matrices[&ind[1]].to_owned(),
        };
        let (diagonal_mat, qc2cx) = two_qubit_decompose_up_to_diagonal(mat1.view()).unwrap();
        diagonal_rollover.insert(ind[0], qc2cx);
        let new_mat2 = mat2.dot(&diagonal_mat);
        new_matrices.insert(ind[1], new_mat2);
    }
    let last_idx = ind2q.last().unwrap();
    let qc3_seq = two_qubit_decomposer
        .call_inner(new_matrices[last_idx].view(), None, true, None)
        .unwrap();
    let phase = Param::Float(qc3_seq.global_phase());
    let qc3 = CircuitData::from_packed_operations(
        2,
        0,
        qc3_seq
            .into_gates()
            .into_iter()
            .map(|(gate, params, qubits)| {
                Ok((
                    gate,
                    params.into_iter().map(Param::Float).collect(),
                    qubits.into_iter().map(|q| Qubit(q as u32)).collect(),
                    vec![],
                ))
            }),
        phase,
    )?;
    diagonal_rollover.insert(*last_idx, qc3);
    let mut out_circ = CircuitData::clone_empty_like(
        circ,
        Some(
            circ.data().len()
                + diagonal_rollover
                    .values()
                    .map(|x| x.data().len())
                    .sum::<usize>()
                - ind2q.len(),
        ),
        VarsMode::Alike,
        BlocksMode::Drop,
    )?;
    for (idx, inst) in circ.data().iter().enumerate() {
        if let Some(new_circ) = diagonal_rollover.get(&idx) {
            let block_index_map = circ.get_qargs(circ.data()[idx].qubits);

            // Build interned key map for possible combinations of 2q unitary qargs in out circuit
            let qarg_vals = [
                out_circ.add_qargs(&[block_index_map[0]]),
                out_circ.add_qargs(&[block_index_map[1]]),
                out_circ.add_qargs(&[block_index_map[0], block_index_map[1]]),
                out_circ.add_qargs(&[block_index_map[1], block_index_map[0]]),
            ];
            // Build lookup table for possible interned qubits in custom decomposition
            // in 2q new circuit
            let interned_default = out_circ.qargs_interner().get_default();
            let interned_map: [&[Qubit]; 4] = [
                &[Qubit(0)],
                &[Qubit(1)],
                &[Qubit(0), Qubit(1)],
                &[Qubit(1), Qubit(0)],
            ];
            let interned_map = interned_map.map(|qubits| {
                new_circ
                    .qargs_interner()
                    .try_key(qubits)
                    .unwrap_or(interned_default)
            });
            // Map new circuit interned qubits -> out_circuit interned qubits
            let qarg_lookup = |qargs: Interned<[Qubit]>| {
                qarg_vals[interned_map
                    .iter()
                    .position(|key| qargs == *key)
                    .expect("not a 1q or 2q gate in the qargs block")]
            };
            for inst in new_circ.data() {
                let out_inst = PackedInstruction {
                    op: inst.op.clone(),
                    qubits: qarg_lookup(inst.qubits),
                    clbits: inst.clbits,
                    params: inst.params.clone(),
                    label: inst.label.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                };
                out_circ.push(out_inst)?;
            }
            out_circ.add_global_phase(new_circ.global_phase())?;
        } else {
            out_circ.push(inst.clone())?;
        }
    }
    Ok(out_circ)
}

/// Decomposes a unitary matrix into one and two qubit gates using Quantum Shannon Decomposition,
/// based on the Block ZXZ-Decomposition.
/// This decomposition is described in Krol and Al-Ars [2] and improves the method of
/// Shende et al. [1].
///
/// .. code-block:: text
///
///       ┌───┐              ┌───┐     ┌───┐
///      ─┤   ├─      ────□──┤ H ├──□──┤ H ├──□──
///       │   │    ≃    ┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐
///     /─┤   ├─      ──┤ C ├─────┤ B ├─────┤ A ├
///       └───┘         └───┘     └───┘     └───┘
///
/// The number of :class:`.CXGate`\ s generated with the decomposition without optimizations is
/// the same as the unoptimized method in [1]:
///
/// .. math::
///
///     \frac{9}{16} 4^n - \frac{3}{2} 2^n
///
/// If ``opt_a1 = True``, the CX count is reduced, improving [1], by:
///
/// .. math::
///
///     \frac{2}{3} (4^{n - 2} - 1).
///
/// Saving two :class:`.CXGate`\ s instead of one in each step of the recursion.
///
/// If ``opt_a2 = True``, the CX count is reduced, as in [1], by:
///
/// .. math::
///
///     4^{n-2} - 1.
///
/// Hence, the number of :class:`.CXGate`\ s generated with the decomposition with optimizations is
///
/// .. math::
///
///     \frac{22}{48} 4^n - \frac{3}{2} 2^n + \frac{5}{3}.
///
///
/// Args:
///     mat: unitary matrix to decompose
///     opt_a1: whether to try optimization A.1 from [1, 2].
///         This should eliminate 2 ``cx`` per call.
///     opt_a2: whether to try optimization A.2 from [1, 2].
///         This decomposes two qubit unitaries into a diagonal gate and
///         a two ``cx`` unitary and reduces overall ``cx`` count by :math:`4^{n-2} - 1`.
///         This optimization should not be done if the original unitary is controlled.
///
/// Returns:
///     QuantumCircuit: Decomposed quantum circuit.
///
/// References:
///
/// [1] Shende, Bullock, Markov, *Synthesis of Quantum Logic Circuits*,
///        `arXiv:0406176 [quant-ph] <https://arxiv.org/abs/quant-ph/0406176>`_
/// [2] Krol, Al-Ars, *Beyond Quantum Shannon: Circuit Construction for General
///        n-Qubit Gates Based on Block ZXZ-Decomposition*,
///        `arXiv:2403.13692 <https://arxiv.org/abs/2403.13692>`_
#[pyfunction]
pub fn qs_decomposition(
    mat: PyReadonlyArray2<Complex64>,
    opt_a1: Option<bool>,
    opt_a2: Option<bool>,
    one_qubit_decomposer_basis_string: Option<String>,
    two_qubit_decomposer: Option<&TwoQubitBasisDecomposer>,
) -> PyResult<PyCircuitData> {
    let array: ArrayView2<Complex64> = mat.as_array();
    let mut one_qubit_decomposer_basis_set = EulerBasisSet::new();
    let one_qubit_decomposer = if let Some(basis_string) = one_qubit_decomposer_basis_string {
        let basis = basis_string
            .parse::<EulerBasis>()
            .map_err(|_| QSDError::UnknownBasis(basis_string))?;
        one_qubit_decomposer_basis_set.add_basis(basis);
        Some(&one_qubit_decomposer_basis_set)
    } else {
        None
    };
    let res = quantum_shannon_decomposition(
        array,
        opt_a1,
        opt_a2,
        one_qubit_decomposer,
        two_qubit_decomposer,
    )?;
    Ok(res.into())
}

pub fn qsd_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(qs_decomposition, m)?)?;
    Ok(())
}
