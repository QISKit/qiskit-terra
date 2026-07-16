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

use pyo3::prelude::*;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports::PAULI_EVOLUTION_GATE;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PauliBased, PauliProductMeasurement, PauliProductRotation,
    PyInstruction, PyOpKind, StandardGate, StandardInstruction, multiply_param, radd_param,
};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{BlocksMode, Qubit, VarsMode};

use super::common::{
    MINIMUM_TOL, average_gate_fidelity_below_tol, is_angle_close_to_multiple_of_pi_k,
};
use crate::TranspilerError;
use num_complex::Complex64;
use qiskit_quantum_info::clifford::{Clifford, Pauli1q};
use qiskit_quantum_info::sparse_observable::{BitTerm, SparseObservable};

use smallvec::smallvec;
use std::f64::consts::{FRAC_PI_4, FRAC_PI_8, PI};

// List of gate/instruction names supported by the pass: the pass raises an error if the circuit
// contains instruction with names outside of this list.
static SUPPORTED_INSTRUCTION_NAMES: [&str; 26] = [
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "sx",
    "sxdg",
    "cx",
    "cz",
    "cy",
    "swap",
    "iswap",
    "ecr",
    "dcx",
    "t",
    "tdg",
    "rz",
    "rx",
    "ry",
    "p",
    "u1",
    "measure",
    "pauli_product_rotation",
    "pauli_product_measurement",
];

// List of instruction names which are modified by the pass: the pass is skipped if the circuit
// contains no instructions with names in this list.
static HANDLED_INSTRUCTION_NAMES: [&str; 10] = [
    "t",
    "tdg",
    "rz",
    "rx",
    "ry",
    "p",
    "u1",
    "measure",
    "pauli_product_rotation",
    "pauli_product_measurement",
];

#[pyfunction]
#[pyo3(signature = (dag, fix_clifford=true, insert_barrier=false, use_ppr=false, approximation_degree=1.0))]
pub fn run_litinski_transformation(
    dag: &DAGCircuit,
    fix_clifford: bool,
    insert_barrier: bool,
    use_ppr: bool,
    approximation_degree: f64,
) -> PyResult<Option<DAGCircuit>> {
    let op_counts = dag.get_op_counts();
    let tol = MINIMUM_TOL.max(1.0 - approximation_degree);

    // Skip the pass if there are no rotation or measurement gates, including PPRs and PPMs.
    if op_counts
        .keys()
        .all(|k| !HANDLED_INSTRUCTION_NAMES.contains(&k.as_str()))
    {
        return Ok(None);
    }

    // Skip the pass if there are unsupported gates.
    if !op_counts
        .keys()
        .all(|k| SUPPORTED_INSTRUCTION_NAMES.contains(&k.as_str()))
    {
        let unsupported: Vec<_> = op_counts
            .keys()
            .filter(|k| !SUPPORTED_INSTRUCTION_NAMES.contains(&k.as_str()))
            .collect();

        return Err(TranspilerError::new_err(format!(
            "Unable to run Litinski transformation as the circuit contains instructions not supported by the pass: {:?}",
            unsupported
        )));
    }
    // note that this count may not be accurate since non-clifford gates with pi/2 angles
    // are treated as cliffords by this pass
    let non_clifford_handled_count: usize = HANDLED_INSTRUCTION_NAMES
        .iter()
        .filter_map(|name| op_counts.get(*name))
        .sum();
    let clifford_count = dag.size(false)? - non_clifford_handled_count;

    let new_dag = dag.copy_empty_like_with_same_capacity(VarsMode::Alike, BlocksMode::Keep);
    let mut new_dag = new_dag.into_builder();

    let num_qubits = dag.num_qubits();
    let mut clifford = Clifford::identity(num_qubits);

    let mut qargs = Vec::new();

    // Keep track of the update to the global phase (produced when converting T/Tdg gates
    // to RZ-rotations).
    let mut global_phase_update = Param::Float(0.);

    // Keep track of the clifford operations in the circuit.
    let mut clifford_ops: Vec<&PackedInstruction> = Vec::with_capacity(clifford_count);
    // Apply the Litinski transformation: that is, express a given circuit as a sequence of Pauli
    // product rotations and Pauli product measurements, followed by a final Clifford operator.
    for node_index in dag.topological_op_nodes(false) {
        // Convert T and Tdg gates to RZ rotations.
        if let NodeType::Operation(inst) = &dag[node_index] {
            let name = inst.op.name();
            let mut is_clifford = false; // indicates if it is a pi/2 rotation gate which is a clifford

            match inst.op.view() {
                OperationRef::StandardGate(StandardGate::I) => is_clifford = true,
                OperationRef::StandardGate(StandardGate::X) => {
                    clifford.append_x(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::Y) => {
                    clifford.append_y(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::Z) => {
                    clifford.append_z(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::H) => {
                    clifford.append_h(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::S) => {
                    clifford.append_s(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::Sdg) => {
                    clifford.append_sdg(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::SX) => {
                    clifford.append_sx(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::SXdg) => {
                    clifford.append_sxdg(dag.get_qargs(inst.qubits)[0].index());
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::CX) => {
                    clifford.append_cx(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::CZ) => {
                    clifford.append_cz(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::CY) => {
                    clifford.append_cy(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::Swap) => {
                    clifford.append_swap(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::ISwap) => {
                    clifford.append_iswap(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::ECR) => {
                    clifford.append_ecr(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::DCX) => {
                    clifford.append_dcx(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    );
                    is_clifford = true
                }
                OperationRef::StandardGate(StandardGate::T)
                | OperationRef::StandardGate(StandardGate::Tdg)
                | OperationRef::StandardGate(StandardGate::RZ)
                | OperationRef::StandardGate(StandardGate::RX)
                | OperationRef::StandardGate(StandardGate::RY)
                | OperationRef::StandardGate(StandardGate::Phase)
                | OperationRef::StandardGate(StandardGate::U1) => {
                    let qubit = dag.get_qargs(inst.qubits)[0].index();
                    let param = inst.params_view();

                    // closure to process the rotation gates
                    let mut process_rot_gate = |gate: StandardGate,
                                                pauli: Pauli1q|
                     -> Option<(Pauli1q, Param)> {
                        if let Param::Float(angle) = param[0] {
                            if let Some(multiple) =
                                is_angle_close_to_multiple_of_pi_k(gate, 2, angle, tol)
                            {
                                is_clifford = true;
                                match gate {
                                    StandardGate::RZ | StandardGate::Phase | StandardGate::U1 => {
                                        clifford.append_rz(qubit, multiple)
                                    }
                                    StandardGate::RX => clifford.append_rx(qubit, multiple),
                                    StandardGate::RY => clifford.append_ry(qubit, multiple),
                                    _ => unreachable!(
                                        "We cannot have gates other than RZ/RX/RY/P/U1 at this point."
                                    ),
                                }
                                return None;
                            }
                        }
                        Some((pauli, param[0].clone()))
                    };

                    // Convert T and Tdg gates to RZ rotations
                    let (result, phase_update) = match inst.op.view() {
                        OperationRef::StandardGate(StandardGate::T) => (
                            Some((Pauli1q::Z, Param::Float(FRAC_PI_4))),
                            Param::Float(FRAC_PI_8),
                        ),
                        OperationRef::StandardGate(StandardGate::Tdg) => (
                            Some((Pauli1q::Z, Param::Float(-FRAC_PI_4))),
                            Param::Float(-FRAC_PI_8),
                        ),
                        OperationRef::StandardGate(StandardGate::RZ) => (
                            process_rot_gate(StandardGate::RZ, Pauli1q::Z),
                            Param::Float(0.),
                        ),
                        OperationRef::StandardGate(StandardGate::Phase)
                        | OperationRef::StandardGate(StandardGate::U1) => (
                            process_rot_gate(StandardGate::RZ, Pauli1q::Z),
                            multiply_param(&param[0], 0.5),
                        ),
                        OperationRef::StandardGate(StandardGate::RX) => (
                            process_rot_gate(StandardGate::RX, Pauli1q::X),
                            Param::Float(0.),
                        ),
                        OperationRef::StandardGate(StandardGate::RY) => (
                            process_rot_gate(StandardGate::RY, Pauli1q::Y),
                            Param::Float(0.),
                        ),
                        _ => {
                            unreachable!(
                                "We cannot have gates other than T/Tdg/RZ/RX/RY/P/U1 at this point."
                            );
                        }
                    };

                    // rotation gate is non-clifford
                    if let Some((pauli, angle)) = result {
                        global_phase_update = radd_param(global_phase_update, phase_update);

                        // Evolving the single qubit pauli (X, Y or Z) by the Clifford.
                        // Returns the evolved Pauli in the sparse format: (sign, pauli z, pauli x, indices),
                        // where signs `true` and `false` correspond to coefficients `-1` and `+1` respectively.
                        let (sign, z, x, indices) = clifford.evolve_single_qubit_pauli(
                            pauli,
                            dag.get_qargs(inst.qubits)[0].index(),
                        );
                        qargs.clear();
                        qargs.extend(bytemuck::cast_slice(&indices));

                        // In the legacy path, we add PauliEvolutionGate as rotation gates, otherwise
                        // we add PauliProductRotation. The new path should not call Python at any
                        // point.
                        let (packed_op, param) = if use_ppr {
                            let angle = if sign {
                                multiply_param(&angle, -1.0)
                            } else {
                                angle
                            };
                            let ppr = PauliProductRotation {
                                z,
                                x,
                                angle: angle.clone(),
                            };
                            (PauliBased::PauliProductRotation(ppr).into(), angle)
                        } else {
                            let time = if sign {
                                multiply_param(&angle, -0.5)
                            } else {
                                multiply_param(&angle, 0.5)
                            };
                            let obs = sparse_obs_from_zx(&z, &x);
                            let py_gate = Python::attach(|py| -> PyResult<_> {
                                let py_evo = PAULI_EVOLUTION_GATE
                                    .get_bound(py)
                                    .call1((obs, time.clone()))?;
                                Ok(PyInstruction {
                                    qubits: indices.len() as u32,
                                    clbits: 0,
                                    params: 1,
                                    op_name: "PauliEvolution".to_string(),
                                    ob: py_evo.into(),
                                    kind: PyOpKind::Gate,
                                })
                            })?;
                            (py_gate.into(), time)
                        };

                        new_dag.apply_operation_back(
                            packed_op,
                            &qargs,
                            &[],
                            Some(Parameters::Params(smallvec![param])),
                            None,
                            #[cfg(feature = "cache_pygates")]
                            None,
                        )?;
                    }
                }
                OperationRef::PauliProductRotation(rotation) => {
                    // Synthesize PPR
                    let in_z = &rotation.z;
                    let in_x = &rotation.x;
                    let angle = &rotation.angle;
                    let qargs_in = dag.get_qargs(inst.qubits);
                    let indices_in: Vec<u32> = (0..qargs_in.len())
                        .map(|i| qargs_in[i].index() as u32)
                        .collect();

                    if let Param::Float(angle) = angle {
                        // PPR has pi/2 angle, and so is a clifford
                        if let Some(multiple) =
                            is_ppr_angle_close_to_multiple_of_pi2(in_z, in_x, *angle, tol)
                        {
                            is_clifford = true;
                            clifford.append_ppr(in_z, in_x, &indices_in, multiple)
                        }
                    }
                    if !is_clifford {
                        // PPR is not clifford
                        // Evolve PPR by the clifford
                        let (sign, z, x, indices) = clifford.evolve_pauli(in_z, in_x, &indices_in);

                        let out_sign = if sign { -1.0 } else { 1.0 };
                        let angle = multiply_param(angle, out_sign);
                        let ppr = PauliProductRotation {
                            z,
                            x,
                            angle: angle.clone(),
                        };
                        qargs.clear();
                        qargs.extend(bytemuck::cast_slice(&indices));

                        new_dag.apply_operation_back(
                            PauliBased::PauliProductRotation(ppr).into(),
                            &qargs,
                            &[],
                            Some(Parameters::Params(smallvec![angle])),
                            None,
                            #[cfg(feature = "cache_pygates")]
                            None,
                        )?;
                    }
                }
                OperationRef::StandardInstruction(StandardInstruction::Measure) => {
                    // Evolve a measurement in the Z-basis by a Clifford.
                    // Returns the evolved Pauli in the sparse format: (sign, pauli z, pauli x, indices),
                    // where signs `true` and `false` correspond to coefficients `-1` and `+1` respectively.
                    let (sign, z, x, indices) = clifford.evolve_single_qubit_pauli(
                        Pauli1q::Z,
                        dag.get_qargs(inst.qubits)[0].index(),
                    );
                    qargs.clear();
                    qargs.extend(bytemuck::cast_slice(&indices));

                    let ppm = PauliProductMeasurement { z, x, neg: sign };

                    let ppm_clbits = dag.get_cargs(inst.clbits);

                    new_dag.apply_operation_back(
                        PauliBased::PauliProductMeasurement(ppm).into(),
                        &qargs,
                        ppm_clbits,
                        None,
                        None,
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                }
                OperationRef::PauliProductMeasurement(pp_meas) => {
                    // Evolve PPM by the clifford
                    let in_z = &pp_meas.z;
                    let in_x = &pp_meas.x;

                    let qargs_in = dag.get_qargs(inst.qubits);
                    let indices_in: Vec<u32> = (0..qargs_in.len())
                        .map(|i| qargs_in[i].index() as u32)
                        .collect();

                    let (sign, z, x, indices) = clifford.evolve_pauli(in_z, in_x, &indices_in);
                    let ppm = PauliProductMeasurement { z, x, neg: sign };
                    qargs.clear();
                    qargs.extend(bytemuck::cast_slice(&indices));

                    let ppm_clbits = dag.get_cargs(inst.clbits);

                    new_dag.apply_operation_back(
                        PauliBased::PauliProductMeasurement(ppm).into(),
                        &qargs,
                        ppm_clbits,
                        None,
                        None,
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                }
                _ => unreachable!(
                    "We cannot have unsupported names at this step of Litinski Transformation: {}",
                    name
                ),
            }
            if is_clifford && fix_clifford {
                clifford_ops.push(inst);
            }
        }
    }

    new_dag.add_global_phase(&global_phase_update)?;

    // Add Clifford gates to the Qiskit circuit (when required).
    // Since we aim to preserve the global phase of the circuit, we add the Clifford operations from
    // the original circuit (and not the final Clifford operator).
    if fix_clifford {
        // If specified, insert barriers between the Clifford and the rest of the circuit.
        if insert_barrier {
            let barrier = StandardInstruction::Barrier(dag.num_qubits() as u32).into();
            let qubits = (0..dag.num_qubits() as u32).map(Qubit).collect::<Vec<_>>();
            new_dag.apply_operation_back(
                barrier,
                &qubits,
                &[],
                None,
                None,
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
        }
        for inst in clifford_ops.into_iter() {
            new_dag.push_back(inst.clone())?;
        }
    }

    Ok(Some(new_dag.build()))
}

fn sparse_obs_from_zx(z: &[bool], x: &[bool]) -> SparseObservable {
    let bit_terms: Vec<BitTerm> = z
        .iter()
        .zip(x)
        .filter_map(|(&zi, &xi)| {
            if zi || xi {
                Some(non_identity_zx_to_bitterm(zi, xi))
            } else {
                None
            }
        })
        .collect();

    let boundaries = vec![0, bit_terms.len()];
    let coeffs = vec![Complex64::new(1.0, 0.0)];
    let num_qubits = bit_terms.len() as u32;
    let indices = (0..num_qubits).collect();

    // SAFETY: We manually built the internal data checking it is consistent.
    unsafe { SparseObservable::new_unchecked(num_qubits, coeffs, bit_terms, indices, boundaries) }
}

fn non_identity_zx_to_bitterm(z: bool, x: bool) -> BitTerm {
    match (z, x) {
        (false, false) => panic!("Identity terms not allowed."),
        (false, true) => BitTerm::X,
        (true, true) => BitTerm::Y,
        (true, false) => BitTerm::Z,
    }
}

/// For a given angle, if it is a multiple of PI/2, calculate the multiple mod (4),
/// Otherwise, return `None`.
/// I.e, if the angle is a multiple m of PI/2 then it returns m, where 0 <= m < 4.
fn is_ppr_angle_close_to_multiple_of_pi2(
    z: &[bool],
    x: &[bool],
    angle: f64,
    tol: f64,
) -> Option<usize> {
    let closest_ratio = 2.0 * angle / PI;
    let closest_integer = closest_ratio.round();
    let closest_angle = closest_integer * PI / 2.0;
    let theta = angle - closest_angle;

    // direct calculation of dim and tr_over_dim
    let num_qubits = z.iter().zip(x.iter()).filter(|(z, x)| **z || **x).count();
    let dim = 2u32.pow(num_qubits as u32);
    let tr_over_dim = if num_qubits == 0 {
        // This is an identity Pauli rotation.
        (Complex64::new(0.0, -theta / 2.)).exp()
    } else {
        Complex64::new((theta / 2.).cos(), 0.)
    };

    if average_gate_fidelity_below_tol(tr_over_dim, dim.into(), tol).is_some() {
        Some((closest_integer as i64).rem_euclid(4) as usize)
    } else {
        None
    }
}

pub fn litinski_transformation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_litinski_transformation))?;
    Ok(())
}
