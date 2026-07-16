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
use num_complex::Complex64;
use pyo3::prelude::*;
use qiskit_circuit::dag_circuit::DAGError;
use rayon::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use rustworkx_core::petgraph::visit::NodeIndexable;

use crate::commutation_checker::try_matrix_with_definition;
use crate::gate_metrics::rotation_trace_and_dim;
use crate::passes::common::{MINIMUM_TOL, average_gate_fidelity_below_tol};
use crate::target::Target;
use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_util::getenv_use_multiple_threads;

/// Possible errors returned by the `RemoveIdentityEquiv` pass.
#[derive(Debug, thiserror::Error)]
pub enum RemoveIdentityEquivError {
    #[error(transparent)]
    DAGCircuit(#[from] DAGError),
    // TODO: Remove once Pauli Gates are in Rust.
    #[error(transparent)]
    PyPauliRotationTraceAndDim(PyErr),
}

impl From<RemoveIdentityEquivError> for PyErr {
    fn from(value: RemoveIdentityEquivError) -> Self {
        match value {
            RemoveIdentityEquivError::DAGCircuit(dagcircuit_inner_error) => {
                dagcircuit_inner_error.into()
            }
            RemoveIdentityEquivError::PyPauliRotationTraceAndDim(py_err) => py_err,
        }
    }
}

// The point at which to start running the analysis in parallel.
// This value was found experimentally during the development of
// https://github.com/Qiskit/qiskit/pull/14719 it can be tweaked
// if the performance of this pass changes over time.
const PARALLEL_THRESHOLD: usize = 50_000;

/// Fidelity-based computation to check whether an operation `inst` is equivalent
/// to identity up to a global phase.
///
/// # Arguments
///
/// * `inst`: the packed instruction
/// * `matrix_from_definition`: if `true`, can call the Python-space `Operator` class to
///   construct the matrix.
/// * `matrix_from_definition_max_qubits`: maximum number of qubits allowed for matrix-based
///   checks.
/// * `error_cutoff_fn`: function to compute the allowed error tolerance.
///
/// # Returns
///
/// * `Some(update to the global phase)` if the operation can be removed.
/// * `None` if the operation cannot be removed.
pub fn is_identity_equiv<F>(
    inst: &PackedInstruction,
    matrix_from_definition: bool,
    matrix_from_definition_max_qubits: Option<u32>,
    error_cutoff_fn: F,
) -> Result<Option<f64>, RemoveIdentityEquivError>
where
    F: Fn(&PackedInstruction) -> f64,
{
    if inst.is_parameterized() {
        // Skip parameterized gates
        return Ok(None);
    }

    let view = inst.op.view();

    if let OperationRef::StandardGate(gate) = view {
        let (tr_over_dim, dim) = match gate {
            StandardGate::I => {
                return Ok(Some(0.));
            }
            StandardGate::GlobalPhase => {
                if let Param::Float(angle) = inst.params_view()[0] {
                    return Ok(Some(angle));
                } else {
                    // We cannot get here since we skip parameterized gates,
                    // but in theory we could return Ok((true, param)) here.
                    return Ok(None);
                }
            }
            StandardGate::RX
            | StandardGate::RY
            | StandardGate::RZ
            | StandardGate::Phase
            | StandardGate::U1
            | StandardGate::RXX
            | StandardGate::RYY
            | StandardGate::RZX
            | StandardGate::RZZ
            | StandardGate::CRX
            | StandardGate::CRY
            | StandardGate::CRZ
            | StandardGate::CU1
            | StandardGate::CPhase => {
                if let Param::Float(angle) = inst.params_view()[0] {
                    let (tr_over_dim, dim) = rotation_trace_and_dim(gate, angle).expect(
                        "Since only supported rotation gates are given, the result is not None",
                    );
                    (tr_over_dim, dim)
                } else {
                    return Ok(None);
                }
            }
            StandardGate::H
            | StandardGate::X
            | StandardGate::Y
            | StandardGate::Z
            | StandardGate::S
            | StandardGate::Sdg
            | StandardGate::SX
            | StandardGate::SXdg
            | StandardGate::T
            | StandardGate::Tdg
            | StandardGate::CX
            | StandardGate::CY
            | StandardGate::CZ
            | StandardGate::CH
            | StandardGate::DCX
            | StandardGate::ECR
            | StandardGate::Swap
            | StandardGate::ISwap
            | StandardGate::CS
            | StandardGate::CSdg
            | StandardGate::CSX
            | StandardGate::CCX
            | StandardGate::CCZ
            | StandardGate::CSwap
            | StandardGate::RCCX
            | StandardGate::C3X
            | StandardGate::C3SX
            | StandardGate::RC3X => {
                return Ok(None);
            }
            _ => {
                // The remaining standard gates are R, U, U2, U3, CU, CU3, XXMinusYY and XXPlusYY.
                // We could consider extending the function rotation_trace_and_dim to handle
                // these gates, without needing to compute actual matrices.
                if let Some(matrix) = inst.try_matrix() {
                    let dim = matrix.shape()[0] as f64;
                    let tr_over_dim = matrix.diag().iter().sum::<Complex64>() / dim;
                    (tr_over_dim, dim)
                } else {
                    return Ok(None);
                }
            }
        };

        return Ok(average_gate_fidelity_below_tol(
            tr_over_dim,
            dim,
            error_cutoff_fn(inst),
        ));
    }

    // Special handling for pauli product rotation gates.
    if let OperationRef::PauliProductRotation(ppr) = view {
        if let Some((tr_over_dim, dim)) = ppr.rotation_trace_and_dim() {
            return Ok(average_gate_fidelity_below_tol(
                tr_over_dim,
                dim,
                error_cutoff_fn(inst),
            ));
        } else {
            // Parameterized rotation
            return Ok(None);
        }
    }

    // Special handling for pauli evolution gates.
    if view.name() == "PauliEvolution" {
        if let OperationRef::PyCustom(py_gate) = view {
            let result = Python::attach(|py| -> PyResult<Option<(Complex64, usize)>> {
                let result = imports::PAULI_ROTATION_TRACE_AND_DIM
                    .get_bound(py)
                    .call1((py_gate.ob.clone_ref(py),))?
                    .extract()?;
                Ok(result)
            })
            .map_err(RemoveIdentityEquivError::PyPauliRotationTraceAndDim)?;

            if let Some((tr_over_dim, dim)) = result {
                return Ok(average_gate_fidelity_below_tol(
                    tr_over_dim,
                    dim as f64,
                    error_cutoff_fn(inst),
                ));
            }
        }
    }

    // If matrix_from_definition is false and view.matrix() returns None, we skip the operation.
    // If matrix_from_definition is true, we also attempt to construct the matrix from the python Operator.
    if let Some(matrix) = match matrix_from_definition {
        false => inst.try_matrix(),
        true => {
            try_matrix_with_definition(&view, inst.params_view(), matrix_from_definition_max_qubits)
        }
    } {
        let dim = matrix.shape()[0] as f64;
        let tr_over_dim = matrix.diag().iter().sum::<Complex64>() / dim;
        return Ok(average_gate_fidelity_below_tol(
            tr_over_dim,
            dim,
            error_cutoff_fn(inst),
        ));
    }

    Ok(None)
}

#[pyfunction]
#[pyo3(name = "remove_identity_equiv", signature=(dag, approx_degree=Some(1.0), target=None))]
pub fn py_remove_identity_equiv(
    py: Python,
    dag: &mut DAGCircuit,
    approx_degree: Option<f64>,
    target: Option<&Target>,
) -> PyResult<()> {
    // TODO: This is a hack to avoid panicking in the case that the global phase contains `Py`
    // pointers (such as backrefs to `ParameterVector` objects in an expression `Symbol`) that would
    // get cloned when updating the global phase.  It's easier to do it out here than to try to
    // reattach to Python-space within a pure-Rust function but only if we can spot a hiding `Py`
    // (or we'd break standalone C).
    //
    // This doesn't account for control-flow blocks which _also_ might have set global phases, byt
    // `run_remove_identity_equiv` as of Qiskit 2.4 doesn't recurse, so the hack should hold.
    let old_phase = dag.set_global_phase_f64(0.0);

    // Explicitly release GIL because threads may call Python to get
    // the matrix for a PyGate
    py.detach(|| run_remove_identity_equiv(dag, approx_degree, target))?;

    dag.add_global_phase(&old_phase)?;
    Ok(())
}

pub fn run_remove_identity_equiv(
    dag: &mut DAGCircuit,
    approx_degree: Option<f64>,
    target: Option<&Target>,
) -> Result<(), RemoveIdentityEquivError> {
    // Minimum threshold to compare average gate fidelity to 1. This is chosen to account
    // for roundoff errors and to be consistent with other places.
    let get_error_cutoff = |inst: &PackedInstruction| -> f64 {
        match approx_degree {
            Some(degree) => {
                if degree == 1.0 {
                    MINIMUM_TOL
                } else {
                    match target {
                        Some(target) => {
                            let qargs: Vec<PhysicalQubit> = dag
                                .get_qargs(inst.qubits)
                                .iter()
                                .map(|x| PhysicalQubit::new(x.0))
                                .collect();
                            let error_rate = target.get_error(inst.op.name(), &qargs);
                            match error_rate {
                                Some(err) => err * degree,
                                None => MINIMUM_TOL.max(1. - degree),
                            }
                        }
                        None => MINIMUM_TOL.max(1. - degree),
                    }
                }
            }
            None => match target {
                Some(target) => {
                    let qargs: Vec<PhysicalQubit> = dag
                        .get_qargs(inst.qubits)
                        .iter()
                        .map(|x| PhysicalQubit::new(x.0))
                        .collect();
                    let error_rate = target.get_error(inst.op.name(), &qargs);
                    match error_rate {
                        Some(err) => err,
                        None => MINIMUM_TOL,
                    }
                }
                None => MINIMUM_TOL,
            },
        }
    };

    let process_node = |op_node: NodeIndex, inst: &PackedInstruction| {
        is_identity_equiv(inst, false, None, get_error_cutoff)
            .map(|result| result.map(|x| (op_node, x)))
    };
    let run_in_parallel = getenv_use_multiple_threads();
    let remove_list: Vec<(NodeIndex, f64)> =
        if dag.num_ops() >= PARALLEL_THRESHOLD && run_in_parallel {
            (0..dag.dag().node_bound())
                .into_par_iter()
                .filter_map(|index_val| {
                    let index = NodeIndex::new(index_val);
                    if let Some(NodeType::Operation(inst)) = dag.dag().node_weight(index) {
                        process_node(index, inst).transpose()
                    } else {
                        None
                    }
                })
                .collect::<Result<Vec<(NodeIndex, f64)>, RemoveIdentityEquivError>>()?
        } else {
            dag.op_nodes(false)
                .filter_map(|x| process_node(x.0, x.1).transpose())
                .collect::<Result<Vec<(NodeIndex, f64)>, RemoveIdentityEquivError>>()?
        };

    for (node, phase_update) in remove_list {
        dag.remove_op_node(node);
        dag.add_global_phase(&Param::Float(phase_update))
            .expect("The global phase is guaranteed to be a float");
    }
    Ok(())
}

pub fn remove_identity_equiv_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_remove_identity_equiv))?;
    Ok(())
}
