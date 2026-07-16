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

use pyo3::prelude::*;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use crate::QiskitError;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{OperationRef, Param, StandardGate, add_param};
use qiskit_synthesis::ross_selinger::{gridsynth_cleanup, gridsynth_rz};

const MINIMUM_EPSILON: f64 = 1e-12; // minimum epsilon for synthesis

/// Finds a canonical representation of an angle.
///
/// Given `angle`, this returns `(interval, angle_normalized)` such that
/// angle_normalized = angle (mod pi/2)
/// (angle - angle_normalized - interval * pi/2) = 0 (mod 4pi)
///
/// The canonical representation is limited by the f64 representation.
/// Any angle that differs in decimal places beyond f64 will be non-unique.
fn canonicalize_angle(angle: f64) -> (u8, f64) {
    if (0. ..FRAC_PI_2).contains(&angle) {
        return (0, angle);
    }

    let angle_normalized = angle.rem_euclid(FRAC_PI_2);
    let interval = ((angle - angle_normalized) / FRAC_PI_2)
        .round()
        .rem_euclid(8.) as u8;
    (interval, angle_normalized)
}

/// Lookup table for fixing the circuit based on interval computed during
/// canonicalization.
///
/// The table is based on the properties of `Rz(theta)` such as:
/// * `Rz(theta + pi/2) = Rz(theta).S`, up to a global phase of `-pi/4`,
/// * `Rz(theta + pi) = Rz(theta).Z`, up to a global phase of `-pi/2`,
/// * `Rz(theta + 2*pi) = -Rz(theta)`, up to a global phase of `-pi`.
static PHASE_GATE_LUT: [(f64, Option<StandardGate>); 8] = [
    (0.0, None),
    (-FRAC_PI_4, Some(StandardGate::S)),
    (-FRAC_PI_2, Some(StandardGate::Z)),
    (-3. * FRAC_PI_4, Some(StandardGate::Sdg)),
    (PI, None),
    (-5. * FRAC_PI_4, Some(StandardGate::S)),
    (-6. * FRAC_PI_4, Some(StandardGate::Z)),
    (-7. * FRAC_PI_4, Some(StandardGate::Sdg)),
];

/// Approximates RZ-rotation using gridsynth.
///
/// Returns the sequence of gates in the synthesized circuit and
/// an update to the global phase.
fn synthesize_rz_gate_via_gridsynth(
    angle: f64,
    epsilon: f64,
) -> PyResult<(Vec<StandardGate>, Param)> {
    let circ_data = gridsynth_rz(angle, epsilon)?;

    // obtain phase from circuit data
    let phase = circ_data.global_phase().clone();

    // get sequence of standard gates
    let sequence: Vec<StandardGate> = circ_data
        .data()
        .iter()
        .map(|inst| {
            if let OperationRef::StandardGate(gate) = inst.op.view() {
                gate
            } else {
                unreachable!("gridsynth only produces standard gates");
            }
        })
        .collect();

    Ok((sequence, phase))
}

/// Synthesize RZ gates in the circuit, modifying the circuit in-place.
///
/// # Arguments
///
/// - `dag`: The DAG circuit in which the RZ gates will be synthesized.
/// - `approximation_degree`: Controls the overall degree of approximation.
/// - `synthesis_error`: Maximum allowed error for the approximate synthesis of
///   :math:`RZ(\theta)`.
/// - `cache_error`: Maximum allowed error when reusing a cached synthesis
///   result for angles close to :math:`\theta`.
///
/// If both `synthesis_error` and `cache_error` are provided, they specify the error budget
/// due to approximate synthesis and due to caching respectively. If either value is not
/// specified, the total allowed error is derived from `approximation_degree`, and
/// suitable values for `synthesis_error` and `cache_error` are computed automatically.
#[pyfunction]
#[pyo3(name = "synthesize_rz_rotations")]
#[pyo3(signature = (dag, approximation_degree=None, synthesis_error=None, cache_error=None))]
pub fn py_run_synthesize_rz_rotations(
    dag: &mut DAGCircuit,
    approximation_degree: Option<f64>,
    synthesis_error: Option<f64>,
    cache_error: Option<f64>,
) -> PyResult<()> {
    // Skip the pass if there are no RZ rotation gates.
    if dag.get_op_counts().keys().all(|k| k != "rz") {
        return Ok(());
    }

    // rsgridsynth internally caches results of various computations that depend on the
    // number of precision bits. Generally speaking, these caches become invalid when
    // precision changes. Fortunately, rsgridsynth exposes a method to clear some (though not all)
    // of these caches. The current approach is to clear the caches at the start of each run
    // of SynthesizeRZRotations, while the cached values can be safely reused for all
    // rotation gates in the circuit.
    // This is not a complete or perfect solution, so a major refactor of
    // gridsynth is probably needed.
    gridsynth_cleanup();

    // Compute error budgets. When approximation degree is used, the total error is
    // computed as 1 - approximation_degree, and the error budget for synthesis and for
    // caching are distributed equally.
    let (synthesis_error, cache_error) = match (synthesis_error, cache_error) {
        (Some(synthesis_error), Some(cache_error)) => (synthesis_error, cache_error),
        _ => {
            let total_error = if let Some(approximation_degree) = approximation_degree {
                MINIMUM_EPSILON.max(1. - approximation_degree)
            } else {
                MINIMUM_EPSILON // use minimum epsilon per default 
            };
            (total_error / 2., total_error / 2.)
        }
    };

    // By an explicit computation one can show that if the current angle is within
    // 4.0 * arcsin(cache_error / 2) from the previous angle, the error due to reusing the synthesis
    // result for the previous angle is precisely cache_error. Contact a Qiskit synthesis developer
    // for more details!
    let bin_width = 4. * (cache_error / 2.).asin();

    // Iterate over nodes in the DAG and collect nodes that have RZ gates.
    // Canonicalize angles already at this stage, so that we can use them for sorting.
    let mut candidates: Vec<_> = dag
        .op_nodes(false)
        .filter_map(|(node_index, inst)| {
            if let OperationRef::StandardGate(StandardGate::RZ) = inst.op.view() {
                if let Param::Float(angle) = inst.params_view()[0] {
                    let (interval_index, canonical_angle) = canonicalize_angle(angle);
                    Some((node_index, canonical_angle, interval_index))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    // Sort candidates based on the canonicalized angles
    candidates.sort_unstable_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .expect("Angles are never NaN here, so we can compare f64.")
    });

    let mut prev_result: Option<(f64, (Vec<StandardGate>, Param))> = None;

    for (node_index, angle, interval_index) in candidates {
        // Get or compute the sequence and phase update.
        let should_recompute = prev_result
            .as_ref()
            .is_none_or(|(prev_angle, _)| *prev_angle + bin_width < angle);

        if should_recompute {
            let (sequence, phase_update) = synthesize_rz_gate_via_gridsynth(angle, synthesis_error)
                .map_err(|e| QiskitError::new_err(e.to_string()))?;

            prev_result = Some((angle, (sequence, phase_update)));
        }

        let (sequence, phase_update) = &prev_result
            .as_ref()
            .expect("is_none_or ensures prev_result is never None")
            .1;

        // Add the gates and phase update to DAG, remove old node
        for new_gate in sequence {
            dag.insert_1q_on_incoming_qubit((*new_gate, &[]), node_index);
        }
        if let Some(gate) = PHASE_GATE_LUT[interval_index as usize].1 {
            dag.insert_1q_on_incoming_qubit((gate, &[]), node_index);
        }
        dag.remove_1q_sequence(&[node_index]);

        let phase_update_with_shift =
            add_param(phase_update, PHASE_GATE_LUT[interval_index as usize].0);
        dag.add_global_phase(&phase_update_with_shift)?;
    }

    Ok(())
}

pub fn synthesize_rz_rotations_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_synthesize_rz_rotations))?;
    Ok(())
}
