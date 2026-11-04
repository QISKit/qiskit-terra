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

use qiskit_circuit::Qubit;
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::operations::{Param, StandardGate};
const EPS: f64 = 1e-10;

/// This function synthesizes UCRZ without the final CX gate,
/// unless _vw_type = ``all``.
pub(crate) fn get_ucrz(
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
        } else if vw_type_all && num_qubits > 1 {
            let q_ctrl_index = num_qubits - 2;
            let _ = out.push_standard_gate(
                StandardGate::CX,
                &[],
                &[q_controls[q_ctrl_index], q_target],
            );
        }
    }
    Ok(out)
}

/// Calculates rotation angles for a uniformly controlled Rz gate with a C-NOT gate at
/// the end of the circuit. The rotation angles are stored in `angles[start_index..end_index]`.
/// If `reversed_dec` is true, decomposes the gate such that there is a C-NOT gate at the
/// start of the circuit (the circuit topology is the reverse of the original decomposition).
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

/// Calculate the new rotation angles according to Shende's decomposition.
fn update_angle(angle_1: f64, angle_2: f64) -> [f64; 2] {
    [(angle_1 + angle_2) / 2., (angle_1 - angle_2) / 2.]
}

#[cfg(test)]
mod tests {
    use super::get_ucrz;
    use approx::abs_diff_eq;
    use ndarray::Array2;
    use num_complex::Complex64;

    use crate::matrix::sim::sim_unitary_circuit;

    #[test]
    fn test_get_ucrz_all_synthesizes_controlled_rz() {
        let angles = [0.2, 0.6];
        let circuit = get_ucrz(2, &mut angles.to_vec(), true).unwrap();
        let unitary = sim_unitary_circuit(&circuit).unwrap();

        let mut expected = Array2::zeros((4, 4));
        for (control, angle) in angles.iter().enumerate() {
            expected[[2 * control, 2 * control]] = Complex64::new(0.0, -angle / 2.0).exp();
            expected[[2 * control + 1, 2 * control + 1]] = Complex64::new(0.0, angle / 2.0).exp();
        }
        assert!(abs_diff_eq!(unitary, expected, epsilon = 1e-12));
    }

    #[test]
    fn test_get_ucrz_without_final_cx() {
        let mut angles = vec![0.2, 0.6];
        let circuit = get_ucrz(2, &mut angles, false).unwrap();

        assert_eq!(circuit.data().len(), 3);
    }
}
