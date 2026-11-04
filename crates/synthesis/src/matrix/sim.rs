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

use ndarray::Array2;
use num_complex::Complex64;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardInstruction};
use qiskit_quantum_info::unitary_compose;

use crate::QiskitError;

// The code is based on top of unitary_compose. For circuits with 13 or more qubits, einsum
// throws an "index out of bounds" error.
const MAX_NUM_QUBITS: usize = 12;

/// Create a unitary matrix for a circuit.
pub fn sim_unitary_circuit(circuit: &CircuitData) -> Result<Array2<Complex64>, String> {
    if circuit.num_clbits() > 0 {
        return Err("Cannot simulate circuit involving classical bits.".to_string());
    }

    let num_qubits = circuit.num_qubits();

    if num_qubits > MAX_NUM_QUBITS {
        return Err(format!(
            "The number of circuit qubits ({num_qubits}) exceeds the maximum allowed number of qubits allowed for simulation ({MAX_NUM_QUBITS})."
        ));
    }

    // e^{i * global_phase}
    let global_phase_exp: Complex64 = if let Param::Float(p) = circuit.global_phase() {
        Complex64::new(0., *p).exp()
    } else {
        return Err("Cannot simulate circuit involving non-float global phase.".to_string());
    };

    // Product matrix holding the result
    let mut product_mat: Array2<Complex64> =
        Array2::<Complex64>::eye(2_usize.pow(num_qubits as u32)) * global_phase_exp;
    let mut qubits = Vec::new();

    for inst in circuit.data() {
        if !circuit.get_cargs(inst.clbits).is_empty() {
            return Err(
                "Cannot simulate circuit with instructions involving classical bits".to_string(),
            );
        }

        // Ignore barriers
        if let OperationRef::StandardInstruction(StandardInstruction::Barrier(_)) = inst.op.view() {
            continue;
        }

        qubits.clear();
        qubits.extend(circuit.get_qargs(inst.qubits).iter().map(|q| q.index()));

        let mat = inst
            .try_matrix()
            .ok_or_else(|| format!("Cannot extract matrix for operation {:?}.", inst.op.name()))?;

        product_mat = unitary_compose::compose(&product_mat.view(), &mat.view(), &qubits, false)?;
    }

    Ok(product_mat)
}

/// Create a unitary matrix for a circuit.
#[pyfunction]
#[pyo3(name = "sim_unitary_circuit")]
pub fn py_sim_unitary_circuit(py: Python, circuit: &PyCircuitData) -> PyResult<Py<PyAny>> {
    let product_mat = sim_unitary_circuit(circuit).map_err(QiskitError::new_err)?;
    Ok(product_mat.into_pyarray(py).into_any().unbind())
}

pub fn unitary_sim(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_sim_unitary_circuit))?;
    Ok(())
}

#[cfg(all(test, not(miri)))]
mod test {
    use super::sim_unitary_circuit;
    use approx::abs_diff_eq;
    use qiskit_circuit::operations::StandardGate;

    #[test]
    fn test_sim_ecr_definition() {
        // Check that the definition circuit for the ECRGate (which in particular involves
        // a non-trivial global phase) gets simulated correctly.
        let ecr_definition = StandardGate::ECR.definition(&[]).unwrap();

        let simulated_matrix = sim_unitary_circuit(&ecr_definition).unwrap();
        let expected_matrix = StandardGate::ECR.matrix(&[]).unwrap();

        let close = abs_diff_eq!(simulated_matrix, expected_matrix, epsilon = 1e-12);
        assert!(close);
    }
}
