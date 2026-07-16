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

use nalgebra::{Matrix2, Matrix3};
use numpy::{Complex64, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyString;
use pyo3::{prelude::*, types::PyList};
use qiskit_circuit::NoBlocks;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::instruction::Instruction;
use qiskit_circuit::operations::{OperationRef, Param, PyInstruction, PyOpKind, StandardGate};

use crate::discrete_basis::basic_approximations::{BasicApproximations, GateSequence};

use super::basic_approximations::DiscreteBasisError;
use super::math::{self, group_commutator_decomposition};

/// A stateful implementation of Solovay Kitaev.
///
/// The code is based mainly on https://arxiv.org/pdf/quant-ph/0505030.
///
/// This generates the basic approximation set once as R-tree and re-uses it for
/// each queried decomposition.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct SolovayKitaevSynthesis {
    /// The set of basic approximations.
    basic_approximations: BasicApproximations,
    /// Whether to perform runtime checks on the handled matrices/data.
    do_checks: bool,
}

impl SolovayKitaevSynthesis {
    /// Initialize a new instance by generating a set of basic approximations, given a list
    /// of discrete standard gates and a depth. Returns an error if any gate is not discrete.
    pub fn new(
        basis_gates: &[StandardGate],
        depth: usize,
        tol: Option<f64>,
        do_checks: bool,
    ) -> Result<Self, DiscreteBasisError> {
        let basic_approximations = BasicApproximations::generate_from(basis_gates, depth, tol)?;
        Ok(Self {
            basic_approximations,
            do_checks,
        })
    }

    /// Run the Solovay Kitaev algorithm, given the initial unitary as U(2) matrix.
    ///
    /// This matrix is given using [Complex64] numbers, which can limit the precision of the
    /// algorithm. It is preferred to run the algorithm using [Self::synthesize_gate], which lets
    /// Qiskit attempt to construct the underlying SO(3) matrix representation using [f64] at
    /// quadruple precision, instead of inferring it from the double precision U(2) matrix.
    pub fn synthesize_matrix(
        &self,
        matrix_u2: &Matrix2<Complex64>,
        recursion_degree: usize,
    ) -> Result<CircuitData, DiscreteBasisError> {
        let (matrix_so3, phase) = math::u2_to_so3(matrix_u2);
        let mut output = self.recurse(&matrix_so3, recursion_degree);

        output.inverse_cancellation();

        // Compute the error in the SO(3) representation to return with the circuit.
        let circuit = output.to_circuit(Some((matrix_u2, phase)))?;
        Ok(circuit)
    }

    /// Run the Solovay Kitaev algorithm on a standard gate + gate parameters.
    ///
    /// This attempts to directly construct the SO(3) matrix representation to minimize roundoff
    /// errors. If unsuccessful, this falls back onto constructing the standard U(2) matrix
    /// representation and converting it to SO(3).
    pub fn synthesize_standard_gate(
        &self,
        gate: &StandardGate,
        params: &[Param],
        recursion_degree: usize,
    ) -> Result<CircuitData, DiscreteBasisError> {
        let (matrix_so3, phase) = math::standard_gates_to_so3(gate, params)?;
        let mut output = self.recurse(&matrix_so3, recursion_degree);

        output.inverse_cancellation();

        // Compute the error in the SO(3) representation to return with the circuit.
        let array_u2 = gate
            .matrix(params)
            .expect("StandardGate::matrix() should return a matrix.");
        let matrix_u2 = math::array2_to_matrix2(&array_u2.view());
        let circuit = output.to_circuit(Some((&matrix_u2, phase)))?;
        Ok(circuit)
    }

    /// Run the Solovay Kitaev algorithm on an operation.
    pub fn synthesize_operation(
        &self,
        op: &OperationRef,
        params: &[Param],
        recursion_degree: usize,
    ) -> Result<CircuitData, DiscreteBasisError> {
        match op {
            OperationRef::StandardGate(gate) => {
                self.synthesize_standard_gate(gate, params, recursion_degree)
            }
            OperationRef::Unitary(unitary) => {
                let matrix = unitary.matrix_view();
                let matrix_nalgebra: Matrix2<Complex64> = Matrix2::from_fn(|i, j| matrix[(i, j)]);
                self.synthesize_matrix(&matrix_nalgebra, recursion_degree)
            }
            OperationRef::PyCustom(
                gate @ PyInstruction {
                    kind: PyOpKind::Gate,
                    ..
                },
            ) => {
                let matrix = gate.matrix().ok_or(DiscreteBasisError::NoMatrix)?;
                let matrix_nalgebra: Matrix2<Complex64> = Matrix2::from_fn(|i, j| matrix[(i, j)]);
                self.synthesize_matrix(&matrix_nalgebra, recursion_degree)
            }
            OperationRef::PauliProductRotation(ppr) => {
                let matrix = ppr.matrix().expect("PPR should have a matrix defined");
                let matrix_nalgebra: Matrix2<Complex64> = Matrix2::from_fn(|i, j| matrix[(i, j)]);
                self.synthesize_matrix(&matrix_nalgebra, recursion_degree)
            }
            _ => Err(DiscreteBasisError::NoMatrix),
        }
    }

    /// Run a recursion step for a gate sequence, given a recursion degree.
    ///
    /// If the degree is 0 (recursion root), return the closest approximation in the set of
    /// basic approximations. Otherwise, decompose the difference between the approximation
    /// and the target unitary as balanced group commutator, and recurse on each element.
    fn recurse(&self, matrix_so3: &Matrix3<f64>, degree: usize) -> GateSequence {
        // Recursion root: return the best approximation in the precomputed set.
        if degree == 0 {
            if self.do_checks {
                math::assert_so3("Recursion root", matrix_so3);
            }
            let basic_approximation = self
                .basic_approximations
                .query(matrix_so3)
                .expect("No basic approximation in root found")
                .clone();
            return basic_approximation;
        }

        // Find a basic approximation of the target sequence...
        let u_n1 = self.recurse(matrix_so3, degree - 1);

        // ... and then improve the delta in between that approximation and the target.
        let delta = matrix_so3 * u_n1.matrix_so3.transpose();
        let (matrix_vn, matrix_wn) = group_commutator_decomposition(&delta, self.do_checks);

        // Recurse on the group commutator elements.
        let v_n1 = self.recurse(&matrix_vn, degree - 1);
        let w_n1 = self.recurse(&matrix_wn, degree - 1);

        v_n1.dot(&w_n1)
            .dot(&v_n1.adjoint())
            .dot(&w_n1.adjoint())
            .dot(&u_n1)
    }

    /// Store the basic approximations into a file.
    fn save(&self, filename: &str) -> ::std::io::Result<()> {
        self.basic_approximations.save(filename)
    }

    /// Load basic approximation from a file to instantiate this class.
    fn from_basic_approximations(filename: &str, do_checks: bool) -> ::std::io::Result<Self> {
        let basic_approximations = BasicApproximations::load(filename)?;
        Ok(Self {
            basic_approximations,
            do_checks,
        })
    }
}

#[pymethods]
impl SolovayKitaevSynthesis {
    /// Args:
    ///     basis_gates (list[Gate] | None): A list of discrete (i.e., non-parameterized) standard
    ///         gates. Defaults to ``[H, T, Tdg]``.
    ///     depth (int): The number of basis gate combinations to consider in the basis set. This
    ///         determines how fast (and if) the algorithm converges and should be chosen
    ///         sufficiently high. Defaults to 12,
    ///     tol (float | None): A tolerance determining the granularity of the basic approximations.
    ///         Any sequence whose SO(3) representation is within :math:`\sqrt{\texttt{tol}}` of
    ///         an existing point, will be discarded. Defaults to ``1e-12``.
    #[new]
    #[pyo3 (signature = (basis_gates=None, depth=12, tol=None, do_checks=false))]
    fn py_new(
        basis_gates: Option<&Bound<PyList>>,
        depth: usize,
        tol: Option<f64>,
        do_checks: bool,
    ) -> PyResult<Self> {
        // Extract list of standard gates from the input. Errors if a gate is not a standard gate.
        let basis_gates: Vec<StandardGate> = match basis_gates {
            None => vec![StandardGate::H, StandardGate::T, StandardGate::Tdg],
            Some(py_gates) => py_gates
                .iter()
                .map(|el| {
                    let py_op = el.extract::<OperationFromPython<NoBlocks>>()?;
                    match py_op.operation.view() {
                        OperationRef::StandardGate(gate) => Ok(gate),
                        _ => Err(PyValueError::new_err("Only standard gates accepted.")),
                    }
                })
                .collect::<PyResult<_>>()?,
        };

        // Construct self. Errors if a gate is not a discrete standard gate.
        Self::new(&basis_gates, depth, tol, do_checks).map_err(|err| err.into())
    }

    /// Getter for whether to perform runtime checks on the inputs.
    #[getter]
    fn get_do_checks(&self) -> PyResult<bool> {
        Ok(self.do_checks)
    }

    /// Setter for whether to perform runtime checks on the inputs.
    #[setter]
    fn set_do_checks(&mut self, value: bool) -> PyResult<()> {
        self.do_checks = value;
        Ok(())
    }

    /// Run the Solovay-Kitaev algorithm on a :math:`U(2)` input matrix.
    ///
    /// For better accuracy, it is suggested to use :meth:`synthesize`, which provides the
    /// :class:`.Gate` to decompose and allows Qiskit to internally create a high-accuracy
    /// representation.
    ///
    /// Args:
    ///     matrix (np.ndarray): A 2x2 complex matrix representing a 1-qubit gate.
    ///     recursion_degree (int): The recursion degree of the algorithm.
    ///
    /// Returns:
    ///     CircuitData: The ``CircuitData`` implementing the approximation.
    #[pyo3(name = "synthesize_matrix")]
    #[pyo3(signature = (gate_matrix, recursion_degree))]
    fn py_synthesize_matrix(
        &self,
        gate_matrix: PyReadonlyArray2<Complex64>,
        recursion_degree: usize,
    ) -> PyResult<PyCircuitData> {
        let view = matrix2_from_pyreadonly(&gate_matrix);
        self.synthesize_matrix(&view, recursion_degree)
            .map(Into::into)
            .map_err(|err| err.into())
    }

    /// Run the Solovay-Kitaev algorithm on an operation.
    ///
    /// Args:
    ///     gate (Gate): The operation to approximate.
    ///     recursion_degree (int): The recursion degree of the algorithm.
    ///
    /// Returns:
    ///     CircuitData: The ``CircuitData`` implementing the approximation.
    fn synthesize(
        &self,
        gate: OperationFromPython<NoBlocks>,
        recursion_degree: usize,
    ) -> PyResult<PyCircuitData> {
        self.synthesize_operation(&gate.operation.view(), gate.params_view(), recursion_degree)
            .map(Into::into)
            .map_err(|err| err.into())
    }

    /// Query the basic approximation for a [GateSequence].
    ///
    /// Legacy compat.
    fn find_basic_approximation(&self, sequence: &GateSequence) -> GateSequence {
        let approximation = self
            .basic_approximations
            .query(&sequence.matrix_so3)
            .expect("No basic approximation found");

        approximation.clone()
    }

    /// Query the basic approximation for a :class:`.Gate`.
    ///
    /// Args:
    ///     gate (Gate): The gate sequence to find the approximation of.
    ///
    /// Returns:
    ///     CircuitData: The sequence in the set of basic approximations closest to the input.
    fn query_basic_approximation(
        &self,
        gate: OperationFromPython<NoBlocks>,
    ) -> PyResult<PyCircuitData> {
        let matrix_u2 = match gate.try_matrix() {
            Some(matrix) => Matrix2::new(
                matrix[(0, 0)],
                matrix[(0, 1)],
                matrix[(1, 0)],
                matrix[(1, 1)],
            ),
            None => {
                return Err(PyValueError::new_err(
                    "Failed to construct matrix representation.",
                ));
            }
        };

        let (matrix_so3, phase) = math::u2_to_so3(&matrix_u2);
        let approximation = self
            .basic_approximations
            .query(&matrix_so3)
            .expect("No basic approximation found");

        approximation
            .to_circuit(Some((&matrix_u2, phase)))
            .map(Into::into)
            .map_err(|e| e.into())
    }

    /// Query the basic approximation for an U(2) matrix.
    ///
    /// Args:
    ///     matrix (np.ndarray): The gate sequence to find the approximation of.
    ///
    /// Returns:
    ///     CircuitData: The sequence in the set of basic approximations closest to the input.
    fn query_basic_approximation_matrix(
        &self,
        matrix: PyReadonlyArray2<Complex64>,
    ) -> PyResult<PyCircuitData> {
        let matrix_u2 = matrix2_from_pyreadonly(&matrix);
        let (matrix_so3, phase) = math::u2_to_so3(&matrix_u2);

        let approximation = self
            .basic_approximations
            .query(&matrix_so3)
            .expect("No basic approximation found");

        approximation
            .to_circuit(Some((&matrix_u2, phase)))
            .map(Into::into)
            .map_err(|e| e.into())
    }

    /// Store the basic approximations.
    fn save_basic_approximations(&self, filename: &Bound<'_, PyString>) -> PyResult<()> {
        let filename = filename.extract::<String>()?;
        self.save(filename.as_str())
            .map_err(PyRuntimeError::new_err)
    }

    /// Load from basic approximations.
    #[staticmethod]
    #[pyo3(name = "from_basic_approximations")]
    fn py_from_basic_approximations(
        filename: &Bound<'_, PyString>,
        do_checks: bool,
    ) -> PyResult<Self> {
        let filename = filename.extract::<String>()?;
        Self::from_basic_approximations(filename.as_str(), do_checks)
            .map_err(PyRuntimeError::new_err)
    }

    /// Load from a list of [GateSequence]s.
    #[staticmethod]
    fn from_sequences(sequences: Vec<GateSequence>, do_checks: bool) -> Self {
        let basic_approximations = BasicApproximations::load_from_sequences(&sequences);
        Self {
            basic_approximations,
            do_checks,
        }
    }

    /// Get a list of all [GateSequence]s in the basic approximations.
    ///
    /// Legacy compatibility.
    fn get_gate_sequences(&self) -> Vec<GateSequence> {
        self.basic_approximations
            .approximations
            .values()
            .cloned()
            .collect()
    }
}

#[inline]
fn matrix2_from_pyreadonly(array: &PyReadonlyArray2<Complex64>) -> Matrix2<Complex64> {
    Matrix2::new(
        *array.get((0, 0)).unwrap(),
        *array.get((0, 1)).unwrap(),
        *array.get((1, 0)).unwrap(),
        *array.get((1, 1)).unwrap(),
    )
}
