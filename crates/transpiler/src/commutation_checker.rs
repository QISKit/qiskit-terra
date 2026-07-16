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

use hashbrown::{HashMap, HashSet};
use ndarray::Array2;
use ndarray::linalg::kron;
use num_complex::Complex64;
use num_complex::ComplexFloat;
use qiskit_circuit::object_registry::PyObjectAsKey;
use qiskit_circuit::standard_gate::standard_generators::standard_gate_exponent;
use qiskit_quantum_info::sparse_observable::{PySparseObservable, SparseObservable};
use smallvec::SmallVec;
use std::fmt::Debug;

use numpy::PyReadonlyArray2;
use pyo3::BoundObject;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyTuple};

use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_node::DAGOpNode;
use qiskit_circuit::imports::QI_OPERATOR;
use qiskit_circuit::instruction::{Instruction, Parameters};
use qiskit_circuit::object_registry::ObjectRegistry;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PyOpKind, STANDARD_GATE_SIZE, StandardGate,
};
use qiskit_circuit::{Clbit, Qubit};
use qiskit_quantum_info::unitary_compose;
use thiserror::Error;

use crate::QiskitError;
use crate::gate_metrics;
use crate::standard_gates_commutations;

#[derive(Error, Debug)]
pub enum CommutationError {
    #[error("Einsum error: {0}")]
    EinsumError(&'static str),
    #[error("First instruction must have at most as many qubits as the second instruction")]
    FirstInstructionTooLarge,
    #[error("Invalid hash value: NaN or inf")]
    HashingNaN,
    #[error("Invalid hash type: parameterized")]
    HashingParameter,
}

impl From<CommutationError> for PyErr {
    fn from(value: CommutationError) -> Self {
        match value {
            // For backward compatibility we keep these two errors as QiskitErrors
            CommutationError::HashingParameter | CommutationError::FirstInstructionTooLarge => {
                QiskitError::new_err(value.to_string())
            }
            _ => PyRuntimeError::new_err(value.to_string()),
        }
    }
}

const fn build_supported_ops() -> [bool; STANDARD_GATE_SIZE] {
    let mut lut = [false; STANDARD_GATE_SIZE];
    lut[StandardGate::RXX as usize] = true;
    lut[StandardGate::RYY as usize] = true;
    lut[StandardGate::RZZ as usize] = true;
    lut[StandardGate::RZX as usize] = true;
    lut[StandardGate::CRX as usize] = true;
    lut[StandardGate::CRY as usize] = true;
    lut[StandardGate::CRZ as usize] = true;
    lut[StandardGate::H as usize] = true;
    lut[StandardGate::X as usize] = true;
    lut[StandardGate::Y as usize] = true;
    lut[StandardGate::Z as usize] = true;
    lut[StandardGate::SX as usize] = true;
    lut[StandardGate::SXdg as usize] = true;
    lut[StandardGate::T as usize] = true;
    lut[StandardGate::Tdg as usize] = true;
    lut[StandardGate::S as usize] = true;
    lut[StandardGate::Sdg as usize] = true;
    lut[StandardGate::CX as usize] = true;
    lut[StandardGate::CY as usize] = true;
    lut[StandardGate::CZ as usize] = true;
    lut[StandardGate::Swap as usize] = true;
    lut[StandardGate::ISwap as usize] = true;
    lut[StandardGate::ECR as usize] = true;
    lut[StandardGate::CCX as usize] = true;
    lut[StandardGate::CCZ as usize] = true;
    lut[StandardGate::CS as usize] = true;
    lut[StandardGate::CSdg as usize] = true;
    lut[StandardGate::CSX as usize] = true;
    lut[StandardGate::I as usize] = true;
    lut[StandardGate::GlobalPhase as usize] = true;
    lut[StandardGate::CSwap as usize] = true;
    lut
}

static SUPPORTED_OP: [bool; STANDARD_GATE_SIZE] = build_supported_ops();

// Map rotation gates to their generators (or to ``None`` if we cannot currently efficiently
// represent the generator in Rust and store the commutation relation in the commutation dictionary).
const fn build_supported_rotations() -> [Option<Option<StandardGate>>; STANDARD_GATE_SIZE] {
    let mut lut = [None; STANDARD_GATE_SIZE];
    lut[StandardGate::RX as usize] = Some(Some(StandardGate::X));
    lut[StandardGate::RY as usize] = Some(Some(StandardGate::Y));
    lut[StandardGate::RZ as usize] = Some(Some(StandardGate::Z));
    lut[StandardGate::Phase as usize] = Some(Some(StandardGate::Z));
    lut[StandardGate::U1 as usize] = Some(Some(StandardGate::Z));
    lut[StandardGate::CPhase as usize] = Some(Some(StandardGate::CZ));
    // CRXGate, CRYGate, CRZGate, RXXGate, RYYGate, RZXGate, and RZZGate are supported by the commutation dictionary
    lut[StandardGate::CRX as usize] = Some(None);
    lut[StandardGate::CRY as usize] = Some(None);
    lut[StandardGate::CRZ as usize] = Some(None);
    lut[StandardGate::RXX as usize] = Some(None);
    lut[StandardGate::RYY as usize] = Some(None);
    lut[StandardGate::RZX as usize] = Some(None);
    lut[StandardGate::RZZ as usize] = Some(None);
    lut
}

static SUPPORTED_ROTATIONS: [Option<Option<StandardGate>>; STANDARD_GATE_SIZE] =
    build_supported_rotations();

/// Try extracting the operator from a Python `PauliGate`. Upon failure, `None` is returned.
///
/// # Safety
///
/// Behavior is undefined if ``qubits`` contains a qubit index larger equal to ``num_qubits``,
/// or if the number of Paulis in ``pauli_gate`` does not match the number of indices in ``qubits``.
fn try_extract_op_from_pauli_gate(
    pauli_gate: &OperationRef,
    qubits: &[Qubit],
    num_qubits: u32,
) -> Option<SparseObservable> {
    let OperationRef::PyCustom(gate) = pauli_gate else {
        // Gate is called "pauli" but is not a Python gate
        return None;
    };

    // Call the internal _extract_sparse_observable method. If the gate doesn't have this,
    // it means the user passed a Python gate called "pauli" which is not actually a PauliGate.
    // In that case we return None.
    Python::attach(|py| {
        let py_obs = gate
            .ob
            .call_method0(py, intern!(py, "_extract_sparse_observable"))
            .ok()? // Return None if the method didn't exist
            .cast_bound::<PySparseObservable>(py)
            .expect("Failed casting to PySparseObservable")
            .borrow();
        let local = py_obs.as_inner().expect("Failed to read");

        let out = SparseObservable::identity(num_qubits);
        Some(out.compose_map(&local, |i| qubits[i as usize].0))
    })
}

fn try_extract_op_from_pauli_evo(
    operation: &OperationRef,
    qubits: &[Qubit],
    num_qubits: u32,
) -> Option<SparseObservable> {
    let OperationRef::PyCustom(gate) = operation else {
        // Gate is called "PauliEvo" but is not a Python gate
        return None;
    };

    Python::attach(|py| {
        let py_obs = gate
            .ob
            .call_method0(py, intern!(py, "_extract_sparse_observable"))
            .ok()? // Return None if the method didn't exist
            .cast_bound::<PySparseObservable>(py)
            .expect("Failed casting to PySparseObservable")
            .borrow();
        let local = py_obs.as_inner().expect("Failed to read");

        let out = SparseObservable::identity(num_qubits);
        Some(out.compose_map(&local, |i| qubits[i as usize].0))
    })
}

fn xz_to_observable(x: &[bool], z: &[bool]) -> SparseObservable {
    let mut indices = Vec::new();
    let mut bit_terms = Vec::new();
    for (i, (x_i, z_i)) in x.iter().zip(z.iter()).enumerate() {
        // The only failure case possible here is the identity, because of how we're
        // constructing the value to convert.
        let Ok(term) = ::bytemuck::checked::try_cast(((*x_i as u8) << 1) | (*z_i as u8)) else {
            continue;
        };
        indices.push(i as u32);
        bit_terms.push(term);
    }

    // The sign of the PPM doesn't matter for commutation checking so we just use 1
    let coeffs = vec![Complex64::new(1., 0.)];
    let boundaries = vec![0, bit_terms.len()];

    let num_qubits = x.len() as u32;
    // SAFETY: We know the data is consistent since we constructed it manually, and that indices are
    // sorted in ascending order. Hence, calling new_unchecked is safe.
    unsafe { SparseObservable::new_unchecked(num_qubits, coeffs, bit_terms, indices, boundaries) }
}

fn try_extract_op_from_ppm(
    operation: &OperationRef,
    qubits: &[Qubit],
    num_qubits: u32,
) -> Option<SparseObservable> {
    let OperationRef::PauliProductMeasurement(ppm) = operation else {
        // Gate is called "pauli_product_measurement" but is not actually a PPM...
        return None;
    };
    let local = xz_to_observable(&ppm.x, &ppm.z);
    let out = SparseObservable::identity(num_qubits);
    Some(out.compose_map(&local, |i| qubits[i as usize].0))
}

fn try_extract_op_from_ppr(
    operation: &OperationRef,
    qubits: &[Qubit],
    num_qubits: u32,
) -> Option<SparseObservable> {
    let OperationRef::PauliProductRotation(rotation) = operation else {
        // Gate is called "pauli_product_rotation" but is not actually the right type
        return None;
    };
    let local = xz_to_observable(&rotation.x, &rotation.z);
    let out = SparseObservable::identity(num_qubits);
    Some(out.compose_map(&local, |i| qubits[i as usize].0))
}

/// Attempt to extract generator of a Pauli-based gate in the form of a sparse observable.
fn try_sparse_observable_generator_for_pauli_based(
    operation: &OperationRef,
    qubits: &[Qubit],
    num_qubits: u32,
) -> Option<SparseObservable> {
    match operation.name() {
        "pauli" => try_extract_op_from_pauli_gate(operation, qubits, num_qubits),
        "PauliEvolution" => try_extract_op_from_pauli_evo(operation, qubits, num_qubits),
        "pauli_product_measurement" => try_extract_op_from_ppm(operation, qubits, num_qubits),
        "pauli_product_rotation" => try_extract_op_from_ppr(operation, qubits, num_qubits),
        _ => None,
    }
}

/// Attemp to extract generator of a standard gate in the form of a sparse observable.
fn try_sparse_observable_generator_for_standard_gate(
    operation: &OperationRef,
    params: &[Param],
    qubits: &[Qubit],
    num_qubits: u32,
) -> Option<SparseObservable> {
    if let OperationRef::StandardGate(gate) = operation {
        if let Some(local) = standard_gate_exponent(*gate, params) {
            let out = SparseObservable::identity(num_qubits);
            return Some(out.compose_map(&local, |i| qubits[i as usize].0));
        }
    }
    None
}

/// Attempt to extract generator of a Pauli-based gate in the form of a single Pauli.
/// When successful, return the generator in the (Z, X) form.
pub fn try_pauli_generator_for_pauli_based<'a>(
    operation: &'a OperationRef,
) -> Option<(&'a Vec<bool>, &'a Vec<bool>)> {
    match operation {
        OperationRef::PauliProductRotation(ppr) => Some((&ppr.z, &ppr.x)),
        OperationRef::PauliProductMeasurement(ppm) => Some((&ppm.z, &ppm.x)),
        _ => None,
    }
}

fn get_bits_from_py<T>(
    py_bits1: &Bound<'_, PyTuple>,
    py_bits2: &Bound<'_, PyTuple>,
) -> PyResult<(Vec<T>, Vec<T>)>
where
    T: From<u32> + Copy + Debug,
    u32: From<T>,
{
    // Using `PyObjectAsKey` here is a total hack, but this is a short-term workaround before a
    // larger refactor of the commutation checker.
    let mut registry: ObjectRegistry<T, PyObjectAsKey> = ObjectRegistry::new();

    for bit in py_bits1.iter().chain(py_bits2.iter()) {
        registry.add_allow_existing(bit.into())?;
    }

    Ok((
        registry
            .map_objects(py_bits1.iter().map(|bit| bit.into()))?
            .collect(),
        registry
            .map_objects(py_bits2.iter().map(|bit| bit.into()))?
            .collect(),
    ))
}

/// This is the internal structure for the Python CommutationChecker class
/// It handles the actual commutation checking, and library lookups. It's
/// not meant to be a public facing Python object though and only used
/// internally by the Python class.
#[pyclass(module = "qiskit._accelerate.commutation_checker")]
pub struct CommutationChecker {
    library: CommutationLibrary,
    #[pyo3(get)]
    gates: Option<HashSet<String>>,
}

#[pymethods]
impl CommutationChecker {
    #[pyo3(signature = (standard_gate_commutations=None, gates=None))]
    #[new]
    fn py_new(
        standard_gate_commutations: Option<Bound<PyAny>>,
        gates: Option<HashSet<String>>,
    ) -> Self {
        let library = CommutationLibrary::py_new(standard_gate_commutations);
        CommutationChecker::new(Some(library), gates)
    }

    #[pyo3(signature=(op1, op2, max_num_qubits=None, approximation_degree=1., matrix_max_num_qubits=3))]
    fn commute_nodes(
        &self,
        py: Python,
        op1: &DAGOpNode,
        op2: &DAGOpNode,
        max_num_qubits: Option<u32>,
        approximation_degree: f64,
        matrix_max_num_qubits: u32,
    ) -> PyResult<bool> {
        let (qargs1, qargs2) = get_bits_from_py::<Qubit>(
            op1.instruction.qubits.bind(py),
            op2.instruction.qubits.bind(py),
        )?;
        let (cargs1, cargs2) = get_bits_from_py::<Clbit>(
            op1.instruction.clbits.bind(py),
            op2.instruction.clbits.bind(py),
        )?;

        Ok(self.commute(
            &op1.instruction.operation.view(),
            op1.instruction.parameters(),
            &qargs1,
            &cargs1,
            &op2.instruction.operation.view(),
            op2.instruction.parameters(),
            &qargs2,
            &cargs2,
            max_num_qubits,
            matrix_max_num_qubits,
            approximation_degree,
        )?)
    }

    #[pyo3(name="commute", signature=(op1, qargs1, cargs1, op2, qargs2, cargs2, max_num_qubits=None, approximation_degree=1., matrix_max_num_qubits=3))]
    #[allow(clippy::too_many_arguments)]
    fn py_commute(
        &mut self,
        op1: OperationFromPython<Py<PyAny>>,
        qargs1: &Bound<'_, PyTuple>,
        cargs1: &Bound<'_, PyTuple>,
        op2: OperationFromPython<Py<PyAny>>,
        qargs2: &Bound<'_, PyTuple>,
        cargs2: &Bound<'_, PyTuple>,
        max_num_qubits: Option<u32>,
        approximation_degree: f64,
        matrix_max_num_qubits: u32,
    ) -> PyResult<bool> {
        let (qargs1, qargs2) = get_bits_from_py::<Qubit>(qargs1, qargs2)?;
        let (cargs1, cargs2) = get_bits_from_py::<Clbit>(cargs1, cargs2)?;

        Ok(self.commute(
            &op1.operation.view(),
            op1.parameters(),
            &qargs1,
            &cargs1,
            &op2.operation.view(),
            op2.parameters(),
            &qargs2,
            &cargs2,
            max_num_qubits,
            matrix_max_num_qubits,
            approximation_degree,
        )?)
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new(py);
        out_dict.set_item("library", self.library.library.clone().into_pyobject(py)?)?;
        out_dict.set_item("gates", self.gates.clone())?;
        Ok(out_dict.unbind())
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let dict_state = state.cast_bound::<PyDict>(py)?;
        self.library = CommutationLibrary {
            library: dict_state.get_item("library")?.unwrap().extract()?,
        };
        self.gates = dict_state.get_item("gates")?.unwrap().extract()?;
        Ok(())
    }
}

impl CommutationChecker {
    /// Create a new commutation checker.
    ///
    /// # Arguments
    ///
    /// - `library`: An optional existing [CommutationLibrary].
    /// - `gates`: An optional set of gates (by name) to check commutations for. If `None`,
    ///   commutation is checked for all gates.
    pub fn new(library: Option<CommutationLibrary>, gates: Option<HashSet<String>>) -> Self {
        // Initialize sets before they are used in the commutation checker
        CommutationChecker {
            library: library.unwrap_or(CommutationLibrary { library: None }),
            gates,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn commute<T>(
        &self,
        op1: &OperationRef,
        params1: Option<&Parameters<T>>,
        qargs1: &[Qubit],
        cargs1: &[Clbit],
        op2: &OperationRef,
        params2: Option<&Parameters<T>>,
        qargs2: &[Qubit],
        cargs2: &[Clbit],
        max_num_qubits: Option<u32>,
        matrix_max_num_qubits: u32,
        approximation_degree: f64,
    ) -> Result<bool, CommutationError> {
        if let Some(gates) = &self.gates {
            if !gates.contains(op1.name()) || !gates.contains(op2.name()) {
                return Ok(false);
            }
        }

        // If the average gate infidelity is below this tolerance, they commute. The tolerance
        // is set to max(1e-12, 1 - approximation_degree), to account for roundoffs and for
        // consistency with other places in Qiskit.
        let tol = 1e-12_f64.max(1. - approximation_degree);

        let params1 = params1
            .map(|p| match p {
                Parameters::Params(p) => p.as_slice(),
                _ => &[],
            })
            .unwrap_or_default();
        let params2 = params2
            .map(|p| match p {
                Parameters::Params(p) => p.as_slice(),
                _ => &[],
            })
            .unwrap_or_default();

        // if we have rotation gates, we attempt to map them to their generators, for example
        // RX -> X or CPhase -> CZ
        let (op1_gate, params1, trivial1) = map_rotation(op1, params1, tol);
        if trivial1 {
            return Ok(true);
        }
        let op1 = if let Some(gate) = op1_gate {
            &OperationRef::StandardGate(gate)
        } else {
            op1
        };
        let (op2_gate, params2, trivial2) = map_rotation(op2, params2, tol);
        if trivial2 {
            return Ok(true);
        }
        let op2 = if let Some(gate) = op2_gate {
            &OperationRef::StandardGate(gate)
        } else {
            op2
        };

        let precheck_status = commutation_precheck(
            op1,
            qargs1,
            cargs1,
            params1,
            op2,
            qargs2,
            cargs2,
            params2,
            max_num_qubits,
        );

        // Check if we can already return after the pre-check, or if further evaluations are required
        match precheck_status {
            PrecheckStatus::Commuting => return Ok(true),
            PrecheckStatus::NonCommuting => return Ok(false),
            _ => (),
        };

        // Special handling for commutativity of two pauli product measurements in the case they write to
        // the same classical bit. In this case, it's generally incorrect to interchange them, so we only
        // do this if they have the same pauli generators.
        if let (
            OperationRef::PauliProductMeasurement(ppm1),
            OperationRef::PauliProductMeasurement(ppm2),
        ) = (op1, op2)
        {
            if cargs1 == cargs2 {
                if (ppm1.neg != ppm2.neg) || (qargs1.len() != qargs2.len()) {
                    return Ok(false);
                }
                // Mark all qubit indices in qargs1
                let qarg_map = qargs1
                    .iter()
                    .enumerate()
                    .map(|(i, q)| (q.index(), i))
                    .collect::<HashMap<_, _>>();
                // Check that every qubit index in qargs2 is marked and has the same z,x values.
                for (j, &q) in qargs2.iter().enumerate() {
                    let Some(&i) = qarg_map.get(&q.index()) else {
                        return Ok(false);
                    };
                    if i == usize::MAX {
                        return Ok(false);
                    }
                    if ppm1.z[i] != ppm2.z[j] || ppm1.x[i] != ppm2.x[j] {
                        return Ok(false);
                    }
                }
                return Ok(true);
            }
        }

        // Sort the arguments, such that `op2` always is the larger one.
        let reversed = (op1.num_qubits(), op1.name().len(), op1.name())
            > (op2.num_qubits(), op2.name().len(), op2.name());

        let (op1, op2, params1, params2, qargs1, qargs2) = if reversed {
            (op2, op1, params2, params1, qargs2, qargs1)
        } else {
            (op1, op2, params1, params2, qargs1, qargs2)
        };

        // Handle commutations using Pauli-based generators.
        if let (Some((z1, x1)), Some((z2, x2))) = (
            try_pauli_generator_for_pauli_based(op1),
            try_pauli_generator_for_pauli_based(op2),
        ) {
            let qarg_map = qargs1
                .iter()
                .enumerate()
                .map(|(i, q)| (q.index(), i))
                .collect::<HashMap<_, _>>();
            let mut parity = false;
            for (j, &q) in qargs2.iter().enumerate() {
                if let Some(&i) = qarg_map.get(&q.index()) {
                    parity ^= (x1[i] && z2[j]) ^ (z1[i] && x2[j]);
                }
            }
            return Ok(!parity);
        }

        // Handle commutations between Pauli-based gates among themselves, and with standard gates
        // TODO Support trivial commutations of standard gates with identities in the Paulis
        let size = qargs1.iter().chain(qargs2.iter()).max().unwrap().0 + 1;
        let maybe_pauli1 = try_sparse_observable_generator_for_pauli_based(op1, qargs1, size);
        let maybe_pauli2 = try_sparse_observable_generator_for_pauli_based(op2, qargs2, size);

        match (maybe_pauli1, maybe_pauli2) {
            (None, None) => (), // No gate is Pauli-based, continue
            (None, Some(pauli2)) => {
                if let Some(pauli1) =
                    try_sparse_observable_generator_for_standard_gate(op1, params1, qargs1, size)
                {
                    return Ok(pauli1.commutes(&pauli2, tol));
                }
            }
            (Some(pauli1), None) => {
                if let Some(pauli2) =
                    try_sparse_observable_generator_for_standard_gate(op2, params2, qargs2, size)
                {
                    return Ok(pauli1.commutes(&pauli2, tol));
                }
            }
            (Some(pauli1), Some(pauli2)) => return Ok(pauli1.commutes(&pauli2, tol)),
        };

        // Now there are no more parameterized gates that are allowed
        if matches!(precheck_status, PrecheckStatus::Parameterized) {
            return Ok(false);
        }

        // Query commutation library
        let relative_placement = get_relative_placement(qargs1, qargs2);
        if let Some(is_commuting) =
            self.library
                .check_commutation_entries(op1, op2, &relative_placement)
        {
            return Ok(is_commuting);
        }

        if qargs2.len() > matrix_max_num_qubits as usize {
            return Ok(false);
        }

        // Perform matrix multiplication to determine commutation
        let is_commuting = self.commute_matmul(op1, params1, qargs1, op2, params2, qargs2, tol)?;

        Ok(is_commuting)
    }

    #[allow(clippy::too_many_arguments)]
    fn commute_matmul(
        &self,
        first_op: &OperationRef,
        first_params: &[Param],
        first_qargs: &[Qubit],
        second_op: &OperationRef,
        second_params: &[Param],
        second_qargs: &[Qubit],
        tol: f64,
    ) -> Result<bool, CommutationError> {
        // Compute relative positioning of qargs of the second gate to the first gate.
        // Since the qargs come out the same BitData, we already know there are no accidental
        // bit-duplications, but this code additionally maps first_qargs to [0..n] and then
        // computes second_qargs relative to that. For example, it performs the mappings
        //  (first_qargs, second_qargs) = ( [1, 2], [0, 2] ) --> ( [0, 1], [2, 1] )
        //  (first_qargs, second_qargs) = ( [1, 2, 0], [0, 3, 4] ) --> ( [0, 1, 2], [2, 3, 4] )
        // This re-shuffling is done to compute the correct kronecker product later.
        let mut qarg: HashMap<&Qubit, Qubit> = HashMap::from_iter(
            first_qargs
                .iter()
                .enumerate()
                .map(|(i, q)| (q, Qubit::new(i))),
        );
        let mut num_qubits = first_qargs.len() as u32;
        for q in second_qargs {
            if !qarg.contains_key(q) {
                qarg.insert(q, Qubit(num_qubits));
                num_qubits += 1;
            }
        }

        let first_indices = Vec::from_iter(0..first_qargs.len());
        let second_indices = second_qargs
            .iter()
            .map(|q| qarg[q].index())
            .collect::<Vec<_>>();

        if first_indices.len() > second_indices.len() {
            return Err(CommutationError::FirstInstructionTooLarge);
        };
        let first_mat = match try_matrix_with_definition(first_op, first_params, None) {
            Some(matrix) => matrix,
            None => return Ok(false),
        };

        let second_mat = match try_matrix_with_definition(second_op, second_params, None) {
            Some(matrix) => matrix,
            None => return Ok(false),
        };

        // TODO Optimize this bit to avoid unnecessary Kronecker products:
        //  1. We currently sort the operations for the cache by operation size, putting the
        //     *smaller* operation first: (smaller op, larger op)
        //  2. This code here expands the first op to match the second -- hence we always
        //     match the operator sizes.
        // This whole extension logic could be avoided since we know the second one is larger.
        let extra_qarg2 = num_qubits - first_indices.len() as u32;
        let first_mat = if extra_qarg2 > 0 {
            let id_op = Array2::<Complex64>::eye(usize::pow(2, extra_qarg2));
            kron(&id_op, &first_mat)
        } else {
            first_mat
        };

        // the 1 qubit case cannot happen, since that would already have been captured
        // by the previous if clause; first_qarg == second_qarg (if they overlap they must
        // be the same)
        let op12 = match unitary_compose::compose(
            &first_mat.view(),
            &second_mat.view(),
            &second_indices,
            false,
        ) {
            Ok(matrix) => matrix,
            Err(e) => return Err(CommutationError::EinsumError(e)),
        };
        let op21 = match unitary_compose::compose(
            &first_mat.view(),
            &second_mat.view(),
            &second_indices,
            true,
        ) {
            Ok(matrix) => matrix,
            Err(e) => return Err(CommutationError::EinsumError(e)),
        };
        let (fid, phase) = gate_metrics::gate_fidelity(&op12.view(), &op21.view());

        // we consider the gates as commuting if the process fidelity of
        // AB (BA)^\dagger is approximately the identity and there is no global phase difference
        // let dim = op12.ncols() as f64;
        // let matrix_tol = tol * dim.powi(2);
        let matrix_tol = tol;
        Ok(phase.abs() <= tol && (1.0 - fid).abs() <= matrix_tol)
    }
}

/// A pre-check status.
///
/// Used to differentiate between the reasons why gates might not commute, which allow
/// the commutation checker to handle cases individually. E.g. a ``PauliEvolutionGate`` can
/// still be checked even if the gate is parameterized.
#[derive(Debug, Clone)]
enum PrecheckStatus {
    Commuting,     // gates commute for sure
    NonCommuting,  // gates do not commute for sure
    Parameterized, // a gate is parameterized and is likely disqualified from being checked
    Unknown,       // the precheck is inconclusive
}

#[allow(clippy::too_many_arguments)]
fn commutation_precheck(
    op1: &OperationRef,
    qargs1: &[Qubit],
    cargs1: &[Clbit],
    params1: &[Param],
    op2: &OperationRef,
    qargs2: &[Qubit],
    cargs2: &[Clbit],
    params2: &[Param],
    max_num_qubits: Option<u32>,
) -> PrecheckStatus {
    if matches!(op1, OperationRef::ControlFlow { .. })
        || matches!(op2, OperationRef::ControlFlow { .. })
    {
        return PrecheckStatus::NonCommuting;
    }

    // assuming the number of involved qubits to be small, this might be faster than set operations
    if !qargs1.iter().any(|e| qargs2.contains(e)) && !cargs1.iter().any(|e| cargs2.contains(e)) {
        return PrecheckStatus::Commuting;
    }

    if let Some(limit) = max_num_qubits {
        if qargs1.len() > limit as usize || qargs2.len() > limit as usize {
            return PrecheckStatus::NonCommuting;
        }
    }

    if let OperationRef::StandardGate(gate_1) = op1 {
        if let OperationRef::StandardGate(gate_2) = op2 {
            if SUPPORTED_OP[(*gate_1) as usize] && SUPPORTED_OP[(*gate_2) as usize] {
                return PrecheckStatus::Unknown;
            }
        }
    }

    if [op1, op2].iter().any(|op| match op {
        OperationRef::StandardInstruction(_) => true,
        OperationRef::PyCustom(inst) => inst.kind == PyOpKind::Instruction,
        _ => false,
    }) {
        return PrecheckStatus::NonCommuting;
    }

    if is_parameterized(params1) || is_parameterized(params2) {
        return PrecheckStatus::Parameterized;
    }

    PrecheckStatus::Unknown
}

/// Returns matrix representation of the specified operation.
///
/// For custom python gates:
/// - The matrix is constructed from the gate's definition when the gate does
///   not provide a direct `matrix` method.
/// - To prevent generating excessively large matrices, use
///   `matrix_from_definition_max_qubits` to set an upper limit on the number
///   of qubits for which the construction is applied.
pub fn try_matrix_with_definition(
    operation: &OperationRef,
    params: &[Param],
    matrix_from_definition_max_qubits: Option<u32>,
) -> Option<Array2<Complex64>> {
    match operation {
        OperationRef::StandardGate(gate) => gate.matrix(params),
        OperationRef::PauliProductRotation(ppr) => ppr.matrix(),
        OperationRef::Unitary(unitary) => unitary.matrix(),
        OperationRef::PyCustom(inst) => Python::attach(|py| -> Option<_> {
            if let Some(matrix) = inst.matrix() {
                return Some(matrix);
            }
            if matrix_from_definition_max_qubits.is_some_and(|max| max < inst.qubits)
                || inst.kind == PyOpKind::Instruction
            {
                return None;
            }
            Some(
                QI_OPERATOR
                    .get_bound(py)
                    .call1((inst.ob.clone_ref(py),))
                    .ok()?
                    .getattr(intern!(py, "data"))
                    .ok()?
                    .extract::<PyReadonlyArray2<Complex64>>()
                    .ok()?
                    .as_array()
                    .to_owned(),
            )
        }),
        _ => None,
    }
}

fn is_parameterized(params: &[Param]) -> bool {
    params
        .iter()
        .any(|x| matches!(x, Param::ParameterExpression(_)))
}

/// Check if a given operation can be mapped onto a generator.
///
/// If ``op`` is in the ``SUPPORTED_ROTATIONS`` hashmap, it is a rotation and we
///   (1) check whether the rotation is so small (modulo pi) that we assume it is the
///       identity and it commutes trivially with every other operation
///   (2) otherwise, we check whether a generator of the rotation is given (e.g. X for RX)
///       and we return the generator
///
/// Returns (operation, parameters, commutes_trivially).
fn map_rotation<'a>(
    op: &'a OperationRef<'a>,
    params: &'a [Param],
    tol: f64,
) -> (Option<StandardGate>, &'a [Param], bool) {
    if let OperationRef::StandardGate(gate) = op {
        if let Some(generator) = SUPPORTED_ROTATIONS[(*gate) as usize] {
            // If the rotation angle is below the tolerance, the gate is assumed to
            // commute with everything, and we simply return the operation with the flag that
            // it commutes trivially.
            if let Param::Float(angle) = params[0] {
                let (tr_over_dim, dim) = gate_metrics::rotation_trace_and_dim(*gate, angle)
                    .expect("All rotation should be covered at this point");
                let gate_fidelity = tr_over_dim.abs().powi(2);
                let process_fidelity = (dim * gate_fidelity + 1.) / (dim + 1.);
                if (1. - process_fidelity).abs() <= tol {
                    return (Some(*gate), params, true);
                };
            };

            // Otherwise we need to cover two cases -- either a generator is given, in which case
            // we return it, or we don't have a generator yet, but we know we have the operation
            // stored in the commutation library. For example, RXX does not have a generator in Rust
            // yet (PauliGate is not in Rust currently), but it is stored in the library, so we
            // can strip the parameters and just return the gate.
            if let Some(gate) = generator {
                return (Some(gate), &[], false);
            };
        }
    }
    (None, params, false)
}

fn get_relative_placement(
    first_qargs: &[Qubit],
    second_qargs: &[Qubit],
) -> SmallVec<[Option<Qubit>; 2]> {
    let mut qubits_g2: HashMap<&Qubit, Qubit> = HashMap::with_capacity(second_qargs.len());
    second_qargs
        .iter()
        .enumerate()
        .for_each(|(i_g1, q_g1)| unsafe {
            qubits_g2.insert_unique_unchecked(q_g1, Qubit::new(i_g1));
        });

    first_qargs
        .iter()
        .map(|q_g0| qubits_g2.get(q_g0).copied())
        .collect()
}

#[derive(Clone, Debug)]
#[pyclass(skip_from_py_object)]
pub struct CommutationLibrary {
    pub library: Option<HashMap<(String, String), CommutationLibraryEntry>>,
}

impl CommutationLibrary {
    pub(crate) fn with_capacity(size: usize) -> Self {
        CommutationLibrary {
            library: Some(HashMap::with_capacity(size)),
        }
    }

    pub(crate) fn add_entry(&mut self, key: [&str; 2], value: CommutationLibraryEntry) {
        if let Some(library) = &mut self.library {
            library.insert((key[0].to_string(), key[1].to_string()), value);
        } else {
            let mut library = HashMap::new();
            library.insert((key[0].to_string(), key[1].to_string()), value);
            self.library = Some(library);
        }
    }

    fn check_commutation_entries(
        &self,
        first_op: &OperationRef,
        second_op: &OperationRef,
        relative_placement: &SmallVec<[Option<Qubit>; 2]>,
    ) -> Option<bool> {
        if let Some(library) = &self.library {
            match library.get(&(first_op.name().to_string(), second_op.name().to_string())) {
                Some(CommutationLibraryEntry::Commutes(b)) => Some(*b),
                Some(CommutationLibraryEntry::QubitMapping(qm)) => {
                    qm.get(relative_placement).copied()
                }
                _ => None,
            }
        } else {
            None
        }
    }
}

#[pymethods]
impl CommutationLibrary {
    #[new]
    #[pyo3(signature=(py_any=None))]
    fn py_new(py_any: Option<Bound<PyAny>>) -> Self {
        match py_any {
            Some(pyob) => CommutationLibrary {
                library: pyob
                    .extract::<HashMap<(String, String), CommutationLibraryEntry>>()
                    .ok(),
            },
            None => CommutationLibrary {
                library: Some(HashMap::new()),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum CommutationLibraryEntry {
    Commutes(bool),
    QubitMapping(HashMap<SmallVec<[Option<Qubit>; 2]>, bool>),
}

impl<'py> IntoPyObject<'py> for CommutationLibraryEntry {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_out = match self {
            CommutationLibraryEntry::Commutes(b) => b.into_pyobject(py)?.into_bound().into_any(),
            CommutationLibraryEntry::QubitMapping(qm) => {
                let out = PyDict::new(py);
                for (k, v) in qm {
                    let key = PyTuple::new(py, k.iter().map(|q| q.map(|t| t.0)))?;
                    let value = PyBool::new(py, v);
                    out.set_item(key, value)?;
                }
                out.into_any()
            }
        };
        Ok(py_out)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for CommutationLibraryEntry {
    type Error = PyErr;

    fn extract(b: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(b) = b.extract::<bool>() {
            return Ok(CommutationLibraryEntry::Commutes(b));
        }
        let dict = b.cast::<PyDict>()?;
        let mut ret = hashbrown::HashMap::with_capacity(dict.len());
        for (k, v) in &*dict {
            let raw_key: SmallVec<[Option<u32>; 2]> = k.extract()?;
            let v: bool = v.extract()?;
            let key = raw_key.into_iter().map(|key| key.map(Qubit)).collect();
            ret.insert(key, v);
        }
        Ok(CommutationLibraryEntry::QubitMapping(ret))
    }
}

#[pyfunction]
pub fn get_standard_commutation_checker() -> CommutationChecker {
    let library = standard_gates_commutations::get_commutation_library();
    CommutationChecker {
        library,
        gates: None,
    }
}

pub fn commutation_checker(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CommutationLibrary>()?;
    m.add_class::<CommutationChecker>()?;
    m.add_wrapped(wrap_pyfunction!(get_standard_commutation_checker))?;
    Ok(())
}
