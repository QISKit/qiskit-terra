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

use std::cmp::Ordering;
use std::str::FromStr;
use std::sync::OnceLock;

use hashbrown::HashSet;
use num_complex::Complex64;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use ndarray::prelude::*;
use rayon::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::{Operation, OperationRef, Param};
use qiskit_util::getenv_use_multiple_threads;

use crate::target::{Target, TargetOperation};
use qiskit_circuit::instruction::Instruction;
use qiskit_circuit::{PhysicalQubit, gate_matrix};
use qiskit_synthesis::euler_one_qubit_decomposer::{
    EULER_BASES, EULER_BASIS_NAMES, EulerBasis, EulerBasisSet, OneQubitGateSequence,
    angles_from_unitary, generate_circuit,
};

fn compute_error_term_from_target(gate: &str, target: &Target, qubit: PhysicalQubit) -> f64 {
    1. - target.get_error(gate, &[qubit]).unwrap_or(0.)
}

/// Resynthesize a 1q unitary in each Euler basis of the set and return the candidate with the
/// lowest error for the target (falling back to the lowest gate count when there is no target),
/// together with its computed error.  Unlike selecting on gate count alone, this accounts for
/// otherwise-tied candidates having very different error rates on the target.
fn resynthesize_run_min_error(
    operator: ArrayView2<Complex64>,
    target_basis_set: &EulerBasisSet,
    qubit: PhysicalQubit,
    target: Option<&Target>,
) -> Option<(OneQubitGateSequence, (f64, usize))> {
    target_basis_set
        .get_bases()
        .map(|target_basis| {
            let [theta, phi, lam, phase] = angles_from_unitary(operator, target_basis);
            let sequence =
                generate_circuit(&target_basis, theta, phi, lam, phase, true, None).unwrap();
            let error = compute_error_from_target_one_qubit_sequence(&sequence, qubit, target);
            (sequence, error)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
}

fn compute_error_from_target_one_qubit_sequence(
    circuit: &OneQubitGateSequence,
    qubit: PhysicalQubit,
    target: Option<&Target>,
) -> (f64, usize) {
    match target {
        Some(target) => {
            let num_gates = circuit.gates.len();
            let gate_fidelities: f64 = circuit
                .gates
                .iter()
                .map(|gate| compute_error_term_from_target(gate.0.name(), target, qubit))
                .product();
            (1. - gate_fidelities, num_gates)
        }
        None => (circuit.gates.len() as f64, circuit.gates.len()),
    }
}

#[derive(Clone, Debug)]
enum BasisGatesPerQubit {
    All,
    Gates(HashSet<String>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for BasisGatesPerQubit {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(value) = obj.extract::<&str>() {
            if value == "All" {
                Ok(Self::All)
            } else {
                Err(PyValueError::new_err(format!(
                    "Value {obj:?} is not a valid value for a basis gates per qubit"
                )))
            }
        } else if let Ok(gates) = obj.extract::<HashSet<String>>() {
            Ok(Self::Gates(gates))
        } else {
            Err(PyValueError::new_err(format!(
                "Value {obj:?} is not a valid value for a basis gates per qubit"
            )))
        }
    }
}

impl<'py> IntoPyObject<'py> for BasisGatesPerQubit {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::All => Ok("All".into_pyobject(py)?.into_any()),
            Self::Gates(gates) => Ok(gates.clone().into_pyobject(py)?.into_any()),
        }
    }
}

#[pyclass(module = "qiskit._accelerate.optimize_1q_gates_decomposition")]
pub struct Optimize1qGatesDecompositionState {
    target_basis_per_qubit: Vec<OnceLock<EulerBasisSet>>,
    basis_gates_per_qubit: Vec<OnceLock<BasisGatesPerQubit>>,
    global: bool,
}

type Optimize1qGatesDecompositionStatePickle = (
    Vec<Option<Vec<String>>>,
    Vec<Option<BasisGatesPerQubit>>,
    bool,
);

#[pymethods]
impl Optimize1qGatesDecompositionState {
    fn __getstate__(&self) -> Optimize1qGatesDecompositionStatePickle {
        (
            self.target_basis_per_qubit
                .iter()
                .map(|raw_set| {
                    raw_set.get().map(|basis_set| {
                        basis_set
                            .get_bases()
                            .map(|x| x.as_str().to_string())
                            .collect()
                    })
                })
                .collect(),
            self.basis_gates_per_qubit
                .iter()
                .map(|x| x.get().cloned())
                .collect(),
            self.global,
        )
    }

    fn __setstate__(&mut self, state: Optimize1qGatesDecompositionStatePickle) {
        self.target_basis_per_qubit = state
            .0
            .into_iter()
            .map(|set| match set {
                Some(set) => {
                    let mut euler_set = EulerBasisSet::new();
                    for basis_str in set {
                        euler_set.add_basis(EulerBasis::from_str(basis_str.as_str()).unwrap());
                    }
                    let out_set = OnceLock::new();
                    out_set.set(euler_set).expect("Qubit already populated");
                    out_set
                }
                None => OnceLock::new(),
            })
            .collect();
        self.basis_gates_per_qubit = state
            .1
            .into_iter()
            .map(|x| {
                let locked = OnceLock::new();
                if let Some(x) = x {
                    locked.set(x).expect("basis gates not set");
                }
                locked
            })
            .collect();
        self.global = state.2;
    }

    #[new]
    #[pyo3(signature = (num_qubits=0))]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            target_basis_per_qubit: vec![OnceLock::new(); num_qubits.max(1)],
            basis_gates_per_qubit: vec![OnceLock::new(); num_qubits.max(1)],
            global: num_qubits == 0,
        }
    }
}

impl Optimize1qGatesDecompositionState {
    fn get_basis_set(
        &self,
        qubit: PhysicalQubit,
        target: Option<&Target>,
        basis_gates: Option<&HashSet<String>>,
    ) -> Option<HashSet<String>> {
        if self.global {
            match target {
                Some(target) => Some(
                    target
                        .operations()
                        .filter_map(|op| {
                            if op.operation.num_qubits() == 1 {
                                Some(op.operation.name().to_string())
                            } else {
                                None
                            }
                        })
                        .collect::<HashSet<_>>(),
                ),
                None => basis_gates.cloned(),
            }
        } else {
            let Some(target) = target else {
                unreachable!("Target must be true to be non-global");
            };
            Some(
                target
                    .operation_names_for_qargs(&[qubit])
                    .expect("Qubit has operations")
                    .into_iter()
                    .map(|x| x.to_string())
                    .filter(|gate_name| {
                        let target_op = target.operation_from_name(gate_name).unwrap();
                        let TargetOperation::Normal(gate) = target_op else {
                            return false;
                        };
                        if let OperationRef::StandardGate(_) = gate.operation.view() {
                            // For standard gates check that the target entry accepts any
                            // params and if so then we can use the gate in the pass
                            // else filter the operation since arbitrary angles are not
                            // supported
                            gate.params_view()
                                .iter()
                                .all(|x| matches!(x, Param::ParameterExpression(_)))
                        } else {
                            // For all other gates pass it through
                            true
                        }
                    })
                    .collect(),
            )
        }
    }

    fn get_euler_basis_set(
        &self,
        qubit: PhysicalQubit,
        target: Option<&Target>,
        global_decomposers: Option<&Vec<String>>,
    ) -> EulerBasisSet {
        if self.global {
            let mut target_basis_set = EulerBasisSet::new();
            if target.is_some() {
                EULER_BASES
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, gates)| {
                        if !gates.iter().all(|gate| {
                            let BasisGatesPerQubit::Gates(gates) =
                                self.basis_gates_per_qubit[0].get().unwrap()
                            else {
                                unreachable!("The target path always provides a hash set");
                            };
                            gates.contains(*gate)
                        }) {
                            return None;
                        }
                        let basis = EULER_BASIS_NAMES[idx];
                        Some(basis)
                    })
                    .for_each(|basis| target_basis_set.add_basis(basis));
                if target_basis_set.basis_supported(EulerBasis::U3)
                    && target_basis_set.basis_supported(EulerBasis::U321)
                {
                    target_basis_set.remove(EulerBasis::U3);
                }
                if target_basis_set.basis_supported(EulerBasis::ZSX)
                    && target_basis_set.basis_supported(EulerBasis::ZSXX)
                {
                    target_basis_set.remove(EulerBasis::ZSX);
                }
            } else {
                match &global_decomposers {
                    Some(bases) => {
                        for basis in bases.iter() {
                            target_basis_set.add_basis(
                                EulerBasis::__new__(basis).expect("a valid basis string"),
                            )
                        }
                    }
                    None => match self.basis_gates_per_qubit[0].get().unwrap() {
                        BasisGatesPerQubit::Gates(gates) => EULER_BASES
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, basis_gates)| {
                                if !gates
                                    .iter()
                                    .all(|gate| basis_gates.as_ref().contains(&gate.as_str()))
                                {
                                    return None;
                                }
                                let basis = EULER_BASIS_NAMES[idx];
                                Some(basis)
                            })
                            .for_each(|basis| target_basis_set.add_basis(basis)),
                        BasisGatesPerQubit::All => target_basis_set.support_all(),
                    },
                }
                if target_basis_set.basis_supported(EulerBasis::U3)
                    && target_basis_set.basis_supported(EulerBasis::U321)
                {
                    target_basis_set.remove(EulerBasis::U3);
                }
                if target_basis_set.basis_supported(EulerBasis::ZSX)
                    && target_basis_set.basis_supported(EulerBasis::ZSXX)
                {
                    target_basis_set.remove(EulerBasis::ZSX);
                }
            }
            target_basis_set
        } else {
            let mut target_basis_set = EulerBasisSet::new();
            EULER_BASES
                .iter()
                .enumerate()
                .filter_map(|(idx, gates)| {
                    if !gates.iter().all(|gate| {
                        let BasisGatesPerQubit::Gates(gates) =
                            self.basis_gates_per_qubit[qubit.index()].get().unwrap()
                        else {
                            unreachable!("the target path always provides a hash set")
                        };
                        gates.contains(*gate)
                    }) {
                        return None;
                    }
                    let basis = EULER_BASIS_NAMES[idx];
                    Some(basis)
                })
                .for_each(|basis| target_basis_set.add_basis(basis));
            if target_basis_set.basis_supported(EulerBasis::U3)
                && target_basis_set.basis_supported(EulerBasis::U321)
            {
                target_basis_set.remove(EulerBasis::U3);
            }
            if target_basis_set.basis_supported(EulerBasis::ZSX)
                && target_basis_set.basis_supported(EulerBasis::ZSXX)
            {
                target_basis_set.remove(EulerBasis::ZSX);
            }
            target_basis_set
        }
    }
}

#[pyfunction]
#[pyo3(name = "optimize_1q_gates_decomposition", signature = (dag, state, *, target=None, basis_gates=None, global_decomposers=None))]
pub fn run_optimize_1q_gates_decomposition(
    dag: &mut DAGCircuit,
    state: &Optimize1qGatesDecompositionState,
    target: Option<&Target>,
    basis_gates: Option<HashSet<String>>,
    global_decomposers: Option<Vec<String>>,
) -> PyResult<()> {
    let runs: Vec<Vec<NodeIndex>> = dag.collect_1q_runs().unwrap().collect();
    let process_run =
        |raw_run: &[NodeIndex], dag: &DAGCircuit| -> PyResult<Option<OneQubitGateSequence>> {
            let mut error = match target {
                Some(_) => 1.,
                None => raw_run.len() as f64,
            };
            let qubit: PhysicalQubit = if let NodeType::Operation(inst) = &dag[raw_run[0]] {
                PhysicalQubit::new(dag.get_qargs(inst.qubits)[0].0)
            } else {
                unreachable!("nodes in runs will always be op nodes")
            };
            let basis_gates = if state.global {
                state.basis_gates_per_qubit[0].get_or_init(|| {
                    match state.get_basis_set(PhysicalQubit::new(0), target, basis_gates.as_ref()) {
                        Some(x) => BasisGatesPerQubit::Gates(x),
                        None => BasisGatesPerQubit::All,
                    }
                })
            } else {
                state.basis_gates_per_qubit[qubit.index()].get_or_init(|| {
                    match state.get_basis_set(qubit, target, basis_gates.as_ref()) {
                        Some(x) => BasisGatesPerQubit::Gates(x),
                        None => BasisGatesPerQubit::All,
                    }
                })
            };
            let target_basis_set = if state.global {
                state.target_basis_per_qubit[0].get_or_init(|| {
                    state.get_euler_basis_set(
                        PhysicalQubit::new(0),
                        target,
                        global_decomposers.as_ref(),
                    )
                })
            } else {
                state.target_basis_per_qubit[qubit.index()].get_or_init(|| {
                    state.get_euler_basis_set(qubit, target, global_decomposers.as_ref())
                })
            };
            let operator = raw_run
                .iter()
                .map(|node_index| {
                    let node = &dag[*node_index];
                    if let NodeType::Operation(inst) = node {
                        if let Some(target) = target {
                            error *= compute_error_term_from_target(inst.op.name(), target, qubit);
                        }
                        inst.try_matrix_as_static_1q()
                            .expect("collect_1q_runs only collects gates that can produce a matrix")
                    } else {
                        unreachable!("Can only have op nodes here")
                    }
                })
                .fold(gate_matrix::ONE_QUBIT_IDENTITY, |mut operator, node| {
                    matmul_1q_with_slice(&mut operator, &node);
                    operator
                });

            let old_error = if target.is_some() {
                (1. - error, raw_run.len())
            } else {
                (error, raw_run.len())
            };
            let Some((sequence, new_error)) =
                resynthesize_run_min_error(aview2(&operator), target_basis_set, qubit, target)
            else {
                return Ok(None);
            };

            let mut outside_basis = false;
            if let BasisGatesPerQubit::Gates(basis) = basis_gates {
                for node in raw_run {
                    if let NodeType::Operation(inst) = &dag[*node] {
                        if !basis.contains(inst.op.name()) {
                            outside_basis = true;
                            break;
                        }
                    }
                }
            } else {
                outside_basis = false;
            }
            if outside_basis
                || new_error < old_error
                || new_error.0.abs() < 1e-9 && old_error.0.abs() >= 1e-9
            {
                Ok(Some(sequence))
            } else {
                Ok(None)
            }
        };
    if getenv_use_multiple_threads() {
        let sequences = runs
            .par_iter()
            .map(|raw_run| process_run(raw_run, dag))
            .collect::<PyResult<Vec<_>>>()?;
        runs.into_iter()
            .zip(sequences)
            .filter_map(|(raw_run, sequence)| sequence.map(|x| (raw_run, x)))
            .try_for_each(|(raw_run, sequence)| -> PyResult<()> {
                for gate in sequence.gates {
                    dag.insert_1q_on_incoming_qubit((gate.0, &gate.1), raw_run[0]);
                }
                dag.add_global_phase(&Param::Float(sequence.global_phase))?;
                dag.remove_1q_sequence(&raw_run);
                Ok(())
            })?;
    } else {
        for raw_run in runs {
            let sequence = process_run(&raw_run, dag)?;
            if let Some(sequence) = sequence {
                for gate in sequence.gates {
                    dag.insert_1q_on_incoming_qubit((gate.0, &gate.1), raw_run[0]);
                }
                dag.add_global_phase(&Param::Float(sequence.global_phase))?;
                dag.remove_1q_sequence(&raw_run);
            }
        }
    }
    Ok(())
}

#[inline(always)]
pub(crate) fn matmul_1q(operator: &mut [[Complex64; 2]; 2], other: Array2<Complex64>) {
    *operator = [
        [
            other[[0, 0]] * operator[0][0] + other[[0, 1]] * operator[1][0],
            other[[0, 0]] * operator[0][1] + other[[0, 1]] * operator[1][1],
        ],
        [
            other[[1, 0]] * operator[0][0] + other[[1, 1]] * operator[1][0],
            other[[1, 0]] * operator[0][1] + other[[1, 1]] * operator[1][1],
        ],
    ];
}

/// Computes matrix product ``other * operator`` and stores the result within ``operator``.
#[inline(always)]
pub fn matmul_1q_with_slice(operator: &mut [[Complex64; 2]; 2], other: &[[Complex64; 2]; 2]) {
    *operator = [
        [
            other[0][0] * operator[0][0] + other[0][1] * operator[1][0],
            other[0][0] * operator[0][1] + other[0][1] * operator[1][1],
        ],
        [
            other[1][0] * operator[0][0] + other[1][1] * operator[1][0],
            other[1][0] * operator[0][1] + other[1][1] * operator[1][1],
        ],
    ];
}

pub fn optimize_1q_gates_decomposition_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_optimize_1q_gates_decomposition))?;
    m.add_class::<Optimize1qGatesDecompositionState>()?;
    Ok(())
}
