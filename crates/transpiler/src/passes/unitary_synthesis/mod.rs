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

mod decomposers;

use hashbrown::{HashMap, HashSet};
use nalgebra::Matrix2;
use ndarray::prelude::*;
use num_complex::Complex64;
use qiskit_util::IndexSet;
use rayon::prelude::*;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::visit::NodeIndexable;
use std::cell::RefCell;
use std::hash;
use thread_local::ThreadLocal;

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::{intern, wrap_pyfunction};

pub(crate) use self::decomposers::Direction2q;

use self::decomposers::{Decomposer2q, DecomposerCache, FlipDirection};
use crate::QiskitError;
use crate::target::Target;
use qiskit_circuit::bit::QuantumRegister;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder, NodeType};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{Operation, OperationRef, Param, PythonOperation, StandardGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{BlocksMode, PhysicalQubit, Qubit, VarsMode};
use qiskit_synthesis::euler_one_qubit_decomposer::{
    OneQubitGateSequence, unitary_to_gate_sequence_inner,
};
use qiskit_synthesis::qsd::quantum_shannon_decomposition;
use qiskit_synthesis::two_qubit_decompose::TwoQubitGateSequence;
use qiskit_util::getenv_use_multiple_threads;

#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

/// Threshold in number of gates to synthesize to run the pass multithreaded
/// vs serially. At smaller gate counts the overhead of multithreading outweighs the speedup
const PARALLEL_THRESHOLD: usize = 50;

/// The fidelity of the 2q basis gate used in a decomposer.
///
/// This is "normalised" in the sense that the value is guaranteed (when constructed safely) to be
/// between 0.0 and 1.0 inclusive, and the allowed floating-point values are standardised so we only
/// use one canonical representation of each (i.e. no negative zero).  These conditions make it safe
/// to use with total equality and hashing, which lets it be put in a hashmap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedFidelity(f64);
impl NormalizedFidelity {
    #[inline]
    pub fn new(val: f64) -> Option<Self> {
        // The `abs` is normalising signed zero.
        (0.0..=1.0).contains(&val).then(|| Self(val.abs()))
    }
    /// Get the value.  This is guaranteed to be finite, sign positive and in `[0.0, 1.0]`.
    #[inline]
    pub fn get(&self) -> f64 {
        self.0
    }
}
// `impl Eq` is safe for this float-derived quantity because we only permit the range `[0.0, 1.0]`
// and forbid negative zero.
impl Eq for NormalizedFidelity {}
impl hash::Hash for NormalizedFidelity {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        // This is safe because we're in the range `[0.0, 1.0]` and normalised out negative zero.
        self.0.to_le_bytes().hash(state)
    }
}

/// The strategy to use for approximate synthesis (or `Exact` for exact synthesis).
///
/// This is used internally in the Rust code, because the Python form `float | None` has been
/// frequently accidentally misinterpreted (there, `None` does not mean exact, and `1.0` has a
/// very different meaning to `1.0 - 1ULP`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Approximation {
    /// Do perfect synthesis, regardless of reported gate errors.
    #[default]
    Exact,
    /// Scale the reported gate fidelity by the given amount.
    ScaleFidelity(f64),
}
impl Approximation {
    /// Convert from the Python-space representation of `approximation_degree`.
    pub fn from_py_approximation_degree(val: Option<f64>) -> Self {
        match val {
            // ... yeah, I don't know why we've historically done this either.
            None => Self::ScaleFidelity(1.0),
            Some(1.0) => Self::Exact,
            Some(val) => Self::ScaleFidelity(val),
        }
    }

    /// Get the fidelity target that should be used for a given gate error.
    ///
    /// Returns `Err` with the value of an out-of-bounds requested fidelity.
    pub fn synthesis_fidelity(&self, gate_error: f64) -> Result<NormalizedFidelity, f64> {
        match self {
            Self::Exact => Ok(NormalizedFidelity::new(1.0).expect("1.0 should be in bounds")),
            Self::ScaleFidelity(scale) => {
                let fidelity = scale * (1.0 - gate_error);
                NormalizedFidelity::new(fidelity).ok_or(fidelity)
            }
        }
    }
}

/// A borrowed view onto the hardware constraint.
#[derive(Clone, Copy, Debug)]
pub enum QpuConstraint<'a> {
    Target(&'a Target),
    Loose {
        basis_gates: &'a IndexSet<&'a str>,
        coupling: &'a HashSet<[PhysicalQubit; 2]>,
    },
}
impl QpuConstraint<'_> {
    /// Which kind of coupling constraint is this?
    fn kind(&self) -> QpuConstraintKind {
        match self {
            Self::Target(_) => QpuConstraintKind::Target,
            Self::Loose { .. } => QpuConstraintKind::Loose,
        }
    }
}
impl<'a> From<&'a Target> for QpuConstraint<'a> {
    fn from(value: &'a Target) -> Self {
        Self::Target(value)
    }
}
/// Which kind of hardware constraint is in use?
///
/// This doesn't actually store the constraint or a reference to it, it's just a type homomoprhic to
/// the discriminant of [QpuConstraint] while discarding its lifetime, so it can be a hashed key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum QpuConstraintKind {
    Target,
    Loose,
}

/// The settings to apply to 2q pulse-aware decomposers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UsePulseOptimizer {
    /// Require the use of a pulse-optimal decomposer.  This may produce compilation errors if the
    /// hardware target uses basis gates that the pulse-optimal decomposer cannot handle.
    Require,
    /// Never use a pulse-optimizing decomposer.
    Forbid,
    /// Use the pulse-optimising decomposer if it's available for a given hardware link, and
    /// produces better results.
    IfBetter,
}
impl UsePulseOptimizer {
    /// Convert the legacy Python-space `pulse_optimize` argument into the Rust-native explicit form.
    pub fn from_py_pulse_optimize(val: Option<bool>) -> Self {
        match val {
            None => Self::IfBetter,
            Some(false) => Self::Forbid,
            Some(true) => Self::Require,
        }
    }
    /// Convert back to the legacy Python-space `pulse_optimize` argument.
    pub fn to_py_pulse_optimize(self) -> Option<bool> {
        match self {
            Self::IfBetter => None,
            Self::Forbid => Some(false),
            Self::Require => Some(true),
        }
    }
}
/// Which directions to allow for a 2q gate in a decomposition.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DecompositionDirection2q {
    /// The 2q gate can be used in either direction, regardless of whether the hardware is
    /// bidirectional or not.
    Any,
    /// If only one direction is hardware-native, then use that direction.  If both are valid,
    /// choose the one that minimizes some error heuristic (allowing either direction if they are
    /// both the same).
    BestValid,
    /// The same as [BestValid], except that an error is raised if both directions are valid and
    /// have the same (or no) error heuristics.
    UniquelyBestValid,
}
impl DecompositionDirection2q {
    pub fn from_py_natural_direction(val: Option<bool>) -> Self {
        match val {
            None => Self::BestValid,
            Some(false) => Self::Any,
            Some(true) => Self::UniquelyBestValid,
        }
    }
}

/// Configuration options for the default unitary synthesis plugin.
///
/// This implements `Default`, which is a convenient constructor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitarySynthesisConfig {
    /// Whether to do approximate synthesis.
    pub approximation: Approximation,
    pub use_pulse_optimizer: UsePulseOptimizer,
    pub decomposition_direction_2q: DecompositionDirection2q,
    /// Whether to allow use of Python-space decomposers.
    pub run_python_decomposers: bool,
}
impl Default for UnitarySynthesisConfig {
    fn default() -> Self {
        Self {
            approximation: Approximation::Exact,
            use_pulse_optimizer: UsePulseOptimizer::IfBetter,
            decomposition_direction_2q: DecompositionDirection2q::BestValid,
            run_python_decomposers: false,
        }
    }
}

/// State of a unitary synthesis run.
#[derive(Clone, Debug, Default)]
pub struct UnitarySynthesisState {
    config: UnitarySynthesisConfig,
    cache: DecomposerCache,
}
impl UnitarySynthesisState {
    pub fn new(config: UnitarySynthesisConfig) -> Self {
        Self {
            config,
            cache: Default::default(),
        }
    }
}

/// The matcher for the set of standard gates that the TwoQubitControlledUDecomposer
/// supports
macro_rules! PARAM_SET {
    // Make sure that this is kept in sync with the static array PARAM_GATES below
    () => {
        StandardGate::RXX
            | StandardGate::RYY
            | StandardGate::RZZ
            | StandardGate::RZX
            | StandardGate::CRX
            | StandardGate::CRY
            | StandardGate::CRZ
            | StandardGate::CPhase
    };
}

// Make sure that this is kept in sync with the macro PARAM_SET above
static PARAM_SET_BASIS_GATES: [StandardGate; 8] = [
    StandardGate::RXX,
    StandardGate::RYY,
    StandardGate::RZZ,
    StandardGate::RZX,
    StandardGate::CRX,
    StandardGate::CRY,
    StandardGate::CRZ,
    StandardGate::CPhase,
];

/// The matcher for the set of standard gates that the TwoQubitBasisDecomposer
/// supports
macro_rules! TWO_QUBIT_BASIS_SET {
    // Make sure that this is kept in sync with the static array TWO_QUBIT_BASIS_SET_GATES below
    () => {
        StandardGate::CX
            | StandardGate::CY
            | StandardGate::CZ
            | StandardGate::CH
            | StandardGate::DCX
            | StandardGate::ISwap
            | StandardGate::ECR
    };
}

// Make sure this is kept in sync with the macro TWO_QUBIT_BASIS_SET above
static TWO_QUBIT_BASIS_SET_GATES: [StandardGate; 7] = [
    StandardGate::CX,
    StandardGate::CY,
    StandardGate::CZ,
    StandardGate::CH,
    StandardGate::DCX,
    StandardGate::ISwap,
    StandardGate::ECR,
];

pub(crate) use {PARAM_SET, TWO_QUBIT_BASIS_SET};

/// The different output types from a synthesis.
///
/// The different synthesis types have different outputs which make different
/// assumptions about the output constraints based on the synthesis being
/// performed (e.g. one qubit synthesis doesn't need to specify qubit indices).
enum SynthesisOutput {
    /// A global phase update
    Phase(f64),
    /// A two qubit synthesis result
    TwoQ(TwoQSynthesisResult<f64>),
    /// A single qubit synthesis result from the Euler one qubit synthesis module
    OneQ(OneQubitGateSequence),
    /// A single qubit synthesis result from the Solovay Kitaev module
    OneQSK(CircuitData),
    /// A multiqubit synthesis output from the Quantum Shannon Decomposition
    Qsd(CircuitData),
}

/// Run synthesis analysis in parallel using multithreading and collect results into a mapping from node
/// indices to synthesis result.
fn parallel_synthesis(
    dag: &DAGCircuit,
    synth_gates: &HashSet<String>,
    qubit_indices: &[PhysicalQubit],
    constraint: QpuConstraint,
    state: &mut UnitarySynthesisState,
    min_qubits: usize,
) -> PyResult<HashMap<NodeIndex, SynthesisOutput>> {
    let thread_local_states = ThreadLocal::new();
    (0..dag.dag().node_bound())
        .into_par_iter()
        .filter_map(|idx| -> Option<PyResult<(NodeIndex, SynthesisOutput)>> {
            let index = NodeIndex::new(idx);
            let synthesis_state: &RefCell<UnitarySynthesisState> =
                thread_local_states.get_or(|| RefCell::new(state.clone()));
            let Some(NodeType::Operation(inst)) = dag.dag().node_weight(index) else {
                return None;
            };
            if !(synth_gates.contains(inst.op.name()) && inst.op.num_qubits() >= min_qubits as u32)
            {
                return None;
            }
            let unitary = inst.try_cow_array()?;
            let result = synthesize_matrix(
                unitary,
                qubit_indices,
                dag.get_qargs(inst.qubits),
                &mut synthesis_state.borrow_mut(),
                constraint,
            )
            .transpose()?;
            match result {
                Ok(result) => Some(Ok((index, result))),
                Err(e) => Some(Err(e)),
            }
        })
        .collect::<PyResult<_>>()
}

/// Return a new DAG that takes a mapping as returned by [`parallel_synthesis`] and replaces those
/// nodes in the mapping with the synthesis result.
fn apply_synthesis(
    dag: &DAGCircuit,
    mut node_replace_map: HashMap<NodeIndex, SynthesisOutput>,
    qubit_indices: &[PhysicalQubit],
    synth_gates: &HashSet<String>,
    min_qubits: usize,
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
) -> PyResult<DAGCircuit> {
    let rebuilder_callback =
        |out: &mut DAGCircuitBuilder, inst: &PackedInstruction, node_index: NodeIndex| {
            let Some(cf) = dag.try_view_control_flow(inst) else {
                // Handle regular instructions - this path is where we end up most of the time.
                if let Some(sequence) = node_replace_map.remove(&node_index) {
                    apply_matrix_result_onto(sequence, out, dag.get_qargs(inst.qubits))?;
                } else {
                    // No synthesis was necessary, so reinstate the operation.
                    out.push_back(inst.clone())?;
                }
                return Ok(());
            };
            // If we make it here, we've got control flow and have to set ourselves up to recurse.
            //
            // TODO: Running this in parallel will potentially deadlock if the target decomposition gate is
            // defined in python and there are unitaries in a control flow block inside the recursion run
            // internally.
            let blocks = cf
                .blocks()
                .into_iter()
                .map(|orig_block| {
                    let qubit_indices = dag
                        .get_qargs(inst.qubits)
                        .iter()
                        .map(|q| qubit_indices[q.index()])
                        .collect::<Vec<_>>();
                    run_unitary_synthesis(
                        orig_block,
                        synth_gates,
                        min_qubits,
                        &qubit_indices,
                        state,
                        constraint,
                        true,
                    )
                    .map(|block| {
                        if let Some(block) = block {
                            out.add_block(block)
                        } else {
                            out.add_block(orig_block.clone())
                        }
                    })
                })
                .collect::<PyResult<_>>()?;
            out.push_back(PackedInstruction::from_control_flow(
                inst.op.control_flow().clone(),
                blocks,
                inst.qubits,
                inst.clbits,
                inst.label.as_deref().cloned(),
            ))?;
            Ok(())
        };
    dag.rebuild_dag_with(rebuilder_callback, VarsMode::Alike, BlocksMode::Drop)
}

/// Iterate over `DAGCircuit` to perform unitary synthesis.  For each eligible gate: find
/// decomposers, select the synthesis method with the highest fidelity score and apply
/// decompositions. The available methods are:
///
/// * 1q synthesis:
///     * OneQubitEulerDecomposer
///     * SolovayKitaevSynthesis
/// * 2q synthesis:
///     * TwoQubitBasisDecomposer
///     * TwoQubitControlledUDecomposer
///     * XXDecomposer (Python, only if target is provided)
/// * 3q+ synthesis: QuantumShannonDecomposer
///
/// # Args
///
/// - `dag`: The DAGCircuit to run the pass on
/// - `synth_gates`: The set of gate names to synthesize the unitary matrix of.
/// - `min_qubits`: The minimum number of qubits to require gates in synth_gates to act on to
///   synthesize. For example, if `min_qubits=3` `UnitaryGate` instances acting on 1 or 2 qubits
///   will not be synthesized (assuming `"unitary"` is in `synth_gates`).
/// - `qubit_indices`: The mapping to apply when evaluating qubits in the circuit in the target.
///   This is typically only used when recursing into control flow blocks.
/// - `state`: The pass state, used to cache the decomposers to use for qubits
/// - `constraint`: The qpu constraints that apply to synthesis
/// - `force_serial`: Whether or not force running the pass serially even if
///   otherwise the pass would be multithreaded.
pub fn run_unitary_synthesis(
    dag: &DAGCircuit,
    synth_gates: &HashSet<String>,
    min_qubits: usize,
    qubit_indices: &[PhysicalQubit],
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
    force_serial: bool,
) -> PyResult<Option<DAGCircuit>> {
    let op_counts = dag.count_ops(true)?;
    let gate_counts: usize = synth_gates
        .iter()
        .map(|name| op_counts.get(name).unwrap_or(&0usize))
        .copied()
        .sum();
    if gate_counts == 0 {
        return Ok(None);
    }
    let run_in_parallel = getenv_use_multiple_threads();
    if run_in_parallel && gate_counts > PARALLEL_THRESHOLD && !force_serial {
        let node_replace_map = parallel_synthesis(
            dag,
            synth_gates,
            qubit_indices,
            constraint,
            state,
            min_qubits,
        )?;
        Ok(Some(apply_synthesis(
            dag,
            node_replace_map,
            qubit_indices,
            synth_gates,
            min_qubits,
            state,
            constraint,
        )?))
    } else {
        Ok(Some(serial_run_unitary_synthesis(
            dag,
            synth_gates,
            min_qubits,
            qubit_indices,
            state,
            constraint,
        )?))
    }
}

/// Run unitary synthesis serially in a single loop over the dag.
///
/// This is as opposed to the combination of [`parallel_synthesis`] and [`apply_synthesis`] that
/// will iterate over the nodes twice, but one will done in parallel using multithreading.
fn serial_run_unitary_synthesis(
    dag: &DAGCircuit,
    synth_gates: &HashSet<String>,
    min_qubits: usize,
    qubit_indices: &[PhysicalQubit],
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
) -> PyResult<DAGCircuit> {
    // This method is the actual distribution logic of unitary synthesis, but there are several
    // paths through it that return `Ok(false)`, meaning "no error and no synthesis needed", so the
    // caller is responsible for propagating the old instruction through to wherever is necessary.
    let synthesize_onto = |out: &mut DAGCircuitBuilder,
                           state: &mut UnitarySynthesisState,
                           inst: &PackedInstruction|
     -> PyResult<bool> {
        if !(synth_gates.contains(inst.op.name()) && inst.op.num_qubits() >= min_qubits as u32) {
            return Ok(false);
        }
        let Some(unitary) = inst.try_cow_array() else {
            return Ok(false);
        };
        synthesize_matrix_onto(
            out,
            unitary,
            qubit_indices,
            dag.get_qargs(inst.qubits),
            state,
            constraint,
        )
    };

    let rebuilder_callback =
        |out: &mut DAGCircuitBuilder, inst: &PackedInstruction, _node_index: NodeIndex| {
            let Some(cf) = dag.try_view_control_flow(inst) else {
                // Handle regular instructions - this path is where we end up most of the time.
                if !synthesize_onto(out, state, inst)? {
                    // No synthesis was necessary, so reinstate the operation.
                    out.push_back(inst.clone())?;
                }
                return Ok(());
            };
            // If we make it here, we've got control flow and have to set ourselves up to recurse.
            let blocks = cf
                .blocks()
                .into_iter()
                .map(|orig_block| {
                    let qubit_indices = dag
                        .get_qargs(inst.qubits)
                        .iter()
                        .map(|q| qubit_indices[q.index()])
                        .collect::<Vec<_>>();
                    run_unitary_synthesis(
                        orig_block,
                        synth_gates,
                        min_qubits,
                        &qubit_indices,
                        state,
                        constraint,
                        true,
                    )
                    .map(|block| {
                        if let Some(block) = block {
                            out.add_block(block)
                        } else {
                            out.add_block(orig_block.clone())
                        }
                    })
                })
                .collect::<PyResult<_>>()?;
            out.push_back(PackedInstruction::from_control_flow(
                inst.op.control_flow().clone(),
                blocks,
                inst.qubits,
                inst.clbits,
                inst.label.as_deref().cloned(),
            ))?;
            Ok(())
        };
    dag.rebuild_dag_with(rebuilder_callback, VarsMode::Alike, BlocksMode::Drop)
}

/// Synthesize a unitary matrix and return the synthesis output
fn synthesize_matrix(
    unitary: CowArray<Complex64, Ix2>,
    qubits_phys: &[PhysicalQubit],
    qubits_local: &[Qubit],
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
) -> PyResult<Option<SynthesisOutput>> {
    let num_qubits = qubits_local.len();
    debug_assert_eq!(unitary.shape(), &[1 << num_qubits, 1 << num_qubits]);
    match *qubits_local {
        [] => Ok(Some(SynthesisOutput::Phase(unitary[[0, 0]].arg()))),
        [q_virt] => {
            let q_phys = qubits_phys[q_virt.index()];
            synthesize_1q_matrix(unitary.view(), q_phys, state, constraint)
        }
        [q1_virt, q2_virt] => {
            let q_virt = [q1_virt, q2_virt];
            let q_phys = q_virt.map(|q| qubits_phys[q.index()]);
            let result =
                synthesize_2q_matrix(unitary, q_phys, state, constraint, fidelity_2q_sequence)?;
            Ok(result.map(SynthesisOutput::TwoQ))
        }
        _ => {
            if let QpuConstraint::Loose { basis_gates, .. } = constraint {
                if basis_gates.is_empty() {
                    return Ok(None);
                }
            }
            Ok(Some(SynthesisOutput::Qsd(quantum_shannon_decomposition(
                unitary.view(),
                None,
                None,
                None,
                None,
            )?)))
        }
    }
}

fn apply_matrix_result_onto(
    result: SynthesisOutput,
    out: &mut DAGCircuitBuilder,
    qubits_virt: &[Qubit],
) -> PyResult<()> {
    match result {
        SynthesisOutput::Phase(phase) => {
            out.add_global_phase(&Param::Float(phase))?;
        }
        SynthesisOutput::OneQ(sequence) => {
            let qubits = out.insert_qargs(qubits_virt);
            let clbits = out.insert_cargs(&[]);
            out.add_global_phase(&Param::Float(sequence.global_phase))?;
            for (gate, params) in sequence.gates {
                let params = (!params.is_empty()).then(|| {
                    Box::new(Parameters::Params(
                        params.iter().map(|p| Param::Float(*p)).collect(),
                    ))
                });
                out.push_back(PackedInstruction {
                    op: gate.into(),
                    qubits,
                    clbits,
                    params,
                    label: None,
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                })?;
            }
        }
        SynthesisOutput::OneQSK(circuit) => {
            let qubits = out.insert_qargs(qubits_virt);
            let clbits = out.insert_cargs(&[]);
            out.add_global_phase(circuit.global_phase())?;
            for inst in circuit.into_data_iter() {
                out.push_back(PackedInstruction {
                    qubits,
                    clbits,
                    ..inst
                })?;
            }
        }
        SynthesisOutput::TwoQ(result) => {
            // ... now apply the best sequence.
            let order = result.dir.as_indices();
            let out_qargs = [
                qubits_virt[order[0] as usize],
                qubits_virt[order[1] as usize],
            ];
            let qubit_keys = [
                out.insert_qargs(&[out_qargs[0]]),
                out.insert_qargs(&[out_qargs[1]]),
                out.insert_qargs(&[out_qargs[0], out_qargs[1]]),
                out.insert_qargs(&[out_qargs[1], out_qargs[0]]),
            ];
            out.add_global_phase(&Param::Float(result.sequence.global_phase()))?;
            for (gate, params, qubits) in result.sequence.gates() {
                let qubits = match qubits.as_slice() {
                    [0] => qubit_keys[0],
                    [1] => qubit_keys[1],
                    [0, 1] => qubit_keys[2],
                    [1, 0] => qubit_keys[3],
                    _ => panic!(
                        "internal logic error: decomposed sequence contained unexpected qargs"
                    ),
                };
                let op = match gate.view() {
                    OperationRef::StandardGate(gate) => PackedOperation::from(gate),
                    OperationRef::PyCustom(py_gate) => Python::attach(|py| -> PyResult<_> {
                        let gate = py_gate.py_copy(py)?;
                        gate.ob.setattr(py, intern!(py, "params"), params)?;
                        Ok(gate.into())
                    })?,
                    _ => panic!("internal logic error: decomposed sequence contains a non-gate"),
                };
                let params = (!params.is_empty()).then(|| {
                    Box::new(Parameters::Params(
                        params.iter().copied().map(Param::Float).collect(),
                    ))
                });
                out.push_back(PackedInstruction {
                    op,
                    qubits,
                    clbits: Default::default(),
                    params,
                    label: None,
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                })?;
            }
        }
        SynthesisOutput::Qsd(circuit) => {
            let map = out.merge_qargs(circuit.qargs_interner(), |q| Some(qubits_virt[q.index()]));
            out.add_global_phase(circuit.global_phase())?;
            for inst in circuit.into_data_iter() {
                out.push_back(PackedInstruction {
                    qubits: map[inst.qubits],
                    ..inst
                })?;
            }
        }
    }
    Ok(())
}

/// Synthesise a matrix onto the DAG.
fn synthesize_matrix_onto(
    out: &mut DAGCircuitBuilder,
    unitary: CowArray<Complex64, Ix2>,
    qubits_phys: &[PhysicalQubit],
    qubits_local: &[Qubit],
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
) -> PyResult<bool> {
    let num_qubits = qubits_local.len();
    debug_assert_eq!(unitary.shape(), &[1 << num_qubits, 1 << num_qubits]);
    match *qubits_local {
        [] => {
            out.add_global_phase(&Param::Float(unitary[[0, 0]].arg()))?;
            Ok(true)
        }
        [q_virt] => {
            let q_phys = qubits_phys[q_virt.index()];
            synthesize_1q_matrix_onto(out, unitary.view(), q_phys, q_virt, state, constraint)
        }
        [q1_virt, q2_virt] => {
            let q_virt = [q1_virt, q2_virt];
            let q_phys = q_virt.map(|q| qubits_phys[q.index()]);
            synthesize_2q_matrix_onto(out, unitary, q_phys, q_virt, state, constraint)
        }
        _ => {
            if let QpuConstraint::Loose { basis_gates, .. } = constraint {
                if basis_gates.is_empty() {
                    return Ok(false);
                }
            }
            let circuit = quantum_shannon_decomposition(unitary.view(), None, None, None, None)?;
            let map = out.merge_qargs(circuit.qargs_interner(), |q| Some(qubits_local[q.index()]));
            out.add_global_phase(circuit.global_phase())?;
            for inst in circuit.into_data_iter() {
                out.push_back(PackedInstruction {
                    qubits: map[inst.qubits],
                    ..inst
                })?;
            }
            Ok(true)
        }
    }
}

/// Synthesize a 1q unitary matrix and return the output
fn synthesize_1q_matrix(
    unitary: ArrayView2<Complex64>,
    qubit_phys: PhysicalQubit,
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
) -> PyResult<Option<SynthesisOutput>> {
    if let Some(sk) = state.cache.try_solovay_kitaev(qubit_phys, constraint) {
        let circuit = sk
            .synthesize_matrix(&Matrix2::from_fn(|i, j| unitary[[i, j]]), 5)
            .expect("hardcoded standard gates should not include parametric gates");
        Ok(Some(SynthesisOutput::OneQSK(circuit)))
    } else {
        let sequence = unitary_to_gate_sequence_inner(
            unitary,
            &state.cache.get_euler_1q(qubit_phys, constraint),
            qubit_phys.index(),
            None,
            true,
            None,
        );
        let Some(sequence) = sequence else {
            return Ok(None);
        };
        Ok(Some(SynthesisOutput::OneQ(sequence)))
    }
}

/// Synthesize a 1q unitary matrix and apply the result onto a DAGCircuit
fn synthesize_1q_matrix_onto(
    out: &mut DAGCircuitBuilder,
    unitary: ArrayView2<Complex64>,
    qubit_phys: PhysicalQubit,
    qubit_virt: Qubit,
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
) -> PyResult<bool> {
    // TODO: we possibly want to invert this logic and do Euler synthesis if possible, and SK only
    // if we have to.  If nothing else, it simplifies the logic of `try_solovay_kitaev` - instead of
    // "is this _only_ Clifford+T?" it can become "does this permit a Clifford+T decomposition",
    // which would allow us to remove all the "ignored" non-Clifford+T instructions.
    // If the qubit permits only known discrete-basis instructions, we'll do SK synthesis...
    if let Some(sk) = state.cache.try_solovay_kitaev(qubit_phys, constraint) {
        let circuit = sk
            .synthesize_matrix(&Matrix2::from_fn(|i, j| unitary[[i, j]]), 5)
            .expect("hardcoded standard gates should not include parametric gates");
        let qubits = out.insert_qargs(&[qubit_virt]);
        let clbits = out.insert_cargs(&[]);
        out.add_global_phase(circuit.global_phase())?;
        for inst in circuit.into_data_iter() {
            out.push_back(PackedInstruction {
                qubits,
                clbits,
                ..inst
            })?;
        }
        return Ok(true);
    };
    // ... otherwise, we do continuous Euler-angle synthesis.
    let sequence = unitary_to_gate_sequence_inner(
        unitary,
        &state.cache.get_euler_1q(qubit_phys, constraint),
        qubit_phys.index(),
        None,
        true,
        None,
    );
    let Some(sequence) = sequence else {
        return Ok(false);
    };
    let qubits = out.insert_qargs(&[qubit_virt]);
    let clbits = out.insert_cargs(&[]);
    out.add_global_phase(&Param::Float(sequence.global_phase))?;
    for (gate, params) in sequence.gates {
        let params = (!params.is_empty()).then(|| {
            Box::new(Parameters::Params(
                params.iter().map(|p| Param::Float(*p)).collect(),
            ))
        });
        out.push_back(PackedInstruction {
            op: gate.into(),
            qubits,
            clbits,
            params,
            label: None,
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        })?;
    }
    Ok(true)
}

#[derive(Debug)]
pub struct TwoQSynthesisResult<S> {
    pub sequence: TwoQubitGateSequence,
    pub dir: Direction2q,
    pub score: Option<S>,
}

/// Estimate the fidelity of a synthesized two qubit unitary synthesis output
///
/// The estimated fidelity is computed by taking the error rate from the target
/// and computing the product of the fidelities of each gate in the circuit. In
/// the absence of error rates (either QpuConstraint doesn't have a target or
/// the target contains no error rates) then a fidelity of 1 is returned.
///
/// # Args
/// `dir` - The direction of the synthesis (forward or reverse)
/// `sequence` - The synthesis sequence
/// `constraint` - The qpu constraints used for the synthesis, typically just a Target
/// `qargs_phys` - The qpu qargs the unitary is run on
///
/// # Returns
///
/// The computed fidelity estimate of the synthesis output sequence
#[inline]
pub(crate) fn fidelity_2q_sequence(
    dir: &Direction2q,
    sequence: &TwoQubitGateSequence,
    constraint: &QpuConstraint,
    qargs_phys: [PhysicalQubit; 2],
) -> f64 {
    let QpuConstraint::Target(target) = &constraint else {
        return 1.;
    };
    let order = dir.as_indices();
    let phys = [qargs_phys[order[0] as usize], qargs_phys[order[1] as usize]];
    sequence
        .gates()
        .iter()
        .map(|(op, _, qubits)| {
            let qargs: &[_] = match *qubits.as_slice() {
                [q] => &[phys[q as usize]],
                [q1, q2] => &[phys[q1 as usize], phys[q2 as usize]],
                _ => panic!("sequences should only contain 1q and 2q gates"),
            };
            // TODO: this does not handle the possibility of a 2q decomposer (like the
            // XXDecomposer) using specialised instructions whose operation names do not match
            // their target key.
            1. - target.get_error(op.name(), qargs).unwrap_or(0.)
        })
        .product()
}

/// Synthesize a given two qubit unitary matrix into a gate sequence.
///
/// For overcomplete targets this will run all compatible decomposers and pick the output synthesis
/// with the highest estimated fidelity.
///
/// # Args
/// `unitary` - The unitary to synthesize
/// `qargs_phys` - The physical qubits the unitary is being applied to
/// `state` - The internal state of the pass, this includes the configured synthesizers
/// `constraint` - The qpu constraints used for the synthesis, typically just a Target
/// `fidelity_calculation` - A callable that is called with a two qubit synthesis output sequence and
///   the context around it. It is expected to return a fidelity score to compare that synthesis output
///   against other decomposers. The synthesis output with the maximum score is selected and is
///   what is returned by the `synthesize_2q_matrix`.
///
/// # Returns
///
/// When the return is `Ok` this returns an Option<TwoQSynthesisResult>, when the return is None
/// this indicates that synthesis failed or is not necessary. The `Some` case will return a
/// `TwoQSynthesisResult` representing the "best" output synthesis from the available decomposers
/// determined by `fidelity_calculation`. This function returns an Error if the internal
/// decomposition errors and it returns the directly from the decomposer.
pub(crate) fn synthesize_2q_matrix<F, S>(
    mut unitary: CowArray<Complex64, Ix2>,
    qargs_phys: [PhysicalQubit; 2],
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
    mut fidelity_calculation: F,
) -> PyResult<Option<TwoQSynthesisResult<S>>>
where
    F: FnMut(&Direction2q, &TwoQubitGateSequence, &QpuConstraint, [PhysicalQubit; 2]) -> S,
    S: PartialOrd,
{
    let decomposer_cache = &mut state.cache;
    let config = &state.config;

    let mut cur_direction = Direction2q::Forwards;
    // This closure takes ownership of its two stateful values, `cur_direction` and `unitary`, via
    // the `move`, so that the rest of the function can't accidentally view them in unknown states.
    let mut single_decomposition = move |decomposer: &Decomposer2q, direction: Direction2q| {
        if cur_direction != direction {
            conjugate_with_swaps(unitary.view_mut());
        }
        cur_direction = direction;
        decomposer.decompose(unitary.view())
    };
    let mut sequences = decomposer_cache
        .get_2q(qargs_phys, config, constraint)?
        .map(|(decomposer, flip)| -> PyResult<_> {
            match flip {
                FlipDirection::No => single_decomposition(decomposer, Direction2q::Forwards)
                    .map(|seq| (Direction2q::Forwards, seq)),
                FlipDirection::Yes => single_decomposition(decomposer, Direction2q::Backwards)
                    .map(|seq| (Direction2q::Backwards, seq)),
                FlipDirection::Ensure(dir) => {
                    let normal = single_decomposition(decomposer, Direction2q::Forwards)?;
                    if normal
                        .gates()
                        .iter()
                        .filter_map(|(_, _, qubits)| (qubits.len() == 2).then_some(qubits))
                        .all(|qubits| qubits.as_slice() == dir.as_indices())
                    {
                        Ok((Direction2q::Forwards, normal))
                    } else {
                        // TODO: we assume that if the decomposer flipped when requesting one
                        // direction, then it'll flip when requesting the other too.  This is a
                        // historical assumption, and we'd be better just fixing the decomposers so
                        // they're not flaky with respect to direction.
                        single_decomposition(decomposer, Direction2q::Backwards)
                            .map(|seq| (Direction2q::Backwards, seq))
                    }
                }
            }
        });

    let Some(first) = sequences.next().transpose()? else {
        // TODO: The historical behaviour of this pass is to treat "failed to synthesise" the same
        // way as "synthesis was not necessary" (i.e. propagate the base unitary/gate to the output
        // without any synthesis).  This is probably not an ideal choice.
        //
        // Actually, there were also some convoluted circumstances specifically in the case of only
        // a `Target` being specified where it could instead raise an error, but these were
        // inconsistent; either it should be an error in _all_ circumstances if synthesis fails or
        // in _none_.  It's tricky to recreate the pre-Qiskit-2.4 behaviour bug-for-bug in the new
        // refactor because of how the split between decomposer construction and use works now.
        return Ok(None);
    };

    // We only need to calculate the best score if there's more than one sequence.
    let mut best_fidelity = None;
    let mut best_pair = first;
    for sequence in sequences {
        let sequence = sequence?;
        let prev_fidelity = best_fidelity.unwrap_or_else(|| {
            fidelity_calculation(&best_pair.0, &best_pair.1, &constraint, qargs_phys)
        });
        let this_fidelity = fidelity_calculation(&sequence.0, &sequence.1, &constraint, qargs_phys);
        if this_fidelity > prev_fidelity {
            best_fidelity = Some(this_fidelity);
            best_pair = sequence;
        } else {
            best_fidelity = Some(prev_fidelity);
        }
    }
    Ok(Some(TwoQSynthesisResult {
        sequence: best_pair.1,
        dir: best_pair.0,
        score: best_fidelity,
    }))
}

/// Synthesize a 2q unitary matrix and apply the result onto a DAGCircuit
fn synthesize_2q_matrix_onto(
    out: &mut DAGCircuitBuilder,
    unitary: CowArray<Complex64, Ix2>,
    qargs_phys: [PhysicalQubit; 2],
    qargs_virt: [Qubit; 2],
    state: &mut UnitarySynthesisState,
    constraint: QpuConstraint,
) -> PyResult<bool> {
    let Some(result) =
        synthesize_2q_matrix(unitary, qargs_phys, state, constraint, fidelity_2q_sequence)?
    else {
        return Ok(false);
    };
    // ... now apply the best sequence.
    let order = result.dir.as_indices();
    let out_qargs = [qargs_virt[order[0] as usize], qargs_virt[order[1] as usize]];
    let qubit_keys = [
        out.insert_qargs(&[out_qargs[0]]),
        out.insert_qargs(&[out_qargs[1]]),
        out.insert_qargs(&[out_qargs[0], out_qargs[1]]),
        out.insert_qargs(&[out_qargs[1], out_qargs[0]]),
    ];
    out.add_global_phase(&Param::Float(result.sequence.global_phase()))?;
    for (gate, params, qubits) in result.sequence.gates() {
        let qubits = match qubits.as_slice() {
            [0] => qubit_keys[0],
            [1] => qubit_keys[1],
            [0, 1] => qubit_keys[2],
            [1, 0] => qubit_keys[3],
            _ => panic!("internal logic error: decomposed sequence contained unexpected qargs"),
        };
        let op = match gate.view() {
            OperationRef::StandardGate(gate) => PackedOperation::from(gate),
            OperationRef::PyCustom(inst) => Python::attach(|py| -> PyResult<_> {
                let inst = inst.py_copy(py)?;
                inst.ob.setattr(py, intern!(py, "params"), params)?;
                Ok(inst.into())
            })?,
            _ => panic!("internal logic error: decomposed sequence contains a non-gate"),
        };
        let params = (!params.is_empty()).then(|| {
            Box::new(Parameters::Params(
                params.iter().copied().map(Param::Float).collect(),
            ))
        });
        out.push_back(PackedInstruction {
            op,
            qubits,
            clbits: Default::default(),
            params,
            label: None,
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        })?;
    }
    Ok(true)
}

/// Mutate the given 2q matrix in place to be `swap @ m @ swap`, i.e. "`m`, but applied to the qubit
/// arguments in reverse".
fn conjugate_with_swaps(mut m: ArrayViewMut2<Complex64>) {
    // Swap rows 1 and 2
    let (mut row_1, mut row_2) = m.multi_slice_mut((s![1, ..], s![2, ..]));
    azip!((x in &mut row_1, y in &mut row_2) (*x, *y) = (*y, *x));

    // Swap columns 1 and 2
    let (mut col_1, mut col_2) = m.multi_slice_mut((s![.., 1], s![.., 2]));
    azip!((x in &mut col_1, y in &mut col_2) (*x, *y) = (*y, *x));
}

/// Python entry point to [run_unitary_synthesis].
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "run_main_loop", signature=(dag, qubit_indices, min_qubits, target, basis_gates, synth_gates, coupling_edges, approximation_degree=None, natural_direction=None, pulse_optimize=None))]
pub fn py_unitary_synthesis(
    py: Python,
    dag: &DAGCircuit,
    qubit_indices: Vec<PhysicalQubit>,
    min_qubits: usize,
    target: Option<&Target>,
    basis_gates: HashSet<String>,
    synth_gates: HashSet<String>,
    coupling_edges: HashSet<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<bool>,
) -> PyResult<Option<DAGCircuit>> {
    let config = UnitarySynthesisConfig {
        approximation: Approximation::from_py_approximation_degree(approximation_degree),
        use_pulse_optimizer: UsePulseOptimizer::from_py_pulse_optimize(pulse_optimize),
        decomposition_direction_2q: DecompositionDirection2q::from_py_natural_direction(
            natural_direction,
        ),
        run_python_decomposers: true,
    };
    let mut state = UnitarySynthesisState::new(config);
    let mut basis_gates_set: IndexSet<&str>;
    let constraint = match target {
        Some(target) => QpuConstraint::Target(target),
        None => {
            basis_gates_set = basis_gates.iter().map(String::as_str).collect();
            basis_gates_set.sort();
            QpuConstraint::Loose {
                basis_gates: &basis_gates_set,
                coupling: &coupling_edges,
            }
        }
    };
    let op_counts = dag.count_ops(true)?;
    let gate_counts: usize = synth_gates
        .iter()
        .map(|name| op_counts.get(name).unwrap_or(&0usize))
        .copied()
        .sum();
    if gate_counts == 0 {
        return Ok(None);
    }
    let run_in_parallel = getenv_use_multiple_threads();
    if run_in_parallel && gate_counts > PARALLEL_THRESHOLD {
        // Release GIL in case there are python gates being processed as the synthesis target gate
        // and the GIL is needed by worker threads
        let node_replace_map = py.detach(|| {
            parallel_synthesis(
                dag,
                &synth_gates,
                &qubit_indices,
                constraint,
                &mut state,
                min_qubits,
            )
        })?;
        Ok(Some(apply_synthesis(
            dag,
            node_replace_map,
            &qubit_indices,
            &synth_gates,
            min_qubits,
            &mut state,
            constraint,
        )?))
    } else {
        Ok(Some(serial_run_unitary_synthesis(
            dag,
            &synth_gates,
            min_qubits,
            &qubit_indices,
            &mut state,
            constraint,
        )?))
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "synthesize_unitary_matrix", signature=(unitary, qubit_indices, target, basis_gates, coupling_edges, approximation_degree=None, natural_direction=None, pulse_optimize=None))]
pub fn py_synthesize_unitary_matrix(
    unitary: PyReadonlyArray2<Complex64>,
    qubit_indices: Vec<PhysicalQubit>,
    target: Option<&Target>,
    basis_gates: HashSet<String>,
    coupling_edges: HashSet<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
    pulse_optimize: Option<bool>,
) -> PyResult<DAGCircuit> {
    let config = UnitarySynthesisConfig {
        approximation: Approximation::from_py_approximation_degree(approximation_degree),
        use_pulse_optimizer: UsePulseOptimizer::from_py_pulse_optimize(pulse_optimize),
        decomposition_direction_2q: DecompositionDirection2q::from_py_natural_direction(
            natural_direction,
        ),
        run_python_decomposers: true,
    };
    let mut state = UnitarySynthesisState::new(config);
    let mut basis_gates_set: IndexSet<&str>;
    let constraint = match target {
        Some(target) => QpuConstraint::Target(target),
        None => {
            basis_gates_set = basis_gates.iter().map(String::as_str).collect();
            basis_gates_set.sort();
            QpuConstraint::Loose {
                basis_gates: &basis_gates_set,
                coupling: &coupling_edges,
            }
        }
    };

    let mat = unitary.as_array();

    let num_qubits = mat.shape()[0].trailing_zeros();

    let mut out_dag = DAGCircuit::new();
    let qubits = QuantumRegister::new_owning("q", num_qubits);
    out_dag.add_qreg(qubits)?;
    let mut out_dag = out_dag.into_builder();
    let out_qargs: Vec<Qubit> = (0..num_qubits).map(Qubit).collect();
    if !synthesize_matrix_onto(
        &mut out_dag,
        CowArray::from(mat.view()),
        &qubit_indices,
        &out_qargs,
        &mut state,
        constraint,
    )? {
        return Err(QiskitError::new_err("Failed to decompose unitary"));
    }
    Ok(out_dag.build())
}

pub fn unitary_synthesis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_unitary_synthesis))?;
    m.add_wrapped(wrap_pyfunction!(py_synthesize_unitary_matrix))?;
    Ok(())
}
