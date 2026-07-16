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

use hashbrown::HashMap;
use hashbrown::HashSet;
use ndarray::prelude::*;
use pyo3::Bound;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::converters::QuantumCircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::gate_matrix::CX_GATE;
use qiskit_circuit::imports::HLS_SYNTHESIZE_OP_USING_PLUGINS;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, StandardGate, StandardInstruction, radd_param,
};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{BlocksMode, Clbit, Qubit, VarsMode};
use qiskit_synthesis::pauli_products::synthesize_ppm;
use qiskit_synthesis::pauli_products::synthesize_ppr;
use smallvec::SmallVec;

use crate::TranspilerError;
use crate::equivalence::EquivalenceLibrary;
use crate::target::Qargs;
use crate::target::Target;
use qiskit_circuit::PhysicalQubit;
use qiskit_synthesis::euler_one_qubit_decomposer::EulerBasis;
use qiskit_synthesis::euler_one_qubit_decomposer::angles_from_unitary;
use qiskit_synthesis::qsd::quantum_shannon_decomposition;
use qiskit_synthesis::two_qubit_decompose::TwoQubitBasisDecomposer;

use qiskit_circuit::instruction::{Instruction, Parameters};

/// Track global qubits by their state.
/// The global qubits are numbered by consecutive integers starting at `0`,
/// and the states are distinguished into clean (:math:`|0\rangle`)
/// and dirty (unknown).
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
struct QubitTracker {
    /// The total number of global qubits
    num_qubits: usize,
    /// Stores the state for each qubit: `true` means clean, `false` means dirty
    state: Vec<bool>,
    /// Stores whether qubits are allowed be used
    enabled: Vec<bool>,
    /// Used internally for keeping the computations in `O(n)`
    ignored: Vec<bool>,
}

impl QubitTracker {
    /// Sets state of the given qubits to dirty
    fn set_dirty(&mut self, qubits: &[Qubit]) {
        for q in qubits {
            self.state[q.index()] = false;
        }
    }

    /// Sets state of the given qubits to clean
    fn set_clean(&mut self, qubits: &[Qubit]) {
        for q in qubits {
            self.state[q.index()] = true;
        }
    }

    /// Disables using the given qubits
    fn disable<T: Iterator<Item = Qubit>>(&mut self, qubits: T) {
        for q in qubits {
            self.enabled[q.index()] = false;
        }
    }

    /// Enable using the given qubits
    fn enable(&mut self, qubits: &[Qubit]) {
        for q in qubits {
            self.enabled[q.index()] = true;
        }
    }

    /// Returns the number of enabled clean qubits, ignoring the given qubits
    fn num_clean(&mut self, ignored_qubits: &[Qubit]) -> usize {
        // TODO: check if it's faster to avoid using ignored
        for q in ignored_qubits {
            self.ignored[q.index()] = true;
        }

        let count = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && self.state[*q])
            .count();

        for q in ignored_qubits {
            self.ignored[q.index()] = false;
        }

        count
    }

    /// Returns the number of enabled dirty qubits, ignoring the given qubits
    fn num_dirty(&mut self, ignored_qubits: &[Qubit]) -> usize {
        // TODO: check if it's faster to avoid using ignored
        for q in ignored_qubits {
            self.ignored[q.index()] = true;
        }

        let count = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && !self.state[*q])
            .count();

        for q in ignored_qubits {
            self.ignored[q.index()] = false;
        }

        count
    }

    /// Get `num_qubits` enabled qubits, excluding `ignored_qubits`, and returning the
    /// clean qubits first.
    fn borrow(&mut self, num_qubits: usize, ignored_qubits: &[Qubit]) -> Vec<Qubit> {
        // TODO: check if it's faster to avoid using ignored
        for q in ignored_qubits {
            self.ignored[q.index()] = true;
        }

        let clean_ancillas = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && self.state[*q])
            .map(Qubit::new);
        let dirty_ancillas = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && !self.state[*q])
            .map(Qubit::new);
        let out: Vec<Qubit> = clean_ancillas
            .chain(dirty_ancillas)
            .take(num_qubits)
            .collect();

        for q in ignored_qubits {
            self.ignored[q.index()] = false;
        }
        out
    }

    /// Replaces the state of the given qubits by their state in the `other` tracker
    fn replace_state<T: Iterator<Item = Qubit>>(&mut self, other: &QubitTracker, qubits: T) {
        for q in qubits {
            self.state[q.index()] = other.state[q.index()]
        }
    }

    /// Returns whether `qubit` is clean
    fn is_qubit_clean(&self, qubit: Qubit) -> bool {
        self.state[qubit.index()]
    }
}

#[pymethods]
impl QubitTracker {
    #[new]
    fn new(num_qubits: usize, qubits_initially_zero: bool) -> Self {
        QubitTracker {
            num_qubits,
            state: vec![qubits_initially_zero; num_qubits],
            enabled: vec![true; num_qubits],
            ignored: vec![false; num_qubits],
        }
    }

    /// Returns the number of enabled dirty qubits, ignoring the given qubits
    #[pyo3(name = "num_dirty")]
    fn py_num_dirty(&mut self, ignored_qubits: Vec<Qubit>) -> usize {
        self.num_dirty(&ignored_qubits)
    }

    /// Returns the number of enabled clean qubits, ignoring the given qubits
    #[pyo3(name = "num_clean")]
    fn py_num_clean(&mut self, ignored_qubits: Vec<Qubit>) -> usize {
        self.num_clean(&ignored_qubits)
    }

    /// enable using the given qubits
    #[pyo3(name = "enable")]
    fn py_enable(&mut self, qubits: Vec<Qubit>) {
        self.enable(&qubits)
    }

    /// Disables using the given qubits
    #[pyo3(name = "disable")]
    fn py_disable(&mut self, qubits: Vec<Qubit>) {
        self.disable(qubits.into_iter())
    }

    /// Sets state of the given qubits to dirty
    #[pyo3(name = "set_dirty")]
    fn py_set_dirty(&mut self, qubits: Vec<Qubit>) {
        self.set_dirty(&qubits)
    }

    /// Sets state of the given qubits to clean
    #[pyo3(name = "set_clean")]
    fn py_set_clean(&mut self, qubits: Vec<Qubit>) {
        self.set_clean(&qubits)
    }

    /// Get `num_qubits` enabled qubits, excluding `ignored_qubits`, and returning the
    /// clean qubits first.
    #[pyo3(name = "borrow")]
    fn py_borrow(&mut self, num_qubits: usize, ignored_qubits: Vec<Qubit>) -> Vec<usize> {
        self.borrow(num_qubits, &ignored_qubits)
            .into_iter()
            .map(|q| q.index())
            .collect()
    }

    /// Replaces the state of the given qubits by their state in the `other` tracker
    #[pyo3(name = "replace_state")]
    fn py_replace_state(&mut self, other: &QubitTracker, qubits: Vec<Qubit>) {
        self.replace_state(other, qubits.into_iter())
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns whether qubit is clean
    #[pyo3(name = "is_qubit_clean")]
    fn py_is_qubit_clean(&self, qubit: Qubit) -> bool {
        self.is_qubit_clean(qubit)
    }

    /// Copies the contents
    fn copy(&self) -> Self {
        QubitTracker {
            num_qubits: self.num_qubits,
            state: self.state.clone(),
            enabled: self.enabled.clone(),
            ignored: self.ignored.clone(),
        }
    }

    /// Pretty-prints
    fn __str__(&self) -> String {
        let mut out = String::from("QubitTracker(");
        for q in 0..self.num_qubits {
            out.push_str(&q.to_string());
            out.push(':');
            out.push(' ');
            if !self.enabled[q] {
                out.push('_');
            } else if self.state[q] {
                out.push('0');
            } else {
                out.push('*');
            }
            if q != self.num_qubits - 1 {
                out.push(';');
                out.push(' ');
            } else {
                out.push(')');
            }
        }
        out
    }
}

/// Internal class that encapsulates immutable data required by the HighLevelSynthesis transpiler pass.
#[pyclass(
    module = "qiskit._accelerate.high_level_synthesis",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct HighLevelSynthesisData {
    // The high-level-synthesis config that specifies the synthesis methods
    // to use for high-level-objects in the circuit.
    // This is only accessed from the Python space.
    #[pyo3(get)]
    hls_config: Py<PyAny>,

    // The high-level-synthesis plugin manager that specifies the synthesis methods
    // available for various high-level-objects.
    // This is only accessed from the Python space.
    #[pyo3(get)]
    hls_plugin_manager: Py<PyAny>,

    // The names of high-level objects with available synthesis plugins.
    // This is an optimization to avoid calling python when an object has no
    // synthesis plugins.
    #[pyo3(get)]
    hls_op_names: HashSet<String>,

    // Optional, directed graph represented as a coupling map.
    // This is only accessedfrom the Python space (when passing the coupling map to
    // high-level synthesis plugins).
    #[pyo3(get)]
    coupling_map: Py<PyAny>,

    // Optional, the backend target to use for this pass. If it is specified,
    // it will be used instead of the coupling map.
    // It needs to be used both from python and rust, and hence is represented
    // as Py<Target> to avoid cloning.
    #[pyo3(get)]
    target: Option<Py<Target>>,

    // The equivalence library used (instructions in this library will not
    // be unrolled by this pass).
    #[pyo3(get)]
    equivalence_library: Option<Py<EquivalenceLibrary>>,

    // Supported instructions in case that target is not specified.
    #[pyo3(get)]
    device_insts: HashSet<String>,

    // A flag indicating whether the qubit indices of high-level-objects in the
    // circuit correspond to qubit indices on the target backend.
    #[pyo3(get)]
    use_physical_indices: bool,

    // The minimum number of qubits for operations in the input dag to translate.
    #[pyo3(get)]
    min_qubits: usize,

    // Indicates whether to use custom definitions.
    #[pyo3(get)]
    unroll_definitions: bool,

    // Indicates whether default synthesis methods for high-level-objects should
    // prioritize methods for Clifford+T basis set.
    #[pyo3(get)]
    optimize_clifford_t: bool,
}

#[pymethods]
impl HighLevelSynthesisData {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn __new__(
        hls_config: Py<PyAny>,
        hls_plugin_manager: Py<PyAny>,
        hls_op_names: HashSet<String>,
        coupling_map: Py<PyAny>,
        target: Option<Py<Target>>,
        equivalence_library: Option<Py<EquivalenceLibrary>>,
        device_insts: HashSet<String>,
        use_physical_indices: bool,
        min_qubits: usize,
        unroll_definitions: bool,
        optimize_clifford_t: bool,
    ) -> Self {
        Self {
            hls_config,
            hls_plugin_manager,
            hls_op_names,
            coupling_map,
            target,
            equivalence_library,
            device_insts,
            use_physical_indices,
            min_qubits,
            unroll_definitions,
            optimize_clifford_t,
        }
    }

    fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (
            self.hls_config.clone_ref(py),
            self.hls_plugin_manager.clone_ref(py),
            self.hls_op_names.clone(),
            self.coupling_map.clone_ref(py),
            self.target.clone(),
            self.equivalence_library.clone(),
            self.device_insts.clone(),
            self.use_physical_indices,
            self.min_qubits,
            self.unroll_definitions,
            self.optimize_clifford_t,
        )
            .into_py_any(py)
    }

    fn __str__(&self) -> String {
        format!(
            "HighLevelSynthesisData(hls_config: {:?}, hls_plugin_manager: {:?}, hls_op_names: {:?}, coupling_map: {:?}, target: {:?},  equivalence_library: {:?}, device_insts: {:?}, use_physical_indices: {:?}, min_qubits: {:?}, unroll_definitions: {:?}, optimize_clifford_t: {:?})",
            self.hls_config,
            self.hls_plugin_manager,
            self.hls_op_names,
            self.coupling_map,
            self.target,
            self.equivalence_library,
            self.device_insts,
            self.use_physical_indices,
            self.min_qubits,
            self.unroll_definitions,
            self.optimize_clifford_t,
        )
    }
}

/// A super-fast check whether all operations in `op_names` are natively supported.
/// This check is based only on the names of the operations in the circuit.
fn all_instructions_supported(
    py: Python,
    data: &Bound<HighLevelSynthesisData>,
    dag: &DAGCircuit,
) -> PyResult<bool> {
    let ops = dag.count_ops(true)?;
    let mut op_keys = ops.keys();

    let borrowed_data = data.borrow();

    match &borrowed_data.target {
        Some(target) => {
            let target = target.borrow(py);
            if target.num_qubits.is_some() {
                // If we have the target and HighLevelSynthesis runs pre-routing,
                // we check whether every operation name in op_names is supported
                // by the target.
                if borrowed_data.use_physical_indices {
                    return Ok(false);
                }
                Ok(op_keys
                    .all(|name| target.instruction_supported(name, &Qargs::Global, &[], false)))
            } else {
                // If we do not have the target, we check whether every operation
                // in op_names is inside the basis gates.
                Ok(op_keys.all(|name| borrowed_data.device_insts.contains(name)))
            }
        }
        None => Ok(op_keys.all(|name| borrowed_data.device_insts.contains(name))),
    }
}

/// Check whether an operation is natively supported.
fn instruction_supported(
    py: Python,
    data: &Bound<HighLevelSynthesisData>,
    name: &str,
    qubits: &[Qubit],
) -> bool {
    let borrowed_data = data.borrow();
    match &borrowed_data.target {
        Some(target) => {
            let target = target.borrow(py);
            if target.num_qubits.is_some() {
                if borrowed_data.use_physical_indices {
                    let physical_qubits: Qargs =
                        qubits.iter().map(|q| PhysicalQubit(q.0)).collect();
                    target.instruction_supported(name, &physical_qubits, &[], false)
                } else {
                    target.instruction_supported(name, &Qargs::Global, &[], false)
                }
            } else {
                borrowed_data.device_insts.contains(name)
            }
        }
        None => borrowed_data.device_insts.contains(name),
    }
}

/// Check whether an operation does not need to be synthesized.
fn definitely_skip_op(
    py: Python,
    data: &Bound<HighLevelSynthesisData>,
    op: &PackedOperation,
    qubits: &[Qubit],
) -> bool {
    let borrowed_data: PyRef<'_, HighLevelSynthesisData> = data.borrow();

    if qubits.len() < borrowed_data.min_qubits {
        return true;
    }

    if op.directive() {
        return true;
    }

    if op.try_control_flow().is_some() {
        return false;
    }

    // If the operation is natively supported, we can skip it.
    if instruction_supported(py, data, op.name(), qubits) {
        return true;
    }

    // If there are available plugins for this operation, we should try them
    // before checking the equivalence library.
    if borrowed_data.hls_op_names.iter().any(|s| s == op.name()) {
        return false;
    }

    if let Some(equiv_lib) = &borrowed_data.equivalence_library {
        if equiv_lib.borrow(py).has_entry(op) {
            return true;
        }
    }

    false
}

/// Recursively synthesizes a circuit. This circuit is either the original circuit,
/// the definition circuit for one of the gates, or a circuit returned by a plugin.
///
/// The input to this function is the circuit to be synthesized and the global
/// qubits over which it is defined.
///
/// The output is the synthesized circuit and the global qubits over which it is
/// defined. Note that by using auxiliary qubits, the output circuit may be defined
/// over more qubits than the input circuit.
///
/// The function also updates in-place the qubit tracker, which keeps track of the
/// state of each global qubits (whether it's clean, dirty, or cannot be used).
fn run_on_circuitdata(
    py: Python,
    input_circuit: &CircuitData,
    input_qubits: &[Qubit],
    data: &Bound<HighLevelSynthesisData>,
    tracker: &mut QubitTracker,
) -> PyResult<(CircuitData, Vec<Qubit>)> {
    if input_circuit.num_qubits() != input_qubits.len() {
        return Err(TranspilerError::new_err(format!(
            "HighLevelSynthesis: number of input qubits ({}) does not match the circuit size ({})",
            input_qubits.len(),
            input_circuit.num_qubits()
        )));
    }

    // We iteratively process circuit instructions in the order they appear in the input circuit,
    // and add the synthesized instructions to the output circuit. Note that in the process the
    // output circuit may need to be extended with additional qubits. In addition, we keep track
    // of the state of the global qubits using the qubits tracker.
    //
    // Note: This is a first version of a potentially more elaborate approach to find
    // good operation/ancilla allocations. The current approach is greedy and just gives
    // all available ancilla qubits to the current operation ("the-first-takes-all" approach).
    // It does not distribute ancilla qubits between different operations present in the circuit.

    let mut output_circuit: CircuitData =
        CircuitData::copy_empty_like(input_circuit, VarsMode::Alike, BlocksMode::Drop)?;
    let mut output_qubits = input_qubits.to_vec();

    // The "inverse" map from the global qubits to the output circuit's qubits.
    // This map may be extended if additional auxiliary qubits get used.
    let mut global_to_local: HashMap<Qubit, Qubit> = output_qubits
        .iter()
        .enumerate()
        .map(|(i, j)| (*j, Qubit::new(i)))
        .collect();

    for inst in input_circuit.iter() {
        // op's qubits as viewed globally
        let op_qubits = input_circuit
            .get_qargs(inst.qubits)
            .iter()
            .map(|q| input_qubits[q.index()])
            .collect::<Vec<_>>();
        let op_clbits = input_circuit.get_cargs(inst.clbits);

        // Start by handling special operations.
        // In the future, we can also consider other possible optimizations, e.g.:
        //   - improved qubit tracking after a SWAP gate
        //   - automatically simplify control gates with control at 0.
        if ["id", "delay", "barrier"].contains(&inst.op.name()) {
            output_circuit.push(inst.clone())?;
            // tracker is not updated, these are no-ops
            continue;
        }

        if inst.op.name() == "reset" {
            output_circuit.push(inst.clone())?;
            tracker.set_clean(&op_qubits);
            continue;
        }

        // Check if synthesis for this operation can be skipped
        if definitely_skip_op(py, data, &inst.op, &op_qubits) {
            if let Some(cf) = input_circuit.try_view_control_flow(inst) {
                let blocks: Vec<_> = cf
                    .blocks()
                    .into_iter()
                    .map(|b| output_circuit.add_block(b.clone()))
                    .collect();
                output_circuit.push(PackedInstruction {
                    params: (!blocks.is_empty()).then(|| Box::new(Parameters::Blocks(blocks))),
                    ..inst.clone()
                })?;
            } else {
                output_circuit.push(inst.clone())?;
            }
            tracker.set_dirty(&op_qubits);
            continue;
        }

        // Recursively handle control-flow.
        // Currently we do not allow subcircuits within the control flow to use auxiliary qubits
        // and mark all the usable qubits as dirty. This is done in order to avoid complications
        // that different subcircuits may choose to use different auxiliary global qubits, and to
        // avoid complications related to tracking qubit status for while- loops.
        // In the future, this handling can potentially be improved.
        if let Some(control_flow) = input_circuit.try_view_control_flow(inst) {
            // We do not allow using any additional qubits outside of the block.
            let mut block_tracker = tracker.clone();
            let to_disable = (0..tracker.num_qubits())
                .map(Qubit::new)
                .filter(|q| !op_qubits.contains(q));
            block_tracker.disable(to_disable);
            block_tracker.set_dirty(&op_qubits);

            let blocks = control_flow
                .blocks()
                .into_iter()
                .map(|block| -> PyResult<_> {
                    let (new_block, _) =
                        run_on_circuitdata(py, block, &op_qubits, data, &mut block_tracker)?;
                    Ok(output_circuit.add_block(new_block))
                })
                .collect::<PyResult<_>>()?;
            let packed_instruction = PackedInstruction::from_control_flow(
                inst.op.control_flow().clone(),
                blocks,
                inst.qubits,
                inst.clbits,
                inst.label.as_deref().cloned(),
            );
            output_circuit.push(packed_instruction)?;
            tracker.set_dirty(&op_qubits);
            continue;
        }

        // Now we synthesize the operation.
        // The function synthesize_operation returns either None if the operation does not need to be
        // synthesized, or returns a quantum circuit together with the global qubits on which this
        // circuit is defined. Note that the synthesized circuit may involve auxiliary
        // global qubits not used by the input circuit.
        let synthesize_operation_result = synthesize_operation(
            py,
            data,
            tracker,
            &op_qubits,
            &inst.op,
            inst.params
                .as_deref()
                .map(|p| match p {
                    Parameters::Params(params) => params.as_slice(),
                    Parameters::Blocks(_) => panic!("control flow should not be present"),
                })
                .unwrap_or_default(),
            inst.label.as_deref().map(|l| l.as_str()),
        )?;

        match synthesize_operation_result {
            None => {
                // If the synthesis did not change anything, we add the operation to the output circuit
                // and update the qubit tracker.
                output_circuit.push(inst.clone())?;
                tracker.set_dirty(&op_qubits);
            }
            Some((synthesized_circuit, synthesized_circuit_qubits)) => {
                // This pedantic check can possibly be removed.
                if synthesized_circuit.num_qubits() != synthesized_circuit_qubits.len() {
                    return Err(TranspilerError::new_err(format!(
                        "HighLevelSynthesis: number of output qubits ({}) does not match the circuit size ({})",
                        synthesized_circuit_qubits.len(),
                        synthesized_circuit.num_qubits()
                    )));
                }

                // If the synthesized circuit uses (auxiliary) global qubits that are not in the output circuit,
                // we add these qubits to the output circuit.
                if synthesized_circuit_qubits.len() > op_qubits.len() {
                    for q in &synthesized_circuit_qubits {
                        if !global_to_local.contains_key(q) {
                            global_to_local.insert(*q, Qubit::new(output_qubits.len()));
                            output_qubits.push(*q);
                            output_circuit.add_qubit(ShareableQubit::new_anonymous(), false)?;
                        }
                    }
                }

                // Add the operations from the circuit synthesized for the current operation to the output circuit.
                // The correspondence between qubits is:
                // qubit index in the synthesized circuit -> corresponding global qubit -> corresponding qubit in the output circuit
                let qubit_map: HashMap<Qubit, Qubit> = synthesized_circuit_qubits
                    .iter()
                    .enumerate()
                    .map(|(i, q)| (Qubit::new(i), global_to_local[q]))
                    .collect();

                for inst_inner in synthesized_circuit.iter() {
                    let inst_inner_qubits = synthesized_circuit.get_qargs(inst_inner.qubits);
                    let inst_inner_clbits = synthesized_circuit.get_cargs(inst_inner.clbits);

                    let inst_outer_qubits: Vec<Qubit> =
                        inst_inner_qubits.iter().map(|q| qubit_map[q]).collect();
                    let inst_outer_clbits: Vec<Clbit> = inst_inner_clbits
                        .iter()
                        .map(|c| op_clbits[c.index()])
                        .collect();

                    output_circuit.push_packed_operation(
                        inst_inner.op.clone(),
                        inst_inner.params.as_deref().cloned(),
                        &inst_outer_qubits,
                        &inst_outer_clbits,
                    )?;
                }

                let updated_global_phase = radd_param(
                    output_circuit.global_phase().clone(),
                    synthesized_circuit.global_phase().clone(),
                );
                output_circuit.set_global_phase_param(updated_global_phase)?;
            }
        }
    }

    // Another pedantic check that can possibly be removed.
    if output_circuit.num_qubits() != output_qubits.len() {
        return Err(TranspilerError::new_err(format!(
            "HighLevelSynthesis: number of output qubits ({}) does not match the circuit size ({})",
            output_qubits.len(),
            output_circuit.num_qubits()
        )));
    }

    Ok((output_circuit, output_qubits))
}

/// Produces a definition circuit for an operation.
///
/// Essentially this function constructs a default definition for a unitary gate, in which case
/// ``op.definition`` purposefully returns ``None``.
/// For all other operation types, it simply calls ``op.definition``.
fn extract_definition(op: &PackedOperation, params: &[Param]) -> PyResult<Option<CircuitData>> {
    match op.view() {
        OperationRef::Unitary(unitary) => {
            let unitary = unitary.matrix_view();
            let shape = unitary.shape();
            match shape {
                // Run 1q synthesis
                [2, 2] => {
                    let [theta, phi, lam, phase] =
                        angles_from_unitary(unitary.view(), EulerBasis::U);
                    let mut circuit_data: CircuitData =
                        CircuitData::with_capacity(1, 0, 1, Param::Float(phase))?;
                    circuit_data.push_standard_gate(
                        StandardGate::U,
                        &[Param::Float(theta), Param::Float(phi), Param::Float(lam)],
                        &[Qubit(0)],
                    )?;
                    Ok(Some(circuit_data))
                }
                // Run 2q synthesis
                [4, 4] => {
                    let decomposer = TwoQubitBasisDecomposer::new_inner(
                        StandardGate::CX.into(),
                        SmallVec::new(),
                        aview2(&CX_GATE),
                        1.0,
                        EulerBasis::U,
                        None,
                    )?;
                    let two_qubit_sequence =
                        decomposer.call_inner(unitary.view(), None, false, None)?;
                    let circuit_data = CircuitData::from_packed_operations(
                        2,
                        0,
                        two_qubit_sequence.gates().iter().map(
                            |(gate, params_floats, qubit_indices)| {
                                let params: SmallVec<[Param; 3]> =
                                    params_floats.iter().map(|p| Param::Float(*p)).collect();
                                let qubits =
                                    qubit_indices.iter().map(|q| Qubit(*q as u32)).collect();
                                Ok((gate.clone(), params, qubits, vec![]))
                            },
                        ),
                        Param::Float(two_qubit_sequence.global_phase()),
                    )?;
                    Ok(Some(circuit_data))
                }
                // Run 3q+ synthesis
                _ => {
                    let synth_circ =
                        quantum_shannon_decomposition(unitary.view(), None, None, None, None)?;
                    Ok(Some(synth_circ))
                }
            }
        }
        OperationRef::StandardGate(g) => Ok(g.definition(params)),
        OperationRef::PyCustom(i) => Ok(i.definition()),
        OperationRef::PauliProductMeasurement(ppm) => Ok(Some(synthesize_ppm(ppm)?)),
        OperationRef::PauliProductRotation(rotation) => Ok(Some(synthesize_ppr(rotation)?)),
        OperationRef::StandardInstruction(i) => match i {
            StandardInstruction::Measure
            | StandardInstruction::Reset
            | StandardInstruction::Barrier(_)
            | StandardInstruction::Delay(_) => Ok(None),
        },
        OperationRef::ControlFlow(_) => Ok(None),
        OperationRef::CustomOperation(custom_gate) => Ok(custom_gate.definition(params)),
    }
}

/// Recursively synthesizes a single operation.
///
/// The input to this function is the operation to be synthesized (consisting of a
/// packed operation, params and extra attributes) and a list of global qubits over
/// which this operation is defined.
///
/// The function returns the synthesized circuit and the global qubits over which this
/// synthesized circuit is defined. Note that by using auxiliary qubits, the output circuit
/// may be defined over more qubits than the input operation. In addition, the output
/// circuit may be ``None``, which means that the operation should remain as it is.
///
/// The function also updates in-place the qubit tracker which keeps track of the state of
/// each global qubit (whether it's clean, dirty, or cannot be used).
fn synthesize_operation(
    py: Python,
    data: &Bound<HighLevelSynthesisData>,
    tracker: &mut QubitTracker,
    input_qubits: &[Qubit],
    op: &PackedOperation,
    params: &[Param],
    label: Option<&str>,
) -> PyResult<Option<(CircuitData, Vec<Qubit>)>> {
    if op.num_qubits() != input_qubits.len() as u32 {
        return Err(TranspilerError::new_err(format!(
            "HighLevelSynthesis: number of operation's qubits ({}) does not match the circuit size ({})",
            op.num_qubits(),
            input_qubits.len()
        )));
    }

    let borrowed_data: PyRef<'_, HighLevelSynthesisData> = data.borrow();

    let mut output_circuit_and_qubits: Option<(CircuitData, Vec<Qubit>)> = None;

    // If this function is called, the operation is not supported by the target, however may have
    // high-level synthesis plugins, and/or be in the equivalence library, and/or have a definition
    // circuit. The priorities are as follows:
    // - First, we try running the battery of high-level synthesis plugins.
    // - Second, we check if the operation is present in the equivalence library.
    // - Third, we unroll custom definitions.
    //
    // If we obtain a new quantum circuit, it needs to be recursively synthesized, so
    // that the final result only consists of supported operations. If there is no
    // change, we return None.

    // Try to synthesize using plugins.
    if borrowed_data.hls_op_names.iter().any(|s| s == op.name()) {
        output_circuit_and_qubits = synthesize_op_using_plugins(
            py,
            data,
            tracker,
            input_qubits,
            &op.view(),
            params,
            label,
        )?;
    }

    // Check if present in the equivalent library.
    if output_circuit_and_qubits.is_none() {
        if let Some(equiv_lib) = &borrowed_data.equivalence_library {
            if equiv_lib.borrow(py).has_entry(op) {
                return Ok(None);
            }
        }
    }

    // Extract definition.
    if output_circuit_and_qubits.is_none() && borrowed_data.unroll_definitions {
        let definition_circuit = extract_definition(op, params)?;
        match definition_circuit {
            Some(definition_circuit) => {
                output_circuit_and_qubits = Some((definition_circuit, input_qubits.to_vec()));
            }
            None => {
                return Err(TranspilerError::new_err(format!(
                    "HighLevelSynthesis is unable to synthesize {:?}",
                    op.name()
                )));
            }
        }
    }

    // Output circuit is a quantum circuit which we want to process recursively.
    // Currently, neither 'synthesize_op_using_plugins' nor 'get_custom_definition'
    // update the tracker (we might want to change this in the future), which makes
    // sense because we have not synthesized the output circuit yet.
    // So we pass the tracker to 'run_on_circuitdata' but make sure to restore the status of
    // clean ancilla qubits after the circuit is synthesized. In order to do that,
    // we save the current state of the tracker.
    if let Some((current_circuit, current_qubits)) = output_circuit_and_qubits {
        let saved_tracker = tracker.copy();
        let (synthesized_circuit, synthesized_qubits) =
            run_on_circuitdata(py, &current_circuit, &current_qubits, data, tracker)?;

        if synthesized_qubits.len() > input_qubits.len() {
            tracker.replace_state(
                &saved_tracker,
                synthesized_qubits[input_qubits.len()..].iter().cloned(),
            );
        }

        output_circuit_and_qubits = Some((synthesized_circuit, synthesized_qubits));
    }

    Ok(output_circuit_and_qubits)
}

/// Attempts to synthesize an operation using available plugins.
///
/// The input to this function is the operation to be synthesized and a list of global
/// qubits over which this operation is defined.
///
/// The function returns either the synthesized quantum circuit and the global qubits over
/// which it's defined, or ``None`` in the case that no synthesis methods are available or
/// applicable (for instance, when there is an insufficient number of auxiliary qubits).
///
/// Internally, this function calls the Python function.
///
/// Currently, this function does not update the qubit tracker, which is handled upstream.
fn synthesize_op_using_plugins(
    py: Python,
    data: &Bound<HighLevelSynthesisData>,
    tracker: &mut QubitTracker,
    input_qubits: &[Qubit],
    op: &OperationRef,
    params: &[Param],
    label: Option<&str>,
) -> PyResult<Option<(CircuitData, Vec<Qubit>)>> {
    let mut output_circuit_and_qubits: Option<(CircuitData, Vec<Qubit>)> = None;

    let op_py = match op {
        OperationRef::ControlFlow(_) => panic!("control flow should not be present"),
        OperationRef::StandardGate(standard) => standard
            .create_py_op(py, Some(params.iter().cloned().collect()), label)?
            .into_any(),
        OperationRef::StandardInstruction(instruction) => instruction
            .create_py_op(py, Some(params.iter().cloned().collect()), label)?
            .into_any(),
        OperationRef::PyCustom(inst) => inst.ob.clone_ref(py),
        OperationRef::Unitary(unitary) => unitary.create_py_op(py, label)?.into_any(),
        OperationRef::PauliProductMeasurement(ppm) => ppm.create_py_op(py, label)?.into_any(),
        OperationRef::PauliProductRotation(rotation) => {
            rotation.create_py_op(py, label)?.into_any()
        }
        OperationRef::CustomOperation(_) => {
            return Err(PyNotImplementedError::new_err(
                "Custom Operations from Rust cannot be exposed to Python.",
            ));
        }
    };

    let res = HLS_SYNTHESIZE_OP_USING_PLUGINS
        .get_bound(py)
        .call1((
            op_py,
            input_qubits.iter().map(|x| x.0).collect::<Vec<_>>(),
            data,
            tracker.clone(),
        ))?
        .extract::<Option<(QuantumCircuitData, Vec<Qubit>)>>()?;

    if let Some((quantum_circuit_data, qubits)) = res {
        output_circuit_and_qubits = Some((quantum_circuit_data.data, qubits));
    }

    Ok(output_circuit_and_qubits)
}

/// Synthesizes an operation.
///
/// This function is currently called by the default plugin for annotated operations to
/// synthesize the base operation. Here `py_op` is a subclass of `Operation` (on the Python
/// side).
#[pyfunction]
#[pyo3(name = "synthesize_operation", signature = (py_op, input_qubits, data, tracker))]
fn py_synthesize_operation(
    py: Python,
    py_op: Bound<PyAny>,
    input_qubits: Vec<Qubit>,
    data: &Bound<HighLevelSynthesisData>,
    tracker: &mut QubitTracker,
) -> PyResult<Option<(PyCircuitData, Vec<usize>)>> {
    let op: OperationFromPython<Py<PyAny>> = py_op.extract()?;

    // Check if the operation can be skipped.
    if definitely_skip_op(py, data, &op.operation, &input_qubits) {
        return Ok(None);
    }

    let result = synthesize_operation(
        py,
        data,
        tracker,
        &input_qubits,
        &op.operation,
        op.params_view(),
        op.label.as_deref().map(|l| l.as_str()),
    )?;

    Ok(result.map(|res: (CircuitData, Vec<Qubit>)| {
        (res.0.into(), res.1.iter().map(|x| x.index()).collect())
    }))
}

/// Synthesizes a circuit.
///
/// This function is only used in the Python testing of the HighLevelSynthesis qubit tracking mechanism.
#[pyfunction]
#[pyo3(name = "synthesize_circuit", signature = (circuit, input_qubits, data, tracker))]
fn py_synthesize_circuit(
    py: Python,
    circuit: &PyCircuitData,
    input_qubits: Vec<Qubit>,
    data: &Bound<HighLevelSynthesisData>,
    tracker: &mut QubitTracker,
) -> PyResult<(PyCircuitData, Vec<usize>)> {
    let res = run_on_circuitdata(py, circuit, &input_qubits, data, tracker)?;

    Ok((res.0.into(), res.1.iter().map(|x| x.index()).collect()))
}

/// Runs HighLevelSynthesis transpiler pass.
///
/// This is the main function called from the Python space. If the pass does not need
/// to do anything, it returns None, meaning that the DAG should remain unchanged.
/// Otherwise, the new DAG is returned.
#[pyfunction]
#[pyo3(name = "run_on_dag", signature = (dag, data, qubits_initially_zero))]
pub fn run_high_level_synthesis(
    py: Python,
    dag: &DAGCircuit,
    data: &Bound<HighLevelSynthesisData>,
    qubits_initially_zero: bool,
) -> PyResult<Option<DAGCircuit>> {
    // Fast-path: check if HighLevelSynthesis can be skipped altogether. This is only
    // done at the top-level since this does not track the qubit states.

    // First, we apply a super-fast (but incomplete) check to see if all the operations
    // present in the circuit are supported by the target / are in the basis.
    if all_instructions_supported(py, data, dag)? {
        return Ok(None);
    }

    // Second, we apply a slightly slower (but still fast) that considers each operation
    // one-by-one.
    let mut fast_path: bool = true;

    for (_, inst) in dag.op_nodes(false) {
        let qubits = dag.get_qargs(inst.qubits);
        if !definitely_skip_op(py, data, &inst.op, qubits) {
            fast_path = false;
            break;
        }
    }

    if fast_path {
        Ok(None)
    } else {
        // Regular-path: we synthesize the circuit recursively. Except for
        // this conversion from DAGCircuit to CircuitData and back, all
        // the recursive functions work with CircuitData objects only.
        let circuit = CircuitData::from_dag_ref(dag)?;

        let num_qubits = circuit.num_qubits();
        let input_qubits: Vec<Qubit> = (0..num_qubits).map(Qubit::new).collect();
        let mut tracker = QubitTracker::new(num_qubits, qubits_initially_zero);

        let (output_circuit, _) =
            run_on_circuitdata(py, &circuit, &input_qubits, data, &mut tracker)?;

        // Using this constructor so name and metadata are not lost
        let new_dag = DAGCircuit::from_circuit(
            QuantumCircuitData {
                data: output_circuit,
                name: dag.get_name().cloned(),
                metadata: dag.get_metadata().map(|m| m.bind(py)).cloned(),
                transpile_layout: None,
            },
            false,
            None,
            None,
        )?;

        Ok(Some(new_dag))
    }
}

pub fn high_level_synthesis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_high_level_synthesis))?;
    m.add_wrapped(wrap_pyfunction!(py_synthesize_operation))?;
    m.add_wrapped(wrap_pyfunction!(py_synthesize_circuit))?;

    m.add_class::<QubitTracker>()?;
    m.add_class::<HighLevelSynthesisData>()?;
    Ok(())
}
