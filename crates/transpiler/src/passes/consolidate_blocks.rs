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

use std::cell::RefCell;

use rayon::prelude::*;
use thread_local::ThreadLocal;

use super::optimize_1q_gates_decomposition::matmul_1q;
use hashbrown::{HashMap, HashSet};
use nalgebra::{Matrix2, Matrix4, U4};
use ndarray::ArrayView2;
use ndarray::{Array2, aview2};
use num_complex::Complex64;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyIndexError;
use pyo3::intern;
use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::gate_matrix::{
    CH_GATE, CX_GATE, CY_GATE, CZ_GATE, DCX_GATE, ECR_GATE, ISWAP_GATE, ONE_QUBIT_IDENTITY,
};
use qiskit_circuit::imports::QI_OPERATOR;
use qiskit_circuit::interner::Interned;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::operations::{ArrayType, Operation, Param, UnitaryGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_synthesis::euler_one_qubit_decomposer::EulerBasis;
use qiskit_synthesis::linalg::nalgebra_array_view;
use qiskit_synthesis::matrix::two_qubit::{
    blocks_to_matrix, get_1q_matrix_from_inst, get_2q_matrix_from_inst, get_matrix_from_inst,
};
use qiskit_synthesis::two_qubit_decompose::RXXEquivalent;
use qiskit_synthesis::two_qubit_decompose::{
    TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer,
};
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::SmallVec;

use crate::passes::unitary_synthesis::{PARAM_SET, TWO_QUBIT_BASIS_SET};
use crate::target::{Qargs, Target};
use qiskit_circuit::PhysicalQubit;
use qiskit_util::getenv_use_multiple_threads;

static IDENTITY_2Q: Matrix4<Complex64> = Matrix4::new(
    // Row 1
    Complex64::ONE,
    Complex64::ZERO,
    Complex64::ZERO,
    Complex64::ZERO,
    // Row 2
    Complex64::ZERO,
    Complex64::ONE,
    Complex64::ZERO,
    Complex64::ZERO,
    // Row 3
    Complex64::ZERO,
    Complex64::ZERO,
    Complex64::ONE,
    Complex64::ZERO,
    // Row 4
    Complex64::ZERO,
    Complex64::ZERO,
    Complex64::ZERO,
    Complex64::ONE,
);

/// Threshold in number of blocks to run the pass multithreaded vs serially
/// At smaller block counts the overhead of multithreading outweighs the speedup
const PARALLEL_THRESHOLD: usize = 50;

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum DecomposerType {
    TwoQubitBasis(TwoQubitBasisDecomposer),
    TwoQubitControlledU(TwoQubitControlledUDecomposer),
}
// TODO: we shouldn't need to clone a decomposer on entry from Python space, but should rely on a
// reference type.  This implementation was added during the transition to PyO3 0.28, to avoid
// proliferating the cloning derive of `FromPyObject` to `TwoQubitBasisDecomposer` and
// `TwoQubitControlledUDecomposer`; that clone is usually a mistake.  Using a reference type within
// this code would require a greater refactor so that the Python-space method
// `py_run_consolidate_blocks` is not the actual worker function.
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for DecomposerType {
    type Error = pyo3::CastError<'a, 'py>;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(ob) = ob.cast::<TwoQubitBasisDecomposer>() {
            Ok(Self::TwoQubitBasis(ob.borrow().clone()))
        } else {
            ob.cast::<TwoQubitControlledUDecomposer>()
                .map(|ob| Self::TwoQubitControlledU(ob.borrow().clone()))
        }
    }
}

fn get_matrix(gate: &StandardGate) -> ArrayView2<'_, Complex64> {
    match gate {
        StandardGate::CX => aview2(&CX_GATE),
        StandardGate::CY => aview2(&CY_GATE),
        StandardGate::CZ => aview2(&CZ_GATE),
        StandardGate::CH => aview2(&CH_GATE),
        StandardGate::DCX => aview2(&DCX_GATE),
        StandardGate::ISwap => aview2(&ISWAP_GATE),
        StandardGate::ECR => aview2(&ECR_GATE),
        _ => unreachable!("Unsupported gate"),
    }
}

/// Helper function that extracts the decomposer and basis gate directly from the [Target].
#[inline]
fn get_decomposer_and_basis_gate(
    target: Option<&Target>,
    approximation_degree: f64,
) -> (DecomposerType, StandardGate) {
    if let Some(target) = target {
        // Targets from C should only support Standard gates.
        if let Some(gate) = target.operations().find_map(|op| {
            op.operation
                .try_standard_gate()
                .and_then(|gate| matches!(gate, PARAM_SET!()).then_some(gate))
        }) {
            return (
                DecomposerType::TwoQubitControlledU(
                    TwoQubitControlledUDecomposer::new(RXXEquivalent::Standard(gate), "ZXZ")
                        .unwrap_or_else(|_| {
                            panic!(
                                "Error while creating TwoQubitControlledUDecomposer using a {} gate.",
                                gate.name()
                            )
                        }),
                ),
                gate,
            );
        }
        if let Some(gate) = target.operations().find_map(|op| {
            op.operation
                .try_standard_gate()
                .and_then(|gate| matches!(gate, TWO_QUBIT_BASIS_SET!()).then_some(gate))
        }) {
            return (
                DecomposerType::TwoQubitBasis(
                    TwoQubitBasisDecomposer::new_inner(
                        gate.into(),
                        SmallVec::default(),
                        get_matrix(&gate),
                        approximation_degree,
                        EulerBasis::U,
                        None,
                    )
                    .unwrap_or_else(|_| {
                        panic!(
                            "Error while creating TwoQubitBasisDecomposer using a {} gate.",
                            gate.name()
                        )
                    }),
                ),
                gate,
            );
        }
    }
    let gate = StandardGate::CX;
    (
        DecomposerType::TwoQubitBasis(
            TwoQubitBasisDecomposer::new_inner(
                gate.into(),
                SmallVec::default(),
                aview2(&CX_GATE),
                1.0,
                EulerBasis::U,
                None,
            )
            .expect("Error while creating TwoQubitBasisDecomposer using a 'cx' gate."),
        ),
        gate,
    )
}

fn is_supported(
    target: Option<&Target>,
    basis_gates: Option<&HashSet<String>>,
    name: &str,
    qargs: &[PhysicalQubit],
) -> bool {
    match target {
        Some(target) => {
            let physical_qargs: Qargs = qargs.iter().map(|bit| PhysicalQubit(bit.0)).collect();
            target.instruction_supported(name, &physical_qargs, &[], false)
        }
        None => match basis_gates {
            Some(basis_gates) => basis_gates.contains(name),
            None => true,
        },
    }
}

// If depth > 20, there will be 1q gates to consolidate.
const MAX_2Q_DEPTH: usize = 20;

struct PhysQargsMap {
    map: Option<Vec<PhysicalQubit>>,
    cache: HashMap<Interned<[Qubit]>, Vec<PhysicalQubit>>,
}
impl PhysQargsMap {
    fn new(map: Option<Vec<PhysicalQubit>>) -> Self {
        Self {
            map,
            cache: Default::default(),
        }
    }
    fn get<'a>(&'a mut self, dag: &'a DAGCircuit, key: Interned<[Qubit]>) -> &'a [PhysicalQubit] {
        let Some(map) = self.map.as_ref() else {
            return PhysicalQubit::lift_slice(dag.get_qargs(key));
        };
        self.cache
            .entry(key)
            .or_insert_with(|| dag.get_qargs(key).iter().map(|q| map[q.index()]).collect())
    }
}

/// The different actions to take from the analysis
enum ConsolidateResult {
    /// The block is equivalent to an identity and should be removed
    Identity,
    /// No consolidation should occur and the block should be left as is
    NoConsolidate,
    /// The block should be consolidated into a `UnitaryGate` of a given
    /// matrix
    Matrix(HashMap<Qubit, usize>, UnitaryGate),
    /// The single gate block should be replaced with the UnitaryGate
    Replace(UnitaryGate),
}

#[allow(clippy::too_many_arguments)]
fn should_substitute(
    dag: &DAGCircuit,
    decomposer: Option<&DecomposerType>,
    target: Option<&Target>,
    basis_gates: Option<&HashSet<String>>,
    basis_gate_name: &str,
    block: &[NodeIndex],
    block_qargs: &mut HashSet<Qubit>,
    phys_qargs: &mut PhysQargsMap,
    force_consolidate: bool,
) -> PyResult<ConsolidateResult> {
    block_qargs.clear();
    if block.len() == 1 {
        let inst_node = block[0];
        let inst = dag[inst_node].unwrap_operation();
        if !is_supported(
            target,
            basis_gates,
            inst.op.name(),
            phys_qargs.get(dag, inst.qubits),
        ) {
            let num_qubits = inst.op.num_qubits();
            let unitary_gate = if num_qubits == 1 {
                let matrix = match get_1q_matrix_from_inst(inst) {
                    Ok(mat) => mat,
                    Err(_) => return Ok(ConsolidateResult::NoConsolidate),
                };
                UnitaryGate {
                    array: ArrayType::OneQ(matrix),
                }
            } else if num_qubits == 2 {
                let matrix = match get_2q_matrix_from_inst(inst) {
                    Ok(mat) => mat,
                    Err(_) => return Ok(ConsolidateResult::NoConsolidate),
                };
                UnitaryGate {
                    array: ArrayType::TwoQ(matrix),
                }
            } else {
                let matrix = match get_matrix_from_inst(inst) {
                    Ok(mat) => mat,
                    Err(_) => return Ok(ConsolidateResult::NoConsolidate),
                };
                UnitaryGate {
                    array: ArrayType::NDArray(matrix),
                }
            };
            return Ok(ConsolidateResult::Replace(unitary_gate));
        }
    }
    let mut basis_count: usize = 0;
    let mut outside_basis = false;
    for node in block {
        let inst = dag[*node].unwrap_operation();
        block_qargs.extend(dag.get_qargs(inst.qubits));
        if inst.op.name() == basis_gate_name {
            basis_count += 1;
        }
        if !is_supported(
            target,
            basis_gates,
            inst.op.name(),
            phys_qargs.get(dag, inst.qubits),
        ) {
            outside_basis = true;
        }
    }
    if block_qargs.len() > 2 {
        let mut qargs: Vec<Qubit> = block_qargs.iter().copied().collect();
        qargs.sort();
        let block_index_map: HashMap<Qubit, usize> = qargs
            .into_iter()
            .enumerate()
            .map(|(idx, qubit)| (qubit, idx))
            .collect();
        let circuit_data = CircuitData::from_packed_operations(
            block_qargs.len() as u32,
            0,
            block.iter().map(|node| {
                let inst = dag[*node].unwrap_operation();

                Ok((
                    inst.op.clone(),
                    inst.params_view().iter().cloned().collect(),
                    dag.get_qargs(inst.qubits)
                        .iter()
                        .map(|x| Qubit::new(block_index_map[x]))
                        .collect(),
                    vec![],
                ))
            }),
            Param::Float(0.),
        )?;
        let matrix = Python::attach(|py| -> PyResult<_> {
            let circuit = circuit_data.into_py_quantum_circuit(py)?;
            let matrix = QI_OPERATOR
                .get_bound(py)
                .call1((circuit,))?
                .getattr(intern!(py, "data"))?
                .extract::<PyReadonlyArray2<Complex64>>()?
                .as_array()
                .to_owned();
            Ok(matrix)
        })?;
        let identity: Array2<Complex64> = Array2::eye(2usize.pow(block_qargs.len() as u32));
        if approx::abs_diff_eq!(identity, matrix.view()) {
            return Ok(ConsolidateResult::Identity);
        } else {
            let unitary_gate = UnitaryGate {
                array: ArrayType::NDArray(matrix),
            };
            return Ok(ConsolidateResult::Matrix(block_index_map, unitary_gate));
        }
    } else {
        let block_index_map = [
            *block_qargs.iter().min().unwrap(),
            *block_qargs.iter().max().unwrap(),
        ];
        let matrix = blocks_to_matrix(dag, block, block_index_map).ok();
        if let Some(matrix) = matrix {
            let consolidate = if force_consolidate
                || block.len() > MAX_2Q_DEPTH
                || (basis_gates.is_some() && outside_basis)
                || (target.is_some() && outside_basis)
            {
                true
            } else {
                let num_basis_gates = if let Some(ref decomposer) = decomposer {
                    match decomposer {
                        DecomposerType::TwoQubitBasis(decomp) => decomp.num_basis_gates_inner(
                            nalgebra_array_view::<Complex64, U4, U4>(matrix.as_view()),
                        )?,
                        DecomposerType::TwoQubitControlledU(decomp) => decomp
                            .num_basis_gates_inner(nalgebra_array_view::<Complex64, U4, U4>(
                                matrix.as_view(),
                            ))?,
                    }
                } else {
                    unreachable!("A decomposer is always set unless force_consolidate is true");
                };
                num_basis_gates < basis_count
            };
            if consolidate {
                if approx::abs_diff_eq!(IDENTITY_2Q, matrix) {
                    return Ok(ConsolidateResult::Identity);
                } else {
                    let unitary_gate = UnitaryGate {
                        array: ArrayType::TwoQ(matrix),
                    };
                    let qubit_pos_map = block_index_map
                        .into_iter()
                        .enumerate()
                        .map(|(idx, qubit)| (qubit, idx))
                        .collect();
                    return Ok(ConsolidateResult::Matrix(qubit_pos_map, unitary_gate));
                }
            }
        }
    }
    Ok(ConsolidateResult::NoConsolidate)
}

#[allow(clippy::too_many_arguments)]
fn consolidation_analysis_parallel<'a>(
    dag: &DAGCircuit,
    decomposer: Option<DecomposerType>,
    target: Option<&Target>,
    basis_gates: Option<&HashSet<String>>,
    basis_gate_name: &str,
    blocks: &'a [Vec<NodeIndex>],
    qubit_map: Option<Vec<PhysicalQubit>>,
    force_consolidate: bool,
) -> PyResult<Vec<(&'a [NodeIndex], ConsolidateResult)>> {
    let block_qargs = ThreadLocal::new();
    let phys_qargs = ThreadLocal::new();
    blocks
        .par_iter()
        .map(|block| {
            let block_qargs = block_qargs.get_or(|| RefCell::new(HashSet::with_capacity(2)));
            let phys_qargs =
                phys_qargs.get_or(|| RefCell::new(PhysQargsMap::new(qubit_map.clone())));
            let consolidate_result = should_substitute(
                dag,
                decomposer.as_ref(),
                target,
                basis_gates,
                basis_gate_name,
                block,
                &mut block_qargs.borrow_mut(),
                &mut phys_qargs.borrow_mut(),
                force_consolidate,
            )?;
            Ok((block.as_slice(), consolidate_result))
        })
        .collect()
}

fn apply_consolidation(
    dag: &mut DAGCircuit,
    block: &[NodeIndex],
    consolidation: ConsolidateResult,
) -> PyResult<()> {
    match consolidation {
        ConsolidateResult::NoConsolidate => {}
        ConsolidateResult::Identity => {
            for index in block {
                dag.remove_op_node(*index);
            }
        }
        ConsolidateResult::Matrix(block_index_map, unitary_gate) => {
            let clbit_pos_map = HashMap::new();
            dag.replace_block(
                block,
                PackedOperation::from_unitary(Box::new(unitary_gate)),
                None,
                None,
                false,
                &block_index_map,
                &clbit_pos_map,
            )?;
        }
        ConsolidateResult::Replace(unitary_gate) => {
            dag.substitute_op(
                block[0],
                PackedOperation::from_unitary(Box::new(unitary_gate)),
                None,
                None,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "consolidate_blocks", signature = (dag, decomposer, basis_gate_name, force_consolidate, target=None, basis_gates=None, blocks=None, runs=None, qubit_map=None))]
fn py_run_consolidate_blocks(
    py: Python,
    dag: &mut DAGCircuit,
    decomposer: Option<DecomposerType>,
    basis_gate_name: &str,
    force_consolidate: bool,
    target: Option<&Target>,
    basis_gates: Option<HashSet<String>>,
    blocks: Option<Vec<Vec<usize>>>,
    runs: Option<Vec<Vec<usize>>>,
    qubit_map: Option<Vec<PhysicalQubit>>,
) -> PyResult<()> {
    // If we don't have a decomposer and force consolidate is not set then there is not any
    // consolidation to do.
    if decomposer.is_none() && !force_consolidate {
        return Ok(());
    }
    // The node indices that enter from `blocks` and `runs` come from Python space, and we can't
    // trust that they come from a correct analysis (or the block/run collection might have been
    // invalidated). Rather than panicking, we should raise Python-space exceptions. We don't have
    // to check indices that are generated by trusted Rust-only methods within this function.
    let valid_op_node = |dag: &mut DAGCircuit, index: usize| -> PyResult<NodeIndex> {
        let index = NodeIndex::new(index);
        match dag.dag().node_weight(index) {
            Some(NodeType::Operation(_)) => Ok(index),
            _ => Err(PyIndexError::new_err(
                "node index in run or block was not a valid operation",
            )),
        }
    };
    let blocks = match blocks {
        Some(runs) => runs
            .into_iter()
            .map(|run| {
                run.into_iter()
                    .map(|index| valid_op_node(dag, index))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?,
        // If runs are specified but blocks are none we're in a legacy configuration where external
        // collection passes are being used. In this case don't collect blocks because it's
        // unexpected.
        None => match runs {
            Some(_) => vec![],
            None => dag.collect_2q_runs().unwrap(),
        },
    };

    let runs: Option<Vec<Vec<NodeIndex>>> = runs
        .map(|runs| {
            runs.into_iter()
                .map(|run| {
                    run.into_iter()
                        .map(|index| valid_op_node(dag, index))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let run_in_parallel = getenv_use_multiple_threads();
    if run_in_parallel && blocks.len() > PARALLEL_THRESHOLD {
        let consolidations = py.detach(|| {
            consolidation_analysis_parallel(
                dag,
                decomposer,
                target,
                basis_gates.as_ref(),
                basis_gate_name,
                &blocks,
                qubit_map.clone(),
                force_consolidate,
            )
        })?;
        for (block, result) in consolidations {
            apply_consolidation(dag, block, result)?;
        }
    } else {
        // In most cases, the qargs in a block will not exceed 2 qubits.
        let mut block_qargs: HashSet<Qubit> = HashSet::with_capacity(2);
        let mut phys_qargs = PhysQargsMap::new(qubit_map.clone());
        for block in &blocks {
            let result = should_substitute(
                dag,
                decomposer.as_ref(),
                target,
                basis_gates.as_ref(),
                basis_gate_name,
                block,
                &mut block_qargs,
                &mut phys_qargs,
                force_consolidate,
            )?;
            apply_consolidation(dag, block, result)?;
        }
    }
    if let Some(runs) = runs {
        let all_block_gates: HashSet<NodeIndex> = blocks.iter().flatten().copied().collect();
        let mut phys_qargs = PhysQargsMap::new(qubit_map);
        for run in runs {
            if run.iter().any(|node| all_block_gates.contains(node)) {
                continue;
            }
            let first_inst_node = run[0];
            let first_inst = dag[first_inst_node].unwrap_operation();
            let first_qubits = phys_qargs.get(dag, first_inst.qubits);

            if run.len() == 1
                && !is_supported(
                    target,
                    basis_gates.as_ref(),
                    first_inst.op.name(),
                    first_qubits,
                )
            {
                // Runs are necessarily single qubit gates so if it has a matrix then it
                // must have a single qubit matrix or it's not a run and the blocks handling above
                // would have covered it
                let matrix = match get_1q_matrix_from_inst(first_inst) {
                    Ok(mat) => mat,
                    Err(_) => continue,
                };
                let unitary_gate = UnitaryGate {
                    array: ArrayType::OneQ(matrix),
                };
                dag.substitute_op(
                    first_inst_node,
                    PackedOperation::from_unitary(Box::new(unitary_gate)),
                    None,
                    None,
                )?;
                continue;
            }
            let mut matrix = ONE_QUBIT_IDENTITY;

            let mut already_in_block = false;
            for node in &run {
                if all_block_gates.contains(node) {
                    already_in_block = true;
                }
                let gate = dag[*node].unwrap_operation();
                let operator = match get_matrix_from_inst(gate) {
                    Ok(mat) => mat,
                    Err(_) => {
                        // Set this to skip this run because we can't compute the matrix of the
                        // operation.
                        already_in_block = true;
                        break;
                    }
                };
                matmul_1q(&mut matrix, operator);
            }
            if already_in_block {
                continue;
            }
            if approx::abs_diff_eq!(aview2(&ONE_QUBIT_IDENTITY), aview2(&matrix)) {
                for node in run {
                    dag.remove_op_node(node);
                }
            } else {
                let array: Matrix2<Complex64> =
                    Matrix2::from_row_iterator(matrix.into_iter().flat_map(|x| x.into_iter()));
                let unitary_gate = UnitaryGate {
                    array: ArrayType::OneQ(array),
                };
                let dag_qubit = dag.get_qargs(first_inst.qubits)[0];
                let mut block_index_map: HashMap<Qubit, usize> = HashMap::with_capacity(1);
                block_index_map.insert(dag_qubit, 0);
                let clbit_pos_map = HashMap::new();
                dag.replace_block(
                    &run,
                    PackedOperation::from_unitary(Box::new(unitary_gate)),
                    None,
                    None,
                    false,
                    &block_index_map,
                    &clbit_pos_map,
                )?;
            }
        }
    }

    Ok(())
}

/// Replaces each block of consecutive gates by a single unitary node.
///
/// This is function is the Rust entry point for the `ConsolidateBlocks` transpiler pass
/// which replaces uninterrupted sequences of gates acting on the same pair of qubits
/// into a [`UnitaryGate`] representing the unitary of that two qubit block if it estimated to
/// to optimize the circuit. This [`UnitaryGate`] subsequently will be synthesized by the
/// unitary synthesis pass into a more optimal subcircuit to replace that block.
///
/// # Arguments
/// * `dag` - The circuit for which we will consolidate gates.
/// * `force_consolidate` - Decides whether to force all consolidations or not.
/// * `approximation_degree` - A float between `[0.0, 1.0]`. Lower approximates more.
/// * `target` - The target representing the backend for which the pass is consolidating.
pub fn run_consolidate_blocks(
    dag: &mut DAGCircuit,
    force_consolidate: bool,
    approximation_degree: Option<f64>,
    target: Option<&Target>,
) -> PyResult<()> {
    let approximation_degree = approximation_degree.unwrap_or(1.0);
    let (decomposer, basis_gate): (Option<DecomposerType>, Option<StandardGate>) =
        if force_consolidate {
            (None, None)
        } else {
            let (decomposer, basis_gate) =
                get_decomposer_and_basis_gate(target, approximation_degree);
            (Some(decomposer), Some(basis_gate))
        };
    let run_in_parallel = getenv_use_multiple_threads();
    let blocks = dag.collect_2q_runs().unwrap();
    if run_in_parallel {
        let consolidations = consolidation_analysis_parallel(
            dag,
            decomposer,
            target,
            None,
            basis_gate.as_ref().map(|x| x.name()).unwrap_or("cx"),
            &blocks,
            None,
            force_consolidate,
        )?;
        for (block, result) in consolidations {
            apply_consolidation(dag, block, result)?;
        }
    } else {
        // In most cases, the qargs in a block will not exceed 2 qubits.
        let mut block_qargs: HashSet<Qubit> = HashSet::with_capacity(2);
        let mut phys_qargs = PhysQargsMap::new(None);
        for block in &blocks {
            let result = should_substitute(
                dag,
                decomposer.as_ref(),
                target,
                None,
                basis_gate.as_ref().map(|x| x.name()).unwrap_or("cx"),
                block,
                &mut block_qargs,
                &mut phys_qargs,
                force_consolidate,
            )?;
            apply_consolidation(dag, block, result)?;
        }
    }
    Ok(())
}

pub fn consolidate_blocks_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_consolidate_blocks))?;
    Ok(())
}

#[cfg(all(test, not(miri)))]
mod test_consolidate_blocks {

    use qiskit_util::IndexMap;

    use qiskit_circuit::{
        PhysicalQubit, Qubit, circuit_data::CircuitData, dag_circuit::DAGCircuit,
        operations::StandardGate,
    };
    use smallvec::smallvec;

    use crate::target::{Qargs, Target};

    use super::run_consolidate_blocks;

    #[test]
    fn test_identity_unitary_is_removed() {
        let circuit: CircuitData = CircuitData::from_standard_gates(
            2,
            [
                (StandardGate::H, smallvec![], smallvec![Qubit(0)]),
                (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                (StandardGate::H, smallvec![], smallvec![Qubit(0)]),
            ],
            0.0.into(),
        )
        .expect("Error while creating the circuit");

        let mut target = Target::default();
        target
            .add_instruction(
                StandardGate::H.into(),
                None,
                None,
                Some(IndexMap::from_iter([
                    (
                        Qargs::Concrete(smallvec![PhysicalQubit(0)]).to_owned(),
                        None,
                    ),
                    (
                        Qargs::Concrete(smallvec![PhysicalQubit(1)]).to_owned(),
                        None,
                    ),
                ])),
            )
            .expect("Error while adding HGate to target");
        target
            .add_instruction(
                StandardGate::CX.into(),
                None,
                None,
                Some(IndexMap::from_iter([
                    (
                        Qargs::Concrete(smallvec![PhysicalQubit(0), PhysicalQubit(1)]).to_owned(),
                        None,
                    ),
                    (
                        Qargs::Concrete(smallvec![PhysicalQubit(1), PhysicalQubit(0)]).to_owned(),
                        None,
                    ),
                ])),
            )
            .expect("Error while adding CXGate to target");

        // Convert the circuit to a DAG.
        let mut circ_as_dag =
            DAGCircuit::from_circuit_data(&circuit, false, None, None, None, None)
                .expect("Error converting circuit to DAG.");
        // Run the pass
        run_consolidate_blocks(&mut circ_as_dag, false, None, Some(&target))
            .expect("Error while running the consolidate blocks pass.");

        let circ_result = CircuitData::from_dag_ref(&circ_as_dag)
            .expect("Error while converting the DAG to a circuit.");

        let data = circ_result.data();
        if !data.is_empty() {
            panic!(
                "The output circuit had {} instructions but all instruction should have cancelled out",
                data.len()
            );
        }
    }
}
