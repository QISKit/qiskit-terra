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

use crate::TranspilerError;
use crate::target::Target;
use hashbrown::HashSet;
use pyo3::prelude::*;
use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::bit::{QuantumRegister, Register};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{OperationRef, StandardGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{
    Qubit,
    dag_circuit::{DAGCircuit, PyDAGCircuit},
    operations::Operation,
    operations::Param,
};
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::SmallVec;
use std::f64::consts::PI;
//#########################################################################
//              CheckGateDirection analysis pass functions
//#########################################################################

/// Check if the two-qubit gates follow the right direction with respect to the coupling map.
///
/// Args:
///     dag: the DAGCircuit to analyze
///
///     coupling_edges: set of edge pairs representing a directed coupling map, against which gate directionality is checked
///
/// Returns:
///     true iff all two-qubit gates comply with the coupling constraints
#[pyfunction]
#[pyo3(name = "check_gate_direction_coupling")]
fn py_check_direction_coupling_map(
    dag: &PyDAGCircuit,
    coupling_edges: HashSet<[Qubit; 2]>,
) -> PyResult<bool> {
    check_direction_coupling_map(dag.try_read()?, coupling_edges)
}

pub fn check_direction_coupling_map(
    dag: &DAGCircuit,
    coupling_edges: HashSet<[Qubit; 2]>,
) -> PyResult<bool> {
    let coupling_map_check =
        |_: &PackedInstruction, op_args: &[Qubit]| -> bool { coupling_edges.contains(op_args) };

    check_gate_direction(dag, &coupling_map_check, None)
}

/// Check if the two-qubit gates follow the right direction with respect to instructions supported in the given target.
///
/// Args:
///     dag: the DAGCircuit to analyze
///
///     target: the Target against which gate directionality compliance is checked
///
/// Returns:
///     true iff all two-qubit gates comply with the target's coupling constraints
#[pyfunction]
#[pyo3(name = "check_gate_direction_target")]
fn py_check_direction_target(dag: &PyDAGCircuit, target: &Target) -> PyResult<bool> {
    check_direction_target(dag.try_read()?, target)
}

/// Check if the two-qubit gates follow the right direction with respect to instructions supported in the given target.
///
/// # Arguments:
/// * `dag` - the [`DAGCircuit`] to analyze
/// * `target` - the [`Target`] against which gate directionality compliance is checked
///
/// # Returns:
/// `true` iff all two-qubit gates comply with the target's coupling constraints
pub fn check_direction_target(dag: &DAGCircuit, target: &Target) -> PyResult<bool> {
    let target_check = |inst: &PackedInstruction, op_args: &[Qubit]| -> bool {
        let qargs = [
            PhysicalQubit::new(op_args[0].0),
            PhysicalQubit::new(op_args[1].0),
        ];

        target.instruction_supported(inst.op.name(), &qargs, &[], false)
    };

    check_gate_direction(dag, &target_check, None)
}

// The main routine for checking gate directionality.
//
// gate_complies: a function returning true iff the two-qubit gate direction complies with directionality constraints
//
// qubit_mapping: used for mapping the index of a given qubit within an instruction qargs vector to the corresponding qubit index of the
//  original DAGCircuit the pass was called with. This mapping is required since control flow blocks are represented by nested DAGCircuit
//  objects whose instruction qubit indices are relative to the parent DAGCircuit they reside in, thus when we recurse into nested DAGs, we need
//  to carry the mapping context relative to the original DAG.
//  When qubit_mapping is None, the identity mapping is assumed
fn check_gate_direction<T>(
    dag: &DAGCircuit,
    gate_complies: &T,
    qubit_mapping: Option<&[Qubit]>,
) -> PyResult<bool>
where
    T: Fn(&PackedInstruction, &[Qubit]) -> bool,
{
    for (_, packed_inst) in dag.op_nodes(false) {
        let inst_qargs = dag.get_qargs(packed_inst.qubits);

        if let Some(control_flow) = dag.try_view_control_flow(packed_inst) {
            for block in control_flow.blocks() {
                let block_ok = if let Some(mapping) = qubit_mapping {
                    let mapping = inst_qargs // Create a temp mapping for the recursive call
                        .iter()
                        .map(|q| mapping[q.index()])
                        .collect::<Vec<Qubit>>();

                    check_gate_direction(block, gate_complies, Some(&mapping))?
                } else {
                    check_gate_direction(block, gate_complies, Some(inst_qargs))?
                };

                if !block_ok {
                    return Ok(false);
                }
            }
            continue;
        }

        if inst_qargs.len() == 2
            && !match qubit_mapping {
                // Check gate direction based either on a given custom mapping or the identity mapping
                Some(mapping) => gate_complies(
                    packed_inst,
                    &[
                        mapping[inst_qargs[0].index()],
                        mapping[inst_qargs[1].index()],
                    ],
                ),
                None => gate_complies(packed_inst, inst_qargs),
            }
        {
            return Ok(false);
        }
    }

    Ok(true)
}

//#########################################################################
//              GateDirection transformation pass functions
//#########################################################################

/// Try to swap two-qubit gate directions using pre-defined mapping to follow the right direction with respect to the coupling map.
///
/// Args:
///     dag: the DAGCircuit to analyze
///
///     coupling_edges: set of edge pairs representing a directed coupling map, against which gate directionality is checked
///
/// Returns:
///     the transformed DAGCircuit
#[pyfunction]
#[pyo3(name = "fix_gate_direction_coupling")]
fn py_fix_direction_coupling_map(
    dag: &mut PyDAGCircuit,
    coupling_edges: HashSet<[Qubit; 2]>,
) -> PyResult<()> {
    fix_direction_coupling_map(dag.try_write()?, coupling_edges)
}
/// Try to swap two-qubit gate directions using pre-defined mapping to follow the right direction with respect to the coupling map.
///
/// # Arguments:
/// * `dag` - the DAGCircuit to analyze
/// * `coupling_edges` - set of edge pairs representing a directed coupling map, against which gate directionality is checked
///
/// # Returns:
/// The transformed DAGCircuit
pub fn fix_direction_coupling_map(
    dag: &mut DAGCircuit,
    coupling_edges: HashSet<[Qubit; 2]>,
) -> PyResult<()> {
    if coupling_edges.is_empty() {
        return Ok(());
    }

    let coupling_map_check =
        |_: &PackedInstruction, op_args: &[Qubit]| -> bool { coupling_edges.contains(op_args) };

    fix_gate_direction(dag, &coupling_map_check, None)
}

/// Try to swap two-qubit gate directions using pre-defined mapping to follow the right direction with respect to the given target.
///
/// Args:
///     dag: the DAGCircuit to analyze
///
///     coupling_edges: set of edge pairs representing a directed coupling map, against which gate directionality is checked
///
/// Returns:
///     the transformed DAGCircuit
#[pyfunction]
#[pyo3(name = "fix_gate_direction_target")]
fn py_fix_direction_target(dag: &mut PyDAGCircuit, target: &Target) -> PyResult<()> {
    fix_direction_target(dag.try_write()?, target)
}
/// Try to swap two-qubit gate directions using pre-defined mapping to follow the right direction with respect to the given target.
///
/// # Arguments:
/// * `dag` - the [`DAGCircuit`] to analyze
/// * `coupling_edges` - set of edge pairs representing a directed coupling map, against which gate directionality is checked
///
/// # Returns:
/// The transformed [`DAGCircuit`]
pub fn fix_direction_target(dag: &mut DAGCircuit, target: &Target) -> PyResult<()> {
    let target_check = |inst: &PackedInstruction, op_args: &[Qubit]| -> bool {
        let qargs: &[PhysicalQubit] = &[
            PhysicalQubit::new(op_args[0].0),
            PhysicalQubit::new(op_args[1].0),
        ];

        target.instruction_supported(inst.op.name(), qargs, inst.params_view(), false)
    };

    fix_gate_direction(dag, &target_check, None)
}

// The main routine for fixing gate direction. Same parameters as check_gate_direction
fn fix_gate_direction<T>(
    dag: &mut DAGCircuit,
    gate_complies: &T,
    qubit_mapping: Option<&[Qubit]>,
) -> PyResult<()>
where
    T: Fn(&PackedInstruction, &[Qubit]) -> bool,
{
    let mut nodes_to_replace: Vec<(NodeIndex, DAGCircuit)> = Vec::new();
    let mut ops_to_replace: Vec<(NodeIndex, Vec<DAGCircuit>)> = Vec::new();

    for (node, packed_inst) in dag.op_nodes(false) {
        let op_args = dag.get_qargs(packed_inst.qubits);

        if let Some(control_flow) = dag.try_view_control_flow(packed_inst) {
            let blocks = control_flow.blocks();
            let mut blocks_to_replace = Vec::with_capacity(blocks.len());
            for inner_dag in blocks {
                let mut inner_dag = inner_dag.clone();
                if let Some(mapping) = qubit_mapping {
                    let mapping = op_args // Create a temp mapping for the recursive call
                        .iter()
                        .map(|q| mapping[q.index()])
                        .collect::<Vec<Qubit>>();

                    fix_gate_direction(&mut inner_dag, gate_complies, Some(&mapping))?;
                } else {
                    fix_gate_direction(&mut inner_dag, gate_complies, Some(op_args))?;
                };

                blocks_to_replace.push(inner_dag);
            }
            // Store this for replacement outside the dag.op_nodes loop
            ops_to_replace.push((node, blocks_to_replace));

            continue;
        }

        if op_args.len() != 2 {
            continue;
        };

        // Take into account qubit index mapping if we're inside a control-flow block
        let (op_args0, op_args1) = if let Some(mapping) = qubit_mapping {
            (mapping[op_args[0].index()], mapping[op_args[1].index()])
        } else {
            (op_args[0], op_args[1])
        };

        if gate_complies(packed_inst, &[op_args0, op_args1]) {
            continue;
        }

        // If the op has a pre-defined replacement - replace if the other direction is supported otherwise error
        // If no pre-defined replacement for the op - if the other direction is supported error saying no pre-defined rule otherwise error saying op is not supported
        if let OperationRef::StandardGate(std_gate) = packed_inst.op.view() {
            match std_gate {
                StandardGate::CX
                | StandardGate::ECR
                | StandardGate::CZ
                | StandardGate::Swap
                | StandardGate::RXX
                | StandardGate::RYY
                | StandardGate::RZZ
                | StandardGate::RZX => {
                    if gate_complies(packed_inst, &[op_args1, op_args0]) {
                        // Store this for replacement outside the dag.op_nodes loop
                        nodes_to_replace.push((node, replace_dag(std_gate, packed_inst)?));
                        continue;
                    } else {
                        return Err(TranspilerError::new_err(format!(
                            "The circuit requires a connection between physical qubits {:?} for {}",
                            op_args,
                            packed_inst.op.name()
                        )));
                    }
                }
                _ => {}
            }
        }
        // No matching replacement found
        if gate_complies(packed_inst, &[op_args1, op_args0]) {
            return Err(TranspilerError::new_err(format!(
                "{} would be supported on {:?} if the direction was swapped, but no rules are known to do that. {:?} can be automatically flipped.",
                packed_inst.op.name(),
                op_args,
                vec!["cx", "cz", "ecr", "swap", "rzx", "rxx", "ryy", "rzz"]
            )));
            // NOTE: Make sure to update the list of the supported gates if adding more replacements
        } else {
            return Err(TranspilerError::new_err(format!(
                "{} with parameters {:?} is not supported on qubits {:?} in either direction.",
                packed_inst.op.name(),
                packed_inst.params.as_deref(),
                op_args
            )));
        }
    }

    for (node, op_blocks) in ops_to_replace {
        let blocks = {
            let packed_inst = dag[node].unwrap_operation();
            packed_inst
                .params
                .as_deref()
                .map(|b| match b {
                    Parameters::Blocks(blocks) => blocks.clone(),
                    Parameters::Params(_) => panic!("control flow should not have params"),
                })
                .unwrap_or_default()
        };
        for (block, replacement) in blocks.iter().zip(op_blocks) {
            *dag.view_block_mut(*block) = replacement;
        }
    }

    for (node, replacement_dag) in nodes_to_replace {
        dag.substitute_node_with_dag(node, &replacement_dag, None, None, None, None)?;
    }

    Ok(())
}

// Return a replacement DAG for the given standard gate in the supported list
// TODO: optimize it by caching the DAGs of the non-parametric gates and caching and
// mutating upon request the DAGs of the parametric gates
fn replace_dag(std_gate: StandardGate, inst: &PackedInstruction) -> PyResult<DAGCircuit> {
    match std_gate {
        StandardGate::CX => cx_replacement_dag(),
        StandardGate::ECR => ecr_replacement_dag(),
        StandardGate::CZ => cz_replacement_dag(),
        StandardGate::Swap => swap_replacement_dag(),
        StandardGate::RXX => rxx_replacement_dag(inst.params_view()),
        StandardGate::RYY => ryy_replacement_dag(inst.params_view()),
        StandardGate::RZZ => rzz_replacement_dag(inst.params_view()),
        StandardGate::RZX => rzx_replacement_dag(inst.params_view()),
        _ => panic!("Mismatch in supported gates assumption"),
    }
}

//###################################################
// Utility functions to build the replacement dags
//
// TODO: replace this once we have a Rust version of QuantumRegister
#[inline]
fn add_qreg(dag: &mut DAGCircuit, num_qubits: u32) -> PyResult<Vec<Qubit>> {
    let qreg = QuantumRegister::new_owning("q".to_string(), num_qubits);
    dag.add_qreg(qreg.clone())?;
    let mut qargs = Vec::new();

    for qubit in qreg.bits() {
        qargs.push(
            dag.qubits()
                .find(&qubit)
                .expect("Qubit should have been stored in the DAGCircuit"),
        );
    }

    Ok(qargs)
}

#[inline]
fn apply_operation_back(
    dag: &mut DAGCircuit,
    gate: StandardGate,
    qargs: &[Qubit],
    param: Option<SmallVec<[Param; 3]>>,
) -> PyResult<()> {
    dag.apply_operation_back(
        PackedOperation::from_standard_gate(gate),
        qargs,
        &[],
        param.map(Parameters::Params),
        None,
        #[cfg(feature = "cache_pygates")]
        None,
    )?;

    Ok(())
}

fn cx_replacement_dag() -> PyResult<DAGCircuit> {
    let mut new_dag = DAGCircuit::new();
    let qargs = add_qreg(&mut new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(&mut new_dag, StandardGate::H, &[qargs[0]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::H, &[qargs[1]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::CX, &[qargs[1], qargs[0]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::H, &[qargs[0]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::H, &[qargs[1]], None)?;

    Ok(new_dag)
}

fn ecr_replacement_dag() -> PyResult<DAGCircuit> {
    let mut new_dag = DAGCircuit::new();
    new_dag.add_global_phase(&Param::Float(-PI / 2.0))?;
    let qargs = add_qreg(&mut new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(&mut new_dag, StandardGate::S, &[qargs[0]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::SX, &[qargs[0]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::Sdg, &[qargs[0]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::Sdg, &[qargs[1]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::SX, &[qargs[1]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::S, &[qargs[1]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::ECR, &[qargs[1], qargs[0]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::H, &[qargs[0]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::H, &[qargs[1]], None)?;

    Ok(new_dag)
}

fn cz_replacement_dag() -> PyResult<DAGCircuit> {
    let mut new_dag = DAGCircuit::new();
    let qargs = add_qreg(&mut new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(&mut new_dag, StandardGate::CZ, &[qargs[1], qargs[0]], None)?;

    Ok(new_dag)
}

fn swap_replacement_dag() -> PyResult<DAGCircuit> {
    let mut new_dag = DAGCircuit::new();
    let qargs = add_qreg(&mut new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(
        &mut new_dag,
        StandardGate::Swap,
        &[qargs[1], qargs[0]],
        None,
    )?;

    Ok(new_dag)
}

fn rxx_replacement_dag(param: &[Param]) -> PyResult<DAGCircuit> {
    let mut new_dag = DAGCircuit::new();
    let qargs = add_qreg(&mut new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(
        &mut new_dag,
        StandardGate::RXX,
        &[qargs[1], qargs[0]],
        Some(SmallVec::from(param)),
    )?;

    Ok(new_dag)
}

fn ryy_replacement_dag(param: &[Param]) -> PyResult<DAGCircuit> {
    let mut new_dag = DAGCircuit::new();
    let qargs = add_qreg(&mut new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(
        &mut new_dag,
        StandardGate::RYY,
        &[qargs[1], qargs[0]],
        Some(SmallVec::from(param)),
    )?;

    Ok(new_dag)
}

fn rzz_replacement_dag(param: &[Param]) -> PyResult<DAGCircuit> {
    let mut new_dag = DAGCircuit::new();
    let qargs = add_qreg(&mut new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(
        &mut new_dag,
        StandardGate::RZZ,
        &[qargs[1], qargs[0]],
        Some(SmallVec::from(param)),
    )?;

    Ok(new_dag)
}

fn rzx_replacement_dag(param: &[Param]) -> PyResult<DAGCircuit> {
    let mut new_dag = DAGCircuit::new();
    let qargs = add_qreg(&mut new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(&mut new_dag, StandardGate::H, &[qargs[0]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::H, &[qargs[1]], None)?;
    apply_operation_back(
        &mut new_dag,
        StandardGate::RZX,
        &[qargs[1], qargs[0]],
        Some(SmallVec::from(param)),
    )?;
    apply_operation_back(&mut new_dag, StandardGate::H, &[qargs[0]], None)?;
    apply_operation_back(&mut new_dag, StandardGate::H, &[qargs[1]], None)?;

    Ok(new_dag)
}

pub fn gate_direction_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_check_direction_coupling_map))?;
    m.add_wrapped(wrap_pyfunction!(py_check_direction_target))?;
    m.add_wrapped(wrap_pyfunction!(py_fix_direction_coupling_map))?;
    m.add_wrapped(wrap_pyfunction!(py_fix_direction_target))?;
    Ok(())
}
