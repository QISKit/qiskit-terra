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

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::{BlocksMode, Qubit, VarsMode};

/// Run the ElidePermutations pass on `dag`.
///
/// Args:
///     dag (DAGCircuit): the DAG to be optimized.
/// Returns:
///     An `Option`: the value of `None` indicates that no optimization was
///     performed and the original `dag` should be used, otherwise it's a
///     tuple consisting of the optimized DAG and the induced qubit permutation.
#[pyfunction]
#[pyo3(name = "run")]
pub fn run_elide_permutations(dag: &DAGCircuit) -> PyResult<Option<(DAGCircuit, Vec<usize>)>> {
    let permutation_gate_names = ["swap".to_string(), "permutation".to_string()];
    let op_counts = dag.get_op_counts();
    if !permutation_gate_names
        .iter()
        .any(|name| op_counts.contains_key(name))
    {
        return Ok(None);
    }
    let mut mapping: Vec<usize> = (0..dag.num_qubits()).collect();

    // note that DAGCircuit::copy_empty_like clones the interners
    let mut new_dag = dag.copy_empty_like_with_capacity(0, 0, VarsMode::Alike, BlocksMode::Keep);
    for node_index in dag.topological_op_nodes(false) {
        if let NodeType::Operation(inst) = &dag[node_index] {
            match inst.op.view() {
                OperationRef::StandardGate(StandardGate::Swap) => {
                    let qargs = dag.get_qargs(inst.qubits);
                    let index0 = qargs[0].index();
                    let index1 = qargs[1].index();
                    mapping.swap(index0, index1);
                }
                OperationRef::PyCustom(
                    gate @ qiskit_circuit::operations::PyInstruction {
                        kind: qiskit_circuit::operations::PyOpKind::Gate,
                        ..
                    },
                ) if gate.name() == "permutation" => {
                    Python::attach(|py| -> PyResult<()> {
                        let params = inst.params_view();
                        if let Param::Obj(ref pyobj) = params[0] {
                            let pyarray: PyReadonlyArray1<i32> = pyobj.extract(py)?;
                            let pattern = pyarray.as_array();

                            let qindices: Vec<usize> = dag
                                .get_qargs(inst.qubits)
                                .iter()
                                .map(|q| q.index())
                                .collect();

                            let new_values: Vec<usize> = (0..qindices.len())
                                .map(|i| mapping[qindices[pattern[i] as usize]])
                                .collect();

                            for i in 0..qindices.len() {
                                mapping[qindices[i]] = new_values[i];
                            }
                        } else {
                            unreachable!();
                        }
                        Ok(())
                    })?;
                }
                _ => {
                    // General instruction
                    let qargs = dag.get_qargs(inst.qubits);
                    let cargs = dag.get_cargs(inst.clbits);
                    let mapped_qargs: Vec<Qubit> = qargs
                        .iter()
                        .map(|q| Qubit::new(mapping[q.index()]))
                        .collect();

                    new_dag.apply_operation_back(
                        inst.op.clone(),
                        &mapped_qargs,
                        cargs,
                        inst.params.as_deref().cloned(),
                        inst.label.as_ref().map(|x| x.as_ref().clone()),
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                }
            }
        } else {
            unreachable!();
        }
    }
    Ok(Some((new_dag, mapping)))
}

pub fn elide_permutations_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_elide_permutations))?;
    Ok(())
}
