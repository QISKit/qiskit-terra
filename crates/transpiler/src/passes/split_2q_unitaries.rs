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
use std::f64::consts::PI;
const PI4: f64 = PI / 4.;

use nalgebra::Matrix2;
use num_complex::Complex64;
use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::{SmallVec, smallvec};

use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder, NodeType, Wire};
use qiskit_circuit::operations::{ArrayType, Operation, OperationRef, Param, UnitaryGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{BlocksMode, Qubit, VarsMode};

use qiskit_synthesis::two_qubit_decompose::{Specialization, TwoQubitWeylDecomposition};

#[pyfunction]
#[pyo3(name = "split_2q_unitaries")]
pub fn run_split_2q_unitaries(
    dag: &mut DAGCircuit,
    requested_fidelity: f64,
    split_swaps: bool,
) -> PyResult<Option<(DAGCircuit, Vec<usize>)>> {
    if !dag.get_op_counts().contains_key("unitary") {
        return Ok(None);
    }
    let nodes: Vec<NodeIndex> = dag.op_node_indices(false).collect();
    let mut has_swaps = false;
    for node in nodes {
        if let NodeType::Operation(inst) = &dag[node] {
            // We only attempt to split UnitaryGate objects, but this could be extended in future
            // -- however we need to ensure that we can compile the resulting single-qubit unitaries
            // to the supported basis gate set.
            let OperationRef::Unitary(unitary_gate) = inst.op.view() else {
                continue;
            };
            if unitary_gate.num_qubits() != 2 {
                continue;
            }
            let temp = dag.get_qargs(inst.qubits);
            let qubits: [Qubit; 2] = [temp[0], temp[1]];
            let matrix = unitary_gate.matrix_view();
            let decomp =
                TwoQubitWeylDecomposition::new_inner(matrix, Some(requested_fidelity), None)?;
            if matches!(decomp.specialization, Specialization::SWAPEquiv) {
                has_swaps = true;
            }
            if matches!(decomp.specialization, Specialization::IdEquiv) {
                let k1r_arr = decomp.k1r_view();
                let k1l_arr = decomp.k1l_view();

                let insert_fn = |edge: Wire| -> (PackedOperation, SmallVec<[Param; 3]>) {
                    let Wire::Qubit(qubit) = edge else {
                        panic!("must only be called on ops with no classical wires");
                    };
                    if qubit == qubits[0] {
                        let mat: Matrix2<Complex64> = [
                            [k1r_arr[[0, 0]], k1r_arr[[1, 0]]],
                            [k1r_arr[[0, 1]], k1r_arr[[1, 1]]],
                        ]
                        .into();
                        let k1r_gate = Box::new(UnitaryGate {
                            array: ArrayType::OneQ(mat),
                        });
                        (PackedOperation::from_unitary(k1r_gate), smallvec![])
                    } else {
                        let mat: Matrix2<Complex64> = [
                            [k1l_arr[[0, 0]], k1l_arr[[1, 0]]],
                            [k1l_arr[[0, 1]], k1l_arr[[1, 1]]],
                        ]
                        .into();

                        let k1l_gate = Box::new(UnitaryGate {
                            array: ArrayType::OneQ(mat),
                        });

                        (PackedOperation::from_unitary(k1l_gate), smallvec![])
                    }
                };
                dag.replace_node_with_1q_ops(node, insert_fn);
                dag.add_global_phase(&Param::Float(decomp.global_phase))?;
            }
        }
    }
    if !split_swaps || !has_swaps {
        return Ok(None);
    }
    // We have swap-like unitaries, so we create a new DAG in a manner similar to
    // The Elide Permutations pass, while also splitting the unitaries to 1-qubit gates
    let mut mapping: Vec<usize> = (0..dag.num_qubits()).collect();
    let rebuilder_callback =
        |new_dag: &mut DAGCircuitBuilder, inst: &PackedInstruction, _node: NodeIndex| {
            if let OperationRef::Unitary(unitary_gate) = inst.op.view() {
                if unitary_gate.num_qubits() == 2 {
                    let decomp = TwoQubitWeylDecomposition::new_inner(
                        unitary_gate.matrix_view(),
                        Some(requested_fidelity),
                        None,
                    )?;
                    if matches!(decomp.specialization, Specialization::SWAPEquiv) {
                        let k1r_arr = decomp.k1r_view();
                        let k1r_mat: Matrix2<Complex64> = [
                            [k1r_arr[[0, 0]], k1r_arr[[1, 0]]],
                            [k1r_arr[[0, 1]], k1r_arr[[1, 1]]],
                        ]
                        .into();
                        let k1r_gate = Box::new(UnitaryGate {
                            array: ArrayType::OneQ(k1r_mat),
                        });
                        let k1l_arr = decomp.k1l_view();
                        let k1l_mat: Matrix2<Complex64> = [
                            [k1l_arr[[0, 0]], k1l_arr[[1, 0]]],
                            [k1l_arr[[0, 1]], k1l_arr[[1, 1]]],
                        ]
                        .into();
                        let k1l_gate = Box::new(UnitaryGate {
                            array: ArrayType::OneQ(k1l_mat),
                        });
                        // perform the virtual swap
                        let qargs = dag.get_qargs(inst.qubits);
                        let index0 = qargs[0].index();
                        let index1 = qargs[1].index();
                        mapping.swap(index0, index1);
                        // now add the two 1-qubit gates
                        new_dag.apply_operation_back(
                            PackedOperation::from_unitary(k1r_gate),
                            &[Qubit::new(mapping[index0])],
                            &[],
                            None,
                            None,
                            #[cfg(feature = "cache_pygates")]
                            None,
                        )?;
                        new_dag.apply_operation_back(
                            PackedOperation::from_unitary(k1l_gate),
                            &[Qubit::new(mapping[index1])],
                            &[],
                            None,
                            None,
                            #[cfg(feature = "cache_pygates")]
                            None,
                        )?;
                        new_dag.add_global_phase(&Param::Float(decomp.global_phase + PI4))?;
                        return Ok(());
                    }
                }
            }
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
                inst.label.as_ref().map(|x| x.to_string()),
                #[cfg(feature = "cache_pygates")]
                inst.py_op.get().cloned(),
            )?;
            Ok(())
        };
    dag.rebuild_dag_with(rebuilder_callback, VarsMode::Alike, BlocksMode::Keep)
        .map(|x| Some((x, mapping)))
}

pub fn split_2q_unitaries_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_split_2q_unitaries))?;
    Ok(())
}
