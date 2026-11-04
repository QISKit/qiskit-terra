# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Choose a Layout by finding the most connected subset of qubits."""


import numpy as np
import rustworkx

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
from qiskit._accelerate import disjoint_utils

from qiskit._accelerate.dense_layout import best_subset


class DenseLayout(AnalysisPass):
    """Choose a Layout by finding the most connected subset of qubits.

    This pass associates a physical qubit (int) to each virtual qubit
    of the circuit (Qubit).

    Note:
        Even though a ``'layout'`` is not strictly a property of the DAG,
        in the transpiler architecture it is best passed around between passes
        by being set in ``property_set``.
    """

    def __init__(self, coupling_map=None, target=None):
        """DenseLayout initializer.

        Args:
            coupling_map (Coupling): directed graph representing a coupling map.
            target (Target): A target representing the target backend.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.target = target
        self.adjacency_matrix = None
        if target is not None:
            self.coupling_map = target.build_coupling_map()

    def run(self, dag):
        """Run the DenseLayout pass on `dag`.

        Pick a convenient layout depending on the best matching
        qubit connectivity, and set the property `layout`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        if self.coupling_map is None:
            raise TranspilerError(
                "A coupling_map or target with constrained qargs is necessary to run the pass."
            )
        if self.target is not None:
            layout_components = disjoint_utils.run_pass_over_connected_components(
                dag,
                self.target,
                self._inner_run,
            )
            if layout_components is None:
                target = Target.from_configuration(
                    basis_gates=["u", "cx"], coupling_map=self.coupling_map
                )
                layout_components = disjoint_utils.run_pass_over_connected_components(
                    dag,
                    target,
                    self._inner_run,
                )
        else:
            target = Target.from_configuration(
                basis_gates=["u", "cx"], coupling_map=self.coupling_map
            )
            layout_components = disjoint_utils.run_pass_over_connected_components(
                dag,
                target,
                self._inner_run,
            )
        layout_mapping = {}
        for component in layout_components:
            layout_mapping.update(component)
        layout = Layout(layout_mapping)
        for qreg in dag.qregs.values():
            layout.add_register(qreg)
        self.property_set["layout"] = layout

    def _inner_run(self, dag, coupling_map):
        num_dag_qubits = len(dag.qubits)
        if num_dag_qubits > coupling_map.size():
            raise TranspilerError("Number of qubits greater than device.")
        num_cx = 0
        num_meas = 0

        if self.target is not None:
            num_cx = 1
            num_meas = 1
        else:
            # Get avg number of cx and meas per qubit
            ops = dag.count_ops(recurse=True)
            if "cx" in ops.keys():
                num_cx = ops["cx"]
            if "measure" in ops.keys():
                num_meas = ops["measure"]

        best_sub = self._best_subset(num_dag_qubits, num_meas, num_cx, coupling_map)
        layout_mapping = {
            qubit: coupling_map.graph[int(best_sub[i])] for i, qubit in enumerate(dag.qubits)
        }
        return layout_mapping

    def _best_subset(self, num_qubits, num_meas, num_cx, coupling_map):
        """Computes the qubit mapping with the best connectivity.

        Args:
            num_qubits (int): Number of subset qubits to consider.
            num_meas (int): The number of measure measurement operations in the circuit.
            num_cx (int): The number of CXGates that are in the circuit
            coupling_map (CouplingMap): The coupling map representing the connectivity of
                the QPU.

        Returns:
            ndarray: Array of qubits to use for best connectivity mapping.
        """
        from scipy.sparse import coo_matrix, csgraph

        if num_qubits == 1:
            return np.array([0])
        if num_qubits == 0:
            return []

        adjacency_matrix = rustworkx.adjacency_matrix(coupling_map.graph)
        reverse_index_map = {v: k for k, v in enumerate(coupling_map.graph.nodes())}

        error_mat, use_error = _build_error_matrix(
            coupling_map.size(),
            reverse_index_map,
            target=self.target,
        )

        rows, cols, best_map = best_subset(
            num_qubits,
            adjacency_matrix,
            num_meas,
            num_cx,
            use_error,
            coupling_map.is_symmetric,
            error_mat,
        )
        data = [1] * len(rows)
        sp_sub_graph = coo_matrix((data, (rows, cols)), shape=(num_qubits, num_qubits)).tocsr()
        perm = csgraph.reverse_cuthill_mckee(sp_sub_graph)
        best_map = best_map[perm]
        return best_map


def _build_error_matrix(num_qubits, qubit_map, target=None):
    error_mat = np.zeros((num_qubits, num_qubits))
    use_error = False
    if target is not None and target.qargs is not None:
        for qargs in target.qargs:
            # Ignore gates over 2q. DenseLayout only works with 2q
            if len(qargs) > 2:
                continue
            error = 0.0
            ops = target.operation_names_for_qargs(qargs)
            for op in ops:
                props = target[op].get(qargs, None)
                if props is not None and props.error is not None:
                    # Use max error rate to represent operation error
                    # on a qubit(s). If there is more than 1 operation available
                    # we don't know what will be used on the qubits eventually
                    # so we take the highest error operation as a proxy for
                    # the possible worst case.
                    error = max(error, props.error)
            max_error = error
            if any(qubit not in qubit_map for qubit in qargs):
                continue
            # TODO: Factor in T1 and T2 to error matrix after #7736
            if len(qargs) == 1:
                qubit = qubit_map[qargs[0]]
                error_mat[qubit][qubit] = max_error
                use_error = True
            elif len(qargs) == 2:
                error_mat[qubit_map[qargs[0]]][qubit_map[qargs[1]]] = max_error
                use_error = True
    return error_mat, use_error
