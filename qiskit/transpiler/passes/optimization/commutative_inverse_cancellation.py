# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cancel pairs of inverse gates exploiting commutation relations."""
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit import CircuitError

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.transpiler.basepasses import TransformationPass


class CommutativeInverseCancellation(TransformationPass):
    """Cancel pairs of inverse gates exploiting commutation relations."""

    def __init__(self, matrix_based: bool = False, max_qubits: int = 4):
        """
        Args:
            matrix_based: If ``True``, uses matrix representations to check whether two
                operations are inverse of each other. This makes the checks more powerful,
                and, in addition, allows canceling pairs of operations that are inverse up to a
                phase, while updating the global phase of the circuit accordingly.
                Generally this leads to more reductions at the expense of increased runtime.
            max_qubits: Limits the number of qubits in matrix-based commutativity and
                inverse checks.
        """
        self._matrix_based = matrix_based
        self._max_qubits = max_qubits
        self.comm_checker = scc
        super().__init__()

    def _skip_node(self, node):
        """Returns True if we should skip this node for the analysis."""
        if not isinstance(node, DAGOpNode):
            return True

        # We are currently taking an over-conservative approach with respect to which nodes
        # can be inverses of which other nodes, and do not allow reductions for barriers, measures,
        # conditional gates or parameterized gates. Possibly both this and commutativity
        # checking can be extended to cover additional cases.
        if getattr(node.op, "_directive", False) or node.name in {"measure", "reset", "delay"}:
            return True
        if getattr(node, "condition", None):
            return True
        if node.is_control_flow():
            return True
        if getattr(node.op, "is_parameterized", None) is not None and node.op.is_parameterized():
            return True
        return False

    def _get_inverse(self, op):
        """
        Returns an inverse of the given op, or ``None`` if the inverse
        does not exist or is too expensive to compute.
        """
        # Some instructions (such as Initialize) cannot be inverted
        try:
            inverse = op.inverse()
        except (CircuitError, AttributeError):
            inverse = None
        return inverse

    def _check_equal_upto_phase(self, op1, op2, matrix_based):
        """
        Checks whether op1 and op2 are equal up to a phase, that is whether
        ``op2 = e^{i * d} op1`` for some phase difference ``d``.

        If this is the case, we can replace ``op2 * op1^{-1}`` by ``e^{i * d} I``.

        The output is a tuple representing whether the two ops
        are equal up to a phase and that phase difference.
        """
        phase_difference = 0
        are_equal = op1 == op2
        if not are_equal and matrix_based:
            mat1 = Operator(op1).data
            mat2 = Operator(op2).data
            props = {}
            are_equal = matrix_equal(mat1, mat2, ignore_phase=True, props=props)
            if are_equal:
                # mat2 = e^{i * phase_difference} mat1
                phase_difference = props["phase_difference"]
        return are_equal, phase_difference

    def run(self, dag: DAGCircuit):
        """
        Run the CommutativeInverseCancellation pass on `dag`.

        Args:
            dag: the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        topo_sorted_nodes = list(dag.topological_op_nodes())

        circ_size = len(topo_sorted_nodes)

        removed = [False for _ in range(circ_size)]

        phase_update = 0

        for idx1 in range(circ_size):
            # if the node should be skipped or does not have an inverse, continue
            if self._skip_node(topo_sorted_nodes[idx1]):
                continue
            if (op1_inverse := self._get_inverse(topo_sorted_nodes[idx1].op)) is None:
                continue

            matrix_based = (
                self._matrix_based and topo_sorted_nodes[idx1].num_qubits <= self._max_qubits
            )

            matched_idx2 = -1

            for idx2 in range(idx1 - 1, -1, -1):
                if removed[idx2]:
                    continue

                if (
                    not self._skip_node(topo_sorted_nodes[idx2])
                    and topo_sorted_nodes[idx2].qargs == topo_sorted_nodes[idx1].qargs
                    and topo_sorted_nodes[idx2].cargs == topo_sorted_nodes[idx1].cargs
                ):
                    is_inverse, phase = self._check_equal_upto_phase(
                        op1_inverse,
                        topo_sorted_nodes[idx2].op,
                        matrix_based,
                    )
                    if is_inverse:
                        phase_update += phase
                        matched_idx2 = idx2
                        break

                if not self.comm_checker.commute_nodes(
                    topo_sorted_nodes[idx1],
                    topo_sorted_nodes[idx2],
                    max_num_qubits=self._max_qubits,
                ):
                    break

            if matched_idx2 != -1:
                removed[idx1] = True
                removed[matched_idx2] = True

        for idx in range(circ_size):
            if removed[idx]:
                dag.remove_op_node(topo_sorted_nodes[idx])

        if phase_update != 0:
            dag.global_phase += phase_update

        return dag
