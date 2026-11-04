# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Combine consecutive Cliffords over the same qubits."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.quantum_info.operators import Clifford


class OptimizeCliffords(TransformationPass):
    """Combine consecutive Cliffords over the same qubits.
    This serves as an example of extra capabilities enabled by storing
    Cliffords natively on the circuit.
    """

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the OptimizeCliffords pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        blocks = []
        prev_node = None
        cur_block = []

        # Iterate over all nodes and collect consecutive Cliffords over the
        # same qubits. In this very first proof-of-concept implementation
        # we require the same ordering of qubits, but this restriction will
        # be shortly removed. An interesting question is whether we may also
        # want to compose Cliffords over different sets of qubits, such as
        # cliff1 over qubits [1, 2, 3] and cliff2 over [2, 3, 4].
        for node in dag.topological_op_nodes():
            if isinstance(node.op, Clifford):
                if prev_node is None:
                    blocks.append(cur_block)
                    cur_block = [node]
                elif prev_node.qargs == node.qargs:
                    cur_block.append(node)
                else:
                    blocks.append(cur_block)
                    cur_block = [node]

                prev_node = node

            else:
                # not a clifford
                if cur_block:
                    blocks.append(cur_block)
                prev_node = None
                cur_block = []

        if cur_block:
            blocks.append(cur_block)

        # Replace every discovered block of cliffords by a single clifford
        # based on the Cliffords' compose function.
        for cur_nodes in blocks:
            # Create clifford functions only out of blocks with at least 2 gates
            if len(cur_nodes) <= 1:
                continue

            wire_pos_map = {qb: ix for ix, qb in enumerate(cur_nodes[0].qargs)}

            # Construct a linear circuit
            cliff = cur_nodes[0].op
            for i, node in enumerate(cur_nodes):
                if i > 0:
                    cliff = Clifford.compose(node.op, cliff, front=True)

            # Replace the block by the composed clifford
            dag.replace_block_with_op(cur_nodes, cliff, wire_pos_map, cycle_check=False)

        return dag
