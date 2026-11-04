# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Remove reset when it is the final instruction on a qubit."""

from qiskit.circuit import Reset, Qubit
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass


class RemoveFinalReset(TransformationPass):
    """Remove reset when it is the final instruction on a qubit wire."""

    def run(self, dag):
        """Run the RemoveFinalReset pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        for output_node in dag.output_map.values():
            if isinstance(output_node.wire, Qubit):
                pred = next(dag.predecessors(output_node))
                if isinstance(pred, DAGOpNode) and isinstance(pred.op, Reset):
                    dag.remove_op_node(pred)
        return dag
