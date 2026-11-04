# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Synthesize LinearFunctions."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.circuit.library import PermutationGate
from qiskit.circuit.exceptions import CircuitError


class LinearFunctionsToPermutations(TransformationPass):
    """Promotes linear functions to permutations when possible."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the LinearFunctionsToPermutations pass on `dag`.
        Args:
            dag: input dag.
        Returns:
            Output dag with LinearFunctions synthesized.
        """

        for node in dag.named_nodes("linear_function"):
            try:
                pattern = node.op.permutation_pattern()
            except CircuitError:
                continue

            permutation = PermutationGate(pattern)
            dag.substitute_node(node, permutation)
        return dag
