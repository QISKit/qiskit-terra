# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Commutative Optimization transpiler pass."""


from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc

from qiskit._accelerate import commutative_optimization

from qiskit.transpiler.passes.utils.control_flow import trivial_recurse


class CommutativeOptimization(TransformationPass):
    """
    Cancel/merge gates exploiting commutativity relations.

    Specifically, the pass:

    * Cancels pairs of inverse gates, including pairs that are
      inverse up to a global phase (adjusting the global phase
      if necessary).
    * Attempts to merge consecutive gates when possible, for example
      sequences of RZ-gates, RX-gates, Pauli rotations, and so on.

    This pass unifies and extends the functionality of both
    :class:`.CommutativeCancellation` and
    :class:`.CommutativeInverseCancellation`.
    """

    def __init__(self, approximation_degree: float = 1.0, matrix_max_num_qubits: int = 0):
        """
        Args:
            approximation_degree: the threshold used in the average gate fidelity
                computation to decide whether pairs of gates can be considered as
                canceling or commuting.
            matrix_max_num_qubits: Upper-bound on the number of qubits for the matrix-based
                commutativity and inverse checks.
        """
        super().__init__()
        self.commutation_checker = scc.cc
        self.approximation_degree = approximation_degree
        self.matrix_max_num_qubits = matrix_max_num_qubits

    @trivial_recurse
    def run(self, dag):
        """Run the CommutativeOptimization pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        result = commutative_optimization.commutative_optimization(
            dag, self.commutation_checker, self.approximation_degree, self.matrix_max_num_qubits
        )

        # If the pass did not do anything, the result is None
        if result is None:
            return dag

        return result
