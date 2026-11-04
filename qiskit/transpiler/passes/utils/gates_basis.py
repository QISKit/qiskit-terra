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

"""Check if all gates in the DAGCircuit are in the specified basis gates."""

from qiskit.transpiler.basepasses import AnalysisPass

from qiskit._accelerate.gates_in_basis import (
    any_gate_missing_from_basis,
    any_gate_missing_from_target,
)


class GatesInBasis(AnalysisPass):
    """Check if all gates in a DAG are in a given set of gates"""

    def __init__(self, basis_gates=None, target=None):
        """Initialize the GatesInBasis pass.

        Args:
            basis_gates (list): The list of strings representing the set of basis gates.
            target (Target): The target representing the backend. If specified
                this will be used instead of the ``basis_gates`` parameter
        """
        super().__init__()
        self._basis_gates = None
        if basis_gates is not None:
            self._basis_gates = set(basis_gates).union(
                {"measure", "reset", "barrier", "snapshot", "delay", "store"}
            )
        self._target = target

    def run(self, dag):
        """Run the GatesInBasis pass on `dag`."""
        if self._basis_gates is None and self._target is None:
            self.property_set["all_gates_in_basis"] = True
            return
        if self._target is not None:
            gates_out_of_basis = any_gate_missing_from_target(dag, self._target)
        else:
            gates_out_of_basis = any_gate_missing_from_basis(dag, self._basis_gates)
        self.property_set["all_gates_in_basis"] = not gates_out_of_basis
