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

"""Check if the gates follow the right direction with respect to the coupling map."""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit._accelerate.gate_direction import (
    check_gate_direction_coupling,
    check_gate_direction_target,
)


class CheckGateDirection(AnalysisPass):
    """Check if the two-qubit gates follow the right direction with
    respect to the coupling map.
    """

    def __init__(self, coupling_map, target=None):
        """CheckGateDirection initializer.

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            target (Target): The backend target to use for this pass. If this is specified
                it will be used instead of the coupling map
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.target = target

    def run(self, dag):
        """Run the CheckGateDirection pass on `dag`.

        If `dag` is mapped and the direction is correct the property
        `is_direction_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to check.
        """
        # Only use "check_gate_direction_target" if a target exists and target.operation_names
        # is not empty, else use "check_gate_direction_coupling".
        if self.target is None:
            self.property_set["is_direction_mapped"] = check_gate_direction_coupling(
                dag, set(self.coupling_map.get_edges())
            )
        elif len(self.target.operation_names) == 0:
            # A  _FakeTarget path, no basis gates, just use the coupling map
            self.property_set["is_direction_mapped"] = check_gate_direction_coupling(
                dag, set(self.target.build_coupling_map().get_edges())
            )
        else:
            self.property_set["is_direction_mapped"] = check_gate_direction_target(dag, self.target)
