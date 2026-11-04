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

"""Optimize sequences of single-qubit Clifford+T gates."""

from collections.abc import Sequence
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.transpiler.basepasses import TransformationPass
from qiskit._accelerate.optimize_clifford_t import optimize_clifford_t

SKIPPED_NAMES = {
    "measure",
    "box",
    "reset",
    "delay",
    "if_else",
    "switch_case",
    "while_loop",
    "for_loop",
}


class OptimizeCliffordT(TransformationPass):
    """
    Optimize sequences of consecutive Clifford+T gates.

    This pass rewrites maximal chains of consecutive single-qubit
    Clifford+T gates, reducing each chain to an equivalent sequence
    that uses the minimum possible number of T gates.

    For a chain of length :math:`m`, the pass runs in linear time,
    :math:`O(m)`.
    """

    def __init__(self, basis_gates: Sequence[str] | None = None):
        if basis_gates is not None:
            name_map = get_standard_gate_name_mapping()
            std_gates = []
            for name in basis_gates:
                if name in SKIPPED_NAMES:
                    continue
                if (gate := name_map.get(name, None)) is None:
                    raise ValueError("Invalid basis gate: %s", name)
                if (std_gate := gate._standard_gate) is None:
                    raise ValueError("Non-standard gate passed: %s", name)

                std_gates.append(std_gate)

            self.basis_gates = std_gates
        else:
            self.basis_gates = None

        super().__init__()

    def run(self, dag: DAGCircuit):
        """
        Run the OptimizeCliffordT pass on `dag`.

        Args:
            dag: The directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """

        optimize_clifford_t(dag, self.basis_gates)
        return dag
