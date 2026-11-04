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

"""Choose a Layout by assigning ``n`` circuit qubits to device qubits ``0, .., n-1``."""

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target


class TrivialLayout(AnalysisPass):
    """Choose a Layout by assigning ``n`` circuit qubits to device qubits ``0, .., n-1``.

    A pass for choosing a Layout of a circuit onto a Coupling graph, using a simple
    round-robin order.

    This pass associates a physical qubit (int) to each virtual qubit
    of the circuit (Qubit) in increasing order.

    Does not assume any ancilla.
    """

    def __init__(self, coupling_map):
        """TrivialLayout initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): directed graph representing a coupling map.

        Raises:
            TranspilerError: if invalid options
        """
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map

    def run(self, dag):
        """Run the TrivialLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: if dag wider than the target backend
        """
        if self.target is not None:
            if dag.num_qubits() > self.target.num_qubits:
                raise TranspilerError("Number of qubits greater than device.")
        elif dag.num_qubits() > self.coupling_map.size():
            raise TranspilerError("Number of qubits greater than device.")
        self.property_set["layout"] = Layout.generate_trivial_layout(
            *(dag.qubits + list(dag.qregs.values()))
        )
