# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Evaluate how good the layout selection was.

No CX direction is considered.
Saves in `property_set['layout_score']` the sum of distances for each circuit CX.
The lower the number, the better the selection.
Therefore, 0 is a perfect layout selection.
"""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.target import Target


class Layout2qDistance(AnalysisPass):
    """Evaluate how good the layout selection was.

    Saves in ``property_set['layout_score']`` (or the property name in property_name)
    the sum of distances for each circuit CX.
    The lower the number, the better the selection. Therefore, 0 is a perfect layout selection.
    No CX direction is considered.
    """

    def __init__(self, coupling_map, property_name="layout_score"):
        """Layout2qDistance initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): Directed graph representing a coupling map.
            property_name (str): The property name to save the score. Default: layout_score
        """
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        self.property_name = property_name

    def run(self, dag):
        """
        Run the Layout2qDistance pass on `dag`.
        Args:
            dag (DAGCircuit): DAG to evaluate.
        """
        layout = self.property_set["layout"]

        if layout is None:
            return

        if self.coupling_map is None or len(self.coupling_map.graph) == 0:
            self.property_set[self.property_name] = 0
            return

        self.coupling_map.compute_distance_matrix()

        sum_distance = 0

        virtual_physical_map = layout.get_virtual_bits()
        dist_matrix = self.coupling_map.distance_matrix
        for gate in dag.two_qubit_ops():
            physical_q0 = virtual_physical_map[gate.qargs[0]]
            physical_q1 = virtual_physical_map[gate.qargs[1]]

            sum_distance += dist_matrix[physical_q0, physical_q1] - 1

        self.property_set[self.property_name] = sum_distance
