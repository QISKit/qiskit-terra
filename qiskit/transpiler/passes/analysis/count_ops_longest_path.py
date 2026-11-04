# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Count the operations on the longest path in a DAGCircuit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.basepasses import AnalysisPass


class CountOpsLongestPath(AnalysisPass):
    """Count the operations on the longest path in a :class:`.DAGCircuit`.

    The result is saved in ``property_set['count_ops_longest_path']`` as an integer.
    """

    def run(self, dag: DAGCircuit) -> None:
        """Run the CountOpsLongestPath pass on ``dag``."""
        self.property_set["count_ops_longest_path"] = dag.count_ops_longest_path()
