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

"""Return the longest path in a :class:`.DAGCircuit` as a list of DAGNodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.basepasses import AnalysisPass


class DAGLongestPath(AnalysisPass):
    """Return the longest path in a :class:`.DAGCircuit` as a list of
    :class:`.DAGOpNode`\\ s, :class:`.DAGInNode`\\ s, and :class:`.DAGOutNode`\\ s."""

    def run(self, dag: DAGCircuit) -> None:
        """Run the DAGLongestPath pass on ``dag``."""
        self.property_set["dag_longest_path"] = dag.longest_path()
