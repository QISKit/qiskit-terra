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

"""Calculate the size of a DAG circuit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.basepasses import AnalysisPass


class Size(AnalysisPass):
    """Calculate the size of a DAG circuit.

    The result is saved in ``property_set['size']`` as an integer.
    """

    def __init__(self, *, recurse: bool = False) -> None:
        """
        Args:
            recurse: whether to allow recursion into control flow.  If this is ``False`` (default),
                the pass will throw an error when control flow is present, to avoid returning a
                number with little meaning.
        """
        super().__init__()
        self.recurse = recurse

    def run(self, dag: DAGCircuit) -> None:
        """Run the Size pass on ``dag``."""
        self.property_set["size"] = dag.size(recurse=self.recurse)
