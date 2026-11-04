# This code is part of Qiskit.
#
# (C) Copyright IBM 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Set the ``layout`` property to the given layout."""
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import InvalidLayoutError
from qiskit.transpiler.layout import Layout


class SetLayout(AnalysisPass):
    """Set the ``layout`` property to the given layout.

    This pass associates a physical qubit (int) to each virtual qubit
    of the circuit (Qubit) in increasing order.
    """

    def __init__(self, layout):
        """SetLayout initializer.

        Args:
            layout (Layout or List[int]): the layout to set. It can be:

                * a :class:`Layout` instance: sets that layout.
                * a list of integers: takes the index in the list as the physical position in which the
                  virtual qubit is going to be mapped.

        """
        super().__init__()
        self.layout = layout

    def run(self, dag):
        """Run the SetLayout pass on ``dag``.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: the original DAG.
        """
        if isinstance(self.layout, list):
            if len(self.layout) != len(dag.qubits):
                raise InvalidLayoutError(
                    "The length of the layout is different than the size of the "
                    f"circuit: {len(self.layout)} <> {len(dag.qubits)}"
                )
            if len(set(self.layout)) != len(self.layout):
                raise InvalidLayoutError(
                    f"The provided layout {self.layout} contains duplicate qubits"
                )
            layout = Layout({phys: dag.qubits[i] for i, phys in enumerate(self.layout)})
        elif isinstance(self.layout, Layout):
            layout = self.layout.copy()
        elif self.layout is None:
            layout = None
        else:
            raise InvalidLayoutError(
                f"SetLayout was initialized with the layout type: {type(self.layout)}"
            )
        self.property_set["layout"] = layout
        return dag
