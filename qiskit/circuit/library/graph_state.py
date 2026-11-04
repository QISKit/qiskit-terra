# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Graph State circuit and gate."""

from __future__ import annotations

import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_func


class GraphState(QuantumCircuit):
    r"""Circuit to prepare a graph state.

    Given a graph G = (V, E), with the set of vertices V and the set of edges E,
    the corresponding graph state is defined as

    .. math::

        |G\rangle = \prod_{(a,b) \in E} CZ_{(a,b)} {|+\rangle}^{\otimes V}

    Such a state can be prepared by first preparing all qubits in the :math:`+`
    state, then applying a :math:`CZ` gate for each corresponding graph edge.

    Graph state preparation circuits are Clifford circuits, and thus
    easy to simulate classically. However, by adding a layer of measurements
    in a product basis at the end, there is evidence that the circuit becomes
    hard to simulate [2].

    Reference Circuit:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import GraphState
       from qiskit.visualization.library import _generate_circuit_library_visualization
       import rustworkx as rx
       G = rx.generators.cycle_graph(5)
       circuit = GraphState(rx.adjacency_matrix(G))
       circuit.name = "Graph state"
       _generate_circuit_library_visualization(circuit)

    References:

    [1] M. Hein, J. Eisert, H.J. Briegel, Multi-party Entanglement in Graph States,
        `arXiv:0307130 <https://arxiv.org/pdf/quant-ph/0307130.pdf>`_
    [2] D. Koh, Further Extensions of Clifford Circuits & their Classical Simulation Complexities.
        `arXiv:1512.07892 <https://arxiv.org/pdf/1512.07892.pdf>`_
    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use qiskit.circuit.library.GraphStateGate instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(self, adjacency_matrix: list | np.ndarray) -> None:
        """Create graph state preparation circuit.

        Args:
            adjacency_matrix: input graph as n-by-n list of 0-1 lists

        Raises:
            CircuitError: If adjacency_matrix is not symmetric.

        The circuit prepares a graph state with the given adjacency
        matrix.
        """

        adjacency_matrix = np.asarray(adjacency_matrix)
        if not np.allclose(adjacency_matrix, adjacency_matrix.transpose()):
            raise CircuitError("The adjacency matrix must be symmetric.")

        graph_state_gate = GraphStateGate(adjacency_matrix)
        super().__init__(graph_state_gate.num_qubits, name=f"graph: {adjacency_matrix}")
        self.compose(graph_state_gate, qubits=self.qubits, inplace=True)


class GraphStateGate(Gate):
    r"""A gate representing a graph state.

    Given a graph G = (V, E), with the set of vertices V and the set of edges E,
    the corresponding graph state is defined as

    .. math::

        |G\rangle = \prod_{(a,b) \in E} CZ_{(a,b)} {|+\rangle}^{\otimes V}

    Such a state can be prepared by first preparing all qubits in the :math:`+`
    state, then applying a :math:`CZ` gate for each corresponding graph edge.

    Graph state preparation circuits are Clifford circuits, and thus
    easy to simulate classically. However, by adding a layer of measurements
    in a product basis at the end, there is evidence that the circuit becomes
    hard to simulate [2].

    Reference Circuit:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import GraphStateGate
        import rustworkx as rx

        G = rx.generators.cycle_graph(5)
        circuit = QuantumCircuit(5)
        circuit.append(GraphStateGate(rx.adjacency_matrix(G)), [0, 1, 2, 3, 4])
        circuit.decompose().draw('mpl')

    References:

    [1] M. Hein, J. Eisert, H.J. Briegel, Multi-party Entanglement in Graph States,
    `arXiv:0307130 <https://arxiv.org/pdf/quant-ph/0307130.pdf>`_

    [2] D. Koh, Further Extensions of Clifford Circuits & their Classical Simulation Complexities.
    `arXiv:1512.07892 <https://arxiv.org/pdf/1512.07892.pdf>`_
    """

    def __init__(self, adjacency_matrix: list | np.ndarray) -> None:
        """
        Args:
            adjacency_matrix: input graph as n-by-n list of 0-1 lists

        Raises:
            CircuitError: If adjacency_matrix is not symmetric.

        The gate represents a graph state with the given adjacency matrix.
        """

        adjacency_matrix = np.asarray(adjacency_matrix)
        if not np.allclose(adjacency_matrix, adjacency_matrix.transpose()):
            raise CircuitError("The adjacency matrix must be symmetric.")
        num_qubits = len(adjacency_matrix)

        super().__init__(name="graph_state", num_qubits=num_qubits, params=[adjacency_matrix])

    def _define(self):
        adjacency_matrix = self.adjacency_matrix
        circuit = QuantumCircuit(self.num_qubits)
        circuit.h(range(self.num_qubits))
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if adjacency_matrix[i][j] == 1:
                    circuit.cz(i, j)
        self.definition = circuit

    def validate_parameter(self, parameter):
        """Parameter validation"""
        return parameter

    @property
    def adjacency_matrix(self):
        """Returns the adjacency matrix."""
        return self.params[0]

    def __eq__(self, other):
        return (
            isinstance(other, GraphStateGate)
            and self.num_qubits == other.num_qubits
            and np.all(self.adjacency_matrix == other.adjacency_matrix)
        )
