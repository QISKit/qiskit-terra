# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of graph state circuits."""

import unittest
import numpy as np

from qiskit.circuit import Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import GraphState, GraphStateGate
from qiskit.quantum_info import Clifford, Operator
from test import QiskitTestCase


class TestGraphStateLibrary(QiskitTestCase):
    """Test the graph state circuit."""

    def assertGraphStateIsCorrect(self, adjacency_matrix, graph_state):
        """Check the stabilizers of the graph state circuit/gate against the expected stabilizers.
        Based on https://arxiv.org/pdf/quant-ph/0307130.pdf, Eq. (6).
        """
        if isinstance(graph_state, Gate):
            cliff = Clifford(graph_state.definition)
        else:
            cliff = Clifford(graph_state)

        stabilizers = [stabilizer[1:] for stabilizer in cliff.to_labels(mode="S")]

        expected_stabilizers = []  # keep track of all expected stabilizers
        num_vertices = len(adjacency_matrix)
        for vertex_a in range(num_vertices):
            stabilizer = [None] * num_vertices  # Paulis must be put into right place
            for vertex_b in range(num_vertices):
                if vertex_a == vertex_b:  # self-connection --> 'X'
                    stabilizer[vertex_a] = "X"
                elif adjacency_matrix[vertex_a][vertex_b] != 0:  # vertices connected --> 'Z'
                    stabilizer[vertex_b] = "Z"
                else:  # else --> 'I'
                    stabilizer[vertex_b] = "I"

            # need to reverse for Qiskit's tensoring order
            expected_stabilizers.append("".join(stabilizer)[::-1])

        self.assertListEqual(expected_stabilizers, stabilizers)

    def test_graph_state_circuit(self):
        """Verify the GraphState by checking if the circuit has the expected stabilizers."""
        adjacency_matrix = [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
        with self.assertWarns(DeprecationWarning):
            graph_state = GraphState(adjacency_matrix)
        self.assertGraphStateIsCorrect(adjacency_matrix, graph_state)

    def test_non_symmetric_circuit_raises(self):
        """Test that adjacency matrix is required to be symmetric."""
        adjacency_matrix = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
        with self.assertRaises(CircuitError):
            with self.assertWarns(DeprecationWarning):
                GraphState(adjacency_matrix)

    def test_graph_state_gate(self):
        """Verify correctness of GraphStatGate by checking that the gate's definition circuit
        has the expected stabilizers.
        """
        adjacency_matrix = [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
        graph_state = GraphStateGate(adjacency_matrix)
        self.assertGraphStateIsCorrect(adjacency_matrix, graph_state)

    def test_non_symmetric_gate_raises(self):
        """Test that adjacency matrix is required to be symmetric."""
        adjacency_matrix = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
        with self.assertRaises(CircuitError):
            GraphStateGate(adjacency_matrix)

    def test_circuit_and_gate_equivalence(self):
        """Test that GraphState-circuit and GraphStateGate yield equal operators."""
        adjacency_matrix = [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
        graph_state_gate = GraphStateGate(adjacency_matrix)
        with self.assertWarns(DeprecationWarning):
            graph_state_circuit = GraphState(adjacency_matrix)
        self.assertEqual(Operator(graph_state_gate), Operator(graph_state_circuit))

    def test_adjacency_matrix(self):
        """Test GraphStateGate's adjacency_matrix method."""
        adjacency_matrix = [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
        graph_state_gate = GraphStateGate(adjacency_matrix)
        self.assertTrue(np.all(graph_state_gate.adjacency_matrix == adjacency_matrix))

    def test_equality(self):
        """Test GraphStateGate's equality method."""
        mat1 = [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
        mat2 = [
            [0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
        with self.subTest("equal"):
            gate1 = GraphStateGate(mat1)
            gate2 = GraphStateGate(mat1)
            self.assertEqual(gate1, gate2)
        with self.subTest("not equal"):
            gate1 = GraphStateGate(mat1)
            gate2 = GraphStateGate(mat2)
            self.assertNotEqual(gate1, gate2)


if __name__ == "__main__":
    unittest.main()
