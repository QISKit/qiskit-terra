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


import unittest

import numpy as np
import rustworkx as rx

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.exceptions import CouplingError
from qiskit.utils import optionals
from test import QiskitTestCase

from ..visualization.visualization import QiskitVisualizationTestCase, path_to_diagram_reference
from ..legacy_cmaps import RUESCHLIKON_CMAP


class CouplingTest(QiskitTestCase):
    def test_empty_coupling_class(self):
        coupling = CouplingMap()
        self.assertEqual(0, coupling.size())
        self.assertEqual([], coupling.physical_qubits)
        self.assertEqual([], coupling.get_edges())
        self.assertFalse(coupling.is_connected())
        self.assertEqual("", str(coupling))

    def test_coupling_str(self):
        coupling_list = [[0, 1], [0, 2], [1, 2]]
        coupling = CouplingMap(coupling_list)
        expected = "[[0, 1], [0, 2], [1, 2]]"
        self.assertEqual(expected, str(coupling))

    def test_coupling_distance(self):
        coupling_list = [(0, 1), (0, 2), (1, 2)]
        coupling = CouplingMap(coupling_list)
        self.assertTrue(coupling.is_connected())
        physical_qubits = coupling.physical_qubits
        result = coupling.distance(physical_qubits[0], physical_qubits[1])
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_add_physical_qubits(self):
        coupling = CouplingMap()
        self.assertEqual("", str(coupling))
        coupling.add_physical_qubit(0)
        self.assertEqual([0], coupling.physical_qubits)
        self.assertEqual("", str(coupling))

    def test_add_physical_qubits_not_int(self):
        coupling = CouplingMap()
        self.assertRaises(CouplingError, coupling.add_physical_qubit, "q")

    def test_add_edge(self):
        coupling = CouplingMap()
        self.assertEqual("", str(coupling))
        coupling.add_edge(0, 1)
        expected = "[[0, 1]]"
        self.assertEqual(expected, str(coupling))

    def test_neighbors(self):
        """Test neighboring qubits are found correctly."""
        coupling = CouplingMap([[0, 1], [0, 2], [1, 0]])

        physical_qubits = coupling.physical_qubits
        self.assertEqual(set(coupling.neighbors(physical_qubits[0])), {1, 2})
        self.assertEqual(set(coupling.neighbors(physical_qubits[1])), {0})
        self.assertEqual(set(coupling.neighbors(physical_qubits[2])), set())

    def test_distance_error(self):
        """Test distance between unconnected physical_qubits."""
        graph = CouplingMap()
        graph.add_physical_qubit(0)
        graph.add_physical_qubit(1)
        self.assertRaises(CouplingError, graph.distance, 0, 1)

    def test_distance_self_loop(self):
        """Test distance between the same physical qubit."""
        graph = CouplingMap()
        graph.add_physical_qubit(0)
        graph.add_physical_qubit(1)
        self.assertEqual(0.0, graph.distance(0, 0))

    def test_init_with_couplinglist(self):
        coupling_list = [[0, 1], [1, 2]]
        coupling = CouplingMap(coupling_list)

        qubits_expected = [0, 1, 2]
        edges_expected = [(0, 1), (1, 2)]

        self.assertEqual(coupling.physical_qubits, qubits_expected)
        self.assertEqual(coupling.get_edges(), edges_expected)
        self.assertEqual(2, coupling.distance(0, 2))

    def test_successful_reduced_map(self):
        """Generate a reduced map"""
        cmap = RUESCHLIKON_CMAP
        coupling_map = CouplingMap(cmap)
        out = coupling_map.reduce([12, 11, 10, 9]).get_edges()
        ans = [(1, 2), (3, 2), (0, 1)]
        self.assertEqual(set(out), set(ans))

    def test_bad_reduced_map(self):
        """Generate disconnected reduced map"""
        cmap = RUESCHLIKON_CMAP
        coupling_map = CouplingMap(cmap)
        with self.assertRaises(CouplingError):
            coupling_map.reduce([12, 11, 10, 3])

    def test_disconnected_reduced_map_allowed(self):
        """Generate disconnected reduced map but do not error"""
        cmap = RUESCHLIKON_CMAP
        coupling_map = CouplingMap(cmap)
        reduced_map = coupling_map.reduce([12, 11, 10, 3], check_if_connected=False)
        reduced_edges = reduced_map.get_edges()
        qubits_expected = [0, 1, 2, 3]
        edges_expected = [(0, 1), (1, 2)]
        self.assertEqual(qubits_expected, reduced_map.physical_qubits)
        self.assertEqual(set(reduced_edges), set(edges_expected))

    def test_symmetric_small_true(self):
        coupling_list = [[0, 1], [1, 0]]
        coupling = CouplingMap(coupling_list)

        self.assertTrue(coupling.is_symmetric)

    def test_symmetric_big_false(self):
        coupling_list = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [9, 8],
            [9, 10],
            [7, 8],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]
        coupling = CouplingMap(coupling_list)

        self.assertFalse(coupling.is_symmetric)

    def test_make_symmetric(self):
        coupling_list = [[0, 1], [0, 2]]
        coupling = CouplingMap(coupling_list)

        coupling.make_symmetric()
        edges = coupling.get_edges()

        self.assertEqual(set(edges), {(0, 1), (0, 2), (2, 0), (1, 0)})

    def test_full_factory(self):
        coupling = CouplingMap.from_full(4)
        edges = coupling.get_edges()
        expected = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_line_factory(self):
        coupling = CouplingMap.from_line(4)
        edges = coupling.get_edges()
        expected = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
        self.assertEqual(set(edges), set(expected))

    def test_grid_factory(self):
        coupling = CouplingMap.from_grid(2, 3)
        edges = coupling.get_edges()
        expected = [
            (0, 3),
            (0, 1),
            (3, 0),
            (3, 4),
            (1, 0),
            (1, 4),
            (1, 2),
            (4, 1),
            (4, 3),
            (4, 5),
            (2, 1),
            (2, 5),
            (5, 2),
            (5, 4),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_grid_factory_unidirectional(self):
        coupling = CouplingMap.from_grid(2, 3, bidirectional=False)
        edges = coupling.get_edges()
        expected = [(0, 3), (0, 1), (3, 4), (1, 4), (1, 2), (4, 5), (2, 5)]
        self.assertEqual(set(edges), set(expected))

    def test_heavy_hex_factory(self):
        coupling = CouplingMap.from_heavy_hex(3, bidirectional=False)
        edges = coupling.get_edges()
        expected = [
            (0, 9),
            (0, 13),
            (1, 13),
            (1, 14),
            (2, 14),
            (3, 9),
            (3, 15),
            (4, 15),
            (4, 16),
            (5, 12),
            (5, 16),
            (6, 17),
            (7, 17),
            (7, 18),
            (8, 12),
            (8, 18),
            (10, 14),
            (10, 16),
            (11, 15),
            (11, 17),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_heavy_hex_factory_bidirectional(self):
        coupling = CouplingMap.from_heavy_hex(3, bidirectional=True)
        edges = coupling.get_edges()
        expected = [
            (0, 9),
            (0, 13),
            (1, 13),
            (1, 14),
            (2, 14),
            (3, 9),
            (3, 15),
            (4, 15),
            (4, 16),
            (5, 12),
            (5, 16),
            (6, 17),
            (7, 17),
            (7, 18),
            (8, 12),
            (8, 18),
            (9, 0),
            (9, 3),
            (10, 14),
            (10, 16),
            (11, 15),
            (11, 17),
            (12, 5),
            (12, 8),
            (13, 0),
            (13, 1),
            (14, 1),
            (14, 2),
            (14, 10),
            (15, 3),
            (15, 4),
            (15, 11),
            (16, 4),
            (16, 5),
            (16, 10),
            (17, 6),
            (17, 7),
            (17, 11),
            (18, 7),
            (18, 8),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_heavy_square_factory(self):
        coupling = CouplingMap.from_heavy_square(3, bidirectional=False)
        edges = coupling.get_edges()
        expected = [
            (0, 15),
            (1, 16),
            (2, 11),
            (3, 12),
            (3, 17),
            (4, 18),
            (5, 11),
            (6, 12),
            (6, 19),
            (7, 20),
            (9, 15),
            (9, 17),
            (10, 16),
            (10, 18),
            (13, 17),
            (13, 19),
            (14, 18),
            (14, 20),
            (15, 1),
            (16, 2),
            (17, 4),
            (18, 5),
            (19, 7),
            (20, 8),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_heavy_square_factory_bidirectional(self):
        coupling = CouplingMap.from_heavy_square(3, bidirectional=True)
        edges = coupling.get_edges()
        expected = [
            (0, 15),
            (1, 15),
            (1, 16),
            (2, 11),
            (2, 16),
            (3, 12),
            (3, 17),
            (4, 17),
            (4, 18),
            (5, 11),
            (5, 18),
            (6, 12),
            (6, 19),
            (7, 19),
            (7, 20),
            (8, 20),
            (9, 15),
            (9, 17),
            (10, 16),
            (10, 18),
            (11, 2),
            (11, 5),
            (12, 3),
            (12, 6),
            (13, 17),
            (13, 19),
            (14, 18),
            (14, 20),
            (15, 0),
            (15, 1),
            (15, 9),
            (16, 1),
            (16, 2),
            (16, 10),
            (17, 3),
            (17, 4),
            (17, 9),
            (17, 13),
            (18, 4),
            (18, 5),
            (18, 10),
            (18, 14),
            (19, 6),
            (19, 7),
            (19, 13),
            (20, 7),
            (20, 8),
            (20, 14),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_hexagonal_lattice_2_2_factory(self):
        coupling = CouplingMap.from_hexagonal_lattice(2, 2, bidirectional=False)
        edges = coupling.get_edges()
        expected = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (0, 5),
            (2, 7),
            (4, 9),
            (6, 11),
            (8, 13),
            (10, 15),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_hexagonal_lattice_2_2_factory_bidirectional(self):
        coupling = CouplingMap.from_hexagonal_lattice(2, 2, bidirectional=True)
        edges = coupling.get_edges()
        expected = [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (5, 6),
            (6, 5),
            (6, 7),
            (7, 6),
            (7, 8),
            (8, 7),
            (8, 9),
            (9, 8),
            (9, 10),
            (10, 9),
            (11, 12),
            (12, 11),
            (12, 13),
            (13, 12),
            (13, 14),
            (14, 13),
            (14, 15),
            (15, 14),
            (0, 5),
            (5, 0),
            (2, 7),
            (7, 2),
            (4, 9),
            (9, 4),
            (6, 11),
            (11, 6),
            (8, 13),
            (13, 8),
            (10, 15),
            (15, 10),
        ]
        self.assertEqual(set(edges), set(expected))

    def test_implements_iter(self):
        """Test that the object is implicitly iterable."""
        coupling = CouplingMap.from_line(3)
        expected = [(0, 1), (1, 0), (1, 2), (2, 1)]
        self.assertEqual(sorted(coupling), expected)

    def test_disjoint_coupling_map(self):
        cmap = CouplingMap([[0, 1], [1, 0], [2, 3], [3, 2]])
        self.assertFalse(cmap.is_connected())
        distance_matrix = cmap.distance_matrix
        expected = np.array(
            [
                [0, 1, np.inf, np.inf],
                [1, 0, np.inf, np.inf],
                [np.inf, np.inf, 0, 1],
                [np.inf, np.inf, 1, 0],
            ]
        )
        np.testing.assert_array_equal(expected, distance_matrix)

    def test_disjoint_coupling_map_distance_no_path_qubits(self):
        cmap = CouplingMap([[0, 1], [1, 0], [2, 3], [3, 2]])
        self.assertFalse(cmap.is_connected())
        with self.assertRaises(CouplingError):
            cmap.distance(0, 3)

    def test_component_mapping(self):
        cmap = CouplingMap([[0, 1], [1, 0], [2, 3], [3, 2]])
        components = cmap.connected_components()
        self.assertEqual(components[1].graph[0], 2)
        self.assertEqual(components[1].graph[1], 3)
        self.assertEqual(components[0].graph[0], 0)
        self.assertEqual(components[0].graph[1], 1)

    def test_components_connected_graph(self):
        cmap = CouplingMap.from_line(5)
        self.assertTrue(cmap.is_connected())
        subgraphs = cmap.connected_components()
        self.assertEqual(len(subgraphs), 1)
        self.assertTrue(rx.is_isomorphic(cmap.graph, subgraphs[0].graph))

    def test_components_disconnected_graph(self):
        cmap = CouplingMap([[0, 1], [1, 2], [3, 4], [4, 5]])
        self.assertFalse(cmap.is_connected())
        subgraphs = cmap.connected_components()
        self.assertEqual(len(subgraphs), 2)
        expected_subgraph = CouplingMap([[0, 1], [1, 2]])
        self.assertTrue(rx.is_isomorphic(expected_subgraph.graph, subgraphs[0].graph))
        self.assertTrue(rx.is_isomorphic(expected_subgraph.graph, subgraphs[1].graph))

    def test_equality(self):
        """Test that equality checks that the graphs have the same nodes, node labels, and edges."""

        # two coupling maps with 4 nodes and the same edges
        coupling0 = CouplingMap([(0, 1), (0, 2), (2, 3)])
        coupling1 = CouplingMap([(0, 1), (0, 2), (2, 3)])
        self.assertEqual(coupling0, coupling1)

        # coupling map with 5 nodes not equal to the previous 2
        coupling2 = CouplingMap([(0, 1), (0, 2), (2, 4)])
        self.assertNotEqual(coupling0, coupling2)

        # coupling map isomorphic to coupling0, but with cyclically shifted labels
        coupling3 = CouplingMap([(1, 2), (1, 3), (3, 0)])
        self.assertNotEqual(coupling0, coupling3)

        # additional test for comparison to a non-CouplingMap object
        self.assertNotEqual(coupling0, 1)

    def test_ignores_self_coupling(self):
        edges = [(0, 1), (1, 1), (1, 2), (2, 3)]
        with self.assertWarnsRegex(Warning, "qubits cannot be coupled to themselves"):
            coupling = CouplingMap(edges)
        self.assertEqual(set(coupling.get_edges()), {(a, b) for (a, b) in edges if a != b})
        with self.assertWarnsRegex(Warning, "qubits cannot be coupled to themselves"):
            coupling.add_edge(2, 2)
        self.assertEqual(set(coupling.get_edges()), {(a, b) for (a, b) in edges if a != b})


class CouplingVisualizationTest(QiskitVisualizationTestCase):
    @unittest.skipUnless(optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(optionals.HAS_PIL, "Pillow not installed")
    def test_coupling_draw(self):
        """Test that the coupling map drawing with respect to the reference file is correct."""
        cmap = CouplingMap([[0, 1], [1, 2], [2, 3], [2, 4], [2, 5], [2, 6]])
        image_ref = path_to_diagram_reference("coupling_map.png")
        image = cmap.draw()
        self.assertImagesAreEqual(image, image_ref, diff_tolerance=0.01)
