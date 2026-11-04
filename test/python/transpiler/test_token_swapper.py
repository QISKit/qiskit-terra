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

# Copyright 2019 Andrew M. Childs, Eddie Schoute, Cem M. Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test cases for the permutation.complete package"""

import itertools

import rustworkx as rx
from numpy import random

from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper
from qiskit.transpiler.passes.routing.algorithms import util

from test import QiskitTestCase


class TestGeneral(QiskitTestCase):
    """The test cases"""

    def setUp(self) -> None:
        """Set up test cases."""
        super().setUp()
        random.seed(0)

    def test_simple(self) -> None:
        """Test a simple permutation on a path graph of size 4."""
        graph = rx.generators.path_graph(4)
        permutation = {0: 0, 1: 3, 3: 1, 2: 2}
        swapper = ApproximateTokenSwapper(graph)  # type: ApproximateTokenSwapper[int]

        out = list(swapper.map(permutation))
        self.assertEqual(3, len(out))
        util.swap_permutation([out], permutation)
        self.assertEqual({i: i for i in range(4)}, permutation)

    def test_small(self) -> None:
        """Test an inverting permutation on a small path graph of size 8"""
        graph = rx.generators.path_graph(8)
        permutation = {i: 7 - i for i in range(8)}
        swapper = ApproximateTokenSwapper(graph)  # type: ApproximateTokenSwapper[int]

        out = list(swapper.map(permutation))
        util.swap_permutation([out], permutation)
        self.assertEqual({i: i for i in range(8)}, permutation)

    def test_bug1(self) -> None:
        """Tests for a bug that occurred in happy swap chains of length >2."""
        graph = rx.PyGraph()
        graph.extend_from_edge_list(
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (3, 6)]
        )
        permutation = {0: 4, 1: 0, 2: 3, 3: 6, 4: 2, 6: 1}
        swapper = ApproximateTokenSwapper(graph)  # type: ApproximateTokenSwapper[int]

        out = list(swapper.map(permutation))
        util.swap_permutation([out], permutation)
        self.assertEqual({i: i for i in permutation}, permutation)

    def test_partial_simple(self) -> None:
        """Test a partial mapping on a small graph."""
        graph = rx.generators.path_graph(4)
        mapping = {0: 3}
        swapper = ApproximateTokenSwapper(graph)  # type: ApproximateTokenSwapper[int]
        out = list(swapper.map(mapping))
        self.assertEqual(3, len(out))
        util.swap_permutation([out], mapping, allow_missing_keys=True)
        self.assertEqual({3: 3}, mapping)

    def test_partial_small(self) -> None:
        """Test an partial inverting permutation on a small path graph of size 5"""
        graph = rx.generators.path_graph(4)
        permutation = {i: 3 - i for i in range(2)}
        swapper = ApproximateTokenSwapper(graph)  # type: ApproximateTokenSwapper[int]

        out = list(swapper.map(permutation))
        self.assertEqual(5, len(out))
        util.swap_permutation([out], permutation, allow_missing_keys=True)
        self.assertEqual({i: i for i in permutation.values()}, permutation)

    def test_large_partial_random(self) -> None:
        """Test a random (partial) mapping on a large randomly generated graph"""
        size = 100
        # Note that graph may have "gaps" in the node counts, i.e. the numbering is noncontiguous.
        graph = rx.undirected_gnm_random_graph(size, size**2 // 10)
        for i in graph.node_indexes():
            try:
                graph.remove_edge(i, i)  # Remove self-loops.
            except rx.NoEdgeBetweenNodes:
                continue
        # Make sure the graph is connected by adding C_n
        graph.add_edges_from_no_data([(i, i + 1) for i in range(len(graph) - 1)])
        swapper = ApproximateTokenSwapper(graph)  # type: ApproximateTokenSwapper[int]

        # Generate a randomized permutation.
        rand_perm = random.permutation(graph.nodes())
        permutation = dict(zip(graph.nodes(), rand_perm))
        mapping = dict(itertools.islice(permutation.items(), 0, size, 2))  # Drop every 2nd element.

        out = list(swapper.map(mapping, trials=40))
        util.swap_permutation([out], mapping, allow_missing_keys=True)
        self.assertEqual({i: i for i in mapping.values()}, mapping)
