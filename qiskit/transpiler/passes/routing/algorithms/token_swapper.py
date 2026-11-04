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

"""Permutation algorithms for general graphs."""

from __future__ import annotations
import logging
from collections.abc import Mapping

import numpy as np
import rustworkx as rx

from .types import Swap, Permutation
from .util import PermutationCircuit, permutation_circuit


logger = logging.getLogger(__name__)


class ApproximateTokenSwapper:
    """A class for computing approximate solutions to the Token Swapping problem.

    Internally caches the graph and associated datastructures for re-use.
    """

    def __init__(self, graph: rx.PyGraph, seed: int | np.random.Generator | None = None) -> None:
        """Construct an ApproximateTokenSwapping object.

        Args:
            graph: Undirected graph representing a coupling map.
            seed: Seed to use for random trials.
        """
        self.graph = graph
        self.shortest_paths = rx.graph_distance_matrix(graph)
        if isinstance(seed, np.random.Generator):
            self.seed = seed
        else:
            self.seed = np.random.default_rng(seed)

    def distance(self, vertex0: int, vertex1: int) -> int:
        """Compute the distance between two nodes in `graph`."""
        return self.shortest_paths[vertex0, vertex1]

    def permutation_circuit(self, permutation: Permutation, trials: int = 4) -> PermutationCircuit:
        """Perform an approximately optimal Token Swapping algorithm to implement the permutation.

        Args:
          permutation: The partial mapping to implement in swaps.
          trials: The number of trials to try to perform the mapping. Minimize over the trials.

        Returns:
          The circuit to implement the permutation
        """
        sequential_swaps = self.map(permutation, trials=trials)

        parallel_swaps = [[swap] for swap in sequential_swaps]
        return permutation_circuit(parallel_swaps)

    def map(
        self, mapping: Mapping[int, int], trials: int = 4, parallel_threshold: int = 50
    ) -> list[Swap[int]]:
        """Perform an approximately optimal Token Swapping algorithm to implement the permutation.

        Supports partial mappings (i.e. not-permutations) for graphs with missing tokens.

        Based on the paper: Approximation and Hardness for Token Swapping by Miltzow et al. (2016)
        ArXiV: https://arxiv.org/abs/1602.05150
        and generalization based on our own work.

        Args:
          mapping: The partial mapping to implement in swaps.
          trials: The number of trials to try to perform the mapping. Minimize over the trials.
          parallel_threshold: The number of nodes in the graph beyond which the algorithm
                will use parallel processing

        Returns:
          The swaps to implement the mapping
        """
        # Since integer seed is used in rustworkx, take random integer from np.random.randint
        # and use that for the seed.
        seed = self.seed.integers(1, 10000)
        return rx.graph_token_swapper(self.graph, mapping, trials, seed, parallel_threshold)
