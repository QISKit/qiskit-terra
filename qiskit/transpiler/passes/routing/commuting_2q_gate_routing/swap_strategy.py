# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Defines a swap strategy class."""
from __future__ import annotations
from typing import Any
import copy
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.transpiler.coupling import CouplingMap


class SwapStrategy:
    """A class representing swap strategies for coupling maps.

    A swap strategy is a tuple of swap layers to apply to the coupling map to
    route blocks of commuting two-qubit gates. Each swap layer is specified by a set of tuples
    which correspond to the edges of the coupling map that are swapped. At each swap layer
    SWAP gates are applied to the corresponding edges. These SWAP gates must be executable in
    parallel. This means that a qubit can only be present once in a swap layer. For example, the
    following swap layers represent the optimal swap strategy for a line with five qubits

    .. code-block:: text

        (
            ((0, 1), (2, 3)),  # Swap layer no. 1
            ((1, 2), (3, 4)),  # Swap layer no. 2
            ((0, 1), (2, 3)),  # Swap layer no. 3
        )

    This strategy is optimal in the sense that it reaches full qubit-connectivity in the least
    amount of swap gates. More generally, a swap strategy is optimal for a given block of
    commuting two-qubit gates and a given coupling map if it minimizes the number of gates
    applied when routing the commuting two-qubit gates to the coupling map. Finding the optimal
    swap strategy is a non-trivial problem but can be done for certain coupling maps such as a
    line coupling map. This class stores the permutations of the qubits resulting from the swap
    strategy. See https://arxiv.org/abs/2202.03459 for more details.
    """

    def __init__(
        self, coupling_map: CouplingMap, swap_layers: tuple[tuple[tuple[int, int], ...], ...]
    ) -> None:
        """
        Args:
            coupling_map: The coupling map the strategy is implemented for.
            swap_layers: The swap layers of the strategy, specified as tuple of swap layers.
                Each swap layer is a tuple of edges to which swaps are applied simultaneously.
                Each swap is specified as an edge which is a tuple of two integers.

        Raises:
            QiskitError: If the coupling map is not specified.
            QiskitError: if the swap strategy is not valid. A swap strategy is valid if all
                swap gates, specified as tuples, are contained in the edge set of the coupling map.
                A swap strategy is also invalid if a layer has multiple swaps on the same qubit.
        """
        self._coupling_map = coupling_map
        self._num_vertices = coupling_map.size()
        self._swap_layers = swap_layers
        self._distance_matrix: np.ndarray | None = None
        self._possible_edges: set[tuple[int, int]] | None = None
        self._missing_couplings: set[tuple[int, int]] | None = None
        self._inverse_composed_permutation = {0: list(range(self._num_vertices))}

        edge_set = set(self._coupling_map.get_edges())

        for i, layer in enumerate(self._swap_layers):
            for edge in layer:
                if edge not in edge_set:
                    raise QiskitError(
                        f"The {i}th swap layer contains the edge {edge} which is not "
                        f"part of the underlying coupling map with {edge_set} edges."
                    )

            layer_qubits = [qubit for edge in layer for qubit in edge]
            if len(layer_qubits) != len(set(layer_qubits)):
                raise QiskitError(f"The {i}th swap layer contains a qubit with multiple swaps.")

    @classmethod
    def from_line(cls, line: list[int], num_swap_layers: int | None = None) -> SwapStrategy:
        """Creates a swap strategy for a line graph with the specified number of SWAP layers.

        This SWAP strategy will use the full line if instructed to do so (i.e. num_variables
        is None or equal to num_vertices). If instructed otherwise then the first num_variables
        nodes of the line will be used in the swap strategy.

        Args:
            line: A line given as a list of nodes, e.g. ``[0, 2, 3, 4]``.
            num_swap_layers: Number of swap layers the swap manager should be initialized with.

        Returns:
            A swap strategy that reaches full connectivity on a linear coupling map.

        Raises:
            ValueError: If the ``num_swap_layers`` is negative.
            ValueError: If the ``line`` has less than 2 elements and no swap strategy can be applied.
        """
        if len(line) < 2:
            raise ValueError(f"The line cannot have less than two elements, but is {line}")

        if num_swap_layers is None:
            num_swap_layers = len(line) - 2

        elif num_swap_layers < 0:
            raise ValueError(f"Negative number {num_swap_layers} passed for number of swap layers.")

        swap_layer0 = tuple((line[i], line[i + 1]) for i in range(0, len(line) - 1, 2))
        swap_layer1 = tuple((line[i], line[i + 1]) for i in range(1, len(line) - 1, 2))

        base_layers = [swap_layer0, swap_layer1]

        swap_layers = tuple(base_layers[i % 2] for i in range(num_swap_layers))

        couplings = []
        for idx in range(len(line) - 1):
            couplings.append((line[idx], line[idx + 1]))
            couplings.append((line[idx + 1], line[idx]))

        return cls(coupling_map=CouplingMap(couplings), swap_layers=tuple(swap_layers))

    def __len__(self) -> int:
        """Return the length of the strategy as the number of layers.

        Returns:
            The number of layers of the swap strategy.
        """
        return len(self._swap_layers)

    def __repr__(self) -> str:
        """Representation of the swap strategy.

        Returns:
            The representation of the swap strategy.
        """
        description = [f"{self.__class__.__name__} with swap layers:\n"]

        for layer in self._swap_layers:
            description.append(f"{layer},\n")

        description.append(f"on {self._coupling_map} coupling map.")
        description = "".join(description)

        return description

    def swap_layer(self, idx: int) -> list[tuple[int, int]]:
        """Return the layer of swaps at the given index.

        Args:
            idx: The index of the returned swap layer.

        Returns:
            A copy of the swap layer at ``idx`` to avoid any unintentional modification to
            the swap strategy.
        """
        return list(self._swap_layers[idx])

    @property
    def distance_matrix(self) -> np.ndarray:
        """A matrix describing when qubits become adjacent in the swap strategy.

        Returns:
            The distance matrix for the SWAP strategy as an array that cannot be written to. Here,
            the entry (i, j) corresponds to the number of SWAP layers that need to be applied to
            obtain a connection between physical qubits i and j.
        """
        if self._distance_matrix is None:
            self._distance_matrix = np.full((self._num_vertices, self._num_vertices), -1, dtype=int)

            for i in range(self._num_vertices):
                self._distance_matrix[i, i] = 0

            for i in range(len(self._swap_layers) + 1):
                for j, k in self.swapped_coupling_map(i).get_edges():

                    # This if ensures that the smallest distance is used.
                    if self._distance_matrix[j, k] == -1:
                        self._distance_matrix[j, k] = i
                        self._distance_matrix[k, j] = i

            self._distance_matrix.setflags(write=False)

        return self._distance_matrix

    def new_connections(self, idx: int) -> list[set[int]]:
        """
        Returns the new connections obtained after applying the SWAP layer specified by idx, i.e.
        a list of qubit pairs that are adjacent to one another after idx steps of the SWAP strategy.

        Args:
            idx: The index of the SWAP layer. 1 refers to the first SWAP layer whereas an ``idx``
                of 0 will return the connections present in the original coupling map.

        Returns:
            A list of edges representing the new qubit connections.
        """
        connections = []
        for i in range(self._num_vertices):
            for j in range(i):
                if self.distance_matrix[i, j] == idx:
                    connections.append({i, j})
        return connections

    def _build_edges(self) -> set[tuple[int, int]]:
        """Build the possible edges that the swap strategy accommodates."""

        possible_edges = set()
        for swap_layer_idx in range(len(self) + 1):
            for edge in self.swapped_coupling_map(swap_layer_idx).get_edges():
                possible_edges.add(edge)
                possible_edges.add(edge[::-1])

        return possible_edges

    @property
    def possible_edges(self) -> set[tuple[int, int]]:
        """Return the qubit connections that can be generated.

        Returns:
            The qubit connections that can be accommodated by the swap strategy.
        """
        if self._possible_edges is None:
            self._possible_edges = self._build_edges()

        return self._possible_edges

    @property
    def missing_couplings(self) -> set[tuple[int, int]]:
        """Return the set of couplings that cannot be reached.

        Returns:
            The couplings that cannot be reached as a set of Tuples of int. Here,
            each int corresponds to a qubit in the coupling map.
        """
        if self._missing_couplings is None:
            self._missing_couplings = set(zip(*(self.distance_matrix == -1).nonzero()))

        return self._missing_couplings

    def swapped_coupling_map(self, idx: int) -> CouplingMap:
        """Returns the coupling map after applying ``idx`` swap layers of strategy.

        Args:
            idx: The number of swap layers to apply. For idx = 0, the original coupling
                map is returned.

        Returns:
            The swapped coupling map.
        """
        permutation = self.inverse_composed_permutation(idx)

        edges = [[permutation[i], permutation[j]] for i, j in self._coupling_map.get_edges()]

        return CouplingMap(couplinglist=edges)

    def apply_swap_layer(
        self, list_to_swap: list[Any], idx: int, inplace: bool = False
    ) -> list[Any]:
        """Permute the elements of ``list_to_swap`` based on layer indexed by ``idx``.

        Args:
            list_to_swap: The list of elements to swap.
            idx: The index of the swap layer to apply.
            inplace: A boolean which if set to True will modify the list inplace. By default
                this value is False.

        Returns:
            The list with swapped elements
        """
        if inplace:
            x = list_to_swap
        else:
            x = copy.copy(list_to_swap)

        for i, j in self._swap_layers[idx]:
            x[i], x[j] = x[j], x[i]

        return x

    def inverse_composed_permutation(self, idx: int) -> list[int]:
        """
        Returns the inversed composed permutation of all swap layers applied up to layer
        ``idx``. Permutations are represented by list of integers where the ith element
        corresponds to the mapping of i under the permutation.

        Args:
            idx: The number of swap layers to apply.

        Returns:
            The inversed permutation as a list of integer values.
        """
        # Only compute the inverse permutation if it has not been computed before
        if idx not in self._inverse_composed_permutation:
            self._inverse_composed_permutation[idx] = self.apply_swap_layer(
                self.inverse_composed_permutation(idx - 1), idx - 1
            )

        return self._inverse_composed_permutation[idx]
