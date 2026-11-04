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

"""A swap strategy pass for blocks of commuting gates."""
from __future__ import annotations
from collections import defaultdict

from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from .swap_strategy import SwapStrategy
from .commuting_2q_block import Commuting2qBlock


class Commuting2qGateRouter(TransformationPass):
    """A class to swap route one or more commuting gates to the coupling map.

    This pass routes blocks of commuting two-qubit gates encapsulated as
    :class:`.Commuting2qBlock` instructions. This pass will not apply to other instructions.
    The mapping to the coupling map is done using swap strategies, see :class:`.SwapStrategy`.
    The swap strategy should suit the problem and the coupling map. This transpiler pass
    should ideally be executed before the quantum circuit is enlarged with any idle ancilla
    qubits. Otherwise, we may swap qubits outside the portion of the chip we want to use.
    Therefore, the swap strategy and its associated coupling map do not represent physical
    qubits. Instead, they represent an intermediate mapping that corresponds to the physical
    qubits once the initial layout is applied. The example below shows how to map a four
    qubit :class:`.PauliEvolutionGate` to qubits 0, 1, 3, and 4 of the five qubit device with
    the coupling map

    .. code-block:: text

        0 -- 1 -- 2
             |
             3
             |
             4

    To do this we use a line swap strategy for qubits 0, 1, 3, and 4 defined in terms
    of virtual qubits 0, 1, 2, and 3.

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.transpiler import Layout, CouplingMap, PassManager
        from qiskit.transpiler.passes import FullAncillaAllocation
        from qiskit.transpiler.passes import EnlargeWithAncilla
        from qiskit.transpiler.passes import ApplyLayout
        from qiskit.transpiler.passes import SetLayout

        from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
            SwapStrategy,
            FindCommutingPauliEvolutions,
            Commuting2qGateRouter,
        )

        # Define the circuit on virtual qubits
        op = SparsePauliOp.from_list([("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)])
        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(4))

        # Define the swap strategy on qubits before the initial_layout is applied.
        swap_strat = SwapStrategy.from_line([0, 1, 2, 3])

        # Choose qubits 0, 1, 3, and 4 from the backend coupling map shown above.
        backend_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (1, 3), (3, 4)])
        initial_layout = Layout.from_intlist([0, 1, 3, 4], *circ.qregs)

        pm_pre = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat),
                SetLayout(initial_layout),
                FullAncillaAllocation(backend_cmap),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ]
        )

        # Insert swap gates, map to initial_layout and finally enlarge with ancilla.
        pm_pre.run(circ).draw("mpl")

    This pass manager relies on the ``current_layout`` which corresponds to the qubit layout as
    swap gates are applied. The pass will traverse all nodes in the dag. If a node should be
    routed using a swap strategy then it will be decomposed into sub-instructions with swap
    layers in between and the ``current_layout`` will be modified. Nodes that should not be
    routed using swap strategies will be added back to the dag taking the ``current_layout``
    into account.
    """

    def __init__(
        self,
        swap_strategy: SwapStrategy | None = None,
        edge_coloring: dict[tuple[int, int], int] | None = None,
    ) -> None:
        r"""
        Args:
            swap_strategy: An instance of a :class:`.SwapStrategy` that holds the swap layers
                that are used, and the order in which to apply them, to map the instruction to
                the hardware. If this field is not given, it should be contained in the
                property set of the pass. This allows other passes to determine the most
                appropriate swap strategy at run-time.
            edge_coloring: An optional edge coloring of the coupling map (I.e. no two edges that
                share a node have the same color). If the edge coloring is given then the commuting
                gates that can be simultaneously applied given the current qubit permutation are
                grouped according to the edge coloring and applied according to this edge
                coloring. Here, a color is an int which is used as the index to define and
                access the groups of commuting gates that can be applied simultaneously.
                If the edge coloring is not given then the sets will be built-up using a
                greedy algorithm. The edge coloring is useful to position gates such as
                ``RZZGate``\s next to swap gates to exploit CX cancellations.
        """
        super().__init__()
        self._swap_strategy = swap_strategy
        self._bit_indices: dict[Qubit, int] | None = None
        self._edge_coloring = edge_coloring

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass by decomposing the nodes it applies on.

        Args:
            dag: The dag to which we will add swaps.

        Returns:
            A dag where swaps have been added for the intended gate type.

        Raises:
            TranspilerError: If the swap strategy was not given at init time and there is
                no swap strategy in the property set.
            TranspilerError: If the quantum circuit contains more than one qubit register.
            TranspilerError: If there are qubits that are not contained in the quantum register.
        """
        if self._swap_strategy is None:
            swap_strategy = self.property_set["swap_strategy"]

            if swap_strategy is None:
                raise TranspilerError("No swap strategy given at init or in the property set.")
        else:
            swap_strategy = self._swap_strategy

        if len(dag.qregs) != 1:
            raise TranspilerError(
                f"{self.__class__.__name__} runs on circuits with one quantum register."
            )

        if len(dag.qubits) != next(iter(dag.qregs.values())).size:
            raise TranspilerError("Circuit has qubits not contained in the qubit register.")

        # Fix output permutation -- copied from ElidePermutations
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping

        new_dag = dag.copy_empty_like()
        current_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        # Used to keep track of nodes that do not decompose using swap strategies.
        accumulator = new_dag.copy_empty_like()
        accumulator.global_phase = 0

        for node in dag.topological_op_nodes():
            if isinstance(node.op, Commuting2qBlock):

                # Check that the swap strategy creates enough connectivity for the node.
                self._check_edges(dag, node, swap_strategy)

                # Compose any accumulated non-swap strategy gates to the dag
                accumulator = self._compose_non_swap_nodes(accumulator, current_layout, new_dag)

                # Decompose the swap-strategy node and add to the dag.
                new_dag.compose(self.swap_decompose(dag, node, current_layout, swap_strategy))
            else:
                accumulator.apply_operation_back(node.op, node.qargs, node.cargs)

        self._compose_non_swap_nodes(accumulator, current_layout, new_dag)

        self.property_set["virtual_permutation_layout"] = current_layout

        return new_dag

    def _compose_non_swap_nodes(
        self, accumulator: DAGCircuit, layout: Layout, new_dag: DAGCircuit
    ) -> DAGCircuit:
        """Add all the non-swap strategy nodes that we have accumulated up to now.

        This method also resets the node accumulator to an empty dag.

        Args:
            layout: The current layout that keeps track of the swaps.
            new_dag: The new dag that we are building up.
            accumulator: A DAG to keep track of nodes that do not decompose
                using swap strategies.

        Returns:
            A new accumulator with the same registers as ``new_dag``.
        """
        # Add all the non-swap strategy nodes that we have accumulated up to now.
        order = layout.reorder_bits(new_dag.qubits)
        order_bits: list[int | None] = [None] * len(layout)
        for idx, val in enumerate(order):
            order_bits[val] = idx

        new_dag.compose(accumulator, qubits=order_bits)

        # Re-initialize the node accumulator
        accumulator = new_dag.copy_empty_like()
        accumulator.global_phase = 0
        return accumulator

    def _position_in_cmap(self, dag: DAGCircuit, j: int, k: int, layout: Layout) -> tuple[int, ...]:
        """A helper function to track the movement of virtual qubits through the swaps.

        Args:
            dag: The dag the pass is being run on.
            j: The index of decision variable j (i.e. virtual qubit).
            k: The index of decision variable k (i.e. virtual qubit).
            layout: The current layout that takes into account previous swap gates.

        Returns:
            The position in the coupling map of the virtual qubits j and k as a tuple.
        """
        bit0 = dag.find_bit(layout.get_physical_bits()[j]).index
        bit1 = dag.find_bit(layout.get_physical_bits()[k]).index

        return bit0, bit1

    def _build_sub_layers(
        self, current_layer: dict[tuple[int, int], Gate]
    ) -> list[dict[tuple[int, int], Gate]]:
        """A helper method to build-up sets of gates to simultaneously apply.

        This is done with an edge coloring if the ``edge_coloring`` init argument was given or with
        a greedy algorithm if not. With an edge coloring all gates on edges with the same color
        will be applied simultaneously. These sublayers are applied in the order of their color,
        which is an int, in increasing color order.

        Args:
            current_layer: All gates in the current layer can be applied given the qubit ordering
                of the current layout. However, not all gates in the current layer can be applied
                simultaneously. This function creates sub-layers by building up sub-layers
                of gates. All gates in a sub-layer can simultaneously be applied given the coupling
                map and current qubit configuration.

        Returns:
             A list of gate dicts that can be applied. The gates a position 0 are applied first.
             A gate dict has the qubit tuple as key and the gate to apply as value.
        """
        if self._edge_coloring is not None:
            return self._edge_coloring_build_sub_layers(current_layer)
        else:
            return self._greedy_build_sub_layers(current_layer)

    def _edge_coloring_build_sub_layers(
        self, current_layer: dict[tuple[int, int], Gate]
    ) -> list[dict[tuple[int, int], Gate]]:
        """The edge coloring method of building sub-layers of commuting gates."""
        sub_layers: list[dict[tuple[int, int], Gate]] = [
            {} for _ in set(self._edge_coloring.values())
        ]
        for edge, gate in current_layer.items():
            color = self._edge_coloring[edge]
            sub_layers[color][edge] = gate

        return sub_layers

    @staticmethod
    def _greedy_build_sub_layers(
        current_layer: dict[tuple[int, int], Gate],
    ) -> list[dict[tuple[int, int], Gate]]:
        """The greedy method of building sub-layers of commuting gates."""
        sub_layers = []
        while len(current_layer) > 0:
            current_sub_layer, remaining_gates = {}, {}
            blocked_vertices: set[tuple] = set()

            for edge, evo_gate in current_layer.items():
                if blocked_vertices.isdisjoint(edge):
                    current_sub_layer[edge] = evo_gate

                    # A vertex becomes blocked once a gate is applied to it.
                    blocked_vertices = blocked_vertices.union(edge)
                else:
                    remaining_gates[edge] = evo_gate

            current_layer = remaining_gates
            sub_layers.append(current_sub_layer)

        return sub_layers

    def swap_decompose(
        self, dag: DAGCircuit, node: DAGOpNode, current_layout: Layout, swap_strategy: SwapStrategy
    ) -> DAGCircuit:
        """Take an instance of :class:`.Commuting2qBlock` and map it to the coupling map.

        The mapping is done with the swap strategy.

        Args:
            dag: The dag which contains the :class:`.Commuting2qBlock` we route.
            node: A node whose operation is a :class:`.Commuting2qBlock`.
            current_layout: The layout before the swaps are applied. This function will
                modify the layout so that subsequent gates can be properly composed on the dag.
            swap_strategy: The swap strategy used to decompose the node.

        Returns:
            A dag that is compatible with the coupling map where swap gates have been added
            to map the gates in the :class:`.Commuting2qBlock` to the hardware.
        """
        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        gate_layers = self._make_op_layers(dag, node.op, current_layout, swap_strategy)

        # Iterate over and apply gate layers
        max_distance = max(gate_layers.keys())

        circuit_with_swap = QuantumCircuit(len(dag.qubits))

        for i in range(max_distance + 1):
            # Get current layer and replace the problem indices j,k by the corresponding
            # positions in the coupling map. The current layer corresponds
            # to all the gates that can be applied at the ith swap layer.
            current_layer = {}
            for (j, k), local_gate in gate_layers.get(i, {}).items():
                current_layer[self._position_in_cmap(dag, j, k, current_layout)] = local_gate

            # Not all gates that are applied at the ith swap layer can be applied at the same
            # time. We therefore greedily build sub-layers.
            sub_layers = self._build_sub_layers(current_layer)

            # Apply sub-layers
            for sublayer in sub_layers:
                for edge, local_gate in sublayer.items():
                    circuit_with_swap.append(local_gate, edge)

            # Apply SWAP gates
            if i < max_distance:
                for swap in swap_strategy.swap_layer(i):
                    (j, k) = [trivial_layout.get_physical_bits()[vertex] for vertex in swap]

                    circuit_with_swap.swap(j, k)
                    current_layout.swap(j, k)

        return circuit_to_dag(circuit_with_swap)

    def _make_op_layers(
        self, dag: DAGCircuit, op: Commuting2qBlock, layout: Layout, swap_strategy: SwapStrategy
    ) -> dict[int, dict[tuple, Gate]]:
        """Creates layers of two-qubit gates based on the distance in the swap strategy."""

        gate_layers: dict[int, dict[tuple, Gate]] = defaultdict(dict)

        for node in op.node_block:
            edge = (dag.find_bit(node.qargs[0]).index, dag.find_bit(node.qargs[1]).index)

            bit0 = layout.get_virtual_bits()[dag.qubits[edge[0]]]
            bit1 = layout.get_virtual_bits()[dag.qubits[edge[1]]]

            distance = swap_strategy.distance_matrix[bit0, bit1]

            gate_layers[distance][edge] = node.op

        return gate_layers

    def _check_edges(self, dag: DAGCircuit, node: DAGOpNode, swap_strategy: SwapStrategy):
        """Check if the swap strategy can create the required connectivity.

        Args:
            dag: The dag to check edges from.
            node: The dag node for which to check if the swap strategy provides enough connectivity.
            swap_strategy: The swap strategy that is being used.

        Raises:
            TranspilerError: If there is an edge that the swap strategy cannot accommodate
                and if the pass has been configured to raise on such issues.
        """
        required_edges = set()

        for sub_node in node.op:
            edge = (dag.find_bit(sub_node.qargs[0]).index, dag.find_bit(sub_node.qargs[1]).index)
            required_edges.add(edge)

        # Check that the swap strategy supports all required edges
        if not required_edges.issubset(swap_strategy.possible_edges):
            raise TranspilerError(
                f"{swap_strategy} cannot implement all edges in {required_edges}."
            )
