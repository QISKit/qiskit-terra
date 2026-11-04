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


"""Various ways to divide a DAG into blocks of nodes, to split blocks of nodes
into smaller sub-blocks, and to consolidate blocks."""
from __future__ import annotations

from collections.abc import Iterable, Callable

from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister, Bit
from qiskit.circuit.controlflow import condition_resources
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagdependency import DAGDependency
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.dagcircuit.dagdepnode import DAGDepNode
from .exceptions import DAGCircuitError


class BlockCollector:
    """This class implements various strategies of dividing a DAG (directed acyclic graph)
    into blocks of nodes that satisfy certain criteria. It works both with the
    :class:`~qiskit.dagcircuit.DAGCircuit` and
    :class:`~qiskit.dagcircuit.DAGDependency` representations of a DAG, where
    the latter takes into account commutativity between nodes.

    Collecting nodes from DAGDependency generally leads to more optimal results, but is
    slower, as it requires to construct a DAGDependency beforehand. Thus, DAGCircuit should
    be used with lower transpiler settings, and DAGDependency should be used with higher
    transpiler settings.

    In general, there are multiple ways to collect maximal blocks. The approaches used
    here are of the form 'starting from the input nodes of a DAG, greedily collect
    the largest block of nodes that match certain criteria'. For additional details,
    see https://github.com/Qiskit/qiskit-terra/issues/5775.
    """

    def __init__(self, dag: DAGCircuit | DAGDependency):
        """
        Args:
            dag (Union[DAGCircuit, DAGDependency]): The input DAG.

        Raises:
            DAGCircuitError: the input object is not a DAG.
        """

        self.dag = dag
        self._pending_nodes: list[DAGOpNode | DAGDepNode] | None = None
        self._in_degree: dict[DAGOpNode | DAGDepNode, int] | None = None
        self._collect_from_back = False

        if isinstance(dag, DAGCircuit):
            self.is_dag_dependency = False

        elif isinstance(dag, DAGDependency):
            self.is_dag_dependency = True

        else:
            raise DAGCircuitError("not a DAG.")

    def _setup_in_degrees(self):
        """For an efficient implementation, for every node we keep the number of its
        unprocessed immediate predecessors (called ``_in_degree``). This ``_in_degree``
        is set up at the start and updated throughout the algorithm.
        A node is leaf (or input) node iff its ``_in_degree`` is 0.
        When a node is (marked as) collected, the ``_in_degree`` of each of its immediate
        successor is updated by subtracting 1.
        Additionally, ``_pending_nodes`` explicitly keeps the list of nodes whose
        ``_in_degree`` is 0.
        """
        self._pending_nodes = []
        self._in_degree = {}
        for node in self._op_nodes():
            deg = len(self._direct_preds(node))
            self._in_degree[node] = deg
            if deg == 0:
                self._pending_nodes.append(node)

    def _op_nodes(self) -> Iterable[DAGOpNode | DAGDepNode]:
        """Returns DAG nodes."""
        if not self.is_dag_dependency:
            return self.dag.op_nodes()
        else:
            return self.dag.get_nodes()

    def _direct_preds(self, node):
        """Returns direct predecessors of a node. This function takes into account the
        direction of collecting blocks, that is node's predecessors when collecting
        backwards are the direct successors of a node in the DAG.
        """
        if not self.is_dag_dependency:
            if self._collect_from_back:
                return [pred for pred in self.dag.successors(node) if isinstance(pred, DAGOpNode)]
            else:
                return [pred for pred in self.dag.predecessors(node) if isinstance(pred, DAGOpNode)]
        elif self._collect_from_back:
            return [
                self.dag.get_node(pred_id) for pred_id in self.dag.direct_successors(node.node_id)
            ]
        else:
            return [
                self.dag.get_node(pred_id) for pred_id in self.dag.direct_predecessors(node.node_id)
            ]

    def _direct_succs(self, node):
        """Returns direct successors of a node. This function takes into account the
        direction of collecting blocks, that is node's successors when collecting
        backwards are the direct predecessors of a node in the DAG.
        """
        if not self.is_dag_dependency:
            if self._collect_from_back:
                return [succ for succ in self.dag.predecessors(node) if isinstance(succ, DAGOpNode)]
            else:
                return [succ for succ in self.dag.successors(node) if isinstance(succ, DAGOpNode)]
        elif self._collect_from_back:
            return [
                self.dag.get_node(succ_id) for succ_id in self.dag.direct_predecessors(node.node_id)
            ]
        else:
            return [
                self.dag.get_node(succ_id) for succ_id in self.dag.direct_successors(node.node_id)
            ]

    def _have_uncollected_nodes(self):
        """Returns whether there are uncollected (pending) nodes"""
        return len(self._pending_nodes) > 0

    def collect_matching_block(
        self, filter_fn: Callable, max_block_width: int | None
    ) -> list[DAGOpNode | DAGDepNode]:
        """Iteratively collects the largest block of input nodes (that is, nodes with
        ``_in_degree`` equal to 0) that match a given filtering function.
        Examples of this include collecting blocks of swap gates,
        blocks of linear gates (CXs and SWAPs), blocks of Clifford gates, blocks of single-qubit gates,
        blocks of two-qubit gates, etc.  Here 'iteratively' means that once a node is collected,
        the ``_in_degree`` of each of its immediate successor is decreased by 1, allowing more nodes
        to become input and to be eligible for collecting into the current block.
        Returns the block of collected nodes.
        """
        current_block = []
        current_block_qargs = set()
        unprocessed_pending_nodes = self._pending_nodes
        self._pending_nodes = []

        # Iteratively process unprocessed_pending_nodes:
        # - any node that does not match filter_fn is added to pending_nodes
        # - any node that match filter_fn is added to the current_block,
        #   and some of its successors may be moved to unprocessed_pending_nodes.
        while unprocessed_pending_nodes:
            node = unprocessed_pending_nodes.pop()

            if max_block_width is not None:
                # for efficiency, only update new_qargs when max_block_width is specified
                new_qargs = current_block_qargs.copy()
                new_qargs.update(node.qargs)
                width_within_budget = len(new_qargs) <= max_block_width
            else:
                new_qargs = set()
                width_within_budget = True

            if filter_fn(node) and width_within_budget:
                current_block.append(node)
                current_block_qargs = new_qargs

                # update the _in_degree of node's successors
                for suc in self._direct_succs(node):
                    self._in_degree[suc] -= 1
                    if self._in_degree[suc] == 0:
                        unprocessed_pending_nodes.append(suc)
            else:
                self._pending_nodes.append(node)

        return current_block

    def collect_all_matching_blocks(
        self,
        filter_fn,
        split_blocks=True,
        min_block_size=2,
        split_layers=False,
        collect_from_back=False,
        max_block_width=None,
    ):
        """Collects all blocks that match a given filtering function filter_fn.
        This iteratively finds the largest block that does not match filter_fn,
        then the largest block that matches filter_fn, and so on, until no more uncollected
        nodes remain. Intuitively, finding larger blocks of non-matching nodes helps to
        find larger blocks of matching nodes later on.

        After the blocks are collected, they can be optionally refined. The option
        ``split_blocks`` allows to split collected blocks into sub-blocks over disjoint
        qubit subsets. The option ``split_layers`` allows to split collected blocks
        into layers of non-overlapping instructions. The option ``min_block_size``
        specifies the minimum number of gates in the block for the block to be collected.
        The option ``max_block_width`` specifies the maximum number of qubits over
        which a block can be defined.

        By default, blocks are collected in the direction from the inputs towards the outputs
        of the circuit. The option ``collect_from_back`` allows to change this direction,
        that is collect blocks from the outputs towards the inputs of the circuit.

        Returns the list of matching blocks only.
        """

        def not_filter_fn(node):
            """Returns the opposite of filter_fn."""
            return not filter_fn(node)

        # Note: the collection direction must be specified before setting in-degrees
        self._collect_from_back = collect_from_back
        self._setup_in_degrees()

        # Iteratively collect non-matching and matching blocks.
        matching_blocks: list[list[DAGOpNode | DAGDepNode]] = []
        while self._have_uncollected_nodes():
            self.collect_matching_block(not_filter_fn, max_block_width=None)
            matching_block = self.collect_matching_block(filter_fn, max_block_width=max_block_width)
            if matching_block:
                matching_blocks.append(matching_block)

        # If the option split_layers is set, refine blocks by splitting them into layers
        # of non-overlapping instructions (in other words, into depth-1 sub-blocks).
        if split_layers:
            tmp_blocks = []
            for block in matching_blocks:
                tmp_blocks.extend(split_block_into_layers(block))
            matching_blocks = tmp_blocks

        # If the option split_blocks is set, refine blocks by splitting them into sub-blocks over
        # disconnected qubit subsets.
        if split_blocks:
            tmp_blocks = []
            for block in matching_blocks:
                tmp_blocks.extend(BlockSplitter().run(block))
            matching_blocks = tmp_blocks

        # If we are collecting from the back, both the order of the blocks
        # and the order of nodes in each block should be reversed.
        if self._collect_from_back:
            matching_blocks = [block[::-1] for block in matching_blocks[::-1]]

        # Keep only blocks with at least min_block_sizes.
        matching_blocks = [block for block in matching_blocks if len(block) >= min_block_size]

        return matching_blocks


class BlockSplitter:
    """Splits a block of nodes into sub-blocks over disjoint qubits.
    The implementation is based on the Disjoint Set Union data structure."""

    def __init__(self):
        self.leader = {}  # qubit's group leader
        self.group = {}  # qubit's group

    def find_leader(self, index):
        """Find in DSU."""
        if index not in self.leader:
            self.leader[index] = index
            self.group[index] = []
            return index
        if self.leader[index] == index:
            return index
        self.leader[index] = self.find_leader(self.leader[index])
        return self.leader[index]

    def union_leaders(self, index1, index2):
        """Union in DSU."""
        leader1 = self.find_leader(index1)
        leader2 = self.find_leader(index2)
        if leader1 == leader2:
            return
        if len(self.group[leader1]) < len(self.group[leader2]):
            leader1, leader2 = leader2, leader1

        self.leader[leader2] = leader1
        self.group[leader1].extend(self.group[leader2])
        self.group[leader2].clear()

    def run(self, block):
        """Splits block of nodes into sub-blocks over disjoint qubits."""
        for node in block:
            indices = node.qargs
            if not indices:
                continue
            first = indices[0]
            for index in indices[1:]:
                self.union_leaders(first, index)
            self.group[self.find_leader(first)].append(node)

        blocks = []
        for index, item in self.leader.items():
            if index == item:
                blocks.append(self.group[index])

        return blocks


def split_block_into_layers(block: list[DAGOpNode | DAGDepNode]):
    """Splits a block of nodes into sub-blocks of non-overlapping instructions
    (or, in other words, into depth-1 sub-blocks).
    """
    bit_depths: dict[Bit, int] = {}
    layers: list[list[DAGOpNode | DAGDepNode]] = []

    for node in block:
        cur_bits = set(node.qargs)
        cur_bits.update(node.cargs)

        cond = getattr(node.op, "_condition", None)
        if cond is not None:
            cur_bits.update(condition_resources(cond).clbits)

        cur_depth = max(bit_depths.get(bit, 0) for bit in cur_bits)
        while len(layers) <= cur_depth:
            layers.append([])

        for bit in cur_bits:
            bit_depths[bit] = cur_depth + 1
        layers[cur_depth].append(node)

    return layers


class BlockCollapser:
    """This class implements various strategies of consolidating blocks of nodes
    in a DAG (directed acyclic graph). It works both with
    the :class:`~qiskit.dagcircuit.DAGCircuit`
    and :class:`~qiskit.dagcircuit.DAGDependency` DAG representations.
    """

    def __init__(self, dag):
        """
        Args:
            dag (Union[DAGCircuit, DAGDependency]): The input DAG.

        Raises:
            DAGCircuitError: the input object is not a DAG.
        """

        self.dag = dag

    def collapse_to_operation(self, blocks, collapse_fn):
        """For each block, constructs a quantum circuit containing instructions in the block,
        then uses collapse_fn to collapse this circuit into a single operation.
        """
        global_index_map = {wire: idx for idx, wire in enumerate(self.dag.qubits)}
        global_index_map.update({wire: idx for idx, wire in enumerate(self.dag.clbits)})

        for block in blocks:
            # Find the sets of qubits/clbits used in this block (which might be much smaller
            # than the set of all qubits/clbits).
            cur_qubits = set()
            cur_clbits = set()

            # Additionally, find the set of classical registers used in conditions over full registers
            # (in such a case, we need to add that register to the block circuit, not just its clbits).
            cur_clregs = set()

            for node in block:
                cur_qubits.update(node.qargs)
                cur_clbits.update(node.cargs)
                cond = getattr(node.op, "_condition", None)
                if cond is not None:
                    cur_clbits.update(condition_resources(cond).clbits)
                    if isinstance(cond[0], ClassicalRegister):
                        cur_clregs.add(cond[0])

            # For reproducibility, order these qubits/clbits compatibly with the global order.
            sorted_qubits = sorted(cur_qubits, key=lambda x: global_index_map[x])
            sorted_clbits = sorted(cur_clbits, key=lambda x: global_index_map[x])

            qc = QuantumCircuit(sorted_qubits, sorted_clbits)

            # Add classical registers used in conditions over registers
            for reg in cur_clregs:
                qc.add_register(reg)

            # Construct a quantum circuit from the nodes in the block, remapping the qubits.
            wire_pos_map = {qb: ix for ix, qb in enumerate(sorted_qubits)}
            wire_pos_map.update({qb: ix for ix, qb in enumerate(sorted_clbits)})

            for node in block:
                qc.append(CircuitInstruction(node.op, node.qargs, node.cargs))

            # Collapse this quantum circuit into an operation.
            op = collapse_fn(qc)

            # Replace the block of nodes in the DAG by the constructed operation
            # (the function replace_block_with_op is implemented both in DAGCircuit and DAGDependency).
            self.dag.replace_block_with_op(block, op, wire_pos_map, cycle_check=False)
        return self.dag
