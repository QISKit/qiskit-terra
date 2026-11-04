# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Search for star connectivity patterns and replace them with."""
import itertools

from collections.abc import Iterable
from math import floor, log10

from qiskit.circuit import Barrier
from qiskit.circuit.library import SwapGate
from qiskit.dagcircuit import (
    DAGOpNode,
    DAGOutNode,
    DAGInNode,
    DAGDepNode,
    DAGDependency,
    DAGCircuit,
)
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout


class StarBlock:
    """Defines blocks representing star-shaped pieces of a circuit."""

    def __init__(self, nodes=None, center=None, num2q=0):
        self.center = center
        self.num2q = num2q
        self.nodes = [] if nodes is None else nodes

    def get_nodes(self):
        """Returns the list of nodes used in the block."""
        return self.nodes

    def append_node(self, node):
        """
        If node can be added to block while keeping the block star-shaped, and
        return True. Otherwise, does not add node to block and returns False.
        """

        added = False

        if len(node.qargs) == 1:
            self.nodes.append(node)
            added = True
        elif self.center is None:
            self.center = set(node.qargs)
            self.nodes.append(node)
            self.num2q += 1
            added = True
        elif isinstance(self.center, set):
            if node.qargs[0] in self.center:
                self.center = node.qargs[0]
                self.nodes.append(node)
                self.num2q += 1
                added = True
            elif node.qargs[1] in self.center:
                self.center = node.qargs[1]
                self.nodes.append(node)
                self.num2q += 1
                added = True
        elif self.center in node.qargs:
            self.nodes.append(node)
            self.num2q += 1
            added = True

        return added

    def size(self):
        """
        Returns the number of two-qubit quantum gates in this block.
        """
        return self.num2q


class StarPreRouting(TransformationPass):
    """Run star to linear pre-routing

    This pass is a logical optimization pass that rewrites any
    solely 2q gate star connectivity subcircuit as a linear connectivity
    equivalent with swaps.

    For example:

      .. plot::
         :alt: Circuit diagram output by the previous code.
         :include-source:

         from qiskit.circuit import QuantumCircuit
         from qiskit.transpiler.passes import StarPreRouting

         qc = QuantumCircuit(10)
         qc.h(0)
         qc.cx(0, range(1, 5))
         qc.h(9)
         qc.cx(9, range(8, 4, -1))
         qc.measure_all()
         StarPreRouting()(qc).draw("mpl")

    This pass was inspired by a similar pass described in Section IV of:
    C. Campbell et al., "Superstaq: Deep Optimization of Quantum Programs,"
    2023 IEEE International Conference on Quantum Computing and Engineering (QCE),
    Bellevue, WA, USA, 2023, pp. 1020-1032, doi: 10.1109/QCE57702.2023.00116.
    """

    def __init__(self):
        """StarPreRouting"""

        self._pending_nodes: list[DAGOpNode | DAGDepNode] | None = None
        self._in_degree: dict[DAGOpNode | DAGDepNode, int] | None = None
        super().__init__()

    def _setup_in_degrees(self, dag):
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
        for node in self._op_nodes(dag):
            deg = len(self._direct_preds(dag, node))
            self._in_degree[node] = deg
            if deg == 0:
                self._pending_nodes.append(node)

    def _op_nodes(self, dag) -> Iterable[DAGOpNode | DAGDepNode]:
        """Returns DAG nodes."""
        if not isinstance(dag, DAGDependency):
            return dag.op_nodes()
        else:
            return dag.get_nodes()

    def _direct_preds(self, dag, node):
        """Returns direct predecessors of a node. This function takes into account the
        direction of collecting blocks, that is node's predecessors when collecting
        backwards are the direct successors of a node in the DAG.
        """
        if not isinstance(dag, DAGDependency):
            return [pred for pred in dag.predecessors(node) if isinstance(pred, DAGOpNode)]
        else:
            return [dag.get_node(pred_id) for pred_id in dag.direct_predecessors(node.node_id)]

    def _direct_succs(self, dag, node):
        """Returns direct successors of a node. This function takes into account the
        direction of collecting blocks, that is node's successors when collecting
        backwards are the direct predecessors of a node in the DAG.
        """
        if not isinstance(dag, DAGDependency):
            return [succ for succ in dag.successors(node) if isinstance(succ, DAGOpNode)]
        else:
            return [dag.get_node(succ_id) for succ_id in dag.direct_successors(node.node_id)]

    def _have_uncollected_nodes(self):
        """Returns whether there are uncollected (pending) nodes"""
        return len(self._pending_nodes) > 0

    def collect_matching_block(self, dag, filter_fn):
        """Iteratively collects the largest block of input nodes (that is, nodes with
        ``_in_degree`` equal to 0) that match a given filtering function.
        Examples of this include collecting blocks of swap gates,
        blocks of linear gates (CXs and SWAPs), blocks of Clifford gates, blocks of single-qubit gates,
        blocks of two-qubit gates, etc.  Here 'iteratively' means that once a node is collected,
        the ``_in_degree`` of each of its immediate successor is decreased by 1, allowing more nodes
        to become input and to be eligible for collecting into the current block.
        Returns the block of collected nodes.
        """
        unprocessed_pending_nodes = self._pending_nodes
        self._pending_nodes = []

        current_block = StarBlock()

        # Iteratively process unprocessed_pending_nodes:
        # - any node that does not match filter_fn is added to pending_nodes
        # - any node that match filter_fn is added to the current_block,
        #   and some of its successors may be moved to unprocessed_pending_nodes.
        while unprocessed_pending_nodes:
            new_pending_nodes = []
            for node in unprocessed_pending_nodes:
                added = filter_fn(node) and current_block.append_node(node)
                if added:
                    # update the _in_degree of node's successors
                    for suc in self._direct_succs(dag, node):
                        self._in_degree[suc] -= 1
                        if self._in_degree[suc] == 0:
                            new_pending_nodes.append(suc)
                else:
                    self._pending_nodes.append(node)
            unprocessed_pending_nodes = new_pending_nodes

        return current_block

    def collect_all_matching_blocks(
        self,
        dag,
        min_block_size=2,
    ):
        """Collects all blocks that match a given filtering function filter_fn.
        This iteratively finds the largest block that does not match filter_fn,
        then the largest block that matches filter_fn, and so on, until no more uncollected
        nodes remain. Intuitively, finding larger blocks of non-matching nodes helps to
        find larger blocks of matching nodes later on. The option ``min_block_size``
        specifies the minimum number of gates in the block for the block to be collected.

        By default, blocks are collected in the direction from the inputs towards the outputs
        of the circuit. The option ``collect_from_back`` allows to change this direction,
        that is collect blocks from the outputs towards the inputs of the circuit.

        Returns the list of matching blocks only.
        """

        def filter_fn(node):
            """Specifies which nodes can be collected into star blocks."""
            return (
                len(node.qargs) <= 2
                and len(node.cargs) == 0
                and getattr(node.op, "condition", None) is None
                and not isinstance(node.op, Barrier)
            )

        def not_filter_fn(node):
            """Returns the opposite of filter_fn."""
            return not filter_fn(node)

        # Note: the collection direction must be specified before setting in-degrees
        self._setup_in_degrees(dag)

        # Iteratively collect non-matching and matching blocks.
        matching_blocks: list[StarBlock] = []
        processing_order = []
        while self._have_uncollected_nodes():
            self.collect_matching_block(dag, filter_fn=not_filter_fn)
            matching_block = self.collect_matching_block(dag, filter_fn=filter_fn)
            if matching_block.size() >= min_block_size:
                matching_blocks.append(matching_block)
            processing_order.append(matching_block)

        processing_order = [n for p in processing_order for n in p.nodes]

        return matching_blocks, processing_order

    def run(self, dag):
        # Extract StarBlocks from DAGCircuit / DAGDependency / DAGDependencyV2
        star_blocks, processing_order = self.determine_star_blocks_processing(dag, min_block_size=2)

        if not star_blocks:
            return dag

        if all(b.size() < 3 for b in star_blocks):
            # we only process blocks with less than 3 two-qubit gates in this pre-routing pass
            # if they occur in a collection of larger stars, otherwise we consider them to be 'lines'
            return dag

        # Create a new DAGCircuit / DAGDependency / DAGDependencyV2, replacing each
        # star block by a linear sequence of gates
        new_dag, qubit_mapping = self.star_preroute(dag, star_blocks, processing_order)

        # Fix output permutation -- copied from ElidePermutations
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping

        new_layout = Layout({dag.qubits[out]: idx for idx, out in enumerate(qubit_mapping)})
        if current_layout := self.property_set["virtual_permutation_layout"]:
            self.property_set["virtual_permutation_layout"] = new_layout.compose(
                current_layout, dag.qubits
            )
        else:
            self.property_set["virtual_permutation_layout"] = new_layout

        return new_dag

    def determine_star_blocks_processing(
        self, dag: DAGCircuit | DAGDependency, min_block_size: int
    ) -> tuple[list[StarBlock], list[DAGOpNode] | list[DAGDepNode]]:
        """Returns star blocks in dag and the processing order of nodes within these star blocks
        Args:
            dag (DAGCircuit or DAGDependency): a dag on which star blocks should be determined.
            min_block_size (int): minimum number of two-qubit gates in a star block.

        Returns:
            List[StarBlock]: a list of star blocks in the given dag
            Union[List[DAGOpNode], List[DAGDepNode]]: a list of operations specifying processing order
        """
        blocks, processing_order = self.collect_all_matching_blocks(
            dag, min_block_size=min_block_size
        )
        return blocks, processing_order

    def star_preroute(self, dag, blocks, processing_order):
        """Returns star blocks in dag and the processing order of nodes within these star blocks
        Args:
            dag (DAGCircuit or DAGDependency): a dag on which star prerouting should be performed.
            blocks (List[StarBlock]): a list of star blocks in the given dag.
            processing_order (Union[List[DAGOpNode], List[DAGDepNode]]): a list of operations specifying
            processing order

        Returns:
            new_dag: a dag specifying the pre-routed circuit
            qubit_mapping: the final qubit mapping after pre-routing
        """
        node_to_block_id = {}
        for i, block in enumerate(blocks):
            for node in block.get_nodes():
                node_to_block_id[node] = i

        new_dag = dag.copy_empty_like()
        processed_block_ids = set()
        qubit_mapping = list(range(len(dag.qubits)))

        def _apply_mapping(qargs, qubit_mapping, qubits):
            return tuple(qubits[qubit_mapping[dag.find_bit(qubit).index]] for qubit in qargs)

        is_first_star = True
        last_2q_gate = [
            op
            for op in reversed(processing_order)
            if ((len(op.qargs) > 1) and (op.name != "barrier"))
        ]
        if len(last_2q_gate) > 0:
            last_2q_gate = last_2q_gate[0]
        else:
            last_2q_gate = None

        int_digits = floor(log10(len(processing_order))) + 1
        processing_order_index_map = {
            node: f"a{str(index).zfill(int(int_digits))}"
            for index, node in enumerate(processing_order)
        }

        def tie_breaker_key(node):
            processing_order = processing_order_index_map.get(node, None)
            if processing_order is not None:
                return processing_order
            if isinstance(node, (DAGInNode, DAGOutNode)):
                return str(node.wire)
            return ",".join(
                f"{dag.find_bit(q).index:04d}" for q in itertools.chain(node.qargs, node.cargs)
            )

        for node in dag.topological_op_nodes(key=tie_breaker_key):
            block_id = node_to_block_id.get(node, None)
            if block_id is not None:
                if block_id in processed_block_ids:
                    continue

                processed_block_ids.add(block_id)

                # process the whole block
                block = blocks[block_id]
                sequence = block.nodes
                center_node = block.center

                if len(sequence) == 2:
                    for inner_node in sequence:
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                            check=False,
                        )
                    continue
                swap_source = None
                prev = None
                for inner_node in sequence:
                    if (len(inner_node.qargs) == 1) or (inner_node.qargs == prev):
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                            check=False,
                        )
                        continue
                    if is_first_star and swap_source is None:
                        swap_source = center_node
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                            check=False,
                        )

                        prev = inner_node.qargs
                        continue
                    # place 2q-gate and subsequent swap gate
                    new_dag.apply_operation_back(
                        inner_node.op,
                        _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                        inner_node.cargs,
                        check=False,
                    )

                    if inner_node is not last_2q_gate and not isinstance(inner_node.op, Barrier):
                        new_dag.apply_operation_back(
                            SwapGate(),
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                            check=False,
                        )
                        # Swap mapping
                        index_0 = dag.find_bit(inner_node.qargs[0]).index
                        index_1 = dag.find_bit(inner_node.qargs[1]).index
                        qubit_mapping[index_1], qubit_mapping[index_0] = (
                            qubit_mapping[index_0],
                            qubit_mapping[index_1],
                        )

                    prev = inner_node.qargs
                is_first_star = False
            else:
                # the node is not part of a block
                new_dag.apply_operation_back(
                    node.op,
                    _apply_mapping(node.qargs, qubit_mapping, dag.qubits),
                    node.cargs,
                    check=False,
                )
        return new_dag, qubit_mapping
