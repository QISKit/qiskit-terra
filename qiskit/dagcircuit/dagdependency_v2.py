# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""_DAGDependencyV2 class for representing non-commutativity in a circuit."""

import itertools
import math
from collections import OrderedDict, namedtuple
from typing import Any
from collections.abc import Generator

import rustworkx as rx

from qiskit.circuit import (
    QuantumRegister,
    ClassicalRegister,
    Qubit,
    Clbit,
    ParameterExpression,
)
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit import Bit
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.dagcircuit.exceptions import DAGDependencyError
from qiskit.circuit.commutation_checker import CommutationChecker


BitLocations = namedtuple("BitLocations", ("index", "registers"))


class _DAGDependencyV2:
    """Object to represent a quantum circuit as a Directed Acyclic Graph (DAG)
    via operation dependencies (i.e. lack of commutation).

    .. warning::

        This is not part of the public API.

    The nodes in the graph are operations represented by quantum gates.
    The edges correspond to non-commutation between two operations
    (i.e. a dependency). A directed edge from node A to node B means that
    operation A does not commute with operation B.
    The object's methods allow circuits to be constructed.

    **Example:**

    Bell circuit with no measurement.

    .. code-block:: text

              ┌───┐
        qr_0: ┤ H ├──■──
              └───┘┌─┴─┐
        qr_1: ─────┤ X ├
                   └───┘

    The dependency DAG for the above circuit is represented by two nodes.
    The first one corresponds to Hadamard gate, the second one to the CNOT gate
    as the gates do not commute there is an edge between the two nodes.

    **Reference:**

    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
    Exact and practical pattern matching for quantum circuit optimization.
    `arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

    """

    def __init__(self):
        """
        Create an empty _DAGDependencyV2.
        """
        # Circuit name
        self.name = None

        # Circuit metadata
        self.metadata = {}

        # Cache of dag op node sort keys
        self._key_cache = {}

        # Directed multigraph whose nodes are operations(gates) and edges
        # represent non-commutativity between two gates.
        self._multi_graph = rx.PyDAG()

        # Map of qreg/creg name to Register object.
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()

        # List of Qubit/Clbit wires that the DAG acts on.
        self.qubits: list[Qubit] = []
        self.clbits: list[Clbit] = []

        # Dictionary mapping of Qubit and Clbit instances to a tuple comprised of
        # 0) corresponding index in dag.{qubits,clbits} and
        # 1) a list of Register-int pairs for each Register containing the Bit and
        # its index within that register.
        self._qubit_indices: dict[Qubit, BitLocations] = {}
        self._clbit_indices: dict[Clbit, BitLocations] = {}

        self._global_phase = 0

        # Map of number of each kind of op, keyed on op name
        self._op_names = {}

        self.duration = None
        self.unit = "dt"

        self.comm_checker = CommutationChecker()

    @property
    def global_phase(self):
        """Return the global phase of the circuit."""
        return self._global_phase

    @global_phase.setter
    def global_phase(self, angle):
        """Set the global phase of the circuit.

        Args:
            angle (float, ParameterExpression): The angle to set the global phase to.
        """
        if isinstance(angle, ParameterExpression):
            self._global_phase = angle
        else:
            # Set the phase to the [0, 2π) interval
            angle = float(angle)
            if not angle:
                self._global_phase = 0
            else:
                self._global_phase = angle % (2 * math.pi)

    def size(self):
        """Returns the number of gates in the circuit"""
        return len(self._multi_graph)

    def depth(self):
        """Return the circuit depth.
        Returns:
            int: the circuit depth
        """
        depth = rx.dag_longest_path_length(self._multi_graph)
        return depth if depth >= 0 else 0

    def width(self):
        """Return the total number of qubits + clbits used by the circuit."""
        return len(self.qubits) + len(self.clbits)

    def num_qubits(self):
        """Return the total number of qubits used by the circuit."""
        return len(self.qubits)

    def num_clbits(self):
        """Return the total number of classical bits used by the circuit."""
        return len(self.clbits)

    def add_qubits(self, qubits):
        """Add individual qubit wires."""
        if any(not isinstance(qubit, Qubit) for qubit in qubits):
            raise DAGDependencyError("not a Qubit instance.")

        duplicate_qubits = set(self.qubits).intersection(qubits)
        if duplicate_qubits:
            raise DAGDependencyError(f"duplicate qubits {duplicate_qubits}")

        for qubit in qubits:
            self.qubits.append(qubit)
            self._qubit_indices[qubit] = BitLocations(len(self.qubits) - 1, [])

    def add_clbits(self, clbits):
        """Add individual clbit wires."""
        if any(not isinstance(clbit, Clbit) for clbit in clbits):
            raise DAGDependencyError("not a Clbit instance.")

        duplicate_clbits = set(self.clbits).intersection(clbits)
        if duplicate_clbits:
            raise DAGDependencyError(f"duplicate clbits {duplicate_clbits}")

        for clbit in clbits:
            self.clbits.append(clbit)
            self._clbit_indices[clbit] = BitLocations(len(self.clbits) - 1, [])

    def add_qreg(self, qreg):
        """Add qubits in a quantum register."""
        if not isinstance(qreg, QuantumRegister):
            raise DAGDependencyError("not a QuantumRegister instance.")
        if qreg.name in self.qregs:
            raise DAGDependencyError(f"duplicate register {qreg.name}")
        self.qregs[qreg.name] = qreg
        existing_qubits = set(self.qubits)
        for j in range(qreg.size):
            if qreg[j] in self._qubit_indices:
                self._qubit_indices[qreg[j]].registers.append((qreg, j))
            if qreg[j] not in existing_qubits:
                self.qubits.append(qreg[j])
                self._qubit_indices[qreg[j]] = BitLocations(
                    len(self.qubits) - 1, registers=[(qreg, j)]
                )

    def add_creg(self, creg):
        """Add clbits in a classical register."""
        if not isinstance(creg, ClassicalRegister):
            raise DAGDependencyError("not a ClassicalRegister instance.")
        if creg.name in self.cregs:
            raise DAGDependencyError(f"duplicate register {creg.name}")
        self.cregs[creg.name] = creg
        existing_clbits = set(self.clbits)
        for j in range(creg.size):
            if creg[j] in self._clbit_indices:
                self._clbit_indices[creg[j]].registers.append((creg, j))
            if creg[j] not in existing_clbits:
                self.clbits.append(creg[j])
                self._clbit_indices[creg[j]] = BitLocations(
                    len(self.clbits) - 1, registers=[(creg, j)]
                )

    def find_bit(self, bit: Bit) -> BitLocations:
        """
        Finds locations in the _DAGDependencyV2, by mapping the Qubit and Clbit to positional index
        BitLocations is defined as: BitLocations = namedtuple("BitLocations", ("index", "registers"))

        Args:
            bit (Bit): The bit to locate.

        Returns:
            namedtuple(int, List[Tuple(Register, int)]): A 2-tuple. The first element (``index``)
                contains the index at which the ``Bit`` can be found (in either
                :obj:`~_DAGDependencyV2.qubits`, :obj:`~_DAGDependencyV2.clbits`, depending on its
                type). The second element (``registers``) is a list of ``(register, index)``
                pairs with an entry for each :obj:`~Register` in the circuit which contains the
                :obj:`~Bit` (and the index in the :obj:`~Register` at which it can be found).

          Raises:
            DAGDependencyError: If the supplied :obj:`~Bit` was of an unknown type.
            DAGDependencyError: If the supplied :obj:`~Bit` could not be found on the circuit.
        """
        try:
            if isinstance(bit, Qubit):
                return self._qubit_indices[bit]
            elif isinstance(bit, Clbit):
                return self._clbit_indices[bit]
            else:
                raise DAGDependencyError(f"Could not locate bit of unknown type: {type(bit)}")
        except KeyError as err:
            raise DAGDependencyError(
                f"Could not locate provided bit: {bit}. Has it been added to the DAGDependency?"
            ) from err

    def apply_operation_back(self, operation, qargs=(), cargs=()):
        """Add a DAGOpNode to the graph and update the edges.

        Args:
            operation (qiskit.circuit.Operation): operation as a quantum gate
            qargs (list[~qiskit.circuit.Qubit]): list of qubits on which the operation acts
            cargs (list[Clbit]): list of classical wires to attach to
        """
        new_node = DAGOpNode(
            op=operation,
            qargs=qargs,
            cargs=cargs,
        )
        new_node._node_id = self._multi_graph.add_node(new_node)
        self._update_edges()
        self._increment_op(new_node.op)

    def _update_edges(self):
        """
        Updates DagDependencyV2 by adding edges to the newly added node (max_node)
        from the previously added nodes.
        For each previously added node (prev_node), an edge from prev_node to max_node
        is added if max_node is "reachable" from prev_node (this means that the two
        nodes can be made adjacent by commuting them with other nodes), but the two nodes
        themselves do not commute.

        Currently. this function is only used when creating a new _DAGDependencyV2 from another
        representation of a circuit, and hence there are no removed nodes (this is why
        iterating over all nodes is fine).
        """
        max_node_id = len(self._multi_graph) - 1
        max_node = self._get_node(max_node_id)

        reachable = [True] * max_node_id

        # Analyze nodes in the reverse topological order.
        # An improvement to the original algorithm is to consider only direct predecessors
        # and to avoid constructing the lists of forward and backward reachable predecessors
        # for every node when not required.
        for prev_node_id in range(max_node_id - 1, -1, -1):
            if reachable[prev_node_id]:
                prev_node = self._get_node(prev_node_id)

                if not self.comm_checker.commute(
                    prev_node.op,
                    prev_node.qargs,
                    prev_node.cargs,
                    max_node.op,
                    max_node.qargs,
                    max_node.cargs,
                ):
                    # If prev_node and max_node do not commute, then we add an edge
                    # between the two, and mark all direct predecessors of prev_node
                    # as not reaching max_node.
                    self._multi_graph.add_edge(prev_node_id, max_node_id, {"commute": False})

                    predecessor_ids = sorted(
                        [node._node_id for node in self.predecessors(self._get_node(prev_node_id))]
                    )
                    for predecessor_id in predecessor_ids:
                        reachable[predecessor_id] = False
            else:
                # If prev_node cannot reach max_node, then none of its predecessors can
                # reach max_node either.
                predecessor_ids = sorted(
                    [node._node_id for node in self.predecessors(self._get_node(prev_node_id))]
                )
                for predecessor_id in predecessor_ids:
                    reachable[predecessor_id] = False

    def _get_node(self, node_id):
        """
        Args:
            node_id (int): label of considered node.

        Returns:
            node: corresponding to the label.
        """
        return self._multi_graph.get_node_data(node_id)

    def named_nodes(self, *names):
        """Get the set of "op" nodes with the given name."""
        named_nodes = []
        for node in self._multi_graph.nodes():
            if node.name in names:
                named_nodes.append(node)
        return named_nodes

    def _increment_op(self, op):
        if op.name in self._op_names:
            self._op_names[op.name] += 1
        else:
            self._op_names[op.name] = 1

    def _decrement_op(self, op):
        if self._op_names[op.name] == 1:
            del self._op_names[op.name]
        else:
            self._op_names[op.name] -= 1

    def count_ops(self):
        """Count the occurrences of operation names."""
        return self._op_names

    def op_nodes(self):
        """
        Returns:
            generator(dict): iterator over all the nodes.
        """
        return iter(self._multi_graph.nodes())

    def topological_nodes(self, key=None) -> Generator[DAGOpNode, Any, Any]:
        """
        Yield nodes in topological order.

        Args:
            key (Callable): A callable which will take a DAGNode object and
                return a string sort key. If not specified the
                :attr:`~qiskit.dagcircuit.DAGNode.sort_key` attribute will be
                used as the sort key for each node.

        Returns:
            generator(DAGOpNode): node in topological order
        """

        def _key(x):
            return ",".join(
                f"{self.find_bit(q).index:04d}" for q in itertools.chain(x.qargs, x.cargs)
            )

        if key is None:
            key = _key

        return iter(rx.lexicographical_topological_sort(self._multi_graph, key=key))

    def topological_op_nodes(self, key=None) -> Generator[DAGOpNode, Any, Any]:
        """
        Yield nodes in topological order. This is a wrapper for topological_nodes since
        all nodes are op nodes. It's here so that calls to dag.topological_op_nodes can
        use either DAGCircuit or _DAGDependencyV2.

        Returns:
            generator(DAGOpNode): nodes in topological order.
        """
        return self.topological_nodes(key)

    def successors(self, node):
        """Returns iterator of the successors of a node as DAGOpNodes."""
        return iter(self._multi_graph.successors(node._node_id))

    def predecessors(self, node):
        """Returns iterator of the predecessors of a node as DAGOpNodes."""
        return iter(self._multi_graph.predecessors(node._node_id))

    def is_successor(self, node, node_succ):
        """Checks if a second node is in the successors of node."""
        return self._multi_graph.has_edge(node._node_id, node_succ._node_id)

    def is_predecessor(self, node, node_pred):
        """Checks if a second node is in the predecessors of node."""
        return self._multi_graph.has_edge(node_pred._node_id, node._node_id)

    def ancestors(self, node):
        """Returns set of the ancestors of a node as DAGOpNodes."""
        return {self._multi_graph[x] for x in rx.ancestors(self._multi_graph, node._node_id)}

    def descendants(self, node):
        """Returns set of the descendants of a node as DAGOpNodes."""
        return {self._multi_graph[x] for x in rx.descendants(self._multi_graph, node._node_id)}

    def bfs_successors(self, node):
        """
        Returns an iterator of tuples of (DAGOpNode, [DAGOpNodes]) where the DAGOpNode is the
        current node and [DAGOpNode] is its successors in  BFS order.
        """
        return iter(rx.bfs_successors(self._multi_graph, node._node_id))

    def copy_empty_like(self):
        """Return a copy of self with the same structure but empty.

        That structure includes:
            * name and other metadata
            * global phase
            * duration
            * all the qubits and clbits, including the registers.

        Returns:
            _DAGDependencyV2: An empty copy of self.
        """
        target_dag = _DAGDependencyV2()
        target_dag.name = self.name
        target_dag._global_phase = self._global_phase
        target_dag.metadata = self.metadata
        target_dag._key_cache = self._key_cache
        target_dag.comm_checker = self.comm_checker

        target_dag.add_qubits(self.qubits)
        target_dag.add_clbits(self.clbits)

        target_dag.duration = self.duration
        target_dag.unit = self.unit

        for qreg in self.qregs.values():
            target_dag.add_qreg(qreg)
        for creg in self.cregs.values():
            target_dag.add_creg(creg)

        return target_dag

    def draw(self, scale=0.7, filename=None, style="color"):
        """
        Draws the _DAGDependencyV2 graph.

        This function needs `pydot <https://github.com/erocarrera/pydot>`, which in turn needs
        Graphviz <https://www.graphviz.org/>` to be installed.

        .. warning::
            This function will call the system Graphviz tool on a file involving user-controllable
            strings (such as gate labels or register names).  It is recommended to only call this
            function on trusted input.

        Args:
            scale (float): scaling factor
            filename (str): file path to save image to (format inferred from name)
            style (str): 'plain': B&W graph
                         'color' (default): color input/output/op nodes

        Returns:
            Ipython.display.Image: if in Jupyter notebook and not saving to file, otherwise None.
        """
        from qiskit.visualization.dag_visualization import dag_drawer

        return dag_drawer(dag=self, scale=scale, filename=filename, style=style)

    def replace_block_with_op(self, node_block, op, wire_pos_map, cycle_check=True):
        """Replace a block of nodes with a single node.

        This is used to consolidate a block of DAGOpNodes into a single
        operation. A typical example is a block of CX and SWAP gates consolidated
        into a LinearFunction. This function is an adaptation of a similar
        function from DAGCircuit.

        It is important that such consolidation preserves commutativity assumptions
        present in _DAGDependencyV2. As an example, suppose that every node in a
        block [A, B, C, D] commutes with another node E. Let F be the consolidated
        node, F = A o B o C o D. Then F also commutes with E, and thus the result of
        replacing [A, B, C, D] by F results in a valid _DAGDependencyV2. That is, any
        deduction about commutativity in consolidated _DAGDependencyV2 is correct.
        On the other hand, suppose that at least one of the nodes, say B, does not commute
        with E. Then the consolidated _DAGDependencyV2 would imply that F does not commute
        with E. Even though F and E may actually commute, it is still safe to assume that
        they do not. That is, the current implementation of consolidation may lead to
        suboptimal but not to incorrect results.

        Args:
            node_block (List[DAGOpNode]): A list of dag nodes that represents the
                node block to be replaced
            op (qiskit.circuit.Operation): The operation to replace the
                block with
            wire_pos_map (Dict[~qiskit.circuit.Qubit, int]): The dictionary mapping the qarg to
                the position. This is necessary to reconstruct the qarg order
                over multiple gates in the combined single op node.
            cycle_check (bool): When set to True this method will check that
                replacing the provided ``node_block`` with a single node
                would introduce a cycle (which would invalidate the
                ``_DAGDependencyV2``) and will raise a ``DAGDependencyError`` if a cycle
                would be introduced. This checking comes with a run time
                penalty. If you can guarantee that your input ``node_block`` is
                a contiguous block and won't introduce a cycle when it's
                contracted to a single node, this can be set to ``False`` to
                improve the runtime performance of this method.
        Raises:
            DAGDependencyError: if ``cycle_check`` is set to ``True`` and replacing
                the specified block introduces a cycle or if ``node_block`` is
                empty.
        """
        block_qargs = set()
        block_cargs = set()
        block_ids = [x._node_id for x in node_block]

        # If node block is empty return early
        if not node_block:
            raise DAGDependencyError("Can't replace an empty node_block")

        for nd in node_block:
            block_qargs |= set(nd.qargs)
            block_cargs |= set(nd.cargs)
            cond = getattr(nd.op, "condition", None)
            if cond is not None:
                block_cargs.update(condition_resources(cond).clbits)

        # Create replacement node
        new_node = DAGOpNode(
            op,
            qargs=sorted(block_qargs, key=lambda x: wire_pos_map[x]),
            cargs=sorted(block_cargs, key=lambda x: wire_pos_map[x]),
            dag=self,
        )

        try:
            new_node._node_id = self._multi_graph.contract_nodes(
                block_ids, new_node, check_cycle=cycle_check
            )
        except rx.DAGWouldCycle as ex:
            raise DAGDependencyError(
                "Replacing the specified node block would introduce a cycle"
            ) from ex

        self._increment_op(op)

        for nd in node_block:
            self._decrement_op(nd.op)

        return new_node
