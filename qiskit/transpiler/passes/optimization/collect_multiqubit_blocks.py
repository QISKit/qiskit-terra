# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Collect sequences of uninterrupted gates acting on a number of qubits."""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGOpNode, DAGInNode


class CollectMultiQBlocks(AnalysisPass):
    """Collect sequences of uninterrupted gates acting on groups of qubits.
    ``max_block_size`` specifies the maximum number of qubits that can be acted upon
    by any single group of gates

    Traverse the DAG and find blocks of gates that act consecutively on
    groups of qubits. Write the blocks to ``property_set`` as a list of blocks
    of the form::

        [[g0, g1, g2], [g4, g5]]

    Blocks are reported in a valid topological order. Further, the gates
    within each block are also reported in topological order
    Some gates may not be present in any block (e.g. if the number
    of operands is greater than ``max_block_size``)

    By default, blocks are collected in the direction from the inputs towards the
    outputs of the DAG. The option ``collect_from_back`` allows to change this
    direction, that is to collect blocks from the outputs towards the inputs.
    Note that the blocks are still reported in a valid topological order.

    A Disjoint Set Union data structure (DSU) is used to maintain blocks as
    gates are processed. This data structure points each qubit to a set at all
    times and the sets correspond to current blocks. These change over time
    and the data structure allows these changes to be done quickly.
    """

    def __init__(self, max_block_size=2, collect_from_back=False):
        super().__init__()
        self.parent = {}  # parent array for the union

        # the dicts below are keyed by a qubit signifying the root of a
        #    set in the DSU data structure
        self.bit_groups = {}  # current groups of bits stored at top of trees
        self.gate_groups = {}  # current gate lists for the groups

        self.max_block_size = max_block_size  # maximum block size
        self.collect_from_back = collect_from_back  # backward collection

    def find_set(self, index):
        """DSU function for finding root of set of items
        If my parent is myself, I am the root. Otherwise we recursively
        find the root for my parent. After that, we assign my parent to be
        my root, saving recursion in the future.
        """

        if index not in self.parent:
            self.parent[index] = index
            self.bit_groups[index] = [index]
            self.gate_groups[index] = []
        if self.parent[index] == index:
            return index
        self.parent[index] = self.find_set(self.parent[index])
        return self.parent[index]

    def union_set(self, set1, set2):
        """DSU function for unioning two sets together
        Find the roots of each set. Then assign one to have the other
        as its parent, thus linking the sets.
        Merges smaller set into larger set in order to have better runtime
        """

        set1 = self.find_set(set1)
        set2 = self.find_set(set2)
        if set1 == set2:
            return
        if len(self.gate_groups[set1]) < len(self.gate_groups[set2]):
            set1, set2 = set2, set1
        self.parent[set2] = set1
        self.gate_groups[set1].extend(self.gate_groups[set2])
        self.bit_groups[set1].extend(self.bit_groups[set2])
        self.gate_groups[set2].clear()
        self.bit_groups[set2].clear()

    def run(self, dag):
        """Run the CollectMultiQBlocks pass on `dag`.

        The blocks contain "op" nodes in topological sort order
        such that all gates in a block act on the same set of
        qubits and are adjacent in the circuit.

        The blocks are built by examining predecessors and successors of
        "cx" gates in the circuit. u1, u2, u3, cx, id gates will be included.

        After the execution, ``property_set['block_list']`` is set to
        a list of tuples of ``DAGNode`` objects
        """

        self.parent = {}  # reset all variables on run
        self.bit_groups = {}
        self.gate_groups = {}

        block_list = []

        def collect_key(x):
            """special key function for topological ordering.
            Heuristic for this is to push all gates involving measurement
            or barriers, etc. as far back as possible (because they force
            blocks to end). After that, we process gates in order of lowest
            number of qubits acted on to largest number of qubits acted on
            because these have less chance of increasing the size of blocks
            The key also processes all the non operation nodes first so that
            input nodes do not mess with the top sort of op nodes
            """
            if isinstance(x, DAGInNode):
                return "a"
            if not isinstance(x, DAGOpNode):
                return "d"
            if isinstance(x.op, Gate):
                if x.op.is_parameterized() or getattr(x.op, "_condition", None) is not None:
                    return "c"
                return "b" + chr(ord("a") + len(x.qargs))
            return "d"

        op_nodes = dag.topological_op_nodes(key=collect_key)

        # When collecting from the back, the order of nodes is reversed
        if self.collect_from_back:
            op_nodes = reversed(list(op_nodes))

        for nd in op_nodes:
            can_process = True
            makes_too_big = False

            # check if the node is a gate and if it is parameterized
            if (
                getattr(nd.op, "_condition", None) is not None
                or nd.op.is_parameterized()
                or not isinstance(nd.op, Gate)
            ):
                can_process = False

            cur_qubits = {dag.find_bit(bit).index for bit in nd.qargs}

            if can_process:
                # if the gate is valid, check if grouping up the bits
                # in the gate would fit within our desired max size
                c_tops = set()
                for bit in cur_qubits:
                    c_tops.add(self.find_set(bit))
                tot_size = 0
                for group in c_tops:
                    tot_size += len(self.bit_groups[group])
                if tot_size > self.max_block_size:
                    makes_too_big = True

            if not can_process:
                # resolve the case where we cannot process this node
                for bit in cur_qubits:
                    # create a gate out of me
                    bit = self.find_set(bit)
                    if len(self.gate_groups[bit]) == 0:
                        continue
                    block_list.append(self.gate_groups[bit][:])
                    cur_set = set(self.bit_groups[bit])
                    for v in cur_set:
                        # reset this bit
                        self.parent[v] = v
                        self.bit_groups[v] = [v]
                        self.gate_groups[v] = []

            if makes_too_big:
                # adding in all of the new qubits would make the group too big
                # we must block off sub portions of the groups until the new
                # group would no longer be too big
                savings = {}
                tot_size = 0
                for bit in cur_qubits:
                    top = self.find_set(bit)
                    if top in savings:
                        savings[top] = savings[top] - 1
                    else:
                        savings[top] = len(self.bit_groups[top]) - 1
                        tot_size += len(self.bit_groups[top])
                slist = []
                for item, value in savings.items():
                    slist.append((value, item))
                slist.sort(reverse=True)
                savings_need = tot_size - self.max_block_size
                for item in slist:
                    # remove groups until the size created would be acceptable
                    # start with blocking out the group that would decrease
                    # the new size the most. This heuristic for which blocks we
                    # create does not necessarily give the optimal blocking. Other
                    # heuristics may be worth considering
                    if savings_need > 0:
                        savings_need = savings_need - item[0]
                        if len(self.gate_groups[item[1]]) >= 1:
                            block_list.append(self.gate_groups[item[1]][:])
                        cur_set = set(self.bit_groups[item[1]])
                        for v in cur_set:
                            self.parent[v] = v
                            self.bit_groups[v] = [v]
                            self.gate_groups[v] = []

            if can_process:
                # if the operation is a gate, either skip it if it is too large
                # or group up all of the qubits involved in the gate
                if len(cur_qubits) > self.max_block_size:
                    # gates acting on more qubits than max_block_size cannot
                    #   be a part of any block and thus we skip them here.
                    # we have already finalized the blocks involving the gate's
                    #   qubits in the above makes_too_big block
                    continue  # unable to be part of a group
                prev = -1
                for bit in cur_qubits:
                    if prev != -1:
                        self.union_set(prev, bit)
                    prev = bit
                self.gate_groups[self.find_set(prev)].append(nd)
        # need to turn all groups that still exist into their own blocks
        for index, item in self.parent.items():
            if item == index and len(self.gate_groups[index]) != 0:
                block_list.append(self.gate_groups[index][:])

        # When collecting from the back, both the order of the blocks
        # and the order of nodes in each block should be reversed.
        if self.collect_from_back:
            block_list = [block[::-1] for block in block_list[::-1]]

        self.property_set["block_list"] = block_list

        return dag
