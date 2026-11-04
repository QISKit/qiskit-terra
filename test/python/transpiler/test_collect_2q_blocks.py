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

"""
Tests for the Collect2qBlocks transpiler pass.
"""

import unittest
from math import pi

from ddt import ddt, data, unpack

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.circuit.library import CXGate, U1Gate, U2Gate, RXXGate, RXGate, RZGate
from test import QiskitTestCase


@ddt
class TestCollect2qBlocks(QiskitTestCase):
    """
    Tests to verify that blocks of 2q interactions are found correctly.
    """

    def test_blocks_in_topological_order(self):
        """the pass returns blocks in correct topological order
                                                    ______
        q0:--[p]-------.----      q0:-------------|      |--
                       |                 ______   |  U2  |
        q1:--[u]--(+)-(+)---   =  q1:---|      |--|______|--
                   |                    |  U1  |
        q2:--------.--------      q2:---|______|------------
        """
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.p(0.5, qr[0])
        qc.u(0.0, 0.2, 0.6, qr[1])
        qc.cx(qr[2], qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)

        topo_ops = list(dag.topological_op_nodes())
        block_1 = [topo_ops[1], topo_ops[2]]
        block_2 = [topo_ops[0], topo_ops[3]]

        pass_ = Collect2qBlocks()
        pass_.run(dag)
        self.assertTrue(pass_.property_set["block_list"], [block_1, block_2])

    def test_block_interrupted_by_gate(self):
        """Test that blocks interrupted by a gate that can't be added
        to the block can be collected correctly

        This was raised in #2775 where a measure in the middle of a block
        stopped the block collection from working properly. This was because
        the pass didn't expect to have measures in the middle of the circuit.

        blocks : [['cx', 'id', 'id', 'id'], ['id', 'cx']]

                ┌───┐┌───┐┌─┐     ┌───┐┌───┐
        q_0: |0>┤ X ├┤ I ├┤M├─────┤ I ├┤ X ├
                └─┬─┘├───┤└╥┘┌───┐└───┘└─┬─┘
        q_1: |0>──■──┤ I ├─╫─┤ I ├───────■──
                     └───┘ ║ └───┘
         c_0: 0 ═══════════╩════════════════

        """
        qc = QuantumCircuit(2, 1)
        qc.cx(1, 0)
        qc.id(0)
        qc.id(1)
        qc.measure(0, 0)
        qc.id(0)
        qc.id(1)
        qc.cx(1, 0)

        dag = circuit_to_dag(qc)
        pass_ = Collect2qBlocks()
        pass_.run(dag)

        # list from Collect2QBlocks of nodes that it should have put into blocks
        good_names = ["cx", "u1", "u2", "u3", "id"]
        dag_nodes = [node for node in dag.topological_op_nodes() if node.name in good_names]

        # we have to convert them to sets as the ordering can be different
        # there is no implied topology in a block, so this isn't an issue
        dag_nodes = [set(dag_nodes[:4]), set(dag_nodes[4:])]
        pass_nodes = [set(bl) for bl in pass_.property_set["block_list"]]

        self.assertEqual(dag_nodes, pass_nodes)

    @unpack
    @data(
        (CXGate(), U1Gate(0.1), U2Gate(0.2, 0.3)),
        (RXXGate(pi / 2), RZGate(0.1), RXGate(pi / 2)),
        (
            Gate("custom2qgate", 2, []),
            Gate("custom1qgate1", 1, []),
            Gate("custom1qgate2", 1, []),
        ),
    )
    def test_collect_arbitrary_gates(self, twoq_gate, oneq_gate1, oneq_gate2):
        """Validate we can collect blocks irrespective of gate types in the circuit."""

        qc = QuantumCircuit(3)

        # Block 1 - q[0] and q[1]
        qc.append(oneq_gate1, [0])
        qc.append(oneq_gate2, [1])
        qc.append(twoq_gate, [0, 1])
        qc.append(oneq_gate1, [0])
        qc.append(oneq_gate2, [1])

        # Block 2 - q[1] and q[2]
        qc.append(oneq_gate1, [1])
        qc.append(oneq_gate2, [2])
        qc.append(twoq_gate, [1, 2])
        qc.append(oneq_gate1, [1])
        qc.append(oneq_gate2, [2])

        # Block 3 - q[0] and q[1]
        qc.append(oneq_gate1, [0])
        qc.append(oneq_gate2, [1])
        qc.append(twoq_gate, [0, 1])
        qc.append(oneq_gate1, [0])
        qc.append(oneq_gate2, [1])

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())

        pass_manager.run(qc)
        self.assertEqual(len(pass_manager.property_set["block_list"]), 3)


if __name__ == "__main__":
    unittest.main()
