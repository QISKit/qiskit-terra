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

import math
import unittest

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CollectMultiQBlocks
from test import QiskitTestCase


class TestCollect2qBlocks(QiskitTestCase):
    """
    Tests to verify that blocks of 2q interactions are found correctly.
    """

    def test_blocks_in_topological_order(self):
        """the pass returns blocks in correct topological order
                                                    ______
        q0:--[u1]-------.----      q0:-------------|      |--
                        |                 ______   |  U2  |
        q1:--[u2]--(+)-(+)---   =  q1:---|      |--|______|--
                    |                    |  U1  |
        q2:---------.--------      q2:---|______|------------
        """
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.p(0.5, qr[0])
        qc.u(math.pi / 2, 0.2, 0.6, qr[1])
        qc.cx(qr[2], qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)

        topo_ops = list(dag.topological_op_nodes())
        block_1 = [topo_ops[1], topo_ops[2]]
        block_2 = [topo_ops[0], topo_ops[3]]

        pass_ = CollectMultiQBlocks()
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
        pass_ = CollectMultiQBlocks()
        pass_.run(dag)

        # list from Collect2QBlocks of nodes that it should have put into blocks
        good_names = ["cx", "u1", "u2", "u3", "id"]
        dag_nodes = [node for node in dag.topological_op_nodes() if node.name in good_names]

        # we have to convert them to sets as the ordering can be different
        # there is no implied topology in a block, so this isn't an issue
        dag_nodes = [set(dag_nodes[:4]), set(dag_nodes[4:])]

        pass_nodes = [set(bl) for bl in pass_.property_set["block_list"]]

        self.assertEqual(dag_nodes, pass_nodes)

    def test_block_with_classical_register(self):
        """Test that only blocks that share quantum wires are added to the block.
        It was the case that gates which shared a classical wire could be added to
        the same block, despite not sharing the same qubits. This was fixed in #2956.

                                    ┌─────────────────────┐
        q_0: |0>────────────────────┤ U2(0.25*pi,0.25*pi) ├
                     ┌─────────────┐└──────────┬──────────┘
        q_1: |0>──■──┤ U1(0.25*pi) ├───────────┼───────────
                ┌─┴─┐└──────┬──────┘           │
        q_2: |0>┤ X ├───────┼──────────────────┼───────────
                └───┘    ┌──┴──┐            ┌──┴──┐
        c0_0: 0 ═════════╡ = 0 ╞════════════╡ = 0 ╞════════
                         └─────┘            └─────┘

        Previously the blocks collected were : [['cx', 'u1', 'u2']]
        This is now corrected to : [['cx', 'u1']]
        """

        qasmstr = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c0[1];

        cx q[1],q[2];
        if(c0==0) u1(0.25*pi) q[1];
        if(c0==0) u2(0.25*pi, 0.25*pi) q[0];
        """
        qc = QuantumCircuit.from_qasm_str(qasmstr)

        pass_manager = PassManager()
        pass_manager.append(CollectMultiQBlocks())

        pass_manager.run(qc)

        self.assertEqual(
            [["cx"]], [[n.name for n in block] for block in pass_manager.property_set["block_list"]]
        )

    def test_do_not_go_across_barrier(self):
        """Validate that blocks are not collected across barriers
                   ░
        q_0: ──■───░───■──
             ┌─┴─┐ ░ ┌─┴─┐
        q_1: ┤ X ├─░─┤ X ├
             └───┘ ░ └───┘
        q_2: ──────░──────
                   ░
        """
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)
        qc.barrier()
        qc.cx(0, 1)

        pass_manager = PassManager()
        pass_manager.append(CollectMultiQBlocks())

        pass_manager.run(qc)
        for block in pass_manager.property_set["block_list"]:
            self.assertTrue(len(block) <= 1)

    def test_optimal_blocking(self):
        """Test that blocks are created optimally in at least the two quibit case.
        Here, if the topological ordering of nodes is wrong then we might create
        an extra block
              ┌───┐
        qr_0: ┤ X ├───────■────■───────
              └───┘┌───┐┌─┴─┐┌─┴─┐┌───┐
        qr_1: ──■──┤ H ├┤ X ├┤ X ├┤ H ├
              ┌─┴─┐├───┤└───┘└───┘└───┘
        qr_2: ┤ X ├┤ X ├───────────────
              └───┘└───┘
        """
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.x(0)
        qc.cx(1, 2)
        qc.h(1)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(1)
        qc.x(2)

        pass_manager = PassManager()
        pass_manager.append(CollectMultiQBlocks())

        pass_manager.run(qc)
        self.assertTrue(len(pass_manager.property_set["block_list"]) == 2)

    def test_ignore_measurement(self):
        """Test that doing a measurement on one qubit will not prevent
        gates from being added to the block that do not act on the qubit
        that was measured
                       ┌─┐
        q_0: ──■───────┤M├──────────
             ┌─┴─┐     └╥┘     ┌───┐
        q_1: ┤ X ├──■───╫───■──┤ X ├
             └───┘┌─┴─┐ ║ ┌─┴─┐├───┤
        q_2: ─────┤ X ├─╫─┤ X ├┤ H ├
                  └───┘ ║ └───┘└───┘
        c_0: ═══════════╩═══════════
        """
        qc = QuantumCircuit(3, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure(0, 0)
        qc.cx(1, 2)
        qc.x(1)
        qc.h(2)

        pass_manager = PassManager()
        pass_manager.append(CollectMultiQBlocks(max_block_size=3))

        pass_manager.run(qc)
        self.assertTrue(len(pass_manager.property_set["block_list"]) == 1)

    def test_larger_blocks(self):
        """Test that a max block size of 4 is still being processed
        reasonably. Currently, this test just makes sure that the circuit can be run.
        This is because the current multiqubit block collector is not optimal for this case
        q_0: ──■──────────────■───────
             ┌─┴─┐            │
        q_1: ┤ X ├──■─────────■───────
             └───┘┌─┴─┐     ┌─┴─┐
        q_2: ─────┤ X ├──■──┤ X ├─────
                  └───┘┌─┴─┐└───┘
        q_3: ──────────┤ X ├──■────■──
                       └───┘┌─┴─┐┌─┴─┐
        q_4: ───────────────┤ X ├┤ X ├
                            └───┘└───┘
        """
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.ccx(0, 1, 2)
        qc.cx(3, 4)

        pass_manager = PassManager()
        pass_manager.append(CollectMultiQBlocks(max_block_size=4))

        pass_manager.run(qc)

    def test_collect_from_back(self):
        """Test the option to collect blocks from the outputs towards
        the inputs.
             ┌───┐
        q_0: ┤ H ├──■────■────■───────
             └───┘┌─┴─┐  │    │
        q_1: ─────┤ X ├──┼────┼───────
                  └───┘┌─┴─┐  │
        q_2: ──────────┤ X ├──┼───────
                       └───┘┌─┴─┐┌───┐
        q_3: ───────────────┤ X ├┤ H ├
                            └───┘└───┘
        """
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.h(3)

        dag = circuit_to_dag(qc)
        # For the circuit above, the topological order is unique
        topo_ops = list(dag.topological_op_nodes())

        # When collecting blocks of size-3 using the default direction,
        # the first block should contain the H-gate and two CX-gates,
        # and the second block should contain a single CX-gate and an H-gate.
        pass_ = CollectMultiQBlocks(max_block_size=3, collect_from_back=False)
        pass_.run(dag)
        expected_blocks = [[topo_ops[0], topo_ops[1], topo_ops[2]], [topo_ops[3], topo_ops[4]]]
        self.assertEqual(pass_.property_set["block_list"], expected_blocks)

        # When collecting blocks of size-3 using the opposite direction,
        # the first block should contain the H-gate and a single CX-gate,
        # and the second block should contain two CX-gates and an H-gate.
        pass_ = CollectMultiQBlocks(max_block_size=3, collect_from_back=True)
        pass_.run(dag)
        expected_blocks = [[topo_ops[0], topo_ops[1]], [topo_ops[2], topo_ops[3], topo_ops[4]]]
        self.assertEqual(pass_.property_set["block_list"], expected_blocks)


if __name__ == "__main__":
    unittest.main()
