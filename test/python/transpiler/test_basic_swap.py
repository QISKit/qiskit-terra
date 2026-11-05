# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the BasicSwap pass"""

import unittest
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler import CouplingMap, Target
from qiskit.circuit.library import CXGate
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit
from test import QiskitTestCase


class TestBasicSwap(QiskitTestCase):
    """Tests the BasicSwap pass."""

    def test_trivial_case(self):
        """No need to have any swap, the CX are distance 1 to each other
        q0:--(+)-[U]-(+)-
              |       |
        q1:---.-------|--
                      |
        q2:-----------.--

        CouplingMap map: [1]--[0]--[2]
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_trivial_in_same_layer(self):
        """No need to have any swap, two CXs distance 1 to each other, in the same layer
        q0:--(+)--
              |
        q1:---.---

        q2:--(+)--
              |
        q3:---.---

        CouplingMap map: [0]--[1]--[2]--[3]
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[0], qr[1])

        dag = circuit_to_dag(circuit)
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_a_single_swap(self):
        """Adding a swap
        q0:-------

        q1:--(+)--
              |
        q2:---.---

        CouplingMap map: [1]--[0]--[2]

        q0:--X---.---
             |   |
        q1:--X---|---
                 |
        q2:-----(+)--

        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[0])
        expected.cx(qr[0], qr[2])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_a_single_swap_with_target(self):
        """Adding a swap
        q0:-------

        q1:--(+)--
              |
        q2:---.---

        CouplingMap map: [1]--[0]--[2]

        q0:--X---.---
             |   |
        q1:--X---|---
                 |
        q2:-----(+)--

        """
        target = Target()
        target.add_instruction(CXGate(), {(0, 1): None, (0, 2): None})

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[0])
        expected.cx(qr[0], qr[2])

        pass_ = BasicSwap(target)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_a_single_swap_bigger_cm(self):
        """Swapper in a bigger coupling map
        q0:-------

        q1:---.---
              |
        q2:--(+)--

        CouplingMap map: [1]--[0]--[2]--[3]

        q0:--X---.---
             |   |
        q1:--X---|---
                 |
        q2:-----(+)--

        """
        coupling = CouplingMap([[0, 1], [0, 2], [2, 3]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[0])
        expected.cx(qr[0], qr[2])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_keep_layout(self):
        """After a swap, the following gates also change the wires.
        qr0:---.---[H]--
               |
        qr1:---|--------
               |
        qr2:--(+)-------

        CouplingMap map: [0]--[1]--[2]

        qr0:--X-----------
              |
        qr1:--X---.--[H]--
                  |
        qr2:-----(+)------
        """
        coupling = CouplingMap([[1, 0], [1, 2]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.cx(qr[1], qr[2])
        expected.h(qr[1])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap(self):
        """A far swap that affects coming CXs.
        qr0:--(+)---.--
               |    |
        qr1:---|----|--
               |    |
        qr2:---|----|--
               |    |
        qr3:---.---(+)-

        CouplingMap map: [0]--[1]--[2]--[3]

        qr0:--X--------------
              |
        qr1:--X--X-----------
                 |
        qr2:-----X--(+)---.--
                     |    |
        qr3:---------.---(+)-

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[3], qr[2])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap_with_gate_the_front(self):
        """A far swap with a gate in the front.
        q0:------(+)--
                  |
        q1:-------|---
                  |
        q2:-------|---
                  |
        q3:--[H]--.---

        CouplingMap map: [0]--[1]--[2]--[3]

        q0:-----------(+)--
                       |
        q1:---------X--.---
                    |
        q2:------X--X------
                 |
        q3:-[H]--X---------

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr[3])
        expected.swap(qr[3], qr[2])
        expected.swap(qr[2], qr[1])
        expected.cx(qr[1], qr[0])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap_with_gate_the_back(self):
        """A far swap with a gate in the back.
        q0:--(+)------
              |
        q1:---|-------
              |
        q2:---|-------
              |
        q3:---.--[H]--

        CouplingMap map: [0]--[1]--[2]--[3]

        q0:-------(+)------
                   |
        q1:-----X--.--[H]--
                |
        q2:--X--X----------
             |
        q3:--X-------------

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[3], qr[2])
        expected.swap(qr[2], qr[1])
        expected.cx(qr[1], qr[0])
        expected.h(qr[1])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap_with_gate_the_middle(self):
        """A far swap with a gate in the middle.
        q0:--(+)-------.--
              |        |
        q1:---|--------|--
              |
        q2:---|--------|--
              |        |
        q3:---.--[H]--(+)-

        CouplingMap map: [0]--[1]--[2]--[3]

        q0:-------(+)-------.---
                   |        |
        q1:-----X--.--[H]--(+)--
                |
        q2:--X--X---------------
             |
        q3:--X------------------

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        circuit.cx(qr[0], qr[3])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[3], qr[2])
        expected.swap(qr[2], qr[1])
        expected.cx(qr[1], qr[0])
        expected.h(qr[1])
        expected.cx(qr[0], qr[1])

        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_fake_run(self):
        """A fake run, doesn't change dag
        q0:--(+)-------.--
              |        |
        q1:---|--------|--
              |
        q2:---|--------|--
              |        |
        q3:---.--[H]--(+)-

        CouplingMap map: [0]--[1]--[2]--[3]

        q0:-------(+)-------.---
                   |        |
        q1:-----X--.--[H]--(+)--
                |
        q2:--X--X---------------
             |
        q3:--X------------------

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        circuit.cx(qr[0], qr[3])

        fake_pm = PassManager([BasicSwap(coupling, fake_run=True)])
        real_pm = PassManager([BasicSwap(coupling, fake_run=False)])

        self.assertEqual(circuit, fake_pm.run(circuit))
        self.assertNotEqual(circuit, real_pm.run(circuit))
        self.assertIsInstance(fake_pm.property_set["final_layout"], Layout)
        self.assertEqual(fake_pm.property_set["final_layout"], real_pm.property_set["final_layout"])

    def test_preserves_global_phase(self):
        """The global phase is preserved when no swaps are needed."""
        coupling = CouplingMap.from_line(3)

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.global_phase = 0.5
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])

        dag = circuit_to_dag(circuit)
        after = BasicSwap(coupling).run(dag)

        self.assertEqual(after.global_phase, dag.global_phase)

    def test_preserves_global_phase_when_swaps_inserted(self):
        """The global phase is preserved when routing inserts swaps."""
        coupling = CouplingMap.from_line(3)

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.global_phase = 0.5
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(after.global_phase, dag.global_phase)
        self.assertEqual(after.count_ops().get("swap", 0), 1)
        self.assertIsInstance(pass_.property_set["final_layout"], Layout)


if __name__ == "__main__":
    unittest.main()
