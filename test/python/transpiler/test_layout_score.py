# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Layout Score pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.transpiler.passes import Layout2qDistance
from qiskit.transpiler import CouplingMap, Layout
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.target import Target
from test import QiskitTestCase


class TestLayoutScoreError(QiskitTestCase):
    """Test error-ish of Layout Score"""

    def test_no_layout(self):
        """No Layout. Empty Circuit CouplingMap map: None. Result: None"""
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        coupling = CouplingMap()
        layout = None

        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(coupling)
        pass_.property_set["layout"] = layout
        pass_.run(dag)

        self.assertIsNone(pass_.property_set["layout_score"])


class TestTrivialLayoutScore(QiskitTestCase):
    """Trivial layout scenarios"""

    def test_no_cx(self):
        """Empty Circuit CouplingMap map: None. Result: 0"""
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        coupling = CouplingMap()
        layout = Layout().generate_trivial_layout(qr)

        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(coupling)
        pass_.property_set["layout"] = layout
        pass_.run(dag)

        self.assertEqual(pass_.property_set["layout_score"], 0)

    def test_swap_mapped_true(self):
        """Mapped circuit. Good Layout
        qr0 (0):--(+)---(+)-
                   |     |
        qr1 (1):---.-----|--
                         |
        qr2 (2):---------.--

        CouplingMap map: [1]--[0]--[2]
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        coupling = CouplingMap([[0, 1], [0, 2]])
        layout = Layout().generate_trivial_layout(qr)

        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(coupling)
        pass_.property_set["layout"] = layout
        pass_.run(dag)

        self.assertEqual(pass_.property_set["layout_score"], 0)

    def test_swap_mapped_false(self):
        """Needs [0]-[1] in a [0]--[2]--[1] Result:1
        qr0:--(+)--
               |
        qr1:---.---

        CouplingMap map: [0]--[2]--[1]
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = CouplingMap([[0, 2], [2, 1]])
        layout = Layout().generate_trivial_layout(qr)

        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(coupling)
        pass_.property_set["layout"] = layout
        pass_.run(dag)

        self.assertEqual(pass_.property_set["layout_score"], 1)

    def test_swap_mapped_true_target(self):
        """Mapped circuit. Good Layout
        qr0 (0):--(+)---(+)-
                   |     |
        qr1 (1):---.-----|--
                         |
        qr2 (2):---------.--

        CouplingMap map: [1]--[0]--[2]
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        target = Target()
        target.add_instruction(CXGate(), {(0, 1): None, (0, 2): None})
        layout = Layout().generate_trivial_layout(qr)

        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(target)
        pass_.property_set["layout"] = layout
        pass_.run(dag)

        self.assertEqual(pass_.property_set["layout_score"], 0)

    def test_swap_mapped_false_target(self):
        """Needs [0]-[1] in a [0]--[2]--[1] Result:1
        qr0:--(+)--
               |
        qr1:---.---

        CouplingMap map: [0]--[2]--[1]
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        target = Target()
        target.add_instruction(CXGate(), {(0, 2): None, (2, 1): None})

        layout = Layout().generate_trivial_layout(qr)

        dag = circuit_to_dag(circuit)
        pass_ = Layout2qDistance(target)
        pass_.property_set["layout"] = layout
        pass_.run(dag)

        self.assertEqual(pass_.property_set["layout_score"], 1)


if __name__ == "__main__":
    unittest.main()
