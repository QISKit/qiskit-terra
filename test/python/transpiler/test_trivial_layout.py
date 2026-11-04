# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the TrivialLayout pass"""

import unittest

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.target import Target
from qiskit.circuit.library import CXGate
from qiskit.transpiler import TranspilerError
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase

from ..legacy_cmaps import RUESCHLIKON_CMAP, TENERIFE_CMAP


class TestTrivialLayout(QiskitTestCase):
    """Tests the TrivialLayout pass"""

    def setUp(self):
        super().setUp()
        self.cmap5 = TENERIFE_CMAP
        self.cmap16 = RUESCHLIKON_CMAP

    def test_3q_circuit_5q_coupling(self):
        """Test finds trivial layout for 3q circuit on 5q device."""
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[1], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = TrivialLayout(CouplingMap(self.cmap5))
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        for i in range(3):
            self.assertEqual(layout[qr[i]], i)

    def test_3q_circuit_5q_coupling_with_target(self):
        """Test finds trivial layout for 3q circuit on 5q device."""
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[1], qr[2])

        dag = circuit_to_dag(circuit)
        target = Target()
        target.add_instruction(CXGate(), {tuple(edge): None for edge in self.cmap5})
        pass_ = TrivialLayout(target)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        for i in range(3):
            self.assertEqual(layout[qr[i]], i)

    def test_9q_circuit_16q_coupling(self):
        """Test finds trivial layout for 9q circuit with 2 registers on 16q device."""
        qr0 = QuantumRegister(4, "q0")
        qr1 = QuantumRegister(5, "q1")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr0, qr1, cr)
        circuit.cx(qr0[1], qr0[2])
        circuit.cx(qr0[0], qr1[3])
        circuit.cx(qr1[4], qr0[2])
        circuit.measure(qr1[1], cr[0])
        circuit.measure(qr0[2], cr[1])

        dag = circuit_to_dag(circuit)
        pass_ = TrivialLayout(CouplingMap(self.cmap16))
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        for i in range(4):
            self.assertEqual(layout[qr0[i]], i)
        for i in range(5):
            self.assertEqual(layout[qr1[i]], i + 4)

    def test_raises_wider_circuit(self):
        """Test error is raised if the circuit is wider than coupling map."""
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0, qr1)

        dag = circuit_to_dag(circuit)
        with self.assertRaises(TranspilerError):
            pass_ = TrivialLayout(CouplingMap(self.cmap5))
            pass_.run(dag)


if __name__ == "__main__":
    unittest.main()
