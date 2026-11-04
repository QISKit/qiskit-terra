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

"""ResourceEstimation pass testing"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ResourceEstimation
from test import QiskitTestCase


class TestResourceEstimationPass(QiskitTestCase):
    """Tests for PropertySet methods."""

    def test_empty_dag(self):
        """Empty DAG."""
        circuit = QuantumCircuit()
        passmanager = PassManager()
        passmanager.append(ResourceEstimation())
        passmanager.run(circuit)

        self.assertEqual(passmanager.property_set["size"], 0)
        self.assertEqual(passmanager.property_set["depth"], 0)
        self.assertEqual(passmanager.property_set["width"], 0)
        self.assertDictEqual(passmanager.property_set["count_ops"], {})

    def test_just_qubits(self):
        """A dag with 8 operations and no classic bits"""
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])

        passmanager = PassManager()
        passmanager.append(ResourceEstimation())
        passmanager.run(circuit)

        self.assertEqual(passmanager.property_set["size"], 8)
        self.assertEqual(passmanager.property_set["depth"], 7)
        self.assertEqual(passmanager.property_set["width"], 2)
        self.assertDictEqual(passmanager.property_set["count_ops"], {"cx": 6, "h": 2})


if __name__ == "__main__":
    unittest.main()
