# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the global rotation circuit."""

import unittest
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GR, GRX, GRY, GRZ, RGate, RZGate
from test import QiskitTestCase


class TestGlobalRLibrary(QiskitTestCase):
    """Test library of global R gates."""

    def test_gr_equivalence(self):
        """Test global R gate is same as 3 individual R gates."""
        circuit = GR(num_qubits=3, theta=np.pi / 3, phi=2 * np.pi / 3)
        expected = QuantumCircuit(3, name="gr")
        for i in range(3):
            expected.append(RGate(theta=np.pi / 3, phi=2 * np.pi / 3), [i])
        self.assertEqual(expected, circuit.decompose())

    def test_grx_equivalence(self):
        """Test global RX gates is same as 3 individual RX gates."""
        circuit = GRX(num_qubits=3, theta=np.pi / 3)
        expected = GR(num_qubits=3, theta=np.pi / 3, phi=0)
        self.assertEqual(expected, circuit)

    def test_gry_equivalence(self):
        """Test global RY gates is same as 3 individual RY gates."""
        circuit = GRY(num_qubits=3, theta=np.pi / 3)
        expected = GR(num_qubits=3, theta=np.pi / 3, phi=np.pi / 2)
        self.assertEqual(expected, circuit)

    def test_grz_equivalence(self):
        """Test global RZ gate is same as 3 individual RZ gates."""
        circuit = GRZ(num_qubits=3, phi=2 * np.pi / 3)
        expected = QuantumCircuit(3, name="grz")
        for i in range(3):
            expected.append(RZGate(phi=2 * np.pi / 3), [i])
        self.assertEqual(expected, circuit)


if __name__ == "__main__":
    unittest.main()
