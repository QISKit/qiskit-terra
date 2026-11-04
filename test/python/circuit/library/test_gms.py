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

"""Test library of Global Mølmer–Sørensen gate."""

import unittest
import numpy as np

from qiskit.circuit.library import GMS, RXXGate
from qiskit.quantum_info import Operator
from test import QiskitTestCase


class TestGMSLibrary(QiskitTestCase):
    """Test library of Global Mølmer–Sørensen gate."""

    def test_twoq_equivalence(self):
        """Test GMS on 2 qubits is same as RXX."""
        with self.assertWarns(DeprecationWarning):
            circuit = GMS(num_qubits=2, theta=[[0, np.pi / 3], [0, 0]])
        expected = RXXGate(np.pi / 3)
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))


if __name__ == "__main__":
    unittest.main()
