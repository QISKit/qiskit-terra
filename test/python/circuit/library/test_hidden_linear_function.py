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

"""Test library of Hidden Linear Function circuits."""

import unittest
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import HiddenLinearFunction, hidden_linear_function
from qiskit.quantum_info import Operator
from test import QiskitTestCase


class TestHiddenLinearFunctionLibrary(QiskitTestCase):
    """Test library of Hidden Linear Function circuits."""

    def assertHLFIsCorrect(self, hidden_function, hlf):
        """Assert that the HLF circuit produces the correct matrix.

        Number of qubits is equal to the number of rows (or number of columns)
        of hidden_function.
        """
        num_qubits = len(hidden_function)
        hidden_function = np.asarray(hidden_function)
        simulated = Operator(hlf)

        expected = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        for i in range(2**num_qubits):
            i_qiskit = int(bin(i)[2:].zfill(num_qubits)[::-1], 2)
            x_vec = np.asarray(list(map(int, bin(i)[2:].zfill(num_qubits)[::-1])))
            expected[i_qiskit, i_qiskit] = 1j ** (
                np.dot(x_vec.transpose(), np.dot(hidden_function, x_vec))
            )

        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))
        qc = Operator(qc)
        expected = qc.compose(Operator(expected)).compose(qc)
        self.assertTrue(expected.equiv(simulated))

    def test_hlf(self):
        """Test if the HLF matrix produces the right matrix."""
        hidden_function = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        with self.assertWarns(DeprecationWarning):
            hlf = HiddenLinearFunction(hidden_function)
        self.assertHLFIsCorrect(hidden_function, hlf)

    def test_non_symmetric_raises(self):
        """Test that adjacency matrix is required to be symmetric."""
        with self.assertRaises(CircuitError):
            with self.assertWarns(DeprecationWarning):
                HiddenLinearFunction([[1, 1, 0], [1, 0, 1], [1, 1, 1]])

    def test_hlf_function(self):
        """Test if the HLF matrix produces the right matrix."""
        hidden_function = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        hlf = hidden_linear_function(hidden_function)
        self.assertHLFIsCorrect(hidden_function, hlf)

    def test_non_symmetric_raises_function(self):
        """Test that adjacency matrix is required to be symmetric."""
        with self.assertRaises(CircuitError):
            hidden_linear_function([[1, 1, 0], [1, 0, 1], [1, 1, 1]])


if __name__ == "__main__":
    unittest.main()
