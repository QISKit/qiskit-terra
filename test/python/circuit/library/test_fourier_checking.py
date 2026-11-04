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

"""Test library of quantum circuits."""

import unittest
from ddt import ddt, data, unpack
import numpy as np

from qiskit.circuit.library import FourierChecking, fourier_checking
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import Operator
from test import QiskitTestCase


@ddt
class TestFourierCheckingLibrary(QiskitTestCase):
    """Test the Fourier Checking circuit."""

    def assertFourierCheckingIsCorrect(self, f_truth_table, g_truth_table, fc_circuit):
        """Assert that the Fourier Checking circuit produces the correct matrix."""

        simulated = Operator(fc_circuit)

        num_qubits = int(np.log2(len(f_truth_table)))

        # create Hadamard matrix, re-use approach from TestMCMT
        h_i = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        h_tot = np.array([1])
        for _ in range(num_qubits):
            h_tot = np.kron(h_tot, h_i)

        f_mat = np.diag(f_truth_table)
        g_mat = np.diag(g_truth_table)

        expected = np.linalg.multi_dot([h_tot, g_mat, h_tot, f_mat, h_tot])
        expected = Operator(expected)

        self.assertTrue(expected.equiv(simulated))

    @data(([1, -1, -1, -1], [1, 1, -1, -1]), ([1, 1, 1, 1], [1, 1, 1, 1]))
    @unpack
    def test_fourier_checking(self, f_truth_table, g_truth_table):
        """Test if the Fourier Checking circuit produces the correct matrix."""
        with self.assertWarns(DeprecationWarning):
            fc_circuit = FourierChecking(f_truth_table, g_truth_table)
        self.assertFourierCheckingIsCorrect(f_truth_table, g_truth_table, fc_circuit)

    @data(([1, -1, -1, -1], [1, 1, -1]), ([1], [-1]), ([1, -1, -1, -1, 1], [1, 1, -1, -1, 1]))
    @unpack
    def test_invalid_input_raises(self, f_truth_table, g_truth_table):
        """Test that invalid input truth tables raise an error."""
        with self.assertRaises(CircuitError):
            with self.assertWarns(DeprecationWarning):
                FourierChecking(f_truth_table, g_truth_table)

    @data(([1, -1, -1, -1], [1, 1, -1, -1]), ([1, 1, 1, 1], [1, 1, 1, 1]))
    @unpack
    def test_fourier_checking_function(self, f_truth_table, g_truth_table):
        """Test if the Fourier Checking circuit produces the correct matrix."""
        fc_circuit = fourier_checking(f_truth_table, g_truth_table)
        self.assertFourierCheckingIsCorrect(f_truth_table, g_truth_table, fc_circuit)

    @data(([1, -1, -1, -1], [1, 1, -1]), ([1], [-1]), ([1, -1, -1, -1, 1], [1, 1, -1, -1, 1]))
    @unpack
    def test_invalid_input_raises_function(self, f_truth_table, g_truth_table):
        """Test that invalid input truth tables raise an error."""
        with self.assertRaises(CircuitError):
            fourier_checking(f_truth_table, g_truth_table)


if __name__ == "__main__":
    unittest.main()
