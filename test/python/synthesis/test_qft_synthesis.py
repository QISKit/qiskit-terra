# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for QFT synthesis methods."""


import unittest
from test import combine
from ddt import ddt, data

from qiskit.circuit.library import QFT
from qiskit.synthesis.qft import synth_qft_line, synth_qft_full
from qiskit.quantum_info import Operator
from qiskit.synthesis.linear.linear_circuits_utils import check_lnn_connectivity
from test import QiskitTestCase


@ddt
class TestQFTLNN(QiskitTestCase):
    """Tests for QFT synthesis functions."""

    @combine(num_qubits=[2, 3, 4, 5, 6, 7, 8], do_swaps=[True, False])
    def test_qft_lnn(self, num_qubits, do_swaps):
        """Assert that the original and synthesized QFT circuits are the same."""
        with self.assertWarns(DeprecationWarning):
            qft_circ = QFT(num_qubits, do_swaps=do_swaps)
        qft_lnn = synth_qft_line(num_qubits, do_swaps=do_swaps)

        with self.subTest(msg="original and synthesized QFT circuits are not the same"):
            self.assertEqual(Operator(qft_circ), Operator(qft_lnn))

        # Check that the output circuit has LNN connectivity
        with self.subTest(msg="synthesized QFT circuit do not have LNN connectivity"):
            self.assertTrue(check_lnn_connectivity(qft_lnn))

    @combine(num_qubits=[5, 6, 7, 8], do_swaps=[True, False], approximation_degree=[2, 3])
    def test_qft_lnn_approximated(self, num_qubits, do_swaps, approximation_degree):
        """Assert that the original and synthesized QFT circuits are the same with approximation."""
        with self.assertWarns(DeprecationWarning):
            qft_circ = QFT(num_qubits, do_swaps=do_swaps, approximation_degree=approximation_degree)
        qft_lnn = synth_qft_line(
            num_qubits, do_swaps=do_swaps, approximation_degree=approximation_degree
        )

        with self.subTest(msg="original and synthesized QFT circuits are not the same"):
            self.assertEqual(Operator(qft_circ), Operator(qft_lnn))

        # Check that the output circuit has LNN connectivity
        with self.subTest(msg="synthesized QFT circuit do not have LNN connectivity"):
            self.assertTrue(check_lnn_connectivity(qft_lnn))

    @data(50, 100, 1000)
    def test_create_large_circuit(self, num_qubits):
        """Test creating large QFT circuits."""
        qft = synth_qft_line(num_qubits)
        self.assertEqual(set(qft.count_ops()), {"p", "cx", "h"})


@ddt
class TestQFTFull(QiskitTestCase):
    """Tests for QFT synthesis using all-to-all connectivity."""

    @data(2, 3, 4, 5, 6)
    def test_synthesis_default(self, num_qubits):
        """Assert that the original and synthesized QFT circuits are the same."""
        with self.assertWarns(DeprecationWarning):
            original = QFT(num_qubits)
        synthesized = synth_qft_full(num_qubits)
        self.assertEqual(Operator(original), Operator(synthesized))

    @combine(num_qubits=[5, 6, 7, 8], do_swaps=[False, True])
    def test_synthesis_do_swaps(self, num_qubits, do_swaps):
        """Assert that the original and synthesized QFT circuits are the same."""
        with self.assertWarns(DeprecationWarning):
            original = QFT(num_qubits, do_swaps=do_swaps)
        synthesized = synth_qft_full(num_qubits, do_swaps=do_swaps)
        self.assertEqual(Operator(original), Operator(synthesized))

    @combine(num_qubits=[5, 6, 7, 8], approximation_degree=[0, 1, 2, 3])
    def test_synthesis_arpproximate(self, num_qubits, approximation_degree):
        """Assert that the original and synthesized QFT circuits are the same."""
        with self.assertWarns(DeprecationWarning):
            original = QFT(num_qubits, approximation_degree=approximation_degree)
        synthesized = synth_qft_full(num_qubits, approximation_degree=approximation_degree)
        self.assertEqual(Operator(original), Operator(synthesized))

    @combine(num_qubits=[5, 6, 7, 8], inverse=[False, True])
    def test_synthesis_inverse(self, num_qubits, inverse):
        """Assert that the original and synthesized QFT circuits are the same."""
        with self.assertWarns(DeprecationWarning):
            original = QFT(num_qubits, inverse=inverse)
        synthesized = synth_qft_full(num_qubits, inverse=inverse)
        self.assertEqual(Operator(original), Operator(synthesized))

    @combine(num_qubits=[5, 6, 7, 8], insert_barriers=[False, True])
    def test_synthesis_insert_barriers(self, num_qubits, insert_barriers):
        """Assert that the original and synthesized QFT circuits are the same."""
        with self.assertWarns(DeprecationWarning):
            original = QFT(num_qubits, insert_barriers=insert_barriers)
        synthesized = synth_qft_full(num_qubits, insert_barriers=insert_barriers)
        self.assertEqual(Operator(original), Operator(synthesized))

    @data(5, 6, 7, 8)
    def test_synthesis_name(self, num_qubits):
        """Assert that the original and synthesized QFT circuits are the same."""
        with self.assertWarns(DeprecationWarning):
            original = QFT(num_qubits, name="SomeRandomName")
        synthesized = synth_qft_full(num_qubits, name="SomeRandomName")
        self.assertEqual(original.name, synthesized.name)

    @data(50, 100, 1000)
    def test_create_large_circuit(self, num_qubits):
        """Test creating large QFT circuits."""
        qft = synth_qft_full(num_qubits)
        self.assertEqual(set(qft.count_ops()), {"cp", "h", "swap"})


if __name__ == "__main__":
    unittest.main()
