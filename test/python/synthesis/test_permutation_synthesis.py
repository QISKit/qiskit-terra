# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test permutation synthesis functions."""


import unittest
import numpy as np
from ddt import ddt, data

from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import LinearFunction, PermutationGate
from qiskit.synthesis.permutation import (
    synth_permutation_acg,
    synth_permutation_depth_lnn_kms,
    synth_permutation_basic,
    synth_permutation_reverse_lnn_kms,
    _inverse_pattern,
    _validate_permutation,
)
from test import QiskitTestCase


@ddt
class TestPermutationSynthesis(QiskitTestCase):
    """Test the permutation synthesis functions."""

    @data(0, 4, 5, 10, 15, 20)
    def test_inverse_pattern(self, width):
        """Test _inverse_pattern function produces correct index map."""
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)
            inverse = _inverse_pattern(pattern)
            for ii, jj in enumerate(pattern):
                self.assertTrue(inverse[jj] == ii)

    @data(10, 20)
    def test_invalid_permutations(self, width):
        """Check that _validate_permutation raises exceptions when the
        input is not a permutation."""
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)

            pattern_out_of_range = np.copy(pattern)
            pattern_out_of_range[0] = -1
            with self.assertRaises(ValueError) as exc:
                _validate_permutation(pattern_out_of_range)
                self.assertIn("input contains a negative number", str(exc.exception))

            pattern_out_of_range = np.copy(pattern)
            pattern_out_of_range[0] = width
            with self.assertRaises(ValueError) as exc:
                _validate_permutation(pattern_out_of_range)
                self.assertIn(f"input has length {width} and contains {width}", str(exc.exception))

            pattern_duplicate = np.copy(pattern)
            pattern_duplicate[-1] = pattern[0]
            with self.assertRaises(ValueError) as exc:
                _validate_permutation(pattern_duplicate)
                self.assertIn(f"input contains {pattern[0]} more than once", str(exc.exception))

    @data(0, 4, 5, 10, 15, 20)
    def test_synth_permutation_basic(self, width):
        """Test synth_permutation_basic function produces the correct
        circuit."""
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)
            qc = synth_permutation_basic(pattern)

            # Check that the synthesized circuit consists of SWAP gates only.
            for instruction in qc.data:
                self.assertEqual(instruction.operation.name, "swap")

            # Construct a linear function from the synthesized circuit, and
            # check that its permutation pattern matches the original pattern.
            synthesized_pattern = LinearFunction(qc).permutation_pattern()
            self.assertTrue(np.array_equal(synthesized_pattern, pattern))

    @data(0, 4, 5, 10, 15, 20)
    def test_synth_permutation_acg(self, width):
        """Test synth_permutation_acg function produces the correct
        circuit."""
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)
            qc = synth_permutation_acg(pattern)

            # Check that the synthesized circuit consists of SWAP gates only.
            for instruction in qc.data:
                self.assertEqual(instruction.operation.name, "swap")

            # Check that the depth of the circuit (measured in terms of SWAPs) is at most 2.
            self.assertLessEqual(qc.depth(), 2)

            # Construct a linear function from the synthesized circuit, and
            # check that its permutation pattern matches the original pattern.
            synthesized_pattern = LinearFunction(qc).permutation_pattern()
            self.assertTrue(np.array_equal(synthesized_pattern, pattern))

    @data(0, 4, 5, 10, 15, 20)
    def test_synth_permutation_depth_lnn_kms(self, width):
        """Test synth_permutation_depth_lnn_kms function produces the correct
        circuit."""
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)
            qc = synth_permutation_depth_lnn_kms(pattern)

            # Check that the synthesized circuit consists of SWAP gates only,
            # and that these SWAPs adhere to the LNN connectivity.
            for instruction in qc.data:
                self.assertEqual(instruction.operation.name, "swap")
                q0 = qc.find_bit(instruction.qubits[0]).index
                q1 = qc.find_bit(instruction.qubits[1]).index
                dist = abs(q0 - q1)
                self.assertEqual(dist, 1)

            # Check that the depth of the circuit (measured in #SWAPs)
            # does not exceed the number of qubits.
            self.assertLessEqual(qc.depth(), width)

            # Construct a linear function from the synthesized circuit, and
            # check that its permutation pattern matches the original pattern.
            synthesized_pattern = LinearFunction(qc).permutation_pattern()
            self.assertTrue(np.array_equal(synthesized_pattern, pattern))

    @data(0, 1, 2, 3, 4, 5, 10, 15, 20)
    def test_synth_permutation_reverse_lnn_kms(self, num_qubits):
        """Test synth_permutation_reverse_lnn_kms function produces the correct
        circuit."""
        pattern = list(reversed(range(num_qubits)))
        qc = synth_permutation_reverse_lnn_kms(num_qubits)
        self.assertListEqual((LinearFunction(qc).permutation_pattern()).tolist(), pattern)

        # Check that the CX depth of the circuit is at 2*n+2
        self.assertTrue(qc.depth() <= 2 * num_qubits + 2)

        # Check that the synthesized circuit consists of CX gates only,
        # and that these CXs adhere to the LNN connectivity.
        for instruction in qc.data:
            self.assertEqual(instruction.operation.name, "cx")
            q0 = qc.find_bit(instruction.qubits[0]).index
            q1 = qc.find_bit(instruction.qubits[1]).index
            dist = abs(q0 - q1)
            self.assertEqual(dist, 1)

    @data(4, 5, 6, 7)
    def test_permutation_matrix(self, width):
        """Test that the unitary matrix constructed from permutation pattern
        is correct."""
        np.random.seed(1)
        for _ in range(5):
            pattern = np.random.permutation(width)
            qc = synth_permutation_depth_lnn_kms(pattern)
            expected = Operator(qc)
            constructed = Operator(PermutationGate(pattern))
            self.assertEqual(expected, constructed)


if __name__ == "__main__":
    unittest.main()
