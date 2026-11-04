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

"""Randomized tests of Clifford operator class."""
import unittest
from hypothesis import given, strategies, settings

from qiskit.quantum_info import random_clifford
from qiskit.quantum_info import Clifford


class TestClifford(unittest.TestCase):
    """Test random_clifford"""

    def assertValidClifford(self, value, num_qubits):
        """Assertion from test/python/quantum_info/operators/test_random.py:
        TestRandomClifford:test_valid"""
        self.assertIsInstance(value, Clifford)
        self.assertEqual(value.num_qubits, num_qubits)

    @given(
        strategies.integers(min_value=0, max_value=2**32 - 1),
        strategies.integers(min_value=1, max_value=211),
    )
    @settings(deadline=None)
    def test_random_clifford_valid(self, seed, num_qubits):
        """Test random_clifford."""
        value = random_clifford(num_qubits, seed=seed)
        self.assertValidClifford(value, num_qubits=num_qubits)


if __name__ == "__main__":
    unittest.main()
