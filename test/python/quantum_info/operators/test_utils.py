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

"""Tests utility functions for operator classes."""

import unittest

from ddt import data, ddt, unpack

from qiskit.quantum_info import SparsePauliOp, anti_commutator, commutator, double_commutator
from test import QiskitTestCase

I = SparsePauliOp("I")  # noqa: E741
X = SparsePauliOp("X")
Y = SparsePauliOp("Y")
Z = SparsePauliOp("Z")
zero = SparsePauliOp("I", 0)


@ddt
class TestOperatorUtils(QiskitTestCase):
    """Test utility functions for operator classes."""

    @unpack
    @data((Z, I, zero), (Z, X, 2j * Y))
    def test_commutator(self, a, b, com):
        """Test commutator function on SparsePauliOp."""
        self.assertTrue(commutator(a, b).equiv(com))

    @unpack
    @data((Z, X, zero), (Z, I, 2 * Z))
    def test_anti_commutator(self, a, b, com):
        """Test anti_commutator function on SparsePauliOp."""
        self.assertTrue(anti_commutator(a, b).equiv(com))

    @unpack
    @data(
        (X, Y, Z, True, zero),
        (X, Y, X, True, -4 * Y),
        (X, Y, Z, False, 4j * I),
        (X, Y, X, False, zero),
    )
    def test_double_commutator(self, a, b, c, com, expected):
        """Test double_commutator function on SparsePauliOp."""
        self.assertTrue(double_commutator(a, b, c, commutator=com).equiv(expected))


if __name__ == "__main__":
    unittest.main()
