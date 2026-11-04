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


"""Tests for PauliList utility functions."""

import unittest

from qiskit.quantum_info import PauliList, pauli_basis
from test import QiskitTestCase


class TestPauliBasis(QiskitTestCase):
    """Test pauli_basis function"""

    def test_standard_order_1q(self):
        """Test 1-qubit pauli_basis function."""
        labels = ["I", "X", "Y", "Z"]
        target = PauliList(labels)
        self.assertEqual(pauli_basis(1), target)

    def test_weight_order_1q(self):
        """Test 1-qubit pauli_basis function with weight=True."""
        labels = ["I", "X", "Y", "Z"]
        target = PauliList(labels)
        self.assertEqual(pauli_basis(1, weight=True), target)

    def test_standard_order_2q(self):
        """Test 2-qubit pauli_basis function."""
        labels = [
            "II",
            "IX",
            "IY",
            "IZ",
            "XI",
            "XX",
            "XY",
            "XZ",
            "YI",
            "YX",
            "YY",
            "YZ",
            "ZI",
            "ZX",
            "ZY",
            "ZZ",
        ]
        target = PauliList(labels)
        self.assertEqual(pauli_basis(2), target)

    def test_weight_order_2q(self):
        """Test 2-qubit pauli_basis function with weight=True."""
        labels = [
            "II",
            "IX",
            "IY",
            "IZ",
            "XI",
            "YI",
            "ZI",
            "XX",
            "XY",
            "XZ",
            "YX",
            "YY",
            "YZ",
            "ZX",
            "ZY",
            "ZZ",
        ]
        target = PauliList(labels)
        self.assertEqual(pauli_basis(2, weight=True), target)

    def test_standard_order_3q(self):
        """Test 3-qubit pauli_basis function."""
        labels = [
            "III",
            "IIX",
            "IIY",
            "IIZ",
            "IXI",
            "IXX",
            "IXY",
            "IXZ",
            "IYI",
            "IYX",
            "IYY",
            "IYZ",
            "IZI",
            "IZX",
            "IZY",
            "IZZ",
            "XII",
            "XIX",
            "XIY",
            "XIZ",
            "XXI",
            "XXX",
            "XXY",
            "XXZ",
            "XYI",
            "XYX",
            "XYY",
            "XYZ",
            "XZI",
            "XZX",
            "XZY",
            "XZZ",
            "YII",
            "YIX",
            "YIY",
            "YIZ",
            "YXI",
            "YXX",
            "YXY",
            "YXZ",
            "YYI",
            "YYX",
            "YYY",
            "YYZ",
            "YZI",
            "YZX",
            "YZY",
            "YZZ",
            "ZII",
            "ZIX",
            "ZIY",
            "ZIZ",
            "ZXI",
            "ZXX",
            "ZXY",
            "ZXZ",
            "ZYI",
            "ZYX",
            "ZYY",
            "ZYZ",
            "ZZI",
            "ZZX",
            "ZZY",
            "ZZZ",
        ]
        target = PauliList(labels)
        self.assertEqual(pauli_basis(3), target)

    def test_weight_order_3q(self):
        """Test 3-qubit pauli_basis function with weight=True."""
        labels = [
            "III",
            "IIX",
            "IIY",
            "IIZ",
            "IXI",
            "IYI",
            "IZI",
            "XII",
            "YII",
            "ZII",
            "IXX",
            "IXY",
            "IXZ",
            "IYX",
            "IYY",
            "IYZ",
            "IZX",
            "IZY",
            "IZZ",
            "XIX",
            "XIY",
            "XIZ",
            "XXI",
            "XYI",
            "XZI",
            "YIX",
            "YIY",
            "YIZ",
            "YXI",
            "YYI",
            "YZI",
            "ZIX",
            "ZIY",
            "ZIZ",
            "ZXI",
            "ZYI",
            "ZZI",
            "XXX",
            "XXY",
            "XXZ",
            "XYX",
            "XYY",
            "XYZ",
            "XZX",
            "XZY",
            "XZZ",
            "YXX",
            "YXY",
            "YXZ",
            "YYX",
            "YYY",
            "YYZ",
            "YZX",
            "YZY",
            "YZZ",
            "ZXX",
            "ZXY",
            "ZXZ",
            "ZYX",
            "ZYY",
            "ZYZ",
            "ZZX",
            "ZZY",
            "ZZZ",
        ]
        target = PauliList(labels)
        self.assertEqual(pauli_basis(3, weight=True), target)


if __name__ == "__main__":
    unittest.main()
