# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test library of quantum circuits."""
import copy

from qiskit.circuit import Qubit, QuantumRegister
from test import QiskitTestCase


class TestQubitClass(QiskitTestCase):
    """Test library of boolean logic quantum circuits."""

    def test_bit_eq_invalid_type_comparison(self):
        orig_reg = QuantumRegister(3)
        test_bit = Qubit(orig_reg, 0)
        self.assertNotEqual(test_bit, 3.14)

    def test_old_style_bit_equality(self):
        test_reg = QuantumRegister(size=3, name="foo")

        self.assertEqual(Qubit(test_reg, 0), Qubit(test_reg, 0))
        self.assertNotEqual(Qubit(test_reg, 0), Qubit(test_reg, 2))

        reg_copy = copy.copy(test_reg)

        self.assertEqual(Qubit(test_reg, 0), Qubit(reg_copy, 0))
        self.assertNotEqual(Qubit(test_reg, 0), Qubit(reg_copy, 1))

        reg_larger = QuantumRegister(size=4, name="foo")

        self.assertNotEqual(Qubit(test_reg, 0), Qubit(reg_larger, 0))

        reg_renamed = QuantumRegister(size=3, name="bar")

        self.assertNotEqual(Qubit(test_reg, 0), Qubit(reg_renamed, 0))

        reg_difftype = QuantumRegister(size=3, name="bar")

        self.assertNotEqual(Qubit(test_reg, 0), Qubit(reg_difftype, 0))

    def test_old_style_bit_deepcopy(self):
        """Verify deep-copies of bits are equal."""
        test_reg = QuantumRegister(size=3, name="foo")

        bit1 = Qubit(test_reg, 0)
        bit2 = copy.deepcopy(bit1)

        # Bits are fully immutable.
        self.assertIs(bit1, bit2)

    def test_old_style_bit_copy(self):
        """Verify copies of bits are the same instance."""
        bit1 = Qubit()
        bit2 = copy.copy(bit1)

        self.assertIs(bit1, bit2)


class TestNewStyleQubit(QiskitTestCase):
    """Test behavior of new-style bits."""

    def test_bits_do_not_require_registers(self):
        """Verify we can create a bit outside the context of a register."""
        self.assertIsInstance(Qubit(), Qubit)

    def test_new_style_bit_deepcopy(self):
        """Verify deep-copies of bits are the same instance."""
        bit1 = Qubit()
        bit2 = copy.deepcopy(bit1)

        self.assertIs(bit1, bit2)

    def test_new_style_bit_copy(self):
        """Verify copies of bits are the same instance."""
        bit1 = Qubit()
        bit2 = copy.copy(bit1)

        self.assertIs(bit1, bit2)

    def test_new_style_bit_equality(self):
        """Verify bits instances are equal only to themselves."""
        bit1 = Qubit()
        bit2 = Qubit()

        self.assertEqual(bit1, bit1)
        self.assertNotEqual(bit1, bit2)
        self.assertNotEqual(bit1, 3.14)
