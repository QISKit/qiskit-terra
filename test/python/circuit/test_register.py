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

from ddt import data, ddt

from qiskit.circuit import Qubit, Clbit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import AncillaRegister
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.exceptions import CircuitError
from test import QiskitTestCase


@ddt
class TestRegisterClass(QiskitTestCase):
    """Tests for Register class."""

    @data(QuantumRegister, ClassicalRegister, AncillaRegister)
    def test_raise_on_init_with_invalid_size(self, reg_type):
        with self.assertRaises(TypeError):
            _ = reg_type(1j, "foo")

    @data(QuantumRegister, ClassicalRegister, AncillaRegister)
    def test_raise_if_init_passed_both_size_and_bits(self, reg_type):
        bits = [reg_type.bit_type()]
        with self.assertRaisesRegex(CircuitError, "Exactly one of the size or bits"):
            _ = reg_type(1, "foo", bits)

    @data(QuantumRegister, ClassicalRegister, AncillaRegister)
    def test_raise_on_init_with_duplicated_bits(self, reg_type):
        bits = [reg_type.bit_type()] * 2
        with self.assertRaisesRegex(CircuitError, "bits must not be duplicated"):
            _ = reg_type(bits=bits)

    @data(QuantumRegister, ClassicalRegister, AncillaRegister)
    def test_init_raise_if_bits_of_incorrect_type(self, reg_type):
        bits = [Qubit(), Clbit()]
        with self.assertRaises(TypeError):
            _ = reg_type(bits=bits)

    @data(QuantumRegister, ClassicalRegister, AncillaRegister)
    def test_init_with_zero_size(self, reg_type):
        register = reg_type(0)
        self.assertEqual(register.size, 0)

    @data(QuantumRegister, ClassicalRegister, AncillaRegister)
    def test_init_raise_if_negative_size(self, reg_type):
        with self.assertRaisesRegex(CircuitError, "Register size must be non-negative"):
            _ = reg_type(-1)

    @data(QuantumRegister, ClassicalRegister, AncillaRegister)
    def test_implicit_bit_construction_from_size(self, reg_type):
        reg = reg_type(2)
        self.assertEqual(len(reg), 2)
        self.assertEqual(reg.size, 2)
        self.assertTrue(all(isinstance(bit, reg.bit_type) for bit in reg))

    @data(QuantumRegister, ClassicalRegister, AncillaRegister)
    def test_implicit_size_calculation_from_bits(self, reg_type):
        bits = [reg_type.bit_type() for _ in range(3)]
        reg = reg_type(bits=bits)
        self.assertEqual(reg.size, 3)

    @data(QuantumRegister, ClassicalRegister, AncillaRegister)
    def test_oldstyle_register_eq(self, reg_type):
        test_reg = reg_type(3, "foo")
        self.assertEqual(test_reg, test_reg)

        reg_copy = reg_type(3, "foo")
        self.assertEqual(reg_copy, test_reg)

        reg_larger = reg_type(4, "foo")
        self.assertNotEqual(reg_larger, test_reg)

        reg_renamed = reg_type(3, "bar")
        self.assertNotEqual(reg_renamed, test_reg)

        difftype = ({QuantumRegister, ClassicalRegister, AncillaRegister} - {reg_type}).pop()
        reg_difftype = difftype(3, "foo")
        self.assertNotEqual(reg_difftype, test_reg)

    @data(QuantumRegister, ClassicalRegister, AncillaRegister)
    def test_newstyle_register_eq(self, reg_type):
        test_bits = [reg_type.bit_type() for _ in range(3)]
        test_reg = reg_type(name="foo", bits=test_bits)
        self.assertEqual(test_reg, test_reg)

        reg_samebits = reg_type(name="foo", bits=test_bits)
        self.assertEqual(reg_samebits, test_reg)

        test_diffbits = [reg_type.bit_type() for _ in range(3)]
        reg_diffbits = reg_type(name="foo", bits=test_diffbits)
        self.assertNotEqual(reg_diffbits, test_reg)

        reg_oldstyle = reg_type(3, "foo")
        self.assertNotEqual(reg_oldstyle, test_reg)

        test_largerbits = [reg_type.bit_type() for _ in range(4)]
        reg_larger = reg_type(name="foo", bits=test_largerbits)
        self.assertNotEqual(reg_larger, test_reg)

        reg_renamed = reg_type(name="bar", bits=test_bits)
        self.assertNotEqual(reg_renamed, test_reg)

        difftype = ({QuantumRegister, ClassicalRegister, AncillaRegister} - {reg_type}).pop()
        bits_difftype = [difftype.bit_type() for _ in range(3)]
        reg_difftype = difftype(name="foo", bits=bits_difftype)
        self.assertNotEqual(reg_difftype, test_reg)
