# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test QuantumCircuit.find_bit."""

from ddt import ddt, data, unpack

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit, Clbit, AncillaRegister
from qiskit.circuit.exceptions import CircuitError
from test import QiskitTestCase


@ddt
class TestQuantumCircuitFindBit(QiskitTestCase):
    """Test cases for QuantumCircuit.find_bit."""

    @data(Qubit, Clbit)
    def test_bit_not_in_circuit(self, bit_type):
        """Verify we raise if the bit has not been attached to the circuit."""
        qc = QuantumCircuit()
        bit = bit_type()

        with self.assertRaisesRegex(CircuitError, r"Could not locate provided bit"):
            qc.find_bit(bit)

    @data(Qubit, Clbit)
    def test_registerless_bit_constructor(self, bit_type):
        """Verify we find individual bits added via QuantumCircuit constructor."""
        bits = [bit_type() for _ in range(5)]

        qc = QuantumCircuit(bits)

        for idx, bit in enumerate(bits):
            self.assertEqual(qc.find_bit(bit), (idx, []))

    @data(Qubit, Clbit)
    def test_registerless_add_bits(self, bit_type):
        """Verify we find individual bits added via QuantumCircuit.add_bits."""
        bits = [bit_type() for _ in range(5)]

        qc = QuantumCircuit()
        qc.add_bits(bits)

        for idx, bit in enumerate(bits):
            self.assertEqual(qc.find_bit(bit), (idx, []))

    def test_registerless_add_int(self):
        """Verify we find bits and implicit registers added via QuantumCircuit(int, int)."""
        qc = QuantumCircuit(5, 2)

        qubits = qc.qubits
        clbits = qc.clbits

        # N.B. After deprecation of implicit register creation via
        # QuantumCircuit(int, int) in PR#6582 and subsequent removal, this test
        # should be updated to verify no registers are found.

        qr = qc.qregs[0]
        cr = qc.cregs[0]

        for idx, bit in enumerate(qubits):
            self.assertEqual(qc.find_bit(bit), (idx, [(qr, idx)]))

        for idx, bit in enumerate(clbits):
            self.assertEqual(qc.find_bit(bit), (idx, [(cr, idx)]))

    @data(QuantumRegister, ClassicalRegister)
    def test_register_bit_reg_constructor(self, reg_type):
        """Verify we find register bits added via QuantumCicrcuit(reg)."""
        reg = reg_type(5, "reg")

        qc = QuantumCircuit(reg)

        for idx, bit in enumerate(reg):
            self.assertEqual(qc.find_bit(bit), (idx, [(reg, idx)]))

    @data(QuantumRegister, ClassicalRegister)
    def test_register_bit_add_reg(self, reg_type):
        """Verify we find register bits added QuantumCircuit.add_register."""
        reg = reg_type(5, "reg")

        qc = QuantumCircuit()
        qc.add_register(reg)

        for idx, bit in enumerate(reg):
            self.assertEqual(qc.find_bit(bit), (idx, [(reg, idx)]))

    def test_ancilla_register_add_register(self):
        """Verify AncillaRegisters are found by find_bit by their locations in qubits/qregs."""
        qreg = QuantumRegister(3, "qr")
        areg = AncillaRegister(5, "ar")

        qc = QuantumCircuit()
        qc.add_register(qreg)

        qc.add_register(areg)

        for idx, bit in enumerate(areg):
            self.assertEqual(qc.find_bit(bit), (idx + len(qreg), [(areg, idx)]))

    @data([Qubit, QuantumRegister], [Clbit, ClassicalRegister])
    @unpack
    def test_multiple_register_from_bit(self, bit_type, reg_type):
        """Verify we find individual bits in multiple registers."""

        bits = [bit_type() for _ in range(10)]

        even_reg = reg_type(bits=bits[::2])
        odd_reg = reg_type(bits=bits[1::2])

        fwd_reg = reg_type(bits=bits)
        rev_reg = reg_type(bits=bits[::-1])

        qc = QuantumCircuit()
        qc.add_bits(bits)
        qc.add_register(even_reg, odd_reg, fwd_reg, rev_reg)

        for idx, bit in enumerate(bits):
            if idx % 2:
                self.assertEqual(
                    qc.find_bit(bit),
                    (idx, [(odd_reg, idx // 2), (fwd_reg, idx), (rev_reg, 9 - idx)]),
                )
            else:
                self.assertEqual(
                    qc.find_bit(bit),
                    (idx, [(even_reg, idx // 2), (fwd_reg, idx), (rev_reg, 9 - idx)]),
                )

    @data(QuantumRegister, ClassicalRegister)
    def test_multiple_register_from_reg(self, reg_type):
        """Verify we find register bits in multiple registers."""

        reg1 = reg_type(6, "reg1")
        reg2 = reg_type(4, "reg2")

        even_reg = reg_type(bits=(reg1[:] + reg2[:])[::2])
        odd_reg = reg_type(bits=(reg1[:] + reg2[:])[1::2])

        qc = QuantumCircuit(reg1, reg2, even_reg, odd_reg)

        for idx, bit in enumerate(reg1):
            if idx % 2:
                self.assertEqual(qc.find_bit(bit), (idx, [(reg1, idx), (odd_reg, idx // 2)]))
            else:
                self.assertEqual(qc.find_bit(bit), (idx, [(reg1, idx), (even_reg, idx // 2)]))

        for idx, bit in enumerate(reg2):
            circ_idx = len(reg1) + idx
            if idx % 2:
                self.assertEqual(
                    qc.find_bit(bit), (circ_idx, [(reg2, idx), (odd_reg, circ_idx // 2)])
                )
            else:
                self.assertEqual(
                    qc.find_bit(bit), (circ_idx, [(reg2, idx), (even_reg, circ_idx // 2)])
                )


class TestQuantumCircuitBitLocations(QiskitTestCase):
    """Test cases for BitLocations in the QuantumCircuit."""

    def test_bit_indexable(self):
        """Tests whether a BitLocations instance can be indexed correctly"""
        qc = QuantumCircuit(5)
        # Retrieve the register this circuit contains
        reg = qc.qregs[0]

        # Check that all bits are localizable and that the `BitLocations` instance
        # can be indexed.
        for index, bit in enumerate(qc.qubits):
            location = qc.find_bit(bit)
            # Test index by integer
            self.assertEqual(
                location[1],
                [
                    (reg, index),
                ],
            )
            self.assertEqual(location[0], index)
            self.assertEqual(
                location[-1],
                [
                    (reg, index),
                ],
            )
            self.assertEqual(location[-2], index)

            # Test index by slice positive range
            self.assertEqual(location[0:], tuple(location))
            self.assertEqual(location[0:9], tuple(location))

            # Test index by slice negative range
            self.assertEqual(location[::-1], tuple(reversed(location)))
            self.assertEqual(location[-1::], (location[-1],))
            self.assertEqual(location[-1:1], ())

            # Test invalid indices
            with self.assertRaisesRegex(IndexError, "out of range"):
                return location[2]
            with self.assertRaisesRegex(IndexError, "out of range"):
                return location[-3]
            with self.assertRaisesRegex(IndexError, "out of range"):
                return location[-999999999]
            with self.assertRaisesRegex(IndexError, "out of range"):
                return location[999999999]
