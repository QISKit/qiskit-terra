# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Non-string identifiers for circuit and record identifiers test"""

import unittest

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.exceptions import CircuitError
from test import QiskitTestCase


class TestAnonymousIds(QiskitTestCase):
    """Test the anonymous use of registers."""

    def test_create_anonymous_classical_register(self):
        """ClassicalRegister with no name."""
        cr = ClassicalRegister(size=3)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_anonymous_quantum_register(self):
        """QuantumRegister with no name."""
        qr = QuantumRegister(size=3)
        self.assertIsInstance(qr, QuantumRegister)

    def test_create_anonymous_classical_registers(self):
        """Several ClassicalRegister with no name."""
        cr1 = ClassicalRegister(size=3)
        cr2 = ClassicalRegister(size=3)
        self.assertNotEqual(cr1.name, cr2.name)

    def test_create_anonymous_quantum_registers(self):
        """Several QuantumRegister with no name."""
        qr1 = QuantumRegister(size=3)
        qr2 = QuantumRegister(size=3)
        self.assertNotEqual(qr1.name, qr2.name)

    def test_create_anonymous_mixed_registers(self):
        """Several Registers with no name."""
        cr0 = ClassicalRegister(size=3)
        qr0 = QuantumRegister(size=3)
        # Get the current index count of the registers
        cr_index = int(cr0.name[1:])
        qr_index = int(qr0.name[1:])

        cr1 = ClassicalRegister(size=3)
        _ = QuantumRegister(size=3)
        qr2 = QuantumRegister(size=3)

        # Check that the counters for each kind are incremented separately.
        cr_current = int(cr1.name[1:])
        qr_current = int(qr2.name[1:])
        self.assertEqual(cr_current, cr_index + 1)
        self.assertEqual(qr_current, qr_index + 2)

    def test_create_circuit_noname(self):
        """Create_circuit with no name."""
        qr = QuantumRegister(size=3)
        cr = ClassicalRegister(size=3)
        qc = QuantumCircuit(qr, cr)
        self.assertIsInstance(qc, QuantumCircuit)


class TestInvalidIds(QiskitTestCase):
    """Circuits and records with invalid IDs"""

    def test_invalid_type_circuit_name(self):
        """QuantumCircuit() with invalid type name."""
        qr = QuantumRegister(size=3)
        cr = ClassicalRegister(size=3)
        self.assertRaises(CircuitError, QuantumCircuit, qr, cr, name=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
