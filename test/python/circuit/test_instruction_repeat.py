# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test Qiskit's repeat instruction operation."""

import unittest

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import SGate, CXGate, UnitaryGate
from qiskit.circuit import Instruction, Measure, Gate
from qiskit.circuit.exceptions import CircuitError
from test import QiskitTestCase


class TestRepeatInt1Q(QiskitTestCase):
    """Test gate_q1.repeat() with integer"""

    def test_standard_1Q_two(self):
        """Test standard gate.repeat(2) method."""
        qr = QuantumRegister(1, "qr")
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SGate(), [qr[0]])
        expected_circ.append(SGate(), [qr[0]])
        expected = expected_circ.to_instruction()

        result = SGate().repeat(2)

        self.assertEqual(result.name, "s*2")
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Gate)

    def test_standard_1Q_one(self):
        """Test standard gate.repeat(1) method."""
        qr = QuantumRegister(1, "qr")
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SGate(), [qr[0]])
        expected = expected_circ.to_instruction()

        result = SGate().repeat(1)

        self.assertEqual(result.name, "s*1")
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Gate)


class TestRepeatInt2Q(QiskitTestCase):
    """Test gate_q2.repeat() with integer"""

    def test_standard_2Q_two(self):
        """Test standard 2Q gate.repeat(2) method."""
        qr = QuantumRegister(2, "qr")
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(CXGate(), [qr[0], qr[1]])
        expected_circ.append(CXGate(), [qr[0], qr[1]])
        expected = expected_circ.to_instruction()

        result = CXGate().repeat(2)

        self.assertEqual(result.name, "cx*2")
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Gate)

    def test_standard_2Q_one(self):
        """Test standard 2Q gate.repeat(1) method."""
        qr = QuantumRegister(2, "qr")
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(CXGate(), [qr[0], qr[1]])
        expected = expected_circ.to_instruction()

        result = CXGate().repeat(1)

        self.assertEqual(result.name, "cx*1")
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Gate)


class TestRepeatIntMeasure(QiskitTestCase):
    """Test Measure.repeat() with integer"""

    def test_measure_two(self):
        """Test Measure.repeat(2) method."""
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        expected_circ = QuantumCircuit(qr, cr)
        expected_circ.append(Measure(), [qr[0]], [cr[0]])
        expected_circ.append(Measure(), [qr[0]], [cr[0]])
        expected = expected_circ.to_instruction()

        result = Measure().repeat(2)

        self.assertEqual(result.name, "measure*2")
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)
        self.assertNotIsInstance(result, Gate)

    def test_measure_one(self):
        """Test Measure.repeat(1) method."""
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        expected_circ = QuantumCircuit(qr, cr)
        expected_circ.append(Measure(), [qr[0]], [cr[0]])
        expected = expected_circ.to_instruction()

        result = Measure().repeat(1)

        self.assertEqual(result.name, "measure*1")
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)
        self.assertNotIsInstance(result, Gate)


class TestRepeatErrors(QiskitTestCase):
    """Test when Gate.repeat() should raise."""

    def test_unitary_no_int(self):
        """Test UnitaryGate.repeat(2/3) method. Raises, since n is not int."""
        with self.assertRaises(CircuitError) as context:
            _ = UnitaryGate([[0, 1j], [-1j, 0]]).repeat(2 / 3)
        self.assertIn("strictly positive integer", str(context.exception))

    def test_standard_no_int(self):
        """Test standard Gate.repeat(2/3) method. Raises, since n is not int."""
        with self.assertRaises(CircuitError) as context:
            _ = SGate().repeat(2 / 3)
        self.assertIn("strictly positive integer", str(context.exception))

    def test_measure_zero(self):
        """Test Measure.repeat(0) method. Raises, since n<1"""
        with self.assertRaises(CircuitError) as context:
            _ = Measure().repeat(0)
        self.assertIn("strictly positive integer", str(context.exception))

    def test_standard_1Q_zero(self):
        """Test standard 2Q gate.repeat(0) method. Raises, since n<1."""
        with self.assertRaises(CircuitError) as context:
            _ = SGate().repeat(0)
        self.assertIn("strictly positive integer", str(context.exception))

    def test_standard_1Q_minus_one(self):
        """Test standard 2Q gate.repeat(-1) method. Raises, since n<1."""
        with self.assertRaises(CircuitError) as context:
            _ = SGate().repeat(-1)
        self.assertIn("strictly positive integer", str(context.exception))

    def test_standard_2Q_minus_one(self):
        """Test standard 2Q gate.repeat(-1) method. Raises, since n<1."""
        with self.assertRaises(CircuitError) as context:
            _ = CXGate().repeat(-1)
        self.assertIn("strictly positive integer", str(context.exception))

    def test_measure_minus_one(self):
        """Test Measure.repeat(-1) method. Raises, since n<1"""
        with self.assertRaises(CircuitError) as context:
            _ = Measure().repeat(-1)
        self.assertIn("strictly positive integer", str(context.exception))

    def test_standard_2Q_zero(self):
        """Test standard 2Q gate.repeat(0) method. Raises, since n<1."""
        with self.assertRaises(CircuitError) as context:
            _ = CXGate().repeat(0)
        self.assertIn("strictly positive integer", str(context.exception))


if __name__ == "__main__":
    unittest.main()
