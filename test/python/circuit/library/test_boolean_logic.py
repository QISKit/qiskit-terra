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

"""Test the boolean logic circuits."""

import unittest
from ddt import ddt, data, unpack
import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import (
    XOR,
    InnerProduct,
    AND,
    OR,
    AndGate,
    OrGate,
    BitwiseXorGate,
    InnerProductGate,
)
from qiskit.quantum_info import Statevector, Operator
from test import QiskitTestCase


@ddt
class TestBooleanLogicLibrary(QiskitTestCase):
    """Test library of boolean logic quantum circuits."""

    def assertBooleanFunctionIsCorrect(self, boolean_object, reference):
        """Assert that ``boolean_object`` implements the reference boolean function correctly."""
        circuit = QuantumCircuit(boolean_object.num_qubits)
        circuit.h(list(range(boolean_object.num_variable_qubits)))

        if isinstance(boolean_object, Gate):
            circuit.append(boolean_object, list(range(boolean_object.num_qubits)))
        else:
            circuit.append(boolean_object.to_instruction(), list(range(boolean_object.num_qubits)))

        # compute the statevector of the circuit
        statevector = Statevector.from_label("0" * circuit.num_qubits)
        statevector = statevector.evolve(circuit)

        # trace out ancillas
        probabilities = statevector.probabilities(
            qargs=list(range(boolean_object.num_variable_qubits + 1))
        )

        # compute the expected outcome by computing the entries of the statevector that should
        # have a 1 / sqrt(2**n) factor
        expectations = np.zeros_like(probabilities)
        for x in range(2**boolean_object.num_variable_qubits):
            bits = np.array(list(bin(x)[2:].zfill(boolean_object.num_variable_qubits)), dtype=int)
            result = reference(bits[::-1])

            entry = int(str(int(result)) + bin(x)[2:].zfill(boolean_object.num_variable_qubits), 2)
            expectations[entry] = 1 / 2**boolean_object.num_variable_qubits

        np.testing.assert_array_almost_equal(probabilities, expectations)

    def test_xor(self):
        """Test xor circuit.

        TODO add a test using assertBooleanFunctionIsCorrect
        """
        circuit = QuantumCircuit(3)
        circuit.append(BitwiseXorGate(num_qubits=3, amount=4), range(3))
        expected = QuantumCircuit(3)
        expected.x(2)
        self.assertEqual(circuit.decompose(), expected)

    def test_xor_gate(self):
        """Test XOR-gate."""
        xor_gate = BitwiseXorGate(num_qubits=3, amount=4)
        expected = QuantumCircuit(3)
        expected.x(2)
        self.assertEqual(Operator(xor_gate), Operator(expected))

    @data(
        (5, 12),
        (6, 21),
    )
    @unpack
    def test_xor_equivalence(self, num_qubits, amount):
        """Test that XOR-circuit and BitwiseXorGate yield equal operators."""
        xor_gate = BitwiseXorGate(num_qubits, amount)
        with self.assertWarns(DeprecationWarning):
            xor_circuit = XOR(num_qubits, amount)
        self.assertEqual(Operator(xor_gate), Operator(xor_circuit))

    def test_xor_eq(self):
        """Test BitwiseXorGate's equality method."""
        xor1 = BitwiseXorGate(num_qubits=5, amount=10)
        xor2 = BitwiseXorGate(num_qubits=5, amount=10)
        xor3 = BitwiseXorGate(num_qubits=5, amount=11)
        self.assertEqual(xor1, xor2)
        self.assertNotEqual(xor1, xor3)

    def test_xor_inverse(self):
        """Test correctness of the BitwiseXorGate's inverse."""
        xor_gate = BitwiseXorGate(num_qubits=5, amount=10)
        xor_gate_inverse = xor_gate.inverse()
        self.assertEqual(xor_gate, xor_gate_inverse)
        self.assertEqual(Operator(xor_gate), Operator(xor_gate_inverse).adjoint())

    def test_inner_product(self):
        """Test inner product circuit.

        TODO add a test using assertBooleanFunctionIsCorrect
        """
        with self.assertWarns(DeprecationWarning):
            circuit = InnerProduct(num_qubits=3)
        expected = QuantumCircuit(*circuit.qregs)
        expected.cz(0, 3)
        expected.cz(1, 4)
        expected.cz(2, 5)
        self.assertEqual(circuit.decompose(), expected)

    def test_inner_product_gate(self):
        """Test inner product gate."""
        inner_product = InnerProductGate(num_qubits=3)
        expected = QuantumCircuit(6)
        expected.cz(0, 3)
        expected.cz(1, 4)
        expected.cz(2, 5)
        self.assertEqual(Operator(inner_product), Operator(expected))

    @data(4, 5, 6)
    def test_inner_product_equivalence(self, num_qubits):
        """Test that XOR-circuit and BitwiseXorGate yield equal operators."""
        inner_product_gate = InnerProductGate(num_qubits)
        with self.assertWarns(DeprecationWarning):
            inner_product_circuit = InnerProduct(num_qubits)
        self.assertEqual(Operator(inner_product_gate), Operator(inner_product_circuit))

    @data(
        (2, None, "noancilla"),
        (5, None, "noancilla"),
        (2, [-1, 1], "v-chain"),
        (2, [-1, 1], "noancilla"),
        (5, [0, 0, -1, 1, -1], "noancilla"),
        (5, [-1, 0, 0, 1, 1], "v-chain"),
    )
    @unpack
    def test_or(self, num_variables, flags, mcx_mode):
        """Test the or circuit."""
        with self.assertWarns(DeprecationWarning):
            or_circuit = OR(num_variables, flags, mcx_mode=mcx_mode)
        flags = flags or [1] * num_variables

        def reference(bits):
            flagged = []
            for flag, bit in zip(flags, bits):
                if flag < 0:
                    flagged += [1 - bit]
                elif flag > 0:
                    flagged += [bit]
            return np.any(flagged)

        self.assertBooleanFunctionIsCorrect(or_circuit, reference)

    @data(
        (2, None),
        (2, [-1, 1]),
        (5, [0, 0, -1, 1, -1]),
        (5, [-1, 0, 0, 1, 1]),
    )
    @unpack
    def test_or_gate(self, num_variables, flags):
        """Test correctness of the OrGate."""
        or_gate = OrGate(num_variables, flags)
        flags = flags or [1] * num_variables

        def reference(bits):
            flagged = []
            for flag, bit in zip(flags, bits):
                if flag < 0:
                    flagged += [1 - bit]
                elif flag > 0:
                    flagged += [bit]
            return np.any(flagged)

        self.assertBooleanFunctionIsCorrect(or_gate, reference)

    @data(
        (2, None),
        (2, [-1, 1]),
        (5, [0, 0, -1, 1, -1]),
        (5, [-1, 0, 0, 1, 1]),
    )
    @unpack
    def test_or_gate_inverse(self, num_variables, flags):
        """Test correctness of the OrGate's inverse."""
        or_gate = OrGate(num_variables, flags)
        or_gate_inverse = or_gate.inverse()
        self.assertEqual(Operator(or_gate), Operator(or_gate_inverse).adjoint())

    @data(
        (2, None),
        (2, [-1, 1]),
        (5, [0, 0, -1, 1, -1]),
        (5, [-1, 0, 0, 1, 1]),
    )
    @unpack
    def test_or_equivalence(self, num_variables, flags):
        """Test that OR-circuit and OrGate yield equal operators
        (when not using ancilla qubits).
        """
        or_gate = OrGate(num_variables, flags)
        with self.assertWarns(DeprecationWarning):
            or_circuit = OR(num_variables, flags)
        self.assertEqual(Operator(or_gate), Operator(or_circuit))

    @data(
        (2, None, "noancilla"),
        (2, [-1, 1], "v-chain"),
        (5, [0, 0, -1, 1, -1], "noancilla"),
        (5, [-1, 0, 0, 1, 1], "v-chain"),
    )
    @unpack
    def test_and(self, num_variables, flags, mcx_mode):
        """Test the AND-circuit."""
        with self.assertWarns(DeprecationWarning):
            and_circuit = AND(num_variables, flags, mcx_mode=mcx_mode)
        flags = flags or [1] * num_variables

        def reference(bits):
            flagged = []
            for flag, bit in zip(flags, bits):
                if flag < 0:
                    flagged += [1 - bit]
                elif flag > 0:
                    flagged += [bit]
            return np.all(flagged)

        self.assertBooleanFunctionIsCorrect(and_circuit, reference)

    @data(
        (2, None),
        (2, [-1, 1]),
        (5, [0, 0, -1, 1, -1]),
        (5, [-1, 0, 0, 1, 1]),
    )
    @unpack
    def test_and_gate(self, num_variables, flags):
        """Test correctness of the AndGate."""
        and_gate = AndGate(num_variables, flags)
        flags = flags or [1] * num_variables

        def reference(bits):
            flagged = []
            for flag, bit in zip(flags, bits):
                if flag < 0:
                    flagged += [1 - bit]
                elif flag > 0:
                    flagged += [bit]
            return np.all(flagged)

        self.assertBooleanFunctionIsCorrect(and_gate, reference)

    @data(
        (2, None),
        (2, [-1, 1]),
        (5, [0, 0, -1, 1, -1]),
        (5, [-1, 0, 0, 1, 1]),
    )
    @unpack
    def test_and_gate_inverse(self, num_variables, flags):
        """Test correctness of the AND-gate inverse."""
        and_gate = AndGate(num_variables, flags)
        and_gate_inverse = and_gate.inverse()
        self.assertEqual(Operator(and_gate), Operator(and_gate_inverse).adjoint())

    @data(
        (2, None),
        (2, [-1, 1]),
        (5, [0, 0, -1, 1, -1]),
        (5, [-1, 0, 0, 1, 1]),
    )
    @unpack
    def test_and_equivalence(self, num_variables, flags):
        """Test that AND-circuit and AND-gate yield equal operators
        (when not using ancilla qubits).
        """
        and_gate = AndGate(num_variables, flags)
        with self.assertWarns(DeprecationWarning):
            and_circuit = AND(num_variables, flags)
        self.assertEqual(Operator(and_gate), Operator(and_circuit))


if __name__ == "__main__":
    unittest.main()
