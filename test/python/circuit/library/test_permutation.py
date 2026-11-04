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

"""Test permutation quantum circuits, permutation gates, and quantum circuits that
contain permutation gates."""

import io

import unittest
import numpy as np

from qiskit import QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import Permutation, PermutationGate
from qiskit.quantum_info import Operator
from qiskit.qpy import dump, load
from qiskit.qasm2 import dumps
from test import QiskitTestCase


class TestPermutationLibrary(QiskitTestCase):
    """Test library of permutation logic quantum circuits."""

    def test_permutation(self):
        """Test permutation circuit."""
        with self.assertWarns(DeprecationWarning):
            circuit = Permutation(num_qubits=4, pattern=[1, 0, 3, 2])
        expected = QuantumCircuit(4)
        expected.swap(0, 1)
        expected.swap(2, 3)
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))

    def test_permutation_bad(self):
        """Test that [0,..,n-1] permutation is required (no -1 for last element)."""
        with self.assertRaises(CircuitError):
            with self.assertWarns(DeprecationWarning):
                _ = Permutation(4, [1, 0, -1, 2])


class TestPermutationGate(QiskitTestCase):
    """Tests for the PermutationGate class."""

    def test_permutation(self):
        """Test that Operator can be constructed."""
        perm = PermutationGate(pattern=[1, 0, 3, 2])
        expected = QuantumCircuit(4)
        expected.swap(0, 1)
        expected.swap(2, 3)
        expected = Operator(expected)
        simulated = Operator(perm)
        self.assertTrue(expected.equiv(simulated))

    def test_permutation_bad(self):
        """Test that [0,..,n-1] permutation is required (no -1 for last element)."""
        self.assertRaises(CircuitError, PermutationGate, [1, 0, -1, 2])

    def test_permutation_array(self):
        """Test correctness of the ``__array__`` method."""
        perm = PermutationGate([1, 2, 0])
        # The permutation pattern means q1->q0, q2->q1, q0->q2, or equivalently
        # q0'=q1, q1'=q2, q2'=q0, where the primed values are the values after the
        # permutation. The following matrix is the expected unitary matrix for this.
        # As an example, the second column represents the result of applying
        # the permutation to |001>, i.e. to q2=0, q1=0, q0=1. We should get
        # q2'=q0=1, q1'=q2=0, q0'=q1=0, that is the state |100>. This corresponds
        # to the "1" in the 5 row.
        expected_op = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.array_equal(perm.__array__(dtype=int), expected_op))

    def test_pattern(self):
        """Test the ``pattern`` method."""
        pattern = [1, 3, 5, 0, 4, 2]
        perm = PermutationGate(pattern)
        self.assertTrue(np.array_equal(perm.pattern, pattern))

    def test_inverse(self):
        """Test correctness of the ``inverse`` method."""
        perm = PermutationGate([1, 3, 5, 0, 4, 2])

        # We have the permutation 1->0, 3->1, 5->2, 0->3, 4->4, 2->5.
        # The inverse permutations is 0->1, 1->3, 2->5, 3->0, 4->4, 5->2, or
        # after reordering 3->0, 0->1, 5->2, 1->3, 4->4, 2->5.
        inverse_perm = perm.inverse()
        expected_inverse_perm = PermutationGate([3, 0, 5, 1, 4, 2])
        self.assertTrue(np.array_equal(inverse_perm.pattern, expected_inverse_perm.pattern))

    def test_repeat(self):
        """Test the ``repeat`` method."""
        pattern = [2, 4, 1, 3, 0]
        perm = PermutationGate(pattern)
        self.assertTrue(np.allclose(Operator(perm.repeat(2)), Operator(perm) @ Operator(perm)))


class TestPermutationGatesOnCircuit(QiskitTestCase):
    """Tests for quantum circuits containing permutations."""

    def test_append_to_circuit(self):
        """Test method for adding Permutations to quantum circuit."""
        qc = QuantumCircuit(5)
        qc.append(PermutationGate([1, 2, 0]), [0, 1, 2])
        qc.append(PermutationGate([2, 3, 0, 1]), [1, 2, 3, 4])
        self.assertIsInstance(qc.data[0].operation, PermutationGate)
        self.assertIsInstance(qc.data[1].operation, PermutationGate)

    def test_inverse(self):
        """Test inverse method for circuits with permutations."""
        qc = QuantumCircuit(5)
        qc.append(PermutationGate([1, 2, 3, 0]), [0, 4, 2, 1])
        qci = qc.inverse()
        qci_pattern = qci.data[0].operation.pattern
        expected_pattern = [3, 0, 1, 2]

        # The inverse permutations should be defined over the same qubits but with the
        # inverse permutation pattern.
        self.assertTrue(np.array_equal(qci_pattern, expected_pattern))
        self.assertTrue(np.array_equal(qc.data[0].qubits, qci.data[0].qubits))

    def test_reverse_ops(self):
        """Test reverse_ops method for circuits with permutations."""
        qc = QuantumCircuit(5)
        qc.append(PermutationGate([1, 2, 3, 0]), [0, 4, 2, 1])
        qcr = qc.reverse_ops()

        # The reversed circuit should have the permutation gate with the same pattern and over the
        # same qubits.
        self.assertTrue(np.array_equal(qc.data[0].operation.pattern, qcr.data[0].operation.pattern))
        self.assertTrue(np.array_equal(qc.data[0].qubits, qcr.data[0].qubits))

    def test_qasm(self):
        """Test qasm for circuits with permutations."""
        qr = QuantumRegister(5, "q0")
        circuit = QuantumCircuit(qr)
        pattern = [2, 4, 3, 0, 1]
        permutation = PermutationGate(pattern)
        circuit.append(permutation, [0, 1, 2, 3, 4])
        circuit.h(qr[0])

        expected_qasm = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "gate permutation__2_4_3_0_1_ q0,q1,q2,q3,q4 { swap q2,q3; swap q1,q4; swap q0,q3; }\n"
            "qreg q0[5];\n"
            "permutation__2_4_3_0_1_ q0[0],q0[1],q0[2],q0[3],q0[4];\n"
            "h q0[0];"
        )
        self.assertEqual(expected_qasm, dumps(circuit))

    def test_qpy(self):
        """Test qpy for circuits with permutations."""
        circuit = QuantumCircuit(6, 1)
        circuit.cx(0, 1)
        circuit.append(PermutationGate([1, 2, 0]), [2, 4, 5])
        circuit.h(4)

        qpy_file = io.BytesIO()
        dump(circuit, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]

        self.assertEqual(circuit, new_circuit)


if __name__ == "__main__":
    unittest.main()
