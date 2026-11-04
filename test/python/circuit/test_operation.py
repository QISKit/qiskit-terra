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

"""Test Qiskit's Operation class."""

import unittest

import numpy as np

from qiskit.circuit import QuantumCircuit, Barrier, Measure, Reset, Gate, Operation
from qiskit.circuit.library import XGate, CXGate, Initialize, Isometry
from qiskit.quantum_info.operators import Clifford, CNOTDihedral, Pauli
from test import QiskitTestCase


class TestOperationClass(QiskitTestCase):
    """Testing qiskit.circuit.Operation"""

    def test_measure_as_operation(self):
        """Test that we can instantiate an object of class
        :class:`~qiskit.circuit.Measure` and that
        it has the expected name, num_qubits and num_clbits.
        """
        op = Measure()
        self.assertTrue(op.name == "measure")
        self.assertTrue(op.num_qubits == 1)
        self.assertTrue(op.num_clbits == 1)
        self.assertIsInstance(op, Operation)

    def test_reset_as_operation(self):
        """Test that we can instantiate an object of class
        :class:`~qiskit.circuit.Reset` and that
        it has the expected name, num_qubits and num_clbits.
        """
        op = Reset()
        self.assertTrue(op.name == "reset")
        self.assertTrue(op.num_qubits == 1)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_barrier_as_operation(self):
        """Test that we can instantiate an object of class
        :class:`~qiskit.circuit.Barrier` and that
        it has the expected name, num_qubits and num_clbits.
        """
        num_qubits = 4
        op = Barrier(num_qubits)
        self.assertTrue(op.name == "barrier")
        self.assertTrue(op.num_qubits == num_qubits)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_clifford_as_operation(self):
        """Test that we can instantiate an object of class
        :class:`~qiskit.quantum_info.operators.Clifford` and that
        it has the expected name, num_qubits and num_clbits.
        """
        num_qubits = 4
        qc = QuantumCircuit(4, 0)
        qc.h(2)
        qc.cx(0, 1)
        op = Clifford(qc)
        self.assertTrue(op.name == "clifford")
        self.assertTrue(op.num_qubits == num_qubits)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_cnotdihedral_as_operation(self):
        """Test that we can instantiate an object of class
        :class:`~qiskit.quantum_info.operators.CNOTDihedral` and that
        it has the expected name, num_qubits and num_clbits.
        """
        num_qubits = 4
        qc = QuantumCircuit(4)
        qc.t(0)
        qc.x(0)
        qc.t(0)
        op = CNOTDihedral(qc)
        self.assertTrue(op.name == "cnotdihedral")
        self.assertTrue(op.num_qubits == num_qubits)
        self.assertTrue(op.num_clbits == 0)

    def test_pauli_as_operation(self):
        """Test that we can instantiate an object of class
        :class:`~qiskit.quantum_info.operators.Pauli` and that
        it has the expected name, num_qubits and num_clbits.
        """
        num_qubits = 4
        op = Pauli("I" * num_qubits)
        self.assertTrue(op.name == "pauli")
        self.assertTrue(op.num_qubits == num_qubits)
        self.assertTrue(op.num_clbits == 0)

    def test_isometry_as_operation(self):
        """Test that we can instantiate an object of class
        :class:`~qiskit.circuit.library.Isometry` and that
        it has the expected name, num_qubits and num_clbits.
        """
        op = Isometry(np.eye(4, 4), 3, 2)
        self.assertTrue(op.name == "isometry")
        self.assertTrue(op.num_qubits == 7)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_initialize_as_operation(self):
        """Test that we can instantiate an object of class
        :class:`~qiskit.circuit.library.Initialize` and that
        it has the expected name, num_qubits and num_clbits.
        """
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        op = Initialize(desired_vector)
        self.assertTrue(op.name == "initialize")
        self.assertTrue(op.num_qubits == 2)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_gate_as_operation(self):
        """Test that we can instantiate an object of class
        :class:`~qiskit.circuit.Gate` and that
        it has the expected name, num_qubits and num_clbits.
        """
        name = "test_gate_name"
        num_qubits = 3
        op = Gate(name, num_qubits, [])
        self.assertTrue(op.name == name)
        self.assertTrue(op.num_qubits == num_qubits)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_xgate_as_operation(self):
        """Test that we can instantiate an object of class
        :class:`~qiskit.circuit.library.XGate` and that
        it has the expected name, num_qubits and num_clbits.
        """
        op = XGate()
        self.assertTrue(op.name == "x")
        self.assertTrue(op.num_qubits == 1)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_cxgate_as_operation(self):
        """Test that we can instantiate an object of class
        :class:`~qiskit.circuit.library.CXGate` and that
        it has the expected name, num_qubits and num_clbits.
        """
        op = CXGate()
        self.assertTrue(op.name == "cx")
        self.assertTrue(op.num_qubits == 2)
        self.assertTrue(op.num_clbits == 0)
        self.assertIsInstance(op, Operation)

    def test_can_append_to_quantum_circuit(self):
        """Test that we can add various objects with Operation interface to a Quantum Circuit."""
        qc = QuantumCircuit(6, 1)
        qc.append(XGate(), [2])
        qc.append(Barrier(3), [1, 2, 4])
        qc.append(CXGate(), [0, 1])
        qc.append(Measure(), [1], [0])
        qc.append(Reset(), [0])
        qc.cx(3, 4)
        qc.append(Gate("some_gate", 3, []), [1, 2, 3])
        qc.append(Initialize([0.5, 0.5, 0.5, 0.5]), [4, 5])
        qc.append(Isometry(np.eye(4, 4), 0, 0), [3, 4])
        qc.append(Pauli("II"), [0, 1])

        # Appending Clifford
        circ1 = QuantumCircuit(2)
        circ1.h(1)
        circ1.cx(0, 1)
        qc.append(Clifford(circ1), [0, 1])

        # Appending CNOTDihedral
        circ2 = QuantumCircuit(2)
        circ2.t(0)
        circ2.x(0)
        circ2.t(1)
        qc.append(CNOTDihedral(circ2), [2, 3])

        # If we got to here, we have successfully appended everything to qc
        self.assertIsInstance(qc, QuantumCircuit)


if __name__ == "__main__":
    unittest.main()
