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


"""HamiltonianGate tests"""

import numpy as np
from numpy.testing import assert_allclose

from qiskit.circuit.library import HamiltonianGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.converters import circuit_to_dag, dag_to_circuit
from test import QiskitTestCase


class TestHamiltonianGate(QiskitTestCase):
    """Tests for the HamiltonianGate class."""

    def test_set_matrix(self):
        """Test instantiation"""
        hamiltonian = HamiltonianGate([[0, 1], [1, 0]], 1)
        self.assertEqual(hamiltonian.num_qubits, 1)

    def test_set_matrix_raises(self):
        """test non-unitary"""
        with self.assertRaises(ValueError):
            HamiltonianGate([[1, 0], [1, 1]], 1)

    def test_complex_time_raises(self):
        """test non-unitary"""
        with self.assertRaises(ValueError):
            HamiltonianGate([[1, 0], [1, 1]], 1j)

    def test_conjugate(self):
        """test conjugate"""
        ham = HamiltonianGate([[0, 1j], [-1j, 2]], np.pi / 4)
        np.testing.assert_array_almost_equal(ham.conjugate().to_matrix(), np.conj(ham.to_matrix()))

    def test_transpose(self):
        """test transpose"""
        ham = HamiltonianGate([[15, 1j], [-1j, -2]], np.pi / 7)
        np.testing.assert_array_almost_equal(
            ham.transpose().to_matrix(), np.transpose(ham.to_matrix())
        )

    def test_adjoint(self):
        """test adjoint operation"""
        ham = HamiltonianGate([[3, 4j], [-4j, -0.2]], np.pi * 0.143)
        np.testing.assert_array_almost_equal(
            ham.adjoint().to_matrix(), np.transpose(np.conj(ham.to_matrix()))
        )

    def test_repeat(self):
        """test repeat operation"""
        ham = HamiltonianGate(np.array([[1, 0.5 + 4j], [0.5 - 4j, -0.2]]), np.pi * 0.143)
        operator = Operator(ham)
        self.assertTrue(np.allclose(Operator(ham.repeat(2)), operator @ operator))


class TestHamiltonianCircuit(QiskitTestCase):
    """Hamiltonian gate circuit tests."""

    def test_1q_hamiltonian(self):
        """test 1 qubit hamiltonian"""
        qr = QuantumRegister(1, "q0")
        cr = ClassicalRegister(1, "c0")
        qc = QuantumCircuit(qr, cr)
        matrix = np.zeros((2, 2))
        qc.x(qr[0])
        theta = Parameter("theta")
        qc.append(HamiltonianGate(matrix, theta), [qr[0]])
        qc = qc.assign_parameters({theta: 1})

        # test of text drawer
        self.log.info(qc)
        dag = circuit_to_dag(qc)
        dag_nodes = dag.named_nodes("hamiltonian")
        self.assertTrue(len(dag_nodes) == 1)
        dnode = dag_nodes[0]
        self.assertIsInstance(dnode.op, HamiltonianGate)
        self.assertEqual(dnode.qargs, tuple(qc.qubits))
        assert_allclose(dnode.op.to_matrix(), np.eye(2))

    def test_2q_hamiltonian(self):
        """test 2 qubit hamiltonian"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        matrix = Operator.from_label("XY")
        qc.x(qr[0])
        theta = Parameter("theta")
        uni2q = HamiltonianGate(matrix, theta)
        qc.append(uni2q, [qr[0], qr[1]])
        qc2 = qc.assign_parameters({theta: -np.pi / 2})
        dag = circuit_to_dag(qc2)
        nodes = dag.two_qubit_ops()
        self.assertEqual(len(nodes), 1)
        dnode = nodes[0]
        self.assertIsInstance(dnode.op, HamiltonianGate)
        self.assertEqual(dnode.qargs, (qr[0], qr[1]))
        # Equality based on Pauli exponential identity
        np.testing.assert_array_almost_equal(dnode.op.to_matrix(), 1j * matrix.data)
        qc3 = dag_to_circuit(dag)
        self.assertEqual(qc2, qc3)

    def test_3q_hamiltonian(self):
        """test 3 qubit hamiltonian on non-consecutive bits"""
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr)
        qc.x(qr[0])
        matrix = Operator.from_label("XZY")
        theta = Parameter("theta")
        uni3q = HamiltonianGate(matrix, theta)
        qc.append(uni3q, [qr[0], qr[1], qr[3]])
        qc.cx(qr[3], qr[2])
        # test of text drawer
        self.log.info(qc)
        qc = qc.assign_parameters({theta: -np.pi / 2})
        dag = circuit_to_dag(qc)
        nodes = dag.multi_qubit_ops()
        self.assertEqual(len(nodes), 1)
        dnode = nodes[0]
        self.assertIsInstance(dnode.op, HamiltonianGate)
        self.assertEqual(dnode.qargs, (qr[0], qr[1], qr[3]))
        np.testing.assert_almost_equal(dnode.op.to_matrix(), 1j * matrix.data)

    def test_decomposes_into_correct_unitary(self):
        """test 2 qubit hamiltonian"""
        qc = QuantumCircuit(2)
        matrix = Operator.from_label("XY")
        theta = Parameter("theta")
        uni2q = HamiltonianGate(matrix, theta)
        qc.append(uni2q, [0, 1])
        qc = qc.assign_parameters({theta: -np.pi / 2}).decompose()
        decomposed_ham = qc.data[0].operation
        self.assertEqual(Operator(decomposed_ham), 1j * Operator.from_label("XY"))
