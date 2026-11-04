# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Tests for Operator matrix linear operator class."""

import unittest
import logging
import copy

from test import combine
import numpy as np
import ddt
from numpy.testing import assert_allclose
import scipy.stats
import scipy.linalg

from qiskit import QiskitError
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import library
from qiskit.circuit.library import HGate, CHGate, CXGate, QFTGate
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.layout import Layout, TranspileLayout
from qiskit.quantum_info.operators import Operator, ScalarOp
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.operator_utils import _equal_with_ancillas
from qiskit.compiler.transpiler import transpile
from qiskit.circuit import Qubit
from qiskit.circuit.library import PermutationGate
from qiskit.utils import optionals

from qiskit._accelerate import unitary_sim

from test import QiskitTestCase

logger = logging.getLogger(__name__)


class OperatorTestCase(QiskitTestCase):
    """Test utils for Operator"""

    # Pauli-matrix unitaries
    UI = np.eye(2)
    UX = np.array([[0, 1], [1, 0]])
    UY = np.array([[0, -1j], [1j, 0]])
    UZ = np.diag([1, -1])
    UH = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    @classmethod
    def rand_rho(cls, n):
        """Return random density matrix"""
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        logger.debug("rand_rho default_rng seeded with seed=%s", seed)
        rng = np.random.default_rng(seed)

        psi = rng.random(n) + 1j * rng.random(n)
        rho = np.outer(psi, psi.conj())
        rho /= np.trace(rho)
        return rho

    @classmethod
    def rand_matrix(cls, rows, cols=None, real=False):
        """Return a random matrix."""
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        logger.debug("rand_matrix default_rng seeded with seed=%s", seed)
        rng = np.random.default_rng(seed)

        if cols is None:
            cols = rows
        if real:
            return rng.random(size=(rows, cols))
        return rng.random(size=(rows, cols)) + 1j * rng.random(size=(rows, cols))

    def simple_circuit_no_measure(self):
        """Return a unitary circuit and the corresponding unitary array."""
        qr = QuantumRegister(3)
        circ = QuantumCircuit(qr)
        circ.h(qr[0])
        circ.x(qr[1])
        circ.ry(np.pi / 2, qr[2])
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = Operator(np.kron(y90, np.kron(self.UX, self.UH)))
        return circ, target

    def simple_circuit_with_measure(self):
        """Return a unitary circuit with measurement."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        circ = QuantumCircuit(qr, cr)
        circ.h(qr[0])
        circ.x(qr[1])
        circ.measure(qr, cr)
        return circ


@ddt.ddt
class TestOperator(OperatorTestCase):
    """Tests for Operator linear operator class."""

    def test_init_array_qubit(self):
        """Test subsystem initialization from N-qubit array."""
        # Test automatic inference of qubit subsystems
        mat = self.rand_matrix(8, 8)
        op = Operator(mat)
        assert_allclose(op.data, mat)
        self.assertEqual(op.dim, (8, 8))
        self.assertEqual(op.input_dims(), (2, 2, 2))
        self.assertEqual(op.output_dims(), (2, 2, 2))
        self.assertEqual(op.num_qubits, 3)

        op = Operator(mat, input_dims=8, output_dims=8)
        assert_allclose(op.data, mat)
        self.assertEqual(op.dim, (8, 8))
        self.assertEqual(op.input_dims(), (2, 2, 2))
        self.assertEqual(op.output_dims(), (2, 2, 2))
        self.assertEqual(op.num_qubits, 3)

    def test_init_array(self):
        """Test initialization from array."""
        mat = np.eye(3)
        op = Operator(mat)
        assert_allclose(op.data, mat)
        self.assertEqual(op.dim, (3, 3))
        self.assertEqual(op.input_dims(), (3,))
        self.assertEqual(op.output_dims(), (3,))
        self.assertIsNone(op.num_qubits)

        mat = self.rand_matrix(2 * 3 * 4, 4 * 5)
        op = Operator(mat, input_dims=[4, 5], output_dims=[2, 3, 4])
        assert_allclose(op.data, mat)
        self.assertEqual(op.dim, (4 * 5, 2 * 3 * 4))
        self.assertEqual(op.input_dims(), (4, 5))
        self.assertEqual(op.output_dims(), (2, 3, 4))
        self.assertIsNone(op.num_qubits)

    def test_init_array_except(self):
        """Test initialization exception from array."""
        mat = self.rand_matrix(4, 4)
        self.assertRaises(QiskitError, Operator, mat, input_dims=[4, 2])
        self.assertRaises(QiskitError, Operator, mat, input_dims=[2, 4])
        self.assertRaises(QiskitError, Operator, mat, input_dims=5)

    def test_init_operator(self):
        """Test initialization from Operator."""
        op1 = Operator(self.rand_matrix(4, 4))
        op2 = Operator(op1)
        self.assertEqual(op1, op2)

    def test_circuit_init(self):
        """Test initialization from a circuit."""
        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        op = Operator(circuit)
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = np.kron(y90, np.kron(self.UX, self.UH))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of Controlled-Phase gate
        lam = np.pi / 4
        circuit = QuantumCircuit(2)
        circuit.cp(lam, 0, 1)
        op = Operator(circuit)
        target = np.diag([1, 1, 1, np.exp(1j * lam)])
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of controlled-H gate
        circuit = QuantumCircuit(2)
        circuit.ch(0, 1)
        op = Operator(circuit)
        target = np.kron(self.UI, np.diag([1, 0])) + np.kron(self.UH, np.diag([0, 1]))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_instruction_init(self):
        """Test initialization from a circuit."""
        gate = CXGate()
        op = Operator(gate).data
        target = gate.to_matrix()
        global_phase_equivalent = matrix_equal(op, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        gate = CHGate()
        op = Operator(gate).data
        had = HGate().to_matrix()
        target = np.kron(had, np.diag([0, 1])) + np.kron(np.eye(2), np.diag([1, 0]))
        global_phase_equivalent = matrix_equal(op, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, Operator, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        mat = self.rand_matrix(2, 2, real=True)
        self.assertEqual(Operator(np.array(mat, dtype=complex)), Operator(mat))
        mat = self.rand_matrix(4, 4)
        self.assertEqual(Operator(mat.tolist()), Operator(mat))

    def test_data(self):
        """Test Operator representation string property."""
        mat = self.rand_matrix(2, 2)
        op = Operator(mat)
        assert_allclose(mat, op.data)

    def test_to_matrix(self):
        """Test Operator to_matrix method."""
        mat = self.rand_matrix(2, 2)
        op = Operator(mat)
        assert_allclose(mat, op.to_matrix())

    def test_dim(self):
        """Test Operator dim property."""
        mat = self.rand_matrix(4, 4)
        self.assertEqual(Operator(mat).dim, (4, 4))
        self.assertEqual(Operator(mat, input_dims=[4], output_dims=[4]).dim, (4, 4))
        self.assertEqual(Operator(mat, input_dims=[2, 2], output_dims=[2, 2]).dim, (4, 4))

    def test_input_dims(self):
        """Test Operator input_dims method."""
        op = Operator(self.rand_matrix(2 * 3 * 4, 4 * 5), input_dims=[4, 5], output_dims=[2, 3, 4])
        self.assertEqual(op.input_dims(), (4, 5))
        self.assertEqual(op.input_dims(qargs=[0, 1]), (4, 5))
        self.assertEqual(op.input_dims(qargs=[1, 0]), (5, 4))
        self.assertEqual(op.input_dims(qargs=[0]), (4,))
        self.assertEqual(op.input_dims(qargs=[1]), (5,))

    def test_output_dims(self):
        """Test Operator output_dims method."""
        op = Operator(self.rand_matrix(2 * 3 * 4, 4 * 5), input_dims=[4, 5], output_dims=[2, 3, 4])
        self.assertEqual(op.output_dims(), (2, 3, 4))
        self.assertEqual(op.output_dims(qargs=[0, 1, 2]), (2, 3, 4))
        self.assertEqual(op.output_dims(qargs=[2, 1, 0]), (4, 3, 2))
        self.assertEqual(op.output_dims(qargs=[2, 0, 1]), (4, 2, 3))
        self.assertEqual(op.output_dims(qargs=[0]), (2,))
        self.assertEqual(op.output_dims(qargs=[1]), (3,))
        self.assertEqual(op.output_dims(qargs=[2]), (4,))
        self.assertEqual(op.output_dims(qargs=[0, 2]), (2, 4))
        self.assertEqual(op.output_dims(qargs=[2, 0]), (4, 2))

    def test_reshape(self):
        """Test Operator reshape method."""
        op = Operator(self.rand_matrix(8, 8))
        reshaped1 = op.reshape(input_dims=[8], output_dims=[8])
        reshaped2 = op.reshape(input_dims=[4, 2], output_dims=[2, 4])
        self.assertEqual(op.output_dims(), (2, 2, 2))
        self.assertEqual(op.input_dims(), (2, 2, 2))
        self.assertEqual(reshaped1.output_dims(), (8,))
        self.assertEqual(reshaped1.input_dims(), (8,))
        self.assertEqual(reshaped2.output_dims(), (2, 4))
        self.assertEqual(reshaped2.input_dims(), (4, 2))

    def test_reshape_num_qubits(self):
        """Test Operator reshape method with num_qubits."""
        op = Operator(self.rand_matrix(8, 8), input_dims=(4, 2), output_dims=(2, 4))
        reshaped = op.reshape(num_qubits=3)
        self.assertEqual(reshaped.num_qubits, 3)
        self.assertEqual(reshaped.output_dims(), (2, 2, 2))
        self.assertEqual(reshaped.input_dims(), (2, 2, 2))

    def test_reshape_raise(self):
        """Test Operator reshape method with invalid args."""
        op = Operator(self.rand_matrix(3, 3))
        self.assertRaises(QiskitError, op.reshape, num_qubits=2)

    def test_copy(self):
        """Test Operator copy method"""
        mat = np.eye(2)
        with self.subTest("Deep copy"):
            orig = Operator(mat)
            cpy = orig.copy()
            cpy._data[0, 0] = 0.0
            self.assertFalse(cpy == orig)
        with self.subTest("Shallow copy"):
            orig = Operator(mat)
            clone = copy.copy(orig)
            clone._data[0, 0] = 0.0
            self.assertTrue(clone == orig)

    def test_is_unitary(self):
        """Test is_unitary method."""
        # X-90 rotation
        X90 = scipy.linalg.expm(-1j * 0.5 * np.pi * np.array([[0, 1], [1, 0]]) / 2)
        self.assertTrue(Operator(X90).is_unitary())
        # Non-unitary should return false
        self.assertFalse(Operator([[1, 0], [0, 0]]).is_unitary())

    def test_to_operator(self):
        """Test to_operator method."""
        op1 = Operator(self.rand_matrix(4, 4))
        op2 = op1.to_operator()
        self.assertEqual(op1, op2)

    def test_conjugate(self):
        """Test conjugate method."""
        matr = self.rand_matrix(2, 4, real=True)
        mati = self.rand_matrix(2, 4, real=True)
        op = Operator(matr + 1j * mati)
        uni_conj = op.conjugate()
        self.assertEqual(uni_conj, Operator(matr - 1j * mati))

    def test_transpose(self):
        """Test transpose method."""
        matr = self.rand_matrix(2, 4, real=True)
        mati = self.rand_matrix(2, 4, real=True)
        op = Operator(matr + 1j * mati)
        uni_t = op.transpose()
        self.assertEqual(uni_t, Operator(matr.T + 1j * mati.T))

    def test_adjoint(self):
        """Test adjoint method."""
        matr = self.rand_matrix(2, 4, real=True)
        mati = self.rand_matrix(2, 4, real=True)
        op = Operator(matr + 1j * mati)
        uni_adj = op.adjoint()
        self.assertEqual(uni_adj, Operator(matr.T - 1j * mati.T))

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, Operator(np.eye(2)).compose, Operator(np.eye(3)))
        self.assertRaises(QiskitError, Operator(np.eye(2)).compose, 2)

    def test_compose(self):
        """Test compose method."""

        op1 = Operator(self.UX)
        op2 = Operator(self.UY)

        targ = Operator(np.dot(self.UY, self.UX))
        self.assertEqual(op1.compose(op2), targ)
        self.assertEqual(op1 & op2, targ)

        targ = Operator(np.dot(self.UX, self.UY))
        self.assertEqual(op2.compose(op1), targ)
        self.assertEqual(op2 & op1, targ)

    def test_dot(self):
        """Test dot method."""
        op1 = Operator(self.UY)
        op2 = Operator(self.UX)

        targ = Operator(np.dot(self.UY, self.UX))
        self.assertEqual(op1.dot(op2), targ)
        self.assertEqual(op1 @ op2, targ)

        targ = Operator(np.dot(self.UX, self.UY))
        self.assertEqual(op2.dot(op1), targ)
        self.assertEqual(op2 @ op1, targ)

    def test_compose_front(self):
        """Test front compose method."""

        opYX = Operator(self.UY).compose(Operator(self.UX), front=True)
        matYX = np.dot(self.UY, self.UX)
        self.assertEqual(opYX, Operator(matYX))

        opXY = Operator(self.UX).compose(Operator(self.UY), front=True)
        matXY = np.dot(self.UX, self.UY)
        self.assertEqual(opXY, Operator(matXY))

    def test_compose_subsystem(self):
        """Test subsystem compose method."""
        # 3-qubit operator
        mat = self.rand_matrix(8, 8)
        mat_a = self.rand_matrix(2, 2)
        mat_b = self.rand_matrix(2, 2)
        mat_c = self.rand_matrix(2, 2)
        op = Operator(mat)
        op1 = Operator(mat_a)
        op2 = Operator(np.kron(mat_b, mat_a))
        op3 = Operator(np.kron(mat_c, np.kron(mat_b, mat_a)))

        # op3 qargs=[0, 1, 2]
        targ = np.dot(np.kron(mat_c, np.kron(mat_b, mat_a)), mat)
        self.assertEqual(op.compose(op3, qargs=[0, 1, 2]), Operator(targ))
        self.assertEqual(op.compose(op3([0, 1, 2])), Operator(targ))
        self.assertEqual(op & op3([0, 1, 2]), Operator(targ))
        # op3 qargs=[2, 1, 0]
        targ = np.dot(np.kron(mat_a, np.kron(mat_b, mat_c)), mat)
        self.assertEqual(op.compose(op3, qargs=[2, 1, 0]), Operator(targ))
        self.assertEqual(op & op3([2, 1, 0]), Operator(targ))

        # op2 qargs=[0, 1]
        targ = np.dot(np.kron(np.eye(2), np.kron(mat_b, mat_a)), mat)
        self.assertEqual(op.compose(op2, qargs=[0, 1]), Operator(targ))
        self.assertEqual(op & op2([0, 1]), Operator(targ))
        # op2 qargs=[2, 0]
        targ = np.dot(np.kron(mat_a, np.kron(np.eye(2), mat_b)), mat)
        self.assertEqual(op.compose(op2, qargs=[2, 0]), Operator(targ))
        self.assertEqual(op & op2([2, 0]), Operator(targ))

        # op1 qargs=[0]
        targ = np.dot(np.kron(np.eye(4), mat_a), mat)
        self.assertEqual(op.compose(op1, qargs=[0]), Operator(targ))
        self.assertEqual(op & op1([0]), Operator(targ))
        # op1 qargs=[1]
        targ = np.dot(np.kron(np.eye(2), np.kron(mat_a, np.eye(2))), mat)
        self.assertEqual(op.compose(op1, qargs=[1]), Operator(targ))
        self.assertEqual(op & op1([1]), Operator(targ))
        # op1 qargs=[2]
        targ = np.dot(np.kron(mat_a, np.eye(4)), mat)
        self.assertEqual(op.compose(op1, qargs=[2]), Operator(targ))
        self.assertEqual(op & op1([2]), Operator(targ))

    def test_dot_subsystem(self):
        """Test subsystem dot method."""
        # 3-qubit operator
        mat = self.rand_matrix(8, 8)
        mat_a = self.rand_matrix(2, 2)
        mat_b = self.rand_matrix(2, 2)
        mat_c = self.rand_matrix(2, 2)
        op = Operator(mat)
        op1 = Operator(mat_a)
        op2 = Operator(np.kron(mat_b, mat_a))
        op3 = Operator(np.kron(mat_c, np.kron(mat_b, mat_a)))

        # op3 qargs=[0, 1, 2]
        targ = np.dot(mat, np.kron(mat_c, np.kron(mat_b, mat_a)))
        self.assertEqual(op.dot(op3, qargs=[0, 1, 2]), Operator(targ))
        self.assertEqual(op.dot(op3([0, 1, 2])), Operator(targ))
        # op3 qargs=[2, 1, 0]
        targ = np.dot(mat, np.kron(mat_a, np.kron(mat_b, mat_c)))
        self.assertEqual(op.dot(op3, qargs=[2, 1, 0]), Operator(targ))
        self.assertEqual(op.dot(op3([2, 1, 0])), Operator(targ))

        # op2 qargs=[0, 1]
        targ = np.dot(mat, np.kron(np.eye(2), np.kron(mat_b, mat_a)))
        self.assertEqual(op.dot(op2, qargs=[0, 1]), Operator(targ))
        self.assertEqual(op.dot(op2([0, 1])), Operator(targ))
        # op2 qargs=[2, 0]
        targ = np.dot(mat, np.kron(mat_a, np.kron(np.eye(2), mat_b)))
        self.assertEqual(op.dot(op2, qargs=[2, 0]), Operator(targ))
        self.assertEqual(op.dot(op2([2, 0])), Operator(targ))

        # op1 qargs=[0]
        targ = np.dot(mat, np.kron(np.eye(4), mat_a))
        self.assertEqual(op.dot(op1, qargs=[0]), Operator(targ))
        self.assertEqual(op.dot(op1([0])), Operator(targ))
        # op1 qargs=[1]
        targ = np.dot(mat, np.kron(np.eye(2), np.kron(mat_a, np.eye(2))))
        self.assertEqual(op.dot(op1, qargs=[1]), Operator(targ))
        self.assertEqual(op.dot(op1([1])), Operator(targ))
        # op1 qargs=[2]
        targ = np.dot(mat, np.kron(mat_a, np.eye(4)))
        self.assertEqual(op.dot(op1, qargs=[2]), Operator(targ))
        self.assertEqual(op.dot(op1([2])), Operator(targ))

    def test_compose_front_subsystem(self):
        """Test subsystem front compose method."""
        # 3-qubit operator
        mat = self.rand_matrix(8, 8)
        mat_a = self.rand_matrix(2, 2)
        mat_b = self.rand_matrix(2, 2)
        mat_c = self.rand_matrix(2, 2)
        op = Operator(mat)
        op1 = Operator(mat_a)
        op2 = Operator(np.kron(mat_b, mat_a))
        op3 = Operator(np.kron(mat_c, np.kron(mat_b, mat_a)))

        # op3 qargs=[0, 1, 2]
        targ = np.dot(mat, np.kron(mat_c, np.kron(mat_b, mat_a)))
        self.assertEqual(op.compose(op3, qargs=[0, 1, 2], front=True), Operator(targ))
        # op3 qargs=[2, 1, 0]
        targ = np.dot(mat, np.kron(mat_a, np.kron(mat_b, mat_c)))
        self.assertEqual(op.compose(op3, qargs=[2, 1, 0], front=True), Operator(targ))

        # op2 qargs=[0, 1]
        targ = np.dot(mat, np.kron(np.eye(2), np.kron(mat_b, mat_a)))
        self.assertEqual(op.compose(op2, qargs=[0, 1], front=True), Operator(targ))
        # op2 qargs=[2, 0]
        targ = np.dot(mat, np.kron(mat_a, np.kron(np.eye(2), mat_b)))
        self.assertEqual(op.compose(op2, qargs=[2, 0], front=True), Operator(targ))

        # op1 qargs=[0]
        targ = np.dot(mat, np.kron(np.eye(4), mat_a))
        self.assertEqual(op.compose(op1, qargs=[0], front=True), Operator(targ))

        # op1 qargs=[1]
        targ = np.dot(mat, np.kron(np.eye(2), np.kron(mat_a, np.eye(2))))
        self.assertEqual(op.compose(op1, qargs=[1], front=True), Operator(targ))

        # op1 qargs=[2]
        targ = np.dot(mat, np.kron(mat_a, np.eye(4)))
        self.assertEqual(op.compose(op1, qargs=[2], front=True), Operator(targ))

    def test_power(self):
        """Test power method."""
        X90 = scipy.linalg.expm(-1j * 0.5 * np.pi * np.array([[0, 1], [1, 0]]) / 2)
        op = Operator(X90)
        self.assertEqual(op.power(2), Operator([[0, -1j], [-1j, 0]]))
        self.assertEqual(op.power(4), Operator(-1 * np.eye(2)))
        self.assertEqual(op.power(8), Operator(np.eye(2)))

    def test_floating_point_power(self):
        """Test handling floating-point powers."""
        circuit = QuantumCircuit(2)
        circuit.crz(np.pi, 0, 1)
        op = Operator(circuit)

        expected_circuit = QuantumCircuit(2)
        expected_circuit.crz(np.pi / 4, 0, 1)
        expected_op = Operator(expected_circuit)

        self.assertEqual(op.power(0.25), expected_op)

    def test_power_of_nonunitary(self):
        """Test power method for a non-unitary matrix."""
        data = [[1, 1], [0, -1]]
        powered = Operator(data).power(0.5)
        expected = Operator([[1.0 + 0.0j, 0.5 - 0.5j], [0.0 + 0.0j, 0.0 + 1.0j]])
        assert_allclose(powered.data, expected.data)

    @ddt.data(
        (0.5, False),
        (1.0 / 3.0, False),
        (0.25, False),
        (0.5, True),
        (1.0 / 3.0, True),
        (0.25, True),
    )
    @ddt.unpack
    def test_root_stability(self, root, assume_unitary):
        """Test that the root of operators that have eigenvalues that are -1 up to floating-point
        imprecision stably choose the positive side of the principal-root branch cut."""
        rng = np.random.default_rng(2024_10_22)

        eigenvalues = np.array([1.0, -1.0], dtype=complex)
        # We have the eigenvalues exactly, so this will safely find the principal root.
        root_eigenvalues = eigenvalues**root

        # Now, we can construct a bunch of Haar-random unitaries with our chosen eigenvalues.  Since
        # we already know their eigenvalue decompositions exactly (up to floating-point precision in
        # the matrix multiplications), we can also compute the principal values of the roots of the
        # complete matrices.
        bases = scipy.stats.unitary_group.rvs(2, size=16, random_state=rng)
        matrices = [basis.conj().T @ np.diag(eigenvalues) @ basis for basis in bases]
        expected = [basis.conj().T @ np.diag(root_eigenvalues) @ basis for basis in bases]
        self.assertEqual(
            [Operator(matrix).power(root, assume_unitary=assume_unitary) for matrix in matrices],
            [Operator(single) for single in expected],
        )

    @ddt.data(True, False)
    def test_root_branch_cut(self, assume_unitary):
        """Test that we can choose where the branch cut appears in the root."""
        z_op = Operator(library.ZGate())
        # Depending on the direction we move the branch cut, we should be able to select the root to
        # be either of the two valid options.
        self.assertEqual(
            z_op.power(0.5, branch_cut_rotation=1e-3, assume_unitary=assume_unitary),
            Operator(library.SGate()),
        )
        self.assertEqual(
            z_op.power(0.5, branch_cut_rotation=-1e-3, assume_unitary=assume_unitary),
            Operator(library.SdgGate()),
        )

    def test_expand(self):
        """Test expand method."""
        mat1 = self.UX
        mat2 = np.eye(3, dtype=complex)

        mat21 = np.kron(mat2, mat1)
        op21 = Operator(mat1).expand(Operator(mat2))
        self.assertEqual(op21.dim, (6, 6))
        assert_allclose(op21.data, Operator(mat21).data)

        mat12 = np.kron(mat1, mat2)
        op12 = Operator(mat2).expand(Operator(mat1))
        self.assertEqual(op12.dim, (6, 6))
        assert_allclose(op12.data, Operator(mat12).data)

    def test_tensor(self):
        """Test tensor method."""
        mat1 = self.UX
        mat2 = np.eye(3, dtype=complex)

        mat21 = np.kron(mat2, mat1)
        op21 = Operator(mat2).tensor(Operator(mat1))
        self.assertEqual(op21.dim, (6, 6))
        assert_allclose(op21.data, Operator(mat21).data)

        mat12 = np.kron(mat1, mat2)
        op12 = Operator(mat1).tensor(Operator(mat2))
        self.assertEqual(op12.dim, (6, 6))
        assert_allclose(op12.data, Operator(mat12).data)

    def test_power_except(self):
        """Test power method raises exceptions if not square."""
        op = Operator(self.rand_matrix(2, 3))
        # Non-integer power raises error
        self.assertRaises(QiskitError, op.power, 0.5)

    def test_add(self):
        """Test add method."""
        mat1 = self.rand_matrix(4, 4)
        mat2 = self.rand_matrix(4, 4)
        op1 = Operator(mat1)
        op2 = Operator(mat2)
        self.assertEqual(op1._add(op2), Operator(mat1 + mat2))
        self.assertEqual(op1 + op2, Operator(mat1 + mat2))
        self.assertEqual(op1 - op2, Operator(mat1 - mat2))

    def test_add_except(self):
        """Test add method raises exceptions."""
        op1 = Operator(self.rand_matrix(2, 2))
        op2 = Operator(self.rand_matrix(3, 3))
        self.assertRaises(QiskitError, op1._add, op2)

    def test_add_qargs(self):
        """Test add method with qargs."""
        mat = self.rand_matrix(8, 8)
        mat0 = self.rand_matrix(2, 2)
        mat1 = self.rand_matrix(2, 2)

        op = Operator(mat)
        op0 = Operator(mat0)
        op01 = Operator(np.kron(mat1, mat0))

        with self.subTest(msg="qargs=[0]"):
            value = op + op0([0])
            target = op + Operator(np.kron(np.eye(4), mat0))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1]"):
            value = op + op0([1])
            target = op + Operator(np.kron(np.kron(np.eye(2), mat0), np.eye(2)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2]"):
            value = op + op0([2])
            target = op + Operator(np.kron(mat0, np.eye(4)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 1]"):
            value = op + op01([0, 1])
            target = op + Operator(np.kron(np.eye(2), np.kron(mat1, mat0)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1, 0]"):
            value = op + op01([1, 0])
            target = op + Operator(np.kron(np.eye(2), np.kron(mat0, mat1)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 2]"):
            value = op + op01([0, 2])
            target = op + Operator(np.kron(mat1, np.kron(np.eye(2), mat0)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2, 0]"):
            value = op + op01([2, 0])
            target = op + Operator(np.kron(mat0, np.kron(np.eye(2), mat1)))
            self.assertEqual(value, target)

    def test_sub_qargs(self):
        """Test subtract method with qargs."""
        mat = self.rand_matrix(8, 8)
        mat0 = self.rand_matrix(2, 2)
        mat1 = self.rand_matrix(2, 2)

        op = Operator(mat)
        op0 = Operator(mat0)
        op01 = Operator(np.kron(mat1, mat0))

        with self.subTest(msg="qargs=[0]"):
            value = op - op0([0])
            target = op - Operator(np.kron(np.eye(4), mat0))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1]"):
            value = op - op0([1])
            target = op - Operator(np.kron(np.kron(np.eye(2), mat0), np.eye(2)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2]"):
            value = op - op0([2])
            target = op - Operator(np.kron(mat0, np.eye(4)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 1]"):
            value = op - op01([0, 1])
            target = op - Operator(np.kron(np.eye(2), np.kron(mat1, mat0)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1, 0]"):
            value = op - op01([1, 0])
            target = op - Operator(np.kron(np.eye(2), np.kron(mat0, mat1)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 2]"):
            value = op - op01([0, 2])
            target = op - Operator(np.kron(mat1, np.kron(np.eye(2), mat0)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2, 0]"):
            value = op - op01([2, 0])
            target = op - Operator(np.kron(mat0, np.kron(np.eye(2), mat1)))
            self.assertEqual(value, target)

    def test_multiply(self):
        """Test multiply method."""
        mat = self.rand_matrix(4, 4)
        val = np.exp(5j)
        op = Operator(mat)
        self.assertEqual(op._multiply(val), Operator(val * mat))
        self.assertEqual(val * op, Operator(val * mat))
        self.assertEqual(op * val, Operator(mat * val))

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        op = Operator(self.rand_matrix(2, 2))
        self.assertRaises(QiskitError, op._multiply, "s")
        self.assertRaises(QiskitError, op.__rmul__, "s")
        self.assertRaises(QiskitError, op._multiply, op)
        self.assertRaises(QiskitError, op.__rmul__, op)

    def test_negate(self):
        """Test negate method"""
        mat = self.rand_matrix(4, 4)
        op = Operator(mat)
        self.assertEqual(-op, Operator(-1 * mat))

    def test_equiv(self):
        """Test negate method"""
        mat = np.diag([1, np.exp(1j * np.pi / 2)])
        phase = np.exp(-1j * np.pi / 4)
        op = Operator(mat)
        self.assertTrue(op.equiv(phase * mat))
        self.assertTrue(op.equiv(Operator(phase * mat)))
        self.assertFalse(op.equiv(2 * mat))

    def test_reverse_qargs(self):
        """Test reverse_qargs method"""
        circ1 = QFTGate(5).definition
        circ2 = circ1.reverse_bits()

        state1 = Operator(circ1)
        state2 = Operator(circ2)
        self.assertEqual(state1.reverse_qargs(), state2)

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    def test_drawings(self):
        """Test draw method"""
        qc1 = QFTGate(5).definition
        op = Operator.from_circuit(qc1)
        with self.subTest(msg="str(operator)"):
            str(op)
        for drawtype in ["repr", "text", "latex_source"]:
            with self.subTest(msg=f"draw('{drawtype}')"):
                op.draw(drawtype)
        with self.subTest(msg=" draw('latex')"):
            op.draw("latex")

    def test_from_circuit_constructor_no_layout(self):
        """Test initialization from a circuit using the from_circuit constructor."""
        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        op = Operator.from_circuit(circuit)
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = np.kron(y90, np.kron(self.UX, self.UH))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of Controlled-Phase gate
        lam = np.pi / 4
        circuit = QuantumCircuit(2)
        circuit.cp(lam, 0, 1)
        op = Operator.from_circuit(circuit)
        target = np.diag([1, 1, 1, np.exp(1j * lam)])
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of controlled-H gate
        circuit = QuantumCircuit(2)
        circuit.ch(0, 1)
        op = Operator.from_circuit(circuit)
        target = np.kron(self.UI, np.diag([1, 0])) + np.kron(self.UH, np.diag([0, 1]))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_from_circuit_initial_layout_final_layout(self):
        """Test initialization from a circuit with a non-trivial initial_layout and final_layout as given
        by a transpiled circuit."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(2, 1)
        qc.cx(1, 2)
        qc.cx(1, 0)
        qc.cx(1, 3)
        qc.cx(1, 4)
        qc.h(2)

        qc_transpiled = transpile(
            qc,
            coupling_map=CouplingMap.from_line(5),
            initial_layout=[2, 3, 4, 0, 1],
            optimization_level=1,
            seed_transpiler=17,
        )

        self.assertTrue(Operator.from_circuit(qc_transpiled).equiv(qc))

    def test_from_circuit_constructor_reverse_embedded_layout(self):
        """Test initialization from a circuit with an embedded reverse layout."""
        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(2)
        circuit.x(1)
        circuit.ry(np.pi / 2, 0)
        circuit._layout = TranspileLayout(
            Layout({circuit.qubits[2]: 0, circuit.qubits[1]: 1, circuit.qubits[0]: 2}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        op = Operator.from_circuit(circuit)
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = np.kron(y90, np.kron(self.UX, self.UH))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of Controlled-Phase gate
        lam = np.pi / 4
        circuit = QuantumCircuit(2)
        circuit.cp(lam, 1, 0)
        circuit._layout = TranspileLayout(
            Layout({circuit.qubits[1]: 0, circuit.qubits[0]: 1}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        op = Operator.from_circuit(circuit)
        target = np.diag([1, 1, 1, np.exp(1j * lam)])
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of controlled-H gate
        circuit = QuantumCircuit(2)
        circuit.ch(1, 0)
        circuit._layout = TranspileLayout(
            Layout({circuit.qubits[1]: 0, circuit.qubits[0]: 1}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        op = Operator.from_circuit(circuit)
        target = np.kron(self.UI, np.diag([1, 0])) + np.kron(self.UH, np.diag([0, 1]))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_from_circuit_constructor_reverse_embedded_layout_from_transpile(self):
        """Test initialization from a circuit with an embedded final layout."""
        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        output = transpile(circuit, initial_layout=[2, 1, 0])
        op = Operator.from_circuit(output)
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = np.kron(y90, np.kron(self.UX, self.UH))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_from_circuit_constructor_reverse_embedded_layout_from_transpile_with_registers(self):
        """Test initialization from a circuit with an embedded final layout."""
        # Test tensor product of 1-qubit gates
        qr = QuantumRegister(3, name="test_reg")
        circuit = QuantumCircuit(qr)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        output = transpile(circuit, initial_layout=[2, 1, 0])
        op = Operator.from_circuit(output)
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = np.kron(y90, np.kron(self.UX, self.UH))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_from_circuit_constructor_reverse_embedded_layout_and_final_layout(self):
        """Test initialization from a circuit with an embedded final layout."""
        # Test tensor product of 1-qubit gates
        qr = QuantumRegister(3, name="test_reg")
        circuit = QuantumCircuit(qr)
        circuit.h(2)
        circuit.x(1)
        circuit.ry(np.pi / 2, 0)
        circuit._layout = TranspileLayout(
            Layout({circuit.qubits[2]: 0, circuit.qubits[1]: 1, circuit.qubits[0]: 2}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
            Layout({circuit.qubits[0]: 2, circuit.qubits[1]: 0, circuit.qubits[2]: 1}),
        )
        circuit.swap(0, 1)
        circuit.swap(1, 2)
        op = Operator.from_circuit(circuit)
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = np.kron(y90, np.kron(self.UX, self.UH))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_from_circuit_constructor_reverse_embedded_layout_and_manual_final_layout(self):
        """Test initialization from a circuit with an embedded final layout."""
        # Test tensor product of 1-qubit gates
        qr = QuantumRegister(3, name="test_reg")
        circuit = QuantumCircuit(qr)
        circuit.h(2)
        circuit.x(1)
        circuit.ry(np.pi / 2, 0)
        circuit._layout = TranspileLayout(
            Layout({circuit.qubits[2]: 0, circuit.qubits[1]: 1, circuit.qubits[0]: 2}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        final_layout = Layout({circuit.qubits[0]: 2, circuit.qubits[1]: 0, circuit.qubits[2]: 1})
        circuit.swap(0, 1)
        circuit.swap(1, 2)
        op = Operator.from_circuit(circuit, final_layout=final_layout)
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = np.kron(y90, np.kron(self.UX, self.UH))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_from_circuit_constructor_reverse_embedded_layout_ignore_set_layout(self):
        """Test initialization from a circuit with an ignored embedded reverse layout."""
        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(2)
        circuit.x(1)
        circuit.ry(np.pi / 2, 0)
        circuit._layout = TranspileLayout(
            Layout({circuit.qubits[2]: 0, circuit.qubits[1]: 1, circuit.qubits[0]: 2}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        op = Operator.from_circuit(circuit, ignore_set_layout=True).reverse_qargs()
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = np.kron(y90, np.kron(self.UX, self.UH))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of Controlled-Phase gate
        lam = np.pi / 4
        circuit = QuantumCircuit(2)
        circuit.cp(lam, 1, 0)
        circuit._layout = TranspileLayout(
            Layout({circuit.qubits[1]: 0, circuit.qubits[0]: 1}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        op = Operator.from_circuit(circuit, ignore_set_layout=True).reverse_qargs()
        target = np.diag([1, 1, 1, np.exp(1j * lam)])
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of controlled-H gate
        circuit = QuantumCircuit(2)
        circuit.ch(1, 0)
        circuit._layout = TranspileLayout(
            Layout({circuit.qubits[1]: 0, circuit.qubits[0]: 1}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        op = Operator.from_circuit(circuit, ignore_set_layout=True).reverse_qargs()
        target = np.kron(self.UI, np.diag([1, 0])) + np.kron(self.UH, np.diag([0, 1]))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_from_circuit_constructor_reverse_user_specified_layout(self):
        """Test initialization from a circuit with a user specified reverse layout."""
        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(2)
        circuit.x(1)
        circuit.ry(np.pi / 2, 0)
        layout = Layout({circuit.qubits[2]: 0, circuit.qubits[1]: 1, circuit.qubits[0]: 2})
        op = Operator.from_circuit(circuit, layout=layout)
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = np.kron(y90, np.kron(self.UX, self.UH))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of Controlled-Phase gate
        lam = np.pi / 4
        circuit = QuantumCircuit(2)
        circuit.cp(lam, 1, 0)
        layout = Layout({circuit.qubits[1]: 0, circuit.qubits[0]: 1})
        op = Operator.from_circuit(circuit, layout=layout)
        target = np.diag([1, 1, 1, np.exp(1j * lam)])
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of controlled-H gate
        circuit = QuantumCircuit(2)
        circuit.ch(1, 0)
        layout = Layout({circuit.qubits[1]: 0, circuit.qubits[0]: 1})
        op = Operator.from_circuit(circuit, layout=layout)
        target = np.kron(self.UI, np.diag([1, 0])) + np.kron(self.UH, np.diag([0, 1]))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_from_circuit_constructor_ghz_out_of_order_layout(self):
        """Test an out of order ghz state with a layout set."""
        circuit = QuantumCircuit(5)
        circuit.h(3)
        circuit.cx(3, 4)
        circuit.cx(3, 2)
        circuit.cx(3, 0)
        circuit.cx(3, 1)
        circuit._layout = TranspileLayout(
            Layout(
                {
                    circuit.qubits[3]: 0,
                    circuit.qubits[4]: 1,
                    circuit.qubits[2]: 2,
                    circuit.qubits[0]: 3,
                    circuit.qubits[1]: 4,
                }
            ),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        result = Operator.from_circuit(circuit)
        expected = QuantumCircuit(5)
        expected.h(0)
        expected.cx(0, 1)
        expected.cx(0, 2)
        expected.cx(0, 3)
        expected.cx(0, 4)
        expected_op = Operator(expected)
        self.assertTrue(expected_op.equiv(result))

    def test_from_circuit_empty_circuit_empty_layout(self):
        """Test an out of order ghz state with a layout set."""
        circuit = QuantumCircuit()
        circuit._layout = TranspileLayout(Layout(), {})
        op = Operator.from_circuit(circuit)
        self.assertEqual(Operator([1]), op)

    def test_from_circuit_constructor_empty_layout(self):
        """Test an out of order ghz state with a layout set."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        layout = Layout()
        with self.assertRaises(KeyError):
            Operator.from_circuit(circuit, layout=layout)

    def test_compose_scalar(self):
        """Test that composition works with a scalar-valued operator over no qubits."""
        base = Operator(np.eye(2, dtype=np.complex128))
        scalar = Operator(np.array([[-1.0 + 0.0j]]))
        composed = base.compose(scalar, qargs=[])
        self.assertEqual(composed, Operator(-np.eye(2, dtype=np.complex128)))

    def test_compose_scalar_op(self):
        """Test that composition works with an explicit scalar operator over no qubits."""
        base = Operator(np.eye(2, dtype=np.complex128))
        scalar = ScalarOp(coeff=-1.0 + 0.0j)
        composed = base.compose(scalar, qargs=[])
        self.assertEqual(composed, Operator(-np.eye(2, dtype=np.complex128)))

    def test_from_circuit_single_flat_default_register_transpiled(self):
        """Test a transpiled circuit with layout set from default register."""
        circuit = QuantumCircuit(5)
        circuit.h(3)
        circuit.cx(3, 4)
        circuit.cx(3, 2)
        circuit.cx(3, 0)
        circuit.cx(3, 1)
        init_layout = Layout(
            {
                circuit.qubits[0]: 3,
                circuit.qubits[1]: 4,
                circuit.qubits[2]: 1,
                circuit.qubits[3]: 2,
                circuit.qubits[4]: 0,
            }
        )
        tqc = transpile(circuit, initial_layout=init_layout)
        result = Operator.from_circuit(tqc)
        self.assertTrue(Operator.from_circuit(circuit).equiv(result))

    def test_from_circuit_loose_bits_transpiled(self):
        """Test a transpiled circuit with layout set from loose bits."""
        bits = [Qubit() for _ in range(5)]
        circuit = QuantumCircuit()
        circuit.add_bits(bits)
        circuit.h(3)
        circuit.cx(3, 4)
        circuit.cx(3, 2)
        circuit.cx(3, 0)
        circuit.cx(3, 1)
        init_layout = Layout(
            {
                circuit.qubits[0]: 3,
                circuit.qubits[1]: 4,
                circuit.qubits[2]: 1,
                circuit.qubits[3]: 2,
                circuit.qubits[4]: 0,
            }
        )
        tqc = transpile(circuit, initial_layout=init_layout)
        result = Operator.from_circuit(tqc)
        self.assertTrue(Operator(circuit).equiv(result))

    def test_from_circuit_multiple_registers_bits_transpiled(self):
        """Test a transpiled circuit with layout set from loose bits."""
        regs = [QuantumRegister(1, name=f"custom_reg-{i}") for i in range(5)]
        circuit = QuantumCircuit()
        for reg in regs:
            circuit.add_register(reg)
        circuit.h(3)
        circuit.cx(3, 4)
        circuit.cx(3, 2)
        circuit.cx(3, 0)
        circuit.cx(3, 1)
        tqc = transpile(circuit, initial_layout=[3, 4, 1, 2, 0])
        result = Operator.from_circuit(tqc)
        self.assertTrue(Operator(circuit).equiv(result))

    def test_from_circuit_single_flat_custom_register_transpiled(self):
        """Test a transpiled circuit with layout set from loose bits."""
        circuit = QuantumCircuit(QuantumRegister(5, name="custom_reg"))
        circuit.h(3)
        circuit.cx(3, 4)
        circuit.cx(3, 2)
        circuit.cx(3, 0)
        circuit.cx(3, 1)
        tqc = transpile(circuit, initial_layout=[3, 4, 1, 2, 0])
        result = Operator.from_circuit(tqc)
        self.assertTrue(Operator(circuit).equiv(result))

    def test_from_circuit_mixed_reg_loose_bits_transpiled(self):
        """Test a transpiled circuit with layout set from loose bits."""
        bits = [Qubit(), Qubit()]
        circuit = QuantumCircuit()
        circuit.add_bits(bits)
        circuit.add_register(QuantumRegister(3, name="a_reg"))
        circuit.h(3)
        circuit.cx(3, 4)
        circuit.cx(3, 2)
        circuit.cx(3, 0)
        circuit.cx(3, 1)
        init_layout = Layout(
            {
                circuit.qubits[0]: 3,
                circuit.qubits[1]: 4,
                circuit.qubits[2]: 1,
                circuit.qubits[3]: 2,
                circuit.qubits[4]: 0,
            }
        )
        tqc = transpile(circuit, initial_layout=init_layout)
        result = Operator.from_circuit(tqc)
        self.assertTrue(Operator(circuit).equiv(result))

    def test_from_circuit_into_larger_map(self):
        """Test from_circuit method when the number of physical
        qubits is larger than the number of original virtual qubits."""

        # original circuit on 3 qubits
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        # transpile into 5-qubits
        tqc = transpile(qc, coupling_map=CouplingMap.from_line(5), initial_layout=[0, 2, 4])

        # qc expanded with ancilla qubits
        expected = QuantumCircuit(5)
        expected.h(0)
        expected.cx(0, 1)
        expected.cx(1, 2)

        self.assertEqual(Operator.from_circuit(tqc), Operator(expected))

    def test_apply_permutation_back(self):
        """Test applying permutation to the operator,
        where the operator is applied first and the permutation second."""
        op = Operator(self.rand_matrix(64, 64))
        pattern = [1, 2, 0, 3, 5, 4]

        # Consider several methods of computing this operator and show
        # they all lead to the same result.

        # Compose the operator with the operator constructed from the
        # permutation circuit.
        op2 = op.copy()
        perm_op = Operator(PermutationGate(pattern))
        op2 &= perm_op

        # Compose the operator with the operator constructed from the
        # permutation gate.
        op3 = op.copy()
        perm_op = Operator(PermutationGate(pattern))
        op3 &= perm_op

        # Modify the operator using apply_permutation method.
        op4 = op.copy()
        op4 = op4.apply_permutation(pattern, front=False)

        self.assertEqual(op2, op3)
        self.assertEqual(op2, op4)

    def test_apply_permutation_front(self):
        """Test applying permutation to the operator,
        where the permutation is applied first and the operator second"""
        op = Operator(self.rand_matrix(64, 64))
        pattern = [1, 2, 0, 3, 5, 4]

        # Consider several methods of computing this operator and show
        # they all lead to the same result.

        # Compose the operator with the operator constructed from the
        # permutation circuit.
        op2 = op.copy()
        perm_op = Operator(PermutationGate(pattern))
        op2 = perm_op & op2

        # Compose the operator with the operator constructed from the
        # permutation gate.
        op3 = op.copy()
        perm_op = Operator(PermutationGate(pattern))
        op3 = perm_op & op3

        # Modify the operator using apply_permutation method.
        op4 = op.copy()
        op4 = op4.apply_permutation(pattern, front=True)

        self.assertEqual(op2, op3)
        self.assertEqual(op2, op4)

    def test_apply_permutation_qudits_back(self):
        """Test applying permutation to the operator with heterogeneous qudit spaces,
        where the operator O is applied first and the permutation P second.
        The matrix of the resulting operator is the product [P][O] and
        corresponds to suitably permuting the rows of O's matrix.
        """
        mat = np.array(range(6 * 6)).reshape((6, 6))
        op = Operator(mat, input_dims=(2, 3), output_dims=(2, 3))
        perm = [1, 0]
        actual = op.apply_permutation(perm, front=False)

        # Rows of mat are ordered to 00, 01, 02, 10, 11, 12;
        # perm maps these to 00, 10, 20, 01, 11, 21,
        # while the default ordering is 00, 01, 10, 11, 20, 21.
        permuted_mat = mat.copy()[[0, 2, 4, 1, 3, 5]]
        expected = Operator(permuted_mat, input_dims=(2, 3), output_dims=(3, 2))
        self.assertEqual(actual, expected)

    def test_apply_permutation_qudits_front(self):
        """Test applying permutation to the operator with heterogeneous qudit spaces,
        where the permutation P is applied first and the operator O is applied second.
        The matrix of the resulting operator is the product [O][P] and
        corresponds to suitably permuting the columns of O's matrix.
        """
        mat = np.array(range(6 * 6)).reshape((6, 6))
        op = Operator(mat, input_dims=(2, 3), output_dims=(2, 3))
        perm = [1, 0]
        actual = op.apply_permutation(perm, front=True)

        # Columns of mat are ordered to 00, 01, 02, 10, 11, 12;
        # perm maps these to 00, 10, 20, 01, 11, 21,
        # while the default ordering is 00, 01, 10, 11, 20, 21.
        permuted_mat = mat.copy()[:, [0, 2, 4, 1, 3, 5]]
        expected = Operator(permuted_mat, input_dims=(3, 2), output_dims=(2, 3))
        self.assertEqual(actual, expected)

    @combine(
        dims=((2, 3, 4, 5), (5, 2, 4, 3), (3, 5, 2, 4), (5, 3, 4, 2), (4, 5, 2, 3), (4, 3, 2, 5))
    )
    def test_reverse_qargs_as_apply_permutation(self, dims):
        """Test reversing qargs by pre- and post-composing with reversal
        permutation.
        """
        perm = [3, 2, 1, 0]
        op = Operator(
            np.array(range(120 * 120)).reshape((120, 120)), input_dims=dims, output_dims=dims
        )
        op2 = op.reverse_qargs()
        op3 = op.apply_permutation(perm, front=True).apply_permutation(perm, front=False)
        self.assertEqual(op2, op3)

    def test_apply_permutation_exceptions(self):
        """Checks that applying permutation raises an error when dimensions do not match."""
        op = Operator(
            np.array(range(24 * 30)).reshape((24, 30)), input_dims=(6, 5), output_dims=(2, 3, 4)
        )

        with self.assertRaises(QiskitError):
            op.apply_permutation([1, 0], front=False)
        with self.assertRaises(QiskitError):
            op.apply_permutation([2, 1, 0], front=True)

    def test_apply_permutation_dimensions(self):
        """Checks the dimensions of the operator after applying permutation."""
        op = Operator(
            np.array(range(24 * 30)).reshape((24, 30)), input_dims=(6, 5), output_dims=(2, 3, 4)
        )
        op2 = op.apply_permutation([1, 2, 0], front=False)
        self.assertEqual(op2.output_dims(), (4, 2, 3))

        op = Operator(
            np.array(range(24 * 30)).reshape((30, 24)), input_dims=(2, 3, 4), output_dims=(6, 5)
        )
        op2 = op.apply_permutation([2, 0, 1], front=True)
        self.assertEqual(op2.input_dims(), (4, 2, 3))

    def test_equality_with_ancillas(self):
        """Check correctness of the equal_with_ancillas method."""

        # The two circuits below are equal provided that qubit 1 is initially |0>.
        qc1 = QuantumCircuit(4)
        qc1.x(0)
        qc1.x(2)
        qc1.cx(1, 0)
        qc1.cx(1, 2)
        qc1.cx(1, 3)
        op1 = Operator(qc1)

        qc2 = QuantumCircuit(4)
        qc2.x(0)
        qc2.x(2)
        op2 = Operator(qc2)

        self.assertNotEqual(op1, op2)
        self.assertFalse(_equal_with_ancillas(op1, op2, []))
        self.assertTrue(_equal_with_ancillas(op1, op2, [1]))
        self.assertFalse(_equal_with_ancillas(op1, op2, [2]))
        self.assertTrue(_equal_with_ancillas(op1, op2, [2, 1]))


class TestUnitarySim(QiskitTestCase):
    """Test simulation of unitary circuits in Rust."""

    def test_equivalence(self):
        """Test equivalence of Python and Rust code"""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.swap(1, 2)
        qc.s(1)
        qc.cx(3, 1)
        qc.rz(0.1, 2)
        qc.ccx(3, 4, 0)
        qc.s(4)
        python_mat = Operator(qc).data
        rust_mat = unitary_sim.sim_unitary_circuit(qc._data)
        assert_allclose(python_mat, rust_mat)

    def test_equivalence_with_barriers(self):
        """Test equivalence of Python and Rust code where the
        circuit includes barriers"""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        qc.swap(1, 2)
        qc.s(1)
        qc.cx(3, 1)
        qc.rz(0.1, 2)
        qc.barrier()
        qc.ccx(3, 4, 0)
        qc.s(4)
        qc.barrier()

        python_mat = Operator(qc).data
        rust_mat = unitary_sim.sim_unitary_circuit(qc._data)
        assert_allclose(python_mat, rust_mat)

    def test_raises_on_classical_bits(self):
        """Test that simulating circuits with classical bits
        raises an error."""
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        qc.cx(0, 1)

        with self.assertRaises(QiskitError):
            _ = unitary_sim.sim_unitary_circuit(qc._data)


if __name__ == "__main__":
    unittest.main()
