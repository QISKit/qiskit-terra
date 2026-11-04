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

"""Tests for SuperOp quantum channel representation class."""

import copy
import unittest
import numpy as np
from numpy.testing import assert_allclose

from qiskit import QiskitError, QuantumCircuit
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.operators.channel import SuperOp
from .channel_test_case import ChannelTestCase


class TestSuperOp(ChannelTestCase):
    """Tests for SuperOp channel representation."""

    def test_init(self):
        """Test initialization"""
        chan = SuperOp(self.sopI)
        assert_allclose(chan.data, self.sopI)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        mat = np.zeros((4, 16))
        chan = SuperOp(mat)
        assert_allclose(chan.data, mat)
        self.assertEqual(chan.dim, (4, 2))
        self.assertIsNone(chan.num_qubits)

        chan = SuperOp(mat.T)
        assert_allclose(chan.data, mat.T)
        self.assertEqual(chan.dim, (2, 4))
        self.assertIsNone(chan.num_qubits)

        # Wrong input or output dims should raise exception
        self.assertRaises(QiskitError, SuperOp, mat, input_dims=[4], output_dims=[4])

    def test_circuit_init(self):
        """Test initialization from a circuit."""
        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        op = SuperOp(circuit)
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = SuperOp(Operator(np.kron(y90, np.kron(self.UX, self.UH))))
        self.assertEqual(target, op)

        # Test decomposition of Controlled-Phase gate
        lam = np.pi / 4
        circuit = QuantumCircuit(2)
        circuit.cp(lam, 0, 1)
        op = SuperOp(circuit)
        target = SuperOp(Operator(np.diag([1, 1, 1, np.exp(1j * lam)])))
        self.assertEqual(target, op)

        # Test decomposition of controlled-H gate
        circuit = QuantumCircuit(2)
        circuit.ch(0, 1)
        op = SuperOp(circuit)
        target = SuperOp(
            Operator(np.kron(self.UI, np.diag([1, 0])) + np.kron(self.UH, np.diag([0, 1])))
        )
        self.assertEqual(target, op)

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, SuperOp, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        mat = self.rand_matrix(4, 4)
        self.assertEqual(SuperOp(mat), SuperOp(mat))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(4)
        with self.subTest("Deep copy"):
            orig = SuperOp(mat)
            cpy = orig.copy()
            cpy._data[0, 0] = 0.0
            self.assertFalse(cpy == orig)
        with self.subTest("Shallow copy"):
            orig = SuperOp(mat)
            clone = copy.copy(orig)
            clone._data[0, 0] = 0.0
            self.assertTrue(clone == orig)

    def test_clone(self):
        """Test clone method"""
        mat = np.eye(4)
        orig = SuperOp(mat)
        clone = copy.copy(orig)
        clone._data[0, 0] = 0.0
        self.assertTrue(clone == orig)

    def test_evolve(self):
        """Test evolve method."""
        input_rho = DensityMatrix([[0, 0], [0, 1]])
        # Identity channel
        chan = SuperOp(self.sopI)
        target_rho = DensityMatrix([[0, 0], [0, 1]])
        self.assertEqual(input_rho.evolve(chan), target_rho)

        # Hadamard channel
        mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        chan = SuperOp(np.kron(mat.conj(), mat))
        target_rho = DensityMatrix(np.array([[1, -1], [-1, 1]]) / 2)
        self.assertEqual(input_rho.evolve(chan), target_rho)

        # Completely depolarizing channel
        chan = SuperOp(self.depol_sop(1))
        target_rho = DensityMatrix(np.eye(2) / 2)
        self.assertEqual(input_rho.evolve(chan), target_rho)

    def test_evolve_subsystem(self):
        """Test subsystem evolve method."""

        # Single-qubit random superoperators
        op_a = SuperOp(self.rand_matrix(4, 4))
        op_b = SuperOp(self.rand_matrix(4, 4))
        op_c = SuperOp(self.rand_matrix(4, 4))
        id1 = SuperOp(np.eye(4))
        id2 = SuperOp(np.eye(16))
        rho = DensityMatrix(self.rand_rho(8))

        # Test evolving single-qubit of 3-qubit system
        op = op_a

        # Evolve on qubit 0
        full_op = id2.tensor(op_a)
        rho_targ = rho.evolve(full_op)
        rho_test = rho.evolve(op, qargs=[0])
        self.assertEqual(rho_test, rho_targ)

        # Evolve on qubit 1
        full_op = id1.tensor(op_a).tensor(id1)
        rho_targ = rho.evolve(full_op)
        rho_test = rho.evolve(op, qargs=[1])
        self.assertEqual(rho_test, rho_targ)

        # Evolve on qubit 2
        full_op = op_a.tensor(id2)
        rho_targ = rho.evolve(full_op)
        rho_test = rho.evolve(op, qargs=[2])
        self.assertEqual(rho_test, rho_targ)

        # Test 2-qubit evolution
        op = op_b.tensor(op_a)

        # Evolve on qubits [0, 2]
        full_op = op_b.tensor(id1).tensor(op_a)
        rho_targ = rho.evolve(full_op)
        rho_test = rho.evolve(op, qargs=[0, 2])
        self.assertEqual(rho_test, rho_targ)
        # Evolve on qubits [2, 0]
        full_op = op_a.tensor(id1).tensor(op_b)
        rho_targ = rho.evolve(full_op)
        rho_test = rho.evolve(op, qargs=[2, 0])
        self.assertEqual(rho_test, rho_targ)

        # Test 3-qubit evolution
        op = op_c.tensor(op_b).tensor(op_a)

        # Evolve on qubits [0, 1, 2]
        full_op = op
        rho_targ = rho.evolve(full_op)
        rho_test = rho.evolve(op, qargs=[0, 1, 2])
        self.assertEqual(rho_test, rho_targ)

        # Evolve on qubits [2, 1, 0]
        full_op = op_a.tensor(op_b).tensor(op_c)
        rho_targ = rho.evolve(full_op)
        rho_test = rho.evolve(op, qargs=[2, 1, 0])
        self.assertEqual(rho_test, rho_targ)

    def test_is_cptp(self):
        """Test is_cptp method."""
        self.assertTrue(SuperOp(self.depol_sop(0.25)).is_cptp())
        # Non-CPTP should return false
        self.assertFalse(SuperOp(1.25 * self.sopI - 0.25 * self.depol_sop(1)).is_cptp())

    def test_conjugate(self):
        """Test conjugate method."""
        mat = self.rand_matrix(4, 4)
        chan = SuperOp(mat)
        targ = SuperOp(np.conjugate(mat))
        self.assertEqual(chan.conjugate(), targ)

    def test_transpose(self):
        """Test transpose method."""
        mat = self.rand_matrix(4, 4)
        chan = SuperOp(mat)
        targ = SuperOp(np.transpose(mat))
        self.assertEqual(chan.transpose(), targ)

    def test_adjoint(self):
        """Test adjoint method."""
        mat = self.rand_matrix(4, 4)
        chan = SuperOp(mat)
        targ = SuperOp(np.transpose(np.conj(mat)))
        self.assertEqual(chan.adjoint(), targ)

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, SuperOp(np.eye(4)).compose, SuperOp(np.eye(16)))
        self.assertRaises(QiskitError, SuperOp(np.eye(4)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        # UnitaryChannel evolution
        chan1 = SuperOp(self.sopX)
        chan2 = SuperOp(self.sopY)
        chan = chan1.compose(chan2)
        targ = SuperOp(self.sopZ)
        self.assertEqual(chan, targ)

        # 50% depolarizing channel
        chan1 = SuperOp(self.depol_sop(0.5))
        chan = chan1.compose(chan1)
        targ = SuperOp(self.depol_sop(0.75))
        self.assertEqual(chan, targ)

        # Random superoperator
        mat1 = self.rand_matrix(4, 4)
        mat2 = self.rand_matrix(4, 4)
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        targ = SuperOp(np.dot(mat2, mat1))
        self.assertEqual(chan1.compose(chan2), targ)
        self.assertEqual(chan1 & chan2, targ)
        targ = SuperOp(np.dot(mat1, mat2))
        self.assertEqual(chan2.compose(chan1), targ)
        self.assertEqual(chan2 & chan1, targ)

        # Compose different dimensions
        chan1 = SuperOp(self.rand_matrix(16, 4))
        chan2 = SuperOp(
            self.rand_matrix(4, 16),
        )
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dim, (2, 2))
        chan = chan2.compose(chan1)
        self.assertEqual(chan.dim, (4, 4))

    def test_dot(self):
        """Test dot method."""
        # UnitaryChannel evolution
        chan1 = SuperOp(self.sopX)
        chan2 = SuperOp(self.sopY)
        targ = SuperOp(self.sopZ)
        self.assertEqual(chan1.dot(chan2), targ)
        self.assertEqual(chan1 @ chan2, targ)

        # 50% depolarizing channel
        chan1 = SuperOp(self.depol_sop(0.5))
        targ = SuperOp(self.depol_sop(0.75))
        self.assertEqual(chan1.dot(chan1), targ)
        self.assertEqual(chan1 @ chan1, targ)

        # Random superoperator
        mat1 = self.rand_matrix(4, 4)
        mat2 = self.rand_matrix(4, 4)
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        targ = SuperOp(np.dot(mat2, mat1))
        self.assertEqual(chan2.dot(chan1), targ)
        targ = SuperOp(np.dot(mat1, mat2))

        # Compose different dimensions
        chan1 = SuperOp(self.rand_matrix(16, 4))
        chan2 = SuperOp(self.rand_matrix(4, 16))
        chan = chan1.dot(chan2)
        self.assertEqual(chan.dim, (4, 4))
        chan = chan1 @ chan2
        self.assertEqual(chan.dim, (4, 4))
        chan = chan2.dot(chan1)
        self.assertEqual(chan.dim, (2, 2))
        chan = chan2 @ chan1
        self.assertEqual(chan.dim, (2, 2))

    def test_compose_front(self):
        """Test front compose method."""
        # DEPRECATED
        # UnitaryChannel evolution
        chan1 = SuperOp(self.sopX)
        chan2 = SuperOp(self.sopY)
        chan = chan1.compose(chan2, front=True)
        targ = SuperOp(self.sopZ)
        self.assertEqual(chan, targ)

        # 50% depolarizing channel
        chan1 = SuperOp(self.depol_sop(0.5))
        chan = chan1.compose(chan1, front=True)
        targ = SuperOp(self.depol_sop(0.75))
        self.assertEqual(chan, targ)

        # Random superoperator
        mat1 = self.rand_matrix(4, 4)
        mat2 = self.rand_matrix(4, 4)
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        targ = SuperOp(np.dot(mat2, mat1))
        self.assertEqual(chan2.compose(chan1, front=True), targ)
        targ = SuperOp(np.dot(mat1, mat2))
        self.assertEqual(chan1.compose(chan2, front=True), targ)

        # Compose different dimensions
        chan1 = SuperOp(self.rand_matrix(16, 4))
        chan2 = SuperOp(self.rand_matrix(4, 16))
        chan = chan1.compose(chan2, front=True)
        self.assertEqual(chan.dim, (4, 4))
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dim, (2, 2))

    def test_compose_subsystem(self):
        """Test subsystem compose method."""
        # 3-qubit superoperator
        mat = self.rand_matrix(64, 64)
        mat_a = self.rand_matrix(4, 4)
        mat_b = self.rand_matrix(4, 4)
        mat_c = self.rand_matrix(4, 4)
        iden = SuperOp(np.eye(4))

        op = SuperOp(mat)
        op1 = SuperOp(mat_a)
        op2 = SuperOp(mat_b).tensor(SuperOp(mat_a))
        op3 = SuperOp(mat_c).tensor(SuperOp(mat_b)).tensor(SuperOp(mat_a))
        # op3 qargs=[0, 1, 2]
        full_op = SuperOp(mat_c).tensor(SuperOp(mat_b)).tensor(SuperOp(mat_a))
        targ = np.dot(full_op.data, mat)
        self.assertEqual(op.compose(op3, qargs=[0, 1, 2]), SuperOp(targ))
        self.assertEqual(op & op3([0, 1, 2]), SuperOp(targ))
        # op3 qargs=[2, 1, 0]
        full_op = SuperOp(mat_a).tensor(SuperOp(mat_b)).tensor(SuperOp(mat_c))
        targ = np.dot(full_op.data, mat)
        self.assertEqual(op.compose(op3, qargs=[2, 1, 0]), SuperOp(targ))
        self.assertEqual(op & op3([2, 1, 0]), SuperOp(targ))

        # op2 qargs=[0, 1]
        full_op = iden.tensor(SuperOp(mat_b)).tensor(SuperOp(mat_a))
        targ = np.dot(full_op.data, mat)
        self.assertEqual(op.compose(op2, qargs=[0, 1]), SuperOp(targ))
        self.assertEqual(op & op2([0, 1]), SuperOp(targ))
        # op2 qargs=[2, 0]
        full_op = SuperOp(mat_a).tensor(iden).tensor(SuperOp(mat_b))
        targ = np.dot(full_op.data, mat)
        self.assertEqual(op.compose(op2, qargs=[2, 0]), SuperOp(targ))
        self.assertEqual(op & op2([2, 0]), SuperOp(targ))

        # op1 qargs=[0]
        full_op = iden.tensor(iden).tensor(SuperOp(mat_a))
        targ = np.dot(full_op.data, mat)
        self.assertEqual(op.compose(op1, qargs=[0]), SuperOp(targ))
        self.assertEqual(op & op1([0]), SuperOp(targ))

        # op1 qargs=[1]
        full_op = iden.tensor(SuperOp(mat_a)).tensor(iden)
        targ = np.dot(full_op.data, mat)
        self.assertEqual(op.compose(op1, qargs=[1]), SuperOp(targ))
        self.assertEqual(op & op1([1]), SuperOp(targ))

        # op1 qargs=[2]
        full_op = SuperOp(mat_a).tensor(iden).tensor(iden)
        targ = np.dot(full_op.data, mat)
        self.assertEqual(op.compose(op1, qargs=[2]), SuperOp(targ))
        self.assertEqual(op & op1([2]), SuperOp(targ))

    def test_dot_subsystem(self):
        """Test subsystem dot method."""
        # 3-qubit operator
        mat = self.rand_matrix(64, 64)
        mat_a = self.rand_matrix(4, 4)
        mat_b = self.rand_matrix(4, 4)
        mat_c = self.rand_matrix(4, 4)
        iden = SuperOp(np.eye(4))
        op = SuperOp(mat)
        op1 = SuperOp(mat_a)
        op2 = SuperOp(mat_b).tensor(SuperOp(mat_a))
        op3 = SuperOp(mat_c).tensor(SuperOp(mat_b)).tensor(SuperOp(mat_a))

        # op3 qargs=[0, 1, 2]
        full_op = SuperOp(mat_c).tensor(SuperOp(mat_b)).tensor(SuperOp(mat_a))
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.dot(op3, qargs=[0, 1, 2]), SuperOp(targ))
        # op3 qargs=[2, 1, 0]
        full_op = SuperOp(mat_a).tensor(SuperOp(mat_b)).tensor(SuperOp(mat_c))
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.dot(op3, qargs=[2, 1, 0]), SuperOp(targ))

        # op2 qargs=[0, 1]
        full_op = iden.tensor(SuperOp(mat_b)).tensor(SuperOp(mat_a))
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.dot(op2, qargs=[0, 1]), SuperOp(targ))
        # op2 qargs=[2, 0]
        full_op = SuperOp(mat_a).tensor(iden).tensor(SuperOp(mat_b))
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.dot(op2, qargs=[2, 0]), SuperOp(targ))

        # op1 qargs=[0]
        full_op = iden.tensor(iden).tensor(SuperOp(mat_a))
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.dot(op1, qargs=[0]), SuperOp(targ))
        # op1 qargs=[1]
        full_op = iden.tensor(SuperOp(mat_a)).tensor(iden)
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.dot(op1, qargs=[1]), SuperOp(targ))

        # op1 qargs=[2]
        full_op = SuperOp(mat_a).tensor(iden).tensor(iden)
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.dot(op1, qargs=[2]), SuperOp(targ))

    def test_compose_front_subsystem(self):
        """Test subsystem front compose method."""
        # 3-qubit operator
        mat = self.rand_matrix(64, 64)
        mat_a = self.rand_matrix(4, 4)
        mat_b = self.rand_matrix(4, 4)
        mat_c = self.rand_matrix(4, 4)
        iden = SuperOp(np.eye(4))
        op = SuperOp(mat)
        op1 = SuperOp(mat_a)
        op2 = SuperOp(mat_b).tensor(SuperOp(mat_a))
        op3 = SuperOp(mat_c).tensor(SuperOp(mat_b)).tensor(SuperOp(mat_a))

        # op3 qargs=[0, 1, 2]
        full_op = SuperOp(mat_c).tensor(SuperOp(mat_b)).tensor(SuperOp(mat_a))
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.compose(op3, qargs=[0, 1, 2], front=True), SuperOp(targ))
        # op3 qargs=[2, 1, 0]
        full_op = SuperOp(mat_a).tensor(SuperOp(mat_b)).tensor(SuperOp(mat_c))
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.compose(op3, qargs=[2, 1, 0], front=True), SuperOp(targ))

        # op2 qargs=[0, 1]
        full_op = iden.tensor(SuperOp(mat_b)).tensor(SuperOp(mat_a))
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.compose(op2, qargs=[0, 1], front=True), SuperOp(targ))
        # op2 qargs=[2, 0]
        full_op = SuperOp(mat_a).tensor(iden).tensor(SuperOp(mat_b))
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.compose(op2, qargs=[2, 0], front=True), SuperOp(targ))

        # op1 qargs=[0]
        full_op = iden.tensor(iden).tensor(SuperOp(mat_a))
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.compose(op1, qargs=[0], front=True), SuperOp(targ))

        # op1 qargs=[1]
        full_op = iden.tensor(SuperOp(mat_a)).tensor(iden)
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.compose(op1, qargs=[1], front=True), SuperOp(targ))

        # op1 qargs=[2]
        full_op = SuperOp(mat_a).tensor(iden).tensor(iden)
        targ = np.dot(mat, full_op.data)
        self.assertEqual(op.compose(op1, qargs=[2], front=True), SuperOp(targ))

    def test_expand(self):
        """Test expand method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(self.sopX)

        # X \otimes I
        chan = chan1.expand(chan2)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # I \otimes X
        chan = chan2.expand(chan1)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_tensor(self):
        """Test tensor method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(self.sopX)

        # X \otimes I
        chan = chan2.tensor(chan1)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan2 ^ chan1
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        # I \otimes X
        chan = chan1.tensor(chan2)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan1 ^ chan2
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_power(self):
        """Test power method."""
        # 10% depolarizing channel
        p_id = 0.9
        depol = SuperOp(self.depol_sop(1 - p_id))

        # Compose 3 times
        p_id3 = p_id**3
        chan3 = depol.power(3)
        targ3 = SuperOp(self.depol_sop(1 - p_id3))
        self.assertEqual(chan3, targ3)

    def test_add(self):
        """Test add method."""
        mat1 = 0.5 * self.sopI
        mat2 = 0.5 * self.depol_sop(1)
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        targ = SuperOp(mat1 + mat2)
        self.assertEqual(chan1._add(chan2), targ)
        self.assertEqual(chan1 + chan2, targ)
        targ = SuperOp(mat1 - mat2)
        self.assertEqual(chan1 - chan2, targ)

    def test_add_qargs(self):
        """Test add method with qargs."""
        mat = self.rand_matrix(8**2, 8**2)
        mat0 = self.rand_matrix(4, 4)
        mat1 = self.rand_matrix(4, 4)

        op = SuperOp(mat)
        op0 = SuperOp(mat0)
        op1 = SuperOp(mat1)
        op01 = op1.tensor(op0)
        eye = SuperOp(self.sopI)

        with self.subTest(msg="qargs=[0]"):
            value = op + op0([0])
            target = op + eye.tensor(eye).tensor(op0)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1]"):
            value = op + op0([1])
            target = op + eye.tensor(op0).tensor(eye)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2]"):
            value = op + op0([2])
            target = op + op0.tensor(eye).tensor(eye)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 1]"):
            value = op + op01([0, 1])
            target = op + eye.tensor(op1).tensor(op0)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1, 0]"):
            value = op + op01([1, 0])
            target = op + eye.tensor(op0).tensor(op1)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 2]"):
            value = op + op01([0, 2])
            target = op + op1.tensor(eye).tensor(op0)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2, 0]"):
            value = op + op01([2, 0])
            target = op + op0.tensor(eye).tensor(op1)
            self.assertEqual(value, target)

    def test_sub_qargs(self):
        """Test subtract method with qargs."""
        mat = self.rand_matrix(8**2, 8**2)
        mat0 = self.rand_matrix(4, 4)
        mat1 = self.rand_matrix(4, 4)

        op = SuperOp(mat)
        op0 = SuperOp(mat0)
        op1 = SuperOp(mat1)
        op01 = op1.tensor(op0)
        eye = SuperOp(self.sopI)

        with self.subTest(msg="qargs=[0]"):
            value = op - op0([0])
            target = op - eye.tensor(eye).tensor(op0)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1]"):
            value = op - op0([1])
            target = op - eye.tensor(op0).tensor(eye)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2]"):
            value = op - op0([2])
            target = op - op0.tensor(eye).tensor(eye)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 1]"):
            value = op - op01([0, 1])
            target = op - eye.tensor(op1).tensor(op0)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1, 0]"):
            value = op - op01([1, 0])
            target = op - eye.tensor(op0).tensor(op1)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 2]"):
            value = op - op01([0, 2])
            target = op - op1.tensor(eye).tensor(op0)
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2, 0]"):
            value = op - op01([2, 0])
            target = op - op0.tensor(eye).tensor(op1)
            self.assertEqual(value, target)

    def test_add_except(self):
        """Test add method raises exceptions."""
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(np.eye(16))
        self.assertRaises(QiskitError, chan1._add, chan2)
        self.assertRaises(QiskitError, chan1._add, 5)

    def test_multiply(self):
        """Test multiply method."""
        chan = SuperOp(self.sopI)
        val = 0.5
        targ = SuperOp(val * self.sopI)
        self.assertEqual(chan._multiply(val), targ)
        self.assertEqual(val * chan, targ)
        targ = SuperOp(self.sopI * val)
        self.assertEqual(chan * val, targ)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = SuperOp(self.sopI)
        self.assertRaises(QiskitError, chan._multiply, "s")
        self.assertRaises(QiskitError, chan.__rmul__, "s")
        self.assertRaises(QiskitError, chan._multiply, chan)
        self.assertRaises(QiskitError, chan.__rmul__, chan)

    def test_negate(self):
        """Test negate method"""
        chan = SuperOp(self.sopI)
        targ = SuperOp(-self.sopI)
        self.assertEqual(-chan, targ)


if __name__ == "__main__":
    unittest.main()
