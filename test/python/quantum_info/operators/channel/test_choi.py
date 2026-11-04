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


"""Tests for Choi quantum channel representation class."""

import copy
import unittest
import numpy as np
from numpy.testing import assert_allclose

from qiskit import QiskitError
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info.operators.channel import Choi
from .channel_test_case import ChannelTestCase


class TestChoi(ChannelTestCase):
    """Tests for Choi channel representation."""

    def test_init(self):
        """Test initialization"""
        mat4 = np.eye(4) / 2.0
        chan = Choi(mat4)
        assert_allclose(chan.data, mat4)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        mat8 = np.eye(8) / 2.0
        chan = Choi(mat8, input_dims=4)
        assert_allclose(chan.data, mat8)
        self.assertEqual(chan.dim, (4, 2))
        self.assertIsNone(chan.num_qubits)

        chan = Choi(mat8, input_dims=2)
        assert_allclose(chan.data, mat8)
        self.assertEqual(chan.dim, (2, 4))
        self.assertIsNone(chan.num_qubits)

        mat16 = np.eye(16) / 4
        chan = Choi(mat16)
        assert_allclose(chan.data, mat16)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(chan.num_qubits, 2)

        # Wrong input or output dims should raise exception
        self.assertRaises(QiskitError, Choi, mat8, input_dims=[4], output_dims=[4])

    def test_circuit_init(self):
        """Test initialization from a circuit."""
        circuit, target = self.simple_circuit_no_measure()
        op = Choi(circuit)
        target = Choi(target)
        self.assertEqual(op, target)

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, Choi, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        mat = self.rand_matrix(4, 4)
        self.assertEqual(Choi(mat), Choi(mat))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(2)
        with self.subTest("Deep copy"):
            orig = Choi(mat)
            cpy = orig.copy()
            cpy._data[0, 0] = 0.0
            self.assertFalse(cpy == orig)
        with self.subTest("Shallow copy"):
            orig = Choi(mat)
            clone = copy.copy(orig)
            clone._data[0, 0] = 0.0
            self.assertTrue(clone == orig)

    def test_clone(self):
        """Test clone method"""
        mat = np.eye(4)
        orig = Choi(mat)
        clone = copy.copy(orig)
        clone._data[0, 0] = 0.0
        self.assertTrue(clone == orig)

    def test_is_cptp(self):
        """Test is_cptp method."""
        self.assertTrue(Choi(self.depol_choi(0.25)).is_cptp())
        # Non-CPTP should return false
        self.assertFalse(Choi(1.25 * self.choiI - 0.25 * self.depol_choi(1)).is_cptp())

    def test_conjugate(self):
        """Test conjugate method."""
        # Test channel measures in Z basis and prepares in Y basis
        # Zp -> Yp, Zm -> Ym
        Zp, Zm = np.diag([1, 0]), np.diag([0, 1])
        Yp, Ym = np.array([[1, -1j], [1j, 1]]) / 2, np.array([[1, 1j], [-1j, 1]]) / 2
        chan = Choi(np.kron(Zp, Yp) + np.kron(Zm, Ym))
        # Conjugate channel swaps Y-basis states
        targ = Choi(np.kron(Zp, Ym) + np.kron(Zm, Yp))
        chan_conj = chan.conjugate()
        self.assertEqual(chan_conj, targ)

    def test_transpose(self):
        """Test transpose method."""
        # Test channel measures in Z basis and prepares in Y basis
        # Zp -> Yp, Zm -> Ym
        Zp, Zm = np.diag([1, 0]), np.diag([0, 1])
        Yp, Ym = np.array([[1, -1j], [1j, 1]]) / 2, np.array([[1, 1j], [-1j, 1]]) / 2
        chan = Choi(np.kron(Zp, Yp) + np.kron(Zm, Ym))
        # Transpose channel swaps basis
        targ = Choi(np.kron(Yp, Zp) + np.kron(Ym, Zm))
        chan_t = chan.transpose()
        self.assertEqual(chan_t, targ)

    def test_adjoint(self):
        """Test adjoint method."""
        # Test channel measures in Z basis and prepares in Y basis
        # Zp -> Yp, Zm -> Ym
        Zp, Zm = np.diag([1, 0]), np.diag([0, 1])
        Yp, Ym = np.array([[1, -1j], [1j, 1]]) / 2, np.array([[1, 1j], [-1j, 1]]) / 2
        chan = Choi(np.kron(Zp, Yp) + np.kron(Zm, Ym))
        # Ajoint channel swaps Y-basis elements and Z<->Y bases
        targ = Choi(np.kron(Ym, Zp) + np.kron(Yp, Zm))
        chan_adj = chan.adjoint()
        self.assertEqual(chan_adj, targ)

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, Choi(np.eye(4)).compose, Choi(np.eye(8)))
        self.assertRaises(QiskitError, Choi(np.eye(4)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        # UnitaryChannel evolution
        chan1 = Choi(self.choiX)
        chan2 = Choi(self.choiY)
        chan = chan1.compose(chan2)
        targ = Choi(self.choiZ)
        self.assertEqual(chan, targ)

        # 50% depolarizing channel
        chan1 = Choi(self.depol_choi(0.5))
        chan = chan1.compose(chan1)
        targ = Choi(self.depol_choi(0.75))
        self.assertEqual(chan, targ)

        # Measure and rotation
        Zp, Zm = np.diag([1, 0]), np.diag([0, 1])
        Xp, Xm = np.array([[1, 1], [1, 1]]) / 2, np.array([[1, -1], [-1, 1]]) / 2
        chan1 = Choi(np.kron(Zp, Xp) + np.kron(Zm, Xm))
        chan2 = Choi(self.choiX)
        # X-gate second does nothing
        targ = Choi(np.kron(Zp, Xp) + np.kron(Zm, Xm))
        self.assertEqual(chan1.compose(chan2), targ)
        self.assertEqual(chan1 & chan2, targ)
        # X-gate first swaps Z states
        targ = Choi(np.kron(Zm, Xp) + np.kron(Zp, Xm))
        self.assertEqual(chan2.compose(chan1), targ)
        self.assertEqual(chan2 & chan1, targ)

        # Compose different dimensions
        chan1 = Choi(np.eye(8) / 4, input_dims=2, output_dims=4)
        chan2 = Choi(np.eye(8) / 2, input_dims=4, output_dims=2)
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dim, (2, 2))
        chan = chan2.compose(chan1)
        self.assertEqual(chan.dim, (4, 4))

    def test_dot(self):
        """Test dot method."""
        # UnitaryChannel evolution
        chan1 = Choi(self.choiX)
        chan2 = Choi(self.choiY)
        targ = Choi(self.choiZ)
        self.assertEqual(chan1.dot(chan2), targ)
        self.assertEqual(chan1 @ chan2, targ)

        # 50% depolarizing channel
        chan1 = Choi(self.depol_choi(0.5))
        targ = Choi(self.depol_choi(0.75))
        self.assertEqual(chan1.dot(chan1), targ)
        self.assertEqual(chan1 @ chan1, targ)

        # Measure and rotation
        Zp, Zm = np.diag([1, 0]), np.diag([0, 1])
        Xp, Xm = np.array([[1, 1], [1, 1]]) / 2, np.array([[1, -1], [-1, 1]]) / 2
        chan1 = Choi(np.kron(Zp, Xp) + np.kron(Zm, Xm))
        chan2 = Choi(self.choiX)
        # X-gate second does nothing
        targ = Choi(np.kron(Zp, Xp) + np.kron(Zm, Xm))
        self.assertEqual(chan2.dot(chan1), targ)
        self.assertEqual(chan2 @ chan1, targ)
        # X-gate first swaps Z states
        targ = Choi(np.kron(Zm, Xp) + np.kron(Zp, Xm))
        self.assertEqual(chan1.dot(chan2), targ)
        self.assertEqual(chan1 @ chan2, targ)

        # Compose different dimensions
        chan1 = Choi(np.eye(8) / 4, input_dims=2, output_dims=4)
        chan2 = Choi(np.eye(8) / 2, input_dims=4, output_dims=2)
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
        # UnitaryChannel evolution
        chan1 = Choi(self.choiX)
        chan2 = Choi(self.choiY)
        chan = chan1.compose(chan2, front=True)
        targ = Choi(self.choiZ)
        self.assertEqual(chan, targ)

        # 50% depolarizing channel
        chan1 = Choi(self.depol_choi(0.5))
        chan = chan1.compose(chan1, front=True)
        targ = Choi(self.depol_choi(0.75))
        self.assertEqual(chan, targ)

        # Measure and rotation
        Zp, Zm = np.diag([1, 0]), np.diag([0, 1])
        Xp, Xm = np.array([[1, 1], [1, 1]]) / 2, np.array([[1, -1], [-1, 1]]) / 2
        chan1 = Choi(np.kron(Zp, Xp) + np.kron(Zm, Xm))
        chan2 = Choi(self.choiX)
        # X-gate second does nothing
        chan = chan2.compose(chan1, front=True)
        targ = Choi(np.kron(Zp, Xp) + np.kron(Zm, Xm))
        self.assertEqual(chan, targ)
        # X-gate first swaps Z states
        chan = chan1.compose(chan2, front=True)
        targ = Choi(np.kron(Zm, Xp) + np.kron(Zp, Xm))
        self.assertEqual(chan, targ)

        # Compose different dimensions
        chan1 = Choi(np.eye(8) / 4, input_dims=2, output_dims=4)
        chan2 = Choi(np.eye(8) / 2, input_dims=4, output_dims=2)
        chan = chan1.compose(chan2, front=True)
        self.assertEqual(chan.dim, (4, 4))
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dim, (2, 2))

    def test_expand(self):
        """Test expand method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Choi(self.choiI)
        chan2 = Choi(self.choiX)

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

        # Completely depolarizing
        chan_dep = Choi(self.depol_choi(1))
        chan = chan_dep.expand(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_tensor(self):
        """Test tensor method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Choi(self.choiI)
        chan2 = Choi(self.choiX)

        # X \otimes I
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        chan = chan2.tensor(chan1)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan2 ^ chan1
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # I \otimes X
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        chan = chan1.tensor(chan2)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan1 ^ chan2
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # Completely depolarizing
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        chan_dep = Choi(self.depol_choi(1))
        chan = chan_dep.tensor(chan_dep)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan_dep ^ chan_dep
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_power(self):
        """Test power method."""
        # 10% depolarizing channel
        p_id = 0.9
        depol = Choi(self.depol_choi(1 - p_id))

        # Compose 3 times
        p_id3 = p_id**3
        chan3 = depol.power(3)
        targ3 = Choi(self.depol_choi(1 - p_id3))
        self.assertEqual(chan3, targ3)

    def test_add(self):
        """Test add method."""
        mat1 = 0.5 * self.choiI
        mat2 = 0.5 * self.depol_choi(1)
        chan1 = Choi(mat1)
        chan2 = Choi(mat2)
        targ = Choi(mat1 + mat2)
        self.assertEqual(chan1._add(chan2), targ)
        self.assertEqual(chan1 + chan2, targ)
        targ = Choi(mat1 - mat2)
        self.assertEqual(chan1 - chan2, targ)

    def test_add_qargs(self):
        """Test add method with qargs."""
        mat = self.rand_matrix(8**2, 8**2)
        mat0 = self.rand_matrix(4, 4)
        mat1 = self.rand_matrix(4, 4)

        op = Choi(mat)
        op0 = Choi(mat0)
        op1 = Choi(mat1)
        op01 = op1.tensor(op0)
        eye = Choi(self.choiI)

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

        op = Choi(mat)
        op0 = Choi(mat0)
        op1 = Choi(mat1)
        op01 = op1.tensor(op0)
        eye = Choi(self.choiI)

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
        chan1 = Choi(self.choiI)
        chan2 = Choi(np.eye(8))
        self.assertRaises(QiskitError, chan1._add, chan2)
        self.assertRaises(QiskitError, chan1._add, 5)

    def test_multiply(self):
        """Test multiply method."""
        chan = Choi(self.choiI)
        val = 0.5
        targ = Choi(val * self.choiI)
        self.assertEqual(chan._multiply(val), targ)
        self.assertEqual(val * chan, targ)
        targ = Choi(self.choiI * val)
        self.assertEqual(chan * val, targ)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = Choi(self.choiI)
        self.assertRaises(QiskitError, chan._multiply, "s")
        self.assertRaises(QiskitError, chan.__rmul__, "s")
        self.assertRaises(QiskitError, chan._multiply, chan)
        self.assertRaises(QiskitError, chan.__rmul__, chan)

    def test_negate(self):
        """Test negate method"""
        chan = Choi(self.choiI)
        targ = Choi(-1 * self.choiI)
        self.assertEqual(-chan, targ)


if __name__ == "__main__":
    unittest.main()
