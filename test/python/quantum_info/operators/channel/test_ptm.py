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

"""Tests for PTM quantum channel representation class."""

import copy
import unittest
import numpy as np
from numpy.testing import assert_allclose

from qiskit import QiskitError
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info.operators.channel import PTM
from .channel_test_case import ChannelTestCase


class TestPTM(ChannelTestCase):
    """Tests for PTM channel representation."""

    def test_init(self):
        """Test initialization"""
        mat4 = np.eye(4) / 2.0
        chan = PTM(mat4)
        assert_allclose(chan.data, mat4)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        mat16 = np.eye(16) / 4
        chan = PTM(mat16)
        assert_allclose(chan.data, mat16)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(chan.num_qubits, 2)

        # Wrong input or output dims should raise exception
        self.assertRaises(QiskitError, PTM, mat16, input_dims=2, output_dims=4)

        # Non multi-qubit dimensions should raise exception
        self.assertRaises(QiskitError, PTM, np.eye(6) / 2, input_dims=3, output_dims=2)

    def test_circuit_init(self):
        """Test initialization from a circuit."""
        circuit, target = self.simple_circuit_no_measure()
        op = PTM(circuit)
        target = PTM(target)
        self.assertEqual(op, target)

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, PTM, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        mat = self.rand_matrix(4, 4, real=True)
        self.assertEqual(PTM(mat), PTM(mat))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(4)
        with self.subTest("Deep copy"):
            orig = PTM(mat)
            cpy = orig.copy()
            cpy._data[0, 0] = 0.0
            self.assertFalse(cpy == orig)
        with self.subTest("Shallow copy"):
            orig = PTM(mat)
            clone = copy.copy(orig)
            clone._data[0, 0] = 0.0
            self.assertTrue(clone == orig)

    def test_clone(self):
        """Test clone method"""
        mat = np.eye(4)
        orig = PTM(mat)
        clone = copy.copy(orig)
        clone._data[0, 0] = 0.0
        self.assertTrue(clone == orig)

    def test_is_cptp(self):
        """Test is_cptp method."""
        self.assertTrue(PTM(self.depol_ptm(0.25)).is_cptp())
        # Non-CPTP should return false
        self.assertFalse(PTM(1.25 * self.ptmI - 0.25 * self.depol_ptm(1)).is_cptp())

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, PTM(np.eye(4)).compose, PTM(np.eye(16)))
        self.assertRaises(QiskitError, PTM(np.eye(4)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = PTM(self.ptmX)
        chan2 = PTM(self.ptmY)
        chan = chan1.compose(chan2)
        rho_targ = rho.evolve(PTM(self.ptmZ))
        self.assertEqual(rho.evolve(chan), rho_targ)

        # 50% depolarizing channel
        chan1 = PTM(self.depol_ptm(0.5))
        chan = chan1.compose(chan1)
        rho_targ = rho.evolve(PTM(self.depol_ptm(0.75)))
        self.assertEqual(rho.evolve(chan), rho_targ)

        # Compose random
        ptm1 = self.rand_matrix(4, 4, real=True)
        ptm2 = self.rand_matrix(4, 4, real=True)
        chan1 = PTM(ptm1, input_dims=2, output_dims=2)
        chan2 = PTM(ptm2, input_dims=2, output_dims=2)
        rho_targ = rho.evolve(chan1).evolve(chan2)
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho.evolve(chan), rho_targ)
        chan = chan1 & chan2
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho.evolve(chan), rho_targ)

    def test_dot(self):
        """Test dot method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = PTM(self.ptmX)
        chan2 = PTM(self.ptmY)
        rho_targ = rho.evolve(PTM(self.ptmZ))
        self.assertEqual(rho.evolve(chan2.dot(chan1)), rho_targ)
        self.assertEqual(rho.evolve(chan2 @ chan1), rho_targ)

        # Compose random
        ptm1 = self.rand_matrix(4, 4, real=True)
        ptm2 = self.rand_matrix(4, 4, real=True)
        chan1 = PTM(ptm1, input_dims=2, output_dims=2)
        chan2 = PTM(ptm2, input_dims=2, output_dims=2)
        rho_targ = rho.evolve(chan1).evolve(chan2)
        self.assertEqual(rho.evolve(chan2.dot(chan1)), rho_targ)
        self.assertEqual(rho.evolve(chan2 @ chan1), rho_targ)

    def test_compose_front(self):
        """Test deprecated front compose method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = PTM(self.ptmX)
        chan2 = PTM(self.ptmY)
        chan = chan2.compose(chan1, front=True)
        rho_targ = rho.evolve(PTM(self.ptmZ))
        self.assertEqual(rho.evolve(chan), rho_targ)

        # Compose random
        ptm1 = self.rand_matrix(4, 4, real=True)
        ptm2 = self.rand_matrix(4, 4, real=True)
        chan1 = PTM(ptm1, input_dims=2, output_dims=2)
        chan2 = PTM(ptm2, input_dims=2, output_dims=2)
        rho_targ = rho.evolve(chan1).evolve(chan2)
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho.evolve(chan), rho_targ)

    def test_expand(self):
        """Test expand method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = PTM(self.ptmI)
        chan2 = PTM(self.ptmX)

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
        chan_dep = PTM(self.depol_ptm(1))
        chan = chan_dep.expand(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_tensor(self):
        """Test tensor method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = PTM(self.ptmI)
        chan2 = PTM(self.ptmX)

        # X \otimes I
        chan = chan2.tensor(chan1)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # I \otimes X
        chan = chan1.tensor(chan2)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # Completely depolarizing
        chan_dep = PTM(self.depol_ptm(1))
        chan = chan_dep.tensor(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_power(self):
        """Test power method."""
        # 10% depolarizing channel
        p_id = 0.9
        depol = PTM(self.depol_ptm(1 - p_id))

        # Compose 3 times
        p_id3 = p_id**3
        chan3 = depol.power(3)
        targ3 = PTM(self.depol_ptm(1 - p_id3))
        self.assertEqual(chan3, targ3)

    def test_add(self):
        """Test add method."""
        mat1 = 0.5 * self.ptmI
        mat2 = 0.5 * self.depol_ptm(1)

        chan1 = PTM(mat1)
        chan2 = PTM(mat2)

        targ = PTM(mat1 + mat2)
        self.assertEqual(chan1._add(chan2), targ)
        self.assertEqual(chan1 + chan2, targ)
        targ = PTM(mat1 - mat2)
        self.assertEqual(chan1 - chan2, targ)

    def test_add_qargs(self):
        """Test add method with qargs."""
        mat = self.rand_matrix(8**2, 8**2)
        mat0 = self.rand_matrix(4, 4)
        mat1 = self.rand_matrix(4, 4)

        op = PTM(mat)
        op0 = PTM(mat0)
        op1 = PTM(mat1)
        op01 = op1.tensor(op0)
        eye = PTM(self.ptmI)

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

        op = PTM(mat)
        op0 = PTM(mat0)
        op1 = PTM(mat1)
        op01 = op1.tensor(op0)
        eye = PTM(self.ptmI)

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
        chan1 = PTM(self.ptmI)
        chan2 = PTM(np.eye(16))
        self.assertRaises(QiskitError, chan1._add, chan2)
        self.assertRaises(QiskitError, chan1._add, 5)

    def test_multiply(self):
        """Test multiply method."""
        chan = PTM(self.ptmI)
        val = 0.5
        targ = PTM(val * self.ptmI)
        self.assertEqual(chan._multiply(val), targ)
        self.assertEqual(val * chan, targ)
        targ = PTM(self.ptmI * val)
        self.assertEqual(chan * val, targ)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = PTM(self.ptmI)
        self.assertRaises(QiskitError, chan._multiply, "s")
        self.assertRaises(QiskitError, chan.__rmul__, "s")
        self.assertRaises(QiskitError, chan._multiply, chan)
        self.assertRaises(QiskitError, chan.__rmul__, chan)

    def test_negate(self):
        """Test negate method"""
        chan = PTM(self.ptmI)
        targ = PTM(-self.ptmI)
        self.assertEqual(-chan, targ)


if __name__ == "__main__":
    unittest.main()
