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

"""Tests for Chi quantum channel representation class."""

import copy
import unittest
import numpy as np
from numpy.testing import assert_allclose

from qiskit import QiskitError
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info.operators.channel import Chi
from .channel_test_case import ChannelTestCase


class TestChi(ChannelTestCase):
    """Tests for Chi channel representation."""

    def test_init(self):
        """Test initialization"""
        mat4 = np.eye(4) / 2.0
        chan = Chi(mat4)
        assert_allclose(chan.data, mat4)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        mat16 = np.eye(16) / 4
        chan = Chi(mat16)
        assert_allclose(chan.data, mat16)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(chan.num_qubits, 2)

        # Wrong input or output dims should raise exception
        self.assertRaises(QiskitError, Chi, mat16, input_dims=2, output_dims=4)

        # Non multi-qubit dimensions should raise exception
        self.assertRaises(QiskitError, Chi, np.eye(6) / 2, input_dims=3, output_dims=2)

    def test_circuit_init(self):
        """Test initialization from a circuit."""
        circuit, target = self.simple_circuit_no_measure()
        op = Chi(circuit)
        target = Chi(target)
        self.assertEqual(op, target)

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, Chi, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        mat = self.rand_matrix(4, 4, real=True)
        self.assertEqual(Chi(mat), Chi(mat))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(4)
        with self.subTest("Deep copy"):
            orig = Chi(mat)
            cpy = orig.copy()
            cpy._data[0, 0] = 0.0
            self.assertFalse(cpy == orig)
        with self.subTest("Shallow copy"):
            orig = Chi(mat)
            clone = copy.copy(orig)
            clone._data[0, 0] = 0.0
            self.assertTrue(clone == orig)

    def test_is_cptp(self):
        """Test is_cptp method."""
        self.assertTrue(Chi(self.depol_chi(0.25)).is_cptp())
        # Non-CPTP should return false
        self.assertFalse(Chi(1.25 * self.chiI - 0.25 * self.depol_chi(1)).is_cptp())

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, Chi(np.eye(4)).compose, Chi(np.eye(16)))
        self.assertRaises(QiskitError, Chi(np.eye(4)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = Chi(self.chiX)
        chan2 = Chi(self.chiY)
        chan = chan1.compose(chan2)
        target = rho.evolve(Chi(self.chiZ))
        output = rho.evolve(chan)
        self.assertEqual(output, target)

        # 50% depolarizing channel
        chan1 = Chi(self.depol_chi(0.5))
        chan = chan1.compose(chan1)
        target = rho.evolve(Chi(self.depol_chi(0.75)))
        output = rho.evolve(chan)
        self.assertEqual(output, target)

        # Compose random
        chi1 = self.rand_matrix(4, 4, real=True)
        chi2 = self.rand_matrix(4, 4, real=True)
        chan1 = Chi(chi1, input_dims=2, output_dims=2)
        chan2 = Chi(chi2, input_dims=2, output_dims=2)
        target = rho.evolve(chan1).evolve(chan2)
        chan = chan1.compose(chan2)
        output = rho.evolve(chan)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(output, target)
        chan = chan1 & chan2
        output = rho.evolve(chan)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(output, target)

    def test_dot(self):
        """Test dot method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = Chi(self.chiX)
        chan2 = Chi(self.chiY)
        target = rho.evolve(Chi(self.chiZ))
        output = rho.evolve(chan2.dot(chan1))
        self.assertEqual(output, target)

        # Compose random
        chi1 = self.rand_matrix(4, 4, real=True)
        chi2 = self.rand_matrix(4, 4, real=True)
        chan1 = Chi(chi1, input_dims=2, output_dims=2)
        chan2 = Chi(chi2, input_dims=2, output_dims=2)
        target = rho.evolve(chan1).evolve(chan2)
        chan = chan2.dot(chan1)
        output = rho.evolve(chan)
        self.assertEqual(output, target)
        chan = chan2 @ chan1
        output = rho.evolve(chan)
        self.assertEqual(output, target)

    def test_compose_front(self):
        """Test front compose method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = Chi(self.chiX)
        chan2 = Chi(self.chiY)
        chan = chan2.compose(chan1, front=True)
        target = rho.evolve(Chi(self.chiZ))
        output = rho.evolve(chan)
        self.assertEqual(output, target)

        # Compose random
        chi1 = self.rand_matrix(4, 4, real=True)
        chi2 = self.rand_matrix(4, 4, real=True)
        chan1 = Chi(chi1, input_dims=2, output_dims=2)
        chan2 = Chi(chi2, input_dims=2, output_dims=2)
        target = rho.evolve(chan1).evolve(chan2)
        chan = chan2.compose(chan1, front=True)
        output = rho.evolve(chan)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(output, target)

    def test_expand(self):
        """Test expand method."""
        # Pauli channels
        paulis = [self.chiI, self.chiX, self.chiY, self.chiZ]
        targs = 4 * np.eye(16)  # diagonals of Pauli channel Chi mats
        for i, chi1 in enumerate(paulis):
            for j, chi2 in enumerate(paulis):
                chan1 = Chi(chi1)
                chan2 = Chi(chi2)
                chan = chan1.expand(chan2)
                # Target for diagonal Pauli channel
                targ = Chi(np.diag(targs[i + 4 * j]))
                self.assertEqual(chan.dim, (4, 4))
                self.assertEqual(chan, targ)

        # Completely depolarizing
        rho = DensityMatrix(np.diag([1, 0, 0, 0]))
        chan_dep = Chi(self.depol_chi(1))
        chan = chan_dep.expand(chan_dep)
        target = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        output = rho.evolve(chan)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(output, target)

    def test_tensor(self):
        """Test tensor method."""
        # Pauli channels
        paulis = [self.chiI, self.chiX, self.chiY, self.chiZ]
        targs = 4 * np.eye(16)  # diagonals of Pauli channel Chi mats
        for i, chi1 in enumerate(paulis):
            for j, chi2 in enumerate(paulis):
                chan1 = Chi(chi1)
                chan2 = Chi(chi2)
                chan = chan2.tensor(chan1)
                # Target for diagonal Pauli channel
                targ = Chi(np.diag(targs[i + 4 * j]))
                self.assertEqual(chan.dim, (4, 4))
                self.assertEqual(chan, targ)
                # Test overload
                chan = chan2 ^ chan1
                self.assertEqual(chan.dim, (4, 4))
                self.assertEqual(chan, targ)

        # Completely depolarizing
        rho = DensityMatrix(np.diag([1, 0, 0, 0]))
        chan_dep = Chi(self.depol_chi(1))
        chan = chan_dep.tensor(chan_dep)
        target = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        output = rho.evolve(chan)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(output, target)
        # Test operator overload
        chan = chan_dep ^ chan_dep
        output = rho.evolve(chan)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(output, target)

    def test_power(self):
        """Test power method."""
        # 10% depolarizing channel
        p_id = 0.9
        depol = Chi(self.depol_chi(1 - p_id))

        # Compose 3 times
        p_id3 = p_id**3
        chan3 = depol.power(3)
        targ3 = Chi(self.depol_chi(1 - p_id3))
        self.assertEqual(chan3, targ3)

    def test_add(self):
        """Test add method."""
        mat1 = 0.5 * self.chiI
        mat2 = 0.5 * self.depol_chi(1)
        chan1 = Chi(mat1)
        chan2 = Chi(mat2)

        targ = Chi(mat1 + mat2)
        self.assertEqual(chan1._add(chan2), targ)
        self.assertEqual(chan1 + chan2, targ)

        targ = Chi(mat1 - mat2)
        self.assertEqual(chan1 - chan2, targ)

    def test_add_qargs(self):
        """Test add method with qargs."""
        mat = self.rand_matrix(8**2, 8**2)
        mat0 = self.rand_matrix(4, 4)
        mat1 = self.rand_matrix(4, 4)

        op = Chi(mat)
        op0 = Chi(mat0)
        op1 = Chi(mat1)
        op01 = op1.tensor(op0)
        eye = Chi(self.chiI)

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

        op = Chi(mat)
        op0 = Chi(mat0)
        op1 = Chi(mat1)
        op01 = op1.tensor(op0)
        eye = Chi(self.chiI)

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
        chan1 = Chi(self.chiI)
        chan2 = Chi(np.eye(16))
        self.assertRaises(QiskitError, chan1._add, chan2)
        self.assertRaises(QiskitError, chan1._add, 5)

    def test_multiply(self):
        """Test multiply method."""
        chan = Chi(self.chiI)
        val = 0.5
        targ = Chi(val * self.chiI)
        self.assertEqual(chan._multiply(val), targ)
        self.assertEqual(val * chan, targ)
        targ = Chi(self.chiI * val)
        self.assertEqual(chan * val, targ)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = Chi(self.chiI)
        self.assertRaises(QiskitError, chan._multiply, "s")
        self.assertRaises(QiskitError, chan.__rmul__, "s")
        self.assertRaises(QiskitError, chan._multiply, chan)
        self.assertRaises(QiskitError, chan.__rmul__, chan)

    def test_negate(self):
        """Test negate method"""
        chan = Chi(self.chiI)
        targ = Chi(-self.chiI)
        self.assertEqual(-chan, targ)


if __name__ == "__main__":
    unittest.main()
