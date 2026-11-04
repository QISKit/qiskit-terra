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

"""Tests for Kraus quantum channel representation class."""

import copy
import unittest
import numpy as np
from numpy.testing import assert_allclose

from qiskit import QiskitError
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info import Kraus, Operator
from .channel_test_case import ChannelTestCase


class TestKraus(ChannelTestCase):
    """Tests for Kraus channel representation."""

    def test_init(self):
        """Test initialization"""
        # Initialize from unitary
        chan = Kraus(self.UI)
        assert_allclose(chan.data, [self.UI])
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        # Initialize from Kraus
        chan = Kraus(self.depol_kraus(0.5))
        assert_allclose(chan.data, self.depol_kraus(0.5))
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        # Initialize from Non-CPTP
        kraus_l, kraus_r = [self.UI, self.UX], [self.UY, self.UZ]
        chan = Kraus((kraus_l, kraus_r))
        assert_allclose(chan.data, (kraus_l, kraus_r))
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        # Initialize with redundant second op
        chan = Kraus((kraus_l, kraus_l))
        assert_allclose(chan.data, kraus_l)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        # Initialize from rectangular
        kraus = [np.zeros((4, 2))]
        chan = Kraus(kraus)
        assert_allclose(chan.data, kraus)
        self.assertEqual(chan.dim, (2, 4))
        self.assertIsNone(chan.num_qubits)

        # Wrong input or output dims should raise exception
        self.assertRaises(QiskitError, Kraus, kraus, input_dims=4, output_dims=4)

    def test_circuit_init(self):
        """Test initialization from a circuit."""
        circuit, target = self.simple_circuit_no_measure()
        op = Kraus(circuit)
        target = Kraus(target)
        # The given separable circuit should only have a single Kraus operator.
        self.assertEqual(len(op.data), 1)
        self.assertEqual(len(target.data), 1)
        kraus_op = Operator(op.data[0])
        kraus_target = Operator(target.data[0])
        # THe Kraus representation is not unique, but for a single operator, the only gauge freedom
        # is the global phase.
        self.assertTrue(kraus_op.equiv(kraus_target))

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, Kraus, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        kraus = [self.rand_matrix(2, 2) for _ in range(2)]
        self.assertEqual(Kraus(kraus), Kraus(kraus))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(2)
        with self.subTest("Deep copy"):
            orig = Kraus(mat)
            cpy = orig.copy()
            cpy._data[0][0][0, 0] = 0.0
            self.assertFalse(cpy == orig)
        with self.subTest("Shallow copy"):
            orig = Kraus(mat)
            clone = copy.copy(orig)
            clone._data[0][0][0, 0] = 0.0
            self.assertTrue(clone == orig)

    def test_clone(self):
        """Test clone method"""
        mat = np.eye(4)
        orig = Kraus(mat)
        clone = copy.copy(orig)
        clone._data[0][0][0, 0] = 0.0
        self.assertTrue(clone == orig)

    def test_is_cptp(self):
        """Test is_cptp method."""
        self.assertTrue(Kraus(self.depol_kraus(0.5)).is_cptp())
        self.assertTrue(Kraus(self.UX).is_cptp())
        # Non-CPTP should return false
        self.assertFalse(Kraus(([self.UI], [self.UX])).is_cptp())
        self.assertFalse(Kraus([self.UI, self.UX]).is_cptp())

    def test_conjugate(self):
        """Test conjugate method."""
        kraus_l, kraus_r = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)
        # Single Kraus list
        targ = Kraus([np.conjugate(k) for k in kraus_l])
        chan1 = Kraus(kraus_l)
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (2, 4))
        # Double Kraus list
        targ = Kraus(([np.conjugate(k) for k in kraus_l], [np.conjugate(k) for k in kraus_r]))
        chan1 = Kraus((kraus_l, kraus_r))
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (2, 4))

    def test_transpose(self):
        """Test transpose method."""
        kraus_l, kraus_r = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)
        # Single Kraus list
        targ = Kraus([np.transpose(k) for k in kraus_l])
        chan1 = Kraus(kraus_l)
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))
        # Double Kraus list
        targ = Kraus(([np.transpose(k) for k in kraus_l], [np.transpose(k) for k in kraus_r]))
        chan1 = Kraus((kraus_l, kraus_r))
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))

    def test_adjoint(self):
        """Test adjoint method."""
        kraus_l, kraus_r = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)
        # Single Kraus list
        targ = Kraus([np.transpose(k).conj() for k in kraus_l])
        chan1 = Kraus(kraus_l)
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))
        # Double Kraus list
        targ = Kraus(
            ([np.transpose(k).conj() for k in kraus_l], [np.transpose(k).conj() for k in kraus_r])
        )
        chan1 = Kraus((kraus_l, kraus_r))
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, Kraus(np.eye(2)).compose, Kraus(np.eye(4)))
        self.assertRaises(QiskitError, Kraus(np.eye(2)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = Kraus(self.UX)
        chan2 = Kraus(self.UY)
        chan = chan1.compose(chan2)
        targ = rho & Kraus(self.UZ)
        self.assertEqual(rho & chan, targ)

        # 50% depolarizing channel
        chan1 = Kraus(self.depol_kraus(0.5))
        chan = chan1.compose(chan1)
        targ = rho & Kraus(self.depol_kraus(0.75))
        self.assertEqual(rho & chan, targ)

        # Compose different dimensions
        kraus1, kraus2 = self.rand_kraus(2, 4, 4), self.rand_kraus(4, 2, 4)
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = rho & chan1 & chan2
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho & chan, targ)
        chan = chan1 & chan2
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho & chan, targ)

    def test_dot(self):
        """Test dot method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = Kraus(self.UX)
        chan2 = Kraus(self.UY)
        targ = rho.evolve(Kraus(self.UZ))
        self.assertEqual(rho.evolve(chan1.dot(chan2)), targ)
        self.assertEqual(rho.evolve(chan1 @ chan2), targ)

        # 50% depolarizing channel
        chan1 = Kraus(self.depol_kraus(0.5))
        targ = rho & Kraus(self.depol_kraus(0.75))
        self.assertEqual(rho.evolve(chan1.dot(chan1)), targ)
        self.assertEqual(rho.evolve(chan1 @ chan1), targ)

        # Compose different dimensions
        kraus1, kraus2 = self.rand_kraus(2, 4, 4), self.rand_kraus(4, 2, 4)
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = rho & chan1 & chan2
        self.assertEqual(rho.evolve(chan2.dot(chan1)), targ)
        self.assertEqual(rho.evolve(chan2 @ chan1), targ)

    def test_compose_front(self):
        """Test deprecated front compose method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = Kraus(self.UX)
        chan2 = Kraus(self.UY)
        chan = chan1.compose(chan2, front=True)
        targ = rho & Kraus(self.UZ)
        self.assertEqual(rho & chan, targ)

        # 50% depolarizing channel
        chan1 = Kraus(self.depol_kraus(0.5))
        chan = chan1.compose(chan1, front=True)
        targ = rho & Kraus(self.depol_kraus(0.75))
        self.assertEqual(rho & chan, targ)

        # Compose different dimensions
        kraus1, kraus2 = self.rand_kraus(2, 4, 4), self.rand_kraus(4, 2, 4)
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = rho & chan1 & chan2
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho & chan, targ)

    def test_expand(self):
        """Test expand method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Kraus(self.UI)
        chan2 = Kraus(self.UX)

        # X \otimes I
        chan = chan1.expand(chan2)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)

        # I \otimes X
        chan = chan2.expand(chan1)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)

        # Completely depolarizing
        chan_dep = Kraus(self.depol_kraus(1))
        chan = chan_dep.expand(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)

    def test_tensor(self):
        """Test tensor method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Kraus(self.UI)
        chan2 = Kraus(self.UX)

        # X \otimes I
        chan = chan2.tensor(chan1)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)

        # I \otimes X
        chan = chan1.tensor(chan2)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)

        # Completely depolarizing
        chan_dep = Kraus(self.depol_kraus(1))
        chan = chan_dep.tensor(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init & chan, rho_targ)

    def test_power(self):
        """Test power method."""
        # 10% depolarizing channel
        rho = DensityMatrix(np.diag([1, 0]))
        p_id = 0.9
        chan = Kraus(self.depol_kraus(1 - p_id))

        # Compose 3 times
        p_id3 = p_id**3
        chan3 = chan.power(3)
        targ3a = rho & chan & chan & chan
        self.assertEqual(rho & chan3, targ3a)
        targ3b = rho & Kraus(self.depol_kraus(1 - p_id3))
        self.assertEqual(rho & chan3, targ3b)

    def test_add(self):
        """Test add method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))
        kraus1, kraus2 = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)

        # Random Single-Kraus maps
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = (rho & chan1) + (rho & chan2)
        chan = chan1._add(chan2)
        self.assertEqual(rho & chan, targ)
        chan = chan1 + chan2
        self.assertEqual(rho & chan, targ)

        # Random Single-Kraus maps
        chan = Kraus((kraus1, kraus2))
        targ = 2 * (rho & chan)
        chan = chan._add(chan)
        self.assertEqual(rho & chan, targ)

    def test_add_qargs(self):
        """Test add method with qargs."""
        rho = DensityMatrix(self.rand_rho(8))
        kraus = self.rand_kraus(8, 8, 4)
        kraus0 = self.rand_kraus(2, 2, 4)

        op = Kraus(kraus)
        op0 = Kraus(kraus0)
        eye = Kraus(self.UI)

        with self.subTest(msg="qargs=[0]"):
            value = op + op0([0])
            target = op + eye.tensor(eye).tensor(op0)
            self.assertEqual(rho & value, rho & target)

        with self.subTest(msg="qargs=[1]"):
            value = op + op0([1])
            target = op + eye.tensor(op0).tensor(eye)
            self.assertEqual(rho & value, rho & target)

        with self.subTest(msg="qargs=[2]"):
            value = op + op0([2])
            target = op + op0.tensor(eye).tensor(eye)
            self.assertEqual(rho & value, rho & target)

    def test_sub_qargs(self):
        """Test sub method with qargs."""
        rho = DensityMatrix(self.rand_rho(8))
        kraus = self.rand_kraus(8, 8, 4)
        kraus0 = self.rand_kraus(2, 2, 4)

        op = Kraus(kraus)
        op0 = Kraus(kraus0)
        eye = Kraus(self.UI)

        with self.subTest(msg="qargs=[0]"):
            value = op - op0([0])
            target = op - eye.tensor(eye).tensor(op0)
            self.assertEqual(rho & value, rho & target)

        with self.subTest(msg="qargs=[1]"):
            value = op - op0([1])
            target = op - eye.tensor(op0).tensor(eye)
            self.assertEqual(rho & value, rho & target)

        with self.subTest(msg="qargs=[2]"):
            value = op - op0([2])
            target = op - op0.tensor(eye).tensor(eye)
            self.assertEqual(rho & value, rho & target)

    def test_subtract(self):
        """Test subtract method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))
        kraus1, kraus2 = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)

        # Random Single-Kraus maps
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = (rho & chan1) - (rho & chan2)
        chan = chan1 - chan2
        self.assertEqual(rho & chan, targ)

        # Random Single-Kraus maps
        chan = Kraus((kraus1, kraus2))
        targ = 0 * (rho & chan)
        chan = chan - chan
        self.assertEqual(rho & chan, targ)

    def test_multiply(self):
        """Test multiply method."""
        # Random initial state and Kraus ops
        rho = DensityMatrix(self.rand_rho(2))
        val = 0.5
        kraus1, kraus2 = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)

        # Single Kraus set
        chan1 = Kraus(kraus1)
        targ = val * (rho & chan1)
        chan = chan1._multiply(val)
        self.assertEqual(rho & chan, targ)
        chan = val * chan1
        self.assertEqual(rho & chan, targ)
        targ = (rho & chan1) * val
        chan = chan1 * val
        self.assertEqual(rho & chan, targ)

        # Double Kraus set
        chan2 = Kraus((kraus1, kraus2))
        targ = val * (rho & chan2)
        chan = chan2._multiply(val)
        self.assertEqual(rho & chan, targ)
        chan = val * chan2
        self.assertEqual(rho & chan, targ)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = Kraus(self.depol_kraus(1))
        self.assertRaises(QiskitError, chan._multiply, "s")
        self.assertRaises(QiskitError, chan.__rmul__, "s")
        self.assertRaises(QiskitError, chan._multiply, chan)
        self.assertRaises(QiskitError, chan.__rmul__, chan)

    def test_negate(self):
        """Test negate method"""
        rho = DensityMatrix(np.diag([1, 0]))
        targ = DensityMatrix(np.diag([-0.5, -0.5]))
        chan = -Kraus(self.depol_kraus(1))
        self.assertEqual(rho & chan, targ)


if __name__ == "__main__":
    unittest.main()
