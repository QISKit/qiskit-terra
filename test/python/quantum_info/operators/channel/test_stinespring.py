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

"""Tests for Stinespring quantum channel representation class."""

import copy
import unittest
import numpy as np
from numpy.testing import assert_allclose

from qiskit import QiskitError
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info import Stinespring, Operator
from .channel_test_case import ChannelTestCase


class TestStinespring(ChannelTestCase):
    """Tests for Stinespring channel representation."""

    def test_init(self):
        """Test initialization"""
        # Initialize from unitary
        chan = Stinespring(self.UI)
        assert_allclose(chan.data, self.UI)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        # Initialize from Stinespring
        chan = Stinespring(self.depol_stine(0.5))
        assert_allclose(chan.data, self.depol_stine(0.5))
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        # Initialize from Non-CPTP
        stine_l, stine_r = self.rand_matrix(4, 2), self.rand_matrix(4, 2)
        chan = Stinespring((stine_l, stine_r))
        assert_allclose(chan.data, (stine_l, stine_r))
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        # Initialize with redundant second op
        chan = Stinespring((stine_l, stine_l))
        assert_allclose(chan.data, stine_l)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(chan.num_qubits, 1)

        # Wrong input or output dims should raise exception
        self.assertRaises(QiskitError, Stinespring, stine_l, input_dims=4, output_dims=4)

    def test_circuit_init(self):
        """Test initialization from a circuit."""
        circuit, target = self.simple_circuit_no_measure()
        op = Stinespring(circuit)
        target = Stinespring(target)
        # If the Stinespring is CPTP (and it should be), it's defined in terms of a single
        # rectangular operator, which has global-phase gauge freedom.
        self.assertTrue(op.is_cptp())
        self.assertTrue(Operator(op.data).equiv(Operator(target.data)))

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, Stinespring, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        stine = tuple(self.rand_matrix(4, 2) for _ in range(2))
        self.assertEqual(Stinespring(stine), Stinespring(stine))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(4)
        with self.subTest("Deep copy"):
            orig = Stinespring(mat)
            cpy = orig.copy()
            cpy._data[0][0, 0] = 0.0
            self.assertFalse(cpy == orig)
        with self.subTest("Shallow copy"):
            orig = Stinespring(mat)
            clone = copy.copy(orig)
            clone._data[0][0, 0] = 0.0
            self.assertTrue(clone == orig)

    def test_clone(self):
        """Test clone method"""
        mat = np.eye(4)
        orig = Stinespring(mat)
        clone = copy.copy(orig)
        clone._data[0][0, 0] = 0.0
        self.assertTrue(clone == orig)

    def test_is_cptp(self):
        """Test is_cptp method."""
        self.assertTrue(Stinespring(self.depol_stine(0.5)).is_cptp())
        self.assertTrue(Stinespring(self.UX).is_cptp())
        # Non-CP
        stine_l, stine_r = self.rand_matrix(4, 2), self.rand_matrix(4, 2)
        self.assertFalse(Stinespring((stine_l, stine_r)).is_cptp())
        self.assertFalse(Stinespring(self.UI + self.UX).is_cptp())

    def test_is_cptp_honors_tolerances(self):
        """Test that is_cptp uses caller-provided tolerance values."""
        channel = Stinespring((1 + 4e-4) * np.eye(2))

        self.assertFalse(channel.is_cptp(atol=1e-5, rtol=0))
        self.assertTrue(channel.is_cptp(atol=1e-3, rtol=0))

    def test_conjugate(self):
        """Test conjugate method."""
        stine_l, stine_r = self.rand_matrix(16, 2), self.rand_matrix(16, 2)
        # Single Stinespring list
        targ = Stinespring(stine_l.conj(), output_dims=4)
        chan1 = Stinespring(stine_l, output_dims=4)
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (2, 4))
        # Double Stinespring list
        targ = Stinespring((stine_l.conj(), stine_r.conj()), output_dims=4)
        chan1 = Stinespring((stine_l, stine_r), output_dims=4)
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (2, 4))

    def test_transpose(self):
        """Test transpose method."""
        stine_l, stine_r = self.rand_matrix(4, 2), self.rand_matrix(4, 2)
        # Single square Stinespring list
        targ = Stinespring(stine_l.T, 4, 2)
        chan1 = Stinespring(stine_l, 2, 4)
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))
        # Double square Stinespring list
        targ = Stinespring((stine_l.T, stine_r.T), 4, 2)
        chan1 = Stinespring((stine_l, stine_r), 2, 4)
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))

    def test_adjoint(self):
        """Test adjoint method."""
        stine_l, stine_r = self.rand_matrix(4, 2), self.rand_matrix(4, 2)
        # Single square Stinespring list
        targ = Stinespring(stine_l.T.conj(), 4, 2)
        chan1 = Stinespring(stine_l, 2, 4)
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))
        # Double square Stinespring list
        targ = Stinespring((stine_l.T.conj(), stine_r.T.conj()), 4, 2)
        chan1 = Stinespring((stine_l, stine_r), 2, 4)
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, Stinespring(np.eye(2)).compose, Stinespring(np.eye(4)))
        self.assertRaises(QiskitError, Stinespring(np.eye(2)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        # Random input test state
        rho_init = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = Stinespring(self.UX)
        chan2 = Stinespring(self.UY)
        chan = chan1.compose(chan2)
        rho_targ = rho_init & Stinespring(self.UZ)
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # 50% depolarizing channel
        chan1 = Stinespring(self.depol_stine(0.5))
        chan = chan1.compose(chan1)
        rho_targ = rho_init & Stinespring(self.depol_stine(0.75))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # Compose different dimensions
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(8, 4)
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        chan2 = Stinespring(stine2, input_dims=4, output_dims=2)
        rho_targ = rho_init & chan1 & chan2
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan1 & chan2
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_dot(self):
        """Test deprecated front compose method."""
        # Random input test state
        rho_init = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = Stinespring(self.UX)
        chan2 = Stinespring(self.UY)
        rho_targ = rho_init.evolve(Stinespring(self.UZ))
        self.assertEqual(rho_init.evolve(chan1.dot(chan2)), rho_targ)
        self.assertEqual(rho_init.evolve(chan1 @ chan2), rho_targ)

        # 50% depolarizing channel
        chan1 = Stinespring(self.depol_stine(0.5))
        rho_targ = rho_init & Stinespring(self.depol_stine(0.75))
        self.assertEqual(rho_init.evolve(chan1.dot(chan1)), rho_targ)
        self.assertEqual(rho_init.evolve(chan1 @ chan1), rho_targ)

        # Compose different dimensions
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(8, 4)
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        chan2 = Stinespring(stine2, input_dims=4, output_dims=2)
        rho_targ = rho_init & chan1 & chan2
        self.assertEqual(rho_init.evolve(chan2.dot(chan1)), rho_targ)
        self.assertEqual(rho_init.evolve(chan2 @ chan1), rho_targ)

    def test_compose_front(self):
        """Test deprecated front compose method."""
        # Random input test state
        rho_init = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = Stinespring(self.UX)
        chan2 = Stinespring(self.UY)
        chan = chan1.compose(chan2, front=True)
        rho_targ = rho_init & Stinespring(self.UZ)
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # 50% depolarizing channel
        chan1 = Stinespring(self.depol_stine(0.5))
        chan = chan1.compose(chan1, front=True)
        rho_targ = rho_init & Stinespring(self.depol_stine(0.75))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # Compose different dimensions
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(8, 4)
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        chan2 = Stinespring(stine2, input_dims=4, output_dims=2)
        rho_targ = rho_init & chan1 & chan2
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_expand(self):
        """Test expand method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Stinespring(self.UI)
        chan2 = Stinespring(self.UX)

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
        chan_dep = Stinespring(self.depol_stine(1))
        chan = chan_dep.expand(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_tensor(self):
        """Test tensor method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Stinespring(self.UI)
        chan2 = Stinespring(self.UX)

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
        chan_dep = Stinespring(self.depol_stine(1))
        chan = chan_dep.tensor(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_power(self):
        """Test power method."""
        # 10% depolarizing channel
        rho_init = DensityMatrix(np.diag([1, 0]))
        p_id = 0.9
        chan1 = Stinespring(self.depol_stine(1 - p_id))

        # Compose 3 times
        p_id3 = p_id**3
        chan = chan1.power(3)
        rho_targ = rho_init & chan1 & chan1 & chan1
        self.assertEqual(rho_init & chan, rho_targ)
        rho_targ = rho_init & Stinespring(self.depol_stine(1 - p_id3))
        self.assertEqual(rho_init & chan, rho_targ)

    def test_add(self):
        """Test add method."""
        # Random input test state
        rho_init = DensityMatrix(self.rand_rho(2))
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(16, 2)

        # Random Single-Stinespring maps
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        chan2 = Stinespring(stine2, input_dims=2, output_dims=4)
        rho_targ = (rho_init & chan1) + (rho_init & chan2)
        chan = chan1._add(chan2)
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = chan1 + chan2
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # Random Single-Stinespring maps
        chan = Stinespring((stine1, stine2))
        rho_targ = 2 * (rho_init & chan)
        chan = chan._add(chan)
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_subtract(self):
        """Test subtract method."""
        # Random input test state
        rho_init = DensityMatrix(self.rand_rho(2))
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(16, 2)

        # Random Single-Stinespring maps
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        chan2 = Stinespring(stine2, input_dims=2, output_dims=4)
        rho_targ = (rho_init & chan1) - (rho_init & chan2)
        chan = chan1 - chan2
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # Random Single-Stinespring maps
        chan = Stinespring((stine1, stine2))
        rho_targ = 0 * (rho_init & chan)
        chan = chan - chan
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_add_qargs(self):
        """Test add method with qargs."""
        rho = DensityMatrix(self.rand_rho(8))
        stine = self.rand_matrix(32, 8)
        stine0 = self.rand_matrix(8, 2)

        op = Stinespring(stine)
        op0 = Stinespring(stine0)
        eye = Stinespring(self.UI)

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
        stine = self.rand_matrix(32, 8)
        stine0 = self.rand_matrix(8, 2)

        op = Stinespring(stine)
        op0 = Stinespring(stine0)
        eye = Stinespring(self.UI)

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

    def test_multiply(self):
        """Test multiply method."""
        # Random initial state and Stinespring ops
        rho_init = DensityMatrix(self.rand_rho(2))
        val = 0.5
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(16, 2)

        # Single Stinespring set
        chan1 = Stinespring(stine1, input_dims=2, output_dims=4)
        rho_targ = val * (rho_init & chan1)
        chan = chan1._multiply(val)
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = val * chan1
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        rho_targ = (rho_init & chan1) * val
        chan = chan1 * val
        self.assertEqual(rho_init.evolve(chan), rho_targ)

        # Double Stinespring set
        chan2 = Stinespring((stine1, stine2), input_dims=2, output_dims=4)
        rho_targ = val * (rho_init & chan2)
        chan = chan2._multiply(val)
        self.assertEqual(rho_init.evolve(chan), rho_targ)
        chan = val * chan2
        self.assertEqual(rho_init.evolve(chan), rho_targ)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = Stinespring(self.depol_stine(1))
        self.assertRaises(QiskitError, chan._multiply, "s")
        self.assertRaises(QiskitError, chan.__rmul__, "s")
        self.assertRaises(QiskitError, chan._multiply, chan)
        self.assertRaises(QiskitError, chan.__rmul__, chan)

    def test_negate(self):
        """Test negate method"""
        rho_init = DensityMatrix(np.diag([1, 0]))
        rho_targ = DensityMatrix(np.diag([-0.5, -0.5]))
        chan = -Stinespring(self.depol_stine(1))
        self.assertEqual(rho_init.evolve(chan), rho_targ)


if __name__ == "__main__":
    unittest.main()
