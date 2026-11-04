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


"""Tests for quantum channel representation transformations."""

import unittest

import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.channel.stinespring import Stinespring
from qiskit.quantum_info.operators.channel.ptm import PTM
from qiskit.quantum_info.operators.channel.chi import Chi
from .channel_test_case import ChannelTestCase


class TestTransformations(ChannelTestCase):
    """Tests for Operator channel representation."""

    unitary_mat = [
        ChannelTestCase.UI,
        ChannelTestCase.UX,
        ChannelTestCase.UY,
        ChannelTestCase.UZ,
        ChannelTestCase.UH,
    ]
    unitary_choi = [
        ChannelTestCase.choiI,
        ChannelTestCase.choiX,
        ChannelTestCase.choiY,
        ChannelTestCase.choiZ,
        ChannelTestCase.choiH,
    ]
    unitary_chi = [
        ChannelTestCase.chiI,
        ChannelTestCase.chiX,
        ChannelTestCase.chiY,
        ChannelTestCase.chiZ,
        ChannelTestCase.chiH,
    ]
    unitary_sop = [
        ChannelTestCase.sopI,
        ChannelTestCase.sopX,
        ChannelTestCase.sopY,
        ChannelTestCase.sopZ,
        ChannelTestCase.sopH,
    ]
    unitary_ptm = [
        ChannelTestCase.ptmI,
        ChannelTestCase.ptmX,
        ChannelTestCase.ptmY,
        ChannelTestCase.ptmZ,
        ChannelTestCase.ptmH,
    ]

    def test_operator_to_operator(self):
        """Test Operator to Operator transformation."""
        # Test unitary channels
        for mat in self.unitary_mat:
            chan1 = Operator(mat)
            chan2 = Operator(chan1)
            self.assertEqual(chan1, chan2)

    def test_operator_to_choi(self):
        """Test Operator to Choi transformation."""
        # Test unitary channels
        for mat, choi in zip(self.unitary_mat, self.unitary_choi):
            chan1 = Choi(choi)
            chan2 = Choi(Operator(mat))
            self.assertEqual(chan1, chan2)

    def test_operator_to_superop(self):
        """Test Operator to SuperOp transformation."""
        # Test unitary channels
        for mat, sop in zip(self.unitary_mat, self.unitary_sop):
            chan1 = SuperOp(sop)
            chan2 = SuperOp(Operator(mat))
            self.assertEqual(chan1, chan2)

    def test_operator_to_kraus(self):
        """Test Operator to Kraus transformation."""
        # Test unitary channels
        for mat in self.unitary_mat:
            chan1 = Kraus(mat)
            chan2 = Kraus(Operator(mat))
            self.assertEqual(chan1, chan2)

    def test_operator_to_stinespring(self):
        """Test Operator to Stinespring transformation."""
        # Test unitary channels
        for mat in self.unitary_mat:
            chan1 = Stinespring(mat)
            chan2 = Stinespring(Operator(chan1))
            self.assertEqual(chan1, chan2)

    def test_operator_to_chi(self):
        """Test Operator to Chi transformation."""
        # Test unitary channels
        for mat, chi in zip(self.unitary_mat, self.unitary_chi):
            chan1 = Chi(chi)
            chan2 = Chi(Operator(mat))
            self.assertEqual(chan1, chan2)

    def test_operator_to_ptm(self):
        """Test Operator to PTM transformation."""
        # Test unitary channels
        for mat, ptm in zip(self.unitary_mat, self.unitary_ptm):
            chan1 = PTM(ptm)
            chan2 = PTM(Operator(mat))
            self.assertEqual(chan1, chan2)

    def test_choi_to_operator(self):
        """Test Choi to Operator transformation."""
        # Test unitary channels
        for mat, choi in zip(self.unitary_mat, self.unitary_choi):
            chan1 = Operator(mat)
            chan2 = Operator(Choi(choi))
            self.assertTrue(matrix_equal(chan2.data, chan1.data, ignore_phase=True))

    def test_choi_to_choi(self):
        """Test Choi to Choi transformation."""
        # Test unitary channels
        for choi in self.unitary_choi:
            chan1 = Choi(choi)
            chan2 = Choi(chan1)
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Choi(self.depol_choi(p))
            chan2 = Choi(chan1)
            self.assertEqual(chan1, chan2)

    def test_choi_to_superop(self):
        """Test Choi to SuperOp transformation."""
        # Test unitary channels
        for choi, sop in zip(self.unitary_choi, self.unitary_sop):
            chan1 = SuperOp(sop)
            chan2 = SuperOp(Choi(choi))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = SuperOp(self.depol_sop(p))
            chan2 = SuperOp(Choi(self.depol_choi(p)))
            self.assertEqual(chan1, chan2)

    def test_choi_to_kraus(self):
        """Test Choi to Kraus transformation."""
        # Test unitary channels
        for mat, choi in zip(self.unitary_mat, self.unitary_choi):
            chan1 = Kraus(mat)
            chan2 = Kraus(Choi(choi))
            self.assertTrue(matrix_equal(chan2.data[0], chan1.data[0], ignore_phase=True))
        # Test depolarizing channels
        rho = DensityMatrix(np.diag([1, 0]))
        for p in [0.25, 0.5, 0.75, 1]:
            target = rho.evolve(Kraus(self.depol_kraus(p)))
            output = rho.evolve(Kraus(Choi(self.depol_choi(p))))
            self.assertEqual(output, target)

    def test_choi_to_kraus_does_not_mutate_input(self):
        """Converting to Kraus should not modify the input Choi data."""
        choi = Choi(self.choiX)
        original = choi.data.copy()

        _ = Kraus(choi)

        np.testing.assert_array_equal(choi.data, original)

    def test_choi_to_stinespring(self):
        """Test Choi to Stinespring transformation."""
        # Test unitary channels
        for mat, choi in zip(self.unitary_mat, self.unitary_choi):
            chan1 = Kraus(mat)
            chan2 = Kraus(Choi(choi))
            self.assertTrue(matrix_equal(chan2.data[0], chan1.data[0], ignore_phase=True))
        # Test depolarizing channels
        rho = DensityMatrix(np.diag([1, 0]))
        for p in [0.25, 0.5, 0.75, 1]:
            target = rho.evolve(Stinespring(self.depol_stine(p)))
            output = rho.evolve(Stinespring(Choi(self.depol_choi(p))))
            self.assertEqual(output, target)

    def test_choi_to_chi(self):
        """Test Choi to Chi transformation."""
        # Test unitary channels
        for choi, chi in zip(self.unitary_choi, self.unitary_chi):
            chan1 = Chi(chi)
            chan2 = Chi(Choi(choi))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Chi(self.depol_chi(p))
            chan2 = Chi(Choi(self.depol_choi(p)))
            self.assertEqual(chan1, chan2)

    def test_choi_to_ptm(self):
        """Test Choi to PTM transformation."""
        # Test unitary channels
        for choi, ptm in zip(self.unitary_choi, self.unitary_ptm):
            chan1 = PTM(ptm)
            chan2 = PTM(Choi(choi))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = PTM(self.depol_ptm(p))
            chan2 = PTM(Choi(self.depol_choi(p)))
            self.assertEqual(chan1, chan2)

    def test_superop_to_operator(self):
        """Test SuperOp to Operator transformation."""
        for mat, sop in zip(self.unitary_mat, self.unitary_sop):
            chan1 = Operator(mat)
            chan2 = Operator(SuperOp(sop))
            self.assertTrue(matrix_equal(chan2.data, chan1.data, ignore_phase=True))
        self.assertRaises(QiskitError, Operator, SuperOp(self.depol_sop(0.5)))

    def test_superop_to_choi(self):
        """Test SuperOp to Choi transformation."""
        # Test unitary channels
        for choi, sop in zip(self.unitary_choi, self.unitary_sop):
            chan1 = Choi(choi)
            chan2 = Choi(SuperOp(sop))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0, 0.25, 0.5, 0.75, 1]:
            chan1 = Choi(self.depol_choi(p))
            chan2 = Choi(SuperOp(self.depol_sop(p)))
            self.assertEqual(chan1, chan2)

    def test_superop_to_superop(self):
        """Test SuperOp to SuperOp transformation."""
        # Test unitary channels
        for sop in self.unitary_sop:
            chan1 = SuperOp(sop)
            chan2 = SuperOp(chan1)
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0, 0.25, 0.5, 0.75, 1]:
            chan1 = SuperOp(self.depol_sop(p))
            chan2 = SuperOp(chan1)
            self.assertEqual(chan1, chan2)

    def test_superop_to_kraus(self):
        """Test SuperOp to Kraus transformation."""
        # Test unitary channels
        for mat, sop in zip(self.unitary_mat, self.unitary_sop):
            chan1 = Kraus(mat)
            chan2 = Kraus(SuperOp(sop))
            self.assertTrue(matrix_equal(chan2.data[0], chan1.data[0], ignore_phase=True))
        # Test depolarizing channels
        rho = DensityMatrix(np.diag([1, 0]))
        for p in [0.25, 0.5, 0.75, 1]:
            target = rho.evolve(Kraus(self.depol_kraus(p)))
            output = rho.evolve(Kraus(SuperOp(self.depol_sop(p))))
            self.assertEqual(output, target)

    def test_superop_to_stinespring(self):
        """Test SuperOp to Stinespring transformation."""
        # Test unitary channels
        for mat, sop in zip(self.unitary_mat, self.unitary_sop):
            chan1 = Stinespring(mat)
            chan2 = Stinespring(SuperOp(sop))
            self.assertTrue(matrix_equal(chan2.data[0], chan1.data[0], ignore_phase=True))
        # Test depolarizing channels
        rho = DensityMatrix(np.diag([1, 0]))
        for p in [0.25, 0.5, 0.75, 1]:
            target = rho.evolve(Stinespring(self.depol_stine(p)))
            output = rho.evolve(Stinespring(SuperOp(self.depol_sop(p))))
            self.assertEqual(output, target)

    def test_superop_to_chi(self):
        """Test SuperOp to Chi transformation."""
        # Test unitary channels
        for sop, ptm in zip(self.unitary_sop, self.unitary_ptm):
            chan1 = PTM(ptm)
            chan2 = PTM(SuperOp(sop))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Chi(self.depol_chi(p))
            chan2 = Chi(SuperOp(self.depol_sop(p)))
            self.assertEqual(chan1, chan2)

    def test_superop_to_ptm(self):
        """Test SuperOp to PTM transformation."""
        # Test unitary channels
        for sop, ptm in zip(self.unitary_sop, self.unitary_ptm):
            chan1 = PTM(ptm)
            chan2 = PTM(SuperOp(sop))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = PTM(self.depol_ptm(p))
            chan2 = PTM(SuperOp(self.depol_sop(p)))
            self.assertEqual(chan1, chan2)

    def test_kraus_to_operator(self):
        """Test Kraus to Operator transformation."""
        for mat in self.unitary_mat:
            chan1 = Operator(mat)
            chan2 = Operator(Kraus(mat))
            self.assertTrue(matrix_equal(chan2.data, chan1.data, ignore_phase=True))
        self.assertRaises(QiskitError, Operator, Kraus(self.depol_kraus(0.5)))

    def test_kraus_to_choi(self):
        """Test Kraus to Choi transformation."""
        # Test unitary channels
        for mat, choi in zip(self.unitary_mat, self.unitary_choi):
            chan1 = Choi(choi)
            chan2 = Choi(Kraus(mat))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Choi(self.depol_choi(p))
            chan2 = Choi(Kraus(self.depol_kraus(p)))
            self.assertEqual(chan1, chan2)

    def test_kraus_to_superop(self):
        """Test Kraus to SuperOp transformation."""
        # Test unitary channels
        for mat, sop in zip(self.unitary_mat, self.unitary_sop):
            chan1 = SuperOp(sop)
            chan2 = SuperOp(Kraus(mat))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = SuperOp(self.depol_sop(p))
            chan2 = SuperOp(Kraus(self.depol_kraus(p)))
            self.assertEqual(chan1, chan2)

    def test_kraus_to_kraus(self):
        """Test Kraus to Kraus transformation."""
        # Test unitary channels
        for mat in self.unitary_mat:
            chan1 = Kraus(mat)
            chan2 = Kraus(chan1)
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Kraus(self.depol_kraus(p))
            chan2 = Kraus(chan1)
            self.assertEqual(chan1, chan2)

    def test_kraus_to_stinespring(self):
        """Test Kraus to Stinespring transformation."""
        # Test unitary channels
        for mat in self.unitary_mat:
            chan1 = Stinespring(mat)
            chan2 = Stinespring(Kraus(mat))
            self.assertTrue(matrix_equal(chan2.data[0], chan1.data[0], ignore_phase=True))
        # Test depolarizing channels
        rho = DensityMatrix(np.diag([1, 0]))
        for p in [0.25, 0.5, 0.75, 1]:
            target = rho.evolve(Stinespring(self.depol_stine(p)))
            output = rho.evolve(Stinespring(Kraus(self.depol_kraus(p))))
            self.assertEqual(output, target)

    def test_kraus_to_chi(self):
        """Test Kraus to Chi transformation."""
        # Test unitary channels
        for mat, chi in zip(self.unitary_mat, self.unitary_chi):
            chan1 = Chi(chi)
            chan2 = Chi(Kraus(mat))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Chi(self.depol_chi(p))
            chan2 = Chi(Kraus(self.depol_kraus(p)))
            self.assertEqual(chan1, chan2)

    def test_kraus_to_ptm(self):
        """Test Kraus to PTM transformation."""
        # Test unitary channels
        for mat, ptm in zip(self.unitary_mat, self.unitary_ptm):
            chan1 = PTM(ptm)
            chan2 = PTM(Kraus(mat))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = PTM(self.depol_ptm(p))
            chan2 = PTM(Kraus(self.depol_kraus(p)))
            self.assertEqual(chan1, chan2)

    def test_stinespring_to_operator(self):
        """Test Stinespring to Operator transformation."""
        for mat in self.unitary_mat:
            chan1 = Operator(mat)
            chan2 = Operator(Stinespring(mat))
            self.assertTrue(matrix_equal(chan2.data, chan1.data, ignore_phase=True))
        self.assertRaises(QiskitError, Operator, Stinespring(self.depol_stine(0.5)))

    def test_stinespring_to_choi(self):
        """Test Stinespring to Choi transformation."""
        # Test unitary channels
        for mat, choi in zip(self.unitary_mat, self.unitary_choi):
            chan1 = Choi(choi)
            chan2 = Choi(Stinespring(mat))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Choi(self.depol_choi(p))
            chan2 = Choi(Stinespring(self.depol_stine(p)))
            self.assertEqual(chan1, chan2)

    def test_stinespring_to_superop(self):
        """Test Stinespring to SuperOp transformation."""
        # Test unitary channels
        for mat, sop in zip(self.unitary_mat, self.unitary_sop):
            chan1 = SuperOp(sop)
            chan2 = SuperOp(Kraus(mat))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = SuperOp(self.depol_sop(p))
            chan2 = SuperOp(Stinespring(self.depol_stine(p)))
            self.assertEqual(chan1, chan2)

    def test_stinespring_to_kraus(self):
        """Test Stinespring to Kraus transformation."""
        # Test unitary channels
        for mat in self.unitary_mat:
            chan1 = Kraus(mat)
            chan2 = Kraus(Stinespring(mat))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Kraus(self.depol_kraus(p))
            chan2 = Kraus(Stinespring(self.depol_stine(p)))
            self.assertEqual(chan1, chan2)

    def test_stinespring_to_stinespring(self):
        """Test Stinespring to Stinespring transformation."""
        # Test unitary channels
        for mat in self.unitary_mat:
            chan1 = Stinespring(mat)
            chan2 = Stinespring(chan1)
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Stinespring(self.depol_stine(p))
            chan2 = Stinespring(chan1)
            self.assertEqual(chan1, chan2)

    def test_stinespring_to_chi(self):
        """Test Stinespring to Chi transformation."""
        # Test unitary channels
        for mat, chi in zip(self.unitary_mat, self.unitary_chi):
            chan1 = Chi(chi)
            chan2 = Chi(Stinespring(mat))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Chi(self.depol_chi(p))
            chan2 = Chi(Stinespring(self.depol_stine(p)))
            self.assertEqual(chan1, chan2)

    def test_stinespring_to_ptm(self):
        """Test Stinespring to PTM transformation."""
        # Test unitary channels
        for mat, ptm in zip(self.unitary_mat, self.unitary_ptm):
            chan1 = PTM(ptm)
            chan2 = PTM(Stinespring(mat))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = PTM(self.depol_ptm(p))
            chan2 = PTM(Stinespring(self.depol_stine(p)))
            self.assertEqual(chan1, chan2)

    def test_chi_to_operator(self):
        """Test Chi to Operator transformation."""
        for mat, chi in zip(self.unitary_mat, self.unitary_chi):
            chan1 = Operator(mat)
            chan2 = Operator(Chi(chi))
            self.assertTrue(matrix_equal(chan2.data, chan1.data, ignore_phase=True))
        self.assertRaises(QiskitError, Operator, Chi(self.depol_chi(0.5)))

    def test_chi_to_choi(self):
        """Test Chi to Choi transformation."""
        # Test unitary channels
        for chi, choi in zip(self.unitary_chi, self.unitary_choi):
            chan1 = Choi(choi)
            chan2 = Choi(Chi(chi))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Choi(self.depol_choi(p))
            chan2 = Choi(Chi(self.depol_chi(p)))
            self.assertEqual(chan1, chan2)

    def test_chi_to_superop(self):
        """Test Chi to SuperOp transformation."""
        # Test unitary channels
        for chi, sop in zip(self.unitary_chi, self.unitary_sop):
            chan1 = SuperOp(sop)
            chan2 = SuperOp(Chi(chi))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = SuperOp(self.depol_sop(p))
            chan2 = SuperOp(Chi(self.depol_chi(p)))
            self.assertEqual(chan1, chan2)

    def test_chi_to_kraus(self):
        """Test Chi to Kraus transformation."""
        # Test unitary channels
        for mat, chi in zip(self.unitary_mat, self.unitary_chi):
            chan1 = Kraus(mat)
            chan2 = Kraus(Chi(chi))
            self.assertTrue(matrix_equal(chan2.data[0], chan1.data[0], ignore_phase=True))
        # Test depolarizing channels
        rho = DensityMatrix(np.diag([1, 0]))
        for p in [0.25, 0.5, 0.75, 1]:
            target = rho.evolve(Kraus(self.depol_kraus(p)))
            output = rho.evolve(Kraus(Chi(self.depol_chi(p))))
            self.assertEqual(output, target)

    def test_chi_to_stinespring(self):
        """Test Chi to Stinespring transformation."""
        # Test unitary channels
        for mat, chi in zip(self.unitary_mat, self.unitary_chi):
            chan1 = Kraus(mat)
            chan2 = Kraus(Chi(chi))
            self.assertTrue(matrix_equal(chan2.data[0], chan1.data[0], ignore_phase=True))
        # Test depolarizing channels
        rho = DensityMatrix(np.diag([1, 0]))
        for p in [0.25, 0.5, 0.75, 1]:
            target = rho.evolve(Stinespring(self.depol_stine(p)))
            output = rho.evolve(Stinespring(Chi(self.depol_chi(p))))
            self.assertEqual(output, target)

    def test_chi_to_chi(self):
        """Test Chi to Chi transformation."""
        # Test unitary channels
        for chi in self.unitary_chi:
            chan1 = Chi(chi)
            chan2 = Chi(chan1)
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Chi(self.depol_chi(p))
            chan2 = Chi(chan1)
            self.assertEqual(chan1, chan2)

    def test_chi_to_ptm(self):
        """Test Chi to PTM transformation."""
        # Test unitary channels
        for chi, ptm in zip(self.unitary_chi, self.unitary_ptm):
            chan1 = PTM(ptm)
            chan2 = PTM(Chi(chi))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = PTM(self.depol_ptm(p))
            chan2 = PTM(Chi(self.depol_chi(p)))
            self.assertEqual(chan1, chan2)

    def test_ptm_to_operator(self):
        """Test PTM to Operator transformation."""
        for mat, ptm in zip(self.unitary_mat, self.unitary_ptm):
            chan1 = Operator(mat)
            chan2 = Operator(PTM(ptm))
            self.assertTrue(matrix_equal(chan2.data, chan1.data, ignore_phase=True))
        self.assertRaises(QiskitError, Operator, PTM(self.depol_ptm(0.5)))

    def test_ptm_to_choi(self):
        """Test PTM to Choi transformation."""
        # Test unitary channels
        for ptm, choi in zip(self.unitary_ptm, self.unitary_choi):
            chan1 = Choi(choi)
            chan2 = Choi(PTM(ptm))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Choi(self.depol_choi(p))
            chan2 = Choi(PTM(self.depol_ptm(p)))
            self.assertEqual(chan1, chan2)

    def test_ptm_to_superop(self):
        """Test PTM to SuperOp transformation."""
        # Test unitary channels
        for ptm, sop in zip(self.unitary_ptm, self.unitary_sop):
            chan1 = SuperOp(sop)
            chan2 = SuperOp(PTM(ptm))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = SuperOp(self.depol_sop(p))
            chan2 = SuperOp(PTM(self.depol_ptm(p)))
            self.assertEqual(chan1, chan2)

    def test_ptm_to_kraus(self):
        """Test PTM to Kraus transformation."""
        # Test unitary channels
        for mat, ptm in zip(self.unitary_mat, self.unitary_ptm):
            chan1 = Kraus(mat)
            chan2 = Kraus(PTM(ptm))
            self.assertTrue(matrix_equal(chan2.data[0], chan1.data[0], ignore_phase=True))
        # Test depolarizing channels
        rho = DensityMatrix(np.diag([1, 0]))
        for p in [0.25, 0.5, 0.75, 1]:
            target = rho.evolve(Kraus(self.depol_kraus(p)))
            output = rho.evolve(Kraus(PTM(self.depol_ptm(p))))
            self.assertEqual(output, target)

    def test_ptm_to_stinespring(self):
        """Test PTM to Stinespring transformation."""
        # Test unitary channels
        for mat, ptm in zip(self.unitary_mat, self.unitary_ptm):
            chan1 = Kraus(mat)
            chan2 = Kraus(PTM(ptm))
            self.assertTrue(matrix_equal(chan2.data[0], chan1.data[0], ignore_phase=True))
        # Test depolarizing channels
        rho = DensityMatrix(np.diag([1, 0]))
        for p in [0.25, 0.5, 0.75, 1]:
            target = rho.evolve(Stinespring(self.depol_stine(p)))
            output = rho.evolve(Stinespring(PTM(self.depol_ptm(p))))
            self.assertEqual(output, target)

    def test_ptm_to_chi(self):
        """Test PTM to Chi transformation."""
        # Test unitary channels
        for chi, ptm in zip(self.unitary_chi, self.unitary_ptm):
            chan1 = Chi(chi)
            chan2 = Chi(PTM(ptm))
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = Chi(self.depol_chi(p))
            chan2 = Chi(PTM(self.depol_ptm(p)))
            self.assertEqual(chan1, chan2)

    def test_ptm_to_ptm(self):
        """Test PTM to PTM transformation."""
        # Test unitary channels
        for ptm in self.unitary_ptm:
            chan1 = PTM(ptm)
            chan2 = PTM(chan1)
            self.assertEqual(chan1, chan2)
        # Test depolarizing channels
        for p in [0.25, 0.5, 0.75, 1]:
            chan1 = PTM(self.depol_ptm(p))
            chan2 = PTM(chan1)
            self.assertEqual(chan1, chan2)


if __name__ == "__main__":
    unittest.main()
