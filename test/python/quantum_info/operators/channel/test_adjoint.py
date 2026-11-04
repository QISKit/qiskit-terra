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

"""Equivalence tests for quantum channel methods."""

import unittest

import numpy as np

from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.channel.stinespring import Stinespring
from qiskit.quantum_info.operators.channel.ptm import PTM
from qiskit.quantum_info.operators.channel.chi import Chi
from .channel_test_case import ChannelTestCase


class TestEquivalence(ChannelTestCase):
    """Tests for channel equivalence.

    Tests that performing conjugate, transpose, adjoint
    in the Operator representation is equivalent to performing the same
    operations in other representations.
    """

    unitaries = [
        ChannelTestCase.UI,
        ChannelTestCase.UX,
        ChannelTestCase.UY,
        ChannelTestCase.UZ,
        ChannelTestCase.UH,
    ]

    chois = [
        ChannelTestCase.choiI,
        ChannelTestCase.choiX,
        ChannelTestCase.choiY,
        ChannelTestCase.choiZ,
        ChannelTestCase.choiH,
    ]

    chis = [
        ChannelTestCase.chiI,
        ChannelTestCase.chiX,
        ChannelTestCase.chiY,
        ChannelTestCase.chiZ,
        ChannelTestCase.chiH,
    ]

    sops = [
        ChannelTestCase.sopI,
        ChannelTestCase.sopX,
        ChannelTestCase.sopY,
        ChannelTestCase.sopZ,
        ChannelTestCase.sopH,
    ]

    ptms = [
        ChannelTestCase.ptmI,
        ChannelTestCase.ptmX,
        ChannelTestCase.ptmY,
        ChannelTestCase.ptmZ,
        ChannelTestCase.ptmH,
    ]

    def _compare_transpose_to_operator(self, chans, mats):
        """Test transpose is equivalent"""
        unitaries = [Operator(np.transpose(i)) for i in mats]
        channels = [i.transpose() for i in chans]
        for chan, uni in zip(channels, unitaries):
            self.assertEqual(chan, chan.__class__(uni))

    def _compare_conjugate_to_operator(self, chans, mats):
        """Test conjugate is equivalent"""
        unitaries = [Operator(np.conjugate(i)) for i in mats]
        channels = [i.conjugate() for i in chans]
        for chan, uni in zip(channels, unitaries):
            self.assertEqual(chan, chan.__class__(uni))

    def _compare_adjoint_to_operator(self, chans, mats):
        """Test adjoint is equivalent"""
        unitaries = [Operator(np.conjugate(np.transpose(i))) for i in mats]
        channels = [i.adjoint() for i in chans]
        for chan, uni in zip(channels, unitaries):
            self.assertEqual(chan, chan.__class__(uni))

    def test_choi_conjugate(self):
        """Test conjugate of Choi matrices is correct."""
        mats = self.unitaries
        chans = [Choi(mat) for mat in self.chois]
        self._compare_conjugate_to_operator(chans, mats)

    def test_superop_conjugate(self):
        """Test conjugate of SuperOp matrices is correct."""
        mats = self.unitaries
        chans = [SuperOp(mat) for mat in self.sops]
        self._compare_conjugate_to_operator(chans, mats)

    def test_kraus_conjugate(self):
        """Test conjugate of Kraus matrices is correct."""
        mats = self.unitaries
        chans = [Kraus(mat) for mat in mats]
        self._compare_conjugate_to_operator(chans, mats)

    def test_stinespring_conjugate(self):
        """Test conjugate of Stinespring matrices is correct."""
        mats = self.unitaries
        chans = [Stinespring(mat) for mat in mats]
        self._compare_conjugate_to_operator(chans, mats)

    def test_chi_conjugate(self):
        """Test conjugate of Chi matrices is correct."""
        mats = self.unitaries
        chans = [Chi(mat) for mat in self.chis]
        self._compare_conjugate_to_operator(chans, mats)

    def test_ptm_conjugate(self):
        """Test conjugate of PTM matrices is correct."""
        mats = self.unitaries
        chans = [PTM(mat) for mat in self.ptms]
        self._compare_conjugate_to_operator(chans, mats)

    def test_choi_conjugate_random(self):
        """Test conjugate of Choi matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Choi(Operator(mat)) for mat in mats]
        self._compare_conjugate_to_operator(chans, mats)

    def test_superop_conjugate_random(self):
        """Test conjugate of SuperOp matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [SuperOp(Operator(mat)) for mat in mats]
        self._compare_conjugate_to_operator(chans, mats)

    def test_kraus_conjugate_random(self):
        """Test conjugate of Kraus matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Kraus(Operator(mat)) for mat in mats]
        self._compare_conjugate_to_operator(chans, mats)

    def test_stinespring_conjugate_random(self):
        """Test conjugate of Stinespring matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Stinespring(Operator(mat)) for mat in mats]
        self._compare_conjugate_to_operator(chans, mats)

    def test_chi_conjugate_random(self):
        """Test conjugate of Chi matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Chi(Operator(mat)) for mat in mats]
        self._compare_conjugate_to_operator(chans, mats)

    def test_ptm_conjugate_random(self):
        """Test conjugate of PTM matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [PTM(Operator(mat)) for mat in mats]
        self._compare_conjugate_to_operator(chans, mats)

    def test_choi_transpose(self):
        """Test transpose of Choi matrices is correct."""
        mats = self.unitaries
        chans = [Choi(mat) for mat in self.chois]
        self._compare_transpose_to_operator(chans, mats)

    def test_superop_transpose(self):
        """Test transpose of SuperOp matrices is correct."""
        mats = self.unitaries
        chans = [SuperOp(mat) for mat in self.sops]
        self._compare_transpose_to_operator(chans, mats)

    def test_kraus_transpose(self):
        """Test transpose of Kraus matrices is correct."""
        mats = self.unitaries
        chans = [Kraus(mat) for mat in mats]
        self._compare_transpose_to_operator(chans, mats)

    def test_stinespring_transpose(self):
        """Test transpose of Stinespring matrices is correct."""
        mats = self.unitaries
        chans = [Stinespring(mat) for mat in mats]
        self._compare_transpose_to_operator(chans, mats)

    def test_chi_transpose(self):
        """Test transpose of Chi matrices is correct."""
        mats = self.unitaries
        chans = [Chi(mat) for mat in self.chis]
        self._compare_transpose_to_operator(chans, mats)

    def test_ptm_transpose(self):
        """Test transpose of PTM matrices is correct."""
        mats = self.unitaries
        chans = [PTM(mat) for mat in self.ptms]
        self._compare_transpose_to_operator(chans, mats)

    def test_choi_transpose_random(self):
        """Test transpose of Choi matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Choi(Operator(mat)) for mat in mats]
        self._compare_transpose_to_operator(chans, mats)

    def test_superop_transpose_random(self):
        """Test transpose of SuperOp matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [SuperOp(Operator(mat)) for mat in mats]
        self._compare_transpose_to_operator(chans, mats)

    def test_kraus_transpose_random(self):
        """Test transpose of Kraus matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Kraus(Operator(mat)) for mat in mats]
        self._compare_transpose_to_operator(chans, mats)

    def test_stinespring_transpose_random(self):
        """Test transpose of Stinespring matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Stinespring(Operator(mat)) for mat in mats]
        self._compare_transpose_to_operator(chans, mats)

    def test_chi_transpose_random(self):
        """Test transpose of Chi matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Chi(Operator(mat)) for mat in mats]
        self._compare_transpose_to_operator(chans, mats)

    def test_ptm_transpose_random(self):
        """Test transpose of PTM matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [PTM(Operator(mat)) for mat in mats]
        self._compare_transpose_to_operator(chans, mats)

    def test_choi_adjoint(self):
        """Test adjoint of Choi matrices is correct."""
        mats = self.unitaries
        chans = [Choi(mat) for mat in self.chois]
        self._compare_adjoint_to_operator(chans, mats)

    def test_superop_adjoint(self):
        """Test adjoint of SuperOp matrices is correct."""
        mats = self.unitaries
        chans = [SuperOp(mat) for mat in self.sops]
        self._compare_adjoint_to_operator(chans, mats)

    def test_kraus_adjoint(self):
        """Test adjoint of Kraus matrices is correct."""
        mats = self.unitaries
        chans = [Kraus(mat) for mat in mats]
        self._compare_adjoint_to_operator(chans, mats)

    def test_stinespring_adjoint(self):
        """Test adjoint of Stinespring matrices is correct."""
        mats = self.unitaries
        chans = [Stinespring(mat) for mat in mats]
        self._compare_adjoint_to_operator(chans, mats)

    def test_chi_adjoint(self):
        """Test adjoint of Chi matrices is correct."""
        mats = self.unitaries
        chans = [Chi(mat) for mat in self.chis]
        self._compare_adjoint_to_operator(chans, mats)

    def test_ptm_adjoint(self):
        """Test adjoint of PTM matrices is correct."""
        mats = self.unitaries
        chans = [PTM(mat) for mat in self.ptms]
        self._compare_adjoint_to_operator(chans, mats)

    def test_choi_adjoint_random(self):
        """Test adjoint of Choi matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Choi(Operator(mat)) for mat in mats]
        self._compare_adjoint_to_operator(chans, mats)

    def test_superop_adjoint_random(self):
        """Test adjoint of SuperOp matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [SuperOp(Operator(mat)) for mat in mats]
        self._compare_adjoint_to_operator(chans, mats)

    def test_kraus_adjoint_random(self):
        """Test adjoint of Kraus matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Kraus(Operator(mat)) for mat in mats]
        self._compare_adjoint_to_operator(chans, mats)

    def test_stinespring_adjoint_random(self):
        """Test adjoint of Stinespring matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Stinespring(Operator(mat)) for mat in mats]
        self._compare_adjoint_to_operator(chans, mats)

    def test_chi_adjoint_random(self):
        """Test adjoint of Chi matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Chi(Operator(mat)) for mat in mats]
        self._compare_adjoint_to_operator(chans, mats)

    def test_ptm_adjoint_random(self):
        """Test adjoint of PTM matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [PTM(Operator(mat)) for mat in mats]
        self._compare_adjoint_to_operator(chans, mats)


if __name__ == "__main__":
    unittest.main()
