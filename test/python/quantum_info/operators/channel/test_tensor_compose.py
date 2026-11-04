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

    Tests that performing compose, tensor, conjugate, transpose, adjoint
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

    def _compare_tensor_to_operator(self, chans, mats):
        """Test tensor product is equivalent"""
        unitaries = [Operator(np.kron(j, i)) for i in mats for j in mats]
        channels = [j.tensor(i) for i in chans for j in chans]
        for chan, uni in zip(channels, unitaries):
            self.assertEqual(chan, chan.__class__(uni))

    def _compare_expand_to_operator(self, chans, mats):
        """Test expand product is equivalent"""
        unitaries = [Operator(np.kron(i, j)) for i in mats for j in mats]
        channels = [j.expand(i) for i in chans for j in chans]
        for chan, uni in zip(channels, unitaries):
            self.assertEqual(chan, chan.__class__(uni))

    def _compare_compose_to_operator(self, chans, mats):
        """Test compose is equivalent"""
        unitaries = [Operator(np.dot(i, j)) for i in mats for j in mats]
        channels = [j.compose(i) for i in chans for j in chans]
        for chan, uni in zip(channels, unitaries):
            self.assertEqual(chan, chan.__class__(uni))

    def _check_tensor_other_reps(self, chan):
        """Check tensor works for other representations"""
        current_rep = chan.__class__
        other_reps = [Operator, Choi, SuperOp, Kraus, Stinespring, Chi, PTM]
        for rep in other_reps:
            self.assertEqual(current_rep, chan.tensor(rep(chan)).__class__)

    def _check_expand_other_reps(self, chan):
        """Check expand works for other representations"""
        current_rep = chan.__class__
        other_reps = [Operator, Choi, SuperOp, Kraus, Stinespring, Chi, PTM]
        for rep in other_reps:
            self.assertEqual(current_rep, chan.expand(rep(chan)).__class__)

    def _check_compose_other_reps(self, chan):
        """Check compose works for other representations"""
        current_rep = chan.__class__
        other_reps = [Operator, Choi, SuperOp, Kraus, Stinespring, Chi, PTM]
        for rep in other_reps:
            self.assertEqual(current_rep, chan.compose(rep(chan)).__class__)

    def test_choi_tensor(self):
        """Test tensor of Choi matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Choi(mat) for mat in [self.choiI, self.choiX, self.choiY, self.choiZ, self.choiH]]
        self._compare_tensor_to_operator(chans, mats)

    def test_choi_tensor_random(self):
        """Test tensor of random Choi matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [Choi(Operator(mat)) for mat in mats]
        self._compare_tensor_to_operator(chans, mats)

    def test_choi_tensor_other_reps(self):
        """Test tensor of Choi works with other reps."""
        chan = Choi(self.choiI)
        self._check_tensor_other_reps(chan)

    def test_superop_tensor(self):
        """Test tensor of SuperOp matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [SuperOp(mat) for mat in [self.sopI, self.sopX, self.sopY, self.sopZ, self.sopH]]
        self._compare_tensor_to_operator(chans, mats)

    def test_superop_tensor_random(self):
        """Test tensor of SuperOp matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [SuperOp(Operator(mat)) for mat in mats]
        self._compare_tensor_to_operator(chans, mats)

    def test_superop_tensor_other_reps(self):
        """Test tensor of SuperOp works with other reps."""
        chan = SuperOp(self.sopI)
        self._check_tensor_other_reps(chan)

    def test_kraus_tensor(self):
        """Test tensor of Kraus matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Kraus(mat) for mat in mats]
        self._compare_tensor_to_operator(chans, mats)

    def test_kraus_tensor_random(self):
        """Test tensor of Kraus matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [Kraus(Operator(mat)) for mat in mats]
        self._compare_tensor_to_operator(chans, mats)

    def test_kraus_tensor_other_reps(self):
        """Test tensor of Kraus works with other reps."""
        chan = Kraus(self.UI)
        self._check_tensor_other_reps(chan)

    def test_stinespring_tensor(self):
        """Test tensor of Stinespring matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Stinespring(mat) for mat in mats]
        self._compare_tensor_to_operator(chans, mats)

    def test_stinespring_tensor_random(self):
        """Test tensor of Stinespring matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [Stinespring(Operator(mat)) for mat in mats]
        self._compare_tensor_to_operator(chans, mats)

    def test_stinespring_tensor_other_reps(self):
        """Test tensor of Stinespring works with other reps."""
        chan = Stinespring(self.UI)
        self._check_tensor_other_reps(chan)

    def test_chi_tensor(self):
        """Test tensor of Chi matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Chi(mat) for mat in [self.chiI, self.chiX, self.chiY, self.chiZ, self.chiH]]
        self._compare_tensor_to_operator(chans, mats)

    def test_chi_tensor_random(self):
        """Test tensor of Chi matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [Chi(Operator(mat)) for mat in mats]
        self._compare_tensor_to_operator(chans, mats)

    def test_chi_tensor_other_reps(self):
        """Test tensor of Chi works with other reps."""
        chan = Chi(self.chiI)
        self._check_tensor_other_reps(chan)

    def test_ptm_tensor(self):
        """Test tensor of PTM matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [PTM(mat) for mat in [self.ptmI, self.ptmX, self.ptmY, self.ptmZ, self.ptmH]]
        self._compare_tensor_to_operator(chans, mats)

    def test_ptm_tensor_random(self):
        """Test tensor of PTM matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [PTM(Operator(mat)) for mat in mats]
        self._compare_tensor_to_operator(chans, mats)

    def test_ptm_tensor_other_reps(self):
        """Test tensor of PTM works with other reps."""
        chan = PTM(self.ptmI)
        self._check_tensor_other_reps(chan)

    def test_choi_expand(self):
        """Test expand of Choi matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Choi(mat) for mat in [self.choiI, self.choiX, self.choiY, self.choiZ, self.choiH]]
        self._compare_expand_to_operator(chans, mats)

    def test_choi_expand_random(self):
        """Test expand of random Choi matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [Choi(Operator(mat)) for mat in mats]
        self._compare_expand_to_operator(chans, mats)

    def test_choi_expand_other_reps(self):
        """Test expand of Choi works with other reps."""
        chan = Choi(self.choiI)
        self._check_expand_other_reps(chan)

    def test_superop_expand(self):
        """Test expand of SuperOp matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [SuperOp(mat) for mat in [self.sopI, self.sopX, self.sopY, self.sopZ, self.sopH]]
        self._compare_expand_to_operator(chans, mats)

    def test_superop_expand_random(self):
        """Test expand of SuperOp matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [SuperOp(Operator(mat)) for mat in mats]
        self._compare_expand_to_operator(chans, mats)

    def test_superop_expand_other_reps(self):
        """Test expand of SuperOp works with other reps."""
        chan = SuperOp(self.sopI)
        self._check_expand_other_reps(chan)

    def test_kraus_expand(self):
        """Test expand of Kraus matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Kraus(mat) for mat in mats]
        self._compare_expand_to_operator(chans, mats)

    def test_kraus_expand_random(self):
        """Test expand of Kraus matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [Kraus(Operator(mat)) for mat in mats]
        self._compare_expand_to_operator(chans, mats)

    def test_kraus_expand_other_reps(self):
        """Test expand of Kraus works with other reps."""
        chan = Kraus(self.UI)
        self._check_expand_other_reps(chan)

    def test_stinespring_expand(self):
        """Test expand of Stinespring matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Stinespring(mat) for mat in mats]
        self._compare_expand_to_operator(chans, mats)

    def test_stinespring_expand_random(self):
        """Test expand of Stinespring matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [Stinespring(Operator(mat)) for mat in mats]
        self._compare_expand_to_operator(chans, mats)

    def test_stinespring_expand_other_reps(self):
        """Test expand of Stinespring works with other reps."""
        chan = Stinespring(self.UI)
        self._check_expand_other_reps(chan)

    def test_chi_expand(self):
        """Test expand of Chi matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Chi(mat) for mat in [self.chiI, self.chiX, self.chiY, self.chiZ, self.chiH]]
        self._compare_expand_to_operator(chans, mats)

    def test_chi_expand_random(self):
        """Test expand of Chi matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [Chi(Operator(mat)) for mat in mats]
        self._compare_expand_to_operator(chans, mats)

    def test_chi_expand_other_reps(self):
        """Test expand of Chi works with other reps."""
        chan = Chi(self.chiI)
        self._check_expand_other_reps(chan)

    def test_ptm_expand(self):
        """Test expand of PTM matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [PTM(mat) for mat in [self.ptmI, self.ptmX, self.ptmY, self.ptmZ, self.ptmH]]
        self._compare_expand_to_operator(chans, mats)

    def test_ptm_expand_random(self):
        """Test expand of PTM matrices is correct."""
        mats = [self.rand_matrix(2, 2) for _ in range(4)]
        chans = [PTM(Operator(mat)) for mat in mats]
        self._compare_expand_to_operator(chans, mats)

    def test_ptm_expand_other_reps(self):
        """Test expand of PTM works with other reps."""
        chan = PTM(self.ptmI)
        self._check_expand_other_reps(chan)

    def test_choi_compose(self):
        """Test compose of Choi matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Choi(mat) for mat in [self.choiI, self.choiX, self.choiY, self.choiZ, self.choiH]]
        self._compare_compose_to_operator(chans, mats)

    def test_choi_compose_random(self):
        """Test compose of Choi matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Choi(Operator(mat)) for mat in mats]
        self._compare_compose_to_operator(chans, mats)

    def test_choi_compose_other_reps(self):
        """Test compose of Choi works with other reps."""
        chan = Choi(self.choiI)
        self._check_compose_other_reps(chan)

    def test_superop_compose(self):
        """Test compose of SuperOp matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [SuperOp(mat) for mat in [self.sopI, self.sopX, self.sopY, self.sopZ, self.sopH]]
        self._compare_compose_to_operator(chans, mats)

    def test_superop_compose_random(self):
        """Test compose of SuperOp matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [SuperOp(Operator(mat)) for mat in mats]
        self._compare_compose_to_operator(chans, mats)

    def test_superop_compose_other_reps(self):
        """Test compose of Superop works with other reps."""
        chan = SuperOp(self.sopI)
        self._check_compose_other_reps(chan)

    def test_kraus_compose(self):
        """Test compose of Kraus matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Kraus(mat) for mat in mats]
        self._compare_compose_to_operator(chans, mats)

    def test_kraus_compose_random(self):
        """Test compose of Kraus matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Kraus(Operator(mat)) for mat in mats]
        self._compare_compose_to_operator(chans, mats)

    def test_kraus_compose_other_reps(self):
        """Test compose of Kraus works with other reps."""
        chan = Kraus(self.UI)
        self._check_compose_other_reps(chan)

    def test_stinespring_compose(self):
        """Test compose of Stinespring matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Stinespring(mat) for mat in mats]
        self._compare_compose_to_operator(chans, mats)

    def test_stinespring_compose_random(self):
        """Test compose of Stinespring matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Stinespring(Operator(mat)) for mat in mats]
        self._compare_compose_to_operator(chans, mats)

    def test_stinespring_compose_other_reps(self):
        """Test compose of Stinespring works with other reps."""
        chan = Stinespring(self.UI)
        self._check_compose_other_reps(chan)

    def test_chi_compose(self):
        """Test compose of Chi matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [Chi(mat) for mat in [self.chiI, self.chiX, self.chiY, self.chiZ, self.chiH]]
        self._compare_compose_to_operator(chans, mats)

    def test_chi_compose_random(self):
        """Test compose of Chi matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [Chi(Operator(mat)) for mat in mats]
        self._compare_compose_to_operator(chans, mats)

    def test_chi_compose_other_reps(self):
        """Test compose of Chi works with other reps."""
        chan = Chi(self.chiI)
        self._check_compose_other_reps(chan)

    def test_ptm_compose(self):
        """Test compose of PTM matrices is correct."""
        mats = [self.UI, self.UX, self.UY, self.UZ, self.UH]
        chans = [PTM(mat) for mat in [self.ptmI, self.ptmX, self.ptmY, self.ptmZ, self.ptmH]]
        self._compare_compose_to_operator(chans, mats)

    def test_ptm_compose_random(self):
        """Test compose of PTM matrices is correct."""
        mats = [self.rand_matrix(4, 4) for _ in range(4)]
        chans = [PTM(Operator(mat)) for mat in mats]
        self._compare_compose_to_operator(chans, mats)

    def test_ptm_compose_other_reps(self):
        """Test compose of PTM works with other reps."""
        chan = PTM(self.ptmI)
        self._check_compose_other_reps(chan)


if __name__ == "__main__":
    unittest.main()
