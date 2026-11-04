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
from numpy.testing import assert_allclose

from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.channel.stinespring import Stinespring
from qiskit.quantum_info.operators.channel.ptm import PTM
from qiskit.quantum_info.operators.channel.chi import Chi
from .channel_test_case import ChannelTestCase


class TestEvolve(ChannelTestCase):
    """Random tests for equivalence of channel evolution."""

    qubits_test_cases = (1, 2)
    repetitions = 2

    def _unitary_to_other(self, rep, qubits_test_cases, repetitions):
        """Test Operator to Other evolution."""
        for nq in qubits_test_cases:
            dim = 2**nq
            for _ in range(repetitions):
                rho = self.rand_rho(dim)
                mat = self.rand_matrix(dim, dim)
                chan = Operator(mat)
                rho1 = DensityMatrix(rho).evolve(chan).data
                rho2 = DensityMatrix(rho).evolve(rep(chan)).data
                assert_allclose(rho1, rho2)

    def _other_to_operator(self, rep, qubits_test_cases, repetitions):
        """Test Other to Operator evolution."""
        for nq in qubits_test_cases:
            dim = 2**nq
            for _ in range(repetitions):
                rho = self.rand_rho(dim)
                mat = self.rand_matrix(dim, dim)
                chan = rep(Operator(mat))
                rho1 = DensityMatrix(rho).evolve(chan).data
                rho2 = DensityMatrix(rho).evolve(Operator(chan)).data
                assert_allclose(rho1, rho2)

    def _choi_to_other_cp(self, rep, qubits_test_cases, repetitions):
        """Test CP Choi to Other evolution."""
        for nq in qubits_test_cases:
            dim = 2**nq
            for _ in range(repetitions):
                rho = self.rand_rho(dim)
                mat = dim * self.rand_rho(dim**2)
                chan = Choi(mat)
                rho1 = DensityMatrix(rho).evolve(chan).data
                rho2 = DensityMatrix(rho).evolve(rep(chan)).data
                assert_allclose(rho1, rho2)

    def _choi_to_other_noncp(self, rep, qubits_test_cases, repetitions):
        """Test CP Choi to Other evolution."""
        for nq in qubits_test_cases:
            dim = 2**nq
            for _ in range(repetitions):
                rho = self.rand_rho(dim)
                mat = self.rand_matrix(dim**2, dim**2)
                chan = Choi(mat)
                rho1 = DensityMatrix(rho).evolve(chan).data
                rho2 = DensityMatrix(rho).evolve(rep(chan)).data
                assert_allclose(rho1, rho2)

    def _superop_to_other(self, rep, qubits_test_cases, repetitions):
        """Test SuperOp to Other evolution."""
        for nq in qubits_test_cases:
            dim = 2**nq
            for _ in range(repetitions):
                rho = self.rand_rho(dim)
                mat = self.rand_matrix(dim**2, dim**2)
                chan = SuperOp(mat)
                rho1 = DensityMatrix(rho).evolve(chan).data
                rho2 = DensityMatrix(rho).evolve(rep(chan)).data
                assert_allclose(rho1, rho2)

    def _kraus_to_other_single(self, rep, qubits_test_cases, repetitions):
        """Test single Kraus to Other evolution."""
        for nq in qubits_test_cases:
            dim = 2**nq
            for _ in range(repetitions):
                rho = self.rand_rho(dim)
                kraus = self.rand_kraus(dim, dim, dim**2)
                chan = Kraus(kraus)
                rho1 = DensityMatrix(rho).evolve(chan).data
                rho2 = DensityMatrix(rho).evolve(rep(chan)).data
                assert_allclose(rho1, rho2)

    def _kraus_to_other_double(self, rep, qubits_test_cases, repetitions):
        """Test double Kraus to Other evolution."""
        for nq in qubits_test_cases:
            dim = 2**nq
            for _ in range(repetitions):
                rho = self.rand_rho(dim)
                kraus_l = self.rand_kraus(dim, dim, dim**2)
                kraus_r = self.rand_kraus(dim, dim, dim**2)
                chan = Kraus((kraus_l, kraus_r))
                rho1 = DensityMatrix(rho).evolve(chan).data
                rho2 = DensityMatrix(rho).evolve(rep(chan)).data
                assert_allclose(rho1, rho2)

    def _stinespring_to_other_single(self, rep, qubits_test_cases, repetitions):
        """Test single Stinespring to Other evolution."""
        for nq in qubits_test_cases:
            dim = 2**nq
            for _ in range(repetitions):
                rho = self.rand_rho(dim)
                mat = self.rand_matrix(dim**2, dim)
                chan = Stinespring(mat)
                rho1 = DensityMatrix(rho).evolve(chan).data
                rho2 = DensityMatrix(rho).evolve(rep(chan)).data
                assert_allclose(rho1, rho2)

    def _stinespring_to_other_double(self, rep, qubits_test_cases, repetitions):
        """Test double Stinespring to Other evolution."""
        for nq in qubits_test_cases:
            dim = 2**nq
            for _ in range(repetitions):
                rho = self.rand_rho(dim)
                mat_l = self.rand_matrix(dim**2, dim)
                mat_r = self.rand_matrix(dim**2, dim)
                chan = Stinespring((mat_l, mat_r))
                rho1 = DensityMatrix(rho).evolve(chan).data
                rho2 = DensityMatrix(rho).evolve(rep(chan)).data
                assert_allclose(rho1, rho2)

    def _chi_to_other(self, rep, qubits_test_cases, repetitions):
        """Test Chi to Other evolution."""
        for nq in qubits_test_cases:
            dim = 2**nq
            for _ in range(repetitions):
                rho = self.rand_rho(dim)
                mat = self.rand_matrix(dim**2, dim**2, real=True)
                chan = Chi(mat)
                rho1 = DensityMatrix(rho).evolve(chan).data
                rho2 = DensityMatrix(rho).evolve(rep(chan)).data
                assert_allclose(rho1, rho2)

    def _ptm_to_other(self, rep, qubits_test_cases, repetitions):
        """Test PTM to Other evolution."""
        for nq in qubits_test_cases:
            dim = 2**nq
            for _ in range(repetitions):
                rho = self.rand_rho(dim)
                mat = self.rand_matrix(dim**2, dim**2, real=True)
                chan = PTM(mat)
                rho1 = DensityMatrix(rho).evolve(chan).data
                rho2 = DensityMatrix(rho).evolve(rep(chan)).data
                assert_allclose(rho1, rho2)

    def test_unitary_to_choi(self):
        """Test Operator to Choi evolution."""
        self._unitary_to_other(Choi, self.qubits_test_cases, self.repetitions)

    def test_unitary_to_superop(self):
        """Test Operator to SuperOp evolution."""
        self._unitary_to_other(SuperOp, self.qubits_test_cases, self.repetitions)

    def test_unitary_to_kraus(self):
        """Test Operator to Kraus evolution."""
        self._unitary_to_other(Kraus, self.qubits_test_cases, self.repetitions)

    def test_unitary_to_stinespring(self):
        """Test Operator to Stinespring evolution."""
        self._unitary_to_other(Stinespring, self.qubits_test_cases, self.repetitions)

    def test_unitary_to_chi(self):
        """Test Operator to Chi evolution."""
        self._unitary_to_other(Chi, self.qubits_test_cases, self.repetitions)

    def test_unitary_to_ptm(self):
        """Test Operator to PTM evolution."""
        self._unitary_to_other(PTM, self.qubits_test_cases, self.repetitions)

    def test_choi_to_operator(self):
        """Test Choi to Operator evolution."""
        self._other_to_operator(Choi, self.qubits_test_cases, self.repetitions)

    def test_choi_to_superop_cp(self):
        """Test CP Choi to SuperOp evolution."""
        self._choi_to_other_cp(SuperOp, self.qubits_test_cases, self.repetitions)

    def test_choi_to_kraus_cp(self):
        """Test CP Choi to Kraus evolution."""
        self._choi_to_other_cp(Kraus, self.qubits_test_cases, self.repetitions)

    def test_choi_to_stinespring_cp(self):
        """Test CP Choi to Stinespring evolution."""
        self._choi_to_other_cp(Stinespring, self.qubits_test_cases, self.repetitions)

    def test_choi_to_chi_cp(self):
        """Test CP Choi to Chi evolution."""
        self._choi_to_other_cp(Chi, self.qubits_test_cases, self.repetitions)

    def test_choi_to_ptm_cp(self):
        """Test CP Choi to PTM evolution."""
        self._choi_to_other_cp(PTM, self.qubits_test_cases, self.repetitions)

    def test_choi_to_superop_noncp(self):
        """Test CP Choi to SuperOp evolution."""
        self._choi_to_other_noncp(SuperOp, self.qubits_test_cases, self.repetitions)

    def test_choi_to_kraus_noncp(self):
        """Test CP Choi to Kraus evolution."""
        self._choi_to_other_noncp(Kraus, self.qubits_test_cases, self.repetitions)

    def test_choi_to_stinespring_noncp(self):
        """Test CP Choi to Stinespring evolution."""
        self._choi_to_other_noncp(Stinespring, self.qubits_test_cases, self.repetitions)

    def test_choi_to_chi_noncp(self):
        """Test Choi to Chi evolution."""
        self._choi_to_other_noncp(Chi, self.qubits_test_cases, self.repetitions)

    def test_choi_to_ptm_noncp(self):
        """Test Non-CP Choi to PTM evolution."""
        self._choi_to_other_noncp(PTM, self.qubits_test_cases, self.repetitions)

    def test_superop_to_operator(self):
        """Test SuperOp to Operator evolution."""
        self._other_to_operator(SuperOp, self.qubits_test_cases, self.repetitions)

    def test_superop_to_choi(self):
        """Test SuperOp to Choi evolution."""
        self._superop_to_other(Choi, self.qubits_test_cases, self.repetitions)

    def test_superop_to_kraus(self):
        """Test SuperOp to Kraus evolution."""
        self._superop_to_other(Kraus, self.qubits_test_cases, self.repetitions)

    def test_superop_to_stinespring(self):
        """Test SuperOp to Stinespring evolution."""
        self._superop_to_other(Stinespring, self.qubits_test_cases, self.repetitions)

    def test_superop_to_chi(self):
        """Test SuperOp to Chi evolution."""
        self._superop_to_other(Chi, self.qubits_test_cases, self.repetitions)

    def test_superop_to_ptm(self):
        """Test SuperOp to PTM evolution."""
        self._superop_to_other(PTM, self.qubits_test_cases, self.repetitions)

    def test_kraus_to_operator(self):
        """Test Kraus to Operator evolution."""
        self._other_to_operator(Kraus, self.qubits_test_cases, self.repetitions)

    def test_kraus_to_choi_single(self):
        """Test single Kraus to Choi evolution."""
        self._kraus_to_other_single(Choi, self.qubits_test_cases, self.repetitions)

    def test_kraus_to_superop_single(self):
        """Test single Kraus to SuperOp evolution."""
        self._kraus_to_other_single(SuperOp, self.qubits_test_cases, self.repetitions)

    def test_kraus_to_stinespring_single(self):
        """Test single Kraus to Stinespring evolution."""
        self._kraus_to_other_single(Stinespring, self.qubits_test_cases, self.repetitions)

    def test_kraus_to_chi_single(self):
        """Test single Kraus to Chi evolution."""
        self._kraus_to_other_single(Chi, self.qubits_test_cases, self.repetitions)

    def test_kraus_to_ptm_single(self):
        """Test single Kraus to PTM evolution."""
        self._kraus_to_other_single(PTM, self.qubits_test_cases, self.repetitions)

    def test_kraus_to_choi_double(self):
        """Test single Kraus to Choi evolution."""
        self._kraus_to_other_double(Choi, self.qubits_test_cases, self.repetitions)

    def test_kraus_to_superop_double(self):
        """Test single Kraus to SuperOp evolution."""
        self._kraus_to_other_double(SuperOp, self.qubits_test_cases, self.repetitions)

    def test_kraus_to_stinespring_double(self):
        """Test single Kraus to Stinespring evolution."""
        self._kraus_to_other_double(Stinespring, self.qubits_test_cases, self.repetitions)

    def test_kraus_to_chi_double(self):
        """Test single Kraus to Chi evolution."""
        self._kraus_to_other_double(Chi, self.qubits_test_cases, self.repetitions)

    def test_kraus_to_ptm_double(self):
        """Test single Kraus to PTM evolution."""
        self._kraus_to_other_double(PTM, self.qubits_test_cases, self.repetitions)

    def test_stinespring_to_operator(self):
        """Test Stinespring to Operator evolution."""
        self._other_to_operator(Stinespring, self.qubits_test_cases, self.repetitions)

    def test_stinespring_to_choi_single(self):
        """Test single Stinespring to Choi evolution."""
        self._stinespring_to_other_single(Choi, self.qubits_test_cases, self.repetitions)

    def test_stinespring_to_superop_single(self):
        """Test single Stinespring to SuperOp evolution."""
        self._stinespring_to_other_single(SuperOp, self.qubits_test_cases, self.repetitions)

    def test_stinespring_to_kraus_single(self):
        """Test single Stinespring to Kraus evolution."""
        self._stinespring_to_other_single(Kraus, self.qubits_test_cases, self.repetitions)

    def test_stinespring_to_chi_single(self):
        """Test single Stinespring to Chi evolution."""
        self._stinespring_to_other_single(Chi, self.qubits_test_cases, self.repetitions)

    def test_stinespring_to_ptm_single(self):
        """Test single Stinespring to PTM evolution."""
        self._stinespring_to_other_single(PTM, self.qubits_test_cases, self.repetitions)

    def test_stinespring_to_choi_double(self):
        """Test single Stinespring to Choi evolution."""
        self._stinespring_to_other_double(Choi, self.qubits_test_cases, self.repetitions)

    def test_stinespring_to_superop_double(self):
        """Test single Stinespring to SuperOp evolution."""
        self._stinespring_to_other_double(SuperOp, self.qubits_test_cases, self.repetitions)

    def test_stinespring_to_kraus_double(self):
        """Test single Stinespring to Kraus evolution."""
        self._stinespring_to_other_double(Kraus, self.qubits_test_cases, self.repetitions)

    def test_stinespring_to_chi_double(self):
        """Test single Stinespring to Chi evolution."""
        self._stinespring_to_other_double(Chi, self.qubits_test_cases, self.repetitions)

    def test_stinespring_to_ptm_double(self):
        """Test single Stinespring to PTM evolution."""
        self._stinespring_to_other_double(PTM, self.qubits_test_cases, self.repetitions)

    def test_ptm_to_operator(self):
        """Test PTM to Operator evolution."""
        self._other_to_operator(PTM, self.qubits_test_cases, self.repetitions)

    def test_ptm_to_choi(self):
        """Test PTM to Choi evolution."""
        self._ptm_to_other(Choi, self.qubits_test_cases, self.repetitions)

    def test_ptm_to_superop(self):
        """Test PTM to SuperOp evolution."""
        self._ptm_to_other(SuperOp, self.qubits_test_cases, self.repetitions)

    def test_ptm_to_kraus(self):
        """Test PTM to Kraus evolution."""
        self._ptm_to_other(Kraus, self.qubits_test_cases, self.repetitions)

    def test_ptm_to_stinespring(self):
        """Test PTM to Stinespring evolution."""
        self._ptm_to_other(Stinespring, self.qubits_test_cases, self.repetitions)

    def test_ptm_to_chi(self):
        """Test PTM to Chi evolution."""
        self._ptm_to_other(Chi, self.qubits_test_cases, self.repetitions)

    def test_chi_to_operator(self):
        """Test Chi to Operator evolution."""
        self._other_to_operator(Chi, self.qubits_test_cases, self.repetitions)

    def test_chi_to_choi(self):
        """Test Chi to Choi evolution."""
        self._chi_to_other(Choi, self.qubits_test_cases, self.repetitions)

    def test_chi_to_superop(self):
        """Test Chi to SuperOp evolution."""
        self._chi_to_other(SuperOp, self.qubits_test_cases, self.repetitions)

    def test_chi_to_kraus(self):
        """Test Chi to Kraus evolution."""
        self._chi_to_other(Kraus, self.qubits_test_cases, self.repetitions)

    def test_chi_to_stinespring(self):
        """Test Chi to Stinespring evolution."""
        self._chi_to_other(Stinespring, self.qubits_test_cases, self.repetitions)

    def test_chi_to_ptm(self):
        """Test Chi to PTM evolution."""
        self._chi_to_other(PTM, self.qubits_test_cases, self.repetitions)


if __name__ == "__main__":
    unittest.main()
