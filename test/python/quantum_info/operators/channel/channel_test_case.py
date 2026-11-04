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

"""Tests for quantum channel representation class."""

import numpy as np

from qiskit.quantum_info.operators.channel import SuperOp
from ..test_operator import OperatorTestCase


class ChannelTestCase(OperatorTestCase):
    """Tests for Channel representations."""

    # Pauli-matrix superoperators
    sopI = np.eye(4)
    sopX = np.kron(OperatorTestCase.UX.conj(), OperatorTestCase.UX)
    sopY = np.kron(OperatorTestCase.UY.conj(), OperatorTestCase.UY)
    sopZ = np.kron(OperatorTestCase.UZ.conj(), OperatorTestCase.UZ)
    sopH = np.kron(OperatorTestCase.UH.conj(), OperatorTestCase.UH)

    # Choi-matrices for Pauli-matrix unitaries
    choiI = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
    choiX = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    choiY = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])
    choiZ = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
    choiH = np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]) / 2

    # Chi-matrices for Pauli-matrix unitaries
    chiI = np.diag([2, 0, 0, 0])
    chiX = np.diag([0, 2, 0, 0])
    chiY = np.diag([0, 0, 2, 0])
    chiZ = np.diag([0, 0, 0, 2])
    chiH = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]])

    # PTM-matrices for Pauli-matrix unitaries
    ptmI = np.diag([1, 1, 1, 1])
    ptmX = np.diag([1, 1, -1, -1])
    ptmY = np.diag([1, -1, 1, -1])
    ptmZ = np.diag([1, -1, -1, 1])
    ptmH = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]])

    def simple_circuit_no_measure(self):
        """Return a unitary circuit and the corresponding unitary array."""
        # Override OperatorTestCase to return a SuperOp
        circ, target = super().simple_circuit_no_measure()
        return circ, SuperOp(target)

    # Depolarizing channels
    def depol_kraus(self, p):
        """Depolarizing channel Kraus operators"""
        return [
            np.sqrt(1 - p * 3 / 4) * self.UI,
            np.sqrt(p / 4) * self.UX,
            np.sqrt(p / 4) * self.UY,
            np.sqrt(p / 4) * self.UZ,
        ]

    def depol_sop(self, p):
        """Depolarizing channel superoperator matrix"""
        return (1 - p) * self.sopI + p * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
        ) / 2

    def depol_choi(self, p):
        """Depolarizing channel Choi-matrix"""
        return (1 - p) * self.choiI + p * np.eye(4) / 2

    def depol_chi(self, p):
        """Depolarizing channel Chi-matrix"""
        return 2 * np.diag([1 - 3 * p / 4, p / 4, p / 4, p / 4])

    def depol_ptm(self, p):
        """Depolarizing channel PTM"""
        return np.diag([1, 1 - p, 1 - p, 1 - p])

    def depol_stine(self, p):
        """Depolarizing channel Stinespring matrix"""
        kraus = self.depol_kraus(p)
        basis = np.eye(4).reshape((4, 4, 1))
        return np.sum([np.kron(k, b) for k, b in zip(kraus, basis)], axis=0)

    def rand_kraus(self, input_dim, output_dim, n):
        """Return a random (non-CPTP) Kraus operator map"""
        return [self.rand_matrix(output_dim, input_dim) for _ in range(n)]
