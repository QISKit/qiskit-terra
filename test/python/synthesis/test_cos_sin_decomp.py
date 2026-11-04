# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for cosine-sine decomposition."""

import math

from ddt import ddt
import numpy as np

from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.circuit._utils import _compute_control_matrix
from qiskit._accelerate.cos_sin_decomp import cossin

from test import combine
from test import QiskitTestCase


@ddt
class TestCSD(QiskitTestCase):
    """Test the cosine-sine decomposition."""

    def assertCossinDecompositionIsCorrect(self, mat):
        """Run QSD on mat and verify that it is correct."""
        n = mat.shape[0] // 2

        # Run QSD: u = [u1, u2], v = [v1, v2]
        csd_u, csd_thetas, csd_v = cossin(mat)

        # Check that the computed matrices are unitary
        self.assertTrue(is_unitary_matrix(csd_u[0], atol=1e-13))
        self.assertTrue(is_unitary_matrix(csd_u[1], atol=1e-13))
        self.assertTrue(is_unitary_matrix(csd_v[0], atol=1e-13))
        self.assertTrue(is_unitary_matrix(csd_v[1], atol=1e-13))

        # Create appropriate 2n x 2n matrices and multiply
        zero_mat = np.zeros((n, n), dtype=complex)
        c_mat = np.diag([math.cos(theta) for theta in csd_thetas])
        s_mat = np.diag([math.sin(theta) for theta in csd_thetas])
        cs_mat = np.vstack([np.hstack([c_mat, -1 * s_mat]), np.hstack([s_mat, c_mat])])
        u_mat = np.vstack([np.hstack([csd_u[0], zero_mat]), np.hstack([zero_mat, csd_u[1]])])
        v_mat = np.vstack([np.hstack([csd_v[0], zero_mat]), np.hstack([zero_mat, csd_v[1]])])
        recomputed_mat = u_mat @ cs_mat @ v_mat

        np.testing.assert_allclose(
            mat,
            recomputed_mat,
            atol=1e-7,
        )

    @combine(num_qubits=[1, 2, 3, 4], seed=list(range(100)))
    def test_random_unitary(self, num_qubits, seed):
        """Test CSD for random unitary matrices."""
        mat = random_unitary(2**num_qubits, seed=seed).data
        self.assertCossinDecompositionIsCorrect(mat)

    @combine(num_controls=[1, 2, 3], num_targets=[1, 2, 3], seed=list(range(50)))
    def test_controlled_random_unitary(self, num_controls, num_targets, seed):
        """Test CSD for controlled random unitary matrices."""
        # Note that scipy has numerical stability problems on controlled
        # random unitary matrices.
        base_mat = random_unitary(2**num_targets, seed=seed).data
        mat = _compute_control_matrix(base_mat, num_controls)
        self.assertCossinDecompositionIsCorrect(mat)

    @combine(num_qubits=[1, 2, 3, 4], seed=list(range(20)))
    def test_random_hermitian(self, num_qubits, seed):
        """Test CSD for random Hermitian matrices."""
        umat = random_unitary(2**num_qubits, seed=seed).data
        np.random.seed(seed)
        dmat = np.diag(np.exp(1j * np.random.normal(size=2**num_qubits)))
        mat = umat.T.conjugate() @ dmat @ umat
        self.assertCossinDecompositionIsCorrect(mat)

    @combine(num_qubits=[1, 2, 3, 4], seed=list(range(20)))
    def test_random_diagonal(self, num_qubits, seed):
        """Test CSD for random diagonal matrices."""
        np.random.seed(seed)
        dmat = np.diag(np.exp(1j * np.random.normal(size=2**num_qubits)))
        self.assertCossinDecompositionIsCorrect(dmat)
