# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for Layer2Q implementation.
"""

import unittest
from random import randint
import numpy as np

import qiskit.synthesis.unitary.aqc.fast_gradient.layer as lr
from qiskit.synthesis.unitary.aqc.fast_gradient.pmatrix import PMatrix
from test import QiskitTestCase
import test.python.synthesis.aqc.fast_gradient.utils_for_testing as tut


class TestLayer2q(QiskitTestCase):
    """
    Tests for Layer2Q class.
    """

    max_num_qubits = 5  # maximum number of qubits in tests
    num_repeats = 50  # number of repetitions in tests

    def setUp(self):
        super().setUp()
        np.random.seed(0x0696969)

    def test_layer2q_matrix(self):
        """
        Tests: (1) the correctness of Layer2Q matrix construction;
        (2) matrix multiplication interleaved with permutations.
        """

        mat_kind = "complex"
        _eps = 100.0 * np.finfo(float).eps
        max_rel_err = 0.0
        for n in range(2, self.max_num_qubits + 1):

            dim = 2**n
            iden = tut.eye_int(n)
            for j in range(n):
                for k in range(n):
                    if j == k:
                        continue
                    m_mat = tut.rand_matrix(dim=dim, kind=mat_kind)
                    t_mat, g_mat = tut.make_test_matrices4x4(n=n, j=j, k=k, kind=mat_kind)
                    lmat = lr.Layer2Q(num_qubits=n, j=j, k=k, g4x4=g_mat)
                    g2, perm, inv_perm = lmat.get_attr()
                    self.assertTrue(m_mat.dtype == t_mat.dtype == g_mat.dtype == g2.dtype)
                    self.assertTrue(np.all(g_mat == g2))
                    self.assertTrue(np.all(iden[perm].T == iden[inv_perm]))

                    g_mat = np.kron(tut.eye_int(n - 2), g_mat)

                    # T == P^t @ G @ P.
                    err = tut.relative_error(t_mat, iden[perm].T @ g_mat @ iden[perm])
                    self.assertLess(err, _eps, f"err = {err:0.16f}")
                    max_rel_err = max(max_rel_err, err)

                    # Multiplication by permutation matrix of the left can be
                    # replaced by row permutations.
                    tm = t_mat @ m_mat
                    err1 = tut.relative_error(iden[perm].T @ g_mat @ m_mat[perm], tm)
                    err2 = tut.relative_error((g_mat @ m_mat[perm])[inv_perm], tm)

                    # Multiplication by permutation matrix of the right can be
                    # replaced by column permutations.
                    mt = m_mat @ t_mat
                    err3 = tut.relative_error(m_mat @ iden[perm].T @ g_mat @ iden[perm], mt)
                    err4 = tut.relative_error((m_mat[:, perm] @ g_mat)[:, inv_perm], mt)

                    self.assertTrue(
                        err1 < _eps and err2 < _eps and err3 < _eps and err4 < _eps,
                        f"err1 = {err1:f},  err2 = {err2:f},  "
                        f"err3 = {err3:f},  err4 = {err4:f}",
                    )
                    max_rel_err = max(max_rel_err, err1, err2, err3, err4)

    def test_pmatrix_class(self):
        """
        Test the class PMatrix.
        """

        _eps = 100.0 * np.finfo(float).eps
        mat_kind = "complex"
        max_rel_err = 0.0
        for n in range(2, self.max_num_qubits + 1):

            dim = 2**n
            tmp1 = np.ndarray((dim, dim), dtype=np.complex128)
            tmp2 = tmp1.copy()
            for _ in range(self.num_repeats):
                j0 = randint(0, n - 1)
                k0 = (j0 + randint(1, n - 1)) % n
                j1 = randint(0, n - 1)
                k1 = (j1 + randint(1, n - 1)) % n
                j2 = randint(0, n - 1)
                k2 = (j2 + randint(1, n - 1)) % n
                j3 = randint(0, n - 1)
                k3 = (j3 + randint(1, n - 1)) % n
                j4 = randint(0, n - 1)
                k4 = (j4 + randint(1, n - 1)) % n

                t0, g0 = tut.make_test_matrices4x4(n=n, j=j0, k=k0, kind=mat_kind)
                t1, g1 = tut.make_test_matrices4x4(n=n, j=j1, k=k1, kind=mat_kind)
                t2, g2 = tut.make_test_matrices4x4(n=n, j=j2, k=k2, kind=mat_kind)
                t3, g3 = tut.make_test_matrices4x4(n=n, j=j3, k=k3, kind=mat_kind)
                t4, g4 = tut.make_test_matrices4x4(n=n, j=j4, k=k4, kind=mat_kind)

                c0 = lr.Layer2Q(num_qubits=n, j=j0, k=k0, g4x4=g0)
                c1 = lr.Layer2Q(num_qubits=n, j=j1, k=k1, g4x4=g1)
                c2 = lr.Layer2Q(num_qubits=n, j=j2, k=k2, g4x4=g2)
                c3 = lr.Layer2Q(num_qubits=n, j=j3, k=k3, g4x4=g3)
                c4 = lr.Layer2Q(num_qubits=n, j=j4, k=k4, g4x4=g4)

                m_mat = tut.rand_matrix(dim=dim, kind=mat_kind)
                ttmtt = t0 @ t1 @ m_mat @ np.conj(t2).T @ np.conj(t3).T

                pmat = PMatrix(n)
                pmat.set_matrix(m_mat)
                pmat.mul_left_q2(layer=c1, temp_mat=tmp1)
                pmat.mul_left_q2(layer=c0, temp_mat=tmp1)
                pmat.mul_right_q2(layer=c2, temp_mat=tmp1)
                pmat.mul_right_q2(layer=c3, temp_mat=tmp1)
                alt_ttmtt = pmat.finalize(temp_mat=tmp1)

                err1 = tut.relative_error(alt_ttmtt, ttmtt)
                self.assertLess(err1, _eps, f"relative error: {err1:f}")

                prod = np.complex128(np.trace(ttmtt @ t4))
                alt_prod = pmat.product_q2(layer=c4, tmp1=tmp1, tmp2=tmp2)
                err2 = abs(alt_prod - prod) / abs(prod)
                self.assertLess(err2, _eps, f"relative error: {err2:f}")

                max_rel_err = max(max_rel_err, err1, err2)


if __name__ == "__main__":
    unittest.main()
