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
Tests for utility functions.
"""

import unittest
import random
import numpy as np

import qiskit.synthesis.unitary.aqc.fast_gradient.fast_grad_utils as myu
from qiskit.synthesis.unitary.aqc.elementary_operations import rx_matrix as _rx
from qiskit.synthesis.unitary.aqc.elementary_operations import ry_matrix as _ry
from qiskit.synthesis.unitary.aqc.elementary_operations import rz_matrix as _rz
from test import QiskitTestCase
import test.python.synthesis.aqc.fast_gradient.utils_for_testing as tut


class TestUtils(QiskitTestCase):
    """
    Tests utility functions used by the fast gradient implementation.
    """

    max_nqubits_rev_bits = 7  # max. number of qubits in test_reverse_bits
    max_nqubits_mk_unit = 5  # max. number of qubits in test_make_unit_vector
    max_nqubits_swap = 7  # max. number of qubits in test_swap
    max_nqubits_perms = 5  # max. number of qubits in test_permutations
    num_repeats_gates2x2 = 100  # number of repetitions in test_gates2x2

    def setUp(self):
        super().setUp()
        np.random.seed(0x0696969)

    def test_reverse_bits(self):
        """
        Tests the function utils.reverse_bits().
        """

        nbits = self.max_nqubits_rev_bits
        for x in range(2**nbits):
            y = myu.reverse_bits(x, nbits, enable=True)
            self.assertTrue(x == myu.reverse_bits(y, nbits, enable=True))
            for i in range(nbits):
                self.assertTrue(((x >> i) & 1) == ((y >> (nbits - i - 1)) & 1))

        x = np.arange(2**nbits, dtype=np.int64)
        y = myu.reverse_bits(x, nbits, enable=True)
        self.assertTrue(np.all(myu.reverse_bits(y, nbits, enable=True) == x))

    def test_make_unit_vector(self):
        """
        Tests the function utils_for_testing.make_unit_vector().
        """

        # Computes unit vector as Kronecker product of (1 0) or (0 1) sub-vectors.
        def _make_unit_vector_simple(num: int, nbits: int) -> np.ndarray:
            self.assertTrue(isinstance(num, int) and isinstance(nbits, int))
            self.assertTrue(0 <= num < 2**nbits)
            vec = 1
            for i in range(nbits):
                x = (num >> i) & 1
                vec = np.kron(vec, np.asarray([1 - x, x], dtype=np.int64))
            self.assertTrue(vec.dtype == np.int64 and vec.shape == (2**nbits,))
            return vec

        for n in range(1, self.max_nqubits_mk_unit + 1):
            for k in range(2**n):
                v1 = _make_unit_vector_simple(k, n)
                v2 = tut.make_unit_vector(k, n)
                self.assertTrue(np.all(v1 == v2))

    def test_swap(self):
        """
        Tests the function utils.swap_bits().
        """

        nbits = self.max_nqubits_swap
        for x in range(2**nbits):
            for a in range(nbits):
                for b in range(nbits):
                    y = myu.swap_bits(x, a, b)
                    self.assertTrue(x == myu.swap_bits(y, a, b))
                    self.assertTrue(((x >> a) & 1) == ((y >> b) & 1))
                    self.assertTrue(((x >> b) & 1) == ((y >> a) & 1))

    def test_permutations(self):
        """
        Tests various properties of permutations and their compositions.
        """

        def _perm2_mat(perm: np.ndarray) -> np.ndarray:
            """Creates permutation matrix from a permutation."""
            return np.eye(perm.size, dtype=np.int64)[perm]

        def _permutation_and_matrix(size: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
            """Creates direct and inverse permutations and their matrices."""
            perm = np.random.permutation(np.arange(size, dtype=np.int64))
            inv_perm = myu.inverse_permutation(perm)
            p_mat, q_mat = _perm2_mat(perm), _perm2_mat(inv_perm)
            return perm, inv_perm, p_mat, q_mat

        for n in range(1, self.max_nqubits_perms + 1):

            dim = 2**n
            for _ in range(100):
                perm1, inv_perm1, p1_mat, q1_mat = _permutation_and_matrix(dim)
                perm2, inv_perm2, p2_mat, q2_mat = _permutation_and_matrix(dim)

                iden = tut.eye_int(n)
                self.assertTrue(np.all(p1_mat == q1_mat.T) and np.all(p1_mat.T == q1_mat))
                self.assertTrue(np.all(p2_mat == q2_mat.T) and np.all(p2_mat.T == q2_mat))
                self.assertTrue(np.all(p1_mat @ q1_mat == iden) and np.all(q1_mat @ p1_mat == iden))
                self.assertTrue(np.all(p2_mat @ q2_mat == iden) and np.all(q2_mat @ p2_mat == iden))

                self.assertTrue(np.all(perm2[perm1] == np.take(perm2, perm1)))
                self.assertTrue(np.all(perm1[perm2] == np.take(perm1, perm2)))

                self.assertTrue(np.all(p1_mat @ p2_mat == _perm2_mat(perm2[perm1])))
                self.assertTrue(np.all(p2_mat @ p1_mat == _perm2_mat(perm1[perm2])))

                self.assertTrue(np.all(p1_mat @ q2_mat == _perm2_mat(inv_perm2[perm1])))
                self.assertTrue(np.all(q2_mat @ p1_mat == _perm2_mat(perm1[inv_perm2])))

                self.assertTrue(np.all(p2_mat @ q1_mat == _perm2_mat(inv_perm1[perm2])))
                self.assertTrue(np.all(q1_mat @ p2_mat == _perm2_mat(perm2[inv_perm1])))

                _, inv_perm3, p3_mat, _ = _permutation_and_matrix(dim)
                perm4, _, _, q4_mat = _permutation_and_matrix(dim)

                mat = tut.rand_matrix(dim=dim, kind="randint")

                pmq = p1_mat @ mat @ q2_mat
                self.assertTrue(np.all(pmq == mat[perm1][:, perm2]))
                self.assertTrue(np.all(pmq == np.take(np.take(mat, perm1, axis=0), perm2, axis=1)))
                pmq = p1_mat @ mat @ p2_mat
                self.assertTrue(np.all(pmq == mat[perm1][:, inv_perm2]))
                self.assertTrue(
                    np.all(pmq == np.take(np.take(mat, perm1, axis=0), inv_perm2, axis=1))
                )

                ppmpq = p1_mat @ p2_mat @ mat @ p3_mat @ q4_mat
                self.assertTrue(np.all(ppmpq == mat[perm2][perm1][:, inv_perm3][:, perm4]))
                self.assertTrue(np.all(ppmpq == mat[perm2[perm1]][:, inv_perm3[perm4]]))
                self.assertTrue(
                    np.all(
                        ppmpq
                        == np.take(np.take(mat, perm2[perm1], axis=0), inv_perm3[perm4], axis=1)
                    )
                )

    def test_gates2x2(self):
        """
        Tests implementation of 2x2 gate matrices vs the ones that create and
        return temporary array upon every call (but have more compact code).
        """

        tol = np.finfo(np.float64).eps * 2.0
        out = np.full((2, 2), fill_value=0, dtype=np.complex128)
        for test in range(self.num_repeats_gates2x2):
            phi = random.random() * 2.0 * np.pi
            self.assertTrue(np.allclose(myu.make_rx(phi, out=out), _rx(phi), atol=tol, rtol=tol))
            self.assertTrue(np.allclose(myu.make_ry(phi, out=out), _ry(phi), atol=tol, rtol=tol))
            self.assertTrue(np.allclose(myu.make_rz(phi, out=out), _rz(phi), atol=tol, rtol=tol))


if __name__ == "__main__":
    unittest.main()
