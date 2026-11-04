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
Tests equivalence of the default and fast gradient computation routines.
"""

import unittest
from time import perf_counter
import numpy as np

from qiskit.synthesis.unitary.aqc.fast_gradient.fast_gradient import FastCNOTUnitObjective
from qiskit.synthesis.unitary.aqc.cnot_unit_objective import DefaultCNOTUnitObjective
from test import QiskitTestCase
from test.python.synthesis.aqc.fast_gradient.utils_for_testing import (
    rand_circuit,
    rand_su_mat,
)


class TestCompareGradientImpls(QiskitTestCase):
    """
    Tests equivalence of the default and fast gradient implementations.
    """

    max_num_qubits = 3  # maximum number of qubits in tests
    max_depth = 10  # maximum circuit depth in tests

    def setUp(self):
        super().setUp()
        np.random.seed(0x0696969)

    def _compare(
        self, num_qubits: int, depth: int
    ) -> tuple[int, int, float, float, float, float, float]:
        """
        Calculates gradient and objective function value for the original
        and the fast implementations, and compares the outputs from both.
        Returns relative errors. Also, accumulates performance metrics.
        """

        cnots = rand_circuit(num_qubits=num_qubits, depth=depth)
        depth = cnots.shape[1]  # might be less than initial depth
        u_mat = rand_su_mat(2**num_qubits)
        dflt_obj = DefaultCNOTUnitObjective(num_qubits=num_qubits, cnots=cnots)
        fast_obj = FastCNOTUnitObjective(num_qubits=num_qubits, cnots=cnots)
        thetas = np.random.rand(4 * depth + 3 * num_qubits) * 2 * np.pi
        thetas = thetas.astype(np.float64)

        dflt_obj.target_matrix = u_mat
        fast_obj.target_matrix = u_mat

        # Compute fast gradient.
        start = perf_counter()
        f1 = fast_obj.objective(param_values=thetas)
        g1 = fast_obj.gradient(param_values=thetas)
        t1 = perf_counter() - start

        # Compute default gradient.
        start = perf_counter()
        f2 = dflt_obj.objective(param_values=thetas)
        g2 = dflt_obj.gradient(param_values=thetas)
        t2 = perf_counter() - start

        fobj_rel_err = abs(f1 - f2) / f2
        grad_rel_err = np.linalg.norm(g1 - g2) / np.linalg.norm(g2)
        speedup = t2 / max(t1, np.finfo(np.float64).eps ** 2)

        tol = float(np.sqrt(np.finfo(np.float64).eps))
        self.assertLess(fobj_rel_err, tol)
        self.assertLess(grad_rel_err, tol)
        self.assertTrue(np.allclose(g1, g2, atol=tol, rtol=tol))
        return int(num_qubits), int(depth), fobj_rel_err, grad_rel_err, speedup, t1, t2

    def test_cmp_gradients(self):
        """
        Tests equivalence of gradients.
        """

        # Configurations of the number of qubits and depths we want to try.
        configs = [
            (n, depth)
            for n in range(2, self.max_num_qubits + 1)
            for depth in np.sort(
                np.random.permutation(
                    np.arange(3 if n <= 3 else 7, 9 if n <= 3 else self.max_depth)
                )[0:10]
            )
        ]

        results = []

        # Run the tests sequentially.
        for nqubits, depth in configs:
            results.append(self._compare(nqubits, depth))

        tol = float(np.sqrt(np.finfo(float).eps))
        for _, _, ferr, gerr, _, _, _ in results:
            self.assertTrue(ferr < tol)
            self.assertTrue(gerr < tol)


if __name__ == "__main__":
    unittest.main()
