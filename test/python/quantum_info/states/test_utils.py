# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests utility functions for QuantumState classes."""

import unittest
import numpy as np

from qiskit.quantum_info.states import Statevector, DensityMatrix
from qiskit.quantum_info.states import partial_trace, shannon_entropy, schmidt_decomposition
from test import QiskitTestCase


class TestStateUtils(QiskitTestCase):
    """Test utility functions for QuantumState classes."""

    def test_statevector_partial_trace(self):
        """Test partial_trace function on statevectors"""
        psi = Statevector.from_label("10+")
        self.assertEqual(partial_trace(psi, [0, 1]), DensityMatrix.from_label("1"))
        self.assertEqual(partial_trace(psi, [0, 2]), DensityMatrix.from_label("0"))
        self.assertEqual(partial_trace(psi, [1, 2]), DensityMatrix.from_label("+"))
        self.assertEqual(partial_trace(psi, [0]), DensityMatrix.from_label("10"))
        self.assertEqual(partial_trace(psi, [1]), DensityMatrix.from_label("1+"))
        self.assertEqual(partial_trace(psi, [2]), DensityMatrix.from_label("0+"))
        self.assertEqual(partial_trace(psi, []), DensityMatrix(psi))

    def test_density_matrix_partial_trace(self):
        """Test partial_trace function on density matrices"""
        rho = DensityMatrix.from_label("10+")
        self.assertEqual(partial_trace(rho, [0, 1]), DensityMatrix.from_label("1"))
        self.assertEqual(partial_trace(rho, [0, 2]), DensityMatrix.from_label("0"))
        self.assertEqual(partial_trace(rho, [1, 2]), DensityMatrix.from_label("+"))
        self.assertEqual(partial_trace(rho, [0]), DensityMatrix.from_label("10"))
        self.assertEqual(partial_trace(rho, [1]), DensityMatrix.from_label("1+"))
        self.assertEqual(partial_trace(rho, [2]), DensityMatrix.from_label("0+"))
        self.assertEqual(partial_trace(rho, []), rho)

    def test_shannon_entropy(self):
        """Test shannon_entropy function"""
        input_pvec = np.array([0.5, 0.3, 0.07, 0.1, 0.03])
        # Base 2
        self.assertAlmostEqual(1.7736043871504037, shannon_entropy(input_pvec))
        # Base e
        self.assertAlmostEqual(1.229368880382052, shannon_entropy(input_pvec, np.e))
        # Base 10
        self.assertAlmostEqual(0.533908120973504, shannon_entropy(input_pvec, 10))

    def test_schmidt_decomposition_separable(self):
        """Test schmidt_decomposition for separable 2-level system without subsystem permutation"""

        target = Statevector.from_label("l10")
        schmidt_comps = schmidt_decomposition(target, [0])

        # check decomposition elements
        self.assertAlmostEqual(schmidt_comps[0][0], 1)
        self.assertEqual(schmidt_comps[0][1], -1 * Statevector.from_label("l1"))
        self.assertEqual(schmidt_comps[0][2], -1 * Statevector.from_label("0"))

        # check that state can be properly reconstructed
        state = Statevector(sum(suv[0] * np.kron(suv[1], suv[2]) for suv in schmidt_comps))
        self.assertEqual(state, target)

    def test_schmidt_decomposition_separable_with_permutation(self):
        """Test schmidt_decomposition for separable 2-level system with subsystem permutation"""

        target = Statevector.from_label("0l1")
        schmidt_comps = schmidt_decomposition(Statevector.from_label("l10"), [2, 1])

        # check decomposition elements
        self.assertAlmostEqual(schmidt_comps[0][0], 1)
        self.assertEqual(schmidt_comps[0][1], Statevector.from_label("0"))
        self.assertEqual(schmidt_comps[0][2], Statevector.from_label("l1"))

        # check that state can be properly reconstructed
        state = Statevector(sum(suv[0] * np.kron(suv[1], suv[2]) for suv in schmidt_comps))
        self.assertEqual(state, target)

    def test_schmidt_decomposition_entangled(self):
        """Test schmidt_decomposition for entangled 2-level system"""

        target = np.sqrt(1 / 3) * Statevector.from_label("00") + np.sqrt(
            2 / 3
        ) * Statevector.from_label("11")
        schmidt_comps = schmidt_decomposition(target, [0])

        # check decomposition elements
        self.assertAlmostEqual(schmidt_comps[0][0], np.sqrt(2 / 3))
        self.assertEqual(schmidt_comps[0][1], Statevector.from_label("1"))
        self.assertEqual(schmidt_comps[0][2], Statevector.from_label("1"))
        self.assertAlmostEqual(schmidt_comps[1][0], np.sqrt(1 / 3))
        self.assertEqual(schmidt_comps[1][1], Statevector.from_label("0"))
        self.assertEqual(schmidt_comps[1][2], Statevector.from_label("0"))

        # check that state can be properly reconstructed
        state = Statevector(sum(suv[0] * np.kron(suv[1], suv[2]) for suv in schmidt_comps))
        self.assertEqual(state, target)

    def test_schmidt_decomposition_3_level_system(self):
        """Test schmidt_decomposition for entangled 3-level system"""

        target = Statevector(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]) * 1 / np.sqrt(3), dims=(3, 3))
        schmidt_comps = schmidt_decomposition(target, [0])

        # check decomposition elements
        self.assertAlmostEqual(schmidt_comps[0][0], 1 / np.sqrt(3))
        self.assertEqual(schmidt_comps[0][1], Statevector(np.array([1, 0, 0]), dims=3))
        self.assertEqual(schmidt_comps[0][2], Statevector(np.array([1, 0, 0]), dims=3))
        self.assertAlmostEqual(schmidt_comps[1][0], 1 / np.sqrt(3))
        self.assertEqual(schmidt_comps[1][1], Statevector(np.array([0, 1, 0]), dims=3))
        self.assertEqual(schmidt_comps[1][2], Statevector(np.array([0, 1, 0]), dims=3))
        self.assertAlmostEqual(schmidt_comps[2][0], 1 / np.sqrt(3))
        self.assertEqual(schmidt_comps[2][1], Statevector(np.array([0, 0, 1]), dims=3))
        self.assertEqual(schmidt_comps[2][2], Statevector(np.array([0, 0, 1]), dims=3))

        # check that state can be properly reconstructed
        state = Statevector(
            sum(suv[0] * np.kron(suv[1], suv[2]) for suv in schmidt_comps), dims=(3, 3)
        )
        self.assertEqual(state, target)


if __name__ == "__main__":
    unittest.main()
