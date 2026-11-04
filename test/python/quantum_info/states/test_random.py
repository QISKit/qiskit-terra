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

"""Test random operator functions."""

import unittest
from test import combine
from ddt import ddt
import numpy as np

from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.quantum_info import random_statevector
from qiskit.quantum_info import random_density_matrix
from test import QiskitTestCase


@ddt
class TestRandomStatevector(QiskitTestCase):
    """Testing random_unitary function."""

    @combine(dims=[(2,), (3,), (2, 2), (2, 3)])
    def test_tuple_dims(self, dims):
        """Test random_statevector is valid with dims {dims}."""
        value = random_statevector(dims)
        self.assertIsInstance(value, Statevector)
        self.assertTrue(value.is_valid())
        self.assertEqual(value.dims(), dims)

    @combine(dim=[2, 3, 4, 5])
    def test_int_dims(self, dim):
        """Test random_statevector is valid with dims {dim}."""
        value = random_statevector(dim)
        self.assertIsInstance(value, Statevector)
        self.assertTrue(value.is_valid())
        self.assertEqual(np.prod(value.dims()), dim)

    def test_fixed_seed(self):
        """Test fixing seed fixes output"""
        seed = 1532
        value1 = random_statevector(4, seed=seed)
        value2 = random_statevector(4, seed=seed)
        self.assertEqual(value1, value2)

    def test_not_global_seed(self):
        """Test fixing seed is locally scoped."""
        seed = 314159
        test_cases = 100
        random_statevector(2, seed=seed)
        rng_before = np.random.randint(1000, size=test_cases)
        random_statevector(2, seed=seed)
        rng_after = np.random.randint(1000, size=test_cases)
        self.assertFalse(np.all(rng_before == rng_after))


@ddt
class TestRandomDensityMatrix(QiskitTestCase):
    """Testing random_density_matrix function."""

    @combine(dims=[(2,), (3,), (2, 2), (2, 3)], method=["Hilbert-Schmidt", "Bures"])
    def test_tuple_dims(self, dims, method):
        """Test random_density_matrix {method} method is valid with dims {dims}."""
        value = random_density_matrix(dims, method=method)
        self.assertIsInstance(value, DensityMatrix)
        self.assertTrue(value.is_valid())
        self.assertEqual(value.dims(), dims)

    @combine(dim=[2, 3, 4, 5], method=["Hilbert-Schmidt", "Bures"])
    def test_int_dims(self, dim, method):
        """Test random_density_matrix {method} method is valid with dims {dim}."""
        value = random_density_matrix(dim, method=method)
        self.assertIsInstance(value, DensityMatrix)
        self.assertTrue(value.is_valid())
        self.assertEqual(np.prod(value.dims()), dim)

    @combine(method=["Hilbert-Schmidt", "Bures"])
    def test_fixed_seed(self, method):
        """Test fixing seed fixes output ({method} method)"""
        seed = 1532
        value1 = random_density_matrix(4, method=method, seed=seed)
        value2 = random_density_matrix(4, method=method, seed=seed)
        self.assertEqual(value1, value2)

    @combine(method=["Hilbert-Schmidt", "Bures"])
    def test_not_global_seed(self, method):
        """Test fixing seed is locally scoped ({method} method)."""
        seed = 314159
        test_cases = 100
        random_density_matrix(2, method=method, seed=seed)
        rng_before = np.random.randint(1000, size=test_cases)
        random_density_matrix(2, method=method, seed=seed)
        rng_after = np.random.randint(1000, size=test_cases)
        self.assertFalse(np.all(rng_before == rng_after))


if __name__ == "__main__":
    unittest.main()
