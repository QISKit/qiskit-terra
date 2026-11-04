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


"""Tests for Operator matrix linear operator class."""

import unittest
import logging

import numpy as np

from qiskit.quantum_info.operators.predicates import is_identity_matrix

from test import QiskitTestCase

logger = logging.getLogger(__name__)


class PredicatesTestCase(QiskitTestCase):
    """Test methods in predicates"""

    def test_is_identity_matrix(self):
        """Test is_identity_matrix for various sizes."""
        for size in [1, 2, 4, 20]:
            self.assertTrue(is_identity_matrix(np.eye(size)))
            self.assertTrue(is_identity_matrix(np.eye(size, dtype=complex)))


if __name__ == "__main__":
    unittest.main()
