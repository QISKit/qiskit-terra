# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Unit tests for PrimitiveResult."""

import numpy as np

from qiskit.primitives.containers import DataBin, PrimitiveResult, PubResult
from test import QiskitTestCase


class PrimitiveResultCase(QiskitTestCase):
    """Test the PrimitiveResult class."""

    def test_primitive_result(self):
        """Test the PrimitiveResult class."""
        alpha = np.empty((10, 20), dtype=np.uint16)
        beta = np.empty((10, 20), dtype=int)

        pub_results = [
            PubResult(DataBin(alpha=alpha, beta=beta, shape=(10, 20))),
            PubResult(DataBin(alpha=alpha, beta=beta, shape=(10, 20))),
        ]
        result = PrimitiveResult(pub_results, {"x": 2})

        self.assertTrue(result[0] is pub_results[0])
        self.assertTrue(result[1] is pub_results[1])
        self.assertTrue(list(result)[0] is pub_results[0])
        self.assertEqual(len(result), 2)
        self.assertEqual(result.metadata, {"x": 2})
