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

"""Tests for functions used in state visualization"""

import unittest
import numpy as np

from qiskit.quantum_info import Statevector
from qiskit.visualization.state_visualization import _paulivec_data
from test import QiskitTestCase


class TestStatePlotTools(QiskitTestCase):
    """State Plotting Tools"""

    def test_state_paulivec(self):
        """Test paulivec."""

        sv = Statevector.from_label("+-rl")
        output = _paulivec_data(sv)
        labels = [
            "IIII",
            "IIIY",
            "IIYI",
            "IIYY",
            "IXII",
            "IXIY",
            "IXYI",
            "IXYY",
            "XIII",
            "XIIY",
            "XIYI",
            "XIYY",
            "XXII",
            "XXIY",
            "XXYI",
            "XXYY",
        ]
        values = [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1]
        self.assertEqual(output[0], labels)
        self.assertTrue(np.allclose(output[1], values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
