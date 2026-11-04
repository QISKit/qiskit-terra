# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tests building up CNOT unit structures.
"""

import numpy as np
from ddt import ddt, data, unpack

from qiskit.synthesis.unitary.aqc import make_cnot_network
from test import QiskitTestCase
from test.python.synthesis.aqc.sample_data import (
    CARTAN_4,
    CARTAN_3,
)


@ddt
class TestCNOTNetworks(QiskitTestCase):
    """Tests constructing various CNOT structures."""

    @data(
        # number of qubits, network layout, connectivity, depth, output
        (3, "spin", "full", 3, [[0, 1, 0], [1, 2, 1]]),
        (4, "spin", "full", 4, [[0, 2, 1, 0], [1, 3, 2, 1]]),
        (3, "sequ", "full", 4, [[0, 0, 1, 0], [1, 2, 2, 1]]),
        (4, "sequ", "full", 7, [[0, 0, 0, 1, 1, 2, 0], [1, 2, 3, 2, 3, 3, 1]]),
        (3, "sequ", "line", 3, [[0, 1, 0], [1, 2, 1]]),
        (4, "sequ", "line", 4, [[0, 1, 2, 0], [1, 2, 3, 1]]),
        (3, "sequ", "star", 3, [[0, 0, 0], [1, 2, 1]]),
        (4, "sequ", "star", 4, [[0, 0, 0, 0], [1, 2, 3, 1]]),
        (3, "cart", "full", 0, CARTAN_3),
        (4, "cart", "full", 0, CARTAN_4),
        (3, "cyclic_spin", "full", 3, [[0, 1, 0], [1, 2, 1]]),
        (4, "cyclic_spin", "full", 5, [[0, 2, 1, 3, 0], [1, 3, 2, 0, 1]]),
        (3, "cyclic_line", "line", 4, [[0, 1, 2, 0], [1, 2, 0, 1]]),
        (4, "cyclic_line", "line", 5, [[0, 1, 2, 3, 0], [1, 2, 3, 0, 1]]),
    )
    @unpack
    def test_basic_networks(self, num_qubits, network_layout, connectivity, depth, output):
        """Tests basic CNOT networks."""
        cnots = make_cnot_network(
            num_qubits=num_qubits,
            network_layout=network_layout,
            connectivity_type=connectivity,
            depth=depth,
        )
        np.testing.assert_array_equal(cnots, output)
