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


"""Tests for Weyl coordinate routines."""

import unittest
import numpy as np
from numpy.testing import assert_allclose

from qiskit._accelerate.two_qubit_decompose import weyl_coordinates
from qiskit.quantum_info import random_unitary
from qiskit.synthesis.two_qubit.local_invariance import (
    two_qubit_local_invariants,
    local_equivalence,
)
from test import QiskitTestCase


class TestWeyl(QiskitTestCase):
    """Test Weyl coordinate routines"""

    def test_weyl_coordinates_simple(self):
        """Check Weyl coordinates against known cases."""
        # Identity [0,0,0]
        U = np.identity(4, dtype=complex)
        weyl = weyl_coordinates(U)
        assert_allclose(weyl, [0, 0, 0])

        # CNOT [pi/4, 0, 0]
        U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_allclose(weyl, [np.pi / 4, 0, 0], atol=1e-07)

        # SWAP [pi/4, pi/4 ,pi/4]
        U = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

        weyl = weyl_coordinates(U)
        assert_allclose(weyl, [np.pi / 4, np.pi / 4, np.pi / 4])

        # SQRT ISWAP [pi/8, pi/8, 0]
        U = np.array(
            [
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
                [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ],
            dtype=complex,
        )

        weyl = weyl_coordinates(U)
        assert_allclose(weyl, [np.pi / 8, np.pi / 8, 0])

    def test_weyl_coordinates_random(self):
        """Randomly check Weyl coordinates with local invariants."""
        for _ in range(10):
            U = random_unitary(4).data
            weyl = weyl_coordinates(U)
            local_equiv = local_equivalence(weyl)
            local = two_qubit_local_invariants(U)
            assert_allclose(local, local_equiv)


if __name__ == "__main__":
    unittest.main()
