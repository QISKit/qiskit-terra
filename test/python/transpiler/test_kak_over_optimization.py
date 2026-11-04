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

"""Test KAK over optimization"""

import unittest
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import CU1Gate
from test import QiskitTestCase


class TestKAKOverOptim(QiskitTestCase):
    """Tests to verify that KAK decomposition
    does not over optimize.
    """

    def test_cz_optimization(self):
        """Test that KAK does not run on a cz gate"""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.cz(qr[0], qr[1])

        cz_circ = transpile(
            qc,
            None,
            coupling_map=[[0, 1], [1, 0]],
            basis_gates=["u1", "u2", "u3", "id", "cx"],
            optimization_level=3,
        )
        ops = cz_circ.count_ops()
        self.assertEqual(ops["u2"], 2)
        self.assertEqual(ops["cx"], 1)
        self.assertFalse("u3" in ops.keys())

    def test_cu1_optimization(self):
        """Test that KAK does run on a cu1 gate and
        reduces the cx count from two to one.
        """
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.append(CU1Gate(np.pi), [qr[0], qr[1]])

        cu1_circ = transpile(
            qc,
            None,
            coupling_map=[[0, 1], [1, 0]],
            basis_gates=["u1", "u2", "u3", "id", "cx"],
            optimization_level=3,
        )
        ops = cu1_circ.count_ops()
        self.assertEqual(ops["cx"], 1)


if __name__ == "__main__":
    unittest.main()
