# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test CZ circuits synthesis functions."""

import unittest
import numpy as np
from ddt import ddt

from qiskit import QuantumCircuit
from qiskit.circuit.library import PermutationGate
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr
from qiskit.synthesis.linear.linear_circuits_utils import check_lnn_connectivity
from qiskit.quantum_info import Clifford
from test import combine
from test import QiskitTestCase


@ddt
class TestCZSynth(QiskitTestCase):
    """Test the linear reversible circuit synthesis functions."""

    @combine(num_qubits=[3, 4, 5, 6, 7])
    def test_cz_synth_lnn(self, num_qubits):
        """Test the CZ synthesis code for linear nearest neighbor connectivity."""
        seed = 1234
        rng = np.random.default_rng(seed)
        num_gates = 10
        num_trials = 5
        for _ in range(num_trials):
            mat = np.zeros((num_qubits, num_qubits))
            qctest = QuantumCircuit(num_qubits)

            # Generate a random CZ circuit
            for _ in range(num_gates):
                i = rng.integers(num_qubits)
                j = rng.integers(num_qubits)
                if i != j:
                    qctest.cz(i, j)
                    if j > i:
                        mat[i][j] = (mat[i][j] + 1) % 2
                    else:
                        mat[j][i] = (mat[j][i] + 1) % 2

            qc = synth_cz_depth_line_mr(mat)
            # Check that the output circuit 2-qubit depth equals to 2*n+2
            depth2q = qc.depth(filter_function=lambda x: x.operation.num_qubits == 2)
            self.assertTrue(depth2q == 2 * num_qubits + 2)
            # Check that the output circuit has LNN connectivity
            self.assertTrue(check_lnn_connectivity(qc))
            # Assert that we get the same element, up to reverse order of qubits

            perm = QuantumCircuit(num_qubits)
            perm.append(PermutationGate(pattern=range(num_qubits)[::-1]), range(num_qubits))
            qctest = qctest.compose(perm)
            self.assertEqual(Clifford(qc), Clifford(qctest))

    def test_cz_synth_lnn_accepts_trivial_inputs(self):
        zero = np.zeros((0, 0), dtype=bool)
        empty = QuantumCircuit(0)
        empty.ensure_physical()
        self.assertEqual(synth_cz_depth_line_mr(zero), empty)

        one = np.zeros((1, 1), dtype=bool)
        self.assertEqual(synth_cz_depth_line_mr(one), QuantumCircuit(1))
        one = np.eye(1, dtype=bool)
        self.assertEqual(synth_cz_depth_line_mr(one), QuantumCircuit(1))

    def test_cz_synth_lnn_rejects_nonsquare(self):
        wide = np.zeros((2, 3))
        with self.assertRaisesRegex(ValueError, "matrix must be square"):
            synth_cz_depth_line_mr(wide)
        tall = np.zeros((5, 2))
        with self.assertRaisesRegex(ValueError, "matrix must be square"):
            synth_cz_depth_line_mr(tall)


if __name__ == "__main__":
    unittest.main()
