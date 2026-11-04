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

"""Test the diagonal circuit."""

import unittest
from ddt import ddt, data
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info.operators.predicates import matrix_equal
from test import QiskitTestCase


@ddt
class TestDiagonalGate(QiskitTestCase):
    """Test diagonal circuit."""

    @data(
        [0, 0],
        [0, 0.8],
        [0, 0, 1, 1],
        [0, 1, 0.5, 1],
        (2 * np.pi * np.random.rand(2**3)),
        (2 * np.pi * np.random.rand(2**4)),
        (2 * np.pi * np.random.rand(2**5)),
    )
    def test_diag_gate(self, phases):
        """Test correctness of diagonal decomposition."""
        diag = [np.exp(1j * ph) for ph in phases]
        num_qubits = int(np.log2(len(phases)))
        qc = QuantumCircuit(num_qubits)
        qc.append(DiagonalGate(diag), range(num_qubits))
        simulated_diag = Statevector(Operator(qc).data.diagonal()).data
        ref_diag = Statevector(diag).data

        self.assertTrue(matrix_equal(simulated_diag, ref_diag, ignore_phase=False))


if __name__ == "__main__":
    unittest.main()
