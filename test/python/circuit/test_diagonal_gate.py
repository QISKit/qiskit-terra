# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Diagonal gate tests."""

import unittest
import numpy as np

from qiskit import QuantumCircuit
from qiskit import QiskitError
from qiskit.compiler import transpile
from qiskit.circuit.library.generalized_gates import DiagonalGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info import Operator
from test import QiskitTestCase


class TestDiagonalGate(QiskitTestCase):
    """
    Diagonal gate tests.
    """

    def test_diag_gate(self):
        """Test diagonal gates."""
        for phases in [
            [0, 0],
            [0, 0.8],
            [0, 0, 1, 1],
            [0, 1, 0.5, 1],
            (2 * np.pi * np.random.rand(2**3)).tolist(),
            (2 * np.pi * np.random.rand(2**4)).tolist(),
            (2 * np.pi * np.random.rand(2**5)).tolist(),
        ]:
            with self.subTest(phases=phases):
                diag = [np.exp(1j * ph) for ph in phases]
                num_qubits = int(np.log2(len(diag)))
                qc = QuantumCircuit(num_qubits)
                gate = DiagonalGate(diag)
                qc.append(gate, qc.qubits)

                # Decompose the gate
                qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"], optimization_level=0)
                # Simulate the decomposed gate
                unitary = Operator(qc)
                unitary_desired = _get_diag_gate_matrix(diag)
                self.assertTrue(matrix_equal(unitary, unitary_desired, ignore_phase=False))

    def test_mod1_entries(self):
        """Test that diagonal raises if entries do not have modules of 1."""
        from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT

        with self.assertRaises(QiskitError):
            DiagonalGate([1, 1 - 2 * ATOL_DEFAULT - RTOL_DEFAULT])

    def test_npcomplex_params_conversion(self):
        """Verify diagonal gate converts numpy.complex to complex."""
        # ref: https://github.com/Qiskit/qiskit-aer/issues/696
        diag = np.array([1 + 0j, 1 + 0j])
        qc = QuantumCircuit(1)
        gate = DiagonalGate(diag)
        qc.append(gate, [0])

        params = qc.data[0].operation.params
        self.assertTrue(
            all(isinstance(p, complex) and not isinstance(p, np.number) for p in params)
        )

    def test_repeat(self):
        """Test the repeat() method."""
        for phases in [
            [0, 0],
            np.array([0, 0.8, 1, 0]),
            (2 * np.pi * np.random.rand(2**3)).tolist(),
        ]:
            with self.subTest(phases=phases):
                diag = [np.exp(1j * ph) for ph in phases]
                gate = DiagonalGate(diag)
                operator = Operator(gate)
                self.assertTrue(np.allclose(Operator(gate.repeat(2)), operator @ operator))


def _get_diag_gate_matrix(diag):
    return np.diagflat(diag)


if __name__ == "__main__":
    unittest.main()
