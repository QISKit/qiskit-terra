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


"""
Tests for uniformly controlled single-qubit unitaries.
"""

import unittest
from ddt import ddt
from test import combine
import numpy as np
from scipy.linalg import block_diag

from qiskit.circuit.library.generalized_gates import UCGate
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import random_unitary
from qiskit.compiler import transpile
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info import Operator
from test import QiskitTestCase

_id = np.eye(2, 2)
_not = np.matrix([[0, 1], [1, 0]])
_had = 1 / np.sqrt(2) * np.matrix([[1, 1], [1, -1]])
_rand = random_unitary(2, seed=541234).data


@ddt
class TestUCGate(QiskitTestCase):
    """Qiskit UCGate tests."""

    @combine(
        squs=[
            [_not],
            [_id],
            [_id, _id],
            [_id, 1j * _id],
            [_id, _not, _id, _not],
            [_rand, _had, _rand, _had, _rand, _had, _rand, _had],
            [_had, _had, _had, _had, _had, _had, _had, _had],
            [random_unitary(2, seed=541234).data for _ in range(2**2)],
            [random_unitary(2, seed=975163).data for _ in range(2**3)],
            [random_unitary(2, seed=629462).data for _ in range(2**4)],
        ],
        up_to_diagonal=[True, False],
        mux_simp=[True, False],
    )
    def test_ucg(self, squs, up_to_diagonal, mux_simp):
        """Test uniformly controlled gates."""
        num_con = int(np.log2(len(squs)))
        q = QuantumRegister(num_con + 1)
        qc = QuantumCircuit(q)

        uc = UCGate(squs, up_to_diagonal=up_to_diagonal, mux_simp=mux_simp)
        qc.append(uc, q)

        # Decompose the gate
        qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"])

        # Simulate the decomposed gate
        unitary = Operator(qc).data
        if up_to_diagonal:
            ucg = UCGate(squs, up_to_diagonal=up_to_diagonal, mux_simp=mux_simp)
            diag = np.diagflat(ucg._get_diagonal())
            unitary = np.dot(diag, unitary)

        unitary_desired = _get_ucg_matrix(squs)
        self.assertTrue(
            matrix_equal(unitary_desired, unitary, ignore_phase=True),
            f"{unitary_desired}\ndoes not equal\n{unitary}",
        )

    def test_global_phase_ucg(self):
        """Test global phase of uniformly controlled gates"""
        gates = [random_unitary(2).data for _ in range(2**2)]
        num_con = int(np.log2(len(gates)))
        q = QuantumRegister(num_con + 1)
        qc = QuantumCircuit(q)

        uc = UCGate(gates, up_to_diagonal=False)
        qc.append(uc, q)

        unitary = Operator(qc).data
        unitary_desired = _get_ucg_matrix(gates)

        self.assertTrue(np.allclose(unitary_desired, unitary))

    def test_inverse_ucg(self):
        """Test inverse function of uniformly controlled gates"""
        gates = [random_unitary(2, seed=42 + s).data for s in range(2**2)]
        num_con = int(np.log2(len(gates)))
        q = QuantumRegister(num_con + 1)
        qc = QuantumCircuit(q)

        uc = UCGate(gates, up_to_diagonal=False)
        qc.append(uc, q)
        qc.append(qc.inverse(), qc.qubits)

        unitary = Operator(qc).data
        unitary_desired = np.identity(2**qc.num_qubits)

        self.assertTrue(np.allclose(unitary_desired, unitary))

    def test_ucge(self):
        """test ucg simplification"""
        gate_list = [_had, _had, _had, _had, _had, _had, _had, _had]

        qc1 = QuantumCircuit(4)
        uc1 = UCGate(gate_list, up_to_diagonal=False, mux_simp=False)
        qc1.append(uc1, range(4))
        op1 = Operator(qc1).data

        qc2 = QuantumCircuit(4)
        uc2 = UCGate(gate_list, up_to_diagonal=False, mux_simp=True)
        qc2.append(uc2, range(4))
        op2 = Operator(qc2).data
        self.assertTrue(np.allclose(op1, op2))

    def test_repeat(self):
        """test repeat operation"""
        gates = [random_unitary(2, seed=seed).data for seed in [124435, 876345, 687462, 928365]]

        uc = UCGate(gates, up_to_diagonal=False)
        self.assertTrue(np.allclose(Operator(uc.repeat(2)), Operator(uc) @ Operator(uc)))


def _get_ucg_matrix(squs):
    return block_diag(*squs)


if __name__ == "__main__":
    unittest.main()
