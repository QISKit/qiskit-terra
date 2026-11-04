# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Tests for local invariance routines."""

import unittest
from numpy.testing import assert_allclose

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.synthesis.two_qubit.local_invariance import two_qubit_local_invariants
from qiskit.quantum_info import Operator
from test import QiskitTestCase


class TestLocalInvariance(QiskitTestCase):
    """Test local invariance routines"""

    def test_2q_local_invariance_simple(self):
        """Check the local invariance parameters
        for known simple cases.
        """

        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)
        U = Operator(qc)
        vec = two_qubit_local_invariants(U)
        assert_allclose(vec, [1, 0, 3])

        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)
        qc.cx(qr[1], qr[0])
        U = Operator(qc)
        vec = two_qubit_local_invariants(U)
        assert_allclose(vec, [0, 0, 1])

        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)
        qc.cx(qr[1], qr[0])
        qc.cx(qr[0], qr[1])
        U = Operator(qc)
        vec = two_qubit_local_invariants(U)
        assert_allclose(vec, [0, 0, -1])

        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)
        qc.swap(qr[1], qr[0])
        U = Operator(qc)
        vec = two_qubit_local_invariants(U)
        assert_allclose(vec, [-1, 0, -3])


if __name__ == "__main__":
    unittest.main()
