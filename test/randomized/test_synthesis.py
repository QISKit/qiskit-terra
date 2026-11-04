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

"""Randomized tests of quantum synthesis."""
import unittest
from test.python.synthesis.test_synthesis import CheckDecompositions
from hypothesis import given, strategies, settings
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.quantum_info import random_unitary
from qiskit.synthesis.two_qubit.two_qubit_decompose import (
    two_qubit_cnot_decompose,
    TwoQubitBasisDecomposer,
)
from qiskit._accelerate.two_qubit_decompose import Ud


class TestSynthesis(CheckDecompositions):
    """Test synthesis"""

    seed = strategies.integers(min_value=0, max_value=2**32 - 1)
    rotation = strategies.floats(min_value=-np.pi * 10, max_value=np.pi * 10)

    @given(seed)
    def test_1q_random(self, seed):
        """Checks one qubit decompositions"""
        unitary = random_unitary(2, seed=seed)
        self.check_one_qubit_euler_angles(unitary)
        self.check_one_qubit_euler_angles(unitary, "U3")
        self.check_one_qubit_euler_angles(unitary, "U1X")
        self.check_one_qubit_euler_angles(unitary, "PSX")
        self.check_one_qubit_euler_angles(unitary, "ZSX")
        self.check_one_qubit_euler_angles(unitary, "ZYZ")
        self.check_one_qubit_euler_angles(unitary, "ZXZ")
        self.check_one_qubit_euler_angles(unitary, "XYX")
        self.check_one_qubit_euler_angles(unitary, "RR")

    @settings(deadline=None)
    @given(seed)
    def test_2q_random(self, seed):
        """Checks two qubit decompositions"""
        unitary = random_unitary(4, seed=seed)
        self.check_exact_decomposition(unitary.data, two_qubit_cnot_decompose)

    @given(strategies.tuples(*[seed] * 5))
    def test_exact_supercontrolled_decompose_random(self, seeds):
        """Exact decomposition for random supercontrolled basis and random target"""
        k1 = np.kron(random_unitary(2, seed=seeds[0]).data, random_unitary(2, seed=seeds[1]).data)
        k2 = np.kron(random_unitary(2, seed=seeds[2]).data, random_unitary(2, seed=seeds[3]).data)
        basis_unitary = k1 @ Ud(np.pi / 4, 0, 0) @ k2
        decomposer = TwoQubitBasisDecomposer(UnitaryGate(basis_unitary))
        self.check_exact_decomposition(random_unitary(4, seed=seeds[4]).data, decomposer)

    @given(strategies.tuples(*[rotation] * 6))
    def test_cx_equivalence_0cx_random(self, rnd):
        """Check random circuits with  0 cx gates locally equivalent to identity."""
        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        unitary = Operator(qc)
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 0)

    @given(strategies.tuples(*[rotation] * 12))
    def test_cx_equivalence_1cx_random(self, rnd):
        """Check random circuits with 1 cx gates locally equivalent to a cx."""
        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[6], rnd[7], rnd[8], qr[0])
        qc.u(rnd[9], rnd[10], rnd[11], qr[1])

        unitary = Operator(qc)
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 1)

    @given(strategies.tuples(*[rotation] * 18))
    def test_cx_equivalence_2cx_random(self, rnd):
        """Check random circuits with 2 cx gates locally equivalent to some circuit with 2 cx."""
        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[6], rnd[7], rnd[8], qr[0])
        qc.u(rnd[9], rnd[10], rnd[11], qr[1])

        qc.cx(qr[0], qr[1])

        qc.u(rnd[12], rnd[13], rnd[14], qr[0])
        qc.u(rnd[15], rnd[16], rnd[17], qr[1])

        unitary = Operator(qc)
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 2)

    @given(strategies.tuples(*[rotation] * 24))
    def test_cx_equivalence_3cx_random(self, rnd):
        """Check random circuits with 3 cx gates are outside the 0, 1, and 2 qubit regions."""
        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[6], rnd[7], rnd[8], qr[0])
        qc.u(rnd[9], rnd[10], rnd[11], qr[1])

        qc.cx(qr[0], qr[1])

        qc.u(rnd[12], rnd[13], rnd[14], qr[0])
        qc.u(rnd[15], rnd[16], rnd[17], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[18], rnd[19], rnd[20], qr[0])
        qc.u(rnd[21], rnd[22], rnd[23], qr[1])

        unitary = Operator(qc)
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 3)


if __name__ == "__main__":
    unittest.main()
