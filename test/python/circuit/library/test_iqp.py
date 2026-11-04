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

"""Test library of IQP circuits."""

import unittest
from ddt import ddt, data
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import IQP, iqp, random_iqp
from qiskit.quantum_info import Operator
from test import QiskitTestCase


@ddt
class TestIQPLibrary(QiskitTestCase):
    """Test library of IQP quantum circuits."""

    @data(True, False)
    def test_iqp(self, use_function):
        """Test iqp circuit.

             ┌───┐                   ┌─────────┐  ┌───┐
        q_0: ┤ H ├─■─────────■───────┤ P(3π/2) ├──┤ H ├────────
             ├───┤ │P(5π/2)  │       └─────────┘ ┌┴───┴─┐ ┌───┐
        q_1: ┤ H ├─■─────────┼─────────■─────────┤ P(π) ├─┤ H ├
             ├───┤           │P(π/2)   │P(3π/2) ┌┴──────┴┐├───┤
        q_2: ┤ H ├───────────■─────────■────────┤ P(π/2) ├┤ H ├
             └───┘                              └────────┘└───┘
        """

        interactions = np.array([[6, 5, 1], [5, 4, 3], [1, 3, 2]])

        if use_function:
            circuit = iqp(interactions)
        else:
            with self.assertWarns(DeprecationWarning):
                circuit = IQP(interactions)

        expected = QuantumCircuit(3)
        expected.h([0, 1, 2])
        expected.cp(5 * np.pi / 2, 0, 1)
        expected.cp(3 * np.pi / 2, 1, 2)
        expected.cp(1 * np.pi / 2, 0, 2)
        expected.p(6 * np.pi / 4, 0)
        expected.p(4 * np.pi / 4, 1)
        expected.p(2 * np.pi / 4, 2)
        expected.h([0, 1, 2])
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))

    @data(True, False)
    def test_iqp_bad(self, use_function):
        """Test an error is raised if the interactions matrix is not symmetric."""
        interactions = [[6, 5], [2, 4]]
        with self.assertRaises(CircuitError):
            if use_function:
                _ = iqp(interactions)
            else:
                with self.assertWarns(DeprecationWarning):
                    _ = IQP(interactions)

    def test_random_iqp(self):
        """Test generating a random IQP circuit."""

        circuit = random_iqp(num_qubits=4, seed=426)
        self.assertEqual(circuit.num_qubits, 4)

        ops = circuit.count_ops()
        allowed = {"p", "h", "cp"}

        # we pick a seed where neither the diagonal, nor the off-diagonal is completely 0,
        # therefore each gate is expected to be present
        self.assertEqual(set(ops.keys()), allowed)

    def test_random_iqp_seed(self):
        """Test setting the seed."""

        seed = 236321
        circuit1 = random_iqp(num_qubits=3, seed=seed)
        circuit2 = random_iqp(num_qubits=3, seed=seed)
        self.assertEqual(circuit1, circuit2)

        circuit3 = random_iqp(num_qubits=3, seed=seed + 1)
        self.assertNotEqual(circuit1, circuit3)


if __name__ == "__main__":
    unittest.main()
