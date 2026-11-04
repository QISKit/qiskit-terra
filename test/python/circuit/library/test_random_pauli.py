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

"""Test library of quantum volume circuits."""

import unittest
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliTwoDesign
from test import QiskitTestCase


class TestPauliTwoDesign(QiskitTestCase):
    """Test the Random Pauli circuit."""

    def test_random_pauli(self):
        """Test the Random Pauli circuit."""
        with self.assertWarns(DeprecationWarning):
            circuit = PauliTwoDesign(4, seed=12, reps=1)

        qr = QuantumRegister(4, "q")
        params = circuit.ordered_parameters

        # expected circuit for the random seed 12
        #
        #      ┌─────────┐┌──────────┐   ┌──────────┐
        # q_0: ┤ Ry(π/4) ├┤ Ry(θ[0]) ├─■─┤ Rx(θ[4]) ├────────────
        #      ├─────────┤├──────────┤ │ └──────────┘┌──────────┐
        # q_1: ┤ Ry(π/4) ├┤ Rx(θ[1]) ├─■──────■──────┤ Rx(θ[5]) ├
        #      ├─────────┤├──────────┤        │      ├──────────┤
        # q_2: ┤ Ry(π/4) ├┤ Rz(θ[2]) ├─■──────■──────┤ Rx(θ[6]) ├
        #      ├─────────┤├──────────┤ │ ┌──────────┐└──────────┘
        # q_3: ┤ Ry(π/4) ├┤ Rz(θ[3]) ├─■─┤ Rx(θ[7]) ├────────────
        #      └─────────┘└──────────┘   └──────────┘
        expected = QuantumCircuit(qr)

        # initial RYs
        expected.ry(np.pi / 4, qr)

        # first random Pauli layer
        expected.ry(params[0], 0)
        expected.rx(params[1], 1)
        expected.rz(params[2], 2)
        expected.rz(params[3], 3)

        # entanglement
        expected.cz(0, 1)
        expected.cz(2, 3)
        expected.cz(1, 2)

        # second random Pauli layer
        expected.rx(params[4], 0)
        expected.rx(params[5], 1)
        expected.rx(params[6], 2)
        expected.rx(params[7], 3)

        self.assertEqual(circuit.decompose(), expected)

    def test_resize(self):
        """Test resizing the Random Pauli circuit preserves the gates."""
        with self.assertWarns(DeprecationWarning):
            circuit = PauliTwoDesign(1)
        top_gates = [instruction.operation.name for instruction in circuit.decompose().data]

        circuit.num_qubits = 3
        decomposed = circuit.decompose()
        with self.subTest("assert existing gates remain"):
            new_top_gates = []
            for instruction in decomposed:
                if instruction.qubits == (decomposed.qubits[0],):  # if top qubit
                    new_top_gates.append(instruction.operation.name)

            self.assertEqual(top_gates, new_top_gates)

    def test_assign_keeps_one_initial_layer(self):
        """Test assigning parameters does not add an additional initial layer."""
        with self.assertWarns(DeprecationWarning):
            circuit = PauliTwoDesign(2)

        values = list(range(circuit.num_parameters))

        bound0 = circuit.assign_parameters(values)
        bound1 = circuit.assign_parameters(values)
        bound2 = circuit.assign_parameters(values)

        self.assertEqual(bound0, bound1)
        self.assertEqual(bound0, bound2)


if __name__ == "__main__":
    unittest.main()
