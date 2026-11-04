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

"""Test library of weighted adder circuits."""

import unittest
from collections import defaultdict
from ddt import ddt, data
import numpy as np

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import WeightedAdder, WeightedSumGate
from qiskit.quantum_info import Statevector
from qiskit.transpiler.passes import HighLevelSynthesis
from test import QiskitTestCase


@ddt
class TestWeightedAdder(QiskitTestCase):
    """Test the weighted adder circuit."""

    def assertSummationIsCorrect(self, adder, num_ancillas=None):
        """Assert that ``adder`` correctly implements the summation w.r.t. its set weights."""
        num_state_qubits = adder.num_state_qubits
        num_sum_qubits = adder.num_sum_qubits

        if isinstance(adder, QuantumCircuit):
            num_ancillas = adder.num_ancillas
            weights = adder.weights
        else:
            weights = adder.params

        circuit = QuantumCircuit(num_state_qubits + num_sum_qubits + num_ancillas)
        circuit.h(list(range(num_state_qubits)))
        circuit.append(adder, list(range(adder.num_qubits)))

        tqc = transpile(circuit, basis_gates=["u", "cx"])
        statevector = Statevector(tqc)

        probabilities = defaultdict(float)
        for i, statevector_amplitude in enumerate(statevector):
            i = bin(i)[2:].zfill(circuit.num_qubits)[num_ancillas:]
            probabilities[i] += np.real(np.abs(statevector_amplitude) ** 2)

        expectations = defaultdict(float)
        for x in range(2**num_state_qubits):
            bits = np.array(list(bin(x)[2:].zfill(num_state_qubits)), dtype=int)
            summation = bits.dot(weights[::-1])

            entry = bin(summation)[2:].zfill(num_sum_qubits) + bin(x)[2:].zfill(num_state_qubits)
            expectations[entry] = 1 / 2**num_state_qubits

        for state, probability in probabilities.items():
            self.assertAlmostEqual(probability, expectations[state])

    @data([0], [1, 2, 1], [4], [1, 2, 1, 1, 4])
    def test_summation(self, weights):
        """Test the weighted adder on some examples."""
        for use_gate in [False, True]:
            if use_gate:
                adder = WeightedSumGate(len(weights), weights)
                num_ancillas = adder.num_sum_qubits - 1
                num_ancillas += int(adder.num_sum_qubits > 2)
            else:
                with self.assertWarns(DeprecationWarning):
                    adder = WeightedAdder(len(weights), weights)
                num_ancillas = None

            with self.subTest(use_gate=use_gate):
                self.assertSummationIsCorrect(adder, num_ancillas)

    def test_too_few_aux(self):
        """Test a warning if raised if there are not sufficient auxiliary qubits present."""
        adder = WeightedSumGate(3, [1, 0, 1])
        circuit = QuantumCircuit(adder.num_qubits)  # would need an additional auxiliary qubit
        circuit.append(adder, circuit.qubits)

        with self.assertWarnsRegex(
            UserWarning, expected_regex="Cannot synthesize a WeightedSumGate on 3 state qubits"
        ):
            _ = HighLevelSynthesis()(circuit)

    def test_mutability(self):
        """Test the mutability of the weighted adder."""
        with self.assertWarns(DeprecationWarning):
            adder = WeightedAdder()

        with self.subTest(msg="missing number of state qubits"):
            with self.assertRaises(AttributeError):
                _ = str(adder.draw())

        with self.subTest(msg="default weights"):
            adder.num_state_qubits = 3
            default_weights = 3 * [1]
            self.assertListEqual(adder.weights, default_weights)

        with self.subTest(msg="specify weights"):
            adder.weights = [3, 2, 1]
            self.assertSummationIsCorrect(adder)

        with self.subTest(msg="mismatching number of state qubits and weights"):
            with self.assertRaises(ValueError):
                adder.weights = [0, 1, 2, 3]
                _ = str(adder.draw())

        with self.subTest(msg="change all attributes"):
            adder.num_state_qubits = 4
            adder.weights = [2, 0, 1, 1]
            self.assertSummationIsCorrect(adder)


if __name__ == "__main__":
    unittest.main()
