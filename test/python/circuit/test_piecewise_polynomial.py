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

"""Test the piecewise polynomial Pauli rotations."""

import unittest
from collections import defaultdict
import numpy as np
from ddt import ddt, data, unpack

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.arithmetic.piecewise_polynomial_pauli_rotations import (
    PiecewisePolynomialPauliRotations,
)
from qiskit.quantum_info import Statevector
from test import QiskitTestCase


@ddt
class TestPiecewisePolynomialRotations(QiskitTestCase):
    """Test the piecewise polynomial Pauli rotations."""

    def assertFunctionIsCorrect(self, function_circuit, reference):
        """Assert that ``function_circuit`` implements the reference function ``reference``."""
        num_state_qubits = function_circuit.num_state_qubits
        num_ancilla_qubits = function_circuit.num_ancillas
        circuit = QuantumCircuit(num_state_qubits + 1 + num_ancilla_qubits)
        circuit.h(list(range(num_state_qubits)))
        circuit.append(function_circuit.to_instruction(), list(range(circuit.num_qubits)))

        statevector = Statevector(circuit)

        probabilities = defaultdict(float)
        for i, statevector_amplitude in enumerate(statevector):
            i = bin(i)[2:].zfill(circuit.num_qubits)[num_ancilla_qubits:]
            probabilities[i] += np.real(np.abs(statevector_amplitude) ** 2)

        unrolled_probabilities = []
        unrolled_expectations = []
        for i, probability in probabilities.items():
            x, last_qubit = int(i[1:], 2), i[0]
            if last_qubit == "0":
                expected_amplitude = np.cos(reference(x)) / np.sqrt(2**num_state_qubits)
            else:
                expected_amplitude = np.sin(reference(x)) / np.sqrt(2**num_state_qubits)

            unrolled_probabilities += [probability]
            unrolled_expectations += [np.real(np.abs(expected_amplitude) ** 2)]

        np.testing.assert_almost_equal(unrolled_probabilities, unrolled_expectations)

    @data(
        (1, [0], [[1]]),
        (2, [0, 2], [[2], [-0.5, 1]]),
        (3, [0, 2, 5], [[1, 0, -1], [2, 1], [1, 1, 1]]),
        (4, [2, 5, 7, 16], [[1, -1], [1, 2, 3], [1, 2, 3, 4]]),
        (3, [0, 1], [[1, 0], [1, -2]]),
    )
    @unpack
    def test_piecewise_polynomial_function(self, num_state_qubits, breakpoints, coeffs):
        """Test the piecewise linear rotations."""

        def pw_poly(x):
            for i, point in enumerate(reversed(breakpoints[: len(coeffs)])):
                if x >= point:
                    # Rescale the coefficients to take into account the 2 * theta argument from the
                    # rotation gates
                    rescaled_c = [coeff / 2 for coeff in coeffs[-(i + 1)][::-1]]
                    return np.poly1d(rescaled_c)(x)
            return 0

        with self.assertWarns(DeprecationWarning):
            pw_polynomial_rotations = PiecewisePolynomialPauliRotations(
                num_state_qubits, breakpoints, coeffs
            )

        self.assertFunctionIsCorrect(pw_polynomial_rotations, pw_poly)

    def test_piecewise_polynomial_rotations_mutability(self):
        """Test the mutability of the linear rotations circuit."""

        def pw_poly(x):
            for i, point in enumerate(reversed(breakpoints[: len(coeffs)])):
                if x >= point:
                    # Rescale the coefficients to take into account the 2 * theta argument from the
                    # rotation gates
                    rescaled_c = [coeff / 2 for coeff in coeffs[-(i + 1)][::-1]]
                    return np.poly1d(rescaled_c)(x)
            return 0

        with self.assertWarns(DeprecationWarning):
            pw_polynomial_rotations = PiecewisePolynomialPauliRotations()

        with self.subTest(msg="missing number of state qubits"):
            with self.assertRaises(AttributeError):  # no state qubits set
                _ = str(pw_polynomial_rotations.draw())

        with self.subTest(msg="default setup, just setting number of state qubits"):
            pw_polynomial_rotations.num_state_qubits = 2
            self.assertFunctionIsCorrect(pw_polynomial_rotations, lambda x: 1 / 2)

        with self.subTest(msg="setting non-default values"):
            breakpoints = [0, 2]
            coeffs = [[0, -2 * 1.2], [-2 * 1, 2 * 1, 2 * 3]]
            pw_polynomial_rotations.breakpoints = breakpoints
            pw_polynomial_rotations.coeffs = coeffs
            self.assertFunctionIsCorrect(pw_polynomial_rotations, pw_poly)

        with self.subTest(msg="changing all values"):
            pw_polynomial_rotations.num_state_qubits = 4
            breakpoints = [1, 3, 6]
            coeffs = [[0, -2 * 1.2], [-2 * 1, 2 * 1, 2 * 3], [-2 * 2]]
            pw_polynomial_rotations.breakpoints = breakpoints
            pw_polynomial_rotations.coeffs = coeffs
            self.assertFunctionIsCorrect(pw_polynomial_rotations, pw_poly)


if __name__ == "__main__":
    unittest.main()
