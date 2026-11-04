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

"""Test the piecewise Chebyshev approximation."""

import unittest
from collections import defaultdict
import numpy as np
from ddt import ddt, data, unpack

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import (
    PiecewiseChebyshev,
    PiecewiseChebyshevGate,
)
from qiskit.quantum_info import Statevector
from test import QiskitTestCase


@ddt
class TestPiecewiseChebyshev(QiskitTestCase):
    """Test the piecewise Chebyshev approximation."""

    def assertFunctionIsCorrect(self, function_circuit, reference, num_ancillas=None):
        """Assert that ``function_circuit`` implements the reference function ``reference``."""
        function_circuit._build()
        num_ancillas = function_circuit.num_ancillas if num_ancillas is None else num_ancillas
        num_state_qubits = function_circuit.num_qubits - num_ancillas - 1

        circuit = QuantumCircuit(num_state_qubits + 1 + num_ancillas)
        circuit.h(list(range(num_state_qubits)))
        circuit.compose(function_circuit, inplace=True)

        tqc = transpile(circuit, basis_gates=["u", "cx"])
        statevector = Statevector(tqc)
        probabilities = defaultdict(float)
        for i, statevector_amplitude in enumerate(statevector):
            i = bin(i)[2:].zfill(circuit.num_qubits)[num_ancillas:]
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

        np.testing.assert_array_almost_equal(
            unrolled_probabilities, unrolled_expectations, decimal=1
        )

    @data(
        (lambda x: np.arcsin(1 / x), 2, [2, 4], 2),
        (lambda x: x / 8, 1, [1, 8], 3),
        (np.sqrt, 2, None, 2),
    )
    @unpack
    def test_piecewise_chebyshev(self, f_x, degree, breakpoints, num_state_qubits):
        """Test the piecewise Chebyshev approximation."""

        def pw_poly(x):
            if breakpoints:
                if len(breakpoints) > 1:
                    start = breakpoints[0]
                    end = breakpoints[-1]
                else:
                    start = breakpoints[0]
                    end = 2**num_state_qubits
            else:
                start = 0
                end = 2**num_state_qubits
            if start <= x < end:
                return f_x(x)
            return np.arcsin(1)

        for use_gate in [True, False]:
            with self.subTest(use_gate=use_gate):
                if use_gate:
                    pw_approximation = PiecewiseChebyshevGate(
                        f_x, num_state_qubits, degree, breakpoints
                    )
                    num_ancillas = 0 if breakpoints is None else int(len(breakpoints) > 1)
                else:
                    with self.assertWarns(DeprecationWarning):
                        pw_approximation = PiecewiseChebyshev(
                            f_x, degree, breakpoints, num_state_qubits
                        )
                    num_ancillas = None

        self.assertFunctionIsCorrect(pw_approximation, pw_poly, num_ancillas)

    def test_piecewise_chebyshev_mutability(self):
        """Test the mutability of the piecewise Chebyshev approximation."""

        def pw_poly(x, f_x):
            if breakpoints[0] <= x < breakpoints[-1]:
                return f_x(x)
            return np.arcsin(1)

        def f_x_1(x):
            return x / 2

        with self.assertWarns(DeprecationWarning):
            pw_approximation = PiecewiseChebyshev(f_x_1)

        with self.subTest(msg="missing number of state qubits"):
            with self.assertRaises(AttributeError):  # no state qubits set
                _ = str(pw_approximation.draw())

        with self.subTest(msg="default setup, just setting number of state qubits"):
            pw_approximation.num_state_qubits = 2
            pw_approximation.f_x = f_x_1
            # set to the default breakpoints for pw_poly
            breakpoints = [0, 4]
            pw_approximation.breakpoints = breakpoints
            self.assertFunctionIsCorrect(pw_approximation, lambda x: pw_poly(x, f_x_1))

        def f_x_2(x):
            return x / 4

        with self.subTest(msg="setting non-default values"):
            breakpoints = [0, 2]
            degree = 2
            pw_approximation.breakpoints = breakpoints
            pw_approximation.degree = degree
            pw_approximation.f_x = f_x_2
            self.assertFunctionIsCorrect(pw_approximation, lambda x: pw_poly(x, f_x_2))

        def f_x_3(x):
            return x**2

        with self.subTest(msg="changing all values"):
            pw_approximation.num_state_qubits = 4
            breakpoints = [1, 3, 6]
            degree = 3
            pw_approximation.breakpoints = breakpoints
            pw_approximation.degree = degree
            pw_approximation.f_x = f_x_3
            self.assertFunctionIsCorrect(pw_approximation, lambda x: pw_poly(x, f_x_3))


if __name__ == "__main__":
    unittest.main()
