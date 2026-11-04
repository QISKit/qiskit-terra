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

"""Test the linear amplitude function."""

import unittest
from functools import partial
from collections import defaultdict
from ddt import ddt, data, unpack
import numpy as np

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import LinearAmplitudeFunction, LinearAmplitudeFunctionGate
from qiskit.quantum_info import Statevector
from test import QiskitTestCase


@ddt
class TestLinearAmplitudeFunctional(QiskitTestCase):
    """Test the functional Pauli rotations."""

    def assertFunctionIsCorrect(self, function_circuit, reference, num_ancillas=None):
        """Assert that ``function_circuit`` implements the reference function ``reference``."""
        num_ancillas = function_circuit.num_ancillas if num_ancillas is None else num_ancillas
        num_state_qubits = function_circuit.num_qubits - num_ancillas - 1

        circuit = QuantumCircuit(function_circuit.num_qubits)
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

        np.testing.assert_almost_equal(unrolled_probabilities, unrolled_expectations)

    def evaluate_function(
        self, x_int, num_qubits, slope, offset, domain, image, rescaling_factor, breakpoints=None
    ):
        """A helper function to get the expected value of the linear amplitude function."""
        a, b = domain
        c, d = image

        # map integer x to the domain of the function
        x = a + (b - a) / (2**num_qubits - 1) * x_int

        # apply the function
        if breakpoints is None:
            value = offset + slope * x
        else:
            value = 0
            for i, point in enumerate(reversed(breakpoints)):
                if x >= point:
                    value = offset[-(i + 1)] + slope[-(i + 1)] * (x - point)
                    break

        # map the value to [0, 1]
        normalized = (value - c) / (d - c)

        # prepared value for taylor approximation
        return np.pi / 4 + np.pi * rescaling_factor / 2 * (normalized - 0.5)

    @data(
        (2, 1, 0, (0, 3), (0, 3), 0.1, None),
        (3, 1, 0, (0, 1), (0, 1), 0.01, None),
        (1, [0, 0], [0, 0], (0, 2), (0, 1), 0.1, [0, 1]),
        (2, [1, -1], [0, 1], (0, 2), (0, 1), 0.1, [0, 1]),
        (3, [1, 0, -1, 0], [0, 0.5, -0.5, -0.5], (0, 2.5), (-0.5, 0.5), 0.1, [0, 0.5, 1, 2]),
    )
    @unpack
    def test_polynomial_function(
        self, num_state_qubits, slope, offset, domain, image, rescaling_factor, breakpoints
    ):
        """Test the polynomial rotation."""
        reference = partial(
            self.evaluate_function,
            num_qubits=num_state_qubits,
            slope=slope,
            offset=offset,
            domain=domain,
            image=image,
            rescaling_factor=rescaling_factor,
            breakpoints=breakpoints,
        )

        for use_gate in [True, False]:
            constructor = LinearAmplitudeFunctionGate if use_gate else LinearAmplitudeFunction
            num_ancillas = int(len(breakpoints or [0]) > 1) if use_gate else None

            if use_gate:
                linear_f = constructor(
                    num_state_qubits, slope, offset, domain, image, rescaling_factor, breakpoints
                )
            else:
                with self.assertWarns(DeprecationWarning):
                    linear_f = constructor(
                        num_state_qubits,
                        slope,
                        offset,
                        domain,
                        image,
                        rescaling_factor,
                        breakpoints,
                    )

            with self.subTest(use_gate=use_gate):
                self.assertFunctionIsCorrect(linear_f, reference, num_ancillas)

    def test_not_including_start_in_breakpoints(self):
        """Test not including the start of the domain works."""
        num_state_qubits = 1
        slope = [0, 0]
        offset = [0, 0]
        domain = (0, 2)
        image = (0, 1)
        rescaling_factor = 0.1
        breakpoints = [1]

        reference = partial(
            self.evaluate_function,
            num_qubits=num_state_qubits,
            slope=slope,
            offset=offset,
            domain=domain,
            image=image,
            rescaling_factor=rescaling_factor,
            breakpoints=breakpoints,
        )

        with self.assertWarns(DeprecationWarning):
            linear_f = LinearAmplitudeFunction(
                num_state_qubits, slope, offset, domain, image, rescaling_factor, breakpoints
            )

        self.assertFunctionIsCorrect(linear_f, reference)

    def test_invalid_inputs_raise(self):
        """Test passing invalid inputs to the LinearAmplitudeFunction raises an error."""
        # default values
        num_state_qubits = 1
        slope = [0, 0]
        offset = [0, 0]
        domain = (0, 2)
        image = (0, 1)
        rescaling_factor = 0.1
        breakpoints = [0, 1]

        with self.subTest("mismatching breakpoints size"):
            with self.assertRaises(ValueError):
                with self.assertWarns(DeprecationWarning):
                    _ = LinearAmplitudeFunction(
                        num_state_qubits, slope, offset, domain, image, rescaling_factor, [0]
                    )

        with self.subTest("mismatching offsets"):
            with self.assertRaises(ValueError):
                with self.assertWarns(DeprecationWarning):
                    _ = LinearAmplitudeFunction(
                        num_state_qubits, slope, [0], domain, image, rescaling_factor, breakpoints
                    )

        with self.subTest("mismatching slopes"):
            with self.assertRaises(ValueError):
                with self.assertWarns(DeprecationWarning):
                    _ = LinearAmplitudeFunction(
                        num_state_qubits, [0], offset, domain, image, rescaling_factor, breakpoints
                    )

        with self.subTest("breakpoints outside of domain"):
            with self.assertRaises(ValueError):
                with self.assertWarns(DeprecationWarning):
                    _ = LinearAmplitudeFunction(
                        num_state_qubits,
                        slope,
                        offset,
                        (0, 0.2),
                        image,
                        rescaling_factor,
                        breakpoints,
                    )

        with self.subTest("breakpoints not sorted"):
            with self.assertRaises(ValueError):
                with self.assertWarns(DeprecationWarning):
                    _ = LinearAmplitudeFunction(
                        num_state_qubits, slope, offset, domain, image, rescaling_factor, [1, 0]
                    )

    def test_post_processing(self):
        """Test the ``post_processing`` method."""
        num_state_qubits = 2
        slope = 1
        offset = -2
        domain = (0, 2)
        image = (-2, 0)
        rescaling_factor = 0.1

        with self.assertWarns(DeprecationWarning):
            circuit = LinearAmplitudeFunction(
                num_state_qubits, slope, offset, domain, image, rescaling_factor
            )

        values = [0, 0.2, 0.5, 0.9, 1]

        def reference_post_processing(x):
            x = 2 / np.pi / rescaling_factor * (x - 0.5) + 0.5
            return image[0] + (image[1] - image[0]) * x

        expected = [reference_post_processing(value) for value in values]
        actual = [circuit.post_processing(value) for value in values]

        self.assertListEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
