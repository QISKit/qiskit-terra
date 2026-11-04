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

"""Test the quadratic form."""

import unittest
from ddt import ddt, data
import numpy as np

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import QuadraticForm, QuadraticFormGate
from qiskit.quantum_info import Statevector
from test import QiskitTestCase


@ddt
class TestQuadraticForm(QiskitTestCase):
    """Test the QuadraticForm circuit."""

    def assertQuadraticFormIsCorrect(self, m, quadratic, linear, offset, circuit):
        """Assert ``circuit`` implements the quadratic form correctly."""

        def q_form(x, num_bits):
            x = np.array([int(val) for val in reversed(x)])
            res = x.T.dot(quadratic).dot(x) + x.T.dot(linear) + offset
            # compute 2s complement
            res = (2**num_bits + int(res)) % 2**num_bits
            twos = bin(res)[2:].zfill(num_bits)
            return twos

        n = len(quadratic)  # number of value qubits
        ref = np.zeros(2 ** (n + m), dtype=complex)
        for x in range(2**n):
            x_bin = bin(x)[2:].zfill(n)
            index = q_form(x_bin, m) + x_bin
            index = int(index, 2)
            ref[index] = 1 / np.sqrt(2**n)

        actual = QuantumCircuit(circuit.num_qubits)
        actual.h(list(range(n)))
        actual.compose(circuit, inplace=True)
        self.assertTrue(Statevector.from_instruction(actual).equiv(ref))

    @data(True, False)
    def test_endian(self, little_endian):
        """Test the outcome for different endianness."""
        with self.assertWarns(DeprecationWarning):
            qform = QuadraticForm(2, linear=[0, 1], little_endian=little_endian)

        circuit = QuantumCircuit(4)
        circuit.x(1)
        circuit.compose(qform, inplace=True)

        # the result is x_0 linear_0 + x_1 linear_1 = 1 = '0b01'
        result = "01"

        # the state is encoded as |q(x)>|x>, |x> = |x_1 x_0> = |10>
        index = (result if little_endian else result[::-1]) + "10"
        ref = np.zeros(2**4, dtype=complex)
        ref[int(index, 2)] = 1

        self.assertTrue(Statevector.from_instruction(circuit).equiv(ref))

    @data(True, False)
    def test_required_result_qubits(self, use_gate):
        """Test getting the number of required result qubits."""

        constructor = QuadraticFormGate if use_gate else QuadraticForm

        with self.subTest("positive bound"):
            quadratic = [[1, -50], [100, 0]]
            linear = [-5, 5]
            offset = 0
            num_result_qubits = constructor.required_result_qubits(quadratic, linear, offset)
            self.assertEqual(num_result_qubits, 1 + int(np.ceil(np.log2(106 + 1))))

        with self.subTest("negative bound"):
            quadratic = [[1, -50], [10, 0]]
            linear = [-5, 5]
            offset = 0
            num_result_qubits = constructor.required_result_qubits(quadratic, linear, offset)
            self.assertEqual(num_result_qubits, 1 + int(np.ceil(np.log2(55))))

        with self.subTest("empty"):
            num_result_qubits = constructor.required_result_qubits([[]], [], 0)
            self.assertEqual(num_result_qubits, 1)

    @data(True, False)
    def test_quadratic_form(self, use_gate):
        """Test the quadratic form circuit."""

        constructor = QuadraticFormGate if use_gate else QuadraticForm

        with self.subTest("empty"):
            if use_gate:
                circuit = constructor()
            else:
                with self.assertWarns(DeprecationWarning):
                    circuit = constructor()

            self.assertQuadraticFormIsCorrect(1, [[0]], [0], 0, circuit)

        with self.subTest("1d case"):
            quadratic = np.array([[1]])
            linear = np.array([2])
            offset = -1

            if use_gate:
                circuit = constructor(quadratic=quadratic, linear=linear, offset=offset)
            else:
                with self.assertWarns(DeprecationWarning):
                    circuit = constructor(quadratic=quadratic, linear=linear, offset=offset)

            self.assertQuadraticFormIsCorrect(3, quadratic, linear, offset, circuit)

        with self.subTest("negative"):
            quadratic = np.array([[-2]])
            linear = np.array([0])
            offset = -1
            m = 2

            if use_gate:
                circuit = constructor(m, quadratic, linear, offset)
            else:
                with self.assertWarns(DeprecationWarning):
                    circuit = constructor(m, quadratic, linear, offset)

            self.assertQuadraticFormIsCorrect(m, quadratic, linear, offset, circuit)

        with self.subTest("missing quadratic"):
            quadratic = np.zeros((3, 3))
            linear = np.array([-2, 0, 1])
            offset = -1

            if use_gate:
                circuit = constructor(linear=linear, offset=offset)
            else:
                with self.assertWarns(DeprecationWarning):
                    circuit = constructor(linear=linear, offset=offset)

            self.assertQuadraticFormIsCorrect(3, quadratic, linear, offset, circuit)

        with self.subTest("missing linear"):
            quadratic = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])
            linear = np.zeros(3)
            offset = -1
            m = 2

            if use_gate:
                circuit = constructor(m, quadratic, None, offset)
            else:
                with self.assertWarns(DeprecationWarning):
                    circuit = constructor(m, quadratic, None, offset)

            self.assertQuadraticFormIsCorrect(m, quadratic, linear, offset, circuit)

        with self.subTest("missing offset"):
            quadratic = np.array([[2, 1], [-1, -2]])
            linear = np.array([2, 0])
            offset = 0
            m = 2

            if use_gate:
                circuit = constructor(m, quadratic, linear)
            else:
                with self.assertWarns(DeprecationWarning):
                    circuit = constructor(m, quadratic, linear)

            self.assertQuadraticFormIsCorrect(m, quadratic, linear, offset, circuit)

    def test_quadratic_form_parameterized(self):
        """Test the quadratic form circuit with parameters."""

        theta = ParameterVector("th", 7)

        p_quadratic = [[theta[0], theta[1]], [theta[2], theta[3]]]
        p_linear = [theta[4], theta[5]]
        p_offset = theta[6]

        quadratic = np.array([[2, 1], [-1, -2]])
        linear = np.array([2, 0])
        offset = 0
        m = 2

        with self.assertWarns(DeprecationWarning):
            circuit = QuadraticForm(m, p_quadratic, p_linear, p_offset)
        param_dict = dict(zip(theta, [*quadratic[0]] + [*quadratic[1]] + [*linear] + [offset]))

        circuit.assign_parameters(param_dict, inplace=True)

        self.assertQuadraticFormIsCorrect(m, quadratic, linear, offset, circuit)


if __name__ == "__main__":
    unittest.main()
