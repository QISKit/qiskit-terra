# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Uniform Superposition Gate test.
"""

import unittest
import math
from test import QiskitTestCase
import numpy as np
from ddt import ddt, data

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from qiskit.circuit.library.data_preparation import (
    UniformSuperpositionGate,
)


@ddt
class TestUniformSuperposition(QiskitTestCase):
    """Test initialization with UniformSuperpositionGate class"""

    @data(2, 3, 5)
    def test_uniform_superposition_gate(self, num_superpos_states):
        """Test Uniform Superposition Gate"""
        n = math.ceil(math.log2(num_superpos_states))
        desired_sv = (1 / np.sqrt(num_superpos_states)) * np.array(
            [1.0] * num_superpos_states + [0.0] * (2**n - num_superpos_states)
        )
        gate = UniformSuperpositionGate(num_superpos_states, n)
        actual_sv = Statevector(gate)
        np.testing.assert_allclose(desired_sv, actual_sv)

    @data(2, 3, 5, 13)
    def test_inverse_uniform_superposition_gate(self, num_superpos_states):
        """Test Inverse Uniform Superposition Gate"""
        n = math.ceil(math.log2(num_superpos_states))
        gate = UniformSuperpositionGate(num_superpos_states, n)
        qc = QuantumCircuit(n)
        qc.append(gate, list(range(n)))
        qc.append(gate.inverse(annotated=True), list(range(n)))
        actual_unitary_matrix = np.array(Operator(qc).data)
        desired_unitary_matrix = np.eye(2**n)
        np.testing.assert_allclose(desired_unitary_matrix, actual_unitary_matrix, atol=1e-14)

    @data(-2, -1, 0, 1)
    def test_incompatible_num_superpos_states(self, num_superpos_states):
        """Test error raised if num_superpos_states not valid"""
        n = 1
        with self.assertRaises(ValueError):
            UniformSuperpositionGate(num_superpos_states, n)

    @data(1, 2, 3, 4)
    def test_incompatible_int_num_superpos_states_and_qubit_args(self, n):
        """Test error raised if number of qubits not compatible with integer
        state num_superpos_states (n >= log2(num_superpos_states) )"""
        num_superpos_states = 50
        with self.assertRaises(ValueError):
            UniformSuperpositionGate(num_superpos_states, n)

    @data(2, 3, 5)
    def test_extra_qubits(self, num_superpos_states):
        """Tests for cases where n >= log2(num_superpos_states)"""
        num_extra_qubits = 2
        n = math.ceil(math.log2(num_superpos_states)) + num_extra_qubits
        desired_sv = (1 / np.sqrt(num_superpos_states)) * np.array(
            [1.0] * num_superpos_states + [0.0] * (2**n - num_superpos_states)
        )
        gate = UniformSuperpositionGate(num_superpos_states, n)
        actual_sv = Statevector(gate)
        np.testing.assert_allclose(desired_sv, actual_sv)

    @data(2, 3, 5)
    def test_no_qubit_args(self, num_superpos_states):
        """Test Uniform Superposition Gate without passing the number of qubits as an argument"""
        n = math.ceil(math.log2(num_superpos_states))
        desired_sv = (1 / np.sqrt(num_superpos_states)) * np.array(
            [1.0] * num_superpos_states + [0.0] * (2**n - num_superpos_states)
        )
        gate = UniformSuperpositionGate(num_superpos_states)
        actual_sv = Statevector(gate)
        np.testing.assert_allclose(desired_sv, actual_sv)


if __name__ == "__main__":
    unittest.main()
