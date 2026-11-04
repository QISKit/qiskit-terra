# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the TranslateParameterizedGates pass"""

import unittest

from qiskit.circuit import ParameterVector, Parameter, Gate, QuantumCircuit
from qiskit.circuit.library import n_local
from qiskit.exceptions import QiskitError
from qiskit.transpiler.passes import TranslateParameterizedGates
from qiskit.providers.fake_provider import GenericBackendV2
from test import QiskitTestCase


class TestTranslateParameterized(QiskitTestCase):
    """Test the pass to translate parameterized gates, but keep others as is."""

    def test_only_parameterized_is_unrolled(self):
        """Test only parameterized gates are unrolled."""
        x = ParameterVector("x", 4)
        block1 = QuantumCircuit(1)
        block1.rx(x[0], 0)

        sub_block = QuantumCircuit(2)
        sub_block.cx(0, 1)
        sub_block.rz(x[2], 0)

        block2 = QuantumCircuit(2)
        block2.ry(x[1], 0)
        block2.append(sub_block.to_gate(), [0, 1])

        block3 = QuantumCircuit(3)
        block3.ccx(0, 1, 2)

        circuit = QuantumCircuit(3)
        circuit.append(block1.to_gate(), [1])
        circuit.append(block2.to_gate(), [0, 1])
        circuit.append(block3.to_gate(), [0, 1, 2])
        circuit.cry(x[3], 0, 2)

        supported_gates = ["rx", "ry", "rz", "cp", "crx", "cry", "crz"]
        unroller = TranslateParameterizedGates(supported_gates)
        unrolled = unroller(circuit)

        expected = QuantumCircuit(3)
        expected.rx(x[0], 1)
        expected.ry(x[1], 0)
        expected.cx(0, 1)
        expected.rz(x[2], 0)
        expected.append(block3.to_gate(), [0, 1, 2])
        expected.cry(x[3], 0, 2)

        self.assertEqual(unrolled, expected)

    def test_target(self):
        """Test unrolling with a target."""
        target = GenericBackendV2(num_qubits=5).target
        circuit = n_local(2, "rz", "cx", reps=2, entanglement="linear")

        translator = TranslateParameterizedGates(target=target)
        translated = translator(circuit)

        expected_ops = {"cx": 2, "rz": 6}

        self.assertEqual(translated.count_ops(), expected_ops)

    def test_no_supported_gates_or_target(self):
        """Test an error is raised if neither of ``supported_gates`` and ``target`` is supported."""
        with self.assertRaises(ValueError):
            _ = TranslateParameterizedGates(supported_gates=None, target=None)

    def test_translation_impossible(self):
        """Test translating a parameterized gate without definition does not work."""

        x = Parameter("x")
        mission_actually_impossible = Gate("mysterious", 1, [x])

        circuit = QuantumCircuit(1)
        circuit.rx(x, 0)
        circuit.append(mission_actually_impossible, [0])

        translator = TranslateParameterizedGates(["rx", "ry", "rz"])

        with self.assertRaises(QiskitError):
            _ = translator(circuit)


if __name__ == "__main__":
    unittest.main()
