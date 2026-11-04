# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the UnrollForLoops pass"""

import math
import unittest

from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister, ClassicalRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.utils.unroll_forloops import UnrollForLoops
from test import QiskitTestCase


class TestUnrollForLoops(QiskitTestCase):
    """Test UnrollForLoops pass"""

    def test_range(self):
        """Check simples unrolling case"""
        qreg, creg = QuantumRegister(5, "q"), ClassicalRegister(2, "c")

        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, [0, 1, 2])

        circuit = QuantumCircuit(qreg, creg)
        circuit.for_loop(indexset, loop_parameter, body, [1, 2, 3], [1])

        expected = QuantumCircuit(qreg, creg)
        for index_loop in indexset:
            expected.rx(index_loop, [1, 2, 3])

        passmanager = PassManager()
        passmanager.append(UnrollForLoops())
        result = passmanager.run(circuit)

        self.assertEqual(result, expected)

    def test_parameterless_range(self):
        """Check simples unrolling case when there is not parameter"""
        qreg, creg = QuantumRegister(5, "q"), ClassicalRegister(2, "c")

        body = QuantumCircuit(3, 1)
        indexset = range(0, 10, 2)

        body.h([0, 1, 2])

        circuit = QuantumCircuit(qreg, creg)
        circuit.for_loop(indexset, None, body, [1, 2, 3], [1])

        expected = QuantumCircuit(qreg, creg)
        for _ in indexset:
            expected.h([1, 2, 3])

        passmanager = PassManager()
        passmanager.append(UnrollForLoops())
        result = passmanager.run(circuit)

        self.assertEqual(result, expected)

    def test_preserves_body_global_phase(self):
        """Check that body global phase is accumulated once per loop iteration."""
        indexset = range(3)
        body = QuantumCircuit(1, global_phase=math.pi / 7)
        body.x(0)

        circuit = QuantumCircuit(1)
        circuit.for_loop(indexset, None, body, [0], [])

        expected = QuantumCircuit(1)
        for _ in indexset:
            expected.compose(body, inplace=True)

        result = UnrollForLoops()(circuit)

        self.assertEqual(result, expected)

    def test_preserves_builder_body_global_phase(self):
        """Check global phase from a builder-created body is accumulated once per iteration."""
        indexset = range(3)

        circuit = QuantumCircuit(1)
        with circuit.for_loop(indexset):
            circuit.global_phase = math.pi / 7
            circuit.x(0)

        body = QuantumCircuit(1, global_phase=math.pi / 7)
        body.x(0)
        expected = QuantumCircuit(1)
        for _ in indexset:
            expected.compose(body, inplace=True)

        result = UnrollForLoops()(circuit)

        self.assertEqual(result, expected)

    def test_nested_forloop(self):
        """Test unrolls only one level of nested for-loops"""
        circuit = QuantumCircuit(1)
        twice = range(2)
        with circuit.for_loop(twice):
            with circuit.for_loop(twice):
                circuit.h(0)

        expected = QuantumCircuit(1)
        for _ in twice:
            for _ in twice:
                expected.h(0)

        passmanager = PassManager()
        passmanager.append(UnrollForLoops())
        result = passmanager.run(circuit)

        self.assertEqual(result, expected)

    def test_skip_continue_loop(self):
        """Unrolling should not be done when a `continue;` in the body"""
        parameter = Parameter("x")
        loop_body = QuantumCircuit(1)
        loop_body.rx(parameter, 0)
        loop_body.continue_loop()

        qc = QuantumCircuit(2)
        qc.for_loop([0, 3, 4], parameter, loop_body, [1], [])
        qc.x(0)

        passmanager = PassManager()
        passmanager.append(UnrollForLoops())
        result = passmanager.run(qc)

        self.assertEqual(result, qc)

    def test_skip_continue_in_conditional(self):
        """Unrolling should not be done when a `continue;` is in a nested condition"""
        parameter = Parameter("x")

        true_body = QuantumCircuit(1)
        true_body.continue_loop()
        false_body = QuantumCircuit(1)
        false_body.rx(parameter, 0)

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        loop_body = QuantumCircuit(qr, cr)
        loop_body.if_else((cr, 0), true_body, false_body, [1], [])
        loop_body.x(0)

        qc = QuantumCircuit(qr, cr)
        qc.for_loop([0, 3, 4], parameter, loop_body, qr, cr)

        passmanager = PassManager()
        passmanager.append(UnrollForLoops())
        result = passmanager.run(qc)

        self.assertEqual(result, qc)

    def test_max_target_depth(self):
        """Unrolling should not be done when results over `max_target_depth`"""

        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)
        body = QuantumCircuit(3, 1)
        body.rx(loop_parameter, [0, 1, 2])

        qreg, creg = QuantumRegister(5, "q"), ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qreg, creg)
        circuit.for_loop(indexset, loop_parameter, body, [1, 2, 3], [1])

        passmanager = PassManager()
        passmanager.append(UnrollForLoops(max_target_depth=2))
        result = passmanager.run(circuit)

        self.assertEqual(result, circuit)


if __name__ == "__main__":
    unittest.main()
