# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests preset pass managers with 1Q backend"""

from ddt import ddt

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.transpiler import TranspilerError
from test import combine
from test import QiskitTestCase


def fake_1q_v2():
    return GenericBackendV2(
        num_qubits=1, basis_gates=["u1", "u2", "u3"], coupling_map=None, dtm=1.3333, seed=42
    )


def emptycircuit():
    """Empty circuit"""
    return QuantumCircuit()


def circuit_3516():
    """Circuit from https://github.com/Qiskit/qiskit-terra/issues/3516 should fail"""
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.ry(0.11, 1)
    circuit.measure([0], [0])
    return circuit


@ddt
class Test1QV2Failing(QiskitTestCase):
    """1QV2 tests that should fail."""

    @combine(
        circuit=[circuit_3516],
        level=[0, 1, 2, 3],
        dsc="Transpiling {circuit.__name__} at level {level} should fail",
        name="{circuit.__name__}_level{level}_fail_v2",
    )
    def test(self, circuit, level):
        """All the levels with all the 1Q backendV2"""
        with self.assertRaises(TranspilerError):
            transpile(circuit(), backend=fake_1q_v2(), optimization_level=level, seed_transpiler=42)


@ddt
class TestBasicSimulatorWorking(QiskitTestCase):
    """All the levels with a simulator backend"""

    @combine(
        circuit=[circuit_3516],
        level=[0, 1, 2, 3],
        dsc="Transpiling {circuit.__name__} at level {level} should work for simulator",
        name="{circuit.__name__}_level{level}_valid",
    )
    def test_simulator(self, circuit, level):
        """All the levels with a simulator backend"""
        backend = BasicSimulator()
        result = transpile(circuit(), backend=backend, optimization_level=level, seed_transpiler=42)
        self.assertIsInstance(result, QuantumCircuit)


@ddt
class Test1QV2Working(QiskitTestCase):
    """1QV2 tests that should work."""

    @combine(
        circuit=[emptycircuit],
        level=[0, 1, 2, 3],
        dsc="Transpiling {circuit.__name__} at level {level} should work",
        name="{circuit.__name__}_level{level}_valid_v2",
    )
    def test_device(self, circuit, level):
        """All the levels with all the 1Q backendV2"""
        result = transpile(
            circuit(), backend=fake_1q_v2(), optimization_level=level, seed_transpiler=42
        )
        self.assertIsInstance(result, QuantumCircuit)
