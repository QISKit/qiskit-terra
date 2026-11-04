# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Ross-Selinger synthesis and plugin."""

import unittest
import numpy as np

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    RZGate,
    IGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
)
from qiskit.quantum_info import Operator
from qiskit.quantum_info import get_clifford_gate_names
from qiskit.quantum_info import random_unitary
from qiskit.synthesis import gridsynth_rz, gridsynth_unitary
from qiskit.converters import dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnitarySynthesis, Collect1qRuns, ConsolidateBlocks
from qiskit.transpiler.passes.synthesis import RossSelingerSynthesis

from test import QiskitTestCase


# Set of single-qubit Clifford gates
CLIFFORD_GATES_1Q_SET = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg"}

# Set of Clifford+T gates
CLIFFORD_T_GATES_SET = set(get_clifford_gate_names() + ["t", "tdg"])


@ddt
class TestRossSelingerSynthesis(QiskitTestCase):
    """Test Ross-Selinger synthesis methods."""

    def test_gridsynth_rz_correct(self):
        """Test that gridsynth_rz works correctly."""
        num_trials = 40
        for angle in np.linspace(-2 * np.pi, 2 * np.pi, num_trials):
            with self.subTest(angle=angle):
                # Approximate RZ-rotation
                approximate_circuit = gridsynth_rz(angle, 1e-10)
                # Check the operators are (almost) equal
                self.assertEqual(Operator(approximate_circuit), Operator(RZGate(angle)))

    @data(10, -10)
    def test_gridsynth_rz_with_nonstandard_angles(self, angle):
        """Test that gridsynth_rz works correctly."""
        # Approximate RZ-rotation
        approximate_circuit = gridsynth_rz(angle, 1e-10)
        # Check the operators are (almost) equal
        self.assertEqual(Operator(approximate_circuit), Operator(RZGate(angle)))

    def test_gridsynth_unitary_correct(self):
        """Test that gridsynth_unitary works correctly."""
        num_trials = 50
        for seed in range(num_trials):
            with self.subTest(seed=seed):
                # Create a random 1q unitary.
                unitary = random_unitary(2, seed=seed)
                # Approximate unitary
                approximate_circuit = gridsynth_unitary(unitary.data, 1e-10)
                # Check the operators are (almost) equal
                self.assertEqual(Operator(approximate_circuit), Operator(unitary))

    def test_gridsynth_rz_deterministic(self):
        """Test that calling gridsynth_rz multiple times produces the same circuit."""
        angle = 1.2345
        num_trials = 10
        approximate_circuits = [gridsynth_rz(angle, 1e-10) for _ in range(num_trials)]

        for idx in range(1, len(approximate_circuits)):
            self.assertEqual(approximate_circuits[idx], approximate_circuits[0])

    def test_gridsynth_unitary_deterministic(self):
        """Test that calling gridsynth_unitary multiple times produces the same circuit."""
        unitary = random_unitary(2, seed=12345)
        num_trials = 10
        approximate_circuits = [gridsynth_unitary(unitary.data, 1e-10) for _ in range(num_trials)]

        for idx in range(1, len(approximate_circuits)):
            self.assertEqual(approximate_circuits[idx], approximate_circuits[0])

    @data(IGate, XGate, YGate, ZGate, HGate, SGate, SdgGate, SXGate, SXdgGate)
    def test_clifford_matrix(self, clifford_cls):
        """Test that the Ross-Selinger algorithm does not return T-gates when approximating
        Clifford-gate matrices.
        """
        circuit = QuantumCircuit(1)
        circuit.append(clifford_cls(), [0])
        matrix = Operator(circuit).data
        approximate_circuit = gridsynth_unitary(matrix)
        self.assertLessEqual(set(approximate_circuit.count_ops()), CLIFFORD_GATES_1Q_SET)
        self.assertEqual(Operator(circuit), Operator(approximate_circuit))

    def test_t_matrix(self):
        """Test that Ross-Selinger algorithm returns a single T-gate for the T-gate matrix."""
        # note that this requires up-to-phase support
        circuit = QuantumCircuit(1)
        circuit.t(0)
        matrix = Operator(circuit).data
        approximate_circuit = gridsynth_unitary(matrix)
        self.assertEqual(approximate_circuit.count_ops().get("t", 0), 1)
        self.assertEqual(approximate_circuit.count_ops().get("tdg", 0), 0)
        self.assertEqual(Operator(circuit), Operator(approximate_circuit))

    def test_tdg_matrix(self):
        """Test that Ross-Selinger algorithm returns a single T-gate for the T-gate matrix."""
        # note that this requires up-to-phase support
        circuit = QuantumCircuit(1)
        circuit.tdg(0)
        matrix = Operator(circuit).data
        approximate_circuit = gridsynth_unitary(matrix)
        self.assertEqual(approximate_circuit.count_ops().get("t", 0), 1)
        self.assertEqual(approximate_circuit.count_ops().get("tdg", 0), 0)
        self.assertEqual(Operator(circuit), Operator(approximate_circuit))

    @data(1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
    def test_approximation_error(self, epsilon):
        """Test that the argument ``epsilon`` works correctly,"""
        approximate_circuit = gridsynth_rz(0.8, epsilon)
        error_matrix = Operator(RZGate(0.8)).data - Operator(approximate_circuit).data
        spectral_norm = np.linalg.norm(error_matrix, 2)
        self.assertLessEqual(spectral_norm, epsilon)


@ddt
class TestRossSelingerPlugin(QiskitTestCase):
    """Test the Ross-Selinger unitary synthesis plugin."""

    def test_unitary_synthesis(self):
        """Test the unitary synthesis transpiler pass with Ross-Selinger algorithm."""
        circuit = QuantumCircuit(2)
        circuit.rx(0.8, 0)
        circuit.cx(0, 1)
        circuit.x(1)

        _1q = Collect1qRuns()
        _cons = ConsolidateBlocks()
        _synth = UnitarySynthesis(["h", "t", "tdg"], method="gridsynth")
        passes = PassManager([_1q, _cons, _synth])
        compiled = passes.run(circuit)

        # The approximation should be good enough for the Operator-equality check to pass
        self.assertEqual(Operator(circuit), Operator(compiled))
        self.assertLessEqual(set(compiled.count_ops()), CLIFFORD_T_GATES_SET)

    def test_plugin(self):
        """Test calling the Ross-Selinger plugin directly."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        unitary = Operator(circuit).data

        plugin = RossSelingerSynthesis()
        compiled_dag = plugin.run(unitary)
        compiled = dag_to_circuit(compiled_dag)

        # The approximation should be good enough for the Operator-equality check to pass
        self.assertEqual(Operator(circuit), Operator(compiled))
        self.assertLessEqual(set(compiled.count_ops()), CLIFFORD_T_GATES_SET)

    def test_plugin_config(self):
        """Test the plugin configs are propagated correctly."""
        circuit = QuantumCircuit(1)
        circuit.rx(1.0, 0)

        unitary = Operator(circuit).data
        plugin = RossSelingerSynthesis()

        epsilons = [1e-6, 1e-8, 1e-10]
        t_expected = [62, 81, 105]

        for eps, t_expect in zip(epsilons, t_expected):
            with self.subTest(eps=eps, t_expect=t_expect):
                compiled_dag = plugin.run(unitary, method="gridsynth", config={"epsilon": eps})
                t_count = compiled_dag.count_ops().get("t", 0)
                self.assertLessEqual(t_count, t_expect)


if __name__ == "__main__":
    unittest.main()
