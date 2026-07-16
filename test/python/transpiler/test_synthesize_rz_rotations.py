# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the SynthesizeRZRotations pass"""

import numpy as np

from ddt import ddt, data

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import RZGate, QFTGate
from qiskit.quantum_info import Operator, get_clifford_gate_names
from qiskit.synthesis import gridsynth_rz
from qiskit.transpiler.passes.synthesis import SynthesizeRZRotations

from test import QiskitTestCase


@ddt
class TestSynthesizeRzRotations(QiskitTestCase):
    """Test Synthesize Rz rotations method"""

    def test_synthesize_rz_rotations(self):
        """Test that synthesize_rz_rotations works correctly."""
        num_trials = 40
        for angle in np.linspace(-2 * np.pi, 2 * np.pi, num_trials):
            with self.subTest(angle=angle):
                qc = QuantumCircuit(1)
                qc.rz(angle, 0)
                synthesized_circ = SynthesizeRZRotations()(qc)
                self.assertEqual(Operator(synthesized_circ), Operator(RZGate(angle)))

    @data(10, -10)
    def test_synthesize_rz_with_nonstandard_angles(self, angle):
        """Test that synthesize_rz_rotations works correctly."""
        # Approximate RZ-rotation
        qc = QuantumCircuit(1)
        qc.rz(angle, 0)
        synthesized_circ = SynthesizeRZRotations()(qc)
        self.assertEqual(Operator(synthesized_circ), Operator(RZGate(angle)))

    def test_approximation_error(self):
        """Test that the argument ``approximation_degree`` works correctly,"""
        qc = QuantumCircuit(1)
        theta = 2.3579
        qc.rz(theta, 0)

        for eps in 10.0 ** np.arange(-2, -13, -1):
            with self.subTest(eps=eps):
                approximate_circuit = SynthesizeRZRotations(1 - eps)(qc)

                # Check the norm is satisfied
                self.assertLessEqual(operator_norm_distance(approximate_circuit, theta), eps)

    def test_cache_error(self):
        """Test that the cache_error argument works as expected
        when both synthesis_error and cache_error are given.
        """
        qc = QuantumCircuit(1)
        qc.rz(0.0, 0)
        qc.rz(0.1, 0)
        qc.rz(0.2, 0)
        qc.rz(0.3, 0)

        # Sets a very high cache_error. Because both synthesis_error and
        # cache_error are given, we expect them to be used instead of the
        # approximation degree.
        # Exploiting our knowledge of the inner workings
        # of the SynthesizeRZRotations pass, the RZ(0.0, 0) gate will get synthesized
        # first (requiring no T-gates), while the other gates should reuse its synthesis
        # result (hence also requiring no T-gates).
        qct = SynthesizeRZRotations(synthesis_error=1e-10, cache_error=0.5)(qc)
        t_count = qct.count_ops().get("t", 0)
        self.assertEqual(t_count, 0)

    def test_synthesis_error(self):
        """Test that the synthesis_error argument works as expected
        when both synthesis_error and cache_error are given.
        """
        angle = 1.2345

        qc = QuantumCircuit(1)
        qc.rz(angle, 0)

        for eps in 10.0 ** np.arange(-2, -13, -1):
            with self.subTest(eps=eps):
                qct = SynthesizeRZRotations(synthesis_error=eps, cache_error=0.0)(qc)
                t_count = qct.count_ops().get("t", 0)

                expected_circ = gridsynth_rz(angle, eps)
                t_expect = expected_circ.count_ops().get("t", 0)

                self.assertLessEqual(operator_norm_distance(qct, angle), eps)
                self.assertEqual(t_count, t_expect)

    def test_direct_errors_take_precedence(self):
        """Test synthesis and cache errors take precedence if set."""
        angle = 0.1
        qc = QuantumCircuit(1)
        qc.rz(angle, 0)
        qct = SynthesizeRZRotations(
            approximation_degree=1 - 1e-10, synthesis_error=0.1, cache_error=0.1
        )(qc)
        error = operator_norm_distance(qct, angle)

        self.assertLessEqual(error, 0.2)
        self.assertGreaterEqual(error, 1e-10)

    def test_param_angle(self):
        """Test to see if parametrized angles remain unaffected"""
        p = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rz(p, 0)
        synthesized_circ = SynthesizeRZRotations()(qc)
        self.assertEqual(qc, synthesized_circ)

    def test_angle_canonicalization(self):
        """Test that the angle canonicalization works."""
        theta = 0.23
        qc = QuantumCircuit(1)
        qc.rz(theta, 0)

        qc_shifted = QuantumCircuit(1)
        qc_shifted.rz(theta + np.pi / 2, 0)

        synth_rz = SynthesizeRZRotations()
        expected = synth_rz(qc)
        # shift by RZ(pi/2) = S exp(-i pi/4)
        expected.s(0)
        expected.global_phase -= np.pi / 4

        self.assertEqual(expected, synth_rz(qc_shifted))

    def test_qft(self):
        """Test that synthesize_rz_rotations works correctly for larger circuits."""
        num_qubits = 5
        qft = QuantumCircuit(num_qubits)
        qft.append(QFTGate(num_qubits), qft.qubits)

        qft_rz = transpile(qft, basis_gates=get_clifford_gate_names() + ["rz"])
        synthesized_circ = SynthesizeRZRotations(approximation_degree=1 - 1e-10)(qft_rz)
        difference = Operator(synthesized_circ).data - Operator(qft_rz).data

        self.assertLessEqual(operator_norm(difference), 1e-8)

    def test_t(self):
        """Test a T-rotation is synthesized with a single T gate."""
        qc = QuantumCircuit(1)
        qc.rz(np.pi / 4, 0)

        synthesized = SynthesizeRZRotations()(qc)
        self.assertEqual(synthesized.count_ops().get("t", 0), 1)

    def test_called_with_different_error_budgets(self):
        """
        Test SynthesizeRZRotations when called multiple times with different error
        budgets.

        Regression test of https://github.com/Qiskit/qiskit/issues/16385.
        """
        # Each run represents a triple (angle, synthesis error, cache error)
        runs = [(0.0, 0.5, 0.1), (1.0, 0.1, 0.1), (2.0, 0.5, 0.2), (-1.0, 0.3, 0.4)]

        for angle, synthesis_error, cache_error in runs:
            qc = QuantumCircuit(1)
            qc.rz(angle, 0)
            qct = SynthesizeRZRotations(synthesis_error=synthesis_error, cache_error=cache_error)(
                qc
            )
            self.assertLessEqual(operator_norm_distance(qct, angle), synthesis_error)


def operator_norm_distance(circuit, angle):
    """Return the operator norm distance of the circuit to RZ(angle)."""
    difference = Operator(circuit).data - Operator(RZGate(angle)).data
    return operator_norm(difference)


def operator_norm(matrix):
    """Return the operator norm (the max eigenvalue by magnitude) of a matrix."""
    return np.max(np.abs(np.linalg.eigvals(matrix)))
