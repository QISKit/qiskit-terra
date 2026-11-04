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

"""Test Clifford+T optimization pass"""

import numpy as np

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes import SolovayKitaev, OptimizeCliffordT
from qiskit.quantum_info import Operator
from test import QiskitTestCase


@ddt
class TestOptimizeCliffordT(QiskitTestCase):
    """Test the OptimizeCliffordT pass."""

    angles = np.linspace(0, 2 * np.pi, 10)

    @data(*angles)
    def test_solovay_kitaev_rx(self, angle):
        """Test optimization of circuits coming out of the Solovay-Kitaev pass."""
        qc = QuantumCircuit(1)
        qc.rx(angle, 0)

        # Run Solovay-Kitaev pass on qc
        transpiled = SolovayKitaev()(qc)
        self.assertLessEqual(set(transpiled.count_ops()), {"h", "t", "tdg"})

        # Run Clifford+T optimization pass on the transpiled circuit
        optimized = OptimizeCliffordT()(transpiled)
        self.assertTrue(Operator(transpiled), Operator(optimized))

    def test_removes_t_tdg_gates(self):
        """A simple test that a pair of T,Tdg-gates gets removed."""
        qc = QuantumCircuit(1)
        qc.t(0)
        qc.s(0)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        qc.tdg(0)
        optimized = OptimizeCliffordT()(qc)

        # The Clifford gates in the middle correspond to an identity
        # Clifford up a global phase of pi/4. The T and Tdg gates
        # should cancel out.
        expected = QuantumCircuit(1, global_phase=np.pi / 4)
        self.assertEqual(Operator(qc), Operator(expected))

        self.assertEqual(optimized, expected)

    def test_combines_t_gates(self):
        """A simple test that a pair of T-gates gets combined."""
        qc = QuantumCircuit(1)
        qc.t(0)
        qc.s(0)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        qc.t(0)
        optimized = OptimizeCliffordT()(qc)

        # The Clifford gates in the middle correspond to an identity
        # Clifford up a global phase of pi/4. The two T gates at the ends
        # can be combined into an S-gate.
        expected = QuantumCircuit(1, global_phase=np.pi / 4)
        expected.s(0)
        self.assertEqual(Operator(qc), Operator(expected))

        self.assertEqual(optimized, expected)

    def test_combines_tdg_gates(self):
        """A simple test that a pair of Tdg-gates gets removed."""
        qc = QuantumCircuit(1)
        qc.tdg(0)
        qc.s(0)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        qc.tdg(0)
        optimized = OptimizeCliffordT()(qc)

        # The Clifford gates in the middle correspond to an identity
        # Clifford up a global phase of pi/4. The two Tdg gates on the ends
        # can be combined into an Sdg-gate (which gets implemented as S,Z).
        expected = QuantumCircuit(1, global_phase=np.pi / 4)
        expected.s(0)
        expected.z(0)
        self.assertEqual(Operator(qc), Operator(expected))

        self.assertEqual(optimized, expected)

    def test_does_not_remove_t_gates(self):
        """A simple test for the case that a pair T-gates cannot be removed."""
        qc = QuantumCircuit(1)
        qc.t(0)
        qc.h(0)
        qc.t(0)

        # The pass should not remove anything.
        optimized = OptimizeCliffordT()(qc)

        self.assertEqual(optimized, qc)

    def test_non_clifford_t_gates(self):
        """A simple test that the pass optimizes T-gates when there are
        also non-{Clifford,T}-gates in the circuit.
        """
        qc = QuantumCircuit(2, 1)
        qc.t(0)
        qc.s(0)
        qc.t(0)
        qc.rx(0.5, 1)
        qc.t(1)
        qc.z(1)
        qc.tdg(1)
        qc.measure(0, 0)

        # The pass should perform some optimizations, even when
        # the circuit also contains non-{Clifford,T}-gates.
        optimized = OptimizeCliffordT()(qc)

        expected = QuantumCircuit(2, 1)
        expected.z(0)
        expected.rx(0.5, 1)
        expected.z(1)
        expected.measure(0, 0)

        self.assertEqual(optimized, expected)

    def test_non_clifford_t_gate_in_the_middle(self):
        """A simple test that the pass does not optimize
        across non-{Clifford, T}-gates.
        """
        qc = QuantumCircuit(1)
        qc.t(0)
        qc.rx(0.5, 0)
        qc.t(0)
        qc.t(0)
        qc.t(0)
        qc.ry(0.5, 0)
        qc.tdg(0)

        # The pass should perform some optimizations, but not
        # across non-{Clifford,T}-gates.
        optimized = OptimizeCliffordT()(qc)

        expected = QuantumCircuit(1)
        expected.t(0)
        expected.rx(0.5, 0)
        expected.t(0)
        expected.s(0)
        expected.ry(0.5, 0)
        expected.tdg(0)

        self.assertEqual(optimized, expected)
        self.assertEqual(Operator(qc), Operator(expected))

    def test_reduces_clifford_gates(self):
        """A simple test that the pass reduces the number
        of Clifford gates even when the number of T-gates
        remains the same.
        """
        qc = QuantumCircuit(1)
        qc.s(0)
        qc.t(0)
        qc.s(0)

        optimized = OptimizeCliffordT()(qc)

        expected = QuantumCircuit(1)
        expected.t(0)
        expected.z(0)

        self.assertEqual(optimized, expected)
        self.assertEqual(Operator(qc), Operator(expected))

    def test_reduces_clifford_gates_2(self):
        """A simple test that the pass reduces the number
        of Clifford gates even when the number of T-gates
        remains the same.
        """
        qc = QuantumCircuit(1)
        qc.s(0)
        qc.h(0)
        qc.sx(0)
        qc.t(0)
        qc.h(0)
        qc.z(0)
        qc.x(0)

        optimized = OptimizeCliffordT()(qc)

        expected = QuantumCircuit(1, global_phase=np.pi / 4)
        expected.h(0)
        expected.tdg(0)
        expected.h(0)
        expected.x(0)

        self.assertEqual(optimized, expected)
        self.assertEqual(Operator(qc), Operator(expected))
