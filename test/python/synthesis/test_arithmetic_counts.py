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

"""Test gate counts of synthesis algorithms for arithmetic circuits."""

from __future__ import annotations

import unittest

from ddt import data, ddt, unpack

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.arithmetic.adders import FullAdderGate, ModularAdderGate
from qiskit.circuit.library.arithmetic.multipliers import MultiplierGate
from qiskit.synthesis.arithmetic import adder_modular_v17, adder_ripple_c04
from qiskit.transpiler import (
    generate_preset_clifford_t_pass_manager,
    generate_preset_pass_manager,
)

from test import QiskitTestCase


@ddt
class TestAdderSynthesisCounts(QiskitTestCase):
    """Test gate counts of synthesis algorithms for adder circuits."""

    def setUp(self):
        super().setUp()
        # Need optimization level 2 for small modular adder counts
        self.pm = generate_preset_pass_manager(
            optimization_level=2, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        self.clifford_t_pm = generate_preset_clifford_t_pass_manager(
            optimization_level=2, seed_transpiler=12345
        )

    @data(
        (1, 1),
        (2, 10),
        (3, 21),
        (4, 40),
        (5, 65),
    )
    @unpack
    def test_small_modular_adder_cx_count(self, num_ctrl_gates: int, expected_cx: int):
        """Test gate counts of small modular adder."""

        qc = QuantumCircuit(2 * num_ctrl_gates)
        qc.append(ModularAdderGate(num_ctrl_gates), range(2 * num_ctrl_gates))
        transpiled = self.pm.run(qc)
        cx_count = transpiled.count_ops().get("cx", 0)

        self.assertLessEqual(cx_count, expected_cx)

    @data(*range(2, 15, 2))
    def test_vrg_modular_adder_counts(self, num_qubits):
        """Test gate counts of VRG modular adder."""
        qc = adder_modular_v17(num_qubits)
        transpiled = self.pm.run(qc)
        cx_count = transpiled.count_ops().get("cx", 0)
        self.assertLessEqual(cx_count, 16 * num_qubits - 13)
        self.assertEqual(transpiled.num_qubits, 2 * num_qubits)

    @data(("fixed", -8, -8), ("half", 1, 0), ("full", 1, 0))
    @unpack
    def test_cdkm_adder_counts(self, kind, cx_offset, t_offset):
        """Test CX and T-count upper bounds of the optimized CDKM adder."""
        for num_qubits in range(1, 10):
            circuit = adder_ripple_c04(num_qubits, kind=kind)
            cx_count = self.pm.run(circuit).count_ops().get("cx", 0)
            clifford_t_ops = self.clifford_t_pm.run(circuit).count_ops()
            t_count = clifford_t_ops.get("t", 0) + clifford_t_ops.get("tdg", 0)
            self.assertLessEqual(cx_count, 10 * num_qubits + cx_offset)
            self.assertLessEqual(t_count, 8 * num_qubits + t_offset)

    @data((4, 40), (5, 42))
    @unpack
    def test_modular_adder_with_clean_ancilla_counts(self, num_qubits, expected_cx):
        """Test metric-aware default synthesis when a clean ancilla is available."""
        circuit = QuantumCircuit(2 * num_qubits + 1)
        circuit.append(ModularAdderGate(num_qubits), range(2 * num_qubits))

        cx_count = self.pm.run(circuit).count_ops().get("cx", 0)
        clifford_t_ops = self.clifford_t_pm.run(circuit).count_ops()
        t_count = clifford_t_ops.get("t", 0) + clifford_t_ops.get("tdg", 0)

        self.assertLessEqual(cx_count, expected_cx)
        self.assertLessEqual(t_count, 8 * (num_qubits - 1))

    def test_single_qubit_full_adder_counts(self):
        """Test metric-aware synthesis of a single-qubit full adder."""
        gate = FullAdderGate(1)
        circuit = QuantumCircuit(gate.num_qubits)
        circuit.append(gate, circuit.qubits)

        cx_count = self.pm.run(circuit).count_ops().get("cx", 0)
        clifford_t_ops = self.clifford_t_pm.run(circuit).count_ops()
        t_count = clifford_t_ops.get("t", 0) + clifford_t_ops.get("tdg", 0)

        self.assertLessEqual(cx_count, 10)
        self.assertLessEqual(t_count, 8)


@ddt
class TestMultiplierSynthesisCounts(QiskitTestCase):
    """Test gate counts of synthesis algorithms for multiplier circuits."""

    def setUp(self):
        super().setUp()
        self.cx_pm = generate_preset_pass_manager(
            optimization_level=2, basis_gates=["u", "cx"], seed_transpiler=0
        )
        self.clifford_t_pm = generate_preset_clifford_t_pass_manager(
            optimization_level=2, seed_transpiler=0
        )

    @data(
        # Truncated result register: num_result_qubits < 2 * num_state_qubits.
        (2, 2, 30, 106),
        (3, 3, 76, 285),
        (4, 4, 156, 544),
        (5, 5, 478, 883),
        (6, 6, 876, 1302),
        (7, 7, 1350, 1801),
        # Full-width result register: num_result_qubits = 2 * num_state_qubits.
        (2, 4, 117, 145),
        (3, 6, 374, 424),
        (4, 8, 808, 896),
        (5, 10, 1288, 1415),
        (6, 12, 1864, 2034),
        (7, 14, 2536, 2753),
    )
    @unpack
    def test_multiplier_gate_counts(
        self, num_state_qubits, num_result_qubits, expected_cx, expected_t
    ):
        """Test CX and T-count upper bounds for truncated and full-width multipliers."""
        gate = MultiplierGate(num_state_qubits, num_result_qubits)
        circuit = QuantumCircuit(gate.num_qubits)
        circuit.append(gate, circuit.qubits)

        cx_counts = self.cx_pm.run(circuit).count_ops()
        clifford_t_counts = self.clifford_t_pm.run(circuit).count_ops()

        self.assertLessEqual(cx_counts.get("cx", 0), expected_cx)
        self.assertLessEqual(
            clifford_t_counts.get("t", 0) + clifford_t_counts.get("tdg", 0), expected_t
        )


if __name__ == "__main__":
    unittest.main()
