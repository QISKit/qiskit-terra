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
from qiskit.circuit.library.arithmetic.adders import ModularAdderGate
from qiskit.circuit.library.arithmetic.multipliers import MultiplierGate
from qiskit.synthesis.arithmetic import adder_modular_v17
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
