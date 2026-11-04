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

"""Test synthesis algorithms for multi-controlled gates."""

import unittest
from test import QiskitTestCase, combine

import numpy as np
from ddt import data, ddt

from qiskit.circuit import Gate, QuantumCircuit, Parameter
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int
from qiskit.circuit.library import (
    C3XGate,
    C4XGate,
    CCXGate,
    CXGate,
    CZGate,
    HGate,
    MCXGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    SdgGate,
    SGate,
    SXdgGate,
    SXGate,
    TdgGate,
    TGate,
    U1Gate,
    U2Gate,
    U3Gate,
    UGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.operator_utils import (
    _equal_with_ancillas,
    matrix_equal,
)
from qiskit.synthesis.multi_controlled import (
    synth_c3x,
    synth_c4x,
    synth_mcp_noaux_default,
    synth_mcp_noaux_sp22,
    synth_mcp_noaux_v24,
    synth_mcx_1_clean_b95,
    synth_mcx_1_clean_kg24,
    synth_mcx_1_dirty_kg24,
    synth_mcx_2_clean_kg24,
    synth_mcx_2_dirty_kg24,
    synth_mcx_gray_code,
    synth_mcx_n_clean_m15,
    synth_mcx_n_dirty_i15,
    synth_mcx_n_dirty_m15,
    synth_mcx_noaux_hp24,
    synth_mcx_noaux_sp22,
    synth_mcx_noaux_v24,
)
from qiskit.transpiler import generate_preset_pass_manager, generate_preset_clifford_t_pass_manager


@ddt
class TestMCSynthesisCorrectness(QiskitTestCase):
    """Test correctness of synthesis methods for multi-controlled gates."""

    @staticmethod
    def mc_matrix(base_gate: Gate, num_ctrl_qubits: int):
        """Return matrix for the MC gate with the given base gate and the number of control qubits."""
        base_mat = base_gate.to_matrix()
        return _compute_control_matrix(base_mat, num_ctrl_qubits)

    def assertSynthesisCorrect(
        self,
        base_gate: Gate,
        num_ctrl_qubits: int,
        synthesized_circuit: QuantumCircuit,
        clean_ancillas: bool,
    ):
        """Check correctness of a quantum circuit produced by a synthesis algorithm for multi-controlled
        gates, taking the additional ancilla qubits into account.

        This check is based on comparing the synthesized and the expected matrices and thus only
        works for synthesized circuits with up to about 10 qubits.

        Args:
            base_gate: the base gate of the MC gate.
            num_ctrl_qubits: the number of control qubits in the MC gate.
            synthesized_circuit: the quantum circuit synthesizing the MC gate.
            clean_ancillas: True if the algorithm uses clean ancilla qubits.

        Note: currently we do not have any MC synthesis algorithms that use both clean and dirty
        ancilla qubits. When we do, we will need to extend this function.
        """
        original_op = Operator(self.mc_matrix(base_gate, num_ctrl_qubits))
        synthesized_op = Operator(synthesized_circuit)

        num_qubits_original = original_op._op_shape._num_qargs_l
        num_qubits_synthesized = synthesized_circuit.num_qubits

        expected_op = Operator(
            np.kron(np.eye(2 ** (num_qubits_synthesized - num_qubits_original)), original_op)
        )
        if clean_ancillas:
            ancilla_qubits = list(range(num_qubits_original, num_qubits_synthesized))
        else:
            ancilla_qubits = []

        result = _equal_with_ancillas(
            synthesized_op,
            expected_op,
            ancilla_qubits,
        )
        self.assertTrue(result)

    @data(0, 1, 2, 3, 4, 5, 6)
    def test_mcp_noaux_v24(self, num_ctrl_qubits: int):
        """Test synth_mcp_noaux_v24 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcp_noaux_v24(num_ctrl_qubits, phase=0.123)
        self.assertSynthesisCorrect(
            PhaseGate(0.123), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6)
    def test_mcp_noaux_default(self, num_ctrl_qubits: int):
        """Test synth_mcp_noaux_default by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcp_noaux_default(num_ctrl_qubits, phase=0.123)
        self.assertSynthesisCorrect(
            PhaseGate(0.123), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6)
    def test_mcp_noaux_sp22(self, num_ctrl_qubits: int):
        """Test synth_mcp_noaux_sp22 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcp_noaux_sp22(num_ctrl_qubits, phase=0.123)
        self.assertSynthesisCorrect(
            PhaseGate(0.123), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6)
    def test_mcp_noaux_v24_with_params(self, num_ctrl_qubits: int):
        """Test synth_mcp_noaux_v24 works correctly with parametric angles."""
        theta = Parameter("theta")
        val = 0.456
        circuit = synth_mcp_noaux_v24(num_ctrl_qubits, phase=theta)
        bound_circuit = circuit.assign_parameters([val])
        self.assertSynthesisCorrect(
            PhaseGate(val), num_ctrl_qubits, bound_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6)
    def test_mcp_noaux_default_with_params(self, num_ctrl_qubits: int):
        """Test synth_mcp_noaux_default works correctly with parametric angles."""
        theta = Parameter("theta")
        val = 0.456
        circuit = synth_mcp_noaux_default(num_ctrl_qubits, phase=theta)
        bound_circuit = circuit.assign_parameters([val])
        self.assertSynthesisCorrect(
            PhaseGate(val), num_ctrl_qubits, bound_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6)
    def test_mcp_noaux_sp22_with_params(self, num_ctrl_qubits: int):
        """Test synth_mcp_noaux_sp22 works correctly with parametric angles."""
        theta = Parameter("theta")
        val = 0.456
        circuit = synth_mcp_noaux_sp22(num_ctrl_qubits, phase=theta)
        bound_circuit = circuit.assign_parameters([val])
        self.assertSynthesisCorrect(
            PhaseGate(val), num_ctrl_qubits, bound_circuit, clean_ancillas=False
        )

    def test_mcp_noaux_v24_with_expr_param(self):
        """Test synth_mcp_noaux_v24 works correctly with a parameter expression."""
        a, b = Parameter("a"), Parameter("b")
        val_a, val_b = 0.456, 0.123
        circuit = synth_mcp_noaux_v24(3, phase=2 * a + b)
        bound_circuit = circuit.assign_parameters({a: val_a, b: val_b})
        self.assertSynthesisCorrect(
            PhaseGate(2 * val_a + val_b), 3, bound_circuit, clean_ancillas=False
        )

    def test_mcp_noaux_default_with_expr_param(self):
        """Test synth_mcp_noaux_default works correctly with a parameter expression."""
        a, b = Parameter("a"), Parameter("b")
        val_a, val_b = 0.456, 0.123
        circuit = synth_mcp_noaux_default(3, phase=2 * a + b)
        bound_circuit = circuit.assign_parameters({a: val_a, b: val_b})
        self.assertSynthesisCorrect(
            PhaseGate(2 * val_a + val_b), 3, bound_circuit, clean_ancillas=False
        )

    def test_mcp_noaux_sp22_with_expr_param(self):
        """Test synth_mcp_noaux_sp22 works correctly with a parameter expression."""
        a, b = Parameter("a"), Parameter("b")
        val_a, val_b = 0.456, 0.123
        circuit = synth_mcp_noaux_sp22(3, phase=2 * a + b)
        bound_circuit = circuit.assign_parameters({a: val_a, b: val_b})
        self.assertSynthesisCorrect(
            PhaseGate(2 * val_a + val_b), 3, bound_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6)
    def test_mcx_n_dirty_i15(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_dirty_i15 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_n_dirty_i15(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6)
    def test_mcx_n_dirty_m15(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_dirty_m15, including arbitrary dirty-ancilla states."""
        synthesized_circuit = synth_mcx_n_dirty_m15(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6)
    def test_mcx_n_clean_m15(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_clean_m15 by comparing synthesized and expected matrices."""
        # Note: the method requires at least 3 control qubits
        synthesized_circuit = synth_mcx_n_clean_m15(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=True
        )

    @data(0, 1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_1_clean_b95(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_clean_b95 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_1_clean_b95(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=True
        )

    @data(0, 1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_1_clean_kg24(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_clean_kg24 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_1_clean_kg24(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=True
        )

    @data(0, 1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_1_dirty_kg24(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_dirty_kg24 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_1_dirty_kg24(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_2_clean_kg24(self, num_ctrl_qubits: int):
        """Test synth_mcx_2_clean_kg24 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_2_clean_kg24(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=True
        )

    @data(0, 1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_2_dirty_kg24(self, num_ctrl_qubits: int):
        """Test synth_mcx_2_dirty_kg24 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_2_dirty_kg24(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_gray_code(self, num_ctrl_qubits: int):
        """Test synth_mcx_gray_code by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_gray_code(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_noaux_sp22(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_sp22 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_noaux_sp22(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_noaux_v24(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_v24 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_noaux_v24(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(0, 1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_noaux_hp24(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_hp24 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_noaux_hp24(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    def test_c3x(self):
        """Test synth_c3x by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_c3x()
        self.assertSynthesisCorrect(XGate(), 3, synthesized_circuit, clean_ancillas=False)

    def test_c4x(self):
        """Test synth_c4x by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_c4x()
        self.assertSynthesisCorrect(XGate(), 4, synthesized_circuit, clean_ancillas=False)

    @combine(
        num_ctrl_qubits=[0, 1, 2, 3, 4, 5, 6, 7],
        base_gate=[
            XGate(),
            YGate(),
            ZGate(),
            HGate(),
            PhaseGate(0.123),
            SGate(),
            SdgGate(),
            TGate(),
            TdgGate(),
            SXGate(),
            SXdgGate(),
            RXGate(0.789),
            RYGate(0.123),
            RZGate(0.456),
            UGate(0.1, 0.2, 0.3),
            U1Gate(0.1),
            U2Gate(0.1, 0.2),
            U3Gate(0.1, 0.2, 0.3),
        ],
        annotated=[False, True],
    )
    def test_create_mc_gates(self, num_ctrl_qubits, base_gate, annotated):
        """Test that creating various multi-controlled gates with small number of controls
        and no ancillas yields correct unitaries.
        """
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(
            base_gate.control(num_ctrl_qubits, annotated=annotated), range(num_ctrl_qubits + 1)
        )
        test_op = Operator(qc).data
        cop_mat = self.mc_matrix(base_gate, num_ctrl_qubits)
        self.assertTrue(matrix_equal(cop_mat, test_op))

    @combine(
        num_ctrl_qubits_original=[1, 2, 3, 4, 5],
        ctrl_state_original=[None, 0, 1],
        num_ctrl_qubits_new=[2],
        ctrl_state_new=[None, 1, 2],
        annotated=[False, True],
    )
    def test_create_open_controlled_mcx_gates(
        self,
        num_ctrl_qubits_original,
        ctrl_state_original,
        num_ctrl_qubits_new,
        ctrl_state_new,
        annotated,
    ):
        """Test that creating open controlled multi-controlled X gates works correctly,
        including correctly combining the control states of the original and the additional
        control lines.
        """
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits_original, ctrl_state=ctrl_state_original)
        cgate = gate.control(
            num_ctrl_qubits=num_ctrl_qubits_new, ctrl_state=ctrl_state_new, annotated=annotated
        )

        num_ctrl_qubits_joint = num_ctrl_qubits_original + num_ctrl_qubits_new
        ctrl_state_joint = (
            _ctrl_state_to_int(ctrl_state_original, num_ctrl_qubits_original) << num_ctrl_qubits_new
        ) | _ctrl_state_to_int(ctrl_state_new, num_ctrl_qubits_new)

        expected_gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits_joint, ctrl_state=ctrl_state_joint)
        self.assertEqual(Operator(cgate), Operator(expected_gate))


@ddt
class TestMCSynthesisCounts(QiskitTestCase):
    """Test gate counts produced by multi-controlled synthesis methods."""

    def setUp(self):
        super().setUp()
        self.pm = generate_preset_pass_manager(
            optimization_level=0,
            basis_gates=["u", "cx"],
            seed_transpiler=12345,
            qubits_initially_zero=False,
        )
        self.pm_cliff_t = generate_preset_clifford_t_pass_manager(
            optimization_level=0,
            basis_gates=["h", "t", "tdg", "x", "cx"],
            seed_transpiler=12345,
            qubits_initially_zero=False,
        )

    @data(10, 15, 20)
    def test_mcp_noaux_v24_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcp_noaux_v24 bound on CX count."""
        synthesized_circuit = synth_mcp_noaux_v24(num_ctrl_qubits, phase=0.123)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the documentation of synth_mcp_noaux_v24
        self.assertEqual(cx_count, 8 * num_ctrl_qubits**2 - 16 * num_ctrl_qubits - 60)

    @data(10, 15, 20)
    def test_synth_mcp_noaux_sp22(self, num_ctrl_qubits: int):
        """Test synth_mcp_noaux_sp22 bound on CX count."""
        synthesized_circuit = synth_mcp_noaux_sp22(num_ctrl_qubits, phase=0.123)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the documentation of synth_mcp_noaux_sp22
        self.assertEqual(cx_count, 4 * num_ctrl_qubits**2 - 4 * num_ctrl_qubits + 2)

    @data(10, 15, 20)
    def test_synth_mcp_noaux_default(self, num_ctrl_qubits: int):
        """Test synth_mcp_noaux_default bound on CX count."""
        synthesized_circuit = synth_mcp_noaux_default(num_ctrl_qubits, phase=0.123)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the documentation of synth_mcp_noaux_sp22
        self.assertEqual(cx_count, 4 * num_ctrl_qubits**2 - 4 * num_ctrl_qubits + 2)

    @data(5, 10, 15)
    def test_mcx_n_dirty_i15_counts(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_dirty_i15 bound on CX count and T count."""
        synthesized_circuit = synth_mcx_n_dirty_i15(num_ctrl_qubits)
        transpiled_circuit = self.pm_cliff_t.run(synthesized_circuit)
        count_ops = transpiled_circuit.count_ops()
        cx_count = count_ops["cx"]
        t_count = count_ops["t"] + count_ops["tdg"]
        # The bound from the documentation of synth_mcx_n_dirty_i15
        self.assertEqual(synthesized_circuit.num_qubits, 2 * num_ctrl_qubits - 1)
        self.assertEqual(cx_count, 8 * num_ctrl_qubits - 6)
        self.assertEqual(t_count, 8 * num_ctrl_qubits - 2)

    @data(3, 4, 5, 10, 15)
    def test_mcx_n_dirty_m15_resources(self, num_ctrl_qubits: int):
        """Test the exact ancillary-qubit, T, and CX counts of synth_mcx_n_dirty_m15."""
        circuit = synth_mcx_n_dirty_m15(num_ctrl_qubits)
        transpiled_circuit = self.pm_cliff_t.run(circuit)
        counts = transpiled_circuit.count_ops()

        expected_ancillas = 1 if num_ctrl_qubits == 3 else (num_ctrl_qubits - 1) // 2
        expected_cx = 14 if num_ctrl_qubits == 3 else 8 * num_ctrl_qubits - 12
        expected_t = 16 if num_ctrl_qubits == 3 else 8 * num_ctrl_qubits - 8

        self.assertEqual(circuit.num_qubits, num_ctrl_qubits + 1 + expected_ancillas)
        self.assertEqual(counts["cx"], expected_cx)
        self.assertEqual(counts["t"] + counts["tdg"], expected_t)

    @data(3, 4, 5, 10, 15)
    def test_mcx_n_clean_m15_resources(self, num_ctrl_qubits: int):
        """Test the exact ancillary-qubit, T, and CX counts of synth_mcx_n_clean_m15."""
        circuit = synth_mcx_n_clean_m15(num_ctrl_qubits)
        transpiled_circuit = self.pm_cliff_t.run(circuit)
        counts = transpiled_circuit.count_ops()

        self.assertEqual(circuit.num_qubits, num_ctrl_qubits + 1 + (num_ctrl_qubits - 1) // 2)
        self.assertEqual(counts["cx"], 6 * num_ctrl_qubits - 6)
        self.assertEqual(counts["t"] + counts["tdg"], 8 * num_ctrl_qubits - 9)

    @data(5, 6, 7, 10, 15)
    def test_mcx_1_clean_b95_counts(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_clean_b95 bound on CX count."""
        synthesized_circuit = synth_mcx_1_clean_b95(num_ctrl_qubits)
        transpiled_circuit = self.pm_cliff_t.run(synthesized_circuit)
        count_ops = transpiled_circuit.count_ops()

        cx_count = count_ops["cx"]
        t_count = count_ops["t"] + count_ops["tdg"]
        if num_ctrl_qubits == 5:
            expected_cx_counts = 48
        elif num_ctrl_qubits == 6:
            expected_cx_counts = 64
        else:
            expected_cx_counts = 16 * num_ctrl_qubits - 24
            expected_t_counts = 16 * num_ctrl_qubits - 12
            self.assertEqual(t_count, expected_t_counts)
        # The bound from the documentation of synth_mcx_1_clean_b95
        self.assertEqual(cx_count, expected_cx_counts)

    @data(3, 5, 10, 15)
    def test_mcx_1_clean_kg24_counts(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_clean_kg24 bound on CX count and T count."""
        synthesized_circuit = synth_mcx_1_clean_kg24(num_ctrl_qubits)
        transpiled_circuit = self.pm_cliff_t.run(synthesized_circuit)
        count_ops = transpiled_circuit.count_ops()

        cx_count = count_ops["cx"]
        t_count = count_ops["t"] + count_ops["tdg"]
        # Based on the bound from the Sec 5.1 of arXiv:2407.17966, assuming relative-phase Toffoli
        # decomposition
        self.assertEqual(cx_count, 6 * num_ctrl_qubits - 6)
        self.assertEqual(t_count, 8 * num_ctrl_qubits - 9)

    @data(3, 5, 10, 15)
    def test_mcx_1_dirty_kg24_counts(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_dirty_kg24 bound on CX count and T count."""
        synthesized_circuit = synth_mcx_1_dirty_kg24(num_ctrl_qubits)
        transpiled_circuit = self.pm_cliff_t.run(synthesized_circuit)
        count_ops = transpiled_circuit.count_ops()

        cx_count = count_ops["cx"]
        t_count = count_ops["t"] + count_ops["tdg"]
        # Based on the bound from the Sec 5.3 of arXiv:2407.17966, assuming relative-phase Toffoli
        # decomposition
        self.assertEqual(cx_count, 12 * num_ctrl_qubits - 18)
        self.assertEqual(t_count, 16 * num_ctrl_qubits - 26)

    @data(3, 5, 10, 15)
    def test_mcx_2_clean_kg24_counts(self, num_ctrl_qubits: int):
        """Test synth_mcx_2_clean_kg24 bound on CX count and T count."""
        synthesized_circuit = synth_mcx_2_clean_kg24(num_ctrl_qubits)
        transpiled_circuit = self.pm_cliff_t.run(synthesized_circuit)
        count_ops = transpiled_circuit.count_ops()

        cx_count = count_ops["cx"]
        t_count = count_ops["t"] + count_ops["tdg"]
        # Based on the bound from the Sec 5.2 of arXiv:2407.17966, assuming relative-phase Toffoli
        # decomposition
        self.assertEqual(cx_count, 6 * num_ctrl_qubits - 6)
        self.assertEqual(t_count, 8 * num_ctrl_qubits - 9)

    @data(3, 5, 10, 15)
    def test_mcx_2_dirty_kg24_counts(self, num_ctrl_qubits: int):
        """Test synth_mcx_2_dirty_kg24 bound on CX count and T count."""
        synthesized_circuit = synth_mcx_2_dirty_kg24(num_ctrl_qubits)
        transpiled_circuit = self.pm_cliff_t.run(synthesized_circuit)
        count_ops = transpiled_circuit.count_ops()

        cx_count = count_ops["cx"]
        t_count = count_ops["t"] + count_ops["tdg"]
        # Based on the bound from the Sec 5.4 of arXiv:2407.17966, assuming relative-phase Toffoli
        # decomposition
        self.assertEqual(cx_count, 12 * num_ctrl_qubits - 18)
        self.assertEqual(t_count, 16 * num_ctrl_qubits - 26)

    def test_c3x_cx_count(self):
        """Test synth_c3x bound on CX count."""
        synthesized_circuit = synth_c3x()
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the default construction for C3X
        self.assertEqual(cx_count, 14)

    def test_c4x_cx_count(self):
        """Test synth_c4x bound on CX count."""
        synthesized_circuit = synth_c4x()
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the default construction for C4X
        self.assertEqual(cx_count, 36)

    @combine(
        num_ctrl_qubits=[5, 10, 15],
        base_gate=[RXGate(0.123), RYGate(0.456), RZGate(0.789)],
        annotated=[False, True],
    )
    def test_mc_rotation_gates_cx_count(
        self, num_ctrl_qubits: int, base_gate: Gate, annotated: bool
    ):
        """Test bounds on the number of CX gates for mcrx / mcry / mcrz."""
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(
            base_gate.control(num_ctrl_qubits, annotated=annotated), range(num_ctrl_qubits + 1)
        )
        transpiled_circuit = self.pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The synthesis of mcrx/mcry/mcrz gates uses _mcsu2_real_diagonal.
        # The bounds are given in arXiv:2302.06377, Theorem 3.
        # In practice, we actually get better bounds for small values of num_ctrl_qubits.
        expected_cx_count = 16 * (num_ctrl_qubits + 1) - 40
        self.assertLessEqual(cx_count, expected_cx_count)

    @data(10, 15, 20)
    def test_synth_mcx_noaux_sp22(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_sp22 bound on CX count."""
        synthesized_circuit = synth_mcx_noaux_sp22(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the documentation of synth_mcx_noaux_sp22
        self.assertEqual(cx_count, 4 * num_ctrl_qubits**2 - 4 * num_ctrl_qubits + 2)

    @data(10, 15, 20)
    def test_mcx_noaux_v24_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_v24 bound on CX count."""
        synthesized_circuit = synth_mcx_noaux_v24(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the documentation of synth_mcp_noaux_v24
        self.assertEqual(cx_count, 8 * num_ctrl_qubits**2 - 16 * num_ctrl_qubits - 60)

    @data(25, 30, 35, 40, 45, 50, 55, 60)
    def test_mcx_noaux_hp24_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_hp24 bound on CX count (for large values of ``num_ctrl_qubits``."""
        synthesized_circuit = synth_mcx_noaux_hp24(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        num_qubits = num_ctrl_qubits + 1
        # bound from the paper
        if num_qubits % 2 == 0:
            expected_cx_bound = 124 * num_qubits - 214
        else:
            expected_cx_bound = 132 * num_qubits - 358
        self.assertEqual(cx_count, expected_cx_bound)

    @combine(
        num_ctrl_qubits=[5, 8, 10, 13, 15],
        base_gate=[
            XGate(),
            YGate(),
            ZGate(),
            HGate(),
            SGate(),
            SdgGate(),
            TGate(),
            TdgGate(),
            SXGate(),
            SXdgGate(),
            PhaseGate(0.345),
        ],
        annotated=[False, True],
    )
    def test_mcx_equiv_noaux_cx_count(self, num_ctrl_qubits: int, base_gate: Gate, annotated: bool):
        """Test bounds on the number of CX-gates when synthesizing multi-controlled gates
        which are locally equivalent to MCX.
        """
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(
            base_gate.control(num_ctrl_qubits, annotated=annotated), range(num_ctrl_qubits + 1)
        )
        transpiled_circuit = self.pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bounds should be the same as for synth_mcx_noaux_sp22
        self.assertLessEqual(cx_count, 4 * num_ctrl_qubits**2 - 4 * num_ctrl_qubits + 2)

    @combine(
        num_ctrl_qubits=[10, 15, 20],
        annotated=[False, True],
    )
    def test_mcu_noaux_cx_count(self, num_ctrl_qubits: int, annotated: bool):
        """Test bounds on the number of CX-gates when synthesizing multi-controlled single-qubit
        unitary gates.
        """
        base_gate = UGate(0.123, 0.456, 0.789)
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(
            base_gate.control(num_ctrl_qubits, annotated=annotated), range(num_ctrl_qubits + 1)
        )
        transpiled_circuit = self.pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The synthesis of MCU(n) uses two MCRZ(n), one MCRY(n), and one MCPhase(n-1).
        # Thus the number of CX-gate should be upper-bounded by
        # 3 * (16 * (n + 1) - 40) + 8 * (n - 1) ** 2 - 16 * (n -1) - 60
        self.assertLessEqual(cx_count, 8 * num_ctrl_qubits**2 + 16 * num_ctrl_qubits - 108)

    @combine(
        num_ctrl_qubits=[1, 2, 3, 4, 5, 6, 7, 8],
        base_gate=[
            XGate(),
            YGate(),
            ZGate(),
            HGate(),
            PhaseGate(0.123),
            SGate(),
            SdgGate(),
            TGate(),
            TdgGate(),
            SXGate(),
            SXdgGate(),
            RXGate(0.789),
            RYGate(0.123),
            RZGate(0.456),
            UGate(0.1, 0.2, 0.3),
            U1Gate(0.1),
            U2Gate(0.1, 0.2),
            U3Gate(0.1, 0.2, 0.3),
            CZGate(),
        ],
        annotated=[False, True],
    )
    def test_small_mc_gates_cx_count(self, num_ctrl_qubits: int, base_gate: Gate, annotated: bool):
        """Test that transpiling various multi-controlled gates with small number of controls and no
        ancillas yields the expected number of CX gates.

        This test prevents making changes to the synthesis algorithms that would deteriorate the
        quality of the synthesized circuits.
        """
        qc = QuantumCircuit(num_ctrl_qubits + base_gate.num_qubits)
        qc.append(base_gate.control(num_ctrl_qubits, annotated=annotated), qc.qubits)
        transpiled_circuit = self.pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]

        if isinstance(base_gate, (XGate, YGate, ZGate, HGate)):
            # MCX gate and other locally equivalent multi-controlled gates
            expected = {1: 1, 2: 6, 3: 14, 4: 36, 5: 82, 6: 122, 7: 170, 8: 226}
        elif isinstance(
            base_gate, (PhaseGate, SGate, SdgGate, TGate, TdgGate, SXGate, SXdgGate, U1Gate)
        ):
            # MCPhase gate and other locally equivalent multi-controlled gates
            expected = {1: 2, 2: 6, 3: 20, 4: 44, 5: 82, 6: 122, 7: 170, 8: 226}
        elif isinstance(base_gate, RZGate):
            expected = {1: 2, 2: 4, 3: 14, 4: 24, 5: 40, 6: 56, 7: 80, 8: 104}
        elif isinstance(base_gate, (RXGate, RYGate)):
            expected = {1: 2, 2: 8, 3: 20, 4: 24, 5: 40, 6: 56, 7: 80, 8: 104}
        elif isinstance(base_gate, (UGate, U2Gate, U3Gate)):
            expected = {1: 2, 2: 22, 3: 54, 4: 92, 5: 164, 6: 250, 7: 362, 8: 482}
        elif isinstance(base_gate, CZGate):
            expected = {1: 6, 2: 14, 3: 36, 4: 82, 5: 122, 6: 170, 7: 226, 8: 290}
        else:
            raise NotImplementedError

        self.assertEqual(cx_count, expected[num_ctrl_qubits])

    @combine(
        gate_class=[XGate, CXGate, CCXGate, C3XGate, C4XGate],
        ctrl_state_original=[None, 0, 1],
        num_ctrl_qubits_new=[2],
        ctrl_state_new=[None, 1, 2],
        annotated=[False, True],
    )
    def test_open_controlled_x_family_gates_count(
        self,
        gate_class,
        ctrl_state_original,
        num_ctrl_qubits_new,
        ctrl_state_new,
        annotated,
    ):
        """Test that transpiling controlled X, CX, CCX, C3X, C4X gates works correctly
        and produces expected CX-counts.
        """
        if gate_class == XGate:
            gate = gate_class()
        else:
            gate = gate_class(ctrl_state=ctrl_state_original)

        cgate = gate.control(
            num_ctrl_qubits=num_ctrl_qubits_new, ctrl_state=ctrl_state_new, annotated=annotated
        )
        qc = QuantumCircuit(cgate.num_qubits)
        qc.append(cgate, qc.qubits)
        transpiled_circuit = self.pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        self.assertEqual(Operator(cgate), Operator(transpiled_circuit))

        expected = {1: 1, 2: 6, 3: 14, 4: 36, 5: 84, 6: 140, 7: 220, 8: 324}
        self.assertLessEqual(cx_count, expected[cgate.num_ctrl_qubits])

    @combine(
        num_ctrl_qubits_original=[1, 2, 3, 4, 5],
        ctrl_state_original=[None, 0, 1],
        num_ctrl_qubits_new=[2],
        ctrl_state_new=[None, 1, 2],
        annotated=[False, True],
    )
    def test_open_controlled_mcx_gates_count(
        self,
        num_ctrl_qubits_original,
        ctrl_state_original,
        num_ctrl_qubits_new,
        ctrl_state_new,
        annotated,
    ):
        """Test that transpiling controlled multi-controlled X gates works correctly
        and produces expected CX-counts.
        """
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits_original, ctrl_state=ctrl_state_original)
        cgate = gate.control(
            num_ctrl_qubits=num_ctrl_qubits_new, ctrl_state=ctrl_state_new, annotated=annotated
        )
        qc = QuantumCircuit(cgate.num_qubits)
        qc.append(cgate, qc.qubits)
        transpiled_circuit = self.pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        self.assertEqual(Operator(cgate), Operator(transpiled_circuit))

        expected = {1: 1, 2: 6, 3: 14, 4: 36, 5: 82, 6: 122, 7: 170, 8: 324}
        self.assertEqual(cx_count, expected[cgate.num_ctrl_qubits])


@ddt
class TestMCSynthesisDepth(QiskitTestCase):
    """Test circuit depth produced by multi-controlled synthesis methods."""

    def setUp(self):
        super().setUp()
        self.pm = generate_preset_pass_manager(
            optimization_level=0,
            basis_gates=["u", "cx"],
            seed_transpiler=12345,
            qubits_initially_zero=False,
        )

    @data(10, 15, 20)
    def test_synth_mcp_noaux_sp22(self, num_ctrl_qubits: int):
        """Test synth_mcp_noaux_sp22 bound on depth."""
        synthesized_circuit = synth_mcp_noaux_sp22(num_ctrl_qubits, phase=0.123)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        depth2q = transpiled_circuit.depth(filter_function=lambda x: x.operation.num_qubits == 2)
        # For the exact calculation see Fig. 1 in SP22:
        # The total number of CRX/CP layers is:
        # n + (n-1) + (n-2) + (n-1) + (n-1) + (n-2) + (n-3) + (n-2) = 8*n-12
        # where n=num_ctrl_qubits.
        # Now, since each CRX/CP gate can be decomposed with 2 CX gates,
        # the bound on the 2-qubit depth after transpilation is: 16*n-24
        self.assertLessEqual(depth2q, 16 * num_ctrl_qubits - 24)

    @data(10, 15, 20)
    def test_synth_mcp_noaux_default(self, num_ctrl_qubits: int):
        """Test synth_mcp_noaux_default bound on depth."""
        synthesized_circuit = synth_mcp_noaux_default(num_ctrl_qubits, phase=0.123)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        depth2q = transpiled_circuit.depth(filter_function=lambda x: x.operation.num_qubits == 2)
        self.assertLessEqual(depth2q, 16 * num_ctrl_qubits - 24)

    @data(10, 15, 20)
    def test_synth_mcx_noaux_sp22(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_sp22 bound on depth."""
        synthesized_circuit = synth_mcx_noaux_sp22(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        depth2q = transpiled_circuit.depth(filter_function=lambda x: x.operation.num_qubits == 2)
        # For the exact calculation see test_synth_mcp_noaux_sp22:
        self.assertLessEqual(depth2q, 16 * num_ctrl_qubits - 24)

    @data(5, 10, 25)
    def test_synth_mcx_2_clean_kg24_depth(self, num_ctrl_qubits: int):
        """Test synth_mcx_2_clean_kg24 circuit depth bound.

        The algorithm itself uses O(log k) depth with binary-tree AND reduction (Khattar & Gidney,
        Sec 5.2), but this test checks depth after transpilation (all gate types,
        optimization_level=0), matching how the reference depths below were obtained from the Rust
        implementation. Basis-gate decomposition adds overhead and can affect parallelism, so these
        post-transpilation numbers do not themselves scale as O(log k); this is a regression guard
        against depth blowup, not a check of the algorithm's asymptotic complexity.

        Reference depths (Rust, after transpilation with optimization_level=0):
        k=5: 70, k=10: 104, k=25: 178
        """
        synthesized_circuit = synth_mcx_2_clean_kg24(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        depth = transpiled_circuit.depth()  # General depth (all gates)

        # Expected bounds from the reference (Rust) implementation.
        expected = {5: 70, 10: 104, 25: 178}
        if num_ctrl_qubits in expected:
            self.assertLessEqual(depth, expected[num_ctrl_qubits])

    @data(5, 10, 25)
    def test_synth_mcx_2_dirty_kg24_depth(self, num_ctrl_qubits: int):
        """Test synth_mcx_2_dirty_kg24 circuit depth bound (logarithmic with toggle detection).

        Dirty variant repeats the log-depth tree reduction for toggle detection (Khattar & Gidney, Sec 5.4).
        Depth is measured after transpilation (all gate types, optimization_level=0), matching how
        the reference depths below were obtained from the Rust implementation.

        Reference depths (Rust, after transpilation with optimization_level=0):
        k=5: 118, k=10: 186, k=25: 334
        """
        synthesized_circuit = synth_mcx_2_dirty_kg24(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        depth = transpiled_circuit.depth()  # General depth (all gates)

        # Expected bounds from the reference (Rust) implementation.
        expected = {5: 118, 10: 186, 25: 334}
        if num_ctrl_qubits in expected:
            self.assertLessEqual(depth, expected[num_ctrl_qubits])


if __name__ == "__main__":
    unittest.main()
