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

"""Test OptimizeAnnotated pass"""

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import SwapGate, XGate, CXGate, HGate
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    ControlModifier,
    InverseModifier,
    PowerModifier,
)
from qiskit.transpiler.passes import OptimizeAnnotated
from qiskit.quantum_info import Operator
from test import QiskitTestCase


class TestOptimizeSwapBeforeMeasure(QiskitTestCase):
    """Test optimizations related to annotated operations."""

    def test_combine_modifiers(self):
        """Test that the pass correctly combines modifiers."""
        gate1 = AnnotatedOperation(
            SwapGate(),
            [
                InverseModifier(),
                ControlModifier(2),
                PowerModifier(4),
                InverseModifier(),
                ControlModifier(1),
                PowerModifier(-0.5),
            ],
        )
        gate2 = AnnotatedOperation(SwapGate(), [InverseModifier(), InverseModifier()])
        gate3 = AnnotatedOperation(
            AnnotatedOperation(CXGate(), ControlModifier(2)), ControlModifier(1)
        )
        gate4 = AnnotatedOperation(
            AnnotatedOperation(SwapGate(), InverseModifier()), InverseModifier()
        )
        gate5 = CXGate()

        gate1_expected = AnnotatedOperation(
            SwapGate(), [InverseModifier(), PowerModifier(2), ControlModifier(3)]
        )
        gate2_expected = SwapGate()
        gate3_expected = AnnotatedOperation(CXGate(), ControlModifier(3))
        gate4_expected = SwapGate()
        gate5_expected = CXGate()

        qc = QuantumCircuit(6)
        qc.append(gate1, [3, 2, 4, 0, 5])
        qc.append(gate2, [1, 5])
        qc.append(gate3, [5, 4, 3, 2, 1])
        qc.append(gate4, [1, 2])
        qc.append(gate5, [4, 2])

        qc_optimized = OptimizeAnnotated()(qc)

        qc_expected = QuantumCircuit(6)
        qc_expected.append(gate1_expected, [3, 2, 4, 0, 5])
        qc_expected.append(gate2_expected, [1, 5])
        qc_expected.append(gate3_expected, [5, 4, 3, 2, 1])
        qc_expected.append(gate4_expected, [1, 2])
        qc_expected.append(gate5_expected, [4, 2])

        self.assertEqual(qc_optimized, qc_expected)

    def test_combine_open_control_modifiers(self):
        """Test recursively combining open-controlled modifiers.
        Regression test of https://github.com/Qiskit/qiskit/issues/16410.
        """
        with self.subTest("control0-control1"):
            gate = AnnotatedOperation(
                AnnotatedOperation(XGate(), ControlModifier(1, ctrl_state=0)),
                ControlModifier(1, ctrl_state=1),
            )

            qc = QuantumCircuit(3)
            qc.append(gate, [0, 1, 2])

            qct = OptimizeAnnotated()(qc)
            self.assertEqual(Operator(qc), Operator(qct))

            expected = QuantumCircuit(3)
            expected.append(
                AnnotatedOperation(XGate(), ControlModifier(2, ctrl_state=1)), [0, 1, 2]
            )
            self.assertEqual(qct, expected)

        with self.subTest("control0-control11"):
            gate = AnnotatedOperation(
                AnnotatedOperation(XGate(), ControlModifier(1, ctrl_state=0)),
                ControlModifier(2, ctrl_state=3),
            )

            qc = QuantumCircuit(4)
            qc.append(gate, [0, 1, 2, 3])

            qct = OptimizeAnnotated()(qc)
            self.assertEqual(Operator(qc), Operator(qct))

            expected = QuantumCircuit(4)
            expected.append(
                AnnotatedOperation(XGate(), ControlModifier(3, ctrl_state=3)), [0, 1, 2, 3]
            )
            self.assertEqual(qct, expected)

        with self.subTest("control1-control0"):
            gate = AnnotatedOperation(
                AnnotatedOperation(XGate(), ControlModifier(1, ctrl_state=1)),
                ControlModifier(1, ctrl_state=0),
            )

            qc = QuantumCircuit(3)
            qc.append(gate, [0, 1, 2])

            qct = OptimizeAnnotated()(qc)
            self.assertEqual(Operator(qc), Operator(qct))

            expected = QuantumCircuit(3)
            expected.append(
                AnnotatedOperation(XGate(), ControlModifier(2, ctrl_state=2)), [0, 1, 2]
            )
            self.assertEqual(qct, expected)

        with self.subTest("control11-control0"):
            gate = AnnotatedOperation(
                AnnotatedOperation(XGate(), ControlModifier(2, ctrl_state=3)),
                ControlModifier(1, ctrl_state=0),
            )

            qc = QuantumCircuit(4)
            qc.append(gate, [0, 1, 2, 3])

            qct = OptimizeAnnotated()(qc)
            self.assertEqual(Operator(qc), Operator(qct))

            expected = QuantumCircuit(4)
            expected.append(
                AnnotatedOperation(XGate(), ControlModifier(3, ctrl_state=6)), [0, 1, 2, 3]
            )
            self.assertEqual(qct, expected)

    def test_optimize_definitions(self):
        """Test that the pass descends into gate definitions when basis_gates are defined."""
        qc_def = QuantumCircuit(3)
        qc_def.cx(0, 2)
        qc_def.append(AnnotatedOperation(CXGate(), [InverseModifier(), InverseModifier()]), [0, 1])
        qc_def.append(
            AnnotatedOperation(
                SwapGate(), [InverseModifier(), ControlModifier(1), InverseModifier()]
            ),
            [0, 1, 2],
        )

        expected_qc_def_optimized = QuantumCircuit(3)
        expected_qc_def_optimized.cx(0, 2)
        expected_qc_def_optimized.cx(0, 1)
        expected_qc_def_optimized.append(
            AnnotatedOperation(SwapGate(), ControlModifier(1)), [0, 1, 2]
        )

        gate = Gate("custom_gate", 3, [])
        gate.definition = qc_def

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.append(gate, [0, 1, 3])

        # Add "swap" to the basis gates to prevent conjugate reduction from replacing
        # control-[SWAP] by CX(0,1) -- CCX(1, 0) -- CX(0, 1)
        qc_optimized = OptimizeAnnotated(basis_gates=["cx", "u", "swap"])(qc)
        self.assertEqual(qc_optimized[1].operation.definition, expected_qc_def_optimized)

    def test_do_not_optimize_definitions_without_basis_gates(self):
        """
        Test that the pass does not descend into gate definitions when neither the
        target nor basis_gates are defined.
        """
        qc_def = QuantumCircuit(3)
        qc_def.cx(0, 2)
        qc_def.append(AnnotatedOperation(CXGate(), [InverseModifier(), InverseModifier()]), [0, 1])
        qc_def.append(
            AnnotatedOperation(
                SwapGate(), [InverseModifier(), ControlModifier(1), InverseModifier()]
            ),
            [0, 1, 2],
        )

        gate = Gate("custom_gate", 3, [])
        gate.definition = qc_def

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.append(gate, [0, 1, 3])

        qc_optimized = OptimizeAnnotated()(qc)
        self.assertEqual(qc_optimized[1].operation.definition, qc_def)

    def test_do_not_optimize_definitions_without_recurse(self):
        """
        Test that the pass does not descend into gate definitions when recurse is
        False.
        """
        qc_def = QuantumCircuit(3)
        qc_def.cx(0, 2)
        qc_def.append(AnnotatedOperation(CXGate(), [InverseModifier(), InverseModifier()]), [0, 1])
        qc_def.append(
            AnnotatedOperation(
                SwapGate(), [InverseModifier(), ControlModifier(1), InverseModifier()]
            ),
            [0, 1, 2],
        )

        gate = Gate("custom_gate", 3, [])
        gate.definition = qc_def

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.append(gate, [0, 1, 3])

        qc_optimized = OptimizeAnnotated(basis_gates=["cx", "u"], recurse=False)(qc)
        self.assertEqual(qc_optimized[1].operation.definition, qc_def)

    def test_if_else(self):
        """Test optimizations with if-else block."""

        true_body = QuantumCircuit(3)
        true_body.h(0)
        true_body.append(
            AnnotatedOperation(CXGate(), [InverseModifier(), InverseModifier()]), [0, 1]
        )
        false_body = QuantumCircuit(3)
        false_body.append(
            AnnotatedOperation(
                SwapGate(), [InverseModifier(), ControlModifier(1), InverseModifier()]
            ),
            [0, 1, 2],
        )

        qc = QuantumCircuit(3, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.if_else((0, True), true_body, false_body, [0, 1, 2], [])

        qc_optimized = OptimizeAnnotated()(qc)

        expected_true_body_optimized = QuantumCircuit(3)
        expected_true_body_optimized.h(0)
        expected_true_body_optimized.append(CXGate(), [0, 1])
        expected_false_body_optimized = QuantumCircuit(3)
        expected_false_body_optimized.append(
            AnnotatedOperation(SwapGate(), ControlModifier(1)), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3, 1)
        expected_qc.h(0)
        expected_qc.measure(0, 0)
        expected_qc.if_else(
            (0, True), expected_true_body_optimized, expected_false_body_optimized, [0, 1, 2], []
        )

        self.assertEqual(qc_optimized, expected_qc)

    def test_conjugate_reduction(self):
        """Test conjugate reduction optimization."""

        # Create a control-annotated operation.
        # The definition of the base operation has conjugate decomposition P -- Q -- R with R = P^{-1}
        qc_def = QuantumCircuit(6)
        qc_def.cx(0, 1)  # P
        qc_def.z(0)  # P
        qc_def.s(0)  # P
        qc_def.cx(0, 4)  # P
        qc_def.cx(4, 3)  # P
        qc_def.y(3)  # Q
        qc_def.cx(3, 0)  # Q
        qc_def.cx(4, 3)  # R
        qc_def.cx(0, 4)  # R
        qc_def.sdg(0)  # R
        qc_def.z(0)  # R
        qc_def.cx(0, 1)  # R
        qc_def.z(5)  # P
        qc_def.z(5)  # R
        qc_def.x(2)  # Q
        custom = qc_def.to_gate().control(annotated=True)

        # Create a quantum circuit with an annotated operation
        qc = QuantumCircuit(8)
        qc.cx(0, 2)
        qc.append(custom, [0, 1, 3, 4, 5, 7, 6])
        qc.h(0)
        qc.z(4)

        qc_keys = qc.count_ops().keys()
        self.assertIn("annotated", qc_keys)

        # Run optimization pass
        qc_optimized = OptimizeAnnotated(basis_gates=["cx", "u"])(qc)

        # The pass should simplify the gate
        qc_optimized_keys = qc_optimized.count_ops().keys()
        self.assertIn("optimized", qc_optimized_keys)
        self.assertNotIn("annotated", qc_optimized_keys)
        self.assertEqual(Operator(qc), Operator(qc_optimized))

    def test_conjugate_reduction_collection(self):
        """Test conjugate reduction optimization including an assertion on which gates
        are collected (using annotated gate from the previous example).
        """

        # Create a control-annotated operation.
        # The definition of the base operation has conjugate decomposition P -- Q -- R with R = P^{-1}
        qc_def = QuantumCircuit(6)
        qc_def.cx(0, 1)  # P
        qc_def.z(0)  # P
        qc_def.s(0)  # P
        qc_def.cx(0, 4)  # P
        qc_def.cx(4, 3)  # P
        qc_def.y(3)  # Q
        qc_def.cx(3, 0)  # Q
        qc_def.cx(4, 3)  # R
        qc_def.cx(0, 4)  # R
        qc_def.sdg(0)  # R
        qc_def.z(0)  # R
        qc_def.cx(0, 1)  # R
        qc_def.z(5)  # P
        qc_def.z(5)  # R
        qc_def.x(2)  # Q
        custom = qc_def.to_gate().control(annotated=True)

        # Create a quantum circuit with an annotated operation
        qc = QuantumCircuit(8)
        qc.append(custom, [0, 1, 3, 4, 5, 7, 6])

        # Run optimization pass
        qc_optimized = OptimizeAnnotated(basis_gates=["cx", "u"])(qc)

        # Check that the optimization is correct
        self.assertEqual(Operator(qc), Operator(qc_optimized))

        # Check that the optimization finds correct pairs of inverse gates
        new_def_ops = dict(qc_optimized[0].operation.definition.count_ops())
        self.assertEqual(new_def_ops, {"annotated": 1, "s": 1, "sdg": 1, "z": 4, "cx": 6})

    def test_conjugate_reduction_consecutive_gates(self):
        """Test conjugate reduction optimization including an assertion on which gates
        are collected (multiple consecutive gates on the same pair of qubits).
        """

        # Create a control-annotated operation.
        # the definition of the base operation has conjugate decomposition P -- Q -- R with R = P^{-1}
        qc_def = QuantumCircuit(6)
        qc_def.cx(0, 1)  # P
        qc_def.swap(0, 1)  # P
        qc_def.cz(1, 2)  # Q
        qc_def.swap(0, 1)  # R
        qc_def.cx(0, 1)  # R
        custom = qc_def.to_gate().control(annotated=True)

        # Create a quantum circuit with an annotated operation.
        qc = QuantumCircuit(8)
        qc.append(custom, [0, 1, 3, 4, 5, 7, 6])

        # Run optimization pass
        qc_optimized = OptimizeAnnotated(basis_gates=["cx", "u"])(qc)

        # Check that the optimization is correct
        self.assertEqual(Operator(qc), Operator(qc_optimized))

        # Check that the optimization finds correct pairs of inverse gates
        new_def_ops = dict(qc_optimized[0].operation.definition.count_ops())
        self.assertEqual(new_def_ops, {"annotated": 1, "cx": 2, "swap": 2})

    def test_conjugate_reduction_chain_of_gates(self):
        """Test conjugate reduction optimization including an assertion on which gates
        are collected (chain of gates).
        """

        # Create a control-annotated operation.
        # the definition of the base operation has conjugate decomposition P -- Q -- R with R = P^{-1}
        qc_def = QuantumCircuit(6)
        qc_def.cx(0, 1)  # P
        qc_def.cx(1, 2)  # P
        qc_def.cx(2, 3)  # P
        qc_def.h(3)  # Q
        qc_def.cx(2, 3)  # R
        qc_def.cx(1, 2)  # R
        qc_def.cx(0, 1)  # R
        custom = qc_def.to_gate().control(annotated=True)

        # Create a quantum circuit with an annotated operation.
        qc = QuantumCircuit(8)
        qc.append(custom, [0, 1, 3, 4, 5, 7, 6])

        # Run optimization pass
        qc_optimized = OptimizeAnnotated(basis_gates=["cx", "u"])(qc)

        # Check that the optimization is correct
        self.assertEqual(Operator(qc), Operator(qc_optimized))

        # Check that the optimization finds correct pairs of inverse gates
        new_def_ops = dict(qc_optimized[0].operation.definition.count_ops())
        self.assertEqual(new_def_ops, {"annotated": 1, "cx": 6})

    def test_conjugate_reduction_empty_middle(self):
        """Test conjugate reduction optimization including an assertion on which gates
        are collected (with no gates in the middle circuit).
        """

        # Create a control-annotated operation.
        # the definition of the base operation has conjugate decomposition P -- Q -- R with R = P^{-1}
        qc_def = QuantumCircuit(6)
        qc_def.cx(0, 1)  # P
        qc_def.swap(0, 1)  # P
        qc_def.cz(1, 2)  # P
        qc_def.cz(1, 2)  # R
        qc_def.swap(0, 1)  # R
        qc_def.cx(0, 1)  # R
        custom = qc_def.to_gate().control(annotated=True)

        # Create a quantum circuit with an annotated operation.
        qc = QuantumCircuit(8)
        qc.append(custom, [0, 1, 3, 4, 5, 7, 6])

        # Run optimization pass
        qc_optimized = OptimizeAnnotated(basis_gates=["cx", "u"])(qc)

        # Check that the optimization is correct
        self.assertEqual(Operator(qc), Operator(qc_optimized))

        # Check that the optimization finds correct pairs of inverse gates
        new_def_ops = dict(qc_optimized[0].operation.definition.count_ops())
        self.assertEqual(new_def_ops, {"annotated": 1, "cx": 2, "cz": 2, "swap": 2})

    def test_conjugate_reduction_parallel_gates(self):
        """Test conjugate reduction optimization including an assertion on which gates
        are collected (multiple gates in front and back layers).
        """

        # Create a control-annotated operation.
        # the definition of the base operation has conjugate decomposition P -- Q -- R with R = P^{-1}
        qc_def = QuantumCircuit(6)
        qc_def.cx(0, 1)  # P
        qc_def.swap(2, 3)  # P
        qc_def.cz(4, 5)  # P
        qc_def.h(0)  # Q
        qc_def.h(1)  # Q
        qc_def.cx(0, 1)  # R
        qc_def.swap(2, 3)  # R
        qc_def.cz(4, 5)  # R
        custom = qc_def.to_gate().control(annotated=True)

        # Create a quantum circuit with an annotated operation.
        qc = QuantumCircuit(8)
        qc.append(custom, [0, 1, 3, 4, 5, 7, 6])

        # Run optimization pass
        qc_optimized = OptimizeAnnotated(basis_gates=["cx", "u"])(qc)

        # Check that the optimization is correct
        self.assertEqual(Operator(qc), Operator(qc_optimized))

        # Check that the optimization finds correct pairs of inverse gates
        new_def_ops = dict(qc_optimized[0].operation.definition.count_ops())
        self.assertEqual(new_def_ops, {"annotated": 1, "cx": 2, "cz": 2, "swap": 2})

    def test_conjugate_reduction_cswap(self):
        """Test conjugate reduction optimization for control-SWAP."""

        # Create a circuit with a control-annotated swap
        qc = QuantumCircuit(4)
        qc.append(SwapGate().control(2, annotated=True), [0, 1, 2, 3])

        # Run optimization pass
        qc_optimized = OptimizeAnnotated(basis_gates=["cx", "u"])(qc)

        # Check that the optimization is correct
        self.assertEqual(Operator(qc), Operator(qc_optimized))

        # CSwap(0, 1) gets translated to CCX(0, 1), CCX(1, 0), CCX(0, 1).
        # The first and the last of the CXs should be detected as inverse of each other.
        new_def_ops = dict(qc_optimized[0].operation.definition.count_ops())
        self.assertEqual(new_def_ops, {"annotated": 1, "cx": 2})

    def test_standalone_var(self):
        """Test that standalone vars work."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        qc = QuantumCircuit(3, 3, inputs=[a])
        qc.add_var(b, 12)
        qc.append(AnnotatedOperation(HGate(), [ControlModifier(1), ControlModifier(1)]), [0, 1, 2])
        qc.append(AnnotatedOperation(CXGate(), [InverseModifier(), InverseModifier()]), [0, 1])
        qc.measure([0, 1, 2], [0, 1, 2])
        qc.store(a, expr.logic_and(qc.clbits[0], qc.clbits[1]))

        expected = qc.copy_empty_like()
        expected.store(b, 12)
        expected.append(HGate().control(2, annotated=True), [0, 1, 2])
        expected.cx(0, 1)
        expected.measure([0, 1, 2], [0, 1, 2])
        expected.store(a, expr.logic_and(expected.clbits[0], expected.clbits[1]))

        self.assertEqual(OptimizeAnnotated()(qc), expected)
