# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the ResetAfterMeasureSimplification pass"""

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Clbit
from qiskit.transpiler.passes.optimization import ResetAfterMeasureSimplification
from qiskit.circuit import IfElseOp
from test import QiskitTestCase


class TestResetAfterMeasureSimplificationt(QiskitTestCase):
    """Test ResetAfterMeasureSimplification transpiler pass."""

    def test_simple(self):
        """Test simple"""
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        qc.reset(0)
        new_qc = ResetAfterMeasureSimplification()(qc)

        ans_qc = QuantumCircuit(1, 1)
        ans_qc.measure(0, 0)
        x_body = QuantumCircuit(1)
        x_body.x(0)
        new_x = IfElseOp((ans_qc.clbits[0], 1), x_body)
        ans_qc.append(new_x, [ans_qc.qubits[0]])

        self.assertEqual(new_qc, ans_qc)

    def test_simple_null(self):
        """Test simple no change in circuit"""
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        qc.x(0)
        qc.reset(0)
        new_qc = ResetAfterMeasureSimplification()(qc)

        self.assertEqual(new_qc, qc)

    def test_simple_multi_reg(self):
        """Test simple, multiple registers"""
        cr1 = ClassicalRegister(1, "c1")
        cr2 = ClassicalRegister(1, "c2")
        qr = QuantumRegister(1, "q")
        qc = QuantumCircuit(qr, cr1, cr2)
        qc.measure(0, 1)
        qc.reset(0)
        new_qc = ResetAfterMeasureSimplification()(qc)

        ans_qc = QuantumCircuit(qr, cr1, cr2)
        ans_qc.measure(0, 1)
        x_body = QuantumCircuit(1)
        x_body.x(0)
        new_x = IfElseOp((cr2[0], 1), x_body)
        ans_qc.append(new_x, [ans_qc.qubits[0]])

        self.assertEqual(new_qc, ans_qc)

    def test_simple_multi_reg_null(self):
        """Test simple, multiple registers, null change"""
        cr1 = ClassicalRegister(1, "c1")
        cr2 = ClassicalRegister(1, "c2")
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr, cr1, cr2)
        qc.measure(0, 1)
        qc.reset(1)  # reset not on same qubit as meas
        new_qc = ResetAfterMeasureSimplification()(qc)
        self.assertEqual(new_qc, qc)

    def test_simple_multi_resets(self):
        """Only first reset is collapsed"""
        qc = QuantumCircuit(1, 2)
        qc.measure(0, 0)
        qc.reset(0)
        qc.reset(0)
        new_qc = ResetAfterMeasureSimplification()(qc)

        ans_qc = QuantumCircuit(1, 2)
        ans_qc.measure(0, 0)
        x_body = QuantumCircuit(1)
        x_body.x(0)
        new_x = IfElseOp((ans_qc.clbits[0], 1), x_body)
        ans_qc.append(new_x, [ans_qc.qubits[0]])
        ans_qc.reset(0)
        self.assertEqual(new_qc, ans_qc)

    def test_simple_multi_resets_with_resets_before_measure(self):
        """Reset BEFORE measurement not collapsed"""
        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        qc.reset(0)
        qc.reset(1)
        qc.measure(1, 1)

        new_qc = ResetAfterMeasureSimplification()(qc)

        ans_qc = QuantumCircuit(2, 2)
        ans_qc.measure(0, 0)
        x_body = QuantumCircuit(1)
        x_body.x(0)
        new_x = IfElseOp((Clbit(ClassicalRegister(2, "c"), 0), 1), x_body)
        ans_qc.append(new_x, [ans_qc.qubits[0]])
        ans_qc.reset(1)
        ans_qc.measure(1, 1)

        self.assertEqual(new_qc, ans_qc)

    def test_barriers_work(self):
        """Test that barriers block consolidation"""
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        qc.barrier(0)
        qc.reset(0)

        new_qc = ResetAfterMeasureSimplification()(qc)
        self.assertEqual(new_qc, qc)

    def test_bv_circuit(self):
        """Test Bernstein Vazirani circuit with midcircuit measurement."""
        bitstring = "11111"
        qc = QuantumCircuit(2, len(bitstring))
        qc.x(1)
        qc.h(1)
        for idx, bit in enumerate(bitstring[::-1]):
            qc.h(0)
            if int(bit):
                qc.cx(0, 1)
            qc.h(0)
            qc.measure(0, idx)
            if idx != len(bitstring) - 1:
                qc.reset(0)
                # reset control
                qc.reset(1)
                qc.x(1)
                qc.h(1)
        new_qc = ResetAfterMeasureSimplification()(qc)
        for op in new_qc.data:
            if op.operation.name == "reset":
                self.assertEqual(op.qubits[0], new_qc.qubits[1])

    def test_simple_if_else(self):
        """Test that the pass recurses into an if-else."""
        pass_ = ResetAfterMeasureSimplification()

        base_test = QuantumCircuit(1, 1)
        base_test.measure(0, 0)
        base_test.reset(0)

        base_expected = QuantumCircuit(1, 1)
        base_expected.measure(0, 0)
        x_body = QuantumCircuit(1)
        x_body.x(0)
        new_x = IfElseOp((base_expected.clbits[0], True), x_body)
        base_expected.append(new_x, [base_expected.qubits[0]])

        test = QuantumCircuit(1, 1)
        test.if_else(
            (test.clbits[0], True), base_test.copy(), base_test.copy(), test.qubits, test.clbits
        )

        expected = QuantumCircuit(1, 1)
        expected.if_else(
            (expected.clbits[0], True),
            base_expected.copy(),
            base_expected.copy(),
            expected.qubits,
            expected.clbits,
        )

        self.assertEqual(pass_(test), expected)

    def test_nested_control_flow(self):
        """Test that the pass recurses into nested control flow."""
        pass_ = ResetAfterMeasureSimplification()

        base_test = QuantumCircuit(1, 1)
        base_test.measure(0, 0)
        base_test.reset(0)

        base_expected = QuantumCircuit(1, 1)
        base_expected.measure(0, 0)
        with base_expected.if_test((0, True)):
            base_expected.x(0)

        body_test = QuantumCircuit(1, 1)
        body_test.for_loop((0,), None, base_expected.copy(), body_test.qubits, body_test.clbits)

        body_expected = QuantumCircuit(1, 1)
        body_expected.for_loop(
            (0,), None, base_expected.copy(), body_expected.qubits, body_expected.clbits
        )

        test = QuantumCircuit(1, 1)
        test.while_loop((test.clbits[0], True), body_test, test.qubits, test.clbits)

        expected = QuantumCircuit(1, 1)
        expected.while_loop(
            (expected.clbits[0], True), body_expected, expected.qubits, expected.clbits
        )
        self.assertEqual(pass_(test), expected)
