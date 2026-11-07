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
from qiskit.circuit.controlflow import ForLoopOp
from qiskit.circuit.classical import expr, types
from qiskit.transpiler import PassManager
from qiskit.transpiler.exceptions import TranspilerError
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

    def test_unroll_constant_expr_range(self):
        """Constant expr.Range unrolls with OpenQASM 3 inclusive-stop semantics."""
        range_expr = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(5, types.Uint(8)))
        circuit = QuantumCircuit(1, 1)
        with circuit.for_loop(range_expr):
            circuit.h(0)
            circuit.measure(0, 0)

        expected = QuantumCircuit(1, 1)
        # Inclusive [0:1:5] → six iterations (0..5).
        for _ in range(6):
            expected.h(0)
            expected.measure(0, 0)

        passmanager = PassManager()
        passmanager.append(UnrollForLoops())
        result = passmanager.run(circuit)
        self.assertEqual(result, expected)

    def test_unroll_constant_expr_range_with_var(self):
        """Constant expr.Range with an expr.Var loop variable unrolls by substituting the
        Var with a constant Value on every iteration.

        The loop variable indexes a register (it is the body's input var, so the body cannot
        also capture an outer ``expr.Var``); unrolling substitutes the Var in the index
        expression of each iteration.
        """
        indexset = expr.Range(
            expr.lift(0, types.Uint(8)), expr.lift(10, types.Uint(8)), expr.lift(2, types.Uint(8))
        )
        cr = ClassicalRegister(16, "cr")
        circuit = QuantumCircuit(QuantumRegister(1), cr)
        with circuit.for_loop(indexset) as i:
            circuit.store(expr.index(cr, i), expr.lift(True))

        passmanager = PassManager()
        passmanager.append(UnrollForLoops())
        result = passmanager.run(circuit)

        self.assertFalse(any(inst.operation.name == "for_loop" for inst in result.data))
        store_indices = [
            inst.operation.lvalue.index for inst in result.data if inst.operation.name == "store"
        ]
        self.assertEqual(
            store_indices,
            [expr.lift(value, types.Uint(8)) for value in range(0, 11, 2)],
        )

    def test_unroll_constant_expr_range_with_explicit_var(self):
        """Explicit expr.Var loop variables unroll the same as auto-generated ones."""
        loop_var = expr.Var.new("i", types.Uint(8))
        indexset = expr.Range(
            expr.lift(0, types.Uint(8)), expr.lift(6, types.Uint(8)), expr.lift(2, types.Uint(8))
        )
        cr = ClassicalRegister(16, "cr")
        circuit = QuantumCircuit(QuantumRegister(1), cr)
        with circuit.for_loop(indexset, loop_var):
            circuit.store(expr.index(cr, loop_var), expr.lift(True))

        passmanager = PassManager()
        passmanager.append(UnrollForLoops())
        result = passmanager.run(circuit)

        self.assertFalse(any(inst.operation.name == "for_loop" for inst in result.data))
        store_indices = [
            inst.operation.lvalue.index for inst in result.data if inst.operation.name == "store"
        ]
        self.assertEqual(
            store_indices,
            [expr.lift(value, types.Uint(8)) for value in range(0, 7, 2)],
        )

    def test_unroll_var_in_if_else_condition(self):
        """Unrolling substitutes expr.Var loop variables in nested if-else conditions.

        The nested bodies act on qubits/clbits only (not an outer ``expr.Var``) because a body
        with the input loop var cannot also capture outer variables.
        """
        indexset = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(3, types.Uint(8)))
        loop_var = expr.Var.new("i", types.Uint(8))

        circuit = QuantumCircuit(1, 1)

        true_body = QuantumCircuit(1, 1)
        true_body.x(0)
        false_body = QuantumCircuit(1, 1)
        false_body.measure(0, 0)

        body = QuantumCircuit(1, 1, inputs=[loop_var])
        body.if_else(
            expr.equal(loop_var, expr.lift(1, types.Uint(8))),
            true_body,
            false_body,
            [0],
            [0],
        )

        circuit.append(ForLoopOp(indexset, loop_var, body), [0], [0])

        passmanager = PassManager()
        passmanager.append(UnrollForLoops())
        result = passmanager.run(circuit)

        if_ops = [inst for inst in result.data if inst.operation.name == "if_else"]
        # Inclusive Range(0, 3) → iterations 0,1,2,3.
        self.assertEqual(len(if_ops), 4)
        self.assertEqual(
            [inst.operation.condition for inst in if_ops],
            [
                expr.equal(expr.lift(0, types.Uint(8)), expr.lift(1, types.Uint(8))),
                expr.equal(expr.lift(1, types.Uint(8)), expr.lift(1, types.Uint(8))),
                expr.equal(expr.lift(2, types.Uint(8)), expr.lift(1, types.Uint(8))),
                expr.equal(expr.lift(3, types.Uint(8)), expr.lift(1, types.Uint(8))),
            ],
        )

    def test_skip_continue_expr_range_with_var(self):
        """Unrolling is skipped when a continue appears in an expr.Range loop body."""
        indexset = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(5, types.Uint(8)))
        cr = ClassicalRegister(16, "cr")
        circuit = QuantumCircuit(QuantumRegister(1), cr)
        with circuit.for_loop(indexset) as loop_var:
            circuit.store(expr.index(cr, loop_var), expr.lift(True))
            circuit.continue_loop()

        result = UnrollForLoops()(circuit)
        self.assertEqual(result, circuit)

    def test_skip_non_constant_expr_range(self):
        """Non-constant expr.Range is left unchanged when strict is False."""
        qc = QuantumCircuit(1)
        start_var = qc.add_var("start", expr.lift(0, types.Uint(8)))
        stop_var = qc.add_var("stop", expr.lift(10, types.Uint(10)))
        range_expr = expr.Range(start_var, stop_var)
        with qc.for_loop(range_expr):
            qc.h(0)

        passmanager = PassManager()
        passmanager.append(UnrollForLoops(strict=False))
        result = passmanager.run(qc)
        self.assertEqual(result, qc)

    def test_strict_raises_non_constant_expr_range(self):
        """Non-constant expr.Range raises with strict=True."""
        qc = QuantumCircuit(1)
        start_var = qc.add_var("start", expr.lift(0, types.Uint(8)))
        stop_var = qc.add_var("stop", expr.lift(10, types.Uint(10)))
        range_expr = expr.Range(start_var, stop_var)
        with qc.for_loop(range_expr):
            qc.h(0)

        passmanager = PassManager()
        passmanager.append(UnrollForLoops(strict=True))
        with self.assertRaises(TranspilerError):
            passmanager.run(qc)

    def test_max_target_depth_constant_expr_range(self):
        """max_target_depth applies after materializing constant expr.Range."""
        indexset = expr.Range(
            expr.lift(0, types.Uint(8)), expr.lift(10, types.Uint(8)), expr.lift(2, types.Uint(8))
        )
        cr = ClassicalRegister(16, "cr")
        circuit = QuantumCircuit(QuantumRegister(3), cr)
        with circuit.for_loop(indexset) as i:
            circuit.h([0, 1, 2])
            circuit.store(expr.index(cr, i), expr.lift(True))

        passmanager = PassManager()
        # Body has quantum depth 1, inclusive range has 6 values: 6*1 > max_target_depth=2 → skip.
        passmanager.append(UnrollForLoops(max_target_depth=2))
        result = passmanager.run(circuit)
        self.assertEqual(result, circuit)


if __name__ == "__main__":
    unittest.main()
