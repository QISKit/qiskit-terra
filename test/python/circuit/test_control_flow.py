# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test operations on control flow for dynamic QuantumCircuits."""

import copy
import math

from ddt import ddt, data, unpack, idata

from qiskit.circuit import Clbit, ClassicalRegister, Instruction, Parameter, QuantumCircuit, Qubit
from qiskit import transpile
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import CASE_DEFAULT, condition_resources, node_resources
from qiskit.circuit.library import XGate, RXGate
from qiskit.circuit.exceptions import CircuitError

from qiskit.circuit.controlflow import (
    ControlFlowOp,
    WhileLoopOp,
    ForLoopOp,
    IfElseOp,
    ContinueLoopOp,
    BreakLoopOp,
    SwitchCaseOp,
    BoxOp,
)
from test import QiskitTestCase


CONDITION_PARAMETRISATION = (
    (Clbit(), True),
    (ClassicalRegister(3, "test_creg"), 3),
    (ClassicalRegister(3, "test_creg"), True),
    expr.lift(Clbit()),
    expr.logic_not(Clbit()),
    expr.equal(ClassicalRegister(3, "test_creg"), 3),
    expr.not_equal(1, ClassicalRegister(3, "test_creg")),
)


@ddt
class TestCreatingControlFlowOperations(QiskitTestCase):
    """Tests instantiation of instruction subclasses for dynamic QuantumCircuits."""

    @idata(CONDITION_PARAMETRISATION)
    def test_while_loop_instantiation(self, condition):
        """Verify creation and properties of a WhileLoopOp."""
        body = QuantumCircuit(3, 1)
        resources = condition_resources(condition)
        body.add_bits(resources.clbits)
        for reg in resources.cregs:
            body.add_register(reg)

        op = WhileLoopOp(condition, body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "while_loop")
        self.assertEqual(op.num_qubits, body.num_qubits)
        self.assertEqual(op.num_clbits, body.num_clbits)
        self.assertEqual(op.condition, condition)
        self.assertEqual(op.params, [body])
        self.assertEqual(op.blocks, (body,))

    def test_while_loop_invalid_instantiation(self):
        """Verify we catch invalid instantiations of WhileLoopOp."""
        body = QuantumCircuit(3, 1)
        condition = (body.clbits[0], True)

        with self.assertRaisesRegex(CircuitError, r"A classical condition should be a 2-tuple"):
            _ = WhileLoopOp(0, body)

        with self.assertRaisesRegex(CircuitError, r"A classical condition should be a 2-tuple"):
            _ = WhileLoopOp((Clbit(), None), body)

        with self.assertRaisesRegex(CircuitError, r"type 'Bool\(\)'"):
            _ = WhileLoopOp(expr.Value(2, types.Uint(2)), body)

        with self.assertRaisesRegex(CircuitError, r"of type QuantumCircuit"):
            _ = WhileLoopOp(condition, XGate())

    def test_while_loop_invalid_params_setter(self):
        """Verify we catch invalid param settings for WhileLoopOp."""
        body = QuantumCircuit(3, 1)
        condition = (body.clbits[0], True)

        bad_body = QuantumCircuit(2, 1)
        op = WhileLoopOp(condition, body)
        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the WhileLoopOp"
        ):
            op.params = [bad_body]

    def test_for_loop_iterable_instantiation(self):
        """Verify creation and properties of a ForLoopOp using an iterable indexset."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = iter(range(0, 10, 2))

        body.rx(loop_parameter, 0)

        op = ForLoopOp(indexset, loop_parameter, body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "for_loop")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [tuple(range(0, 10, 2)), loop_parameter, body])
        self.assertEqual(op.blocks, (body,))

    def test_for_loop_range_instantiation(self):
        """Verify creation and properties of a ForLoopOp using a range indexset."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, 0)

        op = ForLoopOp(indexset, loop_parameter, body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "for_loop")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [indexset, loop_parameter, body])
        self.assertEqual(op.blocks, (body,))

    def test_for_loop_no_parameter_instantiation(self):
        """Verify creation and properties of a ForLoopOp without a loop_parameter."""
        body = QuantumCircuit(3, 1)
        loop_parameter = None
        indexset = range(0, 10, 2)

        body.rx(3.14, 0)

        op = ForLoopOp(indexset, loop_parameter, body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "for_loop")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [indexset, loop_parameter, body])
        self.assertEqual(op.blocks, (body,))

    def test_for_loop_negative_indexset(self):
        """Verify that ForLoopOp can be instantiated with negative integers in a list."""
        body = QuantumCircuit(1)
        body.x(0)
        indexset = [-1, 0, 1]
        op = ForLoopOp(indexset, None, body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertEqual(op.name, "for_loop")
        self.assertEqual(tuple(op.params[0]), (-1, 0, 1))

    def test_for_loop_invalid_instantiation(self):
        """Verify we catch invalid instantiations of ForLoopOp."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, 0)

        with self.assertWarnsRegex(UserWarning, r"loop_parameter was not found"):
            _ = ForLoopOp(indexset, Parameter("foo"), body)

        with self.assertRaisesRegex(CircuitError, r"to be of type QuantumCircuit"):
            _ = ForLoopOp(indexset, loop_parameter, RXGate(loop_parameter))

    def test_for_loop_expr_range_with_parameter_raises(self):
        """An ``expr.Range`` indexset cannot be paired with a compile-time ``Parameter``."""
        qc = QuantumCircuit(1)
        start_var = qc.add_var("start", expr.lift(0, types.Uint(8)))
        stop_var = qc.add_var("stop", expr.lift(10, types.Uint(10)))
        range_expr = expr.Range(start_var, stop_var)
        loop_parameter = Parameter("i")
        body = QuantumCircuit(1)
        body.rx(loop_parameter, 0)
        with self.assertRaisesRegex(CircuitError, r"compile-time Parameter"):
            ForLoopOp(range_expr, loop_parameter, body)

    def test_for_loop_python_range_with_var_counter(self):
        """A Python ``range``/integer list may be paired with an ``expr.Var`` real-time counter;
        the loop variable is the body's ``input`` variable."""
        var = expr.Var.new("i", types.Uint(8))
        body = QuantumCircuit(1, inputs=[var])
        body.h(0)
        op = ForLoopOp(range(5), var, body)
        self.assertEqual(op.params[1], var)
        self.assertEqual(op.blocks[0].num_input_vars, 1)

    def test_for_loop_var_loop_parameter_requires_body_input_var(self):
        """An ``expr.Var`` loop parameter must equal the body's single ``input`` variable."""
        var = expr.Var.new("i", types.Uint(8))
        body = QuantumCircuit(1)
        body.h(0)
        with self.assertRaisesRegex(CircuitError, r"body does not contain one"):
            ForLoopOp(range(5), var, body)

    def test_for_loop_non_constant_expr_range_without_parameter(self):
        """Non-constant expr.Range without a loop parameter is allowed."""
        qc = QuantumCircuit(1)
        start_var = qc.add_var("start", expr.lift(0, types.Uint(8)))
        stop_var = qc.add_var("stop", expr.lift(10, types.Uint(10)))
        range_expr = expr.Range(start_var, stop_var)
        body = QuantumCircuit(1)
        body.h(0)
        op = ForLoopOp(range_expr, None, body)
        self.assertEqual(op.params[0], range_expr)
        self.assertIsNone(op.params[1])

    def test_for_loop_with_var_loop_variable_explicit(self):
        """A ForLoopOp built manually with an expr.Var loop variable is valid and the
        Var appears as the body's input variable."""
        i = expr.Var.new("i", types.Uint(8))
        body = QuantumCircuit(1, inputs=[i])
        body.h(0)
        op = ForLoopOp(expr.Range(0, 4), i, body)
        self.assertEqual(op.params[1], i)
        self.assertIn(i, list(op.params[2].iter_input_vars()))

    def test_for_loop_context_manager_auto_generates_var_for_expr_range(self):
        """The for-loop context manager auto-generates an expr.Var whose type matches
        the Range, and drops it from the ForLoopOp if the user never references it."""
        qc = QuantumCircuit(1)
        # Unused: the auto-generated Var should be dropped to None.
        with qc.for_loop(expr.Range(0, 5)):
            qc.h(0)
        self.assertIsNone(qc.data[-1].operation.params[1])

        # Used inside the body (indexing a register by the loop var): the auto-generated Var is
        # kept.  We index a register rather than closing over an outer ``expr.Var`` because a body
        # with an input variable cannot also have captured variables.
        cr = ClassicalRegister(5, "cr")
        qc2 = QuantumCircuit(1)
        qc2.add_register(cr)
        with qc2.for_loop(expr.Range(0, 5)) as i:
            qc2.store(expr.index(cr, i), True)
        loop_var = qc2.data[-1].operation.params[1]
        self.assertIsInstance(loop_var, expr.Var)
        # Auto-generated Var's type follows the Range's type.
        self.assertEqual(loop_var.type, expr.Range(0, 5).type)
        # The Var is the body's input var (owned by the loop header, input into the body).
        self.assertIn(loop_var, list(qc2.data[-1].operation.params[2].iter_input_vars()))

    def test_for_loop_context_manager_keeps_explicit_var(self):
        """An explicitly-supplied Var is kept on the ForLoopOp even if the user
        never references it inside the body, matching the explicit-Parameter behavior."""
        i = expr.Var.new("i", types.Uint(8))
        qc = QuantumCircuit(1)
        with qc.for_loop(expr.Range(0, 5), i):
            qc.h(0)
        self.assertEqual(qc.data[-1].operation.params[1], i)
        self.assertIn(i, list(qc.data[-1].operation.params[2].iter_input_vars()))

    def test_for_loop_invalid_params_setter(self):
        """Verify we catch invalid param settings for ForLoopOp."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, 0)

        op = ForLoopOp(indexset, loop_parameter, body)

        with self.assertWarnsRegex(UserWarning, r"loop_parameter was not found"):
            op.params = [indexset, Parameter("foo"), body]

        with self.assertRaisesRegex(CircuitError, r"to be of type QuantumCircuit"):
            op.params = [indexset, loop_parameter, RXGate(loop_parameter)]

        bad_body = QuantumCircuit(2, 1)
        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the ForLoopOp"
        ):
            op.params = [indexset, loop_parameter, bad_body]

        with self.assertRaisesRegex(
            CircuitError, r"to be either of type Parameter, expr\.Var, or None"
        ):
            _ = ForLoopOp(indexset, "foo", body)

        qc_vars = QuantumCircuit(1)
        start_var = qc_vars.add_var("start", expr.lift(0, types.Uint(8)))
        stop_var = qc_vars.add_var("stop", expr.lift(10, types.Uint(10)))
        non_const_range = expr.Range(start_var, stop_var)
        with self.assertRaisesRegex(CircuitError, r"compile-time Parameter"):
            op.params = [non_const_range, loop_parameter, body]

    @idata(CONDITION_PARAMETRISATION)
    def test_if_else_instantiation_with_else(self, condition):
        """Verify creation and properties of a IfElseOp with an else branch."""
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        op = IfElseOp(condition, true_body, false_body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "if_else")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [true_body, false_body])
        self.assertEqual(op.condition, condition)
        self.assertEqual(op.blocks, (true_body, false_body))

    @idata(CONDITION_PARAMETRISATION)
    def test_if_else_instantiation_without_else(self, condition):
        """Verify creation and properties of a IfElseOp without an else branch."""
        true_body = QuantumCircuit(3, 1)

        op = IfElseOp(condition, true_body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "if_else")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [true_body, None])
        self.assertEqual(op.condition, condition)
        self.assertEqual(op.blocks, (true_body,))

    def test_if_else_invalid_instantiation(self):
        """Verify we catch invalid instantiations of IfElseOp."""
        condition = (Clbit(), True)
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        with self.assertRaisesRegex(CircuitError, r"A classical condition should be a 2-tuple"):
            _ = IfElseOp(1, true_body, false_body)

        with self.assertRaisesRegex(CircuitError, r"A classical condition should be a 2-tuple"):
            _ = IfElseOp((1, 2), true_body, false_body)

        with self.assertRaisesRegex(CircuitError, r"type 'Bool\(\)'"):
            _ = IfElseOp(expr.Value(2, types.Uint(2)), true_body, false_body)

        with self.assertRaisesRegex(CircuitError, r"true_body parameter of type QuantumCircuit"):
            _ = IfElseOp(condition, XGate())

        with self.assertRaisesRegex(CircuitError, r"false_body parameter of type QuantumCircuit"):
            _ = IfElseOp(condition, true_body, XGate())

        bad_body = QuantumCircuit(4, 2)

        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the IfElseOp"
        ):
            _ = IfElseOp(condition, true_body, bad_body)

    def test_if_else_invalid_params_setter(self):
        """Verify we catch invalid param settings for IfElseOp."""
        condition = (Clbit(), True)
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        op = IfElseOp(condition, true_body, false_body)

        with self.assertRaisesRegex(CircuitError, r"true_body parameter of type QuantumCircuit"):
            op.params = [XGate(), None]

        with self.assertRaisesRegex(CircuitError, r"false_body parameter of type QuantumCircuit"):
            op.params = [true_body, XGate()]

        bad_body = QuantumCircuit(4, 2)

        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the IfElseOp"
        ):
            op.params = [true_body, bad_body]

        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the IfElseOp"
        ):
            op.params = [bad_body, false_body]

        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the IfElseOp"
        ):
            op.params = [bad_body, bad_body]

    def test_continue_loop_instantiation(self):
        """Verify creation and properties of a ContinueLoopOp."""
        op = ContinueLoopOp(3, 1)

        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "continue_loop")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [])

    def test_break_loop_instantiation(self):
        """Verify creation and properties of a BreakLoopOp."""
        op = BreakLoopOp(3, 1)

        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "break_loop")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [])

    def test_switch_clbit(self):
        """Test that a switch statement can be constructed with a bit as a condition."""
        qubit = Qubit()
        clbit = Clbit()
        case1 = QuantumCircuit([qubit, clbit])
        case1.x(0)
        case2 = QuantumCircuit([qubit, clbit])
        case2.z(0)

        op = SwitchCaseOp(clbit, [(True, case1), (False, case2)])
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "switch_case")
        self.assertEqual(op.num_qubits, 1)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.target, clbit)
        self.assertEqual(op.cases(), {True: case1, False: case2})
        self.assertEqual(list(op.blocks), [case1, case2])

    def test_switch_register(self):
        """Test that a switch statement can be constructed with a register as a condition."""
        qubit = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)
        case3 = QuantumCircuit([qubit], creg)
        case3.z(0)

        op = SwitchCaseOp(creg, [(0, case1), (1, case2), (2, case3)])
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "switch_case")
        self.assertEqual(op.num_qubits, 1)
        self.assertEqual(op.num_clbits, 2)
        self.assertEqual(op.target, creg)
        self.assertEqual(op.cases(), {0: case1, 1: case2, 2: case3})
        self.assertEqual(list(op.blocks), [case1, case2, case3])

    def test_switch_expr_uint(self):
        """Test that a switch statement can be constructed with a Uint `Expr` as a condition."""
        qubit = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)
        case3 = QuantumCircuit([qubit], creg)
        case3.z(0)

        op = SwitchCaseOp(expr.lift(creg), [(0, case1), (1, case2), (2, case3)])
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "switch_case")
        self.assertEqual(op.num_qubits, 1)
        self.assertEqual(op.num_clbits, 2)
        self.assertEqual(op.target, expr.Var(creg, types.Uint(creg.size)))
        self.assertEqual(op.cases(), {0: case1, 1: case2, 2: case3})
        self.assertEqual(list(op.blocks), [case1, case2, case3])

    def test_switch_expr_bool(self):
        """Test that a switch statement can be constructed with a Bool `Expr` as a condition."""
        qubit = Qubit()
        clbit = Clbit()
        case1 = QuantumCircuit([qubit, clbit])
        case1.x(0)
        case2 = QuantumCircuit([qubit, clbit])
        case2.z(0)

        op = SwitchCaseOp(expr.logic_not(clbit), [(True, case1), (False, case2)])
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "switch_case")
        self.assertEqual(op.num_qubits, 1)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(
            op.target,
            expr.Unary(expr.Unary.Op.LOGIC_NOT, expr.Var(clbit, types.Bool()), types.Bool()),
        )
        self.assertEqual(op.cases(), {True: case1, False: case2})
        self.assertEqual(list(op.blocks), [case1, case2])

    def test_switch_with_default(self):
        """Test that a switch statement can be constructed with a default case at the end."""
        qubit = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)
        case3 = QuantumCircuit([qubit], creg)
        case3.z(0)

        op = SwitchCaseOp(creg, [(0, case1), (1, case2), (CASE_DEFAULT, case3)])
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "switch_case")
        self.assertEqual(op.num_qubits, 1)
        self.assertEqual(op.num_clbits, 2)
        self.assertEqual(op.target, creg)
        self.assertEqual(op.cases(), {0: case1, 1: case2, CASE_DEFAULT: case3})
        self.assertEqual(list(op.blocks), [case1, case2, case3])

    def test_switch_expr_with_default(self):
        """Test that a switch statement can be constructed with a default case at the end when the
        target is an `Expr`."""
        qubit = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)
        case3 = QuantumCircuit([qubit], creg)
        case3.z(0)

        target = expr.bit_xor(creg, 0b11)
        op = SwitchCaseOp(target, [(0, case1), (1, case2), (CASE_DEFAULT, case3)])
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "switch_case")
        self.assertEqual(op.num_qubits, 1)
        self.assertEqual(op.num_clbits, 2)
        self.assertEqual(op.target, target)
        self.assertEqual(op.cases(), {0: case1, 1: case2, CASE_DEFAULT: case3})
        self.assertEqual(list(op.blocks), [case1, case2, case3])

    def test_switch_multiple_cases_to_same_block(self):
        """Test that it is possible to add multiple cases that apply to the same block, if they are
        given as a compound value.  This is an allowed special case of block fall-through."""
        qubit = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        op = SwitchCaseOp(creg, [(0, case1), ((1, 2), case2)])
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "switch_case")
        self.assertEqual(op.num_qubits, 1)
        self.assertEqual(op.num_clbits, 2)
        self.assertEqual(op.target, creg)
        self.assertEqual(op.cases(), {0: case1, 1: case2, 2: case2})
        self.assertEqual(list(op.blocks), [case1, case2])

    def test_switch_reconstruction(self):
        """Test that the `cases_specifier` method can be used to reconstruct an equivalent op."""
        qubit = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        base = SwitchCaseOp(creg, [(0, case1), ((1, 2), case2)])
        self.assertEqual(base, SwitchCaseOp(creg, base.cases_specifier()))

    def test_switch_rejects_separate_cases_to_same_block(self):
        """Test that the switch statement rejects cases that are supplied separately, but point to
        the same QuantumCircuit."""
        qubit = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        with self.assertRaisesRegex(CircuitError, "ungrouped cases cannot point to the same block"):
            SwitchCaseOp(creg, [(0, case1), (1, case2), (2, case1)])

    def test_switch_rejects_cases_over_different_bits(self):
        """Test that a switch statement fails to build if its individual cases are not all defined
        over the same numbers of bits."""
        qubits = [Qubit() for _ in [None] * 3]
        clbits = [Clbit(), Clbit()]
        case1 = QuantumCircuit(qubits, clbits)
        case2 = QuantumCircuit(qubits[1:], clbits)

        for case in (case1, case2):
            case.h(1)
            case.cx(1, 0)
            case.measure(0, 0)

        with self.assertRaisesRegex(CircuitError, r"incompatible bits between cases"):
            SwitchCaseOp(Clbit(), [(True, case1), (False, case2)])

    def test_switch_rejects_cases_with_bad_types(self):
        """Test that a switch statement will fail to build if it contains cases whose types are not
        matched to the switch expression."""
        qubit = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        with self.assertRaisesRegex(CircuitError, "case values must be"):
            SwitchCaseOp(creg, [(1.3, case1), (4.5, case2)])

    def test_switch_rejects_cases_after_default(self):
        """Test that a switch statement will fail to build if there are cases after the default
        case."""
        qubit = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        with self.assertRaisesRegex(CircuitError, "cases after the default are unreachable"):
            SwitchCaseOp(creg, [(CASE_DEFAULT, case1), (1, case2)])

    def test_if_else_rejects_input_vars(self):
        """Bodies must not contain input variables."""
        cond = (Clbit(), False)
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        bad_body = QuantumCircuit(inputs=[a])
        good_body = QuantumCircuit(captures=[a], declarations=[(b, expr.lift(False))])

        with self.assertRaisesRegex(
            CircuitError, "Only for-loop blocks can contain input variables"
        ):
            IfElseOp(cond, bad_body, None)
        with self.assertRaisesRegex(
            CircuitError, "Only for-loop blocks can contain input variables"
        ):
            IfElseOp(cond, bad_body, good_body)
        with self.assertRaisesRegex(
            CircuitError, "Only for-loop blocks can contain input variables"
        ):
            IfElseOp(cond, good_body, bad_body)

    def test_while_rejects_input_vars(self):
        """Bodies must not contain input variables."""
        cond = (Clbit(), False)
        a = expr.Var.new("a", types.Bool())
        bad_body = QuantumCircuit(inputs=[a])
        with self.assertRaisesRegex(
            CircuitError, "Only for-loop blocks can contain input variables"
        ):
            WhileLoopOp(cond, bad_body)

    def test_for_input_vars(self):
        """If a body has an input variable, it must match the loop parameter."""
        a = expr.Var.new("a", types.Uint(32))
        body = QuantumCircuit(inputs=[a])
        with self.assertRaisesRegex(CircuitError, "loop variable.*disagree"):
            ForLoopOp(range(3), None, body)
        with self.assertRaisesRegex(CircuitError, "loop variable.*disagree"):
            ForLoopOp(range(3), expr.Var.new("a", types.Uint(32)), body)
        with self.assertRaisesRegex(CircuitError, "loop variable.*disagree"):
            ForLoopOp(range(3), Parameter("a"), body)
        with self.assertRaisesRegex(
            CircuitError, "ForLoopOp expects a Var loop_parameter parameter to be of type Uint"
        ):
            ForLoopOp(range(3), expr.Var.new("b", types.Bool()), body)
        self.assertEqual(list(ForLoopOp(range(3), a, body).params), [range(3), a, body])

    def test_switch_rejects_input_vars(self):
        """Bodies must not contain input variables."""
        target = ClassicalRegister(3, "cr")
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        bad_body = QuantumCircuit(inputs=[a])
        good_body = QuantumCircuit(captures=[a], declarations=[(b, expr.lift(False))])

        with self.assertRaisesRegex(
            CircuitError, "Only for-loop blocks can contain input variables"
        ):
            SwitchCaseOp(target, [(0, bad_body)])
        with self.assertRaisesRegex(
            CircuitError, "Only for-loop blocks can contain input variables"
        ):
            SwitchCaseOp(target, [(0, good_body), (1, bad_body)])

    def test_box_instantiation(self):
        """Test that we can instantiate a `BoxOp` directly."""
        body = QuantumCircuit(3, 1)
        op = BoxOp(body)
        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "box")
        self.assertEqual(op.num_qubits, body.num_qubits)
        self.assertEqual(op.num_clbits, body.num_clbits)
        self.assertEqual(op.params, [body])
        self.assertEqual(op.blocks, (body,))
        self.assertEqual(op.duration, None)
        self.assertEqual(op.label, None)

        op = BoxOp(body, duration=10.0, unit="ms", label="hello, world")
        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "box")
        self.assertEqual(op.num_qubits, body.num_qubits)
        self.assertEqual(op.num_clbits, body.num_clbits)
        self.assertEqual(op.params, [body])
        self.assertEqual(op.blocks, (body,))
        self.assertEqual(op.duration, 10.0)
        self.assertEqual(op.unit, "ms")
        self.assertEqual(op.label, "hello, world")

        a = expr.Stretch.new("a")
        op = BoxOp(body, duration=a)
        self.assertEqual(op.duration, a)
        self.assertEqual(op.unit, "expr")

        op = BoxOp(body, duration=expr.mul(2, a))
        self.assertEqual(op.duration, expr.mul(2, a))
        self.assertEqual(op.unit, "expr")

    def test_box_valid_params_setter(self):
        """Verify that valid sets to `params` function."""
        a = Parameter("a")
        body = QuantumCircuit(2, 1)
        body.h(0)
        body.rzx(a, 0, 1)
        body.measure(0, 0)

        op = BoxOp(body.copy())

        assigned = body.assign_parameters({a: 2.5}, inplace=False)
        op.params = [assigned.copy()]
        self.assertEqual(op.params, [assigned])
        self.assertEqual(op.blocks, (assigned,))
        self.assertEqual(op.body, assigned)

    def test_box_invalid_params_setter(self):
        """Verify we catch invalid param settings for BoxOp."""
        body = QuantumCircuit(3, 1)

        bad_body = QuantumCircuit(2, 1)
        op = BoxOp(body)
        with self.assertRaisesRegex(CircuitError, r"different than that of the BoxOp"):
            op.params = [bad_body]
        with self.assertRaisesRegex(CircuitError, "body parameter of type QuantumCircuit"):
            op.params = [None]

    def test_box_body(self):
        """The `body` property should work and be unsettable."""
        body = QuantumCircuit(2)
        body.h(0)
        body.cx(0, 1)

        op = BoxOp(body)
        self.assertEqual(op.body, body)
        with self.assertRaises(AttributeError):
            op.body = body.copy()

    def test_box_replace_blocks(self):
        """Test all properties copied across."""
        body0 = QuantumCircuit(1)
        body0.x(0)
        body1 = QuantumCircuit(1)
        body1.h(0)

        op = BoxOp(body0, duration=300.0, unit="ns", label="hello, world")
        expected = BoxOp(body1, duration=300.0, unit="ns", label="hello, world")
        self.assertEqual(op.replace_blocks([body1]), expected)

    def test_box_equal_respects_all_items(self):
        """Test equality of custom fields of `Box`."""
        body_a = QuantumCircuit(1)
        body_a.x(0)
        body_b = QuantumCircuit(1)
        body_b.h(0)

        self.assertEqual(BoxOp(body_a), BoxOp(body_a))
        self.assertNotEqual(BoxOp(body_a), BoxOp(body_b))

        # 'label' is not typically considered by `Instruction.__eq__`.

        base = BoxOp(body_a, duration=300.0, unit="ns")
        self.assertEqual(BoxOp(body_a, duration=300.0, unit="ns"), base)
        self.assertNotEqual(BoxOp(body_b, duration=300.0, unit="ns"), base)
        self.assertNotEqual(BoxOp(body_a, duration=100.0, unit="ns"), base)
        self.assertNotEqual(BoxOp(body_a, duration=300.0, unit="ms"), base)

    def test_box_copy_is_deep(self):
        """Copying a circuit with a box should deepcopy the body."""
        qc = QuantumCircuit(2)
        with qc.box():
            qc.cx(0, 1)

        qc_copy = qc.copy()

        self.assertIsNot(
            qc[0].operation.blocks[0],
            qc_copy[0].operation.blocks[0],
        )

        qc_deepcopy = copy.deepcopy(qc)

        self.assertIsNot(
            qc[0].operation.blocks[0],
            qc_deepcopy[0].operation.blocks[0],
        )


@ddt
class TestAddingControlFlowOperations(QiskitTestCase):
    """Tests of instruction subclasses for dynamic QuantumCircuits."""

    @data(
        (Clbit(), [False, True]),
        (ClassicalRegister(3, "test_creg"), [3, 1]),
        (ClassicalRegister(3, "test_creg"), [0, (1, 2), CASE_DEFAULT]),
        (expr.lift(Clbit()), [False, True]),
        (expr.lift(ClassicalRegister(3, "test_creg")), [3, 1]),
        (expr.bit_not(ClassicalRegister(3, "test_creg")), [0, (1, 2), CASE_DEFAULT]),
    )
    @unpack
    def test_appending_switch_case_op(self, target, labels):
        """Verify we can append a SwitchCaseOp to a QuantumCircuit."""
        bodies = [QuantumCircuit(3, 1) for _ in labels]

        op = SwitchCaseOp(target, zip(labels, bodies))

        qc = QuantumCircuit(5, 2)
        if isinstance(target, ClassicalRegister):
            qc.add_register(target)
        elif isinstance(target, Clbit):
            qc.add_bits([target])
        else:
            resources = node_resources(target)
            qc.add_bits(resources.clbits)
            for reg in resources.cregs:
                qc.add_register(reg)
        qc.append(op, [1, 2, 3], [1])

        self.assertEqual(qc.data[0].operation.name, "switch_case")
        self.assertEqual(qc.data[0].operation.params, bodies[: len(labels)])
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits[1:4]))
        self.assertEqual(qc.data[0].clbits, (qc.clbits[1],))

    @data(
        (Clbit(), [False, True]),
        (ClassicalRegister(3, "test_creg"), [3, 1]),
        (ClassicalRegister(3, "test_creg"), [0, (1, 2), CASE_DEFAULT]),
        (expr.lift(Clbit()), [False, True]),
        (expr.lift(ClassicalRegister(3, "test_creg")), [3, 1]),
        (expr.bit_not(ClassicalRegister(3, "test_creg")), [0, (1, 2), CASE_DEFAULT]),
    )
    @unpack
    def test_quantumcircuit_switch(self, target, labels):
        """Verify we can use the `QuantumCircuit.switch` method."""
        bodies = [QuantumCircuit(3, 1) for _ in labels]

        qc = QuantumCircuit(5, 2)
        if isinstance(target, ClassicalRegister):
            qc.add_register(target)
        elif isinstance(target, Clbit):
            qc.add_bits([target])
        else:
            resources = node_resources(target)
            qc.add_bits(resources.clbits)
            for reg in resources.cregs:
                qc.add_register(reg)
        qc.switch(target, zip(labels, bodies), [1, 2, 3], [1])

        self.assertEqual(qc.data[0].operation.name, "switch_case")
        self.assertEqual(qc.data[0].operation.params, bodies[: len(labels)])
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits[1:4]))
        self.assertEqual(qc.data[0].clbits, (qc.clbits[1],))

    @idata(CONDITION_PARAMETRISATION)
    def test_appending_while_loop_op(self, condition):
        """Verify we can append a WhileLoopOp to a QuantumCircuit."""
        body = QuantumCircuit(3, 1)

        op = WhileLoopOp(condition, body)

        qc = QuantumCircuit(5, 2)
        qc.append(op, [1, 2, 3], [1])

        self.assertEqual(qc.data[0].operation.name, "while_loop")
        self.assertEqual(qc.data[0].operation.params, [body])
        self.assertEqual(qc.data[0].operation.condition, condition)
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits[1:4]))
        self.assertEqual(qc.data[0].clbits, (qc.clbits[1],))

    @idata(CONDITION_PARAMETRISATION)
    def test_quantumcircuit_while_loop(self, condition):
        """Verify we can append a WhileLoopOp to a QuantumCircuit via qc.while_loop."""
        body = QuantumCircuit(3, 1)

        qc = QuantumCircuit(5, 2)
        resources = condition_resources(condition)
        qc.add_bits(resources.clbits)
        for reg in resources.cregs:
            qc.add_register(reg)
        qc.while_loop(condition, body, [1, 2, 3], [1])

        self.assertEqual(qc.data[0].operation.name, "while_loop")
        self.assertEqual(qc.data[0].operation.params, [body])
        self.assertEqual(qc.data[0].operation.condition, condition)
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits[1:4]))
        self.assertEqual(qc.data[0].clbits, (qc.clbits[1],))

    def test_appending_for_loop_op(self):
        """Verify we can append a ForLoopOp to a QuantumCircuit."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, [0, 1, 2])

        op = ForLoopOp(indexset, loop_parameter, body)

        qc = QuantumCircuit(5, 2)
        qc.append(op, [1, 2, 3], [1])

        self.assertEqual(qc.data[0].operation.name, "for_loop")
        self.assertEqual(qc.data[0].operation.params, [indexset, loop_parameter, body])
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits[1:4]))
        self.assertEqual(qc.data[0].clbits, (qc.clbits[1],))

    def test_quantumcircuit_for_loop_op(self):
        """Verify we can append a ForLoopOp to a QuantumCircuit via qc.for_loop."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, [0, 1, 2])

        qc = QuantumCircuit(5, 2)
        qc.for_loop(indexset, loop_parameter, body, [1, 2, 3], [1])

        self.assertEqual(qc.data[0].operation.name, "for_loop")
        self.assertEqual(qc.data[0].operation.params, [indexset, loop_parameter, body])
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits[1:4]))
        self.assertEqual(qc.data[0].clbits, (qc.clbits[1],))

    def test_for_loop_loop_parameter_not_in_outer_parameters(self):
        """A ForLoopOp's loop_parameter must NOT appear in the outer circuit's parameters."""
        theta = Parameter("θ")
        body = QuantumCircuit(1)
        body.rx(theta, 0)

        qc = QuantumCircuit(1)
        qc.append(ForLoopOp(range(3), theta, body), [0])

        self.assertNotIn(theta, qc.parameters)
        with self.assertRaisesRegex(CircuitError, r"not present in the circuit"):
            qc.assign_parameters({theta: math.pi / 2}, inplace=False)

    def test_for_loop_var_loop_parameter_not_in_outer_vars(self):
        """A ForLoopOp's expr.Var loop variable is the body's input var and must NOT appear in
        the outer circuit's vars."""
        loop_var = expr.Var.new("i", types.Uint(8))
        body = QuantumCircuit(1, inputs=[loop_var])
        body.h(0)

        qc = QuantumCircuit(1)
        qc.append(ForLoopOp(expr.Range(0, 3), loop_var, body), [0])

        self.assertNotIn(loop_var, qc.iter_vars())
        for_loop = qc.data[0].operation
        self.assertEqual(for_loop.params[1], loop_var)
        self.assertIn(loop_var, list(for_loop.params[2].iter_input_vars()))

    @idata(CONDITION_PARAMETRISATION)
    def test_appending_if_else_op(self, condition):
        """Verify we can append a IfElseOp to a QuantumCircuit."""
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        op = IfElseOp(condition, true_body, false_body)

        qc = QuantumCircuit(5, 2)
        resources = condition_resources(condition)
        qc.add_bits(resources.clbits)
        for reg in resources.cregs:
            qc.add_register(reg)
        qc.append(op, [1, 2, 3], [1])

        self.assertEqual(qc.data[0].operation.name, "if_else")
        self.assertEqual(qc.data[0].operation.params, [true_body, false_body])
        self.assertEqual(qc.data[0].operation.condition, condition)
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits[1:4]))
        self.assertEqual(qc.data[0].clbits, (qc.clbits[1],))

    @idata(CONDITION_PARAMETRISATION)
    def test_quantumcircuit_if_else_op(self, condition):
        """Verify we can append a IfElseOp to a QuantumCircuit via qc.if_else."""
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        qc = QuantumCircuit(5, 2)
        resources = condition_resources(condition)
        qc.add_bits(resources.clbits)
        for reg in resources.cregs:
            qc.add_register(reg)
        qc.if_else(condition, true_body, false_body, [1, 2, 3], [1])

        self.assertEqual(qc.data[0].operation.name, "if_else")
        self.assertEqual(qc.data[0].operation.params, [true_body, false_body])
        self.assertEqual(qc.data[0].operation.condition, condition)
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits[1:4]))
        self.assertEqual(qc.data[0].clbits, (qc.clbits[1],))

    @idata(CONDITION_PARAMETRISATION)
    def test_quantumcircuit_if_test_op(self, condition):
        """Verify we can append a IfElseOp to a QuantumCircuit via qc.if_test."""
        true_body = QuantumCircuit(3, 1)

        qc = QuantumCircuit(5, 2)
        resources = condition_resources(condition)
        qc.add_bits(resources.clbits)
        for reg in resources.cregs:
            qc.add_register(reg)
        qc.if_test(condition, true_body, [1, 2, 3], [1])

        self.assertEqual(qc.data[0].operation.name, "if_else")
        self.assertEqual(qc.data[0].operation.params, [true_body, None])
        self.assertEqual(qc.data[0].operation.condition, condition)
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits[1:4]))
        self.assertEqual(qc.data[0].clbits, (qc.clbits[1],))

    @idata(CONDITION_PARAMETRISATION)
    def test_appending_if_else_op_with_condition_outside(self, condition):
        """Verify we catch if IfElseOp has a condition outside outer circuit."""
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        qc = QuantumCircuit(5, 2)

        with self.assertRaisesRegex(CircuitError, r".* is not present in this circuit\."):
            qc.if_test(condition, true_body, [1, 2, 3], [1])

        with self.assertRaisesRegex(CircuitError, r".* is not present in this circuit\."):
            qc.if_else(condition, true_body, false_body, [1, 2, 3], [1])

    def test_appending_continue_loop_op(self):
        """Verify we can append a ContinueLoopOp to a QuantumCircuit."""
        op = ContinueLoopOp(3, 1)

        qc = QuantumCircuit(3, 1)
        qc.append(op, [0, 1, 2], [0])

        self.assertEqual(qc.data[0].operation.name, "continue_loop")
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits))
        self.assertEqual(qc.data[0].clbits, tuple(qc.clbits))

    def test_quantumcircuit_continue_loop_op(self):
        """Verify we can append a ContinueLoopOp to a QuantumCircuit via qc.continue_loop."""
        qc = QuantumCircuit(3, 1)
        qc.continue_loop()

        self.assertEqual(qc.data[0].operation.name, "continue_loop")
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits))
        self.assertEqual(qc.data[0].clbits, tuple(qc.clbits))

    def test_appending_break_loop_op(self):
        """Verify we can append a BreakLoopOp to a QuantumCircuit."""
        op = BreakLoopOp(3, 1)

        qc = QuantumCircuit(3, 1)
        qc.append(op, [0, 1, 2], [0])

        self.assertEqual(qc.data[0].operation.name, "break_loop")
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits))
        self.assertEqual(qc.data[0].clbits, tuple(qc.clbits))

    def test_quantumcircuit_break_loop_op(self):
        """Verify we can append a BreakLoopOp to a QuantumCircuit via qc.break_loop."""
        qc = QuantumCircuit(3, 1)
        qc.break_loop()

        self.assertEqual(qc.data[0].operation.name, "break_loop")
        self.assertEqual(qc.data[0].qubits, tuple(qc.qubits))
        self.assertEqual(qc.data[0].clbits, tuple(qc.clbits))

    def test_appending_box_op(self):
        """Verify that we can append a BoxOp to a QuantumCircuit."""
        body = QuantumCircuit(3, 1)
        op = BoxOp(body, duration=10.0, unit="ms", label="hello, world")

        qc = QuantumCircuit(5, 2)
        qc.append(op, [0, 2, 3], [1])
        self.assertEqual(qc.data[0].name, "box")
        self.assertEqual(qc.data[0].qubits, (qc.qubits[0], qc.qubits[2], qc.qubits[3]))
        self.assertEqual(qc.data[0].clbits, (qc.clbits[1],))
        self.assertEqual(qc.data[0].operation.duration, 10.0)
        self.assertEqual(qc.data[0].operation.unit, "ms")
        self.assertEqual(qc.data[0].label, "hello, world")

    def test_quantumcircuit_box(self):
        """Verify that we can append a BoxOp to a QuantumCircuit via qc.box."""
        body = QuantumCircuit(3, 1)
        qc = QuantumCircuit(5, 2)
        qc.box(body, [0, 2, 3], [1], duration=10.0, unit="ms", label="hello, world")

        self.assertEqual(qc.data[0].name, "box")
        self.assertEqual(qc.data[0].qubits, (qc.qubits[0], qc.qubits[2], qc.qubits[3]))
        self.assertEqual(qc.data[0].clbits, (qc.clbits[1],))
        self.assertEqual(qc.data[0].operation.duration, 10.0)
        self.assertEqual(qc.data[0].operation.unit, "ms")
        self.assertEqual(qc.data[0].label, "hello, world")

    def test_nested_parameters_are_recognised(self):
        """Verify that parameters added inside a control-flow operator get added to the outer
        circuit table."""
        x, y = Parameter("x"), Parameter("y")

        with self.subTest("if/else"):
            body1 = QuantumCircuit(1, 1)
            body1.rx(x, 0)
            body2 = QuantumCircuit(1, 1)
            body2.rx(y, 0)

            main = QuantumCircuit(1, 1)
            main.if_else((main.clbits[0], 0), body1, body2, [0], [0])
            self.assertEqual({x, y}, set(main.parameters))

        with self.subTest("while"):
            body = QuantumCircuit(1, 1)
            body.rx(x, 0)

            main = QuantumCircuit(1, 1)
            main.while_loop((main.clbits[0], 0), body, [0], [0])
            self.assertEqual({x}, set(main.parameters))

        with self.subTest("for"):
            body = QuantumCircuit(1, 1)
            body.rx(x, 0)

            main = QuantumCircuit(1, 1)
            main.for_loop(range(1), None, body, [0], [0])
            self.assertEqual({x}, set(main.parameters))

        with self.subTest("box"):
            body = QuantumCircuit(1)
            body.rz(x, 0)

            main = QuantumCircuit(1)
            main.box(body, [0], [])
            self.assertEqual({x}, set(main.parameters))

    def test_nested_parameters_can_be_assigned(self):
        """Verify that parameters added inside a control-flow operator can be assigned by calls to
        the outer circuit."""
        x, y = Parameter("x"), Parameter("y")

        with self.subTest("if/else"):
            body1 = QuantumCircuit(1, 1)
            body1.rx(x, 0)
            body2 = QuantumCircuit(1, 1)
            body2.rx(y, 0)

            test = QuantumCircuit(1, 1)
            test.if_else((test.clbits[0], 0), body1, body2, [0], [0])
            self.assertEqual({x, y}, set(test.parameters))
            assigned = test.assign_parameters({x: math.pi, y: 0.5 * math.pi})
            self.assertEqual(set(), set(assigned.parameters))

            expected = QuantumCircuit(1, 1)
            expected.if_else(
                (expected.clbits[0], 0),
                body1.assign_parameters({x: math.pi}),
                body2.assign_parameters({y: 0.5 * math.pi}),
                [0],
                [0],
            )

            self.assertEqual(assigned, expected)

        with self.subTest("while"):
            body = QuantumCircuit(1, 1)
            body.rx(x, 0)

            test = QuantumCircuit(1, 1)
            test.while_loop((test.clbits[0], 0), body, [0], [0])
            self.assertEqual({x}, set(test.parameters))
            assigned = test.assign_parameters({x: math.pi})
            self.assertEqual(set(), set(assigned.parameters))

            expected = QuantumCircuit(1, 1)
            expected.while_loop(
                (expected.clbits[0], 0),
                body.assign_parameters({x: math.pi}),
                [0],
                [0],
            )

            self.assertEqual(assigned, expected)

        with self.subTest("for"):
            body = QuantumCircuit(1, 1)
            body.rx(x, 0)

            test = QuantumCircuit(1, 1)
            test.for_loop(range(1), None, body, [0], [0])
            self.assertEqual({x}, set(test.parameters))
            assigned = test.assign_parameters({x: math.pi})
            self.assertEqual(set(), set(assigned.parameters))

            expected = QuantumCircuit(1, 1)
            expected.for_loop(
                range(1),
                None,
                body.assign_parameters({x: math.pi}),
                [0],
                [0],
            )

            self.assertEqual(assigned, expected)

        with self.subTest("box"):
            body = QuantumCircuit(1)
            body.rz(x, 0)
            main = QuantumCircuit(1)
            main.box(body, [0], [])
            assigned = main.assign_parameters({x: math.pi})
            self.assertEqual(set(), set(assigned.parameters))

            expected = QuantumCircuit(1)
            expected.box(body.assign_parameters({x: math.pi}), [0], [])
            self.assertEqual(assigned, expected)

    def test_can_add_op_with_captures_of_inputs(self):
        """Test circuit methods can capture input variables."""
        outer = QuantumCircuit(1, 1)
        a = outer.add_input("a", types.Bool())

        inner = QuantumCircuit(1, 1, captures=[a])

        outer.if_test((outer.clbits[0], False), inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "if_else")
        self.assertEqual(set(added.blocks[0].iter_captured_vars()), {a})

        outer.if_else((outer.clbits[0], False), inner.copy(), inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "if_else")
        self.assertEqual(set(added.blocks[0].iter_captured_vars()), {a})
        self.assertEqual(set(added.blocks[1].iter_captured_vars()), {a})

        outer.while_loop((outer.clbits[0], False), inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "while_loop")
        self.assertEqual(set(added.blocks[0].iter_captured_vars()), {a})

        outer.for_loop(range(3), None, inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "for_loop")
        self.assertEqual(set(added.blocks[0].iter_captured_vars()), {a})

        outer.switch(outer.clbits[0], [(False, inner.copy()), (True, inner.copy())], [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "switch_case")
        self.assertEqual(set(added.blocks[0].iter_captured_vars()), {a})
        self.assertEqual(set(added.blocks[1].iter_captured_vars()), {a})

        outer.box(inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "box")
        self.assertEqual(set(added.blocks[0].iter_captured_vars()), {a})

    def test_can_add_op_with_captures_of_captures(self):
        """Test circuit methods can capture captured variables."""
        outer = QuantumCircuit(1, 1)
        a = expr.Var.new("a", types.Bool())
        b = expr.Stretch.new("b")
        outer.add_capture(a)
        outer.add_capture(b)

        inner = QuantumCircuit(1, 1, captures=[a, b])

        outer.if_test((outer.clbits[0], False), inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "if_else")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})

        outer.if_else((outer.clbits[0], False), inner.copy(), inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "if_else")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})
        self.assertEqual(set(added.blocks[1].iter_captures()), {a, b})

        outer.while_loop((outer.clbits[0], False), inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "while_loop")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})

        outer.for_loop(range(3), None, inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "for_loop")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})

        outer.switch(outer.clbits[0], [(False, inner.copy()), (True, inner.copy())], [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "switch_case")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})
        self.assertEqual(set(added.blocks[1].iter_captures()), {a, b})

        outer.box(inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "box")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})

    def test_can_add_op_with_captures_of_locals(self):
        """Test circuit methods can capture declared variables."""
        outer = QuantumCircuit(1, 1)
        a = outer.add_var("a", expr.lift(True))
        b = outer.add_stretch("b")

        inner = QuantumCircuit(1, 1, captures=[a, b])

        outer.if_test((outer.clbits[0], False), inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "if_else")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})

        outer.if_else((outer.clbits[0], False), inner.copy(), inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "if_else")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})
        self.assertEqual(set(added.blocks[1].iter_captures()), {a, b})

        outer.while_loop((outer.clbits[0], False), inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "while_loop")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})

        outer.for_loop(range(3), None, inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "for_loop")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})

        outer.switch(outer.clbits[0], [(False, inner.copy()), (True, inner.copy())], [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "switch_case")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})
        self.assertEqual(set(added.blocks[1].iter_captures()), {a, b})

        outer.box(inner.copy(), [0], [0])
        added = outer.data[-1].operation
        self.assertEqual(added.name, "box")
        self.assertEqual(set(added.blocks[0].iter_captures()), {a, b})

    def test_cannot_capture_unknown_variables_methods(self):
        """Control-flow operations should not be able to capture variables that don't exist in the
        outer circuit."""
        outer = QuantumCircuit(1, 1)

        a = expr.Var.new("a", types.Bool())
        inner = QuantumCircuit(1, 1, captures=[a])

        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.if_test((outer.clbits[0], False), inner.copy(), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.if_else((outer.clbits[0], False), inner.copy(), inner.copy(), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.while_loop((outer.clbits[0], False), inner.copy(), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.for_loop(range(3), None, inner.copy(), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.switch(outer.clbits[0], [(False, inner.copy()), (True, inner.copy())], [0], [0])

    def test_cannot_capture_unknown_stretches_methods(self):
        """Control-flow operations should not be able to capture stretches that don't exist in the
        outer circuit."""
        outer = QuantumCircuit(1, 1)

        a = expr.Stretch.new("b")
        inner = QuantumCircuit(1, 1, captures=[a])

        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.if_test((outer.clbits[0], False), inner.copy(), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.if_else((outer.clbits[0], False), inner.copy(), inner.copy(), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.while_loop((outer.clbits[0], False), inner.copy(), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.for_loop(range(3), None, inner.copy(), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.switch(outer.clbits[0], [(False, inner.copy()), (True, inner.copy())], [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.box(inner.copy(), [0], [0])

    def test_cannot_capture_unknown_variables_append(self):
        """Control-flow operations should not be able to capture variables that don't exist in the
        outer circuit."""
        outer = QuantumCircuit(1, 1)

        a = expr.Var.new("a", types.Bool())
        inner = QuantumCircuit(1, 1, captures=[a])

        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(IfElseOp((outer.clbits[0], False), inner.copy(), None), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(IfElseOp((outer.clbits[0], False), inner.copy(), inner.copy()), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(WhileLoopOp((outer.clbits[0], False), inner.copy()), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(ForLoopOp(range(3), None, inner.copy()), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(
                SwitchCaseOp(outer.clbits[0], [(False, inner.copy()), (True, inner.copy())]),
                [0],
                [0],
            )
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(BoxOp(inner.copy()), [0], [0])

    def test_cannot_capture_unknown_stretches_append(self):
        """Control-flow operations should not be able to capture stretches that don't exist in the
        outer circuit."""
        outer = QuantumCircuit(1, 1)

        a = expr.Stretch.new("a")
        inner = QuantumCircuit(1, 1, captures=[a])

        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(IfElseOp((outer.clbits[0], False), inner.copy(), None), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(IfElseOp((outer.clbits[0], False), inner.copy(), inner.copy()), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(WhileLoopOp((outer.clbits[0], False), inner.copy()), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(ForLoopOp(range(3), None, inner.copy()), [0], [0])
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(
                SwitchCaseOp(outer.clbits[0], [(False, inner.copy()), (True, inner.copy())]),
                [0],
                [0],
            )
        with self.assertRaisesRegex(CircuitError, "not in this circuit"):
            outer.append(BoxOp(inner.copy()), [0], [0])

    def test_transpiling_with_control_flow(self):
        """Test we don't run into compilation errors when transpiling circuits with control flows"""
        qc = QuantumCircuit(1, 1)

        with qc.for_loop(range(1000)):
            qc.h(0)
            qc.measure(0, 0)
            with qc.if_test((0, False)):
                qc.continue_loop()
            qc.break_loop()

        transpiled = transpile(qc)
        self.assertIsNotNone(transpiled)
