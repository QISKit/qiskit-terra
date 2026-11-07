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


import itertools

from test import QiskitTestCase

from qiskit.circuit import QuantumCircuit, CircuitError, Clbit, ClassicalRegister, Store, Parameter
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import ForLoopOp, IfElseOp, WhileLoopOp, BoxOp
from qiskit.circuit.controlflow.switch_case import SwitchCaseOp, CASE_DEFAULT


class TestCircuitVars(QiskitTestCase):
    """Tests for variable-manipulation routines on circuits.  More specific functionality is likely
    tested in the suites of the specific methods."""

    def test_initialise_inputs(self):
        vars_ = [
            expr.Var.new("a", types.Bool()),
            expr.Var.new("b", types.Uint(16)),
        ]
        qc = QuantumCircuit(inputs=vars_)
        self.assertEqual(set(vars_), set(qc.iter_vars()))
        self.assertEqual(qc.num_vars, len(vars_))
        self.assertEqual(qc.num_input_vars, len(vars_))
        self.assertEqual(qc.num_captured_vars, 0)
        self.assertEqual(qc.num_declared_vars, 0)

    def test_initialise_captures(self):
        vars_ = [
            expr.Var.new("a", types.Bool()),
            expr.Var.new("b", types.Uint(16)),
        ]
        stretches_ = [
            expr.Stretch.new("c"),
        ]
        qc = QuantumCircuit(captures=itertools.chain(vars_, stretches_))
        self.assertEqual(set(vars_), set(qc.iter_vars()))
        self.assertEqual(set(stretches_), set(qc.iter_stretches()))
        self.assertEqual(qc.num_vars, len(vars_))
        self.assertEqual(qc.num_stretches, len(stretches_))
        self.assertEqual(qc.num_input_vars, 0)
        self.assertEqual(qc.num_captured_vars, len(vars_))
        self.assertEqual(qc.num_captured_stretches, len(stretches_))
        self.assertEqual(qc.num_declared_vars, 0)
        self.assertEqual(qc.num_declared_stretches, 0)

    def test_initialise_declarations_iterable(self):
        vars_ = [
            (expr.Var.new("a", types.Bool()), expr.lift(True)),
            (expr.Var.new("b", types.Uint(16)), expr.lift(0xFFFF)),
        ]
        qc = QuantumCircuit(declarations=vars_)

        self.assertEqual({var for var, _initialiser in vars_}, set(qc.iter_vars()))
        self.assertEqual(qc.num_vars, len(vars_))
        self.assertEqual(qc.num_input_vars, 0)
        self.assertEqual(qc.num_captured_vars, 0)
        self.assertEqual(qc.num_declared_vars, len(vars_))
        operations = [
            (instruction.operation.name, instruction.operation.lvalue, instruction.operation.rvalue)
            for instruction in qc.data
        ]
        self.assertEqual(
            operations,
            [("store", lvalue, rvalue) for lvalue, rvalue in vars_],
        )

    def test_initialise_declarations_mapping(self):
        # Dictionary iteration order is guaranteed to be insertion order.
        vars_ = {
            expr.Var.new("a", types.Bool()): expr.lift(True),
            expr.Var.new("b", types.Uint(16)): expr.lift(0xFFFF),
        }
        qc = QuantumCircuit(declarations=vars_)

        self.assertEqual(set(vars_), set(qc.iter_vars()))
        operations = [
            (instruction.operation.name, instruction.operation.lvalue, instruction.operation.rvalue)
            for instruction in qc.data
        ]
        self.assertEqual(
            operations, [("store", lvalue, rvalue) for lvalue, rvalue in vars_.items()]
        )

    def test_initialise_declarations_dependencies(self):
        """Test that the circuit initializer can take in declarations with dependencies between
        them, provided they're specified in a suitable order."""
        a = expr.Var.new("a", types.Bool())
        vars_ = [
            (a, expr.lift(True)),
            (expr.Var.new("b", types.Bool()), a),
        ]
        qc = QuantumCircuit(declarations=vars_)

        self.assertEqual({var for var, _initialiser in vars_}, set(qc.iter_vars()))
        operations = [
            (instruction.operation.name, instruction.operation.lvalue, instruction.operation.rvalue)
            for instruction in qc.data
        ]
        self.assertEqual(operations, [("store", lvalue, rvalue) for lvalue, rvalue in vars_])

    def test_initialise_inputs_declarations(self):
        a = expr.Var.new("a", types.Uint(16))
        b = expr.Var.new("b", types.Uint(16))
        b_init = expr.bit_and(a, 0xFFFF)
        qc = QuantumCircuit(inputs=[a], declarations={b: b_init})

        self.assertEqual({a}, set(qc.iter_input_vars()))
        self.assertEqual({b}, set(qc.iter_declared_vars()))
        self.assertEqual({a, b}, set(qc.iter_vars()))
        self.assertEqual(qc.num_vars, 2)
        self.assertEqual(qc.num_input_vars, 1)
        self.assertEqual(qc.num_captured_vars, 0)
        self.assertEqual(qc.num_declared_vars, 1)
        operations = [
            (instruction.operation.name, instruction.operation.lvalue, instruction.operation.rvalue)
            for instruction in qc.data
        ]
        self.assertEqual(operations, [("store", b, b_init)])

    def test_initialise_captures_declarations(self):
        a = expr.Var.new("a", types.Uint(16))
        b = expr.Var.new("b", types.Uint(16))
        b_init = expr.bit_and(a, 0xFFFF)
        qc = QuantumCircuit(captures=[a], declarations={b: b_init})

        self.assertEqual({a}, set(qc.iter_captured_vars()))
        self.assertEqual({b}, set(qc.iter_declared_vars()))
        self.assertEqual({a, b}, set(qc.iter_vars()))
        self.assertEqual(qc.num_vars, 2)
        self.assertEqual(qc.num_input_vars, 0)
        self.assertEqual(qc.num_captured_vars, 1)
        self.assertEqual(qc.num_declared_vars, 1)
        operations = [
            (instruction.operation.name, instruction.operation.lvalue, instruction.operation.rvalue)
            for instruction in qc.data
        ]
        self.assertEqual(operations, [("store", b, b_init)])

    def test_add_uninitialized_var(self):
        a = expr.Var.new("a", types.Bool())
        qc = QuantumCircuit()
        qc.add_uninitialized_var(a)
        self.assertEqual({a}, set(qc.iter_vars()))
        self.assertEqual([], list(qc.data))

    def test_add_var_returns_good_var(self):
        qc = QuantumCircuit()
        a = qc.add_var("a", expr.lift(True))
        self.assertEqual(a.name, "a")
        self.assertEqual(a.type, types.Bool())

        b = qc.add_var("b", expr.Value(0xFF, types.Uint(8)))
        self.assertEqual(b.name, "b")
        self.assertEqual(b.type, types.Uint(8))

    def test_add_stretch_returns_good_var(self):
        qc = QuantumCircuit()
        a = qc.add_stretch("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.type, types.Duration())

    def test_add_var_returns_input(self):
        """Test that the `Var` returned by `add_var` is the same as the input if `Var`."""
        a = expr.Var.new("a", types.Bool())
        qc = QuantumCircuit()
        a_other = qc.add_var(a, expr.lift(True))
        self.assertEqual(a, a_other)

    def test_add_stretch_returns_input(self):
        a = expr.Stretch.new("a")
        qc = QuantumCircuit()
        a_other = qc.add_stretch(a)
        self.assertEqual(a, a_other)

    def test_stretch_circuit_equality(self):
        a = expr.Stretch.new("a")
        b = expr.Stretch.new("b")
        c = expr.Stretch.new("c")

        # Capture order doesn't matter in circuit equality!
        qc1 = QuantumCircuit(captures=[a, b, c])
        self.assertEqual(qc1, QuantumCircuit(captures=[a, b, c]))
        self.assertEqual(qc1, QuantumCircuit(captures=[c, b, a]))

        qc1 = QuantumCircuit()
        qc1.add_stretch(a)
        qc1.add_stretch(b)
        qc1.add_stretch(c)

        qc2 = QuantumCircuit()
        qc2.add_stretch(c)
        qc2.add_stretch(b)
        qc2.add_stretch(a)

        # But declaration order does!
        self.assertNotEqual(qc1, qc2)

        qc1 = QuantumCircuit(captures=[a, b, c])
        qc2 = QuantumCircuit(captures=[a])
        qc2.add_stretch(b)
        qc2.add_stretch(c)
        self.assertNotEqual(qc1, qc2)

        qc1 = qc2.copy()
        self.assertEqual(qc1, qc2)

        qc2 = QuantumCircuit(captures=[a])
        qc2.add_stretch(c)
        qc2.add_stretch(b)
        self.assertNotEqual(qc1, qc2)

    def test_add_input_returns_good_var(self):
        qc = QuantumCircuit()
        a = qc.add_input("a", types.Bool())
        self.assertEqual(a.name, "a")
        self.assertEqual(a.type, types.Bool())

        b = qc.add_input("b", types.Uint(8))
        self.assertEqual(b.name, "b")
        self.assertEqual(b.type, types.Uint(8))

    def test_add_input_returns_input(self):
        """Test that the `Var` returned by `add_input` is the same as the input if `Var`."""
        a = expr.Var.new("a", types.Bool())
        qc = QuantumCircuit()
        a_other = qc.add_input(a)
        self.assertEqual(a, a_other)

    def test_cannot_have_both_inputs_and_captures(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        c = expr.Stretch.new("c")

        with self.assertRaisesRegex(CircuitError, "circuits with input.*cannot be closures"):
            QuantumCircuit(inputs=[a], captures=[b])

        with self.assertRaisesRegex(CircuitError, "circuits with input.*cannot be closures"):
            QuantumCircuit(inputs=[a], captures=[c])

        qc = QuantumCircuit(inputs=[a])
        with self.assertRaisesRegex(CircuitError, "circuits with input.*cannot be closures"):
            qc.add_capture(b)

        qc = QuantumCircuit(inputs=[a])
        with self.assertRaisesRegex(CircuitError, "circuits with input.*cannot be closures"):
            qc.add_capture(c)

        qc = QuantumCircuit(captures=[a])
        with self.assertRaisesRegex(CircuitError, "circuits to be enclosed.*cannot have input"):
            qc.add_input(b)

        qc = QuantumCircuit(captures=[c])
        with self.assertRaisesRegex(CircuitError, "circuits to be enclosed.*cannot have input"):
            qc.add_input(b)

    def test_cannot_add_cyclic_declaration(self):
        a = expr.Var.new("a", types.Bool())
        with self.assertRaisesRegex(CircuitError, "not present in this circuit"):
            QuantumCircuit(declarations=[(a, a)])

        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "not present in this circuit"):
            qc.add_var(a, a)

    def test_initialise_inputs_equal_to_add_input(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(16))

        qc_init = QuantumCircuit(inputs=[a, b])
        qc_manual = QuantumCircuit()
        qc_manual.add_input(a)
        qc_manual.add_input(b)
        self.assertEqual(list(qc_init.iter_vars()), list(qc_manual.iter_vars()))

        qc_manual = QuantumCircuit()
        a = qc_manual.add_input("a", types.Bool())
        b = qc_manual.add_input("b", types.Uint(16))
        qc_init = QuantumCircuit(inputs=[a, b])
        self.assertEqual(list(qc_init.iter_vars()), list(qc_manual.iter_vars()))

    def test_initialise_captures_equal_to_add_capture(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(16))
        c = expr.Stretch.new("c")

        qc_init = QuantumCircuit(captures=[a, b, c])
        qc_manual = QuantumCircuit()
        qc_manual.add_capture(a)
        qc_manual.add_capture(b)
        qc_manual.add_capture(c)
        self.assertEqual(list(qc_init.iter_vars()), list(qc_manual.iter_vars()))
        self.assertEqual(list(qc_init.iter_stretches()), list(qc_manual.iter_stretches()))

    def test_initialise_declarations_equal_to_add_var(self):
        a = expr.Var.new("a", types.Bool())
        a_init = expr.lift(False)
        b = expr.Var.new("b", types.Uint(16))
        b_init = expr.lift(0xFFFF)

        qc_init = QuantumCircuit(declarations=[(a, a_init), (b, b_init)])
        qc_manual = QuantumCircuit()
        qc_manual.add_var(a, a_init)
        qc_manual.add_var(b, b_init)
        self.assertEqual(list(qc_init.iter_vars()), list(qc_manual.iter_vars()))
        self.assertEqual(qc_init.data, qc_manual.data)

        qc_manual = QuantumCircuit()
        a = qc_manual.add_var("a", a_init)
        b = qc_manual.add_var("b", b_init)
        qc_init = QuantumCircuit(declarations=[(a, a_init), (b, b_init)])
        self.assertEqual(list(qc_init.iter_vars()), list(qc_manual.iter_vars()))
        self.assertEqual(qc_init.data, qc_manual.data)

    def test_declarations_widen_integer_literals(self):
        a = expr.Var.new("a", types.Uint(8))
        b = expr.Var.new("b", types.Uint(16))
        qc = QuantumCircuit(declarations=[(a, 3)])
        qc.add_var(b, 5)
        actual_initializers = [
            (op.lvalue, op.rvalue)
            for instruction in qc
            if isinstance((op := instruction.operation), Store)
        ]
        expected_initializers = [
            (a, expr.Value(3, types.Uint(8))),
            (b, expr.Value(5, types.Uint(16))),
        ]
        self.assertEqual(actual_initializers, expected_initializers)

    def test_declaration_does_not_widen_bool_literal(self):
        # `bool` is a subclass of `int` in Python (except some arithmetic operations have different
        # semantics...).  It's not in Qiskit's value type system, though.
        a = expr.Var.new("a", types.Uint(8))
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "explicit cast is required"):
            qc.add_var(a, True)

    def test_cannot_shadow_vars(self):
        """Test that exact duplicate ``Var`` nodes within different combinations of the inputs are
        detected and rejected."""
        a = expr.Var.new("a", types.Bool())
        a_init = expr.lift(True)
        with self.assertRaisesRegex(CircuitError, "already present"):
            QuantumCircuit(inputs=[a, a])
        with self.assertRaisesRegex(CircuitError, "already present"):
            QuantumCircuit(captures=[a, a])
        with self.assertRaisesRegex(CircuitError, "already present"):
            QuantumCircuit(declarations=[(a, a_init), (a, a_init)])
        with self.assertRaisesRegex(CircuitError, "already present"):
            QuantumCircuit(inputs=[a], declarations=[(a, a_init)])
        with self.assertRaisesRegex(CircuitError, "already present"):
            QuantumCircuit(captures=[a], declarations=[(a, a_init)])

    def test_cannot_shadow_stretches(self):
        """Test that exact duplicate ``Stretch`` nodes within different combinations of the inputs are
        detected and rejected."""
        a = expr.Stretch.new("a")
        with self.assertRaisesRegex(CircuitError, "already present"):
            QuantumCircuit(captures=[a, a])
        qc = QuantumCircuit(captures=[a])
        with self.assertRaisesRegex(CircuitError, "already present"):
            qc.add_stretch(a)
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "already present"):
            qc.add_stretch(a)
            qc.add_stretch(a)

    def test_cannot_shadow_names(self):
        """Test that exact duplicate ``Var`` nodes within different combinations of the inputs are
        detected and rejected."""
        a_bool1 = expr.Var.new("a", types.Bool())
        a_bool2 = expr.Var.new("a", types.Bool())
        a_uint = expr.Var.new("a", types.Uint(16))
        a_bool_init = expr.lift(True)
        a_uint_init = expr.lift(0xFFFF)

        tests = [
            ((a_bool1, a_bool_init), (a_bool2, a_bool_init)),
            ((a_bool1, a_bool_init), (a_uint, a_uint_init)),
        ]
        for (left, left_init), (right, right_init) in tests:
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                QuantumCircuit(inputs=(left, right))
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                QuantumCircuit(captures=(left, right))
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                QuantumCircuit(declarations=[(left, left_init), (right, right_init)])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                QuantumCircuit(inputs=[left], declarations=[(right, right_init)])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                QuantumCircuit(captures=[left], declarations=[(right, right_init)])

            qc = QuantumCircuit(inputs=[left])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                qc.add_input(right)
            qc = QuantumCircuit(inputs=[left])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                qc.add_var(right, right_init)

            qc = QuantumCircuit(captures=[left])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                qc.add_capture(right)
            qc = QuantumCircuit(captures=[left])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                qc.add_var(right, right_init)

            qc = QuantumCircuit(inputs=[left])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                qc.add_var(right, right_init)

        qc = QuantumCircuit()
        qc.add_var("a", expr.lift(True))
        with self.assertRaisesRegex(CircuitError, "its name shadows"):
            qc.add_var("a", expr.lift(True))
        with self.assertRaisesRegex(CircuitError, "its name shadows"):
            qc.add_var("a", expr.lift(0xFF))

        a_stretch1 = expr.Stretch.new("a")
        a_stretch2 = expr.Stretch.new("a")
        with self.assertRaisesRegex(CircuitError, "its name shadows"):
            QuantumCircuit(captures=[a_stretch1, a_stretch2])
        with self.assertRaisesRegex(CircuitError, "its name shadows"):
            QuantumCircuit(captures=[a_stretch1, a_bool1])
        with self.assertRaisesRegex(CircuitError, "its name shadows"):
            QuantumCircuit(captures=[a_bool1, a_stretch1])
        with self.assertRaisesRegex(CircuitError, "its name shadows"):
            QuantumCircuit(captures=[a_stretch1], declarations=[(a_bool1, a_bool_init)])
        qc = QuantumCircuit(declarations=[(a_bool1, a_bool_init)])
        with self.assertRaisesRegex(CircuitError, "its name shadows"):
            qc.add_stretch(a_stretch1)
        with self.assertRaisesRegex(CircuitError, "its name shadows"):
            qc.add_stretch("a")
        qc = QuantumCircuit()
        qc.add_stretch("a")
        with self.assertRaisesRegex(CircuitError, "its name shadows"):
            qc.add_var("a", expr.lift(True))

    def test_cannot_add_vars_wrapping_clbits(self):
        a = expr.Var(Clbit(), types.Bool())
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(inputs=[a])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_input(a)
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(captures=[a])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_capture(a)
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(declarations=[(a, expr.lift(True))])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_var(a, expr.lift(True))

    def test_cannot_add_vars_wrapping_cregs(self):
        a = expr.Var(ClassicalRegister(8, "cr"), types.Uint(8))
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(inputs=[a])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_input(a)
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(captures=[a])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_capture(a)
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(declarations=[(a, expr.lift(0xFF))])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_var(a, expr.lift(0xFF))

    def test_get_var_success(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        qc = QuantumCircuit(inputs=[a], declarations={b: expr.Value(0xFF, types.Uint(8))})
        self.assertEqual(qc.get_var("a"), a)
        self.assertEqual(qc.get_var("b"), b)

        qc = QuantumCircuit(captures=[a, b])
        self.assertEqual(qc.get_var("a"), a)
        self.assertEqual(qc.get_var("b"), b)

        qc = QuantumCircuit(
            inputs=[],
            declarations={
                a: expr.lift(True),
                b: expr.Value(0xFF, types.Uint(8)),
            },
        )
        self.assertEqual(qc.get_var("a"), a)
        self.assertEqual(qc.get_var("b"), b)

    def test_get_stretch_success(self):
        a = expr.Stretch.new("a")
        b = expr.Stretch.new("b")

        qc = QuantumCircuit(captures=[a])
        qc.add_stretch(b)
        self.assertEqual(qc.get_stretch("a"), a)
        self.assertEqual(qc.get_stretch("b"), b)

        qc = QuantumCircuit(captures=[a, b])
        self.assertEqual(qc.get_stretch("a"), a)
        self.assertEqual(qc.get_stretch("b"), b)

    def test_get_var_missing(self):
        qc = QuantumCircuit()
        with self.assertRaises(KeyError):
            qc.get_var("a")

        a = expr.Var.new("a", types.Bool())
        qc.add_input(a)
        with self.assertRaises(KeyError):
            qc.get_var("b")

    def test_get_stretch_missing(self):
        qc = QuantumCircuit()
        with self.assertRaises(KeyError):
            qc.get_stretch("a")

        a = expr.Stretch.new("a")
        qc.add_capture(a)
        with self.assertRaises(KeyError):
            qc.get_stretch("b")

    def test_get_var_default(self):
        qc = QuantumCircuit()
        self.assertEqual(qc.get_var("a", None), None)

        missing = "default"
        a = expr.Var.new("a", types.Bool())
        qc.add_input(a)
        self.assertEqual(qc.get_var("c", missing), missing)
        self.assertEqual(qc.get_var("c", a), a)

    def test_get_stretch_default(self):
        qc = QuantumCircuit()
        self.assertEqual(qc.get_stretch("a", None), None)

        missing = "default"
        a = expr.Stretch.new("a")
        qc.add_stretch(a)
        self.assertEqual(qc.get_stretch("c", missing), missing)
        self.assertEqual(qc.get_stretch("c", a), a)

    def test_has_var(self):
        a = expr.Var.new("a", types.Bool())
        self.assertFalse(QuantumCircuit().has_var("a"))
        self.assertTrue(QuantumCircuit(inputs=[a]).has_var("a"))
        self.assertTrue(QuantumCircuit(captures=[a]).has_var("a"))
        self.assertTrue(QuantumCircuit(declarations={a: expr.lift(True)}).has_var("a"))
        self.assertTrue(QuantumCircuit(inputs=[a]).has_var(a))
        self.assertTrue(QuantumCircuit(captures=[a]).has_var(a))
        self.assertTrue(QuantumCircuit(declarations={a: expr.lift(True)}).has_var(a))

        # When giving an `Var`, the match must be exact, not just the name.
        self.assertFalse(QuantumCircuit(inputs=[a]).has_var(expr.Var.new("a", types.Uint(8))))
        self.assertFalse(QuantumCircuit(inputs=[a]).has_var(expr.Var.new("a", types.Bool())))
        self.assertFalse(QuantumCircuit(inputs=[a]).has_var(expr.Var.new("a", types.Float())))

    def test_has_stretch(self):
        a = expr.Stretch.new("a")
        self.assertFalse(QuantumCircuit().has_stretch("a"))
        self.assertTrue(QuantumCircuit(captures=[a]).has_stretch("a"))
        self.assertTrue(QuantumCircuit(captures=[a]).has_stretch(a))

        # When giving an `Stretch`, the match must be exact, not just the name.
        self.assertFalse(QuantumCircuit(captures=[a]).has_stretch(expr.Stretch.new("a")))


class TestSubstituteVars(QiskitTestCase):
    """Tests for :meth:`~.QuantumCircuit.substitute_vars` and the per-operation
    ``substitute`` helpers it dispatches to."""

    def test_substitute_var_in_store(self):
        """Store lvalues and rvalues are rewritten."""
        qc = QuantumCircuit()
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        loop_var = expr.Var.new("i", types.Uint(8))
        qc.add_uninitialized_var(loop_var)
        qc.store(target, loop_var)

        result = qc.substitute_vars({loop_var: expr.lift(3, types.Uint(8))})

        store_rvalues = [
            inst.operation.rvalue for inst in result.data if inst.operation.name == "store"
        ]
        self.assertIn(expr.lift(3, types.Uint(8)), store_rvalues)
        self.assertNotIn(loop_var, list(result.iter_vars()))

    def test_substitute_inplace_mutates_and_returns_none(self):
        """``inplace=True`` rewrites the circuit in place and returns ``None``."""
        qc = QuantumCircuit()
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        loop_var = expr.Var.new("i", types.Uint(8))
        qc.add_uninitialized_var(loop_var)
        qc.store(target, loop_var)

        out = qc.substitute_vars({loop_var: expr.lift(3, types.Uint(8))}, inplace=True)

        self.assertIsNone(out)
        store_rvalues = [
            inst.operation.rvalue for inst in qc.data if inst.operation.name == "store"
        ]
        self.assertIn(expr.lift(3, types.Uint(8)), store_rvalues)
        self.assertNotIn(loop_var, list(qc.iter_vars()))

    def test_substitute_copy_leaves_input_untouched(self):
        """``inplace=False`` (default) leaves the input circuit unchanged."""
        qc = QuantumCircuit()
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        loop_var = expr.Var.new("i", types.Uint(8))
        qc.add_uninitialized_var(loop_var)
        qc.store(target, loop_var)

        result = qc.substitute_vars({loop_var: expr.lift(3, types.Uint(8))})

        self.assertIsNot(result, qc)
        self.assertIn(loop_var, list(qc.iter_vars()))
        self.assertNotIn(loop_var, list(result.iter_vars()))

    def test_substitute_var_in_if_else_condition(self):
        """Classical if-else conditions are rewritten."""
        loop_var = expr.Var.new("i", types.Uint(8))
        true_body = QuantumCircuit(1)
        true_body.x(0)
        false_body = QuantumCircuit(1)
        qc = QuantumCircuit(1)
        qc.append(
            IfElseOp(
                expr.equal(loop_var, expr.lift(1, types.Uint(8))),
                true_body,
                false_body,
            ),
            [0],
        )

        result = qc.substitute_vars({loop_var: expr.lift(1, types.Uint(8))}, strict=False)

        condition = result.data[0].operation.condition
        self.assertEqual(
            condition,
            expr.equal(expr.lift(1, types.Uint(8)), expr.lift(1, types.Uint(8))),
        )

    def test_substitute_var_in_for_loop_body(self):
        """The loop variable is bound by the loop, so substituting it is a no-op on the body.

        The loop variable is the body's ``input`` variable (it cannot be substituted away without
        leaving the body disagreeing with ``loop_parameter``); it indexes a register rather than
        closing over an outer ``expr.Var`` (a body with an input var cannot also have captures).
        """
        cr = ClassicalRegister(16, "cr")
        qc = QuantumCircuit(1)
        qc.add_register(cr)
        indexset = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(4, types.Uint(8)))
        with qc.for_loop(indexset) as loop_var:
            qc.store(expr.index(cr, loop_var), expr.lift(True))

        result = qc.substitute_vars({loop_var: expr.lift(2, types.Uint(8))}, strict=False)

        for_loop = next(inst.operation for inst in result.data if inst.operation.name == "for_loop")
        self.assertEqual(for_loop.params[0], indexset)
        self.assertEqual(for_loop.params[1], loop_var)
        body_indices = [
            inst.operation.lvalue.index
            for inst in for_loop.params[2].data
            if inst.operation.name == "store"
        ]
        self.assertEqual(body_indices, [loop_var])

    def test_empty_substitution_returns_independent_circuit(self):
        """An empty mapping returns a copy whose mutation does not affect the input."""
        qc = QuantumCircuit(1)
        qc.add_var("target", expr.lift(0, types.Uint(8)))
        qc.x(0)

        result = qc.substitute_vars({})

        self.assertIsNot(result, qc)
        original_len = len(qc.data)
        result.x(0)
        self.assertEqual(len(qc.data), original_len)
        self.assertEqual(len(result.data), original_len + 1)

    def test_declared_var_initialization_preserved(self):
        """Initial values of preserved declared vars survive substitution."""
        qc = QuantumCircuit()
        qc.add_var("kept", expr.lift(7, types.Uint(8)))
        loop_var = expr.Var.new("i", types.Uint(8))
        qc.add_uninitialized_var(loop_var)

        result = qc.substitute_vars({loop_var: expr.lift(3, types.Uint(8))})

        store_rvalues = [
            inst.operation.rvalue for inst in result.data if inst.operation.name == "store"
        ]
        self.assertIn(expr.lift(7, types.Uint(8)), store_rvalues)

    def test_substitute_var_in_switch_case_roundtrip(self):
        """Switch-case targets and bodies are rewritten and reconstructed correctly."""
        target_var = expr.Var.new("t", types.Uint(8))
        body_zero = QuantumCircuit(1)
        body_zero.x(0)
        body_default = QuantumCircuit(1)
        body_default.h(0)

        qc = QuantumCircuit(1)
        qc.append(
            SwitchCaseOp(target_var, [(0, body_zero), (CASE_DEFAULT, body_default)]),
            [0],
        )

        result = qc.substitute_vars({target_var: expr.lift(0, types.Uint(8))}, strict=False)

        switch = result.data[0].operation
        self.assertEqual(switch.target, expr.lift(0, types.Uint(8)))
        cases = list(switch.cases_specifier())
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0][0], (0,))
        self.assertEqual(cases[1][0], (CASE_DEFAULT,))

    def test_for_loop_with_python_range_indexset_recurses_into_body(self):
        """A Python ``range`` indexset passes through untouched while the body is rewritten."""
        captured = expr.Var.new("flag", types.Uint(8))
        loop_param = Parameter("i")

        body = QuantumCircuit(1)
        body.add_capture(captured)
        target = body.add_var("target", expr.lift(0, types.Uint(8)))
        body.store(target, captured)

        qc = QuantumCircuit(1)
        qc.add_uninitialized_var(captured)
        qc.append(ForLoopOp(range(4), loop_param, body), [0])

        result = qc.substitute_vars({captured: expr.lift(7, types.Uint(8))})

        for_loop = next(inst.operation for inst in result.data if inst.operation.name == "for_loop")
        self.assertEqual(for_loop.params[0], range(4))
        self.assertIsInstance(for_loop.params[1], Parameter)
        self.assertEqual(for_loop.params[1].name, loop_param.name)
        body_stores = [
            inst.operation.rvalue
            for inst in for_loop.params[2].data
            if inst.operation.name == "store"
        ]
        self.assertIn(expr.lift(7, types.Uint(8)), body_stores)
        self.assertNotIn(captured, list(result.iter_vars()))

    def test_substitute_var_inside_box_body(self):
        """Vars inside a box body are rewritten (the box itself has no condition)."""
        flag = expr.Var.new("flag", types.Uint(8))

        box_body = QuantumCircuit(1)
        box_body.add_capture(flag)
        sink = box_body.add_var("sink", expr.lift(0, types.Uint(8)))
        box_body.store(sink, flag)

        qc = QuantumCircuit(1)
        qc.add_uninitialized_var(flag)
        qc.append(BoxOp(box_body), [0])

        result = qc.substitute_vars({flag: expr.lift(4, types.Uint(8))})

        box = result.data[-1].operation
        self.assertIsInstance(box, BoxOp)
        body_stores = [
            inst.operation.rvalue for inst in box.blocks[0].data if inst.operation.name == "store"
        ]
        self.assertIn(expr.lift(4, types.Uint(8)), body_stores)
        self.assertNotIn(flag, list(result.iter_vars()))

    def test_substitute_var_through_three_level_nesting(self):
        """Three-level nesting: for > if > store — substitution reaches the deepest body."""
        deep_var = expr.Var.new("deep", types.Uint(8))

        innermost = QuantumCircuit(1)
        innermost.add_capture(deep_var)
        sink = innermost.add_var("sink", expr.lift(0, types.Uint(8)))
        innermost.store(sink, deep_var)

        if_block = QuantumCircuit(1)
        if_block.add_capture(deep_var)
        if_block.append(
            IfElseOp(expr.equal(deep_var, expr.lift(2, types.Uint(8))), innermost, None),
            [0],
        )

        qc = QuantumCircuit(1)
        qc.add_uninitialized_var(deep_var)
        qc.append(ForLoopOp(range(2), Parameter("i"), if_block), [0])

        result = qc.substitute_vars({deep_var: expr.lift(2, types.Uint(8))})

        for_body = result.data[-1].operation.params[2]
        inner_if = for_body.data[0].operation
        self.assertEqual(
            inner_if.condition,
            expr.equal(expr.lift(2, types.Uint(8)), expr.lift(2, types.Uint(8))),
        )
        innermost_body = inner_if.blocks[0]
        innermost_stores = [
            inst.operation.rvalue for inst in innermost_body.data if inst.operation.name == "store"
        ]
        self.assertIn(expr.lift(2, types.Uint(8)), innermost_stores)
        self.assertNotIn(deep_var, list(result.iter_vars()))

    def test_op_substitute_store(self):
        """:meth:`Store.substitute` rewrites both lvalue and rvalue."""
        a = expr.Var.new("a", types.Uint(8))
        b = expr.Var.new("b", types.Uint(8))
        store = Store(a, b)
        new = store.substitute({a: a, b: expr.lift(5, types.Uint(8))})
        self.assertEqual(new.rvalue, expr.lift(5, types.Uint(8)))

    def test_op_substitute_while_loop_rewrites_condition_and_body(self):
        """:meth:`WhileLoopOp.substitute` rewrites an expr condition and recurses into the body."""
        cond_var = expr.Var.new("c", types.Uint(8))
        body_var = expr.Var.new("v", types.Uint(8))

        body = QuantumCircuit(1)
        body.add_capture(cond_var)
        body.add_uninitialized_var(body_var)
        sink = body.add_var("sink", expr.lift(0, types.Uint(8)))
        body.store(sink, body_var)

        op = WhileLoopOp(
            expr.equal(cond_var, expr.lift(0, types.Uint(8))),
            body,
        )

        replacement = expr.lift(3, types.Uint(8))
        new = op.substitute({body_var: replacement})

        # Condition: cond_var is not in the substitution map, so it is unchanged.
        self.assertEqual(
            new.condition,
            expr.equal(cond_var, expr.lift(0, types.Uint(8))),
        )
        # Body: body_var is replaced by the literal 3.
        body_stores = [
            inst.operation.rvalue for inst in new.blocks[0].data if inst.operation.name == "store"
        ]
        self.assertIn(replacement, body_stores)
        # The original body is not mutated.
        original_body_stores = [
            inst.operation.rvalue for inst in op.blocks[0].data if inst.operation.name == "store"
        ]
        self.assertIn(body_var, original_body_stores)

    def test_op_substitute_while_loop_rewrites_condition_var(self):
        """:meth:`WhileLoopOp.substitute` rewrites a Var that appears in the condition."""
        cond_var = expr.Var.new("c", types.Uint(8))
        body = QuantumCircuit(1)
        body.add_capture(cond_var)

        op = WhileLoopOp(expr.equal(cond_var, expr.lift(0, types.Uint(8))), body)

        new_cond_var = expr.Var.new("d", types.Uint(8))
        new = op.substitute({cond_var: new_cond_var})

        self.assertEqual(
            new.condition,
            expr.equal(new_cond_var, expr.lift(0, types.Uint(8))),
        )

    def test_op_substitute_while_loop_legacy_tuple_condition_unchanged(self):
        """:meth:`WhileLoopOp.substitute` leaves a legacy two-tuple condition untouched.

        A legacy condition is a ``(Clbit, bool)`` or ``(ClassicalRegister, int)`` two-tuple
        rather than an :class:`~.expr.Expr`.  It carries no :class:`~.expr.Var` nodes so the
        substitution map has nothing to apply to it.
        """
        cr = ClassicalRegister(1, "cr")
        body = QuantumCircuit(1)
        body.add_register(cr)
        op = WhileLoopOp((cr[0], True), body)

        # The tuple condition has no expr.Var nodes; substitution must not touch it.
        some_var = expr.Var.new("x", types.Bool())
        new = op.substitute({some_var: expr.lift(True)})

        self.assertEqual(new.condition, (cr[0], True))

    def test_op_substitute_for_loop_keeps_loop_parameter(self):
        """:meth:`ForLoopOp.substitute` rewrites the Range indexset but not the loop var."""
        stop = expr.Var.new("n", types.Uint(8))
        loop_var = expr.Var.new("i", types.Uint(8))
        indexset = expr.Range(expr.lift(0, types.Uint(8)), stop)
        body = QuantumCircuit(1, inputs=[loop_var])
        op = ForLoopOp(indexset, loop_var, body)

        new = op.substitute({stop: expr.lift(8, types.Uint(8))})

        self.assertEqual(
            new.params[0], expr.Range(expr.lift(0, types.Uint(8)), expr.lift(8, types.Uint(8)))
        )
        self.assertEqual(new.params[1], loop_var)

    def test_substitute_vars_strict_raises_for_unknown_var(self):
        """`substitute_vars(strict=True)` raises when a key is not declared in the circuit."""
        qc = QuantumCircuit()
        a = qc.add_var("a", expr.lift(0, types.Uint(8)))
        ghost = expr.Var.new("ghost", types.Uint(8))  # not in qc

        with self.assertRaisesRegex(CircuitError, "Cannot substitute variables.*ghost"):
            qc.substitute_vars({ghost: a}, strict=True)

    def test_substitute_vars_strict_false_ignores_unknown_var(self):
        """`substitute_vars(strict=False)` silently skips keys absent from the circuit."""
        qc = QuantumCircuit()
        a = qc.add_var("a", expr.lift(0, types.Uint(8)))
        ghost = expr.Var.new("ghost", types.Uint(8))  # not in qc

        # No error; the ghost key is simply ignored and 'a' is preserved as-is.
        out = qc.substitute_vars({ghost: a}, strict=False)
        self.assertIsInstance(out, QuantumCircuit)
        self.assertTrue(out.has_var(a))

    def test_substitute_vars_strict_default_is_true(self):
        """`substitute_vars` raises by default (strict=True) for an unknown var."""
        qc = QuantumCircuit()
        qc.add_var("a", expr.lift(0, types.Uint(8)))
        ghost = expr.Var.new("ghost", types.Uint(8))

        with self.assertRaises(CircuitError):
            qc.substitute_vars({ghost: expr.lift(0, types.Uint(8))})

    # --- Case 2: self-referential substitution (key appears in its own replacement) ---

    def test_self_referential_substitution_keeps_key_in_scope(self):
        """A key that appears in its own replacement is kept in scope, not removed."""
        qc = QuantumCircuit()
        # Use add_input so x has no implicit initializer store (which would create an x lvalue
        # that becomes non-lvalue x+1 after substitution and trigger a Store error).
        x = qc.add_input("x", types.Uint(8))
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        qc.store(target, x)

        # x → x + 1: every READ of x becomes x+1, but x must stay declared as an input.
        out = qc.substitute_vars({x: expr.add(x, expr.lift(1, types.Uint(8)))})

        self.assertTrue(
            out.has_var(x), "x must remain declared since it appears in its replacement"
        )
        self.assertIn(x, list(out.iter_input_vars()))
        store_rvalues = [
            inst.operation.rvalue for inst in out.data if inst.operation.name == "store"
        ]
        self.assertIn(expr.add(x, expr.lift(1, types.Uint(8))), store_rvalues)

    def test_self_referential_substitution_cross_key_keeps_referenced_key(self):
        """A key referenced inside *another* key's replacement is also kept in scope."""
        qc = QuantumCircuit()
        # Use input vars to avoid implicit initializer stores that would turn x into a lvalue.
        x = qc.add_input("x", types.Uint(8))
        y = qc.add_input("y", types.Uint(8))
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        qc.store(target, x)
        qc.store(target, y)

        # y → x + 1: y is fully replaced, x is referenced inside y's replacement so x stays.
        out = qc.substitute_vars({y: expr.add(x, expr.lift(1, types.Uint(8)))})

        self.assertTrue(
            out.has_var(x), "x must remain declared since it appears in y's replacement"
        )
        self.assertFalse(out.has_var(y), "y must be removed as it was fully substituted")

    # --- Case 3: two keys share the same new replacement Var ---

    def test_two_keys_same_replacement_var_compatible_scopes(self):
        """Two input vars that both map to the same new Var → new Var added as input once."""
        qc = QuantumCircuit()
        a = qc.add_input("a", types.Uint(8))
        b = qc.add_input("b", types.Uint(8))
        z = expr.Var.new("z", types.Uint(8))  # not in qc

        out = qc.substitute_vars({a: z, b: z}, strict=False)

        self.assertTrue(out.has_var(z))
        self.assertIn(z, list(out.iter_input_vars()))
        self.assertFalse(out.has_var(a))
        self.assertFalse(out.has_var(b))

    def test_two_keys_same_replacement_var_conflicting_scopes_raises(self):
        """An input var and a declared var both mapping to the same new Var raises CircuitError."""
        qc = QuantumCircuit()
        a = qc.add_input("a", types.Uint(8))
        b = qc.add_var("b", expr.lift(0, types.Uint(8)))  # declared scope
        z = expr.Var.new("z", types.Uint(8))  # not in qc

        with self.assertRaisesRegex(CircuitError, "Cannot determine scope.*incompatible scopes"):
            qc.substitute_vars({a: z, b: z}, strict=False)

    # ---------------------------------------------------------------------------
    # keep_substituted_vars flag
    # ---------------------------------------------------------------------------

    def test_keep_substituted_vars_normal_rename_ghost_declaration(self):
        """keep_substituted_vars=True keeps the substituted key as a ghost declaration."""
        qc = QuantumCircuit()
        x = qc.add_input("x", types.Uint(8))
        y = expr.Var.new("y", types.Uint(8))

        out = qc.substitute_vars({x: y}, keep_substituted_vars=True)

        # x is now a ghost: declared but not referenced anywhere
        self.assertTrue(out.has_var(x), "x must be retained as a ghost declaration")
        self.assertIn(x, list(out.iter_input_vars()), "x must keep its input scope")
        # y was added with x's inferred scope
        self.assertTrue(out.has_var(y))
        self.assertIn(y, list(out.iter_input_vars()))

    def test_keep_substituted_vars_false_removes_key(self):
        """keep_substituted_vars=False (default) removes the substituted key."""
        qc = QuantumCircuit()
        x = qc.add_input("x", types.Uint(8))
        y = expr.Var.new("y", types.Uint(8))

        out = qc.substitute_vars({x: y})

        self.assertFalse(out.has_var(x), "x must be removed by default")
        self.assertTrue(out.has_var(y))

    def test_keep_substituted_vars_subsequent_strict_pass_finds_ghost(self):
        """A ghost declaration is found by strict=True on a second substitution, but the
        structural rewrite has no effect because the ghost is not referenced anywhere."""
        qc = QuantumCircuit()
        x = qc.add_input("x", types.Uint(8))
        y = expr.Var.new("y", types.Uint(8))
        z = expr.Var.new("z", types.Uint(8))

        # First pass: substitute x → y, keep x as ghost.
        mid = qc.substitute_vars({x: y}, keep_substituted_vars=True, strict=False)
        self.assertTrue(mid.has_var(x))

        # Second pass: strict=True finds x (ghost) and passes validation; structural
        # effect is zero because x does not appear in any expression.
        out = mid.substitute_vars({x: z}, strict=True, keep_substituted_vars=False)

        # x ghost is now dropped (keep=False), z is added with x's scope
        self.assertFalse(out.has_var(x))
        self.assertTrue(out.has_var(z))
        self.assertIn(z, list(out.iter_input_vars()))

    def test_keep_substituted_vars_self_referential_always_kept(self):
        """Self-referential keys are always retained regardless of keep_substituted_vars."""
        qc = QuantumCircuit()
        x = qc.add_input("x", types.Uint(8))
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        qc.store(target, x)

        # keep_substituted_vars=False but x must stay because it appears in its own replacement.
        out = qc.substitute_vars(
            {x: expr.add(x, expr.lift(1, types.Uint(8)))},
            keep_substituted_vars=False,
        )
        self.assertTrue(out.has_var(x), "x must be kept — it appears in its own replacement")

    def test_keep_substituted_vars_cross_referential_keeps_key_y_as_ghost(self):
        """With keep=True, the fully-substituted cross-referential key y becomes a ghost."""
        qc = QuantumCircuit()
        x = qc.add_input("x", types.Uint(8))
        y = qc.add_input("y", types.Uint(8))
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        qc.store(target, y)

        # y → x + 1: y is substituted away, x is kept by keys_to_keep.
        # With keep=True, y is also retained as a ghost.
        out = qc.substitute_vars(
            {y: expr.add(x, expr.lift(1, types.Uint(8)))},
            keep_substituted_vars=True,
        )
        self.assertTrue(out.has_var(y), "y must be retained as ghost with keep=True")
        self.assertIn(y, list(out.iter_input_vars()))
        self.assertTrue(out.has_var(x), "x retained by keys_to_keep regardless of flag")

    def test_keep_substituted_vars_strict_false_key_not_in_circuit_no_op(self):
        """keep_substituted_vars=True has no effect for keys not declared in the circuit
        (strict=False path) — there is nothing to retain at the top level."""
        qc = QuantumCircuit()
        declared = qc.add_var("declared", expr.lift(0, types.Uint(8)))
        ghost_key = expr.Var.new("ghost_key", types.Uint(8))  # not in qc

        out = qc.substitute_vars(
            {ghost_key: expr.lift(0, types.Uint(8))},
            strict=False,
            keep_substituted_vars=True,
        )

        # ghost_key was never declared here, so keep has nothing to act on.
        self.assertFalse(out.has_var(ghost_key))
        # The var that was declared and not substituted survives unchanged.
        self.assertTrue(out.has_var(declared))

    def test_keep_substituted_vars_two_keys_same_new_var_both_ghosts(self):
        """With keep=True and two keys mapping to the same new var (compatible scopes),
        both original keys are retained as ghosts and the new var is added once."""
        qc = QuantumCircuit()
        a = qc.add_input("a", types.Uint(8))
        b = qc.add_input("b", types.Uint(8))
        z = expr.Var.new("z", types.Uint(8))

        out = qc.substitute_vars({a: z, b: z}, strict=False, keep_substituted_vars=True)

        self.assertIn(z, list(out.iter_input_vars()), "z must be added as input")
        self.assertIn(a, list(out.iter_input_vars()), "a retained as ghost")
        self.assertIn(b, list(out.iter_input_vars()), "b retained as ghost")
