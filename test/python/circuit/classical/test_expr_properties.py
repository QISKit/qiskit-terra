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


import copy
import pickle
import uuid

import ddt

from qiskit.circuit import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr, types
from test import QiskitTestCase


@ddt.ddt
class TestExprProperties(QiskitTestCase):
    def test_bool_type_is_singleton(self):
        """The `Bool` type is meant (and used) as a Python singleton object for efficiency.  It must
        always be referentially equal to all other references to it."""
        self.assertIs(types.Bool(), types.Bool())
        self.assertIs(types.Bool(), copy.copy(types.Bool()))
        self.assertIs(types.Bool(), copy.deepcopy(types.Bool()))
        self.assertIs(types.Bool(), pickle.loads(pickle.dumps(types.Bool())))

    @ddt.data(types.Bool(), types.Uint(8))
    def test_types_can_be_cloned(self, obj):
        """Test that various ways of cloning a `Type` object are valid and produce equal output."""
        self.assertEqual(obj, copy.copy(obj))
        self.assertEqual(obj, copy.deepcopy(obj))
        self.assertEqual(obj, pickle.loads(pickle.dumps(obj)))

    @ddt.data(
        expr.Var(ClassicalRegister(3, "c"), types.Uint(3)),
        expr.Value(3, types.Uint(2)),
        expr.Cast(expr.Value(1, types.Uint(8)), types.Bool()),
        expr.Unary(expr.Unary.Op.LOGIC_NOT, expr.Value(False, types.Bool()), types.Bool()),
        expr.Binary(
            expr.Binary.Op.LOGIC_OR,
            expr.Value(False, types.Bool()),
            expr.Value(True, types.Bool()),
            types.Bool(),
        ),
        expr.Index(
            expr.Var.new("a", types.Uint(3)),
            expr.Binary(
                expr.Binary.Op.SHIFT_LEFT,
                expr.Binary(
                    expr.Binary.Op.SHIFT_RIGHT,
                    expr.Var.new("b", types.Uint(3)),
                    expr.Value(1, types.Uint(1)),
                    types.Uint(3),
                ),
                expr.Value(1, types.Uint(1)),
                types.Uint(3),
            ),
            types.Bool(),
        ),
    )
    def test_expr_can_be_cloned(self, obj):
        """Test that various ways of cloning an `Expr` object are valid and produce equal output."""
        self.assertEqual(obj, copy.copy(obj))
        self.assertEqual(obj, copy.deepcopy(obj))
        self.assertEqual(obj, pickle.loads(pickle.dumps(obj)))

    def test_var_equality(self):
        """Test that various types of :class:`.expr.Var` equality work as expected both in equal and
        unequal cases."""
        var_a_bool = expr.Var.new("a", types.Bool())
        self.assertEqual(var_a_bool, var_a_bool)

        # Allocating a new variable should not compare equal, despite the name match.  A semantic
        # equality checker can choose to key these variables on only their names and types, if it
        # knows that that check is valid within the semantic context.
        self.assertNotEqual(var_a_bool, expr.Var.new("a", types.Bool()))

        # Manually constructing the same object with the same UUID should cause it compare equal,
        # though, for serialization ease.
        self.assertEqual(var_a_bool, expr.Var(var_a_bool.var, types.Bool(), name="a"))

        # This is a badly constructed variable because it's using a different type to refer to the
        # same storage location (the UUID) as another variable.  It is an IR error to generate this
        # sort of thing, but we can't fully be responsible for that and a pass would need to go out
        # of its way to do this incorrectly, but we can still ensure that the direct equality check
        # would spot the error.
        self.assertNotEqual(
            var_a_bool, expr.Var(var_a_bool.var, types.Uint(8), name=var_a_bool.name)
        )

        # This is also badly constructed because it uses a different name to refer to the "same"
        # storage location.
        self.assertNotEqual(var_a_bool, expr.Var(var_a_bool.var, types.Bool(), name="b"))

        # Obviously, two variables of different types and names should compare unequal.
        self.assertNotEqual(expr.Var.new("a", types.Bool()), expr.Var.new("b", types.Uint(8)))
        # As should two variables of the same name but different storage locations and types.
        self.assertNotEqual(expr.Var.new("a", types.Bool()), expr.Var.new("a", types.Uint(8)))

    def test_var_uuid_clone(self):
        """Test that :class:`.expr.Var` instances that have an associated UUID and name roundtrip
        through pickle and copy operations to produce values that compare equal."""
        var_a_u8 = expr.Var.new("a", types.Uint(8))

        self.assertEqual(var_a_u8, pickle.loads(pickle.dumps(var_a_u8)))
        self.assertEqual(var_a_u8, copy.copy(var_a_u8))
        self.assertEqual(var_a_u8, copy.deepcopy(var_a_u8))

    def test_var_standalone(self):
        """Test that the ``Var.standalone`` property is set correctly."""
        self.assertTrue(expr.Var.new("a", types.Bool()).standalone)
        self.assertTrue(expr.Var.new("a", types.Uint(8)).standalone)
        self.assertFalse(expr.Var(Clbit(), types.Bool()).standalone)
        self.assertFalse(expr.Var(ClassicalRegister(8, "cr"), types.Uint(8)).standalone)

    def test_big_int_values(self):
        """Test that ``Value`` of type ``Uint`` can handle arbitrarily large integer"""
        for raw_value in [10, 10**20, 10**50]:
            val = (expr.Value(raw_value, types.Uint(1)),)
            self.assertEqual(raw_value, val[0].value)

    def test_var_hashable(self):
        clbits = [Clbit(), Clbit()]
        cregs = [ClassicalRegister(2, "cr1"), ClassicalRegister(2, "cr2")]

        vars_ = [
            expr.Var.new("a", types.Bool()),
            expr.Var.new("b", types.Uint(16)),
            expr.Var(clbits[0], types.Bool()),
            expr.Var(clbits[1], types.Bool()),
            expr.Var(cregs[0], types.Uint(2)),
            expr.Var(cregs[1], types.Uint(2)),
        ]
        duplicates = [
            expr.Var(uuid.UUID(bytes=vars_[0].var.bytes), types.Bool(), name=vars_[0].name),
            expr.Var(uuid.UUID(bytes=vars_[1].var.bytes), types.Uint(16), name=vars_[1].name),
            expr.Var(clbits[0], types.Bool()),
            expr.Var(clbits[1], types.Bool()),
            expr.Var(cregs[0], types.Uint(2)),
            expr.Var(cregs[1], types.Uint(2)),
        ]

        # Smoke test.
        self.assertEqual(vars_, duplicates)
        # Actual test of hashability properties.
        self.assertEqual(set(vars_ + duplicates), set(vars_))
