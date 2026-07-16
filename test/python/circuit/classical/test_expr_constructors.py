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


import ddt

from qiskit.circuit import Clbit, ClassicalRegister, Duration
from qiskit.circuit.classical import expr, types
from test import QiskitTestCase


@ddt.ddt
class TestExprConstructors(QiskitTestCase):
    def test_lift_legacy_condition(self):
        cr = ClassicalRegister(3, "c")
        clbit = Clbit()
        cond = (cr, 7)
        self.assertEqual(
            expr.lift_legacy_condition(cond),
            expr.Binary(
                expr.Binary.Op.EQUAL,
                expr.Var(cr, types.Uint(cr.size)),
                expr.Value(7, types.Uint(cr.size)),
                types.Bool(),
            ),
        )
        cond = (cr, 255)
        self.assertEqual(
            expr.lift_legacy_condition(cond),
            expr.Binary(
                expr.Binary.Op.EQUAL,
                expr.Cast(expr.Var(cr, types.Uint(cr.size)), types.Uint(8), implicit=True),
                expr.Value(255, types.Uint(8)),
                types.Bool(),
            ),
        )
        cond = (clbit, False)
        self.assertEqual(
            expr.lift_legacy_condition(cond),
            expr.Unary(
                expr.Unary.Op.LOGIC_NOT,
                expr.Var(clbit, types.Bool()),
                types.Bool(),
            ),
        )
        cond = (clbit, True)
        self.assertEqual(
            expr.lift_legacy_condition(cond),
            expr.Var(clbit, types.Bool()),
        )

    def test_value_lifts_qiskit_scalars(self):
        cr = ClassicalRegister(3, "c")
        self.assertEqual(expr.lift(cr), expr.Var(cr, types.Uint(cr.size)))

        big = ClassicalRegister((1 << 32) - 1, "BIG")  # Limit is 32-bit.
        self.assertEqual(expr.lift(big), expr.Var(big, types.Uint(big.size)))

        clbit = Clbit()
        self.assertEqual(expr.lift(clbit), expr.Var(clbit, types.Bool()))

        duration = Duration.dt(1000)
        self.assertEqual(expr.lift(duration), expr.Value(duration, types.Duration()))

    def test_value_lifts_python_builtins(self):
        self.assertEqual(expr.lift(True), expr.Value(True, types.Bool()))
        self.assertEqual(expr.lift(False), expr.Value(False, types.Bool()))
        self.assertEqual(expr.lift(7), expr.Value(7, types.Uint(3)))
        self.assertEqual(expr.lift(7.0), expr.Value(7.0, types.Float()))

    def test_value_ensures_nonzero_width(self):
        self.assertEqual(expr.lift(0), expr.Value(0, types.Uint(1)))

    def test_value_type_representation(self):
        self.assertEqual(expr.lift(5), expr.Value(5, types.Uint((5).bit_length())))
        self.assertEqual(expr.lift(5, types.Uint(8)), expr.Value(5, types.Uint(8)))

        cr = ClassicalRegister(3, "c")
        self.assertEqual(expr.lift(cr, types.Uint(8)), expr.Var(cr, types.Uint(8)))

    def test_value_does_not_allow_downcast(self):
        with self.assertRaisesRegex(TypeError, "the explicit type .* is not suitable"):
            expr.lift(0xFF, types.Uint(2))
        with self.assertRaisesRegex(TypeError, "the explicit type .* is not suitable"):
            expr.lift(1.1, types.Uint(2))

    def test_value_rejects_bad_values(self):
        with self.assertRaisesRegex(TypeError, "failed to infer a type"):
            expr.lift("1")
        with self.assertRaisesRegex(ValueError, "cannot represent a negative value"):
            expr.lift(-1)

    def test_cast_adds_explicit_nodes(self):
        """A specific request to add a cast in means that we should respect that in the type tree,
        even if the cast is a no-op."""
        base = expr.Value(5, types.Uint(8))
        self.assertEqual(
            expr.cast(base, types.Uint(8)), expr.Cast(base, types.Uint(8), implicit=False)
        )

    def test_cast_allows_lossy_downcasting(self):
        """An explicit 'cast' call should allow lossy casts to be performed."""
        base = expr.Value(5, types.Uint(16))
        self.assertEqual(
            expr.cast(base, types.Uint(8)), expr.Cast(base, types.Uint(8), implicit=False)
        )
        self.assertEqual(
            expr.cast(base, types.Bool()), expr.Cast(base, types.Bool(), implicit=False)
        )
        self.assertEqual(
            expr.cast(base, types.Float()), expr.Cast(base, types.Float(), implicit=False)
        )

    @ddt.data(
        (expr.bit_not, ClassicalRegister(3)),
        (expr.logic_not, ClassicalRegister(3)),
        (expr.logic_not, False),
        (expr.logic_not, Clbit()),
        (expr.negate, 7.0),
        (expr.negate, Duration.dt(1000)),
    )
    @ddt.unpack
    def test_unary_functions_lift_scalars(self, function, scalar):
        self.assertEqual(function(scalar), function(expr.lift(scalar)))

    def test_bit_not_explicit(self):
        cr = ClassicalRegister(3)
        self.assertEqual(
            expr.bit_not(cr),
            expr.Unary(
                expr.Unary.Op.BIT_NOT, expr.Var(cr, types.Uint(cr.size)), types.Uint(cr.size)
            ),
        )
        clbit = Clbit()
        self.assertEqual(
            expr.bit_not(clbit),
            expr.Unary(expr.Unary.Op.BIT_NOT, expr.Var(clbit, types.Bool()), types.Bool()),
        )

    @ddt.data(expr.bit_not)
    def test_unary_bitwise_forbidden(self, function):
        with self.assertRaisesRegex(TypeError, "cannot apply"):
            function(7.0)
        with self.assertRaisesRegex(TypeError, "cannot apply"):
            function(Duration.dt(1000))

    def test_logic_not_explicit(self):
        cr = ClassicalRegister(3)
        self.assertEqual(
            expr.logic_not(cr),
            expr.Unary(
                expr.Unary.Op.LOGIC_NOT,
                expr.Cast(expr.Var(cr, types.Uint(cr.size)), types.Bool(), implicit=True),
                types.Bool(),
            ),
        )
        clbit = Clbit()
        self.assertEqual(
            expr.logic_not(clbit),
            expr.Unary(expr.Unary.Op.LOGIC_NOT, expr.Var(clbit, types.Bool()), types.Bool()),
        )

    @ddt.data(expr.logic_not)
    def test_unary_logical_forbidden(self, function):
        with self.assertRaisesRegex(TypeError, "cannot apply"):
            function(7.0)
        with self.assertRaisesRegex(TypeError, "cannot apply"):
            function(Duration.dt(1000))

    @ddt.data(
        (expr.bit_and, ClassicalRegister(3), ClassicalRegister(3)),
        (expr.bit_or, ClassicalRegister(3), ClassicalRegister(3)),
        (expr.bit_xor, ClassicalRegister(3), ClassicalRegister(3)),
        (expr.logic_and, Clbit(), True),
        (expr.logic_or, False, ClassicalRegister(3)),
        (expr.equal, ClassicalRegister(8), 255),
        (expr.not_equal, ClassicalRegister(8), 255),
        (expr.less, ClassicalRegister(3), 6),
        (expr.less_equal, ClassicalRegister(3), 5),
        (expr.greater, 4, ClassicalRegister(3)),
        (expr.greater_equal, ClassicalRegister(3), 5),
        (expr.add, ClassicalRegister(3), 6),
        (expr.sub, ClassicalRegister(3), 5),
        (expr.mul, 4, ClassicalRegister(3)),
        (expr.div, ClassicalRegister(3), 5),
        (expr.equal, 8.0, 255.0),
        (expr.not_equal, 8.0, 255.0),
        (expr.less, 3.0, 6.0),
        (expr.less_equal, 3.0, 5.0),
        (expr.greater, 4.0, 3.0),
        (expr.greater_equal, 3.0, 5.0),
        (expr.add, 3.0, 6.0),
        (expr.sub, 3.0, 5.0),
        (expr.mul, 4.0, 3.0),
        (expr.div, 3.0, 5.0),
        (expr.equal, Duration.dt(1000), Duration.dt(1000)),
        (expr.not_equal, Duration.dt(1000), Duration.dt(1000)),
        (expr.less, Duration.dt(1000), Duration.dt(1000)),
        (expr.less_equal, Duration.dt(1000), Duration.dt(1000)),
        (expr.greater, Duration.dt(1000), Duration.dt(1000)),
        (expr.greater_equal, Duration.dt(1000), Duration.dt(1000)),
        (expr.add, Duration.dt(1000), Duration.dt(1000)),
        (expr.sub, Duration.dt(1000), Duration.dt(1000)),
        (expr.div, Duration.dt(1000), Duration.dt(1000)),
    )
    @ddt.unpack
    def test_binary_functions_lift_scalars(self, function, left, right):
        self.assertEqual(function(left, right), function(expr.lift(left), right))
        self.assertEqual(function(left, right), function(left, expr.lift(right)))
        self.assertEqual(function(left, right), function(expr.lift(left), expr.lift(right)))

    @ddt.data(
        (expr.bit_and, expr.Binary.Op.BIT_AND),
        (expr.bit_or, expr.Binary.Op.BIT_OR),
        (expr.bit_xor, expr.Binary.Op.BIT_XOR),
    )
    @ddt.unpack
    def test_binary_bitwise_explicit(self, function, opcode):
        cr = ClassicalRegister(8, "c")
        self.assertEqual(
            function(cr, 255),
            expr.Binary(
                opcode, expr.Var(cr, types.Uint(8)), expr.Value(255, types.Uint(8)), types.Uint(8)
            ),
        )
        self.assertFalse(function(cr, 255).const)

        self.assertEqual(
            function(255, cr),
            expr.Binary(
                opcode, expr.Value(255, types.Uint(8)), expr.Var(cr, types.Uint(8)), types.Uint(8)
            ),
        )
        self.assertFalse(function(255, cr).const)

        clbit = Clbit()
        self.assertEqual(
            function(True, clbit),
            expr.Binary(
                opcode,
                expr.Value(True, types.Bool()),
                expr.Var(clbit, types.Bool()),
                types.Bool(),
            ),
        )
        self.assertFalse(function(True, clbit).const)

        self.assertEqual(
            function(clbit, False),
            expr.Binary(
                opcode,
                expr.Var(clbit, types.Bool()),
                expr.Value(False, types.Bool()),
                types.Bool(),
            ),
        )
        self.assertFalse(function(clbit, True).const)

        self.assertEqual(
            function(255, 255),
            expr.Binary(
                opcode,
                expr.Value(255, types.Uint(8)),
                expr.Value(255, types.Uint(8)),
                types.Uint(8),
            ),
        )
        self.assertTrue(function(255, 255).const)

    @ddt.data(
        (expr.bit_and, expr.Binary.Op.BIT_AND),
        (expr.bit_or, expr.Binary.Op.BIT_OR),
        (expr.bit_xor, expr.Binary.Op.BIT_XOR),
    )
    @ddt.unpack
    def test_binary_bitwise_uint_inference(self, function, opcode):
        """The binary bitwise functions have specialized inference for the widths of integer
        literals, since the bitwise functions require the operands to already be of exactly the same
        width without promotion."""
        cr = ClassicalRegister(8, "c")
        self.assertEqual(
            function(cr, 5),
            expr.Binary(
                opcode,
                expr.Var(cr, types.Uint(8)),
                expr.Value(5, types.Uint(8)),  # Note the inference should be Uint(8) not Uint(3).
                types.Uint(8),
            ),
        )
        self.assertEqual(
            function(5, cr),
            expr.Binary(
                opcode,
                expr.Value(5, types.Uint(8)),
                expr.Var(cr, types.Uint(8)),
                types.Uint(8),
            ),
        )

        # Inference between two integer literals is "best effort".  This behavior isn't super
        # important to maintain if we want to change the expression system.
        self.assertEqual(
            function(5, 255),
            expr.Binary(
                opcode,
                expr.Value(5, types.Uint(8)),
                expr.Value(255, types.Uint(8)),
                types.Uint(8),
            ),
        )

    @ddt.data(expr.bit_and, expr.bit_or, expr.bit_xor)
    def test_binary_bitwise_forbidden(self, function):
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(ClassicalRegister(3, "c"), Clbit())
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(3.0, 3.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(3, 3.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(3.0, 3)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Duration.dt(1000), Duration.dt(1000))
        # Unlike most other functions, the bitwise functions should error if the two bit-like types
        # aren't of the same width, except for the special inference for integer literals.
        with self.assertRaisesRegex(TypeError, "binary bitwise operations .* same width"):
            function(ClassicalRegister(3, "a"), ClassicalRegister(5, "b"))

    @ddt.data(
        (expr.logic_and, expr.Binary.Op.LOGIC_AND),
        (expr.logic_or, expr.Binary.Op.LOGIC_OR),
    )
    @ddt.unpack
    def test_binary_logical_explicit(self, function, opcode):
        cr = ClassicalRegister(8, "c")
        clbit = Clbit()

        self.assertEqual(
            function(cr, clbit),
            expr.Binary(
                opcode,
                expr.Cast(expr.Var(cr, types.Uint(cr.size)), types.Bool(), implicit=True),
                expr.Var(clbit, types.Bool()),
                types.Bool(),
            ),
        )
        self.assertFalse(function(cr, clbit).const)

        self.assertEqual(
            function(cr, 3),
            expr.Binary(
                opcode,
                expr.Cast(expr.Var(cr, types.Uint(cr.size)), types.Bool(), implicit=True),
                expr.Cast(expr.Value(3, types.Uint(2)), types.Bool(), implicit=True),
                types.Bool(),
            ),
        )
        self.assertFalse(function(cr, 3).const)

        self.assertEqual(
            function(False, clbit),
            expr.Binary(
                opcode,
                expr.Value(False, types.Bool()),
                expr.Var(clbit, types.Bool()),
                types.Bool(),
            ),
        )
        self.assertFalse(function(False, clbit).const)

    @ddt.data(expr.logic_and, expr.logic_or)
    def test_binary_logic_forbidden(self, function):
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(3.0, 3.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(3, 3.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(3.0, 3)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Duration.dt(1000), Duration.dt(1000))

    @ddt.data(
        (expr.equal, expr.Binary.Op.EQUAL),
        (expr.not_equal, expr.Binary.Op.NOT_EQUAL),
    )
    @ddt.unpack
    def test_binary_equal_explicit(self, function, opcode):
        cr = ClassicalRegister(8, "c")
        clbit = Clbit()

        self.assertEqual(
            function(cr, 255),
            expr.Binary(
                opcode, expr.Var(cr, types.Uint(8)), expr.Value(255, types.Uint(8)), types.Bool()
            ),
        )
        self.assertFalse(function(cr, 255).const)

        self.assertEqual(
            function(7, cr),
            expr.Binary(
                opcode,
                expr.Value(7, types.Uint(8)),
                expr.Var(cr, types.Uint(8)),
                types.Bool(),
            ),
        )
        self.assertFalse(function(7, cr).const)

        self.assertEqual(
            function(clbit, True),
            expr.Binary(
                opcode,
                expr.Var(clbit, types.Bool()),
                expr.Value(True, types.Bool()),
                types.Bool(),
            ),
        )
        self.assertFalse(function(clbit, True).const)

        self.assertEqual(
            function(expr.lift(7.0), 7.0),
            expr.Binary(
                opcode,
                expr.Value(7.0, types.Float()),
                expr.Value(7.0, types.Float()),
                types.Bool(),
            ),
        )
        self.assertTrue(function(expr.lift(7.0), 7.0).const)

        self.assertEqual(
            function(expr.lift(Duration.ms(1000)), Duration.s(1)),
            expr.Binary(
                opcode,
                expr.Value(Duration.ms(1000), types.Duration()),
                expr.Value(Duration.s(1), types.Duration()),
                types.Bool(),
            ),
        )
        self.assertTrue(function(expr.lift(Duration.ms(1000)), Duration.s(1)).const)

    @ddt.data(expr.equal, expr.not_equal)
    def test_binary_equal_forbidden(self, function):
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Clbit(), ClassicalRegister(3, "c"))
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(ClassicalRegister(3, "c"), False)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(5, True)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(True, 5.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(5, 5.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(ClassicalRegister(3, "c"), 5.0)

    @ddt.data(
        (expr.less, expr.Binary.Op.LESS),
        (expr.less_equal, expr.Binary.Op.LESS_EQUAL),
        (expr.greater, expr.Binary.Op.GREATER),
        (expr.greater_equal, expr.Binary.Op.GREATER_EQUAL),
    )
    @ddt.unpack
    def test_binary_relation_explicit(self, function, opcode):
        cr = ClassicalRegister(8, "c")

        self.assertEqual(
            function(cr, 200),
            expr.Binary(
                opcode, expr.Var(cr, types.Uint(8)), expr.Value(200, types.Uint(8)), types.Bool()
            ),
        )
        self.assertFalse(function(cr, 200).const)

        self.assertEqual(
            function(12, cr),
            expr.Binary(
                opcode,
                expr.Value(12, types.Uint(8)),
                expr.Var(cr, types.Uint(8)),
                types.Bool(),
            ),
        )
        self.assertFalse(function(12, cr).const)

        self.assertEqual(
            function(expr.lift(12.0, types.Float()), expr.lift(12.0)),
            expr.Binary(
                opcode,
                expr.Value(12.0, types.Float()),
                expr.Value(12.0, types.Float()),
                types.Bool(),
            ),
        )
        self.assertTrue(function(expr.lift(12.0, types.Float()), expr.lift(12.0)).const)

        self.assertEqual(
            function(
                expr.lift(Duration.ms(1000), types.Duration()),
                expr.lift(Duration.s(1)),
            ),
            expr.Binary(
                opcode,
                expr.Value(Duration.ms(1000), types.Duration()),
                expr.Value(Duration.s(1), types.Duration()),
                types.Bool(),
            ),
        )
        self.assertTrue(
            function(
                expr.lift(Duration.ms(1000), types.Duration()),
                expr.lift(Duration.s(1)),
            ).const
        )

    @ddt.data(expr.less, expr.less_equal, expr.greater, expr.greater_equal)
    def test_binary_relation_forbidden(self, function):
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Clbit(), ClassicalRegister(3, "c"))
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(ClassicalRegister(3, "c"), False)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Clbit(), Clbit())
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(True, 5.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(5, 5.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(ClassicalRegister(3, "c"), 5.0)

    def test_index_explicit(self):
        cr = ClassicalRegister(4, "c")
        a = expr.Var.new("a", types.Uint(8))

        self.assertEqual(
            expr.index(cr, 3),
            expr.Index(expr.Var(cr, types.Uint(4)), expr.Value(3, types.Uint(2)), types.Bool()),
        )
        self.assertFalse(expr.index(cr, 3).const)

        self.assertEqual(
            expr.index(a, cr),
            expr.Index(a, expr.Var(cr, types.Uint(4)), types.Bool()),
        )
        self.assertFalse(expr.index(a, cr).const)

        self.assertEqual(
            expr.index(255, 1),
            expr.Index(expr.Value(255, types.Uint(8)), expr.Value(1, types.Uint(1)), types.Bool()),
        )
        self.assertTrue(expr.index(255, 1).const)

    def test_index_forbidden(self):
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.index(Clbit(), 3)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.index(ClassicalRegister(3, "a"), False)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.index(ClassicalRegister(3, "a"), 1.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.index(0xFFFF, 1.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.index(ClassicalRegister(3, "a"), 1.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.index(Duration.dt(1000), 1)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.index(Duration.dt(1000), Duration.dt(1000))

    @ddt.data(
        (expr.shift_left, expr.Binary.Op.SHIFT_LEFT),
        (expr.shift_right, expr.Binary.Op.SHIFT_RIGHT),
    )
    @ddt.unpack
    def test_shift_explicit(self, function, opcode):
        cr = ClassicalRegister(8, "c")
        a = expr.Var.new("a", types.Uint(4))

        self.assertEqual(
            function(cr, 5),
            expr.Binary(
                opcode, expr.Var(cr, types.Uint(8)), expr.Value(5, types.Uint(3)), types.Uint(8)
            ),
        )
        self.assertFalse(function(cr, 5).const)

        self.assertEqual(
            function(a, cr),
            expr.Binary(opcode, a, expr.Var(cr, types.Uint(8)), types.Uint(4)),
        )
        self.assertFalse(function(a, cr).const)

        self.assertEqual(
            function(3, 5, types.Uint(8)),
            expr.Binary(
                opcode, expr.Value(3, types.Uint(8)), expr.Value(5, types.Uint(3)), types.Uint(8)
            ),
        )
        self.assertTrue(function(3, 5, types.Uint(8)).const)

    @ddt.data(expr.shift_left, expr.shift_right)
    def test_shift_forbidden(self, function):
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Clbit(), ClassicalRegister(3, "c"))
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(ClassicalRegister(3, "c"), False)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Clbit(), Clbit())
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(0xFFFF, 2.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(255.0, 1)
        with self.assertRaisesRegex(TypeError, "cannot losslessly represent"):
            function(expr.lift(5.0), 3, types.Uint(8))
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Duration.dt(1000), 1)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Duration.dt(1000), Duration.dt(1000))

    @ddt.data(
        (expr.add, expr.Binary.Op.ADD),
        (expr.sub, expr.Binary.Op.SUB),
    )
    @ddt.unpack
    def test_binary_sum_explicit(self, function, opcode):
        cr = ClassicalRegister(8, "c")

        self.assertEqual(
            function(cr, 200),
            expr.Binary(
                opcode, expr.Var(cr, types.Uint(8)), expr.Value(200, types.Uint(8)), types.Uint(8)
            ),
        )
        self.assertFalse(function(cr, 200).const)

        self.assertEqual(
            function(12, cr),
            expr.Binary(
                opcode,
                expr.Value(12, types.Uint(8)),
                expr.Var(cr, types.Uint(8)),
                types.Uint(8),
            ),
        )
        self.assertFalse(function(12, cr).const)

        self.assertEqual(
            function(12.5, 2.0),
            expr.Binary(
                opcode,
                expr.Value(12.5, types.Float()),
                expr.Value(2.0, types.Float()),
                types.Float(),
            ),
        )
        self.assertTrue(function(12.5, 2.0).const)

        self.assertEqual(
            function(
                expr.lift(Duration.ms(1000), types.Duration()),
                expr.lift(Duration.s(1)),
            ),
            expr.Binary(
                opcode,
                expr.Value(Duration.ms(1000), types.Duration()),
                expr.Value(Duration.s(1), types.Duration()),
                types.Duration(),
            ),
        )
        self.assertTrue(
            function(
                expr.lift(Duration.ms(1000), types.Duration()),
                expr.lift(Duration.s(1)),
            ).const
        )

    @ddt.data(expr.add, expr.sub)
    def test_binary_sum_forbidden(self, function):
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Clbit(), ClassicalRegister(3, "c"))
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(ClassicalRegister(3, "c"), False)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Clbit(), Clbit())
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(0xFFFF, 2.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(255.0, 1)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Duration.dt(1000), 1)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Duration.dt(1000), 1.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            function(Duration.dt(1000), expr.lift(1.0))

    def test_mul_explicit(self):
        cr = ClassicalRegister(8, "c")

        self.assertEqual(
            expr.mul(cr, 200),
            expr.Binary(
                expr.Binary.Op.MUL,
                expr.Var(cr, types.Uint(8)),
                expr.Value(200, types.Uint(8)),
                types.Uint(8),
            ),
        )
        self.assertFalse(expr.mul(cr, 200).const)

        self.assertEqual(
            expr.mul(12, cr),
            expr.Binary(
                expr.Binary.Op.MUL,
                expr.Value(12, types.Uint(8)),
                expr.Var(cr, types.Uint(8)),
                types.Uint(8),
            ),
        )
        self.assertFalse(expr.mul(12, cr).const)

        self.assertEqual(
            expr.mul(expr.lift(12), cr),
            expr.Binary(
                expr.Binary.Op.MUL,
                # Explicit cast required to get from Uint(4) to Uint(8)
                expr.Cast(expr.Value(12, types.Uint(4)), types.Uint(8), implicit=False),
                expr.Var(cr, types.Uint(8)),
                types.Uint(8),
            ),
        )
        self.assertFalse(expr.mul(12, cr).const)

        self.assertEqual(
            expr.mul(expr.lift(12, types.Uint(8)), expr.lift(12)),
            expr.Binary(
                expr.Binary.Op.MUL,
                expr.Value(12, types.Uint(8)),
                expr.Cast(
                    expr.Value(12, types.Uint(4)),
                    types.Uint(8),
                    implicit=False,
                ),
                types.Uint(8),
            ),
        )
        self.assertTrue(expr.mul(expr.lift(12, types.Uint(8)), expr.lift(12)).const)

        self.assertEqual(
            expr.mul(expr.lift(12.0, types.Float()), expr.lift(12.0)),
            expr.Binary(
                expr.Binary.Op.MUL,
                expr.Value(12.0, types.Float()),
                expr.Value(12.0, types.Float()),
                types.Float(),
            ),
        )
        self.assertTrue(expr.mul(expr.lift(12.0, types.Float()), expr.lift(12.0)).const)

        self.assertEqual(
            expr.mul(Duration.ms(1000), 2.0),
            expr.Binary(
                expr.Binary.Op.MUL,
                expr.Value(Duration.ms(1000), types.Duration()),
                expr.Value(2.0, types.Float()),
                types.Duration(),
            ),
        )
        self.assertTrue(expr.mul(Duration.ms(1000), 2.0).const)

        self.assertEqual(
            expr.mul(2.0, Duration.ms(1000)),
            expr.Binary(
                expr.Binary.Op.MUL,
                expr.Value(2.0, types.Float()),
                expr.Value(Duration.ms(1000), types.Duration()),
                types.Duration(),
            ),
        )
        self.assertTrue(expr.mul(2.0, Duration.ms(1000)).const)

        self.assertEqual(
            expr.mul(2, Duration.ms(1000)),
            expr.Binary(
                expr.Binary.Op.MUL,
                expr.Value(2, types.Uint(2)),
                expr.Value(Duration.ms(1000), types.Duration()),
                types.Duration(),
            ),
        )
        self.assertTrue(expr.mul(2, Duration.ms(1000)).const)

    def test_mul_forbidden(self):
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.mul(Clbit(), ClassicalRegister(3, "c"))
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.mul(ClassicalRegister(3, "c"), False)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.mul(Clbit(), Clbit())
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.mul(0xFFFF, 2.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.mul(255.0, 1)
        with self.assertRaisesRegex(TypeError, "cannot multiply two durations"):
            expr.mul(Duration.dt(1000), Duration.dt(1000))

    def test_div_explicit(self):
        cr = ClassicalRegister(8, "c")

        self.assertEqual(
            expr.div(cr, 200),
            expr.Binary(
                expr.Binary.Op.DIV,
                expr.Var(cr, types.Uint(8)),
                expr.Value(200, types.Uint(8)),
                types.Uint(8),
            ),
        )
        self.assertFalse(expr.div(cr, 200).const)

        self.assertEqual(
            expr.div(12, cr),
            expr.Binary(
                expr.Binary.Op.DIV,
                expr.Value(12, types.Uint(8)),
                expr.Var(cr, types.Uint(8)),
                types.Uint(8),
            ),
        )
        self.assertFalse(expr.div(12, cr).const)

        self.assertEqual(
            expr.div(expr.lift(12), cr),
            expr.Binary(
                expr.Binary.Op.DIV,
                # Explicit cast required to get from Uint(4) to Uint(8)
                expr.Cast(expr.Value(12, types.Uint(4)), types.Uint(8), implicit=False),
                expr.Var(cr, types.Uint(8)),
                types.Uint(8),
            ),
        )
        self.assertFalse(expr.div(expr.lift(12), cr).const)

        self.assertEqual(
            expr.div(expr.lift(12, types.Uint(8)), expr.lift(12)),
            expr.Binary(
                expr.Binary.Op.DIV,
                expr.Value(12, types.Uint(8)),
                expr.Cast(
                    expr.Value(12, types.Uint(4)),
                    types.Uint(8),
                    implicit=False,
                ),
                types.Uint(8),
            ),
        )
        self.assertTrue(expr.div(expr.lift(12, types.Uint(8)), expr.lift(12)).const)

        self.assertEqual(
            expr.div(expr.lift(12.0, types.Float()), expr.lift(12.0)),
            expr.Binary(
                expr.Binary.Op.DIV,
                expr.Value(12.0, types.Float()),
                expr.Value(12.0, types.Float()),
                types.Float(),
            ),
        )
        self.assertTrue(expr.div(expr.lift(12.0, types.Float()), expr.lift(12.0)).const)

        self.assertEqual(
            expr.div(Duration.ms(1000), 2.0),
            expr.Binary(
                expr.Binary.Op.DIV,
                expr.Value(Duration.ms(1000), types.Duration()),
                expr.Value(2.0, types.Float()),
                types.Duration(),
            ),
        )
        self.assertTrue(expr.div(Duration.ms(1000), 2.0).const)

        self.assertEqual(
            expr.div(Duration.ms(1000), 2),
            expr.Binary(
                expr.Binary.Op.DIV,
                expr.Value(Duration.ms(1000), types.Duration()),
                expr.Value(2, types.Uint(2)),
                types.Duration(),
            ),
        )
        self.assertTrue(expr.div(Duration.ms(1000), 2).const)

        self.assertEqual(
            expr.div(Duration.ms(1000), Duration.ms(1000)),
            expr.Binary(
                expr.Binary.Op.DIV,
                expr.Value(Duration.ms(1000), types.Duration()),
                expr.Value(Duration.ms(1000), types.Duration()),
                types.Float(),
            ),
        )
        self.assertTrue(expr.div(Duration.ms(1000), Duration.ms(1000)).const)

    def test_div_forbidden(self):
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.div(Clbit(), ClassicalRegister(3, "c"))
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.div(ClassicalRegister(3, "c"), False)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.div(Clbit(), Clbit())
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.div(0xFFFF, 2.0)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.div(255.0, 1)
        with self.assertRaisesRegex(TypeError, "invalid types"):
            expr.div(255.0, Duration.dt(1000))

    def test_unary_negate_forbidden(self):
        with self.assertRaisesRegex(TypeError, "cannot apply"):
            expr.negate(True)
        with self.assertRaisesRegex(TypeError, "cannot apply"):
            expr.negate(Clbit())
        with self.assertRaisesRegex(TypeError, "cannot apply"):
            cr = ClassicalRegister(3)
            expr.negate(cr)
        with self.assertRaisesRegex(TypeError, "cannot apply"):
            expr.negate(expr.Value(5, types.Uint(3)))

    def test_unary_negate_explicit(self):

        self.assertEqual(
            expr.negate(7.0),
            expr.Unary(
                expr.Unary.Op.NEGATE,
                expr.Value(7.0, types.Float()),
                types.Float(),
            ),
        )

        self.assertEqual(
            expr.negate(Duration.dt(1000)),
            expr.Unary(
                expr.Unary.Op.NEGATE,
                expr.Value(Duration.dt(1000), types.Duration()),
                types.Duration(),
            ),
        )
