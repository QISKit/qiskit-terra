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

"""Test parameter expression."""

import cmath
import math
import unittest

from test import combine
from test import QiskitTestCase

import ddt

from qiskit.circuit import Parameter, ParameterExpression
from qiskit.utils.optionals import HAS_SYMPY

param_x = Parameter("x")
param_y = Parameter("y")
nested_expr = param_x + param_y - param_x
nested_expr = nested_expr.subs({param_y: param_x})

operands = [
    Parameter("a"),
    Parameter("dai"),
    complex(3.14, -3.14),
    complex(1.0, 1.0),
    complex(0, 1),
    complex(-1, 0),
    2.3,
    5,
    (-5),
    1.0,
    -1.0,
    0,
    0.0,
    complex(0, 0),
    Parameter("ab") + 2 - 2,
    Parameter("abc") ** 1.0,
    Parameter("abcd") / 1,
    Parameter("X") * 1.0,
    Parameter("Y") ** complex(1.0, 0),
    Parameter("abcd_complex") / complex(1, 0),
    nested_expr,
]

bind_values = [math.pi, -math.pi, 5, -5, complex(2, 1), complex(-1, 2), 0, complex(0, 0)]
real_values = [0.41, 0.9, -0.83, math.pi, -math.pi / 124, -42.42]


@ddt.ddt
class TestParameterExpression(QiskitTestCase):
    """Test parameter expression."""

    @combine(
        left=operands,
        right=operands,
        bind_value=bind_values,
        name="{left}_plus_{right}_bind_{bind_value}",
    )
    def test_addition_simple(self, left, right, bind_value):
        """Test expression addition."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left + right
            res = expr.bind(dict.fromkeys(expr.parameters, bind_value))
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), 2.0 * bind_value)
            elif not isinstance(left, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), left + bind_value)
            elif not isinstance(right, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), right + bind_value)

    @combine(
        left=operands,
        right=operands,
        bind_value=bind_values,
        name="{left}_minus_{right}_bind_{bind_value}",
    )
    def test_subtraction_simple(self, left, right, bind_value):
        """Test expression subtraction."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left - right
            res = expr.bind(dict.fromkeys(expr.parameters, bind_value))
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), 0.0)
            elif not isinstance(left, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), left - bind_value)
            elif not isinstance(right, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), bind_value - right)

    @combine(
        left=operands,
        right=operands,
        bind_value=bind_values,
        name="{left}_mul_{right}_bind_{bind_value}",
    )
    def test_multiplication_simple(self, left, right, bind_value):
        """Test expression multiplication."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left * right
            res = expr.bind(dict.fromkeys(expr.parameters, bind_value))
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), bind_value * bind_value)
            elif not isinstance(left, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), left * bind_value)
            elif not isinstance(right, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), bind_value * right)

    @combine(
        left=operands,
        right=operands,
        bind_value=bind_values,
        name="{left}_div_{right}_bind_{bind_value}",
    )
    def test_division_simple(self, left, right, bind_value):
        """Test expression division."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            if not isinstance(right, ParameterExpression) and right == 0:
                with self.assertRaises(ZeroDivisionError):
                    _ = left / right
                return
            expr = left / right
            try:
                res = expr.bind(dict.fromkeys(expr.parameters, bind_value))
            except ZeroDivisionError:
                self.assertIsInstance(right, ParameterExpression)
                self.assertAlmostEqual(bind_value, 0)
                return
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), 1.0)
            elif not isinstance(left, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), left / bind_value)
            elif not isinstance(right, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), bind_value / right)

    @combine(
        left=operands,
        right=operands,
        name="{left}_pow_{right}",
    )
    def test_pow_simple(self, left, right):
        """Test expression pow."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left**right
            res = expr.bind(dict.fromkeys(expr.parameters, 1.0))
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), 1.0)
            elif not isinstance(left, ParameterExpression):
                if isinstance(left, complex):
                    self.assertAlmostEqual(res.numeric(), left)
                else:
                    self.assertAlmostEqual(res.numeric(), left)
            elif not isinstance(right, ParameterExpression):
                self.assertAlmostEqual(res.numeric(), 1.0**right)

    @combine(
        left=operands,
        right=operands,
        name="{left}_pow_{right}",
    )
    def test_pow_complex_binding(self, left, right):
        """Test expression pow with complex binding."""
        if isinstance(left, ParameterExpression) or isinstance(right, ParameterExpression):
            expr = left**right
            res = expr.bind({x: complex(1.0, 1.0) for x in expr.parameters})
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
                self.assertAlmostEqual(complex(res), complex(1.0, 1.0) ** complex(1.0, 1.0))
            elif not isinstance(left, ParameterExpression):
                if left != 0:
                    self.assertAlmostEqual(complex(res), left ** complex(1.0, 1.0))
                else:
                    with self.assertRaises(ZeroDivisionError):
                        _ = left ** complex(1.0, 1.0)
            elif not isinstance(right, ParameterExpression):
                self.assertAlmostEqual(complex(res), complex(1.0, 1.0) ** right)

    def test_pow_creates_complex(self):
        """Test a complex is created when appropriate."""
        param_a = Parameter("A")
        param_b = Parameter("B")
        param_c = Parameter("C")
        expr = param_a + param_b + param_c
        expr = expr.subs({param_b: param_a + 2 * param_c})
        expr = expr.subs({param_a: -param_a, param_c: -param_c})
        expr = expr**0.5
        res = expr.bind({param_a: 2, param_c: 2})
        self.assertFalse(res.is_real())
        # Expected is sqrt(-10):
        self.assertAlmostEqual(complex(0, 3.1622776601683795), complex(res))

    @combine(expression=operands, bind_value=bind_values, name="{expression}_bind_{bind_value}")
    def test_abs_simple(self, expression, bind_value):
        """Test expression abs."""
        if isinstance(expression, ParameterExpression):
            expr = abs(expression)
            res = expr.bind(dict.fromkeys(expr.parameters, bind_value))
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, abs(bind_value))

    @combine(expression=operands)
    def test_acos_simple(self, expression):
        """Test expression arccos."""
        if isinstance(expression, ParameterExpression):
            expr = expression.arccos()
            res = expr.bind(dict.fromkeys(expr.parameters, 0.2))
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.acos(0.2))
            # Test negative
            expr = expression.arccos()
            res = expr.bind(dict.fromkeys(expr.parameters, -0.3))
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.acos(-0.3))

    @combine(expression=operands)
    def test_asin_simple(self, expression):
        """Test expression arcsin."""
        if isinstance(expression, ParameterExpression):
            expr = expression.arcsin()
            res = expr.bind(dict.fromkeys(expr.parameters, 0.2))
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.asin(0.2))
            # Test negative
            expr = expression.arcsin()
            res = expr.bind(dict.fromkeys(expr.parameters, -0.3))
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.asin(-0.3))

    @combine(expression=operands)
    def test_atan_simple(self, expression):
        """Test expression arctan."""
        if isinstance(expression, ParameterExpression):
            expr = expression.arctan()
            res = expr.bind(dict.fromkeys(expr.parameters, 0.2))
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.atan(0.2))
            # Test negative
            expr = expression.arctan()
            res = expr.bind(dict.fromkeys(expr.parameters, -0.3))
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, math.atan(-0.3))

    @combine(expression=operands, bind_value=bind_values, name="{expression}_bind_{bind_value}")
    def test_conjugate_simple(self, expression, bind_value):
        """Test expression conjugate."""
        if isinstance(expression, ParameterExpression):
            expr = expression.conjugate()
            res = expr.bind(dict.fromkeys(expr.parameters, bind_value))
            self.assertIsInstance(res, ParameterExpression)
            numeric = res.numeric()
            if isinstance(numeric, complex):
                self.assertEqual(res, bind_value.conjugate())
            else:
                self.assertEqual(res, bind_value)

    @combine(expression=operands, bind_value=bind_values, name="{expression}_bind_{bind_value}")
    def test_cos_simple(self, expression, bind_value):
        """Test expression cos."""
        if isinstance(expression, ParameterExpression):
            expr = expression.cos()
            res = expr.bind(dict.fromkeys(expr.parameters, bind_value))
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(bind_value, complex):
                self.assertAlmostEqual(res.numeric(), cmath.cos(bind_value))
            else:
                self.assertAlmostEqual(res.numeric(), math.cos(bind_value))

    @combine(expression=operands, bind_value=bind_values, name="{expression}_bind_{bind_value}")
    def test_sin_simple(self, expression, bind_value):
        """Test expression sin."""
        if isinstance(expression, ParameterExpression):
            expr = expression.sin()
            res = expr.bind(dict.fromkeys(expr.parameters, bind_value))
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(bind_value, complex):
                self.assertAlmostEqual(res.numeric(), cmath.sin(bind_value))
            else:
                self.assertAlmostEqual(res.numeric(), math.sin(bind_value))

    @combine(expression=operands, bind_value=bind_values, name="{expression}_bind_{bind_value}")
    def test_tan_simple(self, expression, bind_value):
        """Test expression tan."""
        if isinstance(expression, ParameterExpression):
            expr = expression.tan()
            res = expr.bind(dict.fromkeys(expr.parameters, bind_value))
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(bind_value, complex):
                self.assertAlmostEqual(res.numeric(), cmath.tan(bind_value))
            else:
                self.assertAlmostEqual(res.numeric(), math.tan(bind_value))

    @combine(expression=operands, bind_value=bind_values, name="{expression}_bind_{bind_value}")
    def test_exp_simple(self, expression, bind_value):
        """Test expression exp."""
        if isinstance(expression, ParameterExpression):
            expr = expression.exp()
            res = expr.bind(dict.fromkeys(expr.parameters, bind_value))
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(bind_value, complex):
                self.assertAlmostEqual(res.numeric(), cmath.exp(bind_value))
            else:
                self.assertAlmostEqual(res.numeric(), math.exp(bind_value))

    @combine(expression=operands, bind_value=bind_values, name="{expression}_bind_{bind_value}")
    def test_log_simple(self, expression, bind_value):
        """Test expression log."""
        if isinstance(expression, ParameterExpression) and bind_value != 0:
            expr = expression.log()
            res = expr.bind(dict.fromkeys(expr.parameters, bind_value))
            self.assertIsInstance(res, ParameterExpression)
            if isinstance(bind_value, complex):
                self.assertAlmostEqual(res.numeric(), cmath.log(bind_value))
            elif bind_value > 0:
                self.assertAlmostEqual(res.numeric(), math.log(bind_value))
            else:
                self.assertAlmostEqual(res.numeric(), cmath.log(bind_value))

    @combine(expression=operands)
    def test_sign_simple(self, expression):
        """Test expression sign."""
        if isinstance(expression, ParameterExpression) and expression.is_real():
            expr = expression.sign()
            res = expr.bind(dict.fromkeys(expr.parameters, -0.1))
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, -1)
            expr = expression.sign()
            res = expr.bind(dict.fromkeys(expr.parameters, 0.1))
            self.assertEqual(res, 1)
            expr = expression.sign()
            res = expr.bind(dict.fromkeys(expr.parameters, 0.0))
            self.assertEqual(res, 0)
            expr = expression.sign()
            res = expr.bind(dict.fromkeys(expr.parameters, -2))
            self.assertIsInstance(res, ParameterExpression)
            self.assertEqual(res, -1)
            expr = expression.sign()
            res = expr.bind(dict.fromkeys(expr.parameters, 5))
            self.assertEqual(res, 1)
            expr = expression.sign()
            res = expr.bind(dict.fromkeys(expr.parameters, 0))
            self.assertEqual(res, 0)

    @combine(expression=operands)
    def test_is_real(self, expression):
        """Test the is_real() method."""
        if isinstance(expression, ParameterExpression):
            res = expression.bind({x: complex(1.0, 0.0) for x in expression.parameters})
            self.assertTrue(res.is_real())
            res = expression.bind({x: complex(1.0, 1.0) for x in expression.parameters})
            self.assertFalse(res.is_real())
            res = expression.bind(dict.fromkeys(expression.parameters, 1.0))
            self.assertTrue(res.is_real())
            res = expression.bind(dict.fromkeys(expression.parameters, 5))
            self.assertTrue(res.is_real())
            self.assertFalse(expression.is_real())

    @combine(expression=operands)
    def test_casting(self, expression):
        """Test casting"""
        if isinstance(expression, ParameterExpression):
            res = expression.bind({x: complex(1.0, 0.0) for x in expression.parameters})
            self.assertIsInstance(complex(res), complex)
            self.assertIsInstance(float(res), float)
            self.assertIsInstance(int(res), int)
            self.assertEqual(res, 1)
            res = expression.bind({x: complex(1.0, 1.0) for x in expression.parameters})
            self.assertIsInstance(complex(res), complex)
            with self.assertRaises(TypeError):
                float(res)
            with self.assertRaises(TypeError):
                int(res)
            self.assertAlmostEqual(complex(res), complex(1.0, 1.0))
            res = expression.bind(dict.fromkeys(expression.parameters, 1.0))
            self.assertIsInstance(complex(res), complex)
            self.assertIsInstance(float(res), float)
            self.assertIsInstance(int(res), int)
            self.assertEqual(res, 1.0)
            res = expression.bind(dict.fromkeys(expression.parameters, 5))
            self.assertIsInstance(complex(res), complex)
            self.assertIsInstance(float(res), float)
            self.assertIsInstance(int(res), int)
            self.assertEqual(res, 5)

    @combine(expression=operands)
    def test_numeric(self, expression):
        """Test numeric"""
        if isinstance(expression, ParameterExpression):
            res = expression.bind({x: complex(1.0, 0.0) for x in expression.parameters}).numeric()
            self.assertIsInstance(res, float)
            self.assertEqual(res, 1)
            res = expression.bind({x: complex(1.0, 1.0) for x in expression.parameters}).numeric()
            self.assertIsInstance(res, complex)
            self.assertAlmostEqual(complex(res), complex(1.0, 1.0))
            res = expression.bind(dict.fromkeys(expression.parameters, 1.0)).numeric()
            self.assertIsInstance(float(res), float)
            self.assertEqual(res, 1.0)
            res = expression.bind(dict.fromkeys(expression.parameters, 5)).numeric()
            self.assertIsInstance(int(res), int)
            self.assertEqual(res, 5)

    def test_derivatives_numeric(self):
        """Test derivatives with numerical values."""
        methods_and_references = [
            ("abs", lambda x: x / abs(x)),
            ("exp", math.exp),
            ("log", lambda x: 1 / x),
            ("sin", math.cos),
            ("cos", lambda x: -math.sin(x)),
            ("tan", lambda x: 1 / math.cos(x) ** 2),
            ("arccos", lambda x: -((1 - x**2) ** (-0.5))),
            ("arcsin", lambda x: (1 - x**2) ** (-0.5)),
            ("arctan", lambda x: 1 / (1 + x**2)),
            ("conjugate", lambda _: 1),
        ]

        x = Parameter("x")
        for method, reference in methods_and_references:
            expr = getattr(x, method)()
            d_expr = expr.gradient(x)

            for value in real_values:
                if method == "log" and value <= 0:
                    continue  # log not defined
                if method in ["arccos", "arcsin", "arctan"] and abs(value) >= 1 - 1e-10:
                    continue  # arc-funcs not defined

                with self.subTest(method=method, value=value):
                    ref = reference(value)
                    if isinstance(d_expr, ParameterExpression):
                        val = d_expr.bind({x: value}).numeric()
                    else:
                        val = d_expr  # d/dx conj(x) == 1

                    self.assertAlmostEqual(ref, val)

    def test_sign_derivative_errors(self):
        """Test the derivative of sign errors (not supported right now)."""
        x = Parameter("x")
        expr = x.sign()

        with self.assertRaises(RuntimeError):
            _ = expr.gradient(x)

    @unittest.skipUnless(HAS_SYMPY, "Sympy is required for this test")
    def test_sympify_all_ops(self):
        """Test the sympify function works for all the supported operations."""

        import sympy

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        d = Parameter("d")

        expression = (a + b.sin() * 0.25) * c**2
        final_expr = (
            (expression.cos() + d.arccos() - d.arcsin() + d.arctan() + d.tan()) / d.exp()
            + expression.gradient(a)
            + expression.log().sign()
            - a.sin()
            - b.conjugate()
        )
        final_expr = final_expr.abs()
        final_expr = final_expr.subs({c: a})
        result = final_expr.sympify()

        a = sympy.Symbol("a")
        b = sympy.Symbol("b")
        c = sympy.Symbol("c")
        d = sympy.Symbol("d")
        expression = (a + sympy.sin(b) * 0.25) * c**2
        expected = (
            (sympy.cos(expression) + sympy.acos(d) - sympy.asin(d) + sympy.atan(d) + sympy.tan(d))
            / sympy.exp(d)
            + expression.diff(a)
            + sympy.sign(sympy.log(expression))
            - sympy.sin(a)
            - sympy.conjugate(b)
        )
        expected = sympy.Abs(expected)
        expected = expected.subs({c: a})

        self.assertEqual(result, expected)
