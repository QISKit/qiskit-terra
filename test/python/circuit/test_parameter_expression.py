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
import pickle
import copy
import functools
import itertools

from test import combine
from test import QiskitTestCase

import ddt

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.utils.optionals import HAS_SYMPY

param_x = Parameter("x")
param_y = Parameter("y")
param_z = Parameter("z")
nested_expr = param_x + param_y - param_x
nested_expr = nested_expr.subs({param_y: param_x})

vector = ParameterVector("vec", 1000)
nested_vector_expr = vector[500] + vector[256] - vector[500]
for i in range(1000):
    nested_vector_expr += vector[i] - vector[i]


operands = [
    Parameter("a"),
    Parameter("dai"),
    ParameterVector("a", 100)[42],
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
    ParameterVector("b", 1)[0] + (0 * 1) * Parameter("ZERO"),
    nested_expr,
    nested_vector_expr,
]

bind_values = [math.pi, -math.pi, 5, -5, complex(2, 1), complex(-1, 2), 0, complex(0, 0)]
real_values = [0.41, 0.9, -0.83, math.pi, -math.pi / 124, -42.42]


@ddt.ddt
class TestParameterExpression(QiskitTestCase):
    """Test parameter expression."""

    def assertStructurallyEqualResult(
        self, left: ParameterExpression, right: ParameterExpression, result: bool
    ):
        """Assert that ``left.structurally_equal(right) == result`` with a better error message."""
        self.assertIsInstance(left, ParameterExpression)
        self.assertIsInstance(right, ParameterExpression)
        if left.structurally_equal(right) != result:
            op = "not ==" if result else "=="
            msg = f"Assertion failure: '{left}' should {op} '{right}'."
            raise self.failureException(msg)

    def assertStructurallyEqual(self, left, right):
        self.assertStructurallyEqualResult(left, right, True)

    def assertNotStructurallyEqual(self, left, right):
        self.assertStructurallyEqualResult(left, right, False)

    @ddt.data(param_x, param_x + param_y, (param_x + 1.0).bind({param_x: 1.0}))
    def test_num_parameters(self, expr):
        """Do the two ways of getting the number of unbound parameters agree?"""
        self.assertEqual(len(expr.parameters), expr.num_parameters)

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
            ("conjugate", lambda x: 1),
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
                        # allow unknown parameters since the derivative could evaluate to a const
                        val = d_expr.bind({x: value}, allow_unknown_parameters=True).numeric()
                    else:
                        val = d_expr  # d/dx conj(x) == 1

                    self.assertAlmostEqual(ref, val)

    def test_sign_derivative_errors(self):
        """Test the derivative of sign errors (not supported right now)."""
        x = Parameter("x")
        expr = x.sign()

        with self.assertRaises(RuntimeError):
            _ = expr.gradient(x)

    def test_gradient_constant_derivatives(self):
        """Test gradient method returns numeric values for constant derivatives."""
        x = Parameter("x")
        y = Parameter("y")

        test_cases = [
            (x, x, 1.0),
            (x + 0, x, 1.0),
            (0 * x, x, 0.0),
            (x / 2, x, 0.5),
            (x - x, x, 0.0),
            (5 + x - x, x, 0.0),
            (2 * x + y - x, x, 1.0),
        ]

        for expr, param, expected in test_cases:
            with self.subTest(expr=str(expr), param=str(param)):
                result = expr.gradient(param)
                self.assertIsInstance(result, (int, float, complex))
                self.assertEqual(result, expected)

    @ddt.idata(operand for operand in operands if isinstance(operand, ParameterExpression))
    def test_pickle_roundtrip(self, expr):
        pickled = pickle.loads(pickle.dumps(expr))
        copied = copy.copy(expr)
        deep_copied = copy.deepcopy(expr)
        self.assertEqual([expr] * 3, [pickled, copied, deep_copied])
        self.assertEqual(
            [expr.parameters] * 3, [pickled.parameters, copied.parameters, deep_copied.parameters]
        )

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

    @unittest.skipUnless(HAS_SYMPY, "Sympy is required for this test")
    def test_sympify_subs_vector(self):
        """Test an expression with subbed ParameterVectorElements is sympifiable"""
        import sympy

        p_vec = ParameterVector("p", length=2)
        theta = Parameter("theta")

        expression = theta + 1
        expression = expression.subs({theta: p_vec[0]})
        result = expression.sympify()
        expected = sympy.Symbol("p[0]") + 1
        self.assertEqual(expected, result)

    @unittest.skipUnless(HAS_SYMPY, "Sympy is required for this test")
    def test_sympify_rpow_operand_order(self):
        """Test that sympify correctly handles RPOW operations with swapped operands.

        This test verifies the fix for issue #15583, where ParameterExpression.sympify()
        was incorrectly swapping operands for reverse power operations (RPOW).
        """
        import sympy

        a, b = Parameter("a"), Parameter("b")
        # This creates a ** (b - 2), which uses RPOW internally
        res = a ** (b - 2)

        # The string representation should be correct
        self.assertEqual(str(res), "a**(-2 + b)")

        # The sympify result should match the correct order: a ** (b - 2)
        sympy_result = res.sympify()
        expected = sympy.Symbol("a") ** (sympy.Symbol("b") - 2)

        self.assertEqual(sympy_result, expected)

        # Test with a product expression as exponent (as mentioned in the issue)
        res2 = a ** (2 * b)
        sympy_result2 = res2.sympify()
        expected2 = sympy.Symbol("a") ** (2 * sympy.Symbol("b"))
        self.assertEqual(sympy_result2, expected2)

    @unittest.skipUnless(HAS_SYMPY, "Sympy is required for this test")
    def test_sympify_reverse_operations(self):
        """Test that sympify correctly handles all reverse operations (RPOW, RDIV, RSUB).

        This test ensures that reverse operations correctly swap operands when converting
        to sympy expressions. Reverse operations are used when the left operand is numeric
        and the right operand is a ParameterExpression.
        """
        import sympy

        a, b = Parameter("a"), Parameter("b")

        # Test RPOW: a ** (b - 2) should convert to a ** (b - 2)
        # This uses RPOW internally when the exponent is a ParameterExpression
        res_pow = a ** (b - 2)
        sympy_pow = res_pow.sympify()
        expected_pow = sympy.Symbol("a") ** (sympy.Symbol("b") - 2)
        self.assertEqual(sympy_pow, expected_pow)

        # Test RDIV: 2 / a should convert to 2 / a
        # RDIV is used when left operand is numeric and right is ParameterExpression
        res_div = 2 / a
        sympy_div = res_div.sympify()
        expected_div = 2 / sympy.Symbol("a")
        self.assertEqual(sympy_div, expected_div)

        # Test RSUB: 2 - a should convert to 2 - a
        # RSUB is used when left operand is numeric and right is ParameterExpression
        res_sub = 2 - a
        sympy_sub = res_sub.sympify()
        expected_sub = 2 - sympy.Symbol("a")
        self.assertEqual(sympy_sub, expected_sub)

    def test_division_with_addition(self):
        """Test that (v / p) + p evaluates correctly when bound.

        Regression test for bug where (v / p) + p was incorrectly evaluated.

        See https://github.com/Qiskit/qiskit/issues/16263
        """
        p = Parameter("p")

        # Case 1: (v / p) + p
        qc1 = QuantumCircuit(1)
        qc1.rx((2 / p) + p, 0)

        bound1 = qc1.assign_parameters({p: 3})
        actual1 = float(bound1.data[0].operation.params[0])
        expected1 = (2 / 3) + 3

        self.assertAlmostEqual(actual1, expected1, places=10)

        # Case 2: p + (v / p)
        qc2 = QuantumCircuit(1)
        qc2.rx(p + (2 / p), 0)

        bound2 = qc2.assign_parameters({p: 3})
        actual2 = float(bound2.data[0].operation.params[0])
        expected2 = 3 + (2 / 3)

        self.assertAlmostEqual(actual2, expected2, places=10)

    def test_division_with_subtraction(self):
        """Test that (v / p) - p and p - (v / p) evaluate correctly when bound.

        Regression test for bug where (v / p) - p and p - (v / p) were incorrectly evaluated.

        See https://github.com/Qiskit/qiskit/issues/16263
        """
        p = Parameter("p")

        # Case 1: (v / p) - p
        qc1 = QuantumCircuit(1)
        qc1.rx((3 / p) - p, 0)

        bound1 = qc1.assign_parameters({p: 5})
        actual1 = float(bound1.data[0].operation.params[0])
        expected1 = (3 / 5) - 5

        self.assertAlmostEqual(actual1, expected1, places=10)

        # Case 2: p - (v / p)
        qc2 = QuantumCircuit(1)
        qc2.rx(p - (2 / p), 0)

        bound2 = qc2.assign_parameters({p: 3})
        actual2 = float(bound2.data[0].operation.params[0])
        expected2 = 3 - (2 / 3)

        self.assertAlmostEqual(actual2, expected2, places=10)

    def test_nested_division(self):
        """Test that nested division expressions evaluate correctly when bound.

        Regression test for bug where expressions like (a / x) / (b / y) were incorrectly evaluated.

        See https://github.com/Qiskit/qiskit/issues/16262
        """
        x = Parameter("x")
        y = Parameter("y")
        p = Parameter("p")
        q = Parameter("q")

        # Case 1: (a / x) / (b / y)
        qc1 = QuantumCircuit(1)
        qc1.rx((2 / x) / (3 / y), 0)

        bound1 = qc1.assign_parameters({x: 5, y: 7})
        actual1 = float(bound1.data[0].operation.params[0])
        expected1 = (2 / 5) / (3 / 7)

        self.assertAlmostEqual(actual1, expected1, places=10)

        # Case 2: (a * p) / (b / q)
        qc2 = QuantumCircuit(1)
        qc2.rx((2 * p) / (3 / q), 0)

        bound2 = qc2.assign_parameters({p: 5, q: 7})
        actual2 = float(bound2.data[0].operation.params[0])
        expected2 = (2 * 5) / (3 / 7)

        self.assertAlmostEqual(actual2, expected2, places=10)

        # Case 3: (a / p) / (b * q)
        qc3 = QuantumCircuit(1)
        qc3.rx((2 / p) / (3 * q), 0)

        bound3 = qc3.assign_parameters({p: 5, q: 7})
        actual3 = float(bound3.data[0].operation.params[0])
        expected3 = (2 / 5) / (3 * 7)

        self.assertAlmostEqual(actual3, expected3, places=10)

    def test_gradient_composite_base_constant_exponent(self):
        """Test gradient of composite base with constant exponent.

        Regression test for bug where d/dp (f(p))^n was incorrectly computed.
        The correct derivative is: n * (f(p))^(n-1) * f'(p)

        See https://github.com/Qiskit/qiskit/issues/16260
        """
        p = Parameter("p")

        # Case: d/dp (2p)^3 = 3 * (2p)^2 * 2 = 6 * (2p)^2
        expr = (2 * p) ** 3
        gradient = expr.gradient(p)
        actual = float(gradient.bind({p: 2}))
        expected = 3 * (2 * 2) ** 2 * 2  # 3 * 16 * 2 = 96

        self.assertAlmostEqual(actual, expected, places=10)

        # Case: d/dp 2^(3p) = 2^(3p) * ln(2) * 3
        expr = 2 ** (3 * p)
        gradient = expr.gradient(p)
        actual = float(gradient.bind({p: 1}))
        expected = (2**3) * math.log(2) * 3  # 8 * ln(2) * 3

        self.assertAlmostEqual(actual, expected, places=10)

    def test_power_expression_with_addition_subtraction(self):
        """Test that theta**2 + phi**2 and theta**2 - phi**2 evaluate correctly.

        Regression test for bug where power expressions combined with addition/subtraction
        were incorrectly evaluated when parameters are bound.
        See https://github.com/Qiskit/qiskit/issues/16259
        """
        theta = Parameter("theta")
        phi = Parameter("phi")

        # Subtraction case: theta**2 - phi**2
        qc_sub = QuantumCircuit(1)
        qc_sub.rx(theta**2 - phi**2, 0)

        actual_sub = qc_sub.assign_parameters({theta: 5, phi: 3})
        expected_sub_value = 5**2 - 3**2  # 25 - 9 = 16
        actual_sub_value = float(actual_sub.data[0].operation.params[0])

        self.assertAlmostEqual(actual_sub_value, expected_sub_value, places=10)

        # Addition case: theta**2 + phi**2
        qc_add = QuantumCircuit(1)
        qc_add.rx(theta**2 + phi**2, 0)

        actual_add = qc_add.assign_parameters({theta: 5, phi: 3})
        expected_add_value = 5**2 + 3**2  # 25 + 9 = 34
        actual_add_value = float(actual_add.data[0].operation.params[0])

        self.assertAlmostEqual(actual_add_value, expected_add_value, places=10)

    def test_conjugate_gradient(self):
        """Test that gradient of conjugate expression is computed correctly.

        Regression test for bug where d/dp conj(f(p)) returned 1.0 instead of
        the actual derivative f'(p) (assuming real parameters, conj is identity).
        """
        p = Parameter("p")
        q = Parameter("q")

        # d/dp conj(q) should be 0 (q does not depend on p)
        expr = q.conjugate()
        gradient = expr.gradient(p)
        actual = (
            float(gradient.bind({p: 1, q: 2})) if hasattr(gradient, "bind") else float(gradient)
        )
        self.assertAlmostEqual(actual, 0.0, places=10)

        # d/dp conj(p) should be 1 (conj is identity for real p)
        expr = p.conjugate()
        gradient = expr.gradient(p)
        actual = float(gradient.bind({p: 1})) if hasattr(gradient, "bind") else float(gradient)
        self.assertAlmostEqual(actual, 1.0, places=10)

        # d/dp conj(2*p + 3) should be 2
        expr = (2 * p + 3).conjugate()
        gradient = expr.gradient(p)
        actual = float(gradient.bind({p: 5})) if hasattr(gradient, "bind") else float(gradient)
        self.assertAlmostEqual(actual, 2.0, places=10)

        # d/dp (conj(p) + conj(q)) should be 1 (only first term depends on p)
        expr = p.conjugate() + q.conjugate()
        gradient = expr.gradient(p)
        actual = (
            float(gradient.bind({p: 1, q: 2})) if hasattr(gradient, "bind") else float(gradient)
        )
        self.assertAlmostEqual(actual, 1.0, places=10)

    def test_mul_expand_rhs_div(self):
        """Test that multiplication distributes correctly over division.

        Regression test for bug where expand((A+B)*(p/q)) incorrectly computed
        (A+B)*p*q instead of (A*p + B*p)/q.
        """

        a = Parameter("a")
        b = Parameter("b")
        p = Parameter("p")
        q = Parameter("q")

        vals = {a: 2, b: 3, p: 5, q: 7}

        # (a+b)*(p/q) should give a*p/q + b*p/q
        expr = (a + b) * (p / q)
        expected = float((a * p / q + b * p / q).bind(vals))
        actual = float(expr.bind(vals))
        self.assertAlmostEqual(actual, expected, places=10)

        # sin((a+b)*(p/q)) - sin(a*p*q + b*p*q) must NOT be zero
        # (the bug caused these to incorrectly simplify to 0)
        wrong = (a * p * q + b * p * q).sin()
        combined = expr.sin() - wrong
        combined_val = float(combined.bind(vals))
        expected_val = math.sin(float(expr.bind(vals))) - math.sin(
            float((a * p * q + b * p * q).bind(vals))
        )
        self.assertAlmostEqual(combined_val, expected_val, places=10)
        self.assertNotAlmostEqual(combined_val, 0.0, places=10)

    def test_mul_expand_self_div(self):
        """Test that (A/b)*v expands correctly to A*v/b.

        Regression test for bug where mul_expand((A/b)*v) incorrectly computed
        A/(v*b) instead of A*v/b.
        """
        p = Parameter("p")
        q = Parameter("q")
        a = Parameter("a")

        vals = {p: 1, q: 2, a: 1}

        # ((2.0*p + 2.0*q)/a) * 6.0 = 12*(p+q)/a
        # ((p + q)/a) * 3.0 = 3*(p+q)/a
        # These are different; their sins must NOT simplify to 0
        e1 = ((2.0 * p + 2.0 * q) / a) * 6.0
        e2 = ((p + q) / a) * 3.0

        e1_val = float(e1.bind(vals))
        e2_val = float(e2.bind(vals))
        self.assertAlmostEqual(e1_val, 36.0, places=10)
        self.assertAlmostEqual(e2_val, 9.0, places=10)

        combined = e1.sin() - e2.sin()
        combined_val = float(combined.bind(vals))
        expected_val = math.sin(e1_val) - math.sin(e2_val)
        self.assertAlmostEqual(combined_val, expected_val, places=10)
        self.assertNotAlmostEqual(combined_val, 0.0, places=10)

    def test_chain_rule_gradient(self):
        """Test that the chain rule is correctly applied for all function applications.

        Each sub-test uses a non-trivial inner function g(p) = 2*p + 1 (with g'(p) = 2)
        so that a bug which drops the inner derivative would produce the wrong answer.
        For arcsin/arccos, h(p) = 0.1*p + 0.1 is used to keep values in (-1, 1).
        """
        p = Parameter("p")
        val = 0.5  # evaluation point

        # g(p) = 2p + 1,  g'(p) = 2,  g(0.5) = 2.0
        g = 2 * p + 1
        gval = 2 * val + 1  # 2.0

        with self.subTest("neg"):
            # d/dp (-(2p+1)) = -2
            gradient = (-g).gradient(p)
            expected = -2.0
            self.assertAlmostEqual(gradient, expected, places=10)

        with self.subTest("abs"):
            # d/dp |g(p)| = g(p)/|g(p)| * g'(p)
            gradient = abs(g).gradient(p)
            expected = (gval / abs(gval)) * 2
            actual = gradient.bind({p: val})
            self.assertAlmostEqual(actual, expected, places=10)

        with self.subTest("exp"):
            # d/dp exp(g(p)) = exp(g(p)) * 2
            gradient = g.exp().gradient(p)
            actual = gradient.bind({p: val})
            self.assertAlmostEqual(actual, math.exp(gval) * 2, places=10)

        with self.subTest("log"):
            # d/dp log(g(p)) = 2 / g(p)
            gradient = g.log().gradient(p)
            actual = gradient.bind({p: val})
            self.assertAlmostEqual(actual, 2 / gval, places=10)

        with self.subTest("pow_negative_integer_exponent"):
            # d/dp g(p)^(-2) = -2 * g(p)^(-3) * 2
            gradient = (g**-2).gradient(p)
            expected = -2 * gval**-3 * 2
            actual = gradient.bind({p: val})
            self.assertAlmostEqual(actual, expected, places=10)

        with self.subTest("pow_float_exponent"):
            # d/dp g(p)^0.5 = 0.5 * g(p)^(-0.5) * 2
            gradient = (g**0.5).gradient(p)
            expected = 0.5 * gval**-0.5 * 2
            actual = gradient.bind({p: val})
            self.assertAlmostEqual(actual, expected, places=10)

        with self.subTest("sin"):
            # d/dp sin(g(p)) = cos(g(p)) * 2
            gradient = g.sin().gradient(p)
            actual = gradient.bind({p: val})
            self.assertAlmostEqual(actual, math.cos(gval) * 2, places=10)

        with self.subTest("cos"):
            # d/dp cos(g(p)) = -sin(g(p)) * 2
            gradient = g.cos().gradient(p)
            actual = gradient.bind({p: val})
            self.assertAlmostEqual(actual, -math.sin(gval) * 2, places=10)

        with self.subTest("tan"):
            # d/dp tan(g(p)) = 2 / cos(g(p))^2
            gradient = g.tan().gradient(p)
            actual = gradient.bind({p: val})
            self.assertAlmostEqual(actual, 2 / math.cos(gval) ** 2, places=10)

        # h(p) = 0.1p + 0.1,  h'(p) = 0.1,  h(0.5) = 0.15  (safe for arcsin/arccos)
        h = 0.1 * p + 0.1
        hval = 0.1 * val + 0.1  # 0.15

        with self.subTest("arcsin"):
            # d/dp arcsin(h(p)) = 0.1 / sqrt(1 - h(p)^2)
            gradient = h.arcsin().gradient(p)
            expected = 0.1 / math.sqrt(1 - hval**2)
            actual = gradient.bind({p: val})
            self.assertAlmostEqual(actual, expected, places=10)

        with self.subTest("arccos"):
            # d/dp arccos(h(p)) = -0.1 / sqrt(1 - h(p)^2)
            gradient = h.arccos().gradient(p)
            expected = -0.1 / math.sqrt(1 - hval**2)
            actual = gradient.bind({p: val})
            self.assertAlmostEqual(actual, expected, places=10)

        with self.subTest("arctan"):
            # d/dp arctan(g(p)) = 2 / (1 + g(p)^2)
            gradient = g.arctan().gradient(p)
            actual = gradient.bind({p: val})
            self.assertAlmostEqual(actual, 2 / (1 + gval**2), places=10)

    def test_simplify_multi_parameter_cancellation(self):
        """Test that simplify() handles cancellation across multiple parameters."""
        a = Parameter("a")
        b = Parameter("b")
        expr = a + b - a - b
        simplified = expr.simplify()
        self.assertEqual(simplified.parameters, set())
        self.assertEqual(simplified.numeric(), 0)

    def test_simplify_no_cancellation(self):
        """Test that simplify() leaves non-cancelling expressions intact."""
        a = Parameter("a")
        b = Parameter("b")
        expr = a + b
        self.assertEqual(expr, expr.simplify())

    @ddt.data("__add__", "__sub__", "__mul__", "__truediv__")
    def test_accumulation(self, meth):
        """Test on-the-fly accumulation of numerical values.

        Regression test of
        """
        symbol = Parameter("p")
        values = [(i + 1) / 11 for i in range(50)]

        def accumulator(total, value):
            return getattr(total, meth)(value)

        expression = functools.reduce(accumulator, [symbol] + values)
        if meth == "__truediv__":
            prod = functools.reduce(lambda total, value: total * value, values)
            reference = symbol / prod
        elif meth == "__sub__":
            summed = functools.reduce(lambda total, value: total + value, values)
            reference = symbol - summed
        else:
            accumulated = functools.reduce(accumulator, values)
            reference = accumulator(symbol, accumulated)

        self.assertEqual(reference, expression)

    def test_huge_addition(self):
        """Test additions are simplified on the fly (aka. simpliflied).

        Regression test of #16676.
        """
        p = Parameter("p")
        for _ in range(int(1e6)):
            p += 3.14

        self.assertEqual(p, p.simplify())

    @ddt.data("__add__", "__sub__")
    def test_optimization_same_symbol(self, method):
        """Test optimizations with the same symbol."""
        x = Parameter("x")
        add_sub_terms = [1 + x, -1 + x, 1 - x, -1 - x]
        pow_terms = [x**2, x ** (-2)]
        rpow_terms = [2**x, (-2) ** x]
        mul_terms = [2 * x, -2 * x]
        rdiv_terms = [2 / x, -2 / x]
        div_terms = [x / 2, x / (-2)]

        value = 1.234

        for terms in [add_sub_terms, pow_terms, rpow_terms, mul_terms, rdiv_terms, div_terms]:
            for lhs, rhs in itertools.permutations(terms, 2):
                with self.subTest(lhs=lhs, rhs=rhs):
                    reference = getattr(lhs.bind({x: value}), method)(rhs.bind({x: value}))
                    expression = getattr(lhs, method)(rhs)

                    self.assertEqual(reference, expression.bind({x: value}))

    def test_sub_sub(self):
        """Regression test for two nested subtractions with numeric values."""
        x, y = Parameter("x"), Parameter("y")
        expr1 = x + y
        sub1 = expr1 - 1.0
        sub2 = sub1 - 2.0

        expected = x + y - 3.0
        self.assertEqual(expected, sub2)

    def test_deep_string_parse(self):
        """Test that the string parser can handle very deep expressions."""
        n = 100_000
        a = Parameter("a")
        expr_str = "a" + " ** a" * n
        # This is an explicitly private constructor, but the purpose of the test is for _any_ string
        # constructor; we can change it over to a new API if/when we expose one.
        out = ParameterExpression({"a": a}, expr_str)
        expected = a
        for _ in range(n):
            # TODO: actually this seems like a weirdness in the parser: ` ** ` should be
            # right-associative (so it should be `a**expected`).  We have to use `**` in the test to
            # avoid complexity explosion via attempted simplification.
            expected = expected**a
        self.assertStructurallyEqual(out, expected)

    def test_structurally_equal_simplification(self):
        x = Parameter("x")
        y = Parameter("y")

        # These two expressions do not canonicalise to each other, as of the introduction of the
        # test (2026-09-07), so they should not appear structurally equal.  If canonicalisation is
        # introduced, the test case should be changed (after we've checked that `structurally_equal`
        # is actually correct).
        left = x + y + 1
        right = 1 + y + x
        self.assertEqual(left, right)
        self.assertNotStructurallyEqual(left, right)

    @ddt.unpack
    @ddt.data(
        (param_x, param_x, True),
        (param_x, param_y, False),
        (param_x + 1, param_x + 1, True),
        (param_x + 1, param_x + 2, False),
        (param_x + 1, param_y + 1, False),
        (-param_x, -param_x, True),
        (-param_x, -param_y, False),
        (param_x + (2 * param_y), param_x + (2 * param_y), True),
        (param_x + (2 * param_y), param_x + (2 * param_z), False),
        ((-param_x) ** param_y, (-param_x) ** param_y, True),
        ((-param_x) ** param_y, (-param_y) ** (param_x), False),
        (param_x**-param_y, param_x**-param_y, True),
        (param_x + param_y, param_x - param_y, False),
        (-param_x, param_x.sin(), False),
    )
    def test_structurally_equal(self, left, right, expected):
        self.assertStructurallyEqualResult(left, right, expected)

    @ddt.data("__add__", "__sub__")
    def test_optimization_with_shared_terms(self, method):
        """Adding or subtracting two expressions with a shared term should fold the constants and
        either double or cancel the shared term.

        Tests all combinations of ``x+1, x-1, -x+1, -x-1`` with add/sub, comparing against the
        expected simplified expression.
        """
        # (expression, coefficient of x, constant); the coefficient and constant are used to
        # compute the expected expression.
        terms = [
            (param_x + 1, 1, 1),
            (param_x - 1, 1, -1),
            (-param_x + 1, -1, 1),
            (-param_x - 1, -1, -1),
        ]
        # all ordered pairs, including an expression paired with itself
        for (lhs, left_coeff, left_const), (rhs, right_coeff, right_const) in itertools.product(
            terms, repeat=2
        ):
            with self.subTest(lhs=str(lhs), method=method, rhs=str(rhs)):
                expression = getattr(lhs, method)(rhs)

                # compute expected simplified expression directly
                coeff = getattr(left_coeff, method)(right_coeff)
                const = getattr(left_const, method)(right_const)
                expected = coeff * param_x + const

                self.assertStructurallyEqual(expression, expected)
