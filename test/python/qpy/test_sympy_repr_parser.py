# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the qpy sympy repr parser"""

from ddt import ddt, idata, unpack

from qiskit.circuit import Parameter, ParameterExpression
from qiskit.qpy import QpyError
from qiskit.qpy.binary_io.parse_sympy_repr import parse_sympy_repr

from test import QiskitTestCase

TEST_PARAMS = [
    ("Add(Symbol('a'), Symbol('b'))", lambda: Parameter("a") + Parameter("b")),
    ("Sub(Symbol('a'), Symbol('b'))", lambda: Parameter("a") - Parameter("b")),
    ("Mul(Symbol('a'), Symbol('b'))", lambda: Parameter("a") * Parameter("b")),
    ("Div(Symbol('a'), Symbol('b'))", lambda: Parameter("a") / Parameter("b")),
    ("Pow(Symbol('a'), Symbol('b'))", lambda: Parameter("a") ** Parameter("b")),
    ("Add(Symbol('a'), 3.12345)", lambda: Parameter("a") + 3.12345),
    ("Sub(Symbol('a'), 3.12345)", lambda: Parameter("a") - 3.12345),
    ("Mul(Symbol('a'), 3.12345)", lambda: Parameter("a") * 3.12345),
    ("Div(Symbol('a'), 3.12345)", lambda: Parameter("a") / 3.12345),
    ("Add(3.12345, Symbol('a'))", lambda: 3.12345 + Parameter("a")),
    ("Sub(3.12345, Symbol('a'))", lambda: 3.12345 - Parameter("a")),
    ("Mul(3.12345, Symbol('a'))", lambda: 3.12345 * Parameter("a")),
    ("Div(3.12345, Symbol('a'))", lambda: 3.12345 / Parameter("a")),
    ("Pow(Symbol('a'), 3.1234)", lambda: Parameter("a") ** 3.1234),
    ("Add(Symbol('a'), sin(Symbol('b')))", lambda: Parameter("a") + Parameter("b").sin()),
    ("Sub(Symbol('a'), sin(Symbol('b')))", lambda: Parameter("a") - Parameter("b").sin()),
    ("Mul(Symbol('a'), sin(Symbol('b')))", lambda: Parameter("a") * Parameter("b").sin()),
    ("Div(Symbol('a'), sin(Symbol('b')))", lambda: Parameter("a") / Parameter("b").sin()),
    ("Pow(Symbol('a'), sin(Symbol('b')))", lambda: Parameter("a") ** Parameter("b").sin()),
    ("Add(Symbol('a'), cos(Symbol('b')))", lambda: Parameter("a") + Parameter("b").cos()),
    ("Sub(Symbol('a'), cos(Symbol('b')))", lambda: Parameter("a") - Parameter("b").cos()),
    ("Mul(Symbol('a'), cos(Symbol('b')))", lambda: Parameter("a") * Parameter("b").cos()),
    ("Div(Symbol('a'), cos(Symbol('b')))", lambda: Parameter("a") / Parameter("b").cos()),
    ("Pow(Symbol('a'), cos(Symbol('b')))", lambda: Parameter("a") ** Parameter("b").cos()),
    ("Add(Symbol('a'), tan(Symbol('b')))", lambda: Parameter("a") + Parameter("b").tan()),
    ("Sub(Symbol('a'), tan(Symbol('b')))", lambda: Parameter("a") - Parameter("b").tan()),
    ("Mul(Symbol('a'), tan(Symbol('b')))", lambda: Parameter("a") * Parameter("b").tan()),
    ("Div(Symbol('a'), tan(Symbol('b')))", lambda: Parameter("a") / Parameter("b").tan()),
    ("Pow(Symbol('a'), tan(Symbol('b')))", lambda: Parameter("a") ** Parameter("b").tan()),
    ("Add(Symbol('a'), log(Symbol('b')))", lambda: Parameter("a") + Parameter("b").log()),
    ("Sub(Symbol('a'), log(Symbol('b')))", lambda: Parameter("a") - Parameter("b").log()),
    ("Mul(Symbol('a'), log(Symbol('b')))", lambda: Parameter("a") * Parameter("b").log()),
    ("Div(Symbol('a'), log(Symbol('b')))", lambda: Parameter("a") / Parameter("b").log()),
    ("Pow(Symbol('a'), log(Symbol('b')))", lambda: Parameter("a") ** Parameter("b").log()),
    ("Add(Symbol('a'), asin(Symbol('b')))", lambda: Parameter("a") + Parameter("b").arcsin()),
    ("Sub(Symbol('a'), asin(Symbol('b')))", lambda: Parameter("a") - Parameter("b").arcsin()),
    ("Mul(Symbol('a'), asin(Symbol('b')))", lambda: Parameter("a") * Parameter("b").arcsin()),
    ("Div(Symbol('a'), asin(Symbol('b')))", lambda: Parameter("a") / Parameter("b").arcsin()),
    ("Pow(Symbol('a'), asin(Symbol('b')))", lambda: Parameter("a") ** Parameter("b").arcsin()),
    ("Add(Symbol('a'), acos(Symbol('b')))", lambda: Parameter("a") + Parameter("b").arccos()),
    ("Sub(Symbol('a'), acos(Symbol('b')))", lambda: Parameter("a") - Parameter("b").arccos()),
    ("Mul(Symbol('a'), acos(Symbol('b')))", lambda: Parameter("a") * Parameter("b").arccos()),
    ("Div(Symbol('a'), acos(Symbol('b')))", lambda: Parameter("a") / Parameter("b").arccos()),
    ("Pow(Symbol('a'), acos(Symbol('b')))", lambda: Parameter("a") ** Parameter("b").arccos()),
    ("Add(Symbol('a'), atan(Symbol('b')))", lambda: Parameter("a") + Parameter("b").arctan()),
    ("Sub(Symbol('a'), atan(Symbol('b')))", lambda: Parameter("a") - Parameter("b").arctan()),
    ("Mul(Symbol('a'), atan(Symbol('b')))", lambda: Parameter("a") * Parameter("b").arctan()),
    ("Div(Symbol('a'), atan(Symbol('b')))", lambda: Parameter("a") / Parameter("b").arctan()),
    ("Pow(Symbol('a'), atan(Symbol('b')))", lambda: Parameter("a") ** Parameter("b").arctan()),
    ("Add(Symbol('a'), exp(Symbol('b')))", lambda: Parameter("a") + Parameter("b").exp()),
    ("Sub(Symbol('a'), exp(Symbol('b')))", lambda: Parameter("a") - Parameter("b").exp()),
    ("Mul(Symbol('a'), exp(Symbol('b')))", lambda: Parameter("a") * Parameter("b").exp()),
    ("Div(Symbol('a'), exp(Symbol('b')))", lambda: Parameter("a") / Parameter("b").exp()),
    ("Pow(Symbol('a'), exp(Symbol('b')))", lambda: Parameter("a") ** Parameter("b").exp()),
    (
        "Add(Symbol('a'), conjugate(Symbol('b')))",
        lambda: Parameter("a") + Parameter("b").conjugate(),
    ),
    (
        "Sub(Symbol('a'), conjugate(Symbol('b')))",
        lambda: Parameter("a") - Parameter("b").conjugate(),
    ),
    (
        "Mul(Symbol('a'), conjugate(Symbol('b')))",
        lambda: Parameter("a") * Parameter("b").conjugate(),
    ),
    (
        "Div(Symbol('a'), conjugate(Symbol('b')))",
        lambda: Parameter("a") / Parameter("b").conjugate(),
    ),
    (
        "Pow(Symbol('a'), conjugate(Symbol('b')))",
        lambda: Parameter("a") ** Parameter("b").conjugate(),
    ),
    ("Complex(3.14)", lambda: 3.14 + 0j),
    ("Float(3)", lambda: float(3)),
    ("Rational(3.14)", lambda: 3.14),
    ("Rational(3.14, 2)", lambda: 3.14 / 2),
    ("Integer(3.14)", lambda: 3),
    (
        "Abs(Add(Symbol('a'), conjugate(Symbol('b'))))",
        (Parameter("a") + Parameter("b").conjugate()).abs,
    ),
    (
        "Abs(Sub(Symbol('a'), conjugate(Symbol('b'))))",
        (Parameter("a") - Parameter("b").conjugate()).abs,
    ),
    (
        "Abs(Mul(Symbol('a'), conjugate(Symbol('b'))))",
        (Parameter("a") * Parameter("b").conjugate()).abs,
    ),
    (
        "Abs(Div(Symbol('a'), conjugate(Symbol('b'))))",
        (Parameter("a") / Parameter("b").conjugate()).abs,
    ),
    (
        "Abs(Pow(Symbol('a'), conjugate(Symbol('b'))))",
        (Parameter("a") ** Parameter("b").conjugate()).abs,
    ),
]


@ddt
class TestQPYRoundtrip(QiskitTestCase):
    """Test QPY's sympy repr parser."""

    @unpack
    @idata(TEST_PARAMS)
    def test_expressions(self, srepr_str, generator_expected):
        expected = generator_expected()
        if isinstance(expected, ParameterExpression):
            name_map = {x.name: x for x in expected.parameters}
        else:
            name_map = {}
        result = parse_sympy_repr(srepr_str, name_map)
        self.assertEqual(result, expected, f"{result} != {expected}")

    def test_invalid_ops(self):
        py_repr_str = "parse_sympy_repr('foo')"
        name_map = {"foo": Parameter("foo")}
        with self.assertRaises(QpyError):
            parse_sympy_repr(py_repr_str, name_map)

    @unpack
    @idata(
        [
            ("Symbol('missing_quote)", SyntaxError),
            ("Rational(a, b, c)", IndexError),
            ("Symbol('open paren'", SyntaxError),
            ("Symbol('missing name map')", QpyError),
            ("Mul()", IndexError),
        ]
    )
    def test_malformed_reprs(self, malformed, expected_exception):
        with self.assertRaises(expected_exception):
            parse_sympy_repr(malformed, {})

    def test_large_expr(self):
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        d = Parameter("d")
        final_expr = (
            a**2
            + ((a**2 * (a + 0.25 * b.sin())).cos() + d.tan() + d.arccos() - d.arcsin() + d.arctan())
            * (-d).exp()
            + (a**2 * (a + 0.25 * b.sin())).log()
            - a.sin()
            - b.conjugate()
        ).abs()
        srepr_str = "Abs(Add(Pow(Symbol('a'), Integer(2)), Mul(Add(cos(Mul(Pow(Symbol('a'), Integer(2)), Add(Symbol('a'), Mul(Rational(1, 4), sin(Symbol('b')))))), tan(Symbol('d')), acos(Symbol('d')), Mul(Integer(-1), asin(Symbol('d'))), atan(Symbol('d'))), exp(Mul(Integer(-1), Symbol('d')))), log(Mul(Pow(Symbol('a'), Integer(2)), Add(Symbol('a'), Mul(Rational(1, 4), sin(Symbol('b')))))), Mul(Integer(-1), sin(Symbol('a'))), Mul(Integer(-1), conjugate(Symbol('b')))))"
        name_map = {"a": a, "b": b, "c": c, "d": d}
        result = parse_sympy_repr(srepr_str, name_map)
        self.assertEqual(result, final_expr, f"{result} != {final_expr}")
