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

"""Tests the BooleanExpression related code"""

import unittest
import ast
from os import path
from ddt import ddt, unpack, data

from test import QiskitTestCase
from qiskit import transpile
from qiskit.providers.basic_provider import BasicSimulator

from qiskit.synthesis.boolean.boolean_expression_visitor import (
    BooleanExpressionEvalVisitor,
    BooleanExpressionArgsCollectorVisitor,
)
from qiskit.synthesis.boolean.boolean_expression import BooleanExpression, TruthTable


class TestTruthTable(QiskitTestCase):
    """Test the truth table class"""

    def test_explicit_representation(self):
        """Test truth table on small number of variables"""

        def func(vals):
            x0, x1, x2, x3 = vals
            return ((x0 and x1) or not x2) ^ x3

        table = TruthTable(func, 4)
        self.assertEqual(str(table), "1111000100001110")
        for assignment, assignment_string in [
            ((False, True, True, False), "0110"),
            ((True, True, True, True), "1111"),
            ((True, False, True, False), "1010"),
        ]:
            self.assertEqual(table[assignment], func(assignment))
            self.assertEqual(table[assignment_string], func(assignment))

    def test_implicit_representation(self):
        """Test truth table on a large number of variables"""

        def func(vals):
            return sum(1 for val in vals if val) % 3 == 0

        table = TruthTable(func, 30)
        for assignment_string, expected_result in [
            ("110101111010100110110111111110", True),
            ("010011111100001011001101110011", False),
            ("011001001111011000011001011001", True),
            ("100001100100100010001111101110", False),
            ("010110010010001110110101000100", False),
        ]:
            self.assertEqual(table[assignment_string], expected_result)
            assignment = tuple(val == "1" for val in assignment_string)
            self.assertEqual(table[assignment], expected_result)


@ddt
class TestBooleanExpression(QiskitTestCase):
    """Test boolean expression."""

    @data(
        ("x | x", "1", True),
        ("x & x", "0", False),
        ("(x0 & x1 | ~x2) ^ x4", "0110", False),
        ("xx & xxx | ( ~z ^ zz)", "0111", True),
    )
    @unpack
    def test_evaluate(self, expression, input_bitstring, expected):
        """Test simulate"""
        expression = BooleanExpression(expression)
        result = expression.simulate(input_bitstring)
        self.assertEqual(result, expected)

    @data(
        ("x | x", "01"),
        ("~x", "10"),
        ("x & y", "0001"),
        ("x & ~y", "0100"),
        ("(x0 & x1 | ~x2) ^ x4", "1111000100001110"),
        ("x & y ^ ( ~z1 | z2)", "1110000111101110"),
    )
    @unpack
    def test_truth_table(self, expression_string, truth_table_string):
        """Test the boolean expression's truth table is correctly generated"""
        expression = BooleanExpression(expression_string)
        self.assertEqual(truth_table_string, str(expression.truth_table))

    @data(
        ("x", False),
        ("not x", True),
        ("x & ~x", False),
        ("(x0 & x1 | ~x2) ^ x4", True),
        ("xx & xxx | ( ~z ^ zz)", True),
    )
    @unpack
    def test_synth(self, expression, expected):
        """Test synth"""
        expression = BooleanExpression(expression)
        expr_circ = expression.synth()
        num_qubits = len(expression.args)
        new_creg = expr_circ._create_creg(1, "c")
        expr_circ.add_register(new_creg)
        expr_circ.measure(num_qubits, new_creg)

        backend = BasicSimulator()
        [result] = (
            backend.run(
                transpile(expr_circ, backend),
                shots=1,
                seed_simulator=14,
            )
            .result()
            .get_counts()
            .keys()
        )

        self.assertEqual(bool(int(result)), expected)

    def test_errors(self):
        """Tests correct identification of errors"""
        exp = "x1 & x2 | x3"
        with self.assertRaisesRegex(ValueError, "var_order missing.*x2"):
            BooleanExpression(exp, var_order=["x1", "x3"])

        bool_exp = BooleanExpression(exp)
        with self.assertRaisesRegex(ValueError, "bitstring length differs.*2 != 3"):
            bool_exp.simulate("01")
        with self.assertRaisesRegex(ValueError, "'circuit_type' must be either 'bit' or 'phase'"):
            bool_exp.synth(circuit_type="z_flip")


class TestBooleanExpressionDIMACS(QiskitTestCase):
    """Loading from a cnf file"""

    def normalize_filenames(self, filename):
        """Given a filename, returns the directory in terms of __file__."""
        dirname = path.dirname(__file__)
        return path.join(dirname, filename)

    def test_simple(self):
        """Loads simple_v3_c2.cnf and simulate"""
        filename = self.normalize_filenames("dimacs/simple_v3_c2.cnf")
        simple = BooleanExpression.from_dimacs_file(filename)
        self.assertEqual(simple.num_bits, 3)
        self.assertTrue(simple.simulate("101"))
        self.assertFalse(simple.simulate("001"))

    def test_quinn(self):
        """Loads quinn.cnf and simulate"""
        filename = self.normalize_filenames("dimacs/quinn.cnf")
        simple = BooleanExpression.from_dimacs_file(filename)
        self.assertEqual(simple.num_bits, 16)
        self.assertFalse(simple.simulate("1010101010101010"))

    def test_bad_formatting(self):
        """Tests DIMACS parsing on edge cases"""

        dimacs = """p cnf 2 1
         
        1 2 0"""  # has empty line with whitespace - should ignore it
        exp = BooleanExpression.from_dimacs(dimacs)
        self.assertEqual(str(exp.truth_table), "0111")

        bad_filename = "bad_filename"
        with self.assertRaisesRegex(FileNotFoundError, f"{bad_filename} does not exist"):
            BooleanExpression.from_dimacs_file(bad_filename)


class TestBooleanParse(QiskitTestCase):
    """Tests the boolean expression parsers"""

    def test_args_collection(self):
        """Tests that args are collected based of appearance in the string"""
        exp = "x2 & x1"
        args_collector = BooleanExpressionArgsCollectorVisitor()
        args_collector.visit(ast.parse(exp))
        res = args_collector.get_sorted_args()
        expected_result = ["x2", "x1"]
        self.assertEqual(res, expected_result)

        exp = "(x3 | x1) ^ x2 & (x2 | x3)"  # some vars appear twice
        args_collector = BooleanExpressionArgsCollectorVisitor()
        args_collector.visit(ast.parse(exp))
        res = args_collector.get_sorted_args()
        expected_result = ["x3", "x1", "x2"]
        self.assertEqual(res, expected_result)

    def test_unrecognized_op(self):
        """Boolean expression has an unknown op"""
        exp = "x1 + x2"  # + should not be recognized as it is not a boolean op
        parser = BooleanExpressionEvalVisitor()
        parser.arg_values = {"x1": True, "x2": True}
        with self.assertRaisesRegex(ValueError, "Unknown op.*Add"):
            parser.visit(ast.parse(exp))

    def test_undefined_var(self):
        """Boolean expression has an variable with no known value"""
        exp = "x1 & z2"  # value for z2 is unknown
        parser = BooleanExpressionEvalVisitor()
        parser.arg_values = {"x1": True, "x2": True}
        with self.assertRaisesRegex(ValueError, "Undefined value.*z2"):
            parser.visit(ast.parse(exp))

    def test_incorrect_format(self):
        """Parsed string is not a python expression"""
        exp = "return x5"  # statement, not expression
        parser = BooleanExpressionEvalVisitor()
        with self.assertRaisesRegex(ValueError, "Incorrectly formatted boolean expression"):
            parser.visit(ast.parse(exp))


if __name__ == "__main__":
    unittest.main()
