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

"""Node visitor as defined in https://docs.python.org/3/library/ast.html#ast.NodeVisitor
This module is used internally by ``qiskit.synthesis.boolean.BooleanExpression``.
"""

import ast
import _ast


class BooleanExpressionEvalVisitor(ast.NodeVisitor):
    """Node visitor to compute the value of the expression, given the boolean values of the args
    as defined in https://docs.python.org/3/library/ast.html#ast.NodeVisitor"""

    bitops = {
        _ast.BitAnd: lambda values: values[0] and values[1],
        _ast.And: lambda values: values[0] and values[1],
        _ast.BitOr: lambda values: values[0] or values[1],
        _ast.Or: lambda values: values[0] or values[1],
        _ast.BitXor: lambda values: values[0] ^ values[1],
        _ast.Not: lambda values: not values[0],
        _ast.Invert: lambda values: not values[0],
    }

    def __init__(self):
        self.arg_values = {}
        super().__init__()

    def bit_binop(self, op, values):
        """Performs the operation, if it is recognized"""
        op_type = type(op)
        if op_type not in self.bitops:
            raise ValueError(f"Unknown op: {op_type}")
        return self.bitops[op_type](values)

    def visit_BinOp(self, node):
        """Handles ``&``, ``^``, and ``|``."""
        return self.bit_binop(node.op, [self.visit(node.left), self.visit(node.right)])

    def visit_UnaryOp(self, node):
        """Handles ``~``."""
        return self.bit_binop(node.op, [self.visit(node.operand)])

    def visit_Name(self, node):
        """Reduce variable names."""
        if node.id not in self.arg_values:
            raise ValueError(f"Undefined value for {node.id}")
        return self.arg_values[node.id]

    def visit_Module(self, node):
        """Returns the value of the single expression comprising the boolean expression"""
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise ValueError("Incorrectly formatted boolean expression")
        return self.visit(node.body[0])

    def visit_Expr(self, node):
        """Returns the value of the expression"""
        return self.visit(node.value)

    def generic_visit(self, node):
        """Catch all for the unhandled nodes."""
        raise ValueError(f"Unknown node: {type(node)}")


class BooleanExpressionArgsCollectorVisitor(ast.NodeVisitor):
    """Node visitor to collect the name of the args of the expression
    as defined in https://docs.python.org/3/library/ast.html#ast.NodeVisitor"""

    def __init__(self):
        self.args = set()
        self.args_pos = {}
        super().__init__()

    def visit_Name(self, node):
        """Collect arg name."""
        self.args.add(node.id)
        if node.id not in self.args_pos or (
            self.args_pos[node.id] > (node.lineno, node.col_offset)
        ):
            self.args_pos[node.id] = (node.lineno, node.col_offset)

    def get_sorted_args(self):
        """Returns a list of the args, sorted by their appearance locations"""
        return sorted(self.args, key=lambda arg: self.args_pos[arg])
