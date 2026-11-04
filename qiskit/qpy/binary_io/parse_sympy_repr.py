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

"""Parser for sympy expressions srepr from ParameterExpression internals."""

import ast
import operator

from qiskit.qpy.exceptions import QpyError
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression


ALLOWED_CALLERS = {
    "Abs",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Symbol",
    "Integer",
    "Rational",
    "Complex",
    "Float",
    "log",
    "sin",
    "cos",
    "tan",
    "atan",
    "acos",
    "asin",
    "exp",
    "conjugate",
}

METHOD_MAPPING = {
    "log": "log",
    "Abs": "abs",
    "sin": "sin",
    "tan": "tan",
    "cos": "cos",
    "atan": "arctan",
    "acos": "arccos",
    "asin": "arcsin",
    "exp": "exp",
    "conjugate": "conjugate",
}

FUNCTION_MAPPING = {
    "Integer": int,
    "Complex": complex,
    "Float": float,
}

OPERATOR_FUNCTIONS = {
    "Add": operator.add,
    "Sub": operator.sub,
    "Mul": operator.mul,
    "Div": operator.truediv,
    "Pow": operator.pow,
}

UNARY = {
    "sin",
    "cos",
    "tan",
    "atan",
    "acos",
    "asin",
    "conjugate",
    "exp",
    "log",
    "Symbol",
    "Integer",
    "Complex",
    "Abs",
    "Float",
}


class ParseSympyWalker(ast.NodeVisitor):
    """A custom ast walker that is passed the sympy srepr from QPY < 13 and creates a custom
    expression."""

    def __init__(self, name_map: dict[str, Parameter]):
        self.stack = []
        self.name_map = name_map

    def visit_UnaryOp(self, node: ast.UnaryOp):
        """Visit a python unary op node"""
        self.visit(node.operand)
        arg = self.stack.pop()
        if isinstance(node.op, ast.UAdd):
            self.stack.append(+arg)
        elif isinstance(node.op, ast.USub):
            self.stack.append(-arg)
        elif isinstance(node.op, ast.Not):
            self.stack.append(not arg)
        elif isinstance(node.op, ast.Invert):
            self.stack.append(~arg)
        else:
            raise QpyError(f"Invalid unary op as part of sympy srepr: {node.op}")

    def visit_Constant(self, node: ast.Constant):
        """Visit a constant node."""
        self.stack.append(node.value)

    def visit_Call(self, node: ast.Call):
        """Visit a call node

        This can only be parameter expression allowed sympy call types.
        """

        if isinstance(node.func, ast.Name):
            name = node.func.id
        else:
            raise QpyError(f"Unknown node type: {node.func}")

        if name not in ALLOWED_CALLERS:
            raise QpyError(f"{name} is not part of a valid sympy expression srepr")

        args = node.args
        if name in UNARY:
            if len(args) != 1:
                raise QpyError(f"{name} has an invalid number of args in sympy srepr")
            self.visit(args[0])
            method = METHOD_MAPPING.get(name, None)
            if method is not None:
                obj = getattr(self.stack.pop(), method)()
            elif name == "Symbol":
                name = self.stack.pop()
                obj = self.name_map.get(name, None)
                if obj is None:
                    raise QpyError(f"Parameter: {name} not defined in the QPY payload")
            else:
                function = FUNCTION_MAPPING[name]
                obj = function(self.stack.pop())
            self.stack.append(obj)
        else:
            for arg in args:
                self.visit(arg)
            out_args = [self.stack.pop() for _ in range(len(args))]
            func = OPERATOR_FUNCTIONS.get(name, None)
            if func is not None:
                obj = out_args.pop()
                out_args.reverse()
                for arg in out_args:
                    obj = func(obj, arg)
            elif name == "Rational":
                # If rational has one arg it's a no-op because
                # ParameterExpression doesn't have a Rational type
                if len(out_args) < 2:
                    obj = out_args[0]
                else:
                    lhs = out_args.pop()
                    rhs = out_args.pop()
                    # If there is a 3rd argument that is the GCD which isn't supported by
                    # ParameterExpression
                    if len(out_args) == 1:
                        raise QpyError(
                            "An expression can not contain a Sympy Rational with a GCD set"
                        )
                    elif len(out_args) > 0:
                        raise QpyError(
                            f"Invalid Rational too many arguments Rational({lhs}, {rhs}, *{out_args})"
                        )
                    obj = lhs / rhs
            else:
                function = FUNCTION_MAPPING[name]
                out_args.reverse()
                obj = function(*out_args)
            self.stack.append(obj)


def parse_sympy_repr(sympy_repr: str, name_map: dict[str, Parameter]) -> ParameterExpression:
    """Parse a given sympy srepr into a symbolic expression object."""
    tree = ast.parse(sympy_repr, mode="eval")
    visitor = ParseSympyWalker(name_map)
    visitor.visit(tree)
    return visitor.stack.pop()
