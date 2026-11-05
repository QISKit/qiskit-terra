# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
ParameterExpression Class to enable creating simple expressions of Parameters.
"""

from __future__ import annotations


from qiskit.utils.optionals import HAS_SYMPY
import qiskit._accelerate.circuit

Parameter = qiskit._accelerate.circuit.Parameter
ParameterExpression = qiskit._accelerate.circuit.ParameterExpression
OpCode = qiskit._accelerate.circuit.OpCode


_OP_CODE_MAP = (
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__pow__",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "exp",
    "log",
    "sign",
    "gradient",
    "conjugate",
    "subs",
    "abs",
    "arctan",
    "__rsub__",
    "__rtruediv__",
    "__rpow__",
)


def op_code_to_method(op_code: OpCode | int) -> str:
    """Return the method name for a given op_code."""
    return _OP_CODE_MAP[int(op_code)]


@HAS_SYMPY.require_in_call
def sympify(expression):
    """Return symbolic expression as a raw Sympy object.

    .. note::

        This is for interoperability only.  Qiskit will not accept or work with raw Sympy or
        Symengine expressions in its parameters, because they do not contain the tracking
        information used in circuit-parameter binding and assignment.

    .. warning::

        This method will pass strings from the expression to sympy which is
        inherently unsafe as sympy heavily relies on Python's :func:`.eval`
        function for the processing of strings. This method is not safe to
        use with untrusted input.
    """
    import sympy

    # if the expression is a plain Parameter, we just need to return the Symbol
    if isinstance(expression, Parameter):
        return sympy.Symbol(expression.name)
    elif expression.is_symbol():
        return sympy.Symbol(expression.parameters.pop().name)

    try:
        value = expression.numeric(strict=False)
        return sympy.Number(value)
    except TypeError as _:
        pass

    # Otherwise we rebuild the expression from the QPY replay. We keep track of expressions
    # in a stack, which will be queried if an lhs or rhs element is None (meaning that it must
    # be applied to the previous expression).
    stack = []
    for inst in expression._qpy_replay:
        for operand in [inst.lhs, inst.rhs]:
            if operand is not None:
                if isinstance(operand, ParameterExpression):
                    stack.append(operand.sympify())
                else:
                    stack.append(operand)

        method_str = op_code_to_method(inst.op)
        # checks if we apply a binary operation, requiring lhs and rhs
        if inst.op in {
            OpCode.ADD,
            OpCode.SUB,
            OpCode.MUL,
            OpCode.DIV,
            OpCode.POW,
            OpCode.RSUB,
            OpCode.RDIV,
            OpCode.RPOW,
        }:
            rhs = stack.pop()
            lhs = stack.pop()

            # Handle reverse operations (RSUB, RDIV, RPOW) by swapping operands
            if inst.op in {OpCode.RSUB, OpCode.RDIV, OpCode.RPOW}:
                # For reverse operations, the Rust code already swapped operands in the replay,
                # so we need to apply the operation with rhs op lhs to get the correct result.
                # For RPOW: rhs ** lhs (Rust already swapped, so we use __pow__ directly)
                # For RDIV: rhs / lhs (construct division directly: 2 / a)
                # For RSUB: rhs - lhs (construct subtraction directly: 2 - a)
                if inst.op == OpCode.RPOW:
                    # Rust already swapped operands, so we use __pow__ directly: rhs ** lhs
                    stack.append(getattr(rhs, "__pow__")(lhs))
                elif inst.op == OpCode.RDIV:
                    # Construct division directly: rhs / lhs (e.g., 2 / a)
                    # Use sympy's division operator to handle the case where
                    # __rtruediv__ returns NotImplemented
                    stack.append(rhs / lhs)
                elif inst.op == OpCode.RSUB:
                    # Construct subtraction directly: rhs - lhs (e.g., 2 - a)
                    # Use sympy's subtraction operator to handle the case where
                    # __rsub__ returns NotImplemented
                    stack.append(rhs - lhs)
            elif (
                not isinstance(lhs, sympy.Basic)
                and isinstance(rhs, sympy.Basic)
                and inst.op in [OpCode.ADD, OpCode.MUL]
            ):
                if inst.op == OpCode.ADD:
                    method_str = "__radd__"
                elif inst.op == OpCode.MUL:
                    method_str = "__rmul__"
                stack.append(getattr(rhs, method_str)(lhs))
            else:
                stack.append(getattr(lhs, method_str)(rhs))

        # these are unary operands, which only require a lhs operand
        else:
            lhs = stack.pop()

            if inst.op == OpCode.ACOS:
                stack.append(getattr(sympy, "acos")(lhs))
            elif inst.op == OpCode.ASIN:
                stack.append(getattr(sympy, "asin")(lhs))
            elif inst.op == OpCode.ATAN:
                stack.append(getattr(sympy, "atan")(lhs))
            elif inst.op == OpCode.ABS:
                stack.append(getattr(sympy, "Abs")(lhs))
            else:
                stack.append(getattr(sympy, method_str)(lhs))

    return stack.pop()


# Redefine the type so external imports get an evaluated reference; Sphinx needs this to understand
# the type hints.
ParameterValueType = ParameterExpression | float
