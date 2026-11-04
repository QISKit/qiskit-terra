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

"""Expression-tree nodes."""


__all__ = [
    "Binary",
    "Cast",
    "Expr",
    "Index",
    "Stretch",
    "Unary",
    "Value",
    "Var",
]

import enum

from qiskit._accelerate.circuit.classical.expr import (
    Expr,
    Var,
    Stretch,
    Value,
    Cast,
    Unary,
    Binary,
    Index,
)


class _UnaryOp(enum.Enum):
    """Enumeration of the opcodes for unary operations.

    The bitwise negation :data:`BIT_NOT` takes a single bit or an unsigned integer of known
    width, and returns a value of the same type.

    The logical negation :data:`LOGIC_NOT` takes an input that is implicitly coerced to a
    Boolean, and returns a Boolean.

    The arithmetic negation :data:`NEGATE` takes an input that is either float or duration and
    returns a value of the same type.
    """

    # If adding opcodes, remember to add helper constructor functions in `constructors.py`.
    # The opcode integers should be considered a public interface; they are used by
    # serialization formats that may transfer data between different versions of Qiskit.
    #
    # !!! YOU MUST ALSO UPDATE the underlying Rust enum if you touch this.
    BIT_NOT = 1
    """Bitwise negation. ``~operand``."""
    LOGIC_NOT = 2
    """Logical negation. ``!operand``."""
    NEGATE = 3
    """Arithmetic negation. ``-operand``."""

    def __str__(self):
        return f"Unary.{super().__str__()}"

    def __repr__(self):
        return f"Unary.{super().__repr__()}"


# Setting these tricks Sphinx into thinking that this enum is actually
# defined as an inner class of the Rust pyclass.
_UnaryOp.__module__ = "qiskit._accelerate.circuit.classical.expr"
_UnaryOp.__name__ = "Op"
_UnaryOp.__qualname__ = "Unary.Op"


class _BinaryOp(enum.Enum):
    """Enumeration of the opcodes for binary operations.

    The bitwise operations :data:`BIT_AND`, :data:`BIT_OR` and :data:`BIT_XOR` apply to two
    operands of the same type, which must be a single bit or an unsigned integer of fixed width.
    The resultant type is the same as the two input types.

    The logical operations :data:`LOGIC_AND` and :data:`LOGIC_OR` first implicitly coerce their
    arguments to Booleans, and then apply the logical operation.  The resultant type is always
    Boolean.

    The binary mathematical relations :data:`EQUAL`, :data:`NOT_EQUAL`, :data:`LESS`,
    :data:`LESS_EQUAL`, :data:`GREATER` and :data:`GREATER_EQUAL` take unsigned integers
    (with an implicit cast to make them the same width), and return a Boolean.

    The bitshift operations :data:`SHIFT_LEFT` and :data:`SHIFT_RIGHT` can take bit-like
    container types (e.g. unsigned integers) as the left operand, and any integer type as the
    right-hand operand.  In all cases, the output bit width is the same as the input, and zeros
    fill in the "exposed" spaces.

    The binary arithmetic operators :data:`ADD`, :data:`SUB`, :data:`MUL`, and :data:`DIV`
    can be applied to two floats or two unsigned integers, which should be made to be of
    the same width during construction via a cast.
    The :data:`ADD`, :data:`SUB`, and :data:`DIV` operators can be applied on two durations
    yielding another duration, or a float in the case of :data:`DIV`. The :data:`MUL` operator
    can also be applied to a duration and a numeric type, yielding another duration. Finally,
    the :data:`DIV` operator can be used to divide a duration by a numeric type, yielding a
    duration.
    """

    # If adding opcodes, remember to add helper constructor functions in `constructors.py`
    # The opcode integers should be considered a public interface; they are used by
    # serialization formats that may transfer data between different versions of Qiskit.
    #
    # !!! YOU MUST ALSO UPDATE the underlying Rust enum if you touch this.
    BIT_AND = 1
    """Bitwise "and". ``lhs & rhs``."""
    BIT_OR = 2
    """Bitwise "or". ``lhs | rhs``."""
    BIT_XOR = 3
    """Bitwise "exclusive or". ``lhs ^ rhs``."""
    LOGIC_AND = 4
    """Logical "and". ``lhs && rhs``."""
    LOGIC_OR = 5
    """Logical "or". ``lhs || rhs``."""
    EQUAL = 6
    """Numeric equality. ``lhs == rhs``."""
    NOT_EQUAL = 7
    """Numeric inequality. ``lhs != rhs``."""
    LESS = 8
    """Numeric less than. ``lhs < rhs``."""
    LESS_EQUAL = 9
    """Numeric less than or equal to. ``lhs <= rhs``"""
    GREATER = 10
    """Numeric greater than. ``lhs > rhs``."""
    GREATER_EQUAL = 11
    """Numeric greater than or equal to. ``lhs >= rhs``."""
    SHIFT_LEFT = 12
    """Zero-padding bitshift to the left.  ``lhs << rhs``."""
    SHIFT_RIGHT = 13
    """Zero-padding bitshift to the right.  ``lhs >> rhs``."""
    ADD = 14
    """Addition. ``lhs + rhs``."""
    SUB = 15
    """Subtraction. ``lhs - rhs``."""
    MUL = 16
    """Multiplication. ``lhs * rhs``."""
    DIV = 17
    """Division. ``lhs / rhs``."""

    def __str__(self):
        return f"Binary.{super().__str__()}"

    def __repr__(self):
        return f"Binary.{super().__repr__()}"


# Setting these tricks Sphinx into thinking that this enum is actually
# defined as an inner class of the Rust pyclass.
_BinaryOp.__module__ = "qiskit._accelerate.circuit.classical.expr"
_BinaryOp.__name__ = "Op"
_BinaryOp.__qualname__ = "Binary.Op"
