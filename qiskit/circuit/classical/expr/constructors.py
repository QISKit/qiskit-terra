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

"""User-space constructor functions for the expression tree, which do some of the inference and
lifting boilerplate work."""


from __future__ import annotations

__all__ = [
    "add",
    "bit_and",
    "bit_not",
    "bit_or",
    "bit_xor",
    "cast",
    "div",
    "equal",
    "greater",
    "greater_equal",
    "index",
    "less",
    "less_equal",
    "lift",
    "lift_legacy_condition",
    "logic_and",
    "logic_not",
    "logic_or",
    "mul",
    "not_equal",
    "shift_left",
    "shift_right",
    "sub",
]

import typing

from .expr import Expr, Var, Value, Unary, Binary, Cast, Index
from ..types import CastKind, cast_kind
from .. import types

if typing.TYPE_CHECKING:
    import qiskit


def _coerce_lossless(expr: Expr, type: types.Type) -> Expr | None:
    """Coerce ``expr`` to ``type`` by inserting a suitable :class:`Cast` node, if the cast is
    lossless.  Otherwise, return ``None``."""
    kind = cast_kind(expr.type, type)
    if kind is CastKind.EQUAL:
        return expr
    if kind is CastKind.IMPLICIT:
        return Cast(expr, type, implicit=True)
    if kind is CastKind.LOSSLESS:
        return Cast(expr, type, implicit=False)
    return None


def lift_legacy_condition(
    condition: tuple[qiskit.circuit.Clbit | qiskit.circuit.ClassicalRegister, int], /
) -> Expr:
    """Lift a legacy two-tuple equality condition into a new-style :class:`Expr`."""
    from qiskit.circuit import Clbit

    target, value = condition
    if isinstance(target, Clbit):
        bool_ = types.Bool()
        return Var(target, bool_) if value else Unary(Unary.Op.LOGIC_NOT, Var(target, bool_), bool_)
    left = Var(target, types.Uint(width=target.size))
    if value.bit_length() > target.size:
        left = Cast(left, types.Uint(width=value.bit_length()), implicit=True)
    right = Value(value, left.type)
    return Binary(Binary.Op.EQUAL, left, right, types.Bool())


def lift(value: typing.Any, /, type: types.Type | None = None) -> Expr:
    """Lift the given Python ``value`` to a :class:`~.expr.Value` or :class:`~.expr.Var`.

    If an explicit ``type`` is given, the typing in the output will reflect that.

    Examples:
        Lifting simple circuit objects to be :class:`~.expr.Var` instances::

            >>> from qiskit.circuit import Clbit, ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.lift(Clbit())
            Var(<clbit>, Bool())
            >>> expr.lift(ClassicalRegister(3, "c"))
            Var(ClassicalRegister(3, "c"), Uint(3))

        The type of the return value can be influenced, if the given value could be interpreted
        losslessly as the given type (use :func:`cast` to perform a full set of casting
        operations, including lossy ones)::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr, types
            >>> expr.lift(ClassicalRegister(3, "c"), types.Uint(5))
            Var(ClassicalRegister(3, "c"), Uint(5))
            >>> expr.lift(5, types.Uint(4))
            Value(5, Uint(4))
    """
    if isinstance(value, Expr):
        if type is not None:
            raise ValueError("use 'cast' to cast existing expressions, not 'lift'")
        return value
    from qiskit.circuit import Clbit, ClassicalRegister, Duration

    inferred: types.Type
    if value is True or value is False or isinstance(value, Clbit):
        inferred = types.Bool()
        constructor = Value if value is True or value is False else Var
    elif isinstance(value, ClassicalRegister):
        inferred = types.Uint(width=value.size)
        constructor = Var
    elif isinstance(value, int):
        if value < 0:
            raise ValueError("cannot represent a negative value")
        inferred = types.Uint(width=value.bit_length() or 1)
        constructor = Value
    elif isinstance(value, float):
        inferred = types.Float()
        constructor = Value
    elif isinstance(value, Duration):
        inferred = types.Duration()
        constructor = Value
    else:
        raise TypeError(f"failed to infer a type for '{value}'")
    if type is None:
        type = inferred
    if types.is_supertype(type, inferred):
        return constructor(value, type)
    raise TypeError(
        f"the explicit type '{type}' is not suitable for representing '{value}';"
        f" it must be non-strict supertype of '{inferred}'"
    )


def cast(operand: typing.Any, type: types.Type, /) -> Expr:
    """Create an explicit cast from the given value to the given type.

    Examples:
        Add an explicit cast node that explicitly casts a higher precision type to a lower precision
        one::

            >>> from qiskit.circuit.classical import expr, types
            >>> value = expr.value(5, types.Uint(32))
            >>> expr.cast(value, types.Uint(8))
            Cast(Value(5, types.Uint(32)), types.Uint(8), implicit=False)
    """
    operand = lift(operand)
    if cast_kind(operand.type, type) is CastKind.NONE:
        raise TypeError(f"cannot cast '{operand}' to '{type}'")
    return Cast(operand, type)


def bit_not(operand: typing.Any, /) -> Expr:
    """Create a bitwise 'not' expression node from the given value, resolving any implicit casts and
    lifting the value into a :class:`Value` node if required.

    Examples:
        Bitwise negation of a :class:`.ClassicalRegister`::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.bit_not(ClassicalRegister(3, "c"))
            Unary(Unary.Op.BIT_NOT, Var(ClassicalRegister(3, 'c'), Uint(3)), Uint(3))
    """
    operand = lift(operand)
    if operand.type.kind not in (types.Bool, types.Uint):
        raise TypeError(f"cannot apply '{Unary.Op.BIT_NOT}' to type '{operand.type}'")
    return Unary(Unary.Op.BIT_NOT, operand, operand.type)


def logic_not(operand: typing.Any, /) -> Expr:
    """Create a logical 'not' expression node from the given value, resolving any implicit casts and
    lifting the value into a :class:`Value` node if required.

    Examples:
        Logical negation of a :class:`.ClassicalRegister`::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.logic_not(ClassicalRegister(3, "c"))
            Unary(\
Unary.Op.LOGIC_NOT, \
Cast(Var(ClassicalRegister(3, 'c'), Uint(3)), \
Bool(), implicit=True), \
Bool())
    """
    operand = lift(operand)
    coerced_operand = _coerce_lossless(operand, types.Bool())
    if coerced_operand is None:
        raise TypeError(f"cannot apply '{Unary.Op.LOGIC_NOT}' to type '{operand.type}'")
    return Unary(Unary.Op.LOGIC_NOT, coerced_operand, coerced_operand.type)


def _lift_binary_operands(left: typing.Any, right: typing.Any) -> tuple[Expr, Expr]:
    """Lift two binary operands simultaneously, inferring the widths of integer literals in either
    position to match the other operand."""
    left_int = isinstance(left, int) and not isinstance(left, bool)
    right_int = isinstance(right, int) and not isinstance(right, bool)
    if not (left_int or right_int):
        left = lift(left)
        right = lift(right)
    elif not right_int:
        right = lift(right)
        if right.type.kind is types.Uint:
            if left.bit_length() > right.type.width:
                raise TypeError(
                    f"integer literal '{left}' is wider than the other operand '{right}'"
                )
            left = Value(left, right.type)
        else:
            left = lift(left)
    elif not left_int:
        left = lift(left)
        if left.type.kind is types.Uint:
            if right.bit_length() > left.type.width:
                raise TypeError(
                    f"integer literal '{right}' is wider than the other operand '{left}'"
                )
            right = Value(right, left.type)
        else:
            right = lift(right)
    else:
        # Both are `int`, so we take our best case to make things work.
        uint = types.Uint(max(left.bit_length(), right.bit_length(), 1))
        left = Value(left, uint)
        right = Value(right, uint)
    return left, right


def _binary_bitwise(op: Binary.Op, left: typing.Any, right: typing.Any) -> Expr:
    left, right = _lift_binary_operands(left, right)
    type: types.Type
    if left.type.kind is right.type.kind is types.Bool:
        type = types.Bool()
    elif left.type.kind is types.Uint and right.type.kind is types.Uint:
        if left.type != right.type:
            raise TypeError(
                "binary bitwise operations are defined between unsigned integers of the same width,"
                f" but got {left.type.width} and {right.type.width}."
            )
        type = left.type
    else:
        raise TypeError(f"invalid types for '{op}': '{left.type}' and '{right.type}'")
    return Binary(op, left, right, type)


def bit_and(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a bitwise 'and' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Bitwise 'and' of a classical register and an integer literal::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.bit_and(ClassicalRegister(3, "c"), 0b111)
            Binary(\
Binary.Op.BIT_AND, \
Var(ClassicalRegister(3, 'c'), Uint(3)), \
Value(7, Uint(3)), \
Uint(3))
        """
    return _binary_bitwise(Binary.Op.BIT_AND, left, right)


def bit_or(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a bitwise 'or' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Bitwise 'or' of a classical register and an integer literal::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.bit_or(ClassicalRegister(3, "c"), 0b101)
            Binary(\
Binary.Op.BIT_OR, \
Var(ClassicalRegister(3, 'c'), Uint(3)), \
Value(5, Uint(3)), \
Uint(3))
    """
    return _binary_bitwise(Binary.Op.BIT_OR, left, right)


def bit_xor(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a bitwise 'exclusive or' expression node from the given value, resolving any implicit
    casts and lifting the values into :class:`Value` nodes if required.

    Examples:
        Bitwise 'exclusive or' of a classical register and an integer literal::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.bit_xor(ClassicalRegister(3, "c"), 0b101)
            Binary(\
Binary.Op.BIT_XOR, \
Var(ClassicalRegister(3, 'c'), Uint(3)), \
Value(5, Uint(3)), \
Uint(3))
    """
    return _binary_bitwise(Binary.Op.BIT_XOR, left, right)


def _binary_logical(op: Binary.Op, left: typing.Any, right: typing.Any) -> Expr:
    bool_ = types.Bool()
    left = lift(left)
    right = lift(right)
    coerced_left = _coerce_lossless(left, bool_)
    coerced_right = _coerce_lossless(right, bool_)
    if coerced_left is None or coerced_right is None:
        raise TypeError(f"invalid types for '{op}': '{left.type}' and '{right.type}'")
    return Binary(op, coerced_left, coerced_right, bool_)


def logic_and(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a logical 'and' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Logical 'and' of two classical bits::

            >>> from qiskit.circuit import Clbit
            >>> from qiskit.circuit.classical import expr
            >>> expr.logic_and(Clbit(), Clbit())
            Binary(Binary.Op.LOGIC_AND, Var(<clbit 0>, Bool()), Var(<clbit 1>, Bool()), Bool())
    """
    return _binary_logical(Binary.Op.LOGIC_AND, left, right)


def logic_or(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a logical 'or' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Logical 'or' of two classical bits

            >>> from qiskit.circuit import Clbit
            >>> from qiskit.circuit.classical import expr
            >>> expr.logic_or(Clbit(), Clbit())
            Binary(Binary.Op.LOGIC_OR, Var(<clbit 0>, Bool()), Var(<clbit 1>, Bool()), Bool())
    """
    return _binary_logical(Binary.Op.LOGIC_OR, left, right)


def _equal_like(op: Binary.Op, left: typing.Any, right: typing.Any) -> Expr:
    left, right = _lift_binary_operands(left, right)
    if left.type.kind is not right.type.kind:
        raise TypeError(f"invalid types for '{op}': '{left.type}' and '{right.type}'")
    type = types.greater(left.type, right.type)
    # Note that we don't check the return value of _coerce_lossless for these
    # since 'left' and 'right' are guaranteed to be the same kind here.
    return Binary(op, _coerce_lossless(left, type), _coerce_lossless(right, type), types.Bool())


def equal(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create an 'equal' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Equality between a classical register and an integer::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.equal(ClassicalRegister(3, "c"), 7)
            Binary(Binary.Op.EQUAL, \
Var(ClassicalRegister(3, "c"), Uint(3)), \
Value(7, Uint(3)), \
Uint(3))
    """
    return _equal_like(Binary.Op.EQUAL, left, right)


def not_equal(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a 'not equal' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Inequality between a classical register and an integer::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.not_equal(ClassicalRegister(3, "c"), 7)
            Binary(Binary.Op.NOT_EQUAL, \
Var(ClassicalRegister(3, "c"), Uint(3)), \
Value(7, Uint(3)), \
Uint(3))
    """
    return _equal_like(Binary.Op.NOT_EQUAL, left, right)


def _binary_relation(op: Binary.Op, left: typing.Any, right: typing.Any) -> Expr:
    left, right = _lift_binary_operands(left, right)
    if left.type.kind is not right.type.kind or left.type.kind is types.Bool:
        raise TypeError(f"invalid types for '{op}': '{left.type}' and '{right.type}'")
    type = types.greater(left.type, right.type)
    # Note that we don't check the return value of _coerce_lossless for these
    # since 'left' and 'right' are guaranteed to be the same kind here.
    return Binary(op, _coerce_lossless(left, type), _coerce_lossless(right, type), types.Bool())


def less(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a 'less than' expression node from the given value, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Query if a classical register is less than an integer::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.less(ClassicalRegister(3, "c"), 5)
            Binary(Binary.Op.LESS, \
Var(ClassicalRegister(3, "c"), Uint(3)), \
Value(5, Uint(3)), \
Uint(3))
    """
    return _binary_relation(Binary.Op.LESS, left, right)


def less_equal(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a 'less than or equal to' expression node from the given value, resolving any implicit
    casts and lifting the values into :class:`Value` nodes if required.

    Examples:
        Query if a classical register is less than or equal to another::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.less_equal(ClassicalRegister(3, "a"), ClassicalRegister(3, "b"))
            Binary(Binary.Op.LESS_EQUAL, \
Var(ClassicalRegister(3, "a"), Uint(3)), \
Var(ClassicalRegister(3, "b"), Uint(3)), \
Uint(3))
    """
    return _binary_relation(Binary.Op.LESS_EQUAL, left, right)


def greater(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a 'greater than' expression node from the given value, resolving any implicit casts
    and lifting the values into :class:`Value` nodes if required.

    Examples:
        Query if a classical register is greater than an integer::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.greater(ClassicalRegister(3, "c"), 5)
            Binary(Binary.Op.GREATER, \
Var(ClassicalRegister(3, "c"), Uint(3)), \
Value(5, Uint(3)), \
Uint(3))
    """
    return _binary_relation(Binary.Op.GREATER, left, right)


def greater_equal(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a 'greater than or equal to' expression node from the given value, resolving any
    implicit casts and lifting the values into :class:`Value` nodes if required.

    Examples:
        Query if a classical register is greater than or equal to another::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.greater_equal(ClassicalRegister(3, "a"), ClassicalRegister(3, "b"))
            Binary(Binary.Op.GREATER_EQUAL, \
Var(ClassicalRegister(3, "a"), Uint(3)), \
Var(ClassicalRegister(3, "b"), Uint(3)), \
Uint(3))
    """
    return _binary_relation(Binary.Op.GREATER_EQUAL, left, right)


def _shift_like(
    op: Binary.Op, left: typing.Any, right: typing.Any, type: types.Type | None
) -> Expr:
    if type is not None and type.kind is not types.Uint:
        raise TypeError(f"type '{type}' is not a valid bitshift operand type")
    if isinstance(left, Expr):
        if type is not None:
            coerced_left = _coerce_lossless(left, type)
            if coerced_left is None:
                raise TypeError(f"type '{type}' cannot losslessly represent '{left.type}'")
            left = coerced_left
    else:
        left = lift(left, type)
    right = lift(right)
    if left.type.kind != types.Uint or right.type.kind != types.Uint:
        raise TypeError(f"invalid types for '{op}': '{left.type}' and '{right.type}'")
    return Binary(op, left, right, left.type)


def shift_left(left: typing.Any, right: typing.Any, /, type: types.Type | None = None) -> Expr:
    """Create a 'bitshift left' expression node from the given two values, resolving any implicit
    casts and lifting the values into :class:`Value` nodes if required.

    If ``type`` is given, the ``left`` operand will be coerced to it (if possible).

    Examples:
        Shift the value of a standalone variable left by some amount::

            >>> from qiskit.circuit.classical import expr, types
            >>> a = expr.Var.new("a", types.Uint(8))
            >>> expr.shift_left(a, 4)
            Binary(Binary.Op.SHIFT_LEFT, \
Var(<UUID>, Uint(8), name='a'), \
Value(4, Uint(3)), \
Uint(8))

        Shift an integer literal by a variable amount, coercing the type of the literal::

            >>> expr.shift_left(3, a, types.Uint(16))
            Binary(Binary.Op.SHIFT_LEFT, \
Value(3, Uint(16)), \
Var(<UUID>, Uint(8), name='a'), \
Uint(16))
    """
    return _shift_like(Binary.Op.SHIFT_LEFT, left, right, type)


def shift_right(left: typing.Any, right: typing.Any, /, type: types.Type | None = None) -> Expr:
    """Create a 'bitshift right' expression node from the given values, resolving any implicit casts
    and lifting the values into :class:`Value` nodes if required.

    If ``type`` is given, the ``left`` operand will be coerced to it (if possible).

    Examples:
        Shift the value of a classical register right by some amount::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.shift_right(ClassicalRegister(8, "a"), 4)
            Binary(Binary.Op.SHIFT_RIGHT, \
Var(ClassicalRegister(8, "a"), Uint(8)), \
Value(4, Uint(3)), \
Uint(8))
    """
    return _shift_like(Binary.Op.SHIFT_RIGHT, left, right, type)


def index(target: typing.Any, index: typing.Any, /) -> Expr:
    """Index into the ``target`` with the given integer ``index``, lifting the values into
    :class:`Value` nodes if required.

    This can be used as the target of a :class:`.Store`, if the ``target`` is itself an lvalue.

    Examples:
        Index into a classical register with a literal::

            >>> from qiskit.circuit import ClassicalRegister
            >>> from qiskit.circuit.classical import expr
            >>> expr.index(ClassicalRegister(8, "a"), 3)
            Index(Var(ClassicalRegister(8, "a"), Uint(8)), Value(3, Uint(2)), Bool())
    """
    target, index = lift(target), lift(index)
    if target.type.kind is not types.Uint or index.type.kind is not types.Uint:
        raise TypeError(f"invalid types for indexing: '{target.type}' and '{index.type}'")
    return Index(target, index, types.Bool())


def _binary_sum(op: Binary.Op, left: typing.Any, right: typing.Any) -> Expr:
    left, right = _lift_binary_operands(left, right)
    if left.type.kind is right.type.kind and left.type.kind in {
        types.Uint,
        types.Float,
        types.Duration,
    }:
        type = types.greater(left.type, right.type)
        return Binary(
            op,
            _coerce_lossless(left, type),
            _coerce_lossless(right, type),
            type,
        )
    raise TypeError(f"invalid types for '{op}': '{left.type}' and '{right.type}'")


def add(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create an addition expression node from the given values, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Addition of two floating point numbers::

            >>> from qiskit.circuit.classical import expr
            >>> expr.add(5.0, 2.0)
            Binary(\
Binary.Op.ADD, \
Value(5.0, Float()), \
Value(2.0, Float()), \
Float())

        Addition of two durations::

            >>> from qiskit.circuit import Duration
            >>> from qiskit.circuit.classical import expr
            >>> expr.add(Duration.dt(1000), Duration.dt(1000))
            Binary(\
Binary.Op.ADD, \
Value(Duration.dt(1000), Duration()), \
Value(Duration.dt(1000), Duration()), \
Duration())
        """
    return _binary_sum(Binary.Op.ADD, left, right)


def sub(left: typing.Any, right: typing.Any, /) -> Expr:
    """Create a subtraction expression node from the given values, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    Examples:
        Subtraction of two floating point numbers::

            >>> from qiskit.circuit.classical import expr
            >>> expr.sub(5.0, 2.0)
            Binary(\
Binary.Op.SUB, \
Value(5.0, Float()), \
Value(2.0, Float()), \
Float())

        Subtraction of two durations::

            >>> from qiskit.circuit import Duration
            >>> from qiskit.circuit.classical import expr
            >>> expr.add(Duration.dt(1000), Duration.dt(1000))
            Binary(\
Binary.Op.SUB, \
Value(Duration.dt(1000), Duration()), \
Value(Duration.dt(1000), Duration()), \
Duration())
        """
    return _binary_sum(Binary.Op.SUB, left, right)


def mul(left: typing.Any, right: typing.Any) -> Expr:
    """Create a multiplication expression node from the given values, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    This can be used to multiply numeric operands of the same type kind, or to multiply a duration
    operand by a numeric operand.

    Examples:
        Multiplication of two floating point numbers::

            >>> from qiskit.circuit.classical import expr
            >>> expr.mul(5.0, 2.0)
            Binary(\
Binary.Op.MUL, \
Value(5.0, Float()), \
Value(2.0, Float()), \
Float())

        Multiplication of a duration by a float::

            >>> from qiskit.circuit import Duration
            >>> from qiskit.circuit.classical import expr
            >>> expr.mul(Duration.dt(1000), 0.5)
            Binary(\
Binary.Op.MUL, \
Value(Duration.dt(1000), Duration()), \
Value(0.5, Float()), \
Duration())
    """
    left, right = _lift_binary_operands(left, right)
    type: types.Type
    if left.type.kind is right.type.kind is types.Duration:
        raise TypeError("cannot multiply two durations")
    if left.type.kind is right.type.kind and left.type.kind in {types.Uint, types.Float}:
        type = types.greater(left.type, right.type)
        left = _coerce_lossless(left, type)
        right = _coerce_lossless(right, type)
    elif left.type.kind is types.Duration and right.type.kind in {types.Uint, types.Float}:
        type = left.type
    elif right.type.kind is types.Duration and left.type.kind in {types.Uint, types.Float}:
        type = right.type
    else:
        raise TypeError(f"invalid types for '{Binary.Op.MUL}': '{left.type}' and '{right.type}'")
    return Binary(
        Binary.Op.MUL,
        left,
        right,
        type,
    )


def div(left: typing.Any, right: typing.Any) -> Expr:
    """Create a division expression node from the given values, resolving any implicit casts and
    lifting the values into :class:`Value` nodes if required.

    This can be used to divide numeric operands of the same type kind, to divide a
    :class:`~.types.Duration` operand by a numeric operand, or to divide two
    :class:`~.types.Duration` operands which yields an expression of type
    :class:`~.types.Float`.

    Examples:
        Division of two floating point numbers::

            >>> from qiskit.circuit.classical import expr
            >>> expr.div(5.0, 2.0)
            Binary(\
Binary.Op.DIV, \
Value(5.0, Float()), \
Value(2.0, Float()), \
Float())

        Division of two durations::

            >>> from qiskit.circuit import Duration
            >>> from qiskit.circuit.classical import expr
            >>> expr.div(Duration.dt(10000), Duration.dt(1000))
            Binary(\
Binary.Op.DIV, \
Value(Duration.dt(10000), Duration()), \
Value(Duration.dt(1000), Duration()), \
Float())


        Division of a duration by a float::

            >>> from qiskit.circuit import Duration
            >>> from qiskit.circuit.classical import expr
            >>> expr.div(Duration.dt(10000), 12.0)
            Binary(\
Binary.Op.DIV, \
Value(Duration.dt(10000), Duration()), \
Value(12.0, types.Float()), \
Duration())
    """
    left, right = _lift_binary_operands(left, right)
    type: types.Type
    if left.type.kind is right.type.kind and left.type.kind in {
        types.Duration,
        types.Uint,
        types.Float,
    }:
        if left.type.kind is types.Duration:
            type = types.Float()
        elif types.order(left.type, right.type) is not types.Ordering.NONE:
            type = types.greater(left.type, right.type)
            left = _coerce_lossless(left, type)
            right = _coerce_lossless(right, type)
    elif left.type.kind is types.Duration and right.type.kind in {types.Uint, types.Float}:
        type = left.type
    else:
        raise TypeError(f"invalid types for '{Binary.Op.DIV}': '{left.type}' and '{right.type}'")
    return Binary(
        Binary.Op.DIV,
        left,
        right,
        type,
    )


def negate(operand: typing.Any, /) -> Expr:
    """Negate an expression node from the given value, resolving any implicit casts and
    lifting the value into a :class:`Value` node if required.
    
    Examples:
        Negation of a floating point number::

            >>> from qiskit.circuit.classical import expr
            >>> expr.negate(5.0)
            Unary(\
Unary.Op.NEGATE, \
Value(5.0, Float()), \
Float())

        Negation of a duration::

            >>> from qiskit.circuit import Duration
            >>> from qiskit.circuit.classical import expr
            >>> expr.negate(Duration.dt(1000))
            Unary(\
Unary.Op.NEGATE, \
Value(Duration.dt(1000), Duration()), \
Duration())
    """
    operand = lift(operand)
    if operand.type.kind not in (types.Float, types.Duration):
        raise TypeError(f"cannot apply '{Unary.Op.NEGATE}' to type '{operand.type}'")
    return Unary(Unary.Op.NEGATE, operand, operand.type)
