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

"""Tools for working with the partial ordering of the type system."""

from __future__ import annotations

__all__ = [
    "CastKind",
    "Ordering",
    "cast_kind",
    "greater",
    "is_subtype",
    "is_supertype",
    "order",
]

import enum

from .types import Type, Bool, Duration, Float, Uint


# While the type system is simple, it's overkill to represent the complete partial ordering graph of
# the set of types in an explicit graph form.  The strategy here is to assume that two types have no
# ordering between them, and an ordering is defined by putting a function `Type * Type -> Ordering`
# into the `_ORDERERS`.


class Ordering(enum.Enum):
    """Enumeration listing the possible relations between two types.  Types only have a partial
    ordering, so it's possible for two types to have no sub-typing relationship.

    Note that the sub-/supertyping relationship is not the same as whether a type can be explicitly
    cast from one to another."""

    LESS = enum.auto()
    """The left type is a strict subtype of the right type."""
    EQUAL = enum.auto()
    """The two types are equal."""
    GREATER = enum.auto()
    """The left type is a strict supertype of the right type."""
    NONE = enum.auto()
    """There is no typing relationship between the two types."""

    def __repr__(self):
        return str(self)


def _order_uint_uint(left: Uint, right: Uint, /) -> Ordering:
    if left.width < right.width:
        return Ordering.LESS
    if left.width == right.width:
        return Ordering.EQUAL
    return Ordering.GREATER


_ORDERERS = {
    (Bool, Bool): lambda _a, _b, /: Ordering.EQUAL,
    (Uint, Uint): _order_uint_uint,
    (Float, Float): lambda _a, _b, /: Ordering.EQUAL,
    (Duration, Duration): lambda _a, _b, /: Ordering.EQUAL,
}


def order(left: Type, right: Type, /) -> Ordering:
    """Get the ordering relationship between the two types as an enumeration value.

    Examples:
        Compare two :class:`Uint` types of different widths::

            >>> from qiskit.circuit.classical import types
            >>> types.order(types.Uint(8), types.Uint(16))
            Ordering.LESS

        Compare two types that have no ordering between them::

            >>> types.order(types.Uint(8), types.Bool())
            Ordering.NONE
    """
    if (orderer := _ORDERERS.get((left.kind, right.kind))) is None:
        return Ordering.NONE
    return orderer(left, right)


def is_subtype(left: Type, right: Type, /, strict: bool = False) -> bool:
    r"""Does the relation :math:`\text{left} \le \text{right}` hold?  If there is no ordering
    relation between the two types, then this returns ``False``.  If ``strict``, then the equality
    is also forbidden.

    Examples:
        Check if one type is a subclass of another::

            >>> from qiskit.circuit.classical import types
            >>> types.is_subtype(types.Uint(8), types.Uint(16))
            True

        Check if one type is a strict subclass of another::

            >>> types.is_subtype(types.Bool(), types.Bool())
            True
            >>> types.is_subtype(types.Bool(), types.Bool(), strict=True)
            False
    """
    order_ = order(left, right)
    return order_ is Ordering.LESS or (not strict and order_ is Ordering.EQUAL)


def is_supertype(left: Type, right: Type, /, strict: bool = False) -> bool:
    r"""Does the relation :math:`\text{left} \ge \text{right}` hold?  If there is no ordering
    relation between the two types, then this returns ``False``.  If ``strict``, then the equality
    is also forbidden.

    Examples:
        Check if one type is a superclass of another::

            >>> from qiskit.circuit.classical import types
            >>> types.is_supertype(types.Uint(8), types.Uint(16))
            False

        Check if one type is a strict superclass of another::

            >>> types.is_supertype(types.Bool(), types.Bool())
            True
            >>> types.is_supertype(types.Bool(), types.Bool(), strict=True)
            False
    """
    order_ = order(left, right)
    return order_ is Ordering.GREATER or (not strict and order_ is Ordering.EQUAL)


def greater(left: Type, right: Type, /) -> Type:
    """Get the greater of the two types, assuming that there is an ordering relation between them.
    Technically, this is a slightly restricted version of the concept of the 'meet' of the two
    types in that the return value must be one of the inputs. In practice in the type system there
    is no concept of a 'sum' type, so the 'meet' exists if and only if there is an ordering between
    the two types, and is equal to the greater of the two types.

    Returns:
        The greater of the two types.

    Raises:
        TypeError: if there is no ordering relation between the two types.

    Examples:
        Find the greater of two :class:`Uint` types::

            >>> from qiskit.circuit.classical import types
            >>> types.greater(types.Uint(8), types.Uint(16))
            types.Uint(16)
    """
    order_ = order(left, right)
    if order_ is Ordering.NONE:
        raise TypeError(f"no ordering exists between '{left}' and '{right}'")
    return left if order_ is Ordering.GREATER else right


class CastKind(enum.Enum):
    """A return value indicating the type of cast that can occur from one type to another."""

    EQUAL = enum.auto()
    """The two types are equal; no cast node is required at all."""
    IMPLICIT = enum.auto()
    """The 'from' type can be cast to the 'to' type implicitly.  A :class:`~.expr.Cast` node with
    ``implicit==True`` is the minimum required to specify this."""
    LOSSLESS = enum.auto()
    """The 'from' type can be cast to the 'to' type explicitly, and the cast will be lossless.  This
    requires a :class:`~.expr.Cast` node with ``implicit=False``, but there's no danger from
    inserting one."""
    DANGEROUS = enum.auto()
    """The 'from' type has a defined cast to the 'to' type, but depending on the value, it may lose
    data.  A user would need to manually specify casts."""
    NONE = enum.auto()
    """There is no casting permitted from the 'from' type to the 'to' type."""


def _uint_cast(from_: Uint, to_: Uint, /) -> CastKind:
    if from_.width == to_.width:
        return CastKind.EQUAL
    if from_.width < to_.width:
        return CastKind.LOSSLESS
    return CastKind.DANGEROUS


_ALLOWED_CASTS = {
    (Bool, Bool): lambda _a, _b, /: CastKind.EQUAL,
    (Bool, Uint): lambda _a, _b, /: CastKind.LOSSLESS,
    (Bool, Float): lambda _a, _b, /: CastKind.LOSSLESS,
    (Uint, Bool): lambda _a, _b, /: CastKind.IMPLICIT,
    (Uint, Uint): _uint_cast,
    (Uint, Float): lambda _a, _b, /: CastKind.DANGEROUS,
    (Float, Float): lambda _a, _b, /: CastKind.EQUAL,
    (Float, Uint): lambda _a, _b, /: CastKind.DANGEROUS,
    (Float, Bool): lambda _a, _b, /: CastKind.DANGEROUS,
    (Duration, Duration): lambda _a, _b, /: CastKind.EQUAL,
}


def cast_kind(from_: Type, to_: Type, /) -> CastKind:
    """Determine the sort of cast that is required to move from the left type to the right type.

    Examples:

        .. plot::
           :include-source:
           :nofigs:


            >>> from qiskit.circuit.classical import types
            >>> types.cast_kind(types.Bool(), types.Bool())
            <CastKind.EQUAL: 1>
            >>> types.cast_kind(types.Uint(8), types.Bool())
            <CastKind.IMPLICIT: 2>
            >>> types.cast_kind(types.Bool(), types.Uint(8))
            <CastKind.LOSSLESS: 3>
            >>> types.cast_kind(types.Uint(16), types.Uint(8))
            <CastKind.DANGEROUS: 4>
    """
    if (coercer := _ALLOWED_CASTS.get((from_.kind, to_.kind))) is None:
        return CastKind.NONE
    return coercer(from_, to_)
