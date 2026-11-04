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

"""The 'Store' operation."""

from __future__ import annotations

import typing

from .exceptions import CircuitError
from .classical import expr, types
from .instruction import Instruction


def _handle_equal_types(lvalue: expr.Expr, rvalue: expr.Expr, /) -> tuple[expr.Expr, expr.Expr]:
    return lvalue, rvalue


def _handle_implicit_cast(lvalue: expr.Expr, rvalue: expr.Expr, /) -> tuple[expr.Expr, expr.Expr]:
    return lvalue, expr.Cast(rvalue, lvalue.type, implicit=True)


def _requires_lossless_cast(lvalue: expr.Expr, rvalue: expr.Expr, /) -> typing.NoReturn:
    raise CircuitError(f"an explicit cast is required from '{rvalue.type}' to '{lvalue.type}'")


def _requires_dangerous_cast(lvalue: expr.Expr, rvalue: expr.Expr, /) -> typing.NoReturn:
    raise CircuitError(
        f"an explicit cast is required from '{rvalue.type}' to '{lvalue.type}', which may be lossy"
    )


def _no_cast_possible(lvalue: expr.Expr, rvalue: expr.Expr) -> typing.NoReturn:
    raise CircuitError(f"no cast is possible from '{rvalue.type}' to '{lvalue.type}'")


_HANDLE_CAST = {
    types.CastKind.EQUAL: _handle_equal_types,
    types.CastKind.IMPLICIT: _handle_implicit_cast,
    types.CastKind.LOSSLESS: _requires_lossless_cast,
    types.CastKind.DANGEROUS: _requires_dangerous_cast,
    types.CastKind.NONE: _no_cast_possible,
}


class Store(Instruction):
    """A manual storage of some classical value to a classical memory location.

    This is a low-level primitive of the classical-expression handling (similar to how
    :class:`~.circuit.Measure` is a primitive for quantum measurement), and is not safe for
    subclassing."""

    # This is a compiler/backend intrinsic operation, separate to any quantum processing.
    _directive = True

    def __init__(self, lvalue: expr.Expr, rvalue: expr.Expr):
        """
        Args:
            lvalue: the memory location being stored into.
            rvalue: the expression result being stored.
        """
        if not expr.is_lvalue(lvalue):
            raise CircuitError(f"'{lvalue}' is not an l-value")

        cast_kind = types.cast_kind(rvalue.type, lvalue.type)
        if (handler := _HANDLE_CAST.get(cast_kind)) is None:
            raise RuntimeError(f"unhandled cast kind required: {cast_kind}")
        lvalue, rvalue = handler(lvalue, rvalue)

        super().__init__("store", 0, 0, [lvalue, rvalue])

    @property
    def lvalue(self):
        """Get the l-value :class:`~.expr.Expr` node that is being stored to."""
        return self.params[0]

    @property
    def rvalue(self):
        """Get the r-value :class:`~.expr.Expr` node that is being written into the l-value."""
        return self.params[1]
