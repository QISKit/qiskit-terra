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

"""Expression visitors."""

from __future__ import annotations

__all__ = [
    "ExprVisitor",
    "is_lvalue",
    "iter_identifiers",
    "iter_vars",
    "structurally_equivalent",
]

import typing

from . import expr

_T_co = typing.TypeVar("_T_co", covariant=True)


class ExprVisitor(typing.Generic[_T_co]):
    """Base class for visitors to the :class:`Expr` tree.  Subclasses should override whichever of
    the ``visit_*`` methods that they are able to handle, and should be organized such that
    non-existent methods will never be called."""

    # The method names are self-explanatory and docstrings would just be noise.

    __slots__ = ()

    def visit_generic(self, node: expr.Expr, /) -> _T_co:  # pragma: no cover
        raise RuntimeError(f"expression visitor {self} has no method to handle expr {node}")

    def visit_var(self, node: expr.Var, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)

    def visit_stretch(self, node: expr.Stretch, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)

    def visit_value(self, node: expr.Value, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)

    def visit_unary(self, node: expr.Unary, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)

    def visit_binary(self, node: expr.Binary, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)

    def visit_cast(self, node: expr.Cast, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)

    def visit_index(self, node: expr.Index, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)

    def visit_range(self, node: expr.Range, /) -> _T_co:  # pragma: no cover
        return self.visit_generic(node)


class _VarWalkerImpl(ExprVisitor[typing.Iterable[expr.Var]]):
    # We don't want docstrings for the inherited visitor methods, which are self-explanatory and
    # would just be noise.

    __slots__ = ()

    def visit_var(self, node, /):
        yield node

    def visit_stretch(self, node, /):

        yield from ()

    def visit_value(self, node, /):

        yield from ()

    def visit_unary(self, node, /):
        yield from node.operand.accept(self)

    def visit_binary(self, node, /):
        yield from node.left.accept(self)
        yield from node.right.accept(self)

    def visit_cast(self, node, /):
        yield from node.operand.accept(self)

    def visit_index(self, node, /):
        yield from node.target.accept(self)
        yield from node.index.accept(self)

    def visit_range(self, node, /):
        yield from node.start.accept(self)
        yield from node.stop.accept(self)
        if node.step is not None:
            yield from node.step.accept(self)


class _IdentWalkerImpl(ExprVisitor[typing.Iterable[expr.Var | expr.Stretch]]):
    __slots__ = ()

    def visit_var(self, node, /):
        yield node

    def visit_stretch(self, node, /):
        yield node

    def visit_value(self, node, /):
        yield from ()

    def visit_unary(self, node, /):
        yield from node.operand.accept(self)

    def visit_binary(self, node, /):
        yield from node.left.accept(self)
        yield from node.right.accept(self)

    def visit_cast(self, node, /):
        yield from node.operand.accept(self)

    def visit_index(self, node, /):
        yield from node.target.accept(self)
        yield from node.index.accept(self)

    def visit_range(self, node, /):
        yield from node.start.accept(self)
        yield from node.stop.accept(self)
        if node.step is not None:
            yield from node.step.accept(self)


_VAR_WALKER = _VarWalkerImpl()
_IDENT_WALKER = _IdentWalkerImpl()


def iter_vars(node: expr.Expr) -> typing.Iterator[expr.Var]:
    """Get an iterator over the :class:`~.expr.Var` nodes referenced at any level in the given
    :class:`~.expr.Expr`.

    Examples:
        Print out the name of each :class:`.ClassicalRegister` encountered::

            from qiskit.circuit import ClassicalRegister
            from qiskit.circuit.classical import expr

            cr1 = ClassicalRegister(3, "a")
            cr2 = ClassicalRegister(3, "b")

            for node in expr.iter_vars(expr.bit_and(expr.bit_not(cr1), cr2)):
                if isinstance(node.var, ClassicalRegister):
                    print(node.var.name)

    .. seealso::
        :func:`iter_identifiers`
            Get an iterator over all identifier nodes in the expression, including
            both :class:`~.expr.Var` and :class:`~.expr.Stretch` nodes.
    """
    yield from node.accept(_VAR_WALKER)


def iter_identifiers(node: expr.Expr) -> typing.Iterator[expr.Var | expr.Stretch]:
    """Get an iterator over the :class:`~.expr.Var` and :class:`~.expr.Stretch`
    nodes referenced at any level in the given :class:`~.expr.Expr`.

    Examples:
        Print out the name of each :class:`.ClassicalRegister` encountered::

            from qiskit.circuit import ClassicalRegister
            from qiskit.circuit.classical import expr

            cr1 = ClassicalRegister(3, "a")
            cr2 = ClassicalRegister(3, "b")

            for node in expr.iter_vars(expr.bit_and(expr.bit_not(cr1), cr2)):
                if isinstance(node.var, ClassicalRegister):
                    print(node.var.name)

    .. seealso::
        :func:`iter_vars`
            Get an iterator over just the :class:`~.expr.Var` nodes in the expression.
    """
    yield from node.accept(_IDENT_WALKER)


class _StructuralEquivalenceImpl(ExprVisitor[bool]):
    # The strategy here is to continue to do regular double dispatch through the visitor format,
    # since we simply exit out with a ``False`` as soon as the structure of the two trees isn't the
    # same; we never need to do any sort of "triple" dispatch.  To recurse through both trees
    # simultaneously, we hold a pointer to the "full" expression of the other (at the given depth)
    # in the stack variables of each visit function, and pass the next "deeper" pointer via the
    # `other` state in the class instance.

    __slots__ = (
        "other",
        "other_key",
        "self_key",
    )

    def __init__(self, other: expr.Expr, self_key, other_key):
        self.self_key = self_key
        self.other_key = other_key
        self.other = other

    def visit_var(self, node, /):
        if self.other.__class__ is not node.__class__ or self.other.type != node.type:
            return False
        if self.self_key is None or (self_var := self.self_key(node.var)) is None:
            self_var = node.var
        if self.other_key is None or (other_var := self.other_key(self.other.var)) is None:
            other_var = self.other.var
        return self_var == other_var

    def visit_stretch(self, node, /):
        if self.other.__class__ is not node.__class__:
            return False
        return node.var == self.other.var

    def visit_value(self, node, /):
        return (
            node.__class__ is self.other.__class__
            and node.type == self.other.type
            and node.value == self.other.value
        )

    def visit_unary(self, node, /):
        if (
            self.other.__class__ is not node.__class__
            or self.other.op is not node.op
            or self.other.type != node.type
        ):
            return False
        self.other = self.other.operand
        return node.operand.accept(self)

    def visit_binary(self, node, /):
        if (
            self.other.__class__ is not node.__class__
            or self.other.op is not node.op
            or self.other.type != node.type
        ):
            return False
        other = self.other
        self.other = other.left
        if not node.left.accept(self):
            return False
        self.other = other.right
        return node.right.accept(self)

    def visit_cast(self, node, /):
        if self.other.__class__ is not node.__class__ or self.other.type != node.type:
            return False
        self.other = self.other.operand
        return node.operand.accept(self)

    def visit_index(self, node, /):
        if self.other.__class__ is not node.__class__ or self.other.type != node.type:
            return False
        other = self.other
        self.other = other.target
        if not node.target.accept(self):
            return False
        self.other = other.index
        return node.index.accept(self)


def structurally_equivalent(
    left: expr.Expr,
    right: expr.Expr,
    left_var_key: typing.Callable[[typing.Any], typing.Any] | None = None,
    right_var_key: typing.Callable[[typing.Any], typing.Any] | None = None,
) -> bool:
    """Do these two expressions have exactly the same tree structure, up to some key function for
    the :class:`~.expr.Var` objects?

    In other words, are these two expressions the exact same trees, except we compare the
    :attr:`.Var.var` fields by calling the appropriate ``*_var_key`` function on them, and comparing
    that output for equality.  This function does not allow any semantic "equivalences" such as
    asserting that ``a == b`` is equivalent to ``b == a``; the evaluation order of the operands
    could, in general, cause such a statement to be false (consider hypothetical ``extern``
    functions that access global state).

    There's no requirements on the key functions, except that their outputs should have general
    ``__eq__`` methods.  If a key function returns ``None``, the variable will be used verbatim
    instead.

    Args:
        left: one of the :class:`~.expr.Expr` nodes.
        right: the other :class:`~.expr.Expr` node.
        left_var_key: a callable whose output should be used when comparing :attr:`.Var.var`
            attributes.  If this argument is ``None`` or its output is ``None`` for a given
            variable in ``left``, the variable will be used verbatim.
        right_var_key: same as ``left_var_key``, but used on the variables in ``right`` instead.

    Examples:
        Comparing two expressions for structural equivalence, with no remapping of the variables.
        These are different because the different :class:`.Clbit` instances compare differently::

            >>> from qiskit.circuit import Clbit
            >>> from qiskit.circuit.classical import expr
            >>> left_bits = [Clbit(), Clbit()]
            >>> right_bits = [Clbit(), Clbit()]
            >>> left = expr.logic_and(expr.logic_not(left_bits[0]), left_bits[1])
            >>> right = expr.logic_and(expr.logic_not(right_bits[0]), right_bits[1])
            >>> expr.structurally_equivalent(left, right)
            False

        Comparing the same two expressions, but this time using mapping functions that associate
        the bits with simple indices::

            >>> left_key = {var: i for i, var in enumerate(left_bits)}.get
            >>> right_key = {var: i for i, var in enumerate(right_bits)}.get
            >>> expr.structurally_equivalent(left, right, left_key, right_key)
            True
    """
    return left.accept(_StructuralEquivalenceImpl(right, left_var_key, right_var_key))


class _IsLValueImpl(ExprVisitor[bool]):
    __slots__ = ()

    def visit_var(self, node, /):
        return True

    def visit_stretch(self, node, /):
        return False

    def visit_value(self, node, /):
        return False

    def visit_unary(self, node, /):
        return False

    def visit_binary(self, node, /):
        return False

    def visit_cast(self, node, /):
        return False

    def visit_index(self, node, /):
        return node.target.accept(self)


_IS_LVALUE = _IsLValueImpl()


def is_lvalue(node: expr.Expr, /) -> bool:
    """Return whether this expression can be used in l-value positions, that is, whether it has a
    well-defined location in memory, such as one that might be writeable.

    Being an l-value is necessary but not sufficient for this location to be writeable; it is
    permissible that a larger object containing this memory location may not allow writing from
    the scope that attempts to write to it.  This would be an access property of the containing
    program, however, and not an inherent property of the expression system.

    A constant expression is never an lvalue.

    Examples:
        Literal values are never l-values; there's no memory location associated with (for example)
        the constant ``1``::

            >>> from qiskit.circuit.classical import expr
            >>> expr.is_lvalue(expr.lift(2))
            False

        :class:`~.expr.Var` nodes are always l-values, because they always have some associated
        memory location::

            >>> from qiskit.circuit.classical import types
            >>> from qiskit.circuit import Clbit
            >>> expr.is_lvalue(expr.Var.new("a", types.Bool()))
            True
            >>> expr.is_lvalue(expr.lift(Clbit()))
            True

        Currently there are no unary or binary operations on variables that can produce an l-value
        expression, but it is likely in the future that some sort of "indexing" operation will be
        added, which could produce l-values::

            >>> a = expr.Var.new("a", types.Uint(8))
            >>> b = expr.Var.new("b", types.Uint(8))
            >>> expr.is_lvalue(a) and expr.is_lvalue(b)
            True
            >>> expr.is_lvalue(expr.bit_and(a, b))
            False
    """
    return node.accept(_IS_LVALUE)
