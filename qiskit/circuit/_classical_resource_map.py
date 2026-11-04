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

"""Shared helper utility for mapping classical resources from one circuit or DAG to another."""

from __future__ import annotations

import typing

from . import Bit, Clbit, ClassicalRegister
from .classical import expr


class VariableMapper(expr.ExprVisitor[expr.Expr]):
    """Stateful helper class that manages the mapping of variables in conditions and expressions.

    This is designed to be used by both :class:`.QuantumCircuit` and :class:`.DAGCircuit` when
    managing operations that need to map classical resources from one circuit to another.

    The general usage is to initialise this at the start of a many-block mapping operation, then
    call its :meth:`map_condition`, :meth:`map_target` or :meth:`map_expr` methods as appropriate,
    which will return the new object that should be used.

    If an ``add_register`` callable is given to the initializer, the mapper will use it to attempt
    to add new aliasing registers to the outer circuit object, if there is not already a suitable
    register for the mapping available in the circuit.  If this parameter is not given, a
    ``ValueError`` will be raised instead.  The given ``add_register`` callable may choose to raise
    its own exception."""

    # We don't want docstrings for the inherited visitor methods, which are self-explanatory and
    # would just be noise.

    __slots__ = ("add_register", "bit_map", "register_map", "target_cregs", "var_map")

    def __init__(
        self,
        target_cregs: typing.Iterable[ClassicalRegister],
        bit_map: typing.Mapping[Bit, Bit],
        var_map: typing.Mapping[expr.Var | expr.Stretch, expr.Var | expr.Stretch] | None = None,
        *,
        add_register: typing.Callable[[ClassicalRegister], None] | None = None,
    ):
        self.target_cregs = tuple(target_cregs)
        self.register_map = {}
        self.bit_map = bit_map
        self.var_map = var_map or {}
        self.add_register = add_register

    def _map_register(self, theirs: ClassicalRegister) -> ClassicalRegister:
        """Map the target's registers to suitable equivalents in the destination, adding an
        extra one if there's no exact match."""
        if (mapped_theirs := self.register_map.get(theirs.name)) is not None:
            return mapped_theirs
        mapped_bits = [self.bit_map[bit] for bit in theirs]
        for ours in self.target_cregs:
            if mapped_bits == list(ours):
                mapped_theirs = ours
                break
        else:
            if self.add_register is None:
                raise ValueError(f"Register '{theirs.name}' has no counterpart in the destination.")
            mapped_theirs = ClassicalRegister(bits=mapped_bits)
            self.add_register(mapped_theirs)
        self.register_map[theirs.name] = mapped_theirs
        return mapped_theirs

    def map_condition(self, condition, /, *, allow_reorder=False):
        """Map the given ``condition`` so that it only references variables in the destination
        circuit (as given to this class on initialization).

        If ``allow_reorder`` is ``True``, then when a legacy condition (the two-tuple form) is made
        on a register that has a counterpart in the destination with all the same (mapped) bits but
        in a different order, then that register will be used and the value suitably modified to
        make the equality condition work.  This is maintaining legacy (tested) behavior of
        :meth:`.DAGCircuit.compose`; nowhere else does this, and in general this would require *far*
        more complex classical rewriting than Terra needs to worry about in the full expression era.
        """
        if condition is None:
            return None
        if isinstance(condition, expr.Expr):
            return self.map_expr(condition)
        target, value = condition
        if isinstance(target, Clbit):
            return (self.bit_map[target], value)
        if not allow_reorder:
            return (self._map_register(target), value)
        # This is maintaining the legacy behavior of `DAGCircuit.compose`.  We don't attempt to
        # speed-up this lookup with a cache, since that would just make the more standard cases more
        # annoying to deal with.
        mapped_bits_order = [self.bit_map[bit] for bit in target]
        mapped_bits_set = set(mapped_bits_order)
        for register in self.target_cregs:
            if mapped_bits_set == set(register):
                mapped_theirs = register
                break
        else:
            if self.add_register is None:
                raise self.exc_type(
                    f"Register '{target.name}' has no counterpart in the destination."
                )
            mapped_theirs = ClassicalRegister(bits=mapped_bits_order)
            self.add_register(mapped_theirs)
        new_order = {bit: i for i, bit in enumerate(mapped_bits_order)}
        value_bits = f"{value:0{len(target)}b}"[::-1]  # Little-index-indexed binary bitstring.
        mapped_value = int("".join(value_bits[new_order[bit]] for bit in mapped_theirs)[::-1], 2)
        return (mapped_theirs, mapped_value)

    def map_target(self, target, /):
        """Map the real-time variables in a ``target`` of a :class:`.SwitchCaseOp` to the new
        circuit, as defined in the ``circuit`` argument of the initializer of this class."""
        if isinstance(target, Clbit):
            return self.bit_map[target]
        if isinstance(target, ClassicalRegister):
            return self._map_register(target)
        return self.map_expr(target)

    def map_expr(self, node: expr.Expr, /) -> expr.Expr:
        """Map the variables in an :class:`~.expr.Expr` node to the new circuit."""
        return node.accept(self)

    def visit_var(self, node, /):
        if isinstance(node.var, Clbit):
            return expr.Var(self.bit_map[node.var], node.type)
        if isinstance(node.var, ClassicalRegister):
            return expr.Var(self._map_register(node.var), node.type)
        return self.var_map.get(node, node)

    def visit_stretch(self, node, /):
        return self.var_map.get(node, node)

    def visit_value(self, node, /):
        return expr.Value(node.value, node.type)

    def visit_unary(self, node, /):
        return expr.Unary(node.op, node.operand.accept(self), node.type)

    def visit_binary(self, node, /):
        return expr.Binary(node.op, node.left.accept(self), node.right.accept(self), node.type)

    def visit_cast(self, node, /):
        return expr.Cast(node.operand.accept(self), node.type, implicit=node.implicit)

    def visit_index(self, node, /):
        return expr.Index(node.target.accept(self), node.index.accept(self), node.type)
