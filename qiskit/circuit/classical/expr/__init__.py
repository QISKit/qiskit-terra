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

"""
==================================================
Expressions (:mod:`qiskit.circuit.classical.expr`)
==================================================

The necessary components for building expressions are all exported from the
:mod:`~.qiskit.circuit.classical.expr` namespace within :mod:`qiskit.circuit.classical`, so you can
choose whether to use qualified access (for example :class:`.expr.Value`) or import the names you
need directly and call them without the prefix.

There are two pathways for constructing expressions.  The classes that form :ref:`the
representation of the expression system <circuit-classical-expressions-expr-representation>`
have constructors that perform zero type checking; it is up to the caller to ensure that they
are building valid objects.  For a more user-friendly interface to direct construction, there
are helper functions associated with most of the classes that do type validation and inference.
These are described below, in :ref:`circuit-classical-expressions-expr-construction`.

.. _circuit-classical-expressions-expr-representation:

Representation
==============

The expression system is based on tree representation.  All nodes in the tree are final
(uninheritable) instances of the abstract base class:

.. autoclass:: Expr

These objects are mutable and should not be reused in a different location without a copy.

All :class:`Expr` instances define a boolean :attr:`~Expr.const` attribute, which indicates
whether the expression can be evaluated at compile time. Most expression classes infer this
during construction based on the const-ness of their operands.

The base for dynamic variables is the :class:`Var`, which can be either an arbitrarily typed
real-time variable, or a wrapper around a :class:`.Clbit` or :class:`.ClassicalRegister`.

.. autoclass:: Var
    :members: var, name, new

Similarly, literals used in expressions (such as integers) should be lifted to :class:`Value` nodes
with associated types. A :class:`Value` is always considered a constant expression.

.. autoclass:: Value

Stretch variables for use in duration expressions are represented by the :class:`Stretch` node.

.. autoclass:: Stretch
    :members: var, name, new

The operations traditionally associated with pre-, post- or infix operators in programming are
represented by the :class:`Unary` and :class:`Binary` nodes as appropriate.  These each take an
operation type code, which are exposed as enumerations inside each class as :class:`Unary.Op`
and :class:`Binary.Op` respectively.

.. autoclass:: Unary
    :members: Op
    :member-order: bysource
.. autoclass:: Binary
    :members: Op
    :member-order: bysource

Bit-like types (unsigned integers) can be indexed by integer types, represented by :class:`Index`.
The result is a single bit.  The resulting expression has an associated memory location (and so can
be used as an lvalue for :class:`.Store`, etc) if the target is also an lvalue.

.. autoclass:: Index

When constructing expressions, one must ensure that the types are valid for the operation.
Attempts to construct expressions with invalid types will raise a regular Python ``TypeError``.

Expressions in this system are defined to act only on certain sets of types.  However, values
may be cast to a suitable supertype in order to satisfy the typing requirements.  In these
cases, a node in the expression tree is used to represent the promotion.  In all cases where
operations note that they "implicitly cast" or "coerce" their arguments, the expression tree
must have this node representing the conversion.

.. autoclass:: Cast


.. _circuit-classical-expressions-expr-construction:

Construction
============

Constructing the tree representation directly is verbose and easy to make a mistake with the
typing.  In many cases, much of the typing can be inferred, scalar values can automatically
be promoted to :class:`Value` instances, and any required promotions can be resolved into
suitable :class:`Cast` nodes.

The functions and methods described in this section are a more user-friendly way to build the
expression tree, while staying close to the internal representation.  All these functions will
automatically lift valid Python scalar values into corresponding :class:`Var` or :class:`Value`
objects, and will resolve any required implicit casts on your behalf.  If you want to directly use
some scalar value as an :class:`Expr` node, you can manually :func:`lift` it yourself.

.. autofunction:: lift

Typically you should create memory-owning :class:`Var` instances by using the
:meth:`.QuantumCircuit.add_var` method to declare them in some circuit context, since a
:class:`.QuantumCircuit` will not accept an :class:`Expr` that contains variables that are not
already declared in it, since it needs to know how to allocate the storage and how the variable will
be initialized.  However, should you want to do this manually, you should use the low-level
:meth:`Var.new` call to safely generate a named variable for usage.

You can manually specify casts in cases where the cast is allowed in explicit form, but may be
lossy (such as the cast of a higher precision :class:`~.types.Uint` to a lower precision one).

.. autofunction:: cast

There are helper constructor functions for each of the unary operations.

.. autofunction:: bit_not
.. autofunction:: logic_not
.. autofunction:: negate

Similarly, the binary operations and relations have helper functions defined.

.. autofunction:: bit_and
.. autofunction:: bit_or
.. autofunction:: bit_xor
.. autofunction:: logic_and
.. autofunction:: logic_or
.. autofunction:: equal
.. autofunction:: not_equal
.. autofunction:: less
.. autofunction:: less_equal
.. autofunction:: greater
.. autofunction:: greater_equal
.. autofunction:: shift_left
.. autofunction:: shift_right
.. autofunction:: add
.. autofunction:: sub
.. autofunction:: mul
.. autofunction:: div

You can index into unsigned integers and bit-likes using another unsigned integer of any width.
This includes in storing operations, if the target of the index is writeable.

.. autofunction:: index

Qiskit's legacy method for specifying equality conditions for use in conditionals is to use a
two-tuple of a :class:`.Clbit` or :class:`.ClassicalRegister` and an integer.  This represents an
exact equality condition, and there are no ways to specify any other relations.  The helper function
:func:`lift_legacy_condition` converts this legacy format into the new expression syntax.

.. autofunction:: lift_legacy_condition

Working with the expression tree
================================

A typical consumer of the expression tree wants to recursively walk through the tree, potentially
statefully, acting on each node differently depending on its type.  This is naturally a
double-dispatch problem; the logic of 'what is to be done' is likely stateful and users should be
free to define their own operations, yet each node defines 'what is being acted on'.  We enable this
double dispatch by providing a base visitor class for the expression tree.

.. autoclass:: ExprVisitor
    :members:
    :undoc-members:

Consumers of the expression tree should subclass the visitor, and override the ``visit_*`` methods
that they wish to handle.  Any non-overridden methods will call :meth:`~ExprVisitor.visit_generic`,
which unless overridden will raise a ``RuntimeError`` to ensure that you are aware if new nodes
have been added to the expression tree that you are not yet handling.

For the convenience of simple visitors that only need to inspect the variables in an expression and
not the general structure, the iterator method :func:`iter_vars` is provided.

.. autofunction:: iter_vars

To iterator over all variables including stretch variables, the iterator method
:func:`iter_identifiers` is provided.

.. autofunction:: iter_identifiers

Two expressions can be compared for direct structural equality by using the built-in Python ``==``
operator.  In general, though, one might want to compare two expressions slightly more semantically,
allowing that the :class:`Var` nodes inside them are bound to different memory-location descriptions
between two different circuits.  In this case, one can use :func:`structurally_equivalent` with two
suitable "key" functions to do the comparison.

.. autofunction:: structurally_equivalent

Some expressions have associated memory locations, and others may be purely temporary.
You can use :func:`is_lvalue` to determine whether an expression has an associated memory location.

.. autofunction:: is_lvalue
"""

__all__ = [
    "Binary",
    "Cast",
    "Expr",
    "ExprVisitor",
    "Index",
    "Stretch",
    "Unary",
    "Value",
    "Var",
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
    "is_lvalue",
    "iter_identifiers",
    "iter_vars",
    "less",
    "less_equal",
    "lift",
    "lift_legacy_condition",
    "logic_and",
    "logic_not",
    "logic_or",
    "mul",
    "negate",
    "not_equal",
    "shift_left",
    "shift_right",
    "structurally_equivalent",
    "sub",
]

from .expr import Expr, Var, Value, Cast, Unary, Binary, Index, Stretch
from .visitors import ExprVisitor, iter_vars, iter_identifiers, structurally_equivalent, is_lvalue
from .constructors import (
    lift,
    cast,
    bit_not,
    logic_not,
    bit_and,
    bit_or,
    bit_xor,
    logic_and,
    logic_or,
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
    shift_left,
    shift_right,
    index,
    add,
    sub,
    mul,
    div,
    lift_legacy_condition,
    negate,
)
