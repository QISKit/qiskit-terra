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
==============================================
Typing (:mod:`qiskit.circuit.classical.types`)
==============================================

Representation
==============

The type system of the expression tree is exposed through this module.  This is inherently linked to
the expression system in the :mod:`~.classical.expr` module, as most expressions can only be
understood with the context of the types that they act on.

All types inherit from an abstract base class:

.. autoclass:: Type

Types should be considered immutable objects, and you must not mutate them.  It is permissible to
reuse a :class:`Type` that you take from another object without copying it, and generally this will
be the best approach for performance.  :class:`Type` objects are designed to be small amounts of
data, and it's best to point to the same instance of the data where possible rather than
heap-allocating a new version of the same thing.  Where possible, the class constructors will return
singleton instances to facilitate this.

The :class:`Bool` type represents :class:`.Clbit` and the literals ``True`` and ``False``, the
:class:`Uint` type represents :class:`.ClassicalRegister` and Python integers, the :class:`Float`
type represents Python floats, and the :class:`Duration` type represents a duration for use in
timing-aware circuit operations.

.. autoclass:: Bool
.. autoclass:: Uint
.. autoclass:: Float
.. autoclass:: Duration

Working with types
==================

There are some additional functions on these types documented in the subsequent sections.
These are mostly expected to be used only in manipulations of the expression tree;
users who are building expressions using the
:ref:`user-facing construction interface <circuit-classical-expressions-expr-construction>` should
not need to use these.

Partial ordering of types
=========================

The type system is equipped with a partial ordering, where :math:`a < b` is interpreted as
":math:`a` is a strict subtype of :math:`b`".  Note that the partial ordering is a subset of the
directed graph that describes the allowed explicit casting operations between types.  The partial
ordering defines when one type may be lossless directly interpreted as another.

The low-level interface to querying the subtyping relationship is the :func:`order` function.

.. autofunction:: order

The return value is an enumeration :class:`Ordering` that describes what, if any, subtyping
relationship exists between the two types.

.. autoclass:: Ordering
    :member-order: bysource

Some helper methods are then defined in terms of this low-level :func:`order` primitive:

.. autofunction:: is_subtype
.. autofunction:: is_supertype
.. autofunction:: greater


Casting between types
=====================

It is common to need to cast values of one type to another type.  The casting rules for this are
embedded into the :mod:`types` module.  You can query the casting kinds using :func:`cast_kind`:

.. autofunction:: cast_kind

The return values from this function are an enumeration explaining the types of cast that are
allowed from the left type to the right type.

Note that casts between :class:`Float` and :class:`Uint` are considered dangerous in either
direction, and must be done explicitly.

.. autoclass:: CastKind
"""

__all__ = [
    "Bool",
    "CastKind",
    "Duration",
    "Float",
    "Ordering",
    "Type",
    "Uint",
    "cast_kind",
    "greater",
    "is_subtype",
    "is_supertype",
    "order",
]

from .types import Type, Bool, Duration, Float, Uint
from .ordering import Ordering, order, is_subtype, is_supertype, greater, CastKind, cast_kind
