# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Operator Mixins
"""

from inspect import getdoc

from .group import GroupMixin
from .adjoint import AdjointMixin
from .linear import LinearMixin
from .multiply import MultiplyMixin
from .tolerances import TolerancesMixin


__all__ = [
    "AdjointMixin",
    "GroupMixin",
    "LinearMixin",
    "MultiplyMixin",
    "TolerancesMixin",
    "getdoc",
]


def generate_apidocs(cls):
    """Decorator to format API docstrings for classes using Mixins.

    This runs string replacement on the docstrings of the mixin
    methods to replace the placeholder CLASS with the class
    name `cls.__name__`.

    Args:
        cls (type): The class to format docstrings.

    Returns:
        cls: the original class with updated docstrings.
    """

    def _replace_name(mixin, methods):
        if issubclass(cls, mixin):
            for i in methods:
                meth = getattr(cls, i)
                doc = getdoc(meth)
                if doc is not None:
                    meth.__doc__ = doc.replace("CLASS", cls.__name__)

    _replace_name(GroupMixin, ("tensor", "expand", "compose", "dot", "power"))
    _replace_name(AdjointMixin, ("transpose", "conjugate", "adjoint"))
    _replace_name(MultiplyMixin, ("_multiply",))
    _replace_name(LinearMixin, ("_add",))
    return cls
