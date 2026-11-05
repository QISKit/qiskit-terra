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
Mixin for linear operator interface.
"""

from abc import ABC, abstractmethod

from .multiply import MultiplyMixin

from typing import Self


class LinearMixin(MultiplyMixin, ABC):
    """Abstract Mixin for linear operator.

    This class defines the following operator overloads:

        - ``+`` / ``__add__``
        - ``-`` / ``__sub__``
        - ``*`` / ``__rmul__`
        - ``/`` / ``__truediv__``
        - ``__neg__``

    The following abstract methods must be implemented by subclasses
    using this mixin

        - ``_add(self, other, qargs=None)``
        - ``_multiply(self, other)``
    """

    def __add__(self, other) -> Self:
        # enable easy use of sum(...)
        if not isinstance(other, type(self)) and other == 0:
            return self

        qargs = getattr(other, "qargs", None)
        return self._add(other, qargs=qargs)

    def __radd__(self, other) -> Self:
        # enable easy use of sum(...)
        if not isinstance(other, type(self)) and other == 0:
            return self

        qargs = getattr(other, "qargs", None)
        return self._add(other, qargs=qargs)

    def __sub__(self, other) -> Self:
        qargs = getattr(other, "qargs", None)
        return self._add(-other, qargs=qargs)

    def __rsub__(self, other) -> Self:
        qargs = getattr(other, "qargs", None)
        return (-self)._add(other, qargs=qargs)

    @abstractmethod
    def _add(self, other, qargs=None):
        """Return the CLASS self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (CLASS): an operator object.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            CLASS: the CLASS self + other.
        """
