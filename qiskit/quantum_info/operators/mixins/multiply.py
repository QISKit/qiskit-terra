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
Mixin for operator scalar multiplication interface.
"""

from abc import ABC, abstractmethod

from typing import Self


class MultiplyMixin(ABC):
    """Abstract Mixin for scalar multiplication.

    This class defines the following operator overloads:

        - ``*`` / ``__rmul__`
        - ``/`` / ``__truediv__``
        - ``__neg__``

    The following abstract methods must be implemented by subclasses
    using this mixin

        - ``_multiply(self, other)``
    """

    def __rmul__(self, other) -> Self:
        return self._multiply(other)

    def __mul__(self, other) -> Self:
        return self._multiply(other)

    def __truediv__(self, other) -> Self:
        return self._multiply(1 / other)

    def __neg__(self) -> Self:
        return self._multiply(-1)

    @abstractmethod
    def _multiply(self, other):
        """Return the CLASS other * self.

        Args:
            other (complex): a complex number.

        Returns:
            CLASS: the CLASS other * self.
        """
