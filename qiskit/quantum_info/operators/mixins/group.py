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
Mixin for gate operator interface.
"""

from abc import ABC, abstractmethod
from numbers import Integral

from qiskit.exceptions import QiskitError

from typing import Self


class GroupMixin(ABC):
    """Abstract Mixin for operator group operations.

    This class defines the following methods

        - :meth:`compose`
        - :meth:`dot`
        - :meth:`tensor`
        - :meth:`expand`
        - :meth:`power`

    And the following operator overloads:

        - ``&``, ``__and__`` -> :meth:`compose`
        - ``@``, ``__matmul__`` -> :meth:`dot`
        - ``^``, ``__xor__`` -> :meth:`tensor`
        - ``**``, ``__pow__`` -> :meth:`power`

    The following abstract methods must be implemented by subclasses
    using this mixin

        - ``compose(self, other, qargs=None, inplace=False)``
        - ``tensor(self, other)``
        - ``expand(self, other)``
    """

    def __and__(self, other) -> Self:
        return self.compose(other)

    def __pow__(self, n) -> Self:
        return self.power(n)

    def __xor__(self, other) -> Self:
        return self.tensor(other)

    def __matmul__(self, other) -> Self:
        # Allow Statevector @ Operator to call Statevector.evolve
        if other.__class__.__name__ == "Statevector":
            return other.evolve(self)
        return self.dot(other)

    @abstractmethod
    def tensor(self, other) -> Self:
        r"""Return the tensor product with another CLASS.

        Args:
            other (CLASS): a CLASS object.

        Returns:
            CLASS: the tensor product :math:`a \otimes b`, where :math:`a`
                is the current CLASS, and :math:`b` is the other CLASS.

        .. note::
            The tensor product can be obtained using the ``^`` binary operator.
            Hence ``a.tensor(b)`` is equivalent to ``a ^ b``.

        .. note:
            Tensor uses reversed operator ordering to :meth:`expand`.
            For two operators of the same type ``a.tensor(b) = b.expand(a)``.
        """

    @abstractmethod
    def expand(self, other) -> Self:
        r"""Return the reverse-order tensor product with another CLASS.

        Args:
            other (CLASS): a CLASS object.

        Returns:
            CLASS: the tensor product :math:`b \otimes a`, where :math:`a`
                is the current CLASS, and :math:`b` is the other CLASS.

        .. note:
            Expand is the opposite operator ordering to :meth:`tensor`.
            For two operators of the same type ``a.expand(b) = b.tensor(a)``.
        """

    @abstractmethod
    def compose(self, other, qargs=None, front=False) -> Self:
        """Return the operator composition with another CLASS.

        Args:
            other (CLASS): a CLASS object.
            qargs (list or None):  a list of subsystem positions to
                                  apply other on. If None apply on all
                                  subsystems (default: None).
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            CLASS: The composed CLASS.

        Raises:
            QiskitError: if other cannot be converted to an operator, or has
                         incompatible dimensions for specified subsystems.

        .. note::
            Composition (``&``) by default is defined as `left` matrix multiplication for
            matrix operators, while ``@`` (equivalent to :meth:`dot`) is defined as `right` matrix
            multiplication. That is that ``A & B == A.compose(B)`` is equivalent to
            ``B @ A == B.dot(A)`` when ``A`` and ``B`` are of the same type.

            Setting the ``front=True`` kwarg changes this to `right` matrix
            multiplication and is equivalent to the :meth:`dot` method
            ``A.dot(B) == A.compose(B, front=True)``.
        """

    def dot(self, other, qargs=None) -> Self:
        """Return the right multiplied operator self * other.

        Args:
            other (CLASS): an operator object.
            qargs (list or None):  a list of subsystem positions to
                                  apply other on. If None apply on all
                                  subsystems (default: None).

        Returns:
            CLASS: The right matrix multiplied CLASS.

        .. note::
            The dot product can be obtained using the ``@`` binary operator.
            Hence ``a.dot(b)`` is equivalent to ``a @ b``.
        """
        return self.compose(other, qargs=qargs, front=True)

    def power(self, n) -> Self:
        """Return the composition of an operator with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).

        Returns:
            CLASS: the n-times composed operator.

        Raises:
            QiskitError: if the input and output dimensions of the operator
                         are not equal, or the power is not a positive integer.
        """
        # NOTE: if a subclass can have negative or non-integer powers
        # this method should be overridden in that class.
        if not isinstance(n, Integral) or n < 1:
            raise QiskitError("Can only power with positive integer powers.")
        ret = self
        for _ in range(1, n):
            ret = ret.dot(self)
        return ret
