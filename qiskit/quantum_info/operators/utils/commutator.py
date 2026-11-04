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

"""Commutator function."""

from __future__ import annotations
from typing import TypeVar

from qiskit.quantum_info.operators.linear_op import LinearOp

OperatorTypeT = TypeVar("OperatorTypeT", bound=LinearOp)


def commutator(a: OperatorTypeT, b: OperatorTypeT) -> OperatorTypeT:
    r"""Compute commutator of a and b.

    .. math::

        ab - ba.

    Args:
        a: Operator a.
        b: Operator b.
    Returns:
        The commutator
    """
    return a @ b - b @ a
