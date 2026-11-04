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

"""Double commutator function."""

from __future__ import annotations
from typing import TypeVar

from qiskit.quantum_info.operators.linear_op import LinearOp

OperatorTypeT = TypeVar("OperatorTypeT", bound=LinearOp)


def double_commutator(
    a: OperatorTypeT, b: OperatorTypeT, c: OperatorTypeT, *, commutator: bool = True
) -> OperatorTypeT:
    r"""Compute symmetric double commutator of a, b and c.

    See also Equation (13.6.18) in [1].

    If `commutator` is `True`, it returns

    .. math::

         [[A, B], C]/2 + [A, [B, C]]/2
         = (2ABC + 2CBA - BAC - CAB - ACB - BCA)/2.

    If `commutator` is `False`, it returns

    .. math::
         \lbrace[A, B], C\rbrace/2 + \lbrace A, [B, C]\rbrace/2
         = (2ABC - 2CBA - BAC + CAB - ACB + BCA)/2.

    Args:
        a: Operator a.
        b: Operator b.
        c: Operator c.
        commutator: If ``True`` compute the double commutator,
            if ``False`` the double anti-commutator.

    Returns:
        The double commutator

    References:

        [1]: R. McWeeny.
            Methods of Molecular Quantum Mechanics.
            2nd Edition, Academic Press, 1992.
            ISBN 0-12-486552-6.
    """
    sign_num = -1 if commutator else 1

    ab = a @ b
    ba = b @ a
    ac = a @ c
    ca = c @ a

    abc = ab @ c
    cba = c @ ba
    bac = ba @ c
    cab = c @ ab
    acb = ac @ b
    bca = b @ ca

    res = abc - sign_num * cba + 0.5 * (-bac + sign_num * cab - acb + sign_num * bca)

    return res
