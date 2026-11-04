# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""SI unit utilities"""
from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from qiskit.circuit.parameterexpression import ParameterExpression


def apply_prefix(value: float | ParameterExpression, unit: str) -> float | ParameterExpression:
    """
    Given a SI unit prefix and value, apply the prefix to convert to
    standard SI unit.

    Args:
        value: The number to apply prefix to.
        unit: String prefix.

    Returns:
        Converted value.

    .. note::

        This may induce tiny value error due to internal representation of float object.
        See https://docs.python.org/3/tutorial/floatingpoint.html for details.

    Raises:
        ValueError: If the ``unit`` isn't recognized.
    """
    prefactors = {
        "f": -15,
        "p": -12,
        "n": -9,
        "u": -6,
        "µ": -6,
        "m": -3,
        "k": 3,
        "M": 6,
        "G": 9,
        "T": 12,
        "P": 15,
    }

    if not unit or len(unit) == 1:
        # for example, "m" can represent meter
        return value

    if unit[0] not in prefactors:
        raise ValueError(f"Could not understand unit: {unit}")

    pow10 = prefactors[unit[0]]

    # to avoid round-off error of prefactor
    if pow10 < 0:
        return value / pow(10, -pow10)

    return value * pow(10, pow10)


def detach_prefix(value: float, decimal: int | None = None) -> tuple[float, str]:
    """
    Given a SI unit value, find the most suitable prefix to scale the value.

    For example, the ``value = 1.3e8`` will be converted into a tuple of ``(130.0, "M")``,
    which represents a scaled value and auxiliary unit that may be used to display the value.
    In above example, that value might be displayed as ``130 MHz`` (unit is arbitrary here).

    Example:

        >>> value, prefix = detach_prefix(1e4)
        >>> print(f"{value} {prefix}Hz")
        10 kHz

    Args:
        value: The number to find prefix.
        decimal: Optional. An arbitrary integer number to represent a precision of the value.
            If specified, it tries to round the mantissa and adjust the prefix to rounded value.
            For example, 999_999.91 will become 999.9999 k with ``decimal=4``,
            while 1.0 M with ``decimal=3`` or less.

    Returns:
        A tuple of scaled value and prefix.

    .. note::

        This may induce tiny value error due to internal representation of float object.
        See https://docs.python.org/3/tutorial/floatingpoint.html for details.

    Raises:
        ValueError: If the ``value`` is out of range.
        ValueError: If the ``value`` is not real number.
    """
    prefactors = {
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "µ",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
    }

    if not np.isreal(value):
        raise ValueError(f"Input should be real number. Cannot convert {value}.")

    if np.abs(value) != 0:
        pow10 = int(np.floor(np.log10(np.abs(value)) / 3) * 3)
    else:
        pow10 = 0

    # to avoid round-off error of prefactor
    if pow10 > 0:
        mant = value / pow(10, pow10)
    else:
        mant = value * pow(10, -pow10)

    if decimal is not None:
        # Corner case handling
        # For example, 999_999.99 can be rounded to 1000.0 k rather than 1.0 M.
        mant = np.round(mant, decimal)
        if mant >= 1000:
            mant /= 1000
            pow10 += 3

    if pow10 not in prefactors:
        raise ValueError(f"Value is out of range: {value}")

    return mant, prefactors[pow10]
