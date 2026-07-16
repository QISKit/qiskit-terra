# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Check if number close to values of PI"""

from fractions import Fraction

import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.exceptions import QiskitError

MAX_FRAC = 99
POW_LIST = np.pi ** np.arange(2, 5)


def _format_raw(single_inpt, output, ndigits):
    if output == "qasm":
        return f"{single_inpt:#}" if ndigits is None else f"{single_inpt:#.{ndigits}g}"
    return f"{single_inpt}" if ndigits is None else f"{single_inpt:.{ndigits}g}"


def _format_pi_multiple(numer, denom, pi, output, neg_str):
    """Format (numer * pi) / denom, omitting unit coefficients."""
    if denom == 1:
        if numer == 1:
            return f"{neg_str}{pi}"
        if output == "qasm":
            return f"{neg_str}{numer}*{pi}"
        return f"{neg_str}{numer}{pi}"
    if output == "latex":
        if numer == 1:
            return f"\\frac{{{neg_str}{pi}}}{{{denom}}}"
        return f"\\frac{{{neg_str}{numer}{pi}}}{{{denom}}}"
    if output == "qasm":
        if numer == 1:
            return f"{neg_str}{pi}/{denom}"
        return f"{neg_str}{numer}*{pi}/{denom}"
    if numer == 1:
        return f"{neg_str}{pi}/{denom}"
    return f"{neg_str}{numer}{pi}/{denom}"


def _format_pi_divisor(numer, denom, pi, output, neg_str):
    """Format numer / (denom * pi), omitting unit coefficients."""
    denom_str = "" if denom == 1 and output != "qasm" else str(denom)
    if output == "latex":
        return f"\\frac{{{neg_str}{numer}}}{{{denom_str}{pi}}}"
    if output == "qasm":
        return f"{neg_str}{numer}/({denom}*{pi})"
    return f"{neg_str}{numer}/{denom_str}{pi}"


def _match_fraction(val, eps, limit):
    """Return n/d close to val with denominator at most limit, else None."""
    frac = Fraction(val).limit_denominator(limit)
    if frac.numerator == 0 or frac.numerator > limit:
        return None
    if abs(val - frac) < eps:
        return frac
    return None


def pi_check(inpt, eps=1e-9, output="text", ndigits=None):
    """Computes if a number is close to an integer
    fraction or multiple of PI and returns the
    corresponding string.

    Args:
        inpt (float): Number to check.
        eps (float): EPS to check against.
        output (str): Options are 'text' (default),
                      'latex', 'mpl', and 'qasm'.
        ndigits (int or None): Number of digits to print
                               if returning raw inpt.
                               If `None` (default), Python's
                               default float formatting is used.

    Returns:
        str: string representation of output.

    Raises:
        QiskitError: if output is not a valid option.
    """
    if isinstance(inpt, ParameterExpression):
        param_str = str(inpt)
        values = inpt._values()
        for val in values:
            pi = pi_check(abs(float(val)), eps=eps, output=output, ndigits=ndigits)
            try:
                _ = float(pi)
            except (ValueError, TypeError):
                import qiskit._accelerate.circuit

                # we need to match the precise string representation of the pi-value,
                # therefore we use _Value instead of just str(abs(val))
                sym_str = str(qiskit._accelerate.circuit.ParameterExpression._Value(abs(val)))
                param_str = param_str.replace(sym_str, pi)
        return param_str
    elif isinstance(inpt, str):
        return inpt

    def normalize(single_inpt):
        if abs(single_inpt) < eps:
            return "0"

        if output == "text":
            pi = "π"
        elif output == "qasm":
            pi = "pi"
        elif output == "latex":
            pi = "\\pi"
        elif output == "mpl":
            pi = "$\\pi$"
        else:
            raise QiskitError("pi_check parameter output should be text, latex, mpl, or qasm.")

        neg_str = "-" if single_inpt < 0 else ""
        abs_inpt = abs(single_inpt)

        if abs_inpt > np.pi:
            power = np.where(abs(abs_inpt - POW_LIST) < eps)
            if power[0].shape[0]:
                if output == "qasm":
                    return _format_raw(single_inpt, output, ndigits)
                if output == "latex":
                    return f"{neg_str}{pi}^{power[0][0] + 2}"
                if output == "mpl":
                    return f"{neg_str}{pi}$^{power[0][0] + 2}$"
                return f"{neg_str}{pi}**{power[0][0] + 2}"

        frac = _match_fraction(abs_inpt / np.pi, eps, MAX_FRAC)
        if frac is not None:
            return _format_pi_multiple(frac.numerator, frac.denominator, pi, output, neg_str)

        frac = _match_fraction(abs_inpt * np.pi, eps, MAX_FRAC)
        if frac is not None:
            return _format_pi_divisor(frac.numerator, frac.denominator, pi, output, neg_str)

        return _format_raw(single_inpt, output, ndigits)

    complex_inpt = complex(inpt)
    real, imag = map(normalize, [complex_inpt.real, complex_inpt.imag])

    jstr = "\\jmath" if output == "latex" else "j"
    if real == "0" and imag != "0":
        str_out = imag + jstr
    elif real != "0" and imag != "0":
        op_str = "+"
        # Remove + if imag negative except for latex fractions
        if complex_inpt.imag < 0 and (output != "latex" or "\\frac" not in imag):
            op_str = ""
        str_out = f"{real}{op_str}{imag}{jstr}"
    else:
        str_out = real
    return str_out
