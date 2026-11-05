# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tools to create LaTeX arrays.
"""

import numpy as np

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.utils.optionals import HAS_SYMPY


@HAS_SYMPY.require_in_call("Create a latex representation of a ket expression")
def _num_to_latex(raw_value, decimals=15, first_term=True, coefficient=False):
    """Convert a complex number to latex code suitable for a ket expression

    Args:
        raw_value (complex): Value to convert.
        decimals (int): Number of decimal places to round to (default 15).
        coefficient (bool): Whether the number is to be used as a coefficient
                            of a ket.
        first_term (bool): If a coefficient, whether this number is the first
                           coefficient in the expression.
    Returns:
        str: latex code
    """
    import sympy  # runtime import

    raw_value = np.around(raw_value, decimals=decimals).item()
    value = sympy.nsimplify(raw_value, rational=False)

    if isinstance(value, sympy.core.numbers.Rational) and value.denominator > 50:
        # Avoid showing ugly fractions (e.g. 50498971964399/62500000000000)
        value = value.evalf()  # Display as float

    if isinstance(value, sympy.core.numbers.Float):
        value = round(value, decimals)

    element = sympy.latex(value, full_prec=False)

    if not coefficient:
        return element

    if isinstance(value, sympy.core.Add):
        # element has two terms
        element = f"({element})"

    if element == "1":
        element = ""

    if element == "-1":
        element = "-"

    if not first_term and not element.startswith("-"):
        element = f"+{element}"

    return element


def _matrix_to_latex(matrix, decimals=10, prefix="", max_size=(8, 8)):
    """Latex representation of a complex numpy array (with maximum dimension 2)

    Args:
        matrix (ndarray): The matrix to be converted to latex, must have dimension 2.
        decimals (int): For numbers not close to integers, the number of decimal places
                         to round to.
        prefix (str): Latex string to be prepended to the latex, intended for labels.
        max_size (list(```int```)): Indexable containing two integers: Maximum width and maximum
                          height of output Latex matrix (including dots characters). If the
                          width and/or height of matrix exceeds the maximum, the centre values
                          will be replaced with dots. Maximum width or height must be greater
                          than 3.

    Returns:
        str: Latex representation of the matrix

    Raises:
        ValueError: If minimum value in max_size < 3
        MissingOptionalLibraryError: If sympy is not installed
    """
    if min(max_size) < 3:
        raise ValueError("""Smallest value in max_size must be greater than or equal to 3""")

    out_string = f"\n{prefix}\n"
    out_string += "\\begin{bmatrix}\n"

    def _elements_to_latex(elements):
        # Takes a list of elements (a row) and creates a latex
        # string from it; Each element separated by `&`
        el_string = ""
        for el in elements:
            num_string = _num_to_latex(el, decimals=decimals)
            el_string += num_string + " & "
        el_string = el_string[:-2]  # remove trailing ampersands
        return el_string

    def _rows_to_latex(rows, max_width):
        # Takes a list of lists (list of 'rows') and creates a
        # latex string from it
        row_string = ""
        for r in rows:
            if len(r) <= max_width:
                row_string += _elements_to_latex(r)
            else:
                row_string += _elements_to_latex(r[: max_width // 2])
                row_string += "& \\cdots & "
                row_string += _elements_to_latex(r[-max_width // 2 + 1 :])
            row_string += " \\\\\n "
        return row_string

    max_width, max_height = max_size
    if matrix.ndim == 1:
        out_string += _rows_to_latex([matrix], max_width)

    elif len(matrix) > max_height:
        # We need to truncate vertically, so we process the rows at the beginning
        # and end, and add a line of vertical ellipse (dots) characters between them
        out_string += _rows_to_latex(matrix[: max_height // 2], max_width)

        if max_width >= matrix.shape[1]:
            out_string += "\\vdots & " * matrix.shape[1]
        else:
            # In this case we need to add the diagonal dots in line with the column
            # of horizontal dots
            pre_vdots = max_width // 2
            post_vdots = max_width // 2 + np.mod(max_width, 2) - 1
            out_string += "\\vdots & " * pre_vdots
            out_string += "\\ddots & "
            out_string += "\\vdots & " * post_vdots

        out_string = out_string[:-2] + "\\\\\n "
        out_string += _rows_to_latex(matrix[-max_height // 2 + 1 :], max_width)

    else:
        out_string += _rows_to_latex(matrix, max_width)
    out_string += "\\end{bmatrix}\n"
    return out_string


def array_to_latex(array, precision=10, prefix="", source=False, max_size=8):
    """Latex representation of a complex numpy array (with dimension 1 or 2)

    .. warning::

       This method will relies on SymPy internally which is
       inherently unsafe as SymPy heavily relies on Python's :func:`.eval`
       function. This method is not safe to use with untrusted input.

    Args:
        array (ndarray): The array to be converted to latex, must have dimension 1 or 2 and
                         contain only numerical data.
        precision (int): For numbers not close to integers or common terms, the number of
                         decimal places to round to.
        prefix (str): Latex string to be prepended to the latex, intended for labels.
        source (bool): If ``False``, will return IPython.display.Latex object. If display is
                       ``True``, will instead return the LaTeX source string.
        max_size (list(int) or int): The maximum size of the output Latex array.

            * If list(``int``), then the 0th element of the list specifies the maximum
              width (including dots characters) and the 1st specifies the maximum height
              (also inc. dots characters).
            * If a single ``int`` then this value sets the maximum width _and_ maximum
              height.

    Returns:
        str or IPython.display.Latex: If ``source`` is ``True``, a ``str`` of the LaTeX
            representation of the array, else an ``IPython.display.Latex`` representation of
            the array.

    Raises:
        TypeError: If array can not be interpreted as a numerical numpy array.
        ValueError: If the dimension of array is not 1 or 2.
        MissingOptionalLibraryError: If ``source`` is ``False`` and ``IPython.display.Latex`` cannot be
                     imported. Or if sympy is not installed.
    """
    try:
        array = np.asarray(array)
        _ = array[0] + 1  # Test first element contains numerical data
    except TypeError as err:
        raise TypeError(
            """array_to_latex can only convert numpy arrays containing numerical data,
        or types that can be converted to such arrays"""
        ) from err

    if array.ndim > 2:
        raise ValueError("array_to_latex can only convert numpy ndarrays of dimension 1 or 2")

    if isinstance(max_size, int):
        max_size = (max_size, max_size)

    outstr = _matrix_to_latex(array, decimals=precision, prefix=prefix, max_size=max_size)

    if source is True:
        return outstr

    try:
        from IPython.display import Latex
    except ImportError as err:
        raise MissingOptionalLibraryError(
            libname="IPython",
            name="array_to_latex",
            pip_install="pip install ipython",
        ) from err
    return Latex(f"$${outstr}$$")
