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


"""Routines that use local invariants to compute properties
of two-qubit unitary operators.
"""
from __future__ import annotations
import numpy as np
from qiskit._accelerate.two_qubit_decompose import two_qubit_local_invariants as tqli_rs
from qiskit._accelerate.two_qubit_decompose import local_equivalence as le_rs


def two_qubit_local_invariants(U: np.ndarray) -> np.ndarray:
    """Computes the local invariants for a two-qubit unitary.

    Args:
        U (ndarray): Input two-qubit unitary.

    Returns:
        ndarray: NumPy array of local invariants [g0, g1, g2].

    Raises:
        ValueError: Input not a 2q unitary.

    Notes:
        Y. Makhlin, Quant. Info. Proc. 1, 243-252 (2002).
        Zhang et al., Phys Rev A. 67, 042313 (2003).
    """
    U = np.asarray(U, dtype=complex)
    if U.shape != (4, 4):
        raise ValueError("Unitary must correspond to a two-qubit gate.")
    (a, b, c) = tqli_rs(U)
    return np.array([round(a, 12), round(b, 12), round(c, 12)])


def local_equivalence(weyl: np.ndarray) -> np.ndarray:
    """Computes the equivalent local invariants from the
    Weyl coordinates.

    Args:
        weyl (ndarray): Weyl coordinates.

    Returns:
        ndarray: Local equivalent coordinates [g0, g1, g3].

    Notes:
        This uses Eq. 30 from Zhang et al, PRA 67, 042313 (2003),
        but we multiply weyl coordinates by 2 since we are
        working in the reduced chamber.
    """
    mat = np.asarray(weyl, dtype=float)
    (a, b, c) = le_rs(mat)
    return np.array([round(a, 12), round(b, 12), round(c, 12)])
