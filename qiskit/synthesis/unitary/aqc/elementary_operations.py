# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
These are a number of elementary functions that are required for the AQC routines to work.
"""

import numpy as np

from qiskit.circuit.library import RXGate, RZGate, RYGate


def place_unitary(unitary: np.ndarray, n: int, j: int) -> np.ndarray:
    """
    Computes I(j - 1) tensor product U tensor product I(n - j), where U is a unitary matrix
    of size ``(2, 2)``.

    Args:
        unitary: a unitary matrix of size ``(2, 2)``.
        n: num qubits.
        j: position where to place a unitary.

    Returns:
        a unitary of n qubits with u in position j.
    """
    return np.kron(np.kron(np.eye(2**j), unitary), np.eye(2 ** (n - 1 - j)))


def place_cnot(n: int, j: int, k: int) -> np.ndarray:
    """
    Places a CNOT from j to k.

    Args:
        n: number of qubits.
        j: control qubit.
        k: target qubit.

    Returns:
        a unitary of n qubits with CNOT placed at ``j`` and ``k``.
    """
    if j < k:
        unitary = np.kron(
            np.kron(np.eye(2**j), [[1, 0], [0, 0]]), np.eye(2 ** (n - 1 - j))
        ) + np.kron(
            np.kron(
                np.kron(np.kron(np.eye(2**j), [[0, 0], [0, 1]]), np.eye(2 ** (k - j - 1))),
                [[0, 1], [1, 0]],
            ),
            np.eye(2 ** (n - 1 - k)),
        )
    else:
        unitary = np.kron(
            np.kron(np.eye(2**j), [[1, 0], [0, 0]]), np.eye(2 ** (n - 1 - j))
        ) + np.kron(
            np.kron(
                np.kron(np.kron(np.eye(2**k), [[0, 1], [1, 0]]), np.eye(2 ** (j - k - 1))),
                [[0, 0], [0, 1]],
            ),
            np.eye(2 ** (n - 1 - j)),
        )
    return unitary


def rx_matrix(phi: float) -> np.ndarray:
    """
    Computes an RX rotation by the angle of ``phi``.

    Args:
        phi: rotation angle.

    Returns:
        an RX rotation matrix.
    """
    return RXGate(phi).to_matrix()


def ry_matrix(phi: float) -> np.ndarray:
    """
    Computes an RY rotation by the angle of ``phi``.

    Args:
        phi: rotation angle.

    Returns:
        an RY rotation matrix.
    """
    return RYGate(phi).to_matrix()


def rz_matrix(phi: float) -> np.ndarray:
    """
    Computes an RZ rotation by the angle of ``phi``.

    Args:
        phi: rotation angle.

    Returns:
        an RZ rotation matrix.
    """
    return RZGate(phi).to_matrix()
