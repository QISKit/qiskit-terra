# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Layer classes for the fast gradient implementation.
"""
from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
from .fast_grad_utils import (
    bit_permutation_1q,
    reverse_bits,
    inverse_permutation,
    bit_permutation_2q,
    make_rz,
    make_ry,
)


class LayerBase(ABC):
    """
    Base class for any layer implementation. Each layer here is represented
    by a 2x2 or 4x4 gate matrix ``G`` (applied to 1 or 2 qubits respectively)
    interleaved with the identity ones:
    ``Layer = I kron I kron ... kron G kron ... kron I kron I``
    """

    @abstractmethod
    def set_from_matrix(self, mat: np.ndarray):
        """
        Updates this layer from an external gate matrix.

        Args:
            mat: external gate matrix that initializes this layer's one.
        """
        raise NotImplementedError

    @abstractmethod
    def get_attr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns gate matrix, direct and inverse permutations.

        Returns:
            (1) gate matrix; (2) direct permutation; (3) inverse permutations.
        """
        raise NotImplementedError


class Layer1Q(LayerBase):
    """
    Layer represents a simple circuit where 1-qubit gate matrix (of size 2x2)
    interleaves with the identity ones.
    """

    def __init__(self, num_qubits: int, k: int, g2x2: np.ndarray | None = None):
        """
        Args:
            num_qubits: number of qubits.
            k: index of the bit where gate is applied.
            g2x2: 2x2 matrix that makes up this layer along with identity ones,
                  or None (should be set up later).
        """
        super().__init__()

        # 2x2 gate matrix (1-qubit gate).
        self._gmat = np.full((2, 2), fill_value=0, dtype=np.complex128)
        if isinstance(g2x2, np.ndarray):
            np.copyto(self._gmat, g2x2)

        bit_flip = True
        dim = 2**num_qubits
        row_perm = reverse_bits(
            bit_permutation_1q(n=num_qubits, k=k), nbits=num_qubits, enable=bit_flip
        )
        col_perm = reverse_bits(np.arange(dim, dtype=np.int64), nbits=num_qubits, enable=bit_flip)
        self._perm = np.full((dim,), fill_value=0, dtype=np.int64)
        self._perm[row_perm] = col_perm
        self._inv_perm = inverse_permutation(self._perm)

    def set_from_matrix(self, mat: np.ndarray):
        """See base class description."""
        np.copyto(self._gmat, mat)

    def get_attr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """See base class description."""
        return self._gmat, self._perm, self._inv_perm


class Layer2Q(LayerBase):
    """
    Layer represents a simple circuit where 2-qubit gate matrix (of size 4x4)
    interleaves with the identity ones.
    """

    def __init__(self, num_qubits: int, j: int, k: int, g4x4: np.ndarray | None = None):
        """
        Args:
            num_qubits: number of qubits.
            j: index of the first (control) bit.
            k: index of the second (target) bit.
            g4x4: 4x4 matrix that makes up this layer along with identity ones,
                  or None (should be set up later).
        """
        super().__init__()

        # 4x4 gate matrix (2-qubit gate).
        self._gmat = np.full((4, 4), fill_value=0, dtype=np.complex128)
        if isinstance(g4x4, np.ndarray):
            np.copyto(self._gmat, g4x4)

        bit_flip = True
        dim = 2**num_qubits
        row_perm = reverse_bits(
            bit_permutation_2q(n=num_qubits, j=j, k=k), nbits=num_qubits, enable=bit_flip
        )
        col_perm = reverse_bits(np.arange(dim, dtype=np.int64), nbits=num_qubits, enable=bit_flip)
        self._perm = np.full((dim,), fill_value=0, dtype=np.int64)
        self._perm[row_perm] = col_perm
        self._inv_perm = inverse_permutation(self._perm)

    def set_from_matrix(self, mat: np.ndarray):
        """See base class description."""
        np.copyto(self._gmat, mat)

    def get_attr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """See base class description."""
        return self._gmat, self._perm, self._inv_perm


def init_layer1q_matrices(thetas: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Initializes 2x2 matrices of 1-qubit gates defined in the paper.

    Args:
        thetas: n x 3 matrix of gate parameters for every qubit, where
                "n" is the number of qubits.
        dst: destination array of size n x 2 x 2 that will receive gate
             matrices of each qubit.

    Returns:
        Returns the "dst" array.
    """
    n = thetas.shape[0]
    tmp = np.full((4, 2, 2), fill_value=0, dtype=np.complex128)
    for k in range(n):
        th = thetas[k]
        a = make_rz(th[0], out=tmp[0])
        b = make_ry(th[1], out=tmp[1])
        c = make_rz(th[2], out=tmp[2])
        np.dot(np.dot(a, b, out=tmp[3]), c, out=dst[k])
    return dst


def init_layer1q_deriv_matrices(thetas: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Initializes 2x2 derivative matrices of 1-qubit gates defined in the paper.

    Args:
        thetas: n x 3 matrix of gate parameters for every qubit, where
                "n" is the number of qubits.
        dst: destination array of size n x 3 x 2 x 2 that will receive gate
             derivative matrices of each qubit; there are 3 parameters per gate,
             hence, 3 derivative matrices per qubit.

    Returns:
        Returns the "dst" array.
    """
    n = thetas.shape[0]
    y = np.asarray([[0, -0.5], [0.5, 0]], dtype=np.complex128)
    z = np.asarray([[-0.5j, 0], [0, 0.5j]], dtype=np.complex128)
    tmp = np.full((5, 2, 2), fill_value=0, dtype=np.complex128)
    for k in range(n):
        th = thetas[k]
        a = make_rz(th[0], out=tmp[0])
        b = make_ry(th[1], out=tmp[1])
        c = make_rz(th[2], out=tmp[2])

        za = np.dot(z, a, out=tmp[3])
        np.dot(np.dot(za, b, out=tmp[4]), c, out=dst[k, 0])
        yb = np.dot(y, b, out=tmp[3])
        np.dot(a, np.dot(yb, c, out=tmp[4]), out=dst[k, 1])
        zc = np.dot(z, c, out=tmp[3])
        np.dot(a, np.dot(b, zc, out=tmp[4]), out=dst[k, 2])
    return dst


def init_layer2q_matrices(thetas: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Initializes 4x4 matrices of 2-qubit gates defined in the paper.

    Args:
        thetas: depth x 4 matrix of gate parameters for every layer, where
                "depth" is the number of layers.
        dst: destination array of size depth x 4 x 4 that will receive gate
             matrices of each layer.

    Returns:
        Returns the "dst" array.
    """
    depth = thetas.shape[0]
    for k in range(depth):
        th = thetas[k]
        cs0 = np.cos(0.5 * th[0]).item()
        sn0 = np.sin(0.5 * th[0]).item()
        ep1 = np.exp(0.5j * th[1]).item()
        en1 = np.exp(-0.5j * th[1]).item()
        cs2 = np.cos(0.5 * th[2]).item()
        sn2 = np.sin(0.5 * th[2]).item()
        cs3 = np.cos(0.5 * th[3]).item()
        sn3 = np.sin(0.5 * th[3]).item()
        ep1cs0 = ep1 * cs0
        ep1sn0 = ep1 * sn0
        en1cs0 = en1 * cs0
        en1sn0 = en1 * sn0
        sn2cs3 = sn2 * cs3
        sn2sn3 = sn2 * sn3
        cs2cs3 = cs2 * cs3
        sn3cs2j = 1j * sn3 * cs2
        sn2sn3j = 1j * sn2sn3

        flat_dst = dst[k].ravel()
        flat_dst[:] = [
            -(sn2sn3j - cs2cs3) * en1cs0,
            -(sn2cs3 + sn3cs2j) * en1cs0,
            (sn2cs3 + sn3cs2j) * en1sn0,
            (sn2sn3j - cs2cs3) * en1sn0,
            (sn2cs3 - sn3cs2j) * en1cs0,
            (sn2sn3j + cs2cs3) * en1cs0,
            -(sn2sn3j + cs2cs3) * en1sn0,
            (-sn2cs3 + sn3cs2j) * en1sn0,
            (-sn2sn3j + cs2cs3) * ep1sn0,
            -(sn2cs3 + sn3cs2j) * ep1sn0,
            -(sn2cs3 + sn3cs2j) * ep1cs0,
            (-sn2sn3j + cs2cs3) * ep1cs0,
            (sn2cs3 - sn3cs2j) * ep1sn0,
            (sn2sn3j + cs2cs3) * ep1sn0,
            (sn2sn3j + cs2cs3) * ep1cs0,
            (sn2cs3 - sn3cs2j) * ep1cs0,
        ]
    return dst


def init_layer2q_deriv_matrices(thetas: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Initializes 4 x 4 derivative matrices of 2-qubit gates defined in the paper.

    Args:
        thetas: depth x 4 matrix of gate parameters for every layer, where
                "depth" is the number of layers.
        dst: destination array of size depth x 4 x 4 x 4 that will receive gate
             derivative matrices of each layer; there are 4 parameters per gate,
             hence, 4 derivative matrices per layer.

    Returns:
        Returns the "dst" array.
    """
    depth = thetas.shape[0]
    for k in range(depth):
        th = thetas[k]
        cs0 = np.cos(0.5 * th[0]).item()
        sn0 = np.sin(0.5 * th[0]).item()
        ep1 = np.exp(0.5j * th[1]).item() * 0.5
        en1 = np.exp(-0.5j * th[1]).item() * 0.5
        cs2 = np.cos(0.5 * th[2]).item()
        sn2 = np.sin(0.5 * th[2]).item()
        cs3 = np.cos(0.5 * th[3]).item()
        sn3 = np.sin(0.5 * th[3]).item()
        ep1cs0 = ep1 * cs0
        ep1sn0 = ep1 * sn0
        en1cs0 = en1 * cs0
        en1sn0 = en1 * sn0
        sn2cs3 = sn2 * cs3
        sn2sn3 = sn2 * sn3
        sn3cs2 = sn3 * cs2
        cs2cs3 = cs2 * cs3
        sn2cs3j = 1j * sn2cs3
        sn2sn3j = 1j * sn2sn3
        sn3cs2j = 1j * sn3cs2
        cs2cs3j = 1j * cs2cs3

        flat_dst = dst[k, 0].ravel()
        flat_dst[:] = [
            (sn2sn3j - cs2cs3) * en1sn0,
            (sn2cs3 + sn3cs2j) * en1sn0,
            (sn2cs3 + sn3cs2j) * en1cs0,
            (sn2sn3j - cs2cs3) * en1cs0,
            (-sn2cs3 + sn3cs2j) * en1sn0,
            -(sn2sn3j + cs2cs3) * en1sn0,
            -(sn2sn3j + cs2cs3) * en1cs0,
            (-sn2cs3 + sn3cs2j) * en1cs0,
            (-sn2sn3j + cs2cs3) * ep1cs0,
            -(sn2cs3 + sn3cs2j) * ep1cs0,
            (sn2cs3 + sn3cs2j) * ep1sn0,
            (sn2sn3j - cs2cs3) * ep1sn0,
            (sn2cs3 - sn3cs2j) * ep1cs0,
            (sn2sn3j + cs2cs3) * ep1cs0,
            -(sn2sn3j + cs2cs3) * ep1sn0,
            (-sn2cs3 + sn3cs2j) * ep1sn0,
        ]

        flat_dst = dst[k, 1].ravel()
        flat_dst[:] = [
            -(sn2sn3 + cs2cs3j) * en1cs0,
            (sn2cs3j - sn3cs2) * en1cs0,
            -(sn2cs3j - sn3cs2) * en1sn0,
            (sn2sn3 + cs2cs3j) * en1sn0,
            -(sn2cs3j + sn3cs2) * en1cs0,
            (sn2sn3 - cs2cs3j) * en1cs0,
            (-sn2sn3 + cs2cs3j) * en1sn0,
            (sn2cs3j + sn3cs2) * en1sn0,
            (sn2sn3 + cs2cs3j) * ep1sn0,
            (-sn2cs3j + sn3cs2) * ep1sn0,
            (-sn2cs3j + sn3cs2) * ep1cs0,
            (sn2sn3 + cs2cs3j) * ep1cs0,
            (sn2cs3j + sn3cs2) * ep1sn0,
            (-sn2sn3 + cs2cs3j) * ep1sn0,
            (-sn2sn3 + cs2cs3j) * ep1cs0,
            (sn2cs3j + sn3cs2) * ep1cs0,
        ]

        flat_dst = dst[k, 2].ravel()
        flat_dst[:] = [
            -(sn2cs3 + sn3cs2j) * en1cs0,
            (sn2sn3j - cs2cs3) * en1cs0,
            -(sn2sn3j - cs2cs3) * en1sn0,
            (sn2cs3 + sn3cs2j) * en1sn0,
            (sn2sn3j + cs2cs3) * en1cs0,
            (-sn2cs3 + sn3cs2j) * en1cs0,
            (sn2cs3 - sn3cs2j) * en1sn0,
            -(sn2sn3j + cs2cs3) * en1sn0,
            -(sn2cs3 + sn3cs2j) * ep1sn0,
            (sn2sn3j - cs2cs3) * ep1sn0,
            (sn2sn3j - cs2cs3) * ep1cs0,
            -(sn2cs3 + sn3cs2j) * ep1cs0,
            (sn2sn3j + cs2cs3) * ep1sn0,
            (-sn2cs3 + sn3cs2j) * ep1sn0,
            (-sn2cs3 + sn3cs2j) * ep1cs0,
            (sn2sn3j + cs2cs3) * ep1cs0,
        ]

        flat_dst = dst[k, 3].ravel()
        flat_dst[:] = [
            -(sn2cs3j + sn3cs2) * en1cs0,
            (sn2sn3 - cs2cs3j) * en1cs0,
            (-sn2sn3 + cs2cs3j) * en1sn0,
            (sn2cs3j + sn3cs2) * en1sn0,
            -(sn2sn3 + cs2cs3j) * en1cs0,
            (sn2cs3j - sn3cs2) * en1cs0,
            -(sn2cs3j - sn3cs2) * en1sn0,
            (sn2sn3 + cs2cs3j) * en1sn0,
            -(sn2cs3j + sn3cs2) * ep1sn0,
            (sn2sn3 - cs2cs3j) * ep1sn0,
            (sn2sn3 - cs2cs3j) * ep1cs0,
            -(sn2cs3j + sn3cs2) * ep1cs0,
            -(sn2sn3 + cs2cs3j) * ep1sn0,
            (sn2cs3j - sn3cs2) * ep1sn0,
            (sn2cs3j - sn3cs2) * ep1cs0,
            -(sn2sn3 + cs2cs3j) * ep1cs0,
        ]
    return dst
