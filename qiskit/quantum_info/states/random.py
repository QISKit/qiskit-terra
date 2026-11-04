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

"""
Random state generation.
"""

from __future__ import annotations
from typing import Literal

import numpy as np
from numpy.random import default_rng

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.random import random_unitary
from .statevector import Statevector
from .densitymatrix import DensityMatrix


def random_statevector(
    dims: int | tuple, seed: int | np.random.Generator | None = None
) -> Statevector:
    """Generate a random Statevector.

    The statevector is sampled from the uniform distribution. This is the measure
    induced by the Haar measure on unitary matrices.

    Args:
        dims (int or tuple): the dimensions of the state.
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Statevector: the random statevector.

    Reference:
        K. Zyczkowski and H. Sommers (2001), "Induced measures in the space of mixed quantum states",
        `J. Phys. A: Math. Gen. 34 7111 <https://arxiv.org/abs/quant-ph/0012101>`__.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    dim = np.prod(dims)
    vec = rng.standard_normal(dim).astype(complex)
    vec += 1j * rng.standard_normal(dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec, dims=dims)


def random_density_matrix(
    dims: int | tuple,
    rank: int | None = None,
    method: Literal["Hilbert-Schmidt", "Bures"] = "Hilbert-Schmidt",
    seed: int | np.random.Generator | None = None,
) -> DensityMatrix:
    """Generate a random density matrix.

    Args:
        dims (int or tuple): the dimensions of the DensityMatrix.
        rank (int or None):  the rank of the density matrix.
                            The default value is full-rank.
        method (string): Optional. The method to use.
            'Hilbert-Schmidt': (Default) sample from the Hilbert-Schmidt metric.
            'Bures': sample from the Bures metric.
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        DensityMatrix: the random density matrix.

    Raises:
        QiskitError: if the method is not valid.
    """
    # Flatten dimensions
    dim = np.prod(dims)
    if rank is None:
        rank = dim  # Use full rank

    if method == "Hilbert-Schmidt":
        rho = _random_density_hs(dim, rank, seed)
    elif method == "Bures":
        rho = _random_density_bures(dim, rank, seed)
    else:
        raise QiskitError(f"Error: unrecognized method {method}")
    return DensityMatrix(rho, dims=dims)


def _ginibre_matrix(nrow, ncol, seed):
    """Return a normally distributed complex random matrix.

    Args:
        nrow (int): number of rows in output matrix.
        ncol (int): number of columns in output matrix.
        seed(int or np.random.Generator): default rng.

    Returns:
        ndarray: A complex rectangular matrix where each real and imaginary
            entry is sampled from the normal distribution.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    ginibre = rng.normal(size=(nrow, ncol)) + rng.normal(size=(nrow, ncol)) * 1j
    return ginibre


def _random_density_hs(dim, rank, seed):
    """
    Generate a random density matrix from the Hilbert-Schmidt metric.

    Args:
        dim (int): the dimensions of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int or np.random.Generator): default rng.

    Returns:
        ndarray: rho (N,N)  a density matrix.
    """
    mat = _ginibre_matrix(dim, rank, seed)
    mat = mat.dot(mat.conj().T)
    return mat / np.trace(mat)


def _random_density_bures(dim, rank, seed):
    """Generate a random density matrix from the Bures metric.

    Args:
        dim (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int or np.random.Generator): default rng.

    Returns:
        ndarray: rho (N,N) a density matrix.
    """
    density = np.eye(dim) + random_unitary(dim, seed=seed).data
    mat = density.dot(_ginibre_matrix(dim, rank, seed))
    mat = mat.dot(mat.conj().T)
    return mat / np.trace(mat)
