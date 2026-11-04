# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Random symplectic operator functions
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng

from qiskit._accelerate.synthesis.clifford import random_clifford_tableau

from .clifford import Clifford
from .pauli import Pauli
from .pauli_list import PauliList


def random_pauli(
    num_qubits: int, group_phase: bool = False, seed: int | np.random.Generator | None = None
) -> Pauli:
    """Return a random :class:`Pauli`.

    Args:
        num_qubits: The number of qubits.
        group_phase: Optional. If ``True`` generate random phase.
                     Otherwise the phase will be set so that the
                     Pauli coefficient is +1 (default: ``False``).
        seed: Optional. Set a fixed seed or generator for RNG.

    Returns:
        A random Pauli.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)
    z = rng.integers(2, size=num_qubits, dtype=bool)
    x = rng.integers(2, size=num_qubits, dtype=bool)
    phase = rng.integers(4) if group_phase else 0
    pauli = Pauli((z, x, phase))
    return pauli


def random_pauli_list(
    num_qubits: int,
    size: int = 1,
    seed: int | np.random.Generator | None = None,
    phase: bool = True,
) -> PauliList:
    """Return a random :class:`PauliList`.

    Args:
        num_qubits: The number of qubits.
        size: Optional. The length of the Pauli list (Default: 1).
        seed: Optional. Set a fixed seed or generator for RNG.
        phase: Optional. If ``True`` the Pauli phases are randomized, otherwise the phases are fixed
            to 0 (Default: ``True``).

    Returns:
        A random :class:`PauliList`.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    z = rng.integers(2, size=(size, num_qubits)).astype(bool)
    x = rng.integers(2, size=(size, num_qubits)).astype(bool)
    if phase:
        _phase = rng.integers(4, size=size)
        return PauliList.from_symplectic(z, x, _phase)
    return PauliList.from_symplectic(z, x)


def random_clifford(num_qubits: int, seed: int | np.random.Generator | None = None) -> Clifford:
    """Return a random Clifford operator.

    The Clifford is sampled using the method of Reference [1].

    Args:
        num_qubits: The number of qubits for the Clifford.
        seed: Optional. Set a fixed seed or generator for RNG.

    Returns:
        A random Clifford operator.

    References:

    [1] S. Bravyi and D. Maslov (2020). Hadamard-free circuits expose the structure of the
    Clifford group. `arXiv:2003.09412 <https://arxiv.org/abs/2003.09412>`__
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    seed = rng.integers(100000, size=1, dtype=np.uint64)[0]
    tableau = random_clifford_tableau(num_qubits, seed=seed)
    return Clifford(tableau, validate=False)
