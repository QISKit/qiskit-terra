# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Random CNOTDihedral operator functions
"""

from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from .dihedral import CNOTDihedral


def random_cnotdihedral(num_qubits, seed=None):
    """Return a random CNOTDihedral element.

    Args:
        num_qubits (int): the number of qubits for the CNOTDihedral object.
        seed (int or RandomState): Optional. Set a fixed seed or
                                   generator for RNG.
    Returns:
        CNOTDihedral: a random CNOTDihedral element.
    """

    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    elem = CNOTDihedral(num_qubits=num_qubits)

    # Random phase polynomial weights
    weight_1 = rng.integers(8, size=num_qubits)
    elem.poly.weight_1 = weight_1
    weight_2 = 2 * rng.integers(4, size=int(num_qubits * (num_qubits - 1) / 2))
    elem.poly.weight_2 = weight_2
    weight_3 = 4 * rng.integers(2, size=int(num_qubits * (num_qubits - 1) * (num_qubits - 2) / 6))
    elem.poly.weight_3 = weight_3

    # Random affine function
    # Random invertible binary matrix
    from qiskit.synthesis.linear import (
        random_invertible_binary_matrix,
    )

    seed = rng.integers(100000, size=1, dtype=np.uint64)[0]
    linear = random_invertible_binary_matrix(num_qubits, seed=seed).astype(int, copy=False)
    elem.linear = linear

    # Random shift
    shift = rng.integers(2, size=num_qubits)
    elem.shift = shift

    return elem
