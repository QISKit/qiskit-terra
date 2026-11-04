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

"""Depth-efficient synthesis algorithm for Permutation gates."""

from __future__ import annotations
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit._accelerate.synthesis.permutation import _synth_permutation_depth_lnn_kms


def synth_permutation_depth_lnn_kms(pattern: list[int] | np.ndarray[int]) -> QuantumCircuit:
    """Synthesize a permutation circuit for a linear nearest-neighbor
    architecture using the Kutin, Moulton, Smithline method.

    This is the permutation synthesis algorithm from [1], section 6.
    It synthesizes any permutation of n qubits over linear nearest-neighbor
    architecture using SWAP gates with depth at most :math:`n` and size at most
    :math:`n(n-1)/2` (where both depth and size are measured with respect to SWAPs).

    Args:
        pattern: Permutation pattern, describing
            which qubits occupy the positions 0, 1, 2, etc. after applying the
            permutation. That is, ``pattern[k] = m`` when the permutation maps
            qubit ``m`` to position ``k``. As an example, the pattern ``[2, 4, 3, 0, 1]``
            means that qubit ``2`` goes to position ``0``, qubit ``4`` goes to
            position ``1``, etc.

    Returns:
        The synthesized quantum circuit.

    References:
        1. Samuel A. Kutin, David Petrie Moulton and Lawren M. Smithline.
           *Computation at a distance.*,
           `arXiv:quant-ph/0701194v1 <https://arxiv.org/abs/quant-ph/0701194>`_
    """

    # In Qiskit, the permutation pattern [2, 4, 3, 0, 1] means that
    # the permutation that maps qubit 2 to position 0, 4 to 1, 3 to 2, 0 to 3, and 1 to 4.
    # In the permutation synthesis code below the notation is opposite:
    # [2, 4, 3, 0, 1] means that 0 maps to 2, 1 to 4, 2 to 3, 3 to 0, and 4 to 1.
    # This is why we invert the pattern.
    return QuantumCircuit._from_circuit_data(
        _synth_permutation_depth_lnn_kms(pattern), legacy_qubits=True
    )
