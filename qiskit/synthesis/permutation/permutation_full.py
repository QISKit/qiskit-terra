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

"""Synthesis algorithm for Permutation gates for full-connectivity."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit._accelerate.synthesis.permutation import (
    _synth_permutation_basic,
    _synth_permutation_acg,
)


def synth_permutation_basic(pattern: list[int] | np.ndarray[int]) -> QuantumCircuit:
    """Synthesize a permutation circuit for a fully-connected
    architecture using sorting.

    More precisely, if the input permutation is a cycle of length ``m``,
    then this creates a quantum circuit with ``m-1`` SWAPs (and of depth ``m-1``);
    if the input  permutation consists of several disjoint cycles, then each cycle
    is essentially treated independently.

    Args:
        pattern: Permutation pattern, describing
            which qubits occupy the positions 0, 1, 2, etc. after applying the
            permutation. That is, ``pattern[k] = m`` when the permutation maps
            qubit ``m`` to position ``k``. As an example, the pattern ``[2, 4, 3, 0, 1]``
            means that qubit ``2`` goes to position ``0``, qubit ``4`` goes to
            position ``1``, etc.

    Returns:
        The synthesized quantum circuit.
    """
    return QuantumCircuit._from_circuit_data(_synth_permutation_basic(pattern), legacy_qubits=True)


def synth_permutation_acg(pattern: list[int] | np.ndarray[int]) -> QuantumCircuit:
    """Synthesize a permutation circuit for a fully-connected
    architecture using the Alon, Chung, Graham method.

    This produces a quantum circuit of depth 2 (measured in the number of SWAPs).

    This implementation is based on the Proposition 4.1 in reference [1] with
    the detailed proof given in Theorem 2 in reference [2]

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
        1. N. Alon, F. R. K. Chung, and R. L. Graham.
           *Routing Permutations on Graphs Via Matchings.*,
           Proceedings of the Twenty-Fifth Annual ACM Symposium on Theory of Computing(1993).
           Pages 583–591.
           `(Extended abstract) 10.1145/167088.167239 <https://doi.org/10.1145/167088.167239>`_
        2. N. Alon, F. R. K. Chung, and R. L. Graham.
           *Routing Permutations on Graphs Via Matchings.*,
           `(Full paper) <https://www.cs.tau.ac.il/~nogaa/PDFS/r.pdf>`_
    """
    return QuantumCircuit._from_circuit_data(_synth_permutation_acg(pattern), legacy_qubits=True)
