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
Implementation of the GraySynth algorithm for synthesizing CNOT-Phase
circuits with efficient CNOT cost, and the Patel-Hayes-Markov algorithm
for optimal synthesis of linear (CNOT-only) reversible circuits.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit

from qiskit._accelerate.synthesis.linear import synth_cnot_count_full_pmh as fast_pmh


def synth_cnot_count_full_pmh(
    state: list[list[bool]] | np.ndarray[bool], section_size: int | None = None
) -> QuantumCircuit:
    r"""
    Synthesize linear reversible circuits for all-to-all architecture
    using Patel, Markov and Hayes method.

    This function is an implementation of the Patel, Markov and Hayes algorithm from [1]
    for optimal synthesis of linear reversible circuits for all-to-all architecture,
    as specified by an :math:`n \times n` matrix.

    Args:
        state: :math:`n \times n` boolean invertible matrix, describing
            the state of the input circuit.
        section_size: The size of each section in the Patel–Markov–Hayes algorithm [1].
            If ``None`` it is chosen to be :math:`\max(2, \alpha\log_2(n))` with
            :math:`\alpha = 0.56`, which approximately minimizes the upper bound on the number
            of row operations given in [1] Eq. (3).

    Returns:
        A CX-only circuit implementing the linear transformation.

    Raises:
        ValueError: When ``section_size`` is larger than the number of columns.

    References:
        1. Patel, Ketan N., Igor L. Markov, and John P. Hayes,
           *Optimal synthesis of linear reversible circuits*,
           Quantum Information & Computation 8.3 (2008): 282-294.
           `arXiv:quant-ph/0302002 [quant-ph] <https://arxiv.org/abs/quant-ph/0302002>`_
    """
    normalized = np.asarray(state).astype(bool)
    if section_size is not None and normalized.shape[1] < section_size:
        raise ValueError(
            f"The section_size ({section_size}) cannot be larger than the number of columns "
            f"({normalized.shape[1]})."
        )

    # call Rust implementation with normalized input
    circuit_data = fast_pmh(normalized, section_size)

    # construct circuit from the data
    return QuantumCircuit._from_circuit_data(circuit_data, legacy_qubits=True)
