# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Synthesis of an n-qubit circuit containing only CZ gates for
linear nearest neighbor (LNN) connectivity, using CX and phase (S, Sdg or Z) gates.
The two-qubit depth of the circuit is bounded by 2*n+2.
This algorithm reverts the order of qubits.

References:
    [1]: Dmitri Maslov, Martin Roetteler,
         Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations,
         `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit

from qiskit._accelerate.synthesis.linear_phase import (
    synth_cz_depth_line_mr as synth_cz_depth_line_mr_inner,
)


def synth_cz_depth_line_mr(mat: np.ndarray) -> QuantumCircuit:
    r"""Synthesis of a CZ circuit for linear nearest neighbor (LNN) connectivity,
    based on Maslov and Roetteler.

    Note that this method *reverts* the order of qubits in the circuit,
    and returns a circuit containing :class:`.CXGate`\s and phase gates
    (:class:`.SGate`, :class:`.SdgGate` or :class:`.ZGate`).

    Args:
        mat: a square upper-diagonal matrix of `bool` representing the CZ circuit.
            ``mat[i][j]=1 for i<j`` represents a ``cz(i,j)`` gate.  Only the upper triangle is read
            from; the diagonal and lower triangle have no effect.

    Returns:
        A circuit implementation of the CZ circuit of depth :math:`2n+2` for LNN
        connectivity.

    References:
        1. Dmitri Maslov, Martin Roetteler,
           *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,
           `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
    """

    # Call Rust implementation
    return QuantumCircuit._from_circuit_data(
        synth_cz_depth_line_mr_inner(mat.astype(bool)), legacy_qubits=True
    )
