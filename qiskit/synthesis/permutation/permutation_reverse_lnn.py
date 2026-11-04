# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Synthesis of a reverse permutation for LNN connectivity.
"""

from qiskit.circuit import QuantumCircuit
from qiskit._accelerate.synthesis.permutation import (
    synth_permutation_reverse_lnn_kms as synth_permutation_reverse_lnn_kms_inner,
)


def _append_cx_stage1(qc, n):
    """A single layer of CX gates."""
    for i in range(n // 2):
        qc.cx(2 * i, 2 * i + 1)
    for i in range((n + 1) // 2 - 1):
        qc.cx(2 * i + 2, 2 * i + 1)
    return qc


def _append_cx_stage2(qc, n):
    """A single layer of CX gates."""
    for i in range(n // 2):
        qc.cx(2 * i + 1, 2 * i)
    for i in range((n + 1) // 2 - 1):
        qc.cx(2 * i + 1, 2 * i + 2)
    return qc


def _append_reverse_permutation_lnn_kms(qc: QuantumCircuit, num_qubits: int) -> None:
    """
    Append reverse permutation to a QuantumCircuit for linear nearest-neighbor architectures
    using Kutin, Moulton, Smithline method.

    Synthesis algorithm for reverse permutation from [1], section 5.
    This algorithm synthesizes the reverse permutation on :math:`n` qubits over
    a linear nearest-neighbor architecture using CX gates with depth :math:`2 * n + 2`.

    Args:
        qc: The original quantum circuit.
        num_qubits: The number of qubits.

    Returns:
        The quantum circuit with appended reverse permutation.

    References:
        1. Kutin, S., Moulton, D. P., Smithline, L.,
           *Computation at a distance*, Chicago J. Theor. Comput. Sci., vol. 2007, (2007),
           `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_
    """

    for _ in range((num_qubits + 1) // 2):
        _append_cx_stage1(qc, num_qubits)
        _append_cx_stage2(qc, num_qubits)
    if (num_qubits % 2) == 0:
        _append_cx_stage1(qc, num_qubits)


def synth_permutation_reverse_lnn_kms(num_qubits: int) -> QuantumCircuit:
    """
    Synthesize reverse permutation for linear nearest-neighbor architectures using
    Kutin, Moulton, Smithline method.

    Synthesis algorithm for reverse permutation from [1], section 5.
    This algorithm synthesizes the reverse permutation on :math:`n` qubits over
    a linear nearest-neighbor architecture using CX gates with depth :math:`2 * n + 2`.

    Args:
        num_qubits: The number of qubits.

    Returns:
        The synthesized quantum circuit.

    References:
        1. Kutin, S., Moulton, D. P., Smithline, L.,
           *Computation at a distance*, Chicago J. Theor. Comput. Sci., vol. 2007, (2007),
           `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_
    """

    # Call Rust implementation
    return QuantumCircuit._from_circuit_data(
        synth_permutation_reverse_lnn_kms_inner(num_qubits), legacy_qubits=True
    )
