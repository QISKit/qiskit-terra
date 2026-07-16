# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute modular sum of two qubit registers without any ancillary qubits."""

from __future__ import annotations

from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import XGate, MCMTGate


def adder_modular_v17(num_qubits: int) -> QuantumCircuit:
    r"""
    Construct a modular adder circuit with no ancillary qubits based on the Van Rentergem-style
    adder in Fig. 15 of [1]. The implementation uses at most :math:`16k - 13` CX gates for an
    adder with `k` qubits in each register, where `k = num_qubits`.

    Args:
        num_qubits: The size of the register.

    Returns:
        The quantum circuit implementing the modular adder.

    Raises:
        ValueError: If ``num_qubits`` is less than 1.

    References:

    [1] Gidney, Factoring with n+2 clean qubits and n-1 dirty qubits, 2017.
    `arxiv:1706.07884 <https://arxiv.org/abs/1706.07884>`_

    """

    if num_qubits < 1:
        raise ValueError("The number of qubits must be at least 1.")

    qr_a = QuantumRegister(num_qubits, "a")
    qr_b = QuantumRegister(num_qubits, "b")
    qc = QuantumCircuit(qr_a, qr_b)

    if num_qubits == 1:
        qc.cx(qr_a[0], qr_b[0])
        return qc

    mcmt = MCMTGate(XGate(), 1, 2 * num_qubits - 1)

    qc.compose(mcmt, [qr_a[-1]] + qr_a[:-1] + qr_b[:], inplace=True)

    # Ripple forward.
    # Use the following facts:
    #   1. CSWAP(a, b, c) = CX(c, b) CCX(a, b, c) CX(c, b)
    #   2. Use RCCX instead of CCX because CSWAP gates appear in a compute-uncompute pattern.
    for i in range(num_qubits - 1):
        qc.cx(qr_a[-1], qr_b[i])
        qc.cx(qr_a[i], qr_a[-1])
        qc.rccx(qr_b[i], qr_a[-1], qr_a[i])
        qc.cx(qr_a[i], qr_a[-1])

    qc.cx(qr_a[-1], qr_b[-1])  # High bit toggle.

    # Ripple backward.
    for i in range(num_qubits - 2, -1, -1):
        qc.cx(qr_a[i], qr_a[-1])
        qc.rccx(qr_b[i], qr_a[-1], qr_a[i])
        qc.cx(qr_a[i], qr_a[-1])
        qc.cx(qr_a[i], qr_b[i])

    qc.compose(mcmt, [qr_a[-1]] + qr_a[:-1] + qr_b[:], inplace=True)

    return qc
