# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute the sum of two qubit registers using QFT."""

import numpy as np

from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import QFTGate


def adder_qft_d00(
    num_state_qubits: int, kind: str = "half", annotated: bool = False
) -> QuantumCircuit:
    r"""A circuit that uses QFT to perform in-place addition on two qubit registers.

    For registers with :math:`n` qubits, the QFT adder can perform addition modulo
    :math:`2^n` (with ``kind="fixed"``) or ordinary addition by adding a carry qubits (with
    ``kind="half"``). The fixed adder uses :math:`(3n^2 - n)/2` :class:`.CPhaseGate` operators,
    with an additional :math:`n` for the half adder.

    As an example, a non-fixed_point QFT adder circuit that performs addition on two 2-qubit sized
    registers is as follows:

    .. parsed-literal::

        a_0: ─────────■──────■────────■──────────────────────────────────
                      │      │        │
        a_1: ─────────┼──────┼────────┼────────■──────■──────────────────
             ┌──────┐ │      │        │P(π/4)  │      │P(π/2) ┌─────────┐
        b_0: ┤0     ├─┼──────┼────────■────────┼──────■───────┤0        ├
             │      │ │      │P(π/2)           │P(π)          │         │
        b_1: ┤1 Qft ├─┼──────■─────────────────■──────────────┤1 qft_dg ├
             │      │ │P(π)                                   │         │
       cout: ┤2     ├─■───────────────────────────────────────┤2        ├
             └──────┘                                         └─────────┘

    Args:
        num_state_qubits: The number of qubits in either input register for
            state :math:`|a\rangle` or :math:`|b\rangle`. The two input
            registers must have the same number of qubits.
        kind: The kind of adder, can be ``"half"`` for a half adder or
            ``"fixed"`` for a fixed-sized adder. A half adder contains a carry-out to represent
            the most-significant bit, but the fixed-sized adder doesn't and hence performs
            addition modulo ``2 ** num_state_qubits``.
        annotated: If ``True``, creates appropriate control and inverse operations as
            ``AnnotatedOperation`` objects.

    References:

    [1] T. G. Draper, Addition on a Quantum Computer, 2000.
    `arXiv:quant-ph/0008033 <https://arxiv.org/pdf/quant-ph/0008033.pdf>`_

    [2] Ruiz-Perez et al., Quantum arithmetic with the Quantum Fourier Transform, 2017.
    `arXiv:1411.5949 <https://arxiv.org/pdf/1411.5949.pdf>`_

    [3] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_

    """

    if kind == "full":
        raise ValueError("The DraperQFTAdder only supports 'half' and 'fixed' as ``kind``.")

    if num_state_qubits < 1:
        raise ValueError("The number of qubits must be at least 1.")

    qr_a = QuantumRegister(num_state_qubits, name="a")
    qr_b = QuantumRegister(num_state_qubits, name="b")
    qr_list = [qr_a, qr_b]

    if kind == "half":
        qr_z = QuantumRegister(1, name="cout")
        qr_list.append(qr_z)

    # add registers
    circuit = QuantumCircuit(*qr_list)

    # define register containing the sum and number of qubits for QFT circuit
    qr_sum = qr_b[:] if kind == "fixed" else qr_b[:] + qr_z[:]
    num_sum = num_state_qubits if kind == "fixed" else num_state_qubits + 1

    # build QFT adder circuit
    qft = QFTGate(num_sum)
    circuit.append(qft, qr_sum[:])

    for j in range(num_state_qubits):
        for k in range(num_sum - j):
            lam = np.pi / (2**k)
            # note: if we were able to remove the final swaps from the QFTGate, we could
            # simply use qr_sum[(j + k)] here and avoid synthesizing two swap networks, which
            # can be elided and cancelled by the compiler
            circuit.cp(lam, qr_a[j], qr_sum[~(j + k)])

    circuit.append(qft.inverse(annotated=annotated), qr_sum[:])

    return circuit
