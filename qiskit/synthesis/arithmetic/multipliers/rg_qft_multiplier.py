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

"""Compute the product of two qubit registers using QFT."""

from __future__ import annotations

import numpy as np

from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import PhaseGate, QFTGate


def multiplier_qft_r17(
    num_state_qubits: int, num_result_qubits: int | None = None
) -> QuantumCircuit:
    r"""A QFT multiplication circuit to store product of two input registers out-of-place.

    Multiplication in this circuit is implemented using the procedure of Fig. 3 in [1], where
    weighted sum rotations are implemented as given in Fig. 5 in [1]. QFT is used on the output
    register and is followed by rotations controlled by input registers. The rotations
    transform the state into the product of two input registers in QFT base, which is
    reverted from QFT base using inverse QFT.
    For example, on 3 state qubits, a full multiplier is given by:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:

        from qiskit.synthesis.arithmetic import multiplier_qft_r17

        num_state_qubits = 3
        circuit = multiplier_qft_r17(num_state_qubits)
        circuit.draw("mpl")

    Args:
        num_state_qubits: The number of qubits in either input register for
            state :math:`|a\rangle` or :math:`|b\rangle`. The two input
            registers must have the same number of qubits.
        num_result_qubits: The number of result qubits to limit the output to.
            If number of result qubits is :math:`n`, multiplication modulo :math:`2^n` is performed
            to limit the output to the specified number of qubits. Default
            value is ``2 * num_state_qubits`` to represent any possible
            result from the multiplication of the two inputs.

    Raises:
        ValueError: If ``num_result_qubits`` is given and not valid, meaning not
            in ``[num_state_qubits, 2 * num_state_qubits]``.

    References:

    [1] Ruiz-Perez et al., Quantum arithmetic with the Quantum Fourier Transform, 2017.
    `arXiv:1411.5949 <https://arxiv.org/pdf/1411.5949.pdf>`_

    """
    # define the registers
    if num_result_qubits is None:
        num_result_qubits = 2 * num_state_qubits
    elif num_result_qubits < num_state_qubits or num_result_qubits > 2 * num_state_qubits:
        raise ValueError(
            f"num_result_qubits ({num_result_qubits}) must be in between num_state_qubits "
            f"({num_state_qubits}) and 2 * num_state_qubits ({2 * num_state_qubits})"
        )

    qr_a = QuantumRegister(num_state_qubits, name="a")
    qr_b = QuantumRegister(num_state_qubits, name="b")
    qr_out = QuantumRegister(num_result_qubits, name="out")

    # build multiplication circuit
    circuit = QuantumCircuit(qr_a, qr_b, qr_out)
    qft = QFTGate(num_result_qubits)

    circuit.append(qft, qr_out[:])

    for j in range(1, num_state_qubits + 1):
        for i in range(1, num_state_qubits + 1):
            for k in range(1, num_result_qubits + 1):
                lam = (2 * np.pi) / (2 ** (i + j + k - 2 * num_state_qubits))

                # note: if we can synthesize the QFT without swaps, we can implement this circuit
                # more efficiently and just apply phase gate on qr_out[(k - 1)] instead
                circuit.append(
                    PhaseGate(lam).control(2, annotated=False),
                    [qr_a[num_state_qubits - j], qr_b[num_state_qubits - i], qr_out[~(k - 1)]],
                )

    circuit.append(qft.inverse(), qr_out[:])

    return circuit
