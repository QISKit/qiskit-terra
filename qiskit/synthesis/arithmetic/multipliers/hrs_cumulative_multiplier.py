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

"""Compute the product of two qubit registers using classical multiplication approach."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import HalfAdderGate, ModularAdderGate


def multiplier_cumulative_h18(
    num_state_qubits: int, num_result_qubits: int | None = None
) -> QuantumCircuit:
    r"""A multiplication circuit to store product of two input registers out-of-place.

    The circuit uses the approach from Ref. [1]. As an example, a multiplier circuit that
    performs a non-modular multiplication on two 3-qubit sized registers is:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:

        from qiskit.synthesis.arithmetic import multiplier_cumulative_h18

        num_state_qubits = 3
        circuit = multiplier_cumulative_h18(num_state_qubits)
        circuit.draw("mpl")

    Multiplication in this circuit is implemented in a classical approach by performing
    a series of shifted additions using one of the input registers while the qubits
    from the other input register act as control qubits for the adders.

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

    [1] Häner et al., Optimizing Quantum Circuits for Arithmetic, 2018.
    `arXiv:1805.12445 <https://arxiv.org/pdf/1805.12445.pdf>`_

    """
    if num_result_qubits is None:
        num_result_qubits = 2 * num_state_qubits
    elif num_result_qubits < num_state_qubits or num_result_qubits > 2 * num_state_qubits:
        raise ValueError(
            f"num_result_qubits ({num_result_qubits}) must be in between num_state_qubits "
            f"({num_state_qubits}) and 2 * num_state_qubits ({2 * num_state_qubits})"
        )

    # define the registers
    qr_a = QuantumRegister(num_state_qubits, name="a")
    qr_b = QuantumRegister(num_state_qubits, name="b")
    qr_out = QuantumRegister(num_result_qubits, name="out")

    circuit = QuantumCircuit(qr_a, qr_b, qr_out)

    # prepare adder as controlled gate

    adder = HalfAdderGate(num_state_qubits)
    controlled_adder = adder.control(annotated=True)

    # build multiplication circuit
    for i in range(num_state_qubits):
        excess_qubits = max(0, num_state_qubits + i + 1 - num_result_qubits)
        if excess_qubits == 0:
            num_adder_qubits = num_state_qubits
            this_controlled = controlled_adder
        else:
            num_adder_qubits = num_state_qubits - excess_qubits + 1
            modular = ModularAdderGate(num_adder_qubits)
            this_controlled = modular.control(annotated=True)

        qr_list = (
            [qr_a[i]]
            + qr_b[:num_adder_qubits]
            + qr_out[i : num_state_qubits + i + 1 - excess_qubits]
        )
        circuit.append(this_controlled, qr_list)

    return circuit
