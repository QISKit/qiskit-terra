# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Integer comparator based on 2s complement."""

import math

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import OrGate


def synth_integer_comparator_2s(
    num_state_qubits: int, value: int, geq: bool = True
) -> QuantumCircuit:
    r"""Implement an integer comparison based on 2s complement.

    This is based on Appendix B of [1].

    Args:
        num_state_qubits: The number of qubits encoding the value to compare to.
        value: The value to compare to.
        geq: If ``True`` flip the target bit if the qubit state is :math:`\geq` than the value,
            otherwise implement :math:`<`.

    Returns:
        A circuit implementing the integer comparator.

    References:

        [1] J. Gacon et al. "Quantum-enhanced simulation-based optimization"
            `arXiv:2005.10780 <https://arxiv.org/abs/2005.10780>`__.
    """
    circuit = QuantumCircuit(2 * num_state_qubits)
    qr_state = circuit.qubits[:num_state_qubits]
    q_compare = circuit.qubits[num_state_qubits]
    qr_ancilla = circuit.qubits[num_state_qubits + 1 :]

    if value <= 0:  # condition always satisfied for non-positive values
        if geq:  # otherwise the condition is never satisfied
            circuit.x(q_compare)
    # condition never satisfied for values larger than or equal to 2^n
    elif value < pow(2, num_state_qubits):

        if num_state_qubits > 1:
            twos = _get_twos_complement(num_state_qubits, value)
            for i in range(num_state_qubits):
                if i == 0:
                    if twos[i] == 1:
                        circuit.cx(qr_state[i], qr_ancilla[i])
                elif i < num_state_qubits - 1:
                    if twos[i] == 1:
                        circuit.append(OrGate(2), [qr_state[i], qr_ancilla[i - 1], qr_ancilla[i]])
                    else:
                        circuit.ccx(qr_state[i], qr_ancilla[i - 1], qr_ancilla[i])
                elif twos[i] == 1:
                    # OR needs the result argument as qubit not register, thus
                    # access the index [0]
                    circuit.append(OrGate(2), [qr_state[i], qr_ancilla[i - 1], q_compare])
                else:
                    circuit.ccx(qr_state[i], qr_ancilla[i - 1], q_compare)

            # flip result bit if geq flag is false
            if not geq:
                circuit.x(q_compare)

            # uncompute ancillas state
            for i in reversed(range(num_state_qubits - 1)):
                if i == 0:
                    if twos[i] == 1:
                        circuit.cx(qr_state[i], qr_ancilla[i])
                elif twos[i] == 1:
                    circuit.append(OrGate(2), [qr_state[i], qr_ancilla[i - 1], qr_ancilla[i]])
                else:
                    circuit.ccx(qr_state[i], qr_ancilla[i - 1], qr_ancilla[i])
        else:

            # num_state_qubits == 1 and value == 1:
            circuit.cx(qr_state[0], q_compare)

            # flip result bit if geq flag is false
            if not geq:
                circuit.x(q_compare)

    elif not geq:  # otherwise the condition is never satisfied
        circuit.x(q_compare)

    return circuit


def _get_twos_complement(num_bits: int, value: int) -> list[int]:
    """Returns the 2's complement of ``self.value`` as array.

    Returns:
            The 2's complement of ``self.value``.
    """
    twos_complement = pow(2, num_bits) - math.ceil(value)
    twos_complement = f"{twos_complement:b}".rjust(num_bits, "0")
    twos_complement = [
        1 if twos_complement[i] == "1" else 0 for i in reversed(range(len(twos_complement)))
    ]
    return twos_complement
