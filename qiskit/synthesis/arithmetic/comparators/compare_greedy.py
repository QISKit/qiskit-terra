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

"""Integer comparator based on an exponential number of multi-controlled gates."""

import numpy as np

from qiskit.circuit import QuantumCircuit


def synth_integer_comparator_greedy(
    num_state_qubits: int, value: int, geq: bool = True
) -> QuantumCircuit:
    r"""Implement an integer comparison based on value-by-value comparison.

    For ``value`` smaller than ``2 ** (num_state_qubits - 1)`` this circuit implements
    ``value`` multi-controlled gates with control states 0, 1, ..., ``value - 1``, such that
    the target qubit is flipped if the qubit state represents any of the allowed values.
    For ``value`` larger than that, ``geq`` is flipped. This implementation can
    require an exponential number of gates. If auxiliary qubits are available, the implementation
    provided by :func:`.synth_integer_comparator_2s` is more efficient.

    Args:
        num_state_qubits: The number of qubits encoding the value to compare to.
        value: The value to compare to.
        geq: If ``True`` flip the target bit if the qubit state is :math:`\geq` than the value,
            otherwise implement :math:`<`.

    Returns:
        A circuit implementing the integer comparator.

    """
    circuit = QuantumCircuit(num_state_qubits + 1)

    if value <= 0:  # condition always satisfied for non-positive values
        if geq:  # otherwise the condition is never satisfied
            circuit.x(num_state_qubits)

        return circuit

    # make sure to always choose the comparison where we have to place less than
    # (2 ** n)/2 MCX gates
    value = int(np.ceil(value))
    if (value < 2 ** (num_state_qubits - 1) and geq) or (
        value > 2 ** (num_state_qubits - 1) and not geq
    ):
        geq = not geq
        circuit.x(num_state_qubits)

    if geq:
        accepted_values = range(value, 2**num_state_qubits)
    else:
        accepted_values = range(value)

    for accepted_value in accepted_values:
        circuit.mcx(list(range(num_state_qubits)), num_state_qubits, ctrl_state=accepted_value)

    return circuit
