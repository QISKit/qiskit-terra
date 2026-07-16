# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Implement an integer-weighted sum over a set of qubits."""

from __future__ import annotations

import typing
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.synthesis.multi_controlled import synth_mcx_n_clean_m15

if typing.TYPE_CHECKING:
    from qiskit.circuit.library import WeightedSumGate


def synth_weighted_sum_carry(weighted_sum: WeightedSumGate) -> QuantumCircuit:
    """Synthesize a weighted sum gate, by the number of state qubits and the qubit weights.

    This method is described in Appendix A of [1].

    Reference:

        [1] Stamatopoulos et al. Option Pricing using Quantum Computers (2020)
            `Quantum 4, 291 <https://doi.org/10.22331/q-2020-07-06-291>`__
    """

    num_sum_qubits = weighted_sum.num_sum_qubits
    num_state_qubits = weighted_sum.num_qubits - num_sum_qubits

    num_carry_qubits = num_sum_qubits - 1
    num_control_qubits = int(num_sum_qubits > 2)

    qr_state = QuantumRegister(num_state_qubits, "state")
    qr_sum = QuantumRegister(num_sum_qubits, "sum")
    qr_carry = AncillaRegister(num_carry_qubits, "carry")
    qr_control = AncillaRegister(num_control_qubits, "control")
    circuit = QuantumCircuit(qr_state, qr_sum, qr_carry, qr_control)

    # we use a MCX v-chain decomposition of C3X here, synthesize it once and re-use
    c3x = synth_mcx_n_clean_m15(3)

    # loop over state qubits and corresponding weights
    weights = weighted_sum.params
    for i, weight in enumerate(weights):
        # only act if non-trivial weight
        if np.isclose(weight, 0):
            continue

        # get state control qubit
        q_state = qr_state[i]

        # get bit representation of current weight
        weight_binary = f"{int(weight):b}".rjust(num_sum_qubits, "0")[::-1]

        # loop over bits of current weight and add them to sum and carry registers
        for j, bit in enumerate(weight_binary):
            if bit == "1":
                if num_sum_qubits == 1:
                    circuit.cx(q_state, qr_sum[j])
                elif j == 0:
                    # compute (q_sum[0] + 1) into (q_sum[0], q_carry[0])
                    # - controlled by q_state[i]
                    circuit.ccx(q_state, qr_sum[j], qr_carry[j])
                    circuit.cx(q_state, qr_sum[j])
                elif j == num_sum_qubits - 1:
                    # compute (q_sum[j] + q_carry[j-1] + 1) into (q_sum[j])
                    # - controlled by q_state[i] / last qubit,
                    # no carry needed by construction
                    circuit.cx(q_state, qr_sum[j])
                    circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])
                else:
                    # compute (q_sum[j] + q_carry[j-1] + 1) into (q_sum[j], q_carry[j])
                    # - controlled by q_state[i]
                    circuit.x(qr_sum[j])
                    circuit.x(qr_carry[j - 1])
                    circuit.compose(
                        c3x,
                        [q_state, qr_sum[j], qr_carry[j - 1], qr_carry[j]] + qr_control[:],
                        inplace=True,
                    )
                    circuit.cx(q_state, qr_carry[j])
                    circuit.x(qr_sum[j])
                    circuit.x(qr_carry[j - 1])
                    circuit.cx(q_state, qr_sum[j])
                    circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])
            elif num_sum_qubits == 1:
                pass  # nothing to do, since nothing to add
            elif j == 0:
                pass  # nothing to do, since nothing to add
            elif j == num_sum_qubits - 1:
                # compute (q_sum[j] + q_carry[j-1]) into (q_sum[j])
                # - controlled by q_state[i] / last qubit,
                # no carry needed by construction
                circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])
            else:
                # compute (q_sum[j] + q_carry[j-1]) into (q_sum[j], q_carry[j])
                # - controlled by q_state[i]
                circuit.compose(
                    c3x,
                    [q_state, qr_sum[j], qr_carry[j - 1], qr_carry[j]] + qr_control[:],
                    inplace=True,
                )
                circuit.ccx(q_state, qr_carry[j - 1], qr_sum[j])

        # uncompute carry qubits
        for j in reversed(range(len(weight_binary))):
            bit = weight_binary[j]
            if bit == "1":
                if num_sum_qubits == 1:
                    pass
                elif j == 0:
                    circuit.x(qr_sum[j])
                    circuit.ccx(q_state, qr_sum[j], qr_carry[j])
                    circuit.x(qr_sum[j])
                elif j == num_sum_qubits - 1:
                    pass
                else:
                    circuit.x(qr_carry[j - 1])
                    circuit.compose(
                        c3x,
                        [q_state, qr_sum[j], qr_carry[j - 1], qr_carry[j]] + qr_control[:],
                        inplace=True,
                    )
                    circuit.cx(q_state, qr_carry[j])
                    circuit.x(qr_carry[j - 1])
            elif num_sum_qubits == 1:
                pass
            elif j == 0:
                pass
            elif j == num_sum_qubits - 1:
                pass
            else:
                # compute (q_sum[j] + q_carry[j-1]) into (q_sum[j], q_carry[j])
                # - controlled by q_state[i]
                circuit.x(qr_sum[j])
                circuit.compose(
                    c3x,
                    [q_state, qr_sum[j], qr_carry[j - 1], qr_carry[j]] + qr_control[:],
                    inplace=True,
                )
                circuit.x(qr_sum[j])

    return circuit
