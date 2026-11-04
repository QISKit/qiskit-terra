# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Stabilizer to circuit function
"""
from __future__ import annotations

from collections.abc import Collection

import numpy as np

from qiskit.quantum_info import PauliList, Clifford
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit


def synth_circuit_from_stabilizers(
    stabilizers: Collection[str],
    allow_redundant: bool = False,
    allow_underconstrained: bool = False,
    invert: bool = False,
) -> QuantumCircuit:
    """Synthesis of a circuit that generates a state stabilized by the stabilizers
    using Gaussian elimination with Clifford gates.
    If the stabilizers are underconstrained, and ``allow_underconstrained`` is ``True``,
    the circuit will output one of the states stabilized by the stabilizers.
    Based on stim implementation.

    Args:
        stabilizers: List of stabilizer strings
        allow_redundant: Allow redundant stabilizers (i.e., some stabilizers
            can be products of the others)
        allow_underconstrained: Allow underconstrained set of stabilizers (i.e.,
            the stabilizers do not specify a unique state)
        invert: Return inverse circuit

    Returns:
        A circuit that generates a state stabilized by ``stabilizers``.

    Raises:
        QiskitError: if the stabilizers are invalid, do not commute, or contradict each other,
                     if the list is underconstrained and ``allow_underconstrained`` is ``False``,
                     or if the list is redundant and ``allow_redundant`` is ``False``.

    References:
        1. https://github.com/quantumlib/Stim/blob/c0dd0b1c8125b2096cd54b6f72884a459e47fe3e/src/stim/stabilizers/conversions.inl#L469
        2. https://quantumcomputing.stackexchange.com/questions/12721/how-to-calculate-destabilizer-group-of-toric-and-other-codes

    """
    stabilizer_list = PauliList(stabilizers)
    if np.any(stabilizer_list.phase % 2):
        raise QiskitError("Some stabilizers have an invalid phase")
    if len(stabilizer_list.commutes_with_all(stabilizer_list)) < len(stabilizer_list):
        raise QiskitError("Some stabilizers do not commute.")

    num_qubits = stabilizer_list.num_qubits
    circuit = QuantumCircuit(num_qubits)

    used = 0
    for i, stabilizer in enumerate(stabilizer_list):
        curr_stab = stabilizer.evolve(Clifford(circuit), frame="s")

        # Find pivot.
        pivot = used
        while pivot < num_qubits:
            if curr_stab[pivot].x or curr_stab[pivot].z:
                break
            pivot += 1

        if pivot == num_qubits:
            if curr_stab.x.any():
                raise QiskitError(
                    f"Stabilizer {i} ({stabilizer}) anti-commutes with some of "
                    "the previous stabilizers."
                )
            if curr_stab.phase == 2:
                raise QiskitError(
                    f"Stabilizer {i} ({stabilizer}) contradicts "
                    "some of the previous stabilizers."
                )
            if curr_stab.z.any() and not allow_redundant:
                raise QiskitError(
                    f"Stabilizer {i} ({stabilizer}) is a product of the others "
                    "and allow_redundant is False. Add allow_redundant=True "
                    "to the function call if you want to allow redundant stabilizers."
                )
            continue

        # Change pivot basis to the Z axis.
        if curr_stab[pivot].x:
            if curr_stab[pivot].z:
                circuit.h(pivot)
                circuit.s(pivot)
                circuit.h(pivot)
                circuit.s(pivot)
                circuit.s(pivot)
            else:
                circuit.h(pivot)

        # Cancel other terms in Pauli string.
        for j in range(num_qubits):
            if j == pivot or not (curr_stab[j].x or curr_stab[j].z):
                continue
            p = curr_stab[j].x + curr_stab[j].z * 2
            if p == 1:  # X
                circuit.h(pivot)
                circuit.cx(pivot, j)
                circuit.h(pivot)
            elif p == 2:  # Z
                circuit.cx(j, pivot)
            elif p == 3:  # Y
                circuit.h(pivot)
                circuit.s(j)
                circuit.s(j)
                circuit.s(j)
                circuit.cx(pivot, j)
                circuit.h(pivot)
                circuit.s(j)

        # Move pivot to diagonal.
        if pivot != used:
            circuit.swap(pivot, used)

        # fix sign
        curr_stab = stabilizer.evolve(Clifford(circuit), frame="s")
        if curr_stab.phase == 2:
            circuit.x(used)
        used += 1

    if used < num_qubits and not allow_underconstrained:
        raise QiskitError(
            "Stabilizers are underconstrained and allow_underconstrained is False."
            " Add allow_underconstrained=True  to the function call "
            "if you want to allow underconstrained stabilizers."
        )
    if invert:
        return circuit
    return circuit.inverse()
