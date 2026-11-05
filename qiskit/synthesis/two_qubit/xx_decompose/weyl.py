# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Simple circuit constructors for Weyl reflections.
"""

from __future__ import annotations
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate


reflection_options = {
    "no reflection": ([1, 1, 1], 1, []),
    "reflect XX, YY": ([-1, -1, 1], 1, [RZGate]),
    "reflect XX, ZZ": ([-1, 1, -1], 1, [RYGate]),
    "reflect YY, ZZ": ([1, -1, -1], 1, [RXGate]),
}
"""
A table of available reflection transformations on canonical coordinates.
Entries take the form

    readable_name: (reflection scalars, global phase, [gate constructors]),

where reflection scalars (a, b, c) model the map (x, y, z) |-> (ax, by, cz),
global phase is a complex unit, and gate constructors are applied in sequence
and by conjugation to the first qubit and are passed pi as a parameter.
"""

shift_options = {
    "no shift": ([0, 0, 0], 1, []),
    "Z shift": ([0, 0, 1], 1j, [RZGate]),
    "Y shift": ([0, 1, 0], -1j, [RYGate]),
    "Y,Z shift": ([0, 1, 1], 1, [RYGate, RZGate]),
    "X shift": ([1, 0, 0], -1j, [RXGate]),
    "X,Z shift": ([1, 0, 1], 1, [RXGate, RZGate]),
    "X,Y shift": ([1, 1, 0], -1, [RXGate, RYGate]),
    "X,Y,Z shift": ([1, 1, 1], -1j, [RXGate, RYGate, RZGate]),
}
"""
A table of available shift transformations on canonical coordinates.  Entries
take the form

    readable name: (shift scalars, global phase, [gate constructors]),

where shift scalars model the map

    (x, y, z) |-> (x + a pi / 2, y + b pi / 2, z + c pi / 2) ,

global phase is a complex unit, and gate constructors are applied to the first
and second qubits and are passed pi as a parameter.
"""


def apply_reflection(reflection_name, coordinate):
    """
    Given a reflection type and a canonical coordinate, applies the reflection
    and describes a circuit which enacts the reflection + a global phase shift.
    """
    reflection_scalars, reflection_phase_shift, source_reflection_gates = reflection_options[
        reflection_name
    ]
    reflected_coord = [x * y for x, y in zip(reflection_scalars, coordinate)]
    source_reflection = QuantumCircuit(2)
    for gate in source_reflection_gates:
        source_reflection.append(gate(np.pi), [0])

    return reflected_coord, source_reflection, reflection_phase_shift


def apply_shift(shift_name, coordinate):
    """
    Given a shift type and a canonical coordinate, applies the shift and
    describes a circuit which enacts the shift + a global phase shift.
    """
    shift_scalars, shift_phase_shift, source_shift_gates = shift_options[shift_name]
    shifted_coord = [np.pi / 2 * x + y for x, y in zip(shift_scalars, coordinate)]

    source_shift = QuantumCircuit(2)
    for gate in source_shift_gates:
        source_shift.append(gate(np.pi), [0])
        source_shift.append(gate(np.pi), [1])

    return shifted_coord, source_shift, shift_phase_shift


def canonical_rotation_circuit(first_index, second_index):
    """
    Given a pair of distinct indices 0 ≤ (first_index, second_index) ≤ 2,
    produces a two-qubit circuit which rotates a canonical gate

        a0 XX + a1 YY + a2 ZZ

    into

        a[first] XX + a[second] YY + a[other] ZZ .
    """
    conj = QuantumCircuit(2)

    if (0, 1) == (first_index, second_index):
        pass  # no need to do anything
    elif (0, 2) == (first_index, second_index):
        conj.rx(-np.pi / 2, [0])
        conj.rx(np.pi / 2, [1])
    elif (1, 0) == (first_index, second_index):
        conj.rz(-np.pi / 2, [0])
        conj.rz(-np.pi / 2, [1])
    elif (1, 2) == (first_index, second_index):
        conj.rz(np.pi / 2, [0])
        conj.rz(np.pi / 2, [1])
        conj.ry(np.pi / 2, [0])
        conj.ry(-np.pi / 2, [1])
    elif (2, 0) == (first_index, second_index):
        conj.rz(np.pi / 2, [0])
        conj.rz(np.pi / 2, [1])
        conj.rx(np.pi / 2, [0])
        conj.rx(-np.pi / 2, [1])
    elif (2, 1) == (first_index, second_index):
        conj.ry(np.pi / 2, [0])
        conj.ry(-np.pi / 2, [1])

    return conj
