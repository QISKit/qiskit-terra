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
Tools for building optimal circuits out of XX interactions.

Inputs:
 + A set of native XX operations, described as strengths.
 + A right-angled path, computed using the methods in `paths.py`.

Output:
 + A circuit which implements the target operation (expressed exactly as the exponential of
 `a XX + b YY + c ZZ`) using the native operations and local gates.
"""

from __future__ import annotations
import cmath
from functools import reduce
import math
from operator import itemgetter

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate, RZGate
from qiskit.exceptions import QiskitError

from .paths import decomposition_hop
from .utilities import EPSILON, safe_arccos
from .weyl import (
    apply_reflection,
    apply_shift,
    canonical_rotation_circuit,
    reflection_options,
    shift_options,
)


def decompose_xxyy_into_xxyy_xx(a_target, b_target, a_source, b_source, interaction):
    """
    Consumes a target canonical interaction CAN(a_target, b_target) and source interactions
    CAN(a1, b1), CAN(a2), then manufactures a circuit identity of the form

    CAN(a_target, b_target) = (Zr, Zs) CAN(a_source, b_source) (Zu, Zv) CAN(interaction) (Zx, Zy).

    Returns the 6-tuple (r, s, u, v, x, y).
    """

    cplus, cminus = math.cos(a_source + b_source), math.cos(a_source - b_source)
    splus, sminus = math.sin(a_source + b_source), math.sin(a_source - b_source)
    ca, sa = np.cos(interaction), np.sin(interaction)

    uplusv = (
        1
        / 2
        * safe_arccos(
            cminus**2 * ca**2 + sminus**2 * sa**2 - np.cos(a_target - b_target) ** 2,
            2 * cminus * ca * sminus * sa,
        )
    )
    uminusv = (
        1
        / 2
        * safe_arccos(
            cplus**2 * ca**2 + splus**2 * sa**2 - np.cos(a_target + b_target) ** 2,
            2 * cplus * ca * splus * sa,
        )
    )

    u, v = (uplusv + uminusv) / 2, (uplusv - uminusv) / 2

    # NOTE: the target matrix is phase-free
    middle_matrix = reduce(
        np.dot,
        [
            RXXGate(2 * a_source).to_matrix() @ RYYGate(2 * b_source).to_matrix(),
            np.kron(RZGate(2 * u).to_matrix(), RZGate(2 * v).to_matrix()),
            RXXGate(2 * interaction).to_matrix(),
        ],
    )

    phase_solver = np.array(
        [
            [
                1 / 4,
                1 / 4,
                1 / 4,
                1 / 4,
            ],
            [
                1 / 4,
                -1 / 4,
                -1 / 4,
                1 / 4,
            ],
            [
                1 / 4,
                1 / 4,
                -1 / 4,
                -1 / 4,
            ],
            [
                1 / 4,
                -1 / 4,
                1 / 4,
                -1 / 4,
            ],
        ]
    )
    inner_phases = [
        cmath.phase(middle_matrix[0, 0]),
        cmath.phase(middle_matrix[1, 1]),
        cmath.phase(middle_matrix[1, 2]) + np.pi / 2,
        cmath.phase(middle_matrix[0, 3]) + np.pi / 2,
    ]
    r, s, x, y = np.dot(phase_solver, inner_phases)

    # If there's a phase discrepancy, need to conjugate by an extra Z/2 (x) Z/2.
    generated_matrix = reduce(
        np.dot,
        [
            np.kron(RZGate(2 * r).to_matrix(), RZGate(2 * s).to_matrix()),
            middle_matrix,
            np.kron(RZGate(2 * x).to_matrix(), RZGate(2 * y).to_matrix()),
        ],
    )
    if (abs(cmath.phase(generated_matrix[3, 0]) - np.pi / 2) < 0.01 and a_target > b_target) or (
        abs(cmath.phase(generated_matrix[3, 0]) + np.pi / 2) < 0.01 and a_target < b_target
    ):
        x += np.pi / 4
        y += np.pi / 4
        r -= np.pi / 4
        s -= np.pi / 4

    return r, s, u, v, x, y


def xx_circuit_step(source, strength, target, embodiment):
    """
    Builds a single step in an XX-based circuit.

    `source` and `target` are positive canonical coordinates; `strength` is the interaction strength
    at this step in the circuit as a canonical coordinate (so that CX = RZX(pi/2) corresponds to
    pi/4); and `embodiment` is a Qiskit circuit which enacts the canonical gate of the prescribed
    interaction `strength`.
    """

    permute_source_for_overlap, permute_target_for_overlap = None, None

    # apply all possible reflections, shifts to the source
    for source_reflection_name in reflection_options:
        reflected_source_coord, source_reflection, reflection_phase_shift = apply_reflection(
            source_reflection_name, source
        )
        for source_shift_name in shift_options:
            shifted_source_coord, source_shift, shift_phase_shift = apply_shift(
                source_shift_name, reflected_source_coord
            )

            # check for overlap, back out permutation
            source_shared, target_shared = None, None
            for i, j in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:

                if (
                    abs(np.mod(abs(shifted_source_coord[i] - target[j]), np.pi)) < EPSILON
                    or abs(np.mod(abs(shifted_source_coord[i] - target[j]), np.pi) - np.pi)
                    < EPSILON
                ):
                    source_shared, target_shared = i, j
                    break
            if source_shared is None:
                continue

            # pick out the other coordinates
            source_first, source_second = (x for x in [0, 1, 2] if x != source_shared)
            target_first, target_second = (x for x in [0, 1, 2] if x != target_shared)

            # check for arccos validity
            r, s, u, v, x, y = decompose_xxyy_into_xxyy_xx(
                float(target[target_first]),
                float(target[target_second]),
                float(shifted_source_coord[source_first]),
                float(shifted_source_coord[source_second]),
                float(strength),
            )
            if any(math.isnan(val) for val in (r, s, u, v, x, y)):
                continue

            # OK: this combination of things works.
            # save the permutation which rotates the shared coordinate into ZZ.
            permute_source_for_overlap = canonical_rotation_circuit(source_first, source_second)
            permute_target_for_overlap = canonical_rotation_circuit(target_first, target_second)
            break

        if permute_source_for_overlap is not None:
            break

    if permute_source_for_overlap is None:
        raise QiskitError(
            "Error during RZX decomposition: Could not find a suitable Weyl "
            f"reflection to match {source} to {target} along {strength}."
        )

    prefix_circuit, affix_circuit = QuantumCircuit(2), QuantumCircuit(2)

    # the basic formula we're trying to work with is:
    # target^p_t_f_o =
    #     rs * (source^s_reflection * s_shift)^p_s_f_o * uv * operation * xy
    # but we're rearranging it into the form
    #   target = affix source prefix
    # and computing just the prefix / affix circuits.

    # the outermost prefix layer comes from the (inverse) target permutation.
    prefix_circuit.compose(permute_target_for_overlap.inverse(), inplace=True)
    # the middle prefix layer comes from the local Z rolls.
    prefix_circuit.rz(2 * x, [0])
    prefix_circuit.rz(2 * y, [1])
    prefix_circuit.compose(embodiment, inplace=True)
    prefix_circuit.rz(2 * u, [0])
    prefix_circuit.rz(2 * v, [1])
    # the innermost prefix layer is source_reflection, shifted by source_shift,
    # finally conjugated by p_s_f_o.
    prefix_circuit.compose(permute_source_for_overlap, inplace=True)
    prefix_circuit.compose(source_reflection, inplace=True)
    prefix_circuit.global_phase -= cmath.phase(reflection_phase_shift)
    prefix_circuit.global_phase -= cmath.phase(shift_phase_shift)

    # the affix circuit is constructed in reverse.
    # first (i.e., innermost), we install the other half of the source transformations and p_s_f_o.
    affix_circuit.compose(source_reflection.inverse(), inplace=True)
    affix_circuit.compose(source_shift, inplace=True)
    affix_circuit.compose(permute_source_for_overlap.inverse(), inplace=True)
    # then, the other local rolls in the middle.
    affix_circuit.rz(2 * r, [0])
    affix_circuit.rz(2 * s, [1])
    # finally, the other half of the p_t_f_o conjugation.
    affix_circuit.compose(permute_target_for_overlap, inplace=True)

    return {"prefix_circuit": prefix_circuit, "affix_circuit": affix_circuit}


def canonical_xx_circuit(target, strength_sequence, basis_embodiments):
    """
    Assembles a Qiskit circuit from a specified `strength_sequence` of XX-type interactions which
    emulates the canonical gate at canonical coordinate `target`.  The circuits supplied by
    `basis_embodiments` are used to instantiate the individual XX actions.

    NOTE: The elements of `strength_sequence` are expected to be normalized so that np.pi/2
        corresponds to RZX(np.pi/2) = CX; `target` is taken to be a positive canonical coordinate;
        and `basis_embodiments` maps `strength_sequence` elements to circuits which instantiate
        these gates.
    """
    # empty decompositions are easy!
    if len(strength_sequence) == 0:
        return QuantumCircuit(2)

    # assemble the prefix / affix circuits
    prefix_circuit, affix_circuit = QuantumCircuit(2), QuantumCircuit(2)
    while len(strength_sequence) > 1:
        source = decomposition_hop(target, strength_sequence)
        strength = strength_sequence[-1]

        preceding_prefix_circuit, preceding_affix_circuit = itemgetter(
            "prefix_circuit", "affix_circuit"
        )(xx_circuit_step(source, strength / 2, target, basis_embodiments[strength]))

        prefix_circuit.compose(preceding_prefix_circuit, inplace=True)
        affix_circuit.compose(preceding_affix_circuit, inplace=True, front=True)

        target, strength_sequence = source, strength_sequence[:-1]

    circuit = prefix_circuit

    # lastly, deal with the "leading" gate.
    if target[0] <= np.pi / 4:
        circuit.compose(basis_embodiments[strength_sequence[0]], inplace=True)
    else:
        _, source_reflection, reflection_phase_shift = apply_reflection("reflect XX, YY", [0, 0, 0])
        _, source_shift, shift_phase_shift = apply_shift("X shift", [0, 0, 0])

        circuit.compose(source_reflection, inplace=True)
        circuit.compose(basis_embodiments[strength_sequence[0]], inplace=True)
        circuit.compose(source_reflection.inverse(), inplace=True)
        circuit.compose(source_shift, inplace=True)
        circuit.global_phase -= cmath.phase(shift_phase_shift)
        circuit.global_phase -= cmath.phase(reflection_phase_shift)

    circuit.compose(affix_circuit, inplace=True)

    return circuit
