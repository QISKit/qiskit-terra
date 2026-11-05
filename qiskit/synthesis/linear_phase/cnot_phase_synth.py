# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Implementation of the GraySynth algorithm for synthesizing CNOT-Phase
circuits with efficient CNOT cost, and the Patel-Hayes-Markov algorithm
for optimal synthesis of linear (CNOT-only) reversible circuits.
"""
from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit._accelerate.synthesis.linear_phase import (
    synth_cnot_phase_aam as synth_cnot_phase_aam_xlated,
)


def synth_cnot_phase_aam(
    cnots: list[list[int]], angles: list[str], section_size: int | None = 2
) -> QuantumCircuit:
    r"""This function is an implementation of the `GraySynth` algorithm of
    Amy, Azimadeh and Mosca.

    GraySynth is a heuristic algorithm from [1] for synthesizing small parity networks.
    It is inspired by Gray codes. Given a set of binary strings :math:`S`
    (called ``cnots`` below), the algorithm synthesizes a parity network for :math:`S` by
    repeatedly choosing an index :math:`i` to expand and then effectively recursing on
    the co-factors :math:`S_0` and :math:`S_1`, consisting of the strings :math:`y \in S`,
    with :math:`y_i = 0` or :math:`1` respectively. As a subset :math:`S` is recursively expanded,
    ``cx`` gates are applied so that a designated target bit contains the
    (partial) parity :math:`\chi_y(x)` where :math:`y_i = 1` if and only if :math:`y'_i = 1` for all
    :math:`y' \in S`. If :math:`S` contains a single element :math:`\{y'\}`, then :math:`y = y'`,
    and the target bit contains the value :math:`\chi_{y'}(x)` as desired.

    Notably, rather than uncomputing this sequence of ``cx`` (CNOT) gates when a subset :math:`S`
    is finished being synthesized, the algorithm maintains the invariant
    that the remaining parities to be computed are expressed over the current state
    of bits. This allows the algorithm to avoid the 'backtracking' inherent in
    uncomputing-based methods.

    The algorithm is described in detail in section 4 of [1].

    Args:
        cnots: A matrix whose columns are the parities to be synthesized
            e.g.::

                [[0, 1, 1, 1, 1, 1],
                 [1, 0, 0, 1, 1, 1],
                 [1, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 1, 0]]

            corresponds to::

                 x1^x2 + x0 + x0^x3 + x0^x1^x2 + x0^x1^x3 + x0^x1

        angles: A list containing all the phase-shift gates which are
            to be applied, in the same order as in ``cnots``. A number is
            interpreted as the angle of :math:`p(angle)`, otherwise the elements
            have to be ``'t'``, ``'tdg'``, ``'s'``, ``'sdg'`` or ``'z'``.

        section_size: The size of every section in the Patel–Markov–Hayes algorithm.
            ``section_size`` must be a factor of the number of qubits.

    Returns:
        The decomposed quantum circuit.

    Raises:
        QiskitError: when dimensions of ``cnots`` and ``angles`` don't align or when an angle is
            neither a floating-point value, nor one of the supported gates
            't', 'tdg', 's', 'sdg' or 'z'.

    References:
        1. Matthew Amy, Parsiad Azimzadeh, and Michele Mosca.
           *On the controlled-NOT complexity of controlled-NOT–phase circuits.*,
           Quantum Science and Technology 4.1 (2018): 015002.
           `arXiv:1712.01859 <https://arxiv.org/abs/1712.01859>`_
    """

    cnots_array = np.asarray(cnots).astype(bool)
    _circuit_data = synth_cnot_phase_aam_xlated(cnots_array, angles, section_size)
    return QuantumCircuit._from_circuit_data(_circuit_data, legacy_qubits=True)
