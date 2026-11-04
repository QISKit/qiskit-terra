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
Tests for synthesis/xx_decompose/circuits.py .
"""

from operator import itemgetter
import unittest

import ddt
import numpy as np

from qiskit._accelerate.two_qubit_decompose import weyl_coordinates
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RZGate, UnitaryGate
import qiskit.quantum_info.operators
from qiskit.synthesis.two_qubit.xx_decompose.circuits import (
    decompose_xxyy_into_xxyy_xx,
    xx_circuit_step,
)

from .utilities import canonical_matrix

EPSILON = 0.001


@ddt.ddt
class TestMonodromyCircuits(unittest.TestCase):
    """Check circuit synthesis step routines."""

    def __init__(self, *args, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed)

    def _generate_xxyy_test_case(self):
        """
        Generates a random pentuple of values (a_source, b_source), beta, (a_target, b_target) s.t.

            CAN(a_source, b_source) * exp(-i(s ZI + t IZ)) * CAN(beta)
                                    =
            exp(-i(u ZI + v IZ)) * CAN(a_target, b_target) * exp(-i(x ZI + y IZ))

        admits a solution in (s, t, u, v, x, y).

        Returns (source_coordinate, interaction, target_coordinate).
        """
        source_coordinate = [self.rng.random(), self.rng.random(), 0.0]
        source_coordinate = [
            source_coordinate[0] * np.pi / 8,
            source_coordinate[1] * source_coordinate[0] * np.pi / 8,
            0.0,
        ]
        interaction = [self.rng.random() * np.pi / 8]
        z_angles = [self.rng.random() * np.pi / 8, self.rng.random() * np.pi / 8]
        prod = (
            canonical_matrix(*source_coordinate)
            @ np.kron(RZGate(2 * z_angles[0]).to_matrix(), RZGate(2 * z_angles[1]).to_matrix())
            @ canonical_matrix(interaction[0], 0.0, 0.0)
        )
        target_coordinate = weyl_coordinates(prod)

        self.assertAlmostEqual(target_coordinate[-1], 0.0, delta=EPSILON)

        return source_coordinate, interaction, target_coordinate

    # pylint:disable=invalid-name
    def test_decompose_xxyy(self):
        """
        Test that decompose_xxyy_into_xxyy_xx correctly recovers decompositions.
        """

        for _ in range(100):
            source_coordinate, interaction, target_coordinate = self._generate_xxyy_test_case()

            r, s, u, v, x, y = decompose_xxyy_into_xxyy_xx(
                target_coordinate[0],
                target_coordinate[1],
                source_coordinate[0],
                source_coordinate[1],
                interaction[0],
            )

            prod = (
                np.kron(RZGate(2 * r).to_matrix(), RZGate(2 * s).to_matrix())
                @ canonical_matrix(*source_coordinate)
                @ np.kron(RZGate(2 * u).to_matrix(), RZGate(2 * v).to_matrix())
                @ canonical_matrix(interaction[0], 0.0, 0.0)
                @ np.kron(RZGate(2 * x).to_matrix(), RZGate(2 * y).to_matrix())
            )
            expected = canonical_matrix(*target_coordinate)
            self.assertTrue(np.all(np.abs(prod - expected) < EPSILON))

    def test_xx_circuit_step(self):
        """
        Test that `xx_circuit_step` correctly generates prefix/affix circuits relating source
        canonical coordinates to target canonical coordinates along prescribed interactions, all
        randomly selected.
        """

        for _ in range(100):
            source_coordinate, interaction, target_coordinate = self._generate_xxyy_test_case()

            source_embodiment = qiskit.QuantumCircuit(2)
            source_embodiment.append(UnitaryGate(canonical_matrix(*source_coordinate)), [0, 1])
            interaction_embodiment = qiskit.QuantumCircuit(2)
            interaction_embodiment.append(UnitaryGate(canonical_matrix(*interaction)), [0, 1])

            prefix_circuit, affix_circuit = itemgetter("prefix_circuit", "affix_circuit")(
                xx_circuit_step(
                    source_coordinate, interaction[0], target_coordinate, interaction_embodiment
                )
            )

            target_embodiment = QuantumCircuit(2)
            target_embodiment.compose(prefix_circuit, inplace=True)
            target_embodiment.compose(source_embodiment, inplace=True)
            target_embodiment.compose(affix_circuit, inplace=True)
            self.assertTrue(
                np.all(
                    np.abs(
                        qiskit.quantum_info.operators.Operator(target_embodiment).data
                        - canonical_matrix(*target_coordinate)
                    )
                    < EPSILON
                )
            )
