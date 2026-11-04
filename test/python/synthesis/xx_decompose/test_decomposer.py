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
Tests for synthesis/xx_decompose/qiskit.py .
"""

from statistics import mean
import unittest

import ddt
import numpy as np
from scipy.stats import unitary_group

import qiskit
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.two_qubit.xx_decompose.decomposer import (
    XXDecomposer,
    TwoQubitWeylDecomposition,
)

from .utilities import canonical_matrix

EPSILON = 1e-8


@ddt.ddt
class TestXXDecomposer(unittest.TestCase):
    """Tests for decomposition of two-qubit unitaries over discrete gates from XX family."""

    decomposer = XXDecomposer(euler_basis="PSX")

    def __init__(self, *args, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_state = np.random.Generator(np.random.PCG64(seed))

    def test_random_compilation(self):
        """Test that compilation gives correct results."""
        for _ in range(100):
            unitary = unitary_group.rvs(4, random_state=self._random_state)
            unitary /= np.linalg.det(unitary) ** (1 / 4)

            # decompose into CX, CX/2, and CX/3
            circuit = self.decomposer(unitary, approximate=False)
            decomposed_unitary = Operator(circuit).data

            self.assertTrue(np.all(unitary - decomposed_unitary < EPSILON))

    def test_compilation_determinism(self):
        """Test that compilation is stable under multiple calls."""
        for _ in range(10):
            unitary = unitary_group.rvs(4, random_state=self._random_state)
            unitary /= np.linalg.det(unitary) ** (1 / 4)

            # decompose into CX, CX/2, and CX/3
            circuit1 = self.decomposer(unitary, approximate=False)
            circuit2 = self.decomposer(unitary, approximate=False)

            self.assertEqual(circuit1, circuit2)

    @ddt.data(np.pi / 3, np.pi / 5, np.pi / 2)
    def test_default_embodiment(self, angle):
        """Test that _default_embodiment actually does yield XX gates."""
        embodiment = self.decomposer._default_embodiment(angle)
        embodiment_matrix = Operator(embodiment).data
        self.assertTrue(np.all(canonical_matrix(angle, 0, 0) - embodiment_matrix < EPSILON))

    def test_check_embodiment(self):
        """Test that XXDecomposer._check_embodiments correctly diagnoses il/legal embodiments."""

        # build the member of the XX family corresponding to a single CX
        good_angle = np.pi / 2
        good_embodiment = qiskit.QuantumCircuit(2)
        good_embodiment.h(0)
        good_embodiment.cx(0, 1)
        good_embodiment.h(1)
        good_embodiment.rz(np.pi / 2, 0)
        good_embodiment.rz(np.pi / 2, 1)
        good_embodiment.h(1)
        good_embodiment.h(0)
        good_embodiment.global_phase += np.pi / 4

        # mismatch two members of the XX family
        bad_angle = np.pi / 10
        bad_embodiment = qiskit.QuantumCircuit(2)

        # "self.assertDoesNotRaise"
        XXDecomposer(embodiments={good_angle: good_embodiment})

        self.assertRaises(
            qiskit.exceptions.QiskitError, XXDecomposer, embodiments={bad_angle: bad_embodiment}
        )

    def test_compilation_improvement(self):
        """Test that compilation to CX, CX/2, CX/3 improves over CX alone."""
        slope, offset = (64 * 90) / 1000000, 909 / 1000000 + 1 / 1000
        strength_table = self.decomposer._strength_to_infidelity(
            basis_fidelity={
                strength: 1 - (slope * strength / (np.pi / 2) + offset)
                for strength in [np.pi / 2, np.pi / 4, np.pi / 6]
            },
            approximate=True,
        )
        limited_strength_table = {np.pi / 2: strength_table[np.pi / 2]}

        clever_costs = []
        naive_costs = []
        for _ in range(200):
            unitary = unitary_group.rvs(4, random_state=self._random_state)
            unitary /= np.linalg.det(unitary) ** (1 / 4)

            weyl_decomposition = TwoQubitWeylDecomposition(unitary)
            target = [getattr(weyl_decomposition, x) for x in ("a", "b", "c")]
            if target[-1] < -EPSILON:
                target = [np.pi / 2 - target[0], target[1], -target[2]]

            # decompose into CX, CX/2, and CX/3
            clever_costs.append(self.decomposer._best_decomposition(target, strength_table)["cost"])
            naive_costs.append(
                self.decomposer._best_decomposition(target, limited_strength_table)["cost"]
            )

        # the following are taken from Fig 14 of the XX synthesis paper
        self.assertAlmostEqual(mean(clever_costs), 1.445e-2, delta=5e-3)
        self.assertAlmostEqual(mean(naive_costs), 2.058e-2, delta=5e-3)

    def test_error_on_empty_basis_fidelity(self):
        """Test synthesizing entangling gate with no entangling basis fails."""
        decomposer = XXDecomposer(basis_fidelity={})
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        mat = Operator(qc).to_matrix()
        with self.assertRaisesRegex(
            qiskit.exceptions.QiskitError,
            "Attempting to synthesize entangling gate with no controlled gates in basis set.",
        ):
            decomposer(mat)

    def test_no_error_on_empty_basis_fidelity_trivial_target(self):
        """Test synthesizing non-entangling gate with no entangling basis succeeds."""
        decomposer = XXDecomposer(basis_fidelity={})
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.y(1)
        mat = Operator(qc).to_matrix()
        dqc = decomposer(mat)
        self.assertTrue(np.allclose(mat, Operator(dqc).to_matrix()))
