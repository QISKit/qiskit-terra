# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tests analytical gradient vs the one computed via finite differences.
"""

import unittest
import numpy as np

from qiskit.synthesis.unitary.aqc.cnot_structures import make_cnot_network
from qiskit.synthesis.unitary.aqc.cnot_unit_objective import DefaultCNOTUnitObjective
from test import QiskitTestCase
from test.python.synthesis.aqc.sample_data import (
    ORIGINAL_CIRCUIT,
    INITIAL_THETAS,
)


class TestGradientAgainstFiniteDiff(QiskitTestCase):
    """
    Compares analytical gradient vs the one computed via finite difference
    approximation. Also, the test demonstrates that the difference between
    analytical and numerical gradients is up to quadratic term in Taylor
    expansion for small deltas.
    """

    def setUp(self):
        super().setUp()
        np.random.seed(0x0696969)

    def test_gradient(self):
        """
        Gradient test for specified number of qubits and circuit depth.
        """
        num_qubits = 3
        num_cnots = 14

        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=num_cnots
        )

        # we pick a target matrix from the existing sample data
        target_matrix = ORIGINAL_CIRCUIT
        objective = DefaultCNOTUnitObjective(num_qubits, cnots)
        objective.target_matrix = target_matrix

        # thetas = np.random.rand(objective.num_thetas) * (2.0 * np.pi)
        thetas = INITIAL_THETAS
        fobj0 = objective.objective(thetas)
        grad0 = objective.gradient(thetas)

        grad0_dir = grad0 / np.linalg.norm(grad0)
        numerical_grad = np.zeros(thetas.size)
        thetas_delta = np.zeros(thetas.size)

        # Every angle has a magnitude between 0 and 2*pi. We choose the
        # angle increment (tau) about the same order of magnitude, tau <= 1,
        # and then gradually decrease it towards zero.
        tau = 1.0
        diff_prev = 0.0
        orders = []
        errors = []
        steps = 9
        for step in range(steps):
            # Estimate gradient approximation error.
            for i in range(thetas.size):
                np.copyto(thetas_delta, thetas)
                thetas_delta[i] -= tau
                fobj1 = objective.objective(thetas_delta)
                np.copyto(thetas_delta, thetas)
                thetas_delta[i] += tau
                fobj2 = objective.objective(thetas_delta)
                numerical_grad[i] = (fobj2 - fobj1) / (2.0 * tau)
            errors.append(np.linalg.norm(grad0 - numerical_grad) / np.linalg.norm(grad0))

            # Estimate approximation order (should be quadratic for small tau).
            # Note, we take perturbation in gradient direction. More rigorous
            # approach would take a random direction, although quadratic
            # convergence is less pronounced in this case.
            perturbation = grad0_dir * tau
            # circuit.set_thetas(thetas + perturbation)
            fobj = objective.objective(thetas + perturbation)
            diff = abs(fobj - fobj0 - np.dot(grad0, perturbation))
            orders.append(
                0.0 if step == 0 else float((np.log(diff_prev) - np.log(diff)) / np.log(2.0))
            )

            tau /= 2.0
            diff_prev = diff

        # check errors
        prev_error = errors[0]
        for error in errors[1:]:
            self.assertLess(error, prev_error * 0.75)
            prev_error = error

        # check orders, skipping first zero
        self.assertTrue(np.count_nonzero(np.asarray(orders[1:]) > 1.8) >= 3)
        self.assertTrue(np.count_nonzero(np.asarray(orders[1:]) < 3.0) >= 3)


if __name__ == "__main__":
    unittest.main()
