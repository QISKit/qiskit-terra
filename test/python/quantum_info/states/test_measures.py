# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quick program to test the quantum information states modules."""

import unittest
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info import StabilizerState
from qiskit.quantum_info import purity
from qiskit.quantum_info import entropy
from qiskit.quantum_info import concurrence
from qiskit.quantum_info import entanglement_of_formation
from qiskit.quantum_info import mutual_information
from qiskit.quantum_info.states import shannon_entropy
from qiskit.quantum_info import negativity
from test import QiskitTestCase
from qiskit import QuantumCircuit


class TestStateMeasures(QiskitTestCase):
    """Tests state measure functions"""

    def test_state_fidelity_statevector(self):
        """Test state_fidelity function for statevector inputs"""

        psi1 = [0.70710678118654746, 0, 0, 0.70710678118654746]
        psi2 = [0.0, 0.70710678118654746, 0.70710678118654746, 0.0]
        self.assertAlmostEqual(state_fidelity(psi1, psi1), 1.0, places=7, msg="vector-vector input")
        self.assertAlmostEqual(state_fidelity(psi2, psi2), 1.0, places=7, msg="vector-vector input")
        self.assertAlmostEqual(state_fidelity(psi1, psi2), 0.0, places=7, msg="vector-vector input")

        psi1 = Statevector([0.70710678118654746, 0, 0, 0.70710678118654746])
        psi2 = Statevector([0.0, 0.70710678118654746, 0.70710678118654746, 0.0])
        self.assertAlmostEqual(state_fidelity(psi1, psi1), 1.0, places=7, msg="vector-vector input")
        self.assertAlmostEqual(state_fidelity(psi2, psi2), 1.0, places=7, msg="vector-vector input")
        self.assertAlmostEqual(state_fidelity(psi1, psi2), 0.0, places=7, msg="vector-vector input")

        psi1 = Statevector([1, 0, 0, 1])  # invalid state
        psi2 = Statevector([1, 0, 0, 0])
        self.assertRaises(QiskitError, state_fidelity, psi1, psi2)
        self.assertRaises(QiskitError, state_fidelity, psi1, psi2, validate=True)
        self.assertEqual(state_fidelity(psi1, psi2, validate=False), 1)

    def test_state_fidelity_stabilizer_state(self):
        """Test state_fidelity function for stabilizer state inputs."""
        state0 = StabilizerState(QuantumCircuit(1))

        qc_plus = QuantumCircuit(1)
        qc_plus.h(0)
        state_plus = StabilizerState(qc_plus)

        qc_one = QuantumCircuit(1)
        qc_one.x(0)
        state_one = StabilizerState(qc_one)

        self.assertAlmostEqual(state_fidelity(state0, state0), 1.0)
        self.assertAlmostEqual(state_fidelity(state0, state_plus), 0.5)
        self.assertAlmostEqual(state_fidelity(state0, state_one), 0.0)

    def test_state_fidelity_stabilizer_complex_state(self):
        """Test state_fidelity for nontrivial stabilizer states."""

        qc1 = QuantumCircuit(3)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.cx(1, 2)
        qc1.s(2)

        qc2 = QuantumCircuit(3)
        qc2.h(1)
        qc2.cx(1, 0)
        qc2.cx(0, 2)
        qc2.z(2)

        stabilizer1 = StabilizerState(qc1)
        stabilizer2 = StabilizerState(qc2)

        reference1 = Statevector.from_instruction(qc1)
        reference2 = Statevector.from_instruction(qc2)
        expected = state_fidelity(reference1, reference2)

        self.assertAlmostEqual(state_fidelity(stabilizer1, stabilizer2), expected)

    def test_state_fidelity_stabilizer_statevector(self):
        """Test state_fidelity for mixed StabilizerState and Statevector inputs."""
        qc_zero = QuantumCircuit(1)
        state_zero = StabilizerState(qc_zero)

        qc_plus = QuantumCircuit(1)
        qc_plus.h(0)
        state_plus = StabilizerState(qc_plus)
        statevector_plus = Statevector.from_instruction(qc_plus)

        self.assertAlmostEqual(state_fidelity(state_plus, statevector_plus), 1.0, places=7)
        self.assertAlmostEqual(state_fidelity(statevector_plus, state_plus), 1.0, places=7)
        self.assertAlmostEqual(state_fidelity(state_zero, statevector_plus), 0.5, places=7)
        self.assertAlmostEqual(state_fidelity(statevector_plus, state_zero), 0.5, places=7)

    def test_state_fidelity_stabilizer_density_matrix(self):
        """Test state_fidelity for mixed StabilizerState and DensityMatrix inputs."""
        qc_zero = QuantumCircuit(1)
        state_zero = StabilizerState(qc_zero)

        qc_plus = QuantumCircuit(1)
        qc_plus.h(0)
        state_plus = StabilizerState(qc_plus)
        density_matrix_plus = DensityMatrix(qc_plus)

        self.assertAlmostEqual(state_fidelity(state_plus, density_matrix_plus), 1.0, places=7)
        self.assertAlmostEqual(state_fidelity(density_matrix_plus, state_plus), 1.0, places=7)
        self.assertAlmostEqual(state_fidelity(state_zero, density_matrix_plus), 0.5, places=7)
        self.assertAlmostEqual(state_fidelity(density_matrix_plus, state_zero), 0.5, places=7)

    def test_state_fidelity_density_matrix(self):
        """Test state_fidelity function for density matrix inputs"""
        rho1 = [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
        mix = [[0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, 0.25]]
        self.assertAlmostEqual(state_fidelity(rho1, rho1), 1.0, places=7, msg="matrix-matrix input")
        self.assertAlmostEqual(state_fidelity(mix, mix), 1.0, places=7, msg="matrix-matrix input")
        self.assertAlmostEqual(state_fidelity(rho1, mix), 0.25, places=7, msg="matrix-matrix input")

        rho1 = DensityMatrix(rho1)
        mix = DensityMatrix(mix)
        self.assertAlmostEqual(state_fidelity(rho1, rho1), 1.0, places=7, msg="matrix-matrix input")
        self.assertAlmostEqual(state_fidelity(mix, mix), 1.0, places=7, msg="matrix-matrix input")
        self.assertAlmostEqual(state_fidelity(rho1, mix), 0.25, places=7, msg="matrix-matrix input")

        rho1 = DensityMatrix([1, 0, 0, 0])
        mix = DensityMatrix(np.diag([1, 0, 0, 1]))
        self.assertRaises(QiskitError, state_fidelity, rho1, mix)
        self.assertRaises(QiskitError, state_fidelity, rho1, mix, validate=True)
        self.assertEqual(state_fidelity(rho1, mix, validate=False), 1)

    def test_state_fidelity_mixed(self):
        """Test state_fidelity function for statevector and density matrix inputs"""
        psi1 = Statevector([0.70710678118654746, 0, 0, 0.70710678118654746])
        rho1 = DensityMatrix([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
        mix = DensityMatrix([[0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, 0.25]])
        self.assertAlmostEqual(state_fidelity(psi1, rho1), 1.0, places=7, msg="vector-matrix input")
        self.assertAlmostEqual(state_fidelity(psi1, mix), 0.25, places=7, msg="vector-matrix input")
        self.assertAlmostEqual(state_fidelity(rho1, psi1), 1.0, places=7, msg="matrix-vector input")

    def test_purity_statevector(self):
        """Test purity function on statevector inputs"""
        psi = Statevector([1, 0, 0, 0])
        self.assertEqual(purity(psi), 1)
        self.assertEqual(purity(psi, validate=True), 1)
        self.assertEqual(purity(psi, validate=False), 1)

        psi = [0.70710678118654746, 0.70710678118654746]
        self.assertAlmostEqual(purity(psi), 1)
        self.assertAlmostEqual(purity(psi, validate=True), 1)
        self.assertAlmostEqual(purity(psi, validate=False), 1)

        psi = np.array([0.5, 0.5j, -0.5j, -0.5])
        self.assertAlmostEqual(purity(psi), 1)
        self.assertAlmostEqual(purity(psi, validate=True), 1)
        self.assertAlmostEqual(purity(psi, validate=False), 1)

        psi = Statevector([1, 0, 0, 1])
        self.assertRaises(QiskitError, purity, psi)
        self.assertRaises(QiskitError, purity, psi, validate=True)
        self.assertEqual(purity(psi, validate=False), 4)

    def test_purity_density_matrix(self):
        """Test purity function on density matrix inputs"""
        rho = DensityMatrix(np.diag([1, 0, 0, 0]))
        self.assertEqual(purity(rho), 1)
        self.assertEqual(purity(rho, validate=True), 1)
        self.assertEqual(purity(rho, validate=False), 1)

        rho = np.diag([0.25, 0.25, 0.25, 0.25])
        self.assertEqual(purity(rho), 0.25)
        self.assertEqual(purity(rho, validate=True), 0.25)
        self.assertEqual(purity(rho, validate=False), 0.25)

        rho = [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
        self.assertEqual(purity(rho), 1)
        self.assertEqual(purity(rho, validate=True), 1)
        self.assertEqual(purity(rho, validate=False), 1)

        rho = np.diag([1, 0, 0, 1])
        self.assertRaises(QiskitError, purity, rho)
        self.assertRaises(QiskitError, purity, rho, validate=True)
        self.assertEqual(purity(rho, validate=False), 2)

    def test_purity_equivalence(self):
        """Test purity is same for equivalent inputs"""
        for alpha, beta in [
            (0, 0),
            (0, 0.25),
            (0.25, 0),
            (0.33, 0.33),
            (0.5, 0.5),
            (0.75, 0.25),
            (0, 0.75),
        ]:
            psi = Statevector([alpha, beta, 0, 1j * np.sqrt(1 - alpha**2 - beta**2)])
            rho = DensityMatrix(psi)
            self.assertAlmostEqual(purity(psi), purity(rho))

    def test_entropy_statevector(self):
        """Test entropy function on statevector inputs"""
        test_psis = [
            [1, 0],
            [0, 1, 0, 0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5j, -0.5j, 0.5],
            [0.70710678118654746, 0, 0, -0.70710678118654746j],
            [0.70710678118654746] + (14 * [0]) + [0.70710678118654746j],
        ]
        for psi_ls in test_psis:
            self.assertEqual(entropy(psi_ls), 0)
            self.assertEqual(entropy(np.array(psi_ls)), 0)

    def test_entropy_density_matrix(self):
        """Test entropy function on density matrix inputs"""
        # Density matrix input
        rhos = [DensityMatrix(np.diag([0.5] + (n * [0]) + [0.5])) for n in range(1, 5)]
        for rho in rhos:
            self.assertAlmostEqual(entropy(rho), 1)
            self.assertAlmostEqual(entropy(rho, base=2), 1)
            self.assertAlmostEqual(entropy(rho, base=np.e), -1 * np.log(0.5))
        # Array input
        for prob in [0.001, 0.3, 0.7, 0.999]:
            rho = np.diag([prob, 1 - prob])
            self.assertAlmostEqual(entropy(rho), shannon_entropy([prob, 1 - prob]))
            self.assertAlmostEqual(
                entropy(rho, base=np.e), shannon_entropy([prob, 1 - prob], base=np.e)
            )
            self.assertAlmostEqual(entropy(rho, base=2), shannon_entropy([prob, 1 - prob], base=2))
        # List input
        rho = [[0.5, 0], [0, 0.5]]
        self.assertAlmostEqual(entropy(rho), 1)

    def test_entropy_equivalence(self):
        """Test entropy is same for equivalent inputs"""
        for alpha, beta in [
            (0, 0),
            (0, 0.25),
            (0.25, 0),
            (0.33, 0.33),
            (0.5, 0.5),
            (0.75, 0.25),
            (0, 0.75),
        ]:
            psi = Statevector([alpha, beta, 0, 1j * np.sqrt(1 - alpha**2 - beta**2)])
            rho = DensityMatrix(psi)
            self.assertAlmostEqual(entropy(psi), entropy(rho))

    def test_concurrence_statevector(self):
        """Test concurrence function on statevector inputs"""
        # Statevector input
        psi = Statevector([0.70710678118654746, 0, 0, -0.70710678118654746j])
        self.assertAlmostEqual(concurrence(psi), 1)
        # List input
        psi = [1, 0, 0, 0]
        self.assertAlmostEqual(concurrence(psi), 0)
        # Array input
        psi = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(concurrence(psi), 0)
        # Larger than 2 qubit input
        psi_ls = [0.70710678118654746] + (14 * [0]) + [0.70710678118654746j]
        psi = Statevector(psi_ls, dims=(2, 8))
        self.assertAlmostEqual(concurrence(psi), 1)
        psi = Statevector(psi_ls, dims=(4, 4))
        self.assertAlmostEqual(concurrence(psi), 1)
        psi = Statevector(psi_ls, dims=(8, 2))
        self.assertAlmostEqual(concurrence(psi), 1)

    def test_concurrence_density_matrix(self):
        """Test concurrence function on density matrix inputs"""
        # Density matrix input
        rho1 = DensityMatrix([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
        rho2 = DensityMatrix([[0, 0, 0, 0], [0, 0.5, -0.5j, 0], [0, 0.5j, 0.5, 0], [0, 0, 0, 0]])
        self.assertAlmostEqual(concurrence(rho1), 1)
        self.assertAlmostEqual(concurrence(rho2), 1)
        self.assertAlmostEqual(concurrence(0.5 * rho1 + 0.5 * rho2), 0)
        self.assertAlmostEqual(concurrence(0.75 * rho1 + 0.25 * rho2), 0.5)
        # List input
        rho = [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.assertEqual(concurrence(rho), 0)
        # Array input
        rho = np.diag([0.25, 0.25, 0.25, 0.25])
        self.assertEqual(concurrence(rho), 0)

    def test_concurrence_equivalence(self):
        """Test concurrence is same for equivalent inputs"""
        for alpha, beta in [
            (0, 0),
            (0, 0.25),
            (0.25, 0),
            (0.33, 0.33),
            (0.5, 0.5),
            (0.75, 0.25),
            (0, 0.75),
        ]:
            psi = Statevector([alpha, beta, 0, 1j * np.sqrt(1 - alpha**2 - beta**2)])
            rho = DensityMatrix(psi)
            self.assertAlmostEqual(concurrence(psi), concurrence(rho))

    def test_entanglement_of_formation_statevector(self):
        """Test entanglement of formation function on statevector inputs"""
        # Statevector input
        psi = Statevector([0.70710678118654746, 0, 0, -0.70710678118654746j])
        self.assertAlmostEqual(entanglement_of_formation(psi), 1)
        # List input
        psi = [1, 0, 0, 0]
        self.assertAlmostEqual(entanglement_of_formation(psi), 0)
        # Array input
        psi = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(entanglement_of_formation(psi), 0)
        # Larger than 2 qubit input
        psi_ls = [0.70710678118654746] + (14 * [0]) + [0.70710678118654746j]
        psi = Statevector(psi_ls, dims=(2, 8))
        self.assertAlmostEqual(entanglement_of_formation(psi), 1)
        psi = Statevector(psi_ls, dims=(4, 4))
        self.assertAlmostEqual(entanglement_of_formation(psi), 1)
        psi = Statevector(psi_ls, dims=(8, 2))
        self.assertAlmostEqual(entanglement_of_formation(psi), 1)

    def test_entanglement_of_formation_density_matrix(self):
        """Test entanglement of formation function on density matrix inputs"""
        # Density matrix input
        rho1 = DensityMatrix([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
        rho2 = DensityMatrix([[0, 0, 0, 0], [0, 0.5, -0.5j, 0], [0, 0.5j, 0.5, 0], [0, 0, 0, 0]])
        self.assertAlmostEqual(entanglement_of_formation(rho1), 1)
        self.assertAlmostEqual(entanglement_of_formation(rho2), 1)
        self.assertAlmostEqual(entanglement_of_formation(0.5 * rho1 + 0.5 * rho2), 0)
        self.assertAlmostEqual(
            entanglement_of_formation(0.75 * rho1 + 0.25 * rho2), 0.35457890266527003
        )
        # List input
        rho = [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.assertEqual(entanglement_of_formation(rho), 0)
        # Array input
        rho = np.diag([0.25, 0.25, 0.25, 0.25])
        self.assertEqual(entanglement_of_formation(rho), 0)

    def test_entanglement_of_formation_equivalence(self):
        """Test entanglement of formation is same for equivalent inputs"""
        for alpha, beta in [
            (0, 0),
            (0, 0.25),
            (0.25, 0),
            (0.33, 0.33),
            (0.5, 0.5),
            (0.75, 0.25),
            (0, 0.75),
        ]:
            psi = Statevector([alpha, beta, 0, 1j * np.sqrt(1 - alpha**2 - beta**2)])
            rho = DensityMatrix(psi)
            self.assertAlmostEqual(entanglement_of_formation(psi), entanglement_of_formation(rho))

    def test_mutual_information_statevector(self):
        """Test mutual_information function on statevector inputs"""
        # Statevector input
        psi = Statevector([0.70710678118654746, 0, 0, -0.70710678118654746j])
        self.assertAlmostEqual(mutual_information(psi), 2)
        # List input
        psi = [1, 0, 0, 0]
        self.assertAlmostEqual(mutual_information(psi), 0)
        # Array input
        psi = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(mutual_information(psi), 0)
        # Larger than 2 qubit input
        psi_ls = [0.70710678118654746] + (14 * [0]) + [0.70710678118654746j]
        psi = Statevector(psi_ls, dims=(2, 8))
        self.assertAlmostEqual(mutual_information(psi), 2)
        psi = Statevector(psi_ls, dims=(4, 4))
        self.assertAlmostEqual(mutual_information(psi), 2)
        psi = Statevector(psi_ls, dims=(8, 2))
        self.assertAlmostEqual(mutual_information(psi), 2)

    def test_mutual_information_density_matrix(self):
        """Test mutual_information  function on density matrix inputs"""
        # Density matrix input
        rho1 = DensityMatrix([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
        rho2 = DensityMatrix([[0, 0, 0, 0], [0, 0.5, -0.5j, 0], [0, 0.5j, 0.5, 0], [0, 0, 0, 0]])
        self.assertAlmostEqual(mutual_information(rho1), 2)
        self.assertAlmostEqual(mutual_information(rho2), 2)
        # List input
        rho = [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.assertEqual(mutual_information(rho), 0)
        # Array input
        rho = np.diag([0.25, 0.25, 0.25, 0.25])
        self.assertEqual(mutual_information(rho), 0)

    def test_mutual_information_equivalence(self):
        """Test mutual_information is same for equivalent inputs"""
        for alpha, beta in [
            (0, 0),
            (0, 0.25),
            (0.25, 0),
            (0.33, 0.33),
            (0.5, 0.5),
            (0.75, 0.25),
            (0, 0.75),
        ]:
            psi = Statevector([alpha, beta, 0, 1j * np.sqrt(1 - alpha**2 - beta**2)])
            rho = DensityMatrix(psi)
            self.assertAlmostEqual(mutual_information(psi), mutual_information(rho))

    def test_negativity_statevector(self):
        """Test negativity function on statevector inputs"""
        # Constructing separable quantum statevector
        state = Statevector([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])
        negv = negativity(state, [0])
        self.assertAlmostEqual(negv, 0, places=7)
        # Constructing entangled quantum statevector
        state = Statevector([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0])
        negv = negativity(state, [1])
        self.assertAlmostEqual(negv, 0.5, places=7)

    def test_negativity_density_matrix(self):
        """Test negativity function on density matrix inputs"""
        # Constructing separable quantum state
        rho = DensityMatrix.from_label("10+")
        negv = negativity(rho, [0, 1])
        self.assertAlmostEqual(negv, 0, places=7)
        negv = negativity(rho, [0, 2])
        self.assertAlmostEqual(negv, 0, places=7)
        # Constructing entangled quantum state
        rho = DensityMatrix([[0, 0, 0, 0], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [0, 0, 0, 0]])
        negv = negativity(rho, [0])
        self.assertAlmostEqual(negv, 0.5, places=7)
        negv = negativity(rho, [1])
        self.assertAlmostEqual(negv, 0.5, places=7)


if __name__ == "__main__":
    unittest.main()
