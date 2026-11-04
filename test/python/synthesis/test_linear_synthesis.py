# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test linear reversible circuits synthesis functions."""

import unittest

import numpy as np
from ddt import ddt, data
from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearFunction
from qiskit.synthesis.linear import (
    synth_cnot_count_full_pmh,
    synth_cnot_depth_line_kms,
    random_invertible_binary_matrix,
    check_invertible_binary_matrix,
    calc_inverse_matrix,
    binary_matmul,
)
from qiskit.synthesis.linear.linear_circuits_utils import transpose_cx_circ, optimize_cx_4_options
from test import QiskitTestCase


@ddt
class TestLinearSynth(QiskitTestCase):
    """Test the linear reversible circuit synthesis functions."""

    def test_lnn_circuit(self):
        """Test the synthesis of a CX circuit with LNN connectivity."""

        n = 5
        qc = QuantumCircuit(n)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        mat = LinearFunction(qc).linear

        for optimized in [True, False]:
            optimized_qc = optimize_cx_4_options(
                synth_cnot_count_full_pmh, mat, optimize_count=optimized
            )
            self.assertEqual(optimized_qc.depth(), 4)
            self.assertEqual(optimized_qc.count_ops()["cx"], 4)

    def test_full_circuit(self):
        """Test the synthesis of a CX circuit with full connectivity."""

        n = 5
        qc = QuantumCircuit(n)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j)
        mat = LinearFunction(qc).linear

        for optimized in [True, False]:
            optimized_qc = optimize_cx_4_options(
                synth_cnot_count_full_pmh, mat, optimize_count=optimized
            )
            self.assertEqual(optimized_qc.depth(), 4)
            self.assertEqual(optimized_qc.count_ops()["cx"], 4)

    def test_transpose_circ(self):
        """Test the transpose_cx_circ() function."""
        n = 5
        mat = random_invertible_binary_matrix(n, seed=1234)
        qc = synth_cnot_count_full_pmh(mat)
        transposed_qc = transpose_cx_circ(qc)
        transposed_mat = LinearFunction(transposed_qc).linear.astype(int)
        self.assertTrue((mat.transpose() == transposed_mat).all())

    def test_example_circuit(self):
        """Test the synthesis of an example CX circuit which provides different CX count
        and depth for different optimization methods."""

        qc = QuantumCircuit(9)
        qc.swap(8, 7)
        qc.swap(7, 6)
        qc.cx(5, 6)
        qc.cx(6, 5)
        qc.swap(4, 5)
        qc.cx(3, 4)
        qc.cx(4, 3)
        qc.swap(2, 3)
        qc.cx(1, 2)
        qc.cx(2, 1)
        qc.cx(0, 1)
        qc.cx(1, 0)
        mat = LinearFunction(qc).linear

        optimized_qc = optimize_cx_4_options(synth_cnot_count_full_pmh, mat, optimize_count=True)
        self.assertEqual(optimized_qc.depth(), 17)
        self.assertEqual(optimized_qc.count_ops()["cx"], 20)

        optimized_qc = optimize_cx_4_options(synth_cnot_count_full_pmh, mat, optimize_count=False)
        self.assertEqual(optimized_qc.depth(), 15)
        self.assertEqual(optimized_qc.count_ops()["cx"], 23)

    @data(5, 6)
    def test_invertible_matrix(self, n):
        """Test the functions for generating a random invertible matrix and inverting it."""
        mat = random_invertible_binary_matrix(n, seed=1234)
        out = check_invertible_binary_matrix(mat)
        mat = mat.astype(bool)
        mat_inv = calc_inverse_matrix(mat, verify=True)
        mat_out = binary_matmul(mat, mat_inv)
        self.assertTrue(np.array_equal(mat_out, np.eye(n)))
        self.assertTrue(out)

    @data(5, 6)
    def test_synth_lnn_kms(self, num_qubits):
        """Test that synth_cnot_depth_line_kms produces the correct synthesis."""
        rng = np.random.default_rng(1234)
        num_trials = 10
        seeds = rng.integers(100000, size=num_trials, dtype=np.uint64)
        for seed in seeds:
            mat = random_invertible_binary_matrix(num_qubits, seed=seed)
            mat = np.array(mat, dtype=bool)
            qc = synth_cnot_depth_line_kms(mat)
            mat1 = LinearFunction(qc).linear
            self.assertTrue((mat == mat1).all())

            # Check that the circuit depth is bounded by 5*num_qubits
            depth = qc.depth()
            self.assertTrue(depth <= 5 * num_qubits)

            # Check that the synthesized circuit qc fits LNN connectivity
            for inst in qc.data:
                self.assertEqual(inst.operation.name, "cx")
                q0 = qc.find_bit(inst.qubits[0]).index
                q1 = qc.find_bit(inst.qubits[1]).index
                dist = abs(q0 - q1)
                self.assertEqual(dist, 1)

    @data(None, 2, 4, 5)
    def test_synth_full_pmh(self, section_size):
        """Test the PMH synthesis on pseudo-random matrices."""
        num_qubits = 5
        rng = np.random.default_rng(1234)
        num_trials = 10
        seeds = rng.integers(100000, size=num_trials, dtype=np.uint64)
        for seed in seeds:
            mat = random_invertible_binary_matrix(num_qubits, seed=seed)
            mat = np.array(mat, dtype=bool)

            qc = synth_cnot_count_full_pmh(mat, section_size)
            self.assertEqual(LinearFunction(qc), LinearFunction(mat))

    @data(5, 11)
    def test_pmh_section_sizes(self, num_qubits):
        """Test the PMH algorithm for different section sizes.

        Regression test of Qiskit/qiskit#12166.
        """
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        lin = LinearFunction(qc)

        for section_size in range(1, num_qubits + 1):
            synthesized = synth_cnot_count_full_pmh(lin.linear, section_size=section_size)
            with self.subTest(section_size=section_size):
                self.assertEqual(lin, LinearFunction(synthesized))

    def test_pmh_invalid_section_size(self):
        """Test that a section size larger than the number of qubits raises an error."""
        mat = [[True, False], [True, False]]
        with self.assertRaises(ValueError):
            _ = synth_cnot_count_full_pmh(mat, section_size=3)


if __name__ == "__main__":
    unittest.main()
