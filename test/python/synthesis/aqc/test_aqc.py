# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tests AQC framework using hardcoded and randomly generated circuits.
"""
from functools import partial
import unittest
from ddt import ddt, data
import numpy as np
from scipy.optimize import minimize

from qiskit.quantum_info import Operator
from qiskit.synthesis.unitary.aqc.aqc import AQC
from qiskit.synthesis.unitary.aqc.cnot_structures import make_cnot_network
from qiskit.synthesis.unitary.aqc.cnot_unit_circuit import CNOTUnitCircuit
from qiskit.synthesis.unitary.aqc.cnot_unit_objective import DefaultCNOTUnitObjective
from qiskit.synthesis.unitary.aqc.fast_gradient.fast_gradient import FastCNOTUnitObjective
from test import QiskitTestCase
from test.python.synthesis.aqc.sample_data import (
    ORIGINAL_CIRCUIT,
    INITIAL_THETAS,
)


@ddt
class TestAqc(QiskitTestCase):
    """Main tests of approximate quantum compiler."""

    @data(True, False)
    def test_aqc(self, uses_default):
        """Tests AQC on a hardcoded circuit/matrix."""

        seed = 12345

        num_qubits = round(np.log2(ORIGINAL_CIRCUIT.shape[0]))
        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=0
        )

        if uses_default:
            aqc = AQC(seed=seed)
        else:
            optimizer = partial(minimize, args=(), method="L-BFGS-B", options={"maxiter": 200})
            aqc = AQC(optimizer=optimizer, seed=seed)

        target_matrix = ORIGINAL_CIRCUIT
        approximate_circuit = CNOTUnitCircuit(num_qubits, cnots)
        approximating_objective = DefaultCNOTUnitObjective(num_qubits, cnots)

        aqc.compile_unitary(
            target_matrix=target_matrix,
            approximate_circuit=approximate_circuit,
            approximating_objective=approximating_objective,
            initial_point=INITIAL_THETAS,
        )

        approx_matrix = Operator(approximate_circuit).data
        error = 0.5 * (np.linalg.norm(approx_matrix - ORIGINAL_CIRCUIT, "fro") ** 2)
        self.assertLess(error, 1e-3)

    def test_aqc_fastgrad(self):
        """
        Tests AQC on a MCX circuit/matrix with random initial guess using
        the accelerated implementation.
        """
        seed = 12345
        np.random.seed(seed)

        num_qubits = 3
        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=0
        )

        optimizer = partial(minimize, args=(), method="L-BFGS-B", options={"maxiter": 200})
        aqc = AQC(optimizer=optimizer, seed=seed)

        # Make multi-control CNOT gate matrix.
        # Another option: target_matrix = ORIGINAL_CIRCUIT
        target_matrix = np.eye(2**num_qubits, dtype=np.complex128)
        target_matrix[-2:, -2:] = [[0, 1], [1, 0]]

        circ = CNOTUnitCircuit(num_qubits, cnots)
        objv = FastCNOTUnitObjective(num_qubits, cnots)

        aqc.compile_unitary(
            target_matrix=target_matrix,
            approximate_circuit=circ,
            approximating_objective=objv,
            initial_point=2 * np.pi * np.random.rand(objv.num_thetas),
        )

        approx_matrix = Operator(circ).data
        error = 0.5 * (np.linalg.norm(approx_matrix - target_matrix, "fro") ** 2)
        self.assertTrue(error < 1e-3)

    def test_aqc_determinant_minus_one(self):
        """
        Tests AQC on a matrix with determinant −1.
        """
        seed = 12345

        num_qubits = 3
        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=0
        )

        optimizer = partial(minimize, args=(), method="L-BFGS-B", options={"maxiter": 200})
        aqc = AQC(optimizer=optimizer, seed=seed)

        target_matrix = np.eye(2**num_qubits, dtype=int)
        # Make a unitary with determinant −1 by swapping any two columns
        target_matrix[:, 2], target_matrix[:, 3] = target_matrix[:, 3], target_matrix[:, 2].copy()

        approximate_circuit = CNOTUnitCircuit(num_qubits, cnots)
        approximating_objective = DefaultCNOTUnitObjective(num_qubits, cnots)

        aqc.compile_unitary(
            target_matrix=target_matrix,
            approximate_circuit=approximate_circuit,
            approximating_objective=approximating_objective,
            initial_point=INITIAL_THETAS,
        )

        approx_matrix = Operator(approximate_circuit).data

        error = 0.5 * (np.linalg.norm(approx_matrix - target_matrix, "fro") ** 2)
        self.assertTrue(error < 1e-3)


if __name__ == "__main__":
    unittest.main()
