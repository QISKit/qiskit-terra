# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis, HLSConfig
from qiskit.transpiler import PassManager
from qiskit.circuit.library import (
    LinearFunction,
)
from qiskit.quantum_info.operators.symplectic.random import random_clifford
from qiskit.synthesis.linear import random_invertible_binary_matrix


# TODO: track depth, num. 2q gates, 2q depth
class HLSPluginsSuite:

    def rand_benchmarking_clifford_circuit(self, num_qubits, depth, seed=None):
        """Build Clifford circuit <randomized benchmarking style> (wide and deep)."""
        qc = QuantumCircuit(num_qubits)
        for i in range(0, depth):
            for _ in range(depth):
                cliff = random_clifford(2, seed=seed)
                qc.append(cliff, [2 * i, 2 * i + 1])
        return qc

    def random_wide_clifford_circuit(self, num_qubits, seed=None):
        """Build a wide (but shallow) random Clifford circuit."""
        qc = QuantumCircuit(num_qubits)
        cliff = random_clifford(num_qubits, seed=seed)
        qc.append(cliff, range(num_qubits))
        return qc

    def random_linear_circuit(self, num_qubits, seed=None):
        """Build a wide circuit out of a random linear function."""
        mat = random_invertible_binary_matrix(num_qubits, seed=seed)
        qc = QuantumCircuit(num_qubits)
        qc.append(LinearFunction(mat), list(range(num_qubits)))
        return qc

    def setUp(self):
        # Set seed and num qubits
        self.num_qubits = 100
        self.rng = np.random.default_rng(1234)

        # Define circuits for benchmarking
        self.qc_wide_clifford = self.random_wide_clifford_circuit(self.num_qubits, seed=self.rng)
        self.qc_rand_clifford = self.rand_benchmarking_clifford_circuit(
            self.num_qubits, self.num_qubits // 2, seed=self.rng
        )
        self.qc_linear = self.random_linear_circuit(self.num_qubits, seed=self.rng)

        # Define pm with different synthesis plugin configurations
        self.pm_ag = PassManager([HighLevelSynthesis(hls_config=HLSConfig(clifford=["ag"]))])
        self.pm_bm = PassManager([HighLevelSynthesis(hls_config=HLSConfig(clifford=["bm"]))])
        self.pm_greedy = PassManager(
            [HighLevelSynthesis(hls_config=HLSConfig(clifford=["greedy"]))]
        )
        self.pm_default = PassManager(
            [HighLevelSynthesis(hls_config=HLSConfig(clifford=["default"]))]
        )

        self.pm_pmh = PassManager(
            [HighLevelSynthesis(hls_config=HLSConfig(linear_function=[("pmh", {})]))]
        )
        self.pm_kms = PassManager(
            [HighLevelSynthesis(hls_config=HLSConfig(linear_function=[("kms", {})]))]
        )

    # Time Clifford synthesis plugins with random wide circuit
    def time_clifford_ag_wide_circuit(self):
        _ = self.pm_ag.run(self.qc_wide_clifford)

    def time_clifford_bm_wide_circuit(self):
        _ = self.pm_bm.run(self.qc_wide_clifford)

    def time_clifford_greedy_wide_circuit(self):
        _ = self.pm_greedy.run(self.qc_wide_clifford)

    def time_clifford_default_wide_circuit(self):
        _ = self.pm_default.run(self.qc_wide_clifford)

    # Time Clifford synthesis plugins with randomized benchmarking circuit
    def time_clifford_ag_rand_benchmarking(self):
        _ = self.pm_ag.run(self.qc_rand_clifford)

    def time_clifford_bm_rand_benchmarking(self):
        _ = self.pm_bm.run(self.qc_rand_clifford)

    def time_clifford_greedy_rand_benchmarking(self):
        _ = self.pm_greedy.run(self.qc_rand_clifford)

    def time_clifford_default_rand_benchmarking(self):
        _ = self.pm_default.run(self.qc_rand_clifford)

    # Time Linear synthesis plugins
    def time_linear_func_pmh(self):
        _ = self.pm_pmh.run(self.qc_linear)

    def time_linear_func_kms(self):
        _ = self.pm_kms.run(self.qc_linear)
