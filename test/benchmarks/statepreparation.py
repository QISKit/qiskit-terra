# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.circuit.library.data_preparation import StatePreparation


class StatePreparationTranspileBench:
    params = [4, 5, 6, 7, 8]
    param_names = ["number of qubits in state"]

    def setup(self, n):
        q = QuantumRegister(n)
        qc = QuantumCircuit(q)
        state = np.random.rand(2**n) + np.random.rand(2**n) * 1j
        state = state / np.linalg.norm(state)
        state_gate = StatePreparation(state)
        qc.append(state_gate, q)

        self.circuit = qc

    def track_cnot_counts_after_mapping_to_ibmq_16_melbourne(self, *unused):
        coupling = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]
        circuit = transpile(
            self.circuit,
            basis_gates=["u1", "u3", "u2", "cx"],
            coupling_map=coupling,
            seed_transpiler=0,
        )
        counts = circuit.count_ops()
        cnot_count = counts.get("cx", 0)
        return cnot_count
