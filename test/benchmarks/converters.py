# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from qiskit import converters

from .utils import random_circuit


class ConverterBenchmarks:
    params = ([1, 2, 5, 8, 14, 20, 32, 53], [8, 128, 2048, 8192])
    param_names = ["n_qubits", "depth"]
    timeout = 600

    def setup(self, n_qubits, depth):
        seed = 42
        # NOTE: Remove the benchmarks larger than 20x2048 and 14x8192, this is
        # a tradeoff for speed of benchmarking, creating circuits this size
        # takes more time than is worth it for benchmarks that take a couple
        # seconds
        if n_qubits >= 20:
            if depth >= 2048:
                raise NotImplementedError
        elif n_qubits == 14:
            if depth > 2048:
                raise NotImplementedError
        self.qc = random_circuit(n_qubits, depth, measure=True, conditional=True, seed=seed)
        self.dag = converters.circuit_to_dag(self.qc)

    def time_circuit_to_dag(self, *_):
        converters.circuit_to_dag(self.qc)

    def time_circuit_to_instruction(self, *_):
        converters.circuit_to_instruction(self.qc)

    def time_dag_to_circuit(self, *_):
        converters.dag_to_circuit(self.dag)
