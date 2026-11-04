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


import io
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit import qpy

from .utils import random_circuit


class RandomBenchmarks:

    params = ([20], [256, 1024], [0, 42])

    param_names = ["n_qubits", "depth", "seed"]
    timeout = 300

    def setup(self, n_qubits, depth, seed):
        self.circuit = random_circuit(
            n_qubits,
            depth,
            measure=True,
            conditional=True,
            reset=True,
            seed=seed,
            max_operands=3,
        )

    def time_dump(self, _, __, ___):
        qpy_file = io.BytesIO()
        qpy.dump(self.circuit, qpy_file)

    def time_dump_and_load(self, _, __, ___):
        qpy_file = io.BytesIO()
        qpy.dump(self.circuit, qpy_file)
        qpy_file.seek(0)
        qpy.load(qpy_file)


class CustomGateBenchmarks:

    params = ([200], [100])

    param_names = ["n_qubits", "depth"]
    timeout = 300

    def setup(self, n_qubits, depth):
        custom_gate = QuantumCircuit(2, name="custom_gate")
        custom_gate.h(0)
        custom_gate.x(1)

        qc = QuantumCircuit(n_qubits)
        for _ in range(depth):
            for i in range(n_qubits - 1):
                qc.append(custom_gate.to_gate(), [i, i + 1])
        self.circuit = qc

    def time_dump(self, _, __):
        qpy_file = io.BytesIO()
        qpy.dump(self.circuit, qpy_file)

    def time_dump_and_load(self, _, __):
        qpy_file = io.BytesIO()
        qpy.dump(self.circuit, qpy_file)
        qpy_file.seek(0)
        qpy.load(qpy_file)


class ParameterizedBenchmarks:

    params = ([20, 50], [1, 5, 10])

    param_names = ["n_qubits", "n_params"]
    timeout = 300

    def setup(self, n_qubits, n_params):
        qc = QuantumCircuit(n_qubits)
        params = [Parameter(f"angle{i}") for i in range(n_params)]
        for n in range(n_qubits - 1):
            for i in params:
                qc.rx(i, n)
        self.circuit = qc

    def time_dump(self, _, __):
        qpy_file = io.BytesIO()
        qpy.dump(self.circuit, qpy_file)

    def time_dump_and_load(self, _, __):
        qpy_file = io.BytesIO()
        qpy.dump(self.circuit, qpy_file)
        qpy_file.seek(0)
        qpy.load(qpy_file)
