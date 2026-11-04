# This code is part of Qiskit.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import os

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import GenericBackendV2

from .legacy_cmaps import MUMBAI_CMAP


class TranspilerQualitativeBench:
    params = [0, 1, 2, 3]
    param_names = ["optimization level"]
    timeout = 600

    def setup(self, optimization_level):
        self.backend = GenericBackendV2(num_qubits=27, coupling_map=MUMBAI_CMAP, seed=0)
        self.qasm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qasm"))

        self.depth_4gt10_v1_81 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "depth_4gt10-v1_81.qasm")
        )
        self.depth_4mod5_v0_19 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "depth_4mod5-v0_19.qasm")
        )
        self.depth_mod8_10_178 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "depth_mod8-10_178.qasm")
        )

        self.time_cnt3_5_179 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "time_cnt3-5_179.qasm")
        )
        self.time_cnt3_5_180 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "time_cnt3-5_180.qasm")
        )
        self.time_qft_16 = QuantumCircuit.from_qasm_file(
            os.path.join(self.qasm_path, "time_qft_16.qasm")
        )

    def track_depth_transpile_4gt10_v1_81(self, optimization_level):
        return transpile(
            self.depth_4gt10_v1_81,
            self.backend,
            optimization_level=optimization_level,
            seed_transpiler=0,
        ).depth()

    def track_depth_transpile_4mod5_v0_19(self, optimization_level):
        return transpile(
            self.depth_4mod5_v0_19,
            self.backend,
            optimization_level=optimization_level,
            seed_transpiler=0,
        ).depth()

    def track_depth_transpile_mod8_10_178(self, optimization_level):
        return transpile(
            self.depth_mod8_10_178,
            self.backend,
            optimization_level=optimization_level,
            seed_transpiler=0,
        ).depth()

    def time_transpile_time_cnt3_5_179(self, optimization_level):
        transpile(
            self.time_cnt3_5_179,
            self.backend,
            optimization_level=optimization_level,
            seed_transpiler=0,
        )

    def time_transpile_time_cnt3_5_180(self, optimization_level):
        transpile(
            self.time_cnt3_5_180,
            self.backend,
            optimization_level=optimization_level,
            seed_transpiler=0,
        )

    def time_transpile_time_qft_16(self, optimization_level):
        transpile(
            self.time_qft_16,
            self.backend,
            optimization_level=optimization_level,
            seed_transpiler=0,
        )
