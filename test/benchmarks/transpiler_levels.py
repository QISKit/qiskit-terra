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


import os

from qiskit.compiler import transpile
from qiskit import QuantumCircuit
from qiskit.transpiler import InstructionDurations
from qiskit.providers.fake_provider import GenericBackendV2

from .utils import build_qv_model_circuit
from .legacy_cmaps import MELBOURNE_CMAP


class TranspilerLevelBenchmarks:
    params = [0, 1, 2, 3]
    param_names = ["transpiler optimization level"]
    timeout = 600

    def setup(self, _):
        self.rochester_coupling_map = [
            [0, 5],
            [0, 1],
            [1, 2],
            [1, 0],
            [2, 3],
            [2, 1],
            [3, 4],
            [3, 2],
            [4, 6],
            [4, 3],
            [5, 9],
            [5, 0],
            [6, 13],
            [6, 4],
            [7, 16],
            [7, 8],
            [8, 9],
            [8, 7],
            [9, 10],
            [9, 8],
            [9, 5],
            [10, 11],
            [10, 9],
            [11, 17],
            [11, 12],
            [11, 10],
            [12, 13],
            [12, 11],
            [13, 14],
            [13, 12],
            [13, 6],
            [14, 15],
            [14, 13],
            [15, 18],
            [15, 14],
            [16, 19],
            [16, 7],
            [17, 23],
            [17, 11],
            [18, 27],
            [18, 15],
            [19, 20],
            [19, 16],
            [20, 21],
            [20, 19],
            [21, 28],
            [21, 22],
            [21, 20],
            [22, 23],
            [22, 21],
            [23, 24],
            [23, 22],
            [23, 17],
            [24, 25],
            [24, 23],
            [25, 29],
            [25, 26],
            [25, 24],
            [26, 27],
            [26, 25],
            [27, 26],
            [27, 18],
            [28, 32],
            [28, 21],
            [29, 36],
            [29, 25],
            [30, 39],
            [30, 31],
            [31, 32],
            [31, 30],
            [32, 33],
            [32, 31],
            [32, 28],
            [33, 34],
            [33, 32],
            [34, 40],
            [34, 35],
            [34, 33],
            [35, 36],
            [35, 34],
            [36, 37],
            [36, 35],
            [36, 29],
            [37, 38],
            [37, 36],
            [38, 41],
            [38, 37],
            [39, 42],
            [39, 30],
            [40, 46],
            [40, 34],
            [41, 50],
            [41, 38],
            [42, 43],
            [42, 39],
            [43, 44],
            [43, 42],
            [44, 51],
            [44, 45],
            [44, 43],
            [45, 46],
            [45, 44],
            [46, 47],
            [46, 45],
            [46, 40],
            [47, 48],
            [47, 46],
            [48, 52],
            [48, 49],
            [48, 47],
            [49, 50],
            [49, 48],
            [50, 49],
            [50, 41],
            [51, 44],
            [52, 48],
        ]
        self.basis_gates = ["u1", "u2", "u3", "cx", "id"]
        self.qv_50_x_20 = build_qv_model_circuit(50, 20, 0)
        self.qv_14_x_14 = build_qv_model_circuit(14, 14, 0)
        self.qasm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qasm"))
        large_qasm_path = os.path.join(self.qasm_path, "test_eoh_qasm.qasm")
        self.large_qasm = QuantumCircuit.from_qasm_file(large_qasm_path)
        self.melbourne = GenericBackendV2(num_qubits=14, coupling_map=MELBOURNE_CMAP, seed=0)

        self.durations = InstructionDurations(
            [
                ("u1", None, 0),
                ("id", None, 160),
                ("u2", None, 160),
                ("u3", None, 320),
                ("cx", None, 800),
                ("measure", None, 3200),
            ],
            dt=1e-9,
        )

    def time_quantum_volume_transpile_50_x_20(self, transpiler_level):
        transpile(
            self.qv_50_x_20,
            basis_gates=self.basis_gates,
            coupling_map=self.rochester_coupling_map,
            seed_transpiler=0,
            optimization_level=transpiler_level,
        )

    def track_depth_quantum_volume_transpile_50_x_20(self, transpiler_level):
        return transpile(
            self.qv_50_x_20,
            basis_gates=self.basis_gates,
            coupling_map=self.rochester_coupling_map,
            seed_transpiler=0,
            optimization_level=transpiler_level,
        ).depth()

    def time_transpile_from_large_qasm(self, transpiler_level):
        transpile(
            self.large_qasm,
            basis_gates=self.basis_gates,
            coupling_map=self.rochester_coupling_map,
            seed_transpiler=0,
            optimization_level=transpiler_level,
        )

    def track_depth_transpile_from_large_qasm(self, transpiler_level):
        return transpile(
            self.large_qasm,
            basis_gates=self.basis_gates,
            coupling_map=self.rochester_coupling_map,
            seed_transpiler=0,
            optimization_level=transpiler_level,
        ).depth()

    def time_transpile_from_large_qasm_backend_with_prop(self, transpiler_level):
        transpile(
            self.large_qasm, self.melbourne, seed_transpiler=0, optimization_level=transpiler_level
        )

    def track_depth_transpile_from_large_qasm_backend_with_prop(self, transpiler_level):
        return transpile(
            self.large_qasm, self.melbourne, seed_transpiler=0, optimization_level=transpiler_level
        ).depth()

    def time_transpile_qv_14_x_14(self, transpiler_level):
        transpile(
            self.qv_14_x_14, self.melbourne, seed_transpiler=0, optimization_level=transpiler_level
        )

    def track_depth_transpile_qv_14_x_14(self, transpiler_level):
        return transpile(
            self.qv_14_x_14, self.melbourne, seed_transpiler=0, optimization_level=transpiler_level
        ).depth()

    def time_schedule_qv_14_x_14(self, transpiler_level):
        transpile(
            self.qv_14_x_14,
            self.melbourne,
            seed_transpiler=0,
            optimization_level=transpiler_level,
            scheduling_method="alap",
        )

    # limit optimization levels to reduce time
    time_schedule_qv_14_x_14.params = [0, 1]
