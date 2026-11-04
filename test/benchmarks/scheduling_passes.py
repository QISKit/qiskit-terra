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


from qiskit import transpile
from qiskit.circuit.library.standard_gates import XGate
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler import InstructionDurations
from qiskit.transpiler.passes import (
    TimeUnitConversion,
    ASAPScheduleAnalysis,
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
)
from qiskit.converters import circuit_to_dag

from .utils import random_circuit


class SchedulingPassBenchmarks:

    params = ([5, 10, 20], [500, 1000])
    param_names = ["n_qubits", "depth"]
    timeout = 300

    def setup(self, n_qubits, depth):
        seed = 42
        self.circuit = random_circuit(
            n_qubits, depth, measure=True, conditional=False, reset=False, seed=seed, max_operands=2
        )
        self.basis_gates = ["rz", "sx", "x", "cx", "id", "reset"]
        self.cmap = [
            [0, 1],
            [1, 0],
            [1, 2],
            [1, 6],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [3, 8],
            [4, 3],
            [5, 6],
            [5, 10],
            [6, 1],
            [6, 5],
            [6, 7],
            [7, 6],
            [7, 8],
            [7, 12],
            [8, 3],
            [8, 7],
            [8, 9],
            [9, 8],
            [9, 14],
            [10, 5],
            [10, 11],
            [11, 10],
            [11, 12],
            [11, 16],
            [12, 7],
            [12, 11],
            [12, 13],
            [13, 12],
            [13, 14],
            [13, 18],
            [14, 9],
            [14, 13],
            [15, 16],
            [16, 11],
            [16, 15],
            [16, 17],
            [17, 16],
            [17, 18],
            [18, 13],
            [18, 17],
            [18, 19],
            [19, 18],
        ]
        self.coupling_map = CouplingMap(self.cmap)
        self.transpiled_circuit = transpile(
            self.circuit,
            basis_gates=self.basis_gates,
            coupling_map=self.coupling_map,
            optimization_level=1,
        )
        self.dag = circuit_to_dag(self.transpiled_circuit)
        self.durations = InstructionDurations(
            [
                ("rz", None, 0),
                ("id", None, 160),
                ("sx", None, 160),
                ("x", None, 160),
                ("cx", None, 800),
                ("measure", None, 3200),
                ("reset", None, 3600),
            ],
            dt=1e-9,
        )

    def time_time_unit_conversion_pass(self, _, __):
        TimeUnitConversion(self.durations).run(self.dag)

    def time_alap_schedule_pass(self, _, __):
        dd_sequence = [XGate(), XGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence),
            ]
        )
        pm.run(self.transpiled_circuit)

    def time_asap_schedule_pass(self, _, __):
        dd_sequence = [XGate(), XGate()]
        pm = PassManager(
            [
                ASAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence),
            ]
        )
        pm.run(self.transpiled_circuit)
