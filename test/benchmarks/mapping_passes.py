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


from copy import deepcopy

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import (
    FullAncillaAllocation,
    EnlargeWithAncilla,
    ApplyLayout,
    SabreSwap,
    BasicSwap,
    Layout2qDistance,
    DenseLayout,
    CheckMap,
    TrivialLayout,
    SetLayout,
    SabreLayout,
    GateDirection,
    CheckGateDirection,
    CSPLayout,
)
from qiskit.converters import circuit_to_dag

from .utils import random_circuit


class PassBenchmarks:

    params = ([5, 14, 20], [1024])

    param_names = ["n_qubits", "depth"]
    timeout = 300

    def setup(self, n_qubits, depth):
        seed = 42
        self.circuit = random_circuit(
            n_qubits, depth, measure=True, conditional=True, reset=True, seed=seed, max_operands=2
        )
        self.fresh_dag = circuit_to_dag(self.circuit)
        self.basis_gates = ["u1", "u2", "u3", "cx", "id"]
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

        layout_pass = DenseLayout(self.coupling_map)
        layout_pass.run(self.fresh_dag)
        self.layout = layout_pass.property_set["layout"]
        full_ancilla_pass = FullAncillaAllocation(self.coupling_map)
        full_ancilla_pass.property_set["layout"] = self.layout
        self.full_ancilla_dag = full_ancilla_pass.run(self.fresh_dag)
        enlarge_pass = EnlargeWithAncilla()
        enlarge_pass.property_set["layout"] = self.layout
        self.enlarge_dag = enlarge_pass.run(self.full_ancilla_dag)
        apply_pass = ApplyLayout()
        apply_pass.property_set["layout"] = self.layout
        self.dag = apply_pass.run(deepcopy(self.enlarge_dag))

    def time_sabre_swap(self, _, __):
        swap = SabreSwap(self.coupling_map, seed=42)
        swap.property_set["layout"] = self.layout
        swap.run(self.dag)

    def time_basic_swap(self, _, __):
        swap = BasicSwap(self.coupling_map)
        swap.property_set["layout"] = self.layout
        swap.run(self.dag)

    def time_csp_layout(self, _, __):
        CSPLayout(self.coupling_map, seed=42).run(self.fresh_dag)

    def time_dense_layout(self, _, __):
        DenseLayout(self.coupling_map).run(self.fresh_dag)

    def time_layout_2q_distance(self, _, __):
        layout = Layout2qDistance(self.coupling_map)
        layout.property_set["layout"] = self.layout
        layout.run(self.enlarge_dag)

    def time_apply_layout(self, _, __):
        layout = ApplyLayout()
        layout.property_set["layout"] = self.layout
        layout.run(self.enlarge_dag)

    def time_full_ancilla_allocation(self, _, __):
        ancilla = FullAncillaAllocation(self.coupling_map)
        ancilla.property_set["layout"] = self.layout
        ancilla.run(self.fresh_dag)

    def time_enlarge_with_ancilla(self, _, __):
        ancilla = EnlargeWithAncilla()
        ancilla.property_set["layout"] = self.layout
        ancilla.run(self.full_ancilla_dag)

    def time_check_map(self, _, __):
        CheckMap(self.coupling_map).run(self.dag)

    def time_trivial_layout(self, _, __):
        TrivialLayout(self.coupling_map).run(self.fresh_dag)

    def time_set_layout(self, _, __):
        SetLayout(self.layout).run(self.fresh_dag)

    def time_sabre_layout(self, _, __):
        SabreLayout(self.coupling_map, seed=42).run(self.fresh_dag)


class RoutedPassBenchmarks:
    params = ([5, 14, 20], [1024])

    param_names = ["n_qubits", "depth"]
    timeout = 300

    def setup(self, n_qubits, depth):
        seed = 42
        self.circuit = random_circuit(
            n_qubits, depth, measure=True, conditional=True, reset=True, seed=seed, max_operands=2
        )
        self.fresh_dag = circuit_to_dag(self.circuit)
        self.basis_gates = ["u1", "u2", "u3", "cx", "id"]
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

        layout_pass = DenseLayout(self.coupling_map)
        layout_pass.run(self.fresh_dag)
        self.layout = layout_pass.property_set["layout"]
        full_ancilla_pass = FullAncillaAllocation(self.coupling_map)
        full_ancilla_pass.property_set["layout"] = self.layout
        self.full_ancilla_dag = full_ancilla_pass.run(self.fresh_dag)
        enlarge_pass = EnlargeWithAncilla()
        enlarge_pass.property_set["layout"] = self.layout
        self.enlarge_dag = enlarge_pass.run(self.full_ancilla_dag)
        apply_pass = ApplyLayout()
        apply_pass.property_set["layout"] = self.layout
        self.dag = apply_pass.run(self.enlarge_dag)
        self.routed_dag = SabreSwap(self.coupling_map, seed=42).run(self.dag)

    def time_gate_direction(self, _, __):
        GateDirection(self.coupling_map).run(self.routed_dag)

    def time_check_gate_direction(self, _, __):
        CheckGateDirection(self.coupling_map).run(self.routed_dag)

    def time_check_map(self, _, __):
        CheckMap(self.coupling_map).run(self.routed_dag)
