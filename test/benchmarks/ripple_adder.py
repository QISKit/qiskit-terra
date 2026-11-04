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
from qiskit.transpiler import CouplingMap

from .utils import build_ripple_adder_circuit


class RippleAdderConstruction:
    params = ([10, 50, 100, 200, 500],)
    param_names = ["size"]
    version = 1
    timeout = 600

    def time_build_ripple_adder(self, size):
        build_ripple_adder_circuit(size)


class RippleAdderTranspile:
    params = ([10, 20], [0, 1, 2, 3])
    param_names = ["size", "level"]
    version = 1
    timeout = 600

    def setup(self, size, _):
        edge_len = int((2 * size + 2) ** 0.5) + 1
        self.coupling_map = CouplingMap.from_grid(edge_len, edge_len)
        self.circuit = build_ripple_adder_circuit(size)

    def time_transpile_square_grid_ripple_adder(self, _, level):
        transpile(
            self.circuit,
            coupling_map=self.coupling_map,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            optimization_level=level,
            seed_transpiler=20220125,
        )

    def track_depth_transpile_square_grid_ripple_adder(self, _, level):
        return transpile(
            self.circuit,
            coupling_map=self.coupling_map,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            optimization_level=level,
            seed_transpiler=20220125,
        ).depth()
