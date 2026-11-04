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


"""Module for estimating quantum volume.
See arXiv:1811.12926 [quant-ph]"""

import itertools

import numpy as np

from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreSwap

from .utils import build_qv_model_circuit


class QuantumVolumeBenchmark:
    params = ([1, 2, 3, 5, 8, 14, 20, 27], ["translator", "synthesis"])
    param_names = ["Number of Qubits", "Basis Translation Method"]
    version = 3

    def setup(self, width, _):
        random_seed = np.random.seed(10)
        self.circuit = build_qv_model_circuit(width, width, random_seed)
        self.coupling_map = [
            [0, 1],
            [1, 0],
            [1, 2],
            [1, 4],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 5],
            [4, 1],
            [4, 7],
            [5, 3],
            [5, 8],
            [6, 7],
            [7, 4],
            [7, 6],
            [7, 10],
            [8, 5],
            [8, 9],
            [8, 11],
            [9, 8],
            [10, 7],
            [10, 12],
            [11, 8],
            [11, 14],
            [12, 10],
            [12, 13],
            [12, 15],
            [13, 12],
            [13, 14],
            [14, 11],
            [14, 13],
            [14, 16],
            [15, 12],
            [15, 18],
            [16, 14],
            [16, 19],
            [17, 18],
            [18, 15],
            [18, 17],
            [18, 21],
            [19, 16],
            [19, 20],
            [19, 22],
            [20, 19],
            [21, 18],
            [21, 23],
            [22, 19],
            [22, 25],
            [23, 21],
            [23, 24],
            [24, 23],
            [24, 25],
            [25, 22],
            [25, 24],
            [25, 26],
            [26, 25],
        ]
        self.basis = ["id", "rz", "sx", "x", "cx", "reset"]

    def time_ibmq_backend_transpile(self, _, translation):
        transpile(
            self.circuit,
            basis_gates=self.basis,
            coupling_map=self.coupling_map,
            translation_method=translation,
            seed_transpiler=20220125,
        )


class LargeQuantumVolumeMappingTimeBench:
    timeout = 600.0  # seconds
    heavy_hex_distance = {115: 7, 409: 13, 1081: 21}
    allowed_sizes = {(115, 100), (115, 10), (409, 10), (1081, 10)}
    n_qubits = sorted({n_qubits for n_qubits, _ in allowed_sizes})
    depths = sorted({depth for _, depth in allowed_sizes})

    params = (n_qubits, depths, ["lookahead", "decay"])
    param_names = ["n_qubits", "depth", "heuristic"]

    def setup(self, n_qubits, depth, _):
        if (n_qubits, depth) not in self.allowed_sizes:
            raise NotImplementedError
        seed = 2022_10_27
        self.dag = circuit_to_dag(build_qv_model_circuit(n_qubits, depth, seed))
        self.coupling = CouplingMap.from_heavy_hex(self.heavy_hex_distance[n_qubits])

    def time_sabre_swap(self, _n_qubits, _depth, heuristic):
        pass_ = SabreSwap(self.coupling, heuristic, seed=2022_10_27, trials=1)
        pass_.run(self.dag)


class LargeQuantumVolumeMappingTrackBench:
    timeout = 600.0  # seconds

    allowed_sizes = {(115, 100), (115, 10), (409, 10), (1081, 10)}
    heuristics = ["lookahead", "decay"]
    n_qubits = sorted({n_qubits for n_qubits, _ in allowed_sizes})
    depths = sorted({depth for _, depth in allowed_sizes})

    params = (n_qubits, depths, heuristics)
    param_names = ["n_qubits", "depth", "heuristic"]

    # The benchmarks take a significant amount of time to run, and we don't
    # want to unnecessarily run things twice to get the two pieces of tracking
    # information we're interested in.  We cheat by using the setup cache to do
    # all the calculation work only once, and then each tracker just quickly
    # pulls the result from the cache to return, saving the duplication.

    def setup_cache(self):
        heavy_hex_distance = {115: 7, 409: 13, 1081: 21}
        seed = 2022_10_27

        def setup(n_qubits, depth, heuristic):
            dag = circuit_to_dag(build_qv_model_circuit(n_qubits, depth, seed))
            coupling = CouplingMap.from_heavy_hex(heavy_hex_distance[n_qubits])
            return SabreSwap(coupling, heuristic, seed=seed, trials=1).run(dag)

        state = {}
        for params in itertools.product(*self.params):
            n_qubits, depth, _ = params
            if (n_qubits, depth) not in self.allowed_sizes:
                continue
            dag = setup(*params)
            state[params] = {"depth": dag.depth(), "size": dag.size()}
        return state

    def setup(self, _state, n_qubits, depth, _heuristic):
        if (n_qubits, depth) not in self.allowed_sizes:
            raise NotImplementedError

    def track_depth_sabre_swap(self, state, *params):
        return state[params]["depth"]

    def track_size_sabre_swap(self, state, *params):
        return state[params]["size"]
