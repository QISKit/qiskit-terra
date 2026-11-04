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


import itertools
import math

from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreSwap


def build_model_circuit(qreg, circuit=None):
    """Create quantum fourier transform circuit on quantum register qreg."""
    if circuit is None:
        circuit = QuantumCircuit(qreg, name="qft")

    n = len(qreg)

    for i in range(n):
        for j in range(i):
            # Using negative exponents so we safely underflow to 0 rather than
            # raise `OverflowError`.
            circuit.cp(math.pi * (2.0 ** (j - i)), qreg[i], qreg[j])
        circuit.h(qreg[i])

    return circuit


class QftTranspileBench:
    params = [1, 2, 3, 5, 8, 13, 14]

    def setup(self, n):
        qr = QuantumRegister(n)
        self.circuit = build_model_circuit(qr)

    def time_ibmq_backend_transpile(self, _):
        # Run with ibmq_16_melbourne configuration
        coupling_map = [
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
        transpile(
            self.circuit,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            coupling_map=coupling_map,
            seed_transpiler=20220125,
        )


class LargeQFTMappingTimeBench:
    timeout = 600.0  # seconds

    heavy_hex_size = {115: 7, 409: 13, 1081: 21}
    params = ([115, 409, 1081], ["lookahead", "decay"])
    param_names = ["n_qubits", "heuristic"]

    def setup(self, n_qubits, _heuristic):
        qr = QuantumRegister(n_qubits, name="q")
        self.dag = circuit_to_dag(build_model_circuit(qr))
        self.coupling = CouplingMap.from_heavy_hex(self.heavy_hex_size[n_qubits])

    def time_sabre_swap(self, _n_qubits, heuristic):
        pass_ = SabreSwap(self.coupling, heuristic, seed=2022_10_27, trials=1)
        pass_.run(self.dag)


class LargeQFTMappingTrackBench:
    timeout = 600.0  # seconds, needs to account for the _entire_ setup.

    heavy_hex_size = {115: 7, 409: 13, 1081: 21}
    params = ([115, 409, 1081], ["lookahead", "decay"])
    param_names = ["n_qubits", "heuristic"]

    # The benchmarks take a significant amount of time to run, and we don't
    # want to unnecessarily run things twice to get the two pieces of tracking
    # information we're interested in.  We cheat by using the setup cache to do
    # all the calculation work only once, and then each tracker just quickly
    # pulls the result from the cache to return, saving the duplication.

    def setup_cache(self):
        def setup(n_qubits, heuristic):
            qr = QuantumRegister(n_qubits, name="q")
            dag = circuit_to_dag(build_model_circuit(qr))
            coupling = CouplingMap.from_heavy_hex(self.heavy_hex_size[n_qubits])
            pass_ = SabreSwap(coupling, heuristic, seed=2022_10_27, trials=1)
            return pass_.run(dag)

        state = {}
        for params in itertools.product(*self.params):
            dag = setup(*params)
            state[params] = {"depth": dag.depth(), "size": dag.size()}
        return state

    def track_depth_sabre_swap(self, state, *params):
        return state[params]["depth"]

    def track_size_sabre_swap(self, state, *params):
        return state[params]["size"]
