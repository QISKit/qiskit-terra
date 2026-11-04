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
from functools import lru_cache

import numpy as np

from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import random_clifford
from qiskit.circuit import Gate


class VGate(Gate):
    """V Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new V Gate."""
        super().__init__("v", 1, [])

    def _define(self):
        """V Gate definition."""
        qc = QuantumCircuit(1)
        qc.sdg(0)
        qc.h(0)
        self.definition = qc


class WGate(Gate):
    """W Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new W Gate."""
        super().__init__("w", 1, [])

    def _define(self):
        """W Gate definition."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.s(0)
        self.definition = qc


@lru_cache(maxsize=24)
def clifford_1_qubit_circuit(num):
    qc = QuantumCircuit(1)
    res = []
    for k in (2, 3, 4):
        res.append(num % k)
        num //= k
    i = res[0]
    j = res[1]
    p = res[2]
    if i == 1:
        qc.h(0)
    if j == 1:
        qc.sxdg(0)
    if j == 2:
        qc.s(0)
    if p == 1:
        qc.x(0)
    if p == 2:
        qc.y(0)
    if p == 3:
        qc.z(0)
    return qc


@lru_cache(maxsize=11520)
def clifford_2_qubit_circuit(num):
    sigs = [
        (2, 2, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 4, 4),
    ]
    vals = None
    for i, sig in enumerate(sigs):
        sig_size = 1
        for k in sig:
            sig_size *= k
        if num < sig_size:
            vals = [i]
            res = []
            for k in sig:
                res.append(num % k)
                num //= k
            vals += res
            break
        num -= sig_size
    qc = QuantumCircuit(2)
    if vals[0] == 0 or vals[0] == 3:
        (form, i0, i1, j0, j1, p0, p1) = vals
        k0, k1 = (None, None)
    else:
        (form, i0, i1, j0, j1, k0, k1, p0, p1) = vals
    if i0 == 1:
        qc.h(0)
    if i1 == 1:
        qc.h(1)
    if j0 == 1:
        qc.sxdg(0)
    if j0 == 2:
        qc.s(0)
    if j1 == 1:
        qc.sxdg(1)
    if j1 == 2:
        qc.s(1)
    if form in (1, 2, 3):
        qc.cx(0, 1)
    if form in (2, 3):
        qc.cx(1, 0)
    if form == 3:
        qc.cx(0, 1)
    if form in (1, 2):
        if k0 == 1:
            qc._append(VGate(), [qc.qubits[0]], [])
        if k0 == 2:
            qc._append(WGate(), [qc.qubits[0]], [])
        if k1 == 1:
            qc._append(VGate(), [qc.qubits[1]], [])
        if k1 == 2:
            qc._append(VGate(), [qc.qubits[1]], [])
            qc._append(VGate(), [qc.qubits[1]], [])
    if p0 == 1:
        qc.x(0)
    if p0 == 2:
        qc.y(0)
    if p0 == 3:
        qc.z(0)
    if p1 == 1:
        qc.x(1)
    if p1 == 2:
        qc.y(1)
    if p1 == 3:
        qc.z(1)
    return qc


def build_rb_circuit(qubits, length_vector, num_samples=1, seed=None):
    """
    Randomized Benchmarking sequences.
    """
    if not seed:
        rng = np.random.default_rng(seed=10)
    else:
        rng = np.random.default_rng(seed=seed)
    num_qubits = len(qubits)
    circuits = []
    for _ in range(num_samples):
        for length in length_vector:
            if num_qubits > 2:
                circuits.extend(
                    random_clifford(num_qubits, seed=rng).to_circuit() for _ in range(length)
                )
            if num_qubits == 1:
                samples = rng.integers(24, size=length)
                circuits.extend(clifford_1_qubit_circuit(sample) for sample in samples)
            else:
                samples = rng.integers(11520, size=length)
                circuits.extend(clifford_2_qubit_circuit(sample) for sample in samples)
    return circuits


class RandomizedBenchmarkingBenchmark:
    # parameters for RB (1&2 qubits):
    params = (
        [
            [0],  # Single qubit RB
            [0, 1],  # Two qubit RB
        ],
    )
    param_names = ["qubits"]
    timeout = 600

    def setup(self, qubits):
        length_vector = np.arange(1, 200, 4)
        num_samples = 1
        self.seed = 10
        self.circuits = build_rb_circuit(
            qubits=qubits, length_vector=length_vector, num_samples=num_samples, seed=self.seed
        )

    def teardown(self, _):
        os.environ["QISKIT_IN_PARALLEL"] = "FALSE"

    def time_ibmq_backend_transpile(self, __):
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
            self.circuits,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            coupling_map=coupling_map,
            optimization_level=0,
            seed_transpiler=self.seed,
        )

    def time_ibmq_backend_transpile_single_thread(self, __):
        os.environ["QISKIT_IN_PARALLEL"] = "TRUE"

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
            self.circuits,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            coupling_map=coupling_map,
            optimization_level=0,
            seed_transpiler=self.seed,
        )
