# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Clifford synthesis functions."""

from ddt import ddt
from qiskit.quantum_info import random_clifford
from qiskit.quantum_info.operators import Clifford
from qiskit.synthesis.clifford import (
    synth_clifford_full,
    synth_clifford_ag,
    synth_clifford_bm,
    synth_clifford_greedy,
)

from test import QiskitTestCase
from test import combine


@ddt
class TestCliffordSynthesis(QiskitTestCase):
    """Tests for clifford synthesis functions."""

    @staticmethod
    def _cliffords_1q():
        clifford_dicts = [
            {"stabilizer": ["+Z"], "destabilizer": ["-X"]},
            {"stabilizer": ["-Z"], "destabilizer": ["+X"]},
            {"stabilizer": ["-Z"], "destabilizer": ["-X"]},
            {"stabilizer": ["+Z"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+Z"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-Z"], "destabilizer": ["+Y"]},
            {"stabilizer": ["-Z"], "destabilizer": ["-Y"]},
            {"stabilizer": ["+X"], "destabilizer": ["+Z"]},
            {"stabilizer": ["+X"], "destabilizer": ["-Z"]},
            {"stabilizer": ["-X"], "destabilizer": ["+Z"]},
            {"stabilizer": ["-X"], "destabilizer": ["-Z"]},
            {"stabilizer": ["+X"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+X"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-X"], "destabilizer": ["+Y"]},
            {"stabilizer": ["-X"], "destabilizer": ["-Y"]},
            {"stabilizer": ["+Y"], "destabilizer": ["+X"]},
            {"stabilizer": ["+Y"], "destabilizer": ["-X"]},
            {"stabilizer": ["-Y"], "destabilizer": ["+X"]},
            {"stabilizer": ["-Y"], "destabilizer": ["-X"]},
            {"stabilizer": ["+Y"], "destabilizer": ["+Z"]},
            {"stabilizer": ["+Y"], "destabilizer": ["-Z"]},
            {"stabilizer": ["-Y"], "destabilizer": ["+Z"]},
            {"stabilizer": ["-Y"], "destabilizer": ["-Z"]},
        ]
        return [Clifford.from_dict(i) for i in clifford_dicts]

    def test_decompose_1q(self):
        """Test synthesis for all 1-qubit Cliffords"""
        for cliff in self._cliffords_1q():
            with self.subTest(msg=f"Test circuit {cliff}"):
                target = cliff
                value = Clifford(cliff.to_circuit())
                self.assertEqual(target, value)

    @combine(num_qubits=[0, 1, 2, 3])
    def test_synth_bm(self, num_qubits):
        """Test B&M synthesis for set of {num_qubits}-qubit Cliffords"""
        samples = 50
        for seed in range(samples):
            target = random_clifford(num_qubits, seed=seed)
            synth_circ = synth_clifford_bm(target)
            value = Clifford(synth_circ)
            self.assertEqual(value, target)

    @combine(num_qubits=[0, 1, 2, 3, 4, 5])
    def test_synth_ag(self, num_qubits):
        """Test A&G synthesis for set of {num_qubits}-qubit Cliffords"""
        samples = 50
        for seed in range(samples):
            target = random_clifford(num_qubits, seed)
            synth_circ = synth_clifford_ag(target)
            value = Clifford(synth_circ)
            self.assertEqual(value, target)

    @combine(num_qubits=[0, 1, 2, 3, 4, 5])
    def test_synth_greedy(self, num_qubits):
        """Test greedy synthesis for set of {num_qubits}-qubit Cliffords"""
        samples = 50
        for seed in range(samples):
            target = random_clifford(num_qubits, seed)
            synth_circ = synth_clifford_greedy(target)
            value = Clifford(synth_circ)
            self.assertEqual(value, target)

    @combine(num_qubits=[0, 1, 2, 3, 4, 5])
    def test_synth_full(self, num_qubits):
        """Test synthesis for set of {num_qubits}-qubit Cliffords"""
        samples = 50
        for seed in range(samples):
            target = random_clifford(num_qubits, seed)
            synth_circ = synth_clifford_full(target)
            value = Clifford(synth_circ)
            self.assertEqual(value, target)
