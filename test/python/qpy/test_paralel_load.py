# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import io

from qiskit.circuit.random import random_circuit
from qiskit.qpy import dump, load
from test import QiskitTestCase


class TestParallelLoad(QiskitTestCase):
    """Verify that parallel and sequantial loads produce identical results"""

    def generate_dump_with_random_circuits(self, num_circuits=10, num_qubits=20, depth=50, seed=42):
        circuits = [
            random_circuit(
                num_qubits, depth, measure=True, conditional=True, reset=True, seed=seed + i
            )
            for i in range(num_circuits)
        ]
        buf = io.BytesIO()
        dump(circuits, buf)
        return buf.getvalue()

    def test_parallel_and_sequential_match(self):
        dump = self.generate_dump_with_random_circuits()

        sequential = load(io.BytesIO(dump), parallel=False)
        parallel = load(io.BytesIO(dump), parallel=True)

        self.assertEqual(len(sequential), len(parallel))
        for seq, par in zip(sequential, parallel):
            self.assertEqual(seq, par)
