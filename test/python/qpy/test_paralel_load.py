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
import os
from unittest.mock import patch

from qiskit.circuit.annotated_operation import AnnotatedOperation, ControlModifier, InverseModifier
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
from qiskit.circuit.library.standard_gates.x import CXGate, XGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit import qpy
from qiskit.quantum_info.operators.symplectic.sparse_pauli_op import SparsePauliOp
from qiskit.synthesis.evolution.lie_trotter import LieTrotter
from test import QiskitTestCase


class MyParamGate(Gate):
    "Custom gate for exercise the custom instruciton load"

    def __init__(self, phi):
        super().__init__("custom gate", 1, [phi])

    def _define(self):
        qc = QuantumCircuit(1)
        qc.rx(self.params[0], 0)
        self.definition = qc


class TestParallelLoad(QiskitTestCase):
    """Verify that parallel and sequential loads produce identical results"""

    def generate_dump_with_random_circuits(self, num_circuits=10, num_qubits=20, depth=50, seed=42):
        circuits = [
            random_circuit(
                num_qubits, depth, measure=True, conditional=True, reset=True, seed=seed + i
            )
            for i in range(num_circuits)
        ]
        buf = io.BytesIO()
        qpy.dump(circuits, buf)
        return buf.getvalue()

    def test_parallel_and_sequential_match(self):
        dump = self.generate_dump_with_random_circuits()

        with patch.dict(os.environ, {"QISKIT_IN_PARALLEL": "TRUE"}):
            with io.BytesIO(dump) as buff:
                sequential = qpy.load(buff)
        with patch.dict(os.environ, {"QISKIT_IN_PARALLEL": "FALSE"}):
            with io.BytesIO(dump) as buff:
                parallel = qpy.load(buff)

        self.assertEqual(len(sequential), len(parallel))
        for seq, par in zip(sequential, parallel):
            self.assertEqual(seq, par)

    def generate_dump_with_custom_operations(self):
        """Build several circuits, with multiple operations: custom gate, PauliEvaluation,
        AnnotatedOperation and ControlledGate, in order to exersise GIL acquiring in parallel load
        """
        theta = Parameter("theta")
        circuits = []

        qc1 = QuantumCircuit(3)
        qc1.append(MyParamGate(1.23), [0])
        qc1.append(MyParamGate(theta), [1])
        qc1.append(XGate().control(2, annotated=False), [0, 1, 2])
        circuits.append(qc1)

        qc2 = QuantumCircuit(2)
        p = PauliEvolutionGate(
            SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)]), time=2, synthesis=LieTrotter(reps=2)
        )
        qc2.append(p, [0, 1])
        circuits.append(qc2)

        qc3 = QuantumCircuit(4, 1)
        op1 = AnnotatedOperation(
            CXGate(), [InverseModifier(), ControlModifier(1), InverseModifier()]
        )
        qc3.append(op1, [0, 1, 2])
        qc3.append(MyParamGate(2 * theta), [3])
        qc3.append(CXGate().control(1, annotated=False), [0, 1, 2])
        circuits.append(qc3)

        q4 = QuantumCircuit(2)
        q4.append(PauliEvolutionGate(SparsePauliOp.from_list([("XX", 1)]), time=1), [0, 1])
        q4.append(AnnotatedOperation(XGate(), InverseModifier()), [0])
        circuits.append(q4)

        buf = io.BytesIO()
        qpy.dump(circuits, buf)
        return buf.getvalue()

    def test_parallel_load_extensive_GIL_release(self):
        dump = self.generate_dump_with_custom_operations()
        with patch.dict(os.environ, {"QISKIT_IN_PARALLEL": "TRUE"}):
            with io.BytesIO(dump) as buff:
                sequential = qpy.load(buff)
        with patch.dict(os.environ, {"QISKIT_IN_PARALLEL": "FALSE"}):
            with io.BytesIO(dump) as buff:
                parallel = qpy.load(buff)

        self.assertEqual(len(sequential), len(parallel))
        for seq, par in zip(sequential, parallel):
            self.assertEqual(seq, par)
