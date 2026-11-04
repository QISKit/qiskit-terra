# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test ConvertToPauliRotations pass"""

from ddt import ddt
import numpy as np

from qiskit.circuit import QuantumCircuit, Gate, Parameter, QuantumRegister, ClassicalRegister
from qiskit.transpiler import TranspilerError
from qiskit.transpiler.passes import ConvertToPauliRotations
from qiskit.quantum_info import Operator, Pauli
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.circuit.library import (
    C3SXGate,
    RC3XGate,
    C4XGate,
    MCXGate,
    PauliProductMeasurement,
    PauliProductRotationGate,
)
from test import combine, QiskitTestCase


@ddt
class TestConvertToPauliRotations(QiskitTestCase):
    """Test the ConvertToPauliRotations pass."""

    def setUp(self):
        super().setUp()
        self.standard_gates = get_standard_gate_name_mapping()

    @combine(angle=[0.12, -0.43], global_phase=[0, 1.0, -3.0])
    def test_standard_gates_transpiled(self, angle, global_phase):
        """Test that standard 1-qubit, 2-qubit and 3-qubit gates are translated into
        Pauli product rotatations correctly."""
        for gate in self.standard_gates.values():
            if not isinstance(gate, Gate):
                continue
            num_qubits = gate.num_qubits
            if num_qubits < 4:
                params = [angle * (i + 1) for i in range(len(gate.params))]
                qc = QuantumCircuit(max(num_qubits, 1))
                qc.global_phase = global_phase
                qc.append(gate.base_class(*params), range(num_qubits))
                qct = ConvertToPauliRotations()(qc)
                ops_names = set(qct.count_ops().keys())
                if ops_names:
                    self.assertEqual(ops_names, {"pauli_product_rotation"})
                self.assertEqual(Operator(qct), Operator(qc))

    def test_random_circuit(self):
        """Test that a pesudo-random circuit with 1-qubit, 2-qubit and 3-qubit gates
        is translated into Pauli product rotations correctly."""
        num_qubits = 5
        depth = 200
        seed = 1234
        qc = random_circuit(num_qubits=num_qubits, depth=depth, max_operands=3, seed=seed)
        qct = ConvertToPauliRotations()(qc)
        ops_names = set(qct.count_ops().keys())
        self.assertEqual(ops_names, {"pauli_product_rotation"})
        self.assertEqual(Operator(qct), Operator(qc))

    def test_random_circuit_measure_barrier_delay_reset(self):
        """Test that a pesudo-random circuit with 1-qubit, 2-qubit and 3-qubit gates,
        measurements, delays, resets and barriers,
        is translated into Pauli product rotations correctly."""
        num_qubits = 4
        depth = 10
        seed = 5678
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc1 = random_circuit(num_qubits=num_qubits, depth=depth, max_operands=3, seed=seed)
            qc.compose(qc1, inplace=True)
            qc.delay(i)
            qc.reset((i + 1) % num_qubits)
            qc.barrier()
        qc.measure_all()
        qct = ConvertToPauliRotations()(qc)
        ops_names = set(qct.count_ops().keys())
        self.assertEqual(
            ops_names,
            {"pauli_product_rotation", "pauli_product_measurement", "delay", "reset", "barrier"},
        )

    def test_parametrized_gates(self):
        """Test that a circuit with 1-qubit and 2-qubit parametrized gates
        is translated into Pauli product rotations correctly."""
        symbols = [Parameter("theta"), Parameter("phi"), Parameter("lam"), Parameter("gamma")]
        for gate in self.standard_gates.values():
            if not isinstance(gate, Gate):
                continue
            num_qubits = gate.num_qubits
            num_params = len(gate.params)
            if num_qubits < 3:
                params = symbols[:num_params]
                qc = QuantumCircuit(max(num_qubits, 1))
                qc.append(gate.base_class(*params), range(num_qubits))
                qct = ConvertToPauliRotations()(qc)
                ops_names = set(qct.count_ops().keys())
                if ops_names:
                    self.assertEqual(ops_names, {"pauli_product_rotation"})
                qc_bound = qc.assign_parameters([0.123] * num_params)
                qct_bound = qct.assign_parameters([0.123] * num_params)
                self.assertEqual(Operator(qct_bound), Operator(qc_bound))

    @combine(
        gate=[
            C3SXGate(),
            RC3XGate(),
            C4XGate(),
            MCXGate(5),
        ]
    )
    def test_unsupported_gates_raise_error(self, gate):
        """Test that unsupported gates raise a transpiler error."""
        num_qubits = gate.num_qubits
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(gate, range(num_qubits))
        qc.rzz(0.123, 0, 1)

        with self.assertRaises(TranspilerError):
            _ = ConvertToPauliRotations()(qc)

    def test_control_flow(self):
        """Test that simple control flow circuit works with the pass"""
        qc = QuantumCircuit(2, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc_true = QuantumCircuit(2, 1)
        qc_true.h(0)
        qc.if_else((0, True), qc_true, None, range(2), [0])
        qct = ConvertToPauliRotations()(qc)
        qc_exp = QuantumCircuit(2, 1)
        qc_exp.append(PauliProductRotationGate(Pauli("Y"), np.pi / 2), [0])
        qc_exp.append(PauliProductRotationGate(Pauli("X"), np.pi), [0])
        qc_exp.append(PauliProductMeasurement(Pauli("Z")), [0], [0])
        qc_exp.global_phase = np.pi / 2
        qc_exp_true = QuantumCircuit(2, 1)
        qc_exp_true.global_phase = np.pi / 2
        qc_exp_true.append(PauliProductRotationGate(Pauli("Y"), np.pi / 2), [0])
        qc_exp_true.append(PauliProductRotationGate(Pauli("X"), np.pi), [0])
        qc_exp.if_else((0, True), qc_exp_true, None, range(2), [0])
        self.assertEqual(qct, qc_exp)

    def test_nested_control_flow(self):
        """Test that nested control flow circuit works with the pass"""
        # subcircuit with parameterized gate
        theta = Parameter("theta")
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.ry(theta, 0)
        # build input circuit by composing a subcircuit into nested control flow
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.compose(qc1, [0, 1], inplace=True)
        with qc.for_loop(range(3)):
            with qc.while_loop((cr, 0)):
                qc.compose(qc1, [0, 1], inplace=True)
        qct = ConvertToPauliRotations()(qc)
        # qc2 is the transpiled equivalent of subcircuit qc1
        qc2 = QuantumCircuit(2)
        qc2.append(PauliProductRotationGate(Pauli("Y"), np.pi / 2), [0])
        qc2.append(PauliProductRotationGate(Pauli("X"), np.pi), [0])
        qc2.append(PauliProductRotationGate(Pauli("Y"), theta), [0])
        qc2.global_phase = np.pi / 2
        # expected output circuit of transpiled qc, should be equivalent to qct
        qc_exp = QuantumCircuit(qr, cr)
        qc_exp.compose(qc2, [0, 1], inplace=True)
        with qc_exp.for_loop(range(3)):
            with qc_exp.while_loop((cr, 0)):
                qc_exp.compose(qc2, [0, 1], inplace=True)
        self.assertEqual(qct, qc_exp)

    def test_pprs_and_ppms_are_unchanged(self):
        """Test that the pass leaves Pauli product rotations and measurements unchanged."""
        qc = QuantumCircuit(4, 2)
        qc.append(PauliProductRotationGate(Pauli("XYZ"), 0.1), [1, 2, 3])
        qc.append(PauliProductMeasurement(Pauli("-XZ")), [1, 2], [1])

        qct = ConvertToPauliRotations()(qc)
        self.assertEqual(qc, qct)
