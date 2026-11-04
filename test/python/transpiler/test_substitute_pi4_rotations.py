# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Substitute Pi/4-Rotations optimization pass"""

from ddt import ddt
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_clifford_circuit
from qiskit.transpiler.passes import SubstitutePi4Rotations
from qiskit.quantum_info import Operator, get_clifford_gate_names
from qiskit.circuit.library import (
    RXGate,
    RYGate,
    RZGate,
    PhaseGate,
    U1Gate,
    RZZGate,
    RXXGate,
    RZXGate,
    RYYGate,
    CPhaseGate,
    CU1Gate,
    CRZGate,
    CRXGate,
    CRYGate,
)
from test import combine, QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestSubstitutePi4Rotations(QiskitTestCase):
    """Test the Substitute Pi4-Rotations optimization pass."""

    @combine(
        multiple=[*range(16), 23, 42, -5, -8, -17, -22, -35],
        gate=[RXGate, RYGate, RZGate, PhaseGate, U1Gate, RZZGate, RXXGate, RZXGate, RYYGate],
        global_phase=[0, 1.0, -2.0],
        approximation_degree=[1, 0.99999],
        eps=[0, 1e-10],
    )
    def test_rotation_gates_transpiled(
        self, multiple, gate, global_phase, approximation_degree, eps
    ):
        """Test that rotations gates are translated into Clifford+T+Tdg correctly."""
        angle = np.pi / 4 * multiple + eps
        num_qubits = gate(angle).num_qubits
        qc = QuantumCircuit(num_qubits)
        qc.global_phase = global_phase
        qc.append(gate(angle), range(num_qubits))
        qct = SubstitutePi4Rotations(approximation_degree)(qc)
        ops = qct.count_ops()
        clifford_t_names = get_clifford_gate_names() + ["t"] + ["tdg"]
        self.assertEqual(Operator(qct), Operator(qc))
        self.assertLessEqual(set(ops.keys()), set(clifford_t_names))
        self.assertLessEqual(ops.get("t", 0) + ops.get("tdg", 0), 1)  # at most one t/tdg gate
        if multiple % 2 == 0:  # only clifford gates
            self.assertLessEqual(set(ops.keys()), set(get_clifford_gate_names()))
            self.assertEqual(ops.get("t", 0) + ops.get("tdg", 0), 0)
        if num_qubits == 1 or gate(angle).name == "rzz":
            self.assertLessEqual(qct.size(), 4)
            if multiple % 2 == 0:  # only clifford gates
                self.assertLessEqual(qct.size(), 3)
        else:
            self.assertLessEqual(qct.size(), 8)
            if multiple % 2 == 0:  # only clifford gates
                self.assertLessEqual(qct.size(), 7)

    @combine(
        multiple=[*range(8), 23, 42, -5, -8, -17, -22, -35],
        gate=[CPhaseGate, CU1Gate, CRZGate, CRXGate, CRYGate],
        global_phase=[0, 1.0, -2.0],
        approximation_degree=[1, 0.99999],
        eps=[0, 1e-10],
    )
    def test_controlled_rotation_gates_transpiled(
        self, multiple, gate, global_phase, approximation_degree, eps
    ):
        """Test that controlled rotations gates are translated into Clifford+T+Tdg correctly."""
        angle = np.pi / 2 * multiple + eps
        num_qubits = gate(angle).num_qubits
        qc = QuantumCircuit(num_qubits)
        qc.global_phase = global_phase
        qc.append(gate(angle), range(num_qubits))
        qct = SubstitutePi4Rotations(approximation_degree)(qc)
        ops = qct.count_ops()
        clifford_t_names = get_clifford_gate_names() + ["t"] + ["tdg"]
        self.assertEqual(Operator(qct), Operator(qc))
        self.assertLessEqual(set(ops.keys()), set(clifford_t_names))
        self.assertLessEqual(ops.get("t", 0) + ops.get("tdg", 0), 3)  # at most 3 t/tdg gates
        if gate(angle).name not in {"cp", "cu1"}:
            self.assertLessEqual(ops.get("t", 0) + ops.get("tdg", 0), 2)  # at most 2 t/tdg gates
        if multiple % 2 == 0:  # only clifford gates
            self.assertLessEqual(set(ops.keys()), set(get_clifford_gate_names()))
            self.assertEqual(ops.get("t", 0) + ops.get("tdg", 0), 0)
        self.assertLessEqual(qct.size(), 8)
        if multiple % 2 == 0:  # only clifford gates
            self.assertLessEqual(qct.size(), 4)

    @combine(
        multiple=[*range(16)],
        eps=[0.001, -0.001],
        gate=[
            RXGate,
            RYGate,
            RZGate,
            PhaseGate,
            U1Gate,
            RZZGate,
            RXXGate,
            RZXGate,
            RYYGate,
            CPhaseGate,
            CU1Gate,
            CRZGate,
            CRXGate,
            CRYGate,
        ],
        approximation_degree=[1, 0.9999999],
    )
    def test_rotation_gates_do_not_change(self, multiple, eps, gate, approximation_degree):
        """Test that the transpiler pass does not change the gates for angles that are
        not pi/4 rotations."""
        angle = np.pi / 4 * multiple + eps
        if gate(angle).name in {"cp", "crx", "cry", "crz"}:
            # controlled-rotations: the angle is a multiple of pi/2
            angle = np.pi / 2 * multiple + 2 * eps
        num_qubits = gate(angle).num_qubits
        qc = QuantumCircuit(num_qubits)
        qc.append(gate(angle), range(num_qubits))
        qct = SubstitutePi4Rotations(approximation_degree)(qc)
        self.assertEqual(qc, qct)

    def test_random_clifford_t_rotation_circuit(self):
        """Test that pseudo-random Clifford+T+Tdg+rotation circuits are transpiled correctly."""
        num_qubits = 5
        num_gates = 10
        qc = QuantumCircuit(num_qubits)
        qc.global_phase = 2.0
        for idx in range(num_qubits):
            cliff_qc = random_clifford_circuit(num_qubits, num_gates, seed=idx)
            qc.compose(cliff_qc, inplace=True)
            qc.t(idx)
            qc.tdg((idx + 1) % num_qubits)
            qc.rx(np.pi / 4 * (idx + num_qubits + 2), (idx + 2) % num_qubits)
            qc.ry(np.pi / 4 * (idx + num_qubits + 3), (idx + 3) % num_qubits)
            qc.rz(np.pi / 4 * (idx + num_qubits + 4), (idx + 4) % num_qubits)
            qc.p(np.pi / 4 * (idx + num_qubits + 2), (idx + 4) % num_qubits)
            qc.rzz(np.pi / 4 * idx, idx, (idx + 3) % num_qubits)
            qc.rxx(
                np.pi / 4 * (idx + num_qubits + 1), (idx + 1) % num_qubits, (idx - 1) % num_qubits
            )
            qc.rzx(
                np.pi / 4 * (idx + num_qubits + 2), (idx + 2) % num_qubits, (idx - 2) % num_qubits
            )
            qc.ryy(
                np.pi / 4 * (idx + num_qubits + 3), (idx + 3) % num_qubits, (idx - 3) % num_qubits
            )
            qc.cp(np.pi / 2 * (idx + 1), (idx + 2) % num_qubits, (idx + 3) % num_qubits)
            qc.crz(np.pi / 2 * (idx + 2), (idx + 1) % num_qubits, (idx + 4) % num_qubits)
            qc.crx(np.pi / 2 * (idx + 3), (idx + 2) % num_qubits, (idx + 1) % num_qubits)
            qc.cry(np.pi / 2 * (idx + 4), (idx + 1) % num_qubits, (idx + 2) % num_qubits)

        qct = SubstitutePi4Rotations()(qc)
        clifford_t_names = get_clifford_gate_names() + ["t"] + ["tdg"]
        self.assertEqual(Operator(qct), Operator(qc))
        self.assertLessEqual(set(qct.count_ops().keys()), set(clifford_t_names))

    def test_explicit_general_circuit(self):
        """Test the pass on a general circuit that includes rotation gates, other gates, and barriers."""
        qc = QuantumCircuit(3)
        qc.global_phase = -2.0
        qc.t(0)
        qc.rx(np.pi / 4, 2)
        qc.barrier()
        qc.cx(0, 1)
        qc.ry(np.pi / 2, 1)
        qc.rz(0.1, 0)
        qc.rzz(np.pi / 4, 0, 1)

        qct = SubstitutePi4Rotations()(qc)

        expected = QuantumCircuit(3)
        expected.global_phase = -2.0 - 2 * np.pi / 8
        expected.t(0)
        expected.h(2)
        expected.t(2)
        expected.h(2)
        expected.barrier()
        expected.cx(0, 1)
        expected.z(1)
        expected.h(1)
        expected.rz(0.1, 0)
        expected.cx(0, 1)
        expected.t(1)
        expected.cx(0, 1)

        self.assertEqual(qct, expected)
        self.assertEqual(Operator(qct), Operator(qc))

    def test_nested_control_flow(self):
        """Test unrolling nested control flow blocks."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)
        qc1 = QuantumCircuit(2)
        qc1.rzz(np.pi / 4, 0, 1)
        qc2 = QuantumCircuit(2)
        qc2.global_phase = -np.pi / 8
        qc2.cx(0, 1)
        qc2.t(1)
        qc2.cx(0, 1)

        qc = QuantumCircuit(qr, cr)
        qc.compose(qc1, [0, 1], inplace=True)
        with qc.for_loop(range(3)):
            with qc.while_loop((cr, 0)):
                qc.compose(qc1, [0, 1], inplace=True)
        qct = SubstitutePi4Rotations()(qc)

        qc_exp = QuantumCircuit(qr, cr)
        qc_exp.compose(qc2, [0, 1], inplace=True)
        with qc_exp.for_loop(range(3)):
            with qc_exp.while_loop((cr, 0)):
                qc_exp.compose(qc2, [0, 1], inplace=True)

        self.assertEqual(qct, qc_exp)
