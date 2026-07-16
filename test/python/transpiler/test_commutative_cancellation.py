# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Gate cancellation pass testing"""

import unittest

import numpy as np
import ddt

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import U1Gate, RZGate, PhaseGate, CXGate, SXGate
from qiskit.circuit.parameter import Parameter
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.transpiler.target import Target
from qiskit.transpiler import PassManager, PropertySet
from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation, FixedPoint, Size
from qiskit.quantum_info import Operator
from test import QiskitTestCase, combine


@ddt.ddt
class TestCommutativeCancellation(QiskitTestCase):
    """Test the CommutativeCancellation pass."""

    def setUp(self):
        super().setUp()
        self.com_pass_ = CommutationAnalysis()
        self.pass_ = CommutativeCancellation()
        self.pset = self.pass_.property_set = PropertySet()

    def test_all_gates(self):
        """Test all gates on 1 and 2 qubits

        q0:-[H]-[H]--[x]-[x]--[y]-[y]--[rz]-[rz]--[u1]-[u1]-[rx]-[rx]---.--.--.--.--.--.-
                                                                        |  |  |  |  |  |
        q1:-------------------------------------------------------------X--X--Y--Y--.--.-

        =

        qr0:---[u1]---

        qr1:----------
        """
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.x(qr[0])
        circuit.x(qr[0])
        circuit.y(qr[0])
        circuit.y(qr[0])
        circuit.rz(0.5, qr[0])
        circuit.rz(0.5, qr[0])
        circuit.append(U1Gate(0.5), [qr[0]])  # TODO this should work with Phase gates too
        circuit.append(U1Gate(0.5), [qr[0]])
        circuit.rx(0.5, qr[0])
        circuit.rx(0.5, qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])

        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(RZGate(2.0), [qr[0]])
        expected.rx(1.0, qr[0])
        expected.global_phase = 0.5
        self.assertEqual(expected, new_circuit)

    def test_2pi_multiples(self):
        """Test 2pi multiples are handled with the correct phase they introduce."""
        for eps in [0, 1e-10, -1e-10]:
            for sign in [-1, 1]:
                qc = QuantumCircuit(1)
                qc.rz(sign * np.pi + eps, 0)
                qc.rz(sign * np.pi, 0)

                with self.subTest(msg="single 2pi", sign=sign, eps=eps):
                    tqc = CommutativeCancellation()(qc)
                    self.assertEqual(0, len(tqc.count_ops()))
                    self.assertAlmostEqual(np.pi, tqc.global_phase)

            for sign_x in [-1, 1]:
                for sign_z in [-1, 1]:
                    qc = QuantumCircuit(2)
                    qc.rx(sign_x * np.pi + eps, 0)
                    qc.rx(sign_x * np.pi, 0)
                    qc.rz(sign_z * np.pi, 1)
                    qc.rz(sign_z * np.pi, 1)

                    with self.subTest(msg="two 2pi", sign_x=sign_x, sign_z=sign_z, eps=eps):
                        tqc = CommutativeCancellation()(qc)
                        self.assertEqual(0, len(tqc.count_ops()))
                        self.assertAlmostEqual(0, tqc.global_phase)

    def test_4pi_multiples(self):
        """Test 4pi multiples are removed w/o changing the global phase."""
        for eps in [0, 1e-10, -1e-10]:
            for sign in [-1, 1]:
                qc = QuantumCircuit(1)
                qc.rz(sign * np.pi + eps, 0)
                qc.rz(sign * 6 * np.pi, 0)
                qc.rz(sign * np.pi, 0)

                with self.subTest(sign=sign, eps=eps):
                    tqc = CommutativeCancellation()(qc)
                    self.assertEqual(0, len(tqc.count_ops()))
                    self.assertAlmostEqual(0, tqc.global_phase)

    def test_fixed_rotation_accumulation(self):
        """Test accumulating gates with fixed angles (T, S) works correctly."""
        cc = CommutativeCancellation()

        # test for U1, P and RZ as target gate
        for gate_cls in [RZGate, PhaseGate, U1Gate]:
            qc = QuantumCircuit(1)
            gate = gate_cls(0.2)
            qc.append(gate, [0])
            qc.t(0)
            qc.s(0)

            tqc = cc(qc)
            self.assertTrue(np.allclose(Operator(qc).data, Operator(tqc).data))

    @ddt.data(RZGate(np.pi / 4), PhaseGate(np.pi / 4), U1Gate(np.pi / 4))
    def test_p_u1_2pi_accumulation(self, gate):
        """Test different Z rotations."""
        basis = ["t", "rz", "cx"]
        if gate.name not in basis:
            basis = [gate.name] + basis
        cc = CommutativeCancellation(basis_gates=basis)
        reps = (7, 15, 23, 31)

        tqcs = []
        for rep in reps:
            qc = QuantumCircuit(1)
            qc.append(gate, [0])
            for _ in range(rep):
                qc.t(0)
            tqcs.append(cc(qc))

        phase = -gate.params[0] / 2 if gate.name == "rz" else 0.0
        expected = [QuantumCircuit(1, global_phase=phase)] * len(tqcs)
        self.assertEqual(tqcs, expected)

    def test_commutative_circuit1(self):
        """A simple circuit where three CNOTs commute, the first and the last cancel.

        qr0:----.---------------.--       qr0:------------
                |               |
        qr1:---(+)-----(+)-----(+)-   =   qr1:-------(+)--
                        |                             |
        qr2:---[H]------.----------       qr2:---[H]--.---
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[2])
        circuit.cx(qr[2], qr[1])
        circuit.cx(qr[0], qr[1])

        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr[2])
        expected.cx(qr[2], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_consecutive_cnots(self):
        """A simple circuit equals identity

        qr0:----.- ----.--       qr0:------------
                |      |
        qr1:---(+)----(+)-   =   qr1:------------
        """

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)

        self.assertEqual(expected, new_circuit)

    def test_consecutive_cnots2(self):
        """
        Two CNOTs that equals identity, with rotation gates inserted.
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.rx(np.pi, qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.rx(np.pi, qr[0])

        passmanager = PassManager()
        passmanager.append(
            DoWhileController(
                [
                    CommutationAnalysis(),
                    CommutativeCancellation(),
                    Size(),
                    FixedPoint("size"),
                ],
                do_while=lambda property_set: not property_set["size_fixed_point"],
            )
        )
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(qr, global_phase=np.pi)  # RX(2pi) = -I = exp(i pi) I

        self.assertEqual(expected, new_circuit)
        self.assertTrue(np.allclose(Operator(circuit).data, Operator(expected).data))

    @combine(
        basis_gates=[["rz", "sx", "x"], ["rz", "sx"], ["rz", "rx"]],
        circuit_gate=["x", "rx", "sx"],
        name="basis_gates={basis_gates}_circuit_gate={circuit_gate}",
    )
    def test_xgate_accumulation(self, basis_gates, circuit_gate):
        circuit = QuantumCircuit(2)
        if circuit_gate == "rx":
            circuit.rx(np.pi / 2, 0)
            circuit.rx(np.pi / 2, 0)
        elif circuit_gate == "sx":
            circuit.sx(0)
            circuit.sx(0)
        else:
            circuit.x(0)
            circuit.x(0)
            circuit.x(0)
        commuter_pass = CommutativeCancellation(basis_gates=basis_gates)
        result = commuter_pass(circuit)
        op_counts = result.count_ops()
        self.assertEqual(Operator(circuit), Operator(result))
        if "x" in basis_gates or "x" == circuit_gate:
            self.assertEqual(op_counts.get("x", 0), 1)
            self.assertNotIn("sx", op_counts)
            self.assertNotIn("rx", op_counts)
        elif "sx" in basis_gates or "sx" == circuit_gate:
            self.assertEqual(op_counts.get("sx", 0), 2)
            self.assertNotIn("x", op_counts)
            self.assertNotIn("rx", op_counts)
        else:
            self.assertEqual(op_counts.get("rx", 0), 1)
            self.assertNotIn("sx", op_counts)
            self.assertNotIn("x", op_counts)

    @combine(
        basis_gates=[["rz", "sx", "x"], ["u1", "sx"], ["p", "sx"]],
        circuit_gate=["t", "tdg", "s", "sdg", "rz", "z"],
        name="basis_gates={basis_gates}_circuit_gate={circuit_gate}",
    )
    def test_zgate_accumulation(self, basis_gates, circuit_gate):
        circuit = QuantumCircuit(2)
        if circuit_gate == "t":
            circuit.t(0)
            circuit.t(0)
            circuit.t(0)
            circuit.t(0)
        elif circuit_gate == "tdg":
            circuit.tdg(0)
            circuit.tdg(0)
            circuit.tdg(0)
            circuit.tdg(0)
        elif circuit_gate == "s":
            circuit.s(0)
            circuit.s(0)
        elif circuit_gate == "sdg":
            circuit.sdg(0)
            circuit.sdg(0)
        elif circuit_gate == "rz":
            circuit.rz(np.pi / 2, 0)
            circuit.rz(np.pi / 2, 0)
        else:
            circuit.z(0)
            circuit.z(0)
            circuit.z(0)
        commuter_pass = CommutativeCancellation(basis_gates=basis_gates)
        result = commuter_pass(circuit)
        op_counts = result.count_ops()
        self.assertEqual(Operator(circuit), Operator(result))
        if "rz" in basis_gates or circuit_gate == "rz":
            self.assertEqual(op_counts, {"rz": 1})
        elif "u1" in basis_gates:
            self.assertEqual(op_counts, {"u1": 1})
        else:
            self.assertEqual(op_counts, {"p": 1})

    def test_2_alternating_cnots(self):
        """A simple circuit where nothing should be cancelled.

        qr0:----.- ---(+)-       qr0:----.----(+)-
                |      |                 |     |
        qr1:---(+)-----.--   =   qr1:---(+)----.--

        """

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.cx(qr[0], qr[1])
        expected.cx(qr[1], qr[0])

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot(self):
        """A simple circuit where nothing should be cancelled.

        qr0:----.------[X]------.--       qr0:----.------[X]------.--
                |               |                 |               |
        qr1:---(+)-------------(+)-   =   qr1:---(+)-------------(+)-
        """

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[0])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.cx(qr[0], qr[1])
        expected.x(qr[0])
        expected.cx(qr[0], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot1(self):
        """A simple circuit where the two cnots should be cancelled.

        qr0:----.------[Z]------.--       qr0:---[Z]---
                |               |
        qr1:---(+)-------------(+)-   =   qr1:---------
        """

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.z(qr[0])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.z(qr[0])

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot2(self):
        """A simple circuit where the two cnots should be cancelled.

        qr0:----.------[T]------.--       qr0:---[T]---
                |               |
        qr1:---(+)-------------(+)-   =   qr1:---------
        """

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.t(qr[0])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.t(qr[0])

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot3(self):
        """A simple circuit where the two cnots should be cancelled.

        qr0:----.------[Rz]------.--       qr0:---[Rz]---
                |                |
        qr1:---(+)-------- -----(+)-   =   qr1:----------
        """

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.rz(np.pi / 3, qr[0])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.rz(np.pi / 3, qr[0])

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot4(self):
        """A simple circuit where the two cnots should be cancelled.

        qr0:----.------[T]------.--       qr0:---[T]---
                |               |
        qr1:---(+)-------------(+)-   =   qr1:---------
        """

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.t(qr[0])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.t(qr[0])

        self.assertEqual(expected, new_circuit)

    def test_target_bit_of_cnot(self):
        """A simple circuit where nothing should be cancelled.

        qr0:----.---------------.--       qr0:----.---------------.--
                |               |                 |               |
        qr1:---(+)-----[Z]-----(+)-   =   qr1:---(+)----[Z]------(+)-
        """

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.z(qr[1])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.cx(qr[0], qr[1])
        expected.z(qr[1])
        expected.cx(qr[0], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_target_bit_of_cnot1(self):
        """A simple circuit where nothing should be cancelled.

        qr0:----.---------------.--       qr0:----.---------------.--
                |               |                 |               |
        qr1:---(+)-----[T]-----(+)-   =   qr1:---(+)----[T]------(+)-
        """

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.t(qr[1])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.cx(qr[0], qr[1])
        expected.t(qr[1])
        expected.cx(qr[0], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_target_bit_of_cnot2(self):
        """A simple circuit where nothing should be cancelled.

        qr0:----.---------------.--       qr0:----.---------------.--
                |               |                 |               |
        qr1:---(+)-----[Rz]----(+)-   =   qr1:---(+)----[Rz]-----(+)-
        """

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.rz(np.pi / 3, qr[1])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.cx(qr[0], qr[1])
        expected.rz(np.pi / 3, qr[1])
        expected.cx(qr[0], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_commutative_circuit2(self):
        """
        A simple circuit where three CNOTs commute, the first and the last cancel,
        also two X gates cancel and two Rz gates combine.

        qr0:----.---------------.--------     qr0:-------------
                |               |
        qr1:---(+)---(+)--[X]--(+)--[X]--  =  qr1:--------(+)--
                      |                                    |
        qr2:---[Rz]---.---[Rz]-[T]--[S]--     qr2:--[U1]---.---
        """

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.rz(np.pi / 3, qr[2])
        circuit.cx(qr[2], qr[1])
        circuit.rz(np.pi / 3, qr[2])
        circuit.t(qr[2])
        circuit.s(qr[2])
        circuit.x(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[1])

        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(qr)
        expected.append(RZGate(np.pi * 17 / 12), [qr[2]])
        expected.cx(qr[2], qr[1])
        expected.global_phase = (np.pi * 17 / 12 - (2 * np.pi / 3)) / 2
        self.assertEqual(expected, new_circuit)

    def test_commutative_circuit3(self):
        """
        A simple circuit where three CNOTs commute, the first and the last cancel,
        also two X gates cancel and two Rz gates combine.

        qr0:-------.------------------.-------------     qr0:-------------
                   |                  |
        qr1:------(+)------(+)--[X]--(+)-------[X]--  =  qr1:--------(+)--
                            |                                         |
        qr2:------[Rz]--.---.----.---[Rz]-[T]--[S]--     qr2:--[U1]---.---
                        |        |
        qr3:-[Rz]--[X]-(+)------(+)--[X]-[Rz]-------     qr3:--[Rz]-------
        """

        qr = QuantumRegister(4, "qr")
        circuit = QuantumCircuit(qr)

        circuit.cx(qr[0], qr[1])
        circuit.rz(np.pi / 3, qr[2])
        circuit.rz(np.pi / 3, qr[3])
        circuit.x(qr[3])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[2], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.rz(np.pi / 3, qr[2])
        circuit.t(qr[2])
        circuit.x(qr[3])
        circuit.rz(np.pi / 3, qr[3])
        circuit.s(qr[2])
        circuit.x(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[1])

        passmanager = PassManager()
        passmanager.append(
            DoWhileController(
                [
                    CommutationAnalysis(),
                    CommutativeCancellation(),
                    Size(),
                    FixedPoint("size"),
                ],
                do_while=lambda property_set: not property_set["size_fixed_point"],
            )
        )
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(qr)
        expected.append(RZGate(np.pi * 17 / 12), [qr[2]])
        expected.append(RZGate(np.pi * 2 / 3), [qr[3]])
        expected.cx(qr[2], qr[1])
        expected.global_phase = 3 * np.pi / 8
        self.assertEqual(
            expected, new_circuit, msg=f"expected:\n{expected}\nnew_circuit:\n{new_circuit}"
        )

    def test_cnot_cascade(self):
        """
        A cascade of CNOTs that equals identity.
        """

        qr = QuantumRegister(10, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[5], qr[6])
        circuit.cx(qr[6], qr[7])
        circuit.cx(qr[7], qr[8])
        circuit.cx(qr[8], qr[9])

        circuit.cx(qr[8], qr[9])
        circuit.cx(qr[7], qr[8])
        circuit.cx(qr[6], qr[7])
        circuit.cx(qr[5], qr[6])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[0], qr[1])

        passmanager = PassManager()
        # passmanager.append(CommutativeCancellation())
        passmanager.append(
            DoWhileController(
                [
                    CommutationAnalysis(),
                    CommutativeCancellation(),
                    Size(),
                    FixedPoint("size"),
                ],
                do_while=lambda property_set: not property_set["size_fixed_point"],
            )
        )
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(qr)

        self.assertEqual(expected, new_circuit)

    def test_cnot_cascade1(self):
        """
        A cascade of CNOTs that equals identity, with rotation gates inserted.
        """

        qr = QuantumRegister(10, "qr")
        circuit = QuantumCircuit(qr)
        circuit.rx(np.pi, qr[0])
        circuit.rx(np.pi, qr[1])
        circuit.rx(np.pi, qr[2])
        circuit.rx(np.pi, qr[3])
        circuit.rx(np.pi, qr[4])
        circuit.rx(np.pi, qr[5])
        circuit.rx(np.pi, qr[6])
        circuit.rx(np.pi, qr[7])
        circuit.rx(np.pi, qr[8])
        circuit.rx(np.pi, qr[9])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[5], qr[6])
        circuit.cx(qr[6], qr[7])
        circuit.cx(qr[7], qr[8])
        circuit.cx(qr[8], qr[9])
        circuit.cx(qr[8], qr[9])
        circuit.cx(qr[7], qr[8])
        circuit.cx(qr[6], qr[7])
        circuit.cx(qr[5], qr[6])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[0], qr[1])
        circuit.rx(np.pi, qr[0])
        circuit.rx(np.pi, qr[1])
        circuit.rx(np.pi, qr[2])
        circuit.rx(np.pi, qr[3])
        circuit.rx(np.pi, qr[4])
        circuit.rx(np.pi, qr[5])
        circuit.rx(np.pi, qr[6])
        circuit.rx(np.pi, qr[7])
        circuit.rx(np.pi, qr[8])
        circuit.rx(np.pi, qr[9])
        passmanager = PassManager()
        # passmanager.append(CommutativeCancellation())
        passmanager.append(
            DoWhileController(
                [
                    CommutationAnalysis(),
                    CommutativeCancellation(),
                    Size(),
                    FixedPoint("size"),
                ],
                do_while=lambda property_set: not property_set["size_fixed_point"],
            )
        )
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(qr)

        self.assertEqual(expected, new_circuit)

    def test_basis_01(self):
        """Test basis priority change, phase gate"""
        circuit = QuantumCircuit(1)
        circuit.s(0)
        circuit.z(0)
        circuit.t(0)
        circuit.rz(np.pi, 0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation(basis_gates=["cx", "p", "sx"]))
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(1)
        expected.rz(11 * np.pi / 4, 0)
        expected.global_phase = 11 * np.pi / 4 / 2 - np.pi / 2

        self.assertEqual(new_circuit, expected)

    def test_target_basis_01(self):
        """Test basis priority change, phase gate, with target."""
        circuit = QuantumCircuit(1)
        circuit.s(0)
        circuit.z(0)
        circuit.t(0)
        circuit.rz(np.pi, 0)
        theta = Parameter("theta")
        target = Target(num_qubits=2)
        target.add_instruction(CXGate())
        target.add_instruction(PhaseGate(theta))
        target.add_instruction(SXGate())
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation(target=target))
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(1)
        expected.rz(11 * np.pi / 4, 0)
        expected.global_phase = 11 * np.pi / 4 / 2 - np.pi / 2

        self.assertEqual(new_circuit, expected)

    def test_basis_02(self):
        """Test basis priority change, Rz gate"""
        circuit = QuantumCircuit(1)
        circuit.s(0)
        circuit.z(0)
        circuit.t(0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation(basis_gates=["cx", "rz", "sx"]))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(1)
        expected.rz(7 * np.pi / 4, 0)
        expected.global_phase = 7 * np.pi / 4 / 2
        self.assertEqual(new_circuit, expected)

    def test_basis_03(self):
        """Test no specified basis"""
        circuit = QuantumCircuit(1)
        circuit.s(0)
        circuit.z(0)
        circuit.t(0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(1)
        expected.s(0)
        expected.z(0)
        expected.t(0)
        self.assertEqual(new_circuit, expected)

    def test_basis_global_phase_01(self):
        """Test no specified basis, rz"""
        circ = QuantumCircuit(1)
        circ.rz(np.pi / 2, 0)
        circ.p(np.pi / 2, 0)
        circ.p(np.pi / 2, 0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        ccirc = passmanager.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_basis_global_phase_02(self):
        """Test no specified basis, p"""
        circ = QuantumCircuit(1)
        circ.p(np.pi / 2, 0)
        circ.rz(np.pi / 2, 0)
        circ.p(np.pi / 2, 0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        ccirc = passmanager.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_basis_global_phase_03(self):
        """Test global phase preservation if cumulative z-rotation is 0"""
        circ = QuantumCircuit(1)
        circ.rz(np.pi / 2, 0)
        circ.p(np.pi / 2, 0)
        circ.z(0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        ccirc = passmanager.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_simple_if_else(self):
        """Test that the pass is not confused by if-else."""
        base_test1 = QuantumCircuit(3, 3)
        base_test1.x(1)
        base_test1.cx(0, 1)
        base_test1.x(1)

        base_test2 = QuantumCircuit(3, 3)
        base_test2.rz(0.1, 1)
        base_test2.rz(0.1, 1)

        test = QuantumCircuit(3, 3)
        test.h(0)
        test.x(0)
        test.rx(0.2, 0)
        test.measure(0, 0)
        test.x(0)
        test.if_else(
            (test.clbits[0], True), base_test1.copy(), base_test2.copy(), test.qubits, test.clbits
        )

        expected = QuantumCircuit(3, 3, global_phase=np.pi / 2)
        expected.h(0)
        expected.rx(np.pi + 0.2, 0)  # transforming X into RX(pi) introduces a pi/2 global phase
        expected.measure(0, 0)
        expected.x(0)

        expected_test1 = QuantumCircuit(3, 3)
        expected_test1.cx(0, 1)

        expected_test2 = QuantumCircuit(3, 3)
        expected_test2.rz(0.2, 1)

        expected.if_else(
            (expected.clbits[0], True),
            expected_test1.copy(),
            expected_test2.copy(),
            expected.qubits,
            expected.clbits,
        )

        passmanager = PassManager([CommutationAnalysis(), CommutativeCancellation()])
        new_circuit = passmanager.run(test)
        self.assertEqual(new_circuit, expected)

    def test_nested_control_flow(self):
        """Test that the pass does not add barrier into nested control flow."""
        level2_test = QuantumCircuit(2, 1)
        level2_test.cz(0, 1)
        level2_test.cz(0, 1)
        level2_test.cz(0, 1)
        level2_test.measure(0, 0)

        level1_test = QuantumCircuit(2, 1)
        level1_test.for_loop((0,), None, level2_test.copy(), level1_test.qubits, level1_test.clbits)
        level1_test.h(0)
        level1_test.h(0)
        level1_test.measure(0, 0)

        test = QuantumCircuit(2, 1)
        test.while_loop((test.clbits[0], True), level1_test.copy(), test.qubits, test.clbits)
        test.measure(0, 0)

        level2_expected = QuantumCircuit(2, 1)
        level2_expected.cz(0, 1)
        level2_expected.measure(0, 0)

        level1_expected = QuantumCircuit(2, 1)
        level1_expected.for_loop(
            (0,), None, level2_expected.copy(), level1_expected.qubits, level1_expected.clbits
        )
        level1_expected.measure(0, 0)

        expected = QuantumCircuit(2, 1)
        expected.while_loop(
            (expected.clbits[0], True), level1_expected.copy(), expected.qubits, expected.clbits
        )
        expected.measure(0, 0)

        passmanager = PassManager([CommutationAnalysis(), CommutativeCancellation()])
        new_circuit = passmanager.run(test)
        self.assertEqual(new_circuit, expected)

    def test_cancellation_not_crossing_block_boundary(self):
        """Test that the pass does cancel gates across control flow op block boundaries."""
        test1 = QuantumCircuit(2, 2)
        test1.x(1)
        with test1.if_test((0, False)):
            test1.cx(0, 1)
            test1.x(1)

        passmanager = PassManager([CommutationAnalysis(), CommutativeCancellation()])
        new_circuit = passmanager.run(test1)
        self.assertEqual(new_circuit, test1)

    def test_cancellation_not_crossing_between_blocks(self):
        """Test that the pass does cancel gates in different control flow ops."""
        test2 = QuantumCircuit(2, 2)
        with test2.if_test((0, True)):
            test2.x(1)
        with test2.if_test((0, True)):
            test2.cx(0, 1)
            test2.x(1)

        passmanager = PassManager([CommutationAnalysis(), CommutativeCancellation()])
        new_circuit = passmanager.run(test2)
        self.assertEqual(new_circuit, test2)

    def test_no_intransitive_cancellation(self):
        """Test that no unsound optimization occurs due to "intransitively-commuting" gates.
        See: https://github.com/Qiskit/qiskit-terra/issues/8020.
        """
        circ = QuantumCircuit(1)

        circ.x(0)
        circ.id(0)
        circ.h(0)
        circ.id(0)
        circ.x(0)

        passmanager = PassManager([CommutationAnalysis(), CommutativeCancellation()])
        new_circuit = passmanager.run(circ)
        self.assertEqual(new_circuit, circ)

    def test_overloaded_standard_gate_name(self):
        """Validate the pass works with custom gates using overloaded names

        See: https://github.com/Qiskit/qiskit/issues/13988 for more details.
        """
        qasm_str = """OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1
{
 rx(pi/2) q0;
 rx(pi/2) q1;
 cx q0,q1;
 rz(0.37801308) q1;
 cx q0,q1;
 rx(-pi/2) q0;
 rx(-pi/2) q1;
}
qreg q0[2];
creg c0[2];
z q0[0];
ryy(1.2182379) q0[0],q0[1];
z q0[0];
measure q0[0] -> c0[0];
measure q0[1] -> c0[1];
"""
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        cancellation_pass = CommutativeCancellation()
        res = cancellation_pass(qc)
        # We don't cancel any gates with a custom rzz gate
        self.assertEqual(res.count_ops()["z"], 2)

    def test_determinism(self):
        """Test that the pass produces structurally equivalent circuits."""
        # This is two CZ rings in a row.  If the cancellation order is non-deterministic and each
        # order has an equal chance, the probability of a spurious pass is astronoomical; the edge
        # IDs linking the in- and out-nodes will be different.
        qc = QuantumCircuit(21)
        for _ in range(2):
            for a, b in zip(qc.qubits[:-1], qc.qubits[1:]):
                qc.cz(a, b)
            qc.cz(qc.qubits[-1], qc.qubits[0])

        expected = circuit_to_dag(qc.copy_empty_like())

        left = CommutativeCancellation().run(circuit_to_dag(qc))
        right = CommutativeCancellation().run(circuit_to_dag(qc))

        # Semantic sanity checks.
        self.assertEqual(expected, left)
        self.assertEqual(expected, right)

        # The actual asseertion.
        self.assertTrue(left.structurally_equal(right))


if __name__ == "__main__":
    unittest.main()
