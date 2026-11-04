# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Gate Direction pass"""

import unittest
from math import pi

import ddt

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter, Gate
from qiskit.circuit.library import (
    CXGate,
    CZGate,
    ECRGate,
    RXXGate,
    RYYGate,
    RZXGate,
    RZZGate,
    SwapGate,
)
from qiskit.compiler import transpile
from qiskit.transpiler import TranspilerError, CouplingMap, Target
from qiskit.transpiler.passes import GateDirection
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase


@ddt.ddt
class TestGateDirection(QiskitTestCase):
    """Tests the GateDirection pass."""

    def test_no_cnots(self):
        """Trivial map in a circuit without entanglement
        qr0:---[H]---

        qr1:---[H]---

        qr2:---[H]---

        CouplingMap map: None
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        coupling = CouplingMap()
        dag = circuit_to_dag(circuit)

        pass_ = GateDirection(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_direction_error(self):
        """The mapping cannot be fixed by direction mapper
        qr0:---------

        qr1:----.-----
                |
        qr2:---(+)----

        CouplingMap map: [2] <- [0] -> [1]
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        coupling = CouplingMap([[0, 1], [0, 2]])
        dag = circuit_to_dag(circuit)

        pass_ = GateDirection(coupling)

        with self.assertRaises(TranspilerError):
            pass_.run(dag)

    def test_direction_correct(self):
        """The CX is in the right direction
        qr0:----.----
                |
        qr1:---(+)----

        CouplingMap map: [0] -> [1]
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)

        pass_ = GateDirection(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_multi_register(self):
        """The CX is in the right direction
        qr0:----.-----
                |
        qr1:---(+)----

        CouplingMap map: [0] -> [1]
        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(1, "qr2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.cx(qr1, qr2)
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)

        pass_ = GateDirection(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_direction_flip(self):
        """Flip a CX
        qr0:---(+)---
                |
        qr1:----.----

        CouplingMap map: [0] -> [1]

        qr0:-[H]--.--[H]--
                  |
        qr1:-[H]-(+)--[H]--
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr[0])
        expected.h(qr[1])
        expected.cx(qr[0], qr[1])
        expected.h(qr[0])
        expected.h(qr[1])

        pass_ = GateDirection(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_ecr_flip(self):
        """Flip a ECR gate.
                ┌──────┐
           q_0: ┤1     ├
                │  ECR │
           q_1: ┤0     ├
                └──────┘

        CouplingMap map: [0, 1]
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.ecr(qr[1], qr[0])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)

        #       ┌─────────┐ ┌──────┐┌───┐
        # qr_0: ┤ Ry(π/2) ├─┤0     ├┤ H ├
        #       ├─────────┴┐│  Ecr │├───┤
        # qr_1: ┤ Ry(-π/2) ├┤1     ├┤ H ├
        #       └──────────┘└──────┘└───┘
        expected = QuantumCircuit(qr, global_phase=-pi / 2)
        expected.s(1)
        expected.sx(1)
        expected.sdg(1)
        expected.sdg(0)
        expected.sx(0)
        expected.s(0)
        expected.ecr(0, 1)
        expected.h(0)
        expected.h(1)

        pass_ = GateDirection(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_flip_with_measure(self):
        """
        qr0: -(+)-[m]-
               |   |
        qr1: --.---|--
                   |
        cr0: ------.--

        CouplingMap map: [0] -> [1]

        qr0: -[H]--.--[H]-[m]-
                   |       |
        qr1: -[H]-(+)-[H]--|--
                           |
        cr0: --------------.--
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")

        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[1], qr[0])
        circuit.measure(qr[0], cr[0])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.h(qr[0])
        expected.h(qr[1])
        expected.cx(qr[0], qr[1])
        expected.h(qr[0])
        expected.h(qr[1])
        expected.measure(qr[0], cr[0])

        pass_ = GateDirection(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_regression_gh_8387(self):
        """Regression test for flipping of CZ gate"""
        qc = QuantumCircuit(3)
        qc.cz(1, 0)
        qc.barrier()
        qc.cz(2, 0)

        coupling_map = CouplingMap([[0, 1], [1, 2]])
        _ = transpile(
            qc,
            basis_gates=["cz", "cx", "u3", "u2", "u1"],
            coupling_map=coupling_map,
            optimization_level=2,
        )

    @ddt.data(CXGate(), CZGate(), ECRGate())
    def test_target_static(self, gate):
        """Test that static 2q gates are swapped correctly both if available and not available."""
        circuit = QuantumCircuit(2)
        circuit.append(gate, [0, 1], [])

        matching = Target(num_qubits=2)
        matching.add_instruction(gate, {(0, 1): None})
        self.assertEqual(GateDirection(None, target=matching)(circuit), circuit)

        swapped = Target(num_qubits=2)
        swapped.add_instruction(gate, {(1, 0): None})
        self.assertNotEqual(GateDirection(None, target=swapped)(circuit), circuit)

    @ddt.data(CZGate(), RZXGate(pi / 3), RXXGate(pi / 3), RYYGate(pi / 3), RZZGate(pi / 3))
    def test_target_trivial(self, gate):
        """Test that trivial 2q gates are swapped correctly both if available and not available."""
        circuit = QuantumCircuit(2)
        circuit.append(gate, [0, 1], [])

        matching = Target(num_qubits=2)
        matching.add_instruction(gate, {(0, 1): None})
        self.assertEqual(GateDirection(None, target=matching)(circuit), circuit)

        swapped = Target(num_qubits=2)
        swapped.add_instruction(gate, {(1, 0): None})

        self.assertNotEqual(GateDirection(None, target=swapped)(circuit), circuit)

    @ddt.data(CZGate(), SwapGate(), RXXGate(pi / 3), RYYGate(pi / 3), RZZGate(pi / 3))
    def test_symmetric_gates(self, gate):
        """Test symmetric gates on single direction coupling map."""
        circuit = QuantumCircuit(2)
        circuit.append(gate, [1, 0], [])

        expected = QuantumCircuit(2)
        expected.append(gate, [0, 1], [])

        coupling = CouplingMap.from_line(2, bidirectional=False)
        pass_ = GateDirection(coupling)
        self.assertEqual(pass_(circuit), expected)

    def test_target_parameter_any(self):
        """Test that a parametrized 2q gate is replaced correctly both if available and not
        available."""
        circuit = QuantumCircuit(2)
        circuit.rzx(1.5, 0, 1)

        matching = Target(num_qubits=2)
        matching.add_instruction(RZXGate(Parameter("a")), {(0, 1): None})
        self.assertEqual(GateDirection(None, target=matching)(circuit), circuit)

        swapped = Target(num_qubits=2)
        swapped.add_instruction(RZXGate(Parameter("a")), {(1, 0): None})
        self.assertNotEqual(GateDirection(None, target=swapped)(circuit), circuit)

    def test_target_parameter_exact(self):
        """Test that a parametrized 2q gate is detected correctly both if available and not
        available."""
        circuit = QuantumCircuit(2)
        circuit.rzx(1.5, 0, 1)

        matching = Target(num_qubits=2)
        matching.add_instruction(RZXGate(1.5), {(0, 1): None})
        self.assertEqual(GateDirection(None, target=matching)(circuit), circuit)

        swapped = Target(num_qubits=2)
        swapped.add_instruction(RZXGate(1.5), {(1, 0): None})
        self.assertNotEqual(GateDirection(None, target=swapped)(circuit), circuit)

    def test_target_parameter_mismatch(self):
        """Test that the pass raises if a gate is not supported due to a parameter mismatch."""
        circuit = QuantumCircuit(2)
        circuit.rzx(1.5, 0, 1)

        matching = Target(num_qubits=2)
        matching.add_instruction(RZXGate(2.5), {(0, 1): None})
        pass_ = GateDirection(None, target=matching)
        with self.assertRaises(TranspilerError):
            pass_(circuit)

        swapped = Target(num_qubits=2)
        swapped.add_instruction(RZXGate(2.5), {(1, 0): None})
        pass_ = GateDirection(None, target=swapped)
        with self.assertRaises(TranspilerError):
            pass_(circuit)

    def test_coupling_map_control_flow(self):
        """Test that gates are replaced within nested control-flow blocks."""
        circuit = QuantumCircuit(4, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop((1, 2)):
            circuit.cx(1, 0)
            circuit.cx(0, 1)
            with circuit.if_test((circuit.clbits[0], True)) as else_:
                circuit.ecr(3, 2)
            with else_:
                with circuit.while_loop((circuit.clbits[0], True)):
                    circuit.rzx(2.3, 2, 1)

        expected = QuantumCircuit(4, 1)
        expected.h(0)
        expected.measure(0, 0)
        with expected.for_loop((1, 2)):
            expected.h([0, 1])
            expected.cx(0, 1)
            expected.h([0, 1])
            expected.cx(0, 1)
            with expected.if_test((circuit.clbits[0], True)) as else_:
                expected.global_phase -= pi / 2
                expected.sdg(2)
                expected.sx(2)
                expected.s(2)
                expected.s(3)
                expected.sx(3)
                expected.sdg(3)
                expected.ecr(2, 3)
                expected.h([2, 3])
            with else_:
                with expected.while_loop((circuit.clbits[0], True)):
                    expected.h([1, 2])
                    expected.rzx(2.3, 1, 2)
                    expected.h([1, 2])

        coupling = CouplingMap.from_line(4, bidirectional=False)
        pass_ = GateDirection(coupling)
        self.assertEqual(pass_(circuit), expected)

    def test_target_control_flow(self):
        """Test that gates are replaced within nested control-flow blocks."""
        circuit = QuantumCircuit(4, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop((1, 2)):
            circuit.cx(1, 0)
            circuit.cx(0, 1)
            with circuit.if_test((circuit.clbits[0], True)) as else_:
                circuit.ecr(3, 2)
            with else_:
                with circuit.while_loop((circuit.clbits[0], True)):
                    circuit.rzx(2.3, 2, 1)

        expected = QuantumCircuit(4, 1)
        expected.h(0)
        expected.measure(0, 0)
        with expected.for_loop((1, 2)):
            expected.h([0, 1])
            expected.cx(0, 1)
            expected.h([0, 1])
            expected.cx(0, 1)
            with expected.if_test((circuit.clbits[0], True)) as else_:
                expected.global_phase -= pi / 2
                expected.sdg(2)
                expected.sx(2)
                expected.s(2)
                expected.s(3)
                expected.sx(3)
                expected.sdg(3)
                expected.ecr(2, 3)
                expected.h([2, 3])
            with else_:
                with expected.while_loop((circuit.clbits[0], True)):
                    expected.h([1, 2])
                    expected.rzx(2.3, 1, 2)
                    expected.h([1, 2])

        target = Target(num_qubits=4)
        target.add_instruction(CXGate(), {(0, 1): None})
        target.add_instruction(ECRGate(), {(2, 3): None})
        target.add_instruction(RZXGate(Parameter("a")), {(1, 2): None})
        pass_ = GateDirection(None, target)
        self.assertEqual(pass_(circuit), expected)

    def test_target_cannot_flip_message(self):
        """A suitable error message should be emitted if the gate would be supported if it were
        flipped."""
        gate = Gate("my_2q_gate", 2, [])
        target = Target(num_qubits=2)
        target.add_instruction(gate, properties={(0, 1): None})

        circuit = QuantumCircuit(2)
        circuit.append(gate, (1, 0))

        pass_ = GateDirection(None, target)
        with self.assertRaisesRegex(TranspilerError, "my_2q_gate would be supported.*"):
            pass_(circuit)

    def test_target_unknown_gate_message(self):
        """A suitable error message should be emitted if the gate isn't valid in either direction on
        the target."""
        gate = Gate("my_2q_gate", 2, [])
        target = Target(num_qubits=2)
        target.add_instruction(CXGate(), properties={(0, 1): None})

        circuit = QuantumCircuit(2)
        circuit.append(gate, (0, 1))

        pass_ = GateDirection(None, target)
        with self.assertRaisesRegex(TranspilerError, "my_2q_gate.*not supported on qubits .*"):
            pass_(circuit)


if __name__ == "__main__":
    unittest.main()
