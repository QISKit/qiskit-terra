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

"""Test the Unroll3qOrMore pass"""

import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit, Clbit
from qiskit.circuit.library import CCXGate, RCCXGate, UnitaryGate
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import Target
from test import QiskitTestCase


class TestUnroll3qOrMore(QiskitTestCase):
    """Tests the Unroll3qOrMore pass, for unrolling all
    gates until reaching only 1q or 2q gates."""

    def test_ccx(self):
        """Test decompose CCX."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(1, "qr2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = Unroll3qOrMore()
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes()
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            self.assertIn(node.name, ["h", "t", "tdg", "cx"])

    def test_cswap(self):
        """Test decompose CSwap (recursively)."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(1, "qr2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.cswap(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = Unroll3qOrMore()
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes()
        self.assertEqual(len(op_nodes), 17)
        for node in op_nodes:
            self.assertIn(node.name, ["h", "t", "tdg", "cx"])

    def test_decompose_conditional(self):
        """Test decompose a 3-qubit gate with a conditional."""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with circuit.if_test((cr, 0)):
            circuit.ccx(qr[0], qr[1], qr[2])
        dag = circuit_to_dag(circuit)
        pass_ = Unroll3qOrMore()
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(len(op_nodes[0].op.blocks[0].data), 15)
        for node in op_nodes[0].op.blocks[0].data:
            self.assertIn(node.name, ["h", "t", "tdg", "cx", "if_else"])

    def test_decompose_unitary(self):
        """Test unrolling of unitary gate over 4qubits."""
        qr = QuantumRegister(4, "qr")
        circuit = QuantumCircuit(qr)
        unitary = random_unitary(16, seed=42)
        circuit.unitary(unitary, [0, 1, 2, 3])
        dag = circuit_to_dag(circuit)
        pass_ = Unroll3qOrMore()
        after_dag = pass_.run(dag)
        after_circ = dag_to_circuit(after_dag)
        self.assertTrue(Operator(circuit).equiv(Operator(after_circ)))

    def test_identity(self):
        """Test unrolling of identity gate over 3qubits."""
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        gate = UnitaryGate(np.eye(2**3))
        circuit.append(gate, range(3))
        dag = circuit_to_dag(circuit)
        pass_ = Unroll3qOrMore()
        after_dag = pass_.run(dag)
        after_circ = dag_to_circuit(after_dag)
        self.assertTrue(Operator(circuit).equiv(Operator(after_circ)))

    def test_target(self):
        """Test target is respected by the unroll 3q or more pass."""
        target = Target(num_qubits=3)
        target.add_instruction(CCXGate())
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.append(RCCXGate(), [0, 1, 2])
        unroll_pass = Unroll3qOrMore(target=target)
        res = unroll_pass(qc)
        self.assertIn("ccx", res.count_ops())
        self.assertNotIn("rccx", res.count_ops())

    def test_basis_gates(self):
        """Test basis_gates are respected by the unroll 3q or more pass."""
        basis_gates = ["rccx"]
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.append(RCCXGate(), [0, 1, 2])
        unroll_pass = Unroll3qOrMore(basis_gates=basis_gates)
        res = unroll_pass(qc)
        self.assertNotIn("ccx", res.count_ops())
        self.assertIn("rccx", res.count_ops())

    def test_target_over_basis_gates(self):
        """Test target is respected over basis_gates  by the unroll 3q or more pass."""
        target = Target(num_qubits=3)
        basis_gates = ["rccx"]
        target.add_instruction(CCXGate())
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.append(RCCXGate(), [0, 1, 2])
        unroll_pass = Unroll3qOrMore(target=target, basis_gates=basis_gates)
        res = unroll_pass(qc)
        self.assertIn("ccx", res.count_ops())
        self.assertNotIn("rccx", res.count_ops())

    def test_if_else(self):
        """Test that a simple if-else over 3+ qubits unrolls correctly."""
        pass_ = Unroll3qOrMore(basis_gates=["u", "cx"])

        true_body = QuantumCircuit(3, 1)
        true_body.h(0)
        true_body.ccx(0, 1, 2)
        false_body = QuantumCircuit(3, 1)
        false_body.rccx(2, 1, 0)

        test = QuantumCircuit(3, 1)
        test.h(0)
        test.measure(0, 0)
        test.if_else((0, True), true_body, false_body, [0, 1, 2], [0])

        expected = QuantumCircuit(3, 1)
        expected.h(0)
        expected.measure(0, 0)
        expected.if_else((0, True), pass_(true_body), pass_(false_body), [0, 1, 2], [0])

        self.assertEqual(pass_(test), expected)

    def test_nested_control_flow(self):
        """Test that the unroller recurses into nested control flow."""
        pass_ = Unroll3qOrMore(basis_gates=["u", "cx"])
        qubits = [Qubit() for _ in [None] * 3]
        clbit = Clbit()

        for_body = QuantumCircuit(qubits, [clbit])
        for_body.ccx(0, 1, 2)

        while_body = QuantumCircuit(qubits, [clbit])
        while_body.rccx(0, 1, 2)

        true_body = QuantumCircuit(qubits, [clbit])
        true_body.while_loop((clbit, True), while_body, [0, 1, 2], [0])

        test = QuantumCircuit(qubits, [clbit])
        test.for_loop(range(2), None, for_body, [0, 1, 2], [0])
        test.if_else((clbit, True), true_body, None, [0, 1, 2], [0])

        expected_if_body = QuantumCircuit(qubits, [clbit])
        expected_if_body.while_loop((clbit, True), pass_(while_body), [0, 1, 2], [0])
        expected = QuantumCircuit(qubits, [clbit])
        expected.for_loop(range(2), None, pass_(for_body), [0, 1, 2], [0])
        expected.if_else(range(2), pass_(expected_if_body), None, [0, 1, 2], [0])

        self.assertEqual(pass_(test), expected)

    def test_if_else_in_basis(self):
        """Test that a simple if-else over 3+ qubits unrolls correctly."""
        pass_ = Unroll3qOrMore(basis_gates=["u", "cx", "if_else", "for_loop", "while_loop"])

        true_body = QuantumCircuit(3, 1)
        true_body.h(0)
        true_body.ccx(0, 1, 2)
        false_body = QuantumCircuit(3, 1)
        false_body.rccx(2, 1, 0)

        test = QuantumCircuit(3, 1)
        test.h(0)
        test.measure(0, 0)
        test.if_else((0, True), true_body, false_body, [0, 1, 2], [0])

        expected = QuantumCircuit(3, 1)
        expected.h(0)
        expected.measure(0, 0)
        expected.if_else((0, True), pass_(true_body), pass_(false_body), [0, 1, 2], [0])

        self.assertEqual(pass_(test), expected)

    def test_nested_control_flow_in_basis(self):
        """Test that the unroller recurses into nested control flow."""
        pass_ = Unroll3qOrMore(basis_gates=["u", "cx", "if_else", "for_loop", "while_loop"])
        qubits = [Qubit() for _ in [None] * 3]
        clbit = Clbit()

        for_body = QuantumCircuit(qubits, [clbit])
        for_body.ccx(0, 1, 2)

        while_body = QuantumCircuit(qubits, [clbit])
        while_body.rccx(0, 1, 2)

        true_body = QuantumCircuit(qubits, [clbit])
        true_body.while_loop((clbit, True), while_body, [0, 1, 2], [0])

        test = QuantumCircuit(qubits, [clbit])
        test.for_loop(range(2), None, for_body, [0, 1, 2], [0])
        test.if_else((clbit, True), true_body, None, [0, 1, 2], [0])

        expected_if_body = QuantumCircuit(qubits, [clbit])
        expected_if_body.while_loop((clbit, True), pass_(while_body), [0, 1, 2], [0])
        expected = QuantumCircuit(qubits, [clbit])
        expected.for_loop(range(2), None, pass_(for_body), [0, 1, 2], [0])
        expected.if_else(range(2), pass_(expected_if_body), None, [0, 1, 2], [0])

        self.assertEqual(pass_(test), expected)

    def test_custom_block_over_3q(self):
        """Test a custom instruction is unrolled in a control flow block."""
        pass_ = Unroll3qOrMore(basis_gates=["u", "cx", "if_else", "for_loop", "while_loop"])
        ghz = QuantumCircuit(5, 5)
        ghz.h(0)
        ghz.cx(0, 1)
        ghz.cx(0, 2)
        ghz.cx(0, 3)
        ghz.cx(0, 4)
        ghz.measure(0, 0)
        ghz.measure(1, 1)
        ghz.measure(2, 2)
        ghz.measure(3, 3)
        ghz.measure(4, 4)
        ghz.reset(0)
        ghz.reset(1)
        ghz.reset(2)
        ghz.reset(3)
        ghz.reset(4)
        for_block = QuantumCircuit(5, 5, name="ghz")
        for_block.append(ghz, list(range(5)), list(range(5)))
        qc = QuantumCircuit(5, 5)
        qc.for_loop((1,), None, for_block, [2, 4, 1, 3, 0], [0, 1, 2, 3, 4])
        result = pass_(qc)
        expected = QuantumCircuit(5, 5)
        expected.for_loop((1,), None, ghz, [2, 4, 1, 3, 0], [0, 1, 2, 3, 4])
        self.assertEqual(result, expected)
