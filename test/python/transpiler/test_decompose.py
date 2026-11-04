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

"""Test the decompose pass"""

from numpy import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler.passes import Decompose
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import HGate, CCXGate
from qiskit.quantum_info.operators import Operator, Clifford

from test import QiskitTestCase


class TestDecompose(QiskitTestCase):
    """Tests the decompose pass."""

    def setUp(self):
        super().setUp()
        # example complex circuit
        #       ┌────────┐               ┌───┐┌─────────────┐
        # q2_0: ┤0       ├────────────■──┤ H ├┤0            ├
        #       │        │            │  └───┘│  circuit-57 │
        # q2_1: ┤1 gate1 ├────────────■───────┤1            ├
        #       │        │┌────────┐  │       └─────────────┘
        # q2_2: ┤2       ├┤0       ├──■──────────────────────
        #       └────────┘│        │  │
        # q2_3: ──────────┤1 gate2 ├──■──────────────────────
        #                 │        │┌─┴─┐
        # q2_4: ──────────┤2       ├┤ X ├────────────────────
        #                 └────────┘└───┘
        circ1 = QuantumCircuit(3)
        circ1.h(0)
        circ1.t(1)
        circ1.x(2)
        my_gate = circ1.to_gate(label="gate1")
        circ2 = QuantumCircuit(3)
        circ2.h(0)
        circ2.cx(0, 1)
        circ2.x(2)
        my_gate2 = circ2.to_gate(label="gate2")
        circ3 = QuantumCircuit(2)
        circ3.x(0)
        q_bits = QuantumRegister(5)

        qc = QuantumCircuit(q_bits)
        qc.append(my_gate, q_bits[:3])
        qc.append(my_gate2, q_bits[2:])
        qc.mcx(q_bits[:4], q_bits[4])
        qc.h(0)
        qc.append(circ3, [0, 1])
        self.complex_circuit = qc

        # same circuit but with barriers around the MCX-gate
        # (to make sure that the nodes before/after the MCX-gate remain
        # before/after the gates of the expanded MCX-gate).
        qc = QuantumCircuit(q_bits)
        qc.append(my_gate, q_bits[:3])
        qc.append(my_gate2, q_bits[2:])
        qc.barrier(label="barrier1")
        qc.mcx(q_bits[:4], q_bits[4])
        qc.barrier(label="barrier2")
        qc.h(0)
        qc.append(circ3, [0, 1])
        self.complex_circuit_with_barriers = qc

    def test_basic(self):
        """Test decompose a single H into u2."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = Decompose(HGate)
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, "u")

    def test_decompose_none(self):
        """Test decompose a single H into u2."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = Decompose()
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, "u")

    def test_decompose_only_h(self):
        """Test to decompose a single H, without the rest"""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        dag = circuit_to_dag(circuit)
        pass_ = Decompose(HGate)
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes()
        self.assertEqual(len(op_nodes), 2)
        for node in op_nodes:
            self.assertIn(node.name, ["cx", "u"])

    def test_decompose_toffoli(self):
        """Test decompose CCX."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(1, "qr2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = Decompose(CCXGate)
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes()
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            self.assertIn(node.name, ["h", "t", "tdg", "cx"])

    def test_decompose_oversized_instruction(self):
        """Test decompose on a single-op gate that doesn't use all qubits."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/3440
        qc1 = QuantumCircuit(2)
        qc1.x(0)
        gate = qc1.to_gate()

        qc2 = QuantumCircuit(2)
        qc2.append(gate, [0, 1])

        output = qc2.decompose()

        self.assertEqual(qc1, output)

    def test_decomposition_preserves_qregs_order(self):
        """Test decomposing a gate preserves the order of registers in its definition"""
        qr = QuantumRegister(2, "qr1")
        qc1 = QuantumCircuit(qr)
        qc1.cx(1, 0)
        gate = qc1.to_gate()

        qr2 = QuantumRegister(2, "qr2")
        qc2 = QuantumCircuit(qr2)
        qc2.append(gate, qr2)

        expected = QuantumCircuit(qr2)
        expected.cx(1, 0)

        self.assertEqual(qc2.decompose(), expected)

    def test_decompose_global_phase_1q(self):
        """Test decomposition of circuit with global phase"""
        qc1 = QuantumCircuit(1)
        qc1.rz(0.1, 0)
        qc1.ry(0.5, 0)
        qc1.global_phase += pi / 4
        qcd = qc1.decompose()
        self.assertEqual(Operator(qc1), Operator(qcd))

    def test_decompose_global_phase_2q(self):
        """Test decomposition of circuit with global phase"""
        qc1 = QuantumCircuit(2, global_phase=pi / 4)
        qc1.rz(0.1, 0)
        qc1.rxx(0.2, 0, 1)
        qcd = qc1.decompose()
        self.assertEqual(Operator(qc1), Operator(qcd))

    def test_decompose_global_phase_1q_composite(self):
        """Test decomposition of circuit with global phase in a composite gate."""
        circ = QuantumCircuit(1, global_phase=pi / 2)
        circ.x(0)
        circ.h(0)
        v = circ.to_gate()

        qc1 = QuantumCircuit(1)
        qc1.append(v, [0])
        qcd = qc1.decompose()
        self.assertEqual(Operator(qc1), Operator(qcd))

    def test_decompose_only_h_gate(self):
        """Test decomposition parameters so that only a certain gate is decomposed."""
        circ = QuantumCircuit(2, 1)
        circ.h(0)
        circ.cz(0, 1)
        decom_circ = circ.decompose(["h"])
        dag = circuit_to_dag(decom_circ)
        self.assertEqual(len(dag.op_nodes()), 2)
        self.assertEqual(dag.op_nodes()[0].name, "u")
        self.assertEqual(dag.op_nodes()[1].name, "cz")

    def test_decompose_only_given_label(self):
        """Test decomposition parameters so that only a given label is decomposed."""
        decom_circ = self.complex_circuit.decompose(["gate2"])
        dag = circuit_to_dag(decom_circ)

        self.assertEqual(len(dag.op_nodes()), 7)
        self.assertEqual(dag.op_nodes()[0].op.label, "gate1")
        self.assertEqual(dag.op_nodes()[1].name, "h")
        self.assertEqual(dag.op_nodes()[2].name, "cx")
        self.assertEqual(dag.op_nodes()[3].name, "x")
        self.assertEqual(dag.op_nodes()[4].name, "mcx")
        self.assertEqual(dag.op_nodes()[5].name, "h")
        self.assertRegex(dag.op_nodes()[6].name, "circuit-")

    def test_decompose_only_given_name(self):
        """Test decomposition parameters so that only given name is decomposed."""
        decom_circ = self.complex_circuit_with_barriers.decompose(["mcx"], reps=2)
        dag = circuit_to_dag(decom_circ)
        self.assertEqual(len(dag.op_nodes()), 75)
        self.assertEqual(dag.op_nodes()[0].op.label, "gate1")
        self.assertEqual(dag.op_nodes()[1].op.label, "gate2")
        self.assertEqual(dag.op_nodes()[2].op.label, "barrier1")
        self.assertEqual(dag.op_nodes()[72].op.label, "barrier2")
        self.assertEqual(dag.op_nodes()[73].name, "h")
        self.assertRegex(dag.op_nodes()[74].name, "circuit-")

    def test_decompose_mixture_of_names_and_labels(self):
        """Test decomposition parameters so that mixture of names and labels is decomposed"""
        decom_circ = self.complex_circuit_with_barriers.decompose(["mcx", "gate2"], reps=2)
        dag = circuit_to_dag(decom_circ)

        self.assertEqual(len(dag.op_nodes()), 77)
        self.assertEqual(dag.op_nodes()[0].op.label, "gate1")
        self.assertEqual(dag.op_nodes()[1].name, "h")
        self.assertEqual(dag.op_nodes()[2].name, "cx")
        self.assertEqual(dag.op_nodes()[3].name, "x")
        self.assertEqual(dag.op_nodes()[4].name, "barrier")
        self.assertEqual(dag.op_nodes()[74].name, "barrier")
        self.assertEqual(dag.op_nodes()[75].name, "h")
        self.assertRegex(dag.op_nodes()[76].name, "circuit-")

    def test_decompose_name_wildcards(self):
        """Test decomposition parameters so that name wildcards is decomposed"""
        decom_circ = self.complex_circuit.decompose(["circuit-*"])
        dag = circuit_to_dag(decom_circ)

        self.assertEqual(len(dag.op_nodes()), 9)
        self.assertEqual(dag.op_nodes()[0].name, "h")
        self.assertEqual(dag.op_nodes()[1].name, "t")
        self.assertEqual(dag.op_nodes()[2].name, "x")
        self.assertEqual(dag.op_nodes()[3].name, "h")
        self.assertRegex(dag.op_nodes()[4].name, "cx")
        self.assertEqual(dag.op_nodes()[5].name, "x")
        self.assertEqual(dag.op_nodes()[6].name, "mcx")
        self.assertEqual(dag.op_nodes()[7].name, "h")
        self.assertEqual(dag.op_nodes()[8].name, "x")

    def test_decompose_label_wildcards(self):
        """Test decomposition parameters so that label wildcards is decomposed"""
        decom_circ = self.complex_circuit.decompose(["gate*"])
        dag = circuit_to_dag(decom_circ)

        self.assertEqual(len(dag.op_nodes()), 9)
        self.assertEqual(dag.op_nodes()[0].name, "h")
        self.assertEqual(dag.op_nodes()[1].name, "t")
        self.assertEqual(dag.op_nodes()[2].name, "x")
        self.assertEqual(dag.op_nodes()[3].name, "h")
        self.assertEqual(dag.op_nodes()[4].name, "cx")
        self.assertEqual(dag.op_nodes()[5].name, "x")
        self.assertEqual(dag.op_nodes()[6].name, "mcx")
        self.assertEqual(dag.op_nodes()[7].name, "h")
        self.assertRegex(dag.op_nodes()[8].name, "circuit-")

    def test_decompose_empty_gate(self):
        """Test a gate where the definition is an empty circuit is decomposed."""
        empty = QuantumCircuit(1)
        circuit = QuantumCircuit(1)
        circuit.append(empty.to_gate(), [0])

        decomposed = circuit.decompose()
        self.assertEqual(len(decomposed.data), 0)

    def test_decompose_reps(self):
        """Test decompose reps function is decomposed correctly"""
        decom_circ = self.complex_circuit.decompose(reps=2)
        decomposed = self.complex_circuit.decompose().decompose()
        self.assertEqual(decom_circ, decomposed)

    def test_decompose_single_qubit_clbit(self):
        """Test the decomposition of a block with a single qubit and clbit works.

        Regression test of Qiskit/qiskit-terra#8591.
        """
        block = QuantumCircuit(1, 1)
        block.h(0)

        circuit = QuantumCircuit(1, 1)
        circuit.append(block, [0], [0])

        decomposed = circuit.decompose()

        self.assertEqual(decomposed, block)

    def test_decompose_synthesis(self):
        """Test a high-level object with only a synthesis and no definition is correctly decomposed."""
        qc = QuantumCircuit(1)
        qc.h(0)
        cliff = Clifford(qc)

        bigger = QuantumCircuit(1)
        bigger.append(cliff, [0])

        decomposed = bigger.decompose()

        self.assertEqual(qc, decomposed)

    def test_specify_hls_object(self):
        """Test specifying an HLS object by name works."""
        qc = QuantumCircuit(1)
        qc.h(0)
        cliff = Clifford(qc)

        bigger = QuantumCircuit(1)
        bigger.append(cliff, [0])
        bigger.h(0)  # add another gate that should remain unaffected, but has a definition

        decomposed = bigger.decompose(gates_to_decompose=["clifford"])

        expected = QuantumCircuit(1)
        expected.h(0)
        expected.h(0)

        self.assertEqual(expected, decomposed)

    def test_control_flow_if(self):
        """Test decompose with control flow."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        qc.p(0.2, 0)
        qc.measure(0, 0)

        with qc.if_test((cr[0], 0)) as else_:
            qc.cry(0.5, 0, 1)
        with else_:
            qc.crz(0.5, 0, 1)

        expect = qc.copy_empty_like()
        expect.u(0, 0, 0.2, 0)
        expect.measure(0, 0)

        with expect.if_test((cr[0], 0)) as else_:
            expect.ry(0.25, 1)
            expect.cx(0, 1)
            expect.ry(-0.25, 1)
            expect.cx(0, 1)
        with else_:
            expect.rz(0.25, 1)
            expect.cx(0, 1)
            expect.rz(-0.25, 1)
            expect.cx(0, 1)

        self.assertEqual(expect, qc.decompose())

    def test_control_flow_for(self):
        """Test decompose with control flow."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        qc.p(0.2, 0)
        qc.measure(0, 0)

        with qc.for_loop(range(3)):
            qc.cry(0.5, 0, 1)

        expect = qc.copy_empty_like()
        expect.u(0, 0, 0.2, 0)
        expect.measure(0, 0)

        with expect.for_loop(range(3)):
            expect.ry(0.25, 1)
            expect.cx(0, 1)
            expect.ry(-0.25, 1)
            expect.cx(0, 1)

        self.assertEqual(expect, qc.decompose())
