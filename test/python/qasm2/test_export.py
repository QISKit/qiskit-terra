# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
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
import pathlib
import re
import tempfile
import unittest
from math import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, qasm2
from qiskit.circuit import Parameter, Qubit, Clbit, Gate, library as lib
from qiskit.circuit.classical import expr
from test import QiskitTestCase

# Regex pattern to match valid OpenQASM identifiers
VALID_QASM2_IDENTIFIER = re.compile("[a-z][a-zA-Z_0-9]*")


class TestQASM2Export(QiskitTestCase):
    # We test most of the export by looking at `dumps` because it's easier just to compare strings.

    def test_basic_output(self):
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        cr = ClassicalRegister(3, "cr")
        qc = QuantumCircuit(qr1, qr2, cr)
        qc.p(0.3, qr1[0])
        qc.u(0.3, 0.2, 0.1, qr2[1])
        qc.s(qr2[1])
        qc.sdg(qr2[1])
        qc.cx(qr1[0], qr2[1])
        qc.barrier(qr2)
        qc.cx(qr2[1], qr1[0])
        qc.h(qr2[1])
        with qc.if_test((cr, 0)):
            qc.x(qr2[1])
        with qc.if_test((cr, 1)):
            qc.y(qr1[0])
        with qc.if_test((cr, 2)):
            qc.z(qr1[0])
        qc.barrier(qr1, qr2)
        qc.measure(qr1[0], cr[0])
        qc.measure(qr2[0], cr[1])
        qc.measure(qr2[1], cr[2])
        expected_qasm = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg qr1[1];
qreg qr2[2];
creg cr[3];
p(0.3) qr1[0];
u(0.3,0.2,0.1) qr2[1];
s qr2[1];
sdg qr2[1];
cx qr1[0],qr2[1];
barrier qr2[0],qr2[1];
cx qr2[1],qr1[0];
h qr2[1];
if (cr == 0) x qr2[1];
if (cr == 1) y qr1[0];
if (cr == 2) z qr1[0];
barrier qr1[0],qr2[0],qr2[1];
measure qr1[0] -> cr[0];
measure qr2[0] -> cr[1];
measure qr2[1] -> cr[2];"""
        self.assertEqual(qasm2.dumps(qc), expected_qasm)

    def test_composite_circuit(self):
        composite_circ_qreg = QuantumRegister(2)
        composite_circ = QuantumCircuit(composite_circ_qreg, name="composite_circ")
        composite_circ.h(0)
        composite_circ.x(1)
        composite_circ.cx(0, 1)

        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ.to_gate(), [0, 1])
        qc.measure([0, 1], [0, 1])

        expected_qasm = """\
OPENQASM 2.0;
include "qelib1.inc";
gate composite_circ q0,q1 { h q0; x q1; cx q0,q1; }
qreg qr[2];
creg cr[2];
h qr[0];
cx qr[0],qr[1];
barrier qr[0],qr[1];
composite_circ qr[0],qr[1];
measure qr[0] -> cr[0];
measure qr[1] -> cr[1];"""
        self.assertEqual(qasm2.dumps(qc), expected_qasm)

    def test_multiple_same_composite_circuits(self):
        composite_circ_qreg = QuantumRegister(2)
        composite_circ = QuantumCircuit(composite_circ_qreg, name="composite_circ")
        composite_circ.h(0)
        composite_circ.x(1)
        composite_circ.cx(0, 1)
        composite_circ_gate = composite_circ.to_gate()

        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ_gate, [0, 1])
        qc.append(composite_circ_gate, [0, 1])
        qc.measure([0, 1], [0, 1])

        expected_qasm = """\
OPENQASM 2.0;
include "qelib1.inc";
gate composite_circ q0,q1 { h q0; x q1; cx q0,q1; }
qreg qr[2];
creg cr[2];
h qr[0];
cx qr[0],qr[1];
barrier qr[0],qr[1];
composite_circ qr[0],qr[1];
composite_circ qr[0],qr[1];
measure qr[0] -> cr[0];
measure qr[1] -> cr[1];"""
        self.assertEqual(qasm2.dumps(qc), expected_qasm)

    def test_multiple_composite_circuits_with_same_name(self):
        my_gate = QuantumCircuit(1, name="my_gate")
        my_gate.h(0)
        my_gate_inst1 = my_gate.to_gate()

        my_gate = QuantumCircuit(1, name="my_gate")
        my_gate.x(0)
        my_gate_inst2 = my_gate.to_gate()

        my_gate = QuantumCircuit(1, name="my_gate")
        my_gate.x(0)
        my_gate_inst3 = my_gate.to_gate()

        qr = QuantumRegister(1, name="qr")
        circuit = QuantumCircuit(qr, name="circuit")
        circuit.append(my_gate_inst1, [qr[0]])
        circuit.append(my_gate_inst2, [qr[0]])
        my_gate_inst2_id = id(circuit.data[-1].operation)
        circuit.append(my_gate_inst3, [qr[0]])
        my_gate_inst3_id = id(circuit.data[-1].operation)

        expected_qasm = f"""\
OPENQASM 2.0;
include "qelib1.inc";
gate my_gate q0 {{ h q0; }}
gate my_gate_{my_gate_inst2_id} q0 {{ x q0; }}
gate my_gate_{my_gate_inst3_id} q0 {{ x q0; }}
qreg qr[1];
my_gate qr[0];
my_gate_{my_gate_inst2_id} qr[0];
my_gate_{my_gate_inst3_id} qr[0];"""
        self.assertEqual(qasm2.dumps(circuit), expected_qasm)

    def test_composite_circuit_with_children_composite_circuit(self):
        child_circ = QuantumCircuit(2, name="child_circ")
        child_circ.h(0)
        child_circ.cx(0, 1)

        parent_circ = QuantumCircuit(3, name="parent_circ")
        parent_circ.append(child_circ, range(2))
        parent_circ.h(2)

        grandparent_circ = QuantumCircuit(4, name="grandparent_circ")
        grandparent_circ.append(parent_circ, range(3))
        grandparent_circ.x(3)

        qc = QuantumCircuit(4)
        qc.append(grandparent_circ, range(4))

        expected_qasm = """\
OPENQASM 2.0;
include "qelib1.inc";
gate child_circ q0,q1 { h q0; cx q0,q1; }
gate parent_circ q0,q1,q2 { child_circ q0,q1; h q2; }
gate grandparent_circ q0,q1,q2,q3 { parent_circ q0,q1,q2; x q3; }
qreg q[4];
grandparent_circ q[0],q[1],q[2],q[3];"""

        self.assertEqual(qasm2.dumps(qc), expected_qasm)

    def test_qasm_pi(self):
        """Test circuit qasm() method with pi params."""
        circuit = QuantumCircuit(2)
        circuit.cz(0, 1)
        circuit.u(2 * pi, 3 * pi, -5 * pi, 0)
        self.assertEqual(
            circuit,
            qasm2.loads(qasm2.dumps(circuit), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS),
        )

    def test_composite_circuit_with_one_param(self):
        """Test circuit qasm() method when a composite circuit instruction
        has one param
        """
        original_str = """\
OPENQASM 2.0;
include "qelib1.inc";
gate nG0(param0) q0 { h q0; }
qreg q[3];
creg c[3];
nG0(pi) q[0];"""
        qc = QuantumCircuit.from_qasm_str(original_str)

        self.assertEqual(original_str, qasm2.dumps(qc))

    def test_composite_circuit_with_many_params_and_qubits(self):
        """Test circuit qasm() method when a composite circuit instruction
        has many params and qubits
        """
        original_str = """\
OPENQASM 2.0;
include "qelib1.inc";
gate nG0(param0,param1) q0,q1 { h q0; h q1; }
qreg q[3];
qreg r[3];
creg c[3];
creg d[3];
nG0(pi,pi/2) q[0],r[0];"""
        qc = QuantumCircuit.from_qasm_str(original_str)

        self.assertEqual(original_str, qasm2.dumps(qc))

    def test_c3sxgate_roundtrips(self):
        """Test that C3SXGate correctly round trips.

        Qiskit gives this gate a different name
        ('c3sx') to the name in Qiskit's version of qelib1.inc ('c3sqrtx') gate, which can lead to
        resolution issues."""
        qc = QuantumCircuit(4)
        qc.append(lib.C3SXGate(), qc.qubits, [])
        qasm = qasm2.dumps(qc)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
c3sqrtx q[0],q[1],q[2],q[3];"""
        self.assertEqual(qasm, expected)
        parsed = QuantumCircuit.from_qasm_str(qasm)
        self.assertIsInstance(parsed.data[0].operation, lib.C3SXGate)

    def test_cczgate_qasm(self):
        """Test that CCZ dumps definition as a non-qelib1 gate."""
        qc = QuantumCircuit(3)
        qc.append(lib.CCZGate(), qc.qubits, [])
        qasm = qasm2.dumps(qc)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
gate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }
qreg q[3];
ccz q[0],q[1],q[2];"""
        self.assertEqual(qasm, expected)

    def test_csgate_qasm(self):
        """Test that CS dumps definition as a non-qelib1 gate."""
        qc = QuantumCircuit(2)
        qc.append(lib.CSGate(), qc.qubits, [])
        qasm = qasm2.dumps(qc)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
qreg q[2];
cs q[0],q[1];"""
        self.assertEqual(qasm, expected)

    def test_csdggate_qasm(self):
        """Test that CSdg dumps definition as a non-qelib1 gate."""
        qc = QuantumCircuit(2)
        qc.append(lib.CSdgGate(), qc.qubits, [])
        qasm = qasm2.dumps(qc)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
qreg q[2];
csdg q[0],q[1];"""
        self.assertEqual(qasm, expected)

    def test_rzxgate_qasm(self):
        """Test that RZX dumps definition as a non-qelib1 gate."""
        qc = QuantumCircuit(2)
        qc.rzx(0, 0, 1)
        qc.rzx(pi / 2, 1, 0)
        qasm = qasm2.dumps(qc)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
qreg q[2];
rzx(0) q[0],q[1];
rzx(pi/2) q[1],q[0];"""
        self.assertEqual(qasm, expected)

    def test_ecrgate_qasm(self):
        """Test that ECR dumps its definition as a non-qelib1 gate."""
        qc = QuantumCircuit(2)
        qc.ecr(0, 1)
        qc.ecr(1, 0)
        qasm = qasm2.dumps(qc)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
qreg q[2];
ecr q[0],q[1];
ecr q[1],q[0];"""
        self.assertEqual(qasm, expected)

    def test_unitary_qasm(self):
        """Test that UnitaryGate can be dumped to OQ2 correctly."""
        qc = QuantumCircuit(1)
        qc.unitary([[1, 0], [0, 1]], 0)
        qasm = qasm2.dumps(qc)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
gate unitary q0 { u(0,0,0) q0; }
qreg q[1];
unitary q[0];"""
        self.assertEqual(qasm, expected)

    def test_multiple_unitary_qasm(self):
        """Test that multiple UnitaryGate instances can all dump successfully."""
        custom = QuantumCircuit(1, name="custom")
        custom.unitary([[1, 0], [0, -1]], 0)

        qc = QuantumCircuit(2)
        qc.unitary([[1, 0], [0, 1]], 0)
        qc.unitary([[0, 1], [1, 0]], 1)
        qc.append(custom.to_gate(), [0], [])
        qasm = qasm2.dumps(qc)
        expected = re.compile(
            r"""OPENQASM 2.0;
include "qelib1.inc";
gate unitary q0 { u\(0,0,0\) q0; }
gate (?P<u1>unitary_[0-9]*) q0 { u\(pi,-pi,0\) q0; }
gate (?P<u2>unitary_[0-9]*) q0 { u\(0,0,pi\) q0; }
gate custom q0 { (?P=u2) q0; }
qreg q\[2\];
unitary q\[0\];
(?P=u1) q\[1\];
custom q\[0\];""",
            re.MULTILINE,
        )
        self.assertRegex(qasm, expected)

    def test_unbound_circuit_raises(self):
        """Test circuits with unbound parameters raises."""
        qc = QuantumCircuit(1)
        theta = Parameter("θ")
        qc.rz(theta, 0)
        with self.assertRaises(qasm2.QASM2ExportError):
            qasm2.dumps(qc)

    def test_gate_qasm_with_ctrl_state(self):
        from qiskit.quantum_info import Operator

        qc = QuantumCircuit(2)
        qc.ch(0, 1, ctrl_state=0)
        qasm_str = qasm2.dumps(qc)
        self.assertEqual(Operator(qc), Operator(qasm2.loads(qasm_str)))

    def test_mcx_gate(self):

        qc = QuantumCircuit(4)
        qc.mcx([0, 1, 2], 3)

        # qasm output doesn't support parameterized gate yet.
        # param0 for "gate mcuq(param0) is not used inside the definition
        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
gate mcx q0,q1,q2,q3 { h q3; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q3; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; h q3; }
qreg q[4];
mcx q[0],q[1],q[2],q[3];"""
        self.assertEqual(qasm2.dumps(qc), expected_qasm)

    def test_mcx_gate_variants(self):

        n = 5
        qc = QuantumCircuit(2 * n - 1)
        with self.assertWarns(DeprecationWarning):
            gray = lib.MCXGrayCode(n)
            recursive = lib.MCXRecursive(n)
            vchain = lib.MCXVChain(n)
        qc.append(gray, range(n + 1))
        qc.append(recursive, range(n + 2))
        qc.append(vchain, range(2 * n - 1))

        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
gate mcx_gray q0,q1,q2,q3,q4,q5 { h q5; cu1(pi/16) q4,q5; cx q4,q3; cu1(-pi/16) q3,q5; cx q4,q3; cu1(pi/16) q3,q5; cx q3,q2; cu1(-pi/16) q2,q5; cx q4,q2; cu1(pi/16) q2,q5; cx q3,q2; cu1(-pi/16) q2,q5; cx q4,q2; cu1(pi/16) q2,q5; cx q2,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q3,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q2,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q3,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q1,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q2,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q1,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q2,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; h q5; }
gate mcx_recursive q0,q1,q2,q3,q4,q5,q6 { h q6; t q6; cx q2,q6; tdg q6; cx q3,q6; h q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; cx q3,q6; t q6; cx q2,q6; tdg q6; h q6; h q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; h q5; p(pi/8) q3; p(pi/8) q4; p(pi/8) q6; p(pi/8) q5; cx q3,q4; p(-pi/8) q4; cx q3,q4; cx q4,q6; p(-pi/8) q6; cx q3,q6; p(pi/8) q6; cx q4,q6; p(-pi/8) q6; cx q3,q6; cx q6,q5; p(-pi/8) q5; cx q4,q5; p(pi/8) q5; cx q6,q5; p(-pi/8) q5; cx q3,q5; p(pi/8) q5; cx q6,q5; p(-pi/8) q5; cx q4,q5; p(pi/8) q5; cx q6,q5; p(-pi/8) q5; cx q3,q5; h q5; h q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; h q6; t q6; cx q2,q6; tdg q6; cx q3,q6; h q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; cx q3,q6; t q6; cx q2,q6; tdg q6; h q6; h q5; p(pi/8) q3; p(pi/8) q4; p(pi/8) q6; p(pi/8) q5; cx q3,q4; p(-pi/8) q4; cx q3,q4; cx q4,q6; p(-pi/8) q6; cx q3,q6; p(pi/8) q6; cx q4,q6; p(-pi/8) q6; cx q3,q6; cx q6,q5; p(-pi/8) q5; cx q4,q5; p(pi/8) q5; cx q6,q5; p(-pi/8) q5; cx q3,q5; p(pi/8) q5; cx q6,q5; p(-pi/8) q5; cx q4,q5; p(pi/8) q5; cx q6,q5; p(-pi/8) q5; cx q3,q5; h q5; }
gate mcx_vchain q0,q1,q2,q3,q4,q5,q6,q7,q8 { rccx q0,q1,q6; rccx q2,q6,q7; rccx q3,q7,q8; ccx q4,q8,q5; rccx q3,q7,q8; rccx q2,q6,q7; rccx q0,q1,q6; }
qreg q[9];
mcx_gray q[0],q[1],q[2],q[3],q[4],q[5];
mcx_recursive q[0],q[1],q[2],q[3],q[4],q[5],q[6];
mcx_vchain q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8];"""

        self.assertEqual(qasm2.dumps(qc), expected_qasm)

    def test_registerless_bits(self):
        """Test that registerless bits do not have naming collisions in their registers."""
        initial_registers = [QuantumRegister(2), ClassicalRegister(2)]
        qc = QuantumCircuit(*initial_registers, [Qubit(), Clbit()])
        # Match a 'qreg identifier[3];'-like QASM register declaration.
        register_regex = re.compile(r"\s*[cq]reg\s+(\w+)\s*\[\d+\]\s*", re.M)
        qasm_register_names = set()
        for statement in qasm2.dumps(qc).split(";"):
            match = register_regex.match(statement)
            if match:
                qasm_register_names.add(match.group(1))
        self.assertEqual(len(qasm_register_names), 4)

        # Check that no additional registers were added to the circuit.
        self.assertEqual(len(qc.qregs), 1)
        self.assertEqual(len(qc.cregs), 1)

        # Check that the registerless-register names are recalculated after adding more registers,
        # to avoid naming clashes in this case.
        generated_names = qasm_register_names - {register.name for register in initial_registers}
        for generated_name in generated_names:
            qc.add_register(QuantumRegister(1, name=generated_name))
        qasm_register_names = set()
        for statement in qasm2.dumps(qc).split(";"):
            match = register_regex.match(statement)
            if match:
                qasm_register_names.add(match.group(1))
        self.assertEqual(len(qasm_register_names), 6)

    def test_repeated_instruction_names(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        # Create some random custom gate and name it "custom"
        custom = QuantumCircuit(1)
        custom.h(0)
        custom.y(0)
        gate = custom.to_gate()
        gate.name = "custom"
        # Another random custom gate named "custom" as well
        custom2 = QuantumCircuit(2)
        custom2.x(0)
        custom2.z(1)
        gate2 = custom2.to_gate()
        gate2.name = "custom"
        # Append custom gates with same name to original circuit
        qc.append(gate, [0])
        qc.append(gate2, [1, 0])
        # Expected qasm string will append the id to the second gate with repeated name
        expected_qasm = f"""\
OPENQASM 2.0;
include "qelib1.inc";
gate custom q0 {{ h q0; y q0; }}
gate custom_{id(gate2)} q0,q1 {{ x q0; z q1; }}
qreg q[2];
h q[0];
x q[1];
custom q[0];
custom_{id(gate2)} q[1],q[0];"""
        # Check qasm() produced the correct string
        self.assertEqual(expected_qasm, qasm2.dumps(qc))
        # Check instruction names were not changed by qasm()
        names = ["h", "x", "custom", "custom"]
        for idx, instruction in enumerate(qc._data):
            self.assertEqual(instruction.operation.name, names[idx])

    def test_invalid_identifiers(self):
        qc = QuantumCircuit(2)

        # Create some gate and give it an invalid name
        custom = QuantumCircuit(1)
        custom.x(0)
        custom.u(0, 0, pi, 0)
        gate = custom.to_gate()
        gate.name = "A[$]"

        # Another gate also with invalid name
        custom2 = QuantumCircuit(2)
        custom2.x(0)
        custom2.append(gate, [1])
        gate2 = custom2.to_gate()
        gate2.name = "invalid[name]"

        # Append gates
        qc.append(gate, [0])
        qc.append(gate2, [1, 0])

        # Expected qasm with valid identifiers
        expected_qasm = """\
OPENQASM 2.0;
include "qelib1.inc";
gate gate_A___ q0 { x q0; u(0,0,pi) q0; }
gate invalid_name_ q0,q1 { x q0; gate_A___ q1; }
qreg q[2];
gate_A___ q[0];
invalid_name_ q[1],q[0];"""

        self.assertEqual(expected_qasm, qasm2.dumps(qc))

        names = ["A[$]", "invalid[name]"]
        for idx, instruction in enumerate(qc._data):
            self.assertEqual(instruction.operation.name, names[idx])

    def test_duplicate_invalid_identifiers(self):
        base = QuantumCircuit(1)

        # First gate with invalid name, escapes to "invalid__"
        clash1 = QuantumCircuit(1, name="invalid??")
        clash1.x(0)
        base.append(clash1, [0])

        # Second gate with invalid name that also escapes to "invalid__"
        clash2 = QuantumCircuit(1, name="invalid[]")
        clash2.z(0)
        base.append(clash2, [0])

        # Check qasm is correctly produced
        names = set()
        for match in re.findall(r"gate (\S+)", qasm2.dumps(base)):
            self.assertTrue(VALID_QASM2_IDENTIFIER.fullmatch(match))
            names.add(match)
        self.assertEqual(len(names), 2)

        # Check instruction names were not changed by qasm()
        names = ["invalid??", "invalid[]"]
        for idx, instruction in enumerate(base._data):
            self.assertEqual(instruction.operation.name, names[idx])

    def test_circuit_qasm_escapes_register_names(self):
        """Test that registers that have invalid OpenQASM 2 names get correctly escaped, even when
        they would escape to the same value."""
        qc = QuantumCircuit(QuantumRegister(2, "?invalid"), QuantumRegister(2, "!invalid"))
        qc.cx(0, 1)
        qc.cx(2, 3)
        qasm = qasm2.dumps(qc)
        match = re.fullmatch(
            rf"""OPENQASM 2.0;
include "qelib1.inc";
qreg ({VALID_QASM2_IDENTIFIER.pattern})\[2\];
qreg ({VALID_QASM2_IDENTIFIER.pattern})\[2\];
cx \1\[0\],\1\[1\];
cx \2\[0\],\2\[1\];""",
            qasm,
        )
        self.assertTrue(match)
        self.assertNotEqual(match.group(1), match.group(2))

    def test_escapes_reserved(self):
        """Test that the OpenQASM 2 exporter won't export reserved names."""
        qc = QuantumCircuit(QuantumRegister(1, "qreg"))
        gate = Gate("gate", 1, [])
        gate.definition = QuantumCircuit(1)
        qc.append(gate, [qc.qubits[0]])
        qasm = qasm2.dumps(qc)
        match = re.fullmatch(
            rf"""OPENQASM 2.0;
include "qelib1.inc";
gate ({VALID_QASM2_IDENTIFIER.pattern}) q0 {{  }}
qreg ({VALID_QASM2_IDENTIFIER.pattern})\[1\];
\1 \2\[0\];""",
            qasm,
        )
        self.assertTrue(match)
        self.assertNotEqual(match.group(1), "gate")
        self.assertNotEqual(match.group(1), "qreg")

    def test_double_precision_rotation_angle(self):
        """Test that qasm() emits high precision rotation angles per default."""
        from qiskit.circuit.tools.pi_check import MAX_FRAC

        qc = QuantumCircuit(1)
        qc.p(0.123456789, 0)
        qc.p(pi * pi, 0)
        qc.p(MAX_FRAC * pi + 1, 0)

        boundary = MAX_FRAC * pi + 1
        expected_qasm = f"""\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
p(0.123456789) q[0];
p(9.869604401089358) q[0];
p({boundary:#}) q[0];"""
        self.assertEqual(qasm2.dumps(qc), expected_qasm)

    def test_rotation_angles_close_to_pi(self):
        qc = QuantumCircuit(1)
        qc.p(pi + 1e-11, 0)
        qc.p(pi + 1e-12, 0)
        expected_qasm = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
p(3.141592653599793) q[0];
p(pi) q[0];"""
        self.assertEqual(qasm2.dumps(qc), expected_qasm)

    def test_raises_invalid_custom_gate_no_qubits(self):
        legit_circuit = QuantumCircuit(5, name="legit_circuit")
        empty_circuit = QuantumCircuit(name="empty_circuit")
        legit_circuit.append(empty_circuit)

        with self.assertRaisesRegex(qasm2.QASM2ExportError, "acts on zero qubits"):
            qasm2.dumps(legit_circuit)

    def test_raises_invalid_custom_gate_clbits(self):
        instruction = QuantumCircuit(2, 2, name="inst")
        instruction.cx(0, 1)
        instruction.measure([0, 1], [0, 1])
        custom_instruction = instruction.to_instruction()

        qc = QuantumCircuit(2, 2)
        qc.append(custom_instruction, [0, 1], [0, 1])

        with self.assertRaisesRegex(qasm2.QASM2ExportError, "acts on 2 classical bits"):
            qasm2.dumps(qc)

    def test_permutations(self):
        """Test circuit qasm() method with Permutation gates."""

        qc = QuantumCircuit(4)
        qc.append(lib.PermutationGate([2, 1, 0]), [0, 1, 2])

        expected_qasm = """\
OPENQASM 2.0;
include "qelib1.inc";
gate permutation__2_1_0_ q0,q1,q2 { swap q0,q2; }
qreg q[4];
permutation__2_1_0_ q[0],q[1],q[2];"""
        self.assertEqual(qasm2.dumps(qc), expected_qasm)

    def test_multiple_permutation(self):
        """Test that multiple PermutationGates can be added to a circuit."""
        custom = QuantumCircuit(3, name="custom")
        custom.append(lib.PermutationGate([2, 1, 0]), [0, 1, 2])
        custom.append(lib.PermutationGate([0, 1, 2]), [0, 1, 2])

        qc = QuantumCircuit(4)
        qc.append(lib.PermutationGate([2, 1, 0]), [0, 1, 2], [])
        qc.append(lib.PermutationGate([1, 2, 0]), [0, 1, 2], [])
        qc.append(custom.to_gate(), [1, 3, 2], [])
        qasm = qasm2.dumps(qc)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
gate permutation__2_1_0_ q0,q1,q2 { swap q0,q2; }
gate permutation__1_2_0_ q0,q1,q2 { swap q1,q2; swap q0,q2; }
gate permutation__0_1_2_ q0,q1,q2 {  }
gate custom q0,q1,q2 { permutation__2_1_0_ q0,q1,q2; permutation__0_1_2_ q0,q1,q2; }
qreg q[4];
permutation__2_1_0_ q[0],q[1],q[2];
permutation__1_2_0_ q[0],q[1],q[2];
custom q[1],q[3],q[2];"""
        self.assertEqual(qasm, expected)

    def test_reset(self):
        qc = QuantumCircuit(2)
        qc.reset([0, 1])

        expected_qasm = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
reset q[0];
reset q[1];"""
        self.assertEqual(qasm2.dumps(qc), expected_qasm)

    def test_nested_gate_naming_clashes(self):
        """Test that gates that have naming clashes but only appear in the body of another gate
        still get exported correctly."""

        class Inner(Gate):
            def __init__(self, param):
                super().__init__("inner", 1, [param])

            def _define(self):
                self._definition = QuantumCircuit(1)
                self._definition.rx(self.params[0], 0)

        class Outer(Gate):
            def __init__(self, param):
                super().__init__("outer", 1, [param])

            def _define(self):
                self._definition = QuantumCircuit(1)
                self._definition.append(Inner(self.params[0]), [0], [])

        qc = QuantumCircuit(1)
        qc.append(Outer(1.0), [0], [])
        qc.append(Outer(2.0), [0], [])
        qasm = qasm2.dumps(qc)

        expected = re.compile(
            r"""OPENQASM 2\.0;
include "qelib1\.inc";
gate inner\(param0\) q0 { rx\(1\.0\) q0; }
gate outer\(param0\) q0 { inner\(1\.0\) q0; }
gate (?P<inner1>inner_[0-9]*)\(param0\) q0 { rx\(2\.0\) q0; }
gate (?P<outer1>outer_[0-9]*)\(param0\) q0 { (?P=inner1)\(2\.0\) q0; }
qreg q\[1\];
outer\(1\.0\) q\[0\];
(?P=outer1)\(2\.0\) q\[0\];""",
            re.MULTILINE,
        )
        self.assertRegex(qasm, expected)

    def test_opaque_output(self):
        """Test that gates with no definition are exported as `opaque`."""
        custom = QuantumCircuit(1, name="custom")
        custom.append(Gate("my_c", 1, []), [0])

        qc = QuantumCircuit(2)
        qc.append(Gate("my_a", 1, []), [0])
        qc.append(Gate("my_a", 1, []), [1])
        qc.append(Gate("my_b", 2, [1.0]), [1, 0])
        qc.append(custom.to_gate(), [0], [])
        qasm = qasm2.dumps(qc)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
opaque my_a q0;
opaque my_b(param0) q0,q1;
opaque my_c q0;
gate custom q0 { my_c q0; }
qreg q[2];
my_a q[0];
my_a q[1];
my_b(1.0) q[1],q[0];
custom q[0];"""
        self.assertEqual(qasm, expected)

    def test_sequencial_inner_gates_with_same_name(self):
        """Test if inner gates sequentially added with the same name result in the correct qasm"""
        qubits_range = range(3)

        gate_a = QuantumCircuit(3, name="a")
        gate_a.h(qubits_range)
        gate_a = gate_a.to_instruction()

        gate_b = QuantumCircuit(3, name="a")
        gate_b.append(gate_a, qubits_range)
        gate_b.x(qubits_range)
        gate_b = gate_b.to_instruction()

        qc = QuantumCircuit(3)
        qc.append(gate_b, qubits_range)
        qc.z(qubits_range)

        gate_a_id = id(qc.data[0].operation)

        expected_output = f"""\
OPENQASM 2.0;
include "qelib1.inc";
gate a q0,q1,q2 {{ h q0; h q1; h q2; }}
gate a_{gate_a_id} q0,q1,q2 {{ a q0,q1,q2; x q0; x q1; x q2; }}
qreg q[3];
a_{gate_a_id} q[0],q[1],q[2];
z q[0];
z q[1];
z q[2];"""

        self.assertEqual(qasm2.dumps(qc), expected_output)

    def test_empty_barrier(self):
        """Test that a blank barrier statement in _Qiskit_ acts over all qubits, while an explicitly
        no-op barrier (assuming Qiskit continues to allow this) is not output to OQ2 at all, since
        the statement requires an argument in the spec."""
        qc = QuantumCircuit(QuantumRegister(2, "qr1"), QuantumRegister(3, "qr2"))
        qc.barrier()  # In Qiskit land, this affects _all_ qubits.
        qc.barrier([])  # This explicitly affects _no_ qubits (so is totally meaningless).

        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg qr1[2];
qreg qr2[3];
barrier qr1[0],qr1[1],qr2[0],qr2[1],qr2[2];"""
        self.assertEqual(qasm2.dumps(qc), expected)

    def test_small_angle_valid(self):
        """Test that small angles do not get converted to invalid OQ2 floating-point values."""
        # OQ2 _technically_ requires a decimal point in all floating-point values, even ones that
        # are followed by an exponent.
        qc = QuantumCircuit(1)
        qc.rx(0.000001, 0)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx(1.e-06) q[0];"""
        self.assertEqual(qasm2.dumps(qc), expected)


class TestDumpStream(QiskitTestCase):
    """Specific tests for the stream handling in `dump`."""

    def test_can_write_to_filename(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        prevdir = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                qasm2.dump(qc, "myfile.qasm")
                with open("myfile.qasm") as fptr:
                    written = fptr.read()
            finally:
                os.chdir(prevdir)
        self.assertEqual(qasm2.loads(written), qc)

    def test_can_write_to_pathlike(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        prevdir = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                qasm2.dump(qc, pathlib.Path(".") / "myfile.qasm")
                with open("myfile.qasm") as fptr:
                    written = fptr.read()
            finally:
                os.chdir(prevdir)
        self.assertEqual(qasm2.loads(written), qc)

    def test_can_write_to_stream(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        with io.StringIO() as stream:
            qasm2.dump(qc, stream)
            written = stream.getvalue()
        self.assertEqual(qasm2.loads(written), qc)

    def test_measure_can_be_conditioned(self):
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        qc = QuantumCircuit(qr, cr)
        with qc.if_test((cr, 0)):
            qc.measure(0, 0)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg qr[1];
creg cr[1];
if (cr == 0) measure qr[0] -> cr[0];"""
        self.assertEqual(qasm2.dumps(qc), expected)

    def test_reset_can_be_conditioned(self):
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        qc = QuantumCircuit(qr, cr)
        with qc.if_test((cr, 0)):
            qc.reset(0)
        expected = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg qr[1];
creg cr[1];
if (cr == 0) reset qr[0];"""
        self.assertEqual(qasm2.dumps(qc), expected)

    def test_escaped_register_name_works_in_conditional(self):
        qr = QuantumRegister(1, "qr")
        # This is an invalid register name in OpenQASM 2 because it collides with a keyword.
        cr = ClassicalRegister(1, "measure")
        qc = QuantumCircuit(qr, cr)
        with qc.if_test((cr, 0)):
            qc.x(0)
        expected = r"""OPENQASM 2\.0;
include "qelib1\.inc";
qreg qr\[1\];
creg (\w+)\[1\];
if \(\1 == 0\) x qr\[0\];"""
        match = re.fullmatch(expected, qasm2.dumps(qc), flags=re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertNotEqual(match.group(0), "measure")

    def test_while_loop_fails(self):
        qc = QuantumCircuit(1, 1)
        with qc.while_loop((qc.cregs[0], 0)):
            qc.x(0)
        with self.assertRaisesRegex(
            qasm2.QASM2ExportError, "does not support control-flow constructs"
        ):
            qasm2.dumps(qc)

    def test_for_loop_fails(self):
        qc = QuantumCircuit(1)
        with qc.for_loop(range(3)):
            qc.x(0)
        with self.assertRaisesRegex(
            qasm2.QASM2ExportError, "does not support control-flow constructs"
        ):
            qasm2.dumps(qc)

    def test_switch_case_fails(self):
        qc = QuantumCircuit(1, 2)
        with qc.switch(qc.cregs[0]) as case:
            with case(0):
                qc.x(0)
            with case(1):
                qc.z(0)
            with case(case.DEFAULT):
                qc.h(0)
        with self.assertRaisesRegex(
            qasm2.QASM2ExportError, "does not support control-flow constructs"
        ):
            qasm2.dumps(qc)

    def test_else_fails(self):
        qc = QuantumCircuit(1, 2)
        with qc.if_test((qc.cregs[0], 0)) as else_:
            qc.x(0)
        with else_:
            qc.z(0)
        with self.assertRaisesRegex(qasm2.QASM2ExportError, "does not support 'else' statements"):
            qasm2.dumps(qc)

    def test_compound_if_fails(self):
        # This test could be relaxed in the future by unrolling the `if` into multiple statements.
        qc = QuantumCircuit(1, 1)
        with qc.if_test((qc.cregs[0], 0)):
            qc.x(0)
            qc.z(0)
        with self.assertRaisesRegex(qasm2.QASM2ExportError, "only supports conditionals on single"):
            qasm2.dumps(qc)

    def test_bit_condition_fails(self):
        qc = QuantumCircuit(1, 2)
        with qc.if_test((qc.clbits[0], False)):
            qc.x(0)
        with self.assertRaisesRegex(qasm2.QASM2ExportError, "only supports register-equality"):
            qasm2.dumps(qc)

    def test_expr_condition_fails(self):
        qc = QuantumCircuit(1)
        with qc.if_test(expr.lift(True)):
            qc.x(0)
        with self.assertRaisesRegex(qasm2.QASM2ExportError, "only supports register-equality"):
            qasm2.dumps(qc)

    def test_conditional_barrier_fails(self):
        qc = QuantumCircuit(1, 1)
        with qc.if_test((qc.cregs[0], 0)):
            qc.barrier()
        with self.assertRaisesRegex(qasm2.QASM2ExportError, "barriers cannot be conditional"):
            qasm2.dumps(qc)

    def test_nested_if_fails(self):
        qc = QuantumCircuit(1, 1)
        with qc.if_test((qc.cregs[0], 0)), qc.if_test((qc.cregs[0], 1)):
            qc.x(0)
        with self.assertRaisesRegex(qasm2.QASM2ExportError, "does not support nested conditionals"):
            qasm2.dumps(qc)


if __name__ == "__main__":
    unittest.main()
