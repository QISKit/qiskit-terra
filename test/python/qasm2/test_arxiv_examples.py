# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""These tests are the examples given in the arXiv paper describing OpenQASM 2.  Specifically, there
is a test for each subsection (except the description of 'qelib1.inc') in section 3 of
https://arxiv.org/abs/1707.03429v2. The examples are copy/pasted from the source files there."""


import math
import os
import tempfile

import ddt

from qiskit import qasm2
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit
from qiskit.circuit.library import U1Gate, U3Gate, CU1Gate
from test import QiskitTestCase

from . import gate_builder


def load(string, *args, **kwargs):
    # We're deliberately not using the context-manager form here because we need to use it in a
    # slightly odd pattern.

    temp = tempfile.NamedTemporaryFile(mode="w", delete=False)
    try:
        temp.write(string)
        # NamedTemporaryFile claims not to be openable a second time on Windows, so close it
        # (without deletion) so Rust can open it again.
        temp.close()
        return qasm2.load(temp.name, *args, **kwargs)
    finally:
        # Now actually clean up after ourselves.
        os.unlink(temp.name)


@ddt.ddt
class TestArxivExamples(QiskitTestCase):
    @ddt.data(qasm2.loads, load)
    def test_teleportation(self, parser):
        example = """\
// quantum teleportation example
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c0[1];
creg c1[1];
creg c2[1];
// optional post-rotation for state tomography
gate post q { }
u3(0.3,0.2,0.1) q[0];
h q[1];
cx q[1],q[2];
barrier q;
cx q[0],q[1];
h q[0];
measure q[0] -> c0[0];
measure q[1] -> c1[0];
if(c0==1) z q[2];
if(c1==1) x q[2];
post q[2];
measure q[2] -> c2[0];"""
        parsed = parser(example)

        post = gate_builder("post", [], QuantumCircuit([Qubit()]))

        q = QuantumRegister(3, "q")
        c0 = ClassicalRegister(1, "c0")
        c1 = ClassicalRegister(1, "c1")
        c2 = ClassicalRegister(1, "c2")

        qc = QuantumCircuit(q, c0, c1, c2)
        qc.append(U3Gate(0.3, 0.2, 0.1), [q[0]], [])
        qc.h(q[1])
        qc.cx(q[1], q[2])
        qc.barrier(q)
        qc.cx(q[0], q[1])
        qc.h(q[0])
        qc.measure(q[0], c0[0])
        qc.measure(q[1], c1[0])
        with qc.if_test((c0, 1)):
            qc.z(q[2])
        with qc.if_test((c1, 1)):
            qc.x(q[2])
        qc.append(post(), [q[2]], [])
        qc.measure(q[2], c2[0])

        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_qft(self, parser):
        example = """\
// quantum Fourier transform
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
x q[0];
x q[2];
barrier q;
h q[0];
cu1(pi/2) q[1],q[0];
h q[1];
cu1(pi/4) q[2],q[0];
cu1(pi/2) q[2],q[1];
h q[2];
cu1(pi/8) q[3],q[0];
cu1(pi/4) q[3],q[1];
cu1(pi/2) q[3],q[2];
h q[3];
measure q -> c;"""
        parsed = parser(example)

        qc = QuantumCircuit(QuantumRegister(4, "q"), ClassicalRegister(4, "c"))
        qc.x(0)
        qc.x(2)
        qc.barrier(range(4))
        qc.h(0)
        qc.append(CU1Gate(math.pi / 2), [1, 0])
        qc.h(1)
        qc.append(CU1Gate(math.pi / 4), [2, 0])
        qc.append(CU1Gate(math.pi / 2), [2, 1])
        qc.h(2)
        qc.append(CU1Gate(math.pi / 8), [3, 0])
        qc.append(CU1Gate(math.pi / 4), [3, 1])
        qc.append(CU1Gate(math.pi / 2), [3, 2])
        qc.h(3)
        qc.measure(range(4), range(4))

        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_inverse_qft_1(self, parser):
        example = """\
// QFT and measure, version 1
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q;
barrier q;
h q[0];
measure q[0] -> c[0];
if(c==1) u1(pi/2) q[1];
h q[1];
measure q[1] -> c[1];
if(c==1) u1(pi/4) q[2];
if(c==2) u1(pi/2) q[2];
if(c==3) u1(pi/2+pi/4) q[2];
h q[2];
measure q[2] -> c[2];
if(c==1) u1(pi/8) q[3];
if(c==2) u1(pi/4) q[3];
if(c==3) u1(pi/4+pi/8) q[3];
if(c==4) u1(pi/2) q[3];
if(c==5) u1(pi/2+pi/8) q[3];
if(c==6) u1(pi/2+pi/4) q[3];
if(c==7) u1(pi/2+pi/4+pi/8) q[3];
h q[3];
measure q[3] -> c[3];"""
        parsed = parser(example)

        q = QuantumRegister(4, "q")
        c = ClassicalRegister(4, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q)
        qc.barrier(q)
        qc.h(q[0])
        qc.measure(q[0], c[0])
        with qc.if_test((c, 1)):
            qc.append(U1Gate(math.pi / 2), [q[1]])
        qc.h(q[1])
        qc.measure(q[1], c[1])
        with qc.if_test((c, 1)):
            qc.append(U1Gate(math.pi / 4), [q[2]])
        with qc.if_test((c, 2)):
            qc.append(U1Gate(math.pi / 2), [q[2]])
        with qc.if_test((c, 3)):
            qc.append(U1Gate(math.pi / 4 + math.pi / 2), [q[2]])
        qc.h(q[2])
        qc.measure(q[2], c[2])
        with qc.if_test((c, 1)):
            qc.append(U1Gate(math.pi / 8), [q[3]])
        with qc.if_test((c, 2)):
            qc.append(U1Gate(math.pi / 4), [q[3]])
        with qc.if_test((c, 3)):
            qc.append(U1Gate(math.pi / 8 + math.pi / 4), [q[3]])
        with qc.if_test((c, 4)):
            qc.append(U1Gate(math.pi / 2), [q[3]])
        with qc.if_test((c, 5)):
            qc.append(U1Gate(math.pi / 8 + math.pi / 2), [q[3]])
        with qc.if_test((c, 6)):
            qc.append(U1Gate(math.pi / 4 + math.pi / 2), [q[3]])
        with qc.if_test((c, 7)):
            qc.append(U1Gate(math.pi / 8 + math.pi / 4 + math.pi / 2), [q[3]])
        qc.h(q[3])
        qc.measure(q[3], c[3])

        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_inverse_qft_2(self, parser):
        example = """\
// QFT and measure, version 2
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c0[1];
creg c1[1];
creg c2[1];
creg c3[1];
h q;
barrier q;
h q[0];
measure q[0] -> c0[0];
if(c0==1) u1(pi/2) q[1];
h q[1];
measure q[1] -> c1[0];
if(c0==1) u1(pi/4) q[2];
if(c1==1) u1(pi/2) q[2];
h q[2];
measure q[2] -> c2[0];
if(c0==1) u1(pi/8) q[3];
if(c1==1) u1(pi/4) q[3];
if(c2==1) u1(pi/2) q[3];
h q[3];
measure q[3] -> c3[0];"""
        parsed = parser(example)

        q = QuantumRegister(4, "q")
        c0 = ClassicalRegister(1, "c0")
        c1 = ClassicalRegister(1, "c1")
        c2 = ClassicalRegister(1, "c2")
        c3 = ClassicalRegister(1, "c3")
        qc = QuantumCircuit(q, c0, c1, c2, c3)
        qc.h(q)
        qc.barrier(q)
        qc.h(q[0])
        qc.measure(q[0], c0[0])
        with qc.if_test((c0, 1)):
            qc.append(U1Gate(math.pi / 2), [q[1]])
        qc.h(q[1])
        qc.measure(q[1], c1[0])
        with qc.if_test((c0, 1)):
            qc.append(U1Gate(math.pi / 4), [q[2]])
        with qc.if_test((c1, 1)):
            qc.append(U1Gate(math.pi / 2), [q[2]])
        qc.h(q[2])
        qc.measure(q[2], c2[0])
        with qc.if_test((c0, 1)):
            qc.append(U1Gate(math.pi / 8), [q[3]])
        with qc.if_test((c1, 1)):
            qc.append(U1Gate(math.pi / 4), [q[3]])
        with qc.if_test((c2, 1)):
            qc.append(U1Gate(math.pi / 2), [q[3]])
        qc.h(q[3])
        qc.measure(q[3], c3[0])
        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_ripple_carry_adder(self, parser):
        example = """\
// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
OPENQASM 2.0;
include "qelib1.inc";
gate majority a,b,c
{
  cx c,b;
  cx c,a;
  ccx a,b,c;
}
gate unmaj a,b,c
{
  ccx a,b,c;
  cx c,a;
  cx a,b;
}
qreg cin[1];
qreg a[4];
qreg b[4];
qreg cout[1];
creg ans[5];
// set input states
x a[0]; // a = 0001
x b;    // b = 1111
// add a to b, storing result in b
majority cin[0],b[0],a[0];
majority a[0],b[1],a[1];
majority a[1],b[2],a[2];
majority a[2],b[3],a[3];
cx a[3],cout[0];
unmaj a[2],b[3],a[3];
unmaj a[1],b[2],a[2];
unmaj a[0],b[1],a[1];
unmaj cin[0],b[0],a[0];
measure b[0] -> ans[0];
measure b[1] -> ans[1];
measure b[2] -> ans[2];
measure b[3] -> ans[3];
measure cout[0] -> ans[4];"""
        parsed = parser(example)

        majority_definition = QuantumCircuit([Qubit(), Qubit(), Qubit()])
        majority_definition.cx(2, 1)
        majority_definition.cx(2, 0)
        majority_definition.ccx(0, 1, 2)
        majority = gate_builder("majority", [], majority_definition)

        unmaj_definition = QuantumCircuit([Qubit(), Qubit(), Qubit()])
        unmaj_definition.ccx(0, 1, 2)
        unmaj_definition.cx(2, 0)
        unmaj_definition.cx(0, 1)
        unmaj = gate_builder("unmaj", [], unmaj_definition)

        cin = QuantumRegister(1, "cin")
        a = QuantumRegister(4, "a")
        b = QuantumRegister(4, "b")
        cout = QuantumRegister(1, "cout")
        ans = ClassicalRegister(5, "ans")

        qc = QuantumCircuit(cin, a, b, cout, ans)
        qc.x(a[0])
        qc.x(b)
        qc.append(majority(), [cin[0], b[0], a[0]])
        qc.append(majority(), [a[0], b[1], a[1]])
        qc.append(majority(), [a[1], b[2], a[2]])
        qc.append(majority(), [a[2], b[3], a[3]])
        qc.cx(a[3], cout[0])
        qc.append(unmaj(), [a[2], b[3], a[3]])
        qc.append(unmaj(), [a[1], b[2], a[2]])
        qc.append(unmaj(), [a[0], b[1], a[1]])
        qc.append(unmaj(), [cin[0], b[0], a[0]])
        qc.measure(b, ans[:4])
        qc.measure(cout[0], ans[4])

        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_randomised_benchmarking(self, parser):
        example = """\
// One randomized benchmarking sequence
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
barrier q;
cz q[0],q[1];
barrier q;
s q[0];
cz q[0],q[1];
barrier q;
s q[0];
z q[0];
h q[0];
barrier q;
measure q -> c;
        """
        parsed = parser(example)

        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.barrier(q)
        qc.cz(q[0], q[1])
        qc.barrier(q)
        qc.s(q[0])
        qc.cz(q[0], q[1])
        qc.barrier(q)
        qc.s(q[0])
        qc.z(q[0])
        qc.h(q[0])
        qc.barrier(q)
        qc.measure(q, c)

        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_process_tomography(self, parser):
        example = """\
OPENQASM 2.0;
include "qelib1.inc";
gate pre q { }   // pre-rotation
gate post q { }  // post-rotation
qreg q[1];
creg c[1];
pre q[0];
barrier q;
h q[0];
barrier q;
post q[0];
measure q[0] -> c[0];"""
        parsed = parser(example)

        pre = gate_builder("pre", [], QuantumCircuit([Qubit()]))
        post = gate_builder("post", [], QuantumCircuit([Qubit()]))

        qc = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "c"))
        qc.append(pre(), [0])
        qc.barrier(qc.qubits)
        qc.h(0)
        qc.barrier(qc.qubits)
        qc.append(post(), [0])
        qc.measure(0, 0)

        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_error_correction(self, parser):
        example = """\
// Repetition code syndrome measurement
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg a[2];
creg c[3];
creg syn[2];
gate syndrome d1,d2,d3,a1,a2
{
  cx d1,a1; cx d2,a1;
  cx d2,a2; cx d3,a2;
}
x q[0]; // error
barrier q;
syndrome q[0],q[1],q[2],a[0],a[1];
measure a -> syn;
if(syn==1) x q[0];
if(syn==2) x q[2];
if(syn==3) x q[1];
measure q -> c;"""
        parsed = parser(example)

        syndrome_definition = QuantumCircuit([Qubit() for _ in [None] * 5])
        syndrome_definition.cx(0, 3)
        syndrome_definition.cx(1, 3)
        syndrome_definition.cx(1, 4)
        syndrome_definition.cx(2, 4)
        syndrome = gate_builder("syndrome", [], syndrome_definition)

        q = QuantumRegister(3, "q")
        a = QuantumRegister(2, "a")
        c = ClassicalRegister(3, "c")
        syn = ClassicalRegister(2, "syn")

        qc = QuantumCircuit(q, a, c, syn)
        qc.x(q[0])
        qc.barrier(q)
        qc.append(syndrome(), [q[0], q[1], q[2], a[0], a[1]])
        qc.measure(a, syn)
        with qc.if_test((syn, 1)):
            qc.x(q[0])
        with qc.if_test((syn, 2)):
            qc.x(q[2])
        with qc.if_test((syn, 3)):
            qc.x(q[1])
        qc.measure(q, c)

        self.assertEqual(parsed, qc)
