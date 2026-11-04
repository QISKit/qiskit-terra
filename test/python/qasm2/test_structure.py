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


import io
import math
import os
import pathlib
import pickle
import shutil
import tempfile

import ddt
import numpy as np

import qiskit.qasm2
from qiskit import qpy
from qiskit.circuit import (
    ClassicalRegister,
    Gate,
    Parameter,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
    library as lib,
)
from qiskit.quantum_info import Operator
from test import QiskitTestCase

from . import gate_builder


@ddt.ddt
class TestWhitespace(QiskitTestCase):
    def test_allows_empty(self):
        self.assertEqual(qiskit.qasm2.loads(""), QuantumCircuit())

    @ddt.data("", "\n", "\r\n", "\n  ", "\n\t", "\r\n\t")
    def test_empty_except_comment(self, terminator):
        program = "// final comment" + terminator
        self.assertEqual(qiskit.qasm2.loads(program), QuantumCircuit())

    @ddt.data("", "\n", "\r\n", "\n  ")
    def test_final_comment(self, terminator):
        # This is similar to the empty-circuit test, except that we also have an instruction.
        program = "qreg q[2]; // final comment" + terminator
        self.assertEqual(qiskit.qasm2.loads(program), QuantumCircuit(QuantumRegister(2, "q")))


class TestVersion(QiskitTestCase):
    def test_complete_version(self):
        program = "OPENQASM 2.0;"
        parsed = qiskit.qasm2.loads(program)
        self.assertEqual(parsed, QuantumCircuit())

    def test_incomplete_version(self):
        program = "OPENQASM 2;"
        parsed = qiskit.qasm2.loads(program)
        self.assertEqual(parsed, QuantumCircuit())

    def test_after_comment(self):
        program = """
            // hello, world
            OPENQASM 2.0;
            qreg q[2];
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        self.assertEqual(parsed, qc)


class TestRegisters(QiskitTestCase):
    def test_qreg(self):
        program = "qreg q1[2]; qreg q2[1]; qreg q3[4];"
        parsed = qiskit.qasm2.loads(program)
        regs = [QuantumRegister(2, "q1"), QuantumRegister(1, "q2"), QuantumRegister(4, "q3")]
        self.assertEqual(list(parsed.qregs), regs)
        self.assertEqual(list(parsed.cregs), [])

    def test_creg(self):
        program = "creg c1[2]; creg c2[1]; creg c3[4];"
        parsed = qiskit.qasm2.loads(program)
        regs = [ClassicalRegister(2, "c1"), ClassicalRegister(1, "c2"), ClassicalRegister(4, "c3")]
        self.assertEqual(list(parsed.cregs), regs)
        self.assertEqual(list(parsed.qregs), [])

    def test_interleaved_registers(self):
        program = "qreg q1[3]; creg c1[2]; qreg q2[1]; creg c2[1];"
        parsed = qiskit.qasm2.loads(program)
        qregs = [QuantumRegister(3, "q1"), QuantumRegister(1, "q2")]
        cregs = [ClassicalRegister(2, "c1"), ClassicalRegister(1, "c2")]
        self.assertEqual(list(parsed.qregs), qregs)
        self.assertEqual(list(parsed.cregs), cregs)

    def test_registers_after_gate(self):
        program = "qreg before[2]; CX before[0], before[1]; qreg after[2]; CX after[0], after[1];"
        parsed = qiskit.qasm2.loads(program)
        before = QuantumRegister(2, "before")
        after = QuantumRegister(2, "after")
        qc = QuantumCircuit(before, after)
        qc.cx(before[0], before[1])
        qc.cx(after[0], after[1])
        self.assertEqual(parsed, qc)

    def test_empty_registers(self):
        program = "qreg q[0]; creg c[0];"
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(0, "q"), ClassicalRegister(0, "c"))
        self.assertEqual(parsed, qc)


@ddt.ddt
class TestGateApplication(QiskitTestCase):
    def test_builtin_single(self):
        program = """
            qreg q[2];
            U(0, 0, 0) q[0];
            CX q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.u(0, 0, 0, 0)
        qc.cx(0, 1)
        self.assertEqual(parsed, qc)

    def test_builtin_1q_broadcast(self):
        program = "qreg q[2]; U(0, 0, 0) q;"
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.u(0, 0, 0, 0)
        qc.u(0, 0, 0, 1)
        self.assertEqual(parsed, qc)

    def test_builtin_2q_broadcast(self):
        program = """
            qreg q1[2];
            qreg q2[2];
            CX q1[0], q2;
            barrier;
            CX q1, q2[1];
            barrier;
            CX q1, q2;
        """
        parsed = qiskit.qasm2.loads(program)
        q1 = QuantumRegister(2, "q1")
        q2 = QuantumRegister(2, "q2")
        qc = QuantumCircuit(q1, q2)
        qc.cx(q1[0], q2[0])
        qc.cx(q1[0], q2[1])
        qc.barrier()
        qc.cx(q1[0], q2[1])
        qc.cx(q1[1], q2[1])
        qc.barrier()
        qc.cx(q1[0], q2[0])
        qc.cx(q1[1], q2[1])
        self.assertEqual(parsed, qc)

    def test_3q_broadcast(self):
        program = """
            include "qelib1.inc";
            qreg q1[2];
            qreg q2[2];
            qreg q3[2];

            ccx q1, q2[0], q3[1];
            ccx q1[1], q2, q3[0];
            ccx q1[0], q2[1], q3;
            barrier;

            ccx q1, q2, q3[1];
            ccx q1[1], q2, q3;
            ccx q1, q2[1], q3;
            barrier;

            ccx q1, q2, q3;
        """
        parsed = qiskit.qasm2.loads(program)
        q1 = QuantumRegister(2, "q1")
        q2 = QuantumRegister(2, "q2")
        q3 = QuantumRegister(2, "q3")
        qc = QuantumCircuit(q1, q2, q3)
        qc.ccx(q1[0], q2[0], q3[1])
        qc.ccx(q1[1], q2[0], q3[1])
        qc.ccx(q1[1], q2[0], q3[0])
        qc.ccx(q1[1], q2[1], q3[0])
        qc.ccx(q1[0], q2[1], q3[0])
        qc.ccx(q1[0], q2[1], q3[1])
        qc.barrier()
        qc.ccx(q1[0], q2[0], q3[1])
        qc.ccx(q1[1], q2[1], q3[1])
        qc.ccx(q1[1], q2[0], q3[0])
        qc.ccx(q1[1], q2[1], q3[1])
        qc.ccx(q1[0], q2[1], q3[0])
        qc.ccx(q1[1], q2[1], q3[1])
        qc.barrier()
        qc.ccx(q1[0], q2[0], q3[0])
        qc.ccx(q1[1], q2[1], q3[1])
        self.assertEqual(parsed, qc)

    @ddt.data(True, False)
    def test_broadcast_against_empty_register(self, conditioned):
        cond = "if (cond == 0) " if conditioned else ""
        program = f"""
            OPENQASM 2;
            include "qelib1.inc";
            qreg q1[1];
            qreg q2[1];
            qreg empty1[0];
            qreg empty2[0];
            qreg empty3[0];
            creg cond[1];

            // None of the following statements should produce any gate applications.
            {cond}h empty1;

            {cond}cx q1[0], empty1;
            {cond}cx empty1, q2[0];
            {cond}cx empty1, empty2;

            {cond}ccx empty1, q1[0], q2[0];
            {cond}ccx q1[0], empty2, q2[0];
            {cond}ccx q1[0], q2[0], empty3;

            {cond}ccx empty1, empty2, q1[0];
            {cond}ccx empty1, q1[0], empty2;
            {cond}ccx q1[0], empty1, empty2;

            {cond}ccx empty1, empty2, empty3;
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(
            QuantumRegister(1, "q1"),
            QuantumRegister(1, "q2"),
            QuantumRegister(0, "empty1"),
            QuantumRegister(0, "empty2"),
            QuantumRegister(0, "empty3"),
            ClassicalRegister(1, "cond"),
        )
        self.assertEqual(parsed, qc)

    def test_conditioned(self):
        program = """
            qreg q[2];
            creg cond[1];
            if (cond == 0) U(0, 0, 0) q[0];
            if (cond == 1) CX q[1], q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        cond = ClassicalRegister(1, "cond")
        qc = QuantumCircuit(QuantumRegister(2, "q"), cond)
        with qc.if_test((cond, 0)):
            qc.u(0, 0, 0, 0)
        with qc.if_test((cond, 1)):
            qc.cx(1, 0)
        self.assertEqual(parsed, qc)

    def test_conditioned_broadcast(self):
        program = """
            qreg q1[2];
            qreg q2[2];
            creg cond[1];
            if (cond == 0) U(0, 0, 0) q1;
            if (cond == 1) CX q1[0], q2;
        """
        parsed = qiskit.qasm2.loads(program)
        cond = ClassicalRegister(1, "cond")
        q1 = QuantumRegister(2, "q1")
        q2 = QuantumRegister(2, "q2")
        qc = QuantumCircuit(q1, q2, cond)
        with qc.if_test((cond, 0)):
            qc.u(0, 0, 0, q1[0])
        with qc.if_test((cond, 0)):
            qc.u(0, 0, 0, q1[1])
        with qc.if_test((cond, 1)):
            qc.cx(q1[0], q2[0])
        with qc.if_test((cond, 1)):
            qc.cx(q1[0], q2[1])
        self.assertEqual(parsed, qc)

    def test_constant_folding(self):
        # Most expression-related things are tested in `test_expression.py` instead.
        program = """
            qreg q[1];
            U(4 + 3 * 2 ^ 2, cos(pi) * (1 - ln(1)), 2 ^ 3 ^ 2) q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.u(16.0, -1.0, 512.0, 0)
        self.assertEqual(parsed, qc)

    def test_call_defined_gate(self):
        program = """
            gate my_gate a {
                U(0, 0, 0) a;
            }
            qreg q[2];
            my_gate q[0];
            my_gate q;
        """
        parsed = qiskit.qasm2.loads(program)
        my_gate_def = QuantumCircuit([Qubit()])
        my_gate_def.u(0, 0, 0, 0)
        my_gate = gate_builder("my_gate", [], my_gate_def)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(my_gate(), [0])
        qc.append(my_gate(), [0])
        qc.append(my_gate(), [1])
        self.assertEqual(parsed, qc)

    def test_parameterless_gates_accept_parentheses(self):
        program = """
            qreg q[2];
            CX q[0], q[1];
            CX() q[1], q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.cx(0, 1)
        qc.cx(1, 0)
        self.assertEqual(parsed, qc)

    def test_huge_conditions(self):
        # Something way bigger than any native integer.
        bigint = (1 << 300) + 123456789
        program = f"""
            qreg qr[2];
            creg cr[2];
            creg cond[500];
            if (cond=={bigint}) U(0, 0, 0) qr[0];
            if (cond=={bigint}) U(0, 0, 0) qr;
            if (cond=={bigint}) reset qr[0];
            if (cond=={bigint}) reset qr;
            if (cond=={bigint}) measure qr[0] -> cr[0];
            if (cond=={bigint}) measure qr -> cr;
        """
        parsed = qiskit.qasm2.loads(program)
        qr, cr = QuantumRegister(2, "qr"), ClassicalRegister(2, "cr")
        cond = ClassicalRegister(500, "cond")
        qc = QuantumCircuit(qr, cr, cond)
        with qc.if_test((cond, bigint)):
            qc.u(0, 0, 0, qr[0])
        with qc.if_test((cond, bigint)):
            qc.u(0, 0, 0, qr[0])
        with qc.if_test((cond, bigint)):
            qc.u(0, 0, 0, qr[1])
        with qc.if_test((cond, bigint)):
            qc.reset(qr[0])
        with qc.if_test((cond, bigint)):
            qc.reset(qr[0])
        with qc.if_test((cond, bigint)):
            qc.reset(qr[1])
        with qc.if_test((cond, bigint)):
            qc.measure(qr[0], cr[0])
        with qc.if_test((cond, bigint)):
            qc.measure(qr[0], cr[0])
        with qc.if_test((cond, bigint)):
            qc.measure(qr[1], cr[1])
        self.assertEqual(parsed, qc)


class TestGateDefinition(QiskitTestCase):
    def test_simple_definition(self):
        program = """
            gate not_bell a, b {
                U(0, 0, 0) a;
                CX a, b;
            }
            qreg q[2];
            not_bell q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        not_bell_def = QuantumCircuit([Qubit(), Qubit()])
        not_bell_def.u(0, 0, 0, 0)
        not_bell_def.cx(0, 1)
        not_bell = gate_builder("not_bell", [], not_bell_def)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(not_bell(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_conditioned(self):
        program = """
            gate not_bell a, b {
                U(0, 0, 0) a;
                CX a, b;
            }
            qreg q[2];
            creg cond[1];
            if (cond == 0) not_bell q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        not_bell_def = QuantumCircuit([Qubit(), Qubit()])
        not_bell_def.u(0, 0, 0, 0)
        not_bell_def.cx(0, 1)
        not_bell = gate_builder("not_bell", [], not_bell_def)
        cond = ClassicalRegister(1, "cond")
        qc = QuantumCircuit(QuantumRegister(2, "q"), cond)
        with qc.if_test((cond, 0)):
            qc.append(not_bell(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_constant_folding_in_definition(self):
        program = """
            gate bell a, b {
                U(pi/2, 0, pi) a;
                CX a, b;
            }
            qreg q[2];
            bell q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        bell_def = QuantumCircuit([Qubit(), Qubit()])
        bell_def.u(math.pi / 2, 0, math.pi, 0)
        bell_def.cx(0, 1)
        bell = gate_builder("bell", [], bell_def)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(bell(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_parameterised_gate(self):
        # Most of the tests of deep parameter expressions are in `test_expression.py`.
        program = """
            gate my_gate(a, b) c {
                U(a, b, a + 2 * b) c;
            }
            qreg q[1];
            my_gate(0.25, 0.5) q[0];
            my_gate(0.5, 0.25) q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        a, b = Parameter("a"), Parameter("b")
        my_gate_def = QuantumCircuit([Qubit()])
        my_gate_def.u(a, b, a + 2 * b, 0)
        my_gate = gate_builder("my_gate", [a, b], my_gate_def)
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.append(my_gate(0.25, 0.5), [0])
        qc.append(my_gate(0.5, 0.25), [0])
        self.assertEqual(parsed, qc)

        # Also check the decomposition has come out exactly as expected.  The floating-point
        # assertions are safe as exact equality checks because there are no lossy operations with
        # these parameters, and the answer should be exact.
        decomposed = qc.decompose()
        self.assertEqual(decomposed.data[0].operation.name, "u")
        self.assertEqual(list(decomposed.data[0].operation.params), [0.25, 0.5, 1.25])
        self.assertEqual(decomposed.data[1].operation.name, "u")
        self.assertEqual(list(decomposed.data[1].operation.params), [0.5, 0.25, 1.0])

    def test_parameterless_gate_with_parentheses(self):
        program = """
            gate my_gate() a {
                U(0, 0, 0) a;
            }
            qreg q[1];
            my_gate q[0];
            my_gate() q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        my_gate_def = QuantumCircuit([Qubit()])
        my_gate_def.u(0, 0, 0, 0)
        my_gate = gate_builder("my_gate", [], my_gate_def)
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.append(my_gate(), [0])
        qc.append(my_gate(), [0])
        self.assertEqual(parsed, qc)

    def test_access_includes_in_definition(self):
        program = """
            include "qelib1.inc";
            gate bell a, b {
                h a;
                cx a, b;
            }
            qreg q[2];
            bell q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        bell_def = QuantumCircuit([Qubit(), Qubit()])
        bell_def.h(0)
        bell_def.cx(0, 1)
        bell = gate_builder("bell", [], bell_def)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(bell(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_access_previous_defined_gate(self):
        program = """
            include "qelib1.inc";
            gate bell a, b {
                h a;
                cx a, b;
            }
            gate second_bell a, b {
                bell b, a;
            }
            qreg q[2];
            second_bell q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        bell_def = QuantumCircuit([Qubit(), Qubit()])
        bell_def.h(0)
        bell_def.cx(0, 1)
        bell = gate_builder("bell", [], bell_def)

        second_bell_def = QuantumCircuit([Qubit(), Qubit()])
        second_bell_def.append(bell(), [1, 0])
        second_bell = gate_builder("second_bell", [], second_bell_def)

        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(second_bell(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_qubits_lookup_differently_to_gates(self):
        # The spec is somewhat unclear on this, and this leads to super weird text, but it's
        # technically unambiguously resolvable and this is more permissive.
        program = """
            include "qelib1.inc";
            gate bell h, cx {
                h h;
                cx h, cx;
            }
            qreg q[2];
            bell q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        bell_def = QuantumCircuit([Qubit(), Qubit()])
        bell_def.h(0)
        bell_def.cx(0, 1)
        bell = gate_builder("bell", [], bell_def)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(bell(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_parameters_lookup_differently_to_gates(self):
        # The spec is somewhat unclear on this, and this leads to super weird text, but it's
        # technically unambiguously resolvable and this is more permissive.
        program = """
            include "qelib1.inc";
            gate shadow(rx, rz) a {
                rz(rz) a;
                rx(rx) a;
            }
            qreg q[1];
            shadow(0.5, 2.0) q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        rx, rz = Parameter("rx"), Parameter("rz")
        shadow_def = QuantumCircuit([Qubit()])
        shadow_def.rz(rz, 0)
        shadow_def.rx(rx, 0)
        shadow = gate_builder("shadow", [rx, rz], shadow_def)
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.append(shadow(0.5, 2.0), [0])
        self.assertEqual(parsed, qc)

    def test_unused_parameters_convert_correctly(self):
        # The main risk here is that there might be lazy application in the gate definition
        # bindings, and we might accidentally try and bind parameters that aren't actually in the
        # definition.
        program = """
            gate my_gate(p) q {
                U(0, 0, 0) q;
            }
            qreg q[1];
            my_gate(0.5) q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        # No top-level circuit equality test here, because all the internals of gate application are
        # an implementation detail, and we don't want to tie the tests and implementation together
        # too closely.
        self.assertEqual(list(parsed.qregs), [QuantumRegister(1, "q")])
        self.assertEqual(list(parsed.cregs), [])
        self.assertEqual(len(parsed.data), 1)
        self.assertEqual(parsed.data[0].qubits, (parsed.qubits[0],))
        self.assertEqual(parsed.data[0].clbits, ())
        self.assertEqual(parsed.data[0].operation.name, "my_gate")
        self.assertEqual(list(parsed.data[0].operation.params), [0.5])

        decomposed = QuantumCircuit(QuantumRegister(1, "q"))
        decomposed.u(0, 0, 0, 0)
        self.assertEqual(parsed.decompose(), decomposed)

    def test_qubit_barrier_in_definition(self):
        program = """
            gate my_gate a, b {
                barrier a;
                barrier b;
                barrier a, b;
            }
            qreg q[2];
            my_gate q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        my_gate_def = QuantumCircuit([Qubit(), Qubit()])
        my_gate_def.barrier(0)
        my_gate_def.barrier(1)
        my_gate_def.barrier([0, 1])
        my_gate = gate_builder("my_gate", [], my_gate_def)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(my_gate(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_bare_barrier_in_definition(self):
        program = """
            gate my_gate a, b {
                barrier;
            }
            qreg q[2];
            my_gate q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        my_gate_def = QuantumCircuit([Qubit(), Qubit()])
        my_gate_def.barrier(my_gate_def.qubits)
        my_gate = gate_builder("my_gate", [], my_gate_def)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(my_gate(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_duplicate_barrier_in_definition(self):
        program = """
            gate my_gate a, b {
                barrier a, a;
                barrier b, a, b;
            }
            qreg q[2];
            my_gate q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        my_gate_def = QuantumCircuit([Qubit(), Qubit()])
        my_gate_def.barrier(0)
        my_gate_def.barrier([1, 0])
        my_gate = gate_builder("my_gate", [], my_gate_def)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(my_gate(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_pickleable(self):
        program = """
            include "qelib1.inc";
            gate my_gate(a) b, c {
                rz(2 * a) b;
                h b;
                cx b, c;
            }
            qreg q[2];
            my_gate(0.5) q[0], q[1];
            my_gate(0.25) q[1], q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        a = Parameter("a")
        my_gate_def = QuantumCircuit([Qubit(), Qubit()])
        my_gate_def.rz(2 * a, 0)
        my_gate_def.h(0)
        my_gate_def.cx(0, 1)
        my_gate = gate_builder("my_gate", [a], my_gate_def)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(my_gate(0.5), [0, 1])
        qc.append(my_gate(0.25), [1, 0])
        self.assertEqual(parsed, qc)
        with io.BytesIO() as fptr:
            pickle.dump(parsed, fptr)
            fptr.seek(0)
            loaded = pickle.load(fptr)
        self.assertEqual(parsed, loaded)

    def test_qpy_single_call_roundtrip(self):
        program = """
            include "qelib1.inc";
            gate my_gate(a) b, c {
                rz(2 * a) b;
                h b;
                cx b, c;
            }
            qreg q[2];
            my_gate(0.5) q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)

        # QPY won't persist custom gates by design choice, so instead let us check against the
        # explicit form it uses.
        my_gate_def = QuantumCircuit([Qubit(), Qubit()])
        my_gate_def.rz(1.0, 0)
        my_gate_def.h(0)
        my_gate_def.cx(0, 1)
        my_gate = Gate("my_gate", 2, [0.5])
        my_gate.definition = my_gate_def
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(my_gate, [0, 1])

        with io.BytesIO() as fptr:
            qpy.dump(parsed, fptr)
            fptr.seek(0)
            loaded = qpy.load(fptr)[0]
        self.assertEqual(loaded, qc)

    def test_qpy_double_call_roundtrip(self):
        program = """
            include "qelib1.inc";
            gate my_gate(a) b, c {
                rz(2 * a) b;
                h b;
                cx b, c;
            }
            qreg q[2];
            my_gate(0.5) q[0], q[1];
            my_gate(0.25) q[1], q[0];
        """
        parsed = qiskit.qasm2.loads(program)

        my_gate1_def = QuantumCircuit([Qubit(), Qubit()])
        my_gate1_def.rz(1.0, 0)
        my_gate1_def.h(0)
        my_gate1_def.cx(0, 1)
        my_gate1 = Gate("my_gate", 2, [0.5])
        my_gate1.definition = my_gate1_def

        my_gate2_def = QuantumCircuit([Qubit(), Qubit()])
        my_gate2_def.rz(0.5, 0)
        my_gate2_def.h(0)
        my_gate2_def.cx(0, 1)
        my_gate2 = Gate("my_gate", 2, [0.25])
        my_gate2.definition = my_gate2_def

        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(my_gate1, [0, 1])
        qc.append(my_gate2, [1, 0])

        with io.BytesIO() as fptr:
            qpy.dump(parsed, fptr)
            fptr.seek(0)
            loaded = qpy.load(fptr)[0]
        self.assertEqual(loaded, qc)


class TestOpaque(QiskitTestCase):
    def test_simple(self):
        program = """
            opaque my_gate a;
            opaque my_gate2() a;
            qreg q[2];
            my_gate q[0];
            my_gate() q[1];
            my_gate2 q[0];
            my_gate2() q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(Gate("my_gate", 1, []), [0])
        qc.append(Gate("my_gate", 1, []), [1])
        qc.append(Gate("my_gate2", 1, []), [0])
        qc.append(Gate("my_gate2", 1, []), [1])
        self.assertEqual(parsed, qc)

    def test_parameterised(self):
        program = """
            opaque my_gate(a, b) c, d;
            qreg q[2];
            my_gate(0.5, 0.25) q[1], q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(Gate("my_gate", 2, [0.5, 0.25]), [1, 0])
        self.assertEqual(parsed, qc)


class TestBarrier(QiskitTestCase):
    def test_single_register_argument(self):
        program = """
            qreg first[3];
            qreg second[3];
            barrier first;
            barrier second;
        """
        parsed = qiskit.qasm2.loads(program)
        first = QuantumRegister(3, "first")
        second = QuantumRegister(3, "second")
        qc = QuantumCircuit(first, second)
        qc.barrier(first)
        qc.barrier(second)
        self.assertEqual(parsed, qc)

    def test_single_qubit_argument(self):
        program = """
            qreg first[3];
            qreg second[3];
            barrier first[1];
            barrier second[0];
        """
        parsed = qiskit.qasm2.loads(program)
        first = QuantumRegister(3, "first")
        second = QuantumRegister(3, "second")
        qc = QuantumCircuit(first, second)
        qc.barrier(first[1])
        qc.barrier(second[0])
        self.assertEqual(parsed, qc)

    def test_empty_circuit_empty_arguments(self):
        program = "barrier;"
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit()
        self.assertEqual(parsed, qc)

    def test_one_register_circuit_empty_arguments(self):
        program = "qreg q1[2]; barrier;"
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(2, "q1"))
        qc.barrier(qc.qubits)
        self.assertEqual(parsed, qc)

    def test_multi_register_circuit_empty_arguments(self):
        program = "qreg q1[2]; qreg q2[3]; qreg q3[1]; barrier;"
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(
            QuantumRegister(2, "q1"), QuantumRegister(3, "q2"), QuantumRegister(1, "q3")
        )
        qc.barrier(qc.qubits)
        self.assertEqual(parsed, qc)

    def test_include_empty_register(self):
        program = """
            qreg q[2];
            qreg empty[0];
            barrier empty;
            barrier q, empty;
            barrier;
        """
        parsed = qiskit.qasm2.loads(program)
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, QuantumRegister(0, "empty"))
        qc.barrier(q)
        qc.barrier(qc.qubits)
        self.assertEqual(parsed, qc)

    def test_allows_duplicate_arguments(self):
        # There's nothing in the paper that implies this should be forbidden.
        program = """
            qreg q1[3];
            qreg q2[2];
            barrier q1, q1;
            barrier q1[0], q1;
            barrier q1, q1[0];
            barrier q1, q2, q1;
        """
        parsed = qiskit.qasm2.loads(program)
        q1 = QuantumRegister(3, "q1")
        q2 = QuantumRegister(2, "q2")
        qc = QuantumCircuit(q1, q2)
        qc.barrier(q1)
        qc.barrier(q1)
        qc.barrier(q1)
        qc.barrier(q1, q2)
        self.assertEqual(parsed, qc)


class TestMeasure(QiskitTestCase):
    def test_single(self):
        program = """
            qreg q[1];
            creg c[1];
            measure q[0] -> c[0];
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "c"))
        qc.measure(0, 0)
        self.assertEqual(parsed, qc)

    def test_broadcast(self):
        program = """
            qreg q[2];
            creg c[2];
            measure q -> c;
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"))
        qc.measure(0, 0)
        qc.measure(1, 1)
        self.assertEqual(parsed, qc)

    def test_conditioned(self):
        program = """
            qreg q[2];
            creg c[2];
            creg cond[1];
            if (cond == 0) measure q[0] -> c[0];
            if (cond == 1) measure q -> c;
        """
        parsed = qiskit.qasm2.loads(program)
        cond = ClassicalRegister(1, "cond")
        qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"), cond)
        with qc.if_test((cond, 0)):
            qc.measure(0, 0)
        with qc.if_test((cond, 1)):
            qc.measure(0, 0)
        with qc.if_test((cond, 1)):
            qc.measure(1, 1)
        self.assertEqual(parsed, qc)

    def test_broadcast_against_empty_register(self):
        program = """
            qreg q_empty[0];
            creg c_empty[0];
            measure q_empty -> c_empty;
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(0, "q_empty"), ClassicalRegister(0, "c_empty"))
        self.assertEqual(parsed, qc)

    def test_conditioned_broadcast_against_empty_register(self):
        program = """
            qreg q_empty[0];
            creg c_empty[0];
            creg cond[1];
            if (cond == 0) measure q_empty -> c_empty;
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(
            QuantumRegister(0, "q_empty"),
            ClassicalRegister(0, "c_empty"),
            ClassicalRegister(1, "cond"),
        )
        self.assertEqual(parsed, qc)

    def test_has_to_matrix(self):
        program = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg qr[1];
            gate my_gate(a) q {
                rz(a) q;
                rx(pi / 2) q;
                rz(-a) q;
            }
            my_gate(1.0) qr[0];
        """
        parsed = qiskit.qasm2.loads(program)
        expected = (
            lib.RZGate(-1.0).to_matrix()
            @ lib.RXGate(math.pi / 2).to_matrix()
            @ lib.RZGate(1.0).to_matrix()
        )
        defined_gate = parsed.data[0].operation
        self.assertEqual(defined_gate.name, "my_gate")
        np.testing.assert_allclose(defined_gate.to_matrix(), expected, atol=1e-14, rtol=0)
        # Also test that the standard `Operator` method on the whole circuit still works.
        np.testing.assert_allclose(Operator(parsed), expected, atol=1e-14, rtol=0)


class TestReset(QiskitTestCase):
    def test_single(self):
        program = """
            qreg q[1];
            reset q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.reset(0)
        self.assertEqual(parsed, qc)

    def test_broadcast(self):
        program = """
            qreg q[2];
            reset q;
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.reset(0)
        qc.reset(1)
        self.assertEqual(parsed, qc)

    def test_conditioned(self):
        program = """
            qreg q[2];
            creg cond[1];
            if (cond == 0) reset q[0];
            if (cond == 1) reset q;
        """
        parsed = qiskit.qasm2.loads(program)
        cond = ClassicalRegister(1, "cond")
        qc = QuantumCircuit(QuantumRegister(2, "q"), cond)
        with qc.if_test((cond, 0)):
            qc.reset(0)
        with qc.if_test((cond, 1)):
            qc.reset(0)
        with qc.if_test((cond, 1)):
            qc.reset(1)
        self.assertEqual(parsed, qc)

    def test_broadcast_against_empty_register(self):
        program = """
            qreg empty[0];
            reset empty;
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(0, "empty"))
        self.assertEqual(parsed, qc)

    def test_conditioned_broadcast_against_empty_register(self):
        program = """
            qreg empty[0];
            creg cond[1];
            if (cond == 0) reset empty;
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(0, "empty"), ClassicalRegister(1, "cond"))
        self.assertEqual(parsed, qc)


class TestInclude(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.tmp_dir = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        # Doesn't really matter if the removal fails, since this was a tempdir anyway; it'll get
        # cleaned up by the OS at some point.
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_qelib1_include(self):
        program = """
            include "qelib1.inc";
            qreg q[3];
            u3(0.5, 0.25, 0.125) q[0];
            u2(0.5, 0.25) q[0];
            u1(0.5) q[0];
            cx q[0], q[1];
            id q[0];
            x q[0];
            y q[0];
            z q[0];
            h q[0];
            s q[0];
            sdg q[0];
            t q[0];
            tdg q[0];
            rx(0.5) q[0];
            ry(0.5) q[0];
            rz(0.5) q[0];
            cz q[0], q[1];
            cy q[0], q[1];
            ch q[0], q[1];
            ccx q[0], q[1], q[2];
            crz(0.5) q[0], q[1];
            cu1(0.5) q[0], q[1];
            cu3(0.5, 0.25, 0.125) q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(3, "q"))
        qc.append(lib.U3Gate(0.5, 0.25, 0.125), [0])
        qc.append(lib.U2Gate(0.5, 0.25), [0])
        qc.append(lib.U1Gate(0.5), [0])
        qc.append(lib.CXGate(), [0, 1])
        qc.append(lib.UGate(0, 0, 0), [0])  # Stand-in for id.
        qc.append(lib.XGate(), [0])
        qc.append(lib.YGate(), [0])
        qc.append(lib.ZGate(), [0])
        qc.append(lib.HGate(), [0])
        qc.append(lib.SGate(), [0])
        qc.append(lib.SdgGate(), [0])
        qc.append(lib.TGate(), [0])
        qc.append(lib.TdgGate(), [0])
        qc.append(lib.RXGate(0.5), [0])
        qc.append(lib.RYGate(0.5), [0])
        qc.append(lib.RZGate(0.5), [0])
        qc.append(lib.CZGate(), [0, 1])
        qc.append(lib.CYGate(), [0, 1])
        qc.append(lib.CHGate(), [0, 1])
        qc.append(lib.CCXGate(), [0, 1, 2])
        qc.append(lib.CRZGate(0.5), [0, 1])
        qc.append(lib.CU1Gate(0.5), [0, 1])
        qc.append(lib.CU3Gate(0.5, 0.25, 0.125), [0, 1])
        self.assertEqual(parsed, qc)

    def test_qelib1_after_gate_definition(self):
        program = """
            gate bell a, b {
                U(pi/2, 0, pi) a;
                CX a, b;
            }
            include "qelib1.inc";
            qreg q[2];
            bell q[0], q[1];
            rx(0.5) q[0];
            bell q[1], q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        bell_def = QuantumCircuit([Qubit(), Qubit()])
        bell_def.u(math.pi / 2, 0, math.pi, 0)
        bell_def.cx(0, 1)
        bell = gate_builder("bell", [], bell_def)

        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(bell(), [0, 1])
        qc.rx(0.5, 0)
        qc.append(bell(), [1, 0])
        self.assertEqual(parsed, qc)

    def test_include_can_define_version(self):
        include = """
            OPENQASM 2.0;
            qreg inner_q[2];
        """
        with open(self.tmp_dir / "include.qasm", "w") as fp:
            fp.write(include)
        program = """
            OPENQASM 2.0;
            include "include.qasm";
        """
        parsed = qiskit.qasm2.loads(program, include_path=(self.tmp_dir,))
        qc = QuantumCircuit(QuantumRegister(2, "inner_q"))
        self.assertEqual(parsed, qc)

    def test_can_define_gates(self):
        include = """
            gate bell a, b {
                h a;
                cx a, b;
            }
        """
        with open(self.tmp_dir / "include.qasm", "w") as fp:
            fp.write(include)
        program = """
            OPENQASM 2.0;
            include "qelib1.inc";
            include "include.qasm";
            qreg q[2];
            bell q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program, include_path=(self.tmp_dir,))
        bell_def = QuantumCircuit([Qubit(), Qubit()])
        bell_def.h(0)
        bell_def.cx(0, 1)
        bell = gate_builder("bell", [], bell_def)

        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(bell(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_nested_include(self):
        inner = "creg c[2];"
        with open(self.tmp_dir / "inner.qasm", "w") as fp:
            fp.write(inner)
        outer = """
            qreg q[2];
            include "inner.qasm";
        """
        with open(self.tmp_dir / "outer.qasm", "w") as fp:
            fp.write(outer)
        program = """
            OPENQASM 2.0;
            include "outer.qasm";
        """
        parsed = qiskit.qasm2.loads(program, include_path=(self.tmp_dir,))
        qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"))
        self.assertEqual(parsed, qc)

    def test_first_hit_is_used(self):
        empty = self.tmp_dir / "empty"
        empty.mkdir()
        first = self.tmp_dir / "first"
        first.mkdir()
        with open(first / "include.qasm", "w") as fp:
            fp.write("qreg q[1];")
        second = self.tmp_dir / "second"
        second.mkdir()
        with open(second / "include.qasm", "w") as fp:
            fp.write("qreg q[2];")
        program = 'include "include.qasm";'
        parsed = qiskit.qasm2.loads(program, include_path=(empty, first, second))
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        self.assertEqual(parsed, qc)

    def test_qelib1_ignores_search_path(self):
        with open(self.tmp_dir / "qelib1.inc", "w") as fp:
            fp.write("qreg not_used[2];")
        program = 'include "qelib1.inc";'
        parsed = qiskit.qasm2.loads(program, include_path=(self.tmp_dir,))
        qc = QuantumCircuit()
        self.assertEqual(parsed, qc)

    def test_include_from_current_directory(self):
        include = """
            qreg q[2];
        """
        with open(self.tmp_dir / "include.qasm", "w") as fp:
            fp.write(include)
        program = """
            OPENQASM 2.0;
            include "include.qasm";
        """
        prevdir = os.getcwd()
        os.chdir(self.tmp_dir)
        try:
            parsed = qiskit.qasm2.loads(program)
            qc = QuantumCircuit(QuantumRegister(2, "q"))
            self.assertEqual(parsed, qc)
        finally:
            os.chdir(prevdir)

    def test_load_searches_source_directory(self):
        with open(self.tmp_dir / "include.qasm", "w") as fp:
            fp.write("qreg q[2];")
        program = 'include "include.qasm";'
        with open(self.tmp_dir / "program.qasm", "w") as fp:
            fp.write(program)
        parsed = qiskit.qasm2.load(self.tmp_dir / "program.qasm")
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        self.assertEqual(parsed, qc)

    def test_load_searches_source_directory_last(self):
        first = self.tmp_dir / "first"
        first.mkdir()
        with open(first / "include.qasm", "w") as fp:
            fp.write("qreg q[2];")
        with open(self.tmp_dir / "include.qasm", "w") as fp:
            fp.write("qreg not_used[2];")
        program = 'include "include.qasm";'
        with open(self.tmp_dir / "program.qasm", "w") as fp:
            fp.write(program)
        parsed = qiskit.qasm2.load(self.tmp_dir / "program.qasm", include_path=(first,))
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        self.assertEqual(parsed, qc)

    def test_load_searches_source_directory_prepend(self):
        first = self.tmp_dir / "first"
        first.mkdir()
        with open(first / "include.qasm", "w") as fp:
            fp.write("qreg not_used[2];")
        with open(self.tmp_dir / "include.qasm", "w") as fp:
            fp.write("qreg q[2];")
        program = 'include "include.qasm";'
        with open(self.tmp_dir / "program.qasm", "w") as fp:
            fp.write(program)
        parsed = qiskit.qasm2.load(
            self.tmp_dir / "program.qasm", include_path=(first,), include_input_directory="prepend"
        )
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        self.assertEqual(parsed, qc)

    def test_load_can_ignore_source_directory(self):
        with open(self.tmp_dir / "include.qasm", "w") as fp:
            fp.write("qreg q[2];")
        program = 'include "include.qasm";'
        with open(self.tmp_dir / "program.qasm", "w") as fp:
            fp.write(program)
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "unable to find 'include.qasm'"):
            qiskit.qasm2.load(self.tmp_dir / "program.qasm", include_input_directory=None)


@ddt.ddt
class TestCustomInstructions(QiskitTestCase):
    def test_qelib1_include_overridden(self):
        program = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            u3(0.5, 0.25, 0.125) q[0];
            u2(0.5, 0.25) q[0];
            u1(0.5) q[0];
            cx q[0], q[1];
            id q[0];
            x q[0];
            y q[0];
            z q[0];
            h q[0];
            s q[0];
            sdg q[0];
            t q[0];
            tdg q[0];
            rx(0.5) q[0];
            ry(0.5) q[0];
            rz(0.5) q[0];
            cz q[0], q[1];
            cy q[0], q[1];
            ch q[0], q[1];
            ccx q[0], q[1], q[2];
            crz(0.5) q[0], q[1];
            cu1(0.5) q[0], q[1];
            cu3(0.5, 0.25, 0.125) q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(
            program, custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        )
        qc = QuantumCircuit(QuantumRegister(3, "q"))
        qc.append(lib.U3Gate(0.5, 0.25, 0.125), [0])
        qc.append(lib.U2Gate(0.5, 0.25), [0])
        qc.append(lib.U1Gate(0.5), [0])
        qc.append(lib.CXGate(), [0, 1])
        qc.append(lib.IGate(), [0])
        qc.append(lib.XGate(), [0])
        qc.append(lib.YGate(), [0])
        qc.append(lib.ZGate(), [0])
        qc.append(lib.HGate(), [0])
        qc.append(lib.SGate(), [0])
        qc.append(lib.SdgGate(), [0])
        qc.append(lib.TGate(), [0])
        qc.append(lib.TdgGate(), [0])
        qc.append(lib.RXGate(0.5), [0])
        qc.append(lib.RYGate(0.5), [0])
        qc.append(lib.RZGate(0.5), [0])
        qc.append(lib.CZGate(), [0, 1])
        qc.append(lib.CYGate(), [0, 1])
        qc.append(lib.CHGate(), [0, 1])
        qc.append(lib.CCXGate(), [0, 1, 2])
        qc.append(lib.CRZGate(0.5), [0, 1])
        qc.append(lib.CU1Gate(0.5), [0, 1])
        qc.append(lib.CU3Gate(0.5, 0.25, 0.125), [0, 1])
        self.assertEqual(parsed, qc)

        # Also test that the output matches what Qiskit puts out.
        from_qiskit = QuantumCircuit.from_qasm_str(program)
        self.assertEqual(parsed, from_qiskit)

    def test_qelib1_sparse_overrides(self):
        """Test that the qelib1 special import still works as expected when a couple of gates in the
        middle of it are custom.  As long as qelib1 is handled specially, there is a risk that this
        handling will break in weird ways when custom instructions overlap it."""
        program = """
            include "qelib1.inc";
            qreg q[3];
            u3(0.5, 0.25, 0.125) q[0];
            u2(0.5, 0.25) q[0];
            u1(0.5) q[0];
            cx q[0], q[1];
            id q[0];
            x q[0];
            y q[0];
            z q[0];
            h q[0];
            s q[0];
            sdg q[0];
            t q[0];
            tdg q[0];
            rx(0.5) q[0];
            ry(0.5) q[0];
            rz(0.5) q[0];
            cz q[0], q[1];
            cy q[0], q[1];
            ch q[0], q[1];
            ccx q[0], q[1], q[2];
            crz(0.5) q[0], q[1];
            cu1(0.5) q[0], q[1];
            cu3(0.5, 0.25, 0.125) q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(
            program,
            custom_instructions=[
                qiskit.qasm2.CustomInstruction("id", 0, 1, lib.IGate),
                qiskit.qasm2.CustomInstruction("h", 0, 1, lib.HGate),
                qiskit.qasm2.CustomInstruction("crz", 1, 2, lib.CRZGate),
            ],
        )
        qc = QuantumCircuit(QuantumRegister(3, "q"))
        qc.append(lib.U3Gate(0.5, 0.25, 0.125), [0])
        qc.append(lib.U2Gate(0.5, 0.25), [0])
        qc.append(lib.U1Gate(0.5), [0])
        qc.append(lib.CXGate(), [0, 1])
        qc.append(lib.IGate(), [0])
        qc.append(lib.XGate(), [0])
        qc.append(lib.YGate(), [0])
        qc.append(lib.ZGate(), [0])
        qc.append(lib.HGate(), [0])
        qc.append(lib.SGate(), [0])
        qc.append(lib.SdgGate(), [0])
        qc.append(lib.TGate(), [0])
        qc.append(lib.TdgGate(), [0])
        qc.append(lib.RXGate(0.5), [0])
        qc.append(lib.RYGate(0.5), [0])
        qc.append(lib.RZGate(0.5), [0])
        qc.append(lib.CZGate(), [0, 1])
        qc.append(lib.CYGate(), [0, 1])
        qc.append(lib.CHGate(), [0, 1])
        qc.append(lib.CCXGate(), [0, 1, 2])
        qc.append(lib.CRZGate(0.5), [0, 1])
        qc.append(lib.CU1Gate(0.5), [0, 1])
        qc.append(lib.CU3Gate(0.5, 0.25, 0.125), [0, 1])
        self.assertEqual(parsed, qc)

    def test_user_gate_after_overidden_qelib1(self):
        program = """
            include "qelib1.inc";
            qreg q[1];
            opaque my_gate q;
            my_gate q[0];
        """
        parsed = qiskit.qasm2.loads(
            program, custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        )
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.append(Gate("my_gate", 1, []), [0])
        self.assertEqual(parsed, qc)

    def test_qiskit_extra_builtins(self):
        program = """
            qreg q[5];
            u(0.5 ,0.25, 0.125) q[0];
            p(0.5) q[0];
            sx q[0];
            sxdg q[0];
            swap q[0], q[1];
            cswap q[0], q[1], q[2];
            crx(0.5) q[0], q[1];
            cry(0.5) q[0], q[1];
            cp(0.5) q[0], q[1];
            csx q[0], q[1];
            cu(0.5, 0.25, 0.125, 0.0625) q[0], q[1];
            rxx(0.5) q[0], q[1];
            rzz(0.5) q[0], q[1];
            rccx q[0], q[1], q[2];
            rc3x q[0], q[1], q[2], q[3];
            c3x q[0], q[1], q[2], q[3];
            c3sqrtx q[0], q[1], q[2], q[3];
            c4x q[0], q[1], q[2], q[3], q[4];
        """
        parsed = qiskit.qasm2.loads(
            program, custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        )
        qc = QuantumCircuit(QuantumRegister(5, "q"))
        qc.append(lib.UGate(0.5, 0.25, 0.125), [0])
        qc.append(lib.PhaseGate(0.5), [0])
        qc.append(lib.SXGate(), [0])
        qc.append(lib.SXdgGate(), [0])
        qc.append(lib.SwapGate(), [0, 1])
        qc.append(lib.CSwapGate(), [0, 1, 2])
        qc.append(lib.CRXGate(0.5), [0, 1])
        qc.append(lib.CRYGate(0.5), [0, 1])
        qc.append(lib.CPhaseGate(0.5), [0, 1])
        qc.append(lib.CSXGate(), [0, 1])
        qc.append(lib.CUGate(0.5, 0.25, 0.125, 0.0625), [0, 1])
        qc.append(lib.RXXGate(0.5), [0, 1])
        qc.append(lib.RZZGate(0.5), [0, 1])
        qc.append(lib.RCCXGate(), [0, 1, 2])
        qc.append(lib.RC3XGate(), [0, 1, 2, 3])
        qc.append(lib.C3XGate(), [0, 1, 2, 3])
        qc.append(lib.C3SXGate(), [0, 1, 2, 3])
        qc.append(lib.C4XGate(), [0, 1, 2, 3, 4])
        self.assertEqual(parsed, qc)

        # There's also the 'u0' gate, but this is weird so we don't wildly care what its definition
        # is and it has no Qiskit equivalent, so we'll just test that it using it doesn't produce an
        # error.
        parsed = qiskit.qasm2.loads(
            "qreg q[1]; u0(1) q[0];", custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        )
        self.assertEqual(parsed.data[0].operation.name, "u0")

    def test_qiskit_override_delay_opaque(self):
        program = """
            opaque delay(t) q;
            qreg q[1];
            delay(1) q[0];
        """
        parsed = qiskit.qasm2.loads(
            program, custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        )
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.delay(1, 0, unit="dt")
        self.assertEqual(parsed, qc)

    def test_qiskit_override_u0_opaque(self):
        program = """
            opaque u0(n) q;
            qreg q[1];
            u0(2) q[0];
        """
        parsed = qiskit.qasm2.loads(
            program, custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        )
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.id(0)
        qc.id(0)
        self.assertEqual(parsed.decompose(), qc)

    def test_can_override_u(self):
        program = """
            qreg q[1];
            U(0.5, 0.25, 0.125) q[0];
        """

        class MyGate(Gate):
            def __init__(self, a, b, c):
                super().__init__("u", 1, [a, b, c])

        parsed = qiskit.qasm2.loads(
            program,
            custom_instructions=[qiskit.qasm2.CustomInstruction("U", 3, 1, MyGate, builtin=True)],
        )
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.append(MyGate(0.5, 0.25, 0.125), [0])
        self.assertEqual(parsed, qc)

    def test_can_override_cx(self):
        program = """
            qreg q[2];
            CX q[0], q[1];
        """

        class MyGate(Gate):
            def __init__(self):
                super().__init__("cx", 2, [])

        parsed = qiskit.qasm2.loads(
            program,
            custom_instructions=[qiskit.qasm2.CustomInstruction("CX", 0, 2, MyGate, builtin=True)],
        )
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(MyGate(), [0, 1])
        self.assertEqual(parsed, qc)

    @ddt.data(lambda x: x, reversed)
    def test_can_override_both_builtins_with_other_gates(self, order):
        program = """
            gate unimportant q {}
            qreg q[2];
            U(0.5, 0.25, 0.125) q[0];
            CX q[0], q[1];
        """

        class MyUGate(Gate):
            def __init__(self, a, b, c):
                super().__init__("u", 1, [a, b, c])

        class MyCXGate(Gate):
            def __init__(self):
                super().__init__("cx", 2, [])

        custom = [
            qiskit.qasm2.CustomInstruction("unused", 0, 1, lambda: Gate("unused", 1, [])),
            qiskit.qasm2.CustomInstruction("U", 3, 1, MyUGate, builtin=True),
            qiskit.qasm2.CustomInstruction("CX", 0, 2, MyCXGate, builtin=True),
        ]
        custom = order(custom)
        parsed = qiskit.qasm2.loads(program, custom_instructions=custom)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(MyUGate(0.5, 0.25, 0.125), [0])
        qc.append(MyCXGate(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_custom_builtin_gate(self):
        program = """
            qreg q[1];
            builtin(0.5) q[0];
        """

        class MyGate(Gate):
            def __init__(self, a):
                super().__init__("builtin", 1, [a])

        parsed = qiskit.qasm2.loads(
            program,
            custom_instructions=[
                qiskit.qasm2.CustomInstruction("builtin", 1, 1, MyGate, builtin=True)
            ],
        )
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.append(MyGate(0.5), [0])
        self.assertEqual(parsed, qc)

    def test_can_define_builtin_as_gate(self):
        program = """
            qreg q[1];
            gate builtin(t) q {}
            builtin(0.5) q[0];
        """

        class MyGate(Gate):
            def __init__(self, a):
                super().__init__("builtin", 1, [a])

        parsed = qiskit.qasm2.loads(
            program,
            custom_instructions=[
                qiskit.qasm2.CustomInstruction("builtin", 1, 1, MyGate, builtin=True)
            ],
        )
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.append(MyGate(0.5), [0])
        self.assertEqual(parsed, qc)

    def test_can_define_builtin_as_opaque(self):
        program = """
            qreg q[1];
            opaque builtin(t) q;
            builtin(0.5) q[0];
        """

        class MyGate(Gate):
            def __init__(self, a):
                super().__init__("builtin", 1, [a])

        parsed = qiskit.qasm2.loads(
            program,
            custom_instructions=[
                qiskit.qasm2.CustomInstruction("builtin", 1, 1, MyGate, builtin=True)
            ],
        )
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.append(MyGate(0.5), [0])
        self.assertEqual(parsed, qc)

    def test_can_define_custom_as_gate(self):
        program = """
            qreg q[1];
            gate my_gate(t) q {}
            my_gate(0.5) q[0];
        """

        class MyGate(Gate):
            def __init__(self, a):
                super().__init__("my_gate", 1, [a])

        parsed = qiskit.qasm2.loads(
            program, custom_instructions=[qiskit.qasm2.CustomInstruction("my_gate", 1, 1, MyGate)]
        )
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.append(MyGate(0.5), [0])
        self.assertEqual(parsed, qc)

    def test_can_define_custom_as_opaque(self):
        program = """
            qreg q[1];
            opaque my_gate(t) q;
            my_gate(0.5) q[0];
        """

        class MyGate(Gate):
            def __init__(self, a):
                super().__init__("my_gate", 1, [a])

        parsed = qiskit.qasm2.loads(
            program, custom_instructions=[qiskit.qasm2.CustomInstruction("my_gate", 1, 1, MyGate)]
        )
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.append(MyGate(0.5), [0])
        self.assertEqual(parsed, qc)

    def test_compatible_definition_of_builtin_is_ignored(self):
        program = """
            qreg q[1];
            gate my_gate a { U(0, 0, 0) a; }
            my_gate q[0];
        """

        class MyGate(Gate):
            def __init__(self):
                super().__init__("my_gate", 1, [])

            def _define(self):
                self._definition = QuantumCircuit(1)
                self._definition.z(0)

        parsed = qiskit.qasm2.loads(
            program, custom_instructions=[qiskit.qasm2.CustomInstruction("my_gate", 0, 1, MyGate)]
        )
        self.assertEqual(parsed.data[0].operation.definition, MyGate().definition)

    def test_gates_defined_after_a_builtin_align(self):
        """It's easy to get out of sync between the Rust-space and Python-space components when
        ``builtin=True``. See https://github.com/Qiskit/qiskit/issues/13339."""
        program = """
        OPENQASM 2.0;
        gate first a { U(0, 0, 0) a; }
        gate second a { U(pi, pi, pi) a; }

        qreg q[1];
        first q[0];
        second q[0];
        """
        custom = qiskit.qasm2.CustomInstruction("first", 0, 1, lib.XGate, builtin=True)
        parsed = qiskit.qasm2.loads(program, custom_instructions=[custom])
        # Provided definitions for built-in gates are ignored, so it should be an XGate directly.
        self.assertEqual(parsed.data[0].operation, lib.XGate())
        self.assertEqual(parsed.data[1].operation.name, "second")
        defn = parsed.data[1].operation.definition.copy_empty_like()
        defn.u(math.pi, math.pi, math.pi, 0)
        self.assertEqual(parsed.data[1].operation.definition, defn)


class TestCustomClassical(QiskitTestCase):
    def test_qiskit_extensions(self):
        program = """
            include "qelib1.inc";
            qreg q[1];
            rx(asin(0.3)) q[0];
            ry(acos(0.3)) q[0];
            rz(atan(0.3)) q[0];
        """
        parsed = qiskit.qasm2.loads(program, custom_classical=qiskit.qasm2.LEGACY_CUSTOM_CLASSICAL)
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.rx(math.asin(0.3), 0)
        qc.ry(math.acos(0.3), 0)
        qc.rz(math.atan(0.3), 0)
        self.assertEqual(parsed, qc)

    def test_zero_parameter_custom(self):
        program = """
            qreg q[1];
            U(f(), 0, 0) q[0];
        """
        parsed = qiskit.qasm2.loads(
            program, custom_classical=[qiskit.qasm2.CustomClassical("f", 0, lambda: 0.2)]
        )
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.u(0.2, 0, 0, 0)
        self.assertEqual(parsed, qc)

    def test_multi_parameter_custom(self):
        program = """
            qreg q[1];
            U(f(0.2), g(0.4, 0.1), h(1, 2, 3)) q[0];
        """
        parsed = qiskit.qasm2.loads(
            program,
            custom_classical=[
                qiskit.qasm2.CustomClassical("f", 1, lambda x: 1 + x),
                qiskit.qasm2.CustomClassical("g", 2, math.atan2),
                qiskit.qasm2.CustomClassical("h", 3, lambda x, y, z: z - y + x),
            ],
        )
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.u(1.2, math.atan2(0.4, 0.1), 2, 0)
        self.assertEqual(parsed, qc)

    def test_use_in_gate_definition(self):

        program = """
            gate my_gate(a, b) q {
                U(f(a, b), g(f(b, f(b, a))), b) q;
            }
            qreg q[1];
            my_gate(0.5, 0.25) q[0];
            my_gate(0.25, 0.5) q[0];
        """

        def f(x, y):
            return x - y

        def g(x):
            return 2 * x

        parsed = qiskit.qasm2.loads(
            program,
            custom_classical=[
                qiskit.qasm2.CustomClassical("f", 2, f),
                qiskit.qasm2.CustomClassical("g", 1, g),
            ],
        )
        first_gate = parsed.data[0].operation
        second_gate = parsed.data[1].operation
        self.assertEqual(list(first_gate.params), [0.5, 0.25])
        self.assertEqual(list(second_gate.params), [0.25, 0.5])

        self.assertEqual(
            list(first_gate.definition.data[0].operation.params),
            [
                f(0.5, 0.25),
                g(f(0.25, f(0.25, 0.5))),
                0.25,
            ],
        )
        self.assertEqual(
            list(second_gate.definition.data[0].operation.params),
            [
                f(0.25, 0.5),
                g(f(0.5, f(0.5, 0.25))),
                0.5,
            ],
        )


@ddt.ddt
class TestStrict(QiskitTestCase):
    @ddt.data(
        "gate my_gate(p0, p1$) q0, q1 {}",
        "gate my_gate(p0, p1) q0, q1$ {}",
        "opaque my_gate(p0, p1$) q0, q1;",
        "opaque my_gate(p0, p1) q0, q1$;",
        'include "qelib1.inc"; qreg q[2]; cu3(0.5, 0.25, 0.125$) q[0], q[1];',
        'include "qelib1.inc"; qreg q[2]; cu3(0.5, 0.25, 0.125) q[0], q[1]$;',
        "qreg q[2]; barrier q[0], q[1]$;",
        'include "qelib1.inc"; qreg q[1]; rx(sin(pi$)) q[0];',
    )
    def test_trailing_comma(self, program):
        without = qiskit.qasm2.loads("OPENQASM 2.0;\n" + program.replace("$", ""), strict=True)
        with_ = qiskit.qasm2.loads(program.replace("$", ","), strict=False)
        self.assertEqual(with_, without)

    def test_trailing_semicolon_after_gate(self):
        program = """
            include "qelib1.inc";
            gate bell a, b {
                h a;
                cx a, b;
            };  // <- the important bit of the test
            qreg q[2];
            bell q[0], q[1];
        """
        parsed = qiskit.qasm2.loads(program)
        bell_def = QuantumCircuit([Qubit(), Qubit()])
        bell_def.h(0)
        bell_def.cx(0, 1)
        bell = gate_builder("bell", [], bell_def)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.append(bell(), [0, 1])
        self.assertEqual(parsed, qc)

    def test_empty_statement(self):
        # This is allowed more as a side-effect of allowing the trailing semicolon after gate
        # definitions.
        program = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            h q[0];
            ;
            cx q[0], q[1];
            ;;;;
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(2, "q"))
        qc.h(0)
        qc.cx(0, 1)
        self.assertEqual(parsed, qc)

    def test_single_quoted_path(self):
        program = """
            include 'qelib1.inc';
            qreg q[1];
            h q[0];
        """
        parsed = qiskit.qasm2.loads(program)
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.h(0)
        self.assertEqual(parsed, qc)

    def test_unitary_qasm(self):
        """Test that UnitaryGate can be loaded by OQ2 correctly."""
        qc = QuantumCircuit(1)
        qc.unitary([[1, 0], [0, 1]], 0)
        qasm = """
            OPENQASM 2.0;
            include "qelib1.inc";
            gate unitary q0 { U(0,0,0) q0; }
            qreg q[1];
            unitary q[0];
        """
        parsed = qiskit.qasm2.loads(qasm)
        self.assertIsInstance(parsed, QuantumCircuit)
        self.assertIsInstance(parsed.data[0].operation, qiskit.qasm2.parse._DefinedGate)
        self.assertEqual(Operator.from_circuit(parsed), Operator.from_circuit(qc))
