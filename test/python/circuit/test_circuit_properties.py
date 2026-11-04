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

"""Test Qiskit's inverse gate operation."""

import unittest
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Clbit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import GlobalPhaseGate
from qiskit.circuit.exceptions import CircuitError
from test import QiskitTestCase


class TestCircuitProperties(QiskitTestCase):
    """QuantumCircuit properties tests."""

    def test_qarg_numpy_int(self):
        """Test castable to integer args for QuantumCircuit."""
        n = np.int64(12)
        qc1 = QuantumCircuit(n)
        self.assertEqual(qc1.num_qubits, 12)
        self.assertEqual(type(qc1), QuantumCircuit)

    def test_carg_numpy_int(self):
        """Test castable to integer cargs for QuantumCircuit."""
        n = np.int64(12)
        c1 = ClassicalRegister(n)
        qc1 = QuantumCircuit(c1)
        c_regs = qc1.cregs
        self.assertEqual(c_regs[0], c1)
        self.assertEqual(type(qc1), QuantumCircuit)

    def test_carg_numpy_int_2(self):
        """Test castable to integer cargs for QuantumCircuit."""
        qc1 = QuantumCircuit(12, np.int64(12))
        self.assertEqual(len(qc1.clbits), 12)
        self.assertTrue(all(isinstance(bit, Clbit) for bit in qc1.clbits))
        self.assertEqual(type(qc1), QuantumCircuit)

    def test_qarg_numpy_int_exception(self):
        """Test attempt to pass non-castable arg to QuantumCircuit."""
        self.assertRaises(CircuitError, QuantumCircuit, "string")

    def test_warning_on_noninteger_float(self):
        """Test warning when passing non-integer float to QuantumCircuit"""
        self.assertRaises(CircuitError, QuantumCircuit, 2.2)
        # but an integer float should pass
        qc = QuantumCircuit(2.0)
        self.assertEqual(qc.num_qubits, 2)

    def test_circuit_depth_empty(self):
        """Test depth of empty circuity"""
        q = QuantumRegister(5, "q")
        qc = QuantumCircuit(q)
        self.assertEqual(qc.depth(), 0)

    def test_circuit_depth_no_reg(self):
        """Test depth of no register circuits"""
        qc = QuantumCircuit()
        self.assertEqual(qc.depth(), 0)

    def test_circuit_depth_meas_only(self):
        """Test depth of measurement only"""
        q = QuantumRegister(1, "q")
        c = ClassicalRegister(1, "c")
        qc = QuantumCircuit(q, c)
        qc.measure(q, c)
        self.assertEqual(qc.depth(), 1)

    def test_circuit_depth_barrier(self):
        """Make sure barriers do not add to depth"""

        #         ┌───┐                     ░ ┌─┐
        #    q_0: ┤ H ├──■──────────────────░─┤M├────────────
        #         ├───┤┌─┴─┐                ░ └╥┘┌─┐
        #    q_1: ┤ H ├┤ X ├──■─────────────░──╫─┤M├─────────
        #         ├───┤└───┘  │  ┌───┐      ░  ║ └╥┘┌─┐
        #    q_2: ┤ H ├───────┼──┤ X ├──■───░──╫──╫─┤M├──────
        #         ├───┤       │  └─┬─┘┌─┴─┐ ░  ║  ║ └╥┘┌─┐
        #    q_3: ┤ H ├───────┼────┼──┤ X ├─░──╫──╫──╫─┤M├───
        #         ├───┤     ┌─┴─┐  │  └───┘ ░  ║  ║  ║ └╥┘┌─┐
        #    q_4: ┤ H ├─────┤ X ├──■────────░──╫──╫──╫──╫─┤M├
        #         └───┘     └───┘           ░  ║  ║  ║  ║ └╥┘
        #    c: 5/═════════════════════════════╩══╩══╩══╩══╩═
        #                                      0  1  2  3  4
        q = QuantumRegister(5, "q")
        c = ClassicalRegister(5, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.h(q[4])
        qc.cx(q[0], q[1])
        qc.cx(q[1], q[4])
        qc.cx(q[4], q[2])
        qc.cx(q[2], q[3])
        qc.barrier(q)
        qc.measure(q, c)
        self.assertEqual(qc.depth(), 6)

    def test_circuit_depth_simple(self):
        """Test depth for simple circuit"""
        #      ┌───┐
        # q_0: ┤ H ├──■────────────────────
        #      └───┘  │            ┌───┐┌─┐
        # q_1: ───────┼────────────┤ X ├┤M├
        #      ┌───┐  │  ┌───┐┌───┐└─┬─┘└╥┘
        # q_2: ┤ X ├──┼──┤ X ├┤ X ├──┼───╫─
        #      └───┘  │  └───┘└───┘  │   ║
        # q_3: ───────┼──────────────┼───╫─
        #           ┌─┴─┐┌───┐       │   ║
        # q_4: ─────┤ X ├┤ X ├───────■───╫─
        #           └───┘└───┘           ║
        # c: 1/══════════════════════════╩═
        #                                0
        q = QuantumRegister(5, "q")
        c = ClassicalRegister(1, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[4])
        qc.x(q[2])
        qc.x(q[2])
        qc.x(q[2])
        qc.x(q[4])
        qc.cx(q[4], q[1])
        qc.measure(q[1], c[0])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_multi_reg(self):
        """Test depth for multiple registers"""

        #       ┌───┐
        # q1_0: ┤ H ├──■─────────────────
        #       ├───┤┌─┴─┐
        # q1_1: ┤ H ├┤ X ├──■────────────
        #       ├───┤└───┘  │  ┌───┐
        # q1_2: ┤ H ├───────┼──┤ X ├──■──
        #       ├───┤       │  └─┬─┘┌─┴─┐
        # q2_0: ┤ H ├───────┼────┼──┤ X ├
        #       ├───┤     ┌─┴─┐  │  └───┘
        # q2_1: ┤ H ├─────┤ X ├──■───────
        #       └───┘     └───┘
        q1 = QuantumRegister(3, "q1")
        q2 = QuantumRegister(2, "q2")
        c = ClassicalRegister(5, "c")
        qc = QuantumCircuit(q1, q2, c)
        qc.h(q1[0])
        qc.h(q1[1])
        qc.h(q1[2])
        qc.h(q2[0])
        qc.h(q2[1])
        qc.cx(q1[0], q1[1])
        qc.cx(q1[1], q2[1])
        qc.cx(q2[1], q1[2])
        qc.cx(q1[2], q2[0])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_3q_gate(self):
        """Test depth for 3q gate"""

        #       ┌───┐
        # q1_0: ┤ H ├──■────■─────────────────
        #       ├───┤  │  ┌─┴─┐
        # q1_1: ┤ H ├──┼──┤ X ├──■────────────
        #       ├───┤  │  └───┘  │  ┌───┐
        # q1_2: ┤ H ├──┼─────────┼──┤ X ├──■──
        #       ├───┤┌─┴─┐       │  └─┬─┘┌─┴─┐
        # q2_0: ┤ H ├┤ X ├───────┼────┼──┤ X ├
        #       ├───┤└─┬─┘     ┌─┴─┐  │  └───┘
        # q2_1: ┤ H ├──■───────┤ X ├──■───────
        #       └───┘          └───┘
        q1 = QuantumRegister(3, "q1")
        q2 = QuantumRegister(2, "q2")
        c = ClassicalRegister(5, "c")
        qc = QuantumCircuit(q1, q2, c)
        qc.h(q1[0])
        qc.h(q1[1])
        qc.h(q1[2])
        qc.h(q2[0])
        qc.h(q2[1])
        qc.ccx(q2[1], q1[0], q2[0])
        qc.cx(q1[0], q1[1])
        qc.cx(q1[1], q2[1])
        qc.cx(q2[1], q1[2])
        qc.cx(q1[2], q2[0])
        self.assertEqual(qc.depth(), 6)

    def test_circuit_depth_measurements1(self):
        """Test circuit depth for measurements #1."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├─────────
        #      ├───┤└╥┘┌─┐
        # q_1: ┤ H ├─╫─┤M├──────
        #      ├───┤ ║ └╥┘┌─┐
        # q_2: ┤ H ├─╫──╫─┤M├───
        #      ├───┤ ║  ║ └╥┘┌─┐
        # q_3: ┤ H ├─╫──╫──╫─┤M├
        #      └───┘ ║  ║  ║ └╥┘
        # c: 4/══════╩══╩══╩══╩═
        #            0  1  2  3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.depth(), 2)

    def test_circuit_depth_measurements2(self):
        """Test circuit depth for measurements #2."""

        #      ┌───┐┌─┐┌─┐┌─┐┌─┐
        # q_0: ┤ H ├┤M├┤M├┤M├┤M├
        #      ├───┤└╥┘└╥┘└╥┘└╥┘
        # q_1: ┤ H ├─╫──╫──╫──╫─
        #      ├───┤ ║  ║  ║  ║
        # q_2: ┤ H ├─╫──╫──╫──╫─
        #      ├───┤ ║  ║  ║  ║
        # q_3: ┤ H ├─╫──╫──╫──╫─
        #      └───┘ ║  ║  ║  ║
        # c: 4/══════╩══╩══╩══╩═
        #            0  1  2  3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[0], c[1])
        qc.measure(q[0], c[2])
        qc.measure(q[0], c[3])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_measurements3(self):
        """Test circuit depth for measurements #3."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├─────────
        #      ├───┤└╥┘┌─┐
        # q_1: ┤ H ├─╫─┤M├──────
        #      ├───┤ ║ └╥┘┌─┐
        # q_2: ┤ H ├─╫──╫─┤M├───
        #      ├───┤ ║  ║ └╥┘┌─┐
        # q_3: ┤ H ├─╫──╫──╫─┤M├
        #      └───┘ ║  ║  ║ └╥┘
        # c: 4/══════╩══╩══╩══╩═
        #            0  0  0  0
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[0])
        qc.measure(q[2], c[0])
        qc.measure(q[3], c[0])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_barriers1(self):
        """Test circuit depth for barriers #1."""

        #      ┌───┐      ░
        # q_0: ┤ H ├──■───░───────────
        #      └───┘┌─┴─┐ ░
        # q_1: ─────┤ X ├─░───────────
        #           └───┘ ░ ┌───┐
        # q_2: ───────────░─┤ H ├──■──
        #                 ░ └───┘┌─┴─┐
        # q_3: ───────────░──────┤ X ├
        #                 ░      └───┘
        q = QuantumRegister(4, "q")
        c = ClassicalRegister(4, "c")
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.cx(0, 1)
        circ.barrier(q)
        circ.h(2)
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_depth_barriers2(self):
        """Test circuit depth for barriers #2."""

        #      ┌───┐ ░       ░       ░
        # q_0: ┤ H ├─░───■───░───────░──────
        #      └───┘ ░ ┌─┴─┐ ░       ░
        # q_1: ──────░─┤ X ├─░───────░──────
        #            ░ └───┘ ░ ┌───┐ ░
        # q_2: ──────░───────░─┤ H ├─░───■──
        #            ░       ░ └───┘ ░ ┌─┴─┐
        # q_3: ──────░───────░───────░─┤ X ├
        #            ░       ░       ░ └───┘
        q = QuantumRegister(4, "q")
        c = ClassicalRegister(4, "c")
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.barrier(q)
        circ.cx(0, 1)
        circ.barrier(q)
        circ.h(2)
        circ.barrier(q)
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_depth_barriers3(self):
        """Test circuit depth for barriers #3."""

        #      ┌───┐ ░       ░  ░  ░       ░
        # q_0: ┤ H ├─░───■───░──░──░───────░──────
        #      └───┘ ░ ┌─┴─┐ ░  ░  ░       ░
        # q_1: ──────░─┤ X ├─░──░──░───────░──────
        #            ░ └───┘ ░  ░  ░ ┌───┐ ░
        # q_2: ──────░───────░──░──░─┤ H ├─░───■──
        #            ░       ░  ░  ░ └───┘ ░ ┌─┴─┐
        # q_3: ──────░───────░──░──░───────░─┤ X ├
        #            ░       ░  ░  ░       ░ └───┘
        q = QuantumRegister(4, "q")
        c = ClassicalRegister(4, "c")
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.barrier(q)
        circ.cx(0, 1)
        circ.barrier(q)
        circ.barrier(q)
        circ.barrier(q)
        circ.h(2)
        circ.barrier(q)
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_depth_2qubit(self):
        """Test finding depth of two-qubit gates only."""

        #      ┌───┐
        # q_0: ┤ H ├──■───────────────────
        #      └───┘┌─┴─┐┌─────────┐   ┌─┐
        # q_1: ─────┤ X ├┤ Rz(0.1) ├─■─┤M├
        #      ┌───┐└───┘└─────────┘ │ └╥┘
        # q_2: ┤ H ├──■──────────────┼──╫─
        #      └───┘┌─┴─┐            │  ║
        # q_3: ─────┤ X ├────────────■──╫─
        #           └───┘               ║
        # c: 1/═════════════════════════╩═
        #                               0
        circ = QuantumCircuit(4, 1)
        circ.h(0)
        circ.cx(0, 1)
        circ.h(2)
        circ.cx(2, 3)
        circ.rz(0.1, 1)
        circ.cz(1, 3)
        circ.measure(1, 0)
        self.assertEqual(circ.depth(lambda x: x.operation.num_qubits == 2), 2)

    def test_circuit_depth_first_qubit(self):
        """Test finding depth of gates touching q0 only."""

        #      ┌───┐        ┌───┐
        # q_0: ┤ H ├──■─────┤ T ├─────────
        #      └───┘┌─┴─┐┌──┴───┴──┐   ┌─┐
        # q_1: ─────┤ X ├┤ Rz(0.1) ├─■─┤M├
        #      ┌───┐└───┘└─────────┘ │ └╥┘
        # q_2: ┤ H ├──■──────────────┼──╫─
        #      └───┘┌─┴─┐            │  ║
        # q_3: ─────┤ X ├────────────■──╫─
        #           └───┘               ║
        # c: 1/═════════════════════════╩═
        #                               0
        circ = QuantumCircuit(4, 1)
        circ.h(0)
        circ.cx(0, 1)
        circ.t(0)
        circ.h(2)
        circ.cx(2, 3)
        circ.rz(0.1, 1)
        circ.cz(1, 3)
        circ.measure(1, 0)
        self.assertEqual(circ.depth(lambda x: circ.qubits[0] in x.qubits), 3)

    def test_circuit_depth_0_operands(self):
        """Test that the depth can be found even with zero-bit operands."""
        qc = QuantumCircuit(2, 2)
        qc.append(GlobalPhaseGate(0.0), [], [])
        qc.append(GlobalPhaseGate(0.0), [], [])
        qc.append(GlobalPhaseGate(0.0), [], [])
        self.assertEqual(qc.depth(), 0)
        qc.measure([0, 1], [0, 1])
        self.assertEqual(qc.depth(), 1)

    def test_circuit_depth_expr_condition(self):
        """Test that circuit depth respects `Expr` conditions in `IfElseOp`."""
        # Note that the "depth" of control-flow operations is not well defined, so the assertions
        # here are quite weak.  We're mostly aiming to match legacy behaviour of `c_if` for cases
        # where there's a single instruction within the conditional.
        qc = QuantumCircuit(2, 2)
        a = qc.add_input("a", types.Bool())
        with qc.if_test(a):
            qc.x(0)
        with qc.if_test(expr.logic_and(a, qc.clbits[0])):
            qc.x(1)
        self.assertEqual(qc.depth(), 2)
        qc.measure([0, 1], [0, 1])
        self.assertEqual(qc.depth(), 3)

    def test_circuit_depth_expr_store(self):
        """Test that circuit depth respects `Store`."""
        qc = QuantumCircuit(3, 3)
        a = qc.add_input("a", types.Bool())
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        # Note that `Store` is a "directive", so doesn't increase the depth by default, but does
        # cause qubits 0,1; clbits 0,1 and 'a' to all be depth 3 at this point.
        qc.store(a, qc.clbits[0])
        qc.store(a, expr.logic_and(a, qc.clbits[1]))
        # ... so this use of 'a' should make it depth 4.
        with qc.if_test(a):
            qc.x(2)
        self.assertEqual(qc.depth(), 4)

    def test_circuit_depth_switch(self):
        """Test that circuit depth respects the `target` of `SwitchCaseOp`."""
        qc = QuantumCircuit(QuantumRegister(3, "q"), ClassicalRegister(3, "c"))
        a = qc.add_input("a", types.Uint(3))

        with qc.switch(expr.bit_and(a, qc.cregs[0])) as case:
            with case(case.DEFAULT):
                qc.x(0)
        qc.measure(1, 0)
        self.assertEqual(qc.depth(), 2)

    def test_circuit_size_empty(self):
        """Circuit.size should return 0 for an empty circuit."""
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        self.assertEqual(qc.size(), 0)

    def test_circuit_size_single_qubit_gates(self):
        """Circuit.size should increment for each added single qubit gate."""
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        self.assertEqual(qc.size(), 1)
        qc.h(q[1])
        self.assertEqual(qc.size(), 2)

    def test_circuit_size_2qubit(self):
        """Circuit.size of only 2-qubit gates."""
        size = 3
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.cx(q[0], q[1])
        qc.rz(0.1, q[1])
        qc.rzz(0.1, q[1], q[2])
        self.assertEqual(qc.size(lambda x: x.operation.num_qubits == 2), 2)

    def test_circuit_count_ops(self):
        """Test circuit count ops."""
        q = QuantumRegister(6, "q")
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.x(q[1])
        qc.y(q[2:4])
        qc.z(q[3:])
        result = qc.count_ops()

        expected = {"h": 6, "z": 3, "y": 2, "x": 1}

        self.assertIsInstance(result, dict)
        self.assertEqual(expected, result)

    def test_circuit_nonlocal_gates(self):
        """Test num_nonlocal_gates."""

        #      ┌───┐                   ┌────────┐
        # q_0: ┤ H ├───────────────────┤0       ├
        #      ├───┤   ┌───┐           │        │
        # q_1: ┤ H ├───┤ X ├─────────■─┤        ├
        #      ├───┤   └───┘         │ │        │
        # q_2: ┤ H ├─────■───────────X─┤  Iswap ├
        #      ├───┤     │     ┌───┐ │ │        │
        # q_3: ┤ H ├─────┼─────┤ Z ├─X─┤        ├
        #      ├───┤┌────┴────┐├───┤   │        │
        # q_4: ┤ H ├┤ Ry(0.1) ├┤ Z ├───┤1       ├
        #      ├───┤└──┬───┬──┘└───┘   └───╥────┘
        # q_5: ┤ H ├───┤ Z ├───────────────╫─────
        #      └───┘   └───┘            ┌──╨──┐
        # c: 2/═════════════════════════╡ 0x2 ╞══
        #                               └─────┘
        q = QuantumRegister(6, "q")
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q)
        qc.x(q[1])
        qc.cry(0.1, q[2], q[4])
        qc.z(q[3:])
        qc.cswap(q[1], q[2], q[3])
        result = qc.num_nonlocal_gates()
        expected = 2
        self.assertEqual(expected, result)

    def test_circuit_nonlocal_gates_no_instruction(self):
        """Verify num_nunlocal_gates does not include barriers."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/4500
        n = 3
        qc = QuantumCircuit(n)
        qc.h(range(n))

        qc.barrier()

        self.assertEqual(qc.num_nonlocal_gates(), 0)

    def test_circuit_connected_components_empty(self):
        """Verify num_connected_components is width for empty"""
        q = QuantumRegister(7, "q")
        qc = QuantumCircuit(q)
        self.assertEqual(7, qc.num_connected_components())

    def test_circuit_connected_components_multi_reg(self):
        """Test tensor factors works over multi registers"""

        #       ┌───┐
        # q1_0: ┤ H ├──■─────────────────
        #       ├───┤┌─┴─┐
        # q1_1: ┤ H ├┤ X ├──■────────────
        #       ├───┤└───┘  │  ┌───┐
        # q1_2: ┤ H ├───────┼──┤ X ├──■──
        #       ├───┤       │  └─┬─┘┌─┴─┐
        # q2_0: ┤ H ├───────┼────┼──┤ X ├
        #       ├───┤     ┌─┴─┐  │  └───┘
        # q2_1: ┤ H ├─────┤ X ├──■───────
        #       └───┘     └───┘
        q1 = QuantumRegister(3, "q1")
        q2 = QuantumRegister(2, "q2")
        qc = QuantumCircuit(q1, q2)
        qc.h(q1[0])
        qc.h(q1[1])
        qc.h(q1[2])
        qc.h(q2[0])
        qc.h(q2[1])
        qc.cx(q1[0], q1[1])
        qc.cx(q1[1], q2[1])
        qc.cx(q2[1], q1[2])
        qc.cx(q1[2], q2[0])
        self.assertEqual(qc.num_connected_components(), 1)

    def test_circuit_connected_components_multi_reg2(self):
        """Test tensor factors works over multi registers #2."""

        # q1_0: ──■────────────
        #         │
        # q1_1: ──┼─────────■──
        #         │  ┌───┐  │
        # q1_2: ──┼──┤ X ├──┼──
        #         │  └─┬─┘┌─┴─┐
        # q2_0: ──┼────■──┤ X ├
        #       ┌─┴─┐     └───┘
        # q2_1: ┤ X ├──────────
        #       └───┘
        q1 = QuantumRegister(3, "q1")
        q2 = QuantumRegister(2, "q2")
        qc = QuantumCircuit(q1, q2)
        qc.cx(q1[0], q2[1])
        qc.cx(q2[0], q1[2])
        qc.cx(q1[1], q2[0])
        self.assertEqual(qc.num_connected_components(), 2)

    def test_circuit_connected_components_disconnected(self):
        """Test tensor factors works with 2q subspaces."""

        # q1_0: ──■──────────────────────
        #         │
        # q1_1: ──┼────■─────────────────
        #         │    │
        # q1_2: ──┼────┼────■────────────
        #         │    │    │
        # q1_3: ──┼────┼────┼────■───────
        #         │    │    │    │
        # q1_4: ──┼────┼────┼────┼────■──
        #         │    │    │    │  ┌─┴─┐
        # q2_0: ──┼────┼────┼────┼──┤ X ├
        #         │    │    │  ┌─┴─┐└───┘
        # q2_1: ──┼────┼────┼──┤ X ├─────
        #         │    │  ┌─┴─┐└───┘
        # q2_2: ──┼────┼──┤ X ├──────────
        #         │  ┌─┴─┐└───┘
        # q2_3: ──┼──┤ X ├───────────────
        #       ┌─┴─┐└───┘
        # q2_4: ┤ X ├────────────────────
        #       └───┘
        q1 = QuantumRegister(5, "q1")
        q2 = QuantumRegister(5, "q2")
        qc = QuantumCircuit(q1, q2)
        qc.cx(q1[0], q2[4])
        qc.cx(q1[1], q2[3])
        qc.cx(q1[2], q2[2])
        qc.cx(q1[3], q2[1])
        qc.cx(q1[4], q2[0])
        self.assertEqual(qc.num_connected_components(), 5)

    def test_circuit_connected_components_with_clbits(self):
        """Test tensor components with classical register."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├─────────
        #      ├───┤└╥┘┌─┐
        # q_1: ┤ H ├─╫─┤M├──────
        #      ├───┤ ║ └╥┘┌─┐
        # q_2: ┤ H ├─╫──╫─┤M├───
        #      ├───┤ ║  ║ └╥┘┌─┐
        # q_3: ┤ H ├─╫──╫──╫─┤M├
        #      └───┘ ║  ║  ║ └╥┘
        # c: 4/══════╩══╩══╩══╩═
        #           0  1  2  3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.num_connected_components(), 4)

    def test_circuit_unitary_factors1(self):
        """Test unitary factors empty circuit."""
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)
        self.assertEqual(qc.num_unitary_factors(), 4)

    def test_circuit_unitary_factors2(self):
        """Test unitary factors multi qregs"""
        q1 = QuantumRegister(2, "q1")
        q2 = QuantumRegister(2, "q2")
        c = ClassicalRegister(4, "c")
        qc = QuantumCircuit(q1, q2, c)
        self.assertEqual(qc.num_unitary_factors(), 4)

    def test_circuit_unitary_factors3(self):
        """Test unitary factors measurements and conditionals."""

        #      ┌───┐                                      ┌─┐
        # q_0: ┤ H ├────────■──────────■────■──────────■──┤M├───
        #      ├───┤        │          │    │  ┌─┐     │  └╥┘
        # q_1: ┤ H ├──■─────┼─────■────┼────┼──┤M├─────┼───╫────
        #      ├───┤┌─┴─┐   │   ┌─┴─┐  │    │  └╥┘┌─┐  │   ║
        # q_2: ┤ H ├┤ X ├───┼───┤ X ├──┼────┼───╫─┤M├──┼───╫────
        #      ├───┤└───┘ ┌─┴─┐ └───┘┌─┴─┐┌─┴─┐ ║ └╥┘┌─┴─┐ ║ ┌─┐
        # q_3: ┤ H ├──────┤ X ├──────┤ X ├┤ X ├─╫──╫─┤ X ├─╫─┤M├
        #      └───┘      └─╥─┘      └───┘└───┘ ║  ║ └───┘ ║ └╥┘
        #                ┌──╨──┐                ║  ║       ║  ║
        # c: 4/══════════╡ 0x2 ╞════════════════╩══╩═══════╩══╩═
        #                └─────┘                1  2       0  3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.cx(q[1], q[2])
        qc.cx(q[1], q[2])
        with qc.if_test((c, 2)):
            qc.cx(q[0], q[3])
        qc.cx(q[0], q[3])
        qc.cx(q[0], q[3])
        qc.cx(q[0], q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.num_unitary_factors(), 2)

    def test_circuit_unitary_factors4(self):
        """Test unitary factors measurements go to same cbit."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├─────────
        #      ├───┤└╥┘┌─┐
        # q_1: ┤ H ├─╫─┤M├──────
        #      ├───┤ ║ └╥┘┌─┐
        # q_2: ┤ H ├─╫──╫─┤M├───
        #      ├───┤ ║  ║ └╥┘┌─┐
        # q_3: ┤ H ├─╫──╫──╫─┤M├
        #      └───┘ ║  ║  ║ └╥┘
        # q_4: ──────╫──╫──╫──╫─
        #            ║  ║  ║  ║
        # c: 5/══════╩══╩══╩══╩═
        #            0  0  0  0
        size = 5
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[0])
        qc.measure(q[2], c[0])
        qc.measure(q[3], c[0])
        self.assertEqual(qc.num_unitary_factors(), 5)

    def test_num_qubits_qubitless_circuit(self):
        """Check output in absence of qubits."""
        c_reg = ClassicalRegister(3)
        circ = QuantumCircuit(c_reg)
        self.assertEqual(circ.num_qubits, 0)

    def test_num_qubits_qubitfull_circuit(self):
        """Check output in presence of qubits"""
        q_reg = QuantumRegister(4)
        c_reg = ClassicalRegister(3)
        circ = QuantumCircuit(q_reg, c_reg)
        self.assertEqual(circ.num_qubits, 4)

    def test_num_qubits_registerless_circuit(self):
        """Check output for circuits with direct argument for qubits."""
        circ = QuantumCircuit(5)
        self.assertEqual(circ.num_qubits, 5)

    def test_num_qubits_multiple_register_circuit(self):
        """Check output for circuits with multiple quantum registers."""
        q_reg1 = QuantumRegister(5)
        q_reg2 = QuantumRegister(6)
        q_reg3 = QuantumRegister(7)
        circ = QuantumCircuit(q_reg1, q_reg2, q_reg3)
        self.assertEqual(circ.num_qubits, 18)

    def test_metadata_copy_does_not_share_state(self):
        """Verify mutating the metadata of a circuit copy does not impact original."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/6057

        qc1 = QuantumCircuit(1)
        qc1.metadata = {"a": 0}

        qc2 = qc1.copy()
        qc2.metadata["a"] = 1000

        self.assertEqual(qc1.metadata["a"], 0)

    def test_metadata_is_dict(self):
        """Verify setting metadata to None in the constructor results in an empty dict."""
        qc = QuantumCircuit(1)
        metadata1 = qc.metadata
        self.assertEqual(metadata1, {})

    def test_metadata_raises(self):
        """Test that we must set metadata to a dict."""
        qc = QuantumCircuit(1)
        with self.assertRaises(TypeError):
            qc.metadata = 1

    def test_scheduling(self):
        """Test cannot return schedule information without scheduling."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        with self.assertRaises(AttributeError):
            _ = qc.op_start_times


if __name__ == "__main__":
    unittest.main()
