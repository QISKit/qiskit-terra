# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test QuantumCircuit.tensor()."""

import unittest

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit, Parameter
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import Operator
from test import QiskitTestCase


class TestCircuitCompose(QiskitTestCase):
    """Test composition of two circuits."""

    def test_unnamed_tensor(self):
        """Test composing on unnamed circuits."""
        top = QuantumCircuit(1, 1)
        top.x(0)
        top.measure(0, 0)

        bottom = QuantumCircuit(2)
        bottom.y(0)
        bottom.z(1)

        expect = QuantumCircuit(3, 1)
        expect.x(0)
        expect.y(1)
        expect.z(2)
        expect.measure(0, 0)

        with self.subTest("bottom.tensor(top)"):
            self.assertEqual(bottom.tensor(top), expect)

        expect = QuantumCircuit(3, 1)
        expect.y(0)
        expect.z(1)
        expect.x(2)
        expect.measure(2, 0)

        with self.subTest("top.tensor(bottom)"):
            self.assertEqual(top.tensor(bottom), expect)

    def test_mixed(self):
        """Test composing on named and unnamed registers."""
        qr = QuantumRegister(1, "my_qr")
        cr = ClassicalRegister(1, "my_cr")
        top = QuantumCircuit(qr, cr)
        top.x(0)
        top.measure(0, 0)

        bottom = QuantumCircuit(2)
        bottom.y(0)
        bottom.z(1)

        expect = QuantumCircuit(qr, *bottom.qregs, cr)
        expect.x(0)
        expect.y(1)
        expect.z(2)
        expect.measure(0, 0)

        self.assertEqual(bottom.tensor(top), expect)

    def test_named(self):
        """Test composing on named and unnamed registers."""
        qr0 = QuantumRegister(1, "my_qr")
        cr0 = ClassicalRegister(1, "my_cr")
        top = QuantumCircuit(qr0, cr0)
        top.x(0)
        top.measure(0, 0)

        qr1 = QuantumRegister(2, "my_qr")
        bottom = QuantumCircuit(qr1)
        bottom.y(0)
        bottom.z(1)

        with self.subTest("same name raises"):
            with self.assertRaises(CircuitError):
                _ = bottom.tensor(top)

        qr1 = QuantumRegister(2, "other_qr")
        bottom = QuantumCircuit(qr1)
        bottom.y(0)
        bottom.z(1)

        expect = QuantumCircuit(qr0, qr1, cr0)
        expect.x(0)
        expect.y(1)
        expect.z(2)
        expect.measure(0, 0)

        with self.subTest("assert circuit is correct"):
            self.assertEqual(bottom.tensor(top), expect)

    def test_measure_all(self):
        """Test ``tensor`` works if ``measure_all`` is called on both circuits."""
        qr0 = QuantumRegister(1, "my_qr")
        top = QuantumCircuit(qr0)
        top.x(0)
        top.measure_all()

        qr1 = QuantumRegister(2, "other_qr")
        bottom = QuantumCircuit(qr1)
        bottom.y(0)
        bottom.z(1)
        bottom.measure_all()

        cr = ClassicalRegister(3, "meas")
        expect = QuantumCircuit(qr0, qr1, cr)
        expect.x(0)
        expect.y(1)
        expect.z(2)
        expect.barrier(0)  # since barriers have been applied to the subcircuits
        expect.barrier([1, 2])
        expect.measure([0, 1, 2], [0, 1, 2])

        self.assertEqual(bottom.tensor(top), expect)

    def test_multiple_registers(self):
        """Test tensoring circuits with multiple registers."""
        p, q = Parameter("p"), Parameter("q")
        qrs = [QuantumRegister(1) for _ in range(6)]
        crs = [ClassicalRegister(1) for _ in range(2)]

        #     ┌───┐
        # q0: ┤ H ├───────■──────────────
        #     ├───┤┌───┐  │
        # q1: ┤ X ├┤ H ├──■──────────────
        #     └───┘└───┘  │  ┌───────┐┌─┐
        # q2: ────────────┼──┤ Ry(p) ├┤M├
        #               ┌─┴─┐└───┬───┘└╥┘
        # q3: ──────────┤ X ├────■─────╫─
        #               └───┘          ║
        # c0: 1/═══════════════════════╩═
        #                              0
        top = QuantumCircuit(*qrs[:4], crs[0])
        top.h(0)
        top.x(1)
        top.h(1)
        top.ccx(0, 1, 3)
        top.cry(p, 3, 2)
        top.measure(2, 0)

        #     ┌────┐┌──────┐
        # q4: ┤ √X ├┤ P(q) ├──■─────
        #     └────┘└──────┘┌─┴─┐┌─┐
        # q5: ──────────────┤ X ├┤M├
        #                   └───┘└╥┘
        # c1: 1/══════════════════╩═
        #                         0
        bottom = QuantumCircuit(*qrs[4:], crs[1])
        bottom.sx(0)
        bottom.p(q, 0)
        bottom.cx(0, 1)
        bottom.measure(1, 0)

        #     ┌───┐
        # q0: ┤ H ├───────────■──────────────
        #     ├───┤  ┌───┐    │
        # q1: ┤ X ├──┤ H ├────■──────────────
        #     └───┘  └───┘    │  ┌───────┐┌─┐
        # q2: ────────────────┼──┤ Ry(p) ├┤M├
        #                   ┌─┴─┐└───┬───┘└╥┘
        # q3: ──────────────┤ X ├────■─────╫─
        #     ┌────┐┌──────┐└───┘          ║
        # q4: ┤ √X ├┤ P(q) ├──■────────────╫─
        #     └────┘└──────┘┌─┴─┐   ┌─┐    ║
        # q5: ──────────────┤ X ├───┤M├────╫─
        #                   └───┘   └╥┘    ║
        # c0: 1/═════════════════════╬═════╩═
        #                            ║     0
        # c1: 1/═════════════════════╩═══════
        #                            0
        expect = QuantumCircuit(*qrs, *crs)
        expect.h(0)
        expect.x(1)
        expect.h(1)
        expect.ccx(0, 1, 3)
        expect.cry(p, 3, 2)
        expect.measure(2, 0)

        expect.sx(4)
        expect.p(q, 4)
        expect.cx(4, 5)
        expect.measure(5, 1)

        self.assertEqual(bottom.tensor(top), expect)

    def test_consistent_with_quantum_info(self):
        """Test that the ordering is consistent with quantum_info's Operator."""

        with self.subTest(msg="simple, single register example"):
            x = QuantumCircuit(1)
            x.x(0)

            i = QuantumCircuit(1)

            circuit = x.tensor(i)
            operator = Operator(x).tensor(Operator(i))

            self.assertEqual(Operator(circuit), operator)

        with self.subTest(msg="multi register example"):
            qrs = [QuantumRegister(1) for _ in range(6)]

            top = QuantumCircuit(*qrs[:4])
            top.h(0)
            top.x(1)
            top.h(1)
            top.ccx(0, 1, 3)
            top.cry(-1.2, 3, 2)

            bottom = QuantumCircuit(*qrs[4:])
            bottom.sx(0)
            bottom.p(3.01, 0)
            bottom.cx(0, 1)

            circuit = bottom.tensor(top)
            operator = Operator(bottom).tensor(Operator(top))

            self.assertEqual(Operator(circuit), operator)

        with self.subTest(msg="nested circuits"):
            sub = QuantumCircuit(3)
            sub.h(0)
            sub.swap(0, 1)
            sub.cx(1, 2)
            sub.cx(2, 0)

            larger = QuantumCircuit(4)
            larger.h(range(3))
            larger.append(sub.to_instruction(), [3, 2, 1])
            larger.append(sub.control(annotated=False), [0, 1, 2, 3])

            circuit = larger.tensor(larger)
            operator = Operator(larger).tensor(Operator(larger))

            self.assertEqual(Operator(circuit), operator)


if __name__ == "__main__":
    unittest.main()
