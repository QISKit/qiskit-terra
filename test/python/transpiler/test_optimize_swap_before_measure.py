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

"""Test OptimizeSwapBeforeMeasure pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Qubit
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import OptimizeSwapBeforeMeasure, DAGFixedPoint
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase


class TestOptimizeSwapBeforeMeasure(QiskitTestCase):
    """Test swap-followed-by-measure optimizations."""

    def test_optimize_1swap_1measure(self):
        """Remove a single swap
        qr0:--X--m--       qr0:----
              |  |
        qr1:--X--|--  ==>  qr1:--m-
                 |               |
        cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[1], cr[0])

        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1swap_2measure(self):
        """Remove a single swap affecting two measurements
        qr0:--X--m--         qr0:--m----
              |  |                 |
        qr1:--X--|--m   ==>  qr1:--|--m-
                 |  |              |  |
        cr0:-----.--|--      cr0:--|--.-
        cr1:--------.--      cr1:--.----
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[1], cr[0])
        expected.measure(qr[0], cr[1])

        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_3swap_3measure(self):
        """Remove three swaps affecting three measurements
                ┌─┐         ┌─┐             ┌─┐
        q_0: ─X─┤M├────X──X─┤M├     q_0: ───┤M├───
              │ └╥┘┌─┐ │  │ └╥┘          ┌─┐└╥┘┌─┐
        q_1: ─X──╫─┤M├─┼──┼──╫─     q_1: ┤M├─╫─┤M├
                 ║ └╥┘ │  │  ║  ==>      └╥┘ ║ └╥┘
        q_2: ────╫──╫──X──X──╫─     q_2: ─╫──╫──╫─
                 ║  ║        ║            ║  ║  ║
        c: 2/════╩══╩════════╩═     c: 2/═╩══╩══╩═
                 0  1        0            0  1  0
        """

        qc = QuantumCircuit(3, 2)
        qc.swap(0, 1)
        qc.measure(0, 0)
        qc.swap(0, 2)
        qc.measure(1, 1)
        qc.swap(0, 2)
        qc.measure(0, 0)
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(3, 2)
        expected.measure(1, 0)
        expected.measure(0, 1)
        expected.measure(1, 0)

        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_nswap_nmeasure(self):
        """Remove severals swap affecting multiple measurements
                            ┌─┐                                                   ┌─┐
        q_0: ─X──X─────X────┤M├─────────────────────────────────       q_0: ──────┤M├───────────────
              │  │     │    └╥┘         ┌─┐                                    ┌─┐└╥┘
        q_1: ─X──X──X──X──X──╫─────X────┤M├─────────────────────       q_1: ───┤M├─╫────────────────
                    │     │  ║     │    └╥┘      ┌─┐                        ┌─┐└╥┘ ║
        q_2: ───────X──X──X──╫──X──X─────╫──X────┤M├────────────       q_2: ┤M├─╫──╫────────────────
                       │     ║  │        ║  │    └╥┘┌─┐                     └╥┘ ║  ║    ┌─┐
        q_3: ─X─────X──X─────╫──X──X──X──╫──X─────╫─┤M├─────────       q_3: ─╫──╫──╫────┤M├─────────
              │     │        ║     │  │  ║        ║ └╥┘┌─┐                   ║  ║  ║    └╥┘      ┌─┐
        q_4: ─X──X──X──X─────╫──X──X──X──╫──X─────╫──╫─┤M├──────  ==>  q_4: ─╫──╫──╫─────╫───────┤M├
                 │     │     ║  │        ║  │     ║  ║ └╥┘┌─┐                ║  ║  ║ ┌─┐ ║       └╥┘
        q_5: ────X──X──X──X──╫──X──X─────╫──X──X──╫──╫──╫─┤M├───       q_5: ─╫──╫──╫─┤M├─╫────────╫─
                    │     │  ║     │     ║     │  ║  ║  ║ └╥┘┌─┐             ║  ║  ║ └╥┘ ║ ┌─┐    ║
        q_6: ─X──X──X──X──X──╫──X──X─────╫─────X──╫──╫──╫──╫─┤M├       q_6: ─╫──╫──╫──╫──╫─┤M├────╫─
              │  │     │     ║  │ ┌─┐    ║        ║  ║  ║  ║ └╥┘             ║  ║  ║  ║  ║ └╥┘┌─┐ ║
        q_7: ─X──X─────X─────╫──X─┤M├────╫────────╫──╫──╫──╫──╫─       q_7: ─╫──╫──╫──╫──╫──╫─┤M├─╫─
                             ║    └╥┘    ║        ║  ║  ║  ║  ║              ║  ║  ║  ║  ║  ║ └╥┘ ║
        c: 8/════════════════╩═════╩═════╩════════╩══╩══╩══╩══╩═       c: 8/═╩══╩══╩══╩══╩══╩══╩══╩═
                             0     7     1        2  3  4  5  6              0  1  2  3  4  5  6  7
        """
        circuit = QuantumCircuit(8, 8)
        circuit.swap(3, 4)
        circuit.swap(6, 7)
        circuit.swap(0, 1)
        circuit.swap(6, 7)
        circuit.swap(4, 5)
        circuit.swap(0, 1)
        circuit.swap(5, 6)
        circuit.swap(3, 4)
        circuit.swap(1, 2)
        circuit.swap(6, 7)
        circuit.swap(4, 5)
        circuit.swap(2, 3)
        circuit.swap(0, 1)
        circuit.swap(5, 6)
        circuit.swap(1, 2)
        circuit.swap(6, 7)
        circuit.swap(4, 5)
        circuit.swap(2, 3)
        circuit.swap(3, 4)
        circuit.swap(3, 4)
        circuit.swap(5, 6)
        circuit.swap(1, 2)
        circuit.swap(4, 5)
        circuit.swap(2, 3)
        circuit.swap(5, 6)
        circuit.measure(range(8), range(8))
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(8, 8)
        expected.measure(0, 2)
        expected.measure(1, 1)
        expected.measure(2, 0)
        expected.measure(3, 4)
        expected.measure(4, 7)
        expected.measure(5, 3)
        expected.measure(6, 5)
        expected.measure(7, 6)

        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_nswap_nmeasure_2(self):
        """Remove multiple swaps affecting multiple measurements
                   ┌─┐┌─┐                             ┌─┐┌─┐┌─┐┌─┐┌─┐
        q_0: ─X──X─┤M├┤M├─X──────────────X─      q_0: ┤M├┤M├┤M├┤M├┤M├
              │  │ └╥┘└╥┘ │       ┌─┐    │            └╥┘└╥┘└╥┘└╥┘└╥┘
        q_1: ─X──X──╫──╫──┼─────X─┤M├─X──X─      q_1: ─╫──╫──╫──╫──╫─
                    ║  ║  │ ┌─┐ │ └╥┘ │ ┌─┐  ==>       ║  ║  ║  ║  ║
        q_2: ───────╫──╫──X─┤M├─X──╫──X─┤M├      q_2: ─╫──╫──╫──╫──╫─
                    ║  ║    └╥┘    ║    └╥┘            ║  ║  ║  ║  ║
        c: 1/═══════╩══╩═════╩═════╩═════╩═      c: 1/═╩══╩══╩══╩══╩═
                    0  0     0     0     0             0  0  0  0  0
        """

        qc = QuantumCircuit(3, 1)
        qc.swap(0, 1)
        qc.swap(0, 1)
        qc.measure(0, 0)
        qc.measure(0, 0)
        qc.swap(2, 0)
        qc.measure(2, 0)
        qc.swap(1, 2)
        qc.measure(1, 0)
        qc.swap(1, 2)
        qc.swap(0, 1)
        qc.measure(2, 0)
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(3, 1)
        expected.measure(0, 0)
        expected.measure(0, 0)
        expected.measure(0, 0)
        expected.measure(0, 0)
        expected.measure(0, 0)

        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_cannot_optimize(self):
        """Cannot optimize when swap is not at the end in all of the successors
        qr0:--X-----m--
              |     |
        qr1:--X-[H]-|--
                    |
        cr0:--------.--
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.h(qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        pass_ = OptimizeSwapBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(circuit), after)

    def test_if_else(self):
        """Test that the pass recurses into a simple if-else."""
        pass_ = OptimizeSwapBeforeMeasure()

        base_test = QuantumCircuit(2, 1)
        base_test.swap(0, 1)
        base_test.measure(0, 0)

        base_expected = QuantumCircuit(2, 1)
        base_expected.measure(1, 0)

        test = QuantumCircuit(2, 1)
        test.if_else(
            (test.clbits[0], True), base_test.copy(), base_test.copy(), test.qubits, test.clbits
        )

        expected = QuantumCircuit(2, 1)
        expected.if_else(
            (expected.clbits[0], True),
            base_expected.copy(),
            base_expected.copy(),
            expected.qubits,
            expected.clbits,
        )

        self.assertEqual(pass_(test), expected)

    def test_nested_control_flow(self):
        """Test that the pass recurses into nested control flow."""
        pass_ = OptimizeSwapBeforeMeasure()

        base_test = QuantumCircuit(2, 1)
        base_test.swap(0, 1)
        base_test.measure(0, 0)

        base_expected = QuantumCircuit(2, 1)
        base_expected.measure(1, 0)

        body_test = QuantumCircuit(2, 1)
        body_test.for_loop((0,), None, base_expected.copy(), body_test.qubits, body_test.clbits)

        body_expected = QuantumCircuit(2, 1)
        body_expected.for_loop(
            (0,), None, base_expected.copy(), body_expected.qubits, body_expected.clbits
        )

        test = QuantumCircuit(2, 1)
        test.while_loop((test.clbits[0], True), body_test, test.qubits, test.clbits)

        expected = QuantumCircuit(2, 1)
        expected.while_loop(
            (expected.clbits[0], True), body_expected, expected.qubits, expected.clbits
        )

        self.assertEqual(pass_(test), expected)


class TestOptimizeSwapBeforeMeasureFixedPoint(QiskitTestCase):
    """Test swap-followed-by-measure optimizations in a transpiler, using fixed point."""

    def test_optimize_undone_swap(self):
        """Remove redundant swap
        qr0:--X--X--m--       qr0:--m---
              |  |  |               |
        qr1:--X--X--|--  ==>  qr1:--|--
                    |               |
        cr0:--------.--       cr0:--.--
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_manager = PassManager()
        pass_manager.append(
            DoWhileController(
                [OptimizeSwapBeforeMeasure(), DAGFixedPoint()],
                do_while=lambda property_set: not property_set["dag_fixed_point"],
            )
        )
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)

    def test_optimize_overlap_swap(self):
        """Remove two swaps that overlap
        qr0:--X--------       qr0:--m--
              |                     |
        qr1:--X--X-----       qr1:--|--
                 |       ==>        |
        qr2:-----X--m--       qr2:--|--
                    |               |
        cr0:--------.--       cr0:--.--
        """
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[1], qr[2])
        circuit.measure(qr[2], cr[0])

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_manager = PassManager()
        pass_manager.append(
            DoWhileController(
                [OptimizeSwapBeforeMeasure(), DAGFixedPoint()],
                do_while=lambda property_set: not property_set["dag_fixed_point"],
            )
        )
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)

    def test_optimize_swap_with_anonymous_qubit_and_reg(self):
        """Anonymous bit doesn't mess up the indices
        0: ─────────────       0: ────────
                     ┌─┐               ┌─┐
        q_0: ──────X─┤M├     q_0: ─────┤M├
             ┌───┐ │ └╥┘ ==>      ┌───┐└╥┘
        q_1: ┤ X ├─X──╫─     q_1: ┤ X ├─╫─
             └───┘    ║           └───┘ ║
        c: 1/═════════╩═     c: 1/══════╩═
                      0                 0
        """
        loose_q = Qubit()
        qreg = QuantumRegister(2, "q")
        creg = ClassicalRegister(1, "c")

        # Qubit order: [loose_q, qreg[0], qreg[1]]
        circuit = QuantumCircuit([loose_q], qreg, creg)
        circuit.x(qreg[1])
        circuit.swap(qreg[0], qreg[1])
        circuit.measure(qreg[0], creg[0])

        expected = QuantumCircuit([loose_q], qreg, creg)
        expected.x(qreg[1])
        expected.measure(qreg[1], creg[0])

        after = PassManager([OptimizeSwapBeforeMeasure()]).run(circuit)

        self.assertEqual(expected, after)


if __name__ == "__main__":
    unittest.main()
