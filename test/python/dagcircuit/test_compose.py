# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for the DAGCircuit object"""

import unittest

from qiskit.circuit import (
    QuantumRegister,
    ClassicalRegister,
    QuantumCircuit,
    IfElseOp,
    WhileLoopOp,
    SwitchCaseOp,
    CASE_DEFAULT,
    Store,
)
from qiskit.circuit.classical import expr, types
from qiskit.dagcircuit import DAGCircuit, DAGCircuitError
from qiskit.converters import circuit_to_dag, dag_to_circuit
from test import QiskitTestCase


class TestDagCompose(QiskitTestCase):
    """Test composition of two dags"""

    def setUp(self):
        super().setUp()
        qreg1 = QuantumRegister(3, "lqr_1")
        qreg2 = QuantumRegister(2, "lqr_2")
        creg = ClassicalRegister(2, "lcr")

        self.circuit_left = QuantumCircuit(qreg1, qreg2, creg)
        self.circuit_left.h(qreg1[0])
        self.circuit_left.x(qreg1[1])
        self.circuit_left.p(0.1, qreg1[2])
        self.circuit_left.cx(qreg2[0], qreg2[1])

        self.left_qubit0 = qreg1[0]
        self.left_qubit1 = qreg1[1]
        self.left_qubit2 = qreg1[2]
        self.left_qubit3 = qreg2[0]
        self.left_qubit4 = qreg2[1]
        self.left_clbit0 = creg[0]
        self.left_clbit1 = creg[1]
        self.condition1 = (creg, 1)
        self.condition2 = (creg, 2)

    def test_compose_inorder(self):
        """Composing two dags of the same width, default order.

                      ┌───┐
        lqr_1_0: |0>──┤ H ├───     rqr_0: |0>──■───────
                      ├───┤                    │  ┌───┐
        lqr_1_1: |0>──┤ X ├───     rqr_1: |0>──┼──┤ X ├
                    ┌─┴───┴──┐                 │  ├───┤
        lqr_1_2: |0>┤ P(0.1) ├  +  rqr_2: |0>──┼──┤ Y ├  =
                    └────────┘               ┌─┴─┐└───┘
        lqr_2_0: |0>────■─────     rqr_3: |0>┤ X ├─────
                      ┌─┴─┐                  └───┘┌───┐
        lqr_2_1: |0>──┤ X ├───     rqr_4: |0>─────┤ Z ├
                      └───┘                       └───┘
        lcr_0: 0 ═══════════

        lcr_1: 0 ═══════════


                       ┌───┐
         lqr_1_0: |0>──┤ H ├─────■───────
                       ├───┤     │  ┌───┐
         lqr_1_1: |0>──┤ X ├─────┼──┤ X ├
                     ┌─┴───┴──┐  │  ├───┤
         lqr_1_2: |0>┤ P(0.1) ├──┼──┤ Y ├
                     └────────┘┌─┴─┐└───┘
         lqr_2_0: |0>────■─────┤ X ├─────
                       ┌─┴─┐   └───┘┌───┐
         lqr_2_1: |0>──┤ X ├────────┤ Z ├
                       └───┘        └───┘
         lcr_0: 0 ═══════════════════════

         lcr_1: 0 ═══════════════════════

        """
        qreg = QuantumRegister(5, "rqr")

        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[3])
        circuit_right.x(qreg[1])
        circuit_right.y(qreg[2])
        circuit_right.z(qreg[4])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # default wiring: i <- i
        dag_left.compose(dag_right)
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit0, self.left_qubit3)
        circuit_expected.x(self.left_qubit1)
        circuit_expected.y(self.left_qubit2)
        circuit_expected.z(self.left_qubit4)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_inorder_smaller(self):
        """Composing with a smaller RHS dag, default order.

                      ┌───┐                       ┌─────┐
        lqr_1_0: |0>──┤ H ├───     rqr_0: |0>──■──┤ Tdg ├
                      ├───┤                  ┌─┴─┐└─────┘
        lqr_1_1: |0>──┤ X ├───     rqr_1: |0>┤ X ├───────
                    ┌─┴───┴──┐               └───┘
        lqr_1_2: |0>┤ P(0.1) ├  +                          =
                    └────────┘
        lqr_2_0: |0>────■─────
                      ┌─┴─┐
        lqr_2_1: |0>──┤ X ├───
                      └───┘
        lcr_0: 0 ══════════════

        lcr_1: 0 ══════════════

                       ┌───┐        ┌─────┐
         lqr_1_0: |0>──┤ H ├─────■──┤ Tdg ├
                       ├───┤   ┌─┴─┐└─────┘
         lqr_1_1: |0>──┤ X ├───┤ X ├───────
                     ┌─┴───┴──┐└───┘
         lqr_1_2: |0>┤ P(0.1) ├────────────
                     └────────┘
         lqr_2_0: |0>────■─────────────────
                       ┌─┴─┐
         lqr_2_1: |0>──┤ X ├───────────────
                       └───┘
         lcr_0: 0 ═════════════════════════

         lcr_1: 0 ═════════════════════════

        """
        qreg = QuantumRegister(2, "rqr")

        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # default wiring: i <- i
        dag_left.compose(dag_right)
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit0, self.left_qubit1)
        circuit_expected.tdg(self.left_qubit0)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_permuted(self):
        """Composing two dags of the same width, permuted wires.
                      ┌───┐
        lqr_1_0: |0>──┤ H ├───      rqr_0: |0>──■───────
                      ├───┤                     │  ┌───┐
        lqr_1_1: |0>──┤ X ├───      rqr_1: |0>──┼──┤ X ├
                    ┌─┴───┴──┐                  │  ├───┤
        lqr_1_2: |0>┤ P(0.1) ├      rqr_2: |0>──┼──┤ Y ├
                    └────────┘                ┌─┴─┐└───┘
        lqr_2_0: |0>────■─────  +   rqr_3: |0>┤ X ├─────   =
                      ┌─┴─┐                   └───┘┌───┐
        lqr_2_1: |0>──┤ X ├───      rqr_4: |0>─────┤ Z ├
                      └───┘                        └───┘
        lcr_0: 0 ═════════════

        lcr_1: 0 ═════════════

                      ┌───┐   ┌───┐
        lqr_1_0: |0>──┤ H ├───┤ Z ├
                      ├───┤   ├───┤
        lqr_1_1: |0>──┤ X ├───┤ X ├
                    ┌─┴───┴──┐├───┤
        lqr_1_2: |0>┤ P(0.1) ├┤ Y ├
                    └────────┘└───┘
        lqr_2_0: |0>────■───────■──
                      ┌─┴─┐   ┌─┴─┐
        lqr_2_1: |0>──┤ X ├───┤ X ├
                      └───┘   └───┘
        lcr_0: 0 ══════════════════

        lcr_1: 0 ══════════════════
        """
        qreg = QuantumRegister(5, "rqr")
        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[3])
        circuit_right.x(qreg[1])
        circuit_right.y(qreg[2])
        circuit_right.z(qreg[4])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # permuted wiring
        dag_left.compose(
            dag_right,
            qubits=[
                self.left_qubit3,
                self.left_qubit1,
                self.left_qubit2,
                self.left_qubit4,
                self.left_qubit0,
            ],
        )
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.z(self.left_qubit0)
        circuit_expected.x(self.left_qubit1)
        circuit_expected.y(self.left_qubit2)
        circuit_expected.cx(self.left_qubit3, self.left_qubit4)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_permuted_smaller(self):
        """Composing with a smaller RHS dag, and permuted wires.

                      ┌───┐                       ┌─────┐
        lqr_1_0: |0>──┤ H ├───     rqr_0: |0>──■──┤ Tdg ├
                      ├───┤                  ┌─┴─┐└─────┘
        lqr_1_1: |0>──┤ X ├───     rqr_1: |0>┤ X ├───────
                    ┌─┴───┴──┐               └───┘
        lqr_1_2: |0>┤ P(0.1) ├  +                          =
                    └────────┘
        lqr_2_0: |0>────■─────
                      ┌─┴─┐
        lqr_2_1: |0>──┤ X ├───
                      └───┘
        lcr_0: 0 ═════════════

        lcr_1: 0 ═════════════

                       ┌───┐
         lqr_1_0: |0>──┤ H ├───────────────
                       ├───┤
         lqr_1_1: |0>──┤ X ├───────────────
                     ┌─┴───┴──┐┌───┐
         lqr_1_2: |0>┤ P(0.1) ├┤ X ├───────
                     └────────┘└─┬─┘┌─────┐
         lqr_2_0: |0>────■───────■──┤ Tdg ├
                       ┌─┴─┐        └─────┘
         lqr_2_1: |0>──┤ X ├───────────────
                       └───┘
         lcr_0: 0 ═════════════════════════

         lcr_1: 0 ═════════════════════════
        """
        qreg = QuantumRegister(2, "rqr")
        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # permuted wiring of subset
        dag_left.compose(dag_right, qubits=[self.left_qubit3, self.left_qubit2])
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit3, self.left_qubit2)
        circuit_expected.tdg(self.left_qubit3)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_classical(self):
        """Composing on classical bits.

                      ┌───┐                       ┌─────┐┌─┐
        lqr_1_0: |0>──┤ H ├───     rqr_0: |0>──■──┤ Tdg ├┤M├
                      ├───┤                  ┌─┴─┐└─┬─┬─┘└╥┘
        lqr_1_1: |0>──┤ X ├───     rqr_1: |0>┤ X ├──┤M├───╫─
                    ┌─┴───┴──┐               └───┘  └╥┘   ║
        lqr_1_2: |0>┤ P(0.1) ├  +   rcr_0: 0 ════════╬════╩═  =
                    └────────┘                       ║
        lqr_2_0: |0>────■─────      rcr_1: 0 ════════╩══════
                      ┌─┴─┐
        lqr_2_1: |0>──┤ X ├───
                      └───┘
        lcr_0: 0 ═════════════

        lcr_1: 0 ═════════════

                      ┌───┐
        lqr_1_0: |0>──┤ H ├──────────────────
                      ├───┤        ┌─────┐┌─┐
        lqr_1_1: |0>──┤ X ├─────■──┤ Tdg ├┤M├
                    ┌─┴───┴──┐  │  └─────┘└╥┘
        lqr_1_2: |0>┤ P(0.1) ├──┼──────────╫─
                    └────────┘  │          ║
        lqr_2_0: |0>────■───────┼──────────╫─
                      ┌─┴─┐   ┌─┴─┐  ┌─┐   ║
        lqr_2_1: |0>──┤ X ├───┤ X ├──┤M├───╫─
                      └───┘   └───┘  └╥┘   ║
           lcr_0: 0 ══════════════════╩════╬═
                                           ║
           lcr_1: 0 ═══════════════════════╩═
        """
        qreg = QuantumRegister(2, "rqr")
        creg = ClassicalRegister(2, "rcr")
        circuit_right = QuantumCircuit(qreg, creg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])
        circuit_right.measure(qreg, creg)

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # permuted subset of qubits and clbits
        dag_left.compose(
            dag_right,
            qubits=[self.left_qubit1, self.left_qubit4],
            clbits=[self.left_clbit1, self.left_clbit0],
        )
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit1, self.left_qubit4)
        circuit_expected.tdg(self.left_qubit1)
        circuit_expected.measure(self.left_qubit4, self.left_clbit0)
        circuit_expected.measure(self.left_qubit1, self.left_clbit1)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_expr_condition(self):
        """Test that compose correctly maps clbits and registers in expression conditions."""
        inner = QuantumCircuit(1)
        inner.x(0)
        qr_src = QuantumRegister(1)
        a_src = ClassicalRegister(2, "a_src")
        b_src = ClassicalRegister(2, "b_src")
        source = DAGCircuit()
        source.add_qreg(qr_src)
        source.add_creg(a_src)
        source.add_creg(b_src)

        test_1 = lambda: expr.lift(a_src[0])
        test_2 = lambda: expr.logic_not(b_src[1])
        test_3 = lambda: expr.cast(expr.bit_and(b_src, 2), types.Bool())
        node_1 = source.apply_operation_back(IfElseOp(test_1(), inner.copy(), None), qr_src, [])
        node_2 = source.apply_operation_back(
            IfElseOp(test_2(), inner.copy(), inner.copy()), qr_src, []
        )
        node_3 = source.apply_operation_back(WhileLoopOp(test_3(), inner.copy()), qr_src, [])

        qr_dest = QuantumRegister(1)
        a_dest = ClassicalRegister(2, "a_dest")
        b_dest = ClassicalRegister(2, "b_dest")
        dest = DAGCircuit()
        dest.add_qreg(qr_dest)
        dest.add_creg(a_dest)
        dest.add_creg(b_dest)

        dest.compose(source)

        # Check that the input conditions weren't mutated.
        for in_condition, node in zip((test_1, test_2, test_3), (node_1, node_2, node_3)):
            self.assertEqual(in_condition(), node.op.condition)

        expected = QuantumCircuit(qr_dest, a_dest, b_dest)
        expected.if_test(expr.lift(a_dest[0]), inner.copy(), [0], [])
        expected.if_else(expr.logic_not(b_dest[1]), inner.copy(), inner.copy(), [0], [])
        expected.while_loop(expr.cast(expr.bit_and(b_dest, 2), types.Bool()), inner.copy(), [0], [])
        self.assertEqual(dest, circuit_to_dag(expected))

    def test_compose_expr_target(self):
        """Test that compose correctly maps clbits and registers in expression targets."""
        inner1 = QuantumCircuit(1)
        inner1.x(0)
        inner2 = QuantumCircuit(1)
        inner2.z(0)

        qr_src = QuantumRegister(1)
        a_src = ClassicalRegister(2, "a_src")
        b_src = ClassicalRegister(2, "b_src")
        source = DAGCircuit()
        source.add_qreg(qr_src)
        source.add_creg(a_src)
        source.add_creg(b_src)

        test_1 = lambda: expr.lift(a_src[0])
        test_2 = lambda: expr.logic_not(b_src[1])
        test_3 = lambda: expr.lift(b_src)
        test_4 = lambda: expr.bit_and(b_src, 2)
        node_1 = source.apply_operation_back(
            SwitchCaseOp(test_1(), [(False, inner1.copy()), (True, inner2.copy())]), qr_src, []
        )
        node_2 = source.apply_operation_back(
            SwitchCaseOp(test_2(), [(False, inner1.copy()), (True, inner2.copy())]), qr_src, []
        )
        node_3 = source.apply_operation_back(
            SwitchCaseOp(test_3(), [(0, inner1.copy()), (CASE_DEFAULT, inner2.copy())]), qr_src, []
        )
        node_4 = source.apply_operation_back(
            SwitchCaseOp(test_4(), [(0, inner1.copy()), (CASE_DEFAULT, inner2.copy())]), qr_src, []
        )

        qr_dest = QuantumRegister(1)
        a_dest = ClassicalRegister(2, "a_dest")
        b_dest = ClassicalRegister(2, "b_dest")
        dest = DAGCircuit()
        dest.add_qreg(qr_dest)
        dest.add_creg(a_dest)
        dest.add_creg(b_dest)
        dest.compose(source)

        # Check that the input expressions weren't mutated.
        for in_target, node in zip(
            (test_1, test_2, test_3, test_4), (node_1, node_2, node_3, node_4)
        ):
            self.assertEqual(in_target(), node.op.target)

        expected = QuantumCircuit(qr_dest, a_dest, b_dest)
        expected.switch(
            expr.lift(a_dest[0]), [(False, inner1.copy()), (True, inner2.copy())], [0], []
        )
        expected.switch(
            expr.logic_not(b_dest[1]), [(False, inner1.copy()), (True, inner2.copy())], [0], []
        )
        expected.switch(
            expr.lift(b_dest), [(0, inner1.copy()), (CASE_DEFAULT, inner2.copy())], [0], []
        )
        expected.switch(
            expr.bit_and(b_dest, 2),
            [(0, inner1.copy()), (CASE_DEFAULT, inner2.copy())],
            [0],
            [],
        )

        self.assertEqual(dest, circuit_to_dag(expected))

    def test_join_unrelated_dags(self):
        """This isn't expected to be common, but should work anyway."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        c = expr.Var.new("c", types.Uint(8))

        dest = DAGCircuit()
        dest.add_input_var(a)
        dest.apply_operation_back(Store(a, expr.lift(False)), (), ())
        source = DAGCircuit()
        source.add_declared_var(b)
        source.add_input_var(c)
        source.apply_operation_back(Store(b, expr.lift(True)), (), ())
        dest.compose(source)

        expected = DAGCircuit()
        expected.add_input_var(a)
        expected.add_declared_var(b)
        expected.add_input_var(c)
        expected.apply_operation_back(Store(a, expr.lift(False)), (), ())
        expected.apply_operation_back(Store(b, expr.lift(True)), (), ())

        self.assertEqual(dest, expected)

    def test_join_unrelated_dags_captures(self):
        """This isn't expected to be common, but should work anyway."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        c = expr.Var.new("c", types.Uint(8))

        dest = DAGCircuit()
        dest.add_captured_var(a)
        dest.apply_operation_back(Store(a, expr.lift(False)), (), ())
        source = DAGCircuit()
        source.add_declared_var(b)
        source.add_captured_var(c)
        source.apply_operation_back(Store(b, expr.lift(True)), (), ())
        dest.compose(source, inline_captures=False)

        expected = DAGCircuit()
        expected.add_captured_var(a)
        expected.add_declared_var(b)
        expected.add_captured_var(c)
        expected.apply_operation_back(Store(a, expr.lift(False)), (), ())
        expected.apply_operation_back(Store(b, expr.lift(True)), (), ())

        self.assertEqual(dest, expected)

    def test_inline_capture_var(self):
        """Should be able to append uses onto another DAG."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())

        dest = DAGCircuit()
        dest.add_input_var(a)
        dest.add_input_var(b)
        dest.apply_operation_back(Store(a, expr.lift(False)), (), ())
        source = DAGCircuit()
        source.add_captured_var(b)
        source.apply_operation_back(Store(b, expr.lift(True)), (), ())
        dest.compose(source, inline_captures=True)

        expected = DAGCircuit()
        expected.add_input_var(a)
        expected.add_input_var(b)
        expected.apply_operation_back(Store(a, expr.lift(False)), (), ())
        expected.apply_operation_back(Store(b, expr.lift(True)), (), ())

        self.assertEqual(dest, expected)

    def test_reject_inline_to_nonexistent_var(self):
        """Should not be able to inline a variable that doesn't exist."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())

        dest = DAGCircuit()
        dest.add_input_var(a)
        dest.apply_operation_back(Store(a, expr.lift(False)), (), ())
        source = DAGCircuit()
        source.add_captured_var(b)
        with self.assertRaisesRegex(
            DAGCircuitError, "Variable '.*' to be inlined is not in the base DAG"
        ):
            dest.compose(source, inline_captures=True)

    def test_rejects_rhs_with_too_many_qubits(self):
        """Test that compose rejects a DAG with too many qubits."""
        dest = circuit_to_dag(QuantumCircuit(1))
        other = circuit_to_dag(QuantumCircuit(2))

        with self.assertRaisesRegex(
            DAGCircuitError,
            r"Cannot compose onto a DAGCircuit with fewer qubits \(2 > 1\)\.",
        ):
            dest.compose(other)

    def test_rejects_rhs_with_too_many_clbits(self):
        """Test that compose rejects a DAG with too many classical bits."""
        dest = circuit_to_dag(QuantumCircuit(1, 1))
        other = circuit_to_dag(QuantumCircuit(1, 2))

        with self.assertRaisesRegex(
            DAGCircuitError,
            r"Cannot compose onto a DAGCircuit with fewer classical bits \(2 > 1\)\.",
        ):
            dest.compose(other)


if __name__ == "__main__":
    unittest.main()
