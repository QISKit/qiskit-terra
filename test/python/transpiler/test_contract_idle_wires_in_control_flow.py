# This code is part of Qiskit.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.classical import expr, types
from qiskit.transpiler.passes import ContractIdleWiresInControlFlow

from test import QiskitTestCase


class TestContractIdleWiresInControlFlow(QiskitTestCase):
    def test_simple_body(self):
        qc = QuantumCircuit(3, 1)
        with qc.while_loop((qc.clbits[0], False)):
            qc.cx(0, 1)
            qc.noop(2)

        expected = QuantumCircuit(3, 1)
        with expected.while_loop((expected.clbits[0], False)):
            expected.cx(0, 1)

        self.assertEqual(ContractIdleWiresInControlFlow()(qc), expected)

    def test_nothing_to_do(self):
        qc = QuantumCircuit(3, 1)
        with qc.for_loop(range(3)):
            qc.h(0)
            qc.cx(0, 1)
        self.assertEqual(ContractIdleWiresInControlFlow()(qc), qc)

    def test_break_loop(self):
        qc = QuantumCircuit(3, 1)
        with qc.while_loop((qc.clbits[0], False)):
            qc.cx(0, 1)
            qc.noop(2)
            with qc.if_test((0, True)):
                qc.break_loop()

        expected = QuantumCircuit(3, 1)
        with expected.while_loop((expected.clbits[0], False)):
            expected.cx(0, 1)
            # keep the noop since the optimization currently
            # marks all qubits in break_loop's containing block as used
            expected.noop(2)
            with expected.if_test((0, True)):
                expected.break_loop()

        self.assertEqual(ContractIdleWiresInControlFlow()(qc), expected)

    def test_continue_loop(self):
        qc = QuantumCircuit(3, 1)
        with qc.while_loop((qc.clbits[0], False)):
            qc.cx(0, 1)
            qc.noop(2)
            with qc.if_test((0, True)):
                qc.continue_loop()

        expected = QuantumCircuit(3, 1)
        with expected.while_loop((expected.clbits[0], False)):
            expected.cx(0, 1)
            # keep the noop since the optimization currently
            # marks all qubits in continue_loop's containing block as used
            expected.noop(2)
            with expected.if_test((0, True)):
                expected.continue_loop()
        self.assertEqual(ContractIdleWiresInControlFlow()(qc), expected)

    def test_disparate_if_else_left_alone(self):
        qc = QuantumCircuit(3, 1)
        # The true body only uses 0, the false body only uses (1, 2), but because they're part of
        # the shared op, there is no valid contraction here.
        with qc.if_test((qc.clbits[0], True)) as else_:
            qc.h(0)
        with else_:
            qc.cx(1, 2)
        self.assertEqual(ContractIdleWiresInControlFlow()(qc), qc)

    def test_contract_if_else_both_bodies(self):
        qc = QuantumCircuit(3, 1)
        # Explicit idle in the true body only.
        with qc.if_test((qc.clbits[0], True)) as else_:
            qc.h(0)
            qc.cx(0, 2)
            qc.noop(1)
        with else_:
            qc.cz(0, 2)
        # Explicit idle in the false body only.
        with qc.if_test((qc.clbits[0], True)) as else_:
            qc.h(0)
            qc.cx(0, 1)
        with else_:
            qc.cz(0, 1)
            qc.noop(2)
        # Explicit idle in both bodies.
        with qc.if_test((qc.clbits[0], True)) as else_:
            qc.h(1)
            qc.cx(1, 2)
            qc.noop(0)
        with else_:
            qc.cz(1, 2)
            qc.noop(0)

        expected = QuantumCircuit(3, 1)
        with expected.if_test((expected.clbits[0], True)) as else_:
            expected.h(0)
            expected.cx(0, 2)
        with else_:
            expected.cz(0, 2)
        with expected.if_test((expected.clbits[0], True)) as else_:
            expected.h(0)
            expected.cx(0, 1)
        with else_:
            expected.cz(0, 1)
        with expected.if_test((expected.clbits[0], True)) as else_:
            expected.h(1)
            expected.cx(1, 2)
        with else_:
            expected.cz(1, 2)

        self.assertEqual(ContractIdleWiresInControlFlow()(qc), expected)

    def test_recursively_contract(self):
        qc = QuantumCircuit(3, 1)
        with qc.if_test((qc.clbits[0], True)):
            qc.h(0)
            with qc.if_test((qc.clbits[0], True)):
                qc.cx(0, 1)
                qc.noop(2)
        with qc.while_loop((qc.clbits[0], True)):
            with qc.if_test((qc.clbits[0], True)) as else_:
                qc.h(0)
                qc.noop(1, 2)
            with else_:
                qc.cx(0, 1)
                qc.noop(2)

        expected = QuantumCircuit(3, 1)
        with expected.if_test((expected.clbits[0], True)):
            expected.h(0)
            with expected.if_test((expected.clbits[0], True)):
                expected.cx(0, 1)
        with expected.while_loop((expected.clbits[0], True)):
            with expected.if_test((expected.clbits[0], True)) as else_:
                expected.h(0)
            with else_:
                expected.cx(0, 1)

        actual = ContractIdleWiresInControlFlow()(qc)
        self.assertNotEqual(qc, actual)  # Smoke test.
        self.assertEqual(actual, expected)

    def test_handles_vars_in_contraction(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))
        c = expr.Var.new("c", types.Bool())

        qc = QuantumCircuit(3, inputs=[a])
        qc.add_var(b, 5)
        with qc.if_test(a):
            qc.add_var(c, False)
            with qc.if_test(c):
                qc.x(0)
                qc.noop(1, 2)
        with qc.switch(b) as case:
            with case(0):
                qc.x(0)
            with case(1):
                qc.noop(0, 1)
            with case(case.DEFAULT):
                with qc.if_test(a):
                    qc.x(0)
                    qc.noop(1, 2)

        expected = QuantumCircuit(3, inputs=[a])
        expected.add_var(b, 5)
        with expected.if_test(a):
            expected.add_var(c, False)
            with expected.if_test(c):
                expected.x(0)
        with expected.switch(b) as case:
            with case(0):
                expected.x(0)
            with case(1):
                pass
            with case(case.DEFAULT):
                with expected.if_test(a):
                    expected.x(0)

        actual = ContractIdleWiresInControlFlow()(qc)
        self.assertNotEqual(qc, actual)  # Smoke test.
        self.assertEqual(actual, expected)

    def test_handles_registers_in_contraction(self):
        qr = QuantumRegister(3, "q")
        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")

        qc = QuantumCircuit(qr, cr1, cr2)
        with qc.if_test((cr1, 3)):
            with qc.if_test((cr2, 3)):
                qc.noop(0, 1, 2)
        expected = QuantumCircuit(qr, cr1, cr2)
        with expected.if_test((cr1, 3)):
            with expected.if_test((cr2, 3)):
                pass

        actual = ContractIdleWiresInControlFlow()(qc)
        self.assertNotEqual(qc, actual)  # Smoke test.
        self.assertEqual(actual, expected)

    def test_box_is_ignored(self):
        qc = QuantumCircuit(5)
        with qc.box():
            qc.noop(range(5))
        with qc.if_test(expr.lift(True)):
            with qc.box():
                qc.noop(3)
        actual = ContractIdleWiresInControlFlow()(qc.copy())
        self.assertEqual(actual, qc)
