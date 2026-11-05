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

"""Test RemoveDiagonalGatesBeforeMeasure pass"""

import unittest
from copy import deepcopy

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import U1Gate, CU1Gate
from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase


class TestRemoveDiagonalGatesBeforeMeasure(QiskitTestCase):
    """Test remove_diagonal_gates_before_measure optimizations."""

    def test_optimize_1rz_1measure(self):
        """Remove a single RZGate
        qr0:-RZ--m--       qr0:--m-
                 |               |
        qr1:-----|--  ==>  qr1:--|-
                 |               |
        cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1phase_1measure(self):
        """Remove a single PhaseGate
        qr0:--P--m--       qr0:--m-
                 |               |
        qr1:-----|--  ==>  qr1:--|-
                 |               |
        cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.p(0.1, qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1z_1measure(self):
        """Remove a single ZGate
        qr0:--Z--m--       qr0:--m-
                 |               |
        qr1:-----|--  ==>  qr1:--|-
                 |               |
        cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.z(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1t_1measure(self):
        """Remove a single TGate
        qr0:--T--m--       qr0:--m-
                 |               |
        qr1:-----|--  ==>  qr1:--|-
                 |               |
        cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.t(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1s_1measure(self):
        """Remove a single SGate
        qr0:--S--m--       qr0:--m-
                 |               |
        qr1:-----|--  ==>  qr1:--|-
                 |               |
        cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.s(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1tdg_1measure(self):
        """Remove a single TdgGate
        qr0:-Tdg-m--       qr0:--m-
                 |               |
        qr1:-----|--  ==>  qr1:--|-
                 |               |
        cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.tdg(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1sdg_1measure(self):
        """Remove a single SdgGate
        qr0:-Sdg--m--       qr0:--m-
                  |               |
        qr1:------|--  ==>  qr1:--|-
                  |               |
        cr0:------.--       cr0:--.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.sdg(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1u1_1measure(self):
        """Remove a single U1Gate
        qr0:--U1-m--       qr0:--m-
                 |               |
        qr1:-----|--  ==>  qr1:--|-
                 |               |
        cr0:-----.--       cr0:--.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(U1Gate(0.1), [qr[0]])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1rz_1z_1measure(self):
        """Remove a single RZ and leave the other Z
        qr0:-RZ--m--       qr0:----m-
                 |                 |
        qr1:--Z--|--  ==>  qr1:--Z-|-
                 |                 |
        cr0:-----.--       cr0:----.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.z(qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.z(qr[1])
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_simple_if_else(self):
        """Test that the pass recurses into an if-else."""
        pass_ = RemoveDiagonalGatesBeforeMeasure()

        base_test = QuantumCircuit(1, 1)
        base_test.z(0)
        base_test.measure(0, 0)

        base_expected = QuantumCircuit(1, 1)
        base_expected.measure(0, 0)

        test = QuantumCircuit(1, 1)
        test.if_else(
            (test.clbits[0], True), base_test.copy(), base_test.copy(), test.qubits, test.clbits
        )

        expected = QuantumCircuit(1, 1)
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
        pass_ = RemoveDiagonalGatesBeforeMeasure()

        base_test = QuantumCircuit(2, 1)
        base_test.cz(0, 1)
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


class TestRemoveDiagonalControlGatesBeforeMeasure(QiskitTestCase):
    """Test remove diagonal control gates before measure."""

    def test_optimize_1cz_2measure(self):
        """Remove a single CZGate
        qr0:--Z--m---       qr0:--m---
              |  |                |
        qr1:--.--|-m-  ==>  qr1:--|-m-
                 | |              | |
        cr0:-----.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.cz(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1cs_2measure(self):
        """Remove a single CSGate
        qr0:-CS--m---       qr0:--m---
              |  |                |
        qr1:--.--|-m-  ==>  qr1:--|-m-
                 | |              | |
        cr0:-----.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.cs(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1csdg_2measure(self):
        """Remove a single CSdgGate
        qr0:-CSdg--m---       qr0:--m---
                |  |                |
        qr1:----.--|-m-  ==>  qr1:--|-m-
                   | |              | |
        cr0:-------.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.csdg(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1crz_2measure(self):
        """Remove a single CRZGate
        qr0:-RZ--m---       qr0:--m---
              |  |                |
        qr1:--.--|-m-  ==>  qr1:--|-m-
                 | |              | |
        cr0:-----.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.crz(0.1, qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1cp_2measure(self):
        """Remove a single CPhaseGate
        qr0:-CP--m---       qr0:--m---
              |  |                |
        qr1:--.--|-m-  ==>  qr1:--|-m-
                 | |              | |
        cr0:-----.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.cp(0.1, qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1cu1_2measure(self):
        """Remove a single CU1Gate
        qr0:-CU1-m---       qr0:--m---
              |  |                |
        qr1:--.--|-m-  ==>  qr1:--|-m-
                 | |              | |
        cr0:-----.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(CU1Gate(0.1), [qr[0], qr[1]])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1rzz_2measure(self):
        """Remove a single RZZGate
        qr0:--.----m---       qr0:--m---
              |zz  |                |
        qr1:--.----|-m-  ==>  qr1:--|-m-
                   | |              | |
        cr0:-------.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.rzz(0.1, qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1ccz_3measure(self):
        """Remove a single CCZGate"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.ccz(qr[0], qr[1], qr[2])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        circuit.measure(qr[2], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])
        expected.measure(qr[2], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


class TestRemoveConsecutiveDiagonalGatesBeforeMeasure(QiskitTestCase):
    """Test removing consecutive diagonal gates before measure in a single run."""

    def test_optimize_1t_1tdg_1measure(self):
        """Remove a TGate and a TdgGate in a single run
        qr0:--T--Tdg--m--       qr0:--m--
                      |               |
        cr0:----------.--  ==>  cr0:--.--
        """
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.t(qr[0])
        circuit.tdg(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1s_1tdg_1measure(self):
        """Remove a SGate and a TdgGate, which are not inverses of each other
        qr0:--S--Tdg--m--       qr0:--m--
                      |               |
        cr0:----------.--  ==>  cr0:--.--
        """
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.s(qr[0])
        circuit.tdg(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_rz_z(self):
        """Remove a RZGate and a ZGate in a single run
        qr0:--RZ-Z--m--       qr0:--m--
                    |               |
        cr0:--------.--       cr0:--.--
        """
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.z(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1rz_1z_1t_1p_1measure(self):
        """Remove a long chain of diagonal gates in a single run
        qr0:-RZ--Z--T--P--m--       qr0:--m--
                          |               |
        cr0:--------------.--  ==>  cr0:--.--
        """
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.z(qr[0])
        circuit.t(qr[0])
        circuit.p(0.2, qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_2cz_2measure(self):
        """Remove two CZGates in a single run
        qr0:--Z--Z--m---       qr0:--m---
              |  |  |                |
        qr1:--.--.--|-m-  ==>  qr1:--|-m-
                    | |              | |
        cr0:--------.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.cz(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1rz_1cz_1t_2measure(self):
        """Remove a chain mixing 1Q and 2Q diagonal gates
        qr0:-RZ--Z-----m---       qr0:--m---
                 |     |                |
        qr1:-----.--T--|-m-  ==>  qr1:--|-m-
                       | |              | |
        cr0:-----------.-.-       cr0:--.-.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.cz(qr[0], qr[1])
        circuit.t(qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1ccz_1cz_3measure(self):
        """Remove a CCZGate followed by a CZGate"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.ccz(qr[0], qr[1], qr[2])
        circuit.cz(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        circuit.measure(qr[2], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])
        expected.measure(qr[2], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_mid_circuit_measure(self):
        """Remove diagonal gates before each of several measures on the same qubit
        qr0:--T--S--m--T--m--       qr0:--m--m--
                    |     |               |  |
        cr0:--------.-----.--  ==>  cr0:--.--.--
        """
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.t(qr[0])
        circuit.s(qr[0])
        circuit.measure(qr[0], cr[0])
        circuit.t(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_qubit_reused_after_measure(self):
        """Remove diagonal gates before a measure even if the qubit is used afterwards
        qr0:-RZ--m--H--T--m--       qr0:--m--H--m--
                 |        |               |     |
        cr0:-----.--------.--  ==>  cr0:--.-----.--
        """
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.measure(qr[0], cr[0])
        circuit.h(qr[0])
        circuit.t(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.h(qr[0])
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1cz_2measure_qubits_reused_after_measure(self):
        """Remove a CZGate before measures even if both qubits are used afterwards
        qr0:--Z--m-----H--.--m---       qr0:--m-----H--.--m---
              |  |        |  |                |        |  |
        qr1:--.--|-m-----(+)-|-m-  ==>  qr1:--|-m-----(+)-|-m-
                 | |         | |              | |         | |
        cr0:-----.-.---------.-.-       cr0:--.-.---------.-.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.cz(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])
        expected.h(qr[0])
        expected.cx(qr[0], qr[1])
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


class TestRemoveDiagonalGatesBeforeMeasureOveroptimizations(QiskitTestCase):
    """Test situations where remove_diagonal_gates_before_measure should not optimize"""

    def test_optimize_1cz_1measure(self):
        """Do not remove a CZGate because measure happens on only one of the wires
        Compare with test_optimize_1cz_2measure.

            qr0:--Z--m---
                  |  |
            qr1:--.--|---
                     |
            cr0:-----.---
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.cz(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = deepcopy(dag)

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(expected, after)

    def test_optimize_1ccz_1measure(self):
        """Do not remove a CCZGate because measure happens on only one of the wires"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.ccz(qr[0], qr[1], qr[2])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = deepcopy(dag)

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(expected, after)

    def test_optimize_1rz_1cz_1measure(self):
        """Do not remove a RZGate followed by a CZGate that cannot be removed
        Compare with test_optimize_1rz_1cz_1t_2measure.

            qr0:-RZ--Z--m---
                     |  |
            qr1:-----.--|---
                        |
            cr0:--------.---
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.cz(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = deepcopy(dag)

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(expected, after)

    def test_optimize_1cz_1t_1measure(self):
        """Do not remove a CZGate because the diagonal gates on qr1 are not measured

        qr0:--Z-----m---
              |     |
        qr1:--.--T--|---
                    |
        cr0:--------.---
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.cz(qr[0], qr[1])
        circuit.t(qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = deepcopy(dag)

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(expected, after)

    def test_optimize_1cz_1h_2measure(self):
        """Do not remove a CZGate because a non-diagonal gate precedes one of the measures

        qr0:--Z-----m---
              |     |
        qr1:--.--H--|-m-
                    | |
        cr0:--------.-.-
        """
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.cz(qr[0], qr[1])
        circuit.h(qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)

        expected = deepcopy(dag)

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(expected, after)

    def test_optimize_1t_1h_1t_1measure(self):
        """Remove only the T gate after the HGate, since H is not diagonal
        qr0:--T--H--T--m--       qr0:--T--H--m--
                       |                     |
        cr0:-----------.--  ==>  cr0:--------.--
        """
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.t(qr[0])
        circuit.h(qr[0])
        circuit.t(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.t(qr[0])
        expected.h(qr[0])
        expected.measure(qr[0], cr[0])

        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


if __name__ == "__main__":
    unittest.main()
