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

"""Test the optimize-1q-gate pass"""

import unittest
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, BasisTranslator
from qiskit.converters import circuit_to_dag
from qiskit.circuit import Parameter, Gate
from qiskit.circuit.library import U1Gate, U2Gate, U3Gate, UGate, PhaseGate
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
from test import QiskitTestCase

from qiskit.circuit.library.standard_gates.equivalence_library import (
    StandardEquivalenceLibrary as std_eqlib,
)


class TestOptimize1qGates(QiskitTestCase):
    """Test for 1q gate optimizations."""

    def test_dont_optimize_id(self):
        """Identity gates are like 'wait' commands.
        They should never be optimized (even without barriers).

        See: https://github.com/Qiskit/qiskit-terra/issues/2373
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.id(qr)
        circuit.id(qr)
        dag = circuit_to_dag(circuit)

        pass_ = Optimize1qGates()
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_optimize_h_gates_pass_manager(self):
        """Transpile: qr:--[H]-[H]-[H]-- == qr:--[u2]--"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.h(qr[0])

        expected = QuantumCircuit(qr)
        expected.append(U2Gate(0, np.pi), [qr[0]])

        passmanager = PassManager()
        passmanager.append(BasisTranslator(std_eqlib, ["u2"]))
        passmanager.append(Optimize1qGates())
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_1q_gates_collapse_identity_equivalent(self):
        """test optimize_1q_gates removes u1(2*pi) rotations.

        See: https://github.com/Qiskit/qiskit-terra/issues/159
        """
        #       ┌───┐┌───┐┌────────┐┌───┐┌─────────┐┌───────┐┌─────────┐┌───┐         ┌─┐»
        # qr_0: ┤ H ├┤ X ├┤ U1(2π) ├┤ X ├┤ U1(π/2) ├┤ U1(π) ├┤ U1(π/2) ├┤ X ├─────────┤M├»
        #       └───┘└─┬─┘└────────┘└─┬─┘└─────────┘└───────┘└─────────┘└─┬─┘┌───────┐└╥┘»
        # qr_1: ───────■──────────────■───────────────────────────────────■──┤ U1(π) ├─╫─»
        #                                                                    └───────┘ ║ »
        # cr: 2/═══════════════════════════════════════════════════════════════════════╩═»
        #                                                                             0 »
        # «
        # «qr_0: ────────────
        # «      ┌───────┐┌─┐
        # «qr_1: ┤ U1(π) ├┤M├
        # «      └───────┘└╥┘
        # «cr: 2/══════════╩═
        # «                1
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[1], qr[0])
        qc.append(U1Gate(2 * np.pi), [qr[0]])
        qc.cx(qr[1], qr[0])
        qc.append(U1Gate(np.pi / 2), [qr[0]])  # these three should combine
        qc.append(U1Gate(np.pi), [qr[0]])  # to identity then
        qc.append(U1Gate(np.pi / 2), [qr[0]])  # optimized away.
        qc.cx(qr[1], qr[0])
        qc.append(U1Gate(np.pi), [qr[1]])
        qc.append(U1Gate(np.pi), [qr[1]])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])

        dag = circuit_to_dag(qc)
        simplified_dag = Optimize1qGates().run(dag)

        num_u1_gates_remaining = len(simplified_dag.named_nodes("u1"))
        self.assertEqual(num_u1_gates_remaining, 0)

    def test_optimize_1q_gates_collapse_identity_equivalent_phase_gate(self):
        """test optimize_1q_gates removes u1(2*pi) rotations.

        See: https://github.com/Qiskit/qiskit-terra/issues/159
        """
        #       ┌───┐┌───┐┌───────┐┌───┐┌────────┐┌──────┐┌────────┐┌───┐        ┌─┐»
        # qr_0: ┤ H ├┤ X ├┤ P(2π) ├┤ X ├┤ P(π/2) ├┤ P(π) ├┤ P(π/2) ├┤ X ├────────┤M├»
        #       └───┘└─┬─┘└───────┘└─┬─┘└────────┘└──────┘└────────┘└─┬─┘┌──────┐└╥┘»
        # qr_1: ───────■─────────────■────────────────────────────────■──┤ P(π) ├─╫─»
        #                                                                └──────┘ ║ »
        # cr: 2/══════════════════════════════════════════════════════════════════╩═»
        #                                                                         0 »
        # «
        # «qr_0: ───────────
        # «      ┌──────┐┌─┐
        # «qr_1: ┤ P(π) ├┤M├
        # «      └──────┘└╥┘
        # «cr: 2/═════════╩═
        # «               1
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[1], qr[0])
        qc.append(PhaseGate(2 * np.pi), [qr[0]])
        qc.cx(qr[1], qr[0])
        qc.append(PhaseGate(np.pi / 2), [qr[0]])  # these three should combine
        qc.append(PhaseGate(np.pi), [qr[0]])  # to identity then
        qc.append(PhaseGate(np.pi / 2), [qr[0]])  # optimized away.
        qc.cx(qr[1], qr[0])
        qc.append(PhaseGate(np.pi), [qr[1]])
        qc.append(PhaseGate(np.pi), [qr[1]])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])

        dag = circuit_to_dag(qc)
        simplified_dag = Optimize1qGates(["p", "u2", "u", "cx", "id"]).run(dag)

        num_u1_gates_remaining = len(simplified_dag.named_nodes("p"))
        self.assertEqual(num_u1_gates_remaining, 0)

    def test_in_the_back(self):
        """Optimizations can be in the back of the circuit.
        See https://github.com/Qiskit/qiskit-terra/issues/2004.

        qr0:--[U1]-[U1]-[H]--    qr0:--[U1]-[H]--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U1Gate(0.3), [qr])
        circuit.append(U1Gate(0.4), [qr])
        circuit.h(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.append(U1Gate(0.7), [qr])
        expected.h(qr)

        pass_ = Optimize1qGates()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_in_the_back_phase_gate(self):
        """Optimizations can be in the back of the circuit.
        See https://github.com/Qiskit/qiskit-terra/issues/2004.

        qr0:--[U1]-[U1]-[H]--    qr0:--[U1]-[H]--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(PhaseGate(0.3), [qr])
        circuit.append(PhaseGate(0.4), [qr])
        circuit.h(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.append(PhaseGate(0.7), [qr])
        expected.h(qr)

        pass_ = Optimize1qGates(["p", "u2", "u", "cx", "id"])
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_single_parameterized_circuit(self):
        """Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter("theta")

        qc.append(U1Gate(0.3), [qr])
        qc.append(U1Gate(0.4), [qr])
        qc.append(U1Gate(theta), [qr])
        qc.append(U1Gate(0.1), [qr])
        qc.append(U1Gate(0.2), [qr])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(qr)
        expected.append(U1Gate(theta + 1.0), [qr])

        after = Optimize1qGates().run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_parameterized_circuits(self):
        """Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter("theta")

        qc.append(U1Gate(0.3), [qr])
        qc.append(U1Gate(0.4), [qr])
        qc.append(U1Gate(theta), [qr])
        qc.append(U1Gate(0.1), [qr])
        qc.append(U1Gate(0.2), [qr])
        qc.append(U1Gate(theta), [qr])
        qc.append(U1Gate(0.3), [qr])
        qc.append(U1Gate(0.2), [qr])

        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(qr)
        expected.append(U1Gate(2 * theta + 1.5), [qr])

        after = Optimize1qGates().run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_parameterized_expressions_in_circuits(self):
        """Expressions of Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter("theta")
        phi = Parameter("phi")

        sum_ = theta + phi
        product_ = theta * phi
        qc.append(U1Gate(0.3), [qr])
        qc.append(U1Gate(0.4), [qr])
        qc.append(U1Gate(theta), [qr])
        qc.append(U1Gate(phi), [qr])
        qc.append(U1Gate(sum_), [qr])
        qc.append(U1Gate(product_), [qr])
        qc.append(U1Gate(0.3), [qr])
        qc.append(U1Gate(0.2), [qr])

        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(qr)
        expected.append(U1Gate(2 * theta + 2 * phi + product_ + 1.2), [qr])

        after = Optimize1qGates().run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_global_phase_u3_on_left(self):
        """Check proper phase accumulation with instruction with no definition."""

        class CustomGate(Gate):
            """Custom u1 gate definition."""

            def __init__(self, lam):
                super().__init__("u1", 1, [lam])

            def _define(self):
                qc = QuantumCircuit(1)
                qc.p(*self.params, 0)
                self.definition = qc

            def _matrix(self):
                return U1Gate(*self.params).to_matrix()

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        u1 = CustomGate(0.1)
        u1.definition.global_phase = np.pi / 2
        qc.append(u1, [0])
        qc.global_phase = np.pi / 3
        qc.append(U3Gate(0.1, 0.2, 0.3), [0])

        dag = circuit_to_dag(qc)
        after = Optimize1qGates().run(dag)
        self.assertAlmostEqual(after.global_phase, 5 * np.pi / 6, 8)

    def test_global_phase_u_on_left(self):
        """Check proper phase accumulation with instruction with no definition."""

        class CustomGate(Gate):
            """Custom u1 gate."""

            def __init__(self, lam):
                super().__init__("u1", 1, [lam])

            def _define(self):
                qc = QuantumCircuit(1)
                qc.p(*self.params, 0)
                self.definition = qc

            def _matrix(self):
                return U1Gate(*self.params).to_matrix()

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        u1 = CustomGate(0.1)
        u1.definition.global_phase = np.pi / 2
        qc.append(u1, [0])
        qc.global_phase = np.pi / 3
        qc.append(UGate(0.1, 0.2, 0.3), [0])

        dag = circuit_to_dag(qc)
        after = Optimize1qGates(["u1", "u2", "u", "cx"]).run(dag)
        self.assertAlmostEqual(after.global_phase, 5 * np.pi / 6, 8)


class TestOptimize1qGatesParamReduction(QiskitTestCase):
    """Test for 1q gate optimizations parameter reduction, reduce n in Un"""

    def test_optimize_u3_to_u2(self):
        """U3(pi/2, pi/3, pi/4) ->  U2(pi/3, pi/4)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(np.pi / 2, np.pi / 3, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U2Gate(np.pi / 3, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates())
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_to_u2_round(self):
        """U3(1.5707963267948961, 1.0471975511965971, 0.7853981633974489) ->  U2(pi/3, pi/4)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(1.5707963267948961, 1.0471975511965971, 0.7853981633974489), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U2Gate(np.pi / 3, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates())
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_to_u1(self):
        """U3(0, 0, pi/4) ->  U1(pi/4)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(0, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U1Gate(np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates())
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_to_phase_gate(self):
        """U3(0, 0, pi/4) ->  U1(pi/4)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(0, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(PhaseGate(np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["p", "u2", "u", "cx", "id"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_to_u1_round(self):
        """U3(1e-16, 1e-16, pi/4) ->  U1(pi/4)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(1e-16, 1e-16, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U1Gate(np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates())
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_to_phase_round(self):
        """U3(1e-16, 1e-16, pi/4) ->  U1(pi/4)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(1e-16, 1e-16, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(PhaseGate(np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["p", "u2", "u", "cx", "id"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)


class TestOptimize1qGatesBasis(QiskitTestCase):
    """Test for 1q gate optimizations parameter reduction with basis"""

    def test_optimize_u3_basis_u3(self):
        """U3(pi/2, pi/3, pi/4) (basis[u3]) ->  U3(pi/2, pi/3, pi/4)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(np.pi / 2, np.pi / 3, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u3"]))
        result = passmanager.run(circuit)

        self.assertEqual(circuit, result)

    def test_optimize_u3_basis_u(self):
        """U3(pi/2, pi/3, pi/4) (basis[u3]) ->  U(pi/2, pi/3, pi/4)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(np.pi / 2, np.pi / 3, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u"]))
        result = passmanager.run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(UGate(np.pi / 2, np.pi / 3, np.pi / 4), [qr[0]])

        self.assertEqual(expected, result)

    def test_optimize_u3_basis_u2(self):
        """U3(pi/2, 0, pi/4) ->  U2(0, pi/4)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(np.pi / 2, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U2Gate(0, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u2"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_basis_u2_with_target(self):
        """U3(pi/2, 0, pi/4) ->  U2(0, pi/4)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(np.pi / 2, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U2Gate(0, np.pi / 4), [qr[0]])

        target = Target(num_qubits=1)
        target.add_instruction(U2Gate(Parameter("theta"), Parameter("phi")))

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(target=target))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u_basis_u(self):
        """U(pi/2, pi/3, pi/4) (basis[u3]) ->  U(pi/2, pi/3, pi/4)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(UGate(np.pi / 2, np.pi / 3, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u"]))
        result = passmanager.run(circuit)

        self.assertEqual(circuit, result)

    def test_optimize_u3_basis_u2_cx(self):
        """U3(pi/2, 0, pi/4) ->  U2(0, pi/4). Basis [u2, cx]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(np.pi / 2, 0, np.pi / 4), [qr[0]])
        circuit.cx(qr[0], qr[1])

        expected = QuantumCircuit(qr)
        expected.append(U2Gate(0, np.pi / 4), [qr[0]])
        expected.cx(qr[0], qr[1])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u2", "cx"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u_basis_u2_cx(self):
        """U(pi/2, 0, pi/4) ->  U2(0, pi/4). Basis [u2, cx]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(UGate(np.pi / 2, 0, np.pi / 4), [qr[0]])
        circuit.cx(qr[0], qr[1])

        expected = QuantumCircuit(qr)
        expected.append(U2Gate(0, np.pi / 4), [qr[0]])
        expected.cx(qr[0], qr[1])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u2", "cx"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u1_basis_u2_u3(self):
        """U1(pi/4) ->  U3(0, 0, pi/4). Basis [u2, u3]."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U1Gate(np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u2", "u3"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u1_basis_u2_u(self):
        """U1(pi/4) ->  U3(0, 0, pi/4). Basis [u2, u3]."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U1Gate(np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(UGate(0, 0, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u2", "u"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u1_basis_u2(self):
        """U1(pi/4) ->  Raises. Basis [u2]"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U1Gate(np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u2"]))
        with self.assertRaises(TranspilerError):
            _ = passmanager.run(circuit)

    def test_optimize_u3_basis_u2_u1(self):
        """U3(pi/2, 0, pi/4) ->  U2(0, pi/4). Basis [u2, u1]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(np.pi / 2, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U2Gate(0, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u2", "u1"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_basis_u2_phase_gate(self):
        """U3(pi/2, 0, pi/4) ->  U2(0, pi/4). Basis [u2, p]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(np.pi / 2, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U2Gate(0, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u2", "p"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_basis_u1(self):
        """U3(0, 0, pi/4) ->  U1(pi/4). Basis [u1]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(0, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U1Gate(np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u1"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_basis_phase_gate(self):
        """U3(0, 0, pi/4) ->  p(pi/4). Basis [p]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(0, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(PhaseGate(np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["p"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u_basis_u1(self):
        """U(0, 0, pi/4) ->  U1(pi/4). Basis [u1]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(UGate(0, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U1Gate(np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["u1"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u_basis_phase_gate(self):
        """U(0, 0, pi/4) ->  p(pi/4). Basis [p]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(UGate(0, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(PhaseGate(np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGates(["p"]))
        result = passmanager.run(circuit)

        self.assertEqual(expected, result)

    def test_optimize_u3_with_parameters(self):
        """Test correct behavior for u3 gates."""
        phi = Parameter("φ")
        alpha = Parameter("α")
        qr = QuantumRegister(1, "qr")

        qc = QuantumCircuit(qr)
        qc.ry(2 * phi, qr[0])
        qc.ry(alpha, qr[0])
        qc.ry(0.1, qr[0])
        qc.ry(0.2, qr[0])
        passmanager = PassManager([BasisTranslator(std_eqlib, ["u3"]), Optimize1qGates()])
        result = passmanager.run(qc)

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(2 * phi, 0, 0), [qr[0]])
        expected.append(U3Gate(alpha, 0, 0), [qr[0]])
        expected.append(U3Gate(0.3, 0, 0), [qr[0]])

        self.assertEqual(expected, result)


if __name__ == "__main__":
    unittest.main()
