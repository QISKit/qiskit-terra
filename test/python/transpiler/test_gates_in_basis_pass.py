# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test GatesInBasis pass."""

from qiskit.circuit import QuantumCircuit, ForLoopOp, IfElseOp, SwitchCaseOp, Clbit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import HGate, CXGate, UGate, XGate, ZGate
from qiskit.circuit.measure import Measure
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import GatesInBasis
from qiskit.transpiler.target import Target
from qiskit.providers.fake_provider import GenericBackendV2
from test import QiskitTestCase


class TestGatesInBasisPass(QiskitTestCase):
    """Tests for GatesInBasis pass."""

    def test_all_gates_in_basis(self):
        """Test circuit with all gates in basis."""
        basis_gates = ["cx", "h"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertTrue(property_set["all_gates_in_basis"])

    def test_all_gates_not_in_basis(self):
        """Test circuit with not all gates in basis."""
        basis_gates = ["cx", "u"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_empty_circuit(self):
        """Test circuit with no gates."""
        basis_gates = ["cx", "u"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates)
        circuit = QuantumCircuit(2)
        analysis_pass(circuit, property_set=property_set)
        self.assertTrue(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_empty_basis(self):
        """Test circuit with gates and empty basis."""
        basis_gates = []
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_after_translation(self):
        """Test circuit with gates in basis after conditional translation."""
        basis_gates = ["cx", "u"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])
        pm = PassManager()
        pm.append(analysis_pass)
        pm.append(
            ConditionalController(
                BasisTranslator(SessionEquivalenceLibrary, basis_gates),
                condition=lambda property_set: not property_set["all_gates_in_basis"],
            )
        )
        pm.append(analysis_pass)
        pm.run(circuit)
        self.assertTrue(pm.property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_with_target(self):
        """Test circuit with all gates in basis with target."""
        target = GenericBackendV2(num_qubits=5, basis_gates=["u", "cx"], seed=42).target
        basis_gates = ["cx", "u"]  # not used
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target=target)
        circuit = QuantumCircuit(2)
        circuit.u(0, 0, 0, 0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertTrue(property_set["all_gates_in_basis"])

    def test_all_gates_not_in_basis_with_target(self):
        """Test circuit with not all gates in basis with target."""
        target = GenericBackendV2(num_qubits=5, basis_gates=["u", "cx"], seed=42).target
        basis_gates = ["cx", "h"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target=target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_not_on_all_qubits_with_target(self):
        """Test circuit with gate in global basis but not local basis."""
        target = GenericBackendV2(num_qubits=5, basis_gates=["u", "cx"], seed=42).target
        basis_gates = ["ecr", "cx", "h"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target=target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ecr(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_empty_circuit_with_target(self):
        """Test circuit with no gates with target."""
        target = GenericBackendV2(num_qubits=5, basis_gates=["u", "cx"], seed=42).target
        basis_gates = ["cx", "u"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target=target)
        circuit = QuantumCircuit(2)
        analysis_pass(circuit, property_set=property_set)
        self.assertTrue(property_set["all_gates_in_basis"])

    def test_all_gates_in_empty_target(self):
        """Test circuit with gates and empty basis with target."""
        target = Target()
        basis_gates = []
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target=target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_ideal_sim_target(self):
        """Test with target that has ideal gates."""
        target = Target(num_qubits=2)
        target.add_instruction(HGate())
        target.add_instruction(CXGate())
        target.add_instruction(Measure())
        property_set = {}
        analysis_pass = GatesInBasis(target=target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertTrue(property_set["all_gates_in_basis"])

    def test_all_gates_not_in_ideal_sim_target(self):
        """Test with target that has ideal gates."""
        target = Target()
        target.add_instruction(HGate())
        target.add_instruction(UGate(0, 0, 0))
        target.add_instruction(Measure())
        property_set = {}
        analysis_pass = GatesInBasis(target=target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])

    def test_all_gates_in_basis_after_translation_with_target(self):
        """Test circuit with gates in basis after conditional translation."""
        target = GenericBackendV2(num_qubits=5, basis_gates=["u", "cx"], seed=42).target
        basis_gates = ["cx", "u"]
        property_set = {}
        analysis_pass = GatesInBasis(basis_gates, target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        analysis_pass(circuit, property_set=property_set)
        self.assertFalse(property_set["all_gates_in_basis"])
        pm = PassManager()
        pm.append(analysis_pass)
        pm.append(
            ConditionalController(
                BasisTranslator(SessionEquivalenceLibrary, basis_gates, target=target),
                condition=lambda property_set: not property_set["all_gates_in_basis"],
            )
        )
        pm.append(analysis_pass)
        pm.run(circuit)
        self.assertTrue(pm.property_set["all_gates_in_basis"])

    def test_basis_gates_control_flow(self):
        """Test that the pass recurses into control flow."""
        circuit = QuantumCircuit(4, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop((1, 2)):
            circuit.cx(0, 1)
            with circuit.if_test((circuit.clbits[0], True)) as else_:
                circuit.x(2)
            with else_:
                circuit.z(3)

        one_missing = {"h", "measure", "for_loop", "cx", "if_else", "x"}
        pass_ = GatesInBasis(one_missing)
        pass_(circuit)
        self.assertFalse(pass_.property_set["all_gates_in_basis"])

        complete = one_missing | {"z"}
        pass_ = GatesInBasis(complete)
        pass_(circuit)
        self.assertTrue(pass_.property_set["all_gates_in_basis"])

    def test_basis_gates_target(self):
        """Test that the pass recurses into control flow."""
        circuit = QuantumCircuit(4, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop((1, 2)):
            circuit.cx(0, 1)
            with circuit.if_test((circuit.clbits[0], True)) as else_:
                circuit.x(2)
            with else_:
                circuit.z(3)

        instructions = [
            HGate(),
            Measure(),
            ForLoopOp((), None, QuantumCircuit(4)),
            CXGate(),
            IfElseOp((Clbit(), True), QuantumCircuit(2), QuantumCircuit(2)),
            SwitchCaseOp(Clbit(), [(False, QuantumCircuit(2)), (True, QuantumCircuit(2))]),
            XGate(),
            ZGate(),
        ]
        one_missing = Target(num_qubits=4)
        for instruction in instructions[:-1]:
            one_missing.add_instruction(instruction, {None: None})
        pass_ = GatesInBasis(target=one_missing)
        pass_(circuit)
        self.assertFalse(pass_.property_set["all_gates_in_basis"])

        complete = Target(num_qubits=4)
        for instruction in instructions:
            complete.add_instruction(instruction, {None: None})
        pass_ = GatesInBasis(target=complete)
        pass_(circuit)
        self.assertTrue(pass_.property_set["all_gates_in_basis"])

    def test_store_is_treated_as_builtin_basis_gates(self):
        """Test that `Store` is treated as an automatic built-in when given basis gates."""
        pass_ = GatesInBasis(basis_gates=["h", "cx"])

        a = expr.Var.new("a", types.Bool())
        good = QuantumCircuit(2, inputs=[a])
        good.store(a, False)
        good.h(0)
        good.cx(0, 1)
        _ = pass_(good)
        self.assertTrue(pass_.property_set["all_gates_in_basis"])

        bad = QuantumCircuit(2, inputs=[a])
        bad.store(a, False)
        bad.x(0)
        bad.cz(0, 1)
        _ = pass_(bad)
        self.assertFalse(pass_.property_set["all_gates_in_basis"])

    def test_store_is_treated_as_builtin_target(self):
        """Test that `Store` is treated as an automatic built-in when given a target."""
        target = Target()
        target.add_instruction(HGate(), {(0,): None, (1,): None})
        target.add_instruction(CXGate(), {(0, 1): None, (1, 0): None})
        pass_ = GatesInBasis(target=target)

        a = expr.Var.new("a", types.Bool())
        good = QuantumCircuit(2, inputs=[a])
        good.store(a, False)
        good.h(0)
        good.cx(0, 1)
        _ = pass_(good)
        self.assertTrue(pass_.property_set["all_gates_in_basis"])

        bad = QuantumCircuit(2, inputs=[a])
        bad.store(a, False)
        bad.x(0)
        bad.cz(0, 1)
        _ = pass_(bad)
        self.assertFalse(pass_.property_set["all_gates_in_basis"])
