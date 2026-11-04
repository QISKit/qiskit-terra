# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the RemoveIdentityEquivalent transpiler pass."""

import ddt
import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister, Gate, ParameterVector
from qiskit.circuit.library import (
    CPhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    XXMinusYYGate,
    XXPlusYYGate,
    GlobalPhaseGate,
    UnitaryGate,
    PauliProductRotationGate,
    PauliEvolutionGate,
)
from qiskit.quantum_info import Operator, Pauli
from qiskit.transpiler.passes import RemoveIdentityEquivalent
from qiskit.transpiler.target import Target, InstructionProperties

from test import QiskitTestCase


@ddt.ddt
class TestRemoveIdentityEquivalent(QiskitTestCase):
    """Test the RemoveIdentityEquivalent pass."""

    def test_remove_identity_equiv_pass(self):
        """Test that negligible gates are dropped."""
        qubits = QuantumRegister(2)
        circuit = QuantumCircuit(qubits)
        a, b = qubits
        circuit.append(CPhaseGate(1e-5), [a, b])
        circuit.append(CPhaseGate(1e-8), [a, b])
        circuit.append(RXGate(1e-5), [a])
        circuit.append(RXGate(1e-8), [a])
        circuit.append(RYGate(1e-5), [a])
        circuit.append(RYGate(1e-8), [a])
        circuit.append(RZGate(1e-5), [a])
        circuit.append(RZGate(1e-8), [a])
        circuit.append(RXXGate(1e-5), [a, b])
        circuit.append(RXXGate(1e-8), [a, b])
        circuit.append(RYYGate(1e-5), [a, b])
        circuit.append(RYYGate(1e-8), [a, b])
        circuit.append(RZZGate(1e-5), [a, b])
        circuit.append(RZZGate(1e-8), [a, b])
        circuit.append(XXPlusYYGate(1e-5, 1e-8), [a, b])
        circuit.append(XXPlusYYGate(1e-8, 1e-8), [a, b])
        circuit.append(XXMinusYYGate(1e-5, 1e-8), [a, b])
        circuit.append(XXMinusYYGate(1e-8, 1e-8), [a, b])
        transpiled = RemoveIdentityEquivalent()(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 2)
        self.assertEqual(transpiled.count_ops()["cp"], 1)
        self.assertEqual(circuit.count_ops()["rx"], 2)
        self.assertEqual(transpiled.count_ops()["rx"], 1)
        self.assertEqual(circuit.count_ops()["ry"], 2)
        self.assertEqual(transpiled.count_ops()["ry"], 1)
        self.assertEqual(circuit.count_ops()["rz"], 2)
        self.assertEqual(transpiled.count_ops()["rz"], 1)
        self.assertEqual(circuit.count_ops()["rxx"], 2)
        self.assertEqual(transpiled.count_ops()["rxx"], 1)
        self.assertEqual(circuit.count_ops()["ryy"], 2)
        self.assertEqual(transpiled.count_ops()["ryy"], 1)
        self.assertEqual(circuit.count_ops()["rzz"], 2)
        self.assertEqual(transpiled.count_ops()["rzz"], 1)
        self.assertEqual(circuit.count_ops()["xx_plus_yy"], 2)
        self.assertEqual(transpiled.count_ops()["xx_plus_yy"], 1)
        self.assertEqual(circuit.count_ops()["xx_minus_yy"], 2)
        self.assertEqual(transpiled.count_ops()["xx_minus_yy"], 1)
        np.testing.assert_allclose(
            np.array(Operator(circuit)), np.array(Operator(transpiled)), atol=1e-7
        )

    def test_handles_parameters(self):
        """Test that gates with parameters are ignored gracefully."""
        qubits = QuantumRegister(2)
        circuit = QuantumCircuit(qubits)
        a, b = qubits
        theta = Parameter("theta")
        circuit.append(CPhaseGate(theta), [a, b])
        circuit.append(CPhaseGate(1e-5), [a, b])
        circuit.append(CPhaseGate(1e-8), [a, b])
        transpiled = RemoveIdentityEquivalent()(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 3)
        self.assertEqual(transpiled.count_ops()["cp"], 2)

    def test_approximation_degree(self):
        """Test that approximation degree handled correctly."""
        qubits = QuantumRegister(2)
        circuit = QuantumCircuit(qubits)
        a, b = qubits
        circuit.append(CPhaseGate(1e-4), [a, b])
        # fidelity 0.9999850001249996 which is above the threshold and not excluded
        # so 1e-2 is the only gate remaining
        circuit.append(CPhaseGate(1e-2), [a, b])
        circuit.append(CPhaseGate(1e-20), [a, b])
        transpiled = RemoveIdentityEquivalent(approximation_degree=0.9999999)(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 3)
        self.assertEqual(transpiled.count_ops()["cp"], 1)
        self.assertEqual(transpiled.data[0].operation.params[0], 1e-2)

    def test_target_approx_none(self):
        """Test error rate with target."""

        target = Target()
        props = {(0, 1): InstructionProperties(error=1e-10)}
        target.add_instruction(CPhaseGate(Parameter("theta")), props)
        circuit = QuantumCircuit(2)
        circuit.append(CPhaseGate(1e-4), [0, 1])
        circuit.append(CPhaseGate(1e-2), [0, 1])
        circuit.append(CPhaseGate(1e-20), [0, 1])
        transpiled = RemoveIdentityEquivalent(approximation_degree=None, target=target)(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 3)
        self.assertEqual(transpiled.count_ops()["cp"], 2)

    def test_target_approx_approx_degree(self):
        """Test error rate with target."""

        target = Target()
        props = {(0, 1): InstructionProperties(error=1e-10)}
        target.add_instruction(CPhaseGate(Parameter("theta")), props)
        circuit = QuantumCircuit(2)
        circuit.append(CPhaseGate(1e-4), [0, 1])
        circuit.append(CPhaseGate(1e-2), [0, 1])
        circuit.append(CPhaseGate(1e-20), [0, 1])
        transpiled = RemoveIdentityEquivalent(approximation_degree=0.9999999, target=target)(
            circuit
        )
        self.assertEqual(circuit.count_ops()["cp"], 3)
        self.assertEqual(transpiled.count_ops()["cp"], 2)

    def test_custom_gate_no_matrix(self):
        """Test that opaque gates are ignored."""

        class CustomOpaqueGate(Gate):
            """Custom opaque gate."""

            def __init__(self):
                super().__init__("opaque", 2, [])

        qc = QuantumCircuit(3)
        qc.append(CustomOpaqueGate(), [0, 1])
        transpiled = RemoveIdentityEquivalent()(qc)
        self.assertEqual(qc, transpiled)

    def test_custom_gate_identity_matrix(self):
        """Test that custom gates with matrix are evaluated."""

        class CustomGate(Gate):
            """Custom gate."""

            def __init__(self):
                super().__init__("custom", 3, [])

            def to_matrix(self):
                return np.eye(8, dtype=complex)

        qc = QuantumCircuit(3)
        qc.append(CustomGate(), [0, 1, 2])
        transpiled = RemoveIdentityEquivalent()(qc)
        expected = QuantumCircuit(3)
        self.assertEqual(expected, transpiled)

    @ddt.data(
        RXGate(0),
        RXGate(2 * np.pi),
        RYGate(0),
        RYGate(2 * np.pi),
        RZGate(0),
        RZGate(2 * np.pi),
        UnitaryGate(np.array([[1, 0], [0, 1]])),
        UnitaryGate(np.array([[-1, 0], [0, -1]])),
        UnitaryGate(np.array([[np.exp(1j * np.pi / 4), 0], [0, np.exp(1j * np.pi / 4)]])),
        GlobalPhaseGate(0),
        GlobalPhaseGate(np.pi / 4),
        UnitaryGate(np.exp(-0.123j) * np.eye(2)),
        UnitaryGate(np.exp(-0.123j) * np.eye(4)),
        UnitaryGate(np.exp(-0.123j) * np.eye(8)),
        PauliProductRotationGate(Pauli("XYIZ"), 0),
    )
    def test_remove_identity_up_to_global_phase(self, gate):
        """Test that gates equivalent to identity up to a global phase are removed from the circuit,
        and the global phase of the circuit is updated correctly.
        """
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)
        transpiled = RemoveIdentityEquivalent()(qc)
        self.assertEqual(transpiled.size(), 0)
        self.assertEqual(Operator(qc), Operator(transpiled))

    def test_parameterized_global_phase_ignored(self):
        """Test that parameterized global phase gates are not removed by the pass."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.append(GlobalPhaseGate(theta), [])
        transpiled = RemoveIdentityEquivalent()(qc)
        self.assertEqual(qc, transpiled)

    def test_no_panic_on_parametervector_phase(self):
        """Regression test for gh-16053."""
        pv = ParameterVector("v", 1)
        qc = QuantumCircuit(1, global_phase=pv[0])
        qc.rz(0.0, 0)
        qc = RemoveIdentityEquivalent()(qc)

        expected = QuantumCircuit(1, global_phase=pv[0])
        self.assertEqual(qc, expected)

    def test_pauli_evo_equals_stdgate(self):
        """Test the Pauli evolution gate is consistent with std gates."""

        def get_rz(angle):
            qc = QuantumCircuit(1)
            qc.rz(angle, 0)
            return qc

        def get_paulievo(angle, n):
            evo = PauliEvolutionGate(Pauli("Z" + (n - 1) * "I"), time=angle / 2)
            qc = QuantumCircuit(evo.num_qubits)
            qc.append(evo, qc.qubits)
            return qc

        pass_ = RemoveIdentityEquivalent()

        # It is important to cover a small enough grid around the threshold where we
        # start removing gates, to ensure all cases are handled consistently. That
        # threshold can be calculated by inverting the avg. gate fidelity with the
        # default tol of 1e-12, coming out at an angle of roughly 3.5e-06.
        for angle in np.linspace(1e-5, 1e-6, num=25):
            # number of expected instructions
            rz = get_rz(angle)
            expected = pass_(rz).count_ops().get("rz", 0)

            # check the number matches for different `n`
            for n in [1, 10]:
                with self.subTest(n=n):
                    evo = get_paulievo(angle, n)
                    num_inst = pass_(evo).count_ops().get("PauliEvolution", 0)
                    self.assertEqual(expected, num_inst)
