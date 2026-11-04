# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test transpiler pass that cancels inverse gates while exploiting the commutation relations."""

import unittest
from test import QiskitTestCase

import numpy as np
from ddt import data, ddt

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RZGate, UnitaryGate, U2Gate, Initialize
from qiskit.quantum_info import Operator, Clifford
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeInverseCancellation


@ddt
class TestCommutativeInverseCancellation(QiskitTestCase):
    """Test the CommutativeInverseCancellation pass."""

    # The first suite of tests is adapted from CommutativeCancellation,
    # excluding/modifying the tests the combine rotations gates or do
    # basis priority change.

    @data(False, True)
    def test_commutative_circuit1(self, matrix_based):
        """A simple circuit where three CNOTs commute, the first and the last cancel.

        0:----.---------------.--       0:------------
              |               |
        1:---(+)-----(+)-----(+)-   =   1:-------(+)--
                      |                           |
        2:---[H]------.----------       2:---[H]--.---
        """
        circuit = QuantumCircuit(3)
        circuit.cx(0, 1)
        circuit.h(2)
        circuit.cx(2, 1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(3)
        expected.h(2)
        expected.cx(2, 1)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_consecutive_cnots(self, matrix_based):
        """A simple circuit equals identity

        0:----.- ----.--       0:------------
              |      |
        1:---(+)----(+)-   =   1:------------
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_consecutive_cnots2(self, matrix_based):
        """
        Both CNOTs and rotations should cancel out.
        """
        circuit = QuantumCircuit(2)
        circuit.rx(np.pi / 2, 0)
        circuit.cx(0, 1)
        circuit.cx(0, 1)
        circuit.rx(-np.pi / 2, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_2_alternating_cnots(self, matrix_based):
        """A simple circuit where nothing should be cancelled.

        0:----.- ---(+)-       0:----.----(+)-
              |      |               |     |
        1:---(+)-----.--   =   1:---(+)----.--

        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.cx(1, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.cx(1, 0)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_control_bit_of_cnot(self, matrix_based):
        """A simple circuit where nothing should be cancelled.

        0:----.------[X]------.--       0:----.------[X]------.--
              |               |               |               |
        1:---(+)-------------(+)-   =   1:---(+)-------------(+)-
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.x(0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.x(0)
        expected.cx(0, 1)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_control_bit_of_cnot1(self, matrix_based):
        """A simple circuit where the two cnots should be cancelled.

        0:----.------[Z]------.--       0:---[Z]---
              |               |
        1:---(+)-------------(+)-   =   1:---------
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.z(0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.z(0)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_control_bit_of_cnot2(self, matrix_based):
        """A simple circuit where the two cnots should be cancelled.

        0:----.------[T]------.--       0:---[T]---
              |               |
        1:---(+)-------------(+)-   =   1:---------
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.t(0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.t(0)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_control_bit_of_cnot3(self, matrix_based):
        """A simple circuit where the two cnots should be cancelled.

        0:----.------[Rz]------.--       0:---[Rz]---
              |                |
        1:---(+)--------------(+)-   =   1:----------
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.rz(np.pi / 3, 0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.rz(np.pi / 3, 0)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_control_bit_of_cnot4(self, matrix_based):
        """A simple circuit where the two cnots should be cancelled.

        0:----.------[T]------.--       0:---[T]---
              |               |
        1:---(+)-------------(+)-   =   1:---------
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.t(0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.t(0)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_target_bit_of_cnot(self, matrix_based):
        """A simple circuit where nothing should be cancelled.

        0:----.---------------.--       0:----.---------------.--
              |               |               |               |
        1:---(+)-----[Z]-----(+)-   =   1:---(+)----[Z]------(+)-
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.z(1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.z(1)
        expected.cx(0, 1)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_target_bit_of_cnot1(self, matrix_based):
        """A simple circuit where nothing should be cancelled.

        0:----.---------------.--       0:----.---------------.--
              |               |               |               |
        1:---(+)-----[T]-----(+)-   =   1:---(+)----[T]------(+)-
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.t(1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.t(1)
        expected.cx(0, 1)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_target_bit_of_cnot2(self, matrix_based):
        """A simple circuit where nothing should be cancelled.

        0:----.---------------.--       0:----.---------------.--
              |               |               |               |
        1:---(+)-----[Rz]----(+)-   =   1:---(+)----[Rz]-----(+)-
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.rz(np.pi / 3, 1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.rz(np.pi / 3, 1)
        expected.cx(0, 1)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_commutative_circuit2(self, matrix_based):
        """
        A simple circuit where three CNOTs commute, the first and the last cancel,
        also two X gates cancel.
        """

        circuit = QuantumCircuit(3)
        circuit.cx(0, 1)
        circuit.rz(np.pi / 3, 2)
        circuit.cx(2, 1)
        circuit.rz(np.pi / 3, 2)
        circuit.t(2)
        circuit.s(2)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.x(1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(3)
        expected.rz(np.pi / 3, 2)
        expected.cx(2, 1)
        expected.rz(np.pi / 3, 2)
        expected.t(2)
        expected.s(2)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_commutative_circuit3(self, matrix_based):
        """
        A simple circuit where three CNOTs commute, the first and the last cancel,
        also two X gates cancel and two RX gates cancel.
        """

        circuit = QuantumCircuit(4)

        circuit.cx(0, 1)
        circuit.rz(np.pi / 3, 2)
        circuit.rz(np.pi / 3, 3)
        circuit.x(3)
        circuit.cx(2, 3)
        circuit.cx(2, 1)
        circuit.cx(2, 3)
        circuit.rz(-np.pi / 3, 2)
        circuit.x(3)
        circuit.rz(-np.pi / 3, 3)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.x(1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(4)
        expected.cx(2, 1)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_cnot_cascade(self, matrix_based):
        """
        A cascade of CNOTs that equals identity.
        """

        circuit = QuantumCircuit(10)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.cx(3, 4)
        circuit.cx(4, 5)
        circuit.cx(5, 6)
        circuit.cx(6, 7)
        circuit.cx(7, 8)
        circuit.cx(8, 9)

        circuit.cx(8, 9)
        circuit.cx(7, 8)
        circuit.cx(6, 7)
        circuit.cx(5, 6)
        circuit.cx(4, 5)
        circuit.cx(3, 4)
        circuit.cx(2, 3)
        circuit.cx(1, 2)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(10)

        self.assertEqual(expected, new_circuit)

    # The second suite of tests is adapted from InverseCancellation,
    # modifying tests where more nonconsecutive gates cancel.

    @data(False, True)
    def test_basic_self_inverse(self, matrix_based):
        """Test that a single self-inverse gate as input can be cancelled."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.h(0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("h", gates_after)

    @data(False, True)
    def test_odd_number_self_inverse(self, matrix_based):
        """Test that an odd number of self-inverse gates leaves one gate remaining."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.h(0)
        circuit.h(0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("h", gates_after)
        self.assertEqual(gates_after["h"], 1)

    @data(False, True)
    def test_basic_cx_self_inverse(self, matrix_based):
        """Test that a single self-inverse cx gate as input can be cancelled."""
        circuit = QuantumCircuit(2, 2)
        circuit.cx(0, 1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("cx", gates_after)

    @data(False, True)
    def test_basic_gate_inverse(self, matrix_based):
        """Test that a basic pair of gate inverse can be cancelled."""
        circuit = QuantumCircuit(2, 2)
        circuit.rx(np.pi / 4, 0)
        circuit.rx(-np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("rx", gates_after)

    @data(False, True)
    def test_non_inverse_do_not_cancel(self, matrix_based):
        """Test that non-inverse gate pairs do not cancel."""
        circuit = QuantumCircuit(2, 2)
        circuit.rx(np.pi / 4, 0)
        circuit.rx(np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("rx", gates_after)
        self.assertEqual(gates_after["rx"], 2)

    @data(False, True)
    def test_non_consecutive_gates(self, matrix_based):
        """Test that non-consecutive gates cancel as well."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.h(0)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 1)
        circuit.h(0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("cx", gates_after)
        self.assertNotIn("h", gates_after)

    @data(False, True)
    def test_gate_inverse_phase_gate(self, matrix_based):
        """Test that an inverse pair of a PhaseGate can be cancelled."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("p", gates_after)

    @data(False, True)
    def test_self_inverse_on_different_qubits(self, matrix_based):
        """Test that self_inverse gates cancel on the correct qubits."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.h(1)
        circuit.h(0)
        circuit.h(1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("h", gates_after)

    @data(False, True)
    def test_consecutive_self_inverse_h_x_gate(self, matrix_based):
        """Test that consecutive self-inverse gates cancel."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.h(0)
        circuit.h(0)
        circuit.x(0)
        circuit.x(0)
        circuit.h(0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("x", gates_after)
        self.assertNotIn("h", gates_after)

    @data(False, True)
    def test_inverse_with_different_names(self, matrix_based):
        """Test that inverse gates that have different names."""
        circuit = QuantumCircuit(2, 2)
        circuit.t(0)
        circuit.tdg(0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("t", gates_after)
        self.assertNotIn("tdg", gates_after)

    @data(False, True)
    def test_three_alternating_inverse_gates(self, matrix_based):
        """Test that inverse cancellation works correctly for alternating sequences
        of inverse gates of odd-length."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"], 1)

    @data(False, True)
    def test_four_alternating_inverse_gates(self, matrix_based):
        """Test that inverse cancellation works correctly for alternating sequences
        of inverse gates of even-length."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("p", gates_after)

    @data(False, True)
    def test_five_alternating_inverse_gates(self, matrix_based):
        """Test that inverse cancellation works correctly for alternating sequences
        of inverse gates of odd-length."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"], 1)

    @data(False, True)
    def test_sequence_of_inverse_gates_1(self, matrix_based):
        """Test that inverse cancellation works correctly for more general sequences
        of inverse gates. In this test two pairs of inverse gates are supposed to
        cancel out."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"], 1)

    @data(False, True)
    def test_sequence_of_inverse_gates_2(self, matrix_based):
        """Test that inverse cancellation works correctly for more general sequences
        of inverse gates. In this test, in theory three pairs of inverse gates can
        cancel out, but in practice only two pairs are back-to-back."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"] % 2, 1)

    @data(False, True)
    def test_cx_do_not_wrongly_cancel(self, matrix_based):
        """Test that CX(0,1) and CX(1, 0) do not cancel out, when (CX, CX) is passed
        as an inverse pair."""
        circuit = QuantumCircuit(2, 0)
        circuit.cx(0, 1)
        circuit.cx(1, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("cx", gates_after)
        self.assertEqual(gates_after["cx"], 2)

    # A few more tests from issue 8020

    @data(False, True)
    def test_cancel_both_x_and_z(self, matrix_based):
        """Test that Z commutes with control qubit of CX, and X commutes with the target qubit."""
        circuit = QuantumCircuit(2)
        circuit.z(0)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.z(0)
        circuit.x(1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)

        self.assertEqual(expected, new_circuit)

    @data(False, True)
    def test_gates_do_not_wrongly_cancel(self, matrix_based):
        """Test that X gates do not cancel for X-I-H-I-X."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.id(0)
        circuit.h(0)
        circuit.id(0)
        circuit.x(0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(1)
        expected.x(0)
        expected.h(0)
        expected.x(0)

        self.assertEqual(expected, new_circuit)

    # More tests to cover corner-cases: parameterized gates, directives, reset, etc.

    @data(False, True)
    def test_no_cancellation_across_barrier(self, matrix_based):
        """Test that barrier prevents cancellation."""
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.barrier()
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        self.assertEqual(circuit, new_circuit)

    @data(False, True)
    def test_no_cancellation_across_measure(self, matrix_based):
        """Test that barrier prevents cancellation."""
        circuit = QuantumCircuit(2, 1)
        circuit.cx(0, 1)
        circuit.measure(0, 0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        self.assertEqual(circuit, new_circuit)

    @data(False, True)
    def test_no_cancellation_across_reset(self, matrix_based):
        """Test that reset prevents cancellation."""
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.reset(0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)

        self.assertEqual(circuit, new_circuit)

    @data(False, True)
    def test_cancellation_across_parameterized_gates(self, matrix_based):
        """Test that parameterized gates do not prevent cancellation."""
        theta = Parameter("Theta")
        circuit = QuantumCircuit(1)
        circuit.rz(np.pi / 2, 0)
        circuit.rz(theta, 0)
        circuit.rz(-np.pi / 2, 0)

        expected_circuit = QuantumCircuit(1)
        expected_circuit.rz(theta, 0)

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        self.assertEqual(expected_circuit, new_circuit)

    @data(False, True)
    def test_parameterized_gates_do_not_cancel(self, matrix_based):
        """Test that parameterized gates do not cancel.
        This test should be modified when inverse checking
        gets improved to handle parameterized gates.
        """
        gate = RZGate(Parameter("Theta"))

        circuit = QuantumCircuit(1)
        circuit.append(gate, [0])
        circuit.append(gate.inverse(), [0])

        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=matrix_based))
        new_circuit = passmanager.run(circuit)
        self.assertEqual(circuit, new_circuit)

    def test_phase_difference_rz_p(self):
        """Test inverse rz and p gates which differ by a phase."""
        circuit = QuantumCircuit(1)
        circuit.rz(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)

        # the gates should not cancel when matrix_based is False
        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=False))
        new_circuit = passmanager.run(circuit)
        self.assertEqual(circuit, new_circuit)

        # the gates should be canceled when matrix_based is True
        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=True))
        new_circuit = passmanager.run(circuit)
        self.assertEqual(new_circuit.size(), 0)

        # but check that the operators are the same (global phase is correct)
        self.assertEqual(Operator(circuit), Operator(new_circuit))

    def test_phase_difference_custom(self):
        """Test inverse custom gates that differ by a phase."""
        cx_circuit_with_phase = QuantumCircuit(2)
        cx_circuit_with_phase.cx(0, 1)
        cx_circuit_with_phase.global_phase = np.pi / 4

        circuit = QuantumCircuit(2)
        circuit.append(cx_circuit_with_phase.to_gate(), [0, 1])
        circuit.cx(0, 1)

        # the gates should not cancel when matrix_based is False
        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=False))
        new_circuit = passmanager.run(circuit)
        self.assertEqual(circuit, new_circuit)

        # the gates should be canceled when matrix_based is True
        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=True))
        new_circuit = passmanager.run(circuit)
        self.assertEqual(new_circuit.size(), 0)
        self.assertAlmostEqual(new_circuit.global_phase, np.pi / 4)
        self.assertEqual(Operator(circuit), Operator(new_circuit))

    def test_inverse_unitary_gates(self):
        """Test inverse unitary gates that differ by a phase."""
        circuit = QuantumCircuit(2)
        u1 = UnitaryGate([[1, 0], [0, 1]])
        u2 = UnitaryGate([[-1, 0], [0, -1]])
        circuit.append(u1, [0])
        circuit.append(u2, [0])
        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=True))
        new_circuit = passmanager.run(circuit)
        self.assertEqual(new_circuit.size(), 0)
        self.assertAlmostEqual(new_circuit.global_phase, np.pi)
        self.assertEqual(Operator(circuit), Operator(new_circuit))

    def test_inverse_custom_gates(self):
        """Test inverse custom gates."""
        cx_circuit1 = QuantumCircuit(3)
        cx_circuit1.cx(0, 2)

        cx_circuit2 = QuantumCircuit(3)
        cx_circuit2.cx(0, 1)
        cx_circuit2.cx(1, 2)
        cx_circuit2.cx(0, 1)
        cx_circuit2.cx(1, 2)

        circuit = QuantumCircuit(4)
        circuit.append(cx_circuit1.to_gate(), [0, 1, 2])
        circuit.cx(0, 3)
        circuit.append(cx_circuit2.to_gate(), [0, 1, 2])

        # the two custom gates commute through cx(0, 3) and cancel each other
        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=True))
        new_circuit = passmanager.run(circuit)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.cx(0, 3)
        self.assertEqual(new_circuit, expected_circuit)

    def test_max_qubits(self):
        """Test max_qubits argument."""
        cx_circuit1 = QuantumCircuit(3)
        cx_circuit1.cx(0, 2)
        cx_circuit2 = QuantumCircuit(3)

        cx_circuit2.cx(0, 1)
        cx_circuit2.cx(1, 2)
        cx_circuit2.cx(0, 1)
        cx_circuit2.cx(1, 2)

        circuit = QuantumCircuit(4)
        circuit.append(cx_circuit1.to_gate(), [0, 1, 2])
        circuit.cx(0, 3)
        circuit.append(cx_circuit2.to_gate(), [0, 1, 2])

        # the two custom gates commute through cx(0, 3) and cancel each other, but
        # we avoid the check by limiting max_qubits
        passmanager = PassManager(CommutativeInverseCancellation(matrix_based=True, max_qubits=2))
        new_circuit = passmanager.run(circuit)
        self.assertEqual(circuit, new_circuit)

    def test_2q_pauli_rot_with_non_cached(self):
        """Test a cached 2q-Pauli rotation with a non-cached gate.

        Regression test of #13742.
        """
        circuit = QuantumCircuit(2)
        circuit.rxx(np.pi / 2, 1, 0)
        circuit.append(U2Gate(np.pi / 2, -np.pi), [1])

        pm = PassManager(CommutativeInverseCancellation())
        tqc = pm.run(circuit)

        self.assertEqual(tqc.count_ops().get("u2", 0), 1)
        self.assertEqual(tqc.count_ops().get("rxx", 0), 1)

    def test_clifford(self):
        """Test a circuit that contains a Clifford."""
        cliff_circuit = QuantumCircuit(2)
        cliff_circuit.cx(0, 1)
        cliff = Clifford(cliff_circuit)

        circuit = QuantumCircuit(2)
        circuit.s(0)
        circuit.append(cliff, [0, 1])
        circuit.sdg(0)

        pm = PassManager(CommutativeInverseCancellation())
        tqc = pm.run(circuit)

        # The S and Sdg gates should cancel
        self.assertEqual(tqc.count_ops(), {"clifford": 1})

    def test_control_flow(self):
        """Test a circuit that contains a control-flow operation."""

        circuit = QuantumCircuit(2)
        circuit.h(0)

        with circuit.for_loop(range(3)):
            circuit.cx(1, 0)

        pm = PassManager(CommutativeInverseCancellation())
        tqc = pm.run(circuit)

        # The pass should run successfully but not reduce anything
        self.assertEqual(circuit, tqc)

    def test_initialize(self):
        """Test a circuit with Initialize instruction."""
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        initialize = Initialize(desired_vector)

        circuit = QuantumCircuit(2)
        circuit.append(initialize, [0, 1])
        circuit.x(1)

        pm = PassManager(CommutativeInverseCancellation())
        tqc = pm.run(circuit)

        # The pass should run successfully but not reduce anything
        self.assertEqual(circuit, tqc)

    def test_controlled_state_at_zero(self):
        """Regression test of #14974.

        Two gates with not-all-ones control-states were wrongly
        detected to commute, leading to invalid simplification.
        """
        circuit = QuantumCircuit(2)
        circuit.csdg(0, 1, ctrl_state=0)
        circuit.crx(1, 0, 1, ctrl_state=0)
        circuit.cs(0, 1, ctrl_state=0)
        circuit.ry(1, 1)

        pm = PassManager(CommutativeInverseCancellation())
        tqc = pm.run(circuit)

        # The pass should run successfully but not reduce anything
        self.assertEqual(circuit, tqc)


if __name__ == "__main__":
    unittest.main()
