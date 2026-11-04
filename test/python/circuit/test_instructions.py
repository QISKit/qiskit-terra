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


"""Test Qiskit's Instruction class."""

import unittest.mock
import numpy as np

from qiskit.circuit import (
    Gate,
    Parameter,
    Instruction,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
)
from qiskit.circuit.library import HGate, RZGate, CXGate, SGate, SdgGate, TGate
from qiskit.circuit.annotated_operation import AnnotatedOperation
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.random import random_circuit
from test import QiskitTestCase


class TestInstructions(QiskitTestCase):
    """Instructions tests."""

    def test_instructions_equal(self):
        """Test equality of two instructions."""
        hop1 = Instruction("h", 1, 0, [])
        hop2 = Instruction("s", 1, 0, [])
        hop3 = Instruction("h", 1, 0, [])

        self.assertFalse(hop1 == hop2)
        self.assertTrue(hop1 == hop3)

        uop1 = Instruction("u", 1, 0, [0.4, 0.5, 0.5])
        uop2 = Instruction("u", 1, 0, [0.4, 0.6, 0.5])
        uop3 = Instruction("v", 1, 0, [0.4, 0.5, 0.5])
        uop4 = Instruction("u", 1, 0, [0.4, 0.5, 0.5])

        self.assertFalse(uop1 == uop2)
        self.assertTrue(uop1 == uop4)
        self.assertFalse(uop1 == uop3)

        self.assertTrue(HGate() == HGate())
        self.assertFalse(HGate() == CXGate())
        self.assertFalse(hop1 == HGate())

        eop1 = Instruction("kraus", 1, 0, [np.array([[1, 0], [0, 1]])])
        eop2 = Instruction("kraus", 1, 0, [np.array([[0, 1], [1, 0]])])
        eop3 = Instruction("kraus", 1, 0, [np.array([[1, 0], [0, 1]])])
        eop4 = Instruction("kraus", 1, 0, [np.eye(4)])

        self.assertTrue(eop1 == eop3)
        self.assertFalse(eop1 == eop2)
        self.assertFalse(eop1 == eop4)

    def test_instructions_equal_with_parameters(self):
        """Test equality of instructions for cases with Parameters."""
        theta = Parameter("theta")
        phi = Parameter("phi")

        # Verify we can check params including parameters
        self.assertEqual(
            Instruction("u", 1, 0, [theta, phi, 0.4]), Instruction("u", 1, 0, [theta, phi, 0.4])
        )

        # Verify we can test for correct parameter order
        self.assertNotEqual(
            Instruction("u", 1, 0, [theta, phi, 0]), Instruction("u", 1, 0, [phi, theta, 0])
        )

        # Verify we can still find a wrong fixed param if we use parameters
        self.assertNotEqual(
            Instruction("u", 1, 0, [theta, phi, 0.4]), Instruction("u", 1, 0, [theta, phi, 0.5])
        )

        # Verify we can find cases when param != float
        self.assertNotEqual(
            Instruction("u", 1, 0, [0.3, phi, 0.4]), Instruction("u", 1, 0, [theta, phi, 0.5])
        )

    def test_instructions_soft_compare(self):
        """Test soft comparison between instructions."""
        theta = Parameter("theta")
        phi = Parameter("phi")

        # Verify that we are insensitive when there are parameters.
        self.assertTrue(
            Instruction("u", 1, 0, [0.3, phi, 0.4]).soft_compare(
                Instruction("u", 1, 0, [theta, phi, 0.4])
            )
        )

        # Verify that normal equality still holds.
        self.assertTrue(
            Instruction("u", 1, 0, [0.4, 0.5]).soft_compare(Instruction("u", 1, 0, [0.4, 0.5]))
        )

        # Test that when names differ we get False.
        self.assertFalse(
            Instruction("u", 1, 0, [0.4, phi]).soft_compare(Instruction("v", 1, 0, [theta, phi]))
        )

        # Test that when names are the same but number of qubits differ we get False
        self.assertFalse(Instruction("u", 1, 0, []).soft_compare(Instruction("u", 2, 0, [])))

        # Test that when names are the same but number of clbits differ we get False
        self.assertFalse(Instruction("u", 1, 0, []).soft_compare(Instruction("u", 1, 1, [])))

        # Test cutoff precision.
        self.assertFalse(
            Instruction("v", 1, 0, [0.401, phi]).soft_compare(Instruction("v", 1, 0, [0.4, phi]))
        )

        # Test cutoff precision.
        self.assertTrue(
            Instruction("v", 1, 0, [0.4 + 1.0e-20, phi]).soft_compare(
                Instruction("v", 1, 0, [0.4, phi])
            )
        )

    def test_instructions_equal_with_parameter_expressions(self):
        """Test equality of instructions for cases with ParameterExpressions."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_ = theta + phi
        product_ = theta * phi

        # Verify we can check params including parameters
        self.assertEqual(
            Instruction("u", 1, 0, [sum_, product_, 0.4]),
            Instruction("u", 1, 0, [sum_, product_, 0.4]),
        )

        # Verify we can test for correct parameter order
        self.assertNotEqual(
            Instruction("u", 1, 0, [product_, sum_, 0]), Instruction("u", 1, 0, [sum_, product_, 0])
        )

        # Verify we can still find a wrong fixed param if we use parameters
        self.assertNotEqual(
            Instruction("u", 1, 0, [sum_, phi, 0.4]), Instruction("u", 1, 0, [sum_, phi, 0.5])
        )

        # Verify we can find cases when param != float
        self.assertNotEqual(
            Instruction("u", 1, 0, [0.3, sum_, 0.4]), Instruction("u", 1, 0, [product_, sum_, 0.5])
        )

    def circuit_instruction_circuit_roundtrip(self):
        """test converting between circuit and instruction and back
        preserves the circuit"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ1 = QuantumCircuit(q, c, name="circuit1")
        circ1.h(q[0])
        circ1.crz(0.1, q[0], q[1])
        circ1.id(q[1])
        circ1.u(0.1, 0.2, -0.2, q[0])
        circ1.barrier()
        circ1.measure(q, c)
        circ1.rz(0.8, q[0])
        inst = circ1.to_instruction()

        circ2 = QuantumCircuit(q, c, name="circ2")
        circ2.append(inst, q[:])

        self.assertEqual(circ1, circ2)

    def test_append_opaque_wrong_dimension(self):
        """test appending opaque gate to wrong dimension wires."""
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        opaque_gate = Gate(name="crz_2", num_qubits=2, params=[0.5])
        self.assertRaises(CircuitError, circ.append, opaque_gate, [qr[0]])

    def test_opaque_gate(self):
        """test opaque gate functionality"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name="circ")
        opaque_gate = Gate(name="crz_2", num_qubits=2, params=[0.5])
        circ.append(opaque_gate, [q[2], q[0]])
        self.assertEqual(circ.data[0].operation.name, "crz_2")
        self.assertEqual(circ.decompose(), circ)

    def test_opaque_instruction(self):
        """test opaque instruction does not decompose"""
        q = QuantumRegister(4)
        c = ClassicalRegister(2)
        circ = QuantumCircuit(q, c)
        opaque_inst = Instruction(name="my_inst", num_qubits=3, num_clbits=1, params=[0.5])
        circ.append(opaque_inst, [q[3], q[1], q[0]], [c[1]])
        self.assertEqual(circ.data[0].operation.name, "my_inst")
        self.assertEqual(circ.decompose(), circ)

    def test_reverse_gate(self):
        """test reversing a composite gate"""
        q = QuantumRegister(4)
        circ = QuantumCircuit(q, name="circ")
        circ.h(q[0])
        circ.crz(0.1, q[0], q[1])
        circ.id(q[1])
        circ.u(0.1, 0.2, -0.2, q[0])
        gate = circ.to_gate()

        circ = QuantumCircuit(q, name="circ")
        circ.u(0.1, 0.2, -0.2, q[0])
        circ.id(q[1])
        circ.crz(0.1, q[0], q[1])
        circ.h(q[0])
        gate_reverse = circ.to_gate()
        self.assertEqual(gate.reverse_ops().definition, gate_reverse.definition)

    def test_reverse_instruction(self):
        """test reverseing an instruction with conditionals"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name="circ")
        circ.t(q[1])
        circ.u(0.1, 0.2, -0.2, q[0])
        circ.barrier()
        circ.measure(q[0], c[0])
        circ.rz(0.8, q[0])
        inst = circ.to_instruction()

        circ = QuantumCircuit(q, c, name="circ")
        circ.rz(0.8, q[0])
        circ.measure(q[0], c[0])
        circ.barrier()
        circ.u(0.1, 0.2, -0.2, q[0])
        circ.t(q[1])
        inst_reverse = circ.to_instruction()

        self.assertEqual(inst.reverse_ops().definition, inst_reverse.definition)

    def test_reverse_opaque(self):
        """test opaque gates reverse to themselves"""
        opaque_gate = Gate(name="crz_2", num_qubits=2, params=[0.5])
        self.assertEqual(opaque_gate.reverse_ops(), opaque_gate)
        hgate = HGate()
        self.assertEqual(hgate.reverse_ops(), hgate)

    def test_inverse_and_append(self):
        """test appending inverted gates to circuits"""
        q = QuantumRegister(1)
        circ = QuantumCircuit(q, name="circ")
        circ.s(q)
        circ.append(SGate().inverse(), q[:])
        circ.append(TGate().inverse(), q[:])
        circ.t(q)
        gate = circ.to_instruction()
        circ = QuantumCircuit(q, name="circ")
        circ.inverse()
        circ.tdg(q)
        circ.t(q)
        circ.s(q)
        circ.sdg(q)
        gate_inverse = circ.to_instruction()
        self.assertEqual(gate.inverse().definition, gate_inverse.definition)

    def test_inverse_composite_gate(self):
        """test inverse of composite gate"""
        q = QuantumRegister(4)
        circ = QuantumCircuit(q, name="circ")
        circ.h(q[0])
        circ.crz(0.1, q[0], q[1])
        circ.id(q[1])
        circ.u(0.1, 0.2, -0.2, q[0])
        gate = circ.to_instruction()
        circ = QuantumCircuit(q, name="circ")
        circ.u(-0.1, 0.2, -0.2, q[0])
        circ.id(q[1])
        circ.crz(-0.1, q[0], q[1])
        circ.h(q[0])
        gate_inverse = circ.to_instruction()
        self.assertEqual(gate.inverse().definition, gate_inverse.definition)

    def test_inverse_recursive(self):
        """test that a hierarchical gate recursively inverts"""
        qr0 = QuantumRegister(2)
        circ0 = QuantumCircuit(qr0, name="circ0")
        circ0.t(qr0[0])
        circ0.rx(0.4, qr0[1])
        circ0.cx(qr0[1], qr0[0])
        little_gate = circ0.to_instruction()

        qr1 = QuantumRegister(4)
        circ1 = QuantumCircuit(qr1, name="circuit1")
        circ1.cp(-0.1, qr1[0], qr1[2])
        circ1.id(qr1[1])
        circ1.append(little_gate, [qr1[2], qr1[3]])

        circ_inv = QuantumCircuit(qr1, name="circ1_dg")
        circ_inv.append(little_gate.inverse(), [qr1[2], qr1[3]])
        circ_inv.id(qr1[1])
        circ_inv.cp(0.1, qr1[0], qr1[2])

        self.assertEqual(circ1.inverse(), circ_inv)

    def test_inverse_instruction_with_measure(self):
        """test inverting instruction with measure fails"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name="circ")
        circ.t(q[1])
        circ.u(0.1, 0.2, -0.2, q[0])
        circ.barrier()
        circ.measure(q[0], c[0])
        inst = circ.to_instruction()
        self.assertRaises(CircuitError, inst.inverse)

    def test_inverse_instruction_with_conditional(self):
        """test inverting instruction with conditionals fails"""
        q = QuantumRegister(4)
        c = ClassicalRegister(4)
        circ = QuantumCircuit(q, c, name="circ")
        circ.t(q[1])
        circ.u(0.1, 0.2, -0.2, q[0])
        circ.barrier()
        circ.measure(q[0], c[0])
        circ.rz(0.8, q[0])
        inst = circ.to_instruction()
        self.assertRaises(CircuitError, inst.inverse)

    def test_inverse_opaque(self):
        """test inverting opaque gate fails"""
        opaque_gate = Gate(name="crz_2", num_qubits=2, params=[0.5])
        self.assertRaises(CircuitError, opaque_gate.inverse)

    def test_inverse_empty(self):
        """test inverting empty gate works"""
        q = QuantumRegister(3)
        c = ClassicalRegister(3)
        empty_circ = QuantumCircuit(q, c, name="empty_circ")
        empty_gate = empty_circ.to_instruction()
        self.assertEqual(empty_gate.inverse().definition, empty_gate.definition)

    def test_inverse_with_global_phase(self):
        """test inverting instruction with global phase in definition."""
        q = QuantumRegister(1)
        circ = QuantumCircuit(q, name="circ", global_phase=np.pi / 3)
        circ.x(q)
        gate = circ.to_instruction()
        circ = QuantumCircuit(q, name="circ", global_phase=-np.pi / 3)
        circ.x(q)
        gate_inverse = circ.to_instruction()
        self.assertEqual(gate.inverse().definition, gate_inverse.definition)

    def test_inverse_with_label(self):
        """test inverting gate initialized with label attribute."""
        q = QuantumRegister(2)
        qc = QuantumCircuit(q, name="circ")
        qc.cx(0, 1)
        qc_gate = qc.to_gate()
        qc_gate_inverse = qc_gate.inverse()
        self.assertEqual(qc_gate.name + "_dg", qc_gate_inverse.name)
        qc_gate_inverse_inverse = qc_gate_inverse.inverse()
        self.assertEqual(qc_gate_inverse_inverse.name, qc_gate.name)

    def test_no_broadcast(self):
        """See https://github.com/Qiskit/qiskit-terra/issues/2777
        When creating custom instructions, do not broadcast parameters"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        subcircuit = QuantumCircuit(qr, cr, name="subcircuit")

        subcircuit.x(qr[0])
        subcircuit.h(qr[1])
        subcircuit.measure(qr[0], cr[0])
        subcircuit.measure(qr[1], cr[1])

        inst = subcircuit.to_instruction()
        circuit = QuantumCircuit(qr, cr, name="circuit")
        circuit.append(inst, qr[:], cr[:])
        self.assertEqual(circuit.qregs, [qr])
        self.assertEqual(circuit.cregs, [cr])
        self.assertEqual(circuit.qubits, [qr[0], qr[1]])
        self.assertEqual(circuit.clbits, [cr[0], cr[1]])

    def test_modifying_copied_params_leaves_orig(self):
        """Verify modifying the parameters of a copied instruction does not
        affect the original."""

        inst = Instruction("test", 2, 1, [0, 1, 2])

        cpy = inst.copy()

        cpy.params[1] = 7

        self.assertEqual(inst.params, [0, 1, 2])

    def test_instance_of_instruction(self):
        """Test correct error message is raised when invalid instruction
        is passed to append"""

        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        with self.assertRaisesRegex(CircuitError, r"Object is a subclass of Operation"):
            qc.append(HGate, qr[:], [])

    def test_repr_of_instructions(self):
        """Test the __repr__ method of the Instruction
        class"""

        ins1 = Instruction("test_instruction", 3, 5, [0, 1, 2, 3])
        self.assertEqual(
            repr(ins1),
            f"Instruction(name='{ins1.name}', num_qubits={ins1.num_qubits}, "
            f"num_clbits={ins1.num_clbits}, params={ins1.params})",
        )

        ins2 = random_circuit(num_qubits=4, depth=4, measure=True).to_instruction()
        self.assertEqual(
            repr(ins2),
            f"Instruction(name='{ins2.name}', num_qubits={ins2.num_qubits}, "
            f"num_clbits={ins2.num_clbits}, params={ins2.params})",
        )

    def test_label_type_enforcement(self):
        """Test instruction label type enforcement."""
        with self.subTest("accepts string labels"):
            instruction = Instruction("h", 1, 0, [], label="label")
            self.assertEqual(instruction.label, "label")
        with self.subTest("raises when a non-string label is provided to constructor"):
            with self.assertRaisesRegex(TypeError, r"label expects a string or None"):
                Instruction("h", 1, 0, [], label=0)
        with self.subTest("raises when a non-string label is provided to setter"):
            with self.assertRaisesRegex(TypeError, r"label expects a string or None"):
                instruction = RZGate(0)
                instruction.label = 0


class TestInverseAnnotatedGate(QiskitTestCase):
    """Tests for inverse gates and the AnnotatedOperation class."""

    def test_inverse_cx(self):
        """Test creation of inverse CX gate"""
        gate = CXGate().inverse(annotated=False)
        self.assertIsInstance(gate, CXGate)
        gate = CXGate().inverse(annotated=True)
        self.assertIsInstance(gate, CXGate)

    def test_inverse_s(self):
        """Test creation of inverse S gate"""
        gate = SGate().inverse(annotated=False)
        self.assertIsInstance(gate, SdgGate)
        gate = SGate().inverse(annotated=True)
        self.assertIsInstance(gate, SdgGate)

    def test_inverse_custom(self):
        """Test creation of inverse custom gate"""
        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        circ.h(0)
        gate = circ.to_instruction()
        self.assertIsInstance(gate, Instruction)
        inverse_gate = gate.inverse(annotated=False)
        self.assertIsInstance(inverse_gate, Instruction)
        inverse_gate = gate.inverse(annotated=True)
        self.assertIsInstance(inverse_gate, AnnotatedOperation)


if __name__ == "__main__":
    unittest.main()
