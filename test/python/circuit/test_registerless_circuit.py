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

"""Test registerless QuantumCircuit and Gates on wires"""

import numpy

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit, Clbit, AncillaQubit
from qiskit.circuit.exceptions import CircuitError
from test import QiskitTestCase


class TestRegisterlessCircuit(QiskitTestCase):
    """Test registerless QuantumCircuit."""

    def test_circuit_constructor_qwires(self):
        """Create a QuantumCircuit directly with quantum wires"""
        circuit = QuantumCircuit(2)

        expected = QuantumCircuit(QuantumRegister(2, "q"))

        self.assertEqual(circuit, expected)

    def test_circuit_constructor_wires_wrong(self):
        """Create a registerless QuantumCircuit wrongly"""
        self.assertRaises(CircuitError, QuantumCircuit, 1, 2, 3)  # QuantumCircuit(1, 2, 3)

    def test_circuit_constructor_wires_wrong_mix(self):
        """Create an almost-registerless QuantumCircuit"""
        # QuantumCircuit(1, ClassicalRegister(2))
        self.assertRaises(CircuitError, QuantumCircuit, 1, ClassicalRegister(2))


class TestAddingBitsWithoutRegisters(QiskitTestCase):
    """Test adding Bit instances outside of Registers."""

    def test_circuit_constructor_on_bits(self):
        """Verify we can add bits directly to a circuit."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit()]
        ancillas = [AncillaQubit(), AncillaQubit()]

        qc = QuantumCircuit(qubits, clbits, ancillas)

        self.assertEqual(qc.qubits, qubits + ancillas)
        self.assertEqual(qc.clbits, clbits)
        self.assertEqual(qc.ancillas, ancillas)

        self.assertEqual(qc.qregs, [])
        self.assertEqual(qc.cregs, [])

    def test_circuit_constructor_on_invalid_bits(self):
        """Verify we raise if passed not a Bit."""
        with self.assertRaisesRegex(CircuitError, "Expected an instance of"):
            _ = QuantumCircuit([3.14])

    def test_raise_if_bits_already_present(self):
        """Verify we raise when attempting to add a Bit already in the circuit."""
        qubits = [Qubit()]

        with self.assertRaisesRegex(CircuitError, "bits found already"):
            _ = QuantumCircuit(qubits, qubits)

        qc = QuantumCircuit(qubits)

        with self.assertRaisesRegex(CircuitError, "bits found already"):
            qc.add_bits(qubits)

        qr = QuantumRegister(1, "qr")
        qc = QuantumCircuit(qr)

        with self.assertRaisesRegex(CircuitError, "bits found already"):
            qc.add_bits(qr[:])

    def test_adding_individual_bit(self):
        """Verify we can add a single bit to a circuit."""
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)

        new_bit = Qubit()

        qc.add_bits([new_bit])

        self.assertEqual(qc.qubits, list(qr) + [new_bit])
        self.assertEqual(qc.qregs, [qr])

    def test_inserted_ancilla_bits_are_added_to_qubits(self):
        """Verify AncillaQubits added via .add_bits are added to .qubits."""
        anc = AncillaQubit()
        qb = Qubit()

        qc = QuantumCircuit()
        qc.add_bits([anc, qb])

        self.assertEqual(qc.qubits, [anc, qb])


class TestGatesOnWires(QiskitTestCase):
    """Test gates on wires."""

    def test_circuit_single_wire_h(self):
        """Test circuit on wire (H gate)."""
        qreg = QuantumRegister(2)
        circuit = QuantumCircuit(qreg)
        circuit.h(1)

        expected = QuantumCircuit(qreg)
        expected.h(qreg[1])

        self.assertEqual(circuit, expected)

    def test_circuit_two_wire_cx(self):
        """Test circuit two wires (CX gate)."""
        qreg = QuantumRegister(2)
        expected = QuantumCircuit(qreg)
        expected.cx(qreg[0], qreg[1])

        circuit = QuantumCircuit(qreg)
        circuit.cx(0, 1)

        self.assertEqual(circuit, expected)

    def test_circuit_single_wire_measure(self):
        """Test circuit on wire (measure gate)."""
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)

        circuit = QuantumCircuit(qreg, creg)
        circuit.measure(1, 1)

        expected = QuantumCircuit(qreg, creg)
        expected.measure(qreg[1], creg[1])

        self.assertEqual(circuit, expected)

    def test_circuit_multi_qregs_h(self):
        """Test circuit multi qregs and wires (H gate)."""
        qreg0 = QuantumRegister(2)
        qreg1 = QuantumRegister(2)

        circuit = QuantumCircuit(qreg0, qreg1)
        circuit.h(0)
        circuit.h(2)

        expected = QuantumCircuit(qreg0, qreg1)
        expected.h(qreg0[0])
        expected.h(qreg1[0])

        self.assertEqual(circuit, expected)

    def test_circuit_multi_qreg_cregs_measure(self):
        """Test circuit multi qregs/cregs and wires (measure)."""
        qreg0 = QuantumRegister(2)
        creg0 = ClassicalRegister(2)
        qreg1 = QuantumRegister(2)
        creg1 = ClassicalRegister(2)

        circuit = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        circuit.measure(0, 2)
        circuit.measure(2, 1)

        expected = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        expected.measure(qreg0[0], creg1[0])
        expected.measure(qreg1[0], creg0[1])

        self.assertEqual(circuit, expected)

    def test_circuit_barrier(self):
        """Test barrier on wires."""
        qreg01 = QuantumRegister(2)
        qreg23 = QuantumRegister(2)

        circuit = QuantumCircuit(qreg01, qreg23)
        circuit.barrier(0)
        circuit.barrier(2)

        expected = QuantumCircuit(qreg01, qreg23)
        expected.barrier(qreg01[0])
        expected.barrier(qreg23[0])

        self.assertEqual(circuit, expected)

    def test_circuit_qwire_out_of_range(self):
        """Fail if quantum wire is out of range."""
        qreg = QuantumRegister(2)

        circuit = QuantumCircuit(qreg)
        self.assertRaises(CircuitError, circuit.h, 99)  # circuit.h(99)

    def test_circuit_cwire_out_of_range(self):
        """Fail if classical wire is out of range."""
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)

        circuit = QuantumCircuit(qreg, creg)
        self.assertRaises(CircuitError, circuit.measure, 1, 99)  # circuit.measure(1, 99)

    def test_circuit_initialize(self):
        """Test initialize on wires."""
        init_vector = [0.5, 0.5, 0.5, 0.5]
        qreg01 = QuantumRegister(2)
        qreg23 = QuantumRegister(2)
        circuit = QuantumCircuit(qreg01, qreg23)
        circuit.initialize(init_vector, [0, 2])

        expected = QuantumCircuit(qreg01, qreg23)
        expected.initialize(init_vector, [qreg01[0], qreg23[0]])

        self.assertEqual(circuit, expected)

    def test_circuit_initialize_single_qubit(self):
        """Test initialize on single qubit."""
        init_vector = [numpy.sqrt(0.5), numpy.sqrt(0.5)]
        qreg = QuantumRegister(2)
        circuit = QuantumCircuit(qreg)
        circuit.initialize(init_vector, qreg[0])

        expected = QuantumCircuit(qreg)
        expected.initialize(init_vector, [qreg[0]])

        self.assertEqual(circuit, expected)

    def test_mixed_register_and_registerless_indexing(self):
        """Test indexing if circuit contains bits in and out of registers."""

        bits = [Qubit(), Qubit()]
        qreg = QuantumRegister(3, "q")
        circuit = QuantumCircuit(bits, qreg)
        for i in range(len(circuit.qubits)):
            circuit.rz(i, i)

        expected_qubit_order = bits + qreg[:]
        expected_circuit = QuantumCircuit(bits, qreg)
        for i in range(len(expected_circuit.qubits)):
            expected_circuit.rz(i, expected_qubit_order[i])

        self.assertEqual(circuit.data, expected_circuit.data)


class TestGatesOnWireRange(QiskitTestCase):
    """Test gates on wire range."""

    def test_wire_range(self):
        """Test gate wire range"""
        qreg = QuantumRegister(4)
        circuit = QuantumCircuit(qreg)
        circuit.h(range(2))

        expected = QuantumCircuit(qreg)
        expected.h(qreg[0:2])

        self.assertEqual(circuit, expected)

    def test_circuit_multi_qregs_h(self):
        """Test circuit multi qregs in range of wires (H gate)."""
        qreg0 = QuantumRegister(2)
        qreg1 = QuantumRegister(2)
        circuit = QuantumCircuit(qreg0, qreg1)
        circuit.h(range(3))

        expected = QuantumCircuit(qreg0, qreg1)
        expected.h(qreg0[0])
        expected.h(qreg0[1])
        expected.h(qreg1[0])

        self.assertEqual(circuit, expected)

    def test_circuit_multi_qreg_cregs_measure(self):
        """Test circuit multi qregs in range of wires (measure)."""
        qreg0 = QuantumRegister(2)
        creg0 = ClassicalRegister(2)
        qreg1 = QuantumRegister(2)
        creg1 = ClassicalRegister(2)
        circuit = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        circuit.measure(range(1, 3), range(0, 4, 2))

        expected = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        expected.measure(qreg0[1], creg0[0])
        expected.measure(qreg1[0], creg1[0])

        self.assertEqual(circuit, expected)

    def test_circuit_barrier(self):
        """Test barrier on range of wires with multi regs."""
        qreg01 = QuantumRegister(2)
        qreg23 = QuantumRegister(2)
        circuit = QuantumCircuit(qreg01, qreg23)
        circuit.barrier(range(3))

        expected = QuantumCircuit(qreg01, qreg23)
        expected.barrier(qreg01[0], qreg01[1], qreg23[0])

        self.assertEqual(circuit, expected)

    def test_circuit_initialize(self):
        """Test initialize on wires."""
        init_vector = [0.5, 0.5, 0.5, 0.5]
        qreg01 = QuantumRegister(2)
        qreg23 = QuantumRegister(2)
        circuit = QuantumCircuit(qreg01, qreg23)
        circuit.initialize(init_vector, range(1, 3))

        expected = QuantumCircuit(qreg01, qreg23)
        expected.initialize(init_vector, [qreg01[1], qreg23[0]])

        self.assertEqual(circuit, expected)

    def test_circuit_qwire_out_of_range(self):
        """Fail if quantum wire is out of range."""
        qreg = QuantumRegister(2)
        circuit = QuantumCircuit(qreg)
        self.assertRaises(CircuitError, circuit.h, range(9, 99))  # circuit.h(range(9,99))

    def test_circuit_cwire_out_of_range(self):
        """Fail if classical wire is out of range."""
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        circuit = QuantumCircuit(qreg, creg)
        # circuit.measure(1, range(9,99))
        self.assertRaises(CircuitError, circuit.measure, 1, range(9, 99))


class TestGatesOnWireSlice(QiskitTestCase):
    """Test gates on wire slice."""

    def test_wire_slice(self):
        """Test gate wire slice"""
        qreg = QuantumRegister(4)
        circuit = QuantumCircuit(qreg)
        circuit.h(slice(0, 2))

        expected = QuantumCircuit(qreg)
        expected.h(qreg[0:2])

        self.assertEqual(circuit, expected)

    def test_wire_list(self):
        """Test gate wire list of integers"""
        qreg = QuantumRegister(4)
        circuit = QuantumCircuit(qreg)
        circuit.h([0, 1])

        expected = QuantumCircuit(qreg)
        expected.h(qreg[0:2])

        self.assertEqual(circuit, expected)

    def test_wire_np_int(self):
        """Test gate wire with numpy int"""
        numpy_int = numpy.dtype("int").type(2)
        qreg = QuantumRegister(4)
        circuit = QuantumCircuit(qreg)
        circuit.h(numpy_int)

        expected = QuantumCircuit(qreg)
        expected.h(qreg[2])

        self.assertEqual(circuit, expected)

    def test_wire_np_1d_array(self):
        """Test gate wire with numpy array (one-dimensional)"""
        numpy_arr = numpy.array([0, 1])
        qreg = QuantumRegister(4)
        circuit = QuantumCircuit(qreg)
        circuit.h(numpy_arr)

        expected = QuantumCircuit(qreg)
        expected.h(qreg[0])
        expected.h(qreg[1])

        self.assertEqual(circuit, expected)

    def test_circuit_multi_qregs_h(self):
        """Test circuit multi qregs in slices of wires (H gate)."""
        qreg0 = QuantumRegister(2)
        qreg1 = QuantumRegister(2)
        circuit = QuantumCircuit(qreg0, qreg1)
        circuit.h(slice(0, 3))

        expected = QuantumCircuit(qreg0, qreg1)
        expected.h(qreg0[0])
        expected.h(qreg0[1])
        expected.h(qreg1[0])

        self.assertEqual(circuit, expected)

    def test_circuit_multi_qreg_cregs_measure(self):
        """Test circuit multi qregs in slices of wires (measure)."""
        qreg0 = QuantumRegister(2)
        creg0 = ClassicalRegister(2)
        qreg1 = QuantumRegister(2)
        creg1 = ClassicalRegister(2)
        circuit = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        circuit.measure(slice(1, 3), slice(0, 4, 2))

        expected = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        expected.measure(qreg0[1], creg0[0])
        expected.measure(qreg1[0], creg1[0])

        self.assertEqual(circuit, expected)

    def test_circuit_barrier(self):
        """Test barrier on slice of wires with multi regs."""
        qreg01 = QuantumRegister(2)
        qreg23 = QuantumRegister(2)
        circuit = QuantumCircuit(qreg01, qreg23)
        circuit.barrier(slice(0, 3))

        expected = QuantumCircuit(qreg01, qreg23)
        expected.barrier([qreg01[0], qreg01[1], qreg23[0]])

        self.assertEqual(circuit, expected)

    def test_circuit_initialize(self):
        """Test initialize on wires."""
        init_vector = [0.5, 0.5, 0.5, 0.5]
        qreg01 = QuantumRegister(2)
        qreg23 = QuantumRegister(2)
        circuit = QuantumCircuit(qreg01, qreg23)
        circuit.initialize(init_vector, slice(1, 3))

        expected = QuantumCircuit(qreg01, qreg23)
        expected.initialize(init_vector, [qreg01[1], qreg23[0]])

        self.assertEqual(circuit, expected)

    def test_circuit_qwire_out_of_range(self):
        """Fail if quantum wire is out of range."""
        qreg = QuantumRegister(2)
        circuit = QuantumCircuit(qreg)
        self.assertRaises(CircuitError, circuit.h, slice(9, 99))  # circuit.h(slice(9,99))

    def test_circuit_cwire_out_of_range(self):
        """Fail if classical wire is out of range."""
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        circuit = QuantumCircuit(qreg, creg)
        # circuit.measure(1, slice(9,99))
        self.assertRaises(CircuitError, circuit.measure, 1, slice(9, 99))

    def test_wire_np_2d_array(self):
        """Test gate wire with numpy array (two-dimensional). Raises."""
        numpy_arr = numpy.array([[0, 1], [2, 3]])
        qreg = QuantumRegister(4)
        circuit = QuantumCircuit(qreg)
        self.assertRaises(CircuitError, circuit.h, numpy_arr)  # circuit.h(numpy_arr)
