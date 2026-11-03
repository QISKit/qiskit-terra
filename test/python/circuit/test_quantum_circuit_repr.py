# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test __repr__ method for QuantumCircuit class."""

from qiskit.circuit import QuantumCircuit
from test import QiskitTestCase


class TestQuantumCircuitRepr(QiskitTestCase):
    """Tests for QuantumCircuit.__repr__ method."""

    def test_quantum_circuit_repr_basic(self):
        """Test basic QuantumCircuit repr."""
        qc = QuantumCircuit(2, 2)
        result = repr(qc)
        expected = f"<QuantumCircuit '{qc.name}' with 2 qubits, 2 clbits, 0 instructions, and global_phase={qc.global_phase}>"
        self.assertEqual(result, expected)

    def test_quantum_circuit_repr_with_name(self):
        """Test QuantumCircuit repr with custom name."""
        qc = QuantumCircuit(2, 2, name='Bell')
        result = repr(qc)
        expected = f"<QuantumCircuit 'Bell' with 2 qubits, 2 clbits, 0 instructions, and global_phase={qc.global_phase}>"
        self.assertEqual(result, expected)

    def test_quantum_circuit_repr_with_instructions(self):
        """Test QuantumCircuit repr with instructions."""
        qc = QuantumCircuit(2, 2, name='Bell')
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        result = repr(qc)
        expected = f"<QuantumCircuit 'Bell' with 2 qubits, 2 clbits, {len(qc.data)} instructions, and global_phase={qc.global_phase}>"
        self.assertEqual(result, expected)

    def test_quantum_circuit_repr_with_global_phase(self):
        """Test QuantumCircuit repr with non-zero global phase."""
        qc = QuantumCircuit(1, name='PhaseCircuit', global_phase=1.57)
        result = repr(qc)
        expected = f"<QuantumCircuit 'PhaseCircuit' with 1 qubits, 0 clbits, 0 instructions, and global_phase={qc.global_phase}>"
        self.assertEqual(result, expected)

    def test_quantum_circuit_repr_no_clbits(self):
        """Test QuantumCircuit repr without classical bits."""
        qc = QuantumCircuit(3, name='NoClbits')
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        result = repr(qc)
        expected = f"<QuantumCircuit 'NoClbits' with 3 qubits, 0 clbits, 3 instructions, and global_phase={qc.global_phase}>"
        self.assertEqual(result, expected)

