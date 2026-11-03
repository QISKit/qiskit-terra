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

"""Test __repr__ methods for Gate and ControlledGate classes."""

from qiskit.circuit import Parameter
from qiskit.circuit.library import HGate, CXGate, RXGate, CCXGate
from test import QiskitTestCase


class TestGateRepr(QiskitTestCase):
    """Tests for Gate.__repr__ method."""

    def test_gate_repr_basic(self):
        """Test basic Gate repr without label."""
        gate = HGate()
        result = repr(gate)
        # Use actual class name since some gates are singletons
        expected = f"<{gate.__class__.__name__} 'h' with 1 qubits, 0 clbits and params=[]>"
        self.assertEqual(result, expected)

    def test_gate_repr_with_label(self):
        """Test Gate repr with label."""
        gate = HGate(label='my_hadamard')
        result = repr(gate)
        expected = "<HGate 'h' labeled 'my_hadamard' with 1 qubits, 0 clbits and params=[]>"
        self.assertEqual(result, expected)

    def test_gate_repr_with_params(self):
        """Test Gate repr with parameters."""
        gate = RXGate(1.57)
        result = repr(gate)
        # Build expected string using the actual params from the gate
        expected = f"<RXGate 'rx' with 1 qubits, 0 clbits and params={gate.params}>"
        self.assertEqual(result, expected)

    def test_gate_repr_with_parameter_expression(self):
        """Test Gate repr with Parameter expression."""
        theta = Parameter('theta')
        gate = RXGate(theta)
        result = repr(gate)
        # Build expected string using the actual params from the gate
        expected = f"<RXGate 'rx' with 1 qubits, 0 clbits and params={gate.params}>"
        self.assertEqual(result, expected)


class TestControlledGateRepr(QiskitTestCase):
    """Tests for ControlledGate.__repr__ method."""

    def test_controlled_gate_repr_basic(self):
        """Test basic ControlledGate repr without label."""
        gate = CXGate()
        result = repr(gate)
        # Use actual class name since some gates are singletons
        expected = f"<{gate.__class__.__name__} 'cx' with 1 control qubits, control state = 1 and params=[]>"
        self.assertEqual(result, expected)

    def test_controlled_gate_repr_with_label(self):
        """Test ControlledGate repr with label."""
        gate = CXGate(label='my_cnot')
        result = repr(gate)
        expected = "<CXGate 'cx' labeled 'my_cnot' with 1 control qubits, control state = 1 and params=[]>"
        self.assertEqual(result, expected)

    def test_controlled_gate_repr_multiple_controls(self):
        """Test ControlledGate repr with multiple control qubits."""
        gate = CCXGate()
        result = repr(gate)
        # Use actual class name since some gates are singletons
        expected = f"<{gate.__class__.__name__} 'ccx' with 2 control qubits, control state = 3 and params=[]>"
        self.assertEqual(result, expected)

