# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""FilterOpNodes pass testing"""


from qiskit import QuantumCircuit
from qiskit.transpiler.passes import FilterOpNodes
from test import QiskitTestCase


class TestFilterOpNodes(QiskitTestCase):
    """Tests for FilterOpNodes transformation pass."""

    def test_empty_circuit(self):
        """Empty DAG has does nothing."""
        circuit = QuantumCircuit()
        self.assertEqual(FilterOpNodes(lambda x: False)(circuit), circuit)

    def test_remove_x_gate(self):
        """Test filter removes matching gates."""
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        circuit.measure_all()

        filter_pass = FilterOpNodes(lambda node: node.op.name != "x")

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.cx(1, 0)
        expected.cx(0, 1)
        expected.measure_all()

        self.assertEqual(filter_pass(circuit), expected)

    def test_filter_exception(self):
        """Test a filter function exception passes through."""
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        circuit.measure_all()

        def filter_fn(node):
            raise TypeError("Failure")

        filter_pass = FilterOpNodes(filter_fn)
        with self.assertRaises(TypeError):
            filter_pass(circuit)

    def test_no_matches(self):
        """Test the pass does nothing if there are no filter matches."""
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        circuit.measure_all()

        filter_pass = FilterOpNodes(lambda node: node.op.name != "cz")

        self.assertEqual(filter_pass(circuit), circuit)
