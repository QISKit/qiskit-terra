# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Testing naming functionality of transpiled circuits"""

import unittest

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.transpiler.exceptions import TranspilerError
from test import QiskitTestCase


class TestNamingTranspiledCircuits(QiskitTestCase):
    """Testing the naming functionality for transpiled circuits."""

    def setUp(self):
        super().setUp()
        self.basis_gates = ["u1", "u2", "u3", "cx"]
        self.backend = BasicSimulator()

        self.circuit0 = QuantumCircuit(name="circuit0")
        self.circuit1 = QuantumCircuit(name="circuit1")
        self.circuit2 = QuantumCircuit(name="circuit2")
        self.circuit3 = QuantumCircuit(name="circuit3")

    def test_single_circuit_name_singleton(self):
        """Test output_name with a single circuit
        Given a single circuit and a output name in form of a string, this test
        checks whether that string name is assigned to the transpiled circuit.
        """

        trans_cirkie = transpile(
            self.circuit0, basis_gates=self.basis_gates, output_name="transpiled-cirkie"
        )
        self.assertEqual(trans_cirkie.name, "transpiled-cirkie")

    def test_single_circuit_name_list(self):
        """Test singleton output_name and a single circuit
        Given a single circuit and an output name in form of a single element
        list, this test checks whether the transpiled circuit is mapped with
        that assigned name in the list.
        If list has more than one element, then test checks whether the
        Transpile function raises an error.
        """
        trans_cirkie = transpile(
            self.circuit0, basis_gates=self.basis_gates, output_name=["transpiled-cirkie"]
        )
        self.assertEqual(trans_cirkie.name, "transpiled-cirkie")

    def test_single_circuit_and_multiple_name_list(self):
        """Test multiple output_name and a single circuit"""
        # If List has multiple elements, transpile function must raise error
        with self.assertRaises(TranspilerError):
            transpile(
                self.circuit0,
                basis_gates=self.basis_gates,
                output_name=["cool-cirkie", "new-cirkie", "dope-cirkie", "awesome-cirkie"],
            )

    def test_multiple_circuits_name_singleton(self):
        """Test output_name raise error if a single name is provided to a list of circuits
        Given multiple circuits and a single string as a name, this test checks
        whether the Transpile function raises an error.
        """
        # Raise Error if single name given to multiple circuits
        with self.assertRaises(TranspilerError):
            transpile([self.circuit1, self.circuit2], self.backend, output_name="circ")

    def test_multiple_circuits_name_list(self):
        """Test output_name with a list of circuits
        Given multiple circuits and a list for output names, if
        len(list)=len(circuits), then test checks whether transpile func assigns
        each element in list to respective circuit.
        If lengths are not equal, then test checks whether transpile func raises
        error.
        """
        # combining multiple circuits
        circuits = [self.circuit1, self.circuit2, self.circuit3]
        # equal lengths
        names = ["awesome-circ1", "awesome-circ2", "awesome-circ3"]
        trans_circuits = transpile(circuits, self.backend, output_name=names)
        self.assertEqual(trans_circuits[0].name, "awesome-circ1")
        self.assertEqual(trans_circuits[1].name, "awesome-circ2")
        self.assertEqual(trans_circuits[2].name, "awesome-circ3")

    def test_greater_circuits_name_list(self):
        """Test output_names list greater than circuits list"""
        # combining multiple circuits
        circuits = [self.circuit1, self.circuit2, self.circuit3]
        # names list greater than circuits list
        names = ["awesome-circ1", "awesome-circ2", "awesome-circ3", "awesome-circ4"]
        with self.assertRaises(TranspilerError):
            transpile(circuits, self.backend, output_name=names)

    def test_smaller_circuits_name_list(self):
        """Test output_names list smaller than circuits list"""
        # combining multiple circuits
        circuits = [self.circuit1, self.circuit2, self.circuit3]
        # names list smaller than circuits list
        names = ["awesome-circ1", "awesome-circ2"]
        with self.assertRaises(TranspilerError):
            transpile(circuits, self.backend, output_name=names)


if __name__ == "__main__":
    unittest.main()
