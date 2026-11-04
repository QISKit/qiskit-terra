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

"""NumTensorFactors pass testing"""

import unittest

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import NumTensorFactors
from test import QiskitTestCase


class TestNumTensorsFactorPass(QiskitTestCase):
    """Tests for NumTensorFactors analysis methods."""

    def test_empty_dag(self):
        """Empty DAG has 0 number of tensor factors."""
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)

        pass_ = NumTensorFactors()
        _ = pass_.run(dag)

        self.assertEqual(pass_.property_set["num_tensor_factors"], 0)

    def test_just_qubits(self):
        """A dag with 8 operations and 1 tensor factor."""
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = NumTensorFactors()
        _ = pass_.run(dag)

        self.assertEqual(pass_.property_set["num_tensor_factors"], 1)

    def test_depth_one(self):
        """A dag with operations in parallel (2 tensor factors)"""
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])
        dag = circuit_to_dag(circuit)

        pass_ = NumTensorFactors()
        _ = pass_.run(dag)

        self.assertEqual(pass_.property_set["num_tensor_factors"], 2)


if __name__ == "__main__":
    unittest.main()
