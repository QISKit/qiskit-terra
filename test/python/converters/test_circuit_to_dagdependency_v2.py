# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for the converter dag dependency to circuit and circuit to dag
dependency V2."""

import unittest

from qiskit.converters.dagdependency_to_circuit import dagdependency_to_circuit
from qiskit.converters.circuit_to_dagdependency_v2 import _circuit_to_dagdependency_v2
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from test import QiskitTestCase


class TestCircuitToDAGDependencyV2(QiskitTestCase):
    """Test QuantumCircuit to DAGDependencyV2."""

    def test_circuit_and_dag_canonical(self):
        """Check convert to dag dependency and back"""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit_in = QuantumCircuit(qr, cr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[1])
        circuit_in.measure(qr[0], cr[0])
        circuit_in.measure(qr[1], cr[1])
        circuit_in.measure(qr[0], cr[0])
        circuit_in.measure(qr[1], cr[1])
        circuit_in.measure(qr[2], cr[2])
        dag_dependency = _circuit_to_dagdependency_v2(circuit_in)
        circuit_out = dagdependency_to_circuit(dag_dependency)
        self.assertEqual(circuit_out, circuit_in)

    def test_metadata(self):
        """Test circuit metadata is preserved through conversion."""
        meta_dict = {"experiment_id": "1234", "execution_number": 4}
        qr = QuantumRegister(2)
        circuit_in = QuantumCircuit(qr, metadata=meta_dict)
        circuit_in.h(qr[0])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.measure_all()
        dag_dependency = _circuit_to_dagdependency_v2(circuit_in)
        self.assertEqual(dag_dependency.metadata, meta_dict)
        circuit_out = dagdependency_to_circuit(dag_dependency)
        self.assertEqual(circuit_out.metadata, meta_dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
