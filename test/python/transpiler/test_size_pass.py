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

"""Size pass testing"""

import unittest

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import Size
from test import QiskitTestCase


class TestSizePass(QiskitTestCase):
    """Tests for Depth analysis methods."""

    def test_empty_dag(self):
        """Empty DAG has 0 size"""
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)

        pass_ = Size()
        _ = pass_.run(dag)

        self.assertEqual(pass_.property_set["size"], 0)

    def test_just_qubits(self):
        """A dag with 8 operations and no classic bits"""
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

        pass_ = Size()
        _ = pass_.run(dag)

        self.assertEqual(pass_.property_set["size"], 8)

    def test_depth_one(self):
        """A dag with operations in parallel and size 2"""
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])
        dag = circuit_to_dag(circuit)

        pass_ = Size()
        _ = pass_.run(dag)

        self.assertEqual(pass_.property_set["size"], 2)

    def test_size_control_flow(self):
        """A DAG with control flow still gives an estimate."""
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], True)) as else_:
            qc.x(1)
            qc.cx(2, 3)
        with else_:
            qc.x(1)
            with qc.for_loop(range(3)):
                qc.z(2)
                with qc.for_loop((4, 0, 1)):
                    qc.z(2)
        with qc.while_loop((qc.clbits[0], True)):
            qc.h(0)
            qc.measure(0, 0)
        pass_ = Size(recurse=True)
        pass_(qc)
        self.assertEqual(pass_.property_set["size"], 19)


if __name__ == "__main__":
    unittest.main()
