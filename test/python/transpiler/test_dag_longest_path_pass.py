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

"""DAGFixedPoint pass testing"""

import unittest
from qiskit.transpiler.passes import DAGLongestPath
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase


class TestDAGLongestPathPass(QiskitTestCase):
    """Tests for PropertySet methods."""

    def test_empty_dag_true(self):
        """Test the dag longest path of an empty dag."""
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)

        pass_ = DAGLongestPath()
        pass_.run(dag)
        self.assertListEqual(pass_.property_set["dag_longest_path"], [])

    def test_nonempty_dag_false(self):
        """Test the dag longest path non-empty dag.
        path length = 11 = 9 ops + 2 qubits at start and end of path
        """
        #            ┌───┐┌───┐┌───┐
        # q0_0: ──■──┤ X ├┤ Y ├┤ H ├──■───────────────────■──
        #       ┌─┴─┐└───┘└───┘└───┘┌─┴─┐┌───┐┌───┐┌───┐┌─┴─┐
        # q0_1: ┤ X ├───────────────┤ X ├┤ X ├┤ Y ├┤ H ├┤ X ├
        #       └───┘               └───┘└───┘└───┘└───┘└───┘
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[0])
        circuit.y(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[1])
        circuit.y(qr[1])
        circuit.h(qr[1])
        circuit.cx(qr[0], qr[1])
        dag = circuit_to_dag(circuit)

        pass_ = DAGLongestPath()
        pass_.run(dag)
        self.assertEqual(len(pass_.property_set["dag_longest_path"]), 11)


if __name__ == "__main__":
    unittest.main()
