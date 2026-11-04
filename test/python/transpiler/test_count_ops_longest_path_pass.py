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

"""Depth pass testing"""

import unittest

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import CountOpsLongestPath
from test import QiskitTestCase


class TestCountOpsLongestPathPass(QiskitTestCase):
    """Tests for CountOpsLongestPath analysis methods."""

    def test_empty_dag(self):
        """Empty DAG has empty counts."""
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)

        pass_ = CountOpsLongestPath()
        _ = pass_.run(dag)

        self.assertDictEqual(pass_.property_set["count_ops_longest_path"], {})

    def test_just_qubits(self):
        """A dag with 9 operations (3 CXs, 2Xs, 2Ys and 2 Hs) on the longest
        path
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

        pass_ = CountOpsLongestPath()
        _ = pass_.run(dag)

        count_ops = pass_.property_set["count_ops_longest_path"]
        self.assertDictEqual(count_ops, {"cx": 3, "x": 2, "y": 2, "h": 2})


if __name__ == "__main__":
    unittest.main()
