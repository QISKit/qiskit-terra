# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the RemoveBarriers pass"""

import unittest
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
from test import QiskitTestCase


class TestMergeAdjacentBarriers(QiskitTestCase):
    """Test the MergeAdjacentBarriers pass"""

    def test_remove_barriers(self):
        """Remove all barriers"""

        circuit = QuantumCircuit(2)
        circuit.barrier()
        circuit.barrier()

        pass_ = RemoveBarriers()
        result_dag = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result_dag.size(), 0)

    def test_remove_barriers_other_gates(self):
        """Remove all barriers, leave other gates intact"""

        circuit = QuantumCircuit(1)
        circuit.barrier()
        circuit.x(0)
        circuit.barrier()
        circuit.h(0)

        pass_ = RemoveBarriers()
        result_dag = pass_.run(circuit_to_dag(circuit))

        op_nodes = result_dag.op_nodes()

        self.assertEqual(result_dag.size(), 2)
        for ii, name in enumerate(["x", "h"]):
            self.assertEqual(op_nodes[ii].name, name)

    def test_simple_if_else(self):
        """Test that the pass recurses into an if-else."""
        pass_ = RemoveBarriers()

        base_test = QuantumCircuit(1, 1)
        base_test.barrier()
        base_test.measure(0, 0)

        base_expected = QuantumCircuit(1, 1)
        base_expected.measure(0, 0)

        test = QuantumCircuit(1, 1)
        test.if_else(
            (test.clbits[0], True), base_test.copy(), base_test.copy(), test.qubits, test.clbits
        )

        expected = QuantumCircuit(1, 1)
        expected.if_else(
            (expected.clbits[0], True),
            base_expected.copy(),
            base_expected.copy(),
            expected.qubits,
            expected.clbits,
        )

        self.assertEqual(pass_(test), expected)

    def test_nested_control_flow(self):
        """Test that the pass recurses into nested control flow."""
        pass_ = RemoveBarriers()

        base_test = QuantumCircuit(1, 1)
        base_test.barrier()
        base_test.measure(0, 0)

        base_expected = QuantumCircuit(1, 1)
        base_expected.measure(0, 0)

        body_test = QuantumCircuit(1, 1)
        body_test.for_loop((0,), None, base_expected.copy(), body_test.qubits, body_test.clbits)

        body_expected = QuantumCircuit(1, 1)
        body_expected.for_loop(
            (0,), None, base_expected.copy(), body_expected.qubits, body_expected.clbits
        )

        test = QuantumCircuit(1, 1)
        test.while_loop((test.clbits[0], True), body_test, test.qubits, test.clbits)

        expected = QuantumCircuit(1, 1)
        expected.while_loop(
            (expected.clbits[0], True), body_expected, expected.qubits, expected.clbits
        )

        self.assertEqual(pass_(test), expected)


if __name__ == "__main__":
    unittest.main()
