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

"""Test the MergeAdjacentBarriers pass"""

import random
import unittest

from qiskit.transpiler.passes import MergeAdjacentBarriers
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit
from test import QiskitTestCase


class TestMergeAdjacentBarriers(QiskitTestCase):
    """Test the MergeAdjacentBarriers pass"""

    def test_two_identical_barriers(self):
        """Merges two barriers that are identical into one
                 ░  ░                  ░
        q_0: |0>─░──░─   ->   q_0: |0>─░─
                 ░  ░                  ░
        """
        qr = QuantumRegister(1, "q")

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_numerous_identical_barriers(self):
        """Merges 5 identical barriers in a row into one
                 ░  ░  ░  ░  ░  ░                     ░
        q_0: |0>─░──░──░──░──░──░─    ->     q_0: |0>─░─
                 ░  ░  ░  ░  ░  ░                     ░
        """
        qr = QuantumRegister(1, "q")

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_of_different_sizes(self):
        """Test two barriers of different sizes are merged into one
                 ░  ░                     ░
        q_0: |0>─░──░─           q_0: |0>─░─
                 ░  ░     ->              ░
        q_1: |0>────░─           q_1: |0>─░─
                    ░                     ░
        """
        qr = QuantumRegister(2, "q")

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_not_overlapping_barriers(self):
        """Test two barriers with no overlap are not merged
        (NB in these pictures they look like 1 barrier but they are
            actually 2 distinct barriers, this is just how the text
            drawer draws them)
                 ░                     ░
        q_0: |0>─░─           q_0: |0>─░─
                 ░     ->              ░
        q_1: |0>─░─           q_1: |0>─░─
                 ░                     ░
        """
        qr = QuantumRegister(2, "q")

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.barrier(qr[1])

        expected = QuantumCircuit(qr)
        expected.barrier(qr[0])
        expected.barrier(qr[1])

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_obstacle_before(self):
        """Test with an obstacle before the larger barrier
                  ░   ░                          ░
        q_0: |0>──░───░─           q_0: |0>──────░─
                ┌───┐ ░     ->             ┌───┐ ░
        q_1: |0>┤ H ├─░─           q_1: |0>┤ H ├─░─
                └───┘ ░                    └───┘ ░
        """
        qr = QuantumRegister(2, "q")

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.h(qr[1])
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.h(qr[1])
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_obstacle_after(self):
        """Test with an obstacle after the larger barrier
                 ░   ░                      ░
        q_0: |0>─░───░──           q_0: |0>─░──────
                 ░ ┌───┐    ->              ░ ┌───┐
        q_1: |0>─░─┤ H ├           q_1: |0>─░─┤ H ├
                 ░ └───┘                    ░ └───┘
        """
        qr = QuantumRegister(2, "q")

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.barrier(qr[0])
        circuit.h(qr[1])

        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        expected.h(qr[1])

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_blocking_obstacle(self):
        """Test that barriers don't merge if there is an obstacle that
        is blocking
                 ░ ┌───┐ ░                     ░ ┌───┐ ░
        q_0: |0>─░─┤ H ├─░─    ->     q_0: |0>─░─┤ H ├─░─
                 ░ └───┘ ░                     ░ └───┘ ░
        """
        qr = QuantumRegister(1, "q")

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.h(qr)
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        expected.h(qr)
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_blocking_obstacle_long(self):
        """Test that barriers don't merge if there is an obstacle that
            is blocking
                 ░ ┌───┐ ░                     ░ ┌───┐ ░
        q_0: |0>─░─┤ H ├─░─           q_0: |0>─░─┤ H ├─░─
                 ░ └───┘ ░     ->              ░ └───┘ ░
        q_1: |0>─────────░─           q_1: |0>─────────░─
                         ░                             ░
        """
        qr = QuantumRegister(2, "q")

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.h(qr[0])
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr[0])
        expected.h(qr[0])
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_blocking_obstacle_narrow(self):
        """Test that barriers don't merge if there is an obstacle that
            is blocking
                 ░ ┌───┐ ░                     ░ ┌───┐ ░
        q_0: |0>─░─┤ H ├─░─           q_0: |0>─░─┤ H ├─░─
                 ░ └───┘ ░     ->              ░ └───┘ ░
        q_1: |0>─░───────░─           q_1: |0>─░───────░─
                 ░       ░                     ░       ░
        """
        qr = QuantumRegister(2, "q")

        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.h(qr[0])
        circuit.barrier(qr)

        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        expected.h(qr[0])
        expected.barrier(qr)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_blocking_obstacle_twoQ(self):
        """Test that barriers don't merge if there is an obstacle that
            is blocking

                 ░       ░                     ░       ░
        q_0: |0>─░───────░─           q_0: |0>─░───────░─
                 ░       ░                     ░       ░
        q_1: |0>─░───■─────    ->     q_1: |0>─░───■─────
                 ░ ┌─┴─┐ ░                     ░ ┌─┴─┐ ░
        q_2: |0>───┤ X ├─░─           q_2: |0>───┤ X ├─░─
                   └───┘ ░                       └───┘ ░

        """
        qr = QuantumRegister(3, "q")

        circuit = QuantumCircuit(qr)
        circuit.barrier(0, 1)
        circuit.cx(1, 2)
        circuit.barrier(0, 2)

        expected = QuantumCircuit(qr)
        expected.barrier(0, 1)
        expected.cx(1, 2)
        expected.barrier(0, 2)

        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_output_deterministic(self):
        """Test that the output barriers have a deterministic ordering (independent of
        PYTHONHASHSEED).  This is important to guarantee that any subsequent topological iterations
        through the circuit are also deterministic; it's in general not possible for all transpiler
        passes to produce identical outputs across all valid topological orderings, especially if
        those passes have some stochastic element."""
        order = list(range(20))
        random.Random(2023_02_10).shuffle(order)
        circuit = QuantumCircuit(20)
        circuit.barrier([5, 2, 3])
        circuit.barrier([7, 11, 14, 2, 4])
        circuit.barrier(order)

        # All the barriers should get merged together.
        expected = QuantumCircuit(20)
        expected.barrier(range(20))

        output = MergeAdjacentBarriers()(circuit)
        self.assertEqual(expected, output)
        # This assertion is that the ordering of the arguments in the barrier is fixed.
        self.assertEqual(list(output.data[0].qubits), list(output.qubits))


if __name__ == "__main__":
    unittest.main()
