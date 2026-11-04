# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test RemoveFinalReset pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveFinalReset, DAGFixedPoint
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase


class TestRemoveFinalReset(QiskitTestCase):
    """Test remove-reset-in-zero-state optimizations."""

    def test_optimize_single_reset(self):
        """Remove a single final reset
        qr0:--[H]--|0>--   ==>    qr0:--[H]--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(0)
        circuit.reset(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.h(0)

        pass_ = RemoveFinalReset()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_dont_optimize_non_final_reset(self):
        """Do not remove reset if not final instruction
        qr0:--|0>--[H]--   ==>    qr0:--|0>--[H]--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.reset(qr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.reset(qr)
        expected.h(qr)

        pass_ = RemoveFinalReset()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_single_reset_in_diff_qubits(self):
        """Remove a single final reset in different qubits
        qr0:--[H]--|0>--          qr0:--[H]--
                      ==>
        qr1:--[X]--|0>--          qr1:--[X]----
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(0)
        circuit.x(1)
        circuit.reset(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.h(0)
        expected.x(1)

        pass_ = RemoveFinalReset()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


class TestRemoveFinalResetFixedPoint(QiskitTestCase):
    """Test RemoveFinalReset in a transpiler, using fixed point."""

    def test_two_resets(self):
        """Remove two final resets
        qr0:--[H]-|0>-|0>--   ==>    qr0:--[H]--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.reset(qr[0])
        circuit.reset(qr[0])

        expected = QuantumCircuit(qr)
        expected.h(qr[0])

        pass_manager = PassManager()
        pass_manager.append(
            DoWhileController(
                [RemoveFinalReset(), DAGFixedPoint()],
                do_while=lambda property_set: not property_set["dag_fixed_point"],
            )
        )
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)


if __name__ == "__main__":
    unittest.main()
