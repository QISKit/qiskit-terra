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

"""Test RemoveResetInZeroState pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveResetInZeroState, DAGFixedPoint
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase


class TestRemoveResetInZeroState(QiskitTestCase):
    """Test remove-reset-in-zero-state optimizations."""

    def test_optimize_single_reset(self):
        """Remove a single reset
        qr0:--|0>--   ==>    qr0:----
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.reset(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)

        pass_ = RemoveResetInZeroState()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_dont_optimize_non_zero_state(self):
        """Do not remove reset if not in a zero state
        qr0:--[H]--|0>--   ==>    qr0:--[H]--|0>--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.reset(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr)
        expected.reset(qr)

        pass_ = RemoveResetInZeroState()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_single_reset_in_diff_qubits(self):
        """Remove a single reset in different qubits
        qr0:--|0>--          qr0:----
                      ==>
        qr1:--|0>--          qr1:----
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.reset(qr)
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)

        pass_ = RemoveResetInZeroState()
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


class TestRemoveResetInZeroStateFixedPoint(QiskitTestCase):
    """Test RemoveResetInZeroState in a transpiler, using fixed point."""

    def test_two_resets(self):
        """Remove two initial resets
        qr0:--|0>-|0>--   ==>    qr0:----
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.reset(qr[0])
        circuit.reset(qr[0])

        expected = QuantumCircuit(qr)

        pass_manager = PassManager()
        pass_manager.append(
            DoWhileController(
                [RemoveResetInZeroState(), DAGFixedPoint()],
                do_while=lambda property_set: not property_set["dag_fixed_point"],
            )
        )
        after = pass_manager.run(circuit)

        self.assertEqual(expected, after)


if __name__ == "__main__":
    unittest.main()
