# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tests AQC plugin.
"""
from functools import partial

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes.synthesis import AQCSynthesisPlugin
from test import QiskitTestCase


class TestAQCSynthesisPlugin(QiskitTestCase):
    """Basic tests of the AQC synthesis plugin."""

    def setUp(self):
        super().setUp()
        self._qc = QuantumCircuit(3)
        self._qc.mcx(
            [
                0,
                1,
            ],
            2,
        )

        self._target_unitary = Operator(self._qc).data
        self._seed_config = {"seed": 12345}

    def test_aqc_plugin(self):
        """Basic test of the plugin."""
        plugin = AQCSynthesisPlugin()
        dag = plugin.run(self._target_unitary, config=self._seed_config)

        approx_circuit = dag_to_circuit(dag)
        approx_unitary = Operator(approx_circuit).data

        np.testing.assert_array_almost_equal(self._target_unitary, approx_unitary, 3)

    def test_plugin_setup(self):
        """Tests the plugin via unitary synthesis pass"""
        transpiler_pass = UnitarySynthesis(
            basis_gates=["rx", "ry", "rz", "cx"], method="aqc", plugin_config=self._seed_config
        )

        dag = circuit_to_dag(self._qc)
        dag = transpiler_pass.run(dag)

        approx_circuit = dag_to_circuit(dag)
        approx_unitary = Operator(approx_circuit).data

        np.testing.assert_array_almost_equal(self._target_unitary, approx_unitary, 3)

    def test_plugin_configuration(self):
        """Tests plugin with a custom configuration."""
        optimizer = partial(minimize, args=(), method="SLSQP")
        config = {
            "network_layout": "sequ",
            "connectivity_type": "full",
            "depth": 0,
            "seed": 12345,
            "optimizer": optimizer,
        }
        transpiler_pass = UnitarySynthesis(
            basis_gates=["rx", "ry", "rz", "cx"], method="aqc", plugin_config=config
        )

        dag = circuit_to_dag(self._qc)
        dag = transpiler_pass.run(dag)

        approx_circuit = dag_to_circuit(dag)
        approx_unitary = Operator(approx_circuit).data

        np.testing.assert_array_almost_equal(self._target_unitary, approx_unitary, 3)

    def test_with_pass_manager(self):
        """Tests the plugin via pass manager"""
        qc = QuantumCircuit(3)
        qc.unitary(np.eye(8), [0, 1, 2])
        aqc = PassManager(
            [
                UnitarySynthesis(
                    basis_gates=["u", "cx"], method="aqc", plugin_config=self._seed_config
                )
            ]
        ).run(qc)
        approx_unitary = Operator(aqc).data
        np.testing.assert_array_almost_equal(np.eye(8), approx_unitary, 3)
