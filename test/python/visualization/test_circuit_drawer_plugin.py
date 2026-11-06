# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring

import unittest
from unittest.mock import patch, MagicMock

from qiskit import QuantumCircuit
from qiskit.visualization.circuit.plugin import (
    CircuitDrawerPluginManager,
    list_circuit_drawer_plugins,
    CircuitDrawerPlugin,
)


class MockedPlugin(CircuitDrawerPlugin):
    def draw(
        self,
        circuit: QuantumCircuit,
        scale=None,
        filename=None,
        style=None,
        output=None,
        interactive=False,
        plot_barriers=True,
        reverse_bits=None,
        justify=None,
        vertical_compression="medium",
        idle_wires=True,
        with_layout=True,
        fold=None,
        # The type of ax is matplotlib.axes.Axes, but this is not a fixed dependency, so cannot be
        # safely forward-referenced.
        ax=None,
        initial_state=False,
        cregbundle=None,
        wire_order=None,
        expr_len=30,
    ):
        return "Mocked image"


class TestCircuitDrawerPluginIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_extension_manager = MagicMock()
        self.mock_extension_manager.names.return_value = ["dummy"]
        self.patcher = patch("stevedore.ExtensionManager", return_value=self.mock_extension_manager)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_get_drawer_success(self):
        mock_plugin_instance = MagicMock(name="dummy_instance")
        self.mock_extension_manager.__getitem__.return_value.plugin = mock_plugin_instance

        manager = CircuitDrawerPluginManager()
        plugin = manager.get_drawer("dummy")
        self.assertIs(plugin, mock_plugin_instance)

    def test_get_drawer_failure(self):
        manager = CircuitDrawerPluginManager()
        with self.assertRaises(ValueError) as context:
            manager.get_drawer("nonexistent")
        self.assertIn("Circuit drawer plugin 'nonexistent' not found", str(context.exception))

    def test_list_circuit_drawer_plugins(self):
        plugins = list_circuit_drawer_plugins()
        self.assertListEqual(plugins, ["dummy"])

    def test_quantumcircuit_plugin_drawer(self):
        plugin_manager = CircuitDrawerPluginManager()
        plugin_names = plugin_manager.drawer_plugins.names()
        self.assertIn("dummy", plugin_names)

        with patch.object(
            CircuitDrawerPluginManager, "get_drawer", return_value=MockedPlugin
        ) as mock_get_drawer:
            qc = QuantumCircuit(2)
            image = qc.draw("dummy")
            self.assertEqual(image, "Mocked image")
            mock_get_drawer.assert_called_once_with("dummy")


if __name__ == "__main__":
    unittest.main()
