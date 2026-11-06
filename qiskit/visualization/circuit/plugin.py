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

"""
================================================================================
= Circuit Drawer Plugin Interface (:mod:`qiskit.visualization.circuit.plugin`) =
================================================================================

.. currentmodule:: qiskit.visualization.circuit.plugin.

This module defines the plugin interface for providing custom circuit drawers for
:func:`~.circuit_drawer` function.
This enables external python packages to provide
:class:`~CircuitDrawerPlugin` objects that can be used as a circuit drawing tool.

The plugin interfaces are built using setuptools
`entry points <https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html>`__
which enables external packages to introduce Qiskit circuit drawer(s).

Writing Plugins
===============

To write a new drawer plugin, you need to do the following:
1. Create a subclass of :class:`~qiskit.visualization.circuit.plugin.CircuitDrawerPlugin
The plugin class requires draw method to be implemented which will be drawing the provided circuit.
:meth:`~qiskit.visualization.circuit.plugin.CircuitDrawerPlugin.draw`
The only parameter that it takes is :class:`~qiskit.circuit.quantumcircuit.QuantumCircuit` object
representing the circuit which will be drawn.
2. expose :class:`~qiskit.visualization.circuit.plugin.CircuitDrawerPlugin` as a setuptools
entry point in the package metadata. This is done by simply adding
an ``entry-points`` table in ``pyproject.toml`` for the plugin package with the
necessary entry points under the ``qiskit.circuit_drawer`` namespace. for example:

.. code-block:: toml

    [project.entry-points."qiskit.circuit_drawer"]
    unique = "qiskit_plugin_pkg.plugin:UniqueDrawerPlugin"

There isn't a limit to the number of plugins a single package can
include as long as each plugin has a unique name. So a single package can
expose multiple plugins if necessary.

beware that the drawer already has the following options:
``text``, ``mpl``, ``latex``, ``latex_source``
so make sure that you do not overwrite any of them.

"""

import abc

import stevedore


class CircuitDrawerPlugin(abc.ABC):
    """
    Abstract Circuit Drawer plugin class

    this defines the interface for circuit drawer plugins
    """

    @abc.abstractmethod
    def draw(
        self,
        circuit,
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
        """
        Return an Image which represents the provided Quantum Circuit
        """
        pass


class CircuitDrawerPluginManager:
    """
    Manager class for circuit drawer plugins.

    This class tracks the installed plugins, it has a single property,
    ``drawer_plugins`` which contains a list of stevedore plugin objects.
    """

    def __init__(self):
        super().__init__()
        self.drawer_plugins = stevedore.ExtensionManager(
            "qiskit.circuit_drawer", invoke_on_load=True, propagate_map_exceptions=True
        )

    def get_drawer(self, name):
        """
        Get a circuit drawer plugin by name.
        """
        if name in self.drawer_plugins.names():
            return self.drawer_plugins[name].plugin
        raise ValueError(f"Circuit drawer plugin '{name}' not found")


def list_circuit_drawer_plugins():
    """
    Get a list of installed plugins for circuit drawer

    Returns:
        The list of installed plugin names for the circuit_drawer namespace
    """
    plugin_manager = CircuitDrawerPluginManager()
    return plugin_manager.drawer_plugins.names()
