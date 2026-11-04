# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
====================================================================
Synthesis Plugins (:mod:`qiskit.transpiler.passes.synthesis.plugin`)
====================================================================

.. currentmodule:: qiskit.transpiler.passes.synthesis.plugin

This module defines the plugin interfaces for the synthesis transpiler passes
in Qiskit. These provide a hook point for external python packages to implement
their own synthesis techniques and have them seamlessly exposed as opt-in
options to users when they run :func:`~qiskit.compiler.transpile`.

The plugin interfaces are built using setuptools
`entry points <https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html>`__
which enable packages external to qiskit to advertise they include a synthesis
plugin.

See :mod:`qiskit.transpiler.preset_passmanagers.plugin` for details on how
to write plugins for transpiler stages.

Synthesis Plugin API
====================

Unitary Synthesis Plugin API
----------------------------

.. autosummary::
   :toctree: ../stubs/

   UnitarySynthesisPlugin
   UnitarySynthesisPluginManager
   unitary_synthesis_plugin_names

High-Level Synthesis Plugin API
-------------------------------

.. autosummary::
   :toctree: ../stubs/

   HighLevelSynthesisPlugin
   HighLevelSynthesisPluginManager
   high_level_synthesis_plugin_names

Writing Plugins
===============

Unitary Synthesis Plugins
-------------------------

To write a unitary synthesis plugin there are 2 main steps. The first step is
to create a subclass of the abstract plugin class:
:class:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin`.
The plugin class defines the interface and contract for unitary synthesis
plugins. The primary method is
:meth:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin.run`
which takes in a single positional argument, a unitary matrix as a numpy array,
and is expected to return a :class:`~qiskit.dagcircuit.DAGCircuit` object
representing the synthesized circuit from that unitary matrix. Then to inform
the Qiskit transpiler about what information is necessary for the pass there
are several required property methods that need to be implemented such as
``supports_basis_gates`` and ``supports_coupling_map`` depending on whether the
plugin supports and/or requires that input to perform synthesis. For the full
details refer to the
:class:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin`
documentation for all the required fields. An example plugin class would look
something like::

    from qiskit.transpiler.passes.synthesis import plugin
    from qiskit_plugin_pkg.synthesis import generate_dag_circuit_from_matrix


    class SpecialUnitarySynthesis(plugin.UnitarySynthesisPlugin):
        @property
        def supports_basis_gates(self):
            return True

        @property
        def supports_coupling_map(self):
            return False

        @property
        def supports_natural_direction(self):
            return False

        @property
        def supports_pulse_optimize(self):
            return False

        @property
        def supports_gate_lengths(self):
            return False

        @property
        def supports_gate_errors(self):
            return False

        @property
        def supports_gate_lengths_by_qubit(self):
            return False

        @property
        def supports_gate_errors_by_qubit(self):
            return False

        @property
        def min_qubits(self):
            return None

        @property
        def max_qubits(self):
            return None

        @property
        def supported_bases(self):
            return None

        def run(self, unitary, **options):
            basis_gates = options['basis_gates']
            dag_circuit = generate_dag_circuit_from_matrix(unitary, basis_gates)
            return dag_circuit

If for some reason the available inputs to the
:meth:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin.run`
method are insufficient please open an issue and we can discuss expanding the
plugin interface with new opt-in inputs that can be added in a backwards
compatible manner for future releases. Do note though that this plugin interface
is considered stable and guaranteed to not change in a breaking manner. If
changes are needed (for example to expand the available optional input options)
it will be done in a way that will **not** require changes from existing
plugins.

.. note::

    All methods prefixed with ``supports_`` are reserved on a
    ``UnitarySynthesisPlugin`` derived class for part of the interface. You
    should not define any custom ``supports_*`` methods on a subclass that
    are not defined in the abstract class.


The second step is to expose the
:class:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin` as
a setuptools entry point in the package metadata. This is done by simply adding
an ``entry-points`` table in ``pyproject.toml`` for the plugin package with the necessary entry
points under the ``qiskit.unitary_synthesis`` namespace. For example:

.. code-block:: toml

    [project.entry-points."qiskit.unitary_synthesis"]
    "special" = "qiskit_plugin_pkg.module.plugin:SpecialUnitarySynthesis"

There isn't a limit to the number of plugins a single package can
include as long as each plugin has a unique name. So a single package can
expose multiple plugins if necessary. The name ``default`` is used by Qiskit
itself and can't be used in a plugin.

Unitary Synthesis Plugin Configuration
''''''''''''''''''''''''''''''''''''''

For some unitary synthesis plugins that expose multiple options and tunables
the plugin interface has an option for users to provide a free form
configuration dictionary. This will be passed through to the ``run()`` method
as the ``options`` kwarg. If your plugin has these configuration options you
should clearly document how a user should specify these configuration options
and how they're used as it's a free form field.

High-Level Synthesis Plugins
----------------------------

Writing a high-level synthesis plugin is conceptually similar to writing a
unitary synthesis plugin. The first step is to create a subclass of the
abstract plugin class:
:class:`~qiskit.transpiler.passes.synthesis.plugin.HighLevelSynthesisPlugin`,
which defines the interface and contract for high-level synthesis plugins.
The primary method is
:meth:`~qiskit.transpiler.passes.synthesis.plugin.HighLevelSynthesisPlugin.run`.
The positional argument ``high_level_object`` specifies the "higher-level-object" to
be synthesized, which is any object of type :class:`~qiskit.circuit.Operation`
(including, for example,
:class:`~qiskit.circuit.library.generalized_gates.linear_function.LinearFunction` or
:class:`~qiskit.quantum_info.operators.symplectic.clifford.Clifford`).
The keyword argument ``target`` specifies the target backend, allowing the plugin
to access all target-specific information,
such as the coupling map, the supported gate set, and so on. The keyword argument
``coupling_map`` only specifies the coupling map, and is only used when ``target``
is not specified.
The keyword argument ``qubits`` specifies the list of qubits over which the
higher-level-object is defined, in case the synthesis is done on the physical circuit.
The value of ``None`` indicates that the layout has not yet been chosen and the physical qubits
in the target or coupling map that this operation is operating on has not yet been determined.
Additionally, plugin-specific options and tunables can be specified via ``options``,
which is a free form configuration dictionary.
If your plugin has these configuration options you
should clearly document how a user should specify these configuration options
and how they're used as it's a free form field.
The method
:meth:`~qiskit.transpiler.passes.synthesis.plugin.HighLevelSynthesisPlugin.run`
is expected to return a :class:`~qiskit.circuit.QuantumCircuit` object
representing the synthesized circuit from that higher-level-object.
It is also allowed to return ``None`` representing that the synthesis method is
unable to synthesize the given higher-level-object.
The actual synthesis of higher-level objects is performed by
:class:`~qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesis`
transpiler pass.
For the full details refer to the
:class:`~qiskit.transpiler.passes.synthesis.plugin.HighLevelSynthesisPlugin`
documentation for all the required fields. An example plugin class would look
something like::

    from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPlugin
    from qiskit.synthesis.clifford import synth_clifford_bm


    class SpecialSynthesisClifford(HighLevelSynthesisPlugin):

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if higher_level_object.num_qubits <= 3:
            return synth_clifford_bm(high_level_object)
        else:
            return None

The above example creates a plugin to synthesize objects of type
:class:`.Clifford` that have
at most 3 qubits, using the method ``synth_clifford_bm``.

The second step is to expose the
:class:`~qiskit.transpiler.passes.synthesis.plugin.HighLevelSynthesisPlugin` as
a setuptools entry point in the package metadata. This is done by adding
an ``entry-points`` table in ``pyproject.toml`` for the plugin package with the necessary entry
points under the ``qiskit.synthesis`` namespace. For example:

.. code-block:: toml

    [project.entry-points."qiskit.synthesis"]
    "clifford.special" = "qiskit_plugin_pkg.module.plugin:SpecialSynthesisClifford"

The ``name`` consists of two parts separated by dot ".": the name of the
type of :class:`~qiskit.circuit.Operation` to which the synthesis plugin applies
(``clifford``), and the name of the plugin (``special``).
There isn't a limit to the number of plugins a single package can
include as long as each plugin has a unique name.

Using Plugins
=============

.. _unitary-synth-plugin:

Unitary Synthesis Plugins
-------------------------

To use a plugin all you need to do is install the package that includes a
synthesis plugin. Then Qiskit will automatically discover the installed
plugins and expose them as valid options for the appropriate
:func:`~qiskit.compiler.transpile` kwargs and pass constructors. If there are
any installed plugins which can't be loaded/imported this will be logged to
Python logging.

To get the installed list of installed unitary synthesis plugins you can use the
:func:`qiskit.transpiler.passes.synthesis.plugin.unitary_synthesis_plugin_names`
function.

.. _using-high-level-synthesis-plugins:

High-level Synthesis Plugins
----------------------------

To use a high-level synthesis plugin, you first instantiate an :class:`.HLSConfig` to
store the names of the plugins to use for various high-level objects.
For example::

    HLSConfig(permutation=["acg"], clifford=["layers"], linear_function=["pmh"])

creates a high-level synthesis configuration that uses the ``acg`` plugin
for synthesizing :class:`.PermutationGate` objects, the ``layers`` plugin
for synthesizing :class:`.Clifford` objects, and the ``pmh`` plugin for synthesizing
:class:`.LinearFunction` objects.  The keyword arguments are the :attr:`.Operation.name` fields of
the relevant objects.  For example, all :class:`.Clifford` operations have the
:attr:`~.Operation.name` ``clifford``, so this is used as the keyword argument.  You can specify
any keyword argument here that you have installed plugins to handle, including custom user objects
if you have plugins installed for them.  See :class:`.HLSConfig` for more detail on alternate
formats for configuring the plugins within each argument.

For each high-level object, the list of given plugins are tried in sequence until one of them
succeeds (in the example above, each list only contains a single plugin). In addition to specifying
a plugin by its name, you can instead pass a ``(name, options)`` tuple, where the second element of
the tuple is a dictionary containing options for the plugin.

Once created you then pass this :class:`.HLSConfig` object into the
``hls_config`` argument for :func:`.transpile` or :func:`.generate_preset_pass_manager`
which will use the specified plugins as part of the larger compilation workflow.

To get a list of installed high level synthesis plugins for any given :attr:`.Operation.name`, you
can use the :func:`.high_level_synthesis_plugin_names` function, passing the desired ``name`` as the
argument::

    high_level_synthesis_plugin_names("clifford")

will return a list of all the installed Clifford synthesis plugins.

Available Plugins
=================

Below are the synthesis plugin classes available in Qiskit. These classes should not be
used directly, but instead should be used through the plugin interface documented
above. The classes are listed here to ease finding the documentation for each of the
included plugins and to ease the comparison between different synthesis methods for
a given object.


Unitary Synthesis Plugins
-------------------------

.. automodule:: qiskit.transpiler.passes.synthesis.aqc_plugin
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. automodule:: qiskit.transpiler.passes.synthesis.solovay_kitaev_synthesis
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. automodule:: qiskit.transpiler.passes.synthesis.ross_selinger_plugin
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. automodule:: qiskit.transpiler.passes.synthesis.clifford_unitary_synth_plugin
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. automodule:: qiskit.transpiler.passes.synthesis.default_unitary_synth_plugin
   :no-members:
   :no-inherited-members:
   :no-special-members:


High Level Synthesis
--------------------

High-level synthesis plugins that are directly available in Qiskit include plugins
for synthesizing :class:`.Clifford` objects, :class:`.LinearFunction` objects, and
:class:`.PermutationGate` objects.
Some of these plugins implicitly target all-to-all connectivity. This is not a
practical limitation since
:class:`~qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesis`
typically runs before layout and routing, which will ensure that the final circuit
adheres to the device connectivity by inserting additional SWAP gates. A good example
is the permutation synthesis plugin ``ACGSynthesisPermutation`` which can synthesize
any permutation with at most 2 layers of SWAP gates.
On the other hand, some plugins implicitly target linear connectivity.
Typically, the synthesizing circuits have larger depth and the number of gates,
however no additional SWAP gates would be inserted if the following layout pass chose a
consecutive line of qubits inside the topology of the device. A good example of this is
the permutation synthesis plugin ``KMSSynthesisPermutation`` which can synthesize any
permutation of ``n`` qubits in depth ``n``. Typically, it is difficult to know in advance
which of the two approaches: synthesizing circuits for all-to-all connectivity and
inserting SWAP gates vs. synthesizing circuits for linear connectivity and inserting less
or no SWAP gates lead a better final circuit, so it likely makes sense to try both and
see which gives better results.
Finally, some plugins can target a given connectivity, and hence should be run after the
layout is set. In this case the synthesized circuit automatically adheres to
the topology of the device. A good example of this is the permutation synthesis plugin
``TokenSwapperSynthesisPermutation`` which is able to synthesize arbitrary permutations
with respect to arbitrary coupling maps.
For more detail, please refer to description of each individual plugin.

For each high-level object we give a table that lists all of its plugins available
directly in Qiskit. We include the name of the plugin, the class of the plugin,
the targeted connectivity map and optionally additional information. Recall the plugins
should be used via the previously described :class:`.HLSConfig`, for example::

    HLSConfig(permutation=["kms"])

creates a high-level synthesis configuration that uses the ``kms`` plugin
for synthesizing :class:`.PermutationGate` objects -- i.e. those with
``name = "permutation"``. In this case, the plugin name is "kms", the plugin class
is :class:`~.KMSSynthesisPermutation`. This particular synthesis algorithm created
a circuit adhering to the linear nearest-neighbor connectivity.

.. automodule:: qiskit.transpiler.passes.synthesis.hls_plugins
   :no-members:
   :no-inherited-members:
   :no-special-members:
"""

import abc

import stevedore


class UnitarySynthesisPlugin(abc.ABC):
    """Abstract unitary synthesis plugin class

    This abstract class defines the interface for unitary synthesis plugins.
    """

    @property
    @abc.abstractmethod
    def max_qubits(self):
        """Return the maximum number of qubits the unitary synthesis plugin supports.

        If the size of the unitary to be synthesized exceeds this value the
        ``default`` plugin will be used. If there is no upper bound return
        ``None`` and all unitaries (``>= min_qubits`` if it's defined) will be
        passed to this plugin when it's enabled.
        """

    @property
    @abc.abstractmethod
    def min_qubits(self):
        """Return the minimum number of qubits the unitary synthesis plugin supports.

        If the size of the unitary to be synthesized is below this value the
        ``default`` plugin will be used. If there is no lower bound return
        ``None`` and all unitaries (``<= max_qubits`` if it's defined) will be
        passed to this plugin when it's enabled.
        """

    @property
    @abc.abstractmethod
    def supports_basis_gates(self):
        """Return whether the plugin supports taking ``basis_gates``

        If this returns ``True`` the plugin's ``run()`` method will be
        passed a ``basis_gates`` kwarg with a list of gate names the target
        backend supports. For example, ``['sx', 'x', 'cx', 'id', 'rz']``.
        """

    @property
    @abc.abstractmethod
    def supports_coupling_map(self):
        """Return whether the plugin supports taking ``coupling_map``

        If this returns ``True`` the plugin's ``run()`` method will receive
        one kwarg ``coupling_map``. The ``coupling_map`` kwarg will be set to a
        tuple with the first element being a
        :class:`~qiskit.transpiler.CouplingMap` object representing the qubit
        connectivity of the target backend, the second element will be a list
        of integers that represent the qubit indices in the coupling map that
        unitary is on. Note that if the target backend doesn't have a coupling
        map set, the ``coupling_map`` kwarg's value will be ``(None, qubit_indices)``.
        """

    @property
    @abc.abstractmethod
    def supports_natural_direction(self):
        """Return whether the plugin supports a toggle for considering
        directionality of 2-qubit gates as ``natural_direction``.

        Refer to the documentation for :class:`~qiskit.transpiler.passes.UnitarySynthesis`
        for the possible values and meaning of these values.
        """

    @property
    @abc.abstractmethod
    def supports_pulse_optimize(self):
        """Return whether the plugin supports a toggle to optimize pulses
        during synthesis as ``pulse_optimize``.

        Refer to the documentation for :class:`~qiskit.transpiler.passes.UnitarySynthesis`
        for the possible values and meaning of these values.
        """

    @property
    def supports_gate_lengths_by_qubit(self):
        """Return whether the plugin supports taking ``gate_lengths_by_qubit``

        This differs from ``supports_gate_lengths``/``gate_lengths`` by using a different
        view of the same data. Instead of being keyed by gate name this is keyed by qubit
        and uses :class:`~.Gate` instances to represent gates (instead of gate names)

        ``gate_lengths_by_qubit`` will be a dictionary in the form of
        ``{(qubits,): [Gate, length]}``. For example::

            {
            (0,): [SXGate(): 0.0006149355812506126, RZGate(): 0.0],
            (0, 1): [CXGate(): 0.012012477900732316]
            }

        where the ``length`` value is in units of seconds.

        Do note that this dictionary might not be complete or could be empty
        as it depends on the target backend reporting gate lengths on every
        gate for each qubit.

        This defaults to False
        """
        return False

    @property
    def supports_gate_errors_by_qubit(self):
        """Return whether the plugin supports taking ``gate_errors_by_qubit``

        This differs from ``supports_gate_errors``/``gate_errors`` by using a different
        view of the same data. Instead of being keyed by gate name this is keyed by qubit
        and uses :class:`~.Gate` instances to represent gates (instead of gate names).

        ``gate_errors_by_qubit`` will be a dictionary in the form of
        ``{(qubits,): [Gate, error]}``. For example::

            {
            (0,): [SXGate(): 0.0006149355812506126, RZGate(): 0.0],
            (0, 1): [CXGate(): 0.012012477900732316]
            }

        Do note that this dictionary might not be complete or could be empty
        as it depends on the target backend reporting gate errors on every
        gate for each qubit. The gate error rates reported in ``gate_errors``
        are provided by the target device ``Backend`` object and the exact
        meaning might be different depending on the backend.

        This defaults to False
        """
        return False

    @property
    @abc.abstractmethod
    def supports_gate_lengths(self):
        """Return whether the plugin supports taking ``gate_lengths``

        ``gate_lengths`` will be a dictionary in the form of
        ``{gate_name: {(qubit_1, qubit_2): length}}``. For example::

            {
            'sx': {(0,): 0.0006149355812506126, (1,): 0.0006149355812506126},
            'cx': {(0, 1): 0.012012477900732316, (1, 0): 5.191111111111111e-07}
            }

        where the ``length`` value is in units of seconds.

        Do note that this dictionary might not be complete or could be empty
        as it depends on the target backend reporting gate lengths on every
        gate for each qubit.
        """

    @property
    @abc.abstractmethod
    def supports_gate_errors(self):
        """Return whether the plugin supports taking ``gate_errors``

        ``gate_errors`` will be a dictionary in the form of
        ``{gate_name: {(qubit_1, qubit_2): error}}``. For example::

            {
            'sx': {(0,): 0.0006149355812506126, (1,): 0.0006149355812506126},
            'cx': {(0, 1): 0.012012477900732316, (1, 0): 5.191111111111111e-07}
            }

        Do note that this dictionary might not be complete or could be empty
        as it depends on the target backend reporting gate errors on every
        gate for each qubit. The gate error rates reported in ``gate_errors``
        are provided by the target device ``Backend`` object and the exact
        meaning might be different depending on the backend.
        """

    @property
    @abc.abstractmethod
    def supported_bases(self):
        """Returns a dictionary of supported bases for synthesis

        This is expected to return a dictionary where the key is a string
        basis and the value is a list of gate names that the basis works in.
        If the synthesis method doesn't support multiple bases this should
        return ``None``. For example::

            {
                "XZX": ["rz", "rx"],
                "XYX": ["rx", "ry"],
            }

        If a dictionary is returned by this method the run kwargs will be
        passed a parameter ``matched_basis`` which contains a list of the
        basis strings (i.e. keys in the dictionary) which match the target basis
        gate set for the transpilation. If no entry in the dictionary matches
        the target basis gate set then the ``matched_basis`` kwarg will be set
        to an empty list, and a plugin can choose how to deal with the target
        basis gate set not matching the plugin's capabilities.
        """

    @property
    def supports_target(self):
        """Whether the plugin supports taking ``target`` as an option

        ``target`` will be a :class:`~.Target` object representing the target
        device for the output of the synthesis pass.

        By default this will be ``False`` since the plugin interface predates
        the :class:`~.Target` class. If a plugin returns ``True`` for this
        attribute, it is expected that the plugin will use the
        :class:`~.Target` instead of the values passed if any of
        ``supports_gate_lengths``, ``supports_gate_errors``,
        ``supports_coupling_map``, and ``supports_basis_gates`` are set
        (although ideally all those parameters should contain duplicate
        information).
        """
        return False

    @abc.abstractmethod
    def run(self, unitary, **options):
        """Run synthesis for the given unitary matrix

        Args:
            unitary (numpy.ndarray): The unitary matrix to synthesize to a
                :class:`~qiskit.dagcircuit.DAGCircuit` object
            options: The optional kwargs that are passed based on the output
                the ``support_*`` methods on the class. Refer to the
                documentation for these methods on
                :class:`~qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin`
                to see what the keys and values are.

        Returns:
            DAGCircuit: The dag circuit representation of the unitary. Alternatively, you can return
            a tuple of the form ``(dag, wires)`` where ``dag`` is the dag circuit representation of
            the circuit representation of the unitary and ``wires`` is the mapping wires to use for
            :meth:`qiskit.dagcircuit.DAGCircuit.substitute_node_with_dag`. If you return a tuple
            and ``wires`` is ``None`` this will behave just as if only a
            :class:`~qiskit.dagcircuit.DAGCircuit` was returned. Additionally if this returns
            ``None`` no substitution will be made.

        """


class UnitarySynthesisPluginManager:
    """Unitary Synthesis plugin manager class

    This class tracks the installed plugins, it has a single property,
    ``ext_plugins`` which contains a list of stevedore plugin objects.
    """

    def __init__(self):
        self.ext_plugins = stevedore.ExtensionManager(
            "qiskit.unitary_synthesis", invoke_on_load=True, propagate_map_exceptions=True
        )


def unitary_synthesis_plugin_names():
    """Return a list of installed unitary synthesis plugin names

    Returns:
        list: A list of the installed unitary synthesis plugin names. The plugin names are valid
        values for the :func:`~qiskit.compiler.transpile` kwarg ``unitary_synthesis_method``.
    """
    # NOTE: This is not a shared global instance to avoid an import cycle
    # at load time for the default plugin.
    plugins = UnitarySynthesisPluginManager()
    return plugins.ext_plugins.names()


class HighLevelSynthesisPlugin(abc.ABC):
    """Abstract high-level synthesis plugin class.

    This abstract class defines the interface for high-level synthesis plugins.
    """

    @abc.abstractmethod
    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Operation.

        Args:
            high_level_object (Operation): The Operation to synthesize to a
                :class:`~qiskit.dagcircuit.DAGCircuit` object.
            coupling_map (CouplingMap): The coupling map of the backend
                in case synthesis is done on a physical circuit.
            target (Target): A target representing the target backend.
            qubits (list): List of qubits over which the operation is defined
                in case synthesis is done on a physical circuit.
            options: Additional method-specific optional kwargs.

        Returns:
            QuantumCircuit: The quantum circuit representation of the Operation
                when successful, and ``None`` otherwise.
        """


class HighLevelSynthesisPluginManager:
    """Class tracking the installed high-level-synthesis plugins."""

    def __init__(self):
        self.plugins = stevedore.ExtensionManager(
            "qiskit.synthesis", invoke_on_load=True, propagate_map_exceptions=True
        )

        # The registered plugin names should be of the form <OperationName.SynthesisMethodName>.

        # Create a dict, mapping <OperationName> to the list of its <SynthesisMethodName>s.
        self.plugins_by_op = {}
        for plugin_name in self.plugins.names():
            op_name, method_name = plugin_name.split(".")
            if op_name not in self.plugins_by_op:
                self.plugins_by_op[op_name] = []
            self.plugins_by_op[op_name].append(method_name)

    def method_names(self, op_name):
        """Returns plugin methods for op_name."""
        if op_name in self.plugins_by_op:
            return self.plugins_by_op[op_name]
        else:
            return []

    def method(self, op_name, method_name):
        """Returns the plugin for ``op_name`` and ``method_name``."""
        plugin_name = op_name + "." + method_name
        return self.plugins[plugin_name].obj

    def op_names(self):
        """Returns the names of high-level-objects with available synthesis methods."""
        return list(self.plugins_by_op.keys())


def high_level_synthesis_plugin_names(op_name: str) -> list[str]:
    """Return a list of plugin names installed for a given high level object name

    Args:
        op_name: The operation name to find the installed plugins for. For example,
            if you provide ``"clifford"`` as the input it will find all the installed
            clifford synthesis plugins that can synthesize :class:`.Clifford` objects.
            The name refers to the :attr:`.Operation.name` attribute of the relevant objects.

    Returns:
        A list of installed plugin names for the specified high level operation

    """
    # NOTE: This is not a shared global instance to avoid an import cycle
    # at load time for the default plugins.
    plugin_manager = HighLevelSynthesisPluginManager()
    return plugin_manager.method_names(op_name)
