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
Tests for the UnitarySynthesis transpiler pass.
"""

import functools
import itertools
import unittest.mock

import numpy as np
import stevedore

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager, Target, CouplingMap
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes.synthesis.plugin import (
    UnitarySynthesisPlugin,
    UnitarySynthesisPluginManager,
    unitary_synthesis_plugin_names,
)
from qiskit.transpiler.passes.synthesis.default_unitary_synth_plugin import DefaultUnitarySynthesis
from test import QiskitTestCase


class _MockExtensionManager:
    def __init__(self, plugins):
        self._plugins = {
            name: stevedore.extension.Extension(name, None, plugin, plugin())
            for name, plugin in plugins.items()
        }
        self._stevedore_manager = stevedore.ExtensionManager(
            "qiskit.unitary_synthesis", invoke_on_load=True, propagate_map_exceptions=True
        )

    def names(self):
        """Mock method to replace the stevedore names."""
        return list(self._plugins) + self._stevedore_manager.names()

    def __getitem__(self, value):
        try:
            return self._plugins[value]
        except KeyError:
            pass
        return self._stevedore_manager[value]

    def __contains__(self, value):
        return value in self._plugins or value in self._stevedore_manager

    def __iter__(self):
        return itertools.chain(self._plugins.values(), self._stevedore_manager)


class _MockPluginManager:
    def __init__(self, plugins):
        self.ext_plugins = _MockExtensionManager(plugins)


class ControllableSynthesis(UnitarySynthesisPlugin):
    """A dummy synthesis plugin, which can have its ``supports_`` properties changed to test
    different parts of the synthesis plugin interface.  By default, it accepts all keyword arguments
    and accepts all number of qubits, but if its run method is called, it just returns ``None`` to
    indicate that the gate should not be synthesized."""

    min_qubits = None
    max_qubits = None
    supported_bases = None
    supports_basis_gates = True
    supports_coupling_map = True
    supports_gate_errors = True
    supports_gate_lengths = True
    supports_natural_direction = True
    supports_pulse_optimize = True
    run = unittest.mock.MagicMock(return_value=None)

    @classmethod
    def reset(cls):
        """Reset the state of any internal mocks, and return class properties to their defaults."""
        cls.run.reset_mock()
        cls.min_qubits = None
        cls.max_qubits = None
        cls.supported_bases = None
        cls.support()

    @classmethod
    def support(cls, names=None):
        """Set the plugin to support the given keywords, and reject any that are not given.  If
        no argument is passed, then everything will be supported.  To reject everything, explicitly
        pass an empty iterable."""
        if names is None:

            def value(_name):
                return True

        else:
            names = set(names)

            def value(name):
                return name in names

        prefix = "supports_"
        for name in dir(cls):
            if name.startswith(prefix):
                setattr(cls, name, value(name[len(prefix) :]))


class TestUnitarySynthesisPlugin(QiskitTestCase):
    """Tests for the synthesis plugin interface."""

    # The proliferation of the "disable=no-members" lines are because pylint (very reasonably) can't
    # detect when we've mocked out an internal object with a MagicMock.

    MOCK_PLUGINS = {}
    DEFAULT_PLUGIN = DefaultUnitarySynthesis

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.MOCK_PLUGINS["_controllable"] = ControllableSynthesis
        # Decorate all test functions to patch the plugin manager with a fake one, which inserts our
        # testing mocks into the lookup.  It's easier to do this than to correctly register them,
        # because it's fiddly to trick importlib into registering new entry points metadata without
        # actually installing a package before the interpreter was invoked.  We don't want to leak
        # any testing classes out into public releases, so we just patch.
        decorator = unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPluginManager",
            functools.partial(_MockPluginManager, plugins=cls.MOCK_PLUGINS),
        )
        for name in dir(cls):
            if name.startswith("test_"):
                setattr(cls, name, decorator(getattr(cls, name)))

    def setUp(self):
        super().setUp()
        for plugin in self.MOCK_PLUGINS.values():
            plugin.reset()

    def mock_default_run_method(self):
        """Return a decorator or context manager that replaces the default synthesis plugin's run
        method with a mocked version that behaves normally, except has all the trackers attached to
        it."""
        # We need to mock out DefaultUnitarySynthesis.run, except it will actually get called as an
        # instance method, so we can't just wrap the method defined on the class, but instead we
        # need to wrap a method that has been bound to a particular instance.  This is slightly
        # fragile, because we're likely wrapping a _different_ instance, but since there are no
        # arguments to __init__, and no internal state, it should be ok.  It doesn't matter if we
        # dodged the patching of the manager class that happens elsewhere in this test suite,
        # because we're always accessing something that the patch would delegate to the inner
        # manager anyway.
        inner_default = UnitarySynthesisPluginManager().ext_plugins["default"].obj
        mock = unittest.mock.MagicMock(wraps=inner_default.run)
        return unittest.mock.patch.object(self.DEFAULT_PLUGIN, "run", mock)

    def test_mock_plugins_registered(self):
        """This is a meta test, that the internal registering mechanisms for our dummy test plugins
        exist and that we can call them."""
        registered = unitary_synthesis_plugin_names()
        for plugin in self.MOCK_PLUGINS:
            self.assertIn(plugin, registered)

    def test_call_registered_class(self):
        """Test that a non-default plugin was called."""
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        pm = PassManager([UnitarySynthesis(basis_gates=["u", "cx"], method="_controllable")])
        with self.mock_default_run_method() as mocked:
            _ = pm.run(qc)
            mocked.assert_not_called()
        self.MOCK_PLUGINS["_controllable"].run.assert_called()

    def test_max_qubits_are_respected(self):
        """Test that the default handler gets used if the chosen plugin can't cope with a given
        unitary."""
        self.MOCK_PLUGINS["_controllable"].min_qubits = None
        self.MOCK_PLUGINS["_controllable"].max_qubits = 0
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        pm = PassManager([UnitarySynthesis(basis_gates=["u", "cx"], method="_controllable")])
        with self.mock_default_run_method() as mocked:
            _ = pm.run(qc)
            mocked.assert_called()
        self.MOCK_PLUGINS["_controllable"].run.assert_not_called()

    def test_min_qubits_are_respected(self):
        """Test that the default handler gets used if the chosen plugin can't cope with a given
        unitary."""
        self.MOCK_PLUGINS["_controllable"].min_qubits = 3
        self.MOCK_PLUGINS["_controllable"].max_qubits = None
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        pm = PassManager([UnitarySynthesis(basis_gates=["u", "cx"], method="_controllable")])
        with self.mock_default_run_method() as mocked:
            _ = pm.run(qc)
            mocked.assert_called()
        self.MOCK_PLUGINS["_controllable"].run.assert_not_called()

    def test_all_keywords_passed_to_default_on_fallback(self):
        """Test that all the keywords that the default synthesis plugin needs are passed to it, even
        if the chosen method doesn't support them."""
        # Set the mock plugin to reject all keyword arguments, but also be unable to handle
        # operators of any numbers of qubits.  This will cause fallback to the default handler,
        # which should receive a full set of keywords, still.
        self.MOCK_PLUGINS["_controllable"].min_qubits = np.inf
        self.MOCK_PLUGINS["_controllable"].max_qubits = 0
        self.MOCK_PLUGINS["_controllable"].support([])
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        pm = PassManager([UnitarySynthesis(basis_gates=["u", "cx"], method="_controllable")])
        with self.mock_default_run_method() as mocked:
            _ = pm.run(qc)
            mocked.assert_called()
            call_kwargs = mocked.call_args.kwargs
        expected_kwargs = [
            "basis_gates",
            "coupling_map",
            "gate_errors_by_qubit",
            "gate_lengths_by_qubit",
            "natural_direction",
            "pulse_optimize",
        ]
        for kwarg in expected_kwargs:
            self.assertIn(kwarg, call_kwargs)
        self.MOCK_PLUGINS["_controllable"].run.assert_not_called()

    def test_config_passed_to_non_default(self):
        """Test that a specified non-default plugin gets a config dict passed to it."""
        self.MOCK_PLUGINS["_controllable"].min_qubits = 0
        self.MOCK_PLUGINS["_controllable"].max_qubits = np.inf
        self.MOCK_PLUGINS["_controllable"].support([])
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        return_dag = circuit_to_dag(qc)
        plugin_config = {"option_a": 3.14, "option_b": False}
        pm = PassManager(
            [
                UnitarySynthesis(
                    basis_gates=["u", "cx"], method="_controllable", plugin_config=plugin_config
                )
            ]
        )
        with unittest.mock.patch.object(
            ControllableSynthesis, "run", return_value=return_dag
        ) as plugin_mock:
            _ = pm.run(qc)
            plugin_mock.assert_called()
            call_kwargs = plugin_mock.call_args.kwargs
        expected_kwargs = [
            "config",
        ]
        for kwarg in expected_kwargs:
            self.assertIn(kwarg, call_kwargs)
        self.assertEqual(call_kwargs["config"], plugin_config)

    def test_config_not_passed_to_default_on_fallback(self):
        """Test that all the keywords that the default synthesis plugin needs are passed to it,
        and if if config is specified it is not passed to the default."""
        # Set the mock plugin to reject all keyword arguments, but also be unable to handle
        # operators of any numbers of qubits.  This will cause fallback to the default handler,
        # which should receive a full set of keywords, still.
        self.MOCK_PLUGINS["_controllable"].min_qubits = np.inf
        self.MOCK_PLUGINS["_controllable"].max_qubits = 0
        self.MOCK_PLUGINS["_controllable"].support([])
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        plugin_config = {"option_a": 3.14, "option_b": False}
        pm = PassManager(
            [
                UnitarySynthesis(
                    basis_gates=["u", "cx"], method="_controllable", plugin_config=plugin_config
                )
            ]
        )
        with self.mock_default_run_method() as mocked:
            _ = pm.run(qc)
            mocked.assert_called()
            call_kwargs = mocked.call_args.kwargs
        expected_kwargs = [
            "basis_gates",
            "coupling_map",
            "gate_errors_by_qubit",
            "gate_lengths_by_qubit",
            "natural_direction",
            "pulse_optimize",
        ]
        for kwarg in expected_kwargs:
            self.assertIn(kwarg, call_kwargs)
        self.MOCK_PLUGINS["_controllable"].run.assert_not_called()
        self.assertNotIn("config", call_kwargs)

    def test_target_overrides_basis_gates(self):
        """Test that when both target and basis gates are specified, but the plugin only supports
        basis gates, the basis gates that it gets come from the target."""

        self.MOCK_PLUGINS["_controllable"].support(["basis_gates"])

        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])

        target_basis_gates = ["rx", "rz"]
        standalone_basis_gates = ["cx", "u"]
        target = Target.from_configuration(
            num_qubits=2, basis_gates=target_basis_gates, coupling_map=CouplingMap.from_line(2)
        )
        pm = PassManager(
            [
                UnitarySynthesis(
                    target=target, basis_gates=standalone_basis_gates, method="_controllable"
                )
            ]
        )

        with self.mock_default_run_method() as mocked:
            _ = pm.run(qc)
            mocked.assert_not_called()
            self.MOCK_PLUGINS["_controllable"].run.assert_called()
            call_kwargs = self.MOCK_PLUGINS["_controllable"].run.call_args.kwargs

        self.assertIn("basis_gates", call_kwargs)
        self.assertEqual(call_kwargs["basis_gates"], set(target_basis_gates))
        self.assertNotEqual(call_kwargs["basis_gates"], set(standalone_basis_gates))

    def test_fallback_on_default(self):
        """Test that ``fallback_on_default`` works as expected."""
        self.MOCK_PLUGINS["_controllable"].min_qubits = 0
        self.MOCK_PLUGINS["_controllable"].max_qubits = np.inf
        self.MOCK_PLUGINS["_controllable"].support([])

        circ = QuantumCircuit(1)
        circ.h(0)

        qc = QuantumCircuit(1)
        qc.unitary(Operator(circ).data, [0])

        # The returned circuit consists of a single H-gate.
        return_dag = circuit_to_dag(circ)

        # Plugin returns a DAG and fallback is disabled:
        # the default plugin should not be called.
        with self.subTest("circuit, no fallback"):
            pm = PassManager(
                [
                    UnitarySynthesis(
                        basis_gates=["t", "tdg", "h"],
                        method="_controllable",
                        fallback_on_default=False,
                    )
                ]
            )
            with (
                unittest.mock.patch.object(
                    ControllableSynthesis, "run", return_value=return_dag
                ) as plugin_mock,
                self.mock_default_run_method() as run_mock,
            ):
                _ = pm.run(qc)
                plugin_mock.assert_called()
                run_mock.assert_not_called()

        # Plugin returns None and fallback is disabled:
        # the default plugin should not be called.
        with self.subTest("no circuit, no fallback"):
            pm = PassManager(
                [
                    UnitarySynthesis(
                        basis_gates=["t", "tdg", "h"],
                        method="_controllable",
                        fallback_on_default=False,
                    )
                ]
            )
            with (
                unittest.mock.patch.object(
                    ControllableSynthesis, "run", return_value=None
                ) as plugin_mock,
                self.mock_default_run_method() as run_mock,
            ):
                _ = pm.run(qc)
                plugin_mock.assert_called()
                run_mock.assert_not_called()

        # Plugin returns a DAG and fallback is enabled:
        # the default plugin should not be called.
        with self.subTest("circuit, fallback"):
            pm = PassManager(
                [
                    UnitarySynthesis(
                        basis_gates=["t", "tdg", "h"],
                        method="_controllable",
                        fallback_on_default=True,
                    )
                ]
            )
            with (
                unittest.mock.patch.object(
                    ControllableSynthesis, "run", return_value=return_dag
                ) as plugin_mock,
                self.mock_default_run_method() as run_mock,
            ):
                _ = pm.run(qc)
                plugin_mock.assert_called()
                run_mock.assert_not_called()

        # Plugin returns None and fallback is enabled:
        # this time the default plugin should be called.
        with self.subTest("with fallback, synthesized circuit does not conform"):
            pm = PassManager(
                [
                    UnitarySynthesis(
                        basis_gates=["t", "tdg", "h"],
                        method="_controllable",
                        fallback_on_default=True,
                    )
                ]
            )
            with (
                unittest.mock.patch.object(
                    ControllableSynthesis, "run", return_value=None
                ) as plugin_mock,
                self.mock_default_run_method() as run_mock,
            ):
                _ = pm.run(qc)
                plugin_mock.assert_called()
                run_mock.assert_called()


if __name__ == "__main__":
    unittest.main()
