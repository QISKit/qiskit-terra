# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests the interface for HighLevelSynthesis transpiler pass.
"""
import itertools
import unittest.mock
import numpy as np
from ddt import ddt, data

from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    Gate,
    Qubit,
    Clbit,
    Parameter,
    Operation,
    EquivalenceLibrary,
    Delay,
)
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import (
    XGate,
    ZGate,
    SwapGate,
    CXGate,
    CZGate,
    RZGate,
    PermutationGate,
    U3Gate,
    U2Gate,
    U1Gate,
    UGate,
    CU3Gate,
    CU1Gate,
    QFTGate,
    IGate,
    MCXGate,
    HGate,
    PhaseGate,
    SGate,
    QAOAAnsatz,
    GlobalPhaseGate,
    MultiplierGate,
)
from qiskit.circuit.library import LinearFunction, PauliEvolutionGate
from qiskit.quantum_info import (
    Pauli,
    Clifford,
    Operator,
    Statevector,
    SparsePauliOp,
    SparseObservable,
    random_unitary,
)
from qiskit.synthesis.evolution import synth_pauli_network_rustiq, LieTrotter
from qiskit.synthesis.linear import random_invertible_binary_matrix
from qiskit.synthesis.arithmetic import adder_qft_d00
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.converters import dag_to_circuit, circuit_to_dag, circuit_to_instruction
from qiskit.transpiler import PassManager, TranspilerError, CouplingMap, Target
from qiskit.transpiler.passes.basis import BasisTranslator
from qiskit.transpiler.passes.synthesis.plugin import (
    HighLevelSynthesisPlugin,
    HighLevelSynthesisPluginManager,
    high_level_synthesis_plugin_names,
)
from qiskit._accelerate.high_level_synthesis import (
    synthesize_circuit,
    HighLevelSynthesisData,
    QubitTracker,
)
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis, HLSConfig
from qiskit.transpiler.passes.synthesis.hls_plugins import (
    MCXSynthesis1CleanB95,
    MCXSynthesisNCleanM15,
    MCXSynthesisNDirtyI15,
    MCXSynthesis2CleanKG24,
    MCXSynthesis2DirtyKG24,
    MCXSynthesis1CleanKG24,
    MCXSynthesis1DirtyKG24,
    MCXSynthesisGrayCode,
    MCXSynthesisDefault,
    MCXSynthesisNoAuxV24,
    MCXSynthesisNoAuxHP24,
)
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    ControlModifier,
    InverseModifier,
    PowerModifier,
)
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library.standard_gates.equivalence_library import (
    StandardEquivalenceLibrary as std_eqlib,
)
from test import QiskitTestCase


# In what follows, we create two simple operations OpA and OpB, that potentially mimic
# higher-level objects written by a user.
# For OpA we define two synthesis methods:
#  - "default", which does not require any additional parameters, and
#  - "repeat", which requires a parameter "n" s.t. the "repeat" returns None when "n" is not
#              specified.
# For OpB we define a single synthesis method:
#  - "simple", which does not require any additional parameters.
# Note that OpB does not have a "default" method specified.
# Finally, we will mock the HighLevelSynthesisPluginManager by a dummy class that implements
# a similar functionality, but without depending on the stevedore extension manager.


class OpA(Operation):
    """A simple operation."""

    @property
    def name(self):
        return "op_a"

    @property
    def num_qubits(self):
        return 1

    @property
    def num_clbits(self):
        return 0


class OpB(Operation):
    """Another simple operation."""

    @property
    def name(self):
        return "op_b"

    @property
    def num_qubits(self):
        return 2

    @property
    def num_clbits(self):
        return 0


class OpADefaultSynthesisPlugin(HighLevelSynthesisPlugin):
    """The default synthesis for opA"""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        qc = QuantumCircuit(1)
        qc.id(0)
        return qc


class OpARepeatSynthesisPlugin(HighLevelSynthesisPlugin):
    """The repeat synthesis for opA"""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if "n" not in options:
            return None

        qc = QuantumCircuit(1)
        for _ in range(options["n"]):
            qc.id(0)
        return qc


class OpBSimpleSynthesisPlugin(HighLevelSynthesisPlugin):
    """The simple synthesis for OpB"""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        return qc


class OpBAnotherSynthesisPlugin(HighLevelSynthesisPlugin):
    """Another synthesis plugin for OpB objects.

    This plugin is not registered in MockPluginManager, and is used to
    illustrate the alternative construction mechanism using raw classes.
    """

    def __init__(self, num_swaps=1):
        self.num_swaps = num_swaps

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        num_swaps = options.get("num_swaps", self.num_swaps)

        qc = QuantumCircuit(2)
        for _ in range(num_swaps):
            qc.swap(0, 1)
        return qc


class OpAPluginNeedsCouplingMap(HighLevelSynthesisPlugin):
    """Synthesis plugins for OpA that needs a coupling map to be run."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if coupling_map is None:
            raise TranspilerError("Coupling map should be specified!")
        qc = QuantumCircuit(1)
        qc.id(0)
        return qc


class OpAPluginNeedsQubits(HighLevelSynthesisPlugin):
    """Synthesis plugins for OpA that needs ``qubits`` to be specified."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if qubits is None:
            raise TranspilerError("Qubits should be specified!")
        qc = QuantumCircuit(1)
        qc.id(0)
        return qc


class MockPluginManager:
    """Mocks the functionality of HighLevelSynthesisPluginManager,
    without actually depending on the stevedore extension manager.
    """

    def __init__(self):
        self.plugins = {
            "op_a.default": OpADefaultSynthesisPlugin,
            "op_a.repeat": OpARepeatSynthesisPlugin,
            "op_b.simple": OpBSimpleSynthesisPlugin,
            "op_a.needs_coupling_map": OpAPluginNeedsCouplingMap,
            "op_a.needs_qubits": OpAPluginNeedsQubits,
        }

        self.plugins_by_op = {
            "op_a": ["default", "repeat", "needs_coupling_map", "needs_qubits"],
            "op_b": ["simple"],
        }

    def method_names(self, op_name):
        """Returns plugin methods for op_name."""
        if op_name in self.plugins_by_op:
            return self.plugins_by_op[op_name]
        else:
            return []

    def method(self, op_name, method_name):
        """Returns the plugin for ``op_name`` and ``method_name``."""
        plugin_name = op_name + "." + method_name
        return self.plugins[plugin_name]()

    def op_names(self):
        """Returns the names of high-level-objects with available synthesis methods."""
        return list(self.plugins_by_op.keys())


class MockPlugin(HighLevelSynthesisPlugin):
    """A mock HLS using auxiliary qubits."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run a mock synthesis for high_level_object being anything with a num_qubits property.

        Replaces the high_level_objects by a layer of X gates, applies S gates on clean
        ancillas and T gates on dirty ancillas.
        """

        num_action_qubits = high_level_object.num_qubits
        num_clean = options["num_clean_ancillas"]
        num_dirty = options["num_dirty_ancillas"]
        num_qubits = num_action_qubits + num_clean + num_dirty
        decomposition = QuantumCircuit(num_qubits)
        decomposition.x(range(num_action_qubits))
        if num_clean > 0:
            decomposition.s(range(num_action_qubits, num_action_qubits + num_clean))
        if num_dirty > 0:
            decomposition.t(range(num_action_qubits + num_clean, num_qubits))

        return decomposition


class EmptyPlugin(HighLevelSynthesisPlugin):
    """A mock plugin returning None (i.e. a failed synthesis)."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Elaborate code to return None :)"""
        return None


class GlobalPhaseGatePlugin(HighLevelSynthesisPlugin):
    """Plugin that replaces a global phase gate by a global phase."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Returns a quantum circuit with global phase."""
        decomposition = QuantumCircuit(1)
        decomposition.global_phase = high_level_object.params[0]
        return decomposition


@ddt
class TestHighLevelSynthesisInterface(QiskitTestCase):
    """Tests for the synthesis plugin interface."""

    def create_circ(self):
        """Create a simple circuit used for tests with two OpA gates and one OpB gate."""
        qc = QuantumCircuit(3)
        qc.append(OpA(), [0])
        qc.append(OpB(), [0, 1])
        qc.append(OpA(), [2])
        return qc

    def test_no_config(self):
        """Check the default behavior of HighLevelSynthesis, without
        HighLevelSynthesisConfig specified. In this case, the default
        synthesis methods should be used when defined. OpA has such a
        method, and OpB does not."""
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            pm = PassManager([HighLevelSynthesis()])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # OpA's default method replaces by "id", OpB has no default method
            self.assertNotIn("op_a", ops.keys())
            self.assertEqual(ops["id"], 2)
            self.assertIn("op_b", ops.keys())
            self.assertEqual(ops["op_b"], 1)

    def test_default_config(self):
        """Check the default behavior of HighLevelSynthesis, with
        the default HighLevelSynthesisConfig specified. The behavior should
        be the same as without config."""
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig()
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # OpA's default method replaces by "id", OpB has no default method
            self.assertNotIn("op_a", ops.keys())
            self.assertEqual(ops["id"], 2)
            self.assertIn("op_b", ops.keys())
            self.assertEqual(ops["op_b"], 1)

    def test_non_default_config(self):
        """Check the default behavior of HighLevelSynthesis, specifying
        non-default synthesis methods for OpA and for OpB.
        """
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=[("repeat", {"n": 2})], op_b=[("simple", {})])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            self.assertNotIn("op_a", ops.keys())
            self.assertNotIn("op_b", ops.keys())
            self.assertEqual(ops["id"], 4)
            self.assertEqual(ops["cx"], 1)

    def test_synthesis_returns_none(self):
        """Check that when synthesis method is specified but returns None,
        the operation does not get synthesized.
        """
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=[("repeat", {})])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # The repeat method for OpA without "n" specified returns None.
            self.assertIn("op_a", ops.keys())
            self.assertIn("op_b", ops.keys())

    def test_use_default_on_unspecified_is_false(self):
        """Check that when use_default_on_unspecified is False, the default synthesis
        method is not applied.
        """
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(use_default_on_unspecified=False)
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # The repeat method for OpA without "n" specified returns None.
            self.assertIn("op_a", ops.keys())
            self.assertIn("op_b", ops.keys())

    def test_use_default_on_unspecified_is_true(self):
        """Check that when use_default_on_unspecified is True (which should be the default
        value), the default synthesis method gets applied.
        OpA has such a method, and OpB does not."""
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            pm = PassManager([HighLevelSynthesis()])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # OpA's default method replaces by "id", OpB has no default method
            self.assertNotIn("op_a", ops.keys())
            self.assertEqual(ops["id"], 2)
            self.assertIn("op_b", ops.keys())
            self.assertEqual(ops["op_b"], 1)

    def test_skip_synthesis_with_empty_methods_list(self):
        """Check that when synthesis config is specified, but an operation
        is given an empty list of methods, it is not synthesized.
        """
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=[])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # The repeat method for OpA without "n" specified returns None.
            self.assertIn("op_a", ops.keys())
            self.assertIn("op_b", ops.keys())

    def test_multiple_methods(self):
        """Check that when there are two synthesis methods specified,
        and the first returns None, then the second method gets used.
        """
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=[("repeat", {}), ("default", {})])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # The repeat method for OpA without "n" specified returns None.
            self.assertNotIn("op_a", ops.keys())
            self.assertEqual(ops["id"], 2)
            self.assertIn("op_b", ops.keys())

    def test_multiple_methods_short_form(self):
        """Check that when there are two synthesis methods specified,
        and the first returns None, then the second method gets used.
        In this example, the list of methods is specified without
        explicitly listing empty argument lists.
        """

        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=["repeat", "default"])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # The repeat method for OpA without "n" specified returns None.
            self.assertNotIn("op_a", ops.keys())
            self.assertEqual(ops["id"], 2)
            self.assertIn("op_b", ops.keys())

    def test_synthesis_using_alternate_form(self):
        """Test alternative form of specifying synthesis methods."""

        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            # synthesis using raw class, without extension manager
            plugin = OpBAnotherSynthesisPlugin(num_swaps=6)
            hls_config = HLSConfig(op_b=[(plugin, {})])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            self.assertEqual(ops["swap"], 6)

    def test_synthesis_using_alternate_short_form(self):
        """Test alternative form of specifying synthesis methods."""

        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            # synthesis using raw class, without extension manager
            plugin = OpBAnotherSynthesisPlugin(num_swaps=6)
            hls_config = HLSConfig(op_b=[plugin])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            self.assertEqual(ops["swap"], 6)

    def test_coupling_map_gets_passed_to_plugins(self):
        """Check that passing coupling map works correctly."""
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=["needs_coupling_map"])
            pm_bad = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            pm_good = PassManager(
                [
                    HighLevelSynthesis(
                        hls_config=hls_config, coupling_map=CouplingMap.from_line(qc.num_qubits)
                    )
                ]
            )

            # HighLevelSynthesis is initialized without a coupling map, but calling a plugin that
            # raises a TranspilerError without the coupling map.
            with self.assertRaises(TranspilerError):
                pm_bad.run(qc)

            # Now HighLevelSynthesis is initialized with a coupling map.
            pm_good.run(qc)

    def test_target_gets_passed_to_plugins(self):
        """Check that passing target (and constructing coupling map from the target)
        works correctly.
        """
        qc = QuantumCircuit(3)
        qc.append(OpA(), [0])

        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=["needs_coupling_map"])
            pm_good = PassManager(
                [
                    HighLevelSynthesis(
                        hls_config=hls_config,
                        target=GenericBackendV2(num_qubits=5, basis_gates=["u", "cx", "id"]).target,
                    )
                ]
            )

            # HighLevelSynthesis is initialized with target.
            pm_good.run(qc)

    def test_qubits_get_passed_to_plugins(self):
        """Check that setting ``use_qubit_indices`` works correctly."""
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=["needs_qubits"])
            pm_use_qubits_false = PassManager(
                [HighLevelSynthesis(hls_config=hls_config, use_qubit_indices=False)]
            )
            pm_use_qubits_true = PassManager(
                [HighLevelSynthesis(hls_config=hls_config, use_qubit_indices=True)]
            )

            # HighLevelSynthesis is initialized with use_qubit_indices=False, which means synthesis
            # plugin should see qubits=None and raise a transpiler error.
            with self.assertRaises(TranspilerError):
                pm_use_qubits_false.run(qc)

            # HighLevelSynthesis is initialized with use_qubit_indices=True, which means synthesis
            # plugin should see qubits and complete without errors.
            pm_use_qubits_true.run(qc)

    def test_ancilla_arguments(self):
        """Test ancillas are correctly labelled."""
        gate = Gate(name="duckling", num_qubits=5, params=[])
        hls_config = HLSConfig(duckling=[MockPlugin()])

        qc = QuantumCircuit(10)
        qc.h([0, 8, 9])  # the two last H gates yield two dirty ancillas
        qc.barrier()
        qc.append(gate, range(gate.num_qubits))

        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])

        synthesized = pm.run(qc)

        count = synthesized.count_ops()
        self.assertEqual(count.get("x", 0), gate.num_qubits)  # gate qubits
        self.assertEqual(count.get("s", 0), qc.num_qubits - gate.num_qubits - 2)  # clean
        self.assertEqual(count.get("t", 0), 2)  # dirty

    def test_ancilla_noop(self):
        """Test ancillas states are not affected by no-ops."""
        gate = Gate(name="duckling", num_qubits=1, params=[])
        hls_config = HLSConfig(duckling=[MockPlugin()])
        pm = PassManager([HighLevelSynthesis(hls_config)])

        noops = [Delay(100), IGate()]
        for noop in noops:
            qc = QuantumCircuit(2)
            qc.append(noop, [1])  # this noop should still yield a clean ancilla
            qc.barrier()
            qc.append(gate, [0])

            synthesized = pm.run(qc)
            count = synthesized.count_ops()
            with self.subTest(noop=noop):
                self.assertEqual(count.get("x", 0), gate.num_qubits)  # gate qubits
                self.assertEqual(count.get("s", 0), 1)  # clean ancilla
                self.assertEqual(count.get("t", 0), 0)  # dirty ancilla

    @data(True, False)
    def test_ancilla_reset(self, reset):
        """Test ancillas are correctly freed after a reset operation."""
        gate = Gate(name="duckling", num_qubits=1, params=[])
        hls_config = HLSConfig(duckling=[MockPlugin()])
        pm = PassManager([HighLevelSynthesis(hls_config)])

        qc = QuantumCircuit(2)
        qc.h(1)
        if reset:
            qc.reset(1)  # the reset frees the ancilla qubit again
        qc.barrier()
        qc.append(gate, [0])

        synthesized = pm.run(qc)
        count = synthesized.count_ops()

        expected_clean = 1 if reset else 0
        expected_dirty = 1 - expected_clean

        self.assertEqual(count.get("x", 0), gate.num_qubits)  # gate qubits
        self.assertEqual(count.get("s", 0), expected_clean)  # clean ancilla
        self.assertEqual(count.get("t", 0), expected_dirty)  # clean ancilla

    def test_ancilla_state_maintained(self):
        """Test ancillas states are still dirty/clean after they've been used."""
        gate = Gate(name="duckling", num_qubits=1, params=[])
        hls_config = HLSConfig(duckling=[MockPlugin()])
        pm = PassManager([HighLevelSynthesis(hls_config)])

        qc = QuantumCircuit(3)
        qc.h(2)  # the final ancilla is dirty
        qc.barrier()
        qc.append(gate, [0])
        qc.append(gate, [0])

        # the ancilla states should be unchanged after the synthesis, i.e. qubit 1 is always
        # clean (S gate) and qubit 2 is always dirty (T gate)
        ref = QuantumCircuit(3)
        ref.h(2)
        ref.barrier()
        for _ in range(2):
            ref.x(0)
            ref.s(1)
            ref.t(2)

        self.assertEqual(ref, pm.run(qc))

    def test_synth_fails_definition_exists(self):
        """Test the case that a synthesis fails but the operation can be unrolled."""

        circuit = QuantumCircuit(1)
        circuit.ry(0.2, 0)

        config = HLSConfig(ry=[EmptyPlugin()])
        hls = HighLevelSynthesis(hls_config=config)

        with self.subTest("nothing happened w/o basis gates"):
            out = hls(circuit)
            self.assertEqual(out, circuit)

        hls = HighLevelSynthesis(hls_config=config, basis_gates=["u"])
        with self.subTest("unrolled w/ basis gates"):
            out = hls(circuit)
            self.assertEqual(out.count_ops(), {"u": 1})

    def test_both_basis_gates_and_plugin_specified(self):
        """Test that a gate is not synthesized when it belongs to basis_gates,
        regardless of whether there is a plugin method available.

        See: https://github.com/Qiskit/qiskit/issues/13412 for more
        details.
        """
        with self.assertWarns(DeprecationWarning):
            qc = QAOAAnsatz(SparsePauliOp("Z"), initial_state=QuantumCircuit(1))
        pm = PassManager([HighLevelSynthesis(basis_gates=["PauliEvolution"])])
        qct = pm.run(qc)
        self.assertEqual(qct.count_ops()["PauliEvolution"], 2)

    def test_track_global_phase(self):
        """Test that high-level-synthesis keeps track of the global phases."""

        # Custom plugin that replaces GlobalPhaseGate by global phase.
        hls_config = HLSConfig(global_phase=[GlobalPhaseGatePlugin()])
        hls_pass = HighLevelSynthesis(hls_config=hls_config, basis_gates=["cx", "u"])

        # A circuit that has both a GlobalPhaseGate and a global phase
        qc = QuantumCircuit(2, global_phase=0.2)
        qc.append(GlobalPhaseGate(0.1))
        qc.cx(0, 1)
        qc.append(GlobalPhaseGate(0.5))

        with self.subTest("global phase at top level"):
            transpiled = hls_pass(qc)
            expected = QuantumCircuit(2, global_phase=0.8)
            expected.cx(0, 1)
            self.assertEqual(transpiled, expected)

        with self.subTest("global phase in custom gate"):
            # A circuit with qc as custom gate
            qc2 = QuantumCircuit(4, global_phase=0.1)
            qc2.append(qc.to_gate(), [1, 3])
            qc2.cx(0, 1)
            transpiled = hls_pass(qc2)
            expected = QuantumCircuit(4, global_phase=0.9)
            expected.cx(1, 3)
            expected.cx(0, 1)
            self.assertEqual(transpiled, expected)

        with self.subTest("global phase in control flow op"):
            # A circuit with qc inside control flow blocks
            qc3 = QuantumCircuit(4, 1)
            qc3.if_else((0, True), qc, qc, [0, 1], [])
            transpiled = hls_pass(qc3)
            transpiled_block = transpiled[0].operation.blocks[0]
            expected_block = QuantumCircuit(2, global_phase=0.8)
            expected_block.cx(0, 1)
            self.assertEqual(transpiled_block, expected_block)

    def test_control_flow(self):
        """Test that the pass recurses into control-flow ops."""
        clifford_circuit = QuantumCircuit(3)
        clifford_circuit.cx(1, 0)
        clifford_circuit.cz(0, 2)
        cliff = Clifford(clifford_circuit)

        qc = QuantumCircuit(5, 5)
        with qc.for_loop(range(3)):
            qc.append(cliff, [0, 1, 4])

        transpiled = HighLevelSynthesis(basis_gates=["cx", "u", "for_loop"])(qc)
        transpiled_block = transpiled[0].operation.blocks[0]
        self.assertNotIn("clifford", transpiled_block.count_ops())

    @data(1, 2, 3, 4)
    def test_unitary(self, num_qubits):
        """Test that the pass handles unitary gates."""
        unitary = random_unitary(2**num_qubits, seed=42)
        qc = QuantumCircuit(num_qubits)
        qc.unitary(unitary, qc.qubits)
        target = Target.from_configuration(num_qubits=5, basis_gates=["cx", "u"])
        transpiled = HighLevelSynthesis(target=target)(qc)
        self.assertLessEqual(set(transpiled.count_ops()), {"cx", "u"})

    @data(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13],
    )
    def test_qubit_tracking(self, gate_qubits):
        """Test that the pass tracks qubit states correctly."""

        # Create a quantum circuit with a MultiplierGate (of width 12).
        num_qubits = 25
        qc = QuantumCircuit(num_qubits)
        qc.compose(MultiplierGate(3), gate_qubits, inplace=True)

        # Initialize high-level-synthesis data
        hls_config = HLSConfig()
        hls_plugin_manager = HighLevelSynthesisPluginManager()
        hls_op_names = set(hls_plugin_manager.plugins_by_op.keys())

        target = Target.from_configuration(
            basis_gates=["cx", "u"],
            coupling_map=CouplingMap.from_line(num_qubits),
        )
        coupling_map = target.build_coupling_map()

        hls_data = HighLevelSynthesisData(
            hls_config=hls_config,
            hls_plugin_manager=hls_plugin_manager,
            coupling_map=coupling_map,
            target=target,
            equivalence_library=std_eqlib,
            hls_op_names=hls_op_names,
            device_insts={"cx", "u"},
            use_physical_indices=False,
            min_qubits=0,
            unroll_definitions=True,
            optimize_clifford_t=False,
        )

        # The tracker keeps the state of each qubits in the circuit.
        # Initially all the qubits are clean.
        tracker = QubitTracker(num_qubits, True)

        # Synthesize the circuit, which updates the tracker as a side-effect.
        # Despite the apparent simplicity of this example, there is a lot going on under
        # the hood:
        # - the multiplier is synthesized using the default plugin for multiplier gates,
        #   which produces annotated half-adder gates
        # - the annotated half-adder gates are synthesized using the default
        #   plugin for annotated operations, which produces circuits with MCX gates
        # - the MCX gates are synthesized using the default plugin for MCX gates
        #   and use 1 clean ancilla qubit
        # - the ancilla qubits are supposed to be clean after the synthesis is complete
        _ = synthesize_circuit(qc._data, list(range(25)), hls_data, tracker)

        # Every qubit in the multiplier gate must be "dirty" and every other qubit
        # must be clean.
        for q in range(num_qubits):
            self.assertEqual(tracker.is_qubit_clean(q), q not in gate_qubits)


class TestHighLevelSynthesisQuality(QiskitTestCase):
    """Test the "quality" of circuits produced by HighLevelSynthesis."""

    def test_controlled_x(self):
        """Test default synthesis of controlled-X gate."""
        qc = QuantumCircuit(15)
        qc.append(XGate().control(6, annotated=False), [0, 1, 2, 3, 4, 5, 6])
        qct = HighLevelSynthesis(basis_gates=["cx", "u"])(qc)
        self.assertLessEqual(qct.count_ops()["cx"], 30)

    def test_controlled_cx(self):
        """Test default synthesis of controlled-CX gate."""
        qc = QuantumCircuit(15)
        qc.append(CXGate().control(5, annotated=False), [0, 1, 2, 3, 4, 5, 6])
        qct = HighLevelSynthesis(basis_gates=["cx", "u"])(qc)
        self.assertLessEqual(qct.count_ops()["cx"], 30)

    def test_recursively_controlled_cx(self):
        """Test default synthesis of recursively controlled CX-gate."""
        inner = QuantumCircuit(5)
        inner.append(CXGate().control(3, annotated=True), [0, 1, 2, 3, 4])
        controlled_inner_gate2 = inner.to_gate().control(2, annotated=True)
        qc = QuantumCircuit(15)
        qc.append(controlled_inner_gate2, [0, 1, 2, 3, 4, 5, 6])
        qct = HighLevelSynthesis(basis_gates=["cx", "u"])(qc)
        self.assertLessEqual(qct.count_ops()["cx"], 30)

    def test_controlled_z(self):
        """Test default synthesis of controlled-X gate."""
        qc = QuantumCircuit(15)
        qc.append(ZGate().control(6, annotated=False), [0, 1, 2, 3, 4, 5, 6])
        qct = HighLevelSynthesis(basis_gates=["cx", "u"])(qc)
        self.assertLessEqual(qct.count_ops()["cx"], 30)

    def test_controlled_cz(self):
        """Test default synthesis of controlled-CZ gate."""
        qc = QuantumCircuit(15)
        qc.append(CZGate().control(5, annotated=False), [0, 1, 2, 3, 4, 5, 6])
        qct = HighLevelSynthesis(basis_gates=["cx", "u"])(qc)
        self.assertLessEqual(qct.count_ops()["cx"], 30)

    def test_recursively_controlled_cz(self):
        """Test default synthesis of recursively controlled CZ-gate."""
        inner = QuantumCircuit(5)
        inner.append(CZGate().control(3, annotated=True), [0, 1, 2, 3, 4])
        controlled_inner_gate2 = inner.to_gate().control(2, annotated=True)
        qc = QuantumCircuit(15)
        qc.append(controlled_inner_gate2, [0, 1, 2, 3, 4, 5, 6])
        qct = HighLevelSynthesis(basis_gates=["cx", "u"])(qc)
        self.assertLessEqual(qct.count_ops()["cx"], 30)

    def test_controlled_qft_adder(self):
        """Test QFT-based synthesis of half-adder gate."""
        gate = adder_qft_d00(num_state_qubits=3, kind="half", annotated=True).control(
            num_ctrl_qubits=2, annotated=True
        )
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)
        qct = HighLevelSynthesis(basis_gates=["cx", "u"], qubits_initially_zero=False)(qc)
        self.assertLessEqual(qct.count_ops()["cx"], 450)

    def test_controlled_qft(self):
        """Test controlled QFT-gate."""
        gate = QFTGate(3).control(2, annotated=True)
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)
        qct = HighLevelSynthesis(basis_gates=["cx", "u"], qubits_initially_zero=False)(qc)
        self.assertLessEqual(qct.count_ops()["cx"], 198)


class TestPMHSynthesisLinearFunctionPlugin(QiskitTestCase):
    """Tests for the PMHSynthesisLinearFunction plugin for synthesizing linear functions."""

    @staticmethod
    def construct_linear_circuit(num_qubits: int):
        """Construct linear circuit."""
        qc = QuantumCircuit(num_qubits)
        for i in range(1, num_qubits):
            qc.cx(i - 1, i)
        return qc

    def test_section_size(self):
        """Test that the plugin takes the section size argument into account."""

        mat = random_invertible_binary_matrix(7, seed=1234)
        qc = QuantumCircuit(7)
        qc.append(LinearFunction(mat), [0, 1, 2, 3, 4, 5, 6])

        with self.subTest("section_size_1"):
            hls_config = HLSConfig(linear_function=[("pmh", {"section_size": 1})])
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 30)
            self.assertEqual(qct.depth(), 27)

        with self.subTest("section_size_2"):
            hls_config = HLSConfig(linear_function=[("pmh", {"section_size": 2})])
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 27)
            self.assertEqual(qct.depth(), 23)

        with self.subTest("section_size_3"):
            hls_config = HLSConfig(linear_function=[("pmh", {"section_size": 3})])
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 29)
            self.assertEqual(qct.depth(), 23)

    def test_invert_and_transpose(self):
        """Test that the plugin takes the use_inverted and use_transposed arguments into account."""

        linear_function = LinearFunction(self.construct_linear_circuit(7))

        qc = QuantumCircuit(7)
        qc.append(linear_function, [0, 1, 2, 3, 4, 5, 6])

        with self.subTest("default"):
            hls_config = HLSConfig(linear_function=[("pmh", {})])
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 12)
            self.assertEqual(qct.depth(), 8)

        with self.subTest("invert"):
            hls_config = HLSConfig(linear_function=[("pmh", {"use_inverted": True})])
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 6)
            self.assertEqual(qct.depth(), 6)

        with self.subTest("transpose"):
            hls_config = HLSConfig(linear_function=[("pmh", {"use_transposed": True})])
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 6)
            self.assertEqual(qct.depth(), 6)

        with self.subTest("invert_and_transpose"):
            hls_config = HLSConfig(
                linear_function=[("pmh", {"use_inverted": True, "use_transposed": True})]
            )
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 6)
            self.assertEqual(qct.depth(), 6)

    def test_plugin_selection_all(self):
        """Test setting plugin_selection to all."""

        linear_function = LinearFunction(self.construct_linear_circuit(7))
        qc = QuantumCircuit(7)
        qc.append(linear_function, [0, 1, 2, 3, 4, 5, 6])

        with self.subTest("sequential"):
            # In the default "run sequential" mode, we stop as soon as a plugin
            # in the list returns a circuit.
            # For this specific example the default options lead to a suboptimal circuit.
            hls_config = HLSConfig(linear_function=[("pmh", {}), ("pmh", {"use_inverted": True})])
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 12)
            self.assertEqual(qct.depth(), 8)

        with self.subTest("all"):
            # In the non-default "run all" mode, we examine all plugins in the list.
            # For this specific example we get the better result for the second plugin in the list.
            hls_config = HLSConfig(
                linear_function=[("pmh", {}), ("pmh", {"use_inverted": True})],
                plugin_selection="all",
            )
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 6)
            self.assertEqual(qct.depth(), 6)

    def test_plugin_selection_all_with_metrix(self):
        """Test setting plugin_selection to all and specifying different evaluation functions."""

        # The seed is chosen so that we get different best circuits depending on whether we
        # want to minimize size or depth.
        mat = random_invertible_binary_matrix(7, seed=38)
        qc = QuantumCircuit(7)
        qc.append(LinearFunction(mat), [0, 1, 2, 3, 4, 5, 6])

        with self.subTest("size_fn"):
            # We want to minimize the "size" (aka the number of gates) in the circuit
            hls_config = HLSConfig(
                linear_function=[
                    ("pmh", {}),
                    ("pmh", {"use_inverted": True}),
                    ("pmh", {"use_transposed": True}),
                    ("pmh", {"use_inverted": True, "use_transposed": True}),
                ],
                plugin_selection="all",
                plugin_evaluation_fn=lambda qc: qc.size(),
            )
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 23)
            self.assertEqual(qct.depth(), 19)

        with self.subTest("depth_fn"):
            # We want to minimize the "depth" (aka the number of layers) in the circuit
            hls_config = HLSConfig(
                linear_function=[
                    ("pmh", {}),
                    ("pmh", {"use_inverted": True}),
                    ("pmh", {"use_transposed": True}),
                    ("pmh", {"use_inverted": True, "use_transposed": True}),
                ],
                plugin_selection="all",
                plugin_evaluation_fn=lambda qc: qc.depth(),
            )
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 24)
            self.assertEqual(qct.depth(), 13)

    def test_unfortunate_name(self):
        """Test the synthesis is not triggered for a custom gate with the same name."""
        intruder = QuantumCircuit(2, name="linear_function")
        circuit = QuantumCircuit(2)
        circuit.append(intruder.to_gate(), [0, 1])

        hls = HighLevelSynthesis()
        synthesized = hls(circuit)

        self.assertIn("linear_function", synthesized.count_ops())


class TestKMSSynthesisLinearFunctionPlugin(QiskitTestCase):
    """Tests for the KMSSynthesisLinearFunction plugin for synthesizing linear functions."""

    @staticmethod
    def construct_linear_circuit(num_qubits: int):
        """Construct linear circuit."""
        qc = QuantumCircuit(num_qubits)
        for i in range(1, num_qubits):
            qc.cx(i - 1, i)
        return qc

    def test_invert_and_transpose(self):
        """Test that the plugin takes the use_inverted and use_transposed arguments into account."""

        linear_function = LinearFunction(self.construct_linear_circuit(7))

        qc = QuantumCircuit(7)
        qc.append(linear_function, [0, 1, 2, 3, 4, 5, 6])

        with self.subTest("default"):
            hls_config = HLSConfig(linear_function=[("kms", {})])
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 100)
            self.assertEqual(qct.depth(), 34)

        with self.subTest("invert"):
            hls_config = HLSConfig(linear_function=[("kms", {"use_inverted": True})])
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 101)
            self.assertEqual(qct.depth(), 35)

        with self.subTest("transpose"):
            hls_config = HLSConfig(linear_function=[("kms", {"use_transposed": True})])
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 84)
            self.assertEqual(qct.depth(), 31)

        with self.subTest("invert_and_transpose"):
            hls_config = HLSConfig(
                linear_function=[("kms", {"use_inverted": True, "use_transposed": True})]
            )
            qct = HighLevelSynthesis(hls_config=hls_config)(qc)
            self.assertEqual(LinearFunction(qct), LinearFunction(qc))
            self.assertEqual(qct.size(), 87)
            self.assertEqual(qct.depth(), 32)

    def test_unfortunate_name(self):
        """Test the synthesis is not triggered for a custom gate with the same name."""
        intruder = QuantumCircuit(2, name="linear_function")
        circuit = QuantumCircuit(2)
        circuit.append(intruder.to_gate(), [0, 1])

        hls = HighLevelSynthesis()
        synthesized = hls(circuit)

        self.assertIn("linear_function", synthesized.count_ops())


class TestTokenSwapperPermutationPlugin(QiskitTestCase):
    """Tests for the token swapper plugin for synthesizing permutation gates."""

    def test_token_swapper_in_known_plugin_names(self):
        """Test that "token_swapper" is an available synthesis plugin for permutation gates."""
        self.assertIn(
            "token_swapper", HighLevelSynthesisPluginManager().method_names("permutation")
        )

    def test_abstract_synthesis(self):
        """Test abstract synthesis of a permutation gate (either the coupling map or the set
        of qubits over which the permutation is defined is not specified).
        """

        # Permutation gate
        # 4->0, 6->1, 3->2, 7->3, 1->4, 2->5, 0->6, 5->7
        perm = PermutationGate([4, 6, 3, 7, 1, 2, 0, 5])

        # Circuit with permutation gate
        qc = QuantumCircuit(8)
        qc.append(perm, range(8))

        # Synthesize circuit using the token swapper plugin
        synthesis_config = HLSConfig(permutation=[("token_swapper", {"trials": 10, "seed": 1})])
        qc_transpiled = PassManager(HighLevelSynthesis(synthesis_config)).run(qc)

        self.assertEqual(Operator(qc_transpiled), Operator(qc))

    def test_concrete_synthesis(self):
        """Test concrete synthesis of a permutation gate (we have both the coupling map and the
        set of qubits over which the permutation gate is defined; moreover, the coupling map may
        have more qubits than the permutation gate).
        """

        # Permutation gate
        perm = PermutationGate([0, 1, 4, 3, 2])

        # Circuit with permutation gate
        qc = QuantumCircuit(8)
        qc.append(perm, [3, 4, 5, 6, 7])

        coupling_map = CouplingMap.from_ring(8)

        synthesis_config = HLSConfig(permutation=[("token_swapper", {"trials": 10})])
        qc_transpiled = PassManager(
            HighLevelSynthesis(
                synthesis_config, coupling_map=coupling_map, target=None, use_qubit_indices=True
            )
        ).run(qc)

        qc_expected = QuantumCircuit(8)
        qc_expected.swap(6, 7)
        qc_expected.swap(5, 6)
        qc_expected.swap(6, 7)
        self.assertEqual(qc_transpiled, qc_expected)

    def test_concrete_synthesis_over_disconnected_qubits(self):
        """Test concrete synthesis of a permutation gate over a disconnected set of qubits,
        when synthesis is possible.
        """

        # Permutation gate
        perm = PermutationGate([1, 0, 3, 2])

        # Circuit with permutation gate
        qc = QuantumCircuit(10)
        qc.append(perm, [3, 2, 7, 8])

        coupling_map = CouplingMap.from_ring(10)

        synthesis_config = HLSConfig(permutation=[("token_swapper", {"trials": 10})])
        qc_transpiled = PassManager(
            HighLevelSynthesis(
                synthesis_config, coupling_map=coupling_map, target=None, use_qubit_indices=True
            )
        ).run(qc)

        qc_expected = QuantumCircuit(10)
        qc_expected.swap(2, 3)
        qc_expected.swap(7, 8)

        # Even though the permutation is over a disconnected set of qubits, the synthesis
        # is possible.
        self.assertEqual(qc_transpiled, qc_expected)

    def test_concrete_synthesis_is_not_possible(self):
        """Test concrete synthesis of a permutation gate over a disconnected set of qubits,
        when synthesis is not possible.
        """

        # Permutation gate
        perm = PermutationGate([0, 2, 1, 3])

        # Circuit with permutation gate
        qc = QuantumCircuit(10)
        qc.append(perm, [3, 2, 7, 8])

        coupling_map = CouplingMap.from_ring(10)

        synthesis_config = HLSConfig(permutation=[("token_swapper", {"trials": 10})])
        qc_transpiled = PassManager(
            HighLevelSynthesis(
                synthesis_config, coupling_map=coupling_map, target=None, use_qubit_indices=True
            )
        ).run(qc)

        # The synthesis is not possible. In this case the plugin should return `None`
        # and `HighLevelSynthesis` should not change the original circuit.
        self.assertEqual(qc_transpiled, qc)

    def test_abstract_synthesis_all_permutations(self):
        """Test abstract synthesis of permutation gates, varying permutation gate patterns."""

        edges = [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1), (3, 4), (4, 3)]

        coupling_map = CouplingMap()
        for i in range(5):
            coupling_map.add_physical_qubit(i)
        for edge in edges:
            coupling_map.add_edge(*edge)

        synthesis_config = HLSConfig(permutation=[("token_swapper", {"trials": 10})])
        pm = PassManager(
            HighLevelSynthesis(
                synthesis_config, coupling_map=coupling_map, target=None, use_qubit_indices=False
            )
        )

        for pattern in itertools.permutations(range(4)):
            qc = QuantumCircuit(5)
            qc.append(PermutationGate(pattern), [2, 0, 3, 1])
            self.assertIn("permutation", qc.count_ops())

            qc_transpiled = pm.run(qc)
            self.assertNotIn("permutation", qc_transpiled.count_ops())

            self.assertEqual(Operator(qc), Operator(qc_transpiled))

    def test_concrete_synthesis_all_permutations(self):
        """Test concrete synthesis of permutation gates, varying permutation gate patterns."""

        edges = [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1), (3, 4), (4, 3)]

        coupling_map = CouplingMap()
        for i in range(5):
            coupling_map.add_physical_qubit(i)
        for edge in edges:
            coupling_map.add_edge(*edge)

        synthesis_config = HLSConfig(permutation=[("token_swapper", {"trials": 10})])
        pm = PassManager(
            HighLevelSynthesis(
                synthesis_config, coupling_map=coupling_map, target=None, use_qubit_indices=True
            )
        )

        for pattern in itertools.permutations(range(4)):

            qc = QuantumCircuit(5)
            qc.append(PermutationGate(pattern), [2, 0, 3, 1])
            self.assertIn("permutation", qc.count_ops())

            qc_transpiled = pm.run(qc)
            self.assertNotIn("permutation", qc_transpiled.count_ops())
            self.assertEqual(Operator(qc), Operator(qc_transpiled))

            for inst in qc_transpiled:
                qubits = tuple(qc_transpiled.find_bit(q).index for q in inst.qubits)
                self.assertIn(qubits, edges)

    def test_unfortunate_name(self):
        """Test the synthesis is not triggered for a custom gate with the same name."""
        intruder = QuantumCircuit(2, name="permutation")
        circuit = QuantumCircuit(2)
        circuit.append(intruder.to_gate(), [0, 1])

        hls = HighLevelSynthesis()
        synthesized = hls(circuit)

        self.assertIn("permutation", synthesized.count_ops())


class TestHighLevelSynthesisModifiers(QiskitTestCase):
    """Tests for high-level-synthesis pass."""

    def test_control_basic_gates(self):
        """Test lazy control synthesis of basic gates (each has its class ``control`` method)."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), ControlModifier(2))
        lazy_gate2 = AnnotatedOperation(CXGate(), ControlModifier(1))
        lazy_gate3 = AnnotatedOperation(RZGate(np.pi / 4), ControlModifier(1))
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate1, [0, 1, 2, 3])
        circuit.append(lazy_gate2, [0, 1, 2])
        circuit.append(lazy_gate3, [2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        controlled_gate1 = SwapGate().control(2, annotated=False)
        controlled_gate2 = CXGate().control(1, annotated=False)
        controlled_gate3 = RZGate(np.pi / 4).control(1, annotated=False)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(controlled_gate1, [0, 1, 2, 3])
        expected_circuit.append(controlled_gate2, [0, 1, 2])
        expected_circuit.append(controlled_gate3, [2, 3])

        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_control_custom_gates(self):
        """Test lazy control synthesis of custom gates (which inherits ``control`` method from
        ``Gate``).
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        gate = qc.to_gate()
        circuit = QuantumCircuit(4)
        circuit.append(AnnotatedOperation(gate, ControlModifier(2)), [0, 1, 2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(gate.control(2, annotated=False), [0, 1, 2, 3])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_control_clifford(self):
        """Test lazy control synthesis of Clifford objects (no ``control`` method defined)."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        cliff = Clifford(qc)
        circuit = QuantumCircuit(4)
        circuit.append(AnnotatedOperation(cliff, ControlModifier(2)), [0, 1, 2, 3])
        transpiled_circuit = HighLevelSynthesis(basis_gates=["cx", "u"])(circuit)
        self.assertEqual(transpiled_circuit.count_ops().keys(), {"cx", "u"})

    def test_multiple_controls(self):
        """Test lazy controlled synthesis with multiple control modifiers."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), [ControlModifier(2), ControlModifier(1)])
        circuit = QuantumCircuit(5)
        circuit.append(lazy_gate1, [0, 1, 2, 3, 4])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(5)
        expected_circuit.append(SwapGate().control(3, annotated=False), [0, 1, 2, 3, 4])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_nested_controls(self):
        """Test lazy controlled synthesis of nested lazy gates."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), ControlModifier(2))
        lazy_gate2 = AnnotatedOperation(lazy_gate1, ControlModifier(1))
        circuit = QuantumCircuit(5)
        circuit.append(lazy_gate2, [0, 1, 2, 3, 4])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(5)
        expected_circuit.append(SwapGate().control(3, annotated=False), [0, 1, 2, 3, 4])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_nested_controls_permutation(self):
        """Test lazy controlled synthesis of ``PermutationGate`` with nested lazy gates.
        Note that ``PermutationGate`` currently does not have definition."""
        lazy_gate1 = AnnotatedOperation(PermutationGate([3, 1, 0, 2]), ControlModifier(2))
        lazy_gate2 = AnnotatedOperation(lazy_gate1, ControlModifier(1))
        circuit = QuantumCircuit(7)
        circuit.append(lazy_gate2, [0, 1, 2, 3, 4, 5, 6])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        self.assertEqual(Operator(circuit), Operator(transpiled_circuit))

    def test_inverse_basic_gates(self):
        """Test lazy inverse synthesis of basic gates (each has its class ``control`` method)."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), InverseModifier())
        lazy_gate2 = AnnotatedOperation(CXGate(), InverseModifier())
        lazy_gate3 = AnnotatedOperation(RZGate(np.pi / 4), InverseModifier())
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate1, [0, 2])
        circuit.append(lazy_gate2, [0, 1])
        circuit.append(lazy_gate3, [2])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        inverse_gate1 = SwapGate().inverse()
        inverse_gate2 = CXGate().inverse()
        inverse_gate3 = RZGate(np.pi / 4).inverse()
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(inverse_gate1, [0, 2])
        expected_circuit.append(inverse_gate2, [0, 1])
        expected_circuit.append(inverse_gate3, [2])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_inverse_custom_gates(self):
        """Test lazy control synthesis of custom gates (which inherits ``inverse`` method from
        ``Gate``).
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        gate = qc.to_gate()
        circuit = QuantumCircuit(2)
        circuit.append(AnnotatedOperation(gate, InverseModifier()), [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(gate.inverse(), [0, 1])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_inverse_clifford(self):
        """Test lazy inverse synthesis of Clifford objects (no ``inverse`` method defined)."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        cliff = Clifford(qc)
        circuit = QuantumCircuit(2)
        circuit.append(AnnotatedOperation(cliff, InverseModifier()), [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(cliff.to_instruction().inverse(), [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_two_inverses(self):
        """Test lazy controlled synthesis with multiple inverse modifiers (even)."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), [InverseModifier(), InverseModifier()])
        circuit = QuantumCircuit(2)
        circuit.append(lazy_gate1, [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(SwapGate().inverse().inverse(), [0, 1])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_three_inverses(self):
        """Test lazy controlled synthesis with multiple inverse modifiers (odd)."""
        lazy_gate1 = AnnotatedOperation(
            RZGate(np.pi / 4), [InverseModifier(), InverseModifier(), InverseModifier()]
        )
        circuit = QuantumCircuit(1)
        circuit.append(lazy_gate1, [0])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(1)
        expected_circuit.append(RZGate(np.pi / 4).inverse().inverse().inverse(), [0])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_nested_inverses(self):
        """Test lazy synthesis with nested lazy gates."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), InverseModifier())
        lazy_gate2 = AnnotatedOperation(lazy_gate1, InverseModifier())
        circuit = QuantumCircuit(2)
        circuit.append(lazy_gate2, [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(SwapGate(), [0, 1])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_nested_inverses_permutation(self):
        """Test lazy controlled synthesis of ``PermutationGate`` with nested lazy gates.
        Note that ``PermutationGate`` currently does not have definition."""
        lazy_gate1 = AnnotatedOperation(PermutationGate([3, 1, 0, 2]), InverseModifier())
        lazy_gate2 = AnnotatedOperation(lazy_gate1, InverseModifier())
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate2, [0, 1, 2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        self.assertEqual(Operator(circuit), Operator(transpiled_circuit))

    def test_power_posint_basic_gates(self):
        """Test lazy power synthesis of basic gates with positive and zero integer powers."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), PowerModifier(2))
        lazy_gate2 = AnnotatedOperation(CXGate(), PowerModifier(1))
        lazy_gate3 = AnnotatedOperation(RZGate(np.pi / 4), PowerModifier(3))
        lazy_gate4 = AnnotatedOperation(CXGate(), PowerModifier(0))
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate1, [0, 1])
        circuit.append(lazy_gate2, [1, 2])
        circuit.append(lazy_gate3, [3])
        circuit.append(lazy_gate4, [2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(SwapGate(), [0, 1])
        expected_circuit.append(SwapGate(), [0, 1])
        expected_circuit.append(CXGate(), [1, 2])
        expected_circuit.append(RZGate(np.pi / 4), [3])
        expected_circuit.append(RZGate(np.pi / 4), [3])
        expected_circuit.append(RZGate(np.pi / 4), [3])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_power_negint_basic_gates(self):
        """Test lazy power synthesis of basic gates with negative integer powers."""
        lazy_gate1 = AnnotatedOperation(CXGate(), PowerModifier(-1))
        lazy_gate2 = AnnotatedOperation(RZGate(np.pi / 4), PowerModifier(-3))
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate1, [0, 1])
        circuit.append(lazy_gate2, [2])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(CXGate(), [0, 1])
        expected_circuit.append(RZGate(-np.pi / 4), [2])
        expected_circuit.append(RZGate(-np.pi / 4), [2])
        expected_circuit.append(RZGate(-np.pi / 4), [2])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_power_float_basic_gates(self):
        """Test lazy power synthesis of basic gates with floating-point powers."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), PowerModifier(0.5))
        lazy_gate2 = AnnotatedOperation(CXGate(), PowerModifier(0.2))
        lazy_gate3 = AnnotatedOperation(RZGate(np.pi / 4), PowerModifier(-0.25))
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate1, [0, 1])
        circuit.append(lazy_gate2, [1, 2])
        circuit.append(lazy_gate3, [3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(SwapGate().power(0.5), [0, 1])
        expected_circuit.append(CXGate().power(0.2), [1, 2])
        expected_circuit.append(RZGate(np.pi / 4).power(-0.25), [3])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_power_custom_gates(self):
        """Test lazy power synthesis of custom gates with positive integer powers."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        gate = qc.to_gate()
        circuit = QuantumCircuit(2)
        circuit.append(AnnotatedOperation(gate, PowerModifier(3)), [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(gate, [0, 1])
        expected_circuit.append(gate, [0, 1])
        expected_circuit.append(gate, [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_power_posint_clifford(self):
        """Test lazy power synthesis of Clifford objects with positive integer power."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        cliff = Clifford(qc)
        circuit = QuantumCircuit(2)
        circuit.append(AnnotatedOperation(cliff, PowerModifier(3)), [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(cliff.to_instruction().power(3), [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_power_float_clifford(self):
        """Test lazy power synthesis of Clifford objects with floating-point power."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        cliff = Clifford(qc)
        circuit = QuantumCircuit(2)
        circuit.append(AnnotatedOperation(cliff, PowerModifier(-0.5)), [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(cliff.to_instruction().power(-0.5), [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_multiple_powers(self):
        """Test lazy controlled synthesis with multiple power modifiers."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), [PowerModifier(2), PowerModifier(-1)])
        circuit = QuantumCircuit(2)
        circuit.append(lazy_gate1, [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(SwapGate().power(-2), [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_nested_powers(self):
        """Test lazy synthesis with nested lazy gates."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), PowerModifier(2))
        lazy_gate2 = AnnotatedOperation(lazy_gate1, PowerModifier(-1))
        circuit = QuantumCircuit(2)
        circuit.append(lazy_gate2, [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(SwapGate().power(-2), [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_nested_powers_permutation(self):
        """Test lazy controlled synthesis of ``PermutationGate`` with nested lazy gates.
        Note that ``PermutationGate`` currently does not have definition."""
        lazy_gate1 = AnnotatedOperation(PermutationGate([3, 1, 0, 2]), PowerModifier(2))
        lazy_gate2 = AnnotatedOperation(lazy_gate1, PowerModifier(-1))
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate2, [0, 1, 2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        self.assertEqual(Operator(circuit), Operator(transpiled_circuit))

    def test_reordered_modifiers(self):
        """Test involving gates with different modifiers."""
        lazy_gate1 = AnnotatedOperation(
            PermutationGate([3, 1, 0, 2]), [InverseModifier(), ControlModifier(2), PowerModifier(3)]
        )
        lazy_gate2 = AnnotatedOperation(
            PermutationGate([3, 1, 0, 2]), [PowerModifier(3), ControlModifier(2), InverseModifier()]
        )
        qc1 = QuantumCircuit(6)
        qc1.append(lazy_gate1, [0, 1, 2, 3, 4, 5])
        qc2 = QuantumCircuit(6)
        qc2.append(lazy_gate2, [0, 1, 2, 3, 4, 5])
        self.assertEqual(Operator(qc1), Operator(qc2))
        transpiled1 = HighLevelSynthesis()(qc1)
        transpiled2 = HighLevelSynthesis()(qc2)
        self.assertEqual(Operator(transpiled1), Operator(transpiled2))
        self.assertEqual(Operator(qc1), Operator(transpiled1))

    def test_definition_with_annotations(self):
        """Test annotated gates with definitions involving another annotated gate.
        Note that passing basis_gates makes the pass recursive.
        """
        qc = QuantumCircuit(4)
        lazy_gate1 = AnnotatedOperation(PermutationGate([3, 1, 0, 2]), InverseModifier())
        lazy_gate2 = AnnotatedOperation(SwapGate(), ControlModifier(2))
        qc.append(lazy_gate1, [0, 1, 2, 3])
        qc.append(lazy_gate2, [0, 1, 2, 3])
        custom_gate = qc.to_gate()
        lazy_gate3 = AnnotatedOperation(custom_gate, ControlModifier(2))
        circuit = QuantumCircuit(6)
        circuit.append(lazy_gate3, [0, 1, 2, 3, 4, 5])

        with self.subTest(qubits_initially_zero=False):
            # When transpiling without assuming that qubits are initially zero,
            # we should have that the Operators before and after are equal.
            transpiled_circuit = HighLevelSynthesis(
                basis_gates=["cx", "u"], qubits_initially_zero=False
            )(circuit)
            self.assertEqual(Operator(circuit), Operator(transpiled_circuit))

        with self.subTest(qubits_initially_zero=True):
            # When transpiling assuming that qubits are initially zero,
            # we should have that the Statevectors before and after
            # are equal (but not the full Operators).
            transpiled_circuit = HighLevelSynthesis(
                basis_gates=["cx", "u"], qubits_initially_zero=True
            )(circuit)
            self.assertEqual(Statevector(circuit), Statevector(transpiled_circuit))

    def test_definition_with_high_level_objects(self):
        """Test annotated gates with definitions involving annotations and
        high-level-objects."""
        def_circuit = QuantumCircuit(4)
        def_circuit.append(
            AnnotatedOperation(PermutationGate([1, 0]), ControlModifier(2)), [0, 1, 2, 3]
        )
        gate = def_circuit.to_gate()
        circuit = QuantumCircuit(6)
        circuit.append(gate, [0, 1, 2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(6)
        expected_circuit.append(SwapGate().control(2, annotated=False), [0, 1, 2, 3])
        self.assertEqual(circuit, transpiled_circuit)

    def test_control_high_level_object(self):
        """Test synthesis of high level gates with control modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, ControlModifier(1))
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1, 2])
        backend = GenericBackendV2(num_qubits=5, basis_gates=["u", "cx"])
        qct = HighLevelSynthesis(target=backend.target)(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_transpile_control_high_level_object(self):
        """Test full transpilation of high level gates with control modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, ControlModifier(1))
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1, 2])
        backend = GenericBackendV2(num_qubits=5, basis_gates=["u", "cx"])
        qct = transpile(qc, target=backend.target)
        ops = qct.count_ops().keys()
        for op in ops:
            self.assertIn(op, ["u", "cx", "ecr", "measure"])

    def test_inverse_high_level_object(self):
        """Test synthesis of high level gates with inverse modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, InverseModifier())
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1])
        backend = GenericBackendV2(num_qubits=5, basis_gates=["u", "cx"])
        qct = HighLevelSynthesis(target=backend.target)(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_transpile_inverse_high_level_object(self):
        """Test synthesis of high level gates with inverse modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, InverseModifier())
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1])
        backend = GenericBackendV2(num_qubits=5, basis_gates=["u", "cx"])
        qct = transpile(qc, target=backend.target)
        ops = qct.count_ops().keys()
        for op in ops:
            self.assertIn(op, ["u", "cx", "ecr", "measure"])

    def test_power_high_level_object(self):
        """Test synthesis of high level gates with control modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, PowerModifier(3))
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1])
        backend = GenericBackendV2(num_qubits=5, basis_gates=["u", "cx"])
        qct = HighLevelSynthesis(target=backend.target)(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_transpile_power_high_level_object(self):
        """Test full transpilation of high level gates with control modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, PowerModifier(3))
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1])
        backend = GenericBackendV2(num_qubits=5, basis_gates=["u", "cx"])
        qct = transpile(qc, target=backend.target)
        ops = qct.count_ops().keys()
        for op in ops:
            self.assertIn(op, ["u", "cx", "ecr", "measure"])

    def test_simple_circuit(self):
        """Test HLS on a simple circuit."""
        qc = QuantumCircuit(3)
        qc.cz(1, 2)
        pass_ = HighLevelSynthesis(basis_gates=["cx", "u"])
        qct = pass_(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_simple_circuit2(self):
        """Test HLS on a simple circuit."""
        qc = QuantumCircuit(6)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 3)
        qc.h(5)
        pass_ = HighLevelSynthesis(basis_gates=["cx", "u", "h"])
        qct = pass_(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_circuit_with_recursive_def(self):
        """Test recursive synthesis of the definition circuit."""
        inner = QuantumCircuit(2)
        inner.cz(0, 1)
        qc = QuantumCircuit(3)
        qc.append(inner.to_gate(), [0, 2])
        pass_ = HighLevelSynthesis(basis_gates=["cx", "u"])
        qct = pass_(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_circuit_with_recursive_def2(self):
        """Test recursive synthesis of the definition circuit."""
        inner1 = QuantumCircuit(2)
        inner1.cz(0, 1)
        qc = QuantumCircuit(4)
        qc.append(inner1.to_instruction(), [2, 3])
        pass_ = HighLevelSynthesis(basis_gates=["cz", "cx", "u"])
        qct = pass_(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_circuit_with_recursive_def3(self):
        """Test recursive synthesis of the definition circuit."""
        inner2 = QuantumCircuit(2)
        inner2.h(0)
        inner2.cx(0, 1)

        inner1 = QuantumCircuit(4)
        inner1.cz(0, 1)
        inner1.append(inner2.to_instruction(), [0, 2])

        qc = QuantumCircuit(6)
        qc.h(1)
        qc.h(2)
        qc.cz(1, 2)
        qc.append(inner1.to_instruction(), [2, 0, 4, 3])
        qc.h(2)
        pass_ = HighLevelSynthesis(basis_gates=["h", "z", "cx", "u"])
        qct = pass_(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_circuit_with_mcx(self):
        """Test synthesis with plugins."""
        qc = QuantumCircuit(10)
        qc.mcx([3, 4, 5, 6, 7], 2)
        basis_gates = ["u", "cx"]
        qct = HighLevelSynthesis(basis_gates=basis_gates)(qc)
        self.assertEqual(Statevector(qc), Statevector(qct))

    def test_circuit_with_mcx_def(self):
        """Test synthesis where the plugin is called within the recursive call
        on the definition."""
        circuit = QuantumCircuit(6)
        circuit.mcx([0, 1, 2, 3, 4], 5)
        custom_gate = circuit.to_gate()
        qc = QuantumCircuit(10)
        qc.append(custom_gate, [3, 4, 5, 6, 7, 2])
        basis_gates = ["u", "cx"]
        qct = HighLevelSynthesis(basis_gates=basis_gates)(qc)
        self.assertEqual(Statevector(qc), Statevector(qct))

    def test_circuit_with_mcx_def_rec(self):
        """Test synthesis where the plugin is called within the recursive call
        on the definition."""
        inner2 = QuantumCircuit(6)
        inner2.mcx([0, 1, 2, 3, 4], 5)
        inner1 = QuantumCircuit(7)
        inner1.append(inner2.to_gate(), [1, 2, 3, 4, 5, 6])
        qc = QuantumCircuit(10)
        qc.append(inner1.to_gate(), [2, 3, 4, 5, 6, 7, 8])
        pass_ = HighLevelSynthesis(basis_gates=["h", "z", "cx", "u"])
        qct = pass_(qc)
        self.assertEqual(Statevector(qc), Statevector(qct))

    def test_annotated_gate(self):
        """Test synthesis with annotated gate."""
        qc = QuantumCircuit(10)
        qc.x(1)
        qc.cz(1, 2)
        qc.append(SGate().control(3, annotated=True), [0, 1, 8, 9])
        pass_ = HighLevelSynthesis(basis_gates=["h", "z", "cx", "u"])
        qct = pass_(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_annotated_circuit(self):
        """Test synthesis with annotated custom gate."""
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cy(0, 1)
        qc = QuantumCircuit(10)
        qc.x(1)
        qc.cz(1, 2)
        qc.append(circ.to_gate().control(3, annotated=True), [2, 0, 3, 7, 8])
        pass_ = HighLevelSynthesis(basis_gates=["h", "z", "cx", "u"])
        qct = pass_(qc)
        self.assertEqual(Statevector(qc), Statevector(qct))

    def test_annotated_circuit_with_phase(self):
        """Test controlled-annotated circuits with global phase."""
        inner = QuantumCircuit(2)
        inner.global_phase = 1
        inner.h(0)
        inner.cx(0, 1)
        gate = inner.to_gate()

        qc1 = QuantumCircuit(3)
        qc1.append(gate.control(annotated=False), [0, 1, 2])
        qct1 = HighLevelSynthesis(basis_gates=["cx", "u"])(qc1)

        qc2 = QuantumCircuit(3)
        qc2.append(gate.control(annotated=True), [0, 1, 2])
        qct2 = HighLevelSynthesis(basis_gates=["cx", "u"])(qc2)

        self.assertEqual(Operator(qc1), Operator(qc2))
        self.assertEqual(Operator(qct1), Operator(qct2))

    def test_annotated_rec(self):
        """Test synthesis with annotated custom gates and recursion."""
        inner2 = QuantumCircuit(2)
        inner2.h(0)
        inner2.cy(0, 1)
        inner1 = QuantumCircuit(5)
        inner1.h(1)
        inner1.append(inner2.to_gate().control(2, annotated=True), [1, 2, 3, 4])
        qc = QuantumCircuit(10)
        qc.x(1)
        qc.cz(1, 2)
        qc.append(inner1.to_gate().control(3, annotated=True), [9, 8, 7, 6, 5, 4, 3, 2])
        pass_ = HighLevelSynthesis(basis_gates=["h", "z", "cx", "u"])
        qct = pass_(qc)
        self.assertEqual(Statevector(qc), Statevector(qct))

    def test_annotated_with_empty_modifiers(self):
        """Test synthesis of an annotated gate with an empty list of modifiers."""
        annotated_gate = AnnotatedOperation(SwapGate(), [])
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.append(annotated_gate, [0, 1])

        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.h(0)
        expected_circuit.swap(0, 1)

        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_annotated_rec_with_control_states(self):
        """Test that control states are combined correctly."""
        # qc1 contains h.control('10').control('111')
        inner2 = QuantumCircuit(1)
        inner2.h(0)
        inner1 = QuantumCircuit(3)
        inner1.append(inner2.to_gate().control(2, ctrl_state=2, annotated=True), [0, 1, 2])
        qc1 = QuantumCircuit(6)
        qc1.append(inner1.to_gate().control(3, annotated=True, ctrl_state=7), [0, 1, 2, 3, 4, 5])

        # qc2 contains h.control('10111')
        qc2 = QuantumCircuit(6)
        qc2.append(inner2.to_gate().control(5, annotated=True, ctrl_state=23), [0, 1, 2, 3, 4, 5])

        pass_ = HighLevelSynthesis(basis_gates=["h", "z", "cx", "u"], qubits_initially_zero=False)
        qct1 = pass_(qc1)
        qct2 = pass_(qc2)

        self.assertEqual(Operator(qct1), Operator(qct2))


class TestUnrollerCompatability(QiskitTestCase):
    """Tests backward compatibility with the UnrollCustomDefinitions pass.

    Duplicate of TestUnrollerCompatability from test.python.transpiler.test_basis_translator,
    with UnrollCustomDefinitions replaced by HighLevelSynthesis.
    """

    def test_basic_unroll(self):
        """Test decompose a single H into u2."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["u2"])
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ["u2"])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, "u2")

    def test_unroll_toffoli(self):
        """Test unroll toffoli on multi regs to h, t, tdg, cx."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(1, "qr2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(
            equivalence_library=std_eqlib, basis_gates=["h", "t", "tdg", "cx"]
        )
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ["h", "t", "tdg", "cx"])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            self.assertIn(node.name, ["h", "t", "tdg", "cx"])

    def test_basic_unroll_min_qubits(self):
        """Test decompose a single H into u2."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["u2"], min_qubits=3)
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ["u2"], min_qubits=3)
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, "h")

    def test_unroll_toffoli_min_qubits(self):
        """Test unroll toffoli on multi regs to h, t, tdg, cx."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(1, "qr2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        circuit.sx(qr1[0])
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(
            equivalence_library=std_eqlib, basis_gates=["h", "t", "tdg", "cx"], min_qubits=3
        )
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ["h", "t", "tdg", "cx"], min_qubits=3)
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 16)
        for node in op_nodes:
            self.assertIn(node.name, ["h", "t", "tdg", "cx", "sx"])

    def test_unroll_no_basis(self):
        """Test when a given gate has no decompositions."""
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=[])
        dag = pass_.run(dag)

        pass_ = BasisTranslator(std_eqlib, [])

        with self.assertRaises(QiskitError):
            pass_.run(dag)

    def test_unroll_all_instructions(self):
        """Test unrolling a circuit containing all standard instructions."""

        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.crx(0.5, qr[1], qr[2])
        circuit.cry(0.5, qr[1], qr[2])
        circuit.ccx(qr[0], qr[1], qr[2])
        circuit.ch(qr[0], qr[2])
        circuit.crz(0.5, qr[1], qr[2])
        circuit.cswap(qr[1], qr[0], qr[2])
        circuit.append(CU1Gate(0.1), [qr[0], qr[2]])
        circuit.append(CU3Gate(0.2, 0.1, 0.0), [qr[1], qr[2]])
        circuit.cx(qr[1], qr[0])
        circuit.cy(qr[1], qr[2])
        circuit.cz(qr[2], qr[0])
        circuit.h(qr[1])
        circuit.rx(0.1, qr[0])
        circuit.ry(0.2, qr[1])
        circuit.rz(0.3, qr[2])
        circuit.rzz(0.6, qr[1], qr[0])
        circuit.s(qr[0])
        circuit.sdg(qr[1])
        circuit.swap(qr[1], qr[2])
        circuit.t(qr[2])
        circuit.tdg(qr[0])
        circuit.append(U1Gate(0.1), [qr[1]])
        circuit.append(U2Gate(0.2, -0.1), [qr[0]])
        circuit.append(U3Gate(0.3, 0.0, -0.1), [qr[2]])
        circuit.x(qr[2])
        circuit.y(qr[1])
        circuit.z(qr[0])
        # circuit.snapshot('0')
        # circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["u3", "cx", "id"])
        dag = pass_.run(dag)

        pass_ = BasisTranslator(std_eqlib, ["u3", "cx", "id"])
        unrolled_dag = pass_.run(dag)

        ref_circuit = QuantumCircuit(qr, cr)
        ref_circuit.append(U3Gate(0, 0, np.pi / 2), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(-0.25, 0, 0), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0.25, -np.pi / 2, 0), [qr[2]])
        ref_circuit.append(U3Gate(0.25, 0, 0), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(-0.25, 0, 0), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.cx(qr[0], qr[1])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[1]])
        ref_circuit.cx(qr[0], qr[1])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 2), [qr[2]])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 2), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, 0.25), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, -0.25), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[1]])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, 0.05), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.append(U3Gate(0, 0, 0.05), [qr[0]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, -0.05), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, 0.05), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, -0.05), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(-0.1, 0, -0.05), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[0]])
        ref_circuit.append(U3Gate(0.1, 0.1, 0), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 2), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[1]])
        ref_circuit.append(U3Gate(0.2, 0, 0), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 2), [qr[2]])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[0]])
        ref_circuit.append(U3Gate(0.1, -np.pi / 2, np.pi / 2), [qr[0]])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, 0.6), [qr[0]])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, np.pi / 2), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(np.pi / 2, 0.2, -0.1), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, np.pi), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 2), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, 0.3), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[2], qr[1])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, 0.1), [qr[1]])
        ref_circuit.append(U3Gate(np.pi, np.pi / 2, np.pi / 2), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(0.3, 0.0, -0.1), [qr[2]])
        ref_circuit.append(U3Gate(np.pi, 0, np.pi), [qr[2]])
        # ref_circuit.snapshot('0')
        # ref_circuit.measure(qr, cr)
        # ref_dag = circuit_to_dag(ref_circuit)

        self.assertTrue(Operator(dag_to_circuit(unrolled_dag)).equiv(ref_circuit))

    def test_simple_unroll_parameterized_without_expressions(self):
        """Verify unrolling parameterized gates without expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")

        qc.rz(theta, qr[0])
        dag = circuit_to_dag(qc)

        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["u1", "cx"])
        dag = pass_.run(dag)

        unrolled_dag = BasisTranslator(std_eqlib, ["u1", "cx"]).run(dag)

        expected = QuantumCircuit(qr, global_phase=-theta / 2)
        expected.append(U1Gate(theta), [qr[0]])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_simple_unroll_parameterized_with_expressions(self):
        """Verify unrolling parameterized gates with expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_ = theta + phi

        qc.rz(sum_, qr[0])
        dag = circuit_to_dag(qc)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["p", "cx"])
        dag = pass_.run(dag)

        unrolled_dag = BasisTranslator(std_eqlib, ["p", "cx"]).run(dag)

        expected = QuantumCircuit(qr, global_phase=-sum_ / 2)
        expected.p(sum_, qr[0])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_definition_unroll_parameterized(self):
        """Verify that unrolling complex gates with parameters does not raise."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")

        qc.cp(theta, qr[1], qr[0])
        qc.cp(theta * theta, qr[0], qr[1])
        dag = circuit_to_dag(qc)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["p", "cx"])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(std_eqlib, ["p", "cx"]).run(dag)

        self.assertEqual(out_dag.count_ops(), {"p": 6, "cx": 4})

    def test_unrolling_parameterized_composite_gates(self):
        """Verify unrolling circuits with parameterized composite gates."""
        mock_sel = EquivalenceLibrary(base=std_eqlib)

        qr1 = QuantumRegister(2)
        subqc = QuantumCircuit(qr1)

        theta = Parameter("theta")

        subqc.rz(theta, qr1[0])
        subqc.cx(qr1[0], qr1[1])
        subqc.rz(theta, qr1[1])

        # Expanding across register with shared parameter
        qr2 = QuantumRegister(4)
        qc = QuantumCircuit(qr2)

        sub_instr = circuit_to_instruction(subqc, equivalence_library=mock_sel)
        qc.append(sub_instr, [qr2[0], qr2[1]])
        qc.append(sub_instr, [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        pass_ = HighLevelSynthesis(equivalence_library=mock_sel, basis_gates=["p", "cx"])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(mock_sel, ["p", "cx"]).run(dag)

        # Pick up -1 * theta / 2 global phase four twice (once for each RZ -> P
        # in each of the two sub_instr instructions).
        expected = QuantumCircuit(qr2, global_phase=-1 * 4 * theta / 2.0)
        expected.p(theta, qr2[0])
        expected.p(theta, qr2[2])
        expected.cx(qr2[0], qr2[1])
        expected.cx(qr2[2], qr2[3])
        expected.p(theta, qr2[1])
        expected.p(theta, qr2[3])

        self.assertEqual(circuit_to_dag(expected), out_dag)

        # Expanding across register with shared parameter
        qc = QuantumCircuit(qr2)

        phi = Parameter("phi")
        gamma = Parameter("gamma")

        sub_instr = circuit_to_instruction(subqc, {theta: phi}, mock_sel)
        qc.append(sub_instr, [qr2[0], qr2[1]])
        sub_instr = circuit_to_instruction(subqc, {theta: gamma}, mock_sel)
        qc.append(sub_instr, [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        pass_ = HighLevelSynthesis(equivalence_library=mock_sel, basis_gates=["p", "cx"])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(mock_sel, ["p", "cx"]).run(dag)

        expected = QuantumCircuit(qr2, global_phase=-1 * (2 * phi + 2 * gamma) / 2.0)
        expected.p(phi, qr2[0])
        expected.p(gamma, qr2[2])
        expected.cx(qr2[0], qr2[1])
        expected.cx(qr2[2], qr2[3])
        expected.p(phi, qr2[1])
        expected.p(gamma, qr2[3])

        self.assertEqual(circuit_to_dag(expected), out_dag)

    def test_unroll_with_clbit(self):
        """Test unrolling a custom definition that has qubits and clbits."""
        block = QuantumCircuit(1, 1)
        block.h(0)
        block.measure(0, 0)

        circuit = QuantumCircuit(1, 1)
        circuit.append(block.to_instruction(), [0], [0])

        hls = HighLevelSynthesis(basis_gates=["h", "measure"])
        out = hls(circuit)

        self.assertEqual(block, out)

    def test_unroll_with_clbit_mapping(self):
        """Test unrolling a custom definition that has qubits and clbits
        that require mapping to the global clbits.
        Regression test for: https://github.com/Qiskit/qiskit/issues/14569
        """
        block = QuantumCircuit(2, 2)
        block.h(0)
        block.measure([0, 1], [0, 1])

        circuit = QuantumCircuit(6, 6)
        circuit.append(block.to_instruction(), [0, 1], [0, 1])
        circuit.append(block.to_instruction(), [2, 3], [3, 2])
        circuit.append(block.to_instruction(), [4, 5], [4, 5])

        hls = HighLevelSynthesis(basis_gates=["h", "measure"])
        out = hls(circuit)

        self.assertEqual(
            (out.find_bit(out.data[3].qubits[0]).index, out.find_bit(out.data[3].clbits[0]).index),
            (0, 0),
        )
        self.assertEqual(
            (out.find_bit(out.data[4].qubits[0]).index, out.find_bit(out.data[4].clbits[0]).index),
            (1, 1),
        )
        self.assertEqual(
            (out.find_bit(out.data[5].qubits[0]).index, out.find_bit(out.data[5].clbits[0]).index),
            (3, 2),
        )
        self.assertEqual(
            (out.find_bit(out.data[6].qubits[0]).index, out.find_bit(out.data[6].clbits[0]).index),
            (2, 3),
        )
        self.assertEqual(
            (out.find_bit(out.data[7].qubits[0]).index, out.find_bit(out.data[7].clbits[0]).index),
            (4, 4),
        )
        self.assertEqual(
            (out.find_bit(out.data[8].qubits[0]).index, out.find_bit(out.data[8].clbits[0]).index),
            (5, 5),
        )


class TestGate(Gate):
    """Mock one qubit zero param gate."""

    def __init__(self):
        super().__init__("tg", 1, [])


class TestCompositeGate(Gate):
    """Mock one qubit zero param gate."""

    def __init__(self):
        super().__init__("tcg", 1, [])


class TestUnrollCustomDefinitionsCompatibility(QiskitTestCase):
    """Tests backward compatibility with the UnrollCustomDefinitions pass.

    Duplicate of TestUnrollCustomDefinitions from test.python.transpiler.test_unroll_custom_definitions,
    with UnrollCustomDefinitions replaced by HighLevelSynthesis.
    """

    def test_dont_unroll_a_gate_in_eq_lib(self):
        """Verify we don't unroll a gate found in equivalence_library."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["u3", "cx"]).run(dag)

        expected = qc.copy()
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_dont_unroll_a_gate_in_basis_gates(self):
        """Verify we don't unroll a gate in basis_gates."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["u3", "cx", "tg"]).run(
            dag
        )

        expected = qc.copy()
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_raise_for_opaque_not_in_eq_lib(self):
        """Verify we raise for an opaque gate not in basis_gates or eq_lib."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        with self.assertRaisesRegex(QiskitError, "unable to synthesize"):
            HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["u3", "cx"]).run(dag)

    def test_unroll_gate_until_reach_basis_gates(self):
        """Verify we unroll gates until we hit basis_gates."""
        eq_lib = EquivalenceLibrary()

        gate = TestCompositeGate()
        q = QuantumRegister(1, "q")
        gate.definition = QuantumCircuit(q)
        gate.definition.append(TestGate(), [q[0]], [])

        qc = QuantumCircuit(q)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["u3", "cx", "tg"]).run(
            dag
        )

        expected = QuantumCircuit(1)
        expected.append(TestGate(), [0])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_unroll_twice_until_we_get_to_eqlib(self):
        """Verify we unroll gates until we hit basis_gates."""
        eq_lib = EquivalenceLibrary()

        base_gate = TestGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)

        eq_lib.add_equivalence(base_gate, equiv)

        gate = TestCompositeGate()

        q = QuantumRegister(1, "q")
        gate.definition = QuantumCircuit(q)
        gate.definition.append(TestGate(), [q[0]], [])

        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["u3", "cx"]).run(dag)

        expected = QuantumCircuit(1)
        expected.append(TestGate(), [0])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_if_else(self):
        """Test that a simple if-else unrolls correctly."""
        eq_lib = EquivalenceLibrary()

        equiv = QuantumCircuit(1)
        equiv.h(0)
        eq_lib.add_equivalence(TestGate(), equiv)

        equiv = QuantumCircuit(1)
        equiv.z(0)
        eq_lib.add_equivalence(TestCompositeGate(), equiv)

        pass_ = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["h", "z", "cx"])

        true_body = QuantumCircuit(1)
        true_body.h(0)
        true_body.append(TestGate(), [0])
        false_body = QuantumCircuit(1)
        false_body.append(TestCompositeGate(), [0])

        test = QuantumCircuit(1, 1)
        test.h(0)
        test.measure(0, 0)
        test.if_else((0, True), true_body, false_body, [0], [])

        expected = QuantumCircuit(1, 1)
        expected.h(0)
        expected.measure(0, 0)
        expected.if_else((0, True), pass_(true_body), pass_(false_body), [0], [])

        self.assertEqual(pass_(test), expected)

    def test_nested_control_flow(self):
        """Test that the unroller recurses into nested control flow."""
        eq_lib = EquivalenceLibrary()
        base_gate = TestGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)
        eq_lib.add_equivalence(base_gate, equiv)
        base_gate = TestCompositeGate()
        equiv = QuantumCircuit(1)
        equiv.z(0)
        eq_lib.add_equivalence(base_gate, equiv)

        pass_ = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["h", "z", "cx"])

        qubit = Qubit()
        clbit = Clbit()

        for_body = QuantumCircuit(1)
        for_body.append(TestGate(), [0], [])

        while_body = QuantumCircuit(1)
        while_body.append(TestCompositeGate(), [0], [])

        true_body = QuantumCircuit([qubit, clbit])
        true_body.while_loop((clbit, True), while_body, [0], [])

        test = QuantumCircuit([qubit, clbit])
        test.for_loop(range(2), None, for_body, [0], [])
        test.if_else((clbit, True), true_body, None, [0], [0])

        expected_if_body = QuantumCircuit([qubit, clbit])
        expected_if_body.while_loop((clbit, True), pass_(while_body), [0], [])
        expected = QuantumCircuit([qubit, clbit])
        expected.for_loop(range(2), None, pass_(for_body), [0], [])
        expected.if_else(range(2), pass_(expected_if_body), None, [0], [0])

        self.assertEqual(pass_(test), expected)

    def test_dont_unroll_a_gate_in_basis_gates_with_target(self):
        """Verify we don't unroll a gate in basis_gates."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        target = Target(num_qubits=1)
        target.add_instruction(U3Gate(Parameter("a"), Parameter("b"), Parameter("c")))
        target.add_instruction(CXGate())
        target.add_instruction(TestGate())

        out = HighLevelSynthesis(equivalence_library=eq_lib, target=target).run(dag)

        expected = qc.copy()
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_raise_for_opaque_not_in_eq_lib_target_with_target(self):
        """Verify we raise for an opaque gate not in basis_gates or eq_lib."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        target = Target(num_qubits=1)
        target.add_instruction(U3Gate(Parameter("a"), Parameter("b"), Parameter("c")))
        target.add_instruction(CXGate())

        dag = circuit_to_dag(qc)
        with self.assertRaisesRegex(QiskitError, "unable to synthesize"):
            HighLevelSynthesis(equivalence_library=eq_lib, target=target).run(dag)

    def test_unroll_gate_until_reach_basis_gates_with_target(self):
        """Verify we unroll gates until we hit basis_gates."""
        eq_lib = EquivalenceLibrary()

        gate = TestCompositeGate()
        q = QuantumRegister(1, "q")
        gate.definition = QuantumCircuit(q)
        gate.definition.append(TestGate(), [q[0]], [])

        qc = QuantumCircuit(q)
        qc.append(gate, [0])

        target = Target(num_qubits=1)
        target.add_instruction(U3Gate(Parameter("a"), Parameter("b"), Parameter("c")))
        target.add_instruction(CXGate())
        target.add_instruction(TestGate())

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, target=target).run(dag)

        expected = QuantumCircuit(1)
        expected.append(TestGate(), [0])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_unroll_twice_until_we_get_to_eqlib_with_target(self):
        """Verify we unroll gates until we hit basis_gates."""
        eq_lib = EquivalenceLibrary()

        base_gate = TestGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)

        eq_lib.add_equivalence(base_gate, equiv)

        gate = TestCompositeGate()

        q = QuantumRegister(1, "q")
        gate.definition = QuantumCircuit(q)
        gate.definition.append(TestGate(), [q[0]], [])

        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        target = Target(num_qubits=1)
        target.add_instruction(U3Gate(Parameter("a"), Parameter("b"), Parameter("c")))
        target.add_instruction(CXGate())

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, target=target).run(dag)

        expected = QuantumCircuit(1)
        expected.append(TestGate(), [0])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_unroll_empty_definition(self):
        """Test that a gate with no operations can be unrolled."""
        qc = QuantumCircuit(2)
        qc.append(QuantumCircuit(2).to_gate(), [0, 1], [])
        pass_ = HighLevelSynthesis(equivalence_library=EquivalenceLibrary(), basis_gates=["u"])
        expected = QuantumCircuit(2)
        self.assertEqual(pass_(qc), expected)

    def test_unroll_empty_definition_with_phase(self):
        """Test that a gate with no operations but with a global phase can be unrolled."""
        qc = QuantumCircuit(2)
        qc.append(QuantumCircuit(2, global_phase=0.5).to_gate(), [0, 1], [])
        pass_ = HighLevelSynthesis(equivalence_library=EquivalenceLibrary(), basis_gates=["u"])
        expected = QuantumCircuit(2, global_phase=0.5)
        self.assertEqual(pass_(qc), expected)

    def test_leave_store_alone_basis(self):
        """Don't attempt to synthesize `Store` instructions with basis gates."""

        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["u", "cx"])

        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        qc = QuantumCircuit(2, inputs=[a])
        qc.add_var(b, a)
        qc.compose(bell, [0, 1], inplace=True)
        qc.store(b, a)

        expected = qc.copy_empty_like()
        expected.store(b, a)
        expected.compose(pass_(bell), [0, 1], inplace=True)
        expected.store(b, a)

        self.assertEqual(pass_(qc), expected)

    def test_leave_store_alone_with_target(self):
        """Don't attempt to synthesize `Store` instructions with a `Target`."""

        # Note no store.
        target = Target()
        target.add_instruction(
            UGate(Parameter("a"), Parameter("b"), Parameter("c")), {(0,): None, (1,): None}
        )
        target.add_instruction(CXGate(), {(0, 1): None, (1, 0): None})

        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, target=target)

        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        qc = QuantumCircuit(2, inputs=[a])
        qc.add_var(b, a)
        qc.compose(bell, [0, 1], inplace=True)
        qc.store(b, a)

        expected = qc.copy_empty_like()
        expected.store(b, a)
        expected.compose(pass_(bell), [0, 1], inplace=True)
        expected.store(b, a)

        self.assertEqual(pass_(qc), expected)


@ddt
class TestQFTSynthesisPlugins(QiskitTestCase):
    """Tests related to plugins for QFTGate."""

    def test_supported_names(self):
        """Test that there is a default synthesis plugin for QFTGates."""
        supported_plugin_names = high_level_synthesis_plugin_names("qft")
        self.assertIn("default", supported_plugin_names)

    @data("line", "full")
    def test_qft_plugins_qft(self, qft_plugin_name):
        """Test QFTSynthesisLine plugin for circuits with QFTGates."""
        qc = QuantumCircuit(4)
        qc.append(QFTGate(3), [0, 1, 2])
        qc.cx(1, 3)
        qc.append(QFTGate(3).inverse(), [0, 1, 2])
        hls_config = HLSConfig(qft=[qft_plugin_name])
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        qct = hls_pass(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    @data("line", "full")
    def test_qft_line_plugin_annotated_qft(self, qft_plugin_name):
        """Test QFTSynthesisLine plugin for circuits with annotated QFTGates."""
        qc = QuantumCircuit(4)
        qc.append(QFTGate(3).inverse(annotated=True).control(annotated=True), [0, 1, 2, 3])
        hls_config = HLSConfig(qft=[qft_plugin_name])
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        qct = hls_pass(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    @data("line", "full")
    def test_skip_non_qft(self, qft_plugin_name):
        """Test that synthesis plugins are not applied on gates that are called `qft`, yet
        that are not of type `QFTGate`.
        """
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1, name="qft")
        qc2.s(0)
        qc.append(qc2.to_instruction(), qc.qregs[0])
        hls_config = HLSConfig(qft=[qft_plugin_name])
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        qct = hls_pass(qc)
        # HighLevelSynthesis should replace the custom gate called "qft"
        # by the user-provided definition.
        self.assertEqual(Operator(qc2), Operator(qct))


@ddt
class TestMCXSynthesisPlugins(QiskitTestCase):
    """Tests related to plugins for MCXGate."""

    def test_supported_names(self):
        """Test that there is a default synthesis plugin for MCXGate."""
        supported_plugin_names = high_level_synthesis_plugin_names("mcx")
        self.assertIn("default", supported_plugin_names)

    def test_mcx_plugins_applicability(self):
        """Test applicability of MCX synthesis plugins for MCX gates."""
        gate = MCXGate(5)

        with self.subTest(method="n_clean_m15", num_clean_ancillas=4, num_dirty_ancillas=4):
            # should have a decomposition
            decomposition = MCXSynthesisNCleanM15().run(
                gate, num_clean_ancillas=4, num_dirty_ancillas=4
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="n_clean_m15", num_clean_ancillas=2, num_dirty_ancillas=4):
            # should not have a decomposition
            decomposition = MCXSynthesisNCleanM15().run(
                gate, num_clean_ancillas=2, num_dirty_ancillas=4
            )
            self.assertIsNone(decomposition)

        with self.subTest(method="n_dirty_i15", num_clean_ancillas=4, num_dirty_ancillas=4):
            # should have a decomposition
            decomposition = MCXSynthesisNDirtyI15().run(
                gate, num_clean_ancillas=4, num_dirty_ancillas=4
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="n_dirty_i15", num_clean_ancillas=2, num_dirty_ancillas=2):
            # should have a decomposition
            decomposition = MCXSynthesisNDirtyI15().run(
                gate, num_clean_ancillas=2, num_dirty_ancillas=2
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="n_dirty_i15", num_clean_ancillas=1, num_dirty_ancillas=1):
            # should not have a decomposition
            decomposition = MCXSynthesisNDirtyI15().run(
                gate, num_clean_ancillas=1, num_dirty_ancillas=1
            )
            self.assertIsNone(decomposition)

        with self.subTest(method="2_clean_kg24", num_clean_ancillas=2, num_dirty_ancillas=0):
            # should have a decomposition
            decomposition = MCXSynthesis2CleanKG24().run(
                gate, num_clean_ancillas=2, num_dirty_ancillas=0
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="2_clean_kg24", num_clean_ancillas=1, num_dirty_ancillas=1):
            # should not have a decomposition
            decomposition = MCXSynthesis2CleanKG24().run(
                gate, num_clean_ancillas=1, num_dirty_ancillas=1
            )
            self.assertIsNone(decomposition)

        with self.subTest(method="2_clean_kg24", num_clean_ancillas=0, num_dirty_ancillas=0):
            # should not have a decomposition
            decomposition = MCXSynthesis2CleanKG24().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=0
            )
            self.assertIsNone(decomposition)

        with self.subTest(method="2_dirty_kg24", num_clean_ancillas=0, num_dirty_ancillas=2):
            # should have a decomposition
            decomposition = MCXSynthesis2DirtyKG24().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=2
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="2_dirty_kg24", num_clean_ancillas=1, num_dirty_ancillas=1):
            # should have a decomposition
            decomposition = MCXSynthesis2DirtyKG24().run(
                gate, num_clean_ancillas=1, num_dirty_ancillas=1
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="2_dirty_kg24", num_clean_ancillas=0, num_dirty_ancillas=1):
            # should not have a decomposition
            decomposition = MCXSynthesis2DirtyKG24().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=1
            )
            self.assertIsNone(decomposition)

        with self.subTest(method="2_dirty_kg24", num_clean_ancillas=0, num_dirty_ancillas=0):
            # should not have a decomposition
            decomposition = MCXSynthesis2DirtyKG24().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=0
            )
            self.assertIsNone(decomposition)

        with self.subTest(method="1_clean_kg24", num_clean_ancillas=1, num_dirty_ancillas=0):
            # should have a decomposition
            decomposition = MCXSynthesis1CleanKG24().run(
                gate, num_clean_ancillas=1, num_dirty_ancillas=0
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="1_clean_kg24", num_clean_ancillas=0, num_dirty_ancillas=1):
            # should not have a decomposition
            decomposition = MCXSynthesis1CleanKG24().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=1
            )
            self.assertIsNone(decomposition)

        with self.subTest(method="1_dirty_kg24", num_clean_ancillas=0, num_dirty_ancillas=1):
            # should have a decomposition
            decomposition = MCXSynthesis1DirtyKG24().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=1
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="1_dirty_kg24", num_clean_ancillas=1, num_dirty_ancillas=0):
            # should have a decomposition
            decomposition = MCXSynthesis1DirtyKG24().run(
                gate, num_clean_ancillas=1, num_dirty_ancillas=0
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="1_dirty_kg24", num_clean_ancillas=0, num_dirty_ancillas=0):
            # should not have a decomposition
            decomposition = MCXSynthesis1DirtyKG24().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=0
            )
            self.assertIsNone(decomposition)

        with self.subTest(method="1_clean_b95", num_clean_ancillas=1, num_dirty_ancillas=0):
            # should have a decomposition
            decomposition = MCXSynthesis1CleanB95().run(
                gate, num_clean_ancillas=1, num_dirty_ancillas=0
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="1_clean_b95", num_clean_ancillas=0, num_dirty_ancillas=1):
            # should not have a decomposition
            decomposition = MCXSynthesis1CleanB95().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=1
            )
            self.assertIsNone(decomposition)

        with self.subTest(method="noaux_v24", num_clean_ancillas=1, num_dirty_ancillas=1):
            # should have a decomposition
            decomposition = MCXSynthesisNoAuxV24().run(
                gate, num_clean_ancillas=1, num_dirty_ancillas=1
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="noaux_hp24", num_clean_ancillas=1, num_dirty_ancillas=1):
            # should have a decomposition
            decomposition = MCXSynthesisNoAuxHP24().run(
                gate, num_clean_ancillas=1, num_dirty_ancillas=1
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="noaux_v24", num_clean_ancillas=0, num_dirty_ancillas=0):
            # should have a decomposition
            decomposition = MCXSynthesisNoAuxV24().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=0
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="noaux_hp24", num_clean_ancillas=0, num_dirty_ancillas=0):
            # should have a decomposition
            decomposition = MCXSynthesisNoAuxHP24().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=0
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="gray_code", num_clean_ancillas=1, num_dirty_ancillas=1):
            # should have a decomposition
            decomposition = MCXSynthesisGrayCode().run(
                gate, num_clean_ancillas=1, num_dirty_ancillas=1
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="gray_code", num_clean_ancillas=0, num_dirty_ancillas=0):
            # should have a decomposition
            decomposition = MCXSynthesisGrayCode().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=0
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="default", num_clean_ancillas=1, num_dirty_ancillas=1):
            # should have a decomposition
            decomposition = MCXSynthesisDefault().run(
                gate, num_clean_ancillas=1, num_dirty_ancillas=1
            )
            self.assertIsNotNone(decomposition)

        with self.subTest(method="default", num_clean_ancillas=0, num_dirty_ancillas=0):
            # should have a decomposition
            decomposition = MCXSynthesisDefault().run(
                gate, num_clean_ancillas=0, num_dirty_ancillas=0
            )
            self.assertIsNotNone(decomposition)

    @data(
        "n_clean_m15",
        "n_dirty_i15",
        "2_clean_kg24",
        "2_dirty_kg24",
        "1_clean_kg24",
        "1_dirty_kg24",
        "1_clean_b95",
        "noaux_v24",
        "gray_code",
        "default",
    )
    def test_mcx_plugins_correctness_from_arbitrary(self, mcx_plugin_name):
        """Test that all plugins return a correct Operator when qubits are not
        initially zero."""
        qc = QuantumCircuit(6)
        qc.h(0)
        qc.cx(0, 1)
        qc.mcx(control_qubits=[0, 1, 2], target_qubit=[3])
        qc.mcx(control_qubits=[2, 3, 4, 5], target_qubit=[1])
        qc.mcx(control_qubits=[5, 4, 3, 2, 1], target_qubit=[0])
        hls_config = HLSConfig(mcx=[mcx_plugin_name])
        hls_pass = HighLevelSynthesis(hls_config=hls_config, qubits_initially_zero=False)
        qct = hls_pass(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    @data(
        "n_clean_m15",
        "n_dirty_i15",
        "2_clean_kg24",
        "2_dirty_kg24",
        "1_clean_kg24",
        "1_dirty_kg24",
        "1_clean_b95",
        "noaux_v24",
        "gray_code",
        "default",
    )
    def test_mcx_plugins_correctness_from_zero(self, mcx_plugin_name):
        """Test that all plugins return a correct Statevector when qubits are
        initially zero."""
        qc = QuantumCircuit(6)
        qc.h(0)
        qc.cx(0, 1)
        qc.mcx(control_qubits=[0, 1, 2], target_qubit=[3])
        qc.mcx(control_qubits=[2, 3, 4, 5], target_qubit=[1])
        qc.mcx(control_qubits=[5, 4, 3, 2, 1], target_qubit=[0])
        hls_config = HLSConfig(mcx=[mcx_plugin_name])
        hls_pass = HighLevelSynthesis(hls_config=hls_config, qubits_initially_zero=True)
        qct = hls_pass(qc)
        self.assertEqual(Statevector(qc), Statevector(qct))

    def test_annotated_mcx(self):
        """Test synthesis of annotated MCX gates."""
        qc = QuantumCircuit(6)
        qc.h(0)
        qc.append(MCXGate(3).inverse(annotated=True).control(2, annotated=True), [0, 1, 2, 3, 4, 5])
        qct = transpile(qc, qubits_initially_zero=False)
        self.assertEqual(Operator(qc), Operator(qct))


@ddt
class TestPauliEvolutionSynthesisPlugins(QiskitTestCase):
    """Tests related to plugins for PauliEvolutionGate."""

    def test_supported_names(self):
        """Test that "default" and "rustiq" plugins do exist."""
        supported_plugin_names = high_level_synthesis_plugin_names("PauliEvolution")
        self.assertIn("default", supported_plugin_names)
        self.assertIn("rustiq", supported_plugin_names)

    @data("default", "rustiq")
    def test_correctness(self, plugin_name):
        """Test that plugins return the correct Operator."""
        op = SparsePauliOp(["XXX", "YYY", "IZZ", "XZY"], [1, 2, 3, 4])
        evo = PauliEvolutionGate(op, synthesis=LieTrotter())

        # compile via HLS
        qc = QuantumCircuit(6)
        qc.append(evo, [1, 2, 4])
        hls_config = HLSConfig(PauliEvolution=[plugin_name])
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        qct = hls_pass(qc)

        # compute reference
        ref = QuantumCircuit(6)
        ref.compose(evo.definition, [1, 2, 4], inplace=True)

        self.assertEqual(count_rotation_gates(qct), 4)
        self.assertEqual(Operator(ref), Operator(qct))

    @data("default", "rustiq")
    def test_trivial_rotations(self, plugin_name):
        """Test that plugins return the correct Operator in the presence of
        trivial (all-I) rotations.
        """
        op = SparsePauliOp(["III", "XZY", "III", "III"], [1, 2, 3, 4])
        qc = QuantumCircuit(6)
        qc.append(PauliEvolutionGate(op), [1, 2, 4])
        hls_config = HLSConfig(PauliEvolution=[plugin_name])
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        qct = hls_pass(qc)
        self.assertEqual(Operator(qc), Operator(qct))
        self.assertEqual(count_rotation_gates(qct), 1)

    def test_default_preserve_order(self):
        """Test that option preserve_order is reset."""
        op = SparsePauliOp(["IIIX", "IIXX", "IYYI", "IIZZ"], coeffs=[1, 2, 3, 4])
        qc = QuantumCircuit(6)
        qc.append(PauliEvolutionGate(op), [1, 2, 3, 4])
        with self.subTest("preserve_order_is_reset"):
            hls_config = HLSConfig(PauliEvolution=[("default", {"preserve_order": False})])
            hls_pass = HighLevelSynthesis(hls_config=hls_config)
            qct = hls_pass(qc)
            self.assertEqual(qct.depth(), 3)
            # check that preserve_order is reset and is no longer False
            hls_config = HLSConfig(PauliEvolution=[("default", {})])
            hls_pass = HighLevelSynthesis(hls_config=hls_config)
            qct = hls_pass(qc)
            self.assertEqual(qct.depth(), 4)

    def test_rustiq_upto_options(self):
        """Test non-default Rustiq options upto_phase and upto_clifford."""
        op = SparsePauliOp(["XXXX", "YYYY", "ZZZZ"], coeffs=[1, 2, 3])
        qc = QuantumCircuit(6)
        qc.append(PauliEvolutionGate(op), [1, 2, 3, 4])

        # These calls to Rustiq are deterministic.
        # On the one hand, we may need to change these tests if we switch
        # to a newer version of Rustiq that implements different heuristics.
        # On the other hand, these tests serve to show that the options
        # have the desired effect of reducing the number of CX-gates.
        with self.subTest("default_options"):
            hls_config = HLSConfig(PauliEvolution=[("rustiq", {"upto_phase": False})])
            hls_pass = HighLevelSynthesis(hls_config=hls_config)
            qct = hls_pass(qc)
            cnt_ops = qct.count_ops()
            self.assertEqual(count_rotation_gates(qct), 3)
            self.assertEqual(cnt_ops["cx"], 10)
        with self.subTest("upto_phase"):
            hls_config = HLSConfig(PauliEvolution=[("rustiq", {"upto_phase": True})])
            hls_pass = HighLevelSynthesis(hls_config=hls_config)
            qct = hls_pass(qc)
            cnt_ops = qct.count_ops()
            self.assertEqual(count_rotation_gates(qct), 3)
            self.assertEqual(cnt_ops["cx"], 9)
        with self.subTest("upto_clifford"):
            hls_config = HLSConfig(PauliEvolution=[("rustiq", {"upto_clifford": True})])
            hls_pass = HighLevelSynthesis(hls_config=hls_config)
            qct = hls_pass(qc)
            cnt_ops = qct.count_ops()
            self.assertEqual(count_rotation_gates(qct), 3)
            self.assertEqual(cnt_ops["cx"], 5)

    def test_rustiq_preserve_order(self):
        """Test non-default Rustiq option preserve_order."""
        op = SparsePauliOp(["IXX", "YYI", "IXX", "YYI", "IXX", "YYI"])
        qc = QuantumCircuit(3)
        qc.append(PauliEvolutionGate(op), [0, 1, 2])
        with self.subTest("preserve_order_is_true"):
            hls_config = HLSConfig(PauliEvolution=[("rustiq", {"preserve_order": True})])
            hls_pass = HighLevelSynthesis(hls_config=hls_config)
            qct = hls_pass(qc)
            cnt_ops = qct.count_ops()
            self.assertEqual(count_rotation_gates(qct), 6)
            self.assertEqual(cnt_ops["cx"], 16)
        with self.subTest("preserve_order_is_false"):
            hls_config = HLSConfig(PauliEvolution=[("rustiq", {"preserve_order": False})])
            hls_pass = HighLevelSynthesis(hls_config=hls_config)
            qct = hls_pass(qc)
            cnt_ops = qct.count_ops()
            self.assertEqual(count_rotation_gates(qct), 6)
            self.assertEqual(cnt_ops["cx"], 4)
        with self.subTest("preserve_order_is_reset"):
            hls_config = HLSConfig(PauliEvolution=[("rustiq", {"preserve_order": False})])
            hls_pass = HighLevelSynthesis(hls_config=hls_config)
            qct = hls_pass(qc)
            cnt_ops = qct.count_ops()
            self.assertEqual(count_rotation_gates(qct), 6)
            self.assertEqual(cnt_ops["cx"], 4)
            # check that preserve_order is reset and is no longer False
            hls_config = HLSConfig(PauliEvolution=[("rustiq", {})])
            hls_pass = HighLevelSynthesis(hls_config=hls_config)
            qct = hls_pass(qc)
            cnt_ops = qct.count_ops()
            self.assertEqual(count_rotation_gates(qct), 6)
            self.assertEqual(cnt_ops["cx"], 16)

    def test_rustiq_upto_phase(self):
        """Check that Rustiq synthesis with ``upto_phase=True`` produces a correct
        circuit up to the global phase.
        """
        # On this example Rustiq with the option "upto_phase=True" does produce a circuit
        # with a different global phase.
        op = SparsePauliOp(
            [
                "IIII",
                "XXII",
                "XIXI",
                "XIIX",
                "YYII",
                "YIYI",
                "YIIY",
                "ZZII",
                "ZIZI",
                "ZIIZ",
                "IXIX",
                "IYIY",
                "IZIZ",
            ]
        )
        qc = QuantumCircuit(4)
        qc.append(PauliEvolutionGate(op), [0, 1, 2, 3])
        default_config = HLSConfig(PauliEvolution=["default"])
        qct_default = HighLevelSynthesis(hls_config=default_config)(qc)
        rustiq_config = HLSConfig(PauliEvolution=[("rustiq", {"upto_phase": True})])
        qct_rustiq = HighLevelSynthesis(hls_config=rustiq_config)(qc)
        self.assertEqual(count_rotation_gates(qct_default), 12)
        self.assertEqual(count_rotation_gates(qct_rustiq), 12)
        self.assertTrue(Operator(qct_default).equiv(Operator(qct_rustiq)))

    def test_rustiq_with_parameterized_angles(self):
        """Test Rustiq's synthesis with parameterized angles."""
        alpha = Parameter("alpha")
        beta = Parameter("beta")
        pauli_network = [("XXX", [0, 1, 2], alpha), ("Y", [1], beta)]
        qct = synth_pauli_network_rustiq(
            num_qubits=4, pauli_network=pauli_network, upto_clifford=True
        )
        self.assertEqual(count_rotation_gates(qct), 2)
        self.assertEqual(set(qct.parameters), {alpha, beta})

    def test_rustiq_raises_on_invalid_input(self):
        """Test that we get an error on invalid input."""
        # The Pauli string "1Y" is invalid
        pauli_network = [("XXX", [0, 1, 2], 0.5), ("1Y", [1, 2], -0.5)]
        with self.assertRaises(QiskitError):
            synth_pauli_network_rustiq(num_qubits=4, pauli_network=pauli_network)

    def test_on_sparse_observable(self):
        """Test that plugins handle operators with SparseObservables."""
        obs = SparseObservable.from_sparse_list([("1+XY", (0, 1, 2, 3), 1.5)], num_qubits=4)
        evo = PauliEvolutionGate(obs, time=1)
        qc = QuantumCircuit(4)
        qc.append(evo, [0, 1, 2, 3])
        default_config = HLSConfig(PauliEvolution=["default"])
        qct_default = HighLevelSynthesis(hls_config=default_config)(qc)
        rustiq_config = HLSConfig(PauliEvolution=["rustiq"])
        qct_rustiq = HighLevelSynthesis(hls_config=rustiq_config)(qc)
        self.assertEqual(Operator(qct_default), Operator(qc))
        self.assertEqual(Operator(qct_rustiq), Operator(qc))

    def test_on_list_with_sparse_observable(self):
        """Test that plugins handle operators with SparseObservables."""
        pauli = Pauli("-XYZI")
        op = SparsePauliOp(["IZXY"], 1)
        obs = SparseObservable.from_sparse_list([("1+XY", (0, 1, 2, 3), 1.5)], num_qubits=4)
        evo = PauliEvolutionGate([pauli, op, obs], time=1)
        qc = QuantumCircuit(4)
        qc.append(evo, [0, 1, 2, 3])
        default_config = HLSConfig(PauliEvolution=["default"])
        qct_default = HighLevelSynthesis(hls_config=default_config)(qc)
        rustiq_config = HLSConfig(PauliEvolution=["rustiq"])
        qct_rustiq = HighLevelSynthesis(hls_config=rustiq_config)(qc)
        self.assertEqual(Operator(qct_default), Operator(qc))
        self.assertEqual(Operator(qct_rustiq), Operator(qc))


class TestAnnotatedSynthesisPlugins(QiskitTestCase):
    """Tests related to plugins for AnnotatedOperation."""

    def setUp(self):
        super().setUp()
        self._pass = HighLevelSynthesis(basis_gates=["cx", "u"])

    def test_conjugate_reduction_applies_1(self):
        """Test that conjugate reduction optimization applies when the first and the last gates
        are inverse of each other for the given choice of parameters."""
        qc_inner = QuantumCircuit(1)
        qc_inner.append(PhaseGate(1), [0])
        qc_inner.append(HGate(), [0])
        qc_inner.append(PhaseGate(-1), [0])

        qc_main = QuantumCircuit(5)
        qc_main.append(qc_inner.to_gate().control(4, annotated=True), [0, 1, 2, 3, 4])

        # Optimized circuit with non-controlled phase gates
        qc_expected = QuantumCircuit(5)
        qc_expected.append(PhaseGate(1), [4])
        qc_expected.append(HGate().control(4, annotated=False), [0, 1, 2, 3, 4])
        qc_expected.append(PhaseGate(-1), [4])

        qc_main_tranpiled = self._pass(qc_main)
        qc_expected_transpiled = self._pass(qc_expected)

        self.assertEqual(Operator(qc_main_tranpiled), Operator(qc_expected_transpiled))
        self.assertEqual(qc_main_tranpiled.count_ops(), qc_expected_transpiled.count_ops())

    def test_conjugate_reduction_not_applies_1(self):
        """Test that conjugate reduction optimization does not apply when the first and the
        last gates are not inverse of each other for the given choice of parameters."""
        qc_inner = QuantumCircuit(1)
        qc_inner.append(PhaseGate(1), [0])
        qc_inner.append(HGate(), [0])
        qc_inner.append(PhaseGate(-2), [0])

        qc_main = QuantumCircuit(5)
        qc_main.append(qc_inner.to_gate().control(4, annotated=True), [0, 1, 2, 3, 4])

        # Non-optimized circuit with controlled phase gates
        qc_expected = QuantumCircuit(5)
        qc_expected.append(PhaseGate(1).control(4, annotated=False), [0, 1, 2, 3, 4])
        qc_expected.append(HGate().control(4, annotated=False), [0, 1, 2, 3, 4])
        qc_expected.append(PhaseGate(-2).control(4, annotated=False), [0, 1, 2, 3, 4])

        qc_main_tranpiled = self._pass(qc_main)
        qc_expected_transpiled = self._pass(qc_expected)

        self.assertEqual(Operator(qc_main_tranpiled), Operator(qc_expected_transpiled))
        self.assertEqual(qc_main_tranpiled.count_ops(), qc_expected_transpiled.count_ops())

    def test_conjugate_reduction_applies_2(self):
        """Test that conjugate reduction optimization applies when the first and the last gates
        are inverse of each other for the given choice of parameters, with the inverse represented
        via a modifier."""
        qc_inner = QuantumCircuit(1)
        qc_inner.append(PhaseGate(1), [0])
        qc_inner.append(HGate(), [0])
        qc_inner.append(PhaseGate(1).inverse(annotated=True), [0])

        qc_main = QuantumCircuit(5)
        qc_main.append(qc_inner.to_gate().control(4, annotated=True), [0, 1, 2, 3, 4])

        # Optimized circuit with non-controlled phase gates
        qc_expected = QuantumCircuit(5)
        qc_expected.append(PhaseGate(1), [4])
        qc_expected.append(HGate().control(4, annotated=False), [0, 1, 2, 3, 4])
        qc_expected.append(PhaseGate(-1), [4])

        qc_main_tranpiled = self._pass(qc_main)
        qc_expected_transpiled = self._pass(qc_expected)
        self.assertEqual(Operator(qc_main_tranpiled), Operator(qc_expected_transpiled))
        self.assertEqual(qc_main_tranpiled.count_ops(), qc_expected_transpiled.count_ops())

    def test_conjugate_reduction_not_applies_2(self):
        """Test that conjugate reduction optimization does not apply when the first and the
        last gates are not inverse of each other for the given choice of parameters.
        """
        qc_inner = QuantumCircuit(1)
        qc_inner.append(PhaseGate(1), [0])
        qc_inner.append(HGate(), [0])
        qc_inner.append(PhaseGate(2).inverse(annotated=True), [0])

        qc_main = QuantumCircuit(5)
        qc_main.append(qc_inner.to_gate().control(4, annotated=True), [0, 1, 2, 3, 4])

        # Non-optimized circuit with controlled phase gates
        qc_expected = QuantumCircuit(5)
        qc_expected.append(PhaseGate(1).control(4, annotated=False), [0, 1, 2, 3, 4])
        qc_expected.append(HGate().control(4, annotated=False), [0, 1, 2, 3, 4])
        qc_expected.append(PhaseGate(-2).control(4, annotated=False), [0, 1, 2, 3, 4])

        qc_main_tranpiled = self._pass(qc_main)
        qc_expected_transpiled = self._pass(qc_expected)

        self.assertEqual(Operator(qc_main_tranpiled), Operator(qc_expected_transpiled))
        self.assertEqual(qc_main_tranpiled.count_ops(), qc_expected_transpiled.count_ops())


def count_rotation_gates(qc: QuantumCircuit):
    """Return the number of rotation gates in a quantum circuit."""
    ops = qc.count_ops()
    return (
        ops.get("rx", 0)
        + ops.get("ry", 0)
        + ops.get("rz", 0)
        + ops.get("rxx", 0)
        + ops.get("ryy", 0)
        + ops.get("rzz", 0)
    )


if __name__ == "__main__":
    unittest.main()
