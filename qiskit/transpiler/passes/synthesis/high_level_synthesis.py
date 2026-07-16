# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
High-level-synthesis transpiler pass.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from qiskit.circuit.operation import Operation
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import EquivalenceLibrary
from qiskit.transpiler.optimization_metric import OptimizationMetric
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError

from qiskit._accelerate.high_level_synthesis import (
    QubitTracker,
    HighLevelSynthesisData,
    run_on_dag,
)

from .plugin import HighLevelSynthesisPluginManager


class HLSConfig:
    """The high-level-synthesis config allows to specify a list of "methods" used by
    :class:`~.HighLevelSynthesis` transformation pass to synthesize different types
    of higher-level objects.

    A higher-level object is an object of type :class:`~.Operation` (e.g., :class:`.Clifford` or
    :class:`.LinearFunction`).  Each object is referred to by its :attr:`~.Operation.name` field
    (e.g., ``"clifford"`` for :class:`.Clifford` objects), and the applicable synthesis methods are
    tied to this name.

    In the config, each method is specified in one of several ways:

    1. a tuple consisting of the name of a known synthesis plugin and a dictionary providing
       additional arguments for the algorithm.
    2. a tuple consisting of an instance of :class:`.HighLevelSynthesisPlugin` and additional
       arguments for the algorithm.
    3. a single string of a known synthesis plugin
    4. a single instance of :class:`.HighLevelSynthesisPlugin`.

    The following example illustrates different ways how a config file can be created::

        from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
        from qiskit.transpiler.passes.synthesis.high_level_synthesis import ACGSynthesisPermutation

        # All the ways to specify hls_config are equivalent
        hls_config = HLSConfig(permutation=[("acg", {})])
        hls_config = HLSConfig(permutation=["acg"])
        hls_config = HLSConfig(permutation=[(ACGSynthesisPermutation(), {})])
        hls_config = HLSConfig(permutation=[ACGSynthesisPermutation()])

    The names of the synthesis plugins should be declared in ``entry-points`` table for
    ``qiskit.synthesis`` in ``pyproject.toml``, in the form
    <higher-level-object-name>.<synthesis-method-name>.

    The standard higher-level-objects are recommended to have a synthesis method
    called "default", which would be called automatically when synthesizing these objects,
    without having to explicitly set these methods in the config.

    To avoid synthesizing a given higher-level-object, one can give it an empty list of methods.

    For an explicit example of using such config files, refer to the documentation for
    :class:`~.HighLevelSynthesis`.

    For an overview of the complete process of using high-level synthesis, see
    :ref:`using-high-level-synthesis-plugins`.
    """

    def __init__(
        self,
        use_default_on_unspecified: bool = True,
        plugin_selection: str = "sequential",
        plugin_evaluation_fn: Callable[[QuantumCircuit], int] | None = None,
        **kwargs,
    ):
        """Creates a high-level-synthesis config.

        Args:
            use_default_on_unspecified: if True, every higher-level-object without an
                explicitly specified list of methods will be synthesized using the "default"
                algorithm if it exists.
            plugin_selection: if set to ``"sequential"`` (default), for every higher-level-object
                the synthesis pass will consider the specified methods sequentially, stopping
                at the first method that is able to synthesize the object. If set to ``"all"``,
                all the specified methods will be considered, and the best synthesized circuit,
                according to ``plugin_evaluation_fn`` will be chosen.
            plugin_evaluation_fn: a callable that evaluates the quality of the synthesized
                quantum circuit in the case that ``plugin_selection="sequential"``;
                a smaller value means a better circuit. If ``None``, the
                quality of the circuit is its size (i.e. the number of gates that it contains).
            kwargs: a dictionary mapping higher-level-objects to lists of synthesis methods.
        """
        self.use_default_on_unspecified = use_default_on_unspecified
        self.plugin_selection = plugin_selection
        self.plugin_evaluation_fn = (
            plugin_evaluation_fn if plugin_evaluation_fn is not None else lambda qc: qc.size()
        )
        self.methods = {}

        for key, value in kwargs.items():
            self.set_methods(key, value)

    def set_methods(self, hls_name, hls_methods):
        """Sets the list of synthesis methods for a given higher-level-object. This overwrites
        the lists of methods if also set previously."""
        self.methods[hls_name] = hls_methods


class HighLevelSynthesis(TransformationPass):
    r"""Synthesize higher-level objects and unroll custom definitions.

    The input to this pass is a DAG that may contain higher-level objects,
    including abstract mathematical objects (e.g., objects of type :class:`.LinearFunction`),
    annotated operations (objects of type :class:`.AnnotatedOperation`), and
    custom gates.

    In the most common use-case when either ``basis_gates`` or ``target`` is specified,
    all higher-level objects are synthesized, so the output is a :class:`.DAGCircuit`
    without such objects.
    More precisely, every gate in the output DAG is either directly supported by the target,
    or is in ``equivalence_library``.

    The abstract mathematical objects are synthesized using synthesis plugins, applying
    synthesis methods specified in the high-level-synthesis config (refer to the documentation
    for :class:`~.HLSConfig`).

    As an example, let us assume that ``op_a`` and ``op_b`` are names of two higher-level objects,
    that ``op_a``-objects have two synthesis methods ``default`` which does require any additional
    parameters and ``other`` with two optional integer parameters ``option_1`` and ``option_2``,
    that ``op_b``-objects have a single synthesis method ``default``, and ``qc`` is a quantum
    circuit containing ``op_a`` and ``op_b`` objects. The following code snippet::

        hls_config = HLSConfig(op_b=[("other", {"option_1": 7, "option_2": 4})])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        transpiled_qc = pm.run(qc)

    shows how to run the alternative synthesis method ``other`` for ``op_b``-objects, while using the
    ``default`` methods for all other high-level objects, including ``op_a``-objects.

    The annotated operations (consisting of a base operation and a list of inverse, control and power
    modifiers) are synthesizing recursively, first synthesizing the base operation, and then applying
    synthesis methods for creating inverted, controlled, or powered versions of that).

    The custom gates are synthesized by recursively unrolling their definitions, until every gate
    is either supported by the target or is in the equivalence library.

    When neither ``basis_gates`` nor ``target`` is specified, the pass synthesizes only the top-level
    abstract mathematical objects and annotated operations, without descending into the gate
    ``definitions``. This is consistent with the older behavior of the pass, allowing to synthesize
    some higher-level objects using plugins and leaving the other gates untouched.

    The high-level-synthesis passes information about available auxiliary qubits, and whether their
    state is clean (defined as :math:`|0\rangle`) or dirty (unknown state) to the synthesis routine
    via the respective arguments ``"num_clean_ancillas"`` and ``"num_dirty_ancillas"``.
    If ``qubits_initially_zero`` is ``True`` (default), the qubits are assumed to be in the
    :math:`|0\rangle` state. When appending a synthesized block using auxiliary qubits onto the
    circuit, we first use the clean auxiliary qubits.

    .. note::

        Synthesis methods are assumed to maintain the state of the auxiliary qubits.
        Concretely this means that clean auxiliary qubits must still be in the :math:`|0\rangle`
        state after the synthesized block, while dirty auxiliary qubits are re-used only
        as dirty qubits.

    """

    def __init__(
        self,
        hls_config: HLSConfig | None = None,
        coupling_map: CouplingMap | None = None,
        target: Target | None = None,
        use_qubit_indices: bool = False,
        equivalence_library: EquivalenceLibrary | None = None,
        basis_gates: list[str] | None = None,
        min_qubits: int = 0,
        qubits_initially_zero: bool = True,
        optimization_metric: OptimizationMetric = OptimizationMetric.COUNT_2Q,
    ):
        r"""
        HighLevelSynthesis initializer.

        Args:
            hls_config:  the high-level-synthesis config that specifies synthesis methods
                and parameters for various high-level-objects in the circuit. If it is not specified,
                the default synthesis methods and parameters will be used.
            coupling_map:  directed graph represented as a coupling map.
            target:  the backend target to use for this pass. If it is specified,
                it will be used instead of the coupling map.
            use_qubit_indices: a flag indicating whether this synthesis pass is running before or after
                the layout is set, that is, whether the qubit indices of higher-level-objects correspond
                to qubit indices on the target backend.
            equivalence_library: The equivalence library used (instructions in this library will not
                be unrolled by this pass).
            basis_gates:  target basis names to unroll to, e.g. `['u3', 'cx']`.
                Ignored if ``target`` is also specified.
            min_qubits: The minimum number of qubits for operations in the input
                dag to translate.
            qubits_initially_zero: Indicates whether the qubits are initially in the state
                :math:`|0\rangle`. This allows the high-level-synthesis to use clean auxiliary qubits
                (i.e. in the zero state) to synthesize an operation.
            optimization_metric:  Specifies the optimization criterion used by the default synthesis
                methods for high-level-objects (when available).
        """
        super().__init__()

        # When the config file is not provided, we will use the "default" method
        # to synthesize Operations (when available).
        hls_config = hls_config or HLSConfig(True)
        hls_plugin_manager = HighLevelSynthesisPluginManager()
        hls_op_names = set(hls_plugin_manager.plugins_by_op.keys()).union(
            set(hls_config.methods.keys())
        )

        if target is not None:
            coupling_map = target.build_coupling_map()

        unroll_definitions = not (
            (basis_gates is None or len(basis_gates) == 0)
            and (target is None or len(target.operation_names) == 0)
        )

        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        if unroll_definitions and (target is None or target.num_qubits is None):
            basic_insts = {"measure", "reset", "barrier", "snapshot", "delay", "store"}
            device_insts = basic_insts | set(basis_gates)
        else:
            device_insts = set()

        self.qubits_initially_zero = qubits_initially_zero

        self.data = HighLevelSynthesisData(
            hls_config=hls_config,
            hls_plugin_manager=hls_plugin_manager,
            coupling_map=coupling_map,
            target=target,
            equivalence_library=equivalence_library,
            hls_op_names=hls_op_names,
            device_insts=device_insts,
            use_physical_indices=use_qubit_indices,
            min_qubits=min_qubits,
            unroll_definitions=unroll_definitions,
            optimize_clifford_t=optimization_metric == OptimizationMetric.COUNT_T,
        )

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the HighLevelSynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with higher-level operations synthesized.

        Raises:
            TranspilerError: when the transpiler is unable to synthesize the given DAG
            (for instance, when the specified synthesis method is not available).
        """
        res = run_on_dag(dag, self.data, self.qubits_initially_zero)
        return res if res is not None else dag


def _methods_to_try(data: HighLevelSynthesisData, name: str):
    """Get a sequence of methods to try for a given op name."""
    if (methods := data.hls_config.methods.get(name)) is not None:
        # the operation's name appears in the user-provided config,
        # we use the list of methods provided by the user
        return methods
    if (
        data.hls_config.use_default_on_unspecified
        and "default" in data.hls_plugin_manager.method_names(name)
    ):
        # the operation's name does not appear in the user-specified config,
        # we use the "default" method when instructed to do so and the "default"
        # method is available
        return ["default"]
    return []


def _synthesize_op_using_plugins(
    operation: Operation,
    input_qubits: tuple[int],
    data: HighLevelSynthesisData,
    tracker: QubitTracker,
) -> tuple[QuantumCircuit, tuple[int], QubitTracker] | None:
    """
    Attempts to synthesize an operation using plugin mechanism.

    Input:
        operation: the operation to be synthesized.
        input_qubits: a list of global qubits (qubits in the original circuit) over
            which the operation is defined.
        data: high-level-synthesis data and options.
        tracker: the global tracker, tracking the state of global qubits.
        hls_methods: the list of synthesis methods to try.

    The function is called from within Rust code.

    Returns either the synthesized circuit or ``None`` (which may occur
    when no synthesis methods is available or specified, or when there is
    an insufficient number of auxiliary qubits).
    """
    hls_methods = _methods_to_try(data, operation.name)
    if len(hls_methods) == 0:
        return None

    hls_plugin_manager = data.hls_plugin_manager
    num_clean_ancillas = tracker.num_clean(input_qubits)
    num_dirty_ancillas = tracker.num_dirty(input_qubits)

    best_decomposition = None
    best_score = np.inf

    for method in hls_methods:
        # There are two ways to specify a synthesis method. The more explicit
        # way is to specify it as a tuple consisting of a synthesis algorithm and a
        # list of additional arguments, e.g.,
        #   ("kms", {"all_mats": 1, "max_paths": 100, "orig_circuit": 0}), or
        #   ("pmh", {}).
        # When the list of additional arguments is empty, one can also specify
        # just the synthesis algorithm, e.g.,
        #   "pmh".
        if isinstance(method, tuple):
            plugin_specifier, plugin_args = method
        else:
            plugin_specifier = method
            plugin_args = {}

        # There are two ways to specify a synthesis algorithm being run,
        # either by name, e.g. "kms" (which then should be specified in entry_points),
        # or directly as a class inherited from HighLevelSynthesisPlugin (which then
        # does not need to be specified in entry_points).
        if isinstance(plugin_specifier, str):
            if plugin_specifier not in hls_plugin_manager.method_names(operation.name):
                raise TranspilerError(
                    f"Specified method: {plugin_specifier} not found in available "
                    f"plugins for {operation.name}"
                )
            plugin_method = hls_plugin_manager.method(operation.name, plugin_specifier)
        else:
            plugin_method = plugin_specifier

        # The additional arguments we pass to every plugin include the list of global
        # qubits over which the operation is defined, high-level-synthesis data and options,
        # and the tracker that tracks the state for global qubits.
        #
        # Note: the difference between the argument "qubits" passed explicitly to "run"
        # and "input_qubits" passed via "plugin_args" is that for backwards compatibility
        # the former should be None if the synthesis is done before layout/routing.
        # However, plugins may need access to the global qubits over which the operation
        # is defined, as well as their state, in particular the plugin for AnnotatedOperations
        # requires these arguments to be able to process the base operation recursively.
        #
        # We may want to refactor the inputs and the outputs for the plugins' "run" method,
        # however this needs to be backwards-compatible.
        plugin_args["input_qubits"] = input_qubits
        plugin_args["hls_data"] = data
        plugin_args["qubit_tracker"] = tracker
        plugin_args["num_clean_ancillas"] = num_clean_ancillas
        plugin_args["num_dirty_ancillas"] = num_dirty_ancillas
        if data.optimize_clifford_t:
            plugin_args["optimization_metric"] = OptimizationMetric.COUNT_T
        else:
            plugin_args["optimization_metric"] = OptimizationMetric.COUNT_2Q

        qubits = input_qubits if data.use_physical_indices else None

        decomposition = plugin_method.run(
            operation,
            coupling_map=data.coupling_map,
            target=data.target,
            qubits=qubits,
            **plugin_args,
        )

        # The synthesis methods that are not suited for the given higher-level-object
        # will return None.
        if decomposition is not None:
            if data.hls_config.plugin_selection == "sequential":
                # In the "sequential" mode the first successful decomposition is
                # returned.
                best_decomposition = decomposition
                break

            # In the "run everything" mode we update the best decomposition
            # discovered
            current_score = data.hls_config.plugin_evaluation_fn(decomposition)
            if current_score < best_score:
                best_decomposition = decomposition
                best_score = current_score

    # A synthesis method may have potentially used available ancilla qubits.
    # The following greedily grabs global qubits available. In the additional
    # refactoring mentioned previously, we want each plugin to actually return
    # the global qubits used, especially when the synthesis is done on the physical
    # circuit, and the choice of which ancilla qubits to use really matters.
    output_qubits = input_qubits
    if best_decomposition is not None:
        if best_decomposition.num_qubits > len(input_qubits):
            global_aux_qubits = tracker.borrow(
                best_decomposition.num_qubits - len(input_qubits), input_qubits
            )
            output_qubits = output_qubits + global_aux_qubits

        # This checks (in particular) that there is indeed a sufficient number
        # of ancilla qubits to borrow from the tracker.
        if best_decomposition.num_qubits != len(output_qubits):
            raise TranspilerError(
                "HighLevelSynthesis: the result from 'synthesize_op_using_plugin' is incorrect."
            )

    if best_decomposition is None:
        return None

    return (best_decomposition, output_qubits)
