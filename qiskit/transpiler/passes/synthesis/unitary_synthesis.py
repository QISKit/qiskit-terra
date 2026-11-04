# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Unitary Synthesis Transpiler Pass
"""

from __future__ import annotations
from typing import Any

from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.circuit import CircuitInstruction
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.synthesis.one_qubit import one_qubit_decompose

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.synthesis import plugin
from qiskit.transpiler.target import Target

from qiskit._accelerate.unitary_synthesis import run_main_loop


def _choose_bases(basis_gates, basis_dict=None):
    """Find the matching basis string keys from the list of basis gates from the target."""
    if basis_gates is None:
        basis_set = set()
    else:
        basis_set = set(basis_gates)

    if basis_dict is None:
        basis_dict = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES

    out_basis = []
    for basis, gates in basis_dict.items():
        if set(gates).issubset(basis_set):
            out_basis.append(basis)

    return out_basis


class UnitarySynthesis(TransformationPass):
    """Synthesize gates according to their basis gates."""

    def __init__(
        self,
        basis_gates: list[str] | None = None,
        approximation_degree: float | None = 1.0,
        coupling_map: CouplingMap | None = None,
        pulse_optimize: bool | None = None,
        natural_direction: bool | None = None,
        synth_gates: list[str] | None = None,
        method: str = "default",
        min_qubits: int = 0,
        plugin_config: dict | None = None,
        target: Target | None = None,
        fallback_on_default: bool = False,
    ):
        """Synthesize unitaries over some basis gates.

        This pass can approximate unitaries given some
        gate fidelities (via ``target``).
        More approximation can be forced by setting a heuristic dial
        ``approximation_degree``.

        This pass is multithreaded when the default synthesis plugin is used
        and will potentially launch a thread pool with threads equal to the
        number of CPUs by default. You can tune the number of threads with
        the ``RAYON_NUM_THREADS`` environment variable. For example, setting
        ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.

        Args:
            basis_gates: List of gate names to target. If this is
                not specified the ``target`` argument must be used. If both this
                and the ``target`` are specified the value of ``target`` will
                be used and this will be ignored.
            approximation_degree: Heuristic dial used for circuit approximation
                (1.0=no approximation, 0.0=maximal approximation). Approximation can
                make the synthesized circuit cheaper at the cost of straying from
                the original unitary. If ``None``, approximation is done based on gate fidelities.
            coupling_map: The coupling map of the target
                in case synthesis is done on a physical circuit. The
                directionality of the coupling_map will be taken into
                account if ``pulse_optimize`` is ``True``/``None`` and ``natural_direction``
                is ``True``/``None``.
            pulse_optimize: Whether to optimize pulses during
                synthesis. A value of ``None`` will attempt it but fall
                back if it does not succeed. A value of ``True`` will raise
                an error if pulse-optimized synthesis does not succeed.
            natural_direction: Whether to apply synthesis considering
                directionality of 2-qubit gates. Only applies when
                ``pulse_optimize`` is ``True`` or ``None``. The natural direction is
                determined by first checking to see whether the
                coupling map is unidirectional.  If there is no
                coupling map or the coupling map is bidirectional,
                the gate direction with the shorter
                duration from the target properties will be used. If
                set to True, and a natural direction can not be
                determined, raises :class:`.TranspilerError`. If set to None, no
                exception will be raised if a natural direction can
                not be determined.
            synth_gates: List of gates to synthesize. If None and
                ``pulse_optimize`` is False or None, default to
                ``['unitary']``. If ``None`` and ``pulse_optimize == True``,
                default to ``['unitary', 'swap']``
            method: The unitary synthesis method plugin to use.
            min_qubits: The minimum number of qubits in the unitary to synthesize. If this is set
                and the unitary is less than the specified number of qubits it will not be
                synthesized.
            plugin_config: Optional extra configuration arguments (as a ``dict``)
                which are passed directly to the specified unitary synthesis
                plugin. By default, this will have no effect as the default
                plugin has no extra arguments. Refer to the documentation of
                your unitary synthesis plugin on how to use this.
            target: The optional :class:`~.Target` for the target device the pass
                is compiling for. If specified this will supersede the values
                set for ``basis_gates`` and ``coupling_map``.
            fallback_on_default: Specifies whether the default synthesis method should be used
                in the case that a non-default synthesis ``method`` is specified but is either
                unable to synthesize the operation or the synthesized circuit does not conform
                to the target.

        Raises:
            TranspilerError: if ``method`` was specified but is not found in the
                installed plugins list. The list of installed plugins can be queried with
                :func:`~qiskit.transpiler.passes.synthesis.plugin.unitary_synthesis_plugin_names`
        """
        super().__init__()
        self._basis_gates = set(basis_gates or ())
        self._approximation_degree = approximation_degree
        self._min_qubits = min_qubits
        self.method = method
        self.plugins = None
        if method != "default":
            self.plugins = plugin.UnitarySynthesisPluginManager()
        self._coupling_map = coupling_map
        self._pulse_optimize = pulse_optimize
        self._natural_direction = natural_direction
        self._plugin_config = plugin_config
        self._fallback_on_default = fallback_on_default
        # Bypass target if it doesn't contain any basis gates (i.e it's _FakeTarget), as this
        # not part of the official target model.
        self._target = target if target is not None and len(target.operation_names) > 0 else None
        if target is not None:
            self._coupling_map = target.build_coupling_map()
            self._basis_gates = set(target.operation_names)
        if synth_gates:
            self._synth_gates = synth_gates
        elif pulse_optimize:
            self._synth_gates = ["unitary", "swap"]
        else:
            self._synth_gates = ["unitary"]

        self._synth_gates = set(self._synth_gates) - self._basis_gates

        if self.method != "default" and self.method not in self.plugins.ext_plugins:
            raise TranspilerError(f"Specified method '{self.method}' not found in plugin list")

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the UnitarySynthesis pass on ``dag``.

        Args:
            dag: input dag.

        Returns:
            Output dag with UnitaryGates synthesized to target basis.
        """

        # If there aren't any gates to synthesize in the circuit we can skip all the iteration
        # and just return.
        if not set(self._synth_gates).intersection(dag.count_ops()):
            return dag

        # Special flow for the "default" method
        if self.method == "default":
            _coupling_edges = (
                set(self._coupling_map.get_edges()) if self._coupling_map is not None else set()
            )
            qubit_indices = {bit: i for i, bit in enumerate(dag.qubits)}
            out = run_main_loop(
                dag,
                list(qubit_indices.values()),
                self._min_qubits,
                self._target,
                self._basis_gates,
                self._synth_gates,
                _coupling_edges,
                self._approximation_degree,
                self._natural_direction,
                self._pulse_optimize,
            )
            if out is None:
                return dag
            else:
                return out

        if self.plugins:
            plugin_method = self.plugins.ext_plugins[self.method].obj
        else:
            from qiskit.transpiler.passes.synthesis.default_unitary_synth_plugin import (
                DefaultUnitarySynthesis,
            )

            plugin_method = DefaultUnitarySynthesis()
        plugin_kwargs: dict[str, Any] = {"config": self._plugin_config}
        _gate_lengths = _gate_errors = None
        _gate_lengths_by_qubit = _gate_errors_by_qubit = None

        # If the method is not the default, we still need to initialise the default plugin's
        # keyword arguments in case we have to fall back on it during the actual run.
        default_method = self.plugins.ext_plugins["default"].obj
        default_kwargs = {}
        method_list = [(plugin_method, plugin_kwargs), (default_method, default_kwargs)]

        # Handle approximation degree as a special case for backwards compatibility, it's
        # not part of the plugin interface and only something needed for the default
        # pass.

        default_method._approximation_degree = self._approximation_degree

        qubit_indices = (
            {bit: i for i, bit in enumerate(dag.qubits)}
            if plugin_method.supports_coupling_map or default_method.supports_coupling_map
            else {}
        )

        for method, kwargs in method_list:
            if method.supports_basis_gates:
                kwargs["basis_gates"] = self._basis_gates
            if method.supports_natural_direction:
                kwargs["natural_direction"] = self._natural_direction
            if method.supports_pulse_optimize:
                kwargs["pulse_optimize"] = self._pulse_optimize
            if method.supports_gate_lengths:
                _gate_lengths = _gate_lengths or _build_gate_lengths(self._target)
                kwargs["gate_lengths"] = _gate_lengths
            if method.supports_gate_errors:
                _gate_errors = _gate_errors or _build_gate_errors(self._target)
                kwargs["gate_errors"] = _gate_errors
            if method.supports_gate_lengths_by_qubit:
                _gate_lengths_by_qubit = _gate_lengths_by_qubit or _build_gate_lengths_by_qubit(
                    self._target
                )
                kwargs["gate_lengths_by_qubit"] = _gate_lengths_by_qubit
            if method.supports_gate_errors_by_qubit:
                _gate_errors_by_qubit = _gate_errors_by_qubit or _build_gate_errors_by_qubit(
                    self._target
                )
                kwargs["gate_errors_by_qubit"] = _gate_errors_by_qubit
            supported_bases = method.supported_bases
            if supported_bases is not None:
                kwargs["matched_basis"] = _choose_bases(self._basis_gates, supported_bases)
            if method.supports_target:
                kwargs["target"] = self._target

        out = self._run_main_loop(
            dag, qubit_indices, plugin_method, plugin_kwargs, default_method, default_kwargs
        )
        return out

    def _run_plugin_synthesis(self, method, kwargs, unitary, qubit_indices, qargs):
        """Helper to run the synthesis plugin."""
        if method.supports_coupling_map:
            kwargs["coupling_map"] = (
                self._coupling_map,
                [qubit_indices[x] for x in qargs],
            )
        return method.run(unitary, **kwargs)

    def _run_main_loop(
        self, dag, qubit_indices, plugin_method, plugin_kwargs, default_method, default_kwargs
    ):
        """Inner loop for the optimizer, after all DAG-independent set-up has been completed."""
        for node in dag.op_nodes():
            if node.name not in CONTROL_FLOW_OP_NAMES:
                continue
            new_op = node.op.replace_blocks(
                [
                    dag_to_circuit(
                        self._run_main_loop(
                            circuit_to_dag(block),
                            {
                                inner: qubit_indices[outer]
                                for inner, outer in zip(block.qubits, node.qargs)
                            },
                            plugin_method,
                            plugin_kwargs,
                            default_method,
                            default_kwargs,
                        ),
                        copy_operations=False,
                    )
                    for block in node.op.blocks
                ]
            )
            dag.substitute_node(node, new_op)

        out_dag = dag.copy_empty_like()
        for node in dag.topological_op_nodes():
            if node.name in self._synth_gates and len(node.qargs) >= self._min_qubits:
                synth_dag = None
                unitary = node.matrix
                n_qubits = len(node.qargs)
                use_default_method = (
                    plugin_method.max_qubits is not None and n_qubits > plugin_method.max_qubits
                ) or (plugin_method.min_qubits is not None and n_qubits < plugin_method.min_qubits)

                if use_default_method:
                    method, kwargs = default_method, default_kwargs
                else:
                    method, kwargs = plugin_method, plugin_kwargs

                synth_dag = self._run_plugin_synthesis(
                    method, kwargs, unitary, qubit_indices, node.qargs
                )

                # In the case that a non-default method was used and the option fallback_on_default
                # is set, check whether the unitary was successfully synthesized: the returned
                # circuit is not ``None``. If not, we fall back on running the default plugin.
                if (not use_default_method) and self._fallback_on_default and (synth_dag is None):
                    synth_dag = self._run_plugin_synthesis(
                        default_method, default_kwargs, unitary, qubit_indices, node.qargs
                    )

                if synth_dag is None:
                    out_dag._apply_op_node_back(node)
                    continue
                if isinstance(synth_dag, DAGCircuit):
                    qubit_map = dict(zip(synth_dag.qubits, node.qargs))
                    for node in synth_dag.topological_op_nodes():
                        node.qargs = tuple(qubit_map[x] for x in node.qargs)
                        out_dag._apply_op_node_back(node)
                    out_dag.global_phase += synth_dag.global_phase
                else:
                    node_list, global_phase, gate = synth_dag
                    qubits = node.qargs
                    user_gate_node = DAGOpNode(gate)
                    for (
                        gate,
                        params,
                        qargs,
                    ) in node_list:
                        if gate is None:
                            node = DAGOpNode.from_instruction(
                                user_gate_node._to_circuit_instruction().replace(
                                    params=user_gate_node.params,
                                    qubits=tuple(qubits[x] for x in qargs),
                                )
                            )
                        else:
                            node = DAGOpNode.from_instruction(
                                CircuitInstruction.from_standard(
                                    gate, tuple(qubits[x] for x in qargs), params
                                )
                            )
                        out_dag._apply_op_node_back(node)
                    out_dag.global_phase += global_phase
            else:
                out_dag._apply_op_node_back(node)
        return out_dag


def _build_gate_lengths(target=None):
    """Builds a ``gate_lengths`` dictionary from ``target`` (BackendV2).

    The dictionary has the form:
    {gate_name: {(qubits,): duration}}
    """
    gate_lengths = {}
    if target is not None:
        for gate, prop_dict in target.items():
            gate_lengths[gate] = {}
            for qubit, gate_props in prop_dict.items():
                if gate_props is not None and gate_props.duration is not None:
                    gate_lengths[gate][qubit] = gate_props.duration
    return gate_lengths


def _build_gate_errors(target=None):
    """Builds a ``gate_error`` dictionary from ``target`` (BackendV2).

    The dictionary has the form:
    {gate_name: {(qubits,): error_rate}}
    """
    gate_errors = {}
    if target is not None:
        for gate, prop_dict in target.items():
            gate_errors[gate] = {}
            for qubit, gate_props in prop_dict.items():
                if gate_props is not None and gate_props.error is not None:
                    gate_errors[gate][qubit] = gate_props.error
    return gate_errors


def _build_gate_lengths_by_qubit(target=None):
    """
    Builds a `gate_lengths` dictionary from `target (BackendV2)`.

    The dictionary has the form:
    {(qubits): [Gate, duration]}
    """
    gate_lengths = {}
    if target is not None and target.qargs is not None:
        for qubits in target.qargs:
            names = target.operation_names_for_qargs(qubits)
            operation_and_durations = []
            for name in names:
                operation = target.operation_from_name(name)
                duration = getattr(target[name].get(qubits, None), "duration", None)
                if duration:
                    operation_and_durations.append((operation, duration))
            if operation_and_durations:
                gate_lengths[qubits] = operation_and_durations
    return gate_lengths


def _build_gate_errors_by_qubit(target=None):
    """
    Builds a `gate_error` dictionary from `target (BackendV2)`.

    The dictionary has the form:
    {(qubits): [Gate, error]}
    """
    gate_errors = {}
    if target is not None and target.qargs is not None:
        for qubits in target.qargs:
            names = target.operation_names_for_qargs(qubits)
            operation_and_errors = []
            for name in names:
                operation = target.operation_from_name(name)
                error = getattr(target[name].get(qubits, None), "error", None)
                if error:
                    operation_and_errors.append((operation, error))
            if operation_and_errors:
                gate_errors[qubits] = operation_and_errors
    return gate_errors
