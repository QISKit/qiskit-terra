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

"""Pass Manager Configuration class."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
    from qiskit.transpiler import Target, CouplingMap, Layout
    from qiskit.transpiler.timing_constraints import TimingConstraints
    from .instruction_durations import InstructionDurations


class PassManagerConfig:
    """Pass Manager Configuration."""

    def __init__(
        self,
        initial_layout: Layout | None = None,
        basis_gates: list[str] | None = None,
        coupling_map: CouplingMap | None = None,
        layout_method: str | None = None,
        routing_method: str | None = None,
        translation_method: str | None = None,
        scheduling_method: str | None = None,
        instruction_durations: InstructionDurations | None = None,
        approximation_degree: float | None = None,
        seed_transpiler: int | None = None,
        timing_constraints: TimingConstraints | None = None,
        unitary_synthesis_method: str = "default",
        unitary_synthesis_plugin_config: dict | None = None,
        target: Target | None = None,
        hls_config: HLSConfig | None = None,
        init_method: str | None = None,
        optimization_method: str | None = None,
        qubits_initially_zero: bool = True,
    ):
        """Initialize a PassManagerConfig object

        Args:
            initial_layout: Initial position of virtual qubits on
                physical qubits.
            basis_gates: List of basis gate names to unroll to.
            coupling_map: Directed graph representing a coupling
                map.
            layout_method: the pass to use for choosing initial qubit
                placement. This will be the plugin name if an external layout stage
                plugin is being used.
            routing_method: the pass to use for routing qubits on the
                architecture. This will be a plugin name if an external routing stage
                plugin is being used.
            translation_method: the pass to use for translating gates to
                basis_gates. This will be a plugin name if an external translation stage
                plugin is being used.
            scheduling_method: the pass to use for scheduling instructions. This will
                be a plugin name if an external scheduling stage plugin is being used.
            instruction_durations: Dictionary of duration
                (in dt) for each instruction.
            approximation_degree: Heuristic dial used for circuit approximation, where
                ``1.0`` means no approximation (up to numerical tolerance) and ``0.0``
                means the maximum approximation. If ``target`` is available, a value of ``None``
                indicates that approximation is allowed up to the reported error rate for an operation
                in the target.
            seed_transpiler: Sets random seed for the stochastic parts of
                the transpiler.
            timing_constraints: Hardware time alignment restrictions.
            unitary_synthesis_method: The string method to use for the
                :class:`~qiskit.transpiler.passes.UnitarySynthesis` pass. Will
                search installed plugins for a valid method. You can see a list of
                installed plugins with :func:`.unitary_synthesis_plugin_names`.
            unitary_synthesis_plugin_config: The configuration dictionary that will
                be passed to the specified unitary synthesis plugin. Refer to
                the plugin documentation for how to use this.
            target: The backend target
            hls_config: An optional configuration class to use for
                :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass.
                Specifies how to synthesize various high-level objects.
            init_method: The plugin name for the init stage plugin to use
            optimization_method: The plugin name for the optimization stage plugin
                to use.
            qubits_initially_zero: Indicates whether the input circuit is
                zero-initialized.
        """
        self.initial_layout = initial_layout
        self.basis_gates = basis_gates
        self.coupling_map = coupling_map
        self.init_method = init_method
        self.layout_method = layout_method
        self.routing_method = routing_method
        self.translation_method = translation_method
        self.optimization_method = optimization_method
        self.scheduling_method = scheduling_method
        self.instruction_durations = instruction_durations
        self.approximation_degree = approximation_degree
        self.seed_transpiler = seed_transpiler
        self.timing_constraints = timing_constraints
        self.unitary_synthesis_method = unitary_synthesis_method
        self.unitary_synthesis_plugin_config = unitary_synthesis_plugin_config
        self.target = target
        self.hls_config = hls_config
        self.qubits_initially_zero = qubits_initially_zero

    @classmethod
    def from_backend(cls, backend, _skip_target=False, **pass_manager_options):
        """Construct a configuration based on a backend and user input.

        This method automatically generates a PassManagerConfig object based on the backend's
        features. User options can be used to overwrite the configuration.

        Args:
            backend (BackendV2): The backend that provides the configuration.
            pass_manager_options: User-defined option-value pairs.

        Returns:
            PassManagerConfig: The configuration generated based on the arguments.

        Raises:
            AttributeError: If the backend does not support a `configuration()` method.
        """
        res = cls(**pass_manager_options)
        if res.basis_gates is None:
            res.basis_gates = backend.operation_names
        if res.coupling_map is None:
            res.coupling_map = backend.coupling_map
        if res.instruction_durations is None:
            res.instruction_durations = backend.instruction_durations
        if res.target is None and not _skip_target:
            res.target = backend.target
        if res.scheduling_method is None and hasattr(backend, "get_scheduling_stage_plugin"):
            res.scheduling_method = backend.get_scheduling_stage_plugin()
        if res.translation_method is None and hasattr(backend, "get_translation_stage_plugin"):
            res.translation_method = backend.get_translation_stage_plugin()
        return res

    def __str__(self):
        newline = "\n"
        newline_tab = "\n\t"
        return (
            "Pass Manager Config:\n"
            f"\tinitial_layout: {self.initial_layout}\n"
            f"\tbasis_gates: {self.basis_gates}\n"
            f"\tcoupling_map: {self.coupling_map}\n"
            f"\tlayout_method: {self.layout_method}\n"
            f"\trouting_method: {self.routing_method}\n"
            f"\ttranslation_method: {self.translation_method}\n"
            f"\tscheduling_method: {self.scheduling_method}\n"
            f"\tinstruction_durations: {str(self.instruction_durations).replace(newline, newline_tab)}\n"
            f"\tapproximation_degree: {self.approximation_degree}\n"
            f"\tseed_transpiler: {self.seed_transpiler}\n"
            f"\ttiming_constraints: {self.timing_constraints}\n"
            f"\tunitary_synthesis_method: {self.unitary_synthesis_method}\n"
            f"\tunitary_synthesis_plugin_config: {self.unitary_synthesis_plugin_config}\n"
            f"\tqubits_initially_zero: {self.qubits_initially_zero}\n"
            f"\ttarget: {str(self.target).replace(newline, newline_tab)}\n"
        )


class PassManagerCliffordTConfig:
    """Pass Manager Configuration for Clifford+T transpilation."""

    def __init__(
        self,
        initial_layout: Layout | None = None,
        basis_gates: list[str] | None = None,
        coupling_map: CouplingMap | None = None,
        instruction_durations: InstructionDurations | None = None,
        approximation_degree: float | None = None,
        seed_transpiler: int | None = None,
        timing_constraints: TimingConstraints | None = None,
        unitary_synthesis_method: str = "default",
        unitary_synthesis_plugin_config: dict | None = None,
        target: Target | None = None,
        hls_config: HLSConfig | None = None,
        qubits_initially_zero: bool = True,
        rz_synthesis_config: dict | None = None,
        *,
        _routing_disabled: bool = False,
    ):
        """

        Args:
            initial_layout: Initial position of virtual qubits on
                physical qubits.
            basis_gates: List of basis gate names to unroll to.
            coupling_map: Directed graph representing a coupling
                map.
            instruction_durations: Dictionary of duration
                (in dt) for each instruction.
            approximation_degree: Heuristic dial used for circuit approximation, where
                ``1.0`` means no approximation (up to numerical tolerance) and ``0.0``
                means the maximum approximation. If ``target`` is available, a value of ``None``
                indicates that approximation is allowed up to the reported error rate for an operation
                in the target.
            seed_transpiler: Sets random seed for the stochastic parts of
                the transpiler.
            timing_constraints: Hardware time alignment restrictions.
            unitary_synthesis_method: The string method to use for the
                :class:`~qiskit.transpiler.passes.UnitarySynthesis` pass. Will
                search installed plugins for a valid method. You can see a list of
                installed plugins with :func:`.unitary_synthesis_plugin_names`.
            unitary_synthesis_plugin_config: The configuration dictionary that will
                be passed to the specified unitary synthesis plugin. Refer to
                the plugin documentation for how to use this.
            target: The backend target.
            hls_config: An optional configuration class to use for
                :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass.
                Specifies how to synthesize various high-level objects.
            qubits_initially_zero: Indicates whether the input circuit is
                zero-initialized.
            rz_synthesis_config: An optional configuration class to use for
                :class:`~qiskit.transpiler.passes.SynthesizeRZRotations` pass.
                Specifies how to synthesize RZ rotations in the circuit.
        """
        self.initial_layout = initial_layout
        self.basis_gates = basis_gates
        self.coupling_map = coupling_map
        self.instruction_durations = instruction_durations
        self.approximation_degree = approximation_degree
        self.seed_transpiler = seed_transpiler
        self.timing_constraints = timing_constraints
        self.unitary_synthesis_method = unitary_synthesis_method
        self.unitary_synthesis_plugin_config = unitary_synthesis_plugin_config
        self.target = target
        self.hls_config = hls_config
        self.qubits_initially_zero = qubits_initially_zero
        self.rz_synthesis_config = rz_synthesis_config

        # _routing_disabled is needed for disabling routing when CliffordT pipeline
        # is called from generate_preset_pass_manager with routing_method="none".
        self._routing_disabled = _routing_disabled

    def _to_legacy_config(self) -> PassManagerConfig:
        """
        Returns PassManagerConfig created from PassManagerCliffordTConfig,
        using default values for stage methods.

        This is only used when calling default layout, routing and scheduling
        plugins from the CliffordT compilation pipeline.
        """
        return PassManagerConfig(
            initial_layout=self.initial_layout,
            basis_gates=self.basis_gates,
            coupling_map=self.coupling_map,
            layout_method=None,
            routing_method=None,
            translation_method=None,
            scheduling_method=None,
            instruction_durations=self.instruction_durations,
            approximation_degree=self.approximation_degree,
            seed_transpiler=self.seed_transpiler,
            timing_constraints=self.timing_constraints,
            unitary_synthesis_method=self.unitary_synthesis_method,
            unitary_synthesis_plugin_config=self.unitary_synthesis_plugin_config,
            target=self.target,
            hls_config=self.hls_config,
            init_method="default",
            optimization_method="default",
            qubits_initially_zero=self.qubits_initially_zero,
        )

    @classmethod
    def _from_legacy_config(cls, pass_manager_config: PassManagerConfig):
        """
        Returns PassManagerCliffordTConfig created from PassManagerConfig,
        using default values for RZ-synthesis options.

        This is only used when calling legacy Clifford+T synthesis pipeline
        from `generate_preset_pass_manager` or `transpile`.
        """
        return PassManagerCliffordTConfig(
            initial_layout=pass_manager_config.initial_layout,
            basis_gates=pass_manager_config.basis_gates,
            coupling_map=pass_manager_config.coupling_map,
            instruction_durations=pass_manager_config.instruction_durations,
            approximation_degree=pass_manager_config.approximation_degree,
            seed_transpiler=pass_manager_config.seed_transpiler,
            timing_constraints=pass_manager_config.timing_constraints,
            unitary_synthesis_method=pass_manager_config.unitary_synthesis_method,
            unitary_synthesis_plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
            target=pass_manager_config.target,
            hls_config=pass_manager_config.hls_config,
            qubits_initially_zero=pass_manager_config.qubits_initially_zero,
            rz_synthesis_config=None,
            _routing_disabled=pass_manager_config.routing_method == "none",
        )
