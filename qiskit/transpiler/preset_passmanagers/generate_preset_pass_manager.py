# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Preset pass manager generation function
"""

from __future__ import annotations

import copy
import os
import warnings
import typing


from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES, get_control_flow_name_mapping
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.circuit import Qubit
from qiskit.providers.backend import Backend
from qiskit.quantum_info import get_clifford_gate_names
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passmanager_config import (
    PassManagerConfig,
    PassManagerCliffordTConfig,
)
from qiskit.transpiler.preset_passmanagers.clifford_t import (
    clifford_t_pass_manager,
    clifford_t_pass_manager_legacy,
)
from qiskit.transpiler.preset_passmanagers.common import is_clifford_t_basis, CLIFFORD_T_BASIS
from qiskit.transpiler.target import Target, _FakeTarget
from qiskit import user_config

from .level0 import level_0_pass_manager
from .level1 import level_1_pass_manager
from .level2 import level_2_pass_manager
from .level3 import level_3_pass_manager

if typing.TYPE_CHECKING:
    from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
    from qiskit.transpiler.passmanager import StagedPassManager


OVER_3Q_GATES = ["ccx", "ccz", "cswap", "rccx", "c3x", "c3sx", "rc3x"]


def generate_preset_pass_manager(
    optimization_level: int = 2,
    backend: Backend | None = None,
    target: Target | None = None,
    basis_gates: list[str] | None = None,
    coupling_map: CouplingMap | list | None = None,
    initial_layout: Layout | list[int] = None,
    layout_method: str | None = None,
    routing_method: str | None = None,
    translation_method: str | None = None,
    scheduling_method: str | None = None,
    approximation_degree: float | None = 1.0,
    seed_transpiler: int | None = None,
    unitary_synthesis_method: str = "default",
    unitary_synthesis_plugin_config: dict | None = None,
    hls_config: HLSConfig | None = None,
    init_method: str | None = None,
    optimization_method: str | None = None,
    dt: float | None = None,
    qubits_initially_zero: bool = True,
    *,
    _skip_target=False,
):
    """Generate a preset :class:`~.PassManager`

    This function is used to quickly generate a preset pass manager. Preset pass
    managers are the default pass managers used by the :func:`~.transpile`
    function. This function provides a convenient and simple method to construct
    a standalone :class:`~.PassManager` object that mirrors what the :func:`~.transpile`
    function internally builds and uses.

    The target constraints for the pass manager construction can be specified through a :class:`.Target`
    instance, a :class:`.BackendV2` instance, or via loose constraints
    (``basis_gates``, ``coupling_map``, or ``dt``).
    The order of priorities for target constraints works as follows: if a ``target``
    input is provided, it will take priority over any ``backend`` input or loose constraints.
    If a ``backend`` is provided together with any loose constraint
    from the list above, the loose constraint will take priority over the corresponding backend
    constraint. This behavior is summarized in the table below. The first column
    in the table summarizes the potential user-provided constraints, and each cell shows whether
    the priority is assigned to that specific constraint input or another input
    (`target`/`backend(V1)`/`backend(V2)`).

    ============================ ========= ========================
    User Provided                target    backend(V2)
    ============================ ========= ========================
    **basis_gates**              target    basis_gates
    **coupling_map**             target    coupling_map
    **dt**                       target    dt
    ============================ ========= ========================

    .. note::

        When the target basis consists of Clifford+T gates, this function constructs
        a specialized Clifford+T transpiler pipeline, see
        :func:`.generate_preset_clifford_t_pass_manager` for more detailed documentation. Arguments
        that apply only to transpiling into continuous basis sets are ignored in this flow.
        For example, the ``"unitary_synthesis_method"`` is not taken into account when synthesizing
        single-qubit unitaries into a Clifford+T sequence.

    Args:
        optimization_level: The optimization level to generate a
            :class:`~.StagedPassManager` for. By default optimization level 2
            is used if this is not specified. This can be 0, 1, 2, or 3. Higher
            levels generate potentially more optimized circuits, at the expense
            of longer transpilation time:

                * 0: no optimization
                * 1: light optimization
                * 2: heavy optimization
                * 3: even heavier optimization

        backend: An optional backend object which can be used as the
            source of the default values for the ``basis_gates``,
            ``coupling_map``, and ``target``. If any of those other arguments
            are specified in addition to ``backend`` they will take precedence
            over the value contained in the backend.
        target: The :class:`~.Target` representing a backend compilation
            target. The following attributes will be inferred from this
            argument if they are not set: ``coupling_map`` and ``basis_gates``.
        basis_gates: List of basis gate names to unroll to
            (e.g: ``['u1', 'u2', 'u3', 'cx']``).
        coupling_map: Directed graph represented a coupling
            map. Multiple formats are supported:

            #. ``CouplingMap`` instance
            #. List, must be given as an adjacency matrix, where each entry
               specifies all directed two-qubit interactions supported by backend,
               e.g: ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``
        initial_layout: Initial position of virtual qubits on
            physical qubits.
        layout_method: The :class:`~.Pass` to use for choosing initial qubit
            placement. Valid choices are ``'trivial'``, ``'dense'``,
            and ``'sabre'``, representing :class:`~.TrivialLayout`, :class:`~.DenseLayout` and
            :class:`~.SabreLayout` respectively. This can also
            be the external plugin name to use for the ``layout`` stage of the output
            :class:`~.StagedPassManager`. You can see a list of installed plugins by using
            :func:`~.list_stage_plugins` with ``"layout"`` for the ``stage_name`` argument.
        routing_method: The pass to use for routing qubits on the
            architecture. Valid choices are ``'basic'``, ``'lookahead'``,
            ``'sabre'``, and ``'none'`` representing :class:`~.BasicSwap`,
            :class:`~.LookaheadSwap`, :class:`~.SabreSwap`, and
            erroring if routing is required respectively. This can also be the external plugin
            name to use for the ``routing`` stage of the output :class:`~.StagedPassManager`.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"routing"`` for the ``stage_name`` argument.
        translation_method: The method to use for translating gates to
            basis gates. Valid choices ``'translator'``, ``'synthesis'`` representing
            :class:`~.BasisTranslator`, and :class:`~.UnitarySynthesis` respectively. This can
            also be the external plugin name to use for the ``translation`` stage of the output
            :class:`~.StagedPassManager`. You can see a list of installed plugins by using
            :func:`~.list_stage_plugins` with ``"translation"`` for the ``stage_name`` argument.
        scheduling_method: The pass to use for scheduling instructions. Valid choices
            are ``'alap'`` and ``'asap'``. This can also be the external plugin name to use
            for the ``scheduling`` stage of the output :class:`~.StagedPassManager`. You can
            see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"scheduling"`` for the ``stage_name`` argument.
        approximation_degree: Heuristic dial used for circuit approximation, where
            ``1.0`` means no approximation (up to numerical tolerance) and ``0.0``
            means the maximum approximation. If ``target`` is available, a value of ``None``
            indicates that approximation is allowed up to the reported error rate for an operation
            in the target.
        seed_transpiler: Sets random seed for the stochastic parts of
            the transpiler. If it is not specified here it can also be specified via an environment
            variable: ``QISKIT_TRANSPILER_SEED`` or in a user configuration file. The priority
            order is: this argument, then the environment variable, and finally the user
            configuration option. So setting this argument will take precedence over the other
            methods of setting a seed.
        unitary_synthesis_method (str): The name of the unitary synthesis
            method to use. By default ``'default'`` is used. You can see a list of
            installed plugins with :func:`.unitary_synthesis_plugin_names`.
        unitary_synthesis_plugin_config: An optional configuration dictionary
            that will be passed directly to the unitary synthesis plugin. By
            default this setting will have no effect as the default unitary
            synthesis method does not take custom configuration. This should
            only be necessary when a unitary synthesis plugin is specified with
            the ``unitary_synthesis_method`` argument. As this is custom for each
            unitary synthesis plugin refer to the plugin documentation for how
            to use this option.
        hls_config: An optional configuration class :class:`~.HLSConfig`
            that will be passed directly to :class:`~.HighLevelSynthesis` transformation pass.
            This configuration class allows to specify for various high-level objects
            the lists of synthesis algorithms and their parameters.
        init_method: The plugin name to use for the ``init`` stage of
            the output :class:`~.StagedPassManager`. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"init"`` for the stage
            name argument.
        optimization_method: The plugin name to use for the
            ``optimization`` stage of the output
            :class:`~.StagedPassManager`. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"optimization"`` for the
            ``stage_name`` argument.
        dt: Backend sample time (resolution) in seconds.
            If ``None`` (default) and a backend is provided, ``backend.dt`` is used.
        qubits_initially_zero: Indicates whether the input circuit is
                zero-initialized.

    Returns:
        StagedPassManager: The preset pass manager for the given options

    Raises:
        ValueError: if an invalid value for ``optimization_level`` is passed in.
    """
    config = user_config.get_config()

    if seed_transpiler is None:
        if (seed := os.getenv("QISKIT_TRANSPILER_SEED", None)) is not None:
            seed_transpiler = int(seed)
        else:
            seed_transpiler = config.get("transpiler_seed", None)

    # Handle positional arguments for target and backend. This enables the usage
    # pattern `generate_preset_pass_manager(backend.target)` to generate a default
    # pass manager for a given target.
    if isinstance(optimization_level, Target):
        target, optimization_level = optimization_level, None
    elif isinstance(optimization_level, Backend):
        backend, optimization_level = optimization_level, None
    if optimization_level is None:
        optimization_level = config.get("transpile_optimization_level", 2)

    common_options = _parse_common_options(
        backend=backend,
        target=target,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        initial_layout=initial_layout,
        approximation_degree=approximation_degree,
        seed_transpiler=seed_transpiler,
        unitary_synthesis_method=unitary_synthesis_method,
        unitary_synthesis_plugin_config=unitary_synthesis_plugin_config,
        hls_config=hls_config,
        dt=dt,
        qubits_initially_zero=qubits_initially_zero,
    )

    specific_options = {
        "layout_method": layout_method,
        "routing_method": routing_method,
        "translation_method": translation_method,
        "scheduling_method": scheduling_method,
        "init_method": init_method,
        "optimization_method": optimization_method,
    }

    pm_options = common_options | specific_options

    if is_clifford_t_basis(pm_options["basis_gates"], pm_options["target"]):
        pm_config = PassManagerConfig(**pm_options)
        pm = clifford_t_pass_manager_legacy(pm_config, optimization_level=optimization_level)
        return pm

    if backend is not None:
        pm_options["_skip_target"] = _skip_target
        pm_config = PassManagerConfig.from_backend(backend, **pm_options)
    else:
        pm_config = PassManagerConfig(**pm_options)

    if optimization_level == 0:
        pm = level_0_pass_manager(pm_config)
    elif optimization_level == 1:
        pm = level_1_pass_manager(pm_config)
    elif optimization_level == 2:
        pm = level_2_pass_manager(pm_config)
    elif optimization_level == 3:
        pm = level_3_pass_manager(pm_config)
    else:
        raise ValueError(f"Invalid optimization level {optimization_level}")

    return pm


def generate_preset_clifford_t_pass_manager(
    optimization_level: int = 2,
    target: Target | None = None,
    basis_gates: list[str] | None = None,
    coupling_map: CouplingMap | list | None = None,
    initial_layout: Layout | list[int] = None,
    approximation_degree: float | None = 1.0,
    seed_transpiler: int | None = None,
    unitary_synthesis_method: str = "default",
    unitary_synthesis_plugin_config: dict | None = None,
    hls_config: HLSConfig | None = None,
    dt: float | None = None,
    qubits_initially_zero: bool = True,
    rz_synthesis_config: dict | None = None,
) -> StagedPassManager:
    """Generate a preset Clifford+T :class:`~.StagedPassManager`.

    This function provides a convenient way to construct a preset pass manager for
    Clifford+T compilation. We recommend using this function instead of :func:`~.transpile` or
    :func:`~.generate_preset_pass_manager` with a Clifford+T basis, as it exposes
    arguments specifically tailored for Clifford+T compilations.

    The target constraints for the pass manager construction can be specified through a :class:`.Target`
    instance, or via loose constraints (``basis_gates``, ``coupling_map``, or ``dt``).

    If basis gates are not specified (neither via ``target`` nor via ``basis_gates``), then basis gates
    default to all of the Clifford+T gates in Qiskit. Note, however, that if basis gates are specified
    but do not represent a Clifford+T basis, then an error will be raised.

    Examples:
        Generate and use a simple Clifford+T pass manager::

            from qiskit.circuit import QuantumCircuit
            from qiskit.transpiler import generate_preset_clifford_t_pass_manager

            qc = QuantumCircuit(1)
            qc.rz(2.3579, 0)

            pm = generate_preset_clifford_t_pass_manager()
            qct = pm.run(qc)

        Various options are configurable; see the arguments documentation for more detail::

            rz_synthesis_config = {
                "rz_synthesis_error": 1e-4,
                "rz_cache_error": 1e-5,
            }

            basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
            pm = generate_preset_clifford_t_pass_manager(
                rz_synthesis_config=rz_synthesis_config,
                basis_gates=basis_gates,
                optimization_level=3,
            )

    Args:
        optimization_level: The optimization level to generate a
            :class:`~.StagedPassManager` for. By default optimization level 2
            is used if this is not specified. This can be 0, 1, 2, or 3. Higher
            levels generate potentially more optimized circuits, at the expense
            of potentially longer transpilation time.
        target: The :class:`~.Target` representing a compilation
            target. The following attributes will be inferred from this
            argument if they are not set: ``coupling_map`` and ``basis_gates``.
        basis_gates: List of basis gate names to unroll to (e.g: ``['cx', 's', 'sx', 't', 'tdg']``).
            If both ``target`` and ``basis_gates`` are ``None``, ``basis_gates`` will be set
            to all of the standard Clifford gates together with ``'t'`` and ``'tdg'``.
        coupling_map: Directed graph represented a coupling
            map. Multiple formats are supported:

            #. ``CouplingMap`` instance
            #. List, must be given as an adjacency matrix, where each entry
               specifies all directed two-qubit interactions supported by backend,
               e.g: ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``
        dt: Target sample time (resolution) in seconds.
            If ``None`` (default) and a target is provided, ``target.dt`` is used.
        initial_layout: Initial position of virtual qubits on
            physical qubits.
        approximation_degree: Heuristic dial used for circuit approximation, where
            ``1.0`` means no approximation (up to numerical tolerance) and ``0.0``
            means the maximum approximation. If ``target`` is available, a value of ``None``
            indicates that approximation is allowed up to the reported error rate for an operation
            in the target.
        seed_transpiler: Sets random seed for the stochastic parts of
            the transpiler. If it is not specified here it can also be specified via an environment
            variable: ``QISKIT_TRANSPILER_SEED`` or in a user configuration file. The priority
            order is: this argument, then the environment variable, and finally the user
            configuration option. So setting this argument will take precedence over the other
            methods of setting a seed.
        unitary_synthesis_method: The name of the unitary synthesis
            method to use. By default ``'default'`` is used. You can see a list of
            installed plugins with :func:`.unitary_synthesis_plugin_names`.
        unitary_synthesis_plugin_config: An optional configuration dictionary
            that will be passed directly to the unitary synthesis plugin. By
            default this setting will have no effect as the default unitary
            synthesis method does not take custom configuration. This should
            only be necessary when a unitary synthesis plugin is specified with
            the ``unitary_synthesis_method`` argument. As this is custom for each
            unitary synthesis plugin refer to the plugin documentation for how
            to use this option.
        hls_config: An optional configuration class :class:`~.HLSConfig`
            that will be passed directly to :class:`~.HighLevelSynthesis` transformation pass.
            This configuration class allows to specify for various high-level objects
            the lists of synthesis algorithms and their parameters.
        qubits_initially_zero: Indicates whether the input circuit is
            zero-initialized.
        rz_synthesis_config: An optional configuration class to use for
            :class:`~qiskit.transpiler.passes.SynthesizeRZRotations` pass.
            Specifies how to synthesize RZ rotations in the circuit.

    Returns:
        StagedPassManager: The preset pass manager for the given options.

    Raises:
        TranspilerError: if a basis other than Clifford+T is specified.
        ValueError: if an invalid value for ``optimization_level`` is passed in.
    """
    config = user_config.get_config()
    if seed_transpiler is None:
        if (seed := os.getenv("QISKIT_TRANSPILER_SEED", None)) is not None:
            seed_transpiler = int(seed)
        else:
            seed_transpiler = config.get("transpiler_seed", None)

    if optimization_level is None:
        optimization_level = config.get("transpile_optimization_level", 2)

    if target is None and basis_gates is None:
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]

    common_options = _parse_common_options(
        target=target,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        initial_layout=initial_layout,
        approximation_degree=approximation_degree,
        seed_transpiler=seed_transpiler,
        unitary_synthesis_method=unitary_synthesis_method,
        unitary_synthesis_plugin_config=unitary_synthesis_plugin_config,
        hls_config=hls_config,
        dt=dt,
        qubits_initially_zero=qubits_initially_zero,
    )
    specific_options = {
        "rz_synthesis_config": rz_synthesis_config,
    }
    pm_options = common_options | specific_options

    pm_config = PassManagerCliffordTConfig(**pm_options)

    # Check that the gates in the basis are a subset of Clifford+T gates.
    # (This is slightly different from is_clifford_t_basis since we do not
    # require that either T or T^\dagger must be in the basis).
    if pm_config.target is not None:
        basis = set(pm_config.target.operation_names)
    elif pm_config.basis_gates is not None:
        basis = set(pm_config.basis_gates)
    else:
        basis = set()

    if not basis.issubset(CLIFFORD_T_BASIS):
        raise TranspilerError(
            "generate_preset_clifford_t_pass_manager can only be used with Clifford+T basis"
        )

    pm = clifford_t_pass_manager(pm_config, optimization_level=optimization_level)
    return pm


def _parse_basis_gates(basis_gates, backend):
    standard_gates = get_standard_gate_name_mapping()
    # Add control flow gates by default to basis set and name mapping
    default_gates = {"measure", "delay", "reset"}.union(CONTROL_FLOW_OP_NAMES)
    name_mapping = get_control_flow_name_mapping()
    try:
        instructions = set(basis_gates)
        for name in default_gates:
            if name not in instructions:
                instructions.add(name)
    except TypeError:
        instructions = None

    if backend is None:
        # Check for custom instructions
        if instructions is None:
            return None, name_mapping

        for inst in instructions:
            if inst not in standard_gates and inst not in default_gates:
                raise ValueError(
                    f"Providing non-standard gates ({inst}) through the ``basis_gates`` "
                    "argument is not allowed. Use the ``target`` parameter instead. "
                    "You can build a target instance using ``Target.from_configuration()`` and provide "
                    "custom gate definitions with the ``custom_name_mapping`` argument."
                )

        return list(instructions), name_mapping

    instructions = instructions or backend.operation_names
    name_mapping.update(
        {name: backend.target.operation_from_name(name) for name in backend.operation_names}
    )

    # Check for custom instructions
    for inst in instructions:
        if inst not in standard_gates and inst not in default_gates:
            if inst not in backend.operation_names:
                # do not raise error when the custom instruction comes from the backend
                # (common case with BasicSimulator)
                raise ValueError(
                    f"Providing non-standard gates ({inst}) through the ``basis_gates`` "
                    "argument is not allowed. Use the ``target`` parameter instead. "
                    "You can build a target instance using ``Target.from_configuration()`` and provide "
                    "custom gate definitions with the ``custom_name_mapping`` argument."
                )

    return list(instructions) if instructions else None, name_mapping


def _parse_dt(dt, backend):
    # try getting dt from user, else backend
    if dt is None and backend is not None:
        dt = backend.target.dt
    return dt


def _parse_coupling_map(coupling_map, backend):
    # try getting coupling_map from user, else backend
    if coupling_map is None and backend is not None:
        coupling_map = backend.coupling_map

    # coupling_map could be None, or a list of lists, e.g. [[0, 1], [2, 1]]
    if coupling_map is None or isinstance(coupling_map, CouplingMap):
        return coupling_map
    if isinstance(coupling_map, list) and all(
        isinstance(i, list) and len(i) == 2 for i in coupling_map
    ):
        return CouplingMap(coupling_map)
    else:
        raise TranspilerError(
            "Only a single input coupling map can be used with generate_preset_pass_manager()."
        )


def _parse_instruction_durations(backend, dt):
    """Create a list of ``InstructionDuration``s populated from the backend."""
    final_durations = InstructionDurations()
    backend_durations = InstructionDurations()
    if backend is not None:
        backend_durations = backend.instruction_durations
    final_durations.update(backend_durations, dt or backend_durations.dt)
    return final_durations


def _parse_timing_constraints(backend):
    # If backend is None, we get timing constraints from Target later
    if backend is None:
        return None
    else:
        return backend.target.timing_constraints()


def _parse_initial_layout(initial_layout):
    # initial_layout could be None, or a list of ints, e.g. [0, 5, 14]
    # or a list of tuples/None e.g. [qr[0], None, qr[1]] or a dict e.g. {qr[0]: 0}
    if initial_layout is None or isinstance(initial_layout, Layout):
        return initial_layout
    if isinstance(initial_layout, dict):
        return Layout(initial_layout)
    initial_layout = list(initial_layout)
    if all(phys is None or isinstance(phys, Qubit) for phys in initial_layout):
        return Layout.from_qubit_list(initial_layout)
    return initial_layout


def _parse_approximation_degree(approximation_degree):
    if approximation_degree is None:
        return None
    if approximation_degree < 0.0 or approximation_degree > 1.0:
        raise TranspilerError("Approximation degree must be in [0.0, 1.0]")
    return approximation_degree


def _parse_seed_transpiler(seed_transpiler):
    if seed_transpiler is None:
        return None
    if not isinstance(seed_transpiler, int) or seed_transpiler < 0:
        raise ValueError("Expected non-negative integer as seed for transpiler.")
    return seed_transpiler


def _parse_common_options(
    backend: Backend | None = None,
    target: Target | None = None,
    basis_gates: list[str] | None = None,
    coupling_map: CouplingMap | list | None = None,
    initial_layout: Layout | list[int] = None,
    approximation_degree: float | None = 1.0,
    seed_transpiler: int | None = None,
    unitary_synthesis_method: str = "default",
    unitary_synthesis_plugin_config: dict | None = None,
    hls_config: HLSConfig | None = None,
    dt: float | None = None,
    qubits_initially_zero: bool = True,
) -> dict:
    """Parses options that are used both for standard transpilation and Clifford+T transpilation."""

    # If there are no loose constraints => use backend target if available
    _no_loose_constraints = basis_gates is None and coupling_map is None and dt is None

    # If the only loose constraint is dt => use backend target and modify dt
    _adjust_dt = backend is not None and dt is not None

    # Warn about inconsistencies in backend + loose constraints path (dt shouldn't be a problem)
    if backend is not None and (coupling_map is not None or basis_gates is not None):
        warnings.warn(
            "Providing `coupling_map` and/or `basis_gates` along with `backend` is not "
            "recommended, as this will invalidate the backend's gate durations and error rates.",
            category=UserWarning,
            stacklevel=2,
        )

    # Resolve loose constraints case-by-case against backend constraints.
    # The order of priority is loose constraints > backend.
    dt = _parse_dt(dt, backend)
    instruction_durations = _parse_instruction_durations(backend, dt)
    timing_constraints = _parse_timing_constraints(backend)
    coupling_map = _parse_coupling_map(coupling_map, backend)
    basis_gates, name_mapping = _parse_basis_gates(basis_gates, backend)

    # Check if coupling map has been provided (either standalone or through backend)
    # with user-defined basis_gates, and whether these have 3q or more.
    if coupling_map is not None and basis_gates is not None:
        for gate in OVER_3Q_GATES:
            if gate in basis_gates:
                raise ValueError(
                    f"Gates with 3 or more qubits ({gate}) in `basis_gates` or `backend` are "
                    "incompatible with a custom `coupling_map`. To include 3-qubit or larger "
                    " gates in the transpilation basis, provide a custom `target` instead."
                )

    if target is None:
        if backend is not None and _no_loose_constraints:
            # If a backend is specified without loose constraints, use its target directly.
            target = backend.target
        elif _adjust_dt:
            # If a backend is specified with loose dt, use its target and adjust the dt value.
            target = copy.deepcopy(backend.target)
            target.dt = dt
        elif basis_gates is not None:
            # Build target from constraints.
            target = Target.from_configuration(
                basis_gates=basis_gates,
                num_qubits=backend.num_qubits if backend is not None else None,
                coupling_map=coupling_map,
                instruction_durations=instruction_durations,
                concurrent_measurements=(
                    backend.target.concurrent_measurements if backend is not None else None
                ),
                dt=dt,
                timing_constraints=timing_constraints,
                custom_name_mapping=name_mapping,
            )
        else:
            target = _FakeTarget.from_configuration(
                num_qubits=backend.num_qubits if backend is not None else None,
                coupling_map=coupling_map,
                dt=dt,
            )

    # Update loose constraints to populate pm options
    if coupling_map is None:
        coupling_map = target.build_coupling_map()
    if basis_gates is None and len(target.operation_names) > 0:
        basis_gates = target.operation_names
    if instruction_durations is None:
        instruction_durations = target.durations()
    if timing_constraints is None:
        timing_constraints = target.timing_constraints()

    # Parse non-target dependent pm options
    initial_layout = _parse_initial_layout(initial_layout)
    approximation_degree = _parse_approximation_degree(approximation_degree)
    seed_transpiler = _parse_seed_transpiler(seed_transpiler)

    options = {
        "target": target,
        "basis_gates": basis_gates,
        "coupling_map": coupling_map,
        "instruction_durations": instruction_durations,
        "timing_constraints": timing_constraints,
        "approximation_degree": approximation_degree,
        "seed_transpiler": seed_transpiler,
        "unitary_synthesis_method": unitary_synthesis_method,
        "unitary_synthesis_plugin_config": unitary_synthesis_plugin_config,
        "initial_layout": initial_layout,
        "hls_config": hls_config,
        "qubits_initially_zero": qubits_initially_zero,
    }

    return options
