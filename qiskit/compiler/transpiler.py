# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Circuit transpile function"""

import logging
import os
from time import time
from typing import Any, TypeVar
from collections.abc import Callable

from qiskit import user_config
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.backend import Backend
from qiskit.transpiler import Layout, CouplingMap, PropertySet
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.exceptions import TranspilerError, CircuitTooWideForTarget
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.target import Target

logger = logging.getLogger(__name__)

_CircuitT = TypeVar("_CircuitT", bound=QuantumCircuit | list[QuantumCircuit])


def transpile(
    circuits: _CircuitT,
    backend: Backend | None = None,
    basis_gates: list[str] | None = None,
    coupling_map: CouplingMap | list[list[int]] | None = None,
    initial_layout: Layout | dict | list | None = None,
    layout_method: str | None = None,
    routing_method: str | None = None,
    translation_method: str | None = None,
    scheduling_method: str | None = None,
    dt: float | None = None,
    approximation_degree: float | None = 1.0,
    seed_transpiler: int | None = None,
    optimization_level: int | None = None,
    callback: Callable[[BasePass, DAGCircuit, float, PropertySet, int], Any] | None = None,
    output_name: str | list[str] | None = None,
    unitary_synthesis_method: str = "default",
    unitary_synthesis_plugin_config: dict | None = None,
    target: Target | None = None,
    hls_config: HLSConfig | None = None,
    init_method: str | None = None,
    optimization_method: str | None = None,
    ignore_backend_supplied_default_methods: bool = False,
    num_processes: int | None = None,
    qubits_initially_zero: bool = True,
) -> _CircuitT:
    """Transpile one or more circuits, according to some desired transpilation targets.

    Transpilation is potentially done in parallel using multiprocessing when ``circuits``
    is a list with > 1 :class:`~.QuantumCircuit` object, depending on the local environment
    and configuration.

    The prioritization of transpilation target constraints works as follows: if a ``target``
    input is provided, it will take priority over any ``backend`` input or loose constraints
    (``basis_gates``, ``coupling_map``, or ``dt``). If a ``backend`` is provided
    together with any loose constraint
    from the list above, the loose constraint will take priority over the corresponding backend
    constraint. This behavior is summarized in the table below. The first column
    in the table summarizes the potential user-provided constraints, and each cell shows whether
    the priority is assigned to that specific constraint input or another input
    (`target`/`backend(V2)`).

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
        circuits: Circuit(s) to transpile
        backend: If set, the transpiler will compile the input circuit to this target
            device. If any other option is explicitly set (e.g., ``coupling_map``), it
            will override the backend's. A :class:`~qiskit.transpiler.Target` instance
            may also be passed here as a positional argument, in which case it is treated
            as if it were passed as the ``target`` keyword argument instead. This mirrors
            the behaviour of :func:`~.generate_preset_pass_manager`.
        basis_gates: List of basis gate names to unroll to
            (e.g.: ``['u1', 'u2', 'u3', 'cx']``). If ``None``, do not unroll.
        coupling_map: Directed coupling map (perhaps custom) to target in mapping. If
            the coupling map is symmetric, both directions need to be specified.

            Multiple formats are supported:

            #. ``CouplingMap`` instance
            #. List, must be given as an adjacency matrix, where each entry
               specifies all directed two-qubit interactions supported by backend,
               e.g.: ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``
        initial_layout: Initial position of virtual qubits on physical qubits.
            If this layout makes the circuit compatible with the coupling_map
            constraints, it will be used. The final layout is not guaranteed to be the same,
            as the transpiler may permute qubits through swaps or other means.
            Multiple formats are supported:

            #. ``Layout`` instance
            #. Dict
               * virtual to physical::

                    {qr[0]: 0,
                     qr[1]: 3,
                     qr[2]: 5}

               * physical to virtual::

                    {0: qr[0],
                     3: qr[1],
                     5: qr[2]}

            #. List

               * virtual to physical::

                    [0, 3, 5]  # virtual qubits are ordered (in addition to named)

               * physical to virtual::

                    [qr[0], None, None, qr[1], None, qr[2]]

        layout_method: Name of layout selection pass ('trivial', 'dense', 'sabre').
            This can also be the external plugin name to use for the ``layout`` stage.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"layout"`` for the ``stage_name`` argument.
        routing_method: Name of routing pass
            ('basic', 'lookahead', 'stochastic', 'sabre', 'none').
            This can also be the external plugin name to use for the ``routing`` stage.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"routing"`` for the ``stage_name`` argument.
        translation_method: Name of translation pass (``"default"``, ``"translator"`` or
            ``"synthesis"``). This can also be the external plugin name to use for the
            ``translation`` stage.  You can see a list of installed plugins by using
            :func:`~.list_stage_plugins` with ``"translation"`` for the ``stage_name`` argument.
        scheduling_method: Name of scheduling pass.
            * ``'as_soon_as_possible'``: Schedule instructions greedily, as early as possible
            on a qubit resource. (alias: ``'asap'``)
            * ``'as_late_as_possible'``: Schedule instructions late, i.e. keeping qubits
            in the ground state when possible. (alias: ``'alap'``)
            If ``None``, no scheduling will be done. This can also be the external plugin name
            to use for the ``scheduling`` stage. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"scheduling"`` for the ``stage_name``
            argument.
        dt: Backend sample time (resolution) in seconds.
            If ``None`` (default), ``backend.dt`` is used.
        approximation_degree: Heuristic dial used for circuit approximation, where ``1.0`` means
            no approximation (up to numerical tolerance) and ``0.0`` means the maximum approximation.
            If ``target`` is available, a value of ``None`` indicates that approximation is allowed up
            to the reported error rate for an operation in the target.
        seed_transpiler: Sets a seed for the PRNG used by the stochastic parts of the transpiler.
            This parameter takes precedence over the ``QISKIT_TRANSPILER_SEED`` environment
            variable and ``transpiler_seed`` setting in the user configuration file .
        optimization_level: How much optimization to perform on the circuits.
            Higher levels generate more optimized circuits,
            at the expense of longer transpilation time.

            * 0: no optimization
            * 1: light optimization
            * 2: heavy optimization
            * 3: even heavier optimization

            If ``None``, level 2 will be chosen as default.
        callback: A callback function that will be called after each
            pass execution. The function will be called with 5 keyword
            arguments,
            | ``pass_``: the pass being run.
            | ``dag``: the dag output of the pass.
            | ``time``: the time to execute the pass.
            | ``property_set``: the property set.
            | ``count``: the index for the pass execution.
            The exact arguments passed expose the internals of the pass manager,
            and are subject to change as the pass manager internals change. If
            you intend to reuse a callback function over multiple releases, be
            sure to check that the arguments being passed are the same.
            To use the callback feature, define a function that will
            take in kwargs dict and access the variables. For example::

                def callback_func(**kwargs):
                    pass_ = kwargs['pass_']
                    dag = kwargs['dag']
                    time = kwargs['time']
                    property_set = kwargs['property_set']
                    count = kwargs['count']
                    ...
                transpile(circ, callback=callback_func)

        output_name: A list with strings to identify the output circuits. The length of
            the list should be exactly the length of the ``circuits`` parameter.
        unitary_synthesis_method (str): The name of the unitary synthesis
            method to use. By default ``'default'`` is used. You can see a list of installed
            plugins with :func:`.unitary_synthesis_plugin_names`.
        unitary_synthesis_plugin_config: An optional configuration dictionary
            that will be passed directly to the unitary synthesis plugin. By
            default this setting will have no effect as the default unitary
            synthesis method does not take custom configuration. This should
            only be necessary when a unitary synthesis plugin is specified with
            the ``unitary_synthesis_method`` argument. As this is custom for each
            unitary synthesis plugin refer to the plugin documentation for how
            to use this option.
        target: A backend transpiler target. Normally this is specified as part of
            the ``backend`` argument, but if you have manually constructed a
            :class:`~qiskit.transpiler.Target` object you can specify it manually here.
            This will override the target from ``backend``.
        hls_config: An optional configuration class
            :class:`~qiskit.transpiler.passes.synthesis.HLSConfig` that will be passed directly
            to :class:`~qiskit.transpiler.passes.synthesis.HighLevelSynthesis` transformation pass.
            This configuration class allows specifying the lists of synthesis algorithms and
            their parameters for various high-level objects.
        init_method: The plugin name to use for the ``init`` stage. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"init"`` for the stage
            name argument.
        optimization_method: The plugin name to use for the
            ``optimization`` stage. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"optimization"`` for the
            ``stage_name`` argument.
        ignore_backend_supplied_default_methods: If set to ``True`` any default methods specified by
            a backend will be ignored. Some backends specify alternative default methods
            to support custom compilation target-specific passes/plugins which support
            backend-specific compilation techniques. If you'd prefer that these defaults were
            not used this option is used to disable those backend-specific defaults.
        num_processes: The maximum number of parallel processes to launch for this call to
            transpile if parallel execution is enabled. This argument overrides
            ``num_processes`` in the user configuration file, and the ``QISKIT_NUM_PROCS``
            environment variable. If set to ``None`` the system default or local user configuration
            will be used.
        qubits_initially_zero: Indicates whether the input circuit is zero-initialized.

    Returns:
        The transpiled circuit(s).

    Raises:
        TranspilerError: in case of bad inputs to transpiler (like conflicting parameters)
            or errors in passes
        TypeError: if a :class:`~qiskit.transpiler.Target` is passed as the ``backend``
            positional argument and ``target`` is also specified as a keyword argument.
    """
    arg_circuits_list = isinstance(circuits, list)
    circuits = circuits if arg_circuits_list else [circuits]

    if not circuits:
        return []

    start_time = time()

    if isinstance(backend, Target):
        if target is not None:
            raise TypeError(
                "A 'Target' was passed as the 'backend' positional argument, but 'target' "
                "was also specified as a keyword argument. Please use only one of the two."
            )
        target = backend
        backend = None

    if optimization_level is None:
        # Take optimization level from the configuration or 2 as default.
        config = user_config.get_config()
        optimization_level = config.get("transpile_optimization_level", 2)

    if seed_transpiler is None:
        if (seed := os.getenv("QISKIT_TRANSPILER_SEED", None)) is not None:
            seed_transpiler = int(seed)
        else:
            config = user_config.get_config()
            seed_transpiler = config.get("transpiler_seed", None)

    if not ignore_backend_supplied_default_methods:
        if scheduling_method is None and hasattr(backend, "get_scheduling_stage_plugin"):
            scheduling_method = backend.get_scheduling_stage_plugin()
        if translation_method is None and hasattr(backend, "get_translation_stage_plugin"):
            translation_method = backend.get_translation_stage_plugin()

    output_name = _parse_output_name(output_name, circuits)
    coupling_map = _parse_coupling_map(coupling_map)
    _check_circuits_coupling_map(circuits, coupling_map, backend)

    # Edge cases require using the old model (loose constraints) instead of building a target,
    # but we don't populate the passmanager config with loose constraints unless it's one of
    # the known edge cases to control the execution path.
    pm = generate_preset_pass_manager(
        optimization_level,
        target=target,
        backend=backend,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        initial_layout=initial_layout,
        layout_method=layout_method,
        routing_method=routing_method,
        translation_method=translation_method,
        scheduling_method=scheduling_method,
        approximation_degree=approximation_degree,
        seed_transpiler=seed_transpiler,
        unitary_synthesis_method=unitary_synthesis_method,
        unitary_synthesis_plugin_config=unitary_synthesis_plugin_config,
        hls_config=hls_config,
        init_method=init_method,
        optimization_method=optimization_method,
        dt=dt,
        qubits_initially_zero=qubits_initially_zero,
    )

    out_circuits = pm.run(circuits, callback=callback, num_processes=num_processes)

    for name, circ in zip(output_name, out_circuits):
        circ.name = name
    end_time = time()
    _log_transpile_time(start_time, end_time)
    if arg_circuits_list:
        return out_circuits
    else:
        return out_circuits[0]


def _check_circuits_coupling_map(circuits, cmap, backend):
    # Check circuit width against number of qubits in coupling_map(s)
    max_qubits = None
    if cmap is not None:
        max_qubits = cmap.size()
    elif backend is not None:
        max_qubits = backend.num_qubits

    for circuit in circuits:
        # If coupling_map is not None or num_qubits == 1
        num_qubits = len(circuit.qubits)
        if max_qubits is not None and (num_qubits > max_qubits):
            raise CircuitTooWideForTarget(
                f"Number of qubits ({num_qubits}) in {circuit.name} "
                f"is greater than maximum ({max_qubits}) in the coupling_map"
            )


def _log_transpile_time(start_time, end_time):
    log_msg = f"Total Transpile Time - {((end_time - start_time) * 1000):.5f} (ms)"
    logger.info(log_msg)


def _parse_coupling_map(coupling_map):
    # coupling_map could be None, or a list of lists, e.g. [[0, 1], [2, 1]]
    if isinstance(coupling_map, list) and all(
        isinstance(i, list) and len(i) == 2 for i in coupling_map
    ):
        return CouplingMap(coupling_map)
    elif isinstance(coupling_map, list):
        raise TranspilerError(
            "Only a single input coupling map can be used with transpile() if you need to "
            "target different coupling maps for different circuits you must call transpile() "
            "multiple times"
        )
    else:
        return coupling_map


def _parse_output_name(output_name, circuits):
    # naming and returning circuits
    # output_name could be either a string or a list
    if output_name is not None:
        if isinstance(output_name, str):
            # single circuit
            if len(circuits) == 1:
                return [output_name]
            # multiple circuits
            else:
                raise TranspilerError(
                    "Expected a list object of length equal "
                    + "to that of the number of circuits "
                    + "being transpiled"
                )
        elif isinstance(output_name, list):
            if len(circuits) == len(output_name) and all(
                isinstance(name, str) for name in output_name
            ):
                return output_name
            else:
                raise TranspilerError(
                    "The length of output_name list "
                    "must be equal to the number of "
                    "transpiled circuits and the output_name "
                    "list should be strings."
                )
        else:
            raise TranspilerError(
                "The parameter output_name should be a string or a "
                f"list of strings: {type(output_name)} was used."
            )
    else:
        return [circuit.name for circuit in circuits]
