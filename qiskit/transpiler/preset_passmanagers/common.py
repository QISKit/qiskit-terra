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


"""Common preset passmanager generators."""

from __future__ import annotations

import collections
import typing

from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES

from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes import Error
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.transpiler.passes import Collect1qRuns
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes import HighLevelSynthesis
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import GateDirection
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import CheckGateDirection
from qiskit.transpiler.passes import TimeUnitConversion
from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit.transpiler.passes import ASAPScheduleAnalysis
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import FilterOpNodes
from qiskit.transpiler.passes import PadDelay
from qiskit.transpiler.passes import InstructionDurationCheck
from qiskit.transpiler.passes import ConstrainedReschedule
from qiskit.transpiler.passes import ContainsInstruction
from qiskit.transpiler.passes import VF2PostLayout
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayoutStopReason
from qiskit.transpiler.passes import WrapAngles
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.optimization_metric import OptimizationMetric
from qiskit.utils import deprecate_func
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _CLIFFORD_GATE_NAMES


if typing.TYPE_CHECKING:
    from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
    from qiskit.transpiler.target import Target
    from qiskit.transpiler.coupling import CouplingMap

_ControlFlowState = collections.namedtuple("_ControlFlowState", ("working", "not_working"))

# Any method neither known good nor known bad (i.e. not a Terra-internal pass) is passed through
# without error, since it is being supplied by a plugin and we don't have any knowledge of these.
_CONTROL_FLOW_STATES = {
    "layout_method": _ControlFlowState(
        working={"default", "trivial", "dense", "sabre"}, not_working=set()
    ),
    "routing_method": _ControlFlowState(
        working={"default", "none", "stochastic", "sabre"}, not_working={"lookahead", "basic"}
    ),
    "translation_method": _ControlFlowState(
        working={"default", "translator", "synthesis"},
        not_working=set(),
    ),
    "optimization_method": _ControlFlowState(working={"default"}, not_working=set()),
    "scheduling_method": _ControlFlowState(working=set(), not_working={"alap", "asap"}),
}


def _has_control_flow(property_set):
    return any(property_set[f"contains_{x}"] for x in CONTROL_FLOW_OP_NAMES)


def _without_control_flow(property_set):
    return not any(property_set[f"contains_{x}"] for x in CONTROL_FLOW_OP_NAMES)


class _InvalidControlFlowForBackend:
    # Explicitly stateful closure to allow pickling.

    def __init__(self, basis_gates=(), target=None):
        if target is not None and len(target.operation_names) > 0:
            self.unsupported = [op for op in CONTROL_FLOW_OP_NAMES if op not in target]
        elif basis_gates is not None and len(basis_gates) > 0:
            basis_gates = set(basis_gates)
            self.unsupported = [op for op in CONTROL_FLOW_OP_NAMES if op not in basis_gates]
        else:
            # Pass manager without basis gates or target; assume everything's valid.
            self.unsupported = []

    def message(self, property_set):
        """Create an error message for the given property set."""
        fails = [x for x in self.unsupported if property_set[f"contains_{x}"]]
        if len(fails) == 1:
            return f"The control-flow construct '{fails[0]}' is not supported by the backend."
        return (
            f"The control-flow constructs [{', '.join(repr(op) for op in fails)}]"
            " are not supported by the backend."
        )

    def condition(self, property_set):
        """Checkable condition for the given property set."""
        return any(property_set[f"contains_{x}"] for x in self.unsupported)


def generate_control_flow_options_check(
    layout_method=None,
    routing_method=None,
    translation_method=None,
    optimization_method=None,
    scheduling_method=None,
    basis_gates=(),
    target=None,
):
    """Generate a pass manager that, when run on a DAG that contains control flow, fails with an
    error message explaining the invalid options, and what could be used instead.

    Returns:
        PassManager: a pass manager that populates the ``contains_x`` properties for each of the
        control-flow operations, and raises an error if any of the given options do not support
        control flow, but a circuit with control flow is given.
    """
    bad_options = []
    message = "Some options cannot be used with control flow."
    for stage, given in [
        ("layout", layout_method),
        ("routing", routing_method),
        ("translation", translation_method),
        ("optimization", optimization_method),
        ("scheduling", scheduling_method),
    ]:
        option = stage + "_method"
        method_states = _CONTROL_FLOW_STATES[option]
        if given is not None and given in method_states.not_working:
            if method_states.working:
                message += (
                    f" Got {option}='{given}', but valid values are {list(method_states.working)}."
                )
            else:
                message += (
                    f" Got {option}='{given}', but the entire {stage} stage is not supported."
                )
            bad_options.append(option)
    out = PassManager()
    out.append(ContainsInstruction(CONTROL_FLOW_OP_NAMES, recurse=False))
    if bad_options:
        out.append(ConditionalController(Error(message), condition=_has_control_flow))
    backend_control = _InvalidControlFlowForBackend(basis_gates, target)
    out.append(
        ConditionalController(Error(backend_control.message), condition=backend_control.condition)
    )
    return out


def generate_error_on_control_flow(message):
    """Get a pass manager that always raises an error if control flow is present in a given
    circuit."""
    out = PassManager()
    out.append(ContainsInstruction(CONTROL_FLOW_OP_NAMES, recurse=False))
    out.append(ConditionalController(Error(message), condition=_has_control_flow))
    return out


def if_has_control_flow_else(if_present, if_absent):
    """Generate a pass manager that will run the passes in ``if_present`` if the given circuit
    has control-flow operations in it, and those in ``if_absent`` if it doesn't."""
    if isinstance(if_present, PassManager):
        if_present = if_present.to_flow_controller()
    if isinstance(if_absent, PassManager):
        if_absent = if_absent.to_flow_controller()
    out = PassManager()
    out.append(ContainsInstruction(CONTROL_FLOW_OP_NAMES, recurse=False))
    out.append(ConditionalController(if_present, condition=_has_control_flow))
    out.append(ConditionalController(if_absent, condition=_without_control_flow))
    return out


def generate_unroll_3q(
    target: Target | None,
    basis_gates: list[str] | None = None,
    approximation_degree: float | None = None,
    unitary_synthesis_method: str = "default",
    unitary_synthesis_plugin_config: dict | None = None,
    hls_config: HLSConfig | None = None,
    qubits_initially_zero: bool = True,
    optimization_metric: OptimizationMetric = OptimizationMetric.COUNT_2Q,
    optimization_level: int = 2,
):
    """Generate an unroll >3q :class:`~qiskit.transpiler.PassManager`

    Args:
        target: the :class:`~.Target` object representing the backend
        basis_gates: A list of str gate names that represent the basis
            gates on the backend target.
        approximation_degree: Heuristic dial used for circuit approximation, where
            ``1.0`` means no approximation (up to numerical tolerance) and ``0.0``
            means the maximum approximation. If ``target`` is available, a value of
            ``None`` indicates that approximation is allowed up to the reported error
            rate for an operation in the target.
        unitary_synthesis_method (str): The unitary synthesis method to use. You can see
            a list of installed plugins with :func:`.unitary_synthesis_plugin_names`.
        unitary_synthesis_plugin_config: The optional dictionary plugin
            configuration, this is plugin specific refer to the specified plugin's
            documentation for how to use.
        hls_config: An optional configuration class to use for
            :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass.
            Specifies how to synthesize various high-level objects.
        qubits_initially_zero: Indicates whether the input circuit is
            zero-initialized.
        optimization_metric: the :class:`~.OptimizationMetric` object
            that defines the metric used when optimizing the unrolling.
        optimization_level: The optimization level to use for unrolling, in the case
            that multiple unrolling algorithms are available. Higher
            levels generate potentially more optimized circuits, at the expense
            of longer transpilation time.

    Returns:
        PassManager: The unroll 3q or more pass manager
    """
    unroll_3q = PassManager()
    unroll_3q.append(
        UnitarySynthesis(
            basis_gates,
            approximation_degree=approximation_degree,
            method=unitary_synthesis_method,
            min_qubits=3,
            plugin_config=unitary_synthesis_plugin_config,
            target=target,
        )
    )
    unroll_3q.append(
        HighLevelSynthesis(
            hls_config=hls_config,
            coupling_map=None,
            target=target,
            use_qubit_indices=False,
            equivalence_library=sel,
            basis_gates=basis_gates,
            min_qubits=3,
            qubits_initially_zero=qubits_initially_zero,
            optimization_metric=optimization_metric,
            optimization_level=optimization_level,
        )
    )
    # If there are no target instructions revert to using unroll3qormore so
    # routing works.
    if basis_gates is None and target is None:
        unroll_3q.append(Unroll3qOrMore(target, basis_gates))
    else:
        unroll_3q.append(BasisTranslator(sel, basis_gates, target=target, min_qubits=3))
    return unroll_3q


def generate_embed_passmanager(coupling_map):
    """Generate a layout embedding :class:`~qiskit.transpiler.PassManager`

    This is used to generate a :class:`~qiskit.transpiler.PassManager` object
    that can be used to expand and apply an initial layout to a circuit

    Args:
        coupling_map (Union[CouplingMap, Target]): The coupling map for the backend to embed
            the circuit to.
    Returns:
        PassManager: The embedding passmanager that assumes the layout property
            set has been set in earlier stages
    """
    return PassManager([FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()])


def _layout_not_perfect(property_set):
    """Return ``True`` if the first attempt at layout has been checked and found to be imperfect.
    In this case, perfection means "does not require any swap routing"."""
    return property_set["is_swap_mapped"] is not None and not property_set["is_swap_mapped"]


def _apply_post_layout_condition(property_set):
    # if VF2 Post layout found a solution we need to re-apply the better
    # layout. Otherwise we can skip apply layout.
    return (
        property_set["VF2PostLayout_stop_reason"] is not None
        and property_set["VF2PostLayout_stop_reason"] is VF2PostLayoutStopReason.SOLUTION_FOUND
    )


def generate_routing_passmanager(
    routing_pass,
    target,
    coupling_map=None,
    vf2_call_limit=None,
    seed_transpiler=-1,
    check_trivial=False,
    use_barrier_before_measurement=True,
    vf2_max_trials=None,
):
    """Generate a routing :class:`~qiskit.transpiler.PassManager`

    Args:
        routing_pass (TransformationPass): The pass which will perform the
            routing
        target (Target): the :class:`~.Target` object representing the backend
        coupling_map (CouplingMap): The coupling map of the backend to route
            for
        vf2_call_limit (int): The internal call limit for the vf2 post layout
            pass. If this is ``None`` or ``0`` the vf2 post layout will not be
            run.
        seed_transpiler (int): Sets random seed for the stochastic parts of
            the transpiler. This is currently only used for :class:`.VF2PostLayout` and the
            default value of ``-1`` is strongly recommended (which is no randomization).
            If a value of ``None`` is provided this will seed from system
            entropy.
        check_trivial (bool): If set to true this will condition running the
            :class:`~.VF2PostLayout` pass after routing on whether a trivial
            layout was tried and was found to not be perfect. This is only
            needed if the constructed pass manager runs :class:`~.TrivialLayout`
            as a first layout attempt and uses it if it's a perfect layout
            (as is the case with preset pass manager level 1).
        use_barrier_before_measurement (bool): If true (the default) the
            :class:`~.BarrierBeforeFinalMeasurements` transpiler pass will be run prior to the
            specified pass in the ``routing_pass`` argument.
        vf2_max_trials (int): The maximum number of trials to run VF2 when
            evaluating the vf2 post layout
            pass. If this is ``None`` or ``0`` the vf2 post layout will not be run.
    Returns:
        PassManager: The routing pass manager
    """

    def _run_post_layout_condition(property_set):
        # If we check trivial layout and the found trivial layout was not perfect also
        # ensure VF2 initial layout was not used before running vf2 post layout
        if not check_trivial or _layout_not_perfect(property_set):
            vf2_stop_reason = property_set["VF2Layout_stop_reason"]
            if vf2_stop_reason is None or vf2_stop_reason != VF2LayoutStopReason.SOLUTION_FOUND:
                return True
        return False

    routing = PassManager()
    if target is not None:
        routing.append(CheckMap(target, property_set_field="routing_not_needed"))
    else:
        routing.append(CheckMap(coupling_map, property_set_field="routing_not_needed"))

    def _swap_condition(property_set):
        return not property_set["routing_not_needed"]

    if use_barrier_before_measurement:
        routing.append(
            ConditionalController(
                [
                    BarrierBeforeFinalMeasurements(
                        label="qiskit.transpiler.internal.routing.protection.barrier"
                    ),
                    routing_pass,
                ],
                condition=_swap_condition,
            )
        )
    else:
        routing.append(ConditionalController(routing_pass, condition=_swap_condition))

    is_vf2_fully_bounded = vf2_call_limit and vf2_max_trials
    if target is not None and is_vf2_fully_bounded:
        routing.append(
            ConditionalController(
                VF2PostLayout(
                    target,
                    seed=seed_transpiler,
                    call_limit=vf2_call_limit,
                    max_trials=vf2_max_trials,
                    strict_direction=False,
                ),
                condition=_run_post_layout_condition,
            )
        )
        routing.append(ConditionalController(ApplyLayout(), condition=_apply_post_layout_condition))

    def filter_fn(node):
        return node.label != "qiskit.transpiler.internal.routing.protection.barrier"

    routing.append([FilterOpNodes(filter_fn)])

    return routing


@deprecate_func(
    since="2.0",
    additional_msg=(
        "Translation plugins are now required to respect ISA directionality,"
        " so typically no replacement is necessary."
    ),
    removal_timeline="in Qiskit 3.0",
)
def generate_pre_op_passmanager(target=None, coupling_map=None, remove_reset_in_zero=False):
    """Generate a pre-optimization loop :class:`~qiskit.transpiler.PassManager`

    This pass manager will check to ensure that directionality from the coupling
    map is respected.

    Args:
        target (Target): the :class:`~.Target` object representing the backend
        coupling_map (CouplingMap): The coupling map to use
        remove_reset_in_zero (bool): If ``True`` include the remove reset in
            zero pass in the generated PassManager
    Returns:
        PassManager: The pass manager

    """
    pre_opt = PassManager()
    if coupling_map:
        pre_opt.append(CheckGateDirection(coupling_map, target=target))

        def _direction_condition(property_set):
            return not property_set["is_direction_mapped"]

        pre_opt.append(
            ConditionalController(
                [GateDirection(coupling_map, target=target)],
                condition=_direction_condition,
            )
        )
    if remove_reset_in_zero:
        pre_opt.append(RemoveResetInZeroState())
    return pre_opt


def generate_translation_passmanager(
    target: Target | None,
    basis_gates: list[str] | None = None,
    method: str = "translator",
    approximation_degree: float | None = None,
    coupling_map: CouplingMap | None = None,
    unitary_synthesis_method: str = "default",
    unitary_synthesis_plugin_config: dict | None = None,
    hls_config: HLSConfig | None = None,
    qubits_initially_zero: bool = True,
):
    """Generate a basis translation :class:`~qiskit.transpiler.PassManager`

    Args:
        target: the :class:`~.Target` object representing the backend
        basis_gates: A list of str gate names that represent the basis
            gates on the backend target
        method: The basis translation method to use
        approximation_degree: Heuristic dial used for circuit approximation, where
            ``1.0`` means no approximation (up to numerical tolerance) and ``0.0``
            means the maximum approximation. If ``target`` is available, a value of
            ``None`` indicates that approximation is allowed up
            to the reported error rate for an operation in the target.
        coupling_map: the coupling map of the backend
            in case synthesis is done on a physical circuit. The
            directionality of the coupling_map will be taken into
            account if pulse_optimize is True/None and natural_direction
            is True/None.
        unitary_synthesis_method: The unitary synthesis method to use. You can
            see a list of installed plugins with :func:`.unitary_synthesis_plugin_names`.
        unitary_synthesis_plugin_config: The optional dictionary plugin
            configuration, this is plugin specific refer to the specified plugin's
            documentation for how to use.
        hls_config: An optional configuration class to use for
            :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass.
            Specifies how to synthesize various high-level objects.
        qubits_initially_zero: Indicates whether the input circuit is
            zero-initialized.

    Returns:
        PassManager: The basis translation pass manager

    Raises:
        TranspilerError: If the ``method`` kwarg is not a valid value
    """
    if basis_gates is None and target is None:
        return PassManager([])

    if method == "translator":
        translator = BasisTranslator(sel, basis_gates, target)
        unroll = [
            # Use unitary synthesis for basis aware decomposition of
            # UnitaryGates before custom unrolling
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                plugin_config=unitary_synthesis_plugin_config,
                method=unitary_synthesis_method,
                target=target,
            ),
            HighLevelSynthesis(
                hls_config=hls_config,
                coupling_map=coupling_map,
                target=target,
                use_qubit_indices=True,
                equivalence_library=sel,
                basis_gates=basis_gates,
                qubits_initially_zero=qubits_initially_zero,
            ),
            translator,
        ]
        fix_1q = [translator]
    elif method == "synthesis":
        unroll = [
            # # Use unitary synthesis for basis aware decomposition of
            # UnitaryGates > 2q before collection
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                plugin_config=unitary_synthesis_plugin_config,
                method=unitary_synthesis_method,
                min_qubits=3,
                target=target,
            ),
            HighLevelSynthesis(
                hls_config=hls_config,
                coupling_map=coupling_map,
                target=target,
                use_qubit_indices=True,
                basis_gates=basis_gates,
                min_qubits=3,
                qubits_initially_zero=qubits_initially_zero,
            ),
            Unroll3qOrMore(target=target, basis_gates=basis_gates),
            Collect2qBlocks(),
            Collect1qRuns(),
            ConsolidateBlocks(
                basis_gates=basis_gates, target=target, approximation_degree=approximation_degree
            ),
            UnitarySynthesis(
                basis_gates=basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                plugin_config=unitary_synthesis_plugin_config,
                method=unitary_synthesis_method,
                target=target,
            ),
            HighLevelSynthesis(
                hls_config=hls_config,
                coupling_map=coupling_map,
                target=target,
                use_qubit_indices=True,
                basis_gates=basis_gates,
                qubits_initially_zero=qubits_initially_zero,
            ),
        ]
        fix_1q = [
            Collect1qRuns(),
            ConsolidateBlocks(
                basis_gates=basis_gates, target=target, approximation_degree=approximation_degree
            ),
            UnitarySynthesis(
                basis_gates=basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                plugin_config=unitary_synthesis_plugin_config,
                method=unitary_synthesis_method,
                target=target,
            ),
        ]
    else:
        raise TranspilerError(f"Invalid translation method {method}.")
    # Our built-ins don't 100% guarantee that 2q gate direction is respected, so we might need to
    # run a little bit of fix up on them.  `GateDirection` doesn't guarantee that 1q gates are
    # ISA safe after it runs, so we need another run too.
    if (coupling_map and not coupling_map.is_symmetric) or (
        target is not None and target.get_non_global_operation_names(strict_direction=True)
    ):
        unroll.append(CheckGateDirection(coupling_map, target=target))

        def _direction_condition(property_set):
            return not property_set["is_direction_mapped"]

        unroll.append(
            ConditionalController(
                [GateDirection(coupling_map, target=target)] + fix_1q,
                condition=_direction_condition,
            )
        )
    if target is not None and target.has_angle_bounds():
        unroll.append(WrapAngles(target))
    return PassManager(unroll)


def generate_scheduling(instruction_durations, scheduling_method, timing_constraints, target=None):
    """Generate a post optimization scheduling :class:`~qiskit.transpiler.PassManager`

    Args:
        instruction_durations (dict): The dictionary of instruction durations
        scheduling_method (str): The scheduling method to use, can either be
            ``'asap'``/``'as_soon_as_possible'`` or
            ``'alap'``/``'as_late_as_possible'``
        timing_constraints (TimingConstraints): Hardware time alignment restrictions.
        target (Target): The :class:`~.Target` object representing the backend

    Returns:
        PassManager: The scheduling pass manager

    Raises:
        TranspilerError: If the ``scheduling_method`` kwarg is not a valid value
    """
    scheduling = PassManager()
    if scheduling_method:
        # Do scheduling after unit conversion.
        scheduler = {
            "alap": ALAPScheduleAnalysis,
            "as_late_as_possible": ALAPScheduleAnalysis,
            "asap": ASAPScheduleAnalysis,
            "as_soon_as_possible": ASAPScheduleAnalysis,
        }
        scheduling.append(TimeUnitConversion(instruction_durations, target=target))
        try:
            scheduling.append(scheduler[scheduling_method](instruction_durations, target=target))
        except KeyError as ex:
            raise TranspilerError(f"Invalid scheduling method {scheduling_method}.") from ex
    elif instruction_durations:
        # No scheduling. But do unit conversion for delays.
        def _contains_delay(property_set):
            return property_set["contains_delay"]

        scheduling.append(ContainsInstruction("delay"))
        scheduling.append(
            ConditionalController(
                TimeUnitConversion(instruction_durations, target=target), condition=_contains_delay
            )
        )
    if (
        timing_constraints.granularity != 1
        or timing_constraints.min_length != 1
        or timing_constraints.acquire_alignment != 1
        or timing_constraints.pulse_alignment != 1
    ):
        # Run alignment analysis regardless of scheduling.

        def _require_alignment(property_set):
            return property_set["reschedule_required"]

        scheduling.append(
            InstructionDurationCheck(
                acquire_alignment=timing_constraints.acquire_alignment,
                pulse_alignment=timing_constraints.pulse_alignment,
                target=target,
            )
        )
        scheduling.append(
            ConditionalController(
                ConstrainedReschedule(
                    acquire_alignment=timing_constraints.acquire_alignment,
                    pulse_alignment=timing_constraints.pulse_alignment,
                    target=target,
                ),
                condition=_require_alignment,
            )
        )
    if scheduling_method:
        # Call padding pass if circuit is scheduled
        scheduling.append(PadDelay(target=target, durations=instruction_durations))

    return scheduling


VF2Limits = collections.namedtuple("VF2Limits", ("call_limit", "max_trials"))


def get_vf2_limits(
    optimization_level: int,
    layout_method: str | None = None,
    initial_layout: Layout | None = None,
    exact_match: bool = False,
) -> VF2Limits:
    """Get the VF2 limits for VF2-based layout passes.

    Returns:
        VF2Limits: A namedtuple with optional elements
        ``call_limit`` and ``max_trials``.
    """
    limits = VF2Limits(None, None)
    if layout_method is None and initial_layout is None:
        if optimization_level in {1, 2}:
            limits = VF2Limits(
                50_000,  # Set call limit to ~100ms with rustworkx 0.10.2
                2_500,  # Limits layout scoring to < 600ms on ~400 qubit devices
            )
        elif optimization_level == 3:
            limits = VF2Limits(
                30_000_000,  # Set call limit to ~60 sec with rustworkx 0.10.2
                250_000,  # Limits layout scoring to < 60 sec on ~400 qubit devices
            )
        # In Qiskit 2.2, strict mode still includes heavy Python usage for the semantics and
        # scoring, so we dial the limits way down.
        if exact_match:
            limits = VF2Limits(
                ((limits.call_limit // 100) or 1) if limits.call_limit is not None else None,
                ((limits.max_trials // 100) or 1) if limits.max_trials is not None else None,
            )
    return limits


# Clifford+T basis, consisting of Clifford+T gate names + additional instruction names
# that are a part of every basis
CLIFFORD_T_BASIS = set(_CLIFFORD_GATE_NAMES).union(
    {"t", "tdg", "delay", "barrier", "reset", "measure"}.union(CONTROL_FLOW_OP_NAMES)
)


def is_clifford_t_basis(basis_gates=None, target=None) -> bool:
    """
    Checks whether the given basis set can be considered as Clifford+T.

    For this we require that:
    1. The set only contains Clifford+T gates,
    2. The set contains either T or Tdg gate or both.

    In particular, these conditions guarantee that the empty basis set
    is not considered as Clifford+T.
    """

    if target is not None:
        basis = set(target.operation_names)
    elif basis_gates is not None:
        basis = set(basis_gates)
    else:
        basis = set()

    if ("t" not in basis) and ("tdg" not in basis):
        return False

    return basis.issubset(CLIFFORD_T_BASIS)
