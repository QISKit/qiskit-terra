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

"""Built-in transpiler stage plugins for preset pass managers."""

import os

from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayout
from qiskit.transpiler.passes.optimization.split_2q_unitaries import Split2QUnitaries
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import Error
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import VF2Layout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import ElidePermutations
from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure
from qiskit.transpiler.passes import CommutativeOptimization
from qiskit.transpiler.passes import TwoQubitPeepholeOptimization
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import SynthesizeRZRotations
from qiskit.transpiler.passes import OptimizeCliffordT
from qiskit.transpiler.passes import SubstitutePi4Rotations
from qiskit.transpiler.passes import Collect1qRuns
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import GateDirection
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePlugin,
    PassManagerStagePluginManager,
    PassManagerCliffordTStagePlugin,
)
from qiskit.transpiler.passmanager_config import PassManagerCliffordTConfig
from qiskit.transpiler.passes.optimization import (
    Optimize1qGatesDecomposition,
    CommutativeCancellation,
    ConsolidateBlocks,
    InverseCancellation,
    RemoveIdentityEquivalent,
    ContractIdleWiresInControlFlow,
)
from qiskit.transpiler.optimization_metric import OptimizationMetric
from qiskit.transpiler.passes import Depth, Size, FixedPoint, MinimumPoint, CountOps
from qiskit.transpiler.passes.utils.gates_basis import GatesInBasis
from qiskit.transpiler.passes.synthesis.unitary_synthesis import UnitarySynthesis
from qiskit.passmanager.flow_controllers import ConditionalController, DoWhileController
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.quantum_info.operators.symplectic.clifford_circuits import get_clifford_gate_names
from qiskit.utils import default_num_processes
from qiskit import user_config

CONFIG = user_config.get_config()

_discrete_skipped_ops = {
    "delay",
    "reset",
    "measure",
    "switch_case",
    "if_else",
    "for_loop",
    "while_loop",
}


class DefaultInitPassManager(PassManagerStagePlugin):
    """Plugin class for default init stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        optimization_metric = OptimizationMetric.COUNT_2Q

        if optimization_level == 0:
            init = None
            if (
                pass_manager_config.initial_layout
                or pass_manager_config.coupling_map
                or (
                    pass_manager_config.target is not None
                    and pass_manager_config.target.build_coupling_map() is not None
                )
            ):
                init = common.generate_unroll_3q(
                    pass_manager_config.target,
                    pass_manager_config.basis_gates,
                    pass_manager_config.approximation_degree,
                    pass_manager_config.unitary_synthesis_method,
                    pass_manager_config.unitary_synthesis_plugin_config,
                    pass_manager_config.hls_config,
                    pass_manager_config.qubits_initially_zero,
                    optimization_metric,
                )
        elif optimization_level == 1:
            init = PassManager()
            if (
                pass_manager_config.initial_layout
                or pass_manager_config.coupling_map
                or (
                    pass_manager_config.target is not None
                    and pass_manager_config.target.build_coupling_map() is not None
                )
            ):
                init += common.generate_unroll_3q(
                    pass_manager_config.target,
                    pass_manager_config.basis_gates,
                    pass_manager_config.approximation_degree,
                    pass_manager_config.unitary_synthesis_method,
                    pass_manager_config.unitary_synthesis_plugin_config,
                    pass_manager_config.hls_config,
                    pass_manager_config.qubits_initially_zero,
                    optimization_metric,
                )
            init.append(
                [
                    InverseCancellation(),
                    ContractIdleWiresInControlFlow(),
                ]
            )

        elif optimization_level in {2, 3}:
            init = common.generate_unroll_3q(
                pass_manager_config.target,
                pass_manager_config.basis_gates,
                pass_manager_config.approximation_degree,
                pass_manager_config.unitary_synthesis_method,
                pass_manager_config.unitary_synthesis_plugin_config,
                pass_manager_config.hls_config,
                pass_manager_config.qubits_initially_zero,
                optimization_metric,
            )
            if pass_manager_config.routing_method != "none":
                init.append(ElidePermutations())
            init.append(
                [
                    RemoveDiagonalGatesBeforeMeasure(),
                    # Target not set on RemoveIdentityEquivalent because we haven't applied a Layout
                    # yet so doing anything relative to an error rate in the target is not valid.
                    RemoveIdentityEquivalent(
                        approximation_degree=pass_manager_config.approximation_degree
                    ),
                    InverseCancellation(),
                    ContractIdleWiresInControlFlow(),
                ]
            )
            init.append(CommutativeCancellation())
            init.append(ConsolidateBlocks())

            # If approximation degree is None that indicates a request to approximate up to the
            # error rates in the target. However, in the init stage we don't yet know the target
            # qubits being used to figure out the fidelity so just use the default fidelity parameter
            # in this case.
            split_2q_unitaries_swap = False
            if pass_manager_config.routing_method != "none":
                split_2q_unitaries_swap = True
            if pass_manager_config.approximation_degree is not None:
                init.append(
                    Split2QUnitaries(
                        pass_manager_config.approximation_degree, split_swap=split_2q_unitaries_swap
                    )
                )
            else:
                init.append(Split2QUnitaries(split_swap=split_2q_unitaries_swap))
        else:
            raise TranspilerError(f"Invalid optimization level {optimization_level}")
        return init


class DefaultTranslationPassManager(PassManagerStagePlugin):
    """Plugin class for the default-method translation stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        # For now, this is just a wrapper around the `BasisTranslator`.  It might expand in the
        # future if we want to change the default method to do more context-aware switching, or to
        # start transitioning the default method without breaking the semantics of the default
        # string referring to the `BasisTranslator`.

        return BasisTranslatorPassManager().pass_manager(pass_manager_config, optimization_level)


class BasisTranslatorPassManager(PassManagerStagePlugin):
    """Plugin class for translation stage with :class:`~.BasisTranslator`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        return common.generate_translation_passmanager(
            pass_manager_config.target,
            basis_gates=pass_manager_config.basis_gates,
            method="translator",
            approximation_degree=pass_manager_config.approximation_degree,
            coupling_map=pass_manager_config.coupling_map,
            unitary_synthesis_method=pass_manager_config.unitary_synthesis_method,
            unitary_synthesis_plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
            hls_config=pass_manager_config.hls_config,
            qubits_initially_zero=pass_manager_config.qubits_initially_zero,
        )


class UnitarySynthesisPassManager(PassManagerStagePlugin):
    """Plugin class for translation stage with :class:`~.UnitarySynthesis`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        return common.generate_translation_passmanager(
            pass_manager_config.target,
            basis_gates=pass_manager_config.basis_gates,
            method="synthesis",
            approximation_degree=pass_manager_config.approximation_degree,
            coupling_map=pass_manager_config.coupling_map,
            unitary_synthesis_method=pass_manager_config.unitary_synthesis_method,
            unitary_synthesis_plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
            hls_config=pass_manager_config.hls_config,
            qubits_initially_zero=pass_manager_config.qubits_initially_zero,
        )


class DefaultRoutingPassManager(PassManagerStagePlugin):
    """Plugin class for the "default" routing stage implementation."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        # The Sabre-based PM is the default implementation currently, but semantically the "default"
        # plugin has more scope to change its logic than one called "sabre".  In practice, we don't
        # run the actually `SabreSwap` logic from this pass most of the time, because we do that
        # during default layout; we're looking for the VF2PostLayout stuff mostly.
        return SabreSwapPassManager().pass_manager(pass_manager_config, optimization_level)


class BasicSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.BasicSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        if target is None:
            routing_pass = BasicSwap(coupling_map)
        else:
            routing_pass = BasicSwap(target)

        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 0:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                seed_transpiler=-1,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 2:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 3:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")


class LookaheadSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.LookaheadSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        coupling_map_routing = target
        if coupling_map_routing is None:
            coupling_map_routing = coupling_map
        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 0:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=2, search_width=2)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=4, search_width=4)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                seed_transpiler=-1,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 2:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=5, search_width=6)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 3:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=5, search_width=6)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")


class SabreSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.SabreSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        coupling_map_routing = target
        if coupling_map_routing is None:
            coupling_map_routing = coupling_map
        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 0:
            trial_count = _get_trial_count(5)
            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="basic",
                seed=seed_transpiler,
                trials=trial_count,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            trial_count = _get_trial_count(5)
            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="decay",
                seed=seed_transpiler,
                trials=trial_count,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                seed_transpiler=-1,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 2:
            trial_count = _get_trial_count(20)

            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="decay",
                seed=seed_transpiler,
                trials=trial_count,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 3:
            trial_count = _get_trial_count(20)
            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="decay",
                seed=seed_transpiler,
                trials=trial_count,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")


class NoneRoutingPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with error on routing."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        routing_pass = Error(
            msg="No routing method selected, but circuit is not routed to device. "
            "CheckMap Error: {check_map_msg}",
            action="raise",
        )
        return common.generate_routing_passmanager(
            routing_pass,
            target,
            coupling_map=coupling_map,
            seed_transpiler=-1,
            use_barrier_before_measurement=True,
        )


def _optimization_check_fixed_point():
    def check(property_set):
        return not (property_set["depth_fixed_point"] and property_set["size_fixed_point"])

    setup = [Size(recurse=True), Depth(recurse=True), FixedPoint("size"), FixedPoint("depth")]
    return (setup, check)


def _optimization_check_fixed_point_clifford_t():
    def check(property_set):
        return not property_set["size_and_t_count_fixed_point"]

    def get_size_and_t_count(property_set):
        return (
            property_set["size"],
            property_set["count_ops"].get("t", 0) + property_set["count_ops"].get("tdg", 0),
        )

    setup = [
        CountOps(recurse=True),
        Size(recurse=True),
        FixedPoint("size_and_t_count", getter=get_size_and_t_count),
    ]
    return (setup, check)


def _optimization_check_fixed_point_clifford_rz():
    def check(property_set):
        return not property_set["size_and_rz_count_fixed_point"]

    def get_size_and_rz_count(property_set):
        return (property_set["size"], property_set["count_ops"].get("rz", 0))

    setup = [
        CountOps(recurse=True),
        Size(recurse=True),
        FixedPoint("size_and_rz_count", getter=get_size_and_rz_count),
    ]
    return (setup, check)


def _optimization_check_minimum_point(prefix: str):
    def check(property_set):
        return not property_set[f"{prefix}_minimum_point"]

    setup = [Size(recurse=True), Depth(recurse=True), MinimumPoint(["depth", "size"], prefix)]
    return (setup, check)


class OptimizationPassManager(PassManagerStagePlugin):
    """Plugin class for optimization stage"""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        """Build pass manager for optimization stage."""

        # Use the dedicated plugin for the Clifford+T basis when appropriate.
        match optimization_level:
            case 0:
                return None
            case 1:
                pre_loop = []
                loop = [
                    Optimize1qGatesDecomposition(
                        basis=pass_manager_config.basis_gates, target=pass_manager_config.target
                    ),
                    InverseCancellation(),
                    ContractIdleWiresInControlFlow(),
                ]
                post_loop = []
                loop_check, continue_loop = _optimization_check_fixed_point()
            case 2:
                pre_loop = [
                    TwoQubitPeepholeOptimization(
                        pass_manager_config.target,
                        approximation_degree=pass_manager_config.approximation_degree,
                    ),
                ]
                loop = [
                    RemoveIdentityEquivalent(
                        approximation_degree=pass_manager_config.approximation_degree,
                        target=pass_manager_config.target,
                    ),
                    Optimize1qGatesDecomposition(
                        basis=pass_manager_config.basis_gates, target=pass_manager_config.target
                    ),
                    CommutativeCancellation(target=pass_manager_config.target),
                    ContractIdleWiresInControlFlow(),
                ]
                post_loop = []
                loop_check, continue_loop = _optimization_check_fixed_point()
            case 3:
                pre_loop = []
                loop = [
                    TwoQubitPeepholeOptimization(
                        pass_manager_config.target,
                        approximation_degree=pass_manager_config.approximation_degree,
                    ),
                    RemoveIdentityEquivalent(
                        approximation_degree=pass_manager_config.approximation_degree,
                        target=pass_manager_config.target,
                    ),
                    Optimize1qGatesDecomposition(
                        basis=pass_manager_config.basis_gates, target=pass_manager_config.target
                    ),
                    CommutativeCancellation(target=pass_manager_config.target),
                    ContractIdleWiresInControlFlow(),
                ]
                post_loop = []
                if pass_manager_config.coupling_map and pass_manager_config.target is not None:
                    vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
                        optimization_level,
                        pass_manager_config.layout_method,
                        pass_manager_config.initial_layout,
                        exact_match=True,
                    )
                    if vf2_call_limit and vf2_max_trials:
                        post_loop += [
                            VF2PostLayout(
                                target=pass_manager_config.target,
                                seed=-1,
                                call_limit=vf2_call_limit,
                                max_trials=vf2_max_trials,
                                strict_direction=True,
                            ),
                            ConditionalController(
                                ApplyLayout(), condition=common._apply_post_layout_condition
                            ),
                        ]
                loop_check, continue_loop = _optimization_check_minimum_point("optimization_loop")
            case bad:
                raise TranspilerError(f"Invalid optimization_level: {bad}")

        # Obtain the translation method required for this pass to work
        translation = PassManagerStagePluginManager().get_passmanager_stage(
            "translation",
            pass_manager_config.translation_method or "default",
            pass_manager_config,
            optimization_level=optimization_level,
        )

        def should_unroll(property_set):
            return not property_set["all_gates_in_basis"]

        unroll = [
            GatesInBasis(pass_manager_config.basis_gates, target=pass_manager_config.target),
            ConditionalController(translation.to_flow_controller(), condition=should_unroll),
        ]

        optimization = PassManager()
        optimization.append(pre_loop + loop_check)
        optimization.append(DoWhileController(loop + unroll + loop_check, do_while=continue_loop))
        optimization.append(post_loop)
        return optimization


class AlapSchedulingPassManager(PassManagerStagePlugin):
    """Plugin class for alap scheduling stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build scheduling stage PassManager"""

        instruction_durations = pass_manager_config.instruction_durations
        scheduling_method = pass_manager_config.scheduling_method
        timing_constraints = pass_manager_config.timing_constraints
        target = pass_manager_config.target

        return common.generate_scheduling(
            instruction_durations, scheduling_method, timing_constraints, target
        )


class AsapSchedulingPassManager(PassManagerStagePlugin):
    """Plugin class for asap scheduling stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build scheduling stage PassManager"""

        instruction_durations = pass_manager_config.instruction_durations
        scheduling_method = pass_manager_config.scheduling_method
        timing_constraints = pass_manager_config.timing_constraints
        target = pass_manager_config.target

        return common.generate_scheduling(
            instruction_durations, scheduling_method, timing_constraints, target
        )


class DefaultSchedulingPassManager(PassManagerStagePlugin):
    """Plugin class for default scheduling stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build scheduling stage PassManager"""

        instruction_durations = pass_manager_config.instruction_durations
        scheduling_method = None
        timing_constraints = pass_manager_config.timing_constraints or TimingConstraints()
        target = pass_manager_config.target

        return common.generate_scheduling(
            instruction_durations, scheduling_method, timing_constraints, target
        )


class DefaultLayoutPassManager(PassManagerStagePlugin):
    """Plugin class for default layout stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        _given_layout = SetLayout(pass_manager_config.initial_layout)

        def _choose_layout_condition(property_set):
            return not property_set["layout"]

        def _layout_not_perfect(property_set):
            """Return ``True`` if the first attempt at layout has been checked and found to be
            imperfect.  In this case, perfection means "does not require any swap routing"."""
            return property_set["is_swap_mapped"] is not None and not property_set["is_swap_mapped"]

        def _vf2_match_not_found(property_set):
            # If a layout hasn't been set by the time we run vf2 layout we need to
            # run layout
            if property_set["layout"] is None:
                return True
            # if VF2 layout stopped for any reason other than solution found we need
            # to run layout since VF2 didn't converge.
            return (
                property_set["VF2Layout_stop_reason"] is not None
                and property_set["VF2Layout_stop_reason"] is not VF2LayoutStopReason.SOLUTION_FOUND
            )

        def _swap_mapped(property_set):
            return property_set["final_layout"] is None

        if pass_manager_config.target is None:
            coupling_map = pass_manager_config.coupling_map
        else:
            coupling_map = pass_manager_config.target

        layout = PassManager()
        layout.append(_given_layout)
        if optimization_level == 0:
            if coupling_map is not None:
                layout.append(
                    ConditionalController(
                        TrivialLayout(coupling_map), condition=_choose_layout_condition
                    )
                )
            layout += common.generate_embed_passmanager(coupling_map)
            return layout

        if coupling_map is None:
            # There's nothing to lay out onto.  We only need to embed the initial layout, if given.
            pass
        elif optimization_level == 1:
            layout.append(
                ConditionalController(
                    [TrivialLayout(coupling_map), CheckMap(coupling_map)],
                    condition=_choose_layout_condition,
                )
            )
            choose_layout_1 = VF2Layout(
                coupling_map=pass_manager_config.coupling_map,
                seed=-1,
                call_limit=(50_000, 1_000),
                target=pass_manager_config.target,
            )
            layout.append(ConditionalController(choose_layout_1, condition=_layout_not_perfect))

            trial_count = _get_trial_count(5)

            choose_layout_2 = SabreLayout(
                coupling_map,
                max_iterations=2,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method not in (None, "default", "sabre"),
            )
            layout.append(
                ConditionalController(
                    [
                        BarrierBeforeFinalMeasurements(
                            "qiskit.transpiler.internal.routing.protection.barrier"
                        ),
                        choose_layout_2,
                    ],
                    condition=_vf2_match_not_found,
                )
            )
        elif optimization_level == 2:
            choose_layout_0 = VF2Layout(
                coupling_map=pass_manager_config.coupling_map,
                seed=-1,
                call_limit=(5_000_000, 10_000),
                target=pass_manager_config.target,
            )
            layout.append(
                ConditionalController(choose_layout_0, condition=_choose_layout_condition)
            )

            trial_count = _get_trial_count(20)

            choose_layout_1 = SabreLayout(
                coupling_map,
                max_iterations=2,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method not in (None, "default", "sabre"),
            )
            layout.append(
                ConditionalController(
                    [
                        BarrierBeforeFinalMeasurements(
                            "qiskit.transpiler.internal.routing.protection.barrier"
                        ),
                        choose_layout_1,
                    ],
                    condition=_vf2_match_not_found,
                )
            )
        elif optimization_level == 3:
            choose_layout_0 = VF2Layout(
                coupling_map=pass_manager_config.coupling_map,
                seed=-1,
                call_limit=(30_000_000, 100_000),
                target=pass_manager_config.target,
            )
            layout.append(
                ConditionalController(choose_layout_0, condition=_choose_layout_condition)
            )

            trial_count = _get_trial_count(20)

            choose_layout_1 = SabreLayout(
                coupling_map,
                max_iterations=4,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method not in (None, "default", "sabre"),
            )
            layout.append(
                ConditionalController(
                    [
                        BarrierBeforeFinalMeasurements(
                            "qiskit.transpiler.internal.routing.protection.barrier"
                        ),
                        choose_layout_1,
                    ],
                    condition=_vf2_match_not_found,
                )
            )
        else:
            raise TranspilerError(f"Invalid optimization level: {optimization_level}")

        embed = common.generate_embed_passmanager(coupling_map)
        layout.append(ConditionalController(embed.to_flow_controller(), condition=_swap_mapped))
        return layout


class TrivialLayoutPassManager(PassManagerStagePlugin):
    """Plugin class for trivial layout stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        _given_layout = SetLayout(pass_manager_config.initial_layout)

        def _choose_layout_condition(property_set):
            return not property_set["layout"]

        if pass_manager_config.target is None:
            coupling_map = pass_manager_config.coupling_map
        else:
            coupling_map = pass_manager_config.target

        layout = PassManager()
        layout.append(_given_layout)
        if coupling_map is not None:
            layout.append(
                ConditionalController(
                    TrivialLayout(coupling_map), condition=_choose_layout_condition
                )
            )
        layout += common.generate_embed_passmanager(coupling_map)
        return layout


class DenseLayoutPassManager(PassManagerStagePlugin):
    """Plugin class for dense layout stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        _given_layout = SetLayout(pass_manager_config.initial_layout)

        def _choose_layout_condition(property_set):
            return not property_set["layout"]

        if pass_manager_config.target is None:
            coupling_map = pass_manager_config.coupling_map
        else:
            coupling_map = pass_manager_config.target

        layout = PassManager()
        layout.append(_given_layout)
        if coupling_map is not None:
            layout.append(
                ConditionalController(
                    DenseLayout(
                        coupling_map=pass_manager_config.coupling_map,
                        target=pass_manager_config.target,
                    ),
                    condition=_choose_layout_condition,
                )
            )
        layout += common.generate_embed_passmanager(coupling_map)
        return layout


class SabreLayoutPassManager(PassManagerStagePlugin):
    """Plugin class for sabre layout stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        _given_layout = SetLayout(pass_manager_config.initial_layout)

        def _choose_layout_condition(property_set):
            return not property_set["layout"]

        def _swap_mapped(property_set):
            return property_set["final_layout"] is None

        if pass_manager_config.target is None:
            coupling_map = pass_manager_config.coupling_map
        else:
            coupling_map = pass_manager_config.target

        layout = PassManager()
        layout.append(_given_layout)
        if coupling_map is None:
            layout_pass = None
        elif optimization_level == 0:
            trial_count = _get_trial_count(5)

            layout_pass = SabreLayout(
                coupling_map,
                max_iterations=1,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method not in (None, "default", "sabre"),
            )
        elif optimization_level == 1:
            trial_count = _get_trial_count(5)

            layout_pass = SabreLayout(
                coupling_map,
                max_iterations=2,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method not in (None, "default", "sabre"),
            )
        elif optimization_level == 2:
            trial_count = _get_trial_count(20)

            layout_pass = SabreLayout(
                coupling_map,
                max_iterations=2,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method not in (None, "default", "sabre"),
            )
        elif optimization_level == 3:
            trial_count = _get_trial_count(20)

            layout_pass = SabreLayout(
                coupling_map,
                max_iterations=4,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method not in (None, "default", "sabre"),
            )
        else:
            raise TranspilerError(f"Invalid optimization level: {optimization_level}")
        if layout_pass is not None:
            layout.append(
                ConditionalController(
                    [
                        BarrierBeforeFinalMeasurements(
                            "qiskit.transpiler.internal.routing.protection.barrier"
                        ),
                        layout_pass,
                    ],
                    condition=_choose_layout_condition,
                )
            )
        embed = common.generate_embed_passmanager(coupling_map)
        layout.append(ConditionalController(embed.to_flow_controller(), condition=_swap_mapped))
        return layout


def _get_trial_count(default_trials=5):
    if CONFIG.get("sabre_all_threads", None) or os.getenv("QISKIT_SABRE_ALL_THREADS"):
        return max(default_num_processes(), default_trials)
    return default_trials


class CliffordTInitPassManager(PassManagerCliffordTStagePlugin):
    """
    Clifford+T transpilation stage, which decomposes larger gates into 1-qubit
    and 2-qubits gates and performs logical optimizations.
    """

    # Notes:
    # In theory, we could leave larger-qubit Clifford gates in-place, provided we do not have
    # the layout + routing stages, and the rest of the passes know how to handle larger-qubit
    # Clifford gates.
    def pass_manager(
        self, pass_manager_config: PassManagerCliffordTConfig, optimization_level: int | None = None
    ):
        optimization_metric = OptimizationMetric.COUNT_T
        clifford_rz_gates = get_clifford_gate_names() + ["t", "tdg", "rz"]

        if optimization_level == 0:
            init = None
            if (
                pass_manager_config.initial_layout
                or pass_manager_config.coupling_map
                or (
                    pass_manager_config.target is not None
                    and pass_manager_config.target.build_coupling_map() is not None
                )
            ):
                init = common.generate_unroll_3q(
                    None,
                    clifford_rz_gates,
                    pass_manager_config.approximation_degree,
                    pass_manager_config.unitary_synthesis_method,
                    pass_manager_config.unitary_synthesis_plugin_config,
                    pass_manager_config.hls_config,
                    pass_manager_config.qubits_initially_zero,
                    optimization_metric,
                )
        elif optimization_level == 1:
            init = PassManager()
            if (
                pass_manager_config.initial_layout
                or pass_manager_config.coupling_map
                or (
                    pass_manager_config.target is not None
                    and pass_manager_config.target.build_coupling_map() is not None
                )
            ):
                init += common.generate_unroll_3q(
                    None,
                    clifford_rz_gates,
                    pass_manager_config.approximation_degree,
                    pass_manager_config.unitary_synthesis_method,
                    pass_manager_config.unitary_synthesis_plugin_config,
                    pass_manager_config.hls_config,
                    pass_manager_config.qubits_initially_zero,
                    optimization_metric,
                )
            init.append(
                [
                    InverseCancellation(),
                    ContractIdleWiresInControlFlow(),
                ]
            )

        elif optimization_level in {2, 3}:
            init = common.generate_unroll_3q(
                None,
                clifford_rz_gates,
                pass_manager_config.approximation_degree,
                pass_manager_config.unitary_synthesis_method,
                pass_manager_config.unitary_synthesis_plugin_config,
                pass_manager_config.hls_config,
                pass_manager_config.qubits_initially_zero,
                optimization_metric,
            )
            if not pass_manager_config._routing_disabled:
                init.append(ElidePermutations())
            init.append(
                [
                    RemoveDiagonalGatesBeforeMeasure(),
                    # Target not set on RemoveIdentityEquivalent because we haven't applied a Layout
                    # yet so doing anything relative to an error rate in the target is not valid.
                    RemoveIdentityEquivalent(
                        approximation_degree=pass_manager_config.approximation_degree
                    ),
                    InverseCancellation(),
                    ContractIdleWiresInControlFlow(),
                ]
            )
            init.append(
                CommutativeOptimization(
                    approximation_degree=(
                        pass_manager_config.approximation_degree
                        if pass_manager_config.approximation_degree is not None
                        else 1.0
                    )
                )
            )

            # We do not want to consolidate blocks for a Clifford+T basis set,
            # since this involves resynthesizing 2-qubit unitaries.

            # If approximation degree is None that indicates a request to approximate up to the
            # error rates in the target. However, in the init stage we don't yet know the target
            # qubits being used to figure out the fidelity so just use the default fidelity parameter
            # in this case.
            split_2q_unitaries_swap = not pass_manager_config._routing_disabled
            if pass_manager_config.approximation_degree is not None:
                init.append(
                    Split2QUnitaries(
                        pass_manager_config.approximation_degree, split_swap=split_2q_unitaries_swap
                    )
                )
            else:
                init.append(Split2QUnitaries(split_swap=split_2q_unitaries_swap))
        else:
            raise TranspilerError(f"Invalid optimization level {optimization_level}")
        return init


class TranslateToCliffordRZPassManager(PassManagerCliffordTStagePlugin):
    """
    Clifford+T transpilation stage, which translates circuits into Clifford+RZ+T basis set.
    """

    def pass_manager(
        self, pass_manager_config: PassManagerCliffordTConfig, optimization_level: int | None = None
    ):
        clifford_rz_gates = get_clifford_gate_names() + ["t", "tdg", "rz"]
        translate = PassManager(
            [
                UnitarySynthesis(
                    clifford_rz_gates,
                    approximation_degree=pass_manager_config.approximation_degree,
                    coupling_map=pass_manager_config.coupling_map,
                    plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
                    method=pass_manager_config.unitary_synthesis_method,
                    target=None,
                ),
                HighLevelSynthesis(
                    hls_config=pass_manager_config.hls_config,
                    coupling_map=pass_manager_config.coupling_map,
                    target=None,
                    use_qubit_indices=True,
                    equivalence_library=sel,
                    basis_gates=clifford_rz_gates,
                    qubits_initially_zero=pass_manager_config.qubits_initially_zero,
                    optimization_metric=OptimizationMetric.COUNT_T,
                ),
                # Note that HighLevelSynthesis does not translate gates in the equivalence
                # library, which is BasisTranslator is used as well.
                BasisTranslator(sel, clifford_rz_gates, None),
            ]
        )
        return translate


class OptimizeCliffordRZPassManager(PassManagerCliffordTStagePlugin):
    """
    Clifford+T transpilation stage, which optimizes Clifford+RZ+T circuits.
    """

    def pass_manager(
        self, pass_manager_config: PassManagerCliffordTConfig, optimization_level: int | None = None
    ):
        """Build pass manager for optimization stage."""

        clifford_rz_gates = get_clifford_gate_names() + ["t", "tdg", "rz"]

        match optimization_level:
            case 0:
                return PassManager()
            case 1:
                pre_loop = []
                loop = [
                    InverseCancellation(),
                    ContractIdleWiresInControlFlow(),
                ]
                post_loop = []
                loop_check, continue_loop = _optimization_check_fixed_point_clifford_rz()
            case 2 | 3:
                clifford_t_gates = get_clifford_gate_names() + ["t", "tdg"]

                def consolidate_run_fn(_dag, run):
                    # we keep the run intact if it is only diagonals or only cliffords,
                    # meaning we collect if it's non-diag and non-clifford
                    contains_non_diag = any(
                        node.name not in {"rz", "t", "tdg", "s", "sdg", "z"} for node in run
                    )
                    contains_non_clifford = any(node.name not in clifford_t_gates for node in run)
                    return contains_non_clifford and contains_non_diag

                pre_loop = [
                    Collect1qRuns(consolidate_run_fn),
                    Collect2qBlocks(consolidate_run_fn),
                    ConsolidateBlocks(
                        basis_gates=clifford_rz_gates,
                        target=None,
                        approximation_degree=pass_manager_config.approximation_degree,
                    ),
                    UnitarySynthesis(
                        clifford_rz_gates,
                        approximation_degree=pass_manager_config.approximation_degree,
                        coupling_map=pass_manager_config.coupling_map,
                        method=pass_manager_config.unitary_synthesis_method,
                        plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
                        target=None,
                    ),
                ]
                # The optimization loop runs OptimizeCliffordT + CommutativeCancellation
                # until fixpoint.
                loop = [
                    RemoveIdentityEquivalent(
                        approximation_degree=pass_manager_config.approximation_degree,
                        target=pass_manager_config.target,
                    ),
                    CommutativeOptimization(
                        approximation_degree=(
                            pass_manager_config.approximation_degree
                            if pass_manager_config.approximation_degree is not None
                            else 1.0
                        )
                    ),
                    ContractIdleWiresInControlFlow(),
                ]

                # CommutativeOptimization may produce RX gates, so we need BasisTranslator
                # to convert them back to RZ-gates.
                post_loop = [BasisTranslator(sel, clifford_rz_gates, None)]
                loop_check, continue_loop = _optimization_check_fixed_point_clifford_rz()
            case bad:
                raise TranspilerError(f"Invalid optimization_level: {bad}")

        optimization = PassManager()
        optimization.append(pre_loop + loop_check)
        optimization.append(DoWhileController(loop + loop_check, do_while=continue_loop))
        optimization.append(post_loop)
        return optimization


class TranslateToCliffordTPassManager(PassManagerCliffordTStagePlugin):
    """
    Clifford+T transpilation stage, which translates Clifford+RZ+T circuits
    into Clifford+T circuits.
    """

    def pass_manager(
        self, pass_manager_config: PassManagerCliffordTConfig, optimization_level: int | None = None
    ):
        rz_config = pass_manager_config.rz_synthesis_config
        rz_synthesis_error = rz_config.get("rz_synthesis_error") if rz_config is not None else None
        rz_cache_error = rz_config.get("rz_cache_error") if rz_config is not None else None
        rz_to_t_translation = PassManager(
            [
                SubstitutePi4Rotations(
                    approximation_degree=(
                        pass_manager_config.approximation_degree
                        if pass_manager_config.approximation_degree is not None
                        else 1.0
                    )
                ),
                SynthesizeRZRotations(
                    approximation_degree=(
                        pass_manager_config.approximation_degree
                        if pass_manager_config.approximation_degree is not None
                        else 1.0
                    ),
                    synthesis_error=rz_synthesis_error,
                    cache_error=rz_cache_error,
                ),
            ]
        )
        return rz_to_t_translation


class OptimizeCliffordTPassManager(PassManagerCliffordTStagePlugin):
    """
    Clifford+T transpilation stage, which optimizes Clifford+T circuits.
    """

    def pass_manager(
        self, pass_manager_config: PassManagerCliffordTConfig, optimization_level: int | None = None
    ):
        basis_gates = pass_manager_config.basis_gates
        target = pass_manager_config.target

        def should_fix_direction(property_set):
            res = not property_set["all_gates_in_basis"]
            return res

        # The final stage consists of translating gates to target basis and adhering to
        # the directionality of the coupling map.
        translate_to_target = [
            BasisTranslator(sel, basis_gates, target),
            GatesInBasis(pass_manager_config.basis_gates, target=pass_manager_config.target),
            ConditionalController(
                [
                    GateDirection(
                        coupling_map=pass_manager_config.coupling_map,
                        target=pass_manager_config.target,
                    ),
                    BasisTranslator(sel, basis_gates, target),
                ],
                condition=should_fix_direction,
            ),
        ]

        match optimization_level:
            case 0:
                return PassManager(translate_to_target)

            case 1:
                loop = [
                    InverseCancellation(),
                    OptimizeCliffordT(basis_gates=basis_gates),
                    ContractIdleWiresInControlFlow(),
                ]
                loop_check, continue_loop = _optimization_check_fixed_point_clifford_t()
                post_loop = translate_to_target
            case 2 | 3:
                loop = [
                    OptimizeCliffordT(basis_gates=basis_gates),
                    CommutativeOptimization(
                        approximation_degree=(
                            pass_manager_config.approximation_degree
                            if pass_manager_config.approximation_degree is not None
                            else 1.0
                        )
                    ),
                    SubstitutePi4Rotations(
                        approximation_degree=(
                            pass_manager_config.approximation_degree
                            if pass_manager_config.approximation_degree is not None
                            else 1.0
                        )
                    ),
                    ContractIdleWiresInControlFlow(),
                ]
                loop_check, continue_loop = _optimization_check_fixed_point_clifford_t()
                post_loop = translate_to_target
            case bad:
                raise TranspilerError(f"Invalid optimization_level: {bad}")

        optimization = PassManager()
        optimization.append(loop_check)
        optimization.append(DoWhileController(loop + loop_check, do_while=continue_loop))
        optimization.append(post_loop)
        return optimization
