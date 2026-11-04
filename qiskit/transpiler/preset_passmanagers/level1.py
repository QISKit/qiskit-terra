# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pass manager for optimization level 1, providing light optimization.

Level 1 pass manager: light optimization by simple adjacent gate collapsing.
"""
from __future__ import annotations
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePluginManager,
)


def level_1_pass_manager(pass_manager_config: PassManagerConfig) -> StagedPassManager:
    """Level 1 pass manager: light optimization by simple adjacent gate collapsing.

    This pass manager applies the user-given initial layout. If none is given,
    and a trivial layout (i-th virtual -> i-th physical) makes the circuit fit
    the coupling map, that is used.
    Otherwise, the circuit is mapped to the most densely connected coupling subgraph,
    and swaps are inserted to map. Any unused physical qubit is allocated as ancilla space.
    The pass manager then unrolls the circuit to the desired basis, and transforms the
    circuit to match the coupling map. Finally, optimizations in the form of adjacent
    gate collapse and redundant reset removal are performed.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 1 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    plugin_manager = PassManagerStagePluginManager()
    basis_gates = pass_manager_config.basis_gates
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    init_method = pass_manager_config.init_method or "default"
    layout_method = pass_manager_config.layout_method or "default"
    routing_method = pass_manager_config.routing_method or "default"
    translation_method = pass_manager_config.translation_method or "default"
    optimization_method = pass_manager_config.optimization_method or "default"
    scheduling_method = pass_manager_config.scheduling_method or "default"
    target = pass_manager_config.target

    # Choose routing pass
    routing_pm = plugin_manager.get_passmanager_stage(
        "routing", routing_method, pass_manager_config, optimization_level=1
    )

    # Build full pass manager
    if coupling_map or initial_layout:
        layout = plugin_manager.get_passmanager_stage(
            "layout", layout_method, pass_manager_config, optimization_level=1
        )
        routing = routing_pm

    else:
        layout = None
        routing = None

    translation = plugin_manager.get_passmanager_stage(
        "translation", translation_method, pass_manager_config, optimization_level=1
    )

    optimization = plugin_manager.get_passmanager_stage(
        "optimization", optimization_method, pass_manager_config, optimization_level=1
    )

    sched = plugin_manager.get_passmanager_stage(
        "scheduling", scheduling_method, pass_manager_config, optimization_level=1
    )

    pre_init = common.generate_control_flow_options_check(
        layout_method=layout_method,
        routing_method=routing_method,
        translation_method=translation_method,
        optimization_method=optimization_method,
        scheduling_method=scheduling_method,
        basis_gates=basis_gates,
        target=target,
    )
    init = plugin_manager.get_passmanager_stage(
        "init", init_method, pass_manager_config, optimization_level=1
    )

    return StagedPassManager(
        pre_init=pre_init,
        init=init,
        layout=layout,
        routing=routing,
        translation=translation,
        optimization=optimization,
        scheduling=sched,
    )
