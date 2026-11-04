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

"""Pass manager for optimization level 2, providing medium optimization.

Level 2 pass manager: medium optimization by noise adaptive qubit mapping and
gate cancellation using commutativity rules.
"""
from __future__ import annotations
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePluginManager,
)


def level_2_pass_manager(pass_manager_config: PassManagerConfig) -> StagedPassManager:
    """Level 2 pass manager: medium optimization by initial layout selection and
    gate cancellation using commutativity rules.

    This pass manager applies the user-given initial layout. If none is given, a search
    for a perfect layout (i.e. one that satisfies all 2-qubit interactions) is conducted.
    If no such layout is found, qubits are laid out on the most densely connected subset
    which also exhibits the best gate fidelities.

    The pass manager then transforms the circuit to match the coupling constraints.
    It is then unrolled to the basis, and any flipped cx directions are fixed.
    Finally, optimizations in the form of commutative gate cancellation and redundant
    reset removal are performed.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 2 pass manager.

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
        "routing", routing_method, pass_manager_config, optimization_level=2
    )

    # Build pass manager
    if coupling_map or initial_layout:
        layout = plugin_manager.get_passmanager_stage(
            "layout", layout_method, pass_manager_config, optimization_level=2
        )
        routing = routing_pm
    else:
        layout = None
        routing = None
    translation = plugin_manager.get_passmanager_stage(
        "translation", translation_method, pass_manager_config, optimization_level=2
    )

    optimization = plugin_manager.get_passmanager_stage(
        "optimization", optimization_method, pass_manager_config, optimization_level=2
    )

    sched = plugin_manager.get_passmanager_stage(
        "scheduling", scheduling_method, pass_manager_config, optimization_level=2
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
        "init", init_method, pass_manager_config, optimization_level=2
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
