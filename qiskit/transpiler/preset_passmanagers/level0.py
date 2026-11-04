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

"""Pass manager for optimization level 0, providing no explicit optimization.

Level 0 pass manager: no explicit optimization other than mapping to backend.
"""

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePluginManager,
)


def level_0_pass_manager(pass_manager_config: PassManagerConfig) -> StagedPassManager:
    """Level 0 pass manager: no explicit optimization other than mapping to backend.

    This pass manager applies the user-given initial layout. If none is given, a trivial
    layout consisting of mapping the i-th virtual qubit to the i-th physical qubit is used.
    Any unused physical qubit is allocated as ancilla space.

    The pass manager then unrolls the circuit to the desired basis, and transforms the
    circuit to match the coupling map.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 0 pass manager.

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
        "routing", routing_method, pass_manager_config, optimization_level=0
    )

    # Build pass manager
    if coupling_map or initial_layout:
        layout = plugin_manager.get_passmanager_stage(
            "layout", layout_method, pass_manager_config, optimization_level=0
        )
        routing = routing_pm
    else:
        layout = None
        routing = None

    translation = plugin_manager.get_passmanager_stage(
        "translation", translation_method, pass_manager_config, optimization_level=0
    )

    sched = plugin_manager.get_passmanager_stage(
        "scheduling", scheduling_method, pass_manager_config, optimization_level=0
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
        "init", init_method, pass_manager_config, optimization_level=0
    )
    optimization = plugin_manager.get_passmanager_stage(
        "optimization", optimization_method, pass_manager_config, optimization_level=0
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
