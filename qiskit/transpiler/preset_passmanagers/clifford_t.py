# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Preset pass manager generation function for compiling into Clifford+T.
"""

from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.passmanager_config import PassManagerCliffordTConfig, PassManagerConfig
from qiskit.transpiler.preset_passmanagers.builtin_plugins import (
    CliffordTInitPassManager,
    DefaultLayoutPassManager,
    DefaultRoutingPassManager,
    DefaultSchedulingPassManager,
    OptimizeCliffordRZPassManager,
    OptimizeCliffordTPassManager,
    TranslateToCliffordRZPassManager,
    TranslateToCliffordTPassManager,
)
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePluginManager


def clifford_t_pass_manager(
    pass_manager_config: PassManagerCliffordTConfig, optimization_level: int
) -> StagedPassManager:
    r"""Generate a staged pass manager for transpiling into Clifford+T basis.

    This function is invoked by :func:`.generate_preset_pass_manager` when
    the target basis consists of Clifford+T gates. It generates a specialized
    transpilation pipeline consisting of the following stages:

    * Initialization: Decompose larger gates into 1-qubit and 2-qubits gates and perform
      logical optimizations.
    * Layout: Apply the default layout strategy used for continuous basis sets.
    * Routing: Apply the default routing strategy used for continuous basis sets.
    * RZ translation: Translate the circuit into Clifford+RZ basis.
    * RZ optimization: Optimize the circuit within Clifford+RZ basis.
    * T translation: Translate the circuit into Clifford+T basis.
    * T optimization: Optimizes the circuit within Clifford+T basis.
    * Scheduling: Apply the default scheduling strategy used for continuous basis sets.

    For best results, consider including both :math:`T` and :math:`T^\dagger` into the specified
    Clifford+T basis and as many Clifford gates as possible. For example::

        basis_gates = get_clifford_gate_names() + ["t", "tdg"]

    .. note::

        These stages are still experimental and subject to change. In particular,
        they are not yet exposed as transpiler stage plugins (unlike the stages
        for continuous basis sets).

    Args:
        pass_manager_config: Configuration of the pass manager.
        optimization_level: The optimization level. By default optimization level 2
            is used if this is not specified. This can be 0, 1, 2, or 3. Higher
            levels generate potentially more optimized circuits, at the expense
            of longer transpilation time.
    Returns:
        Staged pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    if optimization_level not in [0, 1, 2, 3]:
        raise TranspilerError(f"Invalid optimization level {optimization_level}")

    legacy_config = pass_manager_config._to_legacy_config()

    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    if coupling_map or initial_layout:
        layout = DefaultLayoutPassManager().pass_manager(
            legacy_config, optimization_level=optimization_level
        )
        routing = DefaultRoutingPassManager().pass_manager(
            legacy_config, optimization_level=optimization_level
        )
    else:
        layout = None
        routing = None

    init = CliffordTInitPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    rz_translation = TranslateToCliffordRZPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    rz_optimization = OptimizeCliffordRZPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    t_translation = TranslateToCliffordTPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    t_optimization = OptimizeCliffordTPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    sched = DefaultSchedulingPassManager().pass_manager(
        legacy_config, optimization_level=optimization_level
    )

    stages = [
        "init",
        "layout",
        "routing",
        "rz_translation",
        "rz_optimization",
        "t_translation",
        "t_optimization",
        "scheduling",
    ]

    return StagedPassManager(
        stages=stages,
        init=init,
        layout=layout,
        routing=routing,
        rz_translation=rz_translation,
        rz_optimization=rz_optimization,
        t_translation=t_translation,
        t_optimization=t_optimization,
        scheduling=sched,
    )


def clifford_t_pass_manager_legacy(
    pass_manager_config: PassManagerConfig, optimization_level: int
) -> StagedPassManager:
    r"""Generate a staged pass manager for transpiling into Clifford+T basis.

    This function is invoked by :func:`.generate_preset_pass_manager` when
    the target basis consists of Clifford+T gates. It generates a specialized
    transpilation pipeline consisting of the six usual stages that can be
    specified by the plugin interface.

    * Initialization: Decompose larger gates into 1-qubit and 2-qubits gates and perform
      logical optimizations.
    * Layout: Apply the default layout strategy used for continuous basis sets.
    * Routing: Apply the default routing strategy used for continuous basis sets.
    * Translation: Translate the circuit into Clifford+T basis.
    * Optimization: Optimizes the circuit within Clifford+T basis.
    * Scheduling: Apply the default scheduling strategy used for continuous basis sets.

    For best results, consider including both :math:`T` and :math:`T^\dagger` into the specified
    Clifford+T basis and as many Clifford gates as possible. For example::

        basis_gates = get_clifford_gate_names() + ["t", "tdg"]

    Args:
        pass_manager_config: Configuration of the pass manager.
        optimization_level: The optimization level. By default optimization level 2
            is used if this is not specified. This can be 0, 1, 2, or 3. Higher
            levels generate potentially more optimized circuits, at the expense
            of longer transpilation time.
    Returns:
        Staged pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    if optimization_level not in [0, 1, 2, 3]:
        raise TranspilerError(f"Invalid optimization level {optimization_level}")

    plugin_manager = PassManagerStagePluginManager()

    # Create CliffordT config with the specified values for common options,
    # and default values for CliffordT-specific options (note that such options
    # cannot be set via the legacy interface).
    clifford_t_pass_manager_config = PassManagerCliffordTConfig._from_legacy_config(
        pass_manager_config
    )

    # If init method is set, we use the specified plugin.
    # Otherwise, we call the Clifford+T init stage.
    if (init_method := pass_manager_config.init_method) is not None:
        init = plugin_manager.get_passmanager_stage(
            "init", init_method, pass_manager_config, optimization_level=optimization_level
        )
    else:
        init = CliffordTInitPassManager().pass_manager(
            clifford_t_pass_manager_config, optimization_level=optimization_level
        )

    # If coupling map or initial layout are provided, use the specified plugins
    # for layout/routing stages, falling back on defaults.
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    if coupling_map or initial_layout:
        layout = plugin_manager.get_passmanager_stage(
            "layout",
            pass_manager_config.layout_method or "default",
            pass_manager_config,
            optimization_level=optimization_level,
        )
        routing = plugin_manager.get_passmanager_stage(
            "routing",
            pass_manager_config.routing_method or "default",
            pass_manager_config,
            optimization_level=optimization_level,
        )
    else:
        layout = None
        routing = None

    # If translation method is set, we use the specified plugin for translation.
    # Otherwise, translation to Clifford+T basis consists of first translating to Clifford+RZ,
    # optimizing Clifford+RZ, and then translating to Clifford+T.
    if (translation_method := pass_manager_config.translation_method) is not None:
        translation = plugin_manager.get_passmanager_stage(
            "init", translation_method, pass_manager_config, optimization_level=optimization_level
        )
    else:
        translation = (
            TranslateToCliffordRZPassManager().pass_manager(
                clifford_t_pass_manager_config, optimization_level=optimization_level
            )
            + OptimizeCliffordRZPassManager().pass_manager(
                clifford_t_pass_manager_config, optimization_level=optimization_level
            )
            + TranslateToCliffordTPassManager().pass_manager(
                clifford_t_pass_manager_config, optimization_level=optimization_level
            )
        )

    # If optimization method is set, we use the specified plugin.
    # Otherwise, we call the Clifford+T optimization stage.
    if (optimization_method := pass_manager_config.optimization_method) is not None:
        optimization = plugin_manager.get_passmanager_stage(
            "optimization",
            optimization_method,
            pass_manager_config,
            optimization_level=optimization_level,
        )
    else:
        optimization = OptimizeCliffordTPassManager().pass_manager(
            clifford_t_pass_manager_config, optimization_level=optimization_level
        )

    # Use the specified plugins for scheduling stage, falling back on default.
    sched = plugin_manager.get_passmanager_stage(
        "scheduling",
        pass_manager_config.scheduling_method or "default",
        pass_manager_config,
        optimization_level=optimization_level,
    )

    return StagedPassManager(
        init=init,
        layout=layout,
        routing=routing,
        translation=translation,
        optimization=optimization,
        scheduling=sched,
    )
