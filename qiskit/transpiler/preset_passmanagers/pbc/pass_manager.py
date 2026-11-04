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
Preset pass manager generation function for compiling into PBC.
"""

import os

from qiskit import user_config

from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.preset_passmanagers.pbc.plugins import (
    PassManagerPBCConfig,
    UnrollPassManager,
    OptimizePassManager,
    TranslateToPBCPassManager,
    OptimizePBCPassManager,
)
from qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager import (
    _parse_approximation_degree,
    _parse_seed_transpiler,
)

from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig


def pbc_pass_manager(
    pass_manager_config: PassManagerPBCConfig, optimization_level: int
) -> StagedPassManager:
    """Generate a staged pass manager for transpiling into PBC.

    This function is invoked by :func:`.generate_preset_pbc_pass_manager`.
    It generates a specialized transpilation pipeline consisting of the following stages:

    * Unrolling: Decompose circuit instructions into a basis consisting of
      standard gates and instructions, pauli product rotations and measurements, and
      control-flow operations.
    * Optimization: Optimize unrolled circuits.
    * PBC translation: Translate unrolled circuits with into Pauli-based circuits.
    * PBC optimization: Optimize Pauli-based circuits.

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

    unrolling = UnrollPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    optimization = OptimizePassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    pbc_translation = TranslateToPBCPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    pbc_optimization = OptimizePBCPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )

    stages = [
        "unrolling",
        "optimization",
        "pbc_translation",
        "pbc_optimization",
    ]

    return StagedPassManager(
        stages=stages,
        unrolling=unrolling,
        optimization=optimization,
        pbc_translation=pbc_translation,
        pbc_optimization=pbc_optimization,
    )


def generate_preset_pbc_pass_manager(
    optimization_level: int = 2,
    approximation_degree: float | None = 1.0,
    seed_transpiler: int | None = None,
    unitary_synthesis_method: str = "default",
    unitary_synthesis_plugin_config: dict | None = None,
    hls_config: HLSConfig | None = None,
    qubits_initially_zero: bool = True,
) -> StagedPassManager:
    """Generate a preset PBC :class:`~.StagedPassManager`.

    This function provides a convenient way to construct a preset pass manager for
    PBC compilation.

    Args:
        optimization_level: The optimization level to generate a
            :class:`~.StagedPassManager` for. By default optimization level 2
            is used if this is not specified. This can be 0, 1, 2, or 3. Higher
            levels generate potentially more optimized circuits, at the expense
            of potentially longer transpilation time.
        approximation_degree: Heuristic dial used for circuit approximation, where
            ``1.0`` means no approximation (up to numerical tolerance) and ``0.0``
            means the maximum approximation.
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

    Returns:
        The preset pass manager for the given options.

    Raises:
        TranspilerError: if an invalid value for ``optimization_level`` is passed in.
    """

    config = user_config.get_config()

    if optimization_level is None:
        optimization_level = config.get("transpile_optimization_level", 2)

    if seed_transpiler is None:
        if (seed := os.getenv("QISKIT_TRANSPILER_SEED", None)) is not None:
            seed_transpiler = int(seed)
        else:
            seed_transpiler = config.get("transpiler_seed", None)

    approximation_degree = _parse_approximation_degree(approximation_degree)
    if approximation_degree is None:
        approximation_degree = 1.0

    seed_transpiler = _parse_seed_transpiler(seed_transpiler)

    pm_options = {
        "approximation_degree": approximation_degree,
        "seed_transpiler": seed_transpiler,
        "unitary_synthesis_method": unitary_synthesis_method,
        "unitary_synthesis_plugin_config": unitary_synthesis_plugin_config,
        "hls_config": hls_config,
        "qubits_initially_zero": qubits_initially_zero,
    }

    pm_config = PassManagerPBCConfig(**pm_options)

    pm = pbc_pass_manager(pm_config, optimization_level=optimization_level)
    return pm
