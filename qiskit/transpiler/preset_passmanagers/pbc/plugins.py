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
Built-in transpiler stage plugins for PBC transpilation.
"""

from __future__ import annotations

import abc
import typing

from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import CommutativeOptimization
from qiskit.transpiler.passes import ContractIdleWiresInControlFlow
from qiskit.transpiler.passes import ConvertToPauliRotations
from qiskit.transpiler.passes import InverseCancellation
from qiskit.transpiler.passes import HighLevelSynthesis
from qiskit.transpiler.passes import RemoveIdentityEquivalent
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.optimization_metric import OptimizationMetric

if typing.TYPE_CHECKING:
    from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig


class PassManagerPBCConfig:
    """Pass Manager Configuration for PBC transpilation."""

    def __init__(
        self,
        approximation_degree: float | None = None,
        seed_transpiler: int | None = None,
        unitary_synthesis_method: str = "default",
        unitary_synthesis_plugin_config: dict | None = None,
        hls_config: HLSConfig | None = None,
        qubits_initially_zero: bool = True,
    ):
        """
        Args:
            approximation_degree: Heuristic dial used for circuit approximation, where
                ``1.0`` means no approximation (up to numerical tolerance) and ``0.0``
                means the maximum approximation. The value of ``None`` is treated
                as ``1.0``.
            seed_transpiler: Sets random seed for the stochastic parts of
                the transpiler.
            unitary_synthesis_method: The string method to use for the
                :class:`~qiskit.transpiler.passes.UnitarySynthesis` pass. Will
                search installed plugins for a valid method. You can see a list of
                installed plugins with :func:`.unitary_synthesis_plugin_names`.
            unitary_synthesis_plugin_config: The configuration dictionary that will
                be passed to the specified unitary synthesis plugin. Refer to
                the plugin documentation for how to use this.
            hls_config: An optional configuration class to use for
                :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass.
                Specifies how to synthesize various high-level objects.
            qubits_initially_zero: Indicates whether the input circuit is
                zero-initialized.
        """
        self.approximation_degree = approximation_degree
        self.seed_transpiler = seed_transpiler
        self.unitary_synthesis_method = unitary_synthesis_method
        self.unitary_synthesis_plugin_config = unitary_synthesis_plugin_config
        self.hls_config = hls_config
        self.qubits_initially_zero = qubits_initially_zero


class PassManagerPBCStagePlugin(abc.ABC):
    """A ``PassManagerPBCStagePlugin`` is a plugin interface object for defining
    stages in :func:`~.generate_preset_pbc_pass_manager`.
    """

    @abc.abstractmethod
    def pass_manager(
        self, pass_manager_config: PassManagerPBCConfig, optimization_level: int | None = None
    ) -> PassManager | None:
        """This method is designed to return a :class:`~.PassManager` for the stage this implements

        Args:
            pass_manager_config: A configuration object that defines all the target device
                specifications and any user specified options to
                :func:`~.generate_preset_pbc_pass_manager`.
            optimization_level: The optimization level of the transpilation, if set this
                should be used to set values for any tunable parameters to trade off runtime
                for potential optimization. Valid values should be ``0``, ``1``, ``2``, or ``3``
                and the higher the number the more optimization is expected.

        Returns:
            the :class:`.PassManager` to run, or ``None`` if nothing is needed for this
            configuration (for example, an optimization plugin might return ``None`` at
            ``optimization_level=0``).
        """


class UnrollPassManager(PassManagerPBCStagePlugin):
    """
    PBC transpilation stage, which decomposes circuit instruction into standard gates and instructions.
    """

    def pass_manager(
        self, pass_manager_config: PassManagerPBCConfig, optimization_level: int | None = None
    ):
        basis_gates = (
            list(get_standard_gate_name_mapping())
            + list(CONTROL_FLOW_OP_NAMES)
            + ["pauli_product_rotation", "pauli_product_measurement"]
        )
        pm = PassManager(
            [
                UnitarySynthesis(
                    basis_gates,
                    target=None,
                    min_qubits=1,
                    approximation_degree=pass_manager_config.approximation_degree,
                    method=pass_manager_config.unitary_synthesis_method,
                    plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
                ),
                HighLevelSynthesis(
                    hls_config=pass_manager_config.hls_config,
                    coupling_map=None,
                    target=None,
                    use_qubit_indices=False,
                    equivalence_library=sel,
                    basis_gates=basis_gates,
                    min_qubits=1,
                    qubits_initially_zero=pass_manager_config.qubits_initially_zero,
                    optimization_metric=OptimizationMetric.COUNT_T,
                ),
            ]
        )
        return pm


class OptimizePassManager(PassManagerPBCStagePlugin):
    """
    PBC transpilation stage, which optimizes circuits with standard gates and instructions.
    """

    def pass_manager(
        self, pass_manager_config: PassManagerPBCConfig, optimization_level: int | None = None
    ):
        match optimization_level:
            case 0:
                pm = PassManager([])
            case 1:
                pm = PassManager(
                    [
                        InverseCancellation(),
                        ContractIdleWiresInControlFlow(),
                    ]
                )
            case 2 | 3:
                pm = PassManager(
                    [
                        RemoveIdentityEquivalent(
                            approximation_degree=pass_manager_config.approximation_degree,
                            target=None,
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
                )
            case bad:
                raise TranspilerError(f"Invalid optimization_level: {bad}")

        return pm


class TranslateToPBCPassManager(PassManagerPBCStagePlugin):
    """
    PBC transpilation stage, which translates circuits with standard gates and instructions
    into Pauli-based circuits.
    """

    def pass_manager(
        self, pass_manager_config: PassManagerPBCConfig, optimization_level: int | None = None
    ):
        # Use BasisTranslator to translate standard gates into gates supported by ConvertToPauliRotations
        unsupported_gates = ["c3sx", "rcccx"]
        supported_gates = [
            gate for gate in get_standard_gate_name_mapping() if gate not in unsupported_gates
        ]
        basis_gates = (
            supported_gates
            + list(CONTROL_FLOW_OP_NAMES)
            + ["pauli_product_rotation", "pauli_product_measurement"]
        )

        pm = PassManager(
            [
                BasisTranslator(sel, basis_gates, target=None, min_qubits=1),
                ConvertToPauliRotations(),
            ]
        )
        return pm


class OptimizePBCPassManager(PassManagerPBCStagePlugin):
    """
    PBC transpilation stage, which optimizes Pauli-based circuits.
    """

    def pass_manager(
        self, pass_manager_config: PassManagerPBCConfig, optimization_level: int | None = None
    ):
        match optimization_level:
            case 0 | 1:
                pm = PassManager([])
            case 2 | 3:
                pm = PassManager(
                    [
                        CommutativeOptimization(
                            approximation_degree=(
                                pass_manager_config.approximation_degree
                                if pass_manager_config.approximation_degree is not None
                                else 1.0
                            )
                        ),
                    ]
                )
            case bad:
                raise TranspilerError(f"Invalid optimization_level: {bad}")

        return pm
