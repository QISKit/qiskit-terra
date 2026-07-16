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

"""Module containing transpiler optimization passes."""

from .optimize_1q_gates import Optimize1qGates
from .optimize_1q_decomposition import Optimize1qGatesDecomposition
from .collect_2q_blocks import Collect2qBlocks
from .collect_multiqubit_blocks import CollectMultiQBlocks
from .consolidate_blocks import ConsolidateBlocks
from .commutation_analysis import CommutationAnalysis
from .commutative_cancellation import CommutativeCancellation
from .commutative_optimization import CommutativeOptimization
from .commutative_inverse_cancellation import CommutativeInverseCancellation
from .optimize_1q_commutation import Optimize1qGatesSimpleCommutation
from .optimize_swap_before_measure import OptimizeSwapBeforeMeasure
from .remove_reset_in_zero_state import RemoveResetInZeroState
from .remove_final_reset import RemoveFinalReset
from .remove_diagonal_gates_before_measure import RemoveDiagonalGatesBeforeMeasure
from .hoare_opt import HoareOptimizer
from .template_optimization import TemplateOptimization
from .inverse_cancellation import InverseCancellation
from .collect_1q_runs import Collect1qRuns
from .collect_linear_functions import CollectLinearFunctions
from .reset_after_measure_simplification import ResetAfterMeasureSimplification
from .optimize_cliffords import OptimizeCliffords
from .collect_cliffords import CollectCliffords
from .elide_permutations import ElidePermutations
from .optimize_annotated import OptimizeAnnotated
from .remove_identity_equiv import RemoveIdentityEquivalent
from .split_2q_unitaries import Split2QUnitaries
from .collect_and_collapse import CollectAndCollapse
from .two_qubit_peephole import TwoQubitPeepholeOptimization
from .contract_idle_wires_in_control_flow import ContractIdleWiresInControlFlow
from .optimize_clifford_t import OptimizeCliffordT
from .litinski_transformation import LitinskiTransformation
from .light_cone import LightCone
from .substitute_pi4_rotations import SubstitutePi4Rotations
from .convert_to_pauli_rotations import ConvertToPauliRotations

__all__ = [
    "Collect1qRuns",
    "Collect2qBlocks",
    "CollectAndCollapse",
    "CollectCliffords",
    "CollectLinearFunctions",
    "CollectMultiQBlocks",
    "CommutationAnalysis",
    "CommutativeCancellation",
    "CommutativeInverseCancellation",
    "CommutativeOptimization",
    "ConsolidateBlocks",
    "ContractIdleWiresInControlFlow",
    "ConvertToPauliRotations",
    "ElidePermutations",
    "HoareOptimizer",
    "InverseCancellation",
    "LightCone",
    "LitinskiTransformation",
    "Optimize1qGates",
    "Optimize1qGatesDecomposition",
    "Optimize1qGatesSimpleCommutation",
    "OptimizeAnnotated",
    "OptimizeCliffordT",
    "OptimizeCliffords",
    "OptimizeSwapBeforeMeasure",
    "RemoveDiagonalGatesBeforeMeasure",
    "RemoveFinalReset",
    "RemoveIdentityEquivalent",
    "RemoveResetInZeroState",
    "ResetAfterMeasureSimplification",
    "Split2QUnitaries",
    "SubstitutePi4Rotations",
    "TemplateOptimization",
    "TwoQubitPeepholeOptimization",
]
