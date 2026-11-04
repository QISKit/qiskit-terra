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

"""
===================================================
Transpiler Passes (:mod:`qiskit.transpiler.passes`)
===================================================

.. currentmodule:: qiskit.transpiler.passes

Layout Selection (Placement)
============================

.. autosummary::
   :toctree: ../stubs/

   SetLayout
   TrivialLayout
   DenseLayout
   SabreLayout
   CSPLayout
   VF2Layout
   ApplyLayout
   Layout2qDistance
   EnlargeWithAncilla
   FullAncillaAllocation
   SabrePreLayout

Routing
=======

.. autosummary::
   :toctree: ../stubs/

   BasicSwap
   Commuting2qGateRouter
   LayoutTransformation
   LookaheadSwap
   SabreSwap
   StarPreRouting

Basis Change
============

.. autosummary::
   :toctree: ../stubs/

   BasisTranslator
   Decompose
   TranslateParameterizedGates
   Unroll3qOrMore
   UnrollCustomDefinitions

Optimizations
=============

.. autosummary::
   :toctree: ../stubs/

   Collect1qRuns
   Collect2qBlocks
   CollectAndCollapse
   CollectCliffords
   CollectLinearFunctions
   CollectMultiQBlocks
   CommutationAnalysis
   CommutativeCancellation
   CommutativeInverseCancellation
   CommutativeOptimization
   ConsolidateBlocks
   ContractIdleWiresInControlFlow
   ConvertToPauliRotations
   ElidePermutations
   HoareOptimizer
   InverseCancellation
   LightCone
   LitinskiTransformation
   Optimize1qGates
   Optimize1qGatesDecomposition
   Optimize1qGatesSimpleCommutation
   OptimizeAnnotated
   OptimizeCliffordT
   OptimizeCliffords
   OptimizeSwapBeforeMeasure
   RemoveDiagonalGatesBeforeMeasure
   RemoveFinalReset
   RemoveIdentityEquivalent
   RemoveResetInZeroState
   ResetAfterMeasureSimplification
   Split2QUnitaries
   SubstitutePi4Rotations
   TemplateOptimization
   TwoQubitPeepholeOptimization

Scheduling
=============

.. autosummary::
   :toctree: ../stubs/

   ALAPScheduleAnalysis
   ASAPScheduleAnalysis
   ConstrainedReschedule
   ContextAwareDynamicalDecoupling
   InstructionDurationCheck
   PadDelay
   PadDynamicalDecoupling
   SetIOLatency
   TimeUnitConversion

Circuit Analysis
================

.. autosummary::
   :toctree: ../stubs/

   CountOps
   CountOpsLongestPath
   DAGLongestPath
   Depth
   NumTensorFactors
   ResourceEstimation
   Size
   Width

Synthesis
=========

The synthesis transpiler plugin documentation can be found in the
:mod:`qiskit.transpiler.passes.synthesis.plugin` page.

.. autosummary::
   :toctree: ../stubs/

   HLSConfig
   HighLevelSynthesis
   LinearFunctionsToPermutations
   SolovayKitaev
   SynthesizeRZRotations
   UnitarySynthesis

Post Layout
===========

These are post qubit selection.

.. autosummary::
   :toctree: ../stubs/

   VF2PostLayout

Additional Passes
=================

.. autosummary::
   :toctree: ../stubs/

   BarrierBeforeFinalMeasurements
   CheckGateDirection
   CheckMap
   ContainsInstruction
   DAGFixedPoint
   Error
   FilterOpNodes
   FixedPoint
   GateDirection
   GatesInBasis
   MergeAdjacentBarriers
   MinimumPoint
   RemoveBarriers
   RemoveFinalMeasurements
   UnrollForLoops
   WrapAngles

Additional data
===============

.. py:data:: qiskit.transpiler.passes.utils.wrap_angles.WRAP_ANGLE_REGISTRY

    A legacy location for :attr:`.WrapAngles.DEFAULT_REGISTRY`.  This path should only be used when
    full compatibility with Qiskit 2.2 is required.
"""

from . import analysis, basis, layout, optimization, routing, scheduling, synthesis, utils

from .analysis import *
from .basis import *
from .layout import *
from .optimization import *
from .routing import *
from .scheduling import *
from .synthesis import *
from .utils import *

__all__ = []
__all__ += analysis.__all__
__all__ += basis.__all__
__all__ += layout.__all__
__all__ += optimization.__all__
__all__ += routing.__all__
__all__ += scheduling.__all__
__all__ += synthesis.__all__
__all__ += utils.__all__
