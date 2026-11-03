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

Working with transpiler passes
==============================

Transpiler passes are the individual building blocks used by
:class:`~qiskit.transpiler.PassManager` objects to analyze or rewrite circuits.
They operate on Qiskit's internal :class:`~qiskit.dagcircuit.DAGCircuit`
representation, but users typically run them through a pass manager using
:class:`~qiskit.circuit.QuantumCircuit` inputs.

For an overview of transpilation, preset pass managers, and compilation
stages, see :mod:`qiskit.transpiler`.

For complete device-aware compilation, start with
:func:`~qiskit.transpiler.generate_preset_pass_manager`. The passes below are
useful when you want to inspect a circuit, build a smaller custom workflow, or
customize one stage of transpilation.

For example, a pass can collect information about a circuit. Here,
:class:`CountOps` stores operation counts in the pass manager's property set::

    from qiskit.circuit import QuantumCircuit
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import CountOps

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    pass_manager = PassManager([CountOps()])
    pass_manager.run(circuit)
    print(pass_manager.property_set["count_ops"])

Passes can be combined in a :class:`~qiskit.transpiler.PassManager` to build a
small custom compilation workflow. This example removes the barrier so that
the two H gates can cancel each other::

    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import HGate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import InverseCancellation, RemoveBarriers

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.barrier()
    circuit.h(0)

    pass_manager = PassManager([RemoveBarriers(), InverseCancellation([HGate()])])
    simplified = pass_manager.run(circuit)

Pass categories
===============

The passes below are loosely grouped by typical use to help readers navigate
this page. These groups are for documentation purposes only, and some passes
may fit into more than one group.

Layout Selection (Placement)
============================

Layout-selection passes choose or validate the mapping from the circuit's
virtual qubits to the target's physical qubits. Some passes simply apply a
provided layout, while others search for a layout that reduces routing or
hardware error.

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

Routing passes insert operations, typically swaps, so that every multi-qubit
operation satisfies the target connectivity. These passes usually run after a
layout has been selected.

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

Basis-change passes rewrite instructions into a requested gate set or target
instruction set. These are often used after routing, when the circuit topology
is compatible with the target but its operations still need to be expressed in
the target basis.

.. autosummary::
   :toctree: ../stubs/

   BasisTranslator
   Decompose
   TranslateParameterizedGates
   Unroll3qOrMore
   UnrollCustomDefinitions

Optimizations
=============

Optimization passes reduce or simplify a circuit while preserving its
semantics. Some are general-purpose, such as cancellation of inverse gates,
while others use target information or specialized synthesis routines.

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

Scheduling passes assign timing information to circuit instructions and insert
delays when required. They are typically used late in the transpilation
pipeline, once the circuit is expressed in target-supported operations.

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

Circuit-analysis passes compute properties such as circuit size, depth, or
operation counts. These passes are useful for reporting and for controlling
later conditional stages in a pass manager.

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

Synthesis passes replace high-level objects, unitaries, or arithmetic
operations with lower-level instructions. They are commonly used before or during
basis translation.

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

Post-layout passes refine the choice of physical qubits for an already-routed
circuit. For example, :class:`VF2PostLayout` searches for a lower-error layout
and records it in the property set for :class:`ApplyLayout` to apply.

.. autosummary::
   :toctree: ../stubs/

   VF2PostLayout

Additional Passes
=================

These utility passes support common pass-manager tasks such as checking target
constraints, removing barriers or measurements, tracking fixed points, and
wrapping angles.

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
