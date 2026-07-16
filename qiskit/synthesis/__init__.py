# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
===========================================
Circuit Synthesis (:mod:`qiskit.synthesis`)
===========================================

.. currentmodule:: qiskit.synthesis

.. _evolution_synthesis:

Evolution Synthesis
===================

.. autosummary::
   :toctree: ../stubs/

   EvolutionSynthesis
   ProductFormula
   LieTrotter
   SuzukiTrotter
   MatrixExponential
   QDrift

Linear Function Synthesis
=========================

.. autofunction:: synth_cnot_count_full_pmh
.. autofunction:: synth_cnot_depth_line_kms

Linear-Phase Synthesis
======================

.. autofunction:: synth_cz_depth_line_mr
.. autofunction:: synth_cx_cz_depth_line_my
.. autofunction:: synth_cnot_phase_aam

Permutation Synthesis
=====================

.. autofunction:: synth_permutation_depth_lnn_kms
.. autofunction:: synth_permutation_basic
.. autofunction:: synth_permutation_acg
.. autofunction:: synth_permutation_reverse_lnn_kms

Clifford Synthesis
==================

.. autofunction:: synth_clifford_full
.. autofunction:: synth_clifford_ag
.. autofunction:: synth_clifford_bm
.. autofunction:: synth_clifford_greedy
.. autofunction:: synth_clifford_layers
.. autofunction:: synth_clifford_depth_lnn

CNOTDihedral Synthesis
======================

.. autofunction:: synth_cnotdihedral_full
.. autofunction:: synth_cnotdihedral_two_qubits
.. autofunction:: synth_cnotdihedral_general

Stabilizer State Synthesis
==========================

.. autofunction:: synth_stabilizer_layers
.. autofunction:: synth_stabilizer_depth_lnn
.. autofunction:: synth_circuit_from_stabilizers

Discrete Basis Synthesis
========================

.. autosummary::
   :toctree: ../stubs/

   SolovayKitaevDecomposition

.. autofunction:: gridsynth_rz
.. autofunction:: gridsynth_unitary
.. autofunction:: generate_basic_approximations

Basis Change Synthesis
======================

.. autofunction:: synth_qft_line
.. autofunction:: synth_qft_full

Unitary Synthesis
=================

Decomposition of general :math:`2^n \times 2^n` unitary matrices for any number of qubits.

.. autofunction:: qs_decomposition

The Approximate Quantum Compiler is available as the module :mod:`qiskit.synthesis.unitary.aqc`.

One-Qubit Synthesis
===================

.. autosummary::
   :toctree: ../stubs/

   OneQubitEulerDecomposer

Two-Qubit Synthesis
===================

.. autosummary::
   :toctree: ../stubs/

   TwoQubitBasisDecomposer
   XXDecomposer
   TwoQubitWeylDecomposition
   TwoQubitControlledUDecomposer

.. autofunction:: two_qubit_cnot_decompose

Multi Controlled Synthesis
==========================

.. autofunction:: synth_mcmt_vchain
.. autofunction:: synth_mcmt_xgate
.. autofunction:: synth_mcx_1_clean_kg24
.. autofunction:: synth_mcx_1_dirty_kg24
.. autofunction:: synth_mcx_2_clean_kg24
.. autofunction:: synth_mcx_2_dirty_kg24
.. autofunction:: synth_mcx_n_dirty_i15
.. autofunction:: synth_mcx_n_clean_m15
.. autofunction:: synth_mcx_1_clean_b95
.. autofunction:: synth_mcx_noaux_v24
.. autofunction:: synth_mcx_noaux_hp24
.. autofunction:: synth_mcx_gray_code
.. autofunction:: synth_c3x
.. autofunction:: synth_c4x

Binary Arithmetic Synthesis
===========================

Adders
------

.. autofunction:: adder_qft_d00
.. autofunction:: adder_ripple_c04
.. autofunction:: adder_ripple_v95
.. autofunction:: adder_ripple_r25
.. autofunction:: adder_modular_v17

Multipliers
-----------

.. autofunction:: multiplier_cumulative_h18
.. autofunction:: multiplier_qft_r17

Sums
----

.. autofunction:: synth_weighted_sum_carry


Unary Arithmetic Synthesis
==========================

Integer comparators
-------------------

.. autofunction:: synth_integer_comparator_2s
.. autofunction:: synth_integer_comparator_greedy

"""


from .evolution import *
from .permutation import *
from .linear import *
from .linear_phase import *
from .clifford import *
from .cnotdihedral import *
from .stabilizer import *
from .discrete_basis import *
from .qft import *
from .unitary.qsd import qs_decomposition
from .unitary import aqc
from .one_qubit import OneQubitEulerDecomposer
from .two_qubit import *
from .multi_controlled import *
from .arithmetic import *

from . import (
    evolution,
    permutation,
    linear,
    linear_phase,
    clifford,
    cnotdihedral,
    stabilizer,
    discrete_basis,
    qft,
    two_qubit,
    multi_controlled,
    arithmetic,
)

__all__ = [
    "OneQubitEulerDecomposer",
    "aqc",
    "arithmetic",
    "clifford",
    "cnotdihedral",
    "discrete_basis",
    "evolution",
    "linear",
    "linear_phase",
    "multi_controlled",
    "permutation",
    "qft",
    "qs_decomposition",
    "stabilizer",
    "two_qubit",
]
__all__ += evolution.__all__
__all__ += permutation.__all__
__all__ += linear.__all__
__all__ += linear_phase.__all__
__all__ += clifford.__all__
__all__ += cnotdihedral.__all__
__all__ += stabilizer.__all__
__all__ += discrete_basis.__all__
__all__ += qft.__all__
__all__ += two_qubit.__all__
__all__ += multi_controlled.__all__
__all__ += arithmetic.__all__
