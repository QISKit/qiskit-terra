# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
================================================
Quantum Information (:mod:`qiskit.quantum_info`)
================================================

.. currentmodule:: qiskit.quantum_info

.. _quantum_info_operators:

Operators
=========

.. autosummary::
   :toctree: ../stubs/

   Operator
   Pauli
   Clifford
   ScalarOp
   SparseObservable
   SparsePauliOp
   PauliLindbladMap
   QubitSparsePauli
   QubitSparsePauliList
   PhasedQubitSparsePauli
   PhasedQubitSparsePauliList
   CNOTDihedral
   PauliList
   pauli_basis
   get_clifford_gate_names

.. _quantum_info_states:

States
======

.. autosummary::
   :toctree: ../stubs/

   Statevector
   DensityMatrix
   StabilizerState

Channels
========
.. autosummary::
   :toctree: ../stubs/

   Choi
   SuperOp
   Kraus
   Stinespring
   Chi
   PTM

Measures
========

.. autofunction:: average_gate_fidelity
.. autofunction:: process_fidelity
.. autofunction:: gate_error
.. autofunction:: diamond_norm
.. autofunction:: state_fidelity
.. autofunction:: purity
.. autofunction:: concurrence
.. autofunction:: entropy
.. autofunction:: entanglement_of_formation
.. autofunction:: mutual_information

Utility Functions
=================

.. autosummary::
   :toctree: ../stubs/

   Quaternion

.. autofunction:: partial_trace
.. autofunction:: schmidt_decomposition
.. autofunction:: shannon_entropy
.. autofunction:: commutator
.. autofunction:: anti_commutator
.. autofunction:: double_commutator

Random
======

.. autofunction:: random_statevector
.. autofunction:: random_density_matrix
.. autofunction:: random_unitary
.. autofunction:: random_hermitian
.. autofunction:: random_pauli
.. autofunction:: random_clifford
.. autofunction:: random_quantum_channel
.. autofunction:: random_cnotdihedral
.. autofunction:: random_pauli_list

Analysis
=========

.. autofunction:: hellinger_distance
.. autofunction:: hellinger_fidelity

.. autosummary::
   :toctree: ../stubs/

   Z2Symmetries

"""

from qiskit._accelerate.pauli_lindblad_map import (
    QubitSparsePauliList,
    QubitSparsePauli,
    PhasedQubitSparsePauli,
    PhasedQubitSparsePauliList,
    PauliLindbladMap,
)
from qiskit._accelerate.sparse_observable import SparseObservable

from . import analysis, operators, states, random
from .analysis import *
from .operators import *
from .states import *
from .random import *
from .quaternion import Quaternion

__all__ = [
    "PauliLindbladMap",
    "PhasedQubitSparsePauli",
    "PhasedQubitSparsePauliList",
    "Quaternion",
    "QubitSparsePauli",
    "QubitSparsePauliList",
    "SparseObservable",
]
__all__ += analysis.__all__
__all__ += operators.__all__
__all__ += states.__all__
__all__ += random.__all__
