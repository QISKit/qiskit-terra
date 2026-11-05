# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=========================================================================
Generating Random Quantum Information (:mod:`qiskit.quantum_info.random`)
=========================================================================

.. currentmodule:: qiskit.quantum_info.random

Overview
========

The :mod:`qiskit.quantum_info.random` module provides functions for generating
random quantum states, operators, and channels. These are useful for testing
and benchmarking.

Random States
=============

.. autofunction:: random_statevector
.. autofunction:: random_density_matrix

Random Operators
================

.. autofunction:: random_unitary
.. autofunction:: random_hermitian
.. autofunction:: random_pauli
.. autofunction:: random_pauli_list
.. autofunction:: random_clifford
.. autofunction:: random_cnotdihedral

Random Channels
===============

.. autofunction:: random_quantum_channel
"""

from qiskit.quantum_info.operators.random import (
    random_clifford,
    random_cnotdihedral,
    random_hermitian,
    random_pauli,
    random_pauli_list,
    random_quantum_channel,
    random_unitary,
)
from qiskit.quantum_info.states.random import random_density_matrix, random_statevector

__all__ = [
    "random_clifford",
    "random_cnotdihedral",
    "random_density_matrix",
    "random_hermitian",
    "random_pauli",
    "random_pauli_list",
    "random_quantum_channel",
    "random_statevector",
    "random_unitary",
]
