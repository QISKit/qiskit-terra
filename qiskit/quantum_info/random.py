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

"""Methods for generating random quantum information objects."""


# The import path via qiskit.quantum_info.random is not API-documented,
# yet it existed for a long time and several packages rely on this import path.
# Since it also mirrors the API-documented ``qiskit.circuit.random``, we keep
# the re-export available here.

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
