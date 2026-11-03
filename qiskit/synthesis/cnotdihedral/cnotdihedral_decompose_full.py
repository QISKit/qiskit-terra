# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for the CNOTDihedral class for all-to-all connectivity.
"""

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import CNOTDihedral

from qiskit._accelerate.synthesis.cnotdihedral import (
    synth_cnotdihedral_full as synth_cnotdihedral_full_inner,
)

from qiskit.synthesis.cnotdihedral.cnotdihedral_decompose_general import _dihedral_parts


def synth_cnotdihedral_full(elem: CNOTDihedral) -> QuantumCircuit:
    r"""Decompose a :class:`.CNOTDihedral` element into a :class:`.QuantumCircuit`.

    For :math:`N \leq 2` qubits this is based on optimal CX-cost decomposition from reference [1].
    For :math:`N > 2` qubits this is done using the general non-optimal compilation
    routine from reference [2].

    Args:
        elem: A :class:`.CNOTDihedral` element.

    Returns:
        A circuit implementation of the :class:`.CNOTDihedral` element.

    References:
        1. Shelly Garion and Andrew W. Cross, *Synthesis of CNOT-Dihedral circuits
           with optimal number of two qubit gates*, `Quantum 4(369), 2020
           <https://quantum-journal.org/papers/q-2020-12-07-369/>`_
        2. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
           *Scalable randomized benchmarking of non-Clifford gates*,
           npj Quantum Inf 2, 16012 (2016).
    """
    return QuantumCircuit._from_circuit_data(
        synth_cnotdihedral_full_inner(*_dihedral_parts(elem)), legacy_qubits=True
    )
