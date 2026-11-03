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
Circuit synthesis for the CNOTDihedral class.
"""

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import CNOTDihedral

from qiskit._accelerate.synthesis.cnotdihedral import (
    synth_cnotdihedral_two_qubits as synth_cnotdihedral_two_qubits_inner,
)

from qiskit.synthesis.cnotdihedral.cnotdihedral_decompose_general import _dihedral_parts


def synth_cnotdihedral_two_qubits(elem: CNOTDihedral) -> QuantumCircuit:
    """Decompose a :class:`.CNOTDihedral` element on a single qubit and two
    qubits into a :class:`.QuantumCircuit`.
    This decomposition has an optimal number of :class:`.CXGate`\\ s.

    Args:
        elem: A :class:`.CNOTDihedral` element.

    Returns:
        A circuit implementation of the :class:`.CNOTDihedral` element.

    Raises:
        QiskitError: if the element is not 1-qubit or 2-qubit :class:`.CNOTDihedral`.

    References:
        1. Shelly Garion and Andrew W. Cross, *On the structure of the CNOT-Dihedral group*,
           `arXiv:2006.12042 [quant-ph] <https://arxiv.org/abs/2006.12042>`_
    """
    return QuantumCircuit._from_circuit_data(
        synth_cnotdihedral_two_qubits_inner(*_dihedral_parts(elem)), legacy_qubits=True
    )
