# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
PauliList utility functions.
"""

from __future__ import annotations

from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList


def pauli_basis(num_qubits: int, weight: bool = False) -> PauliList:
    """Return the ordered PauliList for the n-qubit Pauli basis.

    Args:
        num_qubits (int): number of qubits
        weight (bool): if True optionally return the basis sorted by Pauli weight
                       rather than lexicographic order (Default: False)

    Returns:
        PauliList: the Paulis for the basis
    """
    pauli_1q = PauliList(["I", "X", "Y", "Z"])
    if num_qubits == 1:
        return pauli_1q
    pauli = pauli_1q
    for _ in range(num_qubits - 1):
        pauli = pauli_1q.tensor(pauli)
    if weight:
        return pauli.sort(weight=True)
    return pauli
