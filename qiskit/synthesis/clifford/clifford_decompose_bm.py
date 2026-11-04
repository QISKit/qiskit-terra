# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for 2-qubit and 3-qubit Cliffords based on Bravyi & Maslov
decomposition.
"""

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford

from qiskit._accelerate.synthesis.clifford import (
    synth_clifford_bm as synth_clifford_bm_inner,
)


def synth_clifford_bm(clifford: Clifford) -> QuantumCircuit:
    """Optimal CX-cost decomposition of a :class:`.Clifford` operator on 2 qubits
    or 3 qubits into a :class:`.QuantumCircuit` based on the Bravyi-Maslov method [1].

    Args:
        clifford: A Clifford operator.

    Returns:
        A circuit implementation of the Clifford.

    Raises:
        QiskitError: if Clifford is on more than 3 qubits.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """
    circuit = QuantumCircuit._from_circuit_data(
        synth_clifford_bm_inner(clifford.tableau.astype(bool)),
        legacy_qubits=True,
        name=str(clifford),
    )
    return circuit
