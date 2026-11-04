# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for the Clifford class for all-to-all architecture.
"""

from __future__ import annotations
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.synthesis.clifford.clifford_decompose_ag import synth_clifford_ag
from qiskit.synthesis.clifford.clifford_decompose_bm import synth_clifford_bm
from qiskit.synthesis.clifford.clifford_decompose_greedy import synth_clifford_greedy


def synth_clifford_full(clifford: Clifford, method: str | None = None) -> QuantumCircuit:
    r"""Decompose a :class:`.Clifford` operator into a :class:`.QuantumCircuit`.

    For :math:`N \leq 3` qubits this is based on optimal CX-cost decomposition
    from reference [1]. For :math:`N > 3` qubits this is done using the general
    non-optimal greedy compilation routine from reference [3],
    which typically yields better CX cost compared to the AG method in [2].

    Args:
        clifford: A Clifford operator.
        method:  a synthesis method (``'AG'`` or ``'greedy'``).
             If set this overrides optimal decomposition for :math:`N \leq 3` qubits.

    Returns:
        A circuit implementation of the Clifford.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_

        2. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_

        3. Sergey Bravyi, Shaohan Hu, Dmitri Maslov, Ruslan Shaydulin,
           *Clifford Circuit Optimization with Templates and Symbolic Pauli Gates*,
           `arXiv:2105.02291 [quant-ph] <https://arxiv.org/abs/2105.02291>`_
    """
    num_qubits = clifford.num_qubits

    if method == "AG":
        return synth_clifford_ag(clifford)

    if method == "greedy":
        return synth_clifford_greedy(clifford)

    if num_qubits <= 3:
        return synth_clifford_bm(clifford)

    return synth_clifford_greedy(clifford)
