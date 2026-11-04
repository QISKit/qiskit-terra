# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for a QFT circuit.
"""

from qiskit.circuit import QuantumCircuit
from qiskit._accelerate.synthesis.qft import synth_qft_line as _synth_qft_line

from .qft_decompose_full import _warn_if_precision_loss


def synth_qft_line(
    num_qubits: int, do_swaps: bool = True, approximation_degree: int = 0
) -> QuantumCircuit:
    """Construct a circuit for the Quantum Fourier Transform using linear
    neighbor connectivity.

    The construction is based on Fig 2.b in Fowler et al. [1].

    .. note::

        With the default value of ``do_swaps = True``, this synthesis algorithm creates a
        circuit that faithfully implements the QFT operation. When ``do_swaps = False``,
        this synthesis algorithm creates a circuit that corresponds to "QFT-with-reversal":
        applying the QFT and reversing the order of its output qubits.

    Args:
        num_qubits: The number of qubits on which the Quantum Fourier Transform acts.
        approximation_degree: The degree of approximation (0 for no approximation).
            It is possible to implement the QFT approximately by ignoring
            controlled-phase rotations with the angle beneath a threshold. This is discussed
            in more detail in https://arxiv.org/abs/quant-ph/9601018 or
            https://arxiv.org/abs/quant-ph/0403071.
        do_swaps: Whether to synthesize the "QFT" or the "QFT-with-reversal" operation.

    Returns:
        A circuit implementing the QFT operation.

    References:
        1. A. G. Fowler, S. J. Devitt, and L. C. L. Hollenberg,
           *Implementation of Shor's algorithm on a linear nearest neighbour qubit array*,
           Quantum Info. Comput. 4, 4 (July 2004), 237–251.
           `arXiv:quant-ph/0402196 [quant-ph] <https://arxiv.org/abs/quant-ph/0402196>`_
    """
    _warn_if_precision_loss(num_qubits - approximation_degree - 1)

    return QuantumCircuit._from_circuit_data(
        # From rust
        _synth_qft_line(num_qubits, do_swaps, approximation_degree),
        legacy_qubits=True,
    )
