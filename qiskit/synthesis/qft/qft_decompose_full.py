# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
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

from __future__ import annotations
import warnings
import numpy as np
from qiskit.circuit import QuantumCircuit


def synth_qft_full(
    num_qubits: int,
    do_swaps: bool = True,
    approximation_degree: int = 0,
    insert_barriers: bool = False,
    inverse: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a circuit for the Quantum Fourier Transform using all-to-all connectivity.

    .. note::

        With the default value of ``do_swaps = True``, this synthesis algorithm creates a
        circuit that faithfully implements the QFT operation. This circuit contains a sequence
        of swap gates at the end, corresponding to reversing the order of its output qubits.
        In some applications this reversal permutation can be avoided. Setting ``do_swaps = False``
        creates a circuit without this reversal permutation, at the expense that this circuit
        implements the "QFT-with-reversal" instead of QFT. Alternatively, the
        :class:`~.ElidePermutations` transpiler pass is able to remove these swap gates.

    Args:
        num_qubits: The number of qubits on which the Quantum Fourier Transform acts.
        do_swaps: Whether to synthesize the "QFT" or the "QFT-with-reversal" operation.
        approximation_degree: The degree of approximation (0 for no approximation).
            It is possible to implement the QFT approximately by ignoring
            controlled-phase rotations with the angle beneath a threshold. This is discussed
            in more detail in https://arxiv.org/abs/quant-ph/9601018 or
            https://arxiv.org/abs/quant-ph/0403071.
        insert_barriers: If ``True``, barriers are inserted for improved visualization.
        inverse: If ``True``, the inverse Quantum Fourier Transform is constructed.
        name: The name of the circuit.

    Returns:
        A circuit implementing the QFT operation.

    """
    _warn_if_precision_loss(num_qubits - approximation_degree - 1)
    circuit = QuantumCircuit(num_qubits)

    for j in reversed(range(num_qubits)):
        circuit.h(j)
        num_entanglements = max(0, j - max(0, approximation_degree - (num_qubits - j - 1)))
        for k in reversed(range(j - num_entanglements, j)):
            # Use negative exponents so that the angle safely underflows to zero, rather than
            # using a temporary variable that overflows to infinity in the worst case.
            lam = np.pi * (2.0 ** (k - j))
            circuit.cp(lam, j, k)

        if insert_barriers:
            circuit.barrier()

    if do_swaps:
        for i in range(num_qubits // 2):
            circuit.swap(i, num_qubits - i - 1)

    if inverse:
        circuit = circuit.inverse()

    # It is important to set the name after the circuit's generic "inverse" is called,
    # since that will add ``_dg`` to the name.
    if name is not None:
        circuit.name = name

    return circuit


def _warn_if_precision_loss(max_num_entanglements):
    """Issue a warning if constructing the circuit will lose precision.

    If we need an angle smaller than ``pi * 2**-1022``, we start to lose precision by going into
    the subnormal numbers.  We won't lose _all_ precision until an exponent of about 1075, but
    beyond 1022 we're using fractional bits to represent leading zeros.
    """
    if max_num_entanglements > -np.finfo(float).minexp:  # > 1022 for doubles.
        warnings.warn(
            "precision loss in QFT."
            f" The rotation needed to represent {max_num_entanglements} entanglements"
            " is smaller than the smallest normal floating-point number.",
            category=RuntimeWarning,
            stacklevel=4,
        )
