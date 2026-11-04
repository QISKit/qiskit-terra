# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Instantaneous quantum polynomial circuit."""

from __future__ import annotations
from collections.abc import Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.utils.deprecation import deprecate_func
from qiskit._accelerate.circuit_library import py_iqp, py_random_iqp


class IQP(QuantumCircuit):
    r"""Instantaneous quantum polynomial (IQP) circuit.

    The circuit consists of a column of Hadamard gates,
    a column of powers of T gates,
    a sequence of powers of CS gates (up to
    :math:`\frac{n^2-n}{2}` of them),
    and a final column of Hadamard gates, as introduced in [1].

    The circuit is parameterized by an n x n interactions matrix.
    The powers of each T gate are given by the diagonal elements
    of the interactions matrix. The powers of the CS gates are
    given by the upper triangle of the interactions matrix.

    Reference Circuit:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import IQP
       A = [[6, 5, 3], [5, 4, 5], [3, 5, 1]]
       circuit = IQP(A)
       circuit.draw('mpl')

    Expanded Circuit:

    .. plot::
        :alt: Diagram illustrating the previously described circuit.

        from qiskit.circuit.library import IQP
        from qiskit.visualization.library import _generate_circuit_library_visualization
        A = [[6, 5, 3], [5, 4, 5], [3, 5, 1]]
        circuit = IQP(A)
        _generate_circuit_library_visualization(circuit.decompose())

    References:

    [1] M. J. Bremner et al. Average-case complexity versus approximate
    simulation of commuting quantum computations,
    Phys. Rev. Lett. 117, 080501 (2016).
    `arXiv:1504.07999 <https://arxiv.org/abs/1504.07999>`_
    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use the qiskit.circuit.library.iqp function instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(self, interactions: list | np.ndarray) -> None:
        """Create IQP circuit.

        Args:
            interactions: input n-by-n symmetric matrix.

        Raises:
            CircuitError: if the input is not a symmetric matrix.
        """
        circuit = iqp(interactions)
        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)


def iqp(
    interactions: Sequence[Sequence[int]],
) -> QuantumCircuit:
    r"""Instantaneous quantum polynomial time (IQP) circuit.

    The circuit consists of a column of Hadamard gates, a column of powers of T gates,
    a sequence of powers of CS gates (up to :math:`\frac{n^2-n}{2}` of them), and a final column of
    Hadamard gates, as introduced in [1].

    The circuit is parameterized by an :math:`n \times n` interactions matrix. The powers of each
    T gate are given by the diagonal elements of the interactions matrix. The powers of the CS gates
    are given by the upper triangle of the interactions matrix.

    Reference Circuit:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import iqp
       A = [[6, 5, 3], [5, 4, 5], [3, 5, 1]]
       circuit = iqp(A)
       circuit.draw("mpl")

    Expanded Circuit:

        .. plot::
           :alt: Diagram illustrating the previously described circuit.

           from qiskit.circuit.library import iqp
           from qiskit.visualization.library import _generate_circuit_library_visualization
           A = [[6, 5, 3], [5, 4, 5], [3, 5, 1]]
           circuit = iqp(A)
           _generate_circuit_library_visualization(circuit)

    References:

    [1] M. J. Bremner et al. Average-case complexity versus approximate
    simulation of commuting quantum computations,
    Phys. Rev. Lett. 117, 080501 (2016).
    `arXiv:1504.07999 <https://arxiv.org/abs/1504.07999>`_

    Args:
        interactions: The interactions as symmetric square matrix. If ``None``, then the
            ``num_qubits`` argument must be set and a random IQP circuit will be generated.

    Returns:
        An IQP circuit.
    """
    # if no interactions are given, generate them
    num_qubits = len(interactions)
    interactions = np.asarray(interactions).astype(np.int64)

    # set the label -- if the number of qubits is too large, do not show the interactions matrix
    if num_qubits < 5 and interactions is not None:
        label = np.array_str(interactions)
        name = "iqp:" + label.replace("\n", ";")
    else:
        name = "iqp"

    circuit = QuantumCircuit._from_circuit_data(py_iqp(interactions), legacy_qubits=True)
    circuit.name = name
    return circuit


def random_iqp(
    num_qubits: int,
    seed: int | None = None,
) -> QuantumCircuit:
    r"""A random instantaneous quantum polynomial time (IQP) circuit.

    See :func:`iqp` for more details on the IQP circuit.

    Example:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:

       from qiskit.circuit.library import random_iqp

       circuit = random_iqp(3)
       circuit.draw("mpl")

    Args:
        num_qubits: The number of qubits in the circuit.
        seed: A seed for the random number generator, in case the interactions matrix is
            randomly generated.

    Returns:
        An IQP circuit.
    """
    # set the label -- if the number of qubits is too large, do not show the interactions matrix
    circuit = QuantumCircuit._from_circuit_data(py_random_iqp(num_qubits, seed), legacy_qubits=True)
    circuit.name = "iqp"
    return circuit
