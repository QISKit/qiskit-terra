# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Synthesize a single-qubit gate using Ross-Selinger algorithm."""


from __future__ import annotations

import numpy as np

from qiskit.circuit import QuantumCircuit

from qiskit._accelerate.ross_selinger import gridsynth_rz as gridsynth_rz_rs
from qiskit._accelerate.ross_selinger import gridsynth_unitary as gridsynth_unitary_rs


def gridsynth_rz(angle: float, epsilon: float = 1e-10) -> QuantumCircuit:
    """
    Approximate RZ-rotation using the Ross-Selinger algorithm.

    The algorithm is described in [1]. The source code (in Rust) is available at
    https://github.com/qiskit-community/rsgridsynth.

    Args:
        angle: Specifies the angle of the RZ-rotation.
        epsilon: The allowed approximation error.

    Returns:
        A one-qubit circuit approximating ``RZ(angle)``.

    References:

    [1] Neil J. Ross, Peter Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations,
        `arXiv:1403.2975 <https://arxiv.org/pdf/1403.2975>`_

    """

    approximate_circuit_data = gridsynth_rz_rs(angle, epsilon)
    approximate_circuit = QuantumCircuit._from_circuit_data(
        approximate_circuit_data, legacy_qubits=True
    )
    return approximate_circuit


def gridsynth_unitary(matrix: np.ndarray, epsilon: float = 1e-10) -> QuantumCircuit:
    """
    Approximate a 1-qubit unitary matrix using the Ross-Selinger algorithm.

    The algorithm is described in [1]. The source code (in Rust) is available at
    https://github.com/qiskit-community/rsgridsynth.

    Args:
        matrix: A :math:`2\times 2` unitary matrix.
        epsilon: The allowed approximation error.

    Returns:
        A one-qubit circuit approximating ``matrix``.

    References:

    [1] Neil J. Ross, Peter Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations,
        `arXiv:1403.2975 <https://arxiv.org/pdf/1403.2975>`_

    """

    approximate_circuit_data = gridsynth_unitary_rs(matrix, epsilon)
    approximate_circuit = QuantumCircuit._from_circuit_data(
        approximate_circuit_data, legacy_qubits=True
    )
    return approximate_circuit
