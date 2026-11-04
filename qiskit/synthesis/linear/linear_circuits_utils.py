# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for handling linear reversible circuits."""

import copy
from collections.abc import Callable
import numpy as np

from qiskit.circuit import QuantumCircuit, CircuitError
from qiskit.exceptions import QiskitError
from qiskit._accelerate.synthesis.linear import calc_inverse_matrix, check_invertible_binary_matrix


def transpose_cx_circ(qc: QuantumCircuit):
    """Takes a circuit having only CX gates, and calculates its transpose.
    This is done by recursively replacing CX(i, j) with CX(j, i) in all instructions.

    Args:
        qc: a :class:`.QuantumCircuit` containing only CX gates.

    Returns:
        QuantumCircuit: the transposed circuit.

    Raises:
        CircuitError: if qc has a non-CX gate.
    """
    transposed_circ = QuantumCircuit(qc.qubits, qc.clbits, name=qc.name + "_transpose")
    for instruction in reversed(qc.data):
        if instruction.name != "cx":
            raise CircuitError("The circuit contains non-CX gates.")
        transposed_circ._append(instruction.replace(qubits=reversed(instruction.qubits)))
    return transposed_circ


def optimize_cx_4_options(function: Callable, mat: np.ndarray, optimize_count: bool = True):
    """Get the best implementation of a circuit implementing a binary invertible matrix M,
    by considering all four options: M,M^(-1),M^T,M^(-1)^T.
    Optimizing either the CX count or the depth.

    Args:
        function: the synthesis function.
        mat: a binary invertible matrix.
        optimize_count: True if the number of CX gates is optimized, False if the depth is optimized.

    Returns:
        QuantumCircuit: an optimized :class:`.QuantumCircuit`, has the best depth or CX count of
            the four options.

    Raises:
        QiskitError: if mat is not an invertible matrix.
    """
    if not check_invertible_binary_matrix(mat):
        raise QiskitError("The matrix is not invertible.")

    qc = function(mat)
    best_qc = qc
    best_depth = qc.depth()
    best_count = qc.count_ops()["cx"]

    for i in range(1, 4):
        mat_cpy = copy.deepcopy(mat)
        # i=1 inverse, i=2 transpose, i=3 transpose and inverse
        if i == 1:
            mat_cpy = calc_inverse_matrix(mat_cpy)
            qc = function(mat_cpy)
            qc = qc.inverse()
        elif i == 2:
            mat_cpy = np.transpose(mat_cpy)
            qc = function(mat_cpy)
            qc = transpose_cx_circ(qc)
        elif i == 3:
            mat_cpy = calc_inverse_matrix(np.transpose(mat_cpy))
            qc = function(mat_cpy)
            qc = transpose_cx_circ(qc)
            qc = qc.inverse()

        new_depth = qc.depth()
        new_count = qc.count_ops()["cx"]
        # Prioritize count, and if it has the same count, then also consider depth
        better_count = (optimize_count and best_count > new_count) or (
            not optimize_count and best_depth == new_depth and best_count > new_count
        )
        # Prioritize depth, and if it has the same depth, then also consider count
        better_depth = (not optimize_count and best_depth > new_depth) or (
            optimize_count and best_count == new_count and best_depth > new_depth
        )

        if better_count or better_depth:
            best_count = new_count
            best_depth = new_depth
            best_qc = qc

    return best_qc


def check_lnn_connectivity(qc: QuantumCircuit) -> bool:
    """Check that the synthesized circuit qc fits linear nearest neighbor connectivity.

    Args:
        qc: a :class:`.QuantumCircuit` containing only CX and single qubit gates.

    Returns:
        bool: True if the circuit has linear nearest neighbor connectivity.

    Raises:
        CircuitError: if qc has a non-CX two-qubit gate.
    """
    for instruction in qc.data:
        if instruction.operation.num_qubits > 1:
            if instruction.operation.name == "cx":
                q0 = qc.find_bit(instruction.qubits[0]).index
                q1 = qc.find_bit(instruction.qubits[1]).index
                dist = abs(q0 - q1)
                if dist != 1:
                    return False
            else:
                raise CircuitError("The circuit has two-qubits gates different than CX.")
    return True
