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
Additional utilities for Operators.
"""

from __future__ import annotations

from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal


def _equal_with_ancillas(
    op1: Operator,
    op2: Operator,
    ancilla_qubits: list[int],
    ignore_phase: bool = False,
    rtol: float | None = None,
    atol: float | None = None,
) -> bool:
    r"""Test if two Operators are equal on the subspace where ancilla qubits
    are :math:`|0\rangle`.

    Args:
        op1 (Operator): an operator object.
        op2 (Operator): an operator object.
        ancilla_qubits (list[int]): a list of clean ancilla qubits.
        ignore_phase (bool): ignore complex-phase difference between matrices.
        rtol (float): relative tolerance value for comparison.
        atol (float): absolute tolerance value for comparison.

    Returns:
        bool: True iff operators are equal up to clean ancilla qubits.
    """
    if op1.dim != op2.dim:
        return False

    if atol is None:
        atol = op1.atol
    if rtol is None:
        rtol = op1.rtol

    num_qubits = op1._op_shape._num_qargs_l
    num_non_ancillas = num_qubits - len(ancilla_qubits)

    # Find a permutation that moves all ancilla qubits to the back
    pattern = []
    ancillas = []
    for q in range(num_qubits):
        if q not in ancilla_qubits:
            pattern.append(q)
        else:
            ancillas.append(q)
    pattern = pattern + ancillas

    # Apply this permutation to both operators
    permuted1 = op1.apply_permutation(pattern)
    permuted2 = op2.apply_permutation(pattern)

    # Restrict to the subspace where ancillas are 0
    restricted1 = permuted1.data[: 2**num_non_ancillas, : 2**num_qubits]
    restricted2 = permuted2.data[: 2**num_non_ancillas, : 2**num_qubits]

    return matrix_equal(
        restricted1, restricted2.data, ignore_phase=ignore_phase, rtol=rtol, atol=atol
    )
