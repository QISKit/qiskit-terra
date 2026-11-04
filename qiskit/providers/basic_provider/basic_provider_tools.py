# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contains functions used by the basic provider simulators."""
from __future__ import annotations

from string import ascii_uppercase, ascii_lowercase

import numpy as np

import qiskit.circuit.library.standard_gates as gates
from qiskit.exceptions import QiskitError

# Single qubit gates supported by ``single_gate_params``.
SINGLE_QUBIT_GATES = {
    "U": gates.UGate,
    "u": gates.UGate,
    "u1": gates.U1Gate,
    "u2": gates.U2Gate,
    "u3": gates.U3Gate,
    "h": gates.HGate,
    "p": gates.PhaseGate,
    "s": gates.SGate,
    "sdg": gates.SdgGate,
    "sx": gates.SXGate,
    "sxdg": gates.SXdgGate,
    "t": gates.TGate,
    "tdg": gates.TdgGate,
    "x": gates.XGate,
    "y": gates.YGate,
    "z": gates.ZGate,
    "id": gates.IGate,
    "i": gates.IGate,
    "r": gates.RGate,
    "rx": gates.RXGate,
    "ry": gates.RYGate,
    "rz": gates.RZGate,
}


def single_gate_matrix(gate: str, params: list[float] | None = None) -> np.ndarray:
    """Get the matrix for a single qubit.

    Args:
        gate: the single qubit gate name
        params: the operation parameters op['params']
    Returns:
        array: A numpy array representing the matrix
    Raises:
        QiskitError: If a gate outside the supported set is passed in for the
            ``Gate`` argument.
    """
    if params is None:
        params = []
    if gate in SINGLE_QUBIT_GATES:
        gc = SINGLE_QUBIT_GATES[gate]
    else:
        raise QiskitError(f"Gate is not a valid basis gate for this simulator: {gate}")

    return gc(*params).to_matrix()


# Two qubit gates WITHOUT parameters: name -> matrix
TWO_QUBIT_GATES = {
    "CX": gates.CXGate().to_matrix(),
    "cx": gates.CXGate().to_matrix(),
    "ecr": gates.ECRGate().to_matrix(),
    "cy": gates.CYGate().to_matrix(),
    "cz": gates.CZGate().to_matrix(),
    "swap": gates.SwapGate().to_matrix(),
    "iswap": gates.iSwapGate().to_matrix(),
    "ch": gates.CHGate().to_matrix(),
    "cs": gates.CSGate().to_matrix(),
    "csdg": gates.CSdgGate().to_matrix(),
    "csx": gates.CSXGate().to_matrix(),
    "dcx": gates.DCXGate().to_matrix(),
}

# Two qubit gates WITH parameters: name -> class
TWO_QUBIT_GATES_WITH_PARAMETERS = {
    "cp": gates.CPhaseGate,
    "crx": gates.CRXGate,
    "cry": gates.CRYGate,
    "crz": gates.CRZGate,
    "cu": gates.CUGate,
    "cu1": gates.CU1Gate,
    "cu3": gates.CU3Gate,
    "rxx": gates.RXXGate,
    "ryy": gates.RYYGate,
    "rzz": gates.RZZGate,
    "rzx": gates.RZXGate,
    "xx_minus_yy": gates.XXMinusYYGate,
    "xx_plus_yy": gates.XXPlusYYGate,
}


# Three qubit gates: name -> matrix
THREE_QUBIT_GATES = {
    "ccx": gates.CCXGate().to_matrix(),
    "ccz": gates.CCZGate().to_matrix(),
    "rccx": gates.RCCXGate().to_matrix(),
    "cswap": gates.CSwapGate().to_matrix(),
}


def einsum_matmul_index(gate_indices: list[int], number_of_qubits: int) -> str:
    """Return the index string for Numpy.einsum matrix-matrix multiplication.

    The returned indices are to perform a matrix multiplication A.B where
    the matrix A is an M-qubit matrix, matrix B is an N-qubit matrix, and
    M <= N, and identity matrices are implied on the subsystems where A has no
    support on B.

    Args:
        gate_indices (list[int]): the indices of the right matrix subsystems
                                   to contract with the left matrix.
        number_of_qubits (int): the total number of qubits for the right matrix.

    Returns:
        str: An indices string for the Numpy.einsum function.
    """

    mat_l, mat_r, tens_lin, tens_lout = _einsum_matmul_index_helper(gate_indices, number_of_qubits)

    # Right indices for the N-qubit input and output tensor
    tens_r = ascii_uppercase[:number_of_qubits]

    # Combine indices into matrix multiplication string format
    # for numpy.einsum function
    return f"{mat_l}{mat_r}, {tens_lin}{tens_r}->{tens_lout}{tens_r}"


def einsum_vecmul_index(gate_indices: list[int], number_of_qubits: int) -> str:
    """Return the index string for Numpy.einsum matrix-vector multiplication.

    The returned indices are to perform a matrix multiplication A.v where
    the matrix A is an M-qubit matrix, vector v is an N-qubit vector, and
    M <= N, and identity matrices are implied on the subsystems where A has no
    support on v.

    Args:
        gate_indices (list[int]): the indices of the right matrix subsystems
                                  to contract with the left matrix.
        number_of_qubits (int): the total number of qubits for the right matrix.

    Returns:
        str: An indices string for the Numpy.einsum function.
    """

    mat_l, mat_r, tens_lin, tens_lout = _einsum_matmul_index_helper(gate_indices, number_of_qubits)

    # Combine indices into matrix multiplication string format
    # for numpy.einsum function
    return f"{mat_l}{mat_r}, {tens_lin}->{tens_lout}"


def _einsum_matmul_index_helper(
    gate_indices: list[int], number_of_qubits: int
) -> tuple[str, str, str, str]:
    """Return the index string for Numpy.einsum matrix multiplication.

    The returned indices are to perform a matrix multiplication A.v where
    the matrix A is an M-qubit matrix, matrix v is an N-qubit vector, and
    M <= N, and identity matrices are implied on the subsystems where A has no
    support on v.

    Args:
        gate_indices (list[int]): the indices of the right matrix subsystems
                                   to contract with the left matrix.
        number_of_qubits (int): the total number of qubits for the right matrix.

    Returns:
        tuple: (mat_left, mat_right, tens_in, tens_out) of index strings for
        that may be combined into a Numpy.einsum function string.

    Raises:
        QiskitError: if the total number of qubits plus the number of
        contracted indices is greater than 26.
    """

    # Since we use ASCII alphabet for einsum index labels we are limited
    # to 26 total free left (lowercase) and 26 right (uppercase) indexes.
    # The rank of the contracted tensor reduces this as we need to use that
    # many characters for the contracted indices
    if len(gate_indices) + number_of_qubits > 26:
        raise QiskitError("Total number of free indexes limited to 26")

    # Indices for N-qubit input tensor
    tens_in = ascii_lowercase[:number_of_qubits]

    # Indices for the N-qubit output tensor
    tens_out = list(tens_in)

    # Left and right indices for the M-qubit multiplying tensor
    mat_left = ""
    mat_right = ""

    # Update left indices for mat and output
    for pos, idx in enumerate(reversed(gate_indices)):
        mat_left += ascii_lowercase[-1 - pos]
        mat_right += tens_in[-1 - idx]
        tens_out[-1 - idx] = ascii_lowercase[-1 - pos]
    tens_out = "".join(tens_out)

    # Combine indices into matrix multiplication string format
    # for numpy.einsum function
    return mat_left, mat_right, tens_in, tens_out
