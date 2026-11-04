# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Transformations between QuantumChannel representations.
"""

from __future__ import annotations
import math
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT


def _transform_rep(input_rep, output_rep, data, input_dim, output_dim):
    """Transform a QuantumChannel between representation."""
    if input_rep == output_rep:
        return data
    if output_rep == "Choi":
        return _to_choi(input_rep, data, input_dim, output_dim)
    if output_rep == "Operator":
        return _to_operator(input_rep, data, input_dim, output_dim)
    if output_rep == "SuperOp":
        return _to_superop(input_rep, data, input_dim, output_dim)
    if output_rep == "Kraus":
        return _to_kraus(input_rep, data, input_dim, output_dim)
    if output_rep == "Chi":
        return _to_chi(input_rep, data, input_dim, output_dim)
    if output_rep == "PTM":
        return _to_ptm(input_rep, data, input_dim, output_dim)
    if output_rep == "Stinespring":
        return _to_stinespring(input_rep, data, input_dim, output_dim)
    raise QiskitError(f"Invalid QuantumChannel {output_rep}")


def _to_choi(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Choi representation."""
    if rep == "Choi":
        return data
    if rep == "Operator":
        return _from_operator("Choi", data, input_dim, output_dim)
    if rep == "SuperOp":
        return _superop_to_choi(data, input_dim, output_dim)
    if rep == "Kraus":
        return _kraus_to_choi(data)
    if rep == "Chi":
        return _chi_to_choi(data, input_dim)
    if rep == "PTM":
        data = _ptm_to_superop(data, input_dim)
        return _superop_to_choi(data, input_dim, output_dim)
    if rep == "Stinespring":
        return _stinespring_to_choi(data, input_dim, output_dim)
    raise QiskitError(f"Invalid QuantumChannel {rep}")


def _to_superop(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the SuperOp representation."""
    if rep == "SuperOp":
        return data
    if rep == "Operator":
        return _from_operator("SuperOp", data, input_dim, output_dim)
    if rep == "Choi":
        return _choi_to_superop(data, input_dim, output_dim)
    if rep == "Kraus":
        return _kraus_to_superop(data)
    if rep == "Chi":
        data = _chi_to_choi(data, input_dim)
        return _choi_to_superop(data, input_dim, output_dim)
    if rep == "PTM":
        return _ptm_to_superop(data, input_dim)
    if rep == "Stinespring":
        return _stinespring_to_superop(data, input_dim, output_dim)
    raise QiskitError(f"Invalid QuantumChannel {rep}")


def _to_kraus(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Kraus representation."""
    if rep == "Kraus":
        return data
    if rep == "Stinespring":
        return _stinespring_to_kraus(data, output_dim)
    if rep == "Operator":
        return _from_operator("Kraus", data, input_dim, output_dim)
    # Convert via Choi and Kraus
    if rep != "Choi":
        data = _to_choi(rep, data, input_dim, output_dim)
    return _choi_to_kraus(data, input_dim, output_dim)


def _to_chi(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Chi representation."""
    if rep == "Chi":
        return data
    # Check valid n-qubit input
    _check_nqubit_dim(input_dim, output_dim)
    if rep == "Operator":
        return _from_operator("Chi", data, input_dim, output_dim)
    # Convert via Choi representation
    if rep != "Choi":
        data = _to_choi(rep, data, input_dim, output_dim)
    return _choi_to_chi(data, input_dim)


def _to_ptm(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the PTM representation."""
    if rep == "PTM":
        return data
    # Check valid n-qubit input
    _check_nqubit_dim(input_dim, output_dim)
    if rep == "Operator":
        return _from_operator("PTM", data, input_dim, output_dim)
    # Convert via Superoperator representation
    if rep != "SuperOp":
        data = _to_superop(rep, data, input_dim, output_dim)
    return _superop_to_ptm(data, input_dim)


def _to_stinespring(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Stinespring representation."""
    if rep == "Stinespring":
        return data
    if rep == "Operator":
        return _from_operator("Stinespring", data, input_dim, output_dim)
    # Convert via Superoperator representation
    if rep != "Kraus":
        data = _to_kraus(rep, data, input_dim, output_dim)
    return _kraus_to_stinespring(data, input_dim, output_dim)


def _to_operator(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Operator representation."""
    if rep == "Operator":
        return data
    if rep == "Stinespring":
        return _stinespring_to_operator(data, output_dim)
    # Convert via Kraus representation
    if rep != "Kraus":
        data = _to_kraus(rep, data, input_dim, output_dim)
    return _kraus_to_operator(data)


def _from_operator(rep, data, input_dim, output_dim):
    """Transform Operator representation to other representation."""
    if rep == "Operator":
        return data
    if rep == "SuperOp":
        return np.kron(np.conj(data), data)
    if rep == "Choi":
        vec = np.ravel(data, order="F")
        return np.outer(vec, np.conj(vec))
    if rep == "Kraus":
        return [data], None
    if rep == "Stinespring":
        return data, None
    if rep == "Chi":
        _check_nqubit_dim(input_dim, output_dim)
        data = _from_operator("Choi", data, input_dim, output_dim)
        return _choi_to_chi(data, input_dim)
    if rep == "PTM":
        _check_nqubit_dim(input_dim, output_dim)
        data = _from_operator("SuperOp", data, input_dim, output_dim)
        return _superop_to_ptm(data, input_dim)
    raise QiskitError(f"Invalid QuantumChannel {rep}")


def _kraus_to_operator(data):
    """Transform Kraus representation to Operator representation."""
    if data[1] is not None or len(data[0]) > 1:
        raise QiskitError("Channel cannot be converted to Operator representation")
    return data[0][0]


def _stinespring_to_operator(data, output_dim):
    """Transform Stinespring representation to Operator representation."""
    trace_dim = data[0].shape[0] // output_dim
    if data[1] is not None or trace_dim != 1:
        raise QiskitError("Channel cannot be converted to Operator representation")
    return data[0]


def _superop_to_choi(data, input_dim, output_dim):
    """Transform SuperOp representation to Choi representation."""
    shape = (output_dim, output_dim, input_dim, input_dim)
    return _reshuffle(data, shape)


def _choi_to_superop(data, input_dim, output_dim):
    """Transform Choi to SuperOp representation."""
    shape = (input_dim, output_dim, input_dim, output_dim)
    return _reshuffle(data, shape)


def _kraus_to_choi(data):
    """Transform Kraus representation to Choi representation."""
    choi = 0
    kraus_l, kraus_r = data
    if kraus_r is None:
        for i in kraus_l:
            vec = i.ravel(order="F")
            choi += np.outer(vec, vec.conj())
    else:
        for i, j in zip(kraus_l, kraus_r):
            choi += np.outer(i.ravel(order="F"), j.ravel(order="F").conj())
    return choi


def _choi_to_kraus(data, input_dim, output_dim, atol=ATOL_DEFAULT):
    """Transform Choi representation to Kraus representation."""
    import scipy.linalg

    # Check if hermitian matrix
    if is_hermitian_matrix(data, atol=atol):
        # Ideally we'd use `eigh`, but `scipy.linalg.eigh` has stability problems on macOS (at a
        # minimum from SciPy 1.1 to 1.13 with the bundled OpenBLAS, or ~0.3.6 before they started
        # bundling one in).  The Schur form of a Hermitian matrix is guaranteed diagonal:
        #
        #       H = U T U+   for upper-triangular T.
        #   => H+ = U T+ U+
        #   =>  T = T+       because H = H+, and thus T cannot have super-diagonal elements.
        #
        # So the eigenvalues are on the diagonal, therefore the basis-transformation matrix must be
        # a spanning set of the eigenspace.
        #
        # In addition, to prevent `numpy.linalg` errors when the matrix A is ill-conditioned,
        # we apply a small perturbation, replacing A by A + eI. Since (A + eI)x = kx is
        # equivalent to Ax = (k-e)x, it means that the eigenvectors of A + eI and A are the same,
        # and we can perfectly recover the eigenvalues of A from the eigenvalues of A + eI by
        # subtracting e.
        apply_perturbation = np.linalg.cond(data) >= 1e10

        if apply_perturbation:
            data = data + 1e-10 * np.eye(data.shape[0])

        triangular, vecs = scipy.linalg.schur(data)
        values = triangular.diagonal().real

        if apply_perturbation:
            values = values - 1e-10

        # If we're not a CP map, fall-through back to the generalization handling.  Since we needed
        # to get the eigenvalues anyway, we can do the CP check manually rather than deferring to a
        # separate re-calculation.
        if all(values >= -atol):
            kraus = [
                math.sqrt(value) * vec.reshape((output_dim, input_dim), order="F")
                for value, vec in zip(values, vecs.T)
                if abs(value) > atol
            ]
            # If we are converting a zero matrix, we need to return a Kraus set with a single
            # zero-element Kraus matrix
            if not kraus:
                kraus = [np.zeros((output_dim, input_dim), dtype=complex)]
            return kraus, None
        # Fall through.
    # Non-CP-map generalized Kraus representation.
    mat_u, svals, mat_vh = scipy.linalg.svd(data)
    kraus_l = []
    kraus_r = []
    for val, vec_l, vec_r in zip(svals, mat_u.T, mat_vh.conj()):
        kraus_l.append(np.sqrt(val) * vec_l.reshape((output_dim, input_dim), order="F"))
        kraus_r.append(np.sqrt(val) * vec_r.reshape((output_dim, input_dim), order="F"))
    return kraus_l, kraus_r


def _stinespring_to_kraus(data, output_dim):
    """Transform Stinespring representation to Kraus representation."""
    kraus_pair = []
    for stine in data:
        if stine is None:
            kraus_pair.append(None)
        else:
            trace_dim = stine.shape[0] // output_dim
            iden = np.eye(output_dim)
            kraus = []
            for j in range(trace_dim):
                vec = np.zeros(trace_dim)
                vec[j] = 1
                kraus.append(np.kron(iden, vec[None, :]).dot(stine))
            kraus_pair.append(kraus)
    return tuple(kraus_pair)


def _stinespring_to_choi(data, input_dim, output_dim):
    """Transform Stinespring representation to Choi representation."""
    trace_dim = data[0].shape[0] // output_dim
    stine_l = np.reshape(data[0], (output_dim, trace_dim, input_dim))
    if data[1] is None:
        stine_r = stine_l
    else:
        stine_r = np.reshape(data[1], (output_dim, trace_dim, input_dim))
    return np.reshape(
        np.einsum("iAj,kAl->jilk", stine_l, stine_r.conj()), 2 * [input_dim * output_dim]
    )


def _stinespring_to_superop(data, input_dim, output_dim):
    """Transform Stinespring representation to SuperOp representation."""
    trace_dim = data[0].shape[0] // output_dim
    stine_l = np.reshape(data[0], (output_dim, trace_dim, input_dim))
    if data[1] is None:
        stine_r = stine_l
    else:
        stine_r = np.reshape(data[1], (output_dim, trace_dim, input_dim))
    return np.reshape(
        np.einsum("iAj,kAl->ikjl", stine_r.conj(), stine_l),
        (output_dim * output_dim, input_dim * input_dim),
    )


def _kraus_to_stinespring(data, input_dim, output_dim):
    """Transform Kraus representation to Stinespring representation."""
    stine_pair = [None, None]
    for i, kraus in enumerate(data):
        if kraus is not None:
            num_kraus = len(kraus)
            stine = np.zeros((output_dim * num_kraus, input_dim), dtype=complex)
            for j, mat in enumerate(kraus):
                vec = np.zeros(num_kraus)
                vec[j] = 1
                stine += np.kron(mat, vec[:, None])
            stine_pair[i] = stine
    return tuple(stine_pair)


def _kraus_to_superop(data):
    """Transform Kraus representation to SuperOp representation."""
    kraus_l, kraus_r = data
    superop = 0
    if kraus_r is None:
        for i in kraus_l:
            superop += np.kron(np.conj(i), i)
    else:
        for i, j in zip(kraus_l, kraus_r):
            superop += np.kron(np.conj(j), i)
    return superop


def _chi_to_choi(data, input_dim):
    """Transform Chi representation to a Choi representation."""
    num_qubits = int(math.log2(input_dim))
    return _transform_from_pauli(data, num_qubits)


def _choi_to_chi(data, input_dim):
    """Transform Choi representation to the Chi representation."""
    num_qubits = int(math.log2(input_dim))
    return _transform_to_pauli(data, num_qubits)


def _ptm_to_superop(data, input_dim):
    """Transform PTM representation to SuperOp representation."""
    num_qubits = int(math.log2(input_dim))
    return _transform_from_pauli(data, num_qubits)


def _superop_to_ptm(data, input_dim):
    """Transform SuperOp representation to PTM representation."""
    num_qubits = int(math.log2(input_dim))
    return _transform_to_pauli(data, num_qubits)


def _bipartite_tensor(mat1, mat2, shape1=None, shape2=None):
    """Tensor product (A ⊗ B) to bipartite matrices and reravel indices.

    This is used for tensor product of superoperators and Choi matrices.

    Args:
        mat1 (matrix_like): a bipartite matrix A
        mat2 (matrix_like): a bipartite matrix B
        shape1 (tuple): bipartite-shape for matrix A (a0, a1, a2, a3)
        shape2 (tuple): bipartite-shape for matrix B (b0, b1, b2, b3)

    Returns:
        np.array: a bipartite matrix for reravel(A ⊗ B).

    Raises:
        QiskitError: if input matrices are wrong shape.
    """
    # Convert inputs to numpy arrays
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)

    # Determine bipartite dimensions if not provided
    dim_a0, dim_a1 = mat1.shape
    dim_b0, dim_b1 = mat2.shape
    if shape1 is None:
        sdim_a0 = int(math.sqrt(dim_a0))
        sdim_a1 = int(math.sqrt(dim_a1))
        shape1 = (sdim_a0, sdim_a0, sdim_a1, sdim_a1)
    if shape2 is None:
        sdim_b0 = int(math.sqrt(dim_b0))
        sdim_b1 = int(math.sqrt(dim_b1))
        shape2 = (sdim_b0, sdim_b0, sdim_b1, sdim_b1)
    # Check dimensions
    if len(shape1) != 4 or shape1[0] * shape1[1] != dim_a0 or shape1[2] * shape1[3] != dim_a1:
        raise QiskitError("Invalid shape_a")
    if len(shape2) != 4 or shape2[0] * shape2[1] != dim_b0 or shape2[2] * shape2[3] != dim_b1:
        raise QiskitError("Invalid shape_b")

    return _reravel(mat1, mat2, shape1, shape2)


def _reravel(mat1, mat2, shape1, shape2):
    """Reravel two bipartite matrices."""
    # Reshuffle indices
    left_dims = shape1[:2] + shape2[:2]
    right_dims = shape1[2:] + shape2[2:]
    tensor_shape = left_dims + right_dims
    final_shape = (np.prod(left_dims), np.prod(right_dims))
    # Tensor product matrices
    data = np.kron(mat1, mat2)
    data = np.reshape(
        np.transpose(np.reshape(data, tensor_shape), (0, 2, 1, 3, 4, 6, 5, 7)), final_shape
    )
    return data


def _transform_to_pauli(data, num_qubits):
    """Change of basis of bipartite matrix representation."""
    # Change basis: um_{i=0}^3 |i>><\sigma_i|
    basis_mat = np.array(
        [[1, 0, 0, 1], [0, 1, 1, 0], [0, -1j, 1j, 0], [1, 0j, 0, -1]], dtype=complex
    )
    # Note that we manually renormalized after change of basis
    # to avoid rounding errors from square-roots of 2.
    cob = basis_mat
    for _ in range(num_qubits - 1):
        dim = int(math.sqrt(len(cob)))
        cob = np.reshape(
            np.transpose(
                np.reshape(np.kron(basis_mat, cob), (4, dim * dim, 2, 2, dim, dim)),
                (0, 1, 2, 4, 3, 5),
            ),
            (4 * dim * dim, 4 * dim * dim),
        )
    return np.dot(np.dot(cob, data), cob.conj().T) / 2**num_qubits


def _transform_from_pauli(data, num_qubits):
    """Change of basis of bipartite matrix representation."""
    # Change basis: sum_{i=0}^3 =|\sigma_i>><i|
    basis_mat = np.array(
        [[1, 0, 0, 1], [0, 1, 1j, 0], [0, 1, -1j, 0], [1, 0j, 0, -1]], dtype=complex
    )
    # Note that we manually renormalized after change of basis
    # to avoid rounding errors from square-roots of 2.
    cob = basis_mat
    for _ in range(num_qubits - 1):
        dim = int(math.sqrt(len(cob)))
        cob = np.reshape(
            np.transpose(
                np.reshape(np.kron(basis_mat, cob), (2, 2, dim, dim, 4, dim * dim)),
                (0, 2, 1, 3, 4, 5),
            ),
            (4 * dim * dim, 4 * dim * dim),
        )
    return np.dot(np.dot(cob, data), cob.conj().T) / 2**num_qubits


def _reshuffle(mat, shape):
    """Reshuffle the indices of a bipartite matrix A[ij,kl] -> A[lj,ki]."""
    return np.reshape(
        np.transpose(np.reshape(mat, shape), (3, 1, 2, 0)),
        (shape[3] * shape[1], shape[0] * shape[2]),
    )


def _check_nqubit_dim(input_dim, output_dim):
    """Return true if dims correspond to an n-qubit channel."""
    if input_dim != output_dim:
        raise QiskitError(
            f"Not an n-qubit channel: input_dim ({input_dim}) != output_dim ({output_dim})"
        )
    num_qubits = int(math.log2(input_dim))
    if 2**num_qubits != input_dim:
        raise QiskitError("Not an n-qubit channel: input_dim != 2 ** n")
