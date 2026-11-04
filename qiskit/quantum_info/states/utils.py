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

"""
Quantum information utility functions for states.
"""

from __future__ import annotations
import math

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.operators.channel import SuperOp
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT


def partial_trace(state: Statevector | DensityMatrix, qargs: list) -> DensityMatrix:
    """Return reduced density matrix by tracing out part of quantum state.

    If all subsystems are traced over this returns the
    :meth:`~qiskit.quantum_info.DensityMatrix.trace` of the
    input state.

    Args:
        state (Statevector or DensityMatrix): the input state.
        qargs (list): The subsystems to trace over.

    Returns:
        DensityMatrix: The reduced density matrix.

    Raises:
        QiskitError: if input state is invalid.
    """
    state = _format_state(state, validate=False)

    # Compute traced shape
    traced_shape = state._op_shape.remove(qargs=qargs)
    # Convert vector shape to matrix shape
    traced_shape._dims_r = traced_shape._dims_l
    traced_shape._num_qargs_r = traced_shape._num_qargs_l

    # If we are tracing over all subsystems we return the trace
    if traced_shape.size == 0:
        return state.trace()

    # Statevector case
    if isinstance(state, Statevector):
        trace_systems = len(state._op_shape.dims_l()) - 1 - np.array(qargs)
        arr = state._data.reshape(state._op_shape.tensor_shape)
        rho = np.tensordot(arr, arr.conj(), axes=(trace_systems, trace_systems))
        rho = np.reshape(rho, traced_shape.shape)
        return DensityMatrix(rho, dims=traced_shape._dims_l)

    # Density matrix case
    # Empty partial trace case.
    if not qargs:
        return state.copy()
    # Trace first subsystem to avoid coping whole density matrix
    dims = state.dims(qargs)
    tr_op = SuperOp(np.eye(dims[0]).reshape(1, dims[0] ** 2), input_dims=[dims[0]], output_dims=[1])
    ret = state.evolve(tr_op, [qargs[0]])
    # Trace over remaining subsystems
    for qarg, dim in zip(qargs[1:], dims[1:]):
        tr_op = SuperOp(np.eye(dim).reshape(1, dim**2), input_dims=[dim], output_dims=[1])
        ret = ret.evolve(tr_op, [qarg])
    # Remove traced over subsystems which are listed as dimension 1
    ret._op_shape = traced_shape
    return ret


def shannon_entropy(pvec: list | np.ndarray, base: int = 2) -> float:
    r"""Compute the Shannon entropy of a probability vector.

    The shannon entropy of a probability vector
    :math:`\vec{p} = [p_0, ..., p_{n-1}]` is defined as

    .. math::

        H(\vec{p}) = \sum_{i=0}^{n-1} p_i \log_b(p_i)

    where :math:`b` is the log base and (default 2), and
    :math:`0 \log_b(0) \equiv 0`.

    Args:
        pvec (array_like): a probability vector.
        base (int): the base of the logarithm [Default: 2].

    Returns:
        float: The Shannon entropy H(pvec).
    """
    if base == 2:

        def logfn(x):
            return -x * math.log2(x)

    elif base == np.e:

        def logfn(x):
            return -x * math.log(x)

    else:
        log_base = math.log(base)

        def logfn(x):
            return -x * math.log(x) / log_base

    h_val = 0.0
    for x in pvec:
        if 0 < x < 1:
            h_val += logfn(x)
    return h_val


def schmidt_decomposition(state, qargs):
    r"""Return the Schmidt Decomposition of a pure quantum state.

    For an arbitrary bipartite state:

    .. math::
         |\psi\rangle_{AB} = \sum_{i,j} c_{ij}
                             |x_i\rangle_A \otimes |y_j\rangle_B,

    its Schmidt Decomposition is given by the single-index sum over k:

    .. math::
        |\psi\rangle_{AB} = \sum_{k} \lambda_{k}
                            |u_k\rangle_A \otimes |v_k\rangle_B

    where :math:`|u_k\rangle_A` and :math:`|v_k\rangle_B` are an
    orthonormal set of vectors in their respective spaces :math:`A` and :math:`B`,
    and the Schmidt coefficients :math:`\lambda_k` are positive real values.

    Args:
        state (Statevector or DensityMatrix): the input state.
        qargs (list): the list of Input state positions corresponding to subsystem :math:`B`.

    Returns:
        list: list of tuples ``(s, u, v)``, where ``s`` (float) are the Schmidt coefficients
        :math:`\lambda_k`, and ``u`` (Statevector), ``v`` (Statevector) are the Schmidt vectors
        :math:`|u_k\rangle_A`, :math:`|u_k\rangle_B`, respectively.

    Raises:
        QiskitError: if Input qargs is not a list of positions of the Input state.
        QiskitError: if Input qargs is not a proper subset of Input state.

    .. note::
        In Qiskit, qubits are ordered using little-endian notation, with the least significant
        qubits having smaller indices. For example, a four-qubit system is represented as
        :math:`|q_3q_2q_1q_0\rangle`. Using this convention, setting ``qargs=[0]`` will partition the
        state as :math:`|q_3q_2q_1\rangle_A\otimes|q_0\rangle_B`. Furthermore, qubits will be organized
        in this notation regardless of the order they are passed. For instance, passing either
        ``qargs=[1,2]`` or ``qargs=[2,1]`` will result in partitioning the state as
        :math:`|q_3q_0\rangle_A\otimes|q_2q_1\rangle_B`.
    """
    state = _format_state(state, validate=False)

    # convert to statevector if state is density matrix. Errors if state is mixed.
    if isinstance(state, DensityMatrix):
        state = state.to_statevector()

    # reshape statevector into state tensor
    dims = state.dims()
    state_tens = state._data.reshape(dims[::-1])
    ndim = state_tens.ndim
    qudits = list(range(ndim))

    # check if qargs are valid
    if not isinstance(qargs, (list, np.ndarray)):
        raise QiskitError("Input qargs is not a list of positions of the Input state")
    qargs = set(qargs)
    if qargs == set(qudits) or not qargs.issubset(qudits):
        raise QiskitError("Input qargs is not a proper subset of Input state")

    # define subsystem A and B qargs and dims
    qargs_b = list(qargs)
    qargs_a = [i for i in qudits if i not in qargs_b]
    dims_b = state.dims(qargs_b)
    dims_a = state.dims(qargs_a)
    ndim_b = np.prod(dims_b)
    ndim_a = np.prod(dims_a)

    # permute state for desired qargs order
    qargs_axes = [qudits[::-1].index(i) for i in qargs_b + qargs_a][::-1]
    state_tens = state_tens.transpose(qargs_axes)

    # convert state tensor to matrix of prob amplitudes and perform svd.
    state_mat = state_tens.reshape([ndim_a, ndim_b])
    u_mat, s_arr, vh_mat = np.linalg.svd(state_mat, full_matrices=False)

    schmidt_components = [
        (s, Statevector(u, dims=dims_a), Statevector(v, dims=dims_b))
        for s, u, v in zip(s_arr, u_mat.T, vh_mat)
        if s > ATOL_DEFAULT
    ]

    return schmidt_components


def _format_state(state, validate=True):
    """Format input state into class object"""
    if isinstance(state, list):
        state = np.array(state, dtype=complex)
    if isinstance(state, np.ndarray):
        ndim = state.ndim
        if ndim == 1:
            state = Statevector(state)
        elif ndim == 2:
            dim1, dim2 = state.shape
            if dim2 == 1:
                state = Statevector(state)
            elif dim1 == dim2:
                state = DensityMatrix(state)
    if not isinstance(state, (Statevector, DensityMatrix)):
        raise QiskitError("Input is not a quantum state")
    if validate and not state.is_valid():
        raise QiskitError("Input quantum state is not a valid")
    return state


def _funm_svd(matrix, func):
    """Apply real scalar function to singular values of a matrix.

    Args:
        matrix (array_like): (N, N) Matrix at which to evaluate the function.
        func (callable): Callable object that evaluates a scalar function f.

    Returns:
        ndarray: funm (N, N) Value of the matrix function specified by func
                 evaluated at `A`.
    """
    import scipy.linalg as la

    unitary1, singular_values, unitary2 = la.svd(matrix)
    diag_func_singular = np.diag(func(singular_values))
    return unitary1.dot(diag_func_singular).dot(unitary2)
