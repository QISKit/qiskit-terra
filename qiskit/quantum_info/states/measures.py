# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum information measures, metrics, and related functions for states.
"""

from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.utils import (
    partial_trace,
    shannon_entropy,
    _format_state,
    _funm_svd,
)


def state_fidelity(
    state1: Statevector | DensityMatrix, state2: Statevector | DensityMatrix, validate: bool = True
) -> float:
    r"""Return the state fidelity between two quantum states.

    The state fidelity :math:`F` for density matrix input states
    :math:`\rho_1, \rho_2` is given by

    .. math::
        F(\rho_1, \rho_2) = Tr[\sqrt{\sqrt{\rho_1}\rho_2\sqrt{\rho_1}}]^2.

    If one of the states is a pure state this simplifies to
    :math:`F(\rho_1, \rho_2) = \langle\psi_1|\rho_2|\psi_1\rangle`, where
    :math:`\rho_1 = |\psi_1\rangle\!\langle\psi_1|`.

    Args:
        state1 (Statevector or DensityMatrix): the first quantum state.
        state2 (Statevector or DensityMatrix): the second quantum state.
        validate (bool): check if the inputs are valid quantum states
                         [Default: True]

    Returns:
        float: The state fidelity :math:`F(\rho_1, \rho_2)`.

    Raises:
        QiskitError: if ``validate=True`` and the inputs are invalid quantum states.
    """
    # convert input to numpy arrays
    state1 = _format_state(state1, validate=validate)
    state2 = _format_state(state2, validate=validate)

    # Get underlying numpy arrays
    arr1 = state1.data
    arr2 = state2.data
    if isinstance(state1, Statevector):
        if isinstance(state2, Statevector):
            # Fidelity of two Statevectors
            fid = np.abs(arr2.conj().dot(arr1)) ** 2
        else:
            # Fidelity of Statevector(1) and DensityMatrix(2)
            fid = arr1.conj().dot(arr2).dot(arr1)
    elif isinstance(state2, Statevector):
        # Fidelity of Statevector(2) and DensityMatrix(1)
        fid = arr2.conj().dot(arr1).dot(arr2)
    else:
        # Fidelity of two DensityMatrices
        s1sq = _funm_svd(arr1, np.sqrt)
        s2sq = _funm_svd(arr2, np.sqrt)
        fid = np.linalg.norm(s1sq.dot(s2sq), ord="nuc") ** 2
    # Convert to py float rather than return np.float
    return float(np.real(fid))


def purity(state: Statevector | DensityMatrix, validate: bool = True) -> float:
    r"""Calculate the purity of a quantum state.

    The purity of a density matrix :math:`\rho` is

    .. math::

        \text{Purity}(\rho) = Tr[\rho^2]

    Args:
        state (Statevector or DensityMatrix): a quantum state.
        validate (bool): check if input state is valid [Default: True]

    Returns:
        float: the purity :math:`Tr[\rho^2]`.

    Raises:
        QiskitError: if the input isn't a valid quantum state.
    """
    state = _format_state(state, validate=validate)
    return state.purity()


def entropy(state: Statevector | DensityMatrix, base: int = 2) -> float:
    r"""Calculate the von-Neumann entropy of a quantum state.

    The entropy :math:`S` is given by

    .. math::

        S(\rho) = - Tr[\rho \log(\rho)]

    Args:
        state (Statevector or DensityMatrix): a quantum state.
        base (int): the base of the logarithm [Default: 2].

    Returns:
        float: The von-Neumann entropy S(rho).

    Raises:
        QiskitError: if the input state is not a valid QuantumState.
    """
    import scipy.linalg as la

    state = _format_state(state, validate=True)
    if isinstance(state, Statevector):
        return 0
    # Density matrix case
    evals = np.maximum(np.real(la.eigvals(state.data)), 0.0)
    return shannon_entropy(evals, base=base)


def mutual_information(state: Statevector | DensityMatrix, base: int = 2) -> float:
    r"""Calculate the mutual information of a bipartite state.

    The mutual information :math:`I` is given by:

    .. math::

        I(\rho_{AB}) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})

    where :math:`\rho_A=Tr_B[\rho_{AB}], \rho_B=Tr_A[\rho_{AB}]`, are the
    reduced density matrices of the bipartite state :math:`\rho_{AB}`.

    Args:
        state (Statevector or DensityMatrix): a bipartite state.
        base (int): the base of the logarithm [Default: 2].

    Returns:
        float: The mutual information :math:`I(\rho_{AB})`.

    Raises:
        QiskitError: if the input state is not a valid QuantumState.
        QiskitError: if input is not a bipartite QuantumState.
    """
    state = _format_state(state, validate=True)
    if len(state.dims()) != 2:
        raise QiskitError("Input must be a bipartite quantum state.")
    rho_a = partial_trace(state, [1])
    rho_b = partial_trace(state, [0])
    return entropy(rho_a, base=base) + entropy(rho_b, base=base) - entropy(state, base=base)


def concurrence(state: Statevector | DensityMatrix) -> float:
    r"""Calculate the concurrence of a quantum state.

    The concurrence of a bipartite
    :class:`~qiskit.quantum_info.Statevector` :math:`|\psi\rangle` is
    given by

    .. math::

        C(|\psi\rangle) = \sqrt{2(1 - Tr[\rho_0^2])}

    where :math:`\rho_0 = Tr_1[|\psi\rangle\!\langle\psi|]` is the
    reduced state by taking the
    :func:`~qiskit.quantum_info.partial_trace` of the input state.

    For density matrices the concurrence is only defined for
    2-qubit states, it is given by:

    .. math::

        C(\rho) = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)

    where  :math:`\lambda _1 \ge \lambda _2 \ge \lambda _3 \ge \lambda _4`
    are the ordered eigenvalues of the matrix
    :math:`R=\sqrt{\sqrt{\rho }(Y\otimes Y)\overline{\rho}(Y\otimes Y)\sqrt{\rho}}`.

    Args:
        state (Statevector or DensityMatrix): a 2-qubit quantum state.

    Returns:
        float: The concurrence.

    Raises:
        QiskitError: if the input state is not a valid QuantumState.
        QiskitError: if input is not a bipartite QuantumState.
        QiskitError: if density matrix input is not a 2-qubit state.
    """
    import scipy.linalg as la

    # Concurrence computation requires the state to be valid
    state = _format_state(state, validate=True)
    if isinstance(state, Statevector):
        # Pure state concurrence
        dims = state.dims()
        if len(dims) != 2:
            raise QiskitError("Input is not a bipartite quantum state.")
        qargs = [0] if dims[0] > dims[1] else [1]
        rho = partial_trace(state, qargs)
        return float(np.sqrt(2 * (1 - np.real(purity(rho)))))
    # If input is a density matrix it must be a 2-qubit state
    if state.dim != 4:
        raise QiskitError("Input density matrix must be a 2-qubit state.")
    rho = DensityMatrix(state).data
    yy_mat = np.fliplr(np.diag([-1, 1, 1, -1]))
    sigma = rho.dot(yy_mat).dot(rho.conj()).dot(yy_mat)
    w = np.sort(np.real(la.eigvals(sigma)))
    w = np.sqrt(np.maximum(w, 0.0))
    return max(0.0, w[-1] - np.sum(w[0:-1]))


def entanglement_of_formation(state: Statevector | DensityMatrix) -> float:
    """Calculate the entanglement of formation of quantum state.

    The input quantum state must be either a bipartite state vector, or a
    2-qubit density matrix.

    Args:
        state (Statevector or DensityMatrix): a 2-qubit quantum state.

    Returns:
        float: The entanglement of formation.

    Raises:
        QiskitError: if the input state is not a valid QuantumState.
        QiskitError: if input is not a bipartite QuantumState.
        QiskitError: if density matrix input is not a 2-qubit state.
    """
    state = _format_state(state, validate=True)
    if isinstance(state, Statevector):
        # The entanglement of formation is given by the reduced state
        # entropy
        dims = state.dims()
        if len(dims) != 2:
            raise QiskitError("Input is not a bipartite quantum state.")
        qargs = [0] if dims[0] > dims[1] else [1]
        return entropy(partial_trace(state, qargs), base=2)

    # If input is a density matrix it must be a 2-qubit state
    if state.dim != 4:
        raise QiskitError("Input density matrix must be a 2-qubit state.")
    conc = concurrence(state)
    val = (1 + np.sqrt(1 - (conc**2))) / 2
    return shannon_entropy([val, 1 - val])


def negativity(state, qargs):
    r"""Calculates the negativity

    The mathematical expression for negativity is given by:
    .. math::
        {\cal{N}}(\rho) = \frac{|| \rho^{T_A}|| - 1 }{2}

    Args:
        state (Statevector or DensityMatrix): a quantum state.
        qargs (list): The subsystems to be transposed.

    Returns:
        negv (float): Negativity value of the quantum state

    Raises:
        QiskitError: if the input state is not a valid QuantumState.
    """

    if isinstance(state, Statevector):
        # If input is statevector then converting it into density matrix
        state = DensityMatrix(state)
    # Generating partially transposed state
    state = state.partial_transpose(qargs)
    # Calculating SVD
    singular_values = np.linalg.svd(state.data, compute_uv=False)
    eigvals = np.sum(singular_values)
    # Calculating negativity
    negv = (eigvals - 1) / 2
    return negv
