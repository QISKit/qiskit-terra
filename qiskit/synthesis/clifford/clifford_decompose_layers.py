# This code is part of Qiskit.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for the Clifford class into layers.
"""


from __future__ import annotations
from collections.abc import Callable
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_h,
    _append_s,
    _append_cz,
)
from qiskit.synthesis.linear import (
    synth_cnot_count_full_pmh,
    synth_cnot_depth_line_kms,
)
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr, synth_cx_cz_depth_line_my
from qiskit.synthesis.linear import (
    calc_inverse_matrix,
    compute_rank,
    gauss_elimination,
    gauss_elimination_with_perm,
    binary_matmul,
)


def _default_cx_synth_func(mat):
    """
    Construct the layer of CX gates from a boolean invertible matrix mat.
    """
    CX_circ = synth_cnot_count_full_pmh(mat)
    CX_circ.name = "CX"

    return CX_circ


def _default_cz_synth_func(symmetric_mat):
    """
    Construct the layer of CZ gates from a symmetric matrix.
    """
    nq = symmetric_mat.shape[0]
    qc = QuantumCircuit(nq, name="CZ")

    for j in range(nq):
        for i in range(j):
            if symmetric_mat[i][j]:
                qc.cz(i, j)
    return qc


def synth_clifford_layers(
    cliff: Clifford,
    cx_synth_func: Callable[[np.ndarray], QuantumCircuit] = _default_cx_synth_func,
    cz_synth_func: Callable[[np.ndarray], QuantumCircuit] = _default_cz_synth_func,
    cx_cz_synth_func: Callable[[np.ndarray], QuantumCircuit] | None = None,
    cz_func_reverse_qubits: bool = False,
    validate: bool = False,
) -> QuantumCircuit:
    """Synthesis of a :class:`.Clifford` into layers, it provides a similar
    decomposition to the synthesis described in Lemma 8 of Bravyi and Maslov [1].

    For example, a 5-qubit Clifford circuit is decomposed into the following layers:

    .. code-block:: text

             ┌─────┐┌─────┐┌────────┐┌─────┐┌─────┐┌─────┐┌─────┐┌────────┐
        q_0: ┤0    ├┤0    ├┤0       ├┤0    ├┤0    ├┤0    ├┤0    ├┤0       ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_1: ┤1    ├┤1    ├┤1       ├┤1    ├┤1    ├┤1    ├┤1    ├┤1       ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_2: ┤2 S2 ├┤2 CZ ├┤2 CX_dg ├┤2 H2 ├┤2 S1 ├┤2 CZ ├┤2 H1 ├┤2 Pauli ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_3: ┤3    ├┤3    ├┤3       ├┤3    ├┤3    ├┤3    ├┤3    ├┤3       ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_4: ┤4    ├┤4    ├┤4       ├┤4    ├┤4    ├┤4    ├┤4    ├┤4       ├
             └─────┘└─────┘└────────┘└─────┘└─────┘└─────┘└─────┘└────────┘

    This decomposition is for the default ``cz_synth_func`` and ``cx_synth_func`` functions,
    with other functions one may see slightly different decomposition.

    Args:
        cliff: A Clifford operator.
        cx_synth_func: A function to decompose the CX sub-circuit.
            It gets as input a boolean invertible matrix, and outputs a :class:`.QuantumCircuit`.
        cz_synth_func: A function to decompose the CZ sub-circuit.
            It gets as input a boolean symmetric matrix, and outputs a :class:`.QuantumCircuit`.
        cx_cz_synth_func (Callable): optional, a function to decompose both sub-circuits CZ and CX.
        validate (Boolean): if True, validates the synthesis process.
        cz_func_reverse_qubits (Boolean): True only if ``cz_synth_func`` is
            :func:`.synth_cz_depth_line_mr`, since this function returns a circuit that reverts
            the order of qubits.

    Returns:
        A circuit implementation of the Clifford.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """

    num_qubits = cliff.num_qubits
    if cz_func_reverse_qubits:
        cliff0 = _reverse_clifford(cliff)
    else:
        cliff0 = cliff

    qubit_list = list(range(num_qubits))
    layeredCircuit = QuantumCircuit(num_qubits)

    H1_circ, cliff1 = _create_graph_state(cliff0, validate=validate)

    H2_circ, CZ1_circ, S1_circ, cliff2 = _decompose_graph_state(
        cliff1, validate=validate, cz_synth_func=cz_synth_func
    )

    S2_circ, CZ2_circ, CX_circ = _decompose_hadamard_free(
        cliff2.adjoint(),
        validate=validate,
        cz_synth_func=cz_synth_func,
        cx_synth_func=cx_synth_func,
        cx_cz_synth_func=cx_cz_synth_func,
        cz_func_reverse_qubits=cz_func_reverse_qubits,
    )

    layeredCircuit.append(S2_circ, qubit_list, copy=False)

    if cx_cz_synth_func is None:
        layeredCircuit.append(CZ2_circ, qubit_list, copy=False)

        CXinv = CX_circ.copy().inverse()
        layeredCircuit.append(CXinv, qubit_list, copy=False)

    else:
        # note that CZ2_circ is None and built into the CX_circ when
        # cx_cz_synth_func is not None
        layeredCircuit.append(CX_circ, qubit_list, copy=False)

    layeredCircuit.append(H2_circ, qubit_list, copy=False)
    layeredCircuit.append(S1_circ, qubit_list, copy=False)
    layeredCircuit.append(CZ1_circ, qubit_list, copy=False)

    if cz_func_reverse_qubits:
        H1_circ = H1_circ.reverse_bits()
    layeredCircuit.append(H1_circ, qubit_list, copy=False)

    # Add Pauli layer to fix the Clifford phase signs

    clifford_target = Clifford(layeredCircuit)
    pauli_circ = _calc_pauli_diff(cliff, clifford_target)
    layeredCircuit.append(pauli_circ, qubit_list, copy=False)

    return layeredCircuit


def _reverse_clifford(cliff):
    """Reverse qubit order of a Clifford cliff"""
    cliff_cpy = cliff.copy()
    cliff_cpy.stab_z = np.flip(cliff.stab_z, axis=1)
    cliff_cpy.destab_z = np.flip(cliff.destab_z, axis=1)
    cliff_cpy.stab_x = np.flip(cliff.stab_x, axis=1)
    cliff_cpy.destab_x = np.flip(cliff.destab_x, axis=1)
    return cliff_cpy


def _create_graph_state(cliff, validate=False):
    """Given a Clifford cliff (denoted by U) that induces a stabilizer state U |0>,
    apply a layer H1 of Hadamard gates to a subset of the qubits to make H1 U |0> into a graph state,
    namely to make cliff.stab_x matrix have full rank.
    Returns the QuantumCircuit H1_circ that includes the Hadamard gates and the updated Clifford
    that induces the graph state.
    The algorithm is based on Lemma 6 in [2].

    Args:
        cliff (Clifford): a Clifford operator.
        validate (Boolean): if True, validates the synthesis process.

    Returns:
        H1_circ: a circuit containing a layer of Hadamard gates.
        cliffh: cliffh.stab_x has full rank.

    Raises:
        QiskitError: if there are errors in the Gauss elimination process.

    References:
        2. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """

    num_qubits = cliff.num_qubits
    rank = compute_rank(np.asarray(cliff.stab_x, dtype=bool))
    H1_circ = QuantumCircuit(num_qubits, name="H1")
    cliffh = cliff.copy()

    if rank < num_qubits:
        stab = cliff.stab[:, :-1]
        stab = stab.astype(bool, copy=True)
        gauss_elimination(stab, num_qubits)

        Cmat = stab[rank:num_qubits, num_qubits:]
        Cmat = np.transpose(Cmat)
        perm = gauss_elimination_with_perm(Cmat)
        perm = perm[0 : num_qubits - rank]

        # validate that the output matrix has the same rank
        if validate:
            if compute_rank(Cmat) != num_qubits - rank:
                raise QiskitError("The matrix Cmat after Gauss elimination has wrong rank.")
            if compute_rank(stab[:, 0:num_qubits]) != rank:
                raise QiskitError("The matrix after Gauss elimination has wrong rank.")
            # validate that we have a num_qubits - rank zero rows
            for i in range(rank, num_qubits):
                if stab[i, 0:num_qubits].any():
                    raise QiskitError(
                        "After Gauss elimination, the final num_qubits - rank rows "
                        "contain non-zero elements"
                    )

        for qubit in perm:
            H1_circ.h(qubit)
            _append_h(cliffh, qubit)

        # validate that a layer of Hadamard gates and then appending cliff, provides a graph state.
        if validate:
            stabh = (cliffh.stab_x).astype(bool, copy=False)
            if compute_rank(stabh) != num_qubits:
                raise QiskitError("The state is not a graph state.")

    return H1_circ, cliffh


def _decompose_graph_state(cliff, validate, cz_synth_func):
    """Assumes that a stabilizer state of the Clifford cliff (denoted by U) corresponds to a graph state.
    Decompose it into the layers S1 - CZ1 - H2, such that:
    S1 CZ1 H2 |0> = U |0>,
    where S1_circ is a circuit that can contain only S gates,
    CZ1_circ is a circuit that can contain only CZ gates, and
    H2_circ is a circuit that can contain H gates on all qubits.

    Args:
        cliff (Clifford): a Clifford operator corresponding to a graph state, cliff.stab_x has full rank.
        validate (Boolean): if True, validates the synthesis process.
        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.

    Returns:
        S1_circ: a circuit that can contain only S gates.
        CZ1_circ: a circuit that can contain only CZ gates.
        H2_circ: a circuit containing a layer of Hadamard gates.
        cliff_cpy: a Hadamard-free Clifford.

    Raises:
        QiskitError: if cliff does not induce a graph state.
    """

    num_qubits = cliff.num_qubits
    rank = compute_rank(np.asarray(cliff.stab_x, dtype=bool))
    cliff_cpy = cliff.copy()
    if rank < num_qubits:
        raise QiskitError("The stabilizer state is not a graph state.")

    S1_circ = QuantumCircuit(num_qubits, name="S1")
    H2_circ = QuantumCircuit(num_qubits, name="H2")

    stabx = cliff.stab_x
    stabz = cliff.stab_z
    stabx_inv = calc_inverse_matrix(stabx, validate)
    stabz_update = binary_matmul(stabx_inv, stabz)

    # Assert that stabz_update is a symmetric matrix.
    if validate:
        if (stabz_update != stabz_update.T).any():
            raise QiskitError(
                "The multiplication of stabx_inv and stab_z is not a symmetric matrix."
            )

    CZ1_circ = cz_synth_func(stabz_update)

    for j in range(num_qubits):
        for i in range(j):
            if stabz_update[i][j]:
                _append_cz(cliff_cpy, i, j)

    for i in range(num_qubits):
        if stabz_update[i][i]:
            S1_circ.s(i)
            _append_s(cliff_cpy, i)

    for qubit in range(num_qubits):
        H2_circ.h(qubit)
        _append_h(cliff_cpy, qubit)

    return H2_circ, CZ1_circ, S1_circ, cliff_cpy


def _decompose_hadamard_free(
    cliff, validate, cz_synth_func, cx_synth_func, cx_cz_synth_func, cz_func_reverse_qubits
):
    """Assumes that the Clifford cliff is Hadamard free.
    Decompose it into the layers S2 - CZ2 - CX, where
    S2_circ is a circuit that can contain only S gates,
    CZ2_circ is a circuit that can contain only CZ gates, and
    CX_circ is a circuit that can contain CX gates.

    Args:
        cliff (Clifford): a Hadamard-free clifford operator.
        validate (Boolean): if True, validates the synthesis process.
        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.
        cx_synth_func (Callable): a function to decompose the CX sub-circuit.
        cx_cz_synth_func (Callable): optional, a function to decompose both sub-circuits CZ and CX.
        cz_func_reverse_qubits (Boolean): True only if cz_synth_func is synth_cz_depth_line_mr.

    Returns:
        S2_circ: a circuit that can contain only S gates.
        CZ2_circ: a circuit that can contain only CZ gates.
        CX_circ: a circuit that can contain only CX gates.

    Raises:
        QiskitError: if cliff is not Hadamard free.
    """

    num_qubits = cliff.num_qubits
    destabx = cliff.destab_x
    destabz = cliff.destab_z
    stabx = cliff.stab_x

    if not (stabx == np.zeros((num_qubits, num_qubits))).all():
        raise QiskitError("The given Clifford is not Hadamard-free.")

    destabz_update = binary_matmul(calc_inverse_matrix(destabx), destabz)
    # Assert that destabz_update is a symmetric matrix.
    if validate:
        if (destabz_update != destabz_update.T).any():
            raise QiskitError(
                "The matrix product inv(destabx) * destabz "
                "for the given Clifford is not a symmetric matrix."
            )

    S2_circ = QuantumCircuit(num_qubits, name="S2")
    for i in range(num_qubits):
        if destabz_update[i][i]:
            S2_circ.s(i)

    if cx_cz_synth_func is not None:
        # The cx_cz_synth_func takes as input Mx/Mz representing a CX/CZ circuit
        # and returns the circuit -CZ-CX- implementing them both
        for i in range(num_qubits):
            destabz_update[i][i] = 0

        mat_z = destabz_update
        mat_x = calc_inverse_matrix(destabx.transpose())

        CXCZ_circ = cx_cz_synth_func(mat_x, mat_z)

        return S2_circ, QuantumCircuit(num_qubits), CXCZ_circ

    CZ2_circ = cz_synth_func(destabz_update)

    mat = destabx.transpose()
    if cz_func_reverse_qubits:
        mat = np.flip(mat, axis=0)
    CX_circ = cx_synth_func(mat)

    return S2_circ, CZ2_circ, CX_circ


def _calc_pauli_diff(cliff, cliff_target):
    """Given two Cliffords that differ by a Pauli, we find this Pauli."""

    num_qubits = cliff.num_qubits
    if cliff.num_qubits != cliff_target.num_qubits:
        raise QiskitError("num_qubits is not the same for the original clifford and the target.")

    # Compute the phase difference between the two Cliffords
    phase = [cliff.phase[k] ^ cliff_target.phase[k] for k in range(2 * num_qubits)]
    phase = np.array(phase, dtype=int)

    # compute inverse of our symplectic matrix
    A = cliff.symplectic_matrix
    Ainv = calc_inverse_matrix(A)

    # By carefully writing how X, Y, Z gates affect each qubit, all we need to compute
    # is A^{-1} * (phase)
    C = np.matmul(Ainv, phase) % 2

    # Create the Pauli
    pauli_circ = QuantumCircuit(num_qubits, name="Pauli")
    for k in range(num_qubits):
        destab = C[k]
        stab = C[k + num_qubits]
        if stab and destab:
            pauli_circ.y(k)
        elif stab:
            pauli_circ.x(k)
        elif destab:
            pauli_circ.z(k)

    return pauli_circ


def synth_clifford_depth_lnn(cliff):
    """Synthesis of a :class:`.Clifford` into layers for linear-nearest neighbor connectivity.

    The depth of the synthesized n-qubit circuit is bounded by :math:`7n+2`, which is not optimal.
    It should be replaced by a better algorithm that provides depth bounded by :math:`7n-4` [3].

    Args:
        cliff (Clifford): a Clifford operator.

    Returns:
        QuantumCircuit: a circuit implementation of the Clifford.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
        2. Dmitri Maslov, Martin Roetteler,
           *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,
           `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
        3. Dmitri Maslov, Willers Yang, *CNOT circuits need little help to implement arbitrary
           Hadamard-free Clifford transformations they generate*,
           `arXiv:2210.16195 <https://arxiv.org/abs/2210.16195>`_.
    """
    circ = synth_clifford_layers(
        cliff,
        cx_synth_func=synth_cnot_depth_line_kms,
        cz_synth_func=synth_cz_depth_line_mr,
        cx_cz_synth_func=synth_cx_cz_depth_line_my,
        cz_func_reverse_qubits=True,
    )
    return circ
