# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for the Clifford class.
"""


# ---------------------------------------------------------------------
# Synthesis based on Aaronson & Gottesman decomposition
# ---------------------------------------------------------------------

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_cx,
    _append_h,
    _append_s,
    _append_swap,
    _append_x,
    _append_z,
)
from .clifford_decompose_bm import synth_clifford_bm


def synth_clifford_ag(clifford: Clifford) -> QuantumCircuit:
    """Decompose a :class:`.Clifford` operator into a :class:`.QuantumCircuit`
    based on Aaronson-Gottesman method [1].

    Args:
        clifford: A Clifford operator.

    Returns:
        A circuit implementation of the Clifford.

    References:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """
    # Use 1-qubit decomposition method
    if clifford.num_qubits == 1:
        return synth_clifford_bm(clifford)

    # Compose a circuit which we will convert to an instruction
    circuit = QuantumCircuit(clifford.num_qubits, name=str(clifford))

    # Make a copy of Clifford as we are going to do row reduction to
    # reduce it to an identity
    clifford_cpy = clifford.copy()

    for i in range(clifford.num_qubits):
        # put a 1 one into position by permuting and using Hadamards(i,i)
        _set_qubit_x_true(clifford_cpy, circuit, i)
        # make all entries in row i except ith equal to 0
        # by using phase gate and CNOTS
        _set_row_x_zero(clifford_cpy, circuit, i)
        # treat Zs
        _set_row_z_zero(clifford_cpy, circuit, i)

    for i in range(clifford.num_qubits):
        if clifford_cpy.destab_phase[i]:
            _append_z(clifford_cpy, i)
            circuit.z(i)
        if clifford_cpy.stab_phase[i]:
            _append_x(clifford_cpy, i)
            circuit.x(i)
    # Next we invert the circuit to undo the row reduction and return the
    # result as a gate instruction
    return circuit.inverse()


# ---------------------------------------------------------------------
# Helper functions for Aaronson & Gottesman decomposition
# ---------------------------------------------------------------------


def _set_qubit_x_true(clifford, circuit, qubit):
    """Set destabilizer.X[qubit, qubit] to be True.

    This is done by permuting columns l > qubit or if necessary applying
    a Hadamard
    """
    x = clifford.destab_x[qubit]
    z = clifford.destab_z[qubit]

    if x[qubit]:
        return

    # Try to find non-zero element
    for i in range(qubit + 1, clifford.num_qubits):
        if x[i]:
            _append_swap(clifford, i, qubit)
            circuit.swap(i, qubit)
            return

    # no non-zero element found: need to apply Hadamard somewhere
    for i in range(qubit, clifford.num_qubits):
        if z[i]:
            _append_h(clifford, i)
            circuit.h(i)
            if i != qubit:
                _append_swap(clifford, i, qubit)
                circuit.swap(i, qubit)
            return


def _set_row_x_zero(clifford, circuit, qubit):
    r"""Set destabilizer.X[qubit, i] to False for all i > qubit.

    This is done by applying CNOTs assuming :math:`k \leq N` and A[k][k]=1
    """
    x = clifford.destab_x[qubit]
    z = clifford.destab_z[qubit]

    # Check X first
    for i in range(qubit + 1, clifford.num_qubits):
        if x[i]:
            _append_cx(clifford, qubit, i)
            circuit.cx(qubit, i)

    # Check whether Zs need to be set to zero:
    if np.any(z[qubit:]):
        if not z[qubit]:
            # to treat Zs: make sure row.Z[k] to True
            _append_s(clifford, qubit)
            circuit.s(qubit)

        # reverse CNOTS
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                _append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)
        # set row.Z[qubit] to False
        _append_s(clifford, qubit)
        circuit.s(qubit)


def _set_row_z_zero(clifford, circuit, qubit):
    """Set stabilizer.Z[qubit, i] to False for all i > qubit.

    Implemented by applying (reverse) CNOTS assumes qubit < num_qubits
    and _set_row_x_zero has been called first
    """

    x = clifford.stab_x[qubit]
    z = clifford.stab_z[qubit]

    # check whether Zs need to be set to zero:
    if np.any(z[qubit + 1 :]):
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                _append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)

    # check whether Xs need to be set to zero:
    if np.any(x[qubit:]):
        _append_h(clifford, qubit)
        circuit.h(qubit)
        for i in range(qubit + 1, clifford.num_qubits):
            if x[i]:
                _append_cx(clifford, qubit, i)
                circuit.cx(qubit, i)
        if z[qubit]:
            _append_s(clifford, qubit)
            circuit.s(qubit)
        _append_h(clifford, qubit)
        circuit.h(qubit)
