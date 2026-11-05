# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for a stabilizer state preparation circuit.
"""


from __future__ import annotations

from collections.abc import Callable
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import StabilizerState, Clifford, Pauli
from qiskit.synthesis.linear import calc_inverse_matrix
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr
from qiskit.synthesis.clifford.clifford_decompose_layers import (
    _default_cz_synth_func,
    _reverse_clifford,
    _create_graph_state,
    _decompose_graph_state,
)


def synth_stabilizer_layers(
    stab: StabilizerState,
    cz_synth_func: Callable[[np.ndarray], QuantumCircuit] = _default_cz_synth_func,
    cz_func_reverse_qubits: bool = False,
    validate: bool = False,
) -> QuantumCircuit:
    """Synthesis of a stabilizer state into layers.

    It provides a similar decomposition to the synthesis described in Lemma 8 of reference [1],
    without the initial Hadamard-free sub-circuit which does not affect the stabilizer state.

    For example, a 5-qubit stabilizer state is decomposed into the following layers:

    .. code-block:: text

             ┌─────┐┌─────┐┌─────┐┌─────┐┌────────┐
        q_0: ┤0    ├┤0    ├┤0    ├┤0    ├┤0       ├
             │     ││     ││     ││     ││        │
        q_1: ┤1    ├┤1    ├┤1    ├┤1    ├┤1       ├
             │     ││     ││     ││     ││        │
        q_2: ┤2 H2 ├┤2 S1 ├┤2 CZ ├┤2 H1 ├┤2 Pauli ├
             │     ││     ││     ││     ││        │
        q_3: ┤3    ├┤3    ├┤3    ├┤3    ├┤3       ├
             │     ││     ││     ││     ││        │
        q_4: ┤4    ├┤4    ├┤4    ├┤4    ├┤4       ├
             └─────┘└─────┘└─────┘└─────┘└────────┘

    Args:
        stab: A stabilizer state.
        cz_synth_func: A function to decompose the CZ sub-circuit.
            It gets as input a boolean symmetric matrix, and outputs a :class:`.QuantumCircuit`.
        cz_func_reverse_qubits: ``True`` only if ``cz_synth_func`` is
            :func:`.synth_cz_depth_line_mr`,
            since this function returns a circuit that reverts the order of qubits.
        validate: If ``True``, validates the synthesis process.

    Returns:
        A circuit implementation of the stabilizer state.

    Raises:
        QiskitError: if the input is not a :class:`.StabilizerState`.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """

    if not isinstance(stab, StabilizerState):
        raise QiskitError("The input is not a StabilizerState.")

    cliff = stab.clifford
    num_qubits = cliff.num_qubits

    if cz_func_reverse_qubits:
        cliff0 = _reverse_clifford(cliff)
    else:
        cliff0 = cliff

    H1_circ, cliff1 = _create_graph_state(cliff0, validate=validate)

    H2_circ, CZ1_circ, S1_circ, _ = _decompose_graph_state(
        cliff1, validate=validate, cz_synth_func=cz_synth_func
    )

    qubit_list = list(range(num_qubits))
    layeredCircuit = QuantumCircuit(num_qubits)

    layeredCircuit.append(H2_circ, qubit_list)
    layeredCircuit.append(S1_circ, qubit_list)
    layeredCircuit.append(CZ1_circ, qubit_list)

    if cz_func_reverse_qubits:
        H1_circ = H1_circ.reverse_bits()
    layeredCircuit.append(H1_circ, qubit_list)

    # Add Pauli layer to fix the Clifford phase signs

    clifford_target = Clifford(layeredCircuit)
    pauli_circ = _calc_pauli_diff_stabilizer(cliff, clifford_target)
    layeredCircuit.append(pauli_circ, qubit_list)

    return layeredCircuit


def _calc_pauli_diff_stabilizer(cliff, cliff_target):
    """Given two Cliffords whose stabilizers differ by a Pauli, we find this Pauli."""

    num_qubits = cliff.num_qubits
    if cliff.num_qubits != cliff_target.num_qubits:
        raise QiskitError("num_qubits is not the same for the original clifford and the target.")

    # stabilizer generators of the original clifford
    stab_gen = StabilizerState(cliff).clifford.to_dict()["stabilizer"]

    # stabilizer state of the target clifford
    ts = StabilizerState(cliff_target)

    phase_destab = [False] * num_qubits
    phase_stab = [ts.expectation_value(Pauli(stab_gen[i])) == -1 for i in range(num_qubits)]

    phase = []
    phase.extend(phase_destab)
    phase.extend(phase_stab)
    phase = np.array(phase, dtype=int)

    A = cliff.symplectic_matrix.astype(bool, copy=False)
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


def synth_stabilizer_depth_lnn(stab: StabilizerState) -> QuantumCircuit:
    """Synthesis of an n-qubit stabilizer state for linear-nearest neighbor connectivity,
    in 2-qubit depth :math:`2n+2` and two distinct CX layers, using :class:`.CXGate`\\ s and phase gates
    (:class:`.SGate`, :class:`.SdgGate` or :class:`.ZGate`).

    Args:
        stab: A stabilizer state.

    Returns:
        A circuit implementation of the stabilizer state.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
        2. Dmitri Maslov, Martin Roetteler,
           *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,
           `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
    """

    circ = synth_stabilizer_layers(
        stab,
        cz_synth_func=synth_cz_depth_line_mr,
        cz_func_reverse_qubits=True,
    )
    return circ
