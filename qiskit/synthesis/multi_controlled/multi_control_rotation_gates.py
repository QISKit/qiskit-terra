# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Multiple-Controlled U3 gate utilities. Not using ancillary qubits.
"""

import math
import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import RXGate, RYGate, RZGate, UnitaryGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

from .gray_code import generate_gray_code
from .mcx_synthesis import synth_mcx_n_dirty_i15


def _apply_cu(circuit, theta, phi, lam, control, target, use_basis_gates=True):
    if use_basis_gates:
        #          ┌──────────────┐
        # control: ┤ P(λ/2 + φ/2) ├──■──────────────────────────────────■────────────────
        #          ├──────────────┤┌─┴─┐┌────────────────────────────┐┌─┴─┐┌────────────┐
        #  target: ┤ P(λ/2 - φ/2) ├┤ X ├┤ U(-0.5*0,0,-0.5*λ - 0.5*φ) ├┤ X ├┤ U(0/2,φ,0) ├
        #          └──────────────┘└───┘└────────────────────────────┘└───┘└────────────┘
        circuit.p((lam + phi) / 2, [control])
        circuit.p((lam - phi) / 2, [target])
        circuit.cx(control, target)
        circuit.u(-theta / 2, 0, -(phi + lam) / 2, [target])
        circuit.cx(control, target)
        circuit.u(theta / 2, phi, 0, [target])
    else:
        circuit.cu(theta, phi, lam, 0, control, target)


def _apply_mcu_graycode(circuit, theta, phi, lam, ctls, tgt, use_basis_gates):
    """Apply multi-controlled u gate from ctls to tgt using graycode
    pattern with single-step angles theta, phi, lam."""

    n = len(ctls)

    gray_code = generate_gray_code(n)
    last_pattern = None

    for pattern in gray_code:
        if "1" not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        # find left most set bit
        lm_pos = list(pattern).index("1")

        # find changed bit
        comp = [i != j for i, j in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                circuit.cx(ctls[pos], ctls[lm_pos])
            else:
                indices = [i for i, x in enumerate(pattern) if x == "1"]
                for idx in indices[1:]:
                    circuit.cx(ctls[idx], ctls[lm_pos])
        # check parity and undo rotation
        if pattern.count("1") % 2 == 0:
            # inverse CU: u(theta, phi, lamb)^dagger = u(-theta, -lam, -phi)
            _apply_cu(
                circuit, -theta, -lam, -phi, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates
            )
        else:
            _apply_cu(circuit, theta, phi, lam, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates)
        last_pattern = pattern


def _mcsu2_real_diagonal(
    gate: Gate,
    num_controls: int,
    use_basis_gates: bool = False,
) -> QuantumCircuit:
    """
    Return a multi-controlled SU(2) gate [1]_ with a real main diagonal or secondary diagonal.

    Args:
        gate: SU(2) Gate whose unitary matrix has one real diagonal.
        num_controls: The number of control qubits.
        use_basis_gates: If ``True``, use ``[p, u, cx]`` gates to implement the decomposition.

    Returns:
        A :class:`.QuantumCircuit` implementing the multi-controlled SU(2) gate.

    Raises:
        QiskitError: If the input matrix is invalid.

    References:

        .. [1]: R. Vale et al. Decomposition of Multi-controlled Special Unitary Single-Qubit Gates
            `arXiv:2302.06377 (2023) <https://arxiv.org/abs/2302.06377>`__

    """
    from qiskit.compiler import transpile

    if isinstance(gate, RYGate):
        theta = gate.params[0]
        s_gate = RYGate(-theta / 4)
        is_secondary_diag_real = True
    elif isinstance(gate, RZGate):
        theta = gate.params[0]
        s_gate = RZGate(-theta / 4)
        is_secondary_diag_real = True
    elif isinstance(gate, RXGate):
        theta = gate.params[0]
        s_gate = RZGate(-theta / 4)
        is_secondary_diag_real = False

    else:
        unitary = gate.to_matrix()
        if unitary.shape != (2, 2):
            raise QiskitError(f"The unitary must be a 2x2 matrix, but has shape {unitary.shape}.")

        if not is_unitary_matrix(unitary):
            raise QiskitError(
                f"The unitary for the input gate must be a unitary matrix, but is {unitary}."
            )

        if not np.isclose(1.0, np.linalg.det(unitary)):
            raise QiskitError(
                "Invalid Value _mcsu2_real_diagonal requires det(unitary) equal to one."
            )

        is_main_diag_real = np.isclose(unitary[0, 0].imag, 0.0) and np.isclose(
            unitary[1, 1].imag, 0.0
        )
        is_secondary_diag_real = np.isclose(unitary[0, 1].imag, 0.0) and np.isclose(
            unitary[1, 0].imag, 0.0
        )

        if not is_main_diag_real and not is_secondary_diag_real:
            raise QiskitError("The unitary must have one real diagonal.")

        if is_secondary_diag_real:
            x = unitary[0, 1]
            z = unitary[1, 1]
        else:
            x = -unitary[0, 1].real
            z = unitary[1, 1] - unitary[0, 1].imag * 1.0j

        if np.isclose(z, -1):
            s_op = [[1.0, 0.0], [0.0, 1.0j]]
        else:
            alpha_r = math.sqrt((math.sqrt((z.real + 1.0) / 2.0) + 1.0) / 2.0)
            alpha_i = z.imag / (
                2.0 * math.sqrt((z.real + 1.0) * (math.sqrt((z.real + 1.0) / 2.0) + 1.0))
            )
            alpha = alpha_r + 1.0j * alpha_i
            beta = x / (2.0 * math.sqrt((z.real + 1.0) * (math.sqrt((z.real + 1.0) / 2.0) + 1.0)))

            # S gate definition
            s_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])
        s_gate = UnitaryGate(s_op)

    k_1 = math.ceil(num_controls / 2.0)
    k_2 = math.floor(num_controls / 2.0)

    controls = list(range(num_controls))  # control indices, defined for code legibility
    target = num_controls  # target index, defined for code legibility

    mcx1 = synth_mcx_n_dirty_i15(num_ctrl_qubits=k_1)
    mcx1_num_ancillas = mcx1.num_qubits - k_1 - 1
    mcx1_qubits = controls[:k_1] + [target] + controls[k_1 : k_1 + mcx1_num_ancillas]

    mcx2 = synth_mcx_n_dirty_i15(num_ctrl_qubits=k_2)
    mcx2_num_ancillas = mcx2.num_qubits - k_2 - 1
    mcx2_qubits = controls[k_1:] + [target] + controls[k_1 - mcx2_num_ancillas : k_1]

    circuit = QuantumCircuit(num_controls + 1, name="MCSU2")

    if not is_secondary_diag_real:
        circuit.h(target)

    circuit.compose(mcx1, mcx1_qubits, inplace=True)
    circuit.append(s_gate, [target])
    circuit.compose(mcx2, mcx2_qubits, inplace=True)
    circuit.append(s_gate.inverse(), [target])
    circuit.compose(mcx1, mcx1_qubits, inplace=True)
    circuit.append(s_gate, [target])
    circuit.compose(mcx2, mcx2_qubits, inplace=True)
    circuit.append(s_gate.inverse(), [target])

    if not is_secondary_diag_real:
        circuit.h(target)

    if use_basis_gates:
        circuit = transpile(circuit, basis_gates=["p", "u", "cx"], qubits_initially_zero=False)

    return circuit
