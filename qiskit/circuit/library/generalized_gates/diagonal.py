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


"""Diagonal matrix circuit."""

from __future__ import annotations
from collections.abc import Sequence

import cmath
import math
import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier
from qiskit.utils.deprecation import deprecate_func

from .ucrz import UCRZGate

_EPS = 1e-10


class Diagonal(QuantumCircuit):
    """Circuit implementing a diagonal transformation."""

    @deprecate_func(
        since="2.1",
        additional_msg="Use DiagonalGate instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(self, diag: Sequence[complex]) -> None:
        r"""
        Args:
            diag: List of the :math:`2^k` diagonal entries (for a diagonal gate on :math:`k` qubits).

        Raises:
            CircuitError: if the list of the diagonal entries or the qubit list is in bad format;
                if the number of diagonal entries is not :math:`2^k`, where :math:`k` denotes the
                number of qubits.
        """
        DiagonalGate._check_input(diag)
        num_qubits = int(math.log2(len(diag)))

        super().__init__(num_qubits, name="Diagonal")
        self.append(DiagonalGate(diag), self.qubits)


class DiagonalGate(Gate):
    r"""A generic diagonal quantum gate.

    Matrix form:

    .. math::
        \text{DiagonalGate}\ q_0, q_1, .., q_{n-1} =
            \begin{pmatrix}
                D[0]    & 0         & \dots     & 0 \\
                0       & D[1]      & \dots     & 0 \\
                \vdots  & \vdots    & \ddots    & 0 \\
                0       & 0         & \dots     & D[n-1]
            \end{pmatrix}

    Diagonal gates are useful as representations of Boolean functions,
    as they can map from :math:`\{0,1\}^{2^n}` to :math:`\{0,1\}^{2^n}` space. For example a phase
    oracle can be seen as a diagonal gate with :math:`\{1, -1\}` on the diagonals. Such
    an oracle will induce a :math:`+1` or :math`-1` phase on the amplitude of any corresponding
    basis state.

    Diagonal gates appear in many classically hard oracular problems such as
    Forrelation or Hidden Shift circuits.

    Diagonal gates are represented and simulated more efficiently than a dense
    :math:`2^n \times 2^n` unitary matrix.

    The reference implementation is via the method described in
    Theorem 7 of [1]. The code is based on Emanuel Malvetti's semester thesis
    at ETH in 2018, supervised by Raban Iten and Prof. Renato Renner.

    References:

    [1] Shende et al., Synthesis of Quantum Logic Circuits, 2009
    `arXiv:0406176 <https://arxiv.org/pdf/quant-ph/0406176.pdf>`_
    """

    def __init__(self, diag: Sequence[complex]) -> None:
        r"""
        Args:
            diag: list of the :math:`2^k` diagonal entries (for a diagonal gate on :math:`k` qubits).
        """
        self._check_input(diag)
        num_qubits = int(math.log2(len(diag)))

        super().__init__("diagonal", num_qubits, diag)

    def _define(self):
        # Since the diagonal is a unitary, all its entries have absolute value
        # one and the diagonal is fully specified by the phases of its entries.
        diag_phases = [cmath.phase(z) for z in self.params]
        n = len(diag_phases)
        circuit = QuantumCircuit(self.num_qubits)

        while n >= 2:
            angles_rz = []
            for i in range(0, n, 2):
                diag_phases[i // 2], rz_angle = _extract_rz(diag_phases[i], diag_phases[i + 1])
                angles_rz.append(rz_angle)
            num_act_qubits = int(math.log2(n))
            ctrl_qubits = list(range(self.num_qubits - num_act_qubits + 1, self.num_qubits))
            target_qubit = self.num_qubits - num_act_qubits

            ucrz = UCRZGate(angles_rz)
            circuit.append(ucrz, [target_qubit] + ctrl_qubits)

            n //= 2
        circuit.global_phase += diag_phases[0]

        self.definition = circuit

    def validate_parameter(self, parameter):
        """Diagonal Gate parameter should accept complex
        (in addition to the Gate parameter types) and always return built-in complex."""
        if isinstance(parameter, complex):
            return complex(parameter)
        else:
            return complex(super().validate_parameter(parameter))

    def inverse(self, annotated: bool = False):
        """Return the inverse of the diagonal gate."""
        if annotated:
            return AnnotatedOperation(self.copy(), InverseModifier)

        return DiagonalGate([np.conj(entry) for entry in self.params])

    @staticmethod
    def _check_input(diag):
        """Check if ``diag`` is in valid format."""
        if not isinstance(diag, (list, np.ndarray)):
            raise CircuitError("Diagonal entries must be in a list or numpy array.")
        num_qubits = math.log2(len(diag))
        if num_qubits < 1 or not num_qubits.is_integer():
            raise CircuitError("The number of diagonal entries is not a positive power of 2.")
        if not np.allclose(np.abs(diag), 1, atol=_EPS):
            raise CircuitError("A diagonal element does not have absolute value one.")


def _extract_rz(phi1, phi2):
    """
    Extract a Rz rotation (angle given by first output) such that exp(j*phase)*Rz(z_angle)
    is equal to the diagonal matrix with entries exp(1j*ph1) and exp(1j*ph2).
    """
    phase = (phi1 + phi2) / 2.0
    z_angle = phi2 - phi1
    return phase, z_angle
