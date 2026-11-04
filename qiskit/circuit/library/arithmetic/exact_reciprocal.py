# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Exact reciprocal rotation."""

from math import isclose
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate
from qiskit.circuit.library.generalized_gates import UCRYGate
from qiskit.utils.deprecation import deprecate_func


class ExactReciprocal(QuantumCircuit):
    r"""Exact reciprocal

    .. math::

        |x\rangle |0\rangle \mapsto \cos(1/x)|x\rangle|0\rangle + \sin(1/x)|x\rangle |1\rangle
    """

    @deprecate_func(
        since="2.2",
        additional_msg="Use the class qiskit.circuit.library.ExactReciprocalGate instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self, num_state_qubits: int, scaling: float, neg_vals: bool = False, name: str = "1/x"
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits representing the value to invert.
            scaling: Scaling factor :math:`s` of the reciprocal function, i.e. to compute
                :math:`s / x`.
            neg_vals: Whether :math:`x` might represent negative values. In this case the first
                qubit is the sign, with :math:`|1\rangle` for negative and :math:`|0\rangle` for
                positive.  For the negative case it is assumed that the remaining string represents
                :math:`1 - x`. This is because :math:`e^{-2 \pi i x} = e^{2 \pi i (1 - x)}` for
                :math:`x \in [0,1)`.
            name: The name of the object.

        .. note::

            It is assumed that the binary string :math:`x` represents a number < 1.
        """
        qr_state = QuantumRegister(num_state_qubits, "state")
        qr_flag = QuantumRegister(1, "flag")
        super().__init__(qr_state, qr_flag, name=name)

        reciprocal = ExactReciprocalGate(num_state_qubits, scaling, neg_vals, label=name)
        self.append(reciprocal, self.qubits)


class ExactReciprocalGate(Gate):
    r"""Implements an exact reciprocal function.

    For a state :math:`|x\rangle` and a scaling factor :math:`s`, this gate implements the operation

    .. math::

        |x\rangle |0\rangle \mapsto
            \cos\left(\arcsin\left(s\frac{2^n}{x}\right)\right)|x\rangle|0\rangle +
            \left(s\frac{2^n}{x}\right)|x\rangle|1\rangle.

    States representing :math:`x = 0` or :math:`s 2^n / x \geq 1` are left unchanged, since
    this function would not be defined.
    """

    def __init__(
        self, num_state_qubits: int, scaling: float, neg_vals: bool = False, label: str = "1/x"
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits representing the value to invert.
            scaling: Scaling factor :math:`s` of the reciprocal function, i.e. to compute
                :math:`s / x`.
            neg_vals: Whether :math:`x` might represent negative values. In this case the first
                qubit is the sign, with :math:`|1\rangle` for negative and :math:`|0\rangle` for
                positive.  For the negative case it is assumed that the remaining string represents
                :math:`1 - x`. This is because :math:`e^{-2 \pi i x} = e^{2 \pi i (1 - x)}` for
                :math:`x \in [0,1)`.
            label: The label of the object.

        .. note::

            It is assumed that the binary string :math:`x` represents a number < 1.
        """
        super().__init__("ExactReciprocal", num_state_qubits + 1, [], label=label)

        self.scaling = scaling
        self.neg_vals = neg_vals

    def _define(self):
        num_state_qubits = self.num_qubits - 1
        qr_state = QuantumRegister(num_state_qubits, "state")
        qr_flag = QuantumRegister(1, "flag")
        circuit = QuantumCircuit(qr_state, qr_flag)

        angles = [0.0]
        nl = 2 ** (num_state_qubits - 1) if self.neg_vals else 2**num_state_qubits

        # Angles to rotate by scaling / x, where x = i / nl
        for i in range(1, nl):
            if isclose(self.scaling * nl / i, 1, abs_tol=1e-5):
                angles.append(np.pi)
            elif self.scaling * nl / i < 1:
                angles.append(2 * np.arcsin(self.scaling * nl / i))
            else:
                angles.append(0.0)

        circuit.append(UCRYGate(angles), [qr_flag[0]] + qr_state[: len(qr_state) - self.neg_vals])

        if self.neg_vals:
            circuit.append(
                UCRYGate([-theta for theta in angles]).control(annotated=False),
                [qr_state[-1]] + [qr_flag[0]] + qr_state[:-1],
            )
            angles_neg = [0.0]
            for i in range(1, nl):
                if isclose(self.scaling * (-1) / (1 - i / nl), -1, abs_tol=1e-5):
                    angles_neg.append(-np.pi)
                elif np.abs(self.scaling * (-1) / (1 - i / nl)) < 1:
                    angles_neg.append(2 * np.arcsin(self.scaling * (-1) / (1 - i / nl)))
                else:
                    angles_neg.append(0.0)
            circuit.append(
                UCRYGate(angles_neg).control(annotated=False),
                [qr_state[-1]] + [qr_flag[0]] + qr_state[:-1],
            )

        self.definition = circuit
