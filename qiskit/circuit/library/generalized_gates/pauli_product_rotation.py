# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A generic Pauli rotation gate."""

from __future__ import annotations

import typing
import math
import numpy as np

from qiskit.circuit import QuantumCircuit, CircuitError, Gate
from qiskit._accelerate.synthesis.pauli_products import synth_pauli_product_rotation

if typing.TYPE_CHECKING:
    import qiskit
    from qiskit.circuit.quantumcircuit import ParameterValueType


class PauliProductRotationGate(Gate):
    r"""A generic Pauli rotation.

    This implements the unitary operation

    .. math::

        R_P(\theta) = e^{-i \theta / 2 P}

    for a Pauli :math:`P \in \{I, X, Y, Z\}^{\otimes n}` and a rotation angle
    :math:`\theta \in \mathbb R`, which could be represented by an unbound parameter.

    Example:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import PauliProductRotationGate
        from qiskit.quantum_info import Pauli

        pauli = Pauli("XYZ")
        ppr = PauliProductRotationGate(pauli, angle=0.2)
        circuit = QuantumCircuit(10)
        circuit.append(ppr, [1, 2, 6])

    """

    def __init__(
        self,
        pauli: qiskit.quantum_info.Pauli,
        angle: ParameterValueType,
        label: str | None = None,
    ):
        """
        Args:
            pauli: The Pauli defining the rotation axis. May include a phase of :math:`-1`, but
                not :math:`i` or :math:`-i`.
            angle: The rotation angle.
            label: An optional label for the gate to display in circuit visualizations.
        """
        if len(pauli.x) == 0:
            raise CircuitError("A Pauli rotation requires at least one qubit.")

        if pauli.phase not in [0, 2]:
            raise CircuitError("A Pauli rotation can not have a Pauli phase of i or -i.")

        num_qubits = len(pauli.z)
        self._pauli_z = pauli.z
        self._pauli_x = pauli.x
        # Absorb the sign of the Pauli into the rotation angle
        params = [angle if pauli.phase == 0 else -angle]

        if label is None:
            label = f"R_{pauli.to_label()}"

        super().__init__(
            name="pauli_product_rotation",
            num_qubits=num_qubits,
            params=params,
            label=label,
        )

    @classmethod
    def _from_pauli_data(cls, z, x, angle, label=None):
        """
        Instantiates a PauliProductRotationGate from raw pauli data, the angle, and the label.
        This function is used internally from within the rust code and from QPY
        serialization.
        """
        from qiskit.quantum_info import Pauli

        return cls(Pauli((z, x, 0)), angle, label)

    def inverse(self, annotated=False):
        """Return the inverse; a rotation about the negative angle."""
        return PauliProductRotationGate(self.pauli(), -self.params[0])

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        r"""Return the controlled version of itself.

        The returned gate represents :math:`e^{-i \theta / 2 P_C}`, where :math:`P_C` is the original
        Pauli :math:`P`, tensored with :math:`|0\rangle\langle 0|` and :math:`|1\rangle\langle 1|`
        projectors (depending on the control state).

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
                Ignored if the controlled gate is implemented as an annotated operation.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``.
            annotated: Indicates whether the controlled gate should be implemented as a controlled gate
                or as an annotated operation. If ``None``, treated as ``False``.

        Returns:
            A controlled version of this gate.

        Raises:
            QiskitError: invalid ``num_ctrl_qubits`` or ``ctrl_state``.
        """
        if num_ctrl_qubits == 0:
            return PauliProductRotationGate(self.pauli(), self.params[0], label=label)

        from qiskit.circuit.library import PauliEvolutionGate

        return PauliEvolutionGate(self.pauli(), time=self.params[0] / 2).control(
            num_ctrl_qubits, label, ctrl_state, annotated
        )

    def __eq__(self, other):
        if not isinstance(other, PauliProductRotationGate):
            return False

        return (
            np.all(self._pauli_z == other._pauli_z)
            and np.all(self._pauli_x == other._pauli_x)
            and (self.params[0] == other.params[0])
        )

    def _define(self):
        # In the Pauli tensor order, the label i corresponds to qubit n-(i+1), so we
        # revert the labels. The code below is consistent with PauliEvolutionGate.
        circuit = QuantumCircuit._from_circuit_data(
            synth_pauli_product_rotation(self),
            legacy_qubits=True,
            name="def_pauli_product_rotation",
        )
        self.definition = circuit

    def pauli(self) -> qiskit.quantum_info.Pauli:
        """Return the Pauli rotation axis.

        Note that this does not include any potential sign in the :class:`~.quantum_info.Pauli`
        object used to construct this gate, since the sign is absorbed into the rotation angle,
        which is accessible via the ``params`` attribute.

        Returns:
            The Pauli rotation axis.
        """
        from qiskit.quantum_info import Pauli

        return Pauli((self._pauli_z, self._pauli_x, 0))

    def to_matrix(self):
        r"""Returns a dense matrix representation of
        :class:`~.PauliProductRotationGate`.

        This implements

        .. math::

           e^{-i \theta / 2 P} = \cos(\theta / 2) I - i \sin(\theta / 2) P

        for a rotation angle :math:`\theta` and Pauli :math:`P`.
        """
        angle = self.params[0]
        out = self.pauli().to_matrix(sparse=False)
        out *= -1j * math.sin(angle / 2)
        np.fill_diagonal(out, out.diagonal() + math.cos(angle / 2))
        return out
