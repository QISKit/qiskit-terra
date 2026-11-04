# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An instruction to implement a Pauli Product Measurement."""

from __future__ import annotations

import typing
import numpy as np

from qiskit.circuit import QuantumCircuit, CircuitError, Instruction
from qiskit.quantum_info import Pauli

from qiskit._accelerate.synthesis.pauli_products import synth_pauli_product_measurement

if typing.TYPE_CHECKING:
    import qiskit


class PauliProductMeasurement(Instruction):
    """Pauli Product Measurement instruction.

    A Pauli Product Measurement is a fundamental operation in fault-tolerant quantum
    computing. Mathematically, it corresponds to a joint projective measurement on
    multiple qubits, where the measured observable is a tensor product of Pauli operators.
    The outcome of this measurement is a single eigenvalue, either :math:`+1` or :math:`-1`,
    indicating the eigenstate of the Pauli product.

    References:

    [1] Daniel Litinski.
    "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery"
    `arXiv:1808.02892 <https://arxiv.org/abs/1808.02892>`__
    """

    def __init__(
        self,
        pauli: qiskit.quantum_info.Pauli,
        label: str | None = None,
    ):
        """
        Args:
            pauli: A tensor product of Pauli operators defining the measurement,
                for example ``Pauli("XY")`` or ``Pauli("-XYIZ")``.
                The identity Pauli operator is not permitted.
                The Pauli may include a phase of :math:`-1`, but not :math:`i` or :math:`-i`.
            label: An optional label for the gate to display in circuit visualizations.
                By default, the label is set to ``PPM(<pauli label>)``.

        .. note::

            While Paulis involving ``"I"``-terms are fully supported, it is recommended to remove
            ``"I"``-terms from the Pauli when creating a ``PauliProductMeasurement`` instruction,
            as this does not change the actual measurement but specifies the instruction over
            a smaller set of qubits.

        Raises:
            CircuitError: If the Pauli is the all identity operator, has size 0, or a complex
                phase.
        """

        if not isinstance(pauli, Pauli):
            raise CircuitError(
                "A Pauli Product Measurement instruction can only be "
                "instantiated from a Pauli object."
            )

        if len(pauli.x) == 0:
            raise CircuitError(
                "A Pauli Product Measurement instruction can not have an empty Pauli label."
            )

        if _is_identity_pauli(pauli):
            raise CircuitError(
                "A Pauli Product Measurement instruction can not have an all-'I' Pauli label."
            )

        if pauli.phase not in [0, 2]:
            raise CircuitError(
                "A Pauli Product Measurement instruction can not have a Pauli phase of i or -i."
            )

        num_qubits = len(pauli.z)
        self._pauli_z = pauli.z
        self._pauli_x = pauli.x
        self._pauli_phase = pauli.phase

        if label is None:
            label = _get_default_label(pauli)

        super().__init__(
            name="pauli_product_measurement",
            num_qubits=num_qubits,
            num_clbits=1,
            params=[],
            label=label,
        )

    @classmethod
    def _from_pauli_data(cls, z, x, phase, label):
        """
        Instantiates a PauliProductMeasurement instruction from pauli data and label.
        This function is used internally from within the rust code and from QPY
        serialization.
        """
        return cls(Pauli((z, x, phase)), label)

    def inverse(self, annotated=False):
        """Prevents from calling ``inverse`` on a PauliProductMeasurement instruction."""
        raise CircuitError("PauliProductMeasurement is not invertible.")

    def __eq__(self, other):
        if not isinstance(other, PauliProductMeasurement):
            return False

        return (
            np.all(self._pauli_z == other._pauli_z)
            and np.all(self._pauli_x == other._pauli_x)
            and (self._pauli_phase == other._pauli_phase)
        )

    def _define(self):
        circuit = QuantumCircuit._from_circuit_data(
            synth_pauli_product_measurement(self),
            legacy_qubits=True,
            name="def_ppm",
        )
        self.definition = circuit

    def _to_pauli_data(self):
        """Returns the pauli data that can be used to reconstruct this instruction.
        This function is used internally from QPY serialization.
        """
        return [self._pauli_z, self._pauli_x, self._pauli_phase]

    def pauli(self) -> Pauli:
        """Return the Pauli product implemented by this measurement.

        This is a public accessor that reconstructs and returns the
        :class:`~qiskit.quantum_info.Pauli` corresponding to this
        instruction's underlying tensor product of Pauli operators,
        including its global phase of ``+1`` or ``-1``.

        Returns:
            The Pauli product measured by this instruction.
        """
        return Pauli((self._pauli_z, self._pauli_x, self._pauli_phase))


def _get_default_label(pauli: Pauli):
    """Creates the default label for PauliProductMeasurement instruction,
    used for visualization.
    """
    return "PPM(" + pauli.to_label() + ")"


def _is_identity_pauli(pauli: Pauli):
    """Return whether a Pauli has an all-'I' label (up to a phase)."""
    return not np.any(pauli.z) and not np.any(pauli.x)
