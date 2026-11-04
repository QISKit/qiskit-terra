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


"""Linearly-controlled X, Y or Z rotation."""

from __future__ import annotations


from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError

from .functional_pauli_rotations import FunctionalPauliRotations


class LinearPauliRotations(FunctionalPauliRotations):
    r"""Linearly-controlled X, Y or Z rotation.

    For a register of state qubits :math:`|x\rangle`, a target qubit :math:`|0\rangle` and the
    basis ``'Y'`` this circuit acts as:

    .. code-block:: text

            q_0: ─────────────────────────■───────── ... ──────────────────────
                                          │
                                          .
                                          │
        q_(n-1): ─────────────────────────┼───────── ... ───────────■──────────
                  ┌────────────┐  ┌───────┴───────┐       ┌─────────┴─────────┐
            q_n: ─┤ RY(offset) ├──┤ RY(2^0 slope) ├  ...  ┤ RY(2^(n-1) slope) ├
                  └────────────┘  └───────────────┘       └───────────────────┘

    This can for example be used to approximate linear functions, with :math:`a =` ``slope``:math:`/2`
    and :math:`b =` ``offset``:math:`/2` and the basis ``'Y'``:

    .. math::

        |x\rangle |0\rangle \mapsto \cos(ax + b)|x\rangle|0\rangle + \sin(ax + b)|x\rangle |1\rangle

    Since for small arguments :math:`\sin(x) \approx x` this operator can be used to approximate
    linear functions.
    """

    def __init__(
        self,
        num_state_qubits: int | None = None,
        slope: float = 1,
        offset: float = 0,
        basis: str = "Y",
        name: str = "LinRot",
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits representing the state :math:`|x\rangle`.
            slope: The slope of the controlled rotation.
            offset: The offset of the controlled rotation.
            basis: The type of Pauli rotation ('X', 'Y', 'Z').
            name: The name of the circuit object.
        """
        super().__init__(num_state_qubits=num_state_qubits, basis=basis, name=name)

        # define internal parameters
        self._slope = None
        self._offset = None

        # store parameters
        self.slope = slope
        self.offset = offset

    @property
    def slope(self) -> float:
        """The multiplicative factor in the rotation angle of the controlled rotations.

        The rotation angles are ``slope * 2^0``, ``slope * 2^1``, ... , ``slope * 2^(n-1)`` where
        ``n`` is the number of state qubits.

        Returns:
            The rotation angle common in all controlled rotations.
        """
        return self._slope

    @slope.setter
    def slope(self, slope: float) -> None:
        """Set the multiplicative factor of the rotation angles.

        Args:
            slope: The slope of the rotation angles.
        """
        if self._slope is None or slope != self._slope:
            self._invalidate()
            self._slope = slope

    @property
    def offset(self) -> float:
        """The angle of the single qubit offset rotation on the target qubit.

        Before applying the controlled rotations, a single rotation of angle ``offset`` is
        applied to the target qubit.

        Returns:
            The offset angle.
        """
        return self._offset

    @offset.setter
    def offset(self, offset: float) -> None:
        """Set the angle for the offset rotation on the target qubit.

        Args:
            offset: The offset rotation angle.
        """
        if self._offset is None or offset != self._offset:
            self._invalidate()
            self._offset = offset

    def _reset_registers(self, num_state_qubits: int | None) -> None:
        """Set the number of state qubits.

        Note that this changes the underlying quantum register, if the number of state qubits
        changes.

        Args:
            num_state_qubits: The new number of qubits.
        """
        self.qregs = []

        if num_state_qubits:
            # set new register of appropriate size
            qr_state = QuantumRegister(num_state_qubits, name="state")
            qr_target = QuantumRegister(1, name="target")
            self.qregs = [qr_state, qr_target]

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        valid = True

        if self.num_state_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("The number of qubits has not been set.")

        if self.num_qubits < self.num_state_qubits + 1:
            valid = False
            if raise_on_failure:
                raise CircuitError(
                    "Not enough qubits in the circuit, need at least "
                    f"{self.num_state_qubits + 1}."
                )

        return valid

    def _build(self):
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()
        gate = LinearPauliRotationsGate(self.num_state_qubits, self.slope, self.offset, self.basis)
        self.append(gate, self.qubits)


class LinearPauliRotationsGate(Gate):
    r"""Linearly-controlled X, Y or Z rotation.

    For a register of state qubits :math:`|x\rangle`, a target qubit :math:`|0\rangle` and the
    basis ``'Y'`` this circuit acts as:

    .. parsed-literal::

            q_0: ─────────────────────────■───────── ... ──────────────────────
                                          │
                                          .
                                          │
        q_(n-1): ─────────────────────────┼───────── ... ───────────■──────────
                  ┌────────────┐  ┌───────┴───────┐       ┌─────────┴─────────┐
            q_n: ─┤ RY(offset) ├──┤ RY(2^0 slope) ├  ...  ┤ RY(2^(n-1) slope) ├
                  └────────────┘  └───────────────┘       └───────────────────┘

    This can for example be used to approximate linear functions, with :math:`a =` ``slope``:math:`/2`
    and :math:`b =` ``offset``:math:`/2` and the basis ``'Y'``:

    .. math::

        |x\rangle |0\rangle \mapsto \cos(ax + b)|x\rangle|0\rangle + \sin(ax + b)|x\rangle |1\rangle

    Since for small arguments :math:`\sin(x) \approx x` this operator can be used to approximate
    linear functions.
    """

    def __init__(
        self,
        num_state_qubits: int,
        slope: float = 1,
        offset: float = 0,
        basis: str = "Y",
        label: str | None = None,
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits representing the state :math:`|x\rangle`.
            slope: The slope of the controlled rotation.
            offset: The offset of the controlled rotation.
            basis: The type of Pauli rotation ('X', 'Y', 'Z').
            label: The label of the gate.
        """
        super().__init__("LinPauliRot", num_state_qubits + 1, [], label=label)
        self.slope = slope
        self.offset = offset
        self.basis = basis.lower()

    def _define(self):
        circuit = QuantumCircuit(self.num_qubits)

        # build the circuit
        qr_state = circuit.qubits[: self.num_qubits - 1]
        qr_target = circuit.qubits[-1]

        if self.basis == "x":
            circuit.rx(self.offset, qr_target)
        elif self.basis == "y":
            circuit.ry(self.offset, qr_target)
        else:  # 'Z':
            circuit.rz(self.offset, qr_target)

        for i, q_i in enumerate(qr_state):
            if self.basis == "x":
                circuit.crx(self.slope * pow(2, i), q_i, qr_target)
            elif self.basis == "y":
                circuit.cry(self.slope * pow(2, i), q_i, qr_target)
            else:  # 'Z'
                circuit.crz(self.slope * pow(2, i), q_i, qr_target)

        self.definition = circuit
