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


"""Integer Comparator."""

from __future__ import annotations

from qiskit.circuit import QuantumRegister, AncillaRegister, Gate
from qiskit.circuit.exceptions import CircuitError
from ..blueprintcircuit import BlueprintCircuit


class IntegerComparator(BlueprintCircuit):
    r"""Integer Comparator.

    Operator compares basis states :math:`|i\rangle_n` against a classically given integer
    :math:`L` of fixed value and flips a target qubit if :math:`i \geq L`
    (or :math:`<` depending on the parameter ``geq``):

    .. math::

        |i\rangle_n |0\rangle \mapsto |i\rangle_n |i \geq L\rangle

    This operation is based on two's complement implementation of binary subtraction but only
    uses carry bits and no actual result bits. If the most significant carry bit
    (the results bit) is 1, the :math:`\geq` condition is ``True`` otherwise it is ``False``.
    """

    def __init__(
        self,
        num_state_qubits: int | None = None,
        value: int | None = None,
        geq: bool = True,
        name: str = "cmp",
    ) -> None:
        """
        Args:
            num_state_qubits: Number of state qubits. If this is set it will determine the number
                of qubits required for the circuit.
            value: The fixed value to compare with.
            geq: If True, evaluate a ``>=`` condition, else ``<``.
            name: Name of the circuit.
        """
        super().__init__(name=name)

        self._value = None
        self._geq = None
        self._num_state_qubits = None

        self.value = value
        self.geq = geq
        self.num_state_qubits = num_state_qubits

    @property
    def value(self) -> int:
        """The value to compare the qubit register to.

        Returns:
            The value against which the value of the qubit register is compared.
        """
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        if value != self._value:
            self._invalidate()
            self._value = value

    @property
    def geq(self) -> bool:
        """Return whether the comparator compares greater or less equal.

        Returns:
            True, if the comparator compares ``>=``, False if ``<``.
        """
        return self._geq

    @geq.setter
    def geq(self, geq: bool) -> None:
        """Set whether the comparator compares greater or less equal.

        Args:
            geq: If True, the comparator compares ``>=``, if False ``<``.
        """
        if geq != self._geq:
            self._invalidate()
            self._geq = geq

    @property
    def num_state_qubits(self) -> int:
        """The number of qubits encoding the state for the comparison.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: int | None) -> None:
        """Set the number of state qubits.

        Note that this will change the quantum registers.

        Args:
            num_state_qubits: The new number of state qubits.
        """
        if self._num_state_qubits is None or num_state_qubits != self._num_state_qubits:
            self._invalidate()  # reset data
            self._num_state_qubits = num_state_qubits

            if num_state_qubits is not None:
                # set the new qubit registers
                qr_state = QuantumRegister(num_state_qubits, name="state")
                q_compare = QuantumRegister(1, name="compare")

                self.qregs = [qr_state, q_compare]

                # add ancillas is required
                num_ancillas = num_state_qubits - 1
                if num_ancillas > 0:
                    qr_ancilla = AncillaRegister(num_ancillas)
                    self.add_register(qr_ancilla)

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        valid = True

        if self._num_state_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("Number of state qubits is not set.")

        if self._value is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("No comparison value set.")

        required_num_qubits = 2 * self.num_state_qubits
        if self.num_qubits != required_num_qubits:
            valid = False
            if raise_on_failure:
                raise CircuitError("Number of qubits does not match required number of qubits.")

        return valid

    def _build(self) -> None:
        """If not already built, build the circuit."""
        from qiskit.synthesis.arithmetic.comparators import synth_integer_comparator_2s

        if self._is_built:
            return

        super()._build()

        circuit = synth_integer_comparator_2s(self.num_state_qubits, self.value, self.geq)
        self.append(circuit.to_gate(), self.qubits)


class IntegerComparatorGate(Gate):
    r"""Perform a :math:`\geq` (or :math:`<`) on a qubit register against a classical integer.

    This operator compares basis states :math:`|i\rangle_n` against a classically given integer
    :math:`L` of fixed value and flips a target qubit if :math:`i \geq L`
    (or :math:`<` depending on the parameter ``geq``):

    .. math::

        |i\rangle_n |0\rangle \mapsto |i\rangle_n |i \geq L\rangle

    """

    def __init__(
        self, num_state_qubits: int, value: int, geq: bool = True, label: str | None = None
    ):
        r"""
        Args:
            num_state_qubits: The number of qubits in the registers.
            value: The value :math:`L` to compare to.
            geq: If ``True`` compute :math:`i \geq L`, otherwise compute :math:`i < L`.
            label: An optional label for the gate.
        """
        super().__init__("IntComp", num_state_qubits + 1, [], label=label)
        self.value = value
        self.geq = geq

    def _define(self):
        from qiskit.synthesis.arithmetic.comparators import synth_integer_comparator_greedy

        self.definition = synth_integer_comparator_greedy(self.num_qubits - 1, self.value, self.geq)
