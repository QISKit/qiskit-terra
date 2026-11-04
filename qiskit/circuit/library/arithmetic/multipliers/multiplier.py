# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute the product of two equally sized qubit registers."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.utils.deprecation import deprecate_func


class Multiplier(QuantumCircuit):
    r"""Compute the product of two equally sized qubit registers into a new register.

    For two input registers :math:`|a\rangle_n`, :math:`|b\rangle_n` with :math:`n` qubits each
    and an output register with :math:`2n` qubits, a multiplier performs the following operation

    .. math::

        |a\rangle_n |b\rangle_n |0\rangle_{t} \mapsto |a\rangle_n |b\rangle_n |a \cdot b\rangle_t

    where :math:`t` is the number of bits used to represent the result. To completely store the result
    of the multiplication without overflow we need :math:`t = 2n` bits.

    The quantum register :math:`|a\rangle_n` (analogously :math:`|b\rangle_n` and
    output register)

    .. math::

        |a\rangle_n = |a_0\rangle \otimes \cdots \otimes |a_{n - 1}\rangle,

    for :math:`a_i \in \{0, 1\}`, is associated with the integer value

    .. math::

        a = 2^{0}a_{0} + 2^{1}a_{1} + \cdots + 2^{n - 1}a_{n - 1}.

    """

    @deprecate_func(
        since="2.1",
        additional_msg=(
            "Use the MultiplierGate provided in qiskit.circuit.library.arithmetic instead. "
            "For different multiplier implementations, see "
            "https://quantum.cloud.ibm.com/docs/api/qiskit/synthesis.",
        ),
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        num_state_qubits: int,
        num_result_qubits: int | None = None,
        name: str = "Multiplier",
    ) -> None:
        """
        Args:
            num_state_qubits: The number of qubits in each of the input registers.
            num_result_qubits: The number of result qubits to limit the output to.
                Default value is ``2 * num_state_qubits`` to represent any possible
                result from the multiplication of the two inputs.
            name: The name of the circuit.
        Raises:
            ValueError: If ``num_state_qubits`` is smaller than 1.
            ValueError: If ``num_result_qubits`` is smaller than ``num_state_qubits``.
            ValueError: If ``num_result_qubits`` is larger than ``2 * num_state_qubits``.
        """
        if num_state_qubits < 1:
            raise ValueError("The number of qubits must be at least 1.")

        if num_result_qubits is None:
            num_result_qubits = 2 * num_state_qubits

        if num_result_qubits < num_state_qubits:
            raise ValueError(
                "Number of result qubits is smaller than number of input state qubits."
            )
        if num_result_qubits > 2 * num_state_qubits:
            raise ValueError(
                "Number of result qubits is larger than twice the number of input state qubits."
            )

        super().__init__(name=name)
        self._num_state_qubits = num_state_qubits
        self._num_result_qubits = num_result_qubits

    @property
    def num_state_qubits(self) -> int:
        """The number of state qubits, i.e. the number of bits in each input register.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    @property
    def num_result_qubits(self) -> int:
        """The number of result qubits to limit the output to.

        Returns:
            The number of result qubits.
        """
        return self._num_result_qubits


class MultiplierGate(Gate):
    r"""Compute the product of two equally sized qubit registers into a new register.

    For two input registers :math:`|a\rangle_n`, :math:`|b\rangle_n` with :math:`n` qubits each
    and an output register with :math:`2n` qubits, a multiplier performs the following operation

    .. math::

        |a\rangle_n |b\rangle_n |0\rangle_{t} \mapsto |a\rangle_n |b\rangle_n |a \cdot b\rangle_t

    where :math:`t` is the number of bits used to represent the result. To completely store the result
    of the multiplication without overflow we need :math:`t = 2n` bits.

    The quantum register :math:`|a\rangle_n` (analogously :math:`|b\rangle_n` and
    output register)

    .. math::

        |a\rangle_n = |a_0\rangle \otimes \cdots \otimes |a_{n - 1}\rangle,

    for :math:`a_i \in \{0, 1\}`, is associated with the integer value

    .. math::

        a = 2^{0}a_{0} + 2^{1}a_{1} + \cdots + 2^{n - 1}a_{n - 1}.

    """

    def __init__(
        self,
        num_state_qubits: int,
        num_result_qubits: int | None = None,
        label: str | None = None,
    ) -> None:
        """
        Args:
            num_state_qubits: The number of qubits in each of the input registers.
            num_result_qubits: The number of result qubits to limit the output to.
                Default value is ``2 * num_state_qubits`` to represent any possible
                result from the multiplication of the two inputs.
            label: The optional string label to apply to the instruction.
        Raises:
            ValueError: If ``num_state_qubits`` is smaller than 1.
            ValueError: If ``num_result_qubits`` is smaller than ``num_state_qubits``.
            ValueError: If ``num_result_qubits`` is larger than ``2 * num_state_qubits``.
        """
        if num_state_qubits < 1:
            raise ValueError("The number of state qubits must be at least 1.")

        if num_result_qubits is None:
            num_result_qubits = 2 * num_state_qubits
        elif num_result_qubits < num_state_qubits or num_result_qubits > 2 * num_state_qubits:
            raise ValueError(
                f"num_result_qubits ({num_result_qubits}) must be in between num_state_qubits "
                f"({num_state_qubits}) and 2 * num_state_qubits ({2 * num_state_qubits})"
            )

        super().__init__("Multiplier", 2 * num_state_qubits + num_result_qubits, [], label=label)

        self._num_state_qubits = num_state_qubits
        self._num_result_qubits = num_result_qubits

    @property
    def num_state_qubits(self) -> int:
        """The number of state qubits, i.e. the number of bits in each input register.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    @property
    def num_result_qubits(self) -> int:
        """The number of result qubits to limit the output to.

        Returns:
            The number of result qubits.
        """
        return self._num_result_qubits

    def _define(self):
        """Populates self.definition with some decomposition of this gate."""
        from qiskit.synthesis.arithmetic import multiplier_qft_r17

        # This particular decomposition does not use any ancilla qubits.
        # Note that the transpiler may choose a different decomposition
        # based on the number of ancilla qubits available.
        self.definition = multiplier_qft_r17(self.num_state_qubits, self.num_result_qubits)
