# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Initialize qubit registers to desired arbitrary state.
"""

from __future__ import annotations
from collections.abc import Sequence
import typing

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.instruction import Instruction
from .state_preparation import StatePreparation
from qiskit.circuit.exceptions import CircuitError

if typing.TYPE_CHECKING:
    from qiskit.quantum_info.states.statevector import Statevector

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class Initialize(Instruction):
    """Complex amplitude initialization.

    Class that initializes some flexible collection of qubit registers, implemented by calling
    the :class:`~.library.StatePreparation` class.
    Note that ``Initialize`` is an :class:`~.circuit.Instruction` and not a :class:`.Gate` since it
    contains a reset instruction, which is not unitary.

    The initial state is prepared based on the :class:`~.library.Isometry` synthesis described in [1].

    References:

    [1] Iten et al., Quantum circuits for isometries (2016).
    `Phys. Rev. A 93, 032318
    <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318>`__.

    """

    def __init__(
        self,
        params: Statevector | Sequence[complex] | str | int,
        num_qubits: int | None = None,
        normalize: bool = False,
    ) -> None:
        r"""
        Args:
            params: The state to initialize to, can be either of the following.

                * Statevector or vector of complex amplitudes to initialize to.
                * Labels of basis states of the Pauli eigenstates Z, X, Y. See
                  :meth:`.Statevector.from_label`. Notice the order of the labels is reversed with
                  respect to the qubit index to be applied to. Example label '01' initializes the
                  qubit zero to :math:`|1\rangle` and the qubit one to :math:`|0\rangle`.
                * An integer that is used as a bitmap indicating which qubits to initialize to
                  :math:`|1\rangle`. Example: setting params to 5 would initialize qubit 0 and qubit
                  2 to :math:`|1\rangle` and qubit 1 to :math:`|0\rangle`.

            num_qubits: This parameter is only used if params is an int. Indicates the total
                number of qubits in the `initialize` call. Example: `initialize` covers 5 qubits
                and params is 3. This allows qubits 0 and 1 to be initialized to :math:`|1\rangle`
                and the remaining 3 qubits to be initialized to :math:`|0\rangle`.
            normalize: Whether to normalize an input array to a unit vector.
        """
        self._stateprep = StatePreparation(params, num_qubits, normalize=normalize)

        super().__init__("initialize", self._stateprep.num_qubits, 0, self._stateprep.params)

    def _define(self):
        q = QuantumRegister(self.num_qubits, "q")
        initialize_circuit = QuantumCircuit(q, name="init_def")
        initialize_circuit.reset(q)
        initialize_circuit.append(self._stateprep, q)
        self.definition = initialize_circuit

    def gates_to_uncompute(self) -> QuantumCircuit:
        """Call to create a circuit with gates that take the desired vector to zero.

        Returns:
            QuantumCircuit: circuit to take ``self.params`` vector to :math:`|{00\\ldots0}\\rangle`
        """

        return self._stateprep.definition.inverse()

    @property
    def params(self):
        """Return initialize params."""
        return self._stateprep.params

    @params.setter
    def params(self, parameters: Statevector | Sequence[complex] | str | int) -> None:
        """Set initialize params."""
        self._stateprep.params = parameters

    def broadcast_arguments(self, qargs, cargs):
        return self._stateprep.broadcast_arguments(qargs, cargs)

    def inverse(self, annotated: bool = False):
        """Raises an error as Initialize cannot be inverted."""
        raise CircuitError(
            "Initialize is not unitary thus can not be inverted. "
            "If you want an invertible state preparation, use StatePreparation instead."
        )
