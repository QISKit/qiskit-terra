# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Boolean AND circuit and gate."""

from __future__ import annotations

from qiskit.circuit import QuantumRegister, QuantumCircuit, AncillaRegister, Gate
from qiskit.circuit.library.standard_gates import MCXGate
from qiskit.utils.deprecation import deprecate_func


class AND(QuantumCircuit):
    r"""A circuit implementing the logical AND operation on a number of qubits.

    For the AND operation the state :math:`|1\rangle` is interpreted as ``True``. The result
    qubit is flipped, if the state of all variable qubits is ``True``. In this format, the AND
    operation equals a multi-controlled X gate, which is controlled on all variable qubits.
    Using a list of flags however, qubits can be skipped or negated. Practically, the flags
    allow to skip controls or to apply pre- and post-X gates to the negated qubits.

    The AND gate without special flags equals the multi-controlled-X gate:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import AND
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = AND(5)
       _generate_circuit_library_visualization(circuit)

    Using flags we can negate qubits or skip them. For instance, if we have 5 qubits and want to
    return ``True`` if the first qubit is ``False`` and the last two are ``True`` we use the flags
    ``[-1, 0, 0, 1, 1]``.

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import AND
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = AND(5, flags=[-1, 0, 0, 1, 1])
       _generate_circuit_library_visualization(circuit)

    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use qiskit.circuit.library.AndGate instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        num_variable_qubits: int,
        flags: list[int] | None = None,
        mcx_mode: str = "noancilla",
    ) -> None:
        """
        Args:
            num_variable_qubits: The qubits of which the AND is computed. The result will be written
                into an additional result qubit.
            flags: A list of +1/0/-1 marking negations or omissions of qubits.
            mcx_mode: The mode to be used to implement the multi-controlled X gate.

        """
        self.num_variable_qubits = num_variable_qubits
        self.flags = flags

        # add registers
        qr_variable = QuantumRegister(num_variable_qubits, name="variable")
        qr_result = QuantumRegister(1, name="result")

        circuit = QuantumCircuit(qr_variable, qr_result, name="and")

        # determine the control qubits: all that have a nonzero flag
        flags = flags or [1] * num_variable_qubits
        control_qubits = [q for q, flag in zip(qr_variable, flags) if flag != 0]

        # determine the qubits that need to be flipped (if a flag is < 0)
        flip_qubits = [q for q, flag in zip(qr_variable, flags) if flag < 0]

        # determine the number of ancillas
        num_ancillas = MCXGate.get_num_ancilla_qubits(len(control_qubits), mode=mcx_mode)
        if num_ancillas > 0:
            qr_ancilla = AncillaRegister(num_ancillas, "ancilla")
            circuit.add_register(qr_ancilla)
        else:
            qr_ancilla = AncillaRegister(0)

        if len(flip_qubits) > 0:
            circuit.x(flip_qubits)
        circuit.mcx(control_qubits, qr_result[:], qr_ancilla[:], mode=mcx_mode)
        if len(flip_qubits) > 0:
            circuit.x(flip_qubits)

        super().__init__(*circuit.qregs, name="and")
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)


class AndGate(Gate):
    r"""A gate representing the logical AND operation on a number of qubits.

    For the AND operation the state :math:`|1\rangle` is interpreted as ``True``. The result
    qubit is flipped, if the state of all variable qubits is ``True``. In this format, the AND
    operation equals a multi-controlled X gate, which is controlled on all variable qubits.
    Using a list of flags however, qubits can be skipped or negated. Practically, the flags
    allow to skip controls or to apply pre- and post-X gates to the negated qubits.

    The AndGate gate without special flags equals the multi-controlled-X gate:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit import QuantumCircuit
       from qiskit.circuit.library import AndGate
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = QuantumCircuit(6)
       circuit.append(AndGate(5), [0, 1, 2, 3, 4, 5])
       _generate_circuit_library_visualization(circuit)

    Using flags we can negate qubits or skip them. For instance, if we have 5 qubits and want to
    return ``True`` if the first qubit is ``False`` and the last two are ``True`` we use the flags
    ``[-1, 0, 0, 1, 1]``.

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit import QuantumCircuit
       from qiskit.circuit.library import AndGate
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = QuantumCircuit(6)
       circuit.append(AndGate(5, flags=[-1, 0, 0, 1, 1]), [0, 1, 2, 3, 4, 5])
       _generate_circuit_library_visualization(circuit)

    """

    def __init__(
        self,
        num_variable_qubits: int,
        flags: list[int] | None = None,
    ) -> None:
        """
        Args:
            num_variable_qubits: The qubits of which the AND is computed. The result will be written
                into an additional result qubit.
            flags: A list of +1/0/-1 marking negations or omissions of qubits.
        """
        super().__init__("and", num_variable_qubits + 1, [])
        self.num_variable_qubits = num_variable_qubits
        self.flags = flags

    def _define(self):
        # add registers
        qr_variable = QuantumRegister(self.num_variable_qubits, name="variable")
        qr_result = QuantumRegister(1, name="result")

        # determine the control qubits: all that have a nonzero flag
        flags = self.flags or [1] * self.num_variable_qubits
        control_qubits = [q for q, flag in zip(qr_variable, flags) if flag != 0]

        # determine the qubits that need to be flipped (if a flag is < 0)
        flip_qubits = [q for q, flag in zip(qr_variable, flags) if flag < 0]

        # create the definition circuit
        circuit = QuantumCircuit(qr_variable, qr_result, name="and")

        if len(flip_qubits) > 0:
            circuit.x(flip_qubits)
        circuit.mcx(control_qubits, qr_result[:])
        if len(flip_qubits) > 0:
            circuit.x(flip_qubits)

        self.definition = circuit

    def inverse(self, annotated: bool = False):
        r"""Return inverted AND gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            AndGate: inverse gate (self-inverse).
        """
        return AndGate(self.num_variable_qubits, self.flags)

    def __eq__(self, other):
        return (
            isinstance(other, AndGate)
            and self.num_variable_qubits == other.num_variable_qubits
            and self.flags == other.flags
        )
