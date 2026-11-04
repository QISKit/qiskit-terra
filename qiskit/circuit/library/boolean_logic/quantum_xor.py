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


"""Bitwise XOR circuit and gate."""


import numpy as np
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_func


class XOR(QuantumCircuit):
    """An n_qubit circuit for bitwise xor-ing the input with some integer ``amount``.

    The ``amount`` is xor-ed in bitstring form with the input.

    This circuit can also represent addition by ``amount`` over the finite field GF(2).
    """

    @deprecate_func(
        since="2.1",
        additional_msg="Instead, for xor-ing with a specified amount, use BitwiseXorGate,"
        "and for xor-ing with a random amount, use random_bitwise_xor.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        num_qubits: int,
        amount: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            num_qubits: the width of circuit.
            amount: the xor amount in decimal form.
            seed: random seed in case a random xor is requested.

        Raises:
            CircuitError: if the xor bitstring exceeds available qubits.

        Reference Circuit:

        .. plot::
            :alt: Diagram illustrating the previously described circuit.

            from qiskit.circuit.library import XOR
            from qiskit.visualization.library import _generate_circuit_library_visualization
            circuit = XOR(5, seed=42)
            _generate_circuit_library_visualization(circuit)
        """
        circuit = QuantumCircuit(num_qubits, name="xor")

        if amount is not None:
            if len(bin(amount)[2:]) > num_qubits:
                raise CircuitError("Bits in 'amount' exceed circuit width")
        else:
            rng = np.random.default_rng(seed)
            amount = rng.integers(0, 2**num_qubits)

        for i in range(num_qubits):
            bit = amount & 1
            amount = amount >> 1
            if bit == 1:
                circuit.x(i)

        super().__init__(*circuit.qregs, name="xor")
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)


class BitwiseXorGate(Gate):
    """An n-qubit gate for bitwise xor-ing the input with some integer ``amount``.

    The ``amount`` is xor-ed in bitstring form with the input.

    This gate can also represent addition by ``amount`` over the finite field GF(2).

    Reference Circuit:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit import QuantumCircuit
       from qiskit.circuit.library import BitwiseXorGate
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = QuantumCircuit(5)
       circuit.append(BitwiseXorGate(5, amount=12), [0, 1, 2, 3, 4])
       _generate_circuit_library_visualization(circuit)

    """

    def __init__(
        self,
        num_qubits: int,
        amount: int,
    ) -> None:
        """
        Args:
            num_qubits: the width of circuit.
            amount: the xor amount in decimal form.

        Raises:
            CircuitError: if the xor bitstring exceeds available qubits.
        """
        if len(bin(amount)[2:]) > num_qubits:
            raise CircuitError("Bits in 'amount' exceed circuit width")

        super().__init__("xor", num_qubits, [])
        self.amount = amount

    def _define(self):
        circuit = QuantumCircuit(self.num_qubits, name="xor")
        amount = self.amount
        for i in range(self.num_qubits):
            bit = amount & 1
            amount = amount >> 1
            if bit == 1:
                circuit.x(i)

        self.definition = circuit

    def __eq__(self, other):
        return (
            isinstance(other, BitwiseXorGate)
            and self.num_qubits == other.num_qubits
            and self.amount == other.amount
        )

    def inverse(self, annotated: bool = False):
        r"""Return inverted BitwiseXorGate gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            BitwiseXorGate: inverse gate (self-inverse).
        """
        return BitwiseXorGate(self.num_qubits, self.amount)


def random_bitwise_xor(num_qubits: int, seed: int) -> BitwiseXorGate:
    """
    Create a random BitwiseXorGate.

    Args:
        num_qubits: the width of circuit.
        seed: random seed in case a random xor is requested.
    """

    rng = np.random.default_rng(seed)
    amount = rng.integers(0, 2**num_qubits)
    return BitwiseXorGate(num_qubits, amount)
