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


"""InnerProduct circuit and gate."""


from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate
from qiskit.utils.deprecation import deprecate_func


class InnerProduct(QuantumCircuit):
    r"""A 2n-qubit Boolean function that computes the inner product of
    two n-qubit vectors over :math:`F_2`.

    This implementation is a phase oracle which computes the following transform.

    .. math::

        \mathcal{IP}_{2n} : F_2^{2n} \rightarrow {-1, 1}
        \mathcal{IP}_{2n}(x_1, \cdots, x_n, y_1, \cdots, y_n) = (-1)^{x.y}

    The corresponding unitary is a diagonal, which induces a -1 phase on any inputs
    where the inner product of the top and bottom registers is 1. Otherwise it keeps
    the input intact.

    .. code-block:: text


        q0_0: ─■──────────
               │
        q0_1: ─┼──■───────
               │  │
        q0_2: ─┼──┼──■────
               │  │  │
        q0_3: ─┼──┼──┼──■─
               │  │  │  │
        q1_0: ─■──┼──┼──┼─
                  │  │  │
        q1_1: ────■──┼──┼─
                     │  │
        q1_2: ───────■──┼─
                        │
        q1_3: ──────────■─


    Reference Circuit:
        .. plot::
           :alt: Diagram illustrating the previously described circuit.

           from qiskit.circuit.library import InnerProduct
           from qiskit.visualization.library import _generate_circuit_library_visualization
           circuit = InnerProduct(4)
           _generate_circuit_library_visualization(circuit)
    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use qiskit.circuit.library.InnerProductGate instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(self, num_qubits: int) -> None:
        """
        Args:
            num_qubits: width of top and bottom registers (half total circuit width)
        """
        qr_a = QuantumRegister(num_qubits)
        qr_b = QuantumRegister(num_qubits)
        inner = QuantumCircuit(qr_a, qr_b, name="inner_product")

        for i in range(num_qubits):
            inner.cz(qr_a[i], qr_b[i])

        super().__init__(*inner.qregs, name="inner_product")
        self.compose(inner.to_gate(), qubits=self.qubits, inplace=True)


class InnerProductGate(Gate):
    r"""A 2n-qubit Boolean function that computes the inner product of
    two n-qubit vectors over :math:`F_2`.

    This implementation is a phase oracle which computes the following transform.

    .. math::

        \mathcal{IP}_{2n} : F_2^{2n} \rightarrow {-1, 1}
        \mathcal{IP}_{2n}(x_1, \cdots, x_n, y_1, \cdots, y_n) = (-1)^{x.y}

    The corresponding unitary is a diagonal, which induces a -1 phase on any inputs
    where the inner product of the top and bottom registers is 1. Otherwise, it keeps
    the input intact.

    .. parsed-literal::


        q0_0: ─■──────────
               │
        q0_1: ─┼──■───────
               │  │
        q0_2: ─┼──┼──■────
               │  │  │
        q0_3: ─┼──┼──┼──■─
               │  │  │  │
        q1_0: ─■──┼──┼──┼─
                  │  │  │
        q1_1: ────■──┼──┼─
                     │  │
        q1_2: ───────■──┼─
                        │
        q1_3: ──────────■─


    Reference Circuit:
        .. plot::
           :alt: Diagram illustrating the previously described circuit.

           from qiskit.circuit import QuantumCircuit
           from qiskit.circuit.library import InnerProductGate
           from qiskit.visualization.library import _generate_circuit_library_visualization
           circuit = QuantumCircuit(8)
           circuit.append(InnerProductGate(4), [0, 1, 2, 3, 4, 5, 6, 7])
           _generate_circuit_library_visualization(circuit)
    """

    def __init__(
        self,
        num_qubits: int,
    ) -> None:
        """
        Args:
            num_qubits: width of top and bottom registers (half total number of qubits).
        """
        super().__init__("inner_product", 2 * num_qubits, [])

    def _define(self):
        num_qubits = self.num_qubits // 2
        qr_a = QuantumRegister(num_qubits, name="x")
        qr_b = QuantumRegister(num_qubits, name="y")

        circuit = QuantumCircuit(qr_a, qr_b, name="inner_product")
        for i in range(num_qubits):
            circuit.cz(qr_a[i], qr_b[i])

        self.definition = circuit

    def __eq__(self, other):
        return isinstance(other, InnerProductGate) and self.num_qubits == other.num_qubits
