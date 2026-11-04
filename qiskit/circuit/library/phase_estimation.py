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

"""Phase estimation circuit."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate
from qiskit.utils.deprecation import deprecate_func
from qiskit.circuit.library import QFT


class PhaseEstimation(QuantumCircuit):
    r"""Phase Estimation circuit.

    In the Quantum Phase Estimation (QPE) algorithm [1, 2, 3], the Phase Estimation circuit is used
    to estimate the phase :math:`\phi` of an eigenvalue :math:`e^{2\pi i\phi}` of a unitary operator
    :math:`U`, provided with the corresponding eigenstate :math:`|\psi\rangle`.
    That is

    .. math::

        U|\psi\rangle = e^{2\pi i\phi} |\psi\rangle

    This estimation (and thereby this circuit) is a central routine to several well-known
    algorithms, such as Shor's algorithm or Quantum Amplitude Estimation.

    References:

    [1] Kitaev, A. Y. (1995). Quantum measurements and the Abelian Stabilizer Problem. 1–22.
    `quant-ph/9511026 <https://arxiv.org/abs/quant-ph/9511026>`_

    [2] Michael A. Nielsen and Isaac L. Chuang. 2011.
    Quantum Computation and Quantum Information: 10th Anniversary Edition (10th ed.).
    Cambridge University Press, New York, NY, USA.

    [3] Qiskit
    `textbook <https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/
    quantum-phase-estimation.ipynb>`_

    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use qiskit.circuit.library.phase_estimation instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        num_evaluation_qubits: int,
        unitary: QuantumCircuit,
        iqft: QuantumCircuit | None = None,
        name: str = "QPE",
    ) -> None:
        """
        Args:
            num_evaluation_qubits: The number of evaluation qubits.
            unitary: The unitary operation :math:`U` which will be repeated and controlled.
            iqft: An inverse Quantum Fourier Transform, per default the inverse of
                :class:`~qiskit.circuit.library.QFT` is used. Note that the QFT should not include
                the usual swaps!
            name: The name of the circuit.

        .. note::

            The inverse QFT should not include a swap of the qubit order.

        Reference Circuit:

        .. plot::
            :alt: Diagram illustrating the previously described circuit.

            from qiskit.circuit import QuantumCircuit
            from qiskit.circuit.library import PhaseEstimation
            from qiskit.visualization.library import _generate_circuit_library_visualization
            unitary = QuantumCircuit(2)
            unitary.x(0)
            unitary.y(1)
            circuit = PhaseEstimation(3, unitary)
            _generate_circuit_library_visualization(circuit)
        """
        qr_eval = QuantumRegister(num_evaluation_qubits, "eval")
        qr_state = QuantumRegister(unitary.num_qubits, "q")
        circuit = QuantumCircuit(qr_eval, qr_state, name=name)

        if iqft is None:
            iqft = QFT(num_evaluation_qubits, inverse=True, do_swaps=False).reverse_bits()

        circuit.h(qr_eval)  # hadamards on evaluation qubits

        for j in range(num_evaluation_qubits):  # controlled powers
            circuit.compose(
                unitary.power(2**j).control(annotated=False), qubits=[j] + qr_state[:], inplace=True
            )

        circuit.compose(iqft, qubits=qr_eval[:], inplace=True)  # final QFT

        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)


def phase_estimation(
    num_evaluation_qubits: int,
    unitary: QuantumCircuit | Gate,
    name: str = "QPE",
) -> QuantumCircuit:
    r"""Phase Estimation circuit.

    In the Quantum Phase Estimation (QPE) algorithm [1, 2, 3], the Phase Estimation circuit is used
    to estimate the phase :math:`\phi` of an eigenvalue :math:`e^{2\pi i\phi}` of a unitary operator
    :math:`U`, provided with the corresponding eigenstate :math:`|\psi\rangle`.
    That is

    .. math::

        U|\psi\rangle = e^{2\pi i\phi} |\psi\rangle

    This estimation (and thereby this circuit) is a central routine to several well-known
    algorithms, such as Shor's algorithm or Quantum Amplitude Estimation.

    Args:
        num_evaluation_qubits: The number of evaluation qubits.
        unitary: The unitary operation :math:`U` which will be repeated and controlled. This
            can either be a :class:`.QuantumCircuit` or a :class:`.Gate`. Passing gates can often
            be more performant, as it allows calling optimized control and power subroutines.
        name: The name of the output circuit.

    Reference Circuit:

    .. plot::
       :alt: A phase estimation circuit.
       :include-source:

       from qiskit.circuit.library import phase_estimation, PauliEvolutionGate
       from qiskit.quantum_info import SparsePauliOp

       hamiltonian = SparsePauliOp(["ZZ", "IX", "XI"])
       evo = PauliEvolutionGate(hamiltonian, time=0.1)  # implements exp(-itH)

       circuit = phase_estimation(3, evo)  # QPE for the evolution operator
       circuit.draw("mpl")

    References:

    [1] Kitaev, A. Y. (1995). Quantum measurements and the Abelian Stabilizer Problem. 1–22.
    `quant-ph/9511026 <https://arxiv.org/abs/quant-ph/9511026>`_

    [2] Michael A. Nielsen and Isaac L. Chuang. 2011.
    Quantum Computation and Quantum Information: 10th Anniversary Edition (10th ed.).
    Cambridge University Press, New York, NY, USA.

    [3] Qiskit `textbook <https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/
    quantum-phase-estimation.ipynb>`_

    """

    from qiskit.circuit.library import PermutationGate, QFTGate

    qr_eval = QuantumRegister(num_evaluation_qubits, "eval")
    qr_state = QuantumRegister(unitary.num_qubits, "q")
    circuit = QuantumCircuit(qr_eval, qr_state, name=name)

    circuit.h(qr_eval)  # hadamards on evaluation qubits

    for j in range(num_evaluation_qubits):  # controlled powers
        circuit.compose(
            unitary.power(2**j).control(annotated=False), qubits=[j] + qr_state[:], inplace=True
        )

    circuit.append(QFTGate(num_evaluation_qubits).inverse(), qr_eval[:])

    reversal_pattern = list(reversed(range(num_evaluation_qubits)))
    circuit.append(PermutationGate(reversal_pattern), qr_eval[:])

    return circuit
