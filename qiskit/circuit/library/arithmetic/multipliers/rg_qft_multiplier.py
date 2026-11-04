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

"""Compute the product of two qubit registers using QFT."""


import numpy as np

from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library.standard_gates import PhaseGate
from qiskit.circuit.library.basis_change import QFT

from .multiplier import Multiplier


class RGQFTMultiplier(Multiplier):
    r"""A QFT multiplication circuit to store product of two input registers out-of-place.

    Multiplication in this circuit is implemented using the procedure of Fig. 3 in [1], where
    weighted sum rotations are implemented as given in Fig. 5 in [1]. The QFT is used on the
    output register and is followed by rotations controlled by input registers. The rotations
    transform the state into the product of two input registers in the QFT basis, which is
    reverted from the QFT basis using the inverse QFT.

    As an example, a circuit that performs a modular QFT multiplication on two 2-qubit
    sized input registers with an output register of 2 qubits, is as follows:

    .. code-block:: text

          a_0: ────────────────────────────────────────■───────■──────■──────■────────────────
                                                       │       │      │      │
          a_1: ─────────■───────■───────■───────■──────┼───────┼──────┼──────┼────────────────
                        │       │       │       │      │       │      │      │
          b_0: ─────────┼───────┼───────■───────■──────┼───────┼──────■──────■────────────────
                        │       │       │       │      │       │      │      │
          b_1: ─────────■───────■───────┼───────┼──────■───────■──────┼──────┼────────────────
               ┌──────┐ │P(4π)  │       │P(2π)  │      │P(2π)  │      │P(π)  │       ┌───────┐
        out_0: ┤0     ├─■───────┼───────■───────┼──────■───────┼──────■──────┼───────┤0      ├
               │  qft │         │P(2π)          │P(π)          │P(π)         │P(π/2) │  iqft │
        out_1: ┤1     ├─────────■───────────────■──────────────■─────────────■───────┤1      ├
               └──────┘                                                              └───────┘

    .. seealso::

        The :class:`.MultiplierGate` object represents a multiplication, like this circuit class,
        but allows the compiler to select the optimal decomposition based on the context.
        Specific implementations can be set via the :class:`.HLSConfig`, e.g. this circuit
        can be chosen via ``Multiplier=["qft_r17"]``.

    References:

    [1] Ruiz-Perez et al., Quantum arithmetic with the Quantum Fourier Transform, 2017.
    `arXiv:1411.5949 <https://arxiv.org/pdf/1411.5949.pdf>`_

    """

    def __init__(
        self,
        num_state_qubits: int,
        num_result_qubits: int | None = None,
        name: str = "RGQFTMultiplier",
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            num_result_qubits: The number of result qubits to limit the output to.
                If number of result qubits is :math:`n`, multiplication modulo :math:`2^n` is performed
                to limit the output to the specified number of qubits. Default
                value is ``2 * num_state_qubits`` to represent any possible
                result from the multiplication of the two inputs.
            name: The name of the circuit object.

        """
        super().__init__(num_state_qubits, num_result_qubits, name=name)

        # define the registers
        qr_a = QuantumRegister(num_state_qubits, name="a")
        qr_b = QuantumRegister(num_state_qubits, name="b")
        qr_out = QuantumRegister(self.num_result_qubits, name="out")
        self.add_register(qr_a, qr_b, qr_out)

        # build multiplication circuit
        circuit = QuantumCircuit(*self.qregs, name=name)

        circuit.append(QFT(self.num_result_qubits, do_swaps=False).to_gate(), qr_out[:])

        for j in range(1, num_state_qubits + 1):
            for i in range(1, num_state_qubits + 1):
                for k in range(1, self.num_result_qubits + 1):
                    lam = (2 * np.pi) / (2 ** (i + j + k - 2 * num_state_qubits))
                    circuit.append(
                        PhaseGate(lam).control(2, annotated=False),
                        [qr_a[num_state_qubits - j], qr_b[num_state_qubits - i], qr_out[k - 1]],
                    )

        circuit.append(QFT(self.num_result_qubits, do_swaps=False).inverse().to_gate(), qr_out[:])

        self.append(circuit.to_gate(), self.qubits)
