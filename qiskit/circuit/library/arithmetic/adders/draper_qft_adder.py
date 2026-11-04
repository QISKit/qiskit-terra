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

"""Compute the sum of two qubit registers using QFT."""

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.basis_change import QFT

from .adder import Adder


class DraperQFTAdder(Adder):
    r"""A circuit that uses QFT to perform in-place addition on two qubit registers.

    For registers with :math:`n` qubits, the QFT adder can perform addition modulo
    :math:`2^n` (with ``kind="fixed"``) or ordinary addition by adding a carry qubit (with
    ``kind="half"``).

    As an example, a QFT adder circuit that performs an ordinary addition on two 2-qubit
    sized registers is as follows:

    .. code-block:: text

         a_0: ─────────■──────■────────────────────────■────────────────
                       │      │                        │
         a_1: ─────────┼──────┼────────■──────■────────┼────────────────
              ┌──────┐ │P(π)  │        │      │        │       ┌───────┐
         b_0: ┤0     ├─■──────┼────────┼──────┼────────┼───────┤0      ├
              │      │        │P(π/2)  │P(π)  │        │       │       │
         b_1: ┤1 QFT ├────────■────────■──────┼────────┼───────┤1 IQFT ├
              │      │                        │P(π/2)  │P(π/4) │       │
        cout: ┤2     ├────────────────────────■────────■───────┤2      ├
              └──────┘                                         └───────┘

    .. note::

        The QFT and inverse-QFT blocks in this implementation omit their swap networks,
        which reverses the qubit order for a more efficient implementation. This affects
        which qubits the controlled-phase gates act on. This drawing represents how the
        adder is implemented in Qiskit; it should not be used as instructions for building
        the circuit manually.

    .. seealso::

        The following generic gate objects perform additions, like this circuit class,
        but allow the compiler to select the optimal decomposition based on the context.
        Specific implementations can be set via the :class:`.HLSConfig`, e.g. this
        circuit can be chosen via ``Adder=["qft_d00"]``.

        :class:`.ModularAdderGate`: A generic inplace adder, modulo :math:`2^n`. This
            is functionally equivalent to ``kind="fixed"``.

        :class:`.AdderGate`: A generic inplace adder. This
            is functionally equivalent to ``kind="half"``.

    References:

    [1] T. G. Draper, Addition on a Quantum Computer, 2000.
    `arXiv:quant-ph/0008033 <https://arxiv.org/pdf/quant-ph/0008033.pdf>`_

    [2] Ruiz-Perez et al., Quantum arithmetic with the Quantum Fourier Transform, 2017.
    `arXiv:1411.5949 <https://arxiv.org/pdf/1411.5949.pdf>`_

    [3] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_

    """

    def __init__(
        self, num_state_qubits: int, kind: str = "fixed", name: str = "DraperQFTAdder"
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            kind: The kind of adder, can be ``'half'`` for a half adder or
                ``'fixed'`` for a fixed-sized adder. A half adder contains a carry-out to represent
                the most-significant bit, but the fixed-sized adder doesn't and hence performs
                addition modulo ``2 ** num_state_qubits``.
            name: The name of the circuit object.
        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        """
        if kind == "full":
            raise ValueError("The DraperQFTAdder only supports 'half' and 'fixed' as ``kind``.")

        if num_state_qubits < 1:
            raise ValueError("The number of qubits must be at least 1.")

        super().__init__(num_state_qubits, name=name)

        qr_a = QuantumRegister(num_state_qubits, name="a")
        qr_b = QuantumRegister(num_state_qubits, name="b")
        qr_list = [qr_a, qr_b]

        if kind == "half":
            qr_z = QuantumRegister(1, name="cout")
            qr_list.append(qr_z)

        # add registers
        self.add_register(*qr_list)

        # define register containing the sum and number of qubits for QFT circuit
        qr_sum = qr_b[:] if kind == "fixed" else qr_b[:] + qr_z[:]
        num_qubits_qft = num_state_qubits if kind == "fixed" else num_state_qubits + 1

        circuit = QuantumCircuit(*self.qregs, name=name)

        # build QFT adder circuit
        circuit.append(QFT(num_qubits_qft, do_swaps=False).to_gate(), qr_sum[:])

        for j in range(num_state_qubits):
            for k in range(num_state_qubits - j):
                lam = np.pi / (2**k)
                circuit.cp(lam, qr_a[j], qr_b[j + k])

        if kind == "half":
            for j in range(num_state_qubits):
                lam = np.pi / (2 ** (j + 1))
                circuit.cp(lam, qr_a[num_state_qubits - j - 1], qr_z[0])

        circuit.append(QFT(num_qubits_qft, do_swaps=False).inverse().to_gate(), qr_sum[:])

        self.append(circuit.to_gate(), self.qubits)
