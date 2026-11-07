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

"""Compute the sum of two qubit registers using Classical Addition."""

from __future__ import annotations
from .adder import Adder


class VBERippleCarryAdder(Adder):
    r"""The VBE ripple carry adder [1].

    Carry computations that are later uncomputed use relative-phase Toffoli gates,
    whose phases cancel with their inverse. The final carry computation remains exact
    when the circuit has a carry-out qubit.

    This circuit performs inplace addition of two equally-sized quantum registers.
    As an example, a classical adder circuit that performs full addition (i.e. including
    a carry-in bit) on two 2-qubit sized registers is as follows:

    .. code-block:: text

                  ┌────────┐                       ┌───────────┐┌──────┐
           cin_0: ┤0       ├───────────────────────┤0          ├┤0     ├
                  │        │                       │           ││      │
             a_0: ┤1       ├───────────────────────┤1          ├┤1     ├
                  │        │┌────────┐     ┌──────┐│           ││  Sum │
             a_1: ┤        ├┤1       ├──■──┤1     ├┤           ├┤      ├
                  │        ││        │  │  │      ││           ││      │
             b_0: ┤2 Carry ├┤        ├──┼──┤      ├┤2 Carry_dg ├┤2     ├
                  │        ││        │┌─┴─┐│      ││           │└──────┘
             b_1: ┤        ├┤2 Carry ├┤ X ├┤2 Sum ├┤           ├────────
                  │        ││        │└───┘│      ││           │
          cout_0: ┤        ├┤3       ├─────┤      ├┤           ├────────
                  │        ││        │     │      ││           │
        helper_0: ┤3       ├┤0       ├─────┤0     ├┤3          ├────────
                  └────────┘└────────┘     └──────┘└───────────┘


    Here *Carry* and *Sum* gates correspond to the gates introduced in [1].
    *Carry_dg* correspond to the inverse of the *Carry* gate. Note that
    in this implementation the input register qubits are ordered as all qubits from
    the first input register, followed by all qubits from the second input register.
    This is a different ordering as compared to Figure 2 in [1], which leads to a different
    drawing of the circuit.

    .. seealso::

        The following generic gate objects perform additions, like this circuit class,
        but allow the compiler to select the optimal decomposition based on the context.
        Specific implementations can be set via the :class:`.HLSConfig`, e.g. this circuit
        can be chosen via ``Adder=["ripple_v95"]``.

        :class:`.ModularAdderGate`: A generic inplace adder, modulo :math:`2^n`. This
            is functionally equivalent to ``kind="fixed"``.

        :class:`.AdderGate`: A generic inplace adder. This
            is functionally equivalent to ``kind="half"``.

        :class:`.FullAdderGate`: A generic inplace adder, with a carry-in bit. This
            is functionally equivalent to ``kind="full"``.

    References:

    [1] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_

    """

    def __init__(
        self, num_state_qubits: int, kind: str = "full", name: str = "VBERippleCarryAdder"
    ) -> None:
        """
        Args:
            num_state_qubits: The size of the register.
            kind: The kind of adder, can be ``'full'`` for a full adder, ``'half'`` for a half
                adder, or ``'fixed'`` for a fixed-sized adder. A full adder includes both carry-in
                and carry-out, a half only carry-out, and a fixed-sized adder neither carry-in
                nor carry-out.
            name: The name of the circuit.

        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        """
        from qiskit.synthesis.arithmetic import adder_ripple_v95

        super().__init__(num_state_qubits, name=name)
        circuit = adder_ripple_v95(num_state_qubits, kind)

        self.add_register(*circuit.qregs)
        self.append(circuit.to_gate(), self.qubits)
