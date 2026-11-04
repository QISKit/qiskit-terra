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

"""Compute the sum of two qubit registers using ripple-carry approach."""

from .adder import Adder


class CDKMRippleCarryAdder(Adder):
    r"""A ripple-carry circuit to perform in-place addition on two qubit registers.

    As an example, a ripple-carry adder circuit that performs addition on two 3-qubit sized
    registers with a carry-in bit (``kind="full"``) is as follows:

    .. code-block:: text

                ┌──────┐                                     ┌──────┐
         cin_0: ┤2     ├─────────────────────────────────────┤2     ├
                │      │┌──────┐                     ┌──────┐│      │
           a_0: ┤0     ├┤2     ├─────────────────────┤2     ├┤0     ├
                │      ││      │┌──────┐     ┌──────┐│      ││      │
           a_1: ┤  MAJ ├┤0     ├┤2     ├─────┤2     ├┤0     ├┤  UMA ├
                │      ││      ││      │     │      ││      ││      │
           a_2: ┤      ├┤  MAJ ├┤0     ├──■──┤0     ├┤  UMA ├┤      ├
                │      ││      ││      │  │  │      ││      ││      │
           b_0: ┤1     ├┤      ├┤  MAJ ├──┼──┤  UMA ├┤      ├┤1     ├
                └──────┘│      ││      │  │  │      ││      │└──────┘
           b_1: ────────┤1     ├┤      ├──┼──┤      ├┤1     ├────────
                        └──────┘│      │  │  │      │└──────┘
           b_2: ────────────────┤1     ├──┼──┤1     ├────────────────
                                └──────┘┌─┴─┐└──────┘
        cout_0: ────────────────────────┤ X ├────────────────────────
                                        └───┘

    Here *MAJ* and *UMA* gates correspond to the gates introduced in [1]. Note that
    in this implementation the input register qubits are ordered as all qubits from
    the first input register, followed by all qubits from the second input register.

    Two different kinds of adders are supported. By setting the ``kind`` argument, you can also
    choose a half-adder, which doesn't have a carry-in, and a fixed-sized-adder, which has neither
    carry-in nor carry-out, and thus acts on fixed register sizes. Unlike the full-adder,
    these circuits need one additional helper qubit.

    The circuit diagram for the fixed-point adder (``kind="fixed"``) on 3-qubit sized inputs is

    .. code-block:: text

                ┌──────┐┌──────┐                ┌──────┐┌──────┐
           a_0: ┤0     ├┤2     ├────────────────┤2     ├┤0     ├
                │      ││      │┌──────┐┌──────┐│      ││      │
           a_1: ┤      ├┤0     ├┤2     ├┤2     ├┤0     ├┤      ├
                │      ││      ││      ││      ││      ││      │
           a_2: ┤      ├┤  MAJ ├┤0     ├┤0     ├┤  UMA ├┤      ├
                │      ││      ││      ││      ││      ││      │
           b_0: ┤1 MAJ ├┤      ├┤  MAJ ├┤  UMA ├┤      ├┤1 UMA ├
                │      ││      ││      ││      ││      ││      │
           b_1: ┤      ├┤1     ├┤      ├┤      ├┤1     ├┤      ├
                │      │└──────┘│      ││      │└──────┘│      │
           b_2: ┤      ├────────┤1     ├┤1     ├────────┤      ├
                │      │        └──────┘└──────┘        │      │
        help_0: ┤2     ├────────────────────────────────┤2     ├
                └──────┘                                └──────┘

    It has one less qubit than the full-adder since it doesn't have the carry-out, but uses
    a helper qubit instead of the carry-in, so it only has one less qubit, not two.

    .. seealso::

        The following generic gate objects perform additions, like this circuit class,
        but allow the compiler to select the optimal decomposition based on the context.
        Specific implementations can be set via the :class:`.HLSConfig`, e.g. this circuit
        can be chosen via ``Adder=["ripple_c04"]``.

        :class:`.ModularAdderGate`: A generic inplace adder, modulo :math:`2^n`. This
            is functionally equivalent to ``kind="fixed"``.

        :class:`.AdderGate`: A generic inplace adder. This
            is functionally equivalent to ``kind="half"``.

        :class:`.FullAdderGate`: A generic inplace adder, with a carry-in bit. This
            is functionally equivalent to ``kind="full"``.

    References:

    [1] Cuccaro et al., A new quantum ripple-carry addition circuit, 2004.
    `arXiv:quant-ph/0410184 <https://arxiv.org/pdf/quant-ph/0410184.pdf>`_

    [2] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_

    """

    def __init__(
        self, num_state_qubits: int, kind: str = "full", name: str = "CDKMRippleCarryAdder"
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            kind: The kind of adder, can be ``'full'`` for a full adder, ``'half'`` for a half
                adder, or ``'fixed'`` for a fixed-sized adder. A full adder includes both carry-in
                and carry-out, a half only carry-out, and a fixed-sized adder neither carry-in
                nor carry-out.
            name: The name of the circuit object.
        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        """
        from qiskit.synthesis.arithmetic import adder_ripple_c04

        super().__init__(num_state_qubits, name=name)
        circuit = adder_ripple_c04(num_state_qubits, kind)

        self.add_register(*circuit.qregs)
        self.append(circuit.to_gate(), self.qubits)
