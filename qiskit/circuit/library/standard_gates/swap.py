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

"""Swap gate."""

from __future__ import annotations


import numpy
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit._accelerate.circuit import StandardGate


_SWAP_ARRAY = numpy.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


@with_gate_array(_SWAP_ARRAY)
class SwapGate(SingletonGate):
    r"""The SWAP gate.

    This is a symmetric and Clifford gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.swap` method.

    Circuit symbol:

    .. code-block:: text

        q_0: ─X─
              │
        q_1: ─X─

    Matrix representation:

    .. math::

        SWAP =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    The gate is equivalent to a state swap and is a classical logic gate.

    .. math::

        |a, b\rangle \rightarrow |b, a\rangle
    """

    _standard_gate = StandardGate.Swap

    def __init__(self, label: str | None = None):
        """
        Args:
            label: An optional label for the gate.
        """
        super().__init__("swap", 2, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #           ┌───┐
        # q_0: ──■──┤ X ├──■──
        #      ┌─┴─┐└─┬─┘┌─┴─┐
        # q_1: ┤ X ├──■──┤ X ├
        #      └───┘     └───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.Swap._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the Swap gate.

        For a single control qubit, the controlled gate is implemented as :class:`.CSwapGate`
        (also known as Fredkin gate), regardless of the value of `annotated`.

        For more than one control qubit,
        the controlled gate is implemented as :class:`.ControlledGate` when ``annotated``
        is ``False``, and as :class:`.AnnotatedOperation` when ``annotated`` is ``True``.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
                Ignored if the controlled gate is implemented as an annotated operation.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``.
            annotated: Indicates whether the controlled gate should be implemented as a controlled gate
                or as an annotated operation. If ``None``, defaults to ``False``.

        Returns:
            A controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CSwapGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        """Return inverse Swap gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            SwapGate: inverse gate (self-inverse).
        """
        return SwapGate()  # self-inverse

    def __eq__(self, other):
        return isinstance(other, SwapGate)


@with_controlled_gate_array(_SWAP_ARRAY, num_ctrl_qubits=1)
class CSwapGate(SingletonControlledGate):
    r"""Controlled-SWAP gate, also known as the Fredkin gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cswap` and
    :meth:`~qiskit.circuit.QuantumCircuit.fredkin` methods.

    Circuit symbol:

    .. code-block:: text

        q_0: ─■─
              │
        q_1: ─X─
              │
        q_2: ─X─


    Matrix representation:

    .. math::

        CSWAP\ q_0, q_1, q_2 =
            I \otimes I \otimes |0 \rangle \langle 0| +
            SWAP \otimes |1 \rangle \langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_2. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

            q_0: ─X─
                  │
            q_1: ─X─
                  │
            q_2: ─■─

        .. math::

            CSWAP\ q_2, q_1, q_0 =
                |0 \rangle \langle 0| \otimes I \otimes I +
                |1 \rangle \langle 1| \otimes SWAP =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
                \end{pmatrix}

    In the computational basis, this gate swaps the states of
    the two target qubits if the control qubit is in the
    :math:`|1\rangle` state.

    .. math::
        |0, b, c\rangle \rightarrow |0, b, c\rangle
        |1, b, c\rangle \rightarrow |1, c, b\rangle
    """

    _standard_gate = StandardGate.CSwap

    def __init__(
        self,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create new CSWAP gate."""
        super().__init__(
            "cswap",
            3,
            [],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=SwapGate(label=_base_label),
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        # q_0: ───────■───────
        #      ┌───┐  │  ┌───┐
        # q_1: ┤ X ├──■──┤ X ├
        #      └─┬─┘┌─┴─┐└─┬─┘
        # q_2: ──■──┤ X ├──■──
        #           └───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CSwap._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        """Return inverse CSwap gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CSwapGate: inverse gate (self-inverse).
        """
        return CSwapGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, CSwapGate) and self.ctrl_state == other.ctrl_state
