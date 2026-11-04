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

"""Y and CY gates."""

from __future__ import annotations


from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit._accelerate.circuit import StandardGate

_Y_ARRAY = [[0, -1j], [1j, 0]]


@with_gate_array(_Y_ARRAY)
class YGate(SingletonGate):
    r"""The single-qubit Pauli-Y gate (:math:`\sigma_y`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.y` method.

    Matrix representation:

    .. math::

        Y = \begin{pmatrix}
                0 & -i \\
                i & 0
            \end{pmatrix}

    Circuit symbol:

    .. code-block:: text

             ┌───┐
        q_0: ┤ Y ├
             └───┘

    Equivalent to a :math:`\pi` radian rotation about the Y axis.

    .. note::

        A global phase difference exists between the definitions of
        :math:`RY(\pi)` and :math:`Y`.

        .. math::

            RY(\pi) = \begin{pmatrix}
                        0 & -1 \\
                        1 & 0
                      \end{pmatrix}
                    = -i Y

    The gate is equivalent to a bit and phase flip.

    .. math::

        |0\rangle \rightarrow i|1\rangle \\
        |1\rangle \rightarrow -i|0\rangle
    """

    _standard_gate = StandardGate.Y

    def __init__(self, label: str | None = None):
        """
        Args:
            label: An optional label for the gate.
        """
        super().__init__("y", 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌──────────────┐
        # q: ┤ U(π,π/2,π/2) ├
        #    └──────────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.Y._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the Y gate.

        For a single control qubit, the controlled gate is implemented as :class:`.CYGate`,
        regardless of the value of `annotated`.

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
                or as an annotated operation. If ``None``, treated as ``False``.

        Returns:
            A controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CYGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted Y gate (:math:`Y^{\dagger} = Y`)

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            YGate: inverse gate (self-inverse).
        """
        return YGate()  # self-inverse

    def __eq__(self, other):
        return isinstance(other, YGate)


@with_controlled_gate_array(_Y_ARRAY, num_ctrl_qubits=1)
class CYGate(SingletonControlledGate):
    r"""Controlled-Y gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cy` method.

    Circuit symbol:

    .. code-block:: text

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ Y ├
             └───┘

    Matrix representation:

    .. math::

        CY\ q_0, q_1 =
        I \otimes |0 \rangle\langle 0| + Y \otimes |1 \rangle\langle 1|  =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 0 & -i \\
                0 & 0 & 1 & 0 \\
                0 & i & 0 & 0
            \end{pmatrix}


    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌───┐
            q_0: ┤ Y ├
                 └─┬─┘
            q_1: ──■──

        .. math::

            CY\ q_1, q_0 =
                |0 \rangle\langle 0| \otimes I + |1 \rangle\langle 1| \otimes Y =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & -i \\
                    0 & 0 & i & 0
                \end{pmatrix}

    """

    _standard_gate = StandardGate.CY

    def __init__(
        self,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create new CY gate."""
        super().__init__(
            "cy",
            2,
            [],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=YGate(label=_base_label),
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        # q_0: ─────────■───────
        #      ┌─────┐┌─┴─┐┌───┐
        # q_1: ┤ Sdg ├┤ X ├┤ S ├
        #      └─────┘└───┘└───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CY._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        """Return inverted CY gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CYGate: inverse gate (self-inverse).
        """
        return CYGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, CYGate) and self.ctrl_state == other.ctrl_state
