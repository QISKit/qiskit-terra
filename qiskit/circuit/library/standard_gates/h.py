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

"""Hadamard gate."""

from __future__ import annotations

from math import sqrt

import numpy
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit._accelerate.circuit import StandardGate

_H_ARRAY = 1 / sqrt(2) * numpy.array([[1, 1], [1, -1]], dtype=numpy.complex128)


@with_gate_array(_H_ARRAY)
class HGate(SingletonGate):
    r"""Single-qubit Hadamard gate.

    This gate is a \pi rotation about the X+Z axis, and has the effect of
    changing computation basis from :math:`|0\rangle,|1\rangle` to
    :math:`|+\rangle,|-\rangle` and vice-versa.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.h` method.

    Circuit symbol:

    .. code-block:: text

             ┌───┐
        q_0: ┤ H ├
             └───┘

    Matrix representation:

    .. math::

        H = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                1 & 1 \\
                1 & -1
            \end{pmatrix}
    """

    _standard_gate = StandardGate.H

    def __init__(self, label: str | None = None):
        """
        Args:
            label: An optional label for the gate.
        """
        super().__init__("h", 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌────────────┐
        # q: ┤ U(π/2,0,π) ├
        #    └────────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.H._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the H gate.

        For a single control qubit, the controlled gate is implemented as :class:`.CHGate`,
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
            gate = CHGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted H gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            HGate: inverse gate (self-inverse).
        """
        return HGate()  # self-inverse

    def __eq__(self, other):
        return isinstance(other, HGate)


@with_controlled_gate_array(_H_ARRAY, num_ctrl_qubits=1)
class CHGate(SingletonControlledGate):
    r"""Controlled-Hadamard gate.

    Applies a Hadamard on the target qubit if the control is
    in the :math:`|1\rangle` state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ch` method.

    Circuit symbol:

    .. code-block:: text

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ H ├
             └───┘

    Matrix representation:

    .. math::

        CH\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + H \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\
                0 & 0 & 1 & 0 \\
                0 & \frac{1}{\sqrt{2}} & 0 & -\frac{1}{\sqrt{2}}
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌───┐
            q_0: ┤ H ├
                 └─┬─┘
            q_1: ──■──

        .. math::

            CH\ q_1, q_0 =
                |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes H =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
                    0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
                \end{pmatrix}
    """

    _standard_gate = StandardGate.CH

    def __init__(
        self,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create new CH gate."""
        super().__init__(
            "ch",
            2,
            [],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=HGate(label=_base_label),
            _base_label=_base_label,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        # q_0: ─────────────────■─────────────────────
        #      ┌───┐┌───┐┌───┐┌─┴─┐┌─────┐┌───┐┌─────┐
        # q_1: ┤ S ├┤ H ├┤ T ├┤ X ├┤ Tdg ├┤ H ├┤ Sdg ├
        #      └───┘└───┘└───┘└───┘└─────┘└───┘└─────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CH._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        """Return inverted CH gate (itself)."""
        return CHGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, CHGate) and self.ctrl_state == other.ctrl_state
