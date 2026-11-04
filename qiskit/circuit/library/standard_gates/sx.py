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

"""Sqrt(X) and C-Sqrt(X) gates."""

from __future__ import annotations


from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit._accelerate.circuit import StandardGate


_SX_ARRAY = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
_SXDG_ARRAY = [[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]]


@with_gate_array(_SX_ARRAY)
class SXGate(SingletonGate):
    r"""The single-qubit Sqrt(X) gate (:math:`\sqrt{X}`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.sx` method.

    Matrix representation:

    .. math::

        \sqrt{X} = \frac{1}{2} \begin{pmatrix}
                1 + i & 1 - i \\
                1 - i & 1 + i
            \end{pmatrix}

    Circuit symbol:

    .. code-block:: text

             ┌────┐
        q_0: ┤ √X ├
             └────┘

    .. note::

        A global phase difference exists between the definitions of
        :math:`RX(\pi/2)` and :math:`\sqrt{X}`.

        .. math::

            RX(\pi/2) = \frac{1}{\sqrt{2}} \begin{pmatrix}
                        1 & -i \\
                        -i & 1
                      \end{pmatrix}
                    = e^{-i \pi/4} \sqrt{X}

    """

    _standard_gate = StandardGate.SX

    def __init__(self, label: str | None = None):
        """
        Args:
            label: An optional label for the gate.
        """
        super().__init__("sx", 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        # global phase: π/4
        #    ┌─────┐┌───┐┌─────┐
        # q: ┤ Sdg ├┤ H ├┤ Sdg ├
        #    └─────┘└───┘└─────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.SX._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        """Return inverse SX gate (i.e. SXdg).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.SXdgGate`.

        Returns:
            SXdgGate: inverse of :class:`.SXGate`.
        """
        return SXdgGate()

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the SX gate.

        For a single control qubit, the controlled gate is implemented as :class:`.CSXGate`,
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
            gate = CSXGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def __eq__(self, other):
        return isinstance(other, SXGate)


@with_gate_array(_SXDG_ARRAY)
class SXdgGate(SingletonGate):
    r"""The inverse single-qubit Sqrt(X) gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.sxdg` method.

    .. math::

        \sqrt{X}^{\dagger} = \frac{1}{2} \begin{pmatrix}
                1 - i & 1 + i \\
                1 + i & 1 - i
            \end{pmatrix}


    .. note::

        A global phase difference exists between the definitions of
        :math:`RX(-\pi/2)` and :math:`\sqrt{X}^{\dagger}`.

        .. math::

            RX(-\pi/2) = \frac{1}{\sqrt{2}} \begin{pmatrix}
                        1 & i \\
                        i & 1
                      \end{pmatrix}
                    = e^{-i \pi/4} \sqrt{X}^{\dagger}
    """

    _standard_gate = StandardGate.SXdg

    def __init__(self, label: str | None = None):
        """Create new SXdg gate."""
        super().__init__("sxdg", 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        # global phase: 7π/4
        #    ┌───┐┌───┐┌───┐
        # q: ┤ S ├┤ H ├┤ S ├
        #    └───┘└───┘└───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.SXdg._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        """Return inverse SXdg gate (i.e. SX).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.SXGate`.

        Returns:
            SXGate: inverse of :class:`.SXdgGate`
        """
        return SXGate()

    def __eq__(self, other):
        return isinstance(other, SXdgGate)


@with_controlled_gate_array(_SX_ARRAY, num_ctrl_qubits=1)
class CSXGate(SingletonControlledGate):
    r"""Controlled-√X gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.csx` method.

    Circuit symbol:

    .. code-block:: text

        q_0: ──■──
             ┌─┴──┐
        q_1: ┤ √X ├
             └────┘

    Matrix representation:

    .. math::

        C\sqrt{X} \ q_0, q_1 =
        I \otimes |0 \rangle\langle 0| + \sqrt{X} \otimes |1 \rangle\langle 1|  =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & (1 + i) / 2 & 0 & (1 - i) / 2 \\
                0 & 0 & 1 & 0 \\
                0 & (1 - i) / 2 & 0 & (1 + i) / 2
            \end{pmatrix}


    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be `q_1`. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌────┐
            q_0: ┤ √X ├
                 └─┬──┘
            q_1: ──■──

        .. math::

            C\sqrt{X}\ q_1, q_0 =
                |0 \rangle\langle 0| \otimes I + |1 \rangle\langle 1| \otimes \sqrt{X} =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & (1 + i) / 2 & (1 - i) / 2 \\
                    0 & 0 & (1 - i) / 2 & (1 + i) / 2
                \end{pmatrix}

    """

    _standard_gate = StandardGate.CSX

    def __init__(
        self,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create new CSX gate."""
        super().__init__(
            "csx",
            2,
            [],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=SXGate(label=_base_label),
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        # q_0: ───────■───────
        #      ┌───┐┌─┴─┐┌───┐
        # q_1: ┤ H ├┤ S ├┤ H ├
        #      └───┘└───┘└───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CSX._get_definition(self.params), legacy_qubits=True
        )

    def __eq__(self, other):
        return isinstance(other, CSXGate) and self.ctrl_state == other.ctrl_state
