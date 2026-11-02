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

"""The S, Sdg, CS and CSdg gates."""

from __future__ import annotations


import numpy

from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit._accelerate.circuit import StandardGate

_S_ARRAY = numpy.array([[1, 0], [0, 1j]])
_SDG_ARRAY = numpy.array([[1, 0], [0, -1j]])


@with_gate_array(_S_ARRAY)
class SGate(SingletonGate):
    r"""Single qubit S gate (:math:`\sqrt{Z}`).

    It induces a :math:`\pi/2` phase, and is sometimes called the P gate (phase).

    This is a Clifford gate and a square-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.s` method.

    Matrix representation:

    .. math::

        S = \begin{pmatrix}
                1 & 0 \\
                0 & i
            \end{pmatrix}

    Circuit symbol:

    .. code-block:: text

             ┌───┐
        q_0: ┤ S ├
             └───┘

    Equivalent to a :math:`\pi/2` radian rotation about the Z axis.
    """

    _standard_gate = StandardGate.S

    def __init__(self, label: str | None = None):
        """
        Args:
            label: An optional label for the gate.
        """
        super().__init__("s", 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌────────┐
        # q: ┤ P(π/2) ├
        #    └────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.S._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the S gate.

        For a single control qubit, the controlled gate is implemented as :class:`.CSGate`,
        regardless of the value of ``annotated``.

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
            gate = CSGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        """Return inverse of S (:class:`.SdgGate`).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.SdgGate`.

        Returns:
            SdgGate: inverse of :class:`.SGate`
        """
        return SdgGate()

    def power(self, exponent: float, annotated: bool = False):
        from .p import PhaseGate

        return PhaseGate(0.5 * numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, SGate)


@with_gate_array(_SDG_ARRAY)
class SdgGate(SingletonGate):
    r"""Single qubit S-adjoint gate (:math:`S^\dagger`).

    It induces a :math:`-\pi/2` phase.

    This is a Clifford gate and a square-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.sdg` method.

    Matrix representation:

    .. math::

        S^\dagger = \begin{pmatrix}
                1 & 0 \\
                0 & -i
            \end{pmatrix}

    Circuit symbol:

    .. code-block:: text

             ┌─────┐
        q_0: ┤ Sdg ├
             └─────┘

    Equivalent to a :math:`-\pi/2` radian rotation about the Z axis.
    """

    _standard_gate = StandardGate.Sdg

    def __init__(self, label: str | None = None):
        """
        Args:
            label: An optional label for the gate.
        """
        super().__init__("sdg", 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌─────────┐
        # q: ┤ P(-π/2) ├
        #    └─────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.Sdg._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        r"""Return a controlled version of the :math:`S^\dagger` gate.

        For a single control qubit, the controlled gate is implemented as :class:`.CSdgGate`,
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
            gate = CSdgGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverse of :math:`S^\dagger` (SGate).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.SGate`.

        Returns:
            SGate: inverse of :class:`.SdgGate`
        """
        return SGate()

    def power(self, exponent: float, annotated: bool = False):
        from .p import PhaseGate

        return PhaseGate(-0.5 * numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, SdgGate)


@with_controlled_gate_array(_S_ARRAY, num_ctrl_qubits=1)
class CSGate(SingletonControlledGate):
    r"""Controlled-S gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cs` method.

    Circuit symbol:

    .. code-block:: text

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ S ├
             └───┘

    Matrix representation:

    .. math::

        CS \ q_0, q_1 =
        I \otimes |0 \rangle\langle 0| + S \otimes |1 \rangle\langle 1|  =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & i
            \end{pmatrix}
    """

    _standard_gate = StandardGate.CS

    def __init__(
        self,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create new CS gate."""
        super().__init__(
            "cs",
            2,
            [],
            label=label,
            num_ctrl_qubits=1,
            ctrl_state=ctrl_state,
            base_gate=SGate(label=_base_label),
            _base_label=_base_label,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #      ┌───┐
        # q_0: ┤ T ├──■───────────■───────
        #      └───┘┌─┴─┐┌─────┐┌─┴─┐┌───┐
        # q_1: ─────┤ X ├┤ Tdg ├┤ X ├┤ T ├
        #           └───┘└─────┘└───┘└───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CS._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        """Return inverse of CSGate (CSdgGate).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CSdgGate`.

        Returns:
            CSdgGate: inverse of :class:`.CSGate`
        """
        return CSdgGate(ctrl_state=self.ctrl_state)

    def power(self, exponent: float, annotated: bool = False):
        from .p import CPhaseGate

        return CPhaseGate(0.5 * numpy.pi * exponent, ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, CSGate) and self.ctrl_state == other.ctrl_state


@with_controlled_gate_array(_SDG_ARRAY, num_ctrl_qubits=1)
class CSdgGate(SingletonControlledGate):
    r"""Controlled-:math:`S^\dagger` gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.csdg` method.

    Circuit symbol:

    .. code-block:: text

        q_0: ───■───
             ┌──┴──┐
        q_1: ┤ Sdg ├
             └─────┘

    Matrix representation:

    .. math::

        CS^\dagger \ q_0, q_1 =
        I \otimes |0 \rangle\langle 0| + S^\dagger \otimes |1 \rangle\langle 1|  =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & -i
            \end{pmatrix}
    """

    _standard_gate = StandardGate.CSdg

    def __init__(
        self,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create new CSdg gate."""
        super().__init__(
            "csdg",
            2,
            [],
            label=label,
            num_ctrl_qubits=1,
            ctrl_state=ctrl_state,
            base_gate=SdgGate(label=_base_label),
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #      ┌─────┐
        # q_0: ┤ Tdg ├──■─────────■─────────
        #      └─────┘┌─┴─┐┌───┐┌─┴─┐┌─────┐
        # q_1: ───────┤ X ├┤ T ├┤ X ├┤ Tdg ├
        #             └───┘└───┘└───┘└─────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CSdg._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        """Return inverse of CSdgGate (CSGate).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CSGate`.

        Returns:
            CSGate: inverse of :class:`.CSdgGate`
        """
        return CSGate(ctrl_state=self.ctrl_state)

    def power(self, exponent: float, annotated: bool = False):
        from .p import CPhaseGate

        return CPhaseGate(-0.5 * numpy.pi * exponent, ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, CSdgGate) and self.ctrl_state == other.ctrl_state
