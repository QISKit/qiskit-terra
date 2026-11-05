# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Annotated Operations."""

from __future__ import annotations

import dataclasses


from qiskit.circuit.operation import Operation
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int
from qiskit.circuit.exceptions import CircuitError


class Modifier:
    """The base class that all modifiers of :class:`~.AnnotatedOperation` should
    inherit from."""


@dataclasses.dataclass
class InverseModifier(Modifier):
    """Inverse modifier: specifies that the operation is inverted."""


@dataclasses.dataclass
class ControlModifier(Modifier):
    """Control modifier: specifies that the operation is controlled by ``num_ctrl_qubits``
    and has control state ``ctrl_state``."""

    num_ctrl_qubits: int = 0
    ctrl_state: int | str | None = None

    def __init__(self, num_ctrl_qubits: int = 0, ctrl_state: int | str | None = None):
        if num_ctrl_qubits < 0:
            raise CircuitError("The number of control qubits must be non-negative.")

        self.num_ctrl_qubits = num_ctrl_qubits
        self.ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)


@dataclasses.dataclass
class PowerModifier(Modifier):
    """Power modifier: specifies that the operation is raised to the power ``power``."""

    power: float


class AnnotatedOperation(Operation):
    """Annotated operation."""

    def __init__(self, base_op: Operation, modifiers: Modifier | list[Modifier]):
        """
        Create a new AnnotatedOperation.

        An "annotated operation" allows to add a list of modifiers to the
        "base" operation. For now, the only supported modifiers are of
        types :class:`~.InverseModifier`, :class:`~.ControlModifier` and
        :class:`~.PowerModifier`.

        An annotated operation can be viewed as an extension of
        :class:`~.ControlledGate` (which also allows adding control to the
        base operation). However, an important difference is that the
        circuit definition of an annotated operation is not constructed when
        the operation is declared, and instead happens during transpilation,
        specifically during the :class:`~.HighLevelSynthesis` transpiler pass.

        An annotated operation can be also viewed as a "higher-level"
        or "more abstract" object that can be added to a quantum circuit.
        This enables writing transpiler optimization passes that make use of
        this higher-level representation, for instance removing a gate
        that is immediately followed by its inverse.

        Args:
            base_op: base operation being modified
            modifiers: ordered list of modifiers. Supported modifiers include
                ``InverseModifier``, ``ControlModifier`` and ``PowerModifier``.

        Examples::

            op1 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(2)])

            op2_inner = AnnotatedGate(SGate(), InverseModifier())
            op2 = AnnotatedGate(op2_inner, ControlModifier(2))

        Both op1 and op2 are semantically equivalent to an ``SGate()`` which is first
        inverted and then controlled by 2 qubits.
        """
        self.base_op = base_op
        """The base operation that the modifiers in this annotated operation apply to."""
        self.modifiers = modifiers if isinstance(modifiers, list) else [modifiers]
        """Ordered sequence of the modifiers to apply to :attr:`base_op`.  The modifiers are applied
        in order from lowest index to highest index."""

    @property
    def name(self) -> str:
        """Unique string identifier for operation type."""
        return "annotated"

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        num_ctrl_qubits = 0
        for modifier in self.modifiers:
            if isinstance(modifier, ControlModifier):
                num_ctrl_qubits += modifier.num_ctrl_qubits

        return num_ctrl_qubits + self.base_op.num_qubits

    @property
    def num_clbits(self) -> int:
        """Number of classical bits."""
        return self.base_op.num_clbits

    def __eq__(self, other) -> bool:
        """Checks if two AnnotatedOperations are equal."""
        return (
            isinstance(other, AnnotatedOperation)
            and self.modifiers == other.modifiers
            and self.base_op == other.base_op
        )

    def copy(self) -> AnnotatedOperation:
        """Return a copy of the :class:`~.AnnotatedOperation`."""
        return AnnotatedOperation(base_op=self.base_op.copy(), modifiers=self.modifiers.copy())

    def to_matrix(self):
        """Return a matrix representation (allowing to construct Operator)."""
        from qiskit.quantum_info.operators import Operator

        operator = Operator(self.base_op)

        for modifier in self.modifiers:
            if isinstance(modifier, InverseModifier):
                operator = operator.power(-1)
            elif isinstance(modifier, ControlModifier):
                operator = Operator(
                    _compute_control_matrix(
                        operator.data, modifier.num_ctrl_qubits, modifier.ctrl_state
                    )
                )
            elif isinstance(modifier, PowerModifier):
                operator = operator.power(modifier.power)
            else:
                raise CircuitError(f"Unknown modifier {modifier}.")
        return operator

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ) -> AnnotatedOperation:
        """Return the controlled version of itself.

        Implemented as :class:`.AnnotatedOperation`, regardless of the value of
        ``annotated``.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Ignored.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``.
            annotated: Ignored.

        Returns:
            A controlled version of the given operation.
        """

        extended_modifiers = self.modifiers.copy()
        extended_modifiers.append(
            ControlModifier(num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state)
        )
        return AnnotatedOperation(self.base_op, extended_modifiers)

    def inverse(self, annotated: bool = True):
        """
        Return the inverse version of itself.

        Implemented as an annotated operation, see  :class:`.AnnotatedOperation`.

        Args:
            annotated: ignored (used for consistency with other inverse methods)

        Returns:
            Inverse version of the given operation.
        """

        extended_modifiers = self.modifiers.copy()
        extended_modifiers.append(InverseModifier())
        return AnnotatedOperation(self.base_op, extended_modifiers)

    def power(self, exponent: float, annotated: bool = False):
        """
        Raise this gate to the power of ``exponent``.

        Implemented as an annotated operation, see  :class:`.AnnotatedOperation`.

        Args:
            exponent: the power to raise the gate to
            annotated: ignored (used for consistency with other power methods)

        Returns:
            An operation implementing ``gate^exponent``
        """

        extended_modifiers = self.modifiers.copy()
        extended_modifiers.append(PowerModifier(exponent))
        return AnnotatedOperation(self.base_op, extended_modifiers)

    @property
    def params(self) -> list[ParameterValueType]:
        """The params of the underlying base operation."""
        return getattr(self.base_op, "params", [])

    @params.setter
    def params(self, value: list[ParameterValueType]):
        if hasattr(self.base_op, "params"):
            self.base_op.params = value
        else:
            raise AttributeError(
                f"Cannot set attribute ``params`` on the base operation {self.base_op}."
            )

    def validate_parameter(self, parameter: ParameterValueType) -> ParameterValueType:
        """Validate a parameter for the underlying base operation."""
        if hasattr(self.base_op, "validate_parameter"):
            return self.base_op.validate_parameter(parameter)

        raise AttributeError(f"Cannot validate parameters on the base operation {self.base_op}.")


def _canonicalize_modifiers(modifiers):
    """
    Returns the canonical representative of the modifier list. This is possible
    since all the modifiers commute; also note that InverseModifier is a special
    case of PowerModifier. The current solution is to compute the total number
    of control qubits / control state and the total power. The InverseModifier
    will be present if total power is negative, whereas the power modifier will
    be present only with positive powers different from 1.
    """
    power = 1
    num_ctrl_qubits = 0
    ctrl_state = 0

    for modifier in modifiers:
        if isinstance(modifier, InverseModifier):
            power *= -1
        elif isinstance(modifier, ControlModifier):
            num_ctrl_qubits += modifier.num_ctrl_qubits
            ctrl_state = (ctrl_state << modifier.num_ctrl_qubits) | modifier.ctrl_state
        elif isinstance(modifier, PowerModifier):
            power *= modifier.power
        else:
            raise CircuitError(f"Unknown modifier {modifier}.")

    canonical_modifiers = []
    if power < 0:
        canonical_modifiers.append(InverseModifier())
        power *= -1

    if power != 1:
        canonical_modifiers.append(PowerModifier(power))
    if num_ctrl_qubits > 0:
        canonical_modifiers.append(ControlModifier(num_ctrl_qubits, ctrl_state))

    return canonical_modifiers
