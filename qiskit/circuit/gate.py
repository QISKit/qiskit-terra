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

"""Unitary gate."""

from __future__ import annotations
from collections.abc import Iterator, Iterable
import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_arg
from .annotated_operation import AnnotatedOperation, ControlModifier, PowerModifier
from .instruction import Instruction


class Gate(Instruction):
    """Unitary gate."""

    def __init__(
        self,
        name: str,
        num_qubits: int,
        params: list,
        label: str | None = None,
    ) -> None:
        """
        Args:
            name: The name of the gate.
            num_qubits: The number of qubits the gate acts on.
            params: A list of parameters.
            label: An optional label for the gate.
        """
        self.definition = None
        super().__init__(name, num_qubits, 0, params, label=label)

    # Set higher priority than Numpy array and matrix classes
    __array_priority__ = 20

    def to_matrix(self) -> np.ndarray:
        """Return a Numpy.array for the gate unitary matrix.

        Returns:
            np.ndarray: if the Gate subclass has a matrix definition.

        Raises:
            CircuitError: If a Gate subclass does not implement this method an
                exception will be raised when this base class method is called.
        """
        if hasattr(self, "__array__"):
            return self.__array__(dtype=complex)
        raise CircuitError(f"to_matrix not defined for this {type(self)}")

    def power(self, exponent: float, annotated: bool = False):
        """Raise this gate to the power of ``exponent``.

        Implemented either as a unitary gate (ref. :class:`~.library.UnitaryGate`)
        or as an annotated operation (ref. :class:`.AnnotatedOperation`). In the case of several standard
        gates, such as :class:`.RXGate`, when the power of a gate can be expressed in terms of another
        standard gate that is returned directly.

        Args:
            exponent (float): the power to raise the gate to
            annotated (bool): indicates whether the power gate can be implemented
                as an annotated operation. In the case of several standard
                gates, such as :class:`.RXGate`, this argument is ignored when
                the power of a gate can be expressed in terms of another
                standard gate.

        Returns:
            An operation implementing ``gate^exponent``

        Raises:
            CircuitError: If gate is not unitary
        """

        from qiskit.quantum_info.operators import Operator
        from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate

        if not annotated:
            return UnitaryGate(
                Operator(self).power(exponent, assume_unitary=True), label=f"{self.name}^{exponent}"
            )
        else:
            return AnnotatedOperation(self, PowerModifier(exponent))

    def __pow__(self, exponent: float) -> Gate:
        return self.power(exponent)

    def _return_repeat(self, exponent: float) -> Gate:
        gate = Gate(name=f"{self.name}*{exponent}", num_qubits=self.num_qubits, params=[])
        gate.validate_parameter = self.validate_parameter
        gate.params = self.params
        return gate

    @deprecate_arg(
        name="annotated",
        since="2.3",
        additional_msg=(
            "The method Gate.control() no longer accepts `annotated=None`. The new default is "
            "`annotated=True`, which represents the controlled gate as an `AnnotatedOperation` "
            "(unless a dedicated controlled-gate class already exists). You can explicitly set "
            "`annotated=False` to preserve the previous behavior. However, using `annotated=True` "
            "is recommended, as it defers construction of the controlled circuit to transpiler, "
            "and furthermore enables additional controlled-gate optimizations (typically leading "
            "to higher-quality circuits)."
        ),
        predicate=lambda my_arg: my_arg is None,
        removal_timeline="in Qiskit 3.0",
    )
    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Return the controlled version of itself.

        The controlled gate is implemented as :class:`.ControlledGate` when ``annotated``
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

        Raises:
            QiskitError: invalid ``num_ctrl_qubits`` or ``ctrl_state``.
        """
        if num_ctrl_qubits < 0:
            raise CircuitError("The number of control qubits must be non-negative.")

        # In the special case that we have 0 control qubits, we return the copy of the gate itself.
        if num_ctrl_qubits == 0:
            return self.copy()

        if not annotated:  # captures both None and False

            from ._add_control import add_control

            return add_control(self, num_ctrl_qubits, label, ctrl_state)

        else:
            return AnnotatedOperation(
                self, ControlModifier(num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state)
            )

    @staticmethod
    def _broadcast_single_argument(qarg: list) -> Iterator[tuple[list, list]]:
        """Expands a single argument.

        For example: [q[0], q[1]] -> [q[0]], [q[1]]
        """
        # [q[0], q[1]] -> [q[0]]
        #              -> [q[1]]
        for arg0 in qarg:
            yield [arg0], []

    @staticmethod
    def _broadcast_2_arguments(qarg0: list, qarg1: list) -> Iterator[tuple[list, list]]:
        if len(qarg0) == len(qarg1):
            # [[q[0], q[1]], [r[0], r[1]]] -> [q[0], r[0]]
            #                              -> [q[1], r[1]]
            for arg0, arg1 in zip(qarg0, qarg1):
                yield [arg0, arg1], []
        elif len(qarg0) == 1:
            # [[q[0]], [r[0], r[1]]] -> [q[0], r[0]]
            #                        -> [q[0], r[1]]
            for arg1 in qarg1:
                yield [qarg0[0], arg1], []
        elif len(qarg1) == 1:
            # [[q[0], q[1]], [r[0]]] -> [q[0], r[0]]
            #                        -> [q[1], r[0]]
            for arg0 in qarg0:
                yield [arg0, qarg1[0]], []
        else:
            raise CircuitError(
                f"Not sure how to combine these two-qubit arguments:\n {qarg0}\n {qarg1}"
            )

    @staticmethod
    def _broadcast_3_or_more_args(qargs: list) -> Iterator[tuple[list, list]]:
        if all(len(qarg) == len(qargs[0]) for qarg in qargs):
            for arg in zip(*qargs):
                yield list(arg), []
        else:
            raise CircuitError(f"Not sure how to combine these qubit arguments:\n {qargs}\n")

    def broadcast_arguments(self, qargs: list, cargs: list) -> Iterable[tuple[list, list]]:
        """Validation and handling of the arguments and its relationship.

        For example, ``cx([q[0],q[1]], q[2])`` means ``cx(q[0], q[2]); cx(q[1], q[2])``. This
        method yields the arguments in the right grouping. In the given example::

            in: [[q[0],q[1]], q[2]],[]
            outs: [q[0], q[2]], []
                  [q[1], q[2]], []

        The general broadcasting rules are:

            * If len(qargs) == 1::

                [q[0], q[1]] -> [q[0]],[q[1]]

            * If len(qargs) == 2::

                [[q[0], q[1]], [r[0], r[1]]] -> [q[0], r[0]], [q[1], r[1]]
                [[q[0]], [r[0], r[1]]]       -> [q[0], r[0]], [q[0], r[1]]
                [[q[0], q[1]], [r[0]]]       -> [q[0], r[0]], [q[1], r[0]]

            * If len(qargs) >= 3::

                [q[0], q[1]], [r[0], r[1]],  ...] -> [q[0], r[0], ...], [q[1], r[1], ...]

        Args:
            qargs: List of quantum bit arguments.
            cargs: List of classical bit arguments.

        Returns:
            A tuple with single arguments.

        Raises:
            CircuitError: If the input is not valid. For example, the number of
                arguments does not match the gate expectation.
        """
        if len(qargs) != self.num_qubits or cargs:
            raise CircuitError(
                f"The amount of qubit({len(qargs)})/clbit({len(cargs)}) arguments does"
                f" not match the gate expectation ({self.num_qubits})."
            )

        if any(not qarg for qarg in qargs):
            raise CircuitError("One or more of the arguments are empty")

        if len(qargs) == 0:
            return [
                ([], []),
            ]
        if len(qargs) == 1:
            return Gate._broadcast_single_argument(qargs[0])
        elif len(qargs) == 2:
            return Gate._broadcast_2_arguments(qargs[0], qargs[1])
        elif len(qargs) >= 3:
            return Gate._broadcast_3_or_more_args(qargs)
        else:
            raise CircuitError(f"This gate cannot handle {len(qargs)} arguments")

    def validate_parameter(self, parameter):
        """Gate parameters should be int, float, or ParameterExpression"""
        if isinstance(parameter, ParameterExpression):
            if len(parameter.parameters) > 0:
                return parameter  # expression has free parameters, we cannot validate it
            if not parameter.is_real():
                msg = f"Bound parameter expression is complex in gate {self.name}"
                raise CircuitError(msg)
            return parameter  # per default assume parameters must be real when bound
        if isinstance(parameter, (int, float)):
            return parameter
        elif isinstance(parameter, (np.integer, np.floating)):
            return parameter.item()
        else:
            raise CircuitError(f"Invalid param type {type(parameter)} for gate {self.name}.")
