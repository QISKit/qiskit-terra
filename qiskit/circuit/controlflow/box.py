# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Simple box basic block."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from qiskit.circuit.delay import Delay
from qiskit.circuit.exceptions import CircuitError
from qiskit._accelerate.circuit import ControlFlowType
from .control_flow import ControlFlowOp

if typing.TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit, Annotation
    from qiskit.circuit.classical import expr


class BoxOp(ControlFlowOp):
    """A scoped "box" of operations on a circuit that are treated atomically in the greater context.

    A "box" is a control-flow construct that is entered unconditionally.  The contents of the box
    behave somewhat as if the start and end of the box were barriers, except it is permissible to
    commute operations "all the way" through the box.  The box is also an explicit scope for the
    purposes of variables, stretches and compiler passes.

    A box may be "annotated" with arbitrary user-defined custom :class:`.Annotation` objects.  In
    cases where order is important, these should be interpreted by applying the first annotation in
    the list first, then the second, and so on.  It is generally recommended that annotations should
    not be order-dependent, wherever possible.

    Typically you create this by using the builder-interface form of :meth:`.QuantumCircuit.box`.
    """

    _control_flow_type = ControlFlowType.Box

    def __init__(
        self,
        body: QuantumCircuit,
        duration: None = None,
        unit: typing.Literal["dt", "s", "ms", "us", "ns", "ps", "expr"] | None = None,
        label: str | None = None,
        annotations: typing.Iterable[Annotation] = (),
    ):
        """
        Default constructor of :class:`BoxOp`.

        Args:
            body: the circuit to use as the body of the box.  This should explicitly close over any
                :class:`.expr.Var` variables that must be incident from the outer circuit.  The
                required number of qubits and clbits for the resulting instruction are inferred from
                the number in the circuit, even if they are idle.
            duration: an optional duration for the box as a whole.
            unit: the unit of the ``duration``.
            label: an optional string label for the instruction.
            annotations: any :class:`.Annotation`\\ s to apply to the box.  In cases where order
                is important, annotations are to be interpreted in the same order they appear in
                the iterable.
        """
        super().__init__("box", body.num_qubits, body.num_clbits, [body], label=label)
        self.annotations = list(annotations)
        self.duration, self.unit = Delay._validate_arguments(duration, unit)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, parameters):

        from qiskit.circuit import QuantumCircuit

        (body,) = parameters

        if not isinstance(body, QuantumCircuit):
            raise CircuitError(
                "BoxOp expects a body parameter of type "
                f"QuantumCircuit, but received {type(body)}."
            )
        if body.num_input_vars:
            raise self._unexpected_input_var_error()

        if body.num_qubits != self.num_qubits or body.num_clbits != self.num_clbits:
            raise CircuitError(
                "Attempted to assign a body parameter with a num_qubits or "
                "num_clbits different than that of the BoxOp. "
                f"BoxOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} "
                f"Supplied body num_qubits/clbits: {body.num_qubits}/{body.num_clbits}."
            )

        self._params = [body]

    @property
    def body(self):
        """The ``body`` :class:`.QuantumCircuit` of the operation.

        This is the same object returned as the sole entry in :meth:`params` and :meth:`blocks`.
        """
        # Not settable via this property; the only meaningful way to replace a body is via
        # larger `QuantumCircuit` methods, or using `replace_blocks`.
        return self.params[0]

    @property
    def blocks(self):
        return (self._params[0],)

    def replace_blocks(self, blocks):
        (body,) = blocks
        return BoxOp(
            body,
            duration=self.duration,
            unit=self.unit,
            label=self.label,
            annotations=self.annotations,
        )

    def substitute(self, substitutions: Mapping[expr.Var, expr.Expr]) -> BoxOp:
        """Return a new :class:`BoxOp` with classical :class:`~.expr.Var` nodes replaced.

        A box has no classical condition of its own, so the substitution is applied
        recursively to the box body via :meth:`.QuantumCircuit.substitute_vars`.

        Args:
            substitutions: mapping from :class:`~.expr.Var` to the replacement
                :class:`~.expr.Expr`.

        Returns:
            A new :class:`BoxOp` with the substitutions applied.
        """
        return self.replace_blocks([self.blocks[0].substitute_vars(substitutions, strict=False)])

    def __eq__(self, other):
        return (
            isinstance(other, BoxOp)
            and self.duration == other.duration
            and self.unit == other.unit
            and self.annotations == other.annotations
            and super().__eq__(other)
        )


class BoxContext:
    """Context-manager that powers :meth:`.QuantumCircuit.box`.

    This is not part of the public interface, and should not be instantiated by users.
    """

    __slots__ = ("_annotations", "_circuit", "_duration", "_label", "_unit")

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        duration: None = None,
        unit: typing.Literal["dt", "s", "ms", "us", "ns", "ps"] = "dt",
        label: str | None = None,
        annotations: typing.Iterable[Annotation] = (),
    ):
        """
        Args:
            circuit: the outermost scope of the circuit under construction.
            duration: the final duration of the box.
            unit: the unit of ``duration``.
            label: an optional label for the box.
            annotations: any :class:`.Annotation`\\ s to apply to the box.  In cases where order
                is important, annotations are to be interpreted in the same order they appear in
                the iterable.
        """
        self._circuit = circuit
        self._duration = duration
        self._unit = unit
        self._label = label
        self._annotations = annotations

    def __enter__(self):
        # For a box to have the semantics of internal qubit alignment with a resolvable duration, we
        # can't allow conditional jumps to exit it.  Technically an unconditional `break` or
        # `continue` could work, but we're not getting into that.
        self._circuit._push_scope(allow_jumps=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # If we're leaving the context manager because an exception was raised, there's nothing
            # to do except restore the circuit state.
            self._circuit._pop_scope()
            return False
        scope = self._circuit._pop_scope()
        # Boxes do not need to pass any further resources in, because there's no jumps out of a
        # `box` permitted.
        body = scope.build(scope.qubits(), scope.clbits())
        self._circuit.append(
            BoxOp(
                body,
                duration=self._duration,
                unit=self._unit,
                label=self._label,
                annotations=self._annotations,
            ),
            body.qubits,
            body.clbits,
        )
        return False
