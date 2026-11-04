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

"""Circuit operation representing an ``switch/case`` statement."""

from __future__ import annotations

__all__ = ("CASE_DEFAULT", "SwitchCaseOp")

import contextlib
from typing import Any, Literal, TYPE_CHECKING
from collections.abc import Iterable

from qiskit.circuit import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from qiskit._accelerate.circuit import ControlFlowType

from .builder import InstructionPlaceholder, InstructionResources, ControlFlowBuilderBlock
from .control_flow import ControlFlowOp
from ._builder_utils import unify_circuit_resources, partition_registers, node_resources

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


class _DefaultCaseType:
    # Note: Sphinx uses the docstring of this singleton class object as the documentation of the
    # `CASE_DEFAULT` object.

    """A special object that represents the "default" case of a switch statement.  If you use this
    as a case target, it must be the last case, and will match anything that wasn't already matched.
    When using the builder interface of :meth:`.QuantumCircuit.switch`, this can also be accessed as
    the ``DEFAULT`` attribute of the bound case-builder object."""

    def __repr__(self):
        return "<default case>"


CASE_DEFAULT = _DefaultCaseType()


class SwitchCaseOp(ControlFlowOp):
    """A circuit operation that executes one particular circuit block based on matching a given
    ``target`` against an ordered list of ``values``.  The special value :data:`.CASE_DEFAULT` can
    be used to represent a default condition.
    """

    _control_flow_type = ControlFlowType.SwitchCase

    def __init__(
        self,
        target: Clbit | ClassicalRegister | expr.Expr,
        cases: Iterable[tuple[Any, QuantumCircuit]],
        *,
        label: str | None = None,
    ):
        """
        Args:
            target: the real-time value to switch on.
            cases: an ordered iterable of the corresponding value of the ``target`` and the circuit
                block that should be executed if this is matched.  There is no fall-through between
                blocks, and the order matters.
            label: An optional label for identifying the instruction.
        """

        from qiskit.circuit import QuantumCircuit

        if isinstance(target, expr.Expr):
            if target.type.kind not in (types.Uint, types.Bool):
                raise CircuitError(
                    "the switch target must be an expression with type 'Uint(n)' or 'Bool()',"
                    f" not '{target.type}'"
                )
        elif not isinstance(target, (Clbit, ClassicalRegister)):
            raise CircuitError("the switch target must be a classical bit or register")

        if isinstance(target, expr.Expr):
            target_bits = 1 if target.type.kind is types.Bool else target.type.width
        else:
            target_bits = 1 if isinstance(target, Clbit) else len(target)
        target_max = (1 << target_bits) - 1

        case_ids = set()
        num_qubits, num_clbits = None, None
        self.target = target
        self._case_map = {}
        """Mapping of individual jump values to block indices.  This level of indirection is to let
        us more easily track the case of multiple labels pointing to the same circuit object, so
        it's easier for things like `assign_parameters`, which need to touch each circuit object
        exactly once, to function."""
        self._label_spec: list[tuple[int | Literal[CASE_DEFAULT], ...]] = []
        """List of the normalized jump value specifiers.  This is a list of tuples, where each tuple
        contains the values, and the indexing is the same as the values of `_case_map` and
        `_params`."""
        self._params = []
        """List of the circuit bodies used.  This form makes it simpler for things like
        :meth:`.replace_blocks` and :class:`.QuantumCircuit.assign_parameters` to do their jobs
        without accidentally mutating the same circuit instance more than once."""
        for i, (value_spec, case_) in enumerate(cases):
            values = tuple(value_spec) if isinstance(value_spec, (tuple, list)) else (value_spec,)
            for value in values:
                if value in self._case_map:
                    raise CircuitError(f"duplicate case value {value}")
                if CASE_DEFAULT in self._case_map:
                    raise CircuitError("cases after the default are unreachable")
                if value is not CASE_DEFAULT:
                    if not isinstance(value, int) or value < 0:
                        raise CircuitError("case values must be Booleans or non-negative integers")
                    if value > target_max:
                        raise CircuitError(
                            f"switch target '{target}' has {target_bits} bit(s) of precision,"
                            f" but case {value} is larger than the maximum of {target_max}."
                        )
                self._case_map[value] = i
            self._label_spec.append(values)
            if not isinstance(case_, QuantumCircuit):
                raise CircuitError("case blocks must be QuantumCircuit instances")
            if id(case_) in case_ids:
                raise CircuitError("ungrouped cases cannot point to the same block")
            case_ids.add(id(case_))
            if num_qubits is None:
                num_qubits, num_clbits = case_.num_qubits, case_.num_clbits
            if case_.num_input_vars:
                raise self._unexpected_input_var_error()
            if case_.num_qubits != num_qubits or case_.num_clbits != num_clbits:
                raise CircuitError("incompatible bits between cases")
            self._params.append(case_)
        if not self._params:
            # This condition also implies that `num_qubits` and `num_clbits` must be non-None.
            raise CircuitError("must have at least one case to run")

        super().__init__("switch_case", num_qubits, num_clbits, self._params, label=label)

    def __eq__(self, other):
        # The general __eq__ will compare the blocks in the right order, so we just need to ensure
        # that all the labels point the right way as well.
        return (
            super().__eq__(other)
            and self.target == other.target
            and all(
                set(labels_self) == set(labels_other)
                for labels_self, labels_other in zip(self._label_spec, other._label_spec)
            )
        )

    def cases_specifier(self) -> Iterable[tuple[tuple, QuantumCircuit]]:
        """Return an iterable where each element is a 2-tuple whose first element is a tuple of
        jump values, and whose second is the single circuit block that is associated with those
        values.

        This is an abstract specification of the jump table suitable for creating new
        :class:`.SwitchCaseOp` instances.

        .. seealso::
            :meth:`.SwitchCaseOp.cases`
                Create a lookup table that you can use for your own purposes to jump from values to
                the circuit that would be executed."""
        return zip(self._label_spec, self._params)

    def cases(self):
        """Return a lookup table from case labels to the circuit that would be executed in that
        case.  This object is not generally suitable for creating a new :class:`.SwitchCaseOp`
        because any keys that point to the same object will not be grouped.

        .. seealso::
            :meth:`.SwitchCaseOp.cases_specifier`
                An alternate method that produces its output in a suitable format for creating new
                :class:`.SwitchCaseOp` instances.
        """
        return {key: self._params[index] for key, index in self._case_map.items()}

    @property
    def blocks(self):
        return tuple(self._params)

    def replace_blocks(self, blocks: Iterable[QuantumCircuit]) -> SwitchCaseOp:
        blocks = tuple(blocks)
        if len(blocks) != len(self._params):
            raise CircuitError(f"needed {len(self._case_map)} blocks but received {len(blocks)}")
        return SwitchCaseOp(self.target, zip(self._label_spec, blocks))


class SwitchCasePlaceholder(InstructionPlaceholder):
    """A placeholder instruction to use in control-flow context managers, when calculating the
    number of resources this instruction should block is deferred until the construction of the
    outer loop.

    This generally should not be instantiated manually; only :obj:`.SwitchContext` should do it when
    it needs to defer creation of the concrete instruction.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    def __init__(
        self,
        target: Clbit | ClassicalRegister | expr.Expr,
        cases: list[tuple[Any, ControlFlowBuilderBlock]],
        *,
        label: str | None = None,
    ):
        self.__target = target
        self.__cases = cases
        self.__resources = self._calculate_placeholder_resources()
        super().__init__(
            "switch_case",
            len(self.__resources.qubits),
            len(self.__resources.clbits),
            [],
            label=label,
        )

    def _calculate_placeholder_resources(self):
        qubits = set()
        clbits = set()
        qregs = set()
        cregs = set()
        if isinstance(self.__target, Clbit):
            clbits.add(self.__target)
        elif isinstance(self.__target, ClassicalRegister):
            clbits.update(self.__target)
            cregs.add(self.__target)
        else:
            resources = node_resources(self.__target)
            clbits.update(resources.clbits)
            cregs.update(resources.cregs)
        for _, body in self.__cases:
            qubits |= body.qubits()
            clbits |= body.clbits()
            body_qregs, body_cregs = partition_registers(body.registers)
            qregs |= body_qregs
            cregs |= body_cregs
        return InstructionResources(
            qubits=tuple(qubits),
            clbits=tuple(clbits),
            qregs=tuple(qregs),
            cregs=tuple(cregs),
        )

    def placeholder_resources(self):
        return self.__resources

    def concrete_instruction(self, qubits, clbits):
        cases = [
            (labels, unified_body)
            for (labels, _), unified_body in zip(
                self.__cases,
                unify_circuit_resources(body.build(qubits, clbits) for _, body in self.__cases),
            )
        ]
        if cases:
            resources = InstructionResources(
                qubits=tuple(cases[0][1].qubits),
                clbits=tuple(cases[0][1].clbits),
                qregs=tuple(cases[0][1].qregs),
                cregs=tuple(cases[0][1].cregs),
            )
        else:
            resources = self.__resources
        return (
            SwitchCaseOp(self.__target, cases, label=self.label),
            resources,
        )


class SwitchContext:
    """A context manager for building up ``switch`` statements onto circuits in a natural order,
    without having to construct the case bodies first.

    The return value of this context manager can be used within the created context to build up the
    individual ``case`` statements.  No other instructions should be appended to the circuit during
    the `switch` context.

    This context should almost invariably be created by a :meth:`.QuantumCircuit.switch_case` call,
    and the resulting instance is a "friend" of the calling circuit.  The context will manipulate
    the circuit's defined scopes when it is entered (by pushing a new scope onto the stack) and
    exited (by popping its scope, building it, and appending the resulting :obj:`.SwitchCaseOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        target: Clbit | ClassicalRegister | expr.Expr,
        *,
        in_loop: bool,
        label: str | None = None,
    ):
        self.circuit = circuit
        self._target = target
        if isinstance(target, Clbit):
            self.target_clbits: tuple[Clbit, ...] = (target,)
            self.target_cregs: tuple[ClassicalRegister, ...] = ()
        elif isinstance(target, ClassicalRegister):
            self.target_clbits = tuple(target)
            self.target_cregs = (target,)
        else:
            resources = node_resources(target)
            self.target_clbits = resources.clbits
            self.target_cregs = resources.cregs
        self.in_loop = in_loop
        self.complete = False
        self._op_label = label
        self._cases: list[tuple[tuple[Any, ...], ControlFlowBuilderBlock]] = []
        self._label_set = set()

    def label_in_use(self, label):
        """Return whether a case label is already accounted for in the switch statement."""
        return label in self._label_set

    def add_case(
        self, labels: tuple[int | Literal[CASE_DEFAULT], ...], block: ControlFlowBuilderBlock
    ):
        """Add a sequence of conditions and the single block that should be run if they are
        triggered to the context.  The labels are assumed to have already been validated using
        :meth:`label_in_use`."""
        # The labels were already validated when the case scope was entered, so we don't need to do
        # it again.
        self._label_set.update(labels)
        self._cases.append((labels, block))

    def __enter__(self):
        self.circuit._push_scope(forbidden_message="Cannot have instructions outside a case")
        return CaseBuilder(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.complete = True
        # The popped scope should be the forbidden scope.
        self.circuit._pop_scope()
        if exc_type is not None:
            return False
        # If we're in a loop-builder context, we need to emit a placeholder so that any `break` or
        # `continue`s in any of our cases can be expanded when the loop-builder.  If we're not, we
        # need to emit a concrete instruction immediately.
        placeholder = SwitchCasePlaceholder(self._target, self._cases, label=self._op_label)
        initial_resources = placeholder.placeholder_resources()
        if self.in_loop:
            self.circuit.append(placeholder, initial_resources.qubits, initial_resources.clbits)
        else:
            operation, resources = placeholder.concrete_instruction(
                set(initial_resources.qubits), set(initial_resources.clbits)
            )
            self.circuit.append(operation, resources.qubits, resources.clbits)
        return False


class CaseBuilder:
    """A child context manager for building up the ``case`` blocks of ``switch`` statements onto
    circuits in a natural order, without having to construct the case bodies first.

    This context should never need to be created manually by a user; it is the return value of the
    :class:`.SwitchContext` context manager, which in turn should only be created by suitable
    :meth:`.QuantumCircuit.switch_case` calls.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    DEFAULT = CASE_DEFAULT
    """Convenient re-exposure of the :data:`.CASE_DEFAULT` constant."""

    def __init__(self, parent: SwitchContext):
        self.switch = parent
        self.entered = False

    @contextlib.contextmanager
    def __call__(self, *values):
        if self.entered:
            raise CircuitError(
                "Cannot enter more than one case at once."
                " If you want multiple labels to point to the same block,"
                " pass them all to a single case context,"
                " such as `with case(1, 2, 3):`."
            )
        if self.switch.complete:
            raise CircuitError("Cannot add a new case to a completed switch statement.")
        if not all(value is CASE_DEFAULT or isinstance(value, int) for value in values):
            raise CircuitError("Case values must be integers or `CASE_DEFAULT`")
        seen = set()
        for value in values:
            if self.switch.label_in_use(value) or value in seen:
                raise CircuitError(f"duplicate case label: '{value}'")
            seen.add(value)
        self.switch.circuit._push_scope(
            clbits=self.switch.target_clbits,
            registers=self.switch.target_cregs,
            allow_jumps=self.switch.in_loop,
        )

        try:
            self.entered = True
            yield
        finally:
            self.entered = False
            block = self.switch.circuit._pop_scope()

        # This is outside the `finally` because we only want to add the case to the switch if we're
        # leaving it under normal circumstances.  If there was an exception in the case block, we
        # should discard anything happened during its construction.
        self.switch.add_case(values, block)
