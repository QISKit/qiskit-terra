# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Circuit operation representing a ``for`` loop."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from collections.abc import Iterable, Mapping

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.classical import expr
from qiskit.circuit.classical.types import Uint
from qiskit.circuit.exceptions import CircuitError
from qiskit._accelerate.circuit import ControlFlowType
from qiskit.circuit.classical.expr import Expr, Range, Var
from .control_flow import ControlFlowOp


if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


class ForLoopOp(ControlFlowOp):
    """A circuit operation which repeatedly executes a subcircuit
    (``body``) parameterized by a parameter ``loop_parameter`` through
    the set of integer values provided in ``indexset``.

    The ``indexset`` selects the loop semantics:

    * A Python :class:`range` or an iterable of integers is a compile-time index
      set.  The ``loop_parameter`` may be a :class:`~.Parameter` (the legacy form, unrolled by
      :class:`~qiskit.transpiler.passes.UnrollForLoops` via ``body.assign_parameters``), an
      :class:`~.expr.Var` of type :class:`~.types.Uint` (a real-time counter), or ``None``.
    * A classical :class:`~.expr.Range` (constant or dynamic) is a real-time index
      set.  The ``loop_parameter`` must be an :class:`~.expr.Var` of type :class:`~.types.Uint`
      (or ``None``); a :class:`~.Parameter` cannot represent a runtime-bound iteration value.
      A constant :class:`~.expr.Range` paired with an :class:`~.expr.Var` is still
      unrollable: :class:`~qiskit.transpiler.passes.UnrollForLoops` materializes
      each iteration value and substitutes the :class:`~.expr.Var` in every
      classical expression in the body.

    Mixing a :class:`~.Parameter` with an :class:`~.expr.Range` raises :class:`~.CircuitError`.

    Data model
    ----------

    There is exactly one "block" in a for-loop op, which is the body of the loop.  The circuit block
    may take exactly zero or one ``input`` variable (see :meth:`.QuantumCircuit.add_input_var`).  If
    the body takes one input variable, then ``loop_parameter`` must be equal to that variable, and
    it represents the loop variable: it is owned by the loop's implicit header (which assigns each
    iteration's value) and is *input* into the body, rather than being a local or captured variable
    of the body.  If the body takes zero input variables, ``loop_parameter`` may be either ``None``
    (to indicate no binding) or a :class:`.Parameter` (a legacy form that should be avoided).
    """

    _control_flow_type = ControlFlowType.ForLoop

    def __init__(
        self,
        indexset: Iterable[int] | Range,
        loop_parameter: Parameter | Var | None,
        body: QuantumCircuit,
        label: str | None = None,
    ):
        """
        Args:
            indexset: A collection of integers to loop over, as a Python
                :class:`range`, an iterable of integers, or a classical
                :class:`~.expr.Range`.
            loop_parameter: The placeholder bound to each ``indexset`` value
                inside ``body``. For a Python ``range``/integer list, may be a
                :class:`~.Parameter`, an :class:`~.expr.Var` of type :class:`~.types.Uint`,
                or ``None``. For an :class:`~.expr.Range`, must be an :class:`~.expr.Var` of
                type :class:`~.types.Uint` or ``None`` (a :class:`~.Parameter` is rejected).
                ``None`` simply repeats the body without binding any variable.
            body: The loop body to be repeatedly executed.
            label: An optional label for identifying the instruction.

        Raises:
            CircuitError: if ``loop_parameter``'s type is incompatible with the
                ``indexset`` type.
        """
        num_qubits = body.num_qubits
        num_clbits = body.num_clbits
        super().__init__(
            "for_loop", num_qubits, num_clbits, [indexset, loop_parameter, body], label=label
        )

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, parameters):
        from qiskit.circuit import QuantumCircuit

        indexset, loop_parameter, body = parameters

        if not isinstance(loop_parameter, (Parameter, Var, type(None))):
            raise CircuitError(
                "ForLoopOp expects a loop_parameter parameter to "
                "be either of type Parameter, expr.Var, or None, but received "
                f"{type(loop_parameter)}."
            )
        if isinstance(loop_parameter, expr.Var) and loop_parameter.type.kind is not Uint:
            raise CircuitError(
                "ForLoopOp expects a Var loop_parameter parameter to "
                "be of type Uint, but received "
                f"{type(loop_parameter.type)}."
            )

        if not isinstance(body, QuantumCircuit):
            raise CircuitError(
                "ForLoopOp expects a body parameter to be of type "
                f"QuantumCircuit, but received {type(body)}."
            )
        match body.num_input_vars:
            case 0:
                if isinstance(loop_parameter, expr.Var):
                    raise CircuitError("loop variable is a `Var`, but body does not contain one")
            case 1:
                (expected,) = body.iter_input_vars()
                if loop_parameter != expected:
                    raise CircuitError(
                        f"loop variable ({loop_parameter}) and body ({expected}) disagree"
                    )
            case count:
                raise CircuitError(f"body expects too many input variables ({count})")

        if body.num_qubits != self.num_qubits or body.num_clbits != self.num_clbits:
            raise CircuitError(
                "Attempted to assign a body parameter with a num_qubits or "
                "num_clbits different than that of the ForLoopOp. "
                f"ForLoopOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} "
                f"Supplied body num_qubits/clbits: {body.num_qubits}/{body.num_clbits}."
            )

        if (
            isinstance(loop_parameter, Parameter)
            and loop_parameter not in body.parameters
            and loop_parameter.name in (p.name for p in body.parameters)
        ):
            warnings.warn(
                "The Parameter provided as a loop_parameter was not found "
                "on the loop body and so no binding of the indexset to loop "
                "parameter will occur. A different Parameter of the same name "
                f"({loop_parameter.name}) was found. If you intended to loop "
                "over that Parameter, please use that Parameter instance as "
                "the loop_parameter.",
                stacklevel=2,
            )

        # An `expr.Range` has runtime bounds, so its loop variable must be a real-time `expr.Var`
        # (a compile-time `Parameter` cannot represent a runtime-bound iteration value).  All other
        # combinations are permitted: a Python `range`/integer list pairs with a `Parameter` (the
        # legacy, gate-parameter-unrolling form), with an `expr.Var` (a real-time counter over a
        # compile-time-known range), or with `None`.
        if isinstance(indexset, Range) and isinstance(loop_parameter, Parameter):
            raise CircuitError(
                "Cannot use a compile-time Parameter as a loop variable with an expr.Range "
                "indexset. Use an expr.Var instead, or use a Python range/integer list for "
                "gate-parameter unrolling."
            )

        # Consume indexset into a tuple unless it was provided as a range.
        # Preserve ranges so that they can be exported as OpenQASM 3 ranges.
        indexset = indexset if isinstance(indexset, (range, Range)) else tuple(indexset)

        self._params = [indexset, loop_parameter, body]

    @property
    def blocks(self):
        return (self._params[2],)

    def replace_blocks(self, blocks):
        (body,) = blocks
        return ForLoopOp(self.params[0], self.params[1], body, label=self.label)

    def substitute(self, substitutions: Mapping[Var, Expr]) -> ForLoopOp:
        """Return a new :class:`ForLoopOp` with classical :class:`~.expr.Var` nodes replaced.

        The substitution is applied to the ``indexset`` when it is an :class:`~.expr.Range`
        (a Python :class:`range`/integer-list indexset has no variables and is left unchanged)
        and recursively to the loop body via :meth:`.QuantumCircuit.substitute_vars`.  The
        ``loop_parameter`` itself is left untouched: it is bound by the loop rather than
        substituted away.

        Args:
            substitutions: mapping from :class:`~.expr.Var` to the replacement
                :class:`~.expr.Expr`.

        Returns:
            A new :class:`ForLoopOp` with the substitutions applied.
        """
        indexset, loop_parameter, body = self.params
        if isinstance(indexset, Range):
            indexset = indexset.substitute(substitutions)
        # The loop variable is bound by the loop header (it is the body's ``input`` variable), not
        # a free variable, so it must not be substituted out of the body: doing so would drop the
        # body's input var and leave it disagreeing with ``loop_parameter``.  Exclude it.
        body_substitutions = {
            var: replacement for var, replacement in substitutions.items() if var != loop_parameter
        }
        return ForLoopOp(
            indexset,
            loop_parameter,
            body.substitute_vars(body_substitutions, strict=False),
            label=self.label,
        )


class ForLoopContext:
    """A context manager for building up ``for`` loops onto circuits in a natural order, without
    having to construct the loop body first.

    Within the block, a lot of the bookkeeping is done for you; you do not need to keep track of
    which qubits and clbits you are using, for example, and a loop variable will be allocated for
    you, if you do not supply one yourself.  The type of the auto-generated loop variable follows
    the type of the ``indexset``:

    * Python :class:`range` / integer list → compile-time :class:`~.Parameter`.
    * Classical :class:`~.expr.Range` → real-time :class:`~.expr.Var` of the Range's type.

    In both cases, an auto-generated loop variable that the body never references is dropped
    (the resulting :class:`ForLoopOp` will have ``loop_parameter=None``).  To use a real-time
    :class:`~.expr.Var` counter over a Python :class:`range`, pass it explicitly as
    ``loop_parameter``.

    Mixing a :class:`~.Parameter` with an :class:`~.expr.Range` raises :class:`~.CircuitError`;
    see :class:`ForLoopOp`.

    You generally should never need to instantiate this object directly.  Instead, use
    :obj:`.QuantumCircuit.for_loop` in its context-manager form, i.e. by not supplying a ``body`` or
    sets of qubits and clbits.

    Example usage::

        import math
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2, 1)

        with qc.for_loop(range(5)) as i:
            qc.rx(i * math.pi/4, 0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()

    This context should almost invariably be created by a :meth:`.QuantumCircuit.for_loop` call, and
    the resulting instance is a "friend" of the calling circuit.  The context will manipulate the
    circuit's defined scopes when it is entered (by pushing a new scope onto the stack) and exited
    (by popping its scope, building it, and appending the resulting :obj:`.ForLoopOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    # Class-level variable keep track of the number of auto-generated loop variables, so we don't
    # get naming clashes.
    _generated_loop_parameters = 0

    __slots__ = (
        "_circuit",
        "_generate_loop_parameter",
        "_indexset",
        "_label",
        "_loop_parameter",
        "_used",
    )

    def __init__(
        self,
        circuit: QuantumCircuit,
        indexset: Iterable[int] | Range,
        loop_parameter: Parameter | Var | None = None,
        *,
        label: str | None = None,
    ):
        self._circuit = circuit
        self._generate_loop_parameter = loop_parameter is None
        self._loop_parameter = loop_parameter
        # We can pass through `range` instances because OpenQASM 3 has native support for this type
        # of iterator set.
        self._indexset = indexset if isinstance(indexset, (range, Range)) else tuple(indexset)
        self._label = label
        self._used = False

    def __enter__(self):
        if self._used:
            raise CircuitError("A for-loop context manager cannot be re-entered.")
        self._used = True
        if self._generate_loop_parameter:
            # Auto-generate a loop variable whose type follows the indexset; an
            # auto-generated variable that the body never references is dropped in __exit__.
            self._loop_parameter = self._make_loop_variable()
            type(self)._generated_loop_parameters += 1
        # An `expr.Var` loop parameter is owned by the loop header and input into the body, so it
        # is plumbed through as the scope's `loop_var`.  A `Parameter` (or `None`) needs no scope
        # registration.  `loop_var_explicit` keeps a user-supplied Var even if it is never used,
        # mirroring how an explicit `Parameter` loop variable is kept.
        self._circuit._push_scope(
            loop_var=self._loop_parameter if isinstance(self._loop_parameter, Var) else None,
            loop_var_explicit=not self._generate_loop_parameter,
        )
        return self._loop_parameter

    def _make_loop_variable(self):
        """Construct an auto-generated loop variable matched to the indexset type."""
        name = f"_loop_i_{self._generated_loop_parameters}"
        if isinstance(self._indexset, Range):
            return Var.new(name, self._indexset.type)
        return Parameter(name)

    def _is_loop_variable_referenced(self, body) -> bool:
        """Return whether ``self._loop_parameter`` is referenced inside ``body``.

        A single boundary helper between the two kinds of loop variable Qiskit supports:
        a compile-time :class:`~.Parameter` appears in :attr:`.QuantumCircuit.parameters`,
        a real-time :class:`~.expr.Var` appears as the body's ``input`` variable (the scope only
        emits it when it was referenced).  The rest of the for-loop machinery treats
        ``loop_parameter`` uniformly.
        """
        if self._loop_parameter is None:
            return False
        if isinstance(self._loop_parameter, Var):
            return body.has_var(self._loop_parameter)
        return self._loop_parameter in body.parameters

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # If we're leaving the context manager because an exception was raised, there's nothing
            # to do except restore the circuit state.
            self._circuit._pop_scope()
            return False
        scope = self._circuit._pop_scope()
        # Loops do not need to pass any further resources in, because this scope itself defines the
        # extent of ``break`` and ``continue`` statements.
        body = scope.build(scope.qubits(), scope.clbits())
        # We always bind the loop variable if the user supplied it explicitly, even if unused.
        # For an auto-generated loop variable, we only keep it if the user actually referenced it
        # inside the body, to avoid leaving stray dangling resources.
        if self._generate_loop_parameter and not self._is_loop_variable_referenced(body):
            loop_parameter = None
        else:
            loop_parameter = self._loop_parameter

        self._circuit.append(
            ForLoopOp(self._indexset, loop_parameter, body, label=self._label),
            tuple(body.qubits),
            tuple(body.clbits),
        )
        return False
