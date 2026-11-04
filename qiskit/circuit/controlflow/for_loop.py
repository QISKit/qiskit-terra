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
from collections.abc import Iterable

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.classical import expr
from qiskit.circuit.classical.types import Uint
from qiskit.circuit.exceptions import CircuitError
from qiskit._accelerate.circuit import ControlFlowType
from .control_flow import ControlFlowOp

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


class ForLoopOp(ControlFlowOp):
    """A circuit operation which repeatedly executes a subcircuit
    (``body``) parameterized by a parameter ``loop_parameter`` through
    the set of integer values provided in ``indexset``.

    Data model
    ----------

    There is exactly one "block" in a for-loop op, which is the body of the loop.  The circuit block
    may take exactly zero or one ``input`` variable (see :meth:`.QuantumCircuit.add_input_var`).  If
    the body takes one input variable, then ``loop_parameter`` must be equal to that variable, and
    it represents the loop variable.  If the body takes zero input variables, ``loop_parameter`` may
    be either ``None`` (to indicate no binding) or a :class:`.Parameter`.  The :class:`.Parameter`
    form is a legacy form that should be avoided.
    """

    _control_flow_type = ControlFlowType.ForLoop

    def __init__(
        self,
        indexset: Iterable[int],
        loop_parameter: Parameter | expr.Var | None,
        body: QuantumCircuit,
        label: str | None = None,
    ):
        """
        Args:
            indexset: A collection of integers to loop over.
            loop_parameter: The placeholder parameter to which
                the values from ``indexset`` will be assigned. Can be a
                ``Parameter``, ``expr.Var`` of type ``Uint``, or ``None``.
            body: The loop body to be repeatedly executed.
            label: An optional label for identifying the instruction.
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

        if not isinstance(loop_parameter, (Parameter, expr.Var, type(None))):
            raise CircuitError(
                "ForLoopOp expects a loop_parameter parameter, to "
                "be either of type Parameter, Var or None, but received "
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
            loop_parameter is not None
            and isinstance(loop_parameter, Parameter)
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

        # Consume indexset into a tuple unless it was provided as a range.
        # Preserve ranges so that they can be exported as OpenQASM 3 ranges.
        indexset = indexset if isinstance(indexset, range) else tuple(indexset)

        self._params = [indexset, loop_parameter, body]

    @property
    def blocks(self):
        return (self._params[2],)

    def replace_blocks(self, blocks):
        (body,) = blocks
        return ForLoopOp(self.params[0], self.params[1], body, label=self.label)


class ForLoopContext:
    """A context manager for building up ``for`` loops onto circuits in a natural order, without
    having to construct the loop body first.

    Within the block, a lot of the bookkeeping is done for you; you do not need to keep track of
    which qubits and clbits you are using, for example, and a loop parameter will be allocated for
    you, if you do not supply one yourself.  All normal methods of accessing the qubits on the
    underlying :obj:`~QuantumCircuit` will work correctly, and resolve into correct accesses within
    the interior block.

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
    _generated_loop_vars = 0

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
        indexset: Iterable[int],
        loop_parameter: Parameter | expr.Var | None = None,
        *,
        label: str | None = None,
    ):
        self._circuit = circuit
        self._generate_loop_parameter = loop_parameter is None
        self._loop_parameter = loop_parameter
        # We can pass through `range` instances because OpenQASM 3 has native support for this type
        # of iterator set.
        self._indexset = indexset if isinstance(indexset, range) else tuple(indexset)
        self._label = label
        self._used = False

    def __enter__(self):
        if self._used:
            raise CircuitError("A for-loop context manager cannot be re-entered.")
        self._used = True
        if self._generate_loop_parameter:
            self._loop_parameter = Parameter(f"_loop_i_{self._generated_loop_parameters}")
            type(self)._generated_loop_parameters += 1
        self._circuit._push_scope(
            loop_var=None if isinstance(self._loop_parameter, Parameter) else self._loop_parameter
        )
        return self._loop_parameter

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
        # We always bind the loop parameter if the user gave it to us, even if it isn't actually
        # used, because they requested we do that by giving us a parameter.  However, if they asked
        # us to auto-generate a parameter, then we only add it if they actually used it, to avoid
        # using unnecessary resources.
        if self._generate_loop_parameter and self._loop_parameter not in body.parameters:
            loop_parameter = None
        else:
            loop_parameter = self._loop_parameter

        self._circuit.append(
            ForLoopOp(self._indexset, loop_parameter, body, label=self._label),
            tuple(body.qubits),
            tuple(body.clbits),
        )
        return False
